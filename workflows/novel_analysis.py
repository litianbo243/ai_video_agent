"""novel_analysis 父-workflow:**逐 batch 交错**跑三阶段,产出最终剧本结构。

公开 API:

* ``run(config) -> RunResult``
    顶层,runner 用。派生带时间戳的输出目录 + 注入 agent trace 路径 +
    编译并执行父-graph,把最终 ``FinalReport`` / 输出路径打包返回。
* ``build_graph()``
    编译父-workflow。
* ``WorkflowState``
    节点之间流通的状态契约(``TypedDict``)。
* ``RunResult``
    runner 面向的扁平结果(report + output_dir + output_paths)。


DAG::

    START
      ▼
    detect_input ──[epub]──▶ convert_epub ──┐
        │                                    │
        └──[txt]─────────────────────────────┴──▶ ingest_and_batch
                                                       ▼
                                              interleaved_analysis
                                                       ▼
                                                     write
                                                       ▼
                                                      END

**为什么 interleaved**:character / beat / storyboard 三阶段过去是串行跑完整本书
(先抽完所有人物 → 再切完所有 beats → 再逐 beat 出分镜)。这种做法有副作用:
beat / storyboard 看到的 character 是「读完整本书的剧透态」,弧光 X→Y 被提前
透露,LLM 反而难写出渐进的人物动机。改成逐 batch 内交错跑后,beat / storyboard
看到的 character 是「读到当前 batch 末尾的最新态」,跟人类阅读的体感一致。

**节奏**(每个 batch 内,见 ``_node_interleaved_analysis``)::

    extract_characters.extract_for_batch     → 更新 known_chars
    extract_beats.extract_for_batch          → 更新 beats_so_far
                                              (本批可重写末尾 K 段 = rewrite_window)
    for b in beats_so_far[:-K] if not storyboarded:
        storyboard_beat(b, known_chars, ...) → +1 集

**跨批续写**:beat agent 每批必须复述/修订「末尾 K 段」(``rewrite_window``,默认 1),
LLM 想接着写就重写 summary,想啥都不改就原样照抄。merge 端用前 K 项 in-place 改写
``beats_so_far`` 末尾 K 段(index 沿用,related_batches 追加本批),其后追加新段。
末尾 K 段属于 LLM 可改窗口,storyboard 暂不触发(冷却期);全书结束后一次性 flush。

**子 workflow 的角色变化**:``character_analysis`` / ``beat_analysis`` /
``storyboard_analysis`` 三个子-workflow **仍保留**,但 novel_analysis 不再
``invoke`` 它们的 compiled graph。子-workflow 现在的定位是「dev / debug 时
单独跑某一阶段的全书循环」(比如手工 review 整本书的 character),不再是
novel_analysis 的子节点。

**场景的处理**:本工程不维护独立的场景视觉档案。Beat agent 自己产出
``setting_refs`` 作为字符串 label,storyboard agent 在写每镜 ``description`` 时
用原文 + LLM 常识把视觉环境写到画面里。所以这里没有 setting_analysis 子-workflow。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, TypedDict

from configs import RunConfig
from skills.batch_chapters import Batch
from skills.book_ingest import IngestResult, load_and_batch_txt
from skills.epub_to_txt import epub_to_txt
from agents import extract_beats, extract_characters, extract_storyboard
from agents.extract_beats import Beat, BeatList
from agents.extract_characters import Character, CharacterRoster
from agents.extract_storyboard import (
    Episode,
    ScreenplayAnalysis,
    Storyboard,
    storyboard_beat,
)
from skills.file_io import AgentLLMInfo, FinalReport, ReportMeta, write_final_report

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent LLM 编排:trace 注入 + 配置摘要采集
# ---------------------------------------------------------------------------


AGENT_LLM_MODULES = {
    "extract_characters": extract_characters,
    "extract_beats":      extract_beats,
    "extract_storyboard": extract_storyboard,
}


def setup_agent_traces(out_dir: Path) -> None:
    """为所有 agent 注入运行时 trace 目录(``<out_dir>/llm_trace/<agent>.jsonl``)。

    runner 顶层在派生 out_dir 之后调一次即可。各 agent 内部首次 build LLM
    时会自动 pick up 这个 trace 路径。
    """
    for agent in AGENT_LLM_MODULES.values():
        agent.set_trace_dir(out_dir)


def collect_agent_llm_info() -> Dict[str, AgentLLMInfo]:
    """收集每个 agent 实际用到的 LLM 摘要(base_url + model),供 ReportMeta 落盘。

    会触发 lazy build(若尚未 build);流水线跑到 write 阶段时 3 个 agent 都
    已用过 LLM,这步只是查询,无副作用。
    """
    out: Dict[str, AgentLLMInfo] = {}
    for name, agent in AGENT_LLM_MODULES.items():
        client = agent.get_llm()
        out[name] = AgentLLMInfo(base_url=client.base_url, model=client.model)
    return out


# ---------------------------------------------------------------------------
# State:节点之间流通的薄数据;只放需要跨节点传的内容
# ---------------------------------------------------------------------------


class WorkflowState(TypedDict, total=False):
    """LangGraph 在节点之间传递的状态对象(``TypedDict``)。

    保持薄:子-workflow 内部的迭代/累加状态不放这里,只放跨节点产物。
    LLM 由各 agent 自治(每个 agent 有自己的 ``llm.json``),所以 state 里
    不再有 ``llm_base_url`` / ``llm_model``。
    """

    # --- 启动时填入 ---
    input_path: str
    output_dir: str
    max_batch_chars: int
    max_total_chars: int
    target_episode_duration_sec: int
    recent_beats_window: int
    rewrite_window: int
    storyboard_prev_tail_window: int

    # --- detect_input / convert_epub 之后(ingest 节点会消费) ---
    txt_path: str

    # --- ingest_and_batch 之后(替代旧的 title / raw_chars / total_chars / batches 四件套) ---
    ingest_result: IngestResult

    # --- 各子-workflow 输出 ---
    characters: CharacterRoster
    beats: BeatList
    screenplay: ScreenplayAnalysis

    # --- write 之后 ---
    final_report: FinalReport
    output_paths: Dict[str, str]


# ---------------------------------------------------------------------------
# 节点
# ---------------------------------------------------------------------------


def _node_detect_input(state: WorkflowState) -> WorkflowState:
    src = Path(state["input_path"])
    if not src.is_file():
        raise FileNotFoundError(f"输入文件不存在:{src}")
    suffix = src.suffix.lower()
    if suffix not in {".txt", ".epub"}:
        raise ValueError(f"不支持的文件类型 {suffix!r};只接受 .txt 与 .epub。")
    size_mb = src.stat().st_size / (1024 * 1024)
    logger.info("[detect_input] %s(%s,%.2f MB)", src, suffix, size_mb)
    return {}


def _route_after_detect(state: WorkflowState) -> str:
    suffix = Path(state["input_path"]).suffix.lower()
    route = "epub" if suffix == ".epub" else "txt"
    logger.info("[detect_input] 路由:.%s → %s 分支", suffix.lstrip("."), route)
    return route


def _node_convert_epub(state: WorkflowState) -> WorkflowState:
    logger.info("[convert_epub] 开始 EPUB → TXT:%s", state["input_path"])
    t0 = time.perf_counter()
    txt_path = epub_to_txt(state["input_path"])
    elapsed = time.perf_counter() - t0
    logger.info("[convert_epub] 用时 %.1f 秒,产物 %s", elapsed, txt_path)
    return {"txt_path": str(txt_path)}


def _node_ingest_and_batch(state: WorkflowState) -> WorkflowState:
    src = Path(state.get("txt_path") or state["input_path"])
    title, raw_chars, total_chars, batches = load_and_batch_txt(
        src,
        max_batch_chars=state["max_batch_chars"],
        max_total_chars=state["max_total_chars"],
    )
    ing = IngestResult(
        title=title,
        txt_path=src,
        raw_chars=raw_chars,
        total_chars=total_chars,
        batches=batches,
    )
    return {"txt_path": str(src), "ingest_result": ing}


def _node_interleaved_analysis(state: WorkflowState) -> WorkflowState:
    """逐 batch 交错跑 character → beat → 离开"复述区"的 beats 立即 storyboard。

    模拟「读书」过程:character / beat 是渐进态,storyboard 看到的 character
    是该 beat 离开 LLM 修订窗口那一刻的最新状态(而不是读完整本书后的剧透态)。

    **节拍**(每 batch):

    1) ``extract_characters.extract_for_batch``  → 更新 ``known_chars``
    2) ``extract_beats.extract_for_batch``       → 更新 ``beats_so_far``;
       本批 LLM 可以重写 ``beats_so_far`` 末尾 ``rewrite_window`` 段
    3) 扫 ``beats_so_far[:-rewrite_window]``,尚未 storyboard 的 → 立刻 ``storyboard_beat``

    **storyboard 冷却期**:末尾 ``rewrite_window`` 段还在 LLM 可改窗口内,不立即
    送 storyboard;等下一批不再回看时(或全书结束)再送。

    **尾部 flush**:全部 batch 跑完后,把仍在冷却期里的末尾 ``rewrite_window``
    段一次性补 storyboard(那时再无下批,不会被改了)。
    """
    ing = state["ingest_result"]
    target = state["target_episode_duration_sec"]
    window = state["recent_beats_window"]
    rewrite_window = state["rewrite_window"]
    prev_tail_window = state["storyboard_prev_tail_window"]

    known_chars: Dict[str, Character] = {}
    beats_so_far: List[Beat] = []
    storyboarded_idx: Set[int] = set()
    episodes: List[Episode] = []
    prev_tail: List[Storyboard] = []  # 上集末尾 K 镜,给下集做画面承接

    batches = list(ing.batches)
    batch_lookup: Dict[int, Batch] = {b.index: b for b in batches}

    logger.info("=" * 60)
    logger.info(
        "[interleaved] 启动:共 %d 批,目标每集 %d 秒,近段窗口 %d,rewrite K=%d",
        len(batches), target, window, rewrite_window,
    )
    logger.info("=" * 60)

    t_total = time.perf_counter()
    for i, batch in enumerate(batches, start=1):
        t_batch = time.perf_counter()

        extract_characters.extract_for_batch(
            batch, known_chars, title=ing.title,
        )
        extract_beats.extract_for_batch(
            batch, beats_so_far, known_chars,
            title=ing.title,
            target_duration_sec=target,
            context_window=window,
            rewrite_window=rewrite_window,
        )

        # 末尾 K 段还在 LLM 可改窗口里,先不送 storyboard
        safe_until = len(beats_so_far) - rewrite_window
        new_eps = 0
        for bi, b in enumerate(beats_so_far[:safe_until]):
            if b.index not in storyboarded_idx:
                prev_b = beats_so_far[bi - 1] if bi > 0 else None
                next_b = beats_so_far[bi + 1] if bi + 1 < len(beats_so_far) else None
                ep = storyboard_beat(
                    b, known_chars, batch_lookup,
                    prev_beat=prev_b, next_beat=next_b,
                    prev_tail_storyboards=prev_tail,
                    target_duration_sec=target,
                )
                episodes.append(ep)
                storyboarded_idx.add(b.index)
                new_eps += 1
                prev_tail = ep.storyboards[-prev_tail_window:] if (ep.storyboards and prev_tail_window > 0) else []

        elapsed = time.perf_counter() - t_batch
        logger.info(
            "[interleaved] %d/%d (batch=%d) 完成,%.1f 秒。"
            "累计 %d 人 / %d 段(冷却中 %d 段 → +%d 集,共 %d 集)",
            i, len(batches), batch.index, elapsed,
            len(known_chars), len(beats_so_far),
            len(beats_so_far) - safe_until, new_eps, len(episodes),
        )

    # 全书结束:冷却期里剩下的 K 段一次 flush
    for bi, b in enumerate(beats_so_far):
        if b.index not in storyboarded_idx:
            prev_b = beats_so_far[bi - 1] if bi > 0 else None
            next_b = beats_so_far[bi + 1] if bi + 1 < len(beats_so_far) else None
            ep = storyboard_beat(
                b, known_chars, batch_lookup,
                prev_beat=prev_b, next_beat=next_b,
                prev_tail_storyboards=prev_tail,
                target_duration_sec=target,
            )
            episodes.append(ep)
            storyboarded_idx.add(b.index)
            prev_tail = ep.storyboards[-prev_tail_window:] if (ep.storyboards and prev_tail_window > 0) else []

    total_elapsed = time.perf_counter() - t_total
    logger.info(
        "[interleaved] 全部完成:%d 人 / %d 段 / %d 集,合计 %.1f 秒(%.1f 分钟)",
        len(known_chars), len(beats_so_far), len(episodes),
        total_elapsed, total_elapsed / 60,
    )

    return {
        "characters": CharacterRoster(characters=list(known_chars.values())),
        "beats":      BeatList(beats=beats_so_far),
        "screenplay": ScreenplayAnalysis(episodes=episodes),
    }


def _node_write(state: WorkflowState) -> WorkflowState:
    ing = state["ingest_result"]
    meta = ReportMeta(
        source_path=str(Path(state["input_path"]).resolve()),
        txt_path=str(ing.txt_path),
        title=ing.title,
        total_chars=ing.total_chars,
        batch_count=len(ing.batches),
        max_batch_chars=state["max_batch_chars"],
        max_total_chars=state["max_total_chars"],
        llm_per_agent=collect_agent_llm_info(),
    )
    report = FinalReport(
        screenplay=state["screenplay"],
        characters=state["characters"],
        beats=state["beats"],
        meta=meta,
    )
    out_dir = Path(state["output_dir"])
    logger.info("[write] 开始落盘 → %s", out_dir)
    t0 = time.perf_counter()
    paths = write_final_report(report, out_dir)
    paths["txt_path"] = str(ing.txt_path)
    elapsed = time.perf_counter() - t0
    logger.info(
        "[write] 落盘完成,用时 %.1f 秒,共 %d 个文件",
        elapsed, len(paths),
    )
    return {"final_report": report, "output_paths": paths}


# ---------------------------------------------------------------------------
# 构图
# ---------------------------------------------------------------------------


def build_graph():
    """编译父 workflow。

    `interleaved_analysis` 在单节点内做完三阶段的 batch 循环交错,所以这里不再
    内嵌 character / beat / storyboard 三个子-graph(它们仍作为独立 workflow
    保留,供 dev 单独跑某一层时用)。

    LLM 由各 agent 自治(``agents/extract_*/llm.json``),workflow 不再 build /
    传递 LLM。
    """
    from langgraph.graph import StateGraph, END

    graph = StateGraph(WorkflowState)

    graph.add_node("detect_input", _node_detect_input)
    graph.add_node("convert_epub", _node_convert_epub)
    graph.add_node("ingest_and_batch", _node_ingest_and_batch)
    graph.add_node("interleaved_analysis", _node_interleaved_analysis)
    graph.add_node("write", _node_write)

    graph.set_entry_point("detect_input")

    graph.add_conditional_edges(
        "detect_input",
        _route_after_detect,
        {"epub": "convert_epub", "txt": "ingest_and_batch"},
    )
    graph.add_edge("convert_epub", "ingest_and_batch")

    graph.add_edge("ingest_and_batch", "interleaved_analysis")
    graph.add_edge("interleaved_analysis", "write")
    graph.add_edge("write", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# 顶层入口
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    report: FinalReport
    output_dir: Path
    output_paths: dict


def run(config: RunConfig) -> RunResult:
    """按 ``RunConfig`` 跑完整条 novel_analysis 流水线。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = (Path(config.output_dir) / timestamp).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    logger.info("本次运行输出目录:%s", out_root)

    setup_agent_traces(out_root)
    logger.info(
        "novel_analysis workflow 启动:input=%s output=%s max_batch_chars=%d "
        "max_total_chars=%d episode=%ds window=%d rewrite_K=%d prev_tail_K=%d recursion=%d",
        config.input, out_root, config.max_batch_chars, config.max_total_chars,
        config.target_episode_duration_sec, config.recent_beats_window,
        config.rewrite_window, config.storyboard_prev_tail_window,
        config.langgraph_recursion_limit,
    )

    graph = build_graph()
    initial: WorkflowState = {
        "input_path": str(config.input),
        "output_dir": str(out_root),
        "max_batch_chars": config.max_batch_chars,
        "max_total_chars": config.max_total_chars,
        "target_episode_duration_sec": config.target_episode_duration_sec,
        "recent_beats_window": config.recent_beats_window,
        "rewrite_window": config.rewrite_window,
        "storyboard_prev_tail_window": config.storyboard_prev_tail_window,
    }

    t_start = time.perf_counter()
    final: WorkflowState = graph.invoke(
        initial,
        config={"recursion_limit": config.langgraph_recursion_limit},
    )
    elapsed = time.perf_counter() - t_start
    logger.info(
        "novel_analysis workflow 完成,总用时 %.1f 秒(%.1f 分钟)",
        elapsed, elapsed / 60,
    )

    if "final_report" not in final:
        raise RuntimeError("workflow 结束但未产生 final_report。")

    return RunResult(
        report=final["final_report"],
        output_dir=out_root,
        output_paths=dict(final.get("output_paths", {})),
    )


__all__ = [
    "WorkflowState",
    "RunResult",
    "build_graph",
    "run",
    "setup_agent_traces",
    "collect_agent_llm_info",
]
