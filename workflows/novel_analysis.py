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

**为什么 interleaved**(character + beat 阶段):character / beat 过去是串行
跑完整本书,beat 看到的 character 是「读完整本书的剧透态」,弧光 X→Y 被提前
透露。改成逐 batch 内交错跑后,beat 看到的 character 是「读到当前 batch 末尾
的最新态」,跟人类阅读的体感一致。

**为什么 plan-then-storyboard**:1 段 beat 通常只够 60-120 秒视频体量,而短剧
目标每集 ~300 秒;直接 1 beat = 1 集会导致镜数严重不达标。引入
``episode_planner`` 把 N 段 beat 聚合成 M 集,然后逐集分镜。代价是 character
在分镜阶段看到的是「终态」(不再渐进),换取分集粒度合理 + 集层叙事连贯。

**为什么拆 storyboard 为两个 agent**:一次 LLM 调用既出叙事又出视觉指导职责
太重导致两边都不深入。现在拆成「叙事分镜师」(``extract_storyboard.narrate_episode``)
+「镜头导演」(``shot_director.direct_episode``),前者只关心讲什么,后者拿到
锁定的叙事后专心做视觉决策。代价是每集 2 次 LLM 调用,换来两个 prompt 都更专注,
产出质量更可控。

**节奏**(见 ``_node_interleaved_analysis``)::

    # 阶段 1:逐 batch 跑 character + beat(渐进态)
    for batch in batches:
        extract_characters.extract_for_batch  → 更新 known_chars
        extract_beats.extract_for_batch       → 更新 beats_so_far
                                                (本批可重写末尾 K 段)
    # 阶段 2:全书 beat 抽完后一次性 plan(终态)
    episode_planner.plan_episodes(beats_so_far, ...) → plan_list
    # 阶段 3:按 plan 顺序串行分镜
    for plan in plan_list.plans:
        build_episode_from_plan(plan, ...)    → 叙事 + 视觉 + 合并 → +1 集

**一集 = 两个 agent**:``extract_storyboard.narrate_episode`` 出叙事分镜
(谁 / 在哪 / 说什么 / 想什么 + 集层 director_intent),``shot_director.direct_episode``
出视觉指导(景别 / 运镜 / 起始画面 / 时长 + 集层 visual_style),
``extract_storyboard.merge_episode`` 按 index 合并成 ``Episode``。
封装在 ``workflows.episode_pipeline.build_episode_from_plan``。

**跨批续写**(beat 阶段):beat agent 每批必须复述/修订「末尾 K 段」
(``rewrite_window``,默认 1),LLM 想接着写就重写 summary,想啥都不改就原样
照抄。merge 端用前 K 项 in-place 改写 ``beats_so_far`` 末尾 K 段(index 沿用,
related_batches 追加本批),其后追加新段。**不再有 storyboard 的冷却期**
(plan 是全书一次性,自然不存在冷却问题)。

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
from typing import Dict, List, TypedDict

from configs import RunConfig
from skills.batch_chapters import Batch
from skills.book_ingest import IngestResult, load_and_batch_txt
from skills.epub_to_txt import epub_to_txt
from agents import (
    episode_planner,
    extract_beats,
    extract_characters,
    extract_storyboard,
    shot_director,
)
from schemas.beat import Beat, BeatList
from schemas.character import Character, CharacterList
from schemas.report import AgentLLMInfo, FinalReport, ReportMeta
from schemas.storyboard import Episode, ScreenplayAnalysis, Storyboard
from skills.file_io import write_final_report
from workflows.episode_pipeline import build_episode_from_plan

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent LLM 编排:trace 注入 + 配置摘要采集
# ---------------------------------------------------------------------------


AGENT_LLM_MODULES = {
    "extract_characters": extract_characters,
    "extract_beats":      extract_beats,
    "episode_planner":    episode_planner,
    "extract_storyboard": extract_storyboard,
    "shot_director":      shot_director,
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
    characters: CharacterList
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
    """两阶段:逐 batch 跑 character + beat(渐进态)→ 全书 plan + 串行分镜(终态)。

    **阶段 1**(每 batch):
        extract_characters.extract_for_batch  → 更新 ``known_chars``(渐进)
        extract_beats.extract_for_batch       → 更新 ``beats_so_far``(可重写末尾 K 段)

    **阶段 2**(全书 beat 抽完后,一次性):
        episode_planner.plan_episodes(beats_so_far, known_chars, ...) → plan_list
        for ep_index, plan in enumerate(plan_list.plans, start=1):
            build_episode_from_plan(plan, member_beats, ...) → +1 集

    阶段 2 的 ``known_chars`` 是终态(全书读完);这是 episode_planner 设计的天然
    要求(全局视角才能正确分集)。trade-off:character 弧光在分镜阶段是"剧透态",
    不再是渐进态。
    """
    ing = state["ingest_result"]
    target = state["target_episode_duration_sec"]
    window = state["recent_beats_window"]
    rewrite_window = state["rewrite_window"]
    prev_tail_window = state["storyboard_prev_tail_window"]

    known_chars: Dict[str, Character] = {}
    beats_so_far: List[Beat] = []

    batches = list(ing.batches)
    batch_lookup: Dict[int, Batch] = {b.index: b for b in batches}

    logger.info("=" * 60)
    logger.info(
        "[interleaved] 阶段 1(逐 batch 抽 character + beat):共 %d 批,"
        "近段窗口 %d,rewrite K=%d",
        len(batches), window, rewrite_window,
    )
    logger.info("=" * 60)

    t_total = time.perf_counter()
    for i, batch in enumerate(batches, start=1):
        t_batch = time.perf_counter()

        ch_result = extract_characters.extract_for_batch(
            batch, known_chars, title=ing.title,
        )
        if ch_result.renames:
            extract_beats.apply_character_renames(beats_so_far, ch_result.renames)
        extract_beats.extract_for_batch(
            batch, beats_so_far, known_chars,
            title=ing.title,
            context_window=window,
            rewrite_window=rewrite_window,
        )

        elapsed = time.perf_counter() - t_batch
        logger.info(
            "[interleaved] %d/%d (batch=%d) 完成,%.1f 秒。"
            "累计 %d 人 / %d 段",
            i, len(batches), batch.index, elapsed,
            len(known_chars), len(beats_so_far),
        )

    # 阶段 2:全书 beat 抽完后,plan + 串行分镜
    logger.info("=" * 60)
    logger.info(
        "[interleaved] 阶段 2(全书 plan + 串行分镜):%d 段 beat → 待规划",
        len(beats_so_far),
    )
    logger.info("=" * 60)

    plan_list = episode_planner.plan_episodes(
        beats_so_far, known_chars,
        target_duration_sec=target,
        title=ing.title,
    )

    beat_by_idx: Dict[int, Beat] = {b.index: b for b in beats_so_far}
    episodes: List[Episode] = []
    prev_tail: List[Storyboard] = []  # 上集末尾 K 镜,给下集做画面承接

    for ep_index, plan in enumerate(plan_list.plans, start=1):
        t_ep = time.perf_counter()
        member_beats = [beat_by_idx[i] for i in plan.beat_indices if i in beat_by_idx]
        missing = [i for i in plan.beat_indices if i not in beat_by_idx]
        if missing:
            logger.warning(
                "[interleaved] 集 %d (%s) 的 beat_indices 含未知 index=%s,已跳过",
                ep_index, plan.title, missing,
            )
        if not member_beats:
            logger.warning(
                "[interleaved] 集 %d (%s) 无有效 member_beats,跳过",
                ep_index, plan.title,
            )
            continue

        prev_plan = plan_list.plans[ep_index - 2] if ep_index > 1 else None
        next_plan = plan_list.plans[ep_index] if ep_index < len(plan_list.plans) else None

        ep = build_episode_from_plan(
            ep_index=ep_index,
            plan=plan,
            member_beats=member_beats,
            known_chars=known_chars,
            batch_lookup=batch_lookup,
            prev_plan=prev_plan,
            next_plan=next_plan,
            prev_tail=prev_tail,
            target_duration_sec=target,
        )
        episodes.append(ep)
        prev_tail = (
            ep.storyboards[-prev_tail_window:]
            if (ep.storyboards and prev_tail_window > 0)
            else []
        )

        elapsed = time.perf_counter() - t_ep
        logger.info(
            "[interleaved] 集 %d/%d (%s) 完成,%.1f 秒,%d 镜",
            ep_index, len(plan_list.plans), plan.title, elapsed, len(ep.storyboards),
        )

    total_elapsed = time.perf_counter() - t_total
    logger.info(
        "[interleaved] 全部完成:%d 人 / %d 段 / %d 集,合计 %.1f 秒(%.1f 分钟)",
        len(known_chars), len(beats_so_far), len(episodes),
        total_elapsed, total_elapsed / 60,
    )

    return {
        "characters": CharacterList(characters=list(known_chars.values())),
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
        "max_total_chars=%d episode=%ds window=%d rewrite_K=%d prev_tail_K=%d",
        config.input, out_root, config.max_batch_chars, config.max_total_chars,
        config.target_episode_duration_sec, config.recent_beats_window,
        config.rewrite_window, config.storyboard_prev_tail_window,
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
    final: WorkflowState = graph.invoke(initial)
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
