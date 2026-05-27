"""novel_analysis 主流水线:**逐 batch 交错**跑三阶段,产出最终剧本结构。

按 ``RunConfig.mode`` 早停,由浅到深严格超集:

* ``RunMode.CHARACTER``  —— 只跑人物档案
* ``RunMode.BEAT``       —— + 剧情段切分(interleaved 与 character 同批)
* ``RunMode.EPISODE``    —— + 分集规划
* ``RunMode.SCREENPLAY`` —— + 逐集分镜(完整流水线)

深 mode 跑出的中间产物跟单独跑浅 mode 是同一份,所以"先用浅 mode 调
prompt / 切分,再切到深 mode 跑完整流水线"是无成本的。

DAG::

    START
      ▼
    detect_input ──[epub]──▶ convert_epub ──┐
        │                                    │
        └──[txt]─────────────────────────────┴──▶ ingest_and_batch
                                                       ▼
                                                    analysis   ← 内部按 mode 早停
                                                       ▼
                                                     write     ← 按 mode 按需落盘
                                                       ▼
                                                      END


**为什么 interleaved**(character + beat 阶段):character / beat 过去是串行
跑完整本书,beat 看到的 character 是「读完整本书的剧透态」,弧光 X→Y 被提前
透露。改成逐 batch 内交错跑后,beat 看到的 character 是「读到当前 batch 末尾
的最新态」,跟人类阅读的体感一致。

**为什么 plan-then-shot**:1 段 beat 通常只够 60-120 秒视频体量,而短剧
目标每集 ~300 秒;直接 1 beat = 1 集会导致镜数严重不达标。引入
``episode_planner`` 把 N 段 beat 聚合成 M 集,然后逐集分镜。代价是 character
在分镜阶段看到的是「终态」(不再渐进),换取分集粒度合理 + 集层叙事连贯。

**为什么把分镜拆成两个 agent**:一次 LLM 调用既出叙事又出视觉指导职责
太重导致两边都不深入。现在拆成「叙事分镜师」(``narrative_director.narrate_episode``)
+「镜头导演」(``shot_director.direct_episode``),前者只关心讲什么,后者拿到
锁定的叙事后专心做视觉决策。代价是每集 2 次 LLM 调用,换来两个 prompt 都更专注,
产出质量更可控。

**场景的处理**:本工程不维护独立的场景视觉档案。``beat_segmenter`` 自己产出
``setting_refs`` 作为字符串 label,``shot_director`` 在写每镜 ``description`` 时
用原文 + LLM 常识把视觉环境写到画面里。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, TypedDict

from configs import RunConfig, RunMode, mode_includes
from skills.batch_chapters import Batch
from skills.book_ingest import IngestResult, load_and_batch_txt
from skills.epub_to_txt import epub_to_txt
from skills.file_io import write_partial
from agents import (
    episode_planner,
    beat_segmenter,
    character_profiler,
    narrative_director,
    shot_director,
)
from schemas.beat import Beat, BeatList
from schemas.character import Character, CharacterList
from schemas.episode_plan import EpisodePlanList
from schemas.report import AgentLLMInfo, FinalReport, ReportMeta
from schemas.screenplay import Episode, Screenplay, Shot

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent LLM 编排:trace 注入 + 配置摘要采集
# ---------------------------------------------------------------------------


AGENT_LLM_MODULES = {
    "character_profiler": character_profiler,
    "beat_segmenter":     beat_segmenter,
    "episode_planner":    episode_planner,
    "narrative_director": narrative_director,
    "shot_director":      shot_director,
}


# 各 mode 实际会用到的 agent 子集(用于 collect_agent_llm_info_for_mode)
_AGENTS_BY_MODE: Dict[RunMode, List[str]] = {
    RunMode.CHARACTER:  ["character_profiler"],
    RunMode.BEAT:       ["character_profiler", "beat_segmenter"],
    RunMode.EPISODE:    ["character_profiler", "beat_segmenter", "episode_planner"],
    RunMode.SCREENPLAY: list(AGENT_LLM_MODULES.keys()),
}


def setup_agent_traces(out_dir: Path) -> None:
    """为所有 agent 注入运行时 trace 目录(``<out_dir>/llm_trace/<agent>.jsonl``)。

    runner 顶层在派生 out_dir 之后调一次即可。各 agent 内部首次 build LLM
    时会自动 pick up 这个 trace 路径。
    """
    for agent in AGENT_LLM_MODULES.values():
        agent.set_trace_dir(out_dir)


def collect_agent_llm_info_for_mode(mode: RunMode) -> Dict[str, AgentLLMInfo]:
    """收集当前 mode 实际用到的各 agent 的 LLM 摘要。

    会触发 lazy build(若尚未 build);流水线跑到 write 阶段时对应 agent 都
    已用过 LLM,这步只是查询,无副作用。
    """
    out: Dict[str, AgentLLMInfo] = {}
    for name in _AGENTS_BY_MODE[mode]:
        client = AGENT_LLM_MODULES[name].get_llm()
        out[name] = AgentLLMInfo(base_url=client.base_url, model=client.model)
    return out


# ---------------------------------------------------------------------------
# State:节点之间流通的薄数据;只放需要跨节点传的内容
# ---------------------------------------------------------------------------


class WorkflowState(TypedDict, total=False):
    """LangGraph 在节点之间传递的状态对象(``TypedDict``)。"""

    # --- 启动时填入 ---
    input_path: str
    output_dir: str
    mode: RunMode
    max_batch_chars: int
    max_total_chars: int
    target_episode_duration_sec: int
    recent_beats_window: int
    rewrite_window: int
    shot_prev_tail_window: int

    # --- detect_input / convert_epub 之后 ---
    txt_path: str

    # --- ingest_and_batch 之后 ---
    ingest_result: IngestResult

    # --- 各阶段输出(按 mode 选择性填充) ---
    characters: CharacterList
    beats: BeatList                  # mode >= BEAT
    episode_plans: EpisodePlanList   # mode >= EPISODE
    screenplay: Screenplay           # mode >= SCREENPLAY

    # --- write 之后 ---
    final_report: FinalReport        # 只在 SCREENPLAY mode 填充
    output_paths: Dict[str, str]


# ---------------------------------------------------------------------------
# 节点:输入处理
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


# ---------------------------------------------------------------------------
# 节点:核心分析(按 mode 早停)
# ---------------------------------------------------------------------------


def _stage_character_beat_loop(
    *,
    mode: RunMode,
    batches: List[Batch],
    title: str,
    recent_beats_window: int,
    rewrite_window: int,
) -> tuple[Dict[str, Character], List[Beat]]:
    """逐 batch 抽 character;若 mode >= BEAT,同批顺便跑 beat(interleaved)。

    返回 ``(known_chars, beats_so_far)``。CHARACTER mode 时 beats_so_far 为空列表。
    """
    known_chars: Dict[str, Character] = {}
    beats_so_far: List[Beat] = []
    do_beat = mode_includes(mode, RunMode.BEAT)

    label = "character + beat" if do_beat else "character only"
    logger.info("=" * 60)
    logger.info(
        "[interleaved] 阶段 1(逐 batch 抽 %s):共 %d 批,近段窗口 %d,rewrite K=%d",
        label, len(batches), recent_beats_window, rewrite_window,
    )
    logger.info("=" * 60)

    t_total = time.perf_counter()
    for i, batch in enumerate(batches, start=1):
        t_batch = time.perf_counter()

        ch_result = character_profiler.run_for_batch(
            batch, known_chars, title=title,
        )
        if do_beat:
            if ch_result.renames:
                beat_segmenter.apply_character_renames(beats_so_far, ch_result.renames)
            beat_segmenter.run_for_batch(
                batch, beats_so_far, known_chars,
                title=title,
                context_window=recent_beats_window,
                rewrite_window=rewrite_window,
            )

        elapsed = time.perf_counter() - t_batch
        logger.info(
            "[interleaved] %d/%d (batch=%d) 完成,%.1f 秒。累计 %d 人 / %d 段",
            i, len(batches), batch.index, elapsed,
            len(known_chars), len(beats_so_far),
        )

    total_elapsed = time.perf_counter() - t_total
    logger.info(
        "[interleaved] 阶段 1 完成:%d 人 / %d 段,合计 %.1f 秒",
        len(known_chars), len(beats_so_far), total_elapsed,
    )
    return known_chars, beats_so_far


def _stage_episode_planner(
    *,
    beats: List[Beat],
    known_chars: Dict[str, Character],
    target_duration_sec: int,
    title: str,
) -> EpisodePlanList:
    logger.info("=" * 60)
    logger.info(
        "[plan] 阶段 2(全书分集规划):%d 段 beat → 待规划",
        len(beats),
    )
    logger.info("=" * 60)
    t0 = time.perf_counter()
    plan_list = episode_planner.plan_episodes(
        beats, known_chars,
        target_duration_sec=target_duration_sec,
        title=title,
    )
    logger.info(
        "[plan] 完成:%d 段 beat → %d 集,用时 %.1f 秒",
        len(beats), len(plan_list.plans), time.perf_counter() - t0,
    )
    return plan_list


def _stage_shots(
    *,
    plan_list: EpisodePlanList,
    beats: List[Beat],
    batches: List[Batch],
    known_chars: Dict[str, Character],
    target_duration_sec: int,
    prev_tail_window: int,
) -> List[Episode]:
    logger.info("=" * 60)
    logger.info(
        "[shots] 阶段 3(逐集分镜):%d 集 → narrative + shot direction → Episode",
        len(plan_list.plans),
    )
    logger.info("=" * 60)

    beat_by_idx: Dict[int, Beat] = {b.index: b for b in beats}
    batch_lookup: Dict[int, Batch] = {b.index: b for b in batches}
    episodes: List[Episode] = []
    prev_tail: List[Shot] = []  # 上集末尾 K 镜,给下集做画面承接

    t_total = time.perf_counter()
    for ep_index, plan in enumerate(plan_list.plans, start=1):
        t_ep = time.perf_counter()
        member_beats = [beat_by_idx[i] for i in plan.beat_indices if i in beat_by_idx]
        missing = [i for i in plan.beat_indices if i not in beat_by_idx]
        if missing:
            logger.warning(
                "[shots] 集 %d (%s) 的 beat_indices 含未知 index=%s,已跳过",
                ep_index, plan.title, missing,
            )
        if not member_beats:
            logger.warning(
                "[shots] 集 %d (%s) 无有效 member_beats,跳过",
                ep_index, plan.title,
            )
            continue

        prev_plan = plan_list.plans[ep_index - 2] if ep_index > 1 else None
        next_plan = plan_list.plans[ep_index] if ep_index < len(plan_list.plans) else None

        # 一集的"分镜"= 2 次 LLM 调用 + 1 次合并:
        #   叙事分镜师(谁/在哪/说什么/想什么) → 镜头导演(景别/运镜/画面)
        #   → 按 index 配对合并成完整 Episode
        narrative = narrative_director.narrate_episode(
            ep_index=ep_index,
            plan=plan,
            member_beats=member_beats,
            characters=known_chars,
            batches=batch_lookup,
            prev_plan=prev_plan,
            next_plan=next_plan,
            prev_tail_shots=prev_tail,
            target_duration_sec=target_duration_sec,
        )
        direction = shot_director.direct_episode(
            episode_index=ep_index,
            director_intent=narrative.director_intent,
            narrative_shots=narrative.shots,
            characters=known_chars,
            prev_tail_shots=prev_tail,
            target_duration_sec=target_duration_sec,
        )
        ep = narrative_director.merge_episode(
            ep_index=ep_index,
            plan=plan,
            narrative=narrative,
            direction=direction,
        )
        episodes.append(ep)
        prev_tail = (
            ep.shots[-prev_tail_window:]
            if (ep.shots and prev_tail_window > 0)
            else []
        )
        elapsed = time.perf_counter() - t_ep
        logger.info(
            "[shots] 集 %d/%d (%s) 完成,%.1f 秒,%d 镜",
            ep_index, len(plan_list.plans), plan.title, elapsed, len(ep.shots),
        )

    logger.info(
        "[shots] 阶段 3 完成:%d 集,合计 %.1f 秒",
        len(episodes), time.perf_counter() - t_total,
    )
    return episodes


def _node_analysis(state: WorkflowState) -> WorkflowState:
    """按 mode 串行跑 1-3 个阶段:character[+beat] → [plan] → [shots]。"""
    mode = state["mode"]
    ing = state["ingest_result"]
    batches = list(ing.batches)

    # 阶段 1:character(+ beat,如果 mode >= BEAT)
    known_chars, beats_so_far = _stage_character_beat_loop(
        mode=mode,
        batches=batches,
        title=ing.title,
        recent_beats_window=state["recent_beats_window"],
        rewrite_window=state["rewrite_window"],
    )
    out: Dict = {
        "characters": CharacterList(characters=list(known_chars.values())),
    }
    if mode_includes(mode, RunMode.BEAT):
        out["beats"] = BeatList(beats=beats_so_far)

    if not mode_includes(mode, RunMode.EPISODE):
        return out

    # 阶段 2:分集规划
    plan_list = _stage_episode_planner(
        beats=beats_so_far,
        known_chars=known_chars,
        target_duration_sec=state["target_episode_duration_sec"],
        title=ing.title,
    )
    out["episode_plans"] = plan_list

    if not mode_includes(mode, RunMode.SCREENPLAY):
        return out

    # 阶段 3:逐集分镜
    episodes = _stage_shots(
        plan_list=plan_list,
        beats=beats_so_far,
        batches=batches,
        known_chars=known_chars,
        target_duration_sec=state["target_episode_duration_sec"],
        prev_tail_window=state["shot_prev_tail_window"],
    )
    out["screenplay"] = Screenplay(episodes=episodes)
    return out


# ---------------------------------------------------------------------------
# 节点:落盘(按 mode 落对应产物)
# ---------------------------------------------------------------------------


def _node_write(state: WorkflowState) -> WorkflowState:
    mode = state["mode"]
    ing = state["ingest_result"]
    out_dir = Path(state["output_dir"])

    # meta 在所有 mode 都生成(便于人类区分 run 时间 / 输入 / 用了哪些 LLM)
    meta = ReportMeta(
        source_path=str(Path(state["input_path"]).resolve()),
        txt_path=str(ing.txt_path),
        title=ing.title,
        total_chars=ing.total_chars,
        batch_count=len(ing.batches),
        max_batch_chars=state["max_batch_chars"],
        max_total_chars=state["max_total_chars"],
        llm_per_agent=collect_agent_llm_info_for_mode(mode),
    )

    logger.info("[write] 开始按 mode=%s 落盘 → %s", mode.value, out_dir)
    t0 = time.perf_counter()
    paths = write_partial(
        out_dir,
        characters=state.get("characters"),
        beats=state.get("beats"),
        episode_plans=state.get("episode_plans"),
        screenplay=state.get("screenplay"),
        meta=meta,
    )
    paths["txt_path"] = str(ing.txt_path)
    elapsed = time.perf_counter() - t0
    logger.info(
        "[write] 落盘完成,用时 %.1f 秒,共 %d 个文件",
        elapsed, len(paths),
    )

    update: WorkflowState = {"output_paths": paths}
    # mode >= SCREENPLAY 时三件套齐全,可装完整 FinalReport。
    if mode_includes(mode, RunMode.SCREENPLAY):
        update["final_report"] = FinalReport(
            screenplay=state["screenplay"],
            characters=state["characters"],
            beats=state["beats"],
            meta=meta,
        )
    return update


# ---------------------------------------------------------------------------
# 构图
# ---------------------------------------------------------------------------


def build_graph():
    """编译主 workflow。

    ``analysis`` 节点内部按 ``state['mode']`` 早停,所以 graph 本身不需要按
    mode 分支 —— 一套 graph 跑所有 mode。

    LLM 由各 agent 自治(``agents/<agent_name>/llm.json``),workflow 不再 build /
    传递 LLM。
    """
    from langgraph.graph import StateGraph, END

    graph = StateGraph(WorkflowState)

    graph.add_node("detect_input", _node_detect_input)
    graph.add_node("convert_epub", _node_convert_epub)
    graph.add_node("ingest_and_batch", _node_ingest_and_batch)
    graph.add_node("analysis", _node_analysis)
    graph.add_node("write", _node_write)

    graph.set_entry_point("detect_input")
    graph.add_conditional_edges(
        "detect_input",
        _route_after_detect,
        {"epub": "convert_epub", "txt": "ingest_and_batch"},
    )
    graph.add_edge("convert_epub", "ingest_and_batch")
    graph.add_edge("ingest_and_batch", "analysis")
    graph.add_edge("analysis", "write")
    graph.add_edge("write", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# 顶层入口
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """``run()`` 的扁平返回。各阶段产物按 mode 选择性填充。"""

    mode: RunMode
    output_dir: Path
    output_paths: Dict[str, str] = field(default_factory=dict)
    ingest_result: Optional[IngestResult] = None
    characters: Optional[CharacterList] = None
    beats: Optional[BeatList] = None
    episode_plans: Optional[EpisodePlanList] = None
    screenplay: Optional[Screenplay] = None
    # mode >= SCREENPLAY 时有(三件套 + meta 齐全)
    final_report: Optional[FinalReport] = None


def run(config: RunConfig) -> RunResult:
    """按 ``RunConfig`` 跑工作流。

    跑到哪一阶段由 ``config.mode`` 决定 —— 详见 ``RunMode``。
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = (Path(config.output_dir) / timestamp).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    logger.info("本次运行输出目录:%s", out_root)

    setup_agent_traces(out_root)
    logger.info(
        "workflow 启动:mode=%s input=%s output=%s max_batch_chars=%d "
        "max_total_chars=%d episode=%ds window=%d rewrite_K=%d prev_tail_K=%d",
        config.mode.value, config.input, out_root, config.max_batch_chars,
        config.max_total_chars, config.target_episode_duration_sec,
        config.recent_beats_window, config.rewrite_window,
        config.shot_prev_tail_window,
    )

    graph = build_graph()
    initial: WorkflowState = {
        "input_path": str(config.input),
        "output_dir": str(out_root),
        "mode": config.mode,
        "max_batch_chars": config.max_batch_chars,
        "max_total_chars": config.max_total_chars,
        "target_episode_duration_sec": config.target_episode_duration_sec,
        "recent_beats_window": config.recent_beats_window,
        "rewrite_window": config.rewrite_window,
        "shot_prev_tail_window": config.shot_prev_tail_window,
    }

    t_start = time.perf_counter()
    final: WorkflowState = graph.invoke(initial)
    elapsed = time.perf_counter() - t_start
    logger.info(
        "workflow 完成(mode=%s),总用时 %.1f 秒(%.1f 分钟)",
        config.mode.value, elapsed, elapsed / 60,
    )

    return RunResult(
        mode=config.mode,
        output_dir=out_root,
        output_paths=dict(final.get("output_paths", {})),
        ingest_result=final.get("ingest_result"),
        characters=final.get("characters"),
        beats=final.get("beats"),
        episode_plans=final.get("episode_plans"),
        screenplay=final.get("screenplay"),
        final_report=final.get("final_report"),
    )


__all__ = [
    "WorkflowState",
    "RunResult",
    "build_graph",
    "run",
    "setup_agent_traces",
    "collect_agent_llm_info_for_mode",
]
