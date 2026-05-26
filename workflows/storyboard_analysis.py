"""storyboard_analysis workflow:逐批抽人物 + 切剧情段 → 全书 plan → 串行分镜。

跟 ``novel_analysis`` 的两阶段节拍完全一致(阶段 1 逐 batch 跑 character +
beat;阶段 2 全书 plan 后串行分镜),保证独立跑这个 workflow 时拿到的结果跟
novel_analysis 内部跑出来的语义一致。唯一区别是本 workflow **不写 FinalReport**,
只产 character / beat / screenplay 三件套。

DAG::

    START
      ▼
    ingest ──▶ analyze ──▶ END

* **独立跑**(``run(config)``):2 步全跑。
* **被父 workflow 调**:已注入 ``ingest_result``,ingest 节点 no-op,只跑 analyze。

公开 API:

* ``run(config) -> (IngestResult, CharacterList, BeatList, ScreenplayAnalysis)``
    顶层。
* ``run_with_batches(batches, *, title, target_duration_sec, context_window)
   -> (CharacterList, BeatList, ScreenplayAnalysis)``
    纯计算批循环(analyze 节点用,也可在 notebook 直接调)。注意:character /
    beat 都是这个循环顺手产出的,不能再像旧版那样从外部传入。
* ``build_graph()`` / ``State``
    LangGraph 编译 + 状态契约。

LLM 由各 agent 自治(``agents/extract_storyboard/llm.json`` 及上游 agents 同理),
workflow 不再 build / 传递 LLM。
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Iterable, List, Tuple, TypedDict

from configs import RunConfig
from skills.batch_chapters import Batch
from skills.book_ingest import IngestResult, ingest_book
from agents import episode_planner, extract_beats, extract_characters
from agents.extract_beats import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_REWRITE_WINDOW,
)
from agents.extract_storyboard import DEFAULT_PREV_TAIL_K
from schemas.beat import Beat, BeatList
from schemas.character import Character, CharacterList
from schemas.storyboard import Episode, ScreenplayAnalysis, Storyboard
from workflows.episode_pipeline import build_episode_from_plan

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据契约
# ---------------------------------------------------------------------------


class State(TypedDict, total=False):
    """storyboard_analysis workflow 的状态(``TypedDict``)。

    跟旧版相比:不再有 ``characters`` / ``beats`` 作为「上游产物」输入,
    全部由本 workflow 内部跟 storyboard 交错产出。父 workflow 调本子-workflow
    时只需注入 ``ingest_result``(+ 选填 target/window)。
    """

    config: RunConfig
    ingest_result: IngestResult
    target_duration_sec: int
    context_window: int
    rewrite_window: int
    prev_tail_window: int

    # 输出
    characters: CharacterList
    beats: BeatList
    screenplay: ScreenplayAnalysis


# ---------------------------------------------------------------------------
# 编排:批循环(analyze 节点 + 外部直接调用 共用)
# ---------------------------------------------------------------------------


def run_with_batches(
    batches: Iterable[Batch],
    *,
    title: str = "",
    target_duration_sec: int = 180,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    rewrite_window: int = DEFAULT_REWRITE_WINDOW,
    prev_tail_window: int = DEFAULT_PREV_TAIL_K,
) -> Tuple[CharacterList, BeatList, ScreenplayAnalysis]:
    """跑完所有 batch,顺带产出人物表 + 剧情段 + 分集分镜(config-free 纯计算)。

    **两阶段节拍**:

    阶段 1(逐 batch,渐进态)::

        for batch in batches:
            extract_characters.extract_for_batch  → 更新 known_chars
            extract_beats.extract_for_batch       → 更新 beats_so_far
                                                    (可重写末尾 rewrite_window 段)

    阶段 2(全书 beat 抽完后,终态)::

        episode_planner.plan_episodes(beats_so_far, known_chars, ...) → plan_list
        for ep_index, plan in enumerate(plan_list.plans, start=1):
            build_episode_from_plan(plan, member_beats, ...) → +1 集
    """
    batches = list(batches)
    known_chars: Dict[str, Character] = {}
    beats_so_far: List[Beat] = []
    batch_lookup: Dict[int, Batch] = {b.index: b for b in batches}

    logger.info("=" * 60)
    logger.info(
        "[storyboard_analysis] 阶段 1:共 %d 批,目标每集 %d 秒,近段窗口 %d,"
        "rewrite K=%d,prev_tail K=%d",
        len(batches), target_duration_sec, context_window, rewrite_window,
        prev_tail_window,
    )
    logger.info("=" * 60)

    t_total = time.perf_counter()
    for i, batch in enumerate(batches, start=1):
        t0 = time.perf_counter()

        ch_result = extract_characters.extract_for_batch(
            batch, known_chars, title=title,
        )
        if ch_result.renames:
            extract_beats.apply_character_renames(beats_so_far, ch_result.renames)
        extract_beats.extract_for_batch(
            batch, beats_so_far, known_chars,
            title=title,
            context_window=context_window,
            rewrite_window=rewrite_window,
        )

        elapsed = time.perf_counter() - t0
        logger.info(
            "[storyboard_analysis] %d/%d (batch=%d) 完成,%.1f 秒。"
            "累计 %d 人 / %d 段",
            i, len(batches), batch.index, elapsed,
            len(known_chars), len(beats_so_far),
        )

    # 阶段 2:全书 beat 抽完后,plan + 串行分镜
    logger.info("=" * 60)
    logger.info(
        "[storyboard_analysis] 阶段 2(plan + 串行分镜):%d 段 beat → 待规划",
        len(beats_so_far),
    )
    logger.info("=" * 60)

    plan_list = episode_planner.plan_episodes(
        beats_so_far, known_chars,
        target_duration_sec=target_duration_sec,
        title=title,
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
                "[storyboard_analysis] 集 %d (%s) 的 beat_indices 含未知 index=%s",
                ep_index, plan.title, missing,
            )
        if not member_beats:
            logger.warning(
                "[storyboard_analysis] 集 %d (%s) 无有效 member_beats,跳过",
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
            target_duration_sec=target_duration_sec,
        )
        episodes.append(ep)
        prev_tail = (
            ep.storyboards[-prev_tail_window:]
            if (ep.storyboards and prev_tail_window > 0)
            else []
        )

        elapsed = time.perf_counter() - t_ep
        logger.info(
            "[storyboard_analysis] 集 %d/%d (%s) 完成,%.1f 秒,%d 镜",
            ep_index, len(plan_list.plans), plan.title, elapsed, len(ep.storyboards),
        )

    total_elapsed = time.perf_counter() - t_total
    logger.info(
        "[storyboard_analysis] 全部完成:%d 人 / %d 段 / %d 集,合计 %.1f 秒(%.1f 分钟)",
        len(known_chars), len(beats_so_far), len(episodes),
        total_elapsed, total_elapsed / 60,
    )

    return (
        CharacterList(characters=list(known_chars.values())),
        BeatList(beats=beats_so_far),
        ScreenplayAnalysis(episodes=episodes),
    )


# ---------------------------------------------------------------------------
# LangGraph 节点(全部幂等:state 里有就跳过)
# ---------------------------------------------------------------------------


def _node_ingest(state: State) -> State:
    """ingest 顺手把 target_duration_sec / context_window / rewrite_window /
    prev_tail_window 也从 config 里拎出来(只有独立跑会进此分支;父 workflow
    调时这些通常已注入)。"""
    out: State = {}
    if "ingest_result" not in state:
        if "config" not in state:
            raise RuntimeError(
                "storyboard_analysis workflow 启动时既无 ingest_result 也无 config"
            )
        config = state["config"]
        ing = ingest_book(
            config.input,
            max_batch_chars=config.max_batch_chars,
            max_total_chars=config.max_total_chars,
        )
        out["ingest_result"] = ing
    if "target_duration_sec" not in state and "config" in state:
        out["target_duration_sec"] = state["config"].target_episode_duration_sec
    if "context_window" not in state and "config" in state:
        out["context_window"] = state["config"].recent_beats_window
    if "rewrite_window" not in state and "config" in state:
        out["rewrite_window"] = state["config"].rewrite_window
    if "prev_tail_window" not in state and "config" in state:
        out["prev_tail_window"] = state["config"].storyboard_prev_tail_window
    return out


def _node_analyze(state: State) -> State:
    ing = state["ingest_result"]
    characters, beats, screenplay = run_with_batches(
        ing.batches,
        title=ing.title,
        target_duration_sec=state.get("target_duration_sec", 180),
        context_window=state.get("context_window", DEFAULT_CONTEXT_WINDOW),
        rewrite_window=state.get("rewrite_window", DEFAULT_REWRITE_WINDOW),
        prev_tail_window=state.get("prev_tail_window", DEFAULT_PREV_TAIL_K),
    )
    return {
        "characters": characters,
        "beats":      beats,
        "screenplay": screenplay,
    }


# ---------------------------------------------------------------------------
# 构图 + 顶层入口
# ---------------------------------------------------------------------------


def build_graph():
    """编译 storyboard_analysis workflow(2 节点:ingest → analyze)。"""
    from langgraph.graph import StateGraph, END

    g = StateGraph(State)
    g.add_node("ingest", _node_ingest)
    g.add_node("analyze", _node_analyze)

    g.set_entry_point("ingest")
    g.add_edge("ingest", "analyze")
    g.add_edge("analyze", END)
    return g.compile()


def run(
    config: RunConfig,
) -> Tuple[IngestResult, CharacterList, BeatList, ScreenplayAnalysis]:
    """从 ``RunConfig`` 出发跑完整 storyboard workflow,顺带把 character + beat
    也跑出来。"""
    graph = build_graph()
    final: State = graph.invoke({"config": config})
    missing = [
        k for k in ("ingest_result", "characters", "beats", "screenplay")
        if k not in final
    ]
    if missing:
        raise RuntimeError(f"storyboard_analysis workflow 结束但缺少:{missing}")
    return (
        final["ingest_result"],
        final["characters"],
        final["beats"],
        final["screenplay"],
    )


__all__ = ["State", "build_graph", "run", "run_with_batches"]
