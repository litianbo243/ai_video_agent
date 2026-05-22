"""storyboard_analysis workflow:逐批交错抽人物 + 切剧情段 +(closed beat 立即)出分镜。

跟 ``novel_analysis`` 的批循环节拍完全一致(都是 char → beat → 新 closed beat
立即 storyboard),保证独立跑这个 workflow 时拿到的结果跟 novel_analysis 内部
跑出来的语义一致。唯一区别是本 workflow **不写 FinalReport**,只产
character / beat / screenplay 三件套。

DAG::

    START
      ▼
    ingest ──▶ analyze ──▶ END

* **独立跑**(``run(config)``):2 步全跑。
* **被父 workflow 调**:已注入 ``ingest_result``,ingest 节点 no-op,只跑 analyze。

公开 API:

* ``run(config) -> (IngestResult, CharacterRoster, BeatList, ScreenplayAnalysis)``
    顶层。
* ``run_with_batches(batches, *, title, target_duration_sec, context_window)
   -> (CharacterRoster, BeatList, ScreenplayAnalysis)``
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
from typing import Dict, Iterable, List, Set, Tuple, TypedDict

from configs import RunConfig
from skills.batch_chapters import Batch
from skills.book_ingest import IngestResult, ingest_book
from agents import extract_beats, extract_characters
from agents.extract_beats import (
    Beat,
    BeatList,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_REWRITE_WINDOW,
)
from agents.extract_characters import Character, CharacterRoster
from agents.extract_storyboard import (
    DEFAULT_PREV_TAIL_K,
    Episode,
    ScreenplayAnalysis,
    Storyboard,
    storyboard_beat,
)

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
    characters: CharacterRoster
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
) -> Tuple[CharacterRoster, BeatList, ScreenplayAnalysis]:
    """跑完所有 batch,顺带产出人物表 + 剧情段 + 分集分镜(config-free 纯计算)。

    **节拍**(每 batch):

    1) ``extract_characters.extract_for_batch``  → 更新 ``known_chars``
    2) ``extract_beats.extract_for_batch``       → 更新 ``beats_so_far``;
       LLM 可重写末尾 ``rewrite_window`` 段
    3) 扫 ``beats_so_far[:-rewrite_window]``,尚未 storyboard 的 → 立刻
       ``storyboard_beat``。末尾 K 段还在 LLM 可改窗口里,不送 storyboard

    **尾部 flush**:全部 batch 跑完,把仍在冷却期的末尾 K 段一次性 storyboard
    (那时再无下批,不会被改了)。
    """
    batches = list(batches)
    known_chars: Dict[str, Character] = {}
    beats_so_far: List[Beat] = []
    storyboarded_idx: Set[int] = set()
    episodes: List[Episode] = []
    prev_tail: List[Storyboard] = []  # 上集末尾 K 镜,给下集做画面承接
    batch_lookup: Dict[int, Batch] = {b.index: b for b in batches}

    logger.info("=" * 60)
    logger.info(
        "[storyboard_analysis] 启动:共 %d 批,目标每集 %d 秒,近段窗口 %d,"
        "rewrite K=%d,prev_tail K=%d",
        len(batches), target_duration_sec, context_window, rewrite_window,
        prev_tail_window,
    )
    logger.info("=" * 60)

    t_total = time.perf_counter()
    for i, batch in enumerate(batches, start=1):
        t0 = time.perf_counter()

        extract_characters.extract_for_batch(
            batch, known_chars, title=title,
        )
        extract_beats.extract_for_batch(
            batch, beats_so_far, known_chars,
            title=title,
            target_duration_sec=target_duration_sec,
            context_window=context_window,
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
                    target_duration_sec=target_duration_sec,
                )
                episodes.append(ep)
                storyboarded_idx.add(b.index)
                new_eps += 1
                prev_tail = ep.storyboards[-prev_tail_window:] if (ep.storyboards and prev_tail_window > 0) else []

        elapsed = time.perf_counter() - t0
        logger.info(
            "[storyboard_analysis] %d/%d (batch=%d) 完成,%.1f 秒。"
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
                target_duration_sec=target_duration_sec,
            )
            episodes.append(ep)
            storyboarded_idx.add(b.index)
            prev_tail = ep.storyboards[-prev_tail_window:] if (ep.storyboards and prev_tail_window > 0) else []

    total_elapsed = time.perf_counter() - t_total
    logger.info(
        "[storyboard_analysis] 全部完成:%d 人 / %d 段 / %d 集,合计 %.1f 秒(%.1f 分钟)",
        len(known_chars), len(beats_so_far), len(episodes),
        total_elapsed, total_elapsed / 60,
    )

    return (
        CharacterRoster(characters=list(known_chars.values())),
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
) -> Tuple[IngestResult, CharacterRoster, BeatList, ScreenplayAnalysis]:
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
