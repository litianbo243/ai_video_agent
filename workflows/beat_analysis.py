"""beat_analysis workflow:逐批交错抽人物 + 切剧情段。

跟 ``novel_analysis`` 的批循环节拍一致(都是「读到本批末尾的渐进 character →
当批 beat」),所以独立跑这个 workflow 时 beat agent 看到的 character 跟
novel_analysis 内的 beat agent 看到的完全一样,保证语义统一。

唯一区别是本 workflow **不跑 storyboard**,只产 character + beat 两件套。

DAG::

    START
      ▼
    ingest ──▶ analyze ──▶ END

* **独立跑**(``run(config)``):2 步全跑。
* **被父 workflow 调**:已注入 ``ingest_result``,ingest 节点 no-op,只跑 analyze。

**场景的处理**:beat agent 自己产出 ``setting_refs`` 作为字符串 label,本工程
不再维护独立的场景视觉档案,storyboarder 写每镜 ``description`` 时直接把视觉
环境写到画面里。

公开 API:

* ``run(config) -> (IngestResult, CharacterList, BeatList)``
    顶层。
* ``run_with_batches(batches, *, title, context_window, rewrite_window)
   -> (CharacterList, BeatList)``
    纯计算批循环(analyze 节点用,也可在 notebook 直接调)。注意:character
    是这个循环顺手产出的,不能再像旧版那样从外部传入。
* ``build_graph()`` / ``State``
    LangGraph 编译 + 状态契约。

LLM 由各 agent 自治(``agents/extract_beats/llm.json``、character agent 同理),
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
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_REWRITE_WINDOW,
)
from schemas.beat import Beat, BeatList
from schemas.character import Character, CharacterList

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据契约
# ---------------------------------------------------------------------------


class State(TypedDict, total=False):
    """beat_analysis workflow 的状态(``TypedDict``)。

    跟旧版相比:不再有 ``characters`` 作为「上游产物」输入,character 由本 workflow
    内部跟 beat 交错产出。父 workflow 调本子-workflow 时只需注入 ``ingest_result``。

    本 sub-workflow **不关心目标视频时长**(beat 只按戏剧浓度切自然边界,集层
    聚合由下游 ``episode_planner`` 负责),所以没有 ``target_duration_sec``。

    ``context_window`` 控制 beat agent 在 prompt 里展示的「此前最近 N 段」窗口大小,
    独立跑时从 ``config.recent_beats_window`` 取;父 workflow 调时直接注入。

    ``rewrite_window`` 控制每批 LLM 必须复述/修订的「末尾 K 段」数量,独立跑时
    从 ``config.rewrite_window`` 取。
    """

    config: RunConfig
    ingest_result: IngestResult
    context_window: int
    rewrite_window: int

    characters: CharacterList
    beats: BeatList


# ---------------------------------------------------------------------------
# 编排:批循环(analyze 节点 + 外部直接调用 共用)
# ---------------------------------------------------------------------------


def run_with_batches(
    batches: Iterable[Batch],
    *,
    title: str = "",
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    rewrite_window: int = DEFAULT_REWRITE_WINDOW,
) -> Tuple[CharacterList, BeatList]:
    """跑完所有 batch,顺带产出人物表 + 剧情段列表(config-free 纯计算)。

    **节拍**(每 batch):

    1) ``extract_characters.extract_for_batch`` 更新 ``known_chars``
    2) ``extract_beats.extract_for_batch`` 更新 ``beats_so_far``;LLM 可以
       重写 ``beats_so_far`` 末尾 ``rewrite_window`` 段(自然跨批续写)
    """
    batches = list(batches)
    known_chars: Dict[str, Character] = {}
    beats_so_far: List[Beat] = []

    logger.info("=" * 60)
    logger.info(
        "[beat_analysis] 启动:共 %d 批,近段窗口 %d,rewrite K=%d",
        len(batches), context_window, rewrite_window,
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
            "[beat_analysis] %d/%d (batch=%d) 完成,%.1f 秒。"
            "累计 %d 人 / %d 段",
            i, len(batches), batch.index, elapsed,
            len(known_chars), len(beats_so_far),
        )

    total_elapsed = time.perf_counter() - t_total
    logger.info(
        "[beat_analysis] 全部完成:%d 人 / %d 段,合计 %.1f 秒(%.1f 分钟)",
        len(known_chars), len(beats_so_far),
        total_elapsed, total_elapsed / 60,
    )

    return (
        CharacterList(characters=list(known_chars.values())),
        BeatList(beats=beats_so_far),
    )


# ---------------------------------------------------------------------------
# LangGraph 节点(全部幂等:state 里有就跳过)
# ---------------------------------------------------------------------------


def _node_ingest(state: State) -> State:
    """ingest 顺手把 context_window / rewrite_window 也从 config 里拎出来
    (只有独立跑会进此分支;父 workflow 调时这俩通常已注入)。"""
    out: State = {}
    if "ingest_result" not in state:
        if "config" not in state:
            raise RuntimeError(
                "beat_analysis workflow 启动时既无 ingest_result 也无 config"
            )
        config = state["config"]
        ing = ingest_book(
            config.input,
            max_batch_chars=config.max_batch_chars,
            max_total_chars=config.max_total_chars,
        )
        out["ingest_result"] = ing
    if "context_window" not in state and "config" in state:
        out["context_window"] = state["config"].recent_beats_window
    if "rewrite_window" not in state and "config" in state:
        out["rewrite_window"] = state["config"].rewrite_window
    return out


def _node_analyze(state: State) -> State:
    ing = state["ingest_result"]
    characters, beats = run_with_batches(
        ing.batches,
        title=ing.title,
        context_window=state.get("context_window", DEFAULT_CONTEXT_WINDOW),
        rewrite_window=state.get("rewrite_window", DEFAULT_REWRITE_WINDOW),
    )
    return {"characters": characters, "beats": beats}


# ---------------------------------------------------------------------------
# 构图 + 顶层入口
# ---------------------------------------------------------------------------


def build_graph():
    """编译 beat_analysis workflow(2 节点:ingest → analyze)。"""
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
) -> Tuple[IngestResult, CharacterList, BeatList]:
    """从 ``RunConfig`` 出发跑完整 beat workflow,顺带把 character 也跑出来。"""
    graph = build_graph()
    final: State = graph.invoke({"config": config})
    missing = [k for k in ("ingest_result", "characters", "beats") if k not in final]
    if missing:
        raise RuntimeError(f"beat_analysis workflow 结束但缺少:{missing}")
    return (
        final["ingest_result"],
        final["characters"],
        final["beats"],
    )


__all__ = ["State", "build_graph", "run", "run_with_batches"]
