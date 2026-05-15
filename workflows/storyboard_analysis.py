"""storyboard_analysis workflow:把每段 Beat 展开成一集分镜清单。

依赖 beat + character 产物,但**自己负责拿到它们**——内嵌
``beat_analysis`` 子-graph(beat 又内嵌 character 子-graph),
所以独立跑也能从 ``config`` 一路跑到分镜。runner 不做编排,只是薄壳。

DAG::

    START
      ▼
    ingest ──▶ beat_analysis ──▶ analyze ──▶ END

其中 ``beat_analysis`` 是子-workflow,内部还有自己的 character 节点。
storyboard 这一层不重复写 char 节点,那是 beat 的职责。

**幂等节点**:``ingest`` / ``beat_analysis`` 都先看 state 里对应输出是否已就位,
有就 ``return {}`` 跳过。这样:

* **独立跑**(``run(config)``):只给 ``config``,全链路跑完。
* **被父 workflow 调**:已注入 ``ingest_result`` / ``characters`` /
  ``beats``,storyboard 的 ingest + beat 节点全部 no-op,只跑 analyze。

公开 API:

* ``run(config) -> (IngestResult, CharacterRoster, BeatList, ScreenplayAnalysis)``
    顶层。
* ``run_with_batches(beats, characters, batches, llm, *, target_duration_sec=180, title="") -> ScreenplayAnalysis``
    纯计算循环(逐 Beat 跑一次 LLM,产单集 Episode,组装成 ScreenplayAnalysis)。
* ``build_graph(llm)`` / ``State``
    LangGraph 编译 + 状态契约。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Iterable, List, Tuple, TypedDict

from configs import RunConfig
from llm.client import LLMClient, get_client
from skills.batch_chapters import Batch
from skills.book_ingest import IngestResult, ingest_book
from agents.extract_beats import BeatList
from agents.extract_characters import Character, CharacterRoster
from agents.extract_storyboard import Episode, ScreenplayAnalysis, storyboard_beat
from workflows import beat_analysis

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据契约
# ---------------------------------------------------------------------------


class State(TypedDict, total=False):
    """storyboard_analysis workflow 的状态(``TypedDict``)。

    上游 4 个字段都是"父注入 或 节点自产";父 workflow 调 storyboard 时通常
    全部已就位,对应节点会自动跳过。独立跑时只给 ``config`` 即可。
    """

    config: RunConfig
    ingest_result: IngestResult
    characters: CharacterRoster
    beats: BeatList

    # 可选(不给则用默认)
    target_duration_sec: int
    context_window: int

    # 输出
    screenplay: ScreenplayAnalysis


# ---------------------------------------------------------------------------
# 纯计算:逐 Beat 循环
# ---------------------------------------------------------------------------


def run_with_batches(
    beats: BeatList,
    characters: CharacterRoster,
    batches: Iterable[Batch],
    llm: LLMClient,
    *,
    target_duration_sec: int = 180,
    title: str = "",
) -> ScreenplayAnalysis:
    """跑完所有 Beat,返回 ScreenplayAnalysis(config-free 纯计算)。"""
    beats_list = beats.beats
    char_lookup: Dict[str, Character] = {c.name: c for c in characters.characters}
    batch_lookup: Dict[int, Batch] = {b.index: b for b in batches}

    logger.info("=" * 60)
    logger.info(
        "[storyboard_analysis] 启动:%d 段 → %d 集,目标每集 %d 秒",
        len(beats_list), len(beats_list), target_duration_sec,
    )
    logger.info("=" * 60)

    t_total = time.perf_counter()
    episodes: List[Episode] = []
    for i, beat in enumerate(beats_list, start=1):
        logger.info(
            "[storyboard_analysis] 第 %d/%d 集(beat #%d · %s)开始",
            i, len(beats_list), beat.index, beat.title,
        )
        t0 = time.perf_counter()
        ep = storyboard_beat(
            beat, char_lookup, batch_lookup, llm,
            target_duration_sec=target_duration_sec,
        )
        elapsed = time.perf_counter() - t0
        logger.info(
            "[storyboard_analysis] 第 %d/%d 集完成,用时 %.1f 秒,产 %d 镜",
            i, len(beats_list), elapsed, len(ep.storyboards),
        )
        episodes.append(ep)

    total_elapsed = time.perf_counter() - t_total
    total_storyboards = sum(len(ep.storyboards) for ep in episodes)
    logger.info(
        "[storyboard_analysis] 全部完成:%d 集 / %d 镜,合计用时 %.1f 秒(%.1f 分钟)",
        len(episodes), total_storyboards, total_elapsed, total_elapsed / 60,
    )

    return ScreenplayAnalysis(
        title=title,
        logline="",  # 暂留空,后续可加专门的 logline_writer 子-workflow
        episodes=episodes,
    )


# ---------------------------------------------------------------------------
# LangGraph 节点(全部幂等:state 里有就跳过)
# ---------------------------------------------------------------------------


def _node_ingest(state: State) -> State:
    if "ingest_result" in state:
        return {}
    if "config" not in state:
        raise RuntimeError("storyboard_analysis workflow 启动时既无 ingest_result 也无 config")
    config = state["config"]
    ing = ingest_book(
        config.input,
        max_batch_chars=config.max_batch_chars,
        max_total_chars=config.max_total_chars,
    )
    return {"ingest_result": ing}


def _node_beat_analysis(state: State, sub_graph: Any) -> State:
    """invoke 已编译的 beat_analysis 子-graph,顺带把 characters 透传上来。"""
    if "beats" in state:
        return {}
    inj: Dict[str, Any] = {"ingest_result": state["ingest_result"]}
    if "characters" in state:
        inj["characters"] = state["characters"]
    if "target_duration_sec" in state:
        inj["target_duration_sec"] = state["target_duration_sec"]
    if "context_window" in state:
        inj["context_window"] = state["context_window"]
    final: beat_analysis.State = sub_graph.invoke(inj)
    return {
        "characters": final["characters"],
        "beats":      final["beats"],
    }


def _node_analyze(state: State, llm: LLMClient) -> State:
    ing = state["ingest_result"]
    screenplay = run_with_batches(
        state["beats"], state["characters"], ing.batches, llm,
        target_duration_sec=state.get("target_duration_sec", 180),
        title=ing.title,
    )
    return {"screenplay": screenplay}


# ---------------------------------------------------------------------------
# 构图 + 顶层入口
# ---------------------------------------------------------------------------


def build_graph(llm: LLMClient):
    """编译 storyboard_analysis workflow,内嵌 beat 子-graph(beat 又内嵌 character)。"""
    from langgraph.graph import StateGraph, END

    beat_graph = beat_analysis.build_graph(llm)

    g = StateGraph(State)
    g.add_node("ingest", _node_ingest)
    g.add_node("beat_analysis", lambda s: _node_beat_analysis(s, beat_graph))
    g.add_node("analyze", lambda s: _node_analyze(s, llm))

    g.set_entry_point("ingest")
    g.add_edge("ingest", "beat_analysis")
    g.add_edge("beat_analysis", "analyze")
    g.add_edge("analyze", END)
    return g.compile()


def run(
    config: RunConfig,
) -> Tuple[IngestResult, CharacterRoster, BeatList, ScreenplayAnalysis]:
    """从 ``RunConfig`` 出发跑完整 storyboard workflow,顺带把 character + beat 也跑出来。"""
    llm = get_client(config.llm)
    graph = build_graph(llm)
    final: State = graph.invoke({
        "config": config,
        "target_duration_sec": config.target_episode_duration_sec,
        "context_window": config.recent_beats_window,
    })
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
