"""beat_analysis workflow:把小说原文逐批切成"剧情大纲段"(Beat)。

依赖 character + setting 产物,但**自己负责拿到它们**——内嵌
``character_analysis`` / ``setting_analysis`` 子-graph,并发跑出 roster +
collection 后再做 beat。runner 不再做编排,只是薄壳。

DAG::

    START
      ▼
    ingest ──fan-out──┬──▶ character_analysis ──┐
                      │                          ├──▶ analyze ──▶ END
                      └──▶ setting_analysis ─────┘

**幂等节点**:``ingest`` / ``character_analysis`` / ``setting_analysis`` 都先看
state 里对应输出是否已就位,有就 ``return {}`` 跳过。这样:

* **独立跑**(``run(config)``):只给 ``config``,4 步全跑。
* **被父 workflow 调**:已经注入 ``ingest_result`` / ``characters`` /
  ``settings``,前 3 个节点全部 no-op,只跑 analyze。

公开 API:

* ``run(config) -> (IngestResult, CharacterRoster, SettingCollection, BeatList)``
    顶层。
* ``run_with_batches(batches, characters, settings, llm, *, title="") -> BeatList``
    纯计算批循环(analyze 节点用,也可在 notebook 直接调)。
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
from skills.extract_beats import Beat, BeatExtraction, BeatList, extract_for_batch
from skills.extract_characters import Character, CharacterRoster
from skills.extract_settings import Setting, SettingCollection
from workflows import character_analysis, setting_analysis

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据契约
# ---------------------------------------------------------------------------


class State(TypedDict, total=False):
    """beat_analysis workflow 的状态(``TypedDict``)。

    上游 4 个字段都是"父注入 或 节点自产":父 workflow 调 beat 时通常
    ``ingest_result`` / ``characters`` / ``settings`` 都已就位,对应节点会
    自动跳过。独立跑时只给 ``config``,workflow 自己把链路拉满。
    """

    # 启动注入(独立跑给 config / 父调给后三项)
    config: RunConfig
    ingest_result: IngestResult
    characters: CharacterRoster
    settings: SettingCollection

    # 输出
    beats: BeatList


# ---------------------------------------------------------------------------
# 纯计算:批循环
# ---------------------------------------------------------------------------


def _merge(beats: List[Beat], delta: BeatExtraction, batch_index: int) -> None:
    """追加新段(赋 index + related_batches);延续段把本 batch 加进 related_batches。"""
    for idx in delta.extended_beat_indices:
        i = idx - 1  # 1-based → 0-based
        if 0 <= i < len(beats):
            if batch_index not in beats[i].related_batches:
                beats[i].related_batches.append(batch_index)

    for draft in delta.new_beats:
        beat = Beat(
            **draft.model_dump(),
            index=len(beats) + 1,
            related_batches=[batch_index],
        )
        beats.append(beat)


def run_with_batches(
    batches: Iterable[Batch],
    characters: CharacterRoster,
    settings: SettingCollection,
    llm: LLMClient,
    *,
    title: str = "",
) -> BeatList:
    """跑完所有 batch,返回合并后的剧情段列表(config-free 纯计算)。"""
    batches = list(batches)
    char_lookup: Dict[str, Character] = {c.name: c for c in characters.characters}
    setting_lookup: Dict[str, Setting] = {s.name: s for s in settings.settings}
    beats: List[Beat] = []

    logger.info("=" * 60)
    logger.info(
        "[beat_analysis] 启动:共 %d 批,引用域 %d 人 / %d 场景",
        len(batches), len(char_lookup), len(setting_lookup),
    )
    logger.info("=" * 60)

    t_total = time.perf_counter()
    for i, batch in enumerate(batches, start=1):
        t0 = time.perf_counter()
        delta = extract_for_batch(
            batch, beats, char_lookup, setting_lookup, llm, title=title,
        )
        _merge(beats, delta, batch.index)
        elapsed = time.perf_counter() - t0
        logger.info(
            "[beat_analysis] %d/%d (batch=%d) 完成,用时 %.1f 秒。累计 %d 段",
            i, len(batches), batch.index, elapsed, len(beats),
        )

    total_elapsed = time.perf_counter() - t_total
    logger.info(
        "[beat_analysis] 全部完成:%d 段,合计用时 %.1f 秒(%.1f 分钟)",
        len(beats), total_elapsed, total_elapsed / 60,
    )

    return BeatList(beats=beats)


# ---------------------------------------------------------------------------
# LangGraph 节点(全部幂等:state 里有就跳过)
# ---------------------------------------------------------------------------


def _node_ingest(state: State) -> State:
    if "ingest_result" in state:
        return {}
    if "config" not in state:
        raise RuntimeError("beat_analysis workflow 启动时既无 ingest_result 也无 config")
    config = state["config"]
    ing = ingest_book(
        config.input,
        max_batch_chars=config.max_batch_chars,
        max_total_chars=config.max_total_chars,
    )
    return {"ingest_result": ing}


def _node_character_analysis(state: State, sub_graph: Any) -> State:
    """invoke 已编译的 character_analysis 子-graph。"""
    if "characters" in state:
        return {}
    final: character_analysis.State = sub_graph.invoke(
        {"ingest_result": state["ingest_result"]}
    )
    return {"characters": final["roster"]}


def _node_setting_analysis(state: State, sub_graph: Any) -> State:
    if "settings" in state:
        return {}
    final: setting_analysis.State = sub_graph.invoke(
        {"ingest_result": state["ingest_result"]}
    )
    return {"settings": final["collection"]}


def _node_analyze(state: State, llm: LLMClient) -> State:
    ing = state["ingest_result"]
    beats = run_with_batches(
        ing.batches, state["characters"], state["settings"], llm, title=ing.title,
    )
    return {"beats": beats}


# ---------------------------------------------------------------------------
# 构图 + 顶层入口
# ---------------------------------------------------------------------------


def build_graph(llm: LLMClient):
    """编译 beat_analysis workflow,内嵌 character + setting 子-graph。"""
    from langgraph.graph import StateGraph, END

    char_graph = character_analysis.build_graph(llm)
    setting_graph = setting_analysis.build_graph(llm)

    g = StateGraph(State)
    g.add_node("ingest", _node_ingest)
    g.add_node("character_analysis",
               lambda s: _node_character_analysis(s, char_graph))
    g.add_node("setting_analysis",
               lambda s: _node_setting_analysis(s, setting_graph))
    g.add_node("analyze", lambda s: _node_analyze(s, llm))

    g.set_entry_point("ingest")
    # fan-out:character + setting 并行
    g.add_edge("ingest", "character_analysis")
    g.add_edge("ingest", "setting_analysis")
    # fan-in:两边都完成才跑 analyze
    g.add_edge("character_analysis", "analyze")
    g.add_edge("setting_analysis", "analyze")
    g.add_edge("analyze", END)
    return g.compile()


def run(
    config: RunConfig,
) -> Tuple[IngestResult, CharacterRoster, SettingCollection, BeatList]:
    """从 ``RunConfig`` 出发跑完整 beat workflow,顺带把 character + setting 也跑出来。"""
    llm = get_client(config.llm)
    graph = build_graph(llm)
    final: State = graph.invoke({"config": config})
    missing = [
        k for k in ("ingest_result", "characters", "settings", "beats") if k not in final
    ]
    if missing:
        raise RuntimeError(f"beat_analysis workflow 结束但缺少:{missing}")
    return (
        final["ingest_result"],
        final["characters"],
        final["settings"],
        final["beats"],
    )


__all__ = ["State", "build_graph", "run", "run_with_batches"]
