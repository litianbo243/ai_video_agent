"""setting_analysis workflow:逐批抽取场景 + 增量合并。

公开 API:

* ``run(config) -> (IngestResult, SettingCollection)``
    顶层,runner 用。建 LLM → 编译并执行 graph。
* ``run_with_batches(batches, llm, *, title="") -> SettingCollection``
    纯计算批循环(config-free)。
* ``build_graph(llm)`` / ``State``
    LangGraph 编译 + 状态契约。

两种启动姿势(由 conditional entry 路由):
* 独立跑:外面给 ``config``,自己 ingest。
* 被父 workflow 调用:已注入 ``ingest_result``,跳过 ingest。
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Iterable, Tuple, TypedDict

from configs import RunConfig
from llm.client import LLMClient, get_client
from skills.batch_chapters import Batch
from skills.book_ingest import IngestResult, ingest_book
from agents.extract_settings import (
    Setting,
    SettingCollection,
    SettingExtraction,
    extract_for_batch,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据契约
# ---------------------------------------------------------------------------


class State(TypedDict, total=False):
    """setting_analysis workflow 的状态(``TypedDict``,与 LangGraph 合并语义一致)。"""

    config: RunConfig
    ingest_result: IngestResult
    collection: SettingCollection


# ---------------------------------------------------------------------------
# 纯计算:批循环
# ---------------------------------------------------------------------------


def _merge(known: Dict[str, Setting], delta: SettingExtraction) -> None:
    """同名 → description 非空才覆盖;新名 → 新增并赋全局 index。"""
    for draft in delta.new_or_updated_settings:
        existing = known.get(draft.name)
        if existing is None:
            s = Setting(**draft.model_dump(), index=len(known) + 1)
            known[draft.name] = s
            continue
        if draft.description:
            existing.description = draft.description


def run_with_batches(
    batches: Iterable[Batch],
    llm: LLMClient,
    *,
    title: str = "",
) -> SettingCollection:
    """跑完所有 batch,返回合并后的场景表(config-free 纯计算)。"""
    batches = list(batches)
    known: Dict[str, Setting] = {}

    logger.info("=" * 60)
    logger.info("[setting_analysis] 启动:共 %d 批", len(batches))
    logger.info("=" * 60)

    t_total = time.perf_counter()
    for i, batch in enumerate(batches, start=1):
        t0 = time.perf_counter()
        delta = extract_for_batch(batch, known, llm, title=title)
        _merge(known, delta)
        elapsed = time.perf_counter() - t0
        logger.info(
            "[setting_analysis] %d/%d (batch=%d) 完成,用时 %.1f 秒。累计 %d 处",
            i, len(batches), batch.index, elapsed, len(known),
        )

    total_elapsed = time.perf_counter() - t_total
    logger.info(
        "[setting_analysis] 全部完成:%d 处,合计用时 %.1f 秒(%.1f 分钟)",
        len(known), total_elapsed, total_elapsed / 60,
    )

    return SettingCollection(settings=list(known.values()))


# ---------------------------------------------------------------------------
# LangGraph 节点
# ---------------------------------------------------------------------------


def _route_entry(state: State) -> str:
    if "ingest_result" in state:
        return "skip_ingest"
    if "config" not in state:
        raise RuntimeError("setting_analysis workflow 启动时既无 ingest_result 也无 config")
    return "needs_ingest"


def _node_ingest(state: State) -> State:
    config = state["config"]
    ing = ingest_book(
        config.input,
        max_batch_chars=config.max_batch_chars,
        max_total_chars=config.max_total_chars,
    )
    return {"ingest_result": ing}


def _node_analyze(state: State, llm: LLMClient) -> State:
    ing = state["ingest_result"]
    coll = run_with_batches(ing.batches, llm, title=ing.title)
    return {"collection": coll}


# ---------------------------------------------------------------------------
# 构图 + 顶层入口
# ---------------------------------------------------------------------------


def build_graph(llm: LLMClient):
    from langgraph.graph import StateGraph, END

    g = StateGraph(State)
    g.add_node("ingest", _node_ingest)
    g.add_node("analyze", lambda s: _node_analyze(s, llm))
    g.set_conditional_entry_point(
        _route_entry,
        {"needs_ingest": "ingest", "skip_ingest": "analyze"},
    )
    g.add_edge("ingest", "analyze")
    g.add_edge("analyze", END)
    return g.compile()


def run(config: RunConfig) -> Tuple[IngestResult, SettingCollection]:
    """从 ``RunConfig`` 出发跑完整 setting workflow。"""
    llm = get_client(config.llm)
    graph = build_graph(llm)
    final: State = graph.invoke({"config": config})
    if "ingest_result" not in final or "collection" not in final:
        raise RuntimeError("setting_analysis workflow 结束但 ingest_result / collection 缺失")
    return final["ingest_result"], final["collection"]


__all__ = ["State", "build_graph", "run", "run_with_batches"]
