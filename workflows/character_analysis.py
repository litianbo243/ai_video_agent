"""character_analysis workflow:逐批抽取人物 + 增量合并。

公开 API:

* ``run(config) -> (IngestResult, CharacterRoster)``
    顶层,runner 用。建 LLM → 编译并执行 graph(自带 ingest 节点)。

* ``run_with_batches(batches, llm, *, title="") -> CharacterRoster``
    纯计算批循环(config-free)。workflow 的 analyze 节点直接调它,也可在
    notebook / 测试里手工备好 batches 直接调,不走 LangGraph。

* ``build_graph(llm)``
    编译子-workflow,父-workflow 用 ``sub_graph.invoke(...)`` 调用。

* ``State``
    workflow 的状态数据契约(``TypedDict``,与 LangGraph 原生 dict 合并语义一致)。

两种启动姿势(由 conditional entry 路由):

* **独立跑**(``runner → run(config)``):外面给 ``config``,workflow 自己 ingest。
* **被父 workflow 调用**:外面已注入 ``ingest_result``,跳过 ingest 节点。
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Iterable, Tuple, TypedDict

from configs import RunConfig
from llm.client import LLMClient, get_client
from skills.batch_chapters import Batch
from skills.book_ingest import IngestResult, ingest_book
from agents.extract_characters import (
    Character,
    CharacterExtraction,
    CharacterRoster,
    extract_for_batch,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据契约
# ---------------------------------------------------------------------------


class State(TypedDict, total=False):
    """character_analysis workflow 的状态。

    ``total=False`` → 所有键可选;LangGraph 节点返回的 dict 会被自动合并到当前 state。
    """

    # 启动注入(二选一,由 _route_entry 决定走哪条边)
    config: RunConfig
    ingest_result: IngestResult

    # 输出
    roster: CharacterRoster


# ---------------------------------------------------------------------------
# 纯计算:批循环(analyze 节点 + 外部直接调用 共用)
# ---------------------------------------------------------------------------


def _merge(known: Dict[str, Character], delta: CharacterExtraction) -> None:
    """同名 → 融合;新名 → 新增并赋全局 index。

    aliases 取并集;appearance / personality 非空才覆盖。
    新人物若三大字段全空(LLM 凑出来的空壳),跳过不建档 + warn。
    """
    for draft in delta.new_or_updated_characters:
        existing = known.get(draft.name)
        if existing is None:
            if not draft.appearance and not draft.personality and not draft.aliases:
                logger.warning(
                    "[character_analysis] 跳过空壳新人物 name=%r(无外貌/性格/别名)",
                    draft.name,
                )
                continue
            ch = Character(**draft.model_dump(), index=len(known) + 1)
            known[draft.name] = ch
            continue
        existing.aliases = sorted(set(existing.aliases) | set(draft.aliases))
        if draft.appearance:
            existing.appearance = draft.appearance
        if draft.personality:
            existing.personality = draft.personality


def run_with_batches(
    batches: Iterable[Batch],
    llm: LLMClient,
    *,
    title: str = "",
) -> CharacterRoster:
    """跑完所有 batch,返回合并后的人物表(config-free 纯计算)。"""
    batches = list(batches)
    known: Dict[str, Character] = {}

    logger.info("=" * 60)
    logger.info("[character_analysis] 启动:共 %d 批", len(batches))
    logger.info("=" * 60)

    t_total = time.perf_counter()
    for i, batch in enumerate(batches, start=1):
        t0 = time.perf_counter()
        delta = extract_for_batch(batch, known, llm, title=title)
        _merge(known, delta)
        elapsed = time.perf_counter() - t0
        logger.info(
            "[character_analysis] %d/%d (batch=%d) 完成,用时 %.1f 秒。累计 %d 人",
            i, len(batches), batch.index, elapsed, len(known),
        )

    total_elapsed = time.perf_counter() - t_total
    logger.info(
        "[character_analysis] 全部完成:%d 人,合计用时 %.1f 秒(%.1f 分钟)",
        len(known), total_elapsed, total_elapsed / 60,
    )

    return CharacterRoster(characters=list(known.values()))


# ---------------------------------------------------------------------------
# LangGraph 节点
# ---------------------------------------------------------------------------


def _route_entry(state: State) -> str:
    """有 ingest_result 就跳过 ingest,否则要求 config。"""
    if "ingest_result" in state:
        return "skip_ingest"
    if "config" not in state:
        raise RuntimeError("character_analysis workflow 启动时既无 ingest_result 也无 config")
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
    roster = run_with_batches(ing.batches, llm, title=ing.title)
    return {"roster": roster}


# ---------------------------------------------------------------------------
# 构图 + 顶层入口
# ---------------------------------------------------------------------------


def build_graph(llm: LLMClient):
    """编译 character_analysis workflow。``llm`` 通过闭包注入 analyze 节点。"""
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


def run(config: RunConfig) -> Tuple[IngestResult, CharacterRoster]:
    """从 ``RunConfig`` 出发跑完整 character workflow。"""
    llm = get_client(config.llm)
    graph = build_graph(llm)
    final: State = graph.invoke({"config": config})
    if "ingest_result" not in final or "roster" not in final:
        raise RuntimeError("character_analysis workflow 结束但 ingest_result / roster 缺失")
    return final["ingest_result"], final["roster"]


__all__ = ["State", "build_graph", "run", "run_with_batches"]
