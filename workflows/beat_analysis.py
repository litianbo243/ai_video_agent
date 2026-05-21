"""beat_analysis workflow:把小说原文逐批切成"剧情大纲段"(Beat)。

依赖 character 产物,但**自己负责拿到它**——内嵌 ``character_analysis`` 子-graph,
跑出 roster 后再做 beat。runner 不再做编排,只是薄壳。

DAG::

    START
      ▼
    ingest ──▶ character_analysis ──▶ analyze ──▶ END

**幂等节点**:``ingest`` / ``character_analysis`` 都先看 state 里对应输出是否已就位,
有就 ``return {}`` 跳过。这样:

* **独立跑**(``run(config)``):只给 ``config``,3 步全跑。
* **被父 workflow 调**:已经注入 ``ingest_result`` / ``characters``,前 2 个节点全部 no-op,
  只跑 analyze。

**场景的处理:** beat agent 自己产出 ``setting_refs`` 作为字符串 label,跨 batch 时用
"已用 name 集"提示 LLM 沿用——本工程不再维护独立的场景视觉档案,
那是 storyboarder 从原文 + LLM 常识里写到每镜 ``description`` 的事。

公开 API:

* ``run(config) -> (IngestResult, CharacterRoster, BeatList)``
    顶层。
* ``run_with_batches(batches, characters, *, title="") -> BeatList``
    纯计算批循环(analyze 节点用,也可在 notebook 直接调)。
* ``build_graph()`` / ``State``
    LangGraph 编译 + 状态契约。

LLM 由各 agent 自治(``agents/extract_beats/llm.json``、character agent 同理),
workflow 不再 build / 传递 LLM。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Iterable, List, Set, Tuple, TypedDict

from configs import RunConfig
from skills.batch_chapters import Batch
from skills.book_ingest import IngestResult, ingest_book
from agents.extract_beats import (
    Beat,
    BeatList,
    DEFAULT_CONTEXT_WINDOW,
    extract_for_batch,
)
from agents.extract_characters import Character, CharacterRoster
from workflows import character_analysis

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据契约
# ---------------------------------------------------------------------------


class State(TypedDict, total=False):
    """beat_analysis workflow 的状态(``TypedDict``)。

    上游 3 个字段都是"父注入 或 节点自产":父 workflow 调 beat 时通常
    ``ingest_result`` / ``characters`` 都已就位,对应节点会自动跳过。
    独立跑时只给 ``config``,workflow 自己把链路拉满。

    ``target_duration_sec`` 用来对齐 beat 切粒度(短视频 → 单集紧凑,长视频 →
    单集可承载更多剧情)。独立跑时从 ``config.target_episode_duration_sec`` 取;
    父 workflow 调时直接注入。

    ``context_window`` 控制 beat agent 在 prompt 里展示的「此前最近 N 段」窗口大小,
    独立跑时从 ``config.recent_beats_window`` 取;父 workflow 调时直接注入。
    """

    # 启动注入(独立跑给 config / 父调给其余几项)
    config: RunConfig
    ingest_result: IngestResult
    characters: CharacterRoster
    target_duration_sec: int
    context_window: int

    # 输出
    beats: BeatList


# ---------------------------------------------------------------------------
# 编排:批循环
# 抽取与合并的具体逻辑都在 agents.extract_beats 内。
# ---------------------------------------------------------------------------


def run_with_batches(
    batches: Iterable[Batch],
    characters: CharacterRoster,
    *,
    title: str = "",
    target_duration_sec: int = 180,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
) -> BeatList:
    """跑完所有 batch,返回合并后的剧情段列表(config-free 纯计算)。

    ``target_duration_sec`` 透传给 ``extract_for_batch``,让 beat 粒度跟
    单集目标时长对齐。``context_window`` 控制 prompt 里展示的最近 beats 数量。
    """
    batches = list(batches)
    char_lookup: Dict[str, Character] = {c.name: c for c in characters.characters}
    setting_names: Set[str] = set()
    beats: List[Beat] = []

    logger.info("=" * 60)
    logger.info(
        "[beat_analysis] 启动:共 %d 批,人物名录 %d 人,目标集时长 %d 秒,"
        "近段窗口 %d",
        len(batches), len(char_lookup), target_duration_sec, context_window,
    )
    logger.info("=" * 60)

    t_total = time.perf_counter()
    for i, batch in enumerate(batches, start=1):
        t0 = time.perf_counter()
        extract_for_batch(
            batch, beats, char_lookup, setting_names,
            title=title, target_duration_sec=target_duration_sec,
            context_window=context_window,
        )
        elapsed = time.perf_counter() - t0
        logger.info(
            "[beat_analysis] %d/%d (batch=%d) 完成,用时 %.1f 秒。"
            "累计 %d 段 / %d 处场景 name",
            i, len(batches), batch.index, elapsed, len(beats), len(setting_names),
        )

    total_elapsed = time.perf_counter() - t_total
    logger.info(
        "[beat_analysis] 全部完成:%d 段 / %d 处场景 name,合计用时 %.1f 秒(%.1f 分钟)",
        len(beats), len(setting_names), total_elapsed, total_elapsed / 60,
    )

    return BeatList(beats=beats)


# ---------------------------------------------------------------------------
# LangGraph 节点(全部幂等:state 里有就跳过)
# ---------------------------------------------------------------------------


def _node_ingest(state: State) -> State:
    """ingest 顺手把 target_duration_sec / context_window 也从 config 里拎出来
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
    if "target_duration_sec" not in state and "config" in state:
        out["target_duration_sec"] = state["config"].target_episode_duration_sec
    if "context_window" not in state and "config" in state:
        out["context_window"] = state["config"].recent_beats_window
    return out


def _node_character_analysis(state: State, sub_graph: Any) -> State:
    """invoke 已编译的 character_analysis 子-graph。"""
    if "characters" in state:
        return {}
    final: character_analysis.State = sub_graph.invoke(
        {"ingest_result": state["ingest_result"]}
    )
    return {"characters": final["roster"]}


def _node_analyze(state: State) -> State:
    ing = state["ingest_result"]
    beats = run_with_batches(
        ing.batches, state["characters"],
        title=ing.title,
        target_duration_sec=state.get("target_duration_sec", 180),
        context_window=state.get("context_window", DEFAULT_CONTEXT_WINDOW),
    )
    return {"beats": beats}


# ---------------------------------------------------------------------------
# 构图 + 顶层入口
# ---------------------------------------------------------------------------


def build_graph():
    """编译 beat_analysis workflow,内嵌 character 子-graph。LLM 由各 agent 自治。"""
    from langgraph.graph import StateGraph, END

    char_graph = character_analysis.build_graph()

    g = StateGraph(State)
    g.add_node("ingest", _node_ingest)
    g.add_node("character_analysis",
               lambda s: _node_character_analysis(s, char_graph))
    g.add_node("analyze", _node_analyze)

    g.set_entry_point("ingest")
    g.add_edge("ingest", "character_analysis")
    g.add_edge("character_analysis", "analyze")
    g.add_edge("analyze", END)
    return g.compile()


def run(
    config: RunConfig,
) -> Tuple[IngestResult, CharacterRoster, BeatList]:
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
