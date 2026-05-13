"""novel_analysis 父-workflow:组合 4 个子-workflow,产出最终剧本结构。

公开 API:

* ``run(config) -> RunResult``
    顶层,runner 用。派生带时间戳的输出目录 + 建 LLM + 编译并执行父-graph,
    把最终 ``FinalReport`` / 输出路径打包返回。
* ``build_graph(llm)``
    编译父-workflow(含 4 个子-workflow 的内嵌 compiled graph)。
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
                                                       │
                              ┌────── fan-out ─────────┴──────┐
                              ▼                                ▼
                      character_analysis              setting_analysis
                              └────────── fan-in ─────────────┘
                                              ▼
                                        beat_analysis
                                              ▼
                                      storyboard_analysis
                                              ▼
                                            write
                                              ▼
                                            END

每个 ``*_analysis`` 节点的实现都是 **invoke 对应子-workflow 的 compiled graph**,
而不是直接调内部函数。这样:

* 5 个 workflow(character / setting / beat / storyboard / 本父-workflow)形态完全一致;
* 子-workflow 独立跑时由 ``runner → manager.run(config)`` 触发,
  父-workflow 调用时则注入 ``ingest_result`` 跳过 ingest 节点(由各子 workflow 的
  ``set_conditional_entry_point`` 自动路由);
* 后续给 novel_analysis 加缓存(读上次的 characters.json 跳过 character 子-workflow)
  只需要在这一层加 cache 检查节点,子-workflow 不动。

``character_analysis`` 和 ``setting_analysis`` 没有相互依赖,LangGraph 会**并发执行**
(墙钟时间近乎砍半;LLM 调用费不变)。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, TypedDict

from configs import RunConfig
from llm.client import LLMClient, get_client
from skills.book_ingest import IngestResult, load_and_batch_txt
from skills.epub_to_txt import epub_to_txt
from agents.extract_beats import BeatList
from agents.extract_characters import CharacterRoster
from agents.extract_settings import SettingCollection
from agents.extract_storyboard import ScreenplayAnalysis
from skills.file_io import FinalReport, ReportMeta, write_final_report
from workflows import (
    beat_analysis,
    character_analysis,
    setting_analysis,
    storyboard_analysis,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State:节点之间流通的薄数据;只放需要跨节点传的内容
# ---------------------------------------------------------------------------


class WorkflowState(TypedDict, total=False):
    """LangGraph 在节点之间传递的状态对象(``TypedDict``)。

    保持薄:子-workflow 内部的迭代/累加状态不放这里,只放跨节点产物。
    """

    # --- 启动时填入 ---
    input_path: str
    output_dir: str
    max_batch_chars: int
    max_total_chars: int
    target_episode_duration_sec: int
    llm_base_url: str
    llm_model: str

    # --- detect_input / convert_epub 之后(ingest 节点会消费) ---
    txt_path: str

    # --- ingest_and_batch 之后(替代旧的 title / raw_chars / total_chars / batches 四件套) ---
    ingest_result: IngestResult

    # --- 各子-workflow 输出(可并行写,key 互不冲突)---
    characters: CharacterRoster
    settings: SettingCollection
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


def _node_character_analysis(state: WorkflowState, sub_graph: Any) -> WorkflowState:
    """invoke 已编译的 character_analysis 子-workflow,ingest_result 已就位 → 跳过子图 ingest 节点。"""
    final: character_analysis.State = sub_graph.invoke(
        {"ingest_result": state["ingest_result"]}
    )
    return {"characters": final["roster"]}


def _node_setting_analysis(state: WorkflowState, sub_graph: Any) -> WorkflowState:
    final: setting_analysis.State = sub_graph.invoke(
        {"ingest_result": state["ingest_result"]}
    )
    return {"settings": final["collection"]}


def _node_beat_analysis(state: WorkflowState, sub_graph: Any) -> WorkflowState:
    final: beat_analysis.State = sub_graph.invoke({
        "ingest_result": state["ingest_result"],
        "characters": state["characters"],
        "settings": state["settings"],
    })
    return {"beats": final["beats"]}


def _node_storyboard_analysis(state: WorkflowState, sub_graph: Any) -> WorkflowState:
    final: storyboard_analysis.State = sub_graph.invoke({
        "ingest_result": state["ingest_result"],
        "beats": state["beats"],
        "characters": state["characters"],
        "settings": state["settings"],
        "target_duration_sec": state["target_episode_duration_sec"],
    })
    return {"screenplay": final["screenplay"]}


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
        llm_base_url=state["llm_base_url"],
        llm_model=state["llm_model"],
    )
    report = FinalReport(
        screenplay=state["screenplay"],
        characters=state["characters"],
        settings=state["settings"],
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


def build_graph(llm: LLMClient):
    """编译父 workflow。4 个子-workflow 的 compiled graph 在这里一次性建好,闭包注入节点。"""
    from langgraph.graph import StateGraph, END

    # 子-workflow 各编译一次(节点 lambda 通过闭包持有,避免每次 invoke 重新编译)
    char_graph = character_analysis.build_graph(llm)
    setting_graph = setting_analysis.build_graph(llm)
    beat_graph = beat_analysis.build_graph(llm)
    storyboard_graph = storyboard_analysis.build_graph(llm)

    graph = StateGraph(WorkflowState)

    graph.add_node("detect_input", _node_detect_input)
    graph.add_node("convert_epub", _node_convert_epub)
    graph.add_node("ingest_and_batch", _node_ingest_and_batch)
    graph.add_node("character_analysis",
                   lambda s: _node_character_analysis(s, char_graph))
    graph.add_node("setting_analysis",
                   lambda s: _node_setting_analysis(s, setting_graph))
    graph.add_node("beat_analysis",
                   lambda s: _node_beat_analysis(s, beat_graph))
    graph.add_node("storyboard_analysis",
                   lambda s: _node_storyboard_analysis(s, storyboard_graph))
    graph.add_node("write", _node_write)

    graph.set_entry_point("detect_input")

    graph.add_conditional_edges(
        "detect_input",
        _route_after_detect,
        {"epub": "convert_epub", "txt": "ingest_and_batch"},
    )
    graph.add_edge("convert_epub", "ingest_and_batch")

    # fan-out:char + setting 并行(两条独立的边)
    graph.add_edge("ingest_and_batch", "character_analysis")
    graph.add_edge("ingest_and_batch", "setting_analysis")

    # fan-in:beat 等两边都完成
    graph.add_edge("character_analysis", "beat_analysis")
    graph.add_edge("setting_analysis", "beat_analysis")

    graph.add_edge("beat_analysis", "storyboard_analysis")
    graph.add_edge("storyboard_analysis", "write")
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

    llm = get_client(config.llm)
    logger.info(
        "novel_analysis workflow 启动:input=%s output=%s max_batch_chars=%d "
        "max_total_chars=%d episode=%ds llm=%s @ %s",
        config.input, out_root, config.max_batch_chars, config.max_total_chars,
        config.target_episode_duration_sec, llm.model, llm.base_url,
    )

    graph = build_graph(llm)
    initial: WorkflowState = {
        "input_path": str(config.input),
        "output_dir": str(out_root),
        "max_batch_chars": config.max_batch_chars,
        "max_total_chars": config.max_total_chars,
        "target_episode_duration_sec": config.target_episode_duration_sec,
        "llm_base_url": llm.base_url,
        "llm_model": llm.model,
    }

    t_start = time.perf_counter()
    final: WorkflowState = graph.invoke(initial, config={"recursion_limit": 50})
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


__all__ = ["WorkflowState", "RunResult", "build_graph", "run"]
