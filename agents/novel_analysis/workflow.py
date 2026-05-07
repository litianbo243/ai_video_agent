"""LangGraph 状态机:把 skills 与子-agent 串起来,执行一次小说分析。

节点流向::

    detect_input --[epub]--> convert_epub --> ingest_and_batch
    detect_input --[txt]--> ingest_and_batch
    ingest_and_batch --> analyze
    analyze --[more]--> analyze
    analyze --[done]--> finalize --> write --> END

``finalize`` 节点内部按 Beat 顺序执行 N 次 LLM 调用(每段 Beat 产一集
Episode),不拆成独立 LangGraph 节点是为了避免在节点间传递中间态。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict

from agents.novel_analysis.beat_extractor import extract_beats
from agents.novel_analysis.character_extractor import extract_characters
from agents.novel_analysis.episode_storyboarder import storyboard_beat
from agents.novel_analysis.setting_extractor import extract_settings
from llm.client import LLMClient
from schema.novel_analysis import (
    BatchState,
    BeatList,
    CharacterRoster,
    Episode,
    FinalReport,
    ReportMeta,
    ScreenplayAnalysis,
    SettingCollection,
)
from skills.batch_chapters import (
    load_text,
    split_into_batches,
)
from skills.epub_to_txt import epub_to_txt
from skills.file_io import save_checkpoint, write_final_report

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 节点
# ---------------------------------------------------------------------------


def _node_detect_input(state: BatchState) -> Dict[str, Any]:
    src = Path(state.input_path)
    if not src.exists():
        raise FileNotFoundError(f"输入路径不存在:{src}")
    size_mb = src.stat().st_size / (1024 * 1024)
    logger.info(
        "[detect_input] 检测到输入:%s(%s,%.2f MB)",
        src, src.suffix.lower() or "(无后缀)", size_mb,
    )
    return {}


def _route_after_detect(state: BatchState) -> str:
    suffix = Path(state.input_path).suffix.lower()
    route = "epub" if suffix == ".epub" else "txt"
    logger.info("[detect_input] 路由:%s → 走 %s 分支", suffix or "(无)", route)
    return route


def _node_convert_epub(state: BatchState) -> Dict[str, Any]:
    logger.info("[convert_epub] 开始转换 EPUB → TXT:%s", state.input_path)
    t0 = time.perf_counter()
    txt_path = epub_to_txt(state.input_path)
    elapsed = time.perf_counter() - t0
    logger.info("[convert_epub] 完成,用时 %.1f 秒,产物:%s", elapsed, txt_path)
    return {"txt_path": str(txt_path)}


def _node_ingest_and_batch(state: BatchState) -> Dict[str, Any]:
    """从 ``.txt`` 读全文,可选截断到 ``max_total_chars``,再按 ``max_batch_chars`` 分批。"""
    src = state.txt_path or state.input_path
    logger.info("[ingest_and_batch] 加载文本:%s", src)
    t0 = time.perf_counter()
    title, body = load_text(Path(src))
    raw_chars = len(body)

    # 整书字数上限,超出直接尾部截断
    if state.max_total_chars > 0 and raw_chars > state.max_total_chars:
        body = body[: state.max_total_chars]
        logger.warning(
            "[ingest_and_batch] 全文 %d 字超过 max_total_chars=%d,从尾部截断到 %d 字",
            raw_chars, state.max_total_chars, len(body),
        )

    total_chars = len(body)
    batches = split_into_batches(body, max_chars=state.max_batch_chars)
    elapsed = time.perf_counter() - t0
    logger.info(
        "[ingest_and_batch] 用时 %.1f 秒:书名=%r,原始 %d 字 → 实际 %d 字,打包 %d 批(max_batch_chars=%d)",
        elapsed, title, raw_chars, total_chars, len(batches), state.max_batch_chars,
    )
    return {
        "txt_path": str(src),
        "title": title,
        "total_chars": total_chars,
        "batches": batches,
        "cursor": 0,
    }


def _node_analyze(state: BatchState, llm: LLMClient) -> Dict[str, Any]:
    """单批分析:依次调 character/setting/beat 三个抽取 agent。

    三个 agent **必须按此顺序**:character + setting 是 beat 的引用域,
    beat 在它们之后跑才能用到。
    """
    if state.cursor >= len(state.batches):
        return {}
    batch = state.batches[state.cursor]
    total = len(state.batches)
    pos = state.cursor + 1  # 1-based 显示用

    if pos == 1:
        logger.info("=" * 60)
        logger.info("[analyze] 进入分析阶段:共 %d 批,每批 ≤ %d 字", total, state.max_batch_chars)
        logger.info("=" * 60)

    logger.info(
        "[analyze] 批次 %d/%d(batch.index=%d,%d 字)开始",
        pos, total, batch.index, batch.char_count,
    )
    t0 = time.perf_counter()

    char_delta = extract_characters(state, batch, llm)
    state.merge_characters(char_delta)

    setting_delta = extract_settings(state, batch, llm)
    state.merge_settings(setting_delta)

    beat_delta = extract_beats(state, batch, llm)
    state.merge_beats(beat_delta, batch_index=batch.index)

    save_checkpoint(state)
    elapsed = time.perf_counter() - t0
    logger.info(
        "[analyze] 批次 %d/%d 完成,用时 %.1f 秒。累计:人物=%d 场景=%d 剧情段=%d",
        pos, total, elapsed,
        len(state.characters), len(state.settings), len(state.beats),
    )
    return {
        "characters": state.characters,
        "settings": state.settings,
        "beats": state.beats,
        "last_completed_batch": state.last_completed_batch,
        "cursor": state.cursor + 1,
    }


def _route_after_analyze(state: BatchState) -> str:
    return "more" if state.cursor < len(state.batches) else "done"


# ---------------------------------------------------------------------------
# 综合阶段:逐段 Beat → 单集 storyboard → 组装 FinalReport(单节点内部串行)
# ---------------------------------------------------------------------------


def _node_finalize(state: BatchState, llm: LLMClient) -> Dict[str, Any]:
    """跑完所有 batch 后:按 Beat 顺序逐个产单集分镜,组装最终报告。

    这一步内部是 N 次 LLM 调用(N = Beat 数 = 集数),目前串行。
    未来要并行可在此节点用 ``ThreadPoolExecutor`` 并发 ``storyboard_beat``。
    """
    n_beats = len(state.beats)
    logger.info("=" * 60)
    logger.info(
        "[finalize] 进入分集分镜阶段:共 %d 段(= %d 集)目标每集 %d 秒",
        n_beats, n_beats, state.target_episode_duration_sec,
    )
    logger.info("=" * 60)

    # Step 1: 逐段 Beat 产单集分镜
    t_total = time.perf_counter()
    episodes: list[Episode] = []
    for i, beat in enumerate(state.beats, start=1):
        logger.info("[finalize] 第 %d/%d 集(beat #%d · %s)开始",
                    i, n_beats, beat.index, beat.title)
        t0 = time.perf_counter()
        ep = storyboard_beat(beat, state, llm)
        elapsed = time.perf_counter() - t0
        logger.info(
            "[finalize] 第 %d/%d 集完成,用时 %.1f 秒,产 %d 镜",
            i, n_beats, elapsed, len(ep.storyboards),
        )
        episodes.append(ep)
    total_elapsed = time.perf_counter() - t_total
    total_storyboards = sum(len(ep.storyboards) for ep in episodes)
    logger.info(
        "[finalize] 全部分镜完成,合计 %d 集 / %d 镜,用时 %.1f 秒",
        len(episodes), total_storyboards, total_elapsed,
    )

    screenplay = ScreenplayAnalysis(
        title=state.title,
        logline="",  # 暂留空,后续可加专门的 logline_writer 子-agent
        episodes=episodes,
    )

    # Step 2: 组装 FinalReport
    roster = CharacterRoster(characters=list(state.characters.values()))
    settings = SettingCollection(settings=list(state.settings.values()))
    beats_pack = BeatList(beats=state.beats)
    meta = ReportMeta(
        source_path=state.input_path,
        txt_path=state.txt_path,
        title=state.title,
        total_chars=state.total_chars,
        batch_count=len(state.batches),
        max_batch_chars=state.max_batch_chars,
        max_total_chars=state.max_total_chars,
        llm_base_url=llm.base_url,
        llm_model=llm.model,
    )
    report = FinalReport(
        screenplay=screenplay,
        characters=roster,
        settings=settings,
        beats=beats_pack,
        meta=meta,
    )
    logger.info("[finalize] FinalReport 组装完成")
    return {"final_report": report}


def _node_write(state: BatchState) -> Dict[str, Any]:
    if state.final_report is None:
        raise RuntimeError("finalize 节点没有产出 report")
    logger.info("[write] 开始落盘最终产物到 %s", state.output_dir)
    t0 = time.perf_counter()
    paths = write_final_report(state.final_report, Path(state.output_dir))
    paths["txt_path"] = state.txt_path
    elapsed = time.perf_counter() - t0
    logger.info(
        "[write] 落盘完成,用时 %.1f 秒,共 %d 个文件:\n  %s",
        elapsed, len(paths),
        "\n  ".join(f"{k}: {v}" for k, v in paths.items()),
    )
    return {"output_paths": paths}


# ---------------------------------------------------------------------------
# 构图
# ---------------------------------------------------------------------------


def build_graph(llm: LLMClient):
    """编译 LangGraph StateGraph(惰性 import langgraph)。"""
    from langgraph.graph import StateGraph, END

    graph = StateGraph(BatchState)

    graph.add_node("detect_input", _node_detect_input)
    graph.add_node("convert_epub", _node_convert_epub)
    graph.add_node("ingest_and_batch", _node_ingest_and_batch)
    graph.add_node("analyze", lambda s: _node_analyze(s, llm))
    graph.add_node("finalize", lambda s: _node_finalize(s, llm))
    graph.add_node("write", _node_write)

    graph.set_entry_point("detect_input")

    graph.add_conditional_edges(
        "detect_input",
        _route_after_detect,
        {"epub": "convert_epub", "txt": "ingest_and_batch"},
    )
    graph.add_edge("convert_epub", "ingest_and_batch")
    graph.add_edge("ingest_and_batch", "analyze")
    graph.add_conditional_edges(
        "analyze",
        _route_after_analyze,
        {"more": "analyze", "done": "finalize"},
    )
    graph.add_edge("finalize", "write")
    graph.add_edge("write", END)

    return graph.compile()


def run_workflow(
    input_path: str,
    output_dir: str,
    max_batch_chars: int,
    llm: LLMClient,
    max_total_chars: int = 0,
    target_episode_duration_sec: int = 180,
    recursion_limit: int = 5000,
) -> BatchState:
    """同步编译并执行图,返回最终 state。"""
    logger.info(
        "run_workflow 启动:input=%s output=%s max_batch_chars=%d max_total_chars=%d "
        "episode=%ds llm=%s @ %s",
        input_path, output_dir, max_batch_chars, max_total_chars,
        target_episode_duration_sec, llm.model, llm.base_url,
    )
    t_start = time.perf_counter()
    graph = build_graph(llm)
    initial = BatchState(
        input_path=str(input_path),
        output_dir=str(output_dir),
        max_batch_chars=max_batch_chars,
        max_total_chars=max_total_chars,
        target_episode_duration_sec=target_episode_duration_sec,
    )
    final_dict = graph.invoke(initial, config={"recursion_limit": recursion_limit})
    elapsed = time.perf_counter() - t_start
    logger.info("run_workflow 完成,总用时 %.1f 秒(%.1f 分钟)", elapsed, elapsed / 60)
    return BatchState.model_validate(final_dict)


__all__ = ["build_graph", "run_workflow"]
