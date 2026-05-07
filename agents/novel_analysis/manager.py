"""Manager:执行一次完整的小说分析任务。

把输入路径、输出目录、LLM 配置串起来,跑完整条流水线,返回结果。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from agents.novel_analysis.workflow import run_workflow
from llm.client import get_client
from schema.config import RunConfig
from schema.novel_analysis import BatchState, FinalReport

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_ROOT = Path("output")


@dataclass
class RunResult:
    final_state: BatchState
    report: FinalReport
    output_dir: Path
    output_paths: dict


def run(config: RunConfig) -> RunResult:
    """按 config 执行一次完整的小说分析。"""
    src = Path(config.input).resolve()
    if not src.is_file():
        raise FileNotFoundError(f"输入文件不存在:{src}")
    if src.suffix.lower() not in {".txt", ".epub"}:
        raise ValueError(
            f"不支持的文件类型 {src.suffix!r};当前只接受 .txt 与 .epub。"
        )

    base = Path(config.output_dir) if config.output_dir else DEFAULT_OUTPUT_ROOT
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = (base / timestamp).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    logger.info("本次运行输出目录:%s", out_root)

    llm = get_client(config.llm)
    logger.info(
        "Manager 启动:input=%s output=%s max_batch_chars=%d max_total_chars=%d "
        "episode=%ds llm=%s @ %s",
        src, out_root, config.max_batch_chars, config.max_total_chars,
        config.target_episode_duration_sec, llm.model, llm.base_url,
    )

    final_state = run_workflow(
        input_path=str(src),
        output_dir=str(out_root),
        max_batch_chars=config.max_batch_chars,
        max_total_chars=config.max_total_chars,
        llm=llm,
        target_episode_duration_sec=config.target_episode_duration_sec,
    )

    if final_state.final_report is None:
        raise RuntimeError("流水线结束但未产生 final_report。")

    return RunResult(
        final_state=final_state,
        report=final_state.final_report,
        output_dir=out_root,
        output_paths=dict(final_state.output_paths),
    )


__all__ = ["run", "RunResult"]
