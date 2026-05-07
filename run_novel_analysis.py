"""小说分析 Agent 的命令行入口。"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from agents.novel_analysis.manager import run
from schema.config import RunConfig


def load_config(path: Path) -> RunConfig:
    """读取、校验、并打印 config 文件,返回 ``RunConfig``。"""
    try:
        cfg_dict = json.loads(path.read_text(encoding="utf-8"))
        config = RunConfig.model_validate(cfg_dict)
    except Exception as e:
        raise ValueError(f"加载 config 失败({path}):\n{e}") from e

    print("=" * 60)
    print("配置已加载:")
    print(f"  输入:       {config.input}")
    print(f"  输出:       {config.output_dir or '(默认 output/)'}")
    print(f"  批字数:     ≤ {config.max_batch_chars}")
    print(f"  全书字数限: {config.max_total_chars or '无'}")
    print(f"  集时长目标: {config.target_episode_duration_sec} 秒")
    print(f"  模型:       {config.llm.model}")
    print(f"  端点:       {config.llm.base_url}")
    if config.llm.api_key_env:
        print(f"  Key 环境变量: {config.llm.api_key_env}")
    print(f"  Key 是否就位: {'是' if config.llm.api_key else '否(本地端点可能不需要)'}")
    print(f"  采样温度:    {config.llm.temperature}")
    print("=" * 60)

    return config


def run_novel_analysis(config_path: Path) -> None:
    config = load_config(config_path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    result = run(config)

    print()
    print("=" * 60)
    print(f"完成。输出目录:{result.output_dir}")
    print(f"  书名:   {result.report.meta.title or '(无)'}")
    print(f"  字数:   {result.report.meta.total_chars}")
    print(f"  批次数: {result.report.meta.batch_count}")
    print(f"  LLM:    {result.report.meta.llm_model} @ {result.report.meta.llm_base_url}")
    print(f"  人物表: {len(result.report.characters.characters)} 位")
    print(f"  场景档案: {len(result.report.settings.settings)} 个")
    print(f"  剧情段: {len(result.report.beats.beats)} 段")
    print(f"  分集:   {len(result.report.screenplay.episodes)} 集")
    storyboard_total = sum(len(ep.storyboards) for ep in result.report.screenplay.episodes)
    print(f"  分镜:   {storyboard_total} 镜")
    print("产物文件:")
    for kind, path in result.output_paths.items():
        print(f"  - {kind:>16}  {path}")
    print("=" * 60)


if __name__ == "__main__":
    config_path = Path("configs/novel_analysis.json")
    run_novel_analysis(config_path)
