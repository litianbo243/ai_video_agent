"""跑 workflow 的统一入口。

提供两个入口:

* ``run_novel_analysis`` —— 跑 ``workflows.novel_analysis`` 主流水线
  按 ``RunConfig.mode`` 早停(``character`` / ``beat`` / ``episode`` / ``screenplay``)

* ``run_generate_character_prompt`` —— 跑 ``workflows.generate_character_prompt``
  输入已产出的 ``characters.json``,生成 SDXL 图像提示词

用法::

    # 跑 novel_analysis:
    #   python run_workflow.py novel_analysis

    # 跑 generate_character_prompt:
    #   python run_workflow.py generate_character_prompt
"""

from __future__ import annotations

import logging
from pathlib import Path

from configs import (
    GenerateCharacterPromptConfig,
    RunMode,
    load_config,
    load_generate_character_prompt_config,
)
from workflows import novel_analysis, generate_character_prompt


# ---------------------------------------------------------------------------
# 公共脚手架
# ---------------------------------------------------------------------------


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _format_agent_llms(mode: RunMode) -> str:
    """渲染当前 mode 涉及的各 agent 的 LLM 摘要,作为输出摘要的一行。"""
    info = novel_analysis.collect_agent_llm_info_for_mode(mode)
    return " | ".join(f"{name}={i.model}@{i.base_url}" for name, i in info.items())


# ---------------------------------------------------------------------------
# novel_analysis 入口
# ---------------------------------------------------------------------------


def run_novel_analysis(config_path: Path) -> None:
    """按 ``config.mode`` 跑 novel_analysis 工作流并打印摘要。"""
    config = load_config(config_path)
    _setup_logging()

    result = novel_analysis.run(config)

    ing = result.ingest_result
    mode = result.mode

    print()
    print("=" * 60)
    print(f"完成(mode={mode.value})。输出目录:{result.output_dir}")
    if ing is not None:
        print(f"  书名:    {ing.title or '(无)'}")
        print(f"  字数:    {ing.total_chars}(原文 {ing.raw_chars})")
        print(f"  批次数:  {len(ing.batches)}")
    print(f"  LLM:     {_format_agent_llms(mode)}")
    if result.characters is not None:
        print(f"  人物表:   {len(result.characters.characters)} 位")
    if result.beats is not None:
        print(f"  剧情段:   {len(result.beats.beats)} 段")
    if result.episode_plans is not None:
        avg = (
            sum(len(p.beat_indices) for p in result.episode_plans.plans)
            / max(len(result.episode_plans.plans), 1)
        )
        print(
            f"  分集:     {len(result.episode_plans.plans)} 集"
            f"(平均 {avg:.1f} beat/集)"
        )
    if result.screenplay is not None:
        shot_total = sum(len(ep.shots) for ep in result.screenplay.episodes)
        print(f"  分镜:     {shot_total} 镜")
    print("产物文件:")
    for kind, path in result.output_paths.items():
        print(f"  - {kind:>20}  {path}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# generate_character_prompt 入口
# ---------------------------------------------------------------------------


def run_generate_character_prompt(config_path: Path) -> None:
    """按配置跑 generate_character_prompt workflow。"""
    cfg: GenerateCharacterPromptConfig = load_generate_character_prompt_config(config_path)
    _setup_logging()

    print()
    print("=" * 60)
    print(f"generate_character_prompt 启动")
    print(f"  characters: {cfg.characters_path}")
    print(f"  output:     {cfg.output_dir}")
    print("=" * 60)

    result = generate_character_prompt.run(cfg.characters_path, cfg.output_dir)

    print()
    print("=" * 60)
    print("完成。")
    print(f"  人物提示词: {len(result.prompts.prompts)} 条")
    print(f"  JSON:       {result.prompts_json_path}")
    print(f"  MD:         {result.prompts_md_path}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------


def _print_usage() -> None:
    print("用法:")
    print("  python run_workflow.py novel_analysis             # 跑 novel_analysis")
    print("  python run_workflow.py generate_character_prompt  # 跑 generate_character_prompt")
    print()
    print("配置文件:")
    print("  configs/novel_analysis_config.json")
    print("  configs/generate_character_prompt_config.json")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        _print_usage()
        sys.exit(1)

    cmd = sys.argv[1].lower()
    if cmd in ("novel_analysis", "novel-analysis", "na"):
        run_novel_analysis(Path("configs/novel_analysis_config.json"))
    elif cmd in ("generate_character_prompt", "generate-character-prompt", "gcp"):
        run_generate_character_prompt(Path("configs/generate_character_prompt_config.json"))
    else:
        print(f"未知命令: {cmd}")
        _print_usage()
        sys.exit(1)
