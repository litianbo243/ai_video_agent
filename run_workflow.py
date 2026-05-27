"""跑 workflow 的统一入口。

工作流只有一条:``workflows.novel_analysis``。跑到哪一阶段由
``RunConfig.mode``(写在 JSON 配置文件里)决定:

* ``"character"``  —— 仅人物档案
* ``"beat"``       —— + 剧情段
* ``"episode"``    —— + 分集规划
* ``"screenplay"`` —— + 逐集分镜(完整流水线,默认)

用法::

    # 1. 编辑 configs/run_config.json 把 mode 改成你想跑的那档
    # 2. python run_workflow.py

调试 prompt 时用浅 mode 早停(快 + 省 token),需要完整剧本时切回 screenplay。
4 档 mode 跑出的中间产物是**严格超集**关系,所以早停跟全跑的字符 / 剧情段
完全一致。

**注:** 本工程不再维护独立的场景视觉档案。``beat_segmenter`` 自己产出场景
name 作为字符串 label,``shot_director`` 在写每镜 ``description`` 时把视觉
环境直接写进画面。
"""

from __future__ import annotations

import logging
from pathlib import Path

from configs import RunMode, load_config
from workflows import novel_analysis


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
# 主入口
# ---------------------------------------------------------------------------


def main(config_path: Path) -> None:
    """按 ``config.mode`` 跑工作流并打印摘要。"""
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
# 入口:改 configs/run_config.json 里的 mode 字段切换早停阶段
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    main(Path("configs/run_config.json"))
