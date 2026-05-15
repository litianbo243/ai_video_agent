"""跑 workflow:character / beat / storyboard / novel_analysis 四选一。

在 ``__main__`` 里改最后一行选要跑哪个 workflow。

**注:** 本工程已不再维护独立的场景视觉档案。Beat agent 自己产出场景 name 作为
字符串 label,storyboard agent 在写每镜 ``description`` 时把视觉环境直接写进画面。
所以这里没有 ``run_setting_analysis``。
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from configs import RunConfig, load_config
from workflows import (
    beat_analysis,
    character_analysis,
    novel_analysis,
    storyboard_analysis,
)


# ---------------------------------------------------------------------------
# 公共脚手架
# ---------------------------------------------------------------------------


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _ts_outdir(config: RunConfig) -> Path:
    out_dir = (Path(config.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if config.llm.trace_file is None:
        config.llm.trace_file = str(out_dir / "llm_trace.jsonl")
    return out_dir


# ---------------------------------------------------------------------------
# 4 个 workflow runner
# ---------------------------------------------------------------------------


def run_character_analysis(config_path: Path) -> None:
    """跑 character_analysis 子-workflow:全本人物档案抽取。"""
    config = load_config(config_path)
    _setup_logging()
    out_dir = _ts_outdir(config)

    ing, roster = character_analysis.run(config)

    out_path = out_dir / "characters.json"
    out_path.write_text(
        roster.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print()
    print("=" * 60)
    print(f"完成。输出目录:{out_dir}")
    print(f"  书名:   {ing.title or '(无)'}")
    print(f"  字数:   {ing.total_chars}(原文 {ing.raw_chars})")
    print(f"  批次数: {len(ing.batches)}")
    print(f"  LLM:    {config.llm.model} @ {config.llm.base_url}")
    print(f"  人物表: {len(roster.characters)} 位")
    print(f"产物文件: {out_path}")
    print("=" * 60)


def run_beat_analysis(config_path: Path) -> None:
    """跑 beat_analysis workflow:内部自带 ingest + character + beat 一条龙。"""
    config = load_config(config_path)
    _setup_logging()
    out_dir = _ts_outdir(config)

    ing, roster, beats = beat_analysis.run(config)

    out_paths = {
        "characters": out_dir / "characters.json",
        "beats":      out_dir / "beats.json",
    }
    out_paths["characters"].write_text(
        roster.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8"
    )
    out_paths["beats"].write_text(
        beats.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8"
    )

    setting_names = sorted({s for b in beats.beats for s in b.setting_refs})
    print()
    print("=" * 60)
    print(f"完成。输出目录:{out_dir}")
    print(f"  书名:    {ing.title or '(无)'}")
    print(f"  字数:    {ing.total_chars}(原文 {ing.raw_chars})")
    print(f"  批次数:  {len(ing.batches)}")
    print(f"  LLM:     {config.llm.model} @ {config.llm.base_url}")
    print(f"  人物表:   {len(roster.characters)} 位")
    print(f"  剧情段:   {len(beats.beats)} 段")
    print(f"  场景 name: {len(setting_names)} 处")
    print("产物文件:")
    for kind, path in out_paths.items():
        print(f"  - {kind:>10}  {path}")
    print("=" * 60)


def run_storyboard_analysis(config_path: Path) -> None:
    """跑 storyboard_analysis workflow:内部自带 ingest + character + beat + storyboard 一条龙。"""
    config = load_config(config_path)
    _setup_logging()
    out_dir = _ts_outdir(config)

    ing, roster, beats, screenplay = storyboard_analysis.run(config)

    out_paths = {
        "characters": out_dir / "characters.json",
        "beats":      out_dir / "beats.json",
        "screenplay": out_dir / "screenplay.json",
    }
    out_paths["characters"].write_text(
        roster.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8"
    )
    out_paths["beats"].write_text(
        beats.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8"
    )
    out_paths["screenplay"].write_text(
        screenplay.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8"
    )

    storyboard_total = sum(len(ep.storyboards) for ep in screenplay.episodes)
    setting_names = sorted({s for b in beats.beats for s in b.setting_refs})
    print()
    print("=" * 60)
    print(f"完成。输出目录:{out_dir}")
    print(f"  书名:    {ing.title or '(无)'}")
    print(f"  字数:    {ing.total_chars}(原文 {ing.raw_chars})")
    print(f"  批次数:  {len(ing.batches)}")
    print(f"  LLM:     {config.llm.model} @ {config.llm.base_url}")
    print(f"  人物表:   {len(roster.characters)} 位")
    print(f"  剧情段:   {len(beats.beats)} 段")
    print(f"  场景 name: {len(setting_names)} 处")
    print(f"  分集:     {len(screenplay.episodes)} 集")
    print(f"  分镜:     {storyboard_total} 镜")
    print("产物文件:")
    for kind, path in out_paths.items():
        print(f"  - {kind:>10}  {path}")
    print("=" * 60)


def run_novel_analysis(config_path: Path) -> None:
    """跑 novel_analysis 父-workflow:人物 + 剧情段 + 分集分镜。"""
    config = load_config(config_path)
    _setup_logging()

    result = novel_analysis.run(config)

    print()
    print("=" * 60)
    print(f"完成。输出目录:{result.output_dir}")
    report = result.report
    setting_names = sorted({s for b in report.beats.beats for s in b.setting_refs})
    print(f"  书名:    {report.meta.title or '(无)'}")
    print(f"  字数:    {report.meta.total_chars}")
    print(f"  批次数:  {report.meta.batch_count}")
    print(f"  LLM:     {report.meta.llm_model} @ {report.meta.llm_base_url}")
    print(f"  人物表:   {len(report.characters.characters)} 位")
    print(f"  剧情段:   {len(report.beats.beats)} 段")
    print(f"  场景 name: {len(setting_names)} 处")
    print(f"  分集:     {len(report.screenplay.episodes)} 集")
    storyboard_total = sum(len(ep.storyboards) for ep in report.screenplay.episodes)
    print(f"  分镜:     {storyboard_total} 镜")
    print("产物文件:")
    for kind, path in result.output_paths.items():
        print(f"  - {kind:>16}  {path}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 入口:改下面这一行选要跑的 workflow
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # config_path = Path("configs/novel_analysis.json")
    config_path = Path("configs/small_llm_test.json")

    # run_novel_analysis(config_path)
    # run_character_analysis(config_path)
    run_beat_analysis(config_path)
    # run_storyboard_analysis(config_path)
