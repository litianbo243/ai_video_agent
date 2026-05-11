"""跑 workflow:character / setting / beat / storyboard / novel_analysis 五选一。

在 ``__main__`` 里改最后一行选要跑哪个 workflow。
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
    setting_analysis,
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
    return out_dir


# ---------------------------------------------------------------------------
# 5 个 workflow runner
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


def run_setting_analysis(config_path: Path) -> None:
    """跑 setting_analysis 子-workflow:全本场景档案抽取。"""
    config = load_config(config_path)
    _setup_logging()
    out_dir = _ts_outdir(config)

    ing, collection = setting_analysis.run(config)

    out_path = out_dir / "settings.json"
    out_path.write_text(
        collection.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print()
    print("=" * 60)
    print(f"完成。输出目录:{out_dir}")
    print(f"  书名:    {ing.title or '(无)'}")
    print(f"  字数:    {ing.total_chars}(原文 {ing.raw_chars})")
    print(f"  批次数:  {len(ing.batches)}")
    print(f"  LLM:     {config.llm.model} @ {config.llm.base_url}")
    print(f"  场景档案: {len(collection.settings)} 个")
    print(f"产物文件: {out_path}")
    print("=" * 60)


def run_beat_analysis(config_path: Path) -> None:
    """跑 beat_analysis workflow:内部自带 ingest + character + setting + beat 一条龙。"""
    config = load_config(config_path)
    _setup_logging()
    out_dir = _ts_outdir(config)

    ing, roster, coll, beats = beat_analysis.run(config)

    out_paths = {
        "characters": out_dir / "characters.json",
        "settings":   out_dir / "settings.json",
        "beats":      out_dir / "beats.json",
    }
    out_paths["characters"].write_text(
        roster.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8"
    )
    out_paths["settings"].write_text(
        coll.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8"
    )
    out_paths["beats"].write_text(
        beats.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print()
    print("=" * 60)
    print(f"完成。输出目录:{out_dir}")
    print(f"  书名:    {ing.title or '(无)'}")
    print(f"  字数:    {ing.total_chars}(原文 {ing.raw_chars})")
    print(f"  批次数:  {len(ing.batches)}")
    print(f"  LLM:     {config.llm.model} @ {config.llm.base_url}")
    print(f"  人物表:   {len(roster.characters)} 位")
    print(f"  场景档案: {len(coll.settings)} 个")
    print(f"  剧情段:   {len(beats.beats)} 段")
    print("产物文件:")
    for kind, path in out_paths.items():
        print(f"  - {kind:>10}  {path}")
    print("=" * 60)


def run_storyboard_analysis(config_path: Path) -> None:
    """跑 storyboard_analysis workflow:内部自带 ingest + character + setting + beat + storyboard 一条龙。"""
    config = load_config(config_path)
    _setup_logging()
    out_dir = _ts_outdir(config)

    ing, roster, coll, beats, screenplay = storyboard_analysis.run(config)

    out_paths = {
        "characters": out_dir / "characters.json",
        "settings":   out_dir / "settings.json",
        "beats":      out_dir / "beats.json",
        "screenplay": out_dir / "screenplay.json",
    }
    out_paths["characters"].write_text(
        roster.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8"
    )
    out_paths["settings"].write_text(
        coll.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8"
    )
    out_paths["beats"].write_text(
        beats.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8"
    )
    out_paths["screenplay"].write_text(
        screenplay.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8"
    )

    storyboard_total = sum(len(ep.storyboards) for ep in screenplay.episodes)
    print()
    print("=" * 60)
    print(f"完成。输出目录:{out_dir}")
    print(f"  书名:    {ing.title or '(无)'}")
    print(f"  字数:    {ing.total_chars}(原文 {ing.raw_chars})")
    print(f"  批次数:  {len(ing.batches)}")
    print(f"  LLM:     {config.llm.model} @ {config.llm.base_url}")
    print(f"  人物表:   {len(roster.characters)} 位")
    print(f"  场景档案: {len(coll.settings)} 个")
    print(f"  剧情段:   {len(beats.beats)} 段")
    print(f"  分集:     {len(screenplay.episodes)} 集")
    print(f"  分镜:     {storyboard_total} 镜")
    print("产物文件:")
    for kind, path in out_paths.items():
        print(f"  - {kind:>10}  {path}")
    print("=" * 60)


def run_novel_analysis(config_path: Path) -> None:
    """跑 novel_analysis 父-workflow:人物 + 场景 + 剧情段 + 分集分镜。"""
    config = load_config(config_path)
    _setup_logging()

    result = novel_analysis.run(config)

    print()
    print("=" * 60)
    print(f"完成。输出目录:{result.output_dir}")
    report = result.report
    print(f"  书名:   {report.meta.title or '(无)'}")
    print(f"  字数:   {report.meta.total_chars}")
    print(f"  批次数: {report.meta.batch_count}")
    print(f"  LLM:    {report.meta.llm_model} @ {report.meta.llm_base_url}")
    print(f"  人物表: {len(report.characters.characters)} 位")
    print(f"  场景档案: {len(report.settings.settings)} 个")
    print(f"  剧情段: {len(report.beats.beats)} 段")
    print(f"  分集:   {len(report.screenplay.episodes)} 集")
    storyboard_total = sum(len(ep.storyboards) for ep in report.screenplay.episodes)
    print(f"  分镜:   {storyboard_total} 镜")
    print("产物文件:")
    for kind, path in result.output_paths.items():
        print(f"  - {kind:>16}  {path}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 入口:改下面这一行选要跑的 workflow
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    config_path = Path("configs/novel_analysis.json")

    # run_novel_analysis(config_path)
    # run_character_analysis(config_path)
    # run_setting_analysis(config_path)
    # run_beat_analysis(config_path)
    run_storyboard_analysis(config_path)
