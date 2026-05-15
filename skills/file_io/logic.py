"""小说分析流水线的本地文件 I/O。

负责两件事:

* 读取规范化后的源 ``.txt``;
* 把最终报告同时落成 JSON(机器可读)和 Markdown(人类可读)。
"""

from __future__ import annotations

from pathlib import Path

from agents.extract_beats.schema import Beat
from agents.extract_characters.schema import CharacterRoster
from agents.extract_storyboard.schema import Episode
from skills.file_io.schema import FinalReport


def read_text_file(path: Path) -> str:
    """读取一个 UTF-8 文本文件;文件不存在时抛异常。"""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    return p.read_text(encoding="utf-8")


def write_final_report(report: FinalReport, output_dir: Path) -> dict:
    """把最终报告同时落成 JSON 与 Markdown。

    返回写入的 7 个路径:
    ``{screenplay_json/md, characters_json/md, beats_json/md, meta_json}``。
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        "screenplay_json":  out / "screenplay.json",
        "screenplay_md":    out / "screenplay.md",
        "characters_json":  out / "characters.json",
        "characters_md":    out / "characters.md",
        "beats_json":       out / "beats.json",
        "beats_md":         out / "beats.md",
        "meta_json":        out / "meta.json",
    }

    paths["screenplay_json"].write_text(report.screenplay.model_dump_json(indent=2), encoding="utf-8")
    paths["characters_json"].write_text(report.characters.model_dump_json(indent=2), encoding="utf-8")
    paths["beats_json"].write_text(report.beats.model_dump_json(indent=2), encoding="utf-8")
    paths["meta_json"].write_text(report.meta.model_dump_json(indent=2), encoding="utf-8")

    paths["screenplay_md"].write_text(_render_screenplay_md(report), encoding="utf-8")
    paths["characters_md"].write_text(_render_characters_md(report.characters), encoding="utf-8")
    paths["beats_md"].write_text(_render_beats_md(report.beats.beats), encoding="utf-8")

    return {k: str(v) for k, v in paths.items()}


# ---------------------------------------------------------------------------
# Markdown renderers
# ---------------------------------------------------------------------------


def _render_screenplay_md(report: FinalReport) -> str:
    sp = report.screenplay
    meta = report.meta
    parts: list = []
    title = sp.title or meta.title or "(无标题)"
    parts.append(f"# 剧本分析 · {title}\n")
    if sp.logline:
        parts.append(f"> {sp.logline}\n")

    parts.append("## 元信息\n")
    total_cap = (
        f"(全书字数限 {meta.max_total_chars},超出截断)" if meta.max_total_chars else ""
    )
    parts.append(
        f"- 来源文件: `{meta.source_path}`\n"
        f"- 总字数: {meta.total_chars}{total_cap}\n"
        f"- 批次数: {meta.batch_count}(每批 ≤ {meta.max_batch_chars} 字)\n"
        f"- LLM: {meta.llm_model} @ {meta.llm_base_url}\n"
    )

    if sp.episodes:
        parts.append("## 分集与分镜\n")
        for ep in sp.episodes:
            parts.extend(_render_episode_md(ep))
    else:
        parts.append("_(暂无分集)_\n")

    return "\n".join(parts).rstrip() + "\n"


def _render_episode_md(ep: Episode) -> list:
    out: list = []
    title = ep.title or f"第 {ep.index} 集"
    out.append(f"### {title}")
    out.append("")
    total_dur = sum(s.duration_sec for s in ep.storyboards)
    if total_dur > 0:
        out.append(f"- 总时长: ≈ {total_dur:.0f} 秒 ({len(ep.storyboards)} 镜)")
    if ep.synopsis:
        out.append("")
        out.append(f"**剧情概要**: {ep.synopsis}")
    if ep.storyboards:
        out.append("")
        out.append("| # | 镜头 | 时长 | 出场 | 场景 | 画面 |")
        out.append("|---|------|------|------|------|------|")
        for sb in ep.storyboards:
            chars = ", ".join(sb.characters) if sb.characters else "-"
            setting = sb.setting or "-"
            desc = sb.description.replace("|", "\\|").replace("\n", " ")
            shot = sb.shot_type or "-"
            out.append(
                f"| {sb.index} | {shot} | {sb.duration_sec:.0f}s | {chars} | {setting} | {desc} |"
            )
        if any(sb.dialogue or sb.voiceover for sb in ep.storyboards):
            out.append("")
            out.append("**台词 / 旁白**")
            out.append("")
            for sb in ep.storyboards:
                if sb.dialogue:
                    out.append(f"- 镜{sb.index} 台词: {sb.dialogue}")
                if sb.voiceover:
                    out.append(f"- 镜{sb.index} 旁白: {sb.voiceover}")
    out.append("")
    return out


def _render_beats_md(beats: list[Beat]) -> str:
    parts: list = ["# 剧情大纲段\n"]
    if not beats:
        return parts[0] + "\n_(无剧情段)_\n"
    for b in beats:
        parts.append(f"## 第 {b.index} 段 · {b.title}")
        parts.append("")
        if b.related_batches:
            batches_str = ", ".join(f"第 {x} 批" for x in b.related_batches)
            parts.append(f"- 关联 batch: {batches_str}")
        if b.setting_refs:
            parts.append(f"- 涉及场景: {', '.join(b.setting_refs)}")
        if b.character_refs:
            parts.append(f"- 涉及人物: {', '.join(b.character_refs)}")
        if b.summary:
            parts.append("")
            parts.append(f"**剧情**: {b.summary}")
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def _render_characters_md(roster: CharacterRoster) -> str:
    parts: list = ["# 人物档案\n"]
    if not roster.characters:
        return parts[0] + "\n_(无人物条目)_\n"
    for ch in roster.characters:
        parts.append(f"## #{ch.index} · {ch.name}")
        parts.append("")
        if ch.aliases:
            parts.append(f"- 别名: {', '.join(ch.aliases)}")
        if ch.appearance:
            parts.append("")
            parts.append("**外貌**")
            parts.append("")
            parts.append(ch.appearance)
        if ch.personality:
            parts.append("")
            parts.append("**性格**")
            parts.append("")
            parts.append(ch.personality)
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"
