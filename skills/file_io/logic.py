"""小说分析流水线的本地文件 I/O。

负责两件事:

* 读取规范化后的源 ``.txt``;
* 把流水线产物按需落成 JSON(机器可读)+ Markdown(人类可读)。

``write_partial`` 是统一写盘入口:传哪些字段就落哪些文件,跨 mode 通用。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from schemas.beat import Beat, BeatList
from schemas.character import CharacterList
from schemas.episode_plan import EpisodePlanList
from schemas.report import FinalReport, ReportMeta
from schemas.screenplay import Episode, Screenplay


def read_text_file(path: Path) -> str:
    """读取一个 UTF-8 文本文件;文件不存在时抛异常。"""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    return p.read_text(encoding="utf-8")


def write_partial(
    output_dir: Path,
    *,
    characters: Optional[CharacterList] = None,
    beats: Optional[BeatList] = None,
    episode_plans: Optional[EpisodePlanList] = None,
    screenplay: Optional[Screenplay] = None,
    meta: Optional[ReportMeta] = None,
) -> Dict[str, str]:
    """把工作流产物落到 ``output_dir``,**有什么落什么**。

    每个字段非 ``None`` 时就落对应的 ``.json``(可渲染时同时落 ``.md``):

    * ``characters``    → characters.json + characters.md
    * ``beats``         → beats.json + beats.md
    * ``episode_plans`` → episode_plans.json(无 .md)
    * ``screenplay``    → screenplay.json(+ screenplay.md,需配合 ``meta``)
    * ``meta``          → meta.json

    返回 ``{kind: path_str}`` 映射。
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    if characters is not None:
        paths["characters_json"] = out / "characters.json"
        paths["characters_json"].write_text(
            characters.model_dump_json(indent=2), encoding="utf-8",
        )
        paths["characters_md"] = out / "characters.md"
        paths["characters_md"].write_text(
            _render_characters_md(characters), encoding="utf-8",
        )

    if beats is not None:
        paths["beats_json"] = out / "beats.json"
        paths["beats_json"].write_text(
            beats.model_dump_json(indent=2), encoding="utf-8",
        )
        paths["beats_md"] = out / "beats.md"
        paths["beats_md"].write_text(
            _render_beats_md(beats.beats), encoding="utf-8",
        )

    if episode_plans is not None:
        paths["episode_plans_json"] = out / "episode_plans.json"
        paths["episode_plans_json"].write_text(
            episode_plans.model_dump_json(indent=2), encoding="utf-8",
        )

    if screenplay is not None:
        paths["screenplay_json"] = out / "screenplay.json"
        paths["screenplay_json"].write_text(
            screenplay.model_dump_json(indent=2), encoding="utf-8",
        )
        # screenplay.md 渲染需要 meta 配合(显示元信息);只在两者都有时才写
        if meta is not None:
            report = FinalReport(
                screenplay=screenplay,
                characters=characters or CharacterList(),
                beats=beats or BeatList(),
                meta=meta,
            )
            paths["screenplay_md"] = out / "screenplay.md"
            paths["screenplay_md"].write_text(
                _render_screenplay_md(report), encoding="utf-8",
            )

    if meta is not None:
        paths["meta_json"] = out / "meta.json"
        paths["meta_json"].write_text(
            meta.model_dump_json(indent=2), encoding="utf-8",
        )

    return {k: str(v) for k, v in paths.items()}


# ---------------------------------------------------------------------------
# Markdown renderers
# ---------------------------------------------------------------------------


def _render_screenplay_md(report: FinalReport) -> str:
    sp = report.screenplay
    meta = report.meta
    parts: list = []
    title = meta.title or "(无标题)"
    parts.append(f"# 剧本分析 · {title}\n")

    parts.append("## 元信息\n")
    total_cap = (
        f"(全书字数限 {meta.max_total_chars},超出截断)" if meta.max_total_chars else ""
    )
    base_lines = (
        f"- 来源文件: `{meta.source_path}`\n"
        f"- 总字数: {meta.total_chars}{total_cap}\n"
        f"- 批次数: {meta.batch_count}(每批 ≤ {meta.max_batch_chars} 字)\n"
    )
    if meta.llm_per_agent:
        llm_lines = ["- LLM(各 agent 自选):"]
        for agent_name, info in meta.llm_per_agent.items():
            llm_lines.append(f"  - `{agent_name}`: {info.model} @ {info.base_url}")
        parts.append(base_lines + "\n".join(llm_lines) + "\n")
    else:
        parts.append(base_lines)

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
    total_dur = sum(s.duration_sec for s in ep.shots)
    if total_dur > 0:
        out.append(f"- 总时长: ≈ {total_dur:.0f} 秒 ({len(ep.shots)} 镜)")
    if ep.synopsis:
        out.append("")
        out.append(f"**剧情概要**: {ep.synopsis}")
    if ep.director_intent:
        out.append("")
        out.append(f"**叙事调性**: {ep.director_intent}")
    if ep.visual_style:
        out.append("")
        out.append(f"**视觉调性**: {ep.visual_style}")
    if ep.shots:
        out.append("")
        out.append("| # | 镜头 | 意图 | 剧情 | 时长 | 出场 | 场景 | 画面 |")
        out.append("|---|------|------|------|------|------|------|------|")
        for sb in ep.shots:
            chars = ", ".join(sb.characters) if sb.characters else "-"
            setting = sb.setting or "-"
            desc = sb.description.replace("|", "\\|").replace("\n", " ")
            shot = sb.shot_type or "-"
            intent = (sb.intent or "-").replace("|", "\\|").replace("\n", " ")
            narr = (sb.narrative_description or "-").replace("|", "\\|").replace("\n", " ")
            out.append(
                f"| {sb.index} | {shot} | {intent} | {narr} | {sb.duration_sec:.0f}s | {chars} | {setting} | {desc} |"
            )
        if any(sb.shot_action or sb.camera_motion for sb in ep.shots):
            out.append("")
            out.append("**动作 / 运镜**")
            out.append("")
            for sb in ep.shots:
                if sb.shot_action:
                    out.append(f"- 镜{sb.index} 动作: {sb.shot_action}")
                if sb.camera_motion:
                    out.append(f"- 镜{sb.index} 运镜: {sb.camera_motion}")
        if any(sb.dialogue or sb.voiceover for sb in ep.shots):
            out.append("")
            out.append("**台词 / 旁白**")
            out.append("")
            for sb in ep.shots:
                speaker = sb.speaker or "?"
                if sb.dialogue:
                    out.append(f"- 镜{sb.index} 台词 [{speaker}]: {sb.dialogue}")
                if sb.voiceover:
                    out.append(f"- 镜{sb.index} 内心 [{speaker}]: {sb.voiceover}")
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


def _render_characters_md(roster: CharacterList) -> str:
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
        if ch.background:
            parts.append("")
            parts.append("**背景**")
            parts.append("")
            parts.append(ch.background)
        if ch.personality:
            parts.append("")
            parts.append("**性格(默认态)**")
            parts.append("")
            parts.append(ch.personality)
        if ch.arc:
            parts.append("")
            parts.append("**弧光(剧情演变,带事件锚点)**")
            parts.append("")
            parts.append(ch.arc)
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"
