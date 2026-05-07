"""子-agent:把单段剧情大纲(Beat)展开成单集分集分镜(Episode)。

每段 Beat 一次 LLM 调用。input 包括:
- Beat 本身(title / summary / setting_refs / character_refs)
- 关联场景的视觉档案(Setting.description)
- 关联人物的档案(Character.appearance / personality)
- ★ 关联 batch 的原文(遍历 ``Beat.related_batches`` 从 BatchState.batches 反查并拼接)
  —— 台词 / 旁白 / 视觉细节都从原文里直接挑选改编

output: 单集 ``Episode``(含 storyboards)。
"""

from __future__ import annotations

import logging
from typing import Dict, List

from pydantic import BaseModel, Field

from llm.client import LLMClient
from schema.novel_analysis import (
    Batch,
    Beat,
    BatchState,
    Character,
    Episode,
    Setting,
    Storyboard,
)

logger = logging.getLogger(__name__)


class _StoryboardList(BaseModel):
    """LLM JSON 输出包装(``chat_json`` 要求 schema 是 ``BaseModel`` 子类)。"""

    storyboards: List[Storyboard] = Field(default_factory=list)


SYSTEM_PROMPT = """\
你是专业的中文小说视频改编编剧,正在为小说的某一段剧情(一集)改编分镜。

输入包括:
- 一段剧情大纲(Beat:title + summary + 关联场景/人物)
- 关联场景的视觉档案(给画面用)
- 关联人物的档案(含外貌 + 性格)
- ★ 这段剧情对应的**原文**(整批章节,几千字)
  —— 台词 / 旁白 / 视觉描写**都直接从原文里挑选改编**,不要凭空创作

────────────────────────────────────────────────
你的任务: 编排本集的分镜清单(storyboards)。

每集 ~15-30 个镜头,每镜 3-15 秒,平均 6-8 秒;一集所有镜头时长之和 ≈ {target_duration} 秒。

────────────────────────────────────────────────
【改编规则】

镜头粒度:
- 优先选择**有戏剧张力的画面**(冲突 / 转折 / 决断 / 情绪爆发)
- 平稳过渡可以一镜带过
- 原文里不重要的支线 / 水分内容**直接跳过**(不必逐字改编)

dialogue 字段(从原文里挑选):
- 浏览原文,挑出有戏剧张力的角色对白:揭示关系 / 体现冲突 / 决断 / 反转
- **照抄原文台词**;太长不适合视频节奏 → 浓缩或在切点拆分
- 长台词跨多镜时,**不要重复整句**,在标点处把句子拆成多段
- 普通陈述、寒暄、过渡台词不要选
- 实在挑不到合适的可自由创作,但要保留人物语气

voiceover 字段(从原文挑选):
- 旁白来自小说**叙述性原文**(描写 / 评论 / 心理),非对白
- 把原文的关键叙述**提炼成 1-2 句念白,保留原作语气**
- 长旁白跨多镜时按画面切点切片

description 字段(给图像生成模型):
- 从原文的视觉描写里**提取关键意象**:谁在做什么 + 光线 + 构图
- 出场人物用**正式 name**(从角色档案里取,不要别名)
- 不要把 dialogue 文字塞进 description

setting 字段:
- **必须引用 Beat.setting_refs 里出现过的 Setting.name**(单一地点)
- 不要自由发挥写新地点

duration_sec 字段:
- = max(画面节奏需求, 念完 dialogue + voiceover 所需时间)
- 估算口径:对白 ~3.5 字/秒,旁白 ~3 字/秒
- 整集所有镜头时长之和 ≈ {target_duration} 秒

shot_type 字段:6 选 1 — 远景 / 全景 / 中景 / 近景 / 特写 / 大特写

────────────────────────────────────────────────
【绝对约束】
- 不杜撰人物或事件,所有内容必须有原文支撑
- dialogue / voiceover 必须来自原文(直接搬或浓缩),不要凭空创作
- characters 字段必须是输入"角色档案"里出现过的 name
- setting 字段必须是 Beat.setting_refs 里出现过的 name
- 一集所有镜头时长加起来要接近 {target_duration} 秒
- 严格按 JSON Schema 输出
"""


def _gather_beat_inputs(
    beat: Beat,
    state: BatchState,
) -> tuple[List[Setting], List[Character], str]:
    """按 Beat 检索关联的 Setting / Character / 原文章节。"""
    # 1. Settings(按 setting_refs 匹配)
    relevant_settings: List[Setting] = []
    for ref in beat.setting_refs:
        s = state.settings.get(ref)
        if s is None:
            logger.warning(
                "第 %d 段 Beat 引用的 Setting 不存在:%r(LLM 可能造了个 name)",
                beat.index, ref,
            )
            continue
        relevant_settings.append(s)

    # 2. Characters(按 character_refs 匹配)
    relevant_chars: List[Character] = []
    for ref in beat.character_refs:
        ch = state.characters.get(ref)
        if ch is None:
            logger.warning(
                "第 %d 段 Beat 引用的 Character 不存在:%r",
                beat.index, ref,
            )
            continue
        relevant_chars.append(ch)

    # 3. 原文:按 related_batches 顺序反查 BatchState.batches,拼接所有相关 batch 原文
    parts: List[str] = []
    batch_lookup: dict[int, Batch] = {b.index: b for b in state.batches}
    for bi in beat.related_batches:
        batch = batch_lookup.get(bi)
        if batch is None:
            logger.warning("第 %d 段 Beat 关联的 batch %d 未找到原文", beat.index, bi)
            continue
        parts.append(batch.render_for_prompt())
    raw_text = "\n\n".join(parts)

    return relevant_settings, relevant_chars, raw_text


def _render_settings_for_prompt(settings: List[Setting]) -> str:
    if not settings:
        return "(无)"
    parts: List[str] = []
    for s in settings:
        parts.append(f"### {s.name}\n{s.description}" if s.description else f"### {s.name}")
    return "\n\n".join(parts)


def _render_characters_for_prompt(chars: List[Character]) -> str:
    if not chars:
        return "(无)"
    parts: List[str] = []
    for ch in chars:
        section = [f"### {ch.name}"]
        if ch.aliases:
            section.append(f"别名: {', '.join(ch.aliases)}")
        if ch.appearance:
            section.append(f"外貌: {ch.appearance}")
        if ch.personality:
            section.append(f"性格: {ch.personality}")
        parts.append("\n".join(section))
    return "\n\n".join(parts)


def _render_beat_for_prompt(beat: Beat) -> str:
    return "\n".join([
        f"index: {beat.index}",
        f"title: {beat.title}",
        f"summary: {beat.summary}",
        f"setting_refs: {beat.setting_refs}",
        f"character_refs: {beat.character_refs}",
    ])


def _build_user_prompt(
    beat: Beat,
    settings: List[Setting],
    chars: List[Character],
    raw_text: str,
    target_duration: int,
) -> str:
    return (
        f"=== 本段剧情大纲(Beat)===\n{_render_beat_for_prompt(beat)}\n\n"
        f"=== 关联场景档案 ===\n{_render_settings_for_prompt(settings)}\n\n"
        f"=== 关联人物档案 ===\n{_render_characters_for_prompt(chars)}\n\n"
        f"=== 本段对应的原文(batch {beat.related_batches})===\n{raw_text or '(未找到原文)'}\n\n"
        f"=== 任务 ===\n"
        f"按 JSON Schema 输出本集所有 storyboards。目标总时长 ≈ {target_duration} 秒。\n"
        f"输出格式:`{{\"storyboards\": [...]}}`"
    )


def storyboard_beat(beat: Beat, state: BatchState, llm: LLMClient) -> Episode:
    """把一段 Beat 展开为一集 Episode(含 storyboards)。"""
    settings, chars, raw_text = _gather_beat_inputs(beat, state)

    system = SYSTEM_PROMPT.format(target_duration=state.target_episode_duration_sec)
    user = _build_user_prompt(
        beat, settings, chars, raw_text, state.target_episode_duration_sec,
    )

    logger.info(
        "storyboarder 第 %d 段(%s):场景=%d 人物=%d 原文=%d 字",
        beat.index, beat.title, len(settings), len(chars), len(raw_text),
    )

    sb_pack = llm.chat_json(system, user, _StoryboardList)

    logger.info("第 %d 段产出:%d 镜", beat.index, len(sb_pack.storyboards))

    return Episode(
        index=beat.index,
        title=f"第 {beat.index} 集 · {beat.title}",
        synopsis=beat.summary,
        beat_index=beat.index,
        storyboards=sb_pack.storyboards,
    )


__all__ = ["storyboard_beat", "SYSTEM_PROMPT"]
