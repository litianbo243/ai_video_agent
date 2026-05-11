"""单段 Beat → 单集分镜:一次 LLM 调用。

回查 Beat.related_batches 对应原文,把场景/人物/原文都拼进 prompt。
"""

from __future__ import annotations

import logging
from typing import Dict, List

from llm.client import LLMClient
from skills.batch_chapters import Batch
from skills.extract_beats.schema import Beat
from skills.extract_characters.schema import Character
from skills.extract_settings.schema import Setting
from skills.storyboard_for_beat.schema import Episode, StoryboardList

logger = logging.getLogger(__name__)


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
- 浏览原文,挑出有戏剧张力的**角色对白 / 群众议论 / 测验员或执事式播报**:
  揭示关系 / 体现冲突 / 决断 / 反转 / 当众宣布结果
- **照抄原文台词**;太长不适合视频节奏 → 浓缩或在切点拆分
- 长台词跨多镜时,**不要重复整句**,在标点处把句子拆成多段
- 普通陈述、寒暄、过渡台词不要选
- 说话者不在「角色档案」里(路人、测验员等)时,仍把其**开口说的话**放在 dialogue;
  characters 字段只填本镜画面里可辨认的 roster 人物,没有则留空列表
- 实在挑不到合适的可自由创作,但要保留人物语气

voiceover 字段(从原文挑选,但默认少用):
- **优先留空字符串 ""** —— 能用镜头 + 表演 + 环境音表现的,就不要念旁白
- 仅当原文有**第三人称叙事/作者评论/必要信息**且**画面 alone 难以传达**时,
  才从原文**浓缩为一句**念白(建议 ≤35 字),保留语气
- **禁止**放入以下内容(这些一律进 dialogue,不是旁白):
  * 任何人**说出声**的话(含群众嘲讽、窃窃私语式的直接引语)
  * 测验员 / 执事 / 系统播报式宣读(如「斗之气,X 段」)
  * 带引号的对白原文
- **禁止**与 description 同义重复:若 description 已写清动作、情绪、场面,
  voiceover **必须为空**,不要把小说叙述再念一遍
- 长旁白跨多镜时按画面切点切片,每镜最多一句

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
- 全集约 30%-50% 的镜头 voiceover 应为空;连续多镜不要镜镜念旁白
- characters 字段必须是输入"角色档案"里出现过的 name
- setting 字段必须是 Beat.setting_refs 里出现过的 name
- 一集所有镜头时长加起来要接近 {target_duration} 秒
- 严格按 JSON Schema 输出
"""


def _gather_inputs(
    beat: Beat,
    characters: Dict[str, Character],
    settings: Dict[str, Setting],
    batches: Dict[int, Batch],
) -> tuple[List[Setting], List[Character], str]:
    """按 Beat 检索关联的 Setting / Character / 原文章节。"""
    relevant_settings: List[Setting] = []
    for ref in beat.setting_refs:
        s = settings.get(ref)
        if s is None:
            logger.warning(
                "第 %d 段 Beat 引用的 Setting 不存在:%r(LLM 可能造了个 name)",
                beat.index, ref,
            )
            continue
        relevant_settings.append(s)

    relevant_chars: List[Character] = []
    for ref in beat.character_refs:
        ch = characters.get(ref)
        if ch is None:
            logger.warning(
                "第 %d 段 Beat 引用的 Character 不存在:%r",
                beat.index, ref,
            )
            continue
        relevant_chars.append(ch)

    parts: List[str] = []
    for bi in beat.related_batches:
        batch = batches.get(bi)
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


def storyboard_beat(
    beat: Beat,
    characters: Dict[str, Character],
    settings: Dict[str, Setting],
    batches: Dict[int, Batch],
    llm: LLMClient,
    *,
    target_duration_sec: int = 180,
) -> Episode:
    """把一段 Beat 展开为一集 Episode(含 storyboards)。"""
    settings_list, chars_list, raw_text = _gather_inputs(beat, characters, settings, batches)

    system = SYSTEM_PROMPT.format(target_duration=target_duration_sec)
    user = _build_user_prompt(beat, settings_list, chars_list, raw_text, target_duration_sec)

    logger.info(
        "storyboarder 第 %d 段(%s):场景=%d 人物=%d 原文=%d 字",
        beat.index, beat.title, len(settings_list), len(chars_list), len(raw_text),
    )

    sb_pack = llm.chat_json(system, user, StoryboardList)

    logger.info("第 %d 段产出:%d 镜", beat.index, len(sb_pack.storyboards))

    return Episode(
        index=beat.index,
        title=f"第 {beat.index} 集 · {beat.title}",
        synopsis=beat.summary,
        beat_index=beat.index,
        storyboards=sb_pack.storyboards,
    )


__all__ = ["storyboard_beat", "SYSTEM_PROMPT"]
