"""人物档案 → SDXL tag-style 图像提示词:一次 LLM 调用。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from llm.agent_llm import make_agent_llm_manager
from schemas.character import Character, CharacterList
from schemas.character_prompt import CharacterImagePromptList

logger = logging.getLogger(__name__)


get_llm, set_llm, set_trace_dir = make_agent_llm_manager(
    agent_name="character_image_prompt",
    config_path=Path(__file__).parent / "llm.json",
)


SYSTEM_PROMPT = """\
你是 SDXL 人物图像提示词助手。任务:把中文小说人物档案转换为 SDXL tag-style 提示词,
同时输出**英文主版本**（供 SDXL 生成使用）和**中文对照版本**（供人工 review 对照）。

输出必须是 JSON,符合 CharacterImagePromptList schema。

# 生成目标

- 对输入中的每个人物输出一条 CharacterImagePrompt,不得增删人物,保持 index / name / aliases。
- 每个人物必须同时提供 4 个字段:
  - positive_prompt: 英文逗号分隔 tags（主版本,供 SDXL 使用）
  - positive_prompt_zh: 与 positive_prompt 一一对应的中文翻译（对照版本）
  - negative_prompt: 英文逗号分隔 tags（主版本）
  - negative_prompt_zh: 与 negative_prompt 一一对应的中文翻译（对照版本）
- positive_prompt 聚焦跨场景稳定识别:性别、年龄段、脸部特征、体型、气质、职业 / 服装风格、
  镜头风格、画质风格。
- background / personality / arc 只能转成安全的气质、职业、姿态、表情和氛围标签。
- 不要把剧情事件、暴力、胁迫、性行为、裸露或露骨身体描写写入 positive_prompt。
- 所有人物都按成人角色处理;如果档案年龄不明确,只写 adult。

# SDXL tag 风格

positive_prompt 建议结构（英文）:
masterpiece, best quality, realistic, cinematic portrait, 1girl / 1boy / 1woman / 1man,
adult, [age], [face/body/hair/eyes], [expression/temperament], [profession/style outfit],
[lighting], [composition]

negative_prompt 至少覆盖（英文）:
low quality, worst quality, blurry, bad anatomy, bad hands, extra fingers, missing fingers,
deformed, disfigured, duplicate, watermark, text, logo, jpeg artifacts, child, teen, loli,
shota, nude, nsfw, explicit, sexual act, violence

# 中文对照版本要求

- positive_prompt_zh / negative_prompt_zh 必须与英文版本**严格一一对应**（相同数量、相同顺序）。
- 每个英文 tag 翻译成对应的中文短语，保留专业术语（如 masterpiece → 杰作, loli → 萝莉）。
- 示例:
  English: masterpiece, best quality, 1girl, long hair, red dress
  Chinese: 杰作, 最佳质量, 1女孩, 长发, 红色连衣裙
- 中文版本仅用于人工 review，不送入 SDXL。

# 约束

- 每个 prompt 建议 35-70 个 tags,避免过长。
- 保留东方 / 中文语境角色的视觉身份时,用 Chinese woman / Chinese man 等英文 tag。
- 如果原档案包含敏感或露骨内容,只抽取安全、非露骨的外貌与气质信息。
"""


def _render_character(ch: Character) -> str:
    aliases = ", ".join(ch.aliases) if ch.aliases else "-"
    return (
        f"## #{ch.index} {ch.name}\n"
        f"aliases: {aliases}\n"
        f"appearance: {ch.appearance or '-'}\n"
        f"background: {ch.background or '-'}\n"
        f"personality: {ch.personality or '-'}\n"
        f"arc: {ch.arc or '-'}"
    )


def _build_user_prompt(characters: Iterable[Character]) -> str:
    rendered = "\n\n".join(_render_character(ch) for ch in characters)
    return (
        "=== 人物档案 ===\n"
        f"{rendered or '(无人物)'}\n\n"
        "=== 任务 ===\n"
        "请为每个人物生成 SDXL tag-style positive_prompt 与 negative_prompt。"
    )


def generate_for_characters(characters: CharacterList) -> CharacterImagePromptList:
    """为 ``CharacterList`` 中每个人物生成 SDXL 正负提示词。"""

    llm = get_llm()
    logger.info(
        "character_prompt_generator 生成 %d 位人物提示词,%s @ %s",
        len(characters.characters), llm.model, llm.base_url,
    )
    result = llm.chat_json(
        SYSTEM_PROMPT,
        _build_user_prompt(characters.characters),
        CharacterImagePromptList,
    )
    logger.info(
        "character_prompt_generator 完成:%d 条提示词",
        len(result.prompts),
    )
    return result


__all__ = [
    "SYSTEM_PROMPT",
    "generate_for_characters",
    "get_llm",
    "set_llm",
    "set_trace_dir",
]
