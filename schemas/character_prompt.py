"""人物图像提示词契约(``character_image_prompt`` agent 产出)。

每个提示词同时提供**英文主版本**（供 SDXL 使用）和**中文对照版本**
（供人工 review 对照），中文版本由 LLM 忠实翻译英文 tags 得到。
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class CharacterImagePrompt(BaseModel):
    """单个人物的图像生成正负提示词（中英文对照）。"""

    index: int = Field(default=0, description="对应人物的全局编号")
    name: str = Field(..., description="人物主名")
    aliases: List[str] = Field(default_factory=list, description="人物别名 / 称谓")

    positive_prompt: str = Field(
        ...,
        description="英文逗号分隔的正面提示词(主版本,供 SDXL 使用)",
    )
    positive_prompt_zh: str = Field(
        ...,
        description="中文对照版本(与 positive_prompt 一一对应,供人工 review)",
    )

    negative_prompt: str = Field(
        ...,
        description="英文逗号分隔的负面提示词(主版本,供 SDXL 使用)",
    )
    negative_prompt_zh: str = Field(
        ...,
        description="中文对照版本(与 negative_prompt 一一对应,供人工 review)",
    )


class CharacterImagePromptList(BaseModel):
    """整本书人物图像提示词列表。"""

    prompts: List[CharacterImagePrompt] = Field(default_factory=list)


__all__ = ["CharacterImagePrompt", "CharacterImagePromptList"]
