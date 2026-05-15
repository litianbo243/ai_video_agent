"""batch_chapters skill 的产出类型。

``Batch`` 是 ``split_into_batches`` 的产物,也是 3 个 agent
(``agents/extract_characters`` / ``extract_beats`` / ``extract_storyboard``)
共享的输入契约。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Batch(BaseModel):
    """一段文本(全局 1-based ``index``)。"""

    index: int = Field(..., description="全局批次序号(1-based)")
    text: str = Field(..., description="该批次的原文")

    @property
    def char_count(self) -> int:
        return len(self.text)

    def render_for_prompt(self) -> str:
        """LLM 直接可用的 prompt 正文。"""
        return self.text


__all__ = ["Batch"]
