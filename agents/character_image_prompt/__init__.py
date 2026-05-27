"""character_image_prompt agent:人物档案 → SDXL tag-style 图像提示词。"""

from agents.character_image_prompt.logic import (
    SYSTEM_PROMPT,
    generate_for_characters,
    get_llm,
    set_llm,
    set_trace_dir,
)

__all__ = [
    "SYSTEM_PROMPT",
    "generate_for_characters",
    "get_llm",
    "set_llm",
    "set_trace_dir",
]
