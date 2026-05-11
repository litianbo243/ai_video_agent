"""extract_characters skill:单批人物增量抽取(1 次 LLM 调用)。

LLM-backed skill。给定一批章节正文 + 已知人物表,LLM 返回本批的人物 delta。
合并/累积逻辑由调用方(workflow)负责。

公开 API::

    from skills.extract_characters import (
        extract_for_batch,
        Character, CharacterDraft, CharacterRoster, CharacterExtraction,
    )

    delta: CharacterExtraction = extract_for_batch(
        batch, known_dict, llm, title="斗破苍穹",
    )

数据契约 (``Character`` / ``CharacterRoster`` / ``CharacterDraft`` /
``CharacterExtraction``) 全部住在本 skill 内,不放进顶层 ``schema/``。
"""

from skills.extract_characters.logic import SYSTEM_PROMPT, extract_for_batch
from skills.extract_characters.schema import (
    Character,
    CharacterDraft,
    CharacterExtraction,
    CharacterRoster,
)

__all__ = [
    "Character",
    "CharacterDraft",
    "CharacterExtraction",
    "CharacterRoster",
    "SYSTEM_PROMPT",
    "extract_for_batch",
]
