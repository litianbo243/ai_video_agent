"""extract_settings agent:单批场景增量抽取(1 次 LLM 调用)。

给定一批章节正文 + 已知场景表,LLM 返回本批的场景 delta。

公开 API::

    from agents.extract_settings import (
        extract_for_batch,
        Setting, SettingDraft, SettingCollection, SettingExtraction,
    )

    delta: SettingExtraction = extract_for_batch(batch, known_dict, llm, title="...")
"""

from agents.extract_settings.logic import SYSTEM_PROMPT, extract_for_batch
from agents.extract_settings.schema import (
    Setting,
    SettingCollection,
    SettingDraft,
    SettingExtraction,
)

__all__ = [
    "Setting",
    "SettingCollection",
    "SettingDraft",
    "SettingExtraction",
    "SYSTEM_PROMPT",
    "extract_for_batch",
]
