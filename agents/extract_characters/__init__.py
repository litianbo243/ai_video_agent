"""extract_characters agent:单批人物增量抽取 + 合并。

给定一批章节正文 + 已知人物表,LLM 返回本批的人物 delta,自动合并入名册。

LLM 配置 / 客户端是 agent 自治的:配置住在 ``llm.json``,首次调用时按需
lazy build,trace 由顶层 runner 通过 ``set_trace_dir(out_dir)`` 注入。

公开 API::

    from agents.extract_characters import (
        extract_for_batch, merge_delta,
        Character, CharacterDraft, CharacterRoster, CharacterExtraction,
        get_llm, set_llm, set_trace_dir,
    )

    set_trace_dir(out_dir)                                 # runner 顶层调一次
    delta = extract_for_batch(batch, known_dict, title="斗破苍穹")
    # delta 已 in-place 合并入 known_dict;通常无需读 delta

测试 / notebook 想 mock LLM:``set_llm(fake_client)``,完事后 ``set_llm(None)``
复位即可。
"""

from agents.extract_characters.logic import (
    SYSTEM_PROMPT,
    extract_for_batch,
    get_llm,
    merge_delta,
    set_llm,
    set_trace_dir,
)
from agents.extract_characters.schema import (
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
    "merge_delta",
    "get_llm",
    "set_llm",
    "set_trace_dir",
]
