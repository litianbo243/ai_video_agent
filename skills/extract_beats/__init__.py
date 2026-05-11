"""extract_beats skill:单批剧情段增量抽取(1 次 LLM 调用)。

LLM-backed skill。给定批次正文 + 此前最近 N 段 + 已知人物/场景,
LLM 返回本批的剧情段 delta(可能新起若干段,也可能延续已有段)。

公开 API::

    from skills.extract_beats import (
        extract_for_batch, CONTEXT_WINDOW,
        Beat, BeatDraft, BeatList, BeatExtraction,
    )

    delta: BeatExtraction = extract_for_batch(
        batch, beats_so_far, char_dict, setting_dict, llm, title="...",
    )
"""

from skills.extract_beats.logic import CONTEXT_WINDOW, SYSTEM_PROMPT, extract_for_batch
from skills.extract_beats.schema import (
    Beat,
    BeatDraft,
    BeatExtraction,
    BeatList,
)

__all__ = [
    "Beat",
    "BeatDraft",
    "BeatExtraction",
    "BeatList",
    "CONTEXT_WINDOW",
    "SYSTEM_PROMPT",
    "extract_for_batch",
]
