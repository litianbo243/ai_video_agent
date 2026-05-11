"""storyboard_for_beat skill:把一段 Beat 展开成一集分镜(1 次 LLM 调用)。

LLM-backed skill。给定 Beat + 关联场景/人物档案 + Beat 对应的原文 batches,
LLM 直接产出本集 ~15-30 个分镜(``Episode``)。

公开 API::

    from skills.storyboard_for_beat import (
        storyboard_beat,
        Storyboard, Episode, ScreenplayAnalysis, StoryboardList,
    )

    ep: Episode = storyboard_beat(
        beat, char_dict, setting_dict, batch_dict, llm,
        target_duration_sec=180,
    )
"""

from skills.storyboard_for_beat.logic import SYSTEM_PROMPT, storyboard_beat
from skills.storyboard_for_beat.schema import (
    Episode,
    ScreenplayAnalysis,
    Storyboard,
    StoryboardList,
)

__all__ = [
    "Episode",
    "ScreenplayAnalysis",
    "Storyboard",
    "StoryboardList",
    "SYSTEM_PROMPT",
    "storyboard_beat",
]
