"""extract_storyboard agent:把一段 Beat 展开成一集分镜(1 次 LLM 调用)。

给定 Beat + 关联场景/人物档案 + Beat 对应的原文 batches,LLM 直接产出本集
~15-30 个分镜(``Episode``)。

注:"extract" 这个动词其实不太达意——其它三个 agent 是从原文里抽取已存在
的实体,这里更像是创作。命名按 ``agents/`` 下统一的 ``extract_*`` 前缀走。

公开 API::

    from agents.extract_storyboard import (
        storyboard_beat,
        Storyboard, Episode, ScreenplayAnalysis, StoryboardList,
    )

    ep: Episode = storyboard_beat(
        beat, char_dict, setting_dict, batch_dict, llm,
        target_duration_sec=180,
    )
"""

from agents.extract_storyboard.logic import SYSTEM_PROMPT, storyboard_beat
from agents.extract_storyboard.schema import (
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
