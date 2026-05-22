"""extract_storyboard agent:把一段 Beat 展开成一集分镜(1 次 LLM 调用)。

给定 Beat + 关联场景/人物档案 + Beat 对应的原文 batches,LLM 直接产出本集
~15-30 个分镜(``Episode``)。

注:"extract" 这个动词其实不太达意——其它三个 agent 是从原文里抽取已存在
的实体,这里更像是创作。命名按 ``agents/`` 下统一的 ``extract_*`` 前缀走。

LLM 配置 / 客户端是 agent 自治的:配置住在 ``llm.json``,首次调用时按需
lazy build,trace 由顶层 runner 通过 ``set_trace_dir(out_dir)`` 注入。

公开 API::

    from agents.extract_storyboard import (
        storyboard_beat,
        Storyboard, Episode, ScreenplayAnalysis, StoryboardList,
        get_llm, set_llm, set_trace_dir,
    )

    set_trace_dir(out_dir)                                 # runner 顶层调一次
    ep: Episode = storyboard_beat(
        beat, char_dict, batch_dict, target_duration_sec=180,
    )

测试 / notebook 想 mock LLM:``set_llm(fake_client)``,完事后 ``set_llm(None)``
复位即可。
"""

from agents.extract_storyboard.logic import (
    DEFAULT_PREV_TAIL_K,
    SYSTEM_PROMPT,
    get_llm,
    set_llm,
    set_trace_dir,
    storyboard_beat,
)
from agents.extract_storyboard.schema import (
    Episode,
    ScreenplayAnalysis,
    Storyboard,
    StoryboardList,
)

__all__ = [
    "DEFAULT_PREV_TAIL_K",
    "Episode",
    "ScreenplayAnalysis",
    "Storyboard",
    "StoryboardList",
    "SYSTEM_PROMPT",
    "storyboard_beat",
    "get_llm",
    "set_llm",
    "set_trace_dir",
]
