"""narrative_director agent:一集所含 N 段 Beat → 一集**叙事分镜**(1 次 LLM 调用)。

**职责切分**:本 agent 只管「讲什么」(intent / characters / setting /
speaker / dialogue / voiceover + 集层 ``director_intent``),
镜头视觉(景别 / 运镜 / 起始画面 / 时长)由下游 ``shot_director`` agent 负责。
workflow 调 ``merge_episode`` 把两份按 index 合并成完整 ``Episode``。

**输入颗粒度**:本 agent 处理的是"一集"(由 ``episode_planner`` 给出
``EpisodePlan`` + 对应的 N 段 ``Beat``),不是"一段 beat"。1 集 = N 段
beat,跨 batch 原文已自动拼接。

LLM 配置 / 客户端是 agent 自治的:配置住在 ``llm.json``,首次调用时按需
lazy build,trace 由顶层 runner 通过 ``set_trace_dir(out_dir)`` 注入。

公开 API::

    from agents.narrative_director import (
        narrate_episode,
        merge_episode,
        DEFAULT_PREV_TAIL_K,
        get_llm, set_llm, set_trace_dir,
    )
    from schemas import (
        NarrativeShot, NarrativeShotList,
        Shot, Episode, Screenplay,
    )

    set_trace_dir(out_dir)                                  # runner 顶层调一次
    narrative = narrate_episode(
        ep_index=1, plan=plan, member_beats=beats_in_ep,
        characters=char_dict, batches=batch_dict,
    )
    # 再调 shot_director.direct_episode → ShotDirectionList
    # 最后 merge_episode(ep_index=1, plan=plan, narrative=..., direction=...) → Episode

**schema 不在本模块 re-export**:所有 Pydantic 数据契约集中在顶层 ``schemas/``
包,业务代码请直接 ``from schemas import X``。

测试 / notebook 想 mock LLM:``set_llm(fake_client)``,完事后 ``set_llm(None)``
复位即可。
"""

from agents.narrative_director.logic import (
    DEFAULT_PREV_TAIL_K,
    SYSTEM_PROMPT,
    get_llm,
    merge_episode,
    narrate_episode,
    set_llm,
    set_trace_dir,
)

__all__ = [
    "DEFAULT_PREV_TAIL_K",
    "SYSTEM_PROMPT",
    "get_llm",
    "merge_episode",
    "narrate_episode",
    "set_llm",
    "set_trace_dir",
]
