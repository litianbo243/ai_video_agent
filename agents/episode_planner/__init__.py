"""episode_planner agent:全书 N 段 Beat → M 集 Episode 规划(1 次 LLM 调用)。

LLM 一次性看全部 beat 列表 + 人物档案,把相邻同弧光的 beat 聚合成"集",
每集目标 ~target_duration_sec 秒。产物只有规划元数据(title / synopsis /
director_intent / beat_indices),分镜由下游 ``narrative_director.narrate_episode``
按 ``beat_indices`` 拿对应 beat + 跨 batch 原文展开。

**典型粒度**:1 个 beat ≈ 60-120 秒视频体量;目标 300s/集 → 典型 2-5 beat/集。

**为什么独立成 agent**:之前 1 beat = 1 集导致镜数严重不达标(9 镜对 300s
目标)。把"聚合"的判断从 ``beat_segmenter``(关心"剧情怎么切")抽到独立
agent(关心"几个剧情段组一集"),职责更清晰,prompt 也能更专注。

LLM 配置 / 客户端是 agent 自治的:配置住在 ``llm.json``,首次调用时按需
lazy build,trace 由顶层 runner 通过 ``set_trace_dir(out_dir)`` 注入。

公开 API::

    from agents.episode_planner import (
        plan_episodes,
        get_llm, set_llm, set_trace_dir,
    )
    from schemas import EpisodePlan, EpisodePlanList

    set_trace_dir(out_dir)                              # runner 顶层调一次
    plan_list = plan_episodes(
        beats, characters,
        target_duration_sec=300,
        title="书名",
    )
    # 下游按 plan.beat_indices 拿对应 beat 跑 narrate_episode

**schema 不在本模块 re-export**:所有 Pydantic 数据契约集中在顶层 ``schemas/``
包,业务代码请直接 ``from schemas import X``。

测试 / notebook 想 mock LLM:``set_llm(fake_client)``,完事后 ``set_llm(None)``
复位即可。
"""

from agents.episode_planner.logic import (
    DEFAULT_BEAT_DURATION_SEC,
    MIN_BEATS_PER_EP,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_TEMPLATE,
    get_llm,
    plan_episodes,
    set_llm,
    set_trace_dir,
)

__all__ = [
    "DEFAULT_BEAT_DURATION_SEC",
    "MIN_BEATS_PER_EP",
    "SYSTEM_PROMPT",
    "SYSTEM_PROMPT_TEMPLATE",
    "get_llm",
    "plan_episodes",
    "set_llm",
    "set_trace_dir",
]
