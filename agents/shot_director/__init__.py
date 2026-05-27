"""shot_director agent:一集叙事分镜 → 一集视觉指导(1 次 LLM 调用)。

输入是 ``narrative_director`` 出的一集 ``NarrativeShot`` 列表 + 集层
``director_intent``(叙事调性);本 agent 决定每镜的"怎么拍"(景别 /
运镜 / 动作演变 / 起始画面 / 时长),并给出本集视觉调性 ``visual_style``。

输出 ``ShotDirectionList``,workflow 按 ``index`` 与 ``NarrativeShot`` 一一
配对合并成最终 ``Shot``。

LLM 配置 / 客户端是 agent 自治的:配置住在 ``llm.json``,首次调用 lazy build,
trace 由顶层 runner 通过 ``set_trace_dir(out_dir)`` 注入。

公开 API::

    from agents.shot_director import (
        direct_episode,
        get_llm, set_llm, set_trace_dir,
    )
    from schemas import ShotDirection, ShotDirectionList, ShotType

    set_trace_dir(out_dir)                                 # runner 顶层调一次
    direction = direct_episode(
        episode_index=ep_idx,
        director_intent=narrative.director_intent,
        narrative_shots=narrative.shots,
        characters=char_dict,
        prev_tail_directions=prev_tail,
        target_duration_sec=180,
    )

**schema 不在本模块 re-export**:所有 Pydantic 数据契约集中在顶层 ``schemas/``
包,业务代码请直接 ``from schemas import X``。

测试 / notebook 想 mock LLM:``set_llm(fake_client)``,完事后 ``set_llm(None)``
复位即可。
"""

from agents.shot_director.logic import (
    SYSTEM_PROMPT,
    direct_episode,
    get_llm,
    set_llm,
    set_trace_dir,
)

__all__ = [
    "SYSTEM_PROMPT",
    "direct_episode",
    "get_llm",
    "set_llm",
    "set_trace_dir",
]
