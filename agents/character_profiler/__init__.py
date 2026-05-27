"""character_profiler agent:单批人物增量抽取 + 合并。

给定一批章节正文 + 已知人物表,LLM 返回本批的人物 delta,自动合并入名册。

LLM 配置 / 客户端是 agent 自治的:配置住在 ``llm.json``,首次调用时按需
lazy build,trace 由顶层 runner 通过 ``set_trace_dir(out_dir)`` 注入。

公开 API::

    from agents.character_profiler import (
        run_for_batch, merge_delta, CharacterProfileResult,
        get_llm, set_llm, set_trace_dir,
    )
    from schemas import Character, CharacterDraft, CharacterList, CharacterExtraction

    set_trace_dir(out_dir)                                 # runner 顶层调一次
    result = run_for_batch(batch, known_dict, title="斗破苍穹")
    # known_dict 已就地更新;result.delta 给 trace / debug 看;
    # result.renames 是本批 name 升格事件,caller 应回扫已锁住的 beats:
    #   from agents.beat_segmenter import apply_character_renames
    #   apply_character_renames(beats_so_far, result.renames)

**schema 不在本模块 re-export**:所有 Pydantic 数据契约集中在顶层 ``schemas/``
包,业务代码请直接 ``from schemas import X``。

测试 / notebook 想 mock LLM:``set_llm(fake_client)``,完事后 ``set_llm(None)``
复位即可。
"""

from agents.character_profiler.logic import (
    CharacterProfileResult,
    SYSTEM_PROMPT,
    run_for_batch,
    get_llm,
    merge_delta,
    set_llm,
    set_trace_dir,
)

__all__ = [
    "CharacterProfileResult",
    "SYSTEM_PROMPT",
    "run_for_batch",
    "merge_delta",
    "get_llm",
    "set_llm",
    "set_trace_dir",
]
