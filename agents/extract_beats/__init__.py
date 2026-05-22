"""extract_beats agent:单批剧情段增量抽取 + 合并。

给定批次正文 + 此前最近 N 段 + 已知人物 / 场景,LLM 返回本批的剧情段 delta
(可能新起若干段,也可能延续已有段),自动合并入 beats 列表。

LLM 配置 / 客户端是 agent 自治的:配置住在 ``llm.json``,首次调用时按需
lazy build,trace 由顶层 runner 通过 ``set_trace_dir(out_dir)`` 注入。

公开 API::

    from agents.extract_beats import (
        extract_for_batch, merge_delta, CONTEXT_WINDOW,
        Beat, BeatDraft, BeatList, BeatExtraction,
        get_llm, set_llm, set_trace_dir,
    )

    set_trace_dir(out_dir)                                 # runner 顶层调一次
    delta = extract_for_batch(
        batch, beats_so_far, char_dict, title="...",
    )
    # delta 已 in-place 合并入 beats_so_far

测试 / notebook 想 mock LLM:``set_llm(fake_client)``,完事后 ``set_llm(None)``
复位即可。
"""

from agents.extract_beats.logic import (
    CONTEXT_WINDOW,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_REWRITE_WINDOW,
    SYSTEM_PROMPT,
    extract_for_batch,
    get_llm,
    merge_delta,
    set_llm,
    set_trace_dir,
)
from agents.extract_beats.schema import (
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
    "DEFAULT_CONTEXT_WINDOW",
    "DEFAULT_REWRITE_WINDOW",
    "SYSTEM_PROMPT",
    "extract_for_batch",
    "merge_delta",
    "get_llm",
    "set_llm",
    "set_trace_dir",
]
