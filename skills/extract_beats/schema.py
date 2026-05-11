"""extract_beats skill 的全部数据契约。

层次:

* ``BeatDraft``      —— LLM 直接产出的初稿(无 ``index`` / ``related_batches``)
* ``Beat``           —— 合并后完整的剧情段(含 ``index`` / ``related_batches``)
* ``BeatList``       —— 一次 run 的全部剧情段
* ``BeatExtraction`` —— 单次 LLM 调用的输出包装(新段 + 延续段)

数据语义跟"这次 LLM 调用想抽什么"绑死,所以全部住在 skill 内部。
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class BeatDraft(BaseModel):
    """LLM 抽取剧情段的输出契约(无 ``index`` / ``related_batches``,这两个 merge 自动填)。"""

    title: str = Field(..., description='剧情段标题,如"撕婚约"、"测验失败"')
    summary: str = Field(
        default="",
        description="本段剧情概要(2-4 句),要交代起承转合 + 关键节奏点(小高潮)",
    )
    setting_refs: List[str] = Field(
        default_factory=list,
        description="本段涉及的场景(Setting.name 列表),按时序排列,可多个",
    )
    character_refs: List[str] = Field(
        default_factory=list,
        description="本段涉及的人物(Character.name 列表)",
    )


class Beat(BeatDraft):
    """完整的剧情大纲段(state 内部 + 落盘格式)。

    比 ``BeatDraft`` 多两个 merge 时自动填的字段:
    * ``index``:全局编号(稳定主键)
    * ``related_batches``:涉及的 batch(支持跨批延续)

    一个 Beat 可以涉及多个 Setting / Character,**也可能跨多个 batch**(冲突跨批
    时由后续 batch 的 extractor 把当前 batch 追加到 ``related_batches``)。
    storyboard_analysis 会遍历 ``related_batches`` 把所有相关原文拼起来。
    """

    index: int = Field(default=0, description="全局编号(1-based,merge 时自动赋值)")
    related_batches: List[int] = Field(
        default_factory=list,
        description=(
            "该段涉及的 batch 编号列表(1-based,按时序)。"
            "merge 时自动追加当前 batch;通常为 1-3 个 batch。"
        ),
    )


class BeatList(BaseModel):
    beats: List[Beat] = Field(default_factory=list)


class BeatExtraction(BaseModel):
    """LLM 单批输出:本批的剧情段增量。

    两条通道:
    * ``new_beats``:本批中**新起**的剧情段(典型情况)
    * ``extended_beat_indices``:本批是**已有段的延续**(跨批高潮 / 长冲突),
      给出对应 Beat.index;merge 时自动把本批 batch 追加到它们的
      ``related_batches``,不新建 Beat。
    """

    new_beats: List[BeatDraft] = Field(default_factory=list)
    extended_beat_indices: List[int] = Field(
        default_factory=list,
        description="本批延续了哪几个已有段(给已有段的 index)",
    )


__all__ = [
    "Beat",
    "BeatDraft",
    "BeatExtraction",
    "BeatList",
]
