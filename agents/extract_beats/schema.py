"""extract_beats agent 的数据契约。

* ``BeatDraft``      —— LLM 产出的初稿(无 ``index`` / ``related_batches``)
* ``Beat``           —— 合并后完整段(含 ``index`` / ``related_batches``)
* ``BeatList``       —— 一次 run 的全部剧情段
* ``BeatExtraction`` —— 单次 LLM 调用的输出包装

**写 schema 注意**:class docstring 和 field description 都会经
``model_json_schema()`` 塞进 LLM 的 system prompt,所以只写"是什么 + 怎么填 +
微例",**不放** ``logic.py:xxx`` / "workflow 内部" 这类 LLM 看不懂的内部引用,
也**不复述**决策规则(权威源在 ``logic.py:SYSTEM_PROMPT_TEMPLATE``)。维护指引
只放本模块 docstring —— Pydantic 不抓模块级 docstring。
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class BeatDraft(BaseModel):
    """剧情大纲段(一段 ≈ 一集短视频)。"""

    title: str = Field(
        ...,
        description="2-6 字事件标题,不含地点。例:「会议决裂」",
    )
    summary: str = Field(
        default="",
        description="1-2 句:谁在哪做了什么 + 剧情怎么推进的",
    )
    setting_refs: List[str] = Field(
        default_factory=list,
        description="本段涉及的场景 name 列表(按时序);纯心理活动留空。例:「林家大厅」",
    )
    character_refs: List[str] = Field(
        default_factory=list,
        description="本段所有出场人物 name(含无台词的)",
    )


class Beat(BeatDraft):
    """完整剧情大纲段(带全局编号 + 跨批关联)。"""

    index: int = Field(default=0, description="全局编号,1-based")
    related_batches: List[int] = Field(
        default_factory=list,
        description="该段涉及的批次编号列表,1-based 按时序",
    )


class BeatList(BaseModel):
    """整本小说累积的全部剧情段(落盘格式)。"""

    beats: List[Beat] = Field(default_factory=list)


class BeatExtraction(BaseModel):
    """本批抽取产出的剧情段列表。"""

    beats: List[BeatDraft] = Field(
        default_factory=list,
        description="本批的剧情段列表(填法见 system prompt「增量输出规则」)",
    )


__all__ = [
    "Beat",
    "BeatDraft",
    "BeatExtraction",
    "BeatList",
]
