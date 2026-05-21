"""extract_beats agent 的全部数据契约。

层次:

* ``BeatDraft``      —— LLM 直接产出的初稿(无 ``index`` / ``related_batches``)
* ``Beat``           —— 合并后完整的剧情段(含 ``index`` / ``related_batches`` / ``is_open``)
* ``BeatList``       —— 一次 run 的全部剧情段
* ``BeatExtraction`` —— 单次 LLM 调用的输出包装(新段 + 延续段 + last_beat_open)

数据语义跟"这次 LLM 调用想抽什么"绑死,所以全部住在 agent 内部。

**schema description 写作原则**(给后续维护者):

* description 只讲"字段是什么 / 怎么填 / 一个微例",**不讲决策规则**——
  决策规则的权威源是 ``logic.py:SYSTEM_PROMPT_TEMPLATE``,在 description 里复述
  会让 LLM 注意力分散、以后改一处忘改另一处。
* 不用 markdown(``**bold**`` / ``- bullets``),纯文本中文模型友好。
* 1-2 行能讲完的别写 5 行。
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class BeatDraft(BaseModel):
    """剧情大纲段的 LLM 初稿。一段 ≈ 一集短视频。"""

    title: str = Field(
        ...,
        description="剧情段标题,2-6 字,只描述事件不含地点。如「会议决裂」「揭穿身世」「夜袭仓库」。",
    )
    summary: str = Field(
        default="",
        description="1-2 句话:谁在哪、做了什么、转折是什么。不抄原文台词。",
    )
    setting_refs: List[str] = Field(
        default_factory=list,
        description=(
            "本段涉及的场景 name 列表,按时序。"
            "已用过的 name 沿用不改字;新地点用「宅院/机构+房间」组合(如「林家大厅」「沈宅书房」「明远集团会议室」)。"
            "纯心理活动留空。"
        ),
    )
    character_refs: List[str] = Field(
        default_factory=list,
        description="本段所有出场人物的 name(不只有台词的)。必须出自 prompt 里的人物名录。",
    )


class Beat(BeatDraft):
    """完整剧情大纲段(workflow 内部 + 落盘格式)。"""

    index: int = Field(default=0, description="全局编号,1-based。")
    related_batches: List[int] = Field(
        default_factory=list,
        description="该段涉及的 batch 编号列表,1-based 按时序。",
    )
    is_open: bool = Field(
        default=False,
        description=(
            "本段是否还开着、等下一 batch 续写。"
            "全书任意时刻最多 1 个段 is_open=True。"
        ),
    )


class BeatList(BaseModel):
    beats: List[Beat] = Field(default_factory=list)


class BeatExtraction(BaseModel):
    """单次 LLM 调用的输出包装,三条通道协同(new_beats / continues_open_beat / last_beat_open)。"""

    new_beats: List[BeatDraft] = Field(
        default_factory=list,
        description="本批中新起的剧情段。数量按内容密度决定,无硬上限;典型每 4-6K 字 1 段。",
    )
    continues_open_beat: bool = Field(
        default=False,
        description=(
            "本批是否在续写 prompt 里那个标 [待续] 的段?"
            "true 把本 batch 加到那段的 related_batches;false 让它被自动收束。"
            "prompt 里没 [待续] 段时保持 false。"
        ),
    )
    last_beat_open: bool = Field(
        default=False,
        description=(
            "本批新写的最后一段(new_beats 末尾)是否未到自然收束?"
            "true 让下一批续写它;false 表示已收束。"
            "new_beats 为空时此字段无意义,保持 false。"
        ),
    )


__all__ = [
    "Beat",
    "BeatDraft",
    "BeatExtraction",
    "BeatList",
]
