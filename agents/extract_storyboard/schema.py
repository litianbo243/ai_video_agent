"""extract_storyboard agent 的全部数据契约。

层次由小到大:``Storyboard``(单镜) → ``Episode``(单集) → ``ScreenplayAnalysis``(全本剧本)。
LLM 单次调用产出一集的分镜清单(``StoryboardList``),由 workflow 包装成 ``Episode``,
最后所有 ``Episode`` 组合成 ``ScreenplayAnalysis``。

**schema description 写作原则**(给后续维护者):

* description 只讲"字段是什么 / 怎么填 / 一个微例",**不讲决策规则**——
  决策规则的权威源是 ``logic.py:SYSTEM_PROMPT``,在 description 里复述会让
  LLM 注意力分散、以后改一处忘改另一处。
* 不用 markdown(``**bold**`` / ``- bullets``),纯文本中文模型友好。
* 字段类型已用 ``Literal`` 约束的(如 ``ShotType``)就不再列举枚举值。
"""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field


ShotType = Literal["远景", "全景", "中景", "近景", "特写", "大特写"]


class Storyboard(BaseModel):
    """单个分镜:一个具体镜头(3-15 秒)。"""

    index: int = Field(..., description="在本集中的序号,1-based,从 1 开始连续编号。")
    shot_type: ShotType = Field(
        ...,
        description="镜头类型,从枚举里 6 选 1。",
    )
    description: str = Field(
        default="",
        description=(
            "画面描述,直接给图像生成模型当 prompt。"
            "格式:谁在做什么 + 环境 + 光线 + 构图视角。出场人物用正式 name(不用别名)。"
        ),
    )
    characters: List[str] = Field(
        default_factory=list,
        description="本镜画面里可辨认的出场人物 name(取自人物档案的正式 name)。",
    )
    setting: str = Field(
        default="",
        description=(
            "本镜场景 name,单一地点,必须取自 Beat.setting_refs 里某一个。"
        ),
    )
    dialogue: str = Field(
        default="",
        description=(
            "本镜内开口说话的原文(纯净 TTS 用):对白 / 群众议论 / 测验员播报等。"
            "照抄原文台词,本镜没人说话留空。"
        ),
    )
    voiceover: str = Field(
        default="",
        description=(
            "本镜的第三人称旁白(纯净 TTS 用),默认留空。"
            "仅当画面单独难以传达必要信息时,从原文浓缩为一句(≤35 字)。"
        ),
    )
    duration_sec: float = Field(
        default=0.0,
        description="本镜画面 hold 时长(秒,3-15,平均 6-8)。",
    )


class Episode(BaseModel):
    """一集(对应一段 Beat)。workflow 把 ``StoryboardList`` 包装成 ``Episode``。"""

    index: int = Field(..., description="集序号(1-based,= Beat.index)")
    title: str = Field(default="", description='本集标题,如"第一集 · 废柴觉醒"')
    synopsis: str = Field(default="", description="本集剧情概要(1-2 段)")
    beat_index: int = Field(default=0, description="对应 Beat.index")
    storyboards: List[Storyboard] = Field(
        default_factory=list, description="本集所有分镜(按时序)"
    )


class ScreenplayAnalysis(BaseModel):
    """剧本分析:全书 logline + 所有分集(含分镜)。所有 Episode 的聚合。"""

    title: str = Field(default="")
    logline: str = Field(default="", description="一句话电梯陈述")
    episodes: List[Episode] = Field(default_factory=list)


class StoryboardList(BaseModel):
    """本集所有分镜的有序清单。

    每集 ~15-30 个镜头,所有镜头 ``duration_sec`` 之和 ≈ 目标集时长。
    """

    storyboards: List[Storyboard] = Field(default_factory=list)


__all__ = [
    "Episode",
    "ScreenplayAnalysis",
    "ShotType",
    "Storyboard",
    "StoryboardList",
]
