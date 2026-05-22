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
    """单个分镜:一个具体镜头。"""

    index: int = Field(..., description="本镜在本集中的序号,从 1 开始连续编号。")
    shot_type: ShotType = Field(..., description="镜头类型。")
    intent: str = Field(
        default="",
        description="本镜意图(为什么这么拍 / 让观众感觉什么,精简一句)。",
    )
    description: str = Field(
        default="",
        description="本镜起始画面 + 整体氛围(给图像生成模型当 prompt,也作视频生成的起点画面)。",
    )
    shot_action: str = Field(
        default="",
        description="本镜内画面如何变化(动词为主,如「缓缓抬头,泪痕未干,目光转向窗外」);静态镜头留空。",
    )
    camera_motion: str = Field(
        default="",
        description="本镜运镜方式(推 / 拉 / 摇 / 移 / 升 / 降 / 跟 / 环绕 / 固定);留空 = 固定机位。",
    )
    characters: List[str] = Field(
        default_factory=list,
        description="本镜画面里可辨认的出场人物 name 列表。",
    )
    setting: str = Field(
        default="",
        description="本镜场景 name(单一地点)。",
    )
    speaker: str = Field(
        default="",
        description="本镜发声者 name(无人发声时留空)。",
    )
    dialogue: str = Field(
        default="",
        description="本镜的人物台词原文(不含人名前缀)。",
    )
    voiceover: str = Field(
        default="",
        description="本镜的画外旁白 / 人物内心独白(≤35 字)。",
    )
    duration_sec: float = Field(
        default=0.0,
        description="本镜时长(秒)。",
    )


class Episode(BaseModel):
    """一集(1:1 对应一段 Beat,``index`` 与 ``Beat.index`` 相同)。

    workflow 把 ``StoryboardList`` 包装成 ``Episode``,要看本集对应的 Beat 详情
    用 ``ep.index`` 反查 ``BeatList`` 即可。
    """

    index: int = Field(..., description="集序号(1-based,= 对应 Beat.index)")
    title: str = Field(default="", description='本集标题,如"第一集 · 废柴觉醒"')
    synopsis: str = Field(default="", description="本集剧情概要(1-2 段)")
    director_intent: str = Field(
        default="",
        description="本集导演意图(2-4 句):基调 / 重点 / 节奏 / 视觉锤。",
    )
    storyboards: List[Storyboard] = Field(
        default_factory=list, description="本集所有分镜(按时序)"
    )


class ScreenplayAnalysis(BaseModel):
    """剧本分析:全书所有分集(含分镜)的聚合。

    书名走 ``FinalReport.meta.title``,本类只关心分集内容。
    """

    episodes: List[Episode] = Field(default_factory=list)


class StoryboardList(BaseModel):
    """本集所有分镜的有序清单(填法见 system prompt)。"""

    director_intent: str = Field(
        default="",
        description="本集导演意图(2-4 句):基调 / 重点 / 节奏 / 视觉锤。先填这个,再填 storyboards,让每个镜头都服务于该意图。",
    )
    storyboards: List[Storyboard] = Field(default_factory=list)


__all__ = [
    "Episode",
    "ScreenplayAnalysis",
    "ShotType",
    "Storyboard",
    "StoryboardList",
]
