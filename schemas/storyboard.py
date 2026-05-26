"""跨 agent 聚合容器:由 workflow 编排时组装出来的"成品",非 LLM 直接产出。

* ``Storyboard``         —— 单镜完整契约 = ``NarrativeShot`` ∪ ``ShotDirection``
* ``Episode``            —— 一集完整契约 = ``EpisodePlan`` 元数据 +
                            ``NarrativeShotList`` 叙事调性 +
                            ``ShotDirectionList`` 视觉调性 +
                            ``List[Storyboard]``
* ``ScreenplayAnalysis`` —— 全本契约 = ``List[Episode]``

放在这里的原因:这 3 个类**跨 agent 边界**,不属于任何单一 agent
(``Storyboard`` 字段一半来自叙事分镜师、一半来自镜头导演;``Episode`` 字段
跨 3 个 agent;``ScreenplayAnalysis`` 是顶层聚合)。放在某个 agent 的 schema
下都会形成"反向 import"或归属偏置,集中放此处中立干净。

LLM **不直接产出**这 3 个对象,只用于落盘 / 渲染 / 下游消费,所以字段
description 不必走 LLM-friendly 风格,可保留 ``来自 X.Y`` 等内部引用。
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from schemas.shot_direction import ShotType


class Storyboard(BaseModel):
    """合并后的完整分镜(workflow 把 NarrativeShot + ShotDirection 按 index 合并而成)。

    LLM **不直接产出**此对象 —— 它是叙事 + 视觉两次调用的合并产物,只用于
    落盘 / 渲染 / 下游消费。
    """

    index: int = Field(..., description="本镜在本集中的序号,从 1 开始")
    intent: str = Field(default="", description="本镜抽象意图(叙事维度,≤20 字)")
    narrative_description: str = Field(
        default="",
        description="本镜剧情发生(叙事维度,1-2 句,来自 NarrativeShot.description)",
    )
    characters: List[str] = Field(default_factory=list, description="出场人物 name")
    setting: str = Field(default="", description="场景 name")
    speaker: str = Field(default="", description="发声者 name")
    dialogue: str = Field(default="", description="人物台词")
    voiceover: str = Field(default="", description="内心独白")
    shot_type: ShotType = Field(..., description="镜头景别")
    camera_motion: str = Field(default="", description="运镜方式(留空 = 固定)")
    shot_action: str = Field(default="", description="本镜内画面演变(动词为主)")
    description: str = Field(default="", description="本镜起始画面 + 氛围(视觉维度,图像/视频生成 prompt)")
    duration_sec: float = Field(default=0.0, description="本镜时长(秒)")


class Episode(BaseModel):
    """一集(由 N 个 Beat 聚合而成,N ≥ 1;聚合规则由分集规划决定)。"""

    index: int = Field(..., description="集序号(1-based,与分集规划产出顺序一致)")
    title: str = Field(default="", description='本集标题,如"第 1 集 · 设局陷害"')
    synopsis: str = Field(default="", description="本集剧情概要(1-2 段)")
    beat_indices: List[int] = Field(
        default_factory=list,
        description="本集所含 Beat.index 列表(按时序,1-based);分集规划产出",
    )
    director_intent: str = Field(
        default="",
        description="本集叙事调性(2-4 句):分集规划先给基线,叙事分镜阶段可微调",
    )
    visual_style: str = Field(
        default="",
        description="本集视觉调性(2-3 句,镜头导演产出)",
    )
    storyboards: List[Storyboard] = Field(
        default_factory=list, description="本集所有分镜(按时序)"
    )


class ScreenplayAnalysis(BaseModel):
    """剧本分析:全书所有分集(含分镜)的聚合。

    书名走 ``FinalReport.meta.title``,本类只关心分集内容。
    """

    episodes: List[Episode] = Field(default_factory=list)


__all__ = [
    "Episode",
    "ScreenplayAnalysis",
    "Storyboard",
]
