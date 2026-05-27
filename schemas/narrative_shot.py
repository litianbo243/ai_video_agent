"""叙事分镜相关数据契约(``narrative_director`` agent 产出)。

* ``NarrativeShot``           —— 单镜叙事维度(LLM 直接产出)
* ``NarrativeShotList`` —— 本集所有叙事分镜的有序清单(LLM 单次产出)

**职责切分**:本 agent 写**叙事维度**(谁 / 在哪 / 说什么 / 想什么 + 集层
``director_intent``);**视觉维度**(景别 / 运镜 / 起始画面 / 时长)归
``shot_director`` agent。两者按 ``index`` 配对合并成 ``Shot``
(见 ``schemas/screenplay.py``)。
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class NarrativeShot(BaseModel):
    """单镜的叙事维度(LLM 直接产出;视觉字段不在此 agent 职责内)。"""

    index: int = Field(..., description="本镜在本集中的序号,从 1 开始连续编号。")
    intent: str = Field(
        default="",
        description="本镜抽象意图(为什么这么演 / 让观众感觉什么),精简一句 ≤20 字。",
    )
    description: str = Field(
        default="",
        description="本镜剧情发生(谁做了什么 / 剧情怎么推进),1-2 句。**不写画面 / 镜头 / 光线**(画面归下游镜头导演)。",
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
        description="本镜的人物内心独白(≤35 字)。",
    )
    duration_sec: float = Field(
        default=0.0,
        description="本镜时长(秒)。",
    )


class NarrativeShotList(BaseModel):
    """本集所有叙事分镜的有序清单(LLM 单次产出;填法见 system prompt)。"""

    director_intent: str = Field(
        default="",
        description="本集叙事调性(2-4 句):基调 / 重点 / 节奏 / 视觉锤(供下游镜头导演参考)。先填这个,再填 shots。",
    )
    shots: List[NarrativeShot] = Field(default_factory=list)


__all__ = [
    "NarrativeShot",
    "NarrativeShotList",
]
