"""镜头视觉指导相关数据契约(``shot_director`` agent 产出)。

* ``ShotType``           —— 镜头景别枚举(``Literal``)
* ``ShotDirection``      —— 单镜视觉指导(LLM 直接产出)
* ``ShotDirectionList``  —— 本集所有镜头的视觉指导(LLM 单次产出)

每个 ``ShotDirection.index`` 与上游 ``NarrativeShot.index`` 一一对应,
workflow 按 index 配对合并成 ``Shot``(见 ``schemas/screenplay.py``)。
"""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field


ShotType = Literal["远景", "全景", "中景", "近景", "特写", "大特写"]


class ShotDirection(BaseModel):
    """单镜视觉指导:对一个叙事分镜的"怎么拍"决策。"""

    index: int = Field(
        ...,
        description="对应的叙事分镜 NarrativeShot.index(1-based,严格一一对应)。",
    )
    shot_type: ShotType = Field(..., description="镜头景别。")
    camera_motion: str = Field(
        default="",
        description="本镜运镜方式(推 / 拉 / 摇 / 移 / 升 / 降 / 跟 / 环绕 / 固定);留空 = 固定机位。",
    )
    shot_action: str = Field(
        default="",
        description="本镜内画面如何变化(动词为主,如「缓缓抬头,泪痕未干,目光转向窗外」);静态镜头留空。",
    )
    description: str = Field(
        default="",
        description="本镜起始画面 + 整体氛围(给图像生成模型当 prompt,也作视频生成的起点画面)。",
    )


class ShotDirectionList(BaseModel):
    """本集所有镜头的视觉指导(LLM 单次调用产物)。"""

    visual_style: str = Field(
        default="",
        description="本集视觉调性(2-3 句):色调 / 光影 / 构图 / 运镜偏好 / 视觉锤。先填这个,再填 shots,让每镜视觉服务于该调性。",
    )
    shots: List[ShotDirection] = Field(
        default_factory=list,
        description="本集所有镜头的视觉指导,按 index 升序,与上游叙事分镜一一对应。",
    )


__all__ = [
    "ShotDirection",
    "ShotDirectionList",
    "ShotType",
]
