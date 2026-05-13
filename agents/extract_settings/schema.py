"""extract_settings agent 的全部数据契约。

层次:

* ``SettingDraft``      —— LLM 直接产出的初稿(无 ``index``)
* ``Setting``           —— 合并后完整的场景档案
* ``SettingCollection`` —— 一次 run 的全部场景
* ``SettingExtraction`` —— 单次 LLM 调用的输出包装

数据语义跟"这次 LLM 调用想抽什么"绑死,所以全部住在 agent 内部。
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class SettingDraft(BaseModel):
    """LLM 抽取场景的输出契约(无 ``index``)。"""

    name: str = Field(
        ...,
        description='地点名(全局唯一,跨 Beat 复用),如"萧家祠堂"、"乌坦城广场"',
    )
    description: str = Field(
        default="",
        description=(
            "整段视觉环境描写(150-300 字),含建筑/布局/家具/光线/氛围/关键道具。"
            "给图像生成模型当 prompt 用。"
        ),
    )


class Setting(SettingDraft):
    """完整的场景档案(state 内部 + 落盘格式)。

    比 ``SettingDraft`` 多一个 ``index`` —— 稳定数字主键,setting_analysis
    合并时自动赋值。
    """

    index: int = Field(default=0, description="全局编号(1-based,merge 时自动赋值)")


class SettingCollection(BaseModel):
    settings: List[Setting] = Field(default_factory=list)


class SettingExtraction(BaseModel):
    """LLM 单批输出:本批的场景增量。

    LLM 不输出 ``index``;调用方在合并时按"已知场景数 + 1"赋值。
    """

    new_or_updated_settings: List[SettingDraft] = Field(default_factory=list)


__all__ = [
    "Setting",
    "SettingCollection",
    "SettingDraft",
    "SettingExtraction",
]
