"""人物相关数据契约(``extract_characters`` agent 产出 + workflow 聚合)。

* ``CharacterDraft``      —— LLM 产出的初稿(无 ``index``)
* ``Character``           —— 合并后完整档案(含 ``index``)
* ``CharacterList``     —— 一次 run 的全部人物(落盘格式)
* ``CharacterExtraction`` —— 单次 LLM 调用的输出包装
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class CharacterDraft(BaseModel):
    """有戏剧作用的出场人物档案。"""

    name: str = Field(
        ...,
        description="人物主名,中文。例:林望",
    )
    pre_name: Optional[str] = Field(
        default=None,
        exclude=True,
        description="更新已有人物且 name 与详档原名不同时,填详档里的原 name;否则留空",
    )
    aliases: List[str] = Field(
        default_factory=list,
        description="本批原文出现过的别名 / 称谓 / 外号。例:「高校长」",
    )
    appearance: str = Field(
        default="",
        description="跨章节稳定的整段视觉外貌,给图像生成模型用;上限约 300 字",
    )
    background: str = Field(
        default="",
        description="整段背景档案(职业 / 出身 / 关系 / 经历);上限约 200 字",
    )
    personality: str = Field(
        default="",
        description="整段性格画像(模式级 / 默认态);上限约 300 字",
    )
    arc: str = Field(
        default="",
        description="剧情弧光(累积时序锚点,append-only);无弧光留空;上限约 280 字",
    )


class Character(CharacterDraft):
    """完整人物档案(带全局编号)。"""

    index: int = Field(default=0, description="全局编号,1-based")


class CharacterList(BaseModel):
    """整本小说累积的全部人物档案(落盘格式)。"""

    characters: List[Character] = Field(default_factory=list)


class CharacterExtraction(BaseModel):
    """本批抽取产出的人物增量结果。"""

    new_or_updated_characters: List[CharacterDraft] = Field(
        default_factory=list,
        description="本批新增 / 更新的人物增量",
    )


__all__ = [
    "Character",
    "CharacterDraft",
    "CharacterExtraction",
    "CharacterList",
]
