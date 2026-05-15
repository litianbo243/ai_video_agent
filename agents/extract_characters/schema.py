"""extract_characters agent 的全部数据契约。

层次:

* ``CharacterDraft``     —— LLM 直接产出的初稿(无 ``index``)
* ``Character``          —— 合并后完整的人物档案(``CharacterDraft`` + ``index``)
* ``CharacterRoster``    —— 一次 run 的全部人物
* ``CharacterExtraction``—— 单次 LLM 调用的输出包装

数据语义跟"这次 LLM 调用想抽什么"绑死,所以全部住在 agent 内部。
``Character`` / ``CharacterRoster`` 虽然会跨 workflow 流通,但字段形状完全
由本 agent 的 LLM 抽取契约决定,放在一起最自洽。

**schema description 写作原则**(给后续维护者):

* description 只讲"字段是什么 / 怎么填 / 一个微例",**不讲决策规则**——
  决策规则的权威源是 ``logic.py:SYSTEM_PROMPT``,在 description 里复述会让
  LLM 注意力分散、以后改一处忘改另一处。
* 不用 markdown(``**bold**`` / ``- bullets``),纯文本中文模型友好。
* 1-2 行能讲完的别写 5 行。
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class CharacterDraft(BaseModel):
    """有戏剧作用的出场人物档案。"""

    name: str = Field(
        ...,
        description="人物主名,中文。例:萧炎。同名沿用 prompt 名录里的写法。",
    )
    aliases: List[str] = Field(
        default_factory=list,
        description="本批新出现的别名 / 称谓 / 外号(如「三少爷」「废物」)。",
    )
    appearance: str = Field(
        default="",
        description=(
            "整段外貌描写,150-300 字,连贯叙述不分点,客观第三人称。"
            "给图像生成模型当 prompt 用。本批没新描写就留空。"
        ),
    )
    personality: str = Field(
        default="",
        description=(
            "整段性格分析,150-300 字,连贯叙述不分点。"
            "只基于本批正文实际表现,本批没新内容就留空。"
        ),
    )


class Character(CharacterDraft):
    """完整人物档案(workflow 内部 + 落盘格式)。"""

    index: int = Field(default=0, description="全局编号,1-based。")


class CharacterRoster(BaseModel):
    characters: List[Character] = Field(default_factory=list)


class CharacterExtraction(BaseModel):
    """单次 LLM 调用的输出包装。本批新出现或描写显著更新的人物增量列表。"""

    new_or_updated_characters: List[CharacterDraft] = Field(default_factory=list)


__all__ = [
    "Character",
    "CharacterDraft",
    "CharacterExtraction",
    "CharacterRoster",
]
