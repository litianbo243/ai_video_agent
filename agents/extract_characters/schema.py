"""extract_characters agent 的全部数据契约。

层次:

* ``CharacterDraft``     —— LLM 直接产出的初稿(无 ``index``)
* ``Character``          —— 合并后完整的人物档案(``CharacterDraft`` + ``index``)
* ``CharacterRoster``    —— 一次 run 的全部人物
* ``CharacterExtraction``—— 单次 LLM 调用的输出包装(``new_or_updated_characters``)

数据语义跟"这次 LLM 调用想抽什么"绑死,所以全部住在 agent 内部。
``Character`` / ``CharacterRoster`` 虽然会跨 workflow 流通,但字段形状完全
由本 agent 的 LLM 抽取契约决定,放在一起最自洽。
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class CharacterDraft(BaseModel):
    """LLM 抽取人物的输出契约(无 ``index``)。

    LLM 直接产出此结构;character_analysis workflow 在合并时构造完整的
    ``Character`` 并赋全局 ``index``。
    """

    name: str = Field(..., description="规范化的中文姓名,使用文中最常出现的写法")
    aliases: List[str] = Field(default_factory=list, description="别名 / 称谓 / 外号")
    appearance: str = Field(
        default="",
        description=(
            "整段外貌描写(150-300 字),含性别 + 年龄 + 身材 + 发型发色 + "
            "眼睛 + 服饰 + 标志特征 + 整体气质。给图像生成模型当 prompt 用。"
        ),
    )
    personality: str = Field(
        default="",
        description=(
            "整段性格分析(150-300 字),含行为模式 + 价值观 + 情感倾向 + 弧光走向。"
        ),
    )


class Character(CharacterDraft):
    """完整的人物档案(state 内部 + 落盘格式)。

    比 ``CharacterDraft`` 多一个 ``index`` —— 给下游(出图、外部数据库等)
    用的稳定数字主键,character_analysis 合并时自动赋值。
    """

    index: int = Field(default=0, description="全局编号(1-based,merge 时自动赋值)")


class CharacterRoster(BaseModel):
    characters: List[Character] = Field(default_factory=list)


class CharacterExtraction(BaseModel):
    """LLM 单批输出:本批的人物增量。

    LLM 不输出 ``index``;调用方在合并时按"已知人数 + 1"赋全局 ``index``。
    """

    new_or_updated_characters: List[CharacterDraft] = Field(default_factory=list)


__all__ = [
    "Character",
    "CharacterDraft",
    "CharacterExtraction",
    "CharacterRoster",
]
