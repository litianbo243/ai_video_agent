"""storyboard_for_beat skill 的全部数据契约。

层次由小到大:``Storyboard``(单镜) → ``Episode``(单集) → ``ScreenplayAnalysis``(全本剧本)。
LLM 单次调用产出一集的分镜清单(``StoryboardList``),由 workflow 包装成 ``Episode``,
最后所有 ``Episode`` 组合成 ``ScreenplayAnalysis``。
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class Storyboard(BaseModel):
    """单个分镜(对应一个具体镜头)。LLM 直接按这个 schema 产出。"""

    index: int = Field(..., description="在该集中的序号(1-based)")
    shot_type: str = Field(
        default="",
        description="镜头类型:远景 / 全景 / 中景 / 近景 / 特写 / 大特写",
    )
    description: str = Field(
        default="",
        description="画面描述。直接给图像生成模型当 prompt 用,要写明谁在做什么 + 光线 + 构图。",
    )
    characters: List[str] = Field(
        default_factory=list,
        description="出场人物的正式 name(对应 CharacterRoster)",
    )
    setting: str = Field(
        default="",
        description="本镜场景:引用 Setting.name(单一地点)",
    )
    dialogue: str = Field(
        default="",
        description=(
            "本镜内的**开口说话**原文(纯净 TTS):角色对白、群众议论、测验员/执事播报等。"
            "不要把小说第三人称叙述放在这里。"
            "长台词跨多镜时,每镜只放对应该画面的那一段。"
        ),
    )
    voiceover: str = Field(
        default="",
        description=(
            "本镜内的**第三人称旁白**原文(纯净 TTS),默认可留空。"
            "仅当画面难以单独传达、且不是任何人「说出声」的对白时使用;"
            "若 description 已交代同义信息则必须留空。"
            "长旁白跨多镜时按画面切点切片,每镜最多一句、宜 ≤35 字。"
        ),
    )
    duration_sec: float = Field(
        default=0.0,
        description=(
            "本镜画面 hold 时长(秒),建议 3-15。"
            "= max(画面节奏需求, 念完 dialogue + voiceover 所需时间)。"
            "估算口径:对白 ~3.5 字/秒,旁白 ~3 字/秒。"
        ),
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
    """LLM 单段 Beat 的输出:本集的分镜清单。

    薄包装 —— ``chat_json`` 要求 schema 是 ``BaseModel`` 子类,LLM 实际产
    ``storyboards`` 列表,外面再包成 ``Episode``。
    """

    storyboards: List[Storyboard] = Field(default_factory=list)


__all__ = [
    "Episode",
    "ScreenplayAnalysis",
    "Storyboard",
    "StoryboardList",
]
