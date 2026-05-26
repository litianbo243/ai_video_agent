"""分集规划相关数据契约(``episode_planner`` agent 产出)。

把 N 个 Beat 聚合成 1 个 Episode(目标时长 ~target_duration 秒)。LLM 一次性
看全部 Beat 做规划,主输出 ``EpisodePlanList``(只含元数据 + ``beat_indices``,
不含分镜本身);分镜由下游叙事 / 视觉 agent 按 ``beat_indices`` 取对应
Beat + 跨 batch 原文展开。
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class EpisodePlan(BaseModel):
    """一集的规划元数据(LLM 直接产出;不含分镜)。"""

    title: str = Field(
        ...,
        description="本集标题,4-8 字,概括跨段主线;不含集号。例:「设局陷害」",
    )
    synopsis: str = Field(
        default="",
        description="本集剧情概要(2-3 句),把所含 beats 的 summary 拧成一根主线;不要罗列。",
    )
    director_intent: str = Field(
        default="",
        description="本集叙事调性(2-4 句):基调 / 重点 / 节奏 / 视觉锤。下游分镜 agent 会拿这个当总调子。",
    )
    beat_indices: List[int] = Field(
        ...,
        description="本集所含 Beat.index 列表(按时序,1-based,严格升序连续,不跳号不重号)。",
    )


class EpisodePlanList(BaseModel):
    """全书的分集规划(LLM 单次产出)。

    所有 plan 的 ``beat_indices`` 并集 = 全部 Beat.index(不重不漏)。
    """

    plans: List[EpisodePlan] = Field(default_factory=list)


__all__ = [
    "EpisodePlan",
    "EpisodePlanList",
]
