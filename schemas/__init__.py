"""集中存放整条 pipeline 的所有 Pydantic 数据契约。

设计理念:
* **schemas/ 是唯一来源**:agent 自己的 ``__init__.py`` 不 re-export schema,
  所有业务代码必须 ``from schemas.X import Y``,杜绝"两个真相"。
* **每个 LLM-direct agent 一个文件**:character / beat / episode_plan /
  narrative / shot_direction 各一份,文件名跟 agent 名对应。
* **跨 agent 的聚合容器**(Storyboard / Episode / ScreenplayAnalysis)
  集中放在 ``storyboard.py``,它们不属于任何单一 agent,是 workflow
  组装出来的"成品"。
* **顶层报告契约**(FinalReport / ReportMeta / AgentLLMInfo)放
  ``report.py``,跟 ``skills.file_io`` 配套。

**对 LLM-direct schema 的特别提示**:class docstring + field description
都会经 ``model_json_schema()`` 塞进 LLM 的 system prompt,所以只写
"是什么 / 怎么填 / 微例",**不放** "logic.py:xxx" / "workflow 内部" /
内部 module 名(LLM 看不懂)。决策规则的权威源是各 agent 的
``logic.py:SYSTEM_PROMPT``,在 description 里复述会让 LLM 注意力分散。
维护指引只放本模块 docstring —— Pydantic 不抓模块级 docstring。
"""

from __future__ import annotations

from schemas.beat import Beat, BeatDraft, BeatExtraction, BeatList
from schemas.character import (
    Character,
    CharacterDraft,
    CharacterExtraction,
    CharacterList,
)
from schemas.episode_plan import EpisodePlan, EpisodePlanList
from schemas.narrative import NarrativeShot, NarrativeShotList
from schemas.report import AgentLLMInfo, FinalReport, ReportMeta
from schemas.shot_direction import ShotDirection, ShotDirectionList, ShotType
from schemas.storyboard import Episode, ScreenplayAnalysis, Storyboard

__all__ = [
    "AgentLLMInfo",
    "Beat",
    "BeatDraft",
    "BeatExtraction",
    "BeatList",
    "Character",
    "CharacterDraft",
    "CharacterExtraction",
    "CharacterList",
    "Episode",
    "EpisodePlan",
    "EpisodePlanList",
    "FinalReport",
    "NarrativeShot",
    "NarrativeShotList",
    "ReportMeta",
    "ScreenplayAnalysis",
    "ShotDirection",
    "ShotDirectionList",
    "ShotType",
    "Storyboard",
]
