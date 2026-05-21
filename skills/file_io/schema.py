"""file_io skill 的输入契约。

``write_final_report`` 接 ``FinalReport`` → 落 JSON + Markdown,
``FinalReport`` 是这次 run 全部产物的聚合容器:

* 3 个 agent 的输出(``CharacterRoster`` / ``BeatList`` / ``ScreenplayAnalysis``)
* 一次 run 的元信息(``ReportMeta``,无 LLM 调用,仅记录 input/output/llm/batch 等)

字段类型从 ``agents/`` 下各自的 schema 直接 import,本 schema 是它们的
"装配规范"。工作流(``workflows.novel_analysis``)按这个契约组装,落盘交给本 skill。

**注:** 本工程不维护独立的场景视觉档案。Beat 内 ``setting_refs`` 只是字符串
label;每镜的视觉环境由 storyboard agent 写到 ``Storyboard.description`` 里。
"""

from __future__ import annotations

from typing import Dict

from pydantic import BaseModel, Field

from agents.extract_beats.schema import BeatList
from agents.extract_characters.schema import CharacterRoster
from agents.extract_storyboard.schema import ScreenplayAnalysis


class AgentLLMInfo(BaseModel):
    """单个 agent 用的 LLM(基础 URL + 模型名)。"""
    base_url: str = ""
    model: str = ""


class ReportMeta(BaseModel):
    source_path: str
    txt_path: str = ""
    title: str = ""
    total_chars: int = 0
    batch_count: int = 0
    max_batch_chars: int = 0
    max_total_chars: int = 0
    llm_per_agent: Dict[str, AgentLLMInfo] = Field(
        default_factory=dict,
        description=(
            "每个 agent 用的 LLM 配置摘要(base_url + model),"
            "因为本工程每个 agent 在自己的 ``llm.json`` 里独立选模型"
        ),
    )


class FinalReport(BaseModel):
    """最终产出:剧本 + 人物 + 节拍 + 元信息,平级。"""
    screenplay: ScreenplayAnalysis
    characters: CharacterRoster
    beats: BeatList
    meta: ReportMeta


__all__ = ["AgentLLMInfo", "FinalReport", "ReportMeta"]
