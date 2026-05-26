"""顶层报告契约(``skills.file_io.write_final_report`` 落盘的契约)。

``FinalReport`` 是一次 run 的全部产物聚合:
* 3 个 agent 的输出(``CharacterList`` / ``BeatList`` / ``ScreenplayAnalysis``)
* 元信息(``ReportMeta``,无 LLM 调用,仅记录 input/output/llm/batch 等)

**注:** 本工程不维护独立的场景视觉档案。Beat 内 ``setting_refs`` 只是字符串
label;每镜的视觉环境由 storyboard agent 写到 ``Storyboard.description`` 里。
"""

from __future__ import annotations

from typing import Dict

from pydantic import BaseModel, Field

from schemas.beat import BeatList
from schemas.character import CharacterList
from schemas.storyboard import ScreenplayAnalysis


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
    characters: CharacterList
    beats: BeatList
    meta: ReportMeta


__all__ = [
    "AgentLLMInfo",
    "FinalReport",
    "ReportMeta",
]
