"""file_io skill 的输入契约。

``write_final_report`` 接 ``FinalReport`` → 落 JSON + Markdown,
``FinalReport`` 是这次 run 全部产物的聚合容器:

* 4 个 agent 的输出(``CharacterRoster`` / ``SettingCollection`` /
  ``BeatList`` / ``ScreenplayAnalysis``)
* 一次 run 的元信息(``ReportMeta``,无 LLM 调用,仅记录 input/output/llm/batch 等)

字段类型从 ``agents/`` 下各自的 schema 直接 import,本 schema 是它们的
"装配规范"。工作流(``workflows.novel_analysis``)按这个契约组装,落盘交给本 skill。
"""

from __future__ import annotations

from pydantic import BaseModel

from agents.extract_beats.schema import BeatList
from agents.extract_characters.schema import CharacterRoster
from agents.extract_settings.schema import SettingCollection
from agents.extract_storyboard.schema import ScreenplayAnalysis


class ReportMeta(BaseModel):
    source_path: str
    txt_path: str = ""
    title: str = ""
    total_chars: int = 0
    batch_count: int = 0
    max_batch_chars: int = 0
    max_total_chars: int = 0
    llm_base_url: str = ""
    llm_model: str = ""


class FinalReport(BaseModel):
    """最终产出:剧本 + 人物 + 场景 + 节拍 + 元信息,平级。"""
    screenplay: ScreenplayAnalysis
    characters: CharacterRoster
    settings: SettingCollection
    beats: BeatList
    meta: ReportMeta


__all__ = ["FinalReport", "ReportMeta"]
