"""统一的数据协议:在 agents、skills、workflow 之间共享的 Pydantic 模型。"""

from schema.config import LLMConfig, RunConfig
from schema.novel_analysis import (
    Batch,
    BatchState,
    Beat,
    BeatExtraction,
    BeatList,
    Character,
    CharacterExtraction,
    CharacterRoster,
    Episode,
    FinalReport,
    ReportMeta,
    ScreenplayAnalysis,
    Setting,
    SettingCollection,
    SettingExtraction,
    Storyboard,
)

__all__ = [
    "Batch",
    "BatchState",
    "Beat",
    "BeatExtraction",
    "BeatList",
    "Character",
    "CharacterExtraction",
    "CharacterRoster",
    "Episode",
    "FinalReport",
    "LLMConfig",
    "ReportMeta",
    "RunConfig",
    "ScreenplayAnalysis",
    "Setting",
    "SettingCollection",
    "SettingExtraction",
    "Storyboard",
]
