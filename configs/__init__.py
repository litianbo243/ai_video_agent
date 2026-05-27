"""运行时配置目录。

* ``novel_analysis_config.py`` —— ``RunConfig`` / ``RunMode`` / ``load_config``（novel_analysis 专用）
* ``generate_character_prompt_config.py`` —— ``GenerateCharacterPromptConfig`` / ``load_generate_character_prompt_config``
* ``*.json``        —— 实际配置文件

LLM 配置不在这里,见 ``llm.llm_config.LLMConfig``(每个 agent 自治)。

外部统一用 ``from configs import ...``。
"""

from configs.generate_character_prompt_config import (
    GenerateCharacterPromptConfig,
    load_generate_character_prompt_config,
)
from configs.novel_analysis_config import RunConfig, RunMode, load_config, mode_includes

__all__ = [
    "GenerateCharacterPromptConfig",
    "RunConfig",
    "RunMode",
    "load_config",
    "load_generate_character_prompt_config",
    "mode_includes",
]