"""generate_character_prompt workflow 的配置契约 + 加载入口。

独立于 ``RunConfig``，只包含两个字段：
* ``characters_path`` —— 已产出的 characters.json 路径
* ``output_dir``       —— 提示词输出目录
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel, Field


class GenerateCharacterPromptConfig(BaseModel):
    """generate_character_prompt 的运行参数。"""

    characters_path: str = Field(..., description="已产出的 characters.json 路径")
    output_dir: str = Field(..., description="提示词输出目录（通常与 characters.json 同目录）")


def load_generate_character_prompt_config(path: str | Path) -> GenerateCharacterPromptConfig:
    """读取 + 校验 generate_character_prompt 的 JSON 配置，并打印摘要。"""
    path = Path(path)
    try:
        cfg_dict: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        cfg = GenerateCharacterPromptConfig.model_validate(cfg_dict)
    except Exception as e:
        raise ValueError(f"加载 generate_character_prompt config 失败({path}):\n{e}") from e

    print("=" * 60)
    print(f"配置已加载({path}):")
    print(f"  characters: {cfg.characters_path}")
    print(f"  output:     {cfg.output_dir}")
    print("=" * 60)

    return cfg


__all__ = ["GenerateCharacterPromptConfig", "load_generate_character_prompt_config"]