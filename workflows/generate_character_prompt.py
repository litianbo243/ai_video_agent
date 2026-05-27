"""generate_character_prompt workflow:从 characters.json 生成 SDXL tag-style 图像提示词。

独立于 novel_analysis 流水线,输入是已产出的 characters.json,输出是
character_prompts.json + character_prompts.md。

用法::
    from workflows.generate_character_prompt import run
    from pathlib import Path

    result = run(
        characters_path=Path("outputs/20260527_205624/characters.json"),
        output_dir=Path("outputs/20260527_205624"),
    )
    print(result.prompts_json_path)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from agents import character_image_prompt
from schemas.character import CharacterList
from schemas.character_prompt import CharacterImagePromptList

logger = logging.getLogger(__name__)


@dataclass
class GenerateResult:
    """``run()`` 的返回。"""

    characters_path: Path
    output_dir: Path
    prompts: CharacterImagePromptList
    prompts_json_path: Path
    prompts_md_path: Path


def _render_character_prompts_md(prompts: CharacterImagePromptList) -> str:
    parts: list[str] = ["# SDXL 人物提示词（中英文对照）\n"]
    if not prompts.prompts:
        return parts[0] + "\n_(无人物提示词)_\n"
    for item in prompts.prompts:
        parts.append(f"## #{item.index} · {item.name}")
        parts.append("")
        if item.aliases:
            parts.append(f"- 别名: {', '.join(item.aliases)}")
            parts.append("")
        parts.append("**Positive prompt (EN)**")
        parts.append("")
        parts.append(item.positive_prompt)
        parts.append("")
        parts.append("**Positive prompt (ZH)**")
        parts.append("")
        parts.append(item.positive_prompt_zh)
        parts.append("")
        parts.append("**Negative prompt (EN)**")
        parts.append("")
        parts.append(item.negative_prompt)
        parts.append("")
        parts.append("**Negative prompt (ZH)**")
        parts.append("")
        parts.append(item.negative_prompt_zh)
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def run(characters_path: str | Path, output_dir: str | Path) -> GenerateResult:
    """读取 characters.json,调用 character_image_prompt 生成提示词并落盘。

    Args:
        characters_path: 已产出的 characters.json 路径
        output_dir: 输出目录(通常与 characters.json 所在目录相同)

    Returns:
        GenerateResult,包含 prompts 对象和落盘路径。
    """
    ch_path = Path(characters_path).resolve()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ch_path.is_file():
        raise FileNotFoundError(f"characters.json 不存在:{ch_path}")

    logger.info("[generate_character_prompt] 读取人物档案:%s", ch_path)
    characters = CharacterList.model_validate_json(ch_path.read_text(encoding="utf-8"))

    logger.info("[generate_character_prompt] 开始生成 %d 位人物的 SDXL 提示词", len(characters.characters))
    prompts = character_image_prompt.generate_for_characters(characters)

    json_path = out_dir / "character_prompts.json"
    md_path = out_dir / "character_prompts.md"
    json_path.write_text(prompts.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(_render_character_prompts_md(prompts), encoding="utf-8")

    logger.info(
        "[generate_character_prompt] 完成:%d 条提示词 → %s, %s",
        len(prompts.prompts), json_path.name, md_path.name,
    )

    return GenerateResult(
        characters_path=ch_path,
        output_dir=out_dir,
        prompts=prompts,
        prompts_json_path=json_path,
        prompts_md_path=md_path,
    )


__all__ = ["GenerateResult", "run"]