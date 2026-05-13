"""单批人物抽取:一次 LLM 调用,返回本批增量。

合并(同名融合、新增赋 index)放在 ``manager.py``,这里只负责 prompt + LLM 调用。
"""

from __future__ import annotations

import logging
from typing import Dict, List

from llm.client import LLMClient
from skills.batch_chapters import Batch
from agents.extract_characters.schema import Character, CharacterExtraction

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """\
你是中文长篇小说的人物抽取助手,正在为下游 AI 短视频生产管线构建结构化数据。

工作模式: 增量分析。
- 每次只看一批章节正文 + 此前已知的人物名单;
- 输出"本批次产生的变化"(delta), 不重写已经分析过的章节。

────────────────────────────────────────────────
对每个出场人物,产出:
- name: 规范化的中文姓名(主人物以正名,别号外号放进 aliases)
- aliases: 该批次出现的别名 / 称谓 / 外号
- appearance: **整段外貌描写,150-300 字**
    含性别 + 年龄段 + 身材 + 发型发色 + 眼睛特征 + 服饰风格 + 标志性配饰 + 整体气质
    例:"16 岁少年,身材瘦削挺拔,黑色短发自然垂落耳际,深邃黑眸常带倔强神色,
    常着青色粗布长袍,左手中指戴一枚乌黑纳戒,气质冷峻略带锋芒。"
- personality: **整段性格分析,150-300 字**
    含行为模式 + 价值观 + 情感倾向 + 弧光走向

────────────────────────────────────────────────
合并规则:
- 已在 [此前已知人物名单] 中的角色,**沿用同名,不要改名**
- 仅在你发现新信息(外貌细节 / 性格转变 / 新别名)的老角色才输出更新;
  - appearance / personality:**只在你看到比之前更详细的描写时才填**;否则留空 = 不更新
  - aliases:只列本批新出现的别名,merge 时会与旧别名取并集
- 路人角色(只出场一次、无名 / 无戏剧作用)**不要**输出
- 绝不杜撰: 没有依据的字段直接留空

────────────────────────────────────────────────
严格按 JSON Schema 输出 CharacterExtraction,只输出一个 JSON 对象。
"""


def _condense_known_roster(known: Dict[str, Character]) -> str:
    if not known:
        return "(暂无)"
    items: List[str] = []
    for ch in known.values():
        bits = [ch.name]
        if ch.aliases:
            bits.append("别名:" + "/".join(ch.aliases[:3]))
        items.append(" ".join(bits))
    return "\n".join("- " + s for s in items)


def _build_user_prompt(batch: Batch, known: Dict[str, Character], title: str) -> str:
    roster = _condense_known_roster(known)
    book_title = title or "(未提供书名)"
    return (
        f"书名: {book_title}\n"
        f"批次序号: 第 {batch.index} 批\n"
        f"批次字数: 约 {batch.char_count}\n\n"
        f"=== 此前已知人物名单 ===\n{roster}\n\n"
        f"=== 本批次正文 ===\n{batch.render_for_prompt()}\n\n"
        f"=== 任务 ===\n请按 JSON Schema 输出 CharacterExtraction。"
    )


def extract_for_batch(
    batch: Batch,
    known: Dict[str, Character],
    llm: LLMClient,
    *,
    title: str = "",
) -> CharacterExtraction:
    """对单个 batch 调一次 LLM,返回人物增量(不在此处合并)。"""
    user_prompt = _build_user_prompt(batch, known, title)
    logger.info(
        "character_extractor 第 %d 批(已知 %d 人),%s @ %s",
        batch.index, len(known), llm.model, llm.base_url,
    )
    delta = llm.chat_json(SYSTEM_PROMPT, user_prompt, CharacterExtraction)
    logger.info(
        "character_extractor 第 %d 批产出:%d 人(新增/更新)",
        batch.index, len(delta.new_or_updated_characters),
    )
    return delta


__all__ = ["extract_for_batch", "SYSTEM_PROMPT"]
