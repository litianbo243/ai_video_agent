"""单批人物抽取:一次 LLM 调用,返回本批增量。

合并(同名融合、新增赋 index)放在 ``manager.py``,这里只负责 prompt + LLM 调用。

prompt 上下文采用**两层 index**:

* Tier 1 全员名录(name + 别名,所有已知人物)
  —— 让 LLM 知道"宇宙边界",防止把别名 / 称谓误认成新角色
* Tier 2 本批相关详档(name + aliases + appearance + personality,仅 substring
  扫到的子集)
  —— 让 LLM 判断"appearance / personality 是否需要更新"时有真实对比基准
"""

from __future__ import annotations

import logging
from typing import Dict, List

from llm.client import LLMClient
from skills.batch_chapters import Batch
from agents.extract_characters.schema import Character, CharacterExtraction

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """\
你是中文小说人物抽取助手。增量分析:看一批正文 + 全员名录 + 本批相关详档,输出本批 delta。

字段(每个出场人物):
- name: 规范化中文姓名(主名;别号进 aliases)
- aliases: 本批新出现的别名 / 称谓 / 外号
- appearance: 150-300 字整段外貌(性别 + 年龄 + 身材 + 发型发色 + 眼睛 + 服饰 + 标志特征 + 气质)
- personality: 150-300 字整段性格(行为模式 + 价值观 + 情感倾向 + 弧光)

规则:
- 名录里的角色沿用同名,不改名
- 详档里的人物:appearance / personality 留空 = 不更新;
  仅当本批描写比详档更细才填**完整新版**(整段覆盖,不要差量)
- aliases 只列本批新出现的
- 本批的称谓 / 别名若能对应名录某 name → 沿用那 name,称谓加进 aliases,不新建
- 路人(单次出场、无戏剧作用)不输出
- 没依据的字段留空,不杜撰

只输出一个 JSON 对象,符合 CharacterExtraction schema。
"""


def _render_roster_index(known: Dict[str, Character]) -> str:
    """Tier 1 全员名录:name + 前 3 别名,1 行/人。"""
    if not known:
        return "(暂无)"
    items: List[str] = []
    for ch in known.values():
        bits = [ch.name]
        if ch.aliases:
            bits.append("别名:" + "/".join(ch.aliases[:3]))
        items.append("- " + " ".join(bits))
    return "\n".join(items)


def _scan_relevant(text: str, known: Dict[str, Character]) -> List[Character]:
    """Tier 2 候选筛选:name / aliases 在 batch 正文里出现即算相关。

    约束:单字 key 不参与(避免"云" / "风"误匹配)。
    """
    relevant: List[Character] = []
    for ch in known.values():
        keys = [ch.name, *ch.aliases]
        if any(k and len(k) >= 2 and k in text for k in keys):
            relevant.append(ch)
    return relevant


def _render_full_profile(ch: Character) -> str:
    section = [f"### {ch.name}"]
    if ch.aliases:
        section.append(f"别名: {', '.join(ch.aliases)}")
    if ch.appearance:
        section.append(f"外貌: {ch.appearance}")
    if ch.personality:
        section.append(f"性格: {ch.personality}")
    return "\n".join(section)


def _build_user_prompt(
    batch: Batch, known: Dict[str, Character], title: str,
) -> tuple[str, int]:
    """返回 (prompt, 命中详档人数);命中数让上层 logger 用,免得重复扫一遍。"""
    book_title = title or "(未提供书名)"
    batch_text = batch.render_for_prompt()
    roster_index = _render_roster_index(known)
    relevant = _scan_relevant(batch_text, known)
    profiles = (
        "\n\n".join(_render_full_profile(c) for c in relevant)
        if relevant
        else "(本批未匹配到已知人物;可能全是新角色)"
    )
    prompt = (
        f"书名: {book_title}\n"
        f"批次序号: 第 {batch.index} 批\n"
        f"批次字数: 约 {batch.char_count}\n\n"
        f"=== 全部已知人物名录(共 {len(known)} 人;仅名 + 别名,用于防重名)===\n"
        f"{roster_index}\n\n"
        f"=== 本批可能涉及的人物详档(共 {len(relevant)} 人;"
        f"按 name / 别名扫描命中)===\n{profiles}\n\n"
        f"=== 本批次正文 ===\n{batch_text}\n\n"
        f"=== 任务 ===\n请按 JSON Schema 输出 CharacterExtraction。"
    )
    return prompt, len(relevant)


def extract_for_batch(
    batch: Batch,
    known: Dict[str, Character],
    llm: LLMClient,
    *,
    title: str = "",
) -> CharacterExtraction:
    """对单个 batch 调一次 LLM,返回人物增量(不在此处合并)。"""
    user_prompt, relevant_count = _build_user_prompt(batch, known, title)
    logger.info(
        "character_extractor 第 %d 批(已知 %d 人,本批关联详档 %d 人),%s @ %s",
        batch.index, len(known), relevant_count, llm.model, llm.base_url,
    )
    delta = llm.chat_json(SYSTEM_PROMPT, user_prompt, CharacterExtraction)
    logger.info(
        "character_extractor 第 %d 批产出:%d 人(新增/更新)",
        batch.index, len(delta.new_or_updated_characters),
    )
    return delta


__all__ = ["extract_for_batch", "SYSTEM_PROMPT"]
