"""子-agent:剧情大纲段抽取。

从单批章节正文里提取"剧情大纲段"(Beat)。每段是一段有完整起承转合 + 至少一个
小高潮的连贯剧情,关联多个 Setting / Character,引用本 batch 编号。

input:
- 本批章节正文
- **前 N 段**剧情大纲(默认 10 段,用于节奏接续)
- 已知人物名单(用作 character_refs 的引用域)
- 已知场景名单(用作 setting_refs 的引用域)

合并语义见 ``BatchState.merge_beats``:新段追加(自动赋全局 ``index`` + 把当前
batch 加进 ``related_batches``);延续段把当前 batch 加进对应段的 ``related_batches``。
"""

from __future__ import annotations

import logging
from typing import List

from llm.client import LLMClient
from schema.novel_analysis import Batch, BatchState, Beat, BeatExtraction

logger = logging.getLogger(__name__)


CONTEXT_WINDOW = 10  # 给 LLM 喂多少段最近的 beats 作为节奏接续上下文


SYSTEM_PROMPT = """\
你是中文长篇小说改编编剧,负责把小说切成一段段"剧情大纲段"(Beat)。
每段后续会被独立改编为一集 ~3 分钟的短视频(分镜 25 个左右),所以每段必须:
- 有完整的起承转合,**至少一个小高潮**(冲突 / 决断 / 反转 / 情绪爆发);
- 不是单一动作或独白;
- 不是过渡水分内容。

工作模式: 增量分析。
- 每次只看一批章节正文 + 此前最近 N 段大纲(节奏衔接用)+ 已知人物 / 已知场景;
- 输出**本批的剧情段 delta**:可能是新起几段,也可能是延续已有段。

────────────────────────────────────────────────
你有两条输出通道(都可空,也可同时用):

【1】new_beats: 本批**新起**的剧情段(典型情况)
每个 Beat 字段:
- title: 简短剧情标题,如"撕婚约"、"测验失败"、"初会药老"(不含地点修饰)
- summary: **本段剧情摘要,2-4 句**,要交代起承转合 + 关键节奏点(小高潮)
    例:"萧炎到萧家祠堂迎接前来退婚的纳兰嫣然。纳兰嫣然冷漠陈述退婚理由,
    萧炎反讽对方功利;两人当众撕毁婚约,萧炎冲出祠堂立誓三年内击败纳兰一族。"
- setting_refs: **本段涉及的场景**(Setting.name 列表),按时序排列,可多个
    必须用 [已知场景名单] 中已存在的 name(原样复制),不要造新名
- character_refs: **本段涉及的人物**(Character.name 列表)
    所有出场人物都要列(不只是有台词的);必须用 [已知人物名单] 里已有的 name

【2】extended_beat_indices: 本批**延续**已有段
- 如果你判断本批正文是上一段(或更早)beat 的延续(剧情还没收尾),
  **不要新建** Beat,而是把对应 Beat 的 index 放进 extended_beat_indices。
- 系统会自动把当前 batch 追加到该 Beat 的 related_batches。
- 典型场景:跨批的长冲突 / 大战 / 长心理活动,光是一批装不下整段戏。
- 一批可以同时延续 1 个段 + 新起若干段(常见:批头是上段尾声,批中后是新段)。

注意:Beat 不要存台词原文。下游 storyboarder 会按 related_batches 回查原文,
直接从原文里挑选 / 改编台词,你这里只需要给出剧情骨架 + 引用关系。

────────────────────────────────────────────────
切分规则(关键!):
- 一批 8K 字一般产 1-3 段(看剧情密度);**冲突连续的段不要硬拆**
- 不重要的过渡内容**不要单独成段**(可以并入相邻段或省略)
- 不要造段:本批没有戏剧张力的部分,宁可不输出 Beat,也不要凑数
- 参考 [此前最近 N 段大纲] 的节奏,与之**自然衔接**
- 避免连续两段节奏雷同(都是打斗 / 都是对话)

────────────────────────────────────────────────
严格按 JSON Schema 输出 BeatExtraction,只输出一个 JSON 对象。
"""


def _render_recent_beats(beats: List[Beat], window: int) -> str:
    if not beats:
        return "(暂无, 这是第 1 批)"
    recent = beats[-window:] if window > 0 else beats
    parts: List[str] = []
    for b in recent:
        batches_str = ",".join(str(x) for x in b.related_batches) or "?"
        line = f"段 {b.index} (batch {batches_str}) · {b.title}"
        if b.summary:
            line += f"\n  {b.summary}"
        parts.append(line)
    if len(beats) > len(recent):
        parts.insert(0, f"(此前共 {len(beats)} 段,只展示最近 {len(recent)} 段)")
    return "\n".join(parts)


def _condense_known_characters(state: BatchState) -> str:
    if not state.characters:
        return "(暂无)"
    return "\n".join(f"- {name}" for name in state.characters.keys())


def _condense_known_settings(state: BatchState) -> str:
    if not state.settings:
        return "(暂无)"
    return "\n".join(f"- {name}" for name in state.settings.keys())


def _build_user_prompt(state: BatchState, batch: Batch) -> str:
    recent = _render_recent_beats(state.beats, CONTEXT_WINDOW)
    chars = _condense_known_characters(state)
    settings = _condense_known_settings(state)
    book_title = state.title or "(未提供书名)"
    return (
        f"书名: {book_title}\n"
        f"批次序号: 第 {batch.index} 批\n"
        f"批次字数: 约 {batch.char_count}\n\n"
        f"=== 此前最近 {CONTEXT_WINDOW} 段剧情大纲(节奏接续) ===\n{recent}\n\n"
        f"=== 已知人物名单(character_refs 必须用这里的 name)===\n{chars}\n\n"
        f"=== 已知场景名单(setting_refs 必须用这里的 name)===\n{settings}\n\n"
        f"=== 本批次正文 ===\n{batch.render_for_prompt()}\n\n"
        f"=== 任务 ===\n请按 JSON Schema 输出 BeatExtraction。"
    )


def extract_beats(state: BatchState, batch: Batch, llm: LLMClient) -> BeatExtraction:
    """对单个 batch 调一次 LLM,返回剧情段增量(不在此处合并)。"""
    user_prompt = _build_user_prompt(state, batch)
    logger.info(
        "beat_extractor 第 %d 批(此前 %d 段),%s @ %s",
        batch.index, len(state.beats), llm.model, llm.base_url,
    )
    delta = llm.chat_json(SYSTEM_PROMPT, user_prompt, BeatExtraction)
    logger.info("beat_extractor 第 %d 批产出:%d 段", batch.index, len(delta.new_beats))
    return delta


__all__ = ["extract_beats", "SYSTEM_PROMPT", "CONTEXT_WINDOW"]
