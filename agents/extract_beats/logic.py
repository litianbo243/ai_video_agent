"""单批剧情段抽取:一次 LLM 调用,返回本批增量。

合并(新段赋 index、延续段追加 related_batches)放在 ``workflows/beat_analysis.py``。

prompt 上下文:

* 人物:Tier 1 全员名录(name + 别名) + Tier 2 本批相关详档(`character_refs` 必须取自此名录)
* 场景:**仅 name 集合**——本工程不维护场景视觉档案(那是 storyboarder 用原文 + LLM
  常识写到每镜 description 里的事)。这里给 beat agent 看历史 name 集只是为了让它
  对同一物理地点**沿用同一个 name**(避免"萧家大厅"和"萧家议事厅"指同一处)。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Set

from llm.client import LLMClient
from skills.batch_chapters import Batch
from agents.extract_beats.schema import Beat, BeatExtraction
from agents.extract_characters.schema import Character

logger = logging.getLogger(__name__)


DEFAULT_CONTEXT_WINDOW = 10  # ``RunConfig.recent_beats_window`` 的兜底值

# 向后兼容别名:旧代码 / 测试可能 import 这个名字。新代码请从 RunConfig 取。
CONTEXT_WINDOW = DEFAULT_CONTEXT_WINDOW


SYSTEM_PROMPT_TEMPLATE = """\
你是中文小说改编编剧。把小说切成 Beat(剧情段),每段对应一集 ~{target_minutes} 分钟视频({target_duration} 秒)。

每段要求:
- 有戏剧分量:冲突 / 决断 / 反转 / 情绪爆发 / 重要发现
- 不是单一动作、独白、过渡水分
- 粒度撑得起 ~{target_minutes} 分钟:太碎 → 合并,太重 → 拆分

输出 BeatExtraction(JSON),三个字段:

1. new_beats: 本批新起的段(0-3 段)
   - title: 2-6 字事件标题,不含地点。例:撕婚约
   - summary: 1-2 句,讲清「谁在哪 / 做了什么 / 转折是什么」。不抄原文台词
   - setting_refs: 涉及场景 name。已用过的沿用不改字;新地点用「宅院/机构+房间」组合(如「萧家大厅」),禁止裸名(「大厅」「广场」)
   - character_refs: 所有出场人物 name,必须出自人物名录

2. continues_open_beat (bool): prompt 里若有标 [待续] 的段,本批是否在续写它?
   - 没 [待续] 段 → 一律 false
   - false 时禁止在 new_beats 里复制 [待续] 段的内容

3. last_beat_open (bool): new_beats 末尾段是否戛然而止还没结束?
   - 戛然而止(打斗未分胜负 / 长对话未决断)→ true,下一批续写
   - 自然收束(胜败已定 / 散场)→ false
   - new_beats 为空 → false

[待续] 段决策矩阵(prompt 里有 [待续] 段时,4 选 1):
- 接着写 + 再开新段 → continues=true, new_beats=[新段]
- 只接着写         → continues=true, new_beats=[]
- 不接,只新写     → continues=false, new_beats=[新段]
- 啥也没干         → continues=false, new_beats=[]

切分规则:
- 一批 8K 字典型 1-3 段;没张力宁可空输出
- 避免连续两段节奏雷同(都打斗 / 都对话)

只输出 JSON。
"""


def _build_system_prompt(target_duration_sec: int) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        target_duration=target_duration_sec,
        target_minutes=round(target_duration_sec / 60, 1)
        if target_duration_sec % 60
        else target_duration_sec // 60,
    )


# 默认 prompt(180s = 3 分钟),供单元测试 / notebook 直接看 prompt 文本时用。
# 真实运行链路一律走 ``_build_system_prompt(target_duration_sec)``。
SYSTEM_PROMPT = _build_system_prompt(180)


def _render_recent_beats(beats: List[Beat], window: int) -> str:
    if not beats:
        return "(暂无, 这是第 1 批)"
    recent = beats[-window:] if window > 0 else beats
    parts: List[str] = []
    for b in recent:
        batches_str = ",".join(str(x) for x in b.related_batches) or "?"
        # is_open 段加 [待续] 标记,prompt 里的"决策矩阵"会引导 LLM 处理
        flag = " **[待续:上一批末尾未完成,本批承接的话置 continues_open_beat=true]**" if b.is_open else ""
        line = f"段 {b.index} (batch {batches_str}) · {b.title}{flag}"
        if b.summary:
            line += f"\n  {b.summary}"
        parts.append(line)
    if len(beats) > len(recent):
        parts.insert(0, f"(此前共 {len(beats)} 段,只展示最近 {len(recent)} 段)")
    return "\n".join(parts)


def _render_name_index(names: List[str]) -> str:
    if not names:
        return "(暂无)"
    return "\n".join(f"- {n}" for n in names)


def _scan_relevant_chars(text: str, chars: Dict[str, Character]) -> List[Character]:
    out: List[Character] = []
    for ch in chars.values():
        keys = [ch.name, *ch.aliases]
        if any(k and len(k) >= 2 and k in text for k in keys):
            out.append(ch)
    return out


def _render_char_profile(ch: Character) -> str:
    section = [f"### {ch.name}"]
    if ch.aliases:
        section.append(f"别名: {', '.join(ch.aliases)}")
    if ch.personality:
        section.append(f"性格: {ch.personality}")
    return "\n".join(section)


def _build_user_prompt(
    batch: Batch,
    beats_so_far: List[Beat],
    characters: Dict[str, Character],
    setting_names: Set[str],
    title: str,
    *,
    context_window: int,
) -> tuple[str, int]:
    """返回 (prompt, 命中人物数)。"""
    book_title = title or "(未提供书名)"
    batch_text = batch.render_for_prompt()
    recent = _render_recent_beats(beats_so_far, context_window)
    char_index = _render_name_index(list(characters.keys()))
    setting_index = _render_name_index(sorted(setting_names))

    relevant_chars = _scan_relevant_chars(batch_text, characters)
    char_profiles = (
        "\n\n".join(_render_char_profile(c) for c in relevant_chars)
        if relevant_chars
        else "(本批未匹配到已知人物)"
    )

    prompt = (
        f"书名: {book_title} | 第 {batch.index} 批\n\n"
        f"=== 此前最近 {context_window} 段大纲 ===\n{recent}\n\n"
        f"=== 人物名录({len(characters)} 人,character_refs 必须从这里取)===\n"
        f"{char_index}\n\n"
        f"=== 已用过的场景 name({len(setting_names)} 处,同一地点沿用)===\n"
        f"{setting_index}\n\n"
        f"=== 本批相关人物详档({len(relevant_chars)} 人,辅助理解)===\n"
        f"{char_profiles}\n\n"
        f"=== 本批正文 ===\n{batch_text}\n\n"
        f"输出 BeatExtraction JSON。"
    )
    return prompt, len(relevant_chars)


def extract_for_batch(
    batch: Batch,
    beats_so_far: List[Beat],
    characters: Dict[str, Character],
    setting_names: Set[str],
    llm: LLMClient,
    *,
    title: str = "",
    target_duration_sec: int = 180,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
) -> BeatExtraction:
    """对单个 batch 调一次 LLM,返回剧情段增量。

    ``setting_names`` 是历史 batch 已经用过的场景 name 集合,只作为"沿用提示"
    传给 LLM。本工程不维护场景视觉档案——视觉环境由 storyboarder 从原文 + LLM
    常识写到每镜的 description 里。

    ``target_duration_sec`` 来自 ``RunConfig.target_episode_duration_sec``,
    用于让 LLM 把 beat 粒度跟单集时长对齐(切粒度太碎 / 太重都会反作用于
    storyboarder 出镜数 / 单镜时长)。

    ``context_window`` 来自 ``RunConfig.recent_beats_window``,控制 prompt 里
    展示「此前最近 N 段大纲」的窗口大小。小模型 ctx 紧时调低,大模型可调高。
    """
    system = _build_system_prompt(target_duration_sec)
    user_prompt, rel_chars = _build_user_prompt(
        batch, beats_so_far, characters, setting_names, title,
        context_window=context_window,
    )
    logger.info(
        "beat_extractor 第 %d 批(此前 %d 段 / 窗口 %d;本批关联 %d 人;"
        "场景 name 池 %d;目标集时长 %ds),%s @ %s",
        batch.index, len(beats_so_far), context_window, rel_chars,
        len(setting_names), target_duration_sec, llm.model, llm.base_url,
    )
    delta = llm.chat_json(system, user_prompt, BeatExtraction)
    logger.info("beat_extractor 第 %d 批产出:%d 段", batch.index, len(delta.new_beats))
    return delta


__all__ = [
    "extract_for_batch",
    "SYSTEM_PROMPT",
    "SYSTEM_PROMPT_TEMPLATE",
    "CONTEXT_WINDOW",
    "DEFAULT_CONTEXT_WINDOW",
]
