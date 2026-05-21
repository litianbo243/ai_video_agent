"""单批剧情段增量处理:LLM 抽取 + delta 合并(全部 in-place)。

主 API:

* ``extract_for_batch(batch, beats, characters, setting_names, ...)``
    —— 1 次 LLM 调用 + 自动合并(包含 [待续] 段决策矩阵)
    LLM 客户端从同包 ``llm`` 模块按需 lazy 取,caller 不再传
    返回的 ``BeatExtraction`` 仅给 trace / debug 看,正常 caller 无需读

底层 API(单测 / notebook 用):

* ``merge_delta(beats, delta, batch_index, valid_characters, setting_names)``
    —— 单独跑合并(给手工构造的 delta 用)
* 测试时想 mock LLM:``from agents.extract_beats import set_llm; set_llm(fake)``

workflow 只负责编排(batch 循环 + LangGraph 节点),不写合并语义。

prompt 上下文:

* 人物:全员详档(name + 别名 + background + personality),
  `character_refs` 必须从详档里取 name。
* 场景:**仅 name 集合**——本工程不维护场景视觉档案(由下游按场景生成)。
  这里把历史 name 集给 beat agent,只是为了让它对同一物理地点**沿用同一个 name**
  (避免"林家大厅"和"林家议事厅"指同一处)。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Set

from llm.agent_llm import make_agent_llm_manager
from skills.batch_chapters import Batch
from agents.extract_beats.schema import Beat, BeatExtraction
from agents.extract_characters.schema import Character

logger = logging.getLogger(__name__)


# Per-agent LLM 管理(配置在同目录 ``llm.json``,首次 ``get_llm()`` 才 build)。
# 公开三件套:``get_llm`` / ``set_llm``(测试 mock) / ``set_trace_dir``(runner 注入)。
get_llm, set_llm, set_trace_dir = make_agent_llm_manager(
    agent_name="extract_beats",
    config_path=Path(__file__).parent / "llm.json",
)


def _filter_refs(
    refs: List[str],
    roster: Set[str],
    *,
    batch_index: int,
    field_name: str,
    beat_title: str = "",
) -> List[str]:
    """LLM 写出的 refs 必须在 roster 里;不在的丢弃 + warn。
    保留顺序、去重(同一 ref 只算一次)。
    """
    valid: List[str] = []
    invalid: List[str] = []
    seen: Set[str] = set()
    for r in refs:
        if r in seen:
            continue
        seen.add(r)
        if r in roster:
            valid.append(r)
        else:
            invalid.append(r)
    if invalid:
        logger.warning(
            "[beat_merge batch=%d] %s%s 出现未注册的引用 %s,已丢弃(roster=%d)",
            batch_index, field_name,
            f"({beat_title})" if beat_title else "",
            invalid, len(roster),
        )
    return valid


def _dedup_keep_order(items: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for s in items:
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


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

1. new_beats: 本批新起的段(数量按内容密度决定,无硬性上限)
   - title: 2-6 字事件标题,不含地点。如「会议决裂」「揭穿身世」「夜袭仓库」
   - summary: 1-2 句,讲清「谁在哪 / 做了什么 / 转折是什么」。不抄原文台词
   - setting_refs: 涉及场景 name。已用过的沿用不改字;新地点用「宅院/机构+房间」组合(如「林家大厅」「沈宅书房」「明远集团会议室」),禁止裸名(「大厅」「广场」)
   - character_refs: 所有出场人物 name,必须出自 prompt 里的人物详档

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
- 典型密度:每 ~4-6K 字 1 段;无张力的过渡段宁可空输出,不要硬凑
- 一批字数多就多出段,字数少就少出段;别为了"看着饱满"把过渡水分拔成 beat
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


def _render_char_profile(ch: Character) -> str:
    section = [f"### {ch.name}"]
    if ch.aliases:
        section.append(f"别名: {', '.join(ch.aliases)}")
    if ch.background:
        section.append(f"背景: {ch.background}")
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
) -> str:
    book_title = title or "(未提供书名)"
    batch_text = batch.render_for_prompt()
    recent = _render_recent_beats(beats_so_far, context_window)
    setting_index = _render_name_index(sorted(setting_names))

    char_profiles = (
        "\n\n".join(_render_char_profile(c) for c in characters.values())
        if characters
        else "(尚无已知人物)"
    )

    return (
        f"书名: {book_title} | 第 {batch.index} 批\n\n"
        f"=== 此前最近 {context_window} 段大纲 ===\n{recent}\n\n"
        f"=== 全部已知人物详档({len(characters)} 人,character_refs 必须从这里取 name)===\n"
        f"{char_profiles}\n\n"
        f"=== 已用过的场景 name({len(setting_names)} 处,同一地点沿用)===\n"
        f"{setting_index}\n\n"
        f"=== 本批正文 ===\n{batch_text}\n\n"
        f"输出 BeatExtraction JSON。"
    )


def extract_for_batch(
    batch: Batch,
    beats_so_far: List[Beat],
    characters: Dict[str, Character],
    setting_names: Set[str],
    *,
    title: str = "",
    target_duration_sec: int = 180,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
) -> BeatExtraction:
    """完整处理一批:LLM 抽取 → 合并入 ``beats_so_far`` / ``setting_names``(in-place)。

    LLM 客户端由 ``agents.extract_beats.llm.get_llm()`` lazy 提供
    (配置在同包 ``llm.json``),caller 不需要传。

    返回 LLM 原始 delta,给 logger / trace / debug 看;默认调用方无需读它。
    若想跳过自动合并(notebook 调试场景),可单独调 ``merge_delta``。

    ``setting_names`` 是历史 batch 已经用过的场景 name 集合,既作为"沿用提示"
    传给 LLM,又被 merge_delta 写入新出现的 name。本工程不维护场景视觉档案
    ——视觉环境由 storyboarder 从原文 + LLM 常识写到每镜的 description 里。

    ``target_duration_sec`` 来自 ``RunConfig.target_episode_duration_sec``,
    用于让 LLM 把 beat 粒度跟单集时长对齐(切粒度太碎 / 太重都会反作用于
    storyboarder 出镜数 / 单镜时长)。

    ``context_window`` 来自 ``RunConfig.recent_beats_window``,控制 prompt 里
    展示「此前最近 N 段大纲」的窗口大小。小模型 ctx 紧时调低,大模型可调高。
    """
    llm = get_llm()
    system = _build_system_prompt(target_duration_sec)
    user_prompt = _build_user_prompt(
        batch, beats_so_far, characters, setting_names, title,
        context_window=context_window,
    )
    logger.info(
        "beat_extractor 第 %d 批(此前 %d 段 / 窗口 %d;已知人物 %d;"
        "场景 name 池 %d;目标集时长 %ds),%s @ %s",
        batch.index, len(beats_so_far), context_window, len(characters),
        len(setting_names), target_duration_sec, llm.model, llm.base_url,
    )
    delta = llm.chat_json(system, user_prompt, BeatExtraction)
    logger.info("beat_extractor 第 %d 批产出:%d 段", batch.index, len(delta.new_beats))
    merge_delta(
        beats_so_far, delta, batch.index,
        valid_characters=set(characters.keys()),
        setting_names=setting_names,
    )
    return delta


def merge_delta(
    beats: List[Beat],
    delta: BeatExtraction,
    batch_index: int,
    *,
    valid_characters: Set[str],
    setting_names: Set[str],
) -> None:
    """把本批 LLM delta in-place 合并入 ``beats``;处理 [待续] 段决策矩阵。

    * ``character_refs`` 对 ``valid_characters`` 做白名单过滤
      (character agent 是 character name 的权威源)
    * ``setting_refs`` **不做**白名单过滤——beat agent 自己是 setting name 的权威源,
      只去重 + 顺手把新 name 累加进 ``setting_names``
    * ``is_open`` 不变量:全书任意时刻**最多 1 个**段 is_open=True。

    [待续] 段(``open_idx`` 处)的处置由 ``continues_open_beat`` ×
    ``new_beats 是否非空`` 联合决定:

    ====================  ==============  ==================================
    continues_open_beat   new_beats       旧 [待续] 段去向
    ====================  ==============  ==================================
    False                 任意             step 2 显式 close
    True                  empty           保持 open(本批纯续写,还没结束)
    True                  非空             step 4 隐式 close(被新段接管)
    ====================  ==============  ==================================

    ``last_beat_open`` 只对 ``new_beats`` 末尾段生效,``new_beats`` 为空时无意义。
    """
    open_idx = next((i for i, b in enumerate(beats) if b.is_open), None)

    # 1. continues_open_beat=True → 把本 batch 加进 open 段的 related_batches
    if delta.continues_open_beat:
        if open_idx is None:
            logger.warning(
                "[beat_merge batch=%d] LLM 输出 continues_open_beat=true,"
                "但 prompt 里没 [待续] 段,忽略",
                batch_index,
            )
        else:
            ob = beats[open_idx]
            if batch_index not in ob.related_batches:
                ob.related_batches.append(batch_index)

    # 2. 上批 open 但本批 continues_open_beat=False → 强制 close
    if open_idx is not None and not delta.continues_open_beat:
        ob = beats[open_idx]
        logger.info(
            "[beat_merge batch=%d] 段 %d (%r) 上批标记 [待续],"
            "本批 continues_open_beat=false,强制收束",
            batch_index, ob.index, ob.title,
        )
        ob.is_open = False

    # 3. 处理 new_beats(append + 默认 is_open=False)
    for draft in delta.new_beats:
        clean_setting_refs = _dedup_keep_order(draft.setting_refs)
        clean_char_refs = _filter_refs(
            draft.character_refs, valid_characters,
            batch_index=batch_index, field_name="character_refs",
            beat_title=draft.title,
        )
        beat = Beat(
            **{
                **draft.model_dump(),
                "setting_refs": clean_setting_refs,
                "character_refs": clean_char_refs,
            },
            index=len(beats) + 1,
            related_batches=[batch_index],
        )
        beats.append(beat)
        setting_names.update(clean_setting_refs)

    # 4. 决定最终 is_open 状态
    if delta.new_beats:
        new_tail = beats[-1]
        for b in beats[:-1]:
            if b.is_open:
                b.is_open = False
                logger.info(
                    "[beat_merge batch=%d] 段 %d (%r) 上批 [待续],"
                    "本批续写完成(continues=true + 又新写段 %d) → 由新段接管,收束",
                    batch_index, b.index, b.title, new_tail.index,
                )
        new_tail.is_open = delta.last_beat_open
        if delta.last_beat_open:
            logger.info(
                "[beat_merge batch=%d] 段 %d (%r) 标记 [待续],等下一批续写",
                batch_index, new_tail.index, new_tail.title,
            )
    elif delta.last_beat_open:
        logger.info(
            "[beat_merge batch=%d] last_beat_open=true 但 new_beats 为空,忽略"
            "(此字段仅对 new_beats 末尾段生效)",
            batch_index,
        )


__all__ = [
    "extract_for_batch",
    "merge_delta",
    "SYSTEM_PROMPT",
    "SYSTEM_PROMPT_TEMPLATE",
    "CONTEXT_WINDOW",
    "DEFAULT_CONTEXT_WINDOW",
]
