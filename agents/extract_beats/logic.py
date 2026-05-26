"""单批剧情段增量处理:LLM 抽取 + delta 合并(全部 in-place)。

主 API:

* ``extract_for_batch(batch, beats, characters, ...)``
    1 次 LLM 调用 + 自动合并
* ``merge_delta(beats, delta, ...)`` —— 单独跑合并(notebook / 单测用)

LLM 客户端从同包 ``llm.json`` 按需 lazy 取,caller 不传。
测试 mock:``from agents.extract_beats import set_llm; set_llm(fake)``。

**跨批续写机制**:每批 LLM 必须把「最近 ``rewrite_window`` 段大纲」的最新版
(原样复述或修订过的版本)放在 ``BeatExtraction.beats`` 列表的**前 K 项**,
其后才是本批新起的段。merge_delta 用前 K 项 in-place 改写 ``beats_so_far`` 的
末尾 K 段(index 沿用,related_batches 追加本批),其后追加新段。

* K=0 → 纯增量,LLM 不修改历史;
* K=1(默认)→ LLM 每批重看上批末段,可改可原样;
* K>=2 → LLM 修订空间更大,token 成本随 K 线性增长。

**场景**:本工程不维护场景视觉档案,``Beat.setting_refs`` 只是字符串 label,
storyboard agent 写每镜 ``description`` 时把视觉环境写到画面里。beat agent 内部
**不维护**跨 batch 的「场景 name 一致性池」(各 beat 独立产 setting_refs,
跨 beat 同地点可能写成不同 name,storyboard 不在意因为它只看单 beat)。

prompt 上下文:全员人物详档(name + 别名 + background + personality + arc);
``character_refs`` 必须从详档取 name(``merge_delta`` 做白名单 + alias 规范化)。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

from llm.agent_llm import make_agent_llm_manager
from skills.batch_chapters import Batch
from schemas.beat import Beat, BeatExtraction
from schemas.character import Character

logger = logging.getLogger(__name__)


# Per-agent LLM 管理(配置在同目录 ``llm.json``,首次 ``get_llm()`` 才 build)。
# 公开三件套:``get_llm`` / ``set_llm``(测试 mock) / ``set_trace_dir``(runner 注入)。
get_llm, set_llm, set_trace_dir = make_agent_llm_manager(
    agent_name="extract_beats",
    config_path=Path(__file__).parent / "llm.json",
)


def _normalize_character_refs(
    refs: List[str],
    characters: Dict[str, Character],
    *,
    batch_index: int,
    beat_title: str = "",
) -> List[str]:
    """对 LLM 写出的 character_refs 做白名单过滤 + alias → canonical name 规范化。

    白名单 = ``characters`` 里所有 name ∪ 所有 aliases。
    LLM 在 beat 里常用原文里更生动的别名(如「老七」),需要把它们映射回详档里的
    canonical name(如「陈德志」)再存进 ``Beat.character_refs``,保证下游(storyboard
    等)只跟 canonical name 打交道。

    * 命中 name        → 原样保留
    * 命中某 alias     → 替换为对应人物的 canonical name(debug log,不算 warn)
    * 都不命中         → 丢弃 + warn

    按规范化后的 name 去重、保留顺序。
    """
    alias_to_name: Dict[str, str] = {}
    for c in characters.values():
        for a in c.aliases:
            alias_to_name.setdefault(a, c.name)

    valid: List[str] = []
    invalid: List[str] = []
    aliased: List[str] = []
    seen: Set[str] = set()
    for r in refs:
        if r in characters:
            canonical = r
        elif r in alias_to_name:
            canonical = alias_to_name[r]
            aliased.append(f"{r}→{canonical}")
        else:
            invalid.append(r)
            continue
        if canonical in seen:
            continue
        seen.add(canonical)
        valid.append(canonical)

    title_suffix = f"({beat_title})" if beat_title else ""
    if invalid:
        logger.warning(
            "[beat_merge batch=%d] character_refs%s 出现未注册的引用 %s,已丢弃(roster=%d)",
            batch_index, title_suffix, invalid, len(characters),
        )
    if aliased:
        logger.debug(
            "[beat_merge batch=%d] character_refs%s 别名规范化:%s",
            batch_index, title_suffix, aliased,
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
DEFAULT_REWRITE_WINDOW = 1   # ``RunConfig.rewrite_window`` 的兜底值

# 向后兼容别名:旧代码 / 测试可能 import 这个名字。新代码请从 RunConfig 取。
CONTEXT_WINDOW = DEFAULT_CONTEXT_WINDOW


SYSTEM_PROMPT_TEMPLATE = """\
你是中文小说改编编剧,负责把小说切成 Beat(剧情段)。
每段对应一个**完整的戏剧单元**(情绪 / 信息 / 决断 / 冲突 / 反转的最小自然边界)。

**不需要关心目标时长 / 集长**(集层聚合由下游处理),只关心剧情连贯与边界自然。

# 切分原则

- 每段须有剧情推进(关系变化 / 信息揭示 / 决断 / 冲突 / 反转 / 情绪爆发 之类)
- 按**戏剧浓度**切段:
  · 浓度低(过渡 / 走路过场 / 无信息独白 / 同一动作反复)→ 合到相邻段,不单开
  · 浓度高(多线索交织 / 多场景多冲突)→ 拆成多段,一段一个核心事件
- **段间边界**应自然落在:换场景 + 换主导情绪 / 主线转向 / 关键信息节点
- **段内连贯**:同一事件 / 同一情绪 / 同一信息块不拆开
- 避免连续两段节奏雷同(都对话 / 都打斗)

# 字段

- title: 2-6 字事件标题,不含地点。例:「会议决裂」「揭穿身世」「夜袭仓库」
- summary: 1-2 句,讲清「谁在哪 / 做了什么 / 转折是什么」;不抄原文台词
- setting_refs: 涉及场景 name 列表
  · 用「宅院/机构 + 房间」组合,如「林家大厅」「沈宅书房」「明远集团会议室」;
    **禁止裸名**(「大厅」「广场」之类)
  · 纯心理活动留空
- character_refs: 所有出场人物 name(含无台词的),必须出自人物详档;
  **统一用详档里的 name**,不要用别名(原文里叫「老 X」「X 总」之类,详档 name 是「张 X X」就写「张 X X」)

# 增量输出规则(★ 严格遵守)

本批的 ``beats`` 列表分两部分:

1. **前 {rewrite_window} 项** 依次对应 prompt 里「最近 {rewrite_window} 段大纲」(标 [复述区] 的那几段)
   · 本批原文延续了某段(揭示新细节 / 接续未完场景)→ **重写** 那段的 summary
     覆盖完整剧情(不是追加,是改写,让 summary 一句话能讲清整段从头到尾)
   · 本批原文与该段无关 → **原样复述**(title / summary 完全照抄,字面不变)
   · 即使原样照抄也 **必须输出**,不要少出、不要合并、不要拆成两段
   · title 觉得不准可以改,但段数严格保持 {rewrite_window}

2. **其后** 是本批新起的段(数量按内容密度自行判断,无硬性上限)

第 1 批没有 [复述区](rewrite_window=0),整个 ``beats`` 列表全是新起段。

只输出 JSON。
"""


def _build_system_prompt(
    rewrite_window: int = DEFAULT_REWRITE_WINDOW,
) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(rewrite_window=rewrite_window)


# 默认 prompt(rewrite_window=1),供单元测试 / notebook 直接看 prompt 文本时用。
# 真实运行链路一律走 ``_build_system_prompt(rewrite_window)``。
SYSTEM_PROMPT = _build_system_prompt()


def _render_recent_beats(
    beats: List[Beat],
    window: int,
    rewrite_window: int,
) -> str:
    """渲染「最近 N 段大纲」,末尾 ``rewrite_window`` 段标 [复述区]。"""
    if not beats:
        return "(暂无, 这是第 1 批,rewrite_window=0)"
    recent = beats[-window:] if window > 0 else beats
    # 复述区:末尾 K 段(K 可能 > len(recent),取 min)
    rewrite_start = max(0, len(recent) - rewrite_window)
    parts: List[str] = []
    for i, b in enumerate(recent):
        batches_str = ",".join(str(x) for x in b.related_batches) or "?"
        flag = " **[复述区:请在 beats 前几项重新输出此段(原样或修订)]**" if i >= rewrite_start else ""
        line = f"段 {b.index} (batch {batches_str}) · {b.title}{flag}"
        if b.summary:
            line += f"\n  {b.summary}"
        parts.append(line)
    if len(beats) > len(recent):
        parts.insert(0, f"(此前共 {len(beats)} 段,只展示最近 {len(recent)} 段)")
    return "\n".join(parts)


def _render_char_profile(ch: Character) -> str:
    section = [f"### {ch.name}"]
    if ch.aliases:
        section.append(f"别名: {', '.join(ch.aliases)}")
    if ch.background:
        section.append(f"背景: {ch.background}")
    if ch.personality:
        section.append(f"性格(默认态): {ch.personality}")
    if ch.arc:
        section.append(f"弧光(带事件锚点): {ch.arc}")
    return "\n".join(section)


def _build_user_prompt(
    batch: Batch,
    beats_so_far: List[Beat],
    characters: Dict[str, Character],
    title: str,
    *,
    context_window: int,
    rewrite_window: int,
) -> str:
    book_title = title or "(未提供书名)"
    batch_text = batch.render_for_prompt()
    # 第 1 批没有复述区,展示窗口仍按 context_window 取(实际为空)
    effective_rewrite = min(rewrite_window, len(beats_so_far))
    recent = _render_recent_beats(beats_so_far, context_window, effective_rewrite)

    char_profiles = (
        "\n\n".join(_render_char_profile(c) for c in characters.values())
        if characters
        else "(尚无已知人物)"
    )

    return (
        f"书名: {book_title} | 第 {batch.index} 批 | rewrite_window={effective_rewrite}\n\n"
        f"=== 此前最近 {context_window} 段大纲(末尾 {effective_rewrite} 段为 [复述区])===\n{recent}\n\n"
        f"=== 全部已知人物详档({len(characters)} 人,character_refs 必须从这里取 name)===\n"
        f"{char_profiles}\n\n"
        f"=== 本批正文 ===\n{batch_text}\n\n"
        f"输出 BeatExtraction JSON。beats 列表前 {effective_rewrite} 项 = [复述区]最新版,"
        f"其后是本批新起段。"
    )


def extract_for_batch(
    batch: Batch,
    beats_so_far: List[Beat],
    characters: Dict[str, Character],
    *,
    title: str = "",
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    rewrite_window: int = DEFAULT_REWRITE_WINDOW,
) -> BeatExtraction:
    """完整处理一批:LLM 抽取 → 合并入 ``beats_so_far``(in-place)。

    返回 LLM 原始 delta,给 trace / debug 看;正常 caller 无需读。

    * ``context_window`` —— prompt 里展示「最近 N 段大纲」的窗口;小模型 ctx 紧时调小
    * ``rewrite_window`` —— 本批 LLM 必须复述/修订的「末尾 K 段」数量。K=0 关闭
      跨批续写(纯增量);K=1 默认(LLM 每批重看上批末段);K 越大 LLM 修订空间
      越大但 token 成本越高。语义:LLM 输出 ``beats`` 列表前 K 项 = 「最近 K 段」
      的最新版(原样或修订),其后是本批新起段

    本 agent 不关心目标视频时长 —— 集层聚合由下游 ``episode_planner`` 负责,
    beat 只按戏剧浓度切自然边界。
    """
    effective_rewrite = min(rewrite_window, len(beats_so_far))
    llm = get_llm()
    system = _build_system_prompt(rewrite_window=effective_rewrite)
    user_prompt = _build_user_prompt(
        batch, beats_so_far, characters, title,
        context_window=context_window,
        rewrite_window=effective_rewrite,
    )
    logger.info(
        "beat_extractor 第 %d 批(此前 %d 段 / 窗口 %d / 复述 K=%d;已知人物 %d),"
        "%s @ %s",
        batch.index, len(beats_so_far), context_window, effective_rewrite,
        len(characters), llm.model, llm.base_url,
    )
    delta = llm.chat_json(system, user_prompt, BeatExtraction)
    logger.info("beat_extractor 第 %d 批产出:%d 段", batch.index, len(delta.beats))
    merge_delta(
        beats_so_far, delta, batch.index,
        characters=characters,
        rewrite_window=effective_rewrite,
    )
    return delta


def merge_delta(
    beats: List[Beat],
    delta: BeatExtraction,
    batch_index: int,
    *,
    characters: Dict[str, Character],
    rewrite_window: int,
) -> None:
    """把本批 LLM delta in-place 合并入 ``beats``。

    **协议**:``delta.beats`` 前 K 项依次重写 ``beats`` 末尾 K 段(复述区),其后追加新段。
    K = ``rewrite_window``(已根据 len(beats) 自动收缩,caller 保证 K ≤ len(beats))。

    **白名单 + 规范化**:``character_refs`` 在 ``characters`` 的 name ∪ aliases 内
    过滤,命中 alias 时自动改写为 canonical name(character agent 是 name 权威源);
    ``setting_refs`` 只去重不过滤(每个 beat 自己的 setting label,跨 beat 不强约束)。

    **复述区 ``related_batches``**:沿用旧段历史 ∪ 本批 ``batch_index``(去重),
    无论 LLM 改没改 summary。
    """
    # 1. LLM 输出段数 < K → warn + 自动收缩(LLM 漏复述,把它已输出的当复述区)
    K = rewrite_window
    if len(delta.beats) < K:
        logger.warning(
            "[beat_merge batch=%d] LLM 输出 %d 段 < rewrite_window=%d,自动收缩 K=%d "
            "(LLM 可能漏复述了 %d 段;保留历史末尾不变)",
            batch_index, len(delta.beats), K, len(delta.beats), K - len(delta.beats),
        )
        K = len(delta.beats)

    # 2. 复述区:LLM 前 K 项重写 beats 末尾 K 段(in-place 改写,index 沿用)
    #
    # **复述区 carry-over 语义**:LLM 重写时常顺手把已有的 setting_refs /
    # character_refs 清空(它判断 summary 已经覆盖了就清,这是 LLM 改写时
    # 字段稳定性的典型回归)。这里防御性兜底:
    #   * LLM 给非空 → 用 LLM 的(允许扩展 / 修订)
    #   * LLM 给空 + 旧值非空 → 沿用旧值 + warn(防漏写)
    #   * LLM 给空 + 旧值也空 → 留空(无信息可承继)
    # 合法"清空"场景极少,这个 trade-off 偏向"防漏写"。
    for i in range(K):
        draft = delta.beats[i]
        target_idx = len(beats) - K + i  # 对齐到 beats 末尾 K 段的第 i 个
        old = beats[target_idx]

        if draft.setting_refs:
            clean_setting_refs = _dedup_keep_order(draft.setting_refs)
        else:
            clean_setting_refs = list(old.setting_refs)
            if old.setting_refs:
                logger.warning(
                    "[beat_merge batch=%d] 段 %d(%s)复述区 setting_refs 被 LLM 清空,"
                    "carry-over 旧值 %s",
                    batch_index, old.index, draft.title, old.setting_refs,
                )

        if draft.character_refs:
            clean_char_refs = _normalize_character_refs(
                draft.character_refs, characters,
                batch_index=batch_index,
                beat_title=draft.title,
            )
        else:
            clean_char_refs = list(old.character_refs)
            if old.character_refs:
                logger.warning(
                    "[beat_merge batch=%d] 段 %d(%s)复述区 character_refs 被 LLM 清空,"
                    "carry-over 旧值 %s",
                    batch_index, old.index, draft.title, old.character_refs,
                )

        new_related = list(old.related_batches)
        if batch_index not in new_related:
            new_related.append(batch_index)
        # 覆盖 LLM 给出的字段,index / related_batches 由 server 维护
        beats[target_idx] = Beat(
            **{
                **draft.model_dump(),
                "setting_refs": clean_setting_refs,
                "character_refs": clean_char_refs,
            },
            index=old.index,
            related_batches=new_related,
        )
        if draft.title != old.title or draft.summary != old.summary:
            logger.debug(
                "[beat_merge batch=%d] 段 %d 复述区已修订:title=%r→%r",
                batch_index, old.index, old.title, draft.title,
            )

    # 3. 复述区之后是本批新起段,append
    for draft in delta.beats[K:]:
        clean_setting_refs = _dedup_keep_order(draft.setting_refs)
        clean_char_refs = _normalize_character_refs(
            draft.character_refs, characters,
            batch_index=batch_index,
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


def apply_character_renames(
    beats: List[Beat],
    renames: Iterable[Tuple[str, str]],
) -> int:
    """把已锁住的 ``beats`` 里 ``character_refs`` 的旧 name 替换成新 name(in-place)。

    场景:``extract_characters.merge_delta`` 触发 name 升格(如「老张」→「张志远」)
    后,**此前已经产出的 beats** 仍引用旧 name。本函数遍历回扫,把旧 name 替换
    成新 name,保证 ``Beat.character_refs`` 跟 character roster 的 canonical name
    长期一致。

    保序去重(若新旧 name 在同一段 character_refs 里都出现,新 name 唯一保留)。
    无升格事件(``renames`` 为空)时直接 no-op 返回 0。

    Args:
        beats: 此前已生成的 beats(``beats_so_far``);此函数 in-place 修改。
        renames: ``[(old_name, new_name), ...]``,来自 ``CharacterExtractResult.renames``。

    Returns:
        被改动 ``character_refs`` 的 beat 数量(用于日志 / 调试)。
    """
    rename_map: Dict[str, str] = {old: new for old, new in renames if old != new}
    if not rename_map or not beats:
        return 0

    touched = 0
    for b in beats:
        if not any(name in rename_map for name in b.character_refs):
            continue
        seen: Set[str] = set()
        new_refs: List[str] = []
        for name in b.character_refs:
            canonical = rename_map.get(name, name)
            if canonical in seen:
                continue
            seen.add(canonical)
            new_refs.append(canonical)
        if new_refs != b.character_refs:
            logger.debug(
                "[apply_character_renames] beat %d (%s) character_refs %s → %s",
                b.index, b.title, b.character_refs, new_refs,
            )
            b.character_refs = new_refs
            touched += 1

    if touched:
        logger.info(
            "[apply_character_renames] %d 段 beat 同步了 %d 项 name 升格 %s",
            touched, len(rename_map), list(rename_map.items()),
        )
    return touched


__all__ = [
    "apply_character_renames",
    "extract_for_batch",
    "merge_delta",
    "SYSTEM_PROMPT",
    "SYSTEM_PROMPT_TEMPLATE",
    "CONTEXT_WINDOW",
    "DEFAULT_CONTEXT_WINDOW",
    "DEFAULT_REWRITE_WINDOW",
]
