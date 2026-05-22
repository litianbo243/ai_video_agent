"""单段 Beat → 单集分镜:一次 LLM 调用。

回查 Beat.related_batches 对应原文,把人物档案 + 原文都拼进 prompt。
**场景视觉环境从原文 + LLM 常识里写到每镜 ``description``**——本工程不维护
独立的场景视觉档案。``setting`` 字段只是字符串 label,取自 ``Beat.setting_refs``。

LLM 配置在同目录 ``llm.json``,本模块顶部直接 lazy build。
测试时想 mock:``from agents.extract_storyboard import set_llm; set_llm(fake)``。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

from llm.agent_llm import make_agent_llm_manager
from skills.batch_chapters import Batch
from agents.extract_beats.schema import Beat
from agents.extract_characters.schema import Character
from agents.extract_storyboard.schema import Episode, Storyboard, StoryboardList

logger = logging.getLogger(__name__)


# Per-agent LLM 管理(配置在同目录 ``llm.json``,首次 ``get_llm()`` 才 build)。
# 公开三件套:``get_llm`` / ``set_llm``(测试 mock) / ``set_trace_dir``(runner 注入)。
get_llm, set_llm, set_trace_dir = make_agent_llm_manager(
    agent_name="extract_storyboard",
    config_path=Path(__file__).parent / "llm.json",
)


SYSTEM_PROMPT = """\
你是一名优秀的中文短视频分镜导演,负责把小说的一段剧情改编成一集分镜。

输入包括:
- 一段剧情大纲(Beat:title + summary + setting_refs + character_refs)
- 关联人物档案(外貌 + 性格),用于跨集视觉一致性
- 这段剧情对应的**原文**(整批章节,几千字)
- 可能给到上下集 beat 摘要 + 上集末尾分镜(用于衔接)

────────────────────────────────────────────────
你的任务: 先定本集**导演意图**,再编排分镜清单(storyboards)。

整集时长合计 ≈ {target_duration} 秒。**短视频节奏**:单镜典型 2-8 秒,
情绪 / 信息密集的关键镜可到 10 秒,**绝不超 12 秒**。长情绪 / 长对白要
拆成多镜(换景别 / 切反应镜 / 切局部特写),不要塞到一个长镜里。

────────────────────────────────────────────────
【先定调,再分镜】

**集层** — 填 storyboards 之前,先在 director_intent 字段写 2-4 句话定本集导演意图:
- **基调**:一句话定调(压抑 / 暧昧 / 诡谲 / 凌厉 / 紧张 / 温暖 / 阴郁 / 怀旧 ...)
- **重点**:本集要让观众**感受到**或**记住**的核心(情绪 / 关系 / 冲突 / 反转点)
- **节奏**:整集呼吸感(平稳铺陈 / 张力累积 / 短促爆发 / 留白收尾 / 反转 ...)
- **视觉锤**:1-2 个标志性镜头 / 意象 / 道具(贯穿用,或集末收束)

**镜层** — 每个 storyboard 在 intent 字段写一句话(≤20 字)说明本镜的具体意图,
例如:「试探的压迫感」「反应镜,破防」「视觉锤,定调」「节奏喘息」「钩子,暗示反转」。
intent 决定了本镜的 shot_type / camera_motion / description 怎么选 —
**先想清楚意图再写画面**,不要把 intent 写成 description 换皮(「白洁哭泣」式描述无效)。

整集所有 intent 串起来应该能讲出一条情绪曲线,服务于 director_intent 的总纲。
不要写「就事论事」的客观记录式分镜。

────────────────────────────────────────────────
【改编思路】

镜头编排:
- 用画面讲故事 — 每一镜是一个可被看见的画面
- 集首 1-2 镜定场(谁、在哪、状态);身份 / 背景 / 关系靠画面 +
  服装 + 神态 + 环境自然交代,不要靠旁白念身份介绍
- 原文里水分 / 与本集主线无关的支线可以略过
- 同一 setting 多镜之间保持视觉一致(色调 / 布局 / 整体氛围)

集间衔接(看输入里上下集 beat / 上集末镜):
- 有上一集 → 集首画面跟上集末镜的空间 / 姿态 / 道具自然延续
- 有下一集 → 集末留 1 个钩子镜头(画面 / 表情 / 物件)暗示但不剧透
- 无上一集 → 直接定场亮相;无下一集 → 末镜留一个有余韵的画面收尾
- 上下集 beat / 上集分镜**只读不抄**,不重演已演过的剧情

────────────────────────────────────────────────
【各字段填写】

视觉块(description / shot_action / camera_motion):

把「画面静态」「动作演变」「镜头运动」三件事拆开填,**互不重复**。

- description:本镜**起始那一刻**的画面 + 整体氛围 — 谁在做什么 + 环境 + 光线 +
  构图视角。这字段同时给图像生成模型当 prompt,也给视频生成模型当起点画面
- shot_action:这一镜**内**画面如何变化(动词为主)。例如「缓缓抬头,泪痕未干,
  目光转向窗外」「双手攥紧,指节发白」。如果是静态镜头(对话戏 / 表情特写不动 /
  纯定场)→ **留空**,不要强行编动作
- camera_motion:运镜方式 — 推 / 拉 / 摇 / 移 / 升 / 降 / 跟 / 环绕 / 固定
  (留空 = 固定机位)

公共原则:
- 环境从原文细节 + Beat.setting_refs 的 name 含义 + 常识组合(本工程不另存场景档案)
- 出场人物用**正式 name**(从角色档案取,不用别名);外貌参考人物档案
- 不要把 dialogue 文字塞进画面描述

dialogue(本镜人物开口的台词,TTS 用):
- 原文里人物开口的话(对白 / 群众喊话 / 播报 / 念咒等)优先选取并改编
- 太长就浓缩或拆到下一镜;同镜多人开口时挑张力最强的一句
- 只放台词本身,不带「角色名:」前缀
- 单字感叹(嗯/啊/哎)进 description 写成「轻哼一声」,不进 dialogue

voiceover(角色内心独白,≤35 字):
- **默认留空**。仅当原文中角色心理活动非常需要靠台词表达出来,才填该角色的内心独白
- 同镜与 dialogue 互斥(有 dialogue 时 voiceover 留空)

speaker(本镜发声者,TTS 用,可不在 characters 列表):
- dialogue 说话人 / voiceover 内心独白 → 角色 name(档案内用正式 name,路人用泛称)
- 双空 → 留空

characters:本镜画面里可辨认的出场人物 name(从角色档案取,不用别名)

setting:**1:1 取自 Beat.setting_refs 里某个 name**(单一地点);不自创新地点

shot_type:6 选 1 — 远景 / 全景 / 中景 / 近景 / 特写 / 大特写

duration_sec(秒):
- 单镜典型 2-8s,关键情绪镜可到 10s,**绝不超 12s**
- = max(画面节奏需求, 念完 dialogue + voiceover 所需时间)
- 估算:对白 ~3.5 字/秒,旁白 ~3 字/秒
- 长台词 / 长情绪一定拆多镜,别堆一镜超时

────────────────────────────────────────────────
【硬约束】
- characters 必须是角色档案里出现过的 name
- setting 必须是 Beat.setting_refs 里出现过的 name
- 一集所有镜头时长之和 ≈ {target_duration} 秒
- 严格按 JSON Schema 输出
"""


def _gather_inputs(
    beat: Beat,
    characters: Dict[str, Character],
    batches: Dict[int, Batch],
) -> tuple[List[Character], str]:
    """按 Beat 检索关联的 Character 详档 + 原文章节(场景不再有详档)。"""
    relevant_chars: List[Character] = []
    for ref in beat.character_refs:
        ch = characters.get(ref)
        if ch is None:
            logger.warning(
                "第 %d 段 Beat 引用的 Character 不存在:%r",
                beat.index, ref,
            )
            continue
        relevant_chars.append(ch)

    parts: List[str] = []
    for bi in beat.related_batches:
        batch = batches.get(bi)
        if batch is None:
            logger.warning("第 %d 段 Beat 关联的 batch %d 未找到原文", beat.index, bi)
            continue
        parts.append(batch.render_for_prompt())
    raw_text = "\n\n".join(parts)

    return relevant_chars, raw_text


def _render_characters_for_prompt(chars: List[Character]) -> str:
    if not chars:
        return "(无)"
    parts: List[str] = []
    for ch in chars:
        section = [f"### {ch.name}"]
        if ch.aliases:
            section.append(f"别名: {', '.join(ch.aliases)}")
        if ch.appearance:
            section.append(f"外貌: {ch.appearance}")
        if ch.background:
            section.append(f"背景: {ch.background}")
        if ch.personality:
            section.append(f"性格: {ch.personality}")
        parts.append("\n".join(section))
    return "\n\n".join(parts)


def _render_beat_for_prompt(beat: Beat) -> str:
    return "\n".join([
        f"index: {beat.index}",
        f"title: {beat.title}",
        f"summary: {beat.summary}",
        f"setting_refs: {beat.setting_refs}",
        f"character_refs: {beat.character_refs}",
    ])


def _render_timeline(
    beat: Beat,
    prev_beat: Beat | None,
    next_beat: Beat | None,
) -> str:
    """渲染上下文剧情时间线:前一集 / 本集 / 下一集。"""
    lines: List[str] = []
    if prev_beat is None:
        lines.append("上一集: (本集为全书开场)")
    else:
        lines.append(f"上一集 [{prev_beat.index}]: {prev_beat.title} — {prev_beat.summary}")
    lines.append(f"本  集 [{beat.index}]: {beat.title} — {beat.summary}   ★ 正在分镜")
    if next_beat is None:
        lines.append("下一集: (本集为全书收尾)")
    else:
        lines.append(f"下一集 [{next_beat.index}]: {next_beat.title} — {next_beat.summary}")
    return "\n".join(lines)


def _render_prev_tail(storyboards: List[Storyboard]) -> str:
    """紧凑渲染上集末尾 K 镜,只保留承接相关字段(画面 / 场景 / 台词 / 旁白)。"""
    if not storyboards:
        return "(首集,无上集画面)"
    lines: List[str] = []
    for sb in storyboards:
        head = f"- 镜 {sb.index} [{sb.shot_type}]"
        if sb.setting:
            head += f" @ {sb.setting}"
        lines.append(f"{head}: {sb.description}")
        if sb.dialogue:
            speaker = sb.speaker or "?"
            lines.append(f"  台词[{speaker}]: {sb.dialogue}")
        if sb.voiceover:
            lines.append(f"  旁白: {sb.voiceover}")
    return "\n".join(lines)


def _build_user_prompt(
    beat: Beat,
    chars: List[Character],
    raw_text: str,
    target_duration: int,
    *,
    prev_beat: Beat | None,
    next_beat: Beat | None,
    prev_tail_storyboards: List[Storyboard],
) -> str:
    return (
        f"=== 上下文 · 剧情时间线 ===\n"
        f"{_render_timeline(beat, prev_beat, next_beat)}\n\n"
        f"=== 上集末尾画面(承接参考,共 {len(prev_tail_storyboards)} 镜)===\n"
        f"{_render_prev_tail(prev_tail_storyboards)}\n\n"
        f"=== 本段剧情大纲(Beat)===\n{_render_beat_for_prompt(beat)}\n\n"
        f"=== 关联人物档案 ===\n{_render_characters_for_prompt(chars)}\n\n"
        f"=== 本段对应的原文(batch {beat.related_batches})===\n"
        f"{raw_text or '(未找到原文)'}\n\n"
        f"=== 任务 ===\n"
        f"按 JSON Schema 输出本集所有 storyboards。目标总时长 ≈ {target_duration} 秒。\n"
        f"输出格式:`{{\"storyboards\": [...]}}`"
    )


def _sanitize_storyboards(
    storyboards: List[Storyboard],
    *,
    roster_chars: set,
    beat_setting_refs: List[str],
    beat_index: int,
) -> int:
    """对每镜的 ``characters`` / ``setting`` 做白名单过滤,返回被改动的镜头数。

    * ``characters``: 必须在 character roster(去重 + 丢弃非法 + warn)
    * ``setting``: 必须在 ``beat_setting_refs`` 里;不在则降级为第一个 ref + warn
    """
    fallback_setting = beat_setting_refs[0] if beat_setting_refs else ""
    valid_settings = set(beat_setting_refs)
    touched = 0
    for sb in storyboards:
        clean_chars: List[str] = []
        invalid_chars: List[str] = []
        seen: set = set()
        for name in sb.characters:
            if name in seen:
                continue
            seen.add(name)
            if name in roster_chars:
                clean_chars.append(name)
            else:
                invalid_chars.append(name)
        if invalid_chars:
            logger.warning(
                "[storyboard beat=%d shot=%d] 丢弃未注册 characters=%s",
                beat_index, sb.index, invalid_chars,
            )
            sb.characters = clean_chars
            touched += 1
        if sb.setting and sb.setting not in valid_settings:
            logger.warning(
                "[storyboard beat=%d shot=%d] setting=%r 不在 beat.setting_refs=%s,"
                "降级为 %r",
                beat_index, sb.index, sb.setting, beat_setting_refs, fallback_setting,
            )
            sb.setting = fallback_setting
            touched += 1
    return touched


DEFAULT_PREV_TAIL_K = 3
"""默认从上集末尾取多少镜作为本集承接参考。"""


def storyboard_beat(
    beat: Beat,
    characters: Dict[str, Character],
    batches: Dict[int, Batch],
    *,
    prev_beat: Beat | None = None,
    next_beat: Beat | None = None,
    prev_tail_storyboards: List[Storyboard] | None = None,
    target_duration_sec: int = 180,
) -> Episode:
    """把一段 Beat 展开为一集 Episode(含 storyboards)。

    ``prev_beat`` / ``next_beat``:相邻集 beat summary,LLM 用来做承接 / 钩子决策;
    全书首集 / 末集时传 ``None``,prompt 会显式标注。
    ``prev_tail_storyboards``:上集末尾几镜的 Storyboard 列表(典型 K=3),给 LLM
    看具体画面细节(setting / 姿态 / 服装 / 道具)做承接镜头;首集传 ``None`` / ``[]``。

    LLM 客户端由 ``agents.extract_storyboard.llm.get_llm()`` lazy 提供,
    caller 不需要传。
    """
    llm = get_llm()
    chars_list, raw_text = _gather_inputs(beat, characters, batches)
    prev_tail = prev_tail_storyboards or []

    system = SYSTEM_PROMPT.format(target_duration=target_duration_sec)
    user = _build_user_prompt(
        beat, chars_list, raw_text, target_duration_sec,
        prev_beat=prev_beat, next_beat=next_beat,
        prev_tail_storyboards=prev_tail,
    )

    logger.info(
        "storyboarder 第 %d 段(%s):人物=%d 场景 ref=%d 原文=%d 字;"
        "上下文 prev=%s next=%s prev_tail=%d 镜",
        beat.index, beat.title, len(chars_list),
        len(beat.setting_refs), len(raw_text),
        prev_beat.index if prev_beat else "首集",
        next_beat.index if next_beat else "末集",
        len(prev_tail),
    )

    sb_pack = llm.chat_json(system, user, StoryboardList)

    touched = _sanitize_storyboards(
        sb_pack.storyboards,
        roster_chars=set(characters.keys()),
        beat_setting_refs=beat.setting_refs,
        beat_index=beat.index,
    )

    logger.info(
        "第 %d 段产出:%d 镜%s",
        beat.index, len(sb_pack.storyboards),
        f"(其中 {touched} 镜被白名单过滤修正)" if touched else "",
    )

    return Episode(
        index=beat.index,
        title=f"第 {beat.index} 集 · {beat.title}",
        synopsis=beat.summary,
        director_intent=sb_pack.director_intent,
        storyboards=sb_pack.storyboards,
    )


__all__ = ["storyboard_beat", "SYSTEM_PROMPT", "DEFAULT_PREV_TAIL_K"]
