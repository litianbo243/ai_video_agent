"""一集所含 N 段 Beat → 一集叙事分镜:一次 LLM 调用。

输入是 ``episode_planner`` 给出的 ``EpisodePlan``(含 ``beat_indices`` +
集层 ``director_intent`` 基线)+ 对应的 N 段 ``Beat``。回查所有
``Beat.related_batches`` 跨 batch 拼接原文,把人物档案 + 原文都拼进 prompt。

本 agent **只管叙事维度**(谁 / 在哪 / 说什么 / 想什么 + 集层 director_intent),
镜头视觉决策(景别 / 运镜 / 起始画面 / 时长)归下游 ``shot_director`` agent。

合并:``Shot`` = ``NarrativeShot``(本 agent) ∪ ``ShotDirection``
(下游 agent),由 workflow 调 ``merge_episode`` 按 index 配对。

LLM 配置在同目录 ``llm.json``,本模块顶部直接 lazy build。
测试时想 mock:``from agents.narrative_director import set_llm; set_llm(fake)``。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Set

from llm.agent_llm import make_agent_llm_manager
from skills.batch_chapters import Batch
from schemas.beat import Beat
from schemas.character import Character
from schemas.episode_plan import EpisodePlan
from schemas.narrative_shot import NarrativeShot, NarrativeShotList
from schemas.shot_direction import ShotDirection, ShotDirectionList
from schemas.screenplay import Episode, Shot

logger = logging.getLogger(__name__)


get_llm, set_llm, set_trace_dir = make_agent_llm_manager(
    agent_name="narrative_director",
    config_path=Path(__file__).parent / "llm.json",
)


SYSTEM_PROMPT = """\
你是一名优秀的中文短视频编剧 / 叙事分镜师,负责把一集对应的 N 段剧情(Beat)
拆成一集完整的**叙事分镜**。

你**只关心叙事**:每镜谁在哪、说什么、想什么、想让观众感受什么。
**视觉决策**(景别 / 运镜 / 起始画面 / 时长)归下游镜头导演,你不要操心。

输入:
- 本集所含的 N 段 Beat(剧情大纲;1 集 = 多段 beat 合起来)
- 集层规划基线:title / synopsis / director_intent(上游分集规划已给,你可微调)
- 关联人物档案
- N 段 Beat 对应的原文(可能跨多个 batch 拼接,长度不定,可能数千到上万字)
- 上下集 plan 摘要 + 上集末尾分镜(用于衔接)

────────────────────────────────────────────────
你的任务:先定本集**叙事调性**(director_intent),再编排叙事分镜清单(shots)。

整集时长合计 ≈ {target_duration} 秒,**全集典型 {shots_min}-{shots_max} 镜**(单镜
5-8 秒)。**短视频是高频切镜**,信息密度要切碎到多镜,不要堆一镜里讲完。

────────────────────────────────────────────────
【先定调,再分镜】

集层 director_intent(2-4 句,填 shots 前先写):
- **优先复用 plan 给的 director_intent 基线**;只有看完原文后觉得需要补充 / 微调
  时才改写。如果照搬就原样输出
- **基调**:一句话定调(压抑 / 暧昧 / 诡谲 / 凌厉 / 紧张 / 温暖 / 阴郁 / 怀旧 ...)
- **重点**:本集要让观众**感受到**或**记住**的核心(情绪 / 关系 / 冲突 / 反转点)
- **节奏**:整集呼吸感(平稳铺陈 / 张力累积 / 短促爆发 / 留白收尾 / 反转 ...)
- **视觉锤**:1-2 个标志性镜头 / 意象 / 道具(供下游导演视觉化用)

镜层 — 每镜要分别填好 **intent**(为什么)+ **description**(发生什么):

- **intent**(≤20 字):本镜的**抽象意图** — 为什么拍 / 让观众感觉什么。
  例:「试探的压迫感」「反应镜,破防」「视觉锤,定调」「节奏喘息」「钩子,暗示反转」
- **description**(1-2 句):本镜**剧情上发生了什么** — 谁做了什么 / 推进什么。
  例:「A 推门进入,语气恭敬向 B 询问来意」「B 起身关门,话锋一转抛出关键信息」
- 两者**不要混着写**:intent 是动机,description 是事件。下面是反例:
  反例:`intent="定场亮相,展示 A 兴奋来访"` ← 「A 兴奋来访」是事件,该挪到 description
  正例:`intent="定场,展现兴奋"` + `description="A 兴奋来访,推门进 B 办公室"`
- description 只写**剧情**,**不写画面 / 镜头 / 光线 / 景别**(那些归下游镜头导演)

整集所有 intent 串起来应能讲出一条情绪曲线,所有 description 串起来应能讲出
一条剧情曲线,共同服务于 director_intent。**不要写「就事论事」的客观记录式分镜**。

────────────────────────────────────────────────
【改编思路】

镜头编排:
- 用画面讲故事(虽然视觉字段不由你填,但你设计的每镜应该**能被拍**)
- 集首 1-2 镜定场(谁、在哪、状态);身份 / 背景 / 关系靠场景 + 人物互动自然交代,
  **不要靠旁白念身份介绍**
- 原文里水分 / 与本集主线无关的支线可以略过
- 避免同一节奏连续多镜(全对白 / 全描写)

**集内 beat 间过渡**(本集含多段 beat 时):
- 不要给每段 beat 单独"集首定场""集末钩子" —— **整集**才有 1 个开场 / 1 个收尾
- beat 间用 1-2 镜的过渡镜衔接(空镜 / 时间流逝 / 地点切换 / 反应特写),
  不要硬切两段不同场景
- 集内多段 beat 的情绪曲线要连贯递进,不要让两段独立的小高潮简单拼接

集间衔接(看输入里上下集 plan 摘要 / 上集末镜):
- 有上一集 → 集首跟上集末镜在故事上自然延续(地点 / 角色情绪状态接得上)
- 有下一集 → 集末留 1 个钩子镜(暗示下集冲突,不剧透)
- 无上一集 → 直接定场亮相;无下一集 → 末镜留一个有余韵的画面收尾
- 上下集 plan / 上集分镜**只读不抄**,不重演已演过的剧情

────────────────────────────────────────────────
【人物弧光的解读 —— 防剧透】

人物档案里:
- **性格(默认态)** = 角色一以贯之的基础人设(模式级)
- **弧光(带事件锚点)** = 性格 / 状态在剧情中的演变,格式为
  「在 [事件 X] 后,从 A 转为 B」「在 [事件 Y] 后,从 B 转为 C」

弧光是**全书读完后回看的视角**,不是本集的当前态。你必须**判断本集处于弧光哪段**:

1. 看人物的 arc 字段(若有),拎出里面所有事件锚点(如「被骗婚后」「母亲病故后」)
2. 对每个锚点,扫本集所含 Beat.summary,**判断该事件是否已发生**:
   - 锚点事件在本集**之前**已发生 → 该锚点的 B 状态已**生效**
   - 锚点事件**在本集中**发生 → 本集是**转折点**,前半还是 A 状态,
     转折镜后是 B 状态(画出转变本身)
   - 锚点事件**尚未发生** → 仍是 A 状态(转折前)
3. 没有 arc 的角色 → 直接用 personality 默认态

**严守:不剧透未发生的转变**
- ❌ 第 1 集是「婚礼」,arc 写「被骗婚后转为冷峻戒备」 → 画出已冷峻戒备的眼神(剧透)
- ✓ 第 1 集画新婚甜蜜 / 信赖,把"骗婚"留给后续集的转折点演出来
- ❌ background 没写"父亲去世",arc 写「父亲病故后变独立」,本集父亲尚在 →
  画出"已独立"的疏离感(剧透)
- ✓ 父亲在场时画亲密依赖,等"病故"集再演转变

**典型应用**:dialogue / voiceover / intent 都要反映"此时此刻"的状态,
不要让角色用"未来才会觉醒的语气"说话。

────────────────────────────────────────────────
【各字段填写】

intent / description:见上文「先定调,再分镜」段;intent 是抽象动机(≤20 字),
description 是具体剧情(1-2 句)。

characters:本镜可辨认的出场人物 name
- 用**正式 name**(从角色档案取,不用别名)
- 含无台词但在画面里的人(背景 / 路过 / 旁观)

setting:**1:1 取自本集所含 Beat.setting_refs 的并集里某个 name**(单一地点);不自创新地点

speaker(本镜发声者,TTS 用,可不在 characters 列表):
- dialogue 说话人 / voiceover 内心独白主人 → 角色 name
- 双空(无声镜)→ 留空

dialogue(本镜人物开口的台词,TTS 用):
- 原文里人物开口的话(对白 / 群众喊话 / 念咒等)优先选取并改编
- 太长就浓缩或拆到下一镜;同镜多人开口时挑张力最强的一句
- 只放台词本身,不带「角色名:」前缀
- 单字感叹(嗯 / 啊 / 哎)不进 dialogue

voiceover(角色内心独白,≤35 字):
- **能不用就不用,默认留空**。优先让画面 / 表演 / 对白传递情绪
- 仅当**原文心理描写非常清晰**且**表演 / 画面确实传不出**时才填(典型:决断 /
  信念崩塌 / 难以启齿的真相 / 强烈反差感)
- 同镜与 dialogue 互斥(有 dialogue 时 voiceover 留空)
- 客观陈述 / 画面描述 → 不要进 voiceover,那是 description 的事

duration_sec(秒,本镜时长):
- 典型 5-8s,情绪 / 信息密集镜可到 10s,**绝不超 12s**(超了 = 拆这镜)
- 估算:对白 ~3.5 字/秒,旁白 ~3 字/秒;长台词 / 长情绪一定拆多镜
- 全集所有 duration_sec 加起来 ≈ {target_duration} 秒

────────────────────────────────────────────────
【硬要求 —— 容易踩的坑】

1. **镜数对齐目标时长**:全集 ≈ {shots_min}-{shots_max} 镜。低于 → 你写得太挤,
   拆;高于 → 在合并相似镜。**不要写「每镜 description 长篇大论」式的超大镜**

2. **关键物件 / 物证 / 决断瞬间必须有独立镜**:剧情依赖的关键道具(照片 /
   信件 / 凶器 / 钥匙 / 药品 / 短信 / 手铐 / 武器 ...)、关键面部表情切换、
   关键决策瞬间 → 必须画面单独呈现,不要塞进 description 让观众"靠想象"。
   **特别**:你在 director_intent 里说「视觉锤是 X」,那 X 必须有 ≥1 个
   独立镜(否则就是写支票不兑现)

3. **反应镜配对**:情绪戏 = 动作镜 + 反应镜。强烈冲击 / 决定性时刻给「承受方」
   的特写反应(瞪视 / 颤抖 / 眼神涣散 / 嘴角抽动)。让情绪通过表情传递,
   不要全靠台词

4. **避免重复镜**:同 setting + 同人物的连续镜必须有变化(换景别 / 加道具 /
   换情绪 / 换姿态)。同主题 dialogue 连续 ≥2 镜要合并或换说法

5. **拆长场为多节奏镜**:一场戏(同地点持续 1+ 分钟)不要写成 3-5 个长镜,
   要写成 8-15 个短镜,中间穿插反应 / 道具 / 局部特写,节奏跳起来

────────────────────────────────────────────────
【硬约束】
- characters 必须是角色档案里出现过的 name
- setting 必须是本集所含 Beat.setting_refs 的并集里出现过的 name
- index 从 1 开始连续编号,不跳号、不重号(全集统一编号,跨 beat 也连号)
- 严格按 JSON Schema 输出
"""


def _gather_inputs_for_episode(
    member_beats: List[Beat],
    characters: Dict[str, Character],
    batches: Dict[int, Batch],
    *,
    ep_index: int,
) -> tuple[List[Character], str]:
    """汇总一集所含 N 段 Beat 的关联 Character + 跨 batch 原文。

    * characters:所有 ``member_beats.character_refs`` 并集(保持原档案中出现顺序)
    * raw_text:所有 ``member_beats.related_batches`` 并集,按 batch.index 升序
      去重拼接(同一 batch 被多 beat 引用时只渲染一次)
    """
    char_refs_seen: Set[str] = set()
    relevant_chars: List[Character] = []
    for b in member_beats:
        for ref in b.character_refs:
            if ref in char_refs_seen:
                continue
            char_refs_seen.add(ref)
            ch = characters.get(ref)
            if ch is None:
                logger.warning(
                    "[narrate ep=%d beat=%d] 引用的 Character 不存在:%r",
                    ep_index, b.index, ref,
                )
                continue
            relevant_chars.append(ch)

    batch_indices: Set[int] = set()
    for b in member_beats:
        batch_indices.update(b.related_batches)
    parts: List[str] = []
    for bi in sorted(batch_indices):
        batch = batches.get(bi)
        if batch is None:
            logger.warning(
                "[narrate ep=%d] 关联的 batch %d 未找到原文", ep_index, bi,
            )
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
            section.append(f"性格(默认态): {ch.personality}")
        if ch.arc:
            section.append(f"弧光(带事件锚点): {ch.arc}")
        parts.append("\n".join(section))
    return "\n\n".join(parts)


def _render_plan(plan: EpisodePlan) -> str:
    """渲染本集 plan(title + synopsis + director_intent + beat_indices)。"""
    return "\n".join([
        f"title: {plan.title}",
        f"synopsis: {plan.synopsis}",
        f"director_intent 基线: {plan.director_intent}",
        f"beat_indices: {plan.beat_indices}",
    ])


def _render_member_beats(beats: List[Beat]) -> str:
    """紧凑渲染本集所含 N 段 Beat(按时序)。"""
    if not beats:
        return "(无)"
    lines: List[str] = []
    for b in beats:
        lines.append(f"--- Beat 段 {b.index} · {b.title} ---")
        lines.append(f"summary: {b.summary}")
        lines.append(f"setting_refs: {b.setting_refs}")
        lines.append(f"character_refs: {b.character_refs}")
    return "\n".join(lines)


def _render_plan_timeline(
    ep_index: int,
    plan: EpisodePlan,
    prev_plan: EpisodePlan | None,
    next_plan: EpisodePlan | None,
) -> str:
    """渲染上下文集时间线:前一集 / 本集 / 下一集 plan 摘要。"""
    lines: List[str] = []
    if prev_plan is None:
        lines.append("上一集: (本集为全书开场)")
    else:
        lines.append(
            f"上一集: {prev_plan.title}(beats {prev_plan.beat_indices})"
            f" — {prev_plan.synopsis}"
        )
    lines.append(
        f"本  集 [{ep_index}]: {plan.title}(beats {plan.beat_indices})"
        f" — {plan.synopsis}   ★ 正在分镜"
    )
    if next_plan is None:
        lines.append("下一集: (本集为全书收尾)")
    else:
        lines.append(
            f"下一集: {next_plan.title}(beats {next_plan.beat_indices})"
            f" — {next_plan.synopsis}"
        )
    return "\n".join(lines)


def _render_prev_tail_narrative(shots: List[Shot]) -> str:
    """紧凑渲染上集末尾 K 镜的**叙事维度**(谁在哪 / 说了什么 / 想了什么)。

    本 agent 只关心叙事承接,所以即使 prev_tail 里有视觉字段也不渲染。
    """
    if not shots:
        return "(首集,无上集叙事)"
    lines: List[str] = []
    for sb in shots:
        head = f"- 镜 {sb.index}"
        if sb.setting:
            head += f" @ {sb.setting}"
        if sb.intent:
            head += f" · 意图:{sb.intent}"
        lines.append(head)
        if sb.dialogue:
            speaker = sb.speaker or "?"
            lines.append(f"  台词[{speaker}]: {sb.dialogue}")
        if sb.voiceover:
            speaker = sb.speaker or "?"
            lines.append(f"  内心[{speaker}]: {sb.voiceover}")
    return "\n".join(lines)


def _build_user_prompt(
    *,
    ep_index: int,
    plan: EpisodePlan,
    member_beats: List[Beat],
    chars: List[Character],
    raw_text: str,
    target_duration: int,
    prev_plan: EpisodePlan | None,
    next_plan: EpisodePlan | None,
    prev_tail_shots: List[Shot],
) -> str:
    batch_indices = sorted({bi for b in member_beats for bi in b.related_batches})
    return (
        f"=== 上下文 · 集时间线 ===\n"
        f"{_render_plan_timeline(ep_index, plan, prev_plan, next_plan)}\n\n"
        f"=== 上集末尾叙事(承接参考,共 {len(prev_tail_shots)} 镜)===\n"
        f"{_render_prev_tail_narrative(prev_tail_shots)}\n\n"
        f"=== 本集规划基线 ===\n{_render_plan(plan)}\n\n"
        f"=== 本集所含 {len(member_beats)} 段 Beat(按时序)===\n"
        f"{_render_member_beats(member_beats)}\n\n"
        f"=== 关联人物档案 ===\n{_render_characters_for_prompt(chars)}\n\n"
        f"=== 本集对应的原文(batch {batch_indices})===\n"
        f"{raw_text or '(未找到原文)'}\n\n"
        f"=== 任务 ===\n"
        f"按 JSON Schema 输出:先填 director_intent(2-4 句,优先复用 plan 基线),"
        f"再为每镜填 shots。\n"
        f"目标总时长 ≈ {target_duration} 秒(下游镜头导演会据此估单镜时长)。\n"
        f"输出格式:`{{\"director_intent\": \"...\", \"shots\": [...]}}`"
    )


def _sanitize_narrative_shots(
    shots: List[NarrativeShot],
    *,
    roster_chars: set,
    allowed_settings: List[str],
    ep_index: int,
) -> int:
    """对每镜的 ``characters`` / ``setting`` 做白名单过滤,返回被改动的镜头数。

    * ``characters``: 必须在 character roster(去重 + 丢弃非法 + warn)
    * ``setting``: 必须在 ``allowed_settings``(= 本集所含 Beat.setting_refs 并集);
      不在则降级为第一个 allowed setting + warn
    """
    fallback_setting = allowed_settings[0] if allowed_settings else ""
    valid_settings = set(allowed_settings)
    touched = 0
    for sb in shots:
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
                "[narrate ep=%d shot=%d] 丢弃未注册 characters=%s",
                ep_index, sb.index, invalid_chars,
            )
            sb.characters = clean_chars
            touched += 1
        if sb.setting and sb.setting not in valid_settings:
            logger.warning(
                "[narrate ep=%d shot=%d] setting=%r 不在本集 setting 并集=%s,"
                "降级为 %r",
                ep_index, sb.index, sb.setting, allowed_settings, fallback_setting,
            )
            sb.setting = fallback_setting
            touched += 1
    return touched


def narrate_episode(
    *,
    ep_index: int,
    plan: EpisodePlan,
    member_beats: List[Beat],
    characters: Dict[str, Character],
    batches: Dict[int, Batch],
    prev_plan: EpisodePlan | None = None,
    next_plan: EpisodePlan | None = None,
    prev_tail_shots: List[Shot] | None = None,
    target_duration_sec: int = 180,
) -> NarrativeShotList:
    """把一集所含 N 段 Beat 展开为一集**叙事分镜**(不含视觉字段)。

    Args:
        ep_index: 集序号(1-based,episode_planner 产出的顺序)。
        plan: 本集的 ``EpisodePlan``(title / synopsis / director_intent 基线)。
        member_beats: 本集所含的 N 段 ``Beat``(按时序,= ``plan.beat_indices`` 对应)。
        characters: 全局人物档案(终态)。
        batches: 原文 batch 字典(用于回查所有 beat 的 related_batches,跨 batch 拼接)。
        prev_plan / next_plan: 相邻集 plan 摘要;首集 / 末集传 ``None``。
        prev_tail_shots: 上集末尾几镜的完整 ``Shot`` 列表;
            本 agent 只读其中叙事字段。首集传 ``None`` / ``[]``。
        target_duration_sec: 单集目标时长(秒),用于节奏对齐。

    Returns:
        ``NarrativeShotList``:含 director_intent + shots(NarrativeShot 列表)。
    """
    if not member_beats:
        raise ValueError(f"narrate_episode(ep={ep_index}):member_beats 不能为空")

    llm = get_llm()
    chars_list, raw_text = _gather_inputs_for_episode(
        member_beats, characters, batches, ep_index=ep_index,
    )
    prev_tail = prev_tail_shots or []

    # 镜数典型区间:目标时长 / 8s ~ 目标时长 / 5s(单镜 5-8s 节奏)。
    # LLM 数学不稳,直接给具体上下限引导。
    shots_min = max(1, target_duration_sec // 8)
    shots_max = max(shots_min + 1, target_duration_sec // 5)
    system = SYSTEM_PROMPT.format(
        target_duration=target_duration_sec,
        shots_min=shots_min,
        shots_max=shots_max,
    )
    user = _build_user_prompt(
        ep_index=ep_index,
        plan=plan,
        member_beats=member_beats,
        chars=chars_list,
        raw_text=raw_text,
        target_duration=target_duration_sec,
        prev_plan=prev_plan,
        next_plan=next_plan,
        prev_tail_shots=prev_tail,
    )

    # 本集所含 setting refs 并集(保序去重)
    setting_seen: Set[str] = set()
    allowed_settings: List[str] = []
    for b in member_beats:
        for s in b.setting_refs:
            if s in setting_seen:
                continue
            setting_seen.add(s)
            allowed_settings.append(s)

    logger.info(
        "narrate 第 %d 集(%s):%d 段 beat / %d 人 / %d setting / 原文 %d 字;"
        "上下文 prev=%s next=%s prev_tail=%d 镜",
        ep_index, plan.title, len(member_beats), len(chars_list),
        len(allowed_settings), len(raw_text),
        f"集 {prev_plan.title}" if prev_plan else "首集",
        f"集 {next_plan.title}" if next_plan else "末集",
        len(prev_tail),
    )

    narrative = llm.chat_json(system, user, NarrativeShotList)

    touched = _sanitize_narrative_shots(
        narrative.shots,
        roster_chars=set(characters.keys()),
        allowed_settings=allowed_settings,
        ep_index=ep_index,
    )

    logger.info(
        "narrate 第 %d 集产出:%d 镜叙事%s",
        ep_index, len(narrative.shots),
        f"(其中 {touched} 镜被白名单过滤修正)" if touched else "",
    )

    return narrative


def merge_episode(
    *,
    ep_index: int,
    plan: EpisodePlan,
    narrative: NarrativeShotList,
    direction: ShotDirectionList,
) -> Episode:
    """按 index 把叙事分镜 + 视觉指导合并成 ``Episode``。

    shot_director 已经做过 index 对齐,缺的镜会用空 ``ShotDirection`` 占位
    (shot_type 兜底为「中景」),所以这里假设 narrative 和 direction 的 index
    集合一致。多出来的 direction.shots(不该出现)会被忽略。

    Episode 字段来源:
      * ``title`` / ``synopsis`` / ``beat_indices`` ← ``plan``
      * ``director_intent`` ← ``narrative``(narrate 可能在 plan 基线上微调)
      * ``visual_style`` ← ``direction``
      * ``shots`` ← 按 index 配对合并
    """
    dir_by_idx: Dict[int, ShotDirection] = {d.index: d for d in direction.shots}

    shots: List[Shot] = []
    for n in narrative.shots:
        d = dir_by_idx.get(n.index)
        if d is None:
            logger.warning(
                "[merge ep=%d shot=%d] 视觉指导缺失,以空中景占位",
                ep_index, n.index,
            )
            d = ShotDirection(index=n.index, shot_type="中景")
        shots.append(
            Shot(
                index=n.index,
                intent=n.intent,
                narrative_description=n.description,
                characters=n.characters,
                setting=n.setting,
                speaker=n.speaker,
                dialogue=n.dialogue,
                voiceover=n.voiceover,
                duration_sec=n.duration_sec,
                shot_type=d.shot_type,
                camera_motion=d.camera_motion,
                shot_action=d.shot_action,
                description=d.description,
            )
        )

    return Episode(
        index=ep_index,
        title=f"第 {ep_index} 集 · {plan.title}",
        synopsis=plan.synopsis,
        beat_indices=list(plan.beat_indices),
        director_intent=narrative.director_intent,
        visual_style=direction.visual_style,
        shots=shots,
    )


DEFAULT_PREV_TAIL_K = 3
"""默认从上集末尾取多少镜作为本集承接参考。"""


__all__ = [
    "DEFAULT_PREV_TAIL_K",
    "SYSTEM_PROMPT",
    "merge_episode",
    "narrate_episode",
    "get_llm",
    "set_llm",
    "set_trace_dir",
]
