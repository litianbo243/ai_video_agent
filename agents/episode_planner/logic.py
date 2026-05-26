"""全书 Beat 聚合成 N 集 Episode:一次 LLM 调用 + 兜底校验。

主 API:

* ``plan_episodes(beats, characters, *, target_duration_sec) -> EpisodePlanList``
    LLM 一次性把全书 Beat 列表聚合成 M 集;输出做严格校验
    (beat_indices 不重不漏覆盖所有 Beat.index,严格升序连续)。
    校验失败 → rule-based fallback(贪心装箱,按估算时长贪心切)+ warn。

LLM 配置在同包 ``llm.json``,首次 ``get_llm()`` 才 build。
测试 mock:``from agents.episode_planner import set_llm; set_llm(fake)``。

**聚合粒度**:1 个 beat 自身 ≈ 60-120 秒视频体量;目标每集 {target_duration} 秒。
典型 1 集 = 2-5 个 beat。戏剧浓度高的 beat 可单独成集,浓度低的必须合并。

**职责切分**:本 agent **只做规划**,不写分镜;``EpisodePlan.director_intent``
是给下游 ``narrate_episode`` 的基线调子,后者可在分镜阶段微调。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Set

from llm.agent_llm import make_agent_llm_manager
from schemas.beat import Beat
from schemas.character import Character
from schemas.episode_plan import EpisodePlan, EpisodePlanList

logger = logging.getLogger(__name__)


get_llm, set_llm, set_trace_dir = make_agent_llm_manager(
    agent_name="episode_planner",
    config_path=Path(__file__).parent / "llm.json",
)


SYSTEM_PROMPT_TEMPLATE = """\
你是一名中文短视频剧本规划师。任务:把已经切好的一连串紧凑的 Beat(剧情段)
**聚合**成一集一集的短剧 Episode,每集目标时长 ≈ {target_duration} 秒。

输入:
- 全书已抽取的 Beat 列表(每段含 title / summary / setting_refs / character_refs)
- 全员人物详档(name + 关系)

────────────────────────────────────────────────
【聚合原则】

1. **同一情绪弧**:相邻 beat 若属于同一情绪走向(比如「试探 → 设套 → 得手 → 余韵」)
   合并为 1 集。情绪明显跳脱(悲伤 ↔ 喜剧 / 紧张 ↔ 放松)→ 断开。

2. **同一主线 / 角色组**:相邻 beat 的核心人物 / 主线相同 → 合并;视角切换到
   完全不同角色 / 副线 → 断开(独立成集,或合到下一组同主线)。

3. **集长适配 / 起伏感**:估算 1 个 beat ≈ 60-120 秒视频体量,目标每集 ≈ {target_duration} 秒,
   典型 {beats_per_ep_lo}-{beats_per_ep_hi} 段,**目标均值 ≈ {beats_per_ep_avg} 段/集**。
   **每集 ≥ {min_beats_per_ep} 段**(全书 < {min_beats_per_ep} 段时例外,全部并 1 集)。

   **集长必须随戏剧浓度起伏**(像呼吸节奏),典型分布:
   - 戏剧浓度高的集(强冲突 / 多场景切换 / 多角色交汇 / 关键反转) →
     {beats_per_ep_avg}-{beats_per_ep_hi} 段(铺垫 + 高潮 + 余波)
   - 中等节奏集 → {beats_per_ep_avg} 段左右
   - 紧凑 / 过渡型集 → 下界 {min_beats_per_ep} 段(**少用**,只在剧情天然形成短独立单元时)
   - 戏剧浓度低的 beat(过渡 / 内心戏 / 单一发现瞬间 / 情绪余波)**绝不单独成集**,
     合到前后情绪相连段;**也不要硬凑 {min_beats_per_ep} 段就完事**

   ❌ **所有集 uniformly 都切到下界 {min_beats_per_ep} 段** → 节奏死板,违反"呼吸感";
      宁可让丰满集到 {beats_per_ep_hi} 段、紧凑集到 {min_beats_per_ep} 段,**长短交错**
   ❌ "发现可疑物 / 收到匿名电话 / 一瞬间眼神交错"等单一事件 beat 单独成集 →
      合到前段做余波,或合到后段做开场

4. **节奏对照**:避免连续多集节奏雷同。如果连续多集都是"对话戏",中间安排
   一集"动作 / 视觉冲击戏"(若 beat 本身支持)。

5. **集间清爽**:让每集首 / 末 beat 在情绪上有自然的开/合,集中部分推进情节。
   不要让一个戏剧高潮跨越集边界(高潮和反应都应该在同一集)。

6. **对齐人物弧光锚点**:人物档案 arc 字段里写了「在 [事件 X] 后,从 A 转为 B」
   这类锚点。如果某 beat 正好对应弧光锚点事件(关键转折),**优先把转折前后
   切到不同集**(或同集但作为本集高潮收尾),让人物变化跟集层呼吸节奏对齐。

7. **集末钩子(每集都要有)**:除末集外,每集集末 beat 尽量落在一个**能引出
   下集**的悬念点上 —— 未解之谜 / 关键决断瞬间 / 反转前一拍 / 情绪悬置 /
   关键道具或人物突然出现 / 关系突变 / 危机刚降临。让观众**想看下一集**。
   - 与 #5「高潮不跨集」并行不冲突:**高潮在本集完成,钩子是高潮收尾后的
     余波或新冲突的种子**(典型结构:本集高潮 + 反应 → 末尾抛出新问题 / 新威胁)
   - 切集时若发现下一 beat 才是天然钩子,可微调集边界把这段 beat 划入本集末
   - 末集(收束集)例外:用"余韵 / 闭环"收尾,不强求钩子

────────────────────────────────────────────────
【字段填写】

title(4-8 字):
- 概括本集主线,不重复某个 beat 标题(beat 标题更细)
- 不含集号(集号由 workflow 自动加)
- 例:「初识误会」「设局陷害」「真相反击」「劫后重逢」

synopsis(2-3 句):
- 把本集所含 beat 的 summary **拧成一根主线**,不要罗列
- 体现"起承转合";不抄原文

director_intent(2-4 句):
- **基调**:压抑 / 暧昧 / 凌厉 / 紧张 / 温暖 / 阴郁 ...(一词定调)
- **重点**:让观众**感受到**或**记住**的核心(情绪 / 关系 / 冲突 / 反转点)
- **节奏**:整集呼吸感(平稳铺陈 / 张力累积 / 短促爆发 / 留白收尾 / 反转 ...)
- **视觉锤**:1-2 个标志性意象 / 道具(供下游分镜 agent 复用)
- ⚠️ **不剧透人物弧光的未发生转变**:人物 arc 是全书回看视角,本集 director_intent
  只反映"截至本集所含 beats 末尾"的状态。例:本集 = 初识相处,人物 arc 写
  「被亲密伴侣背叛后转为冷峻戒备」 → director_intent 调子应是"信赖 / 暖意",
  **不要**提前给出"冷峻 / 戒备 / 危险预兆"调子(那是后续集的事)

beat_indices(本集所含 Beat.index 列表):
- 严格按时序,1-based,**严格升序、连续、不跳号、不重号**
- 全书所有 beat **必须不重不漏**地分配到某一集(并集 = 全部 Beat.index)

────────────────────────────────────────────────
【硬约束】
- 所有 beat 必须恰好分配到一集(无漏 / 无重)
- 每集 beat_indices 必须严格升序、连续(不允许跳号或乱序)
- **每集 ≥ {min_beats_per_ep} 段**(全书 < {min_beats_per_ep} 段 → 全部并 1 集)
- **集长要起伏:不要全部都切到下界 {min_beats_per_ep} 段;目标均值 ≈ {beats_per_ep_avg} 段/集**
- 严格按 JSON Schema 输出
"""


# 单 beat 估算时长(秒);用于 prompt 给 LLM 算每集 beat 数,也用于 fallback。
DEFAULT_BEAT_DURATION_SEC = 90

# 每集 beat 数下界:避免 LLM 把单一发现 / 反应瞬间切成单 beat 集(分镜太薄)。
# fallback 装箱也用同一值,保持口径一致。
MIN_BEATS_PER_EP = 3


def _target_avg_beats(target_duration_sec: int) -> int:
    """目标均值(段/集),用于 prompt 锚定 + user prompt 反推目标集数。

    取 ``max(MIN+1, round(target/DEFAULT_BEAT_DUR))``,确保比下界至少高 1 段——
    LLM 看到均值锚点不会贴下界,自然产生集长起伏。
    """
    raw = round(target_duration_sec / DEFAULT_BEAT_DURATION_SEC)
    return max(MIN_BEATS_PER_EP + 1, raw)


def _build_system_prompt(target_duration_sec: int) -> str:
    beats_per_ep_lo = max(MIN_BEATS_PER_EP, target_duration_sec // 120)
    beats_per_ep_hi = max(beats_per_ep_lo + 1, target_duration_sec // 60)
    return SYSTEM_PROMPT_TEMPLATE.format(
        target_duration=target_duration_sec,
        beats_per_ep_lo=beats_per_ep_lo,
        beats_per_ep_hi=beats_per_ep_hi,
        beats_per_ep_avg=_target_avg_beats(target_duration_sec),
        min_beats_per_ep=MIN_BEATS_PER_EP,
    )


# 默认 prompt(180s),供单元测试 / notebook 直接看 prompt 时用。
SYSTEM_PROMPT = _build_system_prompt(180)


def _render_beat_list(beats: List[Beat]) -> str:
    if not beats:
        return "(无)"
    lines: List[str] = []
    for b in beats:
        lines.append(f"段 {b.index} · {b.title}")
        if b.summary:
            lines.append(f"  {b.summary}")
        if b.setting_refs:
            lines.append(f"  setting: {b.setting_refs}")
        if b.character_refs:
            lines.append(f"  characters: {b.character_refs}")
    return "\n".join(lines)


def _render_characters(characters: Dict[str, Character]) -> str:
    if not characters:
        return "(无)"
    parts: List[str] = []
    for ch in characters.values():
        section = [f"### {ch.name}"]
        if ch.aliases:
            section.append(f"别名: {', '.join(ch.aliases)}")
        if ch.background:
            section.append(f"背景: {ch.background}")
        if ch.arc:
            section.append(f"弧光(带事件锚点): {ch.arc}")
        parts.append("\n".join(section))
    return "\n\n".join(parts)


def _build_user_prompt(
    beats: List[Beat],
    characters: Dict[str, Character],
    target_duration_sec: int,
    *,
    title: str,
) -> str:
    target_avg = _target_avg_beats(target_duration_sec)
    target_eps = max(1, round(len(beats) / target_avg))
    return (
        f"书名: {title or '(未提供)'} | 共 {len(beats)} 段 Beat\n\n"
        f"=== 全书 Beat 清单 ===\n{_render_beat_list(beats)}\n\n"
        f"=== 全员人物 ===\n{_render_characters(characters)}\n\n"
        f"=== 任务 ===\n"
        f"按 JSON Schema 输出:{{\"plans\": [...]}}\n"
        f"全书 {len(beats)} 段 beat 必须不重不漏地分配到某一集;每集时长 ≈ {target_duration_sec} 秒。\n"
        f"**目标集数 ≈ {target_eps} 集**(均值 ≈ {target_avg} 段/集);集长要起伏,不要全切到下界。"
    )


def _validate_and_repair(
    plan_list: EpisodePlanList,
    beats: List[Beat],
) -> EpisodePlanList:
    """严格校验 LLM 输出:beat_indices 不重不漏覆盖所有 Beat.index,
    且每集内严格升序连续。

    校验失败 → fallback 到 rule-based(贪心装箱),并 warn。
    """
    all_idx = sorted(b.index for b in beats)
    expected_set = set(all_idx)

    seen: Set[int] = set()
    duplicate: List[int] = []
    out_of_order_plans: List[int] = []
    for pi, plan in enumerate(plan_list.plans, start=1):
        if not plan.beat_indices:
            logger.warning("[episode_planner] 集 %d 的 beat_indices 为空", pi)
            continue
        sorted_idx = sorted(plan.beat_indices)
        if plan.beat_indices != sorted_idx:
            out_of_order_plans.append(pi)
        # 检查是否连续(允许 gap 才算)
        for idx in plan.beat_indices:
            if idx in seen:
                duplicate.append(idx)
            seen.add(idx)

    missing = expected_set - seen
    extra = seen - expected_set

    if missing or duplicate or extra or out_of_order_plans:
        logger.warning(
            "[episode_planner] LLM 输出校验失败:missing=%s duplicate=%s "
            "extra=%s out_of_order_plans=%s;走 rule-based fallback",
            sorted(missing), sorted(set(duplicate)), sorted(extra), out_of_order_plans,
        )
        return _fallback_plan(beats)

    return plan_list


def _fallback_plan(beats: List[Beat]) -> EpisodePlanList:
    """贪心装箱:按时序把 beat 切成每集 ``MIN_BEATS_PER_EP`` 段(口径与 prompt 一致)。

    每集 title / synopsis / director_intent 走"集 X"占位,提示 caller LLM
    plan 失败,需要人工或重跑。
    """
    plans: List[EpisodePlan] = []
    chunk_size = MIN_BEATS_PER_EP
    for i in range(0, len(beats), chunk_size):
        chunk = beats[i:i + chunk_size]
        if not chunk:
            continue
        plans.append(EpisodePlan(
            title=f"集 {len(plans) + 1}(待修订)",
            synopsis=" ".join(b.summary for b in chunk if b.summary),
            director_intent="(LLM 规划失败,fallback 占位;请人工 review 或重跑 episode_planner)",
            beat_indices=[b.index for b in chunk],
        ))
    return EpisodePlanList(plans=plans)


def plan_episodes(
    beats: List[Beat],
    characters: Dict[str, Character],
    *,
    target_duration_sec: int = 300,
    title: str = "",
) -> EpisodePlanList:
    """LLM 一次性把全书 Beat 聚合成 M 集 Episode。

    Args:
        beats: 全书已抽取的剧情段列表(按 ``Beat.index`` 升序)。
        characters: 全员人物档案(终态)。
        target_duration_sec: 每集目标时长(秒),典型 180-300。
        title: 书名(prompt 里用作上下文)。

    Returns:
        ``EpisodePlanList``,每个 plan 含 title / synopsis / director_intent /
        beat_indices。校验失败时返回 rule-based fallback,带 warn。

    若全书只有 0-1 个 beat,直接返回单集 plan(不调 LLM)。
    """
    if len(beats) == 0:
        logger.warning("[episode_planner] beats 为空,返回空 plan")
        return EpisodePlanList(plans=[])
    if len(beats) == 1:
        b = beats[0]
        logger.info("[episode_planner] 全书只 1 段 beat,直接单集 plan(不调 LLM)")
        return EpisodePlanList(plans=[EpisodePlan(
            title=b.title or "集 1",
            synopsis=b.summary,
            director_intent="(单段 beat 直入,未走 planner LLM)",
            beat_indices=[b.index],
        )])

    llm = get_llm()
    system = _build_system_prompt(target_duration_sec)
    user = _build_user_prompt(beats, characters, target_duration_sec, title=title)

    logger.info(
        "[episode_planner] 开始规划:%d 段 beat / %d 人,目标每集 %d 秒,%s @ %s",
        len(beats), len(characters), target_duration_sec, llm.model, llm.base_url,
    )

    raw = llm.chat_json(system, user, EpisodePlanList)

    repaired = _validate_and_repair(raw, beats)

    logger.info(
        "[episode_planner] 完成:%d 段 beat → %d 集(平均 %.1f beat/集)",
        len(beats), len(repaired.plans),
        len(beats) / max(len(repaired.plans), 1),
    )
    return repaired


__all__ = [
    "DEFAULT_BEAT_DURATION_SEC",
    "SYSTEM_PROMPT",
    "SYSTEM_PROMPT_TEMPLATE",
    "get_llm",
    "plan_episodes",
    "set_llm",
    "set_trace_dir",
]
