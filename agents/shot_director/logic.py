"""一集叙事分镜 → 一集视觉指导:一次 LLM 调用。

接到上游 ``extract_storyboard`` 出的一集 ``NarrativeShot`` 列表(已含
intent / characters / setting / speaker / dialogue / voiceover + 集层
``director_intent``),本 agent 决定每镜的"怎么拍":景别 / 运镜 / 动作演变 /
起始画面 / 时长。

输出 ``ShotDirectionList``,workflow 按 ``index`` 与 ``NarrativeShot`` 配对
合并成最终 ``Storyboard``。

LLM 配置在同目录 ``llm.json``,本模块顶部 lazy build。
测试时想 mock:``from agents.shot_director import set_llm; set_llm(fake)``。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

from llm.agent_llm import make_agent_llm_manager
from schemas.character import Character
from schemas.shot_direction import ShotDirection, ShotDirectionList

# ``NarrativeShot`` / ``Storyboard`` 仅做类型注解(``from __future__ import
# annotations`` 已开,运行期靠 duck typing 读字段),关进 TYPE_CHECKING
# 不引入运行时多余 import。schema 中心化后已无循环依赖隐患。
if TYPE_CHECKING:
    from schemas.narrative import NarrativeShot
    from schemas.storyboard import Storyboard

logger = logging.getLogger(__name__)


get_llm, set_llm, set_trace_dir = make_agent_llm_manager(
    agent_name="shot_director",
    config_path=Path(__file__).parent / "llm.json",
)


SYSTEM_PROMPT = """\
你是一名中文短视频镜头导演。上游叙事分镜师已经把一集切成了若干 NarrativeShot
(每镜已锁定 intent / 剧情 / characters / setting / speaker / dialogue /
voiceover / **duration_sec**),你的任务是为每镜出视觉指导:景别 / 运镜 /
动作演变 / 起始画面。

**时长不归你管**(叙事分镜师已估好,你照办即可,但要保证你设计的运镜 / 动作
能在该时长内完成 —— 复杂运镜/动作不要塞 3 秒镜里)。整集总时长 ≈ {target_duration} 秒。

────────────────────────────────────────────────
【先定调,再分镜】

集层 visual_style(2-3 句,先填,再填 shots):
- 色调 / 光影(冷暖 / 明暗 / 特殊光位)
- 构图偏好(对称 / 失衡 / 大量留白 / 极端景别堆叠 ...)
- 运镜偏好(整集多固定 + 关键镜推 / 大量手持跟随 / 静与动对比 ...)
- 视觉锤(1-2 个标志性意象 / 道具,贯穿用,或集末收束)

镜层视觉决策**服务于** NarrativeShot.intent(已锁定,你只能解读不能改),
例如:
- 「试探的压迫感」→ 中景 + 缓推,挤压空间
- 「反应镜,破防」→ 大特写定格,定住眼神
- 「视觉锤,定调」→ 远景留白,人物渺小
- 「钩子,暗示反转」→ 局部特写道具 / 动作,遮住表情

集层 visual_style 是 director_intent 在视觉上的落地,所有 shots 串起来应能
讲出一条统一的视觉曲线。

────────────────────────────────────────────────
【字段填写】

shot_type(6 选 1 — 远景 / 全景 / 中景 / 近景 / 特写 / 大特写):
- 远景 / 全景:定场 / 孤立 / 留白 / 视觉锤
- 中景 / 近景:对话 / 一般动作
- 特写 / 大特写:情绪 / 道具 / 钩子
- 同景别连续超 3 镜 → 换景别(避免视觉疲劳)

camera_motion(运镜,默认留空 = 固定):
- 推:挤压 / 紧张累积  · 拉:释放 / 揭示
- 摇 / 移:展示空间   · 跟:跟随动作
- 升:压迫            · 降:释放
- 环绕:迷失 / 凝重
- **整集 2-3 个有运镜的镜就够,不滥用**

shot_action(画面在本镜**内**如何变化,动词为主):
- 例:「缓缓抬头,泪痕未干」「双手攥紧,指节发白」「转身,背影远去」
- 静态镜头(对白戏 / 表情特写不动 / 纯定场)→ **留空**,不要硬编

description(本镜起始那一刻的画面 + 氛围,给图像 / 视频生成 prompt 用):
- 谁 + 在哪 + 做什么 + 光线 + 构图视角
- **本质是把上游叙事描述视觉化**:NarrativeShot.description(剧情:谁做了什么)
  + setting + 人物档案外貌 + 你选的 shot_type / 光线 → 一个**可被图像生成模型画出**
  的画面
- **不要塞 dialogue 字面**(「她说『我没事』」式无效),让画面自己说话
- 不要复述 NarrativeShot.description(剧情概要),要把它**翻译成视觉细节**
  (例:剧情=「A 敲门进入办公室」→ description=「办公室门口,A 右手叩响实木门,
  侧脸,身着风衣,午后柔和侧光从右侧窗户打来,景别近景」)

────────────────────────────────────────────────
【集间视觉承接】

看输入「上集末尾 K 镜的视觉指导」(若有):
- 色调 / 光线 / 构图基调延续上集
- 集首画面的空间 / 姿态 / 道具跟上集末镜自然延续
- 上集大特写收尾 → 本集可从大特写淡入或反推远景
- 无上集(首集) → 直接定场,自创本集视觉调性

────────────────────────────────────────────────
【硬约束】
- shots 列表的 index 严格对齐输入的每个 NarrativeShot.index,**不增不减不改序**
- 一集所有镜头时长之和 ≈ {target_duration} 秒
- 严格按 JSON Schema 输出
"""


def _build_system_prompt(target_duration_sec: int) -> str:
    return SYSTEM_PROMPT.format(target_duration=target_duration_sec)


def _render_characters_for_prompt(chars: List[Character]) -> str:
    if not chars:
        return "(无)"
    parts: List[str] = []
    for ch in chars:
        section = [f"### {ch.name}"]
        if ch.appearance:
            section.append(f"外貌: {ch.appearance}")
        parts.append("\n".join(section))
    return "\n\n".join(parts)


def _render_narrative_shots(shots: List[NarrativeShot]) -> str:
    """紧凑渲染一集叙事分镜:每镜一段,叙事字段全展开。"""
    if not shots:
        return "(无)"
    parts: List[str] = []
    for s in shots:
        head = f"# 镜 {s.index}"
        if s.duration_sec:
            head += f" ({s.duration_sec:.0f}s)"
        lines = [head]
        if s.intent:
            lines.append(f"intent: {s.intent}")
        if s.description:
            lines.append(f"剧情: {s.description}")
        if s.setting:
            lines.append(f"setting: {s.setting}")
        if s.characters:
            lines.append(f"characters: {', '.join(s.characters)}")
        if s.dialogue:
            speaker = s.speaker or "?"
            lines.append(f"dialogue [{speaker}]: {s.dialogue}")
        if s.voiceover:
            speaker = s.speaker or "?"
            lines.append(f"voiceover [{speaker}]: {s.voiceover}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def _render_prev_tail_visual(prev_tail: List[Storyboard]) -> str:
    """紧凑渲染上集末尾 K 镜的**视觉维度**(景别 / 运镜 / 起始画面 / 动作)。

    本 agent 只关心视觉承接,所以即使 prev_tail 是完整 Storyboard 也只挑视觉字段。
    """
    if not prev_tail:
        return "(首集,无上集视觉)"
    lines: List[str] = []
    for sb in prev_tail:
        head = f"- 镜 {sb.index} [{sb.shot_type}]"
        if sb.camera_motion:
            head += f" / 运镜:{sb.camera_motion}"
        lines.append(f"{head}: {sb.description}")
        if sb.shot_action:
            lines.append(f"  动作: {sb.shot_action}")
    return "\n".join(lines)


def _build_user_prompt(
    *,
    director_intent: str,
    narrative_shots: List[NarrativeShot],
    chars: List[Character],
    prev_tail: List[Storyboard],
    target_duration_sec: int,
) -> str:
    return (
        f"=== 本集叙事调性(已锁定)===\n"
        f"director_intent: {director_intent or '(未提供)'}\n\n"
        f"=== 上集末尾画面(视觉承接参考,共 {len(prev_tail)} 镜)===\n"
        f"{_render_prev_tail_visual(prev_tail)}\n\n"
        f"=== 关联人物档案(外貌一致性)===\n{_render_characters_for_prompt(chars)}\n\n"
        f"=== 本集叙事分镜(共 {len(narrative_shots)} 镜,需要严格按 index 一一配视觉)===\n"
        f"{_render_narrative_shots(narrative_shots)}\n\n"
        f"=== 任务 ===\n"
        f"按 JSON Schema 输出:先填 visual_style(2-3 句),再为每镜填 shots。\n"
        f"目标总时长 ≈ {target_duration_sec} 秒。\n"
        f"输出格式:`{{\"visual_style\": \"...\", \"shots\": [...]}}`"
    )


def _validate_index_alignment(
    direction_list: ShotDirectionList,
    narrative_shots: List[NarrativeShot],
    *,
    episode_index: int,
) -> ShotDirectionList:
    """LLM 输出的 shots 必须严格对齐 narrative_shots 的 index 列表。

    严格模式:不对齐就 raise。轻量模式:warn + 用 index map 修复。
    实践中 LLM 偶尔漏 1-2 镜或乱序,这里用 map 修正(以 narrative 为权威),
    缺的镜降级为空 ShotDirection,多的镜丢弃 + warn。
    """
    by_idx: Dict[int, ShotDirection] = {d.index: d for d in direction_list.shots}
    expected = [s.index for s in narrative_shots]

    fixed: List[ShotDirection] = []
    missing: List[int] = []
    for idx in expected:
        if idx in by_idx:
            fixed.append(by_idx.pop(idx))
        else:
            missing.append(idx)
            fixed.append(
                ShotDirection(
                    index=idx,
                    shot_type="中景",
                )
            )
    extras = sorted(by_idx.keys())

    if missing:
        logger.warning(
            "[shot_director ep=%d] LLM 漏出 %d 个镜(index=%s),已用空中景占位",
            episode_index, len(missing), missing,
        )
    if extras:
        logger.warning(
            "[shot_director ep=%d] LLM 多出 %d 个镜(index=%s),已丢弃",
            episode_index, len(extras), extras,
        )

    return ShotDirectionList(
        visual_style=direction_list.visual_style,
        shots=fixed,
    )


def direct_episode(
    *,
    episode_index: int,
    director_intent: str,
    narrative_shots: List[NarrativeShot],
    characters: Dict[str, Character],
    prev_tail_storyboards: List[Storyboard] | None = None,
    target_duration_sec: int = 180,
) -> ShotDirectionList:
    """为一集叙事分镜出视觉指导(1 次 LLM 调用)。

    Args:
        episode_index: 本集编号(= 对应 Beat.index),仅用于日志 / warn。
        director_intent: 集层叙事调性(extract_storyboard 已写入)。
        narrative_shots: 本集所有叙事分镜(已含 intent / characters / 等)。
        characters: 全局人物档案(取本集涉及人物的 appearance 用于视觉一致性)。
        prev_tail_storyboards: 上集末尾几镜的完整 ``Storyboard``,本 agent 只
            读其中视觉字段做承接;首集传 ``None`` / ``[]``。
        target_duration_sec: 本集目标总时长(秒)。

    Returns:
        ``ShotDirectionList``:含 visual_style 和与 narrative_shots index 一一
        对齐的 shots。
    """
    if not narrative_shots:
        logger.warning("[shot_director ep=%d] 输入 narrative_shots 为空,跳过 LLM 调用", episode_index)
        return ShotDirectionList(visual_style="", shots=[])

    relevant_chars: List[Character] = []
    seen_char: set = set()
    for s in narrative_shots:
        for ref in s.characters:
            if ref in characters and ref not in seen_char:
                seen_char.add(ref)
                relevant_chars.append(characters[ref])

    prev_tail = prev_tail_storyboards or []

    llm = get_llm()
    system = _build_system_prompt(target_duration_sec)
    user = _build_user_prompt(
        director_intent=director_intent,
        narrative_shots=narrative_shots,
        chars=relevant_chars,
        prev_tail=prev_tail,
        target_duration_sec=target_duration_sec,
    )

    logger.info(
        "shot_director 第 %d 集:%d 镜叙事 → 出视觉指导;人物 %d,prev_tail=%d 镜",
        episode_index, len(narrative_shots), len(relevant_chars), len(prev_tail),
    )

    raw = llm.chat_json(system, user, ShotDirectionList)
    aligned = _validate_index_alignment(
        raw, narrative_shots, episode_index=episode_index,
    )

    logger.info(
        "shot_director 第 %d 集产出:%d 镜视觉指导(visual_style: %d 字)",
        episode_index, len(aligned.shots), len(aligned.visual_style),
    )

    return aligned


__all__ = [
    "SYSTEM_PROMPT",
    "direct_episode",
    "get_llm",
    "set_llm",
    "set_trace_dir",
]
