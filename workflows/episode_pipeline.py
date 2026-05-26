"""编排「一集 EpisodePlan(含 N 段 Beat)→ 一集 Episode」的两个 agent 调用 + 合并。

封装顺序:

1. ``extract_storyboard.narrate_episode`` — LLM 叙事分镜(谁 / 在哪 / 说什么 /
   想什么 + 集层 director_intent)
2. ``shot_director.direct_episode``       — LLM 视觉指导(景别 / 运镜 / 起始画面 /
   时长 + 集层 visual_style)
3. ``extract_storyboard.merge_episode``   — 按 index 配对合并成完整 ``Episode``

两个父 workflow(``novel_analysis`` / ``storyboard_analysis``)都用这个 helper,
保持「拆分两个 agent」对 workflow 透明。

**输入颗粒度**:已经升级为"集"(``EpisodePlan`` + ``member_beats``),不再
是单段 beat。1 集 = N 段 beat,跨 batch 原文已自动拼接。
"""

from __future__ import annotations

from typing import Dict, List

from skills.batch_chapters import Batch
from agents.extract_storyboard import merge_episode, narrate_episode
from agents.shot_director import direct_episode
from schemas.beat import Beat
from schemas.character import Character
from schemas.episode_plan import EpisodePlan
from schemas.storyboard import Episode, Storyboard


def build_episode_from_plan(
    *,
    ep_index: int,
    plan: EpisodePlan,
    member_beats: List[Beat],
    known_chars: Dict[str, Character],
    batch_lookup: Dict[int, Batch],
    prev_plan: EpisodePlan | None,
    next_plan: EpisodePlan | None,
    prev_tail: List[Storyboard],
    target_duration_sec: int,
) -> Episode:
    """一集流水线:叙事分镜 → 视觉指导 → 合并 Episode(2 次 LLM 调用)。

    Args:
        ep_index: 集序号(1-based)。
        plan: 本集的 ``EpisodePlan``(episode_planner 产出)。
        member_beats: 本集所含 N 段 ``Beat``(按 ``plan.beat_indices`` 取出)。
        known_chars: 全员人物档案(终态,episode_planner 已用过)。
        batch_lookup: 原文 batch 字典,用于回查所有 ``beat.related_batches`` 跨
            batch 拼接。
        prev_plan / next_plan: 相邻集 ``EpisodePlan`` 摘要;首集 / 末集传 ``None``。
        prev_tail: 上集末尾 K 镜的完整 ``Storyboard`` 列表(叙事 + 视觉);
            两个 agent 各自挑感兴趣的字段渲染。首集传 ``[]``。
        target_duration_sec: 本集目标总时长(秒)。

    Returns:
        完整 ``Episode``(含合并后的 storyboards)。
    """
    narrative = narrate_episode(
        ep_index=ep_index,
        plan=plan,
        member_beats=member_beats,
        characters=known_chars,
        batches=batch_lookup,
        prev_plan=prev_plan,
        next_plan=next_plan,
        prev_tail_storyboards=prev_tail,
        target_duration_sec=target_duration_sec,
    )
    direction = direct_episode(
        episode_index=ep_index,
        director_intent=narrative.director_intent,
        narrative_shots=narrative.shots,
        characters=known_chars,
        prev_tail_storyboards=prev_tail,
        target_duration_sec=target_duration_sec,
    )
    return merge_episode(
        ep_index=ep_index,
        plan=plan,
        narrative=narrative,
        direction=direction,
    )


__all__ = ["build_episode_from_plan"]
