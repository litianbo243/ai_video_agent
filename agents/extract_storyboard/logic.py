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
你是专业的中文小说视频改编编剧,正在为小说的某一段剧情(一集)改编分镜。

输入包括:
- 一段剧情大纲(Beat:title + summary + setting_refs + character_refs)
- 关联人物的档案(含外貌 + 性格)—— 用于跨集视觉一致性
- ★ 这段剧情对应的**原文**(整批章节,几千字)
  —— 台词 / 旁白 / 视觉描写**都直接从原文里挑选改编**,不要凭空创作

────────────────────────────────────────────────
你的任务: 编排本集的分镜清单(storyboards)。

每集 ~15-30 个镜头,每镜 3-15 秒,平均 6-8 秒;一集所有镜头时长之和 ≈ {target_duration} 秒。

────────────────────────────────────────────────
【改编规则】

镜头粒度:
- 优先选择**有戏剧张力的画面**(冲突 / 转折 / 决断 / 情绪爆发)
- 平稳过渡可以一镜带过
- 原文里不重要的支线 / 水分内容**直接跳过**(不必逐字改编)

dialogue 字段(从原文里挑选):
- 浏览原文,挑出有戏剧张力的**角色对白 / 群众议论 / 测验员或执事式播报**:
  揭示关系 / 体现冲突 / 决断 / 反转 / 当众宣布结果
- **照抄原文台词**;太长不适合视频节奏 → 浓缩或在切点拆分
- 长台词跨多镜时,**不要重复整句**,在标点处把句子拆成多段
- 普通陈述、寒暄、过渡台词不要选
- 说话者不在「角色档案」里(路人、测验员等)时,仍把其**开口说的话**放在 dialogue;
  characters 字段只填本镜画面里可辨认的 roster 人物,没有则留空列表
- 实在挑不到合适的可自由创作,但要保留人物语气

voiceover 字段(从原文挑选,但默认少用):
- **优先留空字符串 ""** —— 能用镜头 + 表演 + 环境音表现的,就不要念旁白
- 仅当原文有**第三人称叙事/作者评论/必要信息**且**画面 alone 难以传达**时,
  才从原文**浓缩为一句**念白(建议 ≤35 字),保留语气
- **禁止**放入以下内容(这些一律进 dialogue,不是旁白):
  * 任何人**说出声**的话(含群众嘲讽、窃窃私语式的直接引语)
  * 测验员 / 执事 / 系统播报式宣读(如「斗之气,X 段」)
  * 带引号的对白原文
- **禁止**与 description 同义重复:若 description 已写清动作、情绪、场面,
  voiceover **必须为空**,不要把小说叙述再念一遍
- 长旁白跨多镜时按画面切点切片,每镜最多一句

description 字段(给图像生成模型):
- 直接写**视觉画面**:谁在做什么 + 环境 + 光线 + 构图视角
- **环境描述**从原文细节 + Beat.setting_refs 的 name 含义 + LLM 常识里组合
  (本工程不另存场景视觉档案)。例如 setting_refs=["林家大厅"],你就要自己
  构想中式宅院大厅:朱漆梁柱、上首主位、雕花木椅、香炉等;原文若提到具体器物
  (如"玉石杯"、"袖口云剑纹")要保留进 description
- 出场人物用**正式 name**(从角色档案里取,不要别名);外貌细节参考人物档案
- 不要把 dialogue 文字塞进 description
- 同一集内对同一 setting 的画面描述要保持视觉一致(色调 / 布局 / 整体氛围)

setting 字段:
- **必须 1:1 取自 Beat.setting_refs 列表里的某个 name**(单一地点)
- 不要自由发挥写新地点,不要把 setting 内容塞进 description 重复

duration_sec 字段:
- = max(画面节奏需求, 念完 dialogue + voiceover 所需时间)
- 估算口径:对白 ~3.5 字/秒,旁白 ~3 字/秒
- 整集所有镜头时长之和 ≈ {target_duration} 秒

shot_type 字段:6 选 1 — 远景 / 全景 / 中景 / 近景 / 特写 / 大特写

────────────────────────────────────────────────
【绝对约束】
- 不杜撰人物或事件,所有内容必须有原文支撑
- dialogue / voiceover 必须来自原文(直接搬或浓缩),不要凭空创作
- 全集约 30%-50% 的镜头 voiceover 应为空;连续多镜不要镜镜念旁白
- characters 字段必须是输入"角色档案"里出现过的 name
- setting 字段必须是 Beat.setting_refs 里出现过的 name
- 一集所有镜头时长加起来要接近 {target_duration} 秒
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


def _build_user_prompt(
    beat: Beat,
    chars: List[Character],
    raw_text: str,
    target_duration: int,
) -> str:
    return (
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


def storyboard_beat(
    beat: Beat,
    characters: Dict[str, Character],
    batches: Dict[int, Batch],
    *,
    target_duration_sec: int = 180,
) -> Episode:
    """把一段 Beat 展开为一集 Episode(含 storyboards)。

    LLM 客户端由 ``agents.extract_storyboard.llm.get_llm()`` lazy 提供,
    caller 不需要传。
    """
    llm = get_llm()
    chars_list, raw_text = _gather_inputs(beat, characters, batches)

    system = SYSTEM_PROMPT.format(target_duration=target_duration_sec)
    user = _build_user_prompt(beat, chars_list, raw_text, target_duration_sec)

    logger.info(
        "storyboarder 第 %d 段(%s):人物=%d 场景 ref=%d 原文=%d 字",
        beat.index, beat.title, len(chars_list),
        len(beat.setting_refs), len(raw_text),
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
        beat_index=beat.index,
        storyboards=sb_pack.storyboards,
    )


__all__ = ["storyboard_beat", "SYSTEM_PROMPT"]
