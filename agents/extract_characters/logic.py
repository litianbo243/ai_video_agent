"""单批人物增量处理:LLM 抽取 + delta 合并(全部 in-place)。

主 API:

* ``extract_for_batch(batch, known)`` —— 1 次 LLM 调用 + 自动合并
  返回的 ``CharacterExtraction`` 仅给 trace / debug 看,正常 caller 无需读
  LLM 客户端从同包的 ``llm`` 模块按需 lazy 取,不再由 caller 注入

底层 API(单测 / notebook 用):

* ``merge_delta(known, delta)`` —— 单独跑合并(给手工构造的 delta 用)
* 测试时想 mock LLM:``from agents.extract_characters import set_llm; set_llm(fake)``

workflow 只负责编排(batch 循环 + LangGraph 节点),不写合并语义。

prompt 上下文设计:**全员详档**

* 每次都把已知人物的**完整详档**(name + aliases + appearance + background + personality)
  全部塞进 prompt
* 一本小说主要人物通常 20-30 人,详档总量 ~5-10K token,Grok 4.3 上下文充裕
* 不做 substring scan 召回,**消除漏召回 / 黑名单维护 / 误升格** 等一类 bug
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

from llm.agent_llm import make_agent_llm_manager
from skills.batch_chapters import Batch
from agents.extract_characters.schema import Character, CharacterExtraction

logger = logging.getLogger(__name__)


get_llm, set_llm, set_trace_dir = make_agent_llm_manager(
    agent_name="extract_characters",
    config_path=Path(__file__).parent / "llm.json",
)


SYSTEM_PROMPT = """\
你是中文小说人物抽取助手。增量分析:看一批正文 + 已知人物详档,输出本批 delta。

# 全局原则(适用所有字段)

- 增量语义:appearance / background / personality 留空 = 沿用旧值不更新;
  非空 = 完整新版整段覆盖(不要差量)
- **跨章节稳定 / 故事进度态**:appearance / background / personality 表示
  **故事推进到本批末尾为止的最新状态**(不是单个场景的瞬时切片,也不是某次剧情事件复盘);
  全局**禁用时间锚点**(「今年 / 这天 / 此刻 / 刚结婚两个月」)
- **倒叙 / 插叙处理**:本批若描写角色早期片段(童年 / 大学时 / 多年前)
  → **不更新** appearance / personality 当前态;早期信息归入 background,
  用「曾」「年轻时」「早年」「大学时期」等时态表达
- 长度随原文素材伸缩:主角写到完整画像,配角写到位即止;原文没素材就留空,
  **不杜撰、不补脑**,后续 batch 看到再补
- **跨批人物身份**:输出前先扫详档(name + aliases),已存在的人物**沿用**不要重建
  · 详档里能匹配到 → 用详档里的 name(pre_name 留空)
  · 本批确认了详档某人的真名(详档里 name 是临时称呼,如「X 老师 / 老 X / X 总」
    之类,本批原文出现该人完整本名)→ name 升格为本名,pre_name 填详档里
    的旧 name(原称呼)
  · 全新人物 → 直接出新 name,pre_name 留空
- 路人(单次出场、无戏剧作用)不输出
- **人际关系信息**(与详档里其他角色的师生 / 同窗 / 上下级 / 家庭 / 亲密关系等)
  只写在 background 里,不混进 personality / appearance;**重点抓与已知角色的关系**,
  能写明对象 name 的就写明(如「林望的大学导师」「与张三同窗」「钟离的下属」)

# 字段(每个出场人物)

- name: 规范化中文姓名(主名;别号进 aliases)

- pre_name: 参全局原则「跨批人物身份」决定填什么

- aliases: 本批原文出现过的别名 / 称谓 / 外号,合并入此人 aliases
  ❌ 不要塞通用代词「他 / 她 / 那家伙」

- appearance: 跨场景稳定的视觉画像,客观第三人称,给图像生成模型用,上限约 300 字。
  连贯叙述不分点;
  **风格层抽象,不抄单场景具体穿搭 / 妆容 / 配饰**(那是后续按场景生成的事):
    ❌「米黄毛衣 + 米色过膝裙 + 白色高跟瓢鞋」(单场景穿搭被照抄)
    ✓「偏好干练优雅风格,常穿衬衫配窄裙、平底鞋」(风格层抽象)
  更新已有 appearance → **保留稳定项**,只改 / 增原文新提供的新信息。
  覆盖项(原文有就写):性别 / 年龄段 / 身材体型 / **发型发色**(角色识别核心,
  有就一定写) / 脸型肤色 / 眉眼唇鼻 / 服饰风格 / 标志特征 / 气质

- background: 整段背景档案,上限约 200 字。
  覆盖项:职业 / 出身 / 关键经历 / 社会身份 / **人际关系**
  (与详档里其他角色的家庭 / 同窗 / 师生 / 上下级 / 亲密关系,有就写明对象 name)。
  ⚠️ 履历类信息只放这里,不要混进 appearance 或 personality:
     ❌ appearance 写「前军人转业」「曾任校长」 ← 应归 background
     ❌ personality 写「升职时依赖上级推荐」 ← 应归 background(履历事件)

- personality: 整段性格画像,**模式级而非事件级**,上限约 300 字。
  把一次性情节抽象成**重复模式**,不出现其他角色具体姓名
  (用「权威人物」「亲密伴侣」「陌生人」等关系类型代替):
    ❌「被某位上级在月度会议批评后,回家哭一晚,次日辞职」(单次事件复盘)
    ✓「被批评时表面顺从,事后脑内反复重演找反驳点,下次仍回避正面冲突」
  覆盖项:行为模式(日常 / 高压 / 独处下的默认反应) / 价值观 / 情感倾向
  (对陌生人 / 权威 / 亲密关系的默认反应) / 弧光(整本书的转变方向,
  **强制「X → Y」格式**,如「天真 → 世故」「依附 → 独立」)
  避免强价值判断词(「圣母 / 烂人 / 罪人」),用中性描述

只输出一个 JSON 对象,符合 CharacterExtraction schema。
"""


def _render_full_profile(ch: Character) -> str:
    section = [f"### {ch.name}"]
    if ch.aliases:
        section.append(f"别名: {', '.join(ch.aliases)}")
    if ch.appearance:
        section.append(f"外貌: {ch.appearance}")
    if ch.background:
        section.append(f"背景: {ch.background}")
    if ch.personality:
        section.append(f"性格: {ch.personality}")
    return "\n".join(section)


def _build_user_prompt(
    batch: Batch, known: Dict[str, Character], title: str,
) -> str:
    book_title = title or "(未提供书名)"
    batch_text = batch.render_for_prompt()
    profiles = (
        "\n\n".join(_render_full_profile(c) for c in known.values())
        if known
        else "(尚无已知人物;本批可能全是新角色)"
    )
    return (
        f"书名: {book_title}\n"
        f"批次序号: 第 {batch.index} 批\n"
        f"批次字数: 约 {batch.char_count}\n\n"
        f"=== 全部已知人物详档(共 {len(known)} 人)===\n{profiles}\n\n"
        f"=== 本批次正文 ===\n{batch_text}\n\n"
        f"=== 任务 ===\n请按 JSON Schema 输出 CharacterExtraction。"
    )


def extract_for_batch(
    batch: Batch,
    known: Dict[str, Character],
    *,
    title: str = "",
) -> CharacterExtraction:
    """完整处理一批:LLM 抽取 → 合并入 ``known``(in-place)。

    LLM 客户端由 ``agents.extract_characters.llm.get_llm()`` lazy 提供
    (配置在同包 ``llm.json``),caller 不需要传。

    返回 LLM 原始 delta,给 logger / trace / debug 看;默认调用方无需读它。
    若想跳过自动合并(notebook 调试场景),可单独调 ``merge_delta``。
    """
    llm = get_llm()
    user_prompt = _build_user_prompt(batch, known, title)
    logger.info(
        "character_extractor 第 %d 批(已知 %d 人),%s @ %s",
        batch.index, len(known), llm.model, llm.base_url,
    )
    delta = llm.chat_json(SYSTEM_PROMPT, user_prompt, CharacterExtraction)
    logger.info(
        "character_extractor 第 %d 批产出:%d 人(新增/更新)",
        batch.index, len(delta.new_or_updated_characters),
    )
    merge_delta(known, delta)
    return delta


def merge_delta(known: Dict[str, Character], delta: CharacterExtraction) -> None:
    """把本批 LLM delta in-place 合并入 ``known`` 名册。

    **匹配逻辑**(LLM 显式声明驱动,无代码端反向索引):

    * ``draft.pre_name`` 非空 → 按 ``pre_name`` 找 known(LLM 声明这是更新场景)。
      若 ``draft.name != pre_name`` → 触发 **name 升格**:新 name 替换 canonical name,
      旧 name 自动进 aliases,同步 ``known`` 字典 key。
    * ``draft.pre_name`` 为空 + ``draft.name`` 命中 known → 纯字段更新,不改 name。
    * 都没命中 → 当新人物建。

    **字段更新**:appearance / background / personality 非空才覆盖
    (空 = LLM 表示「本批没新增信息,沿用旧值」)。aliases 始终取并集。

    **空壳保护**:新人物若 appearance / background / personality / aliases 全空 → 跳过 + warn。

    **顺序保证**:函数末尾按 ``index`` 升序重排 ``known`` 的 dict 顺序;
    name 升格走 ``pop + 新 key`` 会把节点甩到 dict 末尾,不 re-sort 的话
    ``list(known.values())`` 会乱序。caller 拿到 known 直接遍历就是有序的。

    **失败模式**:LLM 漏判同人(没填 pre_name,直接用新 name)→ 出现重复条目。
    通过 prompt 教 LLM 「先扫详档判断身份」防御;若仍漏判,看 trace 调整 prompt。
    """
    for draft in delta.new_or_updated_characters:
        lookup_key = draft.pre_name or draft.name
        existing = known.get(lookup_key)

        if existing is None:
            if draft.pre_name:
                logger.warning(
                    "[character_merge] draft 声明 pre_name=%r 但 known 里找不到,"
                    "当新人物处理 name=%r",
                    draft.pre_name, draft.name,
                )
            if (
                not draft.appearance
                and not draft.background
                and not draft.personality
                and not draft.aliases
            ):
                logger.warning(
                    "[character_merge] 跳过空壳新人物 name=%r(无外貌/背景/性格/别名)",
                    draft.name,
                )
                continue
            ch = Character(
                **draft.model_dump(exclude={"pre_name"}),
                index=len(known) + 1,
            )
            known[ch.name] = ch
            continue

        old_name = existing.name
        new_name = draft.name
        union_aliases = set(existing.aliases) | set(draft.aliases)

        if new_name != old_name:
            union_aliases.add(old_name)
            union_aliases.discard(new_name)
            existing.name = new_name
            known.pop(old_name, None)
            known[new_name] = existing
            logger.info(
                "[character_merge] name 升格:%r → %r(旧 name 进 aliases)",
                old_name, new_name,
            )

        existing.aliases = sorted(union_aliases)
        if draft.appearance:
            existing.appearance = draft.appearance
        if draft.background:
            existing.background = draft.background
        if draft.personality:
            existing.personality = draft.personality

    ordered = sorted(known.items(), key=lambda kv: kv[1].index)
    known.clear()
    known.update(ordered)


__all__ = ["extract_for_batch", "merge_delta", "SYSTEM_PROMPT"]
