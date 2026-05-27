"""单批人物增量处理:LLM 抽取 + delta 合并(全部 in-place)。

主 API:

* ``run_for_batch(batch, known)`` —— 1 次 LLM 调用 + 自动合并
  返回的 ``CharacterExtraction`` 仅给 trace / debug 看,正常 caller 无需读
  LLM 客户端从同包的 ``llm`` 模块按需 lazy 取,不再由 caller 注入

底层 API(单测 / notebook 用):

* ``merge_delta(known, delta)`` —— 单独跑合并(给手工构造的 delta 用)
* 测试时想 mock LLM:``from agents.character_profiler import set_llm; set_llm(fake)``

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
from typing import Dict, List, NamedTuple, Tuple

from llm.agent_llm import make_agent_llm_manager
from skills.batch_chapters import Batch
from schemas.character import Character, CharacterExtraction


class CharacterProfileResult(NamedTuple):
    """``run_for_batch`` 的返回值。

    可解构(``delta, renames = result``),也可字段访问。
    """

    delta: CharacterExtraction
    """LLM 原始 delta(给 trace / debug;正常 caller 无需读)。"""

    renames: List[Tuple[str, str]]
    """本批发生的 name 升格事件 ``[(old_name, new_name), ...]``,
    供 caller 同步下游已生成内容(典型:``beat_segmenter`` 的 ``character_refs``)。"""

logger = logging.getLogger(__name__)


get_llm, set_llm, set_trace_dir = make_agent_llm_manager(
    agent_name="character_profiler",
    config_path=Path(__file__).parent / "llm.json",
)


SYSTEM_PROMPT = """\
你是中文小说人物抽取助手,负责从小说中提取人物大纲(档案)。
工作方式:增量分析 —— 每次看一批正文 + 已知人物详档,输出本批 delta。

# 全局原则(适用所有字段)

- 增量语义:appearance / background / personality / arc
  · 留空 = 沿用旧值不更新(本批没新信息时用)
  · 非空 = **完整新版整段覆盖**:必须包含**旧值所有有效信息 + 本批新增 / 修订**
  · ⚠️ **最常见错误**:把"非空"理解为"只写本批新增" → 导致旧档大段丢失
    场景:旧字段已有 3-4 项关键信息,本批揭示 1 项新信息
    ❌ LLM 只输出本批新增的那 1 项(漏写旧档,严重丢档)
    ✓ LLM 输出「旧 1 / 旧 2 / 旧 3 / 旧 4;新 5」—— 旧值全部保留 + 末尾追加新值
  · 写出来的整段必须是"读完后人物档案完整、不依赖记忆补全"
  · ⚠️ **arc 字段特殊 —— append-only 时序结构**:它不是"模式描述",而是
    按时序排列的事件锚点串。本批输出必须是「**旧 arc 全部锚点(原序、原文)
    + 本批新锚点(末尾追加)**」。旧锚点已"盖章",**严禁删除 / 改写 / 精炼**,
    即便本批没提到也照抄。详见下方 arc 字段段。
- **字段职责分工**(强约束,不要越界):
  · appearance / background / personality —— **全局静态信息**(整本书的稳定属性 /
    默认态);**不写剧情中的变化 / 累积**
  · arc —— **剧情弧光**(性格 / 状态在剧情中的转变);专门承担"变化"描述
  互斥:在 personality 里出现「转为」「变得」「从 X 到 Y」等变化措辞 = 越界,
  应迁移到 arc
- **倒叙 / 插叙处理**:本批若描写角色早期片段(童年 / 大学时 / 多年前)
  → 早期信息归入 background,用「曾」「年轻时」「早年」「大学时期」等时态表达;
  不更新 appearance / personality 当前态
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
  只写在 background 里,不混进 personality / appearance / arc;
  **重点抓与已知角色的关系**,能写明对象 name 的就写明
  (如「林望的大学导师」「与张三同窗」「钟离的下属」)

# 字段(每个出场人物)

- name: 规范化中文姓名(主名;别号进 aliases)

- pre_name: 参全局原则「跨批人物身份」决定填什么

- aliases: 本批原文出现过的别名 / 称谓 / 外号,合并入此人 aliases
  ❌ 不要塞通用代词「他 / 她 / 那家伙」

- appearance: 跨场景稳定的视觉画像,客观第三人称,给图像生成模型用,上限约 300 字。
  连贯叙述不分点。影视化"一套造型"逻辑,整本书读者识别同一人。
  · 风格层抽象,不抄单场景具体穿搭 / 妆容 / 配饰(那是后续按场景生成的事):
    ❌「米黄毛衣 + 米色过膝裙 + 白色高跟瓢鞋」(单场景穿搭被照抄)
    ✓「偏好干练优雅风格,常穿衬衫配窄裙、平底鞋」(风格层抽象)
  · **不写剧情中的外貌变化**(剪短发 / 染色 / 怀孕 / 留疤 / 减肥)—— 这些归下游
    画面描述,不进 character 字段
  · 更新已有 appearance → 保留稳定项,只改 / 增原文新提供的新信息
  覆盖项(原文有就写):性别 / 年龄段 / 身材体型 / **发型发色**(角色识别核心,
  有就一定写) / 脸型肤色 / 眉眼唇鼻 / 服饰风格 / 标志特征 / 气质

- background: 整段背景档案,上限约 200 字。
  覆盖项:身世 / 出身 / 教育 / 职业 / 社会身份 / 关键人际关系
  (与详档里其他角色的家庭 / 同窗 / 师生 / 上下级 / 亲密关系,有就写明对象 name)。
  · ⚠️ **不写剧情中的事件累积**(婚姻状况变化 / 工作变动 / 亲友亡故等)——
    这些事件 beat 里已记录,事件造成的性格影响归 arc
    ❌「曾结婚后离婚」「父亲在剧情中段去世」(事件累积)
    ✓「父亲早年病故,与母亲相依为命」(剧情开始前已成立的设定)

- personality: 整段性格画像,上限约 300 字。
  描述角色**长期一以贯之**的行为模式 / 价值观 / 情感倾向(角色的"基础人设")。
  · 把一次性情节抽象成**重复模式**,不出现其他角色具体姓名
    (用「权威人物」「亲密伴侣」「陌生人」等关系类型代替):
    ❌「被某位上级在月度会议批评后,回家哭一晚,次日辞职」(单次事件复盘)
    ✓「被批评时表面顺从,事后脑内反复重演找反驳点,下次仍回避正面冲突」
  · 若角色性格在不同阶段差异巨大 → personality 只写"主轴 / 基础特质",
    具体变化全部归 arc
  覆盖项:行为模式(日常 / 高压 / 独处下的默认反应) / 价值观 / 情感倾向
  (对陌生人 / 权威 / 亲密关系的默认反应)
  避免强价值判断词(「圣母 / 烂人 / 罪人」),用中性描述

- arc: 剧情弧光,**累积时序结构(append-only)**,上限约 280 字。
  锚点串:「锚点 1;锚点 2;锚点 3 ...」按事件发生时序排列。

  · 没有显著弧光的角色 → 留空,**不要为了填空强写**
  · 单个锚点写法:「在 [事件 / 节点] 后,从 X 转为 Y」,事件锚点用
    **自然语言短语**,不要章节号 / 日期:
    ✓「在被亲密伴侣背叛后,从天真信赖转为冷峻戒备」
    ✓「在母亲病故后,从依附家庭转为独立扛事」
    ❌「第 8 章后变得世故」(用章节号当锚点)
    ❌「中期开始变化」(锚点不可识别)
  · 锚点要在 beat 摘要里**找得到 / 对得上**(用关键事件名 / 关系节点)
  · 锚点描述本身仍是**模式级**(避免单次事件复盘),只是带了事件锚点:
    ❌「在 [关键事件] 后,某天大哭一场,撕了所有合照」(单次事件复盘)
    ✓「在 [关键事件] 后,从对亲密关系无保留,转为审慎保留底线」(模式级)

  · ⚠️ **更新规则(最关键)**:旧锚点已"盖章",新批次输出 arc 时必须**完整
    保留旧 arc 全部锚点(原文照抄,保持原序),再在末尾追加本批新锚点**。
    **严禁**:删除旧锚点 / 改写旧锚点 / 把多个旧锚点精简成一个 / "本批不涉及"
    就漏掉旧锚点。即便你觉得旧锚点冗余,也照抄。

    场景:旧 arc 已有 2 个锚点(全书起点 + 中段转折),本批揭示第 3 个锚点
    ❌ LLM 只输出本批的新锚点(漏写之前的全书起点,严重丢档)
    ✓ LLM 输出「旧锚点 1(全书起点);旧锚点 2(中段);新锚点 3(本批)」
      —— 按时序 append,完整时序链

  · 长度策略:接近 280 字上限时,**不要删旧锚点**,而是**精炼新锚点的措辞**
    (保锚点数量,压每个锚点字数)

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
    if ch.arc:
        section.append(f"弧光: {ch.arc}")
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


def run_for_batch(
    batch: Batch,
    known: Dict[str, Character],
    *,
    title: str = "",
) -> CharacterProfileResult:
    """完整处理一批:LLM 抽取 → 合并入 ``known``(in-place)。

    LLM 客户端由 ``agents.character_profiler.llm.get_llm()`` lazy 提供
    (配置在同包 ``llm.json``),caller 不需要传。

    返回 ``CharacterProfileResult(delta, renames)``:

    * ``delta`` —— LLM 原始 delta,给 logger / trace / debug 看;默认 caller 无需读。
    * ``renames`` —— 本批发生的 name 升格事件 ``[(old_name, new_name), ...]``。
      caller 应拿这些事件回扫下游已经生成的内容(典型:``beat_segmenter`` 已经
      锁住的 ``Beat.character_refs`` 还指向旧 name)。空列表 = 本批无升格。

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
    renames = merge_delta(known, delta)
    return CharacterProfileResult(delta=delta, renames=renames)


def merge_delta(
    known: Dict[str, Character], delta: CharacterExtraction,
) -> List[Tuple[str, str]]:
    """把本批 LLM delta in-place 合并入 ``known`` 名册。

    **匹配逻辑**(LLM 显式声明驱动,无代码端反向索引):

    * ``draft.pre_name`` 非空 → 按 ``pre_name`` 找 known(LLM 声明这是更新场景)。
      若 ``draft.name != pre_name`` → 触发 **name 升格**:新 name 替换 canonical name,
      旧 name 自动进 aliases,同步 ``known`` 字典 key。
    * ``draft.pre_name`` 为空 + ``draft.name`` 命中 known → 纯字段更新,不改 name。
    * 都没命中 → 当新人物建。

    **字段更新**:appearance / background / personality / arc 非空才覆盖
    (空 = LLM 表示「本批没新增信息,沿用旧值」)。aliases 始终取并集。

    **空壳保护**:新人物若 appearance / background / personality / arc / aliases 全空 → 跳过 + warn。

    **顺序保证**:函数末尾按 ``index`` 升序重排 ``known`` 的 dict 顺序;
    name 升格走 ``pop + 新 key`` 会把节点甩到 dict 末尾,不 re-sort 的话
    ``list(known.values())`` 会乱序。caller 拿到 known 直接遍历就是有序的。

    **失败模式**:LLM 漏判同人(没填 pre_name,直接用新 name)→ 出现重复条目。
    通过 prompt 教 LLM 「先扫详档判断身份」防御;若仍漏判,看 trace 调整 prompt。

    Returns:
        本批发生的 name 升格事件 ``[(old_name, new_name), ...]``;空列表 = 本批无升格。
        caller 应拿来回扫下游已生成的内容(beats / shots 里残留旧 name 的引用)。
    """
    renames: List[Tuple[str, str]] = []
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
                and not draft.arc
                and not draft.aliases
            ):
                logger.warning(
                    "[character_merge] 跳过空壳新人物 name=%r(无外貌/背景/性格/弧光/别名)",
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
            renames.append((old_name, new_name))
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
        if draft.arc:
            existing.arc = draft.arc

    ordered = sorted(known.items(), key=lambda kv: kv[1].index)
    known.clear()
    known.update(ordered)

    return renames


__all__ = [
    "CharacterProfileResult",
    "SYSTEM_PROMPT",
    "run_for_batch",
    "merge_delta",
]
