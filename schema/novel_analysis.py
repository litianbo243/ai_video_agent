"""小说分析 agent 的 Pydantic 数据模型。

四类产出:
* **人物档案**(``Character``)—— 给 Stage 2 做"角色定调图"
* **场景档案**(``Setting``)—— 给 Stage 2 做"场景定调图",跨 Beat 复用
* **剧情大纲段**(``Beat``)—— 一段"有节奏感"的剧情(至少含一个小高潮),
  关联多个场景与人物,引用一个 batch
* **剧本分析**(``ScreenplayAnalysis``)—— 一段 → 一集,集内含 storyboards

数据流(每批 3 个抽取 agent + 跑完 N 个分镜 agent):
  for each batch:
    character_extractor → CharacterExtraction → state.merge_characters
    setting_extractor   → SettingExtraction   → state.merge_settings
    beat_extractor      → BeatExtraction      → state.merge_beats(batch_index=...)
       ↓
  for each beat:
    episode_storyboarder → Episode(含 storyboards)
       ↓
  ScreenplayAnalysis = logline + Episodes
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 文本批次(由 ``skills.batch_chapters.split_into_batches`` 产出,在
# ``BatchState.batches`` 里作为 LangGraph 状态的一部分流转)
# ---------------------------------------------------------------------------


class Batch(BaseModel):
    """一段文本(全局 1-based ``index``)。"""

    index: int = Field(..., description="全局批次序号(1-based)")
    text: str = Field(..., description="该批次的原文")

    @property
    def char_count(self) -> int:
        return len(self.text)

    def render_for_prompt(self) -> str:
        """LLM 直接可用的 prompt 正文。"""
        return self.text


# ---------------------------------------------------------------------------
# 人物
# ---------------------------------------------------------------------------


class CharacterDraft(BaseModel):
    """LLM 抽取人物的输出契约(无 ``index``)。

    LLM 通过 ``CharacterExtraction`` 输出此结构;``merge_characters`` 在合并时
    构造完整的 ``Character`` 并赋全局 ``index``。
    """

    name: str = Field(..., description="规范化的中文姓名,使用文中最常出现的写法")
    aliases: List[str] = Field(default_factory=list, description="别名 / 称谓 / 外号")
    appearance: str = Field(
        default="",
        description=(
            "整段外貌描写(150-300 字),含性别 + 年龄 + 身材 + 发型发色 + "
            "眼睛 + 服饰 + 标志特征 + 整体气质。给图像生成模型当 prompt 用。"
        ),
    )
    personality: str = Field(
        default="",
        description=(
            "整段性格分析(150-300 字),含行为模式 + 价值观 + 情感倾向 + 弧光走向。"
        ),
    )


class Character(CharacterDraft):
    """完整的人物档案(state 内部 + 落盘格式)。

    比 ``CharacterDraft`` 多一个 ``index`` —— 给下游(Stage 2/3 出图、外部数据库等)
    用的稳定数字主键,``merge_characters`` 自动赋值。
    """

    index: int = Field(default=0, description="全局编号(1-based,merge 时自动赋值)")


class CharacterRoster(BaseModel):
    characters: List[Character] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 场景(物理地点的视觉档案)
# ---------------------------------------------------------------------------


class SettingDraft(BaseModel):
    """LLM 抽取场景的输出契约(无 ``index``)。"""

    name: str = Field(
        ...,
        description='地点名(全局唯一,跨 Beat 复用),如"萧家祠堂"、"乌坦城广场"',
    )
    description: str = Field(
        default="",
        description=(
            "整段视觉环境描写(150-300 字),含建筑/布局/家具/光线/氛围/关键道具。"
            "给图像生成模型当 prompt 用。"
        ),
    )


class Setting(SettingDraft):
    """完整的场景档案(state 内部 + 落盘格式)。

    比 ``SettingDraft`` 多一个 ``index`` —— 稳定数字主键,``merge_settings`` 自动赋值。
    """

    index: int = Field(default=0, description="全局编号(1-based,merge 时自动赋值)")


class SettingCollection(BaseModel):
    settings: List[Setting] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 剧情大纲段
# ---------------------------------------------------------------------------


class BeatDraft(BaseModel):
    """LLM 抽取剧情段的输出契约(无 ``index`` / ``related_batches``,这两个 merge 自动填)。"""

    title: str = Field(..., description='剧情段标题,如"撕婚约"、"测验失败"')
    summary: str = Field(
        default="",
        description="本段剧情概要(2-4 句),要交代起承转合 + 关键节奏点(小高潮)",
    )
    setting_refs: List[str] = Field(
        default_factory=list,
        description="本段涉及的场景(Setting.name 列表),按时序排列,可多个",
    )
    character_refs: List[str] = Field(
        default_factory=list,
        description="本段涉及的人物(Character.name 列表)",
    )


class Beat(BeatDraft):
    """完整的剧情大纲段(state 内部 + 落盘格式)。

    比 ``BeatDraft`` 多两个 merge 时自动填的字段:
    * ``index``:全局编号(稳定主键)
    * ``related_batches``:涉及的 batch(支持跨批延续)

    一个 Beat 可以涉及多个 Setting / Character,**也可能跨多个 batch**(冲突跨批
    时由后续 batch 的 extractor 把当前 batch 追加到 ``related_batches``)。
    storyboarder 会遍历 ``related_batches`` 把所有相关原文拼起来。
    """

    index: int = Field(default=0, description="全局编号(1-based,merge 时自动赋值)")
    related_batches: List[int] = Field(
        default_factory=list,
        description=(
            "该段涉及的 batch 编号列表(1-based,按时序)。"
            "merge 时自动追加当前 batch;通常为 1-3 个 batch。"
        ),
    )


class BeatList(BaseModel):
    beats: List[Beat] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 剧本(分集 + 分镜)
# ---------------------------------------------------------------------------


class Storyboard(BaseModel):
    """单个分镜(对应一个具体镜头)。"""

    index: int = Field(..., description="在该集中的序号(1-based)")
    shot_type: str = Field(
        default="",
        description="镜头类型:远景 / 全景 / 中景 / 近景 / 特写 / 大特写",
    )
    description: str = Field(
        default="",
        description="画面描述。直接给图像生成模型当 prompt 用,要写明谁在做什么 + 光线 + 构图。",
    )
    characters: List[str] = Field(
        default_factory=list,
        description="出场人物的正式 name(对应 CharacterRoster)",
    )
    setting: str = Field(
        default="",
        description="本镜场景:引用 Setting.name(单一地点)",
    )
    dialogue: str = Field(
        default="",
        description=(
            "本镜内的角色台词原文(纯净 TTS 文本,不含动作描写)。"
            "长台词跨多镜时,每镜只放对应该画面的那一段。"
        ),
    )
    voiceover: str = Field(
        default="",
        description=(
            "本镜内的旁白原文(纯净 TTS 文本)。"
            "长旁白跨多镜时,按画面切点切片,每镜放对应片段。"
        ),
    )
    duration_sec: float = Field(
        default=0.0,
        description=(
            "本镜画面 hold 时长(秒),建议 3-15。"
            "= max(画面节奏需求, 念完 dialogue + voiceover 所需时间)。"
            "估算口径:对白 ~3.5 字/秒,旁白 ~3 字/秒。"
        ),
    )


class Episode(BaseModel):
    """一集(对应一段 Beat)。"""

    index: int = Field(..., description="集序号(1-based,= Beat.index)")
    title: str = Field(default="", description='本集标题,如"第一集 · 废柴觉醒"')
    synopsis: str = Field(default="", description="本集剧情概要(1-2 段)")
    beat_index: int = Field(default=0, description="对应 Beat.index")
    storyboards: List[Storyboard] = Field(
        default_factory=list, description="本集所有分镜(按时序)"
    )


class ScreenplayAnalysis(BaseModel):
    """剧本分析:全书 logline + 所有分集(含分镜)。"""

    title: str = Field(default="")
    logline: str = Field(default="", description="一句话电梯陈述")
    episodes: List[Episode] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 跨批次的滚动状态(同时是 LangGraph 的 state)
# ---------------------------------------------------------------------------


class CharacterExtraction(BaseModel):
    """character_extractor 子-agent 的输出(LLM 不输出 index)。"""

    new_or_updated_characters: List[CharacterDraft] = Field(default_factory=list)


class SettingExtraction(BaseModel):
    """setting_extractor 子-agent 的输出(LLM 不输出 index)。"""

    new_or_updated_settings: List[SettingDraft] = Field(default_factory=list)


class BeatExtraction(BaseModel):
    """beat_extractor 子-agent 的输出(LLM 不输出 index / related_batches)。

    LLM 在每批可以做两件事:
    * ``new_beats``:本批中**新起**的剧情段(典型情况)
    * ``extended_beat_indices``:本批是**已有段的延续**(跨批高潮 / 长冲突),
      给出对应 Beat.index;merge 时自动把本批 batch 追加到它们的 ``related_batches``,
      不新建 Beat。
    """

    new_beats: List[BeatDraft] = Field(default_factory=list)
    extended_beat_indices: List[int] = Field(
        default_factory=list,
        description="本批延续了哪几个已有段(给已有段的 index)",
    )


class BatchState(BaseModel):
    """跨 batch 传递的完整滚动状态(也是 LangGraph 的 state 对象)。"""

    input_path: str = ""
    txt_path: str = ""
    output_dir: str = ""
    title: str = ""
    max_batch_chars: int = 8_000
    max_total_chars: int = 0  # 0 = 不截断
    target_episode_duration_sec: int = 180

    batches: List[Batch] = Field(
        default_factory=list,
        description="所有文本批次;storyboarder 按 Beat.related_batches 反查原文",
    )
    cursor: int = Field(default=0, description="下一个要分析的 batch 在 batches 列表中的下标")
    total_chars: int = 0

    characters: Dict[str, Character] = Field(
        default_factory=dict, description="规范化人名 -> Character"
    )
    settings: Dict[str, Setting] = Field(
        default_factory=dict, description="地点名 -> Setting,跨 Beat 复用"
    )
    beats: List[Beat] = Field(default_factory=list)

    last_completed_batch: int = 0

    final_report: Optional["FinalReport"] = Field(
        default=None, description="finalize 节点生成,write 节点消费"
    )
    output_paths: Dict[str, str] = Field(
        default_factory=dict,
        description="write 节点写入完成后填充:产物种类 -> 绝对路径",
    )

    # ------------------------------------------------------------------
    # 三个独立的 merge 方法,跟三个子-agent 的输出一一对应
    # ------------------------------------------------------------------

    def merge_characters(self, delta: CharacterExtraction) -> None:
        """合并人物增量:LLM 输出 ``CharacterDraft``,merge 转 ``Character`` 并赋 index。

        同名 → 融合(aliases 取并集,长描述非空才覆盖);新名 → 新增 + 赋全局 index。
        """
        for draft in delta.new_or_updated_characters:
            existing = self.characters.get(draft.name)
            if existing is None:
                ch = Character(**draft.model_dump(), index=len(self.characters) + 1)
                self.characters[draft.name] = ch
                continue
            existing.aliases = sorted(set(existing.aliases) | set(draft.aliases))
            if draft.appearance:
                existing.appearance = draft.appearance
            if draft.personality:
                existing.personality = draft.personality

    def merge_settings(self, delta: SettingExtraction) -> None:
        """合并场景增量:LLM 输出 ``SettingDraft``,merge 转 ``Setting`` 并赋 index。"""
        for draft in delta.new_or_updated_settings:
            existing = self.settings.get(draft.name)
            if existing is None:
                s = Setting(**draft.model_dump(), index=len(self.settings) + 1)
                self.settings[draft.name] = s
                continue
            if draft.description:
                existing.description = draft.description

    def merge_beats(self, delta: BeatExtraction, batch_index: int) -> None:
        """追加新段 + 把当前 batch 追加到被延续段的 ``related_batches``。

        LLM 输出 ``BeatDraft``,merge 转 ``Beat`` 并赋 index / related_batches。
        延续段:把本批 batch 加到对应 Beat 的 ``related_batches``(去重)。
        """
        # 1) 已有段的延续:把当前 batch 加到它们的 related_batches
        for idx in delta.extended_beat_indices:
            i = idx - 1  # 1-based → 0-based
            if 0 <= i < len(self.beats):
                if batch_index not in self.beats[i].related_batches:
                    self.beats[i].related_batches.append(batch_index)

        # 2) 新起段:Draft → Beat,赋全局 index,补 related_batches
        for draft in delta.new_beats:
            beat = Beat(
                **draft.model_dump(),
                index=len(self.beats) + 1,
                related_batches=[batch_index],
            )
            self.beats.append(beat)

        self.last_completed_batch = batch_index


# ---------------------------------------------------------------------------
# 最终聚合报告
# ---------------------------------------------------------------------------


class ReportMeta(BaseModel):
    source_path: str
    txt_path: str = ""
    title: str = ""
    total_chars: int = 0
    batch_count: int = 0
    max_batch_chars: int = 0
    max_total_chars: int = 0
    llm_base_url: str = ""
    llm_model: str = ""


class FinalReport(BaseModel):
    """最终产出:剧本 + 人物 + 场景 + 节拍 + 元信息,平级。"""
    screenplay: ScreenplayAnalysis
    characters: CharacterRoster
    settings: SettingCollection
    beats: BeatList
    meta: ReportMeta


BatchState.model_rebuild()
