"""单次 run 的流水线编排配置(Pydantic 契约)。

JSON 文件长这样::

    {
      "input": "inputs/your_novel.epub",
      "output_dir": "outputs",
      "max_batch_chars": 8000,
      "max_total_chars": 0,
      "target_episode_duration_sec": 180,
      "recent_beats_window": 10
    }

``RunConfig`` 只管流水线编排参数,**不含 LLM 配置** —— 后者下放到各 agent
自治(每个 agent 在 ``agents/extract_*/llm.json`` 里独立定义,契约见
``llm.llm_config.LLMConfig``)。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RunConfig(BaseModel):
    """单次 run 的完整配置(只含流水线编排参数,不含 LLM —— 后者归各 agent 自治)。"""

    input: str = Field(
        ...,
        description="源小说路径(.txt / .epub)",
    )
    output_dir: str = Field(
        ...,
        description="输出根目录;每次 run 会在它下面建一个时间戳子目录",
    )
    max_batch_chars: int = Field(
        default=8_000,
        gt=0,
        description="每批送给 LLM 的字符上限(用作 split_into_batches 的预算)",
    )
    max_total_chars: int = Field(
        default=0,
        ge=0,
        description=(
            "整本小说要分析的最大字数;超出从尾部直接截断。0 = 不截断(默认)。"
            "用作快速试跑只取前 N 万字看效果。"
        ),
    )
    target_episode_duration_sec: int = Field(
        default=180,
        gt=0,
        description=(
            "期望的每集短视频时长(秒,默认 180 = 3 分钟)。两处用到:"
            "(1) beat agent 切粒度时按此对齐(短视频 → 紧凑;长视频 → 单集多承载剧情);"
            "(2) storyboard agent 据此排镜数 / 单镜时长。"
        ),
    )
    recent_beats_window: int = Field(
        default=10,
        ge=0,
        le=50,
        description=(
            "beat agent 在 prompt 里展示「此前最近 N 段大纲」的窗口大小,"
            "用于跨 batch 的剧情接续判断。"
            "默认 10。本地小模型 ctx 紧时调低(如 5),大模型 / 长上下文可调高。"
            "0 = 不展示历史 beats(冷启动 / 独立 batch 时用)。"
        ),
    )
    rewrite_window: int = Field(
        default=1,
        ge=0,
        le=10,
        description=(
            "beat agent 跨批续写窗口:每批 LLM 必须复述/修订「末尾 K 段」"
            "(原样或修订均可),其后追加本批新起段。"
            "0 = 关闭续写(纯增量,长戏剧段会被批边界切碎);"
            "1 = 默认(LLM 每批重看上批末段,自然接续);"
            ">=2 = 给 LLM 更大修订空间,token 成本随 K 线性增长。"
            "同时是 storyboard 的「冷却期」:末尾 K 段还在可改窗口内,不立即送 storyboard。"
        ),
    )
    storyboard_prev_tail_window: int = Field(
        default=3,
        ge=0,
        le=10,
        description=(
            "storyboard agent 的「上集承接窗口」:每集开始时把上集末尾 K 镜的"
            "画面 / 服装 / 道具状态喂给 LLM 做集首承接镜头。"
            "0 = 关闭(集间无视觉承接,每集独立定场);"
            "3 = 默认(覆盖钩子镜 + 前 1-2 个铺垫镜,足够画面连续);"
            ">=5 = 给 LLM 更长视觉记忆,token 成本随 K 线性增长。"
        ),
    )


__all__ = ["RunConfig"]
