"""单次 run 的流水线编排配置(Pydantic 契约)。

不含 LLM 配置 —— 后者归各 agent 自治(``agents/*/llm.json``)。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RunConfig(BaseModel):
    """单次 run 的流水线编排参数。"""

    input: str = Field(..., description="源小说路径(.txt / .epub)")
    output_dir: str = Field(..., description="输出根目录;每次 run 建时间戳子目录")
    max_batch_chars: int = Field(
        default=8_000, gt=0,
        description="每批送给 LLM 的字符上限",
    )
    max_total_chars: int = Field(
        default=0, ge=0,
        description="整本要分析的最大字数;超出尾部截断。0 = 不截断",
    )
    target_episode_duration_sec: int = Field(
        default=180, gt=0,
        description="每集目标时长(秒),供 episode_planner / storyboard 对齐",
    )
    recent_beats_window: int = Field(
        default=10, ge=0, le=50,
        description="beat agent prompt 里展示最近 N 段历史 beat 的窗口",
    )
    rewrite_window: int = Field(
        default=1, ge=0, le=10,
        description="beat agent 跨批续写窗口:每批 LLM 复述 / 修订末尾 K 段",
    )
    storyboard_prev_tail_window: int = Field(
        default=3, ge=0, le=10,
        description="storyboard agent 拿上集末尾 K 镜做集首承接的窗口",
    )


__all__ = ["RunConfig"]
