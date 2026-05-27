"""单次 run 的流水线编排配置(Pydantic 契约)+ JSON 加载入口。

不含 LLM 配置 —— 后者归各 agent 自治(``agents/<agent_name>/llm.json``)。
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class RunMode(str, Enum):
    """工作流跑到哪个阶段为止。

    深 mode 是浅 mode 的**严格超集**:跑 ``SCREENPLAY`` 时,中间产物跟
    单独跑 ``CHARACTER`` / ``BEAT`` / ``EPISODE`` 时是同一份。
    所以 debug / 试 prompt 时,可以选浅 mode 快速跑出对应阶段产物;
    需要完整剧本时切到 ``SCREENPLAY`` 即可。
    """

    CHARACTER  = "character"   # 仅跑人物档案
    BEAT       = "beat"         # + 剧情段切分
    EPISODE    = "episode"      # + 分集规划
    SCREENPLAY = "screenplay"   # + 逐集分镜(完整流水线)


_STAGE_ORDER = (
    RunMode.CHARACTER,
    RunMode.BEAT,
    RunMode.EPISODE,
    RunMode.SCREENPLAY,
)


def mode_includes(target: RunMode, stage: RunMode) -> bool:
    """跑 ``target`` mode 时,会跑到 ``stage`` 阶段吗?

    例:``mode_includes(RunMode.EPISODE, RunMode.BEAT) == True``。
    """
    return _STAGE_ORDER.index(target) >= _STAGE_ORDER.index(stage)


class RunConfig(BaseModel):
    """单次 run 的流水线编排参数。"""

    input: str = Field(..., description="源小说路径(.txt / .epub)")
    output_dir: str = Field(..., description="产物输出根目录;每次 run 建时间戳子目录")
    log_dir: str = Field(
        default="logs",
        description="LLM trace 和运行日志根目录;每次 run 建时间戳子目录",
    )
    mode: RunMode = Field(
        default=RunMode.SCREENPLAY,
        description=(
            "工作流跑到哪一阶段为止(character / beat / episode / screenplay);"
            "screenplay = 完整流水线,其他三档供 dev 早停 debug 用"
        ),
    )
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
        description="每集目标时长(秒),供 episode_planner / 分镜阶段对齐",
    )
    recent_beats_window: int = Field(
        default=10, ge=0, le=50,
        description="beat agent prompt 里展示最近 N 段历史 beat 的窗口",
    )
    rewrite_window: int = Field(
        default=1, ge=0, le=10,
        description="beat agent 跨批续写窗口:每批 LLM 复述 / 修订末尾 K 段",
    )
    shot_prev_tail_window: int = Field(
        default=3, ge=0, le=10,
        description="shot_director 拿上集末尾 K 镜做集首承接的窗口",
    )


_MODE_HINT = {
    RunMode.CHARACTER:  "仅人物档案",
    RunMode.BEAT:       "人物 + 剧情段",
    RunMode.EPISODE:    "人物 + 剧情段 + 分集规划",
    RunMode.SCREENPLAY: "完整流水线(+ 逐集分镜)",
}


def load_config(path: str | Path) -> RunConfig:
    """读取 + 校验 JSON 配置文件,并把内容打印出来便于人工确认。

    失败时抛带源路径的 ``ValueError``。
    """
    path = Path(path)
    try:
        cfg_dict = json.loads(path.read_text(encoding="utf-8"))
        cfg = RunConfig.model_validate(cfg_dict)
    except Exception as e:
        raise ValueError(f"加载 config 失败({path}):\n{e}") from e

    print("=" * 60)
    print(f"配置已加载({path}):")
    print(f"  输入:           {cfg.input}")
    print(f"  产物目录:       {cfg.output_dir}")
    print(f"  日志目录:       {cfg.log_dir}")
    print(f"  运行模式:       {cfg.mode.value}"
          f"({_MODE_HINT[cfg.mode]})")
    print(f"  批字数:         ≤ {cfg.max_batch_chars}")
    print(f"  全书字数限:     {cfg.max_total_chars or '无'}")
    print(f"  集时长目标:     {cfg.target_episode_duration_sec} 秒"
          f"(beat 切粒度 + 分镜阶段出镜数都按此对齐)")
    print(f"  近段窗口:       {cfg.recent_beats_window} 段"
          f"(beat agent prompt 里展示的最近段数)")
    print(f"  续写窗口:       {cfg.rewrite_window} 段"
          f"(beat agent 每批必须复述/修订的末尾 K 段)")
    print(f"  承接窗口:       {cfg.shot_prev_tail_window} 镜"
          f"(shot_director 每集开头看上集末 K 镜做画面承接)")
    print("  LLM:            各 agent 自治,见 agents/<agent_name>/llm.json")
    print("=" * 60)

    return cfg


__all__ = ["RunConfig", "RunMode", "mode_includes", "load_config"]
