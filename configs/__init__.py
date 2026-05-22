"""运行时配置目录。

* ``config.py`` —— ``RunConfig`` / ``LLMConfig``,即 ``*.json`` 文件的 Pydantic 契约
* ``*.json``   —— 实际配置文件,被 ``RunConfig.model_validate_json`` 解析

外部统一用 ``from configs import RunConfig, LLMConfig, load_config``。
"""

from __future__ import annotations

import json
from pathlib import Path

from configs.config import LLMConfig, RunConfig


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
    print(f"  输出:           {cfg.output_dir}")
    print(f"  批字数:         ≤ {cfg.max_batch_chars}")
    print(f"  全书字数限:     {cfg.max_total_chars or '无'}")
    print(f"  集时长目标:     {cfg.target_episode_duration_sec} 秒"
          f"(beat 切粒度 + storyboard 出镜数都按此对齐)")
    print(f"  近段窗口:       {cfg.recent_beats_window} 段"
          f"(beat agent prompt 里展示的最近段数)")
    print(f"  续写窗口:       {cfg.rewrite_window} 段"
          f"(beat agent 每批必须复述/修订的末尾 K 段)")
    print(f"  承接窗口:       {cfg.storyboard_prev_tail_window} 镜"
          f"(storyboard agent 每集开头看上集末 K 镜做画面承接)")
    print(f"  递归上限:       {cfg.langgraph_recursion_limit}"
          f"(LangGraph 父图安全网)")
    print("  LLM:            各 agent 自治,见 agents/extract_*/llm.json")
    print("=" * 60)

    return cfg


__all__ = ["LLMConfig", "RunConfig", "load_config"]
