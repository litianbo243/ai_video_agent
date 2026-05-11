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
    print(f"  输入:       {cfg.input}")
    print(f"  输出:       {cfg.output_dir}")
    print(f"  批字数:     ≤ {cfg.max_batch_chars}")
    print(f"  全书字数限: {cfg.max_total_chars or '无'}")
    print(f"  集时长目标: {cfg.target_episode_duration_sec} 秒(仅 storyboard 用)")
    print(f"  模型:       {cfg.llm.model}")
    print(f"  端点:       {cfg.llm.base_url}")
    if cfg.llm.api_key_env:
        print(f"  Key 环境变量: {cfg.llm.api_key_env}")
    print(f"  Key 是否就位: {'是' if cfg.llm.api_key else '否(本地端点可能不需要)'}")
    print(f"  采样温度:    {cfg.llm.temperature}")
    print("=" * 60)

    return cfg


__all__ = ["LLMConfig", "RunConfig", "load_config"]
