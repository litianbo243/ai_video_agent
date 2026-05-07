"""运行时配置 —— config 文件的数据契约。

config 文件长这样::

    {
      "input": "input/your_novel.epub",
      "output_dir": "output",
      "max_batch_chars": 8000,
      "max_total_chars": 0,
      "llm": {
        "base_url": "https://api.siliconflow.cn/v1",
        "model": "Pro/deepseek-ai/DeepSeek-V3.2",
        "api_key_env": "SILICONFLOW_API_KEY",
        "temperature": 0.2
      }
    }

API key 不放在 JSON 里,走 .env / 环境变量。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, SecretStr, model_validator


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

_dotenv_loaded = False


def _ensure_dotenv_loaded() -> None:
    """加载 .env 文件。"""
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    _dotenv_loaded = True
    repo_root = Path(__file__).resolve().parent.parent
    candidates = [Path(".env"), repo_root / ".env"]
    for path in candidates:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)
        break


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------


class LLMConfig(BaseModel):
    """LLM 调用配置。任何 OpenAI 兼容协议的端点都用同一套字段定位。"""

    base_url: str = Field(
        ...,
        description="OpenAI 兼容端点 URL(如 https://api.siliconflow.cn/v1)",
    )
    model: str = Field(
        ...,
        description="模型名(如 Pro/deepseek-ai/DeepSeek-V3.2)",
    )
    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="采样温度",
    )
    api_key_env: Optional[str] = Field(
        default=None,
        description="存放 API key 的环境变量名(如 SILICONFLOW_API_KEY);留空则不读 env",
    )
    api_key: Optional[SecretStr] = Field(
        default=None,
        repr=False,
        exclude=True,
        description="运行时 API key,从 env 自动注入,不要写进 JSON",
    )

    @model_validator(mode="after")
    def _populate_api_key_from_env(self) -> "LLMConfig":
        """从 ``api_key_env`` 指定的环境变量读 api_key,没指定就留空。"""
        if self.api_key is not None or not self.api_key_env:
            return self
        _ensure_dotenv_loaded()
        raw = os.environ.get(self.api_key_env)
        if raw:
            self.api_key = SecretStr(raw)
        return self


class RunConfig(BaseModel):
    """单次 run 的完整配置。"""

    input: str = Field(
        ...,
        description="源小说路径(.txt / .epub)",
    )
    output_dir: Optional[str] = Field(
        default=None,
        description="输出目录;留空时自动派生",
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
        description="期望的每集时长(秒);LLM 据此分集 + 分镜",
    )
    llm: LLMConfig = Field(..., description="LLM 调用配置")


__all__ = ["RunConfig", "LLMConfig"]
