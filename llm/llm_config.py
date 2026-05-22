"""LLM client 的配置契约。

每个 agent 在自己目录的 ``llm.json`` 里独立定义自己的 LLM 后端,本模块
提供解析这些 JSON 的 Pydantic 模型 ``LLMConfig``。

任何 OpenAI 兼容协议的端点(DeepSeek / 智谱 / 硅基流动 / vLLM / SGLang /
llama.cpp ……)都用同一套字段定位:``base_url`` + ``model`` + ``api_key`` 即可。

API key 不写进 JSON,走 ``.env`` / 环境变量(由 ``api_key_env`` 字段指定变量名)。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, SecretStr, model_validator


# ---------------------------------------------------------------------------
# .env loader(LLMConfig 用,从 env 注入 api_key)
# ---------------------------------------------------------------------------

_dotenv_loaded = False


def _ensure_dotenv_loaded() -> None:
    """加载 .env 文件(幂等);找不到也不报错。"""
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
    trace_file: Optional[str] = Field(
        default=None,
        description=(
            "可选:把每次 LLM 调用的完整 prompt + response 落成 JSONL 的路径。"
            "留空则不落盘。runner 会自动设为 `<out_dir>/llm_trace.jsonl`。"
        ),
    )
    native_model: bool = Field(
        default=False,
        description=(
            "是否本地原生部署的模型(vLLM / SGLang / llama.cpp 等支持 guided decoding 的服务端)。"
            "True:`chat_json` 走 grammar-constrained decoding(`extra_body={'guided_json': ...}`),"
            "服务端在每个采样步用 schema 屏蔽非法 token,100% 保证结构合规——小模型必备。"
            "False(默认):走 `response_format=json_object` + prompt 里写 schema 的常规模式,"
            "适配 DeepSeek / OpenAI / 智谱等云端 API。"
        ),
    )
    json_max_retries: int = Field(
        default=1,
        ge=0,
        le=5,
        description=(
            "`chat_json` 校验失败时的反馈式重试次数(不含首次)。"
            "失败时把 Pydantic 错误信息回灌给 LLM,让它修正后重出。"
            "0 = 不重试(失败直接抛);1(默认)= 多调一次;通常 1-2 足够。"
        ),
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


__all__ = ["LLMConfig"]
