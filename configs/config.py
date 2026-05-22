"""``configs/*.json`` 配置文件的 Pydantic 数据契约。

JSON 文件长这样::

    {
      "input": "inputs/your_novel.epub",
      "output_dir": "outputs",
      "max_batch_chars": 8000,
      "max_total_chars": 0,
      "target_episode_duration_sec": 180,
      "recent_beats_window": 10,
      "langgraph_recursion_limit": 50
    }

* ``RunConfig`` —— 整个 JSON 文件的根(纯流水线编排参数,不含 LLM 配置)
* ``LLMConfig`` —— 单个 agent 的 LLM 配置;每个 agent 在自己目录的
  ``llm.json`` 里独立定义,详见 ``agents/extract_*/llm.json``

API key 不写进 JSON,走 ``.env`` / 环境变量(由 ``api_key_env`` 字段指定变量名)。

历史变化:``RunConfig.llm`` 字段已下放到各 agent —— 不同 agent 可以选不同的
模型 / endpoint(character agent 用强推理模型,storyboard agent 用便宜模型,
等等),workflow 不再统一 build LLM。
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
    langgraph_recursion_limit: int = Field(
        default=50,
        gt=0,
        description=(
            "LangGraph 父-workflow(novel_analysis)的递归上限,防止节点循环失控。"
            "默认 50 足够 4 个 workflow + ingest + write 这种深度,"
            "若以后加 cache / 重试节点跑挂了可调高。"
        ),
    )


__all__ = ["RunConfig", "LLMConfig"]
