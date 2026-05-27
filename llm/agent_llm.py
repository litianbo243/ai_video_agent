"""Per-agent LLM 客户端管理模板。

每个 agent 拥有**自己的 LLM 配置**(agent 目录下 ``llm.json``),不再由 workflow
统一注入。本模块提供一个工厂 ``make_agent_llm_manager()``,把"加载配置 + lazy build
客户端 + trace 注入 + 测试覆盖"这套样板封装成 3 个闭包函数,各 agent 各自实例化一份。

典型用法(放在 agent 目录的 ``llm.py``)::

    from pathlib import Path
    from llm.agent_llm import make_agent_llm_manager

    get_llm, set_llm, set_trace_dir = make_agent_llm_manager(
        agent_name="character_profiler",
        config_path=Path(__file__).parent / "llm.json",
    )

设计要点:

* **lazy build**:首次 ``get_llm()`` 才读 ``llm.json`` 建 client,之后命中缓存
* **trace 隔离**:每个 agent 一个 trace 文件,落到 ``<out_dir>/llm_trace/<agent_name>.jsonl``
  + 同名 markdown 子目录(由 ``LLMClient.set_trace_file`` 自动建)
* **set_trace_dir 早晚都行**:可在 build 之前调(挂 pending),也可在之后调(立即生效)
* **测试覆盖**:``set_llm(mock)`` 显式塞入 mock client;``set_llm(None)`` 复位
  让下次 get_llm 重新 lazy build
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional, Tuple

from llm.llm_config import LLMConfig
from llm.client import LLMClient, get_client

logger = logging.getLogger(__name__)


GetLLM = Callable[[], LLMClient]
SetLLM = Callable[[Optional[LLMClient]], None]
SetTraceDir = Callable[["Path | str"], None]


def make_agent_llm_manager(
    *,
    agent_name: str,
    config_path: Path,
) -> Tuple[GetLLM, SetLLM, SetTraceDir]:
    """实例化一个 agent 专属的 LLM 管理三件套。

    返回 ``(get_llm, set_llm, set_trace_dir)``,通过闭包持有私有状态
    (``_client`` / ``_pending_trace_dir``),agent 之间互不干扰。

    Args:
        agent_name: agent 名(用于日志 / trace 文件名)。
        config_path: ``llm.json`` 的绝对路径。
    """
    state: dict = {"client": None, "pending_trace_dir": None}

    def _load_config() -> LLMConfig:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        return LLMConfig.model_validate(raw)

    def _apply_trace_dir(client: LLMClient, out_dir: Path) -> None:
        trace_file = out_dir / "llm_trace" / f"{agent_name}.jsonl"
        client.set_trace_file(str(trace_file))

    def get_llm() -> LLMClient:
        if state["client"] is None:
            cfg = _load_config()
            client = get_client(cfg)
            if state["pending_trace_dir"] is not None:
                _apply_trace_dir(client, state["pending_trace_dir"])
            state["client"] = client
            logger.info(
                "[%s] LLM client 就绪:%s @ %s",
                agent_name, client.model, client.base_url,
            )
        return state["client"]

    def set_llm(client: Optional[LLMClient]) -> None:
        """显式覆盖 agent 的 LLM 客户端(测试 / notebook 用)。

        传 ``None`` = 复位缓存,下次 ``get_llm`` 重新 lazy build。
        """
        state["client"] = client

    def set_trace_dir(out_dir: "Path | str") -> None:
        """注入运行时的 out_dir,trace 落到 ``<out_dir>/llm_trace/<agent_name>.jsonl``。

        既可在首次 ``get_llm`` 之前调(记录 pending,build 时生效),
        也可之后调(对现有 client 立即改写)。
        """
        out = Path(out_dir)
        state["pending_trace_dir"] = out
        if state["client"] is not None:
            _apply_trace_dir(state["client"], out)

    return get_llm, set_llm, set_trace_dir


__all__ = ["make_agent_llm_manager"]
