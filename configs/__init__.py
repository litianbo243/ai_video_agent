"""运行时配置目录。

* ``run_config.py`` —— ``RunConfig`` Pydantic 契约 + ``RunMode`` 枚举 +
  ``load_config(path)`` 加载入口
* ``*.json``        —— 实际配置文件,被 ``RunConfig.model_validate`` 解析

LLM 配置不在这里,见 ``llm.llm_config.LLMConfig``(每个 agent 自治)。

外部统一用 ``from configs import RunConfig, RunMode, mode_includes, load_config``。
"""

from configs.run_config import RunConfig, RunMode, load_config, mode_includes

__all__ = ["RunConfig", "RunMode", "load_config", "mode_includes"]
