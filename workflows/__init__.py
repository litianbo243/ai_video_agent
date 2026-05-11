"""workflow 集合。每个子目录是一个独立的 workflow(predefined DAG)。

术语:
* **workflow** 与 **agent** 是同层级的编排概念——前者人工写死流程,
  后者由 LLM 在 ReAct 循环里自主决策(本项目目前没有 agent,只有 workflow)。
* workflow 面向 ``skills/`` 下的原语层编排,自己**不**包含 LLM 调用或纯计算逻辑。
"""
