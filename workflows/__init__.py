"""LangGraph workflow:编排 agent + skill 跑流水线。

只有一个流水线 ``novel_analysis``,跑到哪一阶段由 ``RunConfig.mode`` 决定
(``character`` / ``beat`` / ``episode`` / ``screenplay``,深档是浅档的严格超集)。
"""
