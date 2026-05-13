"""skills:确定性原语池(deterministic primitives)。

每个 skill 是一个独立子目录:

  <name>/__init__.py   -- 重导对外的公共 API
  <name>/logic.py      -- 纯 Python 实现,**不调用任何 LLM**
  <name>/schema.py     -- (可选)skill 的 I/O 数据契约
  <name>/readme.md     -- (可选)契约说明

LLM-backed 的单元住在 ``agents/`` 而不是这里。两者通过 ``workflows/`` 编排到
一起,workflow / agent 都可以使用 skill 做确定性工作。

可选的 ``skills/skills_manifest.json`` 是一个扁平索引,在不导入具体模块的
情况下也能通过它了解可用 skill 的元数据。
"""
