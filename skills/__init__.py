"""原生技能池(Native Skills)。

每个技能都是一个独立的子目录,包含三个文件:

  <name>/__init__.py   -- 重新导出对外的公共 API
  <name>/logic.py      -- 纯 Python 实现,不调用任何 LLM
  <name>/readme.md     -- 给 Agent 阅读的"秘籍"(契约说明)

可选的 ``skills/skills_manifest.json`` 是一个扁平索引,manager agent 在不
导入具体模块的情况下,也能通过它了解可用技能的元数据。
"""
