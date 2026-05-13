"""LLM-backed agents:每个目录是一个"单 LLM 调用 + 上下文 prompt + I/O schema"的最小单元。

约定:

* 每个 agent 是**一次** LLM 调用 + 它的 system prompt + I/O 数据契约 + 必要后处理。
* ``workflows/`` 把多个 agent 串成 multi-agent DAG(并行 / fan-out / 跨批累积)。
* agent 之间默认**不互相 import 运行逻辑**;跨 agent 的数据流靠 workflow state 传递。
  跨 agent 共享的**类型定义**(``Character`` / ``Setting`` / ``Beat`` …)允许直接
  ``from agents.extract_X.schema import ...``,因为类型契约就属于产出它的 agent。
* agent 可以使用 ``skills/`` 下的确定性原语(epub 解析、批分、文件读写);
  反向也允许:``skill`` 可以包装一个 workflow 给外面用(目前没这个需求)。

每个 agent 子目录的标准结构::

    agents/extract_X/
    ├── __init__.py    # 重导公共 API + I/O schema
    ├── logic.py       # SYSTEM_PROMPT + chat_json 调用 + 后处理
    └── schema.py      # I/O 数据契约(LLM 输出包装 + 业务产物类型)

现有 4 个 agent:

* ``extract_characters``   —— 全文人物档案抽取(LLM 输出 → ``CharacterRoster``)
* ``extract_settings``     —— 全文场景档案抽取(LLM 输出 → ``SettingCollection``)
* ``extract_beats``        —— 全文剧情段抽取(依赖前两个的产物)
* ``extract_storyboard``   —— 单段 Beat → 单集分镜(依赖前三个的产物)
"""
