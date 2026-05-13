"""workflow:把 agent + skill 编排成 multi-agent DAG。

每个文件 = 一个独立 workflow(``predefined DAG``)。

术语三件套:

* **agent** (``agents/``):一次 LLM 调用 + prompt + I/O schema。最小的"会思考"
  单元;workflow 把它们串成 multi-agent。
* **skill** (``skills/``):确定性原语(无 LLM 调用),例如文件 I/O、文本分批、
  epub 解码。agent / workflow 都可以调用。
* **workflow** (本目录):用 LangGraph 把多个 agent + 必要 skill 编排成一条流水线。
  自己**不**直接调 LLM 或写纯计算 —— 只负责"什么时候调谁"。

agent 之间默认**不互相 import 运行逻辑**;跨 agent 数据流靠 workflow state 传。
agent 与 workflow 同层级:理论上 skill 也可以包装一个 workflow 给 agent 复用,
不过当前项目没这个需求。
"""
