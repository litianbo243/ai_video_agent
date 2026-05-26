"""LLM-backed agents:每个目录是一个"单 LLM 调用 + 上下文 prompt + I/O schema"的最小单元。

约定:

* 每个 agent 是**一次** LLM 调用 + 它的 system prompt + I/O 数据契约 + 必要后处理。
* ``workflows/`` 把多个 agent 串成 multi-agent DAG(并行 / fan-out / 跨批累积)。
* agent 之间默认**不互相 import 运行逻辑**;跨 agent 的数据流靠 workflow state 传递。
  跨 agent 共享的**类型定义**(``Character`` / ``Beat`` …)允许直接
  ``from agents.extract_X.schema import ...``,因为类型契约就属于产出它的 agent。
* agent 可以使用 ``skills/`` 下的确定性原语(epub 解析、批分、文件读写);
  反向也允许:``skill`` 可以包装一个 workflow 给外面用(目前没这个需求)。

每个 agent 子目录的标准结构::

    agents/extract_X/
    ├── __init__.py    # 重导公共 API + I/O schema
    ├── logic.py       # SYSTEM_PROMPT + chat_json 调用 + 后处理
    └── schema.py      # I/O 数据契约(LLM 输出包装 + 业务产物类型)

现有 5 个 agent:

* ``extract_characters``   —— 全文人物档案抽取(LLM 输出 → ``CharacterList``)
* ``extract_beats``        —— 全文剧情段抽取(依赖 character 产物;场景 name 由
                              本 agent 自己产出,不再有独立的 setting agent)
* ``episode_planner``      —— 全书 N 段 Beat → M 集 Episode 规划(只产元数据 +
                              ``beat_indices``,不产分镜);把"聚合"判断从
                              extract_beats 抽出来,让每个 agent 职责更清晰
* ``extract_storyboard``   —— 单集所含 N 段 Beat → 一集**叙事分镜**(谁 / 在哪 /
                              说什么 / 想什么 + 集层 ``director_intent``);
                              视觉决策不在此 agent 职责内
* ``shot_director``        —— 一集叙事分镜 → 一集**视觉指导**(景别 / 运镜 /
                              起始画面 / 时长 + 集层 ``visual_style``);
                              workflow 调 ``extract_storyboard.merge_episode``
                              按 index 合并成完整 ``Storyboard``

**为什么 storyboard 拆成两个 agent**:一次 LLM 调用既写剧本又当摄影指导职责
太重导致两边都不深入。拆开后,前者只看故事节奏,后者拿到锁定的叙事后专心做
视觉决策。两个 prompt 都更专注,产出质量更可控,代价是每集 2 次 LLM 调用。

**为什么 beat 和 episode 分两层**:Beat 是"戏剧单元"(一个完整冲突 / 揭示 /
决断),Episode 是"一集短视频"(目标时长 ~5 分钟)。两者粒度天然不同 ——
1 beat 通常对应 60-120 秒视频体量,1 集需要聚合 2-5 个 beat 才够时长。把这两
个判断分成两个 agent 后,beat agent 专心切戏剧单元,episode_planner 专心做
分集聚合,产出质量更可控。
"""
