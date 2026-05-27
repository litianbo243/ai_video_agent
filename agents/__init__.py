"""LLM-backed agents:每个目录是一个"单 LLM 调用 + 上下文 prompt + 后处理"的最小单元。

约定:

* 每个 agent 是**一次** LLM 调用 + 它的 system prompt + 必要后处理。
* I/O 数据契约统一放在顶层 ``schemas/``,agent 目录里**不再有** ``schema.py``。
  跨 agent 的类型共享靠 ``from schemas.X import Y``,杜绝"两个真相"。
* ``workflows/`` 把多个 agent 串成 multi-agent DAG(并行 / fan-out / 跨批累积)。
* agent 之间默认**不互相 import 运行逻辑**;跨 agent 的数据流靠 workflow state 传递。
* agent 可以使用 ``skills/`` 下的确定性原语(epub 解析、批分、文件读写);
  反向也允许:``skill`` 可以包装一个 workflow 给外面用(目前没这个需求)。

每个 agent 子目录的标准结构::

    agents/<agent_name>/
    ├── __init__.py    # 重导公共 API(run_for_batch 等)
    ├── logic.py       # SYSTEM_PROMPT + chat_json 调用 + 后处理
    └── llm.json       # 该 agent 的 LLM 配置(model / temperature / …)

现有 5 个 agent:

* ``character_profiler``   —— 全文人物档案抽取 / 维护(产出 ``CharacterList``,
                              支持跨批累积、字段 carry-over、人名升格回扫)
* ``beat_segmenter``       —— 全文剧情段切分(产出 ``BeatList``;场景 name 由
                              本 agent 自己产出,不再有独立 setting agent)
* ``episode_planner``      —— 全书 N 段 Beat → M 集 Episode 规划(只产元数据 +
                              ``beat_indices``,不产分镜);把"聚合"判断从
                              beat_segmenter 抽出来,让每个 agent 职责更清晰
* ``narrative_director``   —— 单集所含 N 段 Beat → 一集**叙事分镜**(谁 / 在哪 /
                              说什么 / 想什么 + 集层 ``director_intent``);
                              视觉决策不在此 agent 职责内
* ``shot_director``        —— 一集叙事分镜 → 一集**视觉指导**(景别 / 运镜 /
                              起始画面 / 时长 + 集层 ``visual_style``);
                              workflow 调 ``narrative_director.merge_episode``
                              按 index 合并成完整 ``Shot``

**为什么分镜拆成两个 agent**:一次 LLM 调用既写剧本又当摄影指导职责
太重导致两边都不深入。拆开后,``narrative_director`` 只看故事节奏,
``shot_director`` 拿到锁定的叙事后专心做视觉决策。两个 prompt 都更专注,
产出质量更可控,代价是每集 2 次 LLM 调用。

**为什么 beat 和 episode 分两层**:Beat 是"戏剧单元"(一个完整冲突 / 揭示 /
决断),Episode 是"一集短视频"(目标时长 ~5 分钟)。两者粒度天然不同 ——
1 beat 通常对应 60-120 秒视频体量,1 集需要聚合 2-5 个 beat 才够时长。把这两
个判断分成两个 agent 后,``beat_segmenter`` 专心切戏剧单元,``episode_planner``
专心做分集聚合,产出质量更可控。
"""
