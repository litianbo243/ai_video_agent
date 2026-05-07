# ai_video

把一本中文小说(`.txt` 或 `.epub`)拆解成可直接驱动 AI 短视频生产的**剧本 / 人物 / 场景**结构化数据,JSON + Markdown 双格式同时输出。底层用 LangGraph 编排;LLM 接任意 OpenAI 兼容端点(云上 / 本地 / 自建网关)。

这是 Stage 1(分析层),Stage 2/3 会基于本阶段产物做角色定调图、场景定调图、分镜出图、视频合成。

## 流水线

```
输入文件 (.txt | .epub)
        │
        ▼
detect_input ──[epub]──▶ convert_epub (skill: epub_to_txt)
        │                         │
        ├──[txt]──────────────────┘
        ▼
ingest_and_batch  (skill: batch_chapters)
        │
        ▼
analyze (每批 3 个抽取 agent 串调)─┐
   ▲   ├─ character_extractor      │  看 batch 原文 + 已知人物名单
   │   ├─ setting_extractor        │  看 batch 原文 + 已知场景名单
   │   └─ beat_extractor           │  看 batch 原文 + 前 10 段大纲 + 人物/场景名单
   │                                │  → Beat(节奏感, setting_refs, character_refs)
   └────── 还有下一批 ─────────────┘
        │
        ▼
finalize:                          ★ 内部按 Beat 顺序 N 次 LLM 调用
  └── episode_storyboarder (×N段)  每段 Beat → 单集分集分镜
        │                          (input: Beat + Setting/Character refs +
        │                                  本段 batch 原文)
        ▼
write (skill: file_io)
        │
        ▼
{output_dir}/{YYYYMMDD_HHMMSS}/    每次运行新建带时间戳的子目录
├── <input_stem>.txt               规范化后的全文 txt
├── batch_state.json               滚动状态检查点(每批后覆盖,只保最新;支持续跑)
├── screenplay.json / .md          剧本分析(logline + 分集 episodes + 分镜 storyboards)
├── characters.json / .md          人物档案(含详细外貌 + 性格,给 Stage 2 做角色定调图)
├── settings.json / .md            场景档案(物理地点视觉描写,跨剧情段复用,给 Stage 2 做场景定调图)
├── beats.json / .md               剧情大纲段(含节奏 + setting_refs + character_refs)
└── meta.json                      运行元信息(书名 / 字数 / 批次数 / LLM)
```

分段分析的循环用的是**子-agent**(进程内 LLM + 滚动状态),而不是 MCP。MCP 是预留给*外部*服务的 —— 详见 [`mcp_connectors/README.md`](mcp_connectors/README.md)。

## 项目结构

```
ai_video/
├── agents/
│   └── novel_analysis/                  # 小说分析 agent(未来:image_generation/ video_generation/ ...)
│       ├── manager.py                   # 顶层协调者
│       ├── character_extractor.py       # 子-agent,按批抽取人物大纲
│       ├── setting_extractor.py         # 子-agent,按批抽取场景大纲
│       ├── beat_extractor.py            # 子-agent,按批抽取剧情大纲段(看前 10 段做接续)
│       ├── episode_storyboarder.py      # 子-agent,逐段 Beat 产单集分镜(读 batch 原文)
│       └── workflow.py                  # LangGraph StateGraph
├── skills/                        # 原生技能(确定性,不调 LLM,跨 agent 共享)
│   ├── epub_to_txt/
│   ├── batch_chapters/
│   ├── file_io/
│   └── skills_manifest.json
├── mcp_connectors/                # 预留给未来的外部 MCP 适配
├── schema/
│   ├── config.py                  # RunConfig:CLI 接受的 JSON 契约(跨 agent 共享)
│   └── novel_analysis.py          # 小说分析的 Pydantic 模型(每个 agent 一份)
├── llm/
│   └── client.py                  # OpenAI 兼容客户端(任意端点都能跑)
├── configs/                       # 每个 agent 一个 JSON 配置
│   └── novel_analysis.json        # 小说分析 agent 的配置
├── input/                         # 源材料(.txt / .epub),git ignore
├── output/                        # 运行产物,git ignore;每次 run 在下面新建一个时间戳子目录
├── run_novel_analysis.py          # CLI 入口(config-driven)
├── requirements.txt
└── .env.example
```

## 安装

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # 只用本地推理(ollama 等)时此步可跳过
```

`.env` 文件**只存 secrets**(`*_API_KEY`)。LLM 的其它配置(`base_url` / `model` / `temperature`)都在 JSON config 里。本地推理完全不需要 `.env`。

## 运行(config-driven)

```bash
# 1. 编辑 configs/novel_analysis.json,填上 input、output_dir、llm 等
# 2. 跑
python run_novel_analysis.py
```

> 配置文件路径在 [`run_novel_analysis.py`](run_novel_analysis.py) 末尾 `__main__` 里写死(``configs/novel_analysis.json``);需要换路径直接改这一行。

### config 结构

```json
{
  "input": "input/your_novel.epub",
  "output_dir": "output",
  "max_batch_chars": 8000,
  "max_total_chars": 0,
  "target_episode_duration_sec": 180,
  "llm": {
    "base_url": "https://api.deepseek.com/v1",
    "model": "deepseek-chat",
    "api_key_env": "DEEPSEEK_API_KEY",
    "temperature": 0.2
  }
}
```

`input` + `llm.base_url` + `llm.model` 是必填,其它字段都有默认值。

| 字段 | 含义 |
|---|---|
| `max_batch_chars` | 每批送给 LLM 的字符上限(默认 8000)|
| `max_total_chars` | **整本小说**字数上限,超出从尾部直接截断;`0` = 不截断(默认)。**用作快速试跑只取前 N 万字看效果** |
| `target_episode_duration_sec` | 期望每集时长(秒),LLM 据此分集 + 分镜;默认 180(3 分钟一集,典型短视频) |

Pydantic 模型定义在 [`schema/config.py`](schema/config.py)。

> **API key 不要放在 JSON 里**,走 `.env` / 环境变量。config 通过 `api_key_env` 字段指明读哪个变量。这样 config 可以放心入 git。

### 几种常见 LLM 配置(改 base_url 一行就切家)

任何 OpenAI 兼容端点都能跑。`.env` 里挂上对应 key,config 里点名 `api_key_env` 即可,多家可以同时挂着。

**SiliconFlow**(国内聚合,新户送 ¥14):
```json
"llm": {
  "base_url": "https://api.siliconflow.cn/v1",
  "model": "Pro/deepseek-ai/DeepSeek-V3.2",
  "api_key_env": "SILICONFLOW_API_KEY"
}
```

**智谱 BigModel**(GLM-4.7-Flash 完全免费):
```json
"llm": {
  "base_url": "https://open.bigmodel.cn/api/paas/v4/",
  "model": "glm-4.7-flash",
  "api_key_env": "ZHIPU_API_KEY"
}
```

**DeepSeek 官方**:
```json
"llm": {
  "base_url": "https://api.deepseek.com/v1",
  "model": "deepseek-chat",
  "api_key_env": "DEEPSEEK_API_KEY"
}
```

**OpenAI 官方**:
```json
"llm": {
  "base_url": "https://api.openai.com/v1",
  "model": "gpt-4o-mini",
  "api_key_env": "OPENAI_API_KEY"
}
```

**阿里通义 DashScope**:
```json
"llm": {
  "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
  "model": "qwen-max",
  "api_key_env": "DASHSCOPE_API_KEY"
}
```

**本地 ollama**(免 API key,启动 `ollama serve` 即可):
```json
"llm": {
  "base_url": "http://localhost:11434/v1",
  "model": "qwen2.5:14b"
}
```

**自建网关 / 公司内代理**:
```json
"llm": {
  "base_url": "http://gateway.internal:8080/v1",
  "model": "internal-gpt-4o",
  "api_key_env": "MY_GATEWAY_KEY"
}
```

`api_key_env` 留空时,`api_key` 取 `"EMPTY"`(本地 / 无鉴权网关够用,SDK 仅要求字符串非空)。

### 在 Python / Notebook 里复用

```python
from schema.config import RunConfig
from agents.novel_analysis.manager import run

config = RunConfig.model_validate_json(open("configs/novel_analysis.json").read())
result = run(config)
print(result.output_paths)
```

## 续跑(resume)

每批合并后写 `batch_state.json`(覆盖)。崩溃后想从指定 run 接着跑:

1. 找到要续跑的那次 run 的目录(`output/{YYYYMMDD_HHMMSS}/`);
2. 用 `load_checkpoint(output_dir)` 读 `batch_state.json` 得到完整 `BatchState`;
3. 把它当作初始 state 传给 `agents.novel_analysis.workflow.build_graph(...)`,而不是新建一个空的 `BatchState`;
4. `analyze` 节点的条件边是按 `cursor` 走的,所以会自动从断点继续。

> 续跑要用**同一份 config**(`max_batch_chars` / `max_total_chars` 必须一样,否则 `Batch.index` 会错位)。

## 新增一个技能

```
skills/<name>/
├── __init__.py     # 暴露公共函数
├── logic.py        # 纯 Python 实现,不调 LLM
└── readme.md       # 给 Agent 读的"秘籍" —— 契约、IO、不变量
```

最后把这条 entry 加到 `skills/skills_manifest.json`,manager agent 才能在不 import 的情况下自我介绍可用技能。

## 新增一个 agent

要做分析性工作(由 LLM 驱动,可能带状态),在 `agents/` 下加一个模块:

```python
def analyze_X(state: BatchState, batch: Batch, llm: LLMClient) -> BatchState:
    delta = llm.chat_json(SYSTEM_PROMPT, build_user_prompt(state, batch), MyDelta)
    state.merge_my_delta(delta, batch_index=batch.index)
    return state
```

然后在 `agents/novel_analysis/workflow.py` 里把它挂成一个节点(其它 agent 同理,各自的 `agents/<agent>/workflow.py`)。

## Cursor IDE 技能 vs 运行时技能(两个不同的层)

`.cursor/skills/*/SKILL.md` 是给**在 Cursor 里编辑这个仓库的 agent**(就是我)看的指引。仓库里 `skills/` 下的运行时项目技能,才是**运行时 agent** 调用的目标。两层可以共享脚本,但读者完全不同。
