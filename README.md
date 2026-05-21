# ai_video_agent

把一本中文小说(`.txt` 或 `.epub`)拆解成可直接驱动 AI 短视频生产的**剧本 / 人物 / 节拍**结构化数据,JSON + Markdown 双格式同时输出。LLM 接任意 OpenAI 兼容端点(云上 / 本地 / 自建网关)。

这是 Stage 1(分析层),Stage 2/3 会基于本阶段产物做角色定调图、场景定调图、分镜出图、视频合成。

## 三层概念

| 层 | 含义 | 位置 |
|---|---|---|
| **agent** | 一次 LLM 调用 + prompt + I/O schema + 自带 LLM 配置(`llm.json` + `llm.py`)。最小"会思考"单元。 | `agents/extract_*/` |
| **skill** | 确定性原语,不调 LLM(epub 解码、文本分批、文件 I/O)。 | `skills/<name>/` |
| **workflow** | 用 LangGraph 把 agent + skill 编排成 DAG(multi-agent);**不再注入 LLM**——每个 agent 自治。 | `workflows/<name>.py` |

agent 之间不互相 import 运行逻辑;跨 agent 数据流靠 workflow state 传。

## 流水线

```
输入文件 (.txt | .epub)
        │
        ▼
   skills/book_ingest.ingest_book  (= epub_to_txt + batch_chapters)
        │
        ▼  IngestResult(title, batches, ...)
   ┌────┴────────────────────────────────────────────────────────┐
   │  3 个 agent,workflow 按依赖串调                              │
   │                                                              │
   │  agents/extract_characters ─→ CharacterRoster               │
   │  agents/extract_beats      ─→ BeatList                      │
   │       (依赖 roster;场景 name 由本 agent 自己产出)            │
   │  agents/extract_storyboard ─→ ScreenplayAnalysis            │
   │       (依赖 beats + roster + batches,逐 beat 调一次 LLM;     │
   │        场景视觉环境从原文 + LLM 常识写到每镜 description)      │
   └────────────────────────────────────────────────────────────┘
        │
        ▼
   skills/file_io.write_final_report
        │
        ▼
{output_dir}/{YYYYMMDD_HHMMSS}/    每次运行新建带时间戳的子目录
├── screenplay.json / .md          剧本分析(logline + 分集 episodes + 分镜 storyboards)
├── characters.json / .md          人物档案(含详细外貌 + 性格,给 Stage 2 做角色定调图)
├── beats.json / .md               剧情大纲段(含节奏 + setting_refs 字符串 label + character_refs)
└── meta.json                      运行元信息(书名 / 字数 / 批次数 / 每个 agent 用的 LLM)
```

> **场景的处理:** 本工程不维护独立的场景视觉档案。Beat 内 ``setting_refs`` 是字符串
> label(如「萧家大厅」),用于跨集场景一致性;每镜的视觉环境由 storyboard agent 从
> 原文 + LLM 常识写到 ``Storyboard.description`` 里。

每个 workflow 都可以**单独跑**(见 [run_workflow.py](run_workflow.py));workflow 内部会按需调起依赖的上游 workflow 或 agent。MCP 是预留给*外部*服务的 —— 详见 [`mcp_connectors/README.md`](mcp_connectors/README.md)。

## 项目结构

```
ai_video_agent/
├── agents/                        # LLM-backed:每个目录 = 一次 LLM 调用
│   ├── extract_characters/
│   │   ├── __init__.py            #   重导公共 API(含 get_llm / set_llm / set_trace_dir)
│   │   ├── logic.py               #   SYSTEM_PROMPT + chat_json + 后处理 + 顶部一行实例化 LLM 3 件套
│   │   ├── schema.py              #   I/O 契约(Character / Roster / Extraction)
│   │   └── llm.json               #   本 agent 的 LLMConfig
│   ├── extract_beats/             # 依赖 character schema;场景 name 由本 agent 产出
│   └── extract_storyboard/        # 依赖前两个 schema;单 beat → 一集分镜
├── skills/                        # 确定性原语,不调 LLM
│   ├── epub_to_txt/
│   ├── batch_chapters/
│   ├── book_ingest/               # = epub_to_txt + batch_chapters 的组合
│   ├── file_io/                   # 写 FinalReport(JSON + Markdown)
│   └── skills_manifest.json
├── workflows/                     # LangGraph DAG,组合 agent + skill
│   ├── character_analysis.py
│   ├── beat_analysis.py           # 内部跑 character,再串到 beat
│   ├── storyboard_analysis.py     # 内部跑 beat_analysis,逐 beat 调 storyboard agent
│   └── novel_analysis.py          # 顶层:全跑一遍,落 FinalReport
├── configs/
│   ├── config.py                  # RunConfig(纯流水线参数;LLMConfig 仍在,被各 agent 用)
│   ├── novel_analysis.json        # 示例 config(不含 llm 段——LLM 在 agent 目录)
│   └── __init__.py                # load_config(...)
├── llm/
│   ├── client.py                  # OpenAI 兼容客户端
│   └── agent_llm.py               # make_agent_llm_manager:per-agent get_llm/set_llm/set_trace_dir
├── mcp_connectors/                # 预留给未来的外部 MCP 适配
├── inputs/                        # 源材料,git ignore
├── outputs/                       # 运行产物,git ignore
├── run_workflow.py                # CLI 入口:__main__ 里选要跑哪个 workflow
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
# 1. 编辑 configs/novel_analysis.json,填上 input、output_dir 等流水线参数
# 2. 编辑 agents/extract_*/llm.json,给每个 agent 选 LLM(可以选不同模型)
# 3. 跑
python run_workflow.py
```

> 配置文件路径和要跑的 workflow 都在 [`run_workflow.py`](run_workflow.py) 末尾 `__main__` 里写死,改对应那两行即可。默认跑 ``run_novel_analysis``(全流程),注释切换到 ``run_character_analysis`` / ``run_beat_analysis`` / ``run_storyboard_analysis`` 可单独跑子-workflow。

### Pipeline config 结构(`configs/*.json`)

```json
{
  "input": "inputs/your_novel.epub",
  "output_dir": "outputs",
  "max_batch_chars": 8000,
  "max_total_chars": 0,
  "target_episode_duration_sec": 180,
  "recent_beats_window": 10,
  "langgraph_recursion_limit": 50
}
```

`input` 是必填,其它字段都有默认值。**注意**:这里**没有 `llm` 段** —— LLM 配置已下放到各 agent,见下一节。

| 字段 | 含义 |
|---|---|
| `max_batch_chars` | 每批送给 LLM 的字符上限(默认 8000)|
| `max_total_chars` | **整本小说**字数上限,超出从尾部直接截断;`0` = 不截断(默认)。**用作快速试跑只取前 N 万字看效果** |
| `target_episode_duration_sec` | 期望每集时长(秒),LLM 据此分集 + 分镜;默认 180(3 分钟一集,典型短视频) |
| `recent_beats_window` | beat agent prompt 里展示的最近段数(默认 10);本地小模型 ctx 紧时调低 |
| `langgraph_recursion_limit` | LangGraph 递归上限(默认 50)|

Pydantic 模型定义在 [`configs/config.py`](configs/config.py)。

### Per-agent LLM 配置(`agents/extract_*/llm.json`)

每个 agent **独立挑模型**——character agent 适合用强推理模型(理解人物关系),
storyboard agent 可以用便宜的对话模型(批量生成分镜),互不耦合。

```
agents/extract_characters/llm.json   ← character agent 用的 LLM
agents/extract_beats/llm.json        ← beat agent 用的 LLM
agents/extract_storyboard/llm.json   ← storyboard agent 用的 LLM
```

每个文件就一份 `LLMConfig`:

```json
{
  "base_url": "https://api.deepseek.com",
  "model": "deepseek-v4-pro",
  "api_key_env": "DEEPSEEK_API_KEY",
  "temperature": 0.2,
  "native_model": false,
  "json_max_retries": 1
}
```

> **API key 不要放在 JSON 里**,走 `.env` / 环境变量,通过 `api_key_env` 字段指明读哪个变量。这样 JSON 可以放心入 git。
>
> 三个 agent 可以共用同一个 LLM(默认就是),也可以各点各家——比如 character 用 OpenAI,beat 用 DeepSeek,storyboard 用本地 ollama。

LLM trace 自动按 agent 分文件落到 `<out_dir>/llm_trace/<agent_name>.{jsonl, .dir/}`,
便于事后定位是哪个 agent 哪一批的调用。

### 几种常见 LLM 配置(写到 `agents/extract_*/llm.json`,改 base_url 一行就切家)

任何 OpenAI 兼容端点都能跑。`.env` 里挂上对应 key,JSON 里点名 `api_key_env` 即可,多家可以同时挂着。

**SiliconFlow**(国内聚合,新户送 ¥14):
```json
{
  "base_url": "https://api.siliconflow.cn/v1",
  "model": "Pro/deepseek-ai/DeepSeek-V3.2",
  "api_key_env": "SILICONFLOW_API_KEY"
}
```

**智谱 BigModel**(GLM-4.7-Flash 完全免费):
```json
{
  "base_url": "https://open.bigmodel.cn/api/paas/v4/",
  "model": "glm-4.7-flash",
  "api_key_env": "ZHIPU_API_KEY"
}
```

**DeepSeek 官方**:
```json
{
  "base_url": "https://api.deepseek.com/v1",
  "model": "deepseek-chat",
  "api_key_env": "DEEPSEEK_API_KEY"
}
```

**OpenAI 官方**:
```json
{
  "base_url": "https://api.openai.com/v1",
  "model": "gpt-4o-mini",
  "api_key_env": "OPENAI_API_KEY"
}
```

**阿里通义 DashScope**:
```json
{
  "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
  "model": "qwen-max",
  "api_key_env": "DASHSCOPE_API_KEY"
}
```

**本地 ollama**(免 API key,启动 `ollama serve` 即可):
```json
{
  "base_url": "http://localhost:11434/v1",
  "model": "qwen2.5:14b"
}
```

**自建网关 / 公司内代理**:
```json
{
  "base_url": "http://gateway.internal:8080/v1",
  "model": "internal-gpt-4o",
  "api_key_env": "MY_GATEWAY_KEY"
}
```

`api_key_env` 留空时,`api_key` 取 `"EMPTY"`(本地 / 无鉴权网关够用,SDK 仅要求字符串非空)。

### 在 Python / Notebook 里复用

```python
from configs import RunConfig
from workflows.novel_analysis import run

config = RunConfig.model_validate_json(open("configs/novel_analysis.json").read())
result = run(config)
print(result.output_paths)
```

## 单独跑某个 workflow

每个 workflow 都暴露 `run(config) -> ...`,内部自己起 LLM + ingest + 依赖的上游 workflow:

```python
from configs import load_config
from workflows import (
    character_analysis,
    beat_analysis, storyboard_analysis, novel_analysis,
)

config = load_config("configs/novel_analysis.json")

# 只要人物
ing, roster = character_analysis.run(config)

# 只要分镜(内部会先跑 character + beat)
ing, roster, beats, screenplay = storyboard_analysis.run(config)

# 全跑
result = novel_analysis.run(config)  # 落 screenplay.json/.md + characters.json/.md + ... + meta.json
```

`run_workflow.py` 末尾 `__main__` 里 4 个函数对应 4 个 workflow,改一行选哪个跑。

依赖关系一图流(workflow 内部自动满足):

| Workflow | 直接依赖的 agent / 上游 workflow |
|---|---|
| `character_analysis` | `agents/extract_characters` |
| `beat_analysis` | character_analysis + `agents/extract_beats` |
| `storyboard_analysis` | beat_analysis + `agents/extract_storyboard` |
| `novel_analysis` | 上面全部 + `skills/file_io.write_final_report` |

## 直接复用 agent

跳过 workflow 编排,直接拼 agent 调用(notebook / 自定义流水线场景):

```python
from configs import load_config
from skills.book_ingest import ingest_book
from agents.extract_characters import extract_for_batch, set_trace_dir

config = load_config("configs/novel_analysis.json")
ing = ingest_book(config.input,
                  max_batch_chars=config.max_batch_chars,
                  max_total_chars=config.max_total_chars)

# 可选:让 LLM trace 落到指定目录,文件名自动 = extract_characters.jsonl
set_trace_dir("outputs/notebook_run")

# extract_for_batch 内部 lazy build LLM(用 agents/extract_characters/llm.json),
# 不需要 caller 显式传;调用即自动合并到 known
known = {}
for batch in ing.batches:
    extract_for_batch(batch, known, title=ing.title)
```

测试时想 mock LLM:`from agents.extract_characters import set_llm; set_llm(fake_client)`,完事后 `set_llm(None)` 复位。

## 新增一个 skill

```
skills/<name>/
├── __init__.py     # 暴露公共函数
├── logic.py        # 纯 Python 实现,不调 LLM
├── schema.py       # (可选)I/O 契约
└── readme.md       # 给 LLM-driven agent 读的"秘籍" —— 契约、IO、不变量
```

如果 skill 有确定性入口,把元数据加到 `skills/skills_manifest.json`。

## 新增一个 agent

```
agents/extract_<X>/
├── __init__.py     # 重导 logic 公共函数 + schema 公共类型(含 get_llm/set_llm/set_trace_dir)
├── logic.py        # SYSTEM_PROMPT + 单次 chat_json 调用 + 后处理;顶部一行实例化 LLM 3 件套
├── schema.py       # I/O 契约:Draft(LLM 原始输出) / Domain(合并后) / Collection / *Extraction
└── llm.json        # 本 agent 用的 LLMConfig(base_url + model + api_key_env + …)
```

`logic.py` 顶部统一这样起 LLM(每个 agent 就改 `agent_name` 这一处):

```python
from pathlib import Path
from llm.agent_llm import make_agent_llm_manager

get_llm, set_llm, set_trace_dir = make_agent_llm_manager(
    agent_name="extract_<X>",
    config_path=Path(__file__).parent / "llm.json",
)
```

约定:
- agent 完全**自治** —— 自己的 prompt、自己的 schema、自己的 LLM 配置;workflow 不再给 agent 注入 LLM。
- agent 只做**一次 LLM 调用**(单批 / 单 beat / 单 prompt)。批次循环、合并累积是 workflow 的职责。
- agent **不直接 import 别的 agent 的 logic**;但可以 `from agents.extract_Y.schema import ...` 复用类型契约。
- agent 可以 `from skills.X import ...` 用确定性原语。
- LLM 客户端 lazy build + 模块级缓存:首次 `get_llm()` 才读 `llm.json` 建客户端,之后命中缓存。
- 把 agent 接到 workflow:在 `workflows/<flow>.py` 里 build 一个 LangGraph 节点,内部循环调 `extract_for_batch`(或 `storyboard_beat`),**不传 llm**。

## Cursor IDE 技能 vs 运行时技能(两个不同的层)

`.cursor/skills/*/SKILL.md` 是给**在 Cursor 里编辑这个仓库的 agent**(就是我)看的指引。仓库里 `skills/` 下的运行时项目技能,才是**运行时 agent** 调用的目标。两层可以共享脚本,但读者完全不同。
