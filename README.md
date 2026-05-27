# ai_video_agent

把一本中文小说(`.txt` 或 `.epub`)拆解成可直接驱动 AI 短视频生产的**剧本 / 人物 / 节拍**结构化数据,JSON + Markdown 双格式同时输出。LLM 接任意 OpenAI 兼容端点(云上 / 本地 / 自建网关)。

这是 Stage 1(分析层),Stage 2/3 会基于本阶段产物做角色定调图、场景定调图、分镜出图、视频合成。

## 三层概念

| 层 | 含义 | 位置 |
|---|---|---|
| **agent** | 一次 LLM 调用 + prompt + 自带 LLM 配置(`llm.json`)。最小"会思考"单元;I/O schema 集中在顶层 `schemas/`。 | `agents/<agent_name>/` |
| **skill** | 确定性原语,不调 LLM(epub 解码、文本分批、文件 I/O)。 | `skills/<name>/` |
| **workflow** | 用 LangGraph 把 agent + skill 编排成 DAG(multi-agent);**不再注入 LLM**——每个 agent 自治。 | `workflows/novel_analysis.py`(唯一入口,mode 决定跑到哪) |

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
   │  4 个 agent,workflow 按依赖串调                              │
   │                                                              │
   │  agents/character_profiler ─→ CharacterList               │
   │  agents/beat_segmenter      ─→ BeatList                      │
   │       (依赖 roster;场景 name 由本 agent 自己产出)            │
   │  agents/narrative_director ─→ NarrativeShotList       │
   │       (依赖 beats + roster + batches,逐 beat 调一次 LLM;     │
   │        只产**叙事维度**:谁 / 在哪 / 说什么 / 想什么 + intent)   │
   │  agents/shot_director       ─→ ShotDirectionList             │
   │       (拿到叙事分镜后,逐集调一次 LLM 配视觉:景别 / 运镜 /    │
   │        起始画面 / 时长 + 集层视觉调性)                        │
   │                                                              │
   │  → workflow 按 index 合并 → 完整 Shot / Episode /       │
   │                              Screenplay              │
   └────────────────────────────────────────────────────────────┘
        │
        ▼
   skills/file_io.write_partial
        │
        ▼
{output_dir}/{YYYYMMDD_HHMMSS}/    每次运行新建带时间戳的子目录
├── characters.json / .md          人物档案(含详细外貌 + 性格 + 弧光,给 Stage 2 做角色定调图)
├── beats.json / .md               剧情大纲段(含节奏 + setting_refs 字符串 label + character_refs)
├── episode_plans.json             分集规划(N 段 beat → M 集元数据 + beat_indices)
├── screenplay.json / .md          剧本分析(分集 episodes + 每集分镜 shots)
└── meta.json                      运行元信息(书名 / 字数 / 批次数 / 每个 agent 用的 LLM)
```

> **场景的处理:** 本工程不维护独立的场景视觉档案。Beat 内 ``setting_refs`` 是字符串
> label(如「萧家大厅」),用于跨集场景一致性;每镜的视觉环境由 ``shot_director``
> agent 从原文 + 人物档案 + LLM 常识写到 ``Shot.description`` 里。
>
> **分镜为何拆 2 个 agent**:一次 LLM 既写剧本又当摄影指导职责太重导致两边
> 都不深入。拆开后 ``narrative_director`` 只看故事节奏(讲什么),``shot_director``
> 拿到锁定的叙事后专心做视觉决策(怎么拍)。代价是每集 2 次 LLM 调用,换来 2 个
> prompt 都更专注,产出质量更可控。

每个 workflow 都可以**单独跑**(见 [run_workflow.py](run_workflow.py));workflow 内部会按需调起依赖的上游 workflow 或 agent。MCP 是预留给*外部*服务的 —— 详见 [`mcp_connectors/README.md`](mcp_connectors/README.md)。

## 项目结构

```
ai_video_agent/
├── agents/                        # LLM-backed:每个目录 = 一次 LLM 调用
│   ├── character_profiler/
│   │   ├── __init__.py            #   重导公共 API(含 get_llm / set_llm / set_trace_dir)
│   │   ├── logic.py               #   SYSTEM_PROMPT + chat_json + 后处理 + 顶部一行实例化 LLM 3 件套
│   │   ├── schema.py              #   I/O 契约(Character / Roster / Extraction)
│   │   └── llm.json               #   本 agent 的 LLMConfig
│   ├── beat_segmenter/             # 依赖 character schema;场景 name 由本 agent 产出
│   ├── narrative_director/        # 依赖前两个 schema;单 beat → 一集**叙事分镜**
│   └── shot_director/             # 叙事分镜 → 一集**视觉指导**(景别/运镜/起始画面/时长)
├── skills/                        # 确定性原语,不调 LLM
│   ├── epub_to_txt/
│   ├── batch_chapters/
│   ├── book_ingest/               # = epub_to_txt + batch_chapters 的组合
│   ├── file_io/                   # 写 FinalReport(JSON + Markdown)
│   └── skills_manifest.json
├── workflows/                     # LangGraph DAG,组合 agent + skill
│   └── novel_analysis.py          # 唯一流水线;mode 早停决定跑到哪一阶段
├── configs/
│   ├── run_config.py              # RunConfig + RunMode(纯流水线参数;LLM 不在此)
│   ├── run_config.json            # 示例 config(不含 llm 段——LLM 在 agent 目录)
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
# 1. 编辑 configs/run_config.json,填上 input / output_dir / mode 等流水线参数
# 2. 编辑 agents/<agent_name>/llm.json,给每个 agent 选 LLM(可以选不同模型)
# 3. 跑
python run_workflow.py
```

> 整套流水线只有**一个入口**:``run_workflow.main(config_path)``,内部按
> ``config.mode`` 决定跑到哪一阶段。改 JSON 即可切早停档,无需改代码。

### 4 档 mode(`RunConfig.mode`)

| mode | 跑到哪一阶段为止 | 用途 |
|---|---|---|
| `character`  | 仅人物档案 | 调 `character_profiler` prompt |
| `beat`       | + 剧情段切分 | 调 `beat_segmenter` prompt |
| `episode`    | + 分集规划 | 调 `episode_planner` prompt / 切集质量 |
| `screenplay` | + 逐集分镜(完整流水线,默认) | 跑完整剧本 |

**4 档是严格超集**:跑 `screenplay` 时跑出的中间产物(人物 / 剧情段 /
分集规划)跟单独跑 `character` / `beat` / `episode` 时是同一份。所以调 prompt
时先用浅 mode 早停快速看产物,确认 OK 再切 `screenplay` 跑完整流水线。

### Pipeline config 结构(`configs/run_config.json`)

```json
{
  "input": "inputs/your_novel.epub",
  "output_dir": "outputs",
  "mode": "screenplay",

  "max_batch_chars": 8000,
  "max_total_chars": 0,

  "target_episode_duration_sec": 180,
  "recent_beats_window": 10,
  "rewrite_window": 1,
  "shot_prev_tail_window": 3
}
```

`input` 是必填,其它字段都有默认值。**注意**:这里**没有 `llm` 段** —— LLM 配置已下放到各 agent,见下一节。

| 字段 | 含义 |
|---|---|
| `mode` | 跑到哪一阶段为止:`character` / `beat` / `episode` / `screenplay`(默认)|
| `max_batch_chars` | 每批送给 LLM 的字符上限(默认 8000)|
| `max_total_chars` | **整本小说**字数上限,超出从尾部直接截断;`0` = 不截断(默认)。**用作快速试跑只取前 N 万字看效果** |
| `target_episode_duration_sec` | 期望每集时长(秒),LLM 据此分集 + 分镜;默认 180(3 分钟一集,典型短视频) |
| `recent_beats_window` | beat agent prompt 里展示的最近段数(默认 10);本地小模型 ctx 紧时调低 |
| `rewrite_window` | beat agent 每批必须复述/修订的末尾 K 段(默认 1);K=0 关闭跨批续写(长戏剧段会被批边界切碎),K=1 推荐(LLM 自然接续),K>=2 给 LLM 更大修订空间但 token 成本随 K 增长。同时是分镜阶段的「冷却期」 |
| `shot_prev_tail_window` | `shot_director` 每集开头看上集末 K 镜做画面承接(默认 3);K=0 关闭(集间无视觉承接),K=3 推荐,K>=5 给 LLM 更长视觉记忆但 token 成本随 K 增长 |

Pydantic 模型 `RunConfig` 定义在 [`configs/run_config.py`](configs/run_config.py);
LLM 客户端配置 `LLMConfig` 定义在 [`llm/llm_config.py`](llm/llm_config.py)(被各 agent 自治使用)。

### Per-agent LLM 配置(`agents/<agent_name>/llm.json`)

每个 agent **独立挑模型**——`character_profiler` 适合用强推理模型(理解人物关系),
`shot_director` 可以用便宜的对话模型(批量生成分镜),互不耦合。

```
agents/character_profiler/llm.json   ← character agent 用的 LLM
agents/beat_segmenter/llm.json        ← beat agent 用的 LLM
agents/narrative_director/llm.json   ← 叙事分镜师 agent 用的 LLM
agents/shot_director/llm.json        ← 镜头导演 agent 用的 LLM(视觉决策)
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
> 四个 agent 可以共用同一个 LLM(默认就是),也可以各点各家——比如 character 用 OpenAI,beat 用 DeepSeek,叙事分镜用 Claude,shot_director 用本地 ollama。

LLM trace 自动按 agent 分文件落到 `<out_dir>/llm_trace/<agent_name>.{jsonl, .dir/}`,
便于事后定位是哪个 agent 哪一批的调用。

### 几种常见 LLM 配置(写到 `agents/<agent_name>/llm.json`,改 base_url 一行就切家)

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

## 早停跑某一阶段(mode-driven)

只有一个入口 ``novel_analysis.run(config)``,跑到哪一阶段由 ``config.mode`` 决定:

```python
from configs import load_config, RunMode
from workflows import novel_analysis

# 改 JSON 里的 "mode" 即可,或者在代码里临时 override:
config = load_config("configs/run_config.json")
config.mode = RunMode.BEAT   # 临时 override 成只跑到 beat

result = novel_analysis.run(config)
# result.mode / result.characters / result.beats /
# result.episode_plans / result.screenplay / result.final_report
# 浅 mode 跑出的字段对应"未跑到"的就是 None
```

各 mode 对应跑到哪一阶段、用哪些 agent、落哪些产物:

| mode | 跑哪些 agent | 落哪些产物 |
|---|---|---|
| `character`  | `character_profiler` | characters.{json,md} |
| `beat`       | + `beat_segmenter`(interleaved) | + beats.{json,md} |
| `episode`    | + `episode_planner` | + episode_plans.json |
| `screenplay` | + `narrative_director` + `shot_director`(逐集 narrate → direct → merge)| + screenplay.{json,md} + meta.json |

`meta.json` 在所有 mode 都会落,记录本次 run 用到的 LLM / 字数 / 批次等元信息。

## 直接复用 agent

跳过 workflow 编排,直接拼 agent 调用(notebook / 自定义流水线场景):

```python
from configs import load_config
from skills.book_ingest import ingest_book
from agents.character_profiler import run_for_batch, set_trace_dir

config = load_config("configs/novel_analysis.json")
ing = ingest_book(config.input,
                  max_batch_chars=config.max_batch_chars,
                  max_total_chars=config.max_total_chars)

# 可选:让 LLM trace 落到指定目录,文件名自动 = character_profiler.jsonl
set_trace_dir("outputs/notebook_run")

# run_for_batch 内部 lazy build LLM(用 agents/character_profiler/llm.json),
# 不需要 caller 显式传;调用即自动合并到 known
known = {}
for batch in ing.batches:
    run_for_batch(batch, known, title=ing.title)
```

测试时想 mock LLM:`from agents.character_profiler import set_llm; set_llm(fake_client)`,完事后 `set_llm(None)` 复位。

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
agents/<agent_name>/
├── __init__.py     # 重导 logic 公共函数(含 get_llm/set_llm/set_trace_dir)
├── logic.py        # SYSTEM_PROMPT + 单次 chat_json 调用 + 后处理;顶部一行实例化 LLM 3 件套
└── llm.json        # 本 agent 用的 LLMConfig(base_url + model + api_key_env + …)

# I/O 数据契约统一放在顶层 `schemas/`,新增一个文件 schemas/<agent_name>.py
# 装 Draft(LLM 原始输出) / Domain(合并后) / Collection 等类型即可。
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
- 把 agent 接到 workflow:在 `workflows/novel_analysis.py` 里 build 一个 LangGraph 节点,内部循环调 `run_for_batch`(或 `plan_episodes` / `narrate_episode` / `direct_episode` 等 agent 入口),**不传 llm**。

## Cursor IDE 技能 vs 运行时技能(两个不同的层)

`.cursor/skills/*/SKILL.md` 是给**在 Cursor 里编辑这个仓库的 agent**(就是我)看的指引。仓库里 `skills/` 下的运行时项目技能,才是**运行时 agent** 调用的目标。两层可以共享脚本,但读者完全不同。
