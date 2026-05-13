# ai_video_agent

把一本中文小说(`.txt` 或 `.epub`)拆解成可直接驱动 AI 短视频生产的**剧本 / 人物 / 场景**结构化数据,JSON + Markdown 双格式同时输出。LLM 接任意 OpenAI 兼容端点(云上 / 本地 / 自建网关)。

这是 Stage 1(分析层),Stage 2/3 会基于本阶段产物做角色定调图、场景定调图、分镜出图、视频合成。

## 流水线

每个分析维度是一个**独立的子-agent**,有自己明确的 I/O 契约;`novel_analysis` 顶层 agent 只负责按依赖顺序把它们串起来。

```
输入文件 (.txt | .epub)
        │
        ▼
agents/novel_analysis/ingest.py  (skill: epub_to_txt + batch_chapters)
        │
        ▼  (title, batches)
   ┌────┴────────────────────────────────────────────────┐
   │  4 个独立 agent,按依赖顺序串调                       │
   │                                                       │
   │  agents/character_analysis ─→ CharacterRoster        │
   │  agents/setting_analysis   ─→ SettingCollection      │
   │  agents/beat_analysis      ─→ BeatList               │
   │       (依赖 roster + settings)                        │
   │  agents/storyboard_analysis─→ ScreenplayAnalysis     │
   │       (依赖 beats + roster + settings + batches)     │
   └─────────────────────────────────────────────────────┘
        │
        ▼
write_final_report (skill: file_io)
        │
        ▼
{output_dir}/{YYYYMMDD_HHMMSS}/    每次运行新建带时间戳的子目录
├── screenplay.json / .md          剧本分析(logline + 分集 episodes + 分镜 storyboards)
├── characters.json / .md          人物档案(含详细外貌 + 性格,给 Stage 2 做角色定调图)
├── settings.json / .md            场景档案(物理地点视觉描写,跨剧情段复用,给 Stage 2 做场景定调图)
├── beats.json / .md               剧情大纲段(含节奏 + setting_refs + character_refs)
└── meta.json                      运行元信息(书名 / 字数 / 批次数 / LLM)
```

每个子-agent 都可以**单独被调用**,只要传它需要的依赖即可(见下方"单独使用某个子-agent")。MCP 是预留给*外部*服务的 —— 详见 [`mcp_connectors/README.md`](mcp_connectors/README.md)。

## 项目结构

```
ai_video_agent/
├── agents/
│   ├── character_analysis/              # 独立子-agent,人物档案
│   │   ├── manager.py                   #   run(batches, llm, *, title="") -> CharacterRoster
│   │   ├── extractor.py                 #   单批 LLM 调用
│   │   └── schema.py                    #   LLM 私有 IO(CharacterExtraction)
│   ├── setting_analysis/                # 独立子-agent,场景档案
│   ├── beat_analysis/                   # 独立子-agent,剧情大纲段
│   ├── storyboard_analysis/             # 独立子-agent,分集 + 分镜
│   └── novel_analysis/                  # 顶层编排:调上面 4 个
│       ├── manager.py                   #   run(config) -> RunResult
│       └── ingest.py                    #   epub→txt + 切批
├── skills/                              # 原生技能(确定性,不调 LLM,跨 agent 共享)
│   ├── epub_to_txt/
│   ├── batch_chapters/
│   ├── file_io/
│   └── skills_manifest.json
├── mcp_connectors/                # 预留给未来的外部 MCP 适配
├── schema/
│   ├── config.py                  # RunConfig:CLI 接受的 JSON 契约
│   └── novel_analysis.py          # 共享领域模型(Character / Setting / Beat / Episode ...)
├── llm/
│   └── client.py                  # OpenAI 兼容客户端
├── configs/
│   └── novel_analysis.json
├── inputs/                        # 源材料,git ignore
├── outputs/                       # 运行产物,git ignore
├── run_workflow.py                # CLI 入口:在 __main__ 里选要跑哪个 workflow
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
python run_workflow.py
```

> 配置文件路径和要跑的 workflow 都在 [`run_workflow.py`](run_workflow.py) 末尾 `__main__` 里写死,改对应那两行即可。默认跑 ``run_novel_analysis``(全流程),注释切换到 ``run_character_analysis`` / ``run_setting_analysis`` 可单独跑子-workflow。

### config 结构

```json
{
  "input": "inputs/your_novel.epub",
  "output_dir": "outputs",
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
from configs import RunConfig
from workflows.novel_analysis import run

config = RunConfig.model_validate_json(open("configs/novel_analysis.json").read())
result = run(config)
print(result.output_paths)
```

## 单独使用某个子-agent

每个子-agent 都分两层 API:

* **顶层 `run(config)`** —— 吃项目 `RunConfig`,内部建 LLM、做 ingest,再调内层。
  runner 用这条。**只有能独立起步的 agent 才暴露**(character / setting)。
* **内层 `run_with_batches(batches, llm, ...)`** —— 纯计算,workflow 用,
  4 个 agent 都有。

最方便的入口(单跑某个 agent):

```python
from workflows import character_analysis, setting_analysis
from configs import load_config

config = load_config("configs/novel_analysis.json")

# 只跑人物(顶层,内部包了 LLM + ingest)
ing, roster = character_analysis.run(config)

# 只跑场景(同上)
ing, settings = setting_analysis.run(config)
```

需要更细控制(自己管 ingest / LLM,例如在 notebook 里复用同一份 batches 跑多个 agent),
用内层 API:

```python
from workflows import character_analysis, setting_analysis, beat_analysis, storyboard_analysis
from configs import load_config
from llm.client import get_client
from skills.book_ingest import ingest_book

config = load_config("configs/novel_analysis.json")
llm = get_client(config.llm)
ing = ingest_book(config.input,
                  max_batch_chars=config.max_batch_chars,
                  max_total_chars=config.max_total_chars)

roster = character_analysis.run_with_batches(ing.batches, llm, title=ing.title)
settings = setting_analysis.run_with_batches(ing.batches, llm, title=ing.title)
beats = beat_analysis.run_with_batches(
    ing.batches, roster, settings, llm, title=ing.title,
)
screenplay = storyboard_analysis.run_with_batches(
    beats, roster, settings, ing.batches, llm,
    target_duration_sec=180, title=ing.title,
)
```

依赖关系一图流:

| Agent | 直接依赖项 |
|---|---|
| `character_analysis` | batches |
| `setting_analysis` | batches |
| `beat_analysis` | batches + `CharacterRoster` + `SettingCollection` |
| `storyboard_analysis` | batches + `BeatList` + `CharacterRoster` + `SettingCollection` |

## 新增一个技能

```
skills/<name>/
├── __init__.py     # 暴露公共函数
├── logic.py        # 纯 Python 实现,不调 LLM
└── readme.md       # 给 Agent 读的"秘籍" —— 契约、IO、不变量
```

最后把这条 entry 加到 `skills/skills_manifest.json`。

## 新增一个 agent

参考 `agents/character_analysis/` 的目录结构:

```
agents/<name>/
├── __init__.py     # 只导出 `run`
├── manager.py      # 公开入口:run(...) -> <DomainOutputModel>
├── extractor.py    # 单批 LLM 调用(prompt + chat_json),不做循环
└── schema.py       # LLM 私有 IO(只本 agent 用)
```

约定:
- **manager.run()** 是该 agent 的对外 API,签名里显式声明所有依赖项(没有隐藏全局状态)。
- **共享领域类型**(`Character` / `Setting` / `Beat` ...)放在 `schema/novel_analysis.py`。
- **LLM 私有产物**(`*Extraction`)放在 agent 自己的 `schema.py`,不外泄。
- **不要**抽 `Input / Output` 包装对象,除非真要跨 CLI / HTTP 边界。

## Cursor IDE 技能 vs 运行时技能(两个不同的层)

`.cursor/skills/*/SKILL.md` 是给**在 Cursor 里编辑这个仓库的 agent**(就是我)看的指引。仓库里 `skills/` 下的运行时项目技能,才是**运行时 agent** 调用的目标。两层可以共享脚本,但读者完全不同。
