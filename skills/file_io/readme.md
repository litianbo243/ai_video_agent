# 技能:`file_io`

本地文件 I/O 的胶水层:读源文本、保存每批检查点、把最终报告同时落成 JSON 与 Markdown。

## 公共 API

```python
from skills.file_io import (
    read_text_file,
    save_checkpoint,
    load_checkpoint,
    write_final_report,
)
```

## 何时调用

- `read_text_file(path)` —— `epub_to_txt` 之后(或者源文件本身就是 `.txt` 时直接调用);
- `save_checkpoint(state, batch_index)` —— 工作流循环里每完成一批就调一次,幂等;
- `load_checkpoint(output_dir)` —— 续跑时,在第一次进入 `analyze` 之前;
- `write_final_report(report, output_dir)` —— 整个 run 结束时调一次。

## 输出约定

```
{output_dir}/
├── batch_states/
│   ├── batch_0001.json       # 每批一个 BatchState 快照
│   ├── batch_0002.json
│   └── ...
├── screenplay.json           # ScreenplayAnalysis(机器可读)
├── screenplay.md             # ScreenplayAnalysis(人类可读)
├── characters.json           # CharacterRoster(机器可读)
├── characters.md             # CharacterRoster(人类可读)
└── meta.json                 # ReportMeta
```

## 不变量

1. 所有 JSON 都通过 Pydantic v2 round-trip(用 `model_validate_json`);
2. 检查点文件名零填充到 4 位,这样 `sorted()` 顺序就是批次顺序;
3. Markdown 是从 JSON 生成的,反方向不允许。
