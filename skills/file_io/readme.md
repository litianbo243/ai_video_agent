# 技能:`file_io`

本地文件 I/O 的胶水层:读源文本、把最终报告同时落成 JSON 与 Markdown。

## 公共 API

```python
from skills.file_io import (
    read_text_file,
    write_final_report,
)
```

## 何时调用

- `read_text_file(path)` —— `epub_to_txt` 之后(或者源文件本身就是 `.txt` 时直接调用);
- `write_final_report(report, output_dir)` —— 整个 run 结束时调一次,`novel_analysis`
  顶层 agent 在 3 个子-agent 跑完后调用。

## 输出约定

```
{output_dir}/
├── screenplay.json / .md     # ScreenplayAnalysis(分集 + 每集分镜)
├── characters.json / .md     # CharacterRoster
├── beats.json / .md          # BeatList(beat.setting_refs 是字符串 label,无独立档案)
└── meta.json                 # ReportMeta
```

## 不变量

1. 所有 JSON 都通过 Pydantic v2 round-trip(用 `model_validate_json`);
2. Markdown 是从 JSON 生成的,反方向不允许。
