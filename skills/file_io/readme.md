# 技能:`file_io`

本地文件 I/O 的胶水层:读源文本、把流水线产物按需落成 JSON + Markdown。

## 公共 API

```python
from skills.file_io import (
    read_text_file,
    write_partial,
)
```

## 何时调用

- `read_text_file(path)` —— `epub_to_txt` 之后(或者源文件本身就是 `.txt` 时直接调用)。
- `write_partial(output_dir, *, characters=None, beats=None, episode_plans=None,
  screenplay=None, meta=None)` —— 工作流末尾调一次,**传哪些字段就落哪些**。
  跨 mode 通用:浅 mode 只传到 `characters` / `beats`,深 mode 全传。

## 输出约定

```
{output_dir}/
├── characters.json / .md      # characters 非 None 时落
├── beats.json / .md           # beats 非 None 时落
├── episode_plans.json         # episode_plans 非 None 时落(无 .md)
├── screenplay.json / .md      # screenplay 非 None 时落;.md 需要 meta 配合
└── meta.json                  # meta 非 None 时落(`novel_analysis` 各 mode 都生成)
```

## 不变量

1. 所有 JSON 都通过 Pydantic v2 round-trip(用 `model_validate_json`);
2. Markdown 是从 JSON 生成的,反方向不允许;
3. `write_partial` **不抛**也不写空文件 —— 传 None 就跳过对应输出。
