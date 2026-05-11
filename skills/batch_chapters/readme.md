# 技能:`batch_chapters`

把整本中文小说的 `.txt` 切成连续的 `Batch`,每个 batch 不超过给定的字符预算。
切分点优先选**段落边界**(`\n\n`),其次**句末标点**(。!?),实在不行才硬切;
保证不会切在词中间。**不关心章节**;每段用稳定的 1-based `Batch.index` 标识。

## 何时调用

- 输入已规范化成 UTF-8 `.txt` 之后;
- 任何会消费正文的 LLM 调用之前;
- 想换一个 LLM 预算重新分批时(用新的 `max_chars` 调用即可)。

## 公共 API

```python
from skills.batch_chapters import load_text, split_into_batches
from skills.batch_chapters import Batch

title, body = load_text(Path("input.txt"))

batches: list[Batch] = split_into_batches(body, max_chars=8_000)
for batch in batches:
    batch.index                # 1-based 批次号
    batch.text                 # 该批次的纯文本
    batch.char_count           # = len(batch.text)
    batch.render_for_prompt()  # 直接可发给 LLM 的正文(等于 .text)
    # Batch 是 Pydantic 模型,自动支持 model_dump_json / model_validate_json
```

## 输入

- 单个 `.txt`(可能含 `# {title}\n\n` 头,会被 `load_text` 自动剥离)。

如果输入是 `.epub`,先跑 `epub_to_txt`。

## 如何选 `max_chars`

| 预算 | 适用场景 | 一本 50w 字小说约多少批 |
|------|----------|------------------------|
| 4,000 | 小模型 / output 紧的本地 LLM | ~125 |
| 8,000 | **默认** —— GLM-4.7-Flash / DeepSeek-V3.2 这一档 | ~63 |
| 16,000 | 大模型 / 长 context | ~32 |

挑选公式:`max_chars = model_context_window - running_state_size - system_prompt_size - expected_output_size`。

## 不变量

1. 文本顺序在所有 batch 之间保留;
2. 切分点选段落 / 句末,**不会**切在词或句子中间;
3. `Batch.index` 全局 1-based 单调递增,适合作 checkpoint / 续跑锚点。
