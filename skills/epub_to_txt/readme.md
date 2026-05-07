# 技能:`epub_to_txt`

把单个 `.epub` 文件转成 UTF-8 `.txt`,保留 spine 中的章节顺序。

## 何时调用

- 输入文件后缀是 `.epub`,而下游流程需要 `.txt`;
- 源文件旁边的缓存 `.txt` 还不存在,或者比源 `.epub` 旧。

何时**不**调用:

- 输入已经是 `.txt`;
- 缓存 `<stem>.txt` 已经比 `<stem>.epub` 新(函数本身也会自动短路,但调用前判断一下能省掉无谓的衔接逻辑)。

## 公共 API

```python
from skills.epub_to_txt import epub_to_txt, epub_to_txt_batch

out_path = epub_to_txt("book.epub")                    # 写到 book.txt(同目录)
out_path = epub_to_txt("book.epub", "out/book.txt")    # 显式指定输出路径
written = epub_to_txt_batch(["a.epub", "b.epub"], "out_dir/")
```

## 输出契约

- 一个 UTF-8 文本文件,路径就是返回的 `Path`;
- 若 OPF 元数据里有书名,文件首行写 `# {title}`,随后是空行;
- 章节正文按 spine 顺序排列,空行分隔;
- 块级 HTML(`<p>`、`<div>`、`<h1>` …)收敛为换行;
- `<script>`、`<style>`、`<head>`、`<title>` 中的内容直接丢弃;
- 行内标签剥离不保留。

## 不变量

1. spine 顺序保留 —— 永远不要按文件名重新排序;
2. 仅依赖标准库,无第三方依赖;
3. 幂等:当 `out.mtime >= src.mtime` 时直接短路。

## 反例

- 不要传非 EPUB 的 ZIP 进来;会抛 `ValueError`;
- 不要指望 PDF 那种复杂排版能保留 —— 仅块级换行能传过去;
- 缓存 `.txt` 是新的就**不要**重新转换。
