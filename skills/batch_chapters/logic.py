"""文本分批 skill。

把整本小说的 ``.txt`` 切成一段段 ``Batch``,每个 batch 不超过 ``max_chars`` 字。
切分点优先选**段落边界**(``\\n\\n``),其次**句末标点**(。!?),实在不行才硬切;
保证不会切在词中间。

不关心章节;每段用稳定的 1-based ``Batch.index`` 标识(可作 checkpoint 锚点)。
``Batch`` 数据结构定义在 ``schema.batch``,跨模块共享。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from skills.batch_chapters.schema import Batch


def _strip_title_marker(text: str) -> Tuple[str, str]:
    """``epub_to_txt`` 在文件最前会加 ``# {title}\\n\\n``,这里剥离。

    返回 ``(book_title, body)``。无标记时 ``book_title`` 为空。
    """
    if text.startswith("# "):
        first_nl = text.find("\n")
        if first_nl == -1:
            return text[2:].strip(), ""
        return text[2:first_nl].strip(), text[first_nl + 1 :].lstrip()
    return "", text


def load_text(path: Path) -> Tuple[str, str]:
    """读一个 ``.txt`` 文件,返回 ``(title, body)``;``title`` 缺失时回退到 stem。"""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    title, body = _strip_title_marker(p.read_text(encoding="utf-8"))
    return title or p.stem, body


def split_into_batches(text: str, max_chars: int = 8_000) -> List[Batch]:
    """把文本切成 ≤ ``max_chars`` 的 batch 列表。

    切分点优先级:段落边界 > 句末标点 > 硬切。
    """
    if max_chars <= 0:
        raise ValueError("max_chars 必须大于 0")

    text = text.strip()
    batches: List[Batch] = []
    idx = 1

    while text:
        if len(text) <= max_chars:
            batches.append(Batch(index=idx, text=text))
            break
        cut = _find_safe_cut(text, max_chars)
        chunk = text[:cut].strip()
        if chunk:
            batches.append(Batch(index=idx, text=chunk))
            idx += 1
        text = text[cut:].lstrip()

    return batches


def _find_safe_cut(text: str, target: int) -> int:
    """在 ``target`` 之前找最优切点(返回字符索引,即切点之后第一个字符)。

    要求切点不能离 ``target`` 太远(留 50% 余量),否则退化到硬切。
    """
    floor = target // 2  # 不要回退超过一半距离

    # 1. 段落边界
    cut = text.rfind("\n\n", floor, target)
    if cut > 0:
        return cut + 2

    # 2. 句末标点
    for punct in ("。", "？", "！", "?", "!", ";", ";"):
        c = text.rfind(punct, floor, target)
        if c > 0:
            return c + 1

    # 3. 硬切
    return target


__all__ = [
    "load_text",
    "split_into_batches",
]
