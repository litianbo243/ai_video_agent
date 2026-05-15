"""文本分批 skill。

把整本小说的 ``.txt`` 切成一段段 ``Batch``,每个 batch 不超过 ``max_chars`` 字。

切分点优先级::

    章节标题(行首独占,严格正则) > 段落边界(\\n\\n) > 句末标点(。!?) > 硬切

为什么章节优先?中文小说"第 N 章"通常天然对齐"剧情段边界",beat agent 跨 batch
做"延续 vs 新起"判断时受益巨大。退化链是逐级安全网:章节匹配不到不会更差,只是
退到原来的段落切分。

不关心章节内部结构;每段用稳定的 1-based ``Batch.index`` 标识(可作 checkpoint 锚点)。
``Batch`` 数据结构定义在 ``schema.batch``,跨模块共享。
"""

from __future__ import annotations

import bisect
import logging
import re
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

from skills.batch_chapters.schema import Batch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 章节正则:严格,宁漏勿错
# ---------------------------------------------------------------------------
#
# 要求:
#   * 行首开始(MULTILINE 下的 ``^``)
#   * 章节号:中文「一二三四五六七八九十百千万零两」或阿拉伯数字
#   * "章" 字
#   * 至少 1 个空白(全角 / 半角 / 普通空格)
#   * 标题正文 ≤ 40 字
#   * 整行结束(``$``)
#
# 这样的设计避免误匹配:
#   * "正如第三章那次..."     ← 不在行首,跳过
#   * "(作者:第二十三章求订阅)"  ← 不在行首,跳过
#   * 跨章引用、章末感言         ← 不在行首,跳过
#
# 真正会被匹配的:
#   * "第一章 陨落的天才"
#   * "第三章 客人【求收藏,求推荐票^_^】"   ← 副标题在 40 字内,OK
CHAPTER_PAT = re.compile(
    r"^(第[一二三四五六七八九十百千万零两\d]+章[\s 　]+.{0,40})$",
    flags=re.MULTILINE,
)


# ---------------------------------------------------------------------------
# 文件读 + 标题剥离
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# 切批主入口
# ---------------------------------------------------------------------------


def split_into_batches(text: str, max_chars: int = 8_000) -> List[Batch]:
    """把文本切成 ≤ ``max_chars`` 的 batch 列表。

    切分点优先级:**章节标题 > 段落边界 > 句末标点 > 硬切**。
    每级落空才退到下一级,所以章节匹配不到不会比"只按段落切"差。
    """
    if max_chars <= 0:
        raise ValueError("max_chars 必须大于 0")

    text = text.strip()
    if not text:
        return []

    chapter_starts = _find_chapter_starts(text)
    if chapter_starts:
        logger.info(
            "[batch] 检测到 %d 个章节标题,启用章节边界优先切批",
            len(chapter_starts),
        )
    else:
        logger.info("[batch] 未检测到章节标题(或正则不匹配),退到段落 / 句末优先")

    cuts: List[Tuple[int, int, str]] = []  # (start, end, by)
    pos = 0
    n = len(text)
    while pos < n:
        if n - pos <= max_chars:
            cuts.append((pos, n, "remainder"))
            break
        target = pos + max_chars
        floor = pos + max_chars // 2
        cut_pos, by = _find_safe_cut(text, target, floor, chapter_starts)
        if cut_pos <= pos:
            # 防御性:如果出现退化也跳不动,硬切兜底,避免死循环
            cut_pos, by = target, "hard"
        cuts.append((pos, cut_pos, by))
        pos = cut_pos

    by_counter = Counter(b for _, _, b in cuts)
    sizes = [e - s for s, e, _ in cuts]
    logger.info(
        "[batch] 切成 %d batch | 切点构成: %s | 字数 min=%d max=%d avg=%d",
        len(cuts),
        ", ".join(f"{k}={v}" for k, v in by_counter.most_common()),
        min(sizes), max(sizes), sum(sizes) // len(sizes),
    )
    if by_counter.get("hard", 0) > 0:
        logger.warning(
            "[batch] 有 %d 个 batch 走了硬切,可能切在不自然的位置",
            by_counter["hard"],
        )

    out: List[Batch] = []
    for i, (s, e, _) in enumerate(cuts, start=1):
        chunk = text[s:e].strip()
        if chunk:
            out.append(Batch(index=len(out) + 1, text=chunk))
    return out


# ---------------------------------------------------------------------------
# 内部:找章节 + 找切点
# ---------------------------------------------------------------------------


def _find_chapter_starts(text: str) -> List[int]:
    """全文扫一遍 ``CHAPTER_PAT``,返回所有章节标题行的起始字符索引(已排序)。"""
    return [m.start() for m in CHAPTER_PAT.finditer(text)]


def _find_safe_cut(
    text: str,
    target: int,
    floor: int,
    chapter_starts: Optional[List[int]],
) -> Tuple[int, str]:
    """在 ``(floor, target]`` 区间找最优切点。

    返回 ``(cut_pos, by)``。``by`` 为 ``'chapter' / 'paragraph' / 'sentence' / 'hard'``。

    优先级:
      1. **章节标题起点**:窗口内有章节匹配,贪心选最靠近 ``target`` 的那个
         (让 batch 字数最大化,避免切得过碎)
      2. **段落边界** ``\\n\\n``:同样选最靠近 ``target``
      3. **句末标点**:。?! 等
      4. **硬切**:都失败时直接 ``target`` 截断
    """
    # 1. 章节边界(贪心选最靠近 target 的)
    if chapter_starts:
        # bisect 找到 chapter_starts 里第一个 >= target 的位置;
        # 它前一个就是 ≤ target 的最大值,再校验是否 > floor。
        idx = bisect.bisect_right(chapter_starts, target)
        if idx > 0:
            candidate = chapter_starts[idx - 1]
            if candidate > floor:
                return candidate, "chapter"

    # 2. 段落边界
    cut = text.rfind("\n\n", floor, target)
    if cut > 0:
        return cut + 2, "paragraph"

    # 3. 句末标点
    for punct in ("。", "？", "！", "?", "!", ";", ";"):
        c = text.rfind(punct, floor, target)
        if c > 0:
            return c + 1, "sentence"

    # 4. 硬切
    return target, "hard"


__all__ = [
    "load_text",
    "split_into_batches",
    "CHAPTER_PAT",  # 暴露给外部 preview / debug 工具用
]
