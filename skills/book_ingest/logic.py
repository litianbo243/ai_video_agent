"""把"原始小说路径"变成"标题 + batches"。

是 ``skills.epub_to_txt`` 和 ``skills.batch_chapters`` 的薄组合,与具体 workflow
无关。两层 API:

* ``load_and_batch_txt(txt_path, ...)`` —— 从 ``.txt`` 加载 + 切批
* ``ingest_book(input_path, ...)``       —— ``.txt`` / ``.epub`` 一把通吃
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from skills.batch_chapters import Batch
from skills.batch_chapters import load_text, split_into_batches
from skills.epub_to_txt import epub_to_txt

logger = logging.getLogger(__name__)


def load_and_batch_txt(
    txt_path: str | Path,
    *,
    max_batch_chars: int = 8_000,
    max_total_chars: int = 0,
) -> Tuple[str, int, int, List[Batch]]:
    """从 ``.txt`` 加载并切批,返回 ``(title, raw_chars, total_chars, batches)``。

    ``max_total_chars`` > 0 时从尾部截断到该字数,用于快速试跑。
    """
    txt_path = Path(txt_path)
    logger.info("[ingest] 加载文本:%s", txt_path)
    t0 = time.perf_counter()
    title, body = load_text(txt_path)
    raw_chars = len(body)

    if max_total_chars > 0 and raw_chars > max_total_chars:
        body = body[:max_total_chars]
        logger.warning(
            "[ingest] 全文 %d 字超过 max_total_chars=%d,从尾部截断到 %d 字",
            raw_chars, max_total_chars, len(body),
        )

    total_chars = len(body)
    batches = split_into_batches(body, max_chars=max_batch_chars)
    elapsed = time.perf_counter() - t0
    logger.info(
        "[ingest] 用时 %.1f 秒:书名=%r,%d 字 → 实际 %d 字,打包 %d 批(≤ %d 字/批)",
        elapsed, title, raw_chars, total_chars, len(batches), max_batch_chars,
    )
    return title, raw_chars, total_chars, batches


@dataclass
class IngestResult:
    title: str
    txt_path: Path
    raw_chars: int       # 截断前的总字数
    total_chars: int     # 截断后实际进 batch 的字数
    batches: List[Batch]


def ingest_book(
    input_path: str | Path,
    *,
    max_batch_chars: int = 8_000,
    max_total_chars: int = 0,
) -> IngestResult:
    """``.txt`` 或 ``.epub`` 一把变成 ``IngestResult``。"""
    src = Path(input_path)
    if not src.is_file():
        raise FileNotFoundError(f"输入文件不存在:{src}")
    suffix = src.suffix.lower()
    if suffix not in {".txt", ".epub"}:
        raise ValueError(f"不支持的文件类型 {suffix!r};只接受 .txt 与 .epub。")

    size_mb = src.stat().st_size / (1024 * 1024)
    logger.info("[ingest] 输入:%s(%s,%.2f MB)", src, suffix, size_mb)

    if suffix == ".epub":
        logger.info("[ingest] 开始 EPUB → TXT")
        t0 = time.perf_counter()
        txt_path = Path(epub_to_txt(str(src)))
        logger.info("[ingest] EPUB → TXT 用时 %.1f 秒,产物 %s",
                    time.perf_counter() - t0, txt_path)
    else:
        txt_path = src

    title, raw_chars, total_chars, batches = load_and_batch_txt(
        txt_path,
        max_batch_chars=max_batch_chars,
        max_total_chars=max_total_chars,
    )
    return IngestResult(
        title=title,
        txt_path=txt_path,
        raw_chars=raw_chars,
        total_chars=total_chars,
        batches=batches,
    )


__all__ = ["IngestResult", "ingest_book", "load_and_batch_txt"]
