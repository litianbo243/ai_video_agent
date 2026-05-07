"""把 .epub 电子书转换成 UTF-8 .txt。

只用标准库 —— 这是刻意为之,任何 agent 在一台干净的机器上都能直接调用,
不必先装第三方依赖。

契约(对应 .cursor/skills/epub-to-txt/SKILL.md):

* 输出是单个 UTF-8 文本文件。
* 如果 OPF 元数据里能拿到书名,文件首行写 ``# {title}``;否则首行就是
  第一章正文。
* 章节正文按 spine 顺序排列,以空行分隔。
* 块级 HTML 标签(``<p>``, ``<div>``, ``<h1>`` ...)收敛成换行。
* 行内标签直接剥掉。
* ``<script>``、``<style>``、``<head>``、``<title>`` 中的内容全部丢弃。
* 缓存:若 ``<stem>.txt`` 比 ``<stem>.epub`` 新,则跳过重复转换。
"""

from __future__ import annotations

import html
import os
import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from xml.etree import ElementTree as ET

_BLOCK_TAGS = {
    "p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
    "blockquote", "pre", "section", "article", "header", "footer",
    "ul", "ol", "table",
}
_DROP_TAGS = {"script", "style", "head", "title", "meta", "link"}

_NS_OPF = "{http://www.idpf.org/2007/opf}"
_NS_DC = "{http://purl.org/dc/elements/1.1/}"
_NS_CONTAINER = "{urn:oasis:names:tc:opendocument:xmlns:container}"


class _BlockTextParser(HTMLParser):
    """把章节 HTML 转成纯文本,块级标签替换为换行。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._buf: List[str] = []
        self._drop_depth = 0

    def handle_starttag(self, tag: str, attrs):  # type: ignore[no-untyped-def]
        tag = tag.lower()
        if tag in _DROP_TAGS:
            self._drop_depth += 1
            return
        if tag in _BLOCK_TAGS:
            self._buf.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _DROP_TAGS:
            self._drop_depth = max(0, self._drop_depth - 1)
            return
        if tag in _BLOCK_TAGS:
            self._buf.append("\n")

    def handle_startendtag(self, tag: str, attrs):  # type: ignore[no-untyped-def]
        if tag.lower() == "br":
            self._buf.append("\n")

    def handle_data(self, data: str) -> None:
        if self._drop_depth:
            return
        self._buf.append(data)

    def get_text(self) -> str:
        text = "".join(self._buf)
        text = html.unescape(text)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def _read_member(zf: zipfile.ZipFile, name: str) -> str:
    return zf.read(name).decode("utf-8", errors="replace")


def _opf_path(zf: zipfile.ZipFile) -> str:
    container = ET.fromstring(zf.read("META-INF/container.xml"))
    rootfile = container.find(f".//{_NS_CONTAINER}rootfile")
    if rootfile is None or rootfile.get("full-path") is None:
        raise ValueError("EPUB 的 container.xml 中缺少 rootfile/full-path")
    return rootfile.get("full-path")  # type: ignore[return-value]


def _parse_opf(zf: zipfile.ZipFile, opf_path: str) -> Tuple[Optional[str], List[str]]:
    """从 OPF 的 manifest + spine 中读出 (书名, 章节文件路径列表)。"""
    opf_root = ET.fromstring(zf.read(opf_path))

    title_el = opf_root.find(f".//{_NS_DC}title")
    title = (title_el.text or "").strip() if title_el is not None else None

    manifest: dict = {}
    for item in opf_root.iter(f"{_NS_OPF}item"):
        item_id = item.get("id")
        href = item.get("href")
        media = (item.get("media-type") or "").lower()
        if item_id and href and ("html" in media or "xml" in media):
            manifest[item_id] = href

    base = os.path.dirname(opf_path)
    chapter_paths: List[str] = []
    for itemref in opf_root.iter(f"{_NS_OPF}itemref"):
        idref = itemref.get("idref")
        if idref and idref in manifest:
            joined = os.path.normpath(os.path.join(base, manifest[idref])) if base else manifest[idref]
            chapter_paths.append(joined.replace("\\", "/"))

    return title, chapter_paths


def _html_to_text(raw: str) -> str:
    parser = _BlockTextParser()
    parser.feed(raw)
    parser.close()
    return parser.get_text()


def epub_to_txt(epub_path: os.PathLike, out_path: Optional[os.PathLike] = None) -> Path:
    """把单个 EPUB 转成 TXT,返回输出路径。

    带缓存:若 ``out_path`` 已存在且比 ``epub_path`` 新,直接返回不重新转换。
    """
    src = Path(epub_path)
    if not src.exists():
        raise FileNotFoundError(src)
    if src.suffix.lower() != ".epub":
        raise ValueError(f"不是 .epub 文件:{src}")

    out = Path(out_path) if out_path is not None else src.with_suffix(".txt")

    if out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
        return out

    with zipfile.ZipFile(src) as zf:
        opf_path = _opf_path(zf)
        title, chapter_paths = _parse_opf(zf, opf_path)
        chapters: List[str] = []
        for cp in chapter_paths:
            try:
                raw = _read_member(zf, cp)
            except KeyError:
                continue
            text = _html_to_text(raw)
            if text:
                chapters.append(text)

    body = "\n\n".join(chapters)
    if title:
        content = f"# {title}\n\n{body}\n"
    else:
        content = body + "\n"

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    return out


def epub_to_txt_batch(
    sources: Iterable[os.PathLike], out_dir: Optional[os.PathLike] = None
) -> List[Path]:
    """批量转换多个 EPUB,按输入顺序返回写入的输出路径。"""
    written: List[Path] = []
    out_root = Path(out_dir) if out_dir is not None else None
    if out_root is not None:
        out_root.mkdir(parents=True, exist_ok=True)
    for src in sources:
        srcp = Path(src)
        if out_root is not None:
            out = out_root / (srcp.stem + ".txt")
        else:
            out = None
        written.append(epub_to_txt(srcp, out))
    return written
