"""file_io skill:读小说源文件 + 按需写工作流产物(JSON + Markdown)。

公开 API::

    from skills.file_io import read_text_file, write_partial
    from schemas import FinalReport, ReportMeta, AgentLLMInfo

``write_partial`` 是统一写盘入口:传哪些字段就落哪些文件,跨 mode 通用。

**schema 不在本模块 re-export**:所有 Pydantic 数据契约集中在顶层 ``schemas/``
包,业务代码请直接 ``from schemas import X``。
"""

from skills.file_io.logic import read_text_file, write_partial

__all__ = [
    "read_text_file",
    "write_partial",
]
