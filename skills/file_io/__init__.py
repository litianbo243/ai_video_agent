"""file_io skill:读小说源文件 + 写最终报告(JSON + Markdown)。

公开 API::

    from skills.file_io import read_text_file, write_final_report
    from schemas import FinalReport, ReportMeta, AgentLLMInfo

**schema 不在本模块 re-export**:所有 Pydantic 数据契约集中在顶层 ``schemas/``
包,业务代码请直接 ``from schemas import X``。
"""

from skills.file_io.logic import read_text_file, write_final_report

__all__ = [
    "read_text_file",
    "write_final_report",
]
