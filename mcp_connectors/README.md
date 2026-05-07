# `mcp_connectors/`

预留给将来的 MCP(Model Context Protocol)适配器。

当前 MVP 不需要 MCP,因为 agent 调用的所有工具都在本地进程内:

- 文件转换:`skills/epub_to_txt/`
- 章节解析 + 分批:`skills/batch_chapters/`
- LLM I/O:`llm/client.py`
- 落盘持久化:`skills/file_io/`

## 什么时候需要在这里加一个 MCP 连接器

需要让 agent 调用 **进程外** 的东西时:

- **把分析器封装成 MCP server**,让 Cursor / Open WebUI / Dify 把它当成可调用工具。把 `agents.novel_analysis.manager.run` 暴露成 MCP 工具(例如 `analyze_novel(path, max_chars) -> FinalReport`),放到 `mcp_connectors/novel_analysis_mcp.py`。
- **从远程源抓小说**(内部 CMS、Notion 数据库等)。在这里加 `mcp_connectors/<source>_mcp.py`,然后在 `agents/novel_analysis/workflow.py` 的 `detect_input` 之前插一个新节点调用它。
- **把报告写到数据库**而不是本地文件系统。加 `mcp_connectors/database_mcp.py` 替换 `write` 节点。

## 编写连接器的约定

每个连接器文件应当:

1. 配置 MCP server endpoint(URL / stdio 命令、鉴权、超时);
2. 暴露一层薄的 Python 门面,返回 [`schema/novel_analysis.py`](../schema/novel_analysis.py) 中的 Pydantic 模型;
3. 在同目录下的 `readme.md` 里说明它提供哪些工具,这样 manager agent 才能学会使用它。
