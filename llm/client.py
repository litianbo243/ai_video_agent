"""LLM 调用层。任何 OpenAI 兼容协议的端点都能跑(云上 / 自建 / 本地)。

可选:把每次调用的完整 prompt + response 落成 JSONL,
通过 ``LLMConfig.trace_file`` 或 ``client.set_trace_file(path)`` 启用。
控制台保持单行摘要,详情翻 trace 文件。
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

if TYPE_CHECKING:
    from configs import LLMConfig


@dataclass
class LLMResponse:
    raw_text: str
    base_url: str
    model: str


class LLMClient:
    """LLM 调用接口。"""

    base_url: str = ""
    model: str = ""

    def chat_json(self, system: str, user: str, schema: Type[T]) -> T:
        raise NotImplementedError

    def chat(self, system: str, user: str) -> LLMResponse:
        raise NotImplementedError

    def set_trace_file(self, path: Optional[str]) -> None:
        """设置 / 清空 JSONL trace 落盘路径(运行时可改)。"""
        raise NotImplementedError


class _TraceWriter:
    """LLM 调用 trace 双轨写入器:

    * ``<jsonl_path>``       —— 机器友好 JSONL,每行一次调用,适合 jq / 喂训练数据
    * ``<jsonl_path>.dir/``  —— 人类友好 markdown,一次调用一个文件,
      ``\\n`` 真换行 + 分段标题,直接打开就能读

    线程安全,失败只 warn 不抛。
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.human_dir = self.path.with_suffix("")
        self.human_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._seq = 0

    def write(self, record: Dict[str, Any]) -> None:
        try:
            line = json.dumps(record, ensure_ascii=False)
        except Exception as e:
            logger.warning("trace 序列化失败(record 跳过):%s", e)
            return
        with self._lock:
            self._seq += 1
            seq = self._seq
            try:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception as e:
                logger.warning("trace JSONL 写入失败(%s):%s", self.path, e)
            try:
                self._write_markdown(seq, record)
            except Exception as e:
                logger.warning("trace markdown 写入失败:%s", e)

    def _write_markdown(self, seq: int, record: Dict[str, Any]) -> None:
        method = str(record.get("method", "?"))
        ts_safe = str(record.get("ts", "")).replace(":", "-")
        fname = f"{seq:04d}_{method}_{ts_safe}.md"
        fpath = self.human_dir / fname

        meta_lines = []
        for k in (
            "ts", "method", "model", "base_url", "temperature", "native_model",
            "attempt", "schema", "latency_ms",
            "system_chars", "user_chars", "response_chars",
        ):
            if k in record:
                meta_lines.append(f"- **{k}**: {record[k]}")
        usage = record.get("usage")
        if isinstance(usage, dict):
            for k, v in usage.items():
                meta_lines.append(f"- **{k}**: {v}")
        if "error" in record:
            meta_lines.append(f"- **error**: {record['error']}")

        parts = [
            f"# {method} #{seq}",
            "",
            "## Meta",
            "",
            *meta_lines,
            "",
            "## SYSTEM",
            "",
            str(record.get("system", "")),
            "",
            "## USER",
            "",
            str(record.get("user", "")),
            "",
            "## RESPONSE",
            "",
        ]
        response = str(record.get("response", ""))
        if record.get("schema"):
            parts.append("```json")
            parts.append(_pretty_json(response))
            parts.append("```")
        else:
            parts.append(response)
        fpath.write_text("\n".join(parts), encoding="utf-8")


def _pretty_json(text: str) -> str:
    """把紧凑 JSON 美化成缩进 + 中文不转义。解析失败就原样返回。"""
    if not text.strip():
        return text
    try:
        obj = json.loads(text)
    except Exception:
        return text
    return json.dumps(obj, ensure_ascii=False, indent=2)


class OpenAICompatibleClient(LLMClient):
    """适配任意 OpenAI 兼容的 chat 端点。"""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.2,
        trace_file: Optional[str] = None,
        native_model: bool = False,
        json_max_retries: int = 1,
    ):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("请先安装 openai 包:`pip install openai>=1.40`") from e
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self.base_url = base_url
        self.model = model
        self._temperature = temperature
        self._native_model = native_model
        self._json_max_retries = max(0, json_max_retries)
        self._trace: Optional[_TraceWriter] = (
            _TraceWriter(trace_file) if trace_file else None
        )
        if native_model:
            logger.info(
                "[LLM] grammar-constrained decoding 启用(guided_json,%s)", model,
            )

    def set_trace_file(self, path: Optional[str]) -> None:
        self._trace = _TraceWriter(path) if path else None
        if path:
            logger.info("LLM trace 启用 → %s", path)

    def chat_json(self, system: str, user: str, schema: Type[T]) -> T:
        """让 LLM 输出指定 schema 的 JSON,自动校验为 Pydantic 实例。

        三层防御:

        1. **Schema in prompt**: 把 JSON Schema 拼进 system,LLM 知道契约
        2. **Grammar decoding**(``native_model=True`` 时): 注入 ``guided_json``,
           让本地服务端(vLLM / SGLang / llama.cpp)在采样阶段做 grammar-constrained
           decoding,token 级保证结构合规
        3. **Validation + feedback retry**: 校验失败时把 Pydantic 错误信息回灌给
           LLM 让它修正,重试 ``json_max_retries`` 次

        每次尝试单独写一条 trace(带 ``attempt`` 字段),便于事后分析"哪些 batch
        触发了 retry,LLM 第几次才修对"。
        """
        schema_dict = schema.model_json_schema()
        schema_json = json.dumps(schema_dict, ensure_ascii=False, indent=2)
        sys_with_schema = (
            system
            + "\n\n严格只输出符合以下 JSON Schema 的单个 JSON 对象,"
              "不要解释,不要 Markdown 代码块。\n\nJSON Schema:\n"
            + schema_json
        )
        mode_tag = "grammar" if self._native_model else "json_obj"
        max_attempts = self._json_max_retries + 1
        current_user = user
        last_validation_error: Optional[Exception] = None

        for attempt in range(1, max_attempts + 1):
            attempt_tag = f"{mode_tag}|{attempt}/{max_attempts}"
            logger.info(
                "[chat_json|%s] %s ← system %d 字 + user %d 字 → %s",
                attempt_tag, self.model,
                len(sys_with_schema), len(current_user), schema.__name__,
            )
            create_kwargs: Dict[str, Any] = {
                "model": self.model,
                "temperature": self._temperature,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": sys_with_schema},
                    {"role": "user", "content": current_user},
                ],
            }
            if self._native_model:
                create_kwargs["extra_body"] = {"guided_json": schema_dict}

            t0 = time.perf_counter()
            try:
                response = self._client.chat.completions.create(**create_kwargs)
            except Exception as e:
                elapsed_ms = int((time.perf_counter() - t0) * 1000)
                logger.error(
                    "LLM 调用失败(%s / %s,%d ms):%s",
                    self.base_url, self.model, elapsed_ms, e,
                )
                self._write_trace(
                    method="chat_json", system=sys_with_schema, user=current_user,
                    response_text="", schema_name=schema.__name__,
                    latency_ms=elapsed_ms, usage=None, error=repr(e),
                    attempt=attempt, max_attempts=max_attempts,
                )
                raise
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            response_text = response.choices[0].message.content or "{}"
            usage = _extract_usage(response)

            try:
                result = _parse_json_into_schema(response_text, schema)
            except (ValueError, ValidationError) as ve:
                last_validation_error = ve
                err_msg = str(ve)
                logger.warning(
                    "[chat_json|%s] %s ⚠️ 校验失败 %d ms,in=%s out=%s: %s",
                    attempt_tag, self.model, elapsed_ms,
                    usage.get("prompt_tokens", "?") if usage else "?",
                    usage.get("completion_tokens", "?") if usage else "?",
                    err_msg[:200],
                )
                self._write_trace(
                    method="chat_json", system=sys_with_schema, user=current_user,
                    response_text=response_text, schema_name=schema.__name__,
                    latency_ms=elapsed_ms, usage=usage,
                    error=f"VALIDATION: {err_msg}",
                    attempt=attempt, max_attempts=max_attempts,
                )
                if attempt < max_attempts:
                    current_user = (
                        user
                        + "\n\n=== 上一次返回不合 schema,需要修正 ===\n"
                        + "错误:\n" + err_msg
                        + "\n\n请只修正这些错误后,重新输出**完整**的 JSON 对象。"
                    )
                    continue
                raise ValueError(
                    f"LLM 返回的 JSON 在 {max_attempts} 次尝试后仍不合 schema "
                    f"{schema.__name__}:{err_msg}"
                ) from ve

            logger.info(
                "[chat_json|%s] %s ✓ %d ms, in=%s out=%s",
                attempt_tag, self.model, elapsed_ms,
                usage.get("prompt_tokens", "?") if usage else "?",
                usage.get("completion_tokens", "?") if usage else "?",
            )
            self._write_trace(
                method="chat_json", system=sys_with_schema, user=current_user,
                response_text=response_text, schema_name=schema.__name__,
                latency_ms=elapsed_ms, usage=usage, error=None,
                attempt=attempt, max_attempts=max_attempts,
            )
            return result

        raise last_validation_error or RuntimeError("chat_json retry 循环异常退出")

    def chat(self, system: str, user: str) -> LLMResponse:
        """普通对话,返回原文。"""
        logger.info(
            "[chat] %s ← system %d 字 + user %d 字",
            self.model, len(system), len(user),
        )
        t0 = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                temperature=self._temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except Exception as e:
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            logger.error(
                "LLM 调用失败(%s / %s,%d ms):%s",
                self.base_url, self.model, elapsed_ms, e,
            )
            self._write_trace(
                method="chat", system=system, user=user, response_text="",
                schema_name=None, latency_ms=elapsed_ms, usage=None, error=repr(e),
            )
            raise
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        response_text = response.choices[0].message.content or ""
        usage = _extract_usage(response)
        logger.info(
            "[chat] %s ✓ %d ms, in=%s out=%s",
            self.model, elapsed_ms,
            usage.get("prompt_tokens", "?") if usage else "?",
            usage.get("completion_tokens", "?") if usage else "?",
        )
        self._write_trace(
            method="chat", system=system, user=user, response_text=response_text,
            schema_name=None, latency_ms=elapsed_ms, usage=usage, error=None,
        )
        return LLMResponse(
            raw_text=response_text,
            base_url=self.base_url,
            model=self.model,
        )

    def _write_trace(
        self,
        *,
        method: str,
        system: str,
        user: str,
        response_text: str,
        schema_name: Optional[str],
        latency_ms: int,
        usage: Optional[Dict[str, int]],
        error: Optional[str],
        attempt: int = 1,
        max_attempts: int = 1,
    ) -> None:
        if self._trace is None:
            return
        record: Dict[str, Any] = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "method": method,
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self._temperature,
            "native_model": self._native_model,
            "attempt": f"{attempt}/{max_attempts}",
            "latency_ms": latency_ms,
            "system_chars": len(system),
            "user_chars": len(user),
            "response_chars": len(response_text),
            "system": system,
            "user": user,
            "response": response_text,
        }
        if schema_name is not None:
            record["schema"] = schema_name
        if usage is not None:
            record["usage"] = usage
        if error is not None:
            record["error"] = error
        self._trace.write(record)


def _extract_usage(response: Any) -> Optional[Dict[str, int]]:
    """从 OpenAI 响应里提取 token usage(没有就返回 None)。"""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    out: Dict[str, int] = {}
    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
        v = getattr(usage, k, None)
        if isinstance(v, int):
            out[k] = v
    return out or None


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json_into_schema(text: str, schema: Type[T]) -> T:
    """JSON 校验:整段失败时退化到第一个 ``{...}`` 块再校验。"""
    try:
        return schema.model_validate_json(text)
    except Exception:
        pass
    m = _JSON_BLOCK_RE.search(text)
    if not m:
        raise ValueError(f"LLM 没有返回 JSON 对象。原始返回:\n{text[:500]}")
    try:
        return schema.model_validate_json(m.group(0))
    except ValidationError as e:
        raise ValueError(
            f"LLM 返回的 JSON 不匹配 schema {schema.__name__}:{e}\n"
            f"原始返回(前 500 字):\n{text[:500]}"
        ) from e


def get_client(llm_config: "LLMConfig") -> LLMClient:
    """从 ``LLMConfig`` 构造客户端。"""
    if not llm_config.base_url:
        raise ValueError("llm.base_url 必填(任意 OpenAI 兼容端点 URL)")
    if not llm_config.model:
        raise ValueError("llm.model 必填")
    api_key = llm_config.api_key.get_secret_value() if llm_config.api_key else "EMPTY"
    return OpenAICompatibleClient(
        base_url=llm_config.base_url,
        api_key=api_key,
        model=llm_config.model,
        temperature=llm_config.temperature,
        trace_file=llm_config.trace_file,
        native_model=llm_config.native_model,
        json_max_retries=llm_config.json_max_retries,
    )


# ---------------------------------------------------------------------------
# 冒烟测试:python -m llm.client(在项目根目录运行)
#   - 测 chat 普通对话
#   - 测 chat_json JSON 结构化输出 + Pydantic 校验
# 默认用智谱免费的 GLM-4.7-Flash,需要 ZHIPU_API_KEY 在 .env 或 export 里。
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import logging
    import sys
    from pathlib import Path
    from typing import List

    # 让 `python llm/client.py` 也能直接跑(从项目根加 sys.path)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from pydantic import Field

    from configs import LLMConfig

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = LLMConfig(
        base_url="https://api.deepseek.com",
        model="deepseek-v4-pro",
        api_key_env="DEEPSEEK_API_KEY",
        temperature=0.2,
    )
  
    if cfg.api_key is None:
        print("⚠️  DEEPSEEK_API_KEY 没读到。请确认 .env 或 export 里有这个变量。")
        sys.exit(1)

    client = get_client(cfg)
    print(f"\n客户端就绪: {client.model} @ {client.base_url}\n")

    # ---------- 测试 1:普通对话 ----------
    print("=== 测试 1:chat() 普通对话 ===")
    resp = client.chat(
        system="你是一个简洁的助手。",
        user="用一句话介绍斗破苍穹的主角。",
    )
    print(f"返回:{resp.raw_text}\n")

    # ---------- 测试 2:chat_json 结构化输出 ----------
    print("=== 测试 2:chat_json() JSON 结构化输出 + Pydantic 校验 ===")

    class CharacterBrief(BaseModel):
        name: str = Field(..., description="人物姓名")
        role: str = Field(..., description="主角 / 配角 / 反派 等")
        traits: List[str] = Field(default_factory=list, description="2-3 个特征")

    char = client.chat_json(
        system="从用户给的描述里提取人物档案。",
        user="萧炎,斗破苍穹的主角,从废柴重新崛起,坚韧、聪慧、有时孤傲。",
        schema=CharacterBrief,
    )
    print(f"返回(已校验为 {type(char).__name__}):")
    print(f"  name:   {char.name}")
    print(f"  role:   {char.role}")
    print(f"  traits: {char.traits}")
    print()
    print("冒烟测试通过 ✓")
