"""LLM 调用层。任何 OpenAI 兼容协议的端点都能跑(云上 / 自建 / 本地)。"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Type, TypeVar

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


class OpenAICompatibleClient(LLMClient):
    """适配任意 OpenAI 兼容的 chat 端点。"""

    def __init__(self, base_url: str, api_key: str, model: str, temperature: float = 0.2):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("请先安装 openai 包:`pip install openai>=1.40`") from e
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self.base_url = base_url
        self.model = model
        self._temperature = temperature

    def chat_json(self, system: str, user: str, schema: Type[T]) -> T:
        """让 LLM 输出指定 schema 的 JSON,自动校验为 Pydantic 实例。"""
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False, indent=2)
        sys_with_schema = (
            system
            + "\n\n严格只输出符合以下 JSON Schema 的单个 JSON 对象,"
              "不要解释,不要 Markdown 代码块。\n\nJSON Schema:\n"
            + schema_json
        )
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                temperature=self._temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": sys_with_schema},
                    {"role": "user", "content": user},
                ],
            )
        except Exception as e:
            logger.error("LLM 调用失败(%s / %s):%s", self.base_url, self.model, e)
            raise
        return _parse_json_into_schema(response.choices[0].message.content or "{}", schema)

    def chat(self, system: str, user: str) -> LLMResponse:
        """普通对话,返回原文。"""
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=self._temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return LLMResponse(
            raw_text=response.choices[0].message.content or "",
            base_url=self.base_url,
            model=self.model,
        )


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
