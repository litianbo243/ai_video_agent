"""单批场景抽取:一次 LLM 调用,返回本批增量。

合并(同名 description 覆盖、新增赋 index)放在 ``manager.py``。
"""

from __future__ import annotations

import logging
from typing import Dict

from llm.client import LLMClient
from skills.batch_chapters import Batch
from skills.extract_settings.schema import Setting, SettingExtraction

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """\
你是中文长篇小说的场景抽取助手,正在为下游 AI 短视频生产管线构建结构化数据。

工作模式: 增量分析。
- 每次只看一批章节正文 + 此前已知的场景名单;
- 输出"本批次产生的变化"(delta)。

────────────────────────────────────────────────
对每个**物理地点**(跨剧情段复用,跟具体事件无关),产出:
- name: 地点名,**跨批稳定**,如"萧家祠堂"、"乌坦城广场"、"加玛帝国魔兽山脉"
    不要含时机修饰("·序章夜")或事件修饰("·撕婚约");
    同一物理地点不同时段、不同事件共用 1 个 Setting。
- description: **整段视觉环境描写,150-300 字**
    含建筑/布局/家具/光线/氛围/关键道具
    例:"萧家祠堂位于宅院中轴线最深处,青砖灰瓦,飞檐翘角;殿内供奉历代家主灵位,
    香烟缭绕,黄铜烛台林立;正中铺设暗红地毯,两侧木雕栏杆透着古朴气息;
    殿顶悬挂三盏八角宫灯,光线幽暗压抑。"

────────────────────────────────────────────────
合并规则:
- 已在 [此前已知场景名单] 中的地点,**沿用同名,不要改名**
- 老地点只在你看到比之前更详细的描写时才输出更新;否则不输出
- 一闪而过的次要地点(走过的街道、路边茶摊)**不要**输出,只挑有戏剧分量的
- 绝不杜撰: 没有依据的字段直接留空

────────────────────────────────────────────────
严格按 JSON Schema 输出 SettingExtraction,只输出一个 JSON 对象。
"""


def _condense_known(known: Dict[str, Setting]) -> str:
    if not known:
        return "(暂无)"
    return "\n".join(f"- {name}" for name in known.keys())


def _build_user_prompt(batch: Batch, known: Dict[str, Setting], title: str) -> str:
    settings_str = _condense_known(known)
    book_title = title or "(未提供书名)"
    return (
        f"书名: {book_title}\n"
        f"批次序号: 第 {batch.index} 批\n"
        f"批次字数: 约 {batch.char_count}\n\n"
        f"=== 此前已知场景名单 ===\n{settings_str}\n\n"
        f"=== 本批次正文 ===\n{batch.render_for_prompt()}\n\n"
        f"=== 任务 ===\n请按 JSON Schema 输出 SettingExtraction。"
    )


def extract_for_batch(
    batch: Batch,
    known: Dict[str, Setting],
    llm: LLMClient,
    *,
    title: str = "",
) -> SettingExtraction:
    """对单个 batch 调一次 LLM,返回场景增量。"""
    user_prompt = _build_user_prompt(batch, known, title)
    logger.info(
        "setting_extractor 第 %d 批(已知 %d 处),%s @ %s",
        batch.index, len(known), llm.model, llm.base_url,
    )
    delta = llm.chat_json(SYSTEM_PROMPT, user_prompt, SettingExtraction)
    logger.info(
        "setting_extractor 第 %d 批产出:%d 处场景",
        batch.index, len(delta.new_or_updated_settings),
    )
    return delta


__all__ = ["extract_for_batch", "SYSTEM_PROMPT"]
