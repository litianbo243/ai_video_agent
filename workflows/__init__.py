"""LangGraph workflow:编排 agent + skill 跑流水线。

提供两个入口:
* ``novel_analysis`` —— 主流水线,按 mode 早停,产出剧本/人物/剧情段
* ``generate_character_prompt`` —— 独立 workflow,输入 characters.json,产出 SDXL 图像提示词
"""
