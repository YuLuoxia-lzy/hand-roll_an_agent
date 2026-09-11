"""Agent 测试2"""

import os
import sys

from ...core import Agents0to1
from ...classic_agent import PlanAndSolveAgent

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def _get_api_key() -> str:
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        raise RuntimeError(
            "未找到 API 密钥。请先设置环境变量 DEEPSEEK_API_KEY 或 LLM_API_KEY。\n"
        )
    return api_key

def play_pAs():
    llm = Agents0to1(
        provider="deepseek",
        model = "deepseek-chat",
        api_key = _get_api_key()
    )

    pAs = PlanAndSolveAgent(
        name = "1号pAs",
        llm = llm,
        system_prompt = "你是心理学的专家，你的语气总是十分的友好并且可以开导任何人的任何难题，以治疗心里疾病。 (在内容最后添加一个 好些了吗)"
    )

    print("您有什么问题吗？")
    while(True):
        pb = input()
        print()
        if pb == "bye":
            break
        print(pAs.run(pb))
        print()
        print("还有什么可以帮到您")

    print("希望您生活愉快")