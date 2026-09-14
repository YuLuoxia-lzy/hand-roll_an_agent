"""ReflectionAgent —— 写初稿 -> 自己评审 -> 按意见改, 迭代几轮

跑法(仓库根目录下):
    python examples/reflection.py
"""

import os
import sys

from agents0to1 import Agents0to1, ReflectionAgent

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _get_api_key() -> str:
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        raise RuntimeError(
            "未找到 API 密钥。请先设置环境变量 DEEPSEEK_API_KEY 或 LLM_API_KEY。\n"
            "也可以把仓库根目录的 .env.example 复制成 .env 再填写。"
        )
    return api_key


def main():
    llm = Agents0to1(
        provider="deepseek",
        model="deepseek-chat",
        api_key=_get_api_key(),
    )

    # max_iterations: 最多"评审 + 改稿"几轮。评审说"无需改进"会提前结束
    agent = ReflectionAgent(
        name="3号reflection",
        llm=llm,
        max_iterations=3,
        system_prompt="你是一位严谨的技术写作者，擅长把复杂概念讲清楚。",
    )

    print("给我一个任务吧（比如：写一段介绍快速排序的说明）")
    while True:
        pb = input()
        print()
        if pb == "bye":
            break

        print(agent.run(pb))
        print()
        print("还有什么可以帮到您")

    print("希望您生活愉快")


if __name__ == "__main__":
    main()
