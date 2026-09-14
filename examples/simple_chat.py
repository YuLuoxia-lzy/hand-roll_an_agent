"""SimpleAgent —— 一问一答, 记住上下文

跑法(仓库根目录下):
    python examples/simple_chat.py
"""

import os
import sys

from agents0to1 import Agents0to1, SimpleAgent

# Windows 控制台默认不是 UTF-8, 不重新配置的话中文会变成乱码
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
    # 先创建 llm api 链接
    llm = Agents0to1(
        provider="deepseek",
        api_key=_get_api_key(),
        model="deepseek-chat",
    )

    simple = SimpleAgent(
        name="0号simper_agent",
        llm=llm,
        system_prompt="你是心理学的专家，你的语气总是十分的友好并且可以开导任何人的任何难题，以治疗心里疾病。 (在内容最后添加一个 好些了吗)",
    )

    print("您有什么问题吗？")
    while True:
        pb = input()
        print()
        if pb == "bye":
            break

        # 流式: 逐块打印正文增量, 最后一条是 final
        for event in simple.stream_run(pb):
            if event.type == "text":
                print(event.text, end="", flush=True)
        print()
        print("还有什么可以帮到您")

    print("希望您生活愉快")


if __name__ == "__main__":
    main()
