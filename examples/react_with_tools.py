"""ReActAgent —— 循环调用工具, 直到不需要为止

这里用 stream_run 跑, 把 AgentEvent 按类型分别渲染:
    text        最终答案的正文增量(逐块)
    thinking    模型调工具之前的过程叙述(整段)
    tool_call   模型请求调用某个工具
    tool_result 该工具的执行结果
    final       最终答案(整段)

跑法(仓库根目录下):
    python examples/react_with_tools.py
"""

import os
import sys

from agents0to1 import Agents0to1, ReActAgent, ToolRegistry, CalculatorTool

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

    # 工具要先注册进 ToolRegistry, 模型才看得见它们(走 API 的 tools 参数)
    tools = ToolRegistry()
    tools.register_tool(CalculatorTool())

    rea = ReActAgent(
        name="2号rea",
        llm=llm,
        tool_registry=tools,
        system_prompt="你是数学的专家，总可以以专业角度分析解决问题并能给出通俗易懂的回答 (在内容最后添加一个 懂了吗)",
    )

    print("您有什么问题吗？")

    while True:
        pb = input()
        print()
        if pb == "bye":
            break

        for event in rea.stream_run(pb):
            if event.type == "text":
                print(event.text, end="", flush=True)
            elif event.type == "thinking":
                print(f"\n[思考] {event.text}", end="\n")
            elif event.type == "tool_call":
                print(f"[调用工具] {event.call.name}({event.call.arguments})")
            elif event.type == "tool_result":
                print(f"[工具结果] {event.result}")
            elif event.type == "final":
                print()

        print("还有什么可以帮到您")

    print("希望您生活愉快")


if __name__ == "__main__":
    main()
