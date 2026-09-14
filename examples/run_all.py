"""一个菜单, 把四个示例串起来。

跑法(仓库根目录下, 注意是 -m 不是直接执行文件 —— 相对导入需要包的上下文):
    python -m examples.run_all
"""

from .plan_and_solve import main as run_plan_and_solve
from .react_with_tools import main as run_react
from .reflection import main as run_reflection
from .simple_chat import main as run_simple

MENU = {
    "1": ("SimpleAgent     一问一答, 记住上下文", run_simple),
    "2": ("ReActAgent      循环调用工具, 直到不需要为止", run_react),
    "3": ("ReflectionAgent 写初稿 -> 自己评审 -> 按意见改", run_reflection),
    "4": ("PlanAndSolveAgent 先拆步骤, 再一步步做", run_plan_and_solve),
}


def main():
    for key, (desc, _) in MENU.items():
        print(f"  {key}. {desc}")

    choice = input("\n选一个 (1-4): ").strip()
    entry = MENU.get(choice)
    if entry is None:
        print("没有这个选项")
        return
    entry[1]()


if __name__ == "__main__":
    main()
