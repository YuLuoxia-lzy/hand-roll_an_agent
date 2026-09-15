"""扩展点 —— 框架的「入站拦截」

AgentEvent 是**出站**事件流(agent 往外吐), hook 是**入站**拦截(外面往里插)。
两者合起来, 框架的骨架才完整: 一个负责"告诉你发生了什么", 一个负责"在它发生前
改掉它"。

【为什么要有这个东西】
加一个记忆层, 之前动了 **9 个文件、178 行** —— 四个子类的构造函数、基类的
_build_messages、ReAct 的 _finish、PlanSolve 的两个方法签名、一个类改名,
外加 PlanSolve 和 Reflection 各自手抄了同样三行调 _memory_context()。

每一处判断都是对的。问题是框架没给别的选择: **记忆成了基类的参数**,
于是每加一种横切能力(记忆流打分、世界感知、trace、情绪、成本熔断),
这套动作就要重来一遍。第 N 种能力的成本是 O(所有子类)。

有了 hook, 加记忆就是 `hooks=[MemoryHook(mem)]` —— 一个新文件, 框架零改动。

【判据】
    加一个新能力 = 写一个新文件 + 挂上去, 不需要改框架里的任何文件。

【五个阶段, 按一次 run() 的时间顺序】
    before_input   改用户输入           —— 世界感知、输入清洗、脱敏
    before_llm     改将要发给模型的消息   —— 记忆注入、RAG、上下文压缩   ← 记忆层在这
    after_llm      观察/改模型响应       —— 用量统计、trace、内容审核
    after_tool     观察/改工具结果       —— 世界状态更新、结果脱敏
    after_run      一轮结束             —— 落库、打分、写记忆流         ← 情景记忆在这

【为什么用 RunContext 而不是把参数摊开】
见下面 RunContext 的文档 —— 它一次性解决了两个"今天靠约定才成立"的约束。
"""

import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RunContext:
    """
    一次 run() 的上下文。**在 hook 之间共享, 也是 hook 唯一的"我在哪"来源。**

    【它值钱的地方: 把两条"靠人记住"的约定变成了"结构上不可能做错"】

    ① 它消灭了「查询该用什么」这个约定
       memory-layer-guide 里有一整节在讲: 检索要用 input_text 参数本身,
       **绝对不要用 messages[-1]["content"]** —— 因为到了 ReAct 第 2 步,
       最后一条是 tool 消息, 拿工具结果去检索, 检索出来的东西和用户问题毫无关系。
       那条约束今天是靠人记住的。而 ctx.input_text **永远是本轮原始输入**,
       不管你挂在哪个阶段、第几次 LLM 调用。

    ② 它解决了「每个 hook 的签名都不一样」的问题
       after_tool 需要 call 和 result, after_run 需要 turn, after_llm 只要 response。
       摊平进签名的话, 每加一个阶段就要改所有 hook 的基类。
       ctx 是那个**会增长的参数** —— 加字段不破坏任何已有 hook。

    > 框架里凡是有"每次调用都需要、而且以后可能变多"的参数, 就该打包成一个
    > context 对象。摊平进签名的每一个参数, 都是一个未来会破的兼容性承诺。
    """

    #: 本轮用户**原始**输入。永远不被改写 —— 这是 hook 用它做检索 query 的前提。
    #: (before_input 改写的是"送进子类的那一份", 不是这个。见 Agent.run。)
    input_text: str
    #: 发起这次 run 的 agent
    agent: Any = None
    #: 本轮是第几轮(agent._turns 的长度, 记录前)
    turn_index: int = 0
    #: 开始时间
    started_at: float = field(default_factory=time.time)
    #: 本轮已经发起了几次 LLM 调用(before_llm 看到的是"这次是第几次", 从 0 开始)
    llm_calls: int = 0
    #: 自由槽位 —— hook 之间传递临时状态用, 框架不碰它
    state: dict = field(default_factory=dict)


class Hook:
    """
    扩展点基类。

    **只覆盖你关心的那几个方法。** 基类里五个都是直接返回入参的 no-op,
    所以一个只想记 trace 的 hook 只需要写 after_llm 一个方法。

    【on_error: 每个 hook 自己声明失败策略】
        "open"   —— 吞掉, 记一条 warning, 继续跑(记忆、trace、埋点用这个)
        "closed" —— 抛出去, 中断本轮(成本熔断、权限校验、内容合规用这个)

    为什么让 hook 自己声明, 而不是框架统一决定:
        框架**没有资格**替应用决定"这个 hook 挂了要不要中断对话"。
        记忆挂了应该继续聊, 但"这个用户欠费了"挂了必须中断。
        两者形状一样, 语义相反。

    【不可变约定: 不得原地修改入参】
    改 messages / result 时必须**返回新对象**(新列表 + 新字典), 不能就地把
    messages[-1]["content"] 改掉。理由见 core/agent.py 里 _chat 那一段:
    ReAct 的 messages 是循环里不断 append 的**活列表**, 原地改会污染
    last_messages 和存进 Turn 的轨迹 —— 而且一个错都不报。
    """

    #: 见上面的说明。子类改这一个类属性就能切换失败策略。
    on_error: str = "open"

    def before_input(self, ctx: RunContext, input_text: str) -> str:
        """改用户输入。返回值才是子类真正收到的那份。**ctx.input_text 不受影响。**"""
        return input_text

    def before_llm(self, ctx: RunContext, messages: List[dict]) -> List[dict]:
        """改将要发给模型的消息。**必须返回新列表**(见类文档的不可变约定)。"""
        return messages

    def after_llm(self, ctx: RunContext, response):
        """
        观察/改模型响应。

        ⚠️ 流式路径上**只能观察, 不能改** —— chunk 早就逐块吐给调用方了,
        改 self._last_response 对已经吐出去的字符没有任何影响。
        所以用它做 usage 统计 / trace 没问题, 做内容审核只有非流式路径有效。
        这个不对称无法消除(除非放弃流式), 只能写清楚。
        """
        return response

    def after_tool(self, ctx: RunContext, call, result: str) -> str:
        """观察/改工具结果。返回值才是回传给模型的那份。"""
        return result

    def after_run(self, ctx: RunContext, turn) -> None:
        """一轮结束了。落库、打分、写记忆流都在这。返回值被忽略。"""
        return None


class HookPipeline:
    """按注册顺序跑一组 hook。"""

    def __init__(self, hooks: Optional[List[Hook]] = None):
        self.hooks: List[Hook] = list(hooks or [])

    def add(self, hook: Hook) -> "HookPipeline":
        """链式注册: pipeline.add(A()).add(B())"""
        self.hooks.append(hook)
        return self

    # ==================== 五个阶段 ====================
    #
    # 每个阶段都是"把上一个 hook 的返回值喂给下一个" —— 所以顺序就是注册顺序,
    # 而且一个 hook 能看见前面所有 hook 的结果。

    def before_input(self, ctx: RunContext, input_text: str) -> str:
        for hook in self.hooks:
            input_text = self._guard(hook, "before_input", ctx, input_text)
        return input_text

    def before_llm(self, ctx: RunContext, messages: List[dict]) -> List[dict]:
        for hook in self.hooks:
            messages = self._guard(hook, "before_llm", ctx, messages)
        return messages

    def after_llm(self, ctx: RunContext, response):
        for hook in self.hooks:
            response = self._guard(hook, "after_llm", ctx, response)
        return response

    def after_tool(self, ctx: RunContext, call, result: str) -> str:
        for hook in self.hooks:
            result = self._guard(hook, "after_tool", ctx, call, result)
        return result

    def after_run(self, ctx: RunContext, turn) -> None:
        for hook in self.hooks:
            self._guard(hook, "after_run", ctx, turn)

    # ==================== 内部 ====================

    @staticmethod
    def _guard(hook: Hook, stage: str, *args):
        """
        跑一个 hook, 按它自己声明的策略处理失败。

        失败时**返回入参的原值** —— 也就是"这个 hook 什么都没做",
        这是 fail-open 唯一正确的退化形态: 调用方拿到的消息和没挂它时**逐字节相同**。
        (注意不是"返回 None"或"返回空"—— 那会把请求整个弄坏。)
        """
        fn = getattr(hook, stage, None)
        if fn is None:                      # 基类已经保证了五个方法都在, 这里防的是
            return args[-1]                 # 传进来的"鸭子类型" hook 少写了某个阶段

        try:
            return fn(*args)
        except Exception as e:
            if getattr(hook, "on_error", "open") == "closed":
                raise                       # 声明了"必须中断"的, 原样穿透
            # warning 而不是 exception: 这在生产里是会反复发生的事(记忆层正对着
            # LLM 漏斗, embedding 独立 provider/独立 key), 每次都打整个堆栈会把日志刷爆。
            logger.warning(
                "%s.%s 失败, 已跳过(本轮行为与没挂它时一致): %s",
                type(hook).__name__, stage, e,
            )
            return args[-1]

    def __len__(self) -> int:
        return len(self.hooks)

    def __repr__(self) -> str:
        return f"HookPipeline({[type(h).__name__ for h in self.hooks]})"
