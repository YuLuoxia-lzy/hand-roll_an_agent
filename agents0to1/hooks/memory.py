"""记忆 hook —— 把记忆层挂进 Agent 的唯一入口

【这个文件的存在本身就是验收证据】
改造之前, 加记忆要动 **9 个文件、178 行**: 四个子类的构造函数各改一遍、
基类加 _prepare_user_message / _record_first_message / _memory_context、
ReAct 单独改 _finish 去还原被注入的内容、PlanSolve 改 Planner.plan 和
Executor.execute 的签名、Reflection 得把一个叫 Memory 的类改名,
最后 PlanSolve 和 Reflection 还得**各手抄同样三行**调 _memory_context()。

改造之后, 就是这一个文件:

    agent = SimpleAgent("a", llm, hooks=[MemoryHook(mem)])

想摘掉记忆? 把 hooks=[...] 里的那一项删掉, agent 完全不知道记忆层存在过。
"""

from typing import Any, Callable, List, Optional

from ..core.hooks import Hook, RunContext
from ..utils.logging import get_logger

logger = get_logger(__name__)


class MemoryHook(Hook):
    """
    语义记忆 / 情景记忆的注入。

    【fail-open: 记忆只是增强, 它没有权力把整个 agent 拖下水】
    memory-layer-guide 5.3 —— 记忆层正对着 LLM 漏斗, 抛出去就是整个 agent 挂掉。
    所以 on_error="open", 而且检索本身还包了一层 try(见 _safe_context),
    两层都退化成"这一轮没有记忆", 而不是"这一轮没有回答"。

    用法:
        mem = SemanticMemory()
        agent = SimpleAgent("a", llm, hooks=[MemoryHook(mem)])

        # ReAct 两个都挂是有意的(见 tools/builtin/knowledge.py):
        #   上下文版: 每次都自动带上    工具版: 模型主动去查
        agent = ReActAgent("a", llm, tool_registry=r,
                           hooks=[MemoryHook(mem)])
    """

    #: 声明失败策略 —— 框架照此办理。记忆挂了继续聊, 见类文档。
    on_error = "open"

    def __init__(self, memory: Optional[Any] = None, once_per_turn: bool = True):
        """
        Args:
            memory:       任何提供 build_context() **或** search() 的对象
                          (SemanticMemory / EpisodicMemory / 你自己写的)
            once_per_turn: 见下面"注入频率"那段。默认 True。

        【类型守卫】搬自原来的 Agent.__init__ —— 在构造函数就报错, 比等到
        第一次调 LLM 时才炸要好得多(那时候报错位置离病因十万八千里)。
        """
        if memory is not None and not (
            hasattr(memory, "search") or hasattr(memory, "build_context")
        ):
            raise TypeError(
                f"memory 需要提供 search() 或 build_context() 方法, "
                f"收到的是 {type(memory).__name__}"
            )
        self.memory = memory
        self.once_per_turn = once_per_turn

        # 取上下文的策略在**构造时**定下来, 而不是每次调用都 hasattr 一遍:
        # 这样"只有 search() 的记忆对象"是真的能用, 而不是等运行期才发现不行。
        self._retrieve: Callable[[str], str] = self._pick_strategy(memory)

    # ==================== 注入频率 ====================
    #
    # 【once_per_turn=True(默认): 一轮只认一个注入目标】
    #     适合"被注入的那条 user 消息**一直在列表里**"的 Agent —— SimpleAgent / ReAct。
    #     ReAct 第 2 步的请求末尾是 tool 消息, 但那条被注入的 user 还在列表中
    #     (倒数第 3 条), 于是"同一份内容放回同一个位置": 每一步的请求里都带着
    #     记忆, 而且只带一份(还是那一条消息, 没有复制)。
    #     到了该给最终答案的那一步, 检索结果还在 —— 这正是原行为。
    #     而 Reflection 第 2、3 次调用的是**新拼的评审/改写 prompt**(内容与目标
    #     不同), 于是自动被跳过: 它只该喂初稿, 见 reflection_agent 的说明。
    #
    # 【once_per_turn=False: 每次 LLM 调用都注入(检索仍然只做一次)】
    #     适合"每次调用都是**全新 prompt**"的 Agent —— PlanAndSolve。
    #     它的规划器和执行器各自拼一条独立的 user 消息直接 _chat, 规划那一次注入
    #     的东西根本流不到执行第几步里去; 对它来说"每一步都有"才等于"记忆生效"。
    #
    #     为什么判据不能是 ctx.llm_calls > 0: PlanSolve 一轮里有 N 次"第一次"
    #     (规划一次 + 每一步一次), 那个计数器对它不成立。所以这里记的是
    #     "这一轮注入到哪条消息上了"(ctx.state["memory_target"]), 它对所有 Agent 都成立。
    #
    #     检索只做一次是刻意的: 每一步都重新检索的话, 同一份资料会在不同的步骤里
    #     被检索出不同的片段, 结果没法复现。

    @staticmethod
    def _pick_strategy(memory: Optional[Any]) -> Callable[[str], str]:
        """
        决定"怎么从记忆里取一段能塞进 prompt 的文本"。

        两种记忆形状:
            build_context()  —— 自己负责检索 + 排版(SemanticMemory / EpisodicMemory)
            search()         —— 只给结果, 排版由这里做
        """
        if memory is None:
            return lambda query: ""

        build = getattr(memory, "build_context", None)
        if callable(build):
            return lambda query: build(query) or ""

        def retrieve_from_search(query: str) -> str:
            items = memory.search(query)
            if not items:
                return ""
            # 由 hook 排版, 但**复用记忆对象自己的 format_items** ——
            # 这样它和 KnowledgeSearchTool 走的是同一个排版, 模型看到的形状一致
            format_items = getattr(memory, "format_items", None)
            if callable(format_items):
                return format_items(items)
            return "\n\n".join(f"[{i}] {item.text}" for i, item in enumerate(items, 1))

        return retrieve_from_search

    # ==================== 钩子本体 ====================

    def before_llm(self, ctx: RunContext, messages: List[dict]) -> List[dict]:
        """
        把检索结果并进**本轮那条 user 消息**。

        【注入形态: 并进已有的 user, 不新增消息】
            单独一条 system 插中间 -> Llama-2 模板不吃
            单独一条 user          -> 连续两条 user, 模板和严格校验器都拒绝
            并进已有的 user        -> 不新增消息、不动角色序列, 谁都吃
        而且塞进 system 会让 `system + history` 这段稳定前缀每轮全变,
        DeepSeek 的自动前缀缓存全部失效(缓存命中和不命中差很多)。

        【为什么目标是"最后一条 user 消息", 而不是"最后一条消息"】
        ReAct 第 2 步的 messages[-1] 是 **tool 消息**, 而本轮那条 user 还在列表里
        (倒数第 3 条)。要是只认 messages[-1], 第 2 步就注入不进去 —— 可第 2 步
        往往才是模型**给最终答案**的那一步:"检索出来的资料, 模型看完就扔了"。
        所以倒着找最后一条 user 消息, 把它换掉。

        【为什么靠"内容"找, 不靠下标】
        下标会因为"这次有没有补 system、历史拼了几条"而错位, 而内容稳。
        靠内容还顺手解决了 Reflection: 它第 2、3 次调用的是**新拼的评审/改写
        prompt**, 内容和注入目标不一样 —— 于是自动跳过, 不需要额外判断。

        【不可变 —— 改造前那个补丁就是为这件事存在的】
        out 是新列表, 里面是新字典。原地改 messages[i]["content"] 会污染
        agent 的工作列表, 于是:
          1. ReAct 的 self.last_messages(=那个活列表)被污染
          2. messages[turn_start:] 存进 Turn 时带着检索结果
          3. 下一轮 _history_messages() 把它当"用户说过的话"重发
          4. _turn_chars() 的预算被白吃
        改造前靠 react_agent._finish 里手工还原第一条(_record_first_message)来
        兜住 2/3/4。现在**返回新列表**让那个补丁整个不需要了 —— 见下面这句
        `out = list(messages)`: 注入的那份从头到尾没进过工作列表。
        """
        if self.memory is None or not messages:
            return messages

        # 1. 目标: 倒着找最后一条 user 消息
        index = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                index = i
                break
        if index < 0:
            # 一条 user 都没有。正常路径不会走到这(每个 Agent 的第一条要注入的
            # 消息都是 user), 真走到了就说一声 —— 静默失效比报错难查得多。
            if not ctx.state.get("memory_skip_warned"):
                ctx.state["memory_skip_warned"] = True
                logger.warning(
                    "记忆注入被跳过: 这次要发给模型的消息里没有 user 消息(%s)—— "
                    "拿别的角色去承载检索结果语义上说不通。",
                    [m.get("role") for m in messages],
                )
            return messages

        target = messages[index].get("content")

        # 2. once_per_turn: 本轮只认**第一个**注入目标
        #
        # 判据不是 ctx.llm_calls(PlanSolve 一轮里有 N 次"第一次"), 而是"这一轮
        # 我注入到哪条消息上了"。同一个目标再来一次 -> 放同一份内容(ReAct 第 2 步);
        # 换了个目标 -> 说明这是另一个 prompt, 不注入(Reflection 的评审/改写)。
        if self.once_per_turn:
            first = ctx.state.get("memory_target")
            if first is None:
                ctx.state["memory_target"] = target
            elif target != first:
                return messages

        # 3. 检索只做一次, 结果缓存在 ctx.state 里 —— 见上面 once_per_turn 的说明
        if "memory_context" not in ctx.state:
            ctx.state["memory_context"] = self._safe_context(ctx.input_text)
        context = ctx.state["memory_context"]

        if not context:
            # 检索成功但没结果 —— 不留任何痕迹, 和没挂记忆时逐字节相同
            return messages

        out = list(messages)
        out[index] = {**messages[index], "content": f"{context}\n\n{target}"}
        return out

    def _safe_context(self, query: str) -> str:
        """
        取上下文, **失败返回空串**。

        【为什么用 ctx.input_text 而不是 messages[-1]["content"]】
        memory-layer-guide 5.2 那条"要靠人记住"的约束, 现在由 RunContext
        **在结构上**保证了: ctx.input_text 永远是本轮原始输入, 不管你挂在哪个阶段、
        第几次 LLM 调用。ReAct 第 2 步 messages[-1] 是 tool 消息, 用它会检索出
        和用户问题毫无关系的东西 —— 而且不报错, 你只会觉得"记忆好像没什么用"。
        """
        try:
            return self._retrieve(query) or ""
        except Exception as e:
            # 用 warning 而不是 exception: 这在生产里会反复发生
            # (embedding 独立 provider、独立 key、DeepSeek 还没有 embedding 端点),
            # 每次都打一整个堆栈会把日志刷爆。
            logger.warning("记忆检索失败, 本轮按『没有记忆』继续: %s", e)
            return ""
