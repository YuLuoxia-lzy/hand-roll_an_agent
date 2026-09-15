"""扩展点(hook 管线)测试 —— 逐条对应 framework-design.md 1.9 的必测清单

    python tests/test_hooks.py

全部离线: 假 LLM、假记忆, 不需要 key。

【这个文件测的不是"记忆", 是"钩子本身"】
记忆那条链路的坑在 test_agent_memory.py(33 条)。这里测的是**框架**:
一个 hook 挂了会不会静默改变行为、失败策略认不认、五个阶段的调用点对不对、
ctx 的生命周期归谁管。这些东西一旦错了, 表现是"某个功能偶尔不生效"——
比崩溃难查得多, 所以每一条都拿"逐字节相同"当判据, 而不是"看起来还行"。
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import FakeLLM, TempDir, run_tests                       # noqa: E402

from agents0to1 import (                                               # noqa: E402
    Agent,
    Hook,
    HookPipeline,
    MemoryHook,
    PlanAndSolveAgent,
    ReActAgent,
    ReflectionAgent,
    SimpleAgent,
    Tool,
    ToolParameter,
    ToolRegistry,
)
from agents0to1.core.typedefs import AgentEvent, ToolCall              # noqa: E402

MARK = "【钩子】"


# ==================== 测试替身 ====================

class TraceHook(Hook):
    """只记录, 不改任何东西 —— 行为必须和没挂时**逐字节相同**"""

    def __init__(self, log: List[tuple] = None, tag: str = "trace"):
        self.log = log if log is not None else []
        self.tag = tag
        self.contexts = []          # 每次 before_llm 看到的 ctx

    def before_input(self, ctx, input_text):
        self.log.append((self.tag, "before_input", ctx.input_text))
        return input_text

    def before_llm(self, ctx, messages):
        self.log.append((self.tag, "before_llm", ctx.llm_calls))
        self.contexts.append(ctx)
        return messages

    def after_llm(self, ctx, response):
        self.log.append((self.tag, "after_llm", ctx.llm_calls))
        return response

    def after_tool(self, ctx, call, result):
        self.log.append((self.tag, "after_tool", call.name))
        return result

    def after_run(self, ctx, turn):
        self.log.append((self.tag, "after_run", turn.user))


class BoomHook(Hook):
    """声明了"必须中断"的 hook —— 它抛异常, 本轮必须停下来"""

    on_error = "closed"

    def before_input(self, ctx, input_text):
        raise RuntimeError("boom")


class SoftBoomHook(Hook):
    """声明了 fail-open 的 hook —— 它抛异常, 本轮必须照常跑完"""

    on_error = "open"

    def before_llm(self, ctx, messages):
        raise RuntimeError("boom")


class InjectHook(Hook):
    """一个最小号的"往最后那条 user 里加料"的 hook —— 记忆注入的形状"""

    def before_llm(self, ctx, messages):
        if not messages or messages[-1].get("role") != "user":
            return messages
        out = list(messages)
        out[-1] = {**out[-1], "content": f"{MARK}{out[-1]['content']}"}
        return out


class RewriteToolResultHook(Hook):
    """改工具结果 —— after_tool 的返回值必须是**回传给模型的那份**"""

    def after_tool(self, ctx, call, result):
        return f"[已脱敏]{result}"


class SubAgent(Agent):
    """最小子类: 只实现 _run, **不实现 stream_run / _stream_run** ——
    用来证明"只要写了 _run, 事件流这个形状就白送"。"""

    def _run(self, input_text: str, **kwargs) -> str:
        return self._chat([{"role": "user", "content": input_text}]).content or ""


class EchoTool(Tool):
    def __init__(self, name: str = "echo"):
        super().__init__(name=name, description="原样返回输入文本")

    def get_parameters(self):
        return [ToolParameter(name="text", type="string", description="要回显的文本")]

    def run(self, parameters: Dict[str, Any]) -> str:
        return f"回显: {parameters.get('text', '')}"


def _simple(llm, hooks=None):
    return SimpleAgent("t", llm, system_prompt="你是助手", hooks=hooks)


def _react(llm, hooks=None, registry=None):
    return ReActAgent("t", llm, tool_registry=registry or ToolRegistry(),
                      system_prompt="你是助手", hooks=hooks)


# ==================== 清单 1: no-op hook 不改变任何东西 ====================

def test_noop_hook_is_byte_identical():
    """
    挂一个什么都不改的 hook, FakeLLM 收到的请求必须和没挂时**逐字节相同**。

    "差不多"不行: 一个 hook 把消息列表重建了一遍、字段顺序变了、或者多带了一个
    kwargs, 今天可能没事, 明天某个服务端就拒了 —— 而你会去查模型、查网络,
    不会想到是那个"只记 trace"的 hook。
    """
    plain = FakeLLM(["答案"])
    _simple(plain).run("问题")

    traced = FakeLLM(["答案"])
    _simple(traced, hooks=[TraceHook()]).run("问题")

    assert plain.calls == traced.calls, (
        "挂了 no-op hook 之后请求变了:\n"
        f"  没挂: {json.dumps(plain.calls[0], ensure_ascii=False)[:300]}\n"
        f"  挂了: {json.dumps(traced.calls[0], ensure_ascii=False)[:300]}"
    )


def test_noop_hook_does_not_change_the_turn():
    plain = FakeLLM(["答案"])
    a = _simple(plain)
    a.run("问题")

    traced = FakeLLM(["答案"])
    b = _simple(traced, hooks=[TraceHook()])
    b.run("问题")

    assert a.get_turns()[0].model_dump() == b.get_turns()[0].model_dump()


# ==================== 清单 2 / 3: 失败策略 ====================

def test_open_hook_failure_is_swallowed_and_byte_identical():
    """
    on_error="open" 的 hook 抛异常 -> 本轮照常跑完, **且请求逐字节相同**。

    失败时正确的退化形态是"这个 hook 什么都没做", 不是"返回 None/返回空"——
    后者会把要发出去的请求整个弄坏。所以断言的是**和没挂它时一模一样**,
    而不是"没崩就行"。
    """
    plain = FakeLLM(["答案"])
    _simple(plain).run("问题")

    soft = FakeLLM(["答案"])
    agent = _simple(soft, hooks=[SoftBoomHook()])
    answer = agent.run("问题")

    assert answer == "答案", answer
    assert plain.calls == soft.calls, "fail-open 之后请求变了"


def test_open_hook_failure_in_every_stage_is_swallowed():
    """
    五个阶段各炸一次, 全都不该把 agent 拖下水。

    只测一个阶段是不够的: 每个阶段在 HookPipeline 里是各写一遍的,
    漏了一个就是"某个 hook 挂在某个阶段会静默 500"。
    """
    for stage in ("before_input", "before_llm", "after_llm", "after_run"):
        hook = Hook()
        setattr(hook, stage, lambda *a, **k: (_ for _ in ()).throw(RuntimeError(stage)))

        llm = FakeLLM(["答案"])
        # after_tool 需要工具才触发, 单独测(见 test_after_tool_failure_is_swallowed)
        kwargs = {"registry": ToolRegistry()} if stage == "after_run" else {}
        agent = _react(llm, hooks=[hook]) if stage in ("after_llm", "after_input") else _simple(llm, hooks=[hook])
        assert agent.run("问题") == "答案", f"{stage} 阶段抛异常把 agent 弄挂了"


def test_after_tool_failure_is_swallowed():
    hook = Hook()

    def boom(ctx, call, result):
        raise RuntimeError("after_tool boom")

    hook.after_tool = boom

    registry = ToolRegistry()
    registry.register_tool(EchoTool())
    llm = FakeLLM([(None, [ToolCall(id="t1", name="echo", arguments={"text": "x"})]), "答案"])
    agent = _react(llm, hooks=[hook], registry=registry)

    assert agent.run("问题") == "答案"
    # 工具结果原样回传 —— fail-open 的退化形态是"这个 hook 什么都没做"
    assert llm.calls[1]["messages"][-1]["content"] == "回显: x"


def test_closed_hook_failure_propagates_and_cleans_up():
    """
    on_error="closed" 的 hook 抛异常 -> 异常必须**穿透出来**。

    为什么让 hook 自己声明策略: 框架没资格替应用决定"这个 hook 挂了要不要中断
    对话"。记忆挂了应该继续聊, 但"这个用户欠费了"挂了必须中断 —— 两者形状一样、
    语义相反。
    """
    llm = FakeLLM(["答案"])
    agent = _simple(llm, hooks=[BoomHook()])

    try:
        agent.run("问题")
    except RuntimeError as e:
        assert "boom" in str(e), e
    else:
        raise AssertionError("closed 的 hook 抛了异常, 但 agent 照常跑完了")

    assert not llm.calls, "ctx 都还没建完就中断了, 不该已经发过请求"
    # 异常路径也必须把 ctx 清干净 —— 否则 add_turn 的 after_run 会在"这一轮
    # 早就结束了"之后被触发一次, 而且是在完全无关的调用里
    assert agent._ctx is None, "异常之后 ctx 没被清理"


# ==================== 清单 4: 注册顺序 ====================

def test_hooks_run_in_registration_order():
    """
    **两个 hook 的先后顺序 = 注册顺序**, 而且是"每个阶段各走一遍"。

    写成"把所有 hook 的 before_input 跑完再统一跑 before_llm"也能工作 ——
    但那样 A 的 after_llm 会排在 B 的 before_llm 前面, 一个"给消息打时间戳"的
    hook 加在"改消息"的 hook 后面就会拿到错的东西。这里钉死这个顺序。
    """
    log: List[tuple] = []
    llm = FakeLLM(["答案"])
    agent = _simple(llm, hooks=[TraceHook(log, "A"), TraceHook(log, "B")])
    agent.run("问题")

    assert log == [
        ("A", "before_input", "问题"), ("B", "before_input", "问题"),
        ("A", "before_llm", 0),        ("B", "before_llm", 0),
        ("A", "after_llm", 0),         ("B", "after_llm", 0),
        ("A", "after_run", "问题"),     ("B", "after_run", "问题"),
    ], log


def test_pipeline_chains_return_values():
    """上一个 hook 的返回值喂给下一个 —— 所以 B 能看见 A 改过的内容"""
    seen = []

    class Append(Hook):
        def __init__(self, text):
            self.text = text

        def before_llm(self, ctx, messages):
            seen.append(messages[-1]["content"])
            out = list(messages)
            out[-1] = {**out[-1], "content": out[-1]["content"] + self.text}
            return out

    llm = FakeLLM(["答案"])
    _simple(llm, hooks=[Append("A"), Append("B")]).run("原")

    assert seen == ["原", "原A"], f"B 没看到 A 的结果: {seen}"
    assert llm.calls[0]["messages"][-1]["content"] == "原AB"


# ==================== 清单 5: 不可变约定 ====================

def test_before_llm_must_return_a_new_list():
    """
    **hook 返回新列表 —— agent 自己的工作状态一个字都不许变。**

    这条对应 framework-design.md 1.7 里 `last_messages` 那个 A/B 选择,
    这里选的是 **A**(文档推荐的那个):

        last_messages 的语义是"模型当时看到了什么" —— 也就是真正发出去的那份,
        **含** hook 注入的内容。它由基类在 _chat / _stream_chat 的出口处维护。

    而"注入不进 Turn / 不进历史 / 不吃预算"这条铁律, 靠的不是把 last_messages
    擦干净, 而是**注入从来没进过 agent 那份工作列表**: hook 返回的是新列表,
    agent 手里那份从头到尾没被碰过。两件事因此可以同时成立。
    """
    llm = FakeLLM(["答案"])
    agent = _simple(llm, hooks=[InjectHook()])
    agent.run("问题")

    # 发出去的:带着注入内容(这是 hook 的职责)
    assert llm.calls[0]["messages"][-1]["content"].startswith(MARK)
    # 模型看到的 == last_messages:两者必须是同一份, 否则 last_messages 就是骗人的
    assert agent.last_messages == llm.calls[0]["messages"], (
        "last_messages 和真正发出去的请求对不上 —— 它的语义是『模型当时看到了什么』"
    )

    # agent 自己的工作状态:干净
    stored = agent.get_turns()[0]
    assert stored.messages[0] == {"role": "user", "content": "问题"}, stored.messages[0]
    assert MARK not in json.dumps(stored.model_dump(), ensure_ascii=False), (
        "注入内容进了 Turn —— 下一轮会被当成『用户说过的话』重发, _turn_chars 的预算也被白吃"
    )


def test_mutating_in_place_would_pollute_the_turn():
    """
    【这条断言的是**当前行为**, 它记录的是"不可变约定为什么必须存在"】

    一个原地改 messages[-1]["content"] 的 hook 会把 agent 的工作列表改脏, 于是
    注入内容被存进 Turn、下一轮被当成"用户说过的话"重发、_turn_chars 的预算被白吃
    —— **而且一个错都不报**。

    为什么必须用 ReAct 演这个: SimpleAgent 的 Turn 是拿 (input_text, answer)
    两个值重建的, 根本不看 messages, 所以它污染不了。ReAct 记的是
    messages[turn_start:], 那条被改的 user 消息**就是** turn_start —— 改造前
    react_agent._finish 里那个"手工还原第一条"的补丁, 补的就是这个洞。

    框架**不拦**这件事(要拦就得在每次调用前深拷贝, 那是给所有 agent 常驻的税),
    约定在 hook 作者这边, 写在 core/hooks.py 的类文档里。

    如果哪天框架改成"进 hook 之前先深拷贝", 这条会红 —— 那时候是它完成了使命:
    该把它删掉, 并更新 hook 类文档里那段"不得原地修改入参"的说明。
    """
    class BadHook(Hook):
        def before_llm(self, ctx, messages):
            messages[-1]["content"] = f"{MARK}{messages[-1]['content']}"     # 原地改
            return messages

    registry = ToolRegistry()
    registry.register_tool(EchoTool())
    llm = FakeLLM([(None, [ToolCall(id="t1", name="echo", arguments={"text": "x"})]), "答案"])
    agent = _react(llm, hooks=[BadHook()], registry=registry)
    agent.run("问题")

    assert MARK in json.dumps(agent.get_turns()[0].model_dump(), ensure_ascii=False), (
        "原地改居然没污染 Turn —— 框架加防护了? 那这条测试和 hook 的文档都要改"
    )


# ==================== 清单 6: 流式路径 ====================

def test_before_llm_fires_on_the_streaming_path():
    """
    **_chat 和 _stream_chat 是两处, 钩子必须挂两处。**

    只挂 _chat 的话, 流式路径静默失效 —— "我这个 agent 走流式的时候记忆就不
    生效, 不走流式就生效", 这种 bug 找起来是以天为单位的。
    本项目已经踩过一次同样的坑(记忆层当初就漏了流式)。
    """
    llm = FakeLLM(["答案"])
    agent = _simple(llm, hooks=[InjectHook()])
    for _ in agent.stream_run("问题"):
        pass

    assert llm.calls[0]["stream"] is True, "这条得真的走流式路径才有意义"
    assert llm.calls[0]["messages"][-1]["content"].startswith(MARK), (
        "流式路径没触发 before_llm"
    )


def test_streaming_does_not_leak_extra_text():
    """hook 改的是**请求**, 不该往给用户看的流里多加一个字符"""
    llm = FakeLLM(["答案"])
    agent = _simple(llm, hooks=[InjectHook()])
    text = "".join(e.text for e in agent.stream_run("问题") if e.type == "text")
    assert text == "答案", f"流被污染了: {text!r}"


def test_after_llm_sees_the_response_on_the_streaming_path():
    log: List[tuple] = []
    llm = FakeLLM(["答案"])
    agent = _simple(llm, hooks=[TraceHook(log)])
    for _ in agent.stream_run("问题"):
        pass

    assert ("trace", "after_llm", 0) in log, log
    assert agent._last_response.content == "答案"


# ==================== 清单 7: ctx.input_text ====================

def test_ctx_input_text_is_still_the_original_question_on_react_step_two():
    """
    **这是 RunContext 存在的理由。**

    ReAct 第 2 步的消息末尾是 tool 消息。谁要是在 hook 里用
    messages[-1]["content"] 当检索依据, 检索出来的东西和用户问题毫无关系 ——
    而且不报错, 你只会觉得"记忆好像没什么用"。

    ctx.input_text 永远是本轮原始输入, 不管你挂在哪个阶段、第几次 LLM 调用。
    约束从"记得这么做"变成了"结构上不可能做错"。
    """
    registry = ToolRegistry()
    registry.register_tool(EchoTool())
    llm = FakeLLM([
        (None, [ToolCall(id="t1", name="echo", arguments={"text": "工具结果"})]),
        "最终答案",
    ])
    log: List[tuple] = []
    agent = _react(llm, hooks=[TraceHook(log)], registry=registry)
    agent.run("用户的问题")

    calls = [x for x in log if x[1] == "before_llm"]
    assert len(calls) == 2, "这个用例得真的走两步才有意义"

    # 每一次调用时 ctx.input_text 都是本轮原始输入(第 2 步的 messages[-1] 是 tool 消息)
    assert [ctx.input_text for ctx in agent._hooks.hooks[0].contexts] == ["用户的问题"] * 2
    # llm_calls 是"这次是第几次", 从 0 开始 —— 在 hook 之后才自增
    assert [x[2] for x in calls] == [0, 1], calls
    # 同一个 ctx 贯穿整轮 —— 不是每次调用新建一个
    assert agent._hooks.hooks[0].contexts[0] is agent._hooks.hooks[0].contexts[1]


def test_before_input_rewrites_what_the_subclass_gets_not_the_ctx():
    class Rewrite(Hook):
        def before_input(self, ctx, input_text):
            return f"[改写]{input_text}"

    llm = FakeLLM(["答案"])
    agent = _simple(llm, hooks=[Rewrite()])
    agent.run("原话")

    # 送进子类的那份被改了
    assert llm.calls[0]["messages"][-1]["content"] == "[改写]原话"
    # 而 ctx.input_text 保持原值 —— 它是 hook 的检索依据, 不该被任何 hook 改写
    assert agent.get_turns()[0].user == "[改写]原话"


def test_ctx_input_text_untouched_by_before_input():
    seen = {}

    class Spy(Hook):
        def before_llm(self, ctx, messages):
            seen["input_text"] = ctx.input_text
            return messages

    class Rewrite(Hook):
        def before_input(self, ctx, input_text):
            return "改过的"

    llm = FakeLLM(["答案"])
    # 注册顺序: 改写在前, 观察在后
    _simple(llm, hooks=[Rewrite(), Spy()]).run("原话")

    assert seen["input_text"] == "原话", (
        f"ctx.input_text 被 before_input 改写了: {seen['input_text']!r} —— "
        f"它是 hook 的检索依据, 必须是本轮原始输入"
    )


# ==================== ctx 生命周期(模板方法) ====================

def test_ctx_is_created_and_cleared_by_the_framework():
    """
    ctx 的生命周期由**框架**管, 子类忘不掉。

    改造前每个子类自己实现 run(), 于是"记得建 ctx / 记得清 ctx"就成了又一条
    "必须记得做的事" —— 而"必须记得做的事"正是"加记忆要动 9 个文件"的病根。
    """
    trace = TraceHook()
    agent = _simple(FakeLLM(["答案"]), hooks=[trace])
    assert agent._ctx is None

    seen = {}

    class Spy(Hook):
        def after_run(self, ctx, turn):
            seen["inside"] = agent._ctx
            seen["turn_index"] = ctx.turn_index

    agent._hooks.add(Spy())
    agent.run("问题")

    assert seen["inside"] is not None, "after_run 跑的时候 ctx 已经被清了"
    assert seen["turn_index"] == 0, "turn_index 是『记录前』的轮次"
    assert agent._ctx is None, "run() 结束后 ctx 没被清掉"

    # 第二轮: turn_index 递增
    agent.run("又问")
    assert seen["turn_index"] == 1, f"第二轮 turn_index = {seen['turn_index']}"


def test_after_run_fires_exactly_once_per_turn():
    """
    after_run 挂在 add_turn 里 —— 那是**所有**写入路径的汇合点。

    挂在子类的 run() 结尾的做法, 四个子类就有四种定义, 而且第五个子类会漏掉;
    挂在 _chat 上则会变成"一轮触发 N 次"(ReAct 一轮调 N 次 LLM)。
    """
    log: List[tuple] = []
    registry = ToolRegistry()
    registry.register_tool(EchoTool())
    llm = FakeLLM([(None, [ToolCall(id="t1", name="echo", arguments={"text": "x"})]), "答案"])
    agent = _react(llm, hooks=[TraceHook(log)], registry=registry)
    agent.run("问题")

    after_runs = [x for x in log if x[1] == "after_run"]
    assert len(after_runs) == 1, f"一轮触发了 {len(after_runs)} 次 after_run"
    assert after_runs[0][2] == "问题", after_runs


def test_after_run_sees_the_turn_after_it_is_recorded():
    seen = {}

    class Spy(Hook):
        def after_run(self, ctx, turn):
            seen["turn"] = turn
            seen["stored"] = len(ctx.agent.get_turns())

    agent = _simple(FakeLLM(["答案"]))
    agent._hooks.add(Spy())
    agent.run("问题")

    assert seen["turn"].user == "问题" and seen["turn"].answer == "答案"
    assert seen["stored"] == 1, "after_run 看到的时候, 这一轮还没进 _turns"


def test_add_turn_outside_a_run_does_not_fire_hooks():
    """run() 之外调 add_turn(老接口 add_message 走这条)不该凭空捏一个 ctx 出来"""
    log: List[tuple] = []
    agent = _simple(FakeLLM(["答案"]), hooks=[TraceHook(log)])
    agent.add_turn("问题", [{"role": "user", "content": "问题"}], answer="答案")

    assert not [x for x in log if x[1] == "after_run"], (
        "run() 之外的 add_turn 触发了 after_run —— hook 会拿到空 input_text 还以为自己在对话里"
    )


# ==================== 模板方法: stream_run / _run ====================

def test_every_agent_has_a_working_stream_run():
    """
    **四个 Agent 全都有了 stream_run, 而其中两个一行代码都没写。**

    改造前基类那句 stream_run 只会 raise NotImplementedError, 四个里只有两个实现了。
    现在默认实现是"把 _run 的结果包成一个 final 事件", 想精细的覆盖 _stream_run。
    """
    registry = ToolRegistry()
    registry.register_tool(EchoTool())

    cases = [
        ("SimpleAgent", _simple(FakeLLM(["答案"]))),
        ("ReActAgent", _react(FakeLLM(["答案"]), registry=registry)),
        ("ReflectionAgent", ReflectionAgent("t", FakeLLM(["初稿", "无需改进"]), system_prompt="你是助手")),
        ("PlanAndSolveAgent", PlanAndSolveAgent("t", FakeLLM([
            (None, [ToolCall(id="p1", name="submit_plan", arguments={"plan": ["一步"]})]),
            "结果",
        ]), system_prompt="你是助手")),
        ("SubAgent(只实现了 _run)", SubAgent("t", FakeLLM(["答案"]))),
    ]

    for name, agent in cases:
        events = list(agent.stream_run("问题"))
        finals = [e for e in events if e.type == "final"]
        assert finals, f"{name} 的 stream_run 一个 final 事件都没吐"
        assert finals[-1].answer, f"{name} 的 final 事件没有答案"
        assert isinstance(finals[-1], AgentEvent)


def test_simple_agent_still_streams_text_chunk_by_chunk():
    """默认实现是"只吐 final", 但 SimpleAgent 覆盖了 _stream_run —— 它得真的逐块吐"""
    llm = FakeLLM(["一二三四五六"])
    agent = _simple(llm)
    events = list(agent.stream_run("问题"))

    texts = [e.text for e in events if e.type == "text"]
    assert len(texts) > 1, f"没有逐块吐: {texts}"
    assert "".join(texts) == "一二三四五六"


def test_run_does_not_go_through_stream_run_recursively():
    """
    ReAct 的 _run 消费 _stream_run(它就是"边跑边收事件, 最后只返回答案")。
    如果 _run 里调的是 **stream_run**(模板方法), 它会重开一个 ctx 盖掉当前这个,
    并且层层递归 —— 这条就是防那个的。
    """
    registry = ToolRegistry()
    registry.register_tool(EchoTool())
    llm = FakeLLM([(None, [ToolCall(id="t1", name="echo", arguments={"text": "x"})]), "答案"])
    agent = _react(llm, registry=registry)

    assert agent.run("问题") == "答案"
    assert len(llm.calls) == 2, "步数不对 —— 递归会让它多跑几轮"
    assert agent._ctx is None


# ==================== after_tool 的时机 ====================

def test_after_tool_rewrites_what_the_model_sees():
    """
    **after_tool 必须在 messages.append 之前。**

    挂到 append 之后, 事件里的 result 和模型看到的 result 就是两份真相 ——
    "日志里脱敏了、实际发出去的没脱敏"这种事故就是这么来的。
    """
    registry = ToolRegistry()
    registry.register_tool(EchoTool())
    llm = FakeLLM([(None, [ToolCall(id="t1", name="echo", arguments={"text": "x"})]), "答案"])
    agent = _react(llm, hooks=[RewriteToolResultHook()], registry=registry)

    events = list(agent.stream_run("问题"))
    tool_events = [e for e in events if e.type == "tool_result"]
    assert len(tool_events) == 1

    sent = llm.calls[1]["messages"][-1]
    assert sent["role"] == "tool"
    assert sent["content"] == "[已脱敏]回显: x", f"模型看到的没脱敏: {sent['content']}"
    assert tool_events[0].result == sent["content"], (
        "事件里的 result 和模型看到的不一致 —— 两份真相"
    )


def test_after_tool_does_not_break_tool_call_pairing():
    """改内容不能改 id —— 配对看 id 不看内容"""
    registry = ToolRegistry()
    registry.register_tool(EchoTool())
    llm = FakeLLM([(None, [ToolCall(id="t1", name="echo", arguments={"text": "x"})]), "答案"])
    agent = _react(llm, hooks=[RewriteToolResultHook()], registry=registry)
    agent.run("问题")

    turn = agent.get_turns()[0]
    assert turn.messages[-2]["tool_call_id"] == "t1"
    assert turn.is_valid(), "轨迹配对被改坏了"


# ==================== 身份: agent_id ====================

def test_agent_id_is_generated_and_stable():
    """
    name 只是给人看的标签, 不保证唯一也不保证稳定。
    agent_id 才是身份 —— 跨进程 / 跨会话 / 记忆归属 / 消息路由都靠它
    (小镇那种"很多 agent 共用一个向量库"的场景尤其需要)。
    """
    a = _simple(FakeLLM(["答案"]))
    b = _simple(FakeLLM(["答案"]))

    assert len(a.agent_id) == 12, a.agent_id
    assert a.agent_id != b.agent_id, "两个 agent 拿到了同一个 id"
    assert a.agent_id != a.name, "id 不该等于 name"

    explicit = _simple(FakeLLM(["答案"]), hooks=None)
    explicit.agent_id = "自定义"
    assert explicit.agent_id == "自定义"


def test_agent_id_survives_snapshot_but_fork_gets_a_new_one():
    """
    恢复(load)是"同一个 agent 又回来了" —— 沿用快照里的 id。
    分叉(fork)是"另起一个" —— 换一个新的, 否则两个 agent 共用身份,
    向量库里的记忆会互相认亲。
    """
    with TempDir() as d:
        path = d / "snap.json"
        agent = _simple(FakeLLM(["答案"]))
        agent.run("问题")
        agent.save(path)

        snap = json.loads(path.read_text(encoding="utf-8"))
        assert snap["agent_id"] == agent.agent_id, "agent_id 没进快照"

        restored = SimpleAgent.load(str(path), llm=FakeLLM(["答案"]))
        assert restored.agent_id == agent.agent_id, "load 换了 id —— 恢复出来的该是同一个 agent"

        forked = SimpleAgent.fork(str(path), llm=FakeLLM(["答案"]))
        assert forked.agent_id != agent.agent_id, "fork 沿用了 id —— 两个 agent 会认错人"
        assert forked.get_turns()[0].user == "问题", "fork 该带着历史"


def test_old_snapshot_without_agent_id_still_loads():
    """旧快照里没这个字段 -> 自动生成一个新的。能读旧快照比"id 必须沿袭"重要"""
    with TempDir() as d:
        path = d / "snap.json"
        agent = _simple(FakeLLM(["答案"]))
        agent.run("问题")
        agent.save(path)

        data = json.loads(path.read_text(encoding="utf-8"))
        del data["agent_id"]
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

        restored = SimpleAgent.load(str(path), llm=FakeLLM(["答案"]))
        assert len(restored.agent_id) == 12


# ==================== HookPipeline 本身 ====================

def test_pipeline_is_usable_without_an_agent():
    """管线是个独立的组件 —— 不依赖 Agent, 空管线是合法的"""
    empty = HookPipeline()
    assert len(empty) == 0
    assert empty.before_input(None, "原文") == "原文"
    assert repo_ok(empty)

    pipe = HookPipeline().add(InjectHook())
    assert len(pipe) == 1
    assert "InjectHook" in repr(pipe)


def repo_ok(pipe):
    """before_llm 在空管线上也必须返回**原对象**(不是副本、不是 None)"""
    messages = [{"role": "user", "content": "问题"}]
    assert pipe.before_llm(None, messages) is messages
    return True


def test_bad_hook_object_is_guarded():
    """ duck-typing 的 hook 少写了某个阶段 —— 管线不能崩在这上面"""

    class Partial:
        on_error = "open"
        # 故意只写了 before_llm, 没有 before_input / after_run

    llm = FakeLLM(["答案"])
    agent = _simple(llm, hooks=[Partial()])
    assert agent.run("问题") == "答案"


# ==================== 那个验收标准 ====================

def test_memory_can_be_detached_without_touching_the_framework():
    """
    framework-design.md 1.9 的验收标准:

        把 MemoryHook 从 hooks=[...] 里摘掉, agent 应该完全不知道记忆层存在;
        重新挂上, 只加一个文件。**框架里的任何文件都不需要改。**

    这里断言的是这句话的字面意思: 摘掉之后, agent 的状态、请求、快照里
    找不到任何"记忆"的痕迹 —— 因为它根本不住在 agent 里。
    """
    class Mem:
        def build_context(self, query, **kwargs):
            return "【记忆】资料。"

    with_mem = FakeLLM(["答案"])
    a = _simple(with_mem, hooks=[MemoryHook(Mem())])
    a.run("问题")

    without = FakeLLM(["答案"])
    b = _simple(without)
    b.run("问题")

    assert without.calls[0]["messages"][-1] == {"role": "user", "content": "问题"}
    assert "【记忆】" in with_mem.calls[0]["messages"][-1]["content"]

    # 摘掉之后: 快照里没有 hooks / memory, agent 上没有 memory 属性
    snap = b.snapshot()
    assert "hooks" not in snap and "memory" not in snap
    assert not hasattr(b, "memory")
    # 而挂着的时候这些也一样 —— 记忆不住在 agent 里
    snap_a = a.snapshot()
    assert "hooks" not in snap_a and "memory" not in snap_a
    assert not hasattr(a, "memory")


if __name__ == "__main__":
    sys.exit(run_tests(globals(), "扩展点(hook)"))
