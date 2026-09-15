"""Agent 接入测试 —— 第五步是整个方案最容易做错的地方

    python tests/test_agent_memory.py

这个文件里每一条都对应指南第五步里的一个坑, 注释写清楚了"如果不这样就
会发生什么"。全部离线, 用假 LLM + 假记忆, 不花钱。

用的都是 FakeLLM 记下来的**"我们到底发了什么给模型"** —— 验证记忆有没有生效,
靠的从来不是"模型说了什么", 而是"请求里有什么"。
"""

import inspect
import json
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import FakeEmbedder, FakeLLM, TempDir, run_tests       # noqa: E402

from agents0to1 import (                                             # noqa: E402
    CalculatorTool,
    Config,
    KnowledgeSearchTool,
    PlanAndSolveAgent,
    ReActAgent,
    ReflectionAgent,
    SemanticMemory,
    SimpleAgent,
    Tool,
    ToolParameter,
    ToolRegistry,
)
from agents0to1.core.typedefs import ToolCall                        # noqa: E402

CONTEXT_MARK = "【记忆】"


# ==================== 测试替身 ====================

class RecordingMemory:
    """
    会记录"被问了什么"的假记忆 —— 用来验证**查询用的是哪个字符串**。

    它同时提供 search() 和 build_context(), 所以两种守卫路径都能走。
    """

    def __init__(self, context: str = f"{CONTEXT_MARK}相关资料在这里。", fail: bool = False):
        self.context = context
        self.fail = fail
        self.queries: List[str] = []
        self.context_calls = 0

    def build_context(self, query, **kwargs) -> str:
        self.queries.append(query)
        self.context_calls += 1
        if self.fail:
            raise RuntimeError("假记忆被要求失败")
        return self.context

    def search(self, query, top_k=None, min_score=None):
        self.queries.append(query)
        if self.fail:
            raise RuntimeError("假记忆被要求失败")
        return []


class EchoTool(Tool):
    """确定性工具 —— 不依赖时间/网络, 便于断言"""

    def __init__(self, name: str = "echo"):
        super().__init__(name=name, description="原样返回输入文本")

    def get_parameters(self):
        return [ToolParameter(name="text", type="string", description="要回显的文本")]

    def run(self, parameters: Dict[str, Any]) -> str:
        return f"回显: {parameters.get('text', '')}"


def _simple(llm, memory=None, **kwargs):
    return SimpleAgent("test", llm, system_prompt="你是助手", memory=memory, **kwargs)


def _react(llm, memory=None, registry=None):
    return ReActAgent(
        "test", llm,
        tool_registry=registry or ToolRegistry(),
        system_prompt="你是助手",
        memory=memory,
    )


# ==================== 基本: 有 / 没有记忆 ====================

def test_memory_none_is_a_noop():
    llm = FakeLLM(["答案"])
    agent = _simple(llm)
    agent.run("问题")

    messages = llm.calls[0]["messages"]
    assert messages[-1] == {"role": "user", "content": "问题"}, messages[-1]


def test_injection_goes_into_the_last_user_message():
    """
    注入形态: **并进最终那条 user 消息**, 不新增消息。

        单独一条 system 插中间 -> Llama-2 模板不吃
        单独一条 user          -> 连续两条 user, 模板和严格校验器都拒绝
        并进最后一条 user      -> 不新增消息、不动角色序列, 谁都吃
    """
    llm = FakeLLM(["答案"])
    agent = _simple(llm, memory=RecordingMemory())
    agent.run("用户的问题")

    messages = llm.calls[0]["messages"]
    assert len(messages) == 2, f"消息条数变了: {[m['role'] for m in messages]}"
    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"].startswith(CONTEXT_MARK), messages[-1]["content"][:40]
    assert messages[-1]["content"].endswith("用户的问题")


def test_injection_is_not_in_system_message():
    """塞进 system 会让 `system + history` 这段稳定前缀每轮全变,
    DeepSeek 的自动前缀缓存全部失效 —— 缓存命中和不命中的价格差很多。"""
    llm = FakeLLM(["答案"])
    agent = _simple(llm, memory=RecordingMemory())
    agent.run("问题")

    system = llm.calls[0]["messages"][0]
    assert system["role"] == "system"
    assert CONTEXT_MARK not in system["content"], "记忆被塞进 system 了"


def test_context_is_after_history_not_before():
    """历史是前缀, 记忆是后缀 —— 顺序反了前缀缓存同样失效"""
    llm = FakeLLM(["答案一"])
    agent = _simple(llm, memory=RecordingMemory())
    agent.run("第一问")
    agent.run("第二问")

    messages = llm.calls[1]["messages"]
    roles = [m["role"] for m in messages]
    # system, user(第一轮), assistant(第一轮), user(第二轮, 带记忆)
    assert roles == ["system", "user", "assistant", "user"], roles
    assert messages[1]["content"] == "第一问", "历史里的 user 被改动了"
    assert CONTEXT_MARK in messages[-1]["content"]


# ==================== 第一条铁律: 注入的内容不进 Turn ====================

def test_injected_context_is_not_stored_in_turn():
    """
    【本文件最重要的一条】

    钩子挂在 _build_messages 上, 而 turn_start 是在它返回**之后**才算的,
    所以插进去的内容天然落在 turn_start 之前 -> 不会被存进 Turn -> 零拷贝。

    如果不小心挂到了 _chat 上、或者插在了 turn_start 之后, 会发生:
    历史膨胀、快照膨胀、下一轮把上一轮的检索结果当历史重发、
    _turn_chars 的预算被白白吃掉 —— 而且**一个错都不报**。
    """
    llm = FakeLLM(["答案"])
    agent = _simple(llm, memory=RecordingMemory())
    agent.run("用户的问题")

    turns = agent.get_turns()
    assert len(turns) == 1
    stored = json.dumps(turns[0].messages, ensure_ascii=False)
    assert CONTEXT_MARK not in stored, f"检索结果被存进 Turn 了:\n{stored[:200]}"
    assert turns[0].messages[0] == {"role": "user", "content": "用户的问题"}
    assert turns[0].user == "用户的问题"


def test_next_turn_does_not_resend_previous_context():
    """上一轮的检索结果不该在下一轮被当成历史重发"""
    llm = FakeLLM(["答案"])
    agent = _simple(llm, memory=RecordingMemory())
    agent.run("第一问")
    agent.run("第二问")

    messages = llm.calls[1]["messages"]
    assert sum(CONTEXT_MARK in (m.get("content") or "") for m in messages) == 1, (
        "历史里带上了上一轮的检索结果 —— 它被存进 Turn 了"
    )


def test_react_turn_does_not_store_the_context():
    """
    【上面那条只证明了 SimpleAgent —— ReAct 得单独盯, 而这个差别害人】

    SimpleAgent 记的是 (input_text, answer) 两个值, 天然干净。
    ReAct 记的是 messages[turn_start:](react_agent.py:239), 而
    turn_start = len(messages) - 1 **正好指向那条已被注入记忆的 user 消息** ——
    所以"注入天然不进 Turn"这句话对 ReAct 是**错的**, 得显式还原。

    这个 bug 是写这个文件时才发现的: 上面那条测试绿着, 因为它是 SimpleAgent。
    """
    registry = ToolRegistry()
    registry.register_tool(EchoTool())
    llm = FakeLLM([
        (None, [ToolCall(id="t1", name="echo", arguments={"text": "x"})]),
        "最终答案",
    ])
    agent = _react(llm, memory=RecordingMemory(), registry=registry)
    agent.run("用户的问题")

    turn = agent.get_turns()[0]
    stored = json.dumps(turn.messages, ensure_ascii=False)
    assert CONTEXT_MARK not in stored, f"检索结果被 ReAct 存进 Turn 了:\n{stored[:200]}"
    assert turn.messages[0] == {"role": "user", "content": "用户的问题"}, turn.messages[0]
    # 还原第一条不能把轨迹弄坏 —— tool 的配对还得完整
    assert [m["role"] for m in turn.messages] == ["user", "assistant", "tool", "assistant"]
    assert turn.is_valid(), "轨迹被改坏了(assistant 的 tool_calls 和 tool 消息对不上)"

    # 第二轮: 上一轮的检索结果不能再出现一遍
    agent.run("第二个问题")
    messages = llm.calls[-1]["messages"]
    assert sum(CONTEXT_MARK in (m.get("content") or "") for m in messages) == 1, (
        "第一轮的检索结果被当成历史重发了"
    )


def test_turn_chars_budget_not_eaten_by_context():
    """_turn_chars 数的是"这一轮实际会发给模型的字符量" —— 检索结果不该被算进去。

    它被算进去的后果: 历史压缩会误以为这一轮很长, 提前砍掉别的轮次。
    """
    question, answer = "问题", "答案"
    llm = FakeLLM([answer])
    agent = _simple(llm, memory=RecordingMemory(context="【记忆】" + "很长的资料。" * 200))
    agent.run(question)

    turn = agent.get_turns()[0]
    assert agent._turn_chars(turn) == len(question) + len(answer), (
        f"检索结果被算进 _turn_chars 了: {agent._turn_chars(turn)}"
    )


# ==================== 查询用哪个字符串 ====================

def test_query_uses_input_text_on_first_step():
    mem = RecordingMemory()
    llm = FakeLLM(["答案"])
    _simple(llm, memory=mem).run("用户的问题")
    assert mem.queries == ["用户的问题"], mem.queries


def test_react_step_two_still_queries_with_input_text():
    """
    【必须用 input_text 参数本身, 绝对不能用 messages[-1]["content"]】

    真写成"从 messages[-1] 取查询词"的话, ReAct 第 2 步的消息末尾是**工具结果**,
    于是每一步都在拿上一步的工具结果去检索 —— 检索出来的东西和用户问题毫无关系,
    而且它不报错, 你只会觉得"记忆好像没什么用"。

    这里的实现是: 在 _build_messages 里用 input_text 检索一次, 然后把这条带记忆的
    user 消息留在列表里, 后续步骤往它**后面**追加。
    """
    mem = RecordingMemory()
    registry = ToolRegistry()
    registry.register_tool(EchoTool())

    llm = FakeLLM([
        (None, [ToolCall(id="t1", name="echo", arguments={"text": "工具参数"})]),
        "最终答案",
    ])
    agent = _react(llm, memory=mem, registry=registry)
    agent.run("用户的问题")

    assert len(llm.calls) == 2, "这个用例得真的走两步才有意义"
    assert mem.queries == ["用户的问题"], (
        f"查询字符串不对: {mem.queries} —— 每次 run 只检索一次, 且用的是 input_text"
    )

    # 第 2 步的请求末尾是 tool 消息(这正是"不能用 messages[-1]"的原因),
    # 而带记忆的那条 user 消息还在列表里 —— 它不会因为后续步骤被丢掉。
    step2 = llm.calls[1]["messages"]
    assert step2[-1]["role"] == "tool", step2[-1]
    assert step2[-3]["role"] == "user" and CONTEXT_MARK in step2[-3]["content"], (
        f"第 2 步把记忆丢了: {[m['role'] for m in step2]}"
    )


def test_plan_solve_does_not_query_with_the_instruction_template():
    """
    PlanAndSolve 的 messages[-1] 是"执行专家, 请只输出当前步骤的答案"这种**指令模板**。
    拿它去检索, 检索出来的是模板本身。
    """
    mem = RecordingMemory(context="")
    llm = FakeLLM([
        (None, [ToolCall(id="p1", name="submit_plan", arguments={"plan": ["第一步"]})]),
        "第一步的结果",
    ])
    agent = PlanAndSolveAgent("test", llm, system_prompt="你是助手", memory=mem)
    agent.run("用户的问题")

    assert mem.queries, "PlanSolve 根本没检索"
    for q in mem.queries:
        assert "执行专家" not in q and "当前步骤" not in q, f"查询字符串是指令模板: {q[:80]}"
        assert "用户的问题" in q, f"查询字符串里没有用户问题: {q[:80]}"


# ==================== 各 Agent 的注入点不一样 ====================

def test_plan_solve_retrieves_only_once():
    """在 run() 里算一次, 显式传进 planner 和 executor —— 每一步都重检的话,
    同一份资料会在不同步骤里被检索出不同片段, 结果没法复现。"""
    mem = RecordingMemory(context="【记忆】部署手册在这里。")
    llm = FakeLLM([
        (None, [ToolCall(id="p1", name="submit_plan", arguments={"plan": ["第一步", "第二步"]})]),
        "结果一",
        "结果二",
    ])
    agent = PlanAndSolveAgent("test", llm, system_prompt="你是助手", memory=mem)
    agent.run("用户的问题")

    assert mem.context_calls == 1, f"检索了 {mem.context_calls} 次, 应该只检索一次"

    prompts = [c["messages"][-1]["content"] for c in llm.calls]
    assert len(prompts) == 3, f"应该是 1 次规划 + 2 次执行, 实际 {len(prompts)} 次"
    for i, p in enumerate(prompts):
        assert "【记忆】" in p, f"第 {i} 次调用没带上记忆"


def test_reflection_feeds_only_the_initial_draft():
    """
    只喂初稿, **不喂评审**。

    理由有两层:
    1. 这个 Agent 的 messages[-1] 是"模型自己的草稿/评审意见", 不是用户问题
    2. 喂到后面几轮没用 —— 检索结果会混进评审意见里, 变成"记忆在评自己", 越评越偏
    """
    mem = RecordingMemory(context="【记忆】参考资料。")
    llm = FakeLLM(["初稿", "无需改进"])
    agent = ReflectionAgent("test", llm, system_prompt="你是助手", memory=mem)
    agent.run("用户的任务")

    assert mem.context_calls == 1, f"检索了 {mem.context_calls} 次"
    first, second = (c["messages"][-1]["content"] for c in llm.calls[:2])
    assert "【记忆】" in first, "初稿没带上记忆"
    assert "【记忆】" not in second, "评审也带上记忆了 —— 会变成记忆在评自己"


def test_reflection_scratch_does_not_clobber_memory():
    """
    【命名冲突的坑】

    ReflectionAgent 原来有个 self.memory(本轮的草稿轨迹), 而基类现在
    也要用 self.memory 放记忆对象 —— 基类那个会被 run() 静默冲掉,
    结果是"记忆挂上了但一个字符都没生效", 从日志里完全看不出来。

    现在: self.memory 归基类, 草稿轨迹叫 self.scratch。
    """
    mem = RecordingMemory()
    llm = FakeLLM(["初稿", "无需改进"])
    agent = ReflectionAgent("test", llm, system_prompt="你是助手", memory=mem)

    assert agent.memory is mem, "self.memory 被草稿轨迹冲掉了"
    agent.run("任务")
    assert agent.memory is mem, "run() 之后 self.memory 被冲掉了"
    assert agent.scratch.get_last_execution() == "初稿"


# ==================== fail-open(最容易写出幽灵 bug 的地方) ====================

def test_fail_open_is_byte_identical_to_no_memory():
    """
    **必须逐字节相同**, 不是"差不多"。

    _chat / _stream_chat 是每一次 LLM 调用的唯一漏斗。embedding 失败很可能发生
    (独立 provider、独立 key、DeepSeek 还没有 embedding 端点)。
    失败时必须退化成和 memory=None **逐字节相同**的行为。
    """
    without = FakeLLM(["答案"])
    _simple(without).run("问题")

    with_broken = FakeLLM(["答案"])
    _simple(with_broken, memory=RecordingMemory(fail=True)).run("问题")

    assert without.calls[0]["messages"] == with_broken.calls[0]["messages"], (
        "fail-open 之后发给模型的消息和『没有记忆』时不一样了"
    )


def test_empty_context_is_also_identical():
    """检索成功但没结果(空字符串)时, 也不该在消息里留下痕迹"""
    without = FakeLLM(["答案"])
    _simple(without).run("问题")

    with_empty = FakeLLM(["答案"])
    _simple(with_empty, memory=RecordingMemory(context="")).run("问题")

    assert without.calls[0]["messages"] == with_empty.calls[0]["messages"]


def test_fail_open_in_streaming_path_yields_nothing_extra():
    """
    【流式路径的 fail-open 最容易写成幽灵 bug】

    如果增强逻辑失败时直接 `return`:
      1. yield from self.llm.stream_invoke(...) 没被执行
      2. -> llm.last_response 没被重置
      3. -> self._last_response 没被更新(只有生成器跑完才会执行)
      4. -> ReAct 读到的是**上一轮的**响应
      5. -> 看到 has_tool_calls -> 把上一步的工具调用**重跑一遍**

    不报错、不崩溃, 只是行为诡异 —— 而且你很难想到要去查那里。

    而且**绝不能额外 yield 任何东西**: chunk 是逐字流给 UI 的,
    多吐一个字符用户就看得见。
    """
    llm = FakeLLM(["答案"])
    agent = _simple(llm, memory=RecordingMemory(fail=True))

    text = "".join(e.text for e in agent.stream_run("问题") if e.type == "text")
    assert text == "答案", f"流被污染了: {text!r}"

    # 第二轮流式: 确认 _last_response 确实被更新了(没有被上一轮卡住)
    llm2 = FakeLLM(["第一轮答案", "第二轮答案"])
    agent2 = _simple(llm2, memory=RecordingMemory(fail=True))
    first = "".join(e.text for e in agent2.stream_run("问题一") if e.type == "text")
    second = "".join(e.text for e in agent2.stream_run("问题二") if e.type == "text")
    assert first == "第一轮答案" and second == "第二轮答案", (first, second)


def test_react_does_not_rerun_tools_when_memory_fails():
    """幽灵 bug 的最终形态: 记忆失败 -> 读到上一轮响应 -> 工具重跑一遍"""
    registry = ToolRegistry()
    registry.register_tool(EchoTool())
    llm = FakeLLM([
        (None, [ToolCall(id="t1", name="echo", arguments={"text": "x"})]),
        "最终答案",
    ])
    agent = _react(llm, memory=RecordingMemory(fail=True), registry=registry)
    answer = agent.run("问题")

    assert answer == "最终答案", answer
    assert len(llm.calls) == 2, f"步数不对, 工具可能被重跑了: {len(llm.calls)} 次调用"


# ==================== 参数与命名(位置调用会静默错绑) ====================

def test_memory_is_the_last_parameter_of_every_agent():
    """
    四个子类都是用**位置参数**调 super().__init__(name, llm, system_prompt, config)。
    把 memory 插在 config 前面的话, 所有位置调用会把 config **静默绑到 memory** 上。
    """
    for cls in (SimpleAgent, ReActAgent, ReflectionAgent, PlanAndSolveAgent):
        params = list(inspect.signature(cls.__init__).parameters)
        assert params[-1] == "memory", f"{cls.__name__} 的 memory 不在参数表最后: {params}"


def test_positional_call_binds_config_correctly():
    cfg = Config(temperature=0.11)
    agent = SimpleAgent("n", FakeLLM(["答案"]), "人设", cfg)
    assert agent.config is cfg, "config 被绑到别的参数上了"
    assert agent.memory is None


def test_bad_memory_is_rejected_early():
    """构造函数加类型守卫, 和项目已有的 isinstance(agent, Agent) 守卫风格一致"""
    try:
        SimpleAgent("n", FakeLLM(["答案"]), memory={"这不是记忆": True})
    except TypeError as e:
        assert "search" in str(e) or "build_context" in str(e), e
        return
    raise AssertionError("传了个字典当 memory, 应该当场报错 —— "
                         "否则会在第一次调 LLM 时才炸, 报错位置离病因十万八千里")


def test_memory_accepts_search_only_object():
    """只提供 search() 的记忆对象也该收(守卫是"search 或 build_context")"""
    class SearchOnly:
        def search(self, query, top_k=None, min_score=None):
            return []

    SimpleAgent("n", FakeLLM(["答案"]), memory=SearchOnly())


# ==================== 快照: memory 不进快照 ====================

def test_snapshot_excludes_memory():
    """
    snapshot() 用 json.dumps(..., default=str)。一个活的向量库句柄塞进去,
    会被**静默序列化**成 "<memory.semantic.SemanticMemory object at 0x...>" ——
    不报错。然后 _from_snapshot 会把那串字符串喂给构造函数, 直到第一次调 LLM
    才炸, 报错位置离病因十万八千里。
    """
    agent = _simple(FakeLLM(["答案"]), memory=RecordingMemory())
    agent.run("问题")

    snap = agent.snapshot()
    assert "memory" not in snap, "memory 进了快照"

    text = json.dumps(snap, ensure_ascii=False, default=str)
    assert "object at 0x" not in text, f"有活对象被字符串化了:\n{text[:300]}"


def test_snapshot_roundtrip_without_memory():
    with TempDir() as d:
        path = d / "snap.json"
        agent = _simple(FakeLLM(["答案"]), memory=RecordingMemory())
        agent.run("问题")
        agent.save(path)

        # 【必须显式传 llm】不传的话 _from_snapshot 会照着快照里的 provider 建一个
        # **真**客户端, 而 "fake" 落进 _resolve_credentials 的 else 分支要 LLM_API_KEY,
        # 于是它去读环境变量 —— 有 .env 的机器上恰好能读到, 这条就"绿"了。
        # 一颗靠环境变量捂住的红灯, 比一颗红的红灯危险得多: 它只在别人机器上红。
        restored = SimpleAgent.load(str(path), llm=FakeLLM(["答案"]))
        assert restored.memory is None, "从快照恢复出来的 agent 不该凭空有记忆"
        assert restored.get_turns()[0].user == "问题"


def test_load_can_attach_memory_explicitly():
    """想让恢复出来的 agent 也带记忆, 显式传 —— setdefault 机制天然支持覆盖"""
    with TempDir() as d:
        path = d / "snap.json"
        agent = _simple(FakeLLM(["答案"]))
        agent.run("问题")
        agent.save(path)

        mem = RecordingMemory()
        # 显式传 llm: 不传的话 _from_snapshot 会照着快照里的 provider 建一个真客户端,
        # 下一行 run() 就会真的去连网 —— 那这个"离线测试"就跑不起来了
        restored = SimpleAgent.load(str(path), llm=FakeLLM(["答案"]), memory=mem)
        assert restored.memory is mem
        restored.run("新问题")
        assert mem.queries == ["新问题"]


# ==================== 工具版(5.1) ====================

def _memory_with_content(**kwargs):
    mem = SemanticMemory(embedder=FakeEmbedder(), path=":memory:", **kwargs)
    mem.remember("部署流程是先装依赖, 再配置密钥, 最后启动服务。", metadata={"source": "手册.md"})
    mem.remember("团队每周三下午开例会。")
    return mem


def test_knowledge_tool_returns_scored_results():
    mem = _memory_with_content()
    tool = KnowledgeSearchTool(mem)
    out = tool.run({"query": "部署流程是什么"})
    assert "相似度" in out and "手册.md" in out, out
    mem.close()


def test_knowledge_tool_output_is_already_within_budget():
    """
    registry.execute 会对结果做 truncate_output, 而它保留头部 + 尾部 _TAIL_CHARS=800,
    会把最后一块的尾巴接到前面去 —— **相关性顺序就乱了**。

    所以工具必须自己按预算裁好, 让外层无东西可截。
    """
    mem = _memory_with_content(context_budget=120)
    for i in range(10):
        mem.remember(f"第{i}份资料讲的是完全不同的主题。", metadata={"source": f"d{i}.md"})

    tool = KnowledgeSearchTool(mem, top_k=8)
    out = tool.run({"query": "资料"})

    assert tool.truncate_output(out) == out, "结果被外层截断了 —— 应该在 run() 内部就裁好"
    assert len(out) < Tool.max_output_chars
    mem.close()


def test_knowledge_tool_empty_query():
    mem = _memory_with_content()
    out = KnowledgeSearchTool(mem).run({"query": "   "})
    assert out.startswith("错误"), out
    mem.close()


def test_knowledge_tool_no_result_is_explicit():
    """明确说"没查到"而不是返回空串 —— 空串会被模型理解成"工具坏了" """
    mem = SemanticMemory(embedder=FakeEmbedder(), path=":memory:")
    out = KnowledgeSearchTool(mem).run({"query": "库里没有的东西"})
    assert "没有找到" in out, out
    mem.close()


def test_knowledge_tool_rejects_episodic_memory():
    """情景记忆是按时间回放的, 没有 search —— 它该挂给 Agent, 不是做成工具"""
    class NoSearch:
        pass

    try:
        KnowledgeSearchTool(NoSearch())
    except TypeError as e:
        assert "search" in str(e)
        return
    raise AssertionError("没有 search() 的对象应该当场被拒绝")


def test_knowledge_tool_survives_embedder_failure():
    """工具失败返回字符串而不是抛 —— 和 registry.execute 的哲学一致,
    错误本身会作为 tool 消息回传给模型, 模型能自己换个问法重试"""
    mem = _memory_with_content()
    mem.embedder.fail = True
    out = KnowledgeSearchTool(mem).run({"query": "部署"})
    assert out.startswith("错误"), out
    mem.close()


# ==================== 两个工具并发(这一步只有并发测得出) ====================

def test_two_concurrent_knowledge_searches_through_react():
    """
    一轮里模型完全可能同时发两个 knowledge_search(这合法)。
    tools/async_executor.py 用 ThreadPoolExecutor 跑它们 —— sqlite 连接默认
    check_same_thread=True, **第二个线程一进来就 ProgrammingError**。

    这条只有并发测得出: 单线程测一百遍都测不出来, 上线才炸。
    而且这里走的是**真实路径**(ReAct -> execute_many_sync -> 两个线程 -> 两次检索),
    不是直接开线程调 store。
    """
    mem = _memory_with_content()
    registry = ToolRegistry()
    registry.register_tool(KnowledgeSearchTool(mem))

    llm = FakeLLM([
        (None, [
            ToolCall(id="k1", name="knowledge_search", arguments={"query": "部署流程"}),
            ToolCall(id="k2", name="knowledge_search", arguments={"query": "例会时间"}),
        ]),
        "最终答案",
    ])
    agent = _react(llm, memory=mem, registry=registry)
    events = list(agent.stream_run("部署流程和例会时间分别是什么"))

    tool_results = [e.result for e in events if e.type == "tool_result"]
    assert len(tool_results) == 2, tool_results
    for result in tool_results:
        assert not result.startswith("错误"), f"并发检索失败了: {result}"
    mem.close()


# ==================== 那条铁律 ====================

def test_agent_never_writes_assistant_answers_into_memory():
    """
    **绝不自动把 assistant 自己的回答入库。**

    否则会形成自我确认循环: 模型编了一个说法 -> 存进库 -> 下次检索出来 ->
    "这是检索到的事实" -> 模型更相信自己编的东西。幻觉会被自己的记忆反复加固。
    """
    mem = _memory_with_content()
    before = mem.count()

    llm = FakeLLM(["模型自己编的一段话, 绝不该进库。"])
    _simple(llm, memory=mem).run("随便问点什么")

    assert mem.count() == before, (
        f"agent 把模型自己的回答写进记忆了(+{mem.count() - before} 条)—— "
        f"这是那条铁律: 幻觉会被自己的记忆反复加固"
    )
    mem.close()


if __name__ == "__main__":
    sys.exit(run_tests(globals(), "Agent 接入"))
