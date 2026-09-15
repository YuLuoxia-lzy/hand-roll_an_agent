"""契约测试 —— 测的不是"这个实现做了什么", 而是"任何实现都该满足什么"

    python tests/test_contracts.py

【为什么框架需要它, 应用不需要】
应用测的是"我的功能对不对", 实现变了测试就该跟着变。
框架测的是"别人按这个接口写的东西能不能跑", 实现变了**契约不能变**。

所以这里的每个用例, 都应该能在**任意**一个符合接口的实现上通过 ——
用例里塞的每一样东西(记忆、hook、store、scorer)都是现写的替身,
而不是框架里的任何一个具体实现。

【一条纪律】
发现某个契约用例要改实现才能过时, 先停下来想: 是契约定错了, 还是实现真的违约了。
**两者都值得停下来, 而不是把测试改成能过。**
(本文件里每一条断言后面都写清了它守的是什么。)

全程离线: 假 LLM / 假 embedder / 假 store, 不需要任何 API key。
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import FakeEmbedder, FakeLLM, TempDir, run_tests          # noqa: E402

from agents0to1 import (                                               # noqa: E402
    Hook,
    MemoryHook,
    PlanAndSolveAgent,
    ReActAgent,
    ReflectionAgent,
    SemanticMemory,
    SimpleAgent,
    Tool,
    ToolParameter,
    ToolRegistry,
)
from agents0to1.core.typedefs import ToolCall                          # noqa: E402


# ==================== 公共替身 ====================

class EchoTool(Tool):
    def __init__(self, name: str = "echo"):
        super().__init__(name=name, description="原样返回输入文本")

    def get_parameters(self):
        return [ToolParameter(name="text", type="string", description="要回显的文本")]

    def run(self, parameters: Dict[str, Any]) -> str:
        return f"回显: {parameters.get('text', '')}"


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_tool(EchoTool())
    return registry


def _cases():
    """
    四种子类 + 各自的剧本 + 恢复时要补的构造参数。

    契约必须对**每一个**子类成立 —— 所以后面几条测试都是循环这个列表,
    而不是只测最好写的那一个。
    """
    registry = _registry()
    return [
        {
            "name": "SimpleAgent",
            "script": ["答案"],
            "build": lambda llm, hooks=None: SimpleAgent(
                "t", llm, system_prompt="你是助手", hooks=hooks),
            "load_kwargs": {},
        },
        {
            "name": "ReActAgent",
            "script": [
                ("我想想", [ToolCall(id="c1", name="echo", arguments={"text": "hi"})]),
                "答案",
            ],
            "build": lambda llm, hooks=None: ReActAgent(
                "t", llm, tool_registry=registry, system_prompt="你是助手", hooks=hooks),
            "load_kwargs": {"tool_registry": registry},
        },
        {
            "name": "ReflectionAgent",
            "script": ["初稿", "无需改进"],
            "build": lambda llm, hooks=None: ReflectionAgent(
                "t", llm, system_prompt="你是助手", hooks=hooks),
            "load_kwargs": {},
        },
        {
            "name": "PlanAndSolveAgent",
            "script": [
                (None, [ToolCall(id="p1", name="submit_plan", arguments={"plan": ["一步"]})]),
                "结果",
            ],
            "build": lambda llm, hooks=None: PlanAndSolveAgent(
                "t", llm, system_prompt="你是助手", hooks=hooks),
            "load_kwargs": {},
        },
    ]


class _Item:
    """一个最小的检索结果 —— 记忆契约不要求一定要是 MemoryItem"""

    def __init__(self, text: str, score: float = 1.0):
        self.text = text
        self.score = score
        self.metadata = {}


# ==================== 契约 1: 任何 memory 对象 ====================

class OnlySearch:
    """只提供 search() —— 排版交给 hook 自己"""

    def search(self, query: str) -> List[_Item]:
        return [_Item(f"来自 OnlySearch 的资料: {query}")]


class OnlyBuildContext:
    """只提供 build_context() —— 自己负责排版"""

    def build_context(self, query: str) -> str:
        return f"来自 OnlyBuildContext 的资料: {query}"


class Both:
    def search(self, query: str) -> List[_Item]:
        return [_Item("search 那条")]

    def build_context(self, query: str) -> str:
        return "build_context 那条"


class Exploding:
    """**每一次调用都会炸的记忆。**

    它必须和"没挂记忆"完全一样地跑完 —— 记忆只是增强, 没有权力把 agent 拖下水。
    """

    def build_context(self, query: str) -> str:
        raise RuntimeError("记忆炸了")

    def search(self, query: str):
        raise RuntimeError("记忆炸了")


def test_any_memory_object_can_be_mounted():
    """**任何**只提供 search() 或 build_context() 的对象, 挂上去都该能用。

    这是 MemoryHook 构造函数里那个类型守卫承诺的东西: "有其中之一就行"。
    两个形状都真的能跑, 而不是"守卫放行了但运行期才发现不行"。
    """
    for memory in (OnlySearch(), OnlyBuildContext(), Both()):
        llm = FakeLLM(["答案"])
        agent = SimpleAgent("t", llm, system_prompt="你是助手",
                            hooks=[MemoryHook(memory)])
        answer = agent.run("问题")

        assert answer == "答案", f"{type(memory).__name__}: 跑都没跑完"
        sent = str(llm.calls[-1]["messages"])
        assert "资料" in sent or "那条" in sent, \
            f"{type(memory).__name__}: 记忆内容根本没进到请求里 —— 挂上了却没用, " \
            f"是最难发现的一种失效"


def test_exploding_memory_is_byte_identical_to_no_memory():
    """**记忆挂了, 行为和没挂时逐字节相同。**

    这是 MemoryHook.on_error="open" + _safe_context 双层兜底要保证的东西,
    也是"记忆层正对 LLM 漏斗, 绝不能抛出去"那条原则的可执行版本。
    """
    plain_llm = FakeLLM(["答案"])
    SimpleAgent("t", plain_llm, system_prompt="你是助手").run("问题")

    broken_llm = FakeLLM(["答案"])
    SimpleAgent("t", broken_llm, system_prompt="你是助手",
                hooks=[MemoryHook(Exploding())]).run("问题")

    assert broken_llm.calls == plain_llm.calls, \
        "记忆炸了之后发出去的请求必须和没挂记忆时一模一样"


def test_memory_object_without_the_interface_is_rejected_early():
    """没有 search / build_context 的对象要在**构造时**就拒绝。

    放它进去的后果是"第一次调 LLM 时才炸", 而那时候报错位置离病因十万八千里。
    """
    try:
        MemoryHook(memory=object())
    except TypeError as e:
        assert "search()" in str(e) and "build_context()" in str(e)
        return
    raise AssertionError("不合规的 memory 对象应该在构造时就被挡住")


# ==================== 契约 2: 任何 hook 抛异常 ====================

class AlwaysBoom(Hook):
    """五个阶段**全部**抛异常。

    注意它是 on_error="open" 的: 框架该吞掉它、记一条 warning、继续跑,
    而且跑出来的东西必须和没挂它时**逐字节相同**。
    """

    on_error = "open"

    def before_input(self, ctx, input_text):
        raise RuntimeError("boom")

    def before_llm(self, ctx, messages):
        raise RuntimeError("boom")

    def after_llm(self, ctx, response):
        raise RuntimeError("boom")

    def after_tool(self, ctx, call, result):
        raise RuntimeError("boom")

    def after_run(self, ctx, turn):
        raise RuntimeError("boom")


def test_any_hook_that_fails_open_does_not_change_behavior():
    """**任何一个会炸的 hook**, 只要它声明了 on_error="open", 就不该改变行为。

    对四个 Agent 各验一遍 —— 埋点在基类和 ReAct 里, 只测 SimpleAgent 测不到
    after_tool 那一处。
    """
    for case in _cases():
        name = case["name"]

        plain_llm = FakeLLM(case["script"])
        plain = case["build"](plain_llm).run("问题")

        boom_llm = FakeLLM(case["script"])
        broken = case["build"](boom_llm, hooks=[AlwaysBoom()]).run("问题")

        assert broken == plain, f"{name}: 炸掉的 hook 改变了最终答案"
        assert boom_llm.calls == plain_llm.calls, \
            f"{name}: 炸掉的 hook 改变了发出去的请求 —— 它必须什么都没做"


def test_hook_that_fails_closed_does_interrupt():
    """反过来: on_error="closed" 的 hook 抛异常, 必须**穿透出来**。

    框架没有资格替应用决定"这个 hook 挂了要不要中断对话" ——
    所以两种策略都必须真的有效, 而不是都往一个方向倒。
    """
    class MustStop(Hook):
        on_error = "closed"

        def before_input(self, ctx, input_text):
            raise RuntimeError("余额不足")

    llm = FakeLLM(["答案"])
    agent = SimpleAgent("t", llm, system_prompt="你是助手", hooks=[MustStop()])

    try:
        agent.run("问题")
    except RuntimeError as e:
        assert "余额不足" in str(e), "异常被换了壳 —— 应用要能认出自己抛的那个"
        assert not llm.calls, "中断要发生在调 LLM 之前"
        return
    raise AssertionError("声明了 closed 的 hook 抛异常必须中断本轮")


# ==================== 契约 3: 任何 Agent 子类的事件流 ====================

#: 框架认得的全部事件类型。加新类型要同步这里 —— 它就是对外承诺的清单。
_EVENT_TYPES = {"text", "thinking", "tool_call", "tool_result", "final"}


def _check_event_contract(name: str, events: List[Any]) -> None:
    """任何 Agent 的 stream_run 都该满足这些 —— 与它内部怎么实现无关。"""
    assert events, f"{name}: stream_run 一个事件都没吐"

    types = [e.type for e in events]
    assert set(types) <= _EVENT_TYPES, f"{name}: 出现了没见过的类型 {set(types) - _EVENT_TYPES}"
    assert types.count("final") == 1, f"{name}: final 必须恰好一个, 实际 {types.count('final')} 个"
    assert types[-1] == "final", f"{name}: final 必须是最后一个事件, 实际 {types}"

    for e in events:
        if e.type in ("text", "thinking"):
            assert e.text is not None, f"{name}: {e.type} 事件没带 text"
        elif e.type == "tool_call":
            assert e.call is not None, f"{name}: tool_call 事件没带 call"
        elif e.type == "tool_result":
            assert e.call is not None and e.result is not None, f"{name}: tool_result 少了 call 或 result"
        elif e.type == "final":
            assert e.answer is not None, f"{name}: final 事件没带 answer"

      # 工具事件必须成对、且结果在调用之后
    call_at = {e.call.id: i for i, e in enumerate(events) if e.type == "tool_call"}
    result_at = {e.call.id: i for i, e in enumerate(events) if e.type == "tool_result"}
    for call_id, index in result_at.items():
        assert call_id in call_at, f"{name}: 没发过 tool_call 就来了 tool_result({call_id})"
        assert call_at[call_id] < index, f"{name}: tool_result({call_id}) 出现在它的调用之前"
    assert set(call_at) == set(result_at), \
        f"{name}: 有 tool_call 没等到结果: {set(call_at) - set(result_at)}"

    # 过程叙述不得混进正文。
    #
    # 【为什么这条是契约而不是风格】
    # 调用方把 text 事件拼起来当"这一轮的回答"—— 这是 typedefs.py 明写的用法。
    # 于是同一段内容只要**既作为 text 又作为 thinking** 出现, 那句"我先查一下…"
    # 就会混进最终答案里。这个错误不报异常、不影响工具执行, 只是答案被弄脏,
    # 而且看起来还挺通顺 —— 属于最该由测试盯住的那一类。
    _joined_text = "".join(e.text for e in events if e.type == "text")
    for e in events:
        if e.type == "thinking" and e.text:
            assert e.text not in _joined_text, (
                f"{name}: 同一段过程叙述既出现在 thinking 里, 又混进了 text —— "
                f"把 text 拼起来当答案的调用方会看到: {e.text!r}"
            )


def test_any_agent_subclass_streams_a_legal_event_sequence():
    """**四种范式, 同一个契约。**

    调用方(UI)要靠这个形状渲染: 一段正文 + 若干工具卡片 + 一个结论。
    任何一个子类吐出不合法的事件序列, 都是在把这份承诺悄悄改掉。
    """
    for case in _cases():
        llm = FakeLLM(case["script"])
        agent = case["build"](llm)
        _check_event_contract(case["name"], list(agent.stream_run("问题")))


def test_a_brand_new_subclass_gets_the_event_stream_for_free():
    """**只实现 _run() 的新范式, 事件流是白送的。**

    这是"加一种新 Agent 范式 = 1 个新文件"那条承诺的可执行版本 ——
    如果哪天它需要子类再写点别的才能有事件流, 这条会红。
    """

    class NewParadigm(SimpleAgent):
        """一个新范式: 只写 _run, 别的什么都不碰"""

        def _run(self, input_text: str, **kwargs) -> str:
            return self._chat([{"role": "user", "content": input_text}]).content or ""

    agent = NewParadigm("t", FakeLLM(["新范式的答案"]), system_prompt="你是助手")
    events = list(agent.stream_run("问题"))

    _check_event_contract("NewParadigm", events)
    assert events[-1].answer == "新范式的答案"


def test_tool_call_preamble_is_not_also_emitted_as_text():
    """**工具调用前的那段话, 只能出现在 thinking 里, 不能同时出现在 text 里。**

    【这条是拿一个真出现过的 bug 写的, 不是假想的】
    ReAct 的 _stream_run 曾经是"边收边吐 text", 而"这轮到底是答案还是前奏"要到
    流结束才知道(判据 response.has_tool_calls 只在那时才有值)。于是那段
    "我先查一下…" 先作为 text 吐了出去, 等发现是工具调用, 又整段作为 thinking
    吐了第二遍 —— 同一句话出现两次, 其中一次还混进了"拼起来当答案"的那一串。

    修法只能是先攒着、流完再定性。这条测试盯的就是那个"攒"没有被人改回去:
    只要有人图省事把 yield text 挪回流循环里, 它立刻红。
    """
    registry = _registry()
    llm = FakeLLM([
        ("我先查一下资料", [ToolCall(id="c1", name="echo", arguments={"text": "hi"})]),
        "查完了, 这是答案",
    ])
    agent = ReActAgent("t", llm, tool_registry=registry, system_prompt="你是助手")
    events = list(agent.stream_run("问题"))

    _check_event_contract("ReActAgent(带前奏)", events)

    thinking = [e.text for e in events if e.type == "thinking"]
    body = "".join(e.text for e in events if e.type == "text")

    assert thinking == ["我先查一下资料"], f"前奏该整段出现在 thinking 里: {thinking}"
    assert "我先查一下资料" not in body, f"前奏混进了正文: {body!r}"
    assert body == "查完了, 这是答案", f"正文该只是最终答案: {body!r}"
    assert events[-1].answer == "查完了, 这是答案"


# ==================== 契约 4: 任何 VectorStore 实现 ====================

class DictStore:
    """
    一个**和 VectorStore 毫无血缘关系**的向量库 —— 一个 dict 而已。

    它只提供 add / search / count / clear / close(就是 2.1 那条缝要求的五个方法),
    证明"换 faiss / chroma 就是换一个类"不是一句空话。
    """

    def __init__(self):
        self.rows: List[tuple] = []          # (text, vector, meta)
        self.closed = False

    def add(self, texts, vectors, metadatas=None, model="unknown"):
        metas = list(metadatas) if metadatas is not None else [{} for _ in texts]
        start = len(self.rows)
        for text, vector, meta in zip(texts, vectors, metas):
            self.rows.append((text, list(vector), dict(meta or {})))
        return list(range(start, len(self.rows)))

    def search(self, query_vector, top_k=5, model=None, filters=None, scorer=None):
        def cos(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = sum(x * x for x in a) ** 0.5
            nb = sum(x * x for x in b) ** 0.5
            return dot / (na * nb) if na and nb else 0.0

        out = []
        for text, vector, meta in self.rows:
            if filters and not all(meta.get(k) == v for k, v in filters.items()):
                continue
            score = scorer(query_vector, vector, metadata=meta, created_at=0.0,
                           query_norm=None, vector_norm=None) if scorer else cos(query_vector, vector)
            out.append((score, text, meta))
        out.sort(key=lambda item: -item[0])
        return out[:top_k]

    def count(self):
        return len(self.rows)

    def stats(self):
        return {"count": len(self.rows), "path": "dict", "table": "dict"}

    def clear(self):
        n = len(self.rows)
        self.rows = []
        return n

    def close(self):
        self.closed = True


def test_any_store_implementation_can_back_the_memory():
    """**add 之后 count() 增加, search 能找回** —— 换成任何实现都该成立。

    这条依赖 2.1 那条缝(SemanticMemory(store=...))。它要证明的是
    "存储后端可替换", 而不是"VectorStore 写得好"。
    """
    store = DictStore()
    mem = SemanticMemory(embedder=FakeEmbedder(), store=store)

    assert mem.count() == 0
    mem.remember("小明住在杭州西湖区。")
    assert mem.count() > 0, "入库之后 count() 没增加 —— 写路径根本没走到这个 store"

    got = mem.search("小明住哪里")
    assert got, "写进去了却搜不回来"
    assert "杭州" in got[0].text

    mem.close()
    assert store.closed, "close() 要穿透到 store —— 不关的话 Windows 上文件一直被占"


# ==================== 契约 5: 任何 scorer ====================

def test_any_scorer_defines_the_final_order():
    """传进去的 scorer, 它给的分**就是**最终排序 —— 框架只负责按它排。

    这里用的是"按字符串长度打分"(文档里点名的那个例子): 越长的排越前。
    分数和语义相似度故意毫无关系, 这样才能证明排序真的来自 scorer,
    而不是"余弦恰好也是这个顺序"。
    """
    def by_text_length(query, vector, *, metadata, **kw):
        return float(len(metadata.get("text", "")))

    mem = SemanticMemory(embedder=FakeEmbedder(), path=":memory:", scorer=by_text_length,
                         min_score=-1.0)     # 长度是正数, 但把阈值放开来免得依赖它
    mem.remember("短", {"text": "短"})
    mem.remember("中等长度的一句话", {"text": "中等长度的一句话"})
    mem.remember("这是一句相当长的话, 长到一定排在最前面", {"text": "这是一句相当长的话, 长到一定排在最前面"})

    got = [item.text for item in mem.search("任意查询", top_k=3)]
    lengths = [len(item.metadata["text"]) for item in mem.search("任意查询", top_k=3)]

    assert lengths == sorted(lengths, reverse=True), f"排序没有跟着 scorer 走: {got}"
    assert len(got) == 3
    mem.close()


def test_default_scorer_is_still_pure_cosine():
    """默认必须是 relevance-only。

    文档型 agent 系统的正确选择: 一份 2023 年的手册不该因为"旧"就沉下去。
    小镇那套三因子是**另一套公式**, 得自己传进来 —— 框架不替应用选。
    """
    store = SemanticMemory(embedder=FakeEmbedder(), path=":memory:")
    assert store.scorer is None, "默认不该预置任何打分策略"
    store.close()


# ==================== 契约 6: 快照往返 ====================

def _public_state(agent) -> dict:
    """
    一份 agent 的"对外可见状态" —— 往返前后必须一致的那些东西。

    【为什么 history 只比 role/content, 不比 timestamp】
    get_history() 是**每次调用现造**的 Message 对象, 而 Turn.messages 里存的是
    {role, content} 这样的裸字典 —— **时间戳从来没被持久化过**, 它是"读的那一刻"。

    这个性质与快照无关: 同一个 agent 连着调两次 get_history(), 只要中间跨过一个
    系统时钟 tick(Windows 上约 15ms), 两次的时间戳就不一样。所以拿它当往返一致性的
    判据, 是**契约定错了** —— 它要求了一个框架从未承诺、也没有地方存的东西。

    (这条是实测出来的: 它曾经真的红过一次。)
    """
    return {
        "name": agent.name,
        "agent_id": agent.agent_id,
        "system_prompt": agent.system_prompt,
        "turns": [t.model_dump() for t in agent.get_turns()],
        "history": [(m.role, m.content) for m in agent.get_history()],
    }


def test_snapshot_roundtrip_preserves_public_state():
    """**四种子类, 同一个契约**: 存下来再读回去, 对外可见状态一模一样。

    快照的意义是"同一个 agent 又回来了"。只要有一项对不上, 恢复出来的就是
    另一个 agent —— 而它长得和原来几乎一样, 查起来极费劲。
    """
    for case in _cases():
        name = case["name"]
        with TempDir() as d:
            path = Path(d) / "snap.json"

            agent = case["build"](FakeLLM(case["script"]))
            agent.run("问题")
            before = _public_state(agent)
            assert before["turns"], f"{name}: 快照测试的前置条件没满足 —— 一轮都没跑"

            agent.save(str(path))
            # 恢复时**必须显式传 llm**(和 tool_registry): 句柄不进快照, 见 core/agent.py
            restored = type(agent).load(
                str(path), llm=FakeLLM(case["script"]), **case["load_kwargs"])

            assert _public_state(restored) == before, f"{name}: 快照往返之后状态对不上"


def test_snapshot_roundtrip_survives_a_second_turn():
    """恢复出来的 agent 要能**接着聊** —— 只会读不会写的恢复没有意义"""
    with TempDir() as d:
        path = Path(d) / "snap.json"

        agent = SimpleAgent("t", FakeLLM(["第一轮"]), system_prompt="你是助手")
        agent.run("问题一")
        agent.save(str(path))

        restored = SimpleAgent.load(str(path), llm=FakeLLM(["第二轮"]))
        answer = restored.run("问题二")

        assert answer == "第二轮"
        assert len(restored.get_turns()) == 2, "恢复之后的第二轮没被记进历史"


if __name__ == "__main__":
    sys.exit(run_tests(globals(), "契约"))
