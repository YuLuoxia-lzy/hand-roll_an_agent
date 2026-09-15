"""情景记忆离线测试 —— 全程不花钱、不联网

    python tests/test_episodic.py

【这个文件补的是一整层的空白】
在这之前, `agents0to1/memory/episodic.py` 有 309 行, 而 tests/ 里**没有任何一个
文件** import 过它 —— 唯一出现 "EpisodicMemory" 字符串的地方是一个**测试名**,
而那个测试用的是本地 `class NoSearch: pass`。

它偏偏是"跨会话"那一层: 向量库搜不到东西你会觉得"检索效果一般", 而情景记忆
写坏了表现为"昨天的对话没了" —— 都是要靠这一层才能发现的。

【为什么最后那条跨进程的用例最重要】
这个模块存在的全部理由就是"跨会话", 而**跨会话 = 跨进程**。
只在一个进程里建两个实例是测不到的: 同一个连接池、同一份 WAL 缓存,
看起来"能读出来", 换个进程就未必了。
"""

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import FakeLLM, REPO_ROOT, TempDir, run_tests          # noqa: E402

from agents0to1 import SimpleAgent                                   # noqa: E402
from agents0to1.core.message import Turn                             # noqa: E402
from agents0to1.memory.episodic import EpisodicMemory                # noqa: E402


def _turn(user: str, answer: str = "") -> Turn:
    """一轮对话 —— 结构照 ReAct 存下来的样子(user -> assistant)"""
    return Turn(
        user=user,
        messages=[
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer},
        ],
        answer=answer,
    )


def _store(**kwargs) -> EpisodicMemory:
    """默认走内存库 —— 每条用例各玩各的, 互不干扰"""
    kwargs.setdefault("path", ":memory:")
    return EpisodicMemory(**kwargs)


# ==================== 写: append / extend / save_agent ====================

def test_append_returns_incrementing_index():
    """序号自动往后排 —— 调用方不需要自己维护计数器, 也就不可能排错"""
    store = _store()
    assert [store.append(_turn(f"第{i}问")) for i in range(3)] == [0, 1, 2]
    assert store.count() == 3
    store.close()


def test_duplicate_turn_index_is_rejected():
    """
    `PRIMARY KEY (session_id, turn_index)` 必须真的在表上 ——
    它是"同一个会话里同一轮不会被悄悄写两遍"的唯一保证。

    光靠 append() 里的 MAX+1 是不够的: 两个进程/两个实例同时写时, 它俩可能算出
    同一个序号。有主键的话第二次插入会**报错**; 没有的话就是静默多出一行,
    而读回来时你会看到一轮莫名其妙的重复。
    """
    import sqlite3

    store = _store()
    store.append(_turn("第一问"))

    try:
        with store._lock:
            # 绕过 append 的序号计算, 直接塞一个已经存在的 (session_id, turn_index)
            store._conn.execute(
                f"INSERT INTO {store.table} "
                f"(session_id, turn_index, user, messages, answer, metadata, created_at) "
                f"VALUES (?, ?, ?, ?, ?, ?, ?)",
                (store.session_id, 0, "偷渡的", "[]", "", "{}", "2026-01-01T00:00:00+00:00"),
            )
    except sqlite3.IntegrityError:
        store.close()
        return
    store.close()
    raise AssertionError("主键没生效: 同一个会话的同一轮被写了两遍")


def test_extend_appends_and_returns_count():
    store = _store()
    assert store.extend([_turn("一"), _turn("二")]) == 2
    assert store.count() == 2
    assert store.extend([]) == 0
    store.close()


def test_save_agent_saves_current_turns():
    store = _store(session_id="alice")
    agent = SimpleAgent("a", FakeLLM(["答一", "答二"]), system_prompt="你是助手")
    agent.run("问一")
    agent.run("问二")

    assert store.save_agent(agent) == 2
    assert [t.user for t in store.recent()] == ["问一", "问二"]
    store.close()


def test_save_agent_limit_keeps_the_newest():
    store = _store()
    agent = SimpleAgent("a", FakeLLM(["答"]), system_prompt="你是助手")
    for i in range(4):
        agent.run(f"第{i}问")

    assert store.save_agent(agent, limit=2) == 2
    assert [t.user for t in store.recent()] == ["第2问", "第3问"]
    store.close()


def test_save_agent_is_append_not_sync():
    """
    **这是这个模块写在注释里的坑, 得有一条用例把它固定下来**:
    save_agent 是追加语义, 不是"同步"。反复调它会把历史重复存进去。

    不在这里自动去重是有理由的(见 episodic.py 末尾那段): 去重需要判断"这一轮
    之前存过没有", 而历史截断会把旧轮丢掉 —— 那些恰恰是已经在库里的轮次。
    靠比对内容去重, 会在"用户问了两次一样的问题"时误判。让调用方决定存哪些更可靠。
    """
    store = _store()
    agent = SimpleAgent("a", FakeLLM(["答"]), system_prompt="你是助手")
    agent.run("问一")

    store.save_agent(agent)                  # 存了 1 轮
    save_again = store.save_agent(agent)     # 又存了 1 轮 —— 同一轮
    assert save_again == 1
    assert store.count() == 2, (
        f"追加语义变了: 存了两遍同一轮, 库里应该是 2 条(重复), 实际 {store.count()} —— "
        f"如果这里变成 1, 说明有人给它加了自动去重, 那会在'问了两次一样的问题'时误判"
    )
    store.close()


# ==================== 读: recent ====================

def test_recent_is_chronological_and_limited():
    """**按时间正序**返回(方便直接拼回历史), limit 取的是最近的几轮"""
    store = _store()
    for i in range(5):
        store.append(_turn(f"第{i}问", f"第{i}答"))

    assert [t.user for t in store.recent(limit=2)] == ["第3问", "第4问"]
    assert [t.user for t in store.recent()] == [f"第{i}问" for i in range(5)]
    assert store.recent(limit=100)[0].user == "第0问", "limit 比库里还大时应该全给, 且仍是正序"
    store.close()


def test_recent_restores_messages_and_answer():
    """Turn 已经是 pydantic 模型, 两头都是现成的序列化 —— 往返必须逐字段一致"""
    store = _store()
    store.append(Turn(
        user="问", messages=[{"role": "user", "content": "问"}], answer="答",
        metadata={"steps": 2},
    ))

    got = store.recent(limit=1)[0]
    assert got.user == "问" and got.answer == "答"
    assert got.messages == [{"role": "user", "content": "问"}]
    assert got.metadata == {"steps": 2}
    assert got.is_valid()
    store.close()


# ==================== 灌回 agent: replay_into ====================

def test_replay_into_puts_history_in_front():
    """
    灌回到**现有历史前面** —— 顺序反了的话, agent 会以为"这段对话发生在我刚说的话之后",
    上下文整个错位, 而且它不报错。
    """
    store = _store(session_id="alice")
    store.extend([_turn("上次的问一", "上次的答一"), _turn("上次的问二", "上次的答二")])

    agent = SimpleAgent("a", FakeLLM(["答"]), system_prompt="你是助手")
    agent.run("这次的问")                     # 让 agent 自己先有一轮

    assert store.replay_into(agent) == 2
    users = [t.user for t in agent.get_turns()]
    assert users == ["上次的问一", "上次的问二", "这次的问"], users
    store.close()


def test_replay_into_respects_history_budget():
    """
    灌进去之后必须走一次 _truncate_history() —— 否则读了一万轮历史就把上下文撑爆了。
    (所以 replay_into 是直接拼 _turns 再统一截断一次, 而不是逐轮 add_turn。)
    """
    from agents0to1 import Config

    store = _store()
    store.extend([_turn(f"旧问{i}", f"旧答{i}") for i in range(50)])

    agent = SimpleAgent(
        "a", FakeLLM(["答"]), system_prompt="你是助手",
        config=Config(max_history_turns=3, max_history_length=100),
    )
    agent.run("新问")
    assert store.replay_into(agent, limit=50) == 50

    turns = agent.get_turns()
    assert len(turns) <= 3, f"历史没被截断, 灌进去 {len(turns)} 轮"
    assert turns[-1].user == "新问", "截断把**最近**的那一轮丢了 —— 应该从最旧的开始丢"
    store.close()


def test_replay_into_empty_store_is_noop():
    store = _store()
    agent = SimpleAgent("a", FakeLLM(["答"]), system_prompt="你是助手")
    agent.run("问")

    assert store.replay_into(agent) == 0
    assert len(agent.get_turns()) == 1
    store.close()


# ==================== 会话隔离 ====================

def test_sessions_are_isolated():
    with TempDir() as d:
        path = str(d / "episodic.sqlite3")
        alice = EpisodicMemory(path, session_id="alice")
        bob = EpisodicMemory(path, session_id="bob")

        alice.extend([_turn("alice 的一"), _turn("alice 的二")])
        bob.append(_turn("bob 的一"))

        assert alice.count() == 2 and bob.count() == 1
        assert [t.user for t in alice.recent()] == ["alice 的一", "alice 的二"]
        assert [t.user for t in bob.recent()] == ["bob 的一"]
        assert alice.sessions() == ["alice", "bob"]

        # 清一个不影响另一个 —— 同一个文件放多个会话的全部意义就在这
        assert alice.clear() == 2
        assert alice.count() == 0
        assert bob.count() == 1, "清了 alice 的会话, bob 的也被清了"
        alice.close()
        bob.close()


def test_count_can_look_at_another_session():
    """count(session_id=...) / recent(session_id=...) 都能跨会话查"""
    with TempDir() as d:
        store = EpisodicMemory(str(d / "e.sqlite3"), session_id="alice")
        store.append(_turn("alice"))
        store.append(_turn("bob 那边的"))                 # 用默认会话
        store._conn.execute(
            f"UPDATE {store.table} SET session_id = 'bob' WHERE user = ?", ("bob 那边的",)
        )
        store._conn.commit()

        assert store.count() == 1
        assert store.count(session_id="bob") == 1
        assert [t.user for t in store.recent(session_id="bob")] == ["bob 那边的"]
        store.close()


# ==================== 给 LLM 看的形态: build_context ====================

def test_build_context_contains_recent_exchanges():
    store = _store()
    store.extend([_turn("我叫小明", "你好小明"), _turn("我住杭州", "好的")])

    ctx = store.build_context()
    assert "小明" in ctx and "杭州" in ctx, ctx
    assert "问:" in ctx and "答:" in ctx
    store.close()


def test_build_context_ignores_the_query():
    """
    情景记忆是**按时间**检索的, 不是按相似度 —— "最近说过什么"和"当前问题像不像"
    基本无关。query 参数留着只是为了和 SemanticMemory.build_context 的签名兼容
    (这样两者都能直接塞给 Agent)。
    """
    store = _store()
    store.append(_turn("今天天气不错", "是挺好的"))

    a = store.build_context("完全无关的问题")
    b = store.build_context(None)
    assert a == b, "query 被用上了 —— 情景记忆不该按相似度检索"
    store.close()


def test_build_context_respects_budget():
    store = _store()
    for i in range(30):
        store.append(_turn(f"第{i}个很长很长的问题" * 5, f"第{i}个很长很长的回答" * 5))

    ctx = store.build_context(limit=30, budget=300)
    assert len(ctx) <= 400, f"预算没生效, 长度 {len(ctx)}"
    store.close()


def test_build_context_is_fail_open():
    """
    **它正对着 LLM 漏斗, 必须 fail-open。**
    它抛出去的后果不是"少了一段上下文", 而是整个 agent 挂掉 ——
    记忆只是增强, 不该有这个权力。这里用"关掉之后再读"来制造失败。
    """
    store = _store()
    store.append(_turn("问", "答"))
    store.close()

    assert store.build_context() == "", "读取失败时应该退化成空字符串, 而不是抛出去"


def test_build_context_empty_store_is_empty_string():
    store = _store()
    assert store.build_context() == ""
    store.close()


# ==================== 持久化 / 跨进程 ====================

def test_persists_across_instances():
    with TempDir() as d:
        path = str(d / "episodic.sqlite3")
        s1 = EpisodicMemory(path, session_id="alice")
        s1.extend([_turn("问一", "答一"), _turn("问二", "答二")])
        s1.close()

        s2 = EpisodicMemory(path, session_id="alice")
        assert s2.count() == 2
        assert [t.user for t in s2.recent()] == ["问一", "问二"]
        # 接着往后写, 序号必须是接着排的 —— 从 0 重来会把老记录撞掉(或报主键冲突)
        assert s2.append(_turn("问三")) == 2
        s2.close()


_CHILD_CODE = """
import sys
sys.path.insert(0, sys.argv[1])
from agents0to1.core.message import Turn
from agents0to1.memory.episodic import EpisodicMemory

store = EpisodicMemory(sys.argv[2], session_id="alice")
print("COUNT_BEFORE", store.count())
store.append(Turn(
    user="子进程问的",
    messages=[{"role": "user", "content": "子进程问的"},
              {"role": "assistant", "content": "子进程答的"}],
    answer="子进程答的",
))
store.close()
print("DONE")
"""


def test_survives_a_real_process_restart():
    """
    **这个模块存在的全部理由就是"跨会话", 而跨会话 = 跨进程。**

    只在一个进程里建两个实例是测不到这件事的 —— 同一个 WAL 缓存、同一个文件句柄,
    看起来"读得出来"。所以这里真的开一个子进程去写, 再回父进程读。
    """
    with TempDir() as d:
        path = str(d / "shared.sqlite3")
        parent = EpisodicMemory(path, session_id="alice")
        parent.append(_turn("父进程问的", "父进程答的"))

        proc = subprocess.run(
            [sys.executable, "-c", _CHILD_CODE, str(REPO_ROOT), path],
            cwd=str(REPO_ROOT),
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            env=dict(os.environ, PYTHONIOENCODING="utf-8"),
        )
        assert proc.returncode == 0, f"子进程失败了:\n{proc.stdout}\n{proc.stderr}"
        assert "COUNT_BEFORE 1" in proc.stdout, (
            f"子进程看不到父进程刚写的那一轮 —— 数据没落盘或被锁住了:\n{proc.stdout}"
        )
        assert "DONE" in proc.stdout

        # 父进程必须能看到子进程写的那一轮 —— 跨进程读得到, 才叫跨会话
        assert parent.count() == 2, f"看不到子进程写的那一轮: {parent.count()}"
        assert [t.user for t in parent.recent()] == ["父进程问的", "子进程问的"]
        assert parent.recent()[-1].answer == "子进程答的"
        parent.close()


def test_close_then_use_fails_clearly():
    """关掉之后再用要给一句人话, 而不是 AttributeError: 'NoneType'"""
    from agents0to1.memory.episodic import EpisodicMemoryException

    store = _store()
    store.close()
    store.close()                                  # 幂等

    try:
        store.count()
    except EpisodicMemoryException as e:
        assert "关闭" in str(e), e
        return
    raise AssertionError("close() 之后 count() 必须报一句清楚的错")


if __name__ == "__main__":
    sys.exit(run_tests(globals(), "情景记忆"))
