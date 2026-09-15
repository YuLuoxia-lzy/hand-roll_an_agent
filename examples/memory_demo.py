"""记忆层端到端演示 —— 把这条链路从头到尾走一遍, 并且**把关键证据打出来**

跑法(仓库根目录下):
    python examples/memory_demo.py            # 离线: 假 embedder + 假 LLM, 零配置零费用
    python examples/memory_demo.py --real     # 真实: 真 embedding + 真 LLM(要先配好)

【这个示例打印的不是"结果好不好看", 而是三件证据】
  ① 发给模型的 messages 里, 检索结果**并进了最后那条 user 消息**(不是新增一条)
  ② Turn 里存的东西**不含**检索结果(下一轮不会把它当历史重发)
  ③ 快照里**没有** memory(活对象进 json.dumps 会被静默字符串化)

离线模式为什么可以"零配置": 假 embedder 用字符 trigram 做哈希袋 —— 它验证不了
语义(那需要真模型), 但足以让切分、入库、检索、注入、记录这五步全都真跑一遍。
所以**它是给你看接线用的, 不是给你看效果用的**。想看效果就配好真 embedding 再跑。

【注意开头的路径修正】
本机 `pip install -e` 装的是原项目, 直接 `import agents0to1` 很可能拿到的是**原项目**
(它没有记忆层)。tests/_harness.py 里的 _bootstrap_path() 会把本副本插到 sys.path
最前面 —— 所以这里复用它, 而不是自己再写一遍。
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from _harness import FakeEmbedder, FakeLLM, TempDir          # noqa: E402

from agents0to1 import (                                     # noqa: E402
    EpisodicMemory,
    KnowledgeSearchTool,
    ReActAgent,
    SemanticMemory,
    SimpleAgent,
    ToolRegistry,
)
from agents0to1.core.typedefs import ToolCall                # noqa: E402


def hr(title: str = "", char: str = "─"):
    print(f"\n{char * 72}")
    if title:
        print(f"  {title}")
        print(char * 72)


def show_request(llm, index: int = 0):
    """把"我们到底发了什么给模型"原样打出来 —— 整个改造最该亲眼看的就是这个"""
    messages = llm.calls[index]["messages"]
    print(f"  发给模型的 {len(messages)} 条消息:")
    for m in messages:
        content = (m.get("content") or "")
        if len(content) > 300:
            content = content[:300] + f"\n     ...(共 {len(content)} 字符)"
        print(f"    [{m['role']}] {content}")
        if m.get("tool_calls"):
            for tc in m["tool_calls"]:
                print(f"      -> 请求调用 {tc['function']['name']}({tc['function']['arguments']})")


SAMPLE_DOC = """\
差旅报销制度

一、住宿标准
一线城市每晚不超过 600 元, 其他城市不超过 400 元。超出部分需要提前申请,
由部门总监审批后据实报销。同城出差原则上不安排住宿。

二、交通标准
高铁二等座、经济舱。市内交通凭票实报实销。夜间到达或行李较多时可以打车,
需要在报销单上注明原因。自驾出行的油费和过路费按实际里程折算, 不再另发补贴。

三、餐补
每人每天 100 元, 出差当天按半天计算。已由对方单位安排用餐的, 当餐不再补贴。
加班到晚上九点以后可以申请一次夜宵补贴, 上限 30 元。

四、提交时限
发票必须在行程结束后 30 天内提交给财务。逾期需要部门总监签字说明,
超过 90 天的原则上不再受理, 特殊情况需报总经理审批。

五、审批权限
单次报销金额在 2000 元以内的由部门总监审批, 2000 元到 10000 元的由财务总监审批,
超过 10000 元的需要总经理签字。审批链条在系统里自动流转, 不需要纸质签字。

六、违规处理
虚开发票、重复报销、虚构行程的, 除全额追回外, 当年绩效考核不得评为优秀,
情节严重的移交公司合规部门处理。出差期间的个人消费不计入报销范围。

七、海外出差
海外出差需要提前两周提交申请, 并附上行程安排和预算。住宿标准按当地消费水平
分三档, 具体额度在行政系统里查询。外币支出按刷卡当天的汇率折算, 汇率损失
由公司承担。出国前需要完成安全培训, 并在系统里登记紧急联系人。

八、通讯补贴
出差期间产生的漫游费、流量费凭票报销, 单次上限 200 元。因公需要开通国际
漫游的, 提前向行政申请, 由公司统一办理。长期驻外人员按季度发放通讯补贴,
不再单独报销通讯费用。

九、附则
本制度自发布之日起施行, 由财务部负责解释。此前发布的《差旅费管理办法》
同时废止。各部门可以在本制度范围内制定更严格的实施细则, 但不得放宽标准。
制度每年评估一次, 根据实际执行情况调整。
"""


def demo_semantic(embedder, tag: str):
    hr("① 语义记忆: 入库 -> 检索")
    print(f"  用的 embedder: {embedder.model_id}  ({tag})")

    with TempDir() as d:
        mem = SemanticMemory(embedder=embedder, path=str(d / "semantic.sqlite3"))

        doc = d / "差旅报销制度.md"
        doc.write_text(SAMPLE_DOC, encoding="utf-8")
        n = mem.ingest_file(str(doc))
        print(f"  入库: {doc.name} -> 切成 {n} 块 (块大小 {mem.chunk_size}, 重叠 {mem.overlap:.0%})")
        print("  ↑ 整篇文档只存一个向量的话, 搜什么都搜不准 —— 切分决定『能不能搜到』。")
        print("     每个块还记着自己是从哪来的(来源/第几块), 否则检索回来你没法引用它。")

        print(f"\n  库里现在有 {mem.count()} 条, 模型指纹 {mem.embedder_id}")
        print(f"  stats: {mem.stats()}")
        if "fake" in mem.embedder_id:
            print("  ↑ 分数低是假 embedder 的正常现象: trigram 词袋在高维空间里几乎两两正交。")
            print("     真模型下相关句子通常 0.5 以上 —— 想看真实效果就跑 --real。")

        for query in ("住宿一晚最多能报多少", "发票最晚什么时候交"):
            items = mem.search(query, top_k=3)
            print(f"\n  检索『{query}』-> {len(items)} 条")
            for i, item in enumerate(items, 1):
                print(f"    [{i}] {item.score:.3f}  {item.text[:52]}...")
                print(f"        来源: {item.citation()}")

        if "fake" in mem.embedder_id:
            print("\n  ↑ 假 embedder 只认字符重叠, 所以上面这个排序**可能和你想的不一样** ——")
            print("     它会把恰好含有相同字词的块排到前面, 哪怕那块里根本没有答案。")
            print("     这不是检索坏了, 是替身的能力边界: 『换个说法问同一件事还能搜到』")
            print("     这件事只有真模型做得到 —— 那正是 test_online.py 要验的东西。")

        print(f"\n  拼成给模型看的上下文(预算 {mem.context_budget} 字符):")
        ctx = mem.build_context("住宿标准")
        print("  " + ctx.replace("\n", "\n  ")[:500])

        # --- 挂到 Agent 上, 看证据 ---
        hr("② 注入: 检索结果去哪了")
        llm = FakeLLM(["出差住宿一线城市每晚不超过 600 元。"])
        agent = SimpleAgent("demo", llm, system_prompt="你是差旅助手", memory=mem)
        answer = agent.run("住宿一晚最多能报多少")

        show_request(llm)
        print(f"\n  ↑ 注意两点: 消息**条数没变**(记忆是并进最后那条 user 的, 不是新增一条),")
        print("     而且**没塞进 system** —— 塞进 system 会让前缀缓存每轮全失效。")

        turn = agent.get_turns()[0]
        print(f"\n  Turn 里实际存下来的 {len(turn.messages)} 条消息:")
        for m in turn.messages:
            print(f"    [{m['role']}] {(m.get('content') or '')[:60]}")
        print(f"\n  ↑ 这里**没有**检索结果 —— 存进去的话, 下一轮它会作为『历史』"
              f"被重新发一遍,")
        print(f"     而 _turn_chars 的预算也会被它白吃。当前这一轮的字符量: "
              f"{agent._turn_chars(turn)}")
        print(f"  答案: {answer}")

        # --- 铁律 ---
        print(f"\n  跑完之后库里还是 {mem.count()} 条 —— agent 不会把模型自己的回答写进记忆。")
        print("  (否则幻觉会被自己的记忆反复加固: 编一个说法 -> 入库 -> 下次当成事实检索出来)")

        # --- 快照 ---
        hr("③ 快照: memory 不该进去")
        snap = agent.snapshot()
        print(f"  快照顶层字段: {sorted(snap)}")
        print(f"  有 'memory' 吗: {'memory' in snap}")
        text = json.dumps(snap, ensure_ascii=False, default=str)
        print(f"  json.dumps(default=str) 之后有被字符串化的活对象吗: "
              f"{'object at 0x' in text}")
        print("  ↑ 活对象塞进快照会被**静默**变成 '<... object at 0x...>', 不报错,")
        print("     直到 load 出来第一次调 LLM 才炸, 报错位置离病因十万八千里。")

        mem.close()


def demo_tool(embedder, tag: str):
    hr("④ 工具版: 让模型自己去查")
    with TempDir() as d:
        mem = SemanticMemory(embedder=embedder, path=str(d / "semantic.sqlite3"))
        mem.remember("差旅住宿标准: 一线城市每晚不超过 600 元, 其他城市 400 元。")
        mem.remember("发票必须在行程结束后 30 天内提交给财务。")

        registry = ToolRegistry()
        registry.register_tool(KnowledgeSearchTool(mem))
        print(f"  注册的工具: {[t['function']['name'] for t in registry.get_tools_schema()]}")

        # 假 LLM 的剧本: 第一步调工具, 第二步给答案
        llm = FakeLLM([
            (None, [ToolCall(id="c1", name="knowledge_search",
                             arguments={"query": "住宿一晚多少钱"})]),
            "一线城市每晚不超过 600 元。",
        ])
        agent = ReActAgent("demo", llm, tool_registry=registry,
                           system_prompt="你是差旅助手", memory=mem)
        answer = agent.run("住宿一晚最多能报多少")

        for i, call in enumerate(llm.calls):
            print(f"\n  --- 第 {i + 1} 次 LLM 调用 ---")
            show_request(llm, i)

        print(f"\n  答案: {answer}")
        print("\n  工具版的三个局限(这就是为什么还要做上下文版):")
        print("    1. 只有 ReActAgent 有工具, 另外三个 Agent 什么都得不到")
        print("    2. 模型得**自己想起来**去调 —— 不调就是静默失效, 日志里看不出来")
        print("    3. 多一次工具往返 = 多一次 LLM 调用, 还得跟模型的『想不想』较劲")
        mem.close()


def demo_episodic(embedder, tag: str):
    hr("⑤ 情景记忆: 换个进程还记得上次聊了什么")
    with TempDir() as d:
        path = str(d / "episodic.sqlite3")

        # --- 会话一 ---
        llm1 = FakeLLM(["好的, 评审会定在周三下午。", "收到, 改到周三下午三点。"])
        a1 = SimpleAgent("demo", llm1, system_prompt="你是会议助手")
        a1.run("帮我把评审会定在周三下午。")
        a1.run("时间改到三点吧。")

        with EpisodicMemory(path=path, session_id="week-37") as ep:
            saved = ep.save_agent(a1)
            print(f"  会话一结束, 存了 {saved} 轮, 库里共 {ep.count()} 轮")

        # --- 会话二: 全新的 agent, 什么都不记得 ---
        llm2 = FakeLLM(["上次我们把评审会定在了周三下午三点。"])
        a2 = SimpleAgent("demo", llm2, system_prompt="你是会议助手")
        print(f"  新 agent 的历史: {len(a2.get_turns())} 轮(全新的, 什么都不记得)")

        with EpisodicMemory(path=path, session_id="week-37") as ep:
            n = ep.replay_into(a2)
            print(f"  回放 {n} 轮之后: {len(a2.get_turns())} 轮")
            for t in a2.get_turns():
                print(f"    [user]      {t.user}")
                print(f"    [assistant] {t.answer}")

        a2.run("评审会定在什么时候来着?")
        print("\n  新会话的请求里带上了上次的对话:")
        show_request(llm2)
        print("\n  ↑ 注意 save_agent 是**追加**语义, 不是同步 —— 同一个 agent 存两次会存成两份。")
        print("     它是『记流水账』, 不是『覆盖快照』。见 episodic.py 末尾那段注释。")


def demo_fail_open(embedder, tag: str):
    hr("⑥ fail-open: embedding 挂了会怎样")
    with TempDir() as d:
        mem = SemanticMemory(embedder=embedder, path=str(d / "semantic.sqlite3"))
        mem.remember("随便一条内容。")

        # 把 embedder 弄挂, 模拟"服务商宕机/欠费/网络不通"
        mem.embedder.fail = True

        try:
            mem.search("随便问")
            print("  search() 没有抛 —— 不对, 它是编程接口, 出错必须让调用方知道")
        except Exception as e:
            print(f"  search() 抛了: {type(e).__name__} —— 这是对的, 编程接口不该偷偷返回空")

        print(f"  build_context() 返回: {mem.build_context('随便问')!r}")
        print("     ↑ 空字符串, 不抛 —— 它正对着 LLM 漏斗, 抛出去的后果是整个 agent 挂掉")

        llm = FakeLLM(["答案"])
        agent = SimpleAgent("demo", llm, system_prompt="你是助手", memory=mem)
        agent.run("问题")
        print(f"  agent 照常跑完, 发给模型的还是 {len(llm.calls[0]['messages'])} 条消息, "
              f"和『没有记忆』逐字节相同")
        print("     ↑ 记忆只是增强, 它没有权力把整个 agent 拖下水")
        mem.close()


def main():
    real = "--real" in sys.argv

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 72)
    print("  记忆层端到端演示")
    print("=" * 72)

    if real:
        from agents0to1 import Agents0to1
        from agents0to1.memory.embedding import EmbeddingClient
        try:
            embedder = EmbeddingClient()
            tag = "真实 embedding"
            print(f"  embedder: {embedder.provider}/{embedder.model}")
        except Exception as e:
            print(f"  真实 embedding 不可用: {e}")
            print("  先装 ollama 并 `ollama pull nomic-embed-text`, 或配一个云端 embedding key。")
            print("  下面退回离线模式。")
            real = False
        else:
            # 顺带把 chat 侧也解析一遍打出来 —— embedding 和 chat 是**两张独立的表**
            # (memory/embedding.py:_EMBEDDING_PROVIDERS vs core/llm.py:_ENV_PROVIDER_KEYS),
            # 所以这里能亲眼看见它们各走各的, 而不是靠"应该不会串"来推断。
            # 只建客户端, 不发请求, 不花钱。
            try:
                chat = Agents0to1()
                print(f"  chat    : {chat.provider}/{chat.model}")
            except Exception as e:
                print(f"  chat 侧没解析出来: {e}")

    if not real:
        embedder = FakeEmbedder(dim=256)
        tag = "假的, 用字符 trigram —— 只看接线, 不看效果"

    demo_semantic(embedder, tag)
    demo_tool(embedder, tag)
    demo_episodic(embedder, tag)
    demo_fail_open(embedder, tag)

    hr()
    print("  跑完了。接下来:" if real else "  离线跑完了(上面全是假 embedder)。接下来:")
    print("    python tests/run_offline.py         # 94 个离线用例, 改完代码的标准动作")
    print("    python tests/bench_storage.py       # BLOB vs JSON 差多少, 亲手量一遍")
    print("    python tests/test_online.py         # 真 embedding 才能验的语义检索")
    print("    python examples/memory_demo.py --real   # 这个示例接真模型再跑一遍")
    print()


if __name__ == "__main__":
    main()
