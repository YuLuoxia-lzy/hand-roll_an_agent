"""联网测试 —— 唯一需要**真实 embedding 服务**的一组

    D:/anaconda_env/agent/python.exe tests/test_online.py

【它测的是离线测试测不到的那一件事: "语义"两个字到底成不成立】

离线测试用的是假 embedder(字符 trigram 的哈希袋), 它能验证"流程通不通"——
向量进得去、出得来、排序是对的、并发不炸。但它**验证不了语义**:
"余弦相似度怎么算"这句话, 在假 embedder 眼里和"今天中午吃什么"一样,
都是些字符片段的重叠。

真 embedder 才回答得了那个问题: **换个说法问同一件事, 能不能搜到?**

【这些用例不花 chat 的钱】
LLM 那一侧全部用 tests/_harness.FakeLLM(本地假实现) —— 我们只需要它**记下请求**,
用来验证"真实检索出来的内容真的被注进 prompt 了"。所以整个文件的成本
≈ 十几次 embedding 调用, 不产生任何对话费用。

【没有可用的 embedding 服务怎么办】
最省事的是本地 ollama(免费):
    ollama pull nomic-embed-text
    set EMBEDDING_PROVIDER=ollama
或者设云端 key: OPENAI_API_KEY / DASHSCOPE_API_KEY / ZHIPUAI_API_KEY ...
跑不起来时这个文件会**整体跳过并打印怎么配**, 不会假装失败。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import FakeLLM, TempDir, run_tests, skip          # noqa: E402

from agents0to1.memory.embedding import EmbeddingClient, EmbeddingException   # noqa: E402
from agents0to1.memory.semantic import SemanticMemory                        # noqa: E402
from agents0to1 import MemoryHook, SimpleAgent                              # noqa: E402

#: 探测出来的可用客户端。None 表示这个环境跑不了, 原因写在 UNAVAILABLE 里
CLIENT = None
UNAVAILABLE = ""

_SETUP_HINT = """\
没找到可用的 embedding 服务。任选一种:

  ① ollama(本地, 免费, 不联网也能用):
       ollama pull nomic-embed-text
       set EMBEDDING_PROVIDER=ollama

  ② 云端(设一个就行, 注意这些 key 和 chat 的 key 是**分开**的):
       set OPENAI_API_KEY=...        # text-embedding-3-small
       set DASHSCOPE_API_KEY=...     # text-embedding-v3
       set ZHIPUAI_API_KEY=...       # embedding-3

  (DeepSeek 没有 embedding 端点 —— 所以它在这里是认不出来的, 这是故意的)
"""


def probe():
    """
    探测一次, 决定这个文件跑不跑。

    **必须真的发一次请求**: 建对象只校验"有没有 key", 而 ollama 常见的情况是
    key 根本不需要(默认就有个 base_url), 但模型没 pull —— 那时会在第一次
    embed() 才炸, 报错位置离病因很远。
    """
    global CLIENT, UNAVAILABLE

    try:
        client = EmbeddingClient()
    except EmbeddingException as e:
        UNAVAILABLE = f"{e}\n\n{_SETUP_HINT}"
        return

    try:
        client.embed_one("连通性测试")
    except Exception as e:                      # noqa: BLE001 — 网络库的异常类型五花八门
        UNAVAILABLE = (
            f"embedding 服务连不上/模型不可用: {type(e).__name__}: {e}\n"
            f"(provider={client.provider} model={client.model} base_url={client.base_url})\n\n"
            f"{_SETUP_HINT}"
        )
        return

    CLIENT = client


def _client():
    if CLIENT is None:
        skip(UNAVAILABLE or "没有可用的 embedding 服务")
    return CLIENT


def _memory(**kwargs) -> SemanticMemory:
    """真实 embedder + 内存库(不落盘, 不污染 data/); 要落盘的显式传 path="""
    kwargs.setdefault("path", ":memory:")
    return SemanticMemory(embedder=_client(), **kwargs)


def _cos(a, b) -> float:
    from agents0to1.memory.vector_store import cosine_similarity
    return cosine_similarity(a, b)


# ==================== 客户端本身 ====================

def test_dim_is_real_and_stable():
    """维度是**服务端说**的, 不是我们猜的 —— 所以 dim 得等第一次调用之后才有值"""
    client = _client()
    v1 = client.embed_one("第一句话")
    v2 = client.embed_one("第二句完全不同的话")

    assert len(v1) == len(v2) == client.dim, (len(v1), len(v2), client.dim)
    assert client.dim > 0
    assert all(isinstance(x, float) for x in v1[:5])
    print(f"         provider={client.provider} model={client.model} dim={client.dim}")


def test_empty_input_makes_no_request():
    """空列表不该发请求 —— 有些服务端收到空 input 直接报 400"""
    client = _client()
    assert client.embed([]) == []


def test_batching_preserves_order():
    """
    【这条是第三步那个坑的联网版】

    一次 embed() 内部会按 batch_size 切成好几段, 一段一个请求。服务端返回的
    顺序**不保证**和输入一致(OpenAI 兼容接口都是靠 data[].index 表达位置的)——
    不按 index 摆回原位, 向量就和文本错位了。

    错位是**静默**的: 库能建、能搜、不报错, 只是搜出来的东西驴唇不对马嘴。
    所以这里用小 batch(强制切多段)和单次请求各跑一遍, 逐行对比:
    同一段文本, 分批拿到的向量必须和一次性拿到的一致。
    """
    texts = [
        "第 0 句: 春天的风把柳絮吹得到处都是。",
        "第 1 句: 递归要有终止条件, 否则栈会溢出。",
        "第 2 句: 向量的模长是各分量平方和的平方根。",
        "第 3 句: 今天午饭吃的是番茄炒蛋配米饭。",
        "第 4 句: sqlite 打开 WAL 之后读写可以并行。",
        "第 5 句: 缓存命中率低的时候, 该检查前缀是不是变了。",
        "第 6 句: 洗衣服之前记得把口袋里的东西掏出来。",
        "第 7 句: 线程池的大小要看任务是 IO 密集还是 CPU 密集。",
        "第 8 句: 下周三是团队例会, 记得提前准备材料。",
        "第 9 句: 相似度分数太低的结果不该喂给模型。",
        "第 10 句: 切分要在句子边界上切, 否则语义会被切断。",
    ]

    client = _client()
    batched = client.embed(texts)                                   # 内部切成多段

    single = EmbeddingClient(
        model=client.model, api_key=client.api_key, base_url=client.base_url,
        provider=client.provider, batch_size=len(texts),            # 一段发完
    ).embed(texts)

    assert len(batched) == len(single) == len(texts)
    for i, (a, b) in enumerate(zip(batched, single)):
        assert _cos(a, b) > 0.99, (
            f"第 {i} 行的向量对不上(cos={_cos(a, b):.4f}) —— "
            f"分批之后顺序错位了, 文本和向量没对上号"
        )


# ==================== 语义: 真 embedder 才能回答的问题 ====================

_FACTS = [
    ("报销流程是先填单子, 再由主管审批, 最后财务打款。", "报销.md"),
    ("那个蓝色的马克杯里装的是咖啡, 别当成茶。", "厨房.md"),
    ("余弦相似度等于两个向量的点积除以它们的模长之积。", "向量笔记.md"),
    ("服务器重启之前, 记得先把流量从负载均衡上摘下来。", "运维.md"),
]


def test_same_meaning_different_words_still_retrieves():
    """
    **整个改造要成立, 靠的就是这一条。**

    问句里一个"余弦""相似度""点积"都没有 —— 只有"像不像""怎么衡量"。
    假 embedder 在这里必然失败(没有字符重叠), 真 embedder 才做得到。
    """
    mem = _memory()
    for text, source in _FACTS:
        mem.remember(text, metadata={"source": source})

    got = mem.search("向量检索里怎么衡量两个向量像不像")
    assert got, "一条都没搜到 —— 检查是不是所有内容都进了同一个块"
    assert got[0].metadata["source"] == "向量笔记.md", (
        f"搜到的是 {got[0].metadata['source']}: {got[0].text}"
    )
    print(f"         命中: {got[0].text}  (相似度 {got[0].score:.3f})")
    mem.close()


def test_unrelated_question_scores_low():
    """
    库里没有的话题, 分数应该明显低 —— 这正是 min_score 阈值存在的意义。
    没有阈值的话, 模型不管问什么都会收到一段"看起来像资料"的噪声。
    """
    mem = _memory()
    for text, source in _FACTS:
        mem.remember(text, metadata={"source": source})

    hit = mem.search("余弦相似度怎么算")[0].score
    miss = mem.search("请介绍一下北宋的科举制度")[0].score

    assert hit > miss, f"相关问题 {hit:.3f} 竟然不高于无关问题 {miss:.3f}"
    print(f"         相关 {hit:.3f} / 无关 {miss:.3f} (这就是该设阈值的理由)")
    mem.close()


def test_threshold_can_filter_with_a_real_model():
    mem = _memory()
    for text, source in _FACTS:
        mem.remember(text, metadata={"source": source})

    got = mem.search("请介绍一下北宋的科举制度", min_score=0.5)
    assert got == [], f"阈值 0.5 没滤掉无关内容: {[(i.text, round(i.score, 3)) for i in got]}"
    mem.close()


# ==================== 端到端: 真实检索 -> 真实注入 ====================

def test_agent_injects_really_retrieved_content():
    """
    真 embedder + 假 LLM。

    假 LLM 不产生任何费用, 但会把**我们到底发了什么**记下来 —— 于是这一条
    真的端到端验证了整条链路: 文本 -> 切分 -> 真向量 -> 检索 -> 并进 prompt。
    离线测试里那条同样的断言, 用的是假向量, 只证明了"接线是通的"。
    """
    with TempDir() as d:
        doc = d / "报销制度.md"
        doc.write_text(
            "差旅报销的上限是每天 500 元。\n"
            "超过 500 元的部分需要总经理签字。\n"
            "发票必须在一个月内提交给财务。\n",
            encoding="utf-8",
        )

        mem = _memory(path=str(d / "semantic.sqlite3"))
        n = mem.ingest_file(str(doc))
        assert n >= 1, "文档没进库"
        before = mem.count()

        llm = FakeLLM(["好的。"])
        agent = SimpleAgent("test", llm, system_prompt="你是助手",
                            hooks=[MemoryHook(mem)])
        agent.run("出差一天最多能报多少钱")

        prompt = llm.calls[0]["messages"][-1]["content"]
        assert "500" in prompt, f"检索到的内容没进 prompt:\n{prompt[:300]}"
        assert "报销制度.md" in prompt, "来源没带上, 模型没法引用"
        assert llm.calls[0]["messages"][-1]["content"].endswith("出差一天最多能报多少钱")

        # 铁律: 模型自己的回答不许进库(否则幻觉会被自己的记忆反复加固)
        assert mem.count() == before, "agent 把模型自己的回答写进记忆了"
        mem.close()


def test_persistence_across_reopen_with_real_vectors():
    """
    真实维度(可能 768/1024/1536)在 sqlite 里存一遍再读出来 —— BLOB 走的是
    float32, 维度大时更该确认没被截断/对齐错。
    """
    with TempDir() as d:
        path = str(d / "semantic.sqlite3")

        mem = SemanticMemory(embedder=_client(), path=path)
        mem.remember("服务器重启之前, 记得先把流量从负载均衡上摘下来。")
        mem.close()

        mem2 = SemanticMemory(embedder=_client(), path=path)
        got = mem2.search("重启服务器要注意什么")
        assert got, "重开之后搜不到了"
        assert "负载均衡" in got[0].text, got[0].text
        assert got[0].score > 0.5, f"重开之后分数异常({got[0].score:.3f}) —— 向量可能存坏了"
        mem2.close()


if __name__ == "__main__":
    probe()

    print()
    if CLIENT is None:
        print("=" * 66)
        print("  联网测试: 整体跳过")
        print("=" * 66)
        print(UNAVAILABLE)
        sys.exit(0)

    print("=" * 66)
    print(f"  联网测试   provider={CLIENT.provider}  model={CLIENT.model}  dim={CLIENT.dim}")
    print("  (LLM 那一侧用的是本地假实现, 不产生对话费用)")
    print("=" * 66)

    # isolate_env=False: 这个文件**就是要**吃 .env 里的真实 key(离线测试才需要隔离)。
    # 而且上面 probe() 已经拿它建好客户端了, 再清环境只会让"跑不起来"变得更难查。
    sys.exit(run_tests(globals(), "联网(真实 embedding)", isolate_env=False))
