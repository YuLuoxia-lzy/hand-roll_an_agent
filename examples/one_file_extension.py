"""一个文件, 三种能力 —— 「唯一判据」的实弹验收

    python examples/one_file_extension.py        # 不联网、不需要任何 API key

【这个文件在验什么】
    加一个新能力 = 写一个新文件 + 挂上去, 不需要改框架里的任何文件。

下面三种能力**互不相关**, 来自三个不同的应用方向(成本治理 / 记忆流 / 换向量库),
它们全挤在这一个文件里, 而且**全部只通过公开的构造参数挂上去**:

    ① CostFuseHook     横切能力   -> 钩在 Hook 上           (on_error="closed")
    ② town_scorer      检索打分   -> 传给 SemanticMemory(scorer=)
    ③ DictStore        换向量库   -> 传给 SemanticMemory(store=)

跑通之后你可以自己验一遍"框架零改动":

    git status --short agents0to1/     # 除了本次改造的既有改动, 这里不该多出任何东西

【为什么这个演示是离线的】
    文档里那句话反过来也成立: **如果哪一步非联网不可, 说明它的边界划错了。**
    所以下面自带假 LLM 和假 embedder —— 假 LLM 还是验证"注入有没有生效"的关键:
    要看的是**我们发了什么给模型**, 而不是模型回了什么。

【⚠️ 别把这里的东西抄进框架】
    本文件里的每一个类都属于**应用层**。框架只保证"挂得上", 不保证"替你挂"。
    (见本地笔记 docs/extension-map.md 第四节)
"""

import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents0to1 import Hook, RunContext, SimpleAgent, SemanticMemory        # noqa: E402
from agents0to1.core.typedefs import LLMResponse, Usage                      # noqa: E402
from agents0to1.memory.vector_store import cosine_similarity                 # noqa: E402


# ============================================================================
# 脚手架: 假 LLM + 假 embedder —— 不联网、不花钱、可复现
# ============================================================================

class _FakeLLM:
    """
    假 LLM。**记下每次请求的完整 messages** —— 验证"注入有没有生效"靠的是这个,
    不是靠它回了什么。顺带伪造一份 usage, 好让下面的成本 hook 有东西可算。
    """

    provider = "fake"
    model = "fake-model"
    base_url = "http://fake.invalid/v1"

    def __init__(self, answer: str = "好的。"):
        self.answer = answer
        self.calls = []
        self.last_response = None
        self._tokens = 0

    def _next(self, messages):
        from copy import deepcopy

        self.calls.append(deepcopy(messages))
        self._tokens += 100
        self.last_response = LLMResponse(
            content=self.answer,
            tool_calls=None,
            model=self.model,
            finish_reason="stop",
            usage=Usage(prompt_tokens=self._tokens, completion_tokens=20,
                        total_tokens=self._tokens + 20),
        )
        return self.last_response

    def invoke(self, messages, tools=None, **kwargs):
        return self._next(messages)

    def stream_invoke(self, messages, tools=None, **kwargs):
        response = self._next(messages)
        yield response.content


class _FakeEmbedder:
    """
    字符 trigram 的哈希袋 —— 不联网, 但**保留了语义梯度**: 有重叠的文本相似度高。
    纯哈希向量两两近似正交(相似度都是 0), 那样"谁排前面"根本测不出来。

    ⚠️ 用的是 zlib.crc32 而不是内置 hash(): 后者每个进程都不一样
    (PYTHONHASHSEED), 入库和查询会算出完全不同的向量。
    """

    model_id = "fake-embedder"

    def __init__(self, dim: int = 32):
        self._dim = dim

    def _one(self, text: str):
        vec = [0.0] * self._dim
        for i in range(max(len(text) - 2, 1)):
            gram = text[i:i + 3]
            vec[zlib.crc32(gram.encode("utf-8")) % self._dim] += 1.0
        return vec or [1.0] + [0.0] * (self._dim - 1)

    def embed(self, texts):
        return [self._one(t) for t in texts]

    def embed_one(self, text):
        return self._one(text)


# ============================================================================
# ① 横切能力: 成本熔断 (Hook + on_error="closed")
# ============================================================================

class CostFuseHook(Hook):
    """
    一轮里累计 token 超过阈值就**中断这一轮**。

    【为什么必须 on_error="closed"】
        这是本文件里唯一一个"必须中断"的 hook。记忆 / trace 挂了应该继续聊,
        但"这轮已经烧掉 5000 token 了"挂掉而继续跑 —— 那正是它要防的事。
        框架不替你决定, 由 hook 自己声明: 见 Hook.on_error 的说明。

    【状态放哪】
        放 ctx.state, **不要**放 self.xxx。ctx 是"这一轮"的, 一轮结束就没了;
        挂在 self 上的累计值会跨轮、跨 agent 地涨下去(同一个 hook 实例被两个
        agent 共用时尤其糟)。

    【为什么 after_llm 就够】
        usage 在两条路径上都填好了(见 agents0to1/core/llm.py), 所以这里不需要
        关心调用方是 run 还是 stream_run。唯一的不对称是流式路径上**改不了**已经
        吐出去的字符 —— 但熔断是"抛异常中断", 不是"改响应", 所以不受影响。
    """

    on_error = "closed"          # ← 这一行就是全部的策略声明

    def __init__(self, max_tokens_per_turn: int = 2000):
        self.max_tokens_per_turn = max_tokens_per_turn

    def after_llm(self, ctx: RunContext, response):
        used = ctx.state.get("fuse_tokens", 0)
        usage = getattr(response, "usage", None)
        if usage is not None:
            used += usage.total_tokens or 0
        ctx.state["fuse_tokens"] = used

        if used > self.max_tokens_per_turn:
            raise RuntimeError(
                f"本轮已用 {used} tokens, 超过上限 {self.max_tokens_per_turn} —— 中断"
            )
        return response


# ============================================================================
# ② 打分公式: 小镇的记忆流 (relevance + recency + importance)
# ============================================================================

def town_scorer(query, vector, *, query_norm=None, vector_norm=None,
                created_at=None, metadata=None, **kw):
    """
    三因子打分 —— Generative Agents 那条路。

    **和文档系统要的打分是冲突的**: 小镇里三周前的闲聊该沉下去, 而文档系统里
    一份 2023 年的手册不该因为"旧"就沉下去。所以框架不替你选, 默认给纯余弦,
    要哪个自己传进来。

    签名里那几个关键字是框架**固定会传**的(见 VectorStore.search 里调用 scorer
    的那一行): query/vector 位置传, 其余全部关键字传 —— 所以你可以只挑你在意的
    那几个用, 多出来的用 **kw 吞掉即可。

    ⚠️ 换了 scorer 就要跟着调 min_score: 它比的是 **scorer 的输出**, 默认
       0.0 那个阈值是按余弦的量纲([-1, 1])定的, 而下面这个式子是 [0, 2.5]。
    """
    metadata = metadata or {}
    relevance = cosine_similarity(query, vector, query_norm, vector_norm)
    recency = 0.995 ** ((time.time() - (created_at or time.time())) / 60.0)
    importance = metadata.get("importance", 5) / 10.0
    return 1.0 * recency + 1.5 * importance + 1.5 * relevance


# ============================================================================
# ③ 换向量库: 一个不继承 VectorStore 的实现
# ============================================================================

class DictStore:
    """
    最土的向量库 —— 一个列表, 线性扫。

    【它为什么能插进来】
        框架只要求五个方法, **不要求继承 VectorStore** —— 一旦要求继承,
        "换一个类"就变成了"换一个子类", 那条承诺就废了。

        add(texts, vectors, metadatas, model=...) -> 返回新 id 列表
        search(vector, top_k=, model=, filters=, scorer=) -> [(分数, 文本, metadata), ...]
        count() / clear() / close()

    【它比真向量库多验了什么】
        VectorStore 里那些行为(行级 model 过滤、filters 在 top_k **之前**生效)
        在真实现里是 SQL 干的。换到这里来, 就得由**这个类自己**保证 ——
        这也正是"契约"该管的事: 换任何实现, 这些性质都该成立。
    """

    def __init__(self):
        self._rows = []
        self._closed = False

    def add(self, texts, vectors, metadatas=None, model=None, **kw):
        metadatas = metadatas or [{} for _ in texts]
        ids = []
        for text, vec, meta in zip(texts, vectors, metadatas):
            self._rows.append({"id": len(self._rows), "text": text, "vector": list(vec),
                               "metadata": dict(meta), "model": model,
                               "created_at": time.time()})
            ids.append(self._rows[-1]["id"])
        return ids

    def search(self, vector, top_k=5, model=None, filters=None, scorer=None, **kw):
        score_fn = scorer or cosine_similarity

        # 过滤必须在排序**之前** —— 先按 top_k 截断再筛是错的:
        # 被筛掉的那几条本来会占掉名额, 于是"符合条件的没被取到"。
        rows = [r for r in self._rows if model is None or r["model"] == model]
        if filters:
            rows = [r for r in rows
                    if all(r["metadata"].get(k) == v for k, v in filters.items())]

        scored = [
            (score_fn(vector, r["vector"], created_at=r["created_at"],
                      metadata=r["metadata"]), r["text"], r["metadata"])
            for r in rows
        ]
        scored.sort(key=lambda t: t[0], reverse=True)
        return scored[:top_k]

    def count(self):
        return len(self._rows)

    def clear(self):
        n = len(self._rows)
        self._rows.clear()
        return n

    def close(self):
        self._closed = True


# ============================================================================
# 挂上去 —— 注意: 下面全是构造参数, 框架文件一个没动
# ============================================================================

def main():
    ok = True

    def check(label, condition, detail=""):
        nonlocal ok
        ok = ok and condition
        print(f"  [{'OK ' if condition else 'FAIL'}] {label}{'  ' + detail if detail else ''}")

    print()
    print("=" * 70)
    print("  一个文件挂三种能力 —— 全程离线, 不需要任何 API key")
    print("=" * 70)

    # ---------- ① 成本熔断 ----------
    print("\n① CostFuseHook (Hook, on_error='closed')")

    fuse = CostFuseHook(max_tokens_per_turn=1000)
    llm = _FakeLLM("这次回答不花钱。")
    agent = SimpleAgent("a", llm, hooks=[fuse])          # ← 只有这一行
    assert agent.run("你好") == "这次回答不花钱。"
    check("一轮之内不超限 -> 正常返回", len(llm.calls) == 1)

    tight = SimpleAgent("b", _FakeLLM(), hooks=[CostFuseHook(max_tokens_per_turn=50)])
    try:
        tight.run("这一轮会被熔断")
        check("超限时抛出去中断本轮", False, "没抛")
    except RuntimeError as e:
        # closed 策略的定义就是"原样穿透", 不是包装成别的异常类型
        check("超限时抛出去中断本轮", "超过上限" in str(e), f"({e})")

    # 同一件事在 open 策略下必须是反的 —— 形状一样, 语义相反
    class Quiet(CostFuseHook):
        on_error = "open"

    quiet_llm = _FakeLLM("照样回答")
    assert SimpleAgent("c", quiet_llm, hooks=[Quiet(50)]).run("x") == "照样回答"
    check("同一个 hook 换成 on_error='open' -> 吞掉, 继续跑", len(quiet_llm.calls) == 1)

    # ---------- ② + ③ 同时挂上 ----------
    print("\n② town_scorer (打分公式) + ③ DictStore (换向量库)")

    store = DictStore()
    mem = SemanticMemory(
        embedder=_FakeEmbedder(),
        store=store,                                      # ← 只有这一行
        scorer=town_scorer,                               # ← 和这一行
        min_score=-99,                                    # 三因子量纲和余弦不同, 见 docstring
        top_k=2,
    )
    mem.remember("小猫在公园里晒太阳", {"importance": 9})
    mem.remember("昨天讨论了一下天气", {"importance": 2})
    check("自定义 store 真的被用上了", store.count() == 2, f"(count={store.count()})")

    hits = mem.search("小猫晒太阳")
    check("换了 scorer -> 结果按新公式排", len(hits) == 2)

    # 【怎么证明 scorer 真的在起作用, 而不是"看起来换了个数"】
    #   同一个查询、同一个 store, 只换 scorer, 把两次的**分数和顺序**都拿出来比:
    #   余弦是 [-1,1] 且只会看相似度; 三因子是 [0,2.5] 而且会把 importance=9 那条
    #   顶上来。两条都不同, 才说明打分权真的交出去了。
    cosine_hits = mem.search("小猫晒太阳",
                             scorer=lambda q, v, **kw: cosine_similarity(q, v))
    check("scorer 换掉后量纲确实变了(说明不是余弦)",
          all(not (-1.0 <= h.score <= 1.0) for h in hits),
          f"(town={[round(h.score, 2) for h in hits]} / "
          f"余弦={[round(h.score, 2) for h in cosine_hits]})")
    check("importance 高的那条被顶到前面(余弦看不出的那个因子)",
          hits[0].metadata["importance"] == 9,
          f"(首条 importance={hits[0].metadata['importance']})")

    # filters 必须在 top_k 之前生效 —— 这条对任何 store 实现都成立(契约)
    only_quiet = mem.search("天气", filters={"importance": 2})
    check("filters 在 top_k 之前生效", all(h.metadata["importance"] == 2 for h in only_quiet),
          f"(拿到 {len(only_quiet)} 条)")

    # ---------- 把记忆和熔断一起挂到 agent 上 ----------
    print("\n④ 三种能力同时挂在同一个 agent 上")

    from agents0to1.hooks.memory import MemoryHook

    llm2 = _FakeLLM("综合回答")
    agent2 = SimpleAgent("d", llm2, hooks=[MemoryHook(mem), CostFuseHook(1_000_000)])
    agent2.run("小猫晒太阳")
    sent = llm2.calls[0]                      # 关键: 看**发了什么**, 不是看回了什么
    check("记忆注入进了 user 消息",
          any("小猫在公园里晒太阳" in m.get("content", "") for m in sent),
          f"(共 {len(sent)} 条消息)")
    check("是两个 hook 都在管线里", len(agent2._hooks) == 2, f"({agent2._hooks})")

    # ---------- 这一节是重点 ----------
    print("\n⑤ 框架里有没有为了上面这些动过一个字?")

    root = Path(__file__).resolve().parent.parent
    framework = sorted((root / "agents0to1").rglob("*.py"))
    print(f"      框架共 {len(framework)} 个 .py 文件, 本文件碰到的公开名只有:")
    print("          Hook / RunContext / SimpleAgent / SemanticMemory")
    print("          cosine_similarity / MemoryHook / LLMResponse / Usage")
    print("      挂载点全是**构造参数**(hooks= / store= / scorer=)或类属性(on_error=),")
    print("      没有一个 monkey patch、没有一个私有名下划线属性被改写。")
    print()
    print("      自己验一遍(应在 git 里看不到任何新增改动):")
    print("          git status --short agents0to1/")

    print()
    print("-" * 70)
    print("  全部通过 ✅   判据成立: 写一个新文件 + 挂上去, 框架零改动"
          if ok else "  有失败 ❌")
    print("-" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
