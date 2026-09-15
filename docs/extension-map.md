# 扩展点地图

> 框架的文档不是「怎么用」，是「**在哪里挂东西**」。
> 这是框架文档和应用文档的分界线 —— 怎么用看 README，挂在哪看这里。

**唯一判据**：

> 加一个新能力，应该只需要写一个新文件 + 挂上去，**不需要改框架里的任何文件**。

---

## 一、横切能力：写成 Hook

`Hook` 有五个阶段，按一次 `run()` 的时间顺序。**只覆盖你关心的那几个方法** ——
基类里五个都是直接返回入参的 no-op。

| 我想加的东西 | 挂在哪 | 写几个文件 | 参考 |
|---|---|---|---|
| 记忆注入 | `Hook.before_llm` | 1 | [agents0to1/hooks/memory.py](../agents0to1/hooks/memory.py) |
| 世界感知 | `Hook.before_input` | 1 | — |
| 对话落库 | `Hook.after_run` | 1 | `EpisodicMemory.save_agent` |
| 成本统计 | `Hook.after_llm` + `LLMResponse.usage` | 1 | — |
| 成本熔断 | `Hook.after_llm` + `on_error="closed"` | 1 | — |
| 工具结果脱敏 | `Hook.after_tool` | 1 | — |
| 内容审核 / 改写 | `Hook.after_llm` | 1 | ⚠️ 见下面的"流式不对称" |
| **记忆流打分（小镇）** | `Hook.before_llm` + 自己的 scorer | 1 | 见第二节 |

**每个 hook 自己声明失败策略**：

- `on_error = "open"` —— 吞掉、记一条 warning、继续跑。记忆 / trace / 埋点用这个
- `on_error = "closed"` —— 抛出去、中断本轮。成本熔断 / 权限校验 / 内容合规用这个

框架不替你决定 —— 「记忆挂了」和「用户欠费了」形状一样，语义相反。

```python
from agents0to1 import Hook

class TraceHook(Hook):                      # 一个文件, 框架零改动
    on_error = "open"
    def after_llm(self, ctx, response):
        logger.info("第 %d 次调用, 用量 %s", ctx.llm_calls, response.usage)
        return response

agent = SimpleAgent("a", llm, hooks=[MemoryHook(mem), TraceHook()])
```

### 两条必须知道的不对称

1. **流式路径的 `after_llm` 只能观察，不能改。** chunk 早就逐块吐给调用方了，
   改 `self._last_response` 对已经吐出去的字符没有任何影响。
   做统计 ✅ / 做内容审核 ❌（只有非流式路径有效）
2. **`_chat` 和 `_stream_chat` 是两条独立路径。** 任何"所有 LLM 调用都要做的事"
   必须在两处都挂 —— 这是 hook 设计的固有代价。框架已经挂好了，你只要用阶段就行。

### `RunContext` 里能拿到什么

| 字段 | 含义 |
|---|---|
| `ctx.input_text` | 本轮用户**原始**输入，永远是原值 —— 拿它做检索 query，不要用 `messages[-1]` |
| `ctx.turn_index` | 本轮是第几轮 |
| `ctx.llm_calls` | 本轮已发起的 LLM 调用数（`before_llm` 看到的是"这次是第几次"，从 0 开始）|
| `ctx.agent` | 发起这次 run 的 agent |
| `ctx.state` | 自由槽位，hook 之间传临时状态用，框架不碰 |

> ⚠️ `llm_calls > 0` 这个判据对 PlanSolve / Reflection **不成立**（它们一轮里有 N 次"第一次"）。
> 要判断"这一轮注入过没有"，用 `ctx.state`。

---

## 二、记忆 / 检索：传参，不写文件

这一层全是**零文件**的开关 —— 换后端、换打分、换过滤，都是构造参数。

| 我想做的事 | 怎么写 | 写几个文件 |
|---|---|---|
| 换向量库（faiss / chroma） | `SemanticMemory(store=你的实现)` | 0 |
| 换检索打分 | `SemanticMemory(scorer=...)` / `VectorStore.search(scorer=...)` | 0 |
| 元数据过滤 | `search(filters={"source": "手册.md"})` | 0 |
| 增量入库 | `SemanticMemory.reindex_file(path)` | 0 |
| 遗忘一批 | `SemanticMemory.forget({"source": ...})` | 0 |
| 按 id / 条件删 | `VectorStore.delete(ids=...)` / `delete(filters=...)` | 0 |
| 文档解析（pdf / docx） | 自己解析完再 `SemanticMemory.remember(text)` | 1 |

### 打分策略：文档系统 vs 小镇

**这两个要的打分是冲突的，所以框架不替你选。**

| | 要的打分 |
|---|---|
| 文档型 agent 系统 | **relevance-only** —— 一份 2023 年的手册不该因为"旧"就沉下去 |
| 小镇（记忆流） | **recency + importance + relevance** —— 三周前的闲聊该沉下去 |

默认是纯余弦。小镇传一个自己的进来（**写在应用层，不进框架**）：

```python
def memory_stream_scorer(query, vector, *, query_norm, vector_norm,
                         created_at, metadata, **kw):
    relevance  = cosine_similarity(query, vector, query_norm, vector_norm)
    recency    = 0.99 ** ((time.time() - created_at) / 3600)     # 指数衰减
    importance = metadata.get("importance", 5) / 10              # LLM 打的分存 metadata
    return 1.0 * recency + 1.0 * importance + 1.5 * relevance

mem = SemanticMemory(scorer=memory_stream_scorer)
```

> ⚠️ 换了 scorer 就要跟着调 `min_score` —— 它比的是 **scorer 的输出**，
> 默认 `0.0` 是按余弦的量纲定的。

**自定义 store 的唯一要求**：有 `add` / `search` / `count` / `clear` / `close` 五个方法，
不必继承 `VectorStore`。`search()` 返回 `[(分数, 文本, metadata), ...]`。

---

## 三、Agent 与工具

| 我想加的东西 | 挂在哪 | 写几个文件 |
|---|---|---|
| 加一种新 Agent 范式 | 继承 `Agent`，实现 `_run()` | 1 |
| 加一个工具 | 继承 `Tool`，`registry.register_tool(...)` | 1 |
| 换 `stream_run` 的粒度 | 覆盖 `_stream_run()` | 0（在子类里）|

**只实现 `_run()`，事件流是白送的** —— 基类的 `stream_run()` 会把结果包成一个 `final` 事件。
想做精细的（工具卡片）再覆盖 `_stream_run()`。

> ⚠️ `_run()` 里要消费事件流时调 **`_stream_run()`**，不要调 `stream_run()` ——
> 后者是模板方法，会重开 ctx 并且无限递归。

### ReAct 的 `text` 为什么不是"边收边吐"

`ReActAgent._stream_run` 会先把这一轮的内容**攒着**，等流结束、知道这轮要不要调工具了，
再把它们吐出去：不调工具 → 逐个作为 `text`；调工具 → 整段作为一个 `thinking`。

因为「这句话是最终答案还是过程叙述」**在流结束前无法判断**（判据
`response.has_tool_calls` 只在整个流被消费完才有值）。边收边吐的后果是那段
"我先查一下…" 会**既出现在 `text` 里又出现在 `thinking` 里**，把 `text` 拼起来当答案的
调用方就中招了。

**代价**：正文不再逐 token 实时到达，而是这一轮流完之后才开始。事件的**形状**没变
（还是按原块边界逐个吐），变的只是时间点。这是把两者分对必须付的钱。

> 事件流的完整契约（谁是谁、谁能拼、谁不能拼）都在 [core/typedefs.py](../agents0to1/core/typedefs.py)
> 里 `AgentEvent` 上面那段。守着它的是
> [tests/test_contracts.py](../tests/test_contracts.py) 里的 `_check_event_contract`。

---

## 四、为什么不写实现

解析器、父子块、混合检索、rerank、多 Agent 编排、小镇的 tick / 世界 / 地点 / 日程 ——
**这些全是应用层。**

框架只保证「挂得上」，不保证「替你挂」。准则：

- 框架给「在哪里挂」，不给「挂什么」
- 硬依赖仍然只有 `openai` + `pydantic`；扩展现有的一切都只用 stdlib
- 每一条缝的验收方式都一样：**换一个实现传进去，框架文件零改动**

---

## 五、这些承诺由什么守着

| 承诺 | 守它的测试 |
|---|---|
| hook 挂了不改变行为 / 两种失败策略 | [tests/test_hooks.py](../tests/test_hooks.py) |
| 任何 memory / hook / store / scorer / Agent 子类都满足契约 | [tests/test_contracts.py](../tests/test_contracts.py) |
| usage 在两条路径上都有值 | [tests/test_usage.py](../tests/test_usage.py) |
| 打分 / 过滤 / 删除 | [tests/test_vector_store.py](../tests/test_vector_store.py)、[tests/test_semantic.py](../tests/test_semantic.py) |

```bash
conda activate agent
python tests/run_offline.py      # 全绿 = 上面每一条都还成立
```
