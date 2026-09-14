# 记忆层改造指南

> 目标：把「一个能对话、会调工具的循环」变成「一个有记忆的 agent」。
> RAG 是这一层里的一个组件，不是一件并列的事。
>
> 本文按**依赖顺序**编排，每一步都能单独跑通、单独验证。任何一步停下来，你手里都有一个能用的东西。

---

## 零、边界与约束

### 本次做什么

| 阶段 | 产物 | 是否依赖 API key |
|---|---|---|
| 一 | 修两个前置地雷 | 否 |
| 二 | Embedding 接口 | **是** |
| 三 | 手搓向量库 | 否 |
| 四 | 切分 + 语义记忆 | 否（用假 embedder） |
| 五 | 接进 Agent 循环 | **是** |
| 六 | 情景记忆持久化（可延后） | 否 |
| 七 | 评测基线 | 大部分否 |

### 本次**不**做什么

- 多 Agent 协作 / 赛博小镇 —— 前置是记忆层 + 异步，不在这份文档里
- MCP 客户端
- 文档格式解析（pdf/docx）—— 先只用纯文本和 markdown

### 两条硬约束

1. **硬依赖仍然只有 `openai` + `pydantic`**。这是这个项目最值钱的一条洁癖（`requirements.txt` 里写得明明白白）。所以：sqlite 用 stdlib、向量运算手搓、numpy 做成可选加速。
2. **Python 3.10 兼容**。`requires-python = ">=3.10"`。`X | Y`、`list[MemoryItem]` 这类新写法都可以用。
   ~~3.9~~ 已经不守了 —— 为什么，见第一步 1.2 的更正说明。

---

## 第一步：先修两个前置地雷

> ✅ **两条都已处理（2026-09-14）**：1.1 删掉了重复 append；1.2 **反着做的** —— 代码保持
> `str | Path` 不动，把 `requires-python` 提到 `>=3.10`（理由见 1.2 末尾的更正）。
> 下面的分析原样保留 —— 它记的是"为什么这两条必须最先修"，比结论本身值钱。

### 为什么必须最先做

不是"顺手修 bug"，是这两条都会**直接放大后面每一步的成本**：

- 第一条让每轮多烧一份问题份量的 token，你后面每加一层，浪费都跟着翻倍
- 第二条让框架在你本机上根本 import 不了，后面什么都跑不起来

两条加起来不到 5 行改动，但它决定了后面每一步是否可定位。

### 1.1 ReAct 重复发送用户问题

`agents0to1/classic_agent/react_agent.py:128-130`：

```python
messages = self._build_messages(input_text)   # 末尾已经拼了 [..., {"role":"user","content":input_text}]
turn_start = len(messages) - 1                # 指向那条 user
messages.append({"role": "user", "content": input_text})   # 又加了一条一模一样的
```

`agents0to1/core/agent.py:80-88` 的 `_build_messages` 返回值末尾**已经**含当前问题。这里又 append 一次。

**后果链（比"发两遍"严重）**：

1. 模型每轮收到两遍同样的问题
2. `turn_start` 指向**第一条** user，所以 `messages[turn_start:]` = `[user, user, assistant, ...]` → **这份重复被存进了 Turn**
3. `Turn.is_valid()`（`agents0to1/core/message.py:62-92`）只检查工具调用配对，不检查重复 → **不会报警**
4. `_history_messages()` 之后每轮都把重复的问题再发一遍 → 一直累积
5. `_turn_chars()`（`agents0to1/core/agent.py:161-173`）重复计数 → 白白吃掉 `max_history_chars` 的预算
6. 在 vLLM + Llama-2 模板上直接就是坏的（连续两条 user，模板不接受）

对照：`SimpleAgent`（`agents0to1/classic_agent/simple_agent.py:33/51`）**没有**这个 append。所以两个 Agent 行为不一致。

**改法**：删掉第 130 行。

**为什么删掉是安全的**：`turn_start` 在 append **之前**就算好了，而且是对着一个已经含 user 消息的列表算的。删掉后 `messages[turn_start:]` = `[user, assistant, ...]`，切片起点不变，没有任何东西读那个被 append 的元素。

### 1.2 `str | Path` 在 3.9 上 import 期就炸

`agents0to1/core/agent.py` 有三处：`save`（:313）、`load`（:371）、`fork`（:383），签名都是 `path: str | Path`。

PEP 604 的 `X | Y` 是 **Python 3.10+**。函数注解在 `def` 执行时求值，所以 3.9 上会在**导入模块时**抛：

```
TypeError: unsupported operand type(s) for |: 'type' and 'type'
```

本机实测确认：默认解释器 `D:\Anaconda\python.exe` 是 **3.9.13**，确实抛这个错。也就是说 `requires-python = ">=3.9"` 今天已经不成立了。

**改法**：`Union[str, Path]`，并从 `typing` 导入 `Union`。这是项目已有的书写习惯——`agents0to1/utils/serialization.py:44/53` 和 `agents0to1/utils/logging.py:22/37` 都是这么写的，跟着来就行。

> ⚠️ **以上的"改法"已作废（2026-09-14）**。
>
> 实测本身没错，错的是前提：3.9 是 `D:\Anaconda\python.exe` 这个**默认**解释器的版本，
> 不是这个项目实际跑的版本。项目跑在 conda 的 `agent` 环境里，是 **3.10.21**，
> `str | Path` 在那儿本来就没问题——那三处注解是作者自己写的，不是笔误。
>
> 最终改成：`requires-python = ">=3.10"`，代码保持 `str | Path` 原样。
>
> 教训：兼容性下限要按**实际运行环境**定，不是按机器上碰巧是默认的那个解释器，
> 更不是按 `pyproject.toml` 里一行没人维护的声明。`utils/` 里那几处 `Union`
> 是作者混着写的，不代表项目要守 3.9。

> 新代码里同理：`list[MemoryItem]` 这种能用（PEP 585），`X | Y` 也能用。

### 验证

```python
import core.agent          # 能 import 就说明 1.2 修好了
```

跑一轮 ReAct，在 `_stream_chat` 前打印 `len(messages)` 和 `messages[-1]`，确认末尾只有一条 user。

---

## 第二步：Embedding 接口

### 为什么不能复用 chat 的 provider 解析

`agents0to1/core/llm.py` 里那套 `_resolve_credentials` 是按 provider 给 key + url 的。看起来"加一个 embeddings 方法、复用同一套解析"很省事——**但会踩坑**：

**DeepSeek 没有 embeddings 端点。**

而 DeepSeek 恰恰是最容易被自动检测到的那个——`DEEPSEEK_API_KEY` 一设上，`_ENV_PROVIDER_KEYS` 就会选中它。如果你图省事复用了 `_ENV_PROVIDER_KEYS`（`agents0to1/core/llm.py:21-30`），DeepSeek 会被正常检测出来，然后你拿着 chat 的 key 和 base_url 去请求一个**根本不存在的端点**——报一个你很难看懂的 404。

所以：embedding 必须有**自己独立的一组配置**。

```
EMBEDDING_PROVIDER   # openai / dashscope / zhipu / ollama / modelscope
EMBEDDING_MODEL      # text-embedding-3-small / text-embedding-v3 / nomic-embed-text ...
EMBEDDING_BASE_URL   # 可选
EMBEDDING_API_KEY    # 可选, 没有就按 provider 找对应的 key
```

**这里的教学点**：`agents0to1/core/llm.py` 里那张 provider 表是**给 chat 用的**，不是一张通用的表。不同的能力有不同的 provider 覆盖范围。这种"看起来能复用、其实语义不同"的地方，是架构里最容易埋雷的。

### 接口设计

放在 `agents0to1/memory/embedding.py`，**不要**塞进 `agents0to1/core/llm.py`。理由：chat 客户端和 embedding 客户端没有共享状态，混在一起只会让 `Agents0to1` 越来越胖。

```python
class EmbeddingClient:
    def __init__(self, model=None, api_key=None, base_url=None,
                 provider=None, dimensions=None, batch_size=32, timeout=30): ...

    def embed(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入。内部按 batch_size 切片, 一次 HTTP 多发几条 —— 不要 for 循环一条条发。"""

    def embed_one(self, text: str) -> List[float]:
        """单条。内部就是 embed([text])[0]。"""

    @property
    def dim(self) -> int:
        """向量维度。第一次调用后才知道, 拿到就缓存住。"""
```

三个要点：

1. **必须批量**。一次请求发 32 条 vs 发 32 次请求，差几十倍。这是 embedding 和 chat 最大的调用习惯差异——chat 你习惯一次一问，embedding 必须攒批。
2. **`dim` 要缓存**。不同模型维度不同（768 / 1024 / 1536 / 3072）。第一次调用后记下来，之后用它做校验。有些 API（OpenAI 的 `text-embedding-3-*`）支持传 `dimensions` 参数截断维度，也可以走这条路。
3. **这里该抛异常**。`EmbeddingClient` 只是个客户端，失败就抛。需要 fail-open 的是**调用它的记忆层**（见第五步 5.3）。这个分层要分清楚，不然异常会被吞在错误的地方。

### 验证

```python
c = EmbeddingClient()
vs = c.embed(["你好", "世界"])
print(len(vs), len(vs[0]))     # 2 <dim>
print(c.dim)                   # 缓存生效
```

---

## 第三步：手搓向量库

### 为什么手搓，而不是直接上 chroma / faiss

三个理由，按重要性排：

1. **硬依赖约束**。这个项目最值钱的就是"硬依赖只有两个"。装一个 chroma 会拖进来几十个包。
2. **手搓过才知道向量库在干什么**。剥开看，核心就是一个余弦相似度：

   ```python
   cos(q, v) = dot(q, v) / (|q| * |v|)
   ```

   就这么简单。理解了这一层，之后看任何向量库的文档都只是"它怎么把这件事做快"。
3. **换起来才便宜**。接口定好了，之后换 faiss / sqlite-vec / chroma 就是换一个类。

### 三个设计点

**① 预存模长**

`|v|` 在**入库时**算一次存下来，查询时只算 query 的模长。省掉每次查询 n 次开方。

```python
def _cosine(query, vec, q_norm, v_norm) -> float:
    return sum(a * b for a, b in zip(query, vec)) / (q_norm * v_norm)
```

**② 向量存储格式**

`array.array("f").tobytes()` 存进 sqlite 的 BLOB，配 `sqlite3.Binary`。比 JSON 文本快很多，而且是 stdlib。

> 学习阶段可以**先用 JSON 文本**（好处是能用眼睛看，调试友好），等觉得慢了再换成 BLOB。**这个转换本身就是一个很好的练习**——你会亲眼看到同一份数据换了存储格式之后快了多少。

**③ 模型和维度必须存进每一行**

不同模型产生的向量**不可比**。如果库里混了两个模型的向量，余弦相似度算出来的是纯粹的噪声，而它**不会报错**——只会给你错误的结果。

所以每行存 `model` 和 `dim`，跨模型搜索要**直接拒绝**，而不是算出一个垃圾分数。

### numpy 可选加速

```python
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False
```

**不要写两套实现**——写一个 `_dot(a, b)`，里面分叉。这样只有一处分叉，其余逻辑两条路共用，不会漂移。

性能量级参考：10k 条 × 1024 维，纯 Python 每次查询约 1 秒级；numpy 约 1 毫秒级。**几千条以内裸奔完全没问题**，到几万条再考虑索引（HNSW / IVF）。别过早优化。

### sqlite 的线程问题（这个必踩）

`agents0to1/tools/async_executor.py:88-89` 用 `ThreadPoolExecutor` 跑工具。一轮里模型完全可能同时发两个 `knowledge_search`（这合法，见 `agents0to1/classic_agent/react_agent.py:164-168`）。

sqlite 连接默认 `check_same_thread=True` → **第二个线程一进来就 `ProgrammingError`**。

**解法**（二选一，推荐第一个）：

- `check_same_thread=False` + 一个 `threading.Lock` 包住所有读写
- 每次调用开一个新连接（sqlite 连接很便宜，但要开 WAL 模式）

**为什么这个坑值得单独讲**：它**只有并发调用两个工具时才会暴露**。你单线程测一百遍都测不出来，上线才炸。所以第七步的离线测试里，**必须专门有一个并发测试**。

### 验证（不需要 API key）

构造几个已知向量，手算余弦，对比：

```python
# 正交向量 -> 0
# 同向向量 -> 1
# 反向向量 -> -1
```

再测：入库 n 条 → 查询 → 确认最像的排最前。**这个测试完全不需要联网**，是你能随时跑的东西。

---

## 第四步：切分 + 语义记忆

### 为什么切分比向量库选型重要

向量库决定「**多快**」，切分决定「**能不能搜到**」。

切分错了，再好的向量库也搜不出正确答案——因为正确答案根本没被完整地切进任何一块里。这是 RAG 里最被低估的一环，也是最值得你花时间调的一环。

### 切分的基本要求

- **按句子边界切**，不要在句子中间断开。中文按 `。！？` 和换行切
- **有 overlap**（相邻块重叠一部分），否则答案正好落在边界上就丢了
- **带 metadata**（来源、位置），否则检索回来你不知道它从哪来，也没法引用

中文从 **400–800 字符 + 10–20% overlap** 起步，然后按实际效果调。

### 检索结果要有分数和来源

```python
class MemoryItem(BaseModel):
    text: str
    score: float          # 相似度, 必须暴露出来
    metadata: Dict[str, Any] = {}   # 来源、位置、时间
```

`score` 必须暴露，因为：

- 你要能**设阈值**——分数太低的不该喂给模型，那只会干扰它
- 你要能**调试**——"为什么这次没检索到"是 RAG 最高频的问题，没有分数你只能猜

### 写入路径必须显式设计

这是最容易漏的一块：**只设计了 `search()`，没人调用 `add()`，库里永远是空的。**

需要明确两个入口：

- `remember(text, metadata)` —— 手动存一条
- `ingest_file(path)` —— 读文件、切分、批量入库

**一条铁律：绝不自动把 assistant 自己的回答入库。**

否则会形成自我确认循环：模型编了一个说法 → 存进库 → 下次检索出来 → "这是检索到的事实" → 模型更相信自己编的东西。**幻觉会被自己的记忆反复加固。**

要存也是存**用户说的**和**外部文档**，不存模型自己生成的。

### 验证（不需要 key）

用一个假 embedder——把文本按字符哈希成一个固定长度的向量。这样：

- 相同文本 → 相同向量
- 不同文本 → 不同向量
- 完全不花钱、不联网

然后测：切分边界对不对、overlap 有没有生效、空输入会不会崩、`ingest_file` 之后 `search` 能不能搜到。

---

## 第五步：接进 Agent 循环

这一步是整个方案里最容易做错的地方。**分两小步走，顺序本身就是教学内容。**

### 5.1 先做工具版

`agents0to1/tools/builtin/knowledge.py` 里的 `KnowledgeSearchTool`，就是对 `SemanticMemory.search()` 的薄封装，走标准 `Tool` 接口。

一天就能跑通，`ReActAgent` 立刻能用。

**但它的真正价值不是"能用"，是让你亲身感受到工具版的三个局限**：

1. **只有 `ReActAgent` 有工具。** 另外三个 Agent 什么都得不到
2. **模型得自己想不起来去调。** 不调 = 静默失效，你从日志里都看不出问题
3. **一次工具往返 = 一次额外的 LLM 调用。** 而且如果模型没调，你还得想办法让它调

感受到这些，你才会真正理解为什么下一步要做上下文版。**先做工具版不是为了省事，是为了建立体感。**

> 顺带注意：`ToolRegistry.execute` 对结果做 `truncate_output`（`agents0to1/tools/registry.py:86`），默认上限 `max_output_chars = 6000`（`agents0to1/tools/base.py:20`）。检索回来好几个块很容易超。**在 `run()` 内部就按预算裁好**，别让外层截断逻辑去砍——`_TAIL_CHARS = 800` 会把最后一块的尾巴接上来，相关性顺序就乱了。
>
> 而且这样还能保证：工具路径和上下文路径**用同一个预算常量**，模型不会看到同一个库的两种形状。

### 5.2 上下文版：为什么不是一处搞定

三种 Agent 的行事逻辑不一样。**不要试图找一个统一的注入点。**

| Agent | `messages[-1]` 是什么 | 怎么处理 |
|---|---|---|
| `SimpleAgent` | 用户问题 ✅ | 基类钩子 |
| `ReActAgent` 第 1 步 | 用户问题 ✅ | 基类钩子 |
| `ReActAgent` 第 2 步起 | **工具结果** ❌ | 基类钩子（查询用 `input_text`，**不是** `messages[-1]`） |
| `PlanAndSolveAgent` | **指令模板** ❌ | 在 `run()` 里算一次，显式传进 prompt |
| `ReflectionAgent` | **模型自己的草稿** ❌ | 只喂初稿，**不喂评审** |

**关键点**：查询要用 `input_text` 参数本身，**绝对不要用 `messages[-1]["content"]`**。因为到了 ReAct 第 2 步，最后一条是 `tool` 消息（`agents0to1/classic_agent/react_agent.py:223-227` 追加的）。拿工具结果去检索，检索出来的东西和用户问题毫无关系。

#### 为什么挂在 `_build_messages` 而不是 `_chat`

一个**反例教学**（我自己先选错了）：

直觉上会选 `_chat`，理由是"所有 Agent 都经过它"。**但这个理由是错的**——`agents0to1/core/agent.py:106-110` 的 `_stream_chat` 是**直接调** `self.llm.stream_invoke`，**不经过 `_chat`**。

钩子挂在 `_chat` 上，**ReAct 主循环（`agents0to1/classic_agent/react_agent.py:133`）和 SimpleAgent 的流式路径（`agents0to1/classic_agent/simple_agent.py:54`）一份都吃不到**。而且它不会报错——只是静默地什么都不做。

而 `_build_messages` 反而**天然正确**：

- 它的返回值末尾才追加 user 消息，而 `turn_start`（`agents0to1/classic_agent/react_agent.py:129`）是在它返回**之后**才算的
- 所以插进去的内容**天然落在 `turn_start` 之前** → 不会被存进 Turn → 零拷贝
- 历史不膨胀、快照不膨胀、下一轮不会把检索结果当历史重发、`_turn_chars` 的预算语义不变

**额外的好处**：`last_messages` 的注释明说是"模型当时看到了什么"（`agents0to1/classic_agent/react_agent.py:73-74`）。插在 `_build_messages` 里，它和真实请求**一致**；插在 `_chat` 里做拷贝，它就对不上了——一个本来用于调试的字段，变成了误导。

> 这个反例值得记住：**"所有 X 都经过 Y" 这种判断，一定要去代码里数一遍，不能靠直觉。** 差别就在一个私有方法有没有转发。

#### 注入形态：并进 user 消息，不要单独一条

| 方案 | 问题 |
|---|---|
| 单独一条 `system` 插在中间 | vLLM 默认的 Llama-2 模板（`agents0to1/core/llm.py:281`）不吃；Ollama 也可能不吃 |
| 单独一条 `user` | 连续两条 user，Llama-2 模板拒绝，严格的校验器也拒绝 |
| **并进最终那条 user 消息** | ✅ 不新增消息、不动角色序列，任何 OpenAI 兼容服务都吃 |

**做法**：在最终那条 user 消息的 content 前面，拼一段带清晰分隔符的上下文块。

**代价**：那条 user 消息的内容不再等于 `input_text`。所以**不能有任何代码依赖这个相等关系**——改之前先搜一遍 `== input_text` 之类的判断。

#### 位置：必须在 `system + history` 之后

不要塞进 system 消息。塞进去会让 `system + history` 这段**稳定前缀每轮全变**，DeepSeek 的自动前缀缓存全部失效。

DeepSeek 缓存命中的价格差很多，这不是小事。

### 5.3 流式路径的 fail-open（最容易写出幽灵 bug 的地方）

`_stream_chat` 是个**生成器**。如果增强逻辑失败时直接 `return`：

1. `yield from self.llm.stream_invoke(...)` 没被执行
2. → `llm.last_response` 没被重置（`agents0to1/core/llm.py:364`）
3. → `self._last_response` 没被更新（`agents0to1/core/agent.py:114`，只有生成器跑完才会执行）
4. → ReAct 在 `agents0to1/classic_agent/react_agent.py:137` 读到的是**上一轮的**响应
5. → 看到 `has_tool_calls` → **把上一步的工具调用重跑一遍**

这是一个典型的**幽灵 bug**：不报错、不崩溃，只是行为诡异，而且你很难想到要去查那里。

**正确做法**：失败时「**原样吐出未增强的流**」，而不是 `return`。

而且**绝不能额外 yield 任何东西**——chunk 是逐字流给 UI 的（`agents0to1/classic_agent/react_agent.py:133-134`、`agents0to1/classic_agent/simple_agent.py:54-56`），多吐一个字符用户就看得见。

**fail-open 是必须的，不是可选的。** `_chat` / `_stream_chat` 是每一次 LLM 调用的**唯一漏斗**。embedding 失败很可能发生（独立 provider、独立 key、DeepSeek 还没有 embedding 端点）。失败时必须退化成和 `memory=None` **逐字节相同**的行为。

这和项目自己的哲学一致：`registry.execute` 工具失败是返回 `"错误:..."` 而不是抛异常（`agents0to1/tools/registry.py:65-83`）——**把错误变成正常流程的一部分**。

### 5.4 参数与命名

**加 `memory` 参数的位置**：加在 `__init__` 参数表的**最后**（`config` 之后）。四个子类也要加在各自参数表的最后。

原因：四个子类都是用**位置参数**调 `super().__init__(name, llm, system_prompt, config)`（`agents0to1/classic_agent/simple_agent.py:21`、`agents0to1/classic_agent/react_agent.py:68`、`agents0to1/classic_agent/reflection_agent.py:101`、`agents0to1/classic_agent/plan_solve_agent.py:194`）。如果你把 `memory` 插在 `config` 前面，所有位置调用会把 `config` 静默绑到 `memory` 上——**又是一个不报错的错误**。

**构造函数加类型守卫**：

```python
if memory is not None and not hasattr(memory, "search"):
    raise TypeError(...)
```

和项目已有的 `isinstance(agent, Agent)` 守卫风格一致（`agents0to1/classic_agent/plan_solve_agent.py:68-71 / 124-127`）。

**⚠️ 命名冲突**：`ReflectionAgent` **已经**有一个 `self.memory`（`agents0to1/classic_agent/reflection_agent.py:103`），而且 `run()` 开头会重新赋值（:117）。

基类再用 `self.memory` 会**被静默冲掉**。两个名字必须改一个。

建议改 Reflection 那个——它本质是"本轮的草稿轨迹"，叫 `self.scratch` 或 `self.trajectory` 更准确。**改完顺便名字也更贴切了**，这种改动是赚的。

**快照里不要放 memory**：

`Agent.snapshot()` 用 `json.dumps(..., default=str)`（`agents0to1/core/agent.py:319`），一个活的向量库句柄会被**静默序列化**成 `"<memory.semantic.SemanticMemory object at 0x...>"`。

然后 `_from_snapshot` 的 `setdefault`（`agents0to1/core/agent.py:352-353`）会把这个**字符串**喂给构造函数 → 第一次调 LLM 才炸 → 报错位置离病因十万八千里。

这就是 `agents0to1/classic_agent/react_agent.py:78-80` 那段注释警告的"悄悄退回默认值"同类问题。

**做法**：明确文档化"memory 不进快照"。`load(path, memory=...)` / `fork(path, memory=...)` 靠现有的 `setdefault` 机制**天然就支持覆盖**，不用额外写代码。

### 验证

跑同样的 prompt 两次：一次 `memory=None`，一次带记忆。对比 `last_messages` 和最终答案。再故意让 embedder 抛异常，确认行为和不带记忆时**逐字节相同**。

---

## 第六步：情景记忆持久化（可延后）

这一层解决"**跨会话记得你说过什么**"，和向量库解决的"**跨会话记得知识**"是两件事。

好在 `Turn` 已经是 pydantic 模型（`agents0to1/core/message.py:43`），`model_dump()` 现成，`_restore` 里 `Turn.model_validate` 也现成（`agents0to1/core/agent.py:327`）。

**一张 sqlite 表就够**：`session_id / turn_index / user / messages(JSON) / answer / metadata(JSON) / created_at`。

> 这张表和向量库可以是同一个 sqlite 文件，也可以分开。**建议分开**——两张表生命周期不同，混在一起以后想单独备份/清理会很别扭。

---

## 第七步：评测基线

### 核心原则：离线测试必须存在，且不依赖 API key

用**假 embedder**（文本 → 哈希 → 固定长度向量），下面这些全都不用联网：

- [ ] 余弦相似度算得对不对（构造已知答案的向量：正交→0、同向→1、反向→-1）
- [ ] 排序对不对（最像的排最前）
- [ ] 切分逻辑（句子边界、overlap、空输入、超长单句）
- [ ] 超预算时检索结果被正确裁剪
- [ ] **并发安全**（两个线程同时 search —— 这个只有并发测得出，见第三步）
- [ ] **fail-open**（embedder 抛异常时，agent 行为和不带记忆时一致）
- [ ] `memory=None` 时一切都是 no-op

**在线测试**（真调 LLM 验证"记忆有没有被用上"）单独放一个文件，标清楚需要 key。

### 为什么这个拆分是这份文档里最值钱的一条

离线那部分，是你**唯一**能"随时跑、立刻知道有没有改坏"的东西。

没有它，你这套东西加完之后，唯一的验证方式是"手动问几句、感觉好像变聪明了"。而那**不是验证**。

---

## 附录 A：决策表

最有价值的部分——**每个选择，被否掉的方案是什么，为什么否掉**。

| 决策 | 选了 | 否掉了 | 为什么 |
|---|---|---|---|
| 钩子位置 | `_build_messages` | `_chat` | `_stream_chat` **不经过** `_chat`，钩子挂那儿 ReAct 主循环和 SimpleAgent 流式路径一份都吃不到 |
| 查询来源 | `input_text` 参数 | `messages[-1]["content"]` | ReAct 第 2 步起最后一条是 **tool 消息**，不是用户问题 |
| 注入形态 | 并进 user 消息 | 独立 system / 独立 user 消息 | Llama-2 模板两者都不吃 |
| 注入位置 | history 之后 | system 消息里 | 保住 DeepSeek 的前缀缓存 |
| 统一注入点 | 分 Agent 处理 | 一处搞定四种 | 三种 Agent 的 `messages[-1]` 语义完全不同 |
| 向量存储 | BLOB (`array`) | JSON 文本 | 快得多（学习阶段可先用 JSON，转换本身就是练习） |
| 向量依赖 | 纯 Python + numpy 可选 | numpy 硬依赖 | 破坏"硬依赖只有两个" |
| sqlite | `check_same_thread=False` + 锁 | 默认行为 | 工具走线程池，一轮两个检索请求就炸 |
| memory 入快照 | 不入 | 入 | `default=str` 会静默字符串化，炸在很远的地方 |
| 自动入库 assistant 回答 | **禁止** | 允许 | 幻觉会被当成事实喂回来，形成自我确认循环 |
| 异常处理 | 记忆层 fail-open | 向上抛 | LLM 唯一漏斗，抛了就是整个 agent 挂掉 |
| 工具结果裁剪 | 在 `run()` 内按预算裁 | 靠 `truncate_output` | 尾部落盘会打乱相关性顺序；且要保证两条路径同一预算 |

## 附录 B：踩坑清单

写代码时对照这张表，能省你几个小时：

1. ~~**`str | Path` 在 3.9 上 import 期就炸**，用 `Union`~~
   → 作废。项目实际跑 3.10，见第一步 1.2 的更正。**教训是**：别拿机器上的默认解释器
   当项目的最低版本，先去问实际跑的那个环境是什么。
2. **embedding 表 ≠ chat provider 表**，DeepSeek 有 chat 没 embedding
3. **`_stream_chat` 不经过 `_chat`** —— 别假设"所有调用都经过某个方法"，去数一遍
4. **ReAct 第 2 步的 `messages[-1]` 是 tool 消息**
5. **流式路径 fail-open 不能用 `return`** —— 会读到上一轮的响应，重跑上一步的工具
6. **sqlite + 线程池** —— 单线程测不出来
7. **memory 参数加在参数表最后** —— 位置调用会静默错绑
8. **`ReflectionAgent` 已有 `self.memory`** —— 命名冲突，且会被 `run()` 冲掉
9. **`json.dumps(default=str)` 会把对象句柄静默字符串化** —— 别让活对象进快照
10. **跨模型/跨维度的向量不可比** —— 存 `model` + `dim`，跨模型直接拒绝，别算出个垃圾分数
11. **检索结果的位置影响前缀缓存** —— 别塞进 system
12. **绝不自动入库模型自己的回答** —— 幻觉自我加固

---

## 建议的提交顺序

每一步一个 commit，都能单独跑、单独回退：

```
✅ 已完成  修 ReAct 重复 user 消息；requires-python 提到 >=3.10  (不到 5 行)
✅ 已完成  包改名 agents0to1 + pyproject 修好 + 入口搬到 examples/  (2026-09-14)
2. agents0to1/memory/embedding.py                    (需要 key)
3. agents0to1/memory/vector_store.py + 离线测试       (不需要 key)
4. agents0to1/memory/semantic.py (切分 + 入库 + 检索) + 离线测试 (不需要 key)
5. agents0to1/tools/builtin/knowledge.py             (工具版, 需要 key)
6. Agent 接入 _build_messages + 四子类参数 + 改名      (上下文版, 需要 key)
7. agents0to1/memory/episodic.py                     (可延后)
8. 端到端评测场景集
```

**每一步做完都跑一次全部离线测试**——这是你能坚持下来的唯一原因。
