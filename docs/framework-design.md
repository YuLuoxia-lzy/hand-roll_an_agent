# 框架化改造指南

> 目标：把 Agents0to1 从「功能超前的单体」变成「骨架合格的框架」。
>
> 它现在**功能上是超前的**——四种 Agent 范式、工具系统、记忆层、快照、事件流，
> 这是应用级的丰富度。但**扩展点还停在单体阶段**：加一个记忆层，动了 9 个文件。
>
> 这份文档只做一件事：**把骨架补上**。
>
> 本文按**依赖顺序**编排，每一步都能单独跑通、单独验证。任何一步停下来，你手里都有一个能用的东西。

---

## 零、边界与约束

### 唯一判据

> **加一个新能力，应该只需要写一个新文件 + 挂上去，不需要改框架里的任何文件。**

这条判据可以直接当验收表用：

| 想做的事 | 今天要动几个文件 | 合格线 |
|---|---|---|
| 加记忆 | **9 个**（实测，178 行） | 1 个 |
| 加 trace / 成本统计 | ?（现在无处可挂） | 1 个 |
| 换向量库（faiss / chroma） | 改 `semantic.py` 源码 | 传个参数 |
| 换检索打分公式 | 改 `search()` 源码 | 传个参数 |
| 加文档解析器（pdf/docx） | 改 `chunker` 源码 | 传个参数 |
| 加一种新 Agent 范式 | 得读基类私有方法 | 1 个 |

### 本次做什么

| 阶段 | 产物 | 是否依赖 API key |
|---|---|---|
| 零 | 清红灯 + 补测试 | 否 |
| 一 | hook 管线（核心） | 否（用假 LLM 测） |
| 二 | 三处缝：store / scorer / delete+filters | 否 |
| 三 | 契约层：usage / 默认 stream_run / agent_id | 否 |
| 四 | 契约测试 | 否 |
| 五 | 扩展点地图 | 否 |

**全程不需要联网。** 这是刻意的——[tests/_harness.py](../tests/_harness.py) 里的 `FakeLLM` / `FakeEmbedder` 足够验证所有这些改造。

### 本次**不**做什么

| 不做 | 为什么 |
|---|---|
| ❌ asyncio / 异步化 | 整个栈同步，OpenAI SDK 也同步。切进去要改穿每一层，收益为零 |
| ❌ 依赖注入容器 / 插件系统 / entry_points | 过度设计。构造函数传参就够了 |
| ❌ 多 Agent 编排器（orchestrator） | **那是应用**。框架只给并发原语 |
| ❌ 任何「小镇」概念（tick / 世界 / 地点 / 日程） | **那是应用** |
| ❌ pdf / docx 解析实现 | 破「硬依赖只有两个」。留**接口**，不实现 |
| ❌ 把 VectorStore 换成 faiss | 手搓的够用几千条。**留缝就行，别换** |
| ❌ Rerank 模型 | 应用层。框架留 scorer 缝即可 |
| ❌ 父子块 / 混合检索实现 | 应用层。同上 |

### 三条硬约束

1. **硬依赖仍然只有 `openai` + `pydantic`。** 这是这个项目最值钱的一条洁癖。
   hook 管线、usage、agent_id、scorer 缝——**全部零新依赖**，用 stdlib 就够。
2. **Python 3.10 兼容。** `X | Y`、`list[dict]` 都能用（PEP 604 / 585）。
   但要记住：**跑之前先 `conda activate agent`**。新开的 shell 里 `python` 是基础环境
   的 3.9（`D:\Anaconda\python.exe`），`import agents0to1` 会报一个看不懂的
   `ModuleNotFoundError`。这个坑本指南作者已经踩过一次了。
3. **框架不替应用做决定，但要让应用的决定容易表达。**
   框架给「在哪里挂」，不给「挂什么」。

---

## 第零步：先清红灯

> ✅ 与框架改造无关，但**必须先做**。理由见下。

### 为什么必须最先做

不是「顺手修 bug」。是这三条**都会直接放大后面每一步的成本**：

- 红灯留着，你后面每次跑测试都要先分辨「这次红的是不是那条老毛病」——**验证能力直接归零**
- `.env` 泄漏那条让「离线测试」变成了「在线测试」，而它现在是**绿的**，最难发现
- episodic 零测试的那一层，恰恰是**后面小镇改造要动的地方**

加起来不到一小时，但它决定了后面每一步是否可定位。

### 0.1 修 `test_cross_model_search_rejected`

**位置**：[tests/test_vector_store.py:133](../tests/test_vector_store.py#L133)

```python
except VectorStoreException as e:
    assert "不可比" in str(e)          # ← 这里失败
```

**病因**：实现抛出的消息是

```python
# agents0to1/memory/vector_store.py:278-280
raise VectorStoreException(
    f"库里的向量来自 {stored_models}, 这次查询用的是 '{model}'。"
)
```

**没有「不可比」三个字**。而 `assert` 没有自定义消息，所以失败输出是**一片空白**——
这就是 `run_offline.py` 里那条 `[FAIL] test_cross_model_search_rejected` 下面没有内容的原因。

**行为是对的**（确实拒绝了跨模型检索），只是文案和断言脱节。

**改法**（推荐改实现，因为那句话对用户更有信息量）：

```python
raise VectorStoreException(
    f"跨模型的向量不可比：库里的向量来自 {stored_models}，"
    f"这次查询用的是 '{model}'。请换用产生这些向量的同一个模型查询。"
)
```

> ⚠️ **但先别急着只改这一行**——第 2.3 节会让这个函数整体重写。这里先记着，
> 到第二步一起改。**不要为了让它变绿而改测试**：那条断言想要的信息（「不可比」）
> 是用户真正需要看到的。

### 0.2 修 `test_snapshot_roundtrip_without_memory`（**.env 泄漏**）

**这条你现在看不见，因为它是绿的。**

**位置**：[tests/test_agent_memory.py:511](../tests/test_agent_memory.py#L511)

```python
SimpleAgent.load(str(path))          # ← 没传 llm=
```

**病因链**：

1. 不传 `llm=`，`Agent._from_snapshot`（[core/agent.py:435-444](../agents0to1/core/agent.py#L435-L444)）会自己造一个：
   ```python
   Agents0to1(provider="fake", base_url="http://fake.invalid/v1")
   ```
2. `fake` 落进 `_resolve_credentials` 的 `else` 分支，**需要 `LLM_API_KEY`**
3. 于是它去读环境变量——而 [agents0to1/__init__.py:_load_dotenv()](../agents0to1/__init__.py#L11-L42) 的
   `find_dotenv(usecwd=True)` 从仓库根**往上找**，捡到了 `Desktop\Agent\.env`
4. 那个 `.env` 里恰好有 `LLM_API_KEY` → **测试过了**

**实测**：

```
从仓库根跑            → Agent 接入: 33/33 全部通过
从 C:\Users\13082 跑  → Agent 接入: 32/33, 失败 ['test_snapshot_roundtrip_without_memory']
```

**为什么这条比 0.1 严重**：它直接违背 [memory-layer-guide.md](memory-layer-guide.md) 第七步的
头号原则——**「离线测试必须存在，且不依赖 API key」**。而且它**现在是绿的**。

> **一颗靠环境变量捂住的红灯，比一颗红的红灯危险得多。**

**改法**：同文件的兄弟测试 [test_agent_memory.py:516](../tests/test_agent_memory.py#L516)
特意传了 `llm=` 并写了注释说明理由，这条漏了。照抄：

```python
agent = SimpleAgent.load(str(path), llm=FakeLLM(["好的。"]))
```

**顺带**：整个测试套件应该有一个**环境隔离**的兜底。在 [tests/_harness.py](../tests/_harness.py)
的 `run_tests()` 开头加一句，把 `.env` 来的 key 清掉：

```python
def run_tests(namespace, title):
    # 离线测试不许吃 .env —— 否则「不依赖 API key」这条保证会被静默破坏
    for name in ("LLM_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ZHIPU_API_KEY"):
        os.environ.pop(name, None)
    ...
```

> ⚠️ 加了这句之后，0.2 那条会**立刻变红**——这正是你要的：
> **让被捂住的失败暴露出来，而不是等换了机器才发现。**

### 0.3 修 `_harness._bootstrap_path()` 自删

**位置**：[tests/_harness.py:35-49](../tests/_harness.py#L35-L49)

```python
sys.path.insert(0, root)                                    # 插入仓库根
sys.path[:] = [p for p in sys.path
               if p and not _is_original_project(p)]        # 又把它删了
```

`_is_original_project` 判定「路径以 `/hand-roll_an_agent` 结尾」——**而你这个仓库就叫这个名字**。
所以它判定「仓库根 = 原项目」，刚插进去就删了。**实测**：

```
REPO_ROOT = C:\Users\13082\Desktop\Agent\hand-roll_an_agent
REPO_ROOT in sys.path ?  False
```

**现在没炸**，是因为 editable 安装的 finder（`site-packages/__editable___agents0to1_0_1_1_finder.py`）
按 MAPPING 指向同一个目录，兜住了。但一旦 `pip uninstall agents0to1`、或者换台没装包的机器，
`tests/` 和 `examples/` 会全线 `ModuleNotFoundError`，**而且报错完全指不到这里**。

**根因**：这段代码的前提已经不成立了。它的注释还在讲「本机 pip install -e 装的是原项目，
副本叫 `hand-roll_an_agent copy`」——但机器上现在只有一份，名字就是 `hand-roll_an_agent`。

**两个改法（二选一）**：

**A. 改成「只信任自己插的这一条」**（推荐，简单且意图明确）：

```python
def _bootstrap_path() -> None:
    root = str(REPO_ROOT)
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)
    # 只把「不是我」的路径按需清理。这里不再猜哪个是原项目 ——
    # 副本和原项目可能同名, 猜错的代价(静默 import 到别的代码)比不清理大得多。
```

**B. 换判据**：不再按目录名猜，改成检查「那个路径下的 `agents0to1/memory/` 存不存在」。
只有不存在的才当原项目挤出去。

> 我推荐 **A**。理由：这段代码想要的是「保证测的一定是副本」，
> 而 `sys.path.insert(0, root)` **已经做到了**——Python 按顺序找，第一条就是它。
> 后面那段过滤是**额外**的保险，但它引入的风险比它消除的大。

### 0.4 给 `EpisodicMemory` 补测试

**309 行，零测试。** `tests/` 里**没有任何一个文件** import 过 `agents0to1.memory.episodic`。
唯一出现 "EpisodicMemory" 字符串的地方是一个**测试名**，而那个测试用的是本地 `class NoSearch: pass`。

完全没被覆盖的：

| 方法 | 该测什么 |
|---|---|
| `append` | 序号自动递增；`PRIMARY KEY (session_id, turn_index)` 冲突 |
| `extend` / `save_agent` | 追加语义；`limit` 参数；**重复保存会重复入库**（[episodic.py:294-309](../agents0to1/memory/episodic.py#L294-L309) 自己写的坑） |
| `recent` | 按时间**正序**返回；`limit` 截断 |
| `replay_into` | 灌回到 `_turns` **前面**；调了 `_truncate_history()`；不撑爆上下文 |
| `sessions` / `count` / `clear` | session 隔离；清一个不影响另一个 |
| `build_context` | fail-open；`query` 参数被忽略（签名兼容用）；budget 截断 |
| 跨进程 | 换个 `EpisodicMemory` 实例读同一个文件 |

**最后一个尤其重要**：这个模块存在的全部理由就是「跨会话」，而**跨会话 = 跨进程**。
只在一个实例里测是测不到的。

**参考**：`tests/test_semantic.py` 的结构可以照抄（它测了 `close` 幂等、fail-open、预算截断）。

### 0.5 拆掉两条「装样子」的测试

[tests/_harness.py:66-69](../tests/_harness.py#L66-L69) 里你自己写着：

> **绝不能用它（SkipTest）掩盖失败** —— 那样测试就变成了装样子，比没有还坏。

这两条比 SkipTest 更糟——它们连 skip 都不算，是**零断言还报 PASS**：

**① [tests/test_semantic.py:247](../tests/test_semantic.py#L247) `test_fail_open_does_not_print_traceback`**

名字说它验证用的是 `logger.warning` 而不是 `logger.exception`。但**没有任何东西观察 logging**。
实现改成打整个堆栈，它照样绿。

**改法**：挂一个 handler 上去抓日志级别：

```python
def test_fail_open_logs_warning_not_traceback():
    """fail-open 时的日志必须是一条 warning, 不是整个堆栈 ——
    生产里这会反复发生(embedding 独立 provider、独立 key), 打堆栈会把日志刷爆。"""
    import logging
    records = []

    class Grab(logging.Handler):
        def emit(self, record):
            records.append(record)

    logger = logging.getLogger("agents0to1.memory.semantic")
    handler = Grab(level=logging.DEBUG)
    logger.addHandler(handler)
    try:
        # ... 构造一个坏 embedder, 调 build_context ...
        pass
    finally:
        logger.removeHandler(handler)

    assert records, "fail-open 时应该留下一条日志, 否则出问题没人看得见"
    assert all(r.levelno == logging.WARNING for r in records), \
        f"fail-open 只该 warning, 收到: {[(r.levelname, r.msg) for r in records]}"
```

**② [tests/test_semantic.py:268](../tests/test_semantic.py#L268) `test_close_is_idempotent`**

只测「不抛」。加一条：`close()` 之后 `count()` 应该给一个**清楚的**报错，而不是 `AttributeError: 'NoneType'`。
（这也顺带修了 [vector_store.py:372-376](../agents0to1/memory/vector_store.py#L372-L376) 的一个小毛病。）

### 0.6 `.gitignore` 补 `data/`

**位置**：[.gitignore](../.gitignore)

`SemanticMemory()` / `EpisodicMemory()` 的默认路径是**相对 CWD** 的 `data/*.sqlite3`。
在仓库根跑一次就会生成 `hand-roll_an_agent/data/`，它会以未跟踪文件的形式出现在 `git status` 里
——**向量库和聊天记录很容易被误提交**。

```gitignore
# 记忆层默认落盘位置(相对 CWD) —— 向量库和对话记录不该进版本库
data/
*.sqlite3
*.sqlite3-wal
*.sqlite3-shm
```

### 0.7 `.env.example` 补 `EMBEDDING_*`

**位置**：[.env.example](../.env.example)

[memory-layer-guide.md 第二步](memory-layer-guide.md) 专门花了篇幅论证「embedding 必须有自己独立的一组配置」，
但 `.env.example` 里**一个都没列**。新用户复制过去会以为配好 `DEEPSEEK_API_KEY` 就够了，
然后在 `EmbeddingClient._detect_provider()` 那里撞上「没能自动检测出 embedding provider」。

```bash
# ==================== 记忆层 / Embedding(和 chat 是两套独立配置) ====================
# DeepSeek 没有 embeddings 端点, 所以 embedding 必须另外配一家。
# 不配的话记忆层构造时会报「没能自动检测出 embedding provider」, 但**不影响 chat**。
#
# EMBEDDING_PROVIDER=zhipu     # openai / dashscope / zhipu / modelscope / ollama
# EMBEDDING_MODEL=embedding-3
# EMBEDDING_API_KEY=           # 不填就按 provider 找对应的 key(如 ZHIPU_API_KEY)

# ==================== 向量库 ====================
# blob(默认, 快) / json(慢, 但能用眼睛看) —— 两种格式可以在同一张表里共存
# VECTOR_STORAGE=blob
```

> ⚠️ **顺带一条真实的隐患**：`.env` 里 embedding 走 zhipu、chat 走 deepseek，
> 这是**检测顺序级**的保证而不是铁律——`DEEPSEEK_API_KEY` 一旦消失，chat 会**静默**变成 glm。
> 两张表在命中多个 key 时都只 `logger.warning`，**不报错**。
> 要钉死就在 `.env` 加 `EMBEDDING_PROVIDER=zhipu`（chat 侧没有 `LLM_PROVIDER` 这类变量可设）。

### 0.8 提交

**这是当前风险最高的状态**：`git status` 显示 **9 个 M + 5 组 `??`**，
记忆层 1475 行 + 测试 2100 行 + 534 行文档，全堆在一个未提交的工作区里。

清完 0.1–0.7 之后，按 [memory-layer-guide.md](memory-layer-guide.md) 末尾那份「建议的提交顺序」补提交。
**分多个 commit**，每个都能单独回退。

### 验证

```bash
conda activate agent                          # ← 别忘了
python tests/run_offline.py                   # 必须 4/4 全绿, 且退出码 0
cd /c/Users/13082 && python <仓库>/tests/test_agent_memory.py   # 换个目录也要全绿(验 .env 隔离)
```

> 第二条命令是关键：**0.2 的修法只有在「换个目录跑」时才验得出来。**

---

## 第一步：hook 管线（核心）

### 1.1 为什么是它

**证据**：加记忆层这一次，动了 **9 个文件、178 行**。

```
agents0to1/__init__.py
agents0to1/core/agent.py
agents0to1/core/exceptions.py
agents0to1/tools/__init__.py
agents0to1/tools/builtin/__init__.py
agents0to1/classic_agent/simple_agent.py
agents0to1/classic_agent/react_agent.py
agents0to1/classic_agent/reflection_agent.py
agents0to1/classic_agent/plan_solve_agent.py
```

拆开看**每一处改动都判断对了**，问题是框架没给你别的选择：

| 改动 | 根本原因 |
|---|---|
| 四个子类各改一遍 `__init__` | memory 成了**基类的参数** |
| `_build_messages` 改造 + 新增 `_prepare_user_message` | 注入逻辑成了**基类的方法** |
| ReAct 单独改 `_finish` | 记录 Turn 时要**还原**被注入的内容 |
| PlanSolve 改 `Planner.plan` + `Executor.execute` 签名 | 它**不走** `_build_messages` |
| Reflection 得把一个叫 `Memory` 的类改名 | 名字冲突 |
| PlanSolve + Reflection **各手抄了同样三行** | 它们不走基类钩子，只能自己调 `_memory_context()` |

> **最后一行是框架内部细节泄漏到子类的典型症状。** 两个子类不抄不行。

**后果是可预测的**：每加一种横切能力（记忆流打分、世界感知、trace、情绪、成本熔断），
这套动作就要重来一遍。**第 N 种能力的成本是 O(所有子类)。**

### 1.2 接口设计

新建 `agents0to1/core/hooks.py`。

```python
"""扩展点 —— 框架的「入站拦截」。

AgentEvent 是出站事件流(agent 往外吐), hook 是入站拦截(外面往里插)。
两者合起来, 框架的骨架才完整。

【五个阶段, 按一次 run() 的时间顺序】
    before_input   改用户输入           —— 世界感知、输入清洗、脱敏
    before_llm     改将要发给模型的消息  —— 记忆注入、RAG、上下文压缩  ← 记忆层在这
    after_llm      观察/改模型响应       —— 用量统计、trace、内容审核
    after_tool     观察/改工具结果       —— 世界状态更新、结果脱敏
    after_run      一轮结束              —— 落库、打分、写记忆流        ← 情景记忆在这

【为什么用 RunContext 而不是把参数摊开】
见 1.3 —— 它一次性解决了 memory-layer-guide 里两个靠「约定」才成立的约束。
"""
```

```python
from dataclasses import dataclass, field
from typing import Any, List, Optional
import time

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RunContext:
    """一次 run() 的上下文。**在 hook 之间共享, 也是 hook 唯一的「我在哪」来源。**"""

    #: 本轮用户**原始**输入。永远不被改写 —— 这是 hook 用它做检索 query 的前提。
    input_text: str
    #: 发起这次 run 的 agent
    agent: Any = None
    #: 本轮是第几轮(agent._turns 的长度, 记录前)
    turn_index: int = 0
    #: 开始时间
    started_at: float = field(default_factory=time.time)
    #: 本轮已经发起了几次 LLM 调用(before_llm 看到的是「这次是第几次」, 从 0 开始)
    llm_calls: int = 0
    #: 自由槽位 —— hook 之间传递临时状态用, 框架不碰它
    state: dict = field(default_factory=dict)


class Hook:
    """
    扩展点基类。

    **只覆盖你关心的那几个方法。** 基类里五个都是直接返回入参的 no-op,
    所以一个只想记 trace 的 hook 只需要写 after_llm 一个方法。

    【on_error: 每个 hook 自己声明失败策略】
        "open"   —— 吞掉, 记一条 warning, 继续跑(记忆、trace、埋点用这个)
        "closed" —— 抛出去, 中断本轮(成本熔断、权限校验、内容合规用这个)

    为什么让 hook 自己声明, 而不是框架统一决定:
        框架**没有资格**替应用决定「这个 hook 挂了要不要中断对话」。
        记忆挂了应该继续聊, 但「这个用户欠费了」挂了必须中断。
        两者形状一样, 语义相反。
    """

    #: 见上面的说明
    on_error: str = "open"

    def before_input(self, ctx: RunContext, input_text: str) -> str:
        return input_text

    def before_llm(self, ctx: RunContext, messages: List[dict]) -> List[dict]:
        return messages

    def after_llm(self, ctx: RunContext, response):
        return response

    def after_tool(self, ctx: RunContext, call, result: str) -> str:
        return result

    def after_run(self, ctx: RunContext, turn) -> None:
        return None
```

```python
class HookPipeline:
    """按注册顺序跑一组 hook。"""

    def __init__(self, hooks: Optional[List[Hook]] = None):
        self.hooks: List[Hook] = list(hooks or [])

    def add(self, hook: Hook) -> "HookPipeline":
        """链式注册: pipeline.add(A()).add(B())"""
        self.hooks.append(hook)
        return self

    # ==================== 五个阶段 ====================

    def before_input(self, ctx: RunContext, input_text: str) -> str:
        for hook in self.hooks:
            input_text = self._guard(hook, "before_input", ctx, input_text)
        return input_text

    def before_llm(self, ctx: RunContext, messages: List[dict]) -> List[dict]:
        for hook in self.hooks:
            messages = self._guard(hook, "before_llm", ctx, messages)
        return messages

    def after_llm(self, ctx: RunContext, response):
        for hook in self.hooks:
            response = self._guard(hook, "after_llm", ctx, response)
        return response

    def after_tool(self, ctx: RunContext, call, result: str) -> str:
        for hook in self.hooks:
            result = self._guard(hook, "after_tool", ctx, call, result)
        return result

    def after_run(self, ctx: RunContext, turn) -> None:
        for hook in self.hooks:
            self._guard(hook, "after_run", ctx, turn)

    # ==================== 内部 ====================

    @staticmethod
    def _guard(hook: Hook, stage: str, *args):
        """
        跑一个 hook, 按它自己声明的策略处理失败。

        失败时**返回入参的原值** —— 也就是「这个 hook 什么都没做」,
        这是 fail-open 唯一正确的退化形态。
        """
        fn = getattr(hook, stage, None)
        if fn is None:
            return args[-1]
        try:
            return fn(*args)
        except Exception as e:
            if getattr(hook, "on_error", "open") == "closed":
                raise
            # warning 而不是 exception: 这在生产里是会反复发生的事,
            # 每次都打整个堆栈会把日志刷爆(memory-layer-guide 5.3 同样的理由)
            logger.warning(
                "%s.%s 失败, 已跳过(本轮行为与没挂它时一致): %s",
                type(hook).__name__, stage, e,
            )
            return args[-1]

    def __len__(self) -> int:
        return len(self.hooks)

    def __repr__(self) -> str:
        return f"HookPipeline({[type(h).__name__ for h in self.hooks]})"
```

### 1.3 RunContext：为什么不是把参数摊开

`before_llm(ctx, messages)` 里那个 `ctx` 是**这次改造里最不显眼、但最值钱的一个设计**。
理由有两条，每一条都在解决一个**现在就靠「约定」才成立的约束**。

#### ① 它消灭了「查询该用什么」这个约定

[memory-layer-guide.md 5.2](memory-layer-guide.md) 里有一整节在讲：

> **关键点**：查询要用 `input_text` 参数本身，**绝对不要用 `messages[-1]["content"]`**。
> 因为到了 ReAct 第 2 步，最后一条是 `tool` 消息。拿工具结果去检索，检索出来的东西和用户问题毫无关系。

那条约束今天是**靠人记住**的。而 `ctx.input_text` **永远是本轮原始输入**，不管你挂在哪个阶段、
第几次 LLM 调用。**约束从「记得这么做」变成了「结构上不可能做错」。**

#### ② 它解决了「每个 hook 的签名都不一样」的问题

`after_tool` 需要 `call` 和 `result`，`after_run` 需要 `turn`，`after_llm` 只需要 `response`。
如果把它们摊平进签名，那每加一个阶段就要改所有 hook 的基类。
`ctx` 是那个**会增长的参数**——加字段不破坏任何已有 hook。

> **教训**：框架里凡是有「每次调用都需要、而且以后可能变多」的参数，
> 就应该打包成一个 context 对象。摊平进签名的每一个参数，都是一个未来会破的兼容性承诺。

### 1.4 埋点位置（逐个方法，附行号）

改 `agents0to1/core/agent.py`。

#### 1.4.1 `Agent.__init__` 加两个属性

```python
def __init__(self, name, llm, system_prompt=None, config=None,
             memory=None, hooks=None):        # ← hooks 加在最后(理由同 memory)
    ...
    self._hooks = HookPipeline(hooks)
    self._ctx: Optional[RunContext] = None
```

> ⚠️ **`hooks` 必须加在参数表最后**——和 `memory` 同一个理由（[core/agent.py:47-48](../agents0to1/core/agent.py#L47-L48) 那段注释）。
> 四个子类用**位置参数**调 `super().__init__(name, llm, system_prompt, config)`，
> 插在前面会让 `config` 静默绑到 `hooks` 上。**又是一个不报错的错误。**

#### 1.4.2 `_chat`（[core/agent.py:165-169](../agents0to1/core/agent.py#L165-L169)）

```python
def _chat(self, messages: list[dict], tools=None, **kwargs) -> LLMResponse:
    outgoing = self._ensure_system(messages)
    outgoing = self._hooks.before_llm(self._ctx, outgoing)      # ← 新增
    response = self.llm.invoke(outgoing, tools=tools, **kwargs)
    response = self._hooks.after_llm(self._ctx, response)       # ← 新增
    self._ctx.llm_calls += 1                                    # ← 新增(在 hook 之后自增)
    return response
```

**为什么 `llm_calls` 在 hook 之后自增**：这样 `before_llm` 里读到的 `ctx.llm_calls`
就是「**这次是第几次**」，从 0 开始。记忆层靠它判断「是不是本轮第一次调用」——
见 1.8。

#### 1.4.3 `_stream_chat`（[core/agent.py:171-179](../agents0to1/core/agent.py#L171-L179)）

```python
def _stream_chat(self, messages: list[dict], tools=None, **kwargs) -> Iterator[str]:
    outgoing = self._ensure_system(messages)
    outgoing = self._hooks.before_llm(self._ctx, outgoing)      # ← 新增

    yield from self.llm.stream_invoke(outgoing, tools=tools, **kwargs)

    self._last_response = self.llm.last_response
    # 流式路径的 after_llm **只能是观察, 不能改** —— 内容早就逐块吐给调用方了。
    # 见 1.6 的说明。
    self._last_response = self._hooks.after_llm(self._ctx, self._last_response)
    self._ctx.llm_calls += 1                                    # ← 新增
```

> ⚠️ **`_chat` 和 `_stream_chat` 是两处，不是一处。**
> [memory-layer-guide.md](memory-layer-guide.md) 里那个「反例教学」讲的正是这件事：
> `_stream_chat` **不经过** `_chat`。当年钩子挂在 `_chat` 上，ReAct 主循环和
> SimpleAgent 的流式路径**一份都吃不到，而且不报错**。
>
> **现在两处都要挂。而且必须挂两处**——这是 hook 设计的固有代价，写进 1.6。

#### 1.4.4 `add_turn` 里触发 `after_run`

`add_turn`（[core/agent.py:189-224](../agents0to1/core/agent.py#L189-L224)）是所有 Agent 的收尾必经之路。
在这里触发，**四种 Agent 全都自动吃到，一个都不用改**：

```python
self._turns.append(turn)
self._truncate_history()

# after_run 放在 add_turn 里, 是为了让四种 Agent 都自动吃到 ——
# 它们都通过 _record_turn 走到这里, 不需要各自改 run()。
if self._ctx is not None:
    self._hooks.after_run(self._ctx, turn)
```

#### 1.4.5 ReAct 的工具结果

改 `agents0to1/classic_agent/react_agent.py:_execute_tool_calls`（[:214-232](../agents0to1/classic_agent/react_agent.py#L214-L232)）：

```python
for call, result in zip(calls, results):
    result = self._hooks.after_tool(self._ctx, call, result)    # ← 新增
    logger.info("工具 %s(%s) -> %s", call.name, call.arguments, result)
    messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
```

> ⚠️ `after_tool` 必须放在 `messages.append` **之前**——否则改的是 `messages` 里的副本，
> 而 `pairs.append((call, result))` 吐给 `tool_result` 事件的还是旧值，**两份真相**。

### 1.5 模板方法：把生命周期收归框架

**问题**：`ctx` 谁来建？谁来清？如果让子类在 `run()` 开头自己调，那**就是下一个「必须记得做的事」**
——而 1.1 那张表里已经有一堆「必须记得做的事」了。

**做法**：把 `run()` 变成**模板方法**，子类实现 `_run()`。

```python
# agents0to1/core/agent.py

def run(self, input_text: str, **kwargs) -> str:
    """
    模板方法: 建上下文 -> 跑子类实现 -> 收尾。

    **子类不要覆盖这个方法**, 覆盖 _run()。
    ctx 的生命周期由框架管, 子类忘不掉。
    """
    ctx = RunContext(
        input_text=input_text,
        agent=self,
        turn_index=len(self._turns),
    )
    self._ctx = ctx
    try:
        input_text = self._hooks.before_input(ctx, input_text)
        return self._run(input_text, **kwargs)
    finally:
        self._ctx = None

@abstractmethod
def _run(self, input_text: str, **kwargs) -> str:
    ...
```

> ⚠️ **`before_input` 在 `ctx.input_text` 已经是原值之后才跑，这是刻意的。**
> `ctx.input_text` 是给 hook 用的「用户到底问了什么」，**它不该被任何 hook 改写**。
> `before_input` 改写的是**送进子类的那份**。如果哪天有 hook 真的改了输入，
> 那 `ctx.input_text` 和实际输入就不一致了——**这个不一致是有意的**，
> 因为它决定了「用哪句话去检索」。

`stream_run` 同样处理：

```python
def stream_run(self, input_text: str, **kwargs) -> Iterator[AgentEvent]:
    """
    默认实现: 任何实现了 _run() 的 Agent, 都能被包成一个只吐 final 的事件流。
    想做精细的(工具事件)再覆盖 _stream_run()。
    """
    ctx = RunContext(input_text=input_text, agent=self, turn_index=len(self._turns))
    self._ctx = ctx
    try:
        input_text = self._hooks.before_input(ctx, input_text)
        yield from self._stream_run(input_text, **kwargs)
    finally:
        self._ctx = None

def _stream_run(self, input_text: str, **kwargs) -> Iterator[AgentEvent]:
    """默认: 退化成 run() 的单个 final 事件。**子类可以覆盖。**"""
    answer = self._run(input_text, **kwargs)
    yield AgentEvent(type="final", answer=answer)
```

**这一下顺手解决了两个老问题：**

| 老问题 | 现在 |
|---|---|
| `stream_run` 只有 2/4 个 Agent 实现 | 基类有默认实现，四个全都有 |
| PlanSolve / Reflection 各手抄三行调 `_memory_context()` | 有了 hook，抄的那三行可以删掉 |

### 1.6 两条必须写进注释的不对称

这两个是 hook 设计的固有代价，**不写清楚一定会有人踩**：

#### ① 流式路径的 `after_llm` **只能观察，不能改**

`_stream_chat` 是生成器，chunk 早就逐块吐给调用方了。`after_llm` 拿到的
`self._last_response` 只是「拼起来之后的完整响应」，**改它对已经吐出去的字符没有任何影响**。

所以：

- 用 `after_llm` 做 **usage 统计 / trace / 计数** → ✅ 两条路径都行
- 用 `after_llm` 做 **内容审核 / 改写** → ❌ 只有非流式路径有效

**这个不对称无法消除**（除非放弃流式），所以只能**明确文档化**。

#### ② 挂一处不够，必须挂两处

`_chat` 和 `_stream_chat` 是两条独立路径（见 1.4.3）。任何「所有 LLM 调用都要做的事」
必须在**两处**都挂。这是 [memory-layer-guide.md](memory-layer-guide.md) 那个反例
花了一整节讲的事——**当年就是漏了流式那一处，静默失效。**

### 1.7 不可变性：`before_llm` 必须返回**新**列表

**这是 hook 设计里最容易写错、也最难查的一条。**

#### 问题

ReAct 的 `messages` 是循环里**不断 append 的活列表**：

```python
messages = self._build_messages(input_text)     # react_agent.py:132
...
self._append_assistant_tool_calls(messages, text, response.tool_calls)
self._execute_tool_calls(messages, response.tool_calls)
...
recorded = list(messages[turn_start:])          # react_agent.py:246
```

如果 `before_llm` **原地改**了 `messages[-1]["content"]`，那么：
1. `self.last_messages`（= 那个活列表）被污染
2. `messages[turn_start:]` 存进 Turn 时带着检索结果
3. 下一轮 `_history_messages()` 把它当「用户说过的话」重发
4. `_turn_chars()` 的预算被白吃

——**这正是今天的 `_record_first_message()` 在补的洞。**

#### 解决

**约定：hook 不得原地修改入参，必须返回新对象。**

```python
# ❌ 错: 原地改
messages[-1]["content"] = f"{context}\n\n{messages[-1]['content']}"
return messages

# ✅ 对: 新列表 + 新字典
out = list(messages)                                         # 浅拷贝列表
out[-1] = {**out[-1], "content": f"{context}\n\n{out[-1]['content']}"}   # 新字典
return out
```

**代价**：`_chat` / `_stream_chat` 必须用**返回值**发请求，而不是 `messages` 本身：

```python
outgoing = self._hooks.before_llm(self._ctx, self._ensure_system(messages))
response = self.llm.invoke(outgoing, ...)        # ← 用 outgoing
```

**收益**：`_record_first_message()` **可以整个删掉**，ReAct 的 `_finish` 也回到朴素写法：

```python
def _finish(self, input_text, final_answer, messages, turn_start):
    self.last_messages = messages                            # 不含检索结果了
    self._record_turn(input_text, final_answer, messages=messages[turn_start:])
    return final_answer
```

> ⚠️ **但 `last_messages` 的语义变了。** 它的注释说「模型当时看到了什么」
> （[react_agent.py:77-78](../agents0to1/classic_agent/react_agent.py#L77-L78)）。
> 不可变约定之后，`messages` 是「agent 的工作列表」，**不是**「模型看到的东西」。
>
> 两个选择，**选一个并写进注释**：
> - **A**：`last_messages` 改成在 `_stream_chat` 里存 `outgoing`（真正发出去的那份）
> - **B**：`last_messages` 重新定义为「agent 的工作列表」，改名 `last_working_messages`
>
> 推荐 **A**——保留原来那个更有用的语义。

### 1.8 迁移：把记忆层改写成 hook

**这是整个第一步的验收标准。** 目标是：**从 9 个文件变成 1 个文件。**

#### 新建 `agents0to1/hooks/memory.py`（或 `agents0to1/memory/hook.py`）

```python
"""记忆 hook —— 把记忆层挂进 Agent 的唯一入口。

【这个文件的存在本身就是验收证据】
改造之前, 加记忆要动 9 个文件、178 行(四个子类的构造函数、基类的 _build_messages、
ReAct 的 _finish、PlanSolve 的两个方法签名、一个类改名)。
改造之后, 就是这一个文件。
"""

from ..core.hooks import Hook, RunContext


class MemoryHook(Hook):
    """
    语义记忆 / 情景记忆的注入。

    **fail-open**: 记忆只是增强, 它没有权力把整个 agent 拖下水。
    (memory-layer-guide 5.3 —— 记忆层正对 LLM 漏斗, 抛出去就是整个 agent 挂掉)
    """

    on_error = "open"           # ← 声明失败策略, 框架照此办理

    def __init__(self, memory):
        if memory is not None and not (
            hasattr(memory, "search") or hasattr(memory, "build_context")
        ):
            raise TypeError(
                f"memory 需要提供 search() 或 build_context() 方法, "
                f"收到的是 {type(memory).__name__}"
            )
        self.memory = memory

    def before_llm(self, ctx: RunContext, messages: list[dict]) -> list[dict]:
        if self.memory is None:
            return messages

        # 只在本轮**第一次** LLM 调用时注入。
        # 为什么: ReAct 第 2 步起 messages[-1] 是 tool 消息, 那时候再注入
        # 就等于把检索结果插进工具结果中间 —— 语义上说不通, 而且每步都注入
        # 会让同一份资料在上下文里出现 N 次。
        if ctx.llm_calls > 0:
            return messages

        context = self.memory.build_context(ctx.input_text)
        if not context:
            return messages

        # 必须返回新列表 + 新字典, 不得原地改 —— 见 framework-design 1.7
        out = list(messages)
        out[-1] = {**out[-1], "content": f"{context}\n\n{out[-1]['content']}"}
        return out
```

> **注意 `ctx.input_text`**：这里用的是它，**不是** `messages[-1]["content"]`。
> [memory-layer-guide.md 5.2](memory-layer-guide.md) 里那条靠人记住的约束，
> 现在由 `RunContext` **在结构上**保证了。

#### 然后从框架里删掉这些东西

| 删掉 | 位置 | 为什么能删 |
|---|---|---|
| `Agent.memory` 属性 + 类型守卫 | [core/agent.py:45-60](../agents0to1/core/agent.py#L45-L60) | 搬进 `MemoryHook.__init__` |
| `Agent._prepare_user_message` | [core/agent.py:110-121](../agents0to1/core/agent.py#L110-L121) | 搬进 `MemoryHook.before_llm` |
| `Agent._record_first_message` | [core/agent.py:123-128](../agents0to1/core/agent.py#L123-L128) | **不可变约定让它变成多余的** |
| `Agent._memory_context` | [core/agent.py:130-153](../agents0to1/core/agent.py#L130-L153) | 搬进 hook，且不再需要那个 `hasattr` 分支 |
| ReAct `_finish` 里的还原逻辑 | [react_agent.py:246-248](../agents0to1/classic_agent/react_agent.py#L246-L248) | 同上 |
| PlanSolve `run()` 里的 `context = self._memory_context(...)` | [plan_solve_agent.py](../agents0to1/classic_agent/plan_solve_agent.py) | 那三行手抄可以删了 |
| Reflection `run()` 里的同上 | [reflection_agent.py](../agents0to1/classic_agent/reflection_agent.py) | 同上 |
| 四个子类的 `memory` 参数 | 四个文件 | 改用 `hooks=[MemoryHook(mem)]` |

#### ⚠️ 一个不能照搬的地方：PlanSolve / Reflection

这两个 Agent **不走 `_build_messages`**——它们自己拼 prompt 直接 `_chat`。
挂上 `MemoryHook` 之后，`before_llm` 会在**每次 `_chat` 时**触发，于是：

- PlanSolve 的规划器调用 → 注入 ✅
- PlanSolve 的**每一步执行**调用 → `ctx.llm_calls > 0` 挡住了 ❌ **但规划那步已经把它变成 1 了**

**所以 `llm_calls > 0` 这个判据对它们不成立**——它们一轮里有 N 次「第一次」。

**两个解法，选一个：**

- **A（推荐）**：`MemoryHook` 加一个 `once_per_turn: bool = True`，
  内部用 `ctx.state` 记「本 ctx 注入过没有」而不是看 `llm_calls`
  ```python
  if ctx.state.get("memory_injected"):
      return messages
  ...
  ctx.state["memory_injected"] = True
  ```
  这个判据对**所有** Agent 都成立，因为它记的是「这个 ctx 里注入过没」。
- **B**：PlanSolve / Reflection 保持现状（自己在 run 里算一次）。

> 推荐 **A**。`ctx.state` 就是为这种「hook 之间的临时状态」准备的——
> 框架不碰它，但它是 hook 能安全共享状态的唯一地方。

### 1.9 验证

**全部离线，用 `FakeLLM` + `FakeEmbedder`，不需要 key。**

```python
# 1) 一个只记 trace 的 hook, 不改任何东西 —— 行为必须和没挂时逐字节相同
class TraceHook(Hook):
    def __init__(self): self.stages = []
    def before_llm(self, ctx, messages):
        self.stages.append(("before_llm", ctx.llm_calls)); return messages
    def after_run(self, ctx, turn):
        self.stages.append(("after_run", turn.user))

# 2) 一个 on_error="closed" 的 hook 抛异常 —— 必须中断
class BoomHook(Hook):
    on_error = "closed"
    def before_input(self, ctx, text): raise RuntimeError("boom")

# 3) 一个 on_error="open" 的 hook 抛异常 —— 必须静默跳过, 行为不变
class SoftBoomHook(Hook):
    on_error = "open"
    def before_llm(self, ctx, messages): raise RuntimeError("boom")
```

必测清单：

- [ ] 挂了 no-op hook 的 agent，`FakeLLM.calls` 和没挂时**逐字节相同**
- [ ] `on_error="open"` 的 hook 抛异常 → agent 照常跑完，**且 `FakeLLM.calls` 逐字节相同**
- [ ] `on_error="closed"` 的 hook 抛异常 → 异常穿透出来
- [ ] 两个 hook 按**注册顺序**执行（用 `TraceHook` 记录顺序）
- [ ] `before_llm` 返回新列表 → `agent.last_messages` **不含**注入内容
- [ ] 流式路径（`_stream_chat`）上也触发了 `before_llm`
- [ ] `ctx.input_text` 在 ReAct 第 2 步仍然是**用户原始问题**（这是 RunContext 存在的理由）
- [ ] 迁移后，原来 `test_agent_memory.py` 那 33 个用例**全部还能过**（只改挂载方式）

> **最后一条是最强的验收**：那 33 个用例是照着「记忆已经接对了」写的。
> 迁移到 hook 之后它们还能全绿，就说明行为**没有退化**。

**验收标准（一句话）**：

> 把 `MemoryHook` 从 `hooks=[...]` 里摘掉，agent 应该完全不知道记忆层存在；
> 重新挂上，只加一个文件。**框架里的任何文件都不需要改。**

---

## 第二步：三处缝

> 这三个都是**「文档声称可换、实际换不了」**的缺口。
> 共同点：**一行到几十行的改动**，但决定了应用层能不能自己选。

### 2.1 向量库可注入

**位置**：[semantic.py:226](../agents0to1/memory/semantic.py#L226)

```python
self.store = VectorStore(path=path, table=table, storage=storage)      # 硬编码
```

但 [vector_store.py:3](../agents0to1/memory/vector_store.py#L3) 的文档写着：

> 接口定好了，之后换 faiss / sqlite-vec / chroma 就是换一个类

**今天做不到——没有那条缝。**

**改法**：

```python
def __init__(self, ..., store: Optional[Any] = None):
    ...
    if store is not None:
        # 鸭子类型守卫, 和项目已有的 isinstance(agent, Agent) 风格一致
        for method in ("add", "search", "count", "clear", "close"):
            if not hasattr(store, method):
                raise TypeError(
                    f"store 需要提供 {method}() 方法, 收到的是 {type(store).__name__}"
                )
        self.store = store
    else:
        self.store = VectorStore(path=path, table=table, storage=storage)
```

`EpisodicMemory` 同理，但**优先级低**——它的「换后端」需求远不如向量库真实。
**先做 semantic 这一个就够。**

### 2.2 检索打分策略（**这条最重要**）

**这是「文档系统」和「小镇」那个冲突的解法。**

| | 要的打分 |
|---|---|
| 文档型 agent 系统 | **relevance-only** —— 一份 2023 年的手册不该因为「旧」就沉下去 |
| 小镇（记忆流） | **recency + importance + relevance** —— 三周前的闲聊该沉下去 |

**这两个打分函数是冲突的。** 框架不该替应用选，该留一条缝。

**位置**：[vector_store.py:286-331](../agents0to1/memory/vector_store.py#L286-L331) 的 `search()`

**关键**：`created_at` 必须能被 scorer 拿到。今天 `search()` 的 SELECT
（[:311-314](../agents0to1/memory/vector_store.py#L311-L314)）**只取了 `text, metadata, norm, storage, vector`**。

**改法**：

```python
#: 打分函数的签名。默认是纯余弦(文档系统的正确选择)。
#: 小镇要换成 recency + importance + relevance, 就传一个自己的进来。
Scorer = Callable[..., float]


def search(
    self,
    query_vector: Sequence[float],
    top_k: int = 5,
    model: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    scorer: Optional[Scorer] = None,
) -> List[Tuple[float, str, Dict[str, Any]]]:
```

SELECT 补上 `created_at`（和 `id`，`delete` 要用）：

```python
SELECT id, text, metadata, norm, storage, vector, created_at FROM {table}
```

然后用：

```python
score_fn = scorer or cosine_similarity
for row in rows:
    ...
    score = score_fn(
        query, vector,
        query_norm=q_norm,
        vector_norm=row["norm"],
        created_at=row["created_at"],       # ← scorer 需要的额外信号
        metadata=meta,
    )
```

**为什么 `created_at` 不往外传**：返回值仍然是 `(score, text, metadata)` 三元组，**零破坏**。
recency 的效果已经编码在 `score` 里了。要暴露 importance 之类，走 `metadata`。

**小镇那边长这样（写在应用层，不进框架）**：

```python
def memory_stream_scorer(query, vector, *, query_norm, vector_norm,
                         created_at, metadata, **kw):
    relevance = cosine_similarity(query, vector, query_norm, vector_norm)
    hours = (time.time() - created_at) / 3600
    recency = 0.99 ** hours                                  # 指数衰减
    importance = metadata.get("importance", 5) / 10          # LLM 打过分的存这儿
    return 1.0 * recency + 1.0 * importance + 1.5 * relevance
```

> 权重和衰减系数按 Generative Agents 那篇的起步值来，之后按实际效果调。

### 2.3 存储原语：`delete` + `filters`

**为什么这是框架的锅**：应用在现有 API 上**做不到**，不是「麻烦」，是「不可能」。

#### ① `filters`：元数据过滤

今天 `search()` 不接受任何过滤条件。应用只能 `search(top_k=100)` 再在 Python 里筛——
**但这是错的**：`top_k` 是在全库上截断的，筛完之后可能一条不剩，
而真正匹配的文档还躺在第 6 到第 50 名。

**改法**（在 SQL 层过滤，和向量检索同一次扫描）：

```python
def _where_clause(self, filters: Optional[Dict[str, Any]]) -> Tuple[str, list]:
    """把 {k: v} 翻译成 SQL。**等值匹配**, 需要范围查询再加。"""
    if not filters:
        return "", []
    clauses, params = [], []
    for key, value in filters.items():
        # json_extract 需要 sqlite 的 json1 扩展。Python 3.10 自带的 sqlite 有 ——
        # 但这不是保证, 所以第一次用之前先探一下(见下面的 _has_json1)。
        clauses.append(f"json_extract(metadata, '$.{key}') = ?")
        params.append(value)
    return " WHERE " + " AND ".join(clauses), params
```

> ⚠️ **先探一次 json1 是否可用**（本机实测可用，`sqlite_version` 自带的都有）：
> ```python
> try:
>     conn.execute("SELECT json_extract('{\"a\":1}', '$.a')")
>     _HAS_JSON1 = True
> except sqlite3.OperationalError:
>     _HAS_JSON1 = False
> ```
> 不可用时的退化方案：把 `filters` 的键值对在**取回后**筛（会退化，但至少不报错），
> 并且**明确记一条 warning**——静默退化比报错更难查。

> ⚠️ **`key` 必须白名单校验**。它是拼进 SQL 字符串的：
> ```python
> if not key.replace("_", "").isalnum():
>     raise VectorStoreException(f"filters 的键只能是字母数字下划线, 收到 '{key}'")
> ```
> 虽然 sqlite 的 `json_extract` 路径参数是字符串字面量，注入面比看起来小，
> 但**不要靠「看起来安全」**。

#### ② `delete`：按条件删除

今天只有 `clear()`——**全清**。想「只更新变了的那份文档」只能全量重建，连别的 source 一起清掉。

```python
def delete(
    self,
    ids: Optional[Sequence[int]] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> int:
    """
    按 id 或元数据条件删除, 返回删了几条。

    【安全设计: 两个都不传就拒绝】
    不带条件的 delete 和 clear() 长得一样, 但语义完全不同 ——
    前者应该是「我确定要删这些」, 后者是「我要清库」。
    允许无参调用, 就等于给了一次「手滑清库」的机会。
    """
    if ids is None and filters is None:
        raise VectorStoreException(
            "delete 必须给出 ids 或 filters 之一。"
            "要清空整张表请显式调用 clear() —— 它会在日志里留一条 warning。"
        )

    clauses, params = [], []
    if ids is not None:
        ids = list(ids)
        if not ids:
            return 0
        clauses.append(f"id IN ({','.join('?' * len(ids))})")
        params.extend(ids)
    if filters:
        where, p = self._where_clause(filters)
        if where:
            clauses.append(where.replace(" WHERE ", ""))

    sql = f"DELETE FROM {self.table} WHERE " + " AND ".join(clauses)
    with self._lock:
        cur = self._conn.execute(sql, params)
        self._conn.commit()
    logger.info("向量库删除 %d 条 (ids=%s filters=%s)", cur.rowcount, ids, filters)
    return cur.rowcount
```

#### ③ `SemanticMemory` 上的两个薄封装

```python
def forget(self, filters: Dict[str, Any]) -> int:
    """按来源/标签忘掉一批。**这是小镇做『遗忘』的入口。**"""
    return self.store.delete(filters=filters)

def reindex_file(self, path: str, metadata: Optional[dict] = None) -> int:
    """重新入库一个文件: 先按 source 删掉旧的, 再灌新的。
    **这才是『增量入库』—— 没有 delete 就做不了, 只能全量重建。**"""
    source = os.path.basename(str(path))
    self.forget({"source": source})
    return self.ingest_file(path, metadata)
```

> ⚠️ **改完 2.3，0.1 那条红灯记得一起处理**——`_assert_same_model` 的错误文案。
> 顺带把它**行级化**（这是第一份分析里发现的问题）：

### 2.4 顺带修：跨模型守卫要「行级」

**这是本次改造里发现的一个真实缺陷。**

今天 [vector_store.py:277-284](../agents0to1/memory/vector_store.py#L277-L284) 的守卫是**表级**的：
只检查「查询用的模型在库里出现过吗」。而 `search()` 的 SELECT **没有 `WHERE model = ?`**。

**后果**：一个 sqlite 文件里混进两个模型的向量时（换 embedding provider 后复用同一个库就会发生），
用新模型查询会**放行**，并把旧模型的向量一起拉进来打分。**实测**：

```
库里模型: ['openai/text-embedding-3-small', 'dashscope/text-embedding-v3']
查询结果:
   1.0000  乙: 牛顿定律      ← dashscope 自己的
   1.0000  甲: 苹果是水果     ← openai 生成的, 却被打了满分
```

**不报错，只给错结果**——和 [memory-layer-guide.md 踩坑清单第 10 条](memory-layer-guide.md)
写的「宁可不给结果，也不给垃圾」正好相反。

**改法**：把模型过滤并进 `WHERE`：

```python
# search() 里, 和 filters 合并
where_parts, params = [], []
if model is not None:
    where_parts.append("model = ?")          # ← 行级过滤, 不是表级校验
    params.append(model)
```

`_assert_same_model` 保留（它给的是**更清楚的报错**：告诉用户「你换了模型，这个库是另一个模型建的」），
但**必须再加行级过滤**。两者不是二选一。

---

## 第三步：契约层

### 3.1 `LLMResponse` 加 `usage`

**位置**：[typedefs.py:16-25](../agents0to1/core/typedefs.py#L16-L25)

```python
@dataclass
class Usage:
    """一次 LLM 调用的用量。**成本、预算、可观测性三件事的共同地基。**"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        """累加 —— 一次 run() 里可能调很多次 LLM, 你最终想知道的是总数。"""
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.total_tokens + other.total_tokens,
        )
```

```python
@dataclass
class LLMResponse:
    content: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Optional[Usage] = None       # ← 新增
    model: Optional[str] = None         # ← 新增(实际服务的模型, 可能和请求的不一样)
    finish_reason: Optional[str] = None # ← 新增("stop" / "tool_calls" / "length")
```

**填充位置**：`core/llm.py` 里 `invoke` 和 `stream_invoke` 构造 `LLMResponse` 的地方。

```python
usage = getattr(response, "usage", None)
LLMResponse(
    content=...,
    tool_calls=...,
    usage=Usage(
        prompt_tokens=getattr(usage, "prompt_tokens", 0),
        completion_tokens=getattr(usage, "completion_tokens", 0),
        total_tokens=getattr(usage, "total_tokens", 0),
    ) if usage else None,
    model=getattr(response, "model", None),
    finish_reason=...,
)
```

#### ⚠️ 流式路径拿不到 usage，除非显式要求

`stream_invoke`（[core/llm.py:362-375](../agents0to1/core/llm.py#L362-L375)）走的是
`_build_request_params(..., stream=True)`，**没有设 `stream_options`**。
OpenAI 兼容接口默认**不在流里返回 usage**。

**改法**：在流式请求里加上

```python
params["stream_options"] = {"include_usage": True}
```

**但这是个有风险的改动**——不是所有 OpenAI 兼容服务都认这个字段：

| 服务 | `stream_options` |
|---|---|
| OpenAI / DeepSeek | ✅ 认 |
| 一些自建 / 老版本 vLLM | ❓ 可能报 400 |

**所以做成可关的**：

```python
if os.getenv("LLM_STREAM_USAGE", "true").lower() == "true":
    params["stream_options"] = {"include_usage": True}
```

**并且在解析流的时候注意**：开了 `include_usage` 之后，**最后一个 chunk 的 `choices` 是空数组**
（只有 usage，没有 delta）。今天的代码里已经有这一句：

```python
if not chunk.choices:
    continue        # core/llm.py:379-380
```

所以**不会崩**，但 usage 要在 `continue` **之前**取出来，否则会被丢掉。

### 3.2 默认 `stream_run`（已在 1.5 给出）

**位置**：基类。任何实现了 `_run()` 的 agent 自动获得事件流能力。

> ⚠️ **一个真实的递归陷阱。**
>
> 今天 ReAct 的 `run()` 是**消费** `stream_run()` 的事件（[react_agent.py:113-117](../agents0to1/classic_agent/react_agent.py#L113-L117)）：
> ```python
> def run(self, input_text, **kwargs):
>     for event in self.stream_run(input_text, **kwargs): ...
> ```
>
> 而基类的默认 `stream_run()` 如果去调 `self.run()`，**就成环了**。
>
> 改造之后必须保证：**`_stream_run()` 和 `_run()` 之间不能互相调用。**
> ReAct 那边应该是：
> ```python
> def _run(self, input_text, **kwargs):
>     answer = ""
>     for event in self._stream_run(input_text, **kwargs):    # ← 调 _stream_run, 不是 stream_run
>         if event.type == "final": answer = event.answer or ""
>     return answer
> ```
> **`_run` 调 `_stream_run` 是对的（模板方法已开好 ctx，不会重复开）；
> `_run` 调 `stream_run` 就会重开 ctx 并且无限递归。**

### 3.3 `agent_id`

```python
import uuid

# Agent.__init__ 里
self.agent_id: str = agent_id or uuid.uuid4().hex[:12]
```

**进快照**（`snapshot()` 加一个字段），**但 `fork()` 时要换新 id**：

```python
@classmethod
def fork(cls, path, **overrides):
    # fork 出来的必须是**新身份** —— 共享 id 会让两个分支在记忆归属、
    # 消息路由上互相打架(而它们本来就是拿来跑对照实验的)
    overrides.setdefault("agent_id", uuid.uuid4().hex[:12])
    return cls.load(path, **overrides)
```

**为什么小镇需要它**：跨进程、跨会话、记忆归属、消息路由——`name` 只是给人和日志看的标签，
不保证唯一，也不保证稳定。

---

## 第四步：契约测试

**这是框架和应用在测试上的分界线。**

你现在那 92 个用例测的是「**这个实现的具体行为**」，很扎实。框架还要一组测
「**任何实现都该满足的性质**」。

新建 `tests/test_contracts.py`：

```python
"""契约测试 —— 测的不是"这个实现做了什么", 而是"任何实现都该满足什么"。

【为什么框架需要它, 应用不需要】
应用测的是"我的功能对不对", 实现变了测试就该变。
框架测的是"别人按这个接口写的东西能不能跑", 实现变了**契约不能变**。

所以这里的每个用例, 都应该能在**任意**一个符合接口的实现上通过。
"""
```

必测清单：

| 契约 | 怎么测 |
|---|---|
| **任何 memory 对象**（只需 `search()` 或 `build_context()`）挂进 agent 都不会弄挂它 | 三个假实现：`OnlySearch` / `OnlyBuildContext` / `Both`，各跑一轮，断言都跑完 |
| **任何 hook 抛异常**，`on_error="open"` 时 agent 行为与没挂时**逐字节相同** | 一个 `AlwaysBoom` hook，对比 `FakeLLM.calls` |
| **任何 Agent 子类**，`stream_run` 的事件顺序合法 | 遍历四个 Agent，断言事件序列满足 `thinking?/text* → tool_* → final` |
| **任何 VectorStore 实现**，`add` 之后 `count()` 增加、`search` 能找回 | 传一个自己写的假 store 进去（这一步依赖 2.1 的缝） |
| **任何 scorer**，返回的排序就是最终排序 | 传一个「按字符串长度打分」的 scorer，断言顺序 |
| **snapshot → load 往返**，所有公开状态一致 | 遍历四个 Agent |

> ⚠️ **契约测试必须对「实现」不敏感。** 如果你发现某个契约用例需要改实现才能过，
> 那要么是契约定错了，要么是实现真的违约了——**两者都值得停下来想清楚，
> 而不是把测试改成能过。**

---

## 第五步：扩展点地图

**框架的文档不是「怎么用」，是「在哪里挂东西」。**
这是框架文档和应用文档的分界线。

新建 `docs/extension-map.md`，一页就够：

```markdown
# 扩展点地图

| 我想加的东西 | 挂在哪 | 写几个文件 | 参考 |
|---|---|---|---|
| 记忆注入 | `Hook.before_llm` | 1 | `agents0to1/hooks/memory.py` |
| 世界感知 | `Hook.before_input` | 1 | — |
| 对话落库 | `Hook.after_run` | 1 | `EpisodicMemory.save_agent` |
| 成本统计 | `Hook.after_llm` + `LLMResponse.usage` | 1 | — |
| 成本熔断 | `Hook.after_llm` + `on_error="closed"` | 1 | — |
| 工具结果脱敏 | `Hook.after_tool` | 1 | — |
| 换向量库 | `SemanticMemory(store=...)` | 0 | — |
| 换检索打分 | `VectorStore.search(scorer=...)` | 0 | 2.2 |
| 元数据过滤 | `VectorStore.search(filters=...)` | 0 | 2.3 |
| 增量入库 | `SemanticMemory.reindex_file` | 0 | 2.3 |
| 文档解析(pdf/docx) | 自己解析完再 `SemanticMemory.remember(text)` | 1 | — |
| 加一种新 Agent 范式 | 继承 `Agent`，实现 `_run()` | 1 | — |
```

**为什么不写实现**：解析器、父子块、混合检索、rerank 都是**应用层**。
框架只保证「挂得上」，不保证「替你挂」。

---

## 附录 A：决策表

最有价值的部分——**每个选择，被否掉的方案是什么，为什么否掉**。

| 决策 | 选了 | 否掉了 | 为什么 |
|---|---|---|---|
| 扩展点机制 | hook 管线 | 继续加基类参数 | 加记忆层动了 9 个文件；第 N 种能力成本是 O(所有子类) |
| 扩展点机制 | hook 管线 | 事件总线 / 发布订阅 | 需要「改」而不只是「听」——`before_llm` 要能改消息 |
| 扩展点机制 | hook 管线 | 中间件洋葱模型 | 洋葱模型适合「请求-响应」，agent 一轮里有 N 次 LLM 调用，形状不匹配 |
| hook 参数形态 | `RunContext` 对象 | 把参数摊平进签名 | 摊平的每个参数都是未来会破的兼容性承诺 |
| 查询来源 | `ctx.input_text` | `messages[-1]["content"]` | ReAct 第 2 步起最后一条是 tool 消息（memory-layer-guide 5.2） |
| 失败策略 | 每个 hook 自己声明 | 框架统一决定 | 框架没资格替应用决定「这个 hook 挂了要不要中断对话」 |
| hook 可变性 | 必须返回新对象 | 允许原地改 | 原地改会污染 `last_messages` 和 Turn，正是今天 `_record_first_message` 在补的洞 |
| ctx 生命周期 | 模板方法（基类管） | 子类自己记得调 | 「必须记得做的事」是 1.1 那张表的病根，不能再加一条 |
| 检索打分 | 可传 `scorer` | 焊死余弦 | 文档系统要 relevance-only，小镇要三因子；**两者冲突，框架不该选** |
| `created_at` | 只传给 scorer，不外传 | 加进返回值 | 零破坏——现有多处调用不用改 |
| 存储过滤 | SQL 层 `WHERE` | 取回来在 Python 筛 | top_k 是先截断的，筛完可能一条不剩，是**错**的 |
| `delete` 无参 | **拒绝** | 允许（等同 clear） | 给一次「手滑清库」的机会 |
| 跨模型守卫 | 行级 `WHERE model=?` | 表级校验 | 表级只在「模型完全不存在」时拦，混库时给满分垃圾 |
| 流式 usage | `stream_options` + 可关 | 硬加 | 不是所有 OpenAI 兼容服务都认，可能 400 |
| `after_llm` 流式 | 观察 only | 也允改 | chunk 早吐出去了，改不了——**不对称只能文档化** |
| `agent_id` | 进快照 | 不进 | 身份要跨会话稳定 |
| `fork` 的 id | 换新的 | 沿用 | fork 是跑对照实验的，共享 id 会让两个分支互相打架 |
| 向量库换 faiss | 留缝不换 | 现在就换 | 手搓的够用几千条，过早优化 |

---

## 附录 B：踩坑清单

写代码时对照这张表，能省你几个小时：

1. **`hooks` 参数必须加在参数表最后** —— 四个子类用位置参数调 `super().__init__`，
   插在前面会让 `config` 静默绑到 `hooks` 上，**不报错**
2. **`_chat` 和 `_stream_chat` 是两处** —— `_stream_chat` 不经过 `_chat`，
   钩子挂一处 = 流式路径静默失效（这个坑本项目已经踩过一次）
3. **`before_llm` 不得原地改入参** —— 会污染 `last_messages` 和 Turn
4. **`_run` 调 `_stream_run`，不调 `stream_run`** —— 后者会重开 ctx 并无限递归
5. **`after_tool` 必须在 `messages.append` 之前** —— 否则事件里的 result 是旧值，两份真相
6. **`ctx.input_text` 不该被 `before_input` 改写** —— 它是 hook 的检索依据
7. **`on_error="closed"` 是给「必须中断」的** —— 成本熔断、权限校验、内容合规；
   记忆/trace 一律 `"open"`
8. **`llm_calls` 判据对 PlanSolve / Reflection 不成立** —— 它们一轮里有 N 次「第一次」，
   用 `ctx.state` 标记
9. **流式 `after_llm` 改不了已经吐出去的字符** —— 只能观察，不对称必须文档化
10. **`stream_options` 不是所有服务都认** —— 做成环境变量可关，默认开
11. **开了 `include_usage` 后最后一个 chunk 的 `choices` 是空的** ——
    usage 要在 `if not chunk.choices: continue` **之前**取
12. **`filters` 的 key 要白名单校验** —— 它是拼进 SQL 的，别靠「看起来安全」
13. **`json_extract` 需要 sqlite 的 json1 扩展** —— 本机实测可用，但要探测 + 退化 + warning
14. **`delete()` 无参必须拒绝** —— 否则和 `clear()` 只剩一个手滑的距离
15. **跨模型守卫要行级** —— 表级守卫在混库时会给满分垃圾，且不报错
16. **`fork()` 要换新 `agent_id`** —— 否则两个分支在记忆归属上互相打架
17. **跑测试前先 `conda activate agent`** —— 新 shell 里的 `python` 是 3.9，
    报的是看不懂的 `ModuleNotFoundError`
18. **离线测试不许吃 `.env`** —— 否则「不依赖 API key」这条保证会被**静默**破坏
    （已经发生了一次：`test_snapshot_roundtrip_without_memory`）

---

## 附录 C：验收清单

改造完成的判据，**逐条打勾**：

**第零步**
- [ ] `python tests/run_offline.py` → **4/4 全绿**，退出码 0
- [ ] 换个工作目录跑 `test_agent_memory.py` → 仍然全绿（`.env` 隔离生效）
- [ ] `_harness` import 之后 `REPO_ROOT in sys.path` → **True**
- [ ] `tests/` 里有文件 import 了 `agents0to1.memory.episodic`
- [ ] 没有零断言还报 PASS 的测试
- [ ] `.gitignore` 有 `data/`
- [ ] `.env.example` 有 `EMBEDDING_*`
- [ ] 全部提交，分多个 commit

**第一步**
- [ ] 摘掉 `MemoryHook` → agent 完全不知道记忆层存在
- [ ] 重新挂上 → **只加一个文件，框架里零改动**
- [ ] 原 `test_agent_memory.py` 33 个用例迁移后全绿
- [ ] `on_error` 两种策略各有一个用例覆盖
- [ ] 四个 Agent 都能 `stream_run`（基类默认实现）

**第二步**
- [ ] `SemanticMemory(store=自己的假实现)` 能跑
- [ ] `search(scorer=...)` 换打分公式，不用改框架
- [ ] `search(filters=...)` 在 SQL 层过滤
- [ ] `delete(filters=...)` / `delete(ids=...)` 能用，无参报错
- [ ] 混库（两个 model）时用其中一个查询，**旧模型的向量不参与打分**

**第三步**
- [ ] `LLMResponse.usage` 在非流式路径有值
- [ ] 流式路径（开了 `stream_options`）也有值
- [ ] `snapshot()['agent_id']` 存在，`fork()` 出来的是新 id

**第四步**
- [ ] `tests/test_contracts.py` 存在，且**不需要改任何实现**就能过

**第五步**
- [ ] `docs/extension-map.md` 存在，一页能看完

---

## 建议的提交顺序

每一步一个 commit，都能单独跑、单独回退：

```
0. 清红灯: 两处测试修正 + _harness 路径 + episodic 测试 + 空测试   (不需要 key)
   ├─ 0a. 修 test_cross_model_search_rejected 的文案
   ├─ 0b. 修 .env 泄漏 + 加环境隔离
   ├─ 0c. 修 _harness._bootstrap_path 自删
   ├─ 0d. 补 tests/test_episodic.py
   ├─ 0e. 拆两条零断言的测试
   └─ 0f. .gitignore + .env.example
1. core/hooks.py: Hook + HookPipeline + RunContext              (不需要 key)
2. core/agent.py: 埋点 + 模板方法 + 默认 stream_run              (不需要 key)
3. 迁移记忆层到 MemoryHook, 删掉基类里的记忆代码                  (不需要 key)
4. 契约测试 test_contracts.py                                    (不需要 key)
5. semantic.py: store 可注入                                     (不需要 key)
6. vector_store.py: 行级模型过滤 + filters + delete + scorer      (不需要 key)
7. typedefs.py + llm.py: Usage                                   (不需要 key)
8. agent_id + fork 换 id                                         (不需要 key)
9. docs/extension-map.md
```

**每一步做完都跑一次全部离线测试**——这是你能坚持下来的唯一原因。

> ⚠️ 注意：这份清单里**没有一个步骤需要 API key**。
> 这不是巧合——**框架改造的每一处，都应该能用假 LLM / 假 embedder 验证。**
> 如果哪一步非联网不可，说明它的边界划错了。
