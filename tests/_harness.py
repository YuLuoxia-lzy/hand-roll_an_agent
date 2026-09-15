"""测试脚手架 —— 路径修正 + 迷你运行器 + 假 embedder / 假 LLM

【为什么要有这个文件而不是直接用 pytest】
这个项目最值钱的一条洁癖是"硬依赖只有 openai + pydantic", 连 numpy 都只做可选。
为了跑测试再塞一个 pytest 进环境, 就把这条洁癖破了。所以这里用 40 行写了个迷你
运行器: 测试函数一律命名为 test_*, 所以**将来装了 pytest 也能直接 pytest tests/ 跑**,
两种方式都行。

用法(每个测试文件结尾):
    if __name__ == "__main__":
        sys.exit(run_tests(globals(), "向量库"))

【那个必须先解决的坑: import 到的是哪个 agents0to1?】
本机 `pip install -e` 装的是**原项目**(hand-roll_an_agent), 不是这个副本。
而 `python tests/test_x.py` 时 sys.path[0] 是 tests/ 目录 —— 于是
`import agents0to1` 会拿到原项目的包, 你改副本、测原项目, **而且完全不报错**,
只会觉得"我明明改了啊怎么没生效"。

所以 _bootstrap_path() 把副本根目录插到 sys.path 最前面, 保证测的一定是副本。
"""

import copy
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

#: 副本根目录(本文件在 <根>/tests/_harness.py, 所以往上两级)
REPO_ROOT = Path(__file__).resolve().parent.parent


def _bootstrap_path() -> None:
    """把副本根目录放到 sys.path 最前面 —— 见模块开头那段说明。"""
    root = str(REPO_ROOT)
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)

    # 顺带把原项目从 sys.path 里挤出去, 免得"副本没改对"时静默用了原项目的代码
    # (editable 安装的路径会指向 .../hand-roll_an_agent)
    def _is_original_project(path: str) -> bool:
        # 只认 ".../hand-roll_an_agent" 结尾的路径 —— 副本叫 ".../hand-roll_an_agent copy",
        # 结尾对不上, 不会被误伤
        return path.replace("\\", "/").rstrip("/").endswith("/hand-roll_an_agent")

    sys.path[:] = [p for p in sys.path if p and not _is_original_project(p)]


_bootstrap_path()

# Windows 控制台默认不是 UTF-8, 中文断言失败信息会变乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ==================== 迷你运行器 ====================

class SkipTest(Exception):
    """
    用例自己判定"这个环境跑不了" —— 记为 SKIP, **不算失败**。

    只该用在"缺外部条件"上: 没有 API key、本地模型没 pull、网络不通。
    **绝不能用它掩盖失败** —— 那样测试就变成了装样子, 比没有还坏。
    """


def skip(reason: str):
    """在用例里调用: 条件不满足就跳过后面的断言"""
    raise SkipTest(reason)


def run_tests(namespace: Dict[str, Any], title: str) -> int:
    """
    跑 namespace 里所有 test_* 函数, 返回进程退出码(0 = 全过)。

    namespace 一般直接传 globals()。只收本模块自己定义的函数 ——
    否则会把 import 进来的 test_ 函数也跑一遍(然后跑两遍)。
    """
    module_name = namespace.get("__name__")
    cases = [
        (name, obj)
        for name, obj in namespace.items()
        if name.startswith("test_") and callable(obj)
        and getattr(obj, "__module__", None) == module_name
    ]

    print(f"\n{'=' * 60}\n{title}  ({len(cases)} 个用例)\n{'=' * 60}")
    failures: List[str] = []
    skipped: List[str] = []
    for name, fn in cases:
        try:
            fn()
            print(f"  [PASS] {name}")
        except SkipTest as e:
            skipped.append(name)
            print(f"  [SKIP] {name}")
            print("         " + str(e).replace("\n", "\n         "))
        except Exception as e:
            failures.append(name)
            print(f"  [FAIL] {name}")
            print("         " + str(e).replace("\n", "\n         "))
            if os.getenv("TEST_VERBOSE"):
                traceback.print_exc()

    ran = len(cases) - len(skipped)
    print(f"\n{'-' * 60}")
    if failures:
        print(f"{title}: {ran - len(failures)}/{ran} 通过, 失败 {len(failures)} 个: {failures}"
              + (f" (另有 {len(skipped)} 个跳过)" if skipped else ""))
        return 1
    print(f"{title}: {ran}/{ran} 全部通过" + (f", 跳过 {len(skipped)} 个" if skipped else ""))
    return 0


# ==================== 临时目录 ====================

class TempDir:
    """with TempDir() as d: ... —— 退出时整棵删掉, 不留测试垃圾在仓库里"""

    def __enter__(self) -> Path:
        self._path = Path(tempfile.mkdtemp(prefix="agents0to1-test-"))
        return self._path

    def __exit__(self, *exc):
        # Windows 上 sqlite 没关干净时 rmtree 会失败 —— 测试垃圾删不掉不该算失败
        shutil.rmtree(self._path, ignore_errors=True)


# ==================== 假 embedder ====================

class FakeEmbedder:
    """
    假 embedder —— 文本 -> 固定长度向量。**不花钱、不联网、结果可复现。**

    做法是"字符 trigram 的哈希袋"(hashing trick):
        相同文本   -> 相同向量            ✓ 指南要求
        不同文本   -> 不同向量            ✓ 指南要求
        有重叠的文本 -> 相似度高            ← 比纯哈希多这一条

    为什么要多那一条: 纯哈希向量两两之间几乎正交(相似度都接近 0), 于是
    "最像的排最前"这种排序测试根本测不出东西 —— 全是 0, 谁排前面都对。
    trigram 重叠让向量带上真实的语义梯度, 排序才可验证。

    【一个容易踩的坑】用 Python 内置的 hash() 是不行的: 字符串的 hash 每个进程
    都不一样(PYTHONHASHSEED 随机化)。那样"这个进程入库、下个进程查询"就会
    算出完全不同的向量, 库直接废掉。所以这里用 zlib.crc32 —— 稳定。
    """

    def __init__(self, dim: int = 64, fail: bool = False):
        self._dim = dim
        self.fail = fail
        self.embed_calls: List[List[str]] = []      # 记录每次批量请求的条数

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_id(self) -> str:
        return f"fake/trigram-{self._dim}"

    def _vector(self, text: str) -> List[float]:
        import zlib

        vec = [0.0] * self._dim
        # 加边界符, 让"开头/结尾"也参与特征
        padded = f"^{text}$"
        for i in range(len(padded) - 2):
            gram = padded[i:i + 3]
            index = zlib.crc32(gram.encode("utf-8")) % self._dim
            vec[index] += 1.0

        norm = sum(x * x for x in vec) ** 0.5
        if norm == 0:            # 空文本 -> 全零向量
            return vec
        return [x / norm for x in vec]

    def embed(self, texts: List[str]) -> List[List[float]]:
        if self.fail:
            raise RuntimeError("FakeEmbedder 被要求失败 (fail=True)")
        self.embed_calls.append(list(texts))
        return [self._vector(t) for t in texts]

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]

    def __repr__(self) -> str:
        return f"FakeEmbedder(dim={self._dim}, fail={self.fail})"


# ==================== 假 LLM ====================

class FakeLLM:
    """
    假 LLM —— 按剧本回话, 并记下**每一次请求的完整 messages**。

    "记下请求"是这一整套测试的关键: 验证"记忆有没有被注入"靠的不是模型说了什么,
    而是**我们到底发了什么给模型**。

    剧本(script)每一项是:
        "最终答案"                     -> 直接给答案, 不带工具调用
        (content, [ToolCall, ...])     -> 先说要调工具, 带正文

    【记下来的 messages 必须**深拷贝**, 不能直接存引用】
    ReAct 是拿着**同一个列表**在循环里不断 append(react_agent.py:132 建列表,
    之后 _append_assistant_message / _execute_tool_calls 往里追加)。
    存引用的话, calls[0] 和 calls[1] 指向的是同一个列表 —— 等 run() 结束再看,
    它们都变成了"最后那一刻"的内容: 你以为在看第 1 步的请求, 其实是第 3 步的。
    于是"第 2 步末尾是 tool 还是 user"这类断言全部失真, 而且**看起来还挺合理**,
    极难发现(这个错在第一版里真的犯了, 断言报出来的是一条 assistant 消息)。
    """

    provider = "fake"
    model = "fake-model"
    base_url = "http://fake.invalid/v1"
    temperature = 0.7
    max_tokens = None

    def __init__(self, script: Optional[List[Any]] = None):
        self.script = list(script or ["好的。"])
        self.calls: List[Dict[str, Any]] = []       # 每次请求的 (messages, tools), 已深拷贝
        self.last_response = None
        self._cursor = 0

    # --- 供 Agent 用 ---

    def invoke(self, messages, tools=None, **kwargs):
        from agents0to1.core.typedefs import LLMResponse

        self.calls.append({
            "messages": copy.deepcopy(messages), "tools": tools, "stream": False, "kwargs": kwargs,
        })
        response = self._next_response()
        self.last_response = response
        return response

    def stream_invoke(self, messages, tools=None, **kwargs) -> Iterator[str]:
        from agents0to1.core.typedefs import LLMResponse

        self.calls.append({
            "messages": copy.deepcopy(messages), "tools": tools, "stream": True, "kwargs": kwargs,
        })
        response = self._next_response()

        # 逐块吐正文 —— 模拟真实的流式, 顺便验证"fail-open 时不能多吐字符"
        for chunk in self._chunks(response.content or ""):
            yield chunk

        # 必须在生成器**跑完之后**才挂上去, 和真客户端一致(core/llm.py:407)
        self.last_response = response

    def _next_response(self):
        """按剧本给出下一条响应。剧本跑完了就重复最后一条。"""
        from agents0to1.core.typedefs import LLMResponse

        item = self.script[min(self._cursor, len(self.script) - 1)]
        self._cursor += 1

        if isinstance(item, str):                    # 直接给答案
            return LLMResponse(content=item, tool_calls=[])
        if isinstance(item, tuple):                  # (正文, [工具调用])
            content, calls = item
            return LLMResponse(content=content or None, tool_calls=list(calls))

        raise TypeError(f"剧本项只能是 str 或 (content, calls), 收到 {type(item).__name__}")

    @staticmethod
    def _chunks(text: str, size: int = 3) -> Iterator[str]:
        for i in range(0, len(text), size):
            yield text[i:i + size]

    # --- 断言辅助 ---

    @property
    def last_messages(self) -> List[dict]:
        return self.calls[-1]["messages"]

    def reset(self):
        self.calls.clear()
        self._cursor = 0

    def __repr__(self) -> str:
        return f"FakeLLM(已调用 {len(self.calls)} 次)"
