"""知识检索工具 —— 语义记忆的**工具版**接法

这就是 SemanticMemory.search() 的一层薄封装, 走标准 Tool 接口
ReActAgent 立刻能用。

【但它的真正价值不是"能用", 是让你亲身感受到工具版的三个局限】
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..base import Tool, ToolParameter
from ...utils.logging import get_logger

if TYPE_CHECKING:
    # 只为类型标注导入, 运行期不 import —— tools/ 不该对 memory/ 产生硬依赖,
    # 也不给循环导入留机会。
    from ...memory.semantic import MemoryItem, SemanticMemory

logger = get_logger(__name__)


DEFAULT_KNOWLEDGE_DESCRIPTION = (
    "在本地知识库(用户提供的文档、资料、之前记下的内容)里做语义检索。"
    "当你需要回答的问题可能和这些资料有关时, 用它查一下再回答 —— "
    "**查不到就直说没查到, 不要编**。"
)


class KnowledgeSearchTool(Tool):
    """
    语义记忆检索工具。

    用法:
        mem = SemanticMemory()
        registry.register_tool(KnowledgeSearchTool(mem))
        agent = ReActAgent("a", llm, tool_registry=registry, hooks=[MemoryHook(mem)])
        #                                                        ^^^^^^^^^^^^^^^^
        # 两个都挂上是有意的: 工具版是"模型主动去查", 上下文版是"每次都自动带上"。
        # 只挂工具版, 你会亲眼看到"模型有时候想不起来查"是什么样子。

    【截断为什么必须在这里做】
    ToolRegistry.execute 对结果做 truncate_output(registry.py:86), 默认上限
    max_output_chars = 6000(base.py:20)。检索回来好几个块很容易超过它。
    而 truncate_output 是"保头 + 保尾 _TAIL_CHARS=800", 它会**把最后一块的尾巴
    接到前面去** —— 相关性顺序就乱了, 而顺序是检索结果的灵魂。

    所以: 在 run() 内部就按预算裁好, 让外层无东西可截。
    而且这样还能保证**工具路径和上下文路径用同一个预算常量**
    (memory.context_budget), 模型不会看到同一个库的两种形状。
    """

    def __init__(
        self,
        memory: "SemanticMemory",
        top_k: int = 5,
        name: str = "knowledge_search",
        description: Optional[str] = None,
    ):
        """
        Args:
            memory:      SemanticMemory 实例
            top_k:       每次检索几条
            name:        工具名。改了记得同步提示词里对它的称呼
            description: 工具描述 —— **这是模型"想不想得起来调"的唯一依据**,
                         值得多花十分钟打磨
        """
        if not hasattr(memory, "search"):
            raise TypeError(
                f"KnowledgeSearchTool 需要一个提供 search() 的记忆对象, "
                f"收到的是 {type(memory).__name__}。"
                f"(情景记忆 EpisodicMemory 是按时间回放的, 它没有 search —— "
                f"那个应该用 hooks=[MemoryHook(mem)] 挂给 Agent, 而不是做成工具)"
            )
        super().__init__(name=name, description=description or DEFAULT_KNOWLEDGE_DESCRIPTION)
        self.memory = memory
        self.top_k = top_k

    def run(self, parameters: Dict[str, Any]) -> str:
        """
        执行检索。

        【失败返回字符串, 不抛异常】和 registry.execute 的哲学一致:
        工具失败是**正常流程的一部分**, 错误本身会作为 tool 消息回传给模型,
        模型看了能自己换个问法重试。抛出去的话整轮就断了。
        """
        query = (parameters.get("query") or "").strip()
        if not query:
            return "错误:查询内容不能为空。请给出你想查的关键词或问题。"

        logger.info("知识库检索: %s", query)

        try:
            items = self.memory.search(query, top_k=self.top_k)
        except Exception as e:
            return f"错误:知识库检索失败: {e}"

        if not items:
            # 明确说"没查到"而不是返回空串: 空串会被模型理解成"工具坏了",
            # 于是它可能换个工具乱试, 或者干脆装没看见
            return "知识库里没有找到相关内容。可以换个说法再查一次, 或者直接回答并说明没有资料依据。"

        # 和上下文路径共用同一个排版 + 同一个预算
        budget = getattr(self.memory, "context_budget", None)
        return self.memory.format_items(items, budget=budget)

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                type="string",
                description="检索用的自然语言问题或关键词。用完整的问题比用零散的关键词效果更好。",
                required=True,
            )
        ]
