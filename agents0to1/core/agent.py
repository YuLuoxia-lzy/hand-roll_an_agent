"""
Agent基类
用于后续创建自己的agent
可以在此地新增各种新功能
在具体agent处去丰富各种功能
"""


import copy
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Optional
from .message import Message, Turn
from .llm import Agents0to1
from .config import Config
from .typedefs import AgentEvent, LLMResponse
from ..utils.logging import get_logger

logger = get_logger(__name__)


class Agent(ABC):
    """Agent基类"""

    def __init__(
            self,
            name: str,
            llm: Agents0to1,
            system_prompt: Optional[str] = None,
            config: Optional[Config] = None,
            memory: Optional[object] = None
            ):
        self.config = config or Config()
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt or self.config.system_prompt
        self._turns: list[Turn] = []
        # self._history : list[Message] = [] #这是内部使用的history

        # 最近一次【流式】调用的完整响应。存在 Agent 上而不是只读 llm.last_response:
        # 同一个 llm 客户端可能被多个 Agent 共用(llm.last_response 里放的是"最后一次流式调用"的结果, 不区分是哪个 Agent 发起的), 存一份在自己的属性上才不会串台。
        self._last_response: Optional[LLMResponse] = None

        # ==================== 记忆 ====================
        #
        # 四个子类都是用**位置参数**调 super().__init__(name, llm, system_prompt, config)
        # 要是把 memory 插在 config 前面, 所有位置调用会把 config 静默绑到 memory上又是一个不报错的错误。
        #
        # 【类型守卫】memory 的合法形状:
        #     search()        —— 语义记忆, 按相似度检索
        #     build_context() —— 情景记忆, 按时间回放 (它没有 search)
        if memory is not None and not (
            hasattr(memory, "search") or hasattr(memory, "build_context")
        ):
            raise TypeError(
                f"memory 需要提供 search() 或 build_context() 方法, "
                f"收到的是 {type(memory).__name__}"
            )
        self.memory = memory

    #声明必须实现这个方法！！
    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        """运行Agent"""
        pass

    def stream_run(self, input_text: str, **kwargs) -> Iterator[AgentEvent]:
        """
        流式运行Agent, 逐块吐出 AgentEvent。

        基类不提供实现(不是所有 Agent 都适合流式), 但把它声明出来,
        """
        raise NotImplementedError(f"{type(self).__name__} 暂不支持流式运行。")

    # ==================== 拼消息 ====================

    def _system_content(self) -> str:
        """
        system 消息的内容。**子类可覆盖**, 用来追加自己的工作方式说明。
        """
        return self.system_prompt

    def _base_messages(self) -> list[dict]:
        """只含 system 的那一条消息(没有 system_prompt 时返回空列表)"""
        content = self._system_content()
        return [{"role": "system", "content": content}] if content else []

    def _history_messages(self) -> list[dict]:
        """
        把历史摊平成 API 消息格式: 一轮接一轮地展开每一轮的上行消息。
        """
        flattened: list[dict] = []
        for turn in self._turns:
            flattened.extend(turn.messages)
        return flattened

    def _build_messages(self, input_text: str) -> list[dict]:
        """
        拼入口消息: system + 历史 + 当前问题(有记忆时, 检索结果并进这条 user 里)。
        """
        return (
            self._base_messages()
            + self._history_messages()
            + [self._prepare_user_message(input_text)]
        )

    # ==================== 记忆注入 ====================

    def _prepare_user_message(self, input_text: str) -> dict:
        """
        组装最后那条 user 消息。有记忆时把检索结果**并进同一条消息**。
        """
        message = {"role": "user", "content": input_text}
        if self.memory is None:
            return message

        context = self._memory_context(input_text)
        if not context:
            return message
        return {"role": "user", "content": f"{context}\n\n{input_text}"}

    def _record_first_message(self, input_text: str) -> dict:
        """
        进 Turn 的第一条消息  必须是原始输入。
        _prepare_user_message 把检索结果并进了这条 user 消息, 但那份内容只该活在
        """
        return {"role": "user", "content": input_text}

    def _memory_context(self, query: str) -> str:
        """
        取记忆上下文。**fail-open 兜底: 任何异常都退化成"没有记忆"。**
        """
        if self.memory is None:
            return ""

        try:
            build = getattr(self.memory, "build_context", None)
            if callable(build):
                return build(query) or ""

            # 只提供 search() 的记忆对象: 由 agent 负责排版,
            # 这样它和 KnowledgeSearchTool 走的是同一个 format_items, 形状一致
            items = self.memory.search(query)
            if not items:
                return ""
            format_items = getattr(self.memory, "format_items", None)
            if callable(format_items):
                return format_items(items)
            return "\n\n".join(f"[{i}] {item.text}" for i, item in enumerate(items, 1))
        except Exception as e:
            logger.warning("记忆检索失败, 本轮按『没有记忆』继续: %s", e)
            return ""

    def _ensure_system(self, messages: list[dict]) -> list[dict]:
        """
        兜底: 消息列表第一条不是 system, 就在最前面补一条。
        """
        if messages and messages[0].get("role") == "system":
            return messages
        return self._base_messages() + list(messages)

    # ==================== 调 LLM ====================

    def _chat(self, messages: list[dict], tools=None, **kwargs) -> LLMResponse:
        """
        所有 LLM 调用的唯一出口。
        """
        return self.llm.invoke(self._ensure_system(messages), tools=tools, **kwargs)

    def _stream_chat(self, messages: list[dict], tools=None, **kwargs) -> Iterator[str]:
        """
        流式版本, 同样强制注入 system。
        """
        yield from self.llm.stream_invoke(self._ensure_system(messages), tools=tools, **kwargs)

        # 只在生成器被完整消费后才会执行到这里 —— 提前 break 的调用方拿不到新值,
        # 这正是"流式调用的固有语义", 不是 bug。
        self._last_response = self.llm.last_response

    # ==================== 历史 ====================
    #
    # 历史 = list[Turn], **以"轮"为原子单位**。
    #
    # 两条写入路径:
    #   add_turn()    完整轨迹(含工具调用) —— ReAct 走这条, 存原始 messages 副本
    #   add_message() 单条消息           —— 老接口, 内部包成一个只含 1 条消息的 Turn

    def add_turn(
        self,
        input_text: str,
        messages: list[dict],
        answer: str = "",
        metadata: Optional[dict] = None,
    ):
        """
        记录完整的一轮: 用户输入 + 本轮全部上行消息 + 最终答案。

        Args:
            input_text: 本轮的用户输入(仅作展示/检索用, 不参与拼消息 ——
                        它本来就在 messages 里, 拼的时候别再加一遍)
            messages:   本轮发给模型的消息序列, 可含 system(会被剥掉)
            answer:     最终答案
            metadata:   步数/用过的工具等
        """
        # 1. 剥掉 system: system 是 Agent 的属性而不是对话内容。
        first = 0
        while first < len(messages) and messages[first].get("role") == "system":
            first += 1

        # 2. 深拷贝。ReAct 的 messages 是循环里不断 append 的**活列表**,
        snapshot = copy.deepcopy(messages[first:])

        turn = Turn(user=input_text, messages=snapshot, answer=answer, metadata=metadata or {})

        # 3. 自检只是烟雾报警器, 不阻断流程 —— 轨迹不合法时记一条 warning,
        if not turn.is_valid():
            logger.warning(
                "本轮轨迹配对异常(assistant 的 tool_calls 与 tool 消息对不上), "
                "已照常记录, 但下一轮模型收到的历史可能不可用。"
            )

        self._turns.append(turn)
        self._truncate_history()

    @staticmethod
    def _turn_chars(turn: Turn) -> int:
        """
        这一轮实际会发给模型的字符量。
        只数 messages —— turn.user / turn.answer 的内容本来就在 messages 里,
        单独再加一遍是重复计数。
        """
        total = 0
        for m in turn.messages:
            total += len(m.get("content") or "")
            for tc in m.get("tool_calls") or []:
                total += len(json.dumps(tc, ensure_ascii=False))
        return total

    @staticmethod
    def _compact_turn(turn: Turn, keep: int) -> bool:
        """
        把这一轮里超长的工具结果压短, 返回是否真的改了。
        只动 tool 消息: 它保留了 tool_call_id, 配对不受影响(配对看 id 不看内容)。
        """
        changed = False
        for m in turn.messages:
            if m.get("role") != "tool":
                continue
            content = m.get("content")
            if not isinstance(content, str) or len(content) <= keep:
                continue
            m["content"] = (
                f"{content[:keep]}"
                f"\n...[已压缩, 省略 {len(content) - keep} 字符; 需要细节请重新调用该工具]..."
            )
            changed = True
        return changed

    def _truncate_history(self):
        """
        三重预算, 从最新往回保留: 轮数 × 条数(切窗口) -> 字符数(先压缩再丢弃)。
        至少保住最近一轮 —— 哪怕它自己就超了预算, 也不能把历史清空。
        """
        kept: list[Turn] = []
        used = 0
        limit = self.config.max_history_length
        for turn in reversed(self._turns):
            size = len(turn.messages)
            if kept and used + size > limit:
                break
            kept.append(turn)
            used += size
            if len(kept) >= self.config.max_history_turns:
                break
        kept.reverse()

        char_limit = self.config.max_history_chars
        if char_limit > 0:
            total = sum(self._turn_chars(t) for t in kept)
            if total > char_limit:
                # kept[-1] 是本轮
                for turn in kept[:-1]:
                    if total <= char_limit:
                        break
                    before = self._turn_chars(turn)
                    if self._compact_turn(turn, self.config.compacted_tool_chars):
                        total -= before - self._turn_chars(turn)
                        logger.info("历史超预算, 已压缩旧轮的工具结果(当前约 %d 字符)。", total)

                # 压缩是丢细节, 丢轮是丢"我问过什么、答过什么" —— 后者损失大得多,
                dropped = 0
                while len(kept) > 1 and total > char_limit:
                    total -= self._turn_chars(kept[0])
                    kept.pop(0)
                    dropped += 1
                if dropped:
                    logger.warning(
                        "压缩后仍超出字符预算 %d, 已丢弃最旧的 %d 轮完整对话。",
                        char_limit, dropped,
                    )

        self._turns = kept

    def add_message(self, message: Message):
        """添加单条消息 —— 老接口, 内部包成一个只含 1 条消息的 Turn"""
        self.add_turn("", [message.to_dict()])

    def _record_turn(self, input_text: str, output_text: str, messages: Optional[list[dict]] = None):
        """
        一轮收尾统一调它。

        messages 传了 -> 存完整轨迹(含工具调用), ReAct 走这条
        messages 没传 -> 退化成"只含 user + assistant 两条的 Turn", 行为与旧版一致
        """
        if messages is not None:
            self.add_turn(input_text, messages, answer=output_text)
            return

        self.add_turn(
            input_text,
            [
                Message(input_text, "user").to_dict(),
                Message(output_text or "", "assistant").to_dict(),
            ],
            answer=output_text,
        )

    def clear_history(self):
        """清空历史记录"""
        self._turns.clear()

    def get_history(self) -> list[Message]:
        """
        获取历史 —— 扁平化的 Message 列表(不含工具调用的中间过程)。
        """
        return [Message(m["content"], m["role"]) for turn in self._turns for m in turn.messages]

    def get_turns(self) -> list[Turn]:
        """
        获取完整轨迹 —— 每一轮的工具调用过程都在里面。
        """
        return list(self._turns)

    # ==================== 快照 ====================
    #
    # 存的是**轨迹**
    #
    # 下面几处类型注解用的是 PEP 604 的 str | Path, 这是 3.10 的语法。
    # 函数注解在 def 执行时求值, 所以 3.9 上 import 这个模块会直接 TypeError。
    # pyproject 里的 requires-python 已经同步成 ">=3.10", 两边别改岔了。
    #
    # 【memory **不进**快照, 这是明确的设计】
    # snapshot() 用的是 json.dumps(..., default=str)。一个活的向量库句柄塞进去,
    # 会被**静默序列化**成 "<memory.semantic.SemanticMemory object at 0x...>"
    # 这样一串字符串 —— 不报错。然后 _from_snapshot 的 setdefault 会把这串
    # 字符串喂给构造函数, 直到第一次调 LLM 才炸, 报错位置离病因十万八千里。
    #
    # 所以: 想让恢复出来的 agent 也带记忆, 显式传 ——
    #     SemanticMemory.load(...) 是不存在的, 直接:
    #     ReActAgent.load("s.json", tool_registry=r, memory=mem)
    # 靠现有的 setdefault 机制**天然就支持覆盖**, 不用额外写代码。
    #

    SNAPSHOT_VERSION = 1

    def _snapshot_state(self) -> dict:
        """
        子类把自己**构造函数参数**里那部分状态塞进来, 快照才是完整的。
        """
        return {}

    def snapshot(self) -> dict:
        """把当前全部状态导出成可 JSON 序列化的字典。"""
        return {
            "version": self.SNAPSHOT_VERSION,
            "agent_type": type(self).__name__,
            "name": self.name,
            "system_prompt": self.system_prompt,
            "llm": {
                "provider": getattr(self.llm, "provider", None),
                "model": getattr(self.llm, "model", None),
                "base_url": getattr(self.llm, "base_url", None),
                "temperature": getattr(self.llm, "temperature", None),
                "max_tokens": getattr(self.llm, "max_tokens", None),
            },
            "config": self.config.model_dump(),
            "turns": [t.model_dump() for t in self._turns],
            "last_messages": getattr(self, "last_messages", []),
            "extra": self._snapshot_state(),
        }

    def save(self, path: str | Path) -> Path:
        """写快照。先写临时文件再替换 —— 中途失败不会把上一份好快照写坏。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(self.snapshot(), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        tmp.replace(path)
        return path

    def _restore(self, data: dict):
        """把快照灌回这个 agent。构造参数已由构造函数恢复, 这里只管运行期状态。"""
        self._turns = [Turn.model_validate(t) for t in data.get("turns", [])]
        if "last_messages" in data:
            self.last_messages = data["last_messages"]
        # fork 可能换了更小的预算, 恢复出来也要守同一条约束
        self._truncate_history()

    @classmethod
    def _from_snapshot(cls, data: dict, **init_kwargs):
        version = data.get("version")
        if version != cls.SNAPSHOT_VERSION:
            raise ValueError(
                f"快照版本不匹配: 文件是 v{version}, 当前框架是 v{cls.SNAPSHOT_VERSION}。"
                f"请重新生成快照, 不要按旧格式硬解 —— 静默按新格式读旧数据, "
                f"坏的是一路传下去的轨迹。"
            )
        if data.get("agent_type") != cls.__name__:
            raise ValueError(
                f"快照来自 {data.get('agent_type')}, 不能用 {cls.__name__} 恢复。"
            )

        # 快照里的值只做**默认值**: 调用方显式传的优先 —— fork 就靠这条实现
        init_kwargs.setdefault("name", data.get("name", "restored"))
        init_kwargs.setdefault("system_prompt", data.get("system_prompt"))
        if data.get("config"):
            init_kwargs.setdefault("config", Config.model_validate(data["config"]))
        for key, value in (data.get("extra") or {}).items():
            init_kwargs.setdefault(key, value)

        if "llm" not in init_kwargs:
            # 密钥不在快照里, 由 Agents0to1 自己去环境变量取。
            saved = data.get("llm") or {}
            init_kwargs["llm"] = Agents0to1(
                provider=saved.get("provider"),
                model=saved.get("model"),
                base_url=saved.get("base_url"),
                temperature=saved.get("temperature"),
                max_tokens=saved.get("max_tokens"),
            )

        agent = cls(**init_kwargs)
        agent._restore(data)
        return agent

    @classmethod
    def load(cls, path: str | Path, **init_kwargs):
        """
        从快照恢复出一个 agent。
        用 llm= 换模型/换供应商; 用 system_prompt= / config= 覆盖快照里的值。
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"快照不存在: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls._from_snapshot(data, **init_kwargs)

    @classmethod
    def fork(cls, path: str | Path, **overrides):
        """
        从同一份快照分叉 —— 每次调用都是一个**全新的、互不影响**的 agent,
        用来做对照实验(同一段历史, 换个提示词/换个模型, 看结果差在哪):

            a = ReActAgent.fork("s.json", tool_registry=r)
            b = ReActAgent.fork("s.json", tool_registry=r, system_prompt="只回答数字")
        """
        return cls.load(path, **overrides)

    def __str__(self) -> str:
        return f"Agent(name={self.name}, provider={getattr(self.llm, 'provider', '?')})"

    def __repr__(self) -> str:
        return self.__str__()
