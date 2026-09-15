"""
Agent基类
用于后续创建自己的agent
可以在此地新增各种新功能
在具体agent处去丰富各种功能
"""


import copy
import json
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Optional
from .message import Message, Turn
from .llm import Agents0to1
from .config import Config
from .typedefs import AgentEvent, LLMResponse
from .hooks import HookPipeline, RunContext
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
            hooks: Optional[list] = None,
            agent_id: Optional[str] = None,
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

        # 最近一次**发出去**的消息(system + 历史 + 本轮提问, 含 hook 注入进去的内容)。
        # 语义是"模型当时看到了什么", 调试和快照用 —— 不是 agent 那份会不断 append
        # 的工作列表(那个是子类内部的东西)。见 _chat / _stream_chat。
        self.last_messages: list[dict] = []

        # ==================== 身份 ====================
        #
        #   name 只是给人和日志看的标签, 不保证唯一、也不保证稳定。
        #   agent_id 才是身份: 跨进程 / 跨会话 / 记忆归属 / 消息路由都靠它
        #   (小镇那种"很多个 agent 共用一个向量库"的场景尤其需要)。
        self.agent_id: str = agent_id or uuid.uuid4().hex[:12]

        # ==================== 扩展点 ====================
        #
        # 四个子类都是用**位置参数**调 super().__init__(name, llm, system_prompt, config)
        # 所以 hooks / agent_id 只能加在 config 后面, 而且子类必须用关键字传进来。
        # 插在 config 前面的话, 所有位置调用会把 config 静默绑到 hooks 上 ——
        # 又是一个不报错的错误。同一个理由, 见原来 memory 参数那段注释。
        self._hooks = HookPipeline(hooks)
        #: 当前这一轮的上下文。run() / stream_run() 建立和清理, 平时是 None。
        self._ctx: Optional[RunContext] = None

    #声明必须实现这个方法！！
    #
    # 【为什么是 _run 而不是 run】
    # run() 现在是**模板方法**(见下面), ctx 的生命周期由框架管, 子类忘不掉。
    # 改造前每个子类自己实现 run(), 于是"记得建 ctx / 记得清 ctx"就成了
    # 又一条"必须记得做的事" —— 而"必须记得做的事"正是加记忆要动 9 个文件的病根。
    @abstractmethod
    def _run(self, input_text: str, **kwargs) -> str:
        """真正干活的实现。**子类覆盖这个, 不要覆盖 run()。**"""
        pass

    def run(self, input_text: str, **kwargs) -> str:
        """
        运行Agent —— 模板方法: 建上下文 -> 跑子类实现 -> 收尾。

        **子类不要覆盖这个方法**, 覆盖 _run()。ctx 的生命周期由框架管。
        """
        ctx = RunContext(input_text=input_text, agent=self, turn_index=len(self._turns))
        self._ctx = ctx
        try:
            # ⚠️ before_input 是在 ctx.input_text 已经存下**原值**之后才跑的, 这是刻意的:
            # ctx.input_text 是给 hook 用的"用户到底问了什么", 它不该被任何 hook 改写 ——
            # 记忆检索的 query 就是它。before_input 改写的是**送进子类的那一份**。
            # 真被改写时两者不一致, 那个不一致是有意的。
            return self._run(self._hooks.before_input(ctx, input_text), **kwargs)
        finally:
            # 清掉: 否则 add_turn 的 after_run 会在"这一轮早就结束了"之后被触发一次
            self._ctx = None

    def stream_run(self, input_text: str, **kwargs) -> Iterator[AgentEvent]:
        """
        流式运行Agent, 逐块吐出 AgentEvent。

        默认实现: 任何实现了 _run() 的 Agent 都能被包成一个"只吐 final"的事件流。
        想要精细的(工具事件)覆盖 _stream_run() 就好。

        改造前这个方法在基类里只会 raise NotImplementedError, 四个 Agent 里
        只有两个实现了它 —— 现在四个都有了, 而且不用各自写一遍。
        """
        ctx = RunContext(input_text=input_text, agent=self, turn_index=len(self._turns))
        self._ctx = ctx
        try:
            yield from self._stream_run(self._hooks.before_input(ctx, input_text), **kwargs)
        finally:
            self._ctx = None

    def _stream_run(self, input_text: str, **kwargs) -> Iterator[AgentEvent]:
        """
        默认: 退化成 run() 的单个 final 事件。**子类可以覆盖。**
        (写不出事件流能力的 Agent, 至少不该连"事件流"这个形状都没有。)

        ⚠️ 这里调的是 _run(), **不是 run()**。
        调 run() 会重新开一个 ctx(把当前这个覆盖掉), 而 ReAct 那种
        "_run 消费 _stream_run" 的形状会直接**无限递归**。
        """
        answer = self._run(input_text, **kwargs)
        yield AgentEvent(type="final", answer=answer)

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
        拼入口消息: system + 历史 + 当前问题。

        **这里没有记忆。** 检索结果由 hook 在 before_llm 阶段并进最后那条 user ——
        改造前它长在这里, 于是每个"想往 prompt 里塞点东西"的能力都得来改这个方法。
        """
        return (
            self._base_messages()
            + self._history_messages()
            + [{"role": "user", "content": input_text}]
        )

    def _ensure_system(self, messages: list[dict]) -> list[dict]:
        """
        兜底: 消息列表第一条不是 system, 就在最前面补一条。
        """
        if messages and messages[0].get("role") == "system":
            return messages
        return self._base_messages() + list(messages)

    # ==================== 调 LLM ====================

    def _hook_ctx(self) -> RunContext:
        """
        当前这一轮的 ctx。**run() 之外直接调 _chat 时兜一个临时的空上下文。**

        什么时候会那样调: 手搓 `Planner(agent).plan(...)`、测试里直接捅 _chat、
        或者谁写了个不走 run() 的循环。那时候 ctx 是 None —— 兜一个临时的,
        比让 hook 吃到 AttributeError 强: 后者会被 fail-open 静默吞掉, 你只会
        觉得"记忆好像没生效", 查半天。

        ⚠️ **不把它存进 self._ctx**: 存了的话, 这次临时调用结束后 add_turn 的
        after_run 会拿这个假上下文触发一次 —— 那才是真的串台。
        """
        if self._ctx is not None:
            return self._ctx
        return RunContext(input_text="", agent=self)

    def _chat(self, messages: list[dict], tools=None, **kwargs) -> LLMResponse:
        """
        所有**非流式** LLM 调用的唯一出口。

        【为什么 hook 挂在这而不是挂在 run() 里】
        run() 只发一次 LLM 的场景(SimpleAgent)看着没问题, 但 ReAct 一轮里调 N 次
        LLM+工具, PlanSolve 一轮里调 2N+1 次。挂在 run() 上等于"一轮只拦一次",
        记忆注入、成本熔断、trace 全都只对第一次调用有效 —— 而且不报错。
        挂在这个出口上, **每一次** LLM 调用都被拦到, 不管是谁发起的。
        """
        ctx = self._hook_ctx()
        outgoing = self._hooks.before_llm(ctx, self._ensure_system(messages))
        response = self.llm.invoke(outgoing, tools=tools, **kwargs)

        # 存"真正发出去的那一份"(含 hook 注入的内容): 语义是"模型当时看到了什么",
        # 调试和快照用。注意它和子类内部那份不断 append 的工作列表**不是一回事**。
        self.last_messages = outgoing

        response = self._hooks.after_llm(ctx, response)
        # 在 hook 之后自增: before_llm / after_llm 里读到的 llm_calls 是"这次是第几次",
        # 从 0 开始 —— 于是 `if ctx.llm_calls == 0` 就是"这是本轮第一次调用"。
        ctx.llm_calls += 1
        return response

    def _stream_chat(self, messages: list[dict], tools=None, **kwargs) -> Iterator[str]:
        """
        流式版本, 同样强制注入 system, 同样过 hook。

        ⚠️ 与 _chat 的**唯一不对称**: after_llm 在流式路径上**只能观察, 不能改**。
        chunk 是边生成边 yield 出去的, 等 after_llm 跑到的时候调用方早就把字符
        拿去用了 —— 改 self._last_response 对已经吐出去的内容没有任何影响。
        做 usage 统计 / trace 没问题, 做内容审核只有非流式路径有效。
        这个不对称是流式的固有代价, 消除不掉, 只能写清楚。
        """
        ctx = self._hook_ctx()
        outgoing = self._hooks.before_llm(ctx, self._ensure_system(messages))
        self.last_messages = outgoing

        yield from self.llm.stream_invoke(outgoing, tools=tools, **kwargs)

        # 只在生成器被完整消费后才会执行到这里 —— 提前 break 的调用方拿不到新值,
        # 这正是"流式调用的固有语义", 不是 bug。
        self._last_response = self._hooks.after_llm(ctx, self.llm.last_response)
        ctx.llm_calls += 1

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

        # 4. after_run —— 一轮真正结束了。落库、打分、写记忆流都在这个阶段。
        #
        # 【为什么挂在 add_turn 里, 而不是子类的 run() 结尾】
        # 因为 add_turn 是**所有**写入路径的汇合点: add_turn / add_message /
        # _record_turn 最后都走到这。挂在这儿,"一轮结束了"这件事只有一个定义 ——
        # 挂在子类里的做法, 四个子类就有四种定义, 而且第五个子类会漏掉。
        #
        # self._ctx is None 的两种情况: ① 有人在 run() 之外直接调 add_turn
        # (add_message 那条老路径) ② 上一轮已经结束、ctx 被清掉了。
        # 两种都不该凭空捏一个 ctx 出来 —— 那会让 hook 拿到空 input_text 还以为
        # 自己在一轮真实的对话里。
        if self._ctx is not None:
            self._hooks.after_run(self._ctx, turn)

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
    # 【hooks **不进**快照, 这是明确的设计】
    # snapshot() 用的是 json.dumps(..., default=str)。一个活的向量库 / 数据库
    # 连接 / HTTP 客户端句柄塞进去, 会被**静默序列化**成
    # "<memory.semantic.SemanticMemory object at 0x...>" 这样一串字符串 ——
    # 不报错。然后 _from_snapshot 的 setdefault 会把这串字符串喂给构造函数,
    # 直到第一次调 LLM 才炸, 报错位置离病因十万八千里。
    #
    # 所以: 想让恢复出来的 agent 也带记忆, 显式传 ——
    #     SimpleMemory.load(...) 是不存在的, 直接:
    #     ReActAgent.load("s.json", tool_registry=r, hooks=[MemoryHook(mem)])
    # 靠现有的 setdefault 机制**天然就支持覆盖**, 不用额外写代码。
    #
    # 【agent_id 进快照, 但 fork() 会换一个新的】
    # 恢复(load)是"同一个 agent 又回来了": 记忆归属、消息路由都该认得它,
    # 所以沿用快照里的 id。
    # fork() 是"另起一个": 同一份历史, 换提示词/换模型跑对照实验 —— 那是两个
    # agent, 共用一个 id 的话, 向量库里两边的记忆会互相认亲。
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
            "agent_id": self.agent_id,
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
        # 旧快照(这个字段之前存的)没有 agent_id -> 这里给 None -> 构造函数重新生成一个。
        # 不强求: 让"能读旧快照"这件事继续成立, 比"id 必须沿袭"重要。
        init_kwargs.setdefault("agent_id", data.get("agent_id"))
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

        【分叉出来的 agent_id 是新的】
        load() 沿用快照里的 id(同一个 agent 又回来了), fork() 换一个新的 ——
        它是"另起一个", 两边共用一个 id 会让记忆归属、消息路由认错人。
        要显式指定就 overrides 里传 agent_id=。
        """
        overrides.setdefault("agent_id", uuid.uuid4().hex[:12])
        return cls.load(path, **overrides)

    def __str__(self) -> str:
        return f"Agent(name={self.name}, provider={getattr(self.llm, 'provider', '?')})"

    def __repr__(self) -> str:
        return self.__str__()
