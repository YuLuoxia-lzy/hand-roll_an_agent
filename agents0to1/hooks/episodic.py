"""对话落库 hook —— 把每一轮写进情景记忆

本地笔记 docs/extension-map.md 里 `Hook.after_run` 那一行写的参考实现就是它。
落库这件事本事全在 memory/episodic.py 里, 这个文件只负责**在正确的时机调它**。

    store = EpisodicMemory("data/episodic.sqlite3", session_id="alice")
    agent = SimpleAgent("a", llm, hooks=[EpisodicHook(store)])

    # 下次启动:
    store.replay_into(agent, limit=20)          # 续聊
    agent = SimpleAgent("a", llm, hooks=[MemoryHook(store)])   # 或者当记忆注入
"""

from typing import Any, Optional

from ..core.hooks import Hook, RunContext
from ..utils.logging import get_logger

logger = get_logger(__name__)


class EpisodicHook(Hook):
    """
    一轮跑完就把**这一轮**存进情景记忆。

    【为什么是"存这一轮", 而不是"跑完调一次 save_agent(agent)"】
    episodic.py 结尾专门记着那个坑: `save_agent()` 是**追加**语义, 不是同步。
    跑完第三轮再调一次, 它会把当前历史里的三轮全再存一遍 —— 库里 6 轮,
    其中 3 轮是重复的。而要回答"哪些已经存过了", 两条路都是死的:
      - 靠下标: `_truncate_history` 丢的是**最旧**的轮, 恰恰是已经存过的那些,
        下标会跟着往前挪, 于是要么漏存要么重存, 两种都不报错
      - 靠内容比对: "用户问了两次一样的问题"会被当成重复而漏掉第二轮
    所以判据不该是"和库里比", 而该是"这一轮是不是刚结束的那一轮"。
    after_run 递过来的 turn 就是它 —— 不可能是旧的, 也不可能是别人的。

    【为什么 on_error = "open"】
    和 MemoryHook 同一个理由: 落库是锦上添花, 它没有权力把一轮**已经跑完**的
    对话变成失败。磁盘满、库被锁、表被删, 都该是"这一轮没存上",
    而不是"这一轮白跑了"。
    """

    #: 见类文档。落库失败不影响已经拿到的回答。
    on_error = "open"

    def __init__(self, store: Optional[Any] = None):
        """
        Args:
            store: 任何提供 `append(turn)` 的对象 —— EpisodicMemory 就是,
                   也可以是别的写着玩的实现(只要方法名对得上)
        """
        if store is not None and not callable(getattr(store, "append", None)):
            raise TypeError(
                f"store 需要提供 append(turn) 方法, 收到的是 {type(store).__name__}"
            )
        self.store = store

    def after_run(self, ctx: RunContext, turn) -> None:
        if self.store is None or turn is None:
            return

        if not turn.is_valid():
            # 内存里存一份坏轨迹只影响这一轮; **落库**的坏轨迹会在下次启动被
            # replay_into 灌回历史, 让新会话第一句话就报错。所以这里不拦
            # (丢数据比留着坏数据更难发现), 但必须说出来。
            logger.warning(
                "这一轮的轨迹配对异常, 已照常落库 —— 注意它会在下次 "
                "replay_into 时被当成历史发出去。"
            )

        self.store.append(turn)
        logger.debug(
            "第 %d 轮已落库 (session=%s)",
            ctx.turn_index,
            getattr(self.store, "session_id", "?"),
        )
