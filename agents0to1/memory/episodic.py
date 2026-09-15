"""
情景记忆 —— 跨会话记得"你说过什么"

【和语义记忆的分工】
    语义记忆(vector_store + semantic): 跨会话记得**知识** —— 文档、事实
    情景记忆(本模块):                跨会话记得**经历** —— 一轮轮对话

两件事。前者靠向量相似度检索, 后者靠时间顺序回放。硬凑成一个东西会很别扭。

【为什么建议和向量库分开存】
两张表的生命周期不同: 向量库是"你灌进去的资料", 情景记忆是"每天长出来的流水"。
混在一个文件里, 以后想单独备份/清理其中一个会很难受。
(技术上完全可以同库不同表, 但**默认分开**更省心。)

【为什么这件事现在很便宜】
Turn 已经是 pydantic 模型(core/message.py), model_dump() 现成,
Agent._restore 里 Turn.model_validate 也现成 —— 两头都是现成的序列化,
中间只要一张表。
"""

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from ..core.exceptions import *
from ..core.message import Turn
from ..utils.logging import get_logger

logger = get_logger(__name__)


def _now_iso() -> str:
    """ISO-8601 字符串。选它而不是 unix 时间戳, 是因为能直接用眼睛读,
    而且字典序 == 时间序, ORDER BY created_at 天然正确。"""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class EpisodicMemory:
    """
    一轮轮对话的持久化。一张表就够。
        session_id | turn_index | user | messages(JSON) | answer | metadata(JSON) | created_at
    session_id 把同一个 sqlite 文件切给多个会话用 —— 换个 session_id 就是一段新对话,
    互不干扰。
    用法:
        store = EpisodicMemory("data/episodic.sqlite3", session_id="alice-2026-09")
        # 存: 跑完之后把这一轮的轨迹写进去
        for turn in agent.get_turns():
            store.append(turn)
        # 取: 下次启动, 把历史灌回去, 接着聊
        store.replay_into(agent, limit=20)
        # 或者: 作为"你说过什么"的上下文注入
        agent = SimpleAgent("a", llm, memory=store)
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS {table} (
        session_id  TEXT    NOT NULL,
        turn_index  INTEGER NOT NULL,
        user        TEXT    NOT NULL DEFAULT '',
        messages    TEXT    NOT NULL DEFAULT '[]',
        answer      TEXT    NOT NULL DEFAULT '',
        metadata    TEXT    NOT NULL DEFAULT '{{}}',
        created_at  TEXT    NOT NULL,
        PRIMARY KEY (session_id, turn_index)
    )
    """

    def __init__(self, path: str = "data/episodic.sqlite3", session_id: str = "default",
                 table: str = "episodic"):
        """
        Args:
            path:       sqlite 文件路径。**建议和语义记忆分开**(见模块开头)
            session_id: 本会话的标识。同一个文件可以放很多会话
            table:      表名
        """
        self.path = str(path)
        self.session_id = session_id
        self.table = table

        # 和向量库同一个理由: 工具走线程池, 连接必须能跨线程用 + 加锁。
        # 见 vector_store.py 里那段更详细的说明。
        self._lock = threading.RLock()

        if self.path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)

        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.execute(self._SCHEMA.format(table=self.table))
            self._conn.commit()

        logger.info("情景记忆就绪: path=%s session=%s", self.path, self.session_id)

    # ==================== 写 ====================

    def append(self, turn: Turn) -> int:
        """
        存一轮, 返回它在会话里的序号。

        序号自动往后排 —— 调用方不需要自己维护计数器, 也就不可能排错。
        """
        with self._lock:
            row = self._conn.execute(
                f"SELECT COALESCE(MAX(turn_index), -1) + 1 AS next FROM {self.table} "
                f"WHERE session_id = ?",
                (self.session_id,),
            ).fetchone()
            index = row["next"]

            self._conn.execute(
                f"INSERT INTO {self.table} "
                f"(session_id, turn_index, user, messages, answer, metadata, created_at) "
                f"VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    self.session_id,
                    index,
                    turn.user,
                    json.dumps(turn.messages, ensure_ascii=False),
                    turn.answer,
                    json.dumps(turn.metadata or {}, ensure_ascii=False, default=str),
                    _now_iso(),
                ),
            )
            self._conn.commit()
        return index

    def extend(self, turns: Sequence[Turn]) -> int:
        """存一批, 返回存了几轮"""
        count = 0
        for turn in turns:
            self.append(turn)
            count += 1
        return count

    def save_agent(self, agent, limit: Optional[int] = None) -> int:
        """
        把 agent 当前的完整轨迹存进来 —— 收尾时调它一行就够。

        Args:
            limit: 只存最近的 N 轮(None = 全存)。**注意它不能替代去重**:
                   反复调 save_agent 会把历史重复存进去, 详见 append 的说明。
        """
        turns = agent.get_turns()
        if limit is not None:
            turns = turns[-limit:]
        n = self.extend(turns)
        logger.info("已保存 %d 轮对话到会话 '%s'", n, self.session_id)
        return n

    # ==================== 读 ====================

    def recent(self, limit: int = 20, session_id: Optional[str] = None) -> List[Turn]:
        """取最近 limit 轮, **按时间正序**返回(方便直接拼回历史)。"""
        sid = session_id or self.session_id
        with self._lock:
            rows = self._conn.execute(
                f"SELECT user, messages, answer, metadata FROM {self.table} "
                f"WHERE session_id = ? ORDER BY turn_index DESC LIMIT ?",
                (sid, limit),
            ).fetchall()

        turns = [
            Turn(
                user=row["user"],
                messages=json.loads(row["messages"]),
                answer=row["answer"],
                metadata=json.loads(row["metadata"]),
            )
            for row in reversed(rows)      # 翻回正序
        ]
        return turns

    def replay_into(self, agent, limit: int = 20, session_id: Optional[str] = None) -> int:
        """
        把历史灌回一个 agent, 实现"跨会话续聊"。

        做法是把 Turn 直接拼到 agent 现有历史**前面**, 再让 agent 自己跑一次
        截断 —— 于是它仍然受 max_history_turns / max_history_chars 的约束,
        不会因为读了一万轮历史就把上下文撑爆。
        """
        turns = self.recent(limit=limit, session_id=session_id)
        if not turns:
            return 0

        # 直接操作 _turns 是故意的: 没有公开的"批量设置历史"接口, 而 add_turn
        # 会逐轮触发一次截断(20 轮就是 20 次全量重算)。这里灌完统一截断一次。
        existing = agent.get_turns()
        agent._turns = turns + existing
        agent._truncate_history()
        logger.info("已把 %d 轮历史灌回 agent '%s'", len(turns), getattr(agent, "name", "?"))
        return len(turns)

    def sessions(self) -> List[str]:
        """库里有哪几个会话"""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT DISTINCT session_id FROM {self.table} ORDER BY session_id"
            ).fetchall()
        return [r["session_id"] for r in rows]

    def count(self, session_id: Optional[str] = None) -> int:
        sid = session_id or self.session_id
        with self._lock:
            return self._conn.execute(
                f"SELECT COUNT(*) AS n FROM {self.table} WHERE session_id = ?", (sid,)
            ).fetchone()["n"]

    # ==================== 给 LLM 看的形态 ====================

    def build_context(
        self,
        query: Optional[str] = None,
        limit: int = 5,
        budget: int = 1200,
    ) -> str:
        """
        把最近几轮对话排成一段文本, 供 Agent 注入。**fail-open**: 失败返回 ""。

        【为什么 query 参数收了却没用】
        情景记忆是**按时间**检索的, 不是按相似度 —— "最近说过什么"和"当前问题像不像"
        基本无关。参数留着是为了和 SemanticMemory.build_context 的签名兼容,
        这样两者都能直接塞给 Agent 的 memory=(见 core/agent.py 的类型守卫)。

        【为什么只取最近的几轮, 而不是全部】
        完整历史本来就该走 agent._turns(那是零成本的, 不需要每次重新拼),
        这里只补**跨会话**那部分。两处都塞会重复。
        """
        del query      # 见上面的说明
        try:
            turns = self.recent(limit=limit)
        except Exception as e:
            logger.warning("读取情景记忆失败, 本轮按『没有记忆』继续: %s", e)
            return ""

        if not turns:
            return ""

        lines: List[str] = []
        for turn in turns:
            if turn.answer:
                lines.append(f"问: {turn.user}\n答: {turn.answer}")
            elif turn.user:
                lines.append(f"问: {turn.user}")
            if sum(len(x) for x in lines) > budget:
                break

        body = "\n\n".join(lines)[:budget]
        if not body:
            return ""

        return (
            "【跨会话记忆 —— 你和这位用户之前聊过的内容】\n"
            f"{body}"
        )

    # ==================== 清理 ====================

    def clear(self, session_id: Optional[str] = None) -> int:
        """清掉一个会话, 返回删了几轮"""
        sid = session_id or self.session_id
        with self._lock:
            n = self.count(sid)
            self._conn.execute(f"DELETE FROM {self.table} WHERE session_id = ?", (sid,))
            self._conn.commit()
        logger.warning("已清空会话 '%s' 的 %d 轮对话", sid, n)
        return n

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __repr__(self) -> str:
        try:
            n = self.count()
        except Exception:
            n = "?"
        return f"EpisodicMemory(path={self.path}, session={self.session_id}, turns={n})"


# ==================== 一个容易踩的坑 ====================
#
# save_agent() 是**追加**语义, 不是"同步"。所以:
#
#     for turn in agent.get_turns(): store.append(turn)   # 第一轮跑完: 存了 1 轮
#     ... 第二轮跑完再存一遍全部 ...                       # 又存了 2 轮 -> 库里 3 轮, 重复 1 轮
#
# 正确做法是只存新增的那些:
#
#     before = len(agent.get_turns())
#     agent.run("...")
#     store.extend(agent.get_turns()[before:])
#
# 之所以不在这里做自动去重: 去重需要"这一轮之前存过没有"的判断, 而历史的
# 截断(_truncate_history)会把旧轮丢掉 —— 那些被丢掉的轮次恰恰是已经在库里的。
# 靠比对内容去重会在"用户问了两次一样的问题"时误判。**让调用方决定存哪些**更可靠。
