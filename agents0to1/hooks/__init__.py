"""现成的 hook 实现

    memory.py    MemoryHook   —— 语义/情景记忆的注入(改造前的"加记忆要动 9 个文件"就是它)
    episodic.py  EpisodicHook —— 对话落库, 挂 after_run

这里是**应用和框架的分界线**: core/hooks.py 定义"在哪里挂", 本目录放
"挂上去的现成东西"。你自己要加的世界感知、成本熔断、trace 也放这种位置 ——
一个能力一个文件, 不由框架替你做决定。
"""

from .episodic import EpisodicHook
from .memory import MemoryHook

__all__ = ["EpisodicHook", "MemoryHook"]
