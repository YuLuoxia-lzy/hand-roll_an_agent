"""
Agents0to1 - 灵活、可扩展的多智能体框架

基于OpenAI原生API构建，提供简洁高效的智能体开发体验。
"""

import os


def _load_dotenv() -> bool:
    """
    可选加载同目录下的 .env。

    装了 python-dotenv 才生效, 没装就静默跳过 —— 不为了读一个 .env 文件
    就给框架强加一个硬依赖。放在最前面调用, 这样后面所有读 os.getenv 的代码
    (模型名、密钥、base_url、LOG_LEVEL) 都能吃到 .env 里的值。
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False
    return load_dotenv()


_load_dotenv()

from .version import __version__, __author__, __email__, __description__

# 核心组件
from .core import *
from .core import __all__ as _core_all

# Agent实现
from .classic_agent import *
from .classic_agent import __all__ as _classic_all

# 工具系统
from .tools import *
from .tools import __all__ as _tools_all

from .utils.logging import setup_logger


def enable_logging(level: str = None, **kwargs):
    """
    打开/调整框架日志。

    包被 import 时已经自动调过一次(级别取 LOG_LEVEL 环境变量, 默认 INFO),
    所以正常用不需要手动调它。

        enable_logging("DEBUG")     # 想看每次请求的细节
        enable_logging("WARNING")   # 只想看警告和错误
        enable_logging("CRITICAL")  # 基本静音
    """
    return setup_logger(level=level or os.getenv("LOG_LEVEL", "INFO"), **kwargs)


# import 时就 wire 一次包根 logger。各模块里的 logger = getLogger(__name__)
# 是它的子节点, 自动继承这个 handler —— 所以工具注册、LLM 调用这些日志
# 开箱就能看到, 不用每个脚本自己配一遍 logging.basicConfig。
enable_logging()


# 直接组合三个子包的 __all__, 不再手写一份总表。
# 上一版是手工维护的, 结果和实际导出长期对不上(比如导出了 SimpleAgent 却漏了
# ReActAgent, 而 __all__ 里还留着一堆注释掉的、早就改名的条目)。
# 组合之后, 子包加了新东西这里自动就有。
__all__ = [
    # 版本信息
    "__version__",
    "__author__",
    "__email__",
    "__description__",

    # 日志开关
    "enable_logging",

    # 核心 / Agent / 工具, 由各子包的 __all__ 汇总而来
    *_core_all,
    *_classic_all,
    *_tools_all,
]
