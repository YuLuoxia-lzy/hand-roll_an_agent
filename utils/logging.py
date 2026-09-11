"""
日志工具

用法:
    from ..utils.logging import get_logger
    logger = get_logger(__name__)
    logger.info("工具 '%s' 已注册。", tool.name)
"""

import logging
import sys
from typing import Optional, Union

_PACKAGE_ROOT = __name__.split(".")[0]

DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DEFAULT_DATEFMT = "%H:%M:%S"


def _resolve_level(level: Union[str, int]) -> int:
    """
    把 "INFO" / "DEBUG" / 20 这类输入统一转成 logging 的数值级别。

    比原来 getattr(logging, level.upper()) 的好处: 传了不认识的名字时退回 INFO,
    而不是抛 AttributeError 让程序在"想开日志"这一步就崩掉。
    """
    if isinstance(level, int):
        return level

    resolved = logging.getLevelName(str(level).upper())
    # getLevelName 对不认识的名字会返回 "Level XXX" 这样的字符串, 用它来判失败
    return resolved if isinstance(resolved, int) else logging.INFO


def setup_logger(
    name: str = _PACKAGE_ROOT,
    level: Union[str, int] = "INFO",
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    配置并返回包根 logger。

    幂等: 已经有 handler 就不重复添加, 所以多处调用是安全的。

    propagate 设为 False: 本框架的日志只走自己的 handler, 不会被外层的
    logging.basicConfig() 再打印一遍(否则同一条日志会出现两次)。
    代价是 pytest 的 caplog 抓不到, 需要时应显式往这个 logger 上挂 handler。
    """
    logger = logging.getLogger(name)
    logger.setLevel(_resolve_level(level))

    if not logger.handlers:
        # 日志走 stderr 而不是 stdout: stdout 是给"程序真正的输出"用的 ——
        # 比如 SimpleAgent 流式吐出来的正文。日志混进去会把正文冲散,
        # 重定向到文件时(> out.txt)更是把两者搅在一起。
        # 分开之后 `2>` 就能单独拿走日志。
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(format_string or DEFAULT_FORMAT, datefmt=DEFAULT_DATEFMT)
        )
        logger.addHandler(handler)

    # propagate=False: 只走自己的 handler。否则外层 logging.basicConfig() 再打印一遍,
    # 同一条日志会出现两次。代价是 pytest 的 caplog 默认抓不到(它靠 root logger 传播),
    # 写测试时注意这一点。
    logger.propagate = False
    return logger


def get_logger(name: str = _PACKAGE_ROOT) -> logging.Logger:
    """获取 logger。模块里请传 __name__, 这样日志会显示来源模块。"""
    return logging.getLogger(name)
