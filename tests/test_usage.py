"""契约层离线测试: Usage / model / finish_reason / stream_options

    python tests/test_usage.py

**不需要任何 API key, 不联网, 不花钱。** 做法是给客户端塞一个假的 openai 对象 ——
我们测的不是"DeepSeek 返不返 usage", 而是"SDK 给了之后, 框架有没有把它接住"。

【为什么这层值得单独测】
usage 是成本统计、预算、熔断的共同地基。它最危险的失效方式不是报错, 而是
**静默为 None**: 流式那条路上, usage 藏在最后一个 choices 为空的 chunk 里,
取晚了就永远拿不到, 而代码看起来完全正常。
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import run_tests                                  # noqa: E402

from agents0to1.core.llm import Agents0to1                       # noqa: E402
from agents0to1.core.typedefs import LLMResponse, Usage          # noqa: E402


# ==================== 假的 openai 客户端 ====================

class _NS:
    """一个能随便挂属性的小对象 —— 用来冒充 SDK 返回的那些结构"""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _message(content=None, tool_calls=None):
    return _NS(content=content, tool_calls=tool_calls)


def _response(content="好的。", usage=None, model="fake-served-model", finish_reason="stop"):
    return _NS(
        choices=[_NS(message=_message(content=content), finish_reason=finish_reason)],
        usage=usage,
        model=model,
    )


class _FakeCompletions:
    def __init__(self, result):
        self.result = result
        self.params = None

    def create(self, **params):
        self.params = params
        return self.result


class _FakeOpenAI:
    """冒充 openai.OpenAI —— 只需要 .chat.completions.create"""

    def __init__(self, result):
        self.chat = _NS(completions=_FakeCompletions(result))


def _client_with(result) -> Agents0to1:
    """
    造一个连着假 openai 的客户端。

    显式传 api_key / base_url / provider, 绕开环境变量 —— **离线测试不许吃 .env**。
    OpenAI(...) 的构造本身不联网, 所以这里不会真的打出去。
    """
    llm = Agents0to1(
        model="test-model",
        api_key="sk-test-not-a-real-key",
        base_url="https://example.invalid/v1",
        provider="openai",
    )
    llm._client = _FakeOpenAI(result)
    return llm


def _chunk(content=None, usage=None, model=None, finish_reason=None, tool_calls=None):
    if content is None and not tool_calls:
        choices = []                      # ← include_usage 的最后一个 chunk 长这样
    else:
        choices = [_NS(delta=_NS(content=content, tool_calls=tool_calls),
                       finish_reason=finish_reason)]
    return _NS(choices=choices, usage=usage, model=model)


# ==================== 非流式 ====================

def test_non_stream_fills_usage():
    raw = _NS(prompt_tokens=11, completion_tokens=7, total_tokens=18)
    llm = _client_with(_response(content="你好", usage=raw))

    resp = llm.invoke([{"role": "user", "content": "hi"}])

    assert isinstance(resp, LLMResponse)
    assert resp.usage == Usage(11, 7, 18), f"usage 没被接住: {resp.usage}"
    assert resp.model == "fake-served-model"
    assert resp.finish_reason == "stop"


def test_non_stream_fills_finish_reason_length():
    """被 max_tokens 截断时是 "length" —— 那意味着"答案不完整", 不是"模型不想说了" """
    llm = _client_with(_response(finish_reason="length"))
    assert llm.invoke([{"role": "user", "content": "hi"}]).finish_reason == "length"


def test_missing_usage_is_none_not_zero():
    """**拿不到就是 None, 不能编造 0。**

    服务不返回 usage 时, 正确的反应是"去开 include_usage";
    而 Usage(0,0,0) 会说"统计到了, 就是零" —— 两种情况的处理完全相反。
    """
    llm = _client_with(_response(usage=None))
    resp = llm.invoke([{"role": "user", "content": "hi"}])

    assert resp.usage is None, f"没有 usage 时该是 None, 实际 {resp.usage}"
    assert not resp.usage, "None 和 0 在两处都要能当'没有'用"


def test_usage_survives_partial_fields():
    """不同厂家的 usage 字段不完全一样 —— 少一个不该把整轮调用带崩"""
    llm = _client_with(_response(usage=_NS(prompt_tokens=5)))     # 没有 completion/total

    resp = llm.invoke([{"role": "user", "content": "hi"}])
    assert resp.usage == Usage(5, 0, 0)


# ==================== 流式 ====================

def test_stream_fills_usage_from_the_empty_choices_chunk():
    """**这条是整个文件的重点。**

    开了 include_usage 之后, usage 只出现在**最后一个 choices 为空的 chunk** 里。
    而 core/llm.py 里有一句 `if not chunk.choices: continue` —— usage 要是放在
    它后面取, 就永远拿不到。**不报错, 只是 usage 一直是 None。**
    """
    chunks = [
        _chunk(content="你", model="fake-served-model"),
        _chunk(content="好"),
        _chunk(content="。", finish_reason="stop"),
        # ↓ 最后一个: choices 空, 只有 usage
        _chunk(usage=_NS(prompt_tokens=3, completion_tokens=4, total_tokens=7)),
    ]
    llm = _client_with(chunks)

    text = "".join(llm.stream_invoke([{"role": "user", "content": "hi"}]))

    assert text == "你好。", f"正文不能受影响, 实际 {text!r}"
    assert llm.last_response is not None
    assert llm.last_response.usage == Usage(3, 4, 7), \
        f"藏在空 choices chunk 里的 usage 被丢了: {llm.last_response.usage}"
    assert llm.last_response.model == "fake-served-model"
    assert llm.last_response.finish_reason == "stop"


def test_stream_without_usage_is_none():
    chunks = [_chunk(content="好", finish_reason="stop")]
    llm = _client_with(chunks)
    "".join(llm.stream_invoke([{"role": "user", "content": "hi"}]))

    assert llm.last_response.usage is None


def test_stream_asks_for_usage_by_default():
    chunks = [_chunk(content="好", usage=_NS(prompt_tokens=1, completion_tokens=1, total_tokens=2))]
    llm = _client_with(chunks)
    "".join(llm.stream_invoke([{"role": "user", "content": "hi"}]))

    params = llm._client.chat.completions.params
    assert params.get("stream") is True
    assert params.get("stream_options") == {"include_usage": True}, \
        "不开 include_usage 的话, 服务端根本不会在流里返回 usage"


def test_stream_options_can_be_switched_off():
    """**不是所有 OpenAI 兼容服务都认这个字段**(一些自建 / 老版本 vLLM 会 400)。

    所以它必须是可关的 —— 关掉之后行为退回原样: 没有 stream_options, usage 为 None。
    """
    old = os.environ.get("LLM_STREAM_USAGE")
    os.environ["LLM_STREAM_USAGE"] = "false"
    try:
        chunks = [_chunk(content="好")]
        llm = _client_with(chunks)
        "".join(llm.stream_invoke([{"role": "user", "content": "hi"}]))

        params = llm._client.chat.completions.params
        assert "stream_options" not in params, f"关掉之后不该再发这个字段: {params}"
        assert params.get("stream") is True, "stream 本身当然还在"
        assert llm.last_response.usage is None
    finally:
        if old is None:
            os.environ.pop("LLM_STREAM_USAGE", None)
        else:
            os.environ["LLM_STREAM_USAGE"] = old


def test_non_stream_never_sends_stream_options():
    """stream_options 只属于流式请求 —— 非流式带上它, 有些服务会 400"""
    llm = _client_with(_response())
    llm.invoke([{"role": "user", "content": "hi"}])
    assert "stream_options" not in llm._client.chat.completions.params


# ==================== Usage 本身 ====================

def test_usage_adds_up():
    """一次 run() 里可能调很多次 LLM, 你最终想知道的是总数"""
    total = Usage(1, 2, 3) + Usage(10, 20, 30)
    assert total == Usage(11, 22, 33)

    # 累加不改变原来的对象(每次都是新实例)
    first = Usage(1, 2, 3)
    first + Usage(9, 9, 9)
    assert first == Usage(1, 2, 3), "__add__ 不该就地改自己"


def test_usage_str_is_readable():
    """日志里要能一眼看懂 —— 成本问题的第一现场就是日志"""
    assert str(Usage(1, 2, 3)) == "prompt=1 completion=2 total=3"


def test_usage_is_importable_from_the_package_root():
    """应用层写 cost hook 时不该被迫 import 到子模块里"""
    import agents0to1
    assert agents0to1.Usage is Usage
    assert "Usage" in agents0to1.__all__


if __name__ == "__main__":
    sys.exit(run_tests(globals(), "用量契约"))
