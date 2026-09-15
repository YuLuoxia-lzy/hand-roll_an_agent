"""切分逻辑离线测试 —— 不联网、不花钱

    python tests/test_chunking.py

向量库决定「多快」, 切分决定「能不能搜到」。**切分错了, 再好的向量库也搜不出
正确答案** —— 因为正确答案根本没被完整地切进任何一块里。

所以这个文件测的都是"边界情况": 空输入、超长单句、正好落在边界上的答案。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import run_tests                                    # noqa: E402

from agents0to1.memory.semantic import (                          # noqa: E402
    SemanticMemoryException,
    chunk_text,
    split_sentences,
)

#: chunk_text 认定的句子结束符, 和实现里那份保持一致
ENDERS = "。！？!?；;…\n"

#: 造测试文本用的词表。
#:
#: 【为什么要"不重复"的文本】如果每句话都长一个样("内容内容内容……"), 那么
#: "第 0 块的尾巴不在这里"这种断言会**假通过**也**假失败** —— 随便截 20 个字
#: 都能在别处找到。测试文本必须有区分度, 断言才有意义。
_WORDS = [
    "春风", "递归", "向量", "记忆", "检索", "分块", "相似", "余弦",
    "缓存", "前缀", "线程", "锁", "快照", "轨迹", "注入", "预算",
    "幻觉", "阈值", "排序", "来源", "切片", "边界", "重叠", "维度",
]


def _make_text(sentences: int, words_per_sentence: int = 12) -> str:
    """第 i 句里有专属的第 i 个标识, 保证句句不同"""
    parts = []
    for i in range(sentences):
        filler = "".join(_WORDS[(i * 7 + j) % len(_WORDS)] for j in range(words_per_sentence))
        parts.append(f"第{i}句讲的是{filler}。")
    return "".join(parts)


# ==================== 句子切分 ====================

def test_split_sentences_keeps_punctuation():
    """标点必须留着 —— 丢了句号, 拼回来的块读起来是断的"""
    got = split_sentences("第一句。第二句！第三句？")
    assert got == ["第一句。", "第二句！", "第三句？"], got


def test_split_sentences_handles_newline_as_boundary():
    got = split_sentences("标题\n正文一。正文二。")
    assert len(got) == 3, got


def test_consecutive_enders_are_one_boundary():
    """中文里 ？！ 和 。。。 连着写很常见, 每个都断一次会切出一堆只有标点的"句子" """
    assert split_sentences("什么？！") == ["什么？！"]
    assert split_sentences("。。。") == ["。。。"]
    assert split_sentences("真的吗……") == ["真的吗……"]


def test_split_sentences_drops_blank():
    assert split_sentences("   ") == []
    assert split_sentences("\n\n") == []


# ==================== 空输入 / 退化输入 ====================

def test_empty_input_returns_empty_list():
    """空输入返回 [] 而不是 [""] —— 后者会真的被拿去 embed, 浪费一次请求"""
    assert chunk_text("") == []
    assert chunk_text("   ") == []
    assert chunk_text("\n\n\t ") == []
    assert chunk_text(None) == []


def test_text_shorter_than_chunk_size_is_one_chunk():
    text = "很短的一句话。"
    assert chunk_text(text, chunk_size=600) == [text]


def test_zero_chunk_size_raises():
    try:
        chunk_text("abc", chunk_size=0)
    except SemanticMemoryException:
        return
    raise AssertionError("chunk_size=0 应该报错, 而不是死循环")


# ==================== 句子边界 ====================

def test_no_chunk_ends_mid_sentence():
    """每块(除最后一块外)都该结束在句子边界上 —— 这是"按句子切"的全部意义"""
    chunks = chunk_text(_make_text(40), chunk_size=200, overlap=0.15)
    assert len(chunks) >= 4, f"应该切出好几块, 实际 {len(chunks)}"
    for i, chunk in enumerate(chunks[:-1]):
        assert chunk.rstrip()[-1] in ENDERS, (
            f"第 {i} 块没有结束在句子边界上: ...{chunk[-20:]!r}"
        )


def test_content_is_not_lost():
    """每一句都必须出现在**至少一块**里 —— 丢了就永远搜不到"""
    text = _make_text(20)
    chunks = chunk_text(text, chunk_size=200, overlap=0.15)
    joined = "".join(chunks)
    for sentence in split_sentences(text):
        assert sentence in joined, f"这句话被切丢了: {sentence[:30]!r}"


# ==================== overlap ====================

def test_overlap_is_actually_applied():
    """相邻块要有重叠 —— 否则答案正好落在边界上就丢了"""
    chunks = chunk_text(_make_text(40), chunk_size=200, overlap=0.15)
    assert len(chunks) >= 3
    for i in range(len(chunks) - 1):
        tail = chunks[i][-20:]
        assert tail in chunks[i + 1], (
            f"第 {i} 块和第 {i+1} 块之间没有 overlap, 答案是会丢的\n"
            f"  上一块结尾: ...{tail!r}\n"
            f"  下一块开头: {chunks[i+1][:60]!r}"
        )


def test_zero_overlap_disables_it():
    chunks = chunk_text(_make_text(20), chunk_size=200, overlap=0)
    assert len(chunks) >= 2
    assert chunks[0][-20:] not in chunks[1]


def test_chunks_do_not_shrink_overlap_to_nothing():
    """overlap 太小等于没有 —— 实现里有个 30% 的下限保护, 这里确认它没被绕过"""
    chunks = chunk_text(_make_text(20), chunk_size=300, overlap=0.15)
    # 目标 overlap = 45 字符, 允许对到句子边界上, 但不该缩到 0
    assert len(chunks[0][-15:]) == 15
    assert chunks[0][-15:] in chunks[1]


# ==================== 超长单句 ====================

# 【chunk_size 的语义, 断言之前先说清楚】
# chunk_size 管的是**新内容**的预算, overlap 是额外加上去的。
# 所以一块的实际长度上限是 chunk_size + overlap_chars, 不是 chunk_size。
# (指南说的"400-800 字符 + 10-20% overlap"就是这个意思。)
#
# 想让"块本身"不超过 chunk_size 的话, 得把 overlap 从预算里扣掉 ——
# 那是另一种设计, 代价是每块能装的新内容变少。

def _max_len(chunk_size: int, overlap: float) -> float:
    return chunk_size + int(chunk_size * overlap)


def test_superlong_single_sentence_is_hard_split():
    """一句话比 chunk_size 还长 -> 没有边界可用, 只能硬切。

    这会切断语义(实现里会记一条 warning), 但总比"整块超长"或"死循环"好。
    """
    text = "啊" * 1000
    chunks = chunk_text(text, chunk_size=300, overlap=0.15)
    assert len(chunks) >= 3, chunks
    for chunk in chunks:
        assert len(chunk) <= _max_len(300, 0.15), f"硬切之后还是超长: {len(chunk)}"


def test_superlong_sentence_mixed_with_normal_ones():
    text = "短句一。" + "长" * 800 + "。短句二。"
    chunks = chunk_text(text, chunk_size=300, overlap=0.15)
    assert all(len(c) <= _max_len(300, 0.15) for c in chunks), [len(c) for c in chunks]


# ==================== 退化块 ====================

def test_no_chunk_is_pure_repetition_of_previous():
    """
    不能出现"整块都是上一块的尾巴"的退化块 —— 那种块是纯重复,
    检索出来就是噪声, 而且它还会挤掉一个 top_k 名额。

    (这是实现里 carry_chars 那个判断在防的事。)
    """
    chunks = chunk_text(_make_text(30, words_per_sentence=8), chunk_size=150, overlap=0.2)
    assert len(chunks) >= 4, f"文本不够长, 测不出退化块: 只有 {len(chunks)} 块"
    for i in range(1, len(chunks)):
        assert chunks[i] not in chunks[i - 1], (
            f"第 {i} 块整块都是上一块的子串:\n  {chunks[i][:80]!r}"
        )


# ==================== 杂项 ====================

def test_crlf_is_normalized():
    """\\r\\n 会被当成两个边界, 切出来的块带着孤立的 \\r"""
    chunks = chunk_text("第一行\r\n第二行\r\n第三行", chunk_size=100)
    assert all("\r" not in c for c in chunks), chunks


def test_short_tail_merges_into_previous():
    """最后一块太短就并进上一块 —— 一个 20 字的碎片几乎没有信息量"""
    text = _make_text(6, words_per_sentence=2) + "尾。"
    chunks = chunk_text(text, chunk_size=200, min_chunk_chars=80)
    assert len(chunks) == 1, [len(c) for c in chunks]
    assert chunks[0].endswith("尾。")


def test_first_chunk_starts_at_the_beginning():
    """第一块必须从原文开头开始 —— 少了开头那段, 前面几页就永远搜不到"""
    chunks = chunk_text(_make_text(20), chunk_size=200, overlap=0.15)
    assert chunks[0].startswith("第0句讲的是"), chunks[0][:30]


if __name__ == "__main__":
    sys.exit(run_tests(globals(), "切分"))
