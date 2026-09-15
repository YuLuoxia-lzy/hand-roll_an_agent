"""BLOB vs JSON 两种存储格式的实测对比

    python tests/bench_storage.py            # 默认 1500 条 / 256 维 / 30 次查询
    python tests/bench_storage.py 5000 512 50

【这不是跑分, 是让你亲眼看到"差异有多大、差异从哪来"】

指南把两种格式都留下了, 但只说了 BLOB 更快 —— 快多少? 快在哪一步?
这个文件回答的就是这两个问题, 顺便让你看到一个反直觉的结论:

    在**纯 Python**(没 numpy)下, 检索耗时的大头是**解码**, 不是算余弦。
    所以存储格式的影响比想象中大; 而装了 numpy 之后算分变便宜, 解码占比更高。

三个数字:
    入库耗时    —— 两种格式都只写一次
    库文件体积  —— BLOB 是二进制 + float32; JSON 是文本 + float64, 还带逗号和括号
    检索耗时    —— 全表扫描: 每行都要解码 + 算一次余弦

【为什么分数会有微小差异】
BLOB 走 array('f') 即 **float32**(7 位有效数字), JSON 走 json.dumps(float) 即
**float64**。精度换体积/速度。差异必须小到不影响排序 —— 这里会直接验一遍。
"""

import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import run_tests, TempDir                              # noqa: E402

from agents0to1.memory.vector_store import (                         # noqa: E402
    STORAGE_BLOB,
    STORAGE_JSON,
    VectorStore,
    has_numpy,
)

N = 1500          # 库里多少条
DIM = 256         # 向量维度
QUERIES = 30      # 检索多少次
TOP_K = 5


def _vectors(n: int, dim: int, seed: int = 0):
    """可复现的随机向量 —— 用固定种子, 保证两次跑的输入完全一样"""
    rng = random.Random(seed)
    return [[rng.uniform(-1.0, 1.0) for _ in range(dim)] for _ in range(n)]


def _time(fn, repeat: int = 1):
    """返回 (总耗时, 每次耗时)。repeat 次里取第一次之后的平均, 摊掉预热。"""
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    total = time.perf_counter() - start
    return total, total / repeat


def _human(seconds: float) -> str:
    return f"{seconds * 1000:7.2f} ms" if seconds < 1 else f"{seconds:7.3f} s "


def bench(storage: str, vectors, texts, queries) -> dict:
    """跑一种格式, 返回它的各项指标"""
    with TempDir() as d:
        path = str(d / f"bench_{storage}.sqlite3")
        store = VectorStore(path, table="bench", storage=storage)

        # ---------- 入库 ----------
        _, add_once = _time(lambda: store.add(texts, vectors, model="m"))

        # ---------- 库文件体积 ----------
        store.close()
        size_bytes = Path(path).stat().st_size
        store = VectorStore(path, table="bench", storage=storage)

        # ---------- 只解码, 不算分 ----------
        #
        # 这里故意伸手去用私有方法: 目的是把"解码"和"算余弦"分开计时,
        # 否则你只知道"慢", 不知道钱花在哪一步。
        col = "vector"
        n_rows = store.count()

        def decode_only():
            rows = store._conn.execute(f"SELECT {col}, storage FROM {store.table}").fetchall()
            for payload, fmt in rows:
                VectorStore._decode(payload, fmt)

        _, decode_once = _time(decode_only, repeat=3)

        # ---------- 全表检索 ----------
        # 按**下标**存结果: 查询向量是个 list, 直接拿它当 dict 的键会
        # TypeError: unhashable type —— 顺手提醒自己 list 不是可哈希的
        results = []

        def search_all():
            results.clear()
            for q in queries:
                results.append(store.search(q, top_k=TOP_K, model="m"))

        _, search_once = _time(search_all, repeat=3)
        search_per_query = search_once / len(queries)

        store.close()

        return {
            "storage": storage,
            "add_total": add_once,
            "size_bytes": size_bytes,
            "decode": decode_once,
            "search": search_per_query,
            "n_rows": n_rows,
            "results": results,
        }


def _parse_args(argv):
    """python tests/bench_storage.py [条数] [维度] [查询次数]"""
    global N, DIM, QUERIES
    nums = [a for a in argv if a.isdigit()]
    if len(nums) > 0:
        N = int(nums[0])
    if len(nums) > 1:
        DIM = int(nums[1])
    if len(nums) > 2:
        QUERIES = int(nums[2])


def main():
    vectors = _vectors(N, DIM)
    queries = _vectors(QUERIES, DIM, seed=99)
    texts = [f"第 {i} 条测试文本, 用来把库撑到 {N} 条。" for i in range(N)]

    print()
    print("=" * 74)
    print(f"BLOB vs JSON   n={N}  dim={DIM}  查询={QUERIES} 次  top_k={TOP_K}")
    print(f"numpy: {'开' if has_numpy() else '关(纯 Python —— 这就是你要的对照条件)'}")
    if not has_numpy():
        print("     装 numpy 再看一次, 你会发现『算分』那一半的占比明显变小,")
        print("     而解码和 I/O 的占比反而更突出。")
    print("=" * 74)

    blob = bench(STORAGE_BLOB, vectors, texts, queries)
    js = bench(STORAGE_JSON, vectors, texts, queries)

    def row(label, b, j, fmt, better):
        """better: 'small' = 越小越好, 'big' = 越大越好"""
        if b and j:
            if better == "small":
                who = "blob" if b < j else ("json" if j < b else "打平")
            else:
                who = "blob" if b > j else ("json" if j > b else "打平")
            ratio = (max(b, j) / min(b, j)) if min(b, j) > 0 else float("inf")
            tail = f"   {who} 胜 {ratio:.2f}x" if who != "打平" else "   打平"
        else:
            tail = ""
        print(f"  {label:<22}{fmt(b):>12}{fmt(j):>12}{tail}")

    print()
    print(f"  {'':<22}{'blob':>12}{'json':>12}")
    print("  " + "-" * 60)
    row("入库(全量)", blob["add_total"], js["add_total"], _human, "small")
    row("库文件体积", blob["size_bytes"], js["size_bytes"],
        lambda v: f"{v / 1024 / 1024:7.2f} MB", "small")
    row("只解码一次全表", blob["decode"], js["decode"], _human, "small")
    row("检索一次(解码+算分)", blob["search"], js["search"], _human, "small")

    print()
    print(f"  {'平均每条':<22}{'blob':>12}{'json':>12}")
    print("  " + "-" * 60)
    for label, key in (("字节/条", "size_bytes"), ("解码/条", "decode")):
        b = blob[key] / N
        j = js[key] / N
        if key == "size_bytes":
            print(f"  {label:<22}{b:9.1f}  B{j:9.1f}  B"
                  f"   blob 省 {100 * (1 - b / j):.0f}%")
        else:
            print(f"  {label:<22}{b * 1e6:9.3f} us{j * 1e6:9.3f} us")

    # ---------- 怎么读这张表 ----------
    #
    # 解码差十几倍, 整体检索却只差几倍 —— 这个"不对劲"正是最该看懂的一处。
    dec_ratio = js["decode"] / max(blob["decode"], 1e-9)
    sch_ratio = js["search"] / max(blob["search"], 1e-9)
    print()
    print(f"  【怎么读】解码那一步差了 {dec_ratio:.1f}x, 整体检索却只差 {sch_ratio:.1f}x。")
    print("     因为两种格式解码完之后, 都要在同一套纯 Python 代码里算余弦 ——")
    print("     那部分是**一样**的, 它把总时间的差距摊薄了。")
    if not has_numpy():
        print("     装了 numpy 之后算分变便宜, 解码的占比会更高, 整体差距会更接近上面那个数。")
    print("     结论: 向量库的性能不是一个数, 是『解码 + 算分 + I/O』三段各自的开销。")
    print("     换格式只动了第一段和第三段 —— 想再快就得动第二段(上 numpy / 上 faiss)。")
    print()

    # ---------- 精度: 分数差多少, 排序会不会变 ----------
    print()
    print("  " + "-" * 60)
    max_diff = 0.0
    same_order = 0
    for i in range(len(queries)):
        a = blob["results"][i]
        b = js["results"][i]
        same_order += ([t for _, t, _ in a] == [t for _, t, _ in b])
        for (s1, _, _), (s2, _, _) in zip(a, b):
            max_diff = max(max_diff, abs(s1 - s2))

    print(f"  float32 vs float64 的最大分数差: {max_diff:.3e}")
    print(f"  top-{TOP_K} 排序完全一致: {same_order}/{QUERIES} 次查询")
    print()
    if same_order == len(queries) and max_diff < 1e-5:
        print("  => 精度损失对结果**没有影响**。BLOB 那个 float32 是划算的:")
        print("     省下的是磁盘和每次检索都要付的解码时间, 换来的是分数第 7 位小数的差别。")
    else:
        print("  => 注意: 排序变了或分数差过大。这在 dim 很大或数据分布极端时可能出现,")
        print("     真遇到了就切 JSON(把 DEFAULT_STORAGE 改成 STORAGE_JSON), 或者改用 float64 的 array('d')。")

    # 一句话结论
    print()
    print("  怎么选:")
    print("    blob —— 默认。生产用它。省体积、省每次检索的解码时间。")
    print("    json —— 调试用它。`sqlite3 data/semantic.sqlite3 'select text, vector from semantic'`")
    print("            能直接读出向量来对, 排查『这条为什么没搜到』时很省事。")
    print("    两种格式可以在同一张表里共存, 切换开关不会把老数据读坏 —— 见")
    print("    test_vector_store.py::test_storage_formats_can_coexist_in_one_table")
    print()


def test_bench_smoke():
    """小规模跑一遍, 保证这个脚本本身没坏(不需要它在离线测试里真的跑全量)"""
    vectors = _vectors(50, 16)
    texts = [f"t{i}" for i in range(50)]
    queries = _vectors(2, 16, seed=7)
    b = bench(STORAGE_BLOB, vectors, texts, queries)
    j = bench(STORAGE_JSON, vectors, texts, queries)

    assert b["n_rows"] == j["n_rows"] == 50
    assert b["size_bytes"] < j["size_bytes"], "BLOB 应该比 JSON 小"
    assert [t for _, t, _ in b["results"][0]] == \
           [t for _, t, _ in j["results"][0]], "两种格式排序必须一致"


if __name__ == "__main__":
    if "--test" in sys.argv:
        sys.exit(run_tests(globals(), "存储对比(自检)"))
    _parse_args(sys.argv[1:])
    main()
