"""一次跑完所有离线测试 —— **不需要任何 API key, 不联网, 不花钱**

    conda run -n agent python tests/run_offline.py      (或 D:/anaconda_env/agent/python.exe tests/run_offline.py)
    python tests/run_offline.py -v                      # 把每个用例的名字都打出来

改完记忆层之后的标准动作。全绿说明: 切分、向量库、语义记忆、Agent 接入四层
都没被改坏 —— 这四层里任何一层错了, 症状都是"检索结果有点怪", 靠肉眼看不出来。

【为什么是子进程而不是 import 进来一起跑】
每个测试文件都想自己掌控 sys.path(见 _harness.py 开头那个坑), 而且 sqlite 连接、
日志配置这些东西在同一个进程里会互相串。一个子进程一个文件, 干净、还能并行看进度。
"""

import os
import subprocess
import sys
import time
from pathlib import Path

# Windows 控制台默认是 GBK(cp936), 中文在这种老编码里没有对应字符时直接抛
# UnicodeEncodeError —— 输出到管道时更明显。测试脚本的第一件事就是把它掰成 UTF-8。
# (子进程也要, 所以下面还传了 PYTHONIOENCODING —— 子进程不一定 import _harness,
#  光靠它自己 reconfigure 是不够的。)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

#: (文件名, 显示名, 这个文件在守什么)
SUITES = [
    ("test_chunking.py",      "切分",      "向量库决定多快, 切分决定能不能搜到"),
    ("test_vector_store.py",  "向量库",    "余弦、跨模型拒绝、并发安全"),
    ("test_semantic.py",      "语义记忆",  "批量入库、预算裁剪、分层 fail-open"),
    ("test_agent_memory.py",  "Agent 接入", "注入点、Turn 不被污染、四个 Agent 各接各的"),
]


def main(argv):
    verbose = "-v" in argv or "--verbose" in argv

    print()
    print("=" * 66)
    print(f"  离线测试  (共 {len(SUITES)} 个文件)")
    print(f"  项目根目录: {ROOT}")
    print(f"  解释器:     {sys.executable}")
    print("=" * 66)

    results = []
    started = time.perf_counter()

    for filename, title, why in SUITES:
        path = HERE / filename
        if not path.exists():
            results.append((title, None, f"文件不存在: {filename}", why))
            continue

        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        proc = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )

        # 每个测试文件自己会打一张表, 这里只留最后一行汇总 + 失败用例
        lines = (proc.stdout or "").splitlines()
        summary = next((ln for ln in reversed(lines) if "全部通过" in ln or "失败" in ln), "")

        # 失败详情: [FAIL] 及其下面那行(缩进的断言信息)
        failures = []
        for i, ln in enumerate(lines):
            if "[FAIL]" in ln:
                detail = lines[i + 1].strip() if i + 1 < len(lines) else ""
                failures.append((ln.split("[FAIL]", 1)[1].strip(), detail))

        ok = proc.returncode == 0 and not failures
        results.append((title, ok, summary.strip() or (proc.stderr or "").strip()[-300:], why))

        if verbose or not ok:
            for name, detail in failures:
                print(f"    [FAIL] {name}")
                if detail:
                    print(f"           {detail}")

    elapsed = time.perf_counter() - started

    print()
    print("-" * 66)
    for title, ok, summary, why in results:
        mark = "OK  " if ok else ("缺失" if ok is None else "FAIL")
        print(f"  [{mark}] {title:<10} {summary}")
        if not ok:
            print(f"         守的是: {why}")
    print("-" * 66)

    failed = [t for t, ok, _, _ in results if not ok]
    total = len(results)
    print(f"  {total - len(failed)}/{total} 个文件通过, 耗时 {elapsed:.1f}s")

    if failed:
        print(f"  失败: {failed}")
        print()
        print("  单独跑某一个:  python tests/" + SUITES[0][0] + "   (换成对应的文件名)")
        return 1

    print()
    print("  下一步可以做:")
    print("    python tests/bench_storage.py          # 亲眼看看 BLOB vs JSON 差多少")
    print("    python tests/test_online.py            # 接真实 embedding 服务(需要 key/ollama)")
    print("    python examples/memory_demo.py         # 端到端跑一个带记忆的 agent")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
