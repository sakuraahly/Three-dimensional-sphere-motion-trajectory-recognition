"""回归测试运行器：项目里**唯一**跑全套测试的地方。

为什么要有它
------------
在这个运行器出现之前，"跑全套回归测试"是靠人敲一句 PowerShell 循环：

    Get-ChildItem tests\\selftest_*.py | ForEach-Object { & $PY $_.FullName }

它有三个具体毛病，都在实际使用中踩到过：

1. **没有分层**。全套要 6~7 分钟，其中 `selftest_stereo_calib` 一个就占
   200+ 秒、`selftest_synthetic` 再占 90+ 秒，**其余的合计远小于这两个之和**。
   只改一行文档也要等全套，于是大家就不跑了 —— 质量门形同虚设。
2. **"哪两个是慢的"要靠人记**。而慢的那个还可能换（实测 `selftest_pair_source`
   在两次运行里分别是 1.5 s 和 39.7 s）—— 写死名单一定会漂。
3. **没有人读得懂的汇总**。PASS/FAIL 靠肉眼数，耗时靠感觉。

所以本模块：

- `fast` / `full` 两档。**`fast` 的名单不是手写的** ——
  它读 `outputs/test_results/timings.json`（上一次实测的耗时），
  把超过 `--slow-threshold`（默认 60 s，见 `DEFAULT_SLOW_THRESHOLD`）的排除掉。
  **第一次跑会自动跑全套并记下耗时**，之后 `fast` 就有依据了。
- 每个测试**独立子进程**跑：一个崩了不会带走其余，退出码原样保留。
- 判定口径与 `AGENTS.md` §7 一致：输出含 `判定结果：全部通过` **且** 退出码 0。
  `[跳过]` 也算通过（缺可选依赖/无相机），但**跳过 ≠ 验证过**，所以单独标出来。
- 结果写 `outputs/test_results/<时间戳>.json`，给机器读。

用法::

    python apps/run_tests.py              # 默认 fast
    python apps/run_tests.py full         # 全套（改代码后必跑）
    python apps/run_tests.py fast --slow-threshold 10
    python apps/run_tests.py --only selftest_calib_facts selftest_doc_facts
    python apps/run_tests.py --list       # 只列清单与分层，不跑
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "sphere_tracker").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TESTS_DIR = ROOT / "tests"
RESULTS_DIR = ROOT / "outputs" / "test_results"
TIMINGS_FILE = RESULTS_DIR / "timings.json"
PASS_MARKER = "判定结果：全部通过"
SKIP_MARKER = "[跳过]"
# 分层阈值：超过它就归入"慢"，fast 档默认不跑。
#
# 实测（本机、多次）：
#   stereo_calib  215 ~ 278 s   ← 真正的大头，必须排除
#   synthetic      78 ~ 102 s   ← 第二大头
#   pair_source   1.5 ~  50 s   ← **方差极大**（要写临时视频素材，取决于磁盘/编码器）
#   其余 12 个合计 ~10 s
#
# 阈值放 60 s：稳稳排除前两个，同时**不**因为 pair_source 偶尔慢就把它踢出去。
# 第一版放 30 s，结果 pair_source 一慢就被排除 —— 而它守的是"同步偏移符号搞反"
# 这类**静默错误**，恰恰最不该被跳过。
DEFAULT_SLOW_THRESHOLD = 60.0
# **保底名单**：无论计时怎么说，fast 档都必须跑这些。
#
# 为什么需要它（这是"自动分层"与"必须跑"之间的取舍）：
# 自动分层的依据是机器计时，而计时会被负载影响；把"跑不跑某个测试"完全交给
# 一个会漂的数字，就意味着某天它会**悄悄不被跑**。这几个测试守的是
# "不跑就发现不了、且后果是静默错误"的东西，所以显式钉住。
# ---- 并行执行的三条约束（2026-10-27）------------------------------------
# `--jobs N` 不是"把循环换成线程池"就完事，下面两组是**真实会互相破坏**的：
#
# 1) 独占：`selftest_mutation_guards` 会**临时改写源码**再还原。
#    它和任何测试并行 = 别的测试在这份被改坏的代码上跑，且结果看起来正常。
EXCLUSIVE = frozenset({"selftest_mutation_guards.py"})
#
# 2) 互斥组：`selftest_doc_facts` 会在 `docs/` 下造"坏引用"探针并跑 `check_standards`；
#    `selftest_entrypoints` 也会跑 `check_standards`。两者并行时，后者会看到前者留下的
#    探针，于是"规范检查报红"——这是并行引入的假失败，和代码无关。
#    （2026-10-27 实测：并发手跑这两个确实会撞出 FileNotFoundError 与红节。）
CONFLICT_GROUPS = (
    frozenset({"selftest_doc_facts.py", "selftest_entrypoints.py"}),
)
# 子进程里把数学库线程压成 1：16 核上开 6 个进程、每个再开满 BLAS 线程会互相抢，
# 反而比串行慢。测试本身没有能吃到多线程的负载。
_PARALLEL_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


MUST_RUN = (
    "selftest_pair_source.py",       # 偏移符号搞反 = 程序照跑、三维系统性偏
    "selftest_config_wiring.py",     # 配置项写了没人读 = 改了不生效
    "selftest_camera_registry.py",   # 索引漂移 = 用错相机还看不出来
)
# **永不进 fast 档**：无论计时数据怎么说（甚至完全没有数据）。
#
# 为什么需要它（`unknown` 的默认行为会咬人）：
# 没有实测耗时的测试会被塞进 fast 里跑掉（见 `classify` 的文档串），
# 而 `selftest_mutation_guards.py` **会临时改写源码文件**（改完靠 SHA256 校验还原）。
# 让它在 `fast` 档里跑有两个后果：
#   ① `启动程序.bat test` 的耗时成倍增加 —— 快速自检就此失去意义；
#   ② 它在运行期把某个源码文件改成"缺陷版"，**同进程/同仓的任何并发观察者
#      都会看到坏代码**（当前运行器是串行的 [已核实]，但把这种前提交给
#      "千万别改成并行"来维持，太脆）。
# 所以：**改源码的测试，必须显式排除在快速档之外。**
NEVER_FAST = (
    "selftest_mutation_guards.py",
)
# 每个测试保留最近几次耗时，取**中位数**分层（比"最后一次"抗抖动）。
TIMING_HISTORY = 5


@dataclass
class TestResult:
    name: str
    status: str = "unknown"          # pass / fail / timeout / error
    seconds: float = 0.0
    exit_code: int | None = None
    skipped: bool = False
    note: str = ""
    slow: bool = False               # 依据上一次实测耗时分层
    tail: list[str] = field(default_factory=list)   # 失败时留最后几行，便于定位

    def to_json(self) -> dict:
        return asdict(self)


def discover() -> list[str]:
    """按文件名排序的测试清单。**清单来自磁盘，不手写**。"""
    return sorted(p.name for p in TESTS_DIR.glob("selftest_*.py"))


def load_timings() -> dict[str, list[float]]:
    """历史耗时（每个测试最近 `TIMING_HISTORY` 次）。第一次跑时为空。"""
    try:
        data = json.loads(TIMINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, list[float]] = {}
    for k, v in data.items():
        series = v if isinstance(v, list) else [v]        # 兼容旧的"单个数字"格式
        vals: list[float] = []
        for x in series:
            try:
                vals.append(float(x))
            except (TypeError, ValueError):
                continue
        if vals:
            out[str(k)] = vals[-TIMING_HISTORY:]
    return out


def save_timings(timings: dict[str, list[float]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TIMINGS_FILE.write_text(
        json.dumps(dict(sorted(timings.items())), indent=2, ensure_ascii=False),
        encoding="utf-8")


def note_timing(timings: dict[str, list[float]], name: str, seconds: float) -> None:
    series = timings.setdefault(name, [])
    series.append(round(float(seconds), 2))
    del series[:-TIMING_HISTORY]


def median(values: list[float]) -> float:
    xs = sorted(values)
    mid = len(xs) // 2
    if not xs:
        return 0.0
    return xs[mid] if len(xs) % 2 else (xs[mid - 1] + xs[mid]) / 2.0


def classify(names: list[str], timings: dict[str, list[float]],
             threshold: float) -> tuple[list[str], list[str], list[str]]:
    """分成 (fast, slow, unknown)。

    分层依据是**历史耗时的中位数**（见 `TIMING_HISTORY` 处的说明），
    但：
    - `MUST_RUN` 里的测试**永远进 fast**，不受计时影响；
    - `NEVER_FAST` 里的测试**永远不进 fast**（改源码的测试不能混进快速自检），
      所以它直接落 slow，连 `unknown` 都不进。

    两个名单**不许重叠**（重叠 = 配置自相矛盾，直接报错而不是挑一个赢）。

    `unknown` = 还没有实测耗时的测试。**第一轮它们会被塞进 fast 里跑掉**
    —— 宁可第一次慢一点，也不要因为"没数据"就把某个测试永久排除在外。
    """
    clash = set(MUST_RUN) & set(NEVER_FAST)
    if clash:
        raise SystemExit(f"MUST_RUN 与 NEVER_FAST 重叠：{sorted(clash)} —— 请修名单")
    fast, slow, unknown = [], [], []
    for n in names:
        if n in MUST_RUN:
            fast.append(n)
            continue
        if n in NEVER_FAST:
            slow.append(n)
            continue
        series = timings.get(n) or []
        if not series:
            unknown.append(n)
        elif median(series) > threshold:
            slow.append(n)
        else:
            fast.append(n)
    return fast, slow, unknown


def run_one(name: str, timeout: float, *,
            extra_env: dict[str, str] | None = None) -> TestResult:
    """跑一个测试，独立子进程。`extra_env` 用于并行时压住数学库线程数。"""
    path = TESTS_DIR / name
    res = TestResult(name=name)
    env = None if not extra_env else {**os.environ, **extra_env}
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout, cwd=str(ROOT), env=env,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        res.exit_code = proc.returncode
    except subprocess.TimeoutExpired as exc:
        res.seconds = time.perf_counter() - t0
        res.status = "timeout"
        res.note = f"超过 {timeout:.0f}s 未结束"
        res.tail = _tail((exc.stdout or "") if isinstance(exc.stdout, str) else "")
        return res
    except OSError as exc:
        res.seconds = time.perf_counter() - t0
        res.status = "error"
        res.note = f"{type(exc).__name__}: {exc}"
        return res

    res.seconds = time.perf_counter() - t0
    res.skipped = SKIP_MARKER in out
    passed = (PASS_MARKER in out) and proc.returncode == 0
    res.status = "pass" if passed else "fail"
    if not passed:
        res.tail = _tail(out)
        if res.exit_code not in (0, None):
            res.note = f"退出码 {res.exit_code}"
        elif PASS_MARKER not in out:
            res.note = "没有打印『" + PASS_MARKER + "』"
    return res


def _tail(text: str, n: int = 6) -> list[str]:
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    return lines[-n:]


def _print_result(idx: int, total: int, r: TestResult) -> None:
    mark = {"pass": "通过", "fail": "不达标", "timeout": "超时",
            "error": "无法运行"}[r.status]
    extra = "（跳过）" if r.skipped else ""
    print(f"\r[{idx}/{total}] {r.name:<38} {r.seconds:7.1f}s  "
          f"{'PASS' if r.status == 'pass' else 'FAIL'}  {mark}{extra}")
    if r.tail:
        for ln in r.tail:
            print(f"          | {ln}")


def run(names: list[str], timeout: float, timings: dict[str, list[float]],
        threshold: float, jobs: int = 1) -> list[TestResult]:
    if jobs > 1 and len(names) > 1:
        return _run_parallel(names, timeout, timings, threshold, jobs)
    return _run_serial(names, timeout, timings, threshold)


def _run_serial(names: list[str], timeout: float,
                timings: dict[str, list[float]], threshold: float
                ) -> list[TestResult]:
    results: list[TestResult] = []
    for i, name in enumerate(names, 1):
        print(f"[{i}/{len(names)}] {name} …", end="", flush=True)
        r = run_one(name, timeout)
        r.slow = median(timings.get(name) or []) > threshold
        _print_result(i, len(names), r)
        results.append(r)
        # 每跑完一个就更新耗时 —— 中途 Ctrl-C 也不丢已测数据
        if r.status in ("pass", "fail") and r.seconds > 0:
            note_timing(timings, name, r.seconds)
            save_timings(timings)
    return results


def _run_parallel(names: list[str], timeout: float,
                  timings: dict[str, list[float]], threshold: float,
                  jobs: int) -> list[TestResult]:
    """并行跑（`--jobs N`）。

    ## 三条必须遵守的约束

    1. **独占的放最后串行**（`EXCLUSIVE`）：`mutation_guards` 会临时改源码。
       放最后还有个额外好处 —— 万一它中途死掉留下变异体，**后面没有测试会在坏代码上跑**
       （`tests/_journal.py` 会在下次启动时还原）。
    2. **互斥组内不许重叠**（`CONFLICT_GROUPS`）：会通过 `docs/` 探针互相污染。
    3. **并行时不写耗时基线**：多个进程抢 CPU，记下来的数会比真实值大得多，
       而 `timings.json` 正是"快/慢分层"的依据 —— 污染它会让 `fast` 档悄悄跑错测试。
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import nullcontext

    group_of: dict[str, int] = {}
    for gi, grp in enumerate(CONFLICT_GROUPS):
        for n in grp:
            group_of[n] = gi
    locks = [threading.Lock() for _ in CONFLICT_GROUPS]
    print_lock = threading.Lock()
    counter = [0]

    rest = [n for n in names if n not in EXCLUSIVE]
    exclusive = [n for n in names if n in EXCLUSIVE]

    print(f"并行 {jobs} 路：{len(rest)} 个（每个子进程 BLAS 线程=1，避免互相抢）")
    if exclusive:
        print(f"独占串行（会临时改源码，且放在最后）：{', '.join(exclusive)}")
    print("说明：并行不更新 outputs/test_results/timings.json —— 抢 CPU 的耗时会污染"
          "快/慢分层")
    print()

    def task(name: str) -> TestResult:
        gi = group_of.get(name)
        with (locks[gi] if gi is not None else nullcontext()):
            r = run_one(name, timeout, extra_env=_PARALLEL_ENV)
        r.slow = median(timings.get(name) or []) > threshold
        with print_lock:
            counter[0] += 1
            _print_result(counter[0], len(names), r)
        return r

    results: list[TestResult] = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        results.extend(pool.map(task, rest))

    for name in exclusive:
        counter[0] += 1
        r = run_one(name, timeout)
        r.slow = median(timings.get(name) or []) > threshold
        _print_result(counter[0], len(names), r)
        results.append(r)

    by_name = {r.name: r for r in results}
    return [by_name[n] for n in names if n in by_name]


def _recover_stale_mutation() -> None:
    """跑任何测试之前，先收拾上次变异测试留下的残局。

    `selftest_mutation_guards` 会临时改坏源码再还原。进程被硬杀时还原不会执行，
    源码里就留着 `# ← 注入` 的变异体 —— 而**下一个测试会在坏代码上跑**，
    结果看起来正常。这套测试是"信任的根"，必须在根上先对账。
    """
    import pathlib as _pathlib                 # 本模块没在顶层 import pathlib
    import sys as _sys
    tests_dir = _pathlib.Path(__file__).resolve().parent.parent / "tests"
    if str(tests_dir) not in _sys.path:
        _sys.path.insert(0, str(tests_dir))
    try:
        from _journal import recover
    except Exception:
        return
    recovered = recover()
    if recovered is not None:
        print(f"[警告] 上次变异测试未走完，已还原 {recovered.name}；本轮结果才可信")
        print()


def _resolve_jobs(raw: str) -> int:
    """把 `--jobs` 解析成正整数；`auto` = CPU 核数（上限 8，再多是抢内存不是提效）。"""
    text = str(raw).strip().lower()
    if text in ("auto", "0"):
        return max(1, min(8, os.cpu_count() or 4))
    try:
        n = int(text)
    except ValueError:
        raise SystemExit(f"--jobs 只接受整数或 auto，收到 {raw!r}")
    return max(1, n)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="回归测试运行器（项目里唯一跑全套测试的地方）")
    ap.add_argument("tier", nargs="?", default="fast", choices=["fast", "full"],
                    help="fast=跳过已知的慢测试（默认）；full=全跑")
    ap.add_argument("--slow-threshold", type=float, default=DEFAULT_SLOW_THRESHOLD,
                    help=f"超过多少秒算『慢』（默认 {DEFAULT_SLOW_THRESHOLD:.0f}s）")
    ap.add_argument("--timeout", type=float, default=900.0,
                    help="单个测试的超时秒数（默认 900）")
    ap.add_argument("--only", nargs="+", metavar="NAME",
                    help="只跑这些（可给简写，如 calib_facts）")
    ap.add_argument("--list", action="store_true", help="只列清单与分层，不跑")
    ap.add_argument("--jobs", "-j", default="1", metavar="N",
                    help="并行路数：整数，或 auto（=CPU 核数，上限 8）。默认 1 = 串行。"
                         "会改源码的测试自动独占串行，见 EXCLUSIVE")
    args = ap.parse_args(argv)

    if not args.list:
        _recover_stale_mutation()

    all_names = discover()
    if not all_names:
        print("tests/ 下没有 selftest_*.py —— 检查是不是跑错目录了")
        return 2

    timings = load_timings()
    fast, slow, unknown = classify(all_names, timings, args.slow_threshold)

    if args.only:
        wanted = []
        for key in args.only:
            hit = [n for n in all_names
                   if n == key or n == f"selftest_{key}.py" or key in n]
            if not hit:
                print(f"没有匹配的测试：{key}")
                return 2
            wanted += hit
        selected = [n for n in all_names if n in set(wanted)]
        tier_label = "指定"
    elif args.tier == "full":
        selected = list(all_names)
        tier_label = "full"
    else:
        selected = [n for n in all_names if n not in set(slow)]
        tier_label = "fast"

    print("=" * 78)
    print(f"回归测试运行器 —— {tier_label}（共 {len(selected)}/{len(all_names)} 个）")
    print("=" * 78)
    if timings:
        print(f"分层依据：outputs/test_results/timings.json（最近 {TIMING_HISTORY} 次的"
              f"**中位数**，阈值 {args.slow_threshold:.0f}s）")
    else:
        print("分层依据：还没有实测耗时 —— 这一次会跑全套，跑完就记住了")
    print(f"保底名单（无论计时如何都跑）：{', '.join(MUST_RUN)}")
    if slow:
        print("本档跳过：")
        for n in slow:
            # ⚠️ `NEVER_FAST` 的测试**可能完全没有计时数据**（第一次就是被排除了），
            # 所以这里不能直接 `timings[n]` —— 那会在第一次引入时就 KeyError 崩掉
            # （2026-10-15 实测：引入 selftest_mutation_guards 后 `test full` 直接崩）。
            series = timings.get(n) or []
            if not series:
                print(f"    · {n}（**从不进 fast 档**：它会临时改源码）")
            else:
                print(f"    · {n}（中位 {median(series):.1f}s）")
    if unknown:
        print(f"尚无耗时记录、本档仍会跑：{len(unknown)} 个")
    print()

    if args.list:
        for n in all_names:
            series = timings.get(n) or []
            med = median(series)
            tag = "?" if not series else ("慢" if med > args.slow_threshold else "快")
            detail = "" if not series else f"中位 {med:7.1f}s  历史 {series}"
            print(f"  [{tag}] {n:<38} {detail}")
        return 0

    jobs = _resolve_jobs(args.jobs)
    if jobs > 1:
        print(f"并行：{jobs} 路（-j {args.jobs}）")
        print()

    t0 = time.perf_counter()
    results = run(selected, args.timeout, timings, args.slow_threshold, jobs)
    total = time.perf_counter() - t0

    n_pass = sum(1 for r in results if r.status == "pass")
    n_skip = sum(1 for r in results if r.skipped and r.status == "pass")
    failed = [r for r in results if r.status != "pass"]

    print()
    print("=" * 78)
    print(f"合计 {len(results)} 个：通过 {n_pass}（其中 {n_skip} 个含跳过项）、"
          f"不达标 {len(failed)}，用时 {total:.1f}s")
    if failed:
        for r in failed:
            print(f"  FAIL  {r.name}  ({r.status}"
                  f"{', ' + r.note if r.note else ''})")
    print("判定结果：" + ("全部通过" if not failed else "有不达标项"))
    print("=" * 78)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = RESULTS_DIR / f"{stamp}.json"
    out_file.write_text(json.dumps({
        "tier": tier_label,
        "started": stamp,
        "slow_threshold": args.slow_threshold,
        "total_seconds": round(total, 2),
        "counts": {"total": len(results), "passed": n_pass,
                   "failed": len(failed), "skipped": n_skip},
        "skipped_slow": slow,
        "results": [r.to_json() for r in results],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"结果已写：{out_file.relative_to(ROOT).as_posix()}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
