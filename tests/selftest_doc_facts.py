"""文档事实自测：在回归套件里跑一遍规范落差检查。

⚠️ **检查逻辑本身在 `apps/tools/check_standards.py`**（2026-10-10 迁走）。
本文件是**测试外壳**：调用那个权威实现并核对结果。

为什么这么分
------------
同一套"引用 vs 谈及"的判别逻辑写两遍，两份迟早不一致 ——
而"同一个假设写在两处就会漂移成两种行为"正是本项目反复踩的坑。
所以分工写死：

| 谁 | 负责 |
| -- | ---- |
| `apps/tools/check_standards.py` | **权威实现**：路径引用 / 规范来源 / 派生事实 三类检查；`启动程序.bat standards` 调它 |
| `apps/tools/sync_facts.py` | **计数的权威实现**（被上面那个委托） |
| 本文件 | **测试外壳**：钉住"清单与登记不退化"，并验证权威实现真的能抓到新漂移 |

本测试钉四件事：

**1. 测试清单**：数量、命名规范（`AGENTS.md` §4 要求 `selftest_<被守护的模块>.py`）、
   以及**每一个都被 `AGENTS.md` §7 的表登记**。
**2. 清单与登记的一致性** —— 新加测试却忘了登记，会在这一条红。
**3. 权威实现可用**：`check_standards.py` 存在、退出码 0、三类检查都真的跑了。
**4. 权威实现**真的**能抓到新漂移**（就地造一处失效引用，跑完删掉）。

反向自证（改了就会红）
--------------------
====================================  ==========================================  ========
改动                                  注入方式                                    结果
====================================  ==========================================  ========
谎报测试数量                          `REAL_TEST_COUNT = 9`                       exit=1
把 `AGENTS.md` 的 (7) 改成 (8)        改 `AGENTS.md` 的分组计数                   exit=1
新增一个未登记的测试文件              在 `tests/` 放一个 `selftest_zzz.py`        exit=1
白名单之外新增失效引用                在 `docs/` 里写一个不存在的路径             exit=1
把权威实现改名 / 删掉                 `apps/tools/check_standards.py` -> 别的名   exit=1
====================================  ==========================================  ========

用法::

    python tests/selftest_doc_facts.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
while not (ROOT / "sphere_tracker").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

AGENTS_MD = ROOT / "AGENTS.md"
CHECKER = ROOT / "apps" / "tools" / "check_standards.py"
SYNCER = ROOT / "apps" / "tools" / "sync_facts.py"

# 本测试自己也是测试，也要被数进去
REAL_TEST_COUNT = 18
# 测试命名规范（AGENTS.md §4）
TEST_NAME_RE = re.compile(r"^selftest_[a-z0-9_]+\.py$")
# 临时探针用的假路径（造一处新漂移用；用完即删）
PROBE_DOC = ROOT / "docs" / "_zz_tmp_drift_probe.md"
# ⚠️ 这个假路径**必须拼出来，不能写成完整字面量**。
# 原因：`check_standards.py` 会把"在源码里作为字符串字面量出现过的路径"视为
# 合法（运行时生成的文件，如 `config/sphere_config.json`）。
# 如果这里写成完整字面量，检查器就会认为它是合法路径、不再报红 ——
# **反向自证会静默失效，测试反而报"不达标"**（第一版就是这样，查了一轮才定位）。
PROBE_REF = "apps/tools/" + "zz_" + "not_a_real_file.py"


def _test_files() -> list[Path]:
    return sorted((ROOT / "tests").glob("selftest_*.py"))


def _run(script: Path, *args: str) -> tuple[int, str]:
    proc = subprocess.run([sys.executable, str(script), *args],
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", cwd=str(ROOT))
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _show(out: str) -> None:
    for ln in out.splitlines():
        s = ln.rstrip()
        if s.startswith(("✅", "  [", "    ·", "    [", "**发现")) or "判定结果" in s:
            print(f"    {s}")


# --------------------------------------------------------------------------
def test_count_and_naming() -> bool:
    """第 1 节：测试数量与命名规范。"""
    print("----- 1. 测试清单与数量 -----")
    files = _test_files()
    names = [f.name for f in files]
    bad_names = [n for n in names if not TEST_NAME_RE.match(n)]

    print(f"  tests/selftest_*.py 实际 {len(files)} 个（本测试认为应当是 {REAL_TEST_COUNT}）")
    for n in names:
        print(f"    {n}")

    checks = [
        (f"数量 = {REAL_TEST_COUNT}（实际 {len(files)}）", len(files) == REAL_TEST_COUNT),
        ("命名都符合 `selftest_<被守护的模块>.py`（全小写下划线）", not bad_names),
    ]
    if bad_names:
        print(f"  不合规的命名：{bad_names}")
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_tests_are_registered_in_agents_md() -> bool:
    """第 2 节：每个测试都被 `AGENTS.md` §7 的表登记（新加的不能漏写）。"""
    print("----- 2. 每个测试都在 AGENTS.md §7 登记 -----")
    text = AGENTS_MD.read_text(encoding="utf-8")
    registered = set(re.findall(r"`(selftest_[a-z0-9_]+\.py)`", text))
    actual = {f.name for f in _test_files()}
    missing = sorted(actual - registered)
    extra = sorted(registered - actual)

    print(f"  AGENTS.md 登记 {len(registered)} 个 / 磁盘上有 {len(actual)} 个")
    if missing:
        print(f"  磁盘上有、AGENTS.md 没写：{missing}")
    if extra:
        print(f"  AGENTS.md 写了、磁盘上没有：{extra}")
    checks = [
        ("没有「磁盘有但文档没登记」的测试", not missing),
        ("没有「文档登记了但磁盘没有」的测试", not extra),
        # 分组表也必须覆盖：这条以前散在这里，现在归 check_standards.py，
        # 但本测试仍断言"权威实现真的在查它"（见第 3 节）。
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_authoritative_checker_passes() -> bool:
    """第 3 节：权威实现存在、可跑、三类检查都做了、且报出已知真漂移。"""
    print("----- 3. 权威实现 check_standards.py -----")
    checks = [
        ("check_standards.py 存在（检查逻辑不许只活在测试里）", CHECKER.exists()),
        ("sync_facts.py 存在（它是计数的权威实现）", SYNCER.exists()),
    ]
    if not CHECKER.exists():
        for name, ok in checks:
            print(f"  [{'通过' if ok else '不达标'}] {name}")
        return False

    code, out = _run(CHECKER, "--quiet")
    print(f"  退出码 = {code}")
    _show(out)
    checks += [
        ("退出码 0（三类检查全过）", code == 0),
        ("确实跑了 A 类：路径引用", "A. 文档里的仓库路径引用" in out),
        ("确实跑了 B 类：派生事实（委托 sync_facts）", "B. 派生事实" in out),
        ("确实跑了 C 类：规范来源", "C. 规范来源是否存在" in out),
        ("已知真漂移仍被报出（`用户手册` 承诺的 config 产物）",
         "stereo_calibration" in out),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_checker_catches_a_brand_new_drift() -> bool:
    """第 4 节：权威实现能抓到**新**漂移（就地反向自证，用完即删）。"""
    print("----- 4. 反向自证：临时造一处新失效引用 -----")
    PROBE_DOC.write_text(f"# 临时探针（本测试创建，跑完就删）\n\n见 `{PROBE_REF}`。\n",
                         encoding="utf-8")
    try:
        code, out = _run(CHECKER, "--quiet")
    finally:
        PROBE_DOC.unlink(missing_ok=True)
    caught = "zz_not_a_real_file.py" in out
    print(f"  造了一处失效引用后：退出码 = {code}，被抓到 = {caught}")
    if caught:
        for ln in out.splitlines():
            if "zz_not_a_real_file.py" in ln:
                print(f"    {ln.strip()}")
    code2, out2 = _run(CHECKER, "--quiet")
    print(f"  删掉探针之后：退出码 = {code2}")

    checks = [
        ("新失效引用让检查器报红", code != 0),
        ("并且指出了那个路径（不只是「有错」）", caught),
        ("删掉之后恢复通过（不是靠永久豁免过的）", code2 == 0),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def main() -> int:
    print("=" * 70)
    print("文档事实自测（外壳：调用 check_standards.py 这个权威实现）")
    print("=" * 70)
    results = []
    for fn in (test_count_and_naming,
               test_tests_are_registered_in_agents_md,
               test_authoritative_checker_passes,
               test_checker_catches_a_brand_new_drift):
        results.append(fn())
        print()
    ok = all(results)
    print("判定结果：" + ("全部通过" if ok else "有不达标项"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
