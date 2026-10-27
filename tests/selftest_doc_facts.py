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
**5. 权威实现的两条**自身**纪律**（2026-10-24 补）：`KNOWN_MENTIONS` 没有死条目、
   `STC/` 引用按仓库外（`..\\STC\\`）解析 —— 否则白名单会变成永久豁免、
   而仓库外引用会被误判成失效。

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
白名单里挂一条永不命中的条目          `KNOWN_MENTIONS` 加一条假 key              exit=1
`STC/` 解析根改回仓库内               `_outside_path` 返回 `ROOT / cand`         exit=1
`STC/` 引用一律跳过（漏报）           在 `_outside_path` 后直接 `continue`        exit=1
====================================  ==========================================  ========

用法::

    python tests/selftest_doc_facts.py
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
while not (ROOT / "sphere_tracker").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

AGENTS_MD = ROOT / "AGENTS.md"
CHECKER = ROOT / "apps" / "tools" / "check_standards.py"
SBOM_TOOL = ROOT / "apps" / "tools" / "build_sbom.py"
SYNCER = ROOT / "apps" / "tools" / "sync_facts.py"

# 本测试自己也是测试，也要被数进去。
#
# ⚠️ **这个数字是"有意的确认门"，不是可以自动推导的东西** —— 别把它改成
# `len(glob(...))`，那就是重言式（自己数自己，恒真）。
# 分工：**"漏登记"由第 2 节自动抓**（集合差），
# 这里只负责"数量变了要有人点头"。往 `tests/` 加文件后，
# 改这个数字就是那次点头 —— 忘了改它会红，那是**设计如此**。
# 它守的是另一件事：**有人悄悄删掉一个测试**时，数量对不上就会有人注意到。
# 但请注意：**这道门不是唯一的门** —— 第 2 节还要求"磁盘上每个
# `selftest_*.py` 都在 `AGENTS.md` §7 登记"，两条一起才成立
# （只靠这里，删掉一个测试再改这个数字就无声无息；只靠第 2 节，
#  删干净后登记表也一起删，同样无声无息）。
# 历次：21 → 22（加 `selftest_window_mode.py`）→ 23（加 `selftest_mutation_guards.py`）
#       → 24（2026-10-26 加 `selftest_plugins.py`：插件注册表不变量 + R5/R7 边界）。
REAL_TEST_COUNT = 24
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
# 仓库外（`STC/`）探针：同样**必须拼出来**，理由同上（别让检查器把它当合法字面量）。
STC_PROBE_DOC = ROOT / "docs" / "_zz_tmp_stc_probe.md"
STC_PROBE_REF = "STC/scripts/zz_" + "not_a_real_stc.ps1"
# 运行期可再生产物（`config/`）探针：同样拼出来，理由同上。
CFG_PROBE_DOC = ROOT / "docs" / "_zz_tmp_cfg_probe.md"
CFG_PROBE_REF = "config/zz_" + "phantom.json"


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
    """第 3 节：权威实现存在、可跑、四类检查都做了。

    ⚠️ **2026-10-15 改了一条断言，记在这里**：原来有一条

        ("已知真漂移仍被报出（`用户手册` 承诺的 config 产物）",
         "stereo_calibration" in out)

    它把"**这两条漂移还在**"钉成了契约。而 2026-10-14 23:08 用户真机标定
    **成功并落盘**了 `config/stereo_calibration.*`，那个漂移真的修好了 ——
    于是检查器按设计报了「已知真漂移里有条目已不再出现（修好了？）……否则它会
    变成永久豁免」，我把两条从 `KNOWN_DRIFT` 删掉，这条断言也就必然失败。

    **那条断言本身是错的**：`KNOWN_DRIFT` 的整个设计意图就是"**应当变空**"
    （`check_standards.py` 里写着"修好之后应当变空，而不是被悄悄从白名单里删掉"）。
    把一个"待修的缺口"钉成测试契约，等于**要求缺陷一直存在** ——
    缺口补上时测试反而变红，那是在惩罚修好它的人。
    现在改成断言"这一类**能力**还在"（条目清空时也不该报"意外失效引用"）。
    """
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
        ("退出码 0（四类检查全过）", code == 0),
        ("确实跑了 A 类：路径引用", "A. 文档里的仓库路径引用" in out),
        ("确实跑了 B 类：派生事实（委托 sync_facts）", "B. 派生事实" in out),
        ("确实跑了 C 类：规范来源", "C. 规范来源是否存在" in out),
        # D 类是 2026-10-27 加的：待办的唯一来源（docs/待办清单.md）自洽性、
        # done 能不能对上 git、文档里的待办章节有没有人登记。
        # 这里只断言"它真的跑了"——具体判据由 check_standards 自己守。
        ("确实跑了 D 类：待办清单", "D. 待办清单" in out),
        # 能力判据（2026-10-15 换掉一条"要求缺陷一直存在"的坏断言，见本函数文档串）：
        # 白名单**已清空**（真机标定落盘后那两条漂移真的消掉了），
        # 此时检查器必须：① 如实报"已知真漂移 0 条"；② **不许**报出任何 `[真漂移，待修]`。
        # 这两条合起来才有意义 —— 下一个人若往 KNOWN_DRIFT 里加新条目，② 就会报红。
        ("白名单已清空且如实报出（`已知真漂移 0 条`）", "已知真漂移 0 条" in out),
        ("当前没有任何 `[真漂移，待修]`（有的话检查器会报出来）",
         "[真漂移，待修]" not in out),
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


def _load_checker():
    """把权威实现当模块加载（要直接看它的白名单与解析结果，不只跑 subprocess）。"""
    spec = importlib.util.spec_from_file_location("check_standards_under_test", CHECKER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_whitelist_vanish_and_outside_repo() -> bool:
    """第 5 节：白名单不许变成永久豁免；`STC/` 引用按仓库外解析。

    这两条钉的是 2026-10-24 那轮修掉的**检查器自身缺陷**：

    - `KNOWN_MENTIONS` 原先只增不减：文档改了、或那个文件真的补上了，
      白名单条目就永远沉默（`KNOWN_DRIFT` 有 vanished 检测，它没有）；
    - `STC/...` 原先只在仓库内解析，实际在 `..\\STC\\`，于是**真实存在的文件被判失效**
      （假阳性），而这批假阳性又因为"算出 `outside` 却从不进 problems"被静默吞掉。

    ## 这些断言怎么才能红（想清楚再改）

    | 断言 | 什么改动会让它红 |
    | ---- | ---------------- |
    | 白名单无死条目 | 往 `KNOWN_MENTIONS` 加一条不命中的键（或删掉 vanished 检测） |
    | 死条目检测真的生效 | 删掉 `check_refs` 里的 `dead_mentions` 分支（下面就地注入一条假条目验证） |
    | `STC/` 存在的引用不算失效 | 把解析根改回 `ROOT`（存在的文件会被判失效，假阳性回来了） |
    | `STC/` 不存在的引用必须报错 | 改成"`STC/` 一律跳过"（漏报回来了） |
    | `config/` 引用不阻断 | 删掉 `collect_refs` 里的 `_is_runtime_reproducible` 分支（它回落进 stale，探针那条会让 exit 1） |
    | `config/` 引用仍被归类 | 改成"`config/` 直接 `continue`"（静默吞掉，`cfg_classified` 变 False） |
    """
    print("----- 5. 白名单 vanished 检测 + STC/ 仓库外解析 -----")
    if not CHECKER.exists():
        print("  [不达标] check_standards.py 不存在")
        return False
    cs = _load_checker()
    stale, outside_present, runtime_repro = cs.collect_refs()

    # ---- ① 白名单无死条目 ----
    dead = sorted(k for k in cs.KNOWN_MENTIONS if k not in stale)
    print(f"  KNOWN_MENTIONS {len(cs.KNOWN_MENTIONS)} 条，命中 {len(cs.KNOWN_MENTIONS) - len(dead)} 条，"
          f"死条目 {len(dead)} 条")
    for k in dead[:10]:
        print(f"    [死条目，请删] {k}")

    # ---- ② 死条目检测本身必须活着（就地注入一条假条目，看检查器报不报） ----
    fake = "docs/__zz_no_such_doc__.md:" + "zz_" + "phantom.py"
    cs.KNOWN_MENTIONS[fake] = "反向自证：这条永远不命中，探 vanished 检测还活着"
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            probs = cs.check_refs(verbose=False)
        caught_vanished = any(fake in p for p in probs)
    finally:
        cs.KNOWN_MENTIONS.pop(fake, None)
    print(f"  注入一条假白名单条目后：检查器{'报出来了' if caught_vanished else '**没报**'}")

    # ---- ③ `STC/` 引用按 `ROOT.parent` 解析 ----
    stc_present = {k: v for k, v in outside_present.items() if v.startswith("STC/")}
    leaked = sorted(k for k in stc_present if k in stale)          # 存在却被判失效 = 假阳性
    ghost = sorted(k for k in stale                                # 不存在却没人管 = 漏报
                   if k.split(":")[1].replace("\\", "/").startswith("STC/"))
    print(f"  仓库外（存在）{len(stc_present)} 处、被判失效 {len(leaked)} 处、"
          f"不存在的 `STC/` 引用 {len(ghost)} 处（应由白名单或 problems 接住）")
    for k in leaked:
        print(f"    [假阳性] {k}")

    # ---- ④ 反向自证：造一处不存在的 `STC/` 引用，检查器必须报红 ----
    STC_PROBE_DOC.write_text(f"# 临时探针（本测试创建，跑完就删）\n\n见 `{STC_PROBE_REF}`。\n",
                             encoding="utf-8")
    try:
        code, out = _run(CHECKER, "--quiet")
    finally:
        STC_PROBE_DOC.unlink(missing_ok=True)
    caught_stc = "zz_not_a_real_stc.ps1" in out
    code2, _out2 = _run(CHECKER, "--quiet")
    print(f"  造一处不存在的 STC 引用：退出码 {code}、被抓到 {caught_stc}；删掉后退出码 {code2}")

    # ---- ⑤ `config/` 属运行期可再生产物：**显式归类、但绝不阻断** ----
    #
    # 钉的是 2026-10-24 那个回归：`config/` 整目录被 `.gitignore` 忽略
    #（`AGENTS.md` §3 规则 3：能重新生成的东西不进版本库），
    # 一段时间里它的 3 条 white-list 条目被当成"死条目"删了，
    # 于是这份可再生产物被**升级成硬承诺** —— 删掉那个 json 就 exit 1。
    # 修法是让豁免挂在**路径类别**上：不管文件在不在，都不进 stale。
    CFG_PROBE_DOC.write_text(f"# 临时探针（本测试创建，跑完就删）\n\n见 `{CFG_PROBE_REF}`。\n",
                             encoding="utf-8")
    try:
        code_cfg, out_cfg = _run(CHECKER, "--quiet")
        # 必须**出现在分类里**（不是被静默吞掉 —— 那和"白名单永久豁免"同样坏）
        _s, _o, repro_cfg = cs.collect_refs()
        cfg_key = f"docs/{CFG_PROBE_DOC.name}:{CFG_PROBE_REF}"
        cfg_classified = cfg_key in repro_cfg
        cfg_in_stale = cfg_key in _s
    finally:
        CFG_PROBE_DOC.unlink(missing_ok=True)
    code_cfg2, _out_cfg2 = _run(CHECKER, "--quiet")
    print(f"  造一处不存在的 config 引用：退出码 {code_cfg}（必须 0）、"
          f"归入「运行期可再生产物」{cfg_classified}、误进 stale {cfg_in_stale}；"
          f"删掉后退出码 {code_cfg2}")

    checks = [
        ("KNOWN_MENTIONS 没有死条目（白名单不会变成永久豁免）", not dead),
        ("死条目会被 check_refs 报出来（不是只算不用）", caught_vanished),
        ("仓库外 `STC/` 引用确实按父目录解析到了（不是空集，断言没空跑）",
         len(stc_present) > 0),
        ("`STC/` 里存在的引用**不被**判为失效（P2 假阳性已修）", not leaked),
        ("不存在的 `STC/` 引用没有漏网（要么进 problems，要么在白名单里）",
         all(k in cs.KNOWN_MENTIONS or k in cs.KNOWN_DRIFT for k in ghost)),
        ("新造一处不存在的 `STC/` 引用会让检查器报红", code != 0),
        ("并且指出了那个路径（不只是「有错」）", caught_stc),
        ("删掉探针后恢复通过（不是靠永久豁免过的）", code2 == 0),
        ("`config/` 是不存在的路径时**也不阻断**（可再生产物不升级成硬承诺）",
         code_cfg == 0),
        ("但它**必须被显式归类**为「运行期可再生产物」（不是静默吞掉）",
         cfg_classified),
        ("并且没有掉进 stale（豁免挂在路径类别上，不挂在文件是否存在上）",
         not cfg_in_stale and any(
             k.split(":")[1].replace("\\", "/").startswith("config/")
             for k in runtime_repro)),
        ("删掉 config 探针后仍通过", code_cfg2 == 0),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_handover_section_13_claims() -> bool:
    """交接文档 §13 里写的具体事实，必须与代码一致。

    ## 为什么这条要有

    `AGENTS.md` §6：「**代码与规范冲突 = 缺陷，不是"谁赢"的问题**……
    并存就是漂移的开始，而这正是本项目反复出现的头号问题。」

    `check_standards.py` 能查**引用**（文件/行号存不存在）与**计数**，
    但它查不出"文档说默认是 25、代码里其实改成了别的"这类**语义漂移**。
    所以这里把 §13（2026-10-13/14 那一轮）里写下的关键事实逐条对着代码核。

    **只钉"文档与代码一致"，不钉具体数值本身** —— 数值可以改，
    但改了就必须同时改文档，否则这条会红。这正是我们要的。
    """
    print("----- 6. 交接文档 §13 的断言 vs 代码 -----")
    handover = (ROOT / "docs" / "交接文档.md").read_text(encoding="utf-8")
    # §13 必须存在（否则下面那些核对是在"文档被删了"的情况下空跑）
    checks: list[tuple[str, bool]] = [
        ("交接文档里有 §13（这一轮的记录没被删）", "## 13. " in handover),
    ]

    # (说明, 文件, 必须出现的片段)
    FACTS = [
        ("标定默认 40 / 20", "apps/calibrate_stereo.py",
         ["default=40", "default=20"]),
        ("GUI 把 --display-width 传下去", "启动程序.py", ["--display-width"]),
        ("GUI 用 resolve_roles（两个角色不撞车）", "启动程序.py", ["resolve_roles"]),
        ("GUI 有任务管理器且杀进程树", "启动程序.py", ["def stop_task", "taskkill"]),
        ("同名相机歧义会被拦下", "sphere_tracker/camera_registry.py",
         ["class AmbiguousMatch", "def resolve_roles"]),
        ("「打开了但没帧」报错且不自动换后端", "sphere_tracker/sources.py",
         ["不会自动换后端", "拔掉相机再插回", "两路必须同后端"]),
        ("搜索有时间预算", "sphere_tracker/sources.py", ["search_budget_s"]),
        ("hough 有自己的半径上限", "sphere_tracker/detectors/hough.py",
         ["hough_max_radius"]),
        ("配置里有 hough_max_radius 项", "sphere_tracker/config.py",
         ["hough_max_radius"]),
        ("显示与检测两级缩放分开", "sphere_tracker/pipeline.py",
         ["display_width", "render_scale", "set_detect_width"]),
        ("Detection.scaled 存在（坐标换算）", "sphere_tracker/types.py",
         ["def scaled"]),
        ("实时检测装了日志", "apps/run_realtime.py", ['setup_logging("realtime")']),
        ("实时检测有 --detect-width", "apps/run_realtime.py", ["--detect-width"]),
        ("采集时实时提示板占比例", "calib/capture.py", ["def area_hint"]),
        ("标定四级判定", "calib/stereo_calib.py", ["勉强可用", "board_area"]),
        ("窗口助手用 setWindowProperty 开全屏", "apps/_window.py",
         ["setWindowProperty", "WND_PROP_FULLSCREEN"]),
    ]
    for label, rel, needles in FACTS:
        p = ROOT / rel
        if not p.exists():
            checks.append((f"{label}（文件存在）", False))
            continue
        src = p.read_text(encoding="utf-8")
        missing = [n for n in needles if n not in src]
        ok = not missing
        if not ok:
            print(f"    {rel} 缺：{missing}")
        checks.append((label, ok))

    # GUI 标定面板的默认值：断言**值**，不断言"源码里有这几个字符串"（2026-10-26）。
    # 旧写法要求 启动程序.py 里出现 '"40"'/'"20"'/'"24"' —— 一旦控件换成 IntVar/Spinbox，
    # 行为完全正确也会红。现在默认值提成模块常量，直接读常量值。
    try:
        import 启动程序 as _gui
        got = (_gui.CALIB_TARGET_PAIRS, _gui.CALIB_MIN_PAIRS, _gui.CALIB_FPS_DEFAULT)
    except Exception as exc:                      # noqa: BLE001 - 导入失败本身就是问题
        got, err = None, f"{type(exc).__name__}: {exc}"
        print(f"    导入 启动程序 失败：{err}")
    checks.append(("GUI 标定面板默认 40 / 20 / 24（按常量值断言，不靠源码字面量）",
                   got == (40, 20, "24")))

    # 默认检测方法必须是 color（§13.5 的核心结论之一）
    cfg = (ROOT / "sphere_tracker" / "config.py").read_text(encoding="utf-8")
    checks.append(("默认检测方法是 color（不是 fusion）", 'method: str = "color"' in cfg))

    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_sbom_builder() -> bool:
    """第 7 节：SBOM 生成器能生成、能检查、坏输入不崩。

    为什么放在这里：SBOM 是"文档与事实"的一部分，且它是**当前机器环境**的快照，
    不适合要求别的机器也一致；所以这里测**结构 / 行为**，不钉死 59 或 66。
    """
    print("----- 7. SBOM 生成器（build_sbom.py） -----")
    checks: list[tuple[str, bool]] = [
        ("build_sbom.py 存在", SBOM_TOOL.exists()),
    ]
    if not SBOM_TOOL.exists():
        for name, ok in checks:
            print(f"  [{'通过' if ok else '不达标'}] {name}")
        return False

    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td)
        spec = importlib.util.spec_from_file_location("sbom_builder", SBOM_TOOL)
        sbom = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sbom)

        code, _out = _run(SBOM_TOOL, "--help")
        checks.append(("--help 退出码 0", code == 0))
        code, _out = _run(SBOM_TOOL, "--out-dir", td, "--quiet")
        checks.append(("生成退出码 0", code == 0))
        md_path = out_dir / "SBOM.md"
        json_path = out_dir / "sbom.cdx.json"
        checks.append(("两份产物都存在", md_path.exists() and json_path.exists()))

        if md_path.exists() and json_path.exists():
            bom = json.loads(json_path.read_text(encoding="utf-8"))
            refs = [c.get("bom-ref") for c in bom.get("components", [])]
            checks.append((
                "CycloneDX 1.5",
                bom.get("bomFormat") == "CycloneDX" and bom.get("specVersion") == "1.5",
            ))
            checks.append(("bom-ref 唯一且非空",
                           bool(refs) and len(refs) == len(set(refs)) and all(refs)))
            checks.append((
                "依赖引用都指向已存在的 component",
                all(x in set(refs) for d in bom.get("dependencies", [])
                    for x in d.get("dependsOn", [])),
            ))
            md = md_path.read_text(encoding="utf-8")
            counts = [int(x) for x in re.findall(
                r"^## [123]\..*?（(?:[^（）]*，)?(\d+)）", md, re.M)]
            pypi = sum(1 for c in bom.get("components", [])
                       if str(c.get("purl", "")).startswith("pkg:pypi/"))
            checks.append(("Markdown 三类计数之和 = pypi 组件数",
                           len(counts) == 3 and sum(counts) == pypi))

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code_q = sbom.check(out_dir, quiet=True)
            checks.append(("刚生成时 --check 退出 0", code_q == 0))
            checks.append(("--quiet 且最新时无输出", buf.getvalue().strip() == ""))

            # 再生成一次：时间戳 / serial 会变，但语义未变 -> --check 仍应 0
            code_r = sbom.write(out_dir, quiet=True)
            code_c = sbom.check(out_dir, quiet=True)
            checks.append(("重新生成（时间戳/serial 变化）后 --check 仍 0",
                           code_r == 0 and code_c == 0))

            # 只改 Markdown 也应报过期（不是只看 JSON）
            md_path.write_text(md + "\n<!-- stale -->\n", encoding="utf-8")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code_m = sbom.check(out_dir, quiet=True)
            checks.append(("只改 Markdown -> --check 退出 1",
                           code_m == 1 and "[过期]" in buf.getvalue()))
            sbom.write(out_dir, quiet=True)     # 恢复

            numpy = next((c for c in bom.get("components", [])
                          if c.get("name") == "numpy"), None)
            if numpy is None:
                checks.append(("能找到 numpy component 以注入漂移", False))
            else:
                numpy["version"] = "9.9.9"
                json_path.write_text(
                    json.dumps(bom, indent=2, ensure_ascii=False), encoding="utf-8")
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    code3 = sbom.check(out_dir)
                out3 = buf.getvalue()
                checks.append(("版本漂移 -> --check 退出 1", code3 == 1))
                checks.append(("漂移输出指出 numpy / 9.9.9",
                               "numpy" in out3 and "9.9.9" in out3))

            json_path.write_text("{not json", encoding="utf-8")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code4 = sbom.check(out_dir)
            out4 = buf.getvalue()
            checks.append(("坏 JSON -> 退出 1，且不抛 traceback",
                           code4 == 1 and "Traceback" not in out4))

            json_path.unlink()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code5 = sbom.check(out_dir, quiet=True)
            checks.append(("缺文件 -> 退出 1 且提示更新",
                           code5 == 1 and "更新" in buf.getvalue()))

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
               test_checker_catches_a_brand_new_drift,
               test_whitelist_vanish_and_outside_repo,
               test_handover_section_13_claims,
               test_sbom_builder):
        results.append(fn())
        print()
    ok = all(results)

    # ⚠️ 失败了就把"是哪一节红的"再打一遍。
    #
    # 为什么需要：本测试有 **7 节、几十条检查**，而运行器
    # （`apps/run_tests.py`）失败时只回显**最后几行** —— 于是"哪一节红的"
    # 恰好被截掉，只剩一个孤零零的 `有不达标项`。
    # 2026-10-14 就撞上过一次：整套里红、单独跑连跑 8 次全绿、
    # `check_standards.py` 连跑 8 次退出码全 0、`启动程序.py` 的相关串也都在，
    # **根因没能定位**（疑为并发改文件时检查器扫到中间态）。
    # 那次之所以难查，一半原因就是**输出被截断**。
    # 所以把结论行放在最后、并且逐节重述 —— 下次至少能一眼看出该查哪一节。
    if not ok:
        names = ("1 测试清单与数量", "2 登记一致", "3 权威实现可用",
                 "4 反向自证能抓新漂移", "5 白名单 vanished + STC 仓库外解析",
                 "6 交接文档 §13 断言 vs 代码", "7 SBOM 生成器")
        print("红的节：")
        for name, good in zip(names, results):
            print(f"  [{'通过' if good else '不达标'}] {name}")
        print("\n单独复跑以定位（运行器的尾部回显会截掉上面那些明细）：")
        print(f"  {sys.executable} tests/selftest_doc_facts.py")

    print("判定结果：" + ("全部通过" if ok else "有不达标项"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
