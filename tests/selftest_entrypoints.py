"""入口一致性自测：三个启动入口必须调**同一个**测试运行器与规范检查器。

为什么需要它
------------
"跑回归测试"这件事在项目里有**三个入口**：

| 入口 | 谁在用 |
| ---- | ------ |
| `启动程序.bat test` | 双击 / 命令行 |
| `启动程序.py` 的按钮   | 图形界面 |
| `启动程序_命令行.py` 的菜单 | 无 tkinter 时 |

它们的**实际缺陷是同一个**：全部硬编码只跑 `selftest_synthetic.py` 一个，
而 `AGENTS.md` §7 要求跑 15 个。这个缺陷在 `交接文档.md` §10.7 第 5 条里
被点名过（"界面上值得加一个『跑全部回归』的入口"），但一直只讨论 `.bat`。

**只修一处不算修好**：另外两处会继续只跑一个测试，而且没人会发现 ——
因为"按钮能用"看起来就是好的。所以本测试把"三个入口指向同一个运行器"
变成可执行的断言。

同理适用于规范自查（`check_standards.py`），以及"一条命令跑完整条链"的
`apps/run_all.py`：它必须是**调用**那三个脚本，而不是把三步逻辑抄进来
（抄进去 = 和 GUI 各一份，迟早漂 —— 本项目第 6 节的第 7 项就守这一条）。

反向自证（改了就会红）
--------------------
========================================  ==========================================  ========
改动                                      注入方式                                    结果
========================================  ==========================================  ========
某一个入口退回硬编码单个测试              把 `.bat` 的 `:do_test` 改回               exit=1
                                          `tests\\selftest_synthetic.py`
删掉运行器                              `apps/run_tests.py` 改名                     exit=1
命令行界面少了 `f` / `s` 菜单项           从 ACTIONS 里删一个键                       exit=1
`run_all.py` 跳步时不说明原因             把 `_skip(...)` 的 reason 改成空串          exit=1
`run_all.py` 把"跳过"当"通过"             退出码固定返回 0                            exit=1
数据目录又出现裸拼 `data/sessions`         往 `apps/run_all.py` 塞                     exit=1
                                          `X = ROOT / "data" / "sessions"`
`stereo_roots()` 顺序反过来               把元组改成 (旧, 新)                        exit=1
========================================  ==========================================  ========

用法::

    python tests/selftest_entrypoints.py
"""

from __future__ import annotations

import json
import os
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

BAT = ROOT / "启动程序.bat"
GUI = ROOT / "启动程序.py"
CLI = ROOT / "启动程序_命令行.py"
RUNNER = ROOT / "apps" / "run_tests.py"
RUN_ALL = ROOT / "apps" / "run_all.py"
STANDARDS = ROOT / "apps" / "tools" / "check_standards.py"

# 旧的、硬编码的写法 —— 出现即视为"没修好"
LEGACY_HARDCODE = "tests\\selftest_synthetic.py"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8-sig", errors="replace")


def test_runner_and_checker_exist() -> bool:
    """第 1 节：被三个入口共同依赖的两个脚本存在。"""
    print("----- 1. 公共设施存在 -----")
    checks = [
        (f"{RUNNER.relative_to(ROOT).as_posix()} 存在", RUNNER.exists()),
        (f"{STANDARDS.relative_to(ROOT).as_posix()} 存在", STANDARDS.exists()),
        ("三个入口文件都在", all(p.exists() for p in (BAT, GUI, CLI))),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_bat_uses_runner() -> bool:
    """第 2 节：`.bat` 的 test / standards 都走公共设施，且不再硬编码。"""
    print("----- 2. 启动程序.bat -----")
    text = _read(BAT)
    body = "\n".join(ln for ln in text.splitlines()
                     if not ln.strip().lower().startswith("rem"))
    has_fast = "run_tests.py fast" in body
    has_full = "run_tests.py full" in body
    has_std = "check_standards.py" in body
    hardcoded = LEGACY_HARDCODE in body
    routes_test = bool(re.search(r'if\s+/i\s+"%MODE%"=="test"\s+goto\s+:do_test', text))
    routes_std = bool(re.search(r'if\s+/i\s+"%MODE%"=="standards"\s+goto\s+:do_standards', text))

    print(f"  test   -> run_tests.py fast: {has_fast} / full: {has_full}")
    print(f"  standards -> check_standards.py: {has_std}")
    print(f"  仍然硬编码 {LEGACY_HARDCODE}: {hardcoded}")

    checks = [
        ("`test` 走 run_tests.py fast", has_fast),
        ("`test full` 走 run_tests.py full", has_full),
        ("`standards` 走 check_standards.py", has_std),
        (f"正文里不再有硬编码的 {LEGACY_HARDCODE}", not hardcoded),
        ("参数分派里有 test 分支", routes_test),
        ("参数分派里有 standards 分支", routes_std),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_gui_uses_runner() -> bool:
    """第 3 节：图形界面的测试/自查按钮走公共设施。"""
    print("----- 3. 启动程序.py（图形界面） -----")
    text = _read(GUI)
    has_fast = '"run_tests.py"), "fast"' in text or "'run_tests.py'), 'fast'" in text
    has_full = '"run_tests.py"), "full"' in text or "'run_tests.py'), 'full'" in text
    has_std = '"check_standards.py"' in text
    hardcoded = "selftest_synthetic.py" in text and "run_tests.py" not in text
    # 三个按钮都要挂在界面上
    buttons = text.count("_run_tests_fast") + text.count("_run_tests_full") + \
        text.count("_run_standards")

    print(f"  快测按钮 -> fast: {has_fast}；全套按钮 -> full: {has_full}")
    print(f"  规范自查按钮 -> check_standards.py: {has_std}")
    print(f"  三个入口函数被引用的次数：{buttons}（每个至少 1 次定义 + 1 次绑定）")

    checks = [
        ("快测按钮走 run_tests.py fast", has_fast),
        ("全套按钮走 run_tests.py full", has_full),
        ("规范自查按钮走 check_standards.py", has_std),
        ("不再只跑 selftest_synthetic.py", not hardcoded),
        ("三个入口函数都定义且被绑定（引用次数 >= 6）", buttons >= 6),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_cli_uses_runner() -> bool:
    """第 4 节：命令行界面的菜单项走公共设施。"""
    print("----- 4. 启动程序_命令行.py -----")
    text = _read(CLI)
    has_fast = '"run_tests.py"), "fast"' in text or "'run_tests.py'), 'fast'" in text
    has_full = '"run_tests.py"), "full"' in text or "'run_tests.py'), 'full'" in text
    has_std = '"check_standards.py"' in text
    keys = re.findall(r'^\s*"([0-9a-z])":\s*action_', text, flags=re.MULTILINE)
    # ⚠️ 菜单键**只能从菜单列表那块解析**，不能扫全文。
    # 第一版用 `\("([0-9a-z])",\s*"` 扫全文，结果把
    # `input("仍要继续？(y/N) > ")` 里的 `"y"` 也当成了菜单项 —— 假阳性。
    # 这类"正则扫到了别处的相似文本"是本项目反复出现的坑（见 check_standards 的分组解析）。
    menu_block = re.search(r"items\s*=\s*\[(.*?)\]", text, flags=re.DOTALL)
    menu_items = re.findall(r'\("([0-9a-z])",\s*"', menu_block.group(1)) if menu_block else []
    # 不走 ACTIONS 字典、由主循环专门处理的键（退出 / 改参数）。
    SPECIAL = {"0", "7"}

    print(f"  ACTIONS 键：{keys}")
    print(f"  菜单项键：{menu_items}")
    missing_from_menu = [k for k in keys if k not in menu_items]
    unhandled = [k for k in menu_items if k not in keys and k not in SPECIAL]

    for k in menu_items:
        if k in SPECIAL:
            handled = bool(re.search(rf'(choice\s*==\s*"{k}"|pick\s*==\s*"{k}")', text))
            print(f"    特别键 {k}：主循环里有显式分支 = {handled}")

    checks = [
        ("快测（9）走 run_tests.py fast", has_fast),
        ("全套（f）走 run_tests.py full", has_full),
        ("规范自查（s）走 check_standards.py", has_std),
        ("ACTIONS 里的每个键都在菜单里露出来（不许有隐藏入口）",
         not missing_from_menu),
        ("菜单里的非特别键都指向存在的 ACTION", not unhandled),
        ("特别键（退出 / 改参数 / 确认）在主循环里都有显式分支",
         all(re.search(rf'(choice\s*==\s*"{k}"|pick\s*==\s*"{k}")', text)
             for k in menu_items if k in SPECIAL)),
    ]
    if missing_from_menu:
        print(f"  ACTIONS 有、菜单没有：{missing_from_menu}")
    if unhandled:
        print(f"  菜单有、ACTIONS 与特别键都没有：{unhandled}")
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_agreement_across_entrypoints() -> bool:
    """第 5 节：三个入口引用的**运行器路径字符串完全一致**（不许各写一份）。"""
    print("----- 5. 三个入口对运行器的引用是否一致 -----")
    variants = {
        "启动程序.bat": "run_tests.py",
        "启动程序.py": "run_tests.py",
        "启动程序_命令行.py": "run_tests.py",
    }
    found: dict[str, str] = {}
    for rel in variants:
        text = _read(ROOT / rel)
        m = re.search(r"""["'\\/]*(apps[\\/]+run_tests\.py)""", text)
        found[rel] = m.group(1) if m else "(没引用)"
    for rel, ref in found.items():
        print(f"  {rel:<24} -> {ref}")

    normalised = {v.replace("\\", "/") for v in found.values()}
    checks = [
        ("三个入口都引用了运行器", all(v != "(没引用)" for v in found.values())),
        ("引用的是同一个路径（只差斜杠方向）", len(normalised) == 1),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def _run_all(*args: str) -> tuple[int, str, str]:
    """跑 `apps/run_all.py`（stdout / stderr 分开，`--json` 时 stdout 必须能直接解析）。"""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.run([sys.executable, str(RUN_ALL), *args], cwd=str(ROOT),
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", env=env)
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def _fake_session(root: Path, *, with_traj: bool = True, name: str = "fake_session") -> Path:
    """造一个"只有文件名、没有内容"的假会话 —— 只验命令拼装，不真跑视频。

    空文件是够的：`plan_steps` 只判 `exists()`，真跑会由被调脚本自己拒绝。
    """
    sess = root / name
    (sess / "derived").mkdir(parents=True, exist_ok=True)
    (sess / "session.json").write_text(json.dumps({
        "schema": "sphere-stereo-session", "version": 1, "id": "fake_session",
        "left": {"role": "left", "video": "left.mkv"},
        "right": {"role": "right", "video": "right.mkv"},
    }), encoding="utf-8")
    (sess / "left.mkv").write_bytes(b"")
    (sess / "right.mkv").write_bytes(b"")
    if with_traj:
        (sess / "derived" / "traj3d.csv").write_text("matched,t,X,Y,Z\n", encoding="utf-8")
    return sess


def test_run_all_orchestrator() -> bool:
    """第 6 节：`apps/run_all.py` 只调用现有脚本、跳过要说明、退出码分得清 0/1/3。"""
    print("----- 6. run_all.py（会话 → 校验 → 三维 → 可视化） -----")
    from apps import run_all as RA          # 只 import，不许顶层执行

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sess = _fake_session(root)
        # ⚠️ 6.1 **必须自带标定 fixture**（2026-10-27）：原来直接跑 CLI 的 dry-run，
        # 而 CLI 用默认的 `config/stereo_calibration.*` 判断 check 步能不能跑。
        # 干净 checkout 里 `config/` 被 gitignore → check 步被 skipped、`cmd=[]`
        # →「三步都转到现有脚本」当场假红（实测：干净副本 4 红之一）。
        # 判据的本意是"命令拼装对不对"，与"这台机器标定过没有"无关，
        # 所以调**同一个纯函数** `plan_steps` 并显式给 calib_files。
        fake_calib = root / "fake_calib.json"
        fake_calib.write_text("{}", encoding="utf-8")
        plan = RA.plan_steps(sess, stereo_mode="angled", calib_files=[fake_calib])
        cmds = {s["name"]: " ".join(s["cmd"]) for s in plan}
        # 统一成正斜杠：`cmd` 里是 Windows 绝对路径（`D:\...\apps\tools\check_dataset.py`），
        # 而断言用的是仓库相对写法；不归一化就会因为分隔符不同而假红。
        out = "\n".join(cmds.values()).replace("\\", "/")

        # 6.0 CLI 通路仍然要走一遍：dry-run 必须退出 0 且把步骤名打出来
        #     （具体命令由上面的纯函数断言，避免依赖本机标定）
        code, cli_out, _err = _run_all("--session", str(sess), "--dry-run",
                                       "--stereo-mode", "angled")
        cli_ok = code == 0 and all(name in cli_out for name in ("check", "tri", "export"))
        print(f"  CLI dry-run：exit={code}，三个步骤名都打印={cli_ok}")

        # 6.1 三步的命令都拼出来了，且**指向现有脚本**（不是抄进来的算法）
        forwarded = {
            "check": "apps/tools/check_dataset.py",
            "tri": "apps/run_stereo_video.py",
            "export": "apps/export_visualization.py",
        }
        for name, rel in forwarded.items():
            print(f"  {name:<6} -> {rel}: {'在' if rel in out else '不在'}")
        strict = "--strict" in out
        method = "--method color" in out
        angled = out.count("--stereo-mode angled") >= 2      # tri + export 都透传
        plot = "--plot" in out
        print(f"  check 的 --strict: {strict}；tri 的 --method color: {method}")
        print(f"  --stereo-mode angled 出现 {out.count('--stereo-mode angled')} 次（tri/export 各一次）")

        # 6.2 缺标定 -> check 被 skipped，且原因点名具体文件
        plan = RA.plan_steps(sess, calib_files=[])
        check = next(s for s in plan if s["name"] == "check")
        print(f"  缺标定：check={check['status']}；原因={check['reason']}")
        no_calib = (check["status"] == "skipped"
                    and "config/stereo_calibration.json" in check["reason"]
                    and check["cmd"] == [])

        # 6.3 缺 traj3d.csv -> export 被 skipped
        sess_no_traj = _fake_session(root, with_traj=False, name="fake_no_traj")
        plan2 = RA.plan_steps(sess_no_traj)
        export = next(s for s in plan2 if s["name"] == "export")
        print(f"  缺轨迹：export={export['status']}；原因={export['reason']}")
        no_traj = (export["status"] == "skipped" and "traj3d.csv" in export["reason"]
                   and export["cmd"] == [])

        # 6.4 全 skipped -> 3（不是 0）
        empty = root / "empty"
        empty.mkdir()
        code3, out3, _e3 = _run_all("--session", str(empty))
        print(f"  全 skipped: exit={code3}（应为 3）")
        all_skip3 = code3 == 3 and "跳过" in out3

        # 6.5 某步 failed -> 1
        code1, out1, _e1 = _run_all("--session", str(root / "nope"), "--steps", "check")
        print(f"  点名要跑但跑不了: exit={code1}（应为 1）")
        failed1 = code1 == 1 and "failed" in out1

        # 6.6 --json 可 json.loads，且 ok 与退出码一致
        js3, so3, _se3 = _run_all("--session", str(empty), "--json")
        try:
            payload = json.loads(so3)
        except Exception as exc:                            # noqa: BLE001
            payload = None
            print(f"  [不达标] --json 解析失败：{exc}")
        ok_field = payload is not None and payload.get("ok") is False
        shape = (payload is not None
                 and set(payload) == {"session", "steps", "ok"}
                 and all(set(s) == {"name", "status", "reason", "seconds", "cmd"}
                         for s in payload["steps"]))
        print(f"  --json: ok={None if payload is None else payload['ok']}，"
              f"exit={js3}；字段={'齐' if shape else '不齐'}")
        json_consistent = payload is not None and (payload["ok"] == (js3 == 0)) and ok_field

        # 6.7 反漂移：不许把三步步逻辑抄进来（等价于"必须 subprocess 调用"）
        src = _read(RUN_ALL)
        no_logic = ("import cv2" not in src and "import stereo" not in src
                    and "subprocess" in src
                    and all(name in src for name in forwarded.values()))
        print(f"  正文不含算法/库 import、只 subprocess 调脚本: {no_logic}")

    checks = [
        ("CLI dry-run 退出 0 且打印三步（通路没断）", cli_ok),
        ("三步都转到现有脚本（check_dataset / run_stereo_video / export_visualization）",
         all(rel in out for rel in forwarded.values())),
        ("每步参数透传正确（--strict / --method / --stereo-mode / --plot）",
         strict and method and angled and plot),
        ("缺标定时 check=skipped，原因点名 config/stereo_calibration.json", no_calib),
        ("缺 traj3d.csv 时 export=skipped", no_traj),
        ("全 skipped 的退出码是 3（不是 0）", all_skip3),
        ("显式点名却跑不了的退出码是 1", failed1),
        ("--json 能 json.loads，字段齐全，ok 与退出码一致", bool(payload) and shape and json_consistent),
        ("不自带算法：只 subprocess 调用现有脚本", no_logic),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_every_cli_parses() -> bool:
    """每个带 argparse 的 app，`--help` 必须能正常退出。

    ## 这条守的是什么（真实事故，2026-10-26）

    改 `apps/record_stereo.py` 时把 `--fps` 的默认值换成常量 `DEFAULT_FPS`，
    却漏了它本来用 `from sphere_tracker.video_profiles import RECORD_DEFAULT`
    这一行导入（我的插入判断"看到 video_profiles 就跳过"，正好跳过）。
    结果是**模块能 import、24 个测试全绿、但 `--help` 一跑就 NameError** ——
    因为没有任何测试执行过它的 `main()`。

    "能 import" ≠ "能跑"。这条用最便宜的方式补上：真跑一次 `--help`。
    只挑**含 argparse 的**脚本（纯 GUI 的没有 `--help`，跳过）。
    """
    import re as _re
    import subprocess

    ap = list(Path(ROOT / "apps").glob("*.py"))
    targets = []
    for p in ap:
        if p.name.startswith("_"):
            continue
        src = p.read_text(encoding="utf-8", errors="replace")
        if "argparse.ArgumentParser" in src:
            targets.append(p)

    bad: list[str] = []
    skipped: list[str] = []
    for p in sorted(targets):
        try:
            r = subprocess.run([sys.executable, str(p), "--help"],
                               capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=90, cwd=str(ROOT))
        except subprocess.TimeoutExpired:
            bad.append(f"{p.name}: --help 超时（90s）")
            continue
        if r.returncode != 0:
            tail = (r.stderr or r.stdout or "").strip().splitlines()
            bad.append(f"{p.name}: exit={r.returncode} | {tail[-1] if tail else '无输出'}")
        elif "usage:" not in (r.stdout or ""):
            bad.append(f"{p.name}: exit=0 但输出里没有 usage（可能没真解析）")

    for p in ap:
        if p not in targets:
            skipped.append(p.name)

    print(f"  带 argparse 的入口 {len(targets)} 个；跳过（无 argparse）{len(skipped)} 个")
    for line in bad:
        print(f"    [不达标] {line}")
    if skipped:
        print(f"  跳过的：{', '.join(sorted(x for x in skipped if not x.startswith('_')))}")
    return not bad


# 单目两个入口共用的八个参数。名字与 `apps/_common.add_mono_pipeline_args()`
# 里登记的一致 —— 这里也**不手写第二份语义**：它只是"哪些参数算公共"的清单，
# 而这个清单本身就是被测对象（公共块少了一个参数，这里也会红）。
MONO_COMMON_FLAGS = ("--config", "--method", "--preset", "--ball-diameter",
                     "--focal", "--predict", "--horizon", "--gravity-fit")
MONO_ENTRIES = ("run_realtime.py", "run_video.py")


def _flag_literals(path: Path, candidates: set[str],
                   minimum: int = 3) -> list[tuple[int, list[str]]]:
    """源码里出现的"手抄清单"：一个列表/元组/集合字面量装了 ≥`minimum` 个候选名。

    和 `selftest_plugins._literal_name_lists` 同一个思路（R7），只是换了一批名字。
    用 AST 而不是正则：正则会把注释、docstring 里的举例也算进去，
    于是"文档里写了三个预设名"就会假红 —— 那是判据写错了，不是代码错。
    """
    import ast as _ast

    try:
        tree = _ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return []
    hits: list[tuple[int, list[str]]] = []
    for node in _ast.walk(tree):
        if not isinstance(node, (_ast.List, _ast.Tuple, _ast.Set)):
            continue
        elems = [e.value for e in node.elts
                 if isinstance(e, _ast.Constant) and isinstance(e.value, str)]
        matched = [v for v in elems if v in candidates]
        if len(matched) >= minimum:
            hits.append((getattr(node, "lineno", 0), matched))
    return hits


def test_annotated_video_atomic_write() -> bool:
    """标注视频必须「写临时文件 + 原子改名」，不能直接写正式名。

    ## 这条守的是什么（真实事故，2026-11-03）

    单目的 `annotated.mp4` 用 mp4v 编码，它的 moov atom **只在 `writer.release()`
    时落盘**。任务管理器「优雅退出」只给 3 秒，跑不完就 `taskkill /F` 强杀——
    强杀时 `release()` 没执行，留下一个「存在但打不开」的 annotated.mp4
    （`moov atom not found`）。查看器 `_resolve_video` 用 `.exists()` 判它存在、
    `cv2.VideoCapture` 却打不开，表现是「3D 有轨迹、视频黑屏/不显示」，且不报错。

    根因不是路径（`data/mono/...` 相对路径是对的），是**视频文件本身损坏**。
    修法：`run_video.py` 与 `run_stereo_video.py` 把标注视频写到 `*.tmp.mp4`，
    release 成功后 `os.replace` 原子改名成正式名。这样中断只留 .tmp 垃圾，
    正式名永远是「完整」或「不存在」，`.exists()` 的判断就重新可靠了。

    ## 断言防什么

    1. `run_video.py` 里必须有「`.tmp` + 后缀」的临时名 + `os.replace`；
    2. `run_stereo_video.py` 同理（双目也有同样的隐患，只是没被中断踩中过）。
    反向自证：删掉 `os.replace` 或把临时名改回直接写正式名 → 断言红。
    """
    from pathlib import Path as _P

    checks: list[tuple[str, bool]] = []
    for name in ("apps/run_video.py", "apps/run_stereo_video.py"):
        src = _P(name).read_text(encoding="utf-8", errors="replace")
        has_tmp = '".tmp" + ' in src or "'.tmp' + " in src
        has_replace = "os.replace(" in src
        checks.append((f"{name} 写 .tmp 临时名", has_tmp))
        checks.append((f"{name} 有 os.replace 原子改名", has_replace))

    for label, good in checks:
        print(f"  [{'通过' if good else '不达标'}] {label}")
    return all(good for _label, good in checks)


def test_stereo_session_layout() -> bool:
    """双目会话必须**按装置分区**，且两种装置的列表不能互相看到。

    ## 这条守的是什么（用户原话，2026-11-04）

    「大角度双目和平行双目的会话没有完全隔离开」。查下来是真的：43 段会话
    全在 ``data/stereo/<时间戳>`` 扁平层，隔离只靠 ``session.json`` 的
    ``stereo_mode`` 在列表里过滤 —— 而 **29/43 段老会话根本没有这个字段**，
    于是两个窗口都列得出来。更进一步：以前录制的 ``--stereo-mode`` 是
    ``auto``（按标定），在 90° 装置上用平行标定录，会话就被标成 parallel，
    元数据与装置不符，事后谁也分不出来。

    ## 断言防什么

    1. 新录的会话落 ``data/stereo/<mode>/``（分区目录就是"属于哪套装置"）；
    2. 扫描能同时认**两种布局**：新的两层 + 旧扁平层（迁移没做也不能丢会话）；
    3. 按模式过滤**只给本装置的**：未记录的不算"可能匹配"，它算"不知道"；
    4. 「改装置模式」同时改字段**并**搬目录（只改字段 = 隔离继续失效）；
    5. 「移到回收站」是软删：目录进 ``_trash``，不是 rmtree。

    反向自证：把 ``record_out_dir`` 改回返回 ``STEREO_ROOT``、或让
    ``scan_sessions`` 不再按 mode 过滤 → 断言红。
    """
    import json
    import tempfile
    from pathlib import Path as _P

    from apps import _stereo_sessions, _stereo_layout
    from apps._gui_stereo import record_out_dir, scan_sessions
    from sphere_tracker.paths import STEREO_ROOT, stereo_mode_root

    checks: list[tuple[str, bool]] = []

    def make_session(parent: _P, name: str, mode: str | None) -> _P:
        d = parent / name
        d.mkdir(parents=True, exist_ok=True)
        raw = {"schema": "sphere-stereo-session", "version": 1, "id": name,
               "created": "2026-01-01 12:00:00", "left": {"frame_count": 10}}
        if mode:
            raw["stereo_mode"] = mode
            raw["rig"] = {"stereo_mode": mode}
        (d / "session.json").write_text(json.dumps(raw), encoding="utf-8")
        return d

    # ① 写入落点：按装置分区（不是扁平根）
    checks.append(("record_out_dir('parallel') 落在 data/stereo/parallel",
                   record_out_dir("parallel").name == "parallel"
                   and record_out_dir("parallel").parent.resolve()
                   == STEREO_ROOT.resolve()))
    checks.append(("record_out_dir('angled') 落在 data/stereo/angled",
                   record_out_dir("angled").name == "angled"))
    checks.append(("record_out_dir() 兜底仍是扁平根（老调用方不炸）",
                   record_out_dir().resolve() == STEREO_ROOT.resolve()))

    with tempfile.TemporaryDirectory() as td:
        data = _P(td) / "data"
        stereo = data / "stereo"
        stereo.mkdir(parents=True)

        # ② 三种会话：扁平层的 parallel / 扁平层的 angled / 无字段的老会话
        make_session(stereo, "20260101-000001", "parallel")
        make_session(stereo, "20260101-000002", "angled")
        make_session(stereo, "20260101-000003", None)
        # 新布局：分区目录里的会话（故意不写字段，靠"摆在哪"认出来）
        make_session(stereo / "angled", "20260101-000004", None)
        # 专供"改模式"用的一段：字段与 rig 都齐全（正常录制产物的形状）。
        # 不拿 000001 改，是为了让前面那几条"按 mode 过滤"的断言不被后续改动影响。
        make_session(stereo, "20260101-000005", "parallel")
        (stereo / "parallel").mkdir(exist_ok=True)

        all_rows = scan_sessions((stereo,))
        checks.append(("两种布局都枚举得到（5 段）", len(all_rows) == 5))
        checks.append(("分区目录本身不被当成会话",
                       all(r["id"] not in ("parallel", "angled") for r in all_rows)))

        par = scan_sessions((stereo,), mode="parallel")
        ang = scan_sessions((stereo,), mode="angled")
        checks.append(("mode='parallel' 只给 parallel（2 段）",
                       sorted(r["id"] for r in par)
                       == ["20260101-000001", "20260101-000005"]))
        checks.append(("mode='angled' 给 angled + 分区里那段（2 段）",
                       sorted(r["id"] for r in ang)
                       == ["20260101-000002", "20260101-000004"]))
        checks.append(("未记录的会话不混进任何一种装置",
                       all(r["id"] != "20260101-000003" for r in par + ang)))

        # ③ 改装置模式：字段 + 目录一起动
        orig = _stereo_sessions.stereo_mode_root
        _stereo_sessions.stereo_mode_root = lambda m: stereo / m
        try:
            moved = _stereo_sessions.set_session_mode(
                stereo / "20260101-000005", "angled")
        finally:
            _stereo_sessions.stereo_mode_root = orig
        rec = json.loads((moved / "session.json").read_text(encoding="utf-8"))
        checks.append(("改模式后字段与 rig 都更新",
                       rec.get("stereo_mode") == "angled"
                       and (rec.get("rig") or {}).get("stereo_mode") == "angled"))
        checks.append(("改模式后目录搬进 angled/",
                       moved.parent.name == "angled"
                       and not (stereo / "20260101-000005").exists()))
        checks.append(("改模式留了 session.json.bak-* 备份",
                       bool(list(moved.glob("session.json.bak-*")))))

        # ④ 移到回收站（软删）
        batch = _stereo_sessions.delete_stereo_session(moved, root=stereo)
        checks.append(("移到回收站：原目录消失", not moved.exists()))
        checks.append(("移到回收站：批次里有这段会话",
                       any(p.name == "session.json"
                           for p in batch.rglob("session.json"))))
        checks.append(("回收站在 data/_trash（不是 rmtree）",
                       batch.parent == data / "_trash"))

        # ⑤ 越界保护：不许删根 / 不许删分区目录本身
        def rejects(fn) -> bool:
            try:
                fn()
            except (OSError, ValueError):
                return True
            return False

        checks.append(("拒绝删 sessions 根",
                       rejects(lambda: _stereo_sessions.delete_stereo_session(
                           stereo, root=stereo))))
        checks.append(("拒绝删分区目录本身",
                       rejects(lambda: _stereo_sessions.delete_stereo_session(
                           stereo / "parallel", root=stereo))))

        # ⑥ 布局推断：位置 vs 记录要能分开看
        checks.append(("layout_mode_of 认得出分区",
                       _stereo_layout.layout_mode_of(stereo / "angled" / "x") == "angled"))
        checks.append(("扁平层的 layout 是 None（位置不表态）",
                       _stereo_layout.layout_mode_of(
                           stereo / "20260101-000001") is None))

    for label, good in checks:
        print(f"  [{'通过' if good else '不达标'}] {label}")
    return all(good for _label, good in checks)


def test_shared_arg_blocks() -> bool:
    """单目两个入口的公共参数必须**出自同一个函数**，预设名单不许再手抄。

    ## 这条守的是什么（真实漂移，2026-10-27 实测）

    `run_realtime.py`（相机）与 `run_video.py`（离线文件）跑同一条流水线，
    八个参数却各写一份。手抄的两份**已经漂了**：

    - `--config`：实时那边写"读取 JSON 配置（标定结果）"，离线那边一个字没有；
    - `--ball-diameter` / `--focal` / `--horizon` / `--gravity-fit`：说明同样只在实时那边；
    - `--preset`：说明里手抄了一份预设名单，而那是预设表的**第三份**副本
      （另外两份在 `calibrate_color.py` 的按键表和 GUI 下拉里）。

    漂移的后果**不是崩溃**，而是"同一个参数在两个入口里解释不一样"，用户只能靠猜 ——
    这类问题没有报错、没有测试会红，只能靠人偶然发现。

    收进 `apps/_common.add_mono_pipeline_args()` 之后，本测试保证它不会再被抄开：
    调用必须在、手抄必须没有、生效必须可验。四条缺一不可 ——
    只查结构会漏"调了但没生效"，只查行为会漏"生效了但旁边又抄了一份"。
    """
    checks: list[tuple[str, bool]] = []

    # ---- 1. 调用在 / 2. 手抄没有 ----
    for name in MONO_ENTRIES:
        path = ROOT / "apps" / name
        if not path.is_file():
            checks.append((f"{name} 存在", False))
            continue
        src = path.read_text(encoding="utf-8", errors="replace")
        calls = src.count("add_mono_pipeline_args(")
        checks.append((f"{name} 调用 add_mono_pipeline_args（实际 {calls} 处，应恰好 1）",
                       calls == 1))
        dup = [flag for flag in MONO_COMMON_FLAGS
               if f'add_argument("{flag}"' in src]
        checks.append((f"{name} 不再自己定义公共参数（发现 {dup or '无'}）", not dup))

    # ---- 3. 生效可验：--help 里八个参数都在 ----
    missing: list[str] = []
    for name in MONO_ENTRIES:
        path = ROOT / "apps" / name
        if not path.is_file():
            continue
        try:
            r = subprocess.run([sys.executable, str(path), "--help"],
                               capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=90, cwd=str(ROOT))
        except subprocess.TimeoutExpired:
            missing.append(f"{name}: --help 超时")
            continue
        text = r.stdout or ""
        gone = [flag for flag in MONO_COMMON_FLAGS if flag not in text]
        if r.returncode != 0 or gone:
            missing.append(f"{name}: exit={r.returncode} 缺 {gone or '（无）'}")
    checks.append((f"两个入口的 --help 都含全部 {len(MONO_COMMON_FLAGS)} 个公共参数"
                   f"{'（' + '；'.join(missing) + '）' if missing else ''}", not missing))

    # ---- 4. 预设名单不许手抄（≥3 个预设名的字面量 = 又抄了一份） ----
    from sphere_tracker.config import ColorConfig

    presets = set(ColorConfig.preset_names())
    scanned = sorted((ROOT / "apps").glob("*.py")) \
        + sorted((ROOT / "apps" / "tools").glob("*.py")) \
        + [GUI, ROOT / "启动程序_命令行.py"]
    offenders: list[str] = []
    for path in scanned:
        if not path.is_file():
            continue
        for lineno, names in _flag_literals(path, presets):
            offenders.append(f"{path.name}:{lineno} {names}")
    checks.append((f"预设名单只此一处（{len(scanned)} 个文件扫描，手抄 {len(offenders)} 处）"
                   f"{'：' + '；'.join(offenders) if offenders else ''}", not offenders))

    for label, good in checks:
        print(f"  [{'通过' if good else '不达标'}] {label}")
    return all(good for _label, good in checks)


PROJECT_PKGS = ("sphere_tracker", "calib", "stereo", "apps")


def test_path_bootstrap_is_marker_based() -> bool:
    """为了 import 本项目包而改 `sys.path` 的文件，必须**按标记上溯**到仓库根。

    ## 为什么（2026-10-27，交接文档 §10.4(a) 挂着的那条）

    全项目 57 个文件各自写了同样 4 行引导（算仓库根 + `sys.path.insert`）。
    写法实测有四种，其中三种是**按位置数**：

    ```python
    ROOT = Path(__file__).resolve().parent        # apps/ 下能用，但插的是 apps/
    ROOT = Path(__file__).resolve().parents[1]    # 换个目录深度就错
    ROOT = Path(__file__).resolve().parents[2]    # apps/tools/ 下才对
    ```

    按位置数的写法**只在文件不动的时候正确**，而且错了**不会当场报错** ——
    它指向一个存在的目录（只是不是仓库根），于是后面的 `import sphere_tracker`
    报 `ModuleNotFoundError`，离真正原因（数错了几层）很远。
    按标记上溯（找含 `sphere_tracker/` 的祖先）与深度无关，也没有这个坑。

    ## 判据为什么定在"语义"而不是"逐字相同"

    两种上溯写法 —— `while not (ROOT / "sphere_tracker").is_dir(): ROOT = ROOT.parent`
    与 `next(p for p in parents if (p / "sphere_tracker").is_dir())` —— **都合规**，
    它们做的是同一件事。要求逐字相同只会制造 57 个文件的无意义 diff，
    而真实风险来自"按位置数"，不是来自"两种等价写法并存"。

    ## 为什么不误伤"插自己目录"的脚本

    `apps/tools/preview_pptx.py` 插的是**自己所在目录**，为的是 import 兄弟模块
    `_preview_lib` —— 那是另一件事，与"找仓库根"无关。所以判据先问
    "这个文件是否在模块层 import 了本项目包"，不 import 的直接不适用。
    """
    import ast as _ast

    targets: list[Path] = []
    for pattern in ("*.py", "apps/*.py", "apps/tools/*.py", "tests/*.py",
                    "stereo/*.py", "calib/*.py"):
        targets += sorted(ROOT.glob(pattern))

    violations: list[str] = []
    applicable = 0
    for path in targets:
        try:
            src = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "sys.path.insert" not in src:
            continue
        try:
            tree = _ast.parse(src)
        except SyntaxError as exc:
            violations.append(f"{path.relative_to(ROOT).as_posix()}: 语法错误 {exc}")
            continue

        imports_project = False
        insert_line: int | None = None
        for node in _ast.walk(tree):
            if isinstance(node, _ast.Import):
                if any(a.name.split(".")[0] in PROJECT_PKGS for a in node.names):
                    imports_project = True
            elif isinstance(node, _ast.ImportFrom):
                if (node.module or "").split(".")[0] in PROJECT_PKGS:
                    imports_project = True
            elif isinstance(node, _ast.Call) and isinstance(node.func, _ast.Attribute):
                inner = node.func.value
                if (isinstance(inner, _ast.Attribute) and inner.attr == "path"
                        and getattr(inner.value, "id", "") == "sys"
                        and node.func.attr == "insert" and insert_line is None):
                    insert_line = node.lineno
        if not imports_project or insert_line is None:
            continue

        applicable += 1
        head = "\n".join(
            line for line in src.splitlines()[:insert_line]
            if not line.lstrip().startswith("#"))       # 注释里的示例不算
        if "is_dir()" not in head and "is_file()" not in head:
            violations.append(
                f"{path.relative_to(ROOT).as_posix()}:{insert_line} "
                f"改 sys.path 前没有按标记上溯（疑似 parent / parents[N]）")

    print(f"  需要引导的项目包使用者 {applicable} 个；按位置数（不合规）{len(violations)} 个")
    for line in violations:
        print(f"    [不达标] {line}")
    print(f"  [{'通过' if not violations else '不达标'}] 全部按标记上溯到含 sphere_tracker/ 的祖先")
    return not violations


#: 裸拼 `data/sessions` 的三种形态：
#: ① 整串成一个字面量 —— `"data/sessions"` / `'data\\sessions'`
#: ② 两段各自成串再拼 —— `ROOT / "data" / "sessions"` / `Path("data", "sessions")`
#: ③ f-string 拼出来的真路径 —— `f"data/sessions/{sid}"`（必须是 f 前缀 + 开头就命中，
#:    避免把 `help="data/sessions/<id> 目录或 session.json"` 这类**说明文字**也算进来）
#: `[/\\,]` 同时接受 `/`、`\` 与逗号（`os.path.join` / `Path(...)` 的写法），
#: `\s*` 允许跨行 —— 8 处旧代码里就有换行写的。
_BARE_LAYOUT = (
    re.compile(r"""['"]\s*data\s*[/\\]+\s*sessions\s*['"]""", re.IGNORECASE),
    re.compile(r"""['"]\s*data\s*['"]\s*[/\\,]\s*['"]\s*sessions\s*['"]""",
               re.IGNORECASE),
    re.compile(r"""(?<![A-Za-z0-9_])(?:rf|fr|f)['"]\s*data\s*[/\\]+\s*sessions""",
               re.IGNORECASE),
)


def _code_only(src: str) -> str:
    """去掉注释但**保留字符串**（路径是字符串，字符串才是要找的东西）。

    不按 `#` 截断：本项目里 `"#0f4c81"` 这类颜色常量到处都是，按 `#` 截会把代码
    截成半句，正则反而永远匹配不到真正的问题。
    """
    import io
    import tokenize as _tokenize

    out: list[str] = []
    try:
        for tok in _tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == _tokenize.COMMENT:
                continue
            out.append(tok.string)
    except (_tokenize.TokenError, IndentationError, SyntaxError):
        return src            # 解析不了就退回原文：宁可多报，不要漏报
    return " ".join(out)


def test_data_layout_single_source() -> bool:
    """`data/` 的目录名只在一处定义：`sphere_tracker/paths.py`。

    ## 为什么它是主判据

    `data/sessions` 这个字面量曾经**散在 8 处**（`data_manager` / `record_stereo` /
    `check_dataset` / `view_3d` / `_gui_app` / `_gui_bench` / `_gui_stereo` / 文档）。
    而 `data/sessions` 里只有**双目**会话、单目另起 `data/mono` —— 于是"会话"在磁盘上
    指两个不同东西（2026-10-29 用户原话："data 下面 session 和 mono 命名有歧义"）。
    只要还允许各处自己拼路径，改布局就必然漏几处，漏掉的几处**不会报错** ——
    只会静默读写一个空目录。这正是本项目反复吃过的形状。

    ## 三条断言分别防什么

    1. **不许裸拼接**：递归扫 `apps/` 与 `stereo/`（`apps/tools/` 也在内）。
       这不是代码风格，是"还有没有第二份真相"。
    2. **顺序即优先级**：`stereo_roots()` 第一项必须是新的 `STEREO_ROOT`。
       顺序反了不会崩，只会在"新旧位置各有一段同名会话"时**选中旧的那段**。
    3. **兼容真的有效**：在 `STEREO_ROOT` 与 `LEGACY_STEREO_ROOT` 各造一段临时会话，
       断言默认扫描（不传参数）**两条都列得出来**。只在注释里写"兼容旧位置"不算数。

    ## 反向自证（改了就会红）

    往 `apps/run_all.py` 里塞一行 `X = ROOT / "data" / "sessions"` → 断言 1 报出
    该文件并返回 False（实测输出见交接报告）。
    """
    import shutil

    from sphere_tracker.paths import LEGACY_STEREO_ROOT, STEREO_ROOT, stereo_roots

    print("----- 数据目录只有一个来源 -----")
    checks: list[tuple[str, bool]] = []

    # ---- 1. 不许有裸拼接 ----
    targets = sorted((ROOT / "apps").rglob("*.py")) + \
        sorted((ROOT / "stereo").rglob("*.py"))
    offenders: list[str] = []
    for path in targets:
        try:
            src = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for pat in _BARE_LAYOUT:
            for m in pat.finditer(_code_only(src)):
                # token 化后行号会错位，报"文件 + 命中文本"足够定位（单个文件都短）
                offenders.append(
                    f"{path.relative_to(ROOT).as_posix()} 命中 {m.group(0)!r}")
    for line in offenders:
        print(f"    [不达标] {line}")
    checks.append((f"{len(targets)} 个文件里没有裸拼 data/sessions"
                   f"（发现 {len(offenders)} 处）", not offenders))

    # ---- 2. stereo_roots() 新的在前 ----
    roots = stereo_roots()
    order_ok = bool(roots) and roots[0] == STEREO_ROOT \
        and STEREO_ROOT in roots and LEGACY_STEREO_ROOT in roots
    print(f"  stereo_roots() = {[r.relative_to(ROOT).as_posix() for r in roots]}")
    checks.append(("stereo_roots() 第一项是新的 data/stereo（新旧都在元组里）", order_ok))

    # ---- 3. 新旧两个根都能被扫到（造真目录，跑完删） ----
    # 名字用 1990 开头：按目录名倒序排永远在最后，不会顶掉界面上的"最近会话"。
    probes = {"new": STEREO_ROOT / "19900101-000001-layout-probe",
              "old": LEGACY_STEREO_ROOT / "19900101-000002-layout-probe"}
    found_default: list[str] = []
    found_per_root: list[str] = []
    err = ""
    try:
        for p in probes.values():
            p.mkdir(parents=True, exist_ok=True)
            (p / "session.json").write_text(
                '{"label": "布局探针", "left": {"frame_count": 1}}', encoding="utf-8")

        from apps._gui_stereo import scan_sessions as gui_scan
        names = {p.name for p in probes.values()}
        found_default = [r["id"] for r in gui_scan() if r["id"] in names]

        # 库层也验一遍：单根 list_sessions 在新/旧根上各能列出自己那一段
        from stereo.sessions import list_sessions
        for key, root in (("new", STEREO_ROOT), ("old", LEGACY_STEREO_ROOT)):
            if probes[key] in list_sessions(root):
                found_per_root.append(key)
    except Exception as exc:                            # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
    finally:
        for p in probes.values():
            shutil.rmtree(p, ignore_errors=True)

    print(f"  默认扫描 scan_sessions() 列出：{sorted(found_default)}"
          f"{'  ' + err if err else ''}")
    checks.append(("默认扫描同时列出新位置与旧位置的会话（兼容真的有效，不是注释）",
                   len(found_default) == 2))
    checks.append((f"库层 list_sessions 在两个根上各能列出（{sorted(found_per_root)}）",
                   len(found_per_root) == 2))

    for label, good in checks:
        print(f"  [{'通过' if good else '不达标'}] {label}")
    return all(good for _label, good in checks)


# ----------------------------------------------------------------------
# 第 11 节：YOLO 数据集导出
# ----------------------------------------------------------------------
TOOL_EXPORT = ROOT / "apps" / "tools" / "export_ball_dataset.py"

#: 夹具里的视频尺寸。取 64×48 而不是正方形，是**故意的**：正方形画面下
#: "宽度除以 W、高度除以 H" 与 "都除以 max(W,H)" 碰巧等价，测不出区别。
FAKE_W, FAKE_H = 64, 48
FAKE_FRAMES = 8


def _tiny_video(path: Path, frames: int = FAKE_FRAMES) -> None:
    """造一段**真能解码**的微型视频（MJPG 内容的 AVI，文件名照生产布局叫 .mkv）。

    为什么不真解码录制会话：真数据 300 帧 1920×1080，一条测试几十秒，
    而这里要验的是"按帧号取图 + 归一化"，跟容器格式无关。
    `cv2.VideoCapture` 只嗅内容、不认扩展名（实测），所以 AVI 存成 .mkv 一样能开 ——
    这样"会话内文件叫 clip.mkv / left.mkv"这条生产约定也不用为测试让步。
    """
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10,
                             (FAKE_W, FAKE_H))
    if not writer.isOpened():
        raise RuntimeError("本机 cv2 写不出 MJPG，无法构造微型视频")
    try:
        for i in range(frames):
            # 纯色帧：标注只依赖 CSV，画面内容不参与任何断言
            writer.write(np.full((FAKE_H, FAKE_W, 3), (i * 30) % 255, np.uint8))
    finally:
        writer.release()


def _fake_calib(path: Path, focal: float = 100.0) -> None:
    """最小可用标定：只给导出要用的 `P1[0,0]`（焦距）与 `image_size`。

    故意把焦距设成 100 这种整数值：`r = f·D/(2Z)` 就能口算，
    期望标签可以写在断言里而不是"跑一遍抄下来"。
    """
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    eye = np.eye(3)
    p1 = eye.copy()
    p1[0, 0] = p1[1, 1] = focal
    p1[0, 2], p1[1, 2] = FAKE_W / 2, FAKE_H / 2
    np.savez(path, K1=eye, D1=np.zeros(5), K2=eye, D2=np.zeros(5),
             R=eye, T=np.array([-0.06, 0.0, 0.0]),
             image_size=np.array([FAKE_W, FAKE_H]), P1=p1, P2=p1)


#: 四段假会话：两段单目 + 两段双目。
#: 为什么是 2+2 而不是 1+1：切分按**类型**分层后，"每类至少 1 段进 val"要求每类 >=2 段，
#: 1+1 的夹具连"正常导出"都做不了（工具会正确地拒绝）。2+2 才能验分层本身。
MONO_IDS = ("20990101-000001", "20990102-000001")
STEREO_IDS = ("20990103-000001", "20990104-000001")


def _make_mono_session(data: Path, sid: str) -> None:
    sess = data / "mono" / sid
    _tiny_video(sess / "clip.mkv")
    (sess / "derived").mkdir(parents=True, exist_ok=True)
    (sess / "derived" / "seed.json").write_text(json.dumps({
        "x": 30.0, "y": 20.0, "radius": 10.0, "frame": 0,
        "image_size": [FAKE_W, FAKE_H], "created": "2099-01-01 00:00:00",
    }), encoding="utf-8")
    # **带 BOM 写**：run_video.py 写出来的 traj.csv 就是 utf-8-sig，
    # 工具的读法必须扛住（否则第一列名变成 \ufeffframe，整段会话"没跟踪结果"）
    (sess / "derived" / "traj.csv").write_text(
        "frame,t,detected,x,y,radius_px,radius_refined_px,gate\n"
        "0,0.0,1,30.0,20.0,9.0,10.0,ok\n"
        "1,0.03,1,31.0,21.0,9.0,10.0,ok\n"
        "2,0.06,1,32.0,22.0,9.0,10.0,ok\n"
        "3,0.09,0,,,,,no_detection\n"
        "4,0.12,0,,,,,no_detection\n"
        "5,0.15,0,,,,,no_detection\n",
        encoding="utf-8-sig")


def _make_stereo_session(data: Path, sid: str, *, legacy_rows: bool = False) -> None:
    """造一段双目会话。

    `legacy_rows=True` 时**照抄 2026-10-30 之前的坏形状**（未匹配行只有 14 格，
    漏掉 confident），用来保证"修复前录的老文件仍然导得对"。一半夹具写新形状、
    一半写老形状，两条读取路径都被踩到；只写一种的话，另一种就是没人管的死角。
    """
    sess = data / "stereo" / sid
    _tiny_video(sess / "left.mkv")
    _fake_calib(sess / "calib" / "stereo_calibration.npz")
    (sess / "derived").mkdir(parents=True, exist_ok=True)
    (sess / "derived" / "seeds.json").write_text(json.dumps({
        "schema": "sphere-stereo-seeds", "version": 1, "session": sid,
        "image_size": [FAKE_W, FAKE_H], "seed_frame": 0, "source": "gui_click",
        "left": {"x": 30.0, "y": 20.0, "radius": 8.0},
        "right": {"x": 10.0, "y": 20.0, "radius": 8.0},
    }), encoding="utf-8")

    order = ("frame", "t", "matched", "X", "Y", "Z", "u_left", "v_left",
             "u_right", "v_right", "epipolar_px", "residual_px", "confident",
             "method_left", "method_right")

    def row(frame: int, matched: int, values: dict) -> str:
        cells = {"frame": frame, "t": f"{frame * 0.03:.2f}", "matched": matched}
        cells.update(values)
        if not matched:
            if legacy_rows:
                # 老文件的坏形状：confident 那一格缺失，method 两列顶上来
                return ",".join(str(cells.get(k, "")) for k in order[:12]) + ",track,track"
            return ",".join(str(cells.get(k, "")) for k in order)   # confident 位是 0
        return ",".join(str(cells.get(k, "")) for k in order)

    # Z=0.4 m、f=100 → r = 100*0.025/(2*0.4) = 3.125 px（口算可验）
    lines = [
        row(i, 1, {"X": 0.01, "Y": 0.02, "Z": 0.4,
                   "u_left": 30.0 + i, "v_left": 20.0 + i,
                   "u_right": 10.0 + i, "v_right": 20.0 + i,
                   "epipolar_px": 0.5, "residual_px": 0.3,
                   "confident": 1, "method_left": "track", "method_right": "track"})
        for i in range(3)
    ] + [row(i, 0, {"confident": 0}) for i in (3, 4, 5)]
    (sess / "derived" / "traj3d.csv").write_text(
        ",".join(order) + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


def _make_export_fixture(root: Path, mono_ids: tuple[str, ...] = MONO_IDS,
                         stereo_ids: tuple[str, ...] = STEREO_IDS) -> Path:
    """造假会话，布局照生产：`<root>/mono|stereo/<id>/`。

    每段都放 frame 0/1/2 三个正样本 + 3/4/5 三个负样本候选 —— **同一个帧号在各段里都有**，
    这样才能验"同帧号必须落到不同 split"（按帧切就会当场露馅）。

    第二段双目写**修复前的老 CSV 形状**，第二段单目留作"只有 1 段"那条断言的素材来源。
    """
    data = root / "data"
    for sid in mono_ids:
        _make_mono_session(data, sid)
    for i, sid in enumerate(stereo_ids):
        _make_stereo_session(data, sid, legacy_rows=(i == 1))
    return data


def _run_export(*args: str) -> tuple[int, str, str]:
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.run([sys.executable, str(TOOL_EXPORT), *args], cwd=str(ROOT),
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", env=env, timeout=180)
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def _stems(root: Path) -> set[str]:
    """`images/train` 下每个文件所属的会话前缀（文件名形如 `<id>_000123.jpg`）。"""
    if not root.is_dir():
        return set()
    return {p.stem.rsplit("_", 1)[0] for p in root.glob("*.jpg")}


def _leaks(out: Path) -> list[str]:
    """同一段会话同时出现在 train 与 val —— 只要非空就是数据泄漏。"""
    return sorted(_stems(out / "images" / "train") & _stems(out / "images" / "val"))


def test_export_ball_dataset() -> bool:
    """第 11 节：`apps/tools/export_ball_dataset.py` 的标注口径与按会话切分。

    这个工具不产生任何物理结论，只做"跟踪结果 → YOLO 格式"。但它有**两个**能悄悄
    毁掉训练、而且不会报错的地方，所以每一处都用可执行断言钉住：

    1. **归一化口径**。YOLO 的 `cx/cy/w/h` 是**逐轴**归一化的（`w = 像素宽 / W`）。
       球在像素里必须是正方框，所以 `w·W` 与 `h·H` 必须相等。
       若把宽高都除以 `max(W,H)`，非正方形画面（本项目 1718×972）上的框会变椭圆；
       而这件事**训练不会崩**，只会让框贴着球的一半或包住半个画面。夹具因此刻意用
       64×48：正方形画面下两种写法恰好等价，测不出来。
    2. **按会话切分**。同一段里相邻帧差 33 ms，按帧随机切 = 让 val 见过 train 的邻帧，
       mAP 会好看得离谱，而这件事**不会报错**。见本节最后的反向自证。

    其余断言（负样本是空文件、`--max-width` 不改标签、1 段必须拒绝、dry-run 不落盘）
    都是"少一个就会静默错"的契约条款。
    """
    print("----- 11. YOLO 数据集导出（标注来自跟踪结果） -----")
    checks: list[tuple[str, bool]] = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        data = _make_export_fixture(tmp_root)
        out = tmp_root / "out"

        # ---- 11.1 正常导出（4 段会话：2 单目 + 2 双目，每类 1 段进 val） ----
        code, stdout, stderr = _run_export(
            "--all", "--data-root", str(data), "--out", str(out),
            "--val-sessions", "2", "--negatives", "--neg-every", "2")
        print(f"  导出：exit={code}；--val-sessions 2；--negatives --neg-every 2")
        if code != 0:
            print(f"    stderr: {stderr.strip().splitlines()[-1:] or '无'}")
        report_path = out / "export_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
        print(f"  报告 keys：{sorted(report)[:6]} ...（共 {len(report)} 项）")

        checks.append(("导出退出码 0", code == 0))
        checks.append(("dataset.yaml 与 export_report.json 都生成了",
                       (out / "dataset.yaml").is_file() and report_path.is_file()))

        # ---- 11.2 数量对得上 ----
        # 每段 3 个正样本 + 负样本 3 帧按 every=2 采样得 2 帧（3、5）→ 5 张；
        # 4 段 = 2 段 train（10）+ 2 段 val（10）
        counts = {}
        for split in ("train", "val"):
            counts[split] = (
                len(list((out / "images" / split).glob("*.jpg"))),
                len(list((out / "labels" / split).glob("*.txt"))),
            )
        print(f"  images/labels：train={counts['train']} val={counts['val']}（期望 (10, 10)）")
        checks.append(("train/val 的图片与标签数量都是 10 且一一对应",
                       counts["train"] == (10, 10) and counts["val"] == (10, 10)))

        # ---- 11.3 负样本是空文件（YOLO 的"无目标"写法，不是缺文件） ----
        empty = {}
        for split in ("train", "val"):
            empty[split] = sum(1 for p in (out / "labels" / split).glob("*.txt")
                               if not p.read_text(encoding="utf-8").strip())
        print(f"  空 label（负样本）：train={empty['train']} val={empty['val']}（期望各 4）")
        checks.append(("负样本落地为**空文件**且数量=4/4",
                       empty["train"] == 4 and empty["val"] == 4))

        # ---- 11.4 标签格式：5 个字段、归一化到 0~1、像素里是正方框 ----
        bad_fields: list[str] = []
        not_square: list[str] = []
        out_of_range: list[str] = []
        for split in ("train", "val"):
            for p in sorted((out / "labels" / split).glob("*.txt")):
                text = p.read_text(encoding="utf-8").strip()
                if not text:
                    continue
                parts = text.split()
                if len(parts) != 5 or parts[0] != "0":
                    bad_fields.append(f"{p.name}:{text!r}")
                    continue
                vals = [float(v) for v in parts[1:]]
                if not all(0.0 <= v <= 1.0 for v in vals):
                    out_of_range.append(f"{p.name}:{text!r}")
                # 还原成像素：宽乘 W、高乘 H，球必须是正方框
                w_px = vals[2] * FAKE_W
                h_px = vals[3] * FAKE_H
                if abs(w_px - h_px) > 1e-3:
                    not_square.append(f"{p.name}: w_px={w_px:.4f} h_px={h_px:.4f}")
        print(f"  字段数不是 5 / 类号不是 0：{bad_fields or '无'}")
        print(f"  值越界：{out_of_range or '无'}；像素边长不相等：{not_square or '无'}")
        checks.append(("每个标签都是 `0 cx cy w h` 五段", not bad_fields))
        checks.append(("四个坐标都在 0~1（真归一化了）", not out_of_range))
        checks.append(("`w·W == h·H`：球在**像素**里是正方框（不是都除以 max(W,H)）",
                       not not_square))

        # 双目那段的半径来自几何：r = f·D/(2Z) = 100·0.025/(2·0.4) = 3.125 px
        # 框边长 = 2·r·1.15 = 7.1875 px → w = 7.1875/64 = 0.112305
        stereo_lbl = sorted((out / "labels" / "val").glob(f"{STEREO_IDS[-1]}_*.txt"))
        got_w = 0.0
        for p in stereo_lbl:
            text = p.read_text(encoding="utf-8").strip()
            if text:
                got_w = float(text.split()[3])
                break
        expect_w = 2 * (100.0 * 0.025 / (2 * 0.4)) * 1.15 / FAKE_W
        print(f"  双目 w：实际 {got_w:.6f}，几何+1.15 期望 {expect_w:.6f}")
        checks.append(("双目半径走的是几何式（f·D/2Z）再乘 box_inflate",
                       abs(got_w - expect_w) < 1e-5))

        # 单目没有 Z，半径只能取 CSV 的 radius_refined_px（=10.0，**不是** radius_px 的 9.0）：
        # 框边长 = 2·10·1.15 = 23 px → w = 23/64 = 0.359375
        mono_lbl = sorted((out / "labels" / "val").glob(f"{MONO_IDS[-1]}_*.txt"))
        got_mono = 0.0
        for p in mono_lbl:
            text = p.read_text(encoding="utf-8").strip()
            if text:
                got_mono = float(text.split()[3])
                break
        expect_mono = 2 * 10.0 * 1.15 / FAKE_W
        print(f"  单目 w：实际 {got_mono:.6f}，refined+1.15 期望 {expect_mono:.6f}")
        checks.append(("单目半径取 radius_refined_px（不是平滑后的 radius_px）",
                       abs(got_mono - expect_mono) < 1e-5))

        # ---- 11.5 dataset.yaml 三件套 ----
        yaml_text = (out / "dataset.yaml").read_text(encoding="utf-8") \
            if (out / "dataset.yaml").is_file() else ""
        checks.append(("dataset.yaml 有 path/train/val/names 且类名是 ball",
                       all(k in yaml_text for k in ("path:", "train: images/train",
                                                    "val: images/val", "0: ball"))))

        # ---- 11.6 报告里该有的字段一个不少 ----
        need = ("created", "sessions", "images", "boxes", "box_inflate", "max_width",
                "box_side_px", "skipped", "val_sessions")
        missing = [k for k in need if k not in report]
        side = report.get("box_side_px") or {}
        print(f"  报告缺字段：{missing or '无'}；box_side_px={side}")
        checks.append((f"export_report.json 含 {len(need)} 个必备字段", not missing))
        checks.append(("box_side_px 有 中位/最小/最大",
                       all(k in side for k in ("median", "min", "max"))))
        splits_in_report = sorted(s["split"] for s in report.get("sessions", []))
        print(f"  报告里的 split：{splits_in_report}；kind_counts={report.get('kind_counts')}")
        checks.append(("报告里的 split 是 2 train / 2 val",
                       splits_in_report == ["train", "train", "val", "val"]))
        # 分层：每类各 1 段进 val —— 这是 team-lead 2026-10-30 要求的规则
        kind_counts = report.get("kind_counts") or {}
        print(f"  切分说明：{report.get('split_notes')}")
        checks.append(("按类型分层：mono 与 stereo 各 1 段进 val",
                       kind_counts.get("mono") == {"train": 1, "val": 1}
                       and kind_counts.get("stereo") == {"train": 1, "val": 1}))

        # ---- 11.7 跨会话不泄漏：同帧号必须分居两侧 ----
        leaks = _leaks(out)
        # 同一个帧号 000001 在两段会话里都存在，若按帧切它们会同时落进两个 split
        f1 = {split: sorted(p.name for p in (out / "images" / split).glob("*_000001.jpg"))
              for split in ("train", "val")}
        print(f"  同时出现在两边的会话：{leaks or '无'}")
        print(f"  frame 000001 的归属：train={f1['train']} val={f1['val']}")
        checks.append(("没有会话同时出现在 train 与 val", not leaks))
        checks.append(("同一个帧号在两段会话里 → 分居 train / val",
                       bool(f1["train"]) and bool(f1["val"])
                       and f1["train"] != f1["val"]))
        checks.append(("val 里正好是每类 id 最大的那一段",
                       _stems(out / "images" / "val") == {MONO_IDS[-1], STEREO_IDS[-1]}))

        # ---- 11.8 --max-width 只改像素，不改标签 ----
        out_small = tmp_root / "out_small"
        code_s, _o, _e = _run_export(
            "--all", "--data-root", str(data), "--out", str(out_small),
            "--val-sessions", "2", "--negatives", "--neg-every", "2",
            "--max-width", "32")
        import cv2
        img = cv2.imread(str(next((out_small / "images" / "train").glob("*.jpg"))))
        same_labels = True
        for split in ("train", "val"):
            for p in (out / "labels" / split).glob("*.txt"):
                q = out_small / "labels" / split / p.name
                if not q.is_file() or q.read_text(encoding="utf-8") != p.read_text(encoding="utf-8"):
                    same_labels = False
        print(f"  --max-width 32：exit={code_s}；存出的图 {img.shape[1]}x{img.shape[0]}；"
              f"标签逐字相同={same_labels}")
        checks.append(("--max-width 真的缩了图（64 → 32）",
                       code_s == 0 and img is not None and img.shape[1] == 32))
        checks.append(("缩小后归一化标签**一字不改**", same_labels))

        # ---- 11.9 只有 1 段：必须拒绝（1 段不能既训又验） ----
        only_mono = tmp_root / "only_mono" / "data"
        _make_export_fixture(tmp_root / "only_mono",
                             mono_ids=(MONO_IDS[0],), stereo_ids=())
        code1, out1, _err1 = _run_export(
            "--all", "--data-root", str(only_mono), "--out", str(tmp_root / "out1"))
        print(f"  只有 1 段：exit={code1}；提到'不能既训又验'={'不能既训又验' in out1}")
        checks.append(("只有 1 段会话时拒绝导出（退出码 2）", code1 == 2))
        checks.append(("拒绝时说明原因（含'不能既训又验'）", "不能既训又验" in out1))
        checks.append(("拒绝时不落任何产物", not (tmp_root / "out1").exists()))

        # ---- 11.9b 每类各 1 段：分层后无 val 可用 -> 也必须拒绝 ----
        # 这是 2026-10-30 那条"单目全落进 val"的镜像问题：一旦要求分层，
        # "1 单目 + 1 双目"就不能既训又验了（哪一类都没法两边各留一段）。
        one_each = tmp_root / "one_each"
        _make_export_fixture(one_each, mono_ids=(MONO_IDS[0],),
                             stereo_ids=(STEREO_IDS[0],))
        code_e, out_e, _e = _run_export(
            "--all", "--data-root", str(one_each / "data"),
            "--out", str(tmp_root / "oute"), "--val-sessions", "1")
        print(f"  每类各 1 段：exit={code_e}；提到'凑够 2 段'={'凑够 2 段' in out_e}")
        checks.append(("每类各 1 段时拒绝（分层后没有可用的 val 段）", code_e == 2))
        checks.append(("拒绝时指明'没有类型凑够 2 段'", "凑够 2 段" in out_e))

        # ---- 11.9c 单目只有 1 段 + 双目 2 段：单目整段进 train（team-lead 要的规则） ----
        one_mono = tmp_root / "one_mono"
        _make_export_fixture(one_mono, mono_ids=(MONO_IDS[0],), stereo_ids=STEREO_IDS)
        out_strat = tmp_root / "out_strat"
        code_t, out_t, _e = _run_export(
            "--all", "--data-root", str(one_mono / "data"), "--out", str(out_strat),
            "--val-sessions", "1")
        train_kinds = _stems(out_strat / "images" / "train")
        val_stems = _stems(out_strat / "images" / "val")
        note_ok = "mono 只有 1 段" in out_t and "未被验证" in out_t
        print(f"  单目 1 段 + 双目 2 段：exit={code_t}；val={sorted(val_stems)}；"
              f"train 含单目={MONO_IDS[0] in train_kinds}；提示={note_ok}")
        checks.append(("单目不足 2 段 -> 整段进 train，val 只含双目",
                       code_t == 0 and val_stems == {STEREO_IDS[-1]}
                       and MONO_IDS[0] in train_kinds))
        checks.append(("并在输出里写明'该类型未被验证'（不是静默处理）", note_ok))

        # ---- 11.9d --no-stratify-kind 退回旧规则（按 id 取最后 K 段） ----
        out_flat = tmp_root / "out_flat"
        code_f, out_f, _e = _run_export(
            "--all", "--data-root", str(data), "--out", str(out_flat),
            "--val-sessions", "1", "--no-stratify-kind")
        print(f"  --no-stratify-kind：exit={code_f}；val={sorted(_stems(out_flat / 'images' / 'val'))}")
        checks.append(("--no-stratify-kind 退回'最后 K 段'（val 是 id 最大的那一段）",
                       code_f == 0
                       and _stems(out_flat / "images" / "val") == {STEREO_IDS[-1]}))

        # ---- 11.10 --val-sessions 越界：报错而不是截断 ----
        # 分层后上限是 sum(每类段数-1)：2+2 段 -> K <= 2
        code_k, out_k, _e = _run_export(
            "--all", "--data-root", str(data), "--out", str(tmp_root / "outk"),
            "--val-sessions", "3")
        print(f"  --val-sessions 3（4 段、每类 2 段，上限 2）：exit={code_k}")
        checks.append(("K 超过分层上限时退出码 2", code_k == 2))

        # ---- 11.11 --dry-run 不落盘 ----
        out_dry = tmp_root / "out_dry"
        code_d, out_d, _e = _run_export(
            "--all", "--data-root", str(data), "--out", str(out_dry),
            "--val-sessions", "2", "--dry-run")
        print(f"  --dry-run：exit={code_d}；目录存在={out_dry.exists()}；"
              f"打印了统计={'框边长' in out_d}")
        checks.append(("--dry-run 退出 0 且**一个文件都没写**",
                       code_d == 0 and not out_dry.exists()))
        checks.append(("--dry-run 仍然打印统计（不是静默空跑）", "框边长" in out_d))

        # ---- 11.12 重跑同一个 --out：上次多出来的帧必须被清掉 ----
        # 不清的后果是**静默泄漏**：换 --val-sessions 后旧文件还在旧 split 里，
        # 同一帧同时进 train 与 val，训练照跑、mAP 虚高、没人发现。
        out_reuse = tmp_root / "out_reuse"
        _run_export("--all", "--data-root", str(data), "--out", str(out_reuse),
                    "--val-sessions", "2", "--negatives", "--neg-every", "2")
        code_r, _o_r, _e_r = _run_export(
            "--all", "--data-root", str(data), "--out", str(out_reuse),
            "--val-sessions", "2")                      # 这次不要负样本 -> 上次那些成了残留
        reuse = {split: (len(list((out_reuse / "images" / split).glob("*.jpg"))),
                         len(list((out_reuse / "labels" / split).glob("*.txt"))))
                 for split in ("train", "val")}
        stale_empty = sum(1 for p in (out_reuse / "labels").rglob("*.txt")
                          if not p.read_text(encoding="utf-8").strip())
        print(f"  重跑同一目录（去掉负样本）：exit={code_r}；train={reuse['train']} "
              f"val={reuse['val']}（期望各 (6, 6)）；残留空标签={stale_empty}")
        checks.append(("重跑同一目录会清掉上次的残留（数量回到 6/6，空标签 0）",
                       code_r == 0 and reuse["train"] == (6, 6)
                       and reuse["val"] == (6, 6) and stale_empty == 0))

        # ---- 11.13 不给 --session/--all：列清单 + 退出码 2，不静默挑一段 ----
        code_n, out_n, _e = _run_export("--data-root", str(data),
                                        "--out", str(tmp_root / "outn"))
        listed = all(sid in out_n for sid in MONO_IDS + STEREO_IDS)
        print(f"  不给选择：exit={code_n}；列出 4 段={listed}")
        checks.append(("不给 --session/--all 时列出可用会话并退出 2",
                       code_n == 2 and listed))

        # ==================================================================
        # 11.14 反向自证：把"按会话切"换成"按帧随机切"，泄漏检查必须红
        # ==================================================================
        # 直接改文件会把仓库搞脏，所以 import 进来**改函数对象**再跑同一个 main()。
        # `frame_split` 是"这一帧进哪个 split"的唯一出口；改成按帧号散列后就等于
        # "按帧随机切"，同段会话必然同时出现在 images/train 与 images/val。
        import contextlib
        import importlib.util
        import io

        spec = importlib.util.spec_from_file_location("_export_probe", TOOL_EXPORT)
        mod = importlib.util.module_from_spec(spec)
        # 必须先登记到 sys.modules：`@dataclass` 会回查 `sys.modules[cls.__module__]`，
        # 没登记就 `NoneType has no __dict__`（第一次跑就是这么炸的）
        sys.modules["_export_probe"] = mod
        spec.loader.exec_module(mod)
        original = mod.frame_split

        def _per_frame(_session, frame, _splits):
            return "val" if (frame.index * 7) % 3 == 0 else "train"

        out_bad = tmp_root / "out_bad"
        mod.frame_split = _per_frame
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                mod.main(["--all", "--data-root", str(data), "--out", str(out_bad),
                          "--val-sessions", "2"])
            bad_leaks = _leaks(out_bad)
        finally:
            mod.frame_split = original        # 无论成败都要还原，否则后面的断言全被污染

        out_restored = tmp_root / "out_restored"
        with contextlib.redirect_stdout(io.StringIO()):
            mod.main(["--all", "--data-root", str(data), "--out", str(out_restored),
                      "--val-sessions", "2"])
        restored_leaks = _leaks(out_restored)

        print(f"  改成正向（按会话）：泄漏 {restored_leaks or '无'}")
        if bad_leaks:
            print(f"    [不达标] 改成'按帧随机切'后，同一会话出现在两个 split：{bad_leaks}")
            print(f"    [通过] 反向自证有效：泄漏检查能抓住按帧切")
        else:
            print("    [不达标] 改成按帧切居然没泄漏 —— 说明检查器是空转的")
        print(f"    [通过] 还原后泄漏 {restored_leaks or '无'}，恢复正常")
        checks.append(("反向自证：改成按帧切 → 泄漏检查报红", bool(bad_leaks)))
        checks.append(("还原成按会话切 → 泄漏消失（检查不是空转）", not restored_leaks))

        # ==================================================================
        # 11.15 CSV 写入侧的列错位判据（team-lead 2026-10-30 修的那处）
        # ==================================================================
        # 为什么这条要放进"入口一致性"测试：`traj3d.csv` 是**跨模块的契约**，
        # 写它的（`apps/run_stereo_video.py`）与读它的（导出器 / 可视化）不在一个文件里。
        # 未匹配行少写一格不会崩、不会报警，只会让"按 confident 过滤"永远得到 0 行 ——
        # 2026-10-30 实测就是这样静默丢了全部负样本。等长断言 + 这一条一起守。
        from apps import run_stereo_video as RSV

        row = RSV.row_unmatched(7, 0.21, None, None)
        print(f"  row_unmatched 长度 {len(row)}，CSV_HEADER 长度 {len(RSV.CSV_HEADER)}，"
              f"confident 位={row[12]!r}")
        checks.append(("row_unmatched 与 CSV_HEADER 等长（列错位会立刻被发现）",
                       len(row) == len(RSV.CSV_HEADER)))
        checks.append(("未匹配行的 confident 位是 0（不是空串、也不是方法名）",
                       RSV.CSV_HEADER[12] == "confident" and row[12] == 0))

    for label, good in checks:
        print(f"  [{'通过' if good else '不达标'}] {label}")
    return all(good for _label, good in checks)


def main() -> int:
    print("=" * 70)
    print("入口一致性自测（三个入口 -> 同一个测试运行器 / 规范检查器）")
    print("=" * 70)
    results = []
    for fn in (test_runner_and_checker_exist,
               test_bat_uses_runner,
               test_gui_uses_runner,
               test_cli_uses_runner,
               test_agreement_across_entrypoints,
               test_every_cli_parses,
               test_path_bootstrap_is_marker_based,
               test_data_layout_single_source,
               test_stereo_session_layout,
               test_shared_arg_blocks,
               test_annotated_video_atomic_write,
               test_run_all_orchestrator,
               test_export_ball_dataset):
        results.append(fn())
        print()
    ok = all(results)
    print("判定结果：" + ("全部通过" if ok else "有不达标项"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
