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
               test_shared_arg_blocks,
               test_run_all_orchestrator):
        results.append(fn())
        print()
    ok = all(results)
    print("判定结果：" + ("全部通过" if ok else "有不达标项"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
