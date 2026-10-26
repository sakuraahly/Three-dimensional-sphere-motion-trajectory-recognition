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
        code, out, _err = _run_all("--session", str(sess), "--dry-run",
                                   "--stereo-mode", "angled")

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
               test_run_all_orchestrator):
        results.append(fn())
        print()
    ok = all(results)
    print("判定结果：" + ("全部通过" if ok else "有不达标项"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
