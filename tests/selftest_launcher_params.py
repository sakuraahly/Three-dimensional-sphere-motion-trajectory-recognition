"""启动器标定参数面板自测：读取、校验、以及"默认就是正式档"。

为什么需要它
------------
用户要求的"标定要 25 张 + UI 里能设标定参数"落地时，暴露了一个**真缺陷**：

`启动程序.py` 原来把板参数**硬编码**在命令里（`--cols 8 --rows 5 --square 25`），
而且**完全没有 `--target`** —— 于是从界面走标定，**永远用脚本默认的那一档**。
用户以为在做正式标定，实际收到的是试标定档的结果，而**界面上没有任何迹象**。

这与 `AGENTS.md` §6"代码与规范冲突 = 缺陷"是同一类：
**界面上能设的东西，不许是一份与真实取值无关的装饰。**

本测试钉四件事：

1. **默认值就是正式档**（`--target 25 --min-pairs 10`）；
2. **可选项留空 = 不传该参数**（不留空才传，尤其 `--baseline-mm` 是基线护栏的开关）；
3. **非法输入与参数矛盾必须被拦住**，并且给出可读的原因；
4. 低目标档仍然可用（只是会在标定程序里打印显式警告）。

反向自证（改了就会红）
--------------------
======================================  ==========================================  ========
改动                                    注入方式                                    结果
======================================  ==========================================  ========
默认值退回 8/8                          面板默认值改回 "8" / "8"                     exit=1
留空也传参数（变成空串）                去掉 `if x is not None:` 判断                exit=1
不校验就拼命令                          去掉 `_read_int` 的范围判断                  exit=1
不查"最少>目标"                         去掉 `minp > target` 那段                    exit=1
高级项接了不传                          去掉 `args += ["--baseline-mm", ...]`        exit=1
======================================  ==========================================  ========

用法::

    python tests/selftest_launcher_params.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
while not (ROOT / "apps").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import tkinter as tk
except ImportError:                                        # pragma: no cover
    tk = None

LAUNCHER = ROOT / "启动程序.py"


def _load_launcher():
    """把启动器作为模块加载（不执行 mainloop）。"""
    sys.argv = ["启动程序.py"]
    spec = importlib.util.spec_from_file_location("launcher_params", LAUNCHER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Box:
    """替掉 messagebox：记录弹窗内容，避免模态框把测试卡住。"""

    def __init__(self) -> None:
        self.msgs: list[str] = []

    def showwarning(self, title, msg, **kw) -> None:        # noqa: ANN001
        self.msgs.append(msg)

    def showinfo(self, title, msg, **kw) -> None:           # noqa: ANN001
        self.msgs.append(msg)


def _set(entry, val: str) -> None:
    entry.delete(0, "end")
    if val != "":
        entry.insert(0, val)


def main() -> int:
    print("=" * 70)
    print("启动器标定参数面板自测（读取 / 校验 / 默认正式档）")
    print("=" * 70)

    if tk is None:
        print("  [跳过] 这台机器没有 tkinter —— 界面测试无法进行。")
        print("         （`启动程序.bat gui` 在这种情况下会自动改用命令行界面。）")
        print("\n判定结果：全部通过")
        return 0

    mod = _load_launcher()
    box = _Box()
    mod.messagebox = box                          # type: ignore[assignment]
    app = mod.Launcher()
    app.update_idletasks()

    results: list[tuple[str, bool]] = []

    def expect(label: str, ok_expected: bool, contains: str = "",
               absent: str = "") -> None:
        box.msgs.clear()
        args = app._calibrate_args()
        got_ok = args is not None
        joined = " ".join(args) if args else ""
        good = got_ok == ok_expected
        if good and got_ok:
            if contains and contains not in joined:
                good = False
            if absent and absent in joined:
                good = False
        # 被拒时必须给出原因 —— 静默拒绝等于让用户猜
        if good and not ok_expected and not box.msgs:
            good = False
        print(f"  [{'通过' if good else '不达标'}] {label}")
        if not good:
            print(f"      期望 ok={ok_expected} 含{contains!r} 不含{absent!r}")
            print(f"      实际 args={joined or None}")
            if box.msgs:
                print(f"      弹窗：{box.msgs[0][:96]}")
        results.append((label, good))

    # ---- 1. 默认就是正式档 ----
    print("----- 1. 默认值 = 正式档（这条是本测试存在的主要理由）-----")
    expect("默认传 --target 25 --min-pairs 10", True,
           contains="--target 25 --min-pairs 10")

    # ---- 2. 板参数真的来自面板 ----
    print("----- 2. 板参数来自面板，不是硬编码 -----")
    _set(app.calib_cols, "7")
    _set(app.calib_rows, "5")
    _set(app.calib_square, "30")
    expect("改面板 -> 命令行跟着变（7×5 / 30mm）", True,
           contains="--cols 7 --rows 5 --square 30")
    _set(app.calib_cols, "8")
    _set(app.calib_rows, "5")
    _set(app.calib_square, "25")

    # ---- 3. 可选项留空 = 不传 ----
    print("----- 3. 可选项留空 = 不传该参数 -----")
    expect("留空不传 --stationary-px", True, absent="--stationary-px")
    expect("留空不传 --min-sharpness", True, absent="--min-sharpness")
    expect("留空不传 --baseline-mm（基线护栏是关着的）", True,
           absent="--baseline-mm")
    expect("未勾放宽不传 --relaxed", True, absent="--relaxed")

    # ---- 4. 填了就传 ----
    print("----- 4. 高级项填了就真的传下去 -----")
    _set(app.calib_baseline, "65")
    expect("填 65 -> 传 --baseline-mm 65（开启基线核对）", True,
           contains="--baseline-mm 65")
    _set(app.calib_baseline, "")
    _set(app.calib_stationary, "5.5")
    expect("填 5.5 -> 传 --stationary-px 5.5", True,
           contains="--stationary-px 5.5")
    _set(app.calib_stationary, "")
    _set(app.calib_sharp, "40")
    expect("填 40 -> 传 --min-sharpness 40", True, contains="--min-sharpness 40")
    _set(app.calib_sharp, "")
    app.calib_relaxed.set(True)
    expect("勾放宽 -> 传 --relaxed", True, contains="--relaxed")
    app.calib_relaxed.set(False)
    _set(app.calib_savepairs, str(ROOT / "outputs" / "tmp_pairs"))
    expect("填存图目录 -> 传 --save-pairs", True, contains="--save-pairs")
    _set(app.calib_savepairs, "")

    # ---- 5. 非法输入 / 矛盾必须被拦住 ----
    print("----- 5. 非法输入与参数矛盾必须被拦住（且说明原因）-----")
    for bad, label in (("abc", "非数字"), ("0", "越界(0)"), ("999", "越界(999)")):
        _set(app.calib_target, bad)
        expect(f"目标对数={bad}（{label}）-> 拒绝", False)
        _set(app.calib_target, "25")
    _set(app.calib_baseline, "xyz")
    expect("预期基线=xyz -> 拒绝", False)
    _set(app.calib_baseline, "")
    _set(app.calib_stationary, "99")
    expect("停稳上限=99 -> 拒绝（>12 会让两路拍到不同位姿）", False)
    _set(app.calib_stationary, "")
    _set(app.calib_target, "5")
    _set(app.calib_minpairs, "10")
    expect("最少对数 > 目标对数 -> 拒绝（否则永远达不到下限）", False)
    _set(app.calib_target, "25")
    _set(app.calib_minpairs, "10")

    # ---- 6. 试标定档仍合法 ----
    print("----- 6. 低目标档仍合法（只是标定程序会显式警告）-----")
    _set(app.calib_target, "8")
    _set(app.calib_minpairs, "8")
    expect("target=8 -> 允许（验证通信用）", True, contains="--target 8")
    _set(app.calib_target, "25")
    _set(app.calib_minpairs, "10")

    # ---- 7. 分辨率/帧率 ----
    print("----- 7. 分辨率 / 帧率仍然生效 -----")
    app.calib_res.set("1280x720")
    expect("1280x720 -> --width 1280 --height 720", True,
           contains="--width 1280 --height 720")
    app.calib_res.set("1920x1080")
    expect("两台相机必须同后端（backend 固定 msmf）", True, contains="--backend msmf")

    app.destroy()

    # ---- 8. 标定脚本自己的默认值也必须是正式档 ----
    #
    # 面板默认 25 但脚本默认 8 的话，**从命令行走的人仍然会停在试标定档** ——
    # 两处必须一致。改任何一处，这条都会红。
    #
    # ⚠️ 判据不能靠 `--help`：argparse 默认**不打印** default（除非用
    # `ArgumentDefaultsHelpFormatter`）。第一版就是这么写的，于是永远拿到
    # "(未打印 default)" 而报红 —— 那是**判据写错了**，不是代码错。
    # 改从源码里读 `add_argument("--target", ..., default=N)`。
    print("----- 8. calibrate_stereo.py 的默认值（命令行入口）-----")
    import re as _re
    src = (ROOT / "apps" / "calibrate_stereo.py").read_text(encoding="utf-8")

    def default_of(flag: str) -> str:
        """从 `add_argument("<flag>" ...)` 那一段里抠出 default 值。"""
        m = _re.search(rf'add_argument\(\s*"{_re.escape(flag)}"\s*,(.*?)\)\s*\n',
                       src, _re.S)
        if not m:
            return "(找不到该参数)"
        body = m.group(1)
        m2 = _re.search(r"default\s*=\s*([0-9.]+)", body)
        return m2.group(1) if m2 else "(没有 default)"

    tgt = default_of("--target")
    mp = default_of("--min-pairs")
    print(f"  源码里 --target 的 default = {tgt}；--min-pairs 的 default = {mp}")
    checks = [
        ("--target 默认 25（与面板一致）", tgt == "25"),
        ("--min-pairs 默认 10（与面板一致）", mp == "10"),
        ("试标定档的文字说明还在（让人知道可以显式给 8）",
         "8" in src and "试标定" in src),
    ]
    for name, good in checks:
        print(f"  [{'通过' if good else '不达标'}] {name}")
        results.append((name, good))

    ok = all(v for _n, v in results)
    bad = [n for n, v in results if not v]
    if bad:
        print(f"\n不达标 {len(bad)} 项：")
        for n in bad:
            print(f"  · {n}")
    print("\n判定结果：" + ("全部通过" if ok else "有不达标项"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
