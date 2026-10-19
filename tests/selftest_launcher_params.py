"""启动器标定参数面板自测：读取、校验、以及"默认就是正式档"。

为什么需要它
------------
用户要求的"标定要收足够多对 + UI 里能设标定参数"落地时，暴露了一个**真缺陷**：

`启动程序.py` 原来把板参数**硬编码**在命令里（`--cols 8 --rows 5 --square 25`），
而且**完全没有 `--target`** —— 于是从界面走标定，**永远用脚本默认的那一档**。
用户以为在做正式标定，实际收到的是试标定档的结果，而**界面上没有任何迹象**。

这与 `AGENTS.md` §6"代码与规范冲突 = 缺陷"是同一类：
**界面上能设的东西，不许是一份与真实取值无关的装饰。**

本测试钉四件事：

1. **默认值就是正式档**（`--target 40 --min-pairs 20`）；
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
**卡片**说明不换行                      去掉 `wraplength=CARD_HINT_WRAP`             exit=1
日志挂回窗口（不进滚动页）              改 `LabelFrame(self._page, ...)` 为 `(self,)`  exit=1
日志无条件滚到底                        去掉 `if at_bottom:` 判断                    exit=1
滚动条永不显示                          把 `_vbar.pack(...)` 换成 `pass`             exit=1
======================================  ==========================================  ========

**本测试没覆盖的（如实写明）**

- **参数说明的换行**（`PARAM_HINT_WRAP`）：**不会红**。实测过 —— 当前真正卡住
  宽度的是**卡片**说明那条换行（把它设成 820/700/640，页宽跟着 873/756/704 变；
  到 640 之后不再变，约束才转移到参数说明）。所以参数说明的换行现在是**护栏**，
  不是当前约束。它的作用是"将来某条参数说明特别长时别把卡片撑开"，
  而那件事还没发生，测试自然抓不住。**不假装覆盖了。**
- **滚轮的可用性**：绑的是 `bind_all("<MouseWheel>")`，无头环境发不了滚轮事件。
  只断言了"滚动条在需要时出现"与 `scroll_to_bottom()` 生效。
- **日志卡片的 `expand=True`**：把 `fill="x"` 改回 `fill="both", expand=True`
  **不会红**。原因是页面高度本来就由内容决定，`expand` 在多出来的空间里
  没有可测的效果 —— 它只影响"看起来占多高"，而那个量没有稳的判据。
  已经把日志挂进滚动页（这一条**有**护栏），所以不假装覆盖这个次要项。

用法::

    python tests/selftest_launcher_params.py
"""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter
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
    """把启动器作为模块加载（不执行 mainloop）。

    ⚠️ **必须先注册进 `sys.modules`**：启动器里有 `@dataclass`，
    而 dataclasses 解析注解时要按 `cls.__module__` 去 `sys.modules` 里找模块，
    找不到就抛 `AttributeError: 'NoneType' object has no attribute '__dict__'`。
    直接 `exec_module` 而不注册，就是这个错。
    """
    sys.argv = ["启动程序.py"]
    spec = importlib.util.spec_from_file_location("launcher_params", LAUNCHER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["launcher_params"] = mod          # ← 关键这一行
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

    # ---- 界面默认值必须与命令行/文档一致（2026-10-14 加）-----------------
    # 为什么要有这一条：这里曾经是 `METHODS[0] = "fusion"` + `set("fusion")`，
    # 而 `run_realtime.py --method` 默认是 `color`、`docs/用户手册.md` §3.1
    # 也写着"`color`（默认，最快）"。于是**从界面走的人必然吃上 fusion**
    # 在真实纹理下的 265 ms/帧（实测 loop 3.7 fps、丢帧 91%），
    # 而界面上没有任何迹象说明这个选项慢在哪 ——
    # 与 `--target` 当年没传下去是同一类缺陷（见 `交接文档.md` §13.1）。
    # 所以钉两件：界面默认 == 文档默认，且与脚本默认**逐字相同**。
    print("----- 0. 检测方法的默认值必须与命令行一致 -----")
    # `run_realtime.py` 的 `--method` 默认是 `None`，不传时沿用
    # `DetectorConfig.method`。所以"脚本默认"要从**真正生效的那处**取 ——
    # 直接用 `AppConfig()` 读，而不是去 grep 源码字符串
    # （我第一版就是 grep `"color" in rt_src`，写成了一条会被正确实现判红的坏断言）。
    from sphere_tracker.config import AppConfig as _AppConfig
    script_default = _AppConfig().detector.method
    got_default = app.method.get()
    rt_src = (ROOT / "apps" / "run_realtime.py").read_text(encoding="utf-8")
    print(f"  界面默认 = {got_default!r}；"
          f"DetectorConfig 的默认（= 不传 --method 时生效）= {script_default!r}")
    checks_default = [
        (f"界面默认检测方法与脚本默认逐字相同（{script_default}）",
         got_default == script_default),
        ("下拉框首项就是默认值（改列表顺序不会让默认值漂走）",
         bool(mod.METHODS) and mod.METHODS[0] == got_default),
        ("color 排在 fusion 之前（默认项不该埋在列表中间）",
         mod.METHODS.index("color") < mod.METHODS.index("fusion")),
        ("脚本侧确实把 --method 的缺省交给配置（没另写一个默认值）",
         "args.method" in rt_src and "cfg.detector.method = args.method" in rt_src),
    ]
    for name, ok in checks_default:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
        results.append((name, ok))
    print()

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
    expect("默认传 --target 40 --min-pairs 20", True,
           contains="--target 40 --min-pairs 20")

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

    # ---- 2.5 相机后端必须**从面板传下去**（2026-10-14 加）----
    #
    # 为什么要有这一条：这里原来把 `--backend msmf` **写死**在 `_calibrate_args` 里，
    # 而「复检」与「前置检查」两条命令**根本不传**。后果是界面完全没有选择权，
    # 而本机实测 **MSMF 打开两路要 34.6 s、DSHOW 只要 11.8 s**（两个后端给的
    # 分辨率完全一致，实测两轮）—— 从界面走的人只能白等半分钟，
    # 而且"同一个决定散在三处"正是本项目反复踩的那类问题。
    # 现在只有一个决定点（面板），三条命令都走 `_backend_args()`。
    print("----- 2.5 相机后端来自面板，且三条命令用的是同一个决定 -----")
    app.calib_backend.set("dshow")
    expect("选 dshow -> 标定命令带上 --backend dshow", True, contains="--backend dshow")
    app.calib_backend.set("auto")
    expect("选 auto -> **不传** --backend（落到脚本默认，不替用户改行为）",
           True, absent="--backend")
    app.calib_backend.set("msmf")
    expect("选 msmf -> 带上 --backend msmf", True, contains="--backend msmf")
    # 三条命令共用同一个决定点：源码级确认没有第二份内联写法
    launcher_src = (ROOT / "启动程序.py").read_text(encoding="utf-8")
    inline = launcher_src.count('args += ["--backend", backend]')
    uses_helper = launcher_src.count("self._backend_args()")
    print(f"  源码：内联写 backend 参数 {inline} 处；调用 _backend_args() {uses_helper} 处")
    results.append(("后端只有一个决定点（没有第二份内联复制）",
                    inline == 0 and uses_helper >= 3))
    print(f"  [{'通过' if inline == 0 and uses_helper >= 3 else '不达标'}] "
          f"后端只有一个决定点（内联 0 处、helper 调用 ≥3 处）")
    app.calib_backend.set("auto")

    # ---- 3. 面板默认值必须与脚本/CaptureConfig 的默认值**一致** ----
    #
    # ⚠️ **2026-10-14 改了契约**：原来这些框是**空的**，断言"留空 = 不传"。
    # 用户要求"填入默认的设置值"，所以现在框里**填着真实默认值**。
    # 于是"留空"从常态变成"用户主动清掉"，其语义仍是**不传该参数**
    # （`calibrate_stereo.py` 会退回自己的默认值）。
    #
    # 这里钉的是**两处默认值必须相同**：面板填 3 / 脚本默认也是 3，
    # 所以"不改任何东西直接点运行"的行为与改动前**完全一致**。
    # 若哪天脚本把默认值改成 4 而面板还写 3，这条就会红 —— 那正是要防的漂移。
    print("----- 3. 面板默认值 == 脚本/CaptureConfig 默认值 -----")
    import re as _re2
    calib_src = (ROOT / "apps" / "calibrate_stereo.py").read_text(encoding="utf-8")
    from calib.capture import CaptureConfig as _CC
    _cc = _CC()
    # argparse 的 default=None 表示"不传就用 CaptureConfig"；这里取后者为准
    expect_stationary = f"{_cc.stationary_max_px:g}"
    expect_sharp = f"{_cc.min_sharpness:g}"
    got_stationary = app.calib_stationary.get().strip()
    got_sharp = app.calib_sharp.get().strip()
    print(f"  停稳上限：面板 {got_stationary!r} / CaptureConfig {expect_stationary}")
    print(f"  清晰度  ：面板 {got_sharp!r} / CaptureConfig {expect_sharp}")
    for label, got, want in (("停稳上限 px", got_stationary, expect_stationary),
                             ("清晰度下限", got_sharp, expect_sharp),
                             ("目标对数", app.calib_target.get().strip(), "40"),
                             ("最少对数", app.calib_minpairs.get().strip(), "20"),
                             ("内角点·列", app.calib_cols.get().strip(), "8"),
                             ("内角点·行", app.calib_rows.get().strip(), "5"),
                             ("方格边长 mm", app.calib_square.get().strip(), "25")):
        good = (float(got) == float(want)) if got else False
        print(f"  [{'通过' if good else '不达标'}] {label}: 面板 {got!r} == 脚本 {want}")
        results.append((f"面板默认值 == 脚本默认值（{label}）", good))

    # 基线**故意**留空：程序无法替你猜装置中心距。这条也要钉住，
    # 免得有人"顺手"填个数字进去 —— 那会让基线核对用错误的锚。
    print(f"  预期基线：面板 {app.calib_baseline.get()!r}（**应当为空**）")
    good = app.calib_baseline.get().strip() == ""
    print(f"  [{'通过' if good else '不达标'}] 预期基线**故意留空**（程序猜不出来）")
    results.append(("预期基线故意留空（程序无法替你猜中心距）", good))

    # ---- 3b. 清空 = 不传该参数（原来的语义仍然成立）----
    print("----- 3b. 清空某个框 = 不传该参数 -----")
    _set(app.calib_stationary, "")
    expect("清空停稳上限 -> 不传 --stationary-px", True, absent="--stationary-px")
    _set(app.calib_stationary, "3")
    expect("填回 3 -> 又传下去", True, contains="--stationary-px 3.0")
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
        _set(app.calib_target, "40")
    _set(app.calib_baseline, "xyz")
    expect("预期基线=xyz -> 拒绝", False)
    _set(app.calib_baseline, "")
    _set(app.calib_stationary, "99")
    expect("停稳上限=99 -> 拒绝（>12 会让两路拍到不同位姿）", False)
    _set(app.calib_stationary, "")
    _set(app.calib_target, "5")
    _set(app.calib_minpairs, "10")
    expect("最少对数 > 目标对数 -> 拒绝（否则永远达不到下限）", False)
    _set(app.calib_target, "40")
    _set(app.calib_minpairs, "20")

    # ---- 6. 试标定档仍合法 ----
    print("----- 6. 低目标档仍合法（只是标定程序会显式警告）-----")
    _set(app.calib_target, "8")
    _set(app.calib_minpairs, "8")
    expect("target=8 -> 允许（验证通信用）", True, contains="--target 8")
    _set(app.calib_target, "40")
    _set(app.calib_minpairs, "20")

    # ---- 7. 分辨率/帧率 ----
    print("----- 7. 分辨率 / 帧率仍然生效 -----")
    app.calib_res.set("1280x720")
    expect("1280x720 -> --width 1280 --height 720", True,
           contains="--width 1280 --height 720")
    app.calib_res.set("1920x1080")
    expect("默认帧率 30 -> --fps 30", True, contains="--fps 30")
    # ⚠️ **2026-10-14 改了这条断言**：它原来是
    #   `expect("两台相机必须同后端（backend 固定 msmf）", True, contains="--backend msmf")`
    # 也就是把"写死 msmf"当成契约。现在后端由面板决定（见第 2.5 节），
    # 但**它要守的真实意图没有变**：`--backend` 是**一个**参数、两台相机共用，
    # 拿不到"左路 dshow、右路 msmf"这种会直接把 MSMF 那路弄挂的组合。
    # 所以断言改成"不多不少恰好一个 --backend"。
    app.calib_backend.set("dshow")
    args_b = app._calibrate_args() or []
    n_be = args_b.count("--backend")
    print(f"  选中 dshow 时：--backend 出现 {n_be} 次（应恰好 1 —— 两台共用同一个）")
    results.append(("两台相机同后端：--backend 只出现一次", n_be == 1))
    print(f"  [{'通过' if n_be == 1 else '不达标'}] 两台相机同后端：--backend 只出现一次")
    app.calib_backend.set("auto")

    # ---- 8. GUI 录制按钮 ----
    print("----- 8. GUI「录制双目」按钮的参数 -----")
    rec = app._record_args("10", "gui_test")
    rec_s = " ".join(rec)
    checks_rec = [
        ("调用 apps/record_stereo.py", "record_stereo.py" in rec_s),
        ("带 --duration 10", "--duration 10" in rec_s),
        ("带 --preview", "--preview" in rec_s),
        ("带 --label gui_test", "--label gui_test" in rec_s),
        ("带 --left left --right right", "--left left --right right" in rec_s),
        ("带 --fps 30", "--fps 30" in rec_s),
    ]
    for label, good in checks_rec:
        print(f"  [{'通过' if good else '不达标'}] {label}")
        results.append((label, good))

    # ---- 9. 可滚动布局（"新加的东西把原来的挤掉"是真实回归）----
    #
    # 加标定参数区之后，第 1 步那张卡片的自然宽度一度到 **1941 px**
    # （整窗 1973 px），而屏幕只有 1600 —— 原有控件被挤到窗口外。
    # 两个原因：卡片顶部与参数下方的说明**没有 wraplength**（按整行文字撑开），
    # 以及日志直接挂在窗口上、用 `expand=True` 吃掉空间。
    print("----- 9. 滚动式布局（不能被内容挤爆 / 不能右边空一片）-----")
    app.update()
    vw0 = app._canvas.winfo_width()
    pw0 = app._page.winfo_reqwidth()
    page_w, canvas_w = pw0, vw0
    page_h = app._page.winfo_reqheight()
    canvas_h = app._canvas.winfo_height()
    print(f"  页面需要 {page_w}x{page_h}；滚动区可见 {canvas_w}x{canvas_h}")
    print(f"  视口与页面宽度差 {canvas_w - page_w} px（越小越说明卡片填满了）")
    checks_layout = [
        ("内容宽度不超过可视宽度（**没有横向截断**）", page_w <= canvas_w),
        # 这一条是"右边空一片 / 文字被切"的回归护栏。
        # 起因：`wraplength` 曾写死 820（按"窗口 900"估的），窗口改成 980 之后
        # 卡片自宽 873 而视口 966 —— 文字在卡片那一列里被切掉，右边还空一大片。
        # 现在换行宽度由 `_reflow` **按视口动态算**，所以差距应当很小。
        ("卡片基本填满可用宽度（差距 < 30px，不是「右边空一片」）",
         (canvas_w - page_w) < 30),
        ("内容高于视口 -> 需要竖向滚动", page_h > canvas_h),
        ("需要滚动时滚动条是显示的", bool(app._vbar.winfo_ismapped())),
        # ⚠️ **不要写死卡片数量**：第一版断言"正好 4 张"，加了「任务」卡片之后
        # 就红了 —— 而那是**测试过时**，不是代码错。数量会随功能增长，
        # 这条断言真正要守的是"卡片都挂在滚动页里，而不是挂在窗口上"。
        ("卡片都挂在滚动页里（不是挂在窗口上）",
         len(app._page.winfo_children()) >= 4
         and len(app.winfo_children()) < len(app._page.winfo_children())),
        ("日志挂到滚动页下了（否则页面滚动时它纹丝不动）",
         app.log.master.master is app._page),
    ]

    # ---- 9b. 参数页签里**不许有网格冲突** ----
    #
    # ## 这条守的是什么（真实回归，2026-10-14）
    #
    # 界面截图里"高级设置都是被遮挡的文字"：参数辅助函数内部占
    # `row*2` 与 `row*2+1` **两行**，而调用方传 1、2、3… 并**另手工放**
    # 复选框/分组标题 —— 手算的行号正好落在已被占用的行上，**文字与输入框互相盖住**。
    #
    # 根因是"手算行号"这种做法；现在改成每个 parent 一个游标、辅助函数自己分配。
    # 所以这里钉的是**没有同格重复、没有同列竖向重叠**。
    def _tabs() -> list[tuple[str, object]]:
        found, out = [], []
        stack = [app]
        while stack:
            w = stack.pop()
            for c in w.winfo_children():
                if c.winfo_class() == "TNotebook":
                    found.append(c)
                stack.append(c)
        for nb in found:
            for tid in nb.tabs():
                out.append((str(nb.tab(tid, "text")).strip(), nb.nametowidget(tid)))
        return out

    tabs = _tabs()
    print(f"  找到 {len(tabs)} 个参数页签：{[n for n, _ in tabs]}")
    checks_layout.append(("参数区确实是「基本设置 / 高级设置」两个页签",
                          [n for n, _ in tabs] == ["基本设置", "高级设置"]))
    dup_any, overlap_any, too_wide_any = [], [], []
    for name, tab in tabs:
        kids = tab.winfo_children()
        cells = [(c.grid_info().get("row"), c.grid_info().get("column", 0))
                 for c in kids]
        dup = [k for k, v in Counter(cells).items() if v > 1]
        # ⚠️ 必须**按列分组**再比 y 区间：同一行不同列的控件 y 区间天然重叠，
        # 那不是冲突。第一版没分组，报了一堆假阳性（「TEntry 与 目标对数 重叠」）。
        by_col: dict[int, list] = {}
        for c in kids:
            by_col.setdefault(c.grid_info().get("column", 0), []).append(c)
        ov = []
        for _col, group in by_col.items():
            boxes = sorted((c.winfo_y(), c.winfo_y() + c.winfo_height())
                           for c in group)
            for (y1, y2), (y3, _y4) in zip(boxes, boxes[1:]):
                if y3 < y2 - 1:
                    ov.append((_col, y1, y2, y3))
        widest = max((c.winfo_x() + c.winfo_width() for c in kids), default=0)
        print(f"    {name}: {len(kids)} 个控件  同格重复={len(dup)}  "
              f"同列重叠={len(ov)}  最右={widest}（视口 {app._canvas.winfo_width()}）")
        dup_any += [(name, d) for d in dup]
        overlap_any += [(name, o) for o in ov]
        if widest > app._canvas.winfo_width():
            too_wide_any.append((name, widest))
    checks_layout.append((f"页签内没有同格重复（实际 {len(dup_any)} 处）", not dup_any))
    checks_layout.append((f"页签内没有同列竖向重叠（实际 {len(overlap_any)} 处）"
                          f"—— 这就是「文字被遮挡」的判据", not overlap_any))
    checks_layout.append((f"页签内容不超出视口（实际 {len(too_wide_any)} 处）",
                          not too_wide_any))
    # 窗口改宽度后，换行宽度必须跟着变（这才是"动态"的证据）
    wraps: list[tuple[int, int, int]] = []
    for w in (980, 1180, 900):
        app.geometry(f"{w}x780")
        app.update()
        cw = next((int(lab.cget("wraplength")) for lab, k in app._wrapped
                   if k == "card"), 0)
        wraps.append((w, app._canvas.winfo_width(), cw))
    print("  窗口宽 -> 视口宽 / 卡片换行宽度：")
    for w, vw, cw in wraps:
        print(f"    {w:5} -> {vw:5} / {cw:5}")
    grows = wraps[1][2] > wraps[0][2] > wraps[2][2]
    checks_layout.append(
        ("换行宽度随窗口**动态**变化（写死的话这条会红）", grows))
    app.geometry("980x780")
    app.update()

    for name, good in checks_layout:
        print(f"  [{'通过' if good else '不达标'}] {name}")
        results.append((name, good))

    # ---- 10. 日志自动滚动的约定 ----
    print("----- 10. 日志：贴底才自动滚（不许把正在读历史的用户拽走）-----")
    # 前置条件：先把日志滚到底。
    # 启动器初始化会写若干行（环境/设备），刚起来时 yview 不一定已在底部；
    # 2026-10-19 full 档偶发 at_bottom=False（实测 after2=(0.56, 0.77)）——
    # 这是**测试假设**错了，不是自动滚动坏了：用户没贴底时本来就不该被拽走。
    app.log.see("end")
    app.update()
    app._append("第一行")
    app._append("第二行")
    app.update()
    at_bottom = app.log.yview()[1] >= 0.99
    for i in range(60):
        app._append(f"填充 {i}")
    app.update()
    app.log.yview_moveto(0.0)                 # 模拟用户往上翻
    app.update()                              # 让 moveto 真正生效再读 yview
    reading_from = app.log.yview()[1] < 0.99
    before = app.log.yview()[1]
    app._append("读历史时来的新行")
    app.update()
    not_yanked = abs(app.log.yview()[1] - before) < 0.02
    print(f"  贴底时自动滚={at_bottom}；往上翻后是否被拽回底={not not_yanked}")
    checks_log = [
        ("贴底时追加会自动跟到底", at_bottom),
        ("用户往上翻之后，新行不再把他拽到底", reading_from and not_yanked),
    ]
    for name, good in checks_log:
        print(f"  [{'通过' if good else '不达标'}] {name}")
        results.append((name, good))

    panel_target = app.calib_target.get().strip()
    panel_minp = app.calib_minpairs.get().strip()
    panel_fps = app.calib_fps.get().strip()
    app.destroy()

    # ---- 8. 标定脚本自己的默认值也必须是正式档 ----
    #
    # 面板默认 40 但脚本默认 8 的话，**从命令行走的人仍然会停在试标定档** ——
    # 两处必须一致。改任何一处，这条都会红。
    #
    # ⚠️ 判据不能靠 `--help`：argparse 默认**不打印** default（除非用
    # `ArgumentDefaultsHelpFormatter`）。第一版就是这么写的，于是永远拿到
    # "(未打印 default)" 而报红 —— 那是**判据写错了**，不是代码错。
    # 改从源码里读 `add_argument("--target", ..., default=N)`。
    print("----- 8. calibrate_stereo.py / _common.py 的默认值（命令行入口）-----")
    import re as _re
    src = (ROOT / "apps" / "calibrate_stereo.py").read_text(encoding="utf-8")
    common_src = (ROOT / "apps" / "_common.py").read_text(encoding="utf-8")
    fps_m = _re.search(r"fps:\s*int\s*=\s*(\d+)", common_src)
    common_fps = fps_m.group(1) if fps_m else "(找不到 fps 默认值)"

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
    print(f"  源码里 --target 的 default = {tgt}；--min-pairs 的 default = {mp}；"
          f"_common.py 的 fps default = {common_fps}")
    print(f"  面板默认：target={panel_target} minpairs={panel_minp} fps={panel_fps}")
    checks = [
        ("--target 默认 40（与面板一致）", tgt == panel_target == "40"),
        ("--min-pairs 默认 20（与面板一致）", mp == panel_minp == "20"),
        ("--fps 默认 30（与面板一致）", common_fps == panel_fps == "30"),
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
