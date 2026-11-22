"""窗口助手自测：全屏 / 尺寸 / 按键约定，以及**各程序有没有接上**。

为什么需要它
------------
用户的原话是"这个窗口太小了，所有程序的视频显示窗口都要有全屏模式"。
根因有两层：

1. `run_realtime` / `run_video` / `calibrate_stereo` / `check_stereo_setup`
   **根本没建窗口** —— `cv2.imshow` 自动建的是 `WINDOW_AUTOSIZE`，
   **尺寸等于图片**，而且**用户拖不动**。显示宽度改成 960 之后，
   窗口就只有 960x540，在 3200x2000 的屏上只占中间一小块。
2. `calibrate_color` / `calibrate_focal` / `cameras` 建了 `WINDOW_NORMAL`，
   但没有全屏。

## 一个必须钉住的坑

实测本机（OpenCV 5.0）：

    cv2.WINDOW_NORMAL     = 0
    cv2.WINDOW_AUTOSIZE   = 1
    cv2.WINDOW_FULLSCREEN = 1     ← 和 AUTOSIZE **同一个值**！

所以 `cv2.namedWindow(name, cv2.WINDOW_FULLSCREEN)` **得不到全屏窗口**，
它等价于 AUTOSIZE。全屏**只能**走 `setWindowProperty`。
本测试会把这个值关系也断言下来 —— 哪天换了 OpenCV 版本、
两者真的分开了，这条会红，提醒可以简化实现（**消掉一个坑也是进展**）。

反向自证（改了就会红）
--------------------
======================================  ==========================================  ========
改动                                    注入方式                                    结果
======================================  ==========================================  ========
用 WINDOW_FULLSCREEN 建窗（不生效）      `namedWindow(name, WINDOW_NORMAL)` 改成         见下方说明
                                        `WINDOW_FULLSCREEN`
Esc 不再优先"退出全屏"                   去掉 `if key == 27` 那段                     exit=1
`f` 抢走别人已用的键                     `toggle_keys` 默认值加上 `ord("s")`          exit=1
某个程序没接上窗口助手                   删掉一处 `wins.show(`                        exit=1
======================================  ==========================================  ========

用法::

    python tests/selftest_window_mode.py
"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
while not (ROOT / "apps").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import cv2
except ImportError:                                          # pragma: no cover
    cv2 = None

# 所有"会开视频窗口"的程序。少一个都算漏 —— 用户说的是"**所有**程序"。
APPS = (
    "apps/run_realtime.py",
    "apps/run_video.py",
    "apps/calibrate_stereo.py",
    "apps/tools/check_stereo_setup.py",
    "apps/tools/cameras.py",
    "apps/tools/calibrate_color.py",
    "apps/tools/calibrate_focal.py",
    # 2026-10-27 补：单板链式标定的收图窗口（双路成对 + 单路）一直没接，
    # 于是它既不能全屏、窗口也拖不动（AUTOSIZE + 固定 2560x720 拼图）。
    # 用户的原话是"所有程序的视频显示窗口都要有全屏模式"—— 这就是漏掉的那个。
    "apps/tools/capture_board.py",
    # 2026-10-29 补：单目录制的预览窗**从来没进过这张表**，于是它用了一个
    # 早就不成立的签名 `wins.ensure(WINDOW)`（签名是 `ensure(name, image)`）——
    # 真机表现是"倒计时走完、开录那一刻"抛 TypeError，10 秒白等。
    # 顺带它也因此一直没有全屏（直接 imshow）。
    "apps/record_mono.py",
)
# 这些程序里 `f` 已经被别的功能占用，全屏只能留给 F11。
# **抢掉别人已用的键比没有全屏更糟** —— 那是把一个既有功能弄坏。
F_KEY_TAKEN_BY_APP = {"apps/tools/calibrate_color.py": "f"}

#: 走查报告（交叉验证用）。默认读最近一次 verify 走查；可用环境变量指到别处。
WALK_REPORT = pathlib.Path(
    __import__("os").environ.get("GUI_WALK_REPORT")
    or "outputs/gui_walk/verify/report.json")
if not WALK_REPORT.is_absolute():
    WALK_REPORT = ROOT / WALK_REPORT


def _descendants(w):
    yield w
    for c in w.winfo_children():
        yield from _descendants(c)


def _scroll_page(win):
    """窗口里的滚动页（鸭子类型：同时有 ``_canvas`` 与 ``content``）。

    不按类名认：ScrollPage 的 Tk 类是 ``TFrame``，按类名会认成普通 Frame
    而漏掉 —— 走查工具第一版就是这么漏过一扇窗的。
    """
    for w in _descendants(win):
        canvas = getattr(w, "_canvas", None)
        content = getattr(w, "content", None)
        if canvas is not None and content is not None:
            return w
    return None


def _blocks(content) -> list:
    """内容区里的每一"块"（卡片 / 折叠区），按竖直位置排。

    为什么要按块而不是按"整页"：整页只有一个数，红了也说不清是哪一步；
    按块能直接说"③ 采集 的下沿超出视口 57 px"。
    """
    kids = list(content.winfo_children())
    kids.sort(key=lambda w: int(w.winfo_y()))
    return kids


def _block_title(block) -> str:
    """这一块叫什么（卡片标题 / 折叠区标题）。"""
    for attr in ("_title",):                 # CollapsibleSection：标题存在 _title
        title = getattr(block, attr, "")
        if title:
            return str(title)
    try:
        text = str(block.cget("text") or "")
    except Exception:                                        # noqa: BLE001
        text = ""
    if text:
        return text.split("（")[0].strip()
    return block.winfo_class()


def _measure(win) -> tuple[list[tuple[str, int]], int]:
    """(每一步的 (标题, 下沿 y), 视口高度)。单位 px。"""
    page = _scroll_page(win)
    if page is None:
        return [], 0
    content, canvas = page.content, page._canvas
    win.update_idletasks()
    win.update()
    vp = int(canvas.winfo_height())
    rows = [(f"{_block_title(b)}", int(b.winfo_y()) + int(b.winfo_reqheight()))
            for b in _blocks(content)]
    return rows, vp


def _folds(win) -> list:
    """窗口里的折叠区（CollapsibleSection 的鸭子类型）。"""
    return [w for w in _descendants(win)
            if hasattr(w, "toggle") and hasattr(w, "expanded")
            and hasattr(w, "content")]


def _expand_folds(win) -> int:
    """展开窗口里所有折叠区，返回展开前的**内容**高度（用来比对增量）。"""
    page = _scroll_page(win)
    base = int(page.content.winfo_reqheight()) if page is not None else 0
    for w in _folds(win):
        if not w.expanded:
            w.toggle()
    win.update_idletasks()
    win.update()
    return base


def _tk_first_screen_checks(ck) -> None:
    """每个 Tk 窗口、每一步：内容下沿必须在视口里（首屏不用滚）。

    另加一条"高级展开"的判据，但它的口径与首屏**不同**，理由写在下面那段。
    """
    print("----- 8. Tk 界面：每一步都要在视口里（不许被裁）-----")
    try:
        import tkinter as tk
    except ImportError:                                      # pragma: no cover
        print("  [跳过] 没有 tkinter")
        return
    try:
        from apps._gui_app import App
        app = App()
        app.update_idletasks()
    except tk.TclError as exc:                               # 无显示器 / 远程会话
        print(f"  [跳过] Tk 起不来（{type(exc).__name__}: {exc}）")
        return

    wins: list[tuple[str, object]] = []
    try:
        wins = [
            ("hub", app.hub),
            ("mono", app.open_mono()),
            ("stereo(parallel)", app.open_stereo("parallel")),
            ("stereo(angled)", app.open_stereo("angled")),
            ("bench", app.open_bench()),
        ]
        app.update_idletasks()

        for name, win in wins:
            win.deiconify()
            win.lift()
            rows, vp = _measure(win)
            if not rows:
                print(f"  [跳过] {name}：没有滚动页（本判据不适用）")
                continue
            print(f"  {name}：窗口 {win.winfo_width()}x{win.winfo_height()}，"
                  f"视口 {vp} px，{len(rows)} 步")
            for label, bottom in rows:
                diff = bottom - vp
                ck(f"{name} · {label}：reqheight {bottom} ≤ 实际 {vp}"
                   f"（差 {diff:+d} px）", diff <= 0)

            # ---- 高级折叠区展开之后 ----
            # 口径**故意**与首屏不同：展开是用户主动要看更多，内容必然变长，
            # 只要窗口能滚到底就成立。若硬要求"展开后也装得下"，唯一出路是
            # 让窗口在用户点开折叠时自己抖一下变高 —— 那比滚动更糟；而大角度
            # 窗在本机已经是 903/904 px，也没有再长的余地。
            # 所以这里断言的是三件与"裁切"真正相关的事：
            #   ① 展开确实加了内容（否则折叠区是死的）；
            #   ② 加的**只有折叠区**那么高（别的地方没跟着乱跳）；
            #   ③ 多出来的部分真的能滚到（scrollregion 覆盖整页）；
            #   ④ 窗口高度没有被内容撑大（高度仍由内容+屏幕决定，不靠"再高一点"）。
            if not _folds(win):
                print("    ├ 没有折叠区，展开后那条判据不适用")
                continue
            h_before = int(win.winfo_height())
            content_before = _expand_folds(win)
            page = _scroll_page(win)
            content_after = int(page.content.winfo_reqheight())
            grew = content_after - content_before
            fold_h = sum(int(w.content.winfo_reqheight()) for w in _folds(win))
            reachable = int(page._canvas.bbox("all")[3]) >= content_after - 2
            rows2, vp2 = _measure(win)
            worst = max((b - vp2 for _l, b in rows2), default=0)
            print(f"    ├ 展开高级后：内容 {content_before} → {content_after}"
                  f"（+{grew}，折叠区本身 {fold_h}），视口 {vp2}，"
                  f"最下沿超出 {worst} px（靠滚动，可滚到={reachable}）")
            ck(f"{name} · 高级展开：内容真的变长（+{grew} px）", grew > 0)
            ck(f"{name} · 高级展开：增量只来自折叠区（+{grew} ≤ {fold_h} + 20）",
               grew <= fold_h + 20)
            ck(f"{name} · 高级展开：能滚到底（scrollregion 覆盖整页）", reachable)
            ck(f"{name} · 高级展开：窗口高度没被内容撑大"
               f"（{win.winfo_height()} = 展开前 {h_before}）",
               int(win.winfo_height()) == h_before)
    finally:
        for _n, win in wins:
            try:
                win.destroy()
            except Exception:                                # noqa: BLE001
                pass
        try:
            app.destroy()
        except Exception:                                    # noqa: BLE001
            pass


def _gui_fingerprint() -> dict[str, str]:
    """当前这版 GUI 源码的指纹：`文件名 -> md5 前 10 位`。

    **与走查工具 `_gui_fingerprint()` 同一算法**（同一批文件、同一段截断）。
    这是两边共用的口径，改算法必须同时改两处 —— 否则版本比对会永远
    "不一致"，本节就变成永远跳过。
    """
    import hashlib
    out: dict[str, str] = {}
    # 必须与走查工具逐字一致（含 `*_dialog.py` 这一类）：2026-11-07 新加的
    # `tools_dialog.py` 一开始落在两边之外，于是它改了也不会让产物失效。
    gui_files = sorted((ROOT / "apps").glob("_gui_*.py")) + \
        sorted((ROOT / "apps").glob("*_dialog.py"))
    for p in gui_files:
        try:
            out[p.name] = hashlib.md5(p.read_bytes()).hexdigest()[:10]
        except OSError:
            out[p.name] = "?"
    for name in ("_task_runner.py", "_collapsible.py"):
        p = ROOT / "apps" / name
        if p.is_file():
            try:
                out[name] = hashlib.md5(p.read_bytes()).hexdigest()[:10]
            except OSError:
                pass
    return out


def _artifact_is_current(data: dict) -> tuple[bool, str]:
    """走查产物是不是**当前这一版 GUI**跑出来的。

    为什么必须有这条判据：产物躺在 `outputs/` 里，跑一次就一直在。
    之后再改 GUI 源码，**报告里的结论就不再代表现在的代码** —— 拿它当门禁
    就是假绿（真的发生过：报告 19:26:40 生成，而 `_gui_app.py` 19:24:42 还在改，
    那次走查是"混版本"的）。版本对不上时本节**不表态**（`[跳过]`），
    而不是给一个看起来正常的绿。
    """
    want = data.get("gui_files")
    if not isinstance(want, dict) or not want:
        return False, "产物里没有 GUI 版本指纹（老版走查工具生成的）"
    have = _gui_fingerprint()
    diff = [k for k in sorted(set(want) | set(have)) if want.get(k) != have.get(k)]
    if diff:
        return False, "这些文件在走查之后又改过：" + "、".join(diff)
    return True, ""


def _overflow_findings(steps: list[dict]) -> tuple[list, list]:
    """把「这一步该不该滚」判出来：返回 ``(首屏要滚的, 豁免却滚不到底的)``。

    抽成纯函数只为了一件事：能拿假步骤反证。判据读的是**走查工具写在报告里的
    理由**（``overflow_exempt``），不是标题文字 —— 旧写法靠 ``"展开" not in title``
    认豁免，标题改一个字就悄悄失效，反向还能把一条真溢出洗白。

    豁免不是空白支票：写「这是滚动页」的同时，走查工具会滚到底量 ``yview``
    终点（``scroll_reach``）。量不到 1.0 就是"最后一屏是死区"，照样算缺陷。
    """
    needs: list[tuple] = []
    dead: list[tuple] = []
    for s in steps:
        if s.get("window") == "external":
            continue
        px = int(s.get("overflow_px") or 0)
        if px <= 20:                      # 与走查工具 _note_overflow 同一阈值
            continue
        if not str(s.get("overflow_exempt") or ""):
            needs.append((s.get("window"), s.get("title"), px))
            continue
        reach = s.get("scroll_reach")
        reach = float(reach) if reach is not None else -1.0
        if reach < 0.999:
            dead.append((s.get("window"), s.get("title"), reach))
    return needs, dead


def _cross_check_walkthrough(ck) -> None:
    """与真人走查产物对账：两边必须说同一句话。"""
    print("----- 9. 与真人走查交叉验证（clipped 必须全为无）-----")
    path = WALK_REPORT
    if not path.is_file():
        print(f"  [跳过] 没有走查产物：{path}")
        print("         生成：python apps/tools/gui_walkthrough.py "
              "--out outputs/gui_walk/verify --only hub,mono,stereo,bench")
        print("         ⚠ 本节本次**没有验证任何东西**（跳过 ≠ 验证过）")
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"  [跳过] 走查产物读不了（{type(exc).__name__}: {exc}）")
        return
    fresh, why = _artifact_is_current(data)
    if not fresh:
        print(f"  [跳过] 走查产物不是当前这一版 GUI：{why}")
        print("         重新生成：python apps/tools/gui_walkthrough.py "
              "--out outputs/gui_walk/verify")
        print("         ⚠ 本节本次**没有验证任何东西**（跳过 ≠ 验证过）")
        return
    uncovered = data.get("uncovered") or []
    if uncovered:
        print(f"  [注意] 该产物是非全量走查，未覆盖：{'、'.join(uncovered)}"
              f" —— 下面的结论**只对走查过的窗口成立**")
    steps = data.get("steps") or []
    print(f"  {path}：{len(steps)} 步")

    clipped = [(s.get("window"), s.get("title"), s.get("clipped"))
               for s in steps if s.get("clipped")]
    for w, t, c in clipped:
        print(f"    ! {w} · {t}：{len(c)} 个控件在窗口外 {c[:2]}")
    ck("走查：没有被窗口边缘切掉的控件（clipped 全为无）", not clipped)

    # 走查工具自己记的缺陷也要算数。以前这里只对账 clipped / overflow 两个**汇总
    # 字段**，`defects` 一张都没人读 —— 于是「工具在某一节新加了一条缺陷」这种
    # 情况，报告里红着，测试照样绿（2026-11-07 加"豁免了却没有滚动容器"那条时
    # 发现的：那条缺陷只在 tools 的工具里产生，ov 可能为 0，永远进不了下面两个
    # 汇总判据）。缺陷清单非空 = 这一版 GUI 不该算验收通过。
    defects = data.get("defects") or []
    for d in defects[:3]:
        print(f"    ! [{d.get('severity')}] {d.get('where')}：{d.get('what')}")
    ck(f"走查：缺陷清单为空（走查工具记了 {len(defects)} 条）", not defects)

    # 首屏步骤必须不用滚。豁免**不靠标题文字**认（旧写法是 `"展开" not in title`：
    # 改一个字标题，判据就悄悄失效；反过来，一条真溢出的步骤只要标题里带「展开」
    # 也会被放过 —— 两边说的不是同一件事）。现在由走查工具在报告里写
    # `overflow_exempt` 理由，判据只认这个字段，并且**逐条验收豁免本身**：
    # 它必须真能滚到底（`scroll_reach` == 1.0），否则"按设计要滚"就是死区。
    # 判定抽在 `_overflow_findings()` 里，下面拿假步骤反证它抓得到、也不误伤。
    # 阈值取 20 px，**与走查工具 `_note_overflow` 的阈值逐字相同**：
    # 这一节的任务是"两边说同一句话"，口径不一致的话红绿就不代表同一件事。
    # （第 8 节是更严的 ≤ 0，管我们自己能控的那几扇窗。）
    for s in steps:
        if not str(s.get("overflow_exempt") or ""):
            continue
        px = int(s.get("overflow_px") or 0)
        if px <= 20:          # 本来就没超标：豁免没被用上，不必打印干扰视线
            continue
        print(f"    · 豁免 {s.get('window')} · {s.get('title')}："
              f"高 {px} px —— {s.get('overflow_exempt')}"
              f"（实测滚到底 {float(s.get('scroll_reach') or -1):.2f}）")
    needs, dead = _overflow_findings(steps)
    for w, t, px in needs:
        print(f"    ! {w} · {t}：内容比视口高 {px} px（首屏要滚）")
    for w, t, reach in dead:
        print(f"    ! {w} · {t}：豁免理由是「滚动页」，但滚到底只到 {reach:.2f}"
              f" —— 最后一屏是死区")
    ck("走查：首屏步骤都不用滚（overflow_px ≤ 20，与走查工具同一阈值）", not needs)
    ck("走查：豁免「按设计要滚」的步骤都真能滚到底（yview 终到 1.00）", not dead)

    # 反证：这套判定自己也得能被喂红。三条假步骤分别代表「真缺陷」
    # 「豁免是空头支票」「合规豁免」—— 只会判绿的空闸门在这里通不过。
    fake_bad = [{"window": "x", "title": "首屏溢出没豁免", "overflow_px": 100}]
    fake_dead = [{"window": "x", "title": "自称滚动页", "overflow_px": 100,
                  "overflow_exempt": "设计如此", "scroll_reach": 0.5}]
    fake_ok = [{"window": "x", "title": "真能滚到底", "overflow_px": 100,
                "overflow_exempt": "设计如此", "scroll_reach": 1.0}]
    ck("反证：首屏溢出没豁免 → 判据会红",
       bool(_overflow_findings(fake_bad)[0]))
    ck("反证：豁免了却滚不到底 → 判据会红",
       bool(_overflow_findings(fake_dead)[1]))
    ck("反证：合规的豁免不会被误判",
       not any(_overflow_findings(fake_ok)))


def _debt_guards(ck) -> None:
    """B-150/161/172/156/149/151/114/124 的可复现静态判据（含反向验证）。

    每条都证明两件事：① 当前源码满足护栏；② 把修复前的坏写法喂回去，
    同一条判据会变红 —— 否则「绿」只是碰巧，半年后照样漂。
    """
    import ast

    def src(rel: str) -> str:
        return (ROOT / rel).read_text(encoding="utf-8")

    mono = src("apps/_gui_mono.py")
    stereo = src("apps/_gui_stereo.py")
    sv = src("apps/run_stereo_video.py")
    rt = src("apps/run_tests.py")
    ms = src("apps/_mono_sessions.py")
    rv = src("apps/run_video.py")
    wt = src("apps/tools/gui_walkthrough.py")
    here = src("tests/selftest_window_mode.py")

    # ---- B-150：不静默改持久化偏好；切换要有状态条说明 ----
    def b150(s: str) -> bool:
        return ("var_detector.set(chosen)" not in s
                and "switch_note" in s and "你的偏好" in s)
    ck("B-150 不静默改 var_detector、切换有状态说明", b150(mono))
    ck("B-150 反向：把 var_detector.set(chosen) 写回去，判据会红",
       not b150(mono.replace("switch_note = \"\"",
                             "switch_note = \"\"\n                self.var_detector.set(chosen)", 1)))

    # ---- B-150 补：**存种子**时同样不许改写持久化偏好 ----
    # 2026-11-07 独立 Critic 抓出：上一轮只修了 `_start_video` 里的
    # `var_detector.set(chosen)`，`_on_seed_saved`（picker 存完种子）里还有一处
    # `var_detector.set(det)` —— 同一型缺陷只修了一半，判据也只盯了一个字面量。
    def b150b(s: str) -> bool:
        return ("var_detector.set(det)" not in s
                and "你的偏好 {pref} 不变" in s)

    ck("B-150 补：存种子不写回 var_detector（本次 vs 长期偏好分离）", b150b(mono))
    ck("B-150 补 反向：把 var_detector.set(det) 写回去，判据会红",
       not b150b(mono.replace(
           "        pref = self.var_detector.get()",
           "        self.var_detector.set(det)\n        pref = self.var_detector.get()", 1)))

    # ---- B-150 丙：**auto 重试**同样不许改写持久化偏好 ----
    # 2026-11-07 Critic 75b 复评抓出第三处：`_on_detected` 里把下拉框直接写成
    # hough，随后 `_start_video` 重跑、`_on_close` 落盘 —— 用户选的 auto 被永久
    # 改成 hough，下次开窗都不再 auto。前两条判据只盯 `set(chosen)`/`set(det)`
    # 两个字面量，抓不到这一处，所以单列一条。
    def b150c(s: str) -> bool:
        return ('var_detector.set("hough")' not in s
                and "_retry_method_override" in s
                and "你的偏好 auto 不变" in s)

    ck("B-150 丙：auto 重试用一次性覆盖，不改 var_detector", b150c(mono))
    ck("B-150 丙 反向：把 var_detector.set(\"hough\") 写回去，判据会红",
       not b150c(mono.replace(
           '            self._retry_method_override = "hough"',
           '            self.var_detector.set("hough")', 1)))

    # ---- B-150 丁：一次性覆盖必须在**任何提前 return 之前**取走 ----
    # Critic 75c 的 P3：`_start_video` 里球径检查就可能直接 return。override
    # 放在它后面取走的话，「球径被改坏 → 本次启动取消」会把 hough 留在字段里，
    # 等改好球径再点启动就**静默用 hough**，而 auto 已被 `and not retry_method` 关掉。
    def b150d(s: str) -> bool:
        start = s.find("def _start_video")
        if start < 0:
            return False
        body = s[start:]
        i_take = body.find("self._retry_method_override = None")
        i_dia = body.find("problem = diameter_problem(self.var_diameter.get())")
        return 0 <= i_take < i_dia

    ck("B-150 丁：override 取走早于球径检查（提前 return 也不泄漏）", b150d(mono))
    _fake_late = ("    def _start_video(self):\n"
                  "        problem = diameter_problem(self.var_diameter.get())\n"
                  "        if problem:\n"
                  "            return\n"
                  "        retry_method = self._retry_method_override\n"
                  "        self._retry_method_override = None\n")
    ck("B-150 丁 反向：取走晚于球径检查，判据会红", not b150d(_fake_late))

    # ---- B-156：按实际跳过集打印，full 档没有真跳过就不打印 ----
    def b156(s: str) -> bool:
        return ("actually_skipped = [n for n in all_names if n not in set(selected)]" in s
                and "if actually_skipped:" in s)
    ck("B-156 按 actually_skipped 打印（full 档不再误报）", b156(rt))
    ck("B-156 反向：旧写法 `if slow: 本档跳过` 不满足判据",
       not b156("    if slow:\n        print(\"本档跳过：\")"))

    # ---- B-161/172：标准 JSON 与 CSV/视频同一落点变量 ----
    def b161(s: str) -> bool:
        return ("csv_name=str(csv_path.resolve())" in s
                and "traj_path = out_dir / \"traj3d.json\"" in s
                and "全部产物（含标准 JSON）写到" in s)
    ck("B-161/172 标准 JSON 与产物从同一个 out_dir 派生", b161(sv))
    ck("B-161 反向：旧写法 csv_name=csv_path.name 不满足判据",
       not b161("session_dir, csv_name=csv_path.name"))

    # ---- B-149：双目窗复用同一份 role_online，缺相机给可执行下一步 ----
    ck("B-149 双目窗复用单目窗同一份 role_online()",
       "def role_online(" in mono and "role_online(" in stereo)
    ck("B-149 缺相机给可执行下一步（identify / 换相机）",
       "identify" in stereo and "换相机" in stereo)

    # ---- B-151：两份 _gui_fingerprint 函数体逐结构一致，漂移即红 ----
    def fp_ast(s: str) -> str:
        tree = ast.parse(s)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "_gui_fingerprint")
        body = list(fn.body)
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            body = body[1:]
        return ast.dump(ast.Module(body=body, type_ignores=[]))
    a, b = fp_ast(wt), fp_ast(here)
    ck("B-151 两份 _gui_fingerprint 函数体 AST 一致（防漂移）", a == b)
    idx = here.index("def _gui_fingerprint")
    drifted = here[:idx] + here[idx:].replace("[:10]", "[:8]", 1)
    ck("B-151 反向：任一份被改坏后判据会红", fp_ast(drifted) != b)

    # ---- B-114：球径写进会话，缺值 null 不是 0（行为级验证）----
    import tempfile
    from apps import _mono_sessions as _ms
    def b114(d: dict) -> bool:
        # 字段必须存在，且缺值不许兜成 0（0 会被当成有效测量）
        return "ball_diameter_mm" in d and d.get("ball_diameter_mm") != 0
    with tempfile.TemporaryDirectory() as _td:
        d1 = _ms.create_mono_session(_td, now=0)             # 不给球径
        raw1 = json.loads((d1 / "session.json").read_text(encoding="utf-8"))
        d2 = _ms.create_mono_session(_td, ball_diameter_mm=40.0, now=1)
        raw2 = json.loads((d2 / "session.json").read_text(encoding="utf-8"))
        ck("B-114 新会话缺球径写 null、已知球径写进元数据",
           b114(raw1) and raw1.get("ball_diameter_mm") is None
           and raw2.get("ball_diameter_mm") == 40.0)
        rows = {r.id: r for r in _ms.scan_mono_sessions(_td)}
        ck("B-114 摘要能读回球径", rows[d2.name].ball_diameter_mm == 40.0)
    ck("B-114 录制时把已知球径写进会话", "ball_diameter_mm=known_d" in mono)
    ck("B-114 反向：把缺值兜成 0（旧习惯）会被判据抓到",
       not b114({"ball_diameter_mm": 0}))

    # ---- B-124：边车相机身份有只读消费点 ----
    ck("B-124 run_video 有 _consume_session_camera 只读消费点",
       "def _consume_session_camera" in rv
       and "_consume_session_camera(" in rv)


def _standard_window_chrome(ck) -> None:
    """§11：每个 GUI 窗口都要有最小化 / 最大化 / 关闭（AGENTS.md §7.1）。

    两层，缺一层都不算验证过：
      ① 静态 AST 扫 `apps/*.py`：不许出现 transient / overrideredirect /
         grab_set / resizable(False) / -toolwindow。比字符串会把自己写的注释
         判成实现，所以走 AST（这个亏 §75 吃过一次）。
      ② 运行时读 Win32 真实窗口样式位。Tk 的 `resizable()` 只能说明"Tk 认为
         能拉伸"，真正决定标题栏按钮的是 `WS_MINIMIZEBOX` / `WS_MAXIMIZEBOX` /
         `WS_SYSMENU` 和 `WS_EX_TOOLWINDOW` —— 实测 `transient()` 会把前两个
         位清成 0，界面上就只剩一个 ×。
    """
    import ast
    import sys

    print("----- 11. 每个 GUI 窗口都要有最小化 / 最大化 / 关闭（AGENTS.md §7.1）-----")

    BANNED = {"transient", "overrideredirect", "grab_set", "toolwindow"}

    def scan(src: str) -> list[tuple[int, str]]:
        bad = []
        for node in ast.walk(ast.parse(src)):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            if name in BANNED:
                bad.append((node.lineno, name))
            elif name == "resizable" and any(
                    isinstance(a, ast.Constant) and a.value is False
                    for a in node.args):
                bad.append((node.lineno, "resizable(False)"))
            for a in node.args:                    # attributes("-toolwindow", ...)
                if isinstance(a, ast.Constant) and isinstance(a.value, str) \
                        and "toolwindow" in a.value:
                    bad.append((node.lineno, "-toolwindow"))
        return bad

    found: list[str] = []
    for p in sorted((ROOT / "apps").glob("*.py")):
        try:
            bad = scan(p.read_text(encoding="utf-8"))
        except (OSError, SyntaxError) as exc:
            found.append(f"{p.name}:{type(exc).__name__}")
            continue
        for line, name in bad:
            found.append(f"{p.name}:{line} {name}")
    for item in found[:6]:
        print(f"    ! {item}")
    ck("静态：apps/*.py 里没有把窗口降成工具窗口 / 锁死尺寸的调用", not found)

    # 反证：把四种坏写法喂回去，同一条判据必须变红（否则是永远判绿的空闸门）。
    fake = ("def f(w, p):\n"
            "    w.transient(p)\n"
            "    w.resizable(False, False)\n"
            "    w.overrideredirect(True)\n"
            "    w.grab_set()\n"
            "    w.attributes('-toolwindow', True)\n")
    bad_fake = scan(fake)
    ck("反证：五种坏写法任何一种都能被静态判据抓到", len(bad_fake) >= 5)
    ck("反证：干净的窗口构造不会被误判",
       not scan("def f(w):\n    w.resizable(True, True)\n    w.lift()\n"))

    if sys.platform != "win32":
        print("  [跳过] 非 Windows：读不到 WM 样式位")
        print("         ⚠ 本节本次**只验证了静态一层**（跳过 ≠ 验证过）")
        return
    try:
        import tkinter as tk
        from apps._gui_app import App
    except (ImportError, tk.TclError) as exc:
        print(f"  [跳过] Tk 起不来（{type(exc).__name__}: {exc}）")
        print("         ⚠ 本节本次**只验证了静态一层**（跳过 ≠ 验证过）")
        return

    try:
        app = App()
        app.update_idletasks()
    except tk.TclError as exc:
        print(f"  [跳过] Tk 起不来（{type(exc).__name__}: {exc}）")
        return

    import ctypes
    from ctypes import wintypes
    u = ctypes.windll.user32
    u.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
    u.GetAncestor.restype = wintypes.HWND
    u.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
    u.GetWindowLongW.restype = ctypes.c_long
    WS_MIN, WS_MAX, WS_SYS = 0x00020000, 0x00010000, 0x00080000
    WS_EX_TOOL = 0x00000080

    targets: list[tuple[str, object]] = []
    try:
        targets = [
            ("App 根窗", app),
            ("门厅", app.hub),
            ("单目窗", app.open_mono()),
            ("平行双目窗", app.open_stereo("parallel")),
            ("大角度双目窗", app.open_stereo("angled")),
            ("会话工作台", app.open_bench()),
            ("控制台", app.open_console()),
        ]
        # 能无参/单参构造的产品对话框，一并量；构造不了的不假装量过。
        from apps.cameras_dialog import CamerasDialog
        from apps.profile_dialog import ProfileDialog
        from apps.tools_dialog import ToolsDialog
        from apps.help_window import HelpWindow
        from apps.junk_dialog import JunkDialog
        from apps.fov_dialog import FovDialog
        targets.append(("相机管理", CamerasDialog(app)))
        targets.append(("装置档案", ProfileDialog(app)))
        targets.append(("标定与训练工具台", ToolsDialog(app.hub, app)))
        targets.append(("操作说明", HelpWindow(app)))
        targets.append(("垃圾管理器", JunkDialog(app)))
        targets.append(("FOV 管理器", FovDialog(app.hub, app)))
        app.update()
    except Exception as exc:                            # noqa: BLE001
        print(f"  [注意] 部分窗口没构造出来（{type(exc).__name__}: {exc}），"
              "下面只对已开出来的窗口成立")

    unverified: list[str] = []
    for label, w in targets:
        try:
            w.update()
            root = u.GetAncestor(int(w.winfo_id()), 2)   # GA_ROOT
            style = u.GetWindowLongW(root, -16) & 0xffffffff
            ex = u.GetWindowLongW(root, -20) & 0xffffffff
            ok = (bool(style & WS_MIN) and bool(style & WS_MAX)
                  and bool(style & WS_SYS) and not bool(ex & WS_EX_TOOL)
                  and tuple(w.resizable()) == (True, True))
        except (tk.TclError, OSError, AttributeError) as exc:
            unverified.append(f"{label}（{type(exc).__name__}）")
            continue
        print(f"    {'✔' if ok else '✘'} {label}: MIN={bool(style & WS_MIN)} "
              f"MAX={bool(style & WS_MAX)} 关闭={bool(style & WS_SYS)} "
              f"工具窗={bool(ex & WS_EX_TOOL)}")
        ck(f"{label} 是标准窗口（最小化 / 最大化 / 关闭都在）", ok)
    if unverified:
        print(f"    ⚠ 没能量到：{'、'.join(unverified)}（跳过 ≠ 验证过）")
    # 这些窗口要么需要独立 Tk 根（同时开两个会抢）、要么会 wait_window 阻塞、
    # 要么要先有真会话目录 —— 运行时没实例化，只靠上面的静态 AST 覆盖。
    # 必须写出来，不许把"没量"伪装成"量过了"。
    print("    · 运行时未实例化（仅静态覆盖）：SeedPicker / MonoSeedPicker / "
          "CleanupDialog / TrashDialog / 改装置模式框 / 经典 Launcher / "
          "数据管理器 / 3D 查看器（独立 Tk 根或需真会话）")

    try:
        app.destroy()
    except tk.TclError:
        pass


def _draggable_panes(ck) -> None:
    """§12：GUI 内容分区必须可拖动（AGENTS.md §7.2）。

    用户截图里两块分区用固定 pack 拼着，内容一长就互相遮挡、也拉不动。
    两层判据：静态把"有内容分区"的窗口逐个钉死 PanedWindow；运行时真的
    改一次 sashpos 再读回，证明分隔条能拖。另外单独验"长错误只留摘要"，
    因为撑爆卡片同样会把相邻分区盖住。
    """
    print("----- 12. GUI 分区必须可以拖动（AGENTS.md §7.2）-----")
    import tkinter as tk
    from tkinter import ttk

    # 这张清单是"两个及以上内容区并排/上下"的窗口，人工核过：
    # 其余 side=left/right 都是「内容控件 + 滚动条/按钮栏」，不算分区。
    parted = ("data_manager.py", "help_window.py", "view_3d.py",
              "cameras_dialog.py")
    def parted_ok(files: dict) -> bool:
        return all("PanedWindow(" in src for src in files.values())
    files = {name: (ROOT / "apps" / name).read_text(encoding="utf-8")
             for name in parted}
    ck("静态：四个有内容分区的窗口都用 ttk.PanedWindow", parted_ok(files))
    broken = dict(files)
    broken["cameras_dialog.py"] = files["cameras_dialog.py"].replace(
        "ttk.PanedWindow(", "ttk.Frame(")
    ck("反证：相机管理换回普通 Frame → 判据会红", not parted_ok(broken))
    broken2 = dict(files)
    broken2["view_3d.py"] = files["view_3d.py"].replace(
        "ttk.PanedWindow(", "ttk.Frame(")
    ck("反证：3D 查看器换回普通 Frame → 判据会红", not parted_ok(broken2))

    dlg_src = files["cameras_dialog.py"]
    ck("长错误压成摘要、卡片状态行固定高度（不许撑爆相邻分区）",
       "def _brief(" in dlg_src and "pack_propagate(False)" in dlg_src
       and "BRIEF_LIMIT" in dlg_src)
    ck("反证：去掉摘要函数 → 判据会红",
       "def _brief(" not in dlg_src.replace("def _brief(", "def _no_brief(", 1))

    try:
        from apps._gui_app import App
        from apps.cameras_dialog import CamerasDialog, _PreviewPane, BRIEF_LIMIT
        app = App()
        app.update_idletasks()
    except tk.TclError as exc:
        print(f"  [跳过] Tk 起不来（{type(exc).__name__}: {exc}）")
        print("         ⚠ 本节本次**只验证了静态一层**（跳过 ≠ 验证过）")
        return

    try:
        dlg = CamerasDialog(app)
        app.update()
        paned = [w for w in _descendants(dlg)
                 if isinstance(w, ttk.PanedWindow)]
        ck("相机管理里确实有 PanedWindow 分区容器", bool(paned))
        if paned:
            p = paned[0]
            ck("分区数 >= 2", len(p.panes()) >= 2)
            try:
                p.sashpos(0, 300)
                app.update()
                pos = p.sashpos(0)
                ck(f"sash 真的能拖动（sashpos 改成 300 后读回 {pos}）",
                   abs(pos - 300) <= 40)
            except tk.TclError as exc:
                ck(f"sash 真的能拖动（{type(exc).__name__}: {exc}）", False)
        # 普通 Frame 没有 sashpos —— 用它反证"可拖"这一层不是空判据。
        plain = ttk.Frame(dlg)
        ck("反证：普通 ttk.Frame 没有可拖的 sash", not hasattr(plain, "sashpos"))
        plain.destroy()

        # 自动核对：设备改名后角色列必须变 ⚠，状态条必须写出“名字 旧 → 新”。
        import dataclasses
        from sphere_tracker import camera_registry as cr
        if not dlg._devices:
            # 首次枚举是后台线程；运行时判据不能靠“等它刚好跑完”。
            try:
                dlg._on_devices(cr.list_devices(), None)
                app.update()
            except Exception as exc:                     # noqa: BLE001
                print(f"    [提示] 现场枚举失败：{exc}")
        if dlg._devices:
            reg = cr.load_registry()
            spec = reg.roles.get("left")
            target = next((d for d in dlg._devices
                           if spec is not None and spec.identity
                           and d.identity
                           and spec.identity.upper() == d.identity.upper()),
                          None)
            if target is not None:
                original = list(dlg._devices)
                changed = [dataclasses.replace(d, name=d.name + " (renamed)")
                           if d.order == target.order else d
                           for d in original]
                dlg._on_devices(changed, cr.device_signature(changed))
                app.update()
                row = dlg.tree.item(str(target.order), "values")
                ck("设备改名后自动核对标 ⚠", "⚠" in str(row[-1]))
                ck("状态条报出名字变化",
                   "名字" in str(dlg.status.cget("text")))
                dlg._on_devices(original, cr.device_signature(original))
                dlg._on_devices([d for d in original
                                 if d.order != target.order], None)
                v = dlg.tree.item(str(target.order), "values") \
                    if dlg.tree.exists(str(target.order)) else ()
                ck("设备拔掉后自动刷新，旧行消失", not v)
            else:
                print("    [跳过] 当前没有与 left 身份匹配的设备（改名核对未验证）")
        else:
            print("    [跳过] 当前没有设备（改名核对未验证；跳过 ≠ 验证过）")

        # 塞一段超长错误进队列，卡片状态行必须只剩一行摘要。
        fake = _PreviewPane(dlg.grid_frame, 99, "fake", lambda: None)
        dlg._previews[99] = fake
        long_error = ("无法打开摄像头。可能原因：" * 30)
        dlg._events.put(("perror", 99, long_error))
        dlg._pump()
        shown = str(fake.status.cget("text"))
        print(f"    长错误 {len(long_error)} 字 -> 卡片显示 {len(shown)} 字")
        ck("超长错误不会撑高卡片（只留摘要）", len(shown) <= BRIEF_LIMIT + 2)
        ck("完整错误仍留得下来（双击可看全文）",
           dlg._full_errors.get(99) == long_error)
        # 自动刷新可能已经把这张假卡片当“设备不存在”停掉；重复 destroy 会抛。
        try:
            fake.frame.destroy()
        except tk.TclError:
            pass
        dlg._previews.pop(99, None)
        dlg._full_errors.pop(99, None)
        dlg._close()
    except Exception as exc:                             # noqa: BLE001
        ck(f"运行时分区检查没跑起来（{type(exc).__name__}: {exc}）", False)
    finally:
        try:
            app.destroy()
        except tk.TclError:
            pass


def _gui_test_discipline(ck) -> None:
    """§13：GUI 验证必须是真的（AGENTS.md §7.3）。

    守的是 2026-11-07 的教训：Treeview 默认没选中行，点「预览这台」什么都不发生，
    而前几轮验证全是 `dlg._preview_all()` 直接调方法 —— 按钮和它的前置条件一次
    都没走过，于是"反复拔插也没用"。

    两条硬判据：
      ① 走查报告里必须有**动作型**步骤「相机管理 · 点「预览这台」」，不能只有
         "观察"；走查工具本身必须带 `.invoke()`。
      ② 相机管理的按钮判据里必须真的 `invoke()`，不能只查源码字符串。
    两条都带反证。
    """
    import json

    print("----- 13. GUI 验证必须点真按钮（AGENTS.md §7.3）-----")

    report = ROOT / "outputs" / "gui_walk" / "verify" / "report.json"
    try:
        data = json.loads(report.read_text(encoding="utf-8"))
        steps = data.get("steps") or []
    except (OSError, ValueError) as exc:
        print(f"  [跳过] 读不到走查报告（{type(exc).__name__}: {exc}）")
        print("         ⚠ 本节本次没验证（跳过 ≠ 验证过）")
        steps = []
    titles = [str(s.get("title") or "") for s in steps]

    def has_action(ts) -> bool:
        return any("相机管理 · 点「预览这台」" in t for t in ts)

    if steps:
        ck("走查里有一个动作型步骤：相机管理 · 点「预览这台」", has_action(titles))
        ck("反证：把这一步退化成「观察」→ 判据会红",
           not has_action([t.replace("点「预览这台」", "观察")
                           for t in titles]))
        # 光有"观察"不算：必须有点击动作。
        observed_only = [t for t in titles if "相机管理" in t]
        ck("相机管理不只是「观察」了一把（有点击动作）",
           any("点「" in t for t in observed_only))

    walk_src = (ROOT / "apps" / "tools" / "gui_walkthrough.py").read_text(
        encoding="utf-8")
    ck("走查工具用 .invoke() 点按钮，而不是直接调窗口内部方法",
       ".invoke()" in walk_src and "相机管理 · 点「预览这台」" in walk_src)
    # 走查里有几十处 .invoke()，只去一处当然还剩 —— 必须全去掉判据才红。
    ck("反证：走查里去掉全部 .invoke() → 判据会红",
       ".invoke()" not in walk_src.replace(".invoke()", ""))

    cam_test = (ROOT / "tests" / "selftest_camera_registry.py").read_text(
        encoding="utf-8")
    invokes = cam_test.count(".invoke()")
    print(f"    相机管理判据里的 .invoke() 次数：{invokes}")
    ck("相机管理的按钮判据真的点按钮（.invoke() ≥ 5 次）", invokes >= 5)
    ck("反证：把 .invoke() 全部去掉 → 判据会红",
       cam_test.replace(".invoke()", "").count(".invoke()") < 5)


def _task_lifecycle_checks(ck) -> None:
    """窗口生命周期：按 X 关窗不许"复苏"，且必须能只停本窗的任务。

    用户原话："检查所有窗口文件，保证鼠标按下 x 这个关闭按钮之后不会自动复苏，
    可以中断任务"。两路只读审计确认：生产代码里没有"关窗后自动 open"的路径，
    真正的两个机制是 —— (a) close_window 的 pop 时序造成的结构性竞态；
    (b) 各窗的 _stop_my_tasks 原来是全局 stop_all，关一个窗杀掉别人的任务。
    这里把这两个机制 + owner 归属钉成判据，防止以后退化。

    前几条真的起子进程（用当前解释器），跑完一定 stop 干净；日志目录用临时目录，
    不往仓库里落任何文件。
    """
    import ast
    import tempfile

    from apps._task_runner import TaskRunner

    print("----- 14. 窗口生命周期：归属中断 + 关窗不复苏 -----")
    py = sys.executable
    sleep_code = "import time; time.sleep(60)"

    # Windows：TaskRunner 的日志 reader 线程可能比被杀的进程晚一步释放文件
    # 句柄，TemporaryDirectory 清理会偶发 WinError 145（目录不是空的）。
    # 上面的 stop 判据已经验完"进程真的停了"，这里只是清理测试夹具的目录，
    # 忽略清理错误不会放过任何行为缺陷。
    with tempfile.TemporaryDirectory(prefix="window_lifecycle_",
                                     ignore_cleanup_errors=True) as td:
        runner = TaskRunner(root=None, on_log=lambda _s: None,
                            log_dir=pathlib.Path(td))
        t_a = runner.start([py, "-c", sleep_code], "生命周期A", owner="win_a")
        t_b = runner.start([py, "-c", sleep_code], "生命周期B", owner="win_b")
        try:
            # 判据 1：start(owner=...) 直接写到 Task 上
            owners = {t.tid: t.owner for t in runner.tasks()}
            ck("start(owner='win_a') 后 task.owner == 'win_a'",
               owners.get(t_a.tid) == "win_a")
            ck("另一个任务的归属没被串改（win_b）",
               owners.get(t_b.tid) == "win_b")

            # 判据 3：空 owner 不许误停（此时两个任务都还在跑）
            ck("stop_by_owner('') 返回 0（空 owner 不许误停）",
               runner.stop_by_owner("") == 0)
            ck("空 owner 调用后两个任务都还在跑",
               t_a.running and t_b.running)

            # 判据 2（核心）：按归属只停 win_a，win_b 必须仍在跑。
            # 若退化成 stop_all，t_b 也会被停 → 这两条立刻红。
            n = runner.stop_by_owner("win_a")
            print(f"    stop_by_owner('win_a') 返回 {n}；"
                  f"win_a.running={t_a.running} win_b.running={t_b.running}")
            ck("stop_by_owner('win_a') 只确认停掉 1 个", n == 1)
            ck("win_a 停掉了", not t_a.running)
            ck("win_b 仍在运行（没退化成全局 stop_all）", t_b.running)
        finally:
            runner.stop_all(graceful=False)

    # 判据 4：close_window 的 pop 必须在 destroy 之后（AST 行号比较，不靠字符串顺序）
    app_src = (ROOT / "apps" / "_gui_app.py").read_text(encoding="utf-8")
    app_tree = ast.parse(app_src)
    closes = [n for n in ast.walk(app_tree)
              if isinstance(n, ast.FunctionDef) and n.name == "close_window"]
    ck("App 只有一个 close_window（判据定位唯一）", len(closes) == 1)
    if closes:
        pop_lines, destroy_lines = [], []
        for node in ast.walk(closes[0]):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            f = node.func
            if (isinstance(f.value, ast.Attribute) and f.value.attr == "_windows"
                    and f.attr == "pop"):
                pop_lines.append(node.lineno)
            if (isinstance(f.value, ast.Name) and f.value.id == "win"
                    and f.attr == "destroy"):
                destroy_lines.append(node.lineno)
        print(f"    close_window: pop@{pop_lines} destroy@{destroy_lines}")
        ck("close_window 里 self._windows.pop 在 win.destroy 之后",
           bool(pop_lines) and bool(destroy_lines)
           and max(pop_lines) > max(destroy_lines))

    # 判据 5：stereo 的 run_next 有 _closed 守卫（关窗后不再递归起下一段）
    st_src = (ROOT / "apps" / "_gui_stereo.py").read_text(encoding="utf-8")
    st_tree = ast.parse(st_src)
    rns = [n for n in ast.walk(st_tree)
           if isinstance(n, ast.FunctionDef) and n.name == "run_next"]
    ck("stereo 只有一个 run_next（判据定位唯一）", len(rns) == 1)
    if rns:
        guarded = any(
            isinstance(node, ast.If)
            and "_closed" in (ast.get_source_segment(st_src, node.test) or "")
            for node in ast.walk(rns[0]))
        ck("run_next 有 _closed 守卫（关窗后不再递归起下一段）", guarded)

    # 判据 6：close_app 弹窗同时说清"是/否"两种后果
    app_fns = [n for n in ast.walk(app_tree)
               if isinstance(n, ast.FunctionDef) and n.name == "close_app"]
    ck("App 只有一个 close_app（判据定位唯一）", len(app_fns) == 1)
    if app_fns:
        body_src = ast.get_source_segment(app_src, app_fns[0]) or ""
        ck("close_app 弹窗文案含「否」", "否" in body_src)
        ck("close_app 弹窗文案含「保持窗口打开」", "保持窗口打开" in body_src)
        ck("点否时留一行日志（退出被取消可查）", "_log_launch_context" in body_src)

    # ---- 第二轮遗留修复的判据 ----
    def _fn_calls(fn_node, attr: str) -> bool:
        """函数体内是否调了某个方法（走 AST，不做脆弱的行字符串匹配）。"""
        return any(isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Attribute)
                   and node.func.attr == attr
                   for node in ast.walk(fn_node))

    # 判据 7：三个窗口的 _stop_my_tasks 必须按归属停，不能退回全局 stop_all。
    # 全局停会误杀别的窗口在跑的离线重建 —— 按钮写着"停我的"，做的却是全局。
    for rel in ("apps/_gui_bench.py", "apps/_gui_mono.py", "apps/_gui_stereo.py"):
        src = (ROOT / rel).read_text(encoding="utf-8")
        tree = ast.parse(src)
        fns = [n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "_stop_my_tasks"]
        ck(f"{rel}: 只有一个 _stop_my_tasks（判据定位唯一）", len(fns) == 1)
        if fns:
            ck(f"{rel}: _stop_my_tasks 不含 stop_all（不误杀别的窗口任务）",
               not _fn_calls(fns[0], "stop_all"))
            ck(f"{rel}: _stop_my_tasks 调 stop_by_owner（按归属停）",
               _fn_calls(fns[0], "stop_by_owner"))

    # 判据 8：fov/tools 对话框按 X 也要停自己的任务。
    # 它们不经 App.open_window、没有 close_window 代劳，必须自己接
    # WM_DELETE_WINDOW → _on_close → stop_by_owner + destroy。
    for rel in ("apps/fov_dialog.py", "apps/tools_dialog.py"):
        src = (ROOT / rel).read_text(encoding="utf-8")
        tree = ast.parse(src)
        has_protocol = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "protocol"
            and any(isinstance(a, ast.Constant) and a.value == "WM_DELETE_WINDOW"
                    for a in node.args)
            for node in ast.walk(tree))
        ck(f"{rel}: 接了 WM_DELETE_WINDOW（按 X 有出口）", has_protocol)
        closers = [n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == "_on_close"]
        ck(f"{rel}: 只有一个 _on_close（判据定位唯一）", len(closers) == 1)
        if closers:
            ck(f"{rel}: _on_close 调 stop_by_owner",
               _fn_calls(closers[0], "stop_by_owner"))
            ck(f"{rel}: _on_close 调 destroy",
               _fn_calls(closers[0], "destroy"))


def _main_impl() -> int:
    print("=" * 70)
    print("窗口助手自测（全屏 / 尺寸 / 按键约定 / 各程序有没有接上）")
    print("=" * 70)
    results: list[tuple[str, bool]] = []

    def ck(label: str, good: bool) -> None:
        results.append((label, good))
        print(f"  [{'通过' if good else '不达标'}] {label}")

    # ---- 1. 那个坑：WINDOW_FULLSCREEN 的取值 ----
    print("----- 1. OpenCV 的 WINDOW_FULLSCREEN 是不是可用的常量 -----")
    if cv2 is None:
        print("  [跳过] 没有 cv2")
        print("\n判定结果：全部通过")
        return 0
    normal = int(cv2.WINDOW_NORMAL)
    autosize = int(cv2.WINDOW_AUTOSIZE)
    fullscreen = int(cv2.WINDOW_FULLSCREEN)
    print(f"  WINDOW_NORMAL={normal}  WINDOW_AUTOSIZE={autosize}  "
          f"WINDOW_FULLSCREEN={fullscreen}")
    same = (fullscreen == autosize)
    print(f"  FULLSCREEN 与 AUTOSIZE 同值：{same}")
    if same:
        print("    → 本构建里 `namedWindow(name, WINDOW_FULLSCREEN)` **得不到全屏**，")
        print("      必须走 setWindowProperty。实现按这个来（见 apps/_window.py）。")
        ck("已知 FULLSCREEN==AUTOSIZE（实现走 setWindowProperty 是**必需**的）", True)
    else:
        print("    → 本构建两者不同，WINDOW_FULLSCREEN 可用；"
              "实现可以简化，但当前写法仍然正确。")
        ck("FULLSCREEN 与 AUTOSIZE 不同（实现可简化，当前写法仍正确）", True)

    # ---- 2. 助手自己的按键约定 ----
    print("----- 2. 按键约定：f/F11 切全屏，Esc 在全屏时只退全屏 -----")
    from apps._window import WindowManager
    w = WindowManager()
    # 不建真窗口：直接查语义（建窗需要 GUI，无头环境会失败）
    ck("按 f 会被助手吃掉（返回 True）", w.handle_key("x", ord("f")) is True)
    ck("按 F11 会被助手吃掉", w.handle_key("x", 122) is True)
    ck("**不在全屏时**按 Esc 不被吃掉（交给程序自己退出）",
       w.handle_key("x", 27) is False)
    # 伪造"处于全屏"再按 Esc，应当被吃掉
    w._state("y").fullscreen = True
    ck("**在全屏时**按 Esc 被吃掉（只退全屏，不关程序）",
       w.handle_key("y", 27) is True)
    ck("Esc 处理过后全屏状态已清掉", w.is_fullscreen("y") is False)
    ck("无关的键不被吃掉（例如 q）", w.handle_key("x", ord("q")) is False)
    ck("提示语口径统一（含全屏说明）",
       "全屏" in w.fullscreen_hint() and "Esc" in w.fullscreen_hint())
    # ⚠️ `s` 是多个程序在用的键，助手**不许**吃它
    ck("助手不吃 `s`（多个程序用它存快照）", w.handle_key("x", ord("s")) is False)

    # ---- 3. 每个程序都接上了 ----
    print("----- 3. 所有会开视频窗口的程序都接上了助手 -----")
    for rel in APPS:
        p = ROOT / rel
        src = p.read_text(encoding="utf-8")
        built = "wins = WindowManager()" in src
        uses_show = "wins.show(" in src
        uses_key = "wins.handle_key(" in src
        # ⚠️ 2026-10-27：这里原来是逐字 `"from apps._window import WindowManager" in src`，
        # 于是只要多导一个名字（`from apps._window import DEFAULT_WINDOW_SIZE, WindowManager`）
        # 就会被判成"没导入" —— 断言耦合了**导入写法**，不是"有没有导入"。
        # 改成认"从 apps._window 导入了 WindowManager"，其余三条（建/show/键）照旧严格。
        import re as _re
        imports = bool(_re.search(
            r"from\s+apps\._window\s+import[^\n]*\bWindowManager\b", src))
        # `wins` 必须在**第一次使用之前**建好。
        # 第一版 `run_realtime` 把提示语写在建管理器之前，跑起来才炸
        # `UnboundLocalError` —— 静态看语法完全正常。
        lines = src.splitlines()
        dline = next((i for i, l in enumerate(lines) if "wins = WindowManager()" in l), None)
        early = [i for i, l in enumerate(lines)
                 if "wins." in l and "wins =" not in l
                 and (dline is None or i < dline)]
        good = built and uses_show and uses_key and imports and not early
        print(f"  {rel:42} 导入={imports} 建={built} show={uses_show} "
              f"键={uses_key} 先用后建={len(early)}")
        ck(f"{rel} 接上了窗口助手（且先建后用）", good)

    # ---- 4. 不许抢别人已用的键 ----
    print("----- 4. 全屏键不许抢程序里已用的键 -----")
    for rel, taken in F_KEY_TAKEN_BY_APP.items():
        src = (ROOT / rel).read_text(encoding="utf-8")
        # 这个程序里 `f` 有别的用途 -> 它必须限制 toggle_keys 不含 ord("f")
        limited = "toggle_keys=(122,)" in src
        print(f"  {rel}：`{taken}` 已被占用 -> 限制为 F11：{limited}")
        ck(f"{rel} 不抢 `{taken}`（限制为 F11）", limited)

    # ---- 5. 仍然直接 imshow 的地方（会绕开全屏）----
    print("----- 5. 不该再有绕开助手的 imshow -----")
    stragglers: list[str] = []
    for rel in APPS:
        for i, line in enumerate((ROOT / rel).read_text(encoding="utf-8").splitlines()):
            s = line.strip()
            if s.startswith("cv2.imshow(") and not s.startswith("#"):
                stragglers.append(f"{rel}:{i + 1}")
    print(f"  直接 cv2.imshow(...) 的位置：{stragglers or '无'}")
    ck("没有绕开助手的 cv2.imshow（否则那个窗口没全屏）", not stragglers)

    # ---- 7. 调用签名：ensure/show 必须给 image ----
    #
    # 为什么要按 AST 数参数（2026-10-29 真机）：`record_mono` 写的是
    # `wins.ensure(WINDOW)`，而签名是 `ensure(name, image)`。后果不是"窗口不好看"，
    # 是**倒计时结束、开录那一刻 TypeError**，10 秒录像白等一场。
    # `py_compile` 查不出（语法合法）、`gui_budget` 不扫它、而这个文件当时
    # 根本不在 APPS 里 —— 三条防线一条都没拦住。
    # 用 AST 而不是 grep：跨行调用、关键字参数都能正确计数。
    import ast
    print("----- 7. 窗口助手的调用签名（ensure / show 必须带 image）-----")
    bad_calls: list[str] = []
    for rel in APPS:
        tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("ensure", "show")
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "wins"):
                if len(node.args) != 2:
                    bad_calls.append(
                        f"{rel}:{node.lineno} 调 .{node.func.attr}() 只给了 "
                        f"{len(node.args)} 个位置参数")
    print(f"  参数个数不对的调用：{bad_calls or '无'}")
    ck("ensure / show 都带 image（否则真机跑到那一步才 TypeError）", not bad_calls)

    # ---- 6. 版式：画布尺寸必须跟着窗口走（窗口 / 全屏两种尺寸都验）----
    #
    # 为什么要钉这条（2026-10-27）：收图标定板的窗口原来把两路各缩到 720 高再
    # `hstack` → 一张**固定的 2560x720** 拼图。窗口是 AUTOSIZE（拖不动），
    # 全屏也无效；提示文字画在原始分辨率上再整体缩放，缩完只剩几个像素。
    # 现在的契约很硬：**请求多大，画布就多大**，于是全屏=屏幕、窗口=客户区，
    # 两种模式共用一条路径（`_canvas_size` 每帧从 `getWindowImageRect` 取）。
    print("----- 6. 版式：画布尺寸 = 请求尺寸（全屏与窗口同一路径）-----")
    from apps.tools import capture_board as cb
    import numpy as np

    spec = cb.BoardSpec(cols=8, rows=5, square_mm=25.0)
    frame = np.zeros((1080, 1920, 3), np.uint8)
    xs = np.linspace(300, 1500, 8)
    ys = np.linspace(300, 800, 5)
    corners = np.array([[x, y] for y in ys for x in xs], np.float32).reshape(-1, 1, 2)
    obs = cb.BoardObservation(found=True, corners=corners, focus=1200.0)
    obs_none = cb.BoardObservation(found=False)

    for label, size in (("窗口 1280x720", (1280, 720)),
                        ("全屏 1920x1080", (1920, 1080))):
        one = cb._compose(
            [{"title": "左", "frame": frame, "obs": obs, "ok": True,
              "reason": "", "spec": spec}], size, "footer", True)
        two = cb._compose(
            [{"title": "左", "frame": frame, "obs": obs, "ok": True,
              "reason": "", "spec": spec},
             {"title": "右", "frame": frame, "obs": obs_none, "ok": False,
              "reason": "未检出棋盘", "spec": spec}], size, "footer", False)
        ck(f"{label}：单路画布尺寸 = 请求尺寸",
           one.shape[:2] == (size[1], size[0]))
        ck(f"{label}：双路画布尺寸 = 请求尺寸（不会被撑成固定 2560 宽）",
           two.shape[:2] == (size[1], size[0]))

    # 角点必须真的画在**画布坐标系**里：有板与无板的两张图不能一样。
    # 若哪天又把文字/角点画回原图再整体缩放，这条会先红。
    with_pts = cb._compose(
        [{"title": "左", "frame": frame, "obs": obs, "ok": True,
          "reason": "", "spec": spec}], (1280, 720), "footer", True)
    without = cb._compose(
        [{"title": "左", "frame": frame, "obs": obs_none, "ok": True,
          "reason": "", "spec": spec}], (1280, 720), "footer", True)
    ck("角点/引导框真的画到了画布上（有板与无板不同）",
       bool((with_pts != without).any()))

    # 窗口还没建好时 `_canvas_size` 不许抛，也不许返回 0
    w0, h0 = cb._canvas_size("__no_such_window__")
    ck(f"窗口未建时画布尺寸有回退值（{w0}x{h0}，>0）", w0 > 0 and h0 > 0)

    # ---- 8. Tk 界面窗口：每一步都要在视口里（2026-10-31 增） ----
    #
    # 守的坑：双目窗写死 `geometry("920x800")`，而四步内容实测 881/950 ——
    # 第 ④ 步的按钮被裁在窗口下沿，用户看到的是"这个界面只有三步"。
    # 现在高度改为**按内容算**（`apps/_gui_geometry.fit_height`，上限=屏幕可用
    # 高度），所以这条判据必须成立；谁再往卡片里加一行而不重新量高度，这里红。
    #
    # 判据按"步"而不是按"整页"：报错要能直接定位是哪一步、差多少 px。
    # 每一步的 reqheight = 这一块**下沿**在内容坐标里的 y（含它之前的全部内容），
    # 实际高度 = 滚动页视口的高度 —— 两者一比，就是"这一步要不要滚才看得见"。
    _tk_first_screen_checks(ck)

    # ---- 9. 与真人走查交叉验证（同一件事，两个独立工具必须说同一句话） ----
    #
    # 为什么不能只信自己：走查工具是"像用户那样点一遍"，本测试是"量控件"。
    # 两边对同一扇窗给出不同结论，说明其中一边的判据写错了 —— 那比两边都红更危险。
    _cross_check_walkthrough(ck)

    # ---- 10. 七条债务的可复现判据（含反向验证）----
    print("----- 10. 债务护栏的静态判据（修复前会红 / 修复后绿）-----")
    _debt_guards(ck)

    # ---- 11. 每个 GUI 窗口的标准标题栏三按钮（AGENTS.md §7.1）----
    _standard_window_chrome(ck)

    # ---- 12. GUI 内容分区必须可拖动（AGENTS.md §7.2）----
    _draggable_panes(ck)

    # ---- 13. GUI 验证必须点真按钮（AGENTS.md §7.3）----
    _gui_test_discipline(ck)

    # ---- 14. 窗口生命周期：关窗不复苏 + 按归属中断任务 ----
    _task_lifecycle_checks(ck)

    print()
    ok = all(v for _n, v in results)
    bad = [n for n, v in results if not v]
    if bad:
        print(f"不达标 {len(bad)} 项：")
        for n in bad:
            print(f"  · {n}")
    print("\n判定结果：" + ("全部通过" if ok else "有不达标项"))
    return 0 if ok else 1


def main() -> int:
    """把跨进程账本指到临时目录再跑主体（try/finally 覆盖早退与异常）。

    本测试会真起 TaskRunner 的任务（第 14 节生命周期），而
    ``register_start/end`` 写的是 ``apps._task_registry`` 的模块全局
    ``REGISTRY_PATH``。不注入的话，每次自测都在真实
    ``logs/tasks/_registry.jsonl`` 里留下"生命周期A/B"夹具记录 —— 测试
    数据混进运行账本。

    两条注入缺一不可：**环境变量**给子进程（新解释器 import 时读到
    ``SPHERE_TASK_REGISTRY``），**模块常量**给已 import 的本进程调用方。
    临时目录随 with 退出自动清理，两者都在 finally 里还原。
    """
    import os
    import tempfile

    import apps._task_registry as _reg

    with tempfile.TemporaryDirectory(prefix="window_mode_registry_") as td:
        path = pathlib.Path(td) / "_registry.jsonl"
        real = _reg.REGISTRY_PATH
        old_env = os.environ.get("SPHERE_TASK_REGISTRY")
        _reg.REGISTRY_PATH = path
        os.environ["SPHERE_TASK_REGISTRY"] = str(path)
        try:
            return _main_impl()
        finally:
            _reg.REGISTRY_PATH = real
            if old_env is None:
                os.environ.pop("SPHERE_TASK_REGISTRY", None)
            else:
                os.environ["SPHERE_TASK_REGISTRY"] = old_env
            n = 0
            if path.exists():
                try:
                    n = len(path.read_text(encoding="utf-8",
                                           errors="replace").splitlines())
                except OSError:
                    pass
            print(f"[账本隔离] 临时账本 {path.name} 共 {n} 行（真实 logs 未写入）")


if __name__ == "__main__":
    raise SystemExit(main())
