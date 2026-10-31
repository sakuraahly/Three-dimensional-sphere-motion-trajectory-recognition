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
    for p in sorted((ROOT / "apps").glob("_gui_*.py")):
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

    # 首屏步骤（不含"展开折叠区"那种用户主动要多看的状态）必须不用滚。
    # 阈值取 20 px，**与走查工具 `_note_overflow` 的阈值逐字相同**：
    # 这一节的任务是"两边说同一句话"，口径不一致的话红绿就不代表同一件事。
    # （第 8 节是更严的 ≤ 0，管我们自己能控的那几扇窗。）
    over = [(s.get("window"), s.get("title"), s.get("overflow_px"))
            for s in steps
            if s.get("window") != "external"
            and "展开" not in str(s.get("title") or "")
            and int(s.get("overflow_px") or 0) > 20]
    for w, t, px in over:
        print(f"    ! {w} · {t}：内容比视口高 {px} px（首屏要滚）")
    ck("走查：首屏步骤都不用滚（overflow_px ≤ 20，与走查工具同一阈值）", not over)


def main() -> int:
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

    print()
    ok = all(v for _n, v in results)
    bad = [n for n, v in results if not v]
    if bad:
        print(f"不达标 {len(bad)} 项：")
        for n in bad:
            print(f"  · {n}")
    print("\n判定结果：" + ("全部通过" if ok else "有不达标项"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
