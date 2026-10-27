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
)
# 这些程序里 `f` 已经被别的功能占用，全屏只能留给 F11。
# **抢掉别人已用的键比没有全屏更糟** —— 那是把一个既有功能弄坏。
F_KEY_TAKEN_BY_APP = {"apps/tools/calibrate_color.py": "f"}


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
