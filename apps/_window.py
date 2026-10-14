"""窗口助手：全屏 / 尺寸 / 退出全屏的键盘约定，**一处实现、所有程序共用**。

## 为什么要单独一个模块（2026-10-14）

项目的视频窗口原来各写各的，于是各有各的毛病：

- `run_realtime` / `run_video` / `calibrate_stereo` / `check_stereo_setup`
  **根本没建窗口** —— `cv2.imshow` 会自动建一个 `WINDOW_AUTOSIZE`
  （= 窗口和图片一样大、且**用户不能拖动改大小**）。显示宽度改成 960 之后，
  窗口就只有 960x540，在高分屏上小得没法看。
- `calibrate_color` / `calibrate_focal` / `cameras` 建了 `WINDOW_NORMAL`，
  但也**没有全屏**。

## 一个必须知道的坑：本构建里 `WINDOW_FULLSCREEN` 不可用

实测这台机器（OpenCV 5.0）：

    cv2.WINDOW_NORMAL     = 0
    cv2.WINDOW_AUTOSIZE   = 1
    cv2.WINDOW_FULLSCREEN = 1     ← 和 AUTOSIZE **同一个值**！

所以 `cv2.namedWindow(name, cv2.WINDOW_FULLSCREEN)` **不会**得到全屏窗口，
它等价于 AUTOSIZE。全屏**只能**走属性：

    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, 1)   # 实测真的全屏
    cv2.getWindowProperty(name, cv2.WND_PROP_FULLSCREEN)      # 读到 1.0
    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, 0)   # 退得回来

（实测图像区从 304x281 变成 1600x1000，再退回去。）

## 键盘约定（全项目统一）

| 键 | 作用 |
| -- | ---- |
| `f` / `F` / F11 | 切换全屏 |
| `Esc` | **在全屏时=退出全屏**；不在全屏时才由各程序自己处理 |

`Esc` 的这条优先级是刻意的：全屏之后用户第一反应就是按 Esc 退出来，
而各程序原来把 Esc 当"退出程序"，直接按下去会把正在跑的任务关掉。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2

# 进入全屏前记住的窗口尺寸（退出时还原）。给不出就用这个保守值。
DEFAULT_WINDOW_SIZE = (1280, 720)


@dataclass
class _Win:
    fullscreen: bool = False
    windowed_size: tuple[int, int] = DEFAULT_WINDOW_SIZE


@dataclass
class WindowManager:
    """管理一组 OpenCV 窗口的全屏状态。

    用法::

        wins = WindowManager()
        wins.show("run_video", vis)          # 建窗口 + imshow
        key = cv2.waitKey(1) & 0xFF
        if wins.handle_key("run_video", key):   # 该键已被全屏逻辑吃掉
            continue
        if key == 27: break                     # 只有非全屏时才会走到这里

    **每个窗口各记各的状态**：同一个程序可能同时开好几个窗口
    （标定就同时有"拼版窗口"和"覆盖率窗口"），全屏其中一个不该影响另一个。
    """

    _wins: dict[str, _Win] = field(default_factory=dict)
    # 不主动改尺寸的窗口名。
    #
    # ⚠️ `calibrate_color` 的窗口上**挂着一排 trackbar**，程序化 `resizeWindow`
    # 会把滑条布局打乱（滑条是跟着窗口走的）。它的全屏照常可用
    # （改的是属性，不动布局），只是不主动 resize。
    _no_resize: set[str] = field(default_factory=set)

    def keep_size(self, name: str) -> None:
        """声明"这个窗口不要程序化改尺寸"（挂 trackbar 的窗口用）。"""
        self._no_resize.add(name)

    # ------------------------------------------------------------------
    def _state(self, name: str) -> _Win:
        return self._wins.setdefault(name, _Win())

    def ensure(self, name: str, image) -> None:
        """按需建窗口。

        **为什么用 `WINDOW_NORMAL`**：只有 NORMAL 窗口才能被
        `resizeWindow` / 全屏属性控制。默认的 AUTOSIZE 窗口
        **尺寸等于图片**，用户既改不了大小、也控制不了全屏。
        """
        st = self._state(name)
        if st.windowed_size != DEFAULT_WINDOW_SIZE:
            return                                   # 已经建过且量过尺寸了
        try:
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        except cv2.error:
            return                                   # 无 GUI 环境：交给 imshow 自己处理
        if name in self._no_resize:
            st.windowed_size = (-1, -1)              # 标记"已建、但不接管尺寸"
            return
        # 首次：给一个"看得清"的初始尺寸，而不是跟着图片大小（960 图太小。
        # 实测 960x540 的图在 3200x2000 屏上只有中间一小块，用户第一反应就是
        # "窗口太小了" —— 而 WINDOW_AUTOSIZE 窗口**用户还拖不大**）。
        h, w = (image.shape[0], image.shape[1]) if image is not None else (720, 1280)
        try:
            sh, sw = self.screen_size()
            # 初始窗口取"图片的 1.5 倍"与"屏幕的 9 成"里较小的那个 ——
            # 图小的时候放大到看得清，图大的时候别超出屏幕。
            want_w = max(320, min(int(w * 1.5), int(sw * 0.9)))
            want_h = max(240, min(int(h * 1.5), int(sh * 0.9)))
            cv2.resizeWindow(name, want_w, want_h)
            st.windowed_size = (want_w, want_h)
        except Exception:                            # noqa: BLE001
            pass

    @staticmethod
    def screen_size() -> tuple[int, int]:
        """主屏 (高, 宽)。拿不到就给 1080p —— 宁可用保守值也不要崩。"""
        try:
            import ctypes

            user32 = ctypes.windll.user32            # type: ignore[attr-defined]
            try:
                user32.SetProcessDPIAware()
            except Exception:                        # noqa: BLE001
                pass
            return int(user32.GetSystemMetrics(1)), int(user32.GetSystemMetrics(0))
        except Exception:                            # noqa: BLE001
            return 1080, 1920

    # ------------------------------------------------------------------
    def show(self, name: str, image) -> None:
        """建窗口（首次）并显示。**所有 imshow 都应当走这里**。"""
        self.ensure(name, image)
        cv2.imshow(name, image)

    def toggle_fullscreen(self, name: str, image=None) -> bool:
        """切换全屏，返回**切换后**是否处于全屏。"""
        st = self._state(name)
        if not st.fullscreen:
            try:
                # 记住当前窗口尺寸，退出全屏时还原回这个样子
                rect = cv2.getWindowImageRect(name)
                if rect and rect[2] > 0 and rect[3] > 0:
                    st.windowed_size = (int(rect[2]), int(rect[3]))
            except Exception:                        # noqa: BLE001
                pass
        st.fullscreen = not st.fullscreen
        self._apply(name, image)
        return st.fullscreen

    def exit_fullscreen(self, name: str, image=None) -> bool:
        """退出全屏。返回**是否原本就在全屏**（调用方据此决定要不要继续处理该键）。"""
        st = self._state(name)
        if not st.fullscreen:
            return False
        st.fullscreen = False
        self._apply(name, image)
        return True

    def _apply(self, name: str, image=None) -> None:
        st = self._state(name)
        try:
            # ⚠️ 只能走属性，不能用 WINDOW_FULLSCREEN 建窗口（见模块文档）
            cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN,
                                  1 if st.fullscreen else 0)
        except cv2.error:
            return
        if not st.fullscreen:
            try:
                if st.windowed_size[0] > 0:          # (-1,-1) = 这个窗口不接管尺寸
                    cv2.resizeWindow(name, *st.windowed_size)
            except cv2.error:
                pass
        if image is not None:
            try:
                cv2.imshow(name, image)
            except cv2.error:
                pass

    # ------------------------------------------------------------------
    def handle_key(self, name: str, key: int, image=None,
                   toggle_keys: tuple[int, ...] = (ord("f"), ord("F"), 122)) -> bool:
        """处理全屏相关的按键。**返回 True 表示这个键已被吃掉**，调用方不要再处理。

        | 键 | 行为 |
        | -- | ---- |
        | `f` / `F` / F11 | 切换全屏（可用 `toggle_keys` 改） |
        | `Esc` | 在全屏时退出全屏（否则返回 False，交给程序自己） |

        ⚠️ **`toggle_keys` 不是可有可无的**：`calibrate_color` 里 `f` 已经是
        "冻结画面"（`frozen`），全屏若抢走它就会把一个既有功能弄坏。
        那种程序传 `toggle_keys=(122,)` 只留 F11。

        这类"快捷键撞车"必须**逐个程序核对**，不能假定 `f` 一定空着 ——
        抢掉别人的键比没有全屏更糟。
        """
        if key in toggle_keys:
            self.toggle_fullscreen(name, image)
            return True
        if key == 27:                                # Esc
            return self.exit_fullscreen(name, image)
        return False

    def is_fullscreen(self, name: str) -> bool:
        return self._state(name).fullscreen

    def fullscreen_hint(self) -> str:
        """拼到各程序的按键提示里，保证口径一致。"""
        return "f 全屏 / Esc 退出全屏"


__all__ = ["DEFAULT_WINDOW_SIZE", "WindowManager"]
