"""启动器窗口几何：窗口多大、放哪、文字多宽 —— **纯计算，不碰 Tk**。

## 为什么单独一个模块

`apps/gui_main.py` 原来写死 ``geometry("980x780")`` + ``minsize(880, 560)``：
在 1024x768 / 1366x768 这类屏幕上，窗口一开就可能比可用区域还大，
右侧控件被切到屏幕外（计划书 §1 有实测记录）。

把"算尺寸"从"建窗口"里拆出来有三个好处：

- 不 import tkinter 也能跑（本文件直接 ``python apps/_gui_geometry.py`` 自测），
  无显示器 / 远程会话里同样能验证；
- 规则只有这一处，主程序只负责把算出的四个数交给 ``geometry()``，
  不会出现"启动器里一套、别处又一套"；
- 窗口尺寸、位置、换行宽度是纯函数，边界（小屏、零屏、恰好卡线）可穷举。

## 口径（调用方必须知道的两件事）

1. **``screen_w`` / ``screen_h`` 是"可用屏幕"，由调用方给出。**
   本模块不查系统、**不扣任务栏**；要避开任务栏，就在调用方传工作区尺寸。
2. **``margin`` 是横/纵方向各自总共预留的像素，不是每边。**
   例：``margin=(80, 100)`` 表示宽度最多比屏幕窄 80、高度最多窄 100；
   居中之后左右各让出 40，不是各让出 80。

屏幕为 0 或负数（Tk 尚未初始化、远程会话异常等）时返回固定兜底
``320x240 @ (0, 0)``：宁可窗口小，也不要算出负坐标或零尺寸。

`wrap_width()` 是同一套思路用在文字上：``wraplength`` 必须跟着视口走，
写死一定会和实际窗口对不上（`apps/gui_main.py` 里 820 被截断过一次）。

`fit_height()` 是同一套思路用在**高度**上：模式窗的四个步骤加起来多高，
由控件的 ``winfo_reqheight`` 说了算，屏幕可用高度只做封顶 —— 不再写死
``geometry("920x800")``（2026-10-31 实测内容 881/950，第 ④ 步被裁）。
"""

_FALLBACK_WINDOW: tuple[int, int] = (320, 240)

#: 高度方向上，客户区之外必须让出的像素（标题栏 + 边框 + 一点呼吸）。
#: 它是**总共**让出的量，不是每边各让一次（与 ``margin`` 同一口径）。
WINDOW_CHROME_H = 48

__all__ = ["fit_window", "wrap_width", "fit_height", "WINDOW_CHROME_H"]


def _axis(screen: int, desired: int, margin: int, minimum: int) -> int:
    """单轴尺寸：先被 ``screen - margin`` 收住，再抬到 ``minimum``，最后屏幕封顶。

    顺序不能反：先抬到 ``minimum`` 再拿 ``screen - margin`` 封顶，会在
    "屏幕只比最小尺寸大一点点"时把窗口压到最小尺寸以下；而最后这道
    ``min(screen, ...)`` 保证屏幕本身小于最小尺寸时退化为屏幕尺寸。
    """
    extent = min(desired, screen - margin)
    extent = max(extent, minimum)
    return min(screen, extent)


def fit_window(screen_w: int, screen_h: int, *,
               desired: tuple[int, int] = (1180, 820),
               margin: tuple[int, int] = (80, 100),
               min_size: tuple[int, int] = (900, 600)) -> tuple[int, int, int, int]:
    """按可用屏幕算出启动器窗口的 ``(宽, 高, x, y)``，居中且不越界。

    规则（按优先级）：

    1. 理想尺寸 ``desired`` 被 ``screen - margin`` 封顶；
    2. 封顶后的结果不得低于 ``min_size`` —— **除非屏幕本身比它小**，
       那时取屏幕尺寸（否则窗口必然出界）；
    3. 居中：``x = max(0, (screen_w - width) // 2)``，``y`` 同理。
       奇数差时少 1 px 落在左侧/上侧，``x + width <= screen_w`` 永远成立；
    4. 屏幕任一维 ``<= 0`` 视为无效，返回兜底 ``(320, 240, 0, 0)``。

    参数：
        screen_w, screen_h: 可用屏幕像素，由调用方给（不要减任务栏，除非
            传进来的就是工作区）。
        desired: 理想窗口尺寸，默认 1180x820。
        margin: 横/纵各自**总共**预留的像素，默认 80x100。
        min_size: 最小窗口尺寸，默认 900x600。

    返回：
        四个 ``int``：``(width, height, x, y)``。屏幕有效时保证
        ``0 <= x``、``0 <= y``、``width <= screen_w``、``height <= screen_h``。
        屏幕无效时返回固定兜底，坐标恒为 ``(0, 0)``。
    """
    sw = int(screen_w)
    sh = int(screen_h)
    if sw <= 0 or sh <= 0:
        # 无效屏幕：不猜、不除零、不产生负坐标
        return _FALLBACK_WINDOW[0], _FALLBACK_WINDOW[1], 0, 0

    desired_w, desired_h = (int(v) for v in desired)
    margin_w, margin_h = (int(v) for v in margin)
    min_w, min_h = (int(v) for v in min_size)

    width = _axis(sw, desired_w, margin_w, min_w)
    height = _axis(sh, desired_h, margin_h, min_h)
    x = max(0, (sw - width) // 2)
    y = max(0, (sh - height) // 2)
    return int(width), int(height), int(x), int(y)


def fit_height(content: int, *, screen_h: int, status: int = 0,
               chrome: int = WINDOW_CHROME_H,
               minimum: int = 480) -> int:
    """按**内容**算窗口客户区高度：``content + status``，上限是可用屏幕减 ``chrome``。

    为什么需要它（2026-10-31，缺陷 #9/#10）
    --------------------------------------
    双目窗原来写死 ``geometry("920x800")``。四步内容实测 881（平行）/ 950
    （大角度）—— 比 800 高，于是第 ④ 步的按钮被裁在窗口下沿。写死的高度
    只是"上次放得下"，内容一长就骗人；而"再把数字调大"只是把同一件事往后推：
    屏幕矮的机器上照样裁。

    所以高度只有两个来源：**内容**与**屏幕**。取小者：

    1. 内容 + 状态条 = 需要的高度（一次装全，不用滚）；
    2. ``screen_h - chrome`` = 允许的高度（别把窗口顶出屏幕/压住任务栏）。

    屏幕无效（``<= 0``，远程会话或 Tk 尚未初始化）时返回**内容**需要的高度，
    而不是退回一个拍脑袋的常数 —— 量不出屏幕时，"按内容"仍是唯一有依据的答案。

    ``status`` 是底部状态条自身的高度：它在内容区之外，但同样要占客户区。
    ``minimum`` 只兜住"内容极短时窗口太扁"，不会把高度抬到 ``screen_h - chrome``
    之上（屏幕比它更小时以屏幕为准，与 ``fit_window`` 同一顺序）。
    """
    need = int(content) + int(status)
    sh = int(screen_h)
    if sh <= 0:
        return max(int(minimum), need)
    cap = sh - int(chrome)
    height = min(need, cap)
    if height < int(minimum):
        height = min(int(minimum), sh)          # 屏幕本身比 minimum 还小时取屏幕
    return int(height)


def wrap_width(viewport: int, *, chrome: int = 44, minimum: int = 240) -> int:
    """按视口宽度算 ``wraplength``：``viewport - chrome``，但不低于 ``minimum``。

    ``chrome`` 是横向被非文字部分吃掉的像素（卡片左右外边距、内边距、
    滚动条等），由调用方按控件结构给；它**不是**每边各扣一次。

    视口 ``<= 0``（首帧还没量出来）时直接返回 ``minimum``：
    这时任何"按比例算"都没有意义，给一个能换行的安全值即可。
    """
    vp = int(viewport)
    floor = int(minimum)
    if vp <= 0:
        return floor
    return max(floor, vp - int(chrome))


# --------------------------------------------------------------------------- #
# 自测：不依赖 Tk / pytest，直接跑本文件即可
# --------------------------------------------------------------------------- #
def _check_fit(label: str, screen: tuple[int, int],
               expected: tuple[int, int, int, int],
               **kwargs) -> None:
    """校验一组 ``fit_window``：返回值、类型、越界、居中、最小尺寸。"""
    sw, sh = screen
    w, h, x, y = fit_window(sw, sh, **kwargs)

    assert all(type(v) is int for v in (w, h, x, y)), \
        f"{label}: 返回值必须是 int，实际 {(w, h, x, y)!r}"
    assert (w, h, x, y) == expected, \
        f"{label}: 期望 {expected}，实际 {(w, h, x, y)}"

    if sw <= 0 or sh <= 0:
        # 无效屏幕：固定兜底，且不能出现负坐标
        assert (w, h, x, y) == (320, 240, 0, 0), f"{label}: 无效屏幕兜底错误 {(w, h, x, y)}"
        print(f"PASS fit_window {label}: 兜底 {w}x{h} at ({x}, {y})")
        return

    min_w, min_h = kwargs.get("min_size", (900, 600))
    min_w, min_h = int(min_w), int(min_h)

    # 不越界：宽高为正、四个方向都在屏幕内
    assert w > 0 and h > 0, f"{label}: 窗口尺寸必须为正，实际 {w}x{h}"
    assert w <= sw and h <= sh, f"{label}: 窗口 {w}x{h} 超出屏幕 {sw}x{sh}"
    assert 0 <= x and 0 <= y, f"{label}: 坐标不能为负，实际 ({x}, {y})"
    assert x + w <= sw and y + h <= sh, \
        f"{label}: 右/下边界越界 {x}+{w}<={sw}, {y}+{h}<={sh}"

    # 居中：屏幕有效时 width/height 已不越界，max(0, ...) 不应再改结果
    assert x == max(0, (sw - w) // 2), f"{label}: x 未居中 {(x, sw, w)}"
    assert y == max(0, (sh - h) // 2), f"{label}: y 未居中 {(y, sh, h)}"

    # 最小尺寸：屏幕放得下就必须不低于 min；屏幕更小则正好取屏幕
    if sw >= min_w:
        assert w >= min_w, f"{label}: 宽度 {w} 低于最小宽度 {min_w}"
    else:
        assert w == sw, f"{label}: 屏幕宽 {sw} < 最小宽 {min_w}，应取屏幕宽"
    if sh >= min_h:
        assert h >= min_h, f"{label}: 高度 {h} 低于最小高度 {min_h}"
    else:
        assert h == sh, f"{label}: 屏幕高 {sh} < 最小高 {min_h}，应取屏幕高"

    print(f"PASS fit_window {label}: {w}x{h} at ({x}, {y})")


def _check_height(label: str, content: int, *, expected: int, **kwargs) -> None:
    """校验 ``fit_height``：值、类型、不越屏、够不够装。"""
    got = fit_height(content, **kwargs)
    assert type(got) is int, f"{label}: 返回值必须是 int，实际 {got!r}"
    assert got == expected, f"{label}: 期望 {expected}，实际 {got}"

    sh = int(kwargs.get("screen_h", 0))
    status = int(kwargs.get("status", 0))
    chrome = int(kwargs.get("chrome", WINDOW_CHROME_H))
    minimum = int(kwargs.get("minimum", 480))
    if sh > 0:
        assert got <= max(sh, minimum), f"{label}: 高度 {got} 超出屏幕 {sh}"
        # 装得下就必须真的装得下：内容 + 状态条一格不少
        if content + status <= sh - chrome:
            assert got >= content + status, \
                f"{label}: 内容需 {content + status}，却只给了 {got}"
    print(f"PASS fit_height {label}: 内容 {content}"
          f"{f' + 状态条 {status}' if status else ''} -> {got}")


def _check_wrap(label: str, viewport: int,
                expected: int, **kwargs) -> None:
    """校验一组 ``wrap_width``：值、类型、下限语义。"""
    got = wrap_width(viewport, **kwargs)
    assert type(got) is int, f"{label}: 返回值必须是 int，实际 {got!r}"
    assert got == expected, f"{label}: 期望 {expected}，实际 {got}"

    floor = kwargs.get("minimum", 240)
    if viewport <= 0:
        assert got == floor, f"{label}: 非正视口应返回 minimum={floor}，实际 {got}"
    else:
        assert got >= floor, f"{label}: 换行宽度 {got} 低于 minimum={floor}"
    print(f"PASS wrap_width {label}: viewport={viewport} -> {got}")


def _selftest() -> int:
    # ---- fit_window：常见屏幕 ----
    _check_fit("1024x768", (1024, 768), (944, 668, 40, 50))
    _check_fit("1366x768", (1366, 768), (1180, 668, 93, 50))
    _check_fit("1920x1080", (1920, 1080), (1180, 820, 370, 130))
    _check_fit("800x600", (800, 600), (800, 600, 0, 0))

    # ---- 恰好卡线 / 边界 ----
    _check_fit("恰好=desired+margin 1260x920", (1260, 920), (1180, 820, 40, 50))
    _check_fit("恰好=desired 1180x820", (1180, 820), (1100, 720, 40, 50))
    _check_fit("恰好=min 900x600", (900, 600), (900, 600, 0, 0))
    _check_fit("比 min 少 1px 899x599", (899, 599), (899, 599, 0, 0))
    _check_fit("介于 min 与 margin 950x650", (950, 650), (900, 600, 25, 25))
    _check_fit("单轴小于 min 800x1080", (800, 1080), (800, 820, 0, 130))
    _check_fit("另一轴小于 min 1920x600", (1920, 600), (1180, 600, 370, 0))

    # ---- 参数覆盖 ----
    _check_fit("desired 小于 min", (1920, 1080), (900, 600, 510, 240),
               desired=(500, 400))
    _check_fit("margin=0", (1000, 700), (1000, 700, 0, 0), margin=(0, 0))
    _check_fit("min 突破 margin 但仍不越界", (1024, 768), (1000, 700, 12, 34),
               min_size=(1000, 700))

    # ---- 无效屏幕 ----
    _check_fit("负宽 -1x1080", (-1, 1080), (320, 240, 0, 0))
    _check_fit("零高 1920x0", (1920, 0), (320, 240, 0, 0))
    _check_fit("零屏幕 0x0", (0, 0), (320, 240, 0, 0))
    _check_fit("双负 -800x-600", (-800, -600), (320, 240, 0, 0))

    # ---- fit_height：按内容算，受屏幕封顶 ----
    # 本机屏幕逻辑高 1000 → 可用工作区 960（任务栏 40）；封顶 = 960 - 48 = 912。
    _check_height("平行双目 内容 846 + 状态条 35（装得下）", 846,
                  screen_h=960, status=35, expected=881)
    _check_height("大角度双目 合并第④步按钮行后 868 + 状态条 35", 868,
                  screen_h=960, status=35, expected=903)
    _check_height("大角度双目 旧布局 915 超上限 → 封顶（剩下交给滚动）", 915,
                  screen_h=960, status=35, expected=912)
    _check_height("小屏（768）封顶，剩下交给滚动", 915,
                  screen_h=768, status=35, expected=720)
    _check_height("内容很短：不缩到 minimum 以下", 200, screen_h=960, status=35,
                  expected=480)
    _check_height("屏幕比 minimum 还矮：取屏幕", 200, screen_h=400, status=35,
                  expected=400)
    _check_height("屏幕无效（0）：按内容，不拍数字", 846, screen_h=0, status=35,
                  expected=881)
    _check_height("chrome 可覆盖", 846, screen_h=960, status=35, chrome=0,
                  expected=881)

    # ---- wrap_width ----
    _check_wrap("零视口", 0, 240)
    _check_wrap("负视口", -100, 240)
    _check_wrap("极小视口", 100, 240)
    _check_wrap("chrome 吃掉全部", 300, 240, chrome=500)
    _check_wrap("恰好=minimum", 284, 240)              # 284 - 44 = 240
    _check_wrap("minimum 之上 1px", 285, 241)
    _check_wrap("1000 宽", 1000, 956)
    _check_wrap("1920 宽", 1920, 1876)
    _check_wrap("启动器 CHROME_PX=64", 1000, 936, chrome=64)
    _check_wrap("minimum 覆盖结果", 500, 600, minimum=600)
    _check_wrap("非正视口用自定义 minimum", -5, 300, minimum=300)
    _check_wrap("chrome=0", 300, 300, chrome=0)

    print("PASS 全部自测通过（fit_window 18 组 / fit_height 8 组 / wrap_width 12 组）")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
