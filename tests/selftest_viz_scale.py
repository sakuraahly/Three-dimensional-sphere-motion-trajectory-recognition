"""叠加层缩放自测：`viz.scale` 必须让**整张图的所有元素**一起缩放。

为什么需要它
------------
`sphere_tracker/pipeline.py` 的 `render()` 会先按 `cfg.viz.scale` 放大画布，
再往上面画叠加。**画布放大了、坐标没放大**，叠加就会画到画面外。

这个 bug 真的存在过（2026-10-10 修）：

| 元素 | 修复前 |
| ---- | ------ |
| 轨迹 `draw_trail` | 用 `TrailPoint` 的**原始坐标** → 画到画面外 |
| 检测圆 `draw_detection` | 用 `det.x/det.y/det.radius` 的**原始值**，连 `det.contour` 也没缩放 |
| 预测线 `draw_prediction` | **只有它乘了 scale**（同一张图里一半缩放、一半没缩放） |
| HUD `draw_hud` | 字号写死 0.52，不跟 scale |

后果：`run_realtime.py --scale 2` 时圆和轨迹画在画面外，
看起来像"检测不到"，而它**只是画错地方了**。

判据：尺度不变性（**不是**像素相等）
-----------------------------------
语义（本次定死）：`viz.scale = k` 的含义是「整张原始图的 k 倍」，几何与文字全按 k 缩放。
所以每个叠加元素的**外接框**在 scale=k 时必须恰好是 scale=1 时的 **k 倍**。

⚠️ 走过的两条弯路，都记在这里免得重犯：

1. **"整图不一致率 < 固定阈值"抓不到 bug** —— 叠加只占全图 ~6%，
   整块画错位置也只有 5~7%，永远低于任何合理阈值。
   变异测试当场证明：把 5 处 scale 逐个去掉，只有 1 处能红。
   → 判据必须**只在内容像素上算**。
2. **"render(k) ≈ resize(render(1))" 在像素上不成立** ——
   抗锯齿是按各自分辨率算的，实测内容像素上差 86%，**那不是 bug**。
   → 不要拿"近似"当等式；改用**尺度不变性**：外接框按 k 缩放。
   它对颜色纯度不敏感（只取包围盒），因此不会被"圆被预测线盖住"这类遮挡影响。

反向自证（改了就会红）
--------------------
==================================  ==========================================  ========
改动                                注入方式                                    结果
==================================  ==========================================  ========
轨迹不乘 scale                       `draw_trail` 里去掉 `* scale`              exit=1
检测圆不乘 scale                     `draw_detection` 里去掉 `* scale`          exit=1
预测线不乘 scale                     `draw_prediction` 里去掉 `* scale`         exit=1
轮廓不乘 scale                       `draw_detection` 里 contour 那段不缩放     exit=1
==================================  ==========================================  ========

**本测试没覆盖的（如实写明，不假装覆盖了）**

- **HUD 内部几何**（`pad` / `lh` / `origin` 不乘 scale）：**不会红**。
  实测过：`pad = int(round(8*scale))` 改成 `pad = 8`，测试照样通过。
  原因是 HUD 面板的外接框**本来就被"原点 + 画布边界"框住** —— 它是从
  `(10*scale, 10*scale)` 画到画布右边界，所以内部调小 padding 根本不影响外框。
  这不是判据写松了，而是**这个量从外部不可观测**。
- **HUD 字号**（`fscale` 不乘 scale）：同理不可观测 —— 面板宽度由 `getTextSize`
  同时决定，两者混在一起分不开。
- 后果评估：这两项只影响"文字相对面板偏小/偏挤"，属**观感**，不是几何错位。
  真正会画错位置的（轨迹 / 圆 / 轮廓 / 预测线 / 面板整体位置）**全部覆盖**。

用法::

    python tests/selftest_viz_scale.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
while not (ROOT / "sphere_tracker").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from sphere_tracker.config import AppConfig, ColorConfig
from sphere_tracker.pipeline import SpherePipeline
from sphere_tracker.stats import Health
from sphere_tracker.types import Detection, FrameResult, TrailPoint
from sphere_tracker.viz import draw_3d_panel, hud_lines

W, H = 320, 240
BG = np.array([40, 40, 40], np.int16)
SCALE = 2.0
# 外接框的容差（像素，按 scale=1 的尺度算）。
# 抗锯齿与线宽/字号取整会让边界抖 1~5 px（实测预测线的起点差 5.0），
# 所以给 6 px；而"没乘 scale"的误差是**几十** px 级（实测差 64 px），仍然拦得住。
BBOX_TOL_PX = 6.0
# 颜色匹配：逐通道、宽容差（要把抗锯齿的边缘也算进去，才能得到完整外接框）
COLOR_TOL = 60
# 每种颜色至少要这么多像素才算"这个元素存在"
MIN_PIXELS = 20

TRAIL = np.array([0, 255, 255], np.int16)      # 青/黄（BGR，见 viz.py 的 base_color）
CIRCLE = np.array([0, 255, 0], np.int16)       # 绿（viz.py 的 color=(0,255,0)）
PRED = np.array([0, 140, 255], np.int16)       # 橙（BGR；viz.py 的 color=(0,140,255)）
ANY = None                                     # 不限颜色：所有非背景


def make_cfg(scale: float, hud: bool = False) -> AppConfig:
    """`hud` 默认**关**。

    ⚠️ 为什么几何判据要把 HUD 关掉：HUD 面板会**超出画布右边界**（文字太长时
    `draw_hud` 用 `min(x+w, frame.shape[1])` 截断）。被截断的元素**不满足**
    尺度不变性 —— scale=1 时被截到 320，scale=2 时被截到 640，
    两者不是 k 倍关系。那是**画布边界**的行为，不是缩放 bug。
    HUD 单独在第 4 节测。
    """
    cfg = AppConfig()
    cfg.viz.scale = scale
    cfg.viz.show_circle = True
    cfg.viz.show_prediction = True
    cfg.viz.show_hud = hud
    cfg.viz.show_mask = False
    cfg.viz.show_axes_3d = False
    # 关掉渐隐：否则轨迹颜色随位置变化，按固定色找不全
    cfg.viz.trail_enabled = True
    cfg.viz.trail_fade = False
    return cfg


def make_result() -> FrameResult:
    """斜穿画面的轨迹 + 一个检测圆（带轮廓）+ 一条预测线。"""
    trail = [TrailPoint(t=i * 0.05, x=40 + i * 8, y=60 + i * 4, score=1.0)
             for i in range(14)]
    cx, cy, r = 150.0, 90.0, 18.0
    contour = cv2.ellipse2Poly((int(cx), int(cy)), (int(r), int(r)), 0, 0, 360, 10)
    det = Detection(x=cx, y=cy, radius=r, score=0.9, method="color",
                    contour=contour.astype(np.float32))
    pred = np.array([[cx + 12 * i, cy + 6 * i] for i in range(6)], dtype=np.float64)
    return FrameResult(t=1.0, frame_index=7, detection=det, trail=trail,
                       prediction=pred, fps=30.0, detect_ms=12.0)


def render(scale: float, hud: bool = False, display_width: int | None = None) -> np.ndarray:
    """纯色背景上渲染（背景必须干净 —— 底图纹理会被算进"内容"）。

    `display_width` 传下去是为了第 6 节：它**只该影响几何，不该影响文字**。
    """
    frame = np.full((H, W, 3), BG.astype(np.uint8), np.uint8)
    return SpherePipeline(make_cfg(scale, hud=hud)).render(
        frame, make_result(), display_width=display_width)


def bbox(img: np.ndarray, color: np.ndarray | None, tol: int = COLOR_TOL):
    """某个颜色的像素外接框 (w, h, x0, y0)。`color=None` 表示所有非背景。"""
    if color is None:
        m = np.abs(img.astype(np.int16) - BG.reshape(1, 1, 3)).max(axis=2) > 25
    else:
        d = np.abs(img.astype(np.int16) - color.reshape(1, 1, 3))
        m = (d < tol).all(axis=2)
    ys, xs = np.nonzero(m)
    if len(xs) < MIN_PIXELS:
        return None, int(m.sum())
    return (int(xs.max() - xs.min()) + 1, int(ys.max() - ys.min()) + 1,
            int(xs.min()), int(ys.min())), int(m.sum())


ELEMENTS = (("轨迹", TRAIL), ("检测圆", CIRCLE), ("预测线", PRED), ("全部内容", ANY))


def test_every_element_scales_by_k() -> bool:
    """核心判据：每个元素的外接框在 scale=k 时恰为 scale=1 的 k 倍。"""
    print("----- 1. 每个叠加元素的尺寸与位置都按 k 缩放 -----")
    small, big = render(1.0), render(SCALE)
    print(f"  scale=1 画布 {small.shape[1]}x{small.shape[0]}，"
          f"scale={SCALE:g} 画布 {big.shape[1]}x{big.shape[0]}")

    checks = [("大图尺寸 = 小图 × k",
               big.shape[:2] == (int(H * SCALE), int(W * SCALE)))]
    for name, color in ELEMENTS:
        b1, n1 = bbox(small, color)
        b2, n2 = bbox(big, color)
        if b1 is None or b2 is None:
            print(f"  {name}：scale=1 {b1}（{n1}px）/ scale=k {b2}（{n2}px）")
            checks.append((f"{name} 在两种缩放下都能找到", False))
            continue
        # 期望：宽高与位置都 ×k；容差按 scale=1 的尺度给
        errs = [abs(b2[i] - b1[i] * SCALE) / SCALE for i in range(4)]
        ok = max(errs) <= BBOX_TOL_PX
        print(f"  {name}：scale=1 bbox(w,h,x,y)={b1} -> 期望 "
              f"{tuple(round(v * SCALE, 1) for v in b1)}；实际 {b2}；"
              f"最大误差 {max(errs):.1f}px（容差 {BBOX_TOL_PX:g}）")
        checks.append((f"{name} 的尺寸与位置都按 k 缩放（误差 ≤ {BBOX_TOL_PX:g}px）", ok))

    for nm, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {nm}")
    return all(ok for _n, ok in checks)


def test_everything_stays_inside_the_canvas() -> bool:
    """叠加不许被画布边缘裁掉 —— 修复前预测线会画到画面之外。"""
    print("----- 2. 叠加不许跑到画面外 -----")
    big = render(SCALE)
    h, w = big.shape[:2]
    b, n = bbox(big, ANY)
    if b is None:
        print("  [不达标] 画面上什么都没有")
        return False
    bw, bh, x0, y0 = b
    x1, y1 = x0 + bw - 1, y0 + bh - 1
    margins = (x0, y0, w - 1 - x1, h - 1 - y1)
    print(f"  内容 bbox=({x0},{y0})-({x1},{y1})  画布 {w}x{h}  像素 {n}")
    print(f"  四边留白 = {margins}"
          f"（修复前预测线会被画到画布外，这里会出现 0 或紧贴边）")

    checks = [
        ("画面里有内容", n > 0),
        ("四条边都留有余量（没有元素被画到画布外）", all(v > 2 for v in margins)),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_scale_one_is_unchanged() -> bool:
    """scale=1 时行为不许变（默认路径）。"""
    print("----- 3. scale=1 的行为不变（默认路径） -----")
    img = render(1.0)
    b, n = bbox(img, ANY)
    print(f"  画布 {img.shape[1]}x{img.shape[0]}，内容像素 {n}"
          f"（{n / img.shape[0] / img.shape[1] * 100:.2f}%）bbox={b}")
    checks = [
        ("画布尺寸不变", img.shape[:2] == (H, W)),
        ("内容仍被画出", b is not None and n > 200),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_hud_scales_too() -> bool:
    """HUD 也要跟 scale —— 单独测（几何判据里它是关着的，见 `make_cfg` 的说明）。

    ⚠️ **能验证到什么程度，这里要说清楚。** 走过的三条弯路：

    - 用**面积**不行：面积随倍率是 k²，但"行数/截断/抗锯齿"都掺进来，
      实测比值只有 3.84（不是 4），容差一放宽就抓不住变异。
    - 用**外接框宽度**也**抓不住字号**：面板自身的宽高也乘了 scale，
      墨迹外接框量到的其实是**面板**而不是文字 ——
      把 `fscale` 改回不缩放，宽度比依然是 2.00。**字体大小无法从外部观测。**
    - 最终判据 = **面板外接框的宽与高都按 k 缩放**。
      这正好覆盖真正会画错的部分（面板位置、内边距、行高）。
      **`fscale` 是否跟着缩放，本测试不声称能验证** —— 它的可见后果仅是文字
      相对面板偏小，属观感问题，不是几何错位。
    """
    print("----- 4. HUD 面板随 scale 变 -----")

    def hud_panel(img: np.ndarray):
        h, w = img.shape[:2]
        region = img[:, : w // 2]      # 只取左半，避开面板被画布右边界截断的影响
        m = np.abs(region.astype(np.int16) - BG.reshape(1, 1, 3)).max(axis=2) > 25
        ys, xs = np.nonzero(m)
        if len(xs) == 0:
            return None, 0
        return (int(xs.max() - xs.min()) + 1, int(ys.max() - ys.min()) + 1), int(m.sum())

    b1, n1 = hud_panel(render(1.0, hud=True))
    b2, n2 = hud_panel(render(SCALE, hud=True))
    if b1 is None or b2 is None:
        print(f"  scale=1 {b1}（{n1}px）/ scale=k {b2}（{n2}px）")
        print("  [不达标] HUD 没有画出来")
        return False
    rw, rh = b2[0] / b1[0], b2[1] / b1[1]
    print(f"  HUD 面板外接框：scale=1 {b1} -> 期望 "
          f"({b1[0] * SCALE:.0f}, {b1[1] * SCALE:.0f})；实际 {b2}")
    print(f"  宽比 {rw:.2f} / 高比 {rh:.2f}（应都约 {SCALE:g}）")
    print("  注：面板的**内部**几何（padding/行高）与字号从外部不可观测 ——")
    print("      面板外框由「原点 + 画布边界」框住，见文件头的「本测试没覆盖的」。")

    checks = [
        ("HUD 确实画出来了", n1 > 50 and n2 > 50),
        (f"面板宽度按 k 缩放（{0.9 * SCALE:.1f}~{1.15 * SCALE:.1f}）",
         0.9 * SCALE < rw < 1.15 * SCALE),
        (f"面板高度按 k 缩放（{0.9 * SCALE:.1f}~{1.15 * SCALE:.1f}）", 
         0.9 * SCALE < rh < 1.15 * SCALE),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_hud_text_is_immune_to_display_width() -> bool:
    """第 6 节：**显示缩放的倍率不许作用在文字上**（2026-10-14 用户的反馈）。

    ## 这一节补的是第 4 节明确"不声称能验证"的那个洞

    第 4 节写着「**`fscale` 是否跟着缩放，本测试不声称能验证** ——
    它的可见后果仅是文字相对面板偏小，属观感问题」。**那句评估是错的**：
    实际后果不是"偏小"，是**根本认不出来**。

    用户实测反馈原话：「为什么字还是如此模糊，这种文字应该设置为
    不受清晰度设置影响」。查下来是第二个、独立的字号问题：

    - 第 4 节覆盖的是「先缩图再画字」那个回归（已修，`pipeline.render()` 有记录）；
    - 而**字号本身**仍被 `display_width` 折进去的 `scale` 乘小：
      实时默认 `display_width=960`（1920 的 0.5 倍）→ `fscale=0.52*0.5=0.26`、
      笔画 `max(1,...)` 退到 **1 px**、行高 `20*0.5=10 px`。
      **字不是被重采样糊掉，是被缩到 10 像素高。**

    `display_width` 存在的唯一理由是省 `cv2.imshow` 的缩放开销
    （实测 17.74 ms → 1.84 ms/帧），它是**性能手段，不是显示意图**，
    所以没有理由把信息面板一起缩小。

    ## 判据为什么这次**能**从外部观测（第 4 节说不能）

    第 4 节量的是**面板**墨迹（相对背景色），面板框由"原点 + 画布边界"决定，
    所以量到的是面板不是字。这里换个量法：`draw_hud` 的**文字是纯白 (255,255,255)**，
    而底图是纯色背景 —— 所以「近白像素」的外接框**就是字形本身**，
    面板的黑底与背景都不满足。
    """
    print("----- 6. HUD 文字不随 display_width 变 -----")

    # ⚠️ 这一节**必须用全尺寸底图（1920x1080）**，不能用本文件默认的 `W, H = 320, 240`。
    # 原因：HUD 是"左上角一块按文字宽度自适应的面板"，实测 3 行文字的框就有
    # 350x53。底图只有 320 宽时，**面板会超出画布右边界被 `min(x+w, ...)` 截断**，
    # 于是"近白像素的外接框"量到的是**被裁掉的字**，而不是字本身 ——
    # 我第一版就是这么搭的台子：量出来 236x91 / 150x72 / 75x50，
    # 看起来像"文字跟着 display_width 缩"，其实**代码是对的、判据是错的**。
    # （`draw_hud` 的截断行为本身见第 4 节的说明，那里也踩过同一个坑。）
    #
    # ⚠️⚠️ **还必须只画 HUD**：轨迹的采样点是纯白 `(255,255,255)` 的实心小圆
    # （`draw_trail` 最后那段 `cv2.circle(..., (255,255,255), -1)`），
    # 它们同样满足"近白"判据，会把外接框一路撑到轨迹所在的位置 ——
    # 我第二版就是被这个骗了：同一个 display_width 下量到 91 而不是 53，
    # 差值恰好等于那些白点。所以这一节用 `hud_only` 把其余叠加全关掉。
    FRAME_W, FRAME_H = 1920, 1080

    def hud_only(scale: float) -> AppConfig:
        cfg = make_cfg(scale, hud=True)
        cfg.viz.trail_enabled = False
        cfg.viz.show_circle = False
        cfg.viz.show_prediction = False
        return cfg

    def render_full(scale: float, display_width: int | None = None) -> np.ndarray:
        frame = np.full((FRAME_H, FRAME_W, 3), BG.astype(np.uint8), np.uint8)
        return SpherePipeline(hud_only(scale)).render(
            frame, make_result(), display_width=display_width)

    def glyph_box(img: np.ndarray):
        """纯白字形的外接框（宽, 高）。底图与面板黑底都不是近白。"""
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ys, xs = np.nonzero(g > 200)
        if len(xs) == 0:
            return None
        return (int(xs.max() - xs.min()) + 1, int(ys.max() - ys.min()) + 1)

    full = glyph_box(render_full(1.0))
    half = glyph_box(render_full(1.0, display_width=FRAME_W // 2))
    quarter = glyph_box(render_full(1.0, display_width=FRAME_W // 4))
    print(f"  字号外接框：display_width={FRAME_W} {full}"
          f" / {FRAME_W // 2} {half} / {FRAME_W // 4} {quarter}")
    if full is None or half is None or quarter is None:
        print("  [不达标] 有一档没量到文字")
        return False

    # 显式放大意图必须仍然生效：viz.scale=2 时文字也该变大（这是既有契约）
    doubled = glyph_box(render_full(SCALE))
    ratio = (doubled[1] / full[1]) if doubled else 0.0
    print(f"  viz.scale={SCALE:g} 时字号外接框 {doubled}，高比 {ratio:.2f}"
          f"（应约 {SCALE:g} —— 显式放大意图仍然生效）")

    checks = [
        (f"display_width 缩到 1/2，文字**逐像素不变**（{full} -> {half}）", full == half),
        (f"display_width 缩到 1/4，文字仍逐像素不变（{full} -> {quarter}）",
         full == quarter),
        ("文字确实量到了（不是空框）", full[1] >= 8 and full[0] >= 40),
        (f"显式 viz.scale={SCALE:g} 仍让文字变大（高比 {ratio:.2f}）",
         0.9 * SCALE < ratio < 1.2 * SCALE),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_3d_panel_text_scales() -> bool:
    """3D 面板内部的**文字**也要跟 scale。

    面板的画布由调用方按 `scale` 给尺寸（`pipeline.py` 传 `size=(320*k, 240*k)`），
    但**字号不会因为画布变大而自动变大** —— `draw_3d_panel` 里原来写死 `0.42/0.38`，
    所以 `viz.scale=2` 时面板里的一切实测比外面小一半。2026-10-10 修。

    判据用**标题文字的高度**：它是干净的一维量，且正是被改的那个。
    不整块量面板，因为面板里还叠着**数据投影**（坐标跟着 `size` 走，是另一回事，
    放大后相对观感本来就不同）——混在一起量会得到一条抓不住 bug 的判据。
    """
    print("----- 5. 3D 面板的标题文字随 scale 变 -----")

    def title_ink_height(scale: float) -> int:
        hist = np.array([[10 + i, 20 + i * 2, 300 + i * 5] for i in range(12)], dtype=np.float64)
        pred = np.array([[30 + i, 40 + i, 350 + i * 5] for i in range(6)], dtype=np.float64)
        panel = draw_3d_panel(size=(int(320 * scale), int(240 * scale)),
                              history=hist, prediction=pred, scale=scale)
        # ⚠️ 量程必须**只圈住标题那一行**。
        # 第一版取"上面 1/4 且靠左 3/4"，把 y=30*scale 处的 "top view" 标签也圈了进去
        # —— 那行的位置也乘了 scale，于是变异后比值仍有 1.81，判据抓不住。
        # 标题的基线在 16*scale、字号 0.42*scale（≈10*scale 高），所以 [0, 22*scale) 够。
        band = panel[: max(2, int(round(22 * scale))), :]
        m = np.abs(band.astype(np.int16) - np.array([28, 28, 28], np.int16)).max(axis=2) > 25
        ys, _xs = np.nonzero(m)
        return 0 if len(ys) == 0 else int(ys.max() - ys.min()) + 1

    h1, h2 = title_ink_height(1.0), title_ink_height(SCALE)
    ratio = h2 / max(h1, 1)
    print(f"  标题文字高度：scale=1 -> {h1}px；{SCALE:g} -> {h2}px；比值 {ratio:.2f}"
          f"（字号×k 时应约 {SCALE:g}）")
    checks = [
        ("两种缩放下标题都画出来了", h1 > 2 and h2 > 2),
        (f"文字高度比 ≈ k（{0.9 * SCALE:.1f}~{1.25 * SCALE:.1f}）"
         f"—— 抓「面板字号没跟 scale」", 0.9 * SCALE < ratio < 1.25 * SCALE),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_content_area_grows_with_k_squared() -> bool:
    """内容**面积**应随 k² 增长（线宽/半径/字号都乘了 k）。"""
    print("----- 8. 内容面积随 k² 增长 -----")
    _, n1 = bbox(render(1.0), ANY)
    _, n2 = bbox(render(SCALE), ANY)
    ratio = n2 / max(n1, 1)
    expect = SCALE ** 2
    print(f"  内容像素：1 -> {n1}；{SCALE:g} -> {n2}；比值 {ratio:.2f}"
          f"（预期 ≈ {expect:g}）")
    checks = [
        (f"面积比值落在 k² 的 0.6~1.6 倍（{expect * 0.6:.1f}~{expect * 1.6:.1f}）",
         expect * 0.6 < ratio < expect * 1.6),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_hud_separates_the_three_frame_rates() -> bool:
    """HUD 必须把 **cap / loop / detect** 三个口径分开写，且**永不出现裸 `FPS`**。

    改之前第一行是 `FPS {result.fps}` —— 而 `result.fps` 量的是**主循环**速率，
    **不是相机采集帧率**。两件事在画面上长得一样，但含义差很远：

    - 主循环 15 fps 可能是"相机 30 fps 但检测太慢"，也可能是"相机只给了 15 fps"；
    - 只看一个数**分不出是哪一种**，"往哪个方向优化"纯靠猜。

    这正是 §10.10 与坑清单反复引用的"走错方向的根因"。
    """
    print("----- 6. HUD 的三个帧率口径分得开 -----")
    cfg = make_cfg(1.0)
    result = make_result()

    # 没接 Health 时：只能说清这是"loop"，不许再叫 FPS
    fallback = " ".join(hud_lines(result, cfg))
    has_bare_fps = "FPS" in fallback.upper().replace("LOOP", "")
    print(f"  未接 Health：{fallback.strip()[:76]}")
    print(f"    含裸 FPS = {has_bare_fps}（必须 False）")

    h = Health()
    h.note_capture(capture_ms=33.3, frame_age_ms=4.0)     # -> cap ≈ 30.0
    h.end_frame(detect_ms=12.5, process_ms=40.0)          # -> loop ≈ 25.0
    line = " ".join(hud_lines(result, cfg, health=h))
    print(f"  接了 Health：{line.strip()[:86]}")
    checks = [
        ("未接 Health 时也不出现裸 `FPS`（裸 FPS 就是歧义来源）", not has_bare_fps),
        ("未接 Health 时明确写 `loop`", "loop" in fallback),
        ("接了 Health 后有 `cap`", "cap" in line),
        ("接了 Health 后有 `loop`", "loop" in line),
        ("接了 Health 后有 `detect`", "detect" in line),
        ("三个数都在同一行（一眼能对比）",
         all(k in line.split("method")[0] for k in ("cap", "loop", "detect"))),
        ("cap 与 loop 是**不同**的值（否则等于没分开）",
         abs(h.capture_fps - h.loop_fps) > 0.5),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_hud_shows_stall() -> bool:
    """采集停住时 HUD 必须打出来。

    **没有这个提示，"球没动"与"相机没了"在画面上完全一样** ——
    那是本项目头号成本（出事了但没人知道）最纯粹的形态。
    """
    print("----- 7. 采集停住时 HUD 打 STALL -----")
    cfg = make_cfg(1.0)
    result = make_result()
    quiet = " ".join(hud_lines(result, cfg))
    result.stalled = True
    loud = " ".join(hud_lines(result, cfg))
    print(f"  stalled=False：{'STALL' in quiet}（必须 False）")
    print(f"  stalled=True ：{'STALL' in loud}（必须 True）")
    line = [ln for ln in hud_lines(result, cfg) if "STALL" in ln]
    if line:
        print(f"    {line[0]}")
    checks = [
        ("正常时不打 STALL（不许误报）", "STALL" not in quiet),
        ("停住时打 STALL", "STALL" in loud),
        ("提示里说清了是什么问题（不只是「卡住」）",
         any(k in loud for k in ("相机", "采集"))),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_display_width_contract() -> bool:
    """`render(display_width=...)` 的契约（2026-10-14 加）。

    ## 这一节守什么

    实测 `cv2.imshow` 把 1920x1080 的图缩进窗口显示要 **17.74 ms/帧**，
    而先把图缩到 960x540 再 imshow 只要 **1.84 ms** —— 便宜 9.6 倍。
    实测对照（同机同相机、各跑 20 秒、取最后三次采样）：

        --display-width 0    （1920x1080 显示，旧行为）  loop 65.2 fps
        --display-width 960  （新默认）                  loop 78.5 fps

    所以 `display_width` 是**流畅度**的一根支柱，钉三件事：

    1. 输出图**宽度确实是**要的那个值（否则"没生效"没人发现）；
    2. **叠加层跟着缩**（否则圈会画在错误的像素位置上 —— 比不缩更糟）；
    3. `None` / `0` 时行为**与从前完全一致**（不改旧路径）。

    第 2 条最要紧：只缩图不缩坐标，圈就偏了。
    """
    print("----- display_width：只缩输出图，叠加层必须跟着缩 -----")
    cfg = AppConfig()
    cfg.viz.show_mask = False
    pipe = SpherePipeline(cfg)
    base = np.full((1080, 1920, 3), 40, np.uint8)
    result = None
    for i in range(8):
        f = base.copy()
        cv2.circle(f, (860 + i * 20, 540), 120, (205, 205, 205), -1)
        result = pipe.process(f, i * 0.033)
    if result is None:                                        # 理论上不会
        print("  [不达标] pipeline 没能产生结果")
        return False

    checks: list[tuple[str, bool]] = []

    for want in (480, 960, 1280):
        out = pipe.render(base, result, display_width=want)
        gw, gh = out.shape[1], out.shape[0]
        expect_h = round(1080 * want / 1920)
        print(f"  display_width={want:5} -> 输出 {gw}x{gh}（高度期望 {expect_h}）")
        checks.append((f"display_width={want} 时输出宽度就是 {want}", gw == want))
        checks.append((f"display_width={want} 时高度按比例", abs(gh - expect_h) <= 2))

    for kw in (None, 0):
        out = pipe.render(base, result, display_width=kw)
        print(f"  display_width={kw!r:5} -> 输出 {out.shape[1]}x{out.shape[0]}（应为原尺寸）")
        checks.append((f"display_width={kw!r} 保持原尺寸（旧路径不变）",
                       out.shape[1] == 1920 and out.shape[0] == 1080))

    # ---- 叠加层跟着缩：比较"内容重心"的相对位置 ----
    #
    # 判据要选对。直接比像素没意义（采样分辨率不同）。
    # 用**亮度重心**：叠加层（亮圈、亮线、白字）会把它拉向球所在的位置。
    # 若坐标没跟着缩，小图里的圈会落在完全错误的地方，重心差到几百像素；
    # 正常缩放的抗锯齿差异只有几个像素 —— 两者量级差两个数量级，判得开。
    big = pipe.render(base, result, display_width=None)
    small = pipe.render(base, result, display_width=960)
    big_small = cv2.resize(cv2.cvtColor(big, cv2.COLOR_BGR2GRAY),
                           (small.shape[1], small.shape[0]),
                           interpolation=cv2.INTER_AREA)
    g_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    def centroid(g: np.ndarray) -> tuple[float, float]:
        m = g.astype(np.float32)
        m -= m.min()
        tot = float(m.sum()) or 1.0
        ys, xs = np.indices(m.shape)
        return float((xs * m).sum() / tot), float((ys * m).sum() / tot)

    cb, cs = centroid(big_small), centroid(g_small)
    dx, dy = abs(cb[0] - cs[0]), abs(cb[1] - cs[1])
    print(f"  内容重心：大图缩后 ({cb[0]:.1f},{cb[1]:.1f})  小图 ({cs[0]:.1f},{cs[1]:.1f})"
          f"  差 ({dx:.1f},{dy:.1f}) px")
    # ⚠️ 这条**是冗余护栏**，如实说明：把 `scale` 改成 1.0（缩了图却按原坐标画）
    # 时，本节第 1~8 节的"轨迹/检测/预测随 k 缩放"断言会**先**红，
    # 所以重心这条抓不到"独家"的 bug。留着它是因为它检查的是**端到端对齐**
    # （叠加层相对图像内容的位置），而不是"某个函数有没有用 scale"——
    # 将来若有叠加层不走 `scale` 参数而是自己算坐标，它才是唯一的守门人。
    checks.append(("叠加层位置随图一起缩（重心偏差 < 40px；不缩会差几百 px）"
                   "（与前面的 scale 断言冗余，留作端到端对齐检查）",
                   dx < 40 and dy < 40))

    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_detect_width_rescales_coordinates() -> bool:
    """`pipe.set_detect_width()` 的契约（2026-10-14 加）。

    ## 这一节守什么

    实测（1080p、**真实纹理**画面 —— 书架那种）：

        color    17.8 ms
        hough  4384.9 ms   ← 4.4 秒一帧
        fusion 3740.5 ms   ← 3.7 秒一帧

    而同样的帧若是**纯色合成图**，`hough` 只要 132 ms。差别在于 `HoughCircles`
    先做 Canny，纹理一多边缘就爆炸。**真实房间永远是纹理丰富的**，
    所以"合成图上测得挺快"没有参考价值 —— 这是本项目的老教训。

    降检测分辨率是有效的杠杆，但降了分辨率就**必须把坐标换算回原图**：
    不换算的话，圈画在错位置，而且**半径错一倍 → 三维深度错一倍**
    （`camera.depth_from_radius` 直接吃半径）。

    判据：**同一个球，在不同 `detect_width` 下解出的圆心/半径必须一致**
    （换算正确 → 一致；漏换算 → 差一个倍率）。
    """
    print("----- detect_width：检测在缩图上跑，坐标必须换算回原图 -----")
    checks: list[tuple[str, bool]] = []

    # 造一个"灰底 + 一个大白球"的帧：color 检测器对它有稳定响应。
    # 球放在**非中心**位置，这样"位置错一个倍率"会立刻暴露。
    H, W = 1080, 1920
    frame = np.full((H, W, 3), 60, np.uint8)
    truth_x, truth_y, truth_r = 1300.0, 400.0, 90.0

    def make() -> np.ndarray:
        f = frame.copy()
        cv2.circle(f, (int(truth_x), int(truth_y)), int(truth_r),
                   (235, 235, 235), -1)
        return f

    got: dict[int, tuple[float, float, float]] = {}
    for dw in (0, 960, 640):
        cfg = AppConfig()
        cfg.detector.color = ColorConfig.preset("white_ball")   # 白色球的预设
        pipe = SpherePipeline(cfg)
        if dw:
            pipe.set_detect_width(dw)
        res = None
        for i in range(3):                          # 跑几帧让内部状态稳
            res = pipe.process(make(), i * 0.033)
        det = res.detection
        if det is None:
            print(f"  detect_width={dw or '原始':>6} -> **没检出**")
            checks.append((f"detect_width={dw} 时能检出球", False))
            continue
        got[dw] = (det.x, det.y, det.radius)
        print(f"  detect_width={dw or '原始':>6} -> "
              f"({det.x:7.1f},{det.y:6.1f}) r={det.radius:6.1f}"
              f"   真值 ({truth_x:.0f},{truth_y:.0f}) r={truth_r:.0f}")

    checks.append(("三种检测宽度都能检出球", len(got) == 3))
    if len(got) == 3:
        base = got[0]
        for dw in (960, 640):
            g = got[dw]
            dx = abs(g[0] - base[0])
            dy = abs(g[1] - base[1])
            dr = abs(g[2] - base[2])
            print(f"  detect_width={dw} 与原始分辨率之差："
                  f"位置 ({dx:.1f},{dy:.1f}) px  半径 {dr:.1f} px")
            # 换算正确 -> 差在抗锯齿/取整的量级（几 px）；
            # 漏换算 -> 位置差几百 px、半径差一个倍率。两者量级完全不同。
            checks.append((f"detect_width={dw} 时**位置**换算正确（误差 < 15px，"
                           f"漏换算会差几百 px）", dx < 15 and dy < 15))
            checks.append((f"detect_width={dw} 时**半径**换算正确（误差 < 15px，"
                           f"漏换算会差一个倍率 —— 深度会跟着错）", dr < 15))
        # 与真值比：确认球确实被找到对的地方（不只是"两次都错得一样"）
        bx, by, br = base
        checks.append(("原始分辨率下与真值一致（位置 < 15px、半径 < 20px）",
                       abs(bx - truth_x) < 15 and abs(by - truth_y) < 15
                       and abs(br - truth_r) < 20))

    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_mono_visualization_contract() -> bool:
    """第 9 节：单目三维可视化的 schema / 扫描 / 0 检出判据（2026-10-29 增）。

    ## 为什么这一节存在

    用户原话："主界面 3D 查看器只能查看双目会话的内容，单目那里没有查看器"。
    查下去比"没有入口"更糟：单目离线检测**什么都没测到却静默报成功**
    （现场 288 帧的一段，默认 ``color`` 40 帧里检出 0 帧，退出码 0、
    CSV 写出但 x/y 全空）。三件事必须钉住：

    1. 单目**本来就有**三维（``Z=f·D/(2r)``），导出的文件要能被查看器读；
    2. 那个文件**不能冒充双目** —— schema 必须不同名（尺度来源不同：
       先验 vs 视差），把 `mode` 读成 mono；
    3. 0 检出不写空文件、且退出码非 0；低检出警告但仍 exit 0。

    全部用 `tempfile`：判据不许依赖这台机器录过什么。
    """
    print("----- 9. 单目可视化：schema / 扫描 / 检出判据 -----")
    import json
    import tempfile

    from apps.run_video import (LOW_HIT_RATE, build_vis, hit_report,
                                session_name)
    from apps.view_3d import (MONO_SCHEMA, is_mono, load_view, scan_sources,
                             source_label)

    checks: list[tuple[str, bool]] = []

    def ck(label: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'通过' if ok else '不达标'}] {label}"
              + (f"（{detail}）" if detail else ""))
        checks.append((label, ok))

    # ---- 1. 生产者 → 磁盘 → 查看器：一条真链路 ----
    rows = [{"frame": i, "t": round(i / 30, 5), "coasting": 0 if i % 5 else 1,
             "X_m": 0.01 * i, "Y_m": 0.02, "Z_m": 0.8 + 0.001 * i}
            for i in range(6)]
    rows.append({"frame": 6, "t": 0.2, "coasting": ""})   # 没有 position_3d 的行
    with tempfile.TemporaryDirectory() as tmp:
        sess = Path(tmp) / "20260101-000000"
        (sess / "derived").mkdir(parents=True)
        vis = build_vis(rows, session=session_name(sess / "clip.mkv"),
                        source="clip.mkv", ball_diameter_mm=21.6, fx=1387.5,
                        created="2026-01-01 00:00:00")
        assert vis is not None
        f = sess / "derived" / "visualization.json"
        f.write_text(json.dumps(vis, ensure_ascii=False, indent=2), encoding="utf-8")

        data = load_view(f)
        ck("查看器能读单目文件，且点数 = 有 position_3d 的行数（6，不是 7）",
           len(data["trajectory"]) == 6, f"{len(data['trajectory'])} 点")
        ck("mode 读成 mono（不冒充双目）",
           data.get("mode") == "mono" and is_mono(data),
           f"schema={data.get('schema')} mode={data.get('mode')}")
        ck("schema 与双目**不同名**（同名会让下游以为在读视差测量）",
           data.get("schema") == MONO_SCHEMA
           and data.get("schema") != "sphere-stereo-visualization",
           str(data.get("schema")))
        ck("尺度来源可追溯（D / fx 都落盘，不是缺省值）",
           abs(float(data["ball_diameter_mm"]) - 21.6) < 1e-6
           and abs(float(data["fx"]) - 1387.5) < 1e-6,
           f"D={data.get('ball_diameter_mm')} fx={data.get('fx')}")
        ck("confident = not coasting（跟踪器外推的点不许算可信）",
           [p["confident"] for p in data["trajectory"]]
           == [not bool(r.get("coasting")) for r in rows[:6]],
           str([p["confident"] for p in data["trajectory"]]))

        found = scan_sources(roots=(Path(tmp),))
        ck("扫描能把单目会话列进来（以前只扫双目根）",
           [q.parent.parent.name for q in found] == ["20260101-000000"],
           str([q.name for q in found]))
        if found:
            label = source_label(found[0])
            ck("下拉文案带 [单目] 前缀（一眼分出类型）",
               label.startswith("[单目]"), label)

        # ---- 2. 空轨迹 / 错 schema：必须抛错，不许"读进来 0 点" ----
        bad = Path(tmp) / "bad_schema" / "derived"
        bad.mkdir(parents=True)
        (bad / "visualization.json").write_text(
            json.dumps({"schema": "sphere-visualization", "version": 1,
                        "trajectory": [{"xyz": [0, 0, 1]}]}), encoding="utf-8")
        try:
            load_view(bad / "visualization.json")
            ck("错 schema 抛错（没有被当成双目放过）", False)
        except ValueError as exc:
            ck("错 schema 抛错（没有被当成双目放过）", True, str(exc)[:40])

        empty = Path(tmp) / "empty" / "derived"
        empty.mkdir(parents=True)
        (empty / "visualization.json").write_text(
            json.dumps({"schema": MONO_SCHEMA, "version": 1, "mode": "mono",
                        "trajectory": []}), encoding="utf-8")
        try:
            load_view(empty / "visualization.json")
            ck("空轨迹抛错（空文件不许被当成「有结果」）", False)
        except ValueError:
            ck("空轨迹抛错（空文件不许被当成「有结果」）", True)

    # ---- 3. 生产侧的判据：0 检出 / 低检出 / 正常 ----
    ck("0 检出 → build_vis 返回 None（不写空文件）",
       build_vis([{"frame": 0, "t": 0.0, "coasting": ""}],
                 session="s", source="c.mkv", ball_diameter_mm=21.6,
                 fx=1000.0) is None)

    code, lines = hit_report(0, 40, "color")
    txt = "\n".join(lines)
    ck("0 检出 → 退出码 2（不是 0）", code == 2, f"EXIT={code}")
    ck("0 检出的诊断里写了当前检测器 + 该改试哪个",
       "color" in txt and "--method hough" in txt and "--method fusion" in txt,
       txt.splitlines()[1].strip() if len(lines) > 1 else "")
    ck("0 检出的诊断说清了「为什么不静默」（提到「看起来成功」）",
       "看起来成功" in txt)

    code_w, lines_w = hit_report(3, 40, "motion")
    ck(f"低检出（7.5% < {LOW_HIT_RATE:.0%}）→ 退出码仍是 0（有数据不假装失败）",
       code_w == 0 and any("[警告]" in ln for ln in lines_w),
       f"EXIT={code_w}")

    code_ok, lines_ok = hit_report(38, 40, "hough")
    ck("正常检出 → 退出码 0 且不告警", code_ok == 0 and not lines_ok)

    print(f"  共 {len(checks)} 项：" + ("全部通过" if all(k for _l, k in checks)
                                      else "有不达标项"))
    return all(k for _l, k in checks)


def test_valid_frame_gate_contract() -> bool:
    """第 10 节：`--seed` 与"有效帧"口径（2026-10-30 增）。

    ## 为什么这一节存在

    上一版的"检出"= 检测器给了东西。同一段录像（`20261029-205522`）实测：

    ==========  ==============  ==================
    方法        有输出          真的落在球上
    ==========  ==============  ==================
    color       0/40            0%
    hough       139/144 = 96%   **2/144 = 1.4%**
    ==========  ==============  ==================

    hough 那 96% 是书架/椅子上的小圆（r≈11~13 px），球的半径约 91 px。
    也就是说"有输出"这个口径**本身**就是那个"看起来成功"的 bug。

    本节的判据只钉**能钉住的部分**（纯函数，不需要素材）：

    1. `--seed x,y,r` 写得不对必须报错退出，不许猜；
    2. 跳变护栏只回退半径、帧仍算有效；带越界要**整帧**不算；
    3. **给了种子时带锚在种子半径上**（`seed_r × [0.75, 1.35]`），不许锚滚动
       中位 —— 旧 `derived/traj.csv` 里 277 帧半径全是 11 px（真值 91），
       中位就是 11，锚中位等于自证，每帧都"在带内"；
    4. 四段报告的拆项**加得起来**（无检测 + 半径不一致 = 总数 − 有效帧），
       且没有种子时必须明说"半径未经确认、绝对深度不可信"；
    5. 丢失按连续段报，不逐帧一行；
    6. 退出码仍然以有效帧为准（0 有效 → 2），且诊断里给 `--seed` 这条出路。

    ⚠️ **本节不声称能验证**"有效帧就是那颗球"。实测：给了**错误**种子
    （书架上的 `300,200,20`）时，CSRT 老实跟住那块纹理，细化每帧都在它周围
    找到边缘，量出 19.5 px（IQR 18.2~20.0），落在种子带 `[15, 27]` 内，
    60/60 全过门。**种子只能把门的锚点钉死，锚错了它照样自洽地错** ——
    种子带挡的是"跟丢后锁到别处"，不是"一开始就点错"。
    """
    print("----- 10. --seed 与「有效帧」口径 -----")
    import json
    import tempfile

    from apps.run_video import (MAX_RADIUS_JUMP, RADIUS_BAND, RADIUS_WINDOW,
                                SEED_BAND_HI, SEED_BAND_LO, RadiusGate,
                                build_vis, consistency_lines, hit_report,
                                loss_segments, parse_seed)

    checks: list[tuple[str, bool]] = []

    def ck(label: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'通过' if ok else '不达标'}] {label}"
              + (f"（{detail}）" if detail else ""))
        checks.append((label, ok))

    # ---- 1. --seed 解析 ----
    seed = parse_seed("895,718,88")
    ck("--seed 895,718,88 解析成 (895,718,88)", seed == (895.0, 718.0, 88.0), str(seed))
    for bad in ("1,2", "a,b,c", "1,2,0", "1,2,-3", ""):
        try:
            parse_seed(bad)
            ok = False
        except SystemExit:
            ok = True
        ck(f"非法 --seed {bad!r} → 报错退出（不猜）", ok)

    # ---- 3. 带必须锚在**种子半径**上，不能锚滚动中位 ----
    # 这一段守两个东西：① 顺序（先护栏后判带，带就永远触发不了，8% < 25%）；
    # ② 锚点（锚滚动中位是**自证**的：一屋子 11 px 的小圆中位就是 11）。
    g2 = RadiusGate(seed_radius=88.0)
    r_used2, verdict2 = g2.accept(11.0)      # 旧 traj.csv 的半径中位
    ck(f"种子 88 px 下，旧 csv 的 11 px（-{1 - 11 / 88:.0%}）→ 整帧被拒",
       r_used2 is None and verdict2 == "inconsistent" and g2.inconsistent == 1,
       f"verdict={verdict2} 带={g2.band()}")
    ck("被拒的帧不进入已接受半径序列（不污染后面的统计）",
       g2.radii == [] and abs(g2.r_prev - 88.0) < 1e-6, f"radii={g2.radii}")
    ck(f"上界给到 +{SEED_BAND_HI:.0%}（球被甩近时半径会变大，实测 +5%）",
       RadiusGate(seed_radius=88.0).accept(88.0 * (1 + SEED_BAND_HI - 0.01))[0]
       is not None)
    ck(f"下界 -{SEED_BAND_LO:.0%} 之外整帧拒（锁到小球）",
       RadiusGate(seed_radius=88.0).accept(88.0 * (1 - SEED_BAND_LO - 0.01))[0]
       is None)
    ck("没给种子时才退回滚动中位（±25%），且它仍能触发",
       RadiusGate().accept(11.0)[0] is not None)   # 前 2 帧样本不足，不过带
    g3 = RadiusGate()
    for _ in range(3):
        g3.accept(11.0)
    ck("无种子 + 滚动中位：偏离中位 2×25% 仍被拒（带不是死代码）",
       g3.accept(11.0 * (1 + 2 * RADIUS_BAND))[0] is None and g3.inconsistent == 1)

    # ---- 3b. 跳变护栏在种子带之后才生效：两者分工不重叠 ----
    g4 = RadiusGate(seed_radius=90.0)
    g4.accept(90.0)
    g4.accept(90.4)
    r_used4, verdict4 = g4.accept(90.0 * (1.0 + 2 * MAX_RADIUS_JUMP))
    ck("带内但跳 16% → jump、沿用上一帧半径（帧仍算有效）",
       verdict4 == "jump" and abs(r_used4 - 90.4) < 1e-6 and g4.jumps == 1,
       f"verdict={verdict4} r={r_used4:.2f}")

    # ---- 4. 四段报告：拆项加得起来，尺度必须交代 ----
    lines = consistency_lines(
        total=10, accepted=7, no_detection=2, inconsistent=1, jumps=3,
        radii=[90.0, 91.0, 92.0], jump_ratios=[0.003, 0.009, 0.014],
        bad_frames=[3, 4, 5, 9], fps=35.0, seed=None)
    txt = "\n".join(lines)
    ck("有效帧写成 7/10 = 70.0%（分母是读到的帧数）", "7/10 = 70.0%" in txt)
    ck("拆项加得起来：无检测 2 + 半径不一致 1 = 10 − 7",
       "无检测 2" in txt and "半径不一致 1" in txt and 10 - 7 == 2 + 1)
    ck("跳变护栏单独报**次数**（回退半径不丢帧，混进拆分会加不平）",
       "跳变护栏触发 3 次" in txt)
    ck("半径给中位与 IQR（只看中位看不出逐帧抖动）",
       "半径 91.0 px" in txt and "IQR" in txt)
    ck("丢失按**连续段**报（2 段：帧 3..5、帧 9..9）",
       "连续丢失 2 段" in txt and "帧 3..5" in txt and "帧 9..9" in txt,
       txt.splitlines()[-2][:40])
    ck("没给种子 → `[尺度]` 明说半径未经确认、绝对深度不可信",
       "[尺度] 无人工种子：半径未经确认，绝对深度不可信" in txt, "")
    seeded_txt = "\n".join(consistency_lines(
        total=2, accepted=2, no_detection=0, inconsistent=0, jumps=0,
        radii=[90.0, 90.1], jump_ratios=[0.001], bad_frames=[],
        fps=35.0, seed=(895.0, 718.0, 88.0)))
    ck("给了种子 → `[尺度]` 报锚点与接受区间（66.0~118.8）",
       "人工种子 88 px 锚定" in seeded_txt and "66.0~118.8" in seeded_txt)
    ck("一帧都没过门时不编造半径统计",
       "半径统计无数据" in "\n".join(consistency_lines(
           total=5, accepted=0, no_detection=5, inconsistent=0, jumps=0,
           radii=[], jump_ratios=[], bad_frames=[0, 1, 2, 3, 4],
           fps=35.0, seed=None)))
    ck("无丢失时打「无」，而不是空着",
       consistency_lines(total=2, accepted=2, no_detection=0, inconsistent=0,
                         jumps=0, radii=[90.0, 90.0], jump_ratios=[0.0],
                         bad_frames=[], fps=35.0, seed=None)[-2].strip()
       == "[丢失]   无")

    # ---- 5. 丢失分段（纯函数）----
    ck("连续段把单帧也当一个段",
       loss_segments([5]) == [(5, 5)] and loss_segments([]) == [])
    ck("段之间断开不合并",
       loss_segments([1, 2, 3, 7, 8]) == [(1, 3), (7, 8)])

    # ---- 6. 退出码以**有效帧**为准 ----
    code, lines0 = hit_report(0, 60, "hough", detector_error="跟丢")
    txt0 = "\n".join(lines0)
    ck("0 有效帧 → 退出码 2", code == 2, f"EXIT={code}")
    ck("0 有效帧的诊断给出 --seed 这条出路（金属球只能人工点一下）",
       "--seed" in txt0 and "金属" in txt0)
    ck("0 有效帧的诊断带上检测器自述原因（不然只能靠猜）", "跟丢" in txt0)
    ck("有效帧 59/60 → 退出码 0（有数据不假装失败）", hit_report(59, 60, "hough")[0] == 0)

    # ---- 7. 可视化文件能溯源：这条轨迹是不是种子跟出来的 ----
    rows = [{"frame": i, "t": i / 30.0, "coasting": 0,
             "X_m": 0.01 * i, "Y_m": 0.0, "Z_m": 0.16} for i in range(4)]
    vis = build_vis(rows, session="s", source="clip.mkv", ball_diameter_mm=21.6,
                    fx=1387.5, seed=(895.0, 718.0, 88.0),
                    accepted_frames=4, rejected_frames=2)
    ck("vis 里有 seed（半年后能看出是人工种子跟出来的）",
       vis is not None and vis["seed"] == [895.0, 718.0, 88.0], str(vis and vis["seed"]))
    ck("vis 里有 accepted/rejected 帧数",
       vis is not None and vis["accepted_frames"] == 4 and vis["rejected_frames"] == 2)
    auto = build_vis(rows, session="s", source="clip.mkv", ball_diameter_mm=21.6,
                     fx=1387.5)
    ck("没给种子时 seed 落成 null（不是缺字段 —— 缺字段会被读成「没这个功能」）",
       auto is not None and "seed" in auto and auto["seed"] is None)
    ck("没给种子 → scale_unverified=true（尺度没人背书，落盘要能看出来）",
       auto is not None and auto.get("scale_unverified") is True)
    ck("给了种子 → scale_unverified=false",
       vis is not None and vis.get("scale_unverified") is False)
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "visualization.json"
        assert vis is not None
        f.write_text(json.dumps(vis, ensure_ascii=False), encoding="utf-8")
        ck("落盘后仍是合法 JSON（种子是元组，不转 list 会写不出来）",
           json.loads(f.read_text(encoding="utf-8"))["seed"] == [895.0, 718.0, 88.0])

    ck(f"窗口长度 {RADIUS_WINDOW} 帧 ≈ 0.86 s @35 fps（覆盖不了更长的目标切换）",
       RADIUS_WINDOW == 30)

    print(f"  共 {len(checks)} 项：" + ("全部通过" if all(k for _l, k in checks)
                                      else "有不达标项"))
    return all(k for _l, k in checks)


def main() -> int:
    print("=" * 70)
    print("叠加层缩放自测（viz.scale：整张图所有元素一起缩放）")
    print("=" * 70)
    results = []
    for fn in (test_every_element_scales_by_k,
               test_everything_stays_inside_the_canvas,
               test_scale_one_is_unchanged,
               test_hud_scales_too,
               test_3d_panel_text_scales,
               test_hud_separates_the_three_frame_rates,
               test_hud_shows_stall,
               test_content_area_grows_with_k_squared,
               test_display_width_contract,
               test_detect_width_rescales_coordinates,
               test_hud_text_is_immune_to_display_width,
               test_mono_visualization_contract,
               test_valid_frame_gate_contract):
        results.append(fn())
        print()
    ok = all(results)
    print("判定结果：" + ("全部通过" if ok else "有不达标项"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
