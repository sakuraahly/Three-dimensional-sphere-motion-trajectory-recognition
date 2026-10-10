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

from sphere_tracker.config import AppConfig
from sphere_tracker.pipeline import SpherePipeline
from sphere_tracker.types import Detection, FrameResult, TrailPoint

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


def render(scale: float, hud: bool = False) -> np.ndarray:
    """纯色背景上渲染（背景必须干净 —— 底图纹理会被算进"内容"）。"""
    frame = np.full((H, W, 3), BG.astype(np.uint8), np.uint8)
    return SpherePipeline(make_cfg(scale, hud=hud)).render(frame, make_result())


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


def test_content_area_grows_with_k_squared() -> bool:
    """内容**面积**应随 k² 增长（线宽/半径/字号都乘了 k）。"""
    print("----- 5. 内容面积随 k² 增长 -----")
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


def main() -> int:
    print("=" * 70)
    print("叠加层缩放自测（viz.scale：整张图所有元素一起缩放）")
    print("=" * 70)
    results = []
    for fn in (test_every_element_scales_by_k,
               test_everything_stays_inside_the_canvas,
               test_scale_one_is_unchanged,
               test_hud_scales_too,
               test_content_area_grows_with_k_squared):
        results.append(fn())
        print()
    ok = all(results)
    print("判定结果：" + ("全部通过" if ok else "有不达标项"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
