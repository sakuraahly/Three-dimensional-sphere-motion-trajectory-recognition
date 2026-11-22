"""颜色分割检测器（HSV 阈值 + 形态学 + 轮廓打分）。

## 三条入口（2026-11-06 补齐）

| 入口 | 怎么触发 | 色相从哪来 | 保证什么 |
| ---- | -------- | ---------- | -------- |
| 预设 | `--preset red_ball` / 默认白球 | `COLOR_PRESETS` 硬编码 | 球色已知且背景可分时最快 |
| 自动 | `--preset auto` | 画面主色（不需要点选） | 不知道球色也能起步 |
| 点选 | `seed(x, y, r)` | 点击点周围的一圈像素 | **种子帧一定有 Detection** |

## 为什么点选后必须"一定识别到"

种子不是"给个大概位置"，而是用户的声明：**这个位置、这个半径就是球**。
所以：

1. 种子帧先从点击点周围取色（`_seed_ranges`），把球的真实 HSV 区间测出来 ——
   这解决了"预设不匹配 → 静默 0 检出"（B-175 的另一半）；
2. 半径带锚在种子半径上，只接受 `[0.6, 1.7]×seed_r` 的候选；
3. 种子帧如果颜色分割一个候选都没给出，**直接用种子本身作为 Detection**
   （`extra["seed_anchor"]`）—— 用户点过的地方不允许报"没看见"；
4. 之后每一帧若颜色分割失败，就沿"上一帧位置 + 速度"用亚像素射线找球缘
   （`extra["edge_follow"]`）。任何球都有边缘，这一路不依赖颜色可分。
5. 取色结果先过质量护栏（`_seed_ranges_verdict`）：判"落在背景/区间过宽"时
   **不采用**，原因写进 `last_error` 与 `_diag`，并回落到预设/auto。点选语义是
   "这里有一颗球"，不是"照抄这里的颜色"—— 球心喂到球不在的帧上时，照抄只会
   把背景色当球色，然后返回点选锚点冒充检出（2026-11-15 修）。

## 多候选时选谁（B-176）

同色干扰物的四项形状得分可以**逐位相同**，`detections[0]` 由扫描顺序决定。
所以候选先按分数筛出"并列第一档"（≥ 最高分 × 0.92），再取离
"上一帧位置 + 速度"最近的那个。分差明显时仍以分数为准，避免为了连续性
把明显更好的候选换掉。
"""

from __future__ import annotations

import cv2
import math
import numpy as np

from .base import BaseDetector
from ..config import AUTO_PRESET, ColorConfig, DetectorConfig
from ..geometry import (
    contour_circularity,
    fit_circle_robust,
    radial_consistency,
    refine_circle_subpixel,
    refine_circle_subpixel_checked,
)
from ..types import Detection

#: 自动取色的分析宽度。只用于**选区间**，真正的检测仍在原分辨率上做。
_AUTO_WIDTH = 480
#: 判"掩膜把背景也吃进来了"的覆盖率。40% 是经验值：一个球占画面 40% 以上时
#: 它已经不是"小目标"，而那正是颜色阈值失效的典型形态（纯色墙/桌布）。
_MASK_COVER_WARN = 0.40
#: 自动取色时"背景被吃进来"的扣分。不是硬拒绝 —— 球贴得很近时覆盖率也可能高，
#: 但同分情况下应优先选更"紧"的区间。
_COVER_PENALTY = 0.55

#: 种子取色的质量护栏（2026-11-15）。`_init_seed_ranges` 以前只判"取到像素没有"，
#: 于是把球心喂给**球不在画面**的帧时，会静默把背景当球色。实测同一会话
#: （`data/stereo/parallel/20261111-153857`，`seed_frame=40`，球心 954.9,1034.6，
#: 半径 24.6）：
#:
#:   · 帧 40（球在）：两段红 `[H146~179 S≥41 V≥67; H0~8 S≥41 V≥67]`，
#:     球外邻环同色占比 **0.0001**、整帧 **0.012**，真检出 score 0.889；
#:   · 帧 0/151/301（球不在）：单段 `[H20~110 S≥2 V≥117]` / `[H14~110 S≥4 V≥117]` /
#:     `[H16~108 S≥3 V≥142]`，S 下界 2/4/3，球外邻环占比 **0.83/0.96/0.94**、
#:     整帧 **0.647/0.749/0.693**，返回 n=1、score=1.000 的**种子锚点**冒充检出。
#:
#: 两条判据都必须是**组合条件**：单看"色相跨度宽"会误伤白球（白光无色相先验），
#: 单看"S 下界低"会误伤白球/浅色球，单看"整帧覆盖高"会因画面里别的同色物体误伤。
_SEED_ANNULUS_LO = 1.35          # 邻环内径 ×种子半径（球缘在 1.0r，这个环已在球外）
_SEED_ANNULUS_HI = 2.50          # 邻环外径 ×种子半径
_SEED_ANNULUS_COVER_MAX = 0.35   # 邻环同色 >35% → 背景（实测 0.0001 vs 0.83~0.96）
_SEED_ANNULUS_MIN_PIXELS = 64    # 邻环太小（球极小/贴边）就不判，宁可放过不可误伤
_SEED_FLAT_HUE_SPAN = 70         # 单段色相跨度 ≥70（≈1/3 色环）＝ 没有色相先验
_SEED_FLAT_S_MAX = 8             # 且 S 下界 ≤8（几乎无彩）—— 背景 2/3/4，红球 41
_SEED_FRAME_COVER_MAX = 0.20     # 且无彩宽区间吃掉整帧 >20%（实测 0.647~0.749 vs 0.012）


# --------------------------------------------------------------------------- #
# 纯函数：HSV 区间估计（可在无检测器的情况下单独测）
# --------------------------------------------------------------------------- #
def _ranges_from_config(c: ColorConfig) -> list[tuple[np.ndarray, np.ndarray]]:
    """把 `ColorConfig` 的两个区间整理成统一的 ``[(lower, upper), ...]``。"""
    out = [(np.array(c.hsv_lower, dtype=np.uint8),
            np.array(c.hsv_upper, dtype=np.uint8))]
    if c.use_second_range:
        out.append((np.array(c.hsv_lower2, dtype=np.uint8),
                    np.array(c.hsv_upper2, dtype=np.uint8)))
    return out


def _mask_from_ranges(hsv: np.ndarray, ranges, *, open_ksize: int,
                      close_ksize: int) -> np.ndarray:
    """按若干段 HSV 区间建掩膜，再做开/闭运算。"""
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        mask |= cv2.inRange(hsv, lower, upper)
    if open_ksize >= 3:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if close_ksize >= 3:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def circular_hue_interval(hues: np.ndarray, margin: int = 0) -> list[tuple[int, int]]:
    """把一堆色相压成**最少段**的闭区间，自动处理红色跨 0°。

    做法：排序后复制一份 +180 的序列，找"覆盖 95% 样本的最短弧"。返回的区间
    已含 ``margin`` 外扩；跨越 0° 时拆成两段（OpenCV 的 H 是 0..179 的环）。
    """
    vals = np.sort(np.asarray(hues, dtype=np.int64).ravel() % 180)
    n = int(vals.size)
    if n == 0:
        return []
    k = max(1, int(round(0.95 * n)))
    ext = np.concatenate([vals, vals + 180])
    best_w = 10 ** 9
    best_start = 0
    for i in range(n):
        w = int(ext[i + k - 1] - ext[i])
        if w < best_w:
            best_w, best_start = w, int(ext[i])
    if best_w >= 170:                       # 几乎整圈 → 不要假装有色相先验
        return [(0, 179)]
    width = best_w + 2 * int(margin)
    start = best_start - int(margin)
    lo, hi = start, start + width
    if lo < 0 and hi < 180:
        return [(0, hi), (lo + 180, 179)]
    if hi >= 180 and lo >= 0:
        return [(lo, 179), (0, hi - 180)]
    if lo < 0 and hi >= 180:
        return [(0, 179)]
    return [(lo, hi)]


def hsv_ranges_from_pixels(pixels: np.ndarray, *,
                           margin_h: int = 6,
                           margin_sv: int = 22) -> list[tuple[np.ndarray, np.ndarray]]:
    """一组 HSV 像素 → 若干段可直接喂给 ``cv2.inRange`` 的区间。

    H 用 :func:`circular_hue_interval`（95% 分位，抗高光/阴影的少量离群）；
    S/V 只取**下界**（低分位减余量），上界一律 255 —— 球面明暗差本来就大，
    压上界只会把亮侧切掉。
    """
    px = np.asarray(pixels, dtype=np.uint8).reshape(-1, 3)
    if px.shape[0] < 8:
        return []
    hues = px[:, 0]
    s_lo = max(0, int(np.percentile(px[:, 1], 5)) - int(margin_sv))
    v_lo = max(0, int(np.percentile(px[:, 2], 5)) - int(margin_sv))
    out = []
    for h_lo, h_hi in circular_hue_interval(hues, margin_h):
        out.append((np.array([h_lo, s_lo, v_lo], dtype=np.uint8),
                    np.array([h_hi, 255, 255], dtype=np.uint8)))
    return out


def dominant_hsv_ranges(frame: np.ndarray, *,
                        margin_h: int = 6,
                        margin_sv: int = 22) -> list[tuple[np.ndarray, np.ndarray]] | None:
    """从整幅画面估"主色"的 HSV 区间；没有足够彩色像素时返回 ``None``。

    只看**有饱和度**的像素（S≥60）：白墙/灰地/黑背景的 S 很低，天然被排除。
    色相直方图按 S 加权后找峰，再向两侧扩到峰值的 1/4 —— 这样同一种颜色的
    明暗变化会落在同一段里，而"另一个颜色的物体"不会。

    这是 ``--preset auto`` 的核心：不需要点选，也不需要事先知道球色。
    """
    if frame is None or frame.size == 0:
        return None
    h, w = frame.shape[:2]
    scale = min(1.0, _AUTO_WIDTH / float(max(w, 1)))
    small = frame if scale >= 1.0 else cv2.resize(
        frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    sel = (sat >= 60) & (val >= 40)
    total = float(sel.size)
    if float(np.count_nonzero(sel)) < max(64.0, 0.002 * total):
        return None

    hues = hsv[:, :, 0][sel].astype(np.int64)
    weights = sat[sel].astype(np.float64) / 255.0
    hist = np.bincount(hues, weights=weights, minlength=180).astype(np.float64)
    # 环形平滑：一个色相单独一根柱可能是噪声，要看邻域
    kernel = np.ones(5, dtype=np.float64) / 5.0
    smooth = np.convolve(np.concatenate([hist[-2:], hist, hist[:2]]),
                         kernel, mode="same")[2:182]
    peak = int(np.argmax(smooth))
    if smooth[peak] <= 1e-9:
        return None
    lo, hi = peak, peak
    for _ in range(90):
        left = smooth[(lo - 1) % 180]
        right = smooth[(hi + 1) % 180]
        if left < 0.25 * smooth[peak] and right < 0.25 * smooth[peak]:
            break
        if left >= right and left >= 0.25 * smooth[peak]:
            lo = (lo - 1) % 180
        elif right >= 0.25 * smooth[peak]:
            hi = (hi + 1) % 180
        else:
            break
        if (hi - lo) % 180 > 70:            # 扩太宽 = 没有明确的"主色"
            return None
    in_hue = np.zeros(180, dtype=bool)
    idx = lo
    while True:
        in_hue[idx] = True
        if idx == hi:
            break
        idx = (idx + 1) % 180
    picks = in_hue[hues] & (sat[sel] >= 40) & (val[sel] >= 30)
    px = hsv[sel][picks]
    return hsv_ranges_from_pixels(px, margin_h=margin_h, margin_sv=margin_sv)


# --------------------------------------------------------------------------- #
class ColorBlobDetector(BaseDetector):
    """HSV 阈值分割 + 形态学 + 轮廓圆度筛选。

    这是"传统方法"里性价比最高的方案：只要球的颜色和背景能分开，
    它的速度和精度都远超 Canny / 霍夫圆，而且天然给出球的边缘轮廓。
    """

    name = "color"

    def __init__(self, cfg: DetectorConfig) -> None:
        super().__init__(cfg)
        self._last_mask: np.ndarray | None = None
        self.last_error = ""
        self.last_warning = ""
        self._diag: dict = {}
        # ---- 人工种子 ----
        self._seed: tuple[float, float, float] | None = None
        self._seed_frame = 0
        self._seed_ranges: list[tuple[np.ndarray, np.ndarray]] | None = None
        # 取色判"不可信"后**只试一次**：否则每帧都会拿球已经离开的种子位置重新
        # 取色，永远在背景上取，日志刷屏、还可能某个巧合帧把背景色当球色收下。
        self._seed_rejected = False
        self._seed_reject_note = ""
        self._frame_idx = -1
        self._anchor_done = False
        # ---- 帧间一致性 ----
        self._prev_center: tuple[float, float] | None = None
        self._prev_radius: float | None = None
        self._prev_frame_idx: int | None = None
        self._vel = (0.0, 0.0)
        # ---- 自动取色 ----
        self._auto_ranges: list[tuple[np.ndarray, np.ndarray]] | None = None
        self._auto_name = ""
        self._auto_at = -(10 ** 9)
        self._auto_misses = 0

    # ------------------------------------------------------------------
    def seed(self, x: float, y: float, radius: float, frame_index: int = 0) -> None:
        """人工种子。**第 frame_index 帧**从点击点周围取色，之后按半径带跟踪。

        与 `SpecularBallDetector.seed` 同签名，所以两个 app 的种子传递逻辑
        不用分支（`hasattr(detector, "seed")` 即可）。
        """
        self._seed = (float(x), float(y), float(radius))
        self._seed_frame = max(0, int(frame_index))
        self._frame_idx = -1
        self._seed_ranges = None
        self._seed_rejected = False
        self._seed_reject_note = ""
        self._anchor_done = False
        self._prev_center = None
        self._prev_radius = None
        self._prev_frame_idx = None
        self._vel = (0.0, 0.0)
        self.last_error = ""

    def reset(self) -> None:
        """换源时重置状态，但**保留**人工种子（与 specular 的约定一致）。"""
        self._frame_idx = -1
        self._seed_ranges = None
        # 换了源就该允许重试取色：上一段视频里"球不在那一帧"不代表这一段也是。
        self._seed_rejected = False
        self._seed_reject_note = ""
        self._anchor_done = False
        self._prev_center = None
        self._prev_radius = None
        self._prev_frame_idx = None
        self._vel = (0.0, 0.0)
        self._auto_ranges = None
        self._auto_name = ""
        self._auto_at = -(10 ** 9)
        self._auto_misses = 0

    # ------------------------------------------------------------------
    # 取色区间
    # ------------------------------------------------------------------
    def _effective_ranges(self):
        if self._seed_ranges is not None:
            return self._seed_ranges
        if self.cfg.color.mode == "auto":
            return self._auto_ranges
        return _ranges_from_config(self.cfg.color)

    def _seed_ranges_verdict(self, frame: np.ndarray, ranges):
        """判「这次取色像不像一颗球」：``None`` = 通过，否则 ``(code, note, stats)``。

        只看**颜色分布**，不做形状判断（形状是 `_detect_from_ranges` 的事）。
        两条判据都是组合条件 —— 单独任何一条都会误伤白球/浅色球：

        A. **球外邻环被同色占满**（`[1.35r, 2.5r]` 里同色像素 > 35%）。球不会把
           自己周围也染成同色；帧 40 实测 0.0001，背景帧 0.83~0.96。球与背景本来
           同色时也会落在这一条上 —— 那种场景颜色分割本来就分不开，文案里说清楚。
        B. **单段 + 色相跨度 ≥70 + S 下界 ≤8 + 整帧 >20%**。无彩色背景的典型形态：
           色相没有先验、饱和度几乎没有、还占住大半画面。帧 40 是两段且 S 下界 41
           （不满足前两项）；合成白球是 V 下界 213 把背景挡在区间外（不满足第二、
           四项，实测邻环 0.0000、整帧 0.0000），所以都不会被这条误伤。

        `stats` 里的数字全部进 `_diag`，供日志排查（阈值怎么定的见模块顶部常量）。
        """
        c = self.cfg.color
        assert self._seed is not None
        sx, sy, sr = self._seed
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = _mask_from_ranges(hsv, ranges, open_ksize=c.open_ksize,
                                 close_ksize=c.close_ksize)
        frame_cover = float(cv2.countNonZero(mask)) / float(max(1, mask.size))

        # 邻环只算球心周围 2.5r 的小窗：整幅 hypot 在 1080p 上是无谓开销
        r_out = _SEED_ANNULUS_HI * sr
        bx0 = max(0, int(math.floor(sx - r_out)))
        bx1 = min(w, int(math.ceil(sx + r_out)) + 1)
        by0 = max(0, int(math.floor(sy - r_out)))
        by1 = min(h, int(math.ceil(sy + r_out)) + 1)
        ann_cover = 0.0
        n_ann = 0
        if bx1 - bx0 >= 3 and by1 - by0 >= 3:
            yy, xx = np.ogrid[by0:by1, bx0:bx1]
            rr = np.hypot(xx - sx, yy - sy)
            ann = (rr >= _SEED_ANNULUS_LO * sr) & (rr <= r_out)
            n_ann = int(np.count_nonzero(ann))
            if n_ann > 0:
                ann_cover = (float(np.count_nonzero(mask[by0:by1, bx0:bx1][ann]))
                             / float(n_ann))

        spans = [int(b[0]) - int(a[0]) for a, b in ranges]
        h_span = max(spans) if spans else 0
        s_min = min(int(a[1]) for a, _b in ranges)
        stats = {
            "seed_annulus_cover": round(float(ann_cover), 4),
            "seed_frame_cover": round(float(frame_cover), 4),
            "seed_h_span": int(h_span),
            "seed_s_min": int(s_min),
        }
        if n_ann >= _SEED_ANNULUS_MIN_PIXELS and ann_cover >= _SEED_ANNULUS_COVER_MAX:
            return ("background", (
                f"种子取色落在背景上（或球与背景同色）：球外邻环 "
                f"{_SEED_ANNULUS_LO:.2f}~{_SEED_ANNULUS_HI:.2f}r 里同色像素占 "
                f"{ann_cover:.0%} —— 球不会把自己周围也染成同色。请重新点选球心，"
                f"并确认种子帧号就是球在画面里的那一帧；若球与背景本来就是同一色，"
                f"颜色分割分不开，改用 --preset auto / 几何方法，或换背景"), stats)
        if (len(ranges) == 1 and h_span >= _SEED_FLAT_HUE_SPAN
                and s_min <= _SEED_FLAT_S_MAX and frame_cover > _SEED_FRAME_COVER_MAX):
            return ("overwide", (
                f"种子取色区间过宽：单段色相跨度 {h_span}（整环只有 180）、S 下界只有 "
                f"{s_min}、占整帧 {frame_cover:.0%} —— 这是无彩色背景的形态，不是球色。"
                f"请重新点选球心，或改用 --preset / --preset auto"), stats)
        return None

    def _init_seed_ranges(self, frame: np.ndarray) -> None:
        """从点击点周围的**圆环**取色（避开中心高光与外圈混色）。

        取到的区间先过 `_seed_ranges_verdict`：判"不可信"时**不设** `_seed_ranges`
        （于是 `_effective_ranges()` 回落到预设/auto），把原因写进 `_diag` 与
        `last_error`。不静默接受、也不抛异常 —— 质量差不中止程序，但必须说出来。
        """
        if self._seed is None:
            return
        c = self.cfg.color
        sx, sy, sr = self._seed
        h, w = frame.shape[:2]
        x0 = max(0, int(math.floor(sx - sr)))
        x1 = min(w, int(math.ceil(sx + sr)) + 1)
        y0 = max(0, int(math.floor(sy - sr)))
        y1 = min(h, int(math.ceil(sy + sr)) + 1)
        if x1 - x0 < 3 or y1 - y0 < 3:
            return
        crop = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        yy, xx = np.ogrid[:crop.shape[0], :crop.shape[1]]
        rr = np.hypot(xx - (sx - x0), yy - (sy - y0))
        ring = (rr >= c.seed_sample_lo * sr) & (rr <= c.seed_sample_hi * sr)
        pixels = crop[ring]
        ranges = hsv_ranges_from_pixels(
            pixels, margin_h=c.seed_hsv_margin_h, margin_sv=c.seed_hsv_margin_sv)
        if not ranges:
            return
        verdict = self._seed_ranges_verdict(frame, ranges)
        if verdict is None:
            self._seed_ranges = ranges
            self._diag["seed_ranges"] = len(ranges)
            self._diag.update({"seed_ranges_rejected": 0})
            return
        code, note, stats = verdict
        self._seed_rejected = True
        self._seed_reject_note = note
        self.last_error = note
        self._diag.update({
            "seed_ranges": 0,
            "seed_ranges_rejected": 1,
            "seed_ranges_reject_code": code,
            "seed_ranges_reject_reason": note,
            **stats,
        })

    def _ensure_auto(self, frame: np.ndarray) -> None:
        """自动取色：锁定一个区间，避免每帧都跑七种候选。"""
        c = self.cfg.color
        if c.mode != "auto":
            return
        fresh = (self._auto_ranges is not None
                 and self._frame_idx - self._auto_at < c.auto_lock_frames
                 and self._auto_misses < c.auto_unlock_after)
        if fresh:
            return
        picked = self._select_auto(frame)
        if picked is not None:
            self._auto_name, self._auto_ranges = picked
            self._auto_at = self._frame_idx
            self._auto_misses = 0
        else:
            # 本轮一个都没选出来：**也要记时间**，否则 `fresh` 永远为假，
            # 之后每一帧都会重跑七种候选（拖慢整段，且日志刷屏）。
            self._auto_at = self._frame_idx
            if self._auto_ranges is None:
                # 从来没锁上过：**不假装成功**（2026-11-13 修 ②）。
                # 以前这里退回 `ColorConfig()` 的默认白球区间（S<=80），对
                # 高饱和红球（S≈117）恒 0 检出，掩膜还会吃掉约 42% 的背景，
                # `_auto_name="fallback:config"` 看着却像已锁定；用户照着
                # "掩膜占画面 40%" 去换背景，方向完全错。
                # 现在保持"无有效区间"：空列表 -> `_effective_ranges()` 返回它
                # -> `_mask_from_ranges` 得到空掩膜；`_auto_name`/`last_error`
                # 明说"未锁定"（见 `_diagnose`）。绝不改环形色相算法。
                self._auto_ranges = []
                self._auto_name = "auto:unlocked"

    def _select_auto(self, frame: np.ndarray):
        """在缩小的图上把六个预设 + 画面主色各试一遍，取形状得分最高的。"""
        h, w = frame.shape[:2]
        scale = min(1.0, _AUTO_WIDTH / float(max(w, 1)))
        small = frame if scale >= 1.0 else cv2.resize(
            frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        candidates: list[tuple[str, list]] = []
        for name in ColorConfig.preset_names():
            candidates.append((name, _ranges_from_config(ColorConfig.preset(name))))
        dom = dominant_hsv_ranges(
            frame, margin_h=self.cfg.color.seed_hsv_margin_h,
            margin_sv=self.cfg.color.seed_hsv_margin_sv)
        if dom:
            candidates.append(("auto:dominant", dom))

        best: tuple[float, str, list] | None = None
        for name, ranges in candidates:
            dets, mask = self._detect_from_ranges(small, ranges, do_subpixel=False,
                                                  radius_scale=scale)
            if not dets:
                continue
            cover = cv2.countNonZero(mask) / float(max(1, mask.size))
            quality = dets[0].score - (_COVER_PENALTY if cover > _MASK_COVER_WARN else 0.0)
            if best is None or quality > best[0]:
                best = (quality, name, ranges)
        if best is None:
            return None
        return best[1], best[2]

    # ------------------------------------------------------------------
    def build_mask(self, frame: np.ndarray) -> np.ndarray:
        """按当前策略建掩膜（公开接口：`calibrate_color.py` 也用它）。"""
        if self.cfg.color.mode == "auto" and self._seed_ranges is None:
            self._ensure_auto(frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ranges = self._effective_ranges()
        if ranges is None:                   # auto 首次调用且没锁上时
            ranges = _ranges_from_config(self.cfg.color)
        c = self.cfg.color
        mask = _mask_from_ranges(hsv, ranges, open_ksize=c.open_ksize,
                                 close_ksize=c.close_ksize)
        self._last_mask = mask
        return mask

    # ------------------------------------------------------------------
    def _detect_from_ranges(self, frame: np.ndarray, ranges, *,
                            do_subpixel: bool,
                            radius_scale: float = 1.0,
                            gray: np.ndarray | None = None):
        """在给定区间上跑一次完整流程。返回 ``(dets, mask)``。

        `radius_scale` 用于"在缩小的图上选区间"：此时像素半径按比例变小，
        尺寸门限必须跟着缩，否则小球会被当成噪点。
        `gray` 由调用方传入时省一次 ``cvtColor``（pipeline 每帧已算过一次）。
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        c = self.cfg.color
        mask = _mask_from_ranges(hsv, ranges, open_ksize=c.open_ksize,
                                 close_ksize=c.close_ksize)
        rmin = float(self.cfg.min_radius) * radius_scale
        rmax = float(self.cfg.max_radius) * radius_scale
        min_area = math.pi * rmin * rmin * 0.5
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        out: list[Detection] = []
        # 2026-11-08：把"被哪道门砍掉"记下来。原来只有一句"0 个候选过形状门"，
        # 梨子（r=205px）被尺寸门砍掉时那句提示说的是"球可能太小"——方向相反，
        # 用户照着它排查只会更远。这里记：面积合格的最大半径、以及最大那个
        # 候选是死在哪道门上（尺寸 / 圆度 / 填充率 / 径向 / 分数）。
        r_seen = 0.0
        r_pass = 0.0
        r_blocked = 0.0
        biggest: dict | None = None
        gate_counts = {"radius": 0, "circ": 0, "fill": 0, "radial": 0, "score": 0}
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            # 用"最小外接圆"算填充率：方块只能填满约 61%，球体能填满 90%+
            (_, _), r_enc = cv2.minEnclosingCircle(cnt)
            r_seen = max(r_seen, float(r_enc))
            first_gate = None
            if r_enc < rmin or r_enc > rmax:
                gate_counts["radius"] += 1
                r_blocked = max(r_blocked, float(r_enc))
                first_gate = "radius"
                circ = contour_circularity(cnt)
                fill = float(area / (math.pi * r_enc * r_enc)) if r_enc > 0 else 0.0
                radial = float("nan")
            else:
                r_pass = max(r_pass, float(r_enc))
                circ = contour_circularity(cnt)
                fill = float(area / (math.pi * r_enc * r_enc)) if r_enc > 0 else 0.0
                radial = float("nan")
                if circ < self.cfg.min_circularity or fill < self.cfg.min_fill_ratio:
                    gate_counts["circ" if circ < self.cfg.min_circularity else "fill"] += 1
                    first_gate = "circ" if circ < self.cfg.min_circularity else "fill"
            if (first_gate is not None
                    and (biggest is None or area > biggest["area"])):
                biggest = {"area": area, "r": float(r_enc), "circ": circ,
                           "fill": fill, "gate": first_gate}
            if first_gate is not None:
                continue
            pts = cnt.reshape(-1, 2)
            if pts.shape[0] >= 5:
                cx, cy, r = fit_circle_robust(pts)
            else:
                (cx, cy), r = cv2.minEnclosingCircle(cnt)
                cx, cy, r = float(cx), float(cy), float(r)
            if r < rmin or r > rmax:
                gate_counts["radius"] += 1
                r_blocked = max(r_blocked, float(r))
                if biggest is None or area > biggest["area"]:
                    biggest = {"area": area, "r": float(r), "circ": circ,
                               "fill": fill, "gate": "radius"}
                continue
            radial = radial_consistency(pts, cx, cy, r)
            score = self._score_candidate(r / radius_scale, circ, fill, radial)
            if score <= 0:
                gate_counts["score"] += 1
                if biggest is None or area > biggest["area"]:
                    biggest = {"area": area, "r": float(r), "circ": circ,
                               "fill": fill, "gate": "score"}
                continue
            out.append(
                Detection(
                    x=float(cx / radius_scale), y=float(cy / radius_scale),
                    radius=float(r / radius_scale), score=score,
                    method=self.name, contour=pts / radius_scale, mask=mask,
                    parts={"circularity": circ, "fill": fill, "radial": radial},
                )
            )
        self._diag.update({
            "r_seen": round(r_seen, 1),
            "r_pass": round(r_pass, 1),
            "r_blocked": round(r_blocked, 1),
            "image_h": int(frame.shape[0]),
            "r_gate": [round(rmin, 1), round(rmax, 1)],
            "gates": dict(gate_counts),
            "biggest": biggest,
        })
        out.sort(key=lambda d: d.score, reverse=True)
        if do_subpixel and self.cfg.subpixel_refine:
            if gray is None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for det in out[:3]:      # 只细化最有希望的几个，避免拖慢帧率
                rx, ry, rr = refine_circle_subpixel(
                    gray, det.x, det.y, det.radius, n_rays=self.cfg.subpixel_rays
                )
                if self.cfg.min_radius <= rr <= self.cfg.max_radius:
                    det.x, det.y, det.radius = float(rx), float(ry), float(rr)
                    det.parts["subpixel"] = 1.0
        return out, mask

    # ------------------------------------------------------------------
    def _anchor_detection(self, note: str = "") -> Detection:
        """种子帧的兜底：用户点过的地方直接作为检测输出。

        构造本身在 `BaseDetector.anchor_detection`（2026-11-07 收敛，见那里的
        注释）。这里只做本类独有的两件事：置 `_anchor_done`、把原因说清楚。
        """
        assert self._seed is not None
        x, y, r = self._seed
        self._anchor_done = True
        return self.anchor_detection(x, y, r, note=note)

    def _rank(self, dets: list[Detection], frame_idx: int) -> list[Detection]:
        """种子门 + 分数门 + 帧间一致性重排。"""
        c = self.cfg.color
        if self._seed is not None:
            sr = self._seed[2]
            lo, hi = sr * c.seed_radius_lo, sr * c.seed_radius_hi
            dets = [d for d in dets if lo <= d.radius <= hi]
        # `min_score` 以前只被 fusion 读，调它不改 color 的行为（B-178）。
        # 现在 color 也走同一道分数门；形状门（圆度/填充率/径向一致性）仍在
        # `_score_candidate` 里，两者是"及格线"与"形状像不像球"两层。
        dets = [d for d in dets if d.score >= self.cfg.min_score]

        if self._seed is not None and frame_idx == self._seed_frame:
            sx, sy, sr = self._seed
            gate = max(c.seed_gate_min_px, c.seed_gate_ratio * sr)
            n_before = len(dets)          # 过了半径带 + 分数门、还没按距离筛
            near = [d for d in dets
                    if math.hypot(d.x - sx, d.y - sy) <= gate]
            dets = near
            if not dets:
                # 说清"筛掉了几层"，否则这句和"掩膜是空的"听起来一模一样，
                # 用户照着排查会往错的方向走。
                return [self._anchor_detection(
                    note=(f"种子帧：半径带 {lo:.1f}~{hi:.1f} 与分数门 "
                          f"{self.cfg.min_score:.2f} 之后剩 {n_before} 个候选，"
                          f"都离点选位置超过 {gate:.0f}px；按 seed 语义返回"
                          f"点选位置，不拿门外候选顶替"))]

        if self._prev_center is not None and len(dets) > 1:
            px, py = self._prev_center
            gap = 1 if self._prev_frame_idx is None else max(
                1, frame_idx - self._prev_frame_idx)
            # 同 `_edge_follow`：最多按 3 帧的速度外推，不拿几十帧前的速度猜位置
            gap = min(gap, 3)
            pred = (px + self._vel[0] * gap, py + self._vel[1] * gap)
            best = dets[0].score
            # 只重排"分数并列"的那一档；分差明显时仍以分数为准
            eligible = [d for d in dets if d.score >= 0.92 * best]
            if len(eligible) > 1:
                eligible.sort(key=lambda d: (
                    math.hypot(d.x - pred[0], d.y - pred[1]) / max(d.radius, 1.0)))
                chosen = eligible[0]
                dets = [chosen] + [d for d in dets if d is not chosen]
        return dets

    def _edge_follow(self, gray: np.ndarray, pred, radius: float) -> Detection | None:
        """颜色失败时沿预测位置找球缘（不依赖颜色，任何球都有边缘）。"""
        c = self.cfg.color
        if not c.seed_edge_follow or self._seed is None:
            return None
        search = max(2.5, 0.30 * radius)
        nx, ny, nr, ok = refine_circle_subpixel_checked(
            gray, pred[0], pred[1], radius,
            n_rays=self.cfg.subpixel_rays, search=search)
        if not ok:
            return None
        if math.hypot(nx - pred[0], ny - pred[1]) > max(3.0, 0.35 * radius):
            return None
        if not (radius * c.seed_radius_lo <= nr <= radius * c.seed_radius_hi):
            return None
        return Detection(x=float(nx), y=float(ny), radius=float(nr), score=0.5,
                         method=f"{self.name}:edge",
                         parts={"edge_follow": 1.0},
                         extra={"edge_follow": True})

    def _zero_candidate_note(self, cover: float = 0.0) -> str:
        """0 候选时说清**是哪道门**，不再一律"球可能太小"（2026-11-08）。

        起因：梨子 r≈205px 被尺寸门（`max_radius=200`）砍掉，提示却说"球可能太小"，
        方向反了 —— 用户按那句话排查只会越走越远。现在按记录下来的
        `biggest.gate`（radius / circ / fill / score）分别给话，并带上实测数值。

        **先排掉"掩膜落在背景上"**：那种情况最大候选往往是墙/柜子（本例 r=367px），
        按"尺寸门"去建议"把相机拿远"是错的 —— 拿远只会让墙更完整。
        """
        big = self._diag.get("biggest") or {}
        gate = big.get("gate")
        lo, hi = self._diag.get("r_gate") or (self.cfg.min_radius, self.cfg.max_radius)
        r_seen = float(self._diag.get("r_seen") or 0.0)
        image_h = float(self._diag.get("image_h") or 0.0)
        big_r = float(big.get("r", 0) or 0)
        if (cover > 0.12 and image_h > 0 and big_r > 0.25 * image_h):
            return (f"掩膜落在背景上了：掩膜占画面 {cover:.0%}，最大连通域半径 "
                    f"{big_r:.0f}px（≈ {big_r / image_h:.2f}×画面高）—— 这不是尺寸门"
                    "的问题，拿远相机只会让背景更完整。先换 --preset / 缩取色区间，"
                    "或改用点选球心（seed）")
        if gate == "radius" and big_r > hi:
            return (f"尺寸门：最大候选半径 {big['r']:.0f}px 超过上限 {hi:.0f}px —— "
                    f"球离得太近或个头太大。把相机拿远，或用 --radius-auto 让尺寸门"
                    f"自动跳档，或手动 --max-radius {big['r'] * 1.2:.0f}")
        if gate == "radius" and float(big.get("r", 0)) < lo:
            return (f"尺寸门：候选半径 {big['r']:.0f}px 小于下限 {lo:.0f}px —— "
                    "球太小/太远，或那是一块噪点")
        if gate in ("circ", "fill"):
            return (f"形状门：最大候选 r={big['r']:.0f}px、圆度 {big['circ']:.2f}、"
                    f"填充率 {big['fill']:.2f} —— 不满足 圆度≥{self.cfg.min_circularity:.2f}"
                    f" / 填充率≥{self.cfg.min_fill_ratio:.2f}。这一颗不是球"
                    "（梨、椭球、被手挡掉一块都会这样）")
        if gate == "score":
            return (f"分数门：候选 r={big['r']:.0f}px 过不了 min_score="
                    f"{self.cfg.min_score:.2f} —— 形状勉强像球但不够好")
        if r_seen <= 0:
            return ("掩膜非空，但没有任何轮廓大过最小面积门 —— 球太小/太远，"
                    "或掩膜只是一些碎点（预设没选对）")
        return ("掩膜非空但 0 个候选过形状门（面积/圆度/填充率/径向一致性）："
                "球可能太小、出画超过 25%、或被同色表面粘住")

    def _diagnose(self, mask: np.ndarray, dets: list[Detection]) -> None:
        """把"静默 0 检出"变成一句能照着排查的话（B-177）。

        `last_error` 只在**没有候选**时写（调用方可以据此打印"这一帧为什么没球"）；
        `last_warning` 记"有候选但结果可疑"（掩膜把背景吃进来）。两者分开，
        避免把"有结果的警告"当成"失败"去处理。
        """
        cover = cv2.countNonZero(mask) / float(max(1, mask.size))
        self.last_warning = ""
        if not dets:
            if (cover <= 0.0 and self.cfg.color.mode == "auto"
                    and self._seed_ranges is None and not self._auto_ranges):
                # 锁不上就是锁不上：别把"没有有效区间"说成"球色与预设不匹配"。
                note = ("auto 未锁定：六个预设 + 画面主色都没给出可信候选，"
                        "当前没有有效 HSV 区间（掩膜为空）—— 点选球心(seed) "
                        "或显式 --preset <颜色>；不要退回默认白球区间假装成功")
            elif cover <= 0.0:
                note = ("颜色掩膜是空的：当前预设/取色区间里没有任何像素 —— "
                        "球色与预设不匹配（换个 --preset，或用 --preset auto / 点选球心）")
            elif cover > _MASK_COVER_WARN:
                note = (f"颜色掩膜占画面超过 {_MASK_COVER_WARN:.0%}：阈值很可能"
                        "把背景也吃进来了 —— 换背景/换预设，或缩小 max_radius")
            else:
                note = self._zero_candidate_note(cover)
            if self._seed_reject_note:
                # 取色被拒是**会话级**事实：不能因为这一帧恰好也没候选就被顶掉
                note = f"{self._seed_reject_note}；本帧仍无候选"
            # 具体数值只进 `_diag`，不进错误文案：文案要稳定，否则日志按字符串
            # 去重会失效（每帧的覆盖率都不一样）。
            self.last_error = note
        else:
            # 锚点是「人给的位置」，它的原因不能说没就没 —— `pipeline` 只在
            # `last_error` 非空时才会把「这一帧返回的其实是点选位置」说出口。
            # 2026-11-07 之前这里无条件清空，于是 color（塑料球 + 点选的生产
            # 路径）这一路的锚点静默了两轮才被 Critic 75c 抓出来。
            if not any((d.extra or {}).get("seed_anchor") for d in dets):
                self.last_error = ""
            if cover > _MASK_COVER_WARN:
                self.last_warning = (
                    f"颜色掩膜占画面超过 {_MASK_COVER_WARN:.0%}，"
                    "结果可能锁到了大面积同色背景")
            if self._seed_reject_note:
                # 取色被拒、靠预设/auto 检到了：也得让人知道球色**不是**从点选
                # 位置取的，否则他会以为"点选取色成功"（质量差可以出结果，不许瞒）
                self.last_warning = (f"{self._seed_reject_note}（已回落到预设/auto 区间）"
                                     + (f"；{self.last_warning}" if self.last_warning else ""))
        self._diag.update({
            "mask_cover": round(float(cover), 4),
            "n_candidates": len(dets),
            "mode": self.cfg.color.mode,
            "auto_range": self._auto_name,
            "has_seed": self._seed is not None,
            "seed_anchor": self._anchor_done,
            "seed_ranges": self._diag.get("seed_ranges", 0),
        })

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray, ctx=None) -> list[Detection]:
        self._frame_idx += 1
        if (self._seed is not None and self._seed_ranges is None
                and not self._seed_rejected
                and self._frame_idx >= self._seed_frame):
            self._init_seed_ranges(frame)

        if self.cfg.color.mode == "auto" and self._seed_ranges is None:
            self._ensure_auto(frame)
        ranges = self._effective_ranges()
        if ranges is None:
            ranges = _ranges_from_config(self.cfg.color)
        # 亚像素细化要用灰度图；ctx 里有就不重算（pipeline 每帧只算一次）
        gray = getattr(ctx, "gray", None) if ctx is not None else None
        if gray is None or gray.shape[:2] != frame.shape[:2]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets, mask = self._detect_from_ranges(frame, ranges, do_subpixel=True,
                                              gray=gray)

        dets = self._rank(dets, self._frame_idx)

        if (not dets and self._seed is not None
                and self._frame_idx > self._seed_frame
                and self._prev_center is not None and self._prev_radius):
            # 最多外推 3 帧：丢了几十帧还按老速度往前推，只会把搜索窗带到画面外，
            # 然后"越推越远、永远找不回"。
            gap = min(3, max(1, self._frame_idx - (self._prev_frame_idx
                                                    if self._prev_frame_idx is not None
                                                    else self._frame_idx)))
            pred = (self._prev_center[0] + self._vel[0] * gap,
                    self._prev_center[1] + self._vel[1] * gap)
            fallback = self._edge_follow(gray, pred, self._prev_radius)
            if fallback is not None:
                dets = [fallback]

        if dets:
            d = dets[0]
            if self._prev_center is not None and self._prev_frame_idx is not None:
                gap = max(1, self._frame_idx - self._prev_frame_idx)
                vx = (d.x - self._prev_center[0]) / gap
                vy = (d.y - self._prev_center[1]) / gap
                if gap <= 3:
                    self._vel = (0.35 * self._vel[0] + 0.65 * vx,
                                 0.35 * self._vel[1] + 0.65 * vy)
                else:
                    self._vel = (vx, vy)
            self._prev_center = (d.x, d.y)
            self._prev_radius = d.radius
            self._prev_frame_idx = self._frame_idx
            if self._seed is not None and self._frame_idx >= self._seed_frame:
                self._anchor_done = True
            if self.cfg.color.mode == "auto":
                self._auto_misses = 0
        elif self.cfg.color.mode == "auto":
            self._auto_misses += 1

        self._diagnose(mask, dets)
        return dets

    # ------------------------------------------------------------------
    def debug_snapshot(self) -> dict:
        return dict(self._diag, error=self.last_error, has_seed=self._seed is not None)
