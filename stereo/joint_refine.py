"""跨视图联合细化：用极线约束把左右圆心**一起**解出来（B-166）。

## 为什么不能各解各的

镜面球的可见轮廓弧只有整圈的 25%~35%（`B-130`）。拿这么短的一段弧去做圆拟合，
圆心在「弧的切线方向」上几乎是自由的 —— 弧往哪儿偏一点，拟合圆心就跟着滑。
实测（`B-165`，79 对双检出）：同一帧左右两路独立细化出的圆心，映射到校正坐标系后
**y 差中位 −28.1 px**；而同一对点由人手点选时只差 0.08 px。几何和标定都没问题，
是**圆心估计各偏一边**。

## 做法：共享 v，各解 u

极线校正之后，真正的同名点必须落在**同一行**（同一个 v）。所以不要问"左圆心在哪、
右圆心在哪"，而要问"哪个 v 让左右两边同时最像球心"。这就把一个二维病态问题
变成一维搜索：

    v 固定 -> 左右各自在自己的打分表上对 u 取 argmax -> 两侧分数相加
    -> 取总分最大的 v -> 两边共用这个 v

单看一侧，v 可能被一段弧"拖"得很远；两侧**共用一个 v** 之后，一段弧没有的信息
由另一段弧补上，交点才是球心。本项目实测的 28 px 差，根因就在这里。

## u 为什么要单独加个弱先验

极线约束只约束 v，**u（视差方向）没有任何跨视图约束**。而弧的切线方向若是沿 u
的，那一侧的 u 同样病态：打分表在 ±搜索半径内几乎是平的，argmax 就变成了"取噪声
最大处"。实测（1080p、r=70、12 点钟方向的弧）这种漂移能到 **+17 px** —— 而 u 偏
1 px 就是视差偏 1 px，深度直接偏掉几个百分点。所以 u 只允许在"有证据"的地方动：
以输入 u 为中心的弱二次先验，有信息时它可忽略，没信息时把人留在原处。
v 不需要这个先验 —— 它的信息来自**另一路**，那才是这个方法的意义。

## 打分表怎么来

对 ROI 求 Sobel 幅值图，再与**环核**（半径 r 的圆环，归一化成均值）做一次
`matchTemplate(TM_CCORR)`，得到"以任意点为圆心、半径 r 的环上平均梯度"的打分表。
一次卷积拿到所有候选中心的分，比 72 条射线逐点采样快两个数量级。
用 `matchTemplate` 而非 `filter2D`：两者取值逐位相同（实测 max diff 0.0），
但前者只输出合法中心的那个 50x50 网格且快约 1.7 倍（1080p 一对 ~2.5 ms）。

## 代码里的两条硬约定

- **不整帧 remap**：只对球周围一个小窗口校正。窗口的**行范围两侧完全相同**
  （见 `refine_pair` 里的 `row0/row1`），所以两边的打分表逐行对齐，v 就是同一个 v；
  这是本模块防止"两边 v 口径错位"的唯一机制，改窗口定义前先看那段。
- **半径一律用输入值**，绝不从边缘重估（`B-130` 已定：镜面球半径量不准）。
"""

from __future__ import annotations

import logging
import math
import weakref
from dataclasses import dataclass

import cv2
import numpy as np

from calib.params import StereoRig
from sphere_tracker.types import Detection
from stereo.triangulate import rectify_point_left, rectify_point_right

_LOGGER = logging.getLogger(__name__)

#: 默认搜索半径 = clip(0.35*radius, 6, 60) px。
#:
#: 0.35 的来处：`B-165` 的搜索窗扫描里，20/40/60 px 三档的中位 dy 依次是
#: −23.1/−12.8/−13.0 px，但窗口给到 60 时开始把背景吸进来（contour 占比掉到 72%）。
#: 真正要解决的是"两侧各偏一边"，不是"窗口不够大"，所以这里只给到能覆盖
#: 检测框中心偏移的量级，剩下的交给 v 的联合搜索。
DEFAULT_SEARCH_FACTOR: float = 0.35
SEARCH_MIN_PX: float = 6.0
SEARCH_MAX_PX: float = 60.0

#: 轮廓支撑（覆盖率 0..1）下限。
#:
#: 0.12 是"真球最差也该有一成多射线有轮廓"的宽松界，但独立验证（`joint-eval`）
#: 在坏几何会话上抓到它**放过了假点**：球被框在木门框/手机上时支撑 0.17 / 0.18
#: 照样通过，三角化出来的 Z≈0.22 m 看起来完全正常。而真球在人手确认处的支撑是
#: 左 0.271 / 右 0.333（健康会话中位）。所以门限必须落在两簇之间：
#: 假点 ≤0.18、真球 ≥0.27 → 取 0.22。
#: 代价写在前面：健康会话里支撑偏低的那部分帧会被一起拒掉（宁可少配，不造假点）。
#: 这个界只有两段录像的样本，**仍需要更多会话校准**，不要当成已定论的物理常数。
DEFAULT_MIN_SUPPORT: float = 0.22

#: 共享 v 之后**保留**的轮廓支撑比例下限（两侧都要满足）。
#:
#: 这条是防"自证"的：共享 v 的定义就是把极线残差压到 0，拿"dv 变小了"当采纳依据
#: 等于把结论当前提 —— 左路是球、右路是灯管的一对，也会被拉到某个 v 上然后过门。
#: 只有跟约束无关的量才作数：每侧**自己**的最优 (u,v) 处有多少轮廓支撑，
#: 共享 v 之后还剩多少。真是同一颗球的同一条轮廓，共享不会丢多少；
#: 丢得多就说明有一侧根本不在球上。
DEFAULT_MIN_SUPPORT_KEEP: float = 0.85

#: v 差至少要改善这么多才值得动坐标。0.5 px 是"比检测噪声小、没有采纳价值"的界；
#: 小于它时采纳只会把两个数改得更不确定，不如原样返回。
DEFAULT_MIN_GAIN_PX: float = 1.0

#: 半径下限：再小的话环核只有几个像素，打分表基本是噪声。
MIN_RADIUS_PX: float = 5.0

# ---- 环核 ---------------------------------------------------------------
#: 环带宽下限 2 px：r 小时环上仍要有足够像素取平均，否则分数被量化噪声主导。
_RING_BAND_MIN = 2.0
#: 带宽随半径走（8%）：r=70 时 5.6 px。
#:
#: 不能更窄：镜面球的半径量不准（`B-130` 只有 25%~35% 的轮廓，半径的相对误差
#: 在 10% 量级），环太窄就等于假设"半径是对的"，分数会随着半径抖动整体塌掉。
_RING_BAND_FRAC = 0.08

# ---- 支撑判据 -----------------------------------------------------------
#: 射线上取到的径向梯度最大值，要比"这块 ROI 的底噪"强多少倍才算"这条射线有轮廓"。
#: 底噪用 Sobel 幅值的**中位数**（ROI 内取样）—— 真实画面里镜面球内部满是反射纹理，
#: 绝对阈值分不出球心，只能跟这块区域自己的底噪比。
_SUPPORT_GRAD_K = 3.0
#: 绝对下限：全黑/全白画面里底噪≈0，没有它的话一点 JPEG 噪声都能"过门"。
_SUPPORT_GRAD_MIN = 8.0
#: 环上射线数。96 的角分辨率 3.75°，对 r>=5 px 的球仍是亚像素级间距。
_SUPPORT_ANGLES = 96
#: 支撑的径向采样带宽：max(3 px, 15% 半径)。
#:
#: 为什么必须是一个**带**而不是"正好在 r 那一圈"：镜面球的半径只有 25%~35% 的
#: 轮廓可量，相对误差 10%~20%（`B-130`），再加上圆心本身 1~2 px 残差、真机相机
#: 把球边缘散焦摊成 5~8 px 过渡带 —— 死卡 r 会把真边缘整圈漏掉。
#: 实测（真实种子帧 20261101-205709 L104）：死卡 r 得 0.03~0.05（全拒），
#: 改成 r ± max(3, 0.15r) 取径向最大值后得 0.30 / 0.91（真球），纯噪声仍是 0.00。
_SUPPORT_BAND_FRAC = 0.15
_SUPPORT_BAND_MIN = 3.0
#: 径向采样步长（px）。0.5 是为了不错过被散焦摊开的软边，1.0 会漏峰。
_SUPPORT_STEP_PX = 0.5

# ---- u 方向的弱先验 -----------------------------------------------------
#: 先验宽度：0.25*搜索半径（至少 2 px）。
#: 物理含义是"检测器给的 u 可信到这个量级"（CSRT 框中心 / 轮廓质心都在 2~5 px 内）。
_U_PRIOR_SIGMA_FRAC = 0.25
#: 偏离 1 个先验宽度时扣掉峰值的比例。10% 足以压住"平顶上的噪声漂移"，
#: 又远小于有信息一侧 1 px 内的分数落差（实测几十个百分点），不会拖偏好解。
_U_PRIOR_REL = 0.10

#: 校正映射缓存条数。一帧一对只算一次，缓存是为了跨帧复用；
#: 上限 8 条足够覆盖"左右 x 一两个分辨率"的常见用法。
_MAP_CACHE_MAX = 8
_MAP_CACHE: dict[tuple, tuple] = {}
#: 环核缓存（半径基本不变，但每帧重算一个 153x153 的核要 0.4 ms）。
_KERNEL_CACHE: dict[float, tuple[np.ndarray, int]] = {}
_KERNEL_CACHE_MAX = 16


@dataclass
class JointRefineResult:
    """一次跨视图联合细化的结果。

    `left` / `right` 是**原始像素坐标**：未采纳时**原样**返回输入（不做任何微调），
    所以调用方可以放心把它当"检测结果的可能改进版"直接替换。
    `dv_before` / `dv_after` 都是"校正坐标系下两路 y 之差的绝对值"，
    **同一把尺**，可以直接相减看改善。`radius` 不在返回值里 —— 半径不参与细化。
    """

    left: tuple[float, float]
    right: tuple[float, float]
    dv_before: float
    dv_after: float
    applied: bool
    reason: str
    support_left: float = 0.0
    support_right: float = 0.0
    #: 该侧**独立最优**时的支撑（各自取自己的 argmax，不受共享 v 约束）。
    #: 唯一用途是算 `support / own` 这个**保留率** —— 共享 v 一定让某侧离开它
    #: 最舒服的位置，掉了多少支撑是这方法付的代价，也是唯一与约束无关的独立量。
    support_left_own: float = 0.0
    support_right_own: float = 0.0


# --------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------
def _as_gray(img: np.ndarray | None) -> np.ndarray | None:
    """把任意常见格式转成单通道灰度；转不了返回 None。

    调用方按约定应直接给灰度图；这里兜一层是因为"三通道传进来"只会让
    Sobel 变成三通道、后续打分静默算成 (H,W,3)，而 argmax 还能跑通 ——
    错得无声无息。
    """
    if img is None:
        return None
    arr = np.asarray(img)
    if arr.size == 0:
        return None
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if arr.shape[2] == 1:
            return arr[:, :, 0]
        if arr.shape[2] == 3:
            return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        if arr.shape[2] == 4:
            return cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY)
    return None


def _bilinear(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """在 img 上按浮点坐标双线性取值（越界钳到边界）。

    不用 `cv2.remap`：这里每次只取几十个点，走 remap 要先造 map 数组，
    开销比取值本身还大。
    """
    h, w = img.shape[:2]
    x = np.clip(np.asarray(xs, np.float64), 0.0, w - 1.0)
    y = np.clip(np.asarray(ys, np.float64), 0.0, h - 1.0)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = x - x0
    fy = y - y0
    return (img[y0, x0].astype(np.float64) * (1.0 - fx) * (1.0 - fy)
            + img[y0, x1].astype(np.float64) * fx * (1.0 - fy)
            + img[y1, x0].astype(np.float64) * (1.0 - fx) * fy
            + img[y1, x1].astype(np.float64) * fx * fy)


def _rectify_maps(rig: StereoRig, side: str, size: tuple[int, int]):
    """(side, 图像尺寸) 对应的校正映射，按 `id(rig)` 缓存。

    ## 为什么不用 `rig.rectify_maps()`

    那个入口只按 `rig.image_size` 生成映射。若送进来的帧尺寸与标定记录不一致，
    映射会按旧尺寸建、`remap` 却按新尺寸索引 —— 结果是"静默地整体错位"，
    比报错难查得多。这里按**实际帧尺寸**建映射。

    ## 为什么用 `rig._side_target / _undistort_target`

    `P` 的 3x3 块在 `none_raw` 下是 `K2·R` 而不是新内参（见
    `calib/params.py::_undistort_target`）。自己去拼 `R1/P1` 就等于把口径
    写死成 parallel，右图会被多转一次 R。口径**只允许有一个来源**，
    所以哪怕那是私有方法也照用 —— 复制一份公式才是真的错。
    """
    w, h = int(size[0]), int(size[1])
    key = (side, w, h, id(rig))
    hit = _MAP_CACHE.get(key)
    if hit is not None:
        ref, map1, map2 = hit
        if ref is not None and ref() is rig:
            return map1, map2
        _MAP_CACHE.pop(key, None)

    K, D, R, P = rig._side_target(side)                      # noqa: SLF001
    R, P = rig._undistort_target(R, P, K)                    # noqa: SLF001
    map1, map2 = cv2.initUndistortRectifyMap(K, D, R, P, (w, h), cv2.CV_32FC1)

    try:
        ref = weakref.ref(rig)
    except TypeError:                                        # 不能被弱引用就别缓存
        return map1, map2
    if len(_MAP_CACHE) >= _MAP_CACHE_MAX:
        _MAP_CACHE.pop(next(iter(_MAP_CACHE)), None)
    _MAP_CACHE[key] = (ref, map1, map2)
    return map1, map2


def _ring_band(radius: float) -> float:
    """环核带宽：至少 2 px（r 小时环上也得有足够像素取平均），大球按 8% 走。"""
    return max(_RING_BAND_MIN, _RING_BAND_FRAC * radius)


def _ring_kernel(radius: float) -> tuple[np.ndarray, int]:
    """半径 `radius` 的环核 + 半边长。返回 (kernel, half)，`kernel` 边长 2*half+1。

    核的**边长必须盖住半径**（`2*ceil(r+band)+1`）：只按带宽开方窗是常见错，
    那时核里一个 1 都没有，归一化的分母为 0，卷积会退化成"原图照抄"，
    打分表看着有值、其实只是中心点自己的梯度 —— 静默失效。
    """
    key = round(float(radius), 2)
    hit = _KERNEL_CACHE.get(key)
    if hit is not None:
        return hit
    band = _ring_band(radius)
    half = int(math.ceil(radius + band))
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float32)
    dist = np.hypot(xx, yy)
    mask = (dist >= radius - band) & (dist <= radius + band)
    total = float(mask.sum())
    kern = mask.astype(np.float32) / total if total > 0.0 else np.ones((1, 1), np.float32)
    if len(_KERNEL_CACHE) >= _KERNEL_CACHE_MAX:
        _KERNEL_CACHE.pop(next(iter(_KERNEL_CACHE)), None)
    _KERNEL_CACHE[key] = (kern, half)
    return kern, half


def _grad_tables(roi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ROI 的 (gx, gy, 幅值)。幅值用 float32，环核卷积直接用。"""
    gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
    return gx, gy, cv2.magnitude(gx, gy)


def _support_band(radius: float) -> float:
    """支撑的径向采样带宽：max(3 px, 15% 半径)。"""
    return max(_SUPPORT_BAND_MIN, _SUPPORT_BAND_FRAC * radius)


def _ray_support(gx: np.ndarray, gy: np.ndarray, floor: float,
                 cx: float, cy: float, radius: float) -> tuple[float, float]:
    """射线上有多少比例的"看到轮廓"，以及这些射线的平均强度。返回 (支撑, 平均幅值)。

    ## 为什么是**沿射线取径向最大值**，不是"正好在 r 那一圈"

    第一版按半径 r 精确采样，真实画面上直接归零（实测种子帧 0.03~0.05，全拒）。
    原因是两个叠加：① 镜面球半径只有 25%~35% 的轮廓可量，相对误差 10%~20%
    （`B-130`）；② 真机相机的球边缘被散焦摊成 5~8 px 的过渡带，峰值根本不在
    r 上。改成 `r ± max(3, 0.15r)` 里取最大值之后，同一帧得 0.30 / 0.91（真球），
    纯噪声仍是 0.00 —— 分离度反而更好，因为它量的是"这条射线附近有没有边"，
    而不是"半径估计准不准"。

    ## 判据是三重

    ① **径向分量要强**：只认沿半径方向的梯度（切向梯度对"圆"没有意义）。
    ② **要比底噪强 K 倍**：底噪取 ROI 的 Sobel 幅值中位数（对真实画面很关键：
       镜面球内部满是反射纹理，绝对阈值分不出球心，只能跟这块区域自己的底噪比）。
    ③ **符号要一致**：真球轮廓在一段弧里要么全"由暗到亮"、要么全"由亮到暗"。

    平均幅值只作打分表的次排序（`support` 相同的那一片里挑幅值大的），
    不参与采纳判据。
    """
    if radius <= 1.0:
        return 0.0, 0.0
    ang = np.linspace(0.0, 2.0 * math.pi, _SUPPORT_ANGLES, endpoint=False)
    cos_a = np.cos(ang)
    sin_a = np.sin(ang)
    band = _support_band(radius)
    rr = np.arange(radius - band, radius + band + 1e-6, _SUPPORT_STEP_PX)
    if rr.size < 2:
        return 0.0, 0.0
    xs = np.clip(cx + np.outer(rr, cos_a), 0.0, gx.shape[1] - 1.0).astype(np.float32)
    ys = np.clip(cy + np.outer(rr, sin_a), 0.0, gx.shape[0] - 1.0).astype(np.float32)
    # 一次 remap 拿到 (径向采样, 射线) 整片，比逐条射线逐点取值快两个量级
    gxs = cv2.remap(gx, xs, ys, cv2.INTER_LINEAR)
    gys = cv2.remap(gy, xs, ys, cv2.INTER_LINEAR)
    gr = gxs * cos_a[None, :] + gys * sin_a[None, :]
    pick = np.argmax(np.abs(gr), axis=0)
    best = gr[pick, np.arange(len(ang))]                 # 每条射线上最强的那一点（带符号）
    thr = max(_SUPPORT_GRAD_K * float(floor), _SUPPORT_GRAD_MIN)
    strong = np.abs(best) >= thr
    if int(strong.sum()) < 3:
        return 0.0, 0.0
    sign = 1.0 if float(np.median(best[strong])) >= 0.0 else -1.0
    ok = strong & (best * sign > 0.0)
    n_ok = int(ok.sum())
    if n_ok == 0:
        return 0.0, 0.0
    return float(n_ok / len(ang)), float(np.abs(best[ok]).mean())


def _best_u(strip: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """每行在 u 方向的 argmax 列号与亚像素偏移。`strip` 形状 (nv, nu)。

    贴边时偏移取 0：抛物线在窗口外没有定义，硬插值只会更差。
    """
    nv, nu = strip.shape
    idx = np.argmax(strip, axis=1)
    rows = np.arange(nv)
    peak = strip[rows, idx].astype(np.float64)
    lo = strip[rows, np.maximum(idx - 1, 0)].astype(np.float64)
    hi = strip[rows, np.minimum(idx + 1, nu - 1)].astype(np.float64)
    denom = lo - 2.0 * peak + hi
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = np.where(np.abs(denom) > 1e-9, 0.5 * (lo - hi) / denom, 0.0)
    delta = np.where(np.isfinite(delta), delta, 0.0)
    delta = np.clip(delta, -0.5, 0.5)
    delta[(idx <= 0) | (idx >= nu - 1)] = 0.0
    return idx, delta


# --------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------
def refine_pair(
    gray_left: np.ndarray,
    gray_right: np.ndarray,
    det_left: Detection,
    det_right: Detection,
    rig: StereoRig,
    *,
    search_px: float | None = None,
    min_support: float = DEFAULT_MIN_SUPPORT,
    min_gain_px: float = DEFAULT_MIN_GAIN_PX,
    min_support_keep: float = DEFAULT_MIN_SUPPORT_KEEP,
) -> JointRefineResult:
    """用极线约束联合细化左右圆心（**共享 v，各解 u**）。

    参数
    ----
    gray_left / gray_right
        **单通道灰度**（调用方保证）。传三通道也能跑（内部 `cvtColor`），
        但那意味着调用方多做了一次无用的转换，1080p 上不值。两幅图尺寸必须一致。
    det_left / det_right
        检测结果，用到 `.x .y .radius`。**半径只当尺度用，不会被改**。
    rig
        `calib.params.StereoRig`。要求已做极线校正（`rectification` 不是
        `none_raw`）—— 未校正时"同一 v"这句话本身不成立，直接降级。
    search_px
        搜索半径（px）。默认 `clip(0.35*radius, 6, 60)`，`radius` 取两路均值。
    min_support
        两侧轮廓支撑下限，低于它不采纳。
    min_support_keep
        共享 v 之后支撑**保留率**的下限（两侧都要满足）。这是唯一的防自证判据：
        `dv_after` 是按定义压出来的，不能当成绩；能作数的是"共享 v 让每侧离开自己
        最舒服的位置之后，还剩下多少轮廓支撑"。掉太多 = 这两处根本不是同一条轮廓。
    min_gain_px
        v 差至少要改善这么多才采纳。

    返回
    ----
    :class:`JointRefineResult`。**未采纳时 `left/right` 是输入的逐位拷贝**
    （不是"算完发现不行就返回算出来的数"），`reason` 永远非空、中文、可读。
    """
    left_in = (float(det_left.x), float(det_left.y))
    right_in = (float(det_right.x), float(det_right.y))

    def _keep(reason: str, dv_before: float = float("nan"),
              sl: float = 0.0, sr: float = 0.0,
              ol: float = 0.0, orr: float = 0.0) -> JointRefineResult:
        return JointRefineResult(left=left_in, right=right_in,
                                 dv_before=dv_before, dv_after=dv_before,
                                 applied=False, reason=reason,
                                 support_left=sl, support_right=sr,
                                 support_left_own=ol, support_right_own=orr)

    try:
        if rig is None:
            return _keep("没有标定参数，无法建立极线约束")
        if str(getattr(rig, "rectification", "")).strip().lower() in (
                "none_raw", "not_applicable"):
            return _keep("该标定未做极线校正（none_raw），没有\"同一 v\"可言，"
                         "不做跨视图细化")

        gl = _as_gray(gray_left)
        gr = _as_gray(gray_right)
        if gl is None or gr is None:
            return _keep("灰度图为空或通道数不支持，未做联合细化")
        if gl.shape[:2] != gr.shape[:2]:
            return _keep(f"左右图尺寸不一致（{gl.shape[:2]} vs {gr.shape[:2]}），"
                         "无法按同一行对齐")

        rl = float(det_left.radius)
        rr = float(det_right.radius)
        if not (math.isfinite(rl) and math.isfinite(rr)) or min(rl, rr) < MIN_RADIUS_PX:
            return _keep(f"半径太小（左 {rl:.1f} / 右 {rr:.1f} px，下限 "
                         f"{MIN_RADIUS_PX:.0f}），环核不足以为圆心打分")

        vl0 = rectify_point_left(rig, *left_in)
        vr0 = rectify_point_right(rig, *right_in)
        dv_before = abs(vl0[1] - vr0[1])

        if search_px is None:
            radius_mean = 0.5 * (rl + rr)
            search_px = float(np.clip(DEFAULT_SEARCH_FACTOR * radius_mean,
                                      SEARCH_MIN_PX, SEARCH_MAX_PX))
        search_px = float(search_px)
        if not math.isfinite(search_px) or search_px <= 0.0:
            return _keep(f"搜索半径非法（{search_px}），未做联合细化", dv_before)

        h, w = gl.shape[:2]
        band_l = _ring_band(rl)
        band_r = _ring_band(rr)
        half_l = int(math.ceil(rl + band_l))
        half_r = int(math.ceil(rr + band_r))
        # 窗口边缘留出环核半径 + 2 px 余量
        rad_win = max(half_l, half_r) + 2

        v_mid = 0.5 * (vl0[1] + vr0[1])
        # ⚠ **行**范围两侧共用（同一起点、同一尺寸）：这是"两边的 v 是同一个 v"
        # 的全部保证 —— 两侧窗口的第 i 行都是整幅校正图的第 row0+i 行，
        # 两张打分表因此可以逐行相加。
        # **列**范围两侧各取各自的（差一个视差）：既不影响 v 对齐，
        # 又不会因为视差大就把窗口撑成一条长条（1080p 视差 100 px 时卷积面积翻倍）。
        row0 = max(0, int(math.floor(v_mid - search_px - rad_win)))
        row1 = min(h, int(math.ceil(v_mid + search_px + rad_win)) + 1)
        col0_l = max(0, int(math.floor(vl0[0] - search_px - rad_win)))
        col1_l = min(w, int(math.ceil(vl0[0] + search_px + rad_win)) + 1)
        col0_r = max(0, int(math.floor(vr0[0] - search_px - rad_win)))
        col1_r = min(w, int(math.ceil(vr0[0] + search_px + rad_win)) + 1)
        if (row1 - row0 < 4 or col1_l - col0_l < 4 or col1_r - col0_r < 4):
            return _keep("球太靠近图像边缘，校正窗口不够大，未做联合细化", dv_before)

        scores: dict[str, np.ndarray] = {}
        maps: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        grads: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for side, gray_full, c0, c1, radius in (("left", gl, col0_l, col1_l, rl),
                                                ("right", gr, col0_r, col1_r, rr)):
            map1, map2 = _rectify_maps(rig, side, (w, h))
            # ⚠ remap 的 src 必须是**整幅**原图：map 里存的是整幅图的坐标。
            # 传裁好的小块 = 索引错位，画面看起来"还在球附近"，只是整体平移了。
            roi = cv2.remap(gray_full, map1[row0:row1, c0:c1],
                            map2[row0:row1, c0:c1], cv2.INTER_LINEAR)
            if roi.size == 0:
                return _keep("校正映射生成失败，未做联合细化", dv_before)
            gx, gy, mag = _grad_tables(roi)
            kern, half = _ring_kernel(radius)
            if roi.shape[0] < kern.shape[0] or roi.shape[1] < kern.shape[1]:
                return _keep("球太靠近图像边缘，校正窗口装不下环核，未做联合细化",
                             dv_before)
            # TM_CCORR 与 filter2D 逐值相同，但只输出合法中心那片网格
            scores[side] = cv2.matchTemplate(mag, kern, cv2.TM_CCORR)
            grads[side] = (gx, gy, mag)
            maps[side] = (map1, map2)

        # 打分表 [i, j] 对应整幅校正图坐标 (row0+half_side+i, col0_side+half_side+j)。
        # ⚠ 两路的 half 不同（半径不同），所以**行号不能共用**：同一批全局 v 在两侧
        # 各有一个行偏移。下面按全局 v 取窗时每侧减自己的 half，取出的行数相同，
        # 逐行对齐的语义因此仍然成立。
        nh_l, nw_l = scores["left"].shape
        nh_r, nw_r = scores["right"].shape
        v_lo = row0 + max(half_l, half_r)
        v_hi = min(row0 + half_l + nh_l - 1, row0 + half_r + nh_r - 1)
        # ⚠ 候选 v 必须同时落在**两侧各自**的搜索窗里，不能只围着 `v_mid` 转。
        # `v_mid` 是两路 v 的平均：一侧若偏了 130 px（右路跟丢时的真实情况），
        # `v_mid` 会被它带着走，窗口跟着平移 —— 那不是"联合细化"，是"顺着错误一起漂"。
        # 交集写成 [max(v_l,v_r)-s, min(v_l,v_r)+s]：两路差得越远窗口越窄，
        # 超过 2×s 时窗口为空，正确地变成"这对不可能是同一颗球"。
        v_min = max(v_lo, int(math.ceil(max(vl0[1], vr0[1]) - search_px)))
        v_max = min(v_hi, int(math.floor(min(vl0[1], vr0[1]) + search_px)))
        ul_lo = col0_l + half_l
        ul_min = max(ul_lo, int(math.ceil(vl0[0] - search_px)))
        ul_max = min(ul_lo + nw_l - 1, int(math.floor(vl0[0] + search_px)))
        ur_lo = col0_r + half_r
        ur_min = max(ur_lo, int(math.ceil(vr0[0] - search_px)))
        ur_max = min(ur_lo + nw_r - 1, int(math.floor(vr0[0] + search_px)))
        if v_max - v_min < 2 or ul_max - ul_min < 1 or ur_max - ur_min < 1:
            if dv_before > 2.0 * search_px:
                # 这条比"贴边"常见得多，也更容易被误读成"算法在修"：
                # 右路跟丢时 v 差能到 200+ px，任何"平均一下"都会给出一个看起来
                # 合理的坐标。这里必须拒绝，并把量级说给用户看。
                return _keep(
                    f"两路校正后 v 差 {dv_before:.0f}px 已超过 2×搜索半径"
                    f"（{2.0 * search_px:.0f}px），不可能是同一颗球，未做联合细化",
                    dv_before)
            return _keep("球太靠近图像边缘，搜索范围不足，未做联合细化", dv_before)

        sigma_u = max(_U_PRIOR_SIGMA_FRAC * search_px, 2.0)
        strips: dict[str, np.ndarray] = {}
        for side, u_lo, u_hi, u0, c0, half in (
                ("left", ul_min, ul_max, vl0[0], col0_l, half_l),
                ("right", ur_min, ur_max, vr0[0], col0_r, half_r)):
            r0 = v_min - row0 - half
            r1 = v_max - row0 - half + 1
            c_lo = u_lo - c0 - half
            c_hi = u_hi - c0 - half + 1
            strip = scores[side][r0:r1, c_lo:c_hi].astype(np.float64)
            s_max = float(strip.max()) if strip.size else 0.0
            if s_max > 0.0:
                # u 方向弱先验：极线约束管不到 u，平顶上不能把噪声当证据（见模块头）
                us = np.arange(u_lo, u_hi + 1, dtype=np.float64)
                strip = strip - _U_PRIOR_REL * s_max * ((us - u0) / sigma_u) ** 2
            strips[side] = strip

        idx_l, d_l = _best_u(strips["left"])
        idx_r, d_r = _best_u(strips["right"])
        rows = np.arange(len(idx_l))
        total = strips["left"][rows, idx_l] + strips["right"][rows, idx_r]
        if not np.all(np.isfinite(total)) or float(total.max()) <= 0.0:
            return _keep("打分表全为 0（画面没有任何可用的轮廓梯度），未做联合细化",
                         dv_before)

        k = int(np.argmax(total))
        dv_sub = 0.0
        if 0 < k < len(total) - 1:
            lo, mid, hi = total[k - 1], total[k], total[k + 1]
            denom = lo - 2.0 * mid + hi
            if abs(denom) > 1e-9:
                dv_sub = float(np.clip(0.5 * (lo - hi) / denom, -0.5, 0.5))
        v_sel = v_min + k + dv_sub                  # 整幅校正图坐标下的同一个 v
        u_l = ul_min + idx_l[k] + float(d_l[k])     # 整幅校正图坐标
        u_r = ur_min + idx_r[k] + float(d_r[k])

        map1_l, map2_l = maps["left"]
        map1_r, map2_r = maps["right"]
        # 反查原图：map 就是"校正图坐标 -> 原图坐标"的映射表，双线性取样即精确反查。
        # 不用 undistortPoints 反解：那是同一套数学的另一次实现，
        # 两条路各算一遍就有了两个"真值"。
        xl = float(_bilinear(map1_l, u_l, v_sel))
        yl = float(_bilinear(map2_l, u_l, v_sel))
        xr = float(_bilinear(map1_r, u_r, v_sel))
        yr = float(_bilinear(map2_r, u_r, v_sel))
        if not all(math.isfinite(v) for v in (xl, yl, xr, yr)):
            return _keep("反查原图坐标失败（映射表给出非有限值），未做联合细化",
                         dv_before)

        # 口径一致地重算一次 v 差：把结果坐标再走一遍 rectify_point_*，
        # 与 dv_before 用的是同一个入口，不自己另立一把尺。
        dv_after = abs(rectify_point_left(rig, xl, yl)[1]
                       - rectify_point_right(rig, xr, yr)[1])

        gx_l, gy_l, mag_l = grads["left"]
        gx_r, gy_r, mag_r = grads["right"]
        floor_l = float(np.median(mag_l[::2, ::2]))
        floor_r = float(np.median(mag_r[::2, ::2]))
        # 共享 v 处的支撑（两侧各自算，位置不同）
        sup_l, amp_l = _ray_support(gx_l, gy_l, floor_l, u_l - col0_l, v_sel - row0, rl)
        sup_r, amp_r = _ray_support(gx_r, gy_r, floor_r, u_r - col0_r, v_sel - row0, rr)

        # 各自独立最优：该侧不受共享 v 约束时最舒服的那个位置。
        # 只用来算保留率 —— 拿"独立最优比共享更优"去否决，等于要求共享 v 不付任何
        # 代价，那不可能；要问的是"代价大不大"，见 DEFAULT_MIN_SUPPORT_KEEP。
        def _own_uv(side: str, u_min: int) -> tuple[float, float]:
            strip = strips[side]
            row = int(np.argmax(strip.max(axis=1)))
            col = int(np.argmax(strip[row]))
            return float(u_min + col), float(v_min + row)

        u_ol, v_ol = _own_uv("left", ul_min)
        u_or, v_or = _own_uv("right", ur_min)
        own_l, _ = _ray_support(gx_l, gy_l, floor_l, u_ol - col0_l, v_ol - row0, rl)
        own_r, _ = _ray_support(gx_r, gy_r, floor_r, u_or - col0_r, v_or - row0, rr)

        gain = dv_before - dv_after
        if min(sup_l, sup_r) < float(min_support):
            return _keep(
                f"轮廓支撑不足（左 {sup_l:.2f} / 右 {sup_r:.2f}，下限 "
                f"{min_support:.2f}）：分不清球和背景，圆心原样返回",
                dv_before, sup_l, sup_r, own_l, own_r)
        keep_l = sup_l / own_l if own_l > 1e-6 else 0.0
        keep_r = sup_r / own_r if own_r > 1e-6 else 0.0
        if min(keep_l, keep_r) < float(min_support_keep):
            return _keep(
                f"强制共享 v 会丢掉 {100.0 * (1.0 - min(keep_l, keep_r)):.0f}% 的轮廓支撑"
                f"（左 {own_l:.2f}->{sup_l:.2f} / 右 {own_r:.2f}->{sup_r:.2f}，"
                f"保留下限 {min_support_keep:.2f}），判为不是同一个球，圆心原样返回",
                dv_before, sup_l, sup_r, own_l, own_r)
        if gain < float(min_gain_px):
            return _keep(
                f"改善不足（|v 差| {dv_before:.2f} -> {dv_after:.2f} px，"
                f"仅 {gain:.2f} px < {min_gain_px:.2f} px），不值得动坐标，"
                f"圆心原样返回", dv_before, sup_l, sup_r, own_l, own_r)

        return JointRefineResult(
            left=(xl, yl), right=(xr, yr),
            dv_before=dv_before, dv_after=dv_after, applied=True,
            reason=(f"共享极线 v：|v 差| {dv_before:.2f} -> {dv_after:.2f} px；"
                    f"圆心分别移动 {_moved(left_in, (xl, yl)):.1f} / "
                    f"{_moved(right_in, (xr, yr)):.1f} px，"
                    f"轮廓支撑 {sup_l:.2f} / {sup_r:.2f}"
                    f"（各侧独立最优 {own_l:.2f} / {own_r:.2f}，保留 "
                    f"{100.0 * keep_l:.0f}% / {100.0 * keep_r:.0f}%）"),
            support_left=sup_l, support_right=sup_r,
            support_left_own=own_l, support_right_own=own_r)
    except (NameError, AttributeError, ImportError, TypeError,
            IndexError, KeyError):
        # 这几类**不是数据问题，是代码写错了**（引用了不存在的名字、签名对不上）。
        # 以前它们被统一 catch 成"计算异常…圆心原样返回"，看起来和"这段画面不行"
        # 一模一样，于是线上的真 bug 会一直藏着。这里显式抛出，让测试/走查当场看见。
        raise
    except Exception as exc:                                  # noqa: BLE001
        # 其余（cv2/numpy 的数值异常）退回原始坐标：联合细化是锦上添花，
        # 绝不能因为它把一帧的检测结果弄丢；但要用 warning 留痕，不能只进 debug。
        _LOGGER.warning("跨视图联合细化失败（%s: %s），本帧原样返回",
                        type(exc).__name__, exc, exc_info=True)
        dvb = locals().get("dv_before", float("nan"))
        return _keep(f"计算异常（{type(exc).__name__}: {exc}），圆心原样返回", dvb)


def _moved(a: tuple[float, float], b: tuple[float, float]) -> float:
    """两点距离。把「改了多少」直接说成像素数，读的人不用再做减法。"""
    return float(math.hypot(b[0] - a[0], b[1] - a[1]))


def describe_result(res: JointRefineResult) -> str:
    """一行中文摘要，直接可进日志/界面。"""
    head = "[已采纳]" if res.applied else "[未采纳]"
    tail = (f"左 ({res.left[0]:.1f}, {res.left[1]:.1f})  "
            f"右 ({res.right[0]:.1f}, {res.right[1]:.1f})")
    if not res.applied:
        tail += "（原样返回）"
    return f"{head}{res.reason}；{tail}"
