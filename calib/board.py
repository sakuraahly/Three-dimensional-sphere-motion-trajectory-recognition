"""标定板定义与检测。

设计目标：为"自动标定"提供**可量化的质量指标**，让程序能自己判断
"这一帧该不该收"。只回答"检测到/没检测到"是不够的 ——
模糊的、贴边的、太小的、过曝的帧都会把标定结果拖坏。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class BoardSpec:
    """标定板规格。

    cols/rows 是**内角点**数量（不是方格数）。例如常见的 9x6 棋盘
    （10 列 9 行方格）内角点是 9x8。

    注意 findChessboardCorners 的参数顺序是 (列, 行)，
    即 pattern_size = (cols, rows)。

    构造时校验范围：3≤cols≤30、3≤rows≤30、5≤square_mm≤500，
    越界抛 ValueError —— 范围的唯一来源在这里，GUI/CLI 不再各抄一份。
    """

    cols: int = 8                 # 内角点列数
    rows: int = 5                 # 内角点行数
    square_mm: float = 25.0       # 方格实际边长（毫米）

    # 范围校验的**唯一来源**（2026-10-27 计划书 §3.1 ①）：此前 GUI 把同一套
    # 范围抄了 3 份、CLI 四处零校验，`--cols 31` 能直接构造出"检测永远
    # 返回 0 张"的板，且没有任何解释。钉在这里之后，所有构造点同时被守住。
    _RANGES = (("cols", 3, 30), ("rows", 3, 30), ("square_mm", 5.0, 500.0))

    def __post_init__(self) -> None:
        for name, lo, hi in self._RANGES:
            value = getattr(self, name)
            if not (lo <= value <= hi):
                raise ValueError(
                    f"BoardSpec.{name}={value} 超出允许范围 [{lo}, {hi}]"
                    f"（cols/rows 是内角点数，至少 3 才成格；square_mm 单位毫米）")

    @property
    def n_points(self) -> int:
        return self.cols * self.rows

    @property
    def pattern_size(self) -> tuple[int, int]:
        return (self.cols, self.rows)

    def object_points(self) -> np.ndarray:
        """棋盘格角点在"板坐标系"下的三维坐标（Z=0），单位与 square_mm 一致。

        用米做单位，方便标定结果直接和物理量对应。
        """
        s = self.square_mm / 1000.0
        objp = np.zeros((self.n_points, 3), np.float32)
        objp[:, :2] = np.mgrid[0 : self.cols, 0 : self.rows].T.reshape(-1, 2) * s
        return objp

    def unit_square_points(self) -> np.ndarray:
        """板平面上的归一化采样点（0..1），用来做位姿多样性度量。"""
        g = np.linspace(0.0, 1.0, 4)
        xs, ys = np.meshgrid(g, g)
        return np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)

    def describe(self) -> str:
        return f"棋盘格 {self.cols}x{self.rows} 内角点，方格 {self.square_mm:g} mm"


@dataclass
class BoardObservation:
    """一帧里对标定板的观测，附带质量指标。"""

    found: bool = False
    corners: np.ndarray | None = None       # (N,1,2) float32
    homography: np.ndarray | None = None    # 板平面(0..1)->图像 的 3x3 单应
    # 棋盘区域的 Laplacian 方差。**仅用于显示/日志**，不要拿它当门限：
    # 同样的模糊程度下，板在画面里越大，这个值越小（实测 240px 宽 ~4373、
    # 960px 宽 ~1017），阈值会随板大小漂移。判"糊不糊"请用 `focus`。
    sharpness: float = 0.0
    # 一个格子占多少像素（相邻内角点间距的中位数：横向取相邻列、纵向取相邻行，
    # 再取两者的中位）。不用"整体宽/列数"是因为斜视时梯形会被平均掉、偏大。
    square_px: float = 0.0
    # **尺度无关的清晰度** = sharpness × square_px。Laplacian 方差随格子像素数
    # 近似反比下降，乘回去就把"板大板小"消掉了 —— 同一模糊程度下几乎恒定
    # （实测 σ=0/0.8/1.6/2.5/4.0 -> ~71k/8.5k/1.5k/480/150）。
    # 所以门限（`CaptureConfig.min_sharpness`）只在 focus 上定义，不随板大小漂移。
    focus: float = 0.0
    area_ratio: float = 0.0                 # 棋盘凸包面积 / 图像面积
    margin_px: float = 0.0                  # 角点到图像边界的最小距离
    brightness: float = 0.0
    precise: bool = True                    # 角点是否来自全分辨率精确检测
    detector: str = ""                      # 角点来自谁：""（没找到）/ "sb" / "classic"
    note: str = ""                          # 未通过时的原因

    def quad(self) -> np.ndarray | None:
        """把板平面四角映射到图像，得到棋盘的四边形轮廓。"""
        if self.homography is None:
            return None
        unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(unit, self.homography).reshape(-1, 2)

    def pose_points(self) -> np.ndarray | None:
        """把归一化采样点映射到图像 —— 位姿多样性/覆盖率的统一表示。"""
        if self.homography is None:
            return None
        unit = BoardSpec().unit_square_points().reshape(-1, 1, 2)
        return cv2.perspectiveTransform(unit, self.homography).reshape(-1, 2)


def _sharpness(gray: np.ndarray, corners: np.ndarray | None) -> float:
    """棋盘区域的清晰度：Laplacian 方差。越小越模糊（运动模糊/失焦）。

    ⚠️ **这个值不是尺度无关的**：棋盘在画面里占的像素越少（板远/分辨率低），
    方格边缘越密，同一模糊程度下方差反而更大（实测同一模糊 σ=0.8：板宽 240px
    时 504、960px 时 125）。所以它只能当"显示用"的参考量，
    **判据必须用 `BoardObservation.focus`（本值 × 格子像素数）**。
    """
    if corners is None or len(corners) == 0:
        return 0.0
    x, y, w, h = cv2.boundingRect(corners.reshape(-1, 1, 2).astype(np.int32))
    pad = 4
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(gray.shape[1], x + w + pad), min(gray.shape[0], y + h + pad)
    roi = gray[y0:y1, x0:x1]
    if roi.size < 100:
        return 0.0
    return float(cv2.Laplacian(roi, cv2.CV_64F).var())


def _square_px(corners: np.ndarray, spec: BoardSpec) -> float:
    """一个方格占多少像素。用**相邻内角点间距的中位数**，不用整体宽/列数。

    为什么不用"整体宽/列数"：板一斜，近端格子大、远端格子小，整体宽除以列数
    拿到的是被梯形平均过的一个数，和棋盘区域里实际占主导的格子尺寸对不上；
    Laplacian 方差的尺度因子正是"格子的像素尺寸"，所以要用**局部**间距。
    """
    if corners is None:
        return 0.0
    pts = np.asarray(corners, np.float32).reshape(-1, 2)
    if pts.shape[0] != spec.n_points or spec.cols < 2 or spec.rows < 2:
        return 0.0
    grid = pts.reshape(spec.rows, spec.cols, 2)      # SB/经典都是行优先
    hgaps = np.linalg.norm(grid[:, 1:, :] - grid[:, :-1, :], axis=2).ravel()
    vgaps = np.linalg.norm(grid[1:, :, :] - grid[:-1, :, :], axis=2).ravel()
    hmed = float(np.median(hgaps)) if hgaps.size else 0.0
    vmed = float(np.median(vgaps)) if vgaps.size else 0.0
    if hmed <= 0.0 or vmed <= 0.0:
        return float(max(hmed, vmed))
    return float(np.median([hmed, vmed]))


def _finish(obs: BoardObservation, corners: np.ndarray, gray: np.ndarray,
            spec: BoardSpec, precise: bool, detector: str) -> BoardObservation:
    """由角点补齐单应、面积、边距、清晰度（sharpness/focus）、亮度等指标。

    `detector` 必须由调用方显式给出（`"sb"` / `"classic"`）——
    角点的**出身**不能丢，原因见 `detect_board()` 的文档串。
    """
    obs.found = True
    obs.precise = precise
    obs.detector = detector
    obs.corners = np.asarray(corners, np.float32).reshape(-1, 1, 2)
    h, w = gray.shape[:2]

    obj_xy = spec.object_points()[:, :2].astype(np.float32)
    # 归一化到**真正的单位正方形**：减最小值、除以跨度。
    #
    # 原来写的是 `obj_xy / np.array([cols - 1, rows - 1])`，那是错的：
    # `object_points()` 返回的已经是米制坐标（本机标定板 0..0.175 × 0..0.1），
    # 除以 (7, 4) 之后只剩 0..0.025；而目标点是像素坐标（600..900）。
    # 源点与目标点差 4~5 个数量级，DLT 严重病态 —— 角点 0.07 px 的检测噪声
    # 被放大成 14~86 px 的单应跳变。
    #
    # 后果有两个，都在真机上踩到了：
    #   1. `pose_distance()` 量到的其实是数值噪声：板明明静止，
    #      "停稳"门限（默认 3 px）却永远过不去 —— 实测 419 次误拒。
    #   2. `quad()` 用同一个单应画棋盘轮廓，会飞出画面几千像素
    #      （预览窗口里那条横贯画面的橙线）。
    #
    # 修好后 pose_distance 与角点位移几乎相等（实测 0.02~0.44 px），与门限量级一致。
    obj_min = obj_xy.min(axis=0)
    obj_span = obj_xy.max(axis=0) - obj_min
    obj_xy = (obj_xy - obj_min) / obj_span
    H, _ = cv2.findHomography(obj_xy, obs.corners.reshape(-1, 2), 0)
    obs.homography = H

    pts = obs.corners.reshape(-1, 2)
    hull = cv2.convexHull(pts.astype(np.float32))
    obs.area_ratio = float(cv2.contourArea(hull) / float(w * h))
    obs.margin_px = float(min(pts[:, 0].min(), pts[:, 1].min(),
                              w - 1 - pts[:, 0].max(), h - 1 - pts[:, 1].max()))
    obs.sharpness = _sharpness(gray, obs.corners)
    obs.square_px = _square_px(obs.corners, spec)
    # 尺度无关清晰度：两者都算得出来才有意义（缺失时保持 0，交给调用方按"不知道"处理）
    obs.focus = obs.sharpness * obs.square_px if (obs.sharpness and obs.square_px) else 0.0
    xs = np.clip(pts[:, 0].astype(np.int32), 0, w - 1)
    ys = np.clip(pts[:, 1].astype(np.int32), 0, h - 1)
    obs.brightness = float(gray[ys, xs].mean())
    return obs


def detect_board(
    gray: np.ndarray,
    spec: BoardSpec,
    use_sb: bool = True,
    normalize: bool = True,
    work_width: int | None = None,
    exhaustive: bool = True,
    allow_classic_fallback: bool = False,
) -> BoardObservation:
    """检测标定板并给出质量指标。

    性能提示（实测 1920x1080）：
        SB + EXHAUSTIVE + ACCURACY   约 300 ms   <- 精确但慢，只能用在"要收的那一帧"
        SB 不开 EXHAUSTIVE           约 93 ms
        缩到 640 宽再检测             约 12 ms    <- 适合每帧跑的粗筛
    EXHAUSTIVE 在**找不到板**时会穷举搜索，所以绝不能每帧开。
    要每帧跑就用 work_width 降分辨率 + exhaustive=False，拿到粗位姿做判断，
    真正收帧时再用 detect_board_in_roi() 在棋盘附近做精确检测
    （裁剪不损失分辨率，只是缩小搜索范围）。

    ⚠️ 关于 `allow_classic_fallback`（默认 **False**，别随手打开）
    ------------------------------------------------------------
    经典检测器 `findChessboardCorners` 与 SB **返回的角点顺序恰好整体反序**
    （实测：直接对应 RMS 323.112 px，整份反序后 0.020 px；分类正好落在 8x5 点阵的
    `rot180` 自同构上）。后果：

    - **单目标定对它免疫**（顺序反序仍是一组自洽对应，RMS -0.0066 vs 0.0704 px），
      所以"每路各自看起来都正常"；
    - **双目标定不免疫**：左右两路若一路 SB、一路经典，同名下标不是同一个物理角点
      → 错配点对 → R/T 垃圾。合成复现：整体 RMS 21~42 px、极线误差上万像素、
      基线被吹到真值的 8~15 倍。

    这个 bug 在真机上极难查，因为**没有报错、单目指标也好看**。所以：

    1. 默认**不许回退**（`False`）—— "从不回退"是可预测且安全的；
    2. 真要开，也只能在**单路诊断**里开，**绝不能**让它出现在标定收帧路径上；
    3. 一旦开了，`obs.detector` 会如实记成 `"classic"`，调用方**必须**核对
       （`AutoPairCollector.offer()` 会拒绝一对之内混用检测器）。

    这组判据钉在 `tests/selftest_board_corner_order.py` 里。
    """
    obs = BoardObservation()
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    full = gray
    scale = 1.0
    if work_width and gray.shape[1] > work_width:
        scale = work_width / float(gray.shape[1])
        gray = cv2.resize(gray, (work_width, max(1, int(round(gray.shape[0] * scale)))),
                          interpolation=cv2.INTER_AREA)

    corners = None
    detector = ""
    if use_sb and hasattr(cv2, "findChessboardCornersSB"):
        flags = cv2.CALIB_CB_NORMALIZE_IMAGE if normalize else 0
        if exhaustive:
            flags |= cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        ok, c = cv2.findChessboardCornersSB(gray, spec.pattern_size, flags=flags)
        if ok:
            corners = np.asarray(c, np.float32)
            detector = "sb"

    # 经典回退：**必须显式允许**。理由见上面的文档串 —— 它与 SB 的角点顺序整体反序，
    # 混进双目标定会静默产出垃圾解。不做成"SB 失败就自动试"。
    if corners is None and exhaustive and allow_classic_fallback:
        flags2 = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK
        if normalize:
            flags2 |= cv2.CALIB_CB_NORMALIZE_IMAGE
        ok, c = cv2.findChessboardCorners(gray, spec.pattern_size, flags=flags2)
        if not ok:
            obs.note = "未检测到棋盘"
            return obs
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
        corners = cv2.cornerSubPix(gray, c, (5, 5), (-1, -1), criteria)
        detector = "classic"

    if corners is None:
        obs.note = "未检测到棋盘"
        return obs
    if scale != 1.0:
        corners = corners / scale          # 换算回全分辨率坐标
    # 指标要在全分辨率图上算（清晰度、亮度才可比）
    return _finish(obs, corners, full, spec, precise=(scale == 1.0 and exhaustive),
                   detector=detector)


def detect_board_in_roi(
    gray: np.ndarray,
    spec: BoardSpec,
    hint: np.ndarray,
    margin_ratio: float = 0.5,
    margin_px: int = 28,
    allow_classic_fallback: bool = False,
) -> BoardObservation | None:
    """在"粗检到的棋盘"附近裁剪出一小块，做全分辨率精确检测。

    为什么要这一步：每帧都做全图精确检测要 300 ms，撑不住实时；
    但直接拿降分辨率粗检的角点去标定，精度又差 3 倍（640 宽下 0.5 px
    ≈ 全分辨率 1.5 px）。裁剪既不损失分辨率，又把搜索范围缩小到棋盘附近，
    实测一次约 40 ms —— 只在"真要收这一帧"时做，完全感觉不到。

    `allow_classic_fallback` 默认 **False**，理由同 `detect_board()`。
    返回的 `BoardObservation.detector` 会如实标出用的是哪一个。
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    pts = np.asarray(hint, np.float32).reshape(-1, 2)
    if len(pts) < 4:
        return None
    x0, y0, w, h = cv2.boundingRect(pts.astype(np.int32))
    pad_x = int(max(margin_px, w * margin_ratio))
    pad_y = int(max(margin_px, h * margin_ratio))
    gx0 = max(0, x0 - pad_x); gy0 = max(0, y0 - pad_y)
    gx1 = min(gray.shape[1], x0 + w + pad_x); gy1 = min(gray.shape[0], y0 + h + pad_y)
    crop = gray[gy0:gy1, gx0:gx1]
    if crop.size < 4000:
        return None
    obs = detect_board(crop, spec, exhaustive=True,
                       allow_classic_fallback=allow_classic_fallback)
    if not obs.found or obs.corners is None:
        return None
    obs.corners = (obs.corners.reshape(-1, 2) + np.array([gx0, gy0], np.float32)).reshape(-1, 1, 2)
    # 指标换成在全分辨率整图上算（detector 出身照原样带过来）
    return _finish(BoardObservation(), obs.corners, gray, spec, precise=True,
                   detector=obs.detector)


def detect_two_boards(gray: np.ndarray, spec: BoardSpec):
    """一张图里检出**两块**同名标定板（L 形 / 垂直双板靶标）。

    做法：先检第一块，把它占的凸包**填白**（棋盘的黑格被抹掉，检测器就不会
    在同一位置再报一次），再检第二块。返回 ``(obs_a, obs_b)`` 或 ``None``。

    这里**不做 A/B 语义分配** —— 哪一块是"左相机看到的 A"，要看它相对哪一路
    相机、落在谁的内参姿态簇附近，由调用方判定（`apps/calibrate_stereo.py`
    的 `_assign_ab`）。

    为什么放在库层（2026-11-04）：双板标定和"拍完先自检"必须用**同一份**
    检测逻辑。以前它只存在于 `calibrate_stereo.py` 里，自检工具想查"这张
    同框图到底能不能同时检出两块板"就得抄一遍——而抄出来的那份迟早会漂。

    已知边界：两块板在图上靠得太近（外扩 12% 的填白区互相重叠）时，第二块
    的角点会被抹掉，返回 ``None``。这不是 bug，是"这个拍摄位置不合格"——
    调用方应当报"没有同时检出两块板"，让用户重拍，而不是放宽检测。
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    first = detect_board(gray, spec, exhaustive=True)
    if not first.found or first.corners is None:
        return None
    masked = gray.copy()
    pts = first.corners.reshape(-1, 2).astype(np.float32)
    center = pts.mean(axis=0)
    grown = ((pts - center) * 1.12 + center).astype(np.int32)
    cv2.fillConvexPoly(masked, cv2.convexHull(grown), 255)
    second = detect_board(masked, spec, exhaustive=True)
    if not second.found or second.corners is None:
        return None
    if float(np.linalg.norm(first.corners.reshape(-1, 2)
                            - second.corners.reshape(-1, 2))) < 1.0:
        return None                                    # 同一块被检了两次
    return first, second


def pose_distance(h1: np.ndarray | None, h2: np.ndarray | None, image_diag: float) -> float:
    """两个位姿的差异（归一化到图像对角线）。

    做法：把板平面上的一组采样点分别用两个单应映射到图像，
    取对应点的均方根距离。它同时包含了平移、缩放、旋转、倾斜，
    比单独比某一个量更能代表"这两个视角差得够不够远"。
    """
    if h1 is None or h2 is None:
        return 0.0
    unit = BoardSpec().unit_square_points().reshape(-1, 1, 2)
    p1 = cv2.perspectiveTransform(unit, h1).reshape(-1, 2)
    p2 = cv2.perspectiveTransform(unit, h2).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum((p1 - p2) ** 2, axis=1))) / max(image_diag, 1.0))
