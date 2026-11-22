"""不依赖颜色的通用近圆球体检测器（几何优先，可选插件）。

## 为什么需要它

`color` 靠 HSV 掩膜。轻微色斑骗不过它（形状门仍会把水果捞回来），真正失效的是
**同色圆干扰物 + 带斑水果同框**：纯色干扰物的掩膜最完整、分数最高，于是颜色
范式把它排到第一，自报 score≈0.95、`last_error` 为空，圆心却离真值 200 px 以上
—— 一个「好看却无意义的数」。这正是计划书 §2/P1、P3 与 §11.7 Q3 要证伪的形态
（不是消融、也不是单球误差：形状门会把碎掩膜直接拦掉，不存在「高置信但单球偏心」）。

本插件**完全不读颜色**：Canny 边缘取闭合外轮廓，再用 ``geometry.py`` 里已验证的
现成函数编排打分，不重写任何算法：

* ``contour_circularity`` / ``fit_circle_robust`` / ``radial_consistency`` —— 形状；
* ``circle_radial_gradient_stats`` —— 「边界像不像球」判据：边缘梯度沿半径、
  且整圈符号一致。**它挡的是「阴阳半圆 / 半明半暗」这类符号混杂的假圆；
  挡不住实心平圆盘和描边圆**（2026-11-07 Critic 实测：那两者符号一致性都是 1.00）。
  所以本插件只证明「有一条圆形的强梯度边界」，**不证明「它是球」** ——
  同色同形多目标时如实输出多候选并标注 ``geometry_ambiguous``。
  **注意：下游目前并不消费这个标注**（``pipeline`` 仍取 ``detections[0]``，
  已登记为待办 B-184），所以现在只能说「标注了歧义」，**不能说「裁决了歧义」**；
* ``circle_edge_support`` —— 圆上确实落在边缘图上；
* ``refine_circle_subpixel_checked`` —— 亚像素圆心 / 半径。

## 阈值放在哪

全在本模块常量里，``DetectorConfig`` 一个字段都不加：加字段会同时撞
``selftest_config_wiring.py:144`` 的「死字段」与 ``:270`` 的
``fusion_members == ["color","hough","motion"]`` 硬断言（见计划书 §10 自审第 2 条）。

## 边界

合成机制验证、真机未验证。同色同形背景 + 多个近圆物体同时运动仍需时序裁决
（本轮不做）。近圆而非正圆：允许椭圆度约 0.9–1.1，但**保留径向一致性下限**，
避免把背景里任意圆形物体吸进来。
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .base import BaseDetector
from ..config import DetectorConfig
from ..geometry import (
    circle_edge_support,
    circle_radial_gradient_stats,
    contour_circularity,
    fit_circle_robust,
    radial_consistency,
    refine_circle_subpixel_checked,
)
from ..types import Detection

# ---- 本插件自己的阈值（不进 config.py）----------------------------------
CANNY_LOW = 40.0
CANNY_HIGH = 120.0
BLUR_KSIZE = 5
MIN_CONTOUR_POINTS = 24
MIN_CIRCULARITY = 0.55        # 与 config 同口径：真球实测 0.61~0.63
MIN_RADIAL = 0.90             # 径向一致性下限：挡方块/胶囊，不吸背景圆
# 「约 0.9–1.1」的门：给像素量化 / 抗锯齿留 ±0.03 余量（实测 fitEllipse 在
# 0.896~0.899 之间抖动，卡死 0.90 会随机漏检整帧）。圆度 + 径向一致性 +
# 梯度符号一致性仍在，放这点余量不会把细长目标或背景圆放进来。
ELLIPSE_RATIO_LO = 0.87
ELLIPSE_RATIO_HI = 1.13
MIN_GRADIENT_SUPPORT = 0.30   # 圆上沿半径方向梯度的最低比例
MIN_SIGN_CONSISTENCY = 0.60   # 梯度符号一致性：真球体整圈同号
MIN_EDGE_SUPPORT = 0.40       # 圆上落在 Canny 边缘图上的比例
GRAD_MIN_MAGNITUDE = 25.0
GRAD_MIN_ALIGNMENT = 0.55
REFINE_MIN_RATIO = 0.75       # 细化半径与原半径差太多就不采信（椭圆/锁错边）
REFINE_MAX_RATIO = 1.30

# ---- 可选「就这颗球」先验（seed）--------------------------------------
# 纯几何判据（圆度 / 径向一致性 / 梯度符号）**无法区分**两个同色近圆物体：
# 两者都合格。这时要么靠下游时序裁决，要么靠用户点选一次给出位置 + 半径带。
# 语义与 color / specular 的 seed 一致（空间门 + 半径带），但全部是模块常量，
# 不往 DetectorConfig 加字段。不给 seed 时插件如实输出全部候选并标注不可区分。
SEED_GATE_RATIO = 2.0
SEED_GATE_MIN_PX = 10.0
SEED_RADIUS_LO = 0.60
SEED_RADIUS_HI = 1.70


class GenericBallDetector(BaseDetector):
    """边缘 + 圆度 + 径向一致性；对任意近圆球体，不看颜色。"""

    name = "generic"

    def __init__(self, cfg: DetectorConfig) -> None:
        super().__init__(cfg)
        self._seed: tuple[float, float, float] | None = None
        # 0 检出时的原因。与 `color.last_error` 同口径：**不许静默**。
        # （2026-11-07 Critic 抓出：本插件原版一处都没有，漏检时下游只能猜。）
        self.last_error: str = ""

    def _anchor_detection(self, note: str = "") -> Detection:
        """种子门内一个候选都没有时的兜底：用户点过的地方直接作为输出。

        构造在 `BaseDetector.anchor_detection`（2026-11-07 收敛，全项目唯一
        的锚点构造点）。**关键：绝不把门外的候选（例如同色圆干扰物）当结果
        返回**，那正是本插件原本"默默占住 detections[0]"的那种失效。
        """
        assert self._seed is not None
        x, y, r = self._seed
        return self.anchor_detection(x, y, r, note=note)

    def seed(self, x: float, y: float, radius: float, frame_index: int = 0) -> None:
        """可选先验：点选一次「就这颗球」，给出位置门 + 半径带。

        不点选也能用（detect 会如实输出全部近圆候选）。这与 color / specular
        的 seed 同语义，fusion 会把种子透传给支持它的成员。
        """
        self._seed = (float(x), float(y), float(radius))

    def reset(self) -> None:
        self._seed = None

    def detect(self, frame: np.ndarray, ctx=None) -> list[Detection]:
        self.last_error = ""
        gray = getattr(ctx, "gray", None) if ctx is not None else None
        if gray is None or gray.shape[:2] != frame.shape[:2]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (BLUR_KSIZE, BLUR_KSIZE), 0)
        edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        n_edge = int(cv2.countNonZero(edges))
        n_cnt = len(contours)

        out: list[Detection] = []
        for cnt in contours:
            if len(cnt) < MIN_CONTOUR_POINTS:
                continue
            pts = cnt.reshape(-1, 2).astype(np.float64)
            circ = contour_circularity(cnt)
            if circ < MIN_CIRCULARITY:
                continue
            cx, cy, r = fit_circle_robust(pts)
            if not (self.cfg.min_radius <= r <= self.cfg.max_radius):
                continue
            radial = radial_consistency(pts, cx, cy, r)
            if radial < MIN_RADIAL:
                continue

            # 椭圆度门：长短轴比落在「约 0.9–1.1」才算近圆球体。
            # fitEllipse 是 OpenCV 现成函数，这里只做一次比值判断，不重写拟合。
            if len(cnt) >= 5:
                try:
                    (_ecx, _ecy), (d1, d2), _ang = cv2.fitEllipse(cnt)
                except cv2.error:
                    continue
                if min(d1, d2) <= 1e-6:
                    continue
                ratio = min(d1, d2) / max(d1, d2)
                if not (ELLIPSE_RATIO_LO <= ratio <= ELLIPSE_RATIO_HI):
                    continue
            else:
                ratio = 1.0

            # 「真物体边界」：梯度沿半径 + 整圈同号。纹理 / 描边圆 / 阴阳圆
            # 在这里符号混杂，会被挡掉（圆形干扰物的判据就钉在这一项）。
            support, consistency = circle_radial_gradient_stats(
                gray, cx, cy, r,
                min_magnitude=GRAD_MIN_MAGNITUDE,
                min_alignment=GRAD_MIN_ALIGNMENT)
            if support < MIN_GRADIENT_SUPPORT or consistency < MIN_SIGN_CONSISTENCY:
                continue
            edge_sup = circle_edge_support(edges, cx, cy, r)
            if edge_sup < MIN_EDGE_SUPPORT:
                continue

            # 亚像素细化：只在结果与粗拟合自洽时采信，否则保留粗拟合。
            fx, fy, fr, ok = refine_circle_subpixel_checked(
                gray, cx, cy, r, n_rays=self.cfg.subpixel_rays)
            if ok and REFINE_MIN_RATIO * r <= fr <= REFINE_MAX_RATIO * r:
                cx, cy, r = fx, fy, fr

            score = self._score_candidate(
                r, circularity=circ, radial=radial,
                edge_support=edge_sup * consistency,
                weights=(0.30, 0.0, 0.40, 0.30))
            if score <= 0:
                continue
            out.append(Detection(
                x=float(cx), y=float(cy), radius=float(r), score=score,
                method=self.name, contour=cnt,
                parts={"circularity": circ, "radial": radial,
                       "gradient_support": support, "sign_consistency": consistency,
                       "edge_support": edge_sup, "ellipse_ratio": ratio},
            ))
        # 给了 seed 就**只认**种子那一颗（空间门 + 半径带）。gate / 带都用种子
        # 半径标定，所以水果走近走远一点也不会被甩掉。
        #
        # ⚠️ 2026-11-07 Critic 抓出的缺陷（原版是 `if near: out = near`）：
        # 带内**没有**候选时，原版会保留 `out`（即门外的候选）并照常返回 ——
        # 于是同色圆干扰物会**默默占住 detections[0]**，恰好是下面注释声称
        # "不会发生"的事，而且 `last_error` 还是空。现在改成与
        # `color._rank` 同一语义：带内空 → 返回锚点（人给的位置），绝不回落门外候选。
        if self._seed is not None:
            sx, sy, sr = self._seed
            gate = max(SEED_GATE_MIN_PX, SEED_GATE_RATIO * sr)
            lo, hi = SEED_RADIUS_LO * sr, SEED_RADIUS_HI * sr
            near = [d for d in out
                    if math.hypot(d.x - sx, d.y - sy) <= gate
                    and lo <= d.radius <= hi]
            if near:
                out = near
            else:
                out = [self._anchor_detection(
                    note=(f"种子门内无候选（种子 ({sx:.0f},{sy:.0f}) r={sr:.1f}；"
                          f"门 {gate:.0f}px、半径带 {lo:.1f}~{hi:.1f}）："
                          f"画面里 {len(out)} 个近圆候选都在门外，按 seed 语义"
                          f"不返回它们，改用点选锚点"))]

        # 没 seed 且多个候选都过门：纯几何判据下它们不可区分，**如实全部输出**
        # 并写明判据，交给下游按半径带 / 时序连续性裁决，绝不假装只有一个答案。
        if len(out) > 1:
            for d in out:
                d.extra["geometry_ambiguous"] = True
                d.extra["geometry_n_candidates"] = len(out)
                d.extra["geometry_note"] = (
                    "同色近圆多目标：圆度/径向一致性/梯度符号一致性均合格，"
                    "纯几何不可区分；需下游按半径带或时序连续性裁决")

        if not out:
            # 0 检出必须说话（对齐 color.last_error 的口径；原版此处完全静默）
            self.last_error = (
                f"无近圆候选：Canny 边缘 {n_edge} px / 外轮廓 {n_cnt} 个；"
                f"门槛 圆度≥{MIN_CIRCULARITY}、径向≥{MIN_RADIAL}、"
                f"梯度符号≥{MIN_SIGN_CONSISTENCY}、边缘支持≥{MIN_EDGE_SUPPORT}、"
                f"椭圆度 {ELLIPSE_RATIO_LO}~{ELLIPSE_RATIO_HI}。"
                f"轮廓很多仍 0 检出 → 多半是表面纹理/斑点把外轮廓打断了"
                f"（Canny 闭合链断开），不是参数没调好")

        out.sort(key=lambda d: d.score, reverse=True)
        return out
