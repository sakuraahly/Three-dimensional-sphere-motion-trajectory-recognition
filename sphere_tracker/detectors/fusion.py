"""多线索投票融合检测器。

## 与旧版的三处差别（2026-11-06）

1. **每个成员交 top-K 候选，而不是只交最好的一个。**
   旧版 `best = found[0]`，于是"画面里两个同色圆"时 color 只能交一个、
   hough 也只能交一个 —— 投票投的是"各成员碰巧都排第一的那一个"，而不是
   "多个成员都看见的那一个"（`纯色球检测方案.md` §2 记着这个失效）。
2. **先聚类再投票。** 候选按中心距离聚成簇（同一簇 = 同一个目标），每簇的
   票数 = 有多少个**不同成员**支持它。这样"color 和 hough 都看见真球、
   只有 color 看见干扰物"会得到 2:1，而不是旧版的 1:1 由分数决胜负。
3. **同分时看时间连续性。** 两簇票数、分数都接近时，取离"上一帧位置 +
   速度"更近的那一簇 —— 同色干扰物通常不会沿着球的轨迹走。

## 什么时候仍然会输

两个同色、同大小、都在动、且三个成员都只看见其中一个的物体，本方法无法
区分（没有任何信息可分）。真正的解法是采集时移走干扰物，或给 color 一个
"跟住这一颗"的种子（`seed()` 会透传给支持的成员）。
"""

from __future__ import annotations

import cv2
import math
import numpy as np

from ..config import DetectorConfig
from .base import BaseDetector
from .factory import build_single_detector
from ..types import Detection


class FusionDetector(BaseDetector):
    """多后端投票融合。

    规则：
      1. 依次跑各个子检测器，各取 **top-K** 候选（`cfg.fusion_topk`）；
      2. 候选按中心距离聚类：距离 < max(agree_px, agree_ratio×半径) 算同一目标；
      3. 每簇得分 = 簇内最高分 + 0.12×(支持它的**不同成员数** − 1)；
      4. 只有**一个成员**支持、且得分低于 `cfg.min_score` 的簇判为不可信；
      5. 票数与分数都接近时，取离上一帧预测位置更近的簇。
    """

    name = "fusion"

    def __init__(self, cfg: DetectorConfig) -> None:
        super().__init__(cfg)
        self.members: dict[str, BaseDetector] = {}
        for name in cfg.fusion_members:
            det = build_single_detector(name, cfg)
            if det is not None:
                self.members[name] = det
        self.last_parts: dict[str, Detection | None] = {}
        self.last_error = ""
        self._prev_center: tuple[float, float] | None = None

    def reset(self) -> None:
        for det in self.members.values():
            det.reset()
        self._prev_center = None

    def seed(self, x: float, y: float, radius: float, frame_index: int = 0) -> None:
        """把种子透传给**支持种子的成员**（当前是 color / specular）。

        不支持种子的成员不报错、也不假装收到了 —— 它们继续在整幅画面里找
        自己那一票，这正是融合想要的"独立第二意见"。
        """
        for det in self.members.values():
            fn = getattr(det, "seed", None)
            if callable(fn):
                fn(x, y, radius, frame_index=frame_index)

    # ------------------------------------------------------------------
    @staticmethod
    def _tol(a: Detection, b: Detection, agree_px: float, agree_ratio: float) -> float:
        return max(agree_px, agree_ratio * max(a.radius, b.radius))

    def detect(self, frame: np.ndarray, ctx=None) -> list[Detection]:
        cands: list[tuple[str, Detection]] = []
        self.last_parts = {}
        for name, det in self.members.items():
            try:
                found = det.detect(frame, ctx)
            except cv2.error:
                found = []
            self.last_parts[name] = found[0] if found else None
            for item in found[: max(1, int(self.cfg.fusion_topk))]:
                cands.append((name, item))
        if not cands:
            notes = [f"{n}: {getattr(d, 'last_error', '')}".strip(": ")
                     for n, d in self.members.items()
                     if getattr(d, "last_error", "")]
            self.last_error = ("融合成员都没有候选"
                               + ("（" + "；".join(notes) + "）" if notes else ""))
            return []

        # ---- 聚类：高分候选先当簇心，避免低分噪声先占位 ----
        ordered = sorted(cands, key=lambda p: p[1].score, reverse=True)
        clusters: list[list[tuple[str, Detection]]] = []
        for name, det in ordered:
            placed = False
            for cl in clusters:
                ref = cl[0][1]
                if math.hypot(det.x - ref.x, det.y - ref.y) <= self._tol(
                        det, ref, self.cfg.agree_px, self.cfg.agree_ratio):
                    cl.append((name, det))
                    placed = True
                    break
            if not placed:
                clusters.append([(name, det)])

        def summarise(cl: list[tuple[str, Detection]]):
            members = sorted({n for n, _ in cl})
            # 代表候选按**成员顺序**取（`fusion_members` 的先后是精度排序：
            # color 直接给轮廓 + 亚像素，hough 只有累加器圆心）。
            # 不能按 score 取：hough 在合成圆上能报 0.99，而它的圆心比 color
            # 差 2~5 px（2026-11-06 实测：按 score 取会让融合比单跑 color 更差）。
            order = {name: i for i, name in enumerate(self.members)}
            top = min(cl, key=lambda p: (order.get(p[0], 99), -p[1].score))
            total = min(1.0, float(top[1].score) + 0.12 * (len(members) - 1))
            return total, len(members), top

        scored = [(summarise(cl), cl) for cl in clusters]
        # 先按 (得分, 成员数) 排；两者的差值都很小时交给时间连续性
        scored.sort(key=lambda item: (item[0][0], item[0][1]), reverse=True)

        (best_total, best_members, (best_name, best)), best_cl = scored[0]
        if len(scored) > 1 and self._prev_center is not None:
            second_total, second_members, _ = scored[1][0]
            if (abs(best_total - second_total) <= 0.03
                    and best_members == second_members):
                px, py = self._prev_center
                def dist(item):
                    return math.hypot(item[0][2][1].x - px, item[0][2][1].y - py)
                (best_total, best_members, (best_name, best)), best_cl = min(
                    scored[:3], key=dist)

        if best_members == 1 and best_total < self.cfg.min_score:
            self.last_error = (f"融合只有一路（{best_name}）支持、得分 {best_total:.2f} "
                               f"< {self.cfg.min_score:.2f}，判为不可信")
            return []

        # 代表候选是锚点时，「为什么退回锚点」这句话属于**这一次输出**，
        # 不能被这里清掉：`pipeline` 靠 `last_error` 决定要不要把「这一帧
        # 返回的其实是点选位置」说出口。2026-11-07 之前这里无条件清空，
        # 于是成员（color / generic / specular）写的原因全白写，fusion
        # 这一路又变回静默。
        if (best.extra or {}).get("seed_anchor"):
            self.last_error = str(
                getattr(self.members.get(best_name), "last_error", "") or "")
        else:
            self.last_error = ""
        self._prev_center = (best.x, best.y)
        out = Detection(
            x=best.x, y=best.y, radius=best.radius, score=best_total,
            method=f"{self.name}:{best_name}", contour=best.contour, mask=best.mask,
            parts={**best.parts, "votes": float(best_members - 1)},
            extra={**best.extra,
                   "fusion_members": sorted({n for n, _ in best_cl})},
        )
        return [out]

    def debug_snapshot(self) -> dict:
        return {"error": self.last_error,
                "members": {n: (None if d is None else d.method)
                            for n, d in self.last_parts.items()}}
