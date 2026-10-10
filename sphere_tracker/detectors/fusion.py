"""多线索投票融合检测器。"""

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
      1. 依次跑各个子检测器，各取最好的一个候选；
      2. 任意两个候选中心距离 < max(agree_px, agree_ratio * 半径)，算"同意"；
      3. 得票最高的候选胜出，每多一票加 0.12 分；
      4. 只有一票、且总分低于 `DetectorConfig.min_score` 时判为不可信（返回空）。
    这样既能用颜色拿精度、用霍夫拿鲁棒性，又能靠"多方一致"压掉误检。
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

    def reset(self) -> None:
        for det in self.members.values():
            det.reset()

    def detect(self, frame: np.ndarray, ctx=None) -> list[Detection]:
        cands: list[tuple[str, Detection]] = []
        self.last_parts = {}
        for name, det in self.members.items():
            try:
                found = det.detect(frame, ctx)
            except cv2.error:
                found = []
            best = found[0] if found else None
            self.last_parts[name] = best
            if best is not None:
                cands.append((name, best))
        if not cands:
            return []

        # 计算两两一致 -> 投票
        votes: dict[str, int] = {name: 0 for name, _ in cands}
        for i in range(len(cands)):
            for j in range(i + 1, len(cands)):
                ni, di = cands[i]
                nj, dj = cands[j]
                tol = max(self.cfg.agree_px, self.cfg.agree_ratio * max(di.radius, dj.radius))
                if math.hypot(di.x - dj.x, di.y - dj.y) <= tol:
                    votes[ni] += 1
                    votes[nj] += 1

        scored: list[tuple[float, str, Detection]] = []
        for name, det in cands:
            total = min(1.0, det.score + 0.12 * votes[name])
            scored.append((total, name, det))
        scored.sort(key=lambda item: item[0], reverse=True)

        total, name, best = scored[0]
        # 阈值读 DetectorConfig.min_score，不再硬编码 0.55
        # （此前硬编码导致改配置不生效：改 min_score 没有任何效果）
        if votes[name] == 0 and total < self.cfg.min_score:
            return []  # 孤证且置信度不高 -> 宁可不报
        best = Detection(
            x=best.x, y=best.y, radius=best.radius, score=total,
            method=f"{self.name}:{name}", contour=best.contour, mask=best.mask,
            parts={**best.parts, "votes": float(votes[name])},
        )
        return [best]
