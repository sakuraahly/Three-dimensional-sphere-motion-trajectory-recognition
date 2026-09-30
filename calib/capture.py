"""自动挑帧：决定"这一帧该不该收"。

标定质量的瓶颈往往不是算法，而是**收进来的数据**：
模糊的、贴边的、全挤在同一个角度的图，会把标定结果拖坏而且看不出来。
这里把"好数据"的标准写成可执行的判据，让程序自己挑。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .board import BoardObservation, BoardSpec, detect_board, pose_distance


@dataclass
class CaptureConfig:
    """自动挑帧的门槛。默认值偏严，宁可少收几帧也要收干净的。"""

    target_pairs: int = 25          # 收够多少对就停
    min_pairs: int = 10             # 少于这么多对不做标定
    min_sharpness: float = 55.0     # 棋盘区域 Laplacian 方差下限（防运动模糊/失焦）
    min_area_ratio: float = 0.012   # 棋盘至少占画面面积的比例（别太远）
    max_area_ratio: float = 0.90    # 也别贴到脸上
    min_margin_px: float = 10.0     # 角点离画面边界至少多少像素（边缘畸变大）
    min_brightness: float = 35.0    # 防过暗
    max_brightness: float = 230.0   # 防过曝
    # 是否要求"板在两路里都基本静止"。
    #   True（默认）：实时采集、以及"手持板走一段停一下"的视频 —— 这个门限会自动
    #                 挑出人停稳的那几帧，同时**顺带解决了左右不同步的问题**
    #                 （静态目标在任意时刻看起来都一样）。
    #   False：独立静帧（图片文件夹、每帧都是不同位姿）时用，此时帧间本来就会大变。
    require_stationary: bool = True
    stationary_max_px: float = 3.0  # 连续两帧位姿变化上限（板要基本静止）
    diversity_min: float = 0.055    # 与已收位姿的最小差异（占图像对角线比例）
    region_grid: int = 3            # 画面分几宫格
    max_per_region: int = 5         # 每宫格最多收几对，强制覆盖全画面


@dataclass
class CapturedPair:
    corners_left: np.ndarray
    corners_right: np.ndarray
    homography_left: np.ndarray
    homography_right: np.ndarray
    center_left: tuple[float, float]
    quality: dict = field(default_factory=dict)


class AutoPairCollector:
    """自动收集标定图像对。

    最重要的一条判据是 **stationary（板在两路里都基本静止）**：
    它保证两台相机拍到的是同一个位姿。有了它，
    两路视频即使没有做时间同步，标定也不受影响 ——
    因为静态目标在任意时刻看起来都一样。
    """

    def __init__(self, spec: BoardSpec, cfg: CaptureConfig | None = None) -> None:
        self.spec = spec
        self.cfg = cfg or CaptureConfig()
        self.pairs: list[CapturedPair] = []
        self._prev_h: dict[str, np.ndarray | None] = {"left": None, "right": None}
        self._region_count: dict[tuple[int, int], int] = {}
        self.rejected: dict[str, int] = {}
        self.last_reason = ""
        self.image_size: tuple[int, int] | None = None  # (w, h)

    # ------------------------------------------------------------------
    @property
    def count(self) -> int:
        return len(self.pairs)

    @property
    def full(self) -> bool:
        return self.count >= self.cfg.target_pairs

    @property
    def enough(self) -> bool:
        return self.count >= self.cfg.min_pairs

    def reset(self) -> None:
        self.pairs.clear()
        self._prev_h = {"left": None, "right": None}
        self._region_count.clear()
        self.rejected.clear()

    def _reject(self, reason: str) -> tuple[bool, str]:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1
        self.last_reason = reason
        return False, reason

    # ------------------------------------------------------------------
    def offer(
        self,
        left_gray: np.ndarray,
        right_gray: np.ndarray,
        spec: BoardSpec | None = None,
    ) -> tuple[bool, str]:
        """尝试收一帧。返回 (是否收下, 原因)。"""
        cfg = self.cfg
        spec = spec or self.spec
        h, w = left_gray.shape[:2]
        self.image_size = (w, h)
        diag = math.hypot(w, h)

        obs_l = detect_board(left_gray, spec)
        obs_r = detect_board(right_gray, spec)

        # 1) 两路都要检测到
        if not obs_l.found or not obs_r.found:
            return self._reject("两路未同时检测到棋盘")
        assert obs_l.corners is not None and obs_r.corners is not None

        # 2) 清晰度
        for name, obs in (("左", obs_l), ("右", obs_r)):
            if obs.sharpness < cfg.min_sharpness:
                return self._reject(f"{name}图模糊(清晰度 {obs.sharpness:.0f})")

        # 3) 大小与位置
        for name, obs in (("左", obs_l), ("右", obs_r)):
            if obs.area_ratio < cfg.min_area_ratio:
                return self._reject(f"{name}图棋盘太小({obs.area_ratio * 100:.1f}%)")
            if obs.area_ratio > cfg.max_area_ratio:
                return self._reject(f"{name}图棋盘太近")
            if obs.margin_px < cfg.min_margin_px:
                return self._reject(f"{name}图棋盘贴边({obs.margin_px:.0f}px)")
            if not (cfg.min_brightness <= obs.brightness <= cfg.max_brightness):
                return self._reject(f"{name}图曝光不合适({obs.brightness:.0f})")

        # 4) 静止：两路都必须和上一帧几乎一样
        #    这条同时解决了两路不同步的问题（静态目标在任意时刻都一样）
        if cfg.require_stationary:
            for side, obs in (("left", obs_l), ("right", obs_r)):
                prev = self._prev_h[side]
                if prev is not None:
                    d = pose_distance(prev, obs.homography, diag)
                    if d * diag > cfg.stationary_max_px:
                        self._prev_h["left"], self._prev_h["right"] = obs_l.homography, obs_r.homography
                        return self._reject("棋盘还在移动，请停稳再收")
        self._prev_h["left"], self._prev_h["right"] = obs_l.homography, obs_r.homography

        # 5) 多样性：别把 25 张全收成同一个角度
        for p in self.pairs:
            d_l = pose_distance(p.homography_left, obs_l.homography, diag)
            d_r = pose_distance(p.homography_right, obs_r.homography, diag)
            if min(d_l, d_r) < cfg.diversity_min:
                return self._reject("与已收位姿太接近")

        # 6) 每个宫格限量，强制覆盖整个画面
        cx = float(obs_l.corners.reshape(-1, 2)[:, 0].mean())
        cy = float(obs_l.corners.reshape(-1, 2)[:, 1].mean())
        gx = min(cfg.region_grid - 1, int(cx / w * cfg.region_grid))
        gy = min(cfg.region_grid - 1, int(cy / h * cfg.region_grid))
        if self._region_count.get((gx, gy), 0) >= cfg.max_per_region:
            return self._reject("该区域已收够，请换个位置")

        self._region_count[(gx, gy)] = self._region_count.get((gx, gy), 0) + 1
        self.pairs.append(
            CapturedPair(
                corners_left=obs_l.corners.copy(),
                corners_right=obs_r.corners.copy(),
                homography_left=obs_l.homography.copy(),
                homography_right=obs_r.homography.copy(),
                center_left=(cx, cy),
                quality={
                    "sharpness_l": obs_l.sharpness, "sharpness_r": obs_r.sharpness,
                    "area_l": obs_l.area_ratio, "area_r": obs_r.area_ratio,
                },
            )
        )
        self.last_reason = f"已收 {self.count}/{cfg.target_pairs}"
        return True, self.last_reason

    # ------------------------------------------------------------------
    def coverage_image(self, size: tuple[int, int] | None = None, scale: float = 0.5) -> np.ndarray:
        """画一张覆盖率示意图：已收位姿的棋盘四边形 + 宫格计数。"""
        if self.image_size is None:
            w, h = 640, 480
        else:
            w, h = self.image_size
        if size is not None:
            w, h = size
        canvas = np.zeros((int(h * scale), int(w * scale), 3), dtype=np.uint8)
        canvas[:] = (28, 28, 32)
        g = self.cfg.region_grid
        for i in range(1, g):
            cv2.line(canvas, (int(w * scale * i / g), 0), (int(w * scale * i / g), canvas.shape[0]), (60, 60, 66), 1)
            cv2.line(canvas, (0, int(h * scale * i / g)), (canvas.shape[1], int(h * scale * i / g)), (60, 60, 66), 1)
        for idx, p in enumerate(self.pairs):
            quad = cv2.perspectiveTransform(
                np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32).reshape(-1, 1, 2),
                p.homography_left,
            ).reshape(-1, 2) * scale
            cv2.polylines(canvas, [quad.astype(np.int32)], True, (0, 200, 90), 1, cv2.LINE_AA)
            c = p.center_left[0] * scale, p.center_left[1] * scale
            cv2.circle(canvas, (int(c[0]), int(c[1])), 2, (0, 220, 255), -1)
        done = len(self.pairs)
        total = self.cfg.target_pairs
        cv2.putText(canvas, f"{done}/{total}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return canvas

    def status_lines(self) -> list[str]:
        """给实时窗口用的状态文本。"""
        lines = [f"collected {self.count}/{self.cfg.target_pairs}"]
        if self.rejected:
            top = sorted(self.rejected.items(), key=lambda kv: -kv[1])[:3]
            lines.append("last: " + self.last_reason)
            for r, n in top:
                lines.append(f"  x{n}  {r}")
        else:
            lines.append("last: " + self.last_reason)
        return lines
