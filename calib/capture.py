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

from .board import BoardObservation, BoardSpec, detect_board, detect_board_in_roi, pose_distance


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
    # 每帧做判断用的"粗检"分辨率。全图精确检测要 300 ms，撑不住实时；
    # 粗检 640 宽只要约 12 ms，足够判断"值不值得收"。真正收帧时再在棋盘
    # 附近做全分辨率精确检测（裁剪不损失分辨率）。
    gate_work_width: int = 640


@dataclass
class CapturedPair:
    corners_left: np.ndarray
    corners_right: np.ndarray
    homography_left: np.ndarray
    homography_right: np.ndarray
    center_left: tuple[float, float]
    quality: dict = field(default_factory=dict)


def diagnose_frame(gray: np.ndarray, work_width: int = 640) -> str:
    """棋盘没检测到时，用**整幅画面**的统计量给一句人话会诊。

    为什么需要：`findChessboardCornersSB` 返回"没找到"是一个黑盒 ——
    画面蒙雾、过曝、太暗、跑焦、板错了(行列数/方格数不符)都会得到同一个结果。
    在标定窗口前反复调整却没有任何反馈，是最耗时间的一种卡住。
    这里只用几个极便宜的统计量（缩到 320 宽算），把最常见的原因指出来。

    返回空字符串表示"画面本身看不出毛病"（那多半是板不对或没在画面里）。
    """
    try:
        h, w = gray.shape[:2]
        if h < 8 or w < 8:
            return ""
        scale = 320.0 / float(w)
        small = cv2.resize(gray, (320, max(8, int(round(h * scale)))),
                           interpolation=cv2.INTER_AREA)
    except cv2.error:
        return ""

    mean = float(small.mean())
    white = float((small > 245).mean())      # 过曝比例
    black = float((small < 12).mean())       # 死黑比例
    sharp = float(cv2.Laplacian(small, cv2.CV_64F).var())
    # 实际用到的动态范围。**"蒙雾"的指纹在这里，不在整幅标准差上**：
    # 蒙雾/脏镜头/起雾会把黑压不到底、白提不到顶，直方图挤成很窄一条；
    # 而画面里有大块纯色（白墙、木桌）也会让标准差偏大，用标准差判会漏判。
    p5, p95 = (float(v) for v in np.percentile(small, (5.0, 95.0)))
    spread = p95 - p5

    if mean < 12:
        return "整幅几乎全黑 —— 镜头被挡住了，或者这条流根本没打开"
    if black > 0.6:
        return f"画面 {black * 100:.0f}% 是死黑 —— 镜头被挡住或没打开"
    if mean < 30:
        return f"画面过暗（均值 {mean:.0f}）—— 补光或调高曝光"
    if white > 0.45 or mean > 240:
        return (f"画面 {white * 100:.0f}% 过曝（均值 {mean:.0f}）——"
                f"多半是逆光/正对强光源，或自动曝光被拉满；"
                f"挪开强光或关掉自动曝光再试")
    if spread < 70.0 and mean > 110:
        return (f"画面动态范围被压到只剩 {spread:.0f} 级（均值 {mean:.0f}）——"
                f"像蒙了一层雾：先擦镜头、撕掉保护膜，检查有没有起雾/结露、"
                f"或有没有东西紧贴镜头")
    if sharp < 25.0:
        return (f"画面细节很少（清晰度 {sharp:.1f}）——"
                f"跑焦或严重运动模糊；对准远处有细节的地方看能不能合焦")
    if white > 0.12:
        return (f"画面有 {white * 100:.0f}% 过曝区 ——"
                f"强光直射会把棋盘对比吃掉，转个方向或降低曝光")
    return ""


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
        # 本帧的粗检结果，供界面直接画框（避免重复检测 —— 那是主要的卡顿来源）
        self.last_obs: tuple[BoardObservation | None, BoardObservation | None] = (None, None)
        self.last_accepted = False

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
        self.last_obs = (None, None)
        self.last_accepted = False

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

        # 每帧只做"粗检"：降分辨率 + 不开 EXHAUSTIVE，约 12 ms/路
        obs_l = detect_board(left_gray, spec, work_width=cfg.gate_work_width, exhaustive=False)
        obs_r = detect_board(right_gray, spec, work_width=cfg.gate_work_width, exhaustive=False)
        self.last_obs = (obs_l, obs_r)
        self.last_accepted = False

        # 1) 两路都要检测到
        #    这里**必须说清是哪一路**没找到：两路画面质量差很多时（比如一路蒙雾、
        #    一路正常），只说"两路未同时检测到"会让人以为是标定板的问题，
        #    在窗口前反复调整却查不到真正的原因。顺手给一句画面会诊。
        if not obs_l.found or not obs_r.found:
            if not obs_l.found and not obs_r.found:
                why = "两路都未检测到棋盘"
            elif not obs_l.found:
                why = "左路未检测到棋盘（右路正常）"
            else:
                why = "右路未检测到棋盘（左路正常）"
            notes = []
            for name, obs, gray in (("左", obs_l, left_gray), ("右", obs_r, right_gray)):
                if not obs.found:
                    note = diagnose_frame(gray, cfg.gate_work_width)
                    if note:
                        notes.append(f"{name}：{note}")
            if notes:
                why += "｜" + "；".join(notes)
            return self._reject(why)
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

        # 7) 全部通过 -> 在棋盘附近做全分辨率精确检测（只在这一帧做，约 40 ms/路）
        acc_l = detect_board_in_roi(left_gray, spec, obs_l.corners)
        acc_r = detect_board_in_roi(right_gray, spec, obs_r.corners)
        if acc_l is None or acc_r is None:
            return self._reject("精确复核失败（板太靠边或被遮挡）")
        # 用精确角点复核关键指标，避免粗检的近似把不合格的帧放进来
        for name, acc in (("左", acc_l), ("右", acc_r)):
            if acc.sharpness < cfg.min_sharpness:
                return self._reject(f"{name}图模糊(精确复核)")
            if acc.margin_px < cfg.min_margin_px:
                return self._reject(f"{name}图棋盘贴边(精确复核)")
            if acc.area_ratio < cfg.min_area_ratio:
                return self._reject(f"{name}图棋盘太小(精确复核)")

        self._region_count[(gx, gy)] = self._region_count.get((gx, gy), 0) + 1
        self.pairs.append(
            CapturedPair(
                corners_left=acc_l.corners.copy(),
                corners_right=acc_r.corners.copy(),
                homography_left=acc_l.homography.copy(),
                homography_right=acc_r.homography.copy(),
                center_left=(cx, cy),
                quality={
                    "sharpness_l": acc_l.sharpness, "sharpness_r": acc_r.sharpness,
                    "area_l": acc_l.area_ratio, "area_r": acc_r.area_ratio,
                },
            )
        )
        self.last_obs = (acc_l, acc_r)
        self.last_accepted = True
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

    def side_lines(self) -> list[str]:
        """**逐路**状态：哪一路找到板、清晰度多少。

        为什么单独要这一行：两路画质差很多时（一路蒙雾一路正常），
        只有合并后的"未同时检测到"完全看不出是哪一路的问题，
        人会一直在窗口前挪板子。把这行打在画面上，一眼就知道该修哪路。
        """
        out = []
        for name, obs in (("左", self.last_obs[0]), ("右", self.last_obs[1])):
            if obs is None:
                out.append(f"{name}: -")
                continue
            if not obs.found:
                out.append(f"{name}: 未找到板")
                continue
            out.append(f"{name}: 清晰度{obs.sharpness:5.0f} 面积{obs.area_ratio * 100:4.1f}%"
                       f" 曝光{obs.brightness:3.0f}")
        return out
