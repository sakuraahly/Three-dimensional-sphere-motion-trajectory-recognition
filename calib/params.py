"""标定结果的保存 / 加载 / 使用。

保存成两个同名文件：
  - xxx.npz   所有矩阵（给程序读）
  - xxx.json  人读的质量报告与元信息（棋盘规格、日期、相机、误差）

这样"这次标定到底靠不靠谱"随时可以翻出来看，而不是只留一堆看不懂的矩阵。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .stereo_calib import StereoCalibrationResult

_MAT_KEYS = ("K1", "D1", "K2", "D2", "R", "T", "E", "F", "R1", "R2", "P1", "P2", "Q")


@dataclass
class StereoRig:
    """一个标定好的双目装置。

    坐标系约定：三维点是在**校正后的左相机坐标系**下，
    原点在左相机光心，Z 轴向前，单位与标定板方格尺寸一致（本项目用米）。
    """

    K1: np.ndarray
    D1: np.ndarray
    K2: np.ndarray
    D2: np.ndarray
    R: np.ndarray
    T: np.ndarray
    image_size: tuple[int, int]
    E: np.ndarray | None = None
    F: np.ndarray | None = None
    R1: np.ndarray | None = None
    R2: np.ndarray | None = None
    P1: np.ndarray | None = None
    P2: np.ndarray | None = None
    Q: np.ndarray | None = None
    quality: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    @property
    def baseline_mm(self) -> float:
        return float(np.linalg.norm(self.T)) * 1000.0

    def describe(self) -> str:
        q = self.quality
        lines = [
            f"双目装置  基线 {self.baseline_mm:.2f} mm   图像 {self.image_size[0]}x{self.image_size[1]}",
            f"  重投影 RMS {q.get('rms', float('nan')):.3f} px   "
            f"极线误差 {q.get('epipolar_rms', float('nan')):.3f} px",
        ]
        if self.meta.get("board"):
            lines.append(f"  标定板 {self.meta['board']}")
        if self.meta.get("created"):
            lines.append(f"  标定时间 {self.meta['created']}")
        if self.baseline_side() == "left":
            # 装反了指标一样好看，但视差整体反号；不提醒的话问题会在
            # "配对率 0%" 那里才暴露，而且报错指不到原因。
            lines.append("  [警告] T[0] < 0：标定时 --left/--right 可能接反了"
                         "（右相机其实在左边）。三角测量仍然正确，"
                         "但请核对 L/R 标签，见 apps/verify_stereo.py。")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def rectify_maps(self, side: str = "both"):
        """生成去畸变+校正的映射表（每帧调用 remap 即可）。

        注意：K/D 不随使用场景变化，但校正映射依赖图像尺寸，
        所以换分辨率要么重新生成映射，要么重新标定。
        """
        def _maps(K, D, R, P):
            return cv2.initUndistortRectifyMap(K, D, R, P, self.image_size, cv2.CV_32FC1)

        if side == "left":
            return _maps(self.K1, self.D1, self.R1, self.P1)
        if side == "right":
            return _maps(self.K2, self.D2, self.R2, self.P2)
        return (_maps(self.K1, self.D1, self.R1, self.P1),
                _maps(self.K2, self.D2, self.R2, self.P2))

    def rectify_image(self, img: np.ndarray, side: str) -> np.ndarray:
        mx, my = self.rectify_maps(side)
        return cv2.remap(img, mx, my, cv2.INTER_LINEAR)

    # ------------------------------------------------------------------
    def rectify_points(self, pts_left: np.ndarray, pts_right: np.ndarray):
        """把原始像素坐标校正到共面极线坐标系（三角测量前必做）。"""
        p1 = cv2.undistortPoints(
            np.asarray(pts_left, np.float64).reshape(-1, 1, 2), self.K1, self.D1,
            R=self.R1, P=self.P1,
        ).reshape(-1, 2)
        p2 = cv2.undistortPoints(
            np.asarray(pts_right, np.float64).reshape(-1, 1, 2), self.K2, self.D2,
            R=self.R2, P=self.P2,
        ).reshape(-1, 2)
        return p1, p2

    def triangulate(self, pts_left: np.ndarray, pts_right: np.ndarray) -> np.ndarray:
        """由左右像素坐标三角测量出三维点（米），返回 (N,3)。"""
        a = np.asarray(pts_left, np.float64).reshape(-1, 2)
        b = np.asarray(pts_right, np.float64).reshape(-1, 2)
        if len(a) != len(b):
            raise ValueError("左右点数不一致")
        p1, p2 = self.rectify_points(a, b)
        pts4d = cv2.triangulatePoints(self.P1, self.P2, p1.T, p2.T)
        w = np.where(np.abs(pts4d[3]) < 1e-12, 1e-12, pts4d[3])
        return (pts4d[:3] / w).T

    def epipolar_error(self, pts_left: np.ndarray, pts_right: np.ndarray) -> float:
        """校正后同名点的 y 之差（RMS，像素）。这是检验装置有没有被碰动的直接指标。"""
        p1, p2 = self.rectify_points(pts_left, pts_right)
        n = min(len(p1), len(p2))
        if n == 0:
            return float("nan")
        dy = p1[:n, 1] - p2[:n, 1]
        return float(np.sqrt(np.mean(dy * dy)))

    # ------------------------------------------------------------------
    def baseline_side(self) -> str:
        """判断右相机到底在左相机的哪一侧 —— 决定校正后视差的正负号。

        为什么必须能查这个：本项目的 `T` 是"右光心在左相机系里的位置"
        （`X_right = R @ X_left + T`），所以 `T[0] > 0` 表示右相机确实在右边。

        如果左右装反了（同型号相机重新插拔后索引互换，而画面完全正常、
        标定指标也好看），`T[0]` 变号，**校正后的视差整体反号**。这时：
          - 三角测量**仍然正确**（P1/P2 本身自洽，三维点没错）；
          - 但任何"假设视差为正"的匹配逻辑会**一个点都配不上**（0%），
            报错只有"配不上"，完全指不到原因。
        所以给出这个能一眼看出来的判定，供加载标定后自检。
        """
        t = np.asarray(self.T, np.float64).ravel()
        if t.size < 1 or abs(float(t[0])) < 1e-9:
            return "unknown"
        return "right" if float(t[0]) > 0 else "left"

    def disparity_sign(self) -> int:
        """校正后正常视差的符号：右相机在右边 -> -1，装反了 -> +1。

        **注意是反的**，别凭直觉填：`cv2.stereoRectify` 的 `P2` 第 4 列
        实测等于 `-f * T[0]`（数值验证见 tests/selftest_stereo_branch.py
        的 test_baseline_sign）。于是 T[0] > 0（右相机在右）时，
        同一个物点在右图里落在**更右边**，视差 x1-x2 为负。

        这个符号不需要调用方记 —— `StereoReconstructor` 会自动从 rig 取。
        """
        side = self.baseline_side()
        if side == "right":
            return -1
        if side == "left":
            return 1
        return 0

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> tuple[Path, Path]:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        arrays = {k: getattr(self, k) for k in _MAT_KEYS if getattr(self, k, None) is not None}
        arrays["image_size"] = np.asarray(self.image_size, dtype=np.int32)
        np.savez(p.with_suffix(".npz"), **arrays)

        meta = dict(self.meta)
        meta.setdefault("created", time.strftime("%Y-%m-%d %H:%M:%S"))
        meta["baseline_mm"] = self.baseline_mm
        meta["image_size"] = list(self.image_size)
        meta["quality"] = self.quality
        p.with_suffix(".json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return p.with_suffix(".npz"), p.with_suffix(".json")

    @classmethod
    def load(cls, path: str | Path) -> "StereoRig":
        p = Path(path)
        npz_path = p.with_suffix(".npz")
        if not npz_path.exists():
            raise FileNotFoundError(f"找不到标定文件：{npz_path}")
        data = np.load(npz_path)
        meta: dict = {}
        js = p.with_suffix(".json")
        if js.exists():
            try:
                meta = json.loads(js.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = {}
        quality = meta.pop("quality", {})
        size = tuple(int(v) for v in data["image_size"]) if "image_size" in data else (0, 0)
        return cls(
            **{k: data[k] for k in _MAT_KEYS if k in data},
            image_size=size, quality=quality, meta=meta,
        )

    @classmethod
    def from_result(cls, res: StereoCalibrationResult, meta: dict | None = None) -> "StereoRig":
        if not res.ok:
            raise ValueError(f"标定结果不可用，拒绝生成装置参数：{res.verdict}")
        return cls(
            K1=res.K1, D1=res.D1, K2=res.K2, D2=res.D2, R=res.R, T=res.T,
            E=res.E, F=res.F, R1=res.R1, R2=res.R2, P1=res.P1, P2=res.P2, Q=res.Q,
            image_size=res.image_size,
            quality={
                "rms": res.rms, "err_left": res.err_left, "err_right": res.err_right,
                "epipolar_rms": res.epipolar_rms, "epipolar_max": res.epipolar_max,
                "n_pairs_used": res.n_pairs_used, "n_pairs_in": res.n_pairs_in,
                "verdict": res.verdict,
            },
            meta=meta or {},
        )


def load_stereo_rig(path: str | Path) -> StereoRig:
    return StereoRig.load(path)


def save_stereo_rig(rig: StereoRig, path: str | Path):
    return rig.save(path)
