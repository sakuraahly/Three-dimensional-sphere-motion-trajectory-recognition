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
            #
            # ⚠️ 2026-10-18：这里原来写的是 "T[0] < 0"，符号同样写反了
            #（那时 `baseline_side()` 也是反的，两处错刚好互相掩盖）。
            # 现在按 `T = -C` 的本义写：**装反 ⟹ T[0] > 0**。
            lines.append("  [警告] T[0] > 0：标定时 --left/--right 可能接反了"
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
        """右相机在左相机的哪一侧。返回 `"right"` / `"left"` / `"unknown"`。

        ## ⚠️ 2026-10-18 修正：这里原来把符号写**反了**

        OpenCV 的约定是 `X_right = R · X_left + T` ——
        把 `X_left = 0`（左光心）代进去得到 `X_right = T`，
        所以 **`T` 是"左光心在右相机系里的位置"**，不是"右光心在左相机系里"。

        设右相机在左相机系里的位置是 `C`，则 `X_right = R·(X_left - C)`，
        即 `T = -R·C`；`R ≈ I` 时 **`T = -C`**。

        | 右相机物理位置 | `C` | 解出的 `T[0]` |
        | -------------- | --- | ------------- |
        | 在右边（正常） | `+B` | **`< 0`** |
        | 在左边（装反） | `-B` | **`> 0`** |

        **合成真值判决**（`outputs/_handoff_read/audit_T_sign.py`）：
        真值右相机放在 `+X`（物理右侧）、`B=65 mm`，`calibrate_stereo()` 解出
        `T = (-0.065, ~0, ~0)` —— 与上表一致。所以原实现
        （`T[0] > 0` 才叫 "right"）**把每一台正常装置都判成了"装反"**，
        于是 `describe()` 每次标定完都打一条假警报。

        假警报的代价不是"多一条日志"：它会**训练用户忽略这条警告** ——
        等真的装反时，人已经习惯性无视它了。
        所以这条必须按真值修，而不是"看起来能跑就不动"。

        装反时的后果（保留原说明，仍然成立）：
          - 三角测量**仍然正确**（`P1/P2` 本身自洽，三维点没错）；
          - 但任何"假设视差为正"的匹配逻辑会**一个点都配不上**（0%），
            报错只有"配不上"，完全指不到原因。
        """
        t = np.asarray(self.T, np.float64).ravel()
        if t.size < 1 or abs(float(t[0])) < 1e-9:
            return "unknown"
        # T = -C  ⟹  T[0] < 0 才是"右相机在右边"
        return "right" if float(t[0]) < 0 else "left"

    def disparity_sign(self) -> int:
        """校正后正常视差的符号（`x_left - x_right` 的符号）。

        | 装置 | `baseline_side()` | 视差符号 |
        | ---- | ----------------- | -------- |
        | 正常（右相机在右） | `"right"` | **`+1`** |
        | 装反（右相机在左） | `"left"`  | `-1` |

        为什么正常时是 `+1`：右相机在右边，同一个物点在**右图里更靠左**，
        所以 `x_left - x_right > 0`。这与合成真值一致
        （真值装置平均视差 `+125.37 px`，见 `audit_T_sign.py`）。

        > 2026-10-18 修正：原实现按**写反的** `baseline_side()` 再取反，
        > 于是"两次错抵消"——**输出数值恰好是对的**，但注释里的推导全错，
        > 而且任何人顺着注释去改就会真的改坏。现在把两处都摆正。

        这个符号不需要调用方记 —— `StereoReconstructor` 会自动从 rig 取。
        """
        side = self.baseline_side()
        if side == "right":
            return 1
        if side == "left":
            return -1
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
                # ---- 不确定度（2026-10-18 加，审查 A5）----
                #
                # 为什么存进 JSON：RMS 是"拟合得多好"，这两个数是"参数能信到什么
                # 程度"。合成真值里**同一份 RMS** 可以对应 4 倍不同的深度精度，
                # 所以只看 RMS 会误判。存下来，事后复盘才有依据。
                "grade": res.grade,
                "board_area": res.board_area,
                "fx_sigma_px": res.fx_sigma(),
                "depth_sigma_mm_at_0_6m": res.depth_sigma_mm(0.6),
            },
            meta=meta or {},
        )


def load_stereo_rig(path: str | Path) -> StereoRig:
    return StereoRig.load(path)


def save_stereo_rig(rig: StereoRig, path: str | Path):
    return rig.save(path)
