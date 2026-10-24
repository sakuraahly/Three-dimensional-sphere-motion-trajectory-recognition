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
from sphere_tracker.geometry import (
    epipolar_line,
    fundamental_matrix_from_pose,
    point_line_distance,
)

_MAT_KEYS = ("K1", "D1", "K2", "D2", "R", "T", "E", "F", "R1", "R2", "P1", "P2", "Q")

#: 表示"**没有做过极线校正**"的 rectification 取值。两种都要认：
#: ``none_raw`` 是现在的 chain 输出，``not_applicable`` 是更早的 chain 落盘值
#: （那时还没定这个字符串）。只认一种的话，旧标定文件就会绕过护栏、在第 20 帧
#: 才以"0 配对"的形式炸掉——兼容意图落空，报错也更难读。
UNCORRECTED_RECTIFICATIONS: tuple[str, ...] = ("none_raw", "not_applicable")


def is_uncorrected(rectification: object) -> bool:
    """``rectification`` 是否表示"未做极线校正"（见 :data:`UNCORRECTED_RECTIFICATIONS`）。"""
    return str(rectification or "").strip().lower() in UNCORRECTED_RECTIFICATIONS


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

    @property
    def stereo_mode(self) -> str:
        """旧标定没有该字段 -> parallel。"""
        value = str(self.meta.get("stereo_mode") or "parallel").strip().lower()
        return "angled" if value in ("angled", "auto", "90", "90deg",
                                     "orthogonal", "angle") else "parallel"

    @property
    def rectification(self) -> str:
        """这套参数是怎么来的：``"stereo_rectify"`` / ``"none_raw"``。

        - ``stereo_rectify``：`R1/R2/P1/P2` 由 `cv2.stereoRectify` 产出，
          极线被摆成水平，（平行装置的）y 差才有几何意义；
        - ``none_raw``：chain（90°）产物 —— **未做极线校正**，
          `R1=R2=I`、`P1=K1·[I|0]`、`P2=K2·[R|T]`、`Q=None`。
          三角测量可用（坐标就在原始 left_camera 系），
          但"校正后 y 差"没有意义。

        旧标定文件没有这个字段：`parallel` 装置按 ``"stereo_rectify"`` 处理；
        chain 在 2026-10-24 之前的落盘值是 ``"not_applicable"``（当时 P1/P2
        还是 None），语义上同样是"未校正"，所以一并按未校正对待。
        """
        for src in (self.meta, self.quality):
            value = src.get("rectification") if isinstance(src, dict) else None
            if value:
                return str(value).strip().lower()
        return "stereo_rectify"

    @property
    def angle_deg(self) -> float:
        """两光轴夹角（度），从 R 计算；不修改/伪造 90.0。"""
        try:
            z2 = self.R.T @ np.array([0.0, 0.0, 1.0])
            n = float(np.linalg.norm(z2))
            if n <= 0:
                return 0.0
            c = float(np.clip(z2[2] / n, -1.0, 1.0))
            return float(np.degrees(np.arccos(c)))
        except Exception:                                   # noqa: BLE001
            return 0.0

    def describe(self) -> str:
        q = self.quality
        lines = [
            f"双目装置  模式 {self.stereo_mode}  基线 {self.baseline_mm:.2f} mm   "
            f"光轴夹角 {self.angle_deg:.1f}°   图像 {self.image_size[0]}x{self.image_size[1]}",
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
            R, P = self._undistort_target(R, P, K)
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
    def _undistort_target(self, R, P, K):
        """去畸变后的目标 (R, newCameraMatrix)。

        ## 为什么需要这一步（2026-10-24，none_raw）

        `undistortPoints(..., R=, P=)` / `initUndistortRectifyMap` 里，`P` 的
        **3x3 块必须是"目标相机的内参"**，旋转只从 `R` 进来。`stereoRectify`
        正是这么产出的（`P` 的 3x3 = 新的 K_rect）。

        而 `none_raw` 的 `P2 = K2·[R|T]` 的 3x3 块是 **K2·R（真实外参）**，
        不是新内参：把它当 newCameraMatrix 用，右图会被**多转一次 R**。
        实测（合成 90° 装置）：右路像素 828 px 被映射到 -9155 px，三角化直接爆掉。
        所以 `none_raw` 下必须让目标回到**各自的原始像素系**（R=I、P=K），
        R/T 只由 `P1/P2` 在三角化那一步承担 —— 这与
        `triangulatePoints(P1, P2, x1_raw, x2_raw)` 的约定完全一致。
        """
        if is_uncorrected(self.rectification):
            return np.eye(3), K
        return R, P

    def _side_target(self, side: str):
        """该路的 (K, D, R, P) —— **原始**参数，尚未按 `rectification` 分流。"""
        if side == "left":
            return self.K1, self.D1, self.R1, self.P1
        if side == "right":
            return self.K2, self.D2, self.R2, self.P2
        raise ValueError(f"未知的 side：{side!r}（只支持 'left' / 'right'）")

    def undistort_points(self, side: str, pts: np.ndarray) -> np.ndarray:
        """把某一路的原始像素去畸变，并映射到 `rectification` 对应的目标像素系。

        **"原始像素 → 三角化用的像素"只有这一个入口**：按 `rectification` 的
        分流只发生在 :meth:`_undistort_target` 里。调用方（如
        `stereo/triangulate.py`）**不要**自己再判一次模式 ——
        两处各判一次，早晚会有一处漏改，而症状是"残差偏大"这种不指向根因的错。

        - ``stereo_rectify``：`R=R*, P=P*`（极线被摆平；`P` 的 3x3 是新内参）；
        - ``none_raw``：`R=I, P=K`（只去畸变，留在各自原始像素系；
          `R/T` 由 `P1/P2` 在三角化那一步承担）。

        返回 (N, 2) float64，单位像素。
        """
        K, D, R, P = self._side_target(side)
        R, P = self._undistort_target(R, P, K)
        src = np.asarray(pts, np.float64).reshape(-1, 1, 2)
        return cv2.undistortPoints(src, K, D, R=R, P=P).reshape(-1, 2)

    def rectify_points(self, pts_left: np.ndarray, pts_right: np.ndarray):
        """把原始像素坐标校正到共面极线坐标系（三角测量前必做）。"""
        return (self.undistort_points("left", pts_left),
                self.undistort_points("right", pts_right))

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
        """校正后同名点的 y 之差（RMS，像素）。这是检验装置有没有被碰动的直接指标。

        ## 只对 ``stereo_rectify`` 成立（2026-10-24 加防御）

        "y 差 = 点到极线的距离"依赖**极线已被摆成水平**。``none_raw``（chain/90°）
        下极线是斜的，斜线上的两点 y 差可以很大而几何误差为零 —— 这个数字会
        又好看又错。与其给一个骗人的数，不如直接报错并指路。
        """
        if is_uncorrected(self.rectification):
            raise ValueError(
                "该标定未做极线校正（none_raw），y 差口径不适用；"
                "请改用 epipolar_error_by_mode('angled')")
        p1, p2 = self.rectify_points(pts_left, pts_right)
        n = min(len(p1), len(p2))
        if n == 0:
            return float("nan")
        dy = p1[:n, 1] - p2[:n, 1]
        return float(np.sqrt(np.mean(dy * dy)))

    # ------------------------------------------------------------------
    # 按模式分流的入口（2026-10-24 加）
    #
    # 为什么另开方法、而不改上面 `epipolar_error()` 的语义：
    # `quality["epipolar_rms"]` 是**落盘字段**，历史标定 JSON 里已经有这个数字，
    # 测试也把它当字面值断言。改它的语义（哪怕只对 angled）会让新旧标定的数字
    # 失去可比性，等于悄悄废掉"历史可对照"。所以旧方法一个字不动，
    # 需要 angled 口径的调用方走下面的 `epipolar_error_by_mode`。
    # ------------------------------------------------------------------
    def epipolar_metric_name(self, mode: str | None = None) -> str:
        """某一模式的极线误差口径（人读字符串，写进报告用）。

        两种模式的数字**不可比**，报告里必须写清楚是哪一种，
        否则 90° 装置会拿到一个"好看但无意义"的数字：

        - ``parallel``：校正后同名点的 **y 差 RMS**（校正后极线是水平线，
                  y 差就是点到极线的距离）；
        - ``angled``  ：**F 矩阵的点线距 RMS**（原始像素去畸变后仍在 K 像素系，
                  逐点算到对应极线的垂直距离）。大角度下校正图像被极端拉伸，
                  y 差不再等于几何误差，所以不能用同一把尺。

        `mode` 省略时取 `self.stereo_mode`（标定 meta 里记录的模式）。
        """
        if (mode or self.stereo_mode) == "angled":
            return "angled：F 矩阵极线点线距 RMS（px，原始像素去畸变后）"
        return "parallel：校正后 y 差 RMS（px）"

    def epipolar_error_by_mode(
        self,
        mode: str,
        pts_left: np.ndarray,
        pts_right: np.ndarray,
    ) -> float:
        """按**显式传入**的 `mode` 计算极线误差（像素 RMS）。

        调用方应传自己实际使用的模式（`rig.stereo_mode` 或会话记录的模式），
        不要依赖隐式默认 —— 报告里写错模式比数字大 10 倍更危险。

        - ``parallel``：直接转调 :meth:`epipolar_error`，**零行为变化**
          （落盘的 `quality["epipolar_rms"]` 与它同口径，仍可对照）；
        - ``angled``  ：F 矩阵点线距 RMS。校正图像在 90° 下被极端拉伸，
          校正后的 y 差与真实几何误差脱钩，必须换口径。
        """
        if mode == "angled":
            return self._epipolar_error_angled(pts_left, pts_right)
        if mode == "parallel":
            return self.epipolar_error(pts_left, pts_right)
        raise ValueError(f"未知的 stereo_mode：{mode!r}（只支持 'parallel' / 'angled'）")

    def _epipolar_error_angled(
        self, pts_left: np.ndarray, pts_right: np.ndarray
    ) -> float:
        """angled 口径：右图点到左图对应极线的垂直距离（RMS，px）。

        像素先 `undistortPoints(..., P=K)` 去畸变并留在 K 像素系 ——
        畸变是非线性的，不先去掉的话 F 只近似成立。
        `F` 优先用标定产物 `self.F`，没有就由 K/R/T 现场构造，
        两者与上面的去畸变方式同一套约定。
        """
        a = np.asarray(pts_left, np.float64).reshape(-1, 2)
        b = np.asarray(pts_right, np.float64).reshape(-1, 2)
        n = min(len(a), len(b))
        if n == 0:
            return float("nan")
        a, b = a[:n], b[:n]

        F = None
        if isinstance(self.F, np.ndarray) and self.F.shape == (3, 3):
            if np.all(np.isfinite(self.F)):
                F = self.F
        if F is None:
            F = fundamental_matrix_from_pose(self.K1, self.K2, self.R, self.T)

        K1, D1 = self.K1, self.D1 if self.D1 is not None else np.zeros(5)
        K2, D2 = self.K2, self.D2 if self.D2 is not None else np.zeros(5)
        pa = cv2.undistortPoints(a.reshape(-1, 1, 2), K1, D1, P=K1).reshape(-1, 2)
        pb = cv2.undistortPoints(b.reshape(-1, 1, 2), K2, D2, P=K2).reshape(-1, 2)

        dists: list[float] = []
        for p_left, p_right in zip(pa, pb):
            try:
                line = epipolar_line(F, p_left)
                d = point_line_distance(line, p_right)
            except ValueError:
                # 退化的极线（点在极点上 / F 病态）：这一个点不参与 RMS，
                # 不留 NaN 污染整体；全部退化时下面返回 nan。
                continue
            if np.isfinite(d):
                dists.append(d)
        if not dists:
            return float("nan")
        arr = np.asarray(dists, np.float64)
        return float(np.sqrt(np.mean(arr * arr)))

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
                # 落盘这一步必须带上：`none_raw` 与 `stereo_rectify` 的
                # `epipolar_error()` 语义不同，丢了它下游就分不清口径。
                "rectification": res.rectification,
                "fx_sigma_px": res.fx_sigma(),
                "depth_sigma_mm_at_0_6m": res.depth_sigma_mm(0.6),
            },
            meta=meta or {},
        )


def load_stereo_rig(path: str | Path) -> StereoRig:
    return StereoRig.load(path)


def save_stereo_rig(rig: StereoRig, path: str | Path):
    return rig.save(path)
