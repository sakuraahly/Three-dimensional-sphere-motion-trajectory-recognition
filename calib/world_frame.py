"""世界坐标系：把左相机系的点变换到"棋盘所在的地面"。

为什么需要
----------
`stereo/triangulate.py` 输出的是**左相机系**（parallel 是校正后左相机系）：
原点在光心、X 右 / Y 下 / Z 前。这个坐标系对"球怎么飞"没有物理意义 ——
重力方向、地面、球桌都在画面之外。世界系的作用就是把它对上：
棋盘平放在地面（或任一已知平面）上，用**左相机**拍一张，PnP 解出
`T_cam_board`，取逆即得 `T_world_cam`。

命名沿用项目其它地方（`calib/relative_pose.py`）的 `T_A_B` 约定：
**把 B 系的点变到 A 系**，即 `p_A = T_A_B @ p_B`。

两个必须知道的约定
------------------
1. **原点取画面里最"左下"的那个内角点**。棋盘对 180° 旋转是对称的，
   SB 与经典检测器的起点正好差 180°（`calib/board.py` 的文档串实测过）；
   不归一化的话，同一块板换一个检测器就会得到差 180° 的世界系，
   下游看到的 X/Y 会毫无征兆地反过来。这里按图像坐标（`x - y` 最小）
   选起点，只做 180° 的重新编号，**几何拟合与 rmse 完全不变**。
   代价：相机若装歪 90°，「左下」仍然可算，但已经不等于物理上的左下角。

2. **Z 是棋盘法向，不是"重力反方向"**。棋盘打印面朝着相机时，
   `BoardSpec.object_points()` 的 `X×Y` 指向**背离相机**的一侧：
   板平放在地面、图案朝上拍时，默认的 +Z 指向**地面以下**。
   要"Z 朝上"传 `ground_axis="-z"`（同时 Y 反向，保持右手系）。
   这个符号只能靠人确认，**程序不替你猜** —— `describe()` 会把它写出来。

限制（必须如实说）
------------------
只处理"平面已知且水平"的情形。棋盘不平、板没放平、或相机与板不同侧，
世界系都会歪，而 `board_rmse_px` **看不出来** —— 它只反映 PnP 拟合得好不好，
不反映"板是不是真平放在地上"。所以 CLI 会要求用户复核这一点。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .board import BoardSpec, detect_board

if TYPE_CHECKING:                                    # 只为类型标注，不引入运行期依赖
    from .params import StereoRig

#: 棋盘 PnP 的重投影误差上限（像素）。超过就说明这一帧不适合当世界系：
#: 板斜得太厉害、角点糊、或者图里根本不是这块板。
MAX_BOARD_RMSE_PX = 2.0

_ALLOWED_GROUND_AXIS = ("z", "-z")


def _check_T(T: np.ndarray, where: str) -> np.ndarray:
    """校验一个 4x4 齐次变换，任何一条不满足就抛 ValueError 并说明是哪一条。

    `where` 只用于报错前缀（文件名 / 调用点），让用户知道该去改哪个文件。
    """
    arr = np.asarray(T, dtype=np.float64)
    if arr.shape != (4, 4):
        raise ValueError(f"{where}：必须是 4x4（实际 {arr.shape}）")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{where}：含 NaN / Inf")
    if not np.allclose(arr[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-9, rtol=0.0):
        raise ValueError(
            f"{where}：最后一行必须是 [0,0,0,1]（实际 {arr[3].tolist()}）")
    R = arr[:3, :3]
    err = float(np.max(np.abs(R.T @ R - np.eye(3))))
    if err > 1e-6:
        raise ValueError(
            f"{where}：旋转块不正交，R^T·R 与单位阵最大偏差 {err:.3e}（容差 1e-6）")
    det = float(np.linalg.det(R))
    if not det > 0.0:
        raise ValueError(
            f"{where}：旋转块 det(R) = {det:.6f}，必须 > 0（<0 表示镜像，不是旋转）")
    return arr


@dataclass
class WorldFrame:
    """世界 ← 左相机 的刚体变换，外加"它是怎么来的"。

    `T_world_cam` 是 4x4：`p_world = T_world_cam @ p_left_camera`。
    """

    T_world_cam: np.ndarray
    source: str
    units: str = "m"
    note: str = ""
    board_rmse_px: float | None = None
    created_at: str = ""

    def __post_init__(self) -> None:
        self.T_world_cam = _check_T(self.T_world_cam, "T_world_cam")

    # ------------------------------------------------------------------
    def apply(self, points: np.ndarray | None) -> np.ndarray | None:
        """左相机系的 (N,3) 点 -> 世界系。`None` 原样返回（便于直通空数据）。"""
        if points is None:
            return None
        arr = np.asarray(points, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"apply() 需要 (N,3) 的点，实际 shape={arr.shape}")
        R = self.T_world_cam[:3, :3]
        t = self.T_world_cam[:3, 3]
        return arr @ R.T + t

    def describe(self) -> str:
        rmse = "—" if self.board_rmse_px is None else f"{self.board_rmse_px:.2f} px"
        return (f"世界系[{self.source}]：原点=棋盘内角点；X=列向；Y=行向；"
                f"Z=地面法向；rmse={rmse}")

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        T = np.asarray(self.T_world_cam, np.float64)
        head = {
            "version": 1,
            "created_at": self.created_at or time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": self.source,
            "units": self.units,
            "note": self.note,
            "board_rmse_px": self.board_rmse_px,
        }
        # 矩阵**一行一个 4x4 行**：默认 json.dumps(indent=2) 会把每个元素拆到
        # 单独一行（16 行），人眼根本读不出这是不是单位阵。
        # 用 17 位有效数字，保证 load() 后逐元素误差远小于 1e-12。
        lines = ["{"]
        for key, value in head.items():
            lines.append(f'  {json.dumps(key)}: {json.dumps(value, ensure_ascii=False)},')
        lines.append('  "T_world_cam": [')
        for i, row in enumerate(T):
            body = ", ".join(f"{float(v):.17g}" for v in row)
            lines.append(f"    [{body}]" + ("," if i < len(T) - 1 else ""))
        lines.extend(["  ]", "}"])
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "WorldFrame":
        """读世界系文件；**任何一条校验不过都抛 ValueError**（绝不回退成单位矩阵）。"""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"没有世界系文件：{p}")
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{p} 不是合法 JSON：{exc}") from exc
        if not isinstance(data, dict) or "T_world_cam" not in data:
            raise ValueError(f"{p} 缺 T_world_cam 字段")
        try:
            T = np.asarray(data["T_world_cam"], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{p}：T_world_cam 不是数值矩阵（{exc}）") from exc
        rmse = data.get("board_rmse_px")
        try:
            return cls(
                T_world_cam=T,
                source=str(data.get("source") or "unknown"),
                units=str(data.get("units") or "m"),
                note=str(data.get("note") or ""),
                board_rmse_px=None if rmse is None else float(rmse),
                created_at=str(data.get("created_at") or ""),
            )
        except ValueError as exc:
            # 矩阵校验在 __post_init__ 里；这里补上**文件名**，
            # 否则用户只知道"矩阵不合法"，不知道该改哪个文件。
            message = str(exc)
            raise ValueError(message if str(p) in message else f"{p}：{message}") from exc

    # ------------------------------------------------------------------
    @classmethod
    def from_camera_board(
        cls,
        T_cam_board: np.ndarray,
        *,
        source: str = "manual",
        units: str = "m",
        note: str = "",
        board_rmse_px: float | None = None,
        ground_axis: str = "z",
    ) -> "WorldFrame":
        """由**已知的** `T_cam_board`（板→左相机）构造世界系。

        留给"没有图、但位姿已经从别处量出来"的场景（例如外部测量/合成真值），
        也是 `from_board()` 与测试共用的那一段数学。
        """
        if ground_axis not in _ALLOWED_GROUND_AXIS:
            raise ValueError(
                f"ground_axis 只支持 {_ALLOWED_GROUND_AXIS}，收到 {ground_axis!r}")
        T_cb = _check_T(T_cam_board, "T_cam_board")
        R_cb, t_cb = T_cb[:3, :3], T_cb[:3, 3]
        # 世界轴相对棋盘轴：默认同向；"-z" 让 Z 翻到另一侧（绕棋盘 X 轴转 180°，
        # 所以 Y 也会反向 —— 这是保持右手系的代价，写在模块文档里）。
        R_wb = np.eye(3) if ground_axis == "z" else np.diag([1.0, -1.0, -1.0])
        T_world_cam = np.eye(4)
        T_world_cam[:3, :3] = R_wb @ R_cb.T
        T_world_cam[:3, 3] = -R_wb @ R_cb.T @ t_cb
        return cls(
            T_world_cam=T_world_cam,
            source=source,
            units=units,
            note=note,
            board_rmse_px=board_rmse_px,
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )


def _left_bottom_order(objp: np.ndarray, corners: np.ndarray):
    """让第 1 个角点是画面里最"左下"的那个，消除检测器之间的 180° 差。

    判据用图像坐标的 `x - y`（y 向下，越小越靠下、越靠左）。
    需要换起点时，物体点整体转 180°（原点落到新起点），角点整份反序；
    两者是同一个物理对应，所以 PnP 的残差不变。
    """
    def score(px: np.ndarray) -> float:
        return float(px[0] - px[1])

    if score(corners[0]) <= score(corners[-1]):
        return objp, corners
    span = objp[:, :2].max(axis=0)
    rotated = objp.copy()
    rotated[:, 0] = span[0] - objp[:, 0]
    rotated[:, 1] = span[1] - objp[:, 1]
    return rotated, corners[::-1]


def from_board(
    image_path: str | Path,
    board_spec: BoardSpec,
    rig: "StereoRig",
    *,
    ground_axis: str = "z",
) -> WorldFrame:
    """用一张"棋盘平放在地面（或已知平面）上"的**左相机**图求世界系。

    流程：`detect_board` -> `solvePnP`（左路 K1/D1）-> `T_cam_board` -> 取逆。
    检测不完整、或者重投影 rmse 超过 :data:`MAX_BOARD_RMSE_PX` 一律抛
    `ValueError` 并说明怎么重拍 —— **不给一个"凑合能用"的世界系**。
    """
    if ground_axis not in _ALLOWED_GROUND_AXIS:
        raise ValueError(
            f"ground_axis 只支持 {_ALLOWED_GROUND_AXIS}，收到 {ground_axis!r}")
    path = Path(image_path)
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"打不开图片：{path}")
    if not isinstance(board_spec, BoardSpec):
        raise TypeError(f"board_spec 必须是 BoardSpec，收到 {type(board_spec).__name__}")

    obs = detect_board(gray, board_spec)
    if not obs.found or obs.corners is None:
        raise ValueError(
            f"{path}：没检测到 {board_spec.cols}x{board_spec.rows} 内角点"
            f"（{obs.note or '原因未知'}）。请重拍：棋盘完整落在画面内、"
            "占画面 10%~30%、对焦清晰、板面与镜头不要接近平行")
    corners = obs.corners.reshape(-1, 2).astype(np.float64)
    if corners.shape[0] != board_spec.n_points:
        raise ValueError(
            f"{path}：只检测到 {corners.shape[0]}/{board_spec.n_points} 个内角点，"
            "检测不完整。请把整块棋盘放进画面（四边留白）后重拍")

    objp, corners = _left_bottom_order(board_spec.object_points().astype(np.float64), corners)

    K = np.asarray(rig.K1, dtype=np.float64)
    D = np.zeros(5) if getattr(rig, "D1", None) is None else np.asarray(rig.D1, np.float64)
    ok, rvec, tvec = cv2.solvePnP(objp, corners.reshape(-1, 1, 2), K, D,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise ValueError(f"{path}：solvePnP 失败（{board_spec.describe()}）。请重拍后再试")

    projected, _ = cv2.projectPoints(objp, rvec, tvec, K, D)
    diff = projected.reshape(-1, 2) - corners
    rmse = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
    if not np.isfinite(rmse) or rmse > MAX_BOARD_RMSE_PX:
        raise ValueError(
            f"{path}：棋盘重投影 rmse = {rmse:.2f} px，超过上限 "
            f"{MAX_BOARD_RMSE_PX:.1f} px。请重拍：板要平整、正对镜头别太斜、"
            "保证清晰（模糊/运动模糊都会把角点拉偏）")

    T_cam_board = np.eye(4)
    T_cam_board[:3, :3] = cv2.Rodrigues(rvec)[0]
    T_cam_board[:3, 3] = tvec.ravel()

    note = (f"{board_spec.describe()}；原点=画面左下内角点；"
            f"ground_axis={ground_axis}（z 为棋盘法向）")
    return WorldFrame.from_camera_board(
        T_cam_board,
        source="board_pnp",
        note=note,
        board_rmse_px=rmse,
        ground_axis=ground_axis,
    )
