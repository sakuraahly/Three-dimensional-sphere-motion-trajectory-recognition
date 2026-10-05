"""双路视频同步。

两台独立相机拍到的帧**不是同一时刻**的。这一点的影响要说清楚：

  - **标定不需要同步**：标定只收"板停稳"的帧，静止目标在任意时刻看起来都一样；
  - **拍运动物体必须同步**：否则三角测量会把"左相机 t1 时刻的球"和
    "右相机 t2 时刻的球"当成同一个点，算出来的三维坐标是错的，
    而且错得很隐蔽（看起来只是个正常的点）。

同步办法（按投入排序，本模块实现第 1 种）：

  1. **闪光/拍手对齐**（本模块）：录一段有闪光或拍手的视频，
     两路的亮度序列会在同一物理事件处出现尖峰，尖峰的位置差就是帧偏移。
  2. 硬件同步：工业相机的外触发，最可靠但要换设备。
  3. UVC 时间戳：需要相机支持。

用法::

    from stereo.sync import SyncConfig, find_offset, brightness_series

    sl = brightness_series(left_frames)     # 逐帧亮度
    sr = brightness_series(right_frames)
    res = find_offset(sl, sr, SyncConfig())
    print(res)                              # offset = 8，表示 left[i] 对应 right[i+8]
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SyncConfig:
    max_offset: int = 180        # 最多搜索多少帧的偏移
    min_overlap: int = 30        # 重叠帧数少于这个就不算数（否则相关性不可信）
    smooth: int = 1              # 亮度序列平滑窗口（1 = 不平滑）


@dataclass
class SyncResult:
    offset: int                  # left[i] 对应 right[i + offset]
    confidence: float            # 归一化互相关系数峰值（1 = 完全一致）
    peak_count: int = 0          # 有几个明显的闪光事件（越多越可信）
    note: str = ""

    def __str__(self) -> str:
        return (f"帧偏移 {self.offset:+d}（left[i] <-> right[i{self.offset:+d}]），"
                f"置信度 {self.confidence:.3f}，闪光事件 {self.peak_count} 个")

    @property
    def usable(self) -> bool:
        return self.confidence >= 0.35 and self.peak_count >= 1


def brightness_series(frames, smooth: int = 1) -> np.ndarray:
    """逐帧平均亮度。frames 可以是 ndarray 序列，也可以是灰度图序列。"""
    vals = []
    for f in frames:
        if f is None:
            continue
        g = f if f.ndim == 2 else cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        vals.append(float(g.mean()))
    arr = np.asarray(vals, dtype=np.float64)
    if smooth > 1 and arr.size >= smooth:
        k = np.ones(smooth) / smooth
        arr = np.convolve(arr, k, mode="same")
    return arr


def _corr_at(da: np.ndarray, db: np.ndarray, offset: int) -> tuple[float, int]:
    """left[i] 对应 right[i+offset] 时，两条差分序列的归一化相关系数。"""
    n = len(da)
    if offset >= 0:
        a, b = da[0: n - offset], db[offset: n]
    else:
        a, b = da[-offset: n], db[0: n + offset]
    m = len(a)
    if m < 2:
        return 0.0, 0
    a = a - a.mean()
    b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0, m
    return float(np.dot(a, b) / (na * nb)), m


def find_offset(
    series_left: np.ndarray,
    series_right: np.ndarray,
    cfg: SyncConfig | None = None,
) -> SyncResult:
    """用亮度突变（闪光/拍手）估计两路视频的帧偏移。

    做法：对两条亮度序列求差分（突变被放大、缓慢变化被压掉），
    再在候选偏移上做归一化互相关，取相关系数最大的那个偏移。
    比"只找最大尖峰"稳，因为多个闪光事件会一起投票。
    """
    cfg = cfg or SyncConfig()
    l = np.asarray(series_left, np.float64)
    r = np.asarray(series_right, np.float64)
    if l.size < 3 or r.size < 3:
        return SyncResult(0, 0.0, 0, "序列太短")

    dl, dr = np.diff(l), np.diff(r)
    n = min(len(dl), len(dr))

    best_off, best_c, best_m = 0, -1.0, 0
    for off in range(-cfg.max_offset, cfg.max_offset + 1):
        c, m = _corr_at(dl[:n], dr[:n], off)
        if m < cfg.min_overlap:
            continue
        if c > best_c:
            best_off, best_c, best_m = off, c, m

    # 数一数有几个明显的闪光事件：差分绝对值超过 4 倍中位绝对偏差的峰
    med = float(np.median(np.abs(dl))) + 1e-9
    peaks = int(np.sum(np.abs(dl) > max(4.0 * med, 1.0)))
    note = ""
    if best_c < 0.35:
        note = ("相关性偏低，可能没拍到闪光，或两路拍的不是同一段。"
                "建议重新录一段：先静止，中间打一次闪光/拍一次手。")
    return SyncResult(best_off, max(best_c, 0.0), peaks, note)


def apply_offset(offset: int) -> str:
    """把偏移换算成"取帧"的说法，避免正负号搞反。"""
    if offset == 0:
        return "两路逐帧对齐"
    if offset > 0:
        return f"左路第 i 帧 对应 右路第 i+{offset} 帧（左路比右路早 {offset} 帧）"
    return f"左路第 i 帧 对应 右路第 i{offset} 帧（左路比右路晚 {-offset} 帧）"
