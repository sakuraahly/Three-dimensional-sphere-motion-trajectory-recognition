"""时间戳配对（金丝雀路径的底层计划器）。

为什么需要它
------------
原来 `PairSource` 假设左右帧号只差一个常数 `sync_offset`。普通 USB 相机
自由运行，采集线程偶尔会卡 0.3–0.9 s，卡顿期间少写一帧；从丢帧位置开始，
常数偏移就再也对不齐，三维结果会系统性错位。

本模块只做**纯计算**：吃两路 `perf_counter_s`，吐一张一一对应、单调递增的
`(left_index, right_index)` 计划。它不碰 OpenCV、不碰相机，因此可以单独测。

金丝雀策略
----------
默认仍走原来的索引配对；只有显式启用 `pairing="timestamp"`，或在
`pairing="canary"` 下计划通过覆盖率 / 残差门禁时才启用本模块。
计划不通过时由 `PairSource` 回退到原方案，不改变旧行为。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TimestampPairPlan:
    """一张只读的配对计划。"""

    pairs: tuple[tuple[int, int], ...]
    skipped_left: int
    skipped_right: int
    median_residual_ms: float
    max_residual_ms: float
    tolerance_ms: float

    @property
    def coverage(self) -> float:
        total = len(self.pairs) + self.skipped_left
        return len(self.pairs) / total if total else 0.0

    def summary(self) -> str:
        return (f"{len(self.pairs)} 对，覆盖 {self.coverage * 100:.1f}%，"
                f"残差中位 {self.median_residual_ms:.1f} ms / "
                f"最大 {self.max_residual_ms:.1f} ms，"
                f"跳过左 {self.skipped_left} / 右 {self.skipped_right}")


def read_time_sidecar(path: str | Path) -> np.ndarray:
    """读 `*_times.csv`，返回按帧号排序的 perf_counter 秒数组。

    兼容 BOM、空行、列顺序变化；只依赖 `frame` / `perf_counter_s` 两列。
    时间戳必须严格递增，否则直接报错 —— 静默排序会把“配对”变成猜。
    """
    p = Path(path)
    rows: list[tuple[int, float]] = []
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{p} 没有表头")
        need = {"frame", "perf_counter_s"}
        if not need.issubset(reader.fieldnames):
            raise ValueError(f"{p} 缺列 {sorted(need - set(reader.fieldnames))}")
        for row in reader:
            raw_i = (row.get("frame") or "").strip()
            raw_t = (row.get("perf_counter_s") or "").strip()
            if not raw_i or not raw_t:
                continue
            rows.append((int(raw_i), float(raw_t)))
    if not rows:
        raise ValueError(f"{p} 没有时间戳")
    rows.sort(key=lambda x: x[0])
    times = np.asarray([t for _i, t in rows], dtype=np.float64)
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError(f"{p} 的 perf_counter_s 不是严格递增")
    return times


def sidecar_for_video(video: str | Path) -> Path | None:
    """给 `left.mkv` 找 `left_times.csv`；不存在返回 None。"""
    p = Path(video)
    candidates = [
        p.with_name(f"{p.stem}_times.csv"),
        p.with_suffix(".times.csv"),
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def plan_timestamp_pairs(
    left_times: np.ndarray,
    right_times: np.ndarray,
    tolerance_s: float = 0.020,
) -> TimestampPairPlan:
    """为每个左帧找最近的右帧；单调、一一对应、超差就跳过。

    不能“每个左帧独立找最近右帧”：右路卡顿时，连续多个左帧会同时指向
    同一个右帧，产生重复配对。这里强制 `right_idx > last_right`，宁可跳过，
    也不让同一右帧被用两次。
    """
    lt = np.asarray(left_times, dtype=np.float64).reshape(-1)
    rt = np.asarray(right_times, dtype=np.float64).reshape(-1)
    if lt.size == 0 or rt.size == 0:
        return TimestampPairPlan((), len(lt), len(rt), 0.0, 0.0,
                                 float(tolerance_s) * 1000.0)
    if np.any(np.diff(lt) <= 0) or np.any(np.diff(rt) <= 0):
        raise ValueError("左右时间戳必须严格递增")

    tol = float(tolerance_s)
    pairs: list[tuple[int, int]] = []
    residuals: list[float] = []
    last_right = -1
    skipped_left = 0
    used_right: set[int] = set()
    for li, t in enumerate(lt):
        j = int(np.searchsorted(rt, t))
        best_j = -1
        best_dt = float("inf")
        for cand in (j - 1, j):
            if 0 <= cand < rt.size and cand > last_right:
                dt = abs(float(rt[cand]) - t)
                if dt < best_dt:
                    best_j, best_dt = cand, dt
        if best_j >= 0 and best_dt <= tol:
            pairs.append((li, best_j))
            used_right.add(best_j)
            last_right = best_j
            residuals.append(best_dt * 1000.0)
        else:
            skipped_left += 1

    skipped_right = rt.size - len(used_right)
    med = float(np.median(residuals)) if residuals else 0.0
    mx = float(np.max(residuals)) if residuals else 0.0
    return TimestampPairPlan(tuple(pairs), skipped_left, skipped_right, med, mx,
                             tol * 1000.0)


def canary_safe(plan: TimestampPairPlan, min_coverage: float = 0.75) -> tuple[bool, str]:
    """金丝雀是否允许接管。

    只要有一项不满足就回退旧方案；这不是“质量警告”，是路由门禁。
    覆盖率门槛 75%：掉帧严重的会话仍能保住大部分有效帧，但不能再低到
    只剩零散数据。
    """
    if not plan.pairs:
        return False, "时间戳没有产生任何配对"
    if plan.coverage < min_coverage:
        return False, (f"覆盖率 {plan.coverage * 100:.1f}% < {min_coverage * 100:.0f}%")
    if plan.max_residual_ms > plan.tolerance_ms:
        return False, (f"最大残差 {plan.max_residual_ms:.1f} ms > "
                       f"容差 {plan.tolerance_ms:.1f} ms")
    return True, plan.summary()
