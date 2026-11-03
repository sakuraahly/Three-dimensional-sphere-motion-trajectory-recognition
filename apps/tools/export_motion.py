"""把三维轨迹导出成**别人能直接拿去滤波**的运动数据（位置 + 速度 + 加速度）。

为什么要有这个工具
------------------
`traj3d.csv` 只有位置。而"速度/加速度"是下游（卡尔曼、Savitzky-Golay 对比、
轨迹识别）真正要的东西，每个人各写一份差分就会各出一套数 —— 本项目已经因为
"同一趟两个深度"返工过，同一趟两个速度只会更糟。

所以：**平滑与求导只在 `stereo/motion.py` 里实现一次**，本工具只负责把它跑一遍、
按固定列序落盘，并把口径（单位 / 坐标系 / 段 / 哪些点没被处理）写进
`motion_meta.json`。下游不需要读代码就能知道该怎么用。

## 输出契约（这两条是给别人用的，不是内部细节）

1. **缺值写空串**，不写 `0`、不写 `nan`。`0` 会被当成"真的静止在那里"，
   `nan` 在 pandas 里是 float 却会被 `to_csv` 写成 `nan` —— 两者都会让下游
   "悄悄算下去"，而空串至少在读入时是个显式的 `None`。
2. **只有 `matched=1` 的点才进文件**。没匹配上的帧**直接缺席**，不补齐：
   把缺口插值成直线会造出一段"不存在的匀速运动"，卡尔曼会当成真观测吃进去。

用法::

    python apps/tools/export_motion.py \\
        --traj outputs/rec/20261103-093820-gui/traj3d.csv \\
        --out  outputs/rec/20261103-093820-gui/motion.csv

    # 不要平滑（只看原始差分长什么样）
    python apps/tools/export_motion.py --traj ... --out ... --no-smooth

    # 换窗口 / 阶数（默认 0.20 s / 2 阶是实测扫出来的，见 stereo/motion.py）
    python apps/tools/export_motion.py --traj ... --out ... --window-s 0.30 --poly 2

输入支持 `traj3d.csv` 和 `traj3d.json`（后者读 `samples` 里的 `frame/t/xyz/matched`）。

退出码：0 = 出了文件；1 = 读不了（文件不存在 / 没有匹配点 / JSON 结构不对），
原因打在 stderr，**不打 traceback** —— 这个工具是给人手敲的，栈回溯只会挡住那行原因。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

# --- 让脚本在任意目录深度都能找到项目根（按标记上溯，不数层数）---
ROOT = Path(__file__).resolve().parent
while not (ROOT / "sphere_tracker").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from stereo.motion import kinematics, split_segments

COLUMNS = ("frame", "t", "x", "y", "z", "vx", "vy", "vz",
           "speed", "ax", "ay", "az", "accel", "segment", "smoothed")

#: 写进 meta 的已知局限。**每条都是下游会踩的坑，不是免责声明。**
LIMITATIONS = [
    "只含 matched=1 的点；未匹配的帧直接缺席（不填 0、不插值）。跨帧对齐请按 frame 联结。",
    "segment 不同的点**不可**一起差分、不可连续滤波；滤波器必须在段边界重置状态。"
    "segment=-1 表示单点段（凑不出差分），其速度为 nan。",
    "段首尾是单边差分，精度比段内中心差分低一档；要一致精度需自行丢掉段首尾点。",
    "加速度是位置的二阶差分，噪声被放大（窗口内 ~4 倍量级）；实测原始 |a| p95 ≈ 15.5 m/s²，"
    "而这段是 0.18 m/s 的慢摆 —— 不做平滑的加速度不可当真值。",
    "时间戳不是严格等间隔（实测中位 33.2 ms、p95 66.4 ms）；做 Savitzky-Golay / 卡尔曼前"
    "应先按段用 stereo.motion.resample_uniform 重采样到固定 dt。",
    "smoothed=false 表示该点没参与平滑（所在段短于 poly+2 或被跳过），此时位置是原始测量值，"
    "速度/加速度也是由原始值差分得到。",
]


def _fail(msg: str) -> int:
    print(f"[export_motion] {msg}", file=sys.stderr)
    return 1


def read_csv_traj(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读 `traj3d.csv`，只取 matched=1 且 X/Y/Z 齐全的行。

    返回 `(frame, t, P)`，P 是 (N,3) 米。顺序按文件顺序（本来就是帧序）。
    """
    frames: list[int] = []
    times: list[float] = []
    pts: list[list[float]] = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        need = {"frame", "t", "X", "Y", "Z"}
        missing = need - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"缺列：{', '.join(sorted(missing))}")
        for row in reader:
            try:
                matched = int(float(row.get("matched") or 0)) == 1
            except ValueError:
                matched = False
            if not matched:
                continue
            raw = [row.get(k, "") for k in ("X", "Y", "Z")]
            if any(v is None or str(v).strip() == "" for v in raw):
                continue                     # 标了 matched 但坐标空：当成没有，不填 0
            frames.append(int(float(row["frame"])))
            times.append(float(row["t"]))
            pts.append([float(v) for v in raw])
    P = np.asarray(pts, dtype=float).reshape(-1, 3)
    return (np.asarray(frames, dtype=int), np.asarray(times, dtype=float), P)


def read_json_traj(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """读 `traj3d.json` 的 `samples`，返回 `(frame, t, P, coordinate_system)`。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    samples = data.get("samples")
    if not isinstance(samples, list):
        raise ValueError("JSON 里没有 samples 数组（不是标准 traj3d.json？）")
    frames: list[int] = []
    times: list[float] = []
    pts: list[list[float]] = []
    for s in samples:
        xyz = s.get("xyz")
        if not s.get("matched") or not isinstance(xyz, list) or len(xyz) != 3:
            continue
        if any(v is None for v in xyz):
            continue
        frames.append(int(s["frame"]))
        times.append(float(s["t"]))
        pts.append([float(v) for v in xyz])
    P = np.asarray(pts, dtype=float).reshape(-1, 3)
    return (np.asarray(frames, dtype=int), np.asarray(times, dtype=float), P,
            str(data.get("coordinate_system") or ""))


def _cell(value: float) -> str:
    """数值 → CSV 单元：nan / inf 一律空串（口径见模块文档第 1 条）。"""
    if value is None or not np.isfinite(value):
        return ""
    return f"{value:.6f}"


def write_motion_csv(path: Path, frame: np.ndarray, kin: dict) -> int:
    """写 motion.csv，返回写入的数据行数。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    t, P, V, A = kin["t"], kin["P"], kin["V"], kin["A"]
    speed, accel, seg, smooth = kin["speed"], kin["accel"], kin["segment"], kin["smoothed"]
    n = 0
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for i in range(t.size):
            w.writerow([
                int(frame[i]), _cell(t[i]),
                _cell(P[i, 0]), _cell(P[i, 1]), _cell(P[i, 2]),
                _cell(V[i, 0]), _cell(V[i, 1]), _cell(V[i, 2]), _cell(speed[i]),
                _cell(A[i, 0]), _cell(A[i, 1]), _cell(A[i, 2]), _cell(accel[i]),
                int(seg[i]), "true" if smooth[i] else "false",
            ])
            n += 1
    return n


def build_meta(
    traj_path: Path,
    out_path: Path,
    frame: np.ndarray,
    kin: dict,
    *,
    smooth: bool,
    window_s: float,
    poly: int,
    max_gap_s: float,
    coordinate: str,
    source_format: str,
) -> dict:
    t = kin["t"]
    seg_ids = kin["segment"]
    segments = []
    for k, seg in enumerate(split_segments(t, max_gap_s)):
        segments.append({
            "index": k,
            "start_t": round(float(t[seg[0]]), 6),
            "end_t": round(float(t[seg[-1]]), 6),
            "n_points": int(seg.size),
            "smoothed": bool(kin["smoothed"][seg].all()),
        })
    n = int(t.size)
    return {
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": traj_path.as_posix(),
        "source_format": source_format,
        "output": out_path.as_posix(),
        "n_points": n,
        "n_segments": len(segments),
        "segments": segments,
        "smoothing": {
            "enabled": bool(smooth),
            "method": "savitzky-golay" if smooth else "none",
            "window_s": float(window_s),
            "poly": int(poly),
            "max_gap_s": float(max_gap_s),
            "impl": "stereo/motion.py::kinematics",
            "n_smoothed": int(kin["smoothed"].sum()),
            "n_velocity": int(np.isfinite(kin["speed"]).sum()),
            "n_acceleration": int(np.isfinite(kin["accel"]).sum()),
        },
        "units": {"t": "s", "position": "m", "velocity": "m/s", "acceleration": "m/s^2"},
        "frame": {
            "coordinate_system": coordinate or "left_camera",
            "x": "right", "y": "down", "z": "forward",
            "note": "轴定义按左相机系：X 向右、Y 向下（与图像 v 同向）、Z 向前。"
                    "JSON 输入时沿用其 coordinate_system 字段。",
        },
        "missing_values": "空串（不是 0、不是 nan 字符串）",
        "known_limitations": LIMITATIONS,
    }


def _stats(kin: dict) -> str:
    """一行摘要：|v| 与 |a| 的中位 / p95。没有有限值时说"无"，不打印 nan。"""
    def fmt(values: np.ndarray) -> str:
        good = values[np.isfinite(values)]
        if good.size == 0:
            return "无"
        return (f"中位 {np.median(good):.3f}  p95 {np.percentile(good, 95):.3f}"
                f"  n={good.size}")

    return (f"  |v| : {fmt(kin['speed'])} m/s\n"
            f"  |a| : {fmt(kin['accel'])} m/s²")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="三维轨迹 → 运动量（位置/速度/加速度），缺值写空串",
    )
    ap.add_argument("--traj", required=True,
                    help="输入 traj3d.csv（或 traj3d.json）")
    ap.add_argument("--out", required=True, help="输出 motion.csv")
    ap.add_argument("--window-s", type=float, default=0.20,
                    help="Savitzky-Golay 窗口（秒），默认 0.20（实测拐点，别随手改）")
    ap.add_argument("--poly", type=int, default=2, help="多项式阶数，默认 2")
    ap.add_argument("--max-gap-s", type=float, default=0.10,
                    help="相邻点时间差超过它就断开成新段，默认 0.10 s")
    ap.add_argument("--no-smooth", action="store_true",
                    help="不做平滑，直接对原始位置差分（加速度会非常抖）")
    args = ap.parse_args(argv)

    traj_path = Path(args.traj)
    out_path = Path(args.out)
    if not traj_path.exists():
        return _fail(f"输入文件不存在：{traj_path}")
    if traj_path.suffix.lower() == ".json":
        source_format = "json"
        try:
            frame, t, P, coordinate = read_json_traj(traj_path)
        except (OSError, ValueError, KeyError) as exc:
            return _fail(f"读不了 {traj_path}：{exc}")
    else:
        source_format = "csv"
        coordinate = "left_camera"           # traj3d.csv 的口径就是左相机系
        try:
            frame, t, P = read_csv_traj(traj_path)
        except (OSError, ValueError, KeyError) as exc:
            return _fail(f"读不了 {traj_path}：{exc}")

    if t.size == 0:
        return _fail(f"{traj_path} 里没有 matched=1 的点 —— 没有东西可导出")
    if t.size < 2:
        return _fail(f"{traj_path} 只有 {t.size} 个匹配点，凑不出速度")

    order = np.argsort(t, kind="stable")     # 时间必须单调，否则差分无意义
    if not np.array_equal(order, np.arange(t.size)):
        frame, t, P = frame[order], t[order], P[order]

    kin = kinematics(t, P, smooth=not args.no_smooth, window_s=args.window_s,
                     poly=args.poly, max_gap_s=args.max_gap_s)

    n = write_motion_csv(out_path, frame, kin)
    meta_path = out_path.with_name(out_path.stem + "_meta.json")
    meta = build_meta(traj_path, out_path, frame, kin, smooth=not args.no_smooth,
                      window_s=args.window_s, poly=args.poly,
                      max_gap_s=args.max_gap_s, coordinate=coordinate,
                      source_format=source_format)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")

    smooth_pct = 100.0 * meta["smoothing"]["n_smoothed"] / n
    print(f"输入：{traj_path}（{source_format}）")
    print(f"输出：{out_path}（{n} 行）+ {meta_path}")
    print(f"  段数：{meta['n_segments']}（max_gap_s={args.max_gap_s}）"
          f"  平滑：{'开' if not args.no_smooth else '关'}"
          f"（window={args.window_s}s / poly={args.poly}）"
          f"  已平滑点 {smooth_pct:.1f}%")
    print(_stats(kin))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
