"""标准数据：双目种子 / 三维轨迹 / 可视化数据的读写与导出。

为什么单独成模块
----------------
`run_stereo_video.py` 之前只写 CSV + PNG，GUI/网页/别的脚本要各自解析；
坐标单位、坐标系、同步偏移、seed 都散在日志里。这里把"一次三维结果"
固定成带 schema/version 的 JSON，别人只读一种格式。

三种文件（都在会话的 `derived/` 下）：

    seeds.json          人工点选/命令行的左右 seed（坐标 + 半径 + 帧号）
    traj3d.json         标准三维轨迹（元数据 + 逐帧 samples）
    visualization.json  给可视化程序用的精简数据（trajectory + bounds + style）

坐标系：`rectified_left_camera`，单位米；XYZ 轴定义见 `docs/数据标准.md`。
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

SEEDS_SCHEMA = "sphere-stereo-seeds"
TRAJECTORY_SCHEMA = "sphere-stereo-trajectory"
VISUALIZATION_SCHEMA = "sphere-stereo-visualization"
VERSION = 1


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _session_json(session_dir: Path) -> dict:
    path = Path(session_dir) / "session.json"
    if not path.exists():
        raise FileNotFoundError(f"没有 session.json：{session_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_seeds(
    session_dir: str | Path,
    left: tuple[float, float, float],
    right: tuple[float, float, float],
    *,
    seed_frame: int = 0,
    image_size: tuple[int, int] | None = None,
    source: str = "manual",
    note: str = "",
) -> Path:
    """写 `derived/seeds.json`。left/right 是 (x, y, radius) 原图像素坐标。"""
    session_dir = Path(session_dir)
    session = _session_json(session_dir)
    if image_size is None:
        image_size = (int(session.get("left", {}).get("width", 0)),
                      int(session.get("left", {}).get("height", 0)))
    payload = {
        "schema": SEEDS_SCHEMA,
        "version": VERSION,
        "session": session.get("id", session_dir.name),
        "created": _now(),
        "image_size": [int(image_size[0]), int(image_size[1])],
        "seed_frame": int(seed_frame),
        "source": source,
        "note": note,
        "left": {"x": float(left[0]), "y": float(left[1]), "radius": float(left[2])},
        "right": {"x": float(right[0]), "y": float(right[1]), "radius": float(right[2])},
    }
    path = session_dir / "derived" / "seeds.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_seeds(path: str | Path) -> dict:
    """读 seeds.json；也接受会话目录（自动找 derived/seeds.json）。"""
    p = Path(path)
    if p.is_dir():
        p = p / "derived" / "seeds.json"
    if not p.exists():
        raise FileNotFoundError(f"没有 seeds 文件：{p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if data.get("schema") != SEEDS_SCHEMA:
        raise ValueError(f"不是 {SEEDS_SCHEMA}：{p}")
    if int(data.get("version", 0)) != VERSION:
        raise ValueError(f"不支持的 seeds version：{data.get('version')}")
    for side in ("left", "right"):
        item = data.get(side) or {}
        for key in ("x", "y", "radius"):
            if key not in item:
                raise ValueError(f"seeds.{side} 缺 {key}")
    return data


def read_trajectory_csv(path: str | Path) -> list[dict[str, Any]]:
    """把 `traj3d*.csv` 读成 samples；空字段 -> None。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"没有轨迹 CSV：{path}")
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        col = {name: i for i, name in enumerate(header)}

        def cell(row: list[str], key: str) -> str:
            i = col.get(key, -1)
            return (row[i] if 0 <= i < len(row) else "").strip()

        def num(row: list[str], key: str) -> float | None:
            raw = cell(row, key)
            return None if raw == "" else float(raw)

        for row in reader:
            if not row or len(row) < 3:
                continue
            if len(row) < len(header):
                row = row + [""] * (len(header) - len(row))
            matched = int(float(cell(row, "matched") or 0)) == 1
            out.append({
                "frame": int(float(cell(row, "frame"))),
                "t": float(cell(row, "t")),
                "matched": matched,
                "xyz": [num(row, "X"), num(row, "Y"), num(row, "Z")] if matched else None,
                "uv_left": [num(row, "u_left"), num(row, "v_left")] if matched else None,
                "uv_right": [num(row, "u_right"), num(row, "v_right")] if matched else None,
                "epipolar_px": num(row, "epipolar_px") if matched else None,
                "residual_px": num(row, "residual_px") if matched else None,
                "confident": bool(int(float(cell(row, "confident") or 0))) if matched else False,
                "method_left": cell(row, "method_left"),
                "method_right": cell(row, "method_right"),
            })
    return out


def build_trajectory_payload(
    session_dir: str | Path,
    *,
    csv_name: str = "traj3d.csv",
) -> dict:
    """把 session + seeds + CSV 合成标准轨迹 JSON。"""
    session_dir = Path(session_dir)
    session = _session_json(session_dir)
    seeds = None
    seed_path = session_dir / "derived" / "seeds.json"
    if seed_path.exists():
        seeds = load_seeds(seed_path)
    samples = read_trajectory_csv(session_dir / "derived" / csv_name)
    traj = [s["xyz"] for s in samples if s["matched"] and s["xyz"] is not None]
    bounds = None
    if traj:
        xs = [p[0] for p in traj]
        ys = [p[1] for p in traj]
        zs = [p[2] for p in traj]
        bounds = {"min": [min(xs), min(ys), min(zs)],
                  "max": [max(xs), max(ys), max(zs)]}
    calib = session.get("calibration") or {}
    return {
        "schema": TRAJECTORY_SCHEMA,
        "version": VERSION,
        "session": session.get("id", session_dir.name),
        "created": _now(),
        "source_csv": f"derived/{csv_name}",
        "unit": "m",
        "coordinate_system": "rectified_left_camera",
        "image_size": [int(session.get("left", {}).get("width", 0)),
                       int(session.get("left", {}).get("height", 0))],
        "sync": session.get("sync") or {},
        "calibration": {
            "baseline_mm": calib.get("baseline_mm"),
            "created": calib.get("created", ""),
            "verdict": calib.get("verdict", ""),
        },
        "seeds": seeds,
        "bounds": bounds,
        "n_samples": len(samples),
        "n_matched": len(traj),
        "samples": samples,
    }


def build_visualization_payload(trajectory: dict) -> dict:
    """从标准轨迹生成可视化专用数据（不重复逐帧 diagnostics）。"""
    samples = trajectory.get("samples") or []
    points = []
    for s in samples:
        if s.get("matched") and s.get("xyz"):
            points.append({
                "frame": s["frame"], "t": s["t"], "xyz": s["xyz"],
                "confident": bool(s.get("confident")),
            })
    return {
        "schema": VISUALIZATION_SCHEMA,
        "version": VERSION,
        "session": trajectory.get("session"),
        "created": _now(),
        "source": "traj3d.json",
        "unit": trajectory.get("unit", "m"),
        "coordinate_system": trajectory.get("coordinate_system"),
        "bounds": trajectory.get("bounds"),
        "n_points": len(points),
        "trajectory": points,
        "style": {
            "line_color": "#2563eb",
            "point_color": "#f97316",
            "background": "#ffffff",
            "axes": {"x": "X right", "y": "Y forward", "z": "Z up"},
        },
        "samples": [
            {"frame": s["frame"], "t": s["t"], "xyz": s["xyz"],
             "matched": s.get("matched"), "confident": s.get("confident")}
            for s in samples
        ],
    }


def export_session_visualization(
    session_dir: str | Path,
    *,
    csv_name: str = "traj3d.csv",
) -> dict[str, Path]:
    """写 `traj3d.json` + `visualization.json`，返回路径。"""
    session_dir = Path(session_dir)
    trajectory = build_trajectory_payload(session_dir, csv_name=csv_name)
    vis = build_visualization_payload(trajectory)
    derived = session_dir / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    traj_path = derived / "traj3d.json"
    vis_path = derived / "visualization.json"
    traj_path.write_text(json.dumps(trajectory, ensure_ascii=False, indent=2), encoding="utf-8")
    vis_path.write_text(json.dumps(vis, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"trajectory": traj_path, "visualization": vis_path}
