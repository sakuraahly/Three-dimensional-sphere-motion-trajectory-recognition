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

坐标系：parallel 用 `rectified_left_camera`；angled 用 `left_camera`，
单位米；XYZ 轴定义见 `docs/数据标准.md`。

世界系：若 `config/world_frame.json` 存在且合法，`visualization.json` 的
点坐标会被换成**世界系**（`coordinate_system="world"`，并附 `world_frame`
来源信息）；不存在时保持左相机系，并在 `notes` 里明说 ——
**绝不静默把相机系当世界系**。真值应用在 `build_visualization_payload()` 里。
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

#: 世界系配置的默认位置（与 `apps/tools/set_world_frame.py --out` 的默认值一致）。
DEFAULT_WORLD_FRAME_PATH = Path(__file__).resolve().parents[1] / "config" / "world_frame.json"

#: 可以称为"左相机系"的取值：parallel 的是**校正后**左相机系。
LEFT_CAMERA_SYSTEMS = ("left_camera", "rectified_left_camera")


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
    stereo_mode: str | None = None,
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
    rig_data = session.get("rig")
    raw_mode = stereo_mode
    if not raw_mode:
        raw_mode = session.get("stereo_mode") or (
            rig_data.get("stereo_mode") if isinstance(rig_data, dict) else None)
    mode = str(raw_mode or "parallel").strip().lower()
    mode = "angled" if mode in ("angled", "auto", "90", "90deg",
                                "orthogonal", "angle") else "parallel"
    coordinate = "left_camera" if mode == "angled" else "rectified_left_camera"
    return {
        "schema": TRAJECTORY_SCHEMA,
        "version": VERSION,
        "session": session.get("id", session_dir.name),
        "created": _now(),
        "source_csv": f"derived/{csv_name}",
        "unit": "m",
        "stereo_mode": mode,
        "coordinate_system": coordinate,
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


def build_visualization_payload(
    trajectory: dict,
    *,
    world_frame_path: str | Path | None = None,
) -> dict:
    """从标准轨迹生成可视化专用数据（不重复逐帧 diagnostics）。

    `world_frame_path` 省略时查 `config/world_frame.json`；
    存在就换成世界系，不存在就保持左相机系（并在 `notes` 里说明）。
    """
    samples = trajectory.get("samples") or []
    points = []
    for s in samples:
        if s.get("matched") and s.get("xyz"):
            points.append({
                "frame": s["frame"], "t": s["t"], "xyz": s["xyz"],
                "confident": bool(s.get("confident")),
            })
    payload = {
        "schema": VISUALIZATION_SCHEMA,
        "version": VERSION,
        "session": trajectory.get("session"),
        "created": _now(),
        "source": "traj3d.json",
        "unit": trajectory.get("unit", "m"),
        "stereo_mode": trajectory.get("stereo_mode", "parallel"),
        "coordinate_system": trajectory.get("coordinate_system"),
        "bounds": trajectory.get("bounds"),
        "n_points": len(points),
        "trajectory": points,
        "style": {
            "line_color": "#2563eb",
            "point_color": "#f97316",
            "background": "#ffffff",
            "axes": {"x": "X right", "y": "Y down", "z": "Z forward"},
        },
        "samples": [
            {"frame": s["frame"], "t": s["t"], "xyz": s["xyz"],
             "matched": s.get("matched"), "confident": s.get("confident")}
            for s in samples
        ],
    }
    return _apply_world_frame(payload, world_frame_path)


def _apply_world_frame(vis: dict, world_frame_path: str | Path | None) -> dict:
    """把可视化数据换成世界系；没有配置就**明说**坐标仍是相机系。

    三条规矩：
    - 有配置：改点、重算 bounds、`coordinate_system="world"`、写 `world_frame`；
    - 没配置：`coordinate_system` 保持左相机系、`notes` 写明，**不冒充世界系**；
    - 配置坏了：抛 `ValueError` 指出文件与原因，**绝不回退成单位矩阵**。
    """
    path = DEFAULT_WORLD_FRAME_PATH if world_frame_path is None else Path(world_frame_path)
    if not path.exists():
        current = str(vis.get("coordinate_system") or "")
        vis["coordinate_system"] = current if current in LEFT_CAMERA_SYSTEMS else "left_camera"
        # 提示语必须跟着**真实**坐标系走：parallel 的点在校正系，说成"左相机系"会误导。
        label = "校正后的左相机系" if vis["coordinate_system"] == "rectified_left_camera" else "左相机系"
        vis["notes"] = [f"未设置 world_frame.json：当前坐标为{label}"]
        return vis

    # 只在真要换坐标时才 import：纯左相机系导出不该多背 calib 的依赖。
    from calib.world_frame import WorldFrame

    # ⚠️ 校正系（parallel 产物）**不能**直接乘 `T_world_cam`。
    # `T_world_cam` 按**原始左相机系**定义，而 parallel 的点满足 `p_rect = R1 · p_cam`；
    # 直接相乘等于把两个不同的系当成一个，误差是整幅校正旋转（本装置 1.4°，大角度下几度）。
    # 正确做法是先 `p_cam = R1⁻¹ · p_rect` 再变换——但那需要 R1 与基线尺度，
    # 属于另一件事；在那之前**拒绝**，别给一个看起来成功的错数。
    frame = str(vis.get("coordinate_system") or "")
    if frame == "rectified_left_camera":
        raise ValueError(
            "世界系变换暂不支持 parallel 模式：本数据的坐标在校正后的左相机系"
            "（rectified_left_camera），而世界系变换按原始左相机系定义；"
            "两者相差 R1 旋转，直接相乘会得到错误坐标。"
            "请用 angled 标定后重跑，或先关闭世界系（--no-world）。"
        )

    try:
        wf = WorldFrame.load(path)
    except ValueError as exc:
        message = str(exc)
        raise ValueError(message if str(path) in message
                         else f"{path}：{message}") from exc

    for item in vis.get("trajectory") or []:
        item["xyz"] = wf.apply([item["xyz"]])[0].tolist()
    for s in vis.get("samples") or []:
        if s.get("xyz"):
            s["xyz"] = wf.apply([s["xyz"]])[0].tolist()

    xs = [it["xyz"] for it in vis.get("trajectory") or []]
    vis["bounds"] = None if not xs else {
        "min": [min(p[i] for p in xs) for i in range(3)],
        "max": [max(p[i] for p in xs) for i in range(3)],
    }
    vis["unit"] = wf.units
    vis["coordinate_system"] = "world"
    vis["world_frame"] = {
        "source": wf.source,
        "note": wf.note,
        "created_at": wf.created_at,
    }
    vis["notes"] = [f"已应用世界系：{path}"]
    # 轴标签描述的是**数据**，换到世界系后必须跟着换；否则标签会说谎。
    vis["style"]["axes"] = {"x": "World X (board columns)",
                            "y": "World Y (board rows)",
                            "z": "World Z (ground normal)"}
    return vis


def export_session_visualization(
    session_dir: str | Path,
    *,
    csv_name: str = "traj3d.csv",
    stereo_mode: str | None = None,
    world_frame_path: str | Path | None = None,
) -> dict[str, Path]:
    """写 `traj3d.json` + `visualization.json`，返回路径。"""
    session_dir = Path(session_dir)
    trajectory = build_trajectory_payload(
        session_dir, csv_name=csv_name, stereo_mode=stereo_mode)
    vis = build_visualization_payload(trajectory, world_frame_path=world_frame_path)
    derived = session_dir / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    traj_path = derived / "traj3d.json"
    vis_path = derived / "visualization.json"
    traj_path.write_text(json.dumps(trajectory, ensure_ascii=False, indent=2), encoding="utf-8")
    vis_path.write_text(json.dumps(vis, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"trajectory": traj_path, "visualization": vis_path}
