"""把一次双目会话导出为可复用的可视化数据。

用法：
    python apps/export_visualization.py data/sessions/<id>
    python apps/export_visualization.py data/sessions/<id> --plot

输出（会话的 derived/ 下）：
    traj3d.json           标准三维轨迹（带 schema/version/坐标/同步/seed）
    visualization.json    给网页/三维查看器用的精简轨迹 + bounds + 样式
    visualization.png     可选，matplotlib 三维图

重导不丢质量状态：导出是**重建**，`quality` / `predictions` / `smoothing` /
`world_frame` 这些"跑三维时才算得出来"的结论会被重建冲掉。所以本工具在导出
后调 `carry_over_runtime_extras()`：轨迹没变就沿用上一版并写 `carried_over`
标记；轨迹变了（计数 / bounds 变了）就丢弃并在 `notes` 里写明 ——
**绝不把旧结论挂到新轨迹上**。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
while not (ROOT / "sphere_tracker").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stereo.visualization import (  # noqa: E402
    carry_over_runtime_extras, export_session_visualization)

#: 判"还是同一段轨迹"时 bounds 的容差（米）：浮点重算的抖动远小于它。
_BOUNDS_TOL = 1e-6
#: 用来判"换了数据源"的计数键：轨迹 JSON 用前两个，可视化 JSON 用 n_points。
_COUNT_KEYS = ("n_matched", "n_samples", "n_points")


def _read_previous(path: Path) -> dict | None:
    """读上一版 JSON；不存在 / 读坏都返回 None（按"首次导出"处理，绝不抛）。"""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"[警告] 上一版读不出来，按首次导出处理：{path}（{exc}）")
        return None
    return data if isinstance(data, dict) else None


def _bounds_differ(old: object, new: object) -> bool:
    if not (isinstance(old, dict) and isinstance(new, dict)):
        return False
    for edge in ("min", "max"):
        a, b = old.get(edge), new.get(edge)
        if not (isinstance(a, list) and isinstance(b, list)):
            continue
        if len(a) != len(b):
            return True
        for x, y in zip(a, b):
            try:
                if abs(float(x) - float(y)) > _BOUNDS_TOL:
                    return True
            except (TypeError, ValueError):
                return True
    return False


def _trajectory_changed(previous: dict, fresh: dict) -> bool:
    """新旧两份 JSON 是不是**同一段轨迹**。

    判据只有两条（不猜别的）：计数不同（`n_matched` / `n_samples` /
    `n_points`）、`bounds` 的 min/max 任一分量差超过 `_BOUNDS_TOL`。
    缺字段不当作"变了" —— 旧文件可能是更早的版本；真换了 CSV，计数一定对不上。
    """
    for key in _COUNT_KEYS:
        a, b = previous.get(key), fresh.get(key)
        if a is None or b is None:
            continue
        try:
            if int(a) != int(b):
                return True
        except (TypeError, ValueError):
            return True
    return _bounds_differ(previous.get("bounds"), fresh.get("bounds"))


def _merge_previous(path: Path, previous: dict | None) -> tuple[bool, list[str]]:
    """把上一版的运行时结论合进刚导出的 `path`，返回 (轨迹是否变了, 搬了哪些键)。

    为什么必须重写整个文件：`carry_over_runtime_extras()` 只做 dict 运算，
    落盘仍然由这里负责（与 `export_session_visualization()` 同一套 dump 风格）。
    """
    if previous is None:
        return False, []
    fresh = json.loads(path.read_text(encoding="utf-8"))
    changed = _trajectory_changed(previous, fresh)
    before = json.dumps(fresh, ensure_ascii=False, sort_keys=True)
    merged = carry_over_runtime_extras(fresh, previous, trajectory_changed=changed)
    if json.dumps(merged, ensure_ascii=False, sort_keys=True) != before:
        path.write_text(json.dumps(merged, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    return changed, list((merged.get("carried_over") or {}).get("keys") or [])


def _plot(session_dir: Path, json_path: Path, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads(json_path.read_text(encoding="utf-8"))
    pts = data.get("trajectory") or []
    if not pts:
        return
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot([p["xyz"][0] for p in pts], [p["xyz"][1] for p in pts],
            [p["xyz"][2] for p in pts], ".-", color="#2563eb", markersize=2)
    ax.set_xlabel("X right (m)")
    ax.set_ylabel("Y down (m)")
    ax.set_zlabel("Z forward (m)")
    ax.set_title(f"3D trajectory · {data.get('session', session_dir.name)}")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="导出标准三维轨迹 + 可视化数据")
    ap.add_argument("session", help="data/sessions/<id> 或 session.json")
    ap.add_argument("--csv", default="traj3d.csv", help="derived 下的轨迹 CSV 文件名")
    ap.add_argument("--plot", action="store_true", help="另存 visualization.png")
    ap.add_argument("--stereo-mode", choices=("auto", "parallel", "angled"),
                    default="auto",
                    help="auto=读 session/标定；angled=大角度（left_camera）")
    args = ap.parse_args()

    session_dir = Path(args.session)
    if session_dir.is_file():
        session_dir = session_dir.parent
    # ⚠️ 顺序：**先读旧文件，再导出**。反过来就只剩刚重建的那份，没什么可搬。
    derived = session_dir / "derived"
    old_traj = _read_previous(derived / "traj3d.json")
    old_vis = _read_previous(derived / "visualization.json")
    try:
        mode = None if args.stereo_mode == "auto" else args.stereo_mode
        paths = export_session_visualization(
            session_dir, csv_name=args.csv, stereo_mode=mode)
    except Exception as exc:                                # noqa: BLE001
        print(f"[错误] 导出失败：{exc}")
        return 1
    # 导出是**重建**：quality / predictions / smoothing / world_frame 重建不出来。
    # 轨迹没变就沿用上一版的结论，并在 JSON 里留 carried_over 标记 —— 让读的人
    # 知道"这几项不是本次算的"，而不是以为它们刚被重新验证过。
    changed_traj, keys_traj = _merge_previous(paths["trajectory"], old_traj)
    changed_vis, _keys_vis = _merge_previous(paths["visualization"], old_vis)
    print(f"轨迹 JSON：{paths['trajectory']}")
    print(f"可视化 JSON：{paths['visualization']}")
    if changed_traj or changed_vis:
        print("[丢弃] 轨迹已变，上一版 quality/predictions 不再适用")
    elif keys_traj:
        extra = [k for k in keys_traj if k not in ("quality", "predictions")]
        line = "[保留] 沿用上一版 quality/predictions（本次未重算）"
        if extra:
            line += f"；另含 {'/'.join(extra)}"
        print(line)
    if args.plot:
        png = session_dir / "derived" / "visualization.png"
        try:
            _plot(session_dir, paths["visualization"], png)
            print(f"三维图：{png}")
        except Exception as exc:                            # noqa: BLE001
            print(f"[警告] 图生成失败：{exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
