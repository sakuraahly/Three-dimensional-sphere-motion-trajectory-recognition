"""把一次双目会话导出为可复用的可视化数据。

用法：
    python apps/export_visualization.py data/sessions/<id>
    python apps/export_visualization.py data/sessions/<id> --plot

输出（会话的 derived/ 下）：
    traj3d.json           标准三维轨迹（带 schema/version/坐标/同步/seed）
    visualization.json    给网页/三维查看器用的精简轨迹 + bounds + 样式
    visualization.png     可选，matplotlib 三维图
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
while not (ROOT / "sphere_tracker").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stereo.visualization import export_session_visualization  # noqa: E402


def _plot(session_dir: Path, json_path: Path, out: Path) -> None:
    import json
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
    ax.set_ylabel("Y forward (m)")
    ax.set_zlabel("Z up (m)")
    ax.set_title(f"3D trajectory · {data.get('session', session_dir.name)}")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="导出标准三维轨迹 + 可视化数据")
    ap.add_argument("session", help="data/sessions/<id> 或 session.json")
    ap.add_argument("--csv", default="traj3d.csv", help="derived 下的轨迹 CSV 文件名")
    ap.add_argument("--plot", action="store_true", help="另存 visualization.png")
    args = ap.parse_args()

    session_dir = Path(args.session)
    if session_dir.is_file():
        session_dir = session_dir.parent
    try:
        paths = export_session_visualization(session_dir, csv_name=args.csv)
    except Exception as exc:                                # noqa: BLE001
        print(f"[错误] 导出失败：{exc}")
        return 1
    print(f"轨迹 JSON：{paths['trajectory']}")
    print(f"可视化 JSON：{paths['visualization']}")
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
