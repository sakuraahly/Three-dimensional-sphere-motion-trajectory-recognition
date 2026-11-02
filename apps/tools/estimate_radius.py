"""从图上估球半径，和 `seeds.json` 里手点/Hough 的半径并排比较。

为什么需要这个工具
------------------
`Z = f·D/(2r)` 里半径是分母：r 错 10%，深度就错 10%，而且是**系统项**，
整条轨迹一起偏、平均不掉。可种子半径一直是"点选时 Hough 给的"，没人复核过。
这个工具把 `geometry.estimate_radius_px()` 的结果和 seeds.json 的原值并排打出来，
**两个数一起看**，不替任何人做决定。

用法::

    python apps/tools/estimate_radius.py data/stereo/20261101-205709
    python apps/tools/estimate_radius.py data/stereo/20261019-200855 --frame 0
    python apps/tools/estimate_radius.py data/stereo/20261101-205709 --overlay outputs/_diag

判读：``score < 0.15`` 表示这一帧的轮廓证据不足（镜面球只有一段弧可量），
此时不要采信估计值，保留原值；``best/r0`` 差 10% 以上才值得人工复核一遍。

只用 `cv2.VideoCapture` 读已录好的视频，**不碰相机**。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

# 按标记上溯（`selftest_entrypoints` 的硬判据）：不写死 `parents[N]`，
# 否则文件一挪层级就静默指错包（本项目已经踩过一次）。
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "sphere_tracker").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sphere_tracker.geometry import estimate_radius_px  # noqa: E402

SIDES = ("left", "right")


def read_seeds(session_dir: Path) -> dict:
    path = session_dir / "derived" / "seeds.json"
    if not path.exists():
        raise SystemExit(f"没有 {path}；先跑一次点选 seed（GUI 的『点选球心』）。")
    return json.loads(path.read_text(encoding="utf-8"))


def read_frame(session_dir: Path, side: str, index: int):
    for ext in (".mkv", ".mp4"):
        path = session_dir / f"{side}{ext}"
        if path.exists():
            break
    else:
        raise SystemExit(f"{session_dir} 下没有 {side}.mkv / {side}.mp4")
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise SystemExit(f"打不开 {path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if not ok:
            raise SystemExit(f"{path} 读不到第 {index} 帧")
        return frame
    finally:
        cap.release()


def main() -> int:
    ap = argparse.ArgumentParser(description="从图上估球半径，与 seeds.json 对比")
    ap.add_argument("session", help="会话目录，例如 data/stereo/20261101-205709")
    ap.add_argument("--frame", type=int, default=None,
                    help="帧号；默认用 seeds.json 里的 seed_frame")
    ap.add_argument("--side", choices=("left", "right", "both"), default="both")
    ap.add_argument("--overlay", default=None,
                    help="把裁剪图（原值圈 + 估计圈）写到这个目录，便于肉眼核对")
    args = ap.parse_args()

    session_dir = Path(args.session)
    seeds = read_seeds(session_dir)
    index = int(seeds.get("seed_frame", 0) if args.frame is None else args.frame)
    sides = SIDES if args.side == "both" else (args.side,)
    overlay = Path(args.overlay) if args.overlay else None
    if overlay is not None:
        overlay.mkdir(parents=True, exist_ok=True)

    print(f"会话 {session_dir}  帧 {index}  （seeds.json 记录 seed_frame="
          f"{seeds.get('seed_frame')}）")
    print(f"{'侧':<6}{'r0':>8}{'best':>8}{'best/r0':>9}{'score':>8}  判读")
    worst = 0
    for side in sides:
        entry = seeds.get(side) or {}
        r0 = float(entry.get("radius", 0.0) or 0.0)
        frame = read_frame(session_dir, side, index)
        r_est, score = estimate_radius_px(frame, float(entry.get("x", 0.0)),
                                          float(entry.get("y", 0.0)), r0)
        ratio = (r_est / r0) if r0 > 1e-6 else float("nan")
        if score < 0.15:
            verdict = "证据不足，保留原值"
        elif abs(ratio - 1.0) >= 0.10:
            verdict = "差 10% 以上，**需要人工复核**"
        else:
            verdict = "与原值一致"
        print(f"{side:<6}{r0:8.1f}{r_est:8.1f}{ratio:9.3f}{score:8.3f}  {verdict}")
        worst = max(worst, abs(ratio - 1.0) if score >= 0.15 else 0.0)

        if overlay is not None:
            cx, cy = float(entry.get("x", 0.0)), float(entry.get("y", 0.0))
            half = int(max(r0, r_est) * 1.35) + 8
            x0, y0 = int(cx) - half, int(cy) - half
            crop = frame[max(0, y0):y0 + 2 * half, max(0, x0):x0 + 2 * half].copy()
            ox, oy = int(cx) - max(0, x0), int(cy) - max(0, y0)
            cv2.circle(crop, (ox, oy), int(round(r0)), (0, 0, 255), 2)
            cv2.circle(crop, (ox, oy), int(round(r_est)), (0, 255, 0), 2)
            cv2.drawMarker(crop, (ox, oy), (255, 255, 0), cv2.MARKER_CROSS, 12, 1)
            out = overlay / f"radius_{session_dir.name}_{side}_f{index}.png"
            cv2.imwrite(str(out), crop)
            print(f"       圈图 -> {out}（红=原值 {r0:.0f}，绿=估计 {r_est:.0f}）")

    print()
    if worst >= 0.10:
        print("结论：至少一路的估计与原值差 ≥10%。**先看证据再看数** —— "
              "score 低的一侧不能据此改半径。")
    else:
        print("结论：估计值与原值一致（或证据不足）。以 seeds.json 为准即可。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
