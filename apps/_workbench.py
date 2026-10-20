"""GUI 会话工作台的事实层：只读 session.json / derived/，不依赖 Tk。

为什么单独成文件
----------------
`启动程序.py` 已经很大；工作台需要回答"这条会话现在到哪一步、能不能点下一步"。
把读取/判定放在这里，GUI 只负责展示和调用现有脚本；测试也不需要开窗口。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

SESSION_SCHEMA = "sphere-stereo-session"


@dataclass
class SessionSummary:
    path: Path
    id: str = ""
    created: str = ""
    frames_left: int = 0
    frames_right: int = 0
    fps: float = 0.0
    sync_method: str = "none"
    sync_offset: int = 0
    sync_confidence: float = 0.0
    mean_left: float | None = None
    mean_right: float | None = None
    has_seeds: bool = False
    has_traj: bool = False
    has_visualization: bool = False
    has_annotated: bool = False
    n_samples: int = 0
    n_matched: int = 0
    z_min: float | None = None
    z_max: float | None = None
    residual_median: float | None = None
    status: str = "empty"       # empty / bad / recorded / seeded / traj
    problems: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        mark = {"empty": "—", "bad": "✗", "recorded": "○",
                "seeded": "◐", "traj": "✓"}.get(self.status, "?")
        return f"{mark} {self.id or self.path.name}"

    @property
    def can_check(self) -> bool:
        return self.frames_left > 0 and self.frames_right > 0

    @property
    def can_seed(self) -> bool:
        return self.can_check

    @property
    def can_run(self) -> bool:
        # color/hough 不需要 seed；工作台按 specular 的主线判断，
        # 调用方可在非 specular 时覆盖。
        return self.has_seeds

    @property
    def can_view(self) -> bool:
        return self.has_traj and self.n_matched > 0


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _video_mean_from_checks(checks: list[str], side: str) -> float | None:
    """从 session.checks 的“平均亮度：left=... right=...”行取数。"""
    for text in checks:
        m = re.search(r"平均亮度[：:]\s*left=([0-9.]+)\s*right=([0-9.]+)", str(text))
        if m:
            return float(m.group(1) if side == "left" else m.group(2))
    return None


def _read_traj_metrics(path: Path) -> tuple[int, int, float | None, float | None, float | None]:
    """返回 (n_samples, n_matched, z_min, z_max, residual_median)。"""
    data = _read_json(path)
    samples = data.get("samples") or []
    matched = [s for s in samples if s.get("matched") and s.get("xyz")]
    zs = [float(s["xyz"][2]) for s in matched]
    res = [float(s["residual_px"]) for s in matched
           if s.get("residual_px") is not None]
    med = None
    if res:
        res.sort()
        med = res[len(res) // 2]
    return len(samples), len(matched), (min(zs) if zs else None), (max(zs) if zs else None), med


def summarize_session(path: str | Path) -> SessionSummary:
    path = Path(path)
    s = SessionSummary(path=path, id=path.name)
    sess_path = path / "session.json"
    if not sess_path.exists():
        s.status = "bad"
        s.problems.append("缺 session.json")
        return s
    try:
        data = _read_json(sess_path)
    except Exception as exc:                                # noqa: BLE001
        s.status = "bad"
        s.problems.append(f"session.json 读不了：{exc}")
        return s
    s.id = str(data.get("id") or path.name)
    s.created = str(data.get("created") or "")
    left = data.get("left") or {}
    right = data.get("right") or {}
    s.frames_left = int(left.get("frame_count") or 0)
    s.frames_right = int(right.get("frame_count") or 0)
    s.fps = float(left.get("fps_measured") or left.get("fps_requested") or 0.0)
    sync = data.get("sync") or {}
    s.sync_method = str(sync.get("method") or "none")
    s.sync_offset = int(sync.get("offset") or 0)
    s.sync_confidence = float(sync.get("confidence") or 0.0)
    checks = [str(x) for x in (data.get("checks") or [])]
    s.mean_left = _video_mean_from_checks(checks, "left")
    s.mean_right = _video_mean_from_checks(checks, "right")

    derived = path / "derived"
    s.has_seeds = (derived / "seeds.json").exists()
    s.has_visualization = (derived / "visualization.json").exists()
    s.has_annotated = (derived / "annotated.mp4").exists()
    traj = derived / "traj3d.json"
    if traj.exists():
        s.has_traj = True
        try:
            s.n_samples, s.n_matched, s.z_min, s.z_max, s.residual_median = _read_traj_metrics(traj)
        except Exception as exc:                            # noqa: BLE001
            s.has_traj = False
            s.problems.append(f"traj3d.json 读不了：{exc}")

    # 判定：坏数据优先，然后按“录制 -> seed -> 轨迹”推进
    if s.frames_left <= 0 or s.frames_right <= 0:
        s.problems.append("缺视频帧数")
        s.status = "bad"
    elif abs(s.frames_left - s.frames_right) > 1:
        s.problems.append(f"左右帧数不一致 {s.frames_left}/{s.frames_right}")
        s.status = "bad"
    if s.sync_method == "none":
        s.problems.append("未同步（sync=none）")
    if any("几乎全黑" in c for c in checks):
        s.problems.append("视频几乎全黑")
        s.status = "bad"
    if s.status != "bad":
        if not s.has_seeds:
            s.status = "recorded"
            s.problems.append("未点选球心 seed")
        elif not s.has_traj:
            s.status = "seeded"
            s.problems.append("尚未跑离线双目")
        elif s.n_matched == 0:
            s.status = "seeded"
            s.problems.append("三维轨迹没有匹配点")
        else:
            s.status = "traj"
    return s


def button_states(s: SessionSummary, specular: bool) -> dict[str, bool]:
    """工作台按钮的启用规则；放在纯逻辑层，GUI 只套用。"""
    data_ok = s.status != "bad"
    return {
        "check": s.can_check,
        "seed": s.can_seed and data_ok,
        "run": data_ok and s.can_check and (s.has_seeds or not specular),
        "view": s.can_view,
        "export": s.has_traj,
        "open": s.path.exists(),
        "files": s.path.exists(),
    }


def scan_sessions(root: str | Path, limit: int = 12) -> list[SessionSummary]:
    """按修改时间倒序扫描会话目录；坏目录也返回摘要，不静默跳过。"""
    root = Path(root)
    if not root.exists():
        return []
    dirs = [p for p in root.iterdir() if p.is_dir()]
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [summarize_session(p) for p in dirs[:max(1, limit)]]
