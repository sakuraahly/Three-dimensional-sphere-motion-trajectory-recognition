"""GUI 会话工作台的事实层：只读 session.json / derived/，不依赖 Tk。

为什么单独成文件
----------------
`apps/gui_main.py` 已经很大；工作台需要回答"这条会话现在到哪一步、能不能点下一步"。
把读取/判定放在这里，GUI 只负责展示和调用现有脚本；测试也不需要开窗口。
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

SESSION_SCHEMA = "sphere-stereo-session"

# 所有面板共用的状态符号：工作台、数据管理都从这里取，避免两套图标各说各话
STATUS_SYMBOLS = {
    "empty": "—",
    "bad": "✗",
    "recorded": "○",
    "seeded": "◐",
    "traj_thin": "◑",      # 跑过三维，但匹配率低到画不出可用轨迹
    "traj": "✓",
}

#: 三维匹配率低于这个比例就判「三维点过少」（◑）而不是「有三维」（✓）。
#: 依据是本机 24 段有 traj3d.json 的会话实测，分布是双峰的：
#:   0.0% / 0.1% / 0.7% / 0.7% / 2.7% / 3.0%   ← 跑了但基本没检出
#:   9.4% / 10.2% / 22.2% / 38.1% / ... / 100%  ← 真轨迹
#: 5% 落在两簇之间的空档（3.0% 与 9.4% 之间），两侧都不贴样本。
#: 不是"质量评分"——`◑` 仍然照常可以打开 3D 查看器，只是不谎称"有三维"。
THIN_MATCH_RATIO = 0.05


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
    has_traj_csv: bool = False
    has_3d_products: bool = False
    has_visualization: bool = False
    has_annotated: bool = False
    n_samples: int = 0
    n_matched: int = 0
    z_min: float | None = None
    z_max: float | None = None
    residual_median: float | None = None
    quality_verdict: str = ""   # traj3d.json["quality"]["verdict"]；没有这个键就是 ""
    quality_note: str = ""      # 同上的 note / reason
    stereo_mode: str = "parallel"
    status: str = "empty"       # empty / bad / recorded / seeded / traj_thin / traj
    problems: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        tag = "  [大角度]" if self.stereo_mode == "angled" else ""
        return f"{self.symbol} {self.id or self.path.name}{tag}"

    @property
    def symbol(self) -> str:
        return STATUS_SYMBOLS.get(self.status, "?")

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


def _read_csv_metrics(path: Path) -> tuple[int, int]:
    """从 `traj3d.csv` 数 (总行数, matched 为 1 的行数)。

    为什么需要它：`traj3d.json` 是后来的格式，2026-10 那批老会话只有 CSV。
    只看 JSON 会让"匹配了 276/304 帧"的会话被判成"未点选球心"
    （实测 data/stereo/20261019-193809）。
    文件读不了就返回 (0, 0)，绝不抛。
    """
    total = 0
    matched = 0
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            for row in csv.reader(fh):
                if not row:
                    continue
                if row[0].strip().lower() == "frame":
                    continue            # 表头
                total += 1
                if len(row) > 2 and row[2].strip() == "1":
                    matched += 1
    except Exception:                                   # noqa: BLE001
        return 0, 0
    return total, matched


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
    if not isinstance(data, dict):
        s.status = "bad"
        s.problems.append("session.json 不是对象")
        return s
    s.id = str(data.get("id") or path.name)
    s.created = str(data.get("created") or "")
    mode = data.get("stereo_mode")
    rig_data = data.get("rig")
    if not mode and isinstance(rig_data, dict):
        mode = rig_data.get("stereo_mode")
    s.stereo_mode = "angled" if str(mode or "parallel").strip().lower() in (
        "angled", "auto", "90", "90deg", "orthogonal", "angle") else "parallel"
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
    traj_json = derived / "traj3d.json"
    traj_csv = derived / "traj3d.csv"
    s.has_traj_csv = traj_csv.is_file()
    if traj_json.exists():
        s.has_traj = True
        try:
            s.n_samples, s.n_matched, s.z_min, s.z_max, s.residual_median = _read_traj_metrics(traj_json)
        except Exception as exc:                        # noqa: BLE001
            s.has_traj = False
            s.problems.append(f"traj3d.json 读不了：{exc}")
        else:
            try:
                data_traj = _read_json(traj_json)
                q = data_traj.get("quality") or {}
                if isinstance(q, dict):
                    s.quality_verdict = str(q.get("verdict") or "")
                    s.quality_note = str(q.get("note") or q.get("reason") or "")
            except Exception:                           # noqa: BLE001
                pass
    elif traj_csv.exists():
        # 老会话只有 CSV：照样能数出匹配数，不能因为缺 JSON 就说"没跑过三维"
        s.has_traj = True
        try:
            s.n_samples, s.n_matched = _read_csv_metrics(traj_csv)
        except Exception:                               # noqa: BLE001
            s.has_traj = False
            s.problems.append("traj3d.csv 读不了")
    s.has_3d_products = bool(s.has_traj or s.has_visualization or s.has_annotated)

    # 判定顺序：**证据强度从硬到软**。
    # 硬事实（元数据缺失 / 帧数矛盾）先压过质量评价；
    # 再按"三维产物三种证据（json > csv > 可视化/标注视频）"逐级降级；
    # 最后才看输入文件（seeds）。绝对不能让某个"输入"或"某个特定产物文件"
    # 当唯一闸门 —— 2026-11-18 用户实测：seeds.json 判过一次、traj3d.json 又判过一次，
    # 两次都是"文件在但状态说没有"。
    judged = False
    if s.frames_left <= 0 or s.frames_right <= 0:
        s.problems.append("缺视频帧数")
        if s.has_seeds or s.has_3d_products:
            # 有产物却读不到帧数 = 元数据与产物矛盾，是真问题
            s.status = "bad"
        else:
            # 什么产物都没有 = 录制根本没起来（相机打不开 / 立刻退出），
            # 这种不是"数据坏了"，是"没录成"，重录即可
            s.status = "empty"
            s.problems.append("录制未完成：左右两路都是 0 帧（视频只有文件头）")
        judged = True
    else:
        # 帧数差多少才算"坏"：**按管线能不能跑**定，不按"差 1 帧就不行"定。
        # 管线遇到帧数不一致只是警告「取到短的那路结束」，照常出轨迹；
        # 实测（2026-11-03）：364/362（0.5%）→ 匹配 317/335，完全可用；
        # 450/378（19%）→ 右路掉到 25 fps，那才是真的废。
        # 所以阈值取 max(5 帧, 5%)：小数差只记"提醒"，大比例差才判死。
        diff = abs(s.frames_left - s.frames_right)
        longer = max(s.frames_left, s.frames_right)
        tol = max(5, int(round(0.05 * longer)))
        if diff > tol:
            s.problems.append(
                f"左右帧数差 {diff}（超过 {tol}）：有一路明显掉帧，三维会断续")
            s.status = "bad"
            judged = True
        elif diff > 1:
            s.problems.append(
                f"左右帧数差 {diff} 帧（管线按短的那路跑，可继续）")
    if s.sync_method == "none":
        s.problems.append("未同步（sync=none）")
    if any("几乎全黑" in c for c in checks):
        s.problems.append("视频几乎全黑")
        s.status = "bad"
        judged = True
    if not judged:
        # 顺序**按产物判**，不按输入判：`seeds.json` 是手动点球心的输入，
        # 而 color/motion 方法不需要它也能跑出三维。拿输入当闸门会让
        # "三维已经跑好、只是没点过球心"的会话永远停在 ○（2026-11-18 用户实测：
        # data/stereo/parallel/20261118-152728 有 163 个匹配点却显示 ○）。
        ratio = (s.n_matched / s.n_samples) if s.n_samples > 0 else 0.0
        if s.has_traj and s.n_matched > 0:
            if s.quality_verdict and s.quality_verdict != "ok":
                # 质检结论（run_stereo_video.py 写入）比匹配率更硬：
                # 几何与标定不一致时，匹配率恰好高也不能谎称"有三维"。
                s.status = "traj_thin"
                s.problems.append(
                    f"质检结论 {s.quality_verdict}：{s.quality_note or '结果仅供参考'}")
            elif s.n_samples > 0 and ratio < THIN_MATCH_RATIO:
                s.status = "traj_thin"
                s.problems.append(
                    f"三维匹配点过少：{s.n_matched}/{s.n_samples}"
                    f"（{ratio * 100:.1f}%，低于 {THIN_MATCH_RATIO * 100:g}%），"
                    "轨迹画不出来；检查球心是否选对 / 换检测方法重跑 / 标定是否匹配")
            else:
                s.status = "traj"
        elif s.has_traj:
            # 跑过三维但一个点都没匹配上：产物在、结果空
            s.status = "seeded"
            s.problems.append("三维轨迹没有匹配点")
        elif s.has_3d_products:
            s.status = "seeded"
            s.problems.append(
                "跑过三维，但缺 traj3d.json / traj3d.csv（只剩可视化或标注视频），"
                "匹配率无法判定；建议用同一份 seeds 重跑一次离线三维")
        elif s.has_seeds:
            s.status = "seeded"
            s.problems.append("尚未跑离线双目")
        else:
            s.status = "recorded"
            s.problems.append("未点选球心 seed")
    return s


def button_states(s: SessionSummary, specular: bool,
                  check_ok: bool | None = None) -> dict[str, bool]:
    """工作台按钮的启用规则；放在纯逻辑层，GUI 只套用。"""
    # `empty` 是"没录成"（0 帧、零产物）的空壳会话：视频里根本没有帧可读，
    # 校验/点球心/跑三维全是白点，和坏数据一样要拦住。它单独成一个状态只是为了
    # 告诉用户"重录即可"而不是"去修数据"。
    data_ok = s.status not in ("bad", "empty")
    return {
        "check": s.can_check,
        "seed": s.can_seed and data_ok and check_ok is not False,
        "run": data_ok and s.can_check and check_ok is not False
               and (s.has_seeds or not specular),
        # 2026-11-11：**没有三维结果也能打开查看器** —— 它会降级成"空场景 +
        # 会话信息 + 为什么没有结果"，所以不再要求 has_traj/n_matched>0。
        # 仍按坏数据拦（status=="bad"）：那时会话本身都读不出来。
        "view": data_ok,
        "export": s.has_traj,
        "open": s.path.exists(),
        "files": s.path.exists(),
    }


def scan_sessions(root: str | Path, limit: int = 12) -> list[SessionSummary]:
    """按会话时间倒序扫描；坏会话也返回摘要，不静默跳过。

    「坏」指**有** ``session.json`` 但内容/帧数不对 —— 那种要列出来让人处理。
    连 ``session.json`` 都没有的半成品这里不列（枚举是严判据，见
    ``sphere_tracker/discovery.py``）：它们跑不了工作台的任何一步，显示出来只会
    让人白点；**要看见它们去数据管理面板**（那边是宽判据，`_trash` 除外）。

    排序用目录名里的 `YYYYMMDD-HHMMSS`，**不用目录 mtime**：
    跑一次离线双目会写 derived/，把旧会话的目录 mtime 刷新到今天，
    结果“昨天刚处理过的会话”会跳到第一条，默认选中就漂了。
    名字不是标准时间戳的目录，退回用 mtime。

    枚举走 `apps/_stereo_layout.candidate_session_dirs`（==
    ``sphere_tracker.discovery`` 的递归口径）：会话按装置分区后（``data/stereo/parallel/<时间戳>``）
    只扫一层会把分区目录本身当会话，工作台里就会出现两条"坏会话"
    （名字叫 parallel / angled）。
    """
    from apps._stereo_layout import candidate_session_dirs

    root = Path(root)
    if not root.exists():
        return []
    dirs = candidate_session_dirs([root])

    def sort_key(p: Path) -> tuple[str, float]:
        m = re.match(r"(\d{8}-\d{6})", p.name)
        try:
            mtime = p.stat().st_mtime
        except OSError:
            mtime = 0.0
        return (m.group(1) if m else "", mtime)

    dirs.sort(key=sort_key, reverse=True)
    return [summarize_session(p) for p in dirs[:max(1, limit)]]
