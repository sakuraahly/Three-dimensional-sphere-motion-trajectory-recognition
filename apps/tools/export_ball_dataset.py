"""把"点一下球心 → 跟踪出来的轨迹"导出成 YOLO 单类检测数据集。

为什么要有这个工具
------------------
用户的目标是"以后不用手动点球心"，也就是要一个**学习出来的检测器**。
但训练的瓶颈通常不是模型而是标注：一帧一框手点几百帧，没人愿意干。

本项目恰好有一个别人没有的便宜：**标注是跟踪结果的副产品**。
每次点选球心后跑出来的 `derived/traj.csv`（单目）与 `derived/traj3d.csv`（双目）
里已经有逐帧圆心与半径 —— 那就是标注，只是格式不对。本工具只做"换格式 + 切分"，
不产生任何新的几何推断（除了双目半径那条几何式，见下）。

两类数据源（路径一律走 `sphere_tracker.paths`，不写字面量）
----------------------------------------------------------
- 单目 `data/mono/<id>/`：`clip.mkv` + `derived/seed.json` + `derived/traj.csv`。
  可用帧 = `gate == "ok"`（半径过了锚定门，`x,y` 是平滑后的圆心）；
  半径取 `radius_refined_px`（**边缘细化**那一列），不是 `radius_px`
  （后者是跟踪器平滑后的框半径，口径不同，混用会让框大小在两套尺子之间跳）。
- 双目 `data/stereo/<id>/`：`left.mkv` + `derived/seeds.json` + `derived/traj3d.csv`。
  可用帧 = `matched == 1`（或 `confident == 1`），圆心取 `u_left,v_left`（左图）。
  **没有逐帧半径**，只能二选一：
    `geometry`：`r = f·D/(2Z)`，`f = P1[0,0]`，`D` 用**名义直径 25 mm**（卡尺实测），
                `Z` 取 traj3d 的 `Z`（米）。逐帧变化，比常数准，是默认。
    `seeds`   ：`seeds.json["left"]["radius"]` 常数，帧间不随距离变，只作兜底。

标注框口径（**只进框，不进 Z**）
--------------------------------
`边长 = 2·r·box_inflate`，`box_inflate` 默认 1.15。
为什么不是 1.0：我们的半径来自高光/边缘细化，比**可见轮廓**系统性小约 13.6%
（`docs/交接文档.md` §35.2，`21.6/25 = 0.864`，其倒数 1.157）。YOLO 的框要覆盖
可见轮廓，所以乘回去。这个系数**不参与任何深度计算** —— `Z = f·D/(2r)` 用的是
偏小的有效半径 D=21.6 mm 配偏小的 r，两处口径必须分开，否则又变成"一个数两个来源"。

切分：按会话，绝不按帧；再按类型分层
------------------------------------
同一段视频里相邻帧几乎一样，按帧随机切会让验证集"见过"训练集，mAP 会好看得离谱。
所以 `--val-sessions K` 是**整段会话**进 val；选择是确定性的（按 id 排序取最后 K 段），
报告里写明谁进了 val，换台机器跑结果一样。

在此之上还有一条 `--stratify-kind`（默认开）：**某类型会话数 < 2 时，这类整段进 train**，
否则每类至少 1 段进 val。起因是 2026-10-30 的实测 —— 当时只有 1 段单目，而它是 id 最大的，
于是任何 K 都把它塞进 val：train 里 0 张单目、val 里 30% 是单目，
**等于训了个没见过单目的模型，再拿单目去考它**。分层之后这类"没得验"的情况会在
输出与报告里**明说**（`split_notes`），而不是悄悄给出一个偏高的指标。

负样本（`--negatives`）
-----------------------
只教"哪里有球"是不够的：镜面球在画面边缘、拖影、空场时都会被误检。
- 单目：`traj.csv` 里"最后一个 ok 帧之后"的帧（球出画/糊掉，`gate=no_detection`），
  按 `--neg-every` 采样；
- 双目：`matched != 1`（= 没配上对）的帧。注意 `confident` 这一列在 2026-10-30 之前
  生成的 `traj3d.csv` 里**列错位**、不可信，判据里不能依赖它，详见 `load_stereo` 的注释。
负样本的 label 文件是**空文件** —— 这就是 YOLO 表示"此图无目标"的写法，不是缺文件。
负样本同样按会话归属 train/val，不跨切分，否则等于把 val 的底片漏进 train。

用法::

    python apps/tools/export_ball_dataset.py --all --out outputs/ball_dataset --dry-run
    python apps/tools/export_ball_dataset.py --all --out outputs/ball_dataset --val-sessions 3
    python apps/tools/export_ball_dataset.py --session 20261029-205522 20261024-173619 \\
        --out outputs/ball_dataset --negatives
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "sphere_tracker").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sphere_tracker.paths import MONO_ROOT, STEREO_ROOT, stereo_roots  # noqa: E402

#: 球的**名义**直径（米）。卡尺实测 25 mm，与"有效直径 21.6 mm"是两回事：
#: 后者是给 `Z = f·D/(2r)` 配偏小半径用的仪器常数，标注框要的是可见轮廓。
BALL_DIAMETER_M = 0.025

#: 半径 -> 可见轮廓的补偿系数，见模块开头。1/0.864 = 1.157，取 1.15。
DEFAULT_BOX_INFLATE = 1.15

#: JPEG 质量。92 在这个尺度下肉眼无损，体积约为 PNG 的 1/10。
JPEG_QUALITY = 92

CLASS_NAME = "ball"


class SkipSession(Exception):
    """这段会话不能用来标注 —— 原因写进报告，不静默丢。"""


@dataclass
class Frame:
    """一帧的标注素材（坐标都是**原始分辨率**像素）。"""

    index: int
    x: float = 0.0
    y: float = 0.0
    radius: float = 0.0
    negative: bool = False


@dataclass
class Session:
    kind: str                       # "mono" / "stereo"
    session_id: str
    path: Path
    video: Path
    width: int
    height: int
    positives: list[Frame] = field(default_factory=list)
    negatives: list[Frame] = field(default_factory=list)
    radius_source: str = ""
    note: str = ""

    @property
    def stem(self) -> str:
        return self.session_id


# ----------------------------------------------------------------------
# 小工具
# ----------------------------------------------------------------------
def _num(row: dict, key: str) -> float | None:
    """CSV 里空字符串表示"这一帧没有"，不是 0。"""
    raw = (row.get(key) or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _rows(path: Path) -> list[dict]:
    """读 CSV。**必须 utf-8-sig**：`run_video.py` 写出来的 traj.csv 带 BOM，
    用 utf-8 读第一个列名会变成 `\\ufeffframe`，然后所有 frame 都取不到。"""
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _video_size(video: Path) -> tuple[int, int]:
    """只问容器要分辨率，不解码。打不开就抛 SkipSession（报告要写"视频坏了"）。"""
    import cv2

    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise SkipSession(f"视频打不开：{video.name}")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    if w <= 0 or h <= 0:
        raise SkipSession(f"视频没有可用分辨率：{video.name}")
    return w, h


# ----------------------------------------------------------------------
# 读两类会话
# ----------------------------------------------------------------------
def load_mono(session_dir: Path, box_source: str) -> Session:
    seed_path = session_dir / "derived" / "seed.json"
    traj_path = session_dir / "derived" / "traj.csv"
    video = session_dir / "clip.mkv"
    if not seed_path.is_file():
        raise SkipSession("没种子（derived/seed.json 不存在）")
    if not traj_path.is_file():
        raise SkipSession("没跟踪结果（derived/traj.csv 不存在）")
    if not video.is_file():
        raise SkipSession("视频坏了（clip.mkv 不存在）")

    seed = json.loads(seed_path.read_text(encoding="utf-8-sig"))
    seed_r = float(seed.get("radius") or 0.0)
    width, height = _video_size(video)

    rows = _rows(traj_path)
    positives: list[Frame] = []
    fallback = 0
    use_seed = box_source == "seeds" or (box_source == "geometry" and seed_r <= 0)
    for row in rows:
        if (row.get("gate") or "").strip() != "ok":
            continue
        x, y = _num(row, "x"), _num(row, "y")
        if x is None or y is None:
            continue
        if use_seed:
            radius = seed_r
        else:
            radius = _num(row, "radius_refined_px") or 0.0
            if radius <= 0:                     # 细化列空 -> 退回种子半径，并记数
                radius = seed_r
                fallback += 1
        if radius <= 0:
            continue
        positives.append(Frame(index=int(row["frame"]), x=x, y=y, radius=radius))

    if not positives:
        raise SkipSession("没跟踪结果（traj.csv 里没有 gate==ok 的帧）")

    # 负样本：最后一个有效帧之后。这些帧在 traj 里是 no_detection，
    # 说明"球的锚定断了"（出画 / 糊掉），正好是模型该学会说"没有"的地方。
    negatives: list[Frame] = []
    last_ok = max(f.index for f in positives)
    for row in rows:
        idx = int(row["frame"])
        if idx > last_ok:
            negatives.append(Frame(index=idx, negative=True))

    if use_seed:
        source = "seeds（常数）"
    else:
        source = "csv:radius_refined_px"
        if fallback:
            source += f"（{fallback} 帧细化列为空，退回种子半径）"
    if box_source == "geometry":
        source += "；geometry 不适用于单目（没有 Z），已退回逐帧半径"

    return Session("mono", session_dir.name, session_dir, video, width, height,
                   positives, negatives, source)


def _stereo_focal(session_dir: Path, video_w: int) -> tuple[float | None, str]:
    """左相机焦距（像素），失败返回 None + 原因（调用方退回种子半径）。

    分辨率不一致时必须按比例缩放：`P1[0,0]` 是**标定图像坐标系**下的焦距，
    视频尺寸与标定不同时直接用会让 `r = f·D/(2Z)` 差一个 `video_w / calib_w`。
    """
    from calib.params import load_stereo_rig

    npz = session_dir / "calib" / "stereo_calibration.npz"
    if not npz.is_file():
        return None, "会话没带 calib/stereo_calibration.npz"
    try:
        rig = load_stereo_rig(npz)
    except Exception as exc:                    # noqa: BLE001 —— 标定坏了要降级，不是崩
        return None, f"标定读不了：{type(exc).__name__}: {exc}"
    if rig.P1 is None:
        return None, "标定没有 P1（未校正），无法取焦距"
    focal = float(rig.P1[0, 0])
    calib_w = int(rig.image_size[0]) if rig.image_size else 0
    if calib_w > 0 and calib_w != video_w:
        # 视频被缩放过：焦距按同一比例缩，否则半径整体偏 (video_w/calib_w)
        focal *= video_w / calib_w
    if focal <= 0:
        return None, "P1[0,0] 不是正数"
    return focal, ""


def load_stereo(session_dir: Path, box_source: str) -> Session:
    seeds_path = session_dir / "derived" / "seeds.json"
    traj_path = session_dir / "derived" / "traj3d.csv"
    video = session_dir / "left.mkv"
    if not seeds_path.is_file():
        raise SkipSession("没种子（derived/seeds.json 不存在）")
    if not traj_path.is_file():
        raise SkipSession("没跟踪结果（derived/traj3d.csv 不存在）")
    if not video.is_file():
        raise SkipSession("视频坏了（left.mkv 不存在）")

    seeds = json.loads(seeds_path.read_text(encoding="utf-8-sig"))
    seed_left = seeds.get("left") or {}
    seed_r = float(seed_left.get("radius") or 0.0)
    width, height = _video_size(video)

    focal, focal_note = _stereo_focal(session_dir, width)
    # auto / geometry 都走几何；只有明确要常数、或几何不可用时才用 seeds
    want_geometry = box_source in ("auto", "geometry") and focal is not None
    if box_source == "seeds":
        want_geometry = False

    rows = _rows(traj_path)
    positives: list[Frame] = []
    fallback = 0
    for row in rows:
        matched = (row.get("matched") or "").strip()
        confident = (row.get("confident") or "").strip()
        if matched != "1" and confident != "1":
            continue
        u, v = _num(row, "u_left"), _num(row, "v_left")
        if u is None or v is None:
            continue
        radius = 0.0
        if want_geometry:
            z = _num(row, "Z")
            if z and z > 1e-6:
                radius = focal * BALL_DIAMETER_M / (2.0 * z)
            else:                                # 有像素没深度：这一帧没法算
                radius = seed_r
                fallback += 1
        else:
            radius = seed_r
        if radius <= 0:
            continue
        positives.append(Frame(index=int(row["frame"]), x=u, y=v, radius=radius))

    if not positives:
        raise SkipSession("没跟踪结果（traj3d.csv 里没有 matched/confident 的帧）")

    # 负样本：**没匹配上**的帧（球不在画面里 / 糊到配对失败）。
    # 判据写成"两边都不是 1"，而不是 `matched == 0 and confident == 0`：
    # **2026-10-30 之前生成的 traj3d.csv 有列错位** —— 未匹配行只写了 14 格
    #（漏掉 confident 那一格），表头是 15 列，csv 于是把 method_left 读成
    # confident（实测值是 "track" / "specular"）。那份老文件里 confident 列
    # **不可信**，只有 matched 可信；写 `confident == "0"` 会永远不成立、
    # 负样本静默为 0（这一天实测：14 段全是 0）。
    # 写入侧已在 `apps/run_stereo_video.py` 修好（CSV_HEADER + row_unmatched
    # + 写前等长断言），新文件两列都可信；这里的写法对新旧文件都成立。
    negatives = [Frame(index=int(row["frame"]), negative=True) for row in rows
                 if (row.get("matched") or "").strip() != "1"
                 and (row.get("confident") or "").strip() != "1"]

    if want_geometry:
        source = f"geometry（f={focal:.1f}px, D={BALL_DIAMETER_M * 1000:.0f}mm）"
        if fallback:
            source += f"（{fallback} 帧没有 Z，退回 seeds 常数半径）"
    else:
        source = "seeds（常数）"
        if box_source in ("auto", "geometry") and focal_note:
            source += f"；几何不可用：{focal_note}"

    return Session("stereo", session_dir.name, session_dir, video, width, height,
                   positives, negatives, source)


# ----------------------------------------------------------------------
# 扫目录
# ----------------------------------------------------------------------
def discover(kind: str, root: Path, box_source: str) -> tuple[list[Session], list[dict]]:
    """扫一个根。返回（可用会话, 跳过清单）。**不静默丢**：跳过的都带原因。"""
    usable: list[Session] = []
    skipped: list[dict] = []
    if not root.is_dir():
        return usable, skipped
    loader = load_mono if kind == "mono" else load_stereo
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        if child.name.startswith("_"):
            continue
        try:
            usable.append(loader(child, box_source))
        except SkipSession as exc:
            skipped.append({"id": child.name, "kind": kind, "reason": str(exc)})
        except Exception as exc:                # noqa: BLE001 —— 坏会话不能让整批失败
            skipped.append({"id": child.name, "kind": kind,
                            "reason": f"读取异常：{type(exc).__name__}: {exc}"})
    return usable, skipped


def discover_all(data_root: Path | None, box_source: str) -> tuple[list[Session], list[dict]]:
    """单目 + 双目。`data_root` 只给测试用（生产走 paths.py 的常量）。"""
    mono_root = (data_root / "mono") if data_root else MONO_ROOT
    stereo_root_list = ((data_root / "stereo"),) if data_root else stereo_roots()

    sessions: list[Session] = []
    skipped: list[dict] = []
    seen: set[str] = set()
    for root in stereo_root_list:              # 新的根在前，同名只认新的
        batch, bad = discover("stereo", root, box_source)
        skipped.extend(bad)
        for s in batch:
            if s.session_id in seen:
                continue
            seen.add(s.session_id)
            sessions.append(s)
    batch, bad = discover("mono", mono_root, box_source)
    skipped.extend(bad)
    for s in batch:
        if s.session_id in seen:                # 单双目同 id：文件名会撞，直接跳过并说清
            skipped.append({"id": s.session_id, "kind": "mono",
                            "reason": "与已有双目会话同名，图片文件名会撞"})
            continue
        seen.add(s.session_id)
        sessions.append(s)
    sessions.sort(key=lambda s: s.session_id)
    return sessions, skipped


# ----------------------------------------------------------------------
# 切分
# ----------------------------------------------------------------------
def _kind_groups(sessions: list[Session]) -> dict[str, list[Session]]:
    """按会话类型分组，组内按 id 升序。"""
    groups: dict[str, list[Session]] = {}
    for s in sorted(sessions, key=lambda s: s.session_id):
        groups.setdefault(s.kind, []).append(s)
    return groups


def split_bounds(sessions: list[Session], stratify: bool = True) -> tuple[int, int, list[str]]:
    """`--val-sessions K` 的合法区间 [lo, hi]，以及分层时要说给人听的话。

    为什么区间不再是"1 .. N-1"：分层之后，**每一类都必须至少留 1 段给训练、
    至少留 1 段给验证**，否则就是"拿没训练过的条件去考试"。只有 1 段的类型
    干脆整段进 train（无法既训又验），它也就不能占用 val 名额。
    """
    groups = _kind_groups(sessions)
    if not stratify:
        n = len(sessions)
        return 1, max(n - 1, 0), []
    eligible = {k: v for k, v in groups.items() if len(v) >= 2}
    notes = [f"{k} 只有 {len(v)} 段 → 全部进 train（无法既训又验）"
             for k, v in sorted(groups.items()) if len(v) < 2]
    lo = len(eligible)
    hi = sum(len(v) - 1 for v in eligible.values())
    return lo, hi, notes


def assign_splits(sessions: list[Session], val_sessions: int, stratify: bool = True
                  ) -> tuple[dict[str, str], list[str]]:
    """按**会话**切分，返回（session_id -> split, 说明）。

    两条纪律合在一起：

    1. **整段会话一个答案**。同一段里相邻帧差 33 ms，按帧随机切 = 让 val 见过
       train 的邻帧，mAP 会好看得离谱，而这件事不会报错。确定性也是刻意的
       （按 id 排序取尾部），换机器、换 run 顺序结果一样。
    2. **按类型分层**（`--stratify-kind`，默认开）。2026-10-30 实测的翻车现场：
       只有 1 段单目，而它是 id 最大的，于是任何 K 都把它放进 val ——
       train 里 0 张单目、val 里 30% 是单目，**训了个没在单目上训练过的模型，
       然后在单目上考它**。现在的规则是：某类型 < 2 段就整段进 train；
       >= 2 段则每类至少 1 段进 val（取该类型 id 最大的几段），
       剩下的名额按类型名轮流分，保持确定性。
    """
    ordered = sorted(sessions, key=lambda s: s.session_id)
    if not stratify:
        val_ids = {s.session_id for s in ordered[len(ordered) - val_sessions:]}
        splits = {s.session_id: ("val" if s.session_id in val_ids else "train")
                  for s in ordered}
        return splits, [f"按 id 排序取最后 {val_sessions} 段进 val（--no-stratify-kind）"]

    groups = _kind_groups(ordered)
    eligible = {k: v for k, v in groups.items() if len(v) >= 2}
    notes = [f"{k} 只有 {len(v)} 段 → 全部进 train；val 只含 "
             f"{'、'.join(sorted(eligible)) or '（空）'}"
             f"（{k} 的表现**未被验证**）"
             for k, v in sorted(groups.items()) if len(v) < 2]

    quota = {k: 1 for k in eligible}
    left = val_sessions - len(eligible)
    names = sorted(eligible)
    while left > 0:
        # 轮流加名额：每轮按类型名顺序各给 1 段，直到满额或没容量
        progressed = False
        for k in names:
            if left <= 0:
                break
            if quota[k] < len(eligible[k]) - 1:
                quota[k] += 1
                left -= 1
                progressed = True
        if not progressed:                  # 容量用尽（调用方应当先截住，这里是兜底）
            break

    val_ids: set[str] = set()
    for k, n_val in quota.items():
        val_ids |= {s.session_id for s in eligible[k][len(eligible[k]) - n_val:]}
    splits = {s.session_id: ("val" if s.session_id in val_ids else "train")
              for s in ordered}
    per_kind = "、".join(f"{k} {quota[k]} 段" for k in names)
    notes.append(f"val 按类型：{per_kind}；共 {sum(quota.values())} 段")
    return splits, notes



# ----------------------------------------------------------------------
# 写盘
# ----------------------------------------------------------------------
def _label_line(frame: Frame, width: int, height: int, inflate: float) -> str:
    """一行 YOLO 标签：`0 cx cy w h`，全部按各自轴的尺寸归一化。

    为什么宽度除以 W、高度除以 H（而不是都除以 max(W,H)）：YOLO 的归一化是
    **逐轴**的，`cx = x/W`、`w = 2r/W`，还原时再各乘回去。球是圆的，
    所以**像素空间**里必须 w_px == h_px == 2·r·inflate；两侧都用 max(W,H)
    除会让非正方形画面（本项目 1718×972）的框在像素里变成椭圆，物理上说不通。
    """
    r = frame.radius * inflate
    cx = min(max(frame.x / width, 0.0), 1.0)
    cy = min(max(frame.y / height, 0.0), 1.0)
    w = min(max(2.0 * r / width, 0.0), 1.0)
    h = min(max(2.0 * r / height, 0.0), 1.0)
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def _iter_frames(video: Path, indices: list[int]):
    """按帧号升序取帧：跳过用 `grab()`（不解码），目标帧才 `retrieve()`。

    逐帧 `read()` 对 300 帧的视频也没问题，但一段会话里要的帧往往带空洞
    （gate 不过的帧不要），`grab` 能把这些帧的解码成本省掉。
    """
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SkipSession(f"视频打不开：{video.name}")
    pos = -1
    try:
        for idx in indices:
            while pos < idx:
                if not cap.grab():
                    return
                pos += 1
            ok, frame = cap.retrieve()
            if not ok:
                return
            yield idx, frame
    finally:
        cap.release()


#: 本工具写出的文件名形状：`<会话 id>_<6 位帧号>`。收拾残留时**只碰**这个形状的东西，
#: 别的文件（用户的笔记、别人的数据集）一个字节都不动 —— 这比"删多少条以内"更靠谱。
_STEM_RE = re.compile(r"^\d{8}-\d{6}_\d{6}$")


def _wanted_frames(session: Session, args) -> list[Frame]:
    """这段会话本次要写哪些帧（正样本 + 采样后的负样本），按帧号升序。"""
    want = list(session.positives)
    if args.negatives:
        # 负样本按 --neg-every 采样：相邻帧几乎一样，全存只会重复同一个负例
        want += session.negatives[::max(1, args.neg_every)]
    want.sort(key=lambda f: f.index)
    return want


def _planned_names(sessions: list[Session], splits: dict[str, str], args
                   ) -> dict[str, set[str]]:
    """这次会写出哪些文件名（**写盘前**就能算出来，用来收拾上次的残留）。

    为什么要"清扫残留"而不是覆盖了事：换 `--val-sessions` 后，上一次进 val 的帧
    这次可能进 train，但旧文件还躺在 `images/val` 里 —— 同一帧同时出现在两边，
    正是按会话切分要防的泄漏，而且是**静默**的（训练照跑，只是 mAP 虚高）。
    """
    planned = {"train": set(), "val": set()}
    for s in sessions:
        for f in _wanted_frames(s, args):
            planned[frame_split(s, f, splits)].add(f"{s.session_id}_{f.index:06d}")
    return planned


def _reconcile_out(out: Path, planned: dict[str, set[str]]) -> tuple[int, int]:
    """把上次的产物对齐到这次的计划：**换 split 的直接搬，多出来的删**。

    2026-10-30 的实测教训：一开始图省事直接 `rmtree(images/labels)` 重写，两个问题
    一起冒出来 —— ① 2431 个文件的批量删除会撞上环境的 safe-delete 守卫（整条命令
    exit 1，一个文件都没写）；② 换 `--val-sessions` 时其实**没有新数据**，
    464 张图只是从 `val/` 挪到 `train/`，删掉再写一遍纯属浪费。

    所以这里按"文件名 = 会话 id + 帧号"这个事实做搬运：目标 split 变了就 `os.replace`
    过去（同盘改名，不动数据）；这次压根不导的才删。返回（搬走, 删掉）。
    """
    where = {stem: split for split, stems in planned.items() for stem in stems}
    moved = removed = 0
    for sub, suffix in (("images", ".jpg"), ("labels", ".txt")):
        for split in ("train", "val"):
            folder = out / sub / split
            if not folder.is_dir():
                continue
            for path in list(folder.iterdir()):
                # 不是本工具的命名 -> 不认识，放手（用户的东西不该被"顺手清理"）
                if path.suffix != suffix or not _STEM_RE.match(path.stem):
                    continue
                target = where.get(path.stem)
                if target == split:
                    continue                    # 位置对，稍后会被覆盖写
                if target:
                    other = out / sub / target
                    other.mkdir(parents=True, exist_ok=True)
                    os.replace(path, other / path.name)
                    moved += 1
                else:
                    path.unlink()
                    removed += 1
    if moved or removed:
        print(f"  对齐上次产物：换 split 搬走 {moved} 个，清掉 {removed} 个（本次不导）")
    return moved, removed


def frame_split(session: Session, frame: Frame, splits: dict[str, str]) -> str:
    """**这一帧**进哪个 split。

    整段会话一个答案 —— 这是本工具最要紧的一条纪律，所以刻意做成**唯一出口**：
    想要"按帧随机"就得改这一处，而测试里有一条反向自证会把这种改法抓出来。
    逐帧随机看似样本更多，实际是让 val 见过 train 的邻帧（相隔 33 ms 的画面几乎相同）。
    """
    return splits[session.session_id]



def build(sessions: list[Session], splits: dict[str, str], out: Path, args) -> dict:
    """真正导出；返回写进 export_report.json 的统计（dry-run 时同一份结构）。"""
    import cv2

    images = {"train": 0, "val": 0}
    boxes = {"train": 0, "val": 0}
    neg_images = {"train": 0, "val": 0}
    box_sides: list[float] = []
    unreadable = 0
    per_session: list[dict] = []

    scale_all: dict[str, float] = {}
    for s in sessions:
        scale_all[s.session_id] = min(1.0, float(args.max_width) / s.width)

    if not args.dry_run:
        for split in ("train", "val"):
            (out / "images" / split).mkdir(parents=True, exist_ok=True)
            (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        # 先把上次的产物对齐到这次的计划（换 split 的搬过去、多出来的删掉），再开始写
        _reconcile_out(out, _planned_names(sessions, splits, args))

    for s in sessions:
        session_split = splits[s.session_id]
        want = _wanted_frames(s, args)

        by_index = {f.index: f for f in want}
        wrote = 0
        wrote_neg = 0
        if args.dry_run:
            for f in want:
                split = frame_split(s, f, splits)
                if f.negative:
                    wrote_neg += 1
                    neg_images[split] += 1
                else:
                    wrote += 1
                    boxes[split] += 1
                    box_sides.append(2.0 * f.radius * args.box_inflate)
                images[split] += 1
        else:
            for idx, frame in _iter_frames(s.video, list(by_index)):
                meta = by_index[idx]
                split = frame_split(s, meta, splits)
                h_px, w_px = frame.shape[:2]
                if w_px != s.width or h_px != s.height:
                    # 容器信息与真实帧不一致时以真实帧为准（归一化必须跟着真实尺寸走）
                    s.width, s.height = w_px, h_px
                scale = scale_all[s.session_id] = min(1.0, float(args.max_width) / w_px)
                if scale < 1.0:
                    frame = cv2.resize(frame, (int(round(w_px * scale)),
                                               int(round(h_px * scale))),
                                       interpolation=cv2.INTER_AREA)
                name = f"{s.session_id}_{idx:06d}.jpg"
                img_path = out / "images" / split / name
                lbl_path = out / "labels" / split / f"{s.session_id}_{idx:06d}.txt"
                if not cv2.imwrite(str(img_path), frame,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]):
                    unreadable += 1
                    continue
                # 缩小只改像素，不改归一化标签 —— 所以标签在缩放前后是同一串数字
                text = "" if meta.negative else _label_line(meta, w_px, h_px, args.box_inflate)
                lbl_path.write_text(text + ("\n" if text else ""), encoding="utf-8")
                images[split] += 1
                if meta.negative:
                    wrote_neg += 1
                    neg_images[split] += 1
                else:
                    wrote += 1
                    boxes[split] += 1
                    box_sides.append(2.0 * meta.radius * args.box_inflate)
            expected = len(want)
            got = wrote + wrote_neg
            if got < expected:
                unreadable += expected - got

        per_session.append({
            "id": s.session_id,
            "kind": s.kind,
            "split": session_split,
            "frames": wrote,
            "negatives": wrote_neg,
            "radius_source": s.radius_source,
            "resolution": f"{s.width}x{s.height}",
            "scale_to_max_width": round(scale_all[s.session_id], 4),
            "video": s.video.relative_to(ROOT).as_posix()
            if s.video.is_relative_to(ROOT) else str(s.video),
        })
        print(f"  [{session_split:5s}] {s.session_id} ({s.kind}) 帧 {wrote} "
              f"负 {wrote_neg} 半径源 {s.radius_source}")


    stats = {
        "images": images,
        "boxes": boxes,
        "negatives": neg_images,
    }
    if box_sides:
        stats["box_side_px"] = {
            "median": round(statistics.median(box_sides), 1),
            "min": round(min(box_sides), 1),
            "max": round(max(box_sides), 1),
            "n": len(box_sides),
        }
    else:
        stats["box_side_px"] = {"median": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    stats["unreadable"] = unreadable
    stats["sessions"] = per_session
    return stats


def write_dataset_yaml(out: Path) -> Path:
    path = out / "dataset.yaml"
    path.write_text(
        "# 由 apps/tools/export_ball_dataset.py 生成，可重新生成，别手改\n"
        f"path: {out.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        f"  0: {CLASS_NAME}\n",
        encoding="utf-8",
    )
    return path


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="导出 YOLO 单类球检测数据集（标注来自跟踪结果）")
    ap.add_argument("--out", default="outputs/ball_dataset", help="输出目录")
    ap.add_argument("--session", nargs="+", default=None,
                    help="只导出这几段会话（按 id）。给了它就不用 --all")
    ap.add_argument("--all", action="store_true", help="导出全部可用会话")
    ap.add_argument("--val-sessions", type=int, default=2,
                    help="整段进 val 的会话数（默认 2；分层后区间由 split_bounds 决定）")
    ap.add_argument("--stratify-kind", dest="stratify_kind", action="store_true",
                    default=True,
                    help="[默认开] 按会话类型分层：某类 <2 段则整段进 train，"
                         "否则每类至少 1 段进 val。防止'单目只有 1 段 -> 全进 val'")
    ap.add_argument("--no-stratify-kind", dest="stratify_kind", action="store_false",
                    help="关掉分层，退回'按 id 排序取最后 K 段'（val 可能与 train 不同类型）")
    ap.add_argument("--max-width", type=int, default=1280,
                    help="存图前等比缩到不超过这个宽度（1920 存 1280 省一半磁盘；"
                         "YOLO 反正会 resize 到 640，标签是归一化的所以不受影响）")
    ap.add_argument("--box-inflate", type=float, default=DEFAULT_BOX_INFLATE,
                    help=f"半径 -> 可见轮廓的补偿系数（默认 {DEFAULT_BOX_INFLATE}，"
                         f"依据 docs/交接文档.md §35.2 的 13.6%% 偏差）")
    ap.add_argument("--box-source", choices=("auto", "seeds", "geometry"), default="auto",
                    help="半径来源：auto=双目几何/单目逐帧；seeds=种子常数；geometry=只用几何")
    ap.add_argument("--negatives", action="store_true",
                    help="额外导出负样本（空 label 文件）")
    ap.add_argument("--neg-every", type=int, default=5,
                    help="负样本采样间隔（相邻帧几乎一样，全存只是重复）")
    ap.add_argument("--data-root", default=None,
                    help="数据根（含 mono/ 与 stereo/）；只给测试用，默认走 sphere_tracker.paths")
    ap.add_argument("--dry-run", action="store_true", help="只统计，不写文件")
    return ap.parse_args(argv)


def _describe(sessions: list[Session], skipped: list[dict]) -> None:
    print(f"发现 {len(sessions)} 段可用会话：")
    for s in sessions:
        print(f"  {s.session_id}  {s.kind:6s} 有效 {len(s.positives):4d} 帧  "
              f"负样本候选 {len(s.negatives):4d}  {s.width}x{s.height}  "
              f"半径 {s.radius_source}")
    if skipped:
        print(f"跳过 {len(skipped)} 段：")
        for item in skipped:
            print(f"  {item['id']} ({item['kind']})：{item['reason']}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # 相对路径按当前工作目录算（用户敲命令时看到的就是它），不做"猜仓库根"的魔法
    out = Path(args.out).expanduser().resolve()

    sessions, skipped = discover_all(
        Path(args.data_root) if args.data_root else None, args.box_source)

    if not args.session and not args.all:
        _describe(sessions, skipped)
        lo, hi, _notes = split_bounds(sessions, args.stratify_kind)
        print()
        if hi:
            print(f"共 {len(sessions)} 段可导出；--val-sessions 合法区间 "
                  f"{lo}..{hi}（默认 {args.val_sessions}；按类型分层）")
        print("用法：加 --all 导出全部，或 --session <id> [<id> ...] 指定几段。")
        print("例：python apps/tools/export_ball_dataset.py --all "
              f"--out {args.out} --dry-run")
        return 2

    if args.session:
        wanted = set(args.session)
        unknown = sorted(wanted - {s.session_id for s in sessions})
        if unknown:
            known = "、".join(s.session_id for s in sessions) or "（没有）"
            print(f"没有这些会话：{'、'.join(unknown)}")
            print(f"可用的有：{known}")
            return 2
        sessions = [s for s in sessions if s.session_id in wanted]

    if len(sessions) < 2:
        only = sessions[0].session_id if sessions else "（无）"
        print(f"只有 {len(sessions)} 段可用会话（{only}），拒绝导出："
              f"1 段不能既训又验。")
        print("需要至少 2 段（一段 train，一段 val）；按会话切分是硬约束，"
              "按帧切会让验证集见过训练集。")
        if skipped:
            print("跳过的会话与原因：")
            for item in skipped:
                print(f"  {item['id']} ({item['kind']})：{item['reason']}")
        return 2

    lo, hi, split_notes = split_bounds(sessions, args.stratify_kind)
    if hi == 0:
        kinds = "、".join(f"{k} {len(v)} 段" for k, v in _kind_groups(sessions).items())
        print(f"拒绝导出：没有任何**类型**凑够 2 段（{kinds}），"
              f"每一类都只有 1 段就不能既训又验。")
        print("要么再录一段同类会话，要么用 --no-stratify-kind 强行按 id 切"
              "（那样 val 里会出现训练时没见过的类型，指标不可信）。")
        return 2
    if not (lo <= args.val_sessions <= hi):
        print(f"--val-sessions {args.val_sessions} 不合法：共 {len(sessions)} 段，"
              f"按类型分层后必须 {lo} <= K <= {hi}。")
        for note in split_notes:
            print(f"  [切分] {note}")
        if args.val_sessions < lo:
            print("  K 太小：分层要求每一类（>=2 段的）至少 1 段进 val。")
        else:
            print("  K 太大：每一类都要留至少 1 段给 train。")
        return 2

    splits, notes = assign_splits(sessions, args.val_sessions, args.stratify_kind)
    for note in notes:
        print(f"[切分] {note}")
    val_ids = [sid for sid, sp in splits.items() if sp == "val"]
    kind_counts: dict[str, dict[str, int]] = {}
    for s in sessions:
        bucket = kind_counts.setdefault(s.kind, {"train": 0, "val": 0})
        bucket[splits[s.session_id]] += 1
    print(f"切分：{len(sessions) - len(val_ids)} 段 train / {len(val_ids)} 段 val"
          f"（val = {'、'.join(val_ids)}）")
    print("  每类：" + "；".join(f"{k} train {v['train']} / val {v['val']}"
                                for k, v in sorted(kind_counts.items())))
    if len(sessions) - len(val_ids) < 1:
        # 理论到不了（上面的区间校验守着），但真到了就是"没训练集"，不能默默跑
        print("train 为空，拒绝导出。")
        return 2

    stats = build(sessions, splits, out, args)

    report = {
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "out": out.as_posix(),
        "dry_run": bool(args.dry_run),
        "sessions": stats["sessions"],
        "images": stats["images"],
        "boxes": stats["boxes"],
        "negatives": stats["negatives"],
        "box_inflate": args.box_inflate,
        "max_width": args.max_width,
        "box_source": args.box_source,
        "box_side_px": stats["box_side_px"],
        "val_sessions": [sid for sid in val_ids],
        "stratify_kind": bool(args.stratify_kind),
        "kind_counts": kind_counts,
        "split_notes": notes,
        "unreadable_frames": stats["unreadable"],
        "skipped": skipped,
        "notes": [
            "box_inflate=1.15 只进标注框，不进 Z；Z 用的是有效直径 21.6 mm",
            "切分按会话，同段会话不会同时出现在 train 与 val",
            "负样本的 label 是空文件（YOLO 的'此图无目标'写法），不是缺失",
            "某类型会话数 <2 时整段进 train：那段类型在 val 上没有体现，"
            "不要把 val 指标当成覆盖了该类型",
        ],
    }

    side = report["box_side_px"]
    print(f"\n图片：train {stats['images']['train']} / val {stats['images']['val']}"
          f"（其中负样本 {stats['negatives']['train']}/{stats['negatives']['val']}）")
    print(f"框：train {stats['boxes']['train']} / val {stats['boxes']['val']}")
    print(f"框边长（原始分辨率像素）：中位 {side['median']}，"
          f"范围 {side['min']}~{side['max']}，n={side['n']}")
    if stats["unreadable"]:
        print(f"⚠ 有 {stats['unreadable']} 帧读不出来（视频坏帧），已跳过")

    if args.dry_run:
        print("\n--dry-run：没有写任何文件。")
        return 0

    out.mkdir(parents=True, exist_ok=True)
    (out / "export_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    yaml_path = write_dataset_yaml(out)
    print(f"\n写出：{out.as_posix()}")
    print(f"  {yaml_path.name} / export_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
