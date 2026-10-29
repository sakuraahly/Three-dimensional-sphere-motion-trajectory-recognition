"""单目 vs 双目：同一段录像，两种测深差多少？

为什么需要这个工具
------------------
`AGENTS.md` §1 写着"深度靠视差，这是双目存在的全部意义；单目只能靠球径先验，
精度差一个量级"。这句断言在这之前**没有任何数字**支撑 —— 于是它既不能被推翻，
也不能被引用：说"差一个量级"的人和说"单目也够用"的人，手里都没有证据。

本工具把这句话变成一个可复现的测量：

1. 只对**左路原始视频**跑单目检测（`left.mkv`，不是 `annotated.mp4` ——
   后者已经叠了圈和文字，检测会被自己的标注干扰）；
2. 用单目内参按 ``Z = f·D/(2r)`` 估深度；
3. 与同一会话的双目 ``derived/traj3d.csv`` 逐帧对比（按 ``frame`` 对齐）。

## 球直径 D 未知，是这件事的**核心**，不是脚注

``Z ∝ D``。D 填错 5%，Z 就错 5%（纯线性），而且是**全帧同号**的系统误差 ——
它不会在统计里表现成"噪声"，只会表现成"整体偏了"。

会话里**没有**记录球直径（`session.json` 只记录相机、标定、同步；
`run_stereo_video.py` 里那个 ``ball_diameter_m=0.04`` 只是喂给跑不用的单目分支，
不是实测值）。所以直接填 40 mm 然后得出"单目很准/很不准"，是自欺。

本工具因此先做**反推**：``D_eff = 2·r·Z_stereo / f``，逐帧算，然后看分布。

- ``D_eff`` 稳定（变异系数小、不随深度漂移）⟹ 半径测量是自洽的，
  单目在"D 正确"时能达到括号里那个精度，并且中位数就是**真值直径的反推值**；
- ``D_eff`` 随深度漂移（近处偏大、远处偏小之类）⟹ 单目的半径随距离/模糊/
  过曝系统性变化，**这时问题不在噪声，在模型**，给再准的 D 也救不回来。

两种情况都打印，且都以"中位 D_eff"作为后续对照的 D —— 这样表里的残差是
**纯半径噪声**，尺度项单独在开头交代。

## 这不是"vs 真值"

双目自己也有误差：本会话标定给出的深度 σ 约 3.85 mm @0.6 m
（`calib/stereo_calibration.json` 的 ``quality.depth_sigma_mm_at_0_6m``）。
所以表里所有偏差都是 **mono − stereo**，不是 **mono − truth**。
工具会把这句话连同当帧距离下换算过的 σ 一起打印出来，避免被当成真值。

用法::

    python apps/tools/mono_vs_stereo.py --session data/sessions/20261024-173619 --max-frames 120
    python apps/tools/mono_vs_stereo.py --session <会话> --method specular --diameter 0.0206
                                        # 已知球径时直接给定，跳过反推

退出码：0 = 跑完并出了数；1 = 跑不起来（缺种子/缺视频/检不到球，带原因）。
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
while not (ROOT / "sphere_tracker").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent                      # 按标记上溯（不写死 parents[N]）
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from calib.params import load_stereo_rig  # noqa: E402
from sphere_tracker.config import AppConfig  # noqa: E402
from sphere_tracker.detectors import build_detector  # noqa: E402
from sphere_tracker.geometry import CameraModel, refine_circle_subpixel  # noqa: E402
from stereo.sessions import load_session  # noqa: E402
from stereo.visualization import load_seeds  # noqa: E402

#: 单目检测里"半径单帧跳变"的容忍度。球在 30 fps 下逐帧半径变化 <1%，
#: 超过 8% 只可能是细化射线锁到了背景边缘 —— 宁可退回上一帧，也不要把
#: 一个错误的半径喂进 Z = f·D/(2r)（半径错 1 px，0.5 m 处深度错约 17 mm）。
_MAX_RADIUS_JUMP = 0.08

#: 细化射线在给定半径附近的搜索半宽（像素）。只有 30 帧/秒的连续帧用它，
#: 半径本身逐帧缓慢变化，不需要宽搜索；宽了反而容易抓到背景。
_REFINE_SEARCH = 3.0


# --------------------------------------------------------------------------- #
# 内参
# --------------------------------------------------------------------------- #
@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple[float, ...]
    source: str

    @property
    def f_eff(self) -> float:
        """单目测深用的等效焦距。

        与 `CameraModel.depth_from_radius` 同口径（fx/fy 取均值）：球在像面上
        是个圆，深度由**平均**角分辨率决定，只取 fx 等于假设像素是正方形。
        """
        return 0.5 * (self.fx + self.fy)


def _load_intrinsics(session_dir: Path, mono_path: Path | None,
                     focal_override: float | None) -> Intrinsics:
    """按优先级取单目内参：``--focal`` > 单目标定文件 > 会话里的左路 K1。

    为什么允许退到会话标定的 K1：那是**同一台相机、同一个镜头**在双目标定时
    解出来的左路内参，物理上就是单目内参；缺的只是"专门为单目标一次"这件事。
    降级必须写在脸上（返回值里的 ``source``），因为它决定了下面所有数字的
    **绝对尺度**，读者有权知道这个 fx 是标出来的还是借来的。
    """
    if focal_override and focal_override > 0:
        fx = float(focal_override)
        return Intrinsics(fx, fx, 0.0, 0.0, (), f"--focal 覆盖 fx={fx:.1f}")

    if mono_path is not None and mono_path.exists():
        from calib.mono_calib import load_mono
        calib = load_mono(mono_path)
        if calib is not None:
            return Intrinsics(
                calib.fx, calib.fy, calib.cx, calib.cy,
                tuple(float(x) for x in calib.D),
                f"{mono_path.name}（RMS={calib.rms_px:.2f} px，"
                f"{calib.views} 视角）",
            )

    npz = session_dir / "calib" / "stereo_calibration.npz"
    if not npz.exists():
        raise SystemExit(f"[错误] 既没有单目标定（{mono_path}），"
                         f"会话里也没有 {npz}，无法确定焦距")
    rig = load_stereo_rig(npz)
    d = np.asarray(rig.D1, np.float64).ravel()
    return Intrinsics(
        float(rig.K1[0, 0]), float(rig.K1[1, 1]),
        float(rig.K1[0, 2]), float(rig.K1[1, 2]),
        tuple(float(x) for x in d[:5]),
        "会话标定的左路 K1/D1（借自双目标定，未单独标单目）",
    )


# --------------------------------------------------------------------------- #
# 双目参考
# --------------------------------------------------------------------------- #
@dataclass
class StereoRow:
    frame: int
    x: float
    y: float
    z: float


def _read_traj(path: Path) -> dict[int, StereoRow]:
    """读 ``traj3d.csv``，只留 matched=1 且三轴齐全的行。

    为什么不用 ``fill``/插值补：双目没配对上的帧本来就没有深度，
    补出来的是双目的"猜测"，拿它当参照等于给单目判卷用了错的答案。
    """
    rows: dict[int, StereoRow] = {}
    if not path.exists():
        return rows
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("matched") != "1":
                continue
            try:
                rows[int(r["frame"])] = StereoRow(
                    frame=int(r["frame"]), x=float(r["X"]),
                    y=float(r["Y"]), z=float(r["Z"]))
            except (TypeError, ValueError):
                continue
    return rows


# --------------------------------------------------------------------------- #
# 单目检测
# --------------------------------------------------------------------------- #
@dataclass
class MonoObs:
    frame: int
    u: float
    v: float
    r_px: float
    r_und: float


def _detect_mono(video: Path, cfg: AppConfig, seeds: dict | None,
                 max_frames: int, cam: CameraModel) -> tuple[list[MonoObs], dict]:
    """跑单目检测：检测器给球心（+粗半径），再逐帧做亚像素圆细化。

    **为什么必须再细化一次半径**：``specular`` 检测器的 ``radius`` 来自 CSRT
    跟踪框的 ``max(w,h)/2``，初始化时是固定的 ``init_box/2``，之后只随框缩放
    抖动 —— 它回答的是"框多大"，不是"球多大"。用它算 Z，等于把深度锁死成常数。
    真正的半径只能从**边缘**量，这正是 `refine_circle_subpixel` 干的事
    （沿 72 条射线找梯度峰值再拟合圆）。

    这个环节属于"测量"，不是"重写检测"：检测（找球在哪）仍然全部由库里的
    检测器负责，本工具只补上库里没提供的"半径怎么量"。
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"[错误] 打不开视频：{video}")

    detector = build_detector(cfg.detector)
    r_prev: float | None = None
    seed_frame = 0
    if seeds is not None and hasattr(detector, "seed"):
        s = seeds["left"]
        seed_frame = int(seeds.get("seed_frame", 0) or 0)
        detector.seed(float(s["x"]), float(s["y"]), float(s["radius"]), seed_frame)
        # 种子帧之前检测器返回空，半径从种子开始接力
        r_prev = float(s["radius"])

    obs: list[MonoObs] = []
    stats = {"read": 0, "no_detection": 0, "refine_rejected": 0, "lost_at": None,
             "last_error": ""}
    idx = -1
    while max_frames <= 0 or idx + 1 < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        stats["read"] = idx + 1
        if idx < seed_frame:
            continue
        dets = detector.detect(frame, None)
        if not dets:
            stats["no_detection"] += 1
            stats["lost_at"] = idx
            stats["last_error"] = getattr(detector, "last_error", "")
            break                       # 跟丢是终态（库里不自动重捕），继续跑没意义
        d = dets[0]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        r0 = r_prev if r_prev is not None else float(d.radius)
        cx, cy, r = refine_circle_subpixel(gray, d.x, d.y, r0,
                                          n_rays=72, search=_REFINE_SEARCH)
        # 半径跳变护栏：细化失败时 refine 会原样返回输入，这不算坏；
        # 但"跳了 8% 以上"说明射线抓到了别的东西，退回上一帧的半径更安全。
        if r_prev is not None and abs(r - r_prev) / max(r_prev, 1e-6) > _MAX_RADIUS_JUMP:
            stats["refine_rejected"] += 1
            r = r_prev
        r_prev = r

        obs.append(MonoObs(frame=idx, u=float(cx), v=float(cy), r_px=float(r),
                           r_und=cam.undistort_radius(float(cx), float(cy), float(r))))

    cap.release()
    return obs, stats


# --------------------------------------------------------------------------- #
# 统计
# --------------------------------------------------------------------------- #
def _abs_stats(err: np.ndarray) -> tuple[float, float, float]:
    """(中位, p90, 有符号中位偏差)。单位与输入一致。"""
    a = np.abs(err)
    return float(np.median(a)), float(np.percentile(a, 90.0)), float(np.median(err))


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _mm(x: float) -> str:
    return f"{x * 1000:.1f}"


def _disp_width(s: str) -> int:
    """终端显示宽度：CJK 占两格。表格对齐必须按这个算，不能按 len()。"""
    return sum(2 if ord(ch) > 0x2E7F else 1 for ch in s)


def _pad(s: str, w: int) -> str:
    return s + " " * max(0, w - _disp_width(s))


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="单目（球径先验）vs 双目（视差）：同一条会话的逐帧深度对照")
    ap.add_argument("--session", required=True,
                    help="会话目录，如 data/sessions/20261024-173619")
    ap.add_argument("--max-frames", type=int, default=120,
                    help="只看前 N 帧（0 = 全部；默认 120）")
    ap.add_argument("--method", default="specular",
                    help="单目检测方法（默认 specular —— 金属球用 color 检不到）")
    ap.add_argument("--mono-calib", type=Path,
                    default=ROOT / "config" / "mono_calibration.json",
                    help="单目标定文件（缺省时退回会话标定的左路 K1）")
    ap.add_argument("--focal", type=float, default=None,
                    help="直接指定 fx（像素），优先于标定文件")
    ap.add_argument("--diameter", type=float, default=None,
                    help="已知球直径（米）。不给则用 D_eff 反推")
    ap.add_argument("--bins", type=int, default=3,
                    help="按双目 Z 分几档（默认 3：近/中/远）")
    args = ap.parse_args(argv)

    session_dir = Path(args.session)
    if not session_dir.is_absolute():
        session_dir = (ROOT / session_dir).resolve()
    if not session_dir.is_dir():
        print(f"[错误] 会话目录不存在：{session_dir}")
        return 1

    session_dir, sess = load_session(session_dir)
    if sess.left is None:
        print("[错误] session.json 里没有左路相机信息")
        return 1
    video = session_dir / sess.left.video

    seeds = None
    seed_path = session_dir / "derived" / "seeds.json"
    if args.method == "specular":
        if not seed_path.exists():
            print(f"[错误] specular 需要人工种子 {seed_path}（GUI 点选球心后才有）")
            return 1
        seeds = load_seeds(seed_path)

    intr = _load_intrinsics(session_dir, Path(args.mono_calib), args.focal)
    rig = load_stereo_rig(session_dir / "calib" / "stereo_calibration.npz")

    print("=" * 78)
    print("单目 vs 双目  深度对照")
    print("=" * 78)
    print(f"会话        {session_dir.name}  （{sess.label or '无标签'}）")
    print(f"左路视频    {sess.left.video}  {sess.left.width}x{sess.left.height}"
          f"  {sess.left.frame_count} 帧")
    print(f"检测方法    {args.method}"
          + ("（人工种子 + CSRT 跟踪 + 亚像素半径细化）" if seeds else ""))
    print(f"单目内参    fx={intr.fx:.1f}  fy={intr.fy:.1f}  "
          f"主点=({intr.cx:.1f},{intr.cy:.1f})  "
          f"畸变={'有' if any(abs(x) > 1e-12 for x in intr.distortion) else '无'}")
    print(f"            来源：{intr.source}")
    q = rig.quality or {}
    sigma06 = float(q.get("depth_sigma_mm_at_0_6m") or float("nan"))
    line = (f"双目标定    基线 {rig.baseline_mm:.1f} mm  "
            f"重投影 RMS {float(q.get('rms') or 0):.2f} px")
    if sigma06 == sigma06:
        line += f"  深度 σ≈{sigma06:.2f} mm @0.6 m"
    print(line)
    print()

    stereo_rows = _read_traj(session_dir / "derived" / "traj3d.csv")
    if not stereo_rows:
        print(f"[错误] {session_dir / 'derived' / 'traj3d.csv'} 没有可用行")
        return 1

    cfg = AppConfig()
    cfg.detector.method = args.method
    cfg.predict.enable = False
    cam = CameraModel(
        width=sess.left.width, height=sess.left.height,
        fx=intr.fx, fy=intr.fy, cx=intr.cx, cy=intr.cy,
        distortion=intr.distortion or None,
        ball_diameter_m=1.0,            # 占位：本工具自己算 Z，不用它的球径
    )

    obs, dstats = _detect_mono(video, cfg, seeds, args.max_frames, cam)
    print(f"单目检测    读到 {dstats['read']} 帧，测到 {len(obs)} 帧"
          f"（半径护栏退回 {dstats['refine_rejected']} 次）")
    if dstats["lost_at"] is not None:
        reason = dstats["last_error"] or "检测器返回空结果（这种方法在这段素材上检不到球）"
        print(f"            第 {dstats['lost_at']} 帧停止：{reason}")
    if not obs:
        print("[错误] 一帧都没测到，无法对照（换会话或换 --method）")
        return 1

    # ---- 逐帧配对 ----
    paired: list[tuple[MonoObs, StereoRow]] = [
        (o, stereo_rows[o.frame]) for o in obs if o.frame in stereo_rows
    ]
    if not paired:
        print("[错误] 单目测到的帧与 traj3d.csv 没有交集")
        return 1
    # 分母只算"被处理到的帧区间内"的双目可用帧 —— 用整份 CSV 的 285 行当分母，
    # 会把"只看了前 120 帧"误读成"配对率只有 41%"。
    last_frame = obs[-1].frame
    in_range = [f for f in stereo_rows if f <= last_frame]
    print(f"            与双目配对 {len(paired)} 帧"
          f"（占该区间双目可用帧 {len(paired) / max(len(in_range), 1) * 100:.0f}%，"
          f"区间内双目共 {len(in_range)} 帧）")
    print()

    f_eff = intr.f_eff
    z_s = np.array([p[1].z for p in paired])
    r_und = np.array([p[0].r_und for p in paired])
    # D_eff：让 Z_mono = f·D/(2r) 恰好等于 Z_stereo 的那个直径
    d_eff = 2.0 * r_und * z_s / f_eff

    # 像素速度（px/帧）：后面三步都要用（解释 D_eff 的离散、解释 Z 的偏差），
    # 只在这里算一次 —— 同一概念两处各算一次，早晚会有一处口径不同。
    frames = np.array([p[0].frame for p in paired], dtype=float)
    us = np.array([p[0].u for p in paired])
    vs = np.array([p[0].v for p in paired])
    speed = np.zeros_like(frames)
    for i in range(1, len(frames)):
        gap = frames[i] - frames[i - 1]
        if gap > 0:
            speed[i] = float(np.hypot(us[i] - us[i - 1], vs[i] - vs[i - 1]) / gap)
    if len(speed) > 1:
        speed[0] = speed[1]
    slow_mask = speed <= np.percentile(speed, 33.3)

    # ---- D_eff 分布 ----
    d_med = float(np.median(d_eff))
    d_cv = float(np.std(d_eff) / max(abs(np.mean(d_eff)), 1e-12))
    d_cv_slow = (float(np.std(d_eff[slow_mask]) / max(abs(np.mean(d_eff[slow_mask])), 1e-12))
                 if slow_mask.sum() > 2 else float("nan"))
    # 漂移：D_eff 对 Z 做一次线性回归，看"同一颗球在不同距离上量出来多大"
    slope, intercept = np.polyfit(z_s, d_eff, 1)
    resid = d_eff - (slope * z_s + intercept)
    ss_tot = float(np.sum((d_eff - d_eff.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else 0.0
    drift = slope * (z_s.max() - z_s.min()) / max(abs(d_med), 1e-12)

    print("-" * 78)
    print("第 1 步：反推等效球直径 D_eff = 2·r·Z_stereo / f")
    print("-" * 78)
    print(f"  中位 {d_med * 1000:.2f} mm   均值 {d_eff.mean() * 1000:.2f} mm   "
          f"四分位 [{np.percentile(d_eff, 25) * 1000:.2f}, "
          f"{np.percentile(d_eff, 75) * 1000:.2f}] mm")
    print(f"  标准差 {np.std(d_eff) * 1000:.2f} mm   "
          f"变异系数 CV = {d_cv * 100:.2f}%"
          + (f"（慢速段只有 {d_cv_slow * 100:.2f}%）" if d_cv_slow == d_cv_slow else ""))
    print(f"  随深度漂移：斜率 {slope * 1000:+.2f} mm/m  "
          f"（在 {z_s.min():.3f}~{z_s.max():.3f} m 全程相当于 "
          f"{drift * 100:+.2f}%），R²={r2:.3f}")

    # 判断口径：CV 与漂移一起看。CV 小 = 半径测量自洽；漂移大 = 半径随距离系统变化。
    # 漂移还要求 R² 足够大：只看斜率的话，一段只有 3 cm 深度跨度的短片段
    # 也能"拟合出" 30 mm/m 的斜率，那是把噪声当趋势。
    drift_ok = abs(drift) >= 0.03 and r2 >= 0.5
    if d_cv < 0.03 and not drift_ok:
        verdict = ("稳定 —— 半径测量自洽，可以认为这张表代表单目在"
                   "\"D 正确\"时的真实能力；真值直径约 "
                   f"{d_med * 1000:.1f} mm")
    elif drift_ok:
        verdict = (f"随深度漂移 {drift * 100:+.1f}%（近/远两端的 D_eff 不一致，"
                   f"R²={r2:.2f}）—— 单目在这条数据上不只是噪声问题，是模型问题："
                   "半径随距离/离焦系统性变化，给再准的 D 也压不平")
    else:
        why = (f"漂移只有 {drift * 100:+.1f}%"
               if abs(drift) < 0.03
               else f"斜率看着有 {drift * 100:+.1f}%，但 R²={r2:.2f} 撑不起这个趋势")
        verdict = (f"离散但无深度漂移（CV {d_cv * 100:.1f}%；{why}）—— "
                   "半径测量有波动，但同一颗球在近处和远处量出来一样大，"
                   "所以不是\"半径随距离变\"，见第 3 步找它的来源")
    print(f"  判断：{verdict}")
    print()

    # ---- 用哪个 D 做对照 ----
    if args.diameter:
        d_used = float(args.diameter)
        d_note = f"--diameter 指定 {d_used * 1000:.2f} mm"
    else:
        d_used = d_med
        d_note = f"取中位 D_eff {d_used * 1000:.2f} mm（此时尺度误差按定义为 0）"
    print(f"  对照所用 D：{d_note}")
    print()

    # ---- 单目三维 ----
    z_m, x_m, y_m = [], [], []
    for o, s in paired:
        z = f_eff * (d_used / 2.0) / max(o.r_und, 1e-6)
        p_raw = cam.backproject(o.u, o.v, z)     # 原始左相机系
        # 双目 CSV 的点在**校正后**左相机系（P1/R1 那套），差一个 R1 旋转。
        # 1.4° 的倾角在 0.5 m 处就是约 12 mm 的 X 偏差 —— 不转这一下，
        # 这 12 mm 会被记到单目头上，而它其实是"两个坐标系没对齐"。
        p_rect = rig.R1 @ p_raw if rig.R1 is not None else p_raw
        z_m.append(float(p_rect[2]))
        x_m.append(float(p_rect[0]))
        y_m.append(float(p_rect[1]))

    z_m = np.array(z_m)
    x_m = np.array(x_m)
    y_m = np.array(y_m)
    x_s = np.array([p[1].x for p in paired])
    y_s = np.array([p[1].y for p in paired])

    dz = z_m - z_s
    dx = x_m - x_s
    dy = y_m - y_s
    rel = np.abs(dz) / np.maximum(z_s, 1e-9)

    print("-" * 78)
    print("第 2 步：逐帧偏差（单目 − 双目，单位 mm）")
    print("-" * 78)
    edges = np.percentile(z_s, np.linspace(0, 100, args.bins + 1))
    names = ["近", "中", "远"] if args.bins == 3 else [f"档{i + 1}" for i in range(args.bins)]
    # 近 = Z 小（离相机近），所以按 Z 升序命名与分档
    hdr = (_pad("深度档", 8) + _pad("Z 范围(m)", 16) + _pad("帧", 5)
           + _pad("ΔZ 中位", 11) + _pad("ΔZ p90", 10) + _pad("|ΔZ|/Z", 10)
           + _pad("ΔX 中位", 11) + _pad("ΔX p90", 10)
           + _pad("ΔY 中位", 11) + "ΔY p90")
    print(hdr)
    print("-" * 78)
    for i in range(args.bins):
        lo, hi = edges[i], edges[i + 1]
        m = (z_s >= lo) & (z_s <= hi) if i == args.bins - 1 else (z_s >= lo) & (z_s < hi)
        if not np.any(m):
            continue
        zm, zp, zb = _abs_stats(dz[m])
        xm, xp, _ = _abs_stats(dx[m])
        ym, yp, _ = _abs_stats(dy[m])
        print(_pad(names[i] if i < len(names) else f"档{i + 1}", 8)
              + _pad(f"{lo:.3f}~{hi:.3f}", 16)
              + _pad(str(int(m.sum())), 5)
              + _pad(_mm(zb), 11) + _pad(_mm(zp), 10)
              + _pad(_pct(float(np.median(rel[m]))), 10)
              + _pad(_mm(xm), 11) + _pad(_mm(xp), 10)
              + _pad(_mm(ym), 11) + _mm(yp))
    print("-" * 78)
    zm, zp, zb = _abs_stats(dz)
    xm, xp, _ = _abs_stats(dx)
    ym, yp, _ = _abs_stats(dy)
    print(_pad("全部", 8) + _pad(f"{z_s.min():.3f}~{z_s.max():.3f}", 16)
          + _pad(str(len(dz)), 5)
          + _pad(_mm(zb), 11) + _pad(_mm(zp), 10)
          + _pad(_pct(float(np.median(rel))), 10)
          + _pad(_mm(xm), 11) + _pad(_mm(xp), 10)
          + _pad(_mm(ym), 11) + _mm(yp))
    print("  注：ΔZ 中位是有符号的（能看出整体偏近还是偏远）；"
          "|ΔZ|/Z 是各帧相对误差的中位数")
    # 档位表容易让人以为"中距离单目就是差"，所以要把最差的那批帧**定位**出来：
    # 若它们挤在同一小段时间里，说明有共同原因（比如一段快速运动），
    # 而不是"某个档位天生差"；若散布全场，那就是各档都有的随机水平。
    worst = np.abs(dz) >= np.percentile(np.abs(dz), 90.0)
    if worst.sum() >= 3:
        wf = frames[worst]
        span = float(wf.max() - wf.min() + 1)
        total = float(frames.max() - frames.min() + 1)
        where = (f"集中在第 {int(wf.min())}~{int(wf.max())} 帧"
                 if span < 0.4 * total else "散布在全场，没有集中在某一段")
        print(f"  最差的 10% 帧（{int(worst.sum())} 帧）{where}，"
              f"其中位速度 {np.median(speed[worst]):.1f} px/帧"
              f"（全场中位 {np.median(speed):.1f}）—— 见第 3 步")
    print()

    # ---- X/Y 为什么跟着 Z 一起错 ----
    # X = (u−cx)·Z/fx，Y 同理 —— 像素位置只定**方向**，深度才定**距离**。
    # 所以 Z 的相对误差会原样搬到 X/Y 上再乘以各自的横向偏角。
    lateral = np.hypot(x_s, y_s) / np.maximum(z_s, 1e-9)
    pred_x = np.abs((x_s / np.maximum(z_s, 1e-9)) * dz)
    pred_y = np.abs((y_s / np.maximum(z_s, 1e-9)) * dz)
    print("  X/Y 的误差来源：X=(u−cx)·Z/fx，像素只定方向，Z 定距离 —— "
          "Z 的相对误差会按比例搬进 X/Y")
    print(f"  横向偏角中位 {np.degrees(np.arctan(np.median(lateral))):.0f}°"
          f"（|X,Y|/Z = {np.median(lateral):.3f}）")
    print(f"  只用「ΔZ 乘以横向偏角」预测的 |ΔX| 中位 {_mm(float(np.median(pred_x)))} mm，"
          f"实测 {_mm(xm)} mm（比值 {xm / max(float(np.median(pred_x)), 1e-9):.2f}）")
    print(f"  同一套预测的 |ΔY| 中位 {_mm(float(np.median(pred_y)))} mm，"
          f"实测 {_mm(ym)} mm（比值 {ym / max(float(np.median(pred_y)), 1e-9):.2f}）")
    print("  比值≈1 说明：X/Y 的误差**不是**独立的，它只是 Z 的误差换了个投影方向；")
    print("  想改善 X/Y 就得先改善 Z，单独调横向检测没有意义"
          "（略大于 1 的余量来自球心像素本身的抖动）")
    print()

    # ---- 误差 vs 运动速度 ----
    # 为什么必须看这个：单目的误差理论上应随半径测量噪声走（白噪声、与运动无关）。
    # 如果它和像素速度强相关，那说明真正的原因是**运动模糊把球的边缘抹开了**，
    # 半径被系统性放大 —— 这是模型问题（模糊下的边缘位置不等于球的边界），
    # 不是"再标定一次就能好"的精度问题。两者的修法完全不同，必须先分清。
    print("-" * 78)
    print("第 3 步：误差与运动速度的关系（白噪声还是运动模糊？）")
    print("-" * 78)
    rho = float(np.corrcoef(speed, rel)[0, 1]) if len(speed) > 3 else float("nan")
    q1, q2 = np.percentile(speed, [33.3, 66.7])
    slow = speed <= q1
    fast = speed >= q2
    print(f"  像素速度：中位 {np.median(speed):.1f} px/帧，"
          f"p90 {np.percentile(speed, 90):.1f} px/帧，最大 {speed.max():.1f}")
    print(f"  rel 误差与速度的相关系数 r = {rho:+.2f}")
    print(f"  慢速段（<={q1:.1f} px/帧，{int(slow.sum())} 帧）："
          f"|ΔZ|/Z 中位 {_pct(float(np.median(rel[slow])))}")
    print(f"  快速段（>={q2:.1f} px/帧，{int(fast.sum())} 帧）："
          f"|ΔZ|/Z 中位 {_pct(float(np.median(rel[fast])))}")
    if rho > 0.5 and np.median(rel[fast]) > 1.5 * max(np.median(rel[slow]), 1e-9):
        print("  判断：误差**跟着速度走** —— 是运动模糊把边缘抹开、半径被放大，"
              "不是白噪声")
        print("        这类误差不会因为\"多标一次内参\"消失；要么提高快门/降模糊，"
              "要么在半径模型里显式考虑模糊。")
    else:
        print("  判断：误差与速度无强相关 —— 当作半径测量的随机噪声看待即可")
    print()

    # ---- 尺度项：拿 config 默认 40 mm 当反例 ----
    print("-" * 78)
    print("第 4 步：尺度项（D 填错的代价）")
    print("-" * 78)
    print("  Z ∝ D 是纯线性：D 偏 x% ⟹ Z 全帧同号偏 x%")
    scale_rel = 0.0
    if not args.diameter:
        default_d = AppConfig().camera.ball_diameter_m
        ratio = default_d / d_med
        scale_rel = abs(ratio - 1.0)
        print(f"  若按 AppConfig 默认 D={default_d * 1000:.0f} mm 猜（而这颗球反推是 "
              f"{d_med * 1000:.1f} mm）")
        print(f"    ⟹ 全部帧的 Z 都乘 {ratio:.3f}，"
              f"即 {scale_rel * 100:.0f}% 的系统偏差 —— "
              f"在 {np.median(z_s):.2f} m 处就是 {scale_rel * np.median(z_s) * 1000:.0f} mm")
        print("  这不是检测算法的问题，是\"先验缺失\"的问题：单目的绝对尺度"
              "完全由这个数决定")
    print()

    # ---- 结论 ----
    rel_med = float(np.median(rel))
    z_med = float(np.median(z_s))
    eq_1m = rel_med * 1000.0                     # 1 m 处等效误差（mm）
    stereo_1m = sigma06 * (1.0 / 0.6) ** 2 if sigma06 == sigma06 else float("nan")
    ratio_1m = eq_1m / stereo_1m if stereo_1m == stereo_1m else float("nan")

    print("=" * 78)
    print("结论")
    print("=" * 78)
    print("  情形 A：D 已知且准确（用反推值）—— 只看半径测量本身的能力")
    print(f"    Z 相对误差中位 {rel_med * 100:.2f}%"
          f"  ⟹  1 m 处等效 {eq_1m:.0f} mm"
          f"（本距离 {z_med:.2f} m 处 {rel_med * z_med * 1000:.1f} mm）")
    if ratio_1m == ratio_1m:
        print(f"    双目在 1 m 处 {stereo_1m:.1f} mm  ⟹  单目 / 双目 ≈ {ratio_1m:.1f} 倍")
    print()
    print("  情形 B：D 未知（真实处境，没人把手里的球量过直径）")
    if scale_rel > 0:
        eq_1m_b = scale_rel * 1000.0
        print(f"    绝对尺度全错，等效相对误差 = |D_猜/D_真 − 1| = {scale_rel * 100:.0f}%"
              f"（按 40 mm 默认值猜这颗 {d_med * 1000:.1f} mm 的球）")
        print(f"    ⟹  1 m 处等效 {eq_1m_b:.0f} mm", end="")
        if ratio_1m == ratio_1m:
            print(f"，是双目的 {eq_1m_b / stereo_1m:.0f} 倍")
        else:
            print()
    else:
        print("    本次给了 --diameter，跳过此项")
    print()
    print("  由此对 AGENTS.md §1 那句\"单目只能靠球径先验，精度差一个量级\"的裁决：")
    if scale_rel > 0 and ratio_1m == ratio_1m:
        big = scale_rel * 1000.0 / stereo_1m
        if big >= 10 > ratio_1m:
            print(f"    结论**成立、但归因要改**：差的是**绝对尺度先验**（{big:.0f} 倍），")
            print(f"    不是\"球的像素半径量不准\"（把 D 给定后只差 {ratio_1m:.1f} 倍）。")
            print("    前者靠\"量一下球径/在已知距离处标一次\"就能消掉，后者才是方法的天花板。")
        elif big >= 10 and ratio_1m >= 10:
            print("    成立，且两段都成立：尺度项与噪声项都差一个量级以上")
        else:
            print(f"    仅尺度项成立（{big:.0f} 倍）；噪声项只有 {ratio_1m:.1f} 倍")
    print()
    print("  口径提醒：以上全是 mono − stereo，**不是** mono − 真值。")
    if sigma06 == sigma06:
        print(f"  双目自身在本会话的深度 σ 约 {sigma06:.2f} mm @0.6 m，"
              f"换算到 {z_med:.2f} m 约 {sigma06 * (z_med / 0.6) ** 2:.2f} mm；")
    print("  真值不存在于任何文件里，这条会话从头到尾只有两台相机互相印证 ——")
    print(f"  所以\"单目 {rel_med * z_med * 1000:.1f} mm\"的意思是"
          f"\"和双目差 {rel_med * z_med * 1000:.1f} mm\"，不是\"错这么多\"。")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
