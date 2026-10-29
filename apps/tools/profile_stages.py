"""离线双目流程的**分段计时**：回答"时间到底花在哪一段"。

## 为什么需要这个工具

`record_stereo` 在 1440p@30 请求下主循环只有约 24 fps（242/300 帧），
`run_realtime` 也有类似上限。但在此之前，**没有人量过**这些时间落在哪一段：
读帧？检测？跟踪？配对？三角测量？叠加渲染？
没有分段数据，任何"优化"都只能靠猜 —— 而本项目已经因为"把 detect_ms
当成采集帧率"走过一次完整的错方向排查（`sphere_tracker/stats.py` 的模块注释）。

## 它凭什么可信：只包计时，不碰计算

本工具**不重写**离线流程，而是把 `apps/run_stereo_video.py` 当模块加载，
复用它自己的 `_make_pipeline` / `_mode_guard` / `_as_detection` / `_draw`，
所以参数语义、模式解析、坐标换算与正式入口是同一份实现。

计时全部在**对象边界**上做（代理 / 临时替换模块函数），被测函数的入参与
返回值原样透传，**不改变任何计算结果**：

- 检测：`SpherePipeline` 本来就支持注入 detector，塞一个计时代理即可；
- 跟踪：同一注入点，塞 tracker 代理；
- 双目配对 / 三角测量：`StereoReconstructor._match_frame` 调的是模块级
  `match_epipolar` / `triangulate_one`，在跑之前临时替换成透传包装、跑完还原。
  **只有这一条路能在不改库的前提下把"配对"和"三角测量"分开**；
- 单目其余（灰度 / 相机自运动 / 单目三维）：用 `process()` 总时长减去
  代理测到的 detect 与 track，不额外采样；
- 读帧 / 叠加渲染 / 视频编码：在循环里直接夹 `perf_counter()`。

## 用法::

    python apps/tools/profile_stages.py --session data/sessions/20261024-173619 \
        --max-frames 60 --repeat 3

    # 只量计算、不量 mp4 编码（编码是离线专有开销，实时链路没有）
    python apps/tools/profile_stages.py --session <dir> --no-video

## 读结果的正确姿势

同机并行跑别的负载（测试 / 别的 agent）时，**绝对值会整体漂移**，
所以每段都跑 `--repeat` 次取中位数，并且**看占比、不看绝对值**。
表里的 p90 用所有重复的逐帧样本一起算，用来看"是否有偶发尖峰"。

## 边界

- 只跑主循环，不做 CSV/JSON 导出、不做 matplotlib 绘图、不做弹道预测
  （`cfg.predict.enable=False` 与正式入口一致；导出与绘图都在循环之外）。
- **CSV 行的字符串格式化没有计入任何阶段**：它是 O(1) 的格式化，
  在 1718x972 的检测开销面前不可能是候选，硬凑一个阶段只会误导。
- 视频写进临时目录，默认跑完删除；`--out` 可保留（用于肉眼核对叠加是否正常）。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import tempfile
import time
import unicodedata
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "sphere_tracker").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calib import StereoRig                                    # noqa: E402
from sphere_tracker.config import AppConfig, ColorConfig       # noqa: E402
from sphere_tracker.plugins import default as default_method   # noqa: E402,F401
from stereo import MatchConfig, PairSource, StereoReconstructor  # noqa: E402
from stereo.sessions import load_session, validate_session     # noqa: E402
from stereo.stereo_mode import mode_label, optical_axes_angle_deg, resolve_auto  # noqa: E402


# --------------------------------------------------------------------------- #
# 阶段名（表里的顺序由占比排序决定，这里只定义"名字 -> 中文标签"）
# --------------------------------------------------------------------------- #
LABELS = {
    "read_pair": "读帧/解码(左右)",
    "detect_left": "单路检测(L)",
    "detect_right": "单路检测(R)",
    "track_left": "跟踪(L)",
    "track_right": "跟踪(R)",
    "mono_rest": "单目其余(灰度/自运动/三维)",
    "stereo_pair": "双目配对(极线)",
    "stereo_triang": "三角测量",
    "stereo_rest": "双目其余(诊断/计数)",
    "stereo_update": "双目(配对+三角)",
    "render_draw": "叠加+缩放+拼接",
    "encode_write": "视频编码写盘",
    "unaccounted": "未归类(计时缝隙)",
}


# --------------------------------------------------------------------------- #
# 计时汇：只存数，不打印
# --------------------------------------------------------------------------- #
class Sink:
    """按阶段累加毫秒与逐帧样本。"""

    def __init__(self) -> None:
        self.sums: dict[str, float] = {}
        self.samples: dict[str, list[float]] = {}

    def add(self, key: str, ms: float) -> None:
        self.sums[key] = self.sums.get(key, 0.0) + ms
        self.samples.setdefault(key, []).append(ms)

    def snapshot(self) -> dict[str, float]:
        return dict(self.sums)


class TimedCall:
    """把一次调用包成计时，返回值原样透传。"""

    __slots__ = ("sink", "key")

    def __init__(self, sink: Sink, key: str) -> None:
        self.sink = sink
        self.key = key

    def __call__(self, fn, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            self.sink.add(self.key, (time.perf_counter() - t0) * 1000.0)


class TimedDetector:
    """检测器计时代理。属性全量转发，`seed()` / `reset()` 等照常可用。"""

    def __init__(self, inner, call: TimedCall) -> None:
        self._inner = inner
        self._call = call

    def detect(self, frame, ctx):
        return self._call(self._inner.detect, frame, ctx)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class TimedTracker:
    """跟踪器计时代理。"""

    def __init__(self, inner, call: TimedCall) -> None:
        self._inner = inner
        self._call = call

    def update(self, detections, t):
        return self._call(self._inner.update, detections, t)

    def __getattr__(self, name):
        return getattr(self._inner, name)


@contextmanager
def timed_module_functions(module, mapping: dict[str, str], sink: Sink):
    """临时把模块级函数换成计时透传包装，退出时**无条件还原**。

    为什么敢这么做：`StereoReconstructor` 调用的是模块命名空间里的名字，
    替换它对调用方完全透明（入参、返回值、异常都原样）。不改库也能把
    配对与三角测量分开计时；这是本项目"只改一个库文件"约束下唯一的路。

    `mapping` 是 `{函数名: 阶段 key}`，阶段 key 统一由这里决定，
    免得同一段时间被记到两个名字下（那会让占比之和超过 100%）。
    """
    saved: dict[str, object] = {}
    try:
        for name, key in mapping.items():
            orig = getattr(module, name)
            saved[name] = orig
            call = TimedCall(sink, key)

            def wrapper(*args, _orig=orig, _call=call, **kwargs):
                return _call(_orig, *args, **kwargs)

            setattr(module, name, wrapper)
        yield
    finally:
        for name, orig in saved.items():
            setattr(module, name, orig)


# --------------------------------------------------------------------------- #
# 复用正式入口的实现（同一个模块，不做第二套参数语义）
# --------------------------------------------------------------------------- #
def load_offline_module():
    """按路径加载 `apps/run_stereo_video.py`（apps/ 不是包，没有 __init__）。"""
    path = ROOT / "apps" / "run_stereo_video.py"
    spec = importlib.util.spec_from_file_location("_offline_stereo_entry", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"加载不了离线入口：{path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def resolve_mode(args, session_dir: Path, rig: StereoRig, rsv) -> str | None:
    """把 auto 落定成并行/大角度。与 `run_stereo_video` 同一套优先级。

    返回 None 表示数据自相矛盾、应当拒跑（与正式入口同一口径），
    而不是"猜一个继续跑" —— 猜错会让三维点整体错位且看不出来。
    """
    try:
        raw = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    except Exception:                                       # noqa: BLE001
        raw = {}
    hint: dict = {}
    if isinstance(raw, dict):
        if "stereo_mode" in raw:
            hint["stereo_mode"] = raw["stereo_mode"]
        rig_hint = raw.get("rig")
        if isinstance(rig_hint, dict) and "stereo_mode" in rig_hint:
            hint["rig"] = {"stereo_mode": rig_hint["stereo_mode"]}
    if args.stereo_mode == "auto":
        mode = resolve_auto(args.stereo_mode, session=hint, calibration={"meta": rig.meta})
    else:
        mode = resolve_auto(args.stereo_mode)
    mode, switched = rsv._mode_guard(mode, rig.rectification)
    if switched:
        print("[模式] 标定是 none_raw 但会话记为 parallel：按 angled 处理")
    angle = optical_axes_angle_deg(rig.R)
    print(f"双目模式：{mode_label(mode)}（{mode}）；光轴夹角 {angle:.1f}°；"
          f"基线 {rig.baseline_mm:.1f} mm")
    if args.stereo_mode == "auto" and (
            (mode == "parallel" and angle >= 40.0)
            or (mode == "angled" and angle < 15.0)):
        print("[错误] 模式与标定实测夹角矛盾，拒绝继续（与 run_stereo_video 同口径）")
        return None
    return mode


# --------------------------------------------------------------------------- #
# 单次运行
# --------------------------------------------------------------------------- #
def run_once(args, rsv, session_dir, s, rig, mode: str, out_dir: Path, run_index: int):
    """跑一遍主循环，返回 (Sink, 帧数, 逐帧总耗时列表, 诊断 dict)。"""
    left_video = session_dir / s.left.video
    right_video = session_dir / s.right.video
    fps = float(s.left.fps_measured or s.left.fps_requested or 30.0)
    offset = args.sync_offset if args.sync_offset is not None else int(s.sync.offset)

    src = PairSource(video=(left_video, right_video), sync_offset=offset,
                     pairing=args.pairing,
                     timestamp_tolerance_s=args.timestamp_tolerance)
    if run_index == 0:
        src.report_warnings()
        if args.pairing != "index":
            print(f"实际配对：{src.describe()}")
    if tuple(src.image_size) != tuple(rig.image_size):
        src.release()
        raise SystemExit(f"[错误] 视频 {src.image_size} 与标定 {rig.image_size} 不一致")

    cfg = AppConfig()
    cfg.detector.method = args.method
    if args.method == "specular" and args.max_epipolar is None:
        args.max_epipolar = 8.0
    cfg.detector.min_radius = float(args.min_radius)
    cfg.detector.max_radius = float(args.max_radius)
    cfg.detector.hough_max_radius = float(args.hough_max_radius)
    cfg.predict.enable = False          # 与 run_stereo_video 一致：离线不做预测
    if args.preset:
        cfg.detector.color = ColorConfig.preset(args.preset)

    pipe_l = rsv._make_pipeline(cfg, rig, "left")
    pipe_r = rsv._make_pipeline(cfg, rig, "right")
    if args.method == "specular":
        seed_path = session_dir / "derived" / "seeds.json"
        if not seed_path.exists():
            src.release()
            raise SystemExit(f"[错误] specular 需要 {seed_path}（先用 GUI/CLI 点选球）")
        seeds = rsv.load_seeds(seed_path)
        pipe_l.detector.seed(float(seeds["left"]["x"]), float(seeds["left"]["y"]),
                             float(seeds["left"]["radius"]),
                             frame_index=int(seeds.get("seed_frame", 0)))
        pipe_r.detector.seed(float(seeds["right"]["x"]), float(seeds["right"]["y"]),
                             float(seeds["right"]["radius"]),
                             frame_index=int(seeds.get("seed_frame", 0)))
    if args.detect_width:
        pipe_l.set_detect_width(args.detect_width)
        pipe_r.set_detect_width(args.detect_width)

    if mode == "angled":
        from stereo.angled import AngledMatchConfig, AngledReconstructor
        epi = 8.0 if args.max_epipolar is None else args.max_epipolar
        rec = AngledReconstructor(rig, AngledMatchConfig(max_epipolar_px=epi))
        parallel = False
    else:
        epi = 2.5 if args.max_epipolar is None else args.max_epipolar
        rec = StereoReconstructor(rig, MatchConfig(max_epipolar_px=epi))
        parallel = True

    sink = Sink()
    pipe_l.detector = TimedDetector(pipe_l.detector, TimedCall(sink, "detect_left"))
    pipe_r.detector = TimedDetector(pipe_r.detector, TimedCall(sink, "detect_right"))
    pipe_l.tracker = TimedTracker(pipe_l.tracker, TimedCall(sink, "track_left"))
    pipe_r.tracker = TimedTracker(pipe_r.tracker, TimedCall(sink, "track_right"))

    writer = None
    if not args.no_video:
        video_path = out_dir / f"profile_run{run_index}.mp4"
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (1920, 540))
        if not writer.isOpened():
            print("[警告] 打不开 VideoWriter，本次不量编码")
            writer = None

    import stereo.triangulate as tri

    trail_l: list[tuple[float, float]] = []
    trail_r: list[tuple[float, float]] = []
    loop_ms: list[float] = []
    n = 0
    t_loop = time.perf_counter()
    ctx_patch = (timed_module_functions(
        tri, {"match_epipolar": "stereo_pair",
             "triangulate_one": "stereo_triang"}, sink)
        if parallel else _nullcontext())
    try:
        with ctx_patch:
            while True:
                t0 = time.perf_counter()
                ok, left_frame, right_frame = src.read()
                t_read = time.perf_counter()
                if not ok or left_frame is None or right_frame is None:
                    break
                if args.max_frames and n >= args.max_frames:
                    break
                sink.add("read_pair", (t_read - t0) * 1000.0)

                t = n / fps
                before = sink.snapshot()

                t_p0 = time.perf_counter()
                res_l = pipe_l.process(left_frame, t)
                res_r = pipe_r.process(right_frame, t)
                t_p1 = time.perf_counter()
                after = sink.snapshot()
                inner = sum(after.get(k, 0.0) - before.get(k, 0.0) for k in
                            ("detect_left", "detect_right", "track_left", "track_right"))
                # 单目其余 = process 总时长 - 代理测到的四段。
                # 夹住下界：代理里 perf_counter 本身有亚微秒误差，负数是噪声不是"负耗时"。
                sink.add("mono_rest", max(0.0, (t_p1 - t_p0) * 1000.0 - inner))

                det_l = rsv._as_detection(res_l)
                det_r = rsv._as_detection(res_r)

                t_s0 = time.perf_counter()
                pt = rec.update(t, det_l, [det_r] if det_r is not None else [])
                t_s1 = time.perf_counter()
                if parallel:
                    pair_ms = sink.sums.get("stereo_pair", 0.0) - before.get("stereo_pair", 0.0)
                    tri_ms = sink.sums.get("stereo_triang", 0.0) - before.get("stereo_triang", 0.0)
                    # 双目其余 = update 总时长 - 配对 - 三角测量（诊断计数/低配率告警）
                    sink.add("stereo_rest", max(0.0, (t_s1 - t_s0) * 1000.0
                                                - pair_ms - tri_ms))
                else:
                    sink.add("stereo_update", (t_s1 - t_s0) * 1000.0)

                if pt is not None:
                    trail_l.append(pt.u_left)
                    trail_r.append(pt.u_right)
                    if args.trail and len(trail_l) > args.trail:
                        trail_l = trail_l[-args.trail:]
                        trail_r = trail_r[-args.trail:]

                t_r0 = time.perf_counter()
                if writer is not None:
                    vl = cv2.resize(rsv._draw(left_frame, res_l, "L", trail_l,
                                              xyz=None if pt is None else pt.xyz,
                                              rig=rig, origin=args._origin,
                                              draw_reference=True, mode=mode),
                                    (960, 540))
                    vr = cv2.resize(rsv._draw(right_frame, res_r, "R", trail_r,
                                              xyz=None if pt is None else pt.xyz),
                                    (960, 540))
                    canvas = np.hstack([vl, vr])
                    t_r1 = time.perf_counter()
                    sink.add("render_draw", (t_r1 - t_r0) * 1000.0)
                    writer.write(canvas)
                    sink.add("encode_write", (time.perf_counter() - t_r1) * 1000.0)
                else:
                    t_r1 = time.perf_counter()
                    sink.add("render_draw", (t_r1 - t_r0) * 1000.0)

                n += 1
                now = time.perf_counter()
                loop_ms.append((now - t0) * 1000.0)
    finally:
        src.release()
        if writer is not None:
            writer.release()
    elapsed = time.perf_counter() - t_loop
    diag = {
        "frames": n,
        "matched": rec.n_matched,
        "no_left": rec.n_no_left,
        "no_right": rec.n_no_right,
        "rejected": rec.n_rejected,
        "wall_s": elapsed,
    }
    return sink, loop_ms, diag


@contextmanager
def _nullcontext():
    yield


# --------------------------------------------------------------------------- #
# 统计与呈现
# --------------------------------------------------------------------------- #
def _disp_width(text: str) -> int:
    """终端显示宽度：中文按 2 列算，否则表格对不齐。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def _pad(text: str, width: int, *, right: bool = False) -> str:
    fill = " " * max(0, width - _disp_width(text))
    return fill + text if right else text + fill


def _p90(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), 90))


def aggregate(runs: list[tuple[Sink, list[float], dict]], warmup: int):
    """把多次运行的样本合并，返回 (阶段行, 总量统计)。

    每段的中位 / p90 用**所有重复的逐帧样本**一起算（样本总量 = 帧数 x 次数），
    另外给"每次运行的中位数范围"，用来暴露并发负载导致的整体漂移。
    """
    pooled: dict[str, list[float]] = {}
    per_run_median: dict[str, list[float]] = {}
    totals: list[float] = []
    total_medians: list[float] = []
    frames = 0
    for sink, loop_ms, _diag in runs:
        # 预热帧不计入统计：首帧含相机模型/检测器懒初始化与解码器填充，
        # 那是启动成本，不是稳态成本。
        skip = {k: list(v[warmup:]) for k, v in sink.samples.items()}
        frame_ms = loop_ms[warmup:]
        frames += len(frame_ms)
        for k, v in skip.items():
            pooled.setdefault(k, []).extend(v)
            if v:
                per_run_median.setdefault(k, []).append(statistics.median(v))
        if frame_ms:
            totals.extend(frame_ms)
            total_medians.append(statistics.median(frame_ms))

    total_sum = sum(totals)
    rows = []
    for key, vals in pooled.items():
        if not vals:
            continue
        ms = statistics.median(vals)
        share = (sum(vals) / total_sum * 100.0) if total_sum > 0 else 0.0
        rows.append({
            "key": key,
            "label": LABELS.get(key, key),
            "median": ms,
            "share": share,
            "p90": _p90(vals),
            "run_min": min(per_run_median.get(key, [0.0])),
            "run_max": max(per_run_median.get(key, [0.0])),
        })
    # 未归类：逐帧总时长 - 各段之和。它必须接近 0，否则说明漏了一段
    accounted = sum(sum(v) for v in pooled.values())
    unacc = total_sum - accounted
    rows.append({
        "key": "unaccounted", "label": LABELS["unaccounted"],
        "median": unacc / max(1, len(totals)),
        "share": unacc / total_sum * 100.0 if total_sum > 0 else 0.0,
        "p90": 0.0, "run_min": 0.0, "run_max": 0.0,
    })
    rows.sort(key=lambda r: r["share"], reverse=True)
    totals_summary = {
        "frames": frames,
        "median_ms": statistics.median(totals) if totals else 0.0,
        "p90_ms": _p90(totals),
        "fps": 1000.0 / statistics.median(totals) if totals else 0.0,
        "run_medians": total_medians,
    }
    return rows, totals_summary


def print_table(rows, summary, runs, warmup: int) -> None:
    headers = ["阶段", "中位 ms", "占比", "p90 ms", "每次中位(最小~最大)"]
    body = []
    for r in rows:
        if r["key"] == "unaccounted":
            body.append([r["label"], f"{r['median']:.2f}", f"{r['share']:.1f}%", "-", "-"])
        else:
            body.append([
                r["label"], f"{r['median']:.2f}", f"{r['share']:.1f}%",
                f"{r['p90']:.2f}", f"{r['run_min']:.2f}~{r['run_max']:.2f}",
            ])
    body.append(["合计(逐帧)", f"{summary['median_ms']:.2f}", "100.0%",
                 f"{summary['p90_ms']:.2f}", "-"])
    widths = [max(_disp_width(h), max(_disp_width(b[i]) for b in body))
              for i, h in enumerate(headers)]
    line = "  ".join(_pad(h, widths[i]) for i, h in enumerate(headers))
    print(line)
    print("-" * _disp_width(line))
    for b in body:
        print("  ".join(_pad(c, widths[i], right=(i > 0)) for i, c in enumerate(b)))
    print()
    fps = summary["fps"]
    print(f"样本：{len(runs)} 次 x 约 {summary['frames'] // max(1, len(runs))} 帧"
          f"（每次丢弃前 {warmup} 帧预热）")
    print(f"主循环：{fps:.2f} fps（{summary['median_ms']:.2f} ms/帧，中位）")
    if summary["run_medians"]:
        spread = "  ".join(f"{m:.2f}" for m in summary["run_medians"])
        print(f"各次运行中位：{spread} ms —— 相差大说明同机有其他负载在抢 CPU")
    print()
    print("注：绝对值受同机并发负载影响，**占比**才是稳定的结论；"
          "「未归类」应为 0 附近，明显偏大说明有段时间没被任何阶段覆盖。")


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description="离线双目分段计时（只包计时，不改任何计算结果）")
    ap.add_argument("--session", required=True, help="data/sessions/<id> 目录或 session.json")
    ap.add_argument("--max-frames", type=int, default=0, help="只跑前 N 对（0=全部）")
    ap.add_argument("--repeat", type=int, default=3,
                    help="重复次数，取中位数抵御抖动（默认 3）")
    ap.add_argument("--warmup", type=int, default=2,
                    help="每次运行丢弃前 N 帧（首帧含懒初始化，默认 2）")
    ap.add_argument("--method", default=default_method("detector"),
                    help="检测方法（默认与 run_stereo_video 一致）")
    ap.add_argument("--preset", default=None, choices=ColorConfig.preset_names())
    ap.add_argument("--detect-width", type=int, default=0)
    ap.add_argument("--min-radius", type=float, default=8.0)
    ap.add_argument("--max-radius", type=float, default=200.0)
    ap.add_argument("--hough-max-radius", type=float, default=80.0)
    ap.add_argument("--max-epipolar", type=float, default=None,
                    help="极线门限（px）；缺省平行 2.5 / angled 8")
    ap.add_argument("--pairing", choices=("index", "timestamp", "canary"), default="canary")
    ap.add_argument("--timestamp-tolerance", type=float, default=0.033)
    ap.add_argument("--sync-offset", type=int, default=None)
    ap.add_argument("--stereo-mode", choices=("auto", "parallel", "angled"), default="auto")
    ap.add_argument("--origin", default="0,0,0", help="叠加图上标出的参照原点（米）")
    ap.add_argument("--trail", type=int, default=300)
    ap.add_argument("--no-video", action="store_true",
                    help="不写标注视频（同时不量「视频编码写盘」，实时链路本来也没有这一段）")
    ap.add_argument("--out", default=None, metavar="DIR",
                    help="标注视频输出目录（默认临时目录，跑完删除）")
    args = ap.parse_args()

    if args.repeat < 1:
        print("[错误] --repeat 至少为 1")
        return 2
    try:
        origin = tuple(float(v) for v in args.origin.split(","))
        if len(origin) != 3:
            raise ValueError
    except ValueError:
        print(f"[错误] --origin 要写成 x,y,z 三个数，现在是 {args.origin!r}")
        return 2
    args._origin = origin

    rsv = load_offline_module()
    session_dir, s = load_session(args.session)
    errors, warnings = validate_session(session_dir)
    for w in warnings:
        print(f"  [警告] {w}")
    if errors:
        for e in errors:
            print(f"  [错误] {e}")
        return 1
    if s.left is None or s.right is None or s.calibration is None:
        print("[错误] session 缺 left/right/calibration")
        return 1

    calib_path = Path(s.calibration.path)
    if not calib_path.is_absolute():
        calib_path = session_dir / calib_path
    rig = StereoRig.load(calib_path)
    mode = resolve_mode(args, session_dir, rig, rsv)
    if mode is None:
        return 2

    # 与 `apps/run_stereo_video.py` 同一条约束（那边 515 行）：`specular` 的种子
    # 是**原图坐标**，一旦缩放检测宽度，种子与检测结果就错位，检测框根本不在球上。
    # 这不是性能取舍，是**语义不成立** —— 剖析工具如果放过这个组合，会测出
    # 一个产品里根本不允许的配置，然后拿它的数字去指导优化（我第一版就这么干了：
    # 960 宽"0 匹配"其实是被自己的错位坑掉的，不是极线门限的问题）。
    if args.detect_width and args.method == "specular":
        print("[错误] --method specular 暂不支持 --detect-width："
              "种子坐标是原图坐标，缩放会错位。")
        return 2

    print(f"会话：{session_dir.name}  {s.left.width}x{s.left.height}  "
          f"偏移 {s.sync.offset:+d} 帧  方法 {args.method}")
    print(f"检测宽度：{args.detect_width or '原图'}；重复 {args.repeat} 次；"
          f"预热丢弃 {args.warmup} 帧")

    tmp = None
    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        tmp = tempfile.TemporaryDirectory(prefix="profile_stages_")
        out_dir = Path(tmp.name)

    runs = []
    try:
        for k in range(args.repeat):
            sink, loop_ms, diag = run_once(args, rsv, session_dir, s, rig, mode,
                                           out_dir, k)
            runs.append((sink, loop_ms, diag))
            print(f"  第 {k + 1}/{args.repeat} 次：{diag['frames']} 对，"
                  f"{diag['wall_s']:.1f}s，匹配 {diag['matched']}，"
                  f"拒绝 {diag['rejected']}")
    finally:
        if tmp is not None:
            tmp.cleanup()

    rows, summary = aggregate(runs, args.warmup)
    print()
    print_table(rows, summary, runs, args.warmup)

    top = rows[0]
    print()
    print(f"最大项：{top['label']}（{top['median']:.2f} ms，占 {top['share']:.1f}%）")
    if top["share"] < 30.0:
        print("结论：没有任何单项占比达到 30%，分段比较平均 —— 数据不支持做单点优化。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
