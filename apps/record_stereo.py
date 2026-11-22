"""双路无损录制：两台相机各写一个无损 MKV（默认 HFYU），并落 `session.json`。

为什么不是直接用 `PairSource` 录
--------------------------------
`PairSource` 的相机模式是"永远取最新帧"——处理慢就丢帧，这是实时预览要的语义，
但**录制要每一帧都留下**。所以这里自己开两路 `CameraSource`，各跑一个采集线程、
各写一个 writer，主线程只负责计时 / 预览 / 停止。

为什么不用 `apps._common.add_stereo_source_args`（**别"统一"它们**）
-------------------------------------------------------------------
两处都定义了 `--left/--right/--backend/--width/--height/--fps`，看起来像重复，
但**不是重复，是有意分开**：

- 那个共用函数服务于"**读**现有素材"（`stereo.from_args` 消费）：多给
  `--auto-cameras`/`--backup-camera`/`--left-video`/`--left-dir` 等 8 个选项；
- 录制器只**开相机**，一个都读不到（`grep args.` 可验证），其中 `--left-video`
  出现在**录制**命令里更是纯误导；
- 两侧默认值语义也不同：共用侧 `--left=None`（= 交给 `auto` 推断），
  录制侧 `--left="left"`（= 缺省就按角色名开工，双击 bat 直接能录）。

数值层面**已经没有第二个来源**：帧率来自 `video_profiles.DEFAULT_FPS`，
而采集档的**兜底默认**来自 `video_profiles.RECORD_DEFAULT`（都注明"全项目只此一处"）。
真正的单一是**数值单一**，不是硬把两份 argparse 揉成一份 —— 揉完就得往共用函数里
塞"录制专用"的帮助文案，耦合方向反而错了。

采集档（2026-10-31）：`--width/--height` **默认 None**。未显式给时，先读标定
meta 记的 `capture_size`（= 标定时的**实际采集分辨率**），与标定**同一档**才能
保证 FOV 裁剪区与 `validate_capture_size` 对得上；标定没记（旧标定）则明确说清、
退回 `RECORD_DEFAULT`。给了显式值就按显式值（不改）。

FOV 生效时（标定 meta 带 `fov`）写盘 / 预览用的是标定的 `fov.target_size`
（两路共同、**不放大**，当前一对约 1684x978），**不是**上面的采集档：
相机仍按采集档打开，再在采集线程里逐帧归一化。
`session.json` 的 `left/right.width/height`、`calibration.image_size` 都记
写盘尺寸；无损 MKV 只接受偶数宽高，奇数 target 会在录制前退出 2。

**angled 模式不做 FOV 对齐**：两路光轴近乎垂直、视场几乎不重叠，单应裁剪
没有物理意义。此时写盘就是两路各自的原始画面（`--width/--height`），
`fov` 记录即使存在也会被忽略（并在日志里说明）。

录完做什么
----------
- 写 `session.json`（分辨率 / 实测 fps / 帧数 / 标定快照 + sha256 / 同步偏移）；
- 用两路逐帧亮度找闪光/拍手偏移（`stereo.sync.find_offset`）；
- 跑 `validate_session`，有 error 就退出码 1（文件保留，方便查）。

用法::

    python apps/record_stereo.py --left left --right right --duration 10 --preview
    python apps/record_stereo.py --left 0 --right 1 --backend dshow --duration 10

⚠️ 角色名 + `--backend dshow` 会被拒绝：角色名解析出的是 PnP 位次，
   DSHOW 编号是另一套命名空间（见 `docs/交接文档.md` §10.13）。
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import shutil
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "sphere_tracker").is_dir())
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps._common import calib_capture_size, setup_logging      # noqa: E402
from calib.params import StereoRig                           # noqa: E402
from stereo.fov_align import (                              # noqa: E402
    MIN_FOV_INLIERS,
    MIN_FOV_QUALITY,
    FovAlignment,
    fov_confidence_problem,
)
from stereo.stereo_mode import (                             # noqa: E402
    optical_axes_angle_deg, resolve_auto,
)
from stereo.profiles import ProfileError, resolve_calibration_base  # noqa: E402
from stereo import fov_policy                               # noqa: E402
from stereo.record_plan import output_size, plan_record     # noqa: E402
from sphere_tracker.camera_registry import (                 # noqa: E402
    list_devices,
    load_registry,
    resolve_camera_arg,
)
from sphere_tracker.sources import (                         # noqa: E402
    INDEX_BACKEND,
    CameraSource,
)
from sphere_tracker.paths import STEREO_ROOT                 # noqa: E402
from sphere_tracker.video_profiles import (                  # noqa: E402
    DEFAULT_FPS,
    RECORD_DEFAULT,
    fps_type,
)
from stereo.sessions import (                                # noqa: E402
    CalibrationInfo,
    CameraInfo,
    SyncInfo,
    create_session,
    describe,
    save_session,
    sha256_file,
    validate_session,
)
from stereo.sync import (                                   # noqa: E402
    SyncConfig,
    brightness_jumps,
    config_for_fps,
    find_offset,
    timestamp_offset,
)

LOG = logging.getLogger("record")
# 1080p30 × 2 的实测吞吐（本机）：HFYU ≈ 34 fps/路；FFV1 ≈ 8 fps/路。
# 所以默认 HFYU（无损、能跟上相机），FFV1 留给"不赶时间"的归档。
# 三元组：(fourcc, 默认扩展名, 说明) —— 仿 record_mono.CODECS 的形状。
# 默认容器只有 mkv；avi 通过 --container 显式选。mp4 **绝不在选项里**：
# HFYU/FFV1 进 mp4 会被 FFmpeg 静默换成 h264（有损），且 isOpened() 假成功。
CODECS = {"hfyuv": ("HFYU", ".mkv", "HuffYUV（无损，默认）"),
          "ffv1": ("FFV1", ".mkv", "FFV1（无损，但本机 2×1080p 只有约 8 fps/路）")}


def _container_of(args) -> str:
    """实际容器名（不含点）：显式 ``--container`` 优先，否则跟随 codec 默认扩展名。

    返回值只可能是 ``mkv`` / ``avi``（``--container`` 的 choices 里没有 mp4，
    理由见 :data:`CODECS` 注释）。所有写盘路径、``session.container``、
    ``CameraInfo.video`` 都必须用它动态拼，不许写死 ``mkv``。
    """
    want = getattr(args, "container", None)
    if want:
        return want
    return CODECS[args.codec][1].lstrip(".")


def _backend_of(args) -> int:
    if args.backend == "msmf":
        return cv2.CAP_MSMF
    if args.backend == "dshow":
        return cv2.CAP_DSHOW
    # auto：角色名/PnP 位次只和 MSMF 对齐；显式编号也默认 MSMF，避免两路各挑一个后端
    return INDEX_BACKEND


def _looks_like_role(value: str) -> bool:
    text = str(value or "").strip().lower()
    return bool(text) and not text.isdigit() and not any(
        text.startswith(p) for p in ("port:", "id:", "name:", "auto"))


def _role_spec(reg, value: str):
    if not _looks_like_role(value):
        return None
    return reg.roles.get(str(value).strip())


def _open_one(side: str, value: str, args, api: int, out: dict) -> None:
    try:
        reg = load_registry()
        idx = resolve_camera_arg(value, reg=reg, interactive=False)
        if idx is None:
            raise RuntimeError(f"{side} 没有解析出相机编号")
        spec = _role_spec(reg, value)
        if args.capture_api in ("auto", "mf") and spec is not None \
                and (spec.identity or spec.port):
            from sphere_tracker.sources import open_camera_source
            src = open_camera_source(
                index=idx, identity=spec.identity, port=spec.port,
                width=args.width, height=args.height, fps=args.fps,
                backend=api, optimize=False, first_frame_timeout_s=30.0,
                capture_api=args.capture_api)
        else:
            src = CameraSource(
                index=idx,
                width=args.width,
                height=args.height,
                fps=args.fps,
                backend=api,
                optimize=False,
                first_frame_timeout_s=30.0,
            )
        out[side] = (idx, src)
    except BaseException as exc:                            # noqa: BLE001
        out[side] = exc


def _open_pair(args) -> tuple[dict, int]:
    if args.backend == "dshow" and (
            _looks_like_role(args.left) or _looks_like_role(args.right)):
        print("[错误] 角色名不能和 --backend dshow 一起用：角色名解析出的是 PnP 位次，")
        print("       DSHOW 编号是另一套命名空间。请改成显式编号（如 --left 0 --right 1），")
        print("       或把后端改回 msmf。")
        return {}, 2

    api = _backend_of(args)
    print(f"[后端] {('MSMF' if api == cv2.CAP_MSMF else 'DSHOW')}；"
          f"请求 {args.width}x{args.height}@{args.fps:g} fps；"
          f"采集 API={args.capture_api}")
    print("[打开] 并行打开两路（首次可能要 20 秒以上，请等）…")
    out: dict = {}
    threads = [
        threading.Thread(target=_open_one, args=("left", args.left, args, api, out),
                         daemon=True, name="open-left"),
        threading.Thread(target=_open_one, args=("right", args.right, args, api, out),
                         daemon=True, name="open-right"),
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    errors = []
    for side in ("left", "right"):
        item = out.get(side)
        if isinstance(item, BaseException):
            errors.append(f"{side}: {item}")
        elif item is None:
            errors.append(f"{side}: 没有结果")
    if errors:
        print("[错误] 相机没打开：")
        for e in errors:
            print(f"  - {e}")
        return {}, 1

    cams = {}
    for side in ("left", "right"):
        idx, src = out[side]
        info = src.info
        print(f"[{side}] camera {idx}: {info.width}x{info.height} @ {info.fps:.1f} fps "
              f"({info.backend})")
        if (info.width, info.height) != (args.width, args.height):
            print(f"[错误] {side} 实际 {info.width}x{info.height}，"
                  f"与请求 {args.width}x{args.height} 不一致；双目要求两路完全一致")
            for item in out.values():
                if isinstance(item, tuple):
                    item[1].release()
            return {}, 1
        cams[side] = (idx, src)
    return cams, 0


def _lossless_selfcheck(width: int, height: int, fps: float, tag: str,
                        container: str = "mkv") -> tuple[bool, str]:
    """写两帧 -> 读回 -> 逐位比较；编码器不可用就提前失败，别录完才发现。

    ``container`` 决定自检文件的扩展名（mkv/avi）—— 真正录制用什么容器，
    自检就得用什么容器验，不能拿 mkv 的自检结果给 avi 录制背书。
    """
    try:
        fourcc = cv2.VideoWriter_fourcc(*tag)
        with tempfile.TemporaryDirectory(prefix="lossless_check_") as td:
            path = Path(td) / f"check.{container}"
            vw = cv2.VideoWriter(str(path), fourcc, float(fps), (width, height))
            if not vw.isOpened():
                return False, f"VideoWriter 打不开 {tag}/{container.upper()}"
            rng = np.random.default_rng(12345)
            frames = [
                rng.integers(0, 256, (height, width, 3), dtype=np.uint8),
                rng.integers(0, 256, (height, width, 3), dtype=np.uint8),
            ]
            for f in frames:
                vw.write(f)
            vw.release()
            cap = cv2.VideoCapture(str(path))
            try:
                ok, got = cap.read()
                if not ok or got is None:
                    return False, f"写出的 {tag} 读不回第一帧"
                if got.shape != frames[0].shape:
                    return False, f"读回尺寸 {got.shape} != {frames[0].shape}"
                if not np.array_equal(got, frames[0]):
                    return False, f"{tag} 第一帧不是逐位一致"
                ok, got = cap.read()
                if not ok or got is None or not np.array_equal(got, frames[1]):
                    return False, f"{tag} 第二帧不是逐位一致"
            finally:
                cap.release()
            return True, f"{tag} 写/读逐位一致"
    except Exception as exc:                                # noqa: BLE001
        return False, f"{tag} 自检异常：{exc}"


class _SideRecorder(threading.Thread):
    def __init__(self, side: str, src: CameraSource, writer, start_evt: threading.Event,
                 stop_evt: threading.Event, preview: dict, lock: threading.Lock,
                 fov: FovAlignment | None = None):
        super().__init__(daemon=True, name=f"record-{side}")
        self.side = side
        self.src = src
        self.writer = writer
        self.start_evt = start_evt
        self.stop_evt = stop_evt
        self.preview = preview
        self.lock = lock
        self.fov = fov
        self.times: list[tuple[int, float, float]] = []
        self.brightness: list[float] = []
        self.frame_count = 0
        self.errors = 0
        self.error = ""

    def run(self) -> None:
        self.start_evt.wait()
        while not self.stop_evt.is_set():
            ok, frame, t = self.src.read()
            if not ok or frame is None:
                self.errors += 1
                if self.errors >= 30:
                    self.error = f"连续 {self.errors} 帧读不到"
                    return
                time.sleep(0.005)
                continue
            self.errors = 0
            # 先按标定里的 FOV 参数归一化，再写盘 / 统计 / 预览 ——
            # 这样录出的两路视频与标定 image_size 一致，不会写入原生视场。
            if self.fov is not None:
                try:
                    frame = self.fov.apply_side(frame, self.side)
                except ValueError as exc:
                    self.error = f"FOV 归一化失败（{self.side}）：{exc}"
                    return
            try:
                self.writer.write(frame)
            except Exception as exc:                        # noqa: BLE001
                self.error = f"writer.write 失败：{exc}"
                return
            self.times.append((self.frame_count, t, time.time()))
            self.brightness.append(float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()))
            self.frame_count += 1
            with self.lock:
                self.preview[self.side] = frame


def _container_fps(info_fps: float, requested: float) -> float:
    """容器（MKV）标称帧率：优先用相机协商值，取不到才退回请求值。

    ``info_fps`` 来自 ``CameraSource.info.fps``（``sources.py`` 里由
    ``cap.get(CAP_PROP_FPS)`` 协商/实测得到，见 ``SourceInfo(... real_fps ...)``）。
    ``--fps`` 只是给相机的**请求提示**，相机可能给别的值；容器头若照抄请求值，
    播放器与 ``FileSource`` 的秒数换算会系统性偏差（2026-10-24 实录 301 帧 /
    10.002 s ≈ 29.99 fps，而容器头被写成 24，慢了 25%）。
    """
    try:
        v = float(info_fps)
    except (TypeError, ValueError):
        v = float("nan")
    if not math.isfinite(v) or v < 1.0:
        v = float(requested)
    return round(v, 3)


#: 单条 USB3 链路的**有效**带宽上限（GB/s）：5 Gbps 扣掉协议/编码开销后的现实值。
#: 判的是**单路**不是两路之和 —— 两台相机通常各占一个口、各吃自己的链路；
#: 按两路求和会把本机已实测跑通的 2560x1440@24（单路 0.265）也报成告警。
USB3_EFFECTIVE_GBPS = 0.5


def stream_bandwidth(width: int, height: int, fps: float) -> tuple[float, float, float]:
    """两路 BGR24 裸流估算：``(单帧 MB, 单路 GB/s, 两路合计 GB/s)``。

    纯函数（不读日志、不碰相机），判据直接调它；日志只是它的消费者。
    未压缩估算 —— 实际码流/带宽以驱动给到的为准（很多相机内部就是 MJPG）。
    """
    frame_mb = width * height * 3 / 1e6
    per_cam_gbps = width * height * float(fps) * 3 / 1e9
    return frame_mb, per_cam_gbps, per_cam_gbps * 2.0


def _make_writer(path: Path, width: int, height: int, fps: float, tag: str):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*tag),
                         float(fps), (width, height))
    if not vw.isOpened():
        raise RuntimeError(f"打不开 {tag} writer：{path}")
    return vw


def _write_times(path: Path, times: list[tuple[int, float, float]]) -> None:
    """每帧时间戳 sidecar：容器 fps 只是标称值，真实节奏看这里。"""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "perf_counter_s", "unix_time_s"])
        for idx, perf, unix in times:
            w.writerow([idx, f"{perf:.6f}", f"{unix:.6f}"])


_PREVIEW_WIN = "record left | right"


def _grab_preview_canvas(left_src, right_src, fov) -> np.ndarray | None:
    """读一对帧、按 fov 归一化、拼成 1280x360 的预览画布；拿不到返回 ``None``。

    抽出来是为了让"等待按键 / 倒计时 / 录制中"三处**共用同一套**归一化逻辑：
    以前只有倒计时那一处做了 fov，录制中直接拿原始帧缩略 —— 一旦 FOV 对齐生效，
    同一场录制里前后两个阶段看到的画面就不一致（一个有裁切一个没有）。
    """
    ok_l, lf, _ = left_src.read()
    ok_r, rf, _ = right_src.read()
    if not (ok_l and ok_r and lf is not None and rf is not None):
        return None
    if fov is not None:
        lf = fov.apply_side(lf, "left")
        rf = fov.apply_side(rf, "right")
    return np.hstack([
        cv2.resize(lf, (640, 360), interpolation=cv2.INTER_AREA),
        cv2.resize(rf, (640, 360), interpolation=cv2.INTER_AREA)])


def _hud(img: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> np.ndarray:
    """在预览画面上叠 HUD；只影响预览，不写进录制的原始视频。"""
    if not lines:
        return img
    out = img.copy()
    band = min(out.shape[0], 10 + 28 * len(lines))
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (out.shape[1], band), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.55, out, 0.45, 0.0)
    for i, (text, color) in enumerate(lines):
        cv2.putText(out, text, (12, 24 + i * 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.72, color, 2, cv2.LINE_AA)
    return out


def _plot_sync(lb, rb, stat_l: dict, stat_r: dict, path: Path) -> None:
    """把两路亮度曲线和检测到的闪光位置画出来，供人一眼确认。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(lb, label="left", linewidth=1.0)
    ax.plot(rb, label="right", linewidth=1.0)
    for stat, color in ((stat_l, "#ef4444"), (stat_r, "#3b82f6")):
        for idx in stat.get("peak_indices", [])[:12]:
            ax.axvline(int(idx) + 1, color=color, alpha=0.18, linewidth=1)
    ax.set_xlabel("frame")
    ax.set_ylabel("mean brightness")
    ax.set_title(f"flash diagnosis: max|ΔL|={stat_l['max_jump']:.2f}, "
                 f"max|ΔR|={stat_r['max_jump']:.2f}, "
                 f"peaks L/R={stat_l['peak_count']}/{stat_r['peak_count']}")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)



def _copy_calibration(session_dir: Path, calib_base: Path) -> tuple[Path, StereoRig]:
    calib_dir = session_dir / "calib"
    calib_dir.mkdir(parents=True, exist_ok=True)
    npz_src = calib_base.with_suffix(".npz")
    json_src = calib_base.with_suffix(".json")
    if not npz_src.exists():
        raise FileNotFoundError(f"找不到标定文件：{npz_src}")
    npz_dst = calib_dir / npz_src.name
    shutil.copy2(npz_src, npz_dst)
    if json_src.exists():
        shutil.copy2(json_src, calib_dir / json_src.name)
    rig = StereoRig.load(npz_dst)
    return npz_dst, rig


def validate_capture_size(rig: StereoRig, width: int, height: int) -> None:
    """录制分辨率必须与标定内参一致；不一致直接报错，不能静默缩放。"""
    actual = tuple(int(v) for v in getattr(rig, "image_size", ()) or ())
    requested = (int(width), int(height))
    if len(actual) != 2:
        raise ValueError("标定缺少 image_size，无法核对录制分辨率；请重新标定")
    if actual != requested:
        raise ValueError(
            f"标定内参基于 {actual[0]}x{actual[1]}，当前录制请求 {requested[0]}x{requested[1]}；"
            "请用同一分辨率重标，或把录制分辨率改成标定值（不要混用）"
        )


def _load_calib_fov(raw: object) -> FovAlignment | None:
    """标定 ``meta["fov"]`` -> :class:`FovAlignment`；缺失 / None 返回 None。

    **只有** 缺失 / ``None`` 才代表旧标定（按原生画面工作）。``{}`` / ``[]`` /
    非 mapping 等是**损坏的记录**：``FovAlignment.from_dict()`` 会抛
    ``ValueError``，调用方必须按损坏退出 2 —— 绝不能因为“假值”就把损坏当成
    旧标定静默跳过（那样会按原生画面录制，尺度错了却不报错）。
    """
    if raw is None:
        return None
    return FovAlignment.from_dict(raw)                      # type: ignore[arg-type]


def decide_record_fov(mode: str, calib_fov, profile):
    """录制侧把装置级 ``fov_apply`` 落到「这一路要不要裁剪」的唯一决策点。

    纯函数：不读文件、不碰相机；``profile`` 由调用方（``main`` 用
    :func:`stereo.profiles.get_active` 读当前装置档案）给，``calib_fov`` 是
    标定 JSON 顶层的 ``fov``（``dict | None``）。测试能用临时 fixture 直接调它
    而不用起相机（见 ``tests/selftest_stereo_branch.py`` 的判据 8）。

    后果由 :func:`stereo.fov_policy.resolve_fov_apply` 统一判定，这里只转发——
    标定与录制服从的是**同一个装置级开关**，不是两套流程逻辑。
    """
    return fov_policy.resolve_fov_apply(profile, mode=mode, calib_fov=calib_fov)


def angled_ignore_message(fov) -> str:
    """angled 下忽略标定里的 FOV 对齐记录时，给用户看的一句警告（纯函数）。

    90° 装置两路几乎不共视，录制按原始画面写盘；若标定 meta 里却带着真裁剪
    参数，必须明确告知「忽略了它」，否则用户以为录的是归一化画面。

    ``fov=None`` 或 ``fov.active=False``（两路视场本就一致、没有真裁剪）时
    返回空串 —— 没有可忽略的东西就不打扰。判据用 ``SimpleNamespace`` 直接调
    这个函数（真跑，不用源码字符串匹配）。
    """
    if fov is None or not getattr(fov, "active", False):
        return ""
    return "[FOV] 标定记录里带裁剪/缩放参数，但模式为 angled：忽略它。"


def _actual_device(idx: int, devices) -> tuple[str, str, str]:
    """本次**实际打开**的那台设备的 ``(name, identity, port)``。

    ⚠ 记录口径（2026-11-11 修）：以前这里写的是**角色绑定**的名字
    （`_role_spec(reg, args.left)`）。用户手动选的左路因此被记成
    "Integrated IR Camera" —— 而算法用的是另一个编号。**名字不能冒充设备**：
    角色名留在 `role` 字段，设备名按实际打开用的 capture index 回查 PnP 表。

    口径同 `AGENTS.md` §9：`#N` 来自 PnP 枚举，MSMF 下实测与 PnP 位次一致
    （这条是**实测**，不是证明）；查不到就留空，不编造一个名字。
    """
    for d in devices:
        if int(getattr(d, "order", -1)) == int(idx):
            return (str(getattr(d, "name", "") or ""),
                    str(getattr(d, "identity", "") or ""),
                    str(getattr(d, "port", "") or ""))
    return ("", "", "")


def _camera_info(side: str, idx: int, src: CameraSource, args,
                 device: tuple[str, str, str],
                 out_width: int, out_height: int) -> CameraInfo:
    """会话 left/right 记录的是**写盘分辨率**（FOV 生效时 = fov.target_size）。

    ``device`` 是**实际打开设备**的 ``(name, identity, port)``，不是角色绑定。
    """
    name, identity, port = device
    return CameraInfo(
        role=side,
        video=f"{side}.{_container_of(args)}",
        width=int(out_width),
        height=int(out_height),
        frame_count=0,
        fps_requested=float(args.fps),
        fps_measured=0.0,
        backend=str(src.info.backend or ""),
        identity=identity,
        name=name,
        port=port,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="双路无损录制（HFYU/FFV1 MKV + session.json）")
    ap.add_argument("--left", default="left", help="左路：角色名 / 编号 / name:...")
    ap.add_argument("--right", default="right", help="右路：角色名 / 编号 / name:...")
    ap.add_argument("--backend", choices=("auto", "msmf", "dshow"), default="auto")
    ap.add_argument(
        "--capture-api", choices=("auto", "mf", "opencv"), default="opencv",
        help="采集后端：opencv=默认，原有编号路径；"
             "mf=按设备 symlink 打开（必须绑定角色）；"
             "auto=角色有 identity/port 时先 MF、失败回退 OpenCV。"
             "显式数字编号只能用 opencv。")
    ap.add_argument("--width", type=int, default=None,
                    help=f"录制采集宽度。**默认不填 = 采用标定记录的采集档 "
                         f"(capture_size)**；标定没记（旧标定）则退回 {RECORD_DEFAULT.label}。"
                         "录制必须与标定同一档，否则写盘尺寸与标定 image_size 对不上")
    ap.add_argument("--height", type=int, default=None,
                    help=f"录制采集高度。默认不填 = 采用标定记录的采集档；"
                         f"标定没记则退回 {RECORD_DEFAULT.label}（说明见 --width）")
    ap.add_argument("--fps", type=fps_type, default=DEFAULT_FPS,
                    help="请求相机的帧率（提示值，可填小数，如 29.97），**不是**容器"
                         "标称帧率；容器帧率取相机协商值（取不到才退回本值）。默认 24")
    ap.add_argument("--codec", choices=tuple(CODECS), default="hfyuv",
                    help="无损编码器：hfyuv（默认，FOV 对齐后 1718x972@30 稳定）/ "
                         "ffv1（更通用但很慢）")
    ap.add_argument("--container", choices=("mkv", "avi"), default=None,
                    help="录制容器：mkv（默认，跟随编码器）/ avi。默认 None = 跟随 "
                         "--codec 的默认扩展名（两种编码器默认都是 mkv）。"
                         "**不支持 mp4**：HFYU/FFV1 在 mp4 里会被 FFmpeg 静默换成 "
                         "h264（有损），isOpened() 还会假成功，等于偷偷破坏无损承诺。"
                         "AVI 有经典单文件 2 GiB 上限，长录制建议用 mkv")
    ap.add_argument("--duration", type=float, default=10.0, help="录制秒数")
    ap.add_argument("--countdown", type=float, default=3.0)
    # 新会话一律写到 `data/stereo/`（这是**写**路径，所以只认新根，不扫旧位置）
    ap.add_argument("--out", default=str(STEREO_ROOT))
    ap.add_argument("--calib", default=None,
                    help="标定文件前缀；缺省从激活装置档案解析，再回落 "
                         "config/stereo_calibration")
    ap.add_argument("--allow-forced", action="store_true",
                    help="⚠ 仅链路测试（默认拒绝）：允许用「强制保存的不可用标定」"
                         "（quality.forced_unusable）录制。录出来的 session.json "
                         "会标「仅测试」，不得用于数据结论")
    ap.add_argument("--baseline-mm", type=float, default=None,
                    help="物理基线（只作核对；不填则会话里不记）")
    ap.add_argument("--label", default="")
    ap.add_argument("--notes", default="")
    ap.add_argument("--stereo-mode", choices=("auto", "parallel", "angled"),
                    default="auto",
                    help="写入会话的双目模式；auto=按标定 meta 推断，缺省 parallel")
    ap.add_argument("--preview", action="store_true", help="显示两路预览（会略增负载）")
    ap.add_argument("--wait-key", choices=("none", "space"), default="none",
                    help="按【空格】才开始倒计时（默认 none = 立刻倒计时）。"
                         "需要 --preview：没有预览窗口就收不到按键。"
                         "为什么要有它：倒计时一开始就只剩几秒，而程序刚启动时"
                         "相机可能还没出画面、窗口也可能还没抢到焦点 —— "
                         "先让人看清两路都有图、球在视野里，再按键开始")
    ap.add_argument("--self-check-only", action="store_true",
                    help="只做编码器写/读逐位自检，不开相机（无相机也能验）")
    args = ap.parse_args()
    # 标定路径：显式 --calib 优先，否则跟随激活装置档案，再回落顶层。
    # 2026-11-04 前这里硬编码顶层，档案"切换装置"是假动作。
    # **必须在这里（最早处）解析**：下面的"采集档"分支用 args.calib 读
    # capture_size，若这里不先把 None 换成真实路径，--self-check-only 也会
    # 在 Path(None) 上崩掉（2026-11-04 实测）。档案缺标定必须**显式报错**：
    # 不能拿顶层那份（别套装置的几何）冒充，否则录出来也没法三维重建。
    if args.calib:
        calib_base = Path(args.calib)
    else:
        try:
            calib_base = resolve_calibration_base()
        except ProfileError as exc:
            raise SystemExit(f"[错误] {exc}") from exc

    # 实际容器名（不含点）：显式 --container 优先，否则跟随 codec 默认（mkv）。
    # 所有写盘路径都从它派生，绝不写死扩展名。
    container = _container_of(args)
    if container == "avi":
        # 只提示、不阻断：AVI 经典单文件 2 GiB 上限，本机 1080p/10s 单路中位约
        # 758 MB（两路 ~1.5 GB/10s）。OpenDML 是否生效未实测，不当成已验证。
        print("[警告] AVI 有经典单文件 2 GiB 上限（本机 1080p/10s 单路中位约 758 MB）；"
              "长录制建议改用 mkv（OpenDML 是否生效未实测，不当成已验证）。")

    # `--wait-key space` 没有预览窗口就收不到按键 —— 这是**参数层的矛盾**，
    # 必须在开相机之前判死：等两台相机都打开、两个 writer 都建好再说"没法等按键"，
    # 用户白等一轮，而且日志里那句错误离"我按了开始录制"隔了好几屏。
    if args.wait_key == "space" and not args.preview:
        print("[错误] --wait-key space 需要 --preview：没有预览窗口就收不到按键。"
              "去掉 --wait-key，或同时开 --preview。")
        return 2

    # ---- 采集档：未显式给 --width/--height 时，**优先用标定记下的采集档** ----
    #
    # 为什么必须这样（2026-10-31）：录制必须与标定**同一档** —— 写盘尺寸、
    # FOV 裁剪区、`validate_capture_size` 都建立在"采集帧 == 标定采集档"上。
    # 而标定档现在可能是 1920x1080（相机实际能力），不再是名义默认 2560x1440。
    # 若还按名义默认请求 1440，`_open_pair` 的「实际 == 请求」校验会直接失败。
    # 标定没记该字段（旧标定）时**明确说清**、退回 RECORD_DEFAULT，不静默假定。
    if args.width is None or args.height is None:
        _cs = calib_capture_size(calib_base)
        if _cs is not None:
            args.width, args.height = _cs
            print(f"[采集档] 采用标定记录的采集档 {_cs[0]}x{_cs[1]}"
                  f"（{calib_base.with_suffix('.json').name} 的 capture_size）")
        else:
            # 标定里没有采集档（旧标定，或这台装置还没标过）→ 问**装置档案**。
            # 这是档案里 width/height 唯一真正有用的场景：新装置第一次录，
            # 还没有标定可依，档案就是它该用的档。比名义默认可靠得多 ——
            # RECORD_DEFAULT 是 2560x1440，而本机实测双路同时开 2560 会让
            # 一路的 MSMF 会话失效（2026-11-05；1920x1080 两台 15/15 稳定）。
            _prof = None
            try:
                from stereo.profiles import get_active
                _prof = get_active()
            except Exception:                            # noqa: BLE001
                _prof = None
            if _prof is not None and _prof.width and _prof.height:
                args.width, args.height = int(_prof.width), int(_prof.height)
                print(f"[采集档] 标定里没有记录，采用装置档案「{_prof.name}」"
                      f"的 {args.width}x{args.height}")
            else:
                args.width, args.height = RECORD_DEFAULT.width, RECORD_DEFAULT.height
                print(f"[采集档] 标定与装置档案都没有采集档，退回默认 "
                      f"{RECORD_DEFAULT.label}。")
                print("         双路同时开 2560x1440 会让其中一路失效；请在装置档案里"
                      "填采集档，或显式给 --width/--height。")

    log_path = setup_logging("record")
    print(f"本次日志：{log_path}")
    LOG.info("=== 双路录制开始 ===")
    LOG.info("参数：left=%s right=%s backend=%s %dx%d@%g duration=%.1f",
             args.left, args.right, args.backend, args.width, args.height,
             args.fps, args.duration)

    if args.self_check_only:
        tag, _ext, label = CODECS[args.codec]
        ok, msg = _lossless_selfcheck(args.width, args.height, args.fps, tag, container)
        print(f"[自检] {label}：{msg}")
        LOG.info("%s/%s 自检（仅自检）：%s（%s）", tag, container,
                 "PASS" if ok else "FAIL", msg)
        return 0 if ok else 1

    tag, _ext, label = CODECS[args.codec]
    print(f"[编码] {label} → {container.upper()}")

    # ---- 带宽可行性：**只提示，不拦截** ----------------------------------
    # 高分辨率 × 高帧率下单路 BGR24 裸流就可能超过一条 USB3 链路的有效带宽。
    # 那种情况下的表现不是报错，而是**相机悄悄降帧**（驱动层丢帧，程序侧
    # 看到的只是 fps 变低）。不说一句，用户会以为"程序没跑到 500 fps"。
    # 注意：这里不改变任何判定、不阻塞、不影响退出码 —— 实际可用帧率只有
    # 相机自己知道，真要量得用 camera_quality.py。
    # 判**单路**：两台相机通常各占一个 USB 口；按两路求和会把正常配置也报成告警。
    _frame_mb, _per_cam_gbps, _total_gbps = stream_bandwidth(
        args.width, args.height, args.fps)
    LOG.info("两路 %dx%d@%g BGR24 裸流：单帧 %.1f MB，单路 %.2f GB/s，"
             "两路合计 %.2f GB/s（未压缩；实际以驱动给到的为准）",
             args.width, args.height, args.fps, _frame_mb,
             _per_cam_gbps, _total_gbps)
    if _per_cam_gbps > USB3_EFFECTIVE_GBPS:
        LOG.warning("单路 %.2f GB/s 已超出常见 USB3 相机链路约 %.1f GB/s 的有效带宽，"
                    "实际帧率会被硬件截断到更低值；程序侧不会因此报错。"
                    "请用 --fps 降到实测可行值（可先跑 apps/tools/camera_quality.py "
                    "量真实吞吐）。两路若共用同一个 USB 控制器还会更早触顶。",
                    _per_cam_gbps, USB3_EFFECTIVE_GBPS)

    if not calib_base.with_suffix(".npz").exists():
        print(f"[错误] 找不到标定：{calib_base.with_suffix('.npz')}")
        return 1

    # ---- 开相机之前先判定"这个组合能不能录" ----
    # 这个判定只依赖标定文件与 --stereo-mode/--width/--height，**不需要相机**。
    # 放在这里是因为 `_open_pair` 会真的申请两台相机：2026-10-28 用 parallel
    # 标定按 angled 录，相机白开一次、会话目录白建一个，才在 validate_capture_size
    # 撞到一行只讲现象不讲出路的报错。判定前移后，代价从"开相机 + 建会话"
    # 降到"读一个 npz"。
    # 读失败（文件损坏等）不在这里报：后面的 `_copy_calibration` 与 FOV 分支
    # 文案更具体，这里抢着报会把它盖掉。
    try:
        _pre = plan_record(StereoRig.load(calib_base),
                           mode_choice=args.stereo_mode,
                           width=args.width, height=args.height,
                           allow_forced=bool(getattr(args, "allow_forced", False)))
    except Exception:                                       # noqa: BLE001
        _pre = None
    if _pre is not None and _pre.ok and getattr(_pre, "forced", False):
        print("\n" + "!" * 66)
        print("⚠ 仅测试：本次录制使用「强制保存的不可用标定」（forced_unusable）")
        print("  录出来的会话只能做链路测试，不得用于数据结论；session.json 会标「仅测试」。")
        print("!" * 66 + "\n")
        LOG.warning("仅测试录制：--allow-forced，标定被标为 forced_unusable")
    if _pre is not None and not _pre.ok:
        print(f"[错误] {_pre.headline}")
        for _line in _pre.detail:
            print(f"       · {_line}")
        for _i, _fix in enumerate(_pre.fixes, 1):
            print(f"       {_i}) {_fix}")
        LOG.error("录制参数不成立，未开相机：%s", _pre.headline)
        return 2

    cams, code = _open_pair(args)
    if code != 0:
        return code
    left_idx, left_src = cams["left"]
    right_idx, right_src = cams["right"]

    try:
        ok, msg = _lossless_selfcheck(args.width, args.height, args.fps, tag, container)
        print(f"[自检] {msg}")
        LOG.info("%s/%s 自检：%s（%s）", tag, container,
                 "PASS" if ok else "FAIL", msg)
        if not ok:
            return 1

        session_dir, session = create_session(
            args.out, label=args.label, notes=args.notes)
        if not (_looks_like_role(args.left) and _looks_like_role(args.right)):
            suffix = (f"backend={args.backend} left_index={left_idx} "
                      f"right_index={right_idx}（显式编号；编号映射已由使用者确认）")
            session.notes = f"{session.notes}；{suffix}" if session.notes else suffix
        if _pre is not None and getattr(_pre, "forced", False):
            # session.json 必须**自己**能证明"这次只是链路测试"。
            mark = ("⚠ 仅测试：使用强制保存的不可用标定（forced_unusable），"
                    "只能做链路测试，不得用于数据结论")
            session.notes = f"{session.notes}；{mark}" if session.notes else mark
        print(f"[会话] {session_dir}")
        LOG.info("会话目录：%s", session_dir)

        try:
            calib_path, rig = _copy_calibration(session_dir, calib_base)
        except Exception as exc:                            # noqa: BLE001
            print(f"[错误] 标定快照失败：{exc}")
            return 1

        # ---- 双目模式：入口层唯一解析点（核心层不接受 auto）----
        # 标定 meta 是**平铺**的（StereoRig.save 直接写 meta），整份传进去。
        mode = resolve_auto(args.stereo_mode, calibration=rig.meta)
        LOG.info("双目模式：%s（--stereo-mode %s）", mode, args.stereo_mode)

        # 新标定在 meta 里带 FOV 对齐参数；**只有** 缺失 / None 才是旧标定 ->
        # fov=None，行为不变。{} / [] / 非 mapping 是损坏记录，必须像 verify
        # 一样退出 2：不能把“解析不出来”静默当成“没有 FOV”。
        raw_fov = rig.meta.get("fov") if isinstance(rig.meta, dict) else None
        try:
            fov = _load_calib_fov(raw_fov)
        except ValueError as exc:
            print(f"[错误] 标定 meta.fov 无法解析：{exc}")
            print("       旧标定本来没有该字段；这份标定的 FOV 记录损坏，"
                  "请重新标定后再录。")
            LOG.error("标定 meta.fov 解析失败，拒绝录制：%s", exc)
            return 2

        # ---- 装置级 fov_apply：录制是否把这份 FOV 应用到两路画面（2026-11-15）----
        # 「启用」= 应用到当前摄像头，是装置级事实，标定与录制都服从它。
        # 这是唯一判定点 resolve_fov_apply 在录制侧的落法：
        #   - error         -> 明确拒绝（return 2），**不静默回退**；
        #   - legacy_follow -> 开关关但标定自带 FOV：沿用标定裁剪 + 醒目提示
        #                      （内参本就是裁剪帧上算的，跟着裁才自洽；旧流程行为不变）；
        #   - 不应用        -> fov=None（标定也没 fov 时按原始画面写盘）；
        #   - 应用          -> 继续用标定 JSON 那份（第 6 条已保证它与档案一致）。
        # 为什么 error 不静默回退成「照旧录制」：FOV 改变的是喂给标定的像素；
        # 用「关 FOV 标出来的内参」去解「开 FOV 录出来的画面」，深度会系统性错，
        # 且屏幕上完全看不出来 —— 这是本项目最贵的那类静默错配，宁可拒绝录制，
        # 也不冒充成功（回退只会让内参与画面口径不一致，错得更隐蔽）。
        try:
            from stereo.profiles import get_active
            _active_prof = get_active()
        except Exception:                                   # noqa: BLE001
            _active_prof = None
        _dec = decide_record_fov(mode, raw_fov, _active_prof)
        if _dec["error"]:
            print(f"[FOV] {_dec['error']}")
            LOG.error("FOV 应用判定拒绝录制：%s", _dec["error"])
            return 2
        if _dec.get("legacy_follow"):
            # 开关关、但标定 JSON 自带 FOV：那次标定的内参是在裁剪帧上算的，
            # 录制必须跟着裁才几何自洽 —— 沿用标定自带的 FOV（旧流程行为不变），
            # 只给一句醒目提示。**不置 fov=None**：fov 保持从标定 JSON 读出来的
            # 原值，照旧裁剪/缩放，否则内参与画面口径错配、且屏幕上完全看不出来。
            print(f"[FOV] {_dec['warning']}")
            LOG.warning("FOV 开关未启用，沿用标定自带的 FOV：%s", _dec["warning"])
        elif not _dec["apply"]:
            if mode != "angled":
                # 关掉开关且标定也没 fov：按原始画面写盘。
                fov = None
                print(f"[FOV] {_dec['reason']}")
                LOG.info("FOV 未应用：%s", _dec["reason"])
            # angled：保持 fov 原值，交给下面原有的 angled 分支自己提示并置 None。
            # 决策块一行都不打，避免与那句「忽略」警告重复；若在这里提前把 fov
            # 置成 None，angled 分支的 `if fov is not None and fov.active` 永远为假，
            # 用户拿带裁剪参数的标定去录 90° 就再也收不到「我忽略了它」的提示。
        else:
            print("[FOV] 已应用装置档案的 FOV 对齐：录制按它裁剪/缩放两路。")
            LOG.info("FOV 已应用：%s", _dec["fov"])
        # angled：两路光轴近乎垂直、视场几乎不重叠，单应裁剪没有物理意义。
        # 写盘就是两路各自的原始画面（坐标系仍是 left_camera），所以不用
        # fov.target_size，也不再拿它去核对 image_size。
        if mode == "angled":
            _ignore_msg = angled_ignore_message(fov)
            if _ignore_msg:
                print(_ignore_msg)
                LOG.warning("angled 模式忽略标定 meta.fov=%s", fov.to_dict())
            fov = None
        # 写盘尺寸只由 `stereo.record_plan.output_size` 决定：体检与录制必须先
        # 算出**同一个数**，否则"界面说能录、后端说不能"就是必然的。
        out_w, out_h = output_size(mode, fov, args.width, args.height)
        # 依据是 **1440p 全尺寸（3.68MP）** 的历史实测（约 24.07fps，242/300 帧）；
        # 现在 FOV 对齐后写盘只有 1718x972（1.67MP，约 45% 像素），30fps 稳定
        # （2026-10-24：301 帧/10.002s）。只有写盘像素数真的很大时才提示。
        if out_w * out_h > 2_500_000 and args.fps > 24:
            print(f"[警告] 写盘 {out_w}x{out_h}（{out_w * out_h / 1e6:.2f}MP）"
                  f"请求 {args.fps:g}fps：这是按 1440p 全尺寸（3.68MP）历史实测"
                  "（约 24.07fps，242/300 帧·10 秒·HFYU）推断的风险，"
                  "当前尺寸未实测；建议先短录确认不丢帧。")
            LOG.warning("写盘 %sx%s（%.2fMP）请求 %gfps：依据 1440p 全尺寸的历史"
                        "实测（约 24.07fps）提示可能丢帧；当前尺寸未实测",
                        out_w, out_h, out_w * out_h / 1e6, args.fps)
        try:
            validate_capture_size(rig, out_w, out_h)
        except ValueError as exc:
            print(f"[错误] {exc}")
            if mode == "angled" and raw_fov:
                print("       这份 angled 标定带了 FOV 对齐记录（内参基于裁剪后尺寸），"
                      "但 angled 录制按原始画面写盘；请重新标定（不强制 --fov），"
                      "或改用与内参一致的写盘尺寸。")
            return 2
        if mode == "angled":
            lw, lh = int(left_src.info.width), int(left_src.info.height)
            rw, rh = int(right_src.info.width), int(right_src.info.height)
            print(f"[FOV] 未做 FOV 对齐（angled）：写盘为两路原始画面；"
                  f"左 {lw}x{lh} / 右 {rw}x{rh}")
            LOG.info("angled：跳过 FOV 对齐，写盘原始画面 左 %sx%s / 右 %sx%s",
                     lw, lh, rw, rh)
        elif fov is not None:
            if fov.active and tuple(fov.target_size) != tuple(rig.image_size):
                print(f"[错误] 标定 meta.fov.target_size {fov.target_size} 与 "
                      f"image_size {rig.image_size} 不一致，拒绝录制。")
                print("       这份标定的 FOV 记录与内参不匹配，请重新标定。")
                LOG.error("fov.target_size %s 与 rig.image_size %s 不一致，拒绝录制",
                          fov.target_size, rig.image_size)
                return 2
            if fov.active:
                # crop_rect 在**广角路原生采集帧**的像素系里，不是 target_size。
                # 预检：必须落在本次相机原生采集帧内；apply_side() 也会报，但那时
                # writer 已打开、会话已建。这里提前拒绝，不产出半截会话。
                cap_w, cap_h = int(args.width), int(args.height)
                cx, cy, cw, ch = fov.crop_rect
                if cx < 0 or cy < 0 or cx + cw > cap_w or cy + ch > cap_h:
                    print(f"[错误] 标定 meta.fov.crop_rect {fov.crop_rect} 超出"
                          f"采集帧 {cap_w}x{cap_h}（裁剪路 {fov.crop_side}），"
                          "拒绝录制。")
                    print("       这份标定的 FOV 裁剪区与当前分辨率不匹配，"
                          "请重新标定。")
                    LOG.error("fov.crop_rect %s 超出采集帧 %sx%s，拒绝录制",
                              fov.crop_rect, cap_w, cap_h)
                    return 2
                if out_w > cw or out_h > ch:
                    print(f"[错误] 标定 meta.fov.target_size {fov.target_size} 大于"
                          f"裁剪区 {(cw, ch)}：该记录要求放大较广一路。")
                    print("       放大会压低该路清晰度（旧版本的做法）；请重新标定，"
                          "让 target_size 取共同（不放大）尺寸。")
                    LOG.error("fov.target_size %s 大于裁剪区 %s，拒绝录制"
                              "（拒绝会放大的旧记录）", fov.target_size, (cw, ch))
                    return 2
                if out_w % 2 or out_h % 2:
                    print(f"[错误] 标定 meta.fov.target_size {fov.target_size} 含"
                          "奇数宽高：无损 MKV 只支持偶数。")
                    print("       奇数尺寸写盘后会被取整，视频与 session / 内参对不上；"
                          "请重新标定。")
                    LOG.error("fov.target_size %s 含奇数宽高，拒绝录制",
                              fov.target_size)
                    return 2
                # 沿用标定里的 FOV 时只警告、不中止：画面已经录下来还能救，
                # 但必须让人知道这次写盘可能被错误裁剪。阈值与标定端同源
                # （stereo/fov_align.py），不在这里另写一份。
                if fov.quality < MIN_FOV_QUALITY or fov.inliers < MIN_FOV_INLIERS:
                    print(f"[警告] 沿用的 FOV 对齐置信度低（{fov_confidence_problem(fov)}）："
                          "写盘画面可能被错误裁剪；本次仍按该参数录制。")
                    print("       建议换有纹理的场景重新标定，或改按原始画面录制。")
                    LOG.warning("沿用低置信度 FOV：%s；%s",
                                fov_confidence_problem(fov), fov.to_dict())
                print(f"[FOV] 沿用标定对齐：参考 {fov.reference_side} / "
                      f"裁剪 {fov.crop_side} / 裁剪区 {fov.crop_rect} -> "
                      f"{out_w}x{out_h}（采集 {cap_w}x{cap_h}，不放大）")
                print(f"      scale={fov.scale:.4f}  inliers={fov.inliers}  "
                      f"quality={fov.quality:.3f}  （平行基线参照：0.6649 / 296 / 0.513）")
                LOG.info("沿用标定 FOV 对齐：%s；写盘 %sx%s（采集 %sx%s）",
                         fov.to_dict(), out_w, out_h, cap_w, cap_h)
            else:
                print("[FOV] 标定记录两路视场一致（crop_side=none），不裁剪")
        else:
            # `fov: null` 是**新标定明确记录**的"本次没有对齐"（parallel 下估计
            # 失败后降级），与"旧标定根本没这个字段"要分开说：两者都按原始画面
            # 写盘，但前者意味着两路像素尺度可能不一致。
            if isinstance(rig.meta, dict) and "fov" in rig.meta and raw_fov is None:
                print("[FOV] 本次无对齐（标定未产生 FOV）：两路各自原始画面")
                LOG.info("标定 meta.fov=null：按两路原始画面录制")
            else:
                print("[FOV] 旧标定没有 fov 记录：按原生画面录制（行为与以前一致）")

        session.calibration = CalibrationInfo(
            path=str(calib_path.relative_to(session_dir)).replace("\\", "/"),
            sha256=sha256_file(calib_path),
            image_size=[int(rig.image_size[0]), int(rig.image_size[1])],
            baseline_mm=float(rig.baseline_mm),
            baseline_measured_mm=args.baseline_mm,
            created=str(rig.meta.get("created") or ""),
            verdict=str((rig.quality or {}).get("verdict") or ""),
        )
        # mode 已在加载标定后由 resolve_auto() 落定，这里不再分叉解析。
        session.stereo_mode = mode
        session.rig = {
            "stereo_mode": mode,
            "angle_deg": float(optical_axes_angle_deg(rig.R)),
            "baseline_mm": float(rig.baseline_mm),
        }
        # 只在真的发生裁剪/缩放时写入，旧会话 schema 与旧标定记录的字段不变。
        if fov is not None and fov.active:
            session.rig["fov"] = fov.to_dict()
        if mode != rig.stereo_mode:
            print(f"[警告] 会话模式 {mode} 与标定 meta {rig.stereo_mode} 不一致；"
                  "请确认这份标定确实对应当前机械装置。")
            LOG.warning("会话 stereo_mode=%s，标定 meta=%s", mode, rig.stereo_mode)
        session.codec = tag
        session.container = container
        session.lossless = True

        left_path = session_dir / f"left.{container}"
        right_path = session_dir / f"right.{container}"
        # 容器标称帧率取相机协商值（`--fps` 只是请求提示）。两路容器头必须一致，
        # 所以取两路协商值里较小的那个；都不合法才退回请求值。
        negotiated = [float(left_src.info.fps), float(right_src.info.fps)]
        usable = [v for v in negotiated if math.isfinite(v) and v >= 1.0]
        container_fps = _container_fps(min(usable) if usable else float("nan"), args.fps)
        fps_src = (f"相机协商 L/R={negotiated[0]:.1f}/{negotiated[1]:.1f}"
                   if usable else "相机未报出有效帧率")
        print(f"[容器] 标称帧率 {container_fps:g} fps（{fps_src}；请求 --fps "
              f"{args.fps:g}）")
        LOG.info("容器标称帧率 %.3f fps（%s；请求 %g fps）",
                 container_fps, fps_src, args.fps)
        try:
            left_writer = _make_writer(left_path, out_w, out_h, container_fps, tag)
            right_writer = _make_writer(right_path, out_w, out_h, container_fps, tag)
        except Exception as exc:                            # noqa: BLE001
            print(f"[错误] {exc}")
            return 1

        start_evt = threading.Event()
        stop_evt = threading.Event()
        preview: dict = {}
        lock = threading.Lock()
        left_rec = _SideRecorder("left", left_src, left_writer, start_evt, stop_evt,
                                 preview, lock, fov=fov)
        right_rec = _SideRecorder("right", right_src, right_writer, start_evt, stop_evt,
                                  preview, lock, fov=fov)

        print()
        print("准备好后看倒计时。录制开始后 **1 秒内闪一次光**（或拍一次手），")
        print("结束前再闪一次；球保持在两路视野内。")
        print(f"本次录 {args.duration:.0f} 秒；按 q / Esc 可提前结束。")

        # ---- 先等【空格】，再进倒计时 ----
        #
        # 为什么要有这一步：倒计时一共只有 `--countdown` 秒，而程序刚起来时
        # 相机常常还没出第一帧、窗口也未必抢到焦点。用户按下「开始录制」后
        # 看到的是一个黑窗 + 已经开始的倒计时，等于把"准备"这段最重要的时间
        # 从人手里抢走了。这一步把控制权交回给人：画面先跑起来，确认两路都有图、
        # 球在视野里，再按空格。
        if args.wait_key == "space":
            print()
            print("预览已开：按【空格】开始倒计时（按 Esc / q 取消）。")
            while True:
                try:
                    canvas = _grab_preview_canvas(left_src, right_src, fov)
                except ValueError as exc:
                    print(f"[错误] 预览 FOV 归一化失败：{exc}")
                    LOG.error("预览 FOV 归一化失败：%s", exc)
                    left_writer.release()
                    right_writer.release()
                    return 2
                if canvas is None:
                    canvas = np.zeros((360, 1280, 3), np.uint8)
                canvas = _hud(canvas, [
                    ("READY - press SPACE to start", (0, 255, 255)),
                    (f"then {max(0, args.countdown):g}s countdown, "
                     f"{args.duration:.0f}s recording", (200, 200, 200)),
                    ("FLASH once within 1s after REC starts", (200, 200, 200)),
                ])
                cv2.imshow(_PREVIEW_WIN, canvas)
                key = cv2.waitKey(30) & 0xFF
                if key == 32:            # 空格
                    break
                if key in (27, ord("q")):
                    print("[取消] 没有开始录制。")
                    left_writer.release()
                    right_writer.release()
                    cv2.destroyWindow(_PREVIEW_WIN)
                    return 0
                # 点了窗口右上角的叉：不处理的话会在这儿空转，且没有任何提示
                try:
                    if cv2.getWindowProperty(_PREVIEW_WIN, cv2.WND_PROP_VISIBLE) < 1:
                        print("[取消] 预览窗口被关掉，没有开始录制。")
                        left_writer.release()
                        right_writer.release()
                        return 0
                except cv2.error:
                    pass

        for i in range(int(max(0, args.countdown)), 0, -1):
            print(f"  {i}…", end="", flush=True)
            t_end = time.perf_counter() + 1.0
            while time.perf_counter() < t_end:
                if args.preview:
                    try:
                        canvas = _grab_preview_canvas(left_src, right_src, fov)
                    except ValueError as exc:
                        print(f"[错误] 倒计时预览 FOV 归一化失败：{exc}")
                        LOG.error("倒计时预览 FOV 归一化失败：%s", exc)
                        left_writer.release()
                        right_writer.release()
                        return 2
                    if canvas is None:
                        canvas = np.zeros((360, 1280, 3), np.uint8)
                    canvas = _hud(canvas, [
                        (f"READY  {max(0.0, t_end - time.perf_counter()):.1f}s", (0, 255, 255)),
                        ("FLASH NOW after the beep; flash again before the end", (200, 200, 200)),
                    ])
                    cv2.imshow(_PREVIEW_WIN, canvas)
                    cv2.waitKey(1)
                else:
                    time.sleep(0.02)
            print()
        print(" 开始！现在闪光！", flush=True)
        try:
            import winsound
            winsound.Beep(1200, 200)
        except Exception:                                   # noqa: BLE001
            pass

        left_rec.start()
        right_rec.start()
        start_evt.set()
        t0 = time.perf_counter()
        flash_seen = False
        try:
            while time.perf_counter() - t0 < args.duration:
                if args.preview:
                    with lock:
                        lf = preview.get("left")
                        rf = preview.get("right")
                    if lf is not None and rf is not None:
                        thumb_l = cv2.resize(lf, (640, 360), interpolation=cv2.INTER_AREA)
                        thumb_r = cv2.resize(rf, (640, 360), interpolation=cv2.INTER_AREA)
                        elapsed = time.perf_counter() - t0
                        remain = max(0.0, args.duration - elapsed)
                        # 实时提示：前 2 秒提示闪，最后 2 秒提示再闪
                        if len(left_rec.brightness) >= 2:
                            jump = abs(left_rec.brightness[-1] - left_rec.brightness[-2])
                            if jump >= 2.0:
                                flash_seen = True
                        lines = [
                            (f"REC {elapsed:.1f}/{args.duration:.1f}s   "
                             f"L {left_rec.frame_count}  R {right_rec.frame_count}",
                             (0, 255, 0)),
                        ]
                        if elapsed < 2.0:
                            lines.append(("FLASH NOW!", (0, 0, 255)))
                        elif remain < 2.0:
                            lines.append(("FLASH AGAIN BEFORE END", (0, 215, 255)))
                        if flash_seen:
                            lines.append(("FLASH DETECTED", (0, 255, 0)))
                        cv2.imshow("record left | right",
                                   _hud(np.hstack([thumb_l, thumb_r]), lines))
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
                else:
                    time.sleep(0.02)
        finally:
            stop_evt.set()
            left_rec.join(timeout=5.0)
            right_rec.join(timeout=5.0)
            if left_rec.is_alive() or right_rec.is_alive():
                print("[警告] 采集线程 5 秒内没退出，强制 release 相机")
                left_src.release()
                right_src.release()
                left_rec.join(timeout=2.0)
                right_rec.join(timeout=2.0)
            left_writer.release()
            right_writer.release()
            if args.preview:
                cv2.destroyAllWindows()
            elapsed = max(1e-6, time.perf_counter() - t0)

        if left_rec.error or right_rec.error:
            print(f"[错误] 采集线程失败：left={left_rec.error or 'ok'} "
                  f"right={right_rec.error or 'ok'}")
            # 不直接 return：文件仍然保留，session 也要写出来，便于查
        left_fps = left_rec.frame_count / elapsed
        right_fps = right_rec.frame_count / elapsed

        session.left = _camera_info("left", left_idx, left_src, args,
                                    _actual_device(left_idx, list_devices()),
                                    out_w, out_h)
        session.right = _camera_info("right", right_idx, right_src, args,
                                      _actual_device(right_idx, list_devices()),
                                      out_w, out_h)
        session.left.frame_count = left_rec.frame_count
        session.right.frame_count = right_rec.frame_count
        session.left.fps_measured = left_fps
        session.right.fps_measured = right_fps
        _write_times(session_dir / "left_times.csv", left_rec.times)
        _write_times(session_dir / "right_times.csv", right_rec.times)

        # 同步：先诊断"有没有拍到闪光"，再谈偏移
        sync_note = ""
        mean_l = mean_r = 0.0
        try:
            lb = np.asarray(left_rec.brightness, dtype=np.float64)
            rb = np.asarray(right_rec.brightness, dtype=np.float64)
            np.save(session_dir / "derived" / "left_brightness.npy", lb)
            np.save(session_dir / "derived" / "right_brightness.npy", rb)
            mean_l, mean_r = float(lb.mean()), float(rb.mean())
            print(f"[画面] 平均亮度 左={mean_l:.1f} 右={mean_r:.1f}")
            if min(mean_l, mean_r) < 20.0:
                print("[警告] 画面几乎全黑：检查镜头盖 / 灯光 / 曝光；这条数据不能做检测")
            stat_l = brightness_jumps(lb)
            stat_r = brightness_jumps(rb)
            max_jump = max(stat_l["max_jump"], stat_r["max_jump"])
            print(f"[同步诊断] 最大亮度跳变 左={stat_l['max_jump']:.2f} "
                  f"右={stat_r['max_jump']:.2f}；明显事件 左={stat_l['peak_count']} "
                  f"右={stat_r['peak_count']}；事件阈值≈{stat_l['threshold']:.2f}；"
                  f"闪光门槛={SyncConfig().flash_min_jump:.2f}")
            try:
                _plot_sync(lb, rb, stat_l, stat_r,
                           session_dir / "derived" / "sync_brightness.png")
            except Exception as exc:                        # noqa: BLE001
                print(f"[警告] 同步诊断图画不出来：{exc}")

            # 搜索窗与最小重叠的语义是**时长**（6 s / 1 s），不是固定 180/30 帧：
            # 500 fps 下 180 帧只有 0.36 s，闪光事件几乎必然落在窗外。
            cfg_sync = config_for_fps(left_fps)
            res = find_offset(lb, rb, cfg_sync)
            if res.usable:
                session.sync = SyncInfo(method="flash", offset=int(res.offset),
                                        confidence=float(res.confidence),
                                        note=str(res.note or ""))
                sync_note = (f"flash {res.offset:+d} 帧，置信度 {res.confidence:.3f}，"
                             f"事件 {res.peak_count} 个")
                print(f"[同步] {res}")
            else:
                # 没有闪光时，用录制时逐帧时间戳做自动兜底（近似，不是硬件同步）
                ts = timestamp_offset(
                    [t[1] for t in left_rec.times],
                    [t[1] for t in right_rec.times],
                    fps=left_fps, max_offset=5)
                if ts.usable:
                    session.sync = SyncInfo(
                        method="timestamp", offset=int(ts.offset),
                        confidence=float(ts.confidence), note=ts.note)
                    sync_note = (f"timestamp {ts.offset:+d} 帧，置信度 "
                                 f"{ts.confidence:.3f}（主机时间戳近似）")
                    print(f"[同步] 没有闪光，改用时间戳兜底：{ts.note}")
                else:
                    if max_jump < cfg_sync.flash_min_jump:
                        why = (f"没有足够强的闪光：最大亮度跳变 {max_jump:.2f} < "
                               f"{cfg_sync.flash_min_jump:.2f}。请重录，并在“开始！现在闪光！”后"
                               "立刻闪一次、结束前再闪一次。")
                    else:
                        why = (f"有亮度变化但没形成可用闪光事件（最大跳变 {max_jump:.2f}，"
                               f"事件 {max(stat_l['peak_count'], stat_r['peak_count'])} 个）。")
                    why += f"；时间戳兜底也不可用（中位差 {ts.median_abs_ms:.1f} ms）"
                    session.sync = SyncInfo(method="none", offset=0,
                                            confidence=float(res.confidence),
                                            note=why)
                    sync_note = f"未同步：{why}"
                    print(f"[同步] 未通过：{why}")
        except Exception as exc:                            # noqa: BLE001
            session.sync = SyncInfo(method="none", note=f"同步计算失败：{exc}")
            print(f"[警告] 同步计算失败：{exc}")

        session.checks = [
            f"{msg}",
            f"帧数：left={left_rec.frame_count} right={right_rec.frame_count}",
            f"实测 fps：left={left_fps:.2f} right={right_fps:.2f}",
            f"平均亮度：left={mean_l:.1f} right={mean_r:.1f}",
            "时间戳 sidecar：left_times.csv / right_times.csv",
        ]
        if min(mean_l, mean_r) < 20.0:
            session.checks.append("画面几乎全黑（<20）：检查镜头盖/灯光/曝光")
        if _pre is not None and getattr(_pre, "forced", False):
            session.checks.append("⚠ 仅测试：使用强制保存的不可用标定"
                                  "（forced_unusable），结果不得用于数据结论")
        if sync_note:
            session.checks.append(f"同步：{sync_note}")
        save_session(session, session_dir)
        print()
        print(describe(session))
        print()

        errors, warnings = validate_session(session_dir)
        if errors:
            print("[校验] 有错误：")
            for e in errors:
                print(f"  ✘ {e}")
        if warnings:
            print("[校验] 警告：")
            for w in warnings:
                print(f"  ! {w}")
        if not errors and not warnings:
            print("[校验] 通过（无错误 / 无警告）")
        LOG.info("会话完成：%s；errors=%s warnings=%s", session_dir, errors, warnings)
        return 1 if errors else 0
    finally:
        left_src.release()
        right_src.release()
        LOG.info("=== 双路录制结束 ===")


if __name__ == "__main__":
    raise SystemExit(main())
