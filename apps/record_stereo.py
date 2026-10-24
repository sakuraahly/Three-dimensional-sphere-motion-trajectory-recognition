"""双路无损录制：两台相机各写一个无损 MKV（默认 HFYU），并落 `session.json`。

为什么不是直接用 `PairSource` 录
--------------------------------
`PairSource` 的相机模式是"永远取最新帧"——处理慢就丢帧，这是实时预览要的语义，
但**录制要每一帧都留下**。所以这里自己开两路 `CameraSource`，各跑一个采集线程、
各写一个 writer，主线程只负责计时 / 预览 / 停止。

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

from apps._common import setup_logging                      # noqa: E402
from calib.params import StereoRig                           # noqa: E402
from stereo.fov_align import FovAlignment                    # noqa: E402
from stereo.stereo_mode import (                             # noqa: E402
    infer_mode, normalize_mode, optical_axes_angle_deg,
)
from sphere_tracker.camera_registry import (                 # noqa: E402
    load_registry,
    resolve_camera_arg,
)
from sphere_tracker.sources import (                         # noqa: E402
    INDEX_BACKEND,
    CameraSource,
)
from sphere_tracker.video_profiles import RECORD_DEFAULT      # noqa: E402
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
    find_offset,
    timestamp_offset,
)

LOG = logging.getLogger("record")
# 1080p30 × 2 的实测吞吐（本机）：HFYU ≈ 34 fps/路；FFV1 ≈ 8 fps/路。
# 所以默认 HFYU（无损、能跟上相机），FFV1 留给"不赶时间"的归档。
CODECS = {"hfyuv": ("HFYU", "HuffYUV（无损，默认）"),
          "ffv1": ("FFV1", "FFV1（无损，但本机 2×1080p 只有约 8 fps/路）")}


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
        idx = resolve_camera_arg(value, reg=load_registry(), interactive=False)
        if idx is None:
            raise RuntimeError(f"{side} 没有解析出相机编号")
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
          f"请求 {args.width}x{args.height}@{args.fps} fps")
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


def _lossless_selfcheck(width: int, height: int, fps: int, tag: str) -> tuple[bool, str]:
    """写两帧 -> 读回 -> 逐位比较；编码器不可用就提前失败，别录完才发现。"""
    try:
        fourcc = cv2.VideoWriter_fourcc(*tag)
        with tempfile.TemporaryDirectory(prefix="lossless_check_") as td:
            path = Path(td) / "check.mkv"
            vw = cv2.VideoWriter(str(path), fourcc, float(fps), (width, height))
            if not vw.isOpened():
                return False, f"VideoWriter 打不开 {tag}/MKV"
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


def _make_writer(path: Path, width: int, height: int, fps: int, tag: str):
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


def _camera_info(side: str, idx: int, src: CameraSource, args, role_spec) -> CameraInfo:
    return CameraInfo(
        role=side,
        video=f"{side}.mkv",
        width=int(src.info.width),
        height=int(src.info.height),
        frame_count=0,
        fps_requested=float(args.fps),
        fps_measured=0.0,
        backend=str(src.info.backend or ""),
        identity=(role_spec.identity if role_spec else ""),
        name=(role_spec.name if role_spec else ""),
        port=(role_spec.port if role_spec else ""),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="双路无损录制（HFYU/FFV1 MKV + session.json）")
    ap.add_argument("--left", default="left", help="左路：角色名 / 编号 / name:...")
    ap.add_argument("--right", default="right", help="右路：角色名 / 编号 / name:...")
    ap.add_argument("--backend", choices=("auto", "msmf", "dshow"), default="auto")
    ap.add_argument("--width", type=int, default=RECORD_DEFAULT.width,
                    help=f"录制宽度（默认 {RECORD_DEFAULT.label}）。真机实测：两台外接"
                         "相机无视 1920/1280 请求、恒给 2560x1440；1440p 主循环"
                         "实测约 24.07fps（242/300 帧/路），24 是稳定目标，"
                         "请求 30 会丢帧。长录/其他机型未验证")
    ap.add_argument("--height", type=int, default=RECORD_DEFAULT.height,
                    help=f"录制高度（默认 {RECORD_DEFAULT.label}；说明见 --width）")
    ap.add_argument("--fps", type=int, default=24,
                    help="请求帧率（默认 24；1440p 主循环实测约 24.07fps，30 会丢帧）")
    ap.add_argument("--codec", choices=tuple(CODECS), default="hfyuv",
                    help="无损编码器：hfyuv（默认，1440p@24 能跟上）/ ffv1（更通用但很慢）")
    ap.add_argument("--duration", type=float, default=10.0, help="录制秒数")
    ap.add_argument("--countdown", type=float, default=3.0)
    ap.add_argument("--out", default=str(ROOT / "data" / "sessions"))
    ap.add_argument("--calib", default=str(ROOT / "config" / "stereo_calibration"))
    ap.add_argument("--baseline-mm", type=float, default=None,
                    help="物理基线（只作核对；不填则会话里不记）")
    ap.add_argument("--label", default="")
    ap.add_argument("--notes", default="")
    ap.add_argument("--stereo-mode", choices=("auto", "parallel", "angled"),
                    default="auto",
                    help="写入会话的双目模式；auto=按标定 meta 推断，缺省 parallel")
    ap.add_argument("--preview", action="store_true", help="显示两路预览（会略增负载）")
    ap.add_argument("--self-check-only", action="store_true",
                    help="只做编码器写/读逐位自检，不开相机（无相机也能验）")
    args = ap.parse_args()

    if (args.width, args.height) == (2560, 1440) and args.fps > 24:
        print(f"[警告] 1440p 主循环实测约 24.07fps（242/300 帧·10 秒·HFYU，"
              f"两路一致）——这是目标值；请求 {args.fps}fps 会丢帧。"
              f"建议 --fps 24（默认已 24）。长录/其他机型未验证。")
        LOG.warning("2560x1440 + %dfps：主循环实测约 24.07fps（242/300），"
                    "请求 >24fps 会丢帧；24fps 是稳定目标；长录/其他机型未验证",
                    args.fps)

    log_path = setup_logging("record")
    print(f"本次日志：{log_path}")
    LOG.info("=== 双路录制开始 ===")
    LOG.info("参数：left=%s right=%s backend=%s %dx%d@%d duration=%.1f",
             args.left, args.right, args.backend, args.width, args.height,
             args.fps, args.duration)

    if args.self_check_only:
        tag, label = CODECS[args.codec]
        ok, msg = _lossless_selfcheck(args.width, args.height, args.fps, tag)
        print(f"[自检] {label}：{msg}")
        LOG.info("%s 自检（仅自检）：%s（%s）", tag, "PASS" if ok else "FAIL", msg)
        return 0 if ok else 1

    tag, label = CODECS[args.codec]
    print(f"[编码] {label}")
    calib_base = Path(args.calib)
    if not calib_base.with_suffix(".npz").exists():
        print(f"[错误] 找不到标定：{calib_base.with_suffix('.npz')}")
        return 1

    cams, code = _open_pair(args)
    if code != 0:
        return code
    left_idx, left_src = cams["left"]
    right_idx, right_src = cams["right"]

    try:
        ok, msg = _lossless_selfcheck(args.width, args.height, args.fps, tag)
        print(f"[自检] {msg}")
        LOG.info("%s 自检：%s（%s）", tag, "PASS" if ok else "FAIL", msg)
        if not ok:
            return 1

        session_dir, session = create_session(
            args.out, label=args.label, notes=args.notes)
        if not (_looks_like_role(args.left) and _looks_like_role(args.right)):
            suffix = (f"backend={args.backend} left_index={left_idx} "
                      f"right_index={right_idx}（显式编号；编号映射已由使用者确认）")
            session.notes = f"{session.notes}；{suffix}" if session.notes else suffix
        print(f"[会话] {session_dir}")
        LOG.info("会话目录：%s", session_dir)

        try:
            calib_path, rig = _copy_calibration(session_dir, calib_base)
        except Exception as exc:                            # noqa: BLE001
            print(f"[错误] 标定快照失败：{exc}")
            return 1

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
        out_w, out_h = ((int(fov.target_size[0]), int(fov.target_size[1]))
                        if fov is not None and fov.active
                        else (int(args.width), int(args.height)))
        try:
            validate_capture_size(rig, out_w, out_h)
        except ValueError as exc:
            print(f"[错误] {exc}")
            return 2
        if fov is not None:
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
                print(f"[FOV] 沿用标定对齐：参考 {fov.reference_side} / "
                      f"裁剪 {fov.crop_side} / 裁剪区 {fov.crop_rect} -> "
                      f"{fov.target_size[0]}x{fov.target_size[1]}")
                LOG.info("沿用标定 FOV 对齐：%s", fov.to_dict())
            else:
                print("[FOV] 标定记录两路视场一致（crop_side=none），不裁剪")
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
        if args.stereo_mode == "auto":
            mode = infer_mode(calibration={"meta": rig.meta})
        else:
            mode = normalize_mode(args.stereo_mode)
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
        session.container = "mkv"
        session.lossless = True

        left_path = session_dir / "left.mkv"
        right_path = session_dir / "right.mkv"
        try:
            left_writer = _make_writer(left_path, out_w, out_h, args.fps, tag)
            right_writer = _make_writer(right_path, out_w, out_h, args.fps, tag)
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
        for i in range(int(max(0, args.countdown)), 0, -1):
            print(f"  {i}…", end="", flush=True)
            t_end = time.perf_counter() + 1.0
            while time.perf_counter() < t_end:
                if args.preview:
                    ok_l, lf, _ = left_src.read()
                    ok_r, rf, _ = right_src.read()
                    if ok_l and ok_r and lf is not None and rf is not None:
                        if fov is not None:
                            try:
                                lf = fov.apply_side(lf, "left")
                                rf = fov.apply_side(rf, "right")
                            except ValueError as exc:
                                print(f"[错误] 倒计时预览 FOV 归一化失败：{exc}")
                                LOG.error("倒计时预览 FOV 归一化失败：%s", exc)
                                left_writer.release()
                                right_writer.release()
                                return 2
                        canvas = np.hstack([
                            cv2.resize(lf, (640, 360), interpolation=cv2.INTER_AREA),
                            cv2.resize(rf, (640, 360), interpolation=cv2.INTER_AREA)])
                    else:
                        canvas = np.zeros((360, 1280, 3), np.uint8)
                    canvas = _hud(canvas, [
                        (f"READY  {max(0.0, t_end - time.perf_counter()):.1f}s", (0, 255, 255)),
                        ("FLASH NOW after the beep; flash again before the end", (200, 200, 200)),
                    ])
                    cv2.imshow("record left | right", canvas)
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

        session.left = _camera_info("left", left_idx, left_src, args, _role_spec(load_registry(), args.left))
        session.right = _camera_info("right", right_idx, right_src, args, _role_spec(load_registry(), args.right))
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

            cfg_sync = SyncConfig(max_offset=180, min_overlap=30)
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
