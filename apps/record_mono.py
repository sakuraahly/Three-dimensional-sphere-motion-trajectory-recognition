"""单目录制：一台相机 → 一个视频文件 + 一份能自证的元数据。

为什么需要它（2026-10-28 用户需求）
----------------------------------
双目有 ``record_stereo.py``，单目一直没有"**纯录制**"入口。唯一能录音频的是
``run_realtime.py --record``，但它录的是**标注后的画面**——上面画了检测圆、
轨迹和 HUD。拿这种视频回去再跑检测，叠加层会干扰分割与圆拟合。

所以"先拍、后调参数再检测"这条路是断的。本入口补的就是这一段：**只写相机原图，
不画任何东西**。

产出两个文件（与 ``capture_board`` 的 ``_capture.json`` 同一思路——这批数据要能
自证是怎么拍的）：

- ``<out>``：视频（默认 ``.mkv`` + HuffYUV 无损）
- ``<out>.json``：分辨率 / 实测 fps / 帧数 / 相机身份 / 编码器 / 时长

用法::

    python apps/record_mono.py                          # 默认 10 秒到 data/mono/<时间>/clip.mkv
    python apps/record_mono.py --duration 30 --label 白球_1m
    python apps/record_mono.py --camera main --width 1920 --height 1080 --fps 30
    python apps/record_mono.py --no-preview             # 不看预览（更省 CPU）

退出码：0 = 录完；1 = 开相机/编码器失败；2 = 参数错。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
while not (ROOT / "sphere_tracker").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent                      # 按标记上溯（不写死 parents[N]）
sys.path.insert(0, str(ROOT))

from apps._window import WindowManager                            # noqa: E402
from sphere_tracker import camera_registry as cr                  # noqa: E402
from sphere_tracker.sources import open_source                    # noqa: E402

#: 编码器：四字符码 / 扩展名 / 说明。默认无损 —— 采集一次就别再想"能不能重拍"。
CODECS = {
    "hfyuv": ("HFYU", ".mkv", "HuffYUV（无损，默认）"),
    "mp4v": ("mp4v", ".mp4", "MP4（有损，体积小、随处可播）"),
}
WINDOW = "record_mono"
# 与 `apps/_gui_mono.py` 的落点**必须一致**：GUI 传的是
# `data/mono/<时间戳>/clip.mkv`，CLI 不给 --out 时也生成同样形态的目录。
# 以前这里叫 `data/mono_takes/` 且是平铺文件，于是"CLI 直录"与"GUI 录"
# 落在两个地方，`data/mono/` 下看不到 CLI 录的东西（2026-10-29 统一）。
DEFAULT_OUT_DIR = ROOT / "data" / "mono"


def _backend_const(name: str) -> int:
    return {"auto": cv2.CAP_ANY, "msmf": cv2.CAP_MSMF,
            "dshow": cv2.CAP_DSHOW}[name]


def _resolve_camera(want: str | None) -> int | None:
    """角色名/索引 → 相机索引。``None`` 表示让 open_source 自动挑。"""
    if not want:
        return None
    try:
        return cr.resolve_camera_arg(want, interactive=False)
    except Exception:                                       # noqa: BLE001
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="单目录制（只写相机原图，不叠标注）")
    ap.add_argument("--camera", default=None,
                    help="相机：角色名（left/right/main）或索引；默认自动挑最优")
    ap.add_argument("--width", type=int, default=2560, help="采集宽度（默认 2560）")
    ap.add_argument("--height", type=int, default=1440, help="采集高度（默认 1440）")
    ap.add_argument("--fps", type=float, default=24.0, help="请求帧率（默认 24）")
    ap.add_argument("--backend", choices=("auto", "msmf", "dshow"),
                    default="auto", help="相机后端（默认 auto）")
    ap.add_argument("--duration", type=float, default=10.0,
                    help="录制秒数；0 = 一直录到按 q（默认 10）")
    ap.add_argument("--countdown", type=float, default=3.0,
                    help="开始前倒计时秒数（默认 3）")
    ap.add_argument("--codec", choices=sorted(CODECS), default="hfyuv",
                    help="编码器（默认 hfyuv 无损）")
    ap.add_argument("--out", default=None,
                    help=f"输出路径（默认 {DEFAULT_OUT_DIR.relative_to(ROOT)}/<时间>/clip.<扩展名>）")
    ap.add_argument("--label", default="", help="给这段视频起个名字（写进元数据）")
    ap.add_argument("--no-preview", action="store_true", help="不显示预览窗口")
    args = ap.parse_args(argv)

    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        print("[参数错误] width/height/fps 必须是正数")
        return 2
    if args.duration < 0:
        print("[参数错误] duration 不能是负数（0 = 按 q 停止）")
        return 2

    tag, ext, codec_desc = CODECS[args.codec]
    print(f"[编码] {codec_desc}")

    # ---- 打开相机 ----
    index = _resolve_camera(args.camera)
    if args.camera and index is None:
        print(f"[错误] 没能把 {args.camera!r} 解析成相机索引。"
              f"先跑 python apps/tools/cameras.py list")
        return 1
    try:
        src = open_source(camera=index, width=args.width, height=args.height,
                          fps=args.fps, backend=_backend_const(args.backend))
    except Exception as exc:                                # noqa: BLE001
        print(f"[错误] 打不开相机：{exc}")
        print("       先跑 python apps/tools/cameras.py probe 看哪台能出图")
        return 1

    # 实际分辨率/帧率可能与请求不同（相机只支持特定档位），writer 必须按**实际**建
    w, h = int(src.info.width), int(src.info.height)
    fps = float(src.info.fps or args.fps)
    print(f"[相机] {src.info.backend}  {w}x{h} @ {fps:.1f} fps")

    # 默认落成"会话目录"形态（与 GUI 一致）：data/mono/<YYYYMMDD-HHMMSS>/clip.<ext>
    out = Path(args.out) if args.out else (
        DEFAULT_OUT_DIR / time.strftime("%Y%m%d-%H%M%S") / f"clip{ext}")
    if out.suffix.lower() not in (".mkv", ".mp4", ".avi"):
        out = out.with_suffix(ext)                          # 扩展名与编码器对齐
    out.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*tag)
    writer = cv2.VideoWriter(str(out), fourcc, fps, (w, h))
    if not writer.isOpened():
        print(f"[错误] 编码器 {tag} 打不开，换个 --codec mp4v 试试")
        src.release()
        return 1

    wins = WindowManager()
    preview = not args.no_preview
    print(f"[输出] {out}")
    if args.duration > 0:
        print(f"[录制] {args.duration:g} 秒；倒计时 {args.countdown:g} 秒后开始")
    else:
        print(f"[录制] 按 q 停止；倒计时 {args.countdown:g} 秒后开始")

    # ---- 倒计时（"开始"那一刻要能被听见/看见，闪光同步才准）----
    t_end = time.monotonic() + args.countdown
    while time.monotonic() < t_end:
        ok, frame, _ts = src.read()
        if not ok or frame is None:
            continue
        if preview:
            left = t_end - time.monotonic()
            cv2.putText(frame, f"starting in {left:.1f}s", (24, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
            wins.show(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if wins.handle_key(WINDOW, key, frame):
                continue
            if key in (ord("q"), 27):
                print("[取消] 倒计时里退出")
                writer.release()
                src.release()
                cv2.destroyAllWindows()
                return 1

    # ---- 正式录制 ----
    frames = 0
    t0 = time.monotonic()
    stop_at = t0 + args.duration if args.duration > 0 else None
    print("[录制] 开始")
    try:
        while True:
            ok, frame, ts = src.read()
            if not ok or frame is None:
                # 掉帧不致命，但连读都失败就是相机没了
                if time.monotonic() - t0 > 3.0 and frames == 0:
                    print("[错误] 3 秒内一帧都没拿到：相机可能被占用或后端不对")
                    break
                continue
            writer.write(frame)
            frames += 1
            if preview:
                shown = frame.copy()
                el = time.monotonic() - t0
                cv2.putText(shown, f"REC  {frames} f  {el:.1f}s", (24, 48),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                wins.show(WINDOW, shown)
                key = cv2.waitKey(1) & 0xFF
                if wins.handle_key(WINDOW, key, shown):
                    continue
                if key in (ord("q"), 27):
                    print("[录制] 用户停止")
                    break
            if stop_at is not None and time.monotonic() >= stop_at:
                break
    finally:
        writer.release()
        src.release()
        if preview:
            cv2.destroyAllWindows()

    elapsed = max(1e-6, time.monotonic() - t0)
    print(f"[完成] {frames} 帧，{elapsed:.1f} 秒，实测 {frames / elapsed:.1f} fps")
    print(f"[完成] {out}（{out.stat().st_size / 1e6:.0f} MB）" if out.exists() else "[错误] 视频没写出来")
    if frames == 0:
        return 1

    # ---- 元数据：让这段视频能自证怎么拍的 ----
    meta = {
        "schema": "mono-take/1",
        "label": args.label,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "video": out.name,
        "codec": tag,
        "codec_desc": codec_desc,
        "requested": {"width": args.width, "height": args.height, "fps": args.fps},
        "actual": {"width": w, "height": h, "fps": round(fps, 3),
                   "measured_fps": round(frames / elapsed, 3)},
        "frames": frames,
        "seconds": round(elapsed, 2),
        "camera": {"requested": args.camera, "index": index,
                   "backend": src.info.backend},
        # 设备**身份**（不只是角色名）。为什么非要有（2026-10-30）：单目深度
        # Z = f·D/(2r) 里的 f 来自另一次标定，而那份标定只能证明"当时叫 left"。
        # 如果中间换过相机（或 USB 重插后索引漂了），fx 就是错的，Z 会按同比例错，
        # 事后却查不出原因 —— 只写 {"requested":"left","index":0} 正是这个漏洞。
        # 取不到设备名时是空串，不抛（元数据不全不该让一趟采集白跑）。
        **cr.describe_identity(args.camera),
        "note": "只写相机原图，不含任何检测叠加层；可直接喂 apps/run_video.py",
    }
    meta_path = out.with_suffix(out.suffix + ".json")
    try:
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
        print(f"[元数据] {meta_path}")
    except OSError as exc:
        print(f"[警告] 元数据写不进：{exc}")

    print()
    print("下一步：")
    # 21.6 是**有效直径**（名义 25 mm，检测器半径偏小 13.6%），别改成 25 —— 见交接文档 §35.2
    print(f"  python apps/run_video.py {out} --ball-diameter 21.6")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
