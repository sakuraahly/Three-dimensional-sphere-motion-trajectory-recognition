"""单目内参标定入口：一组棋盘图 → `config/mono_calibration.json`。

为什么它必须是一个独立入口（2026-10-27 计划书 §3.3 / B-100）
------------------------------------------------------------
``calib/stereo_calib.py:641`` 早就有单目内参算法，chain 标定的左右两支一直在用它，
但它**没有对外的命令行入口** —— 于是想标单目的人只能去读源码、或者误以为
"单目不用标定，填个 fx 就行"。事实是：按 60° 视场猜出来的 fx 误差 1% 就是
0.6 m 处 6 mm，而单目本身在 0.6 m 的预算只有 ±7 mm。

本入口只做三件事：读图、调 :func:`calib.mono_calib.calibrate_mono`、存结果。
**判定逻辑全在库里**（RMS 分档 / 留一法 fx 稳定性 / 视角数 / 姿态覆盖 / 板面积），
这里不重复实现；``--check`` 时连存都不存，只报告。
相机身份从同目录的 ``_capture.json`` 读（读不到就留空，不影响标定）。

用法::

    python apps/calibrate_mono.py --dir outputs/mono_calib_frames
    python apps/calibrate_mono.py --dir <目录> --check          # 只校验
    python apps/calibrate_mono.py --dir <目录> --cols 9 --rows 6 --square 20

退出码：0 = 标定完成（或 --check 通过）；1 = 这组图不可用（带原因）；2 = 参数错。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
while not (ROOT / "sphere_tracker").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent                      # 按标记上溯（不写死 parents[N]）
sys.path.insert(0, str(ROOT))

from calib.board import BoardSpec                            # noqa: E402
from calib.mono_calib import (MonoCalibError, QUALITY, SCHEMA,  # noqa: E402
                              calibrate_mono, camera_note, save_mono)

DEFAULT_DIR = ROOT / "outputs" / "mono_calib_frames"
DEFAULT_OUT = ROOT / "config" / "mono_calibration.json"

LOG = logging.getLogger("calibrate_mono")


def _setup_logging() -> Path:
    """日志落到 ``logs/calib/<年月>/calibrate_mono_<时间>.log``。

    复用 `apps/_common.setup_logging`：这里原来自带一份 ``basicConfig``，路径与
    分区规则各写一遍 —— 改日志布局时最容易漏掉的正是这种"第二处实现"。
    """
    from apps._common import setup_logging

    return setup_logging("calibrate_mono")


def _image_paths(d: Path) -> list[Path]:
    """目录里的图片（png/jpg，不递归，按名字排序）。"""
    items = [p for p in d.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    return sorted(items)


def _camera_from_sidecar(d: Path) -> tuple[str, str]:
    """从 ``<dir>/_capture.json`` 取"这批图是哪台相机拍的"，返回 (标识, 来源)。

    来源是**枚举**（``identity`` / ``role-only`` / ``unknown``），会原样写进
    配置产物，所以这里**只放事实、不放中文**；人话留给 ``report_text`` 与命令行。
    ``identity`` 与 ``role-only`` 的可信度差一个量级，混成一个字段就没法比较了。

    为什么必须记（2026-10-29）：边车原来只有 ``"camera": "left"`` —— 那是**角色名**。
    ``config/cameras.json`` 里 ``left`` 指向的设备可以换、离线的设备会回退到别的索引，
    于是"fx 在 1268~1457 之间差 15%"这种事发生后**无从归因**。身份要跟着图走。

    ``identity`` 时标识形如 ``"aoni webcam A25 left | 0"``（设备名 | OpenCV 索引）。
    **不抛**：边车缺失/损坏只是"这批图的出身不明"，不该让标定跑不起来。
    """
    try:
        raw = json.loads((d / "_capture.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "", "unknown"
    if not isinstance(raw, dict):
        return "", "unknown"
    name = str(raw.get("camera_name") or "")
    ident = str(raw.get("camera_identity") or "")
    port = str(raw.get("camera_port") or "")
    idx = raw.get("camera_index")
    role = str(raw.get("camera") or "")
    label = name or ident or port
    if label and (name or ident):
        # 只有 name / identity 才算"身份"；光有 port 不足以核对设备，退成 role-only。
        return (f"{label} | {idx}"
                if isinstance(idx, (int, str)) and str(idx) else label), "identity"
    if role:
        # 老边车（只有角色名）：标识就是角色名本身，来源如实标 role-only，
        # 别拿角色名冒充身份（这个区分正是本轮要解决的）。
        return role, "role-only"
    return "", "unknown"


def _device_identity(d: Path, camera_label: str) -> str:
    """把这次标定绑到**稳定的 USB 身份**，而不是 `camera="left"` 这种角色名。

    优先用边车里的 `camera_identity`（录制时写下的真实 instance id）；
    老边车没有就退回 `config/cameras.json` 的 roles 表按角色查。
    都查不到就返回空串 —— 读取方会据此提示"没绑定，换相机不会报警"。
    """
    try:
        raw = json.loads((d / "_capture.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raw = {}
    if isinstance(raw, dict):
        ident = str(raw.get("camera_identity") or "")
        if ident:
            return ident
        role = str(raw.get("camera") or "")
    else:
        role = ""
    if not role and camera_label in ("left", "right"):
        role = camera_label
    try:
        reg = json.loads((ROOT / "config" / "cameras.json").read_text(
            encoding="utf-8"))
        spec = (reg.get("roles") or {}).get(role) or {}
        return str(spec.get("identity") or "")
    except (OSError, ValueError):
        return ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="单目内参标定（棋盘图 → config/mono_calibration.json）")
    ap.add_argument("--dir", "--frames", dest="dir", default=str(DEFAULT_DIR),
                    help=f"标定图目录（默认 {DEFAULT_DIR.relative_to(ROOT)}；--frames 同义）")
    ap.add_argument("--cols", type=int, default=8, help="棋盘内角点列数（默认 8）")
    ap.add_argument("--rows", type=int, default=5, help="棋盘内角点行数（默认 5）")
    ap.add_argument("--square", type=float, default=25.0,
                    help="方格边长 mm（默认 25）")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help=f"输出文件，或**目录**（目录时写 <dir>/mono_calibration.json）。"
                         f"默认 {DEFAULT_OUT.relative_to(ROOT)}")
    ap.add_argument("--session", default=None,
                    help="目录里混了多次采集时选哪一次：last=最近一次，"
                         "或 20261029-1922（可只给到分钟）。不给且检出多次采集会被拒")
    ap.add_argument("--check", action="store_true",
                    help="只校验并报告，不写文件")
    ap.add_argument("--fix-principal-point", action="store_true",
                    help="标定时固定主点在图像中心（图少时更稳，但可能引入偏差）")
    args = ap.parse_args(argv)

    log_path = _setup_logging()
    print(f"本次日志：{log_path}")

    d = Path(args.dir)
    if not d.is_dir():
        print(f"[错误] 目录不存在：{d}")
        print("       先拍图：在单目窗第①步点「拍标定图」，或用手册里的采集命令")
        return 2
    try:
        spec = BoardSpec(cols=args.cols, rows=args.rows, square_mm=args.square)
    except ValueError as exc:
        print(f"[参数错误] {exc}")
        return 2

    paths = _image_paths(d)
    cam, cam_src = _camera_from_sidecar(d)
    print(f"[输入] {d}")
    print(f"[输入] {len(paths)} 张图，板 {spec.cols}×{spec.rows}，方格 {spec.square_mm:g} mm")
    print(f"[输入] 相机：{camera_note(cam, cam_src)}")
    if args.session:
        print(f"[输入] 只标一次采集：{args.session}（目录里共 {len(paths)} 张，其余不参与）")
    if not paths:
        print("[错误] 目录里没有图片")
        return 2
    LOG.info("开始标定：%d 张，板 %dx%d，方格 %g mm，相机 %s",
             len(paths), spec.cols, spec.rows, spec.square_mm, cam or "未知")

    try:
        calib = calibrate_mono(paths, spec,
                               fix_principal_point=args.fix_principal_point,
                               session=args.session,
                               camera=cam, camera_source=cam_src)
    except MonoCalibError as exc:
        print()
        print("[不可用] " + str(exc))
        if exc.rejected:
            print(f"[逐图] 共 {len(exc.rejected)} 张被拒：")
            for name, why in exc.rejected[:12]:
                print(f"        · {name}：{why}")
            if len(exc.rejected) > 12:
                print(f"        …… 其余 {len(exc.rejected) - 12} 张略（见日志）")
        print()
        # 建议跟着**实际失败的那一条**走：以前不管什么原因都印同一段"板要摆到四角"，
        # 而焦距不稳定时真正该做的是"加倾斜"（平移多少张都救不回来）。
        msg = str(exc)
        if "次采集" in msg or "没找到会话" in msg:
            print("[下一步] 加 --session last（最近一次）或 --session "
                  "<上面列出的时间，如 20261029-1922> 指定一次采集")
        elif "焦距不稳定" in msg:
            print("[下一步] 这是姿态问题，不是清晰度问题：补拍 3~5 张明显倾斜的"
                  "（±20° 以上）再重标")
        elif "RMS" in msg:
            print("[下一步] 先删掉上面列出的最差几张（拖影/反光/被遮挡），再重标")
        else:
            print("[下一步] 按上面的原因重拍。最常见的两条：")
            print(f"        · 板要摆到画面四角与中央"
                  f"（现在需要覆盖 ≥{QUALITY['min_regions']}/5 个区域）")
            print("        · 每个姿态倾斜 30–45°，不要一直平放在画面中间")
        LOG.warning("标定质量不合格（仅质量差，非程序错误，不中止）：%s", exc)
        # --check 是「只校验不写文件」的显式语义，保持不写。
        if args.check:
            return 0
        # 新原则：质量差也要出结果、只有错误才紧急。单目质量差时库里直接抛
        # MonoCalibError（没有 MonoCalib 对象可存），所以另存一份**诊断 JSON**到
        # 旁路路径供检查：产物形态照 save_mono（单个 .json），带 forced_unusable /
        # grade / verdict 自证「仅链路测试，不得用于数据结论」；正式的
        # config/mono_calibration.json 绝不改动。
        out = Path(args.out)
        if out.suffix.lower() != ".json":
            out = out / "mono_calibration.json"
        stamp = time.strftime("%Y%m%d-%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
        bypass = out.with_name(f"{out.stem}_unusable_{stamp}.json")
        diag = {
            "schema": SCHEMA,
            "version": 1,
            "usable": False,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_dir": str(d),
            "quality": {
                **QUALITY,
                "forced_unusable": True,
                "grade": "不可用",
                "verdict": f"{exc} 【仅链路测试，不得用于数据结论】",
                "reason": str(exc),
                "rejected": [{"name": n, "why": w} for n, w in exc.rejected],
            },
        }
        try:
            bypass.parent.mkdir(parents=True, exist_ok=True)
            bypass.write_text(json.dumps(diag, ensure_ascii=False, indent=2) + "\n",
                              encoding="utf-8")
        except OSError as wexc:
            # 旁路都写不进 = 真错误（磁盘/权限），照常报错中止。
            print(f"[错误] 写不进旁路诊断文件 {bypass}：{wexc}")
            return 1
        print()
        print("!" * 66)
        print("⚠ 标定质量不合格（仅质量差，非程序错误，不中止）")
        print(f"  诊断结果另存为旁路文件供检查：{bypass}")
        print("  激活装置档案未改动（正式标定路径保持原样）。")
        print("!" * 66)
        return 0

    print()
    print(calib.report())
    LOG.info("标定完成：fx=%.2f fy=%.2f RMS=%.3f 视角=%d 焦距抖动=%s 相机=%s",
             calib.fx, calib.fy, calib.rms_px, calib.views,
             f"{calib.fx_spread:.1%}" if calib.fx_spread >= 0 else "未算",
             camera_note(calib.camera, calib.camera_source))

    if args.check:
        print()
        print("[--check] 只校验，不写文件。去掉 --check 即保存。")
        return 0

    # ``--out`` 两种写法都收：``xxx.json`` 是文件；其它一律当**目录**
    # （目录里写固定的 mono_calibration.json）。为什么允许目录：落一份试验性标定
    # 时人不该被迫在命令行里拼文件名，而拼错后缀（``outputs/mono_calib_batch4``
    # 不带 .json）以前会写出一个没有扩展名的文件，行为难猜。
    out = Path(args.out)
    if out.suffix.lower() != ".json":
        out = out / "mono_calibration.json"
    # 绑定设备身份（2026-11-08）：不绑定时换相机只会静默沿用旧 fx。
    calib.device_identity = _device_identity(d, cam)
    try:
        written = save_mono(calib, out)
    except OSError as exc:
        print(f"[错误] 写不进 {out}：{exc}")
        return 1
    print()
    print(f"[已保存] {written}")
    print("[下一步] 运行期会自动读它（run_realtime 会按标定值算焦距，不再按 60° 视场猜）")
    LOG.info("已保存：%s", written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
