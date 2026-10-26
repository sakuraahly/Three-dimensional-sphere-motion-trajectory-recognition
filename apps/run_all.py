"""一条命令：已有会话 → 校验数据 → 三维轨迹 → 可视化 JSON。

为什么要这个入口
----------------
`docs/计划书.md` 的验收口径是"一条命令跑起完整系统（双路采集 → 三维 → 预测 → 可视化）"。
**采集本身不能被编排**（要人摆球、要相机在位），所以这条命令覆盖的是**采集之后**那一段；
采集仍由 `启动程序.bat run` / GUI 第 0 步负责。

它只做"调用 + 判读"，**不复制任何算法**：

    check  → apps/tools/check_dataset.py        （= GUI 第①步 校验数据）
    tri    → apps/run_stereo_video.py           （= GUI 第③步 跑离线双目）
    export → apps/export_visualization.py       （= GUI 第⑤步 导出 JSON）

三步的逻辑一旦抄进这里，就会和 GUI 各漂一份（本项目反复踩的坑）。所以全部走 subprocess。

用法::

    python apps/run_all.py --session data/sessions/20261019-192530
    python apps/run_all.py --session data/sessions/20261019-192530 --dry-run
    python apps/run_all.py --session data/sessions/20261019-192530 --steps tri,export
    python apps/run_all.py --session data/sessions/20261019-192530 --json

退出码：**0 全部通过 / 1 某步失败 / 2 参数错 / 3 有步骤被跳过（已跑的都没失败）**。
3 与 0 必须区分 —— 否则"因为缺标定压根没跑"会被当成"全都好"，这正是静默失败。
`--dry-run` 只打印命令，退出码固定 0（没跑就没有结论）。

步骤被跳过还是失败，取决于**是不是你点名要的**：

- `--steps auto`（默认）：缺前置条件 → `skipped` + 说明缺什么、下一步做什么；
- `--steps check,tri`（显式）：缺前置条件 → `failed`（你点名要跑却跑了，不能装作没要）；
- `--skip-check`：check 一律 `skipped`，原因写明"已跳过、未验证"。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:       # 只为读检测器注册表（R7）；执行链仍是子进程，方式不变
    sys.path.insert(0, str(ROOT))
PY = sys.executable

# import 注册模块即完成注册（R1 要求显式，不扫描目录）
from sphere_tracker.detectors import registry as _detector_registry   # noqa: E402,F401
from sphere_tracker.plugins import default as default_method          # noqa: E402
from sphere_tracker.plugins import describe, names as method_names    # noqa: E402

STEPS = ("check", "tri", "export")
CALIB_LABEL = "config/stereo_calibration.json"
TRAJ_LABEL = "derived/traj3d.csv"

# 每步失败后的"下一步" —— 必须指到**具体入口**，不要只说"检查一下"
NEXT_STEP = {
    "check": "按上面日志的 [错误]/[警告] 逐条修（--strict 下警告也算不通过）："
             "未同步 sync.method=none 要重录闪光或用 --sync-offset；"
             "缺 session.json 只能重录（启动程序.bat run）；缺标定先跑 启动程序.bat calibrate。",
    "tri": "先跑 启动程序.bat 的『相机检查』（启动程序.bat check）确认两台相机都能出图；"
           "再按上面日志换 --detector（高光球用 specular/hough）或在 GUI 第②步点选球心。",
    "export": f"先确认 {TRAJ_LABEL} 里 matched 行不为 0；没有就先重跑 tri"
              "（python apps/run_stereo_video.py <会话>）。",
}

# 从本步日志里挑"关键指标"的行首。**只认本步**，避免把上一步的数字当成本步结果。
METRIC_PREFIX = {
    "check": ("→",),
    "tri": ("匹配：", "三维范围：", "交会残差："),
    "export": ("轨迹 JSON：", "可视化 JSON："),
}


def _default_calib_files(root: Path = ROOT) -> list[Path]:
    """标定存在 = `.json` 或 `.npz` 任一在位（两者由同一次标定一起落盘）。"""
    return [p for p in ((root / "config" / "stereo_calibration.json"),
                        (root / "config" / "stereo_calibration.npz")) if p.exists()]


def _read_session(path: Path) -> tuple[dict | None, str]:
    """返回 (数据, 失败原因)。原因要能直接进人话，所以不抛异常。"""
    if not path.exists():
        return None, "没有这个文件"
    try:
        return json.loads(path.read_text(encoding="utf-8-sig")), ""
    except Exception as exc:                                  # noqa: BLE001
        return None, f"读不了：{exc}"


def _detector_from_session(data: dict | None) -> str:
    """会话里记了检测器就用它，否则 color（与 run_stereo_video 的默认一致）。

    `session.json` 目前**没有**标准字段记检测器（录制器不写），所以这里兼容
    `detector` 与 `detection.method` 两种写法；都没有就落到 color 并在输出里点名，
    不让"用了哪个检测器"变成猜。
    """
    if isinstance(data, dict):
        det = data.get("detector")
        if isinstance(det, str) and det:
            return det
        inner = data.get("detection")
        if isinstance(inner, dict) and isinstance(inner.get("method"), str):
            return inner["method"]
    return "color"


def _skip(name: str, reason: str) -> dict:
    return {"name": name, "status": "skipped", "reason": reason,
            "seconds": 0.0, "cmd": []}


def _pending(name: str, cmd: list[str], note: str = "") -> dict:
    return {"name": name, "status": "pending", "reason": note,
            "seconds": 0.0, "cmd": cmd}


def _blocked(name: str, reason: str, explicit: set[str]) -> dict:
    """前置条件不满足：显式点名 → failed，auto → skipped。"""
    return {"name": name, "status": ("failed" if name in explicit else "skipped"),
            "reason": reason, "seconds": 0.0, "cmd": []}


def plan_steps(
    session: Path,
    *,
    steps: list[str] | tuple[str, ...] = STEPS,
    stereo_mode: str = "auto",
    detector: str | None = None,
    skip_check: bool = False,
    explicit: set[str] | None = None,
    calib_files: list[Path] | None = None,
    root: Path = ROOT,
) -> list[dict]:
    """把"要跑哪几步、每步的命令是什么"算出来 —— **执行与 `--dry-run` 共用这一份**。

    两套拼装逻辑（一套给 dry-run 看、一套给真跑）必然漂，所以只留一份。
    """
    explicit = explicit or set()
    session = Path(session)
    sess_json = session / "session.json"
    data, why = _read_session(sess_json)
    calib = _default_calib_files(root) if calib_files is None else list(calib_files)

    out: list[dict] = []
    for name in steps:
        if name == "check":
            if skip_check:
                out.append(_skip("check", "已按 --skip-check 跳过前置检查（**未验证，不等于通过**）"))
                continue
            if data is None:
                out.append(_blocked(
                    "check", f"缺 {sess_json}（{why}）—— 校验要在会话目录里跑，先录制一次双目",
                    explicit))
                continue
            if not calib:
                out.append(_blocked(
                    "check",
                    f"缺标定文件 {CALIB_LABEL}（.npz/.json 都不在）—— 先跑 启动程序.bat calibrate",
                    explicit))
                continue
            out.append(_pending("check", [
                PY, str(root / "apps" / "tools" / "check_dataset.py"), str(session), "--strict"]))
            continue

        if name == "tri":
            if data is None:
                out.append(_blocked(
                    "tri", f"缺 {sess_json}（{why}）—— 三维这一步读的是会话，先录制一次双目",
                    explicit))
                continue
            missing = []
            for side in ("left", "right"):
                meta = data.get(side) if isinstance(data.get(side), dict) else {}
                rel = meta.get("video")
                if not rel:
                    missing.append(f"{side}.video 字段")
                elif not (session / rel).exists():
                    missing.append(f"{side} 视频 {rel}")
            if missing:
                out.append(_blocked(
                    "tri", f"会话不完整：缺 {'、'.join(missing)} —— 会话可能没录完或被清理，需重录",
                    explicit))
                continue
            det = detector or _detector_from_session(data)
            cmd = [PY, str(root / "apps" / "run_stereo_video.py"), str(session), "--method", det]
            if det == "specular":
                seed = session / "derived" / "seeds.json"
                if seed.exists():
                    cmd += ["--seed-json", str(seed)]
            if stereo_mode != "auto":
                cmd += ["--stereo-mode", stereo_mode]
            note = "" if detector else f"检测器取自{'会话配置' if det != 'color' else '默认值'}：{det}"
            out.append(_pending("tri", cmd, note))
            continue

        if name == "export":
            traj = session / "derived" / "traj3d.csv"
            if not traj.exists():
                out.append(_blocked(
                    "export", f"缺 {TRAJ_LABEL} —— 先跑 tri（python apps/run_stereo_video.py {session}）",
                    explicit))
                continue
            cmd = [PY, str(root / "apps" / "export_visualization.py"), str(session), "--plot"]
            if stereo_mode != "auto":
                cmd += ["--stereo-mode", stereo_mode]
            out.append(_pending("export", cmd))
            continue

        raise ValueError(f"未知步骤：{name}")

    return out


def _brief(line: str) -> str:
    """摘要里的绝对路径缩成项目内相对路径，不然一行 200 字符没法读。"""
    prefix = str(ROOT) + os.sep
    return line.replace(prefix, "").replace(str(ROOT), ".")


def _display(cmd: list[str]) -> str:
    """给人看的命令：项目内的路径显示成相对路径（可原样粘回项目根执行）。"""
    shown = []
    for part in cmd:
        try:
            p = Path(part)
            if p.is_absolute():
                shown.append(p.relative_to(ROOT).as_posix())
                continue
        except ValueError:
            pass
        shown.append(f'"{part}"' if " " in part else part)
    return " ".join(shown)


def _run(cmd: list[str], *, echo, metric_prefix: tuple[str, ...]) -> tuple[int, deque, list[str]]:
    """跑一步：日志**逐行**透传（否则长步骤看起来像卡死），同时留末尾 20 行备用。"""
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(
        cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", env=env)
    tail: deque = deque(maxlen=20)
    metrics: list[str] = []
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip("\r\n")
        echo("  " + line)
        tail.append(line)
        stripped = line.strip()
        if stripped.startswith(metric_prefix):
            metrics.append(stripped)
    return proc.wait(), tail, metrics


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="一条命令：会话 → 校验 → 三维 → 可视化（只调用现有脚本，不含算法）")
    ap.add_argument("--session", required=True,
                    help="已有会话目录（含 session.json）；本工具**不录**（录制要人操作/相机）")
    ap.add_argument("--steps", default="auto",
                    help="auto（默认，按前置条件自动挑）或逗号分隔：check,tri,export")
    ap.add_argument("--stereo-mode", choices=("auto", "angled", "parallel"), default="auto",
                    help="双目模式，透传给 tri / export")
    ap.add_argument("--detector", choices=method_names("detector"), default=None,
                    help="检测器，透传给 tri；默认取会话配置，没有则 %s。可选：%s" % (
                        default_method("detector"),
                        "；".join(f"{d['name']}={d['summary']}"
                                  for d in describe("detector"))))
    ap.add_argument("--skip-check", action="store_true",
                    help="跳过 check（会记为 skipped 并说明，绝不静默）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印将要执行的命令，不执行（退出码固定 0）")
    ap.add_argument("--json", action="store_true",
                    help="stdout 只输出 JSON（人读的日志转 stderr）")
    args = ap.parse_args(argv)

    def say(*parts, **kw) -> None:
        print(*parts, file=(sys.stderr if args.json else sys.stdout), **kw)

    if args.steps.strip() == "auto":
        steps, explicit = list(STEPS), set()
    else:
        picked = [s.strip() for s in args.steps.split(",") if s.strip()]
        bad = [s for s in picked if s not in STEPS]
        if not picked or bad:
            say(f"[参数错] --steps 只认 auto 或 {','.join(STEPS)} 的组合，现在是 {args.steps!r}")
            return 2
        explicit = set(picked)
        steps = [s for s in STEPS if s in explicit]        # 固定顺序，不按用户输入顺序

    session = Path(args.session)
    if session.is_file() and session.name == "session.json":
        session = session.parent
    session = session if session.is_absolute() else (Path.cwd() / session).resolve()

    plan = plan_steps(session, steps=steps, stereo_mode=args.stereo_mode,
                      detector=args.detector, skip_check=args.skip_check,
                      explicit=explicit)

    say(f"[会话] {_display([str(session)])}")
    say(f"[步骤] {' → '.join(s['name'] for s in plan)}"
        f"（{'auto 选择' if not explicit else '显式指定'}）")
    if args.dry_run:
        say("[dry-run] 只打印命令，未执行；退出码固定 0（没跑就没有结论）")
    say()

    for i, step in enumerate(plan, 1):
        head = f"[{i}/{len(plan)}] {step['name']}"
        if step["status"] in ("skipped", "failed"):
            say(f"{head}：{step['status']} —— {step['reason']}")
            if step["status"] == "failed":
                say(f"  下一步：{NEXT_STEP[step['name']]}")
            say()
            continue

        say(f"{head}：{_display(step['cmd'])}")
        if step["reason"]:
            say(f"  说明：{step['reason']}")
        if args.dry_run:
            say()
            continue

        started = time.perf_counter()
        code, tail, metrics = _run(step["cmd"], echo=say,
                                   metric_prefix=METRIC_PREFIX[step["name"]])
        step["seconds"] = round(time.perf_counter() - started, 2)
        if code == 0:
            step["status"] = "ok"
            step["reason"] = " | ".join(_brief(m) for m in metrics) if metrics else ""
            say(f"  [ok] {step['seconds']:.1f}s"
                + (f"  {step['reason']}" if step["reason"] else ""))
        else:
            step["status"] = "failed"
            step["reason"] = f"退出码 {code}"
            say(f"  [失败] 退出码 {code}，耗时 {step['seconds']:.1f}s；末尾 {len(tail)} 行日志：")
            for line in tail:
                say("  | " + line)
            say(f"  下一步：{NEXT_STEP[step['name']]}")
            say(f"  排错可以只跑到这一步：python apps/run_all.py --session "
                f"{_display([str(session)])} --steps {','.join(s['name'] for s in plan[:i])}")
        say()

    failed = [s for s in plan if s["status"] == "failed"]
    skipped = [s for s in plan if s["status"] == "skipped"]
    if args.dry_run:
        exit_code = 0
    elif failed:
        exit_code = 1
    elif skipped:
        exit_code = 3
    else:
        exit_code = 0

    if args.dry_run:
        say("[结论] 以上是**将要执行**的命令，没有真的跑。")
    elif exit_code == 0:
        say(f"[结论] {len(plan)} 步全部通过。")
    elif exit_code == 3:
        say(f"[结论] 已跑的都没失败，但 {len(skipped)} 步被跳过："
            f"{'、'.join(s['name'] for s in skipped)}。"
            " 跳过 ≠ 通过，别当成\"全好\"。")
    else:
        say(f"[结论] 失败 {len(failed)} 步：{'、'.join(s['name'] for s in failed)}；"
            "已停止，后续步骤未执行。")

    if args.json:
        print(json.dumps({"session": str(session), "steps": plan, "ok": exit_code == 0},
                         ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
