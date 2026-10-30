"""校验 `data/stereo/<id>/` 里的采集会话：视频 / 标定 / 同步是否自洽。

用法::

    python apps/tools/check_dataset.py data/stereo/20261019-101500
    python apps/tools/check_dataset.py --root data/stereo             # 换个根
    python apps/tools/check_dataset.py                               # 全部（新旧根都扫）
    python apps/tools/check_dataset.py --strict ...                  # warning 也算失败

为什么是单独一个工具
--------------------
录坏的代价是**重录**，不是重跑一次测试。所以要在录完立刻能查：
- 两路视频能不能打开、分辨率是否与 session/标定一致；
- 帧数差多少（同步偏移只能吸收常数差，掉帧会累积）；
- 标定文件是不是 session 记录的那一份（sha256）；
- 物理基线 vs 算法基线是不是差得离谱（>10% 警告，>20% 报错）。

判定分三档：`[错误]` 硬失败 / `[警告]`（`--strict` 下也失败）/ `[提示]` 只是告知、
永不影响退出码。典型提示：容器标称 fps 与 sidecar 真实帧率不符（sidecar 齐全时
按它分析即可，只是播放器换算会偏），以及两路清晰度对比。

它**只读**，不改任何文件；退出码 0/1 供采集脚本或手工检查使用。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "sphere_tracker").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stereo.sessions import (  # noqa: E402
    describe, list_sessions, load_session, validate_session,
)
from sphere_tracker.paths import stereo_roots  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="校验双目采集会话")
    ap.add_argument("paths", nargs="*", help="会话目录（默认扫描 --root 下全部）")
    # 默认不写死一个根：这是**读**路径，迁移前后都要能列到会话，所以交给 stereo_roots()
    ap.add_argument("--root", default=None,
                    help="会话根目录（默认 data/stereo 与旧位置 data/sessions 都扫）")
    ap.add_argument("--strict", action="store_true",
                    help="warning 也当失败（[提示] 只是告知，永不影响退出码）")
    return ap.parse_args()


def check_one(session_dir: Path, *, strict: bool) -> tuple[int, int, int]:
    errors, warnings, notes = validate_session(session_dir, with_notes=True)
    try:
        _d, session = load_session(session_dir)
        print(describe(session))
    except Exception as exc:            # noqa: BLE001 —— 坏会话要继续报，不是崩
        print(f"  [错误] 读不了 session.json：{exc}")
    for msg in errors:
        print(f"  [错误] {msg}")
    for msg in warnings:
        print(f"  [警告] {msg}")
    # 提示永不导致失败：它是"知道就好"，不是"必须改"
    for msg in notes:
        print(f"  [提示] {msg}")
    ok = not errors and not (strict and warnings)
    print(f"  → {'通过' if ok else '不通过'}"
          f"（错误 {len(errors)}，警告 {len(warnings)}，提示 {len(notes)}）")
    return len(errors), len(warnings), len(notes)


def main() -> int:
    args = parse_args()
    paths = [Path(p) for p in args.paths]
    if not paths:
        # 新的根在前（stereo_roots 的返回顺序）：同一个会话名两边都有时，只查新的那份，
        # 否则迁移中途会把同一段会话查两遍、计数翻倍
        roots = [Path(args.root)] if args.root else list(stereo_roots())
        seen: set[str] = set()
        for root in roots:
            for p in list_sessions(root):
                if p.name in seen:
                    continue
                seen.add(p.name)
                paths.append(p)
        if not paths:
            print("没有找到会话：" + "、".join(str(r) for r in roots))
            print("下一步：先录制一次（P1 录制器），或把已有会话放进 data/stereo/<id>/")
            return 0

    total_e = total_w = total_n = 0
    bad = 0
    for i, path in enumerate(paths, 1):
        print("=" * 74)
        print(f"[{i}/{len(paths)}] {path}")
        e, w, n = check_one(path, strict=args.strict)
        total_e += e
        total_w += w
        total_n += n
        bad += int(e > 0 or (args.strict and w > 0))
        print()

    print("=" * 74)
    print(f"合计 {len(paths)} 个会话：不通过 {bad}，错误 {total_e}，"
          f"警告 {total_w}，提示 {total_n}")
    print("判定结果：" + ("全部通过" if bad == 0 else "有不通过项"))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
