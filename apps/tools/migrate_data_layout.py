"""把旧的双目会话从 `data/sessions/` 搬到 `data/stereo/`（一次性，可重复跑）。

为什么要有这个工具
------------------
`data/sessions/` 这个名字是通用的，但目录里**只有双目**会话；单目另起在
`data/mono/`。于是"会话"在磁盘上指两个不同东西，谁也说不清 `data/sessions`
是不是全部会话（2026-10-29 用户原话："data 下面 session 和 mono 命名有歧义"）。

新布局按**类型**分目录：`data/stereo/` / `data/mono/`。代码侧已经全部改走
`sphere_tracker/paths.py`（`stereo_roots()` 同时返回新旧两个根，所以迁移**前后**
都还能列出会话），剩下的事只有把旧目录里那 34 段数据搬过去。

用法::

    python apps/tools/migrate_data_layout.py --dry-run     # 只打印，不动文件
    python apps/tools/migrate_data_layout.py               # 真的搬

三条硬规则
----------
1. **绝不覆盖**：新位置已经有同名目录就跳过并报出来 —— 那说明这段会话被搬过一次、
   或者同名时间戳各录了一份，两种都不该由脚本替你决定丢弃谁。
2. **只动 `data/sessions/` 的直接子目录**：`data/mono/`、`data/_trash/`、
   `data/calib_pairs/` 一概不碰（它们的名字本来就是对的）。
3. **幂等**：搬完再跑一次输出"移了 0 个"。搬一半中断也能直接重跑，已搬的会被当成
   "新位置已有同名"跳过 —— 所以**不要**为了"重来一次"去删新位置的东西。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "sphere_tracker").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sphere_tracker.paths import LEGACY_STEREO_ROOT, STEREO_ROOT  # noqa: E402


def plan(src: Path, dst: Path) -> tuple[list[Path], list[tuple[str, str]]]:
    """返回 (要搬的目录, [(跳过的名字, 原因)])。

    纯函数、不碰磁盘以外的东西，`--dry-run` 与实际执行走**同一份计划** ——
    dry-run 说的和真跑的不一致，这个工具就没有意义了。
    """
    moves: list[Path] = []
    skips: list[tuple[str, str]] = []
    if not src.is_dir():
        return moves, skips
    for entry in sorted(src.iterdir()):
        if not entry.is_dir():
            skips.append((entry.name, "不是目录"))
            continue
        if (dst / entry.name).exists():
            skips.append((entry.name, "新位置已有同名目录（不覆盖）"))
            continue
        moves.append(entry)
    return moves, skips


def remaining_in_legacy(src: Path) -> int:
    """旧位置还剩几个会话目录（搬完应为 0；跳过的会留在这里）。"""
    if not src.is_dir():
        return 0
    return sum(1 for p in src.iterdir() if p.is_dir())


def move_one(src: Path, dst: Path) -> None:
    """把 src 搬到 dst 下同名。

    同盘用 `os.rename` 语义（`Path.rename`）—— 800 MB 的会话是**改名不是拷贝**，
    瞬间完成；跨盘才退回 `shutil.move`（会真拷，慢但正确）。
    """
    target = dst / src.name
    try:
        src.rename(target)
    except OSError:
        # EXDEV（跨盘）/ 目标被占用等情况：让 shutil 处理或把真实原因抛上去
        shutil.move(str(src), str(target))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="把 data/sessions/* 搬到 data/stereo/*")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印将要做什么，不动任何文件")
    ap.add_argument("--src", default=str(LEGACY_STEREO_ROOT),
                    help="旧位置（默认 data/sessions）")
    ap.add_argument("--dst", default=str(STEREO_ROOT),
                    help="新位置（默认 data/stereo）")
    args = ap.parse_args(argv)

    src, dst = Path(args.src), Path(args.dst)
    print(f"旧位置：{src}")
    print(f"新位置：{dst}")
    if not src.is_dir():
        print("旧位置不存在 —— 已经迁移过（或这台机器没有旧数据），无事可做。")
        print("移了 0 个、跳过 0 个，还剩 0 个在旧位置")
        return 0
    if not args.dry_run:
        dst.mkdir(parents=True, exist_ok=True)
    elif not dst.is_dir():
        print("（新位置还不存在；真跑时会先创建它）")
    if args.dry_run:
        print("[dry-run] 只打印，不动任何文件")
    print()

    moves, skips = plan(src, dst)
    moved = 0
    failed: list[tuple[str, str]] = []
    for d in moves:
        if args.dry_run:
            print(f"  移动 {d.name}")
            continue
        try:
            move_one(d, dst)
        except OSError as exc:
            # 单个会话搬不动不该让其余 33 个也停下 —— 记下来，最后一起报
            failed.append((d.name, str(exc)))
            continue
        moved += 1
        print(f"  移动 {d.name}")

    # 跳过分两类：计划阶段就判定的（同名/不是目录），和执行阶段失败的
    for name, why in skips:
        print(f"  跳过 {name} —— {why}")
    for name, why in failed:
        print(f"  失败 {name} —— {why}")

    left = remaining_in_legacy(src)
    n_skip = len(skips) + len(failed)
    detail = [f"新位置同名 {sum(1 for _n, w in skips if '同名' in w)}",
              f"不是目录 {sum(1 for _n, w in skips if w == '不是目录')}"]
    if failed:
        detail.append(f"移动失败 {len(failed)}")
    print()
    verb = "将移" if args.dry_run else "移了"
    print(f"{verb} {len(moves) if args.dry_run else moved} 个、"
          f"跳过 {n_skip} 个（{' / '.join(detail)}），"
          f"还剩 {left} 个在旧位置")
    if failed:
        print("下一步：看上面的失败原因（通常是文件被播放器/资源管理器占用）；"
              "关掉占用后直接重跑本脚本，已搬的会自动跳过。")
        return 1
    if skips:
        print("下一步：跳过的会话需要你逐个确认（同名 = 两段都有内容）；"
              "确认后手工删掉旧位置那一份，或把它挪到别处。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
