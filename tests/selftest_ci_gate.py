"""CI 门的自律：显式跳过机制不能被悄悄放宽。

为什么需要它
------------
2026-11-22 CI 首次真跑：standards 修好后，8 个测试在 runner 上红、本地全绿 ——
它们依赖本机真机标定（config/calib/*）、走查报告、真实 GUI 显示器与 UTF-8 终端，
CI 的 config/ 只有 .gitkeep、虚拟屏只有 637px、stdout 是 cp1252，**根本无法运行**。
按项目铁律"不为让绿而放松判据"，拍板方案是：**CI 显式跳过这 8 个并逐条写原因**，
本地 pre-push 门每次 push 仍全量跑（26/26），覆盖不打折。

机制（`apps/run_tests.py`）：
  - `CI_SKIP_ENV` 环境变量置 1 时，从测试清单里移除 `CI_LOCAL_ASSET_SKIP` 的测试；
  - 跳过 ≠ 通过：结果 JSON 有 `ci_skipped` 块（逐条带原因），控制台显式打印。

本文件守三条不变量（任何一条被改坏都让 CI 的"跳过"退化成"静默放松"）：
  ① 跳过名单里的每个测试文件都真实存在，且每条都有非空原因；
  ② 不设环境变量时 `--list` 是**全量**（本地行为不受影响）；
  ③ 设了环境变量时正好少 `len(CI_LOCAL_ASSET_SKIP)` 个，且少的正是名单里的那些。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "apps"))
import run_tests  # noqa: E402

PASS_MARKER = "判定结果：全部通过"
LIST_LINE_PREFIX = ("  [?]", "  [慢]", "  [快]")


def _listed(env_extra: dict[str, str] | None) -> list[str]:
    env = {**os.environ}
    env.pop(run_tests.CI_SKIP_ENV, None)      # 先确保干净基线
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(
        [sys.executable, "apps/run_tests.py", "--list"],
        cwd=str(ROOT), capture_output=True, text=True,
        encoding="utf-8", errors="replace", env=env, timeout=60,
    )
    return [ln.strip() for ln in (proc.stdout or "").splitlines()
            if ln.startswith(LIST_LINE_PREFIX)]


def main() -> int:
    checks: list[tuple[str, bool]] = []

    skip = run_tests.CI_LOCAL_ASSET_SKIP
    # ① 名单是真的测试文件 + 每条有原因
    checks.append(("CI 跳过名单非空（否则本机制形同虚设）", len(skip) > 0))
    checks.append(("名单里每个测试文件都真实存在",
                   all((ROOT / "tests" / n).is_file() for n in skip)))
    checks.append(("每条跳过都有非空原因（不许静默跳过）",
                   all(isinstance(w, str) and w.strip() for w in skip.values())))
    checks.append(("名单不会误伤 MUST_RUN 之外的不相关测试（8 个，逐条可数）",
                   len(skip) == 8 and all(n.startswith("selftest_") for n in skip)))

    base = _listed(None)
    ci = _listed({run_tests.CI_SKIP_ENV: "1"})
    # ② 本地全量
    checks.append(("不设环境变量时 --list 至少 29 个（本地全量不受影响）",
                   len(base) >= 29))
    # ③ CI 正好少 8 个
    checks.append((f"CI 模式正好少 {len(skip)} 个",
                   len(base) - len(ci) == len(skip)))
    # --list 行形如 "[快] selftest_x.py   中位 1.3s 历史 [...]"，文件名是第 2 个 token
    base_names = {ln.split()[1] for ln in base}
    ci_names = {ln.split()[1] for ln in ci}
    gone = base_names - ci_names
    checks.append(("少的正是名单里的那些（不是随便砍 8 个）", gone == set(skip)))
    checks.append(("CI 没多出任何测试（清单不是被改乱）", not (ci_names - base_names)))

    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    print(f"  本地 --list {len(base)} 个 / CI --list {len(ci)} 个 / 跳过 {len(gone)} 个")
    ok = all(ok for _, ok in checks)
    print("判定结果：" + ("全部通过" if ok else "有不达标项"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
