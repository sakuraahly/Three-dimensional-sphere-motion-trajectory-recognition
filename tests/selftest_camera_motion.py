"""相机自运动估计自测：迟滞状态机。

为什么需要它
------------
`GlobalMotionEstimator.is_moving()` 的判定结果**同时决定两件事**：

1. `detectors/motion.py`：背景差分要不要切成"先按全局运动对齐、再差分"。
   （相机一动，MOG2 会把整幅画面都判成前景 —— 见交接文档坑 10）
2. `pipeline.py`：像素重力约束还成不成立（匀速平移可以保留，抖动必须退回自由拟合）。

两件事都不希望逐帧翻转，所以这个方法**必须带迟滞并把状态写回 `self._moving`**。

守着的真实 bug（迟滞失效）：`is_moving()` 里 `return` 之后还留着 8 行状态更新代码，
**永远执行不到**，于是 `self._moving` 永远是初始的 `False` —— 迟滞完全失效，
判定退化成"逐帧比较瞬时位移"，在阈值附近来回抖。

用法::

    python tests/selftest_camera_motion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
while not (ROOT / "sphere_tracker").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: F401  (保留：后续用例会用到)

from sphere_tracker.camera_motion import GlobalMotionEstimator, MotionEstimate


def est(dx: float = 0.0, dy: float = 0.0, method: str = "features") -> MotionEstimate:
    """造一个位移已知的估计结果（magnitude = hypot(dx, dy)）。"""
    return MotionEstimate(dx=dx, dy=dy, method=method, response=0.9, inliers=100)


# --------------------------------------------------------------------------
def test_hysteresis() -> bool:
    """状态机语义：门限、滞回带、保持、相位加倍、reset。"""
    print("----- 迟滞状态机（threshold_px=1.2, hysteresis=0.5）-----")
    print("  门限：进入 1.2 px / 退出 0.6 px（相位相关法各乘 2）")

    checks: list[tuple[str, bool]] = []

    def step(e: GlobalMotionEstimator, dx: float, method: str = "features",
             expect: bool | None = None, label: str = "") -> bool:
        got = e.is_moving(est(dx=dx, method=method))
        good = True if expect is None else (got is expect)
        if expect is not None:
            print(f"    {label:<34} 位移 {dx:5.2f} px -> 在动={got}"
                  f"（应为 {expect}）  {'通过' if good else '不达标'}")
        return good

    # --- 进入 / 滞回带 / 退出 ---
    e = GlobalMotionEstimator()
    seq = [
        (0.50, False, "静止：低于进入门限"),
        (1.50, True, "起动了：越过进入门限"),
        (0.80, True, "落回滞回带（0.6~1.2）：必须保持'在动'"),
        (1.10, True, "滞回带内再抖一下：仍保持"),
        (0.70, True, "贴近退出门限上沿：仍保持"),
        (0.50, False, "掉到退出门限以下：终于判静止"),
        (0.90, False, "静止后回到滞回带：不应自行变'在动'"),
    ]
    for dx, expect, label in seq:
        checks.append((label, step(e, dx, expect=expect, label=label)))

    # --- 估计失败时沿用上一帧状态（不要凭空复位）---
    e = GlobalMotionEstimator()
    e.is_moving(est(dx=5.0))                      # 先进入"在动"
    ok_fail_moving = e.is_moving(MotionEstimate(method="none"))
    print(f"    {'估计失败时保持（在动 -> 在动）':<34} 在动={ok_fail_moving}（应为 True）"
          f"  {'通过' if ok_fail_moving else '不达标'}")
    checks.append(("估计失败时保持在动", ok_fail_moving))
    ok_none = e.is_moving(None)
    checks.append(("传 None 时保持在动", ok_none))

    e2 = GlobalMotionEstimator()
    e2.is_moving(est(dx=0.0))                     # 静止
    ok_fail_still = e2.is_moving(MotionEstimate(method="none")) is False
    print(f"    {'估计失败时保持（静止 -> 静止）':<34} 在动={not ok_fail_still}（应为 False）"
          f"  {'通过' if ok_fail_still else '不达标'}")
    checks.append(("估计失败时保持静止", ok_fail_still))

    # --- 相位相关法门限加倍（它会被运动物体带偏，所以更保守）---
    e = GlobalMotionEstimator()
    ok_phase_mid = step(e, 1.50, method="phase", expect=False,
                        label="phase：1.5 px 不到 2.4 px 门限")
    checks.append(("phase 门限加倍（进入）", ok_phase_mid))
    ok_phase_in = step(e, 2.50, method="phase", expect=True, label="phase：越过 2.4 px")
    checks.append(("phase 越过门限判在动", ok_phase_in))
    ok_phase_band = step(e, 1.30, method="phase", expect=True,
                         label="phase：滞回带 1.2~2.4 px 保持")
    checks.append(("phase 滞回带保持", ok_phase_band))
    ok_phase_out = step(e, 1.00, method="phase", expect=False,
                        label="phase：掉到 1.2 px 以下")
    checks.append(("phase 退出", ok_phase_out))

    # --- reset 清状态 ---
    e = GlobalMotionEstimator()
    e.is_moving(est(dx=5.0))
    e.reset()
    ok_reset = e.is_moving(MotionEstimate(method="none")) is False
    print(f"    {'reset() 后回到静止':<34} 在动={not ok_reset}（应为 False）"
          f"  {'通过' if ok_reset else '不达标'}")
    checks.append(("reset 清状态", ok_reset))

    # --- 核心回归：在门限附近抖动时状态不许翻转 ---
    # 旧实现（状态永不写回）在这里会 True/False 交替翻转，正是要防的抖动。
    jitter = [1.30, 1.10, 1.30, 1.10, 1.25, 1.15, 1.30, 1.05, 1.20, 1.10]
    flips = 0
    prev = None
    for dx in jitter:
        cur = e.is_moving(est(dx=dx))
        if prev is not None and cur != prev:
            flips += 1
        prev = cur
    ok_noflap = (flips == 0 and prev is True)
    print(f"    {'门限附近抖动 10 帧的翻转次数':<34} {flips}（应为 0）"
          f"  {'通过' if ok_noflap else '不达标'}")
    checks.append(("滞回带内不翻转", ok_noflap))

    return all(p for _n, p in checks)


def main() -> int:
    print("=" * 62)
    print("相机自运动估计自测")
    print("=" * 62)
    ok = test_hysteresis()
    print()
    print("判定结果：" + ("全部通过" if ok else "有不达标项"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
