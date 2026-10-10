"""相机自运动估计自测：迟滞状态机 + 位移换算方向。

为什么需要它
------------
`GlobalMotionEstimator.is_moving()` 的判定结果**同时决定两件事**：

1. `detectors/motion.py`：背景差分要不要切成"先按全局运动对齐、再差分"。
   （相机一动，MOG2 会把整幅画面都判成前景 —— 见交接文档坑 10）
2. `pipeline.py`：像素重力约束还成不成立（匀速平移可以保留，抖动必须退回自由拟合）。

两件事都不希望逐帧翻转，所以这个方法**必须带迟滞并把状态写回 `self._moving`**。

这里守着 `camera_motion.py` 的两个真实 bug：

**bug 1（迟滞失效）**：`is_moving()` 里 `return` 之后还留着 8 行状态更新代码，
**永远执行不到**，于是 `self._moving` 永远是初始的 `False` —— 迟滞完全失效，
判定退化成"逐帧比较瞬时位移"，在阈值附近来回抖。

**bug 2（换算方向写反）**：`_estimate_features()` 把"小图上解出的相似变换"
换回全分辨率时写成 `M_full = S @ M_small @ Si`，而正确方向是 `Si @ M_small @ S`。
旋转/缩放项不受影响（S 是各向同性的纯缩放），**只有平移项差一个 `s^2`**：
默认 `work_width=320` 配 1920 宽的画面时 s=1/6，报出的位移被缩小 **36 倍**，
于是 `is_moving()` 几乎永不判"在动"，MOG2 的相机补偿（坑 10）永远不会启用。
`_estimate_grid()` / `_estimate_phase()` 两处换算都是对的，**只有主力方法错**，
所以只看"合成场景跑通了"发现不了 —— panning 场景恰好走的是 grid 分支。

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

import cv2
import numpy as np

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


def _textured(width: int = 640, height: int = 360, seed: int = 3) -> np.ndarray:
    """造一张"有角点、有纹理"的灰度图 —— 特征点法要抓得到点才会走 features 分支。"""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (height, width), dtype=np.uint8)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    for (x, y, s) in ((60, 40, 70), (240, 120, 90), (430, 60, 60), (150, 240, 80)):
        cv2.rectangle(img, (x, y), (x + s, y + s), 235, -1)
        cv2.rectangle(img, (x + 10, y + 10), (x + s - 10, y + s - 10), 20, -1)
    return img


def test_estimate() -> bool:
    """真跑一遍 estimate()：静止判静止、平移判在动且位移估得准、换算不随缩放变。"""
    print("----- 真实估计（纹理图平移）-----")
    W, H = 640, 360
    base = _textured(W, H)
    checks: list[tuple[str, bool]] = []

    def shifted(px: float) -> np.ndarray:
        M = np.array([[1.0, 0.0, px], [0.0, 1.0, 0.0]], np.float32)
        return cv2.warpAffine(base, M, (W, H), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)

    def run(px: float, work_width: int = 320):
        """返回 (估计结果, 是否判在动)。第一帧只做初始化，第二次才出结果。"""
        e = GlobalMotionEstimator(work_width=work_width)
        e.estimate(None, base)
        m = e.estimate(base, shifted(px))
        return m, e.is_moving(m)

    # --- 静止 ---
    e = GlobalMotionEstimator()
    e.estimate(None, base)
    m_still = e.estimate(base, base)
    moving_still = e.is_moving(m_still)
    ok_still = (m_still.magnitude < 0.5) and (moving_still is False)
    print(f"  静止（同帧）：方法 {m_still.method:<8} 位移 {m_still.magnitude:.3f} px "
          f"-> 在动={moving_still}  {'通过' if ok_still else '不达标'}")
    checks.append(("静止判静止", ok_still))

    # --- 平移：既要判对"在动"，位移也要估得准 ---
    for px in (6.0, 12.0):
        m, moving = run(px)
        err = abs(m.magnitude - px)
        ok = (m.method != "none") and moving and err <= 1.0
        print(f"  平移 {px:5.1f} px：方法 {m.method:<8} 实测 {m.magnitude:6.2f} px "
              f"（误差 {err:.2f} px）-> 在动={moving}  {'通过' if ok else '不达标'}")
        checks.append((f"平移 {px:.0f}px 判在动且估得准", ok))

    # --- bug 2 的判据：真值不变，报出的位移就不许随 work_width 变 ---
    # 换算写反时（S@M@Si）报出值/真值 恰好等于 s^2：
    # work_width 160/320/640 -> 1.00 / 4.00 / 15.99 px，越缩越小。
    true_px = 16.0
    print(f"  换算方向检查（真值平移固定 {true_px:.0f} px，只改 work_width）：")
    for ww in (160, 320, 640):
        m = GlobalMotionEstimator(work_width=ww)
        m.estimate(None, base)
        r = m.estimate(base, shifted(true_px))
        err = abs(r.magnitude - true_px)
        ok = err <= 1.0
        print(f"    work_width={ww:>3}（s={ww / W:.2f}）：实测 {r.magnitude:6.2f} px "
              f"（报出/真值 {r.magnitude / true_px:.3f}）  {'通过' if ok else '不达标'}")
        checks.append((f"work_width={ww} 位移换算不随缩放变化", ok))

    return all(p for _n, p in checks)


def main() -> int:
    print("=" * 62)
    print("相机自运动估计自测")
    print("=" * 62)
    a = test_hysteresis()
    print()
    b = test_estimate()
    print()
    ok = a and b
    print("判定结果：" + ("全部通过" if ok else "有不达标项"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
