"""角点出身自测：检测器可辨识 + 回退必须显式 + 一对之内禁止混用。

为什么需要它
------------
`calib/board.py` 里有两条角点检测路径：

- `cv2.findChessboardCornersSB`（SB，抗模糊那版，首选）
- `cv2.findChessboardCorners`（经典版，早期是"SB 失败就自动试"的静默回退）

两者返回的角点顺序**恰好整体反序**（实测直接对应 RMS **323.112 px**、
整份反序后 **0.020 px**；分类正好是 8x5 点阵的 `rot180` 自同构）。
后果具有极强的欺骗性：

- **单目标定免疫** —— 顺序反序仍是一组自洽对应（实测 RMS -0.0066 vs 0.0704 px），
  所以**每一路各自看起来都正常**；
- **双目标定不免疫** —— 左右两路若一路 SB、一路经典，同名下标不是同一个物理角点。
  合成复现：整体 RMS 21~42 px、极线误差上万像素、**基线被吹到真值的 8~15 倍**。

更要命的是**它不会报错**，单目指标也好看 —— 真机上只能靠事后猜。所以本测试钉三件事：

**1. 出身必须可辨识。** `BoardObservation.detector` 如实记录 `"sb"` / `"classic"`，
没找到时是 `""`。**不记录出身 = 事后无从归因。**

**2. 回退必须显式。** `allow_classic_fallback` 默认 `False`：
- `exhaustive=False`（每帧粗检那条路）**永远不回退** —— 这是原有的隐式行为，
  现在变成**显式且可断言**的；
- `exhaustive=True` + 默认 → 也不回退（**这是相对旧代码的行为变更**，
  所以下面第 2 节把它单独钉住）。

**3. 一对之内禁止混用。** 两道守卫：粗检后一道、精确复核后一道。
用替身（stub）真造出"左 SB / 右经典"的场景，断言该帧被拒、且原因里
**同时含"出身/检测器"字样与是哪一路**。

这里**不测**"关掉回退会不会少收帧"—— 那条已经实测过（`outputs/_probe_detector_coverage.py`：
高斯模糊 k=15 时 **SB 失败而经典成功**，所以采集器是**显式打开**回退的）。
本测试只保证"开了回退也不会静默混用"。

反向自证（改了就会红）
--------------------
==================================  ================================================  ========
改动                                注入方式                                          结果
==================================  ================================================  ========
让 detector 永远记 "sb"             `obs.detector = "sb"`（_finish 里）                exit=1
把回退恢复成隐式（去掉显式闸门）    `if corners is None and exhaustive:`               exit=1
去掉粗检那道出身守卫                `del` 掉 `offer()` 里第 0 步的分支                 exit=1
去掉精确复核那道守卫                `del` 掉 7b) 的分支                                exit=1
==================================  ================================================  ========

用法::

    python tests/selftest_calib_provenance.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
while not (ROOT / "sphere_tracker").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from calib import capture as cap_module
from calib.board import BoardObservation, BoardSpec, detect_board, detect_board_in_roi
from calib.capture import AutoPairCollector, CaptureConfig

SPEC = BoardSpec(cols=8, rows=5, square_mm=25.0)
LOGGER = logging.getLogger("selftest_calib_provenance")


def render_board(px: int = 60, margin: int = 120) -> np.ndarray:
    w = (SPEC.cols + 1) * px + 2 * margin
    h = (SPEC.rows + 1) * px + 2 * margin
    img = np.full((h, w), 255, np.uint8)
    for r in range(SPEC.rows + 1):
        for c in range(SPEC.cols + 1):
            if (r + c) % 2 == 0:
                img[margin + r * px:margin + (r + 1) * px,
                    margin + c * px:margin + (c + 1) * px] = 0
    return img


def _blank(size: tuple[int, int] = (300, 400)) -> np.ndarray:
    return np.full(size, 255, np.uint8)


# --------------------------------------------------------------------------
def test_provenance_is_recorded() -> bool:
    """第 1 节：`detector` 如实记录出身；没找到时是空串。"""
    print("----- 1. 角点出身必须可辨识 -----")
    img = render_board()
    by_sb = detect_board(img, SPEC, exhaustive=True)
    by_classic = detect_board(img, SPEC, use_sb=False, exhaustive=True,
                              allow_classic_fallback=True)
    not_found = detect_board(_blank(), SPEC, exhaustive=True)

    print(f"  首选路径            -> found={by_sb.found}  detector={by_sb.detector!r}")
    print(f"  use_sb=False + 允许 -> found={by_classic.found}  detector={by_classic.detector!r}")
    print(f"  空白图              -> found={not_found.found}  detector={not_found.detector!r}")

    checks = [
        ("SB 路径记成 'sb'", by_sb.found and by_sb.detector == "sb"),
        ("经典路径记成 'classic'（**不是**混成一个值）",
         by_classic.found and by_classic.detector == "classic"),
        ("两条路的出身确实不同（这是本测试的前提）",
         by_sb.detector != by_classic.detector),
        ("没找到时 found=False 且 detector=''（不假装知道自己是谁）",
         not not_found.found and not_found.detector == ""),
        ("默认构造的 BoardObservation 也是 ''（不默认成 sb）",
         BoardObservation().detector == ""),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_fallback_is_explicit() -> bool:
    """第 2 节：不回退是默认；`exhaustive=False` 永远不回退。"""
    print("----- 2. 回退必须显式（默认不许自作主张） -----")
    img = render_board()

    # 用一张"SB 找不到、经典能找到"的强模糊图来区分两条路 —— 实测 k=15 就是这个分界
    import cv2
    blurred = cv2.GaussianBlur(img, (15, 15), 0)
    sb_alone = detect_board(blurred, SPEC, exhaustive=True)
    with_fb = detect_board(blurred, SPEC, exhaustive=True, allow_classic_fallback=True)
    coarse = detect_board(blurred, SPEC, work_width=640, exhaustive=False,
                          allow_classic_fallback=True)

    print(f"  强模糊图 SB-alone（默认）      -> found={sb_alone.found} detector={sb_alone.detector!r}")
    print(f"  强模糊图 + 允许回退            -> found={with_fb.found} detector={with_fb.detector!r}")
    print(f"  强模糊图 粗检(exhaustive=False)-> found={coarse.found} detector={coarse.detector!r}")

    checks = [
        ("默认**不回退**：SB 找不到就认找不到（相对旧行为的一次刻意变更）",
         not sb_alone.found and sb_alone.detector == ""),
        ("显式允许后，经典能把它救回来（说明回退本身是有用的）",
         with_fb.found and with_fb.detector == "classic"),
        ("**粗检路径即使允许也不回退**（exhaustive=False 是硬闸门）",
         (not coarse.found) or coarse.detector == "sb"),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_roi_passes_detector_through() -> bool:
    """第 3 节：`detect_board_in_roi()` 把出身原样带出来。"""
    print("----- 3. 精确复核（ROI）不许把出身弄丢 -----")
    img = render_board()
    hint = detect_board(img, SPEC, exhaustive=True).corners
    roi_sb = detect_board_in_roi(img, SPEC, hint)
    roi_cl = detect_board_in_roi(img, SPEC, hint, allow_classic_fallback=True)
    # hint 来自经典时，ROI 仍走自己的首选（SB）—— 出身以 **ROI 这一次** 为准
    hint_cl = detect_board(img, SPEC, use_sb=False, exhaustive=True,
                           allow_classic_fallback=True).corners
    roi_from_classic_hint = detect_board_in_roi(img, SPEC, hint_cl,
                                                allow_classic_fallback=True)

    print(f"  ROI 默认          -> detector={roi_sb.detector!r}")
    print(f"  ROI 允许回退      -> detector={roi_cl.detector!r}")
    print(f"  ROI（hint 来自经典）-> detector={roi_from_classic_hint.detector!r}")

    checks = [
        ("ROI 默认记 'sb'", roi_sb is not None and roi_sb.detector == "sb"),
        ("ROI 允许回退时仍走 SB（SB 成功就不回退）",
         roi_cl is not None and roi_cl.detector == "sb"),
        ("**出身以 ROI 这一次为准**，不是照抄 hint 的来源",
         roi_from_classic_hint is not None and roi_from_classic_hint.detector == "sb"),
        ("ROI 的 precise 仍为 True（精度语义没被改动影响）",
         roi_sb is not None and roi_sb.precise is True),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def _offer_with_stub(want_detectors: tuple[str, ...]) -> tuple[bool, str, dict]:
    """用替身造出"两路出身不同"的一帧。

    `want_detectors` **按调用序号**给出身：第 1、2 次调用是粗检（左、右），
    第 3、4 次是精确复核（左、右）。超出范围的调用沿用最后一个值。

    ⚠️ 第一版这里写成了 `want_detectors[i % 2]` —— 只要调用次数超过 2 次
    取模就会错位，于是"精确复核才混用"那个场景**根本没被造出来**，
    测试却以为通过了。是它自己报红才发现的。
    """
    img = render_board()
    real_detect = cap_module.detect_board
    real_roi = cap_module.detect_board_in_roi
    real_obs = real_detect(img, SPEC, exhaustive=True)
    calls = {"n": 0}

    def _pick() -> str:
        i = calls["n"]
        calls["n"] = i + 1
        return want_detectors[min(i, len(want_detectors) - 1)]

    def _make(which: str) -> BoardObservation:
        obs = BoardObservation()
        obs.found = True
        obs.detector = which
        obs.corners = real_obs.corners
        obs.homography = real_obs.homography
        obs.sharpness, obs.area_ratio = 500.0, 0.30
        obs.margin_px, obs.brightness = 100.0, 128.0
        return obs

    def fake_detect(gray, spec, **kw):
        return _make(_pick())

    def fake_roi(gray, spec, hint, **kw):
        return _make(_pick())

    cfg = CaptureConfig()
    cfg.require_stationary = False
    col = AutoPairCollector(SPEC, cfg)
    cap_module.detect_board = fake_detect
    cap_module.detect_board_in_roi = fake_roi
    try:
        ok, why = col.offer(img, img)
    finally:
        cap_module.detect_board = real_detect
        cap_module.detect_board_in_roi = real_roi
    return ok, why, {"count": col.count, "calls": calls["n"]}


def test_mixed_detectors_are_rejected() -> bool:
    """第 4 节：一对之内混用检测器 -> 当场拒绝，并指出是哪一路。"""
    print("----- 4. 一对之内禁止混用（两道守卫） -----")
    # 粗检（第 1、2 次调用）就混用 -> 第 0 步守卫拦下
    ok_coarse, why_coarse, st_coarse = _offer_with_stub(("sb", "classic"))
    # 粗检一致、精确复核（第 3、4 次调用）才混用 -> 第 7b 步守卫拦下
    ok_deep, why_deep, st_deep = _offer_with_stub(("sb", "sb", "sb", "classic"))

    print(f"  粗检就混用      -> ok={ok_coarse}  收下 {st_coarse['count']} 对"
          f"（探测调用 {st_coarse['calls']} 次）")
    print(f"      原因：{why_coarse}")
    print(f"  精确复核才混用  -> ok={ok_deep}  收下 {st_deep['count']} 对"
          f"（探测调用 {st_deep['calls']} 次）")
    print(f"      原因：{why_deep}")

    def mentions_side_and_detector(why: str) -> bool:
        has_side = ("左" in why and "右" in why)
        has_det = ("出身" in why or "检测器" in why)
        return has_side and has_det

    checks = [
        ("粗检混用被拒", ok_coarse is False),
        ("粗检混用一条都没收下", st_coarse["count"] == 0),
        ("原因里同时点出「左/右」与「出身/检测器」（可行动，不只是「失败了」）",
         mentions_side_and_detector(why_coarse)),
        ("原因里带上两路各自的出身值（sb / classic）",
         "sb" in why_coarse and "classic" in why_coarse),
        ("替身确实被调用到精确复核那一步（≥4 次，否则这个场景没被造出来）",
         st_deep["calls"] >= 4),
        ("精确复核混用也被拒", ok_deep is False),
        ("精确复核混用一条都没收下", st_deep["count"] == 0),
        ("该原因同样点出左右与出身", mentions_side_and_detector(why_deep)),
        ("该原因也带上两路出身值", "sb" in why_deep and "classic" in why_deep),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_consistent_pair_still_passes() -> bool:
    """第 5 节：守卫不能误伤 —— 两路出身一致时照收。"""
    print("----- 5. 守卫不误伤：两路一致时照收 -----")
    img = render_board()
    cfg = CaptureConfig()
    cfg.require_stationary = False
    col = AutoPairCollector(SPEC, cfg)
    ok, why = col.offer(img, img)
    det_l = col.last_obs[0].detector
    det_r = col.last_obs[1].detector
    print(f"  真渲染合成帧 -> ok={ok}  收下 {col.count} 对")
    print(f"      两路出身：左={det_l!r} 右={det_r!r}")

    checks = [
        ("两路出身一致（都 sb）", det_l == det_r == "sb"),
        ("该帧被收下（守卫没有误伤正常路径）", ok is True and col.count == 1),
        ("收下的对里角点数正确", col.pairs[0].corners_left.shape[0] == SPEC.n_points),
    ]
    for name, ok_ in checks:
        print(f"  [{'通过' if ok_ else '不达标'}] {name}")
    return all(ok_ for _n, ok_ in checks)


def main() -> int:
    print("=" * 70)
    print("角点出身自测（检测器可辨识 / 回退显式 / 禁止混用）")
    print("=" * 70)
    logging.disable(logging.CRITICAL)
    results = []
    for fn in (test_provenance_is_recorded,
               test_fallback_is_explicit,
               test_roi_passes_detector_through,
               test_mixed_detectors_are_rejected,
               test_consistent_pair_still_passes):
        results.append(fn())
        print()
    ok = all(results)
    print("判定结果：" + ("全部通过" if ok else "有不达标项"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
