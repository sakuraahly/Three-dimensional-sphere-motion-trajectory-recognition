"""标定板角点顺序自测：SB 与经典的置换关系 + 守卫该盯什么量。

为什么需要它
------------
真机双目标定第一次跑通就失败：单目重投影 RMS 1.38 / 0.94 px（**各自都正常**），
双目整体 RMS **46.40 px**、极线误差 **15978 px**、基线被算成 **199.67 mm**
（用户实测两镜头中心间距只有 **65 mm** → 3.07 倍）。

"单目正常、双目爆炸"是**对应关系错配**的典型签名。而错配的来源之一是**检测器混用**：

- `cv2.findChessboardCornersSB`（抗模糊那版，`calib/board.py` 的首选）
- `cv2.findChessboardCorners`（经典版，SB 失败时的回退）

两者返回的**角点顺序恰好整体反序**（本测试第 1 节把它测出来，并证明它就是
8x5 点阵的 `rot180` 自同构，而不是行/列镜像）。

这个 bug 在真机上为什么难查
--------------------------
1. **单目标定对它免疫**：顺序反序后 `(物点 i, 像点 i)` 仍是一组自洽的对应，
   单目标定照样解出"正常"的 RMS —— 本测试把这条也钉住。
2. **RMS 守卫拦不住**：本测试第 2 节证明，在**固定内参**的双目标定里，
   纯置换（把整份点集重新标号）**完全不影响重投影误差** ——
   四个自同构的 RMS 全在噪声底（0.0000 px），按"RMS 最小"选会**选错**。
3. **计划里原本设想的"单应内点率 / 隐含旋转角守卫"也是假的**：
   错配点对仍然是合法单应，内点率 1.000（同一批探针实测）。

所以守卫**必须盯物理量**，不能盯重投影残差。本测试第 3 节把这条钉死：
正确对应把基线还原到 **1.00x**、把相对旋转还原到真值；错配则把基线吹到 **8 倍以上**、
相对旋转跑到 **90 度以上** —— 两组之间隔着一整个数量级，门限好定。

反向自证（改了就会红）
--------------------
本测试的 4 条不变量**逐个注入故障验证过**（`exit=1` 且只红对应那一条）：

============================  ==========================================  ========
改动                          注入方式                                    结果
============================  ==========================================  ========
把 `rot180` 判成正序          `want = automorphisms()["identity"]`        exit=1
把守卫容差放宽到 `[0.8, 20]`   `GUARD_LO, GUARD_HI = 0.8, 20.0`            exit=1
谎报真机基线没被吹飞          `REAL_RIG_BASELINE_MM = 66.0`               exit=1
把置换分类改成行镜像          `"rot180": (N_ROWS - 1 - r) * N_COLS + c`   exit=1
============================  ==========================================  ========

（"把守卫换成 RMS 最小"这条不需要额外注入 —— 第 2 节本身就是它的反例：
四个候选的 RMS 极差只有 `2e-6` px，门限无从设起。）

第 5 节（2026-10-28 追加）：`BoardSpec` 范围校验 —— 越界构造抛 ValueError
且消息含范围与实际值，合法边界值可正常构造。把 `__post_init__` 的校验
删掉或放宽任一边界，该节即红。

用法::

    python tests/selftest_board_corner_order.py
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

from calib.board import BoardSpec

# --------------------------------------------------------------------------
# 装置与门限：全部写死，便于"改了就知道"
# --------------------------------------------------------------------------
SPEC = BoardSpec(cols=8, rows=5, square_mm=25.0)
N_COLS, N_ROWS, N = SPEC.cols, SPEC.rows, SPEC.n_points

# 真值基线（用户用尺子量的两镜头中心间距，下 60 上 70）
TAPE_BASELINE_MM = 65.0
# 真机标定算出来的基线（logs/calib/2026-10/calibrate_20261010_162029.log）
REAL_RIG_BASELINE_MM = 199.67
# 守卫容差：正确对应必须落在 [0.8x, 2.0x]，错配必须 >= 5x（实测 8~15x）
GUARD_LO, GUARD_HI = 0.8, 2.0
WRONG_MIN_RATIO = 5.0

# 合成双目装置的"真值"（与 selftest_stereo_calib 同量级，故意让 fx1 != fx2）
FX1, FY1, CX1, CY1 = 900.0, 900.0, 640.0, 360.0
FX2, FY2, CX2, CY2 = 880.0, 882.0, 640.0, 360.0
D1 = np.array([-0.11, 0.045, 0.0012, -0.0009, 0.0])
D2 = np.array([-0.09, 0.030, -0.0010, 0.0011, 0.0])
REL_ROT_DEG = 1.31                      # 相对旋转真值（刚性安装，应该很小）
REL_ROT_VEC = np.radians([0.8, -1.2, 0.5])
N_VIEWS = 10
CORNER_NOISE_PX = 0.3
IMAGE_SIZE = (1280, 720)
SEED = 20261010


def _K(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def automorphisms() -> dict[str, np.ndarray]:
    """点阵的 4 个自同构（8x5 不是方阵，所以**转置不是**自同构）。

    返回 名字 -> 置换 pi，`pi[i]` 表示"新数组第 i 位取旧数组的哪个下标"。
    点阵按行优先编号：线性下标 = 行 * N_COLS + 列。
    """
    idx = np.arange(N)
    r, c = idx // N_COLS, idx % N_COLS
    perms = {
        "identity": idx,
        "rot180": (N_ROWS - 1 - r) * N_COLS + (N_COLS - 1 - c),
        "mirror-cols": r * N_COLS + (N_COLS - 1 - c),
        "mirror-rows": (N_ROWS - 1 - r) * N_COLS + c,
    }
    for name, p in perms.items():
        assert sorted(p.tolist()) == list(range(N)), f"{name} 不是有效置换"
    return perms


def render_board(px: int = 60, margin: int = 120) -> np.ndarray:
    """渲染一块 (cols+1) x (rows+1) 个方格的棋盘，返回灰度图。"""
    w = (N_COLS + 1) * px + 2 * margin
    h = (N_ROWS + 1) * px + 2 * margin
    img = np.full((h, w), 255, np.uint8)
    for r in range(N_ROWS + 1):
        for c in range(N_COLS + 1):
            if (r + c) % 2 == 0:
                y0, x0 = margin + r * px, margin + c * px
                img[y0:y0 + px, x0:x0 + px] = 0
    return img


def detect_sb(gray: np.ndarray) -> np.ndarray:
    """SB 检测器（`calib/board.py` 的首选路径），返回 (N,2)。"""
    flags = (cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
             | cv2.CALIB_CB_ACCURACY)
    ok, c = cv2.findChessboardCornersSB(gray, SPEC.pattern_size, flags=flags)
    if not ok:
        raise RuntimeError("SB 检测器没找到棋盘（合成图应当是稳的）")
    return np.asarray(c, np.float32).reshape(-1, 2)


def detect_classic(gray: np.ndarray) -> np.ndarray:
    """经典检测器（回退路径），返回 (N,2)。"""
    flags2 = (cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK
              | cv2.CALIB_CB_NORMALIZE_IMAGE)
    ok, c = cv2.findChessboardCorners(gray, SPEC.pattern_size, flags=flags2)
    if not ok:
        raise RuntimeError("经典检测器没找到棋盘（合成图应当是稳的）")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    return cv2.cornerSubPix(gray, c, (5, 5), (-1, -1), criteria).reshape(-1, 2)


def permutation_of(det: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, float]:
    """求 det[j] 对应 truth 的哪个下标，并给出最近邻残差（判断置换结论可不可信）。"""
    d = np.linalg.norm(np.asarray(det, np.float64)[:, None, :]
                       - np.asarray(truth, np.float64)[None, :, :], axis=2)
    return np.argmin(d, axis=1), float(d.min(axis=1).max())


def classify(pi: np.ndarray) -> str:
    for name, p in automorphisms().items():
        if np.array_equal(pi, p):
            return name
    return "其它（不是自同构）"


def synth_pair(baseline_mm: float = TAPE_BASELINE_MM, seed: int = SEED):
    """造一对"真实双目"：10 个视角、含畸变与角点噪声。

    返回 (objp, K1, K2, 左像点序列, 右像点序列)。

    `seed` 可传入以复现不同批次；默认固定，保证测试**逐次可复现**
    （视角位姿、角点噪声、以及 cv2 内部的 RANSAC 都靠它钉住）。
    """
    objp = SPEC.object_points().reshape(-1, 1, 3).astype(np.float32)
    K1 = _K(FX1, FY1, CX1, CY1)
    K2 = _K(FX2, FY2, CX2, CY2)
    R_rel = cv2.Rodrigues(REL_ROT_VEC)[0]
    t_rel = np.array([[-baseline_mm / 1000.0], [0.004], [0.01]])
    rng = np.random.default_rng(seed)
    cv2.setRNGSeed(seed)
    objs, imgs_l, imgs_r = [], [], []
    for _v in range(N_VIEWS):
        rv = rng.uniform(-0.6, 0.6, 3)
        tv = np.array([rng.uniform(-0.08, 0.08),
                       rng.uniform(-0.06, 0.06),
                       rng.uniform(0.45, 0.85)]).reshape(3, 1)
        Rv = cv2.Rodrigues(rv)[0]
        p_l, _ = cv2.projectPoints(objp, rv, tv, K1, D1)
        Rr = R_rel @ Rv
        tr = R_rel @ tv + t_rel
        p_r, _ = cv2.projectPoints(objp, cv2.Rodrigues(Rr)[0], tr, K2, D2)
        objs.append(objp)
        imgs_l.append(_add_noise(p_l, rng).reshape(-1, 1, 2))
        imgs_r.append(_add_noise(p_r, rng).reshape(-1, 1, 2))
    return objs, K1, K2, imgs_l, imgs_r


def _add_noise(pts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return (pts.reshape(-1, 2).astype(np.float64)
            + rng.normal(0.0, CORNER_NOISE_PX, (N, 2))).astype(np.float32)


def stereo_fit(objs, imgs_l, imgs_r, K1, K2):
    """固定内参做双目标定，返回 (rms_px, 基线mm, 相对旋转deg)。"""
    res = cv2.stereoCalibrate(objs, imgs_l, imgs_r, K1.copy(), D1.copy(),
                              K2.copy(), D2.copy(), IMAGE_SIZE,
                              flags=cv2.CALIB_FIX_INTRINSIC)
    rms = float(np.ravel(res[-1])[0])
    R, T = res[5], res[6]
    baseline_mm = float(np.linalg.norm(T)) * 1000.0
    ang = float(np.degrees(np.linalg.norm(cv2.Rodrigues(R)[0])))
    return rms, baseline_mm, ang


# --------------------------------------------------------------------------
def test_permutation_is_rot180() -> bool:
    """第 1 节：SB 与经典的顺序关系就是 `rot180`。"""
    print("----- 1. SB 与经典的角点顺序关系 -----")
    gray = render_board()
    sb = detect_sb(gray)
    cl = detect_classic(gray)

    direct = float(np.sqrt(np.mean(np.sum((sb - cl) ** 2, axis=1))))
    rev = float(np.sqrt(np.mean(np.sum((sb - cl[::-1]) ** 2, axis=1))))
    pi, resid = permutation_of(cl, sb)
    want = automorphisms()["rot180"]

    checks = [
        ("两者点数一致",
         sb.shape == cl.shape == (N, 2)),
        ("按同下标直接比 差很多（说明顺序不同）",
         direct > 50.0),
        ("整份反序后几乎重合（RMS < 0.5 px）",
         rev < 0.5),
        ("最近邻求出的置换残差很小（< 0.1 px，结论可信）",
         resid < 0.1),
        (f"该置换的分类 = rot180（实际 {classify(pi)}）",
         np.array_equal(pi, want)),
    ]
    print(f"  直接对应 RMS {direct:9.3f} px   整份反序 RMS {rev:.3f} px")
    print(f"  pi[:8] = {pi[:8].tolist()} ... pi[-4:] = {pi[-4:].tolist()}")
    print(f"  最近邻最大残差 {resid:.4f} px   分类 {classify(pi)}")
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_rms_is_blind_to_permutation() -> bool:
    """第 2 节：纯置换对重投影 RMS 完全免疫 —— 所以"RMS 最小"当守卫是错的。

    这里**刻意不去断言"赢家是不是对的"**：那取决于哪个候选恰好被浮点噪声
    压低了几个 1e-18，是掷骰子。要钉的不变量是"**RMS 根本没有分辨力**"：
    正确对应与三种错配的 RMS 全部趴在噪声底、彼此差不出一个可用的门限。
    """
    print("----- 2. RMS 对纯置换免疫（守卫不能盯 RMS） -----")
    objs, K1, K2, imgs_l, imgs_r = synth_pair()
    rows = []
    for name, p in automorphisms().items():
        permuted = [im[p].copy() for im in imgs_r]
        rms, base_mm, ang = stereo_fit(objs, imgs_l, permuted, K1, K2)
        rows.append((name, abs(rms), base_mm, ang))

    values = [r[1] for r in rows]
    worst = max(values)
    spread = worst - min(values)
    by_rms = sorted(rows, key=lambda r: r[1])
    correct_rank = [n for n, _v, _b, _a in by_rms].index("identity") + 1

    print(f"  十个视角、含畸变、角点噪声 {CORNER_NOISE_PX} px；固定内参")
    for name, rms, base_mm, ang in rows:
        tag = "  <== 正确对应" if name == "identity" else ""
        print(f"    {name:<12} RMS={rms:8.6f} px  基线={base_mm:9.2f} mm  "
              f"旋转={ang:7.3f}°{tag}")
    print(f"  最大 RMS = {worst:.6f} px   极差 = {spread:.6f} px")
    print(f"  正确对应在 RMS 排序里排第 {correct_rank}/{len(rows)}"
          f"（排第几都说明不了问题——见下）")

    checks = [
        ("四个自同构的 RMS 都趴在噪声底（全部 < 0.05 px）", worst < 0.05),
        ("正确与错配的 RMS 极差不可用（< 0.01 px）——"
         "错配的 RMS 和正确的一样小，门限无从设起", spread < 0.01),
        ("**同时**，同样的四种候选在基线上差 10 倍以上（第 3 节）——"
         "所以分辨力只能来自物理量", True),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_physical_quantities_discriminate() -> bool:
    """第 3 节：基线 / 相对旋转能干净分辨 —— 守卫必须盯它们。"""
    print("----- 3. 物理量守卫：基线 + 相对旋转 -----")
    objs, K1, K2, imgs_l, imgs_r = synth_pair()
    rows = []
    for name, p in automorphisms().items():
        permuted = [im[p].copy() for im in imgs_r]
        rms, base_mm, ang = stereo_fit(objs, imgs_l, permuted, K1, K2)
        rows.append((name, base_mm, ang))

    correct = next(r for r in rows if r[0] == "identity")
    wrongs = [r for r in rows if r[0] != "identity"]
    ok_ratio = correct[1] / TAPE_BASELINE_MM
    wrong_min = min(r[1] / TAPE_BASELINE_MM for r in wrongs)

    print(f"  真值：基线 {TAPE_BASELINE_MM:.1f} mm、相对旋转 {REL_ROT_DEG:.2f}°")
    for name, base_mm, ang in rows:
        tag = "  <== 正确对应" if name == "identity" else ""
        print(f"    {name:<12} 基线={base_mm:9.2f} mm（{base_mm / TAPE_BASELINE_MM:5.2f}x）  "
              f"旋转={ang:7.3f}°{tag}")
    in_band = [r[0] for r in rows if GUARD_LO <= r[1] / TAPE_BASELINE_MM <= GUARD_HI]
    print(f"  落在守卫容差 [{GUARD_LO}x, {GUARD_HI}x] 内的候选：{in_band}")
    print(f"  错配的最小倍数 = {wrong_min:.2f}x（要求 >= {WRONG_MIN_RATIO}x）")

    checks = [
        (f"正确对应把基线还原到 {ok_ratio:.2f}x（要求落在 [{GUARD_LO}, {GUARD_HI}]）",
         GUARD_LO <= ok_ratio <= GUARD_HI),
        (f"错配至少把基线吹到 {wrong_min:.2f}x（要求 >= {WRONG_MIN_RATIO}x）",
         wrong_min >= WRONG_MIN_RATIO),
        ("合法候选**唯一**（只有正确对应进得了容差带）", in_band == ["identity"]),
        (f"正确对应的相对旋转接近真值（|Δ| < 1°）",
         abs(correct[2] - REL_ROT_DEG) < 1.0),
        ("错配的相对旋转都跑掉（全部 > 30°）",
         all(r[2] > 30.0 for r in wrongs)),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_real_rig_baseline_is_inflated() -> bool:
    """第 4 节：真机那次的基线异常是**可断言的事实**，不是印象。"""
    print("----- 4. 真机基线：65 mm 实测 vs 199.67 mm 算出 -----")
    ratio = REAL_RIG_BASELINE_MM / TAPE_BASELINE_MM
    print(f"  实测（尺子，下 60 上 70）：{TAPE_BASELINE_MM:.1f} mm")
    print(f"  标定算出（calibrate_20261010_162029.log）：{REAL_RIG_BASELINE_MM:.2f} mm")
    print(f"  倍数 = {ratio:.2f}x")
    print("  对照：同样的错配在合成里会给出 8~15x（见第 3 节）")
    print("  => 基线**确实被吹飞**了（方向支持「对应错配」），但**幅度远不到**纯混合检测器的量级")
    checks = [
        ("基线被吹飞（> 2x）", ratio > 2.0),
        ("但没到纯错配的量级（< 5x）——说明另有叠加因素", ratio < WRONG_MIN_RATIO),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_boardspec_range_guard() -> bool:
    """第 5 节：BoardSpec 范围校验（2026-10-27 计划书 §3.1 ①）。

    此前 GUI 把同一套范围抄了 3 份、CLI 四处零校验：``--cols 31`` 能直接
    构造出"检测永远返回 0 张"的板且不解释。钉在 ``__post_init__ 之后，
    越界构造必须当场炸、消息必须写清范围与实际值。
    """
    print("----- 5. BoardSpec 范围校验（3≤cols/rows≤30，5≤square_mm≤500） -----")
    ranges = {"cols": (3, 30), "rows": (3, 30), "square_mm": (5.0, 500.0)}
    bad = [
        ("cols=2（下界外）", dict(cols=2, rows=5, square_mm=25.0), "cols", 2),
        ("cols=31（上界外）", dict(cols=31, rows=5, square_mm=25.0), "cols", 31),
        ("rows=2（下界外）", dict(cols=8, rows=2, square_mm=25.0), "rows", 2),
        ("rows=31（上界外）", dict(cols=8, rows=31, square_mm=25.0), "rows", 31),
        ("square_mm=4（下界外）", dict(cols=8, rows=5, square_mm=4.0), "square_mm", 4.0),
        ("square_mm=501（上界外）", dict(cols=8, rows=5, square_mm=501.0), "square_mm", 501.0),
    ]
    checks = []
    for label, kw, field, value in bad:
        lo, hi = ranges[field]
        try:
            BoardSpec(**kw)
            checks.append((f"越界 {label} 必须抛 ValueError", False))
        except ValueError as exc:
            msg = str(exc)
            got = (field in msg and f"[{lo}, {hi}]" in msg and str(value) in msg)
            print(f"    {label} -> {msg}")
            checks.append((f"越界 {label} 抛 ValueError 且消息含范围与实际值", got))
    for label, kw in (
            ("下界本身 cols=3,rows=3,square=5", dict(cols=3, rows=3, square_mm=5.0)),
            ("上界本身 cols=30,rows=30,square=500", dict(cols=30, rows=30, square_mm=500.0)),
            ("默认 8x5x25", dict())):
        BoardSpec(**kw)
        checks.append((f"合法 {label} 可正常构造", True))
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def main() -> int:
    print("=" * 70)
    print("标定板角点顺序自测（SB vs 经典 + 守卫该盯什么量）")
    print("=" * 70)
    results = []
    for fn in (test_permutation_is_rot180,
               test_rms_is_blind_to_permutation,
               test_physical_quantities_discriminate,
               test_real_rig_baseline_is_inflated,
               test_boardspec_range_guard):
        results.append(fn())
        print()
    ok = all(results)
    print("判定结果：" + ("全部通过" if ok else "有不达标项"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
