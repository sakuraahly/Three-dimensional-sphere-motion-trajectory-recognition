"""单目内参标定自测：合成棋盘回收内参 + **判定真的会红**。

为什么这个测试必须存在（2026-10-27 计划书 §3.5）
------------------------------------------------
单目内参是"看起来对但可能错"的典型：RMS 0.3 px 的结果也可能因为图不够、
板没摆开而把畸变系数估歪，而那时**界面上没有任何征兆**。所以判据不能只测
"能标出来"，还要测：

1. **回收精度**：已知 ``K`` 的合成图 → 标定 → ``fx`` 相对误差 ≤ 0.5%；
2. **不够就报错**：3 个姿态必须抛 ``MonoCalibError``，不许给"勉强能用"的结果；
3. **畸形输入**：``BoardSpec(cols=2)`` 必须 ``ValueError``（P0a 的守卫）；
4. **判定真的在跑**：把 ``min_regions`` 抬到不可能满足的值 → 必须抛错。
   这一条是**反向自证** —— 没有它，前三条的"通过"可能只是判定被短路了。

合成方式：先画一张正面棋盘，再用已知 ``K`` 把它的四个外角投影到目标姿态，
``warpPerspective`` 贴到空白画布上。这样得到的图**带有真实的透视与畸变**，
检测器（SB）能按正常流程检出 40 个内角点 —— 实测确认过，不是假设。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
while not (ROOT / "sphere_tracker").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent                      # 按标记上溯（不写死 parents[N]）
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calib.board import BoardSpec                       # noqa: E402
from calib.mono_calib import (MonoCalibError, QUALITY,  # noqa: E402
                              calibrate_mono, load_mono, save_mono)

COLS, ROWS = 8, 5
SQ_MM = 25.0
SQ_PX = 60
W, H = 1920, 1080
#: 合成用的"真值"内参：fx=fy=1663、主点在正中心、无畸变。
K_TRUE = np.array([[1663.0, 0.0, W / 2], [0.0, 1663.0, H / 2], [0.0, 0.0, 1.0]])
D_TRUE = np.zeros(5, np.float64)
SPEC = BoardSpec(cols=COLS, rows=ROWS, square_mm=SQ_MM)


def _board_image() -> np.ndarray:
    """正面棋盘：内角点 (COLS, ROWS) → 图案是 (COLS+1)×(ROWS+1) 个格子。"""
    w, h = (COLS + 1) * SQ_PX, (ROWS + 1) * SQ_PX
    img = np.full((h, w), 255, np.uint8)
    for r in range(ROWS + 1):
        for c in range(COLS + 1):
            if (r + c) % 2 == 0:
                img[r * SQ_PX:(r + 1) * SQ_PX,
                    c * SQ_PX:(c + 1) * SQ_PX] = 0
    return img


def _outer_3d() -> np.ndarray:
    """棋盘图案四个外角的 3D 坐标，单位毫米。

    ⚠ **尺寸是 (cols+1)×(rows+1) 格，不是 (cols-1)×(rows-1)**。
    内角点从第一个格子的右下角开始编号（0,0），所以第一个格子的左上角在
    ``(-sq, -sq)``，第 (cols+1) 格的右下角在 ``(cols·sq, rows·sq)``。
    一开始按"内角点范围 ± 半格"算成 200×125 mm，比真实图案小一整格，
    于是所有内角点整体错位半格 —— 主点被拉偏 86/147 px、畸变系数去补偿它，
    **而 RMS 只有 0.58 px 看不出来**。合成测试里这类几何错比算法错更隐蔽。
    """
    return np.array([[-SQ_MM, -SQ_MM, 0],
                     [COLS * SQ_MM, -SQ_MM, 0],
                     [COLS * SQ_MM, ROWS * SQ_MM, 0],
                     [-SQ_MM, ROWS * SQ_MM, 0]], np.float32)


def _poses() -> list[tuple[np.ndarray, np.ndarray]]:
    """12 个姿态：板中心铺满 5 个区域（四角 + 中央），z 有变化，带小幅倾斜。

    为什么按"目标像素"反算平移：要保证每个区域都真的被覆盖 —— 随机撒点
    很容易全落在中央，那样畸变系数就不可辨（正是本模块要挡的情况）。
    """
    targets = [(x * W, y * H) for x, y in (
        (0.30, 0.30), (0.70, 0.30), (0.30, 0.70), (0.70, 0.70),   # 四角
        (0.50, 0.50), (0.50, 0.30), (0.50, 0.70),                 # 中央 + 上下
        (0.30, 0.50), (0.70, 0.50),                               # 左中、右中
        (0.42, 0.42), (0.58, 0.58), (0.50, 0.38),                 # 补充
    )]
    rng = np.random.default_rng(20261028)
    out = []
    obj = _outer_3d()
    margin = 24                               # 角点离画面边缘至少这么多像素
    for i, (u, v) in enumerate(targets):
        # 平面内旋转用**固定小角度序列**而不是随机 ±180°：板 470×295，转 45°
        # 会把对角对准垂直方向，外接框涨到 1.4 倍，四角姿态立刻出画
        # （实测随机旋转时 5/12 出画 —— 那是测姿问题，检测器背锅）。
        # ⚠ **倾斜必须够大，这是本测试最容易写错的地方**。
        # 一开始只给 ±8° 倾斜，结果 fx=9518（真值 1663，误差 472%）而 RMS 只有
        # 0.53 px —— 所有姿态都近乎正对相机时，板放大两倍与相机靠近一半的投影
        # 无法区分（焦距-距离歧义），`calibrateCamera` 会把 f 和 z 一起推到错处，
        # 而重投影残差照样很小。**这正是"RMS 好看 ≠ 标定正确"的现场演示。**
        # 真实标定同样要求板倾斜 30–45°，否则内参不可辨。
        tilt = ((30, 0), (-30, 0), (0, 30), (0, -30), (0, 0),
                (25, 25), (-25, -25), (25, -25), (-25, 25),
                (30, 10), (-20, 25), (15, -30))[i % 12]
        rz_deg = (0, 15, -15, 25, -25, 0, 20, -20, 10, -10, 22, -22)[i % 12]
        rvec = np.array([np.deg2rad(tilt[0]), np.deg2rad(tilt[1]),
                         np.deg2rad(rz_deg)])
        # **自动退让**：手算位置永远差一点（透视会把靠近的一边放大），
        # 所以出画就按步长往后挪，直到四角全部落在画面内。
        # ⚠ 板原点在**左下角**（object_points 从 0 开始编号），板中心在
        # ((cols-1)/2·sq, (rows-1)/2·sq)。不补这个偏移，板中心会比目标像素
        # 偏出约 200 px（实测 +213/+105），四角姿态全部掉进中央区域。
        R, _ = cv2.Rodrigues(rvec)
        center_mm = np.array([(COLS - 1) * SQ_MM / 2.0,
                              (ROWS - 1) * SQ_MM / 2.0, 0.0])
        tvec = None
        for z in (700.0, 780.0, 860.0, 950.0, 1050.0, 1160.0):
            cand = np.array([(u - K_TRUE[0, 2]) * z / K_TRUE[0, 0],
                             (v - K_TRUE[1, 2]) * z / K_TRUE[1, 1], z]) \
                   - R @ center_mm
            pts = cv2.projectPoints(obj, rvec, cand, K_TRUE, D_TRUE)[0].reshape(4, 2)
            if (pts[:, 0].min() >= margin and pts[:, 1].min() >= margin
                    and pts[:, 0].max() <= W - margin
                    and pts[:, 1].max() <= H - margin):
                tvec = cand
                break
        if tvec is None:                      # 理论上到不了；到了就说明参数要重新算
            tvec = np.array([(u - K_TRUE[0, 2]) * 1160.0 / K_TRUE[0, 0],
                             (v - K_TRUE[1, 2]) * 1160.0 / K_TRUE[1, 1], 1160.0])
        out.append((rvec, tvec))
    return out


def _render(out_dir: Path, count: int) -> list[Path]:
    """把 ``count`` 个姿态渲染成 PNG，返回路径列表。"""
    board = _board_image()
    obj = _outer_3d()
    src = np.array([[0, 0], [board.shape[1], 0],
                    [board.shape[1], board.shape[0]], [0, board.shape[0]]],
                   np.float32)
    paths: list[Path] = []
    for i, (rvec, tvec) in enumerate(_poses()[:count]):
        img_pts, _ = cv2.projectPoints(obj, rvec, tvec, K_TRUE, D_TRUE)
        m = cv2.getPerspectiveTransform(src, img_pts.reshape(4, 2).astype(np.float32))
        canvas = np.full((H, W), 210, np.uint8)
        out = cv2.warpPerspective(board, m, (W, H), dst=canvas,
                                 borderMode=cv2.BORDER_TRANSPARENT)
        p = out_dir / f"view_{i:02d}.png"
        cv2.imwrite(str(p), out)
        paths.append(p)
    return paths


# --------------------------------------------------------------------------- #
def test_recover(tmp: Path) -> list[tuple[str, bool]]:
    """12 个姿态 → 回收内参，fx 相对误差 ≤ 0.5%。"""
    print("----- 1. 合成 12 姿态回收内参 -----")
    paths = _render(tmp, 12)
    checks: list[tuple[str, bool]] = []
    try:
        calib = calibrate_mono(paths, SPEC)
    except MonoCalibError as exc:
        print(f"  [不达标] 标定报错：{exc}")
        return [("12 姿态能标出来", False)]

    rel = abs(calib.fx - K_TRUE[0, 0]) / K_TRUE[0, 0]
    print(f"  fx = {calib.fx:.1f}（真值 {K_TRUE[0, 0]:.1f}，相对误差 {rel:.3%}）")
    print(f"  fy = {calib.fy:.1f}  RMS = {calib.rms_px:.3f} px  视角 {calib.views}")
    print(f"  覆盖 {calib.pose_coverage.get('count')}/5 个区域")
    checks.append(("fx 相对误差 ≤ 0.5%", rel <= 0.005))
    checks.append(("有效视角 ≥ 8", calib.views >= QUALITY["min_views"]))
    checks.append(("覆盖 ≥ 4/5 区域",
                   calib.pose_coverage.get("count", 0) >= QUALITY["min_regions"]))
    checks.append(("RMS 在门限内", calib.rms_px <= QUALITY["rms_px_max"]))

    # 存 / 读往返
    f = tmp / "mono.json"
    save_mono(calib, f)
    back = load_mono(f)
    checks.append(("存读往返不丢 fx",
                   back is not None and abs(back.fx - calib.fx) < 1e-9))
    return checks


def test_too_few(tmp: Path) -> list[tuple[str, bool]]:
    """3 个姿态必须报错（不是"给个凑合的结果"）。"""
    print("----- 2. 姿态不够必须报错 -----")
    paths = _render(tmp / "few", 3)          # 目录由 main() 建好
    try:
        calibrate_mono(paths, SPEC)
        print("  [不达标] 3 个姿态竟然标定成功了 —— 判定被短路")
        return [("3 姿态必须抛 MonoCalibError", False)]
    except MonoCalibError as exc:
        print(f"  按预期报错：{exc}")
        return [("3 姿态必须抛 MonoCalibError", True)]


def test_bad_spec() -> list[tuple[str, bool]]:
    """畸形 BoardSpec 必须在构造时就被拦住（P0a 的守卫）。"""
    print("----- 3. 畸形板参数 -----")
    try:
        BoardSpec(cols=2, rows=5, square_mm=25.0)
        print("  [不达标] cols=2 竟然构造成功")
        return [("BoardSpec(cols=2) 抛 ValueError", False)]
    except ValueError as exc:
        print(f"  按预期报错：{exc}")
        return [("BoardSpec(cols=2) 抛 ValueError", True)]


def test_guard_is_live(tmp: Path) -> list[tuple[str, bool]]:
    """反向自证：把覆盖要求抬到不可能满足 → 必须抛错。

    没有这一条，前面"12 姿态标定成功"可能只是判定被短路了（比如 min_regions
    被写死成 0）。抬高门限后仍然成功，就说明判定根本没在跑。
    """
    print("----- 4. 反向自证：判定真的在跑 -----")
    paths = _render(tmp / "live", 12)        # 目录由 main() 建好
    try:
        calibrate_mono(paths, SPEC, min_regions=QUALITY["regions_total"] + 1)
        print("  [不达标] 门限抬到 6 个区域仍然通过 —— 判定没在跑")
        return [("抬高覆盖门限必须失败", False)]
    except MonoCalibError as exc:
        print(f"  按预期失败：{str(exc)[:70]}")
        return [("抬高覆盖门限必须失败", True)]


def main() -> int:
    print("=" * 74)
    print("单目内参标定自测（合成棋盘，不需要相机）")
    print("=" * 74)
    results: list[tuple[str, bool]] = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        (tmp / "few").mkdir(exist_ok=True)
        (tmp / "live").mkdir(exist_ok=True)
        results += test_recover(tmp)
        results += test_too_few(tmp)
        results += test_bad_spec()
        results += test_guard_is_live(tmp)

    print()
    ok = 0
    for name, passed in results:
        print(f"  [{'通过' if passed else '不达标'}] {name}")
        ok += 1 if passed else 0
    failed = len(results) - ok
    print(f"\n合计 {len(results)} 项：通过 {ok}，不达标 {failed}")
    print("判定结果：全部通过" if not failed else "判定结果：有不达标项")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
