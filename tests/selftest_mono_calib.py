"""单目内参标定自测：合成棋盘回收内参 + **判定真的会红**。

为什么这个测试必须存在（2026-10-27 计划书 §3.5）
------------------------------------------------
单目内参是"看起来对但可能错"的典型：RMS 0.3 px 的结果也可能因为图不够、
板没摆开而把畸变系数估歪，而那时**界面上没有任何征兆**。所以判据不能只测
"能标出来"，还要测：

1. **回收精度**：已知 ``K`` 的合成图 → 标定 → ``fx`` 相对误差 ≤ 0.5%，
   且留一法 fx 极差在警告档内（正常多姿态数据必须自己就稳）；
2. **不够就报错**：3 个姿态必须抛 ``MonoCalibError``，不许给"勉强能用"的结果；
3. **畸形输入**：``BoardSpec(cols=2)`` 必须 ``ValueError``（P0a 的守卫）；
4. **判定真的在跑**：把 ``min_regions`` 抬到不可能满足的值 → 必须抛错。
   这一条是**反向自证** —— 没有它，前三条的"通过"可能只是判定被短路了。
5. **焦距判据**（2026-10-29 加）：同一姿态纯平移的 12 个视角 → 必须报
   "焦距不稳定"；把 ``fx_spread_max`` 调到 99 后同一组必须**通过**（证明拒它的
   只有这一条）；再查门限本身自洽。这一组数据的特点很刺眼：RMS 0.177 px、
   覆盖 5/5、视角 12 个，**旧判据会放行**，而 fx 错 93 倍。
6. **多会话守卫**（2026-10-29 加）：目录里混了 3 次采集（时间戳相差 10 min）
   → 必须拒；``--session last`` / ``--session <时间>`` 必须精确选出一组；
   单会话目录不许要人指定；把阈值调到 1e9（关掉守卫）后不许再切分。

合成方式：先画一张正面棋盘，再用已知 ``K`` 把它的四个外角投影到目标姿态，
``warpPerspective`` 贴到空白画布上。这样得到的图**带有真实的透视与畸变**，
检测器（SB）能按正常流程检出 40 个内角点 —— 实测确认过，不是假设。
"""

from __future__ import annotations

import shutil
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

from calib import mono_calib as mc                      # noqa: E402
from calib.board import BoardSpec                       # noqa: E402
from calib.mono_calib import (MonoCalibError, QUALITY,  # noqa: E402
                              calibrate_mono, load_mono, save_mono, session_id)

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


def _planar_translation_poses() -> list[tuple[np.ndarray, np.ndarray]]:
    """12 个**同一姿态、只平移**的视角：这是"焦距定不出来"的现场。

    板始终正对相机（``rvec = 0``）、距离也固定，只改它在画面里的位置。于是
    "板放大两倍"与"相机靠近一半"在投影上**完全无法区分**（焦距-距离歧义），
    无数个 ``(f, z)`` 解给出同样的重投影。

    实测（2026-10-29，真值 fx=1663）：标出来 **fx=155732**（错 93 倍），
    **而 RMS 只有 0.177 px、姿态覆盖 5/5、视角 12 个** —— 旧判据（RMS ≤ 0.5）
    会**放行**它，然后所有深度都错 93 倍。这就是"RMS 好看 ≠ 内参可用"的实证，
    也是留一法 fx 极差这条判据存在的全部理由。
    """
    targets = [(x * W, y * H) for x, y in (
        (0.30, 0.30), (0.70, 0.30), (0.30, 0.70), (0.70, 0.70),
        (0.50, 0.50), (0.50, 0.30), (0.50, 0.70), (0.30, 0.50),
        (0.70, 0.50), (0.42, 0.42), (0.58, 0.58), (0.50, 0.38),
    )]
    center_mm = np.array([(COLS - 1) * SQ_MM / 2.0,
                          (ROWS - 1) * SQ_MM / 2.0, 0.0])
    z = 900.0
    out = []
    for u, v in targets:
        rvec = np.zeros(3)                     # 正对：不给任何倾斜
        tvec = np.array([(u - K_TRUE[0, 2]) * z / K_TRUE[0, 0],
                         (v - K_TRUE[1, 2]) * z / K_TRUE[1, 1], z]) - center_mm
        out.append((rvec, tvec))
    return out


def _render(out_dir: Path, count: int,
            poses: list[tuple[np.ndarray, np.ndarray]] | None = None) -> list[Path]:
    """把 ``count`` 个姿态渲染成 PNG，返回路径列表。

    ``poses`` 缺省用 :func:`_poses`（正常多姿态）；传 :func:`_planar_translation_poses`
    就得到"同一姿态平移"的退化组。
    """
    board = _board_image()
    obj = _outer_3d()
    src = np.array([[0, 0], [board.shape[1], 0],
                    [board.shape[1], board.shape[0]], [0, board.shape[0]]],
                   np.float32)
    paths: list[Path] = []
    for i, (rvec, tvec) in enumerate((poses or _poses())[:count]):

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
    checks.append(("RMS 在『干净』档（≤ warn）", calib.rms_px <= QUALITY["rms_px_warn"]))
    # 这条是新判据的**正向**面：正常多姿态数据必须自己就稳（否则判据太严，
    # 会把好数据也挡掉 —— 一个只会拒的判据和不存在的判据一样没用）。
    print(f"  留一法 fx 极差 {calib.fx_spread:.2%}（{calib.fx_spread_mode}）")
    checks.append(("留一法 fx 极差 ≤ 3%（warn 档内）",
                   0.0 <= calib.fx_spread <= QUALITY["fx_spread_warn"]))

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


def test_fx_gate(tmp: Path) -> list[tuple[str, bool]]:
    """焦距判据：**同一姿态平移**必须被拒，且拒它的只能是这一条。

    三重断言，缺一条这个判据就是"看起来在跑"：
    ① 退化数据默认必须抛 ``MonoCalibError``，且消息里出现"焦距不稳定"
       （错误信息必须点真实原因 —— 旧文案说"图糊、板太小或角点检错"，
       而这三条在实测里**都不成立**）；
    ② 把 ``fx_spread_max`` 调到 99（等于关掉这条判据）后，同一组数据必须
       **通过**：这证明拒它的既不是 RMS、也不是覆盖、也不是视角数 —— 是这一条；
    ③ 门限本身自洽（警告 < 否决、单位量纲不写错）。②③ 是**反向自证**：
       没有它们，①的"通过/拒绝"可能只是判定被短路了。
    """
    print("----- 5. 焦距判据：同一姿态平移必须被拒 -----")
    paths = _render(tmp / "flat", 12, poses=_planar_translation_poses())
    checks: list[tuple[str, bool]] = []

    # ① 必须拒，且说得出真实原因
    try:
        bad = calibrate_mono(paths, SPEC)
        print(f"  [不达标] 同一姿态平移竟然通过：fx={bad.fx:.0f}"
              f"（真值 {K_TRUE[0, 0]:.0f}），RMS {bad.rms_px:.3f}")
        checks.append(("同一姿态平移必须报「焦距不稳定」", False))
    except MonoCalibError as exc:
        msg = str(exc)
        print("  按预期报错：" + msg.splitlines()[0])
        checks.append(("同一姿态平移必须报「焦距不稳定」", "焦距不稳定" in msg))

    # ② 关掉这条判据后必须通过（否则①的拒绝可能来自别的原因）
    try:
        loose = calibrate_mono(paths, SPEC, fx_spread_max=99.0)
        print(f"  关掉判据后：通过 fx={loose.fx:.0f} RMS={loose.rms_px:.3f} "
              f"spread={loose.fx_spread:.1%} 覆盖={loose.pose_coverage.get('count')}/5")
        sole = (loose.rms_px <= QUALITY["rms_px_max"]
                and loose.pose_coverage.get("count", 0) >= QUALITY["min_regions"]
                and loose.views >= QUALITY["min_views"]
                and loose.fx_spread > QUALITY["fx_spread_max"])
        checks.append(("关掉 fx 判据后同一组通过（拒它的只有这一条）", sole))
    except MonoCalibError as exc:
        print(f"  [不达标] 关掉 fx 判据仍被拒：{str(exc).splitlines()[0]}")
        checks.append(("关掉 fx 判据后同一组通过（拒它的只有这一条）", False))

    # ③ 门限自洽。**不能写 fx_spread_max > rms_px_max**：两者量纲不同
    #    （0.08 是"8% 的比例"，0.8 是"0.8 像素"），8% > 0.8 px 没有意义。
    #    真正会写错的是"把 8% 写成 8（而不是 0.08）"，所以查量纲与档位顺序。
    fx_warn = float(QUALITY["fx_spread_warn"])
    fx_max = float(QUALITY["fx_spread_max"])
    rms_warn = float(QUALITY["rms_px_warn"])
    rms_max = float(QUALITY["rms_px_max"])
    order_ok = 0 < rms_warn < rms_max and 0 < fx_warn < fx_max
    unit_ok = (0.0 < fx_max < 1.0) and (0.1 < rms_max < 5.0)
    print(f"  门限：RMS 警告 {rms_warn} < 否决 {rms_max} px；"
          f"fx 警告 {fx_warn:.0%} < 否决 {fx_max:.0%}；fx 用小数、RMS 用像素")
    checks.append(("门限档位有序（警告 < 否决，两条各自）", order_ok))
    checks.append(("门限量纲正确（fx 是 0~1 小数，RMS 是 px）", unit_ok))
    return checks


def _copy_session(d: Path, srcs: list[Path], start_ms: int, count: int) -> list[Path]:
    """把 ``srcs`` 前 ``count`` 张复制成"另一次采集"的文件名（时间戳每张 +2 s）。

    只碰文件名、不重新渲染：会话切分只依赖 **文件名里的毫秒时间戳**，这条测试
    要验的正是那个解析，而不是再跑一遍 15 张全分辨率的检测（那要 5 s/次）。
    """
    out: list[Path] = []
    for i in range(count):
        p = d / f"shot_{i + 1:02d}_{start_ms + i * 2000}.png"
        shutil.copyfile(srcs[i], p)
        out.append(p)
    return out


def test_session_guard(tmp: Path) -> list[tuple[str, bool]]:
    """多会话守卫：混了多次采集的目录必须拒，``--session`` 能挑出一次。

    为什么必须有（2026-10-29 实测）：``outputs/mono_calib_frames`` 里实际躺着
    4 次采集、跨两天。60 张混标 → fx=1369、留一法只飘 **1.8%**（"看起来很稳"），
    而分批真实值是 1268/1268/1380/1388，**同一台机器上差 15%**。混标的"稳"是把
    4 组不同位姿平均出来的假象 —— 用户下次点"开始标定"就会踩到它。

    用例设计：3 组时间戳相隔 10 min（> 阈值 60 s），各 12/10/9 张，图**内容**复用
    同一批合成棋盘（会话判定只看时间戳）。第 ⑤ 条是反向自证：把阈值调到 1e9
    （等于关掉守卫）后同一目录不该再被切分 —— 否则前面"被拒"可能来自别的原因。
    """
    print("----- 6. 多会话守卫 -----")
    src_dir = tmp / "sess_src"
    src_dir.mkdir(exist_ok=True)             # _render 不建目录（调用方负责）
    src = _render(src_dir, 12)
    d = tmp / "sess_mix"
    d.mkdir(exist_ok=True)
    t0 = 1793272790000                      # 任取一个 13 位毫秒时间戳作基准
    a = _copy_session(d, src, t0, 12)
    b = _copy_session(d, src, t0 + 600_000, 10)          # +10 min
    c = _copy_session(d, src, t0 + 1_200_000, 9)         # +20 min
    everything = a + b + c
    checks: list[tuple[str, bool]] = []

    def _pick(want):
        """选一次采集；选不出来返回 None（**不抛** —— 守卫被关掉时这条要能红，
        而不是把整个自测炸成 traceback）。"""
        try:
            return mc._pick_session(everything, want)
        except MonoCalibError as exc:
            print(f"    选会话失败：{str(exc).splitlines()[0]}")
            return None

    # ① 默认必须拒，且消息里说清"3 次采集"
    try:
        calibrate_mono(everything, SPEC)
        print("  [不达标] 3 次采集混在一起竟然标定成功")
        checks.append(("3 次采集必须被拒", False))
    except MonoCalibError as exc:
        msg = str(exc)
        print("  按预期报错：" + msg.splitlines()[0])
        for line in msg.splitlines()[1:3]:
            print("    " + line)
        checks.append(("3 次采集必须被拒（消息含『3 次采集』）", "3 次采集" in msg))

    # ② --session last：选出最后一组的 9 张（先查选了哪些，再查真标出来的视角数）
    picked = _pick("last")
    same = picked is not None and [p.name for p in picked] == [p.name for p in c]
    try:
        last = calibrate_mono(everything, SPEC, session="last")
        print(f"  --session last：选中 {len(picked or [])} 张，标出 {last.views} 个视角")
        checks.append(("--session last 只算最后一次采集", same and last.views == 9))
    except MonoCalibError as exc:
        print(f"  [不达标] --session last 被拒：{str(exc).splitlines()[0]}")
        checks.append(("--session last 只算最后一次采集", False))

    # ③ 按起始时间选**中间**那组（10 张）——用路径比对，比只看 views 更硬
    sid = session_id(t0 + 600_000)
    picked_mid = _pick(sid)
    mid_ok = picked_mid is not None and [p.name for p in picked_mid] == \
        [p.name for p in b]
    print(f"  --session {sid}：选中 {len(picked_mid or [])} 张（中间那组 10 张）")
    checks.append((f"--session {sid} 选中中间那次采集", mid_ok))

    # ④ 单会话目录（10 张、跨度 18 s）不该要人指定
    try:
        one = calibrate_mono(b, SPEC)
        print(f"  单会话目录：{one.views} 个视角，直接标定成功")
        checks.append(("单会话目录自动通过（不用 --session）", one.views == 10))
    except MonoCalibError as exc:
        print(f"  [不达标] 单会话目录被拒：{str(exc).splitlines()[0]}")
        checks.append(("单会话目录自动通过（不用 --session）", False))

    # ⑤ 反向自证：阈值调到 1e9（= 关掉守卫）后，同一目录不该再被切分
    old = mc.SESSION_GAP_S
    mc.SESSION_GAP_S = 1e9
    try:
        loose = mc._pick_session(everything, None)
        print(f"  关掉守卫（阈值 1e9）：一次选中 {len(loose)} 张（= 3 组全混）")
        checks.append(("关掉守卫后不再切分（拒它的是守卫）", len(loose) == len(everything)))
    finally:
        mc.SESSION_GAP_S = old
    return checks


def main() -> int:
    print("=" * 74)
    print("单目内参标定自测（合成棋盘，不需要相机）")
    print("=" * 74)
    results: list[tuple[str, bool]] = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        (tmp / "few").mkdir(exist_ok=True)
        (tmp / "live").mkdir(exist_ok=True)
        (tmp / "flat").mkdir(exist_ok=True)
        results += test_recover(tmp)
        results += test_too_few(tmp)
        results += test_bad_spec()
        results += test_guard_is_live(tmp)
        results += test_fx_gate(tmp)
        results += test_session_guard(tmp)

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
