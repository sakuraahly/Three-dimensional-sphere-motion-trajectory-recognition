"""chain 标定的输入自检：**拍完先跑这个**，别等标定跑完才发现图不行。

为什么需要它（2026-10-28 的真实事故）
------------------------------------
当天用 90° 装置跑 chain，等到标定结束才看到「左目 RMS 2.18 px 超标 / 参考图
位姿离散 11.83°」—— 那意味着**十几分钟的拍摄全部作废**。而这两个问题在
拍完的那一刻就能查出来：

1. **内参段的姿态覆盖**：板一直放在画面中央时，15 张图只覆盖 5 个区域里的
   1 个。畸变系数不可辨，内参与主点/畸变互相耦合，RMS 必然大。
2. **参考段的板有没有动**：单板模式的前提是"参考图期间板**不动**"（T_AB=I）。
   板挪了 11° 时两路的"同一个位置"其实不是同一个位置，链出来的外参是错的。
3. **双板模式的第 ③ 步有没有被挪**（2026-11-11 补）：参考对各对之间同一角点
   位移超过 :data:`REF_STILL_PX` → 装置在拍 ③ 时动了，链两端位姿不再是同一个
   支架摆放。14:39 实测 188/315 px，而旧版自检只数"N 对、N 对检出整板"，
   判"全部通过"—— **假绿**（标定随后报参考对离散 9.656°/85.49 mm）。
4. **逐图角点质量**（2026-11-11 补）：单应几何残差 + 清晰度。14:39 左目
   8/30 张几何残差 >2 px（最大 4.69 px），左目 RMS 1.899 正是被它们拖高的。
   1~3 张只警告（不判死整段），超过 1/3 才算"必须修"。

两种模式（2026-11-04 加 `--mode dual`）
---------------------------------------
``--mode single``（默认）：``left / right / ref_left / ref_right`` 四段。
``--mode dual``：``left / board_b / ref_pair_left / ref_pair_right / pair`` 四步
（2026-11-09 方案 A 加了**参考对**）。两块刚性固定的板各看一块（垂直双板）：

* ①② `left` / `board_b` 是内参段，板可以随便挪；
* ③ `ref_pair_left` / `ref_pair_right` 要求组合装置**摆好别动**、两路同时各拍
  3 张（GUI 默认；多拍几对便于评估重复性）—— 链上两端的板位姿必须来自同一个
  支架摆放，否则外参会静默错到
  10° / 28 mm 量级（合成真值实测）。这一步不能省，也不能拿内参段的图凑；
* ④ `pair` 是**一台相机同时拍到 A、B 两板**的同框图（求 T_AB）。T_AB 由同框图
  实测、不靠 I，与摆放无关，因此不要求和参考对是同一次拍摄。

早期版本这里写的是"双板模式没有参考段板不动这条、只有三段"——那是错的，
`docs/交接文档.md` §67.1 有同样的错误结论，已一并更正。

本工具只读图片、不写任何文件、不需要相机，也不改标定。

用法::

    python apps/tools/check_chain_inputs.py --root outputs/chain/20261028-135416
    python apps/tools/check_chain_inputs.py --root <目录> --cols 8 --rows 5 --square 25
    python apps/tools/check_chain_inputs.py --root outputs/chain_frames --mode dual

退出码：0 = 可以跑 chain；1 = 有必须先修的问题；2 = 目录/参数不对。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# 按**标记**上溯到含 `sphere_tracker/` 的祖先，不数层数：
# 数层数（parent.parent.parent）在文件被移动、或有人加一层包装目录时会静默指错，
# 而 `selftest_entrypoints` 对这条是硬的（本项目明确的规矩）。
ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "sphere_tracker").is_dir())
sys.path.insert(0, str(ROOT))

from calib.board import (                                 # noqa: E402
    BoardSpec, detect_board, detect_two_boards, pose_distance)
from calib.board_coverage import (                        # noqa: E402
    REF_STILL_PX, coverage_regions, reference_shift)
from calib.capture import focus_gate                      # noqa: E402
from calib.mono_calib import QUALITY                      # noqa: E402

# `REF_STILL_PX` 与两条判据本身都住在 `calib/board_coverage.py` ——
# 采集窗口的 HUD 直接调同一份，这里不再定义任何数字。
# 本文件只保留"打印成给人看的话"这件事。

#: chain 的**硬门槛**：`apps/calibrate_stereo.py` 的 `_run_chain` 只在
#: `len(c_left) < 3 or len(c_right) < 3` 时拒绝。少于这个数**标定跑不了**，
#: 所以本工具也按 3 判"不能跑"。
MIN_VIEWS_HARD = 3
#: **建议**张数，不是门槛：少于它标定仍然能跑，但内参精度会差 ——
#: 而 chain 的外参完全押在内参上（`T_AB = I`，外参误差 ≈ 内参误差 × 基线），
#: 所以少了要**警告**，不是拒绝。
#:
#: 原来这里写的是 `MIN_VIEWS_PER_SIDE = 15`，注释还写"与 calibrate_stereo 的
#: chain 门槛一致"——**不准确**：那边代码里的门槛是 3，15 只是提示文案里的建议。
#: 同一个概念项目里有三个数，别搞混：
#:   3  = chain 硬门槛（本文件 `MIN_VIEWS_HARD`）
#:   8  = `calib/mono_calib.QUALITY["min_views"]`，单目标定用
#:   15 = 建议张数（本文件 `MIN_VIEWS_ADVICE`）
#: 混用后果很具体：拍了 12 张、覆盖 5/5 的用户会被本工具判"不能跑"，
#: 而标定其实能跑 —— 于是白补拍一轮。
MIN_VIEWS_ADVICE = 15

#: 双板模式下"两块板在图上要能分开"的门限，单位是
#: `calib.board.pose_distance` 的返回值（归一化到图像对角线的无量纲距离）。
#:
#: 取 0.02 —— **故意很保守**：它只想拦住"两块板在图上几乎重合/平行到肉眼都
#: 像一块"这种必然让 `_assign_ab` 分不清的拍摄。正常的垂直双板（两板朝向差
#: 约 90°）实测会远大于它。这里**不**照着 `_assign_ab` 的 5° 角度门限换算：
#: 那需要内参做 PnP，而自检发生在标定之前、还没有 K，两个量纲不可混。
PAIR_MIN_POSE_DIST = 0.02
#: 内参段建议的倾斜：只用角点外接框的"短边/长边"比粗略判断有没有斜
#: （不做 PnP，因为这里还没有内参可依赖）

#: 单图"角点可不可信"的**几何**判据（2026-11-11 自检假绿修复）。
#:
#: 棋盘是**平面**：无论板怎么摆，40 个角点都该被同一个单应精确拟合。所以
#: "最佳单应拟合残差"只反映角点本身的抖动/拖影/反光，与板摆在哪、离多远无关，
#: 也不需要内参 —— 这是自检阶段唯一能算的"这张图角点质量"。
#:
#: 门限 = max(绝对 2 px, 本批中位数的 3 倍)：2 px 是"角点定位明显不可信"的线；
#: 相对项防止整批都带轻微噪声时一次点掉半批图。14:39 实测中位 0.64 px、
#: 8/30 张 > 2 px（最大 4.69 px），排序与 mono 逐视图 RMS 一致。
FRAME_GEO_BAD_PX = 2.0
FRAME_GEO_BAD_RATIO = 3.0


def _grid_residual(obs, spec: BoardSpec) -> float:
    """角点相对"最佳单应"的 RMS 残差（px）—— 平面棋盘的单图几何自洽性。

    不需要内参：平面到图像的任意刚体摆放都是一个投影变换，所以理想检测下
    残差应为 0；拖影/反光/误检会让它升高。返回 ``nan`` 表示拟合失败。
    """
    obj_xy = np.asarray(spec.object_points(), np.float64)[:, :2]
    pts = np.asarray(obs.corners, np.float64).reshape(-1, 2)
    H, _ = cv2.findHomography(obj_xy, pts, method=0)
    if H is None:
        return float("nan")
    proj = cv2.perspectiveTransform(
        obj_xy.reshape(-1, 1, 2), H).reshape(-1, 2)
    err = pts - proj
    return float(np.sqrt(np.mean(np.sum(err * err, axis=1))))


def _load_gray(path: Path):
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def _corners(path: Path, spec: BoardSpec):
    g = _load_gray(path)
    if g is None:
        return None, None, "读不出"
    obs = detect_board(g, spec)
    if not obs.found or obs.corners is None:
        return None, None, f"未检出（{obs.note or '未知'}）"
    return g.shape, obs, ""


def check_intrinsics(d: Path, spec: BoardSpec, side: str,
                     warnings: list[str] | None = None) -> tuple[list[str], bool]:
    """检查一路内参图：张数、检出、覆盖、坏图。返回（问题列表, 是否可接受）。

    ``warnings`` 是可选的收集列表：**1~3 张质量偏低的图不该把整段判死**，
    但也绝不能静默绿 —— 这些图会拖高内参 RMS（14:39 左目逐视图 1.96~2.19 px
    的图就是这么来的），所以列成警告，由人决定删掉/补拍。
    """
    warns = warnings if warnings is not None else []
    problems: list[str] = []
    if not d.is_dir():
        return [f"{side}：目录不存在 {d}"], False
    files = sorted(list(d.glob("*.png")) + list(d.glob("*.jpg")))
    if len(files) < MIN_VIEWS_HARD:
        problems.append(f"{side}：只有 {len(files)} 张，chain 标定至少要 "
                        f"{MIN_VIEWS_HARD} 张")
    elif len(files) < MIN_VIEWS_ADVICE:
        # 能跑，但要让人知道代价：内参误差会原样乘进基线（外参 T_AB = I）
        print(f"  [警告] {side}：只有 {len(files)} 张（建议 ≥{MIN_VIEWS_ADVICE}）"
              f"—— 标定能跑，但内参误差会乘进基线")

    views: list[tuple] = []
    quality: list[tuple[str, object]] = []       # 已检出的图，供单图质量判据
    bad: list[str] = []
    areas: list[float] = []
    for p in files:
        shape, obs, why = _corners(p, spec)
        if obs is None:
            bad.append(f"{p.name}（{why}）")
            continue
        views.append((obs.corners.reshape(-1, 2), shape))
        areas.append(obs.area_ratio)
        quality.append((p.name, obs))
        if obs.area_ratio < QUALITY["min_area_ratio"]:
            bad.append(f"{p.name}（板太小 {obs.area_ratio:.1%}）")

    covered, total = coverage_regions(views)
    n_ok = len(files) - len(bad)
    print(f"  {side}: {len(files)} 张，可用 {n_ok}，覆盖 {len(covered)}/{total} 个区域 "
          f"{sorted(covered)}，板占画面 {min(areas):.1%}~{max(areas):.1%}"
          if areas else f"  {side}: {len(files)} 张，全部不可用")
    if len(covered) < QUALITY["min_regions"]:
        problems.append(
            f"{side}：姿态只覆盖 {len(covered)}/{total} 个区域（至少 {QUALITY['min_regions']}）"
            f" —— 板要摆到画面四角与中央，不能一直放中间")
    if bad:
        problems.append(f"{side}：{len(bad)} 张不可用：" + "、".join(bad[:4])
                        + ("…" if len(bad) > 4 else ""))

    # ---- 单图质量（2026-11-11 自检假绿修复）----
    #
    # 以前这里只报覆盖/面积，**逐图角点质量一个字不提** —— 于是 14:39 那种
    # 左目逐视图 RMS 1.96~2.19 px 的坏帧在自检里完全看不见（mono 整体 1.899
    # 是被它们拖上去的）。这里补两条**不需要内参**的逐图判据：
    #   ① 几何一致性（单应残差）：角点抖/拖影/反光；
    #   ② 清晰度（focus 相对本批中位）：与采集端 `focus_gate` 同一判据、同一门限。
    # 1~3 张 → 警告（不判死整段）；超过整段 1/3 → 才算"必须修"。
    if quality:
        geo_pairs = [(_grid_residual(obs, spec), name) for name, obs in quality]
        geos = np.array([g for g, _n in geo_pairs if np.isfinite(g)], np.float64)
        focus_vals = np.array([obs.focus for _n, obs in quality], np.float64)
        med_geo = float(np.median(geos)) if geos.size else float("nan")
        med_focus = (float(np.median(focus_vals[np.isfinite(focus_vals)]))
                     if np.isfinite(focus_vals).any() else float("nan"))
        if geos.size:
            print(f"  {side}: 单图质量 几何残差（单应）中位 {med_geo:.2f} px、"
                  f"最大 {geos.max():.2f} px；清晰度 focus 中位 {med_focus:.0f}")
        geo_bad = sorted(((name, g) for g, name in geo_pairs
                          if np.isfinite(g) and g > max(FRAME_GEO_BAD_PX,
                                                        med_geo * FRAME_GEO_BAD_RATIO)),
                         key=lambda r: -r[1])
        focus_bad = []
        for name, obs in quality:
            ok_f, why_f = focus_gate(obs.focus, label=side, median_focus=med_focus,
                                     n_samples=len(quality))
            if not ok_f:
                focus_bad.append((name, why_f))
        limit = max(3, len(quality) // 3)
        if geo_bad:
            head = "、".join(f"{n}（{g:.1f}px）" for n, g in geo_bad[:4])
            msg = (f"{side}：{len(geo_bad)}/{len(quality)} 张角点几何一致性差：{head}"
                   + ("…" if len(geo_bad) > 4 else "")
                   + " —— 棋盘是平面，这些图的角点连单应都拟合不好（拖影/反光/"
                     "角点抖动），会直接拖高内参 RMS。建议删掉或补拍；"
                     f"其余 {len(quality) - len(geo_bad)} 张仍可标定")
            if len(geo_bad) > limit:
                problems.append(msg + "（坏帧已超过整段 1/3，内参会明显变差，"
                                     "建议整段重拍）")
            else:
                warns.append(msg)
        if focus_bad:
            head = "、".join(f"{n}（{why_f}）" for n, why_f in focus_bad[:3])
            msg = (f"{side}：{len(focus_bad)}/{len(quality)} 张清晰度偏低：{head}"
                   + ("…" if len(focus_bad) > 3 else "")
                   + "。建议删掉或补拍")
            if len(focus_bad) > limit:
                problems.append(msg + "（坏帧超过整段 1/3，内参会明显变差）")
            else:
                warns.append(msg)
    return problems, not problems


def check_reference(d: Path, spec: BoardSpec, side: str) -> tuple[list[str], bool]:
    """检查参考段：板**必须没动**（角点位移 < REF_STILL_PX）。"""
    problems: list[str] = []
    if not d.is_dir():
        return [f"{side}：参考目录不存在 {d}"], False
    files = sorted(list(d.glob("*.png")) + list(d.glob("*.jpg")))
    if not files:
        return [f"{side}：参考目录是空的 {d}"], False
    corners: list[np.ndarray] = []
    for p in files:
        _shape, obs, why = _corners(p, spec)
        if obs is None:
            problems.append(f"{side}：{p.name} {why}")
            continue
        corners.append(obs.corners.reshape(-1, 2))
    if len(corners) < 2:
        problems.append(f"{side}：可用参考图只有 {len(corners)} 张（至少 2 张）")
        return problems, False
    spread, _worst = reference_shift(corners)
    print(f"  {side}: {len(corners)} 张，角点最大位移 {spread:.1f} px "
          f"（门限 {REF_STILL_PX:.0f}）")
    if spread > REF_STILL_PX:
        problems.append(
            f"{side}：参考图之间板动了（角点最大位移 {spread:.0f} px > "
            f"{REF_STILL_PX:.0f}）—— 单板模式要求板在参考段**完全不动**")
    return problems, not problems


def check_ref_pair(root: Path, spec: BoardSpec) -> tuple[list[str], bool]:
    """检查双板模式的**参考对**（第 ③ 步，2026-11-09 方案 A）。

    链上两端的板位姿必须来自**同一个支架摆放**，否则外参会静默错到
    10° / 28 mm 量级。所以除了内参段与同框图，还必须有一组
    「组合装置摆好别动、左右两路同时各拍 3 张（GUI 默认）」的图。

    `capture_board.py --both` 把同一时刻的两张图写成**同名文件**分别落到
    `ref_pair_left/` 与 `ref_pair_right/`，这里按文件名配对；各只放一张、
    名字不同也认（小白不该被文件名卡住）。
    """
    problems: list[str] = []
    dl, dr = root / "ref_pair_left", root / "ref_pair_right"
    for d, side in ((dl, "ref_pair_left"), (dr, "ref_pair_right")):
        if not d.is_dir():
            problems.append(
                f"{side}：目录不存在 {d}\n"
                "     这就是第 ③ 步没拍：把组合装置（A、B 两板固定的支架）"
                "摆好别动，点「采集标定板图」让两路同时各拍 3 张"
                "（GUI 默认；自检会逐对核验）")
    if problems:
        return problems, False

    def _images(d: Path):
        return sorted(list(d.glob("*.png")) + list(d.glob("*.jpg")))

    fl, fr = _images(dl), _images(dr)
    if not fl or not fr:
        problems.append(f"参考对：左 {len(fl)} 张 / 右 {len(fr)} 张 —— "
                        "两边都必须有图（同一时刻各一张）。请补拍第 ③ 步")
        return problems, False

    by_l = {p.name: p for p in fl}
    by_r = {p.name: p for p in fr}
    common = sorted(set(by_l) & set(by_r))
    if common:
        pairs = [(by_l[n], by_r[n]) for n in common]
        how = "同名文件配对"
    elif len(fl) == len(fr) == 1:
        pairs = [(fl[0], fr[0])]
        how = "两边各 1 张，自动配成一对"
    else:
        pairs = []
        how = "没有同名文件"
    if not pairs:
        problems.append(
            f"参考对：左 {len(fl)} 张 / 右 {len(fr)} 张，同名 {len(common)} 对"
            " —— 没有能配成对的图。\n"
            "     两路同时拍时程序会自动写成同名文件；手工整理请把同一时刻"
            "的两张取同一个名字，或两边各只放一张")
        return problems, False

    bad, ok_rows = [], []
    corners_l: list[tuple[str, np.ndarray]] = []
    corners_r: list[tuple[str, np.ndarray]] = []
    for pl, pr in pairs:
        _sl, ol, wl = _corners(pl, spec)
        _sr, orr, wr = _corners(pr, spec)
        if ol is None or orr is None:
            bad.append(f"{pl.name}（左 {wl}）/{pr.name}（右 {wr}）")
        else:
            ok_rows.append((pl.name, ol.area_ratio, orr.area_ratio))
            corners_l.append((pl.name, ol.corners.reshape(-1, 2)))
            corners_r.append((pr.name, orr.corners.reshape(-1, 2)))
    print(f"  参考对：{len(pairs)} 对（{how}），{len(ok_rows)} 对两路都检出整板")
    for name, ar_l, ar_r in ok_rows[:5]:
        print(f"    {name}  左板占画面 {ar_l:.1%} / 右板占画面 {ar_r:.1%}")
    if not ok_rows:
        problems.append(
            "参考对：没有一对是**两路都完整检出整块棋盘**的："
            + "、".join(bad[:3]) + ("…" if len(bad) > 3 else "") + "\n"
            "     第 ③ 步要求左相机完整看到 A 板、右相机完整看到 B 板"
            "（只看到一角不算），且装置摆好别动")
    if bad:
        problems.append(f"参考对：{len(bad)} 对有图没检出整板：" + "、".join(bad[:3])
                        + ("…" if len(bad) > 3 else ""))
    if len(ok_rows) == 1:
        print("  [提示] 参考对只有 1 对 —— 够用，但无法评估重复性；"
              "想更稳可以第 ③ 步多拍 1~2 对（非必需）")

    # ---- 参考段板动检测（2026-11-11 自检假绿修复）----
    #
    # 以前这里只数"N 对、N 对检出整板"，**没有比较各对之间板有没有动** ——
    # 于是 14:39 的第 ③ 步（标定实测参考对离散 9.656°/85.49 mm，拍 ③ 时
    # 装置动过）在自检里是"全部通过"。第 ③ 步的前提是同一个支架摆放；
    # 装置一动，链两端位姿就不是同一个 G，外参会静默错到 10°/28 mm 量级。
    # 判据与单板参考段同源：`reference_shift`（同一角点跨对最大位移）+
    # `REF_STILL_PX`，不另造数字。
    moved: list[tuple[str, float]] = []
    for side, items in (("ref_pair_left", corners_l),
                        ("ref_pair_right", corners_r)):
        if len(items) < 2:
            print(f"  {side}: 只有 {len(items)} 对 —— 参考段板动**未评估**"
                  f"（≥2 对才比得出来）")
            continue
        shift, worst = reference_shift([c for _n, c in items])
        worst_no = (worst[1] + 1) if worst else 0
        print(f"  {side}: {len(items)} 对之间同一角点最大位移 {shift:.1f} px"
              f"（门限 {REF_STILL_PX:.0f}；最差 第 {worst_no} 对）")
        if shift > REF_STILL_PX:
            moved.append((side, shift))
    if moved:
        detail = "、".join(f"{s} {v:.0f} px" for s, v in moved)
        problems.append(
            f"参考段（第 ③ 步）板动了：{detail} > 门限 {REF_STILL_PX:.0f} px —— "
            f"拍第 ③ 步时装置在动，链两端板位姿不再是**同一个支架摆放**"
            f"（标定会报参考对离散度大，外参可错到 10°/28 mm 量级）。\n"
            "     下一步：重拍第 ③ 步 —— 组合装置摆好后手离开，三脚架/桌面别碰，"
            "两路同时各拍 3 张。")
    return problems, not problems


def check_pairs(d: Path, spec: BoardSpec) -> tuple[list[str], bool]:
    """检查同框段（**双板模式专用**）：每张要同时检出 A、B 两块板。

    为什么这条是双板模式的命门：`T_AB`（两板刚性相对位姿）**只能**从这种
    "一台相机同时看到两块板"的图里求。检不出两块 → 那张图被跳过；一张都
    检不出 → 标定直接失败（`_run_chain` 报"没有一张图能同时检出 A、B"）。

    两板"姿态距离"用 `calib.board.pose_distance` 算（对**单应**做比较，
    不需要内参——自检发生在标定之前，还没有 K）。它归一化到图像对角线，
    所以是 0~1 量级的无量纲值，不是角度；门限只用来兜住"两块板在图上几乎
    重合/平行"这种必然让 A/B 分不开的拍摄。

    ⚠️ 同框图必须用 **--pair-camera 指定的那一路**拍（GUI 拼的是 left）。
    姿态是**相对相机**的，换一路拍会让 `_assign_ab` 用错姿态簇把 A/B 判反；
    而两板夹角大时那种错误**不会**触发"差 <5°"护栏，会静默给出反向的 T_AB。
    本工具无法从图里判断是哪一路拍的，所以只能把这句话印出来。
    """
    problems: list[str] = []
    if not d.is_dir():
        return [f"pair：同框图目录不存在 {d}"], False
    files = sorted(list(d.glob("*.png")) + list(d.glob("*.jpg")))
    if not files:
        return [f"pair：同框图目录是空的 {d}"], False

    ok_rows: list[tuple[str, float, float, float]] = []
    bad: list[str] = []
    for p in files:
        g = _load_gray(p)
        if g is None:
            bad.append(f"{p.name}（读不出）")
            continue
        found = detect_two_boards(g, spec)
        if found is None:
            bad.append(f"{p.name}（没有同时检出两块板）")
            continue
        a, b = found
        diag = float(np.hypot(g.shape[1], g.shape[0]))
        # homography 可能是 None（极少数检测路径不带单应）：此时跳过距离，只记检出
        dist = (pose_distance(a.homography, b.homography, diag)
                if a.homography is not None and b.homography is not None else float("nan"))
        ok_rows.append((p.name, a.area_ratio, b.area_ratio, dist))

    for name, ar_a, ar_b, dist in ok_rows:
        dist_txt = "—" if not np.isfinite(dist) else f"{dist:.3f}"
        print(f"  pair: {name} 两板面积 {ar_a:.1%}/{ar_b:.1%}，姿态距离 {dist_txt}")
    print(f"  pair: {len(files)} 张，{len(ok_rows)} 张能同时检出两块板")

    if not ok_rows:
        problems.append(
            "pair：没有一张能**同时**检出 A、B 两块板 —— 双板模式求不出 T_AB。\n"
            "     下一步：两块板都完整入画（不贴边、不互相遮挡）、光照均匀、"
            "板之间留开一点距离（填白检测第二块时靠的就是这个间隙）")
    elif len(ok_rows) < 2:
        # 2026-11-09：核心层已改成「双板 n<2 拒绝出数」，自检不能再只警告 ——
        # 否则出现"自检说可以跑、标定却拒绝"的口径不一致（§98.5 的同类病）。
        problems.append(
            "pair：只有 1 张能同时检出两块板 —— **标定会直接拒绝出数**"
            "（T_AB 只有 1 次测量，A/B 刚性固连没有重复性证据）。\n"
            "     至少补到 2 张，建议 3~5 张；拍这几张时两板都别动")
    if bad:
        problems.append(f"pair：{len(bad)} 张不可用：" + "、".join(bad[:4])
                        + ("…" if len(bad) > 4 else ""))
    dists = [d0 for *_r, d0 in ok_rows if np.isfinite(d0)]
    if dists and min(dists) < PAIR_MIN_POSE_DIST:
        problems.append(
            f"pair：有两块板的姿态距离只有 {min(dists):.3f}"
            f"（门限 {PAIR_MIN_POSE_DIST}）—— 它们在图上几乎重合。\n"
            "     这会让标定分不清哪块是 A、哪块是 B（判反了 T_AB 就是逆矩阵，"
            "外参全错）。请让两块板在图上明显朝向不同")
    return problems, not problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="chain 标定输入自检（不写文件、不开相机）")
    ap.add_argument("--root", required=True, help="各段目录的父目录")
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--square", type=float, default=25.0)
    ap.add_argument("--mode", choices=("single", "dual", "device90"), default="single",
                    help="single=单板四段（left/right/ref_left/ref_right）；"
                         "dual=双板四步（left/board_b/ref_pair_left/ref_pair_right/pair）；"
                         "device90=中轴90°装置（left/board_b 两段正对图）")
    ap.add_argument("--pair-camera", choices=("left", "right"), default="left",
                    help="dual 模式第 ④ 步同框图是由哪一路相机拍的（默认 left）")
    args = ap.parse_args(argv)

    root = Path(args.root)
    if not root.is_dir():
        print(f"[错误] 目录不存在：{root}")
        return 2
    try:
        spec = BoardSpec(cols=args.cols, rows=args.rows, square_mm=args.square)
    except ValueError as exc:
        print(f"[参数错误] {exc}")
        return 2

    print("=" * 72)
    print(f"chain 输入自检（{args.mode}）：{root}")
    print("=" * 72)
    # 警告 ≠ 通过：1~3 张坏帧不该判死整段，但也不能静默绿（2026-11-11）。
    warnings: list[str] = []
    if args.mode == "dual":
        print("前提（**程序查不出来，由拍摄者保证**）：")
        print("  · A、B 两板刚性固连在同一支架上；角点编号/原点朝向三次拍摄一致")
        print("  · ③ 参考对时支架一动不动；①② 的移动图只用来求内参")
        print("  · ④ 同框图至少 2 张（只拍 1 张，标定时会拒绝出数）")
        print("  · 棋盘方格边长填真实值 —— 这是整条链唯一的绝对尺度来源")
        print()
        print("-- 内参段 A 板 / B 板（板应摆满画面四角+中央，每个姿态倾斜 30–45°）--")
        p1, ok1 = check_intrinsics(root / "left", spec, "left(A 板)", warnings)
        p2, ok2 = check_intrinsics(root / "board_b", spec, "board_b(B 板)", warnings)
        print("-- 参考对（第 ③ 步：组合装置摆好别动，两路同时各拍 3 张"
              "（GUI 默认；自检会逐对核验））--")
        p4, ok4 = check_ref_pair(root, spec)
        print(f"-- 同框段（必须用 {args.pair_camera} 相机同时拍到 A、B 两板）--")
        p3, ok3 = check_pairs(root / "pair", spec)
        problems = p1 + p2 + p4 + p3
    elif args.mode == "device90":
        print("-- 中轴 90° 正对参考图（两段均需静止且完整检出棋盘）--")
        p1, ok1 = check_reference(root / "left", spec, "left(① A板正对)")
        p2, ok2 = check_reference(root / "board_b", spec, "board_b(② B板正对)")
        problems = p1 + p2
    else:
        print("-- 内参段（板应摆满画面四角+中央，每个姿态倾斜 30–45°）--")
        p1, ok1 = check_intrinsics(root / "left", spec, "left", warnings)
        p2, ok2 = check_intrinsics(root / "right", spec, "right", warnings)
        print("-- 参考段（板在这几张之间必须完全不动）--")
        p3, ok3 = check_reference(root / "ref_left", spec, "ref_left")
        p4, ok4 = check_reference(root / "ref_right", spec, "ref_right")
        problems = p1 + p2 + p3 + p4

    print()
    if warnings:
        # 警告不等于通过：1~3 张坏帧不该判死整段，但**必须**把数字摆出来，
        # 让人决定删掉/补拍；否则就是"自检全绿、标定才发现内参被拖高"。
        print(f"警告（不阻止跑 chain，但先看数字值不值得重拍 / 删图）：")
        for i, msg in enumerate(warnings, 1):
            print(f"  {i}) {msg}")
        print()
    if problems:
        print("必须先修的问题：")
        for i, msg in enumerate(problems, 1):
            print(f"  {i}) {msg}")
        print("\n判定结果：不能继续标定（修完重拍后再自检一次）")
        return 1
    if warnings:
        print(f"可以继续标定 —— 但有 {len(warnings)} 条警告（上面已列数字）")
    else:
        print("全部通过：可以开始标定了")
    print("下一步（手动跑）：")
    if args.mode == "dual":
        print(f"  python apps/calibrate_stereo.py --method chain "
              f"--left-dir {root / 'left'} --board-b-dir {root / 'board_b'} "
              f"--ref-pair-left {root / 'ref_pair_left'} "
              f"--ref-pair-right {root / 'ref_pair_right'} "
              f"--pair-frame {root / 'pair'} --pair-camera {args.pair_camera} "
              f"--cols {args.cols} --rows {args.rows} --square {args.square:g}")
    elif args.mode == "device90":
        print(f"  python apps/calibrate_device90.py "
              f"--left-dir {root / 'left'} --right-dir {root / 'board_b'} "
              f"--intrinsics <内参.npz或目录> "
              f"--cols {args.cols} --rows {args.rows} --square {args.square:g}")
    else:
        print(f"  python apps/calibrate_stereo.py --method chain --same-board "
              f"--left-dir {root / 'left'} --right-dir {root / 'right'} "
              f"--ref-left {root / 'ref_left'} --ref-right {root / 'ref_right'} "
              f"--cols {args.cols} --rows {args.rows} --square {args.square:g}")
    print("\n判定结果：可以跑 chain")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
