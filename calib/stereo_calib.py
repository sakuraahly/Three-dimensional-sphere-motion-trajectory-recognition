"""双目标定的核心计算。

流程（每一步都是为了让结果"能投入生产"）：

  1. 各自单目标定        -> K1,D1 / K2,D2（用 calibrateCameraExtended，能拿到逐图重投影误差）
  2. 双目联合标定         -> R,T（先固定内参，只优化外参，稳）
  3. 整体精化             -> 放开内参一起优化，RMS 通常能再降一半
  4. 离群剔除 + 重算      -> 用"逐对视差误差"挑出坏数据，丢掉后重算，最多迭代几轮
  5. 立体校正             -> R1,R2,P1,P2,Q
  6. 质量验证             -> 重投影 RMS、极线误差、基线长度，给出可用性判定

第 4 步是最容易被忽略、但对最终质量影响最大的一步：
少量模糊或贴边的图会把整体误差抬高一个量级，而它们在几百个角点里"看不出来"。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import metric_check, relative_pose
from .board import BoardSpec

# 联合求解（`stereoCalibrate`）在 angled 下**至少**需要这么多对共视数据。
#
# 依据：外参只有 6 个自由度（3 旋转 + 3 平移），3 对共视数据在代数上"勉强可解"，
# 但宽基线（90°、几十厘米）从零优化时，代价面的谷底极窄 —— 3 对的角点噪声
# 足以把解推到完全不同的局部极小，而重投影 RMS 仍然好看（合成实验里 3 px 的
# 真实误差可以报成 0.24 px）。8 对是"每个自由度至少有一个以上独立观测"的
# 最低经验线；低于它必须拒绝出数，而不是加一条警告继续跑。
MIN_COVISIBLE_PAIRS_FOR_JOINT_ANGLED = 8

# chain 方法的重复性门限（T_AB 多次测量的离散度）
CHAIN_ROT_GOOD_DEG = 0.5      # 优秀
CHAIN_ROT_OK_DEG = 1.5        # 可用
CHAIN_TRANS_GOOD_MM = 2.0
CHAIN_TRANS_OK_MM = 5.0

# 跨路一致性的告警门限：**两个数取同一档**（"可用"下限），与
# `metric_check.stereo_scale_consistency` 的门限同源。
# 混档（5 mm + 0.5°）会把正常装置天天判成坏：实测 α 正确、1 px 角点噪声时
# 旋转中位数已达 0.585°，而 0.5° 在 metric_check 里是"优秀"的门限。
_CROSS_TRANS_WARN_MM = 5.0
_CROSS_ROT_WARN_DEG = 1.5


@dataclass
class CalibConfig:
    """标定配置。默认值按"20-30 对图像、普通 USB 相机"调好的。"""

    # ---- 外参求解方法（2026-10-24 加）----
    #
    # joint：`cv2.stereoCalibrate` 联合求解。前提是**同一块板同时出现在两路**，
    #        只对平行/有充分共视的装置成立。
    # chain：独立单目 + 刚性双板公共参考（T_{C_R←C_L} = T_{C_R←B}·T_{B←A}·T_{A←C_L}）。
    #        不需要双目共视，90° 装置用它。
    method: str = "joint"

    refine_intrinsics: bool = True     # 第 3 步：放开内参整体精化
    rational_model: bool = False       # 用 8 参数畸变模型（视角很广、畸变很大时再开）
    fix_principal_point: bool = False  # 数据很少时固定主点，避免过拟合
    reject_outliers: bool = True       # 第 4 步：离群剔除
    outlier_sigma: float = 3.0         # 超过 中位数 + k*MAD 就判为坏数据
    outlier_min_px: float = 0.5        # 但至少超过这个绝对量才剔，防止误杀
    max_reject_rounds: int = 4
    min_pairs_after_reject: int = 8
    alpha: float = 0.0                 # stereoRectify 的裁剪系数：0=裁掉黑边，-1=保留全部像素

    # 可用性判定门槛（像素）
    good_rms: float = 0.35
    ok_rms: float = 0.80
    good_epipolar: float = 0.25
    ok_epipolar: float = 0.60

    # 板占画面比例下限：低于它**不许判"优秀"**。
    #
    # ## 为什么要有这条（2026-10-18 实测，见审查计划书 A5）
    #
    # 合成真值 + 0.30 px 亚像素噪声、**同样的 RMS**：
    #
    #     板占 1.9%（只在中央）：RMS 0.414 px   Z 误差 σ 5.76 mm  最大 13.7 mm
    #     板占 7.9%（铺满画面）：RMS 0.413 px   Z 误差 σ 1.40 mm  最大  2.8 mm
    #
    # 即 **RMS 对"板太小"几乎免疫，而深度精度差 4 倍**。
    # 机制：板只在中央时，焦距/主点与畸变在数据上强耦合，换个噪声样本
    # fx 就漂 ±9 px；而 `Z = f·B/d` 对焦距是**一阶**敏感，直接变成深度系统误差。
    #
    # 所以"RMS 好看"不能作为高等级判定的充分条件 —— 必须再卡一个
    # "板在画面里够大"的硬条件。数值取 8%：与报告里"建议 10%~30%"同侧，
    # 留一点余量给"刚好 8~10%"的情况。
    min_board_area_for_good: float = 0.08

    expected_baseline_mm: float | None = None   # 若提供，用于检查标定是否离谱
    baseline_tolerance: float = 0.20            # 允许的相对偏差

    def __post_init__(self) -> None:
        # 非法方法名必须**立刻**报错：否则拼错成 "chna" 会悄悄走 joint 分支，
        # 在 90° 装置上跑出一个看起来正常、其实无意义的解。
        if self.method not in ("joint", "chain"):
            raise ValueError(
                f"CalibConfig.method={self.method!r} 非法：只支持 'joint' / 'chain'"
                f"（joint=双目联合求解，需两路共视；chain=独立单目+刚性双板公共参考）"
            )


@dataclass
class StereoCalibrationResult:
    """标定结果 + 质量报告。"""

    ok: bool = False
    K1: np.ndarray | None = None
    D1: np.ndarray | None = None
    K2: np.ndarray | None = None
    D2: np.ndarray | None = None
    R: np.ndarray | None = None
    T: np.ndarray | None = None
    E: np.ndarray | None = None
    F: np.ndarray | None = None
    R1: np.ndarray | None = None
    R2: np.ndarray | None = None
    P1: np.ndarray | None = None
    P2: np.ndarray | None = None
    Q: np.ndarray | None = None
    roi1: tuple = ()
    roi2: tuple = ()
    image_size: tuple[int, int] = (0, 0)

    n_pairs_in: int = 0
    n_pairs_used: int = 0
    rms: float = float("nan")               # stereoCalibrate 返回的整体 RMS
    err_left: float = float("nan")          # 左目重投影 RMS
    err_right: float = float("nan")
    epipolar_rms: float = float("nan")      # 校正后的极线误差（真正决定三维精度）
    epipolar_max: float = float("nan")
    baseline_mm: float = float("nan")
    per_pair_epipolar: list[float] = field(default_factory=list)
    used_index: list[int] = field(default_factory=list)
    rejected_index: list[int] = field(default_factory=list)
    verdict: str = ""
    warnings: list[str] = field(default_factory=list)
    board_desc: str = ""
    # ---- 外参求解方法（2026-10-24 加）----
    method: str = "joint"
    # "stereo_rectify"：R1/R2/P1/P2 由 cv2.stereoRectify 产出；
    # "none_raw"：chain 产物 —— **未做极线校正**，R1=R2=I、P1/P2 只是
    #             "去畸变 + 原坐标系"的投影矩阵，`Q` 为 None。
    #
    # ## 为什么不是 None（2026-10-24 改）
    #
    # 上一版把 chain 的 `P1/P2` 置 None，结果 `calib/params.py:rectify_maps()`
    # 与 `stereo/triangulate.py` 一读就越界 —— "标出来了但三角化用不了"。
    # 90° 下 stereoRectify 本来就不成立（重投影到共面会把图像拉成极端梯形），
    # 但**三角化并不需要共面极线**：只要给出一致的 (K, R, T) 投影矩阵，
    # `cv2.triangulatePoints` 直接可用。所以这里输出"未校正坐标系"的 P。
    #
    # ⚠️ 用它的地方必须知道：**极线不是水平的、视差与深度不是简单反比**，
    #    任何依赖极线水平对齐/视差的逻辑（SGBM、`epipolar_error` 的 y 差口径）
    #    都不适用。
    rectification: str = "stereo_rectify"
    consistency: dict = field(default_factory=dict)   # T_AB 重复性（仅 chain）
    n_views_left: int = 0                              # chain：左路姿态数
    n_views_right: int = 0                             # chain：右路姿态数
    T_AB: np.ndarray | None = None                     # chain：两板刚性相对位姿 T_{B←A}
    # ---- 板占画面比例（2026-10-14 加）----
    # 为什么要有：实测那次标定**板只占画面 1.2%**，角点像素少、定位精度低，
    # 于是 RMS 0.91 px 超过 ok 门限 0.80 —— 而当时的判定只说
    # "换更多角度、保证清晰、铺满画面"，**没说是"离板太近一点"**。
    # 这个数字一直是算得出来的（采集时每次都在算），只是**没有出口**。
    board_area_l: float = float("nan")      # 左图里板的角点包围面积 / 画面面积
    board_area_r: float = float("nan")
    grade: str = ""                         # 优秀 / 可用 / 勉强可用 / 不可用
    # **可执行的下一步建议**（不是"换更多角度"这种没法照做的话）
    suggestions: list[str] = field(default_factory=list)
    # ---- 参数不确定度（2026-10-18 加，审查 A5）----
    #
    # `cv2.calibrateCameraExtended` 会返回 `stdDeviationsIntrinsics`，
    # 而原来 `_calibrate_one()` **把它丢掉了**（`_sdi` 下划线变量）。
    # 那正是"这个焦距有多可信"的答案 —— 丢掉它，就只能靠 RMS 猜，
    # 而 RMS 对"板太小"这种退化几乎免疫（见 `min_board_area_for_good` 的注释）。
    #
    # 顺序同 OpenCV：[fx, fy, cx, cy, 畸变...]，单位与参数一致（像素）。
    std1: np.ndarray | None = None
    std2: np.ndarray | None = None

    # ---- 2026-10-26 新增（只加输出，不改任何数值路径）----
    #
    # 跨路一致性：用解出的 `T_{C_R←C_L}` 把两路看**同一块板同一位置**的板位姿
    # 搬到同一坐标系，看残差有多大。这是 chain 唯一能自查外参的指标 ——
    # 单目 RMS 只说明"各路的板拟合得好"，完全不涉及两路之间的相对关系。
    # 形状：`{"median_mm","p95_mm","rot_deg","n"}`；拿不到位姿时为 None；
    # 计算抛异常时为 `{"error": "..."}`（**不填 0**）。
    cross_check: dict | None = None
    # 外参不确定度（`calib.uncertainty.extrinsic_sigma` 的原样输出，含 `error` 键）。
    # **只对 joint 有意义**：它需要"同一批板同时被两路看到"的同名角点。
    extrinsic_sigma: dict | None = None
    # 交叉验证结果（`calib.validation.cross_validate` 的输出），或 `{"skipped": 原因}`。
    cv: dict | None = None
    # "parallel" / "angled"：深度公式 `Z=f·B/d` 只在平行装置成立，
    # 报告必须知道这件事才能决定要不要印 σ_Z（见 `depth_sigma_mm`）。
    stereo_mode: str = "parallel"

    @property
    def board_area(self) -> float:
        """两路里**较小**的那个比例（双目要两路都够大才有意义）。"""
        vals = [v for v in (self.board_area_l, self.board_area_r) if np.isfinite(v)]
        return float(min(vals)) if vals else float("nan")

    # ---- 不确定度（审查 A5）---------------------------------------------
    def fx_sigma(self) -> float:
        """左目 `fx` 的标准差估计（像素）；拿不到返回 nan。"""
        if self.std1 is None or np.size(self.std1) < 1:
            return float("nan")
        return float(np.ravel(self.std1)[0])

    def depth_applicable(self) -> bool:
        """`Z = f·B/d` 在本份标定上是否成立（决定 σ_Z 能不能算）。

        ## 为什么必须有这道闸（2026-10-26）

        这个公式假设**两光轴平行、极线水平、视差沿 x 方向**：`d` 是
        "同一物点在左右图的 x 差"，只有满足平行假设时才有 `Z = f·B/d`。
        90° 装置（`rectification="none_raw"`）两光轴近似垂直：没有公共的极线方向，
        "视差"这个量根本不存在 —— 此时若照旧代公式，**算出来的是一个数字，
        但它对应不到任何物理量**（基线方向与深度方向已经不同轴）。
        错数比缺数危险，所以这里直接判"不适用"，让报告去写"请看外参不确定度"。

        只有 `stereo_mode == "parallel"`（近平行）且 `rectification` 是
        `stereo_rectify`（真的做过极线校正）时，公式才成立。
        """
        return self.stereo_mode == "parallel" and self.rectification == "stereo_rectify"

    def depth_sigma_mm(self, z_m: float = 0.6) -> float:
        """在距离 `z_m`（米）处的**深度标准差估计**，单位 mm。

        ⚠️ **只在平行装置上成立**：`angled`/`none_raw` 返回 `nan`（理由见
        `depth_applicable()`）。报告会据此改印"不适用"，而不是印一个错数。

        ## 公式与它的边界

        `Z = f·B/d`，两边取相对误差（忽略协方差）：

            (σ_Z/Z)² = (σ_f/f)² + (σ_d/d)²   [+ (σ_B/B)²]

        - `σ_f`：来左目内参的标准差（`calibrateCameraExtended` 的
          `stdDeviationsIntrinsics`，原来被丢弃）；
        - `d = f·B/z`：该距离处的视差（像素）；
        - `σ_d ≈ √2 · rms`：左右两路各贡献一次角点定位误差，近似独立；
        - **不包含 `σ_B`**：`cv2.stereoCalibrate` 不返回 `T` 的方差，
          所以这里**没有**基线项。真要在意基线精度，得用重采样另估。

        ⚠️ 这是**数量级估计**，不是严格置信区间。已知偏差：用合成数据验证时
        它给出的排序正确（板小 → 不确定度大，约 3 倍差异），但绝对值与实测
        离散度不一定吻合（实测里还混有"收敛到更差局部解"的成分）。
        **用途：让"该信到什么程度"有个可比较的数字，而不是当验收门限。**
        """
        # ⚠️ 90°/未校正 -> 直接 nan（见 `depth_applicable()` 的推导）：
        #    这时"视差"不是可测量，公式给不出物理量。
        if not self.depth_applicable():
            return float("nan")
        f = float(self.K1[0, 0]) if self.K1 is not None else float("nan")
        b = float(np.linalg.norm(np.ravel(self.T))) if self.T is not None else float("nan")
        if not (np.isfinite(f) and np.isfinite(b) and b > 0 and z_m > 0):
            return float("nan")
        d = f * b / float(z_m)
        if not np.isfinite(d) or d <= 0:
            return float("nan")
        sd = float(np.sqrt(2.0) * self.rms) if np.isfinite(self.rms) else float("nan")
        sf = self.fx_sigma()
        rel = 0.0
        if np.isfinite(sd):
            rel += (sd / d) ** 2
        if np.isfinite(sf) and f > 0:
            rel += (sf / f) ** 2
        if rel <= 0:
            return float("nan")
        return float(np.sqrt(rel) * z_m * 1000.0)

    @staticmethod
    def _aggregate_line(c: dict) -> str:
        """`T_AB 聚合` 那一行。

        为什么必须印：`T_AB 重复性` 只说"这几帧散得多开"，**不说外参用了哪一帧**。
        聚合之后链上的 `T_AB` 是多帧的鲁棒均值，与"取一帧"是两份不同的结果；
        不写清 n 和剔除数，事后无法判断这份外参到底吃了几帧的信息。
        """
        agg = c.get("aggregate") or {}
        if c.get("used_aggregate"):
            return (f"T_AB 聚合     : n={agg.get('n', 0)}  "
                    f"旋转散布 {agg.get('rot_spread_deg', float('nan')):.3f}°  "
                    f"平移散布 {agg.get('trans_spread_mm', float('nan')):.3f} mm"
                    f"（剔除 {agg.get('n_outliers', 0)} 个离群，"
                    f"实际使用 {agg.get('used', 0)} 帧）")
        if c.get("n", 0) >= 2:
            return ("T_AB 聚合     : 未启用（T_AB = I 的单板模式：I 是定义不是测量值，"
                    "样本只作重复性证据）")
        return "T_AB 聚合     : 未启用（n<2，单帧原样使用）"

    def _cross_check_line(self) -> str:
        """`跨路一致性` 那一行（chain）。拿不到就说**为什么**拿不到，不印 0。"""
        c = self.cross_check
        if c is None:
            return "跨路一致性    : 无法计算（缺少两路同板位姿）"
        if "error" in c:
            return f"跨路一致性    : 无法计算（{c['error']}）"
        if "insufficient" in c:
            # ⚠️ 单帧时这个残差**恒为 0**：外参 T = T_camR_B·T_AB·inv(T_camL_A)
            # 正是用这一对位姿构造的，代回去当然重合。印 0.00 会被读成"验过且完美"，
            # 而真相是"没有第二份独立观测，什么也没验"。
            return f"跨路一致性    : 无法评估（{c['insufficient']}）"
        return (f"跨路一致性    : 中位 {c['median_mm']:.2f} mm / "
                f"P95 {c['p95_mm']:.2f} mm / {c['rot_deg']:.3f} °（n={c['n']}）")

    def _sigma_lines(self) -> list[str]:
        """外参不确定度 + "旧口径 vs 新口径"的 σ_Z 对照。

        为什么要**并列**印两个数：旧口径（`depth_sigma_mm`，只有 σ_f + σ_d）
        一直少一项 `(Z/B)·σ_B`，而短基线近距装置上这一项是主导项
        （65 mm 基线、1 m 处 `Z/B ≈ 15`）。只印新数，读者无从知道
        "以前那份报告低估了多少倍"；只印旧数就是继续骗自己。
        """
        s = self.extrinsic_sigma
        if s is None:
            if self.method == "chain":
                return ["外参不确定度  : 无法计算（chain 两路角点不是同名点，"
                        "σ_B 需要两路共视）"]
            return ["外参不确定度  : 无法计算（未传入共视角点）"]
        if "error" in s:
            # **原样打印**：错误文本本身带"下一步怎么做"，改写就等于扔证据。
            return [f"外参不确定度  : {s['error']}"]
        sb = float(s["sigma_baseline_mm"])
        bm = float(s["baseline_mm"])
        pct = (sb / bm * 100.0) if bm > 0 else float("nan")
        sa = float(s["sigma_angle_deg"])
        dz_new = float(s["depth_sigma_mm_at"](1.0))
        out = [f"外参不确定度  : σ_B = {sb:.3f} mm（基线 {bm:.2f} mm，{pct:.3f}%）、"
               f"σ_θ = {sa:.3f}°、1 m 处 σ_Z ≈ {dz_new:.1f} mm"]
        old = self.depth_sigma_mm(1.0)
        if np.isfinite(old) and np.isfinite(dz_new):
            # 不给"低估 N 倍"这种带方向的结论：旧口径缺 σ_B 是真的，但它同时含
            # σ_f（小板上这一项可以很大），所以新旧谁大要看数据。如实印比值。
            ratio = (old / dz_new) if dz_new > 0 else float("inf")
            out.append(f"  ↳ σ_Z(1 m) 对照：旧口径 {old:.2f} mm（只含 σ_f+σ_d，"
                       f"**无基线项**）→ 新口径 {dz_new:.2f} mm（含 σ_B）；"
                       f"旧/新 = {ratio:.2f}×（>1 = 旧口径偏大，<1 = 旧口径漏掉主导项）")
        else:
            out.append("  ↳ σ_Z(1 m) 对照：无法给出（旧公式在本模式下不适用或 σ 非有限）")
        return out

    def _cv_line(self) -> str:
        """`交叉验证` 那一行。"""
        cv = self.cv
        if cv is None:
            return "交叉验证      : 未运行"
        if "skipped" in cv:
            return f"交叉验证      : 跳过（{cv['skipped']}）"
        if "error" in cv:
            return f"交叉验证      : 失败（{cv['error']}）"
        return (f"交叉验证      : {cv['k']} 折  训练 RMS 中位 {cv['train_rms_median']:.3f} px / "
                f"测试 {cv['test_rms_median']:.3f} px → 比值 {cv['ratio']:.2f}"
                f"（{cv['verdict']}）")

    def _report_chain(self) -> str:
        """chain 的报告：**不写校正后极线 RMS**（90° 下这个数没有几何意义）。

        取而代之的是 chain 真正能自查的三件事：
        `||T||`（与卷尺值对照）、`T_AB` 的重复性、每路单目 RMS。
        """
        c = self.consistency or {}
        lines = [
            "=" * 62,
            "双目标定结果（chain：独立单目 + 刚性双板公共参考）",
            "=" * 62,
            f"标定板        : {self.board_desc}",
            f"图像尺寸      : {self.image_size[0]}x{self.image_size[1]}",
            f"姿态数        : 左 {self.n_views_left} / 右 {self.n_views_right}"
            f"（chain 不要求两路成对，也不要求共视）",
            f"左目重投影 RMS: {self.err_left:.4f} px",
            f"右目重投影 RMS: {self.err_right:.4f} px",
            f"基线 ||T||    : {self.baseline_mm:.2f} mm",
            f"T_AB 重复性   : n={c.get('n', 0)}  "
            f"旋转最大偏离 {c.get('rot_max_deg', float('nan')):.3f}°  "
            f"平移最大偏离 {c.get('trans_max_mm', float('nan')):.3f} mm",
            self._aggregate_line(c),
            f"两板间距      : {c.get('baseline_mm', float('nan')):.2f} mm"
            f"（||T_AB||，用卷尺核对）",
            # 深度公式 `Z=f·B/d` 在 none_raw 下不成立（没有公共极线方向 -> 无视差），
            # 这一行必须**明说"不适用"**，而不是省掉：省掉会让读者以为"这份标定
            # 没算深度不确定度"，而真相是"这个口径在这里没有物理意义"。
            "深度不确定度  : 不适用（angled 不能用 Z=fB/d；请看下面的外参不确定度）",
            # 跨路一致性 = chain 唯一涉及"两路之间"的证据（单目 RMS 只管各自拟合）
            self._cross_check_line(),
            *self._sigma_lines(),
            self._cv_line(),
            "立体校正      : none_raw —— 未做极线校正；坐标即原始 left_camera 系；"
            "任何依赖极线水平对齐/视差的方法都不适用",
            "投影矩阵      : P1 = K1·[I|0]  P2 = K2·[R|T]（T 与三维点同单位，"
            "本项目为 m；Q 为 None）",
            "-" * 62,
            f"判定          : {self.verdict}",
        ]
        for w in self.warnings:
            lines.append(f"  ! {w}")
        for s in self.suggestions:
            lines.append(f"  → {s}")
        lines.append("=" * 62)
        return "\n".join(lines)

    def report(self) -> str:
        if self.method == "chain":
            return self._report_chain()
        lines = []
        lines.append("=" * 62)
        lines.append("双目标定结果")
        lines.append("=" * 62)
        lines.append(f"标定板        : {self.board_desc}")
        lines.append(f"图像尺寸      : {self.image_size[0]}x{self.image_size[1]}")
        lines.append(f"使用/输入图像 : {self.n_pairs_used}/{self.n_pairs_in}"
                     + (f"（剔除 {len(self.rejected_index)} 对）" if self.rejected_index else ""))
        lines.append(f"整体重投影 RMS: {self.rms:.4f} px")
        lines.append(f"左目重投影 RMS: {self.err_left:.4f} px")
        lines.append(f"右目重投影 RMS: {self.err_right:.4f} px")
        lines.append(f"极线误差 RMS  : {self.epipolar_rms:.4f} px   (最大 {self.epipolar_max:.4f})")
        lines.append(f"基线长度      : {self.baseline_mm:.2f} mm")
        if np.isfinite(self.board_area):
            # 这个数字**必须出现在报告里**：它解释了"为什么 RMS 偏高"，
            # 而用户拿到报告时板已经收起来了 —— 只有这里能告诉他下次该靠近点。
            # 括号里给"建议范围"，因为光看一个百分比不知道多少算合适。
            lines.append(f"板占画面      : {self.board_area * 100:.1f}%"
                         f"（左 {self.board_area_l * 100:.1f}% / 右 {self.board_area_r * 100:.1f}%；"
                         f"建议 10%~30%）")
        # ---- 不确定度（审查 A5）----
        #
        # 为什么一定要打印：RMS 是"拟合得多好"，而这两个数是"参数能信到什么程度"。
        # 合成真值里同一份 RMS 可以对应 4 倍不同的深度精度 —— 只看 RMS 会误判。
        sf = self.fx_sigma()
        if np.isfinite(sf):
            f = float(self.K1[0, 0])
            lines.append(f"焦距不确定度  : σ(fx) = {sf:.2f} px"
                         f"（相对 {sf / f * 100:.3f}%）")
        else:
            lines.append("焦距不确定度  : 不可用（本版 OpenCV 未返回参数标准差）")
        dz = self.depth_sigma_mm(0.6)
        if np.isfinite(dz):
            lines.append(f"深度不确定度  : 在 0.60 m 处约 **±{dz:.1f} mm**"
                         f"（含焦距项 + 视差项，**不含基线项**；数量级估计）")
        elif not self.depth_applicable():
            # 这里**不能**静默省略：省略等于"没这个指标"，而真相是"这个公式不适用"。
            lines.append("深度不确定度  : 不适用（angled 不能用 Z=fB/d；"
                         "请看下面的外参不确定度）")
        for ln in self._sigma_lines():
            lines.append(ln)
        lines.append(self._cv_line())
        lines.append("-" * 62)
        lines.append(f"判定          : {self.verdict}")
        for w in self.warnings:
            lines.append(f"  ! {w}")
        for s in self.suggestions:
            lines.append(f"  → {s}")
        lines.append("=" * 62)
        return "\n".join(lines)


# --------------------------------------------------------------------------
def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    return float(np.sqrt(np.mean(x * x))) if x.size else float("nan")


def _calibrate_one(objp, imgp, size, cfg: CalibConfig):
    """单目标定，返回 (K, D, 逐图重投影误差, 参数标准差)。

    ⚠️ 第 4 个返回值是 2026-10-18 补上的：`calibrateCameraExtended` 一直返回
    `stdDeviationsIntrinsics`，而这里**原来丢掉它**（`_sdi` 下划线）。
    丢掉之后，判定只能靠 RMS —— 而合成真值证明 RMS 对"板太小"几乎免疫
    （板占 1.9% 与 7.9% 的 RMS 都是 0.41 px，深度精度差 4 倍）。
    """
    flags = 0
    if cfg.rational_model:
        flags |= cv2.CALIB_RATIONAL_MODEL
    if cfg.fix_principal_point:
        flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-7)
    out = cv2.calibrateCameraExtended(objp, imgp, size, None, None, criteria=criteria, flags=flags)
    _ret, K, D, _rv, _tv, sdi, _sde, per_view = out
    return (K, D, np.asarray(per_view, dtype=np.float64).ravel(),
            np.asarray(sdi, dtype=np.float64).ravel())


def _stereo_once(objp, ip1, ip2, K1, D1, K2, D2, size, cfg: CalibConfig):
    """跑一次双目联合标定：先固定内参定外参，再整体精化。"""
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7)
    out = cv2.stereoCalibrate(
        objp, ip1, ip2, K1, D1, K2, D2, size,
        criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC,
    )
    rms, K1, D1, K2, D2, R, T, E, F = out

    if cfg.refine_intrinsics:
        flags = cv2.CALIB_USE_INTRINSIC_GUESS
        if cfg.rational_model:
            flags |= cv2.CALIB_RATIONAL_MODEL
        if cfg.fix_principal_point:
            flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        out2 = cv2.stereoCalibrate(
            objp, ip1, ip2, K1, D1, K2, D2, size, criteria=criteria, flags=flags,
        )
        rms2, K1b, D1b, K2b, D2b, R2, T2, E2, F2 = out2
        # 精化不一定总是更好，取更小的那个
        if np.isfinite(rms2) and rms2 <= rms:
            rms, K1, D1, K2, D2, R, T, E, F = rms2, K1b, D1b, K2b, D2b, R2, T2, E2, F2
    return rms, K1, D1, K2, D2, R, T, E, F


def _rectify(K1, D1, K2, D2, size, R, T, alpha: float):
    return cv2.stereoRectify(
        K1, D1, K2, D2, size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha,
    )


def _epipolar_errors(ip1, ip2, K1, D1, K2, D2, R1, R2, P1, P2) -> np.ndarray:
    """逐对图像的极线误差：校正后同名点的 y 之差。

    ## ⚠️ 2026-10-18 修：原来对"左右点数不一致"是**静默截断**

    旧代码：

        n = min(len(p1), len(p2))
        dy = p1[:n, 1] - p2[:n, 1]

    左右角点数不同意味着**对应关系已经不可信**（棋盘角点数量由板决定，两路必然一致；
    不一致只能是检测/关联出了问题）。这时截断后继续比较，等于**拿错配的点对算误差**，
    而且会把一个真实故障掩盖成"极线误差偏大一点点"。

    现在直接报错：让故障在**离根因最近**的地方暴露，而不是变成一个看不懂的数字。
    """
    errs = []
    for i, (c1, c2) in enumerate(zip(ip1, ip2)):
        n1 = int(np.asarray(c1).reshape(-1, 2).shape[0])
        n2 = int(np.asarray(c2).reshape(-1, 2).shape[0])
        if n1 != n2:
            raise ValueError(
                f"第 {i} 对图像的左右角点数不一致（左 {n1} / 右 {n2}）—— "
                f"对应关系不可信，无法计算极线误差。\n"
                f"  常见原因：某一侧漏检/多检了角点，或左右图像配错了对。\n"
                f"  处置：丢弃这一对（或整组重拍），不要截断后继续比较。")
        p1 = cv2.undistortPoints(c1.reshape(-1, 1, 2).astype(np.float64), K1, D1, R=R1, P=P1).reshape(-1, 2)
        p2 = cv2.undistortPoints(c2.reshape(-1, 1, 2).astype(np.float64), K2, D2, R=R2, P=P2).reshape(-1, 2)
        dy = p1[:, 1] - p2[:, 1]
        errs.append(float(np.sqrt(np.mean(dy * dy))))
    return np.asarray(errs, dtype=np.float64)


# --------------------------------------------------------------------------
def reprojection_rms_px(objp, imgpoints1, imgpoints2, K1, D1, K2, D2, R, T) -> float:
    """**固定**内外参，在给定的图像对上算两路重投影 RMS（px）。

    用途：交叉验证的**测试集打分**。`cv2.stereoCalibrate` 回来的 `rms` 是训练集
    自评（同一批角点既拟合又打分），不能拿来当"换一批图还行不行"的证据。
    这里把参数焊死，逐个视图 `solvePnP` 求板位姿（这是测试集里唯一允许的自由度 ——
    板的摆放），再投影回左右两路比对观测角点。

    注意：右路用 `x_R = R·x_L + T`（**不重新求解**板位姿），所以测试 RMS 里
    真正被检验的是 `(R, T)`；左路那一半只反映内参。
    """
    obj_list = _as_objpoints(objp, len(imgpoints1))
    p1s = _as_imgpoints(list(imgpoints1))
    p2s = _as_imgpoints(list(imgpoints2))
    R = np.asarray(R, np.float64).reshape(3, 3)
    T = np.asarray(T, np.float64).reshape(3)
    errs = []
    for obj, q1, q2 in zip(obj_list, p1s, p2s):
        ok, rvec, tvec = cv2.solvePnP(obj, q1, np.asarray(K1, np.float64),
                                      np.asarray(D1, np.float64))
        if not ok:
            raise ValueError("交叉验证打分：测试集某视图 solvePnP 失败（角点/板模型不匹配）")
        uv1, _ = cv2.projectPoints(obj, rvec, tvec, np.asarray(K1, np.float64),
                                   np.asarray(D1, np.float64))
        Rb = cv2.Rodrigues(rvec)[0]
        tb = np.asarray(tvec, np.float64).reshape(3, 1)
        uv2, _ = cv2.projectPoints(obj, cv2.Rodrigues(R @ Rb)[0], R @ tb + T.reshape(3, 1),
                                   np.asarray(K2, np.float64), np.asarray(D2, np.float64))
        errs.append((uv1.reshape(-1, 2) - q1.reshape(-1, 2)).ravel())
        errs.append((uv2.reshape(-1, 2) - q2.reshape(-1, 2)).ravel())
    if not errs:
        return float("nan")
    e = np.concatenate(errs)
    return float(np.sqrt(np.mean(e * e)))


def _as_objpoints(objp, n: int) -> list[np.ndarray]:
    """把物体点规整成"每视图一个 (N,3) float32 数组"。

    允许传单个 (N,3) 数组（所有视图共用同一块标定板，本项目的常态），
    也允许传逐视图的列表（OpenCV 原生风格）。
    顺便统一转成 float32 —— OpenCV 5 对 float64 的物体点会直接报错。
    """
    if isinstance(objp, np.ndarray) and objp.ndim == 2:
        a = objp.astype(np.float32, copy=False)
        return [a] * n
    out = [np.asarray(o, np.float32).reshape(-1, 3) for o in objp]
    if len(out) != n:
        raise ValueError(f"物体点数量 {len(out)} 与视图数量 {n} 不一致")
    return out


def _as_imgpoints(pts: list[np.ndarray]) -> list[np.ndarray]:
    return [np.asarray(p, np.float32).reshape(-1, 1, 2) for p in pts]


def calibrate_mono_points(objp, imgpoints, image_size, cfg: CalibConfig | None = None) -> dict:
    """**单路**内参标定（chain 方法的左/右两支，GUI 侧也复用它）。

    返回 dict：`K, D, rms, rvecs, tvecs, per_view, std`。
    `per_view` 是逐图重投影 RMS，用来挑"这一路最可信的那一帧"作为链上的位姿。
    """
    cfg = cfg or CalibConfig()
    n = len(imgpoints)
    if n < 3:
        raise ValueError(f"单目标定至少要 3 个姿态，实际只有 {n} 个")
    obj = _as_objpoints(objp, n)
    pts = _as_imgpoints(imgpoints)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-7)
    flags = 0
    if cfg.rational_model:
        flags |= cv2.CALIB_RATIONAL_MODEL
    if cfg.fix_principal_point:
        flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
    out = cv2.calibrateCameraExtended(obj, pts, image_size, None, None,
                                      criteria=criteria, flags=flags)
    _ret, K, D, rvecs, tvecs, sdi, _sde, per_view = out
    per_view = np.asarray(per_view, np.float64).ravel()
    return {
        "K": K, "D": D,
        "rms": _rms(per_view),
        "rvecs": [np.asarray(r, np.float64).reshape(3) for r in rvecs],
        "tvecs": [np.asarray(t, np.float64).reshape(3) for t in tvecs],
        "per_view": per_view,
        "std": np.asarray(sdi, np.float64).ravel(),
    }


def board_pose_from_points(objp, corners, K, D) -> np.ndarray:
    """由已检测的角点 + 内参，求**板在相机系**的位姿 `T_cam_A`（4x4）。"""
    obj = np.asarray(objp, np.float64).reshape(-1, 3)
    img = np.asarray(corners, np.float64).reshape(-1, 1, 2)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise ValueError("solvePnP 失败：角点与板模型不匹配（检查行列数/角点顺序）")
    R, _ = cv2.Rodrigues(rvec)
    return relative_pose.rt_to_T(R, tvec.reshape(3))


def _essential_fundamental(K1, K2, R, T) -> tuple[np.ndarray, np.ndarray]:
    tx = np.array([[0.0, -T[2, 0], T[1, 0]],
                   [T[2, 0], 0.0, -T[0, 0]],
                   [-T[1, 0], T[0, 0], 0.0]], np.float64)
    E = tx @ R
    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
    return E, F


def _calibrate_chain(
    objp,
    imgpoints1: list[np.ndarray],
    imgpoints2: list[np.ndarray],
    image_size: tuple[int, int],
    cfg: CalibConfig,
    spec: BoardSpec,
    res: StereoCalibrationResult,
    *,
    stereo_mode: str,
    T_camL_A: np.ndarray | None,
    T_AB: np.ndarray | None,
    T_camR_B: np.ndarray | None,
    T_AB_samples: Sequence[np.ndarray] | None,
    cross_poses: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
) -> StereoCalibrationResult:
    """chain 分支：独立单目 + 刚性双板公共参考。

    ## 数学（不需要两路共视）

        T_{C_R←C_L} = T_{C_R←B} · T_{B←A} · T_{A←C_L}

    `T_{B←A}`（`T_AB`）**必须由调用方实测提供**：它是"两块板刚性固定在同一支架"
    这一物理事实的唯一来源。缺它时整条链不可解 —— 这里直接报错，
    **绝不猜一个值、也不回退到 joint**（静默的错数比明确失败更贵）。
    """
    if T_AB is None:
        raise ValueError(
            "chain 方法缺少 T_AB（两板刚性相对位姿）：这台装置的外参在数学上不可解。\n"
            "  下一步：用**一块刚性双板靶标**，让任一单相机同时拍到 A、B 两板拍一张，\n"
            "          再调用 apps/calibrate_stereo.py --method chain --pair-frame <目录> "
            "--pair-camera left|right。\n"
            "  注意：不能回退到 joint —— 90° 装置两路几乎不共视，联合求解不成立。"
        )
    T_AB = relative_pose._as_T(T_AB, "T_AB")

    # ---- 多帧 T_AB 聚合（2026-10-26）----
    #
    # 以前这里只把 `T_AB_samples` 拿去算重复性、发个警告，**真正进链的仍是调用方
    # 给的那一帧**。而 90° 装置的外参误差主要就来自 `T_AB`，多帧样本被白白丢掉。
    # 现在 n>=2 时用 `average_T`（平移几何中位数 + 四元数加权平均）聚合后再合成。
    #
    # ⚠️ 例外：`T_AB = I` 的单板模式**不聚合**。那里的 I 不是"测量值"而是
    #    "A、B 是同一块板的同一世界系"这一**定义**；样本是逐张参考图算出的
    #    漂移量（真值恒为 I），把它们平均等于把参考图的漂移当成真实的两板
    #    相对位姿，反而引入系统误差。单板模式的证据看 `same_board_ref`。
    provided = [relative_pose._as_T(T, f"T_AB_samples[{i}]")
                for i, T in enumerate(T_AB_samples or [])]
    same_board_TAB = np.allclose(T_AB, np.eye(4), atol=1e-9, rtol=0.0)
    used_aggregate = False
    agg: dict = {}
    if len(provided) >= 2 and not same_board_TAB:
        T_AB, agg = relative_pose.average_T(provided)
        used_aggregate = True

    res.method = "chain"
    res.n_views_left = len(imgpoints1)
    res.n_views_right = len(imgpoints2)
    res.n_pairs_in = len(imgpoints1) + len(imgpoints2)
    res.n_pairs_used = res.n_pairs_in
    res.rectification = "none_raw"
    res.T_AB = T_AB
    if not imgpoints1 or not imgpoints2:
        res.verdict = "数据不足：左/右路至少各需要 3 个姿态"
        return res

    left = calibrate_mono_points(objp, imgpoints1, image_size, cfg)
    right = calibrate_mono_points(objp, imgpoints2, image_size, cfg)
    res.K1, res.D1, res.K2, res.D2 = left["K"], left["D"], right["K"], right["D"]
    res.err_left, res.err_right = left["rms"], right["rms"]
    res.std1, res.std2 = left["std"], right["std"]

    # 若调用方没显式给两路板位姿，就用同一批角点 solvePnP 求（同一份观测，
    # 不引入新假设）；挑**重投影最小**的那一帧 —— 链上的位姿只该用最可信的一帧。
    if T_camL_A is None:
        i = int(np.argmin(left["per_view"]))
        T_camL_A = relative_pose.rt_to_T(
            cv2.Rodrigues(left["rvecs"][i])[0], left["tvecs"][i])
    if T_camR_B is None:
        j = int(np.argmin(right["per_view"]))
        T_camR_B = relative_pose.rt_to_T(
            cv2.Rodrigues(right["rvecs"][j])[0], right["tvecs"][j])

    R, T, T_camR_camL = relative_pose.relative_extrinsics(T_camL_A, T_AB, T_camR_B)
    res.R, res.T = R, T
    res.E, res.F = _essential_fundamental(res.K1, res.K2, R, T)
    # ---- 未校正投影矩阵（2026-10-24）----
    #
    # 90° 下不能跑 stereoRectify（两光轴夹角接近直角，重投影会把画面拉成极端
    # 梯形、有效像素所剩无几），但**三角测量不需要共面极线**：
    # 只要 P1/P2 与 (K, R, T) 自洽，`cv2.triangulatePoints` 就能直接给出
    # 原始 left_camera 系下的三维点。所以这里给"去畸变、不校正"的那一对：
    #
    #     P1 = K1 · [I | 0]       -> 左图去畸变后仍在 K1 像素系
    #     P2 = K2 · [R | T]       -> 右图去畸变后仍在 K2 像素系，R/T 是真实的
    #
    # `T` 在这里**保持与三维点相同的单位**（本项目为米）：若把 T 换算成 mm
    # 而三维点仍是米，三角化出的点会被整体放大 1000 倍（尺度必须一致）。
    #
    # R1=R2=I 让 `initUndistortRectifyMap` / `undistortPoints` 退化成纯去畸变，
    # 这正是"none_raw"的语义：坐标 = 原始 left_camera 系。
    res.R1 = np.eye(3)
    res.R2 = np.eye(3)
    res.P1 = res.K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    res.P2 = res.K2 @ np.hstack([R, T.reshape(3, 1)])
    res.Q = None
    res.roi1 = res.roi2 = ()
    # 90° 下"校正后 y 差"没有意义：显式置 nan，报告里也不写这一行。
    res.epipolar_rms = float("nan")
    res.epipolar_max = float("nan")
    res.rms = float("nan")
    res.baseline_mm = float(np.linalg.norm(T)) * 1000.0

    samples = list(provided) if provided else [T_AB]
    res.consistency = relative_pose.consistency(samples)
    # 聚合的产物必须一起落盘：否则过几天回看，"用了 4 帧聚合"和"只用 1 帧"
    # 这两份结果长得一模一样，而它们的可信度不同。
    res.consistency["used_aggregate"] = used_aggregate
    res.consistency["n_outliers"] = int(agg.get("n_outliers", 0))
    if used_aggregate:
        res.consistency["aggregate"] = {
            "n": int(agg["n"]),
            "used": int(agg["used"]),
            "n_outliers": int(agg["n_outliers"]),
            "outlier_index": list(agg["outlier_index"]),
            "rot_spread_deg": float(agg["rot_spread_deg"]),
            "trans_spread_mm": float(agg["trans_spread_mm"]),
            "iterations": int(agg["iterations"]),
            "converged": bool(agg["converged"]),
        }

    # ---- 判定 ----
    warns: list[str] = []
    sugg: list[str] = []
    # ---- 跨路一致性（2026-10-26）----
    #
    # chain 以前把 `res.rms` 置 NaN，报告却凭"两路单目 RMS 各自很小"判通过 ——
    # 那两个数**只描述各自的板拟合**，对"两路之间的相对位姿对不对"零信息。
    # 这里补上唯一涉及两路的证据：把两路看同一块板同一位置的板位姿用解出的
    # `T_{C_R←C_L}` 搬到同一坐标系，残差应接近 0；若外参尺度偏 k 倍，
    # 残差 ≈ |k-1|·‖T‖（随**基线**线性放大，与板到相机的距离无关；见 metric_check
    # 的推导，实测 k=1.1 / ‖T‖=781 mm -> 78.10 mm，板放 0.4/0.8/1.2 m 都是 10.00 mm）。
    #
    # ⚠️ 只对 `T_AB ≈ I` 的单板模式成立：双板模式下左看 A、右看 B，
    #    两组板位姿根本不在同一个板坐标系里，拿来做这个检查会得到无意义的差。
    if same_board_TAB:
        pairs_src = list(cross_poses or [])
        if not pairs_src and T_camL_A is not None and T_camR_B is not None:
            pairs_src = [(T_camL_A, T_camR_B)]
        if pairs_src and len(pairs_src) < 2:
            # 单帧 = 没有第二份独立观测：残差恒为 0（外参就是这一对构造的）。
            res.cross_check = {"insufficient":
                               f"n={len(pairs_src)}：外参正是由这一对位姿构造的，"
                               f"残差恒为 0，不构成证据；要 ≥2 张同摆放参考图"}
        elif pairs_src:
            try:
                chk = metric_check.stereo_scale_consistency(
                    T_camR_camL, [p[0] for p in pairs_src], [p[1] for p in pairs_src])
                res.cross_check = {
                    "median_mm": float(chk["median_mm"]),
                    "p95_mm": float(chk["p95_mm"]),
                    "rot_deg": float(chk["rotation_deg"]),
                    "n": int(chk["n"]),
                    "verdict": chk["verdict"],
                }
            except Exception as exc:                      # noqa: BLE001 - 原因要原样带回
                res.cross_check = {"error": f"{type(exc).__name__}: {exc}"}
    if res.cross_check and "error" not in res.cross_check \
            and "insufficient" not in res.cross_check:
        cc = res.cross_check
        # 门限必须取**同一档**：5 mm 是 metric_check 的"不可用"边界，与它并列的
        # 旋转门限就该是 `ok_rot_deg=1.5°`。若配 0.5°（那里是"优秀"边界），
        # 两档混用会把正常装置判成坏的 —— 实测 α 正确、1 px 角点噪声时旋转中位数
        # 已达 0.585°，必然天天告警，告警一旦天天响就没人看了。
        if cc["median_mm"] > _CROSS_TRANS_WARN_MM or cc["rot_deg"] > _CROSS_ROT_WARN_DEG:
            warns.append(
                f"跨路一致性差：中位 {cc['median_mm']:.2f} mm / 旋转 {cc['rot_deg']:.3f}° "
                f"（门限 {_CROSS_TRANS_WARN_MM:.0f} mm / {_CROSS_ROT_WARN_DEG:.1f}°，"
                f"即 metric_check 的「可用」下限）—— 外参可疑，或拍参考图时板动了"
                f"（两路的'同一个位置'其实不是同一个位置）")
    # chain 的交叉验证没有意义：两路内参已各自冻结，子集重跑只是重标一次内参，
    # 检验不到"外参能不能泛化"。如实写出来，而不是悄悄不印。
    res.cv = {"skipped": "chain 的单目内参已冻结，子集重跑无意义"}
    if cfg.expected_baseline_mm:
        rel = abs(res.baseline_mm - cfg.expected_baseline_mm) / cfg.expected_baseline_mm
        if rel > cfg.baseline_tolerance:
            warns.append(
                f"基线 {res.baseline_mm:.1f} mm 与预期 {cfg.expected_baseline_mm:.1f} mm "
                f"相差 {rel * 100:.0f}% —— 物理上对不上（先查两镜头中心距、"
                f"两板是否真的刚性固定、同框图 A/B 是否分配反了）")
    c = res.consistency
    if res.n_views_left < 15:
        warns.append(f"左路只有 {res.n_views_left} 个姿态（建议 ≥15），内参/位姿精度受限")
    if res.n_views_right < 15:
        warns.append(f"右路只有 {res.n_views_right} 个姿态（建议 ≥15），内参/位姿精度受限")
    if res.err_left > cfg.ok_rms:
        warns.append(f"左目重投影 RMS {res.err_left:.2f} px 偏大（ok 门限 {cfg.ok_rms}）")
    if res.err_right > cfg.ok_rms:
        warns.append(f"右目重投影 RMS {res.err_right:.2f} px 偏大（ok 门限 {cfg.ok_rms}）")
    if c["n"] < 2:
        warns.append("T_AB 只测了 1 次：**无法评估重复性**，"
                     "block 的刚性/角点精度都没有证据（建议同框图拍 3~5 张）")
        sugg.append("--pair-frame 给一个含 3~5 张同框图的目录（板不要动），"
                    "程序会给出旋转/平移离散度。")
    if T_AB_samples and len(T_AB_samples) >= 2:
        if c["rot_max_deg"] > CHAIN_ROT_OK_DEG or c["trans_max_mm"] > CHAIN_TRANS_OK_MM:
            warns.append(
                f"T_AB 重复性差：旋转最大偏离 {c['rot_max_deg']:.2f}°、"
                f"平移 {c['trans_max_mm']:.2f} mm —— 两板可能没真正刚性固定，"
                f"或同框图角点检测不可靠")
        elif c["rot_max_deg"] > CHAIN_ROT_GOOD_DEG or c["trans_max_mm"] > CHAIN_TRANS_GOOD_MM:
            warns.append(f"T_AB 重复性一般：旋转 {c['rot_max_deg']:.2f}° / "
                         f"平移 {c['trans_max_mm']:.2f} mm")

    bad_mono = (not np.isfinite(res.err_left) or not np.isfinite(res.err_right)
                or res.err_left > cfg.ok_rms or res.err_right > cfg.ok_rms)
    bad_rep = (c["n"] >= 2 and
               (c["rot_max_deg"] > CHAIN_ROT_OK_DEG or c["trans_max_mm"] > CHAIN_TRANS_OK_MM))
    if bad_mono or bad_rep:
        res.grade = "不可用"
        why = ("单目重投影超门限" if bad_mono else "T_AB 重复性超门限")
        res.verdict = f"不可用 —— {why}；chain 链上的误差会被直接放大到外参"
        res.ok = False
    elif (res.err_left <= cfg.good_rms and res.err_right <= cfg.good_rms
          and c["n"] >= 2 and c["rot_max_deg"] <= CHAIN_ROT_GOOD_DEG
          and c["trans_max_mm"] <= CHAIN_TRANS_GOOD_MM):
        res.grade = "优秀"
        res.verdict = "优秀 —— 链式外参 + 可复现的 T_AB 重复性证据"
        res.ok = True
    else:
        res.grade = "可用"
        res.verdict = ("可用 —— 链式外参已解出；T_AB 重复性或单目精度只到「可用」档，"
                       "报告里有具体数字")
        res.ok = True
    if stereo_mode == "angled" and res.ok:
        sugg.append("这组 P1/P2 是 none_raw（未做极线校正）；三角测量可直接用，"
                    "但任何依赖极线水平对齐/视差的方法都不适用。")
    res.warnings = warns
    res.suggestions = sugg
    return res


def calibrate_stereo(
    objp,
    imgpoints1: list[np.ndarray],
    imgpoints2: list[np.ndarray],
    image_size: tuple[int, int],
    cfg: CalibConfig | None = None,
    spec: BoardSpec | None = None,
    *,
    stereo_mode: str = "parallel",
    T_camL_A: np.ndarray | None = None,
    T_AB: np.ndarray | None = None,
    T_camR_B: np.ndarray | None = None,
    T_AB_samples: Sequence[np.ndarray] | None = None,
    cross_poses: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
) -> StereoCalibrationResult:
    """从已配对的角点做完整双目标定（含离群剔除与质量判定）。

    objp 可以是单个 (N,3) 数组（所有视图共用一块板），也可以是逐视图列表。

    外参解法由 `cfg.method` 选择：

    - ``joint``（默认）：`cv2.stereoCalibrate` 联合求解。**要求两路共视**。
      `stereo_mode="angled"` 且共视对 < `MIN_COVISIBLE_PAIRS_FOR_JOINT_ANGLED`
      时**抛 ValueError**（拒绝出数，见该常量的注释）。
    - ``chain``：`T_{C_R←C_L} = T_{C_R←B}·T_{B←A}·T_{A←C_L}`，不需要共视。
      `T_AB`（= `T_{B←A}`）必须由调用方实测传入；`T_camL_A`/`T_camR_B` 可传入，
      缺省时由本函数对 `imgpoints1/2` 做单目标定 + `solvePnP` 求得
      （同一份观测，不引入新假设）。chain **不调用** `stereoRectify`，
      而是给出 `rectification="none_raw"` 的一对投影矩阵
      （`R1=R2=I`、`P1=K1·[I|0]`、`P2=K2·[R|T]`、`Q=None`）——它们等价于
      "只去畸变、不校正"，三角测量直接可用，但极线不水平。
    """
    cfg = cfg or CalibConfig()
    spec = spec or BoardSpec()

    # ---- 护栏：angular + joint + 共视不足 -> 拒绝出数 ----
    # 必须在"数据不足就 return 空结果"**之前**：否则 3 对数据会得到一个
    # 静默的 ok=False，用户看不出这是"方法不成立"还是"手滑少收了两对"。
    n_covis = min(len(imgpoints1), len(imgpoints2))
    if (cfg.method == "joint" and stereo_mode == "angled"
            and n_covis < MIN_COVISIBLE_PAIRS_FOR_JOINT_ANGLED):
        raise ValueError(
            f"angled 且只有 {n_covis} 对共视数据：宽基线从零优化会被噪声主导，拒绝出数。"
            f"下一步：用 --method chain（需刚性双板靶标），"
            f"或补足 ≥{MIN_COVISIBLE_PAIRS_FOR_JOINT_ANGLED} 对两路同时可见的板。")

    res = StereoCalibrationResult(image_size=image_size, board_desc=spec.describe(),
                                  n_pairs_in=len(imgpoints1), method=cfg.method,
                                  stereo_mode=stereo_mode)
    if cfg.method == "chain":
        return _calibrate_chain(
            objp, list(imgpoints1), list(imgpoints2), image_size, cfg, spec, res,
            stereo_mode=stereo_mode, T_camL_A=T_camL_A, T_AB=T_AB,
            T_camR_B=T_camR_B, T_AB_samples=T_AB_samples, cross_poses=cross_poses,
        )

    if len(imgpoints1) < 4 or len(imgpoints1) != len(imgpoints2):
        res.verdict = "数据不足或左右数量不一致"
        return res

    obj_all = _as_objpoints(objp, len(imgpoints1))
    imgpoints1 = _as_imgpoints(imgpoints1)
    imgpoints2 = _as_imgpoints(imgpoints2)

    idx = list(range(len(imgpoints1)))
    best = None
    for round_i in range(cfg.max_reject_rounds if cfg.reject_outliers else 1):
        o = [obj_all[i] for i in idx]
        p1 = [imgpoints1[i] for i in idx]
        p2 = [imgpoints2[i] for i in idx]

        K1, D1, e1, sdi1 = _calibrate_one(o, p1, image_size, cfg)
        K2, D2, e2, sdi2 = _calibrate_one(o, p2, image_size, cfg)
        rms, K1, D1, K2, D2, R, T, E, F = _stereo_once(o, p1, p2, K1, D1, K2, D2, image_size, cfg)
        R1, R2, P1, P2, Q, roi1, roi2 = _rectify(K1, D1, K2, D2, image_size, R, T, cfg.alpha)
        epi = _epipolar_errors(p1, p2, K1, D1, K2, D2, R1, R2, P1, P2)

        # 逐对综合误差：左右重投影误差与极线误差，谁大听谁的
        per_pair = np.maximum(np.maximum(e1, e2), epi)
        med = float(np.median(per_pair))
        mad = float(np.median(np.abs(per_pair - med))) * 1.4826   # 稳健标准差
        thresh = max(med + cfg.outlier_sigma * mad, cfg.outlier_min_px)

        cur = dict(idx=list(idx), K1=K1, D1=D1, K2=K2, D2=D2, R=R, T=T, E=E, F=F,
                   R1=R1, R2=R2, P1=P1, P2=P2, Q=Q, roi1=roi1, roi2=roi2,
                   rms=float(rms), e1=float(_rms(e1)), e2=float(_rms(e2)),
                   epi=epi, per_pair=per_pair, std1=sdi1, std2=sdi2)
        if best is None or cur["rms"] <= best["rms"]:
            best = cur

        if not cfg.reject_outliers:
            break
        bad = [i for i, v in zip(idx, per_pair) if v > thresh]
        if not bad or len(idx) - len(bad) < cfg.min_pairs_after_reject:
            break
        idx = [i for i in idx if i not in set(bad)]

    b = best
    res.K1, res.D1, res.K2, res.D2 = b["K1"], b["D1"], b["K2"], b["D2"]
    res.R, res.T, res.E, res.F = b["R"], b["T"], b["E"], b["F"]
    res.R1, res.R2, res.P1, res.P2, res.Q = b["R1"], b["R2"], b["P1"], b["P2"], b["Q"]
    res.roi1, res.roi2 = b["roi1"], b["roi2"]
    res.rms, res.err_left, res.err_right = b["rms"], b["e1"], b["e2"]
    res.epipolar_rms = _rms(b["epi"])
    res.epipolar_max = float(np.max(b["epi"])) if len(b["epi"]) else float("nan")
    res.per_pair_epipolar = [float(v) for v in b["epi"]]
    res.used_index = list(b["idx"])
    res.rejected_index = [i for i in range(res.n_pairs_in) if i not in set(b["idx"])]
    res.std1, res.std2 = b.get("std1"), b.get("std2")

    # ---- 外参不确定度（2026-10-26）----
    #
    # `cv2.stereoCalibrate` **不返回** `T` 的协方差，于是报告里的 σ_Z 一直缺
    # 基线项 —— 而短基线近距装置上 `(Z/B)·σ_B` 恰恰是主导项。这里用数值雅可比
    # 在解处估 `(R,T)` 的条件协方差（推导见 calib/uncertainty.py），只加输出、
    # 不动任何数值路径。
    #
    # 失败**不吞**：`extrinsic_sigma` 用 `{"error": ...}` 明确失败，报告原样打印。
    try:
        from .uncertainty import extrinsic_sigma as _extrinsic_sigma

        used = list(b["idx"])
        res.extrinsic_sigma = _extrinsic_sigma(
            [obj_all[i] for i in used],
            [imgpoints1[i] for i in used],
            [imgpoints2[i] for i in used],
            res.K1, res.D1, res.K2, res.D2, res.R, res.T, image_size)
    except Exception as exc:                              # noqa: BLE001 - 原文要保留
        res.extrinsic_sigma = {"error": f"{type(exc).__name__}: {exc}"}

    # ---- 板占画面比例：从**实际用上**的那些对的角点算 ----
    #
    # 为什么要凸包：角点数组是**行序网格**（第 1 行从左到右、然后第 2 行…），
    # 而 `cv2.contourArea` 把点当作**多边形顶点按顺序**连起来。
    # 网格顺序连出来的路径会自己折回去，面积**恒为 0**（实测确认）。
    # 必须先用 `convexHull` 求外轮廓 —— 标定板是凸的，凸包就是它的真实轮廓。
    def _area(points: Sequence[np.ndarray]) -> float:
        w, h = image_size
        if not points or w <= 0 or h <= 0:
            return float("nan")
        areas = []
        for pts in points:
            p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
            if p.shape[0] < 3:
                continue
            hull = cv2.convexHull(p.astype(np.float32))
            areas.append(float(cv2.contourArea(hull)))
        if not areas:
            return float("nan")
        return float(np.median(areas)) / float(w * h)

    res.board_area_l = _area([imgpoints1[i] for i in b["idx"] if i < len(imgpoints1)])
    res.board_area_r = _area([imgpoints2[i] for i in b["idx"] if i < len(imgpoints2)])
    res.n_pairs_used = len(b["idx"])
    res.baseline_mm = float(np.linalg.norm(b["T"])) * 1000.0

    # ---- 判定 ----
    #
    # ## 为什么改成分级（2026-10-14）
    #
    # 原来是二值：`rms <= ok_rms and epipolar <= ok_epipolar` 才 `ok=True`，
    # 否则一句"不可用 —— 请重拍"。实测的一次真机标定：
    #
    #     基线 67.28 mm（实际 ~65，只差 3.5% —— 这是对上了）
    #     整体 RMS 0.9143 px（ok 门限 0.80）
    #     极线 RMS 0.8071 px（ok 门限 0.60）
    #
    # **基线核对通过了**（说明方格尺寸、内外参都对），只是精度指标刚过线，
    # 结果被整体否掉、不保存。用户的原话是"我不是很能接受" —— 他不接受是对的：
    #
    # 1. `ok_rms = 0.80 px` 这条线**本来就是"够用"的意思**，不是"超过就废"。
    #    真正决定三维精度的是极线误差，而 0.81 px 在 1080p 下约合
    #    0.81/1920 ≈ 0.04% 的画面宽度 —— 对厘米级的目标完全够。
    # 2. 二值判定丢掉了**最有用的那条信息**：差在哪、下次怎么改。
    #    一次"基线正确但精度刚过线"的标定，其价值与"完全解错"是两回事，
    #    报告里却长得一模一样。
    #
    # 所以现在是**四级**，且**只要基线没问题就仍然保存**：
    #   优秀 / 可用   —— 直接可用于三角测量
    #   勉强可用      —— 保存，但报告里写清差在哪、怎么改进；先能用起来
    #   不可用        —— **只有物理上错了才用**：基线对不上、有效对太少、数值发散
    warns: list[str] = []
    sugg: list[str] = []

    if res.n_pairs_used < 8:
        warns.append(f"有效图像只有 {res.n_pairs_used} 对，建议补到 15 对以上")
    if res.rms > cfg.ok_rms:
        warns.append(f"重投影 RMS {res.rms:.2f} px 偏大（ok 门限 {cfg.ok_rms}）")
    if res.epipolar_rms > cfg.ok_epipolar:
        warns.append(f"极线误差 {res.epipolar_rms:.2f} px 偏大（ok 门限 {cfg.ok_epipolar}）")

    baseline_bad = False
    if cfg.expected_baseline_mm:
        rel = abs(res.baseline_mm - cfg.expected_baseline_mm) / cfg.expected_baseline_mm
        if rel > cfg.baseline_tolerance:
            baseline_bad = True
            warns.append(
                f"基线 {res.baseline_mm:.1f} mm 与预期 {cfg.expected_baseline_mm:.1f} mm "
                f"相差 {rel * 100:.0f}% —— 这是**物理上对不上**，不是精度问题"
            )
            sugg.append("先核对：标定板方格边长（用尺子量实际值，别信打印设置）、"
                        "两镜头中心距、有没有把两台相机搞反。")
    else:
        sugg.append("下次加上 --baseline-mm <两镜头中心距>："
                    "不填的话基线核对这道护栏是关着的，错了也不会有人告诉你。")

    # 数值发散：RMS 到了"像素级垃圾"的程度（实测整份反序会到几百 px）
    diverged = not np.isfinite(res.rms) or res.rms > 5.0

    # ---- 可执行建议：把"换更多角度"换成能照着做的动作 ----
    #
    # ⚠️ **建议与判定是两件事，不要互相压制**。
    #
    # 我第一版把建议整体挂在"没到优秀才给"下面，结果是：板只占 1.12%、
    # 但因为合成噪声理想而解出"优秀"，于是**一条建议都没有** ——
    # 而"板太远"这件事是**真的可以更好**，与判定等级无关。
    #
    # 真正需要避免的自相矛盾只有一种：建议里说"板太近"而结论说"没到目标"
    # 这种**指向相反**的话。所以按条件各给各的，不整体开关。
    if np.isfinite(res.board_area) and res.board_area < 0.08:
        sugg.append(
            f"**板离得太远**：只占画面 {res.board_area * 100:.1f}%，建议靠近到 10%~30%。"
            f"角点像素越多，定位越准 —— 这是把 RMS 降下来最直接的一招"
            f"（采集时窗口里会实时显示这个比例）。")
    if np.isfinite(res.board_area) and res.board_area > 0.45:
        sugg.append(f"板占画面 {res.board_area * 100:.0f}% 有点太近，"
                    f"边角容易出画或畸变过大，退到 10%~30%。")
    if res.n_pairs_used < 20:
        sugg.append(f"再多收几对（现在 {res.n_pairs_used} 对，建议 20 对以上），"
                    f"并且**让板覆盖画面四角**，不只是中间。")
    if res.epipolar_rms > cfg.ok_epipolar:
        sugg.append("重拍时**每个姿势停稳约 1 秒**再移开（板在动的时候两路拍到的是"
                    "不同位姿，这是极线误差偏大的头号原因）。")
    res.suggestions = sugg
    res.warnings = warns

    # ---- 分级 ----
    #
    # ⚠️ 2026-10-18：优秀这一档**多了一道硬条件**：板在画面里必须够大。
    # 依据见 `min_board_area_for_good` 的注释 —— 同样的 RMS 下，
    # 板占 1.9% 与 7.9% 的深度精度差 4 倍。RMS 好看**不足以**判"优秀"。
    area_big_enough = (not np.isfinite(res.board_area)
                       or res.board_area >= cfg.min_board_area_for_good)
    if (res.rms <= cfg.good_rms and res.epipolar_rms <= cfg.good_epipolar
            and not area_big_enough):
        warns.append(
            f"精度指标已达「优秀」档，但板只占画面 {res.board_area * 100:.1f}%"
            f"（< {cfg.min_board_area_for_good * 100:.0f}%）—— 因此**最高只判「可用」**。"
            f"板小会让焦距/主点与畸变强耦合，深度误差被放大而 RMS 看不出来。")

    if diverged or baseline_bad or res.n_pairs_used < 8:
        # 只有"物理上错了"才不可用
        if diverged:
            why = f"求解发散（整体 RMS {res.rms:.1f} px）"
        elif baseline_bad:
            why = "基线对不上"
        else:
            why = f"有效图像只有 {res.n_pairs_used} 对"
        res.grade = "不可用"
        res.verdict = f"不可用 —— {why}，请重拍"
        res.ok = False
    elif (res.rms <= cfg.good_rms and res.epipolar_rms <= cfg.good_epipolar
          and area_big_enough):
        res.grade = "优秀"
        res.verdict = "优秀 —— 可以直接用于三角测量"
        res.ok = True
    elif res.rms <= cfg.ok_rms and res.epipolar_rms <= cfg.ok_epipolar:
        res.grade = "可用"
        res.verdict = "可用 —— 满足三维重建要求"
        res.ok = True
    else:
        # ★ **仍然保存**：精度刚过线不等于废掉。
        res.grade = "勉强可用"
        worst = max(res.rms / cfg.ok_rms, res.epipolar_rms / cfg.ok_epipolar)
        res.verdict = (f"勉强可用 —— 精度略低于目标（最差指标为门限的 {worst:.2f} 倍），"
                       f"**已保存**，可以先用来做三维；按下面的建议重拍会更好")
        res.ok = True
    return res
