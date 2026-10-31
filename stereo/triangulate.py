"""极线约束配对 + 三角测量。

分工：
  - calib.StereoRig 负责数学部分（去畸变 → 立体校正 → 三角测量）；
  - 本模块负责**流程**：怎么把左图的检测点配到右图的候选上、怎么给出置信度。

核心是**极线约束**：校正之后，左图上的点在右图中的对应点必定在**同一条水平线**上。
这一下就把二维搜索降成了一维，是双目能实时跑起来的关键。

用法::

    from calib import StereoRig
    from stereo import StereoReconstructor

    rig = StereoRig.load("config/stereo_calibration")
    rec = StereoReconstructor(rig)
    pt = rec.update(t=1.23, left=det_left, right_dets=dets_right)
    if pt is not None:
        print(pt.xyz, pt.residual_px)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from calib import StereoRig
from calib.params import is_uncorrected
from sphere_tracker.types import Detection

_LOGGER = logging.getLogger(__name__)

# 静默失效门限：帧数够多、却一个都没配上，几乎一定是"装置不是 parallel"。
# 20 帧是"值得报警"的最小样本量；更少的帧数可能只是 1-2 个球没检出。
SILENT_FAILURE_MIN_FRAMES = 20
# 配对率低于这个值：不抛错（也许只是遮挡严重），但必须留下一条明确警告。
LOW_MATCH_RATE = 0.05
# 配对率低于这个值：额外把"极线几何对不对"**量出来**给人看（见 geometry_lines）。
# 取 10% 而不是上面的 5%：5% 才警告的话，1/30（3.3%）这种"几乎全废"的结果
# 在沉默区里待着；而 10% 已经低到"再看也不会是遮挡"的量级。
LOW_MATCH_RATE_REPORT = 0.10
# 几何一致性诊断：样本少于这个帧数就不报 —— 20 帧以下的中位数只会误导。
GEOM_MIN_SAMPLES = 20
# 实测极线偏差中位 / 标定自报 rms 超过这个倍数 -> 判"录像与标定不一致"。
# 4 倍是"再考虑测量噪声也解释不了"的量级（本项目的标定自报本身已含 rms 与 max）。
GEOM_SUSPECT_RATIO = 4.0
# 没有标定自报值时的绝对判据：中位偏差超过 3 px 就算可疑（默认门限 2.5 px 的量级）。
GEOM_SUSPECT_PX = 3.0

# 配对门限 = k × 标定自报极线 RMS。加了这三个常量，是为了把"门限该多大"
# 这件以前写死的事**显式化**：门限随标定质量走，而不是永远钉在 2.5 px。
EPI_GATE_K = 4.0          # 门限 = k × 标定自报极线 RMS
EPI_GATE_MIN_PX = 1.0     # 下限
EPI_GATE_MAX_PX = 6.0     # 上限


def epipolar_gate_px(quality: dict | None, k: float = EPI_GATE_K,
                     fallback: float = 2.5) -> float:
    """按标定质量给出配对门限（像素）。``quality`` 取 ``rig.quality`` 那份字典。

    语义：只有 ``quality["epipolar_rms"]`` 是**有限的、正的实数**时才按它算，
    否则一律回退到 ``fallback``（默认 2.5 px，与历史硬编码门限一致）。

    为什么这么定：

    ① **k = 4.0**：与 :data:`GEOM_SUSPECT_RATIO` 同源 —— 那里判"录像与标定
       不一致"用的就是"实测偏差 ≥ 标定自报的 4 倍"。配对门限取同一个倍数，
       含义统一为"偏差到自报值的 4 倍就算可疑"。本机自报极线 RMS 是 0.584 px，
       0.584 × 4 ≈ 2.34，已经很接近今天的硬编码 2.5 px —— 所以这次**不是放宽
       门限**，只是把"2.5 ≈ 4.3×RMS"这层隐式关系写成显式的 4×RMS。

    ② **要上限**：极线 RMS 大的标定（例如 3 px）若照搬 4 倍会得到 12 px 门限，
       那等于把极线约束关掉了（校正后左右点的 y 差本来只该是亚像素~像素级），
       任何右图候选都能配上，配对率虚高而三维结果不可信。上限 6 px 托底。

    ③ **0.0 要当无效**：若把 RMS = 0 当成合法值，门限会算成 0，于是**每一个**
       候选都因 ``dy > 0`` 被拒，配对率恒为 0；而这不会抛错，只会静默地
       "什么都配不上" —— 典型的静默失效。同理 NaN / inf / 非数 / 键缺失
       也必须回退，绝不能拿它们去乘。
    """
    rms = quality.get("epipolar_rms") if isinstance(quality, dict) else None
    if isinstance(rms, (int, float)) and math.isfinite(rms) and rms > 0.0:
        return min(max(k * rms, EPI_GATE_MIN_PX), EPI_GATE_MAX_PX)
    return fallback


@dataclass
class MatchConfig:
    max_epipolar_px: float = 2.5   # 允许的极线偏差（校正后 y 之差）
    min_disparity_px: float = 0.5  # 视差小于这个 -> 点在无穷远或配错了
    max_disparity_px: float = 600.0
    require_radius_ratio: bool = False  # 是否要求左右半径接近（同型号相机才合理）
    radius_ratio_tol: float = 0.35
    # 视差符号。0 = 不检查（按 |视差| 判范围，两边都接受）；
    # 一般不用手填 —— StereoReconstructor 会用 rig.disparity_sign() 自动定，
    # 见下面 match_epipolar 的说明。
    disparity_sign: int = 0


@dataclass
class StereoPoint:
    """一个时刻的三维点及其置信度。"""

    t: float
    xyz: np.ndarray                # (3,) 米，**校正后左相机系**
    u_left: tuple[float, float]
    u_right: tuple[float, float]
    epipolar_px: float             # 配对时的极线偏差
    residual_px: float             # 三角测量后重投影回两路的误差（交会残差）
    radius_left: float = float("nan")

    @property
    def distance_m(self) -> float:
        return float(np.linalg.norm(self.xyz))

    @property
    def confident(self) -> bool:
        return self.residual_px <= 2.0


def rectify_point_left(rig: StereoRig, x: float, y: float) -> tuple[float, float]:
    """把一个左图像素点映射到"校正后的左相机"坐标系。

    坐标口径**只由** `StereoRig.undistort_points()` 决定，这里不再自己传
    `rig.R1/P1`：`none_raw`（chain/90°）标定下 `P1` 的 3x3 是 `K1` 而不是
    "新内参"，自己拼参数就会把口径写死成 parallel。
    """
    px, py = rig.undistort_points("left", [[x, y]])[0]
    return float(px), float(py)


def rectify_point_right(rig: StereoRig, x: float, y: float) -> tuple[float, float]:
    """把一个右图像素点映射到"校正后的右相机"坐标系（口径同上，统一入口）。

    ⚠️ `none_raw` 下 `P2 = K2·[R|T]`，3x3 块是 **K2·R**、不是新内参；
    直接把它当 `newCameraMatrix` 会**多转一次 R**（实测 828 px -> -9155 px），
    交会残差随之失真。
    """
    px, py = rig.undistort_points("right", [[x, y]])[0]
    return float(px), float(py)


def match_epipolar(
    rig: StereoRig,
    det_left: Detection,
    candidates_right: list[Detection],
    cfg: MatchConfig | None = None,
) -> tuple[Detection | None, float]:
    """在右图候选里挑出与左图检测点对应的那一个。

    返回 (最佳候选, 极线偏差)。找不到合适的返回 (None, inf)。
    """
    cfg = cfg or MatchConfig()
    if not candidates_right:
        return None, float("inf")

    x1, y1 = rectify_point_left(rig, det_left.x, det_left.y)

    best, best_cost, best_dy = None, float("inf"), float("inf")
    for cand in candidates_right:
        x2, y2 = rectify_point_right(rig, cand.x, cand.y)
        dy = abs(y1 - y2)
        if dy > cfg.max_epipolar_px:
            continue
        # 视差用**绝对值**判范围，符号另判。
        #
        # 为什么不能直接写 `if not (min <= disparity <= max): continue`：
        # 那等于把"视差必须为正"这个假设藏进了范围检查里。一旦左右装反
        # （同型号相机插拔后索引互换，画面完全正常、标定也能过），
        # 视差整体反号，于是**每一个点都被静默拒绝**，配对率 0%，
        # 而报错信息只有"配不上"，根本指不到原因。
        # 三角测量本身不受影响（P1/P2 是自洽的），所以这个坑特别隐蔽。
        disparity = x1 - x2
        if not (cfg.min_disparity_px <= abs(disparity) <= cfg.max_disparity_px):
            continue
        if cfg.disparity_sign and math.copysign(1.0, disparity) != cfg.disparity_sign:
            continue
        if cfg.require_radius_ratio and det_left.radius > 0 and cand.radius > 0:
            ratio = abs(cand.radius - det_left.radius) / det_left.radius
            if ratio > cfg.radius_ratio_tol:
                continue
        cost = dy
        if cost < best_cost:
            best, best_cost, best_dy = cand, cost, dy
    return best, (best_dy if best is not None else float("inf"))


def epipolar_dy(rig: StereoRig, det_left: Detection | None,
                candidates_right: list[Detection]) -> float | None:
    """两路都检出时，**校正后** v 差的最小值（px）；缺一边返回 ``None``。

    这是"最像对应点"的那个偏差，也是几何诊断唯一采样的量：只测量、不配对、
    不改任何门限。取最小而不是取"配对成功的那个"，是为了最不容易冤枉几何。

    为什么单列一个函数：这一段以前在 :meth:`StereoReconstructor._match_frame`
    与录前几何守卫里各写一遍，改口径时最怕两边不同步；现在只留这一处。
    """
    if det_left is None or not candidates_right:
        return None
    y1 = rectify_point_left(rig, det_left.x, det_left.y)[1]
    return min(abs(y1 - rectify_point_right(rig, c.x, c.y)[1]) for c in candidates_right)


def triangulate_one(
    rig: StereoRig,
    det_left: Detection,
    det_right: Detection,
    t: float,
    epipolar_px: float = float("nan"),
) -> StereoPoint:
    """单点三角测量 + 交会残差。"""
    pts_l = np.array([[det_left.x, det_left.y]], np.float64)
    pts_r = np.array([[det_right.x, det_right.y]], np.float64)
    xyz = rig.triangulate(pts_l, pts_r)[0]

    # 交会残差：把三维点投回两路，看和用于三角测量的像素差多少。
    #
    # 注意这里必须用**校正后**的量：xyz 是在校正后左相机系下的，
    # 对应的像素也是校正后的像素。用原始 K1/K2 和原始像素去比，
    # 会得到一个几十像素的假残差（立体校正的旋转 R1 没算进去）。
    p1 = np.array(rectify_point_left(rig, det_left.x, det_left.y))
    p2 = np.array(rectify_point_right(rig, det_right.x, det_right.y))
    e_l = float(np.linalg.norm(_project(rig.P1, xyz) - p1))
    e_r = float(np.linalg.norm(_project(rig.P2, xyz) - p2))
    residual = 0.5 * (e_l + e_r)

    return StereoPoint(
        t=float(t), xyz=np.asarray(xyz, np.float64),
        u_left=(det_left.x, det_left.y), u_right=(det_right.x, det_right.y),
        epipolar_px=float(epipolar_px), residual_px=residual,
        radius_left=float(det_left.radius),
    )


def _project(P: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """用 3x4 投影矩阵把（校正后左相机系下的）三维点投影到像素。"""
    h = np.asarray(P, np.float64) @ np.append(np.asarray(xyz, np.float64), 1.0)
    return h[:2] / h[2]


@dataclass
class GeometryVerdict:
    """一段录像的极线几何判读结果（**唯一**判据实现，见 :func:`geometry_verdict`）。

    为什么单独做一个类型：这套判据以前只在 `geometry_lines()` 里、而且只在
    **整段跑完之后**才用得上；`B-131` 的教训是"跑了半天才知道录像与标定对不上"。
    现在提前判读与最后出报告共用同一个函数，避免两处各写一套门限（本项目
    2026-10-30 就因为两处判据打架而互相矛盾过）。
    """

    n: int
    med: float
    p95: float
    epipolar_rms: float = float("nan")   # 标定自报值；取不到就是 nan（**不编数**）
    epipolar_max: float = float("nan")
    suspect: bool = False                # 判"录像与标定不一致"

    @property
    def has_ref(self) -> bool:
        """标定有没有给出可用的自报极线误差。"""
        return math.isfinite(self.epipolar_rms) and self.epipolar_rms > 0

    @property
    def ratio(self) -> float:
        """实测中位 / 标定自报；没有自报值时是 nan。"""
        return self.med / self.epipolar_rms if self.has_ref else float("nan")


def geometry_verdict(dy_samples, quality=None,
                     min_samples: int = GEOM_MIN_SAMPLES) -> GeometryVerdict | None:
    """把逐帧的"校正后左右圆心 y 差"读成判读结果；样本不足返回 ``None``。

    `dy_samples` 是**校正后**的 v 差（原始像素里混着畸变与校正旋转，不能拿去
    跟标定的 `epipolar_rms` 比）。`quality` 是标定的质量块（取 `epipolar_rms` /
    `epipolar_max`）。

    判据两条，缺一不可：
      - 有标定自报值：``中位 / rms >= GEOM_SUSPECT_RATIO`` **且** ``中位 >= GEOM_SUSPECT_PX``；
        （两个都要：只看比值会把"rms 极小 + 中位 1 px"也判成可疑，只看绝对值
        又浪费了标定自报这个可比基准）
      - 没有自报值：只看 ``中位 >= GEOM_SUSPECT_PX`` —— 别拿"不知道"当"没问题"。
    """
    a = np.asarray(list(dy_samples), np.float64)
    a = a[np.isfinite(a)]
    if a.size < min_samples:
        return None
    med = float(np.median(a))
    p95 = float(np.percentile(a, 95))
    q = quality or {}
    try:
        rms = float(q.get("epipolar_rms", float("nan")))
        worst = float(q.get("epipolar_max", float("nan")))
    except (TypeError, ValueError):
        rms = worst = float("nan")
    has_ref = math.isfinite(rms) and rms > 0
    suspect = ((med / rms >= GEOM_SUSPECT_RATIO and med >= GEOM_SUSPECT_PX)
               if has_ref else (med >= GEOM_SUSPECT_PX))
    return GeometryVerdict(n=int(a.size), med=med, p95=p95, epipolar_rms=rms,
                           epipolar_max=worst, suspect=bool(suspect))


@dataclass
class StereoReconstructor:
    """把逐帧的左右检测配成一条三维轨迹。"""

    rig: StereoRig
    cfg: MatchConfig = field(default_factory=MatchConfig)
    trajectory: list[StereoPoint] = field(default_factory=list)

    # 统计
    n_frames: int = 0
    n_matched: int = 0
    n_no_left: int = 0          # 左路没检测
    n_no_right: int = 0         # 右路没有候选
    n_rejected: int = 0         # 有候选但全被极线/视差条件拒掉

    # 非致命诊断（配对率过低等）；由 update() 追加，供日志/报告读取
    warnings: list[str] = field(default_factory=list)
    _low_rate_warned: bool = field(default=False, repr=False)
    #: 给人看的会话目录（**只进报错文案**，不参与任何计算）：0 配对时要直接告诉用户
    #: `--seed-json <会话>/derived/seeds.json` 该填什么路径，重建器自己没有会话概念。
    source_hint: str = ""
    #: 几何一致性诊断的样本：两路都检出时，**校正后** v 差的最小值（px），逐帧一条。
    #: 只测量、不参与配对，用来在低配对率时说清"是几何对不上，还是检测丢了"。
    dy_samples: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        # ---- 使用点护栏：none_raw（链式/未校正）不能走 parallel 重建 ----
        # 为什么必须在这里拦、而不是只靠入口写对 meta：
        # 历史标定文件已经写错过（chain 落了 stereo_mode="parallel"），而
        # max_epipolar_px=2.5 / 视差范围都建立在"极线已水平"的前提上。90° 装置
        # 原始像素的 |dy| 可达几十~几百 px，于是**全部候选被拒、配对率 0、不报错**。
        # 这种静默归零比崩溃难查得多，所以直接用异常拒绝。
        rect = getattr(self.rig, "rectification", "") if self.rig is not None else ""
        if is_uncorrected(rect):
            raise ValueError(
                "该标定未做极线校正（none_raw），不能用 parallel 重建"
                "（dy 门限/视差范围都建立在校正图上）。请用 angled 重建。")
        # 视差符号按装置实际朝向自动定：右相机在右边 -> +1，装反了 -> -1。
        # 不这么做的话，装反时配对率是 0%，而报错只有"配不上"。
        if not self.cfg.disparity_sign:
            sign = self.rig.disparity_sign() if self.rig is not None else 0
            if sign:
                self.cfg.disparity_sign = sign

    def update(
        self,
        t: float,
        left: Detection | None,
        right_dets: list[Detection],
    ) -> StereoPoint | None:
        """喂一帧的左右检测，返回三维点（配不上则返回 None）。

        帧数攒够后才会做一次"静默失效"检查（见 :meth:`_check_diagnostics`）：
        parallel 模式下 0 配对会抛 ``RuntimeError``，而不是安静地输出空轨迹。
        """
        self.n_frames += 1
        pt = self._match_frame(t, left, right_dets)
        self._check_diagnostics()
        return pt

    def _match_frame(
        self,
        t: float,
        left: Detection | None,
        right_dets: list[Detection],
    ) -> StereoPoint | None:
        if left is None:
            self.n_no_left += 1
            return None
        if not right_dets:
            self.n_no_right += 1
            return None
        # 几何诊断的样本：两路都有检出的帧才量，量的是**校正后**的 v 差
        # （原始像素的 y 差里混着畸变与校正旋转，不能拿去跟标定的 epipolar_rms 比）。
        # 取候选里最小的那个 —— 那才是"最像对应点"的偏差，最不容易冤枉几何。
        dy = epipolar_dy(self.rig, left, right_dets)
        if dy is not None:
            self.dy_samples.append(dy)
        matched, dy = match_epipolar(self.rig, left, right_dets, self.cfg)
        if matched is None:
            self.n_rejected += 1
            return None
        pt = triangulate_one(self.rig, left, matched, t, dy)
        self.n_matched += 1
        self.trajectory.append(pt)
        return pt

    # -- 静默失效门禁 ----------------------------------------------------
    def _is_parallel(self) -> bool:
        return self.rig is None or getattr(self.rig, "stereo_mode", "parallel") != "angled"

    def _check_diagnostics(self) -> None:
        """帧数够多后检查配对情况：0 配对 -> 抛错；配对率过低 -> 警告。

        `angled` 模式不做这个门禁：它的配对慢但合法（见 stereo/angled.py），
        而且 `StereoReconstructor` 也不是它的入口。
        """
        if not self._is_parallel():
            return
        if self.n_frames < SILENT_FAILURE_MIN_FRAMES:
            return
        if self.n_matched == 0:
            raise RuntimeError(self._zero_match_message())
        rate = self.match_rate
        if rate < LOW_MATCH_RATE and not self._low_rate_warned:
            self._low_rate_warned = True
            msg = (f"截至第 {self.n_frames} 帧：parallel 模式配对率极低，"
                   f"匹配 {self.n_matched} / 总数 {self.n_frames}"
                   f"（{rate * 100:.1f}% < {LOW_MATCH_RATE * 100:.0f}%）。"
                   + self._low_rate_advice())
            self.warnings.append(msg)
            _LOGGER.warning(msg)

    def dominant_cause(self) -> str:
        """这一趟的**主要矛盾**：``"detect"`` / ``"match"`` / ``"mixed"``。

        `_match_frame` 每帧只会落进三个计数器之一，所以
        ``n_no_left + n_no_right + n_rejected == n_frames`` 恒成立 ——
        "全被拒"等价于两个缺帧计数都是 0，"没检出"等价于 ``n_rejected == 0``。

        判据只在这里写一次：0 配对的报错、运行期的低配对率告警都调它。
        2026-10-30 就是两处各写一套才互相打架 —— 告警说"若装置是大角度，
        请用 angled 重跑"，结束时的 [诊断] 却说"几何与标定不符"。
        """
        if self.n_rejected == 0:
            return "detect"
        if self.n_no_left + self.n_no_right == 0:
            return "match"
        return "mixed"

    def _optical_axis_angle_deg(self) -> float | None:
        """两光轴夹角（度）：优先 `rig.angle_deg`，其次 `rig.meta`，最后从 `rig.R` 算。

        全取不到就返回 ``None`` —— 宁可不提 angled，也不要编一个角度去建议用户
        换模式。算法直接用 `stereo.stereo_mode.optical_axes_angle_deg`（入口
        `run_stereo_video` 打印的是同一个函数），不在这里另算一套。
        """
        raw = getattr(self.rig, "angle_deg", None)
        if raw is None:
            raw = (getattr(self.rig, "meta", None) or {}).get("angle_deg")
        if raw is not None:
            try:
                return float(raw)
            except (TypeError, ValueError):
                return None
        R = getattr(self.rig, "R", None)
        if R is None:
            return None
        try:
            from .stereo_mode import optical_axes_angle_deg
            return float(optical_axes_angle_deg(R))
        except Exception:                                   # noqa: BLE001
            return None                                  # 诊断路径不该因它崩

    def _angled_suggestion(self) -> str:
        """「试 angled」这半句，只在装置**标称**就是大角度时才给（否则空串）。

        判据只有两条：本模式是 parallel，且光轴夹角 > 2°。2° 的门槛很松
        （本项目的平行装置实测 0.7°），它挡的是"根本没这回事"的情况 ——
        2026-10-30 那份数据夹角 0.7°，却被无条件建议"用 angled 重跑"，
        而它交会角只有 3.6~4.8°（< angled 门限 5°），照做 100% 被拒。
        """
        if not self._is_parallel():
            return ""
        angle = self._optical_axis_angle_deg()
        if angle is None or angle <= 2.0:
            return ""
        return f"装置标称光轴夹角 {angle:.1f}°（> 2°），也可试 --stereo-mode angled"

    def _low_rate_advice(self) -> str:
        """低配对率那一句建议。分支口径与 `_zero_match_message` **同源**（`dominant_cause`）。

        为什么不能无条件写"若装置是大角度，请用 --stereo-mode angled 重跑"
        （2026-10-30 实测）：那段录像的交会角只有 3.6~4.8°（< angled 的门限
        `min_angle_deg=5.0`）、残差 3.9~7.2 px（> `max_reproj_px=3.0`），真按它说的
        改，angled 会 100% 被拒 —— 把人从一个坑引到另一个坑，而且和结束时的
        [诊断]（"几何与标定不符"）直接打架。

        运行期只给一行：完整的三步建议在结束时的 [诊断]/报错里，这里重复一遍
        只会把日志刷长。
        """
        base = f"本模式只比较 y 方向、max_epipolar={self.cfg.max_epipolar_px:g}px；"
        # 收尾统一成括号形式，三支完全一致：读者一眼知道完整结论在下面。
        end = "（详见结束时的 [诊断]）。"
        kind = self.dominant_cause()
        seeds = (f"{self.source_hint}/derived/seeds.json" if self.source_hint
                 else "<会话目录>/derived/seeds.json")
        if kind == "detect":
            return (base + "主要问题是检测器没给出圆心（有帧缺左/缺右）；"
                    f"镜面球可加 --method specular --seed-json {seeds}" + end)
        if kind == "mixed":
            # 混合型不能写成"左右都检出了"（那样是假话）：有帧缺、也有帧被拒。
            return (base + "既有帧没检出、也有候选被门限拒掉：先补检测"
                    f"（镜面球用 --method specular --seed-json {seeds}），"
                    "再看门限与标定" + end)
        advice = (base + "左右都检出了、是候选被门限拒掉：检查标定是否被碰动、"
                          "左右是否接反")
        hint = self._angled_suggestion()
        if hint:
            advice += "；" + hint
        return advice + end

    def _zero_match_message(self) -> str:
        """0 配对时的报错。**按分项给不同建议** —— 三种原因的下一步完全不同。

        为什么不能把三种原因混成一句、建议只指向 angled（2026-10-30 实测）：
        用户那次失败的真实原因是"左路 20 帧一帧都没检出"（镜面球 + 默认 `color`
        检测器，见交接文档 §39.3），却被告知"装置夹角可能远大于 parallel 的假设，
        用 angled 重跑" —— 照做只会再失败一次，因为问题根本不在几何。

        判据：`_match_frame` 里每一帧只会落进三个计数器之一，所以
        ``n_no_left + n_no_right + n_rejected == n_frames`` 恒成立，
        于是"纯检测问题"等价于 ``n_rejected == 0``，"纯配对问题"等价于
        ``n_no_left == n_no_right == 0``，其余是两者都有。
        """
        cfg = self.cfg
        n = self.n_frames
        det_miss = self.n_no_left + self.n_no_right
        # 路径没给就写成占位符：宁可像一个模板，也不要凭空编一个会话路径。
        seeds = (f"{self.source_hint}/derived/seeds.json" if self.source_hint
                 else "<会话目录>/derived/seeds.json")
        head = (f"parallel 模式下 0 个配对（共 {n} 帧）。分项："
                f"左路缺 {self.n_no_left} 帧 / 右路无候选 {self.n_no_right} 帧 / "
                f"有候选但被极线或视差条件拒掉 {self.n_rejected} 帧。")

        detect_advice = (
            "检测器根本没给出圆心 —— 这一条与极线、装置夹角、标定**都无关**，"
            "先别改 --stereo-mode。镜面/高光球在 color/hough 下常常恒为 0 检出，"
            f"下一步：--method specular --seed-json {seeds}"
            "（GUI 上是第 3 步『点选球心』）。")
        match_advice = (
            f"两路都检出了，候选被 y 方向门限拒掉（max_epipolar="
            f"{cfg.max_epipolar_px:g}px，本模式只比较 y）。下一步：重新标定"
            "（相机可能被碰动）；或确认极线偏差分布后再决定是否放宽 --max-epipolar")
        hint = self._angled_suggestion()
        if hint:
            match_advice += "；" + hint
        match_advice += "。"

        kind = self.dominant_cause()
        if kind == "detect":
            msg = head + "原因：每一帧都至少有一路没检出 —— " + detect_advice
        elif kind == "match":
            msg = head + "原因：" + match_advice
        else:
            msg = (head + f"原因：检测（{det_miss} 帧）与配对（{self.n_rejected} 帧）各占一部分，"
                   "两条都要查。\n  第 1 步 检测：" + detect_advice
                   + "\n  第 2 步 若检测补上后仍然配不上：" + match_advice)
        return msg + self._geometry_suffix()

    def _geometry_suffix(self) -> str:
        """把几何一致性诊断接在报错后面（样本不足时 `geometry_lines()` 是空的）。"""
        lines = self.geometry_lines()
        return "" if not lines else "\n" + "\n".join(lines)

    def geometry_failure_message(self, v: GeometryVerdict | None = None,
                                 center_box_share: float | None = None) -> str:
        """"录像与标定对不上"的报错文案（**判据不在这里**，判在 :func:`geometry_verdict`）。

        为什么要单独有这句话：`B-131`（2026-10-30 实测）那段会话 dy 中位
        11.3~12.4 px、匹配率 1/30，而用户唯一的反馈是"整段跑完之后的 [诊断]"。
        把它提前说，省下的不只是时间 —— 更重要的是**不给"再调调 --max-epipolar"
        留时间**：把门限放大只会把错配变成看起来正常的数字。

        `center_box_share` 是**召回前**先排除的近因：圆心没能钉到轮廓上的帧占比
        （`center_source == "box"`，由调用方从检测器统计）。`B-132` 已量过：左右高光
        在球面上落在不同位置，圆心退回框中心时 y 差本就错开 ≈6 px —— 与"几何坏掉"
        长得一样，但修法相反。不说出来，程序会把人引去重标定（冤枉且费时）。

        调用方负责决定"要不要因此停下"（`run_stereo_video` 的录前守卫就是停下来）；
        这里只负责把"差在哪、下一步做什么"写清楚。
        """
        v = v or geometry_verdict(self.dy_samples, getattr(self.rig, "quality", None))
        if v is None:                                        # 样本不足：不编结论
            return ""
        ref = (f"标定自报 极线误差 {v.epipolar_rms:.2f} px，相差 {v.ratio:.0f} 倍"
               if v.has_ref else "标定没给自报极线误差，按绝对值判（中位 ≥ 3 px 即可疑）")
        # 只有"多数帧都没钉到轮廓"才提这条：少数帧失败不足以解释整段的中位偏差。
        caveat = ""
        if center_box_share is not None and center_box_share > 0.5:
            # 措辞是"检出"不是"帧"：调用方按**左右两侧的检出**计数，与逐帧的中位
            # 偏差分母不同（少数帧只有一路检出）。说"帧"会比实际略偏。
            caveat = (
                f"⚠ 但先排除一个更近的原因：这段录像的左右检出里有 {center_box_share:.0%} "
                "的球心**没钉到轮廓上**（center_source=box，轮廓梯度不足、亚像素细化失败）。\n"
                "  左右高光在球面上落在不同位置，圆心退回框中心时 y 差本就错开 ≈6 px，"
                "**与几何坏掉长得一样**。\n"
                "  先让轮廓出得来（曝光/对焦/补光）再重跑：y 差回落即说明是圆心口径问题，"
                "不是标定问题。\n")
        return (
            f"parallel 模式：判出**录像的几何与标定对不上** —— 左右圆心（校正后）y 差"
            f"实测中位 {v.med:.1f} px（p95 {v.p95:.1f}，n={v.n}），{ref}；"
            f"当前配对 {self.n_matched}/{self.n_frames}。\n"
            "这不是『门限太紧』，是**这段录像不能用来重建**：放宽 --max-epipolar 只会"
            "把错配变成看起来正常的数字（残差与交会角会同时变差）。\n"
            + caveat
            + "  第 1 步 确认装置没被碰动：相机是否挪过、改过对焦、左/右是否接反"
            "（接反时画面完全正常、标定也能过）。\n"
            "  第 2 步 重新标定（启动程序.bat calibrate）后再重录这一段。\n"
            "  第 3 步 只想看几何差多少、不要它拦：加 --geometry-guard 0。"
            + self._geometry_suffix())

    def geometry_lines(self) -> list[str]:
        """量出"这段录像的极线几何对不对"，只测量、不改任何门限。

        为什么必须由程序说出来：低配对率时用户唯一看到的是"配不上"，最自然的
        反应就是把 `--max_epipolar` 往上调。2026-10-30 那场实测是 中位 12.4 px
        vs 标定自报 0.56 px（差 22 倍）—— 那是"录像和标定对不上"，调到 15 px
        能"跑通"，但配上的点残差 3.9~7.2 px、交会角 4.3°，本来就是错的。
        所以这里给的是"重新标定"，不是"放宽"。

        样本不足（两路都检出的帧 < `GEOM_MIN_SAMPLES`）时返回空：20 帧以下
        的中位数会误导人，宁可不说。标定自报值取不到时省略那半句，**不编数**。

        判据本身在 :func:`geometry_verdict` 里（运行期提前判读走的是同一个）。
        """
        v = geometry_verdict(self.dy_samples, getattr(self.rig, "quality", None))
        if v is None:
            return []
        lines = [f"[诊断] 左右圆心的 y 差实测：中位 {v.med:.1f} px"
                 f"（p95 {v.p95:.1f}，n={v.n}）"]
        if v.has_ref:
            tail = f"（max {v.epipolar_max:.2f}）" if math.isfinite(v.epipolar_max) else ""
            lines.append(f"       这段录像的标定自报 极线误差 {v.epipolar_rms:.2f} px{tail}")
            lead = f"相差 {v.ratio:.0f} 倍："
        else:
            lead = ""
        if v.suspect:
            lines.append(f"       → {lead}录制时的实际几何与标定不一致"
                         "（相机被碰动 / 改过对焦 / 换过装置）")
            lines.append("       → 继续放宽 max_epipolar 只会把错误配对变成数字，"
                         "建议**重新标定**")
        elif v.has_ref:
            lines.append(f"       → 与标定相符（{v.ratio:.1f} 倍）：极线几何没问题，"
                         "低配对率另有原因（检测丢失 / 遮挡 / 配对策略）")
        else:
            lines.append("       → 实测值本身也不大：极线几何看起来没问题，"
                         "低配对率另有原因（检测丢失 / 遮挡 / 配对策略）")
        return lines

    def low_match_diagnostic(self, low_rate: float = LOW_MATCH_RATE_REPORT) -> list[str]:
        """配对率低于 `low_rate` 时给出几何一致性诊断，否则返回空列表。

        `low_rate` 留成参数只为一件事：判据能直接钉住"低于 10% 就说"这个行为，
        而不用去跑一段视频。调用方一律用默认值。
        """
        if self.n_frames == 0 or self.match_rate >= low_rate:
            return []
        return self.geometry_lines()

    def diagnostics(self) -> dict:
        """结构化诊断（计数 + 配对率 + 警告），给报告/日志用。"""
        return {
            "n_frames": self.n_frames,
            "n_matched": self.n_matched,
            "n_no_left": self.n_no_left,
            "n_no_right": self.n_no_right,
            "n_rejected": self.n_rejected,
            "match_rate": self.match_rate,
            "stereo_mode": "angled" if not self._is_parallel() else "parallel",
            "warnings": list(self.warnings),
        }

    @property
    def match_rate(self) -> float:
        return self.n_matched / self.n_frames if self.n_frames else 0.0

    def as_array(self) -> np.ndarray:
        """(N,4) 的 [t, X, Y, Z]，方便直接喂给平滑/预测。"""
        if not self.trajectory:
            return np.zeros((0, 4))
        return np.array([[p.t, *p.xyz] for p in self.trajectory], np.float64)

    def summary(self) -> str:
        if not self.trajectory:
            return (f"没有任何配对成功的三维点"
                    f"（左路缺 {self.n_no_left} 帧 / 右路无候选 {self.n_no_right} 帧 / "
                    f"被条件拒掉 {self.n_rejected} 帧）")
        res = np.array([p.residual_px for p in self.trajectory], np.float64)
        res = res[np.isfinite(res)]
        z = np.array([p.xyz[2] for p in self.trajectory], np.float64)
        return (f"配对率 {self.match_rate * 100:.1f}%  ({self.n_matched}/{self.n_frames})  "
                f"交会残差 中位 {np.median(res):.3f} px / 最大 {res.max():.3f} px  "
                f"深度 {z.min():.2f}–{z.max():.2f} m")
