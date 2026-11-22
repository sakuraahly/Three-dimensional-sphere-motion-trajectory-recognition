"""自动挑帧：决定"这一帧该不该收"。

标定质量的瓶颈往往不是算法，而是**收进来的数据**：
模糊的、贴边的、全挤在同一个角度的图，会把标定结果拖坏而且看不出来。
这里把"好数据"的标准写成可执行的判据，让程序自己挑。
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field

import cv2
import numpy as np

from .board import BoardObservation, BoardSpec, detect_board, detect_board_in_roi, pose_distance

# 库层只取 logger、只写记录；装 handler / 定路径是 app 的事（见 apps/_common.setup_logging）。
logger = logging.getLogger(__name__)


def reject_category(reason: str) -> str:
    """把一条拒绝原因归到**稳定类别**，用于计数。

    为什么必须归一化：原因里嵌着每帧都不同的数字 ——
    "左图模糊(清晰度 42)" 和 "左图模糊(清晰度 51)" 是**两个不同的字符串**。
    直接拿字符串当 key 计数，统计会被打散成几百个只出现一次的条目，
    聚合出来的"最常见原因"完全没有意义 —— 而这恰恰是诊断卡在哪一步的唯一线索。

    规则：截到第一个括号为止，再去掉残留的数字和空白。
    完整原因（含数字）仍然写进日志的逐帧记录，聚合与细节都不丢。
    """
    head = re.split(r"[（(]", reason, maxsplit=1)[0]
    head = re.sub(r"\d+(\.\d+)?\s*(px|%|级)?", "", head)
    return head.strip(" ：:，,。.")


def focus_gate(
    focus: float,
    *,
    label: str = "",
    median_focus: float = float("nan"),
    n_samples: int = 0,
    min_ratio: float = 0.4,
    floor: float = 300.0,
    min_samples: int = 5,
    abs_min: float | None = None,
) -> tuple[bool, str]:
    """清晰度判据：**收还是不收**。返回 ``(可收, 不可收的原因)``。

    纯函数 —— 采集侧（``AutoPairCollector``）、命令行收图器
    （``apps/tools/capture_board.py``）、复检（``apps/verify_stereo.py``）
    都调这一个，不许各写一套：同一条规则写两遍，早晚会分叉成
    "A 放行 B 拒收"，而同一份数据的两种结论是最难查的一类不一致。

    为什么用**相对**判据：``focus`` 没有绝对刻度。同一机位只改对比度，
    这个数就会成倍变化；因此任何"focus < 常数"的门限要么永远不触发
    （常数只要低于检测器能检出的下限，就再也拦不住任何东西 —— 表面有闸、
    实际全放行），要么误杀清晰帧。可靠的信息是**组内相对位置**：
    同一组数据里，明显比中位数糊的那几帧才是真问题。

    判据顺序（先绝对、后相对）：
      1. ``abs_min`` 给了且不达标 -> 拒（老命令的 ``--min-sharpness``，兼容用）；
      2. ``focus < floor`` -> 拒（只拦真正垃圾帧）；
      3. ``n_samples >= min_samples`` 且 ``focus < min_ratio * median`` -> 拒；
      4. 其余全收 —— **样本不足时不做相对判断**，否则前几帧没有基准，
         正常的冷启动会被当成"异常糊"误杀。
    """
    prefix = f"{label}图" if label else "画面"
    if abs_min is not None and focus < abs_min:
        return False, f"{prefix}模糊（focus {focus:.0f} < 绝对下限 {abs_min:.0f}）"
    if focus < floor:
        return False, f"{prefix}糊（focus {focus:.0f} < 地板 {floor:.0f}）"
    med = float(median_focus)
    if (n_samples >= min_samples and np.isfinite(med) and med > 0
            and focus < min_ratio * med):
        return False, (f"{prefix}偏糊（focus {focus:.0f}，只有本组中位数 "
                       f"{med:.0f} 的 {focus / med * 100:.0f}%，"
                       f"阈值 {min_ratio * 100:.0f}%）")
    return True, ""


@dataclass
class CaptureConfig:
    """自动挑帧的门槛。默认值偏严，宁可少收几帧也要收干净的。"""

    target_pairs: int = 25          # 收够多少对就停
    min_pairs: int = 10             # 少于这么多对不做标定
    # ---- 清晰度判据（**相对**，见 focus_gate）----
    #
    # 为什么是相对判据：`BoardObservation.focus` 是个**没有绝对刻度**的量 ——
    # 同一台相机、同一块板，仅改对比度/曝光，focus 就能差几倍。所以
    # "focus < 某个常数就拒" 不是可用的判据：把常数设低（旧值 55、800）它在
    # 任何被检出过的帧上都不会触发，等于没有闸门；设高就会误杀清晰帧。
    # 这里改成只拒**明显比本组中位数糊**的帧，整组都软则只提示、不拒收。
    min_focus_ratio: float = 0.4   # 低于本组中位数的这个比例才拒（相对闸门）
    focus_floor: float = 300.0     # 绝对地板：只拦真正垃圾帧（远低于 SB 可检出下限）
    focus_min_samples: int = 5     # 样本不足时不做相对判断（冷启动没有基准）
    # 兼容用的**额外绝对下限**（语义是 focus）。非 None 时参与判定，
    # 但它在任何被检出的帧上都远低于实际值（SB 检出下限远高于此），
    # 所以不构成有效闸门 —— 真正生效的是上面的 min_focus_ratio。
    min_sharpness: float | None = 800.0
    # 整组收完后若 focus 中位数低于此值，打印**一次**提示（**不拒收**）：
    # 能标定，但角点定位精度会略降。
    soft_focus_hint: float = 4000.0
    min_area_ratio: float = 0.012   # 棋盘至少占画面面积的比例（别太远）
    max_area_ratio: float = 0.90    # 也别贴到脸上
    min_margin_px: float = 10.0     # 角点离画面边界至少多少像素（边缘畸变大）
    min_brightness: float = 35.0    # 防过暗
    max_brightness: float = 230.0   # 防过曝
    # 是否要求"板在两路里都基本静止"。
    #   True（默认）：实时采集、以及"手持板走一段停一下"的视频 —— 这个门限会自动
    #                 挑出人停稳的那几帧，同时**顺带解决了左右不同步的问题**
    #                 （静态目标在任意时刻看起来都一样）。
    #   False：独立静帧（图片文件夹、每帧都是不同位姿）时用，此时帧间本来就会大变。
    require_stationary: bool = True
    stationary_max_px: float = 3.0  # 连续两帧位姿变化上限（板要基本静止）
    diversity_min: float = 0.055    # 与已收位姿的最小差异（占图像对角线比例）
    region_grid: int = 3            # 画面分几宫格
    max_per_region: int = 5         # 每宫格最多收几对，强制覆盖全画面
    # 每帧做判断用的"粗检"分辨率。全图精确检测要 300 ms，撑不住实时；
    # 粗检 640 宽只要约 12 ms，足够判断"值不值得收"。真正收帧时再在棋盘
    # 附近做全分辨率精确检测（裁剪不损失分辨率）。
    gate_work_width: int = 640


@dataclass
class CapturedPair:
    corners_left: np.ndarray
    corners_right: np.ndarray
    homography_left: np.ndarray
    homography_right: np.ndarray
    center_left: tuple[float, float]
    quality: dict = field(default_factory=dict)


def diagnose_frame(gray: np.ndarray, work_width: int = 640) -> str:
    """棋盘没检测到时，用**整幅画面**的统计量给一句人话会诊。

    为什么需要：`findChessboardCornersSB` 返回"没找到"是一个黑盒 ——
    画面蒙雾、过曝、太暗、跑焦、板错了(行列数/方格数不符)都会得到同一个结果。
    在标定窗口前反复调整却没有任何反馈，是最耗时间的一种卡住。
    这里只用几个极便宜的统计量（缩到 320 宽算），把最常见的原因指出来。

    返回空字符串表示"画面本身看不出毛病"（那多半是板不对或没在画面里）。
    """
    try:
        h, w = gray.shape[:2]
        if h < 8 or w < 8:
            return ""
        scale = 320.0 / float(w)
        small = cv2.resize(gray, (320, max(8, int(round(h * scale)))),
                           interpolation=cv2.INTER_AREA)
    except cv2.error:
        return ""

    mean = float(small.mean())
    white = float((small > 245).mean())      # 过曝比例
    black = float((small < 12).mean())       # 死黑比例
    sharp = float(cv2.Laplacian(small, cv2.CV_64F).var())
    # 实际用到的动态范围。**"蒙雾"的指纹在这里，不在整幅标准差上**：
    # 蒙雾/脏镜头/起雾会把黑压不到底、白提不到顶，直方图挤成很窄一条；
    # 而画面里有大块纯色（白墙、木桌）也会让标准差偏大，用标准差判会漏判。
    p5, p95 = (float(v) for v in np.percentile(small, (5.0, 95.0)))
    spread = p95 - p5

    if mean < 12:
        return "整幅几乎全黑 —— 镜头被挡住了，或者这条流根本没打开"
    if black > 0.6:
        return f"画面 {black * 100:.0f}% 是死黑 —— 镜头被挡住或没打开"
    if mean < 30:
        return f"画面过暗（均值 {mean:.0f}）—— 补光或调高曝光"
    if white > 0.45 or mean > 240:
        return (f"画面 {white * 100:.0f}% 过曝（均值 {mean:.0f}）——"
                f"多半是逆光/正对强光源，或自动曝光被拉满；"
                f"挪开强光或关掉自动曝光再试")
    if spread < 70.0 and mean > 110:
        return (f"画面动态范围被压到只剩 {spread:.0f} 级（均值 {mean:.0f}）——"
                f"像蒙了一层雾：先擦镜头、撕掉保护膜，检查有没有起雾/结露、"
                f"或有没有东西紧贴镜头")
    if sharp < 25.0:
        return (f"画面细节很少（清晰度 {sharp:.1f}）——"
                f"跑焦或严重运动模糊；对准远处有细节的地方看能不能合焦")
    if white > 0.12:
        return (f"画面有 {white * 100:.0f}% 过曝区 ——"
                f"强光直射会把棋盘对比吃掉，转个方向或降低曝光")
    return ""


class AutoPairCollector:
    """自动收集标定图像对。

    最重要的一条判据是 **stationary（板在两路里都基本静止）**：
    它保证两台相机拍到的是同一个位姿。有了它，
    两路视频即使没有做时间同步，标定也不受影响 ——
    因为静态目标在任意时刻看起来都一样。
    """

    def __init__(self, spec: BoardSpec, cfg: CaptureConfig | None = None) -> None:
        self.spec = spec
        self.cfg = cfg or CaptureConfig()
        self.pairs: list[CapturedPair] = []
        self._prev_h: dict[str, np.ndarray | None] = {"left": None, "right": None}
        self._region_count: dict[tuple[int, int], int] = {}
        # 计数按**稳定类别**走（原因里带数字，直接当 key 会把统计打散）
        self.rejected: dict[str, int] = {}
        self.reject_examples: dict[str, str] = {}   # 类别 -> 最近一次完整原因
        self.frames = 0                              # 看过的总帧数
        self.last_reason = ""
        self.image_size: tuple[int, int] | None = None  # (w, h)
        # 本帧的粗检结果，供界面直接画框（避免重复检测 —— 那是主要的卡顿来源）
        self.last_obs: tuple[BoardObservation | None, BoardObservation | None] = (None, None)
        self.last_accepted = False
        # ---- 板占画面比例的历史（给"实时提示"和"判定建议"用）----
        #
        # 为什么要一直记：`min_area_ratio` 的拒收信息里**本来就有**这个数字
        # （"左图棋盘太小(1.2%)"），但它只出现在**被拒**的那一刻，
        # 而且只进 `rejected` 计数。结果是用户收完 25 对才从汇总里看到
        # "6.8% 太小" —— 那时板已经收起来了，白举一场。
        #
        # 实测那次标定：**板只占画面 1.2%**，角点定位精度直接受影响，
        # RMS 因此偏高（0.91 px vs ok 门限 0.80）。而 1.2% 这个数字
        # 全程都算得出来，却**没有任何出口** —— 这是"没有出口的字段"的又一個实例。
        self.area_seen: list[float] = []            # 每次算出的比例（各取左右较小者）
        self.frames_with_board = 0                  # 至少一路看到棋盘的帧数
        # 已收下各帧的 focus（按左右分开）：相对判据的基准就是它们的中位数。
        # 只用**已收下**的帧 —— 被拒的帧（比如真的糊了）不该把基准拉低，
        # 否则一组"越来越糊"的帧会互相掩护着全进来。
        self._focus_hist: dict[str, list[float]] = {"左": [], "右": []}
        # "画面偏软"提示只打一次：站在相机前的人需要知道一次，不需要每帧刷屏。
        self._soft_hinted = False

    # ------------------------------------------------------------------
    @property
    def area_ratio_median(self) -> float:
        """板占画面的中位比例（没有样本时返回 nan）。"""
        if not self.area_seen:
            return float("nan")
        return float(np.median(self.area_seen))

    @property
    def area_ratio_max(self) -> float:
        """见过的最大的板占比例 —— 用户"最靠近"到什么程度。"""
        return float(max(self.area_seen)) if self.area_seen else float("nan")

    def area_hint(self) -> str:
        """**实时**建议：板该离多近。

        `min_area_ratio` 是 1.2%，但那是"最远能收"的**下限**。
        实测表明卡在下限（1.2%）会让 RMS 明显偏高 —— 因为角点像素少、
        定位精度低。所以真正的建议是**靠近到 10% 以上**。
        """
        med = self.area_ratio_median
        if not np.isfinite(med):
            return "还没测到标定板"
        pct = med * 100
        if pct < 3.0:
            return (f"板只占画面 {pct:.1f}% —— **太远了**，"
                    f"请靠近到 10% 以上（角点像素越多，标定越准）")
        if pct < 8.0:
            return f"板占画面 {pct:.1f}% —— 建议再靠近些（目标 10%~30%）"
        if pct > 60.0:
            return f"板占画面 {pct:.1f}% —— 有点太近，边角可能出画"
        return f"板占画面 {pct:.1f}% —— 距离合适"

    # ------------------------------------------------------------------
    @property
    def count(self) -> int:
        return len(self.pairs)

    @property
    def full(self) -> bool:
        return self.count >= self.cfg.target_pairs

    @property
    def enough(self) -> bool:
        return self.count >= self.cfg.min_pairs

    def reset(self) -> None:
        self.pairs.clear()
        self._prev_h = {"left": None, "right": None}
        self._region_count.clear()
        self.rejected.clear()
        self.area_seen.clear()
        self.frames_with_board = 0
        self._focus_hist = {"左": [], "右": []}
        self._soft_hinted = False
        self.last_obs = (None, None)
        self.last_accepted = False

    def _focus_verdict(self, name: str, focus: float) -> tuple[bool, str]:
        """对某一路的一帧套用**同一条**清晰度判据（见 ``focus_gate``）。"""
        hist = self._focus_hist[name]
        med = float(np.median(hist)) if hist else float("nan")
        return focus_gate(
            focus, label=name, median_focus=med, n_samples=len(hist),
            min_ratio=self.cfg.min_focus_ratio, floor=self.cfg.focus_floor,
            min_samples=self.cfg.focus_min_samples, abs_min=self.cfg.min_sharpness,
        )

    def _hint_if_soft(self) -> None:
        """整组收完后，若**某一路**偏软，按路提示（不拒收、不影响结果）。

        为什么只在"整组收完"时说：单帧的绝对值没有刻度（见 ``focus_gate``），
        逐帧提示会对着正常的帧反复报软；而一组收完后看中位数才有意义。

        为什么**按路**看而不是两路合并：两路的 focus 没有共同刻度，合并后
        一强一弱会被平均掉 —— 左 3000 / 右 8000 合并成 5500，左路的软
        就再也提示不出来了。所以每路各看各的中位数，软哪路点名哪路。
        """
        if self._soft_hinted or not self.full:
            return
        self._soft_hinted = True
        for name in ("左", "右"):
            vals = self._focus_hist[name]
            if not vals:
                continue
            med = float(np.median(vals))
            if med >= self.cfg.soft_focus_hint:
                continue
            msg = (f"[提示] {name}图整体偏软（focus 中位数 {med:.0f}）——能标定，"
                   f"但该路精度会略降；条件允许请对焦后重拍。")
            print(msg)
            logger.info("%s", msg)

    def _reject(self, reason: str) -> tuple[bool, str]:
        """记一次拒绝。

        计数按**归一化类别**走（见 ``reject_category``），完整原因单独留一份样本 ——
        没有这一步，"收了 8 对就不动了"永远只能靠人在窗口前猜。
        """
        key = reject_category(reason) or reason
        self.rejected[key] = self.rejected.get(key, 0) + 1
        self.reject_examples[key] = reason
        self.last_reason = reason
        logger.debug("拒收 #%d  %s", self.reject_total, reason)
        return False, reason

    # ------------------------------------------------------------------
    def offer(
        self,
        left_gray: np.ndarray,
        right_gray: np.ndarray,
        spec: BoardSpec | None = None,
    ) -> tuple[bool, str]:
        """尝试收一帧。返回 (是否收下, 原因)。"""
        cfg = self.cfg
        spec = spec or self.spec
        self.frames += 1
        h, w = left_gray.shape[:2]
        self.image_size = (w, h)
        diag = math.hypot(w, h)

        # 每帧只做"粗检"：降分辨率 + 不开 EXHAUSTIVE，约 12 ms/路
        #
        # ⚠️ 这里**显式打开**经典回退，是有意的，理由有实测依据：
        # 强行模糊对照（outputs/_probe_detector_coverage.py）显示，高斯模糊 k=15 时
        # **SB 失败而经典成功** —— 真机上"左路偏糊"正是已知现象，关掉回退会真的少收帧。
        # 但回退有代价：经典与 SB 的角点顺序**整体反序**。所以策略是
        # "**允许回退，但绝不允许一对之内混用**" —— 见下面 0) 与 7b) 两道守卫。
        obs_l = detect_board(left_gray, spec, work_width=cfg.gate_work_width,
                             exhaustive=False, allow_classic_fallback=True)
        obs_r = detect_board(right_gray, spec, work_width=cfg.gate_work_width,
                             exhaustive=False, allow_classic_fallback=True)
        self.last_obs = (obs_l, obs_r)
        self.last_accepted = False

        # 这里**不是**一条门限，只是记录 —— 所以编号单独用 ⓿，
        # 免得后面 1)~N) 的门限序号被搅乱。
        #    取左右**较小**者：双目要两路都够大才有意义。
        _areas = [o.area_ratio for o in (obs_l, obs_r) if o.found]
        if _areas:
            self.area_seen.append(min(_areas))
            self.frames_with_board += 1

        # 1) **角点出身必须两路一致**
        #    经典与 SB 的角点顺序整体反序（实测直接对应 RMS 323.112 px、整份反序
        #    0.020 px），单目标定免疫、双目标定不免疫。所以"两路出身不同"这一帧
        #    必须当场丢掉，而不是留到标定结果里去猜。
        #    放在最前面：这一步花钱最少、后果最严重。
        if obs_l.found and obs_r.found and obs_l.detector != obs_r.detector:
            return self._reject(
                f"两路角点出身不同（左 {obs_l.detector} / 右 {obs_r.detector}）"
                f"：顺序相反，标定会解出垃圾 R/T")

        # 2) 两路都要检测到
        #    这里**必须说清是哪一路**没找到：两路画面质量差很多时（比如一路蒙雾、
        #    一路正常），只说"两路未同时检测到"会让人以为是标定板的问题，
        #    在窗口前反复调整却查不到真正的原因。顺手给一句画面会诊。
        if not obs_l.found or not obs_r.found:
            if not obs_l.found and not obs_r.found:
                why = "两路都未检测到棋盘"
            elif not obs_l.found:
                why = "左路未检测到棋盘（右路正常）"
            else:
                why = "右路未检测到棋盘（左路正常）"
            notes = []
            for name, obs, gray in (("左", obs_l, left_gray), ("右", obs_r, right_gray)):
                if not obs.found:
                    note = diagnose_frame(gray, cfg.gate_work_width)
                    if note:
                        notes.append(f"{name}：{note}")
            if notes:
                why += "｜" + "；".join(notes)
            return self._reject(why)
        assert obs_l.corners is not None and obs_r.corners is not None
        # 2) 清晰度（**尺度无关**的 focus + **相对**判据 —— 理由见 focus_gate）
        for name, obs in (("左", obs_l), ("右", obs_r)):
            can_focus, why_focus = self._focus_verdict(name, obs.focus)
            if not can_focus:
                return self._reject(why_focus)

        # 3) 大小与位置
        for name, obs in (("左", obs_l), ("右", obs_r)):
            if obs.area_ratio < cfg.min_area_ratio:
                return self._reject(f"{name}图棋盘太小({obs.area_ratio * 100:.1f}%)")
            if obs.area_ratio > cfg.max_area_ratio:
                return self._reject(f"{name}图棋盘太近")
            if obs.margin_px < cfg.min_margin_px:
                return self._reject(f"{name}图棋盘贴边({obs.margin_px:.0f}px)")
            if not (cfg.min_brightness <= obs.brightness <= cfg.max_brightness):
                return self._reject(f"{name}图曝光不合适({obs.brightness:.0f})")

        # 4) 静止：两路都必须和上一帧几乎一样
        #    这条同时解决了两路不同步的问题（静态目标在任意时刻都一样）
        if cfg.require_stationary:
            for side, obs in (("left", obs_l), ("right", obs_r)):
                prev = self._prev_h[side]
                if prev is not None:
                    d = pose_distance(prev, obs.homography, diag)
                    if d * diag > cfg.stationary_max_px:
                        self._prev_h["left"], self._prev_h["right"] = obs_l.homography, obs_r.homography
                        return self._reject("棋盘还在移动，请停稳再收")
        self._prev_h["left"], self._prev_h["right"] = obs_l.homography, obs_r.homography

        # 5) 多样性：别把 25 张全收成同一个角度
        for p in self.pairs:
            d_l = pose_distance(p.homography_left, obs_l.homography, diag)
            d_r = pose_distance(p.homography_right, obs_r.homography, diag)
            if min(d_l, d_r) < cfg.diversity_min:
                return self._reject("与已收位姿太接近")

        # 6) 每个宫格限量，强制覆盖整个画面
        cx = float(obs_l.corners.reshape(-1, 2)[:, 0].mean())
        cy = float(obs_l.corners.reshape(-1, 2)[:, 1].mean())
        gx = min(cfg.region_grid - 1, int(cx / w * cfg.region_grid))
        gy = min(cfg.region_grid - 1, int(cy / h * cfg.region_grid))
        if self._region_count.get((gx, gy), 0) >= cfg.max_per_region:
            return self._reject("该区域已收够，请换个位置")

        # 7) 全部通过 -> 在棋盘附近做全分辨率精确检测（只在这一帧做，约 40 ms/路）
        #    同样显式允许回退（理由见粗检那一段），并紧跟着核对出身
        acc_l = detect_board_in_roi(left_gray, spec, obs_l.corners, allow_classic_fallback=True)
        acc_r = detect_board_in_roi(right_gray, spec, obs_r.corners, allow_classic_fallback=True)
        if acc_l is None or acc_r is None:
            return self._reject("精确复核失败（板太靠边或被遮挡）")

        # 7b) 再核一次出身 —— 粗检一致、精确复核仍可能不一致
        #（粗检在 640 宽上跑、精确复核在全分辨率裁剪上跑，两者成败可以不同）
        if acc_l.detector != acc_r.detector:
            return self._reject(
                f"两路角点出身不同（左 {acc_l.detector or '?'} / 右 {acc_r.detector or '?'}）"
                f"：顺序相反，标定会解出垃圾 R/T")

        # 用精确角点复核关键指标，避免粗检的近似把不合格的帧放进来
        for name, acc in (("左", acc_l), ("右", acc_r)):
            can_focus, why_focus = self._focus_verdict(name, acc.focus)
            if not can_focus:
                return self._reject(f"{why_focus}（精确复核）")
            if acc.margin_px < cfg.min_margin_px:
                return self._reject(f"{name}图棋盘贴边(精确复核)")
            if acc.area_ratio < cfg.min_area_ratio:
                return self._reject(f"{name}图棋盘太小(精确复核)")

        self._region_count[(gx, gy)] = self._region_count.get((gx, gy), 0) + 1
        self.pairs.append(
            CapturedPair(
                corners_left=acc_l.corners.copy(),
                corners_right=acc_r.corners.copy(),
                homography_left=acc_l.homography.copy(),
                homography_right=acc_r.homography.copy(),
                center_left=(cx, cy),
                quality={
                    "focus_l": acc_l.focus, "focus_r": acc_r.focus,
                    "sharpness_l": acc_l.sharpness, "sharpness_r": acc_r.sharpness,
                    "square_px_l": acc_l.square_px, "square_px_r": acc_r.square_px,
                    "area_l": acc_l.area_ratio, "area_r": acc_r.area_ratio,
                },
            )
        )
        self.last_obs = (acc_l, acc_r)
        self.last_accepted = True
        self.last_reason = f"已收 {self.count}/{cfg.target_pairs}"
        # 收下的帧才进基准（见 _focus_hist 的说明）
        self._focus_hist["左"].append(float(acc_l.focus))
        self._focus_hist["右"].append(float(acc_r.focus))
        self._hint_if_soft()
        logger.info("收下第 %d/%d 对（宫格 %d,%d；focus 左%.0f 右%.0f；"
                    "原始清晰度 左%.0f 右%.0f；格子 左%.1fpx 右%.1fpx）",
                    self.count, cfg.target_pairs, gx, gy,
                    acc_l.focus, acc_r.focus,
                    acc_l.sharpness, acc_r.sharpness,
                    acc_l.square_px, acc_r.square_px)
        return True, self.last_reason

    # ------------------------------------------------------------------
    def coverage_image(
        self,
        out_size: tuple[int, int] | None = None,
        scale: float = 0.5,
    ) -> np.ndarray:
        """覆盖率示意图：已收位姿的棋盘四边形 + 宫格占用数。

        **参数语义（这里踩过坑，写清楚）**：

          - ``out_size``：**输出画布**的 (宽, 高)。给了它就按比例缩放并**居中**，
            已收位姿一定落在画布内。窗口布局要用特定尺寸的面板时给这个。
          - ``scale``：不给 ``out_size`` 时，按源图尺寸乘这个比例。

        原来的实现只有一个 ``size``，语义是"源图尺寸"，而内容按 ``scale`` 缩放 ——
        调用方给了 ``size=(190, 高)`` 又给 ``scale=1.0`` 时，画布变成 190 宽，
        但位姿仍按**全图坐标**（600~900）绘制，**整个画到画布外面去了**，
        面板看起来是空的。用户因此"看不到已经标定的姿态和位置"。
        所以现在明确区分"源"与"输出"，并自己算缩放。
        """
        src_w, src_h = self.image_size if self.image_size else (640, 480)

        if out_size is not None:
            ow, oh = max(1, int(out_size[0])), max(1, int(out_size[1]))
            s = min(ow / src_w, oh / src_h)          # 等比，保证不裁切
        else:
            s = float(scale)
            ow, oh = max(1, int(src_w * s)), max(1, int(src_h * s))

        off_x = (ow - src_w * s) / 2.0               # 居中留边
        off_y = (oh - src_h * s) / 2.0

        canvas = np.zeros((oh, ow, 3), dtype=np.uint8)
        canvas[:] = (28, 28, 32)

        def _to_canvas(pts: np.ndarray) -> np.ndarray:
            return pts * s + np.array([off_x, off_y], np.float32)

        g = self.cfg.region_grid
        for i in range(1, g):
            x = int(off_x + src_w * s * i / g)
            y = int(off_y + src_h * s * i / g)
            cv2.line(canvas, (x, 0), (x, oh), (60, 60, 66), 1)
            cv2.line(canvas, (0, y), (ow, y), (60, 60, 66), 1)

        # 宫格占用数：**直接写在格子上** —— 用户要的是"还缺哪儿"，
        # 让他自己数四边形是"需要思考"，不符合交互原则。
        cell_w, cell_h = src_w * s / g, src_h * s / g
        for (gx, gy), n in sorted(self._region_count.items()):
            cap = self.cfg.max_per_region
            full = n >= cap
            cx = int(off_x + (gx + 0.5) * cell_w)
            cy = int(off_y + (gy + 0.5) * cell_h)
            cv2.putText(canvas, f"{n}/{cap}", (cx - 12, cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (90, 220, 120) if full else (200, 200, 90), 1, cv2.LINE_AA)

        for p in self.pairs:
            quad = cv2.perspectiveTransform(
                np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32).reshape(-1, 1, 2),
                p.homography_left,
            ).reshape(-1, 2)
            cv2.polylines(canvas, [_to_canvas(quad).astype(np.int32)], True,
                          (0, 200, 90), 1, cv2.LINE_AA)
            c = _to_canvas(np.array([p.center_left], np.float32))[0]
            cv2.circle(canvas, (int(c[0]), int(c[1])), 2, (0, 220, 255), -1)

        cv2.putText(canvas, f"{len(self.pairs)}/{self.cfg.target_pairs}", (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return canvas

    def status_lines(self, top: int = 4) -> list[str]:
        """给实时窗口用的状态文本（**放在画面之外的条带里**，不遮图像）。

        每条带上"占已处理帧的百分比"，而不是只有绝对次数 ——
        ``x419`` 看不出严重程度，``419 (78%)`` 一眼就知道卡在哪。
        """
        lines = [f"已收 {self.count}/{self.cfg.target_pairs}"
                 f"   已看 {self.frames} 帧   已拒 {self.reject_total}"]
        # **实时距离提示**（2026-10-14 加）
        # 这是"用户最喜欢的一条信息"：他站在相机前面举着板，需要知道"该不该再近点"。
        # 原来这个判断只能等收完 25 对、从拒收汇总里反推出来 —— 那时板已经收起来了。
        lines.append(self.area_hint())
        lines.append("本帧: " + (self.last_reason or "-"))
        if self.rejected:
            total = max(self.reject_total, 1)
            for r, n in sorted(self.rejected.items(), key=lambda kv: -kv[1])[:top]:
                lines.append(f"    {n:>5} ({n / total * 100:4.1f}%)  {r}")
        return lines

    @property
    def reject_total(self) -> int:
        return sum(self.rejected.values())

    def summary_lines(self, top: int | None = None) -> list[str]:
        """**完整**的拒绝统计（默认全部，不截断）。

        和 ``status_lines`` 的区别：那个给屏幕、只放得下前几条；
        这个给日志和结束报告 —— **被截掉的那几条往往才是真因**。
        """
        items = sorted(self.rejected.items(), key=lambda kv: -kv[1])
        if top is not None:
            items = items[:top]
        total = max(self.reject_total, 1)
        out = [f"共处理 {self.frames} 帧：收下 {self.count} 对，拒收 {self.reject_total} 帧"]
        if not items:
            return out
        out.append("拒收原因（按次数）：")
        for r, n in items:
            out.append(f"  {n:>6} ({n / total * 100:5.1f}%)  {r}")
            example = self.reject_examples.get(r)
            if example and example != r:
                out.append(f"          例：{example}")
        return out

    def next_step_hint(self) -> str:
        """按**占比最高**的那条拒收原因，给一句能照做的话。

        为什么要它：STC/全局规则都要求"反馈引导行动"。只说"收了 8 对，
        不够 10 对"是报告问题；告诉人"把板挪到画面其他宫格"才是解决问题。
        没有这一句，人只能在窗口前反复试同一个动作。
        """
        if not self.rejected:
            return ""
        top = max(self.rejected.items(), key=lambda kv: kv[1])[0]
        pct = self.rejected[top] / max(self.reject_total, 1) * 100.0

        # 顺序有讲究：先匹配更具体的说法，避免被宽泛的规则先截走。
        rules: list[tuple[tuple[str, ...], str]] = [
            (("宫格已收满", "区域已满"),
             "板已经在这个区域收够了 —— **把板移到画面别的宫格**（上/下/左/右/角落），"
             "每个区域最多收 5 对（cfg.max_per_region）"),
            (("与已收位姿太接近",),
             "板的姿态和已收的差不多 —— **换个角度或挪个位置**再停稳；"
             "确实需要更多近似姿态时调小 diversity_min"),
            (("还在移动",),
             "板没停稳 —— 每个姿势**停住约 1 秒**再移开；手抖得厉害就加 --relaxed"),
            (("都未检测到棋盘",),
             "两路都没看到板 —— 把板举到**两路视野重叠区**（先跑 run.bat precheck 看重叠区在哪）"),
            (("未检测到棋盘",),
             "有一路没看到板 —— 看条带里那一侧的「focus / 面积」，"
             "多半是那一路画质或视野问题，跑 run.bat diag 定位"),
            (("模糊", "偏糊", "糊"),
             "画面比本组其他帧明显糊 —— 擦镜头、撕保护膜、加光，或重新对焦。"
             "（判据是**相对**的：只拒明显低于本组中位数的帧；整组偏软只会提示、不拒收）"),
            (("棋盘太小",),
             "板太远 —— **把板拿近**，或换更大的板"),
            (("棋盘贴边",),
             "板压到画面边缘 —— 让整块板都进画面，四边留出余量"),
            (("曝光", "过曝", "过暗"),
             "曝光不合适 —— 调整光照角度，避开强光直射与阴影"),
            (("太近",),
             "板离镜头太近 —— 往后退一点"),
        ]
        for keys, advice in rules:
            if any(k in top for k in keys):
                return f"占比最高的是「{top}」（{pct:.0f}%）—— {advice}"
        return f"占比最高的是「{top}」（{pct:.0f}%）—— 先按这条调整，再看日志里的完整统计"

    def log_summary(self) -> None:
        """把完整统计写进日志（终端 + 文件）。"""
        for line in self.summary_lines():
            logger.info("%s", line)
        hint = self.next_step_hint()
        if hint:
            logger.info("下一步：%s", hint)

    def side_lines(self) -> list[str]:
        """**逐路**状态：哪一路找到板、清晰度多少。

        为什么单独要这一行：两路画质差很多时（一路蒙雾一路正常），
        只有合并后的"未同时检测到"完全看不出是哪一路的问题，
        人会一直在窗口前挪板子。把这一行放上屏，一眼就知道该修哪路。
        """
        out = []
        for name, obs in (("左", self.last_obs[0]), ("右", self.last_obs[1])):
            if obs is None:
                out.append(f"{name}: -")
                continue
            if not obs.found:
                out.append(f"{name}: 未找到板")
                continue
            out.append(f"{name}: focus{obs.focus:6.0f} 清晰度{obs.sharpness:5.0f}"
                       f" 格子{obs.square_px:4.1f}px 面积{obs.area_ratio * 100:4.1f}%"
                       f" 曝光{obs.brightness:3.0f}")
        return out
