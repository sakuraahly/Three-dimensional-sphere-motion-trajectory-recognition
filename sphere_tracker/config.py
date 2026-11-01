"""集中配置。所有阈值都能存成 JSON，方便标定后复用。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# 颜色预设表：**(名字 -> HSV 参数)**，全项目只此一处。
#
# 为什么提到模块级（2026-10-27）：这张表原来藏在 `ColorConfig.preset()` 的
# 函数体里，于是"有哪些预设"这个事实**没有任何可查询的出口**，只能靠人抄 ——
# 实测被抄在了三处：
#   `apps/run_realtime.py` 的 `--preset` 帮助文本、
#   `apps/tools/calibrate_color.py` 的 `PRESETS` 列表、
#   `preset()` 自己的 `if name not in presets`。
# 抄出来的第二、三处没有任何机制保证跟上，加一个预设要改三个文件、还漏得掉。
# 现在 `preset_names()` 从这张表派生，CLI 的 `choices=` 与帮助文本都读它，
# 于是"加一种球色 = 往这里加一行"。
COLOR_PRESETS: dict[str, dict[str, Any]] = {
    # 白色 / 浅色球体（与 `ColorConfig()` 的默认值**逐值相同**，
    # 所以"不传 --preset"等效于 white_ball —— 这句话是事实，不是约定）
    "white_ball": dict(hsv_lower=[0, 0, 175], hsv_upper=[179, 80, 255]),
    # 橙色球（篮球、橙球）
    "orange_ball": dict(hsv_lower=[5, 110, 110], hsv_upper=[22, 255, 255]),
    # 绿色球
    "green_ball": dict(hsv_lower=[40, 70, 60], hsv_upper=[85, 255, 255]),
    # 蓝色球
    "blue_ball": dict(hsv_lower=[95, 90, 60], hsv_upper=[130, 255, 255]),
    # 红色球（跨 0 度，用双区间）
    "red_ball": dict(
        hsv_lower=[0, 110, 70], hsv_upper=[10, 255, 255],
        use_second_range=True,
        hsv_lower2=[170, 110, 70], hsv_upper2=[179, 255, 255],
    ),
    # 黄色球
    "yellow_ball": dict(hsv_lower=[20, 100, 100], hsv_upper=[35, 255, 255]),
}


@dataclass
class ColorConfig:
    """HSV 颜色分割参数。OpenCV 的色相 H 范围是 0-179，S/V 是 0-255。

    默认预设面向"白色 / 高亮小球"（例如乒乓球、白色橡胶球）：
    亮度高、饱和度低。若你的球是彩色的，请用 apps/tools/calibrate_color.py 标定，
    或直接改成对应的 HSV 范围。
    """

    hsv_lower: list[int] = field(default_factory=lambda: [0, 0, 175])
    hsv_upper: list[int] = field(default_factory=lambda: [179, 80, 255])
    # 形态学核大小（去噪点 / 补空洞）
    open_ksize: int = 5
    close_ksize: int = 7
    # 额外支持第二个色相区间（红色跨 0 度时需要）
    use_second_range: bool = False
    hsv_lower2: list[int] = field(default_factory=lambda: [170, 80, 80])
    hsv_upper2: list[int] = field(default_factory=lambda: [179, 255, 255])

    @staticmethod
    def preset_names() -> list[str]:
        """全部预设名，按 :data:`COLOR_PRESETS` 的**定义顺序**（= 界面展示顺序）。

        这是"有哪些预设"的唯一出口：CLI 的 `choices=`、`--help` 文本、
        HSV 标定工具的数字键都从这里取。**不要在别处再抄一份名单** ——
        `tests/selftest_entrypoints.py` 会扫出来。
        """
        return list(COLOR_PRESETS)

    @staticmethod
    def preset(name: str) -> "ColorConfig":
        """按名字取预设。未知名字抛 `KeyError` 并列出可选值。"""
        if name not in COLOR_PRESETS:
            raise KeyError(f"未知预设 {name!r}，可选：{sorted(COLOR_PRESETS)}")
        return ColorConfig(**COLOR_PRESETS[name])


@dataclass
class SpecularConfig:
    """镜面球（金属摆锤）检测器参数。

    实测（1920x1080，`data/sessions/20261019-193809`）：
      - 球半径 35~40 px（不是 640 宽预览里的 8~12 px）；
      - 球速可达 ~80 px/帧；
      - 饱和度与背景白袋几乎一样，不能靠颜色分。
    所以用 **人工 seed + CSRT 跟踪 + 模板/运动门控**；不做全图 Hough 自动锁定。
    """

    min_radius: float = 18.0
    max_radius: float = 100.0
    init_box: int = 64
    max_box: int = 140
    max_jump_px: float = 120.0   # 实测球速可达 ~80 px/帧；50 会把真球误判成跟丢
    min_ncc: float = 0.0       # 仅记录；实测镜面反射下 0.30 会把真球误杀
    min_motion: float = 0.0    # 仅记录；顶点附近球几乎静止，硬门控会误杀
    motion_grace_frames: int = 5
    max_static_frames: int = 20  # 连续这么多帧中心/局部都不动 -> 判为丢了，别继续报背景

    # ---- 丢失后的回捕（2026-11-01）----
    # 为什么要有：绳摆/手抛的球会**摆出某一路画面再回来**。旧行为是"丢一次 =
    # 后面整段空"（实测右路 80~141 之后 160 帧全丢，而球在第 250 帧明明又出现了）。
    # ⚠ **默认关闭**：原型能接回帧（右路 62→174），但在"手持/相机也会动"的录像里
    # 它会锁到静止亮块 —— 2026-11-01 实测直接锁上顶上的日光灯管，跟着它报了
    # 112 帧假数据，而"位移校验"挡不住相机自身的运动。宁可没有数据，也不能造假数据。
    # 打开它之前必须先做"相对背景运动"或形状判别（见 B-163）。
    reacquire: bool = False
    reacquire_after: int = 12        # 丢这么多帧后才开始找（先给 CSRT 自己恢复的机会）
    reacquire_ncc_local: float = 0.45   # 在上次位置附近找：阈值低一点（外观变化小）
    reacquire_ncc_global: float = 0.62  # 全图找：阈值必须高（否则会吸到台灯/白袋子）
    reacquire_confirm: int = 3          # 连续这么多帧命中同一处才重建轨（防一闪而过）
    reacquire_global_after: int = 25    # 丢这么久才允许全图搜（先只在局部找）
    # 回捕成功后必须"真的在动"：灯管、白墙、桌角这些静止亮块外观和球极像
    # （2026-11-01 实测右路回捕直接锁到日光灯管，跟着它报了 112 帧假数据）。
    # 判据：重建轨后这么多帧内，中心累计位移必须超过下面这个像素数，否则撤销。
    reacquire_verify_frames: int = 15
    reacquire_min_travel_px: float = 20.0
    reacquire_cooldown: int = 45        # 撤销后先冷却，别立刻又锁回同一个东西
    static_motion_eps: float = 1.0
    # 轮廓圆心：CSRT 锁的是**高光斑**（最显眼的特征），而高光斑在不同视角落在
    # 球面不同位置，所以左右视图的"框中心"天然错位（实测约 6 px，见 B-132）。
    # 三角化要的是**轮廓圆心**，因此跟踪得到框中心后，再用亚像素射线拟合把圆心
    # 钉到球轮廓上（半径仍取框，因为镜面球的半径无法从边缘量，见 B-130）。
    # 拟合不出足够梯度时原样保留框中心 —— 只改善、不劣化。
    refine_center: bool = True
    # 搜索半径（px）：0 = 自动（框半径的 25%，夹在下面两个界之间）。窗口跟着框半径
    # 走而不是写死 ±2.5：左右视图框半径本就不一样（17.5 vs 24.7），窗口太窄会够不到
    # 真轮廓；又不能太宽，否则会把球内部的强反射（半径明显更小）吸进来。
    refine_center_search_px: float = 0.0
    refine_center_min_search_px: float = 2.5
    # 上限从 8 提到 24（2026-11-01）：搜索窗按 0.25×半径 走，8px 的帽子是**按小球**
    # （r≈35）设的；r≈70 的球本应给 17.5px，被砍到 8 之后细化够不到真轮廓，
    # 圆心就留在 CSRT 框中心上（实测跨视图 dy −28px、落进极线门 0%）。
    # 提到 24 后同一段录像：dy 中位 −23px、落门 13%；再大（40/60）反而变差（开始吸别的）。
    refine_center_max_search_px: float = 24.0


@dataclass
class DetectorConfig:
    """检测器配置。

    method 取值：
        "color"   HSV 颜色分割 + 轮廓圆度筛选（最快，需要球与背景颜色可分）
        "hough"   霍夫圆变换（不依赖颜色，对光照和虚焦较敏感）
        "motion"  背景建模差分（仅适合摄像头固定不动）
        "fusion"  上面几种投票融合（最鲁棒，但**最慢**：见下）
        "yolo"    YOLO 目标检测（可选依赖，需 pip install ultralytics）
        "yolo_sam" YOLO 给框 + SAM 精细分割出像素级边缘（可选依赖）

    ## 默认为什么是 `color` 而不是 `fusion`（2026-10-14 改）

    同一进程内**交替**实测（1920x1080，合成场景；交替是为了避免
    "先测的那个被后测的拖慢"这种假象）：

        color   18.3 ms/帧  ->  54.6 fps
        fusion 120.6 ms/帧  ->   8.3 fps
        color   18.4 ms/帧  ->  54.2 fps   （重复一次，稳定）
        fusion 117.6 ms/帧  ->   8.5 fps

    也就是 **6.5 倍**。`fusion` 要把 `color` + `hough` + `motion` **依次跑完**，
    而全都在**原始分辨率**上（检测器不做降采样），于是单帧时间约等于三者之和；
    单看 `hough` 一个就吃掉约 80 ms。

    8~9 fps 在实时跟踪里是**不能用**的（球一帧就移动很远，预测完全跟不上），
    而用户要的是"看得见球在动"。所以默认改成 `color`：**约 54 fps**，
    颜色是最直接的线索，而本项目跟踪的就是一个**颜色已知**的球。

    代价（要说清，不能只讲好处）：`color` 单独用比 `fusion` 更容易被
    **同色的其他东西**骗到。若现场有同色干扰物，三种补救按代价从低到高：
      ① 用 `--method fusion` 换回鲁棒性，接受 8 fps；
      ② 调 `min_score` / `color_*` 门限把误检压掉（仍是 54 fps）；
      ③ 用 `--preset` 选更贴合的颜色预设。
    """

    method: str = "color"
    # 融合时启用的子检测器。这是 fusion 的**语义**配置（投票成员），不是"有哪些
    # 检测器"的清单来源 —— 清单在 `detectors/registry.py`（规则 R7）。
    # 这 3 个名字**必须**在 detector 注册表里存在：`selftest_plugins` 会断言，
    # 拼错时 fusion 会静默少一路投票（只剩 2 票，"多方一致"的门槛就松了）。
    fusion_members: list[str] = field(default_factory=lambda: ["color", "hough", "motion"])

    # ---- 尺寸/形状约束（像素）。是抑制误检最有效的一招 ----
    min_radius: float = 8.0
    max_radius: float = 200.0
    # 4*pi*A/P^2，完美圆=1。注意：真实摄像头下"圆"的圆度远达不到 1——
    # 掩膜边缘有锯齿、球面有高光和纹理缺口，实测真球只有 0.61~0.63，
    # 原来定 0.62 正好卡在临界点上，导致约 30% 的帧被误杀（合成图抗锯齿渲染
    # 圆度偏高，所以自测抓不到这个问题）。降到 0.55 后真机检出 72.5% -> 100%，
    # 且无球场景误检没有增加。
    # 排方块不归它管（正方形圆度反而是 pi/4≈0.785，比真球高），
    # 排方块/细长目标靠 min_fill_ratio 和 min_radial_consistency。
    min_circularity: float = 0.55
    min_fill_ratio: float = 0.72       # 轮廓面积 / 最小外接圆面积（方块只有 ~0.61）
    min_radial_consistency: float = 0.90   # 轮廓点到圆心距离的一致性（方块 ~0.85，圆 ~0.98）
    hough_min_support: float = 0.45    # 霍夫圆：圆上梯度必须"沿半径方向"的最低比例
    hough_min_sign_consistency: float = 0.70   # 霍夫圆：梯度符号一致性的最低要求
    # 亚像素边缘细化：把圆边界从"阈值分割的像素级"提升到"梯度峰值的亚像素级"。
    # 半径精度直接决定单目测深精度，强烈建议开启。
    subpixel_refine: bool = True
    subpixel_rays: int = 72

    # ---- 颜色 ----
    color: ColorConfig = field(default_factory=ColorConfig)

    # ---- 镜面球（金属摆锤）----
    specular: SpecularConfig = field(default_factory=SpecularConfig)

    # ---- 霍夫圆 ----
    hough_dp: float = 1.2
    hough_min_dist: float = 60.0
    hough_param1: float = 120.0        # Canny 高阈值
    hough_param2: float = 30.0         # 圆心累加器阈值，越小越容易检出也越多误检
    hough_blur_ksize: int = 7
    # ---- hough 的**搜索半径上限**（2026-10-14 加，别与 min/max_radius 混）----
    #
    # `cv2.HoughCircles` 的代价随 `maxRadius - minRadius` **线性上涨**：
    # 它要对每个边缘像素，把这一段里的半径逐个投票。
    #
    # 实测（1080p、书架那种真实纹理）：原来用 `max_radius = 200`
    # （相当于直径 400px 的球）时，hough 要 **4 秒一帧**；
    # 而同样的帧如果是纯色合成图只要 132 ms —— 真实场景的边缘像素多两三个数量级，
    # 于是"搜索 192 种半径 × 每个边缘像素"直接爆炸。
    #
    # 所以给 hough 一个**独立的**上限：物理约束 `max_radius` 还要给几何校验
    # 和 color 用，不该为一个检测器的性能去动它。
    # 80 约合 1080p 下直径 160px 的球，对"球直径 4cm、距离一两米"绰绰有余。
    hough_max_radius: float = 80.0

    # ---- 运动检测 ----
    motion_history: int = 300
    motion_var_threshold: float = 24.0
    motion_detect_shadows: bool = True

    # ---- 相机自运动补偿（手持 / 云台 / 平移拍摄）----
    # 背景差分假设背景静止；相机一平移，整幅画面都变成"前景"，误检会爆炸。
    #   auto  = 检测到相机在动时自动切换成"对齐后再差分"，静止时仍用 MOG2
    #   off   = 永远只做普通背景差分（相机固定的老场景）
    #   force = 永远走补偿差分（相机持续移动时更稳）
    motion_diff_min_threshold: float = 12.0    # 帧差的最低阈值（低于此的差异视为噪声）
    motion_contrast_min: float = 12.0          # 判定"物体真的在这里"所需的最小内外反差
    motion_compensation: str = "auto"
    camera_motion_threshold_px: float = 0.8    # 每帧全局位移超过多少像素判定"相机在动"
    motion_work_width: int = 320               # 估计相机运动时的缩放宽度（越小越快）
    motion_feature_fallback: bool = True       # 平移过大时改用特征点法（可含旋转/缩放）

    # ---- 融合 ----
    # 两个检测中心相距小于 max(agree_px, agree_ratio*半径) 视为"同意"
    agree_px: float = 6.0
    agree_ratio: float = 0.45
    # 只拿到"一票"（没有别的检测器同意）时，总分低于此值就判为不可信、不输出。
    # 注意：**只对孤证生效** —— 多票候选的总分已经加了票数加成，不走这个门限。
    # 默认值 0.55 与旧版硬编码一致（旧版此处是死配置，改了不生效，已修）。
    min_score: float = 0.55

    # ---- AI 后端 ----
    # 权重统一放在项目的 models/ 目录。路径相对项目根解析（不依赖当前工作目录）；
    # 如果文件不存在，会自动退回裸文件名，让 ultralytics 自己去找或下载。
    yolo_weights: str = "models/yolo11n.pt"   # 官方 COCO 权重含 "sports ball" 类
    yolo_conf: float = 0.25
    yolo_imgsz: int = 640
    yolo_device: str = "0"             # "0" = 第一块 GPU，"cpu" = CPU
    yolo_classes: list[int] | None = field(default_factory=lambda: [32])  # 32 = sports ball
    yolo_refine: bool = True           # 用框内颜色/霍夫把框细化成圆
    sam_model: str = "models/sam2.1_t.pt"   # ultralytics 的 SAM2 tiny（首次运行自动下载）
    sam_refine_from: str = "yolo"      # "yolo" | "hough" | "color"：用什么给 SAM 提示框


@dataclass
class TrackerConfig:
    """卡尔曼跟踪参数。"""

    # 状态阶数：1=匀速(CV)，2=匀加速(CA)。球在自由飞行/抛体时用 2 更准
    order: int = 2
    process_noise: float = 1.5e2       # 越大越信任观测、响应快但抖动
    measurement_noise: float = 2.0     # 观测噪声标准差（像素），约等于检测抖动
    gate_chi2: float = 9.21            # 2 自由度 99% 卡方门限，用于剔除野点
    rescue_radius_ratio: float = 3.0   # 门控失败后的"抢救"关联：距离小于 该倍数*球半径 仍归给已有轨迹
    rescue_min_px: float = 25.0        # "抢救"关联的最小距离阈值（像素）
    # ---- 机动检测（对付反弹/击打这种速度突变）----
    # 卡尔曼滤波是"模型驱动"的：球一落地反弹，速度方向瞬间反转，
    # 滤波器还用老速度外推，下一帧就会预测到几十像素之外，位置估计直接被带偏。
    # 靠加大过程噪声救不回来（要加大到 1e11 才能跟上，那样正常飞行又会抖成筛子），
    # 正确做法是：直接检测"急转弯"，然后把速度状态重置成刚测到的速度。
    radius_gate_ratio: float = 0.35    # 半径突变超过这个比例就不许关联（防止跟丢后跳到别的球/别的物体）
    maneuver_min_px_s: float = 250.0   # 速度突变的最小绝对值（像素/秒），低于此不认为是机动
    maneuver_ratio: float = 1.6        # 速度突变超过当前速度的多少倍才判定为机动
    max_q_scale: float = 64.0          # 自适应过程噪声最多放大到基准值的多少倍
    # 连续丢失多久后删除轨迹。
    #
    # ⚠️ 2026-10-18：**优先用秒**（max_coast_s），帧数只作兜底。
    # 只数帧会让"遮挡容忍度"随帧率变：同一段 0.3 s 遮挡，
    # 30 fps 是 9 帧（轨迹延续）、60 fps 是 18 帧（轨迹断开）。
    # 本项目两路常跑 30/60 fps，所以必须用与帧率无关的时间口径。
    max_coast_s: float = 0.4           # 秒；<=0 表示退回按帧数
    max_coast_frames: int = 12         # 兜底：不知道 fps / max_coast_s<=0 时才算它
    init_hits: int = 2                 # 连续命中多少帧才认为轨迹"确认"
    trail_length: int = 240            # 保存多少个历史点用于画拖尾
    # 轨迹点存"原始观测"还是"滤波结果"？
    # 预测器是拿这条轨迹去拟合的，而卡尔曼滤波在机动瞬间会有几十像素的
    # 滞后/超调，这些结构化误差会被拟合放大成"速度误差"。
    # 亚像素检测下，原始观测本身就很干净，所以默认直接用观测值。
    trail_source: str = "measurement"  # "measurement" | "filtered"

    # ---- 帧层半径一致性门（RadiusGate）的五个门限（2026-10-31，S1d′）----
    # ⚠️ 与上面的 `radius_gate_ratio` 是**两道不同的门**，管的是两件事，不要合并：
    #   * `radius_gate_ratio`（**关联层**，`BallTracker.update` 里用）：判"这一帧
    #     的观测配给哪条轨迹"，判不过只影响"检测配给谁"，观测本身不丢；
    #   * 下面这组（**帧层**，`RadiusGate` 里用）：判"这一帧的半径能不能进 Z"，
    #     判不过回退上一帧半径、或整帧不认（`accepted_r=None`）。
    # 拆成五个字段是为了让门限**单点可调**；唯一消费方是 `sphere_tracker.track.RadiusGate`。
    #
    # radius_max_jump（跳变护栏）：实测可测帧的半径逐帧变化中位 0.33%、p95 0.87%、
    # 最大 1.40%，8% 是近 6 倍余量；超过它只可能是细化射线锁到了背景边缘，
    # 此时**只回退上一帧半径**，帧仍算有效（球心与轨迹是这一帧量到的，只有半径没量到）。
    radius_max_jump: float = 0.08
    # radius_window（滚动中位窗口）：30 帧 ≈ 0.86 s @35 fps —— 比这短的波动（抖动）
    # 交给跳变护栏，比这长的漂移（球真的靠近/远离）必须允许，所以窗口不能再短。
    radius_window: int = 30
    # radius_band（无种子时的半径带）：实测可测帧半径的 IQR 只有 ±2.6%（89.2~93.9 px），
    # 25% 是近 10 倍余量 —— 它挡不住随机抖动（那是跳变护栏的事），只挡"锁到了另一个圆"。
    radius_band: float = 0.25
    # radius_seed_band_lo / hi（给了 ``--seed`` 时的半径带）：锚点换成**种子半径**，
    # 带是 ``seed_r × [1-lo, 1+hi]``。为什么必须锚种子、不能锚滚动中位：中位是**自证**的
    # ——旧 traj.csv 里 277 帧半径全是 11 px（真值 91），中位就是 11，每帧都"在带内"，
    # 于是全过。上界给到 35%（不是 25%）是因为球被甩近时半径会变大（实测 +5%）；
    # 下界 25% 用来挡"锁到小球"（11 px 相对 88 px 是 -87.5%）。
    radius_seed_band_lo: float = 0.25
    radius_seed_band_hi: float = 0.35


@dataclass
class CameraConfig:
    """相机与三维重建参数。

    fx/fy/cx/cy 为内参。没有做棋盘格标定时，可用 apps/tools/calibrate_focal.py
    或者下面的简易方式估算：把已知直径的球放在已知距离处，量出像素半径即可。
    """

    fx: float = 0.0                    # 0 表示未标定 -> 由 hfov_deg 与分辨率估算
    fy: float = 0.0
    cx: float = -1.0                   # -1 表示用图像中心
    cy: float = -1.0
    hfov_deg: float = 60.0             # 水平视场角，用于估算 fx
    # 球直径（米）。⚠️ 这里要填**有效直径**，不是卡尺量到的名义值：
    # 检测器给的像素半径系统性偏小（specular 实测 −13.6%），而单目测深是
    # `Z = f·D/(2r)` —— D 只有与 r 同口径才自洽。名义 25 mm 的球 → 21.6 mm。
    # 依据 `apps/tools/mono_vs_stereo.py --truth-diameter-mm 25`（交接文档 §35.2）；
    # 填名义值会让深度系统性偏大 15.7%（1 m 处 157 mm）。
    ball_diameter_m: float = 0.0216
    # 畸变系数 (k1, k2, p1, p2, k3)。空 = 没有畸变模型（不标定的旧行为）。
    # 由 `config/mono_calibration.json` 经 `calib.mono_calib.apply_to` 填入。
    distortion: tuple[float, ...] = ()
    enable_3d: bool = True


@dataclass
class PredictConfig:
    """轨迹预测参数。"""

    enable: bool = True
    method: str = "quadratic"          # "kalman" | "quadratic"
    horizon_s: float = 0.6             # 预测未来多少秒
    steps: int = 30                    # 预测轨迹画多少个点
    min_points: int = 6                # 至少多少个历史点才开始预测
    use_gravity: bool = False          # 二次拟合时把二次项固定为 g/2（需三维/标定）
    gravity: float = 9.80665
    # 拟合前对历史点做滑动平均的长度（1 = 不平滑，默认）。
    # 注意：滑动平均会把"反弹"这个折点向两侧抹开，让紧挨着折点的
    # 新数据混进旧轨迹的信息，直接毁掉刚反弹那几帧的预测。
    # 检测精度已经是亚像素，本来就不需要平滑；只有检测抖动很大时才调到 3-5。
    smooth_window: int = 1
    # ---- 拟合窗口：只用最近这一小段，绝不拿几秒前的数据去拟合 ----
    # 球一被击打/落地反弹，运动规律就换了，用老数据拟合出来的"未来"一定是错的。
    # 这里的策略是：从最长的窗口开始试，谁的拟合残差够小就用谁，
    # 于是"刚反弹完"会自动退化成短窗口，自由飞行时又自动用足长窗口。
    window_s: float = 0.45             # 最长拟合窗口（秒）
    min_window_s: float = 0.12         # 最短拟合窗口（秒）
    residual_tol: float = 2.5          # 残差容差，单位与坐标一致（2D 是像素，3D 是米）
    min_quadratic_points: int = 6      # 少于这么多点就不用自由二次拟合（见 predict.py 注释）
    # 二维预测时，重力加速度在像素坐标下也是已知量：
    #     g_px = fy * 9.80665 / Z      （Z 由球的像素半径反推）
    # 把二次项固定成 g_px，未知数从 3 个降到 2 个，刚反弹完只有 4-5 个点时
    # 也能给出靠谱的预测。这是单目相机做抛体预测的关键一招。
    use_pixel_gravity: bool = True


@dataclass
class VizConfig:
    trail_enabled: bool = True
    trail_fade: bool = True
    show_circle: bool = True
    show_prediction: bool = True
    show_hud: bool = True
    show_mask: bool = False            # 显示分割掩膜小窗
    show_axes_3d: bool = False         # 显示三维坐标轴/地面网格
    scale: float = 1.0                 # 显示缩放：整张图（含所有叠加）按此倍率放大


@dataclass
class AppConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    predict: PredictConfig = field(default_factory=PredictConfig)
    viz: VizConfig = field(default_factory=VizConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return p


def _merge_dataclass(instance: Any, data: dict[str, Any]) -> Any:
    """把 dict 递归合并进 dataclass 实例（未知字段忽略，避免旧配置报错）。"""
    for key, value in data.items():
        if not hasattr(instance, key):
            continue
        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(current, value)
        elif isinstance(current, ColorConfig) and isinstance(value, dict):
            _merge_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


def load_config(path: str | Path | None = None) -> AppConfig:
    """读取 JSON 配置；文件不存在时返回默认配置。"""
    cfg = AppConfig()
    if path is None:
        return cfg
    p = Path(path)
    if not p.exists():
        return cfg
    data = json.loads(p.read_text(encoding="utf-8"))
    return _merge_dataclass(cfg, data)


def save_config(cfg: AppConfig, path: str | Path) -> Path:
    return cfg.save(path)
