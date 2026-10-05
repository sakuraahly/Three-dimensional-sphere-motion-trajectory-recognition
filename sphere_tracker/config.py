"""集中配置。所有阈值都能存成 JSON，方便标定后复用。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


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
    def preset(name: str) -> "ColorConfig":
        """常用预设。"""
        presets: dict[str, dict[str, Any]] = {
            # 白色 / 浅色球体
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
        if name not in presets:
            raise KeyError(f"未知预设 {name!r}，可选：{sorted(presets)}")
        return ColorConfig(**presets[name])


@dataclass
class DetectorConfig:
    """检测器配置。

    method 取值：
        "color"   HSV 颜色分割 + 轮廓圆度筛选（最快，需要球与背景颜色可分）
        "hough"   霍夫圆变换（不依赖颜色，对光照和虚焦较敏感）
        "motion"  背景建模差分（仅适合摄像头固定不动）
        "fusion"  上面几种投票融合（推荐默认，鲁棒性最好）
        "yolo"    YOLO 目标检测（可选依赖，需 pip install ultralytics）
        "yolo_sam" YOLO 给框 + SAM 精细分割出像素级边缘（可选依赖）
    """

    method: str = "fusion"
    # 融合时启用的子检测器
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

    # ---- 霍夫圆 ----
    hough_dp: float = 1.2
    hough_min_dist: float = 60.0
    hough_param1: float = 120.0        # Canny 高阈值
    hough_param2: float = 30.0         # 圆心累加器阈值，越小越容易检出也越多误检
    hough_blur_ksize: int = 7

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
    min_score: float = 0.30            # 低于此分数不输出

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
    max_coast_frames: int = 12         # 连续丢失多少帧后删除轨迹
    init_hits: int = 2                 # 连续命中多少帧才认为轨迹"确认"
    trail_length: int = 240            # 保存多少个历史点用于画拖尾
    smooth_measurement: bool = True    # 用滤波后的状态作为输出（更平滑）
    # 轨迹点存"原始观测"还是"滤波结果"？
    # 预测器是拿这条轨迹去拟合的，而卡尔曼滤波在机动瞬间会有几十像素的
    # 滞后/超调，这些结构化误差会被拟合放大成"速度误差"。
    # 亚像素检测下，原始观测本身就很干净，所以默认直接用观测值。
    trail_source: str = "measurement"  # "measurement" | "filtered"


@dataclass
class CameraConfig:
    """相机与三维重建参数。

    fx/fy/cx/cy 为内参。没有做棋盘格标定时，可用 apps/calibrate_camera.py
    或者下面的简易方式估算：把已知直径的球放在已知距离处，量出像素半径即可。
    """

    fx: float = 0.0                    # 0 表示未标定 -> 由 hfov_deg 与分辨率估算
    fy: float = 0.0
    cx: float = -1.0                   # -1 表示用图像中心
    cy: float = -1.0
    hfov_deg: float = 60.0             # 水平视场角，用于估算 fx
    ball_diameter_m: float = 0.04      # 球直径（米），40 mm 约等于乒乓球
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
    scale: float = 1.0                 # 显示缩放（小窗口时用）


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
