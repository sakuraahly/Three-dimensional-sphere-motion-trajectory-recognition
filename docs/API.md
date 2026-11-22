# 库 API 参考（`sphere_tracker` / `calib` / `stereo`）
> **这份文档管什么**：三个库包的**模块职责 + 真实签名 + 返回值 + 调用顺序与前置条件**。
> **什么时候看**：要**调用**某个函数/类，想确认"默认值多少、返回什么、能不能直接喂下一层"时。
> 想知道"东西放哪"看 `文件结构.md`、"为什么这么设计、踩过哪些坑"看 `交接文档.md`；**入口文档是 `docs/项目总纲.md`**。
> 签名逐个 `read` 源码后用 `inspect.signature` 复核（清单见 §15）；只写代码里**真的存在**的东西，没确认的一律标"未确认"。
## 0. 全局约定（先读这一节）
| 约定     | 内容                                                                                          |
| -------- | --------------------------------------------------------------------------------------------- |
| 像素坐标 | `(x, y)` = `(列, 行)`，原点左上，OpenCV 口径                                                  |
| 单目三维 | **相机坐标系**：Z 向前、米（`CameraModel.depth_from_radius()` + `backproject()`）             |
| 双目三维 | **校正后左相机系**、米（`T` 也是米）；要和原始左相机系的真值比，先乘 `rig.R1`                 |
| 时间 `t` | 秒。视频/图片源 = `帧号/fps`；实时相机 = `perf_counter()`（绝对值无意义，同进程内单调）       |
| 失败约定 | 绝大多数"没结果"返回 `None`/`[]` **不抛**；只有校验类错误抛异常（各节标注）                   |
| 取帧责任 | **`pipeline` 不取帧**：`sources`（单目）/`stereo.PairSource`（双目）由 app 驱动，再把帧喂进库 |

**范围与前提**：本文只覆盖三个库包（`apps/`、`tests/`、启动器、PPT 工具链不在范围内）；文中"实测数字"引自源码注释或既有文档，我**没有**重跑那 6 个 selftest。
**真机标定跑过，但还没有一次可用的结果**（2026-10-14 更正，原写"从未跑过"）：10-10 那轮 8 对判**不可用且未保存**（双目 RMS 46.40 px、基线 199.67 mm vs 实测 65 mm）；10-13 那轮 25 对里 22 对可用、基线 67.28 mm 只差实测 3.5%，但**两路分辨率错配**（1280x720 vs 1920x1080）所以那份 R/T 不能用。详见 `交接文档.md` §11.3、§12.2、§13.2。
## 0b. `sphere_tracker/plugins.py` —— 实现注册表（**新增一种实现前先读这节**）

一张全局表：`kind → name → 工厂`。检测器 / 预测器 / 视频源 / 标定方法 / 导出格式 / 跟踪器
**全部**从这张表里选，不再有 if 链、也不许在别处手写名字清单。

| 函数       | 签名                                                                     | 行为                                                                                                                                            |
| ---------- | ------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `plugin`   | `plugin(kind, name, *, summary, default=False, needs=(), replace=False)` | 装饰器，登记工厂并**返回原对象**                                                                                                                |
| `register` | 同上（把 `factory` 作为位置参数传入）                                    | 非装饰器写法                                                                                                                                    |
| `resolve`  | `resolve(kind, name)`                                                    | 返回工厂。**找不到抛 `UnknownPlugin`**（`LookupError` 子类），消息里带 kind、请求的名字、全部候选、以及候选的来源模块；**绝不 fallback 到默认** |
| `names`    | `names(kind) -> list[str]`                                               | 注册顺序即展示顺序；GUI 下拉、`argparse choices`、`--help` 都用它                                                                               |
| `default`  | `default(kind) -> str`                                                   | 该 kind 的默认名。零个或多于一个 `default=True` 都抛错（**不许靠 import 顺序决定默认**）                                                        |
| `describe` | `describe(kind) -> list[dict]`                                           | `[{name, summary, needs, is_default}]`，给 GUI tooltip / CLI help 用                                                                            |
| `clear`    | `clear()`                                                                | **仅测试用**，清空全表                                                                                                                          |

**调用方必须遵守的三条**（违反任一条都会被 `tests/selftest_plugins.py` 抓到）：

1. **别猜名字**：选项一律 `names(kind)`，默认一律 `default(kind)`。手写 `choices=("color","hough",…)`
   会让不同入口悄悄不一致（历史上四份检测器清单就是 7/5/4/6 四个数）。
2. **别吞 `UnknownPlugin`**：用户拼错名字时要报错并给候选，不要退化成"那就用默认吧"。
3. **别在 `sphere_tracker/`、`calib/`、`stereo/` 里 import `apps/`**（AST 测试守着）。

**当前六类的实际内容**（2026-10-26；以 `python -c "from sphere_tracker.plugins import *; print(names('detector'))"` 为准）：

| kind           | 成员                                                         | 默认        |
| -------------- | ------------------------------------------------------------ | ----------- |
| `detector`     | color / specular / hough / motion / fusion / yolo / yolo_sam | `color`     |
| `predictor`    | kalman / quadratic                                           | `quadratic` |
| `source`       | camera / file / images                                       | `camera`    |
| `calib_method` | joint / chain                                                | `joint`     |
| `exporter`     | json / yaml / toml                                           | `toml`      |
| `tracker`      | ball                                                         | `ball`      |

**新增一种实现的三步**（这就是插件化要买到的东西）：

```python
# 1. 新建文件，写实现
# 2. 在实现所在模块登记（import 即生效；不要在 plugins.py 里集中登记，否则它会反向依赖具体实现）
@plugin("detector", "my_detector", summary="一句话说明什么场景用它")
def _build(cfg): return MyDetector(cfg)
# 3. 跑 python tests/selftest_plugins.py
```

> ⚠️ **`summary` 是必填的**（空串视为缺失会报错）：它直接当 CLI 帮助与 GUI 提示用，
> 缺了就是"用户看到一个没有说明的选项"。
> ⚠️ **`needs=(...)`** 用来声明可选依赖（如 `ultralytics`）。它的语义是"没装也能出现在列表里，
> 但构造/运行时要给明确提示" —— 不要在 import 阶段就炸，那会让 GUI 列不出这一项。

## 1. `sphere_tracker/config.py` —— 全部可调参数
```python
COLOR_PRESETS: dict[str, dict]      # 预设名 -> HSV 参数；**全项目只此一处**
AUTO_PRESET = "auto"                # 取色策略（不是颜色），不进 COLOR_PRESETS
class ColorConfig:  @staticmethod preset(name: str) -> "ColorConfig"   # "auto" -> mode="auto"
                    @staticmethod preset_names() -> list[str]          # 六种真实颜色（定义顺序）
                    @staticmethod preset_choices() -> list[str]        # ["auto", *preset_names()]
    # mode="fixed"|"auto"；seed_* 见 `纯色球检测方案.md` §9.7
    # white_ball / orange_ball / green_ball / blue_ball / red_ball / yellow_ball
class AppConfig:  detector / tracker / camera / predict / viz
    def to_dict(self) -> dict;  def save(self, path: str | Path) -> Path      # mkdir -p 后写 JSON
def load_config(path: str | Path | None = None) -> AppConfig      # None 或文件不存在 -> 全默认
def save_config(cfg: AppConfig, path: str | Path) -> Path          # == cfg.save(path)
```
| 组               | 关键默认值                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DetectorConfig` | `method="color"`、`fusion_members=["color","hough","motion"]`、`min_radius=8.0`、`max_radius=200.0`、`min_circularity=0.55`、`min_fill_ratio=0.72`、`min_radial_consistency=0.90`、`subpixel_refine=True`、`subpixel_rays=72`、`motion_compensation="auto"`、`camera_motion_threshold_px=0.8`、`motion_work_width=320`、`agree_px=6.0`、`agree_ratio=0.45`、`fusion_topk=3`、`yolo_weights="models/yolo11n.pt"`、`sam_model="models/sam2.1_t.pt"`、`yolo_classes=[32]` |
| `TrackerConfig`  | `order=2`、`process_noise=1.5e2`、`measurement_noise=2.0`、`gate_chi2=9.21`、`maneuver_min_px_s=250.0`、`maneuver_ratio=1.6`、`max_coast_frames=12`、`init_hits=2`、`trail_length=240`、`trail_source="measurement"`                                                                                                                                                                                                                                  |
| `CameraConfig`   | `fx=fy=0.0`（未标定）、`cx=cy=-1.0`（用图像中心）、`hfov_deg=60.0`、`ball_diameter_m=0.04`、`enable_3d=True`                                                                                                                                                                                                                                                                                                                                          |
| `PredictConfig`  | `method="quadratic"`、`horizon_s=0.6`、`steps=30`、`min_points=6`、`use_gravity=False`、`use_pixel_gravity=True`、`window_s=0.45`、`min_window_s=0.12`、`residual_tol=2.5`、`smooth_window=1`                                                                                                                                                                                                                                                         |
| `VizConfig`      | `trail_enabled/show_circle/show_prediction/show_hud=True`、`show_mask/show_axes_3d=False`、`scale=1.0`                                                                                                                                                                                                                                                                                                                                                |

**坑**：① 默认颜色预设是**白色/高亮球**（`[0,0,175]`–`[179,80,255]`），**不是橙球** —— 拍橙球必须 `preset("orange_ball")` 或跑 `apps/tools/calibrate_color.py`，否则"检测不到"会被误判成算法问题。
② `DetectorConfig.min_score`（默认 `0.55`）与 `TrackerConfig.max_q_scale`（默认 `64.0`）**现在是活的** —— 接线位置：`detectors/fusion.py:75` 的孤证门限读 `cfg.min_score`、`KalmanFilter.__init__(..., max_q_scale=cfg.max_q_scale)`（`track.py:40` 用它算 `q_max`）。
**但 `_merge_dataclass` 仍然忽略未知字段**，所以**键名拼错会静默失效**（不报错）—— 配置接线由 `tests/selftest_config_wiring.py` 扫全部字段守着。
③ 预设的**名字清单只有 `COLOR_PRESETS` 一份**（2026-10-27）：「有哪些球色」曾被手抄在 CLI 帮助、HSV 工具的数字键表、GUI 下拉三处，加一种球色要改四处。现在 `preset_names()` 是唯一出口，各入口的 `--preset` 还带 `choices=`（拼错当场报 usage，而不是运行时 `KeyError`）。`tests/selftest_entrypoints.py` 扫 `apps/` 与两个启动脚本，出现"≥3 个预设名的字面量"即判红。
（2026-10-10 之前确实是两个死配置项，行为等价修复后默认值对齐了历史硬编码值 `0.55` / `64.0`，见 `交接文档.md` §10.7-1。）
③ `red_ball` 预设自带 `use_second_range=True`（红色跨 0 度），别只抄 `hsv_lower/upper`。
## 2. `sphere_tracker/types.py` —— 数据容器（字段要记准）
**`Detection`**（单帧检测，像素）：`x, y, radius`（亚像素）；`score`（0–1）；`method`（`"color"`/`"hough"`/`"motion"`/`"fusion:color"`/`"yolo"`/`"yolo+color"`/`"yolo_sam"`）；`parts: dict[str,float]`（各子线索原始分，fusion 会塞 `votes`）；`contour: (N,2)|None`（hough/yolo 不给）；`mask`（**整幅**掩膜，**同帧多个 Detection 共享同一数组**）；`extra: dict`（YOLO 放 `"box"`）。属性 `center -> (x,y)`、`diameter -> 2*radius`。

**`TrackState`**（跟踪器输出）：

| 字段                         | 含义                                                               |
| ---------------------------- | ------------------------------------------------------------------ |
| `track_id, t`                | 轨迹号 / 时刻（秒）                                                |
| `x, y, vx, vy, ax, ay`       | 位置(px)、速度(px/s)、加速度(px/s²)，**全部来自卡尔曼滤波**        |
| `radius`                     | 半径（指数滑动平均，α=0.5 硬编码）                                 |
| `z, vz`                      | 单目测深深度(米)、深度速度(m/s) —— **不是 track.py 填的**（见坑①） |
| `confirmed / coasting`       | 命中 `init_hits` 次后 True / 本帧是外推出来的（没有观测）          |
| `missed / age / hits`        | 连续丢失帧数 / 存活帧数 / 累计命中数                               |
| `maneuvers, last_maneuver_t` | 检测到几次速度突变 / 最近一次突变时刻（`None`=从未）               |

属性 `speed_px -> hypot(vx, vy)`。

| 类型           | 字段                                                                                                                                           |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `FrameContext` | `t=0.0, gray=None, motion=None, camera_moving=False`（`motion` 实际是 `MotionEstimate`，标注为 `Any`）                                         |
| `TrailPoint`   | `t, x, y, z=None, radius=0.0, score=0.0`                                                                                                       |
| `FrameResult`  | `t, frame_index, detection, track, trail, prediction, prediction_3d, position_3d, fps, detect_ms, camera_motion, camera_moving, camera_steady, detector_note` |

`FrameResult` 的三个三维字段不同：`prediction` = **(N,2) 像素**（3D 预测也会投影回像素）、`prediction_3d` = **(N,3) 米**、`position_3d` = **(3,) 米**；`trail` 坐标始终是像素。

**坑**：① **`TrackState.z/vz` 不是 `track.py` 填的**：`Track.state()` 永远给 `z=None, vz=0.0`，它们由 `SpherePipeline.process()` 在三维那一步回填 —— 单独用 `Track` 拿不到 z。
② `TrailPoint.score == 0.0` 是**"这一点是 coast 外推的"标志**，`predict.py` 与 `viz.draw_trail` 都靠它；③ 检测器要用 `getattr(ctx, "gray", None)` 这类防御写法取 `FrameContext` 字段（`ctx` 可能是 `None`），`FrameResult.trail` 是 `list(track.trail)` 浅拷贝、点对象共享。
## 3. `sphere_tracker/geometry.py` —— 圆拟合、亚像素、相机模型
```python
def fit_circle_taubin(points) -> tuple[float, float, float]         # 少于 3 点抛 ValueError
def fit_circle_robust(points, iterations=3, trim_sigma=2.0) -> tuple[float, float, float]
def refine_circle_subpixel(gray, cx, cy, radius, n_rays=72, search=2.5, step=0.25, min_gradient=15.0) -> tuple[float, float, float]
def refine_circle_subpixel_checked(gray, cx, cy, radius, ...) -> tuple[float, float, float, bool]   # 末位多一个"是否真的细化成功"
def contour_circularity(contour) -> float;  def radial_consistency(points, cx, cy, radius) -> float
def circle_radial_gradient_stats(gray, cx, cy, radius, samples=96, min_magnitude=40.0, min_alignment=0.65) -> tuple[float, float]
def circle_edge_support(edges, cx, cy, radius, samples=72) -> float
def estimate_focal_from_known_ball(measured_radius_px, distance_m, ball_diameter_m) -> float    # 半径<=1e-6 抛
class CameraModel:
    def __init__(self, width, height, fx=0.0, fy=0.0, cx=-1.0, cy=-1.0, hfov_deg=60.0, ball_diameter_m=0.04) -> None
    def depth_from_radius(self, radius_px) -> float       # 半径<=1e-6 -> inf
    def backproject(self, u, v, z) -> np.ndarray          # (3,) 米
    def project(self, point3d) -> tuple[float, float]     # 属性：width/height/fx/fy/cx/cy/calibrated/ball_diameter_m
```
**调用顺序**：`CameraModel` 要先有帧尺寸才能造（`width/height` 必填）⇒ `SpherePipeline` 是拿到**第一帧后**才建它；典型链路 `detect` →（color 内部）`refine_circle_subpixel` → `depth_from_radius`（Z）→ `backproject`（三维）。

**坑**：① **`refine_circle_subpixel` 失败时不抛异常，而是原样返回输入**（`radius<3.0`、采样起点越界、有效射线 `< max(8, n_rays//3)`）；要判"到底细化成功没有"用 `refine_circle_subpixel_checked` 的末位布尔（**别去比浮点相等**），`sphere_tracker/detectors/specular.py::_refine_center()` 是它的调用方；`fit_circle_robust` 点数 `<5` 时**退回** `fit_circle_taubin`。
② `fx<=0` 用 `hfov_deg` 估；**`fy<=0` 直接令 `fy=fx`**（不按纵横比另算）；`cx/cy<0` 用图像中心；`calibrated` 只看 `fx`。`depth_from_radius` 用 `f=(fx+fy)/2`，单位米，依赖 `ball_diameter_m` 正确。
③ `circle_gradient_support()` **全仓库无调用者**（hough 用的是 `circle_radial_gradient_stats`），改它不影响 hough；旧文档里的 `PinholeModel.to_3d()/radius_from_depth()` **已删除**（`交接文档.md` §10.2-C），类名现在是 `CameraModel`，pipeline 只用 `depth_from_radius/backproject/project`，`estimate_focal_from_known_ball` 只有 `apps/tools/calibrate_focal.py` 在用。
## 4. `sphere_tracker/detectors/` —— 检测器
```python
def resolve_weights(weights: str) -> str        # 相对路径按【项目根】解析；找不到只回文件名给 ultralytics
class BaseDetector:
    def __init__(self, cfg: DetectorConfig) -> None
    def detect(self, frame, ctx=None) -> list[Detection]       # 基类抛 NotImplementedError
    def reset(self) -> None                                    # 换视频源时调用
    def _score_candidate(self, radius, circularity=0.0, fill_ratio=0.0, radial=0.0, edge_support=0.0, weights=(0.30, 0.28, 0.27, 0.15)) -> float
def build_single_detector(name: str, cfg) -> BaseDetector | None;  def build_detector(cfg: DetectorConfig) -> BaseDetector
    # 前者只认 color/hough/motion/specular（其它返回 None）；后者未知名字抛 ValueError，"sam"/"sam2" 等价 "yolo_sam"
class ColorBlobDetector(BaseDetector):  # "color"；def build_mask(frame) -> np.ndarray；给 contour+mask
    # def seed(x,y,r,frame_index=0)：从点击环取 HSV、半径带 [0.6,1.7]×r；
    #   种子帧无候选时输出 seed_anchor，之后可用亚像素边缘跟随（extra 里可分辨）
class HoughDetector(BaseDetector):  # "hough"；无 contour / 无 mask
class MotionDetector(BaseDetector):  # "motion"；last_mode: "bg" | "compensated"
class SpecularBallDetector(BaseDetector):  # "specular"；必须 seed(x,y,r,frame_index=0)；CSRT 返回 0/1
    # 输出半径 = seed_r × (当前框半径 / 初始框半径)，夹在 [seed_scale_lo, hi]；
    # 不是 max(w,h)/2（那是"框多大"，不是"球多大"）
class FusionDetector(BaseDetector):  # "fusion"；返回 0 或 1 个，method="fusion:<子名>"
    # 每个成员交 top-K（fusion_topk）候选 → 按中心聚类 → 按不同成员数投票；
    # seed() 透传给支持种子的成员（color / specular）
class YoloDetector / YoloSamDetector  # "yolo" / "yolo_sam"；def refine_circle_in_box(frame, det, cfg) -> Detection | None
```
`_score_candidate` 的硬规则（所有检测器共用）：半径越界、`radial>0 且 <min_radial_consistency`、`fill>0 且 <min_fill_ratio` → **直接 0**；半径贴边（`<1.3*min` 或 `>0.8*max`）整分 ×0.8；只对"传进来 >0"的指标做加权平均。区分圆与方块主要靠 `radial`（方块顶点比边中点远 41%）。

| 检测器   | 需要 `ctx` 的            | 返回 contour/mask | 额外前置                                                                                                                                                                                                                                                        |
| -------- | ------------------------ | ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| color    | `gray`                   | 有 / 有（整幅）   | 只对**前 3 个**候选做亚像素细化                                                                                                                                                                                                                                 |
| hough    | `gray`                   | 无 / 无           | 两道梯度门：`hough_min_support`、`hough_min_sign_consistency`                                                                                                                                                                                                   |
| motion   | `motion`+`camera_moving` | 有 / 有（`fg`）   | 需 `_prev_frame` 与 `motion.matrix` 才走补偿分支                                                                                                                                                                                                                |
| specular | 不用（内部转灰度）       | 无 / 无           | **必须人工 seed**；CSRT 跑 BGR；丢失返回 `[]`，不 snap 背景；圆心在出口换成**轮廓圆心**（`cfg.specular.refine_center`），出处见 `det.extra["center_source"]`（`contour`/`box`）与 `parts["center_shift_px"]`；**半径仍取框**（`B-130`：镜面球半径不能从边缘量） |
| fusion   | 透传                     | 取胜出者          | 子检测器由 `cfg.fusion_members` 决定                                                                                                                                                                                                                            |
| neural   | 不用                     | SAM 有 / YOLO 无  | 需 `ultralytics`；缺依赖时**抛 RuntimeError**                                                                                                                                                                                                                   |

**坑**：① `detect()` 返回**按 score 降序**，`pipeline` 只取 `[0]` 当"本帧检测"、整表交给 tracker —— 排序是接口契约。
② `build_single_detector` 只认三个名字：把 `"yolo"` 写进 `fusion_members` 会被**静默忽略**（不报错，少一路票）。
③ fusion：每票 +0.12；**只有一票且总分 <0.55 返回空**（硬编码，见 §1 坑②）；docstring 里的 `vote_single_min` **并不存在**；子检测器抛 `cv2.error` 被吞掉当"没检测到"。
④ motion：**换源必须 `reset()`**（重建 MOG2）；`_frames<5` 恒返回 `[]`；从"相机在动"回到"静止"时会用 `learningRate=1.0` 连喂 3 帧重建背景（这 3 帧的检测不可信）。
⑤ AI 后端的失败是**运行时**的：`build_detector` 不报错，`detect()` 抛 `RuntimeError`，`SpherePipeline` 捕获它、只打印一次 `[检测器不可用]`，之后按"没检测到"继续跑（**不中断视频**）；SAM 只处理前 3 个种子框。旧 sphere_tracker/detect.py 垫片已于 2026-10-21 删除，一律从 `sphere_tracker.detectors` 导入。
## 5. `sphere_tracker/camera_motion.py` —— 相机自运动
```python
@dataclass MotionEstimate:
    dx=0.0; dy=0.0                  # 全分辨率像素位移
    rotation_deg=0.0; scale=1.0; response=0.0; inliers=0    # response = 内点比例 / 相位相关响应
    method="none"                   # "features" | "grid" | "phase" | "none"
    matrix: np.ndarray | None = None    # 2x3，把【上一帧】对齐到【当前帧】；magnitude=hypot(dx,dy)、has_transform
class GlobalMotionEstimator:
    def __init__(self, work_width=320, threshold_px=1.2, feature_fallback=True, max_features=250, min_inliers=20, hysteresis=0.5) -> None
    def reset(self) -> None
    def estimate(self, prev_gray: np.ndarray | None, gray: np.ndarray) -> MotionEstimate
    def is_moving(self, motion: MotionEstimate | None) -> bool
def warp_to_current(prev_frame, motion: MotionEstimate | None) -> np.ndarray
```
**调用顺序 / 前置条件**：**第一次 `estimate()` 必然返回 `method="none"`**（内部没有上一帧可比），第二帧起才可能出结果；`reset()` 之后同样要重新等一帧 —— 所以"第一帧相机静止"永远不会在第一帧成立。内部三级降级：`features`（角点 + RANSAC 相似变换）→ `grid`（网格点 + LK + 中位数）→ `phase`（相位相关，仅当 `feature_fallback=True`）；**`"grid"` 是合法 `method` 值**（docstring 只写了另外三个）。`is_moving()` 有迟滞且**把状态写回 `self._moving`**：进门限 `threshold_px`、退出门限 `threshold_px*hysteresis`，`method=="phase"` 时两个门限 ×2，`method=="none"` 时**沿用上一帧状态**（不当成静止）。`SpherePipeline` 传的是 `DetectorConfig.camera_motion_threshold_px`（**0.8**），不是类默认的 1.2。

**坑**：① **`prev_gray` 参数只用来判 `None`**：真正的"上一帧"是实例内部的 `_prev_small`（每次调用时更新）—— 把当前帧误传进去不会报错，只是让"第一帧"判断失效。
② `_estimate_features` 换回全分辨率必须是 `M_full = Si @ M_small @ S`；写成 `S @ M_small @ Si` 时旋转/缩放项碰巧不变、**只有平移项差 `s²`**（默认 320 宽配 1920 画面 = 36 倍），后果是 `is_moving()` 几乎永不判"在动"（`交接文档.md` §10.3-#4）。
③ `warp_to_current` 在 `motion is None` 或 `matrix is None` 时**原样返回 `prev_frame`**（不抛）。
## 6. `sphere_tracker/track.py` —— 卡尔曼跟踪
```python
class KalmanFilter:
    def __init__(self, order=2, process_noise=1.5e2, measurement_noise=2.0) -> None   # dim=4(order1)/6(order2)
    def initialize(self, z, velocity=None) -> None;  def predict(self, dt: float) -> np.ndarray   # 返回状态副本
    def innovation(self, z) -> tuple[np.ndarray, np.ndarray];  def mahalanobis(self, z) -> float   # 奇异时 inf
    def update(self, z) -> float            # 马氏距离²；未初始化时先初始化并返回 0.0
    def adapt_noise(self, d2: float, gate: float) -> None;  def reinitialize(self, z, v) -> None
    # 属性：position(2,) / velocity(2,) / acceleration(2,；order=1 时全 0)
class Track:
    def __init__(self, track_id: int, cfg: TrackerConfig, t: float, det: Detection) -> None
    def predict_to(self, t: float) -> None;  def update(self, det, t) -> None;  def coast(self, t) -> None
    @property dead -> bool                  # misses > max_coast_frames（严格大于）
    def state(self) -> TrackState
class BallTracker:
    def __init__(self, cfg: TrackerConfig) -> None;  def reset(self) -> None
    def update(self, detections: list[Detection], t: float) -> Track | None     # 返回 primary
    @property primary -> Track | None
```
**调用顺序 / 前置条件**：`update()` 第一步对**所有**轨迹 `predict_to(t)` ⇒ `t` 必须**单调不减**（否则 `dt` 被 `max(dt,1e-6)` 夹住，速度/加速度失真）。关联分两轮：紧关联（半径一致 + 卡方门控）→ **抢救关联**（距离 `<= max(rescue_min_px, rescue_radius_ratio*det.radius)`，命中后把 `kf.q` 顶到 `q_max`）；两轮都没配上的检测**立刻新建轨迹**（无 N-of-M 延迟）。半径门控与卡方门控**只在 `hits >= init_hits` 时生效**。

**坑**：① `KalmanFilter.update()` 未初始化时**代替** `initialize()` 并返回 `0.0`（不抛）；`predict(dt<=0)` 或未初始化时原样返回，不推进时间。
② `q_max = q_base*64.0` **硬编码**（`TrackerConfig.max_q_scale` 不生效）；半径 EMA 的 `alpha=0.5` 也硬编码，半径滞后**直接**变成单目测深误差（`Z ∝ 1/r`）。
③ `Track.state()` **不填 `z/vz`**；`age` 由 `BallTracker.update` 递增。
④ 每次 `update()` 结束会删掉 `dead` 的轨迹；`primary` 排序键是 `(confirmed, not coasting, hits, score, radius)`。
## 7. `sphere_tracker/predict.py` —— 轨迹预测
```python
def predict_quadratic(trail: list[TrailPoint], cfg: PredictConfig, dims=2, gravity_axis=1, residual_tol=None, t_now=None, gravity_value=None, t_break=None) -> np.ndarray | None
def predict_kalman(state: TrackState | None, cfg: PredictConfig, dims=2) -> np.ndarray | None
class TrajectoryPredictor:
    def __init__(self, cfg: PredictConfig) -> None
    def predict(self, trail, state, dims=2, residual_tol=None, t_now=None, gravity_value=None, t_break=None) -> np.ndarray | None
```
返回 `(cfg.steps, dims)`，**第一个点就是当前时刻（t+0）**，不是"未来第一个采样"。**要点**：至少 **3 个点**否则 `None`；时间轴以 `t_now` 为基准（缺省用最后一点的时间），传 `t_now=state.t` 才是"从当前帧出发"；`t_break`（一般 `state.last_maneuver_t`）只是**提示**（抬高窗口下界），真正定位分界点的是"最长有效后缀搜索"（逐段丢最老的点，取第一个残差合格的）；已知重力时二次项固定为 `g/2`、每轴未知数 3→2、`min_pts` 可降到 2，`dims=2` 时重力只作用在 `gravity_axis`（默认 1 = y）；`dims=3` 时 **Z 轴强制线性**（给 Z 做二次外推会算出负深度）；2D 时若 `score>0` 的点 ≥4 个，只用这些"真观测"拟合（coast 出来的点丢掉）。

**坑**：① `method="kalman"` 时 `TrajectoryPredictor.predict()` **完全忽略** `trail/gravity_value/t_now/t_break`，直接用 `TrackState` 匀加速外推（`dims=3` 且 `state.z is None` → `None`）；`cfg.enable=False` **直接返回 `None`**（不看 `trail`）。
② `_sanitize`：NaN/inf → `None`；3D 遇到 `z<=0.05` 会**截断**，前 2 个点就无效则整体 `None`。
③ `residual_tol=None` 时用 `cfg.residual_tol`，**2D 单位是像素、3D 是米** —— pipeline 会按 `tol_px*Z/f` 换算成米再传进来，手调容易搞错量纲。
## 8. `sphere_tracker/sources.py` —— 单目帧源
```python
@dataclass SourceInfo:  kind: str ("camera"|"video"|"image");  width: int;  height: int;  fps: float
    backend: str = ""      # "DSHOW"/"MSMF"/"file"（摄像头是索引字符串）      path: str = ""   # 文件路径
class VideoSource:  info: SourceInfo
    def read(self) -> tuple[bool, np.ndarray | None, float]      # (ok, frame, t)
    def release(self) -> None;  @property is_live -> bool        # 基类 False；CameraSource 覆写 True
class LatestFrameCamera:               # 后台线程采集，永远取"最新帧"
    def __init__(self, source: CameraSource) -> None
    def latest(self, last_seq: int = -1) -> tuple[int, np.ndarray | None, float]
    #                                          ^seq  ^frame              ^t_capture
    def wait_first(self, timeout: float = 10.0) -> np.ndarray | None
    def stop(self) -> None             # join(1.5) 并释放底层 source；属性 info / dropped
class CameraSource(VideoSource):
    def __init__(self, index=0, width=1280, height=720, fps=60, fourcc="MJPG", backend=None, optimize=True, target_fps=25.0) -> None
    def measure_fps(self, duration=1.0, warmup=3) -> float;  def optimize_mode(self, target_fps=25.0, budget=0.25) -> None
    def read(self) -> tuple[bool, np.ndarray | None, float]
    # 属性：backend_name / resolution_fallback / mode_probed / fps_measured / info
class FileSource(VideoSource):   def __init__(self, path: str | Path) -> None
class ImageSource(VideoSource):  def __init__(self, paths: Sequence[str | Path], fps: float = 30.0) -> None
def backend_name(api: int | None) -> str;  def open_source(path=None, camera=None, **kwargs) -> VideoSource
def open_camera_source(*, index=None, identity="", port="", symbolic_link="",
                       width=1280, height=720, fps=30, backend=None,
                       optimize=False, capture_api="auto") -> VideoSource
CANDIDATE_MODES: tuple[tuple[int, int], ...]     # 自动探测用的候选分辨率
```
**调用顺序 / 前置条件**：app 里的标准用法是 `src = open_source(...)` → 循环 `ok, frame, t = src.read()` → `pipe.process(frame, t)` → 结束 `src.release()`；**`pipeline` 不 import `sources`**（取帧与编排分层）。`open_source` 分派：`path` 是目录 → `ImageSource`（按扩展名过滤 + 文件名排序）；`path` 是图片扩展名 → `ImageSource([单张])`；否则 → `FileSource`；都不给 → `CameraSource(index=camera or 0)`。要"永远处理最新画面"就用 `LatestFrameCamera(CameraSource(...))`（双目分支就是这么用的），循环里用 `latest()`。

**坑**：① **`latest(last_seq=-1)` 的 `last_seq` 没被使用**：直接返回当前 `(seq, frame, t_capture)`，"有没有新帧"要自己比 `seq`。
①b **第三个值是"这一帧被采到的时刻"，不是"你来取的时刻"**（2026-10-27 修）：它在采集线程里 `read()` 返回时就记下，同一帧取两次返回**同一个**值。用"处理时刻"当时间轴会让轴随负载漂移 —— 项目自己量过：滞后 1 帧（17 ms）→ 三维误差 440 mm。`FramePump` 因此要求源**必须**回三个值，只回两个会抛 `TypeError`（不许静默退化成年龄 0）。
② `CameraSource.__init__` **很重**：依次试后端（DSHOW→MSMF→ANY）与分辨率（用户值→1280x720→640x480→默认），每个组合连读至多 12 帧做"全黑常量流"检查（判据 `frame.std() > 1.0`）；`optimize=True` 还会跑一次 `optimize_mode()`（最坏 10–40 秒，只跑一次，缓存到 `config/camera_modes.json`）。全失败抛 `RuntimeError`（消息里带"占用/红外/遮挡/USB 带宽"四个可能原因）。
③ `info.fps` **尽量是实测值**：`CAP_PROP_FPS` 常返回"你请求的值"而不是真实帧率（实测请求 60、实际 29.8）；`fps_measured` 告诉你这个值是不是实测 —— 写录像帧率要用它；`resolution_fallback=True` = 请求的分辨率不支持、已降级（USB 摄像头常表现为"打开成功但每帧全黑"）。
④ 时间戳语义不同：`CameraSource` 用 `perf_counter()`，`FileSource`/`ImageSource` 用 `帧号/fps`，跨源不可比。
⑤ `open_source(path=...)` 时 `camera=` 被忽略；`**kwargs` 只在 camera/image 分支生效，走 `FileSource` 时**静默丢弃**；读尽后返回 `(False, None, t)`（`ImageSource` 读不出图片时同样 `ok=False`）。

## 8b. `sphere_tracker/mf_camera.py` —— Media Foundation 按设备 symlink 采集

```python
@dataclass(frozen=True) MFDevice:  index; name; symbolic_link; identity; port; serial
def list_devices() -> list[MFDevice]                     # 只枚举，不开相机
def find_device(*, identity="", port="", symbolic_link="", devices=None) -> MFDevice
@dataclass(frozen=True) MediaType: subtype; width; height; fps_num; fps_den
def select_native_type(types, width, height, fps=0) -> MediaType
def decode_frame(raw, subtype, width, height) -> np.ndarray   # NV12/MJPG/YUY2/RGB32 -> BGR
class MFCameraSource(VideoSource):
    def __init__(self, *, identity="", port="", symbolic_link="",
                 width=1920, height=1080, fps=30.0) -> None   # 首次 read 才打开
```

**口径**：`symbolic_link` 来自 `MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK`，
含 PnP 端口；`find_device` 只按 symlink/端口/identity 精确匹配，**不按 FriendlyName
猜**。`MFCameraSource` 枚举 native media type，分辨率优先、fps 其次、subtype
优先 MJPG；请求档不存在时取最近档并置 `resolution_fallback=True`。它**只**是
新增后端：`open_camera_source(capture_api="opencv")` 仍是原 `CameraSource`，
默认也不启用 MF。

**坑**：① MF 的 `SetCurrentMediaType` 给一个“部分类型”可能被接受却落到
640×480；必须用 native type 选档，不能只看请求宽高。② `ReadSample` 的
buffer 可能是 NV12/MJPG/YUY2，按当前 subtype 解码，未知 subtype 抛
`MFError`。③ 两台同型号相机若没有唯一序列号，symlink 仍主要靠 USB 端口；
换口后要重新绑定，不自动按名字找回。

## 8c. `camera_autobind.py` / `camera_fingerprint.py` —— 自动绑定

```python
@dataclass(frozen=True) RoleCandidate: device; parent; serial
def external_candidates(devices, *, settings=None) -> list[DeviceEntry]
def choose_roles(candidates, *, left_serial="", right_serial="") -> tuple[RoleCandidate, RoleCandidate]
# 一台 serial -> 它是 left，另一台 right；台数不是 2 / 两台无 serial 且无锚点 -> AutobindError

@dataclass(frozen=True) Fingerprint: gray; hist
def make_fingerprint(frame) -> Fingerprint          # 64x36 灰度归一化 + HSV 16x8
def similarity(a, b) -> float                       # 0.7 灰度 + 0.3 直方图
def match_fingerprint(fp, refs, *, min_score=0.45, min_margin=0.10) -> (role, score) | None
```

**口径**：`cameras.py autobind` 用 `pnputil ... /properties` 读
`DEVPKEY_Device_Parent`，`serial_from_parent()` 只认不像端口路径的末段；
right 用“唯一另一台外接同型号”排除。OpenCV 索引由 MF 参考帧 + 画面指纹
自动映射；分数不够/两路同索引就**不写**。`RoleSpec.serial` 持久化锚点。

## 9. `sphere_tracker/camera_registry.py` —— 相机设备管理
把"相机身份"从会漂移的 cv2 索引里解耦；配置在 `config/cameras.json`（不入库）。
```python
@dataclass(frozen=True) DeviceEntry:  order: int     # 我们【当作】cv2 索引的枚举位次（启发式）
    name=""; identity=""; port=""; present=True; source="index"     # source: system/registry/index
@dataclass RoleSpec:  identity=""; port=""; name=""; index: int | None = None; verified=False
@dataclass Registry:  v=1; roles: dict[str, RoleSpec]; ignored: list[RoleSpec]; settings: dict
@dataclass(frozen=True) ProbeResult:  index; ok; width=0; height=0; fps=0.0; std=0.0; error=""; entry=None
@dataclass(frozen=True) ResolvedCamera:  role; index; how; entry=None; warnings=()
    # how: "identity" | "port" | "name" | "manual" | "index-fallback"
def current_platform() -> str;  def parse_windows_instance(instance: str) -> tuple[str, str]   # (identity, 端口段)
def read_windows_registry_instances() -> list[str];  def read_windows_pnputil() -> list[dict]
def read_windows_pnp(*, runner=_run_powershell) -> list[dict];  def read_windows_devices_info() -> list[dict]   # pnputil 优先
def windows_devices(*, registry_reader=..., pnp_reader=...) -> list[DeviceEntry];  def list_devices() -> list[DeviceEntry]   # 非 Windows -> []
def probe_index(index, *, width=1280, height=720, fps=30, opener=None) -> ProbeResult   # 异常都收成 ok=False
def probe_mode(index, width, height, *, fps=0, backend=None, sample_s=0.6,
               first_frame_timeout_s=30, opener=None) -> ModeResult   # 只试这一档，真帧尺寸+数帧实测
def probe_modes(index, modes=None, *, fps_list=(0.0,), backend=None, ...) -> list[ModeResult]
def is_avoided(name: str, settings: dict | None = None) -> bool;  def rank_candidates(devices, *, reg=None, settings=None) -> list[DeviceEntry]
def select_cameras(*, need=1, devices=None, probe=probe_index, reg=None, settings=None, indices: list[int] | None = None) -> list[ProbeResult]
def load_registry(path=None) -> Registry;  def save_registry(reg, path=None) -> Path    # 坏文件 -> 空 Registry（不抛）；写不进去抛
def set_role(reg, role, *, entry=None, index=None, verified=False) -> RoleSpec;  def format_device_table(devices, probes=None, settings=None, reg=None) -> str
def resolve_role(role, reg=None, devices=None, *, interactive=True, probe=None) -> ResolvedCamera
def resolve_camera_arg(value, *, reg=None, devices=None, interactive=True) -> int | None
# ---- 「无关」清单（2026-10 新增）----
def is_device_ignored(d: DeviceEntry, reg: Registry | None = None) -> bool
def find_ignored_device(spec: RoleSpec, devices: list[DeviceEntry]) -> DeviceEntry | None   # 严格匹配，不按名字
def ignored_orders(reg: Registry, devices: list[DeviceEntry]) -> set[int]
def role_of_device(reg: Registry, d: DeviceEntry) -> str | None
def mark_ignored(reg, *, entry=None, index=None, verified=False) -> RoleSpec   # 顺手把它从 roles 里摘掉
def unmark_ignored(reg, *, entry=None, index=None) -> int      # 返回摘掉几条
def clear_ignored(reg: Registry) -> int
AUTO_TOKEN = "auto";  DEFAULT_SETTINGS = {...}   # prefer_external / min_width / min_height / min_fps / min_pair_fps / avoid_names / avoid_name_patterns / max_index
# min_fps 只在 ProbeResult.fps_measured=True 时参与 select_cameras 过滤；
# min_pair_fps=None 沿用 AUTO_MIN_PAIR_FPS（14.4），可写绝对下限。
def merged_settings(settings=None, reg=None) -> dict;  def setting_int/setting_float(...)   # settings 的唯一合成/容错读取点
```
`resolve_camera_arg` 规则：`None`/`""` → `None`；`int` → 原样；`"0"` → `0`；`"port:xx"`/`"id:xx"`/`"name:xx"` → 匹配设备；**其它字符串当角色名**（查 `config/cameras.json`）。

**`is_avoided(name, settings)` 的两条规则都来自配置**（2026-10-18 改）：
`avoid_names` 走**子串**、`avoid_name_patterns` 走**正则**，两者是"或"关系。
原来 `infrared` / `\bir\b` 是**写死在函数里**的，用户换一台名字带 `IR` 的正常相机
就会被永久误伤且只能改源码；现在写 `"avoid_name_patterns": []` 即可整体关掉。
**非法正则**只跳过那一条并警告，**不抛异常**（配置写错不该让设备管理跑不起来）。

**`Registry.ignored` = 「无关」清单**：用户明确标记"这台不要用"（笔记本内置 / 红外）。
与 `roles` 同样记**身份**而非编号，所以重插 USB 后仍有效。语义：
`rank_candidates()` 给它 `-1000` 分（比"内置/红外"的 `-200` 还重），`select_cameras()`
把它排到最后、只在其它相机都不够用时才动用**并打印警告**；`mark_ignored()` 会把它
从 `roles` 里摘掉（一台相机不能既是左相机又是「无关」）。
`format_device_table(..., reg=reg)` 会打出 `[无关]` 标记。

**为什么 `_spec_matches_device()` 不按名字匹配**（与 `_match_entry()` 不同）：
**当初**的理由是"两台同型号相机名字一模一样"（本项目两台 aoni 都叫
`aoni webcam A25`），按名字匹配会把"只排除这一台"变成"两台一起排除"。

> ⚠️ **这个前提已经变了（2026-10-18 实测）。** 用户给两台相机改了名，现在系统里是：
>
> | 名字 | 端口 |
> | ---- | ---- |
> | `aoni webcam A25 left` | `6&31324e2c&0&0000` |
> | `aoni webcam A25 right` | `6&4489382&0&0000` |
>
> **名字唯一且带左右标识**，所以"按名字会误伤两台"这个危险**当前不成立**。
> 但代码**仍保持** identity → port 的顺序（拿不到身份才退编号），理由变成
> **保守而非必需**：名字可由用户随时改，而 `identity`/`port` 不会；
> 且改成按名字匹配需要重新评估"用户改名后配置静默失效"这个场景。
> **仍保持保守是已定的决策**（2026-10-18，见 `docs/交接文档.md` §10.12）：
> 名字唯一了也**不**改成按名字匹配 —— 自动兜底省的是几十秒的 `identify`，
> 赌的是"静默用错相机"（左右一换，标定的 R/T 全废且画面看不出异常）。
> **代价不对称，宁可报错让人确认。**

`_match_entry()` 保留"identity → port → name"是给**角色解析**用的（最终只取一台，且前两级优先），
两者刻意不同。

**坑**：① **`DeviceEntry.order` 是启发式**：OpenCV 不提供"索引 N 是哪个物理设备"的查询接口，这里把枚举位次**当作** cv2 索引 —— 所以有 `cameras.py identify`（目视确认后写死身份）与 `check`（查漂移）兜底；非 Windows 时 `list_devices()` 返回 `[]`，整套功能**退化为"按索引"且不报错**。
② **`verified=True` 的角色不允许索引兜底**：身份/端口/名字都对不上时直接抛 `RuntimeError`（或交互让用户选），绝不用配置里的旧索引 —— 这正是"左边外接、右边却成了内置"的成因。
③ `resolve_role(..., probe=...)` 这个参数**在函数体里没被使用**；`resolve_camera_arg("auto")` **抛 `RuntimeError`**（`auto` 只能用 `--auto-cameras` 表达）。
④ `select_cameras()` 挑不满**不抛**，返回实际挑到的几台（调用方自己判 `len`）。
⑤ `probe_index()` 会**真开一次相机**；`_default_opener` 里的 `from .sources import CameraSource` 是**惰性 import**（`camera_registry.py:397`），不探测就不需要开设备。设备清单默认走 `pnputil`（本机实测 ~0.1 s），`Get-PnpDevice` 兜底要 ~10 s。
⑥ **显式编号会绕过「无关」**：`--camera 3` 直接返回 `3`（`resolve_camera_arg` 对纯数字原样返回），
只有 `--auto-cameras` / `--camera auto`（走 `select_cameras`）才跳过「无关」。这是有意的 ——
用户显式点名一台相机时，优先级最高。
⑦ 「无关」清单若在 JSON 里被手改成"同一台既是角色又是无关"，`resolve_role()`
会带一条 `warnings` 提示（不会静默）；`cameras.py check` 把它算作 `[危险]`、退出码 2。
## 10. `sphere_tracker/viz.py` —— 绘制
```python
def draw_trail(frame, trail: list[TrailPoint], fade=True, thickness=2, base_color=(0, 255, 255)) -> None
def draw_detection(frame, det: Detection, color=(0, 255, 0), label=None, contour=True) -> None
def draw_prediction(frame, points, color=(0, 140, 255), dash=6, thickness=2, label=None, dots=10) -> None
def draw_hud(frame, lines: list[str], origin=(10, 10)) -> None;  def paste_mask_preview(frame, mask, width=200) -> None
def draw_3d_panel(size=(300, 220), history=None, prediction=None, title="3D (Z = depth from ball size)") -> np.ndarray   # size=(宽, 高)
def hud_lines(result, cfg) -> list[str]      # result/cfg 未标注类型；期望 FrameResult / AppConfig
```
**坑**：① **除 `draw_3d_panel` 外全部原地改 `frame` 并返回 `None`**（`draw_3d_panel` 相反，返回新面板）—— 别写成 `frame = draw_trail(...)`。
② `cv2.putText` 只能画 ASCII，**中文会变成问号** —— 所以 `hud_lines` 全是英文；要中文界面得改用 PIL。
③ `draw_trail` 用 `TrailPoint.score == 0.0` 判定"外推段"，画成灰色细线。
④ ~~`SpherePipeline.render()` 在 `viz.scale != 1` 时只把预测点乘了 `scale`~~ —— **已修，本条作废**：现在每个 `draw_*` 自己接 `scale`，"画布缩放"只有一个负责处（`pipeline.py:294-336`；预测线不再由调用方预乘）。
`viz.scale = k` 的语义是「**整张原始图的 k 倍**」——几何与文字**全部**按 k 缩放，等式 `render(k) ≈ resize(render(1), ×k)`（像素级不成立：抗锯齿按各自分辨率算，实测内容像素差 86%，那是正常行为）。
判据是**尺度不变性**（每个元素的外接框恰为 k 倍），由 `tests/selftest_viz_scale.py` 守住（5 个变异全红）。
## 11. `sphere_tracker/pipeline.py` —— 单帧总编排
```python
class SpherePipeline:
    def __init__(self, cfg: AppConfig | None = None) -> None
    def reset(self) -> None
    def process(self, frame: np.ndarray, t: float | None = None) -> FrameResult
    def render(self, frame, result: FrameResult) -> np.ndarray            # 返回新图，不改输入
    def process_and_render(self, frame, t=None) -> tuple[np.ndarray, FrameResult]
    # 属性：cfg / detector / tracker / predictor / camera / history3d / motion_estimator
```
`process()` 的固定顺序（要插东西就照这个插）：时间与 FPS → `_ensure_camera`（第一帧）→ 灰度 + 相机自运动 → `FrameContext` → `detector.detect`（`RuntimeError` 被吞，只打印一次）→ `tracker.update` → `FrameResult` → 单目三维（`enable_3d` + `state` + `best.radius>1e-3` + `0.01<z<100`）→ trail 快照 → 预测（先 3D，再 2D）。

**前置条件**：① `__init__` **不造 `CameraModel`**；`_ensure_camera(frame)` 用**第一帧的 `shape`** 建，之后固定 ⇒ 中途改分辨率/换源**不会重建内参**（要自己重建 pipeline）。
② `t` 必须**单调递增**（FPS 的 EMA 与 tracker 的 `dt` 都依赖它）；`t=None` 时内部用 `perf_counter()` 自建基准；`CameraConfig.fx` 必须与**实际帧尺寸**配套（库不知道你从哪个源取帧）。
③ 三维那一步用**原始观测**（`best.x/y/radius`）而不是滤波结果：半径直接决定深度，用滤波半径会引入滞后。
④ 预测：`use_gravity and len(history3d) >= min_points` 才走 3D，并把残差容差按 `tol_m = residual_tol*z/f` 换算成米，否则退回 2D；2D 的像素重力 `g_px = fy*g/z` 只在**相机静止或匀速**（`not camera_moving or camera_steady`）时启用 —— 相机在动时表观加速度 = 球重力 + 相机加速度，没法分离；`camera_steady` 需要至少 3 帧位移历史（`_motion_recent` 是 `maxlen=6`），判据是极差 `<= max(1.0, 0.5*最大)`。

**坑**：① `reset()` 清 tracker/`history3d`/detector/motion_estimator/`_prev_gray`/`_frame_index`/`_prev_t`，**但不清 `_fps_ema`、`_t0`、`_detector_error`**（换源后 FPS 会带旧值平滑，报错也不会再打印）。
② `history3d` 只在三维成立时追加，`state.z/vz` 也在那里回填。
③ 检测器的 `RuntimeError`（AI 后端未装）只打印**一次**，之后静默按"没检测到"跑。
## 11b. `sphere_tracker/multi_target.py` —— 二维多目标关联（未接主流程）
```python
@dataclass TrackConfig:  gate_px=40.0; max_missing=15; predict_kind="const_velocity"
                         reid_radius_px=120.0; reid_after_missing=10; min_hits=3
@dataclass Track:  tid; positions: list[(t,x,y)|None]; state; hits; missing;
                   velocity(px/s); predicted; reid_events; swap_events
                   .last_observation -> (t,x,y)|None;  .length -> 真实观测点数
class MultiTracker:
    def __init__(self, cfg: TrackConfig | None = None)
    def update(self, t, detections: Sequence[tuple[float,float]]) -> list[Track]
    def stats(self) -> dict[str, float]      # n_tracks / n_active / n_reid / n_swap / mean_track_len
    def reset(self) -> None
```
- 返回**当前存活**轨迹（按创建顺序）；`update` 内部顺序：先对全部轨迹按匀速外推 → 贪心一对一关联（按距离升序，全局门限 `gate_px`）→ 身份互换仲裁 → 未命中记 `miss` → 删 `dead` → 未匹配检测进 `pending`（连续 `min_hits` 次才建轨）。
- `positions` 是**逐帧表**：丢失帧写 `None`；`predicted` 是外推值，**不是观测**，不要当下游真值。
- **坑**：① `predict_kind` 只支持 `"const_velocity"`，别的值在 `__init__` 就 `ValueError`（没有加速度/重力模型是刻意的 —— 二维里重力会飘）。
  ② 半径法 reid 只看位置：两球靠近到同一个 `reid_radius_px` 圈内**可能互相认错**，球从别处再入且距离超半径就**认不回**；要根治必须有外观特征（当前检测器只给圆心+半径）。
  ③ **`swap_events` / `stats()["n_swap"]` 是记账，不是修复**：两球重合附近二维位置本质退化，最优匹配胜出后显式记一次交换，而不是假装没发生。看到 `n_swap > 0` 说明这段轨迹的身份**已经不可信**。
  ④ `mean_track_len` 只统计**当前存活**轨迹（被删的短轨迹不参与，否则均值被拉低看不出身份保持好坏）。
  ⑤ 时间 `t` 必须单调；速度是"最后两个观测"差分（单帧噪声直接进速度，外推 10 帧可偏 10–20 px，正是 `reid_radius_px` 要兜的那部分）。

## 12. `calib/` —— 双目标定（离线、一次）
### 12.1 `calib/board.py`
```python
@dataclass BoardSpec:  cols=8; rows=5; square_mm=25.0          # cols/rows 是【内角点】数！pattern_size=(cols, rows)
    def object_points(self) -> np.ndarray        # (N,3) float32，单位【米】(square_mm/1000, Z=0)
    def unit_square_points(self) -> np.ndarray   # (16,2) float32，0..1 归一化板平面；def describe(self) -> str
@dataclass BoardObservation:
    found=False; corners=None (N,1,2 float32 全分辨率); homography=None     # 板平面(0..1)->图像 3x3
    sharpness=0.0    # 棋盘区域 Laplacian 方差（模糊判据）
    area_ratio=0.0   # 凸包面积 / 图像面积；margin_px=0.0 角点到图像边界最小距离
    brightness=0.0; precise=True; note=""    # precise = 角点是否来自全分辨率精确检测
    def quad(self);  def pose_points(self)
def detect_board(gray, spec, use_sb=True, normalize=True, work_width=None, exhaustive=True) -> BoardObservation
def detect_board_in_roi(gray, spec, hint, margin_ratio=0.5, margin_px=28) -> BoardObservation | None
def pose_distance(h1, h2, image_diag: float) -> float           # 归一化到图像对角线
```
**性能与用法（源码注释实测，1920x1080）**：SB + EXHAUSTIVE + ACCURACY ≈ **300 ms**（只能在"真要收的那一帧"用）；不开 EXHAUSTIVE ≈ 93 ms；缩到 640 宽 ≈ 12 ms（适合每帧粗筛）；`detect_board_in_roi()` ≈ 40 ms/路（全分辨率复核）。

**坑**：① `cols/rows` 是**内角点**数，`cv2.findChessboardCorners` 参数顺序是 `(列, 行)`。
② **`exhaustive=True` 找不到板时会穷举**，绝不能每帧开；每帧跑要用 `work_width` + `exhaustive=False`（此时 `precise=False`；640 宽下 0.5 px ≈ 全分辨率 1.5 px）。
③ `object_points()` 单位是**米** —— 这是 `StereoRig.triangulate()` 输出米制坐标的前提。
④ `pose_distance` 任一位姿为 `None` 时返回 **`0.0`**（不是 `inf`），语义别搞反；3 通道输入会自动转灰度，`work_width` 生效时角点会**换算回全分辨率**。
### 12.2 `calib/capture.py`
```text
清晰度判据（唯一实现：focus_gate）：
  1) focus < min_sharpness(800)   → 拒（绝对，低于 SB 可检出下限 ~3200，实际不触发）
  2) focus < focus_floor(300)     → 拒（绝对地板）
  3) 已收样本 ≥ focus_min_samples(5) 且 focus < 0.4 × 本路已收 focus 中位数 → 拒
  4) 整路中位数 < soft_focus_hint(4000) → 组末提示一次（不拒收）
为什么不用绝对阈值：focus 随棋盘占画面的像素数、对比度、镜头而变，一个常数只能在一台装置上调准。
```
```python
@dataclass CaptureConfig:
    target_pairs=25; min_pairs=10; min_sharpness=800.0; min_area_ratio=0.012; max_area_ratio=0.90; min_margin_px=10.0
    # ↑ min_sharpness 的语义在 2026-10-25 变过：现在是 focus（=Laplacian方差(ROI)×格子像素数），
    #   不是老版的 ROI Laplacian 方差。历史日志里的数值不能直接对比。
    min_focus_ratio=0.4; focus_floor=300.0; focus_min_samples=5; soft_focus_hint=4000.0
    min_brightness=35.0; max_brightness=230.0; require_stationary=True; stationary_max_px=3.0
    diversity_min=0.055; region_grid=3; max_per_region=5; gate_work_width=640
@dataclass CapturedPair:
    corners_left; corners_right; homography_left; homography_right
    center_left: tuple[float, float]; quality: dict = {}
class AutoPairCollector:
    def __init__(self, spec: BoardSpec, cfg: CaptureConfig | None = None) -> None;  def reset(self) -> None
    def offer(self, left_gray, right_gray, spec: BoardSpec | None = None) -> tuple[bool, str]
    def coverage_image(self, size: tuple[int, int] | None = None, scale=0.5) -> np.ndarray;  def status_lines(self) -> list[str]
    # 属性：pairs / count / full / enough / rejected / last_reason / image_size / last_obs / last_accepted
```
`offer()` 的**七道门**（顺序即代码顺序，`rejected` 按原因字符串计数）：① 两路都检测到 → ② 清晰度 → ③ 面积/边距/曝光（左右各查一遍）→ ④ **静止**（与上一帧位姿变化 ≤3 px）→ ⑤ 多样性（与已收位姿差异 ≥ `diversity_min`）→ ⑥ 宫格限量 → ⑦ 裁剪区全分辨率**精确复核**。

**坑**：① **最重要的是"静止"门**：它顺带解决了"两路没有时间同步"的问题（静态目标在任意时刻看起来都一样）⇒ **标定不需要 `sync_offset`**；用独立静帧（图片文件夹）时把 `require_stationary=False`。
② 第 ④ 道门只在"上一帧位姿存在"时比较 ⇒ **序列第一帧不受静止门约束**；第 ⑦ 道门用**精确复核后**的角点（不是粗检结果）。
③ `offer()` 期望**灰度图**（传 BGR 也能跑，但每帧多一次 cvtColor 不划算）。
④ `full` = 收够 `target_pairs`；`enough` = 达到 `min_pairs`（但 `calibrate_stereo` 实际要求 ≥4 对）。
### 12.3 `calib/stereo_calib.py`
```python
@dataclass CalibConfig:
    refine_intrinsics=True; rational_model=False; fix_principal_point=False
    reject_outliers=True; outlier_sigma=3.0; outlier_min_px=0.5; max_reject_rounds=4; min_pairs_after_reject=8
    alpha=0.0                    # stereoRectify 裁剪系数：0=裁黑边，-1=保留全部像素
    good_rms=0.35; ok_rms=0.80; good_epipolar=0.25; ok_epipolar=0.60; expected_baseline_mm=None; baseline_tolerance=0.20
@dataclass StereoCalibrationResult:
    ok=False; K1/D1/K2/D2/R/T/E/F/R1/R2/P1/P2/Q = None; roi1=(); roi2=(); image_size=(0,0)
    n_pairs_in=0; n_pairs_used=0; rms/err_left/err_right/epipolar_rms/epipolar_max/baseline_mm = nan
    per_pair_epipolar=[]; used_index=[]; rejected_index=[]; verdict=""; warnings=[]; board_desc="";  def report(self) -> str
def calibrate_stereo(objp, imgpoints1: list[np.ndarray], imgpoints2: list[np.ndarray], image_size: tuple[int, int],
                     cfg: CalibConfig | None = None, spec: BoardSpec | None = None) -> StereoCalibrationResult
```
流程：单目标定（`calibrateCameraExtended`，拿逐图误差）→ 双目联合标定（先 `CALIB_FIX_INTRINSIC` 只优化外参）→ `refine_intrinsics` 时放开内参整体精化（**取 RMS 更小的那个**）→ 按 `max(左误差,右误差,极线误差)` 的 `median + 3*MAD` 剔离群并重算（最多 4 轮）→ `stereoRectify` → 质量判定。

**坑**：① **数据不足直接返回而不抛**：`len(imgpoints1)<4` 或左右数量不等时返回 `verdict="数据不足或左右数量不一致"`、`ok=False`、其余矩阵字段 `None`。
② `objp` 可以是单个 `(N,3)` 数组（所有视图共用一块板，本项目常态）或逐视图列表；内部**强制转 `float32`**（OpenCV 5 对 float64 物体点直接报错）。
③ 离群阈值 `thresh = max(median + sigma*1.4826*MAD, outlier_min_px)`；剔完剩余对数 `< min_pairs_after_reject` 就停；`best` 取的是**各轮中 RMS 最小**那一轮的完整参数（不是最后一轮），因为精化不一定总是更好。
④ 判定：`rms≤0.35 且 epipolar≤0.25` → "优秀"；`≤0.80 / ≤0.60` → "可用"；否则 `ok=False`。**`StereoRig.from_result()` 对 `ok=False` 抛 `ValueError`** ⇒ "能算出矩阵" ≠ "能拿去用"。
⑤ **"优秀"还有一道硬条件**（2026-10-18 加）：`board_area ≥ min_board_area_for_good`（默认 8%）。
   合成真值证明**同一个 RMS 可以对应 4 倍不同的深度精度**（板占 1.9% vs 7.9%，RMS 都是 0.41 px，
   Z 误差 σ 5.76 vs 1.40 mm）—— 所以"指标完美但板太小"最高只判"可用"，并在 `warnings` 里说明。
⑥ 新增不确定度出口：`std1/std2`（来自 `calibrateCameraExtended` 的 `stdDeviationsIntrinsics`，
   **原来被丢弃**）、`fx_sigma()`、`depth_sigma_mm(z)`。后者的公式与**边界**见其文档串 ——
   含焦距项 + 视差项，**不含基线项**，属数量级估计，不是验收门限。
### 12.4 `calib/params.py`
```python
@dataclass StereoRig:
    K1, D1, K2, D2, R, T: np.ndarray            # 必填；image_size: tuple[int, int] 必填 (w, h)
    E=None; F=None; R1=None; R2=None; P1=None; P2=None; Q=None; quality: dict = {}; meta: dict = {}
    @property baseline_mm -> float              # norm(T)*1000;      def describe(self) -> str
    def rectify_maps(self, side="both")         # "left"/"right"/"both"（both 返回一对元组）
    def rectify_image(self, img, side) -> np.ndarray
    def rectify_points(self, pts_left, pts_right) -> tuple[np.ndarray, np.ndarray]
    def triangulate(self, pts_left, pts_right) -> np.ndarray        # (N,3) 米；点数不等抛 ValueError
    def epipolar_error(self, pts_left, pts_right) -> float          # 校正后 y 之差 RMS；空输入 nan
    def baseline_side(self) -> str              # "right" / "left" / "unknown"
    def disparity_sign(self) -> int             # 右相机在右 -> +1；装反 -> -1；未知 -> 0
    def save(self, path) -> tuple[Path, Path]   # 同时写 .npz + .json
    @classmethod load(cls, path) -> "StereoRig";  @classmethod from_result(cls, res, meta=None) -> "StereoRig"   # not res.ok 抛 ValueError
def load_stereo_rig(path) -> StereoRig;  def save_stereo_rig(rig, path)
```
> ⚠️ **`Q` 存了但全项目没有一处用它做重建**（只走 `P1/P2` + `triangulatePoints`）。
> 不是 bug，但"存了两套真值、只校核一套"是维护负担：要么在文档里认下
> "`Q` 仅供外部工具/导出"，要么干脆不出存。**改之前先搜一遍消费者**（2026-10-18 审查 B6）。
> 2026-10-19：`calib/export.py` 已把 `Q` 原样导出给外部工具，属于"仅供导出"的消费者。
**坐标系与单位（最要紧）**：三维点在**校正后的左相机坐标系**，原点在左相机光心、Z 向前、**单位米**（因为 `BoardSpec.object_points()` 是米）。`T` 是**左光心在右相机系里的位置**（`X_right = R @ X_left + T`；等价地 `T = -R·C`，`C` 是右光心在左相机系的位置），单位米。

**坑**：① `rectify_maps` 的映射表**依赖 `image_size`**：换分辨率要重新 `rectify_maps()` 或重标定；`rectify_image(img, side)` 的 `side` 只能是 `"left"`/`"right"`，给 `"both"` 会去解包元组直接报错。
② `disparity_sign()`：右相机在右 → **`+1`**（`T[0] < 0`，同一物点在右图更靠左，`x1-x2` 为正）；装反 → `-1`。不用自己记 —— `StereoReconstructor` 会自动取。
③ **左右装反时标定指标一样漂亮**：`baseline_side()=="left"` 时 `describe()` 会打印警告；三角测量**仍然正确**（P1/P2 自洽），但任何"假设视差为正"的匹配会 0% 配对，报错只有"配不上"。
④ `save()` 同时写 `.npz`（矩阵）和 `.json`（质量报告+元信息）；`load()` 只要求 `.npz` 存在，`.json` 缺失时 `quality/meta` 为空字典（不抛）。
### 12.5 `calib/export.py`
```python
def build_camera(rig, side) -> dict      # side="left"/"right"；K 3x3；D 拉平成 5
def build_rig(rig) -> dict               # R/T/E/F/R1/R2/P1/P2/Q + 基线 + 质量 + 元信息
def build_export(rig) -> dict            # {"left": ..., "right": ..., "rig": ...}
def write_export(rig, out_dir, formats=("json", "yaml", "toml")) -> list[Path]
def toml_dumps(data: dict) -> str        # 最小 TOML 序列化器（stdlib 只有只读 tomllib）
```
**单位 / 约定**：`T` 拉平成 3 个分量、单位**米**；`D` 拉平成 5 个（k1,k2,p1,p2,k3）；三维点仍是校正后左相机系。
**产物**：`left_camera.<ext>` / `right_camera.<ext>` / `stereo_rig.<ext>` + `summary.txt`。
**格式**：JSON（stdlib）、YAML（PyYAML，可选）、TOML（自带 writer）。
请求 YAML 但缺 PyYAML 时**一个文件都不写**（避免留下半套）；未知格式抛 `ValueError`。
### 12.6 `calib/world_frame.py` —— 世界系（棋盘 PnP）
```python
MAX_BOARD_RMSE_PX = 2.0                      # 超过就不许当世界系
@dataclass WorldFrame:
    T_world_cam: np.ndarray                  # 4x4，p_world = T @ p_left_camera
    source: str; units="m"; note=""; board_rmse_px: float|None; created_at: str
    ground_axis: str = "z"                   # "z" / "-z"；随 save/load 往返
    def apply(self, points: (N,3) | None) -> (N,3) | None      # None 原样返回
    def describe(self) -> str                # 两行：几何定义 + ground_axis 符号语义
    def save(self, path) -> Path                               # JSON，矩阵一行一个 4x4 行
    @classmethod load(cls, path) -> WorldFrame                 # 任何校验不过就 ValueError
    @classmethod from_camera_board(cls, T_cam_board, *, source="manual", units="m",
                                   note="", board_rmse_px=None, ground_axis="z") -> WorldFrame
def from_board(image_path, board_spec: BoardSpec, rig: StereoRig, *,
               ground_axis="z", rig_path=None) -> WorldFrame
```
- `from_board`：读图 → **校验分辨率** → `detect_board` → `solvePnP`（用左路 `K1/D1`）→ `T_cam_board` → 取逆。CLI 在 `apps/tools/set_world_frame.py`（`--image --board-cols/--board-rows/--board-square --rig --out --note --ground-axis {z,-z} --dry-run`）。
- **校验规则**（`__post_init__`，全部抛 `ValueError` 并说明是哪一条）：4x4 / 全 `finite` / 最后一行必须是 `[0,0,0,1]` / 旋转块 `RᵀR−I` 最大偏差 ≤ 1e-6 / `det(R) > 0`（<0 是镜像不是旋转）/ `ground_axis ∈ {"z","-z"}`（写别的值等于"轴符号说不清"）。`load()` **绝不回退成单位矩阵**，文件缺失抛 `FileNotFoundError`；老文件没有 `ground_axis` 键 → 按 `"z"` 读。
- **`describe()` 必须含 `ground_axis`**：矩阵本身看不出 Z 朝上还是朝下，`-z` 还会把 Y 一起反向。所以它输出两行 —— 第一行几何（原点/XYZ/单位/rmse），第二行符号语义（`z` → Z 朝地下、Y 朝画面下；`-z` → Z 朝天空、Y 指向画面上方）。`--dry-run` 打印的就是它。
- **拍图分辨率校验（2026-11-20 补）**：`from_board()` 读图后比 `(w,h)` 与 `rig.image_size`，不一致抛 `ValueError`（文案含**两边尺寸**、`rig_path` 对应的标定文件、下一步"用标定同分辨率重拍 / 重标"）；`image_size` 缺失或无效（老 npz 读成 `(0,0)` / `None`）时**跳过校验但不静默**：`print` 一行 `[世界系] 注意：…` 并写进 `WorldFrame.note`。`rig_path` 只用于报错文案。
- **坑**：① 原点取画面**最"左下"的内角点**（按 `x−y` 最小），消掉 SB 与经典检测器之间 180° 的起点差；相机装歪 90° 时"左下"仍可算但已不等于物理左下角。
  ② **`Z` 是棋盘法向，不是"重力反方向"**：板平放地面、图案朝上拍时默认 +Z 指向地面以下，要 Z 朝上必须传 `ground_axis="-z"`（Y 同时反向以保右手系）。符号只能靠人确认，程序不替你猜；两种取值的 rmse **完全相同**，看误差判断不了符号。
  ③ `board_rmse_px` **只反映 PnP 拟合好坏**，看不出来"板是不是真的平放在地上"——板没放平、相机与板不同侧，世界系都会歪而 rmse 依然漂亮。
  ④ 内部计算一律 `left_camera`，只在**导出那一刻**转世界；没这个文件时导出必须写 `coordinate_system="left_camera"`，不许静默当世界系。parallel 的校正系**不支持**世界系变换（差一个 `R1`），会拒绝并说明原因。

### 12.7 `calib/metric_check.py` —— 尺度自检（三种入口，只有一种是真的）
```python
_GOOD_TRANS_MM=2.0; _OK_TRANS_MM=5.0; _GOOD_ROT_DEG=0.5; _OK_ROT_DEG=1.5   # 模块级默认
def stereo_scale_consistency(T_CR_CL, poses_left, poses_right, *,
        good_trans_mm=2.0, ok_trans_mm=5.0,
        good_rot_deg=0.5, ok_rot_deg=1.5) -> dict
def square_size_check(T_CR_CL, objp, corners_left, corners_right, spec,
        K_L, D_L, K_R, D_R, *, good_pct=1.0, ok_pct=3.0) -> dict
def board_scale_error(T_world_cam, objp, corners_list, spec, K, D) -> dict
```
- `stereo_scale_consistency` 返回：`n / translation_mm(均值) / median_mm / p95_mm / max_mm / per_pair_mm / worst_index / rotation_deg / rotation_median_deg / rotation_p95_deg / max_rotation_deg / per_pair_deg / insufficient / verdict / rotation_verdict`。输入 `T` 与位姿**平移单位是米**，输出 mm。
- `square_size_check` 返回：`n / square_mm / square_mm_expected / error_mm / error_pct / scale_ratio / per_point_mm / verdict`。先 `undistortPoints(..., P=K)` 再三角化（针孔口径，与 `none_raw` 的 `P1/P2` 一致），相邻间距取"每点到最近邻的中位数"（不依赖角点顺序）。
- `board_scale_error` 返回：`n_views / reproj_rms_px / per_view_rms_px / tautological(恒 True) / verdict / note`。
- `verdict` 只有四种字符串：`优秀 / 可用 / 不可用 / 数据不足`，**不抛异常**（参数形状非法才 `ValueError`）。

**调用顺序 / 前置条件**：`stereo_scale_consistency` 与 `square_size_check` 都要**先有 `T_{C_R←C_L}`**（来自 `calibrate_stereo` 或链式合成）与**两路对同一批摆放的板位姿**；`poses_left[i]` 与 `poses_right[i]` 必须逐项对应同一物理摆放。跨路一致性**只对单板模式（`T_AB ≈ I`）成立** —— 双板模式下左看 A、右看 B，两组板位姿不在同一个板坐标系里。

**坑**：① **`n<2` 不发合格证**：一对时"残差 = 0"是恒等式（外参可能正是由这一对反向构造的），不构成证据；此时返回 `insufficient=True`、`verdict` 与 `rotation_verdict` 都是「数据不足（n=…）」并说明至少要 2 张同摆放参考图，**不要**把 `median_mm=0.0` 当合格。
② **`board_scale_error` 是自检无效的入口，不是证据**：单目 `solvePnP` 的平移与 `objp` 同尺度（`objp` 由 `spec.square_mm` 生成），缩放 `objp` 会同步缩放 `t`，重投影完全不变 ⇒ 量出的方格边长必然自洽。它只反映"角点与板模型一致"，`tautological=True` 已如实标注。要验尺度请用前两个函数。
③ **残差只随 `‖T‖` 放大，不随板距**：外参尺度错 k 倍时残差 = `|k−1|·‖T‖`（一阶式）。两组实测（**不同参数组，别并成一句**，各自只证明一件事）：
- **证明"与板距无关"**：`‖T‖=100 mm`、`k=1.1`，板放 0.4 / 0.8 / 1.2 m，残差**恒为 10.00 mm**（= 0.1×100 mm）—— 板距变了三倍，数一个毫米都没变；
- **证明"与基线成正比"**：`k=1.1`、`‖T‖=781 mm` → **78.10 mm**（= 0.1×基线），只给绝对值，不含板距变量。

原因：这个差是"两个板原点在同一坐标系里的偏移"，与板本身多远无关 —— 它抓的是**外参尺度**，不是"远处更不准"。
④ 重投影 RMS 对这一类错**完全免疫**（齐次坐标乘常数不变）：所以"RMS 0.2 px + 外参整体偏 10%"可以同时成立，只有这两个函数能看出来。

### 12.8 `calib/validation.py` —— 交叉验证与**泄漏守卫**
```python
GOOD_RATIO_MAX = 1.5; WARN_RATIO_MAX = 2.5
FitFn = Callable[[Sequence[int], Sequence[int]], dict[str, Any]]
def cross_validate(fit_fn, n_items, *, k=4, seed=0) -> dict
```
返回：`k / n_items / seed / degenerate / train_rms_median / test_rms_median / ratio / ratio_per_fold_median / verdict / leak_check("ok"|"unverified") / estimator_declared / n_folds_ok / per_fold / failed_folds / note`。

**`fit_fn` 的契约（2026-10-26 收紧，别漏）**：`fit_fn(train_idx, test_idx) -> dict`，至少含 `"train_rms"`、`"test_rms"`（都是 float，非有限即该折失败），且**必须报**两样东西：`"used_train_idx"`（真正拿去拟合的索引）与 `"train_rms_estimator"` / `"test_rms_estimator"`（两个估计量的名字）。

**两道守卫**（都只防"从外面看不出来"的误用）：
| 情况                                            | 结果                                                                                                             |
| ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| 报了 `used_train_idx` 且与 `train_idx` 集合相等 | `leak_check="ok"`，正常判定                                                                                      |
| 报了但对不上（训练集与测试集重叠）              | 该折抛 `ValueError` → 进 `failed_folds`，`verdict` 带「部分折失败」                                              |
| **没报 `used_train_idx`**                       | `leak_check="unverified"`，`verdict` 整条降级为「未验证（…无法排除训练/测试泄漏）：按比值看是「…」，但不可采信」 |
| 两个 `*_estimator` **不同名**                   | 该折抛 `ValueError`（"这种比值只能说明两把尺子不一样长"）                                                        |
| 两个 `*_estimator` **没声明**                   | `estimator_declared=False`，`verdict` 追加「比值可能混了尺子，仅供参考」                                         |

**为什么泄漏守卫必须有**：这个工具最容易的误用是 `fit_fn` 忽略传入的 `train_idx`、每折都拿全量数据训练 —— 于是 `test_rms == train_rms`、`ratio` 恒 1.0、一路绿灯，**从外面完全看不出来**。

**为什么估计量守卫必须有**：本项目真踩过 —— 训练侧取 `stereoCalibrate.rms`（位姿全自由、DOF 更多、恒偏小），测试侧取"参数焊死的重投影"，两把尺子长度不同，**比值固定偏乐观约 0.2**（已实测，方向不变、量级固定）。现在训练/测试都走 `reprojection_rms_px(..., camera="right")`，同度量。

**`ratio` 只作筛查信号，不作验收依据**，三条理由：① 它随**口径**与**有效对数**变化（同度量后仍会变）；② 左目残差与 `(R,T)` 无关、是纯内参+噪声的常数底，把它合并进来会把比值往 1 压（实测稀释 **约 4%**，不是几十个百分点 —— 那是估计量混用的账，两笔别记在一起）；③ 两笔偏差方向与量级都不同，**不能靠经验系数修正**。报告里因此印三种口径：**判外参泛化只看右目 ratio**；"两目合并"与"左目"只用来回答"被稀释了多少"（左目自检要求 train/test 接近 1，差太多说明口径混了），**不同度量的绝对 RMS 不可横向比大小**。

**其它坑**：① `k=1` 退化（训练集自评，`test_rms` 恒等于 `train_rms`），`degenerate=True` 且 verdict 明写"不能作为泛化证据"；`n_items < k` 抛 `ValueError`（不硬算一个 ratio）。
② 失败折**不记 0**：`_median` 空列表返回 `nan`（0 会被后续比值当成真值）；"忘了返回 `test_rms`"会得到 `ratio=0` 的"完美泛化"，所以缺项一律算失败折。
③ 门限三档：`≤1.5` 泛化良好 / `≤2.5` 轻度过拟合 / `>2.5` 过拟合。**这两个门限没有推导，是经验值**，且实测同度量下的比值距 1.5 只有一成多余量 —— 判定对口径敏感，别把"泛化良好"当合格证。`ratio` 只回答"换一批图还能不能复现"，不回答"绝对精度够不够"（后者看 RMS 本身与板占画面比例）。
④ `seed` 必须固定：同一 seed 必须给同一划分，否则两次的比值没法比较。
⑤ 只对 **joint** 有意义。chain 的单目内参在全量数据上标好后冻结，在子集上"重跑"得到的是另一份内参，检验不到外参泛化 —— 那不是交叉验证，是重标一次。

### 12.9 `calib/uncertainty.py` —— 外参不确定度（数值雅可比 + delta 法）
```python
STEP_ROT = 1e-6; STEP_TRANS = 1e-6      # 中心差分步长（rad / m）
COND_MAX = 1e12                          # 列归一化后的条件数门限
def extrinsic_sigma(objp, imgpoints1, imgpoints2, K1, D1, K2, D2, R, T,
                    image_size, *, noise_px=None) -> dict
```
返回：`sigma_baseline_mm / sigma_angle_deg / sigma_t_mm(3,) / sigma_r_deg(3,) / cov_t_mm2 / cov_r_rad2 / noise_px / noise_px_from_residual / ssr_px2 / dof / n_residuals / n_views / cond / baseline_mm / focal_px / depth_sigma_mm_at(callable) / note`。

**算法**：把 `(R,T)` 当非线性最小二乘的 6 个参数，残差取**右目**重投影（`r_i = project(R·R_i, R·t_i + T) − p2_i`，各视图板位姿 `(R_i,t_i)` 由左目 `solvePnP` 给出并**固定**），在解处取数值雅可比：

```
Σ_θ = σ²·(JᵀJ)⁻¹,   σ² = SSR/(m−6),   m = 2·角点数·视图数
σ_B = √(uᵀ Σ_t u), u = T/‖T‖            # 用完整协方差，不只取对角
σ_angle = √(tr Σ_r)                     # 弧度 -> 度
σ_Z(Z) = Z·√((σ_d/d)² + (σ_B/B)²),  d = f·B/Z,  σ_d = √2·noise_px
```

**最后一条是"这是量级估计、不是置信区间"的实测依据**（2026-10-26，30 次噪声实现）：

| 场景                           | σ_B 预测 / 实测散布 |
| ------------------------------ | ------------------- |
| 0.5 px 噪声、12 视图、54 点    | **0.74**            |
| 基线 60 / 200 mm、1 px、3 视图 | **0.54–0.62**       |
| 2 m 远场（板距 1.2–2.0 m）     | **0.41**            |

即**系统性偏低**（0.54–0.74，即偏低 **1.4–1.9 倍**），远场更糟：远处的板位姿与 `T` 强相关，而条件协方差（各视图位姿、内参固定）吃不到这部分。标定距离附近可信，拿去做验收会偏乐观。

> ⚠️ **不得把 `σ_Z` 当作安全余量。** 它站在偏低的一侧：报告写"±14 mm"时，真实偏差**不会**更小，只会可能更大（远场可达 2 倍以上）。要当界用只能乘一个保守系数并在报告里写明系数来源，不能默认"±X 就是上界"。

**坑**：① **只统计右目**：左目投影完全不含 `(R,T)`，雅可比列恒 0，塞进 SSR 只会把 `σ²` 抬高却不给 `JᵀJ` 任何贡献（有偏）。
② 失败**返回 `{"error": ...}`**（缺 `dof`、基线为 0、雅可比奇异、`JᵀJ` 不可逆），**不返回 inf/nan** —— 否则下游会当成"很大的数"继续算；调用方必须判 `"error" in u`。
③ 奇异判据先**按列归一化再算条件数**（旋转列 px/rad 与平移列 px/m 天然差 1e5 量级，不归一化门限无从设起）；`dead_params` 指出"哪个自由度对残差完全没影响"。
④ `noise_px` 缺省由 `√(SSR/(m−6))` 从**同一批残差**估 —— 既拟合又估噪声，`σ²` 会略偏小（与条件协方差的偏大方向相反，净结果是上面的 0.74）。想显式给就传 `noise_px`。
⑤ **不含 `σ_f`**（要 `calibrateCameraExtended` 的 `stdDeviationsIntrinsics`，见 `StereoCalibrationResult.fx_sigma()`）；两项都要时把平方相加。`T` 必须是**米**，`depth_sigma_mm_at(z)` 的 `z` 也是米、越界返回 `nan`。
⑥ `image_size` 参数**在函数体里没用**（`projectPoints` 不需要），保留只为与 `calibrate_stereo` 调用口径一致。

### 12.10 `calib/axial_calib.py` / `calib/device90_calib.py` —— 两种独立标定求解器（2026-11-18，按源码抄写，未额外跑 inspect.signature）

两个**独立入口**，不是 `calib_method` 插件（见 §0b 注册表仍只有 joint/chain）；产物都是 `StereoCalibrationResult`，能被 `StereoRig.from_result` 直接消费。

```python
# --- calib/axial_calib.py：模式 A 轴向标定（板正对相机 + 已知沿光轴推进量 ΔZ + 已知方格边长 L）---
class AxialCalibError(ValueError)                     # 输入/物理前提不成立（非数值失败）
@dataclass AxialIntrinsics:
    fx, fy, cx, cy, Z0: float                         # 焦距/主点(px)、第一帧(ΔZ=0)板深度(m)
    Z0_x, Z0_y: float                                 # 横/纵分别拟合的 Z0，差异大说明板没正对
    distortion_calibrated: bool                       # 诚实标记：轴向恒 False
    rms_px: float; n_frames: int; delta_z: np.ndarray
    per_frame_rms: list[float] = []; notes: list[str] = []
    def matrix_K_D(self) -> tuple[np.ndarray, np.ndarray]   # D = zeros(5)，恒全 0
def solve_axial_intrinsics(corners_list, delta_z, spec: BoardSpec | None = None, *,
                           min_frames: int = 3) -> AxialIntrinsics
    # 1/p 对 ΔZ 线性拟合解 f/Z0，主点用网格对称性；帧数不足/ΔZ 极差≈0/斜率非正 -> AxialCalibError
def solve_axial_stereo(corners_left, corners_right, delta_z, spec: BoardSpec | None = None,
                       image_size: tuple[int, int] = (0, 0), cfg: CalibConfig | None = None, *,
                       assume_parallel: bool = False) -> StereoCalibrationResult
    # 两路各解内参 + 锁内参 stereoCalibrate 定外参（失败回退逐帧 solvePnP 链）
    # assume_parallel=True 强制 R=I（机械保证两光轴平行时），rectification=none_raw

# --- calib/device90_calib.py：模式 B 中轴 90°（同一块板绕中轴转 90°，分别正对互成 90° 的两相机）---
class Device90CalibError(ValueError)
@dataclass Device90Diagnostics:
    Z_A=Z_B=nan; Q_A=Q_B=None; roll_A_deg=roll_B_deg=nan; rms_A_px=rms_B_px=nan; n_A=n_B=0; notes=[]
def board_center(spec: BoardSpec | None = None) -> np.ndarray       # 板几何中心（米，BoardSpec 系）
def R_AB_matrix(axis: str = "y", angle_deg: float = 90.0) -> np.ndarray   # 板 A→B 纯旋转
def solve_device90(corners_A, corners_B, K1, D1, K2, D2,
                   spec: BoardSpec | None = None, image_size: tuple[int, int] = (0, 0), *,
                   axis: str = "y", angle_deg: float = 90.0,
                   Z_A: float | None = None, Z_B: float | None = None,
                   cfg: CalibConfig | None = None,
                   diagnostics: Device90Diagnostics | None = None) -> StereoCalibrationResult
    # 链式外参 T_{C_B<-C_A}=T_camB_B·T_AB·inv(T_camA_A)，无需共视/同框图；依赖内参 K/D 已知
    # 产物 stereo_mode="angled"、rectification="none_raw"，可喂 angled_geometry.triangulate_pixels
```

**关键口径**：轴向模式只沿光轴推进，畸变不可观测 —— `distortion_calibrated=False`、`D` 全 0，**不得声称已标定畸变**（要畸变走常规多姿态标定）。中轴 90° 模式 `axis/angle` 必须与装置实际中轴/转向一致，装错会给出基线量级不对的"好看"解，务必传 `cfg.expected_baseline_mm` 并把 `baseline_tolerance` 设到 0.05 量级（默认 0.20 会漏掉装错）。两种模式的 meta 字段见 `数据标准.md` §1 的标定模式字段说明；元数据写入靠 CLI（`apps/calibrate_axial.py` / `apps/calibrate_device90.py`）。

## 13. `stereo/` —— 双目逐帧
### 13.1 `stereo/sync.py`
```python
@dataclass SyncConfig:  max_offset=180; min_overlap=30; smooth=1; flash_min_jump=2.0
@dataclass SyncResult:
    offset: int          # left[i] 对应 right[i + offset]
    confidence: float    # 归一化互相关系数峰值
    peak_count=0; note="";  def __str__(self) -> str;  @property usable -> bool      # confidence >= 0.35 且 peak_count >= 1
def brightness_series(frames, smooth=1) -> np.ndarray
def brightness_jumps(series, min_jump=1.0) -> dict   # max_jump / threshold / peak_count / peak_indices
@dataclass TimestampSync:  offset; median_abs_ms; confidence; usable; note
def timestamp_offset(left_times, right_times, fps, max_offset=5) -> TimestampSync
def find_offset(series_left, series_right, cfg: SyncConfig | None = None) -> SyncResult
def apply_offset(offset: int) -> str      # 把偏移翻译成"取帧"的人话，用来核对正负号
```
**坑**：① `find_offset` 在**亮度差分**上做归一化互相关（突变被放大）；序列短于 3 帧返回 `SyncResult(0, 0.0, 0, "序列太短")`；若所有候选偏移都因 `min_overlap` 被跳过，返回的也是 `offset=0, confidence=0.0` —— **别把 `offset=0` 当成"对齐了"**，看 `usable`/`confidence`。`peak_count` 取左右两路较大者（闪光可能只在一路明显）。
② `brightness_series` **会跳过 `None` 帧**，返回序列的下标**不一定等于帧号**，别拿它当帧索引；搜索范围受 `max_offset=180` 限制，偏移正负语义统一是 `left[i] ↔ right[i+offset]`。
③ **没有闪光时也会有一个候选 offset**（普通场景变化也能相关）。录制端先用 `brightness_jumps` 看 `max_jump/peak_count`；`sync.method=none` 的会话三维不可直接用。
④ 录制器在没有闪光时会用 `timestamp_offset` 自动兜底：用每帧 `perf_counter` 时间戳估计 offset，`method=timestamp`。它只是**主机读帧时间**的近似（非硬件触发），中位差超过半帧就拒绝；慢速场景可用，快速球仍应闪光/硬件同步。
### 13.2 `stereo/pair_source.py`
```python
class PairSource:
    def __init__(self, camera: tuple[int, int] | None = None, video: tuple[str | Path, str | Path] | None = None,
                 directory: tuple[str | Path, str | Path] | None = None, sync_offset=0, width=1920, height=1080, fps=60,
                 backup_camera: int | None = None, backend: int | None = None) -> None
    def read(self) -> tuple[bool, np.ndarray | None, np.ndarray | None]     # (ok, left, right)
    def describe(self) -> str;  def report_warnings(self) -> None;  def release(self) -> None
    # 属性：sync_offset / kind / n / warnings / is_live / image_size（取左路）
def from_args(args) -> PairSource      # 读 --left/--right/--auto-cameras/--left-video/... 等
def resolve_backend(value: str | None) -> int | None    # None/""/"auto" -> None；未知名字抛 RuntimeError
def describe_info(info) -> str
class FrameReader(ABC):   def read() -> (ok, frame);  def seek(index) -> (ok, frame)   # 不允许倒退
class VideoFrameReader(FrameReader)  # 用 CAP_PROP_POS_FRAMES 跳帧；class ImageDirFrameReader(FrameReader) 一个图片文件夹 = 一个时刻序列
class CameraPairReader(FrameReader):
    def __init__(self, left_index, right_index, width=1920, height=1080, fps=60, backup_camera=None, backend=None)
    # 属性 info -> (左 SourceInfo, 右 SourceInfo)；seek() 抛 NotImplementedError
CAMERA_WAIT_TIMEOUT = 5.0;  CAMERA_ALIGN_STEPS = 2;  CAMERA_POLL_INTERVAL = 0.005
```
**调用顺序 / 前置条件**：① `camera=/video=/directory=` **只能且必须给一个**，否则构造时抛 `RuntimeError`。② `sync_offset` 语义与 `sync.apply_offset()` 完全一致：**返回的左路第 i 帧与右路第 i+offset 帧是同一物理时刻**；视频/图片分支在**第一次 `read()` 时一次性吸收**偏移，之后逐帧前进（中间不做任何 seek）。③ 相机分支要求左右索引不同，且两路**必须用同一个后端**打开（同进程混用 DSHOW+MSMF 时 MSMF 那路会"打开成功但抓不到帧"）；`read()` 等两路都出新帧（右路至少前进 `CAMERA_ALIGN_STEPS` 帧），最长等 5 s。

**坑**：① **实时相机分支不支持 `sync_offset`**：会往 `warnings` 追加一条说明并**把偏移归零**（不抛）—— 两路各在自己的线程里跑到最新帧，帧号没有共同基准；要真同步得靠硬件触发，或先录成视频再处理。
② 来源不同、"数量不一致"的处理不同：**图片文件夹**数量不等 → 直接 `RuntimeError`；**两段视频**帧数不等 → 只追一条 warning，取到短的那路结束为止；任一路读尽 → `read()` 返回 `(False, None, None)`（不抛）。
③ `FrameReader.seek()` **不允许倒退**（`ValueError`，因为成对取帧只向前走）；`CameraPairReader.seek()` 直接 `NotImplementedError`。
④ `PairSource.image_size` 取**左路**尺寸（两路分辨率不同时不报错，只有 `CameraPairReader` 打印实测差异）。
⑤ `from_args(args)` 读的参数名与 `apps/_common.add_stereo_source_args()` 一一对应；`--left/--right` 可以是索引、**角色名**（查 `config/cameras.json`）或 `auto`；`--auto-cameras` 时两台由 `select_cameras` 挑。
⑥ **默认值的两层，别混**：上面 `PairSource.__init__` / `CameraPairReader.__init__` 的 `width=1920, height=1080, fps=60`（`fps` 已是 **float**）是**库构造签名**的真实默认，没改过；
  而共用 CLI 参数 `apps/_common.add_stereo_source_args()` 的默认已是 `--width/--height=0`（= 逐档自适应协商）、`--fps=DEFAULT_FPS=24`。
  **分辨率档位与帧率建议档的唯一产地在 `sphere_tracker/video_profiles.py`**：8 档（P8K/P5K/P4K/P1440 默认/P1080/P720/P540/P480）、`DEFAULT=1440p`、`DEFAULT_FPS=24`、`FPS_TIERS=(24,30,60,120,240,500,1000)`；
  `fps` 保留小数不截断。库签名的 1920/1080/60 只是显式档的历史默认，改档请先改 `video_profiles.py` 并同步调用方，不要在这份参考里硬改数字。
### 13.3 `stereo/triangulate.py`
```python
@dataclass MatchConfig:
    max_epipolar_px=2.5; min_disparity_px=0.5; max_disparity_px=600.0; require_radius_ratio=False; radius_ratio_tol=0.35
    disparity_sign=0        # 0 = 不检查符号（只按 |视差| 判范围）
@dataclass StereoPoint:
    t: float;  xyz: np.ndarray      # (3,) 米，【校正后左相机系】
    u_left; u_right: tuple[float, float]
    epipolar_px: float              # 配对时的极线偏差（校正后 y 之差）；residual_px 交会残差（投回两路的平均误差）
    radius_left=nan
    @property distance_m -> float;  @property confident -> bool    # residual_px <= 2.0（硬编码）
def rectify_point_left(rig, x, y) -> tuple[float, float];  def rectify_point_right(rig, x, y)
def match_epipolar(rig, det_left: Detection, candidates_right: list[Detection], cfg: MatchConfig | None = None) -> tuple[Detection | None, float]    # 找不到 -> (None, inf)
def triangulate_one(rig, det_left, det_right, t, epipolar_px=float("nan")) -> StereoPoint
def epipolar_dy(rig, det_left: Detection | None, candidates_right: list[Detection]) -> float | None    # 校正后 v 差的最小值；缺一边 -> None
@dataclass GeometryVerdict:
    n: int;  med: float;  p95: float       # 逐帧样本数 / 校正后 y 差的中位 / p95
    epipolar_rms=nan;  epipolar_max=nan    # 标定自报（取不到就是 nan，**不编数**）
    suspect: bool                          # 判"录像几何与标定对不上"
    @property has_ref -> bool;  @property ratio -> float    # 实测中位 / 标定自报；无自报值 -> nan
def geometry_verdict(dy_samples, quality=None, min_samples=GEOM_MIN_SAMPLES) -> GeometryVerdict | None    # 样本 < 20 -> None
@dataclass StereoReconstructor:
    rig: StereoRig;  cfg: MatchConfig = MatchConfig();  trajectory: list[StereoPoint] = []
    n_frames=0; n_matched=0; n_no_left=0; n_no_right=0; n_rejected=0
    dy_samples: list[float] = []       # 两路都检出的帧才攒；几何判读的**唯一**输入
    def __post_init__(self) -> None    # 自动取 rig.disparity_sign()（仅当 cfg.disparity_sign == 0）
    def update(self, t, left: Detection | None, right_dets: list[Detection]) -> StereoPoint | None
    @property match_rate -> float      # n_matched / n_frames
    def as_array(self) -> np.ndarray   # (N,4) = [t, X, Y, Z]；无点时 (0,4) 零数组
    def summary(self) -> str
    def geometry_lines(self) -> list[str]                       # 刚测出来的 [诊断] 几行（样本不足 -> []）
    def low_match_diagnostic(self, low_rate=0.10) -> list[str]   # 配对率 >= low_rate 就返回 []
    def geometry_failure_message(self, v=None, center_box_share=None) -> str
        # "录像与标定对不上"的报错文案；v=None 时自己算。
        # center_box_share>0.5 时**先**给"圆心没钉到轮廓上"这条更近的原因
        # （B-132：左右高光错位本就 ≈6 px，与几何坏掉长得一样，但修法相反）
```
**调用顺序**：
```python
rig = StereoRig.load("config/stereo_calibration")
rec = StereoReconstructor(rig)                             # 自动定视差符号
pt = rec.update(t, left=det_left, right_dets=dets_right)    # left 是【单个】Detection，不是列表
```
- **三维点在"校正后"左相机系**：要跟原始左相机系的真值比，必须先 `truth_rect = rig.R1 @ truth`（`tests/selftest_stereo_branch.py:263` 就是这么做的）；交会残差同理必须用**校正后**的像素算，`triangulate_one` 内部已处理（用原始 K1/K2 去比会得到几十像素的假残差）。
- 视差先按**绝对值**判范围、符号**单独**判（`cfg.disparity_sign`）—— 为了在左右装反时不至于"静默全拒"（那样只会报"配不上"，指不到原因）；`disparity_sign == 0` 表示不检查。

**坑**：① `update()` 的 `left` 是**单个 Detection 或 `None`**，`right_dets` 才是列表；返回 `None` 的三种原因分别计入 `n_no_left`/`n_no_right`/`n_rejected`（`summary()` 会报出来）。
② `match_epipolar` 选的是极线偏差 `dy` **最小**的候选（`cost = dy`），不是"最像球"的那个；`require_radius_ratio=False` 时不比较左右半径（同型号相机才该开）；`triangulate_one` 的 `epipolar_px` 默认 `nan`（想要值就传 `match_epipolar` 的第二个返回值）。
③ `as_array()` 无配对点时返回 `(0,4)` 零数组 —— 形状和内容都要判。
④ **几何守卫**：`rec.dy_samples` 攒的是"两路都检出"那些帧的校正后 v 差最小值，也是 `geometry_verdict()` 的**唯一**输入；`apps/run_stereo_video.py --geometry-guard N`（默认 30，`0`=关）跑满 N 对就判一次，判 `suspect` 则打印 `geometry_failure_message()` 并**退出码 2、不产出任何文件**（拒绝出数；放宽 `--max-epipolar` 只会把错配变成"看起来正常"的数字）。判据只在 `geometry_verdict()` 写一份 —— `geometry_lines()` / `low_match_diagnostic()` / 入口守卫三处共用。守卫**跑满 N 对后一直等到样本够**（`dy_samples < 20` 时 `geometry_verdict()` 返回 `None` = "还没法判"，**不消耗**这次机会，下帧再判），跑完仍未判成会打印「没能判读（样本不足）」—— **没样本 ≠ 没问题**。
### 13.3b `stereo/angled.py` + `stereo/angled_geometry.py`（大角度）

```python
@dataclass AngledMatchConfig:
    max_epipolar_px=8.0; min_angle_deg=5.0; max_reproj_px=3.0
@dataclass AngledPoint:
    t, xyz, u_left, u_right, epipolar_px, residual_px, angle_deg, radius_left
class AngledReconstructor:
    def __init__(rig, cfg=None)
    def update(t, det_left, right_dets) -> AngledPoint | None
    # counters: n_matched / n_no_left / n_no_right / n_rejected
    # as_array() -> (N,4) [t,X,Y,Z]; match_rate; summary()
```

- 用原始像素 + K/D/R/T，不通过 `stereoRectify`；
- 左点算 F 极线，右候选按**到极线距离**匹配；
- 两射线最近点交会，`xyz` 在**原始左相机系**；
- `stereo_mode=angled` 时 `coordinate_system=left_camera`；
- 平行模式仍走 `13.3` 的 `StereoReconstructor`，两套坐标不能混。

`stereo/stereo_mode.py`：`normalize_mode/infer_mode/mode_label/optical_axes_angle_deg/geometry_advice`；
旧 session/标定无字段 → `parallel`。

### 13.3c `stereo/fov_align.py` —— 自动视场对齐

```python
@dataclass(frozen=True)
class FovAlignment:
    reference_side: str                    # "left"/"right"：较窄、更放大的一路；不裁剪，必要时降采样
    crop_side: str                         # "left"/"right"/"none"：被裁剪的一路
    crop_rect: tuple[int, int, int, int]   # (x, y, w, h)，在 crop_side 的像素系里
    target_size: tuple[int, int]           # 两路共同输出尺寸 = 参考帧/裁剪区逐轴取小，再向下取偶（不放大）
    scale: float                           # left -> right 像素尺度；1.0 = 视场基本一致
    method: str                            # "sift"/"orb"
    inliers: int;  quality: float          # quality ∈ [0, 1]
    @property active -> bool               # crop_side != "none"
    def apply_side(self, frame: np.ndarray, side: str) -> np.ndarray
    def apply(self, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]
    def to_dict(self) -> dict[str, Any]
    @classmethod from_dict(cls, data: Mapping[str, Any]) -> "FovAlignment"

def estimate_fov_alignment(left: np.ndarray, right: np.ndarray, *,
                           min_inliers: int = 30,
                           max_scale: float = 2.5) -> FovAlignment
```

**输入 / 边界**：`left`/`right` 支持 BGR / BGRA / 灰度 `np.ndarray`（uint8 最稳）；模块是纯函数 / 纯数据，不读配置、不碰相机、不写文件。

**做法**：SIFT 优先、ORB 兜底；Lowe ratio + 双向交叉校验；RANSAC homography（失败退仿射）。由内点拟合 `left->right` 尺度：`scale <= 1` 以左为参考裁右（视场接近时通常先判 `none`），`scale > 1` 以右为参考裁左。`|scale-1| <= 0.05`、中心偏移 `<= 10%` 短边、参考帧可见面积 `>= 90%` 时返回 `crop_side="none"`。`target_size` = 参考帧与裁剪区**逐轴取小、再向下取偶**（无损 MKV 只支持偶数宽高；**任何一路都不放大**）；`apply()` 对裁剪路做 `crop_rect` 裁剪 + 缩放，对参考路只降采样到 `target_size`，两路最终同尺寸；`crop_side="none"` 时 `target_size` = 参考帧原生尺寸、不裁不缩。取参考帧原生尺寸会把裁剪区放大 1.5x（实测 Laplacian 清晰度 ~103 -> ~20，标定 55 门限整批拒帧），所以这里优先保证不放大。

**失败语义（不会静默退化）**：参数非法或估计失败都抛 `ValueError`；估计失败时错误串带 SIFT/ORB 各自的失败原因 —— 帧非法 / 太小（短边 < 16）、没有足够特征或匹配、内点 `< min_inliers`、尺度超出 `[1/max_scale, max_scale]`、参考帧四角映射失败或退化（含非有限坐标、宽或高 < 2）、参考帧映射后的包围盒只占广角帧 `< 20%` 面积、**参考帧四角映射后任一维越出广角帧 > 16 px**（参考帧未完整落在广角帧内 = 共同视场不足；容忍轻微旋转/取整误差（最多 16 px），**不会**裁掉越界角按“部分重叠”继续主动对齐）。`crop_side="none"` 只在近同视场判据通过时出现，**不是失败兜底**；`min_inliers` 至少 4，`max_scale` 必须 > 1.0。构造 / `from_dict()` 还会校验字段（`reference_side`/`crop_side` 取值、`crop_side != reference_side`、`crop_rect` 为 4 个整数且非负/宽高为正、`target_size` 为正、`scale` 有限正数、`quality` 在 `[0, 1]`）；应用对齐时，`apply()` / `apply_side()` 把两路都缩放到 `target_size`；`apply_side()` 检查裁剪区是否越界，并**拒绝任何放大**（`target_size` 大于裁剪区，或参考帧小于 `target_size`，都抛 `ValueError`）。

**接线**：`apps/calibrate_stereo.py` 启动时只估计一次（实时相机路径最多读 30 帧 / 5 秒），把 `fov.to_dict()` 写进 `StereoRig.meta["fov"]`（落盘后是标定 JSON 顶层的 `fov`），之后每帧 `fov.apply()` 后再检测角点（`--save-pairs` 另存的也是归一化画面）；`record_stereo.py` 在写盘 / 预览前按路 `apply_side()` 归一化 —— 相机仍按 `--width/--height`（2560x1440）打开，但 writer、`session.json` 的 `left/right.width/height` 与 `calibration.image_size` 都用 `fov.target_size`（否则 `validate_session` 会报视频尺寸与 session 不一致）。只有 `meta["fov"]` **缺失 / `None`** 才按旧标定原生画面录，`{}` / `[]` / 非 mapping 一律视为记录损坏退出 2；启用对齐时 `crop_rect` 越出**采集帧**、`target_size` 与 `image_size` 不一致、`target_size` 大于裁剪区（旧版会放大的记录）或含奇数宽高，都在打开 writer 前退出 2。`verify_stereo.py` 只对实时相机输入应用 `fov.apply()`；视频 / 图片不二次裁剪，但会在**首对帧**调用 `estimate_fov_alignment()` 证明输入确实已归一化 —— 仍需要主动对齐（原生画面）、尺度不在 `0.9–1.1` 或纹理不足估不出来都退出 2，显式加 `--fov-input-normalized` 才跳过校验（不静默假定）。旧标定没有 `fov` 键（`rig.meta.get("fov")` 为 `None`）→ 调用方拿到 `None`，按原生画面工作；`session.rig["fov"]` 只在对齐生效（`active=True`）时写入。

**当前真机样例（模块级，不是完整标定 E2E）**：2026-10-24 两路 2560×1440 MSMF 帧，左路（较窄）为参考、右路实测裁剪 `(409,126,1717,986)`（只保留 45.9% 画面；另一次启动估计裁剪区 1685×979）、`scale≈0.651`；共同输出 = 裁剪区**向下取偶** -> **1716×986 / 1684×978**，不是 2560×1440。换机 / 换镜头 / 换分辨率后以启动估计为准，不要写死；旧版（放大到 2560）上的残余尺度 0.966 不能当修复后的验证结论，需用真机帧对重验。

### 13.4 `stereo/sessions.py`
```python
SCHEMA = "sphere-stereo-session"; VERSION = 1; SESSION_FILE = "session.json"
@dataclass CameraInfo:
    role, video, width, height, frame_count=0, fps_requested=0.0, fps_measured=0.0,
    backend="", identity="", name="", port=""
@dataclass CalibrationInfo:
    path, sha256="", image_size=[], baseline_mm=0.0, baseline_measured_mm=None,
    created="", verdict=""
@dataclass SyncInfo: method="none", offset=0, confidence=0.0, note=""
@dataclass Session:
    id, created, label="", notes="", left=None, right=None, calibration=None,
    stereo_mode="parallel", rig={},               # 2026-10-21；旧文件缺省 parallel；rig["fov"] 仅对齐生效时写入
    sync=SyncInfo(), codec="FFV1", container="mkv", lossless=True,   # container: mkv(默认)/avi；录制 --container 选，mp4 不支持
    environment={}, checks=[]
def create_session(root, *, label="", notes="") -> tuple[Path, Session]
def save_session(session, session_dir) -> Path
def load_session(path) -> tuple[Path, Session]              # 目录或 session.json
def validate_session(path) -> tuple[list[str], list[str]]   # (errors, warnings)，不抛
def list_sessions(root) -> list[Path];  def sha256_file(path) -> str
```
**契约**：一次采集 = `data/stereo/<id>/`（`session.json` + 两路视频 + `calib/` 快照 + `derived/`）。
`validate_session` 检查视频可打开 / 分辨率 / 帧数 / 标定 sha256 / 基线偏差
（物理 vs 算法 >10% 警告、>20% 错误），**不抛异常**，坏会话一次列全。
**基线**：`baseline_mm` 是算法结果（几何用它），`baseline_measured_mm` 只作交叉核对，不覆盖算法值。
**路径**：视频/标定路径默认相对会话目录；也接受绝对路径。
**容器**：`container` 记实际写盘容器，`mkv`（默认）/ `avi` 二选一（录制端 `apps/record_stereo.py --container` 指定）；
原始视频扩展名与文件名清单的唯一产地是 `sphere_tracker/video_files.py`（`RAW_VIDEO_EXTENSIONS`、`LEFT/RIGHT_VIDEO_NAMES`，顺序 mkv>mp4>avi）。

### 13.5 `stereo/trajectory.py` + `stereo/depth_fusion.py`
```python
def savgol_trajectory(t, xyz, *, window=11, polyorder=2) -> np.ndarray   # (N,3)
class Kalman3D:
    def __init__(self, process_var=1e-4, measure_var=1e-3)
    def update(self, t, z) -> np.ndarray | None       # 滤波后的 (3,)
    def predict(self, t) -> np.ndarray | None         # 不改状态
    @property velocity -> np.ndarray | None

def stereo_depth(disparity_px, fx_px, baseline_m) -> float
def mono_depth(radius_px, fx_px, diameter_m) -> float
def fuse_depth(z_stereo, sigma_stereo_m, z_mono, sigma_mono_m) -> tuple[float, float]
```
- `savgol_trajectory` 没装 scipy 时退化为滑动平均；`Kalman3D` 常速度模型。
- `fuse_depth` 用逆方差加权；任一路无效就退化为另一路，两路都无效返回 `(inf, inf)`。
- **当前接线**：`run_stereo_video.py --smooth` 已用 S-G；Kalman 与深度融合还是库，
  等镜面球检测稳定后再接主流程（不要在错检测上做平滑/预测）。

### 13.6 `stereo/visualization.py` —— 标准数据
```python
SEEDS_SCHEMA = "sphere-stereo-seeds"; TRAJECTORY_SCHEMA = "sphere-stereo-trajectory"
VISUALIZATION_SCHEMA = "sphere-stereo-visualization"; VERSION = 1
def write_seeds(session_dir, left, right, *, seed_frame=0, image_size=None,
                source="manual", note="") -> Path
def load_seeds(path) -> dict                         # 也接受会话目录
def read_trajectory_csv(path) -> list[dict]          # 补空列；短行不炸
def build_trajectory_payload(session_dir, *, csv_name="traj3d.csv") -> dict
def build_visualization_payload(trajectory) -> dict
def export_session_visualization(session_dir, *, csv_name="traj3d.csv") -> dict[str, Path]
```
- 输出：`derived/traj3d.json` + `derived/visualization.json`；
- `read_trajectory_csv` **按列名读、短行补空**：`matched=0` 的行历史上只有 14 列，
  用 `csv.DictReader` 会把 `track` 错当 `residual_px`（本轮踩过）；
- 坐标/单位/版本规则见 `docs/数据标准.md`；`run_stereo_video` 全片跑完会自动导出，
  部分帧试跑（`_firstN`）不覆盖正式 JSON。
### 13.7 `stereo/predict3d.py` —— 三维弹道外推 + 反弹检测（未接主流程）
```python
@dataclass(frozen=True) PredictConfig:
    g=(0.0,+9.81,0.0); horizons=(0.1,0.2,0.3,0.5); min_samples=5
    max_gap_s=0.25; residual_gate_m=0.05
@dataclass(frozen=True) BallisticState:
    p0; v0; g; rms_m; n; t_ref; dropped=0        # p0 是 t_ref（最后一个有效样本）时刻的位置
def fit_ballistic(times, points, cfg=None, reason=None) -> BallisticState | None
def predict(state: BallisticState | None, horizons) -> dict[float, np.ndarray]   # None -> {}
def detect_bounce(times, points, cfg=None, reason=None) -> int | None
```
- 模型 `p(t)=p0+v0·τ+0.5·g·τ²`，只解每轴 `(p0,v0)` 共 6 个未知数（线性最小二乘，**无迭代优化器**）；`g` 是**已知常向量** —— 三维里它不依赖焦距和深度，这正是三维比二维稳的原因。
- **`fit_ballistic` 什么时候返回 `None`**（原因写进 `reason`，从不硬给一个解）：`n < min_samples` / 时间跨度 < 1 ms / 设计矩阵条件数 > 1e8 / 残差 `rms_m > residual_gate_m`（窗口里多半混进了反弹）。
- **`detect_bounce` 的判据（两步缺一不可）**：先看整段残差，没超门限 → `None`（纯噪声仍是同一条抛物线，不许报反弹）；超标才逐切分点分段拟合，要求**两段各自合格**且明显优于整段（`worst ≤ 0.5·rms_all`），速度沿重力方向由同向变反向（`v·g` 由正变负）只作挑点的加分项。返回的是 `_prepare` 之后的数组下标（已排序、去重时与原始下标不同）。
- **坑**：① **`g` 的方向由坐标系定义**。本项目相机/世界系约定 **Y 向下**，默认 `(0,+9.81,0)`；如果数据是"Y 向上"的数学系，必须显式传 `(0,-9.81,0)`。相机俯拍/侧装或换到世界系时必须用 `PredictConfig(g=...)` 换成该系的真实重力方向（世界系用 `run_stereo_video.world_gravity_world()`），否则 0.3 s 外推能偏几十厘米。
  ② `predict` 的 horizon 相对 `t_ref`（最新观测），不是第一个样本 —— "现在"永远是最新观测。
  ③ 重复/乱序时间戳在 `_prepare` 里排序去重，丢弃数记在 `dropped`；`residual_gate_m=0.05` 是"毫米级残差"的门限，混进反弹就会被拒，**别为了让它出结果而调大**。

## 14. 依赖方向

> **图与规则在根目录 `AGENTS.md` §5（唯一权威），这里不复制。**
> 只补一条 API 视角的说明 —— **为什么 `camera_registry` 和 `sources` 同层**：
> 它只**惰性** import 一次 `sources`（`camera_registry.py:397` 的
> `from .sources import CameraSource`，在 `_default_opener` 内部），
> 其余只用标准库（`json`/`re`/`subprocess`/`sys`/`pathlib`）——
> 不探测相机时连 cv2 都不需要；它既不依赖也不被 `pipeline` 依赖。
> `stereo/` 则在模块顶层同时 import `sphere_tracker.camera_registry`、
> `sphere_tracker.sources`、`sphere_tracker.types`，方向仍然向上。
>
> 反向依赖（比如 `geometry.py` 去 import `pipeline`）会立刻把项目变成一团乱麻。
## 15. 核对记录与本文边界
**已用 `inspect.signature` 复核的 38 个签名**：`geometry` 的 `refine_circle_subpixel`/`fit_circle_robust`/`circle_radial_gradient_stats`/`CameraModel.__init__`/`estimate_focal_from_known_ball`；`camera_motion` 的 `GlobalMotionEstimator.__init__`/`.estimate`/`warp_to_current`；`detectors.base._score_candidate`；`track` 的 `KalmanFilter.__init__`/`Track.__init__`；`predict` 的 `predict_quadratic`/`TrajectoryPredictor.predict`；`sources` 的 `CameraSource.__init__`/`open_source`/`LatestFrameCamera.latest`；`viz` 的 `draw_prediction`/`draw_3d_panel`/`hud_lines`；`pipeline` 的 `SpherePipeline.process`/`.render`；`calib` 的 `detect_board`/`detect_board_in_roi`/`pose_distance`/`AutoPairCollector.offer`/`calibrate_stereo`/`StereoRig.rectify_maps`/`StereoRig.triangulate`；`stereo` 的 `find_offset`/`brightness_series`/`match_epipolar`/`triangulate_one`/`StereoReconstructor.update`/`PairSource.__init__`/`from_args`/`resolve_backend`；以及 §13.3c 的 `estimate_fov_alignment`/`FovAlignment.from_dict`（2 个，2026-10-24 补）。

**边界（诚实标注）**：⓪ 2026-10-26 新增的 §11b `multi_target` / §12.6 `world_frame` / §13.7 `predict3d`
签名是**按源码抄写**的，**没有**跑 `inspect.signature` 复核（不在上面那 38 个之列）；
同日新增的 §12.7 `metric_check` / §12.8 `validation` / §12.9 `uncertainty` 三节同理。① 本文**不覆盖** `apps/`、`tests/`、启动器与 PPT 工具链的内部实现（只在 §13.2 提了一句 `apps/_common.py` 的参数约定）。
② 文中所有"实测数字"（300/93/12/40 ms、~0.1 s vs ~10 s、圆度 0.61~0.63…）**引自源码注释**，我**没有**重跑那 6 个 `tests/selftest_*.py`。
③ ~~§10 的 `viz.scale != 1` 绘制不一致~~ —— **已修，不再是边界**（2026-10-14 的 `feat(ui)` 那轮把 scale 接进每个 `draw_*`，见 §10 坑④）。
④ **未确认**：`MotionDetector._bg_stale` 在长时间手持下的实际重学效果、`optimize_mode()` 在非本机摄像头上的耗时 —— 只有代码注释作依据。
⑤ 本文按三个包**当前源码**写成；若与源码不符，**以源码为准**并顺手改这里。

> **本文自身也是漂移源**（2026-10-14 复核）：§1 坑②与 §10 坑④ 两条"定义但无人读取 / 疑似不一致"在代码修好之后**继续留在正文里**，
> 而 `run.bat standards` 查的是**路径引用与计数**、`selftest_doc_facts` 查的是**测试清单** ——
> **两者都不检查"这段描述还成不成立"**，所以它一路绿灯。教训：**"读代码得出的结论"没有回归测试守着，就会随相邻代码一起过期**；
> 写这类判断时必须同时写出**判据落在哪个测试里**，否则它迟早变成误导下一个人的正文。
