# 库 API 参考（`sphere_tracker` / `calib` / `stereo`）
> **这份文档管什么**：三个库包的**模块职责 + 真实签名 + 返回值 + 调用顺序与前置条件**。
> **什么时候看**：要**调用**某个函数/类，想确认"默认值多少、返回什么、能不能直接喂下一层"时。
> 想知道"东西放哪"看 `文件结构.md`、"为什么这么设计、踩过哪些坑"看 `交接文档.md`；**入口文档是 `docs/项目总纲.md`**。
> 签名逐个 `read` 源码后用 `inspect.signature` 复核（清单见 §15）；只写代码里**真的存在**的东西，没确认的一律标"未确认"。
## 0. 全局约定（先读这一节）
| 约定     | 内容                                                                                     |
| -------- | ---------------------------------------------------------------------------------------- |
| 像素坐标 | `(x, y)` = `(列, 行)`，原点左上，OpenCV 口径                                               |
| 单目三维 | **相机坐标系**：Z 向前、米（`CameraModel.depth_from_radius()` + `backproject()`）          |
| 双目三维 | **校正后左相机系**、米（`T` 也是米）；要和原始左相机系的真值比，先乘 `rig.R1`               |
| 时间 `t` | 秒。视频/图片源 = `帧号/fps`；实时相机 = `perf_counter()`（绝对值无意义，同进程内单调）     |
| 失败约定 | 绝大多数"没结果"返回 `None`/`[]` **不抛**；只有校验类错误抛异常（各节标注）                 |
| 取帧责任 | **`pipeline` 不取帧**：`sources`（单目）/`stereo.PairSource`（双目）由 app 驱动，再把帧喂进库 |

**范围与前提**：本文只覆盖三个库包（`apps/`、`tests/`、启动器、PPT 工具链不在范围内）；双目精度数字**全部来自合成真值，真机标定从未跑过**（`交接文档.md` §10.4(b)），文中"实测数字"引自源码注释或既有文档，我**没有**重跑那 6 个 selftest。
## 1. `sphere_tracker/config.py` —— 全部可调参数
```python
class ColorConfig:  @staticmethod preset(name: str) -> "ColorConfig"
    # white_ball / orange_ball / green_ball / blue_ball / red_ball / yellow_ball
class AppConfig:  detector / tracker / camera / predict / viz
    def to_dict(self) -> dict;  def save(self, path: str | Path) -> Path      # mkdir -p 后写 JSON
def load_config(path: str | Path | None = None) -> AppConfig      # None 或文件不存在 -> 全默认
def save_config(cfg: AppConfig, path: str | Path) -> Path          # == cfg.save(path)
```
| 组               | 关键默认值                                                                                                                                                                                                                                                                                                        |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DetectorConfig` | `method="fusion"`、`fusion_members=["color","hough","motion"]`、`min_radius=8.0`、`max_radius=200.0`、`min_circularity=0.55`、`min_fill_ratio=0.72`、`min_radial_consistency=0.90`、`subpixel_refine=True`、`subpixel_rays=72`、`motion_compensation="auto"`、`camera_motion_threshold_px=0.8`、`motion_work_width=320`、`agree_px=6.0`、`agree_ratio=0.45`、`yolo_weights="models/yolo11n.pt"`、`sam_model="models/sam2.1_t.pt"`、`yolo_classes=[32]` |
| `TrackerConfig`  | `order=2`、`process_noise=1.5e2`、`measurement_noise=2.0`、`gate_chi2=9.21`、`maneuver_min_px_s=250.0`、`maneuver_ratio=1.6`、`max_coast_frames=12`、`init_hits=2`、`trail_length=240`、`trail_source="measurement"`                                                                                                                                                                                                  |
| `CameraConfig`   | `fx=fy=0.0`（未标定）、`cx=cy=-1.0`（用图像中心）、`hfov_deg=60.0`、`ball_diameter_m=0.04`、`enable_3d=True`                                                                                                                                                                                                       |
| `PredictConfig`  | `method="quadratic"`、`horizon_s=0.6`、`steps=30`、`min_points=6`、`use_gravity=False`、`use_pixel_gravity=True`、`window_s=0.45`、`min_window_s=0.12`、`residual_tol=2.5`、`smooth_window=1`                                                                                                                          |
| `VizConfig`      | `trail_enabled/show_circle/show_prediction/show_hud=True`、`show_mask/show_axes_3d=False`、`scale=1.0`                                                                                                                                                                                                            |

**坑**：① 默认颜色预设是**白色/高亮球**（`[0,0,175]`–`[179,80,255]`），**不是橙球** —— 拍橙球必须 `preset("orange_ball")` 或跑 `apps/tools/calibrate_color.py`，否则"检测不到"会被误判成算法问题。
② `DetectorConfig.min_score` 与 `TrackerConfig.max_q_scale` **定义了但全仓库无人读取**（fusion 的孤证门限硬编码 `0.55`、`KalmanFilter.q_max` 硬编码 `q_base*64`）—— 改它们**没有任何效果**；`_merge_dataclass` 还**忽略未知字段**，所以**键名拼错会静默失效**（不报错）。
③ `red_ball` 预设自带 `use_second_range=True`（红色跨 0 度），别只抄 `hsv_lower/upper`。
## 2. `sphere_tracker/types.py` —— 数据容器（字段要记准）
**`Detection`**（单帧检测，像素）：`x, y, radius`（亚像素）；`score`（0–1）；`method`（`"color"`/`"hough"`/`"motion"`/`"fusion:color"`/`"yolo"`/`"yolo+color"`/`"yolo_sam"`）；`parts: dict[str,float]`（各子线索原始分，fusion 会塞 `votes`）；`contour: (N,2)|None`（hough/yolo 不给）；`mask`（**整幅**掩膜，**同帧多个 Detection 共享同一数组**）；`extra: dict`（YOLO 放 `"box"`）。属性 `center -> (x,y)`、`diameter -> 2*radius`。

**`TrackState`**（跟踪器输出）：

| 字段                         | 含义                                                             |
| ---------------------------- | ---------------------------------------------------------------- |
| `track_id, t`                | 轨迹号 / 时刻（秒）                                               |
| `x, y, vx, vy, ax, ay`       | 位置(px)、速度(px/s)、加速度(px/s²)，**全部来自卡尔曼滤波**        |
| `radius`                     | 半径（指数滑动平均，α=0.5 硬编码）                                |
| `z, vz`                      | 单目测深深度(米)、深度速度(m/s) —— **不是 track.py 填的**（见坑①） |
| `confirmed / coasting`       | 命中 `init_hits` 次后 True / 本帧是外推出来的（没有观测）          |
| `missed / age / hits`        | 连续丢失帧数 / 存活帧数 / 累计命中数                               |
| `maneuvers, last_maneuver_t` | 检测到几次速度突变 / 最近一次突变时刻（`None`=从未）               |

属性 `speed_px -> hypot(vx, vy)`。

| 类型           | 字段                                                                                                                              |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `FrameContext` | `t=0.0, gray=None, motion=None, camera_moving=False`（`motion` 实际是 `MotionEstimate`，标注为 `Any`）                             |
| `TrailPoint`   | `t, x, y, z=None, radius=0.0, score=0.0`                                                                                           |
| `FrameResult`  | `t, frame_index, detection, track, trail, prediction, prediction_3d, position_3d, fps, detect_ms, camera_motion, camera_moving, camera_steady` |

`FrameResult` 的三个三维字段不同：`prediction` = **(N,2) 像素**（3D 预测也会投影回像素）、`prediction_3d` = **(N,3) 米**、`position_3d` = **(3,) 米**；`trail` 坐标始终是像素。

**坑**：① **`TrackState.z/vz` 不是 `track.py` 填的**：`Track.state()` 永远给 `z=None, vz=0.0`，它们由 `SpherePipeline.process()` 在三维那一步回填 —— 单独用 `Track` 拿不到 z。
② `TrailPoint.score == 0.0` 是**"这一点是 coast 外推的"标志**，`predict.py` 与 `viz.draw_trail` 都靠它；③ 检测器要用 `getattr(ctx, "gray", None)` 这类防御写法取 `FrameContext` 字段（`ctx` 可能是 `None`），`FrameResult.trail` 是 `list(track.trail)` 浅拷贝、点对象共享。
## 3. `sphere_tracker/geometry.py` —— 圆拟合、亚像素、相机模型
```python
def fit_circle_taubin(points) -> tuple[float, float, float]         # 少于 3 点抛 ValueError
def fit_circle_robust(points, iterations=3, trim_sigma=2.0) -> tuple[float, float, float]
def refine_circle_subpixel(gray, cx, cy, radius, n_rays=72, search=2.5, step=0.25, min_gradient=15.0) -> tuple[float, float, float]
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

**坑**：① **`refine_circle_subpixel` 失败时不抛异常，而是原样返回输入**（`radius<3.0`、采样起点越界、有效射线 `< max(8, n_rays//3)`）—— "返回值和输入完全一样"是**细化失败**的信号；`fit_circle_robust` 点数 `<5` 时**退回** `fit_circle_taubin`。
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
    # 前者只认 color/hough/motion（其它返回 None）；后者未知名字抛 ValueError，"sam"/"sam2" 等价 "yolo_sam"
class ColorBlobDetector(BaseDetector):  # "color"；def build_mask(frame) -> np.ndarray；给 contour+mask
class HoughDetector(BaseDetector):  # "hough"；无 contour / 无 mask
class MotionDetector(BaseDetector):  # "motion"；last_mode: "bg" | "compensated"
class FusionDetector(BaseDetector):  # "fusion"；返回 0 或 1 个，method="fusion:<子名>"
class YoloDetector / YoloSamDetector  # "yolo" / "yolo_sam"；def refine_circle_in_box(frame, det, cfg) -> Detection | None
```
`_score_candidate` 的硬规则（所有检测器共用）：半径越界、`radial>0 且 <min_radial_consistency`、`fill>0 且 <min_fill_ratio` → **直接 0**；半径贴边（`<1.3*min` 或 `>0.8*max`）整分 ×0.8；只对"传进来 >0"的指标做加权平均。区分圆与方块主要靠 `radial`（方块顶点比边中点远 41%）。

| 检测器 | 需要 `ctx` 的            | 返回 contour/mask | 额外前置                                                      |
| ------ | ------------------------ | ----------------- | ------------------------------------------------------------- |
| color  | `gray`                   | 有 / 有（整幅）    | 只对**前 3 个**候选做亚像素细化                                |
| hough  | `gray`                   | 无 / 无           | 两道梯度门：`hough_min_support`、`hough_min_sign_consistency`   |
| motion | `motion`+`camera_moving` | 有 / 有（`fg`）   | 需 `_prev_frame` 与 `motion.matrix` 才走补偿分支               |
| fusion | 透传                     | 取胜出者           | 子检测器由 `cfg.fusion_members` 决定                            |
| neural | 不用                     | SAM 有 / YOLO 无   | 需 `ultralytics`；缺依赖时**抛 RuntimeError**                  |

**坑**：① `detect()` 返回**按 score 降序**，`pipeline` 只取 `[0]` 当"本帧检测"、整表交给 tracker —— 排序是接口契约。
② `build_single_detector` 只认三个名字：把 `"yolo"` 写进 `fusion_members` 会被**静默忽略**（不报错，少一路票）。
③ fusion：每票 +0.12；**只有一票且总分 <0.55 返回空**（硬编码，见 §1 坑②）；docstring 里的 `vote_single_min` **并不存在**；子检测器抛 `cv2.error` 被吞掉当"没检测到"。
④ motion：**换源必须 `reset()`**（重建 MOG2）；`_frames<5` 恒返回 `[]`；从"相机在动"回到"静止"时会用 `learningRate=1.0` 连喂 3 帧重建背景（这 3 帧的检测不可信）。
⑤ AI 后端的失败是**运行时**的：`build_detector` 不报错，`detect()` 抛 `RuntimeError`，`SpherePipeline` 捕获它、只打印一次 `[检测器不可用]`，之后按"没检测到"继续跑（**不中断视频**）；SAM 只处理前 3 个种子框。`sphere_tracker/detect.py` 只是**兼容垫片**，新代码从 `sphere_tracker.detectors` 导入。
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
    def latest(self, last_seq: int = -1) -> tuple[int, np.ndarray | None]
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
CANDIDATE_MODES: tuple[tuple[int, int], ...]     # 自动探测用的候选分辨率
```
**调用顺序 / 前置条件**：app 里的标准用法是 `src = open_source(...)` → 循环 `ok, frame, t = src.read()` → `pipe.process(frame, t)` → 结束 `src.release()`；**`pipeline` 不 import `sources`**（取帧与编排分层）。`open_source` 分派：`path` 是目录 → `ImageSource`（按扩展名过滤 + 文件名排序）；`path` 是图片扩展名 → `ImageSource([单张])`；否则 → `FileSource`；都不给 → `CameraSource(index=camera or 0)`。要"永远处理最新画面"就用 `LatestFrameCamera(CameraSource(...))`（双目分支就是这么用的），循环里用 `latest()`。

**坑**：① **`latest(last_seq=-1)` 的 `last_seq` 没被使用**：直接返回当前 `(seq, frame)`，"有没有新帧"要自己比 `seq`。
② `CameraSource.__init__` **很重**：依次试后端（DSHOW→MSMF→ANY）与分辨率（用户值→1280x720→640x480→默认），每个组合连读至多 12 帧做"全黑常量流"检查（判据 `frame.std() > 1.0`）；`optimize=True` 还会跑一次 `optimize_mode()`（最坏 10–40 秒，只跑一次，缓存到 `config/camera_modes.json`）。全失败抛 `RuntimeError`（消息里带"占用/红外/遮挡/USB 带宽"四个可能原因）。
③ `info.fps` **尽量是实测值**：`CAP_PROP_FPS` 常返回"你请求的值"而不是真实帧率（实测请求 60、实际 29.8）；`fps_measured` 告诉你这个值是不是实测 —— 写录像帧率要用它；`resolution_fallback=True` = 请求的分辨率不支持、已降级（USB 摄像头常表现为"打开成功但每帧全黑"）。
④ 时间戳语义不同：`CameraSource` 用 `perf_counter()`，`FileSource`/`ImageSource` 用 `帧号/fps`，跨源不可比。
⑤ `open_source(path=...)` 时 `camera=` 被忽略；`**kwargs` 只在 camera/image 分支生效，走 `FileSource` 时**静默丢弃**；读尽后返回 `(False, None, t)`（`ImageSource` 读不出图片时同样 `ok=False`）。
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
AUTO_TOKEN = "auto";  DEFAULT_SETTINGS = {...}   # prefer_external / min_width / min_height / min_fps / avoid_names / max_index
```
`resolve_camera_arg` 规则：`None`/`""` → `None`；`int` → 原样；`"0"` → `0`；`"port:xx"`/`"id:xx"`/`"name:xx"` → 匹配设备；**其它字符串当角色名**（查 `config/cameras.json`）。

**`Registry.ignored` = 「无关」清单**：用户明确标记"这台不要用"（笔记本内置 / 红外）。
与 `roles` 同样记**身份**而非编号，所以重插 USB 后仍有效。语义：
`rank_candidates()` 给它 `-1000` 分（比"内置/红外"的 `-200` 还重），`select_cameras()`
把它排到最后、只在其它相机都不够用时才动用**并打印警告**；`mark_ignored()` 会把它
从 `roles` 里摘掉（一台相机不能既是左相机又是「无关」）。
`format_device_table(..., reg=reg)` 会打出 `[无关]` 标记。

**为什么 `_spec_matches_device()` 不按名字匹配**（与 `_match_entry()` 不同）：
两台同型号相机的名字**一模一样**（本项目两台 aoni 都叫 `aoni webcam A25`），
按名字匹配会把"只排除这一台"变成"两台一起排除"。所以「无关」只认 identity → port，
拿不到身份时才退到编号（且只认 `verified=False` 的记录）。
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
④ 疑似不一致（**读代码发现，未实机确认**）：`SpherePipeline.render()` 在 `viz.scale != 1` 时**只把预测点乘了 `scale`**，拖尾/圆/轮廓/掩膜仍按原分辨率坐标画 ⇒ 缩放显示时几何元素错位；默认 `scale=1.0`，日常看不到。
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
```python
@dataclass CaptureConfig:
    target_pairs=25; min_pairs=10; min_sharpness=55.0; min_area_ratio=0.012; max_area_ratio=0.90; min_margin_px=10.0
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
    def disparity_sign(self) -> int             # 右相机在右 -> -1；装反 -> +1；未知 -> 0
    def save(self, path) -> tuple[Path, Path]   # 同时写 .npz + .json
    @classmethod load(cls, path) -> "StereoRig";  @classmethod from_result(cls, res, meta=None) -> "StereoRig"   # not res.ok 抛 ValueError
def load_stereo_rig(path) -> StereoRig;  def save_stereo_rig(rig, path)
```
**坐标系与单位（最要紧）**：三维点在**校正后的左相机坐标系**，原点在左相机光心、Z 向前、**单位米**（因为 `BoardSpec.object_points()` 是米）。`T` 是"右光心在左相机系里的位置"（`X_right = R @ X_left + T`）。

**坑**：① `rectify_maps` 的映射表**依赖 `image_size`**：换分辨率要重新 `rectify_maps()` 或重标定；`rectify_image(img, side)` 的 `side` 只能是 `"left"`/`"right"`，给 `"both"` 会去解包元组直接报错。
② `disparity_sign()` **反直觉**：右相机在右 → **`-1`**（`stereoRectify` 的 `P2` 第四列实测 `= -f*T[0]`，同一物点在右图更靠右，`x1-x2` 为负）；不用自己记 —— `StereoReconstructor` 会自动取。
③ **左右装反时标定指标一样漂亮**：`baseline_side()=="left"` 时 `describe()` 会打印警告；三角测量**仍然正确**（P1/P2 自洽），但任何"假设视差为正"的匹配会 0% 配对，报错只有"配不上"。
④ `save()` 同时写 `.npz`（矩阵）和 `.json`（质量报告+元信息）；`load()` 只要求 `.npz` 存在，`.json` 缺失时 `quality/meta` 为空字典（不抛）。
## 13. `stereo/` —— 双目逐帧
### 13.1 `stereo/sync.py`
```python
@dataclass SyncConfig:  max_offset=180; min_overlap=30; smooth=1
@dataclass SyncResult:
    offset: int          # left[i] 对应 right[i + offset]
    confidence: float    # 归一化互相关系数峰值
    peak_count=0; note="";  def __str__(self) -> str;  @property usable -> bool      # confidence >= 0.35 且 peak_count >= 1
def brightness_series(frames, smooth=1) -> np.ndarray
def find_offset(series_left, series_right, cfg: SyncConfig | None = None) -> SyncResult
def apply_offset(offset: int) -> str      # 把偏移翻译成"取帧"的人话，用来核对正负号
```
**坑**：① `find_offset` 在**亮度差分**上做归一化互相关（突变被放大）；序列短于 3 帧返回 `SyncResult(0, 0.0, 0, "序列太短")`；若所有候选偏移都因 `min_overlap` 被跳过，返回的也是 `offset=0, confidence=0.0` —— **别把 `offset=0` 当成"对齐了"**，看 `usable`/`confidence`。
② `brightness_series` **会跳过 `None` 帧**，返回序列的下标**不一定等于帧号**，别拿它当帧索引；搜索范围受 `max_offset=180` 限制，偏移正负语义统一是 `left[i] ↔ right[i+offset]`。
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
@dataclass StereoReconstructor:
    rig: StereoRig;  cfg: MatchConfig = MatchConfig();  trajectory: list[StereoPoint] = []
    n_frames=0; n_matched=0; n_no_left=0; n_no_right=0; n_rejected=0
    def __post_init__(self) -> None    # 自动取 rig.disparity_sign()（仅当 cfg.disparity_sign == 0）
    def update(self, t, left: Detection | None, right_dets: list[Detection]) -> StereoPoint | None
    @property match_rate -> float      # n_matched / n_frames
    def as_array(self) -> np.ndarray   # (N,4) = [t, X, Y, Z]；无点时 (0,4) 零数组
    def summary(self) -> str
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
**已用 `inspect.signature` 复核的 36 个签名**：`geometry` 的 `refine_circle_subpixel`/`fit_circle_robust`/`circle_radial_gradient_stats`/`CameraModel.__init__`/`estimate_focal_from_known_ball`；`camera_motion` 的 `GlobalMotionEstimator.__init__`/`.estimate`/`warp_to_current`；`detectors.base._score_candidate`；`track` 的 `KalmanFilter.__init__`/`Track.__init__`；`predict` 的 `predict_quadratic`/`TrajectoryPredictor.predict`；`sources` 的 `CameraSource.__init__`/`open_source`/`LatestFrameCamera.latest`；`viz` 的 `draw_prediction`/`draw_3d_panel`/`hud_lines`；`pipeline` 的 `SpherePipeline.process`/`.render`；`calib` 的 `detect_board`/`detect_board_in_roi`/`pose_distance`/`AutoPairCollector.offer`/`calibrate_stereo`/`StereoRig.rectify_maps`/`StereoRig.triangulate`；`stereo` 的 `find_offset`/`brightness_series`/`match_epipolar`/`triangulate_one`/`StereoReconstructor.update`/`PairSource.__init__`/`from_args`/`resolve_backend`。

**边界（诚实标注）**：① 本文**不覆盖** `apps/`、`tests/`、启动器与 PPT 工具链的内部实现（只在 §13.2 提了一句 `apps/_common.py` 的参数约定）。
② 文中所有"实测数字"（300/93/12/40 ms、~0.1 s vs ~10 s、圆度 0.61~0.63…）**引自源码注释**，我**没有**重跑那 6 个 `tests/selftest_*.py`。
③ **未实测确认**：§10 的 `viz.scale != 1` 绘制不一致（读 `pipeline.render()` 发现，没跑起来验证；默认 `scale=1.0` 时不受影响）。
④ **未确认**：`MotionDetector._bg_stale` 在长时间手持下的实际重学效果、`optimize_mode()` 在非本机摄像头上的耗时 —— 只有代码注释作依据。
⑤ 本文按三个包**当前源码**写成；若与源码不符，**以源码为准**并顺手改这里。
