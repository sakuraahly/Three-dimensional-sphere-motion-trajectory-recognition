# 球体运动轨迹识别（检测 → 跟踪 → 三维 → 预测）

用摄像头实时识别球的运动轨迹、标记已有轨迹、预测未来轨迹。
纯 OpenCV 传统方法即可达到亚像素精度与 100+ FPS，**不装任何深度学习框架也能跑**；
同时预留了 YOLO / SAM 的 AI 后端，一行配置就能切换对比。

---

## 一、先说你原来那套为什么不行

原来的做法是：

```python
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)
```

问题不在参数没调好，而在**任务本身不匹配**：

1. **Canny 给的是"所有边缘"，不是"球"**。
   桌子边缘、阴影、反光、背景花纹全在里面，你拿到的是一堆像素点，
   而不是一个"球心 + 半径"。轨迹识别需要的是后者。
2. **没有几何约束**。就算从边缘里找到了圆，也没有任何机制保证它是球而不是别的圆。
3. **没有时间维度的连续性**。单帧噪声（1-2 px）直接差分求速度会被放大成
   几十上百像素/秒的乱跳，轨迹画出来是锯齿，更不可能预测。
4. **边缘位置精度不够**。二值化的边缘有 0.3-0.5 px 偏差，而单目测深的深度
   `Z = f·R/r` 完全由像素半径 r 决定 —— 半径差 1%，深度就差 1%。

顺带修掉了原脚本里几个真实的 bug（见文末"已修复的问题"）。

---

## 二、推荐的流水线（本项目已实现）

```
                ┌──────────── 检测（每帧） ────────────┐
   摄像头帧 ──▶ │ 颜色分割 / 霍夫圆 / 背景差分 → 投票融合 │──▶ 球心(x,y) + 半径r + 置信度
                └──────────────────────────────────────┘
                                   │
                     ┌─────────────┴──────────────┐
                     ▼                            ▼
        亚像素边缘细化（梯度峰值 + 圆拟合）    半径 → 深度 Z = f·R/r
                     │                            │
                     ▼                            ▼
            卡尔曼滤波跟踪（含机动检测）      三维坐标 (X, Y, Z)
                     │                            │
                     └─────────────┬──────────────┘
                                   ▼
              轨迹预测：自适应窗口二次拟合 + 已知重力
                                   ▼
                    可视化：拖尾 / 圆 / 预测虚线 / HUD / 三维视图
```

### 每一步的关键设计

**1. 检测：不要只用一个特征**

| 后端 | 原理 | 适用 | 本项目实测速度 |
|---|---|---|---|
| `color` | HSV 阈值 + 形态学 + 轮廓圆度筛选 | 球与背景颜色可分（**首选**） | 120-180 FPS |
| `hough` | 霍夫圆 + 梯度方向/符号双向校验 | 球与背景同色，但边缘清晰 | 30-45 FPS |
| `motion` | MOG2 背景建模差分 | 摄像头固定不动 | 240 FPS |
| `fusion` | 上面几种投票融合（**默认**） | 通用，最稳 | 27-35 FPS |
| `yolo` / `yolo_sam` | AI 检测 / 分割（可选依赖） | 复杂背景、遮挡、多球 | 见第六节 |

**筛选"是不是球"靠四个几何指标**，缺一不可（`sphere_tracker/detect.py`）：

- **圆度** `4πA/P²` —— 排除锯齿状、细长目标
- **填充率** `A / (π·r_外接²)` —— 方块只能填满外接圆的 61%，球体能填满 90%+
- **径向一致性** `1 - std(到圆心距离)/r` —— **最有判别力的一项**，方块约 0.85，圆约 0.98
- **梯度符号一致性** —— 真球体边缘的梯度方向要么全朝里要么全朝外，纹理凑出来的假圆正负混杂

> 实测：一个和球**同色**的白色方块 + 一个和球**同形**的橙色圆作为干扰物，
> 四项指标必须同时生效才能两个都排除掉。这是自测场景里专门埋的陷阱。

**2. 亚像素边缘细化（精度提升 4 倍，很关键）**

阈值分割得到的圆边界有 0.3-0.5 px 误差。本项目从粗略圆心向 72 个方向发射射线，
在梯度幅值图上找峰值并用抛物线插值求亚像素位置，得到 72 个精确边缘点后重新拟合圆：

| | 半径 RMSE | 深度误差 | 三维 RMSE |
|---|---|---|---|
| 只用阈值轮廓 | 0.35-0.5 px | ~1.4% | 67-100 mm |
| **+ 亚像素细化** | **0.06-0.16 px** | **0.3-0.6%** | **15-42 mm** |

**3. 跟踪：卡尔曼滤波 + 机动检测**

匀速/匀加速卡尔曼滤波给出平滑的位置、速度、加速度。真正难的是**球落地反弹**：

- 反弹瞬间速度方向反转，滤波器还在用老速度外推，下一帧就偏出几十像素；
- 靠"加大过程噪声"救不回来 —— 要大到 `q≈1e11` 才跟得上，那样正常飞行又会抖成筛子；
- **正确做法**：用最近两次观测算出实测速度，和滤波器速度一比，差得太多就判定为机动，
  **直接把状态重置到观测量上**（`KalmanFilter.reinitialize`）。

另外还加了三道闸门：马氏距离门控（剔野点）、**半径一致性门控**（防止跟丢后跳到
画面里另一个尺寸不同的物体上）、抢救关联（门控失败但距离不远时仍归给原轨迹，避免轨迹被打断）。

**4. 预测：绝不能用整条轨迹去拟合**

球一旦反弹或被击打，运动规律就换了，用几秒前的数据拟合出的"未来"必然是错的。
本项目用的是**最长有效后缀搜索**：

> 从"尽量用满历史"（默认 0.45 s）开始，一段段丢掉最老的点，
> 取第一个拟合残差合格的后缀。等价于在问："以当前时刻结尾、且符合同一条抛物线的
> 最长数据段是哪一段？" —— 答案就是"上一次运动突变之后的那一段"。

自由飞行时第一段就合格（用满窗口，最抗噪）；刚反弹完时自动退化成最短的干净段。

还有一个单目相机的关键技巧：**重力在像素坐标下也是已知量**。

```
g_px = fy · g / Z        （Z 由球的像素半径反推：Z = f·R/r）
```

把二次项固定成 `g_px`，未知数从 3 个降到 2 个，刚反弹完只剩两三个点时也能预测准。
如果标定过相机，还可以切到三维米制坐标用真实 g 做预测（透视效应自动被正确处理）。

---

## 三、实测结果（合成数据自测，真值完全已知）

场景：960×540 @ 60 fps，球直径 22 cm，真实三维斜抛 + 落地反弹（恢复系数 0.85）
+ 前后左右碰壁 + 深度变化 3.7-4.4 m，透视投影，含同色方块 + 同形橙色圆两个陷阱干扰物。

```
python apps/selftest_synthetic.py
```

| 场景 | 方法 | 检出率 | 检测RMSE | 跟踪中位 | 半径RMSE | 三维RMSE | 深度误差 | 0.1s预测 | 0.2s预测 | 0.3s预测 | 速度 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clean | **color** | 100% | 0.54 px | 0.65 px | 0.06 px | 15.1 mm | 0.31% | 0.81 px | 1.20 px | 1.98 px | ~160 fps |
| clean | fusion | 100% | 0.54 px | 0.65 px | 0.06 px | 15.1 mm | 0.31% | 0.81 px | 1.20 px | 1.98 px | ~29 fps |
| clean | motion | 85% | 0.54 px | 0.70 px | 0.06 px | 13.5 mm | 0.28% | 0.81 px | 1.10 px | 1.60 px | ~238 fps |
| clean | hough | 100% | 41 px | 0.87 px | 2.79 px | 345 mm | 3.53% | 8.67 px | 9.54 px | 15.4 px | ~37 fps |
| realistic | **color** | 100% | 0.87 px | 0.68 px | 0.16 px | 42.1 mm | 0.58% | 1.33 px | 2.49 px | 4.43 px | ~160 fps |
| realistic | fusion | 100% | 0.87 px | 0.68 px | 0.16 px | 42.1 mm | 0.58% | 1.33 px | 2.49 px | 4.43 px | ~29 fps |

（速度是 960×540 下单线程纯 CPU 的处理帧率，随机器负载浮动；分辨率提高会等比例变慢。）

`clean` = 无噪声无模糊（检验几何链路是否正确）；
`realistic` = 逐帧噪声 + 按曝光时间积分的运动模糊（检验鲁棒性）。

**怎么读这张表：**

- `color` 是本项目的主力，精度和速度都最好 —— 只要球的颜色能和背景分开，就用它。
- `fusion` 精度和 color 持平但更稳（颜色失效时有别的后盾），代价是慢 4-5 倍。
  实时性要求高就用 color，场景复杂就用 fusion。
- `hough` 的中位误差只有 0.87 px，但 RMSE 高达 41 px：**少数帧会跳到别的圆上**。
  它只适合"球清晰、画面干净"的场合，别单独用在关键任务上。
- `motion` 检出率 85% 是因为球开始时会飞出画面上沿 —— 半个球在画面外时轮廓被裁掉，
  圆拟合必然失准，检测器选择不输出才是对的。球完全在画面内的帧里它是 100%。
- **反弹后 0.15 s 内的预测误差会大一个量级**（约 45-55 px @0.2s），这是物理限制：
  速度突变必须靠两次观测才能测出来，那之前无从得知球已经反向。
  约 3-4 帧（0.05-0.07 s）后恢复正常。

---

## 四、快速开始

### 环境

已有 conda 环境 `myenv`（Python 3.14.7 + opencv-contrib-python 5.0.0.93 + numpy 2.5.3）即可，**无需额外安装**。

```powershell
conda activate myenv
cd D:\MY_CODING_PROGRAM\Three-dimensional-sphere-motion-trajectory-recognition
```

> 注意：直接敲 `python` 可能解析到 msys64 的 Python 而不是 myenv。
> 保险写法：`C:\Users\<USER>\miniforge3\envs\myenv\python.exe apps\run_realtime.py`

### 跑起来

```powershell
# 1) 先自测，确认代码在你机器上正常（不需要摄像头）
python apps/selftest_synthetic.py

# 2) 标定颜色阈值（第一次用必做，1 分钟）
python apps/calibrate_color.py --camera 0
#    拖动滑块直到球是白色、背景是黑色；按 1-6 套用预设；按 f 冻结画面；按 s 保存

# 3) 实时运行
python apps/run_realtime.py                       # 默认摄像头 + 融合检测
python apps/run_realtime.py --method color --preset white_ball   # 更快
python apps/run_realtime.py --camera 1 --width 1920 --height 1080 --fps 60

# 4) 处理已有视频 / 图片
python apps/run_video.py material_videos/ball.mp4 --out outputs/videos/annotated.mp4 --csv outputs/traj.csv
```

**实时窗口的快捷键**

| 键 | 功能 |
|---|---|
| `q` / `ESC` | 退出 |
| `r` | 重置轨迹（球飞出去后重新开始） |
| `c` | 切换检测方法（fusion → color → hough → motion → yolo → yolo_sam） |
| `m` | 显示/隐藏分割掩膜小窗（调阈值必备） |
| `p` | 开/关轨迹预测 |
| `3` | 开/关三维视图小面板（俯视 X-Z + 侧视 Y-Z） |
| `s` | 保存快照 + 当前配置 |
| `v` | 开始/停止录制 |
| `空格` | 暂停 |

### 参数速查（`apps/run_realtime.py`）

| 参数 | 说明 |
|---|---|
| `--method` | `color` / `hough` / `motion` / `fusion` / `yolo` / `yolo_sam` |
| `--preset` | `white_ball` / `orange_ball` / `green_ball` / `blue_ball` / `red_ball` / `yellow_ball` |
| `--ball-diameter` | **球的真实直径（米）**。三维测距必需，而且必须准 —— 填错多少，深度就错多少 |
| `--focal` | 相机像素焦距 fx。不填就按水平视场角估算（配置项 `camera.hfov_deg`，默认 60°） |
| `--horizon` | 预测未来多少秒（默认 0.6，建议 0.3-0.6） |
| `--predict` | `quadratic`（默认，推荐）/ `kalman` |
| `--gravity-fit` | 三维预测时用已知重力约束（更准，需要标定） |
| `--config` | 读取标定好的 JSON 配置 |

---

## 五、标定（决定三维精度）

单目相机靠"球在图像里多大"来推距离：`Z = f · R / r`。

1. **焦距 fx**（`apps/calibrate_focal.py`）
   把球放在**量好**的距离上（比如 1.000 m），跑：
   ```powershell
   python apps/calibrate_focal.py --camera 0 --distance 1.0 --ball-diameter 0.04
   ```
   等绿圈稳定后按 `s`，会自动算出 fx 并存进 `config/sphere_config.json`。
2. **球直径**：用卡尺量，填 `--ball-diameter`。这是三维的绝对尺度来源。
3. 之后运行时带上 `--config config/sphere_config.json`。

### 三维的固有局限（必须知道）

- 单目测深的误差直接来自半径误差：`ΔZ/Z = -Δr/r`。
  本项目把半径精度做到了 0.06-0.16 px，所以在 r≈22 px 时深度误差约 0.3-0.6%。
- **球被部分遮挡、或运动模糊严重时，半径会偏**，深度跟着偏。想要真正稳的三维，
  要么上双目/深度相机，要么保证球始终完整可见。
- 这个"三维"是相机坐标系下的伪三维（Z 由球的大小反推），
  不是 LiDAR 那种独立的深度测量。

---

## 六、AI 方案（可选，什么时候值得上）

**结论先说：这个任务上，传统方法在"球颜色可分 + 摄像头固定"的常见场景里
更快、更准、更好调；AI 的价值在于处理传统方法搞不定的情况。**

| 场景 | 建议 |
|---|---|
| 球颜色和背景能分开 | **传统 `color`/`fusion`**，别再折腾模型 |
| 背景杂乱、球会变色/反光、多球、要区分球的种类 | YOLO / YOLOE 零样本检测 |
| 需要像素级精确轮廓（遮挡、形变、深度） | SAM2 分割 |
| 想要"这一个球"而不是"任何球" | YOLOE 文本提示，或自己标 200-500 张微调 |

### 环境是能装上的（已实测确认）

你的机器：RTX 5060 Laptop 8 GB（sm_120 Blackwell）、驱动 577.05、
myenv Python 3.14.7。经过实测的可用组合：

```powershell
$PY = "C:\Users\<USER>\miniforge3\envs\myenv\python.exe"

# 1) PyTorch：必须走 cu128（cu129 缺 cp314 的 torchvision；cu130 需要驱动 >= 580，你的 577 不够）
& $PY -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128

# 2) ultralytics：用 --no-deps，否则它会装 opencv-python 和你已有的 opencv-contrib-python 打架
& $PY -m pip install ultralytics --no-deps
& $PY -m pip install pyyaml psutil polars timm lap

# 3) 验证
& $PY -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_capability())"
# 期望输出：True (12, 0)
```

### 装上之后怎么用

```powershell
python apps/run_realtime.py --method yolo        # 零样本检测（COCO 的 sports ball 类）
python apps/run_realtime.py --method yolo_sam    # YOLO 给框 + SAM 分割出像素级边缘
python apps/selftest_synthetic.py --check-ai     # 检查依赖和 GPU 是否就绪
```

代码里已经写好这两个后端（`sphere_tracker/detect.py` 的 `YoloDetector` / `YoloSamDetector`），
未安装时会打印清晰的安装提示而不是崩掉。相关参数在 `DetectorConfig`：
`yolo_weights`（默认 `yolo11n.pt`，可换 `yolo26n-seg.pt`）、`yolo_conf`、`yolo_imgsz`、
`yolo_classes`（默认 `[32]` = sports ball）、`sam_model`（默认 `sam2.1_t.pt`）。

### 已知的坑（都是实测过的）

- **COCO 的 "sports ball" 对乒乓球不友好**：该类样本以足/篮/网球为主，
  乒乓球在 640 输入下只有几个像素 → 漏检。对策：`--imgsz 1280`、降低 `conf` 到 0.05-0.15、先框 ROI 再检测。
- **更推荐 YOLOE-26 开放词汇模型**（ultralytics ≥ 8.4.0）：
  `YOLOE("yoloe-26n-seg.pt").set_classes(["ping pong ball"])` 直接出框 + 实例 mask，
  不受 COCO 词表限制，还能给一张参考图做视觉提示（比文字稳）。
- **SAM2 官方包只写了支持 Linux**，Windows 建议直接用 ultralytics 的 `SAM("sam2.1_t.pt")`。
  PyPI 上的 `sam2==1.1.0` **不是** Meta 官方包，别装错。
- **速度参考**（官方 CPU 数据 + A100 数据折算）：SAM2.1-tiny 在你的 5060 上估计 30-45 FPS，
  YOLO26n-seg / YOLOE-26n-seg 更快（60+ FPS）。真要 100+ FPS 还是传统方法。
- **学习式轨迹预测（LSTM/Transformer）在这个任务上不划算**：
  单目 + 短历史（<1 s）时，物理模型（重力 + 滤波）明显优于黑盒学习模型，
  而学习模型需要海量标注轨迹。文献（Achterhold et al., L4DC 2023, arXiv:2305.15189）也是这个结论：
  **灰盒（物理 + 少量学习参数 + EKF）> 纯黑盒**。
- `boxmot` 之类的高级跟踪库要求 Python < 3.14，你的环境装不了；
  ultralytics 自带的 6 种跟踪器全是线性卡尔曼，对抛物线下落段有系统偏差 ——
  所以本项目自己做重力感知的滤波与外推。

---

## 七、代码结构

```
sphere_tracker/
  config.py     所有可调参数（dataclass，可存取 JSON）
  types.py      Detection / TrackState / FrameResult / TrailPoint
  geometry.py   亚像素圆拟合、径向一致性、梯度校验、相机模型、单目测深
  detect.py     检测后端：color / hough / motion / fusion / yolo / yolo_sam
  track.py      卡尔曼滤波（含机动检测与重初始化）+ 轨迹管理
  predict.py    轨迹预测：最长有效后缀拟合 + 已知重力约束
  viz.py        拖尾、圆、预测虚线、HUD、三维小面板
  sources.py    摄像头 / 视频文件 / 图片序列
  pipeline.py   单帧总流程编排

apps/
  run_realtime.py       实时摄像头主程序
  run_video.py          离线处理视频/图片，导出 CSV
  calibrate_color.py    HSV 阈值交互标定
  calibrate_focal.py    单点焦距标定（三维必需）
  selftest_synthetic.py 合成数据自测（真值已知，可直接当回归测试）
```

在你自己代码里调用：

```python
import cv2, sys
sys.path.insert(0, r"D:\MY_CODING_PROGRAM\Three-dimensional-sphere-motion-trajectory-recognition")
from sphere_tracker.config import AppConfig
from sphere_tracker.pipeline import SpherePipeline
from sphere_tracker.sources import CameraSource

cfg = AppConfig()
cfg.detector.method = "color"
cfg.detector.color = cfg.detector.color.preset("white_ball")
cfg.camera.ball_diameter_m = 0.04

pipe = SpherePipeline(cfg)
src = CameraSource(index=0)
while True:
    ok, frame, t = src.read()
    if not ok:
        break
    result = pipe.process(frame, t)      # 检测 + 跟踪 + 三维 + 预测
    vis = pipe.render(frame, result)     # 画好标注的图
    if result.track:
        print(result.track.x, result.track.y, result.position_3d, result.prediction)
    cv2.imshow("ball", vis)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
```

---

## 八、调参顺序（遇到问题按这个顺序查）

1. **完全检测不到** → 跑 `apps/calibrate_color.py`，先看掩膜窗口里球是不是白的。
   不是就调 HSV，或者按 `1`-`6` 换预设。
2. **检测到乱七八糟的东西** → 用 `--method color` 单独看；
   提高 `detector.min_radius`（把小球噪声排除）、提高 `min_fill_ratio` / `min_radial_consistency`。
3. **半径/深度不准** → 确认 `--ball-diameter` 填对了；确认 `subpixel_refine=True`；
   球不要被遮挡、不要有严重运动模糊（提高快门速度或降分辨率提帧率）。
4. **轨迹抖动** → 增大 `tracker.process_noise` 会让它更跟手但更抖；
   抖动大就减小它、或增大 `measurement_noise`。
5. **预测不准** → 缩短 `predict.horizon_s`（物理上预测越远越不准）；
   确保 `use_pixel_gravity=True` 且球直径填对；严格按时间戳调用 `process(frame, t)`。
6. **帧率不够** → 用 `--method color`、缩小处理分辨率（`--scale` 只影响显示，
   要提速请用 `cfg.viz` 关闭可视化项，或用 ROI 只处理球所在的区域）。

---

## 九、已修复的问题（原脚本）

| 文件 | 问题 | 修复 |
|---|---|---|
| `video_edge_detect.py` | `OUTPUT_PATH = "../outputs/videos/..."` 会写到**项目上一级目录**，那里没有 outputs 文件夹，VideoWriter 直接失败 | 改成项目内的 `outputs/videos/`，并自动建目录 |
| `video_edge_detect.py` | XVID 编码器 + `isColor=False` 写单通道灰度图，多数机器上写出来是坏文件 | 转成 BGR 三通道写入 |
| `test.py` | `cv2.imread("test.jpg")` 路径错误（图片实际在 `material_images/`），读不到会直接崩 | 修路径并加 None 检查 |
| `test.py` | 保存结果到当前目录，和 `outputs/` 的约定不一致 | 统一存到 `outputs/images/` |

> 另外提醒：`material_images/test.jpg` 的**真实格式是 PNG**（含 alpha 通道），
> 只是扩展名写成了 .jpg。OpenCV 能读，但某些工具会报格式错误，建议改名。

---

## 十、下一步（预测未来轨迹）已经做完了

你原计划"先做标记现有轨迹，再做预测"，其实两者在本项目里是同一条链路 ——
预测就建立在跟踪器给出的速度/加速度状态和轨迹历史之上，所以已经一并实现：

- `predict.method = "quadratic"`（默认）：自适应窗口二次拟合 + 已知重力，稳定段 0.2 s 预测误差 2-3 px
- `predict.method = "kalman"`：直接外推滤波状态，更平滑但对突变迟钝
- 三维模式下开 `use_gravity`，可以正确处理透视效应，输出米制的未来轨迹

如果要继续做下去，比较有价值的方向：
1. **双目/深度相机** → 真正的三维，摆脱"靠球大小猜深度"
2. **落点预测**：把预测轨迹和地面平面求交，直接给出落点坐标
3. **击球检测**：用机动检测的输出反过来识别"什么时候被打了"
4. **多球 + 身份保持**：现在跟踪器已支持多目标，但需要更好的外观特征做关联
