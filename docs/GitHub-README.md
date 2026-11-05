# 球形物体三维运动轨迹识别

> 用两台普通 USB 摄像头，实时识别球体的三维位置、重建运动轨迹并预测后续位置。
> 纯 OpenCV 传统方法即可运行，**不需要 GPU，也不需要深度学习框架**；YOLO / SAM 后端是可选增强。

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](requirements.txt)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)](requirements.txt)

---

## 目录

- [这个项目解决什么问题](#这个项目解决什么问题)
- [主要能力](#主要能力)
- [运行环境与硬件](#运行环境与硬件)
- [五分钟快速开始](#五分钟快速开始)
- [完整工作流程](#完整工作流程)
- [坐标系约定（重要）](#坐标系约定重要)
- [项目结构](#项目结构)
- [命令速查](#命令速查)
- [文档导航](#文档导航)
- [算法概述](#算法概述)
- [数据格式](#数据格式)
- [运行测试](#运行测试)
- [已知限制](#已知限制)
- [常见问题](#常见问题)
- [引用与许可](#引用与许可)

---

## 这个项目解决什么问题

单目相机只能看到球在画面里的二维位置，**深度必须靠假设**（例如已知球的真实直径）才能反推，精度差一个量级，而且球远近变化时误差迅速放大。

本项目用**双目视差**直接求深度：两台相机从不同位置同时看同一个球，同名点在左右画面里的横向偏移（视差）与深度成反比。这样得到的三维轨迹不依赖球径先验，可以直接用于运动学分析（速度、加速度、抛物线拟合、反弹检测）。

项目同时保留了单目链路，用于：没有双目装置时的快速验证、以及作为深度融合中的一路独立观测。

---

## 主要能力

| 环节 | 实现 | 说明 |
| --- | --- | --- |
| 2D 球检测 | 颜色 / 梯度几何 / 背景差分 / 投票融合 | 可插拔注册表，见 `sphere_tracker/detectors/` |
| 亚像素圆拟合 | 径向梯度射线采样 + Taubin / 鲁棒圆拟合 | 圆心与半径细化到亚像素 |
| 2D 跟踪 | 二阶卡尔曼 + 卡方门控 + 机动/反弹重初始化 | `sphere_tracker/track.py` |
| 双目标定 | `joint`（共视充足）/ `chain`（大角度、零共视） | `calib/`，支持平行与近似 90° 装置 |
| 双目三维 | 校正视差 `Z = f·B/d`，或去畸变射线最近点交会 | `stereo/triangulate.py`、`stereo/angled_geometry.py` |
| 深度融合 | 双目视差 / 单目球径先验，逆方差加权 | `stereo/depth_fusion.py` |
| 轨迹平滑 | Savitzky–Golay（等间隔重采样） | `stereo/motion.py` |
| 轨迹预测 | 自适应窗口二次拟合 + 已知重力约束 / 卡尔曼外推 | `sphere_tracker/predict.py`、`stereo/predict3d.py` |
| 可视化 | 视频叠加、3D 轨迹查看器、坐标极值、采样点拾取 | `apps/view_3d.py` |
| 图形界面 | 门厅 → 相机管理 / 录制 / 标定 / 数据管理 / 3D 查看 | `启动程序.py`、内置「操作说明」窗口 |

**不需要训练数据**：默认 `color` 后端是纯图像处理，装完依赖即可跑。AI 后端（YOLO 球检测器）需要用项目自己的数据自举训练，属于可选路径。

---

## 运行环境与硬件

- **操作系统**：Windows 10/11 为主（相机设备枚举、MSMF 后端、`启动程序.bat` 针对 Windows）；核心算法为平台无关的 Python。
- **Python**：3.10+（项目在 3.14 环境开发验证）。
- **依赖**：`opencv-contrib-python`、`numpy`、`scipy`、`matplotlib`、`Pillow`、可选 `python-pptx`。完整列表见 [`requirements.txt`](requirements.txt)；AI 后端另见 [`requirements-ai.txt`](requirements-ai.txt)。
- **相机**：两台 USB 摄像头，建议同型号、固定在同一刚性支架上。**分辨率与帧率必须一致**，实测 1920×1080@30 可稳定双开。
- **标定板**：打印 [`assets/boards/`](assets/boards/) 里的棋盘图（提供 A4 竖版 20 mm 与横版 25 mm 两种，PDF 与 PNG 各一份）。打印后务必用尺子核对实际方格边长，填错会整体缩放三维结果。
- **球体**：建议纯色哑光球，避免高光和透明材质；标定颜色阈值的方法见 [`docs/纯色球检测方案.md`](docs/纯色球检测方案.md)。

> 首次运行 `启动程序.bat` 会自动寻找能 `import cv2` 的 Python、检查并安装依赖，然后打开图形界面。

---

## 五分钟快速开始

```bat
:: 1) 安装依赖（或双击 启动程序.bat 让它自动装）
python -m pip install -r requirements.txt

:: 2) 确认环境与代码正常（约 1 分钟，不需要摄像头）
python apps\run_tests.py fast -j auto

:: 3) 启动图形界面
python 启动程序.py
```

图形界面里按「三步走」引导操作；界面右下角有 **「? 操作说明」**，内置分区目录与搜索，覆盖每个窗口的每个按钮。

想完全走命令行，最小闭环是：

```bat
:: ① 固定左右相机角色（编号会漂移，设备身份不会）
python apps\tools\cameras.py identify

:: ② 双目标定（需要打印好的棋盘板）
python apps\calibrate_stereo.py --method joint --cols 8 --rows 5 --square 25 ^
  --target 40 --left left --right right

:: ③ 录制一段双目视频
python apps\record_stereo.py --left left --right right --duration 10 --out data\stereo

:: ④ 离线重建三维轨迹
python apps\run_stereo_video.py data\stereo\<会话目录>

:: ⑤ 查看三维轨迹（自动打开最新会话）
python apps\view_3d.py
```

标定与录制的完整参数含义见 [`docs/用户手册.md`](docs/用户手册.md) 第 2、4 节。

---

## 完整工作流程

```
        ┌──────────── 一次性 ────────────┐
        │  相机角色绑定  →  双目标定      │
        │  cameras.py identify           │
        │  calibrate_stereo.py           │
        └──────────────┬─────────────────┘
                       │ config/stereo_calibration.*
                       ▼
┌──────────┐    ┌──────────────┐    ┌────────────────┐    ┌──────────────┐
│ 双目录录制 │ → │ 时间戳同步/配对 │ → │ 球检测 + 跟踪   │ → │ 三角测量     │
│ record_* │    │ sync.py       │    │ sphere_tracker │    │ stereo/      │
└──────────┘    └──────────────┘    └────────────────┘    └──────┬───────┘
                                                                  │
                       ┌──────────────────────────────────────────┘
                       ▼
        traj3d.json / visualization.json
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   3D 查看器      运动量导出      平滑 / 预测
   view_3d.py     export_motion   motion.py / predict3d.py
```

三条实用建议：

1. **标定决定精度上限**。录制前跑一次 `python apps\tools\check_stereo_setup.py` 复检装置有没有被碰动。
2. **采集档要一致**。标定图、录制、回放必须用同一分辨率与帧率；装置档案里填写采集档就是为了锁住这一点。
3. **先跑无硬件的自测**。`fast` 档包含合成真值回归，能在没有相机的机器上验证整条链路。

---

## 坐标系约定（重要）

全项目只有一套约定，**相机中心向前为 Z、向右为 X、向下为 Y**，单位米：

| 装置模式 | 原点 | 深度求法 | 输出标记 |
| --- | --- | --- | --- |
| `parallel`（平行双目） | 校正后的左相机光心 | 视差 `d = u_L − u_R`，`Z = f·B/d` | `stereo_mode=parallel` |
| `angled`（大角度 / 近似 90°） | 原始左相机光心 | 去畸变后两条射线做最近点交会 | `stereo_mode=angled` |

- 两者都是**相机系**，不是世界系。若要转到世界系，需用棋盘做 PnP 生成 `world_frame.json`，命令行 `run_stereo_video.py --world-frame`。
- 导出时若没有世界系文件，必须显式写 `coordinate_system="left_camera"`，**不许静默当成世界系**。
- 不同模式的 `xyz` 不能混进同一条轨迹。
- 图示见 [`docs/坐标系统图.png`](docs/坐标系统图.png)，字段定义见 [`docs/数据标准.md`](docs/数据标准.md) 第 6 节。

---

## 项目结构

```
启动程序.bat / 启动程序.py / 启动程序_命令行.py   三个启动入口（双击即用 / GUI / CLI）
sphere_tracker/     单目核心：检测 → 跟踪 → 预测 → 可视化
calib/              双目标定（离线、一次）：内参、外参、质量判定、世界系
stereo/             双目逐帧：同步 → 配对 → 三角测量 → 轨迹 → 导出
apps/               可执行程序与图形界面；tools/ 是一次性设置与诊断工具
tests/              回归测试（fast 档 24 个）
assets/boards/      标定板（PDF / PNG）
config/             运行期生成的本机标定结果（不入库）
data/               录制会话（不入库）
outputs/ logs/      运行产物与日志（不入库，可随时清理）
docs/               说明文档（见下节导航）
```

依赖方向是单向的：`apps → stereo/calib → sphere_tracker`，库代码不反向依赖界面。详见 [`docs/文件结构.md`](docs/文件结构.md) 与 [`docs/API.md`](docs/API.md) 第 14 节。

---

## 命令速查

| 目的 | 命令 |
| --- | --- |
| 启动 GUI / CLI | `启动程序.bat` / `启动程序.bat cli` |
| 列出相机设备 | `python apps\tools\cameras.py list` |
| 绑定左右角色 | `python apps\tools\cameras.py identify` |
| 检查相机漂移 | `启动程序.bat check` |
| 相机体检（真开相机） | `python apps\tools\cameras.py doctor` |
| 双目标定 | `python apps\calibrate_stereo.py --method joint ...` |
| 链式标定（90° 装置） | `python apps\calibrate_stereo.py --method chain --same-board ...` |
| 标定复检 | `启动程序.bat verify` |
| 录制双目 | `python apps\record_stereo.py --left left --right right --duration 10` |
| 离线三维重建 | `python apps\run_stereo_video.py <会话目录>` |
| 3D 查看器 | `python apps\view_3d.py [会话目录]` |
| 运动量导出 | `python apps\tools\export_motion.py ...` |
| 回归测试 | `python apps\run_tests.py fast -j auto` |
| 规范自查 | `启动程序.bat standards` |

完整参数表与故障排查见 [`docs/用户手册.md`](docs/用户手册.md)。

---

## 文档导航

| 想做什么 | 读哪份 |
| --- | --- |
| 第一次接触，了解全貌 | [`docs/项目总纲.md`](docs/项目总纲.md) |
| 上手操作、命令、排障 | [`docs/用户手册.md`](docs/用户手册.md) |
| **理解算法原理** | [`docs/算法解析.md`](docs/算法解析.md) |
| 查函数签名与数据结构 | [`docs/API.md`](docs/API.md) |
| 3D 查看器的鼠标、视角、采样点 | [`docs/3D查看器.md`](docs/3D查看器.md) |
| 标定方法怎么选、为什么 90° 要链式 | [`docs/标定算法对比.md`](docs/标定算法对比.md) |
| 真机标定的逐步操作 | [`docs/真机标定操作清单.md`](docs/真机标定操作清单.md) |
| 球检测器选型与调参 | [`docs/纯色球检测方案.md`](docs/纯色球检测方案.md) |
| 相机角色、端口漂移 | [`docs/摄像头设备管理方案.md`](docs/摄像头设备管理方案.md) |
| 数据文件字段定义 | [`docs/数据标准.md`](docs/数据标准.md) |
| 速度 / 加速度导出 | [`docs/运动数据交付说明.md`](docs/运动数据交付说明.md) |
| 目录放置规则与依赖方向 | [`docs/文件结构.md`](docs/文件结构.md) |
| 扩展新的检测器 / 后端 | [`docs/插件化架构.md`](docs/插件化架构.md) |

图形界面里还有**内置操作说明窗口**（主界面工具条第一个按钮，或各窗口右下角「? 操作说明」），支持按分区浏览与关键词搜索，不需要翻文档也能查到某个按钮是干什么的。

---

## 算法概述

一句话链路：**检测每帧的球心 → 左右画面对同名点配对 → 用标定参数反投影成两条空间射线 → 求交得到三维点 → 连成轨迹 → 平滑与预测**。

- **平行双目**：先做立体校正，让同名点落在同一水平扫描线上，视差 `d = u_L − u_R`，深度 `Z = f·B/d`（`f` 焦距、`B` 基线，均为米/像素一致单位）。深度误差随 `Z²` 放大，所以远处精度下降很快。
- **大角度双目**：两路光轴接近垂直、几乎没有共视，无法用常规 `stereoCalibrate` 求外参。项目改用**链式标定**（用一块板在不同姿态下分别约束两相机），三维点由两条去畸变射线的**最近点交会**求得，不用 `Z=fB/d`。
- **单目兜底**：已知球真实直径 `D` 时，`Z = f·D/(2r)`（`r` 为像素半径）；与双目深度按逆方差加权融合。
- **检测与跟踪**：颜色阈值 / 圆梯度 / 背景差分多路检测后投票融合，径向梯度射线做亚像素细化；二阶卡尔曼跟踪带卡方门控与机动重初始化。
- **轨迹后处理**：Savitzky–Golay 平滑（先等间隔重采样）、自适应窗口二次拟合预测，已知重力时固定二次项。

完整推导、公式、模块索引和误差分析见 [`docs/算法解析.md`](docs/算法解析.md)。

---

## 数据格式

一次采集 = 一个会话目录。核心产物：

| 文件 | 内容 |
| --- | --- |
| `session.json` | 采集参数、双目模式、采集档、帧数 |
| `left/`、`right/` | 两路原始帧或视频 |
| `derived/traj3d.json` | **标准三维轨迹**：逐点 `frame / t / xyz`、坐标系标记、质量位 |
| `derived/visualization.json` | 3D 查看器直接读的数据（含采样点、极值、视频叠加信息） |
| `derived/annotated.mp4` | 带轨迹拖尾与 HUD 的标注视频 |

字段的精确定义、单位、`coordinate_system` 取值与校验规则见 [`docs/数据标准.md`](docs/数据标准.md)。

> 本仓库**不包含任何实验采集数据**：`data/`、`outputs/`、`logs/`、`config/` 都是运行期产物，已被 `.gitignore` 排除。复现实验需要自己接相机录制。

---

## 运行测试

```bat
python apps\run_tests.py fast -j auto
```

`fast` 档 24 个回归测试覆盖：合成真值下的检测/跟踪/测深精度、双目标定与三角测量、数据格式、GUI 参数接线、文档事实一致性。全部不需要摄像头。

`启动程序.bat gate` 会同时跑规范自查与 fast 档，适合提交前执行。

---

## 已知限制

- **标定即精度上限**：相机被碰动、打印的棋盘格尺寸不准、两路分辨率不一致，都会直接污染三维结果。
- **深度误差随距离平方增长**：远距离、小基线、低分辨率都会放大视差量化误差。
- **纹理与光照敏感**：默认 `color` 后端需要稳定的光照与纯色哑光球；强高光、透明球、同色背景会失效。
- **同步依赖时间戳**：两路帧率不一致或丢帧会造成配对错误，采集时务必锁同一档。
- **不是世界系**：默认输出是左相机系；跨会话比较前要确认坐标系标记。
- **单目深度依赖球径先验**：球径填错会整体缩放深度，只能用于粗略估计或融合中的弱观测。
- 更完整的诚实清单见 [`docs/用户手册.md`](docs/用户手册.md) 第 9 节。

---

## 常见问题

**Q：列表里有几台相机，怎么确认哪台是左？**
用 `cameras.py identify` 逐个显示画面并绑定角色。角色按设备身份（而非编号）保存，换 USB 口不会错乱；笔记本内置/红外相机会被标成「无关」并跳过。

**Q：播放器提示「视频只有 N 帧，当前要点第 M 帧」？**
`annotated.mp4` 生成中断或被覆盖，比轨迹短。原始左右视频还在的话，重跑 `run_stereo_video.py` 即可重新生成。

**Q：3D 查看器里曲线跑到坐标框外面 / 背景不动？**
旧版本滚轮只改坐标范围导致的；现在滚轮是整幅缩放、`Shift+滚轮` 才是定点放大。鼠标映射与六视图方向见 [`docs/3D查看器.md`](docs/3D查看器.md)。

**Q：换一台机器要不要重标？**
要。相机内参会变、外参更是与机位绑定，`config/` 整个目录都不入库，换机器请重新标定。

---

## 引用与许可

本项目参考了以下工作（论文原文不在本仓库内，请通过正规渠道获取）：

- *Where Is the Ball? 3D Ball Trajectory Estimation From 2D Monocular Tracking*
- *3D Ball Localization From A Single Calibrated Image*

相机标定与立体视觉的基础方法来自 OpenCV 的 `calibrateCamera` / `stereoCalibrate` / `stereoRectify` 与张正友平面标定法。

**许可**：本仓库未附带 LICENSE 文件，默认保留所有权利；如需用于课程、科研或二次开发，请先联系仓库维护者。第三方依赖各自遵循其原有许可证，见 [`requirements.txt`](requirements.txt)。
