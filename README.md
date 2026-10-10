> 📜 **规则在 [agent.md](agent.md)**（目录 / 命名 / 放置 / 依赖方向 / 提交 / 质量底线 的唯一权威）。
> 本文是面向外部的项目介绍与状态概览。

# 球体运动轨迹识别（检测 → 跟踪 → 三维 → 预测）

用摄像头实时识别球的运动轨迹、标记已有轨迹、预测未来轨迹。
纯 OpenCV 传统方法即可达到亚像素精度与 100+ FPS，**不装任何深度学习框架也能跑**；
同时预留了 YOLO / SAM 的 AI 后端，一行配置就能切换对比。

> 📖 **第一次接触本项目？先读 [`docs/项目总纲.md`](docs/项目总纲.md)。**
> 那份是入口文档：项目能干什么、当前进度、以及**六份文档各自该在什么时候读**。
> 本 README 是功能与命令的详细说明，适合已经知道要做什么、来查具体用法时看。

---

## 〇、当前进度与快速开始

### 做到了什么（全部实测，可复现）

| 环节                                                  | 状态         | 实测结果                                                                                                |
| ----------------------------------------------------- | ------------ | ------------------------------------------------------------------------------------------------------- |
| 球体 2D 检测（颜色 / 梯度几何 / 背景差分 / 投票融合） | ✅ 完成       | 检出率 **100%**，圆心误差 **0.54 px**                                                                   |
| 亚像素边缘细化                                        | ✅ 完成       | 半径误差 **0.06 px**（细化前 0.4–0.5 px）                                                               |
| 卡尔曼跟踪（含机动检测 / 反弹重初始化）               | ✅ 完成       | 跟踪中位误差 **0.65 px**                                                                                |
| 单目三维估计（球直径先验 `Z = fR/r`）                 | ✅ 完成       | 深度误差 **0.31%**，3D RMSE **15.1 mm**                                                                 |
| 轨迹预测（自适应窗口 + 已知重力约束）                 | ✅ 完成       | 0.1 / 0.2 / 0.3 s 分别 **0.81 / 1.20 / 1.98 px**                                                        |
| 可视化（拖尾 / 圆 / 预测虚线 / HUD / 3D 小视图）      | ✅ 完成       | 见 `outputs/videos/annotated.mp4`                                                                       |
| 实时程序 + 离线批处理 + CSV 导出                      | ✅ 完成       | 单路 **~160 fps**（960×540，纯 CPU 单线程）                                                             |
| 合成真值自测（可直接当回归测试）                      | ✅ 完成       | `python tests/selftest_synthetic.py`                                                                    |
| **相机运动适配**（手持/云台/平移拍摄）                | ✅ 完成       | 平移下半径误差 0.23 px（不补偿 1.58 px），见第十一节                                                    |
| 相机自运动估计（迟滞判定 + 位移换算）                 | ✅ 完成       | 位移误差 ≤ 0.01 px 且**不随缩放比例变化**（原先差 `s²`，1080p 下 36 倍，见 `docs/交接文档.md` 坑 20）    |
| **双目自动标定**                                      | ✅ 完成       | 合成真值验证：基线误差 0.006%、端到端三维误差 3.4 mm，见第十二节                                        |
| 双目标定复检（装置是否被碰动）                        | ✅ 完成       | 未动 0.08 px / 碰动 2° 后 27.9 px，可明确判定                                                           |
| 多相机/换相机适配                                     | ✅ 完成       | `apps/tools/list_cameras.py` 探测后端/分辨率/帧率；`cameras.py` 按**设备身份/端口**固定角色，免索引漂移；可把不用的相机标为**「无关」**（自动挑相机时跳过） |
| 一键启动（双击即用，不用敲 Python）                    | ✅ 完成       | 根目录唯一入口 `启动程序.bat`：找 Python（真试一次 `import cv2`）→ 缺依赖自动装 → 启动界面；`启动程序.bat cli/setup/check/precheck/calibrate/verify/run/test` 可直接做某一步 |
| AI 分支环境（YOLO / SAM 后端）                        | ⚙️ 环境就绪   | torch 2.11.0+cu128、ultralytics 8.4.159、CUDA 可用（sm_120）                                            |
| **两路视频同步**                                      | ✅ 完成       | 合成真值验证：人为平移 0 / ±8 / -11 / +37 帧全部找回，置信度 1.000                                      |
| **成对取帧（PairSource）**                            | ✅ 完成       | 按同步偏移取到"同一时刻"的左右帧，帧索引语义有回归测试                                                  |
| **极线配对 + 三角测量**                               | ✅ 完成       | 合成真值：配对率 **100%**，三维误差 **1.3 mm**                                                          |
| **双目整条链路**（真渲染 + 两路真检测）               | ✅ 完成       | clean 配对率 100% / 三维 RMSE **15.6 mm**（折算 1 m 处 **1.13 mm**）；realistic 99.4% / 31.6 mm         |
| **真机标定 + 真机双目录制**                           | ⬜ **未做过** | 当前所有双目指标都是**合成真值**，见第十二节与 `docs/计划书.md` M1                                      |
| 双目接进实时主流程                                    | ⬜ 待做       | `stereo/` 模块已就绪，缺采集与显示编排                                                                  |
| Savitzky-Golay / 3D 轨迹平滑                          | ⬜ 待做       | scipy 1.18.1 已装                                                                                       |
| 三维可视化（Plotly / Unity）                          | ⬜ 待做       | `pip install plotly`（网络现可用）；matplotlib 3.11.2 已装可作备选                                      |

一句话：**单路视频的"检测 → 跟踪 → 单目三维 → 预测 → 可视化"整条链路已经跑通并量化；
双目的"标定 → 同步 → 成对取帧 → 极线配对 → 三角测量"代码侧也已完成并通过回归测试，
两者都只差真机验证。缺的是真机标定、双目实时编排与三维可视化。**

> ⚠️ **双目部分的全部指标都是合成真值**（真渲染 + 真检测，但不涉及真实相机与真实打印标定板）。
> 数值漂亮不等于真机能做到，**接手后第一件事就是真机标定**，见 `docs/交接文档.md` §9。

> 📖 两个 15 mm 不要混：本节上表"单目三维 RMSE **15.1 mm**"来自
> `tests/selftest_synthetic.py`（球径 22 cm、距离 3.7–4.4 m 的单目场景）；
> 双目的 **15.6 mm** 来自 `tests/selftest_stereo_branch.py`（另一套合成场景、双目三角测量）。
> 两者基准不同，**不可直接比较大小**。

> ✅ **2026-10-10 全量复核**：`tests/` 下**九个**回归测试全部输出"全部通过"，
> 且上表的每一项精度指标都与本文档记录**逐位一致**。
> 复核时另修掉 3 个 bug（相机自运动的迟滞失效、位移换算差 `s²`、
> PPT 预览的连线静默不画）。其中位移换算那个一直被合成场景盖住 ——
> **这是"合成真值漂亮 ≠ 真机可用"的活例子**，详见 `docs/交接文档.md` §10.3 与坑 20。

### 三步开始使用

> 💡 **不想敲命令？双击根目录的 `启动程序.bat` 就行**（唯一的双击入口）：
> 它会自动找一台真的装了 opencv 的 Python、缺依赖就问一句然后自动装、
> 然后启动图形界面。只想重做某一步就加参数，例如 `启动程序.bat calibrate`。
> 详见 `docs/用户手册.md` §2.2。

```powershell
conda activate myenv
cd D:\MY_CODING_PROGRAM\Three-dimensional-sphere-motion-trajectory-recognition

# 1) 自测：确认环境与代码正常（不需要摄像头，约 1 分钟）
python tests/selftest_synthetic.py

# 2) 看一遍离线效果（用已生成的合成视频，有真值可对照）
python apps/run_video.py outputs/videos/synthetic_ball.mp4 --out outputs/videos/annotated.mp4 --csv outputs/traj.csv --ball-diameter 0.22 --focal 831
#    处理完打开 outputs/videos/annotated.mp4，就能看到轨迹拖尾与预测虚线

# 3) 接真实摄像头（不写编号，启动时自动挑最优：优先外接、跳过标了「无关」的、跳过全黑流）
python apps/run_realtime.py --method fusion
#    想指定也行：--camera 0 用编号，--camera main 用角色名（需先 identify 固定）
```

### 双目（研究思路主线）

```powershell
# 0) 第一次用：逐个画面确认哪台是左、哪台是右、哪台「无关」，固定成"角色"
#    （索引会漂移，身份不会。不打算用的内置/红外相机就在这里按 n 标成「无关」）
python apps/tools/cameras.py identify      # 窗口里 l=左 r=右 n=无关 m=主 s=跳过 q=退出

# 1) 双目标定（需要两台相机 + 打印好的棋盘格，见 assets/boards/）
python apps/calibrate_stereo.py --left left --right right --cols 8 --rows 5 --square 25
#    也接受具体编号（--left 0 --right 1）或 --auto-cameras（自动挑两台）；
#    结果落在 config/stereo_calibration.*（默认被 gitignore，要存档请手动拷贝）

# 2) 每次拍摄前复检装置是否被碰动（未动通过、碰动不通过）
python apps/verify_stereo.py --calib config/stereo_calibration --left left --right right

# 3) 双目整条链路的回归测试（合成真值，不需要相机）
python tests/selftest_stereo_branch.py
```

> ⚠️ **双目标定（第 1 步）早期未在真机上跑通过**（无法访问摄像头时会报取不到帧）。
> 详见 `docs/交接文档.md` §9「接手后建议的第一件事」。

实时窗口快捷键：`q` 退出、`c` 切检测方法、`m` 看掩膜、`s` 存快照与配置、`v` 录制、`r` 重置轨迹、`3` 开关三维小视图。

**对准真实小球时的顺序**：先跑 `python apps/tools/calibrate_color.py --camera 0` 拖滑块
（按 `1`–`6` 可套预设），直到掩膜窗口里只有球是白的，按 `s` 保存；再用
`python apps/run_realtime.py --config config/sphere_config.json --ball-diameter <球直径米>` 运行。

### 摄像头身份与端口固定（推荐先看这一节）

**编号会漂移，身份不会。** `cv2.VideoCapture(0)` 的 `0` 由系统枚举顺序决定：
换一次 USB 口、多插一个设备，编号 1 可能从"外接相机"变成"笔记本内置"。
把 `--left 0 --right 2` 写死，索引一变就会**悄悄用错相机**，标定白做。
本项目为此做了三层能力：

1. **固定**：把"哪台是左、哪台是右"记成**设备身份**（Windows 上取 PnP `InstanceId` / USB 端口）
   存进 `config/cameras.json`，之后用**角色名**启动。
2. **按端口选**：`--left port:Port_#0001.Hub_#0003`、`--right id:USB\VID_...`、
   `--right name:aoni` 都能直接指定物理设备。
3. **自动挑**：不指定时（`run_realtime.py` 默认、`--auto-cameras`）在启动阶段探测所有索引，
   挑"能打开 + 非全黑 + 分辨率达标"的，优先外接、避开内置/红外。

```powershell
# 列出设备身份与端口（不开相机，任何机器都能跑，~0.1 s）
python apps/tools/cameras.py list
python apps/tools/cameras.py list --probe      # 顺带逐个试开，报实际分辨率 / 是否全黑

# 逐个显示画面，目视确认哪台是左/右/无关，写入 config/cameras.json
python apps/tools/cameras.py identify

# 非交互固定某角色到某设备
python apps/tools/cameras.py pin left  port:Port_#0001.Hub_#0003
python apps/tools/cameras.py pin right 0

# 把不打算用的相机标成「无关」（笔记本内置 / 红外）——自动挑相机会跳过它
python apps/tools/cameras.py ignore 3          # 也可 port:xx / id:xx / name:xx
python apps/tools/cameras.py ignored           # 查看清单（标注在不在线）
python apps/tools/cameras.py unignore 3        # 取消某台；--all 清空整份清单

# 每次拍摄/重插相机前：比对配置与当前设备，有没有漂移 / 被别的设备占用
python apps/tools/cameras.py check             # 有 [危险] 项时退出码为 2
```

**「无关」是什么**：`identify` 的画面里，一台相机有四种去处 ——
`l`=左、`r`=右、`n`=**「无关」**（永远不要用）、`m`=单目主相机；`s`=跳过（不改配置）。
**左 / 右 / 无关三者互斥**：改标一个会自动摘掉旧的。它记的是**设备身份**而非编号，
所以重插 USB 后依然有效。自动挑相机（`--auto-cameras`、`--camera auto`）会跳过它们，
不会再出现"左边是外接、右边却挑到笔记本内置"。**显式写编号（`--camera 3`）仍然会用它** ——
你点名的那台优先级最高。万一其它相机都不够用，程序会兜底动用并**打印警告**，不会静默用错。

启动器与一键脚本里也都有入口：
图形版第 1 步的『刷新设备列表』『相机设备管理（识别/固定/标无关）』『检查相机映射』
『「无关」相机清单』『清空「无关」标记』五个按钮；
命令行版主菜单的『相机设备管理』子菜单（[5] 标无关 / [6] 查看与清除）；
以及根目录的 `启动程序.bat setup` / `启动程序.bat check`（双击入口加参数，不用敲 python）。

**配置与实物对不上时的行为**（刻意设计）：不会替你猜。已 `identify` 确认过（`verified=true`）
的角色若身份消失，直接**报错并附当前设备清单**，让你重新确认；未确认过的角色才会退回旧索引
并打 `[警告]`。交互式终端下还会把设备清单摆出来让你**自己选编号**。

### 本机摄像头（实测样本，编号仅为示例 —— 会漂移）

| 编号 | 设备                                        | 状态                                                                             |
| ---- | ------------------------------------------- | -------------------------------------------------------------------------------- |
| `0`  | **aoni webcam A25**（外接 USB）             | ✅ 可用，成像正常；双目一台                                                       |
| `1`  | **aoni webcam A25**（外接 USB）             | ✅ 可用，成像正常；双目另一台                                                     |
| `2`  | **Integrated Camera**（笔记本内置）         | ⚠️ 可用但**自拍视角**（对着使用者），**不要用于双目**（无法刚性固定），仅应急备选 |
| `3`  | Integrated IR Camera（红外，Windows Hello） | ❌ OpenCV 打不开；且红外**无彩色**，不能用于本课题                                |

> 结论：**别背编号，跑 `python apps/tools/cameras.py list` 看当前顺序。**
> 这套编号会随插口漂移：本文档最早实测是"0、2 = 外接，1 = 内置"，
> 把第二台外接换到另一个 USB 口后变成"0、1 = 外接，2 = 内置"。
> 用 `cameras.py identify` 把 left / right 固定到**设备身份**后，就不用再管编号了。
>
> 左右指派：按"人面对镜头时左手边那台为左"的约定。第一次 `identify` 时窗口里按 `l`/`r` 确认；
> 不确定就先跑「标定前置检查」，看屏幕上"左/右"两个窗口是否与你的物理左右一致，不一致就重新确认。

> ⚠️ **已知限制：同一集线器上的两台同型号相机，双路同时打开会失败。** 实测（2026-10）：
> - `DSHOW` 后端有时枚举不全（早期实测只看到 0、1 两台，看不到第三台）；
> - `MSMF` 后端枚举更全，但**一个进程里同时打开两台会卡死或第二台抓不到帧**
>   （报 `MSMF: can't grab frame. Error: -2147018545`）；
> - 两路**混用不同后端**（一路 DSHOW、一路 MSMF）必然失败 —— 就是上面那个错。
>
> 代码已改为**两路强制用同一个后端、成对地挑**（默认 `--backend auto`：先 DSHOW 再 MSMF），
> 可用 `--backend dshow|msmf` 手动指定。若仍打不开，**把两台外接相机分别插到笔记本机身上
> 两个不同的 USB 口（不要走同一个集线器）**，再重试。

> **踩过的坑（已修复）**：这颗外置摄像头开机后的**前几帧是全黑的**，而旧代码
> "读到第一帧非空就收工"，于是把黑帧当成了正常画面，表现为"这个摄像头是坏的"。
> 现在 `CameraSource` 会等到帧内出现真实内容（std > 1）才接受，
> 并且遇到不支持的分辨率会自动降级重试 —— 这类问题以后不会再骗到你。

**要把这两路变成双目，还差两件事**：

1. **对准同一区域**。现在两台相机一个拍桌面、一个拍人，视野几乎不重叠；
   双目要求小球同时落在两个视角里。
2. **刚性固定并量基线**。两台都上三脚架/横梁，锁死后不要再动；量出两镜头中心的距离（建议 60–150 mm）。

两台**不同型号**是可以做的（标定时各自得到自己的内参 `K` 与畸变 `D`），
但同型号会更省事（畸变特性接近、视场角一致）。

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

| 后端                | 原理                                                          | 适用                         | 本项目实测速度 |
| ------------------- | ------------------------------------------------------------- | ---------------------------- | -------------- |
| `color`             | HSV 阈值 + 形态学 + 轮廓圆度筛选                              | 球与背景颜色可分（**首选**） | 120-180 FPS    |
| `hough`             | 霍夫圆 + 梯度方向/符号双向校验                                | 球与背景同色，但边缘清晰     | 30-45 FPS      |
| `motion`            | 背景差分（固定用 MOG2；**相机在动时自动切成"对齐后再差分"**） | 相机固定或移动都可以         | 96-240 FPS     |
| `fusion`            | 上面几种投票融合（**默认**）                                  | 通用，最稳                   | 27-35 FPS      |
| `yolo` / `yolo_sam` | AI 检测 / 分割（可选依赖）                                    | 复杂背景、遮挡、多球         | 见第六节       |

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

|                  | 半径 RMSE        | 深度误差     | 三维 RMSE    |
| ---------------- | ---------------- | ------------ | ------------ |
| 只用阈值轮廓     | 0.35-0.5 px      | ~1.4%        | 67-100 mm    |
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
python tests/selftest_synthetic.py
```

| 场景      | 方法      | 检出率 | 检测RMSE | 跟踪中位 | 半径RMSE | 三维RMSE | 深度误差 | 0.1s预测 | 0.2s预测 | 0.3s预测 | 速度     |
| --------- | --------- | ------ | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| clean     | **color** | 100%   | 0.54 px  | 0.65 px  | 0.06 px  | 15.1 mm  | 0.31%    | 0.81 px  | 1.20 px  | 1.98 px  | ~160 fps |
| clean     | fusion    | 100%   | 0.54 px  | 0.65 px  | 0.06 px  | 15.1 mm  | 0.31%    | 0.81 px  | 1.20 px  | 1.98 px  | ~29 fps  |
| clean     | motion    | 85%    | 0.54 px  | 0.70 px  | 0.06 px  | 13.5 mm  | 0.28%    | 0.81 px  | 1.10 px  | 1.60 px  | ~238 fps |
| clean     | hough     | 100%   | 41 px    | 0.87 px  | 2.79 px  | 345 mm   | 3.53%    | 8.67 px  | 9.54 px  | 15.4 px  | ~37 fps  |
| realistic | **color** | 100%   | 0.87 px  | 0.68 px  | 0.16 px  | 42.1 mm  | 0.58%    | 1.33 px  | 2.49 px  | 4.43 px  | ~160 fps |
| realistic | fusion    | 100%   | 0.87 px  | 0.68 px  | 0.16 px  | 42.1 mm  | 0.58%    | 1.33 px  | 2.49 px  | 4.43 px  | ~29 fps  |

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

> **`CAP_PROP_FPS` 是假的，别信它。** 实测这两台摄像头：
>
> | 摄像头 | 分辨率 | 上报 fps | **实测采集 fps** |
> |---|---|---|---|
> | cam 0 | 1280x720 (MJPG) | 60 | **10.0** |
> | cam 1 | 1920x1080 (MJPG) | 60 | **30.0** |
>
> 也就是说 `--fps 60` 这个参数只是"请求值"，摄像头不一定给。
> cam 0 在 720p 下实际只有 10 fps，抓快速运动的球会严重拖尾；
> 换成 640x480 通常能明显提升实际帧率。想知道真实帧率，看 HUD 上的 `FPS`，
> 或用 `--width/--height` 换一档再对比。
> 时间戳走的是 `perf_counter` 而不是帧号/fps，所以**即使实际帧率低于请求值，
> 速度和预测的时间轴仍然是对的**。

### 跑起来

```powershell
# 1) 先自测，确认代码在你机器上正常（不需要摄像头）
python tests/selftest_synthetic.py

# 2) 标定颜色阈值（第一次用必做，1 分钟）
python apps/tools/calibrate_color.py --camera 0
#    拖动滑块直到球是白色、背景是黑色；按 1-6 套用预设；按 f 冻结画面；按 s 保存

# 3) 实时运行
python apps/run_realtime.py                       # 不指定相机：启动时自动挑最优 + 融合检测
python apps/run_realtime.py --method color --preset white_ball   # 更快
python apps/run_realtime.py --camera auto                 # 同上，显式写 auto
python apps/run_realtime.py --camera main                 # 用角色名（需先 cameras.py identify）
python apps/run_realtime.py --camera 0 --width 1280 --height 720 --fps 60   # 按编号（会漂移）

# 4) 处理已有视频 / 图片（把自己的视频放到 assets/videos/）
python apps/run_video.py assets/videos/ball.mp4 --out outputs/videos/annotated.mp4 --csv outputs/traj.csv
```

**实时窗口的快捷键**

| 键          | 功能                                                              |
| ----------- | ----------------------------------------------------------------- |
| `q` / `ESC` | 退出                                                              |
| `r`         | 重置轨迹（球飞出去后重新开始）                                    |
| `c`         | 切换检测方法（fusion → color → hough → motion → yolo → yolo_sam） |
| `m`         | 显示/隐藏分割掩膜小窗（调阈值必备）                               |
| `p`         | 开/关轨迹预测                                                     |
| `3`         | 开/关三维视图小面板（俯视 X-Z + 侧视 Y-Z）                        |
| `s`         | 保存快照 + 当前配置                                               |
| `v`         | 开始/停止录制                                                     |
| `空格`      | 暂停                                                              |

### 参数速查（`apps/run_realtime.py`）

| 参数              | 说明                                                                                   |
| ----------------- | -------------------------------------------------------------------------------------- |
| `--method`        | `color` / `hough` / `motion` / `fusion` / `yolo` / `yolo_sam`                          |
| `--preset`        | `white_ball` / `orange_ball` / `green_ball` / `blue_ball` / `red_ball` / `yellow_ball` |
| `--ball-diameter` | **球的真实直径（米）**。三维测距必需，而且必须准 —— 填错多少，深度就错多少             |
| `--focal`         | 相机像素焦距 fx。不填就按水平视场角估算（配置项 `camera.hfov_deg`，默认 60°）          |
| `--horizon`       | 预测未来多少秒（默认 0.6，建议 0.3-0.6）                                               |
| `--predict`       | `quadratic`（默认，推荐）/ `kalman`                                                    |
| `--gravity-fit`   | 三维预测时用已知重力约束（更准，需要标定）                                             |
| `--config`        | 读取标定好的 JSON 配置                                                                 |

---

## 五、标定（决定三维精度）

单目相机靠"球在图像里多大"来推距离：`Z = f · R / r`。

1. **焦距 fx**（`apps/tools/calibrate_focal.py`）
   把球放在**量好**的距离上（比如 1.000 m），跑：
   ```powershell
   python apps/tools/calibrate_focal.py --camera 0 --distance 1.0 --ball-diameter 0.04
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

| 场景                                          | 建议                                    |
| --------------------------------------------- | --------------------------------------- |
| 球颜色和背景能分开                            | **传统 `color`/`fusion`**，别再折腾模型 |
| 背景杂乱、球会变色/反光、多球、要区分球的种类 | YOLO / YOLOE 零样本检测                 |
| 需要像素级精确轮廓（遮挡、形变、深度）        | SAM2 分割                               |
| 想要"这一个球"而不是"任何球"                  | YOLOE 文本提示，或自己标 200-500 张微调 |

### 环境是能装上的（已实测确认）

你的机器：RTX 5060 Laptop 8 GB（sm_120 Blackwell）、驱动 577.05、
myenv Python 3.14.7。经过实测的可用组合：

```powershell
$PY = "C:\Users\<USER>\miniforge3\envs\myenv\python.exe"
$WHL = "$env:TEMP\whl"; New-Item -ItemType Directory -Force -Path $WHL | Out-Null

# 1) PyTorch：必须走 cu128（cu129 缺 cp314 的 torchvision；cu130 需要驱动 >= 580，你的 577 不够）
#
#    【重要】别用官方 index-url https://download.pytorch.org/whl/cu128 ——
#    实测这个源在本机只有 ~0.9 MB/s 而且会中途卡死（连接建着、吞吐为 0），
#    2.6 GB 的 torch 根本下不完；也别直接 pip install torch，PyPI 上的
#    Windows 轮子现在是 +cpu 版，torch.cuda.is_available() 会是 False。
#    用阿里云镜像 + curl 先把 wheel 拉到本地，再离线安装，10 秒级完成：
& curl.exe -s -L -o "$WHL\torch-2.11.0+cu128-cp314-cp314-win_amd64.whl" `
    "https://mirrors.aliyun.com/pytorch-wheels/cu128/torch-2.11.0%2Bcu128-cp314-cp314-win_amd64.whl"
& curl.exe -s -L -o "$WHL\torchvision-0.26.0+cu128-cp314-cp314-win_amd64.whl" `
    "https://mirrors.aliyun.com/pytorch-wheels/cu128/torchvision-0.26.0%2Bcu128-cp314-cp314-win_amd64.whl"
& $PY -m pip install --force-reinstall --no-deps "$WHL\torch-2.11.0+cu128-cp314-cp314-win_amd64.whl" "$WHL\torchvision-0.26.0+cu128-cp314-cp314-win_amd64.whl"

# 2) ultralytics：用 --no-deps，否则它会装 opencv-python 和你的 opencv-contrib-python 打架
& $PY -m pip install -i https://mirrors.aliyun.com/pypi/simple/ ultralytics --no-deps
& $PY -m pip install -i https://mirrors.aliyun.com/pypi/simple/ `
    pyyaml psutil polars timm lap requests scipy pandas ultralytics-thop py-cpuinfo nvidia-ml-py

# 3) 权重：GitHub 直连拿不到，走 gh 代理（或从别的机器拷过来放 models/ 目录）
& curl.exe -s -L -o models\yolo11n.pt `
    "https://ghfast.top/https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"

# 4) 验证
& $PY -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability())"
# 实测输出：2.11.0+cu128 True (12, 0)
```

**跑之前必须设这个环境变量**，否则 conda 的 `libiomp5md.dll` 和 torch 自带的
`libomp.dll` 会冲突，进程直接以 `OMP: Error #15` 退出：

```powershell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
```

### 装上之后怎么用

```powershell
python apps/run_realtime.py --method yolo        # 零样本检测（COCO 的 sports ball 类）
python apps/run_realtime.py --method yolo_sam    # YOLO 给框 + SAM 分割出像素级边缘
python tests/selftest_synthetic.py --check-ai     # 检查依赖和 GPU 是否就绪
```

代码里已经写好这两个后端（`sphere_tracker/detect.py` 的 `YoloDetector` / `YoloSamDetector`），
未安装时会打印清晰的安装提示而不是崩掉。相关参数在 `DetectorConfig`：
`yolo_weights`（默认 `models/yolo11n.pt`，可换 `models/yolo26n-seg.pt`）、`yolo_conf`、`yolo_imgsz`、
`yolo_classes`（默认 `[32]` = sports ball）、`sam_model`（默认 `sam2.1_t.pt`）。

### 已知的坑（都是实测过的）

- **COCO 的 "sports ball" 对小目标不友好**：该类样本以足/篮/网球为主，
  小球在 640 输入下只有几个像素 → 漏检。
  实测（`yolo11n.pt` + 合成球，球半径约 25 px，80 帧）：

  | imgsz    | conf | 检出率    | 耗时             |
  | -------- | ---- | --------- | ---------------- |
  | 640      | 0.25 | 76.2%     | 7.7 ms (131 FPS) |
  | 640      | 0.10 | 92.5%     | 7.9 ms (126 FPS) |
  | **1280** | 0.25 | **97.5%** | 9.7 ms (103 FPS) |
  | 1280     | 0.10 | 97.5%     | 9.9 ms (101 FPS) |

  **结论：直接把 `yolo_imgsz` 设成 1280，几乎不花时间（103 FPS 仍然远超摄像头帧率），
  检出率从 76% 涨到 97.5%。** 640 分辨率下把 conf 降到 0.10 也行，但不如直接提分辨率。
- **COCO 词表是硬限制**：拿一个橘子（橙黄色球状水果）实测，`yolo_classes=[32]`
  一帧都检不出来，改成全类别能检出 7 个目标、其中就有那个橘子。
  也就是说"球"长什么样不是问题，**它是不是 COCO 里的 "sports ball" 才是问题**。
  对策：放宽 `yolo_classes`，或改用下面的 YOLOE 开放词汇。
- **更推荐 YOLOE-26 开放词汇模型**（ultralytics ≥ 8.4.0）：
  `YOLOE("yoloe-26n-seg.pt").set_classes(["ping pong ball"])` 直接出框 + 实例 mask，
  不受 COCO 词表限制，还能给一张参考图做视觉提示（比文字稳）。
- **SAM2 官方包只写了支持 Linux**，Windows 建议直接用 ultralytics 的 `SAM("sam2.1_t.pt")`。
  PyPI 上的 `sam2==1.1.0` **不是** Meta 官方包，别装错。
- **速度参考**：实测 `yolo11n @640` 在你的 5060 上是 **7.6 ms/帧 = 132 FPS**
  （`@1280` 是 9.7 ms = 103 FPS，含 `yolo_refine` 还要再花约 3 ms 做颜色/霍夫细化）。
  SAM2.1-tiny 按官方数据折算大约 30-45 FPS。真要 100+ FPS 且球与背景可分，还是传统 `color` 更划算
  （实测约 1 ms/帧）。
- **学习式轨迹预测（LSTM/Transformer）在这个任务上不划算**：
  单目 + 短历史（<1 s）时，物理模型（重力 + 滤波）明显优于黑盒学习模型，
  而学习模型需要海量标注轨迹。文献（Achterhold et al., L4DC 2023, arXiv:2305.15189）也是这个结论：
  **灰盒（物理 + 少量学习参数 + EKF）> 纯黑盒**。
- `boxmot` 之类的高级跟踪库要求 Python < 3.14，你的环境装不了；
  ultralytics 自带的 6 种跟踪器全是线性卡尔曼，对抛物线下落段有系统偏差 ——
  所以本项目自己做重力感知的滤波与外推。

---

## 七、代码结构

> **规则**（放置 / 命名 / 依赖方向）见 **[agent.md](agent.md)**；**目录逐项说明**见 **[docs/文件结构.md](docs/文件结构.md)**。

```text
启动程序.bat             ★ 根目录唯一入口：双击即可（找Python→查依赖→启动）
                           加参数可直接做某一步：cli/setup/check/precheck/
                           calibrate/verify/run/test
启动程序.py              图形界面（被 .bat 调用）
启动程序_命令行.py        命令行界面（被 .bat 调用；无 tkinter 时自动改用）

sphere_tracker/          单目核心库（可被任意脚本 import）
  config.py              所有可调参数（dataclass，可存取 JSON）
  types.py               Detection / TrackState / FrameResult / TrailPoint
  geometry.py            亚像素圆拟合、径向一致性、梯度校验、相机模型、单目测深
  camera_motion.py       相机自运动估计（手持/云台/平移拍摄的运动补偿基础）
  track.py               卡尔曼滤波（含机动检测与重初始化）+ 轨迹管理
  predict.py             轨迹预测：最长有效后缀拟合 + 已知重力约束
  viz.py                 拖尾、圆、预测虚线、HUD、三维小面板
  sources.py             摄像头 / 视频文件 / 图片序列（含相机模式自动探测、线程化采集）
  camera_registry.py     摄像头设备管理：枚举身份/端口、固定角色、标「无关」、解析索引、探测与自动挑优
  pipeline.py            单帧总流程编排
  detectors/             检测后端，按"线索来源"分文件
    base.py              基类 + 轮廓打分等公共工具
    factory.py           按名字造单个检测器（避开循环导入）
    color.py             颜色分割（主力）
    hough.py             霍夫圆（几何补充）
    motion.py            背景差分（含相机运动补偿）
    fusion.py            多线索投票融合
    neural.py            YOLO / SAM（可选，需要 torch）
  detect.py              兼容垫片：旧导入路径仍可用，新代码请用 detectors/

calib/                   双目标定（离线、一次）
  board.py               标定板检测与质量指标（清晰度/大小/贴边/曝光）
  capture.py             自动挑帧（含位姿多样性与画面覆盖控制）
  stereo_calib.py        标定核心（离群剔除 + 整体精化 + 质量判定）
  params.py              StereoRig：保存/加载/校正/三角测量/极线误差

stereo/                  双目（逐帧把两路合成三维）
  sync.py                双路视频同步（闪光/拍手对齐），算出帧偏移
  pair_source.py         成对帧源：按 sync_offset 取到"同一时刻"的左右两帧
  triangulate.py         极线约束配对 + 三角测量 + 三维轨迹

apps/                    可直接运行的程序
  run_realtime.py        实时摄像头主程序
  run_video.py           离线处理视频/图片，导出 CSV
  calibrate_stereo.py    自动双目标定（相机 / 两段视频 / 两个图片文件夹）
  verify_stereo.py       复检标定：装置是否被碰动
  tools/                 一次性设置与诊断工具
    cameras.py             摄像头设备管理（list / identify / pin / ignore / unignore / ignored / check）
    list_cameras.py        逐个索引试开，报分辨率/帧率/画面是否全黑
    make_calibration_board.py  生成可打印棋盘格
    calibrate_color.py     HSV 阈值交互标定
    calibrate_focal.py     单点焦距标定（三维必需）

tests/                   回归测试（改动代码后必跑，共九个）
  selftest_synthetic.py        单目全链路（clean / realistic / panning）
  selftest_stereo_calib.py     双目标定（合成真值）
  selftest_stereo_pipeline.py  同步 + 极线配对 + 三角测量
  selftest_pair_source.py      成对取帧 + 同步偏移语义（含 find_offset 串联）
  selftest_stereo_branch.py    双目整条链路（真渲染 + 两路检测 + 三角测量）
  selftest_camera_registry.py  摄像头设备管理（全注入假数据，不开相机）
  selftest_camera_motion.py    相机自运动：迟滞状态机 + 位移换算方向
  selftest_preview_lib.py      PPT 预览渲染：连线是否画出来
                               （缺 python-pptx/Pillow 时整体跳过算通过）
  selftest_camera_quality.py   相机画质会诊：蒙雾/过曝/残帧判据（全合成图）

docs/                    文档
  项目总纲.md            ★ 入口：项目是什么 + 文档怎么读
  文件结构.md            目录职责与放置规则
  交接文档.md            现状 / 代码地图 / 踩坑记录
  计划书.md              目标 / 里程碑 / 风险
  API.md                 库 API 参考
  开发指南.md            环境 / 约定 / 怎么加东西 / 方法论
  用户手册.md            面向使用者的操作手册
  PPT生成流程.md         开题报告 PPT 怎么生成 / 校验 / 预览
  开题报告_PPT逐页文稿.md
  refs/                  两篇参考文献 PDF

legacy/                  早期草稿（保留可运行，仅供对照，见其中 README.md）
  01_canny_single_image.py     原 test.py
  02_canny_realtime_edges.py   原 video_edge_detect.py

assets/                  项目素材
  boards/                可打印标定板 PDF/PNG
  images/test.jpg        示例图片（单图调试用）
  videos/                放你要处理的视频（体积大，不入库，见其中 README.md）

models/                  推理权重（*.pt 不入库，见其中 README.md）
  yolo11n.pt             --method yolo 用的 COCO 权重

config/                  本机运行期参数（标定结果、相机模式缓存，不入库）
outputs/                 运行产物（全部可重新生成，不入库）
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

1. **完全检测不到** → 跑 `apps/tools/calibrate_color.py`，先看掩膜窗口里球是不是白的。
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

这两个早期脚本现在归档在 `legacy/`（保留可运行，见 `legacy/README.md`），
文件也改成了能表明用途的名字：

| 现文件                              | 原文件名               | 问题                                                                                                            | 修复                                         |
| ----------------------------------- | ---------------------- | --------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| `legacy/02_canny_realtime_edges.py` | `video_edge_detect.py` | `OUTPUT_PATH = "../outputs/videos/..."` 会写到**项目上一级目录**，那里没有 outputs 文件夹，VideoWriter 直接失败 | 改成项目内的 `outputs/videos/`，并自动建目录 |
| `legacy/02_canny_realtime_edges.py` | `video_edge_detect.py` | XVID 编码器 + `isColor=False` 写单通道灰度图，多数机器上写出来是坏文件                                          | 转成 BGR 三通道写入                          |
| `legacy/01_canny_single_image.py`   | `test.py`              | `cv2.imread("test.jpg")` 路径错误（图片实际在 `assets/images/`），读不到会直接崩                                | 修路径并加 None 检查                         |
| `legacy/01_canny_single_image.py`   | `test.py`              | 保存结果到当前目录，和 `outputs/` 的约定不一致                                                                  | 统一存到 `outputs/images/`                   |

> 另外提醒：`assets/images/test.jpg` 的**真实格式是 PNG**（含 alpha 通道），
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

---

## 十一、相机不固定 / 换相机怎么办（运动适配）

前面几节都默认"相机架在三脚架上不动"。实际拍摄时**相机很可能在动（手持、云台、滑轨），
而且换一套设备就换一次相机**。这一节说明哪些环节会受影响、哪些不会，以及程序里已经做了什么。

### 11.1 会受影响的环节（以及现状）

| 环节                         | 相机移动时                                                                       | 现状                                                           |
| ---------------------------- | -------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| **背景差分（MOG2）**         | ❌ 最严重：相机一平移，整幅画面都变成"前景"，误检爆炸                             | ✅ **已自动切换成"对齐后再差分"**，见下                         |
| 卡尔曼跟踪                   | ⚠️ 图像坐标系里，球的运动 = 球自身 + 相机；跟踪仍成立，但输出的速度是"相对相机的" | 已可用；HUD 会显示 `cam MOVING`                                |
| 轨迹预测                     | ⚠️ 会连同相机运动一起外推                                                         | 匀速平移下**中位误差几乎不变**（实测 1.16 px vs 静止 1.06 px） |
| 像素重力约束 `g_px = fy·g/Z` | ⚠️ 抖动/加减速会污染加速度项                                                      | ✅ 相机匀速时保留，抖动时自动退回自由二次拟合                   |
| 颜色 / 梯度几何检测          | ✅ 不受影响（逐帧判据，与历史无关）                                               | 本来就是主力                                                   |
| 单目测深 `Z = fR/r`          | ✅ 不受影响（逐帧成立）                                                           | —                                                              |
| **双目标定外参 `R, T`**      | ❌ 两台相机若各自移动，相对位姿每帧都变，三角测量失效                             | **见 11.3，这是硬约束**                                        |

### 11.2 运动补偿是怎么做的

相机运动不是靠"假设它在动"，而是**每帧实测**（`sphere_tracker/camera_motion.py`）：

1. **特征点 + RANSAC 相似变换**（主力）：在背景上撒几百个点，
   运动物体只占少数，RANSAC 会把它们当外点剔除，"多数派"的运动就是相机的运动；
2. **网格点 + LK + 中位数**（画面没角点时用）：中位数天然忽略少数派的运动物体；
3. **相位相关**（最后兜底）。

> 踩过的坑：一开始用相位相关做主力，结果在**完全静止**的场景里报出 6-16 px 位移。
> 因为背景不变时，帧间唯一的变化就是那颗移动的球 —— 相位相关锁到球上去了。
> 所以现在的顺序是"多数派优先"，相位相关只兜底。

检测到相机在动之后，背景差分自动切成：
**把上一帧按估计出的运动对齐 → 有符号差分 → 剔除残影 → 在当前帧做局部 Otsu 分割出球**。

其中"剔除残影"是关键：差分图里同时有球的**旧位置**和**新位置**，
叠起来是被切过的月牙，圆度全不合格。判据是——**在当前帧里，
这块区域的内部与周围有没有明显反差**：真位置内部是球、外面是背景；残影内部早已是背景。

### 11.3 双目：相机可以整体移动，但两台相机之间必须刚性固定

这是本课题最重要的一条硬件约束，**建议在买支架/决定机位之前就确认**：

| 情况                                                | 能否做双目三角测量                                                    |
| --------------------------------------------------- | --------------------------------------------------------------------- |
| 两台相机装在同一根横梁/支架上，整个支架可以自由移动 | ✅ **可以**。`R, T` 是常数，标定一次就行；得到的三维是"相对这个支架"的 |
| 两台相机各自移动（比如两部手机各拿一个）            | ❌ **不行**。相对位姿每帧都在变，除非你能实时估计它                    |
| 一台相机 + 一个深度相机，但两者分开移动             | ❌ 同上                                                                |

补充两点：

- **整体移动时，重建出的是"相机坐标系下的轨迹"**，不是固定世界坐标系的。
  这对很多任务其实正是想要的（比如"球相对相机怎么飞"）。
  如果要固定世界系，需要做视觉里程计/SLAM 把相机自身的运动减掉，或者测量期间保持不动。
- **换相机要重新标定**：内参 `K` 和畸变 `D` 是"镜头+分辨率"的属性，
  换相机、改分辨率、变焦、对焦变化都要重标。两台不同型号是可以的（各自标各自的），
  但同型号更省事。标定流程见第五节。

### 11.4 换相机 / 换机器时的检查清单

```powershell
# 1) 先列设备身份与端口（推荐：不开相机、快、能看出插在哪个 USB 口）
python apps/tools/cameras.py list --probe

#    输出示例（名字取自系统 PnP，比注册表猜的准；编号是枚举位次，会漂移）
#    #0  aoni webcam A25      端口 6&31324e2c&0&0000   id USB\VID_1BCF&PID_2A25&MI_00\...
#    #1  aoni webcam A25      端口 6&4489382&0&0000    id USB\VID_1BCF&PID_2A25&MI_00\...
#    #2  Integrated Camera    端口 6&3a5c8ec3&1&0000   id USB\VID_30C9&PID_00EC&MI_00\...  [内置/红外,不推荐]
#    #3  Integrated IR Camera 端口 6&3a5c8ec3&1&0002   id USB\VID_30C9&PID_00EC&MI_02\...  [内置/红外,不推荐]

# 2) 逐个索引真开一遍、报画面是否全黑（老工具，仍可用）
python apps/tools/list_cameras.py
```

- **换设备后先 `python apps/tools/cameras.py check`**：比对 `config/cameras.json` 与当前设备，
  有 `[危险]`（索引被别的设备占用）时会提示重跑 `identify`。
- **红外摄像头不能用于本课题**（无彩色、光学参数不同）。
- 摄像头打开后**前几帧可能是全黑的**，程序会自动等到有真实内容的帧
  （真实传感器一定有噪声：整帧标准差恰好为 0 说明这条流是假的）。
- 不支持的分辨率会**自动降级重试**；`CAP_PROP_FPS` 返回 -1 时会**实测帧率**兜底。
- 不同相机的**色彩响应不同**，每台可能要用各自的 HSV 预设
  （`apps/tools/calibrate_color.py --camera N` 分别标定）。
- 拍摄时**锁定曝光/对焦/白平衡**，否则球在画面里移动会让自动参数来回变，颜色阈值跟着失效。

### 11.5 实测：相机匀速平移时的表现

合成场景：相机以 90 px/s 匀速横移（相当于慢速摇摄），球做斜抛 + 反弹，
其余条件与"realistic"场景相同。

| 方法                                 | 检出率 | 检测RMSE | 跟踪中位 | 半径RMSE    | 0.2s预测中位 |
| ------------------------------------ | ------ | -------- | -------- | ----------- | ------------ |
| `color`（颜色，与相机运动无关）      | 100%   | 0.78 px  | 0.69 px  | 0.23 px     | 1.16 px      |
| `motion_raw`（背景差分，**不补偿**） | 79.4%  | 0.63 px  | 0.86 px  | **1.58 px** | 1.73 px      |
| `motion`（背景差分，**已补偿**）     | 83.9%  | 0.55 px  | 0.68 px  | **0.23 px** | 1.46 px      |
| `fusion`（默认融合）                 | 100%   | 0.78 px  | 0.69 px  | 0.23 px     | 1.16 px      |

**结论**：补偿把背景差分的半径误差从 1.58 px 降到 0.23 px（约 6.9 倍），
而半径直接决定单目测深 —— 深度误差从 4.96% 降到 1.19%。
颜色 / 几何线索本来就与相机是否运动无关，所以在平移场景下几乎不掉分。

复现命令：

```powershell
python tests/selftest_synthetic.py --scenario panning      # 只看平移场景
python tests/selftest_synthetic.py --scenario panning --method motion_raw   # 看不补偿的对照
```

### 11.6 相关配置项（`DetectorConfig`）

| 参数                         | 默认     | 说明                                                            |
| ---------------------------- | -------- | --------------------------------------------------------------- |
| `motion_compensation`        | `"auto"` | `auto` 检测到相机在动就补偿；`off` 永远不补偿；`force` 永远补偿 |
| `camera_motion_threshold_px` | `1.2`    | 每帧全局位移超过多少像素判定"相机在动"（静止实测约 0.0-0.2 px） |
| `motion_work_width`          | `320`    | 估计相机运动时的缩放宽度（越小越快）                            |
| `motion_diff_min_threshold`  | `12.0`   | 帧差的最低阈值，防止把噪声当运动                                |
| `motion_contrast_min`        | `12.0`   | 判定"物体真的在这里"所需的最小内外反差                          |

---

## 十二、双目自动标定

标定是双目方案的入口：**标错了后面全错，而且在真实相机上很难发现**
（重投影误差看着正常，三维却系统性偏）。所以这一节的东西都是围绕
"自动挑好数据 + 自动验证质量"来做的。

### 12.1 三步就能用

```powershell
# 1) 两台实时相机：边看进度边挥板，收够自动标定
python apps/calibrate_stereo.py --left 0 --right 2 --cols 8 --rows 5 --square 25

# 2) 两段视频（先分别录好，再拿来标定）
python apps/calibrate_stereo.py --left-video cam1.mp4 --right-video cam2.mp4 \
    --cols 8 --rows 5 --square 25

# 3) 两个图片文件夹（每张一个位姿）
python apps/calibrate_stereo.py --left-dir imgs/left --right-dir imgs/right
```

`--cols/--rows` 是**内角点**数量（不是方格数，9×6 的棋盘内角点是 9×8），
`--square` 是方格实际边长（毫米）—— **这个值填错，基线就会等比例错**，
程序会用 `--baseline-mm` 帮你核对。

### 12.2 关键认识：标定不需要左右同步

程序只收"**板在两路里都停稳**"的帧。静止目标在任意时刻看起来都一样，
所以左右视频的时间偏移**不影响标定**。
（真正需要同步的是后续拍运动的小球，那是另一回事。）

这个判据还顺带把"人手持板走一段、停一下"的节奏变成了自动挑帧逻辑：
移动中的帧自动丢掉，停稳的帧自动收下。

### 12.3 自动挑帧的七道门限

| 门限                     | 作用                       | 默认                  |
| ------------------------ | -------------------------- | --------------------- |
| 两路都检测到             | 基本前提                   | —                     |
| 清晰度（Laplacian 方差） | 挡掉运动模糊 / 失焦        | ≥ 55                  |
| 棋盘面积占比             | 别太远也别贴脸             | 1.2% – 90%            |
| 贴边距离                 | 边缘畸变大，角点不可靠     | ≥ 10 px               |
| 曝光                     | 挡掉过暗过曝               | 35 – 230              |
| **位姿多样性**           | 别把 25 张全收成同一个角度 | 差异 ≥ 5.5% 对角线    |
| **画面覆盖**             | 每宫格限量，逼你铺满画面   | 3×3 宫格，每格 ≤ 5 对 |

> 位姿多样性是最容易被忽略、但对标定质量影响最大的一条。
> 25 张都从正面拍，标出来的焦距和距离是强耦合的 —— 看起来 RMS 很小，实际没用。

### 12.4 标定核心：稳健化才是关键

1. 各自单目标定 → `K1,D1 / K2,D2`
2. 双目联合（固定内参，只优化外参）→ `R,T`
3. **整体精化**（放开内参一起优化，RMS 通常再降一半）
4. **离群剔除 + 重算**：用"逐对视差误差"挑坏数据，丢掉后重算，最多 4 轮
5. 立体校正 → `R1,R2,P1,P2,Q`
6. 质量判定 → 重投影 RMS、极线误差、基线长度

第 4 步是拉开差距的地方：少数模糊或贴边的图会把整体误差抬高一个量级，
而它们在几百个角点里"看不出来"。

### 12.5 实测：用已知真值的合成双目装置验证

真值：基线 100 mm，fx1 = 900 / fx2 = 880（故意不同型号），含镜头畸变。

| 指标               | 理想（无噪无畸变） | 现实（畸变 + 噪声 + 轻微模糊） |
| ------------------ | ------------------ | ------------------------------ |
| 极线误差 RMS       | **0.077 px**       | **0.067 px**                   |
| 重投影 RMS         | 0.069 px           | 0.067 px                       |
| 焦距误差           | 0.094%             | 0.31%                          |
| **基线误差**       | **0.005%**         | **0.006%**                     |
| 相对旋转误差       | 0.005°             | 0.039°                         |
| **端到端三维误差** | **1.03 mm**        | **3.44 mm**（0.5–0.75 m 处）   |

复现：`python tests/selftest_stereo_calib.py`

> 最后一行才是"能不能投入生产"的真正判据：
> 用**标定出来的参数**去三角测量已知的三维点，看恢复得准不准。
> 前面的 RMS 再小，如果三维偏了也是白搭。

### 12.6 装置被碰动了怎么办

标定一次能用很久，前提是**两台相机之间的相对位置没变**。
云台转动、支架被撞、相机取下再装回，都会让 `R,T` 失效，但表面看不出异常。

```powershell
python apps/verify_stereo.py --calib config/stereo_calibration --left 0 --right 2
```

拿同一块板拍几帧，重算极线误差并与标定时记录的值比较。实测：

| 情况               | 极线误差    | 判定                 |
| ------------------ | ----------- | -------------------- |
| 装置未动           | 0.084 px    | ✅ 通过               |
| 右相机相对旋转 +2° | **27.9 px** | ❌ 不通过，需重新标定 |

差 250 倍，判定毫不含糊。**建议每次正式拍摄前跑一次**，几秒钟的事。

### 12.7 常见失败与对策

| 现象               | 原因                           | 对策                                                      |
| ------------------ | ------------------------------ | --------------------------------------------------------- |
| 收不到几对         | 板太小 / 太模糊 / 只在单路可见 | 打印更大的板、加照明、把板举到两路公共视野中间            |
| 老是"棋盘还在移动" | 板一直在动                     | 每个姿势**停稳约 1 秒**；或改用 `--left-dir` 独立静帧模式 |
| "该区域已收够"     | 一直举在同一个位置             | 按提示换位置，覆盖画面四个角和中心                        |
| 极线误差偏大       | 数据差 / 板不平 / 相机被碰     | 看剔除统计重拍；确认标定板平整；确认支架锁死              |
| 基线明显不对       | `--square` 填错                | 用尺子量方格边长，并用 `--baseline-mm` 核对               |


