"""GUI 文案的**人工种子词典**：中文片段/整句 → 英文。

这是少量**固定词条的精确译文**。完整界面句子在 `apps/i18n_data.py` 中维护；
运行时**不做**逐词/子串替换 —— 那样的输出会把中文和英文拼成
`Record与识别`、`Calibration堆栈`，比保留中文更难读。

新增界面文案由 `apps/tools/i18n_extract.py --write` 登记，再补完整英文句子；
本词典只用于确实稳定的短词条（确定 / 取消 / 保存等）。

约定（改词典前先读）：
- key 必须是**界面上真实出现**的中文片段，不要写整篇译文式的长句；
- 单字 key 尽量少放（"左"/"右" 这类容易在别处误伤），优先放双字以上；
- 带前后空格的片段（如 " 个任务"）是为了拼接句，保留空格；
- value 里不要再出现中文，否则运行时判断"是否译完"会失灵；
- 专有名词保持项目口径：装置=Rig、会话=Session、标定=Calibration。

代码注释、日志格式串、变量名、文件路径**不在这里**，也不应该进来。
"""

from __future__ import annotations

SEED: dict[str, str] = {
    # ---- 通用动作 / 按钮 ----
    "确定": "OK", "取消": "Cancel", "确认": "Confirm", "保存": "Save",
    "删除": "Delete", "刷新": "Refresh", "关闭": "Close", "退出": "Exit",
    "全屏": "Fullscreen", "搜索": "Search", "开始": "Start", "停止": "Stop",
    "暂停": "Pause", "继续": "Continue", "重试": "Retry", "选择": "Select",
    "选择…": "Browse…", "打开": "Open", "导出": "Export", "导入": "Import",
    "新建": "New", "编辑": "Edit", "编辑…": "Edit…", "添加": "Add",
    "移除": "Remove", "已移除": "Removed", "保留": "Keep", "清理": "Clean up",
    "全选": "Select all", "返回": "Back", "下一步": "Next", "完成": "Done",
    "已完成": "Completed", "应用": "Apply", "重置": "Reset", "复制": "Copy",
    "查看": "View", "帮助": "Help", "关于": "About", "设置": "Settings",
    "管理": "Manage", "运行": "Run", "执行": "Run", "中断": "Abort",
    "跳过": "Skip", "进入": "Open", "工具": "Tools", "工具条": "Toolbar",
    "选到此处": "Select up to here", "按时间选择": "Select by time",
    "用基线算夹角": "Angle from baseline", "重新读取标定": "Reload calibration",
    "是": "Yes", "否": "No", "有": "Yes", "无": "No", "全部": "All",
    "未选择": "Nothing selected", "不使用": "None", "左右": "left/right",

    # ---- 状态 / 结果 ----
    "就绪": "Ready", "运行中": "Running", "检查中…": "Checking…",
    "通过": "Pass", "不通过": "Fail", "失败": "Failed", "警告": "Warning",
    "错误": "Error", "已保存": "Saved", "不存在": "Missing",
    "不匹配": "Mismatch", "匹配": "Match", "未记录": "Not recorded",
    "未知名": "Unknown", "参数不对": "Invalid parameters",
    "先选一个任务": "Select a task first",
    "先选一个装置档案": "Select a rig profile first",
    "[失败] ": "[Failed] ", "[错误] ": "[Error] ",
    "[警告] ": "[Warning] ", "[跳过] ": "[Skipped] ",
    "[任务 ": "[Task ", "任务 ": "Task ", " 个任务": " tasks",
    " 个文件 / ": " files / ", " 个": " ", " 张": " images",
    " 帧": " frames", "共 ": "Total ", "已选 ": "Selected ",
    "已确认停掉 ": "Stopped ", "读不了 ": "Cannot read ",
    "打不开 ": "Cannot open ", "（返回码 ": "(exit code ",
    "（读不到：": "(unreadable: ", "，耗时 ": ", elapsed ",
    "，实际 ": ", actual ", "，基线 ": ", baseline ",
    "本次日志：": "Log: ", "下一步：": "Next: ",
    "，缺 ": ", missing ", "缺 ": "Missing ",

    # ---- 相机 / 标定 ----
    "相机": "Camera", "相机设备": "Camera devices",
    "相机设备管理": "Camera devices", "相机设置": "Camera setup",
    "相机检查": "Camera check", "左目": "Left camera", "右目": "Right camera",
    "左": "Left", "右": "Right", "双目": "Stereo", "单目": "Monocular",
    "平行": "Parallel", "大角度": "Angled", "装置": "Rig",
    "装置档案": "Rig profiles", "当前装置：": "Active rig: ",
    "装置：": "Rig: ", "标定": "Calibration", "标定前置检查": "Calibration precheck",
    "双目标定": "Stereo calibration", "标定复检": "Calibration recheck",
    "内参：": "Intrinsics: ", "内参": "Intrinsics", "外参": "Extrinsics",
    "基线": "Baseline", "视差": "Disparity", "焦距": "Focal length",
    "棋盘": "Checkerboard", "内角点·列": "Inner corners · cols",
    "内角点·行": "Inner corners · rows", "方格边长 mm": "Square size (mm)",
    "目标对数": "Target pairs", "最少对数": "Min pairs",
    "停稳上限 px": "Stable threshold (px)", "帧率": "Frame rate",
    "分辨率": "Resolution", "曝光": "Exposure", "画质诊断": "Quality diagnostics",
    "[链式标定] ": "[Chained calibration] ",

    # ---- 会话 / 数据 / 检测 ----
    "会话": "Session", "当前会话 ": "Current session ",
    "会话目录": "Session folder", "历史会话": "Session history",
    "数据管理": "Data manager", "数据管理 · 脱敏导出": "Data manager · anonymized export",
    "操作说明": "Help", "3D 查看器": "3D viewer",
    "打开 3D 查看器": "Open 3D viewer", "垃圾清理": "Junk cleanup",
    "移到回收站": "Move to Recycle Bin", "回收站": "Recycle Bin",
    "任务管理器和运行日志": "Tasks & logs", "任务": "Task",
    "运行日志": "Run log", "状态": "Status", "模式": "Mode",
    "检测方法": "Detector", "颜色预设": "Color preset",
    "点选球心": "Pick ball center", "球心已保存：": "Ball center saved: ",
    "球直径(米)": "Ball diameter (m)", "预测视野(秒)": "Prediction horizon (s)",
    "帧": " frames", "半径": "Radius", "球": "Ball",
    "三维": "3D", "坐标": "Coordinates", "轨迹": "Trajectory",
    "可视化 JSON：": "Visualization JSON: ", "3D + 视频": "3D + video",
    "导出 JSON": "Export JSON", "导出标准数据 ": "Export standard data ",
    "校验数据 ": "Validate data ", "校验数据": "Validate data",
    "① 校验数据": "1. Validate data", "② 点选球心": "2. Pick ball center",
    "跑离线三维": "Run offline 3D", "录制双目": "Record stereo",
    "录制": "Record", "实时摄像头检测": "Live camera detection",
    "离线视频分析": "Offline video analysis", "实时检测": "Live detection",
    "开始识别": "Start detection", "双目工作台": "Stereo workbench",
    "回归测试": "Regression tests", "规范自查": "Standards check",
    "一键启动": "One-click launch", "经典界面（已废止）": "Classic UI (retired)",
    "经典界面已废止，不再启动。\n\n"
    "所有功能都在新界面里：门厅 → 单目 / 平行双目 / 大角度双目；\n"
    "数据管理、3D 查看器、相机设备在门厅下方工具条；\n"
    "标定板、取色、世界系、导出与训练在「标定与训练…」里。\n\n"
    "命令行 `run.bat legacy` 与 `--legacy` 也已关闭，只会打印这条说明。":
        "The classic UI is retired and will not start.\n\n"
        "Everything lives in the new UI: lobby → monocular / parallel stereo / "
        "angled stereo;\n"
        "Data manager, 3D viewer and Camera devices are in the lobby toolbar;\n"
        "Calibration board, color picking, world frame, export and training are "
        "under Calibration & Training...\n\n"
        "The `run.bat legacy` and `--legacy` commands are closed too and only "
        "print this notice.",

    # ---- 门厅 / 帮助 ----
    "球形物体三维运动轨迹识别": "3D Sphere Motion Trajectory Recognition",
    "单目 · 一台相机": "Monocular · one camera",
    "平行双目 · 两路光轴近平行": "Parallel stereo · nearly parallel axes",
    "大角度双目 · 接近直角": "Angled stereo · near right angle",
    "快速上手": "Quick start",
    "三步走：选装置 → 标定 → 录制": "Three steps: rig → calibration → recording",
    "先选装置类型，再按窗口里的三步走；参数只在用到它的那一步出现":
        "Pick a rig first, then follow the three steps; parameters appear only where needed.",
    "（窗口内「数据源」可换任意会话）": "(Use Data Source in the window to switch sessions)",
    "按回车返回主菜单…": "Press Enter to return to the main menu…",
    "按回车继续…": "Press Enter to continue…",
    "输入编号并回车 > ": "Enter a number and press Enter > ",
    "按回车返回": "Press Enter to return",

    # ---- 操作说明：分区与条目标题（窗口里最显眼的文字） ----
    "主界面": "Main window", "常见问题": "FAQ",
    "先搞清楚坐标系：X 右 / Y 下 / Z 前":
        "Coordinate system: X right / Y down / Z forward",
    "三种装置怎么选": "Which rig should I use",
    "门厅三张卡在说什么": "What the three cards mean",
    "工具条各按钮": "Toolbar buttons",
    "装置档案里的「采集档」为什么必须填":
        "Why capture settings are required in a rig profile",
    "录制前会体检什么": "Pre-recording checks",
    "两路相机同时打开失败": "When two cameras cannot open together",
    "录多长、帧率多少": "Clip length and frame rate",
    "标定板参数怎么填": "Checkerboard parameters",
    "joint（两路共视）和链式（单板）怎么选":
        "Joint vs. chained calibration",
    "拍标定图的要领": "Tips for capturing calibration images",
    "鼠标怎么用": "Mouse controls",
    "视角预设：为什么左视右视横轴是 Z":
        "View presets: why Z is the horizontal axis",
    "「最佳视角」按钮做了什么": "What Best view does",
    "曲线上的小圆点是采样点": "Dots on the curve are real samples",
    "左上角「坐标极值」和分度值": "Coordinate extremes and axis steps",
    "播放器：逐帧步进": "Player: frame stepping",
    "视频面板：两格时间和帧号": "Video panel: two timestamps and frame number",
    "状态符号什么意思": "Status icons explained",
    "怎么选一段区间": "Selecting a time range",
    "导出类别怎么勾": "Choosing export categories",
    "曲线跑到框外 / 背景不跟着动":
        "Curve leaves the box / background does not follow",
    "图太小、四周留白很多": "Plot too small with large margins",
    "点「3D 查看器」打开的却是旧会话": "3D viewer opens an old session",
    "标定做完了但录制说尺寸不对":
        "Calibration done but recording reports wrong size",
    "单目和双目的 Z 能直接比吗":
        "Can monocular and stereo Z be compared directly",
    "还不能用：": "Not ready: ", "正在打开 3D 查看器：": "Opening 3D viewer: ",
    "已运行": "Finished", "当前会话": "Current session",

    # ---- 2026-11-07：补「标定与训练工具」用得上的常用词 ----
    # 加这些不是为了那一个对话框，是因为它们**到处都在用**（输入框标签、
    # 按钮、状态条），而词典里一条都没有 —— 英文模式下这些词会原样留中文，
    # 拼出来的句子就变成 "Start训练" 这种中英混排。
    # 注意：自动组词是"按 key 长度降序做短语替换"，短词先放会被长词截走，
    # 所以这里一律放**双字以上的名词/动词**，不加单字。
    "输出": "Output", "目录": "Directory", "设备": "Device",
    "训练": "Train", "数据集": "Dataset", "标定板": "Calibration board",
    "浏览": "Browse", "生成": "Generate", "计算": "Compute",
    "格式": "Format", "备注": "Note", "纸张": "Paper",
    "图片": "Image", "轮数": "Epochs", "尺寸": "Size", "批": "Batch",
    "验证集": "Validation set", "标注": "Annotation",
    # 整词优先于逐字拼：这几条在按钮上反复出现，拼错了很显眼
    "开始训练": "Start training", "生成标定板": "Generate board",
    "打开取色窗口": "Open color picker", "导出数据集": "Export dataset",
    "计算并保存": "Compute and save", "只算不写": "Dry run",
    "只统计不写文件": "Count only, write nothing",
    # 整句标签：自动组词是"逐词替换、不加空格"，所以「输出目录」会拼成
    # OutputDirectory、「验证集会话数」会拼成 Validation setSession数。
    # 界面上真实出现的**标签整句**必须单独登记，拼出来的英文才读得通。
    "输出目录": "Output directory", "标定目录": "Calibration directory",
    "数据集目录": "Dataset directory", "保存到": "Save to",
    "棋盘（内角点）": "Checkerboard (inner corners)",
    "地面棋盘图": "Ground checkerboard image",
    "从图片取色": "Pick color from image", "或从相机取": "Or from camera",
    "只导这一段": "Only this session",
    "验证集会话数": "Validation sessions",
    "训练参数": "Training parameters", "列数": "Columns", "行数": "Rows",
    "边长 mm": "Square size mm", "批大小": "Batch size",
    # ---- 2026-11-07 傍晚之后：分区可拖动 / 错误摘要 ----
    "相机预览详细原因": "Camera preview error details",
    # ---- 2026-11-08：界面回调异常留痕（pythonw 没有 stderr）----
    "界面出过一次错（已自动恢复）：详情见 outputs/logs/gui_errors.log":
        "The interface hit an error (recovered automatically): "
        "see outputs/logs/gui_errors.log",
    "重试这一路": "Retry this camera",
    "用 DSHOW 试开": "Try again with DSHOW",
    # ---- 2026-11-07 傍晚之后：多路预览 ----
    "预览（可同时开多路）": "Preview (multiple cameras at once)",
    "预览全部外接": "Preview all external",
    "停止全部预览": "Stop all previews", "停止预览": "Stop preview",
    "正在打开…": "Opening…", "等待画面…": "Waiting for frames…",
    "预览中": "Live",
    "按画面认清哪台是左/右，然后选中它点「设为左 / 设为右」。预览可以同时开多路，用来并排比对；绑定按设备身份记，换 USB 口也不会漂。":
        "Identify left/right by the picture, then select a device and click Set as left / Set as right. "
        "Previews can run side by side for several cameras at once. Roles are bound to device identity, "
        "so swapping USB ports will not shift them.",
    "这台还没出画面：慢相机可能要 20–30 秒；若一直不出，检查是否被别的程序占着，或拔插一次":
        "No frame yet: a slow camera can take 20-30 seconds. If it stays blank, check whether another "
        "app is using it, or unplug and plug it back in.",
    # ---- 2026-11-07 傍晚：GUI 内相机管理对话框的短标签 ----
    "相机管理": "Camera Manager", "设备": "Device",
    "USB 端口": "USB port", "当前角色": "Current role",
    "预览这台": "Preview this", "设为左": "Set as left",
    "设为右": "Set as right", "设为主相机": "Set as main",
    "标为无关": "Mark unused", "取消无关": "Unmark",
    "刷新列表": "Refresh list", "关闭": "Close",
    "没有预览": "No preview", "读取设备中…": "Reading devices…",
    "正在打开画面…": "Opening preview…",
    "预览（选中一台后点「预览这台」）":
        "Preview (select a device, then click Preview this)",
    "（收够就停；板摆稳、清晰就自动收）":
        "(Stops when enough pairs are collected; sharp, steady board shots are saved automatically)",
    # ---- 2026-11-07 下午：平行窗恢复「目标对数」----
    # 逐词拼会读成 "Target对数 张" 这种混排，整句登记才说得通。
    # 注意：这一轮一度写过「五个角都要拍到，5/5 全绿」，用户明确要求去掉
    # 这个限制，那句与它的译文都已删除 —— 别再往种子词典里放回来。
    "目标对数 张": "Target pairs",
    # ---- 2026-11-07：工具台六张卡的**说明长句** ----
    # 为什么这些也进种子：自动组词是"逐词替换"，一段 60 字的中文说明会被拼成
    # 几十个英文单词硬凑成的一句话 —— 每个词各自对，整句读不懂。说明文字是
    # 用户第一次打开工具台时唯一能读到的东西，宁可人工写三句。
    # （其余三张卡的说明全由上面的常用词拼得出，已经读过，不再单列。）
    ("第一次用、或者手里那块板是别人给的，先在这里生成一份。"
     "打印时必须选「实际大小 / 100%」，不要选「适应页面」——"
     "打完拿尺子量方格边长，量出来是多少，标定时「方格边长」就填多少。"):
        "First time here, or the board in your hand came from someone else: "
        "generate one here. When printing you must pick \"Actual size / 100%\", "
        "never \"Fit to page\" — then measure the square edge with a ruler and "
        "enter that measured value as the square size when calibrating.",
    # 2026-11-20：这句改过一次 —— 原文写的是"让 Y 轴真的指向「上」"，与实现不符
    # （世界系的 Z 才是棋盘/地面法向；`-z` 时 Z 朝上、Y 反而朝上）。中英文一起换掉。
    ("三维轨迹默认落在相机系，只能看形状；要让 Z 轴指向地面法向"
     "（朝上还是朝下由下面的下拉框决定），就得拍一张棋盘平放在地面的左相机图，"
     "在这里算一次 T_world_cam。这是一次性离线标定：拍一张、算一次、落盘，"
     "运行期只读 config/world_frame.json。"
     "注意：Z 朝上（-z）时 Y 会同时反向，这是保持右手系的代价。"):
        "By default the 3D track lives in the camera frame, so only its shape "
        "is meaningful. To make +Z follow the ground normal (up or down is "
        "chosen by the drop-down below), take one left-camera photo with the "
        "checkerboard lying flat on the ground and compute T_world_cam here. "
        "This is a one-off offline calibration: shoot once, compute once, "
        "save it; at run time only config/world_frame.json is read. Note: "
        "choosing Z up (-z) flips Y as well — that is the price of keeping "
        "the frame right-handed.",
    ("镜面金属球、花色水果这类「经典方法全灭」的目标，最后一步是让它学球长什么样。"
     "数据来自上面⑤导出的数据集。验收不看 mAP，看圆心误差 —— "
     "检测结果只是种子，后面跟踪、测深、预测全从它出发，"
     "圆心错 10 px 半径再准也没用。"
     "CPU 上很慢（几十轮可能几小时），有独显就把设备填 cuda:0。"):
        "For specular metal balls, patterned fruit and other targets where all "
        "classic methods fail, the last resort is to let a model learn what "
        "the ball looks like. The data comes from the dataset exported in "
        "step 5 above. Acceptance is not mAP but centre error: the detection "
        "is only a seed for tracking, depth and prediction, so a 10 px centre "
        "error makes even an accurate radius worthless. CPU is very slow "
        "(dozens of epochs may take hours); with a discrete GPU set device to "
        "cuda:0.",

    # ---- 相机管理窗口：预览失败结论 + 动态状态句（2026-11-24）----
    # 这些都是"整句模板"（含 {name}/{path}/{n} 等占位符），运行时由
    # i18n.tf() 先翻模板再填数据。数据（设备名/路径/数值）绝不翻译。
    "采集会话失效（不是相机坏）：先把这台 USB 拔插一次，再点「重试这一路」":
        "Capture session invalidated (the camera is not broken): unplug and "
        "replug this USB device once, then click Retry this feed",
    "打不开：先等 30–60 秒再点「重试这一路」（上一路刚用过要放手）；再查占用（Windows 相机 / QQ）；还不行拔插这台":
        "Cannot open: wait 30–60 s, then click Retry this feed (the previous "
        "feed needs time to release); check for conflicts (Windows Camera / "
        "QQ); if still stuck, replug this camera",
    "打开了但没画面：先拔插这台，再查镜头遮挡/占用":
        "Open but no image: replug this camera first, then check lens "
        "covering / conflicts",
    "为什么这里不切换后端：DSHOW 的编号与这份设备列表不是同一套，按同一个编号打开可能开到别的设备。所以预览只用系统默认后端（MSMF）。":
        "Why the backend is not switched here: DSHOW indices do not match "
        "this device list; opening the same number may reach a different "
        "camera. Preview therefore uses only the system default backend (MSMF).",
    "下一步：① 先点卡片上的「重试这一路」；② 还不行就拔插一次这台相机的 USB，并关掉可能占着相机的程序（微信 / QQ / 钉钉 / OBS / Windows 相机等）；③ 所有相机（含内置）都打不开，先查 设置 → 隐私和安全性 → 相机 → 允许桌面应用访问相机；④ 仍打不开就跑 run.bat doctor 或 python apps/tools/cameras.py doctor —— 它只报告，不改配置、也不绑左右角色。":
        "Next steps: 1) click Retry this feed on the card; 2) if that fails, "
        "replug this camera's USB and close apps that may hold it (WeChat / "
        "QQ / DingTalk / OBS / Windows Camera, etc.); 3) if no camera opens "
        "(including built-in), check Settings → Privacy & security → Camera "
        "→ allow desktop apps; 4) if still stuck run run.bat doctor or python "
        "apps/tools/cameras.py doctor - it only reports, changing no config "
        "and binding no roles.",
    "未实测": "Not measured",
    "\n\n现在开着的、可能占着相机的程序：{hogs}":
        "\n\nApps currently running that may hold the camera: {hogs}",
    "｜现在开着 {hogs}，可能占着相机": " | {hogs} is running and may hold the camera",
    "｜两台都这样＝系统的相机服务卡住了：重启电脑比拔插更彻底":
        " | both feeds like this = the system camera service is stuck: "
        "restarting the PC is more thorough than replugging",
    "（全文：{log}）": "(full text: {log})",
    "（双击状态行看全文）": "(double-click the status line for full text)",
    "相机 #{index}：{brief}{note}{tail}": "Camera #{index}: {brief}{note}{tail}",
    "{stage} 已等 {secs} 秒": "{stage}: waited {secs} s",
    "还没出画面：慢相机可能要 20–30 秒；一直不出就检查占用或拔插一次":
        "No image yet: a slow camera may need 20–30 s; if it never shows, "
        "check conflicts or replug once",
    "预览中": "Previewing",
    "画面显示不出来：{exc}": "Cannot display the image: {exc}",
    "#{index} 出图了；总共 {n} 路画面在同时跑":
        "#{index} is showing frames; {n} feeds running in total",
    "界面刷新出过错（{n} 次，已自动恢复；详情见 outputs/logs/camera_preview.log）":
        "UI refresh errored ({n} times, auto-recovered; see "
        "outputs/logs/camera_preview.log)",
    "⚠ 界面代码在 {stamp} 又改过（{n} 个文件），当前窗口还是旧版 —— **关掉程序重开**才会生效":
        "Warning: UI code changed again at {stamp} ({n} files); this window "
        "is still the old version - close and reopen the app to apply",
    "#{index} 已经在预览了；要重开先点卡片上的「重试这一路」":
        "#{index} is already previewing; to restart, click Retry this feed on the card",
    "#{index}「{name}」是虚拟摄像头，不参与绑定、也不在这里预览（它只注册在别的后端，PnP 列表里没有它）":
        "#{index} '{name}' is a virtual camera - not used for binding or "
        "preview here (it is only registered under another backend, absent "
        "from the PnP list)",
    "提示：#{index} 被当前相机规则标为内置/不推荐；要看外接相机请选择未标记的设备":
        "Note: #{index} is marked built-in/not recommended by the current "
        "camera rules; to see external cameras choose an unmarked device",
    "正在打开 #{index}（首帧最坏约 30 秒）。可以同时再开别的相机，彼此不打断。":
        "Opening #{index} (worst-case first frame about 30 s). You can open "
        "other cameras at the same time; they do not interrupt each other.",
    "已并行打开 {n} 路预览；每路互不干扰，慢的那路继续等首帧，不会拖住别的路":
        "Opened {n} previews in parallel; each is independent, the slow one "
        "keeps waiting for its first frame without holding up the others",
    "没保存：{exc}": "Not saved: {exc}",
    "门限没保存：{exc}": "Thresholds not saved: {exc}",
    "自动刷新：{n} 台｜{check}": "Auto-refresh: {n} cameras | {check}",
    "已写入 {path}{extra}": "Written to {path}{extra}",
    "已写入 {path}。筛选门限从下一次自动挑相机/成对协商起生效":
        "Written to {path}. Filter thresholds take effect from the next "
        "auto camera selection / pair negotiation",
    "#{index}「{name}」是虚拟摄像头，不能设成 {role} —— 它不是物理设备（PnP 列表里没有它），请选一台真实的相机":
        "#{index} '{name}' is a virtual camera and cannot be set as {role} - "
        "it is not a physical device (absent from the PnP list); choose a "
        "real camera",

    # ---- 2026-11-24 第二轮截图反馈：Treeview 单元格 + hint() 工厂文案 ----
    "(没名字)": "(unnamed)",
    "内置": "Built-in",
    "无关": "Ignored",
    "虚拟(不绑定)": "Virtual (not bound)",
    "检查中…": "Checking…",
    "读取相机…": "Reading cameras…",
    "正在读取设备列表（只枚举，不开相机）…":
        "Reading device list (enumeration only, no camera opened)…",
    "已停止全部预览": "All previews stopped",
    "没找到外接相机（列表里只有内置/红外/虚拟，或已标无关）":
        "No external camera found (the list only has built-in/IR/virtual or "
        "marked-ignored devices)",
    "相机管理已经开着了：同一个程序只留一个，避免两处同时抢相机":
        "The camera manager is already open: only one per app, to avoid two "
        "windows competing for the same camera",
    "设备列表是空的：先点「刷新列表」":
        "The device list is empty: click Refresh list first",
    "设备列表是空的：先点「刷新列表」；还是空的就查 USB":
        "The device list is empty: click Refresh list first; if still empty "
        "check the USB connection",
    "「换目录…」平时不用动：只有会话根目录被移走、或录像改放到别的磁盘时，才用它把上面的列表指到新位置。它只改「以后再录的存哪」，不搬动已有会话，也不修复已经丢了的目录。":
        "Change folder… is normally untouched: only use it when the session "
        "root was moved or recordings now live on another disk, to point the "
        "list above to the new location. It only changes where future "
        "recordings go; it does not move existing sessions or repair missing "
        "directories.",
    "「移到回收站」是软删除：整个会话（含原始视频）进 data/_trash，之后能在数据管理里恢复。「改装置模式…」会把会话归到对应装置分区（data/stereo/parallel 或 angled）；「整理旧会话…」一次性把**已记录模式**的旧会话搬进分区（没记录的那些不会动，需要你逐条判断）。":
        "Move to recycle bin is a soft delete: the whole session (including "
        "raw video) goes to data/_trash and can be restored from data "
        "management. Change rig mode files the session under the matching rig "
        "partition (data/stereo/parallel or angled); Tidy old sessions moves "
        "old sessions with a recorded mode into partitions in one pass (those "
        "without a recorded mode are left alone for you to judge one by one).",
    "产物：<输出目录>/*.pdf 与 *.png": "Output: <output dir>/*.pdf and *.png",
    "产物：config/world_frame.json": "Output: config/world_frame.json",
    "产物：images/ labels/ dataset.yaml": "Output: images/ labels/ dataset.yaml",
    "产物：权重 + 逐图圆心误差报告": "Output: weights + per-image centre-error report",
    "指针：config/python_env.txt；切换需重启 GUI/launcher":
        "Pointer: config/python_env.txt; restart the GUI/launcher to switch",
    "每种格式各三份：left_camera / right_camera / stereo_rig":
        "Three copies per format: left_camera / right_camera / stereo_rig",
    "这是个独立窗口：调完按 s 保存，按 q 退出":
        "This is a standalone window: press s to save when done, q to quit",

    # ---- 2026-11-24 第二轮：双目窗说明段落 + 动态状态行模板 ----
    "平行双目的前提：两路光轴**近平行**（夹角很小），且基线远小于工作距离。校正后左右图的行是对齐的。":
        "Prerequisite for parallel stereo: the two optical axes are nearly "
        "parallel (tiny angle) and the baseline is far smaller than the "
        "working distance. After rectification the rows of the left/right "
        "images line up.",
    "大角度双目的前提：两路夹角大（接近直角），不必平行、也不必在每个姿态都看到同一块板。深度靠原始像素三角测量。":
        "Prerequisite for wide-angle stereo: the two axes form a large angle "
        "(near a right angle), need not be parallel, and need not both see "
        "the same board at every pose. Depth comes from raw-pixel triangulation.",
    "这一步决定第 4 步能不能用。平行装置只需要 joint；如果两路看不到同一区域，就该去「大角度双目」窗口。":
        "This step decides whether step 4 can be used. A parallel rig only "
        "needs joint; if the two feeds cannot see the same area, use the "
        "Wide-angle Stereo window.",
    "按「两路还能看到多少共同画面」选方法：还能充分共视 → 两路共视；只剩一处能同时看到整块板 → 单板链式；完全看不到同一区域 → 双板链式（要两块刚性固定的板）。链式产物都直接能在第 4 步用。":
        "Choose by how much the two feeds still overlap: ample common view -> "
        "joint capture; only one spot where both see the whole board -> "
        "single-board chain; no common view at all -> dual-board chain (two "
        "rigidly fixed boards). Chain outputs are directly usable in step 4.",
    "离线重建：读两路视频按标定解三维，强制按 {mode} 解算；3D 查看器同屏加载 derived/annotated.mp4，也能用系统播放器单独打开。":
        "Offline reconstruction: read both videos and solve 3D from the "
        "calibration, forcing {mode} solving; the 3D viewer loads "
        "derived/annotated.mp4 side by side, and it can also be opened "
        "standalone in a system player.",
    "高级：采集门限与相机": "Advanced: capture thresholds and cameras",
    "停稳 / 清晰度 / 放宽 / 相机": "Steady / sharpness / relaxed / camera",
    "半径 = 点选球边得到的像素圆半径（seeds.json），和下面两个门限都不是一回事。":
        "Radius = the pixel radius picked on the ball edge (seeds.json); it "
        "is neither of the two thresholds below.",
    "半径锚定门：用左右点选球心 + 物理球直径校验视半径，须先勾「半径锚点门」且点过球心。":
        "Radius-anchor gate: uses the picked centres in both views plus the "
        "physical ball diameter to check the apparent radius; requires the "
        "radius-anchor gate enabled and a centre already picked.",
    "尺寸门自动阶梯：按 4 档上限逐档试探找最大候选；留空像素门限就走它，物理直径不控制尺寸门。":
        "Size-gate auto ladder: probes 4 upper limits to find the largest "
        "candidate; leaving the pixel threshold empty uses it. The physical "
        "diameter does not control the size gate.",
    "装置档案：{prof}": "Rig profile: {prof}",
    "未设置": "not set",
    "标定模式：{mode}": "Calibration mode: {mode}",
    "基线 {base:.0f} mm": "baseline {base:.0f} mm",
    "夹角 {angle:.1f}°": "angle {angle:.1f}°",
    "尚无标定": "no calibration yet",
    "当前已有标定（{mode}）": "existing calibration ({mode})",
    "标定：{existing}。选好方法点「开始标定」即可产出本窗口需要的标定":
        "Calibration: {existing}. Pick a method and click Start calibration "
        "to produce the calibration this window needs",
    "相机：左={left}｜右={right}": "Cameras: left={left} | right={right}",
    "相机：读取失败（config/cameras.json）":
        "Cameras: read failed (config/cameras.json)",
    "第③步录制会用不了：{why}": "Step 3 recording will not work: {why}",
    "高级：半径与直径": "Advanced: radius and diameter",
    "像素门限留空=自动阶梯": "empty pixel threshold = auto ladder",
    "选择会话 · {kind}": "Choose session · {kind}",
    "平行双目 · 视差测深": "Parallel stereo · disparity depth",
    "大角度双目 · 三角测量": "Wide-angle stereo · triangulation",
}

#: 这些字符串即使含中文也不该被短语替换（专名 / 公式 / 单位 / 标点片段）。
#: 运行时只做"整句精确匹配"时它们本来就不会命中；这里主要防自动组词误伤。
NO_AUTOFILL = (
    "Z = f·D/(2r)", "X 右 / Y 下 / Z 前", "sm_120", "cu128",
)

__all__ = ["SEED", "NO_AUTOFILL"]
