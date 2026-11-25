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

    # ---- 2026-11-24 全面核对：下拉选项 / 窗口标题 / 步骤文案 / 空状态 ----
    "1 年前的": "1 year ago",
    "30 天前的": "30 days ago",
    "7 天前的": "7 days ago",
    "90 天前的": "90 days ago",
    "angled（大角度）": "angled (wide-angle)",
    "parallel（平行）": "parallel",
    "— 空": "— empty",
    "① 校验数据": "1. Validate data",
    "② 点选球心": "2. Pick ball centre",
    "③ 跑离线三维": "3. Run offline 3D",
    "④ 看三维": "4. View 3D",
    "⑤ 导出标准 JSON": "5. Export standard JSON",
    "○ 已录制": "○ Recorded",
    "◐ 已点选": "◐ Centres picked",
    "◑ 三维点过少": "◑ Too few 3D points",
    "✓ 有3D": "✓ 3D available",
    "✗ 坏": "✗ Bad",
    "全部（不按时间筛）": "All (no date filter)",
    "实时（相机）": "Live (camera)",
    "单目 · 球径先验测深": "Monocular · depth from ball diameter prior",
    "会话工作台": "Session workbench",
    "打开标注视频": "Open annotated video",
    "镜面金属球": "Mirror-finish metal ball",
    "纯色塑料球": "Solid-color plastic ball",
    "自定义": "Custom",
    "校验数据": "Validate data",
    "运行中": "Running",
    "运行中（外部）": "Running (external)",
    "新目录名": "New directory name",
    "标签（可空；会写进 session.json）": "Label (optional; saved in session.json)",
    "录多久（秒，1–600）？": "How long to record (seconds, 1–600)?",
    "输入 DELETE 确认永久删除：": "Type DELETE to confirm permanent deletion:",
    "两镜头中心距（基线 B，毫米）：":
        "Distance between camera centres (baseline B, mm):",
    "球到相机的工作距离（D，米）：":
        "Ball-to-camera working distance (D, m):",
    "（没有会话）": "(No sessions)",
    "（这个时间范围内没有可清理的东西）":
        "(Nothing to clean in this time range)",
    "（这个目录下还没有会话）": "(No sessions in this directory yet)",
    "「#编号」只用实测采集索引；角色对上不代表能出图（没开过相机），要确认点「换相机…」。":
        "Use #index only for a measured capture index; a matching role does "
        "not prove the camera can produce frames (it has never been opened). "
        "Confirm with Change camera….",
    "读两路视频，按标定解出三维轨迹，写出 traj3d.json 与 visualization.json。「球种类」决定检测方法：镜面金属球 = specular，必须先做第 ② 步点选球心；纯色塑料球 = color + 取色方式（auto / 颜色预设），球色要和背景分得开。点选球心会交给当前方法：specular 用它跟踪，color 用它取色并锚定半径。":
        "Read both videos and solve the 3D trajectory from the calibration, "
        "writing traj3d.json and visualization.json. Ball kind selects the "
        "detector: mirror metal ball = specular and requires step 2 Pick ball "
        "centre first; solid-color plastic ball = color plus a color preset, "
        "with the ball color distinct from the background. Picked centres go "
        "to the current method: specular tracks with them; color uses them for "
        "color sampling and radius anchoring.",
    "把 traj3d 写成统一的 visualization.json（含 CSV 与示意图），给别的程序读。第 ③ 步已经写过一份，这一步是重新导出。":
        "Write traj3d as the unified visualization.json (with CSV and a preview "
        "image) for other programs. Step 3 already wrote one; this step "
        "re-exports it.",
    "在这一帧的左图、右图上各点一次球心，存成 derived/seeds.json。只有金属球（specular 检测）需要它；彩色球可以跳过。":
        "Click the ball centre once on each of the left and right images; this "
        "is saved as derived/seeds.json. Only metal balls (specular detection) "
        "need it; color balls can skip it.",
    "3D 查看器读的是这段会话的 visualization.json；标注视频用系统播放器打开，用来核对轨迹有没有跳到别的物体上。":
        "The 3D viewer reads this session's visualization.json; the annotated "
        "video opens in the system player so you can check whether the track "
        "jumps to another object.",
    "先看这段会话本身有没有问题：左右帧数是否一致、同步是否可信、视频是否几乎全黑。坏数据进三维只会得到一条看起来像结果的错轨迹。":
        "First check the session itself: whether left/right frame counts match, "
        "whether sync is trustworthy, and whether the video is almost entirely "
        "black. Bad data only produces a plausible-looking wrong 3D track.",
    "球颜色不突出时 color 会 0 检出（本机实测 0/40 帧），改 hough（95%）或 fusion（82.5%）。smart = 先 color，0 检出自动换 hough 重跑一遍。":
        "When the ball color is not distinctive, color may detect 0 frames "
        "(0/40 measured here); use hough (95%) or fusion (82.5%). smart runs "
        "color first and automatically retries with hough when it finds nothing.",
    "纯色塑料球靠颜色分割：球的颜色要和背景分得开（同色背景会把整片区域当成球）。颜色预设选「自动取色」（按画面估主色，不用知道球色）或选一个贴合的球色；不选则用配置里的白球预设。":
        "Solid-color plastic balls are segmented by color: the ball color must "
        "stand out from the background (a same-color background becomes the "
        "ball). Choose Auto color in the color preset (estimates the dominant "
        "color from the frame; no need to know the ball color) or a matching "
        "ball color; otherwise the white-ball preset from config is used.",
    "自定义：检测方法直接取注册表里的实现；换一种球就换一个方法。":
        "Custom: use the detector implementation from the registry directly; "
        "change the method when you change the ball.",
    "镜面金属球只有高光能定位：先「点选球心」人工点一次球心再识别，否则自动检测找不到球（实测 288 帧里 color 落在球上 0 帧）。":
        "A mirror-finish metal ball can only be located by its highlight: click "
        "Pick ball centre once before detection, otherwise automatic detection "
        "will not find the ball (measured: color found the ball in 0 of 288 "
        "frames).",
    "session.json 里左右帧数是 0，这段会话可能是录到一半中断的":
        "session.json reports 0 frames on both sides; this session may have "
        "been interrupted mid-recording",
    "目录名（可用 / 一次建多级，例如 实验1/梨子）":
        "Directory name (use / to create multiple levels, e.g. experiment1/pears)",
    "中文": "Chinese",
    "自动取色": "Auto color",
    "红球": "Red ball", "橙球": "Orange ball", "黄球": "Yellow ball",
    "绿球": "Green ball", "青球": "Cyan ball", "蓝球": "Blue ball",
    "紫球": "Purple ball", "白球": "White ball", "黑球": "Black ball",
    "开始录制（先看状态条）": "Start recording (check the status bar first)",
    "实时 3D（先看状态条）": "Live 3D (check the status bar first)",
    "高级：棋盘尺寸与替代标定":
        "Advanced: checkerboard size and alternative calibration",
    "五步分别做什么": "What the five steps do",
    "每步在干什么、什么时候能点": "What each step does and when it becomes available",
    "内参：": "Intrinsics: ", "相机：": "Cameras: ",
    "自动挑（优先外接）": "Auto-select (prefer external)",
    "{cam}（按实测采集索引，不依赖角色绑定）｜未开相机验证":
        "{cam} (measured capture index; not role-bound) | not opened yet",
    "{cam}（读 config/cameras.json 失败）":
        "{cam} (failed to read config/cameras.json)",
    "{base}｜这台当前不在线，换「#编号」那个选项或重新 pin":
        "{base} | this camera is offline; choose a #index option or pin it again",
    "{base}｜设备还没枚举到": "{base} | device enumeration has not finished",
    "{base}｜未开相机验证": "{base} | not opened yet",
    "{role}：未固定（启动时自动挑）": "{role}: not pinned (auto-select at startup)",
    "{role}：未固定": "{role}: not pinned",
    "未知型号": "unknown model", "不在线": "offline",
    "编号 {n}": "PnP order {n}", "未确认": "unverified",
    "采集 #{n}": "capture #{n}", "采集索引未实测": "capture index not measured",
    "共 {n} 条说明": "{n} help entries",
    "搜到 {n} 条": "{n} results",
    "帧数 {left}/{right}　帧率 {fps:.1f} fps　模式 {mode}":
        "Frames {left}/{right}  ·  {fps:.1f} fps  ·  {mode} mode",
    "同步 {method}{offset:+d}（置信度 {conf:.2f}）":
        "Sync {method}{offset:+d} (confidence {conf:.2f})",
    "匹配 {matched}/{total}": "Matched {matched}/{total}",
    "残差中位 {median:.2f} px": "median residual {median:.2f} px",
    "问题：{text}": "Problems: {text}",
    "界面语言已切换为 {lang}": "Interface language switched to {lang}",
    "{v} 未绑定": "{v} is not bound",
    "{v}（{label}）采集索引未实测":
        "{v} ({label}): capture index not measured",
    "相机：{missing} —— 先跑 python apps/tools/cameras.py identify":
        "Cameras: {missing} - run python apps/tools/cameras.py identify first",
    "相机：左={left}｜右={right}": "Cameras: left={left} | right={right}",
    "相机：读取失败（config/cameras.json）":
        "Cameras: failed to read config/cameras.json",
    "相机：读取中…": "Cameras: reading…",
    "相机枚举完成：{online} 台在线{note}；「#编号」只在采集索引实测后出现，没实测先跑 cameras.py identify":
        "Camera enumeration finished: {online} online{note}; #index appears only "
        "after a measured capture index; if it is not measured, run "
        "cameras.py identify first",
    "名义 {nominal:g} mm 的球在这套检测器下约填 {effective:g} mm —— "
    "高光半径偏小 {shrink:.1f}%，填名义值会让深度偏大 {bias:.1f}%":
        "For a nominal {nominal:g} mm ball this detector wants about "
        "{effective:g} mm: the highlight radius is {shrink:.1f}% small, and "
        "entering the nominal value makes depth {bias:.1f}% too large",
    "误差预算（半径噪声 0.2 px、球径误差 0.2 mm、焦距 1%；{src}）：{parts}":
        "Error budget (radius noise 0.2 px, diameter error 0.2 mm, focal length "
        "1%; {src}): {parts}",
    "半径系统偏差：填名义直径（不是有效值）→ 深度额外 +{pct:.1f}%（1 m 处约 {mm:.0f} mm；实测）":
        "Radius systematic bias: entering the nominal diameter (not the "
        "effective value) adds {pct:.1f}% depth error (about {mm:.0f} mm at "
        "1 m; measured)",
    "球径那一栏是空的，所以算不了误差预算 —— 填毫米数，这颗球填 {effective:g}（有效值）；留空也行，识别时会按 {effective:g} mm 算":
        "The ball diameter field is empty, so the error budget cannot be "
        "computed. Enter millimetres; for this ball use {effective:g} (the "
        "effective value). Leaving it empty is allowed: detection will use "
        "{effective:g} mm.",
    "球径那一栏「{raw}」不是数字，所以算不了误差预算 —— 这里填毫米（例如 {effective:g}），不是 {metres:.4f}（米是命令行那边的单位，界面只认 mm）。改完这一行会自动重算":
        "The ball diameter field \"{raw}\" is not a number, so the error budget "
        "cannot be computed. Enter millimetres here (for example {effective:g}), "
        "not {metres:.4f} (metres are used on the command line; this window "
        "accepts mm only). This line recalculates when you change it.",
    "焦距未标定，按 60° 视场估（fx≈1663 px）":
        "Focal length is not calibrated; estimating from a 60° field of view "
        "(fx≈1663 px)",
    "按 fx = {fx:.0f} px 算": "using fx = {fx:.0f} px",
    "{secs} 秒": "{secs} s",
    "「{secs}」不是数字，按 10 秒":
        "\"{secs}\" is not a number; using 10 seconds",
    "高级：录制时长、检测器说明、会话位置":
        "Advanced: recording duration, detector notes, session location",
    "还没有选中会话。": "No session selected.",
    "还没有会话：先去双目窗录一段，或在双目窗第 ④ 步跑一次离线三维，回来这里就有历史会话可选。":
        "No sessions yet: record one in the stereo window, or run offline 3D "
        "from step 4 there; historical sessions will appear here.",
    "这段会话的数据有问题，先按上面的问题修（重录或修 session.json），别进三维。":
        "This session has data problems. Fix the issues above (re-record or "
        "repair session.json) before running 3D.",
    "这段会话没录成（左右两路都是 0 帧），直接重录即可，不用修数据。":
        "This session was not recorded (0 frames on both sides); record it "
        "again rather than repairing the data.",
    "下一步：① 校验数据；金属球（specular）要在 ② 点选球心后才能跑第 ③ 步。":
        "Next: 1) Validate data; for a metal ball (specular), pick the ball "
        "centre in step 2 before running step 3.",
    "下一步：③ 跑离线三维。": "Next: 3) Run offline 3D.",
    "三维跑出来了但匹配点太少（画不出轨迹）。下一步：确认 ② 的球心选对、换检测方法重跑 ③；④ 仍可打开查看器看那些散点落在哪。":
        "3D finished but too few points matched (no track can be drawn). Next: "
        "check the centres picked in step 2 and rerun step 3 with another "
        "detector; step 4 can still open the viewer to see where those points "
        "land.",
    "下一步：④ 看三维；⑤ 可重新导出标准 JSON。":
        "Next: 4) View 3D; 5) re-export the standard JSON if needed.",
    "点选球心 seeds.json": "Ball centres seeds.json",
    "轨迹 traj3d.json": "Trajectory traj3d.json",
    "三维 visualization.json": "3D visualization.json",
    "标注视频 annotated.mp4": "Annotated video annotated.mp4",
    "缺视频帧数": "Missing frame counts",
    "未同步（sync=none）": "Not synchronized (sync=none)",
    "视频几乎全黑": "Video is almost entirely black",
    "三维轨迹没有匹配点": "3D trajectory has no matched points",
    "尚未跑离线双目": "Offline stereo has not been run",
    "未点选球心 seed": "Ball-centre seed not picked",
    "缺 session.json": "Missing session.json",
    "session.json 不是对象": "session.json is not an object",
    "traj3d.csv 读不了": "Cannot read traj3d.csv",
    "录制未完成：左右两路都是 0 帧（视频只有文件头）":
        "Recording did not complete: 0 frames on both sides (video has only a header)",
    "左右帧数差 {diff}（超过 {tol}）：有一路明显掉帧，三维会断续":
        "Left/right frame difference {diff} (over {tol}): one side dropped "
        "frames; 3D will be discontinuous",
    "左右帧数差 {diff} 帧（管线按短的那路跑，可继续）":
        "Left/right frame difference {diff} frames (pipeline uses the shorter "
        "side and can continue)",
    "session.json 读不了：{exc}": "Cannot read session.json: {exc}",
    "traj3d.json 读不了：{exc}": "Cannot read traj3d.json: {exc}",
    "traj3d.csv 读不了：{exc}": "Cannot read traj3d.csv: {exc}",
    "质检结论 {verdict}：录像几何与标定不一致，结果仅供参考":
        "Quality verdict {verdict}: recording geometry does not match the "
        "calibration; results are for reference only",
    "质检结论 {verdict}：结果仅供参考":
        "Quality verdict {verdict}: results are for reference only",
    "三维匹配点过少：{matched}/{total}（{percent:.1f}%，低于 {threshold:g}%），轨迹画不出来；检查球心是否选对 / 换检测方法重跑 / 标定是否匹配":
        "Too few 3D matches: {matched}/{total} ({percent:.1f}%, below "
        "{threshold:g}%); no track can be drawn. Check the picked ball "
        "centres, rerun with another detector, and verify the calibration.",
    "跑过三维，但缺 traj3d.json / traj3d.csv（只剩可视化或标注视频），匹配率无法判定；建议用同一份 seeds 重跑一次离线三维":
        "3D was run but traj3d.json / traj3d.csv are missing (only "
        "visualization/annotated video remain); the match rate cannot be "
        "determined. Rerun offline 3D with the same seeds.",
    # ---- 2026-11-24：弹窗专项（装置档案 / FOV / 工具台 / 垃圾清理）----
    "采集档": "Capture profile", "标定实测": "Calibration measurements",
    "—（按标定）": "— (from calibration)",
    "当前装置：{name}": "Active rig: {name}",
    "当前装置：{name}　模式：{mode}": "Active rig: {name}  ·  mode: {mode}",
    "尺度 left→right": "Scale left→right", "等效放大": "Effective zoom",
    "裁剪矩形": "Crop rectangle", "共同输出尺寸": "Common output size",
    "测量时分辨率": "Resolution at measurement", "估计质量": "Estimated quality",
    "测量时间": "Measured at", "来源": "Source",
    "大角度（angled）": "Wide-angle (angled)",
    "平行（parallel）": "Parallel (parallel)",
    "世界 Z 轴": "World Z axis", "标定文件": "Calibration file",
    "只统计不写文件（先看有多少可用帧）":
        "Count only, write no files (check usable frames first)",
    "当前：{path}": "Current: {path}",
    "未登记（由启动器/安装器决定）": "not registered (set by launcher/installer)",
    "还没有装置档案 —— 点「新建装置…」建一条（如「平行装置」「90度装置」），各配一份标定":
        "No rig profiles yet - click New rig… to create one (for example "
        "Parallel rig or 90° rig), each with its own calibration",
    "项": "Item", "档": "Tier", "体积": "Size", "最新的一个": "Newest",
    "文件": "Files", "推荐": "Recommended", "谨慎": "Caution",
    "彻底删": "Permanent delete", "（不动）": " (muted)",
    "今天": "today", "昨天": "yesterday",
    "{n} 天前": "{n} days ago", "{n} 个月前": "{n} months ago",
    "{n} 年前": "{n} years ago",
    "Python 字节码缓存": "Python bytecode cache",
    "临时文件残留": "Temporary file leftovers",
    "会话元数据备份": "Session metadata backups",
    "回收站（data/_trash）": "Recycle bin (data/_trash)",
    "outputs/logs（旧日志位置）": "outputs/logs (old log location)",
    "outputs 根下的临时文件": "Temporary files at the outputs root",
    "outputs 根下的其他散落文件": "Other loose files at the outputs root",
    "未标定": "Not calibrated", "已标定": "Calibrated",
    "路径不符": "Path mismatch", "模式冲突": "Mode conflict",
    "全部会话": "All sessions",
    "自动核对：配置读不出来": "Auto-check: cannot read the configuration",
    "自动核对：": "Auto-check: ",
    "未配置": "not configured",
    "{role}={label} 一致": "{role}={label}: match",
    "{role}={label} 采集索引未实测": "{role}={label}: capture index not measured",
    "{role}={label} 不在线/身份变了": "{role}={label}: offline or identity changed",
    "已选 {chosen} 项 / {files} 个文件 / {size}　｜　推荐档 {safe_size}（{safe_n} 项）　谨慎档 {warn_size}（{warn_n} 项）":
        "Selected {chosen} items / {files} files / {size}  |  Recommended: "
        "{safe_size} ({safe_n} items)  Caution: {warn_size} ({warn_n} items)",
    "，其中 {n} 项是「看起来像数据」被自动降级的":
        "; {n} of them were automatically downgraded as \"looks like data\"",
    "　｜　另有 {size}（{files} 个文件）比所选时间范围新，已筛掉":
        "  |  {size} ({files} files) are newer than the selected time range and were filtered out",
    "找不到的位置：{text}": "Missing locations: {text}",
    "选中的「回收站」是彻底删除，删了不可恢复。":
        "The selected Recycle bin entries are permanently deleted and cannot be recovered.",
    "「推荐」= 删了下次跑就有；「谨慎」= 删了要重跑/重拍才能再生；右键某行可「标为不动」。":
        "Recommended = regenerated on the next run; Caution = needs a rerun or "
        "reshoot; right-click a row to mark it as muted.",
    "还原": "Restore", "全屏": "Fullscreen",
    "sessions 根目录：{path}": "sessions root: {path}",
    "回收站：{path}": "Recycle bin: {path}",
    "整个会话（含原始视频）": "Whole session (including raw video)",
    "只清派生结果（derived/）": "Derived results only (derived/)",
    "清理方式": "Cleanup mode", "保留最新": "Keep newest",
    "个（按会话名倒序；0=不保留）":
        " (newest first by session name; 0 = keep none)",
    "预览失败：{exc}": "Preview failed: {exc}",
    "{sessions} 个会话 / {files} 个文件 / {size}":
        "{sessions} sessions / {files} files / {size}",
    "将移动 {summary}；跳过 {n} 项":
        "Will move {summary}; skipping {n} items",
    "清理会话（先移到回收站）": "Clean sessions (move to recycle bin first)",
    "回收站": "Recycle bin",
    "全部会话 ({n})": "All sessions ({n})",
    "平行双目": "Parallel stereo", "大角度双目": "Wide-angle stereo",
    "散会话": "Loose sessions",
    "已选 {chosen} / {total} 个会话\n预计原始数据 {size}":
        "Selected {chosen} / {total} sessions\nEstimated raw data: {size}",
    "选": "Select", "时间": "Time", "大小": "Size",
    "会话元数据（session.json + 时间戳）":
        "Session metadata (session.json + timestamps)",
    "标准派生数据（seeds / traj3d / visualization / CSV）":
        "Standard derived data (seeds / traj3d / visualization / CSV)",
    "派生图（traj3d.png / sync_brightness.png）":
        "Derived plots (traj3d.png / sync_brightness.png)",
    "标注视频（有画面，含隐私风险）":
        "Annotated video (contains imagery; privacy risk)",
    "原始双目视频（很大，有画面，含隐私风险）":
        "Raw stereo videos (large, contain imagery; privacy risk)",
    "该会话的标定参数（JSON/NPZ）": "This session's calibration parameters (JSON/NPZ)",
    "移除相机 identity/name/port": "Remove camera identity/name/port",
    "移除会话 label/notes": "Remove session label/notes",
    "移除 OS platform": "Remove OS platform",
    "批次": "Batch", "方式": "Mode", "项数": "Items",
    # ---- 2026-11-24：3D 查看器 ----
    "[单目]": "[Mono]", "[双目]": "[Stereo]",
    "[降级]": "[Degraded]",
    "3D 轨迹 · {kind}{session} · {n} points":
        "3D trajectory · {kind}{session} · {n} points",
    "共 {n} 个会话有结果（单目 + 双目）":
        "{n} sessions have results (mono + stereo)",
    "没有找到会话结果（data/stereo、data/mono 下都没有 derived/）":
        "No session results found (no derived/ under data/stereo or data/mono)",
    "这段会话没有标注视频": "This session has no annotated video",
    "用系统播放器打开视频（{reason}）":
        "Open video in system player ({reason})",
    "视频不是左右并排（{w}x{h}），按 2:1 切分会错":
        "Video is not side-by-side ({w}x{h}); slicing at 2:1 would be wrong",
    "单目深度由球径先验得到（D={d} mm，Z=f·D/(2r)），尺度依赖这个数；不是视差测量":
        "Monocular depth comes from the ball-diameter prior (D={d} mm, "
        "Z=f·D/(2r)); its scale depends on this value and is not a disparity "
        "measurement",
    "XY 平面": "XY plane", "XZ 平面": "XZ plane", "YZ 平面": "YZ plane",
    "没有可用的三维点": "no usable 3D points",
    "没有三维结果：{reason}{extra}   可查看会话信息 / 视频，若有":
        "No 3D result: {reason}{extra}   You can still view session info / video, if available",
    "   会话：{path}": "   Session: {path}",
    "   （有视频，可在下方查看）": "   (has video; see below)",
    "frame {frame}   t={t:.3f}s   X={x:.4f}  Y={y:.4f}  Z={z:.4f} m   "
    "模式={mode}   坐标={coord}   视角 {elev:.0f}/{azim:.0f}/{roll:.0f}°":
        "frame {frame}   t={t:.3f}s   X={x:.4f}  Y={y:.4f}  Z={z:.4f} m   "
        "mode={mode}   coordinates={coord}   view {elev:.0f}/{azim:.0f}/{roll:.0f}°",

    # ---- 2026-11-24：状态条 / 弹窗动态句（整句模板，数据保持原文）----
    "内参标定失败（返回码 {code}）：{tail} —— 常见原因是标定图不够或姿态太单一（要 8 个以上不同角度）；{log}":
        "Intrinsic calibration failed (exit code {code}): {tail} - usually too "
        "few images or poses (need at least 8 different angles); {log}",
    "已启动实时识别（球径按 {d} mm 传给命令行 {m} m），画面在它自己的窗口里；要停请到「任务管理器和运行日志」里中断":
        "Live detection started (diameter {d} mm sent to the command line as "
        "{m} m); video is in its own window. Stop it from Task manager and run log.",
    "种子已存：{path}（球心 x,y 半径 = {value}）。点「开始识别」就按它跟踪，本次用检测器 {det}（{how}）{pref}":
        "Seed saved: {path} (ball centre x,y radius = {value}). Click Start "
        "detection to track from it; this run uses detector {det} ({how}){pref}",
    "种子是在 {w}x{h} 上点的，这段录像是 {w2}x{h2}，坐标对不上，退回自动检测":
        "The seed was picked at {w}x{h} but this recording is {w2}x{h2}; "
        "coordinates do not match, so automatic detection is used",
    "已启动离线单目：{path}（检测器 {method}）；结果写进 {derived}/（跑完摘要会变成「已检测」，那时才能点「看 3D」）":
        "Offline monocular started: {path} (detector {method}); results go to "
        "{derived}/ (the summary becomes Detected when done, then View 3D is enabled)",
    "「{title}」失败（返回码 {code}）：{tail} —— {log}":
        "\"{title}\" failed (exit code {code}): {tail} - {log}",
    "「{title}」{reason}（返回码 {code}），没跑完就没结果 —— {log}":
        "\"{title}\" {reason} (exit code {code}); no result without a full run - {log}",
    "已打开 3D 查看器：{path}（单目深度由球径先验得到，D={d} mm；不是视差测量）":
        "3D viewer opened: {path} (monocular depth comes from the ball-diameter "
        "prior, D={d} mm; not a disparity measurement)",
    "{id}：结果已写进 derived/（annotated.mp4 + traj.csv + visualization.json）；点「看 3D」看图":
        "{id}: results written to derived/ (annotated.mp4 + traj.csv + "
        "visualization.json); click View 3D to see them",
    "录制没成功（返回码 {code}）：{tail} —— {log}":
        "Recording failed (exit code {code}): {tail} - {log}",
    "已开始录制 {secs} 秒 → {path}/（有倒计时，看到 REC 才开始动）。录完点「开始识别」直接检测这段会话":
        "Recording started for {secs} s → {path}/ (wait for the countdown and "
        "REC). When done, click Start detection to process this session.",
    "旧默认球直径 {d} mm 已清空 —— 球径现在没有默认值，请按实际球的直径（毫米）重填；单目测深 Z 对它是 1:1 的":
        "The old default ball diameter {d} mm was cleared - there is no default "
        "now. Enter the real diameter in millimetres; monocular depth Z scales "
        "1:1 with it.",
    "把整个会话目录 {id} 移到回收站吗？":
        "Move the whole session directory {id} to the recycle bin?",
    "「{title}」完成（返回码 0）：{tail} —— {log}":
        "\"{title}\" completed (exit code 0): {tail} - {log}",
    "新的会话会建在 {root}/<时间戳>/，以前录的还在原位置（点「选择…」只看这个目录下的）":
        "New sessions go to {root}/<timestamp>/; older recordings stay where "
        "they are (Select… only lists this directory).",
    "试开结束，但日志尾既没「打不开」也没「实测帧率」：{tail} —— 开日志看完整输出 {log}":
        "The test open ended without either Cannot open or a measured frame "
        "rate: {tail} - open the log for the full output {log}",
    "离线检测失败（返回码 {code}）：{tail} —— {log}":
        "Offline detection failed (exit code {code}): {tail} - {log}",
    "已开始拍图：板停稳会自动存，存够 {n} 张结束。本次图单独存 {dir}":
        "Capture started: steady frames are saved automatically until {n} "
        "images. This set is stored separately in {dir}",
    "已移到回收站：{path}/{name}（要恢复请用数据管理里的回收站）":
        "Moved to the recycle bin: {path}/{name} (restore it from Data management → Recycle bin)",
    "录好了：{summary}{note}。点「开始识别」（来源「离线」）开始检测，结果写进 derived/":
        "Recorded: {summary}{note}. Click Start detection (source Offline) to "
        "process it; results go to derived/.",
    "。注意：角色 {cam} 指向的相机当前不在线，录制多半开不出画面 —— 换「#编号」那个选项更稳":
        ". Note: the camera for role {cam} is offline; recording will probably "
        "produce no frames - choose a #index option instead.",
    "当前会话换成 {id}（{state}）；「开始识别」（来源「离线」）会直接处理它{note}":
        "Current session is now {id} ({state}); Start detection (source "
        "Offline) will process it directly{note}",
    "上次选的相机角色 {saved} 当前不在线，已改成 {default}（pin 好以后可以再选回来）":
        "The previously selected camera role {saved} is offline; switched to "
        "{default} (pin it again to use it later)",
    "试开结果：这台打不开 —— {tail} —— {log}":
        "Test open: this camera cannot open - {tail} - {log}",
    "因球种需要，本次临时用检测器 {chosen} （你的偏好 {method} 不变，下次打开仍按偏好）":
        "This run temporarily uses detector {chosen} for the ball kind (your "
        "preference {method} is unchanged for next time)",
    "建不了会话目录（{root}）：{exc}":
        "Cannot create the session directory ({root}): {exc}",
    "已换到 {root}；这里列出的是新目录下的会话":
        "Switched to {root}; the sessions listed here come from the new directory",
    "球径那一栏「{raw}」不是有效的毫米数 —— 这里填毫米（例如 25），不是 0.025":
        "The ball diameter field \"{raw}\" is not a valid millimetre value - "
        "enter millimetres here (for example 25), not 0.025",
    "当前会话 {id}：目录已就绪，还没录到视频（录制会写 clip.mkv）":
        "Current session {id}: directory is ready, but no video yet "
        "(recording writes clip.mkv)",
    "（设备列表读不到：{exc}，只能按角色选）":
        "(Cannot read the device list: {exc}; only role selection is available)",
    "种子文件读不了（{exc}），退回自动检测":
        "Cannot read the seed file ({exc}); falling back to automatic detection",
    "内参标定完成，已重新读取标定值（第①步那行会跟着变）；{note}":
        "Intrinsic calibration complete; calibration values were reloaded "
        "(the line in step 1 updates too); {note}",
    "（session.json 没写成：{exc}；视频本身不受影响）":
        "(session.json was not written: {exc}; the video itself is unaffected)",
    "fx 那一栏「{raw}」不是数字，已忽略，仍用{tail}":
        "The fx field \"{raw}\" is not a number and was ignored; still using {tail}",
    "。用你点的种子跟踪（{seed}），不再自动检测":
        ". Tracking from your picked seed ({seed}); automatic detection is disabled.",
    "当前会话 {id} 有问题：{error}":
        "Current session {id} has a problem: {error}",
    "当前会话有问题：{error}{note}":
        "Current session has a problem: {error}{note}",
    "；你的偏好 {pref} 不变，下次打开仍按偏好":
        "; your preference {pref} is unchanged and will be used next time",
    "日志 {path}": "log {path}",
    "标定值 fx={value}": "calibrated fx={value}",
    "。注意：{note}": ". Note: {note}",
    "质量状态：{grade}": "Quality: {grade}",
    "没移成：{exc}": "Move failed: {exc}",
    "已停掉本窗口的 {n} 个任务": "Stopped {n} tasks from this window",
    "有 {n} 个任务在运行，其中可能有相机或标定程序。\n是 = 中断它们并退出；否 = 保持窗口打开、任务继续运行。\n\n如果你只是想收起某个子窗口，请关那个子窗口，不要关这里。":
        "There are {n} running tasks, possibly cameras or calibration jobs.\n"
        "Yes = stop them and exit; No = keep this window open and let them run.\n\n"
        "If you only want to hide a child window, close that window instead of this one.",
    "恢复批次？\n{name}": "Restore this batch?\n{name}",
    "已恢复 {n} 个目录。": "Restored {n} directories.",
    "这会永久删除：\n{path}\n不可恢复。继续？":
        "This permanently deletes:\n{path}\nIt cannot be recovered. Continue?",
    "批次：{batch}\n\n可在数据管理的『回收站』恢复或彻底删除。":
        "Batch: {batch}\n\nRestore or permanently delete it from Data management → Recycle bin.",
    "没有删除。": "Nothing was deleted.",
    "先选一个批次。": "Select a batch first.",
    "没有可移动的目录。": "There are no directories to move.",

    # ---- 2026-11-24：双目/工作台/数据管理的常用动态句 ----
    "球心已保存：{file}；现在跑第 ④ 步会用当前球种类的方法解算（镜面球 specular / 纯色球 color）。想同时用「半径锚点门」，把它勾上再跑（不勾就只是普通点选）":
        "Ball centres saved: {file}. Step 4 now uses the current ball kind "
        "(specular for metal / color for solid colors). To also use the "
        "radius-anchor gate, enable it before running (otherwise this is a "
        "plain centre pick).",
    "两台实际档不一致：{a}={aw}x{ah}，{b}={bw}x{bh}；双路必须同档，未填入。请把两台设成同一个模式后再试":
        "The two actual modes differ: {a}={aw}x{ah}, {b}={bw}x{bh}; both "
        "cameras must use the same mode. Set them to the same mode and retry.",
    "正在检查装置 {left} / {right}（只报告，不改配置）：会开两路取景窗，举着棋盘慢慢移动，按 q 可提前结束。结论会写回这里":
        "Checking rig {left} / {right} (report only, no config changes): two "
        "preview windows open; move the board slowly, press q to finish early. "
        "The conclusion is written here.",
    "正在按 {mode} 解算 {path}（检测方法 {method}）{tail}…":
        "Solving {path} as {mode} (detector {method}){tail}…",
    "正在标定会话 {path} 的图…": "Calibrating images from session {path}…",
    "标定完成。{note}{tail}": "Calibration complete. {note}{tail}",
    "重建完成{note}": "Reconstruction complete{note}",
    "已选会话：{name}": "Selected session: {name}",
    "先选一个会话（点「选择…」）": "Select a session first (click Select…)",
    "这不是会话目录（缺 session.json）：{name}":
        "This is not a session directory (missing session.json): {name}",
    "会话目录不见了：{path}": "Session directory disappeared: {path}",
    "正在解算 {id}（检测方法 {method}）；跑完 ④ 的按钮会自动亮起":
        "Solving {id} (detector {method}); the step 4 buttons enable when done",
    "球心已保存：{file}；现在可以跑第 ③ 步":
        "Ball centres saved: {file}; step 3 can run now",
    "正在打开 3D 查看器：{name}": "Opening 3D viewer: {name}",
    "这个平台不能直接开视频，文件在：{path}":
        "This platform cannot open video directly; file: {path}",
    "已用系统播放器打开 {name}": "Opened {name} in the system player",
    "打不开 {name}：{exc}": "Cannot open {name}: {exc}",
    "正在校验这段会话，逐项结果会写到运行日志":
        "Validating this session; each result goes to the run log",
    "数据校验失败（没通过）：看运行日志里它列的每一项问题":
        "Data validation failed: see each problem listed in the run log",
    "导出标准数据 {id}": "Export standard data {id}",
    "一键全流程 {id}": "One-click full pipeline {id}",
    "校验数据 {id}": "Validate data {id}",
    "离线三维 {id}": "Offline 3D {id}",
    "正在导出 visualization.json / CSV / 示意图":
        "Exporting visualization.json / CSV / preview image",
    "导出失败：看运行日志里它的报错":
        "Export failed: see the error in the run log",
    "当前：{path}": "Current: {path}",
    "已更新 {id}（备份：{name}）": "Updated {id} (backup: {name})",
    "已更新 {id}（备份：未找到）": "Updated {id} (backup: not found)",
    "完成：{name}  {size}": "Completed: {name}  {size}",
    "失败：{exc}": "Failed: {exc}",
    "正在导出，请先等上一个任务结束":
        "Export in progress; wait for the previous task to finish",
    "正在取消导出，等待当前文件收尾…":
        "Cancelling export; waiting for the current file to finish…",
    "导出中…": "Exporting…",
    "先在列表里选一个会话": "Select a session in the list first",
    "至少选一个导出类别": "Select at least one export category",
    "先勾选至少一个会话": "Select at least one session first",
    "先勾选或选中至少一个会话": "Select at least one session first",
    "先只选一条会话": "Select exactly one session first",
    "先在右边勾选会话": "Select sessions on the right first",
    "先点一行作为起点": "Click a row to set the start first",
    "先点一行作为终点": "Click a row to set the end first",
    "先点第一列勾起点，或先按『设为起点』":
        "Check the start in the first column, or click Set as start first",
    "先选一个目录。": "Select a directory first.",
    "选一个自己建的目录再改名。": "Select a directory you created before renaming.",
    "新目录名": "New directory name",
    "目录名不能为空。": "The directory name cannot be empty.",
    "目录名不能以 / 结尾。": "The directory name cannot end with /.",
    "计划开始时间解析失败：{value}": "Cannot parse the start time: {value}",
    "计划结束时间解析失败：{value}": "Cannot parse the end time: {value}",
    "填一个开始或结束时间，例如 2026-10-20。":
        "Enter a start or end time, for example 2026-10-20.",
    "开始时间晚于结束时间。": "The start time is later than the end time.",
    "不是目录：{root}": "Not a directory: {root}",
    "先在列表里选中一条会话。": "Select a session in the list first.",
    "先只选一条会话。": "Select exactly one session first.",
    "输出路径": "Output path", "导出": "Export", "导出完成": "Export complete",
    "导出失败": "Export failed", "扫描失败": "Scan failed",
    "删除目录": "Delete directory", "新建目录": "New directory",
    "改名": "Rename", "放入目录": "Add to directory",
    "移出目录": "Remove from directory", "3D + 视频": "3D + video",
    "打开会话文件夹": "Open session folder",
    "编辑名称/备注": "Edit name/notes",
    "按时间选择": "Select by time", "设为起点": "Set as start",
    "选到此处": "Select up to here",
    "数据目录": "Data directory",
    "基本设置（常用）": "Basic settings (common)",
    "高级设置（精度 / 排障）": "Advanced settings (precision / troubleshooting)",
    "高级（仅单目实时 / 离线视频）":
        "Advanced (live monocular / offline video only)",
    "点选双目球心": "Pick stereo ball centres",
    "点选单目球心": "Pick mono ball centre",
    "左路": "Left view", "右路": "Right view",
    "左：还没点": "Left: not picked yet",
    "等待点击": "Waiting for a click",
    "应用半径": "Apply radius",
    "约 Z = —（还没点球边）": "approx. Z = — (ball edge not picked yet)",
    "约 Z = —（{why}）": "approx. Z = — ({why})",
    "（无时间）": "(no time)",
    "相机 {camera}": "Camera {camera}",
    "{n} 帧": "{n} frames",
    "已录，未检测": "Recorded, not detected",
    "已检测": "Detected",
    "未录制": "Not recorded", "坏会话": "Bad session",
    "还没有当前会话：按「录制视频」会自动新建一个目录；要接着看以前录的，点「选择…」挑一段。":
        "No current session yet: Record video creates a directory automatically; "
        "to continue an old recording, click Select… and choose one.",
    "当前会话 {id} 有问题：{error}":
        "Current session {id} has a problem: {error}",
    "当前会话 {id}：目录已就绪，还没录到视频（录制会写 clip.mkv）":
        "Current session {id}: the directory is ready, but no video yet "
        "(recording writes clip.mkv)",
    "当前会话 {summary}；结果在 derived/，重新检测会覆盖同名文件":
        "Current session {summary}; results are in derived/ and a new run "
        "overwrites files with the same name",
    "当前会话 {summary}；点「开始识别」（来源「离线」）开始检测（结果写进 derived/）":
        "Current session {summary}; click Start detection (source Offline) to "
        "process it (results go to derived/)",
    "吸附Hough圆心": "Snap to Hough centre",
    "在左图点球心，再在右图点球心；圆心=你点击的位置，需要时再点『吸附Hough圆心』；半径可微调。":
        "Click the ball centre in the left image, then in the right image. "
        "The centre is where you click; use Snap to Hough centre if needed; "
        "the radius can be fine-tuned.",
    "拍标定图用棋盘：A4 打印一张，在画面里上下左右都摆一摆，至少 8 个不同姿态（挤在中央会让畸变系数失真）。":
        "For calibration images, print an A4 checkerboard and move it around "
        "the frame: at least 8 different poses (keeping it in the centre "
        "distorts the distortion coefficients).",
    "内角点 = 棋盘格内部交叉点个数，不是格子数。手上那张纸写着 8×5 就填 8 和 5；方格边长按实际量。":
        "Inner corners = the number of internal checkerboard intersections, "
        "not squares. If the printed board says 8×5, enter 8 and 5; measure the "
        "square size with a ruler.",
    "替代方案用的是第②步填的球直径：同一颗球、同一个 D，两处同源，所以不叠加；但换了球一定要改第②步那一栏，否则 fx 会按旧球径算错（f = 2·r·Z/D，D 错多少，fx 就错多少）。":
        "The alternative uses the ball diameter from step 2: the same ball and "
        "the same D are shared, so the error does not add; but when you change "
        "balls you must update step 2, otherwise fx is computed from the old "
        "diameter (f = 2·r·Z/D, so a wrong D gives a wrong fx).",
    "球是镜面的（镀铬、反光）时，自动检测会把书架/衣服当球：先录一段，再点「点选球心」——第一点球心、第二点球边（半径不靠自动估），之后「开始识别」用它跟踪，检测器会自动切成 specular。换会话存放目录的入口在「选择…」弹窗里。":
        "With a mirror-finish (chrome, reflective) ball, automatic detection "
        "mistakes shelves and clothes for the ball: record a clip, then click "
        "Pick ball centre - first the centre, then the edge (the radius is not "
        "estimated automatically). Start detection then tracks from it and the "
        "detector switches to specular. The session-folder picker is in Select….",
    "球种类决定检测方法：镜面金属球 = specular，必须先在「点选球心」里人工点一次球心（没有种子多半 0 检出）；纯色塑料球 = color + 颜色预设，球色要和背景分得开；自定义 = 直接用注册表里的某个方法。":
        "Ball kind selects the detector: mirror metal ball = specular, which "
        "requires picking the centre once under Pick ball centre (without a "
        "seed it usually detects 0 frames); solid-color plastic ball = color "
        "plus a color preset, with the ball color distinct from the background; "
        "Custom = use a detector from the registry directly.",
    "「离线」处理当前会话的录像：结果写进 derived/，跑完「看 3D」才会亮；「实时」直接开相机，不落盘，也没有 3D 结果（单目三维靠球径先验，需要落盘的轨迹）。":
        "Offline processes the current session's recording: results go to "
        "derived/ and View 3D enables when done. Live opens the camera directly, "
        "saves nothing, and has no 3D result (monocular 3D needs a saved track "
        "and the ball-diameter prior).",
    # 2026-11-24 自动绑定功能（基线代码已有、词典漏登记的 9 条）：
    # 英文模式下按钮/状态条全是中文，渲染级判据实测抓到，补整句译文。
    "自动绑定左右": "Auto-bind left / right",
    "画面是左": "This feed is left",
    "画面是右": "This feed is right",
    "自动绑定正在运行，请稍候…": "Auto-bind is running, please wait…",
    "还没拿到实时画面，不能确认采集索引；等画面出来后再点":
        "No live frame yet — cannot confirm the capture index. "
        "Click again once the feed appears",
    "正在自动绑定左右相机并实测采集索引（请稍候，约 10–30 秒）…":
        "Auto-binding left/right cameras and measuring capture indices "
        "(please wait, about 10–30 s)…",
    "自动绑定未完成：{err}": "Auto-bind did not finish: {err}",
    "自动绑定完成：left=采集#{l}，right=采集#{r}":
        "Auto-bind complete: left=capture #{l}, right=capture #{r}",
    "已实测确认 {role} 采集索引 = #{index}（写入 {path}）":
        "Confirmed {role} capture index = #{index} (saved to {path})",

    # ---- 2026-11-24 大角度双目全链路与工具台补充文案 ----
    "① 拍 A 板：正对相机 1（板放稳不动，满 5 张自动收工）":
        "1. Capture Board A: facing camera 1 (keep still; finishes after 5 steady shots)",
    "① 拍左路多姿态内参图（板随便挪，四角+中央各多拍几张）":
        "1. Capture left multi-pose intrinsics (move across corners & center)",
    "② 拍右路多姿态内参图（板随便挪，四角+中央各多拍几张）":
        "2. Capture right multi-pose intrinsics (move across corners & center)",
    "② 绕中轴转 90°，拍 B 板：正对相机 2（满 5 张自动收工）":
        "2. Rotate 90° around mid-axis, capture Board B: facing camera 2 (5 steady shots)",
    "④同框相机": "Pair camera (4)",
    "⑧ 复检双目标定（检查装置是否被碰动）":
        "8. Verify stereo calibration (check if rig moved)",
    "「拍地面棋盘图」已经在跑了 —— 请先在已开的取景窗内完成拍摄":
        "Capture floor board image is already running — finish capture in the open preview window",
    "两步正对图都拍完了。点「检查链式输入」，再点「开始标定」":
        "Both facing poses captured. Click Check chain inputs, then Start calibration",
    "内参文件/目录": "Intrinsics file/dir",
    "切换": "Switch",
    "双目使用一段时间后，用同一块棋盘拍几帧检验极线误差是否漂移；零共视 ~90° 装置若两路无法同时见板，会给出重拍参考对重标的说明。":
        "After extended use, capture a few frames of the same checkerboard to verify epipolar drift; "
        "for zero-overlap ~90° rigs that cannot share a board view, guidance is shown to reshoot reference pairs.",
    "固定主点（少姿态防漂）": "Fix principal point (prevent drift with fewer poses)",
    "复检标定": "Verify calibration",
    "大角度/链式标定不做固有视场裁剪（两路保留完整原始画面）":
        "Angled / chained calibration does not apply FOV cropping (keeps full raw frames on both views)",
    "大角度检查未通过：请确保两路分辨率一致，且检查时棋盘轮流出现在左、右两路视野内（详情看运行日志）":
        "Angled rig check failed: ensure both cameras match resolution and show the checkerboard to left then right in turn (see run log)",
    "大角度装置检查通过：两路分辨率一致且都能单独看清棋盘（日志里有推荐的标定方法）":
        "Angled rig check passed: matching resolution and clear board detection on both views (see run log for method advice)",
    "广角8参数畸变": "Wide-angle 8-param distortion",
    "开始复检": "Start verification",
    "开始重拍选中步": "Reshoot selected step",
    "当前会话没有可恢复的回收批次": "No recoverable trash batches for the current session",
    "恢复旧图": "Restore old images",
    "恢复该批": "Restore batch",
    "打开目录": "Open folder",
    "拍双路内参…": "Capture dual intrinsics…",
    "标定复检结束（详情见运行日志）": "Calibration verification finished (see run log for details)",
    "标定复检被中断了（未完成采样）": "Calibration verification was aborted before collecting enough samples",
    "档案已记录 FOV（裁 {side}{size_txt}）；不勾 = 按原始画面标定":
        "Profile has recorded FOV (crop {side}{size_txt}); uncheck to calibrate on raw frames",
    "检查中轴 90° 正对参考图（左 A 板 / 右 B 板是否静止且完整）…":
        "Checking mid-axis 90° facing reference images (Board A left / Board B right stationary & complete)…",
    "正在复检当前标定（会打开取景窗，举棋盘采集样本，按 q 可提前结束）…":
        "Verifying current calibration (preview window opening; hold checkerboard in view, press q to finish early)…",
    "没有拍下地面棋盘图（请看底部日志确认相机或棋盘是否完整入画）":
        "Failed to capture floor checkerboard image (check run log for camera or board detection)",
    "用左相机拍一张…": "Capture with left camera…",
    "结果写入底部任务日志": "Results are written to the bottom task log",
    "还没有当前标定文件，请先完成第 ② 步「开始标定」":
        "No calibration file yet — complete Step 2 Start calibration first",
    "选一步重拍：该步旧图不删除，会移到 outputs/_trash/ 下的批次。下方可直接切换历史会话、在资源管理器中删掉个别糊图，或恢复之前被移走的旧图批次。":
        "Choose a step to reshoot: existing images are moved safely to a batch under outputs/_trash/ rather than deleted. "
        "Below you can also switch past sessions, open the folder in Explorer to prune blurry frames, or restore a previous trash batch.",
    "选择标定文件（留空 = 当前装置档案）":
        "Select calibration file (leave blank = active rig profile)",
    "选目录…": "Select folder…",
    "链式输入自检已中止": "Chain input check was aborted",
    "链式输入自检通过，但有警告（请在底部日志查看坏帧/数量提示）":
        "Chain input check passed with warnings (see bottom run log for frame quality/count hints)",
    "链式输入自检通过，可以点「开始标定」":
        "Chain input check passed — ready to click Start calibration",
}

#: 纯标点的显示替换。它们没有中文，不能放进 `SEED`（种子词典的契约是
#: “key 含中文”）；但状态条拼接动态句时仍需要按当前语言取分隔符。
PUNCT = {"；": "; ", "，": ", ", "｜": " | "}


#: 这些字符串即使含中文也不该被短语替换（专名 / 公式 / 单位 / 标点片段）。
#: 运行时只做"整句精确匹配"时它们本来就不会命中；这里主要防自动组词误伤。
NO_AUTOFILL = (
    "Z = f·D/(2r)", "X 右 / Y 下 / Z 前", "sm_120", "cu128",
)

__all__ = ["SEED", "PUNCT", "NO_AUTOFILL"]
