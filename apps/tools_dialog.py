"""「标定与训练」工具台：把只能在命令行里走的六条流程搬到界面上。

为什么要有它（2026-11-07 盘点）
--------------------------------
逐条比对 `apps/*.py` 与 `apps/tools/*.py` 的入口后发现，有六条**用户流程**
只能靠敲 python 命令完成，而它们都被 README 写成"下一步做什么"：

    生成标定板       tools/make_calibration_board.py   没有板就没法开始标定
    点选取色         tools/calibrate_color.py          对准真实小球的第一步
    世界坐标系       tools/set_world_frame.py          三维轨迹要落到地面系
    导出标定参数     tools/export_calibration.py       换机器复现 / 交给别人
    导出训练数据集   tools/export_ball_dataset.py      「以后不用手点球心」第一步
    训练球检测器     tools/train_ball_detector.py      通用球类识别的那条路

用户走到这几步就断了：界面上一个入口都没有，只能回头翻文档找命令 ——
而"翻文档"正是分窗重做要消掉的东西。

三条边界
--------
1. **不复制算法**：一条流程 = 一个 `_actions` 构造器 + 一个按钮。
   本文件里不许出现 `"xxx.py"` 字面量（`selftest_gui_budget` 用 AST 扫）。
2. **不替用户猜参数**：输入框留空 = 用脚本默认，构造器负责"空就不传"。
   界面只负责把"留空会怎样"写在说明里。
3. **点了要有回音**：起任务前后都写状态条，并说清产物去哪找 ——
   这条流程里有三条是"跑完只多几个文件"，不指路等于没跑。

为什么不做成模态
----------------
这些动作多数是子进程（开 cv2 窗口 / 训练十几分钟）。模态会把主窗口锁住，
用户既看不到任务日志，也不能在训练时去别的窗口看数据。
"""

from __future__ import annotations

from datetime import datetime
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

# 按**标记**上溯到含 `sphere_tracker/` 的祖先，不数层数（与 `fov_dialog` 同一条
# 纪律：`selftest_entrypoints` 对"数层数"是硬的，移动目录就会静默指错）。
ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "sphere_tracker").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps import _actions, i18n                              # noqa: E402
from apps import setup_env as _setup_env                     # noqa: E402
from apps._gui_widgets import BG, Surface, ScrollPage, StatusBar  # noqa: E402

#: 图片选择框的过滤器。`capture_board` 存的是 png，用户也可能拿手机拍 jpg。
_IMAGE_TYPES = (("图片", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("全部文件", "*.*"))


class ToolsDialog(tk.Toplevel):
    """一张页面上六张卡：每张卡 = 一条流程 + 一个按钮。

    ``app`` 是 `apps._gui_app.App`：要它的 ``tasks``（起子进程）与
    ``open_console``（状态条上的「任务日志」入口）。
    """

    def __init__(self, parent, app) -> None:
        super().__init__(parent)
        self.app = app
        # 这个对话框不经过 App.open_window（没有统一打标），自己登记归属键，
        # 让起任务时的 owner=self.task_owner 与模式窗口径一致。
        self.task_owner = "tools"
        # 不经 App.open_window，没有 close_window 代劳：自己接 WM_DELETE_WINDOW，
        # 按 X 时先停本窗起的任务，再销毁 —— 否则工具进程留在后台继续跑。
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.configure(bg=BG)
        self.title("标定与训练工具")
        self.geometry("880x760")
        self.minsize(760, 520)

        self.surface = Surface(self)
        # 先占底部：pack 从边缘往里分配，先 pack 的优先拿到空间。反过来状态条
        # 会盖住最后一张卡（门厅 2026-10-28 就是这么被盖掉一半的）。
        self.status = StatusBar(self, on_log=app.open_console, help_section="start")
        self.status.pack(fill="x", side="bottom")
        self._page = ScrollPage(self, self.surface)
        self._page.pack(fill="both", expand=True)
        body = self._page.content

        self._build_board(body)
        self._build_pick_color(body)
        self._build_world(body)
        self._build_export_calib(body)
        self._build_dataset(body)
        self._build_train(body)
        self._build_env(body)
        self._build_verify(body)

        self.status.set_status(
            "这些是「跑一次、隔很久再跑」的离线步骤；末尾可以切换 Python 运行环境。"
            "点按钮后台跑，进度在右下角「任务日志」；产物路径写在各卡片的说明里",
            "info")

    # ------------------------------------------------------------------ #
    # 通用小工具
    # ------------------------------------------------------------------ #
    def _param_row(self, parent, label: str,
                   fields: list[tuple[str, str]], *, width: int = 7) -> list[ttk.Entry]:
        """``标签 [子标签][框] [子标签][框] …`` —— 一行多个输入框。

        为什么不直接用 `Surface.param_entry` 一行一个：板参数（列/行/边长）
        拆成三行要 90 px，而这个对话框已经有六张卡，竖着堆会一屏装不下。
        """
        row = self.surface._next_row(parent)
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=(4, 0))
        ents: list[ttk.Entry] = []
        for i, (sub, val) in enumerate(fields):
            ttk.Label(parent, text=sub).grid(
                row=row, column=1 + i * 2, sticky="e", padx=(8, 2), pady=(4, 0))
            ent = ttk.Entry(parent, width=width)
            if val:
                ent.insert(0, val)
            ent.grid(row=row, column=2 + i * 2, sticky="w", pady=(4, 0))
            ents.append(ent)
        return ents

    def _path_row(self, parent, label: str, value: str = "", *,
                  kind: str = "file", title: str = "",
                  filetypes=None) -> ttk.Entry:
        """``标签 [路径框____________] [浏览…]``。

        ``kind="dir"`` 走选目录，其余走选文件 —— 少一次"我怎么填路径"的犹豫。
        ``filetypes`` 只影响文件选择框；解释器选择必须传 Python 过滤器。
        """
        row = self.surface._next_row(parent)
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=(4, 0))
        ent = ttk.Entry(parent, width=46)
        if value:
            ent.insert(0, value)
        ent.grid(row=row, column=1, columnspan=2, sticky="we", padx=(8, 4), pady=(4, 0))

        def pick() -> None:
            # 不传 initialdir 时 Windows 会沿用“上次文件夹”（可能是文档/桌面）。
            # 空值回项目根；已有值则回到它父目录，避免用户每次重新找。
            raw = ent.get().strip()
            cur = Path(raw).expanduser() if raw else None
            start = (str(cur.parent) if cur is not None and cur.is_file()
                     else str(cur) if cur is not None and cur.is_dir()
                     else str(ROOT))
            if kind == "dir":
                got = filedialog.askdirectory(
                    parent=self, title=title or label, initialdir=start)
            else:
                got = filedialog.askopenfilename(
                    parent=self, title=title or label, initialdir=start,
                    filetypes=filetypes or _IMAGE_TYPES)
            if got:
                ent.delete(0, "end")
                ent.insert(0, got)

        ttk.Button(parent, text="浏览…", command=pick).grid(
            row=row, column=3, sticky="w", pady=(4, 0))
        return ent

    def _start(self, argv: list[str], title: str, note: str) -> None:
        """起一个后台任务，并把"去哪看结果"说清楚。

        同名任务在跑就不再起第二个：训练一条要十几分钟，点两下起两个进程会
        互相抢 CPU/显存，而且第一次看到的现象是"越点越慢"。
        """
        for task in self.app.tasks.running():
            if (getattr(task, "title", "") or "") == title:
                self.status.set_status(
                    f"「{title}」已经在跑了 —— 这次没有重复启动。"
                    f"进度在「任务日志」里", "warn")
                return
        self.app.tasks.start(argv, title, owner=self.task_owner)
        self.status.set_status(note, "info")

    @staticmethod
    def _text(ent: ttk.Entry) -> str:
        return ent.get().strip()

    # ------------------------------------------------------------------ #
    # ① 生成标定板
    # ------------------------------------------------------------------ #
    def _build_board(self, body) -> None:
        box = self.surface.card(
            body, "① 生成可打印的标定板",
            "第一次用、或者手里那块板是别人给的，先在这里生成一份。"
            "打印时必须选「实际大小 / 100%」，不要选「适应页面」——"
            "打完拿尺子量方格边长，量出来是多少，标定时「方格边长」就填多少。")
        cols, rows, square = self._param_row(
            box, "棋盘（内角点）",
            [("列数", "8"), ("行数", "5"), ("边长 mm", "25")])
        self._board_cols = cols
        self._board_rows = rows
        self._board_square = square
        page = self.surface.param_combo(
            box, "纸张", ["a4-landscape", "a4-portrait", "a3-landscape", "letter-landscape"],
            "a4-landscape", hint="留空/选默认都是 A4 横向；A3 打出来格子大，远距离更稳。")
        self._board_page = page
        out = self.surface.param_entry(
            box, "输出目录", "", hint="留空 = assets/boards（仓库内，随代码走）。")
        self._board_out = out
        row = self.surface.action_row(box)
        self.surface.primary(row, "生成标定板", self._do_make_board)
        self.surface.inline_note(row, "产物：<输出目录>/*.pdf 与 *.png")

    def _do_make_board(self) -> None:
        argv = _actions.make_board(
            cols=self._text(self._board_cols),
            rows=self._text(self._board_rows),
            square=self._text(self._board_square),
            page=self._board_page.get(),
            out=self._text(self._board_out))
        out = self._text(self._board_out) or "assets/boards"
        self._start(argv, "生成标定板",
                    f"正在生成标定板 → {out}。打印后量一下方格边长，"
                    f"标定时填的数以量出来的为准")

    # ------------------------------------------------------------------ #
    # ② 点选取色
    # ------------------------------------------------------------------ #
    def _build_pick_color(self, body) -> None:
        box = self.surface.card(
            body, "② 点选取色（HSV 滑块）",
            "纯色球在某个场景下检不到时，先用它看一眼颜色到底落在哪段区间："
            "会开一个带滑块的小窗口，拖到球是白的、背景是黑的为止，按 s 保存。"
            "保存的阈值进 config/sphere_config.json，之后界面上选「自动」就沿用它。")
        img = self._path_row(box, "从图片取色", "", title="选一张有球的图")
        self._color_image = img
        cam = self.surface.param_entry(
            box, "或从相机取", "0",
            hint="上面填了图片就用图片；没填图才用这里的相机编号。相机编号见「相机设备」。")
        self._color_camera = cam
        save = self.surface.param_entry(
            box, "保存到", "", hint="留空 = config/sphere_config.json。")
        self._color_save = save
        row = self.surface.action_row(box)
        self.surface.primary(row, "打开取色窗口", self._do_pick_color)
        self.surface.inline_note(row, "这是个独立窗口：调完按 s 保存，按 q 退出")

    def _do_pick_color(self) -> None:
        argv = _actions.calibrate_color(
            image=self._text(self._color_image),
            camera=self._text(self._color_camera),
            save=self._text(self._color_save))
        self._start(argv, "点选取色",
                    "取色窗口已打开（它是独立进程，任务日志里能看到它的输出）。"
                    "拖滑块到「球白背景黑」，按 s 保存、q 退出")

    # ------------------------------------------------------------------ #
    # ③ 世界坐标系
    # ------------------------------------------------------------------ #
    def _build_world(self, body) -> None:
        box = self.surface.card(
            body, "③ 设世界坐标系（棋盘平放在地面）",
            "三维轨迹默认落在相机系，只能看形状；要让 Z 轴指向地面法向"
            "（朝上还是朝下由下面的下拉框决定），就得拍一张棋盘平放在地面的"
            "左相机图，在这里算一次 T_world_cam。这是一次性离线标定：拍一张、"
            "算一次、落盘，运行期只读 config/world_frame.json。"
            "注意：Z 朝上（-z）时 Y 会同时反向，这是保持右手系的代价。")
        img = self._path_row(box, "地面棋盘图", "", title="选一张「棋盘平放地面」的左相机图")
        self._world_image = img
        cols, rows, square = self._param_row(
            box, "棋盘（内角点）",
            [("列数", "8"), ("行数", "5"), ("边长 mm", "25")])
        self._world_cols = cols
        self._world_rows = rows
        self._world_square = square
        axis = self.surface.param_combo(
            box, "世界 Z 轴", ["-z", "z"], "z",
            hint="z = 与棋盘法向同向（默认；板平放、图案朝上拍时 Z 朝地面以下）；"
                 "-z = 反向（Z 朝上，代价是 Y 同时反向）。")
        self._world_axis = axis
        rig = self.surface.param_entry(
            box, "标定文件", "",
            hint="留空 = 当前装置档案的标定。填基名（不含 .npz），"
                 "例如 config\\calib\\<档案名>；拍图分辨率必须与它一致。")
        self._world_rig = rig
        note = self.surface.param_entry(
            box, "备注", "", hint="写进文件的人读说明，例如「球桌左侧地面」。")
        self._world_note = note
        row = self.surface.action_row(box)
        self.surface.primary(row, "计算并保存", self._do_world)
        self.surface.secondary(row, "只算不写", self._do_world_dry)
        self.surface.secondary(row, "用左相机拍一张…", self._do_capture_world_image)
        self.surface.inline_note(row, "产物：config/world_frame.json")

    def _do_world(self) -> None:
        self._run_world(dry_run=False)

    def _do_world_dry(self) -> None:
        self._run_world(dry_run=True)

    def _do_capture_world_image(self) -> None:
        """用左相机现场拍一张地面棋盘图，自动匹配标定分辨率并填入路径。"""
        for task in self.app.tasks.running():
            if (getattr(task, "title", "") or "") == "拍地面棋盘图":
                self.status.set_status(
                    "「拍地面棋盘图」已经在跑了 —— 请先在已开的取景窗内完成拍摄",
                    "warn")
                return
        w, h = 0, 0
        try:
            from apps._common import capture_size_from_profile, current_calibration
            from calib.params import StereoRig
            from stereo.profiles import get_active

            rig_arg = self._text(self._world_rig)
            calib_p = (Path(rig_arg) if rig_arg
                       else current_calibration(need_existing=False)[0])
            try:
                rig = StereoRig.load(calib_p)
                if rig.image_size and len(rig.image_size) == 2:
                    w, h = int(rig.image_size[0]), int(rig.image_size[1])
            except Exception:                                  # noqa: BLE001
                pass
            if w <= 0 or h <= 0:
                w, h, _note = capture_size_from_profile(get_active())
        except Exception:                                      # noqa: BLE001
            w, h = 0, 0

        out_dir = (ROOT / "outputs" / "world_frame"
                   / datetime.now().strftime("%Y%m%d-%H%M%S"))
        out_dir.mkdir(parents=True, exist_ok=True)
        argv = _actions.capture_world_floor_image(
            out_dir=out_dir,
            cols=self._text(self._world_cols),
            rows=self._text(self._world_rows),
            square=self._text(self._world_square),
            width=w, height=h,
        )

        def _on_captured(task) -> None:
            if not self.winfo_exists():
                return
            code = getattr(task, "exit_code", None)
            pics = sorted(
                p for p in out_dir.iterdir()
                if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")
            )
            if code == 0 and pics:
                self._world_image.delete(0, "end")
                self._world_image.insert(0, str(pics[-1]))
                self.status.set_status(
                    f"已拍好地面棋盘图（{pics[-1].name}）并填入路径，点「计算并保存」即可",
                    "ok")
            elif pics:
                self._world_image.delete(0, "end")
                self._world_image.insert(0, str(pics[-1]))
                self.status.set_status(
                    f"拍摄窗口已关闭，但已写入图 {pics[-1].name}；可直接点「计算并保存」",
                    "warn")
            else:
                self.status.set_status(
                    "没有拍下地面棋盘图（请看底部日志确认相机或棋盘是否完整入画）",
                    "warn")

        self.app.tasks.start(argv, "拍地面棋盘图", on_done=_on_captured,
                             owner=self.task_owner)
        res_txt = f"{w}x{h}" if w > 0 and h > 0 else "默认分辨率"
        self.status.set_status(
            f"正在用左相机拍地面棋盘图（目标 {res_txt}）：把棋盘平放地面，停稳自动存 1 张即收工",
            "info")

    def _run_world(self, *, dry_run: bool) -> None:
        image = self._text(self._world_image)
        if not image:
            # 脚本的 `--image` 是 required，空串传下去只会得到一句 argparse 报错，
            # 用户看不出"是因为我没选图"。所以在这里就拦住并说清。
            self.status.set_status(
                "还没选地面棋盘图：先点「浏览…」挑一张（棋盘平放在地面、"
                "左相机拍的那张）", "warn")
            return
        axis = str(self._world_axis.get()).strip() or "z"
        argv = _actions.set_world_frame(
            image=image,
            cols=self._text(self._world_cols),
            rows=self._text(self._world_rows),
            square=self._text(self._world_square),
            note=self._text(self._world_note),
            ground_axis=axis,
            rig=self._text(self._world_rig),
            dry_run=dry_run)
        updown = "Z 朝上（Y 同时反向）" if axis == "-z" else "Z 朝下（与棋盘法向同向）"
        if dry_run:
            self._start(argv, "世界坐标系（试算）",
                        f"正在试算 T_world_cam（不写文件），按 {updown} 算。"
                        "日志里会给出 ground_axis 与重投影误差：误差大说明板子放歪了"
                        "或角点没检全，分辨率不一致会直接报错")
        else:
            self._start(argv, "世界坐标系",
                        f"正在计算 T_world_cam → config/world_frame.json（{updown}）。"
                        "之后跑三维，轨迹就会用这个地面系")

    # ------------------------------------------------------------------ #
    # ④ 导出标定参数
    # ------------------------------------------------------------------ #
    def _build_export_calib(self, body) -> None:
        box = self.surface.card(
            body, "④ 导出标定参数（JSON / YAML / TOML）",
            "把这次双目标定的内参、畸变、R/T、基线、质量导成通用格式，"
            "换台机器复现实验、或者交给别人用别的软件时靠它。只读文件、不碰相机，"
            "什么时候导都安全。")
        calib = self.surface.param_entry(
            box, "标定目录", "", hint="留空 = config/stereo_calibration（当前装置）。")
        self._ec_calib = calib
        out = self.surface.param_entry(
            box, "输出目录", "", hint="留空 = outputs/calib_export。")
        self._ec_out = out
        fmt = self.surface.param_entry(
            box, "格式", "", hint="逗号分隔，留空 = 三种全导（json,yaml,toml）。")
        self._ec_fmt = fmt
        row = self.surface.action_row(box)
        self.surface.primary(row, "导出", self._do_export_calib)
        self.surface.inline_note(row, "每种格式各三份：left_camera / right_camera / stereo_rig")

    def _do_export_calib(self) -> None:
        argv = _actions.export_calibration(
            calib=self._text(self._ec_calib),
            out=self._text(self._ec_out),
            formats=self._text(self._ec_fmt))
        out = self._text(self._ec_out) or "outputs/calib_export"
        self._start(argv, "导出标定参数",
                    f"正在导出 → {out}（每台相机 + 双目整体，各格式一份）")

    # ------------------------------------------------------------------ #
    # ⑤ 导出训练数据集
    # ------------------------------------------------------------------ #
    def _build_dataset(self, body) -> None:
        box = self.surface.card(
            body, "⑤ 导出训练数据集（YOLO 单类）",
            "「以后不用手动点球心」那条路的第一步。你已经点过的每一次球心，"
            "跟踪出来的轨迹里就有逐帧圆心和半径 —— 那就是标注，只是格式不对。"
            "这里只换格式 + 切训练/验证，不产生任何新的几何推断。"
            "建议先勾「只统计不写文件」看一遍能导出多少帧。")
        out = self.surface.param_entry(
            box, "输出目录", "", hint="留空 = outputs/ball_dataset。")
        self._ds_out = out
        session = self.surface.param_entry(
            box, "只导这一段", "",
            hint="填会话目录就只导它（留空 = 全部可用会话）。会话路径可以在「数据管理」里复制。")
        self._ds_session = session
        val = self.surface.param_entry(
            box, "验证集会话数", "", hint="留空 = 2。按会话切分，不按帧切 —— 相邻帧几乎一样。")
        self._ds_val = val
        self._ds_dry = tk.BooleanVar(value=False)
        self.surface.param_check(box, "只统计不写文件（先看有多少可用帧）", self._ds_dry)
        row = self.surface.action_row(box)
        self.surface.primary(row, "导出数据集", self._do_dataset)
        self.surface.inline_note(row, "产物：images/ labels/ dataset.yaml")

    def _do_dataset(self) -> None:
        argv = _actions.export_ball_dataset(
            out=self._text(self._ds_out),
            session=self._text(self._ds_session),
            all_sessions=True,
            val_sessions=self._text(self._ds_val),
            dry_run=bool(self._ds_dry.get()))
        out = self._text(self._ds_out) or "outputs/ball_dataset"
        if self._ds_dry.get():
            self._start(argv, "导出数据集（只统计）",
                        "正在统计可用帧（不写文件）。日志里会按会话列出可用帧数与剔除原因")
        else:
            self._start(argv, "导出数据集",
                        f"正在导出 YOLO 数据集 → {out}。导出完可以在下面⑥直接训练")

    # ------------------------------------------------------------------ #
    # ⑥ 训练球检测器
    # ------------------------------------------------------------------ #
    def _build_train(self, body) -> None:
        box = self.surface.card(
            body, "⑥ 训练球检测器（YOLO11 单类）",
            "镜面金属球、花色水果这类「经典方法全灭」的目标，最后一步是让它学球长什么样。"
            "数据来自上面⑤导出的数据集。验收不看 mAP，看圆心误差 —— 检测结果只是种子，"
            "后面跟踪、测深、预测全从它出发，圆心错 10 px 半径再准也没用。"
            "CPU 上很慢（几十轮可能几小时），有独显就把设备填 cuda:0。")
        data = self.surface.param_entry(
            box, "数据集目录", "", hint="留空 = outputs/ball_dataset（上面⑤的默认产物）。")
        self._tr_data = data
        epochs, imgsz, batch = self._param_row(
            box, "训练参数", [("轮数", ""), ("尺寸", ""), ("批大小", "")], width=6)
        self._tr_epochs, self._tr_imgsz, self._tr_batch = epochs, imgsz, batch
        device = self.surface.param_entry(
            box, "设备", "", hint="留空 = cpu。有独显填 cuda:0（要装 CUDA 版 torch）。")
        self._tr_device = device
        row = self.surface.action_row(box)
        self.surface.primary(row, "开始训练", self._do_train)
        self.surface.inline_note(row, "产物：权重 + 逐图圆心误差报告")

    def _do_train(self) -> None:
        argv = _actions.train_ball_detector(
            data=self._text(self._tr_data),
            epochs=self._text(self._tr_epochs),
            imgsz=self._text(self._tr_imgsz),
            batch=self._text(self._tr_batch),
            device=self._text(self._tr_device))
        self._start(argv, "训练球检测器",
                    "训练已开始（长任务）。进度在「任务日志」；"
                    "中途要停就点状态条上的「中断」。跑完看日志末尾的圆心误差")

    # ------------------------------------------------------------------ #
    # ⑦ Python 运行环境（安装包/启动器同读的指针）
    # ------------------------------------------------------------------ #
    def _build_env(self, body) -> None:
        box = self.surface.card(
            body, "⑦ Python 运行环境",
            "安装包、launcher.exe 和 run.bat 都只读 config/python_env.txt 这一份指针。"
            "这里可以换成另一个人 python.exe；会真的 import 一遍核心依赖，"
            "不合格不会写指针。切换后关掉本程序、重新打开 launcher/GUI 才生效。")
        current = _setup_env.read_pointer()
        self._env_status = self.surface.hint(
            box, i18n.tf("当前：{path}",
                         path=str(current) if current is not None
                         else i18n.t("未登记（由启动器/安装器决定）")))
        self._env_python = self._path_row(
            box, "python.exe", "", title="选择 Python 解释器",
            filetypes=(("Python", "python.exe"), ("可执行文件", "*.exe"),
                       ("全部文件", "*.*")))
        row = self.surface.action_row(box)
        self.surface.primary(row, "验证并切换", self._do_env_switch)
        self.surface.secondary(row, "重新读取", self._refresh_env_status)
        self.surface.inline_note(row, "指针：config/python_env.txt；切换需重启 GUI/launcher")

    def _refresh_env_status(self) -> None:
        current = _setup_env.read_pointer()
        self._env_status.configure(
            text="当前：" + (str(current) if current is not None else "未登记（由启动器/安装器决定）"))

    def _do_env_switch(self) -> None:
        python = self._text(self._env_python)
        if not python:
            self.status.set_status(
                "先选一个 python.exe（点「浏览…」）；空路径不会写指针", "warn")
            return
        self._start(
            _actions.setup_env_adopt(python), "切换 Python 解释器",
            "正在验证所选解释器的 cv2/numpy/matplotlib/PIL/pyyaml/tkinter；"
            "通过后写 config/python_env.txt。切换后请关闭并重开 GUI/launcher")

    # ------------------------------------------------------------------ #
    # ⑧ 复检双目标定
    # ------------------------------------------------------------------ #
    def _build_verify(self, body) -> None:
        box = self.surface.card(
            body, "⑧ 复检双目标定（检查装置是否被碰动）",
            "双目使用一段时间后，用同一块棋盘拍几帧检验极线误差是否漂移；"
            "零共视 ~90° 装置若两路无法同时见板，会给出重拍参考对重标的说明。")
        calib = self._path_row(
            box, "标定文件", "", title="选择标定文件（留空 = 当前装置档案）",
            filetypes=(("标定文件", "*.npz *.json"), ("全部文件", "*.*")))
        self._vf_calib = calib
        cols, rows, square = self._param_row(
            box, "棋盘（内角点）",
            [("列数", "8"), ("行数", "5"), ("边长 mm", "25")])
        self._vf_cols, self._vf_rows, self._vf_square = cols, rows, square
        samples, tol = self._param_row(
            box, "复检参数", [("采样帧数", "8"), ("容差 px", "0.15")], width=8)
        self._vf_samples, self._vf_tol = samples, tol
        row = self.surface.action_row(box)
        self.surface.primary(row, "开始复检", self._do_verify)
        self.surface.inline_note(row, "结果写入底部任务日志")

    def _do_verify(self) -> None:
        calib = self._text(self._vf_calib)
        if calib:
            p = Path(calib)
            if p.suffix.lower() in (".npz", ".json"):
                calib = str(p.with_suffix(""))
        argv = _actions.verify_stereo(
            calib=calib,
            cols=self._text(self._vf_cols),
            rows=self._text(self._vf_rows),
            square=self._text(self._vf_square),
            samples=self._text(self._vf_samples),
            tolerance=self._text(self._vf_tol),
        )
        self._start(argv, "复检双目标定",
                    "正在复检双目标定（会打开取景窗，举棋盘采集样本，按 q 可提前结束）")

    def _on_close(self) -> None:
        """按 X：先停本窗起的任务，再销毁（不经 App.close_window）。"""
        self.app.tasks.stop_by_owner(self.task_owner)
        self.destroy()
