"""收标定板照片 —— 给"单板链式标定"的 4 段引导流程用。

为什么单独做一个工具
--------------------
链式标定要按固定顺序拍 4 组图（左内参 / 右内参 / 左参考 / 右参考），
每组都得**当场确认**"整块板进画面了、手停了、没糊"再按空格。GUI 需要一个
可被 subprocess 调用、能用退出码表达结果的收图器，而不是复制一份标定窗口。

判据沿用 ``calib.board`` 与 ``calib.capture`` 的口径，不另起一套：
  * 整块棋盘检出 —— ``calib.board.detect_board``（SB，绝不混用经典检测器）；
  * 静止 —— 相对上一帧角点的**最大**位移 ≤ ``--stationary-px``；
  * 清晰度 —— **尺度无关的 focus**，判据用 ``calib.capture.focus_gate``
    （与 calibrate_stereo 同一条规则，不是照抄一份）：
    ``focus = 棋盘 ROI 的 Laplacian 方差 × 格子像素数``；只拒**明显比本组中位数糊**
    的帧（默认低于中位数 40%）与低于绝对地板的帧。旧口径（裸 Laplacian 方差）
    会随棋盘占的像素数漂移，板一大就被误判成"糊"，已弃用。

逐帧做的是**粗检**（降分辨率、不开 EXHAUSTIVE，约几十毫秒）；
真正要落盘的那一帧再全分辨率精确复核 —— 和标定窗口的做法一致，
否则 2560x1440 上每帧 300+ ms，窗口会卡到没法对准。

三种用法
--------
1. 单路按键（原来那条路，GUI 的 4 段引导用它）::

    python apps/tools/capture_board.py --out data/calib/L/intr --side left

2. 单路自动（板举着不用按键；判据连稳 N 帧才存，存完冷却一段时间）::

    python apps/tools/capture_board.py --out data/calib/L/intr --side left --auto

3. 双路成对（左右各存一张，同一时间戳；两路都检出整块板才存）::

    python apps/tools/capture_board.py --both --out-left data/L --out-right data/R

自测（视频代替相机，不建窗口）::

    python apps/tools/capture_board.py --out /tmp/x --side left \\
        --source video:tmp/board.mp4 --no-show --target 3
    python apps/tools/capture_board.py --both --out-left /tmp/l --out-right /tmp/r \\
        --source-left video:tmp/board.mp4 --source-right video:tmp/board.mp4 --no-show

退出码：0=收够 target；3=用户提前退出 / 视频放完没到 target；1=打不开设备或目录；2=参数错。
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
while not (ROOT / "sphere_tracker").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "apps"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from _common import log_environment, setup_logging  # noqa: E402
from apps._window import DEFAULT_WINDOW_SIZE, WindowManager  # noqa: E402
from calib.board import BoardObservation, BoardSpec, detect_board  # noqa: E402
from calib.board_coverage import (  # noqa: E402
    REF_STILL_PX, REGIONS_TOTAL, coverage_ok, reference_shift, region_of)
from calib.capture import CaptureConfig, focus_gate  # noqa: E402
from sphere_tracker import camera_registry as cr  # noqa: E402
from sphere_tracker.sources import open_source  # noqa: E402

log = logging.getLogger("capture_board")

# 清晰度判据的默认参数**取自 CaptureConfig**，不在这里另写一份数字 ——
# 否则标定窗口与这个收图器会慢慢漂成两套阈值。
_FOCUS_DEFAULTS = CaptureConfig()

# 粗检用的宽度：960 宽在 1440p 上约 30~50 ms，够实时对准；精确复核仍用全分辨率。
GATE_WORK_WIDTH = 960

# 采集边车文件名。**每个保存目录各写一份自包含的**（双路模式 out_left /
# out_right 两个目录各一份，不写父目录单文件）—— 目录被单独复制/移动后，
# 元数据必须仍然跟着自己的那批图。
SIDECAR_NAME = "_capture.json"
SIDECAR_SCHEMA = "capture-board/1"


def _tool_version() -> str:
    """工具版本：git 短哈希；取不到写 "unknown"（不为它新增依赖）。"""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT),
            capture_output=True, text=True, timeout=5, check=True)
        v = proc.stdout.strip()
        if v:
            return v
    except Exception as exc:                                 # noqa: BLE001
        log.debug("取 git 版本失败（边车写 unknown）：%s", exc)
    return "unknown"


def write_capture_sidecar(out_dir: Path, spec: BoardSpec, *, camera: str,
                          width: int, height: int, fps: float,
                          backend: str, count: int) -> Path:
    """把采集元数据写成 ``<out_dir>/_capture.json``，返回边车路径。

    为什么必须有（2026-10-27 计划书 §3.1 ②）：此前采集只写日志不落盘，
    收完图再改面板的 cols/rows 拿旧图去标定，检测静默报"0 张检出"且不解释。
    有了边车，这批图能**自证**"用什么板、哪路相机、什么分辨率拍的"。

    纯函数：不开相机、不建窗口 —— 自测与 GUI 都能直接调。
    失败抛 OSError，由调用方转成错误输出（写不进元数据与写不进图同一待遇，
    不许静默吞掉）。
    """
    payload = {
        "schema": SIDECAR_SCHEMA,
        "board": {"cols": spec.cols, "rows": spec.rows,
                  "square_mm": spec.square_mm},
        "camera": camera,
        "width": int(width),
        "height": int(height),
        "fps": float(fps),
        "backend": backend,
        "captured_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "count": int(count),
        "tool_version": _tool_version(),
    }
    path = out_dir / SIDECAR_NAME
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return path

# 版式常量见下方 `GUIDE_MARGIN` / `GUIDE_AREA_RANGE` / `CANVAS_MIN`。
# 原来这里有个 `BOTH_VIEW_HEIGHT = 720`：它把两路各缩到 720 高再 hstack →
# 一张固定的 2560x720 拼图，配 AUTOSIZE 窗口 —— 拖不动、全屏无效、字缩没了。
# 2026-10-27 起改成"画布跟着窗口尺寸走"，这个常量随之删掉。


def judge_capture(found: bool, still_px: float, focus: float,
                  stationary_px: float, min_sharpness: float | None,
                  median_focus: float = float("nan"), n_samples: int = 0,
                  min_ratio: float | None = None, floor: float | None = None,
                  min_samples: int | None = None) -> tuple[bool, str]:
    """这一帧能不能存？返回 (可存, 不可存的原因)。

    纯函数：不碰相机、不建窗口、不读文件 —— GUI 与自测都直接调它。
    阈值必须由入参给（``--stationary-px`` / ``--min-sharpness``），这里不写死，
    否则命令行改了参数而判据没跟着变，是最难查的一类不一致。

    清晰度**不是**这里自己判的：直接调 ``calib.capture.focus_gate``，
    与标定窗口**同一条规则**（绝对地板 + 相对中位数）。两处各写一套，
    早晚会分叉成"采集放行、标定拒收"。

    ``focus`` 是**尺度无关**的清晰度（``BoardObservation.focus`` =
    Laplacian 方差 × 格子像素数）。别塞 ``sharpness`` 进来 ——
    那个量随棋盘在画面里的大小变化，板一大就会被误判成"糊"。

    顺序有意：**先报"没检出"再报"在动"再报"糊"**。
    板都没看见时说"太糊"会把人引去擦镜头，实际只是板不在画面里。
    """
    if not found:
        return False, "没检出整块棋盘"
    if still_px > stationary_px:
        return False, f"还在动 {still_px:.1f} px"
    return focus_gate(
        focus, median_focus=median_focus, n_samples=n_samples,
        min_ratio=_FOCUS_DEFAULTS.min_focus_ratio if min_ratio is None else min_ratio,
        floor=_FOCUS_DEFAULTS.focus_floor if floor is None else floor,
        min_samples=(_FOCUS_DEFAULTS.focus_min_samples if min_samples is None
                     else min_samples),
        abs_min=min_sharpness,
    )


def should_autosave(stable_frames: int, auto_hold: int, since_last_s: float,
                    auto_gap: float) -> tuple[bool, str]:
    """自动模式：这一帧该自动存吗？返回 (该存, 不该存的原因)。

    纯函数，阈值全来自入参 —— ``--auto-hold`` / ``--auto-gap`` 改了要能被测出来。

    两道闸门各有各的用处，缺一不可：
      * ``auto_hold``：判据"连续"满足多少帧。单帧判据会被**手在等的时候的
        微小抖动**骗过去（板端部抖 2 px 但瞬时停在门限内），连稳 N 帧才是真停稳；
      * ``auto_gap``：两张之间的最小间隔。没有它，同一个姿势连稳 100 帧就存 100 张 ——
        用户看着"存够了"，但全是同一视角，标定退化。

    等于阈值算通过（``>=``）：用户说 6 帧就是"第 6 帧起可以存"。
    """
    need = max(1, int(auto_hold))
    if stable_frames < need:
        return False, f"稳定 {stable_frames}/{need} 帧"
    gap = max(0.0, float(auto_gap))
    if since_last_s < gap:
        return False, f"冷却中 {gap - since_last_s:.1f} s"
    return True, ""


def corner_shift_px(prev: np.ndarray | None, cur: np.ndarray | None) -> float:
    """两张图的同名角点最大位移（像素）。任一缺失返回 ``inf``（视为"没法判静止"）。

    用**最大**而不是均值：均值会被 40 个几乎不动的角点稀释，
    板一端翘起来（手在转）也照样能过门限 —— 那正是最该拦下的帧。
    """
    if prev is None or cur is None:
        return float("inf")
    a = np.asarray(prev, np.float32).reshape(-1, 2)
    b = np.asarray(cur, np.float32).reshape(-1, 2)
    if a.shape != b.shape or len(a) == 0:
        return float("inf")
    return float(np.max(np.linalg.norm(a - b, axis=1)))


def gate_frame(gray: np.ndarray, spec: BoardSpec, prev_corners: np.ndarray | None,
               stationary_px: float, min_sharpness: float | None,
               focus_hist: list[float] | None = None
               ) -> tuple[BoardObservation, float, bool, str, np.ndarray | None]:
    """粗检一帧并给出判据：返回 (观测, 位移, 可存, 原因, 新基准帧)。

    只有检出时才更新基准帧：板短暂丢失（手挡了一下）不该把下一帧判成"还在动"，
    否则用户会莫名其妙地被卡住。

    ``focus_hist`` 是**已存下**各帧的 focus（相对判据的基准）。只看已存下的帧：
    被拒的帧不该把中位数拉低，否则"越来越糊"的一组会互相掩护着全进来。
    """
    hist = focus_hist if focus_hist is not None else []
    obs = detect_board(gray, spec, work_width=GATE_WORK_WIDTH, exhaustive=False)
    still = corner_shift_px(prev_corners, obs.corners if obs.found else None)
    can, reason = judge_capture(
        obs.found, still, obs.focus, stationary_px, min_sharpness,
        median_focus=float(np.median(hist)) if hist else float("nan"),
        n_samples=len(hist))
    if obs.found:
        prev_corners = obs.corners.copy()
    return obs, still, can, reason, prev_corners


def _backend_const(name: str):
    """CLI 的后端名 -> cv2 常量；``auto`` 交给 CameraSource 自己成对挑。"""
    if not name or name == "auto":
        return None
    return {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}.get(name.lower())


def _open_source(source: str | None, camera: str | None, args,
                 side: str | None = None):
    """按 ``video:``/``--source*`` / ``--camera*`` / 角色打开一路输入。

    类型判定与装配**全部委托** `sphere_tracker.sources.open_source`：
    收图器以前自己抄了一份"video: 前缀 = 视频、否则 = 相机"，于是它只认 2 种源，
    而 `run_video` 认 3 种（图片文件夹在这里会当成相机去开）。现在两边同一份实现。

    ``video:`` 前缀保留原意（**强制**按视频文件打开）；不带前缀时让 `open_source`
    按路径推断 —— 也就是说图片目录/单张图片现在也能直接喂进来。

    失败抛 RuntimeError/FileNotFoundError，由调用方统一转成退出码 1。
    """
    if source:
        raw = str(source)
        forced_video = raw.lower().startswith("video:")
        path = raw[len("video:"):] if forced_video else raw
        return open_source(path, kind="file" if forced_video else None)

    want = camera or side
    index = cr.resolve_camera_arg(want, interactive=False)
    if index is None:
        raise RuntimeError(f"没能把 {want!r} 解析成相机索引；"
                           f"先跑 python apps/tools/cameras.py list --probe")
    return open_source(camera=index, width=args.width, height=args.height,
                       fps=args.fps, backend=_backend_const(args.backend))


CANVAS_MIN = (640, 360)            # 兜底下限：窗口被拖到极小时也不至于除零
GUIDE_MARGIN = 0.06                # 引导框距边 6%：整块板要落在框内
GUIDE_AREA_RANGE = (8.0, 35.0)     # 板占画面面积的建议区间（%）
TARGET_BOX = 0.52                  # 内框占画面宽高的比例（面积≈27%，落在上面区间里）
VIEW_SINGLE = "capture_board"
VIEW_BOTH = "capture_board (both)"


def _canvas_size(name: str) -> tuple[int, int]:
    """按**当前窗口**尺寸给画布尺寸；窗口还没建好就用 1280x720 起步。

    这是"同时适配全屏和窗口模式"的关键：全屏时 `getWindowImageRect` 给的是
    屏幕图像区，于是画布 = 屏幕，imshow 是 1:1（字不糊）；窗口模式下给的是
    窗口客户区，画布 = 窗口，也是 1:1。两种模式共用同一条路径，不需要两套版式。

    首帧的回退值刻意取 `DEFAULT_WINDOW_SIZE`（1280x720）而不是"屏幕的 86%"：
    `WindowManager.ensure()` 会把初始窗口设成"图片的 1.5 倍、屏幕上限定 90%"，
    首帧给一张接近屏幕的画布 → 窗口直接开到 90% 屏，很吓人。给 720p 就刚好。
    """
    try:
        rect = cv2.getWindowImageRect(name)
        if rect and rect[2] > 0 and rect[3] > 0:
            return int(rect[2]), int(rect[3])
    except Exception:                                    # noqa: BLE001
        pass
    return DEFAULT_WINDOW_SIZE


def _fit_into(img: np.ndarray, w: int, h: int):
    """等比缩放到 (w, h) 以内，返回 (图, 左上角偏移 x, y, 缩放比)。"""
    ih, iw = img.shape[:2]
    scale = min(w / max(iw, 1), h / max(ih, 1))
    nw, nh = max(1, int(round(iw * scale))), max(1, int(round(ih * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return (cv2.resize(img, (nw, nh), interpolation=interp),
            (w - nw) // 2, (h - nh) // 2, scale)


def _board_metrics(obs: BoardObservation, shape) -> tuple[float, float]:
    """(板占画面面积 %, 最靠近边缘的角点距边 px)；没检出就 (nan, nan)。"""
    if not obs.found or obs.corners is None or not len(obs.corners):
        return float("nan"), float("nan")
    h, w = shape[:2]
    pts = obs.corners.reshape(-1, 2).astype(np.float32)
    area = float(cv2.contourArea(cv2.convexHull(pts))) / max(1.0, float(w * h)) * 100.0
    edge = float(min(pts[:, 0].min(), pts[:, 1].min(),
                     w - pts[:, 0].max(), h - pts[:, 1].max()))
    return area, edge


def capture_stage(objective: str | None, out_dir: Path | None) -> str:
    """这一段是内参段（``intr``）还是参考段（``ref``）？

    为什么需要分：两段的"进度"是**两个不同的问题** —— 内参段要问"板摆到过
    画面的哪些角"，参考段要问"板是不是还一动没动"。混成一个数字会互相误导
    （参考段板不动，覆盖永远是 1/5，看起来像没拍）。

    默认 ``auto`` 从保存目录名推断（链式标定的目录就是 ``ref_left`` /
    ``ref_right`` 与 ``left`` / ``right``），这样 GUI 那条调用链**不用改**就能
    得到正确提示；``--stage`` 可显式覆盖，避免有人把参考图存进别的名字。
    """
    if objective in ("intr", "ref"):
        return str(objective)
    name = (out_dir.name if out_dir is not None else "").lower()
    return "ref" if "ref" in name else "intr"


def lane_progress(stage: str, covered: set[int],
                  obs: BoardObservation,
                  ref_base: np.ndarray | None) -> dict:
    """一路的 HUD 进度：内参段给覆盖集合，参考段给与第一张的位移。纯函数。

    参考段的位移要**用当前帧**算（不是用已存下的上一张）：提示的意义是
    "你现在手别动"，等保存完再说就已经晚了。基准固定为**第一张已保存的**图 ——
    相邻两张比会漏掉缓慢单向漂移（每张挪 4 px，四张就漂了 12 px）。

    没检出 / 还没有基准张时返回 ``None`` 值：调用方据此显示占位，
    而不是把"没法判"画成"0 px，很稳"。
    """
    if stage == "ref":
        if ref_base is None or not obs.found or obs.corners is None:
            return {"ref_shift": None}
        shift, _pair = reference_shift([ref_base, obs.corners.reshape(-1, 2)])
        return {"ref_shift": shift}
    return {"coverage": (set(covered), REGIONS_TOTAL)}


def _paint_progress(canvas: np.ndarray, x0: int, y0: int, w: int, h: int,
                    fs: float, coverage, ref_shift) -> None:
    """在指标行**上方**再画一行进度：``覆盖 2/5 ▮▮▯▯▯`` 或 ``参考段位移 148 px``。

    为什么单独一行、单独配色：用户拍的时候眼睛只扫两三处。把"这一张合不合格"
    （绿/橙的原因行）和"整段还差什么"（覆盖/位移）混在一句里，前者会淹没后者 ——
    今天那次事故就是 15 张全部"合格"，整段却废了。

    颜色语义有意区分：覆盖不够是**橙色**（继续拍还能补，别慌），
    参考段板在动是**红色 + 明确指令**（这一张已经废了，手必须停下）。
    """
    if coverage is None and ref_shift is None:
        return
    pad = int(10 * fs)
    lh = int(24 * fs)
    top = y0 + h - int(42 * fs) - lh
    if top < y0:
        return                                   # 格子太矮（缩到极小的窗口），宁可不画
    text = ""
    color = (232, 232, 232)
    if coverage is not None:
        got, total = coverage
        # "够不够"由库里那条判据回答（QUALITY.min_regions），这里不写 4
        color = (80, 220, 120) if coverage_ok(got) else (60, 180, 255)
        text = f"覆盖 {len(got)}/{total}"
    else:
        if ref_shift is None:
            text = "参考段：等第一张存下来"
            color = (200, 200, 200)
        elif ref_shift > REF_STILL_PX:
            text = f"参考段位移 {ref_shift:.0f} px  板在动，别挪"
            color = (70, 70, 245)                # BGR 红：与"合格绿"拉开最大距离
        else:
            text = f"参考段位移 {ref_shift:.0f} px  稳"
            color = (80, 220, 120)

    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs * 0.78, 1)
    gap = int(6 * fs)
    side = int(13 * fs)
    # 方块宽度先算进来再画底：否则方块落在深色底之外，压在画面亮部就看不见
    n_box = (coverage[1] if coverage is not None else 0)
    box_w = n_box * (side + gap) + int(12 * fs) if n_box else 0
    cv2.rectangle(canvas, (x0, top),
                  (x0 + tw + int(22 * fs) + box_w, top + lh + pad // 2),
                  (26, 26, 26), -1)
    cv2.putText(canvas, text, (x0 + int(10 * fs), top + th + int(4 * fs)),
                cv2.FONT_HERSHEY_SIMPLEX, fs * 0.78, color, 1, cv2.LINE_AA)

    # 内参段再补 5 个小方块：只给 "2/5" 用户不知道缺哪个角，画出格子才指得动。
    if coverage is None:
        return
    bx = x0 + tw + int(22 * fs)
    for idx in range(n_box):
        cx = bx + idx * (side + gap)
        if cx + side > x0 + w - int(8 * fs):
            break
        fill = (80, 220, 120) if idx in coverage[0] else (70, 70, 245)
        cv2.rectangle(canvas, (cx, top + lh // 2 - side // 2),
                      (cx + side, top + lh // 2 + side // 2), fill, -1)


def _paint_lane(canvas: np.ndarray, x0: int, y0: int, w: int, h: int,
                frame: np.ndarray, spec: BoardSpec, obs: BoardObservation,
                *, ok: bool, reason: str, title: str,
                coverage=None, ref_shift=None) -> None:
    """一路画面 + 引导框 + 角点 + 指标，画进画布的 (x0, y0, w, h) 格子里。

    **角点与文字都画在画布坐标系里**（先算缩放比再映射）。反过来做（在原图上画、
    再整体缩放）会让提示文字跟着分辨率一起缩 —— 2560 宽的拼图缩到窗口里，
    0.55 号字只剩几个像素，那正是"看起来没有提示"的原因。
    """
    img, ox, oy, scale = _fit_into(frame, w, h)
    canvas[y0 + oy:y0 + oy + img.shape[0], x0 + ox:x0 + ox + img.shape[1]] = img

    # 1) 双层引导框：
    #    外框（距边 6%）= **不许出界**（贴边会被"图棋盘贴边"门限拒）；
    #    内框（居中 52%×52%，面积≈27%）= **建议至少填满**，落在 GUIDE_AREA_RANGE 里。
    #    只画外框等于"框住整个画面"，用户看不出该把板放多大 —— 这是第一版的实际问题。
    gx, gy = int(w * GUIDE_MARGIN), int(h * GUIDE_MARGIN)
    cv2.rectangle(canvas, (x0 + gx, y0 + gy), (x0 + w - gx, y0 + h - gy),
                  (96, 96, 96), 1, cv2.LINE_AA)
    tw, th = int(w * TARGET_BOX), int(h * TARGET_BOX)
    tx0, ty0 = x0 + (w - tw) // 2, y0 + (h - th) // 2
    guide = (70, 210, 90) if ok else (120, 170, 210)
    cv2.rectangle(canvas, (tx0, ty0), (tx0 + tw, ty0 + th), guide, 1, cv2.LINE_AA)
    fs = max(0.5, h / 780.0)
    cv2.putText(canvas, "目标区", (tx0 + int(8 * fs), ty0 + int(20 * fs)),
                cv2.FONT_HERSHEY_SIMPLEX, fs * 0.6, guide, 1, cv2.LINE_AA)

    # 2) 角点：映射到画布坐标再画
    if obs.found and obs.corners is not None and len(obs.corners):
        pts = obs.corners.reshape(-1, 2).astype(np.float32) * scale
        pts[:, 0] += x0 + ox
        pts[:, 1] += y0 + oy
        cv2.drawChessboardCorners(canvas, spec.pattern_size,
                                  pts.reshape(-1, 1, 2), True)

    # 3) 文字：字号随格子高度走，全屏下也不会小；**垫一条深色底**，
    #    否则压在画面亮部就看不见（第一版直接写在图上，遇到白背景等于消失）。
    fs = max(0.5, h / 780.0)
    thick = max(1, int(round(fs * 2)))
    color = (80, 220, 120) if ok else (60, 180, 255)
    head = f"{title}  " + ("OK" if ok else (reason or "—"))
    # 原因可能很长（"棋盘还在移动（12px）"之类），超出格宽就截断 —— 宁可少几个字，
    # 也不要让文字横穿到隔壁那一路去（看起来像标错了路）。
    while len(head) > 4 and cv2.getTextSize(
            head, cv2.FONT_HERSHEY_SIMPLEX, fs, thick)[0][0] > w - int(24 * fs):
        head = head[:-2] + "…"
    (hw, hh), _ = cv2.getTextSize(head, cv2.FONT_HERSHEY_SIMPLEX, fs, thick)
    cv2.rectangle(canvas, (x0, y0), (x0 + hw + int(22 * fs), y0 + hh + int(18 * fs)),
                  (26, 26, 26), -1)
    cv2.putText(canvas, head, (x0 + int(10 * fs), y0 + int(30 * fs)),
                cv2.FONT_HERSHEY_SIMPLEX, fs, color, thick, cv2.LINE_AA)

    # 3.5) 整段进度（覆盖度 / 参考段位移）：画在指标行上方，压在深色底上，
    #      与"这一帧合不合格"分开 —— 两件事混一句，后者会被前者淹没。
    _paint_progress(canvas, x0, y0, w, h, fs, coverage, ref_shift)

    area, edge = _board_metrics(obs, frame.shape)
    if np.isfinite(area):
        lo, hi = GUIDE_AREA_RANGE
        warn = "  ← 偏小" if area < lo else ("  ← 偏大" if area > hi else "")
        detail = (f"板 {area:.1f}%{warn}  清晰度 {obs.focus:.0f}  "
                  f"距边 {edge:.0f}px")
    else:
        detail = "未检出：整块入框、别贴边、别斜到看不见格"
    dy = y0 + h - int(12 * fs)
    (dw, dh), _ = cv2.getTextSize(detail, cv2.FONT_HERSHEY_SIMPLEX, fs * 0.78, 1)
    cv2.rectangle(canvas, (x0, dy - dh - int(8 * fs)), (x0 + dw + int(22 * fs), y0 + h),
                  (26, 26, 26), -1)
    cv2.putText(canvas, detail, (x0 + int(10 * fs), dy),
                cv2.FONT_HERSHEY_SIMPLEX, fs * 0.78, (232, 232, 232), 1, cv2.LINE_AA)


def _compose(lanes: list[dict], size: tuple[int, int], footer: str,
             ok_all: bool) -> np.ndarray:
    """把 N 路 lane 拼成一张画布。`lanes` 里每项：
    `{title, frame, obs, ok, reason, spec}`。
    """
    W, H = int(size[0]), int(size[1])
    n = max(1, len(lanes))
    fh = int(np.clip(H // 10, 56, 104))              # 底部状态栏高度
    body = max(1, H - fh)
    cw = max(1, W // n)
    # 底色用深灰而不是纯黑：letterbox 的黑边看起来像"画面坏了"，深灰是有意的底色
    canvas = np.full((H, W, 3), 45, np.uint8)
    for i, lane in enumerate(lanes):
        x0 = i * cw
        w = cw if i < n - 1 else (W - x0)            # 最后一格吃掉余数，右边缘不留黑缝
        _paint_lane(canvas, x0, 0, w, body, lane["frame"], lane["spec"],
                    lane["obs"], ok=lane["ok"], reason=lane["reason"],
                    title=lane["title"], coverage=lane.get("coverage"),
                    ref_shift=lane.get("ref_shift"))
        if i:                                        # 格间分隔线：高对比画面下否则会糊成一片
            cv2.line(canvas, (x0, 0), (x0, body), (90, 90, 90), 1, cv2.LINE_AA)
    # 底部状态栏：进度 + 两路是否"同时"满足 + 按键
    cv2.rectangle(canvas, (0, body), (W, H), (26, 26, 26), -1)
    fs = max(0.6, H / 900.0)
    cv2.putText(canvas, footer, (int(14 * fs), body + int(fh * 0.66)),
                cv2.FONT_HERSHEY_SIMPLEX, fs,
                (80, 220, 120) if ok_all else (60, 180, 255), 2, cv2.LINE_AA)
    return canvas



def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="收标定板照片（单板链式标定的 4 段引导流程用）")
    ap.add_argument("--out", default=None, help="保存目录（不存在则创建）；--both 时改用 --out-left/--out-right")
    ap.add_argument("--side", default=None, choices=["left", "right"],
                    help="用哪一路相机（left/right 查 config/cameras.json 的角色）；--both 时忽略")
    ap.add_argument("--camera", default=None,
                    help="显式相机：索引 / 角色名 / name:...；默认取 --side 的角色")
    ap.add_argument("--backend", choices=["auto", "dshow", "msmf"], default="auto",
                    help="相机后端（默认 auto，由 CameraSource 成对挑）")
    ap.add_argument("--width", type=int, default=2560, help="采集宽度（默认 2560）")
    ap.add_argument("--height", type=int, default=1440, help="采集高度（默认 1440）")
    ap.add_argument("--fps", type=int, default=30, help="采集帧率（默认 30）")
    ap.add_argument("--cols", type=int, default=8, help="标定板内角点列数（默认 8）")
    ap.add_argument("--rows", type=int, default=5, help="标定板内角点行数（默认 5）")
    ap.add_argument("--square", type=float, default=25.0, help="方格边长 mm（默认 25）")
    ap.add_argument("--target", type=int, default=15,
                    help="存够这么多张就退出（--both 下按对数算，默认 15）")
    ap.add_argument("--stationary-px", type=float, default=3.0,
                    help="连续两帧角点最大位移上限（px，默认 3.0）")
    ap.add_argument("--min-sharpness", type=float, default=800.0,
                    help="额外的**绝对**清晰度下限，比的是尺度无关的 focus"
                         "（=Laplacian 方差 × 格子像素数，默认 800.0）。"
                         "绝对阈值没有刻度，真正起作用的是相对判据：focus 低于"
                         "已存帧中位数 40%% 才判糊（阈值取自 CaptureConfig）；越小越宽松")
    ap.add_argument("--auto", action="store_true",
                    help="停稳自动存（不用按空格）")
    ap.add_argument("--auto-hold", type=int, default=6,
                    help="判据连续满足这么多帧才自动存（默认 6）")
    ap.add_argument("--auto-gap", type=float, default=1.2,
                    help="两次自动存的最小间隔秒数（默认 1.2）")
    ap.add_argument("--both", action="store_true",
                    help="双路同时采集：两路都检出整块板才成对保存（需 --out-left/--out-right）")
    ap.add_argument("--out-left", default=None, help="--both：左路保存目录")
    ap.add_argument("--out-right", default=None, help="--both：右路保存目录")
    ap.add_argument("--camera-left", default=None, help="--both：左路相机（默认取 left 角色）")
    ap.add_argument("--camera-right", default=None, help="--both：右路相机（默认取 right 角色）")
    ap.add_argument("--source", default=None,
                    help="输入源：路径（视频文件/图片文件夹/单张图片，按后缀与目录自动判）；"
                         "video:<路径> 强制当视频；只读文件（--both 时用 --source-left/--source-right）")
    ap.add_argument("--source-left", default=None, help="--both 自测：左路视频源")
    ap.add_argument("--source-right", default=None, help="--both 自测：右路视频源")
    ap.add_argument("--timeout", type=float, default=0.0,
                    help="最多跑多少秒（0=不限；视频源下按秒计时）")
    ap.add_argument("--stage", choices=["auto", "intr", "ref"], default="auto",
                    help="这一段拍的是什么：intr=内参（HUD 显示覆盖几个区域）、"
                         "ref=参考（HUD 显示与第一张的位移）、auto=按保存目录名推断"
                         "（含 ref 的是参考段，默认）")
    ap.add_argument("--no-show", action="store_true", help="不建窗口（自测/后台）")
    return ap


def _check_args(ap: argparse.ArgumentParser, args) -> None:
    """参数组合的合法性 —— 全部走 ``ap.error``（退出码 2）。"""
    if args.both:
        if args.out:
            ap.error("--both 用 --out-left/--out-right 两条目录，不能再给 --out")
        if args.source:
            ap.error("--both 与 --source 互斥；双路自测请用 --source-left/--source-right")
        if not (args.out_left and args.out_right):
            ap.error("--both 需要 --out-left 和 --out-right")
        if (args.source_left is None) != (args.source_right is None):
            ap.error("--source-left 与 --source-right 要一起给（两路都读视频）")
        return
    if args.out_left or args.out_right:
        ap.error("--out-left/--out-right 只用于 --both")
    if args.source_left or args.source_right:
        ap.error("--source-left/--source-right 只用于 --both 自测")
    if not args.out:
        ap.error("缺 --out（单路）；双路请用 --both --out-left … --out-right …")
    # `--side` 在单路里**不是必需的**：它只用来给窗口标题和会话记一个"这是哪一路"，
    # 真正的相机由 `--camera`（角色名/索引）决定，都不给就自动挑。
    # 原来这里强制要求 side，导致单目标定（只有一路）根本调不起来
    # （2026-10-28 用户报 "capture_board.py: error: 缺 --side"）。
    # 双路的 `--both` 分支仍然要求 left/right，那是另一段校验。


def _auto_settings(args) -> tuple[int, float]:
    return max(1, int(args.auto_hold)), max(0.0, float(args.auto_gap))


def _make_dir(path: str | None) -> Path | None:
    if not path:
        return None
    p = Path(path).expanduser().resolve()
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"[错误] 保存目录不可用：{p}（{exc}）")
        return None
    return p


def _run_single(args, spec: BoardSpec, out_dir: Path) -> int:
    # 有 --source 一律不建窗口；无窗口时**只能自动存**，否则没有任何保存途径。
    show = not (args.no_show or args.source)
    auto = bool(args.auto) or not show
    hold, gap = _auto_settings(args)
    # 窗口助手必须在**第一次调用它之前**建好（静态看没毛病，跑起来才炸
    # UnboundLocalError —— `run_realtime` 踩过这个，见 `selftest_window_mode` §3）。
    wins = WindowManager()

    try:
        src = _open_source(args.source, args.camera, args, side=args.side)
    except Exception as exc:                                     # noqa: BLE001
        print(f"[错误] 打不开相机/视频：{(str(exc).splitlines() or [type(exc).__name__])[0]}")
        return 1
    log.info("输入 %s %dx%d fps=%.0f %s %s", src.info.kind, src.info.width,
             src.info.height, src.info.fps, src.info.backend, src.info.path)
    if auto:
        log.info("自动保存：连稳 %d 帧、间隔 %.1f s（空格仍可随时存）", hold, gap)

    saved: list[Path] = []
    prev_corners: np.ndarray | None = None
    focus_hist: list[float] = []      # 已存帧的 focus（相对判据的基准）
    # 整段进度（判据来自 calib.board_coverage，这里只攒状态）：
    #   intr —— 已保存的图覆盖了哪几个区域；
    #   ref  —— 第一张已保存的角点当基准，之后每帧与它比位移。
    stage = capture_stage(args.stage, out_dir)
    covered: set[int] = set()
    ref_base: np.ndarray | None = None
    stable = 0
    last_save_ts: float | None = None
    last_cat = ""
    last_print = 0.0
    t0 = time.perf_counter()

    try:
        while True:
            if args.timeout > 0 and time.perf_counter() - t0 >= args.timeout:
                log.info("到 --timeout %.1f s，停止", args.timeout)
                break
            ok, frame, ts = src.read()
            if not ok or frame is None:
                log.info("输入结束（视频放完或相机断流）")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            obs, still, can, reason, prev_corners = gate_frame(
                gray, spec, prev_corners, args.stationary_px, args.min_sharpness,
                focus_hist)
            stable = stable + 1 if can else 0

            since = float("inf") if last_save_ts is None else ts - last_save_ts
            go, auto_reason = (should_autosave(stable, hold, since, gap)
                               if (auto and can) else (False, reason))

            pressed = -1
            if show:
                hint = (f"自动：连稳 {hold} 帧 / 间隔 {gap:.1f} s   空格 立即存   "
                        f"q 退出   {wins.fullscreen_hint()}"
                        if auto else
                        f"空格 保存   q 退出   {wins.fullscreen_hint()}")
                footer = (f"{len(saved)}/{args.target} 张   "
                          f"{'可保存' if can else (reason or '—')}   "
                          f"{src.info.width}x{src.info.height}   {hint}")
                canvas = _compose(
                    [{"title": f"{args.side or '相机'}", "frame": frame,
                      "obs": obs, "ok": can, "reason": reason, "spec": spec,
                      **lane_progress(stage, covered, obs, ref_base)}],
                    _canvas_size(VIEW_SINGLE), footer, can)
                wins.show(VIEW_SINGLE, canvas)
                pressed = cv2.waitKey(1) & 0xFF
                # 全屏键先处理：Esc 在全屏时只退全屏，不该把收图退掉
                if wins.handle_key(VIEW_SINGLE, pressed, canvas):
                    continue
                if pressed == ord("q") or (pressed == 27):
                    log.info("用户退出（q/ESC）")
                    break

            if not can:
                # 不刷屏：原因类别变了、或距上次打印超过 0.5 s 才输出一行
                now = time.perf_counter()
                cat = reason.split()[0]
                if cat != last_cat or now - last_print > 0.5:
                    log.info("跳过：%s", reason)
                    last_cat = cat
                    last_print = now
                continue

            manual = show and pressed == 32
            if not (go or manual):
                now = time.perf_counter()
                # 「冷却中 x.x s」必须打出来：用户看到"板停住了却没存"，
                # 唯一的解释就在这一行里。
                cat = auto_reason.split()[0]
                if cat != last_cat or now - last_print > 0.5:
                    log.info("不存：%s", auto_reason)
                    last_cat = cat
                    last_print = now
                continue

            # 真要落盘：全分辨率精确复核（这一步才贵，所以只在候选帧上做）
            acc = detect_board(gray, spec, exhaustive=True)
            if not acc.found:
                log.info("跳过：精确复核没检出整块棋盘")
                continue
            can_acc, reason_acc = judge_capture(
                True, still, acc.focus, args.stationary_px, args.min_sharpness,
                median_focus=float(np.median(focus_hist)) if focus_hist else float("nan"),
                n_samples=len(focus_hist))
            if not can_acc:
                log.info("跳过：精确复核 %s", reason_acc)
                continue

            name = f"shot_{len(saved) + 1:02d}_{int(time.time() * 1000)}.png"
            path = out_dir / name
            if not cv2.imwrite(str(path), frame):
                print(f"[错误] 写不进文件：{path}")
                return 1
            saved.append(path)
            focus_hist.append(float(acc.focus))
            # 整段进度只在**真正落盘**时更新：被拒的帧没进这批图，不该算"拍到了"。
            if stage == "ref":
                if ref_base is None and acc.corners is not None:
                    ref_base = acc.corners.reshape(-1, 2).copy()   # 第一张当基准
            elif acc.corners is not None:
                idx = region_of(acc.corners, gray.shape[:2])
                if idx >= 0:
                    covered.add(idx)
            verb = "自动保存" if not manual else "已保存"
            print(f"{verb} {len(saved)}/{args.target}  {name}  "
                  f"（focus {acc.focus:.0f}，原始清晰度 {acc.sharpness:.0f}，位移 {still:.1f} px）")
            log.info("保存 %s（focus %.0f，原始清晰度 %.0f，格子 %.1fpx，位移 %.1f px）",
                     path, acc.focus, acc.sharpness, acc.square_px, still)
            stable = 0
            last_save_ts = ts
            if len(saved) >= args.target:
                break
    finally:
        src.release()
        if show:
            cv2.destroyAllWindows()

    print(f"共保存 {len(saved)} 张 → {out_dir}")
    if 0 < len(saved) < 3:
        print("至少需要 3 张才能标定，建议 15 张覆盖画面四角")
    # 边车最后写：count 必须是**实际落盘**的张数（含提前退出/没到 target 的情况）
    try:
        sidecar = write_capture_sidecar(
            out_dir, spec, camera=args.camera or args.side or "",
            width=src.info.width, height=src.info.height, fps=src.info.fps,
            backend=src.info.backend, count=len(saved))
    except OSError as exc:
        print(f"[错误] 元数据边车写不进：{out_dir / SIDECAR_NAME}（{exc}）")
        return 1
    log.info("元数据边车 %s（count=%d）", sidecar, len(saved))
    return 0 if len(saved) >= args.target else 3


def _run_both(args, spec: BoardSpec, out_left: Path, out_right: Path) -> int:
    """双路成对采集：两路都满足判据才存，文件名共用同一毫秒时间戳。

    为什么要"同时"：外参只在**同一时刻**的一对图之间成立。
    分别拍两段再对，只要中间把板/相机碰了一下，标定就会悄悄错 —— 而且看不出来。
    """
    show = not args.no_show
    hold, gap = _auto_settings(args)
    # 同上：先建后用一个字都不能省（`selftest_window_mode` §3 会扫顺序）。
    wins = WindowManager()
    src_l = src_r = None
    try:
        src_l = _open_source(args.source_left, args.camera_left, args, side="left")
        src_r = _open_source(args.source_right, args.camera_right, args, side="right")
    except Exception as exc:                                     # noqa: BLE001
        print(f"[错误] 打不开相机/视频：{(str(exc).splitlines() or [type(exc).__name__])[0]}")
        if src_l is not None:
            src_l.release()
        return 1
    log.info("左路 %s %dx%d；右路 %s %dx%d", src_l.info.kind, src_l.info.width,
             src_l.info.height, src_r.info.kind, src_r.info.width, src_r.info.height)
    if (src_l.info.width, src_l.info.height) != (src_r.info.width, src_r.info.height):
        log.warning("两路分辨率不一致（%dx%d vs %dx%d）—— 标定要求一致",
                    src_l.info.width, src_l.info.height,
                    src_r.info.width, src_r.info.height)
    log.info("双路自动保存：连稳 %d 帧、间隔 %.1f s，目标 %d 对", hold, gap, args.target)

    saved = 0
    prev_l: np.ndarray | None = None
    prev_r: np.ndarray | None = None
    focus_hist_l: list[float] = []      # 两路各记各的（两路画质可以差很多）
    focus_hist_r: list[float] = []
    # 整段进度**两路各算各的**：两路的视野与摆位本来就不同，
    # 把覆盖度合成一个数会掩盖"右路只有 3/5"这种单边短板（今天的实况）。
    stage = capture_stage(args.stage, out_left)
    covered_l: set[int] = set()
    covered_r: set[int] = set()
    ref_base_l: np.ndarray | None = None
    ref_base_r: np.ndarray | None = None
    stable = 0
    last_save_ts: float | None = None
    last_cat = ""
    last_print = 0.0
    t0 = time.perf_counter()

    try:
        while True:
            if args.timeout > 0 and time.perf_counter() - t0 >= args.timeout:
                log.info("到 --timeout %.1f s，停止", args.timeout)
                break
            ok_l, fl, ts_l = src_l.read()
            ok_r, fr, ts_r = src_r.read()
            if not ok_l or fl is None or not ok_r or fr is None:
                log.info("输入结束（有一路放完或断流）")
                break

            gl = cv2.cvtColor(fl, cv2.COLOR_BGR2GRAY)
            gr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            obs_l, still_l, can_l, why_l, prev_l = gate_frame(
                gl, spec, prev_l, args.stationary_px, args.min_sharpness, focus_hist_l)
            obs_r, still_r, can_r, why_r, prev_r = gate_frame(
                gr, spec, prev_r, args.stationary_px, args.min_sharpness, focus_hist_r)
            both_ok = can_l and can_r
            stable = stable + 1 if both_ok else 0
            ts = max(ts_l, ts_r)
            since = float("inf") if last_save_ts is None else ts - last_save_ts
            go, auto_reason = (should_autosave(stable, hold, since, gap)
                               if both_ok else (False, why_l or why_r))

            pressed = -1
            if show:
                # 两路各自的原因 + **共同可见**状态：单板参考图的核心就是
                # "同一块板同一位置，两路同时看到" —— 所以这一条必须一眼可见，
                # 不能只显示"左 OK / 右：还在动"就完事。
                if both_ok:
                    seen = "两路同时可见 ✓ 可以存"
                elif can_l:
                    seen = f"右路未满足：{why_r}"
                elif can_r:
                    seen = f"左路未满足：{why_l}"
                else:
                    seen = f"两路都未满足｜左 {why_l or 'OK'}｜右 {why_r or 'OK'}"
                footer = (f"{saved}/{args.target} 对   {seen}   "
                          f"空格 立即存   q 退出   {wins.fullscreen_hint()}")
                canvas = _compose(
                    [{"title": "左", "frame": fl, "obs": obs_l, "ok": can_l,
                      "reason": why_l, "spec": spec,
                      **lane_progress(stage, covered_l, obs_l, ref_base_l)},
                     {"title": "右", "frame": fr, "obs": obs_r, "ok": can_r,
                      "reason": why_r, "spec": spec,
                      **lane_progress(stage, covered_r, obs_r, ref_base_r)}],
                    _canvas_size(VIEW_BOTH), footer, both_ok)
                wins.show(VIEW_BOTH, canvas)
                pressed = cv2.waitKey(1) & 0xFF
                if wins.handle_key(VIEW_BOTH, pressed, canvas):
                    continue
                if pressed == ord("q") or pressed == 27:
                    log.info("用户退出（q/ESC）")
                    break

            if not both_ok:
                now = time.perf_counter()
                cat = (why_l or why_r).split()[0]
                if cat != last_cat or now - last_print > 0.5:
                    log.info("跳过：左 %s｜右 %s", why_l or "OK", why_r or "OK")
                    last_cat = cat
                    last_print = now
                continue

            manual = show and pressed == 32
            if not (go or manual):
                now = time.perf_counter()
                cat = auto_reason.split()[0]
                if cat != last_cat or now - last_print > 0.5:
                    log.info("不存：%s", auto_reason)
                    last_cat = cat
                    last_print = now
                continue

            # 两路都精确复核：任一路不过就不存这一对（宁可少一对，不要错一对）
            acc_l = detect_board(gl, spec, exhaustive=True)
            acc_r = detect_board(gr, spec, exhaustive=True)
            ok_l2, why_l2 = judge_capture(
                acc_l.found, still_l, acc_l.focus, args.stationary_px, args.min_sharpness,
                median_focus=float(np.median(focus_hist_l)) if focus_hist_l else float("nan"),
                n_samples=len(focus_hist_l))
            ok_r2, why_r2 = judge_capture(
                acc_r.found, still_r, acc_r.focus, args.stationary_px, args.min_sharpness,
                median_focus=float(np.median(focus_hist_r)) if focus_hist_r else float("nan"),
                n_samples=len(focus_hist_r))
            if not (ok_l2 and ok_r2):
                log.info("跳过：精确复核 左 %s｜右 %s", why_l2 or "OK", why_r2 or "OK")
                continue

            n = saved + 1
            ms = int(time.time() * 1000)
            name = f"pair_{n:02d}_{ms}.png"
            pl, pr = out_left / name, out_right / name
            if not (cv2.imwrite(str(pl), fl) and cv2.imwrite(str(pr), fr)):
                print(f"[错误] 写不进文件：{pl} 或 {pr}")
                return 1
            saved = n
            focus_hist_l.append(float(acc_l.focus))
            focus_hist_r.append(float(acc_r.focus))
            # 两路进度各自更新（与单路同一套口径，见 capture_stage / lane_progress）
            if stage == "ref":
                if ref_base_l is None and acc_l.corners is not None:
                    ref_base_l = acc_l.corners.reshape(-1, 2).copy()
                if ref_base_r is None and acc_r.corners is not None:
                    ref_base_r = acc_r.corners.reshape(-1, 2).copy()
            else:
                if acc_l.corners is not None:
                    idx_l = region_of(acc_l.corners, gl.shape[:2])
                    if idx_l >= 0:
                        covered_l.add(idx_l)
                if acc_r.corners is not None:
                    idx_r = region_of(acc_r.corners, gr.shape[:2])
                    if idx_r >= 0:
                        covered_r.add(idx_r)
            print(f"已存第 {n} 对（左 {n} / 右 {n}）  {name}  "
                  f"（focus 左 {acc_l.focus:.0f} / 右 {acc_r.focus:.0f}；"
                  f"原始清晰度 {acc_l.sharpness:.0f} / {acc_r.sharpness:.0f}）")
            log.info("保存一对 %s", name)
            stable = 0
            last_save_ts = ts
            if saved >= args.target:
                break
    finally:
        src_l.release()
        src_r.release()
        if show:
            cv2.destroyAllWindows()

    print(f"共保存 {saved} 对 → {out_left} ; {out_right}")
    # 每个目录各写一份**自包含**的边车：count 按"该目录里的图数"算（= 对数，
    # 每对在两个目录各落一张、文件名 pair_* 前后一致）。
    for out_dir, camera, info in (
            (out_left, args.camera_left or "left", src_l.info),
            (out_right, args.camera_right or "right", src_r.info)):
        try:
            sidecar = write_capture_sidecar(
                out_dir, spec, camera=camera, width=info.width,
                height=info.height, fps=info.fps, backend=info.backend,
                count=saved)
        except OSError as exc:
            print(f"[错误] 元数据边车写不进：{out_dir / SIDECAR_NAME}（{exc}）")
            return 1
        log.info("元数据边车 %s（count=%d）", sidecar, saved)
    return 0 if saved >= args.target else 3


def main() -> int:
    ap = _build_parser()
    args = ap.parse_args()
    _check_args(ap, args)

    try:
        spec = BoardSpec(cols=args.cols, rows=args.rows, square_mm=args.square)
    except ValueError as exc:
        print(f"[参数错误] {exc}")
        return 2

    if args.both:
        out_left = _make_dir(args.out_left)
        out_right = _make_dir(args.out_right)
        if out_left is None or out_right is None:
            return 1
        setup_logging("capture_board")
        log_environment(log)
        log.info("左目录 %s；右目录 %s；板 %s；target=%d 对",
                 out_left, out_right, spec.describe(), args.target)
        return _run_both(args, spec, out_left, out_right)

    out_dir = _make_dir(args.out)
    if out_dir is None:
        return 1
    setup_logging("capture_board")
    log_environment(log)
    log.info("输出目录 %s；板 %s；target=%d", out_dir, spec.describe(), args.target)
    return _run_single(args, spec, out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
