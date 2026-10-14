"""可视化：拖尾、圆、预测虚线、HUD、三维视图小面板。

注意：cv2.putText 只能画 ASCII，中文会变成问号。
所以 HUD 全部用英文/数字；要中文界面请改用 PIL 绘制（见 README 说明）。
"""

from __future__ import annotations

import cv2
import numpy as np

from .types import Detection, TrailPoint, TrackState


def _color_from_progress(p: float, base: tuple[int, int, int] = (0, 255, 255)) -> tuple[int, int, int]:
    """按进度在 深蓝 -> 基色 之间插值，做出"尾巴渐隐"的效果。"""
    p = float(max(0.0, min(1.0, p)))
    start = np.array([120, 40, 0], dtype=np.float64)   # BGR 暗蓝
    end = np.array(base, dtype=np.float64)
    c = start * (1.0 - p) + end * p
    return int(c[0]), int(c[1]), int(c[2])


def draw_trail(
    frame: np.ndarray,
    trail: list[TrailPoint],
    fade: bool = True,
    thickness: int = 2,
    base_color: tuple[int, int, int] = (0, 255, 255),
    scale: float = 1.0,
) -> None:
    """把历史轨迹画成折线。轨迹上每个点画一个小圆点，便于看出采样密度。

    `scale`：**画布相对原始帧的缩放倍率**。轨迹点存的是**原始帧坐标**，
    画布被放大过就必须一起放大，否则会画到画面外（见 `viz.scale` 的语义说明）。
    """
    n = len(trail)
    if n < 2:
        return
    for i in range(1, n):
        p0 = trail[i - 1]
        p1 = trail[i]
        color = _color_from_progress(i / (n - 1), base_color) if fade else base_color
        # 丢失补偿段（score==0）用更细更暗的线，一眼能看出是外推的
        if p1.score == 0.0:
            color = (90, 90, 90)
            th = max(1, thickness - 1)
        else:
            th = thickness
        cv2.line(frame, (int(round(p0.x * scale)), int(round(p0.y * scale))),
                 (int(round(p1.x * scale)), int(round(p1.y * scale))),
                 color, th, cv2.LINE_AA)
    for p in trail[:: max(1, n // 60)]:
        cv2.circle(frame, (int(round(p.x * scale)), int(round(p.y * scale))),
                   max(1, int(round(1 * scale))), (255, 255, 255), -1, cv2.LINE_AA)


def draw_detection(
    frame: np.ndarray,
    det: Detection,
    color: tuple[int, int, int] = (0, 255, 0),
    label: str | None = None,
    contour: bool = True,
    scale: float = 1.0,
    text_scale: float | None = None,
) -> None:
    """画检测到的圆 + 圆心 + 可选的真实轮廓。

    `scale` 的语义同 `draw_trail` —— 检测坐标与轮廓都是**原始帧坐标**。
    `text_scale` 只管**标签文字**（语义与 `draw_hud` 相同：显示缩放的倍率
    不该把文字一起缩小）。默认跟随 `scale`。
    """
    ts = scale if text_scale is None else max(0.7, float(text_scale))
    cx = int(round(det.x * scale))
    cy = int(round(det.y * scale))
    r = max(1, int(round(det.radius * scale)))
    cv2.circle(frame, (cx, cy), r, color, max(1, int(round(2 * scale))), cv2.LINE_AA)
    cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS,
                   max(3, int(round(12 * scale))), max(1, int(round(1 * scale))), cv2.LINE_AA)
    if contour and det.contour is not None and len(det.contour) > 2:
        c = (det.contour.reshape(-1, 2).astype(np.float32) * scale).astype(np.int32)
        cv2.drawContours(frame, [c.reshape(-1, 1, 2)], -1, (255, 0, 255),
                         max(1, int(round(1 * scale))), cv2.LINE_AA)
    if label:
        cv2.putText(frame, label, (cx + r + int(round(6 * scale)), cy - int(round(6 * scale))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * ts, color,
                    max(1, int(round(ts))), cv2.LINE_AA)


def draw_prediction(
    frame: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int] = (0, 140, 255),
    dash: int = 6,
    thickness: int = 2,
    label: str | None = None,
    dots: int = 10,
    scale: float = 1.0,
    text_scale: float | None = None,
) -> None:
    """把预测轨迹画成虚线（和实线的历史轨迹区分开）。

    `scale` 的语义同 `draw_trail`。**注意**：这里的 `points` 由调用方决定是否
    已经乘过 scale —— 为了统一，`render()` 现在**不预乘**，把 scale 交给本函数，
    这样"画布缩放"这件事只有一个地方负责。

    `text_scale` 只管标签文字（语义同 `draw_hud`）。
    """
    ts = scale if text_scale is None else max(0.7, float(text_scale))
    if points is None or len(points) < 2:
        return
    n = len(points)
    for i in range(1, n):
        if (i // max(1, dash // 2)) % 2 == 1:
            continue
        p0 = points[i - 1]
        p1 = points[i]
        cv2.line(
            frame,
            (int(round(p0[0] * scale)), int(round(p0[1] * scale))),
            (int(round(p1[0] * scale)), int(round(p1[1] * scale))),
            color, thickness, cv2.LINE_AA,
        )
    step = max(1, n // max(1, dots))
    for p in points[::step]:
        cv2.circle(frame, (int(round(p[0] * scale)), int(round(p[1] * scale))),
                   max(1, int(round(3 * scale))), color,
                   max(1, int(round(1 * scale))), cv2.LINE_AA)
    if label:
        last = points[-1]
        cv2.putText(frame, label,
                    (int(round(last[0] * scale)) + int(round(8 * scale)),
                     int(round(last[1] * scale)) + int(round(4 * scale))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45 * ts, color,
                    max(1, int(round(ts))), cv2.LINE_AA)


def draw_hud(frame: np.ndarray, lines: list[str], origin: tuple[int, int] = (10, 10),
             scale: float = 1.0, text_scale: float | None = None) -> None:
    """左上角半透明信息面板。

    ## 两个倍率不是一回事（2026-10-14 修）

    | 参数 | 管什么 |
    | ---- | ------ |
    | `scale` | **几何**：面板位置、内边距、行高。必须跟画布走，否则面板相对画面乱跑 |
    | `text_scale` | **字号与笔画**。默认等于 `scale`（保持"整张图 k 倍"的语义）；实时显示时**故意传小于 scale 的值** |

    ## 为什么要把字号单独拆出来

    原来只有 `scale` 一个参数，字号是 `0.52 * scale`。于是实时显示
    （`display_width=960`，1920 的 0.5 倍）把字缩成 `0.52*0.5 = 0.26`、
    笔画 `max(1, ...)` 退到 **1 px**、行高 `20*0.5 = 10 px` ——
    **字不是"被重采样糊掉"，是被缩到 10 像素高，根本认不出来**。
    用户的原话是"字还是如此模糊，这种文字应该设置为不受清晰度设置影响"，
    这个判断是对的：**`display_width` 是"为了省 imshow 缩放开销"而降的显示宽度，
    它没有任何理由把信息面板一起缩小。**

    ⚠️ 注意别把它误读成"先缩图再画字"那个回归（那个已经修好了，
    `pipeline.render()` 的注释里有记录）。这是**第二个、独立的**字号问题：
    即使顺序对了，字号仍被 `display_width` 折进去的 `scale` 乘小。

    `text_scale` 有下限 `0.7`：再小就不可读了，而"看不清的信息面板"
    等于没有 —— 宁可让它在小窗口里略占地方。
    """
    if not lines:
        return
    if text_scale is None:
        text_scale = scale
    ts = max(0.7, float(text_scale))
    font = cv2.FONT_HERSHEY_SIMPLEX
    fscale = 0.52 * ts
    thick = max(1, int(round(ts)))
    pad = int(round(8 * scale))
    lh = max(int(round(20 * ts)), int(round(16 * scale)))
    widths = [cv2.getTextSize(t, font, fscale, thick)[0][0] for t in lines]
    w = max(widths) + 2 * pad
    h = lh * len(lines) + pad
    x, y = int(round(origin[0] * scale)), int(round(origin[1] * scale))
    x2, y2 = min(x + w, frame.shape[1]), min(y + h, frame.shape[0])
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (x + pad, y + pad + lh * (i + 1) - int(round(5 * fscale * 2))),
                    font, fscale, (255, 255, 255), thick, cv2.LINE_AA)


def paste_mask_preview(
    frame: np.ndarray,
    mask: np.ndarray | None,
    width: int = 200,
) -> None:
    """右下角显示分割掩膜的小窗（调试阈值时很有用）。"""
    if mask is None:
        return
    h, w = mask.shape[:2]
    scale = width / float(w)
    small = cv2.resize(mask, (width, max(1, int(h * scale))), interpolation=cv2.INTER_NEAREST)
    small_bgr = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    fh, fw = frame.shape[:2]
    sh, sw = small_bgr.shape[:2]
    y0, x0 = fh - sh - 10, fw - sw - 10
    if y0 < 0 or x0 < 0:
        return
    frame[y0 : y0 + sh, x0 : x0 + sw] = small_bgr
    cv2.rectangle(frame, (x0 - 1, y0 - 1), (x0 + sw, y0 + sh), (200, 200, 200), 1)


def _project_2d(
    points: np.ndarray,
    xs: int,
    ys: int,
    view: str,
    bounds: tuple[float, float, float, float],
) -> np.ndarray:
    """把三维点投影到小面板的像素坐标。view: 'top'(X-Z) 或 'side'(Y-Z)。"""
    x_min, x_max, v_min, v_max = bounds
    if x_max - x_min < 1e-9:
        x_max = x_min + 1e-3
    if v_max - v_min < 1e-9:
        v_max = v_min + 1e-3
    out = np.empty((points.shape[0], 2), dtype=np.float64)
    for i, p in enumerate(points):
        if view == "top":
            a, b = p[0], p[2]
        else:
            a, b = p[1], p[2]
        # 深度 Z 越大越远 -> 画在面板上方
        px = (a - x_min) / (x_max - x_min) * (xs - 1)
        py = (ys - 1) - (b - v_min) / (v_max - v_min) * (ys - 1)
        out[i] = (px, py)
    return out


def draw_3d_panel(
    size: tuple[int, int] = (300, 220),
    history: np.ndarray | None = None,
    prediction: np.ndarray | None = None,
    title: str = "3D (Z = depth from ball size)",
    scale: float = 1.0,
) -> np.ndarray:
    """画一个二维的小面板：俯视图 (X-Z) + 侧视图 (Y-Z)。

    单目相机拿不到真正的三维，这里显示的是"像素位置 + 由球半径反推的深度"。

    `scale` 的语义同 `draw_trail`：**文字与线宽也要跟着面板尺寸走**。
    `size` 只是画布，字号/线宽/内边距不会因为画布变大而自动变大 ——
    不乘 scale 的话，`viz.scale=2` 时面板里的一切实测会显得比外面小一半。
    """
    w, h = size
    panel = np.zeros((h, w, 3), dtype=np.uint8)
    panel[:] = (28, 28, 28)
    # 字号与线宽统一由 scale 决定（画布尺寸由调用方按 scale 给）
    fs_title, fs_label, fs_hint = 0.42 * scale, 0.38 * scale, 0.45 * scale
    th = max(1, int(round(1 * scale)))
    margin = int(round(8 * scale))
    cv2.putText(panel, title, (margin, int(round(16 * scale))), cv2.FONT_HERSHEY_SIMPLEX,
                fs_title, (200, 200, 200), th, cv2.LINE_AA)
    split = h // 2
    cv2.line(panel, (0, split), (w, split), (70, 70, 70), max(1, int(round(1 * scale))), cv2.LINE_AA)

    allpts = []
    for arr in (history, prediction):
        if arr is not None and len(arr):
            allpts.append(np.asarray(arr, dtype=np.float64))
    if not allpts:
        cv2.putText(panel, "no 3D data", (int(round(10 * scale)), split + int(round(24 * scale))),
                    cv2.FONT_HERSHEY_SIMPLEX, fs_hint, (140, 140, 140), th, cv2.LINE_AA)
        return panel
    merged = np.vstack(allpts)
    x_min, x_max = float(merged[:, 0].min()), float(merged[:, 0].max())
    y_min, y_max = float(merged[:, 1].min()), float(merged[:, 1].max())
    z_min, z_max = float(merged[:, 2].min()), float(merged[:, 2].max())
    pad_x = max(0.02, 0.15 * (x_max - x_min))
    pad_y = max(0.02, 0.15 * (y_max - y_min))
    pad_z = max(0.02, 0.15 * (z_max - z_min))

    for idx, (view, bounds, label) in enumerate(
        [
            ("top", (x_min - pad_x, x_max + pad_x, z_min - pad_z, z_max + pad_z), "top view  X-Z"),
            ("side", (y_min - pad_y, y_max + pad_y, z_min - pad_z, z_max + pad_z), "side view Y-Z"),
        ]
    ):
        y_off = 0 if idx == 0 else split
        sub_h = split if idx == 0 else h - split
        cv2.putText(panel, label, (margin, y_off + int(round(30 * scale))),
                    cv2.FONT_HERSHEY_SIMPLEX, fs_label, (160, 220, 160), th, cv2.LINE_AA)
        if history is not None and len(history) > 1:
            pts = _project_2d(np.asarray(history, dtype=np.float64), w, sub_h, view, bounds)
            for i in range(1, len(pts)):
                cv2.line(
                    panel,
                    (int(pts[i - 1][0]), int(pts[i - 1][1]) + y_off),
                    (int(pts[i][0]), int(pts[i][1]) + y_off),
                    (0, 255, 255), max(1, int(round(2 * scale))), cv2.LINE_AA,
                )
        if prediction is not None and len(prediction) > 1:
            pts = _project_2d(np.asarray(prediction, dtype=np.float64), w, sub_h, view, bounds)
            for i in range(1, len(pts)):
                if i % 3 == 0:
                    continue
                cv2.line(
                    panel,
                    (int(pts[i - 1][0]), int(pts[i - 1][1]) + y_off),
                    (int(pts[i][0]), int(pts[i][1]) + y_off),
                    (0, 140, 255), max(1, int(round(2 * scale))), cv2.LINE_AA,
                )
    return panel


def hud_lines(
    result,
    cfg,
    health=None,
) -> list[str]:
    """生成 HUD 文本（纯 ASCII）。

    ## 帧率为什么要分三个写出来（这是本函数改动的全部理由）

    改之前第一行是 **`FPS {result.fps}`** —— 而 `result.fps` 量的是**主循环**
    每秒处理多少帧（`pipeline._fps_ema`），**不是摄像头采集帧率**。
    两件事在画面上长得一样，但含义差很远：

    - 主循环 15 fps 可能是"相机 30 fps 但检测太慢"，也可能是"相机只给了 15 fps"；
    - 光看一个数**分不出是哪一种**，于是"往哪个方向优化"纯靠猜。

    这正是 §10.10 与坑清单反复引用的"走错方向的根因"。
    现在三个口径分开写、各有唯一的名字（与 `stats.Health.summary_line` 同一套叫法）：

    | 名字 | 含义 | 谁算的 |
    | ---- | ---- | ------ |
    | `cap` | 采集帧率（相机侧给了多少） | `Health.capture_fps` |
    | `loop` | 主循环帧率（我们处理得过来多少） | `Health.loop_fps` |
    | `detect` | 单帧检测耗时 | `Health.stages.detect_ms` |

    拿不到 `health`（例如离线视频、或还没接线）时退化成一行明确的
    `loop ...`，**不再出现裸的 `FPS`** —— 裸 `FPS` 正是歧义的来源。

    `STALL` 是采集停住的提示：**没有它的话，"球没动"与"相机没了"在画面上完全一样**
    （见 `sources.LatestFrameCamera.stalled`）。
    """
    lines: list[str] = []
    if health is not None:
        head = (f"cap {health.capture_fps:5.1f}  loop {health.loop_fps:5.1f}  "
                f"detect {health.stages.detect_ms:6.1f} ms")
        if health.dropped_frames:
            head += f"  drop {health.dropped_frames}"
        if health.frame_age_ms > 0:
            head += f"  age {health.frame_age_ms:5.1f} ms"
        if health.skew_ms_valid:
            head += f"  skew {health.skew_ms:+6.1f} ms"
        lines.append(head)
    else:
        # 没有 Health 时也要说清这是哪个"帧率"
        lines.append(f"loop {result.fps:5.1f}  detect {result.detect_ms:6.1f} ms")
    if getattr(result, "stalled", False):
        lines.append("** STALL: 采集停住（相机掉线/被占用？）**")
    det = result.detection
    trk: TrackState | None = result.track
    lines.append(f"method {cfg.detector.method}" + (f"  |  {det.method}" if det else "  |  no target"))
    if det is not None:
        lines.append(f"score {det.score:0.2f}   r = {det.radius:6.1f} px   d = {2 * det.radius:6.1f} px")
    if trk is not None:
        coast = "  COASTING" if trk.coasting else ""
        lines.append(f"track #{trk.track_id}  hits {trk.hits}  age {trk.age}{coast}")
        lines.append(f"px  ({trk.x:7.1f}, {trk.y:7.1f})")
        lines.append(f"v   ({trk.vx:8.1f}, {trk.vy:8.1f}) px/s   |v| {trk.speed_px:7.1f}")
        if trk.ax or trk.ay:
            lines.append(f"a   ({trk.ax:8.1f}, {trk.ay:8.1f}) px/s^2")
    if result.position_3d is not None:
        p = result.position_3d
        lines.append(f"3D  X {p[0]:+0.3f}  Y {p[1]:+0.3f}  Z {p[2]:0.3f} m")
    # 相机自运动：手持/云台拍摄时必须让用户看见，否则会把"画面在动"
    # 误当成"球在动"，预测也会跟着错。
    mot = getattr(result, "camera_motion", None)
    if mot is not None and getattr(mot, "method", "none") != "none":
        tag = "MOVING" if getattr(result, "camera_moving", False) else "static"
        lines.append(
            f"cam {tag}  shift ({mot.dx:+6.2f},{mot.dy:+6.2f}) px  |d| {mot.magnitude:5.2f}"
        )
    if result.prediction is not None:
        method = cfg.predict.method
        if (method == "quadratic" and cfg.predict.use_pixel_gravity
                and getattr(result, "camera_moving", False)
                and not getattr(result, "camera_steady", False)):
            method += " (no g_px)"
        lines.append(f"predict {method} +{cfg.predict.horizon_s:0.2f}s")
    return lines
