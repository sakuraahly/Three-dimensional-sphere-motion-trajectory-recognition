"""镜面球（金属摆锤）检测器：人工 seed + CSRT 跟踪 + 模板/运动门控。

为什么不是 Hough/颜色
--------------------
实测（1920×1080，`20261019-193809`）：

- 真球半径 **35~40 px**，不是 640 宽预览里的 8~12 px；
- 球速可达 **~80 px/帧**；
- 饱和度与背景白袋几乎一样（S≈30 vs 26），颜色阈值不可分；
- Hough 的窄半径 6~14 命中率 0%，宽半径 14~34 也只有 22.7%，
  且会锁到背景圆（右路 frame 37 起锁在 ~(645,605)）；
- **CSRT + 人工 seed** 实测 86.4%@15 px、100%@20 px、20.5 ms/帧。

所以本检测器**要求人工 seed**（`--seed x,y,r`，右路另给 `--seed-right`）。
自动初始化不可靠（Hough top-1 可偏 528 px），不假装全自动。
丢失时返回空列表，**绝不** snap 到远处的 Hough 圆。

圆心与半径**出处不同**（`B-130` / `B-132`）
--------------------------------------------
- **半径**只能来自人工 seed / CSRT 框：镜面球整圈只有 25%~35% 有梯度，且球内
  强反射的半径明显更小，边缘法**量不出**真半径；
- **圆心**不能来自高光斑：球面反射是视角相关的，高光斑在左右视图落在球面不同
  位置，于是"追高光的框中心"在左右图天然错位（实测约 6 px），**双目三角化要的是
  轮廓圆心**。

因此跟踪出框后，默认再用亚像素射线拟合把圆心钉到球轮廓上（`refine_center`），
半径仍取框；拟合不出足够梯度时原样保留框中心。产出的 `Detection.extra`
里记 `center_source`（`"contour"` / `"box"`）与 `center_shift_px`，让"这个圆心
是怎么来的"可自证，而不是靠下游猜。
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from ..config import DetectorConfig
from ..geometry import refine_circle_subpixel_checked
from ..types import Detection
from .base import BaseDetector


class SpecularBallDetector(BaseDetector):
    name = "specular"

    def __init__(self, cfg: DetectorConfig) -> None:
        super().__init__(cfg)
        self._seed: tuple[float, float, float] | None = None
        self._seed_frame = 0
        self._frame_idx = -1
        self._tracker = None
        self._bbox: tuple[int, int, int, int] | None = None
        self._template: np.ndarray | None = None
        self._prev_patch: np.ndarray | None = None
        self._prev_center: tuple[float, float] | None = None
        self._started = False
        self._frames_since_seed = 0
        self._static_frames = 0
        self._recent = deque(maxlen=25)
        self.last_error = ""
        self._snapshot: dict = {}
        # ---- 回捕状态（见 config.SpecularConfig 的 reacquire_*）----
        self._last_good: tuple[float, float, float] | None = None   # 最后一次正常跟踪的 (x,y,r)
        self._last_patch: np.ndarray | None = None                  # 当时的模板（回捕用）
        self._lost = 0                                              # 连续丢了多少帧
        self._relock: dict | None = None                            # 回捕后的校验窗口
        self._reacquire_cooldown = 0                                # 撤销后的冷却
        self._req_hits: list[tuple[float, float]] = []              # 回捕候选的连续命中
        self._vel = (0.0, 0.0)                                      # 速度（px/帧），用于外推

    # ------------------------------------------------------------------
    def seed(self, x: float, y: float, radius: float, frame_index: int = 0) -> None:
        """人工给一个初始球心/半径；第 frame_index 帧才初始化 CSRT。"""
        self._seed = (float(x), float(y), float(radius))
        self._seed_frame = max(0, int(frame_index))
        self._frame_idx = -1
        self._last_good = None
        self._last_patch = None
        self._lost = 0
        self._req_hits = []
        self._vel = (0.0, 0.0)
        self._tracker = None
        self._bbox = None
        self._template = None
        self._prev_patch = None
        self._prev_center = None
        self._started = False
        self._frames_since_seed = 0
        self._static_frames = 0
        self._recent.clear()
        self.last_error = ""

    def reset(self) -> None:
        """重置跟踪器，但**保留**人工 seed（换源后仍可重新初始化）。"""
        self._tracker = None
        self._bbox = None
        self._template = None
        self._prev_patch = None
        self._prev_center = None
        self._started = False
        self._frames_since_seed = 0
        self._static_frames = 0
        self._recent.clear()
        self._frame_idx = -1

    # ------------------------------------------------------------------
    @staticmethod
    def _bbox_at(x: float, y: float, side: int, shape) -> tuple[int, int, int, int]:
        h, w = shape[:2]
        side = max(8, int(side))
        x0 = int(round(x - side / 2))
        y0 = int(round(y - side / 2))
        x0 = max(0, min(w - side, x0))
        y0 = max(0, min(h - side, y0))
        return x0, y0, side, side

    @staticmethod
    def _patch(gray: np.ndarray, bbox, size: int = 64) -> np.ndarray | None:
        x, y, w, h = bbox
        if w <= 2 or h <= 2:
            return None
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(gray.shape[1], x + w), min(gray.shape[0], y + h)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        p = gray[y0:y1, x0:x1]
        return cv2.resize(p, (size, size), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _ncc(a: np.ndarray, b: np.ndarray) -> float:
        if a is None or b is None or a.shape != b.shape:
            return 0.0
        aa = a.astype(np.float32).ravel()
        bb = b.astype(np.float32).ravel()
        aa -= aa.mean()
        bb -= bb.mean()
        na, nb = float(np.linalg.norm(aa)), float(np.linalg.norm(bb))
        if na < 1e-6 or nb < 1e-6:
            return 0.0
        return float(np.dot(aa, bb) / (na * nb))

    def _make_tracker(self):
        if hasattr(cv2, "TrackerCSRT_create"):
            return cv2.TrackerCSRT_create()
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
            return cv2.legacy.TrackerCSRT_create()
        return None

    def _refine_center(self, gray: np.ndarray, cx: float, cy: float,
                       radius: float) -> tuple[float, float, str, float]:
        """把跟踪框中心钉到球**轮廓**上，返回 ``(cx, cy, 来源, 位移 px)``。

        为什么只取圆心、不用它拟合出的半径：镜面球的**半径**无法从边缘量
        （`B-130`：整圈只有 25%~35% 有梯度，球内强反射的半径又明显更小），
        所以半径仍以人工 seed / CSRT 框为准；而**圆心**必须来自轮廓、不能来自
        高光斑（`B-132`：高光斑随视角落在球面不同位置，左右视图中心天然错位
        约 6 px，直接拿它配对会三角化出一个"看起来正常"的错点）。

        拟合不出足够梯度、或拟合结果离原中心太远（多半吸到了背景边缘/内部反射）
        时，**原样保留框中心**（来源记 ``"box"``）—— 只改善，不劣化。
        """
        c = self.cfg.specular
        if not c.refine_center:
            return cx, cy, "box", 0.0
        search = float(c.refine_center_search_px) or float(np.clip(
            0.25 * radius, c.refine_center_min_search_px, c.refine_center_max_search_px))
        nx, ny, _nr, ok = refine_circle_subpixel_checked(gray, cx, cy, radius,
                                                         search=search)
        if not ok:
            return cx, cy, "box", 0.0
        shift = float(np.hypot(nx - cx, ny - cy))
        if shift > 0.5 * radius:
            return cx, cy, "box", shift
        return float(nx), float(ny), "contour", shift

    # -- 建轨 / 回捕 ----------------------------------------------------
    def _lock_at(self, track_frame, gray: np.ndarray, x: float, y: float, r: float) -> bool:
        """在 (x,y,r) 处建一条新轨（seed 与回捕共用同一份逻辑）。"""
        c = self.cfg.specular
        side = int(c.init_box)
        if r > 45.0:                      # 大球（>90px 直径）才放大框；36px 球仍用 64
            side = max(side, int(round(1.8 * r)))
        side = min(side, int(c.max_box))
        bbox = self._bbox_at(x, y, side, gray.shape)
        tracker = self._make_tracker()
        if tracker is None:
            self.last_error = "OpenCV 没有 CSRT tracker（需要 opencv-contrib）"
            return False
        try:
            tracker.init(track_frame, bbox)
        except Exception as exc:                            # noqa: BLE001
            self.last_error = f"CSRT init 失败：{exc}"
            return False
        self._tracker = tracker
        self._bbox = bbox
        self._template = self._patch(gray, bbox)
        self._prev_center = (x, y)
        self._started = True
        self._frames_since_seed = 0
        self._recent.clear()
        self._recent.append((x, y))
        self.last_error = ""
        return True

    def _search_template(self, gray: np.ndarray, center, r: float,
                         thresh: float, side: float | None):
        """用上一次的模板找球。返回 ``(cx, cy, ncc)`` 或 ``None``。

        ``side`` 给了就在「center ± side」里找（快、误吸风险低）；
        给 ``None`` 就全图找（慢，所以降采样一半，并把阈值抬高 —— 全图找最容易
        吸到台灯、白袋子这类大亮块，那会让"回捕"变成"锁到背景"）。
        """
        tpl = self._last_patch
        if tpl is None or tpl.size == 0:
            return None
        th, tw = tpl.shape[:2]
        if side is None:
            small = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            tpl_s = cv2.resize(tpl, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            if small.shape[0] < tpl_s.shape[0] or small.shape[1] < tpl_s.shape[1]:
                return None
            res = cv2.matchTemplate(small, tpl_s, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(res)
            if mx < thresh:
                return None
            return ((loc[0] + tpl_s.shape[1] / 2.0) * 2.0,
                    (loc[1] + tpl_s.shape[0] / 2.0) * 2.0, float(mx))
        half = int(max(side, 2.0 * r))
        x0 = max(0, int(center[0]) - half)
        y0 = max(0, int(center[1]) - half)
        x1 = min(gray.shape[1], int(center[0]) + half)
        y1 = min(gray.shape[0], int(center[1]) + half)
        if (x1 - x0) < tw or (y1 - y0) < th:
            return None
        roi = gray[y0:y1, x0:x1]
        res = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(res)
        if mx < thresh:
            return None
        return (x0 + loc[0] + tw / 2.0, y0 + loc[1] + th / 2.0, float(mx))

    def _reacquire(self, gray: np.ndarray, track_frame) -> list[Detection]:
        """跟丢之后找回球。返回 [] 表示这一帧还没找回来（不是"永远找不回"）。

        三道闸，缺一不可：
          ① 先等 ``reacquire_after`` 帧：CSRT 偶发抖一下会自己恢复，急着找反而抢；
          ② 局部优先、全图其次：局部窗口以"上次位置 + 速度外推"为中心，
             全图搜索要更久（``reacquire_global_after``）且阈值更高；
          ③ 连续 ``reacquire_confirm`` 帧命中同一处才重建轨 —— 一闪而过的假目标不算。
        """
        c = self.cfg.specular
        if not c.reacquire or self._last_good is None:
            return []
        self._lost += 1
        if self._reacquire_cooldown > 0:
            self._reacquire_cooldown -= 1
            self.last_error = f"回捕冷却中（还剩 {self._reacquire_cooldown} 帧）"
            return []
        if self._lost < c.reacquire_after:
            self.last_error = f"跟踪丢失（第 {self._lost} 帧），{c.reacquire_after} 帧后再找"
            return []

        cx0, cy0, r0 = self._last_good
        vx, vy = self._vel
        k = min(self._lost, 3)                # 最多外推 3 帧的量，飞太远就别猜了
        pred = (cx0 + vx * k, cy0 + vy * k)

        cand = self._search_template(gray, pred, r0, c.reacquire_ncc_local,
                                     side=max(3.0 * r0, 160.0))
        if cand is None and self._lost >= c.reacquire_global_after:
            cand = self._search_template(gray, None, r0, c.reacquire_ncc_global, side=None)
        if cand is None:
            self.last_error = f"回捕中：未找到（已丢 {self._lost} 帧）"
            return []

        mx, my, ncc = cand
        if self._req_hits:
            px, py = self._req_hits[-1]
            if float(np.hypot(mx - px, my - py)) > max(0.8 * r0, 40.0):
                self._req_hits.clear()        # 跳得太远：重新计数
        self._req_hits.append((mx, my))
        if len(self._req_hits) < c.reacquire_confirm:
            self.last_error = (f"回捕候选 ncc={ncc:.2f}，确认中 "
                               f"{len(self._req_hits)}/{c.reacquire_confirm}")
            return []

        if not self._lock_at(track_frame, gray, mx, my, r0):
            return []
        self._lost = 0
        self._req_hits.clear()
        self._last_good = (mx, my, r0)
        self._relock = {"start": (mx, my), "n": 0, "from": "reacquire"}
        rx, ry, csrc, cshift = self._refine_center(gray, mx, my, r0)
        self._snapshot = {"center": (round(mx, 1), round(my, 1)), "radius": round(r0, 1),
                          "ncc": round(ncc, 3), "state": "reacquire",
                          "center_source": csrc}
        return [Detection(x=rx, y=ry, radius=r0,
                          score=float(max(0.0, min(1.0, ncc))), method=self.name,
                          parts={"ncc": float(ncc), "center_shift_px": cshift},
                          extra={"state": "reacquire", "bbox": self._bbox,
                                 "center_source": csrc})]

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray, ctx=None) -> list[Detection]:
        c = self.cfg.specular
        # CSRT 在**彩色图**上明显更稳：实测左路灰度 1/11，BGR 10/11@15px。
        track_frame = frame
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # ctx.gray 可能是放大前的原图（pipeline 的 detect-width 坑），形状不符就不用
        if ctx is not None and getattr(ctx, "gray", None) is not None:
            if ctx.gray.shape[:2] == gray.shape[:2]:
                gray = ctx.gray

        if self._seed is None:
            self.last_error = "specular 需要人工 seed：--seed x,y,r（右路 --seed-right）"
            return []
        self._frame_idx += 1
        if self._frame_idx < self._seed_frame:
            return []

        if self._tracker is None:
            if not self._started:
                x, y, r = self._seed
                if not (c.min_radius <= r <= c.max_radius):
                    # seed 半径只做软提示，不硬拒绝（用户可能量的是外径）
                    self.last_error = (f"seed 半径 {r:.1f} 超出建议范围 "
                                       f"{c.min_radius:.0f}~{c.max_radius:.0f}，仍尝试")
                if not self._lock_at(track_frame, gray, x, y, r):
                    return []
                self._last_good = (x, y, r)
                self._last_patch = self._patch(gray, self._bbox) if self._bbox else None
                self._lost = 0
                self._req_hits.clear()
                rx, ry, csrc, cshift = self._refine_center(gray, x, y, r)
                return [Detection(x=rx, y=ry, radius=r, score=1.0, method=self.name,
                                  parts={"center_shift_px": cshift},
                                  extra={"state": "seed", "bbox": self._bbox,
                                         "center_source": csrc})]
            # 已经建过轨但 tracker 没了 = 跟丢过：交给回捕，而不是从此永远返回空。
            return self._reacquire(gray, track_frame)

        ok, bbox = self._tracker.update(track_frame)
        if not ok or bbox is None:
            self.last_error = "CSRT 跟丢；不 snap 到背景圆，请检查 seed 或重新录制"
            self._tracker = None
            return []
        x, y, w, h = [int(round(v)) for v in bbox]
        cx, cy = x + w / 2.0, y + h / 2.0
        radius = max(w, h) / 2.0
        self._frames_since_seed += 1

        if not (c.min_radius <= radius <= c.max_radius):
            self.last_error = f"跟踪框半径 {radius:.1f} 超出 {c.min_radius:.0f}~{c.max_radius:.0f}"
            return []

        jump = 1e9
        if self._prev_center is not None:
            jump = float(np.hypot(cx - self._prev_center[0], cy - self._prev_center[1]))
            if jump > c.max_jump_px:
                self.last_error = f"单帧跳 {jump:.1f}px > {c.max_jump_px:.0f}，判为跟丢"
                self._tracker = None
                return []

        patch = self._patch(gray, (x, y, w, h))
        ncc = self._ncc(self._template, patch) if patch is not None else 0.0
        motion = 0.0
        if self._prev_patch is not None and patch is not None:
            motion = float(cv2.absdiff(patch, self._prev_patch).mean())
        if motion < c.static_motion_eps and jump < 1.0:
            self._static_frames += 1
        else:
            self._static_frames = 0
        if self._static_frames > c.max_static_frames:
            self.last_error = (f"连续 {self._static_frames} 帧中心/局部都不动，"
                               "判为丢了（不再报背景）")
            self._tracker = None
            return []
        if c.min_ncc > 0 and ncc < c.min_ncc:
            self.last_error = f"模板 NCC {ncc:.2f} < {c.min_ncc:.2f}，判为跟丢"
            self._tracker = None
            return []
        if (c.min_motion > 0
                and self._frames_since_seed > c.motion_grace_frames
                and motion < c.min_motion and jump < 1.0):
            self.last_error = (f"局部运动 {motion:.1f} < {c.min_motion:.1f} 且中心几乎不动，"
                               "判为锁到静止背景")
            self._tracker = None
            return []

        self._bbox = (x, y, w, h)
        self._prev_patch = patch
        self._prev_center = (cx, cy)
        self._recent.append((cx, cy))
        if len(self._recent) >= self._recent.maxlen:
            xs = [p[0] for p in self._recent]
            ys = [p[1] for p in self._recent]
            if max(xs) - min(xs) < 5.0 and max(ys) - min(ys) < 5.0:
                self.last_error = (f"最近 {self._recent.maxlen} 帧中心跨度 <5px，"
                                   "判为丢了（不再报静止背景）")
                self._tracker = None
                return []
        # 回捕后的校验：静止亮块（灯管/白墙）外观和球很像，靠"有没有位移"区分。
        # 只在回捕建立的那条轨上查，正常 seed 的轨不受影响（球可能就该不动）。
        if self._relock is not None:
            self._relock["n"] += 1
            if self._relock["n"] >= self.cfg.specular.reacquire_verify_frames:
                sx, sy = self._relock["start"]
                travel = float(np.hypot(cx - sx, cy - sy))
                need = self.cfg.specular.reacquire_min_travel_px
                self._relock = None
                if travel < need:
                    self.last_error = (f"回捕后 {self.cfg.specular.reacquire_verify_frames} 帧只移动 "
                                       f"{travel:.0f}px < {need:.0f}px，判为锁到静止物，撤销")
                    self._tracker = None
                    self._reacquire_cooldown = self.cfg.specular.reacquire_cooldown
                    return []
        # 记账：模板与速度都给"回捕"用 —— 没有这两样，回捕只能瞎猜。
        self._last_good = (cx, cy, radius)
        self._last_patch = patch
        self._lost = 0
        self._req_hits.clear()
        if len(self._recent) >= 2:
            vx = self._recent[-1][0] - self._recent[-2][0]
            vy = self._recent[-1][1] - self._recent[-2][1]
            self._vel = (0.35 * self._vel[0] + 0.65 * vx,
                         0.35 * self._vel[1] + 0.65 * vy)
        # ⚠️ 所有门控（跳变 / 模板 / 静止）都用**框中心**：它们是"CSRT 有没有跟丢"
        # 的判据，混进口径不同的轮廓圆心只会让门限变得无法解释。轮廓圆心只在
        # 出口处替换，供三角化用。
        rx, ry, csrc, cshift = self._refine_center(gray, cx, cy, radius)
        self._snapshot = {
            "center": (round(cx, 1), round(cy, 1)), "radius": round(radius, 1),
            "ncc": round(ncc, 3), "motion": round(motion, 2), "jump": round(jump, 2),
            "bbox": (x, y, w, h), "frame": self._frames_since_seed,
            "center_source": csrc, "center_shift_px": round(cshift, 2),
        }
        self.last_error = ""
        return [Detection(
            x=rx, y=ry, radius=radius,
            score=float(max(0.0, min(1.0, 0.6 * max(0.0, ncc) + 0.4 * min(1.0, motion / 30.0)))),
            method=self.name,
            parts={"ncc": float(ncc), "motion": float(motion), "jump": float(jump),
                   "center_shift_px": cshift},
            extra={"state": "track", "bbox": (x, y, w, h),
                   "frames_since_seed": self._frames_since_seed,
                   "center_source": csrc},
        )]

    def debug_snapshot(self) -> dict:
        return dict(self._snapshot, error=self.last_error, has_seed=self._seed is not None)
