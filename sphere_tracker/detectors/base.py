"""检测器基类与公共工具。"""

from __future__ import annotations

import cv2
import numpy as np
from ..config import DetectorConfig
from ..types import Detection
from pathlib import Path

# CSRT 内部用 OpenCV 的**进程级全局 RNG** 做采样。不固定它，同一段视频每次启动
# 进程的结果都不一样 —— 2026-11-01 实测同一段 302 帧：222 / 198 / 283 帧检出。
# 对一个"要拿数字比精度"的工具，这是致命的：A/B 的差异会被这层噪声盖住
# （我因此误判过一次：把"左路第 100 帧跟丢"当成真实现象，其实是当次的随机结果）。
DETERMINISTIC_RNG_SEED = 20261031
_rng_seeded = False


def ensure_deterministic_rng() -> None:
    """把 OpenCV 全局 RNG 固定下来；**进程内只做一次**。

    为什么是"一次"而不是每个检测器各来一次：这是全局状态，运行中途重置会让
    先建好的检测器与后建好的走不同序列，"同一段视频两次跑不一样"会换个形式回来。
    """
    global _rng_seeded
    if _rng_seeded:
        return
    setter = getattr(cv2, "setRNGSeed", None)
    if setter is not None:
        setter(DETERMINISTIC_RNG_SEED)
    _rng_seeded = True


def resolve_weights(weights: str) -> str:
    """把权重路径解析成实际可用的路径。

    规则：
      1. 绝对路径 -> 原样返回（不存在就退回文件名，交给 ultralytics 处理）；
      2. 相对路径 -> 相对**项目根目录**解析（而不是当前工作目录），
         这样在 apps/ 或任意目录下运行都能找到 models/yolo11n.pt；
      3. 项目里找不到 -> 只返回文件名，让 ultralytics 自己去缓存目录找或下载。
    """
    if not weights:
        return weights
    p = Path(weights)
    if p.is_absolute():
        return str(p) if p.exists() else p.name
    # 本文件在 sphere_tracker/detectors/base.py：
    #   parents[0]=detectors  parents[1]=sphere_tracker  parents[2]=**仓库根**
    # 2026-10-30 之前这里写的是 parents[1]，于是 `models/yolo11n.pt` 永远解析不到，
    # 静默退回裸文件名 `yolo11n.pt` —— ultralytics 会去自己的缓存目录找/下载，
    # 表面一切正常，实际用的**不是仓库里那份权重**。自训权重更直接：报
    # FileNotFoundError，而错误信息里显示的仍是配置里的旧名字，指不到真因。
    root = Path(__file__).resolve().parents[2]
    candidate = root / p
    if candidate.exists():
        return str(candidate)
    return p.name


class BaseDetector:
    name = "base"

    def __init__(self, cfg: DetectorConfig) -> None:
        self.cfg = cfg
        # 所有检测器共用一个确定性入口：谁先被构造谁把 RNG 钉住。
        ensure_deterministic_rng()

    # ctx 是可选的共享上下文（灰度图 / 相机自运动），由 pipeline 每帧构造一次。
    # 不需要它的后端直接忽略即可。
    def detect(self, frame: np.ndarray, ctx=None) -> list[Detection]:
        raise NotImplementedError

    def reset(self) -> None:
        """换视频源时调用（背景模型需要重置）。"""

    # -- 点选锚点：跨检测器的**唯一**构造点 --------------------------------
    def anchor_detection(self, x: float, y: float, radius: float, *,
                         extra: dict | None = None, note: str = "") -> Detection:
        """把「用户点选的位置」当作本帧输出。

        为什么收在基类（2026-11-07，同一个毛病第三次之后）
        --------------------------------------------------
        `color` / `generic` / `specular` 原来各写一份构造，于是同一个概念
        有三种说法：`method` 有的写 ``name:seed``、有的只写 ``name``；
        `extra` 有的带 ``state``/``center_source``、有的只带 ``seed_anchor``；
        `last_error` 有的写原因、有的不写。后果不是"风格不统一"，而是
        `pipeline` 只认 ``extra["seed_anchor"]`` —— `specular` 那一路连键都
        没有，锚点被当成真检测**静默**放过去了，直到第二轮 Critic 才抓出来。
        B-150（偏好被写回）也是同型：同一件事有三处实现，改一处漏两处。

        **加字段请加参数，不要各自 new 一个 `Detection`。**
        判据：``tests/selftest_plugins.py::test_anchor_single_source``
        （AST 扫全树，``extra`` 里出现 ``seed_anchor`` 的 `Detection` 构造只许有一个）。

        ``note`` 是「为什么退回锚点」的一句话，会写进 ``self.last_error``。
        检测器自己知道原因就传（`specular` 是锁定失败、`generic` 是门内无候选）；
        说不出来就留空，`pipeline` 会用一句通用的话兜住 —— **不许静默**。

        语义边界：`x/y/r` 全部来自用户点击，不是算法测量；`score=1.0` 表示
        「这是人给的」，不表示「算法很确信」。
        """
        if note:
            prev = str(getattr(self, "last_error", "") or "")
            # 追加而不是覆盖：`specular` 在锁定前可能刚写过「seed 半径超出
            # 建议范围」，那条也是真的，不该被锚点这句吃掉。
            self.last_error = f"{prev}；{note}" if prev else str(note)
        # 标记写成**构造点上的字面量**：`selftest_plugins` 用 AST 扫「谁在
        # `extra` 里塞了 `seed_anchor`」，写成先组装 dict 再传会扫不到 ——
        # 那样判据就变成永远判绿的空闸门。`seed_anchor` 放在 `**` 之后，
        # 调用方传什么都覆盖不掉它。
        return Detection(x=float(x), y=float(y), radius=float(radius),
                         score=1.0, method=f"{self.name}:seed",
                         parts={"seed_anchor": 1.0},
                         extra={**(extra or {}), "seed_anchor": True})

    # -- 公共打分/筛选 -----------------------------------------------------
    def _score_candidate(
        self,
        radius: float,
        circularity: float = 0.0,
        fill_ratio: float = 0.0,
        radial: float = 0.0,
        edge_support: float = 0.0,
        weights: tuple[float, float, float, float] = (0.30, 0.28, 0.27, 0.15),
    ) -> float:
        """把各种形状指标合成 0-1 的置信度。

        四项指标各有侧重，缺一不可：
          圆度 circularity   排除锯齿状/细长目标
          填充率 fill        排除方块（方块只能填满外接圆的 ~61%）
          径向一致性 radial   排除方块/胶囊/多目标粘连（最有判别力的一项）
          边缘支持度 support  排除"颜色对但边缘不像球"的区域
        """
        if radius < self.cfg.min_radius or radius > self.cfg.max_radius:
            return 0.0
        if radial > 0.0 and radial < self.cfg.min_radial_consistency:
            return 0.0
        if fill_ratio > 0.0 and fill_ratio < self.cfg.min_fill_ratio:
            return 0.0
        size_score = 1.0
        # 半径贴着上下限时降权，避免"卡在阈值边缘"的误检
        if radius < self.cfg.min_radius * 1.3:
            size_score = 0.8
        if radius > self.cfg.max_radius * 0.8:
            size_score = 0.8
        w1, w2, w3, w4 = weights
        total_w = 0.0
        shape = 0.0
        for value, weight in (
            (circularity, w1), (fill_ratio, w2), (radial, w3), (edge_support, w4)
        ):
            if value <= 0.0:
                continue
            shape += value * weight
            total_w += weight
        if total_w <= 0.0:
            return 0.0
        return float(max(0.0, min(1.0, shape / total_w * size_score)))
