"""画质指标：清晰度 / 亮度 / 反差 / 过曝 / 死黑。

**纯计算**：只吃灰度帧（`np.ndarray` 列表），不碰相机、不读文件、不依赖 `cv2.VideoCapture`。
正因为纯计算，`.apps` 与 `.stereo` 可以共用同一份口径 ——
"一套清晰度算法"是这里的全部意义：若各写一份，两个工具迟早给出两个数。

调用方：
- `apps/tools/camera_quality.py`（实时相机会诊，CLI 入口）
- `stereo/sessions.py`（会话校验时的"清晰度偏低"提示，吃录像帧）

本模块在库层（`sphere_tracker/`），**不许反向 import `apps.`**。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FrameMetrics:
    """一组成片的画质指标（中位数）。

    抽成独立纯函数是为了让**只有一套清晰度算法**：校验器（``stereo/sessions.py``）
    也要给"清晰度"提示，若各写一份，两个工具迟早给出两个数。
    """
    sharp: float = 0.0
    bright: float = 0.0
    contrast: float = 0.0
    over: float = 0.0
    under: float = 0.0


def frames_quality(grays: list[np.ndarray]) -> FrameMetrics:
    """中景（避开边缘暗角）的清晰度 / 亮度 / 反差 / 过曝 / 死黑，各取中位数。

    纯计算：只吃灰度帧，不碰相机、不读文件。空输入返回全 0。
    """
    if not grays:
        return FrameMetrics()
    h, w = grays[0].shape[:2]
    cy0, cy1 = int(h * 0.25), int(h * 0.75)
    cx0, cx1 = int(w * 0.25), int(w * 0.75)
    sharps, brights, contrasts, overs, unders = [], [], [], [], []
    for g in grays:
        crop = g[cy0:cy1, cx0:cx1]
        if crop.size < 100:
            crop = g
        sharps.append(float(cv2.Laplacian(crop, cv2.CV_64F).var()))
        brights.append(float(crop.mean()))
        contrasts.append(float(crop.std()))
        overs.append(float((crop > 245).mean()))
        unders.append(float((crop < 12).mean()))
    return FrameMetrics(
        sharp=float(np.median(sharps)),
        bright=float(np.median(brights)),
        contrast=float(np.median(contrasts)),
        over=float(np.median(overs)),
        under=float(np.median(unders)),
    )
