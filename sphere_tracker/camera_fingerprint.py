"""相机画面指纹：用“看到的场景”认相机，而不是名字/端口/编号。

## 用途

两台同型号、无唯一序列号的 UVC 相机，拔插后 Windows 名字/端口/OpenCV 编号
都可能变。画面指纹把它们看到的场景（床 / 阳台门）当作身份：一次登记，
之后按指纹匹配回 left/right。**分不清就返回 None，不猜。**

## 边界

- 相机挪位置/角度或场景剧变后，旧指纹失效，需要重新登记。
- 两台相机看到完全相同的画面时，任何指纹都无法区分；那时只能靠唯一序列号/人工。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Fingerprint:
    """降采样灰度（去均值归一化）+ HSV 2D 直方图。"""

    gray: np.ndarray
    hist: np.ndarray


def make_fingerprint(frame: np.ndarray) -> Fingerprint:
    """从 BGR 帧生成指纹；空帧/非 3 通道抛 ValueError。"""
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ValueError("空帧不能生成相机指纹")
    if frame.ndim == 2:
        gray = frame
        bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 3:
        bgr = frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"只支持灰度/3 通道 BGR，收到 shape={frame.shape}")
    small = cv2.resize(gray, (64, 36)).astype(np.float32)
    small -= float(small.mean())
    norm = float(np.linalg.norm(small))
    if norm > 0:
        small /= norm
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
    return Fingerprint(gray=small.ravel(), hist=hist)


def similarity(a: Fingerprint, b: Fingerprint) -> float:
    """0..1（可能略负）：0.7 灰度相关 + 0.3 HSV 直方图相关。"""
    gray = float(np.dot(a.gray, b.gray))
    hist = float(cv2.compareHist(a.hist, b.hist, cv2.HISTCMP_CORREL))
    return 0.7 * gray + 0.3 * hist


def match_fingerprint(
    fp: Fingerprint,
    refs: dict[str, Fingerprint],
    *,
    min_score: float = 0.45,
    min_margin: float = 0.10,
) -> tuple[str, float] | None:
    """返回最佳 ``(role, score)``；低于阈值或与第二名差距不够 -> None。

    这是**拒绝猜**的实现：两台场景相似时 margin 会很小，返回 None 比
    静默选一个更安全。
    """
    if not refs:
        return None
    scored = sorted(((similarity(fp, ref), name) for name, ref in refs.items()),
                    reverse=True)
    best_score, best_name = scored[0]
    second = scored[1][0] if len(scored) > 1 else -1.0
    if best_score < min_score or best_score - second < min_margin:
        return None
    return best_name, best_score


__all__ = ["Fingerprint", "make_fingerprint", "similarity", "match_fingerprint"]
