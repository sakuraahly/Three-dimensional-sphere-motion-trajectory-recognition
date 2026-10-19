"""深度来源与融合：双目视差 / 单目球径 / 逆方差加权。

这不是"再算一遍深度"，而是把两个**独立**的深度观测按不确定度合成：

    Z_stereo = fx * B / d          （双目，长距离更可信）
    Z_mono   = fx * D / (2 r)      （球径先验，近处/小基线时可能更稳）
    Z_fused  = (Z_s/σ_s² + Z_m/σ_m²) / (1/σ_s² + 1/σ_m²)

双目和单目都来自同一台相机的 fx；两者不是完全独立（球径先验会错），
所以 `sigma` 必须由调用方给出，不能假装融合一定更好。
"""

from __future__ import annotations

import math


def stereo_depth(disparity_px: float, fx_px: float, baseline_m: float) -> float:
    """视差 -> 深度（米）。视差用绝对值；0 视差返回 inf。"""
    d = abs(float(disparity_px))
    if d < 1e-9 or fx_px <= 0 or baseline_m <= 0:
        return math.inf
    return float(fx_px) * float(baseline_m) / d


def mono_depth(radius_px: float, fx_px: float, diameter_m: float) -> float:
    """球半径 -> 深度（米）：Z = fx * D / (2 r)。半径<=0 返回 inf。"""
    r = float(radius_px)
    if r <= 0 or fx_px <= 0 or diameter_m <= 0:
        return math.inf
    return float(fx_px) * float(diameter_m) / (2.0 * r)


def fuse_depth(
    z_stereo: float,
    sigma_stereo_m: float,
    z_mono: float,
    sigma_mono_m: float,
) -> tuple[float, float]:
    """逆方差加权融合，返回 (Z, σ)。

    任何一路无效（inf/nan/σ<=0）就退化为另一路；两路都无效返回 (inf, inf)。
    """
    def ok(z: float, s: float) -> bool:
        return math.isfinite(z) and math.isfinite(s) and s > 0

    s_ok, m_ok = ok(z_stereo, sigma_stereo_m), ok(z_mono, sigma_mono_m)
    if s_ok and m_ok:
        ws = 1.0 / (sigma_stereo_m ** 2)
        wm = 1.0 / (sigma_mono_m ** 2)
        z = (z_stereo * ws + z_mono * wm) / (ws + wm)
        return float(z), float(math.sqrt(1.0 / (ws + wm)))
    if s_ok:
        return float(z_stereo), float(sigma_stereo_m)
    if m_ok:
        return float(z_mono), float(sigma_mono_m)
    return math.inf, math.inf
