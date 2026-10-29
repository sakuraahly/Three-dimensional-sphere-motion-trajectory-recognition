"""标定板的两个"拍完才发现就晚了"的判据 —— **拍摄现场与拍完自检的唯一来源**。

为什么要有这个库（2026-10-28 的真实事故）
----------------------------------------
当天用户拍了 33 张（left 15 / right 15 / ref 3+3）跑 90° 单板链式标定，
等了十几分钟才被拒：left 内参 RMS 2.18 px、参考图板位移 11.83°。
事后复核，四个问题**在拍完那一刻就已经存在**：

1. 内参段板一直在画面中央 —— 15 张图只覆盖 5 个区域里的 2 个。
   畸变系数不可辨，内参与主点/畸变互相耦合，RMS 必然大。
2. 参考段的板在挪 —— ref_left 三张之间同一角点位移 148.4 px。
   单板模式的前提是"参考图期间板不动"（T_AB=I），板动了两路的
   "同一个位置"就不是同一个位置，链出来的外参是错的。

这两条判据原本写死在 ``apps/tools/check_chain_inputs.py`` 里，只能**拍完之后**
跑；现在抽成库，采集窗口的 HUD 在拍摄时就能说话，不用再等标定跑完。

判据来源（一个数都没有重新定义）
--------------------------------
* 区域划分：直接复用 :func:`calib.mono_calib._region_index`（3×3 网格的
  四角 + 中央共 5 个区域），**不在这里复制一份坐标判断**；
* 区域总数与"至少覆盖几个"：``calib.mono_calib.QUALITY`` 的
  ``regions_total`` / ``min_regions`` —— 那是全项目质量门限的唯一来源；
* 参考段门限 :data:`REF_STILL_PX`：从 check_chain_inputs 原样搬来，
  该文件此后从本模块导入，不再自己定义。

本模块是**纯函数**：不读文件、不开相机、不建窗口，采集端与自检端可以
按帧调用，自测也能直接喂合成角点。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from .mono_calib import QUALITY, _region_index

__all__ = [
    "REF_STILL_PX", "REGIONS_TOTAL", "MIN_REGIONS",
    "region_of", "coverage_regions", "coverage_ok", "region_boxes",
    "reference_shift",
]

#: 参考段"板没动"的判据：同一角点在不同张之间的最大位移（像素）。
#: 实测：板不动时 < 2 px（亚像素抖动）；板挪 11° 时几十上百 px。
#: 门限取 5 px 是"手完全离开三脚架/桌面"与"只是呼吸带来的晃动"之间的分界。
REF_STILL_PX = 5.0

#: 区域总数（3×3 网格的四角 + 中央）与"至少覆盖几个"。
#: 一律取自 ``QUALITY``，这样内参标定拒收、拍完自检、现场 HUD 三处的口径
#: 永远同一条 —— 今天的事故正是"三处各自判断"的代价。
REGIONS_TOTAL: int = int(QUALITY["regions_total"])
MIN_REGIONS: int = int(QUALITY["min_regions"])


def region_of(corners: np.ndarray | None, shape: Sequence[int]) -> int:
    """一张图里的板落在哪个区域（0..4；不算数返回 -1）。

    参数
    ----
    corners
        该张图的角点，形状 ``(K, 2)``；``None`` 或空表示这一张没检出。
    shape
        图像形状，按 ``(h, w)`` 给（与 ``frame.shape[:2]`` 同序）。

    用板的**中心**判区域（而不是外接框），因为要判的是"这块板摆到了画面的
    哪个位置"，中心点足以表达；而且与 ``calibrate_mono`` 内部的
    ``pose_coverage`` 计算方式保持一致，否则两处会对同一张图给出不同的区域。
    """
    if corners is None:
        return -1
    c = np.asarray(corners, np.float32).reshape(-1, 2)
    if not len(c):
        return -1
    h, w = int(shape[0]), int(shape[1])
    return _region_index(float(c[:, 0].mean()), float(c[:, 1].mean()), w, h)


def coverage_regions(
        corners_list: Iterable[tuple[np.ndarray | None, Sequence[int]]]
) -> tuple[set[int], int]:
    """一批图覆盖了哪几个区域。返回 ``(覆盖到的区域编号集合, 区域总数)``。

    ``corners_list`` 的每项是 ``(角点 (K,2), 图像形状 (h, w))``：
    区域划分依赖画面尺寸，只有角点算不出位置，所以两者必须成对给。
    没检出 / 落在四边中点的张不计入（``_region_index`` 返回 -1）。

    返回**集合**而不是计数：HUD 要标出"哪几个格子还没覆盖"，
    只有一个数字是画不出来的。
    """
    covered: set[int] = set()
    for corners, shape in corners_list:
        idx = region_of(corners, shape)
        if idx >= 0:
            covered.add(idx)
    return covered, REGIONS_TOTAL


def coverage_ok(covered: Iterable[int], need: int | None = None) -> bool:
    """这批图够不够 ``MIN_REGIONS``（默认取自 ``QUALITY``）。

    ``need`` 只为自测能收紧/放开门限而留 —— 判据可自证，不是生产旋钮。
    """
    goal = MIN_REGIONS if need is None else int(need)
    return len(set(covered)) >= goal


def region_boxes(w: int, h: int) -> dict[int, tuple[int, int, int, int]]:
    """5 个区域的矩形 ``{编号: (x0, y0, x1, y1)}``，给 HUD 画覆盖状态用。

    刻意复用 ``_region_index`` 的同一套 3×3 划分（把每个格子的边界算出来），
    而不是另写一份 —— HUD 上画的格子与判据判出来的区域必须是**同一块地**，
    否则用户会看到"我明明摆进那个格子了，它却说没覆盖"。
    """
    w, h = max(1, int(w)), max(1, int(h))
    tx, ty = w / 3.0, h / 3.0
    cells = {(0, 0): 0, (0, 2): 1, (2, 0): 2, (2, 2): 3}
    boxes: dict[int, tuple[int, int, int, int]] = {}
    for (row, col), idx in cells.items():
        boxes[idx] = (int(col * tx), int(row * ty),
                      int((col + 1) * tx), int((row + 1) * ty))
    boxes[4] = (int(tx), int(ty), int(2 * tx), int(2 * ty))     # 中央
    return boxes


def reference_shift(
        corners_list: Sequence[np.ndarray]
) -> tuple[float, tuple[int, int] | None]:
    """参考段多张之间同一角点的最大位移。返回 ``(位移 px, (基准张, 最差张))``。

    以**第一张**为基准逐张比：参考段的意义是"这几张是同一个位姿"，
    拿相邻两张比会漏掉"缓慢单向漂移"（每步 4 px，四张就漂了 12 px）。

    位移按**坐标分量绝对值的最大**算（``max|dx|, max|dy|``），不是欧氏距离 ——
    这是 check_chain_inputs 原本的算法，原样搬来不改语义：换成欧氏距离会让
    同一批图的读数变一点点，"改判据顺手改了数"正是最难查的那种回归。
    分量口径也符合物理直觉：板沿单轴平移时它就是平移量。

    张数不足 2 时返回 ``(0.0, None)``：没有可比的对象 ≠ 板动了。
    调用方该用"有没有基准张"来决定是否显示，而不是把 0 当成"很稳"。
    """
    arrays = [np.asarray(c, np.float32).reshape(-1, 2) for c in corners_list]
    arrays = [a for a in arrays if len(a)]
    if len(arrays) < 2:
        return 0.0, None
    base = arrays[0]
    worst, worst_idx = 0.0, None
    for i, c in enumerate(arrays[1:], start=1):
        s = float(np.abs(c - base).max())
        if s > worst:
            worst, worst_idx = s, i
    return worst, (0, worst_idx) if worst_idx is not None else None
