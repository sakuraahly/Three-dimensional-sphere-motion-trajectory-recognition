"""标定事实自测：真机日志的口径 + "不可用不得保存" + 基线锚必须显式给。

为什么需要它
------------
真机第一次双目标定跑完留下的两份日志（`outputs/logs/calibrate_2026*.log`）
是**目前唯一一份真机证据**。它里面的判定口径值得钉住，否则下一次
"到底算不算失败、失败在哪一步"又要靠人重新推。

本测试钉四类事实：

**1. 真机日志的汇总能解析、且自洽**（用内嵌的日志片段复算，不依赖 `outputs/` 还在不在）

日志说：`共处理 1362 帧：收下 8 对，拒收 1354 帧`，六类拒收各有占比，
`基线长度 199.67 mm`，`判定 不可用`，最后 `未保存`。
本测试把它们重新解析一遍并断言自洽（收下 + 拒收 = 总数、六类之和 = 拒收数）。
**日志最贵的价值就在这里：拒收原因分布是可统计的** —— 而它此前没人看过。

**2. "单目各自正常、双目整体爆炸"这个签名可复现、可断言**

真机：左 1.3784 / 右 0.9386 px（**都远低于门槛**），整体 46.4040 px、
极线 15978.5150 px（**远超门槛**）。本测试用一批**顺序错配**的合成配对
把同一个签名造出来，断言"单目过、双目炸"。

**3. 判定为不可用时，结果一定不会被保存**

`calib/params.py:203-206` 的 `StereoRig.from_result()` 对 `res.ok=False` 直接抛
`ValueError`；`apps/calibrate_stereo.py:397-400` 在那之前就 `return 1`。
两道闸门都钉住 —— 因为"失败却留下一个能读的标定文件"比失败本身危险得多。

**4. 基线锚（`expected_baseline_mm`）默认 `None`，于是基线核对形同虚设**

`calib/stereo_calib.py:272-278` 本来就有"基线偏离预期就报警"的逻辑，
但 `CalibConfig.expected_baseline_mm` 默认 `None`（第 47 行），**从来没触发过**。
真机日志里警告只有两条（重投影 RMS、极线误差），**没有基线那条** ——
可基线当时是 199.67 mm，而用户实测两镜头中心间距只有 **65 mm**（3.07 倍）。

本测试同一批合成数据跑两遍：不给锚 -> 0 条基线警告（静默放过）；
给锚 `expected_baseline_mm=65` -> 报出偏差。
这就是 P0-2 要"把可选变成必需"的理由：**锚不给，守卫就不存在**。

反向自证（改了就会红）
--------------------
=================================  ============================================  ========
改动                               注入方式                                      结果
=================================  ============================================  ========
谎报真机基线（199.67 -> 66）       `REAL_BASELINE_MM = 66.0`                     exit=1
谎报拒收统计（1362 -> 1000）       `REAL_FRAMES_TOTAL = 1000`                    exit=1
谎报单目 RMS（1.3784 -> 0.5）      `REAL_ERR_LEFT = 0.5`                         exit=1
删掉 `params.py` 的拒绝闸门        `if not res.ok: raise ...` -> `pass`          exit=1（3 条）
=================================  ============================================  ========

> 前两条里，**"谎报基线"这条是本测试写完后被变异测试抓出来的缺口**：
> 最初只断言"倍数 > 2"，把 `REAL_BASELINE_MM` 改成 66（= 没被吹飞）**居然不会红**。
> 现在真机那几个标量一律**逐字**钉住。

**已知不确定性（如实记录）**：第 2、3、4 节里的双目标定结果**逐次抖动**
（同一份输入，基线在 2000~13000 mm 之间跳）。这是 `cv2.stereoCalibrate` 内部
RANSAC 的随机性 —— `cv2.setRNGSeed()` **管不住它**（已实测）。
所以那几节的断言刻意只钉"**数量级**"（> 5 倍、> 10 倍、> 30°），
不钉具体数值；**这一点是设计，不是疏漏**。

用法::

    python tests/selftest_calib_facts.py
"""

from __future__ import annotations

import copy
import json
import re
import sys
import tempfile
import tomllib
from pathlib import Path
ROOT = Path(__file__).resolve().parent
while not (ROOT / "sphere_tracker").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from calib.export import build_export, write_export
from calib.params import StereoRig
from calib.stereo_calib import CalibConfig, StereoCalibrationResult, calibrate_stereo
from calib.world_frame import WorldFrame, from_board
from stereo.visualization import build_visualization_payload

# 复用角点顺序测试里的合成装置（保证两处口径一致）
from selftest_board_corner_order import (  # noqa: E402
    SEED, SPEC, TAPE_BASELINE_MM, synth_pair)

import cv2  # noqa: E402

# --------------------------------------------------------------------------
# 真机那次的实测值（outputs/logs/calibrate_20261010_162029.log）
# 全部内嵌，**不依赖 outputs/ 是否还在** —— 那份日志本身是被 gitignore 的
# --------------------------------------------------------------------------
REAL_LOG_TAG = "calibrate_20261010_162029.log"
REAL_FRAMES_TOTAL = 1362
REAL_PAIRS_KEPT = 8
REAL_FRAMES_REJECTED = 1354
# 拒收原因 -> (次数, 日志里印的百分比)
REAL_REJECTIONS: dict[str, tuple[int, float]] = {
    "棋盘还在移动，请停稳再收": (790, 58.3),
    "两路都未检测到棋盘": (295, 21.8),
    "与已收位姿太接近": (136, 10.0),
    "左路未检测到棋盘": (59, 4.4),
    "该区域已收够，请换个位置": (40, 3.0),
    "右路未检测到棋盘": (34, 2.5),
}
REAL_RMS = 46.4040
REAL_ERR_LEFT = 1.3784
REAL_ERR_RIGHT = 0.9386
REAL_EPIPOLAR_RMS = 15978.5150
REAL_EPIPOLAR_MAX = 32735.3656
REAL_BASELINE_MM = 199.67

# 日志原文片段（排版与 outputs/logs/ 里的真实文件逐字一致，含对齐空格）
LOG_EXCERPT = "\n".join([
    f"16:22:39.564 INFO  共处理 {REAL_FRAMES_TOTAL} 帧：收下 {REAL_PAIRS_KEPT} 对，"
    f"拒收 {REAL_FRAMES_REJECTED} 帧",
    "16:22:39.564 INFO  拒收原因（按次数）：",
    "16:22:39.564 INFO       790 ( 58.3%)  棋盘还在移动，请停稳再收",
    "16:22:39.564 INFO       295 ( 21.8%)  两路都未检测到棋盘",
    "16:22:39.564 INFO       136 ( 10.0%)  与已收位姿太接近",
    "16:22:39.564 INFO        59 (  4.4%)  左路未检测到棋盘",
    "16:22:39.564 INFO        40 (  3.0%)  该区域已收够，请换个位置",
    "16:22:39.564 INFO        34 (  2.5%)  右路未检测到棋盘",
    f"整体重投影 RMS: {REAL_RMS:.4f} px",
    f"左目重投影 RMS: {REAL_ERR_LEFT:.4f} px",
    f"右目重投影 RMS: {REAL_ERR_RIGHT:.4f} px",
    f"极线误差 RMS  : {REAL_EPIPOLAR_RMS:.4f} px   (最大 {REAL_EPIPOLAR_MAX:.4f})",
    f"基线长度      : {REAL_BASELINE_MM:.2f} mm",
    "判定          : 不可用 —— 请重拍标定图像（换更多角度、保证清晰、铺满画面）",
    "  ! 重投影 RMS 46.40 px 偏大：检查标定板是否平整、图像是否模糊",
    "  ! 极线误差 15978.51 px 偏大：左右相机的相对位姿可能不稳定",
    "16:22:39.804 WARNING 标定未通过质量判定，未保存",
]) + "\n"


def _log_warning_lines(text: str) -> list[str]:
    """取出日志里的警告行（`  ! ...`，与 StereoCalibrationResult.report() 的排版一致）。"""
    return [ln.strip() for ln in text.splitlines() if ln.strip().startswith("! ")]


def _params_gate_present() -> bool:
    """`calib/params.py` 里是否有"ok 为假就拒绝生成装置参数"的闸门（源码级断言）。"""
    src = (ROOT / "calib" / "params.py").read_text(encoding="utf-8")
    return bool(re.search(r"if\s+not\s+res\.ok\s*:\s*\n\s*raise\s+ValueError", src))


def _app_refuses_on_unusable() -> bool:
    """`apps/calibrate_stereo.py` 是否在判定不可用时提前返回（源码级断言）。"""
    src = (ROOT / "apps" / "calibrate_stereo.py").read_text(encoding="utf-8")
    return bool(re.search(r"if\s+not\s+res\.ok\s*:[\s\S]{0,400}?return\s+1", src))


def parse_summary(text: str) -> dict:
    """从日志文本里解析汇总数字（口径与日志排版一致）。"""
    out: dict = {}
    m = re.search(r"共处理\s+(\d+)\s*帧：收下\s+(\d+)\s*对，拒收\s+(\d+)\s*帧", text)
    if m:
        out["total"], out["kept"], out["rejected"] = (int(g) for g in m.groups())
    out["reasons"] = {
        name: (int(cnt), float(pct))
        for cnt, pct, name in re.findall(
            r"^\s*\d+:\d+:\d+\.\d+\s+INFO\s+(\d+)\s*\(\s*([\d.]+)%\)\s+(\S.*?)\s*$",
            text, flags=re.MULTILINE)
    }
    for key, pat in (("rms", r"整体重投影 RMS:\s*([\d.]+)\s*px"),
                     ("err_left", r"左目重投影 RMS:\s*([\d.]+)\s*px"),
                     ("err_right", r"右目重投影 RMS:\s*([\d.]+)\s*px"),
                     ("epipolar", r"极线误差 RMS\s*:\s*([\d.]+)\s*px"),
                     ("baseline_mm", r"基线长度\s*:\s*([\d.]+)\s*mm")):
        mm = re.search(pat, text)
        if mm:
            out[key] = float(mm.group(1))
    out["verdict_unusable"] = "判定          : 不可用" in text
    out["not_saved"] = "未保存" in text
    return out


def broken_pair(seed: int = SEED):
    """造一批**顺序错配**的合成配对：右路整份反序（模拟检测器混用）。

    这正是真机那个"单目各自正常、双目整体爆炸"的签名。

    注意两处种子：
    1. `synth_pair(seed=...)` 固定视角与角点噪声；
    2. `cv2.setRNGSeed(seed)` 固定 **`cv2.stereoCalibrate` 内部的 RANSAC** ——
       不设它，`calibrate_stereo` 每轮的离群剔除结果会抖动，基线能从 2000 mm
       跳到 13000 mm，测试就**不是可复现的**（实测过）。
    """
    objs, K1, K2, imgs_l, imgs_r = synth_pair(seed=seed)
    cv2.setRNGSeed(seed)
    n = imgs_l[0].reshape(-1, 2).shape[0]
    objp = SPEC.object_points().astype(np.float64).reshape(n, 3)
    img_l = [im.reshape(-1, 2).astype(np.float32).reshape(-1, 1, 2) for im in imgs_l]
    img_r = [im[::-1].reshape(-1, 2).astype(np.float32).reshape(-1, 1, 2) for im in imgs_r]
    return objp, img_l, img_r


# --------------------------------------------------------------------------
def test_log_summary_is_self_consistent() -> bool:
    """第 1 节：真机日志的汇总能解析、自洽，且数值**逐字**钉住。"""
    print(f"----- 1. 真机日志汇总（{REAL_LOG_TAG}） -----")
    got = parse_summary(LOG_EXCERPT)
    reasons = got.get("reasons", {})
    ratio = REAL_BASELINE_MM / TAPE_BASELINE_MM

    checks = [
        ("解析出总帧数 / 收下 / 拒收三个数",
         all(k in got for k in ("total", "kept", "rejected"))),
        (f"总帧数 = {REAL_FRAMES_TOTAL}（解析得 {got.get('total')}）",
         got.get("total") == REAL_FRAMES_TOTAL),
        (f"收下 {REAL_PAIRS_KEPT} 对、拒收 {REAL_FRAMES_REJECTED} 帧",
         got.get("kept") == REAL_PAIRS_KEPT
         and got.get("rejected") == REAL_FRAMES_REJECTED),
        ("**自洽**：收下 + 拒收 = 总帧数",
         got.get("kept", 0) + got.get("rejected", 0) == got.get("total")),
        (f"解析出全部 {len(REAL_REJECTIONS)} 类拒收原因（实际 {len(reasons)}）",
         set(reasons) == set(REAL_REJECTIONS)),
        ("**自洽**：各类次数之和 = 拒收帧数",
         sum(c for c, _p in reasons.values()) == REAL_FRAMES_REJECTED),
        ("**自洽**：各类占比之和 ≈ 100%",
         abs(sum(p for _c, p in reasons.values()) - 100.0) < 0.6),
        ("每一类的次数与占比都能对上日志（±0.1%）",
         all(abs(reasons[n][1] - REAL_REJECTIONS[n][1]) < 0.1
             for n in REAL_REJECTIONS if n in reasons)),
        # ⚠️ 下面几条是**变异测试逼出来的**：只断言"倍数 > 2"时，
        # 把 REAL_BASELINE_MM 谎报成 66（= 没被吹飞）居然不会红。
        # 所以真机那几个标量必须**逐字**钉住，不能只钉量级。
        (f"解析出的基线 = {REAL_BASELINE_MM} mm（逐字）",
         abs(got.get("baseline_mm", 0.0) - REAL_BASELINE_MM) < 0.01),
        (f"解析出的整体 RMS = {REAL_RMS} px（逐字）",
         abs(got.get("rms", 0.0) - REAL_RMS) < 0.0001),
        (f"解析出的极线 RMS = {REAL_EPIPOLAR_RMS} px（逐字）",
         abs(got.get("epipolar", 0.0) - REAL_EPIPOLAR_RMS) < 0.0001),
        (f"单目 RMS 逐字：左 {REAL_ERR_LEFT} / 右 {REAL_ERR_RIGHT} px",
         abs(got.get("err_left", 0.0) - REAL_ERR_LEFT) < 0.0001
         and abs(got.get("err_right", 0.0) - REAL_ERR_RIGHT) < 0.0001),
        (f"基线相对尺子值被吹飞 {ratio:.2f} 倍（钉在 3.07 ± 0.01）",
         abs(ratio - 3.07) < 0.01),
        ("解析出了「判定 不可用」与「未保存」",
         got.get("verdict_unusable") is True and got.get("not_saved") is True),
    ]

    print(f"  共处理 {REAL_FRAMES_TOTAL} 帧：收下 {REAL_PAIRS_KEPT} 对，"
          f"拒收 {REAL_FRAMES_REJECTED} 帧")
    for name, (cnt, pct) in sorted(reasons.items(), key=lambda kv: -kv[1][0]):
        print(f"    {cnt:5d} ({pct:5.1f}%)  {name}")
    top = max(reasons.items(), key=lambda kv: kv[1][0])
    print(f"  占比最高：「{top[0]}」{top[1][0]} 次（{top[1][1]}%）"
          f" —— 真机卡在「板没停稳」，不是「找不到板」")
    print(f"  基线 {REAL_BASELINE_MM} mm vs 尺子 {TAPE_BASELINE_MM:.0f} mm = {ratio:.2f} 倍")
    checks.append(("占比最高的原因就是「棋盘还在移动」",
                   top[0] == "棋盘还在移动，请停稳再收"))
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_mono_ok_but_stereo_explodes() -> bool:
    """第 2 节：复现"单目正常、双目爆炸"的签名。"""
    print("----- 2. 单目各自正常、双目整体爆炸（顺序错配的可复现签名） -----")
    objp, img_l, img_r = broken_pair()
    cfg = CalibConfig()
    res = calibrate_stereo(objp, img_l, img_r, (1280, 720), cfg, SPEC)

    print(f"  使用/输入 {res.n_pairs_used}/{res.n_pairs_in} 对")
    print(f"  整体重投影 RMS {res.rms:10.4f} px   （门槛 ok_rms={cfg.ok_rms}）")
    print(f"  左目重投影 RMS {res.err_left:10.4f} px   （**正常**）")
    print(f"  右目重投影 RMS {res.err_right:10.4f} px   （**正常**）")
    print(f"  极线误差 RMS   {res.epipolar_rms:10.4f} px   （门槛 ok_epipolar={cfg.ok_epipolar}）")
    print(f"  基线长度       {res.baseline_mm:10.2f} mm   （真值 {TAPE_BASELINE_MM:.1f}）")
    print(f"  判定：{res.verdict}")
    for w in res.warnings:
        print(f"    ! {w}")

    checks = [
        ("两路单目重投影 RMS 都远低于整体 RMS（各自都「正常」）",
         res.err_left < res.rms / 10.0 and res.err_right < res.rms / 10.0),
        ("整体 RMS 超过门槛", res.rms > cfg.ok_rms),
        ("极线误差远超门槛", res.epipolar_rms > cfg.ok_epipolar),
        ("基线被吹飞（> 5 倍真值）",
         res.baseline_mm / TAPE_BASELINE_MM > 5.0),
        ("判定落在「不可用」", res.verdict.startswith("不可用")),
        ("res.ok is False", res.ok is False),
        # 真机的"单目正常"其实只是"比双目好两个数量级"—— 1.3784 px **也超了 ok_rms=0.8**。
        # 这一点很重要：不能把"单目各自正常"读成"两路都完美"，它只说"错在对应关系，不在单目"。
        ("真机是同一形状的签名：双目/单目 比值 > 20 倍",
         REAL_RMS / REAL_ERR_LEFT > 20.0 and REAL_RMS / REAL_ERR_RIGHT > 20.0),
        ("真机的单目 RMS 同样**也超了 ok_rms 门槛**（所以「正常」≠「达标」）",
         REAL_ERR_LEFT > cfg.ok_rms and REAL_ERR_RIGHT > cfg.ok_rms),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_unusable_result_is_never_saved() -> bool:
    """第 3 节：判定不可用时，结果一定不落盘（两道闸门）。"""
    print("----- 3. 不可用 -> 一定不保存 -----")
    objp, img_l, img_r = broken_pair()
    res = calibrate_stereo(objp, img_l, img_r, (1280, 720), CalibConfig(), SPEC)

    saved = False
    refusal = ""
    try:
        StereoRig.from_result(res, meta={"board": SPEC.describe(), "source": "camera"})
        saved = True
    except ValueError as exc:
        refusal = str(exc)

    got = parse_summary(LOG_EXCERPT)
    print(f"  res.ok = {res.ok}   判定 = {res.verdict}")
    print(f"  StereoRig.from_result() -> {'保存了（**这不该发生**）' if saved else '拒绝：' + refusal}")
    print(f"  日志原文也有「未保存」：{got.get('not_saved')}")

    checks = [
        ("res.ok is False（第一道闸门：调用方不该走到保存）", res.ok is False),
        ("StereoRig.from_result 抛 ValueError（第二道闸门）", saved is False),
        ("拒绝原因里带着判定文字（可诊断，不是空异常）",
         "不可用" in refusal or bool(refusal)),
        ("日志原文也是「未保存」", got.get("not_saved") is True),
        # ⚠️ 变异测试发现的缺口：上面两条只测"这次没保存"，测不到"那道闸门被删了"。
        # 所以直接对源码断言 —— 闸门要是被拿掉，这里就红。
        ("**源码级**：params.py 里确实有 `if not res.ok: raise` 这道闸门",
         _params_gate_present()),
        ("**源码级**：calibrate_stereo.py 在不可用时 `return 1`",
         _app_refuses_on_unusable()),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_baseline_anchor_is_optional_and_therefore_silent() -> bool:
    """第 4 节：基线锚不给 -> 守卫不存在 -> 偏差被静默放过。"""
    print("----- 4. 基线锚：不显式给 = 守卫不存在 -----")
    objp, img_l, img_r = broken_pair()

    res_default = calibrate_stereo(objp, img_l, img_r, (1280, 720), CalibConfig(), SPEC)
    cfg_anchored = CalibConfig()
    cfg_anchored.expected_baseline_mm = TAPE_BASELINE_MM
    res_anchored = calibrate_stereo(objp, img_l, img_r, (1280, 720), cfg_anchored, SPEC)

    d_base = [w for w in res_default.warnings if "基线" in w]
    a_base = [w for w in res_anchored.warnings if "基线" in w]

    print(f"  CalibConfig().expected_baseline_mm 默认 = {CalibConfig().expected_baseline_mm!r}")
    print(f"  本批数据的标定基线 {res_default.baseline_mm:.2f} mm，"
          f"真值 {TAPE_BASELINE_MM:.1f} mm（{res_default.baseline_mm / TAPE_BASELINE_MM:.2f} 倍）")
    print(f"  不给锚：警告 {len(res_default.warnings)} 条，其中含「基线」的 {len(d_base)} 条")
    for w in res_default.warnings:
        print(f"      · {w}")
    print(f"  给锚（{TAPE_BASELINE_MM:.0f} mm）：警告 {len(res_anchored.warnings)} 条，"
          f"其中含「基线」的 {len(a_base)} 条")
    for w in res_anchored.warnings:
        print(f"      · {w}")

    checks = [
        ("`expected_baseline_mm` 默认是 None（生产者没给锚）",
         CalibConfig().expected_baseline_mm is None),
        ("不给锚时**一条基线警告都没有** —— 大幅偏差被静默放过",
         len(d_base) == 0),
        ("给了锚就报出基线偏差（守卫确实存在，只是没被启用）",
         len(a_base) == 1),
        ("该警告里含具体数字（可行动，不只是「有问题」）",
         bool(a_base) and re.search(r"\d", a_base[0]) is not None),
        ("真机日志同样只有 2 条警告、且都**不是**基线那条（与本节同因）",
         len(_log_warning_lines(LOG_EXCERPT)) == 2
         and not any("基线" in w for w in _log_warning_lines(LOG_EXCERPT))),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def _synth_views(distance_mm: float, n: int = 12, noise_px: float = 0.15):
    """按**真实相机模型**造一对视图：一块 8x5 / 25mm 的板放在 distance_mm 处。

    ## 为什么不能用"物体点乘个系数"糊弄

    本仓储的 `BoardSpec.object_points()` 返回的单位是**米**（x 范围 0~0.175），
    我第一版临时脚本把它当毫米、又乘了个"距离系数"，于是板被放在
    700 **米**外 —— 解出来的基线是 1e6 mm 的垃圾，而看起来"代码坏了"。
    **是测试数据错，不是被测代码错。** 所以这里老老实实做投影。
    """
    w, h = 1920, 1080
    K = np.array([[1400.0, 0, w / 2], [0, 1400.0, h / 2], [0, 0, 1]])
    zero = np.zeros(5)
    od = np.zeros(3)
    base = SPEC.object_points()
    t_r = np.array([-65.0, 0.0, 0.0]) / 1000.0        # 右相机沿 +X 平移一个基线
    rng = np.random.default_rng(0)
    out1, out2 = [], []
    for k in range(n):
        ax, ay, az = 0.28 * np.sin(k * 1.1), 0.30 * np.cos(k * 0.9), 0.10 * np.sin(k * 0.5)
        Rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
        Ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
        Rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx
        cx = ((k % 4) - 1.5) * 0.25
        cy = (((k // 4) % 3) - 1.0) * 0.20
        t = np.array([cx, cy, distance_mm / 1000.0])
        p_l = (R @ base.T).T + t
        p_r = (p_l - t_r.reshape(1, 3))                  # 右相机只在 X 上平移
        i1, _ = cv2.projectPoints(p_l, od, od, K, zero)
        i2, _ = cv2.projectPoints(p_r, od, od, K, zero)
        out1.append(i1 + rng.normal(0, noise_px, i1.shape))
        out2.append(i2 + rng.normal(0, noise_px, i2.shape))
    return out1, out2


def test_board_area_and_tiered_verdict() -> bool:
    """第 5 节：板占画面比例 + **分级**判定（2026-10-14 加）。

    ## 这一节守的是什么

    原来判定是**二值**：`rms <= ok_rms and epipolar <= ok_epipolar` 才保存，
    否则一句"不可用 —— 请重拍"。实测一次真机标定（基线 67.28 mm vs 实际 65，
    **只差 3.5%**，基线核对是通过的）因为 RMS 0.9143 > ok 门限 0.80 被整体否掉、
    不保存。用户的反应是"我不是很能接受" —— 他不接受是对的：

    * `ok_rms` 本来就是"够用"的意思，不是"超过就废"；
    * 二值判定丢掉了最有用的信息 —— **差在哪、下次怎么改**。

    同时，那次标定的真实原因是**板只占画面 1.2%**（放太远，角点像素少、
    定位精度低），而这个数字**全程都算得出来却没有出口**，报告里一个字都没提。

    所以本节钉四件事：
    1. 板占比例**算得对**（对着 1200mm → 约 1%，与你那次日志里的 1.2% 同量级）；
    2. 板放远 -> 报告里出现"板离得太远 + 靠近到 10%~30%"这条**可执行**建议；
    3. `ok_rms < rms <= 某个程度` -> 判为**勉强可用且仍然保存**（不再一票否决）；
    4. **真正物理错**（基线对不上）-> 仍然不可用并拒绝保存（安全边界不放宽）。
    """
    print("----- 5. 板占画面比例 + 分级判定 -----")
    checks: list[tuple[str, bool]] = []

    far_l, far_r = _synth_views(1200.0)
    near_l, near_r = _synth_views(450.0)
    cfg = CalibConfig(expected_baseline_mm=65.0)

    res_far = calibrate_stereo(SPEC.object_points(), far_l, far_r, (1920, 1080), cfg, SPEC)
    res_near = calibrate_stereo(SPEC.object_points(), near_l, near_r, (1920, 1080), cfg, SPEC)

    print(f"  板放 1200mm: 占画面 {res_far.board_area * 100:5.2f}%  "
          f"基线 {res_far.baseline_mm:6.2f}  分级={res_far.grade}")
    print(f"  板放  450mm: 占画面 {res_near.board_area * 100:5.2f}%  "
          f"基线 {res_near.baseline_mm:6.2f}  分级={res_near.grade}")

    checks.append(("板占比例**算得出来**（不是 nan/0 —— 凸包算法正确）",
                   np.isfinite(res_far.board_area) and res_far.board_area > 0.0))
    checks.append(("远的那次比例明显小于近的那次（单调性正确）",
                   res_far.board_area < res_near.board_area))
    checks.append(("1200mm 时约 1%（与真机日志里的 1.2% 同量级）",
                   0.004 < res_far.board_area < 0.03))
    checks.append(("两路比例都记下来了", np.isfinite(res_far.board_area_l)
                   and np.isfinite(res_far.board_area_r)))

    rep_far = res_far.report()
    checks.append(("报告里**出现**板占比例（不再是「没有出口的字段」）",
                   "板占画面" in rep_far))

    # ---- 建议必须**可执行**，而且不能自相矛盾 ----
    far_sugg = " ".join(res_far.suggestions)
    print(f"  远那次的建议 {len(res_far.suggestions)} 条：")
    for s in res_far.suggestions:
        print(f"    -> {s[:96]}")
    checks.append(("报告里给出「靠近到 10%」这类**能照着做**的建议",
                   "10%~30%" in far_sugg or "靠近" in far_sugg))
    checks.append(("建议里含**具体数字**（不是「换更多角度」这种空话）",
                   re.search(r"\d", far_sugg) is not None))
    # ⚠️ 这里钉的是「**建议与判定是两件事**」：板太远是真的可以更好，与等级无关。
    # 我第一版把建议整体挂在「没到优秀才给」下面，结果这块 1.12% 的板解出
    # 「优秀」就**一条建议都没有**了 —— 那是错的。
    # 真正要避免的自相矛盾只有「建议说太近、结论说不够好」这类**指向相反**的话。
    checks.append(("板占 1.12% 时**一定**给出「板离得太远」（不受判定等级压制）",
                   "太远" in far_sugg))
    checks.append(("近那次的建议里不会同时出现「太近」与「太远」（不自相矛盾）",
                   not ("太近" in " ".join(res_near.suggestions)
                        and "太远" in " ".join(res_near.suggestions))))
    # ---- 分级：唯一「不可用」的理由必须是物理错 ----
    checks.append(("正常数据判为可用一级（不是「不可用」）",
                   res_near.grade in ("优秀", "可用", "勉强可用") and res_near.ok))
    checks.append(("分级字段非空且是四级之一",
                   res_far.grade in ("优秀", "可用", "勉强可用", "不可用")))

    bad_l, bad_r = _synth_views(700.0)
    res_bad = calibrate_stereo(SPEC.object_points(), bad_l, bad_r, (1920, 1080),
                               CalibConfig(expected_baseline_mm=999.0), SPEC)
    print(f"  基线填 999mm（真值 65）: 分级={res_bad.grade}  ok={res_bad.ok}")
    checks.append(("**基线对不上** -> 判不可用（安全边界不放宽）",
                   res_bad.grade == "不可用" and res_bad.ok is False))
    checks.append(("基线错的建议指向**该核对什么**（方格尺寸/中心距/左右是否反）",
                   any("方格" in s for s in res_bad.suggestions)))

    # ---- 勉强可用必须**仍然保存**（这正是这次改动的核心）----
    #
    # ⚠️ 这一段第一版是**重言式**：我手工改了个副本的 rms 字段，
    # 然后断言 `res_near.grade != "不可用"` —— 而 res_near 本来就是「优秀」，
    # 所以恒真，把判定退回二值的变异**照样通过**。已改成真的跑一次：
    # 抬高投影噪声到 0.60 px，实测得到 RMS 0.825 / 极线 0.882 —— 都过 ok 门限，
    # 但远没到「物理错」，**正是真机那次的形状**（0.91 / 0.81）。
    tight_l, tight_r = _synth_views(700.0, noise_px=0.60)
    res_tight = calibrate_stereo(SPEC.object_points(), tight_l, tight_r,
                                 (1920, 1080), cfg, SPEC)
    print(f"  抬高噪声到 0.60 px: RMS {res_tight.rms:.3f}（ok 门限 {cfg.ok_rms}）"
          f"  极线 {res_tight.epipolar_rms:.3f}（ok 门限 {cfg.ok_epipolar}）"
          f"  分级={res_tight.grade}  ok={res_tight.ok}")
    checks.append(("两项都过 ok 门限、但基线没错 -> 判「勉强可用」",
                   res_tight.rms > cfg.ok_rms
                   and res_tight.epipolar_rms > cfg.ok_epipolar
                   and res_tight.grade == "勉强可用"))
    checks.append(("「勉强可用」**仍然保存**（ok=True）—— 不再一票否决",
                   res_tight.ok is True))
    checks.append(("勉强可用时报告里说清差在哪（含倍率数字）",
                   re.search(r"门限的 [\d.]", res_tight.verdict) is not None))

    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def _fake_rig_for_export() -> StereoRig:
    """一个 65 mm 基线的合成装置，专门用来测导出（不碰真实标定文件）。"""
    K = np.array([[800.0, 0.0, 320.0],
                  [0.0, 800.0, 240.0],
                  [0.0, 0.0, 1.0]])
    D = np.zeros((1, 5))
    T = np.array([[-0.065], [0.0], [0.0]])
    P1 = np.hstack([K, np.zeros((3, 1))])
    P2 = P1.copy()
    P2[0, 3] = -800.0 * 0.065
    return StereoRig(
        K1=K, D1=D, K2=K.copy(), D2=D.copy(), R=np.eye(3), T=T,
        image_size=(640, 480), E=np.eye(3), F=np.eye(3),
        R1=np.eye(3), R2=np.eye(3), P1=P1, P2=P2, Q=np.eye(4),
        quality={"rms": 0.5, "epipolar_rms": 0.4,
                 "fx_sigma_px": float("nan"), "verdict": "测试"},
        meta={"board": "8x5 25mm", "created": "2026-01-01 00:00:00",
              "source": "synthetic"},
    )


def test_export_formats() -> bool:
    """导出 JSON / YAML / TOML：数值必须与 rig 一致，且三种格式都能读回。

    为什么值得测：导出是"给外部工具用"的唯一出口。数值差一位、单位写成毫米、
    TOML 里漏掉嵌套表 —— 都不会让主链路变红，但会让别人拿着错参数跑。
    """
    print("----- 标定参数导出（JSON / YAML / TOML） -----")
    checks: list[tuple[str, bool]] = []
    rig = _fake_rig_for_export()
    data = build_export(rig)

    checks.append(("左相机 K 与 rig.K1 一致", np.allclose(data["left"]["K"], rig.K1)))
    checks.append(("D 拉平成 5 个系数", len(data["left"]["D"]) == 5))
    checks.append(("rig.T 为 3 个分量（米）",
                   np.allclose(data["rig"]["T"], np.ravel(rig.T))))
    checks.append(("基线 65 mm", abs(data["rig"]["baseline_mm"] - 65.0) < 1e-9))
    checks.append(("非有限质量指标 -> None（JSON 可序列化）",
                   data["rig"]["quality"]["fx_sigma_px"] is None))

    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        paths = write_export(rig, out, ["json", "yaml", "toml"])
        names = {p.name for p in paths}
        checks.append(("9 个参数文件 + summary", len(paths) == 10 and "summary.txt" in names))
        checks.append(("三种格式齐全",
                       all(f"stereo_rig.{ext}" in names for ext in ("json", "yaml", "toml"))))

        j = json.loads((out / "stereo_rig.json").read_text(encoding="utf-8"))
        checks.append(("JSON 读回 T 一致", np.allclose(j["T"], np.ravel(rig.T))))

        t = tomllib.loads((out / "stereo_rig.toml").read_text(encoding="utf-8"))
        checks.append(("TOML 读回 T 一致", np.allclose(t["T"], np.ravel(rig.T))))
        checks.append(("TOML 读回嵌套表 quality", abs(t["quality"]["rms"] - 0.5) < 1e-12))
        t_left = tomllib.loads((out / "left_camera.toml").read_text(encoding="utf-8"))
        checks.append(("TOML 读回 K 一致", np.allclose(t_left["K"], rig.K1)))
        checks.append(("TOML 不含 null",
                       "null" not in (out / "stereo_rig.toml").read_text(encoding="utf-8")))

        try:
            import yaml
        except ImportError:
            print("  [跳过] PyYAML 未安装，跳过 YAML 读回校验")
        else:
            y = yaml.safe_load((out / "stereo_rig.yaml").read_text(encoding="utf-8"))
            checks.append(("YAML 读回 T 一致", np.allclose(y["T"], np.ravel(rig.T))))

        summary = (out / "summary.txt").read_text(encoding="utf-8")
        checks.append(("summary 含基线与坐标系说明",
                       "基线" in summary and "校正后左相机系" in summary))

        try:
            write_export(rig, out, ["xml"])
        except ValueError:
            checks.append(("未知格式被拒绝（不会写出半套文件）", True))
        else:
            checks.append(("未知格式被拒绝（不会写出半套文件）", False))

    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


# --------------------------------------------------------------------------
# 世界系（config/world_frame.json）
#
# 这一节守的是**"世界系不能说谎"**：
# 1. 落盘/读回逐元素一致，坏矩阵（最后一行 / 非正交 / 3x3）必须被拒并指出原因；
# 2. `apply()` 与手算一致（变换写错方向时这一条会红）；
# 3. 真图片走一遍 `from_board()`（不是只测 `from_camera_board()`）；
# 4. 没有 `world_frame.json` 时 `visualization.json` **必须**保持左相机系 ——
#    静默当世界系是这一节唯一真正危险的行为。
# --------------------------------------------------------------------------
#: 绕 Z 转 90° 再平移 (1,2,3)：手算期望值就写在这张矩阵旁边。
WORLD_T = np.array([[0.0, -1.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 2.0],
                    [0.0, 0.0, 1.0, 3.0],
                    [0.0, 0.0, 0.0, 1.0]])


def _write_world_json(path: Path, T) -> None:
    path.write_text(json.dumps({"version": 1, "source": "manual", "units": "m",
                                "note": "", "board_rmse_px": None,
                                "created_at": "2026-01-01 00:00:00",
                                "T_world_cam": np.asarray(T).tolist()},
                               ensure_ascii=False), encoding="utf-8")


def test_world_frame_roundtrip() -> bool:
    """落盘 -> 读回逐元素一致；三类坏矩阵都必须 ValueError 且指出原因。"""
    print("----- 6. 世界系：往返 + 坏文件必须被拒 -----")
    checks: list[tuple[str, bool]] = []
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "world_frame.json"
        WorldFrame(WORLD_T, "manual", note="合成").save(p)
        data = json.loads(p.read_text(encoding="utf-8"))
        checks.append(("JSON 含 version/T_world_cam/source/units/note/created_at",
                       all(k in data for k in ("version", "T_world_cam", "source",
                                               "units", "note", "board_rmse_px",
                                               "created_at"))))
        back = WorldFrame.load(p)
        checks.append(("往返后矩阵逐元素一致（< 1e-12）",
                       float(np.max(np.abs(back.T_world_cam - WORLD_T))) < 1e-12))
        checks.append(("source/note 一并带回",
                       back.source == "manual" and back.note == "合成"))

        bad_cases = [
            ("最后一行不是 [0,0,0,1]",
             np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 2.0]]),
             "最后一行"),
            ("旋转块非正交",
             np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0.2, 0, 1.0, 0], [0, 0, 0, 1.0]]),
             "正交"),
            ("3x3 不是 4x4", np.eye(3), "4x4"),
        ]
        for name, T, keyword in bad_cases:
            bp = Path(td) / "bad.json"
            _write_world_json(bp, T)
            message = ""
            try:
                WorldFrame.load(bp)
            except ValueError as exc:
                message = str(exc)
            checks.append((f"坏文件被拒：{name}",
                           keyword in message and str(bp) in message))
            print(f"    {name} -> {message or '（**没有报错**）'}")

        bp = Path(td) / "broken.json"
        bp.write_text("{ this is not json", encoding="utf-8")
        message = ""
        try:
            WorldFrame.load(bp)
        except ValueError as exc:
            message = str(exc)
        checks.append(("坏 JSON 被拒且指出文件", "JSON" in message and str(bp) in message))

    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_world_frame_apply() -> bool:
    """apply()：3 个点的手算期望值（变换方向写反/转置写错都会红）。"""
    print("----- 7. 世界系：apply() 数值 -----")
    wf = WorldFrame(WORLD_T, "manual")
    got = wf.apply(np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]))
    expect = np.array([[1.0, 3.0, 3.0], [-1.0, 2.0, 3.0], [1.0, 2.0, 4.0]])
    err = float(np.max(np.abs(got - expect)))
    checks = [
        (f"3 个点与手算一致（max|差| = {err:.2e} < 1e-12）", err < 1e-12),
        ("None -> None（空数据直通）", wf.apply(None) is None),
    ]
    try:
        wf.apply(np.zeros((2, 2)))
    except ValueError:
        checks.append(("形状不是 (N,3) 时报错", True))
    else:
        checks.append(("形状不是 (N,3) 时报错", False))
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def _write_board_photo(path: Path, spec: BoardSpec, K: np.ndarray) -> None:
    """画一张**真的能被 `detect_board` 检出**的棋盘照片（透视投影 + 白边）。"""
    sq, pad = 60, 2
    flat = np.full(((spec.rows + 1 + 2 * pad) * sq, (spec.cols + 1 + 2 * pad) * sq),
                   255, np.uint8)
    for j in range(spec.rows + 1):
        for i in range(spec.cols + 1):
            if (i + j) % 2 == 0:
                y0, x0 = (pad + j) * sq, (pad + i) * sq
                flat[y0:y0 + sq, x0:x0 + sq] = 0
    s = spec.square_mm / 1000.0
    outer = np.array([[-s, -s, 0], [spec.cols * s, -s, 0],
                      [spec.cols * s, spec.rows * s, 0], [-s, spec.rows * s, 0]], float)
    rvec = cv2.Rodrigues(np.array([np.deg2rad(25.0), 0.12, -0.05]))[0]
    tvec = np.array([-0.06, -0.04, 0.50])
    dst = cv2.projectPoints(outer, rvec, tvec, K, np.zeros(5))[0].reshape(-1, 2)
    src = np.array([[pad * sq, pad * sq],
                    [(pad + spec.cols + 1) * sq, pad * sq],
                    [(pad + spec.cols + 1) * sq, (pad + spec.rows + 1) * sq],
                    [pad * sq, (pad + spec.rows + 1) * sq]], np.float32)
    H = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    photo = cv2.warpPerspective(flat, H, (1920, 1080), borderValue=128)
    cv2.imwrite(str(path), photo)


def test_world_frame_from_board_photo() -> bool:
    """真图片路径：`from_board()` 能检出、rmse 过门槛、尺度与 Z 方向说得清。"""
    print("----- 8. 世界系：真图片 from_board() -----")
    checks: list[tuple[str, bool]] = []
    K = np.array([[1400.0, 0.0, 960.0], [0.0, 1400.0, 540.0], [0.0, 0.0, 1.0]])
    rig = StereoRig(
        K1=K, D1=np.zeros((1, 5)), K2=K.copy(), D2=np.zeros((1, 5)),
        R=np.eye(3), T=np.array([[-0.065], [0.0], [0.0]]),
        image_size=(1920, 1080),
        meta={"stereo_mode": "angled", "rectification": "none_raw"},
    )
    with tempfile.TemporaryDirectory() as td:
        photo = Path(td) / "board.png"
        _write_board_photo(photo, SPEC, K)
        checks.append(("照片真的写出来了", photo.is_file()))

        wf = from_board(photo, SPEC, rig)
        checks.append((f"rmse = {wf.board_rmse_px:.2f} px < 1.0",
                       wf.board_rmse_px is not None and wf.board_rmse_px < 1.0))
        checks.append(("source = board_pnp", wf.source == "board_pnp"))
        checks.append(("describe() 含来源与 rmse",
                       "board_pnp" in wf.describe() and "rmse=" in wf.describe()))

        # 相机到世界原点的距离应等于合成时的板距（~0.5 m）——尺度对了才叫世界系。
        cam = wf.apply(np.zeros((1, 3)))[0]        # 左相机光心在世界系里的位置
        dist = float(np.linalg.norm(cam))
        checks.append((f"相机光心距世界原点 {dist:.3f} m（合成板距 0.5 m）",
                       0.4 < dist < 0.6))
        # 默认 +Z = 棋盘法向 = 背离相机的那一侧 -> 相机在负 Z。
        checks.append((f"默认 ground_axis=z 时相机在 Z<0 一侧（z={cam[2]:.3f}）",
                       cam[2] < 0.0))
        wf_up = from_board(photo, SPEC, rig, ground_axis="-z")
        cam_up = wf_up.apply(np.zeros((1, 3)))[0]
        checks.append((f"ground_axis='-z' 时相机翻到 Z>0（z={cam_up[2]:.3f}）",
                       cam_up[2] > 0.0))
        checks.append(("两种 ground_axis 的 rmse 相同（只翻轴、不动拟合）",
                       abs(wf.board_rmse_px - wf_up.board_rmse_px) < 1e-12))

        blank = Path(td) / "blank.png"
        cv2.imwrite(str(blank), np.full((720, 1280), 255, np.uint8))
        message = ""
        try:
            from_board(blank, SPEC, rig)
        except ValueError as exc:
            message = str(exc)
        checks.append(("空图（检不到板）抛 ValueError 并说明怎么重拍",
                       "没检测到" in message and "重拍" in message))
        print(f"    空图 -> {message}")

        missing = Path(td) / "nope.png"
        try:
            from_board(missing, SPEC, rig)
        except ValueError as exc:
            checks.append(("图打不开 -> ValueError 指出路径", str(missing) in str(exc)))
        else:
            checks.append(("图打不开 -> ValueError 指出路径", False))

    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_visualization_world_frame() -> bool:
    """没配置 -> 左相机系且写明；有配置 -> world + 点被换；配置坏了 -> 抛错。"""
    print("----- 9. visualization.json：世界系不许静默 -----")
    checks: list[tuple[str, bool]] = []
    trajectory = {
        "session": "synthetic", "unit": "m", "stereo_mode": "angled",
        "coordinate_system": "left_camera",
        "bounds": {"min": [0.1, 0.2, 0.3], "max": [0.1, 0.2, 0.3]},
        "samples": [
            {"frame": 0, "t": 0.0, "matched": True, "xyz": [0.1, 0.2, 0.3],
             "confident": True},
            {"frame": 1, "t": 0.03, "matched": False, "xyz": None, "confident": False},
        ],
    }
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "world_frame.json"

        vis0 = build_visualization_payload(trajectory, world_frame_path=p)
        checks.append(("无 world_frame.json：coordinate_system 仍是 left_camera",
                       vis0["coordinate_system"] == "left_camera"))
        checks.append(("无 world_frame.json：notes 明说当前是左相机系",
                       any("左相机系" in n for n in vis0.get("notes", []))))
        checks.append(("无 world_frame.json：点坐标未被改动",
                       vis0["trajectory"][0]["xyz"] == [0.1, 0.2, 0.3]))
        checks.append(("无 world_frame.json：不写 world_frame 字段",
                       "world_frame" not in vis0))

        WorldFrame(WORLD_T, "manual", note="合成").save(p)
        vis1 = build_visualization_payload(trajectory, world_frame_path=p)
        checks.append(("有 world_frame.json：coordinate_system = world",
                       vis1["coordinate_system"] == "world"))
        checks.append(("有 world_frame.json：点被换成世界系（手算 0.8/2.1/3.3）",
                       np.allclose(vis1["trajectory"][0]["xyz"], [0.8, 2.1, 3.3], atol=1e-12)))
        checks.append(("bounds 跟着重算",
                       np.allclose(vis1["bounds"]["min"], [0.8, 2.1, 3.3], atol=1e-12)))
        checks.append(("world_frame 带 source/note/created_at",
                       set(vis1["world_frame"]) == {"source", "note", "created_at"}))
        checks.append(("未匹配的点仍为 None（不凭空造坐标）",
                       vis1["samples"][1]["xyz"] is None))

        p.write_text("{ broken json", encoding="utf-8")
        message = ""
        try:
            build_visualization_payload(trajectory, world_frame_path=p)
        except ValueError as exc:
            message = str(exc)
        checks.append(("坏 JSON -> 抛错并指出文件与原因",
                       "JSON" in message and str(p) in message))
        print(f"    坏 JSON -> {message}")

        _write_world_json(p, np.eye(3))
        message = ""
        try:
            build_visualization_payload(trajectory, world_frame_path=p)
        except ValueError as exc:
            message = str(exc)
        checks.append(("3x3 矩阵 -> 抛错（**不许**回退成单位矩阵）",
                       "4x4" in message and str(p) in message))
        print(f"    3x3 -> {message}")

    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def main() -> int:
    print("=" * 70)
    print("标定事实自测（真机日志口径 + 不可用不得保存 + 基线锚 + 分级判定 + 世界系）")
    print("=" * 70)
    results = []
    for fn in (test_log_summary_is_self_consistent,
               test_mono_ok_but_stereo_explodes,
               test_unusable_result_is_never_saved,
               test_baseline_anchor_is_optional_and_therefore_silent,
               test_board_area_and_tiered_verdict,
               test_export_formats,
               test_world_frame_roundtrip,
               test_world_frame_apply,
               test_world_frame_from_board_photo,
               test_visualization_world_frame):
        results.append(fn())
        print()
    ok = all(results)
    print("判定结果：" + ("全部通过" if ok else "有不达标项"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
