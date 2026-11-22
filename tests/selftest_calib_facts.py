"""标定事实自测：真机日志口径 + "质量差也出结果但不覆盖正式档案" + 基线锚必须显式给。

为什么需要它
------------
真机第一次双目标定跑完留下的两份日志（`logs/calib/2026-10/calibrate_2026*.log`）
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

**3. 质量差也出结果，但绝不覆盖正式档案（双轨 + 防呆）**

新原则「凡是质量差，也要能出结果而不是自己中止程序，只有错误才是紧急的」：
默认（无 `--force-save`）把不可用结果另存到 `*_unusable_<时间戳>.npz/.json`
旁路文件供检查，**正式 `--out` 与激活装置档案逐字节不变**；只有显式
`--force-save` 才写正式路径。两条轨道都带 `forced_unusable` / `grade=不可用` /
verdict「仅链路测试，不得用于数据结论」标记。

`calib/params.py` 的 `StereoRig.from_result()` 对 `allow_unusable=False` 仍抛
`ValueError`（第二道闸门保留）；所有 `not res.ok` 的保存都必须显式传
`allow_unusable=True`。下游 `stereo/record_plan.py` 按 `forced_unusable` 仍拒绝
录制 —— 形成「能存、能看、但不能用」。本节钉 8 条判据（含两条源码级护栏）。

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
=======================================  ==========================================  ========
改动                                    注入方式                                    结果
=======================================  ==========================================  ========
谎报真机基线（199.67 -> 66）           `REAL_BASELINE_MM = 66.0`                   exit=1
谎报拒收统计（1362 -> 1000）           `REAL_FRAMES_TOTAL = 1000`                  exit=1
谎报单目 RMS（1.3784 -> 0.5）          `REAL_ERR_LEFT = 0.5`                        exit=1
删掉 `params.py` 的拒绝闸门             `if not res.ok: raise ...` -> `pass`         exit=1（判据 2 + 源码级）
旁路改回「直接 `rig.save(args.out)`」    `finalize_and_save` 去掉旁路改 out            exit=1（判据 3）
去掉 `forced_unusable`/verdict 标记      `finalize_and_save` 去掉标记三行              exit=1（判据 4/5）
=======================================  ==========================================  ========

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
from calib.uncertainty import extrinsic_sigma
from calib.validation import cross_validate
from calib.world_frame import WorldFrame, from_board
from stereo.visualization import build_visualization_payload

# 复用角点顺序测试里的合成装置（保证两处口径一致）
from selftest_board_corner_order import (  # noqa: E402
    SEED, SPEC, TAPE_BASELINE_MM, synth_pair)

import cv2  # noqa: E402

# --------------------------------------------------------------------------
# 真机那次的实测值（logs/calib/2026-10/calibrate_20261010_162029.log）
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

# 日志原文片段（排版与 logs/calib/2026-10/ 里的真实文件逐字一致，含对齐空格）
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


def _app_bypasses_on_unusable() -> bool:
    """`apps/calibrate_stereo.py` 是否把不可用结果**旁路保存**（源码级断言）。

    新原则：质量差不再 `return 1`，而是默认把 `args.out` 改到旁路、显式
    `--force-save` 才写正式路径。护栏换成「统一保存函数 + 默认旁路改 out +
    不再有 `if not res.ok: return 1` 中止」三件事同时成立：删掉旁路、改回
    `rig.save(args.out)`、或重新加回中止，这里都会红。
    """
    src = (ROOT / "apps" / "calibrate_stereo.py").read_text(encoding="utf-8")
    has_finalize = "def finalize_and_save(" in src
    has_bypass = bool(re.search(r"args\.out\s*=\s*_bypass_calibration_out\(", src))
    no_early_return = not re.search(
        r"if\s+not\s+res\.ok\s*:[^\n]*\n(?:[^\n]*\n){0,6}?\s*return\s+1", src)
    return has_finalize and has_bypass and no_early_return


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
    # ---- 泄漏守卫（2026-10-26 加）：这个工具**最容易被误用**成"永远通过" ----
    # 若 fit 忽略传入的 train_idx、每折都拿全量训练，比值恒 1.0 且外部看不出来。
    def _leaky(train_idx, test_idx):
        return {"train_rms": 0.2, "test_rms": 0.2}          # 故意不报 used_train_idx

    rep = cross_validate(_leaky, CV_N, k=4, seed=0)
    print(f"    忽略折号 -> leak_check={rep['leak_check']}  verdict「{rep['verdict'][:28]}…」")
    checks.append(("fit 不报 used_train_idx -> 降级为「未验证」而不是发合格证",
                   rep["leak_check"] == "unverified" and "未验证" in rep["verdict"]))

    def _lying(train_idx, test_idx):
        return {"train_rms": 0.2, "test_rms": 0.2, "used_train_idx": list(range(CV_N))}

    rep = cross_validate(_lying, CV_N, k=4, seed=0)
    checks.append(("谎报 used_train_idx -> 该折判失败（训练/测试重叠）",
                   len(rep["failed_folds"]) == 4
                   and "不一致" in (rep["failed_folds"][0]["error"] or "")))

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


def _synthetic_rig() -> StereoRig:
    """造一套**能存盘**的假标定 —— 只用于测 `save()` 的备份行为。

    为什么不读本机真标定（2026-10-27）：这一节要测的是"覆盖前有没有留旧文件"，
    与这台机器标定到多少毫米**毫无关系**。原来 `StereoRig.load("config/stereo_calibration")`
    直接读真文件，于是在干净 checkout（`config/` 被 gitignore）里**整个测试崩在
    FileNotFoundError**，连带 `run_tests fast` 报红 —— 实测这是"干净副本 4 红"之一。
    """
    import numpy as _np
    K = _np.array([[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]])
    D = _np.zeros(5)
    return StereoRig(
        K1=K.copy(), D1=D.copy(), K2=K.copy(), D2=D.copy(),
        R=_np.eye(3), T=_np.array([[-0.07], [0.0], [0.0]]),   # 基线 70 mm
        image_size=(1280, 720),
    )


def test_save_backs_up_previous() -> bool:
    """第 3b 节：覆盖保存前必须留一份旧标定（2026-10-26 真实事故后加的）。

    事故：一次"只想看指标"的实跑默认写盘，把调好的 124.0 mm 标定覆盖成 69.2 mm。
    标定不可再生（要重拍 30+ 张图），所以 `StereoRig.save()` 覆盖前必须先备份。
    """
    import pathlib as _pl
    import tempfile
    print("----- 3b. 覆盖保存前自动备份 -----")
    d = _pl.Path(tempfile.mkdtemp())
    rig = _synthetic_rig()          # 不再读本机真标定：见 _synthetic_rig 的说明
    target = d / "stereo_calibration"

    rig.save(target)
    first = (d / "backup").exists()
    rig.save(target)
    backups = sorted((d / "backup").glob("*.json"))

    print(f"  首次保存 -> 备份目录被创建：{first}（不该：没有旧文件可备）")
    print(f"  二次保存 -> 备份数 {len(backups)}：{[b.name for b in backups]}")
    checks = [
        ("首次保存不产生备份（无旧文件）", first is False),
        ("二次保存留下 1 份旧标定", len(backups) == 1),
        ("备份与主文件同 stem", bool(backups) and backups[0].stem.startswith(target.stem)),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_unusable_result_is_never_saved() -> bool:
    """第 3 节：质量差也出结果，但绝不覆盖正式档案（双轨 + 第二道闸门 + 防呆）。

    新原则「质量差也要出结果、只有错误才紧急」：默认（无 --force-save）把不可用
    结果另存到旁路文件（.npz + .json）供检查，正式 args.out **逐字节不变**；只有
    显式 --force-save 才写正式路径。两条轨道都带 forced_unusable / grade /
    verdict 标记，下游 record_plan 据此仍拒绝录制 —— 形成「能存能看不能用」。
    """
    import logging
    from types import SimpleNamespace

    import apps.calibrate_stereo as cs
    from stereo.record_plan import plan_record

    print("----- 3. 质量差：默认旁路 / 显式提权 / 第二道闸门 / 录制防呆 -----")
    objp, img_l, img_r = broken_pair()
    res = calibrate_stereo(objp, img_l, img_r, (1280, 720), CalibConfig(), SPEC)

    # ---- 判据 2：第二道闸门（allow_unusable=False）仍抛 ValueError ----
    built = False
    refusal = ""
    try:
        StereoRig.from_result(res, meta={"board": SPEC.describe(), "source": "camera"})
        built = True                      # 没抛 = 闸门被拆了
    except ValueError as exc:
        refusal = str(exc)
    print(f"  res.ok = {res.ok}   grade = {res.grade}   判定 = {res.verdict}")
    print(f"  StereoRig.from_result(allow_unusable=False) -> "
          f"{'**竟然生成了**' if built else '拒绝：' + refusal}")

    log = logging.getLogger("selftest_calib_facts")
    meta = {"board": SPEC.describe(), "source": "camera"}
    checks = [
        # 判据 1：判定逻辑没被放宽
        ("res.ok is False（判定逻辑一字未改）", res.ok is False),
        # 判据 2：第二道闸门还在
        ("StereoRig.from_result(allow_unusable=False) 仍抛 ValueError（第二道闸门）",
         built is False and "不可用" in refusal),
    ]

    with tempfile.TemporaryDirectory() as td:
        # 先在正式路径放一份「好标定」哨兵，逐字节钉住，验证绝不覆盖。
        formal = Path(td) / "stereo_calibration.npz"
        formal_json = formal.with_suffix(".json")
        sentinel = b"GOOD-CALIBRATION-DO-NOT-TOUCH"
        formal.write_bytes(sentinel)
        formal_json.write_text('{"good": true}', encoding="utf-8")

        # ---- 判据 3/4：默认（无 --force-save）跑一次不可用标定 ----
        rig_default = StereoRig.from_result(res, meta=dict(meta), allow_unusable=True)
        args_default = SimpleNamespace(out=str(formal), force_save=False)
        npz_b, js_b, bypassed = cs.finalize_and_save(
            rig_default, args_default, unusable=True, log=log)

        formal_unchanged = (formal.read_bytes() == sentinel
                            and formal_json.read_text(encoding="utf-8") == '{"good": true}')
        bypass_npz, bypass_js = Path(npz_b), Path(js_b)
        bypass_pair = (bypass_npz.is_file() and bypass_js.is_file()
                       and bypass_npz.name.startswith("stereo_calibration_unusable_")
                       and bypass_js.name.startswith("stereo_calibration_unusable_")
                       and bypass_js.suffix == ".json")
        q_default = (json.loads(bypass_js.read_text(encoding="utf-8")).get("quality", {})
                     if bypass_js.is_file() else {})
        marks_default = (q_default.get("forced_unusable") is True
                         and q_default.get("grade") == "不可用"
                         and "仅链路测试" in str(q_default.get("verdict") or ""))
        print(f"  默认：旁路 {bypass_npz.name} / {bypass_js.name}  "
              f"bypassed={bypassed}  正式路径逐字节未改动={formal_unchanged}")
        print(f"  旁路 quality：forced_unusable={q_default.get('forced_unusable')}  "
              f"grade={q_default.get('grade')}  verdict={q_default.get('verdict')}")

        checks += [
            # 判据 3：正式 args.out 逐字节不变（最硬）
            ("默认跑一次：正式 --out 路径内容**逐字节不变**（绝不覆盖好标定）",
             formal_unchanged and bypassed),
            # 判据 4：旁路 npz+json 双产物 + 标记齐全
            ("默认产出旁路 .npz + .json，且 forced_unusable/grade(不可用)/verdict(仅链路测试) 齐全",
             bypass_pair and marks_default),
        ]

        # ---- 判据 5：显式 --force-save 写正式路径且标记齐全 ----
        formal2 = Path(td) / "forced_calibration.npz"
        rig_force = StereoRig.from_result(res, meta=dict(meta), allow_unusable=True)
        args_force = SimpleNamespace(out=str(formal2), force_save=True)
        npz_f, js_f, bypassed_f = cs.finalize_and_save(
            rig_force, args_force, unusable=True, log=log)
        jf = Path(js_f)
        q_force = (json.loads(jf.read_text(encoding="utf-8")).get("quality", {})
                   if jf.is_file() else {})
        force_wrote_formal = (Path(npz_f) == formal2.with_suffix(".npz")
                              and Path(npz_f).is_file()
                              and not bypassed_f)
        marks_force = (q_force.get("forced_unusable") is True
                       and q_force.get("grade") == "不可用"
                       and "仅链路测试" in str(q_force.get("verdict") or ""))
        print(f"  显式 --force-save：写正式路径={force_wrote_formal}  标记齐全={marks_force}")
        checks.append(("显式 --force-save -> 写正式路径且标记齐全（仅链路测试）",
                       force_wrote_formal and marks_force))

        # ---- 判据 6：record_plan 对这份 forced_unusable 标定仍拒绝录制 ----
        loaded = StereoRig.load(bypass_js)          # 读旁路产物
        plan = plan_record(loaded, mode_choice="auto", width=1280, height=720)
        print(f"  record_plan(allow_forced=False) -> ok={plan.ok}  forced={plan.forced}"
              f"  结论：{plan.headline}")
        checks.append(("stereo/record_plan.py 对这份 forced_unusable 标定仍拒绝录制（防呆没破）",
                       plan.ok is False and plan.forced is True))

    # ---- 判据 7：源码级护栏（:153 / :159 更新版）----
    checks.append(("**源码级**：params.py 里仍有 `if not res.ok: raise` 第二道闸门",
                   _params_gate_present()))
    checks.append(("**源码级**：calibrate_stereo.py 走旁路保存、不再 `return 1`",
                   _app_bypasses_on_unusable()))

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


def _write_world_json(path: Path, T, **extra) -> None:
    payload = {"version": 1, "source": "manual", "units": "m",
               "note": "", "board_rmse_px": None,
               "created_at": "2026-01-01 00:00:00",
               "T_world_cam": np.asarray(T).tolist()}
    payload.update(extra)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_world_frame_roundtrip() -> bool:
    """落盘 -> 读回逐元素一致；三类坏矩阵 + 坏 ground_axis 都必须被拒且指出原因。"""
    print("----- 6. 世界系：往返 + 坏文件必须被拒 -----")
    checks: list[tuple[str, bool]] = []
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "world_frame.json"
        WorldFrame(WORLD_T, "manual", note="合成", ground_axis="-z").save(p)
        data = json.loads(p.read_text(encoding="utf-8"))
        checks.append(("JSON 含 version/T_world_cam/source/units/note/created_at",
                       all(k in data for k in ("version", "T_world_cam", "source",
                                               "units", "note", "board_rmse_px",
                                               "created_at"))))
        # 符号必须落地：矩阵本身说不清 Z 朝上还是朝下。
        checks.append(("JSON 含 ground_axis 且写的就是 -z",
                       data.get("ground_axis") == "-z"))
        back = WorldFrame.load(p)
        checks.append(("往返后矩阵逐元素一致（< 1e-12）",
                       float(np.max(np.abs(back.T_world_cam - WORLD_T))) < 1e-12))
        checks.append(("source/note 一并带回",
                       back.source == "manual" and back.note == "合成"))
        checks.append(("ground_axis 往返一致（-z 不会悄悄变回 z）",
                       back.ground_axis == "-z"))

        # 重力必须跟世界系的**Z 轴符号**走：原来预测段写死 (0,-9.81,0)
        # 等于假设"世界 Y 朝上"，与本项目两种 ground_axis 都不符。
        from apps.run_stereo_video import world_gravity_world
        checks.append(("世界系重力：z -> +Z，-z -> -Z，空 -> z",
                       world_gravity_world("z") == (0.0, 0.0, 9.81)
                       and world_gravity_world("-z") == (0.0, 0.0, -9.81)
                       and world_gravity_world("") == (0.0, 0.0, 9.81)))

        # 老文件（没有 ground_axis 字段）按 z 读；写了个非法值必须拒。
        _write_world_json(Path(td) / "old.json", WORLD_T)
        old = WorldFrame.load(Path(td) / "old.json")
        checks.append(("没有 ground_axis 的老文件按 z 读（那时唯一的默认符号）",
                       old.ground_axis == "z"))
        bad_axis = Path(td) / "bad_axis.json"
        _write_world_json(bad_axis, WORLD_T, ground_axis="y")
        message = ""
        try:
            WorldFrame.load(bad_axis)
        except ValueError as exc:
            message = str(exc)
        checks.append(("坏 ground_axis（'y'）被拒且指出文件与允许值",
                       "ground_axis" in message and str(bad_axis) in message
                       and "-z" in message))
        print(f"    ground_axis='y' -> {message or '（**没有报错**）'}")
        try:
            WorldFrame(WORLD_T, "manual", ground_axis="up")
        except ValueError as exc:
            checks.append(("构造 WorldFrame 时 ground_axis 只收 z/-z", "ground_axis" in str(exc)))
        else:
            checks.append(("构造 WorldFrame 时 ground_axis 只收 z/-z", False))

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


def _write_board_photo(path: Path, spec: BoardSpec, K: np.ndarray,
                       size: tuple[int, int] = (1920, 1080)) -> None:
    """画一张**真的能被 `detect_board` 检出**的棋盘照片（透视投影 + 白边）。

    ``size`` 是画布尺寸：默认就是 ``K`` 对应的 1920x1080；传别的值可以造
    "拍图分辨率与标定不一致"的样本（那种图不需要检得到板，尺寸校验先拦）。
    """
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
    photo = cv2.warpPerspective(flat, H, size, borderValue=128)
    cv2.imwrite(str(path), photo)


def test_world_frame_from_board_photo() -> bool:
    """真图片路径：`from_board()` 能检出、rmse 过门槛、尺度与 Z 方向说得清。

    另一条（2026-11-20 补的真实缺口）：**拍图分辨率必须与标定一致**。
    在此之前 `from_board()` 从不看 `rig.image_size` —— 拿 1280x720 的图配
    1920x1080 的内参，PnP 会给出错尺度的 `t`，而 `rmse` 依然很小。
    """
    print("----- 8. 世界系：真图片 from_board()（含分辨率校验 + 符号） -----")
    checks: list[tuple[str, bool]] = []
    K = np.array([[1400.0, 0.0, 960.0], [0.0, 1400.0, 540.0], [0.0, 0.0, 1.0]])
    rig = StereoRig(
        K1=K, D1=np.zeros((1, 5)), K2=K.copy(), D2=np.zeros((1, 5)),
        R=np.eye(3), T=np.array([[-0.065], [0.0], [0.0]]),
        image_size=(1920, 1080),
        meta={"stereo_mode": "angled", "rectification": "none_raw"},
    )
    rig_name = "calib/rig_a"
    with tempfile.TemporaryDirectory() as td:
        photo = Path(td) / "board.png"
        _write_board_photo(photo, SPEC, K)
        checks.append(("照片真的写出来了", photo.is_file()))

        wf = from_board(photo, SPEC, rig, rig_path=rig_name)
        checks.append((f"rmse = {wf.board_rmse_px:.2f} px < 1.0",
                       wf.board_rmse_px is not None and wf.board_rmse_px < 1.0))
        checks.append(("source = board_pnp", wf.source == "board_pnp"))
        checks.append(("describe() 含来源与 rmse",
                       "board_pnp" in wf.describe() and "rmse=" in wf.describe()))
        # ---- ground_axis 必须写进 describe（矩阵本身说不清符号）----
        desc = wf.describe()
        checks.append(("describe() 含 ground_axis 与它的符号说明",
                       "ground_axis=z" in desc and "地面以下" in desc
                       and "+Y" in desc))
        checks.append(("WorldFrame 记住了 ground_axis=z", wf.ground_axis == "z"))

        # 相机到世界原点的距离应等于合成时的板距（~0.5 m）——尺度对了才叫世界系。
        cam = wf.apply(np.zeros((1, 3)))[0]        # 左相机光心在世界系里的位置
        dist = float(np.linalg.norm(cam))
        checks.append((f"相机光心距世界原点 {dist:.3f} m（合成板距 0.5 m）",
                       0.4 < dist < 0.6))
        # 默认 +Z = 棋盘法向 = 背离相机的那一侧 -> 相机在负 Z。
        checks.append((f"默认 ground_axis=z 时相机在 Z<0 一侧（z={cam[2]:.3f}）",
                       cam[2] < 0.0))
        wf_up = from_board(photo, SPEC, rig, ground_axis="-z", rig_path=rig_name)
        cam_up = wf_up.apply(np.zeros((1, 3)))[0]
        checks.append((f"ground_axis='-z' 时相机翻到 Z>0（z={cam_up[2]:.3f}）",
                       cam_up[2] > 0.0))
        checks.append(("两种 ground_axis 的 rmse 相同（只翻轴、不动拟合）",
                       abs(wf.board_rmse_px - wf_up.board_rmse_px) < 1e-12))
        # 数值方向：-z 不是"再转一下"，而是**精确地** X 保持、Y 与 Z 同时取反。
        probe = np.array([[0.12, -0.34, 0.56], [-0.2, 0.3, 0.9]])
        base, flipped = wf.apply(probe), wf_up.apply(probe)
        expect = base * np.array([1.0, -1.0, -1.0])
        checks.append(("-z 的数值方向 = X 不变、Y 与 Z 同时取反（右手系代价）",
                       float(np.max(np.abs(flipped - expect))) < 1e-12))
        desc_up = wf_up.describe()
        checks.append(("ground_axis=-z 的 describe() 说清 Y 也反向（不只是 Z）",
                       "ground_axis=-z" in desc_up and "Y 同时反向" in desc_up
                       and "天空" in desc_up))

        # ---- 拍图分辨率 vs 标定分辨率 ----
        bad_size = Path(td) / "board_720p.png"
        _write_board_photo(bad_size, SPEC, K, size=(1280, 720))
        message = ""
        try:
            from_board(bad_size, SPEC, rig, rig_path=rig_name)
        except ValueError as exc:
            message = str(exc)
        checks.append(("分辨率不一致 -> ValueError，文案含两边尺寸 / 标定文件 / 下一步",
                       "1280x720" in message and "1920x1080" in message
                       # 路径分隔符随平台走（Windows 是 \），比较前先归一
                       and f"{rig_name}.npz" in message.replace("\\", "/")
                       and ("重拍" in message or "重标" in message)))
        print(f"    1280x720 配 1920x1080 标定 -> {message or '（**没有报错**）'}")
        checks.append(("尺寸一致时同一套参数能通过（不误报）",
                       from_board(photo, SPEC, rig, rig_path=rig_name)
                       .source == "board_pnp"))

        # 标定里没有有效 image_size：**明确跳过**并写进 note，不许静默放行。
        orig_size = rig.image_size
        try:
            for label, value in (("(0, 0)（老 npz 缺字段时的读法）", (0, 0)),
                                 ("None", None)):
                rig.image_size = value
                wf_skip = from_board(photo, SPEC, rig, rig_path=rig_name)
                note = wf_skip.note
                checks.append((f"image_size={label} -> 跳过校验但写明原因",
                               "跳过" in note and "image_size" in note
                               and "分辨率" in note))
            del rig.image_size                      # 连属性都没有的装置对象
            wf_skip = from_board(photo, SPEC, rig, rig_path=rig_name)
            checks.append(("image_size 属性缺失 -> 同样明确跳过（读不到就说明读不到）",
                           "跳过" in wf_skip.note and "image_size" in wf_skip.note))
        finally:
            rig.image_size = orig_size

        blank = Path(td) / "blank.png"
        # 尺寸要跟标定一致：否则先被分辨率校验拦下，"检不到板"这条路径就没测到。
        cv2.imwrite(str(blank), np.full((1080, 1920), 255, np.uint8))
        message = ""
        try:
            from_board(blank, SPEC, rig, rig_path=rig_name)
        except ValueError as exc:
            message = str(exc)
        checks.append(("空图（检不到板）抛 ValueError 并说明怎么重拍",
                       "没检测到" in message and "重拍" in message))
        print(f"    空图 -> {message}")

        missing = Path(td) / "nope.png"
        try:
            from_board(missing, SPEC, rig, rig_path=rig_name)
        except ValueError as exc:
            checks.append(("图打不开 -> ValueError 指出路径", str(missing) in str(exc)))
        else:
            checks.append(("图打不开 -> ValueError 指出路径", False))

        # ---- CLI 真的能收下 `--ground-axis -z`（界面发的就是这两段）----
        # argparse 会把单独的 `-z` 当成另一个选项，于是 `--ground-axis -z`
        # 报 "expected one argument" 并退出 2 —— 界面里选「Z 轴朝上」就成了
        # 一个退出码 2 的失败进程。脚本用 `_join_axis_value()` 把两段拼起来。
        from apps.tools.set_world_frame import _join_axis_value, parse_args
        checks.append(("`--ground-axis -z` 两段写法被拼成 `--ground-axis=-z`",
                       _join_axis_value(["--image", "x", "--ground-axis", "-z"])
                       == ["--image", "x", "--ground-axis=-z"]))
        checks.append(("`--ground-axis z` 原样保留（不再多拼一层）",
                       _join_axis_value(["--ground-axis", "z"]) == ["--ground-axis", "z"]))
        try:
            parsed = parse_args(["--image", "x.png", "--ground-axis", "-z"])
        except SystemExit as exc:                       # argparse 退出 2 会走到这里
            checks.append((f"CLI 解析 `--ground-axis -z`（argparse 退出 {exc.code}）", False))
        else:
            checks.append(("CLI 解析 `--ground-axis -z` 得到 ground_axis='-z'",
                           parsed.ground_axis == "-z"))

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


# ==========================================================================
# 第 10-11 节：交叉验证 + 外参不确定度（2026-10-26 加）
#
# 为什么要在这里钉：上面 1-9 节钉的都是"标定结果怎么判"，而**没有任何一节**
# 回答 "这个 RMS 换一批图还复现吗"（`calib/validation.py`）与
# "基线 T 的不确定度是多少"（`calib/uncertainty.py`）。
# 缺了这两个，报告里的 "RMS 0.2 px" 就是**训练集自评**，
# 而 `depth_sigma_mm()` 的 docstring 自己承认"不含 σ_B"。
# ==========================================================================

# 交叉验证用的"假拟合器"口径：数值本身不重要，重要的是判定能不能把它认出来
CV_GOOD = (0.20, 0.25)      # ratio 1.25 -> 泛化良好
CV_OVERFIT = (0.20, 0.60)   # ratio 3.00 -> 过拟合
CV_LIGHT = (0.20, 0.40)     # ratio 2.00 -> 轻度过拟合
CV_N = 20                   # 20 对图像（本项目"建议 20 对以上"那条线）


def _cv_fit(train_rms: float, test_rms: float, boom_at: int | None = None):
    """造一个 `fit_fn`：恒定输出给定 RMS；`boom_at` 指定的那一折抛异常。

    用恒定值而不是真跑标定，是为了**只测判定逻辑本身** —— 若判定写错，
    真标定那点噪声会把错误掩盖成"看起来也差不多"。
    """
    state = {"fold": -1}

    def fit(train_idx, test_idx):
        state["fold"] += 1
        if boom_at is not None and state["fold"] == boom_at:
            raise RuntimeError(f"模拟第 {state['fold']} 折 stereoCalibrate 发散")
        # 新契约（2026-10-26）：必须上报真正参与拟合的索引，否则结果降级为"未验证"
        return {"train_rms": train_rms, "test_rms": test_rms,
                "used_train_idx": train_idx,
                "train_rms_estimator": "same", "test_rms_estimator": "same"}

    return fit


def test_cross_validate_verdicts() -> bool:
    """第 10 节：交叉验证的四种判定（比值 1.25 / 2.0 / 3.0、边界、某折炸掉）。"""
    print("----- 10. 交叉验证：训练/测试比值 -> 判定 -----")
    checks: list[tuple[str, bool]] = []

    rep = cross_validate(_cv_fit(*CV_GOOD), CV_N, k=4, seed=0)
    print(f"    train 0.20 / test 0.25 -> ratio {rep['ratio']:.2f}  "
          f"verdict「{rep['verdict']}」")
    checks.append(("0.20/0.25（ratio 1.25）-> 泛化良好",
                   rep["verdict"] == "泛化良好" and not rep["failed_folds"]))
    checks.append((f"4 折都成功、每折 {CV_N // 4} 个测试样本",
                   rep["n_folds_ok"] == 4
                   and all(len(f["test_idx"]) == CV_N // 4 for f in rep["per_fold"])))

    rep = cross_validate(_cv_fit(*CV_LIGHT), CV_N, k=4, seed=0)
    print(f"    train 0.20 / test 0.40 -> ratio {rep['ratio']:.2f}  "
          f"verdict「{rep['verdict']}」")
    checks.append(("0.20/0.40（ratio 2.00）-> 轻度过拟合",
                   rep["verdict"] == "轻度过拟合：样本偏少或位姿单一"))

    rep = cross_validate(_cv_fit(*CV_OVERFIT), CV_N, k=4, seed=0)
    print(f"    train 0.20 / test 0.60 -> ratio {rep['ratio']:.2f}  "
          f"verdict「{rep['verdict']}」")
    # 用户看到 "RMS 0.2 px" 就以为可靠 —— 3 倍差距必须被明确否掉，而不是"轻度过拟合"
    checks.append(("0.20/0.60（ratio 3.00）-> 过拟合：结果不可用于测量",
                   rep["verdict"] == "过拟合：结果不可用于测量"))

    # 边界：3 个样本拆不出 4 折 —— 必须抛错并说明需要多少帧
    message = ""
    try:
        cross_validate(_cv_fit(*CV_GOOD), 3, k=4, seed=0)
    except ValueError as exc:
        message = str(exc)
    print(f"    n_items=3, k=4 -> ValueError: {message.splitlines()[0]}")
    checks.append(("n_items=3 < k=4 -> ValueError（且写明至少需要 4 帧）",
                   "至少 4 帧" in message and "3 帧" in message))

    # 某折抛异常：**不许静默跳过**。4 折里炸 1 折，结论就不能是"泛化良好"
    rep = cross_validate(_cv_fit(*CV_GOOD, boom_at=2), CV_N, k=4, seed=0)
    print(f"    第 2 折抛异常 -> failed_folds={len(rep['failed_folds'])}  "
          f"n_folds_ok={rep['n_folds_ok']}  verdict「{rep['verdict']}」")
    checks.append(("某折抛异常 -> failed_folds 非空（不是静默跳过）",
                   len(rep["failed_folds"]) == 1
                   and rep["failed_folds"][0]["error"] is not None))
    checks.append(("某折抛异常 -> verdict 不是「泛化良好」",
                   rep["verdict"] != "泛化良好" and "部分折失败" in rep["verdict"]))

    # k=1：退化。test_rms 恒等于 train_rms，ratio 恒为 1 —— 不能冒充泛化证据
    rep = cross_validate(_cv_fit(*CV_OVERFIT), CV_N, k=1, seed=0)
    print(f"    k=1 -> degenerate={rep['degenerate']}  verdict「{rep['verdict']}」")
    checks.append(("k=1 -> degenerate=True 且 verdict 写明「退化」",
                   rep["degenerate"] is True and "退化" in rep["verdict"]))

    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


# --------------------------------------------------------------------------
# 合成双目装置（**已知真值**，用来把"预测的 σ_B"和"实测的基线散布"对上）
# --------------------------------------------------------------------------
RIG_IMG = (1280, 720)
RIG_FX = 1150.0
RIG_D = np.array([-0.08, 0.02, 0.0, 0.0, 0.0])
RIG_BASELINE_MM = 65.0        # 卷尺口径（`selftest_board_corner_order.TAPE_BASELINE_MM`）
RIG_NOISE_PX = 0.3            # 亚像素噪声（与 stereo_calib 注释里的合成实验同量级）
RIG_VIEWS = 20
RIG_TRIALS = 30               # 30 次不同噪声实现
RIG_REL_ROT = np.array([0.0, 0.10, 0.0])
RIG_REL_T = np.array([[-RIG_BASELINE_MM / 1000.0], [0.004], [0.01]])


def _rig_K() -> np.ndarray:
    return np.array([[RIG_FX, 0.0, RIG_IMG[0] / 2.0],
                     [0.0, RIG_FX, RIG_IMG[1] / 2.0],
                     [0.0, 0.0, 1.0]])


def _synth_rig(seed: int):
    """造一台"已知 R、T"的双目：返回 (objp, K, ip1, ip2, R_true, T_true)。"""
    K = _rig_K()
    objp = SPEC.object_points().reshape(-1, 1, 3).astype(np.float32)
    R_rel = cv2.Rodrigues(RIG_REL_ROT)[0]
    rng = np.random.default_rng(seed)
    cv2.setRNGSeed(seed)          # 让 stereoCalibrate 内部也逐次可复现
    ip1, ip2 = [], []
    for _ in range(RIG_VIEWS):
        rvec = rng.uniform(-0.5, 0.5, 3)
        tvec = np.array([rng.uniform(-0.07, 0.07),
                         rng.uniform(-0.05, 0.05),
                         rng.uniform(0.45, 0.80)])
        Rv = cv2.Rodrigues(rvec)[0]
        p1, _ = cv2.projectPoints(objp, rvec, tvec, K, RIG_D)
        rvec_r = cv2.Rodrigues(R_rel @ Rv)[0]
        p2, _ = cv2.projectPoints(objp, rvec_r, R_rel @ tvec + RIG_REL_T.ravel(),
                                  K, RIG_D)
        ip1.append((p1.reshape(-1, 2) + rng.normal(0.0, RIG_NOISE_PX, (SPEC.n_points, 2)))
                   .astype(np.float32).reshape(-1, 1, 2))
        ip2.append((p2.reshape(-1, 2) + rng.normal(0.0, RIG_NOISE_PX, (SPEC.n_points, 2)))
                   .astype(np.float32).reshape(-1, 1, 2))
    return objp, K, ip1, ip2, R_rel, RIG_REL_T


def _fit_extrinsic_only(ip1, ip2, K):
    """固定内参做一次双目标定，返回 (R, T) —— 与 `_synth_rig` 的真值可比。"""
    obj = SPEC.object_points().reshape(-1, 1, 3).astype(np.float32)
    out = cv2.stereoCalibrate([obj] * len(ip1), ip1, ip2,
                              K.copy(), RIG_D.copy(), K.copy(), RIG_D.copy(),
                              RIG_IMG, flags=cv2.CALIB_FIX_INTRINSIC)
    return out[5], out[6]


def test_extrinsic_sigma_vs_empirical() -> bool:
    """第 11a 节：预测的 σ_B vs 30 次实测基线散布（比值必须落在 0.5~2.0）。

    这是**唯一能证明 σ_B 不是瞎编**的实验：合成装置知道真值，
    换 30 次噪声实现就能量出"基线到底抖多少"，预测值必须在同一量级。
    """
    print("----- 11a. σ_B：预测值 vs 30 次实测基线散布 -----")
    checks: list[tuple[str, bool]] = []
    baselines: list[float] = []
    predicted: list[float] = []
    angles: list[float] = []
    for s in range(RIG_TRIALS):
        objp, K, ip1, ip2, _Rt, _Tt = _synth_rig(s)
        R, T = _fit_extrinsic_only(ip1, ip2, K)
        u = extrinsic_sigma(objp, ip1, ip2, K, RIG_D, K, RIG_D, R, T, RIG_IMG)
        if "error" in u:
            print(f"    第 {s} 次 -> 意外奇异：{u['error']}")
            checks.append((f"第 {s} 次不该奇异", False))
            continue
        baselines.append(float(np.linalg.norm(T)) * 1000.0)
        predicted.append(u["sigma_baseline_mm"])
        angles.append(u["sigma_angle_deg"])

    emp = float(np.std(np.asarray(baselines), ddof=1))
    pred = float(np.median(np.asarray(predicted)))
    ratio = pred / emp if emp > 0 else float("inf")
    print(f"    实测基线 {np.mean(baselines):.2f} ± {emp:.4f} mm（n={len(baselines)}，"
          f"真值 {RIG_BASELINE_MM:.1f} mm）")
    print(f"    预测 σ_B 中位数 {pred:.4f} mm  ->  预测/实测 = {ratio:.2f}")
    print(f"    预测 σ_angle 中位数 {np.median(angles):.4f}°")
    # 门限 0.5~2.0：数量级正确即可。预测偏低是因为 σ² 由同一批残差估、
    # 且各视图位姿被 solvePnP 吸收了一部分噪声（条件协方差 vs 联合估计）。
    checks.append((f"预测/实测 = {ratio:.2f}，落在 [0.5, 2.0]", 0.5 <= ratio <= 2.0))
    checks.append((f"σ_B 远小于基线（{pred:.3f} mm << {RIG_BASELINE_MM:.0f} mm），"
                   f"不是把整条基线当误差", pred < 0.1 * RIG_BASELINE_MM))
    checks.append((f"σ_angle = {np.median(angles):.4f}° < 1°（外参旋转可信）",
                   float(np.median(angles)) < 1.0))

    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_depth_sigma_includes_baseline() -> bool:
    """第 11b 节：深度公式**含** σ_B（正是 `depth_sigma_mm()` 缺的那项）+ 奇异不混过去。"""
    print("----- 11b. 深度不确定度含 σ_B + 奇异案例 -----")
    checks: list[tuple[str, bool]] = []
    objp, K, ip1, ip2, _Rt, _Tt = _synth_rig(0)
    R, T = _fit_extrinsic_only(ip1, ip2, K)
    u = extrinsic_sigma(objp, ip1, ip2, K, RIG_D, K, RIG_D, R, T, RIG_IMG)
    checks.append(("正常装置不返回 error", "error" not in u))
    if "error" in u:
        print(f"    意外奇异：{u['error']}")
        for name, ok in checks:
            print(f"  [{'通过' if ok else '不达标'}] {name}")
        return False

    z = 1.0
    dz_full = u["depth_sigma_mm_at"](z)
    # "只用 σ_f"：`stereo_calib.depth_sigma_mm()` 那套口径（σ_d + σ_f，**无** σ_B）。
    # σ_f 取 `calibrateCameraExtended` 在这类合成数据上的典型量级 0.5 px。
    sigma_f_px = 0.5
    dz_f_only = z * sigma_f_px / RIG_FX * 1000.0
    print(f"    σ_B = {u['sigma_baseline_mm']:.4f} mm，基线 {u['baseline_mm']:.2f} mm")
    print(f"    1.0 m 处：含 σ_B 的 σ_Z = {dz_full:.2f} mm；"
          f"只用 σ_f = {dz_f_only:.2f} mm（比值 {dz_full / dz_f_only:.1f}x）")
    checks.append((f"depth_sigma_mm_at(1.0) = {dz_full:.2f} mm > 只用 σ_f 的 "
                   f"{dz_f_only:.2f} mm（σ_B 项真的算进去了）", dz_full > dz_f_only))

    # 光"大于"还不够：1 m 处视差项本来就占 95%，少算 σ_B 只差 5%。
    # 所以这里把公式**逐项复算**一遍 —— 少一项就会对不上。
    b_m = u["baseline_mm"] / 1000.0
    d_px = u["focal_px"] * b_m / z
    sd_px = np.sqrt(2.0) * u["noise_px"]
    term_d = z * sd_px / d_px * 1000.0
    term_b = z * (u["sigma_baseline_mm"] / 1000.0) / b_m * 1000.0
    recomputed = float(np.hypot(term_d, term_b))
    print(f"    逐项复算：视差项 {term_d:.3f} mm + 基线项 {term_b:.3f} mm"
          f"（各占 {(term_d / dz_full) ** 2 * 100:.1f}% / "
          f"{(term_b / dz_full) ** 2 * 100:.1f}% 的方差）-> {recomputed:.3f} mm")
    checks.append((f"σ_Z 逐项复算一致（{recomputed:.3f} vs {dz_full:.3f} mm）："
                   f"σ_Z = Z·√((σ_d/d)² + (σ_B/B)²)，基线项确实在式子里",
                   abs(recomputed - dz_full) <= 1e-9 * max(dz_full, 1.0)))
    b_term_06 = 0.6 * (u["sigma_baseline_mm"] / 1000.0) / b_m * 1000.0
    share_06 = b_term_06 / u["depth_sigma_mm_at"](0.6)
    print(f"    0.6 m 处基线项 {b_term_06:.2f} mm，占 σ_Z 的 {share_06 * 100:.1f}%"
          f"（越近占比越大：它是 Z/B 倍的放大）")
    checks.append((f"0.6 m 处基线项占 σ_Z 的 {share_06 * 100:.1f}% > 5%"
                   f"（不是可忽略的常数）", share_06 > 0.05))
    checks.append((f"σ_Z 随距离增长：1.0 m -> {dz_full:.2f} mm，2.0 m -> "
                   f"{u['depth_sigma_mm_at'](2.0):.2f} mm",
                   u["depth_sigma_mm_at"](2.0) > dz_full))
    checks.append(("σ_t 与 σ_r 都是 3 维且有限",
                   np.all(np.isfinite(u["sigma_t_mm"]))
                   and np.all(np.isfinite(u["sigma_r_deg"]))
                   and u["sigma_t_mm"].shape == (3,)))

    # ---- 奇异：板面与右目光心共面（完全侧视）-> 右目深度维退化 ----
    # 物理含义：板上的点与第二台相机的光心共线，于是"绕这条线的旋转"与
    # "沿板面的平移"在残差上完全等价 —— 6 个自由度里至少有一个不可观测。
    obj_line = np.stack([np.linspace(-0.09, 0.09, SPEC.n_points),
                         np.zeros(SPEC.n_points), np.full(SPEC.n_points, 0.6)], 1).astype(np.float32)
    p1, _ = cv2.projectPoints(obj_line, np.zeros(3), np.zeros(3), K, RIG_D)
    T_side = np.array([[0.07], [0.0], [-0.6]])
    p2, _ = cv2.projectPoints(obj_line, np.zeros(3), T_side, K, RIG_D)
    bad = extrinsic_sigma(obj_line, [p1.astype(np.float32)],
                          [p2.astype(np.float32)], K, RIG_D, K, RIG_D,
                          np.eye(3), T_side, RIG_IMG)
    msg = bad.get("error", "")
    print(f"    侧视退化 -> {msg or bad}")
    checks.append(("退化几何 -> 返回 error，且是「雅可比奇异」那条",
                   "error" in bad and "奇异" in msg))
    checks.append(("退化几何 -> **没有** inf/nan 混过去（error 里无 sigma_* 数值）",
                   not any(k.startswith("sigma_") for k in bad)))

    # ---- 残差维数不足：3 个点 1 个视图 -> m=6，减 6 个自由度后 ≤0 ----
    # 这时**根本没有噪声估计的自由度**，必须明确拒绝而不是给出一个"很小的 σ"。
    few = extrinsic_sigma(obj_line[:3], [p1[:3].astype(np.float32)],
                          [p2[:3].astype(np.float32)], K, RIG_D, K, RIG_D,
                          np.eye(3), T_side, RIG_IMG)
    print(f"    1 视图 x 3 点 -> {few.get('error', few)}")
    checks.append(("残差维数不足 -> error（m−6 ≤ 0 时不假装能估噪声）",
                   "error" in few and "残差维数" in few.get("error", "")))

    # ---- 左右点数不一致：对应关系已不可信，直接抛错 ----
    message = ""
    try:
        extrinsic_sigma(obj_line, [p1.astype(np.float32)],
                        [p2[:4].astype(np.float32)], K, RIG_D, K, RIG_D,
                        np.eye(3), T_side, RIG_IMG)
    except ValueError as exc:
        message = str(exc)
    print(f"    左右点数 40/4 -> ValueError: {message.splitlines()[0][:70]}")
    checks.append(("左右点数不一致 -> ValueError（不是截断后继续算）",
                   "期望" in message and "40" in message))

    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


# --------------------------------------------------------------------------
# A3 + B1：chain 的板占闸 + 以链乘为初值的跨路残差精化（2026-11-18）
#
# chain 走 `method.run` 提前 return，从不到达 joint 收尾处的
# "板占画面比例 / 板太小最高只判可用"。这两节把新接上的闸与新的跨路残差
# 用**已知真值的合成数据 + 真机只读**（后者可选）钉住。
# --------------------------------------------------------------------------

def _chain_synth(z_mm: float, *, n_views: int = 14, seed: int = 7,
                 noise_px: float = 0.12, n_pair: int = 3,
                 pair_noise_px: float = 0.12) -> dict:
    """造一份"双板 L 形 + 90° 装置"的 chain 合成输入（真值全已知）。

    A 板在左相机前 `z_mm`，B 板在右相机前 `z_mm`；B 板面绕 Y 转 45°，
    这样左相机（同框图那一路）也能看见它，不会被当成一条线。
    两块板刚性固连这台支架，参考对与同框图都是**同一摆放**。
    """
    import calib.relative_pose as rp

    W, H = 1280, 720
    K = np.array([[900.0, 0.0, W / 2], [0.0, 900.0, H / 2], [0, 0, 1.0]])
    D = np.zeros(5)
    objp = SPEC.object_points().astype(np.float64)
    z = z_mm / 1000.0

    # ---- 真值外参：右相机在 (0.5,0,0.5)，光轴沿 -X ----
    R_true = cv2.Rodrigues(np.radians([0.0, 90.0, 0.0]))[0]
    C = np.array([0.50, 0.0, 0.50])
    T_true = (-R_true @ C).reshape(3, 1)
    T_cr_cl = rp.rt_to_T(R_true, T_true)

    # ---- 双板：A 正对左相机；B 在右相机系里转 45°（两相机都能看见）----
    T_cl_A = rp.rt_to_T(np.eye(3), [0.0, 0.0, z])
    T_cr_B = rp.rt_to_T(cv2.Rodrigues(np.radians([0.0, 45.0, 0.0]))[0], [0.0, 0.0, z])
    Tab = rp.compose(rp.invert(T_cr_B), T_cr_cl, T_cl_A)   # T_{B←A}
    T_cl_B = rp.compose(rp.invert(T_cr_cl), T_cr_B)        # 同框图里 B 在左相机系的位姿

    rng = np.random.default_rng(seed)
    img_l, img_r = [], []
    for _ in range(n_views):
        dR = cv2.Rodrigues(np.radians(rng.uniform(-7, 7, 3)))[0]
        dt = rng.uniform(-0.012, 0.012, 3)
        for base_T, out in ((T_cl_A, img_l), (T_cr_B, img_r)):
            T = rp.rt_to_T(dR @ base_T[:3, :3], base_T[:3, 3] + dt)
            p = cv2.projectPoints(objp, cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], K, D)[0]
            out.append((p + rng.normal(0, noise_px, p.shape)).astype(np.float32))

    # ---- 同框图：左相机（pair camera ④）同时看到 A、B，支架不动 ----
    pair_frames = []
    for _ in range(n_pair):
        pa = cv2.projectPoints(objp, np.zeros(3), T_cl_A[:3, 3], K, D)[0]
        pb = cv2.projectPoints(objp, cv2.Rodrigues(T_cl_B[:3, :3])[0], T_cl_B[:3, 3], K, D)[0]
        pa = (pa + rng.normal(0, pair_noise_px, pa.shape)).astype(np.float64).reshape(-1, 2)
        pb = (pb + rng.normal(0, pair_noise_px, pb.shape)).astype(np.float64).reshape(-1, 2)
        pair_frames.append((pa, pb))

    return dict(objp=objp, img_l=img_l, img_r=img_r, K=K, D=D, size=(W, H),
                Tab=Tab, R_true=R_true, T_true=T_true,
                T_cl_A=T_cl_A, T_cr_B=T_cr_B, T_cl_B=T_cl_B, pair_frames=pair_frames)


def _run_chain_synth(d: dict, *, Tab=None, pair_frames=None, cfg=None):
    """用合成数据跑一次 chain；`Tab`/`pair_frames` 可替换以做反向实验。"""
    from calib.stereo_calib import CalibConfig

    cfg = cfg or CalibConfig(method="chain")
    Tab = d["Tab"] if Tab is None else Tab
    samples = [Tab, Tab.copy()]             # n=2 且一致 -> 不让"只测 1 次"掩盖 B1
    return calibrate_stereo(
        d["objp"], d["img_l"], d["img_r"], d["size"], cfg, SPEC, stereo_mode="angled",
        T_camL_A=d["T_cl_A"], T_AB=Tab, T_camR_B=d["T_cr_B"],
        T_AB_samples=samples, pair_board_corners=pair_frames)


def test_chain_board_area_caps_verdict() -> bool:
    """A3：chain 的 board_area 必须是实数，且板小把"优秀"封顶为"可用"。

    反向自证锚点：把 `_calibrate_chain` 里那两行 board_area 赋值改空，
    本节的"封顶"判据必须变红（NaN 会让 `area_big_enough=True`）。
    """
    print("\n----- A3：chain 接上板占画面闸 -----")
    import calib.stereo_calib as sc

    cfg = CalibConfig(method="chain")
    small = _chain_synth(900.0)      # 板小（约 1.9% / 1.3%）
    big = _chain_synth(340.0)        # 板大（约 11% / 8.6%）
    r_small = _run_chain_synth(small, cfg=cfg)
    r_big = _run_chain_synth(big, cfg=cfg)

    print(f"  小板：board_area_l={r_small.board_area_l * 100:5.2f}%  "
          f"board_area_r={r_small.board_area_r * 100:5.2f}%  "
          f"min={r_small.board_area * 100:5.2f}%  "
          f"RMS 左/右 {r_small.err_left:.3f}/{r_small.err_right:.3f}  "
          f"判定 {r_small.grade}")
    print(f"  大板：board_area_l={r_big.board_area_l * 100:5.2f}%  "
          f"board_area_r={r_big.board_area_r * 100:5.2f}%  "
          f"min={r_big.board_area * 100:5.2f}%  "
          f"RMS 左/右 {r_big.err_left:.3f}/{r_big.err_right:.3f}  "
          f"判定 {r_big.grade}")

    checks = [
        ("chain 的 board_area_l/r 是实数（不再是 NaN）",
         np.isfinite(r_small.board_area_l) and np.isfinite(r_small.board_area_r)),
        ("小板确实落在门限之下（测试前提成立）",
         r_small.board_area < cfg.min_board_area_for_good),
        ("大板确实落在门限之上（测试前提成立）",
         r_big.board_area >= cfg.min_board_area_for_good),
        ("板占 1.9% 时 verdict 被封顶为「可用」（不是「优秀」）",
         r_small.grade == "可用" and r_small.ok),
        ("板大时仍判「优秀」（没把这一档废掉）", r_big.grade == "优秀"),
        ("封顶时给出原因（「最高只判」入 warnings）",
         any("最高只判" in w for w in r_small.warnings)),
        ("报告里出现板占画面（chain 也有出口了）",
         "板占画面" in r_small.report()),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_chain_cross_path_least_squares() -> bool:
    """B1 正证：链乘初值加 0.3°/1mm 扰动 -> 最小二降 RMS 且回到真值附近。

    直接调生产函数 `_refine_chain_cross_path`（护栏由调用方做，不遮住优化本身）。
    """
    import calib.relative_pose as rp
    import calib.stereo_calib as sc

    d = _chain_synth(500.0)
    # ---- 给链乘初值加 0.3° / 1mm 扰动：扰动进 T_AB，链乘结果随之偏 ----
    delta = rp.rt_to_T(cv2.Rodrigues(np.radians([0.3, 0.0, 0.0]))[0], [0.001, 0.0, 0.0])
    Tab_pert = rp.compose(delta, d["Tab"])
    R_init, T_init, _ = rp.relative_extrinsics(d["T_cl_A"], Tab_pert, d["T_cr_B"])
    init_rot = rp.rotation_angle_deg(R_init, d["R_true"])
    init_tr = float(np.linalg.norm(T_init - d["T_true"])) * 1000.0

    R_opt, T_opt, diag = sc._refine_chain_cross_path(
        d["objp"], d["pair_frames"], d["K"], d["D"],
        d["T_cl_A"], d["T_cr_B"], R_init, T_init)
    opt_rot = rp.rotation_angle_deg(R_opt, d["R_true"])
    opt_tr = float(np.linalg.norm(T_opt - d["T_true"])) * 1000.0
    print(f"  初值真值误差：旋转 {init_rot:.4f}° / 平移 {init_tr:.3f} mm")
    print(f"  优化后 RMS {diag['rms_before']:.4f} -> {diag['rms_after']:.4f} px  "
          f"({diag['n_frames']} 帧 / {diag['n_corners']} 角点)")
    print(f"  优化后真值误差：旋转 {opt_rot:.4f}° / 平移 {opt_tr:.3f} mm  "
          f"改动量 {diag['delta_rot_deg']:.4f}° / {diag['delta_trans_mm']:.3f} mm")

    # ---- 端到端：扰动小于护栏时应当**真被采用**（不是只算不算数）----
    # 同框图角点必须与"被标出来的那组内参"同口径：先不带同框图跑一次拿估计
    # K1/D1，再用它重渲染角点。否则 K 的估计误差会被优化器吸进外参，那不是
    # B1 要测的东西（真机上同框图与内参段本来就是同一台相机）。
    r_warm = _run_chain_synth(d, Tab=d["Tab"], pair_frames=None)
    K_est, D_est = r_warm.K1, r_warm.D1
    rng_e = np.random.default_rng(99)
    pair_est = []
    for _ in range(3):
        pa = cv2.projectPoints(d["objp"], np.zeros(3),
                               d["T_cl_A"][:3, 3], K_est, D_est)[0]
        pb = cv2.projectPoints(d["objp"], cv2.Rodrigues(d["T_cl_B"][:3, :3])[0],
                               d["T_cl_B"][:3, 3], K_est, D_est)[0]
        pair_est.append((
            (pa + rng_e.normal(0, 0.12, pa.shape)).astype(np.float64).reshape(-1, 2),
            (pb + rng_e.normal(0, 0.12, pb.shape)).astype(np.float64).reshape(-1, 2)))
    small_delta = rp.rt_to_T(cv2.Rodrigues(np.radians([0.08, 0.0, 0.0]))[0],
                             [0.0003, 0.0, 0.0])
    Tab_small = rp.compose(small_delta, d["Tab"])
    r_end = _run_chain_synth(d, Tab=Tab_small, pair_frames=pair_est)
    refine = r_end.cross_path_refine or {}
    print(f"  端到端（0.08°/0.3mm）：applied={refine.get('applied')}  "
          f"RMS {refine.get('rms_before', float('nan')):.4f} -> "
          f"{refine.get('rms_after', float('nan')):.4f} px  "
          f"旋转 {refine.get('delta_rot_deg', float('nan')):.4f}° / "
          f"平移 {refine.get('delta_trans_mm', float('nan')):.3f} mm")

    checks = [
        ("初值扰动确实在 0.3°/1mm 量级（测试前提成立）",
         0.15 < init_rot < 0.6 and 0.3 < init_tr < 3.0),
        ("least_squares 把角点重投影 RMS 降下来",
         diag["rms_after"] < diag["rms_before"]),
        ("优化结果回到真值附近（<0.1°）", opt_rot < 0.1),
        ("优化结果回到真值附近（<0.5mm）", opt_tr < 0.5),
        ("端到端小扰动时优化结果**真被采用**",
         refine.get("applied") is True and r_end.ok),
        ("诊断字段有改前/改后数值（不是只写「通过」）",
         np.isfinite(refine.get("rms_before", float("nan")))
         and np.isfinite(refine.get("rms_after", float("nan")))),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_chain_cross_path_guardrail_rejects_180() -> bool:
    """B1 护栏反向：初值翻转 179° -> 必须判不可用且不采用优化结果。

    反向自证锚点：把 `_calibrate_chain` 里"改动幅度护栏"的 `if` 改空
    （例如 `if False and (...)`），本节的"判不可用 / 未采用"两条必须变红。
    """
    import calib.relative_pose as rp
    import calib.stereo_calib as sc

    d = _chain_synth(500.0)
    # 模拟同框图角点倒序造成的 180° 错解：T_AB 被翻 179°
    flip = rp.rt_to_T(cv2.Rodrigues(np.radians([179.0, 0.0, 0.0]))[0], np.zeros(3))
    Tab_flip = rp.compose(flip, d["Tab"])
    chain_R, chain_T, _ = rp.relative_extrinsics(d["T_cl_A"], Tab_flip, d["T_cr_B"])
    r = _run_chain_synth(d, Tab=Tab_flip, pair_frames=d["pair_frames"])
    refine = r.cross_path_refine or {}

    print(f"  179° 翻转：改前 RMS {refine.get('rms_before', float('nan')):.3f} -> "
          f"改后 {refine.get('rms_after', float('nan')):.3f} px，"
          f"改动 {refine.get('delta_rot_deg', float('nan')):.2f}° / "
          f"{refine.get('delta_trans_mm', float('nan')):.2f} mm")
    print(f"  verdict：{r.verdict.splitlines()[0][:120]}")
    print(f"  最终 R/T 与链乘原值一致 = "
          f"{np.allclose(r.R, chain_R) and np.allclose(r.T, chain_T)}")

    checks = [
        ("护栏被触发（applied=False 且 reason 点名护栏）",
         refine.get("applied") is False and "护栏" in refine.get("reason", "")),
        ("判不可用（不是悄悄产出看起来收敛的错解）",
         r.grade == "不可用" and r.ok is False),
        ("优化结果**未被采用**：R 保持链乘原值", np.allclose(r.R, chain_R)),
        ("优化结果**未被采用**：T 保持链乘原值", np.allclose(r.T, chain_T)),
        ("verdict 指向重拍第 ④ 步同框图",
         "同框图" in r.verdict and "重拍" in r.verdict),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_chain_cross_path_real_frames_readonly() -> bool:
    """B1 真机只读：拿 chain_frames 四段图跑一次，机制走通/不崩/不覆盖 config。

    不要求标定变好（那份坏标定的真因是同框图角点倒序，B1 不修那个）。
    任何产物只写 `tempfile.mkdtemp()`，跑完清理；对 outputs/ 只读、
    config/ 绝不触碰。数据目录不存在时**跳过**（判据依然为真）。
    """
    import calib.relative_pose as rp
    from calib import detect_board
    from calib.stereo_calib import calibrate_mono_points

    base = ROOT / "outputs" / "chain_frames" / "20261111-143925"
    if not base.is_dir():
        print("  [跳过] 本机没有 outputs/chain_frames/20261111-143925（只读数据不在）—— 非失败")
        return True

    from apps.calibrate_stereo import (
        _detect_two_boards, _list_images, _pnp_reproj_rms, _ref_pair_poses)

    def _corners(paths):
        out, size = [], None
        for p in paths:
            g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if g is None:
                continue
            if size is None:
                size = (g.shape[1], g.shape[0])
            obs = detect_board(g, SPEC, exhaustive=True)
            if obs.found and obs.corners is not None:
                out.append(obs.corners.reshape(-1, 2).astype(np.float32))
        return out, size

    c_l, size_l = _corners(_list_images(base / "left"))
    c_b, _ = _corners(_list_images(base / "board_b"))
    if len(c_l) < 3 or len(c_b) < 3:
        print(f"  [跳过] 真机角点不足（左 {len(c_l)} / 右 {len(c_b)}）—— 非失败")
        return True
    objp = SPEC.object_points()
    mono_l = calibrate_mono_points(objp, c_l, size_l, CalibConfig())
    mono_r = calibrate_mono_points(objp, c_b, size_l, CalibConfig())

    Tab_p = _list_images(base / "ref_pair_left")
    Tab_r = _list_images(base / "ref_pair_right")
    T_cl_A, T_cr_B, _info = _ref_pair_poses(
        Tab_p, Tab_r, SPEC, objp, mono_l["K"], mono_l["D"],
        mono_r["K"], mono_r["D"], size_l, size_l)

    pair_corners = []
    for p in _list_images(base / "pair"):
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        found = _detect_two_boards(gray, SPEC)
        if found is None:
            continue
        pair_corners.append((found[0].corners.reshape(-1, 2).astype(np.float64),
                             found[1].corners.reshape(-1, 2).astype(np.float64)))
    if pair_corners:
        # T_AB 用同框图直接算（只需一张，作为链乘输入；B1 用全部角点做残差）。
        T1, _ = _pnp_reproj_rms(objp, pair_corners[0][0], mono_l["K"], mono_l["D"])
        T2, _ = _pnp_reproj_rms(objp, pair_corners[0][1], mono_l["K"], mono_l["D"])
        Tab = rp.board_to_board_from_single_camera(T1, T2)
    else:
        Tab = np.eye(4)

    # 产物只写临时目录，跑完删除；实际不落盘，这里只为守住"绝不覆盖 config"。
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        assert out.is_dir()
        res = calibrate_stereo(
            objp, c_l, c_b, size_l, CalibConfig(method="chain"), SPEC,
            stereo_mode="angled", T_camL_A=T_cl_A, T_AB=Tab, T_camR_B=T_cr_B,
            T_AB_samples=[Tab, Tab.copy()], pair_board_corners=pair_corners)

    refine = res.cross_path_refine or {}
    print(f"  真机角点：左 {len(c_l)} / 右 {len(c_b)} / 同框 {len(pair_corners)} 帧")
    print(f"  chain 重投影 RMS（同框图角点级）："
          f"{refine.get('rms_before', float('nan')):.4f} -> "
          f"{refine.get('rms_after', float('nan')):.4f} px")
    print(f"  护栏：{refine.get('applied', '跳过/异常')}  "
          f"改动 {refine.get('delta_rot_deg', float('nan')):.3f}° / "
          f"{refine.get('delta_trans_mm', float('nan')):.2f} mm  "
          f"原因：{str(refine.get('reason', refine.get('skipped', refine.get('error', ''))))[:100]}")
    print(f"  最终 verdict：{res.verdict.splitlines()[0][:120]}")

    checks = [
        ("机制走通：cross_path_refine 有诊断（不是没跑）", bool(refine)),
        ("不崩：R/T 仍是实数矩阵", np.all(np.isfinite(res.R)) and np.all(np.isfinite(res.T))),
        ("chain 的 board_area 也是实数（真机也接上 A3）",
         np.isfinite(res.board_area_l) and np.isfinite(res.board_area_r)),
        ("有明确结论（applied / skipped / error / 护栏原因至少占一样）",
         any(k in refine for k in ("applied", "skipped", "error"))),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def main() -> int:
    print("=" * 70)
    print("标定事实自测（真机日志口径 + 质量差旁路保存/不覆盖正式档案 + 基线锚 + 分级判定"
          " + 世界系 + 交叉验证 + 外参不确定度）")
    print("=" * 70)
    results = []
    for fn in (test_log_summary_is_self_consistent,
               test_mono_ok_but_stereo_explodes,
               test_unusable_result_is_never_saved,
               test_save_backs_up_previous,
               test_baseline_anchor_is_optional_and_therefore_silent,
               test_board_area_and_tiered_verdict,
               test_export_formats,
               test_world_frame_roundtrip,
               test_world_frame_apply,
               test_world_frame_from_board_photo,
               test_visualization_world_frame,
               test_cross_validate_verdicts,
               test_extrinsic_sigma_vs_empirical,
               test_depth_sigma_includes_baseline,
               test_chain_board_area_caps_verdict,
               test_chain_cross_path_least_squares,
               test_chain_cross_path_guardrail_rejects_180,
               test_chain_cross_path_real_frames_readonly):
        results.append(fn())
        print()
    ok = all(results)
    print("判定结果：" + ("全部通过" if ok else "有不达标项"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
