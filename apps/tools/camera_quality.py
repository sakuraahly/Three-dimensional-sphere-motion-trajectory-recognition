"""相机画质 / 链路诊断：到底是**带宽**问题，还是**光学/曝光**问题？

为什么需要这个工具
------------------
"有一路画面明显不如另一路"是最常见、也最容易查错方向的问题。
USB 带宽不足和镜头脏了/逆光/跑焦，在肉眼看来都可以叫"不清晰"，
但**处理方式完全相反**：前者要换线换口，后者要擦镜头/改曝光。
猜错方向就会在硬件上白折腾半天。

这个工具把两者分开，靠的是**它们各自特有的、可测量的指纹**：

| 症状 | 带宽/链路不足 | 光学/曝光问题 |
| ---- | ------------- | ------------- |
| 实测帧率 | 明显低于目标，或两路同开时骤降 | 正常 |
| 残帧（下半幅常量） | **会出现**（传输没传完） | 不会 |
| 实际分辨率 | 被驱动降到更低档 | 正常 |
| 清晰度（Laplacian 方差） | 正常 | **明显偏低** |
| 过曝/死黑比例 | 正常 | 可能明显偏高 |
| 画面反差分 | 正常 | 可能极低（蒙雾） |

判据：**帧率与残帧都没有异常时，带宽就不是原因** —— 带宽问题不会让画面变"糊"，
它只会让画面变"卡"、掉帧、或者下半幅是灰的。

用法::

    python apps/tools/camera_quality.py --left left --right right   # 比两路（推荐）
    python apps/tools/camera_quality.py --auto-cameras
    python apps/tools/camera_quality.py --indices 0,1,2,3           # 逐台看
    python apps/tools/camera_quality.py --left left --right right --together
                                        # 额外测"两路同开"，看带宽够不够

`--together` 为什么重要：单开一台测的是**这台相机的上限**；
两路同开才暴露**共用一条 USB 总线**时的争抢。两者差很多 = 带宽问题。
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
while not (ROOT / "sphere_tracker").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "apps"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from _common import add_stereo_source_args  # noqa: E402
from sphere_tracker import camera_registry as cr  # noqa: E402
# 纯计算部分在库层：`stereo/sessions.py` 也要用同一套口径，库不能反向 import 本文件。
# 这里**重新导出**这两个名字，老用法 `from apps.tools.camera_quality import frames_quality` 照旧可用。
from sphere_tracker.quality import FrameMetrics, frames_quality  # noqa: E402,F401
from sphere_tracker.sources import BLACK_FRAME_STD, CameraSource  # noqa: E402


# --------------------------------------------------------------------------- #
@dataclass
class Quality:
    index: int
    name: str = ""
    req_w: int = 0
    req_h: int = 0
    act_w: int = 0
    act_h: int = 0
    fps: float = 0.0
    frames: int = 0
    partial: int = 0          # 残帧数（下半幅是常量 = 传输没传完）
    sharp: float = 0.0        # 清晰度（Laplacian 方差，越大越锐）
    bright: float = 0.0
    contrast: float = 0.0
    over: float = 0.0         # 过曝像素比例
    under: float = 0.0        # 死黑像素比例
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


def _is_partial(gray: np.ndarray) -> bool:
    """下半幅是不是"常量"？—— 典型的 USB 传输没传完 / 驱动给了半帧。

    判据要稳，不能把"本来就是纯色背景"的画面误判：要求
    **上半幅有明显内容**（标准差够大）而**最下面 1/5 几乎没有内容**。
    """
    h = gray.shape[0]
    if h < 20:
        return False
    top = gray[: h // 2]
    bottom = gray[int(h * 0.8):]
    return float(top.std()) > 6.0 and float(bottom.std()) < 1.0


def measure(index: int, *, seconds: float = 2.0, width: int = 1920,
            height: int = 1080, fps: int = 60, backend: str | None = None,
            warmup: int = 30) -> Quality:
    """开一台相机，量一段时间，返回画质与链路指标。"""
    devices = {d.order: d for d in cr.list_devices()}
    q = Quality(index=index, name=(devices.get(index).name if index in devices else ""),
                req_w=width, req_h=height)
    try:
        # 首帧等待显式给 30 s，不能沿用库默认 3.0（sources.py:388 那是给快设备的）。
        # 本机 aoni 相机首帧约 23 s（AGENTS.md:386-391）；相机链路的实测下限是
        # apps/tools/cameras.py:431 的 PREVIEW_FIRST_FRAME_S = 30.0，
        # apps/record_stereo.py:156、sphere_tracker/camera_registry.py:491 也都显式传 30。
        # 画质诊断工具若沿用 3.0，`--indices N` 单台试开会把慢相机误判成「打不开」。
        src = CameraSource(index=index, width=width, height=height, fps=fps,
                           backend=_backend_const(backend),
                           first_frame_timeout_s=30.0)
    except RuntimeError as exc:
        q.error = (str(exc).splitlines() or [type(exc).__name__])[0]
        return q
    except Exception as exc:                                  # noqa: BLE001
        q.error = f"{type(exc).__name__}: {exc}"
        return q

    try:
        q.act_w = int(src.info.width)
        q.act_h = int(src.info.height)

        # 预热：真实传感器一定有噪声；整帧标准差恰好为 0 说明这条流是假的
        # （外置摄像头开机后前几帧常常是全黑的，别把它算进指标）
        t_end = time.perf_counter() + max(1.0, warmup / 10.0)
        got = 0
        while time.perf_counter() < t_end:
            ok, frame, _t = src.read()
            if not ok or frame is None:
                continue
            if float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).std()) > BLACK_FRAME_STD:
                got += 1
                if got >= warmup:
                    break
        if got == 0:
            q.error = "读不到有效画面（一直全黑或读不到帧）"
            return q

        t0 = time.perf_counter()
        last = t0
        dts: list[float] = []
        shots: list[np.ndarray] = []
        while time.perf_counter() - t0 < seconds:
            ok, frame, _t = src.read()
            if not ok or frame is None:
                continue
            now = time.perf_counter()
            dts.append(now - last)
            last = now
            q.frames += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if _is_partial(gray):
                q.partial += 1
            # 最多留 8 张算画质，别把内存吃满
            if len(shots) < 8:
                shots.append(gray)
        elapsed = max(1e-6, time.perf_counter() - t0)
        q.fps = q.frames / elapsed
    finally:
        src.release()

    if not shots:
        q.error = "没取到可用于评画质的帧"
        return q

    # 画质指标：取中景（避开边缘暗角与贴边物体）
    m = frames_quality(shots)
    q.sharp, q.bright, q.contrast, q.over, q.under = (
        m.sharp, m.bright, m.contrast, m.over, m.under)
    return q


def _backend_const(name: str | None):
    """把 CLI 的后端名翻成 cv2 常量。

    ``auto`` / 没给 = **MSMF**，不是"交给 CameraSource 自己挑"。

    为什么不能 auto：本工具用 `list_devices()` 的 PnP 位次当编号，还把这个位次对应的
    **设备名字**打在报告里。若用 auto 挑到 DSHOW，编号就指向另一台相机 ——
    报告会写着"内置相机 清晰度 812"，而那其实是外接相机的数据。
    完整实测对照见 sphere_tracker/sources.py 的 INDEX_BACKEND 注释。
    """
    if not name or name == "auto":
        from sphere_tracker.sources import INDEX_BACKEND

        return INDEX_BACKEND
    table = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}
    return table.get(name.lower())


# --------------------------------------------------------------------------- #
def print_one(q: Quality) -> None:
    who = f"#{q.index}"
    if q.name:
        who += f"  {q.name}"
    print(f"  {who}")
    if not q.ok:
        print(f"      打不开：{q.error}")
        print()
        return
    res = f"请求 {q.req_w}x{q.req_h} -> 实际 {q.act_w}x{q.act_h}"
    if (q.act_w, q.act_h) != (q.req_w, q.req_h):
        res += "   ← 被驱动换了档（坑 4：请求≠实际）"
    print(f"      分辨率     {res}")
    fps_note = ""
    if q.fps < 20:
        fps_note = "   ← 偏低！可能带宽/后端/占用"
    print(f"      实测帧率   {q.fps:5.1f} fps{fps_note}")
    part = f"{q.partial}"
    if q.partial:
        part += "   ← 有残帧！这是**传输**问题（带宽/线材/口）"
    print(f"      残帧       {part}")
    print(f"      清晰度     {q.sharp:7.1f}   （Laplacian 方差，越大越锐）")
    print(f"      亮度       {q.bright:6.1f}   过曝 {q.over * 100:5.1f}%   死黑 {q.under * 100:5.1f}%")
    print(f"      反差       {q.contrast:6.1f}   （低到个位数 = 画面像蒙了雾）")
    print()


def verdict(pair: list[Quality]) -> None:
    """两路对比 + 结论。这是本工具存在的意义：把'糊'和'卡'分开。"""
    good = [q for q in pair if q.ok]
    print("=" * 66)
    print("对比结论")
    print("=" * 66)
    if len(good) < 2:
        print("  只测到一路可用，没法对比。先解决打不开的那一路（见上面的报错）。")
        return

    link_bad = [q for q in good if q.partial > 0 or q.fps < 20]
    if link_bad:
        who = "、".join(f"#{q.index}" for q in link_bad)
        print(f"  ⚠ 链路（带宽/线材/口）有问题：{who}")
        print(f"     依据：残帧 > 0 或实测帧率 < 20 fps。")
        print(f"     → 先按下面「交换测试」定位是相机、线、还是口；")
        print(f"       两台同型号相机别插在同一个集线器上（会共用同一条总线）。")
        print()

    sharp = [q for q in good if q.sharp > 0]
    if len(sharp) == 2:
        a, b = sharp
        hi, lo = (a, b) if a.sharp >= b.sharp else (b, a)
        ratio = hi.sharp / max(lo.sharp, 1e-6)
        print(f"  清晰度：# {hi.index} = {hi.sharp:.1f}，# {lo.index} = {lo.sharp:.1f}"
              f"（相差 {ratio:.1f} 倍）")
        if ratio >= 1.8:
            print(f"     → 这是**光学 / 曝光**差异，**不是带宽**：")
            print(f"       带宽不足不会让画面变糊，只会让它变卡、掉帧、下半幅发灰。")
            print(f"       而这两路的帧率/残帧都没有异常，说明数据量和链路都是好的。")
            print(f"       按可能性从高到低查 # {lo.index}：")
            print(f"         1) 镜头上有没有**保护膜没撕**（新摄像头极常见）")
            print(f"         2) 镜头**脏了/有指纹**（用眼镜布擦，别用衣角）")
            print(f"         3) **结露/起雾**：从冷处拿到暖湿房间会这样，放十几分钟再看")
            print(f"         4) **跑焦**：对准远处有细节的东西，看能不能变清楚")
            print(f"         5) **逆光/正对强光**：画面大片过曝会把对比拉没，转个方向")
            print(f"         6) 镜头前有没有东西紧贴（窗帘、塑料、纸板、外壳）")
        elif ratio >= 1.3:
            print(f"     → 有点差距（{ratio:.1f} 倍），属于同型号之间的正常个体差异；")
            print(f"       如果标定收不到帧再回头查上面那 6 条。")
        else:
            print(f"     → 两路清晰度相当，画质不是问题。")

    over = [q for q in good if q.over > 0.15]
    if over:
        who = "、".join(f"#{q.index}（过曝 {q.over * 100:.0f}%）" for q in over)
        print(f"  曝光：{who}")
        print(f"     → 大片过曝会把棋盘对比度吃掉，findChessboardCorners 就容易失败。")
        print(f"       让两路**朝向一致**、避开窗口/顶灯直射，或降低曝光（别用自动曝光）。")

    if not link_bad and not (len(sharp) == 2 and sharp[0].sharp / max(sharp[1].sharp, 1e-6) >= 1.8
                             or sharp[1].sharp / max(sharp[0].sharp, 1e-6) >= 1.8):
        print("  → 两路的帧率、残帧、清晰度都没有明显差异。")
        print("    如果主观上还是觉得某一路差，多半是**取景/光照**不同（比如一路对着窗户），")
        print("    而不是设备或链路问题。")

    print()
    print("  交换测试（唯一能定性归因的办法，5 分钟）：")
    print("    把两台相机的 **USB 口对调**（最好连数据线也一起对调），再跑一次本工具。")
    print("      · 问题**跟着相机走** → 是相机本身（光学/传感器）")
    print("      · 问题**留在原来的口/线** → 是链路（线材/集线器/带宽）")
    print("    然后把两台分别插到**不同的 USB 控制器**上（不要走同一个扩展坞），再测一次。")


# --------------------------------------------------------------------------- #
def resolve_targets(args) -> list[tuple[int, str]]:
    """要测哪些相机？返回 [(索引, 角色名)]。"""
    out: list[tuple[int, str]] = []
    for role, value in (("left", args.left), ("right", args.right)):
        if not value:
            continue
        try:
            idx = cr.resolve_camera_arg(value, interactive=False)
        except RuntimeError as exc:
            print(f"[警告] 解析 {role}={value} 失败：{(str(exc).splitlines() or [''])[0]}")
            continue
        if idx is not None:
            out.append((idx, role))
    if out:
        return out
    if args.indices:
        return [(int(i), "") for i in args.indices.split(",") if i.strip().isdigit()]
    # 兜底：自动挑两台
    picks = cr.select_cameras(need=2)
    return [(p.index, "") for p in picks]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="相机画质 / 链路诊断（分清'带宽不足'与'光学/曝光问题'）")
    add_stereo_source_args(ap)
    ap.add_argument("--indices", default=None, help="只测这些索引，如 0,1,2,3")
    ap.add_argument("--seconds", type=float, default=2.0, help="每台测多少秒（默认 2.0）")
    ap.add_argument("--together", action="store_true",
                    help="额外测一次'两路同开'，看共用总线时会不会互相抢带宽")
    args = ap.parse_args()

    targets = resolve_targets(args)
    if not targets:
        print("[错误] 没有可测的相机。先跑：python apps/tools/cameras.py list --probe")
        return 1

    print("=" * 66)
    print("相机画质 / 链路诊断")
    print("=" * 66)
    print(f"每台测 {args.seconds:.1f} 秒；**一次只开一台**，测的是这台自己的上限。")
    print()
    results: list[Quality] = []
    for idx, role in targets:
        tag = f"（{role}）" if role else ""
        print(f"----- #{idx}{tag} -----")
        q = measure(idx, seconds=args.seconds, width=args.width,
                    height=args.height, fps=args.fps, backend=args.backend)
        results.append(q)
        print_one(q)

    if len(results) >= 2:
        verdict(results)

    if args.together and len(targets) >= 2:
        print("=" * 66)
        print("两路同开（暴露共用 USB 总线的争抢）")
        print("=" * 66)
        both = []
        for idx, role in targets[:2]:
            q = measure(idx, seconds=args.seconds, width=args.width,
                        height=args.height, fps=args.fps, backend=args.backend)
            both.append(q)
        for q in both:
            print_one(q)
        solo = {q.index: q.fps for q in results}
        print("  单开 vs 同开：")
        starved = []
        for q in both:
            s = solo.get(q.index, 0.0)
            drop = (1 - q.fps / s) * 100 if s > 1 else 0.0
            flag = ""
            if s > 1 and drop > 25:
                flag = "   ← 明显掉速！总线被抢"
                starved.append(q.index)
            print(f"    #{q.index}  单开 {s:5.1f} fps  ->  同开 {q.fps:5.1f} fps"
                  f"（降 {drop:4.0f}%）{flag}")
        print()
        if starved:
            print("  → 两路同开会互相拖慢 = **带宽/总线**确实紧张。")
            print("    解决：把两台分别插到**不同的 USB 控制器**上（机身不同的口，")
            print("    不要都走同一个扩展坞/集线器）。")
            print("    ⚠️ USB 2.0 的相机插蓝口（3.x）也不会变快，只是**分总线**而已。")
        else:
            print("  → 同开没有明显掉速，**带宽不是问题**。")
    elif args.together:
        print("（只识别到一路，跳过'两路同开'测试）")

    print()
    print("提示：想了解每台的协商细节（后端/请求值/实际值），")
    print("      跑 python apps/tools/cameras.py list --probe")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
