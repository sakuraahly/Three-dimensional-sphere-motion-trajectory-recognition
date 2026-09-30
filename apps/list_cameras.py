"""列出本机可用摄像头：分辨率、帧率、画面是否正常，以及对应的设备名。

换相机/换机器时先跑这个，避免"以为坏了其实是别的问题"：

    python apps/list_cameras.py
    python apps/list_cameras.py --max-index 6 --probe-seconds 2

它会逐个打开 DirectShow 索引 0..N，报告实际拿到的分辨率与实测帧率，
并检查画面是不是"全黑常量流"（真实传感器一定有噪声，
整帧标准差恰好为 0 说明这条流是假的 —— 红外摄像头、隐私挡板、
或者不支持的分辨率都会这样）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from sphere_tracker.sources import CameraSource


def device_names_in_order() -> list[str]:
    r"""按 DirectShow 枚举顺序猜出设备名。

    做法：读注册表里"视频捕获设备"类（{e5323777-...}）的子键顺序 ——
    这个顺序通常与 DirectShow 的索引顺序一致；再从子键名里取出
    USBVID_xxxx&PID_xxxx，去 EnumUSB 下查 FriendlyName。
    读不到就返回空列表，不影响主流程。
    """
    try:
        import winreg  # type: ignore
    except ImportError:
        return []

    class_key = r"SYSTEM\CurrentControlSet\Control\DeviceClasses\{e5323777-f976-4f5b-9b55-b94699c46e44}"
    names: list[str] = []
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, class_key) as key:
            count = winreg.QueryInfoKey(key)[0]
            for i in range(count):
                instance = winreg.EnumKey(key, i)
                parts = [p for p in instance.replace("#", "\\").split("\\") if p]
                vid_pid = next((p for p in parts if p.upper().startswith("VID_")), "")
                name = ""
                if vid_pid:
                    # 设备实例形如 USB\VID_1BCF&PID_2A25&MI_00\6&31324e2c&0&0000
                    # Enum\USB 下的键名就是中间那段，再去它的子键里读 FriendlyName
                    try:
                        usb_key = rf"SYSTEM\CurrentControlSet\Enum\USB\{vid_pid}"
                        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, usb_key) as uk:
                            for j in range(winreg.QueryInfoKey(uk)[0]):
                                sub = winreg.EnumKey(uk, j)
                                try:
                                    with winreg.OpenKey(uk, sub) as sk:
                                        name = str(winreg.QueryValueEx(sk, "FriendlyName")[0])
                                        break
                                except OSError:
                                    continue
                    except OSError:
                        name = ""
                names.append(name or f"({vid_pid or 'unknown'})")
    except OSError:
        return []
    return names


def main() -> int:
    p = argparse.ArgumentParser(description="列出本机可用摄像头")
    p.add_argument("--max-index", type=int, default=5, help="最多探测到第几个索引")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fps", type=int, default=60)
    args = p.parse_args()

    names = device_names_in_order()
    print("DirectShow 索引 -> 设备名（顺序通常一致，供参考）：")
    if names:
        for i, n in enumerate(names):
            print(f"  {i}: {n}")
    else:
        print("  （读不到注册表，跳过；可在设备管理器 > 摄像头 里查看）")
    print()

    print(f"{'索引':<5}{'分辨率':<12}{'实测帧率':<10}{'亮度':>8}{'对比度':>9}  判定")
    print("-" * 62)
    usable: list[int] = []
    for idx in range(args.max_index + 1):
        try:
            src = CameraSource(index=idx, width=args.width, height=args.height, fps=args.fps)
        except RuntimeError:
            print(f"{idx:<5}{'-':<12}{'-':<10}{'-':>8}{'-':>9}  打不开（无此设备或已被占用）")
            continue
        best = None
        for _ in range(20):
            ok, frame, _t = src.read()
            if ok and frame is not None and (best is None or float(frame.std()) > float(best.std())):
                best = frame
        info = src.info
        src.release()
        if best is None:
            print(f"{idx:<5}{info.width}x{info.height:<7}{info.fps:<10.1f}{'-':>8}{'-':>9}  读不到帧")
            continue
        std = float(best.std())
        mean = float(best.mean())
        verdict = "可用" if std > 1.0 else "**画面是全黑常量流，不可用**"
        if std > 1.0:
            usable.append(idx)
        print(f"{idx:<5}{info.width}x{info.height:<7}{info.fps:<10.1f}{mean:8.1f}{std:9.2f}  {verdict}")

    print()
    if usable:
        print(f"可用索引：{usable}")
        print(f"用法示例：python apps/run_realtime.py --camera {usable[0]} --method fusion")
        if len(usable) >= 2:
            print(f"双目可用这前两路（但要先把它们对准同一区域并刚性固定）：{usable[:2]}")
    else:
        print("没有可用摄像头。检查：是否被别的程序占用、镜头隐私挡板、USB 连接。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
