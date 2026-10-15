"""摄像头设备管理：列出设备 / 目视确认角色 / 固定身份 / 标"无关" / 检查漂移。

**索引会漂移，身份不会。** `cv2.VideoCapture(0)` 的 0 由系统枚举顺序决定，
换一次 USB 口就可能从"外接相机"变成"笔记本内置"——启动器里写死 `--left 0`
就会悄悄用错设备、标定白做。这个工具把"哪台是左、哪台是右"记成**设备身份**
（Windows 上取 PnP InstanceId / USB 端口），之后用**角色名**启动即可。

六个子命令：

    python apps/tools/cameras.py list                # 列设备（不开相机，任何机器都能跑）
    python apps/tools/cameras.py list --probe        # 顺带真开一下，报实际分辨率/是否全黑
    python apps/tools/cameras.py identify            # 逐个索引显示画面，让你确认哪个是左/右/无关
    python apps/tools/cameras.py pin left port:Port_#0001.Hub_#0003
    python apps/tools/cameras.py ignore 2            # 把某台标成「无关」（自动挑相机时跳过它）
    python apps/tools/cameras.py unignore --all      # 取消「无关」标记
    python apps/tools/cameras.py check               # 重插相机后比对：有没有漂移/被别的设备占用

`identify` 的按键：窗口里

    l = 记为左相机      r = 记为右相机      n = 标为「无关」（这台不要用）
    m = 记为单目主相机   s = 跳过这台（不改配置）   q = 退出

「左 / 右 / 无关」三者是**互斥**的：把一台相机改标成别的，旧标记会自动摘掉
（一台相机不能既是左相机又是「无关」，否则自动挑相机时两边打架）。
结果写入 `config/cameras.json`（本机相关，不入库）。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
while not (ROOT / "sphere_tracker").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sphere_tracker import camera_registry as cr  # noqa: E402
from apps._window import WindowManager  # noqa: E402

from sphere_tracker.sources import (  # noqa: E402
    CameraSource,
    quiet_repeated_grab_warnings,
)


# --------------------------------------------------------------------------- #
def _first_line(exc: BaseException) -> str:
    return (str(exc).splitlines() or [type(exc).__name__])[0]


def parse_indices(text: str | None, devices: list[cr.DeviceEntry],
                  max_index: int) -> list[int]:
    """`--indices 0,1,2` -> [0,1,2]；不给就优先用枚举到的设备，再补 0..max。"""
    if text:
        out = []
        for piece in text.split(","):
            piece = piece.strip()
            if piece.isdigit():
                out.append(int(piece))
        return out
    if devices:
        return sorted({d.order for d in devices})
    return list(range(max_index + 1))


# --------------------------------------------------------------------------- #
# list
# --------------------------------------------------------------------------- #
def cmd_list(args) -> int:
    devices = cr.list_devices()
    max_index = args.max_index

    probes: dict[int, cr.ProbeResult] = {}
    if args.probe:
        indices = sorted({d.order for d in devices} | set(range(max_index + 1)))
        for i in indices:
            # 用带回退的版本：MSMF 在本机**间歇性**给不出帧（实测一个下午里
            # 好→坏→坏→又好），而 DSHOW 一直正常。没有回退时这里会等 33 秒
            # 然后报"一帧都拿不到"，用户会以为相机坏了。理由详见
            # `camera_registry.probe_index_with_fallback()` 的文档串。
            probes[i] = cr.probe_index_with_fallback(i)

    if args.json:
        payload = {
            "platform": cr.current_platform(),
            "devices": [asdict(d) for d in devices],
            "probes": {str(k): asdict(v) for k, v in probes.items()},
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print(f"平台：{cr.current_platform()}")
    if cr.current_platform() != "windows":
        print("（非 Windows：不做身份识别，功能退化为按索引使用）")
    print("检测到的相机（索引 = 枚举位次，是启发式；请用 identify 目视确认）：")
    print(cr.format_device_table(devices, probes, settings=None, reg=cr.load_registry()))

    if args.probe:
        extra = [i for i in probes if i not in {d.order for d in devices}]
        for i in sorted(extra):
            r = probes[i]
            tail = (f"实测 {r.width}x{r.height}@{r.fps:.1f}" if r.ok
                    else f"实测失败({r.error})")
            print(f"  #{i}  (不在设备枚举里)  {tail}")

    if not devices and not args.probe:
        print("\n提示：加 --probe 可以不管枚举、直接逐个索引试开一次。")
    return 0


# --------------------------------------------------------------------------- #
# identify
# --------------------------------------------------------------------------- #
# 「无关」不是一个"角色"，但它是用户在识别画面时能做的第三种选择，
# 所以和其它三个放在同一张键位表里，由 cmd_identify 分流处理。
IGNORE_TOKEN = "ignore"
_ASSIGN_KEYS = {ord("l"): "left", ord("r"): "right",
                ord("m"): "main", ord("n"): IGNORE_TOKEN}
_KEY_HINT = "l=左相机  r=右相机  n=无关(这台不要用)  m=单目主相机  s=跳过  q=退出"
# 每个索引的预览**实际用上了哪个后端**（`_preview_and_assign` 填）。
# 为什么值得记：标定/检测要用同一个后端才稳，而这个后端不一定等于偏好后端 ——
# 打不开时会回退。不告诉用户，他标定时就得再踩一次同样的坑。
_backend_used: dict[int, str] = {}


def _open_preview(index: int):
    """打开预览源：**偏好后端打不开就换另一个**，并说明用的是哪个。

    ## 为什么不能写死一个后端（2026-10-15 修的真缺陷）

    这里原来是 `CameraSource(index=index, backend=INDEX_BACKEND)`，注释写着
    「**必须固定 MSMF**」，理由是「DSHOW 的枚举与 PnP 错位」。但那条理由
    **只对"不带后端"（auto）成立**；实测本机**两个后端的索引→设备映射一致**
    （idx0=1920x1080 / idx1=640x480，两轮四组读数相同）。

    而写死 MSMF 的后果是**致命的**：本机 **MSMF 对相机 0 读不到任何一帧**
    （`-2147483638`）。用户 2026-10-15 的实测日志：

        [相机 0] ✘ MSMF @ 640x480：读不到画面（已等 10.6s）
          #0 打不开：摄像头 0 用 MSMF 能打开，但**一帧都拿不到**（读 12 次全空）
        ...（相机 1 打开了）
          [right] <- #1 aoni webcam A25（身份已记录）
          [left]  <- #1 aoni webcam A25（身份已记录）   ← 同一台被记成两个角色

    于是 **#0 根本没机会被分配**，而 #1 被先后按了 r 和 l，
    `config/cameras.json` 里 `left` 与 `right` 指向**同一台相机** ——
    标定直接废掉。**识别工具自己把这个坑造了出来。**

    ## 为什么回退是安全的

    回退**不改变"哪个索引是哪台设备"这个前提的可验证性**：画面就在用户眼前，
    标签（`#N 名字 端口`）与画面是同一路流。真要错位，人一眼能看出来 ——
    这正是 `identify` 存在的意义（"目视确认"）。反过来，**打不开**是没法目视确认的。

    ## ⚠️ 2026-10-15 二次修：原来的回退抓不到"**纯黑**"

    第一版只按"能不能打开"回退（`CameraSource` 抛不抛 `RuntimeError`）。
    但本机更常见的坏状态是"**打开成功、每一帧都是纯黑**"，它**不抛异常**。
    实测同一次会话里两个后端的结论正好相反：

        #0（right）  DSHOW: **纯黑** mean=0.0     MSMF: 正常 mean=116.4
        #1（left）   DSHOW: 正常                 MSMF: 正常

    于是"写死 MSMF"和"MSMF 失败才换"都会在某个时刻坏掉。
    现在改用 `camera_registry.pick_backend_for_device()` ——
    它复用 `probe_index`（已经把纯黑判成 `ok=False`），
    **按"真能出图"挑后端**，不再按"能不能打开"。
    """
    from sphere_tracker import camera_registry as _cr
    from sphere_tracker.sources import INDEX_BACKEND_NAME, backend_name

    # `pick_backend_for_device` 自己会把"哪个后端不可用、改用了哪个"打出来，
    # 所以这里不再重复打印 —— **同一个事实只留一个实现**。
    api, probe = _cr.pick_backend_for_device(index, width=640, height=480)
    if api is None:
        raise RuntimeError(probe.error or "打不开")
    used_name = backend_name(api)
    if used_name != INDEX_BACKEND_NAME:
        print(f"  ℹ 标定/检测请相应加 `--backend {used_name.lower()}`", flush=True)
    try:
        src = CameraSource(index=index, width=640, height=480, backend=api)
    except RuntimeError as exc:
        print(f"  #{index} 打不开：{_first_line(exc)}")
        return None
    _backend_used[index] = used_name


def _preview_and_assign(index: int, entry: cr.DeviceEntry, status: str = "") -> str | None:
    """开一台相机显示实时画面，返回用户按下的角色（None=跳过，''=退出）。"""
    label = f"#{index}  {entry.name or '(无名字)'}  端口 {entry.port or '-'}"
    try:
        src, backend_used = _open_preview(index)
    except RuntimeError as exc:
        print(f"  #{index} 打不开：{_first_line(exc)}")
        return None
    _backend_used[index] = backend_used

    win = "相机识别（l=左 r=右 n=无关 m=主 s=跳过 q=退出 / f 全屏）"
    wins = WindowManager()
    # 这窗口画着大字提示与状态，**不主动改尺寸**（靠内容决定大小最稳）
    wins.keep_size(win)
    gui_ok = True
    try:
        import cv2
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    except Exception:                                    # noqa: BLE001
        gui_ok = False

    try:
        if gui_ok:
            print(f"  正在显示 #{index}：{label}")
            if status:
                print(f"  当前状态：{status}（改选会覆盖它）")
            print(f"  {_KEY_HINT}")
            # ⚠️ **相机掉线必须能自己退出来**（2026-10-14 修）。
            # 原来这里 `while True` 里对 `read()` 失败**不做任何处理** ——
            # 相机被别的程序抢走、或者 USB 掉线之后，循环照转，
            # 画面冻在最后一帧、按键也没有反馈，用户看到的就是"卡死了"，
            # 只能去关整个窗口。**这正是"启动器卡死"报告的一个来源。**
            miss = 0
            while True:
                ok, frame, _t = src.read()
                if ok and frame is not None:
                    miss = 0
                    import cv2
                    cv2.putText(frame, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 255), 2, cv2.LINE_AA)
                    if status:
                        cv2.putText(frame, f"now: {status}", (8, 54),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                                    cv2.LINE_AA)
                    try:
                        wins.show(win, frame)
                    except cv2.error:
                        gui_ok = False
                        break
                else:
                    miss += 1
                    if miss == 30:                    # 约 1 秒
                        print(f"  ⚠ #{index} 读不到画面了…（相机被别的程序抢走 / "
                              f"USB 掉线？）再等 1 秒，还不行就跳过这台")
                    if miss >= 60:                    # 约 2 秒
                        print(f"  ✘ #{index} 连续 2 秒读不到画面，跳过这台相机。")
                        print(f"     排查：是不是有别的程序（QQ/会议/浏览器）占着它？"
                              f"拔插一次 USB 往往就好。")
                        return None
                key = cv2.waitKey(1) & 0xFF
                if wins.handle_key(win, key, frame):
                    continue
                if key == ord("q") or key == 27:
                    return ""
                if key in _ASSIGN_KEYS:
                    return _ASSIGN_KEYS[key]
        if not gui_ok:
            # 没有图形界面：打印设备信息，用命令行输入角色名
            print(f"  #{index}：{label}")
            if status:
                print(f"    当前状态：{status}")
            try:
                ans = input(f"  记为哪个？[left/right/main/none(无关)，回车=跳过，q=退出] "
                            ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                return ""
            if ans in ("q", "quit", "exit"):
                return ""
            if ans in ("n", "no", "none", "ignore", "无关", "无"):
                return IGNORE_TOKEN
            return ans or None
        return None
    finally:
        src.release()
        try:
            import cv2
            cv2.destroyWindow(win)
        except Exception:                                # noqa: BLE001
            pass


def _describe_status(reg: cr.Registry, index: int,
                     entry: cr.DeviceEntry) -> str:
    """这台设备当前是什么状态（left / right / main / 无关 / 未设置）。"""
    device = entry if entry.identity or entry.port or entry.name \
        else cr.DeviceEntry(order=index)
    role = cr.role_of_device(reg, device)
    if role:
        return role
    if cr.is_device_ignored(device, reg):
        return "无关"
    return ""


def cmd_identify(args) -> int:
    devices = cr.list_devices()
    by_order = {d.order: d for d in devices}
    indices = parse_indices(args.indices, devices, args.max_index)
    if not indices:
        print("没有可识别的索引。先跑 python apps/tools/cameras.py list 看看设备。")
        return 1

    reg = cr.load_registry()
    print("逐个索引显示画面，请按提示确认哪台是左、哪台是右、哪台「无关」：")
    print(f"（按键：{_KEY_HINT}）\n")

    for idx in indices:
        entry = by_order.get(idx) or cr.DeviceEntry(order=idx)
        status = _describe_status(reg, idx, entry)
        role = _preview_and_assign(idx, entry, status)
        if role == "":                                   # 退出
            break
        if not role:                                     # 跳过：不改配置
            continue

        verified = bool(entry.identity)
        if role == IGNORE_TOKEN:
            # 标「无关」：顺手把它从 left/right/main 里摘掉（mark_ignored 内部做）
            cr.mark_ignored(reg, entry=entry, index=idx, verified=verified)
            where = f"#{idx} {entry.name}" if entry.name else f"#{idx}"
            print(f"  [无关] <- {where}（自动挑相机时会跳过它）")
            continue

        if role not in ("left", "right", "main"):
            print(f"  忽略未知角色「{role}」（只支持 left/right/main/无关）")
            continue
        # 反过来：一旦成为角色，就不再是「无关」
        cr.unmark_ignored(reg, entry=entry, index=idx)
        cr.set_role(reg, role, entry=entry, index=idx, verified=verified)
        if verified:
            print(f"  [{role}] <- #{idx} {entry.name}（身份已记录）")
        else:
            print(f"  [{role}] <- #{idx}（**拿不到设备身份**，只记了索引；"
                  f"重插 USB 后可能漂移，请务必先用 identify 确认）")

    try:
        path = cr.save_registry(reg)
    except OSError as exc:
        print(f"\n[错误] 写配置失败：{exc}")
        return 1
    print(f"\n已写入 {path}")
    print("之后启动标定/检测可直接用角色名：--left left --right right")
    # ⚠️ 角色指向同一台相机 = 配置自相矛盾（2026-10-15 加）。
    # 这次就是这么坏的：相机 #0 用 MSMF 打不开 → 用户只能给 #1 按键 →
    # 先后按了 r 和 l → left 与 right 记成同一台，**而工具一声不响**。
    # 双目拿这个配置去标定必然失败，且失败点离原因很远。
    by_identity: dict[str, list[str]] = {}
    for role, spec in reg.roles.items():
        key = spec.identity or f"idx:{spec.index}"
        by_identity.setdefault(key, []).append(role)
    clash = {k: v for k, v in by_identity.items() if len(v) > 1}
    if clash:
        print("\n⚠ **配置有问题**：以下角色指向的是**同一台**相机 ——")
        for key, roles in clash.items():
            print(f"    {' / '.join(roles)}  ->  {key}")
        print("    双目要求左右是**两台不同的相机**，现在这份配置没法用来标定。")
        print("    原因通常是**有一台没打开**（换后端试试：`--backend dshow`），")
        print("    于是你只能给另一台按键。请重跑 identify，确保两台各按一次。")
    if _backend_used:
        used = "、".join(f"#{i}->{b}" for i, b in sorted(_backend_used.items()))
        print(f"预览实际用的后端：{used}")
        names = {b.lower() for b in _backend_used.values()}
        if len(names) > 1:
            print("⚠ 两台用的后端**不一样**。标定/检测要用同一个后端，"
                  "请显式指定其中一个（建议 `--backend dshow`）。")
    if reg.ignored:
        names = "、".join((s.name or f"#{s.index}") for s in reg.ignored)
        print(f"已标为「无关」的相机（自动挑相机时跳过）：{names}")
    return 0


# --------------------------------------------------------------------------- #
# pin / ignore / unignore 共用的取值解析
# --------------------------------------------------------------------------- #
def _resolve_device_value(
    value: str, devices: list[cr.DeviceEntry]
) -> tuple[cr.DeviceEntry | None, int | None]:
    """`0` / `port:xx` / `id:xx` / `name:xx` -> (设备, 编号)。认不出来返回 (None, None)。

    纯数字即使枚举不到设备也认（返回的 entry 为 None），这样"离线清记录"也能做。
    """
    low = value.lower()
    if value.isdigit():
        idx = int(value)
        return next((d for d in devices if d.order == idx), None), idx
    for prefix, attr in (("port:", "port"), ("id:", "identity"), ("name:", "name")):
        if low.startswith(prefix):
            needle = value[len(prefix):].strip().lower()
            if not needle:
                return None, None
            entry = next((d for d in devices
                          if needle in str(getattr(d, attr)).lower()), None)
            return (entry, entry.order) if entry else (None, None)
    return None, None


_VALUE_HINT = "取值要写成 索引(0) / port:xxx / id:xxx / name:xxx"


# --------------------------------------------------------------------------- #
# pin
# --------------------------------------------------------------------------- #
def cmd_pin(args) -> int:
    role, value = args.role, args.value
    devices = cr.list_devices()
    entry, index = _resolve_device_value(value, devices)

    if index is None:
        print(f"没有匹配 {value} 的相机。\n{_VALUE_HINT}")
        if devices:
            print(f"检测到的相机：\n{cr.format_device_table(devices)}")
        return 1

    reg = cr.load_registry()
    verified = bool(entry and entry.identity)
    old = cr.unmark_ignored(reg, entry=entry, index=index)
    cr.set_role(reg, role, entry=entry, index=index, verified=verified)
    try:
        path = cr.save_registry(reg)
    except OSError as exc:
        print(f"[错误] 写配置失败：{exc}")
        return 1

    where = f"#{index} {entry.name}" if entry else f"#{index}"
    print(f"[{role}] <- {where}  已写入 {path}")
    if old:
        print("（它原来被标为「无关」，已自动摘掉 —— 一台相机不能既是角色又是无关）")
    if not verified:
        print("提醒：这台拿不到设备身份，只记了索引；建议改用 identify 目视确认，"
              "否则重插 USB 后索引可能漂移。")
    return 0


# --------------------------------------------------------------------------- #
# ignore / unignore / ignored ——「无关」清单
# --------------------------------------------------------------------------- #
def cmd_ignore(args) -> int:
    devices = cr.list_devices()
    entry, index = _resolve_device_value(args.value, devices)
    if index is None:
        print(f"没有匹配 {args.value} 的相机。\n{_VALUE_HINT}")
        if devices:
            print(f"检测到的相机：\n{cr.format_device_table(devices)}")
        return 1

    reg = cr.load_registry()
    verified = bool(entry and entry.identity)
    old_role = cr.role_of_device(reg, entry) if entry else None
    cr.mark_ignored(reg, entry=entry, index=index, verified=verified)
    try:
        path = cr.save_registry(reg)
    except OSError as exc:
        print(f"[错误] 写配置失败：{exc}")
        return 1

    where = f"#{index} {entry.name}" if entry else f"#{index}"
    print(f"[无关] <- {where}  已写入 {path}")
    if old_role:
        print(f"（它原来是「{old_role}」，已一并摘掉）")
    print("自动挑相机（--auto-cameras / 默认）以后会跳过它。")
    if not verified:
        print("提醒：这台拿不到设备身份，只记了索引；重插 USB 后标记可能失效。")
    return 0


def cmd_unignore(args) -> int:
    reg = cr.load_registry()
    if args.all:
        n = cr.clear_ignored(reg)
        if not n:
            print("「无关」清单本来就是空的，不用清。")
            return 0
        try:
            path = cr.save_registry(reg)
        except OSError as exc:
            print(f"[错误] 写配置失败：{exc}")
            return 1
        print(f"已清空「无关」清单（{n} 条），写入 {path}")
        return 0

    if not args.value:
        print("要么给一个取值，要么用 --all 清空整份清单。\n" + _VALUE_HINT)
        return 1

    devices = cr.list_devices()
    entry, index = _resolve_device_value(args.value, devices)
    if index is None:
        print(f"没有匹配 {args.value} 的相机。\n{_VALUE_HINT}")
        return 1
    n = cr.unmark_ignored(reg, entry=entry, index=index)
    if not n:
        print(f"#{index} 本来就不在「无关」清单里，没改动。")
        return 0
    try:
        path = cr.save_registry(reg)
    except OSError as exc:
        print(f"[错误] 写配置失败：{exc}")
        return 1
    where = f"#{index} {entry.name}" if entry else f"#{index}"
    print(f"已取消「无关」标记：{where}（{n} 条），写入 {path}")
    print("现在自动挑相机又可以选它了。要固定角色请跑 identify。")
    return 0


def cmd_ignored(args) -> int:
    """打印「无关」清单，并标注每台当前在不在线。"""
    reg = cr.load_registry()
    if not reg.ignored:
        print("「无关」清单是空的 —— 没有任何相机被排除在自动挑选之外。")
        return 0
    devices = cr.list_devices()
    print(f"「无关」清单（{len(reg.ignored)} 台）—— 自动挑相机会跳过它们：")
    for i, spec in enumerate(reg.ignored, 1):
        who = spec.name or spec.identity or spec.port or "(没记名字)"
        # 当前是否在线：用**严格匹配**（不按名字），否则两台同型号会互相认错
        entry = cr.find_ignored_device(spec, devices)
        if entry is not None:
            state = f"在线 #{entry.order}"
        elif not spec.verified and spec.index is not None:
            now = next((d for d in devices if d.order == spec.index), None)
            state = (f"按索引 #{spec.index}（未确认身份，可能是「{now.name}」）"
                     if now is not None else f"离线（原索引 #{spec.index}）")
        else:
            state = "离线（没插 / 换过口）"
        print(f"  {i}. {who:<24} {state}")
    print("\n取消某一台：python apps/tools/cameras.py unignore <编号>")
    print("全部取消：  python apps/tools/cameras.py unignore --all")
    return 0


# --------------------------------------------------------------------------- #
# check
# --------------------------------------------------------------------------- #
def cmd_check(args) -> int:
    reg = cr.load_registry()
    if not reg.roles and not reg.ignored:
        print("还没有固定任何相机（config/cameras.json 为空）。")
        print("第一次使用请先跑：python apps/tools/cameras.py identify")
        return 0

    devices = cr.list_devices()
    by_order = {d.order: d for d in devices}
    if not devices:
        print("（当前枚举不到任何相机设备；非 Windows 或系统枚举不可用 —— 只能按索引用）")

    danger = False
    for role in sorted(reg.roles):
        spec = reg.roles[role]
        tag = "已确认" if spec.verified else "未确认"
        try:
            res = cr.resolve_role(role, reg, devices, interactive=False)
        except RuntimeError as exc:
            print(f"[漂移] 「{role}」({tag}) 解析失败：{_first_line(exc)}")
            continue

        if res.how == "identity":
            status = "一致" if res.index == spec.index else f"漂移 -> 按身份纠正到 #{res.index}"
        elif res.how == "port":
            status = "一致（按端口）" if res.index == spec.index else f"漂移 -> 按端口纠正到 #{res.index}"
        elif res.how == "index-fallback":
            status = f"索引兜底 #{res.index}（未确认身份，可能用错）"
        else:
            status = f"手动指定 #{res.index}"
        print(f"[{'一致' if res.index == spec.index else '漂移'}] 「{role}」({tag}) {status}")

        for w in res.warnings:
            print(f"        [警告] {w}")

        # 危险：配置里记的那个索引，现在正被**另一台**设备占着
        if spec.identity and spec.index is not None:
            now = by_order.get(spec.index)
            if now is not None and now.identity and now.identity.upper() != spec.identity.upper():
                danger = True
                print(f"        [危险] 索引 #{spec.index} 现在指向的是「{now.name or now.identity}」，"
                      f"不是「{role}」原来那台！若用死索引启动就会用错相机。")
        # 危险：身份对上但设备名不符（同型号换了一台 / 记录过期）
        if spec.name and res.entry is not None and res.entry.name \
                and spec.name.lower() not in res.entry.name.lower():
            danger = True
            print(f"        [危险] 期望设备名「{spec.name}」，实际是「{res.entry.name}」。")

    # 「无关」清单：正常是信息，但"角色和无关指向同一台"属于配置自相矛盾
    if reg.ignored:
        print()
        for spec in reg.ignored:
            who = spec.name or spec.identity or spec.port or "(没记名字)"
            entry = cr.find_ignored_device(spec, devices)
            if entry is not None:
                print(f"[无关] 「{who}」在线 #{entry.order}（自动挑相机会跳过它）")
                clash = cr.role_of_device(reg, entry)
                if clash:
                    danger = True
                    print(f"        [危险] 它同时又是角色「{clash}」—— 配置自相矛盾，"
                          f"请重跑 identify 重新确认。")
            else:
                print(f"[无关] 「{who}」不在线（没插 / 换过口）—— 记录保留，插回来仍然生效")

    if danger:
        print("\n存在 [危险] 项：请重跑 python apps/tools/cameras.py identify 重新确认。")
        return 2
    print("\n未发现危险项。若要每次开机都安心，可把这条命令放进启动脚本。")
    return 0


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="摄像头设备管理（列表 / 确认 / 固定 / 标无关 / 检查）",
        epilog="「无关」= 明确标记这台不要用；自动挑相机时会跳过它。",
    )
    sub = p.add_subparsers(dest="cmd")

    pl = sub.add_parser("list", help="列出摄像头设备（默认不开相机）")
    pl.add_argument("--probe", action="store_true", help="真的开一下，报实际分辨率/是否全黑")
    pl.add_argument("--json", action="store_true", help="以 JSON 输出（供脚本消费）")
    pl.add_argument("--max-index", type=int, default=6, help="探测的最大索引（默认 6）")
    pl.set_defaults(func=cmd_list)

    pi = sub.add_parser("identify", help="逐个索引显示画面，确认哪个是左/右/无关")
    pi.add_argument("--indices", default=None, help="只识别这些索引，如 0,2")
    pi.add_argument("--max-index", type=int, default=6, help="没给 --indices 时补探到第几个索引")
    pi.set_defaults(func=cmd_identify)

    pp = sub.add_parser("pin", help="非交互地把某角色固定到某设备")
    pp.add_argument("role", help="角色名，如 left / right / main")
    pp.add_argument("value", help="0 | port:Port_#0001.Hub_#0003 | id:USB\\... | name:aoni")
    pp.set_defaults(func=cmd_pin)

    pig = sub.add_parser("ignore", help="把某台相机标为「无关」（自动挑相机时跳过）")
    pig.add_argument("value", help="0 | port:Port_#0001.Hub_#0003 | id:USB\\... | name:aoni")
    pig.set_defaults(func=cmd_ignore)

    pu = sub.add_parser("unignore", help="取消「无关」标记（单台，或 --all 清空整份清单）")
    pu.add_argument("value", nargs="?", default=None, help="同 ignore 的取值；与 --all 二选一")
    pu.add_argument("--all", action="store_true", help="清空整份「无关」清单")
    pu.set_defaults(func=cmd_unignore)

    pg = sub.add_parser("ignored", help="列出当前「无关」清单（并标注在不在线）")
    pg.set_defaults(func=cmd_ignored)

    pc = sub.add_parser("check", help="比对配置与当前设备：有没有漂移/被占用")
    pc.set_defaults(func=cmd_check)
    return p


def main() -> int:
    quiet_repeated_grab_warnings()
    p = build_parser()
    args = p.parse_args()
    if not getattr(args, "cmd", None):
        p.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
