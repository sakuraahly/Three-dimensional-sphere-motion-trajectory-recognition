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
from sphere_tracker.sources import CameraSource  # noqa: E402


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
            probes[i] = cr.probe_index(i)

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


def _preview_and_assign(index: int, entry: cr.DeviceEntry, status: str = "") -> str | None:
    """开一台相机显示实时画面，返回用户按下的角色（None=跳过，''=退出）。"""
    label = f"#{index}  {entry.name or '(无名字)'}  端口 {entry.port or '-'}"
    try:
        # **必须固定 MSMF**：`entry` 是按 PnP 枚举行的名字，而 `index` 交给 cv2 之后
        # 走的是**后端自己的枚举**。DSHOW 的枚举与 PnP 错位（实测 DSHOW #1 是笔记本
        # 内置、#2 才是外接），于是会出现"画面是 A、标签写着 B"——
        # 人看着 A 的画面按 r，配置却记到了 B 上。MSMF 的枚举与 PnP 一致，标签才是真的。
        # 见 sphere_tracker/sources.py 的 INDEX_BACKEND 注释（那里有完整实测表）。
        from sphere_tracker.sources import INDEX_BACKEND

        src = CameraSource(index=index, width=640, height=480, backend=INDEX_BACKEND)
    except RuntimeError as exc:
        print(f"  #{index} 打不开：{_first_line(exc)}")
        return None

    win = "相机识别（l=左 r=右 n=无关 m=主 s=跳过 q=退出）"
    gui_ok = True
    try:
        import cv2
        try:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        except cv2.error:
            gui_ok = False
    except Exception:                                    # noqa: BLE001
        gui_ok = False

    try:
        if gui_ok:
            print(f"  正在显示 #{index}：{label}")
            if status:
                print(f"  当前状态：{status}（改选会覆盖它）")
            print(f"  {_KEY_HINT}")
            while True:
                ok, frame, _t = src.read()
                if ok and frame is not None:
                    import cv2
                    cv2.putText(frame, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 255), 2, cv2.LINE_AA)
                    if status:
                        cv2.putText(frame, f"now: {status}", (8, 54),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                                    cv2.LINE_AA)
                    try:
                        cv2.imshow(win, frame)
                    except cv2.error:
                        gui_ok = False
                        break
                key = cv2.waitKey(1) & 0xFF
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
    p = build_parser()
    args = p.parse_args()
    if not getattr(args, "cmd", None):
        p.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
