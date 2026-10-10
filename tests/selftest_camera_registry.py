"""摄像头设备管理自测：身份解析 / 端口识别 / 自动选择。

为什么必须有这个测试
--------------------
这个模块的全部风险都藏在**"配置和实物对不上"**这条路径上：身份匹配、端口匹配、
索引兜底、以及"绝不静默用错相机"这条安全底线。这些逻辑**不会在正常使用中暴露**
—— 相机插对了，一切照跑；插错了（换 USB 口、幽灵记录、索引漂移），
程序可能**不报任何错**，只是把左手外接和右手内置配成一对，标定白做。

所以：**全程不打开任何相机**，把系统枚举（注册表 / PowerShell / 探测）全部注入假数据，
在任何机器上都能稳定复现这十条用例。

    python tests/selftest_camera_registry.py            # 无相机即可通过
    python tests/selftest_camera_registry.py --probe    # 有相机时额外验证真机枚举
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
while not (ROOT / "sphere_tracker").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sphere_tracker.camera_registry import (  # noqa: E402
    DeviceEntry,
    ProbeResult,
    Registry,
    RoleSpec,
    _parse_pnputil,
    format_device_table,
    load_registry,
    parse_windows_instance,
    rank_candidates,
    resolve_camera_arg,
    resolve_role,
    save_registry,
    select_cameras,
    set_role,
    windows_devices,
)

# 三台典型设备：外接 A（USB\A）、外接 B（USB\B）、笔记本内置（USB\INT）
MODULE_GUID = "{e5323777-f976-4f5b-9b55-b94699c46e44}"


def dev(order, name="", identity="", port="", present=True, source="powershell"):
    return DeviceEntry(order=order, name=name, identity=identity, port=port,
                       present=present, source=source)


def sample_devices():
    """一台内置 + 两台同型号外接（不同端口）—— 最容易配错的那种组合。"""
    return [
        dev(0, "Integrated Camera", "USB\\INT", "", present=True),
        dev(1, "aoni webcam A25", "USB\\A", "Port_#0001.Hub_#0003"),
        dev(2, "aoni webcam A25", "USB\\B", "Port_#0002.Hub_#0003"),
    ]


def role(identity="", port="", name="", index=None, verified=False):
    return RoleSpec(identity=identity, port=port, name=name, index=index,
                    verified=verified)


def registry_with(left: RoleSpec | None = None, right: RoleSpec | None = None,
                  settings: dict | None = None) -> Registry:
    roles = {}
    if left is not None:
        roles["left"] = left
    if right is not None:
        roles["right"] = right
    return Registry(roles=roles, settings=settings or {})


# --------------------------------------------------------------------------- #
# 1. InstanceId 解析
# --------------------------------------------------------------------------- #
def test_parse_instance() -> bool:
    print("----- 1. PnP InstanceId 解析（身份 / 端口） -----")
    checks: list[tuple[str, bool]] = []

    a = "USB\\VID_1BCF&PID_2A25&MI_00\\6&31324E2C&0&0000"
    ident, port = parse_windows_instance(a)
    checks.append(("反斜杠形式：identity 原样", ident == a))
    checks.append(("反斜杠形式：端口段为最后一节", port == "6&31324E2C&0&0000"))

    b = f"##?#USB#VID_1BCF&PID_2A25&MI_00#6&31324e2c&0&0000#{MODULE_GUID}"
    ident, port = parse_windows_instance(b)
    checks.append(("井号形式去掉 {类 GUID}", ident ==
                   "USB\\VID_1BCF&PID_2A25&MI_00\\6&31324e2c&0&0000"))
    checks.append(("井号形式端口段正确", port == "6&31324e2c&0&0000"))

    c = "USB\\VID_1BCF&PID_2A25\\6&31324E2C&0&0000"
    ident, port = parse_windows_instance(c)
    checks.append(("不带 &MI_00 也能解析", ident == c and port == "6&31324E2C&0&0000"))

    checks.append(("空串 -> ('', '')", parse_windows_instance("") == ("", "")))

    ok = True
    for name, passed in checks:
        print(f"  [{'通过' if passed else '不达标'}] {name}")
        ok = ok and passed
    return ok


# --------------------------------------------------------------------------- #
# 1b. pnputil 输出解析（按位置，不认标签 —— 抗非英文 Windows）
# --------------------------------------------------------------------------- #
def test_pnputil_parse() -> bool:
    print("----- 1b. pnputil 输出解析（字段按位置，不认标签） -----")
    # 故意用"本地化"的字段名，证明解析只依赖位置、不依赖英文
    text = (
        "\nMicrosoft PnP Utility\n\n"
        "实例 ID:                    USB\\VID_1BCF&PID_2A25&MI_00\\6&31324e2c&0&0000\n"
        "设备描述:                   aoni webcam A25\n"
        "类名:                       Camera\n"
        "状态:                       已启动\n\n"
        "实例 ID:                    USB\\VID_30C9&PID_00EC&MI_00\\6&3a5c8ec3&1&0000\n"
        "设备描述:                   Integrated Camera\n"
        "类名:                       Camera\n"
        "状态:                       已启动\n"
    )
    parsed = _parse_pnputil(text)
    checks = [
        ("解析出 2 台设备", len(parsed) == 2),
        ("第 1 台身份正确", parsed[0]["InstanceId"]
         == "USB\\VID_1BCF&PID_2A25&MI_00\\6&31324e2c&0&0000"),
        ("第 1 台名字正确（按第 2 个字段取）", parsed[0]["FriendlyName"] == "aoni webcam A25"),
        ("第 2 台名字正确", parsed[1]["FriendlyName"] == "Integrated Camera"),
        ("connected 列表里都算在线", all(p["Present"] for p in parsed)),
    ]
    ok = True
    for name, passed in checks:
        print(f"  [{'通过' if passed else '不达标'}] {name}")
        ok = ok and passed
    return ok


# --------------------------------------------------------------------------- #
# 2. 注册表 + PnP 合并、幽灵过滤、order 连续
# --------------------------------------------------------------------------- #
def test_windows_devices_merge() -> bool:
    print("----- 2. windows_devices 合并 / 幽灵记录过滤 -----")
    reg = ["USB\\A\\p0", "USB\\B\\p1", "USB\\C\\p2"]
    pnp = [
        {"InstanceId": "USB\\A\\p0", "FriendlyName": "aoni webcam A25",
         "Present": True, "LocationInformation": "Port_#0001.Hub_#0003"},
        {"InstanceId": "USB\\B\\p1", "FriendlyName": "幽灵相机",
         "Present": False, "LocationInformation": "Port_#0002.Hub_#0003"},
        {"InstanceId": "USB\\C\\p2", "FriendlyName": "Integrated Camera",
         "Present": True, "LocationInformation": ""},
    ]
    devs = windows_devices(registry_reader=lambda: reg, pnp_reader=lambda: pnp)

    checks = [
        ("幽灵记录（Present=False）被剔除 -> 剩 2 台", len(devs) == 2),
        ("order 连续 0,1", [d.order for d in devs] == [0, 1]),
        ("名字/端口取自 PnP", devs[0].name == "aoni webcam A25"
         and devs[0].port == "Port_#0001.Hub_#0003"),
        ("第二台是内置（幽灵被挤掉后移位）", devs[1].name == "Integrated Camera"),
        ("来源标记为 system", all(d.source == "system" for d in devs)),
    ]
    ok = True
    for name, passed in checks:
        print(f"  [{'通过' if passed else '不达标'}] {name}")
        ok = ok and passed
    return ok


# --------------------------------------------------------------------------- #
# 3. 降级：PnP 拿不到 -> 不崩、source=registry、present 全 True
# --------------------------------------------------------------------------- #
def test_degraded_no_pnp() -> bool:
    print("----- 3. 降级：PowerShell/PnP 不可用 -----")
    reg = ["USB\\A\\p0", "USB\\B\\p1"]

    def boom():
        raise RuntimeError("PowerShell 被拒绝执行")

    devs = windows_devices(registry_reader=lambda: reg, pnp_reader=boom)
    checks = [
        ("PnP 抛异常时仍返回全部设备", len(devs) == 2),
        ("present 全 True（未知按存在处理）", all(d.present for d in devs)),
        ("source 标记为 registry", all(d.source == "registry" for d in devs)),
        ("端口回退到 InstanceId 末节", devs[0].port == "p0"),
    ]
    ok = True
    for name, passed in checks:
        print(f"  [{'通过' if passed else '不达标'}] {name}")
        ok = ok and passed
    return ok


# --------------------------------------------------------------------------- #
# 4. 解析优先级：identity -> port -> name -> index(兜底)
# --------------------------------------------------------------------------- #
def test_resolve_priority() -> bool:
    print("----- 4. 解析优先级 identity -> port -> name -> index -----")
    devs = sample_devices()
    checks: list[tuple[str, bool]] = []

    r = resolve_role("left", registry_with(left=role(identity="USB\\B")), devs,
                     interactive=False)
    checks.append(("identity 命中 -> 用该设备索引 2",
                   r.index == 2 and r.how == "identity"))

    r = resolve_role("left",
                     registry_with(left=role(identity="USB\\ZZZ", port="Port_#0002")),
                     devs, interactive=False)
    checks.append(("identity 缺失 -> 走端口命中外接 B",
                   r.index == 2 and r.how == "port"))

    r = resolve_role("left",
                     registry_with(left=role(identity="USB\\ZZZ", port="没有这个端口",
                                             name="aoni")),
                     devs, interactive=False)
    checks.append(("端口也缺失 -> 走名字命中", r.how == "name" and r.index in (1, 2)))

    r = resolve_role("left",
                     registry_with(left=role(identity="USB\\Q", index=5)),
                     devs, interactive=False)
    checks.append(("全缺失但记了索引 -> 兜底索引 5（带 warning）",
                   r.index == 5 and r.how == "index-fallback" and r.warnings))

    ok = True
    for name, passed in checks:
        print(f"  [{'通过' if passed else '不达标'}] {name}")
        ok = ok and passed
    return ok


# --------------------------------------------------------------------------- #
# 5. 冲突：端口与名字指向不同设备 -> 端口优先
# --------------------------------------------------------------------------- #
def test_port_beats_name() -> bool:
    print("----- 5. 冲突用例：端口 与 名字 指向不同设备 -----")
    devs = sample_devices()  # 1 和 2 都叫 "aoni webcam A25"
    # 名字最先匹配到 #1；端口指定的是 #2 —— 端口应当赢
    r = resolve_role("left",
                     registry_with(left=role(port="Port_#0002", name="aoni")),
                     devs, interactive=False)
    checks = [
        ("端口优先于名字", r.how == "port"),
        ("解析到端口指定的那台（#2）", r.index == 2),
    ]
    ok = True
    for name, passed in checks:
        print(f"  [{'通过' if passed else '不达标'}] {name}")
        ok = ok and passed
    return ok


# --------------------------------------------------------------------------- #
# 6. 危险用例：索引漂移，绝不静默用错相机
# --------------------------------------------------------------------------- #
def test_never_silently_wrong() -> bool:
    print("----- 6. 危险用例：索引漂移后绝不用错相机 -----")
    checks: list[tuple[str, bool]] = []
    spec = role(identity="USB\\A", index=0, verified=True)

    # 6a. 身份仍在、但索引从 0 漂到了 1：必须按身份解析到 1，绝不返回 0（内置）
    devs = [dev(0, "Integrated Camera", "USB\\INT"),
            dev(1, "aoni webcam A25", "USB\\A", "Port_#0001")]
    r = resolve_role("left", registry_with(left=spec), devs, interactive=False)
    checks.append(("身份命中漂移后的索引 1（不是内置的 0）",
                   r.index == 1 and r.how == "identity"))

    # 6b. 身份彻底消失、只剩内置：已确认过的角色**不能**退回索引 0
    #     （非交互下必须报错；交互下会弹清单让用户重选 —— 二者都不"静默")
    devs2 = [dev(0, "Integrated Camera", "USB\\INT")]
    try:
        resolve_role("left", registry_with(left=spec), devs2, interactive=False)
    except RuntimeError as exc:
        checks.append(("身份消失 + 已确认 -> 报错而非静默用 #0",
                       "不在线" in str(exc)))
    else:
        checks.append(("身份消失 + 已确认 -> 报错而非静默用 #0", False))

    # 6c. 对照：**未确认**的角色才允许索引兜底（带 warning）
    loose = role(identity="USB\\GONE", index=0, verified=False)
    r = resolve_role("left", registry_with(left=loose), devs2, interactive=False)
    checks.append(("未确认角色 -> 索引兜底且带 warning",
                   r.how == "index-fallback" and r.warnings))

    ok = True
    for name, passed in checks:
        print(f"  [{'通过' if passed else '不达标'}] {name}")
        ok = ok and passed
    return ok


# --------------------------------------------------------------------------- #
# 7. resolve_camera_arg：索引 / 角色名 / port: / 未知角色
# --------------------------------------------------------------------------- #
def test_resolve_camera_arg() -> bool:
    print("----- 7. --left/--right 取值解析 -----")
    devs = sample_devices()
    reg = registry_with(left=role(identity="USB\\A"))
    checks: list[tuple[str, bool]] = []

    checks.append(("None -> None", resolve_camera_arg(None) is None))
    checks.append(("int 0 -> 0", resolve_camera_arg(0, devices=devs) == 0))
    checks.append(("'0' -> 0", resolve_camera_arg("0", devices=devs) == 0))
    checks.append(("角色名 'left' -> 命中索引 1",
                   resolve_camera_arg("left", reg=reg, devices=devs) == 1))
    checks.append(("'port:Port_#0002' -> 2",
                   resolve_camera_arg("port:Port_#0002", devices=devs) == 2))
    checks.append(("'id:USB\\B' -> 2",
                   resolve_camera_arg("id:USB\\B", devices=devs) == 2))

    try:
        resolve_camera_arg("nosuchrole", reg=reg, devices=devs, interactive=False)
        checks.append(("未知角色 -> 报错", False))
    except RuntimeError:
        checks.append(("未知角色 -> 报错", True))

    try:
        resolve_camera_arg("auto", devices=devs)
        checks.append(("'auto' 不该在这里被吞掉 -> 报错", False))
    except RuntimeError:
        checks.append(("'auto' 不该在这里被吞掉 -> 报错", True))

    try:
        resolve_camera_arg("port:没有这个口", devices=devs)
        checks.append(("端口匹配不上 -> 报错并附清单", False))
    except RuntimeError as exc:
        checks.append(("端口匹配不上 -> 报错并附清单",
                       "aoni" in str(exc) or "Integrated" in str(exc)))

    ok = True
    for name, passed in checks:
        print(f"  [{'通过' if passed else '不达标'}] {name}")
        ok = ok and passed
    return ok


# --------------------------------------------------------------------------- #
# 8. 配置往返 + 坏文件不崩
# --------------------------------------------------------------------------- #
def test_config_roundtrip(tmp: Path) -> bool:
    print("----- 8. config/cameras.json 读写 -----")
    path = tmp / "cameras.json"
    reg = Registry(settings={"prefer_external": True, "min_width": 640})
    set_role(reg, "left", entry=sample_devices()[1], index=1, verified=True)
    set_role(reg, "right", entry=sample_devices()[2], index=2, verified=False)
    save_registry(reg, path)

    back = load_registry(path)
    checks = [
        ("角色数一致", set(back.roles) == {"left", "right"}),
        ("identity / port / index 往返一致",
         back.roles["left"].identity == "USB\\A"
         and back.roles["left"].port == "Port_#0001.Hub_#0003"
         and back.roles["left"].index == 1),
        ("verified 往返一致（left 确认 / right 未确认）",
         back.roles["left"].verified and not back.roles["right"].verified),
        ("settings 保留", back.settings.get("min_width") == 640),
        ("文件不存在 -> 空 Registry 不抛", len(load_registry(tmp / "nope.json").roles) == 0),
    ]

    bad = tmp / "broken.json"
    bad.write_text("{ 这不是合法 JSON ", encoding="utf-8")
    checks.append(("损坏文件 -> 空 Registry 不抛", len(load_registry(bad).roles) == 0))

    ok = True
    for name, passed in checks:
        print(f"  [{'通过' if passed else '不达标'}] {name}")
        ok = ok and passed
    return ok


# --------------------------------------------------------------------------- #
# 9. select_cameras：排序 / 跳过内置与坏流 / 无可用返回空
# --------------------------------------------------------------------------- #
def test_select_cameras() -> bool:
    print("----- 9. 自动选择（排序 / 跳过内置 / 无可用不抛） -----")
    devs = sample_devices()
    settings = {"max_index": 2, "min_width": 1280, "min_height": 720}

    def probe_from(mapping):
        def probe(index, **kw):
            return mapping.get(index, ProbeResult(index=index, ok=False, error="打不开"))
        return probe

    checks: list[tuple[str, bool]] = []

    # 三台都能开、分辨率达标：内置被降权 -> 外接优先
    good = {i: ProbeResult(i, True, 1280, 720, 30.0, std=20.0) for i in (0, 1, 2)}
    picks = select_cameras(need=1, devices=devs, probe=probe_from(good),
                          settings=settings)
    checks.append(("内置被降权，优先挑外接", len(picks) == 1 and picks[0].index in (1, 2)))

    # 全黑常量流被 probe 判为 ok=False -> 跳过
    black = {1: ProbeResult(1, False, 1280, 720, 30.0, std=0.0, error="全黑")}
    picks = select_cameras(need=1, devices=devs, probe=probe_from(black),
                          settings=settings)
    checks.append(("全黑流被跳过 -> 无可用返回空", picks == []))

    # 分辨率不达标被过滤
    lowres = {1: ProbeResult(1, True, 640, 480, 30.0, std=20.0)}
    picks = select_cameras(need=1, devices=devs, probe=probe_from(lowres),
                          settings=settings)
    checks.append(("分辨率不达标 -> 被过滤", picks == []))

    # 两台外接都可用 -> need=2 优先挑分辨率一致的一对
    pair = {1: ProbeResult(1, True, 1280, 720, 30.0, std=20.0),
            2: ProbeResult(2, True, 1280, 720, 30.0, std=20.0)}
    picks = select_cameras(need=2, devices=devs, probe=probe_from(pair),
                          settings=settings)
    checks.append(("need=2 挑到两台且分辨率一致",
                   len(picks) == 2 and picks[0].width == picks[1].width))

    # 全打不开 -> 空 list，不抛
    picks = select_cameras(need=1, devices=devs, probe=probe_from({}),
                          settings=settings)
    checks.append(("全部打不开 -> 返回空 list 不抛", picks == []))

    # rank_candidates 不抛且内置排最后
    ranked = rank_candidates(devs, settings=settings)
    checks.append(("rank_candidates：内置排在外接之后",
                   ranked[-1].name == "Integrated Camera"))

    ok = True
    for name, passed in checks:
        print(f"  [{'通过' if passed else '不达标'}] {name}")
        ok = ok and passed
    return ok


# --------------------------------------------------------------------------- #
# 10. from_args 兼容性（守护硬约束：全空必须抛 RuntimeError）
# --------------------------------------------------------------------------- #
def test_from_args_compat() -> bool:
    print("----- 10. from_args 兼容性（守护 ns() 硬约束） -----")
    import stereo.pair_source as ps

    checks: list[tuple[str, bool]] = []

    class FakePS:
        def __init__(self, **kw):
            self.kw = kw

    class FakeProbe:
        def __init__(self, index):
            self.index = index

    real_ps, real_select = ps.PairSource, ps.select_cameras
    ps.PairSource = FakePS
    try:
        def ns(**kw):
            d = dict(left=None, right=None, left_video=None, right_video=None,
                     left_dir=None, right_dir=None, width=1280, height=720, fps=30,
                     backup_camera=None, backend=None, auto_cameras=False)
            d.update(kw)
            return SimpleNamespace(**d)

        # 索引字符串 -> 归一化成 int（CameraPairReader 需要 int）
        src = ps.from_args(ns(left="0", right="2"))
        checks.append(("left='0'/right='2' -> camera=(0, 2) int",
                       src.kw.get("camera") == (0, 2)))
        checks.append(("宽度/高度/帧率透传", (src.kw.get("width"), src.kw.get("height"),
                                       src.kw.get("fps")) == (1280, 720, 30)))

        # 硬约束：全空（ns()）仍必须抛 RuntimeError
        try:
            ps.from_args(ns())
        except RuntimeError:
            checks.append(("--left/--right 全空 -> 仍抛 RuntimeError", True))
        except Exception as exc:                      # noqa: BLE001
            print(f"    （期望 RuntimeError，实际 {type(exc).__name__}: {exc}）")
            checks.append(("--left/--right 全空 -> 仍抛 RuntimeError", False))
        else:
            checks.append(("--left/--right 全空 -> 仍抛 RuntimeError", False))

        # --auto-cameras：自动挑两台（注入假的 select_cameras，不碰真相机）
        ps.select_cameras = lambda **kw: [FakeProbe(4), FakeProbe(5)]
        src = ps.from_args(ns(auto_cameras=True))
        checks.append(("--auto-cameras -> camera=(4, 5)",
                       src.kw.get("camera") == (4, 5)))

        # left='auto' 同样触发自动选择
        src = ps.from_args(ns(left="auto", right=None))
        checks.append(("left='auto' -> 触发自动选择", src.kw.get("camera") == (4, 5)))

        # 自动挑不满两台 -> 报错
        ps.select_cameras = lambda **kw: [FakeProbe(4)]
        try:
            ps.from_args(ns(auto_cameras=True))
        except RuntimeError:
            checks.append(("自动挑不满两台 -> 报错", True))
        else:
            checks.append(("自动挑不满两台 -> 报错", False))
    finally:
        ps.PairSource, ps.select_cameras = real_ps, real_select

    ok = True
    for name, passed in checks:
        print(f"  [{'通过' if passed else '不达标'}] {name}")
        ok = ok and passed
    return ok


# --------------------------------------------------------------------------- #
def test_real_probe() -> bool:
    """有相机时额外验证真机枚举（无相机则跳过，不算不达标）。"""
    print("----- 附加：真机枚举（有相机才验证） -----")
    import sphere_tracker.camera_registry as cr

    if cr.current_platform() != "windows":
        print(f"  [跳过] 当前平台是 {cr.current_platform()}，退化为纯索引模式（符合设计）")
        return True
    devs = cr.list_devices()
    if not devs:
        print("  [跳过] 当前没有枚举到相机设备（或注册表不可读）")
        return True
    probes = {d.order: cr.probe_index(d.order) for d in devs}
    print(format_device_table(devs, probes))
    print("  [信息] 以上为真机实测，仅供参考（不参与判定）")
    return True


def make_scratch_dir() -> Path:
    """可写的临时目录（沙箱里 %TEMP% 可能不可写，退回项目内 outputs/）。"""
    import tempfile
    try:
        d = Path(tempfile.mkdtemp(prefix="camera_registry_"))
        (d / ".probe").write_text("ok", encoding="utf-8")
        (d / ".probe").unlink()
        return d
    except OSError:
        d = ROOT / "outputs" / "_camera_registry_tmp"
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)
        return d


def main() -> int:
    ap = argparse.ArgumentParser(description="摄像头设备管理自测")
    ap.add_argument("--probe", action="store_true",
                    help="有相机时额外跑一次真机枚举（默认跳过）")
    args = ap.parse_args()

    print("=" * 62)
    print("摄像头设备管理自测（全注入假数据，不打开任何相机）")
    print("=" * 62)

    tmp = make_scratch_dir()
    print(f"临时目录：{tmp}\n")
    try:
        results = [
            test_parse_instance(),
            test_pnputil_parse(),
            test_windows_devices_merge(),
            test_degraded_no_pnp(),
            test_resolve_priority(),
            test_port_beats_name(),
            test_never_silently_wrong(),
            test_resolve_camera_arg(),
            test_config_roundtrip(tmp),
            test_select_cameras(),
            test_from_args_compat(),
        ]
        print()
        if args.probe:
            results.append(test_real_probe())
        else:
            print("（要顺带验证真机枚举，加 --probe）")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    ok = all(results)
    print("判定结果：" + ("全部通过" if ok else "有不达标项"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
