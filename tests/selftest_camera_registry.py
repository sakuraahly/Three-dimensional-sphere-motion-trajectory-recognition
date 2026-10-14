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
import json
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
    AmbiguousMatch,
    DeviceEntry,
    ProbeResult,
    Registry,
    RoleSpec,
    _parse_pnputil,
    clear_ignored,
    format_device_table,
    ignored_orders,
    is_device_ignored,
    load_registry,
    mark_ignored,
    parse_windows_instance,
    rank_candidates,
    resolve_camera_arg,
    resolve_role,
    role_of_device,
    save_registry,
    select_cameras,
    set_role,
    unmark_ignored,
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

    # ⚠️ **这条在 2026-10-14 改了契约**：原来"绑了身份/端口、但对不上"会**按名字兜底**
    # （`r.how == "name"`）。实测证明那是**危险的猜测**：
    # 两台同型号相机同名，其中一台换了 USB 口 → 名字兜底任意抓一台 →
    # 左右可能互换，而且两路分辨率可能不一致（一台 1280x720、一台 1920x1080），
    # **双目标定直接失去意义，界面上却看不出异常**。
    #
    # 新契约：**有身份信息却对不上 = 那台不在线**，必须报错让用户重新确认，
    # 不许用同名的另一台顶替。
    try:
        resolve_role("left",
                     registry_with(left=role(identity="USB\\ZZZ", port="没有这个端口",
                                             name="aoni")),
                     devs, interactive=False)
        checks.append(("绑了身份却对不上 -> **必须拒绝**（不许按名字猜）", False))
    except RuntimeError as exc:
        msg = str(exc)
        checks.append(("绑了身份却对不上 -> 拒绝，并说清期望的身份/端口",
                       "不在线" in msg and "USB\\ZZZ" in msg))
        checks.append(("拒绝时把当前设备清单摆出来（让用户能自己判断）",
                       "aoni webcam A25" in msg and "#1" in msg))

    # 反过来：**从没绑定过身份**的角色只能按名字找；同名多台时必须报歧义
    # （`sample_devices()` 里 #1 与 #2 同名，所以这里必然歧义）
    try:
        r = resolve_role("left", registry_with(left=role(name="aoni")), devs,
                        interactive=False)
        checks.append(("只有名字且同名多台 -> **必须报歧义**（不许任选一台）", False))
    except AmbiguousMatch as exc:
        checks.append(("只有名字且同名多台 -> 报歧义并列出候选",
                       "匹配到" in str(exc) and "#1" in str(exc)))

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
# 11. 「无关」清单：标记 / 解除 / 影响自动挑选 / 与角色互斥 / 读写往返
# --------------------------------------------------------------------------- #
def test_ignored(tmp: Path) -> bool:
    print("----- 11. 「无关」清单（左/右/无关 三选一，自动挑选跳过无关） -----")
    devs = sample_devices()          # 0=内置, 1=外接A, 2=外接B
    settings = {"max_index": 2, "min_width": 1280, "min_height": 720}

    def probe_from(mapping):
        def probe(index, **kw):
            return mapping.get(index, ProbeResult(index=index, ok=False, error="打不开"))
        return probe

    all_good = {i: ProbeResult(i, True, 1280, 720, 30.0, std=20.0) for i in (0, 1, 2)}
    checks: list[tuple[str, bool]] = []

    # --- 标记 / 识别 ---
    reg = Registry()
    mark_ignored(reg, entry=devs[2], index=2, verified=True)
    checks.append(("标记后进入清单", len(reg.ignored) == 1))
    checks.append(("该台被认作无关", is_device_ignored(devs[2], reg)))
    checks.append(("别的台不受影响", not is_device_ignored(devs[1], reg)))
    checks.append(("ignored_orders 正确", ignored_orders(reg, devs) == {2}))

    # --- 核心价值：按**身份**记，换编号之后依然认得出 ---
    moved = [dev(0, "Integrated Camera", "USB\\INT", ""),
             dev(1, "aoni webcam A25", "USB\\B", "Port_#0002.Hub_#0003"),   # 漂到 1
             dev(2, "aoni webcam A25", "USB\\A", "Port_#0001.Hub_#0003")]
    checks.append(("身份匹配：换编号后仍认作无关",
                   is_device_ignored(moved[1], reg) and not is_device_ignored(moved[2], reg)))

    # --- 与角色互斥：标无关要把它从 left/right 摘掉 ---
    reg2 = Registry()
    set_role(reg2, "left", entry=devs[1], index=1, verified=True)
    mark_ignored(reg2, entry=devs[1], index=1, verified=True)
    checks.append(("标无关会自动摘掉它的旧角色", "left" not in reg2.roles))
    checks.append(("role_of_device 不再报它", role_of_device(reg2, devs[1]) is None))

    # --- 解除 / 清空 ---
    checks.append(("unmark_ignored 摘掉 1 条", unmark_ignored(reg2, entry=devs[1]) == 1))
    checks.append(("摘掉后不再是无关", not is_device_ignored(devs[1], reg2)))
    mark_ignored(reg2, entry=devs[0], index=0, verified=True)
    mark_ignored(reg2, entry=devs[2], index=2, verified=True)
    checks.append(("clear_ignored 清空", clear_ignored(reg2) == 2 and not reg2.ignored))

    # --- 排序：即使"外接 vs 内置"，无关也一律沉到最后 ---
    reg3 = Registry()
    mark_ignored(reg3, entry=devs[1], index=1, verified=True)      # 把一台外接标成无关
    order = [d.order for d in rank_candidates(devs, reg=reg3,
                                             settings={"prefer_external": True})]
    checks.append(("无关沉到内置之后（rank）", order[-1] == 1))

    # --- 自动挑选：有别的可用时绝不碰无关的 ---
    picks = select_cameras(need=1, devices=devs, probe=probe_from(all_good),
                           reg=reg3, settings=settings)
    checks.append(("自动挑选跳过无关（need=1）", len(picks) == 1 and picks[0].index != 1))
    picks2 = select_cameras(need=2, devices=devs, probe=probe_from(all_good),
                            reg=reg3, settings=settings)
    checks.append(("自动挑选跳过无关（need=2，两台都不含 #1）",
                   {p.index for p in picks2} == {0, 2}))

    # --- 兜底：只剩无关的能用时才动用（并会打印警告）---
    reg4 = Registry()
    for d in devs:
        mark_ignored(reg4, entry=d, index=d.order, verified=True)
    picks3 = select_cameras(need=1, devices=devs, probe=probe_from(all_good),
                            reg=reg4, settings=settings)
    checks.append(("全都无关时仍兜底返回（不空手）", len(picks3) == 1))

    # --- 读写往返（含"老配置没有 ignored 键"的向后兼容）---
    path = tmp / "ignored.json"
    reg5 = Registry()
    set_role(reg5, "left", entry=devs[1], index=1, verified=True)
    mark_ignored(reg5, entry=devs[2], index=2, verified=True)
    save_registry(reg5, path)
    back = load_registry(path)
    checks.append(("ignored 往返一致",
                   len(back.ignored) == 1 and back.ignored[0].identity == "USB\\B"
                   and back.ignored[0].verified))
    checks.append(("roles 与 ignored 并存", set(back.roles) == {"left"}))

    legacy = tmp / "legacy.json"
    legacy.write_text(json.dumps({"v": 1, "roles": {"left": {"identity": "USB\\A",
                                                             "index": 1, "verified": True}},
                                  "settings": {}}), encoding="utf-8")
    checks.append(("老配置（没有 ignored 键）能读，清单为空",
                   load_registry(legacy).ignored == []))
    weird = tmp / "weird.json"
    weird.write_text(json.dumps({"ignored": ["不是字典", 42, {"index": 1}]}),
                     encoding="utf-8")
    checks.append(("ignored 里混进垃圾项不崩，只取合法项",
                   len(load_registry(weird).ignored) == 1))

    # --- 角色指向无关设备 -> 必须给出警告（配置自相矛盾）---
    # 注意：这个状态**没法用 API 造出来**（mark_ignored 会把角色摘掉，上一段已验证），
    # 只可能来自手改 JSON。这里直接手搓一份，验证程序遇到它不会装看不见。
    both = role(identity="USB\\B", index=2, verified=True)
    reg6 = Registry(roles={"left": both}, ignored=[role(identity="USB\\B", index=2,
                                                        verified=True)])
    res = resolve_role("left", reg6, devs, interactive=False)
    checks.append(("角色与无关冲突时给出警告", len(res.warnings) == 1
                   and "无关" in res.warnings[0]))

    # --- 清单上看得见 ---
    table = format_device_table(devs, reg=reg3)
    checks.append(("format_device_table 打出 [无关]", "[无关]" in table))
    checks.append(("未标记的台没有 [无关]", table.count("[无关]") == 1))

    ok = True
    for name, passed in checks:
        print(f"  [{'通过' if passed else '不达标'}] {name}")
        ok = ok and passed
    return ok


# --------------------------------------------------------------------------- #
def test_probe_fallback_and_cache_invalidation() -> bool:
    """第 12 节：探测的回退 + 坏缓存必须自己作废（2026-10-14 加）。

    ## 守的是两个**只在真机上才看得见**的缺陷

    **① 探测后端被写死成 MSMF，而它是间歇性坏的。**
    `_default_opener` 用 `INDEX_BACKEND`（= MSMF，2026-10-10 的实测选择），
    但 2026-10-14 一个下午就见到它来回变：
    `MSMF idx=1` 好 9.2s → 好 → **坏 34.3s** → **坏 33.5s** → 又好。
    而同一台相机在 **DSHOW 上每次都秒开、15/15 帧、同样给 1920x1080**。
    坏的时候 `probe_index` 要等 33 秒才失败，于是
    `cameras.py list --probe` / 画质会诊 / 自动挑相机**全报"一帧都拿不到"**，
    用户会以为相机坏了。所以探测路径加**受控回退**（实时取帧路径**刻意不加**，
    因为混用后端会让 MSMF 那路挂掉，见 `sources.py` 的长注释）。

    **② 缓存只在"打不开"时作废，不在"打得开但拿不出帧"时作废。**
    `_raw_open` 已经校验首帧，所以它返回 `None` 就意味着**这条缓存是坏的**；
    旧代码只打印一句就往下搜，**缓存不删** → 每次启动都白等一遍
    （实测 34.3 s）。这正是 `交接文档.md` 坑 35「缓存没有过期概念」的同一件事。

    ## 判据怎么做到"不依赖真机"

    回退那条用**替身**注入（第一次"无帧"、第二次"有帧"），
    并断言替身被调了**两次**且拿到的是第二次的结果 ——
    否则"回退没生效"会与"回退生效了"长得一样（坑 24：替身测试必须断言
    "替身真的走到了那一步"）。缓存那条用临时 JSON 做真实的读写往返。
    """
    print("----- 12. 探测回退 + 坏缓存自动作废 -----")
    import json as _json
    import sphere_tracker.camera_registry as cr

    checks: list[tuple[str, bool]] = []

    # ---- ① 回退：主后端无帧 -> 换一个后端再探 ----
    if cr.current_platform() != "windows":
        print(f"  [跳过] 平台 {cr.current_platform()}（没有第二套后端可回退）")
    else:
        calls: list[int | None] = []

        def fake_probe(idx, *, width=0, height=0, fps=0, backend=None, opener=None):
            calls.append(backend)
            if len(calls) == 1:                      # 第一次：假装主后端无帧
                return cr.ProbeResult(index=idx, ok=False, error="读不到画面")
            return cr.ProbeResult(index=idx, ok=True, width=1280, height=720,
                                  fps=30.0, std=42.0)   # 第二次：另一个后端正常

        real = cr.probe_index
        cr.probe_index = fake_probe                    # type: ignore[assignment]
        try:
            r = cr.probe_index_with_fallback(1)
        finally:
            cr.probe_index = real                      # type: ignore[assignment]
        print(f"  替身调用 {len(calls)} 次，后端依次 = {calls}")
        print(f"  回退结果：ok={r.ok} {r.width}x{r.height} std={r.std} note={r.error!r}")
        checks += [
            ("主后端无帧时会**真的再试一次**（替身被调 2 次）", len(calls) == 2),
            ("两次用的是**不同**的后端（确实换了，不是原地重试）",
             len(calls) == 2 and calls[0] != calls[1]),
            ("回退成功时判定为可用（不再报「一帧都拿不到」）", r.ok is True),
            ("结果带上了「这次是回退来的」痕迹（可诊断）", "改用" in (r.error or "")),
        ]

    # ---- ② 坏缓存：确认给不出帧时必须把自己删掉 ----
    cache_path = cr._MODE_CACHE if hasattr(cr, "_MODE_CACHE") else None
    if cache_path is None:
        from sphere_tracker import sources as _s
        cache_path = _s._MODE_CACHE
    from sphere_tracker import sources as _src
    backup = cache_path.read_text(encoding="utf-8") if cache_path.exists() else None
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(_json.dumps({
            "77": {"v": 2, "backend": "MSMF", "width": 1920, "height": 1080,
                   "actual_width": 1920, "actual_height": 1080, "fps": 30.0},
            "78": {"v": 2, "backend": "DSHOW", "width": 640, "height": 480,
                   "actual_width": 640, "actual_height": 480, "fps": 30.0},
        }), encoding="utf-8")
        dropped = _src._drop_mode_cache(77)
        left = _json.loads(cache_path.read_text(encoding="utf-8"))
        dropped_again = _src._drop_mode_cache(77)
        print(f"  删 77 -> {dropped}；剩下的键 = {sorted(left)}；再删一次 -> {dropped_again}")
        checks += [
            ("作废坏缓存返回 True（真的删到了）", dropped is True),
            ("被作废的那条**不在**缓存里了", "77" not in left),
            ("**别的相机不受影响**（78 还在）", "78" in left),
            ("对不存在的键返回 False（幂等、不误报）", dropped_again is False),
        ]
    finally:
        if backup is None:
            cache_path.unlink(missing_ok=True)
        else:
            cache_path.write_text(backup, encoding="utf-8")

    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


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
            test_ignored(tmp),
            test_probe_fallback_and_cache_invalidation(),
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
