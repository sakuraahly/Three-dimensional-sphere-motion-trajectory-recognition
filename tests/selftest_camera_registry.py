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
import atexit
import io
import json
import os
import shutil
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
while not (ROOT / "sphere_tracker").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sphere_tracker import camera_registry as cr  # noqa: E402

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
    resolve_roles,
    role_of_device,
    save_registry,
    select_cameras,
    set_role,
    unmark_ignored,
    windows_devices,
)

# 三台典型设备：外接 A（USB\A）、外接 B（USB\B）、笔记本内置（USB\INT）
MODULE_GUID = "{e5323777-f976-4f5b-9b55-b94699c46e44}"

# 第 12/13 节要**改写模式缓存**来验证"坏缓存自己作废"与"缓存不许盖过显式请求"。
# 这两节原先直接改写真实的 `config/camera_modes.json` 再还原 —— 在 `-j 6` 下
# `selftest_sources` 会同时走 `_drop_mode_cache()` 写同一个文件，Windows 直接
# 抛 PermissionError（2026-11-02 08:11 全量因此红了一次，单独跑却绿）。
# 缓存本身只是"上次探测的备忘录"，指到进程私有临时文件即可：
# 既不碰用户配置，也没有任何跨进程竞争。
from sphere_tracker import sources as _sources_mod  # noqa: E402
import sphere_tracker.camera_registry as cr_mod  # noqa: E402

_TMP_MODE_CACHE = Path(tempfile.gettempdir()) / f"camera_modes_selftest_{os.getpid()}.json"
_sources_mod._MODE_CACHE = _TMP_MODE_CACHE
atexit.register(lambda: _TMP_MODE_CACHE.unlink(missing_ok=True))


def dev(order, name="", identity="", port="", present=True, source="powershell"):
    return DeviceEntry(order=order, name=name, identity=identity, port=port,
                       present=present, source=source)


def _raises(fn, exc_type) -> bool:
    """`fn()` 抛 `exc_type` 吗？（抛别的 / 不抛都算 False，判据要能报红）"""
    try:
        fn()
    except exc_type:
        return True
    except Exception:                                        # noqa: BLE001
        return False
    return False


def sample_devices():
    """一台内置 + 两台同型号外接（不同端口）—— 最容易配错的那种组合。"""
    return [
        dev(0, "Integrated Camera", "USB\\INT", "", present=True),
        dev(1, "aoni webcam A25", "USB\\A", "Port_#0001.Hub_#0003"),
        dev(2, "aoni webcam A25", "USB\\B", "Port_#0002.Hub_#0003"),
    ]


def role(identity="", port="", name="", index=None, verified=False,
         capture_index=None, location=""):
    return RoleSpec(identity=identity, port=port, name=name, index=index,
                    verified=verified, capture_index=capture_index,
                    location=location)


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

    # ⚠️ 2026-11-12 起**必须**带 `capture_index`：设备身份对上但采集索引没实测时，
    # `resolve_role()` 会抛 `CaptureIndexNotVerified`（不再拿 PnP 位次顶上）。
    # 这里测的是"身份/端口怎么匹配设备"，所以给一个实测值（= 该设备当前位次）。
    r = resolve_role("left", registry_with(left=role(identity="USB\\B", capture_index=2)),
                     devs, interactive=False)
    checks.append(("identity 命中 -> 用该设备索引 2",
                   r.index == 2 and r.how == "identity"))

    r = resolve_role("left",
                     registry_with(left=role(identity="USB\\ZZZ", port="Port_#0002",
                                            capture_index=2)),
                     devs, interactive=False)
    checks.append(("identity 缺失 -> 走端口命中外接 B",
                   r.index == 2 and r.how == "port"))

    # 🚨 语义分离的核心判据（"程序显示的相机 ≠ 真实相机"的根因）：
    # 设备身份对上了、但 `capture_index` 从没实测过 —— 必须**报清晰错误**，
    # 绝不静默用 `entry.order`。报错要能让人照着做（角色/设备名/PnP 位次/下一步）。
    try:
        resolve_role("left", registry_with(left=role(identity="USB\\B")), devs,
                     interactive=False)
        checks.append(("capture_index 缺失 -> 必须拒绝（不许拿 PnP 位次顶上）", False))
    except cr_mod.CaptureIndexNotVerified as exc:
        msg = str(exc)
        checks.append(("capture_index 缺失 -> 抛 CaptureIndexNotVerified（不是静默回退）",
                       True))
        checks.append(("报错含角色 + 设备名 + PnP 位次（知道是哪一路、哪台）",
                       "「left」" in msg and "aoni webcam A25" in msg
                       and "PnP 位次 #2" in msg))
        checks.append(("报错说清「为什么不能猜」（枚举位次 ≠ 采集索引）",
                       "枚举顺序" in msg and "capture_index" in msg))
        checks.append(("报错给出下一步的 identify 命令（可照着做）",
                       "python apps/tools/cameras.py identify" in msg))
    except RuntimeError as exc:
        checks.append(("capture_index 缺失 -> 抛 CaptureIndexNotVerified（不是静默回退）",
                       False))
        print(f"    （期望 CaptureIndexNotVerified，实际 RuntimeError: {exc}）")

    # 诊断路径（`cameras.py check` / 界面在线判定）允许不实测，但**必须标记未验证**，
    # 而且明确写出"这只是 PnP 位次，不能拿来开相机"。
    probe_only = resolve_role("left", registry_with(left=role(identity="USB\\B")), devs,
                              interactive=False, require_capture_index=False)
    checks.append(("诊断路径（require_capture_index=False）可以给出 PnP 位次，"
                   "但标记 capture_index_verified=False",
                   probe_only.index == 2 and probe_only.capture_index_verified is False))
    checks.append(("诊断路径的 warning 明确写「只是 PnP 位次 / 不能拿来开相机」",
                   any("PnP 位次" in w and "不能拿来开相机" in w
                       for w in probe_only.warnings)))

    # ⚠️ **这条在 2026-10-14 改了契约**：原来"绑了身份/端口、但对不上"会**按名字兜底**
    # （`r.how == "name"`）。实测证明那是**危险的猜测**：
    # 两台同型号相机同名，其中一台换了 USB 口 → 名字兜底任意抓一台 →
    # 左右可能互换，而且两路分辨率可能不一致（一台 1280x720、一台 1920x1080），
    # **双目标标定直接失去意义，界面上却看不出异常**。
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

    # 2026-11-20：身份/端口全缺失时，配置里的 index 只是 PnP 位次，
    # **不能**在 require_capture_index=True 的运行路径里拿来开相机。
    try:
        resolve_role("left",
                     registry_with(left=role(identity="USB\\Q", index=5)),
                     devs, interactive=False)
        checks.append(("全缺失 + 记了索引 -> 运行路径必须拒绝（不猜 PnP 位次）", False))
    except cr_mod.CaptureIndexNotVerified as exc:
        checks.append(("全缺失 + 记了索引 -> 拒绝，并点明 PnP 位次不是采集索引",
                       "PnP 位次" in str(exc) and "identify" in str(exc)))
    diag = resolve_role("left",
                        registry_with(left=role(identity="USB\\Q", index=5)),
                        devs, interactive=False, require_capture_index=False)
    checks.append(("诊断路径（require_capture_index=False）才允许显示索引 5",
                   diag.index == 5 and diag.how == "index-fallback"
                   and bool(diag.warnings) and not diag.capture_index_verified))

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
    # （capture_index=2：这条测的是"端口优先于名字"，必须给实测索引才不被拦）
    r = resolve_role("left",
                     registry_with(left=role(port="Port_#0002", name="aoni",
                                             capture_index=2)),
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
    #
    # ⚠️ 2026-11-12：这里传 `require_capture_index=False`，**故意**只看"身份匹配到
    # 哪一台"（这条测的是设备对应关系，不是采集编号）。真正的运行路径要求
    # `capture_index`，缺了就直接拒绝 —— 那件事由 6d 单独钉住。
    devs = [dev(0, "Integrated Camera", "USB\\INT"),
            dev(1, "aoni webcam A25", "USB\\A", "Port_#0001")]
    r = resolve_role("left", registry_with(left=spec), devs, interactive=False,
                     require_capture_index=False)
    checks.append(("身份命中漂移后的索引 1（不是内置的 0）",
                   r.index == 1 and r.how == "identity" and not r.capture_index_verified))

    # 6d. **同一台、同一个身份，但采集索引没实测** -> 必须拒绝，不许用 PnP 位次。
    #     这是"程序显示的相机 ≠ 真实相机"的核心防线。
    try:
        resolve_role("left", registry_with(left=role(identity="USB\\A", index=1,
                                                     verified=True)),
                     devs, interactive=False)
        checks.append(("已确认身份但缺 capture_index -> 拒绝（不猜 PnP 位次）", False))
    except cr_mod.CaptureIndexNotVerified:
        checks.append(("已确认身份但缺 capture_index -> 拒绝（不猜 PnP 位次）", True))
    except RuntimeError as exc:
        checks.append(("已确认身份但缺 capture_index -> 拒绝（不猜 PnP 位次）", False))
        print(f"    （期望 CaptureIndexNotVerified，实际 {type(exc).__name__}: {exc}）")

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

    # 6c. 未确认角色：**运行路径也不许**拿 PnP 位次开相机；只有诊断路径
    #     （require_capture_index=False）才允许显示那个位次作参考。
    loose = role(identity="USB\\GONE", index=0, verified=False)
    try:
        resolve_role("left", registry_with(left=loose), devs2, interactive=False)
        checks.append(("未确认角色 -> 运行路径也拒绝（不拿 PnP 位次开相机）", False))
    except cr_mod.CaptureIndexNotVerified:
        checks.append(("未确认角色 -> 运行路径也拒绝（不拿 PnP 位次开相机）", True))
    r_diag = resolve_role("left", registry_with(left=loose), devs2,
                          interactive=False, require_capture_index=False)
    checks.append(("未确认角色的诊断路径 -> 只显示 PnP 位次且带 warning",
                   r_diag.how == "index-fallback" and bool(r_diag.warnings)
                   and not r_diag.capture_index_verified))

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
    # 角色名 -> 索引这条路要求 `capture_index` 实测过（见第 4 节）。
    reg = registry_with(left=role(identity="USB\\A", capture_index=1))
    # 旧配置（没有 capture_index 字段）单独造一份：它必须能被**读进来**，
    # 只是按角色名启动时会被明确拒绝（不是静默用 PnP 位次）。
    old_reg = registry_with(left=role(identity="USB\\A"))
    checks: list[tuple[str, bool]] = []

    checks.append(("None -> None", resolve_camera_arg(None) is None))
    checks.append(("int 0 -> 0", resolve_camera_arg(0, devices=devs) == 0))
    checks.append(("'0' -> 0", resolve_camera_arg("0", devices=devs) == 0))
    checks.append(("角色名 'left' -> 命中索引 1",
                   resolve_camera_arg("left", reg=reg, devices=devs) == 1))
    # `port:/id:/name:` 是"显式点名设备"，但打开仍要**实测采集索引**。
    # 先给一台绑定了 capture_index 的角色，再验证能取到它；没有实测值必须拒。
    reg_b = registry_with(right=role(identity="USB\\B", capture_index=2))
    checks.append(("'port:Port_#0002' + 已实测角色 -> 2",
                   resolve_camera_arg("port:Port_#0002", reg=reg_b,
                                      devices=devs) == 2))
    checks.append(("'id:USB\\B' + 已实测角色 -> 2",
                   resolve_camera_arg("id:USB\\B", reg=reg_b, devices=devs) == 2))
    checks.append(("'port:Port_#0002' 诊断路径（不要求实测）-> 2",
                   resolve_camera_arg("port:Port_#0002", devices=devs,
                                      require_capture_index=False) == 2))
    try:
        resolve_camera_arg("port:Port_#0002", devices=devs)
        checks.append(("'port:' 命中设备但无 capture_index -> 拒绝（不返回 PnP 位次）", False))
    except cr_mod.CaptureIndexNotVerified:
        checks.append(("'port:' 命中设备但无 capture_index -> 拒绝（不返回 PnP 位次）", True))
    checks.append(("旧配置无 capture_index = None",
                   old_reg.roles["left"].capture_index is None))

    # 旧 gui_state 里存过 "#N 名字"：只有当前实测候选里真有这个 #N 才认，
    # 否则必须丢掉（不能让旧偏好绕过 capture_index 门禁）。
    from apps._gui_mono import MonoWindow
    checks.append(("旧 gui_state 的 '#N' 只在实测候选里才认",
                   MonoWindow._match_saved_camera("#9 old", ["left", "#0 aoni"]) == ""
                   and MonoWindow._match_saved_camera("#0 old", ["#0 aoni"]) == "#0 aoni"))

    # 单目下拉只列**实测 capture_index**，不把 PnP 位次（#dev.order）塞进 --camera。
    # 这条是 Critic 指出的测试缺口：改回 `#{d.order}` 必须变红。
    from types import SimpleNamespace
    fake_state = SimpleNamespace(camera_roles=lambda: {
        "left": {"name": "left cam", "capture_index": 2},
        "right": {"name": "right cam", "capture_index": None},
    })
    fake_win = SimpleNamespace(app=SimpleNamespace(state=fake_state),
                               _devices=[DeviceEntry(0, "pnpo", "USB\\A"),
                                         DeviceEntry(1, "pnpt", "USB\\B")])
    vals = MonoWindow._camera_values(fake_win)
    checks.append(("单目下拉只列实测 #采集索引，不列 PnP 位次",
                   "#2 left cam" in vals
                   and not any(v.startswith("#0") or v.startswith("#1") for v in vals)))

    # 生产入口的 is_avoided 必须传 settings；否则用户配置的 avoid_names 被忽略。
    for rel in ("apps/cli_main.py", "apps/gui_main.py"):
        src = (Path(__file__).resolve().parents[1] / rel).read_text(encoding="utf-8")
        checks.append((f"{rel} 的 is_avoided 传 reg.settings",
                       "is_avoided(name, reg.settings)" in src))

    # 🚨 旧配置**按角色名启动**时必须拒绝（这是"程序显示的相机 ≠ 真实相机"的入口）。
    try:
        resolve_camera_arg("left", reg=old_reg, devices=devs, interactive=False)
        checks.append(("旧配置（无 capture_index）按角色名解析 -> 拒绝", False))
    except cr_mod.CaptureIndexNotVerified:
        checks.append(("旧配置（无 capture_index）按角色名解析 -> 拒绝", True))

    cap_reg = registry_with(left=role(identity="USB\\A", capture_index=2))
    cap_role = resolve_role("left", cap_reg, devs, interactive=False)
    checks.append(("capture_index 覆盖 identity 命中的 PnP 索引",
                   cap_role.index == 2 and cap_role.entry is not None
                   and cap_role.entry.order == 1 and cap_role.how == "identity"))
    checks.append(("resolve_camera_arg 使用 capture_index",
                   resolve_camera_arg("left", reg=cap_reg, devices=devs) == 2))

    both = registry_with(left=role(identity="USB\\A", capture_index=2),
                         right=role(identity="USB\\B", capture_index=0))
    resolved = resolve_roles(["left", "right"], both, devs, interactive=False)
    checks.append(("resolve_roles 使用 capture_index (left=2/right=0)",
                   resolved["left"].index == 2 and resolved["right"].index == 0))
    checks.append(("left/right capture index 不相等",
                   resolved["left"].index != resolved["right"].index))

    miss = registry_with(left=role(identity="USB\\ZZZ", index=1,
                                   verified=False, capture_index=2))
    try:
        resolve_role("left", miss, devs, interactive=False)
        checks.append(("identity 不匹配 + 未确认 -> 运行路径必须拒绝", False))
    except cr_mod.CaptureIndexNotVerified:
        checks.append(("identity 不匹配 + 未确认 -> 运行路径必须拒绝", True))
    miss_diag = resolve_role("left", miss, devs, interactive=False,
                             require_capture_index=False)
    checks.append(("identity 不匹配的诊断路径仍只报 PnP 位次 1",
                   miss_diag.index == 1 and miss_diag.how == "index-fallback"
                   and not miss_diag.capture_index_verified))

    collide = registry_with(left=role(identity="USB\\A", capture_index=2),
                            right=role(identity="USB\\B", capture_index=2))
    try:
        resolve_roles(["left", "right"], collide, devs, interactive=False)
        checks.append(("capture_index 冲突 -> 报错", False))
    except RuntimeError as exc:
        checks.append(("capture_index 冲突 -> 报错",
                       "冲突" in str(exc) and "#2" in str(exc)))

    pin_reg = Registry()
    set_role(pin_reg, "left", entry=devs[1], index=1, verified=True)
    pin_reg.roles["left"].capture_index = 2
    set_role(pin_reg, "left", entry=devs[1], index=1, verified=True)
    checks.append(("set_role 同 identity 保留 capture_index",
                   pin_reg.roles["left"].capture_index == 2))
    buf = io.StringIO()
    with redirect_stdout(buf):
        set_role(pin_reg, "left", entry=devs[2], index=2, verified=True)
    checks.append(("set_role 换 identity 清空并警告",
                   pin_reg.roles["left"].capture_index is None
                   and "capture_index" in buf.getvalue()))

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
    reg.roles["left"].capture_index = 2
    # 「位置」标签：床 / 阳台门（留空 = 未标注）。它只描述"这个机位对着哪儿"，
    # 不参与解析，但必须能**原样往返** —— 否则用户标了一次、下次打开就没了。
    cr_mod.set_location(reg, "left", "床")
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
        ("capture_index 往返一致（left=2 / right=None）",
         back.roles["left"].capture_index == 2
         and back.roles["right"].capture_index is None),
        ("location 往返一致（left=床 / right=未标注）",
         back.roles["left"].location == "床"
         and back.roles["right"].location == ""),
        ("location_text：空 -> 「未标注」",
         cr_mod.location_text(back.roles["right"]) == "未标注"),
        ("set_location 只改位置，不动身份/编号",
         cr_mod.set_location(back, "left", "阳台门") is not None
         and back.roles["left"].location == "阳台门"
         and back.roles["left"].index == 1
         and back.roles["left"].capture_index == 2),
        ("set_location 给不存在的角色 -> ValueError（位置必须有主语）",
         _raises(lambda: cr_mod.set_location(back, "nosuchrole", "床"), ValueError)),
        ("settings 保留", back.settings.get("min_width") == 640),
        ("文件不存在 -> 空 Registry 不抛", len(load_registry(tmp / "nope.json").roles) == 0),
    ]

    # 旧配置（没有 location 键）必须能读，值按"未标注"处理（向后兼容）。
    legacy = tmp / "legacy_no_location.json"
    legacy.write_text(json.dumps({
        "v": 1, "roles": {"left": {"identity": "USB\\A", "index": 1,
                                   "verified": True, "capture_index": 3}},
        "settings": {}}, ensure_ascii=False), encoding="utf-8")
    legacy_back = load_registry(legacy)
    checks.append(("旧配置没有 location 键 -> 读成空串（未标注），不报错",
                   legacy_back.roles["left"].location == ""
                   and legacy_back.roles["left"].capture_index == 3))

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

    # ---- 虚拟摄像头（vivo / OBS / ManyCam…）：**不参与绑定** ----
    # 2026-11-11 现场：点 #3 的 DSHOW 兜底，开出来的是 vivo 的虚拟摄像头
    #（只注册在 DSHOW，PnP 列表里根本没有它）。它绝不能成为 left/right。
    virt = [dev(3, "vivo 办公套件虚拟摄像头", "SW\\VIVO"),
            dev(4, "OBS Virtual Camera", "SW\\OBS")]
    virt_devs = devs + virt
    ranked_v = rank_candidates(virt_devs, settings=settings)
    checks.append(("rank：虚拟摄像头沉到最后（真实相机优先）",
                   {d.order for d in ranked_v[-2:]} == {3, 4}
                   and ranked_v[0].order != 3))
    allgood = {i: ProbeResult(i, True, 1280, 720, 30.0, std=20.0)
               for i in (0, 1, 2, 3, 4)}
    picks_v = select_cameras(need=1, devices=virt_devs, probe=probe_from(allgood),
                             settings=settings)
    checks.append(("自动挑选：有真实相机时**绝不**挑虚拟摄像头",
                   len(picks_v) == 1 and picks_v[0].index in (1, 2)))
    # 真机全失败、只有虚拟摄像头能开 -> 宁可返回空，也不静默选虚拟。
    fail_real = {i: ProbeResult(i, False, error="打不开") for i in (1, 2)}
    virt_ok = {3: ProbeResult(3, True, 1280, 720, 30.0, std=20.0)}
    with redirect_stdout(io.StringIO()):
        picks_v2 = select_cameras(need=1, devices=virt_devs,
                                  probe=probe_from({**fail_real, **virt_ok}),
                                  settings=settings)
    checks.append(("真机全失败时也不自动选虚拟摄像头（返回空）", picks_v2 == []))

    ok = True
    for name, passed in checks:
        print(f"  [{'通过' if passed else '不达标'}] {name}")
        ok = ok and passed
    return ok


# --------------------------------------------------------------------------- #
# 9b. select_cameras：phantom 索引自适应早停（连续失败即停，不误伤真设备）
# --------------------------------------------------------------------------- #
def test_select_cameras_adaptive_phantom() -> bool:
    print("----- 9b. phantom 索引自适应早停（连续失败即停，真设备不误伤） -----")
    checks: list[tuple[str, bool]] = []
    settings = {"min_width": 1280, "min_height": 720}       # max_index 走默认 6

    def make_probe(mapping, seen):
        def probe(index, **kw):
            seen.append(index)          # 记下被真实探测的编号（entry.order，int）
            return mapping.get(index, ProbeResult(index=index, ok=False, error="打不开"))
        return probe

    # 场景①：有真设备(0号)但打不开 + 其后 phantom 全打不开
    # -> 真设备的失败**不计 miss**，phantom 连续失败 2 次即停（只试 [0, 1, 2]）
    # ⚠️ 必须给一台真设备：`devices` 非空才启用早停（见场景④的反面）。
    seen: list[int] = []
    dev0 = DeviceEntry(order=0, name="real cam", identity="USB\\VID_ZZZZ")
    picks = select_cameras(need=1, devices=[dev0], probe=make_probe({}, seen),
                           reg=Registry(), settings=settings)
    checks.append(("① 真设备失败不计 miss；phantom 连续失败 2 次即停（试 [0, 1, 2]）",
                   seen == [0, 1, 2]))
    checks.append(("① 全失败：无可用返回空 list 不抛", picks == []))

    # 场景②：真设备在 3 号位且可用 -> 一定被试到（不被 phantom 早停跳过）
    seen2: list[int] = []
    dev3 = DeviceEntry(order=3, name="real cam", identity="USB\\VID_XXXX")
    ok3 = {3: ProbeResult(3, True, 1920, 1080, 30.0, std=20.0)}
    picks2 = select_cameras(need=1, devices=[dev3], probe=make_probe(ok3, seen2),
                            reg=Registry(), settings=settings)
    checks.append(("② 真设备(3号)一定被试到", 3 in seen2))
    checks.append(("② 真设备(3号)被挑中", len(picks2) == 1 and picks2[0].index == 3))

    # 场景③：真设备全失败 + 一台被标「无关」的真设备在 5 号 -> 5 号仍被试到
    # （它在 keep/drop 重排后被排到 phantom 之后，正是"早停会不会跳过真设备"的关键）
    seen3: list[int] = []
    dev5 = DeviceEntry(order=5, name="ignored cam", identity="USB\\VID_YYYY")
    reg3 = Registry()
    mark_ignored(reg3, entry=dev5, index=5, verified=True)
    picks3 = select_cameras(need=1, devices=[dev5], probe=make_probe({}, seen3),
                            reg=reg3, settings=settings)
    checks.append(("③ 被标「无关」的真设备(5号)仍被试到（早停不跳过真设备）",
                   5 in seen3))
    checks.append(("③ 全失败无可用时返回空 list", picks3 == []))

    # 场景④：**拿不到 PnP 枚举**（devices=[]）时早停必须**禁用** ——
    # 此刻"编号不在清单里"不等于"编号不存在"（非 Windows / 枚举失败会退化成
    # "按索引使用"）。反例：devices=[] 而 #2 实际可开；若早停生效则 seen==[0,1]，
    # 永远选不到 2 号。这条判据就是 Critic 抓到的真回归的守护。
    seen4: list[int] = []
    ok2 = {2: ProbeResult(2, True, 1920, 1080, 30.0, std=20.0)}
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        picks4 = select_cameras(need=1, devices=[], probe=make_probe(ok2, seen4),
                                reg=Registry(), settings=settings)
    checks.append(("④ devices=[] 时早停被禁用：#2 仍被试到（seen 含 2）",
                   2 in seen4))
    checks.append(("④ devices=[] 时 #2 被挑中（picks == [2]，不是空）",
                   len(picks4) == 1 and picks4[0].index == 2))

    # ⑤ 有真设备但全失败 + phantom #0 能开：phantom 不是物理相机，
    #    必须打印警告并跳过，不能静默选中（vivo/DSHOW-only 的入口）。
    seen5: list[int] = []
    dev1 = DeviceEntry(order=1, name="real cam", identity="USB\\VID_REAL")
    ok0 = {0: ProbeResult(0, True, 1280, 720, 30.0, std=20.0)}
    buf5 = io.StringIO()
    with redirect_stdout(buf5):
        picks5 = select_cameras(need=1, devices=[dev1],
                                probe=make_probe(ok0, seen5),
                                reg=Registry(), settings=settings)
    checks.append(("⑤ phantom 能开也不许自动选中", picks5 == []))
    checks.append(("⑤ phantom 成功必须打印「不是枚举到的物理相机」",
                   "不是枚举到的物理相机" in buf5.getvalue()))

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
    both = role(identity="USB\\B", index=2, verified=True, capture_index=2)
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


def test_cached_mode_respects_requested_resolution() -> bool:
    """第 13 节：**缓存里的分辨率不许盖过这次显式请求的**（2026-10-14 加）。

    ## 守的是一个会让标定**静默失败**的真缺陷

    缓存按**编号**存"上次探测出的后端 + 分辨率"，本意是省时间。
    但它原来只校验 `v` 与后端，**不看这次请求的分辨率**。于是
    `calibrate_stereo.py --width 1920 --height 1080` 会拿到一条 `640x480` 的缓存去开：

    ```
    [相机 1] 用上次记住的模式打开：DSHOW 640x480
    相机 1: 640x480 @ 24.4 fps          ← 而相机 0 是 1920x1080
    ! 两路分辨率不一致（(640, 480) vs (1920, 1080)）：标定要求两路完全一致
    ```

    **标定要求两路分辨率完全一致**，所以这一条就是"白跑一轮"。

    而本机还有更硬的一条（受控实测，每档两轮）：**只有请求 `1920x1080` 时两路才一致**
    —— `#0` 无视请求恒给 1920x1080，`#1` 尽量满足请求，其余每一档都对不上。
    已写进 `docs/真机标定操作清单.md` §「要收多少对」。

    ## 判据为什么不依赖真机

    用**子类替身**：`_raw_open` 重写成"记一笔调用、返回 `None`（打不开）"。
    于是 —— **替身被调用 = 走了缓存分支**（那条分支只调 `_raw_open`，
    不自己开设备）；返回 `None` 后构造函数去搜索，最终抛错，
    测试**从头到尾不碰真设备**，在任何机器上都确定性。
    这比"起个真相机看看"可靠得多，也是本项目坑 24 要求的写法：
    替身测试必须能断言"替身真的走到了那一步"。
    """
    print("----- 13. 缓存分辨率不许盖过显式请求 -----")
    import cv2 as _cv2
    from sphere_tracker import sources as _src

    checks: list[tuple[str, bool]] = []
    cache_path = _src._MODE_CACHE
    backup = cache_path.read_text(encoding="utf-8") if cache_path.exists() else None
    key = 991                          # 几乎不可能真存在的索引

    calls: list[tuple[int, int]] = []
    opened_indices: list[int] = []

    class _FakeCap:
        """够 `CameraSource` 走完"打开成功"那条路的最小替身。"""

        def __init__(self, args):                            # noqa: ANN001
            # `cv2.VideoCapture(index, api)` -> args = (idx, api)
            self._idx, self._api = (list(args) + [None, None])[:2]
            self._wh = (0, 0)

        def isOpened(self) -> bool:
            return True

        def set(self, prop, val=None):                       # noqa: ANN001
            import cv2 as _c
            if prop == _c.CAP_PROP_FRAME_WIDTH:
                self._wh = (int(val), self._wh[1])
            elif prop == _c.CAP_PROP_FRAME_HEIGHT:
                self._wh = (self._wh[0], int(val))
                # 每次**请求分辨率**都记一笔：这就是"它打算用哪个分辨率开"
                calls.append(self._wh)
            return True

        def get(self, prop):                                 # noqa: ANN001
            import cv2 as _c
            if prop == _c.CAP_PROP_FRAME_WIDTH:
                return self._wh[0] or 640
            if prop == _c.CAP_PROP_FRAME_HEIGHT:
                return self._wh[1] or 480
            return 30.0

        def read(self):
            import numpy as _np
            return True, _np.full((480, 640, 3), 128, _np.uint8), 0.0

        def release(self) -> None:
            pass

    def build(want_w: int, want_h: int) -> list[tuple[int, int]]:
        """构造真的 `CameraSource`，但把 OpenCV 的设备打开整个替掉。

        为什么替在**这一层**：`_open_attempt()` 走 `_raw_open()`，而**搜索循环
        直接调 `cv2.VideoCapture`** —— 只替 `_raw_open` 会漏掉搜索那一段，
        于是搜索仍去碰真设备（实测：`[相机 991] 试 DSHOW @ 1920x1080…`
        → `DSHOW 整个后端打不开`，`calls` 为空，判据变成假红）。
        替在 `cv2.VideoCapture` 上，缓存分支与搜索分支**一起**被覆盖。
        """
        calls.clear()
        opened_indices.clear()
        real_vc = _src.cv2.VideoCapture

        def _factory(*a, **k):                               # noqa: ANN002, ANN003
            opened_indices.append(int(a[0]) if a else -1)
            return _FakeCap(a)

        _src.cv2.VideoCapture = _factory                     # type: ignore[assignment]
        try:
            try:
                _src.CameraSource(index=key, width=want_w, height=want_h, fps=30,
                                  backend=_cv2.CAP_DSHOW, optimize=True)
            except Exception:                                # noqa: BLE001
                pass
        finally:
            _src.cv2.VideoCapture = real_vc                   # type: ignore[assignment]
        return list(calls)

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({
            str(key): {"v": 2, "backend": "DSHOW", "width": 640, "height": 480,
                       "actual_width": 640, "actual_height": 480, "fps": 30.0},
        }), encoding="utf-8")

        hits = build(640, 480)
        miss = build(1920, 1080)
        print(f"  请求 640x480（与缓存相同）  -> 按这些分辨率依次尝试：{hits[:4]}")
        print(f"  请求 1920x1080（与缓存不同）-> 按这些分辨率依次尝试：{miss[:4]}")
        # ⚠️ 判据**改过两版**，两版都错在同一个地方 —— 记下来免得再犯：
        #
        # 第一版：`len(miss) == 0`（"跳过缓存就该一次都不调"）。
        #   变异测试当场证伪：守卫失效后搜索路径**照样会调** `_raw_open`，
        #   只是**先用缓存里的 640x480 开一次**，于是 `len(miss) > 0`，
        #   注入坏代码**不报红**（坑 36：先分清"判据弱"还是"注入没生效"）。
        # 第二版：`miss[:1] == [(1920, 1080)]` —— **恒假**。因为那版替身永远返回
        #   `None`（"打不开"），搜索路径**抛错退出**，`miss` 是**空的**。
        #   第一次跑就报红，而代码其实是对的（**假红**）。
        #
        # 最终版：替身改成**会成功**（`_FakeCap`），于是调用序列真的等于
        #   "它打算用哪些分辨率去开"。判据只盯**第一个**：
        #   守卫生效 -> 第一个就是请求值；守卫失效 -> 第一个是缓存值。
        # 两版错误的教训是同一句：**判据要盯住"决定用哪个分辨率"，
        # 而不是"函数被调了几次"。**
        checks += [
            ("请求与缓存**相同**时走缓存：第一个尝试就是缓存值 640x480",
             hits[:1] == [(640, 480)]),
            ("请求与缓存**不同**时：第一个尝试是**请求值** 1920x1080（缓存没盖过它）",
             miss[:1] == [(1920, 1080)]),
            ("**不允许**出现「先拿缓存的 640x480 试一次」（那说明守卫没生效）",
             (640, 480) not in miss),
            ("替身确实被用上了（只碰过 991 这个不存在的索引，没碰真相机）",
             bool(opened_indices) and set(opened_indices) == {key}),
        ]
    finally:
        if backup is None:
            cache_path.unlink(missing_ok=True)
        else:
            cache_path.write_text(backup, encoding="utf-8")

    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_identify_open_preview_shape() -> bool:
    """第 14 节：`cameras.py identify` 的 `_open_preview` **两个分支返回形状必须一致**。

    ## 守的是一个把整个 identify 打崩的真缺陷（2026-10-15）

    用户跑「相机设备管理」直接崩：

        File "apps/tools/cameras.py", line 197, in _preview_and_assign
            src, backend_used = _open_preview(index)
        TypeError: cannot unpack non-iterable NoneType object

    原因：我给 `_open_preview` 加后端回退时，把主返回改成 `(src, 名字)`
    两元组，**却漏了"打不开"那条早退分支**（它仍是单值 `return None`）。
    于是**恰好在那台开不了的相机上**崩掉 —— 而"有相机开不了"正是这个工具
    存在的意义，所以这条路径是**常态，不是边缘**。

    ## 判据

    穷举 `_open_preview` 的**每一条返回路径**，断言形状都是两元组。
    用替身（不碰真设备）：把 `pick_backend_for_device` 与 `CameraSource`
    分别换成"挑不到后端"和"打不开"的版本，两条路径都要走到。
    """
    print("----- 14. identify 的 _open_preview 返回形状 -----")
    import importlib.util

    import sphere_tracker.camera_registry as _cr

    cam_path = ROOT / "apps" / "tools" / "cameras.py"
    spec = importlib.util.spec_from_file_location("cam_tool_shape", cam_path)
    cam = importlib.util.module_from_spec(spec)
    sys.modules["cam_tool_shape"] = cam
    assert spec.loader is not None
    spec.loader.exec_module(cam)

    checks: list[tuple[str, bool]] = []
    import cv2 as _cv2

    # ① `_open_preview` 已经**不再挑后端**（2026-10-18 改：挑后端会让标签贴错）。
    #    所以"挑不到后端"这条路径不存在了 —— 直接断言这一点，避免判据过期后
    #    变成永远无法触发的死断言。
    #
    #    ⚠️ 用 **AST** 查"有没有被调用"，而不是 `in src_text`：
    #    `cameras.py` 的**注释里**就写着这个名字（讲这段历史），
    #    字符串匹配会误判成"还在用"——判据本身必须长在语法树上才可靠。
    import ast

    cam_text = cam_path.read_text(encoding="utf-8")
    calls: list[str] = []
    for node in ast.walk(ast.parse(cam_text)):
        if isinstance(node, ast.Call):
            fn = node.func
            name = getattr(fn, "id", None) or getattr(fn, "attr", None)
            if name:
                calls.append(name)
    checks.append(("`_open_preview` 不再按索引挑后端（AST：没有 pick_backend_for_device 调用）",
                   "pick_backend_for_device" not in calls))

    # ② CameraSource 打不开（RuntimeError）-> 必须返回两元组，不能是单值 None
    real_src = cam.CameraSource
    try:
        def _boom(*a, **k):                               # noqa: ANN002, ANN003
            raise RuntimeError("（替身）打不开")

        cam.CameraSource = _boom                            # type: ignore[assignment]
        got2 = cam._open_preview(0, _cv2.CAP_DSHOW)
        ok2 = isinstance(got2, tuple) and len(got2) == 2 and got2[0] is None
        print(f"  ② 打不开 -> 返回 {got2!r}")
    except Exception as exc:                              # noqa: BLE001
        ok2 = False
        print(f"  ② 打不开 -> 抛了 {type(exc).__name__}: {exc}")
    finally:
        cam.CameraSource = real_src                         # type: ignore[assignment]
    checks.append(("CameraSource 抛 RuntimeError 时也返回两元组，不是单值 None",
                   ok2))

    # ③ 静态穷举：所有 `return` 语句要么两元组、要么在 docstring 里
    import re
    body = cam_path.read_text(encoding="utf-8").split("def _open_preview(")[1]
    body = body.split("\ndef ")[0]
    rets = [m.group(1).strip() for m in re.finditer(r"^\s+return\s+(.+)$", body,
                                                    re.MULTILINE)]
    print(f"  ③ _open_preview 的 return 语句：{rets}")
    checks.append(("每一条 return 都是两元组（`X, Y`）",
                   bool(rets) and all("," in r for r in rets)))

    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_usb_port_change_is_diagnosable() -> bool:
    """第 15 节：**换 USB 口**后，报错必须自己说清"是换口，不是相机丢了"。

    ## 守的是用户 2026-10-15 实际踩的那一步

    用户把一个相机换到另一个 USB 口，`cameras.py check` 报：

        [漂移] 「right」(已确认) 解析失败：「right」指定的相机当前不在线
               （期望 身份 USB\\VID_1BCF&PID_2A25&MI_00\\6&212acdc8&0&0000；…）

    **技术上没错，但没告诉用户发生了什么** —— 于是看起来像"相机丢了"。
    真相是 **Windows 的 PnP `InstanceId` 把端口编在里面**：

        USB\\VID_1BCF&PID_2A25&MI_00\\6&212acdc8&0&0000
                                    ^^^^^^^^^^ 这一段就是"插在哪个口"

    换口 → 这一段变（实测 `6&212acdc8` → `7&24e47704`）→ **整条身份变了**。

    ## 顺带钉住第二个缺陷：解析失败必须算危险项

    `cmd_check` 原来遇到解析失败只 `continue`、**不置 danger** ——
    报了一行失败却返回**退出码 0**，而 `cameras.py check --help`
    与用户手册都承诺「有 [危险] 项时退出码为 2」。

    ## 判据（全部用注入数据，不碰真设备）

    - 身份对不上 + 有 1 台同名 → 提示里出现"换"与 `identify`/`pin`
    - 身份对不上 + 同名 0 台 → 提示里出现"不在线"（**不是**换口）
    - 身份对不上 + 同名多台 → 提示里出现"目视确认"（不许替用户猜）
    """
    print("----- 15. 换 USB 口必须可诊断 -----")
    import sphere_tracker.camera_registry as cr

    def dev(order: int, name: str, identity: str, port: str) -> object:
        return cr.DeviceEntry(order=order, name=name, identity=identity,
                              port=port, present=True, source="system")

    # ① 同名 1 台（就是"换了口"的样子）
    stale = cr.RoleSpec(identity="USB\\VID_1BCF&PID_2A25&MI_00\\6&212acdc8&0&0000",
                        port="6&212acdc8&0&0000", name="aoni webcam A25",
                        index=0, verified=True)
    moved = [dev(1, "aoni webcam A25",
                 "USB\\VID_1BCF&PID_2A25&MI_00\\7&24e47704&0&0000",
                 "7&24e47704&0&0000")]
    t1 = cr.explain_role_miss(stale, moved, "right")

    # ② 同名 0 台（真的不在线）
    t2 = cr.explain_role_miss(stale, [dev(2, "Integrated Camera", "USB\\X", "p")], "right")

    # ③ 同名多台（同型号，无法靠名字判断）
    t3 = cr.explain_role_miss(stale, [
        dev(0, "aoni webcam A25", "USB\\A", "6&aaa"),
        dev(1, "aoni webcam A25", "USB\\B", "7&bbb"),
    ], "right")

    # ④ 名字只是**前缀**（right 期望 "aoni webcam A25"，在线的是
    #    "aoni webcam A25 left"）：不许把它当同名候选，更不许建议 pin 过去。
    t4 = cr.explain_role_miss(stale, [
        dev(0, "aoni webcam A25 left", "USB\\L", "6&left"),
    ], "right")

    for label, tips in (("① 同名 1 台", t1), ("② 同名 0 台", t2),
                        ("③ 同名多台", t3), ("④ 只是前缀", t4)):
        print(f"  {label}:")
        for line in tips:
            print(f"      {line[:96]}")

    checks = [
        ("换口场景提到「换」（说清是身份变了，不是相机坏了）",
         any("换" in x for x in t1)),
        ("换口场景给出**可执行**的两条路（identify / pin）",
         any("identify" in x for x in t1) and any("pin" in x for x in t1)),
        ("真不在线时不误报成换口", any("不在线" in x for x in t2)
         and not any("换过 USB 口" in x for x in t2)),
        ("同名多台时要求目视确认（不替用户猜）",
         any("目视确认" in x for x in t3)),
        ("前缀名字不被当成同名候选（不给 pin 过去的建议）",
         not any("pin right port:" in x for x in t4)),
        ("前缀名字明确说「不是同一台」", any("不是同一台" in x for x in t4)),
    ]

    # ④ 非交互报错里也要带上线索（脚本/子进程只看得见这一句）
    try:
        cr.resolve_role("right", reg=cr.Registry(roles={"right": stale}),
                        devices=moved, interactive=False)
        msg = ""
        checks.append(("非交互时应当抛错", False))
    except RuntimeError as exc:
        msg = str(exc)
        checks.append(("非交互时抛 RuntimeError（可被上层接住）", True))
    # ④ 非交互报错里也要带上**线索行**（脚本/子进程只看得见这一句）
    #
    # ⚠️ 判据**改过一次**：第一版是 `"换" in msg and "InstanceId" in msg`，
    # 而报错末尾还有一句**硬编码**的"注意：Windows 的 InstanceId 把端口编在里面"，
    # 于是把线索整段删掉**照样通过** —— 变异测试当场证明那条判据是空的。
    # 现在盯**线索行本身**（`·` 开头、且含可执行命令），那是 `extra` 独有的。
    tip_lines = [ln for ln in msg.splitlines() if ln.strip().startswith("·")]
    print(f"  ④ 非交互报错里的线索行 {len(tip_lines)} 条：")
    for ln in tip_lines:
        print(f"      {ln.strip()[:92]}")
    checks.append(("非交互报错里带上了线索行（`·` 开头；脚本只看得见这句）",
                   bool(tip_lines)))
    checks.append(("线索行里给了**可执行**命令（identify / pin）",
                   any(("identify" in ln or "pin" in ln) for ln in tip_lines)))

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


def test_identify_never_switches_backend() -> bool:
    """第 16 节：`identify` **整场只用一个后端**，绝不在索引之间切换。

    ## 守的是用户 2026-10-18 报的"生产级 bug"

    > 运行相机管理选择的时候，把我的 aoni 相机识别为了 aoni 相机和 integrated
    > 相机，也就是识别了两次

    **根因**：上一版为了绕开"某个后端打不开某台相机"，给**每个索引单独挑后端**
    （先 MSMF，不成再 DSHOW）。而 `#N` 的标签来自 **PnP 枚举**，
    `cv2.VideoCapture(N, backend)` 用的是**后端自己的枚举**。本机实测
    （`outputs/_handoff_read/probe_index_map.py`）：

        PnP 标签：#0 aoni-left   #1 aoni-right   #2 Integrated   #3 Integrated IR
        MSMF     ：#0 有图       #1 有图        #2 纯黑(被遮)   #3 打不开   ← 一致
        DSHOW    ：#0 有图       #1 **纯黑**    #2 **有图**     #3 打不开   ← 错位

    所以任一索引回退到 DSHOW，**标签就贴到了另一台相机上** ——
    用户看到"aoni 被显示成 Integrated"，据此绑定就绑错、标定废掉。

    ## 判据（行为级，不只看源码）

    1. `cameras.py` **不再使用** `pick_backend_for_device`（它是"按索引换后端"的入口）；
    2. `_open_preview(index, backend)` 用的就是**传进来的那个**后端；
    3. 该后端打不开时**返回失败、不再试别的**（替身只应被调用一次）；
    4. `identify` 的 `--backend` 默认 `None`（= 用 `INDEX_BACKEND`）。
    """
    print("----- 16. identify 整场只用一个后端 -----")
    import cv2

    import sphere_tracker.sources as _src
    from sphere_tracker import camera_registry as _cr

    cam_path = ROOT / "apps" / "tools" / "cameras.py"
    import importlib.util

    spec = importlib.util.spec_from_file_location("cam_tool_sw", cam_path)
    cam = importlib.util.module_from_spec(spec)
    sys.modules["cam_tool_sw"] = cam
    assert spec.loader is not None
    spec.loader.exec_module(cam)

    checks: list[tuple[str, bool]] = []

    # （"不再按索引挑后端"这条判据在**本文件第 14 节**用 AST 查过了 ——
    #  这里不重复，避免同一事实两套判据、改一处忘一处。）

    # ---- 行为：传什么后端就用什么后端 ----
    real_src = cam.CameraSource
    used: list[int] = []

    class _Fake:
        def __init__(self, **kw):                         # noqa: ANN003
            used.append(kw.get("backend"))

        def release(self) -> None:
            pass

    try:
        cam.CameraSource = _Fake                        # type: ignore[assignment]
        got = cam._open_preview(3, cv2.CAP_DSHOW)
        ok_used = used == [cv2.CAP_DSHOW]
        ok_name = got[0] is not None and got[1] == "DSHOW"
        print(f"  ① 传 DSHOW -> 实际用 {used}（期望 [{cv2.CAP_DSHOW}]），返回名 {got[1]!r}")
        checks.append(("① `_open_preview` 只用**传入的**后端（不自己挑）", ok_used))
        checks.append(("① 返回的后端名与实际一致", ok_name))

        # ---- 打不开时**绝不**换后端（换 = 重复/贴错，2026-10-18 实测） ----
        #
        # 三次反复都记在这里：换后端 → 标签贴错；不换但隐藏 → 相机消失；
        # 回退只为显示 → **同一台出现两次**。所以这里钉死"只尝试 1 次"。
        used.clear()

        def _boom(**kw):                                # noqa: ANN003
            used.append(kw.get("backend"))
            raise RuntimeError("（替身）打不开")

        cam.CameraSource = _boom                        # type: ignore[assignment]
        got2 = cam._open_preview(3, cv2.CAP_MSMF)
        ok_once = len(used) == 1
        ok_none = got2 == (None, None)
        print(f"  ② 打不开 -> 尝试 {len(used)} 次（期望 1），返回 {got2!r}")
        checks.append(("② 打不开时**只试 1 次**（换后端会导致重复/贴错）", ok_once))
        checks.append(("② 打不开时返回 (None, None)（形状一致、不崩）", ok_none))
    finally:
        cam.CameraSource = real_src                     # type: ignore[assignment]

    # ---- 默认后端 ----
    p = cam.build_parser()
    ns_ = p.parse_args(["identify"])
    checks.append(("`identify --backend` 默认 None（= 用 INDEX_BACKEND）",
                   getattr(ns_, "backend", "缺失") is None))
    checks.append(("默认后端就是 INDEX_BACKEND，且它确实是 MSMF（唯一与 PnP 一致的）",
                   _src.INDEX_BACKEND == cv2.CAP_MSMF
                   and _src.INDEX_BACKEND_NAME == "MSMF"))

    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_preview_label_admits_it_is_a_guess() -> bool:
    """第 17 节：预览标签**必须自己承认是推测**，不许把猜测写成事实。

    ## 守的是用户 2026-10-18 的第二次纠正

    > 你的相机管理的视频画面的内容是不是根本不是通过相机的名字来设置的?
    > 为什么我的命名为 right 的相机却在画面上是 left?

    **他说得对，而且这条比"换后端"更根本。**

    `cv2.VideoCapture(N, backend)` **只能按编号打开，名字不参与**；
    窗口上 `#N <名字>` 是"编号 N ↔ 设备列表第 N 台"这个**假设**的产物。

    而那条假设的证据链有个缺口：

    | 验证过的 | 结论 |
    | -------- | ---- |
    | aoni vs Integrated（MSMF `#2` 黑 = PnP `#2` 内置） | 成立 |
    | **两台一模一样的 aoni 谁在 `#0`/`#1`** | **从未验证** |

    两台 aoni 同 VID/PID、同分辨率、同画面特征 —— **软件层面不可区分**，
    所以这个顺序**原理上无法由 OpenCV 证明**。标签如果写成事实，
    用户就会按错的标签绑角色（"命名为 right 却在画面上是 left"就是这么来的）。

    ## 判据

    - 标签里必须出现「推测」（把不确定性写在用户看得见的地方）；
    - 标签里仍要有名字与端口（否则用户没有任何线索可用）。
    """
    print("----- 17. 预览标签必须自认「推测」 -----")
    import importlib.util

    cam_path = ROOT / "apps" / "tools" / "cameras.py"
    spec = importlib.util.spec_from_file_location("cam_tool_label", cam_path)
    cam = importlib.util.module_from_spec(spec)
    sys.modules["cam_tool_label"] = cam
    assert spec.loader is not None
    spec.loader.exec_module(cam)

    import sphere_tracker.camera_registry as cr

    entry = cr.DeviceEntry(order=0, name="aoni webcam A25 left",
                           identity="USB\\X", port="6&abc&0&0000")
    label = cam._preview_label(0, entry)
    print(f"  标签 = {label!r}")

    blank = cr.DeviceEntry(order=3)
    label2 = cam._preview_label(3, blank)
    print(f"  无名字时 = {label2!r}")

    checks = [
        ("标签写明「推测」（不许把猜测写成事实）", "推测" in label),
        ("标签仍带名字（用户需要线索）", "aoni webcam A25 left" in label),
        ("标签仍带端口（名字不可靠时的第二线索）", "6&abc&0&0000" in label),
        ("没有名字时给出占位而不是空白", "(无名字)" in label2),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_avoid_rules_are_configurable() -> bool:
    """第 18 节：`is_avoided()` 的两条规则**都来自配置**，不再写死在代码里。

    ## 守的是什么（2026-10-18 的"参数化"改动）

    原来 `is_avoided()` 里硬编码了 `"infrared" in low or re.search(r"\\bir\\b", low)`
    —— 用户换一台名字里带 `IR` 的**正常**相机就会被永久误伤，
    而且**只能改源码**才能关掉。

    现在两条规则都在 `settings` 里：`avoid_names`（子串）、
    `avoid_name_patterns`（正则）。

    ## 判据

    - 默认：`infrared` / 独立 `IR` 命中；普通名字不命中；
    - `avoid_name_patterns: []` → 红外启发式**整体关掉**（其余规则仍生效）；
    - 自定义正则确实生效；
    - **非法正则**：跳过那一条并警告，**不抛异常**、不连累其它规则。
    """
    print("----- 18. avoid 规则可配置 -----")
    import sphere_tracker.camera_registry as cr

    checks: list[tuple[str, bool]] = [
        ("默认：`Some IR Camera` 判为不可用（原写死的行为保留）",
         cr.is_avoided("Some IR Camera")),
        ("默认：`MyInfraredThing` 判为不可用", cr.is_avoided("MyInfraredThing")),
        ("默认：普通外接名字**不**误伤", not cr.is_avoided("aoni webcam A25 left")),
        ("默认：内置相机名字仍命中 avoid_names", cr.is_avoided("Integrated Camera")),
        ("`avoid_name_patterns: []` 关掉红外启发式",
         not cr.is_avoided("Some IR Camera", {"avoid_name_patterns": []})),
        ("关掉红外后，`avoid_names` 仍然生效（两套规则互不牵连）",
         cr.is_avoided("Integrated Camera", {"avoid_name_patterns": []})),
        ("自定义正则生效（只排除以 left 结尾）",
         cr.is_avoided("cam left", {"avoid_name_patterns": [r"left$"]})
         and not cr.is_avoided("cam right", {"avoid_name_patterns": [r"left$"]})),
    ]

    try:
        got = cr.is_avoided("Integrated Camera",
                            {"avoid_name_patterns": ["[unclosed"]})
        ok = got is True
        print(f"  非法正则 -> 返回 {got}（未抛异常）")
    except Exception as exc:                              # noqa: BLE001
        ok = False
        print(f"  非法正则 -> 抛了 {type(exc).__name__}: {exc}")
    checks.append(("非法正则不抛异常、且不连累 `avoid_names`", ok))

    # ---- 虚拟摄像头（2026-11-12）：与「内置/红外」**不同类**，不参与绑定 ----
    # 规则同样来自配置 `settings.virtual_names`（子串、大小写不敏感）。
    checks += [
        ("虚拟：vivo 命中（本机实测的虚拟摄像头）",
         cr_mod.is_virtual("vivo 办公套件虚拟摄像头")),
        ("虚拟：OBS Virtual Camera 命中", cr_mod.is_virtual("OBS Virtual Camera")),
        ("虚拟：大小写不敏感（mAnYcAm）", cr_mod.is_virtual("mAnYcAm Video Source")),
        ("虚拟：真实外接相机**不**误伤", not cr_mod.is_virtual("aoni webcam A25")),
        ("虚拟：内置相机不归这一类（它仍是真相机）",
         not cr_mod.is_virtual("Integrated Camera")),
        ("虚拟：`virtual_names: []` 可以整条关掉（配置说了算）",
         not cr_mod.is_virtual("OBS Virtual Camera", {"virtual_names": []})),
        ("虚拟：自定义名单生效",
         cr_mod.is_virtual("My Fake Cam", {"virtual_names": ["fake cam"]})),
        ("is_bindable：虚拟 False / 真实 True",
         not cr_mod.is_bindable("vivo 办公套件虚拟摄像头")
         and cr_mod.is_bindable("aoni webcam A25")),
        ("清单里标出「虚拟,不参与绑定」（与[内置/红外]分开写）",
         "[虚拟,不参与绑定]" in format_device_table(
             [dev(0, "OBS Virtual Camera", "SW\\OBS")], settings={})),
        ("phantom 索引警告写明「这不是枚举到的物理相机」",
         "不是枚举到的物理相机" in cr_mod.phantom_index_warning(5)),
    ]

    # 数据层硬拒绝：虚拟摄像头**不能**被绑成 left/right（任何入口都绕不过去）。
    virt_dev = dev(5, "vivo 办公套件虚拟摄像头", "SW\\VIVO")
    try:
        set_role(Registry(settings={}), "left", entry=virt_dev, verified=False)
        checks.append(("set_role 拒绝虚拟摄像头（ValueError）", False))
    except ValueError as exc:
        checks.append(("set_role 拒绝虚拟摄像头（ValueError），并说清原因",
                       "虚拟摄像头" in str(exc) and "不参与绑定" in str(exc)))
    try:
        set_role(Registry(settings={"virtual_names": []}), "left",
                 entry=virt_dev, verified=False)
        checks.append(("`virtual_names: []` 后同一台就能绑（规则来自配置）", True))
    except ValueError:
        checks.append(("`virtual_names: []` 后同一台就能绑（规则来自配置）", False))

    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_identify_open_interruptible() -> bool:
    """第 21 节：`_open_preview_interruptible` 在**等待首帧期间**就能被 Esc/s 打断。

    ## 守的是用户 2026-10-31 报的"相机管理依然卡死"（P4）

    旧代码里 `CameraSource(...)` 在**构造函数里**开相机 + 等首帧（上限 30 秒），
    而按键循环在 `_preview_and_assign` 里、位于**首帧到达之后**才跑 ——
    所以那 30 秒内按 Esc 完全无效，用户只能理解成"卡死"。

    本判据注入：① 一个"睡眠 1.5 秒才返回"的假 `_open_preview`；
    ② 一个"第 3 次调用返回 27(Esc)"的假 `key_poll`。改后应在 ~0.1 秒内返回
    `(None, None)`（远早于 1.5 秒）。

    ⚠️ **改前对照（Q3）**：改前没有这条可中断通道，这个调用会**等满 1.5 秒**
    才返回（因为它就是直接 `_open_preview(...)` 阻塞），`< 0.8s` 的断言会红。
    """
    print("----- 21. identify 开预览可中断（等待期按键即可放弃）-----")
    try:
        from apps.tools import cameras as _cm
    except Exception as exc:                              # noqa: BLE001
        print(f"  [跳过] 无法导入 apps/tools/cameras.py：{exc}")
        return True

    import time as _time

    class _FakeSrc:
        def __init__(self) -> None:
            self.released = False

        def release(self) -> None:
            self.released = True

    checks: list[tuple[str, bool]] = []
    real_open = _cm._open_preview
    fake = _FakeSrc()

    def _slow_open(index, backend, *, enumerated=False, timeout_s=None):  # noqa: ANN001
        _time.sleep(1.5)
        return fake, "MSMF"

    try:
        # ---- 场景①：第 3 次轮询按 Esc -> 立即返回 (None, None) ----
        _cm._open_preview = _slow_open                  # type: ignore[assignment]
        calls = {"n": 0}

        def _poll():
            calls["n"] += 1
            return 27 if calls["n"] == 3 else None

        t0 = _time.perf_counter()
        got = _cm._open_preview_interruptible(0, 0, enumerated=True,
                                              timeout_s=5.0, key_poll=_poll)
        elapsed = _time.perf_counter() - t0
        # 主判据改成**逻辑不变量**：Esc 在 `_poll()` 第 3 次返回 27，函数就该
        # 在第 3 次轮询那一下退出；若 Esc 被忽略，循环会一直跑到后台线程
        # 结束（假实现 1.5s），轮询次数会多出几十次。这比墙钟阈值更能表达意图。
        #
        # 为什么不再用 0.8s 这个绝对阈值（2026-11-15 全量跑批实测）：主循环每轮
        # 会调 `cv2.imshow + waitKey(1)` 画进度窗，整套跑批时 GUI 调度偶发卡顿，
        # 单跑 0.21~0.23s 的同一段代码会超过 0.8s，把 26/26 打成 25/26 ——
        # 那是判据在测 Windows 的消息循环，不是在测中断逻辑。
        # 保留一条**宽松**护栏 `elapsed < 1.4`：它仍小于假实现的 1.5s，所以
        # 「Esc 被认了却仍 join 后台线程」这种真回归照样抓得到。
        ok1 = (got == (None, None) and calls["n"] <= 5 and elapsed < 1.4)
        print(f"  ① 按 Esc -> 返回 {got!r}，用时 {elapsed:.3f}s，"
              f"轮询 {calls['n']} 次（假实现要 1.5s）")
        checks.append(("等待期按 Esc 在第 3 次轮询即返回 (None, None)，"
                       "且没等满假实现的 1.5s", ok1))

        # ---- 场景②：后台稍后把相机会开出来 -> 必须被 release()（不留悬空）----
        released, waited = False, 0.0
        while waited < 3.0:
            if fake.released:
                released = True
                break
            _time.sleep(0.1)
            waited += 0.1
        print(f"  ② 后台线程结束后 fake.released = {fake.released}（等了 {waited:.1f}s）")
        checks.append(("用户跳过 -> 后台开出图后自动 release()（不留悬空相机）", released))

        # ---- 场景③：不按任何键 + 立即返回 -> 正常拿到 src，且不被误 release ----
        fake2 = _FakeSrc()

        def _fast_open(index, backend, *, enumerated=False, timeout_s=None):  # noqa: ANN001
            return fake2, "MSMF"

        _cm._open_preview = _fast_open                  # type: ignore[assignment]
        got3 = _cm._open_preview_interruptible(0, 0, enumerated=True,
                                               timeout_s=5.0,
                                               key_poll=lambda: None)
        ok3 = (got3 == (fake2, "MSMF")) and (fake2.released is False)
        print(f"  ③ 不按键 -> 拿到 fake={got3[0] is fake2}，"
              f"fake2.released = {fake2.released}")
        checks.append(("不按键 -> 正常拿到 src，且不被误 release", ok3))

        # ---- 场景④：**用户主动跳过**必须记进 `skipped_out`（收尾据此不说
        #      "没能打开、去换个后端试试"）----
        skip_list: list[int] = []
        calls2 = {"n": 0}

        def _poll2():
            calls2["n"] += 1
            return 27 if calls2["n"] == 3 else None

        _cm._open_preview = _slow_open                  # type: ignore[assignment]
        got4 = _cm._open_preview_interruptible(7, 0, enumerated=True,
                                               timeout_s=5.0, key_poll=_poll2,
                                               skipped_out=skip_list)
        checks.append(("④ 用户跳过 -> 记进 skipped_out（供收尾区分「跳过」与「打不开」）",
                       got4 == (None, None) and skip_list == [7]))

        # ---- 场景⑤：**打不开**（非用户跳过）不得进 `skipped_out`，且
        #      `_preview_and_assign` 要把它记进 `failed_out` ----
        skip_list2: list[int] = []

        def _fail_open(index, backend, *, enumerated=False, timeout_s=None):  # noqa: ANN001
            return None, None

        _cm._open_preview = _fail_open                  # type: ignore[assignment]
        _cm._open_preview_interruptible(3, 0, enumerated=True, timeout_s=5.0,
                                        key_poll=lambda: None,
                                        skipped_out=skip_list2)
        checks.append(("⑤ 打不开（非跳过）-> skipped_out 为空", skip_list2 == []))

        failed: list[int] = []
        entry5 = DeviceEntry(order=3, name="real cam", identity="USB\\VID_FAIL")
        role5 = _cm._preview_and_assign(3, entry5, "", backend=0,
                                        first_frame_timeout=5.0,
                                        key_poll=lambda: None,
                                        failed_out=failed)
        checks.append(("⑤ 打不开 -> `_preview_and_assign` 记进 failed_out"
                       "（收尾才对它说「没能打开」）",
                       role5 is None and failed == [3]))

        # ---- 场景⑥：**用户主动跳过**不得进 `failed_out`（本次修复的语义本体：
        #      跳过 ≠ 打不开；回退这条判据，收尾又会把跳过说成"没能打开"）----
        failed2: list[int] = []
        calls3 = {"n": 0}

        def _poll3():
            calls3["n"] += 1
            return 27 if calls3["n"] == 3 else None

        _cm._open_preview = _slow_open                  # type: ignore[assignment]
        entry6 = DeviceEntry(order=9, name="real cam", identity="USB\\VID_SKIP")
        role6 = _cm._preview_and_assign(9, entry6, "", backend=0,
                                        first_frame_timeout=5.0,
                                        key_poll=_poll3, failed_out=failed2)
        checks.append(("⑥ 用户跳过 -> failed_out 保持空（不误报「打不开」）",
                       role6 is None and failed2 == []))
    finally:
        _cm._open_preview = real_open                   # type: ignore[assignment]

    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_check_detects_both_roles_on_same_camera() -> bool:
    """第 22 节：`cameras.py check` **必须自己发现「左右撞车」**。

    ## 守的是用户 2026-10-31 被卡住的死循环

        `pair_source.py` 报「左右两路解析到同一个相机」→ 报错让用户跑 check
        → check **逐角色**比对，两个角色各自都"一致"
        → 输出「未发现危险项」+ **退出码 0** → 用户被弹回原点，还被告知"没事"。

    所以这里钉两件事：

    1. **退出码**：撞车时必须非 0（`check --help` 承诺过"有 [危险] 项时退出码 2"）；
    2. **顺序**：危险的结论必须出现在逐个角色的「[一致]」**之前** —— 否则用户读完
       一屏"一致"就关窗口了，根本看不到后面。

    判据不许自己比 identity/index：那会变成第二份"撞车判据"。这里只断言
    `check` 的输出与退出码，实现上它必须交给 `camera_registry.resolve_roles()`。

    ## 不碰相机

    `cmd_check` 本来就不开相机（只做枚举 + 配置比对），所以这里替换掉
    `list_devices` / `load_registry` 两个入口即可，整段只读假数据。
    """
    print("----- 22. check 必须发现「左右撞车」 -----")
    import importlib.util

    import sphere_tracker.camera_registry as _cr

    cam_path = ROOT / "apps" / "tools" / "cameras.py"
    spec = importlib.util.spec_from_file_location("cam_tool_collide", cam_path)
    cam = importlib.util.module_from_spec(spec)
    sys.modules["cam_tool_collide"] = cam
    assert spec.loader is not None
    spec.loader.exec_module(cam)

    left_dev = dev(0, "aoni webcam A25 left", "USB\\A\\p0", "6&31324e2c&0&0000")
    right_dev = dev(1, "aoni webcam A25 right", "USB\\B\\p1", "6&4489382&0&0000")
    builtin = dev(2, "Integrated Camera", "USB\\INT\\p2", "6&3a5c8ec3&1&0000")
    devs = [left_dev, right_dev, builtin]

    # 撞车：left 与 right 记的**同一台**。
    #
    # ⚠️ 2026-11-08 起必须**直接**构造，不能再调两次 `set_role` —— 现在
    # `set_role` 会把同一台的其它角色摘掉（数据层不变量），用它已经造不出
    # 坏配置了。而"坏配置"仍然会从**别人机器上的旧文件**读进来，所以这条
    # 必须按"历史文件就是这样"的形状构造。
    collide = Registry()
    for role in ("left", "right"):
        collide.roles[role] = RoleSpec(
            identity=left_dev.identity, port=left_dev.port,
            name=left_dev.name, index=0, verified=True,
            # capture_index：2026-11-12 起按角色名解析**必须**有实测值，
            # 否则两条角色都会在"采集索引未实测"那一步被拒绝，撞车就判不出来。
            capture_index=0, location="床")
    # 对照：两台不同（这条照旧用 set_role，顺带确认正常路径没被写坏）
    healthy = Registry()
    set_role(healthy, "left", entry=left_dev, index=0, verified=True,
             capture_index=0, location="床")
    set_role(healthy, "right", entry=right_dev, index=1, verified=True,
             capture_index=1, location="阳台门")
    # 🚨 第三种配置：身份对得上、但 `capture_index` 从未实测。
    # 这是"程序显示的相机 ≠ 真实相机"的根因配置。`check` 必须**报危险**，
    # 而不是崩一个 traceback（用户是照报错来跑 check 的）。
    raw = Registry()
    set_role(raw, "left", entry=left_dev, index=0, verified=True)

    checks: list[tuple[str, bool]] = []
    real_list, real_load = _cr.list_devices, _cr.load_registry
    real_roles = _cr.resolve_roles

    def _fake_roles_raise(*_a, **_k):
        # 判据不许看异常文案：这里把 `resolve_roles` 的报错改成任意无关文本。
        # 若 `check` 靠 `"冲突" in str(exc)` 判断，这一轮就会静默退回 EXIT=0
        # —— 那正是今天这个 bug 的复发方式（改一句文案就重犯，而且没人会发现）。
        raise RuntimeError("duplicate")

    try:
        _cr.list_devices = lambda: list(devs)                  # type: ignore[assignment]

        _cr.load_registry = lambda *a, **k: collide            # type: ignore[assignment]
        buf = io.StringIO()
        with redirect_stdout(buf):
            code_bad = cam.cmd_check(SimpleNamespace())
        out_bad = buf.getvalue()

        _cr.load_registry = lambda *a, **k: healthy            # type: ignore[assignment]
        buf = io.StringIO()
        with redirect_stdout(buf):
            code_ok = cam.cmd_check(SimpleNamespace())
        out_ok = buf.getvalue()

        # 第三轮（在把 resolve_roles 换成必抛之前）：采集索引未实测
        _cr.load_registry = lambda *a, **k: raw                # type: ignore[assignment]
        buf = io.StringIO()
        with redirect_stdout(buf):
            code_raw = cam.cmd_check(SimpleNamespace())
        out_raw = buf.getvalue()

        # 第四/五轮：异常文案改成 "duplicate"（判据必须仍按事实成立）
        _cr.resolve_roles = _fake_roles_raise                   # type: ignore[assignment]
        _cr.load_registry = lambda *a, **k: collide            # type: ignore[assignment]
        buf = io.StringIO()
        with redirect_stdout(buf):
            code_msg = cam.cmd_check(SimpleNamespace())
        out_msg = buf.getvalue()

        _cr.load_registry = lambda *a, **k: healthy            # type: ignore[assignment]
        buf = io.StringIO()
        with redirect_stdout(buf):
            code_msg_ok = cam.cmd_check(SimpleNamespace())
        out_msg_ok = buf.getvalue()
    finally:
        _cr.list_devices, _cr.load_registry = real_list, real_load
        _cr.resolve_roles = real_roles

    print("  撞车时 check 的输出（前 9 行）：")
    for line in out_bad.splitlines()[:9]:
        print(f"      {line[:96]}")
    print(f"  撞车退出码 = {code_bad}；不撞车退出码 = {code_ok}")
    print(f"  采集索引未实测 -> 退出码 = {code_raw}（必须非 0 且不崩）")
    for line in out_raw.splitlines():
        if "采集索引未实测" in line or "identify" in line:
            print(f"      {line.strip()[:96]}")
    print(f"  把异常文案换成「duplicate」后：撞车退出码 = {code_msg}；"
          f"不撞车退出码 = {code_msg_ok}")

    headline = "[危险] left 与 right 解析到同一台相机（#0 aoni webcam A25 left）"
    # 用 find（找不到返回 -1）而不是 index（找不到直接抛）：判据自己要能**报红**，
    # 不许在"守卫被删掉"的情况下先崩成 ValueError —— 那就看不出是哪条判据红的。
    pos_danger = out_bad.find("[危险] left 与 right")
    pos_agree = out_bad.find("[一致]")
    checks += [
        ("撞车 -> 退出码非 0（`check` 承诺过有 [危险] 项时为 2）", code_bad == 2),
        ("撞车 -> 明确报出「left 与 right 解析到同一台相机」并点名那台",
         headline in out_bad),
        ("撞车 -> 说明这两台是双目的前提，并给出两份 pin（不用开相机）",
         "两台相机是双目的前提" in out_bad
         and "pin left" in out_bad and "pin right" in out_bad
         and "不用开相机" in out_bad),
        ("撞车 -> 给出的两个 pin 取值**指向两台不同的相机**（port 各取各的）",
         "port:6&31324e2c&0&0000" in out_bad and "port:6&4489382&0&0000" in out_bad),
        ("撞车 -> 危险结论出现在逐个角色的「[一致]」**之前**（否则用户只看得到一致）",
         pos_danger >= 0 and pos_agree >= 0 and pos_danger < pos_agree),
        ("不撞车 -> 退出码 0 且「未发现危险项」（对照，防止判据永远为红）",
         code_ok == 0 and "未发现危险项" in out_ok and "[危险]" not in out_ok),
        # 下面两条钉死"**不许看异常文案**"：判据一旦退回 `"冲突" in str(exc)`，
        # 第一条立刻红（撞车被漏判），而第二条守住另一头（不许把任何异常都当撞车）。
        ("异常文案改成「duplicate」-> 仍判撞车、退出码 2（判据不看字串）",
         code_msg == 2 and "[危险] left 与 right 解析到同一台相机" in out_msg),
        ("异常文案改了也**不许误报**：不撞车时仍 EXIT 0",
         code_msg_ok == 0 and "未发现危险项" in out_msg_ok),
        # ---- 采集索引未实测：check 必须**报危险**而不是崩 ------------------
        # 用户是照着报错来跑 check 的（`pair_source` 让它去查配置）。这里如果
        # 抛 traceback，他就彻底卡住了 —— 所以要求：非 0 退出码 + 一行能照着做的
        # `[危险] 采集索引未实测`，并且点名 identify。
        ("采集索引未实测 -> check 退出码 2（不是 0，也不是崩）", code_raw == 2),
        ("采集索引未实测 -> 明确打「[危险] 采集索引未实测」",
         "[危险] 采集索引未实测" in out_raw),
        ("采集索引未实测 -> 给出可照做的下一步（identify）",
         "identify" in out_raw),
        ("采集索引未实测 -> 仍然报出该角色的身份是否对上（两件事分开说）",
         "[一致]" in out_raw or "[漂移]" in out_raw),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_no_frame_retries_lowres_and_translates_error() -> bool:
    """第 23 节：「一帧都没有」先补试 640x480；-2147483638 必须翻译成人话。

    ## 守的两个真缺陷（2026-10-31）

    **① 判据被自己吞掉**：`quiet_repeated_grab_warnings()` 为了不被每 10 秒一条
    的 WARN 刷屏，把 `videoio` 的 WARN 压成 ERROR —— 而那条 WARN 里唯一的实质
    信息就是错误码。于是"用户永远看不到判据"。现在报错正文必须把
    `MF_E_VIDEO_RECORDING_DEVICE_INVALIDATED` 的含义与**一条**下一步说出来，
    并且提示"也可能是你自己开了第二个窗口"。

    **② 把"带宽不够"当成"设备死了"**：USB 带宽不足恰恰是"降一档就好"，
    旧代码在一个分辨率上读不到帧就 `break` 掉所有分辨率，于是本来能用的相机
    被判死。现在必须**再试一次 640x480**，且只补这一档（不是全部重试一遍）。

    ## 判据怎么做到"不依赖真机"

    把 `sources.cv2.VideoCapture` 换成替身：一个"永远读不到帧"、一个"只在
    640x480 下给帧"。于是既能断言"补试的是哪一档"，也能断言"补试成功后相机
    真的被救回来了"。全程不打开任何真实设备。
    """
    print("----- 23. 一帧都没有时补试 640x480；错误码翻译 -----")
    import cv2 as _cv2

    from sphere_tracker import sources as _src

    checks: list[tuple[str, bool]] = []

    class _NoFrameCap:
        """能打开、`read()` 永远返回 (False, None)，并记下被请求的分辨率。"""

        def __init__(self, args) -> None:                  # noqa: ANN001
            self._w = 0
            self._h = 0
            self.seen: list[tuple[int, int]] = []

        def isOpened(self) -> bool:
            return True

        def set(self, prop, val=None):                      # noqa: ANN001
            if prop == _cv2.CAP_PROP_FRAME_WIDTH:
                self._w = int(val)
            elif prop == _cv2.CAP_PROP_FRAME_HEIGHT:
                self._h = int(val)
                self.seen.append((self._w, self._h))
            return True

        def get(self, prop):                               # noqa: ANN001
            if prop == _cv2.CAP_PROP_FRAME_WIDTH:
                return self._w or 640
            if prop == _cv2.CAP_PROP_FRAME_HEIGHT:
                return self._h or 480
            return 30.0

        def read(self):
            return False, None

        def release(self) -> None:
            pass

    class _LowresOnlyCap(_NoFrameCap):
        """1920x1080 下读不到帧，640x480 下正常 —— 这就是"带宽不够"的样子。"""

        def read(self):
            import numpy as _np
            if (self._w, self._h) == (640, 480):
                frame = _np.zeros((480, 640, 3), _np.uint8)
                frame[::2] = 255                             # 有噪声，不是全黑
                return True, frame
            return False, None

    made: list[_NoFrameCap] = []

    def factory(kind):
        def _make(*_a, **_k):
            cap = kind(_a)
            made.append(cap)
            return cap
        return _make

    def run(kind):
        made.clear()
        real_vc = _src.cv2.VideoCapture
        _src.cv2.VideoCapture = factory(kind)                # type: ignore[assignment]
        try:
            try:
                return _src.CameraSource(index=993, width=1920, height=1080, fps=30,
                                         backend=_cv2.CAP_MSMF, optimize=False,
                                         search_budget_s=5.0,
                                         first_frame_timeout_s=1.0), ""
            except RuntimeError as exc:
                return None, str(exc)
        finally:
            _src.cv2.VideoCapture = real_vc                   # type: ignore[assignment]

    # ---- ① 永远读不到帧：补试 640x480，只补一档 ----
    _src_unused, msg = run(_NoFrameCap)
    tries = [wh for cap in made for wh in cap.seen]
    print(f"  ① 实际尝试的分辨率序列：{tries}")
    print("  ① 报错前 8 行：")
    for line in msg.splitlines()[:8]:
        print(f"      {line[:96]}")
    checks += [
        ("一帧都没有 -> 紧接着补试的就是 640x480（不是试完所有档才轮到它）",
         tries[:2] == [(1920, 1080), (640, 480)]),
        ("补试**只有一档**（1280x720 没被试，否则又变成「卡死」）",
         (1280, 720) not in tries),
        ("报错写明错误码的含义（MF_E_VIDEO_RECORDING_DEVICE_INVALIDATED）",
         "MF_E_VIDEO_RECORDING_DEVICE_INVALIDATED" in msg),
        ("报错给出**一条**下一步：先关占用它的窗口/程序",
         "下一步（只做这一条）" in msg),
        ("报错提示「也可能是你自己开了第二个窗口」（P2-7）",
         "第二个本程序窗口" in msg),
        ("报错说明已试过 640x480 这一档（结论有依据）", "640x480" in msg),
    ]

    # ---- ② 只认 640x480：必须被救回来（补试的全部意义） ----
    src2, msg2 = run(_LowresOnlyCap)
    tries2 = [wh for cap in made for wh in cap.seen]
    print(f"  ② 实际尝试的分辨率序列：{tries2}")
    ok2 = src2 is not None
    if src2 is not None:
        print(f"  ② 救回来了：{src2.info.width}x{src2.info.height}，"
              f"resolution_fallback={src2.resolution_fallback}")
        src2.release()
    checks += [
        ("一帧都没有时，若 640x480 能出帧 -> 相机**被救回来**（旧逻辑会直接判死）",
         ok2),
        ("救回来时用的是 640x480 那一档", ok2 and tries2[:2] == [(1920, 1080), (640, 480)]),
        ("救回来时标记 resolution_fallback（上层能说出「降了一档」）",
         ok2 and src2.resolution_fallback is True),
    ]

    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_same_camera_message_blames_config_not_cable() -> bool:
    """第 24 节：左右撞车的报错必须说「配置错」，不许再推荐 `--backend dshow`。

    ## 守的是用户 2026-10-31 的真实处境

    报错原文让用户"确认角色绑定的相机在不在线（换过 USB 口）"，
    并推荐 `--left 0 --right 1 --backend dshow`。这两条在本机**都错**：

      · 两台 aoni **都在线**（`list` 里都枚举得到，一台 1.2 s 出图）；
      · DSHOW 的 `#1` 是**笔记本内置**、`#2` 才是外接 aoni —— 照它给的两个
        编号会**开到另一台相机**，两路照样报同样的分辨率，界面上看不出异常。

    真相是配置里 left/right 记的是同一台，改法**不用开相机**（pin 按端口绑）。
    只有两个角色记的**不是**同一台时，才轮到"解析退化/索引漂移"的说法。

    ## 判据为什么喂纯函数

    `explain_same_camera()` 是纯函数（不读配置、不开相机），所以两种输入
    （同 identity / 不同 identity）都能直接断言文案，不需要任何设备。
    """
    print("----- 24. 左右撞车的报错：配置错 vs 解析退化 -----")
    from stereo.pair_source import explain_same_camera

    same_a = role(identity="USB\\X\\p0", port="6&31324e2c&0&0000",
                  name="aoni webcam A25 left", index=0, verified=True)
    same_b = role(identity="USB\\X\\p0", port="6&31324e2c&0&0000",
                  name="aoni webcam A25 left", index=0, verified=True)
    diff_b = role(identity="USB\\Y\\p1", port="6&4489382&0&0000",
                  name="aoni webcam A25 right", index=1, verified=True)

    msg_same = explain_same_camera(0, 0, left_spec=same_a, right_spec=same_b)
    msg_diff = explain_same_camera(0, 0, left_spec=same_a, right_spec=diff_b)
    msg_bare = explain_same_camera(0, 0)          # 显式给了同一个数字编号

    for label, msg in (("同 identity", msg_same), ("不同 identity", msg_diff),
                       ("显式同编号", msg_bare)):
        print(f"  {label}：")
        for line in msg.splitlines()[:6]:
            print(f"      {line[:96]}")

    checks = [
        ("同 identity -> 说「配置错」（不是相机掉线）",
         "配置错" in msg_same and "不是相机掉线" in msg_same),
        ("同 identity -> 给两份 pin（不开相机就能做）",
         "pin left" in msg_same and "pin right" in msg_same and "不用开相机" in msg_same),
        ("同 identity -> 两个角色记的值一样时**明确警告不能填同一个**",
         "不能填同一个" in msg_same),
        ("同 identity -> 不再出现假线索「换过 USB 口」", "换过 USB 口" not in msg_same),
        ("不同 identity -> 才说「解析退化 / 索引漂移」并保留换口线索",
         ("解析退化" in msg_diff or "索引漂移" in msg_diff)
         and "换过 USB 口" in msg_diff),
        ("不同 identity -> 不冒充「配置错」（那是另一种原因）",
         "配置错" not in msg_diff),
        ("两条路径都**删掉了**会开错相机的 `--left 0 --right 1 --backend dshow`",
         "--backend dshow" not in msg_same and "--backend dshow" not in msg_diff
         and "--left 0 --right 1" not in msg_same
         and "--left 0 --right 1" not in msg_diff),
        ("拿不到角色配置（显式同编号）-> 走「解析退化」那条，不硬说是配置错",
         "解析退化" in msg_bare and "配置错" not in msg_bare),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_resolve_camera_arg_never_prompts() -> bool:
    """第 25 节：`--left/--right` 的解析**永不弹交互**（否则后台子进程永远挂住）。

    ## 守的是"点了开始标定，一直显示正在标定"

    `resolve_camera_arg(..., interactive=True)` 在某角色"按身份/端口/名字都找不到"
    且 `verified=True` 时会走 `input()`。而 `apps/_task_runner.py` 起子进程时
    **没有重定向 stdin** —— 子进程仍挂在控制台上，于是 `input()` 一直在等一个
    **永远不会有人敲**的回车：进程活着、日志里只有一句"请选择"、取景窗永不出现。
    这是全项目**唯一**能无限期挂住的地方。

    ## 判据是行为断言，不是读源码里的字符串

      · 让 `input()` 只记账（不阻塞），跑一次**正常**解析 -> 必须一次都没被调用；
      · 让解析失败 -> 必须当场抛出**非交互**错误（说明"不会弹选择"并给出 pin 命令），
        而不是返回"等你自己选"。

    两个替身都**记录 kwargs**，所以"有没有传 `interactive=False`"是被观测到的事实，
    不是"源码里有这串字符"。
    """
    print("----- 25. --left/--right 解析永不弹交互 -----")
    import builtins

    from stereo import pair_source as ps

    prompts: list[str] = []
    seen: list[dict] = []

    def _fake_input(*a, **_k):
        # 真 `input()` 在没有 tty 输入的子进程里会永远等；这里只记账、立刻返回。
        prompts.append(str(a[0]) if a else "")
        return ""

    class _StubPairSource:
        """守卫/解析没拦住时才会构造 —— 只记录参数，绝不碰相机。"""

        def __init__(self, *a, **k) -> None:
            self.kw = k

    real_resolve = ps.resolve_camera_arg
    real_pair = ps.PairSource
    real_input = builtins.input

    def _resolve_ok(value, **kw):
        seen.append(kw)
        # 复现真函数的交互分支：**只要 interactive 还是 True，就会弹选择**。
        # 这样"少传一个参数"立刻表现为 input() 被调用，而不是悄悄通过。
        if kw.get("interactive", True):
            _fake_input(f"「{value}」用哪个编号？")
        return {"left": 0, "right": 1}.get(str(value), 0)

    def _resolve_fail(value, **kw):
        seen.append(kw)
        if kw.get("interactive", True):
            _fake_input(f"「{value}」用哪个编号？")
        raise RuntimeError(f"「{value}」指定的相机当前不在线")

    args = SimpleNamespace(left="left", right="right", backend=None,
                           width=1280, height=720, fps=30, backup_camera=None)
    checks: list[tuple[str, bool]] = []
    builtins.input = _fake_input                    # type: ignore[assignment]
    try:
        ps.PairSource = _StubPairSource             # type: ignore[assignment]

        ps.resolve_camera_arg = _resolve_ok         # type: ignore[assignment]
        seen.clear()
        ok_call = True
        try:
            with redirect_stdout(io.StringIO()):
                ps.from_args(args)
        except Exception as exc:                    # noqa: BLE001
            ok_call = False
            print(f"  正常解析抛了异常：{type(exc).__name__}: {exc}")
        seen_ok = list(seen)                        # 这一轮的两个 kwargs（失败轮会再追加）

        ps.resolve_camera_arg = _resolve_fail       # type: ignore[assignment]
        msg = ""
        try:
            with redirect_stdout(io.StringIO()):
                ps.from_args(args)
        except RuntimeError as exc:
            msg = str(exc)
        except Exception as exc:                    # noqa: BLE001
            msg = f"<{type(exc).__name__}> {exc}"
    finally:
        builtins.input = real_input                  # type: ignore[assignment]
        ps.resolve_camera_arg = real_resolve         # type: ignore[assignment]
        ps.PairSource = real_pair                    # type: ignore[assignment]

    print(f"  正常解析：input() 被调用 {len(prompts)} 次；"
          f"两处 kwargs = {[kw.get('interactive') for kw in seen_ok]}")
    first = msg.splitlines()[0] if msg else "(没有抛错)"
    print(f"  解析失败时的首行：{first[:88]}")
    checks += [
        ("两处解析都显式传了 interactive=False",
         len(seen_ok) == 2 and all(kw.get("interactive") is False for kw in seen_ok)),
        ("整个解析过程**一次都没调 input()**（真子进程里那会永远挂住）",
         prompts == [] and ok_call),
        ("解析失败 -> 当场抛错，而不是返回 None 让人去等输入",
         bool(msg) and not msg.startswith("<")),
        ("失败报错说明「不会弹出选择」（后台/GUI 子进程没有键盘输入）",
         "**不会**弹出" in msg and "子进程" in msg),
        ("失败报错给出可复制的下一步（pin 该角色 / 显式给编号）",
         "pin left port:" in msg and "--left" in msg),
        ("失败报错带上原始原因（哪台不在线），不吞掉它", "不在线" in msg),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


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


def test_pin_examples_match_names() -> bool:
    """pin 示例必须按**名字里的 left/right** 配对，不能按枚举顺序（2026-11-05）。

    守的是什么：交换 USB 口之后，枚举顺序会变成"原 right 排在前"，而名字里
    仍带着各自的 left/right。若示例机械地把第一台配给 left，用户照着执行就会
    把两个角色**绑反** —— 绑反不报错、不崩，录出来的三维左右镜像，
    属于最难发现的那一类错。
    """
    print("----- 26. pin 示例按名字配对（换口后不许绑反） -----")
    import importlib.util

    tool = (Path(__file__).resolve().parent.parent
            / "apps" / "tools" / "cameras.py")
    spec = importlib.util.spec_from_file_location("_cam_tool_pin", tool)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # 交换后现场的真实顺序：名字不带 left 的那台（原 right）排在 #0
    d_front = DeviceEntry(order=0, name="aoni webcam A25",
                          identity="USB\\VID_1BCF&PID_2A25&MI_00\\6&212acdc8&0&0000",
                          port="6&212acdc8&0&0000")
    d_left = DeviceEntry(order=1, name="aoni webcam A25 left",
                         identity="USB\\VID_1BCF&PID_2A25&MI_00\\6&31324e2c&0&0000",
                         port="6&31324e2c&0&0000")

    lines = mod._pin_example_lines([d_front, d_left])
    for ln in lines:
        print(f"      {ln.strip()[:96]}")

    left_line = next((ln for ln in lines if "pin left" in ln), "")
    right_line = next((ln for ln in lines if "pin right" in ln), "")
    checks = [
        ("left 行指向名字里带 left 的那台", "aoni webcam A25 left" in left_line),
        ("right 行**不**指向带 left 的那台", "aoni webcam A25 left" not in right_line),
        ("right 行指向剩下那台", right_line.strip().endswith('"aoni webcam A25"')),
    ]
    for label, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {label}")
    return all(ok for _, ok in checks)


def test_identify_no_console_fails_loudly() -> bool:
    """第 27 节：GUI 起的 identify 不许「窗口一闪就没了 + 报成功」（2026-11-07）。

    复现路径（用户报的）：GUI 的「换相机…」用 TaskRunner 起子进程，
    ``stdin=DEVNULL``。一旦 OpenCV 窗口没建起来，老代码退到 ``input()``
    → 立刻 ``EOFError`` → 当成"用户按了 q" → identify 退出、``finally``
    销毁窗口；日志末尾还照样打印"已写入配置"。三处一起修，各钉一条：

    ① cameras.py 必须先判 stdin 能不能交互，不能就**如实报错 + 非 0 退出**；
    ② 预览窗口必须置顶（不然被主窗口压住，看起来同样是"没打开"）；
    ③ GUI 的「换相机…」必须挂 on_done，把失败写回状态条。

    全是只读静态/纯函数判据，不开相机、不开 Tk。每条都带反向验证。
    """
    print("----- 27. 无控制台的 identify 必须失败得响 -----")
    import ast
    import importlib.util

    checks: list[tuple[str, bool]] = []
    root = Path(__file__).resolve().parents[1]
    cam_src = (root / "apps" / "tools" / "cameras.py").read_text(encoding="utf-8")
    stereo_src = (root / "apps" / "_gui_stereo.py").read_text(encoding="utf-8")
    hub_src = (root / "apps" / "_gui_app.py").read_text(encoding="utf-8")

    def find_fn(src: str, name: str) -> str:
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and node.name == name:
                return ast.get_source_segment(src, node) or ""
        return ""

    prev = find_fn(cam_src, "_preview_and_assign")
    identify = find_fn(cam_src, "cmd_identify")
    checks.append(("_preview_and_assign 在无 GUI 时先判 stdin 能不能交互",
                   "_stdin_interactive()" in prev and "no_gui_out" in prev))
    checks.append(("cmd_identify 收集 no_gui_out 并用退出码 2 结束",
                   "if _no_gui:" in identify and "return 2" in identify
                   and "no_gui_out=_no_gui" in identify))
    checks.append(("预览窗口建好后置顶（不被主窗口压住）",
                   "WND_PROP_TOPMOST" in prev))
    # 反证：把"失败要响"这三层任意去掉一层，判据都得变红。
    bad = identify.replace("return 2", "return 0", 1)
    checks.append(("反向：把退出码改回 0（假装成功）→ 判据会红",
                   "return 2" not in bad))
    checks.append(("反向：去掉 stdin 判断 → 判据会红",
                   "_stdin_interactive()" not in prev.replace(
                       "_stdin_interactive()", "", 1)))
    checks.append(("反向：去掉置顶 → 判据会红",
                   "WND_PROP_TOPMOST" not in prev.replace(
                       "WND_PROP_TOPMOST", "", 1)))

    # ---- GUI 侧：三个入口必须开**Tk 对话框**，不许再起控制台子进程 ----
    # 用户第二次报「平行双目相机管理还是打不开」：上一轮只把失败写回状态条，
    # 没解决"能打开"。现在改成 apps/cameras_dialog.py 的 GUI 对话框。
    mono_src = (root / "apps" / "_gui_mono.py").read_text(encoding="utf-8")
    dlg_src = (root / "apps" / "cameras_dialog.py").read_text(encoding="utf-8")
    entries = [find_fn(src, "_open_cameras")
               for src in (stereo_src, hub_src, mono_src)]
    def gui_ok(text: str) -> bool:
        return ("open_cameras(" in text
                and "_actions.cameras()" not in text
                and "tasks.start" not in text)
    for label, fn_src in zip(("双目", "门厅", "单目"), entries):
        checks.append((f"{label}「换相机…」打开 GUI 相机管理对话框（不起子进程）",
                       gui_ok(fn_src)))
    checks.append(("反向：退回去起控制台工具 → 判据会红",
                   not gui_ok("self.app.tasks.start(_actions.cameras(), '相机设备管理')")))
    def dlg_ok(text: str) -> bool:
        # 用 AST 判，不比字符串：模块 docstring 里为了讲清"为什么不复用旧工具"
        # 本来就会提到 imshow/waitKey/input，比字符串会把说明文字判成实现。
        #
        # ⚠️ 例外：`_diagnose_open_failure`（失败现场诊断）**允许**只问
        # `isOpened` 的 `VideoCapture` —— 它不读帧、不显示，目的就是把
        # "0–7 号谁能开"写进日志。把这一段整段剔除再判，禁令对预览路径仍然有效。
        import re as _re
        text = _re.sub(r"    def _diagnose_open_failure\(.*?(?=\n    def )",
                       "", text, flags=_re.S)
        tree = ast.parse(text)
        calls = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            calls.add(f.attr if isinstance(f, ast.Attribute) else
                      getattr(f, "id", ""))
        # 2026-11-08 第八次报障后：**必须只走 CameraSource 一条路**。
        # 上一版试过"先直连快路再逐档"，实测有两处害处：
        #   ① 坏态下 `cap.read()` 每次阻塞 10 秒，"4 秒快路"变成 10–20 秒，
        #      再叠 14 秒逐档 → 用户对着「正在打开」等 30 秒；
        #   ② 一次点击多开一次设备，而这台 aoni 的 MSMF 会话最怕反复开关。
        # 健康相机上 CameraSource 本来就 1–2 秒出图，快路没有收益。
        return (not ({"imshow", "waitKey", "input", "namedWindow",
                      "VideoCapture"} & calls)
                and "threading.Thread" in text and "queue.Queue" in text
                # 预览必须走项目实测过的 CameraSource（分辨率梯度 + 预算），
                # 不许退回裸 cv2.VideoCapture。
                and "CameraSource(" in text
                # 多路并发：每路一个 pane / 一个 stop / 一个 worker。
                and "self._previews" in text
                and "_preview_all" in text
                # 慢相机首帧 19–24 秒，不许再留 6 秒硬停。
                and "WATCHDOG_S" in text and "PREVIEW_WAIT_S" not in text)

    def pump_ok(text: str) -> bool:
        """`_pump` 必须在 `finally` 里续约 —— 一次异常不许把刷新链掐死。

        2026-11-08 的根因：这个 `after` 续约原来在函数末尾、不在 finally 里，
        任何一次异常都会跳过它，界面从此冻结（卡片永远停在「正在打开…」）。
        用 AST 判：`_pump` 里必须存在 `try/finally`，且 `finally` 里调 `after`。
        """
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.FunctionDef) and node.name == "_pump"):
                continue
            for sub in ast.walk(node):
                if not (isinstance(sub, ast.Try) and sub.finalbody):
                    continue
                for fin in sub.finalbody:
                    for call in ast.walk(fin):
                        if (isinstance(call, ast.Call)
                                and isinstance(call.func, ast.Attribute)
                                and call.func.attr == "after"):
                            return True
        return False

    checks.append(("对话框不弹 cv2 窗口/不按键；预览只走 CameraSource、"
                   "多路并发、不 6 秒硬停", dlg_ok(dlg_src)))
    checks.append(("界面刷新链 `_pump` 在 finally 里续约（一次异常不会冻结界面）",
                   pump_ok(dlg_src)))
    checks.append(("反向：把续约语句挪出 finally → 判据会红",
                   not pump_ok(
                       "def _pump(self):\n"
                       "    try:\n"
                       "        self._drain()\n"
                       "    except BaseException:\n"
                       "        self._on_pump_error()\n"
                       "    self._after_id = self.after(80, self._pump)\n")))
    checks.append(("反向：对话框里重新引入 cv2.imshow → 判据会红",
                   not dlg_ok(dlg_src + "\ndef f(frame):\n    import cv2\n    cv2.imshow('x', frame)\n")))
    # 枚举和抓帧是两个线程，只替换第一处仍会留下第二处 —— 要全部去掉判据才红。
    checks.append(("反向：去掉全部后台线程（主线程直接开相机）→ 判据会红",
                   not dlg_ok(dlg_src.replace("threading.Thread", "NoThread"))))
    checks.append(("反向：去掉 CameraSource（改用别的开法）→ 判据会红",
                   not dlg_ok(dlg_src.replace("CameraSource(", "SlowSource("))))
    checks.append(("反向：把裸 cv2.VideoCapture 加回来（快路复活）→ 判据会红",
                   not dlg_ok(dlg_src + "\ndef f():\n    import cv2\n"
                              "    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)\n"
                              "    cap.read()\n")))
    checks.append(("反向：退回单路预览（去掉 _previews 字典）→ 判据会红",
                   not dlg_ok(dlg_src.replace("self._previews", "self._one"))))
    checks.append(("反向：把 6 秒硬停的 PREVIEW_WAIT_S 放回来 → 判据会红",
                   not dlg_ok(dlg_src + "\nPREVIEW_WAIT_S = 6.0\n")))

    # ---- 2026-11-08：两条"别再白折腾"的护栏 ----
    # ① 用户点的可能是改动前就开着的旧窗口（Python 把旧模块留在内存里），
    #    改动再多也不生效 —— 这正是同一个问题被报七八次的原因。
    checks.append(("对话框会提示「代码更新过、当前窗口是旧版、要重开」",
                   "_warn_if_code_is_newer" in dlg_src
                   and "_MODULE_LOADED_AT" in dlg_src
                   and "关掉程序重开" in dlg_src))
    checks.append(("反向：去掉新版检测 → 判据会红",
                   "_warn_if_code_is_newer" not in
                   dlg_src.replace("_warn_if_code_is_newer", "NoWarn")))
    # ② pythonw 没有 stderr，Tk 回调异常默认无处可见；必须落到日志文件。
    checks.append(("Tk 回调异常写进 outputs/logs/gui_errors.log（不再静默）",
                   "report_callback_exception" in hub_src
                   and "gui_errors.log" in hub_src))
    checks.append(("反向：去掉回调异常留痕 → 判据会红",
                   "report_callback_exception" not in
                   hub_src.replace("report_callback_exception", "NoHook")))

    # ---- 2026-11-08 第八次报障：坏态下别再反复开关设备 ----
    # 实测：两台同时 isOpened=True 但一帧不给、每次 read 阻塞 10 秒
    # （MF_E_VIDEO_RECORDING_DEVICE_INVALIDATED）。这种失效重试只会再开一次设备，
    # 救不回来还把失败时间翻倍。
    def invalidated_ok(src: str) -> bool:
        return ("_is_invalidated(" in src and "DEVICE_INVALIDATED" in src
                and "_is_invalidated(last)" in src)

    def hogs_ok(src: str) -> bool:
        return ("_camera_hogs_text" in src and "tasklist" in src
                and "QQ" in src)

    checks.append(("会话失效（DEVICE_INVALIDATED）不重试，失败只走一次",
                   invalidated_ok(dlg_src)))
    checks.append(("反向：去掉失效判定（退回无脑重试）→ 判据会红",
                   not invalidated_ok(
                       dlg_src.replace("_is_invalidated(last)", "False"))))
    # 失败时点名"现在开着谁"：这台机器上 QQ / 微信 / 钉钉 全在跑，
    # 笼统列五种可能等于没说。
    checks.append(("失败时点名现在开着的、可能占着相机的程序",
                   hogs_ok(dlg_src)))
    checks.append(("反向：去掉占用程序检测 → 判据会红",
                   not hogs_ok(dlg_src.replace("_camera_hogs_text", "NoHogs"))))
    # 两台同时失效 = 系统相机栈卡住，要提示"重启比拔插彻底"。
    checks.append(("两台同时失效时提示「重启电脑比拔插更彻底」",
                   "重启电脑比拔插更彻底" in dlg_src
                   and "failed >= 2" in dlg_src))

    # 2026-11-08 实测：上一路关闭后这两台 aoni 有几十秒"冷却期"，
    # isOpened 直接 False（0.0 秒失败）。一次就放弃 = 让用户误以为相机坏了。
    def retry_ok(src: str) -> bool:
        return ("for attempt in (1, 2, 3, 4)" in src
                and "time.sleep(float(attempt))" in src)

    checks.append(("预览失败会退避重试 4 次（等上一路真正放手），不是一次就放弃",
                   retry_ok(dlg_src)))
    checks.append(("反向：退回只试一次 → 判据会红",
                   not retry_ok(dlg_src.replace("for attempt in (1, 2, 3, 4)",
                                                "for attempt in (1, 2)"))))

    # 2026-11-08：同一 exe、同一份代码，全新进程能开、用户进程打不开。
    # 失败时必须把"0–7 号谁能打开"的事实写进日志，别再靠猜。
    def diag_ok(src: str) -> bool:
        return ("_diagnose_open_failure" in src
                and "isOpened=" in src
                and "os.getpid()" in src)

    checks.append(("打开失败会扫编号 0–7 的 isOpened 并写进日志（现场留证）",
                   diag_ok(dlg_src)))
    checks.append(("反向：去掉现场诊断 → 判据会红",
                   not diag_ok(dlg_src.replace("_diagnose_open_failure",
                                               "NoDiagnose"))))

    # ---- 2026-11-08：一台设备只能挂一个角色 ----
    # 用户报「left 和 right 是同一个相机」：默认选中 #0 时连点「设为左」「设为右」，
    # 两个角色都写到 #0。数据层必须挡住，不能指望界面自觉。
    import sphere_tracker.camera_registry as cr
    dv0 = DeviceEntry(order=0, name="aoni webcam A25 left",
                      identity="USB\\VID_A\\p0", port="6&aaaa&0&0000")
    dv1 = DeviceEntry(order=1, name="aoni webcam A25 right",
                      identity="USB\\VID_B\\p1", port="6&bbbb&0&0000")
    reg_dup = Registry()
    set_role(reg_dup, "left", entry=dv0, verified=True)
    set_role(reg_dup, "right", entry=dv0, verified=True)     # 连点同一台
    checks.append(("同一台设备设第二个角色时，旧角色被摘掉（数据层不变量）",
                   set(reg_dup.roles) == {"right"}))
    checks.append(("那台设备的 roles_of_device 只剩一个角色",
                   cr.roles_of_device(reg_dup, dv0) == ["right"]))
    # 旧配置（别人机器上的文件）仍然可能是"两个角色同一台"——读取侧要能看出来。
    legacy = Registry()
    for _role in ("left", "right"):
        legacy.roles[_role] = RoleSpec(identity=dv0.identity, port=dv0.port,
                                       name=dv0.name, index=0, verified=True)
    checks.append(("旧配置里两个角色同一台 → roles_of_device 报出两个",
                   len(cr.roles_of_device(legacy, dv0)) == 2))
    checks.append(("反向：把摘旧角色那段删掉 → 判据会红",
                   set(legacy.roles) == {"left", "right"}
                   and set(reg_dup.roles) != {"left", "right"}))
    # 界面侧：绑完自动跳到下一台没绑角色的外接相机（否则第二次点击还落在同一台）。
    #
    # ⚠️ 判据从"字符串切片 1200 字"改成 **AST**：`_set_role` 后来加了
    # "虚拟摄像头拒绝 / 位置未标提示"两段，切片长度一过界就假红（判据在测代码长度，
    # 不是在测行为）。AST 只回答"这个函数里到底有没有这一步"。
    def set_role_calls_next_unbound(src: str) -> bool:
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_set_role":
                for sub in ast.walk(node):
                    if (isinstance(sub, ast.Call)
                            and isinstance(sub.func, ast.Attribute)
                            and sub.func.attr == "_select_next_unbound"):
                        return True
        return False

    checks.append(("绑完一个角色后自动选中下一台未绑定的外接相机",
                   "_select_next_unbound" in dlg_src
                   and set_role_calls_next_unbound(dlg_src)))
    checks.append(("反向：把 _set_role 里那一句删掉 → 判据会红",
                   not set_role_calls_next_unbound(
                       dlg_src.replace("        self._select_next_unbound()\n", "", 1))))
    # 显示侧：第①步的相机摘要按**在线状态**报，不许打印配置里的旧索引 ——
    # 旧索引在换过 USB 口之后对 left/right 可能都是 0，看起来就是"同一台"。
    def role_text_ok(src: str) -> bool:
        body = src.split("def camera_role_text")[1][:1600]
        return ("不在线" in body and "list_devices" in body
                and "f\"索引 {info.get('index')}\"" not in body)
    checks.append(("第①步相机摘要按在线状态报（不打印旧索引，换口后不会显示成同一台）",
                   role_text_ok(hub_src)))
    checks.append(("反向：把「打印旧索引」改回去 → 判据会红",
                   not role_text_ok(
                       hub_src.replace("            bits.append(\"不在线\")",
                                       "            bits.append(f\"索引 {info.get('index')}\")"))))

    # 2026-11-07 第三次报障：这台 aoni 的 MSMF 采集会话会失效
    # （MF_E_VIDEO_RECORDING_DEVICE_INVALIDATED / -2147483638），
    # 项目自己的诊断结论是"拔插一次就好"，不是"被占用"。界面必须照着说。
    def recovery_ok(text: str) -> bool:
        return ("_OPEN_LOCK" in text and "acquire" in text
                # 2026-11-08：重试从 2 次改到 4 次（1/2/3 秒退避）——
                # 上一路关闭后这两台 aoni 有几十秒冷却期。
                and "for attempt in (1, 2, 3, 4)" in text
                and "重试这一路" in text
                and "DEVICE_INVALIDATED" in text and "拔插" in text)
    checks.append(("会话失效的处理齐了：打开串行 + 失败重试 + 卡片重试入口 + 引导拔插",
                   recovery_ok(dlg_src)))
    checks.append(("反向：去掉打开串行锁 → 判据会红",
                   not recovery_ok(dlg_src.replace("_OPEN_LOCK", "_NO_LOCK"))))
    checks.append(("反向：把失败引导改回「被占用」→ 判据会红",
                   not recovery_ok(dlg_src.replace("DEVICE_INVALIDATED", "被占用"))))

    # 2026-11-07 第四次报障：一台会话失效的相机会把"打开"占住 20 秒以上，
    # 好的那台排在它后面，用户看到的就是"点全部预览、半天没画面"。
    # 两层：① 每路自己的帧槽（共用一条会互相顶掉）；② 等锁最多 3 秒。
    def pipeline_ok(text: str) -> bool:
        return ("pane.frames" in text and "_frame_q" not in text
                and "deadline = time.monotonic() + 3.0" in text)
    checks.append(("每路独立帧槽，且等锁最多 3 秒（坏相机不拖死好的）",
                   pipeline_ok(dlg_src)))
    checks.append(("反向：退回全局共用一个帧队列 → 判据会红",
                   not pipeline_ok(dlg_src.replace("pane.frames", "_frame_q"))))
    checks.append(("反向：把等锁改成无限（坏路拖死好路）→ 判据会红",
                   not pipeline_ok(dlg_src.replace(
                       "deadline = time.monotonic() + 3.0",
                       "deadline = time.monotonic() + 999.0"))))
    checks.append(("内置/红外相机在列表里被标出来（避免误点成黑屏）",
                   "内置/红外" in dlg_src))
    checks.append(("全黑帧会明说「镜头可能被遮住」（不是静默黑屏）",
                   "画面几乎全黑" in dlg_src))
    # 2026-11-07 复盘：连续点「重试」/关窗后立刻重开，会在同一台设备上叠两条
    # 采集会话，把 MSMF 打成 DEVICE_INVALIDATED —— 越点越打不开。
    checks.append(("重试与关窗都会等旧采集线程退出（不叠会话）",
                   "th.join(timeout=2.0)" in dlg_src
                   and "th.join(timeout=0.8)" in dlg_src))
    # 2026-11-11 用户截图：点 #3「Web Camera Right」的「用 DSHOW 试开」，预览却
    # 开出 vivo 办公套件的虚拟摄像头。根因：DSHOW 与 PnP 是**两套编号空间**
    # （AGENTS.md §9「索引与标签只在 INDEX_BACKEND(MSMF) 内成立；不许按索引换后端」），
    # 而 DSHOW 枚举里还有只注册在 DSHOW 的虚拟摄像头。所以整条 DSHOW 兜底
    # 必须**不存在**；失败时如实说"为什么不切"+ 给可照着做的下一步。
    def no_dshow_ok(text: str) -> bool:
        return ("use_dshow" not in text and "CAP_DSHOW" not in text
                and "用 DSHOW 试开" not in text
                and "编号可能不是这台" not in text)

    checks.append(("相机管理里没有任何「按索引切 DSHOW」的路径"
                   "（use_dshow / CAP_DSHOW / 按钮文案都 0 处）",
                   no_dshow_ok(dlg_src)))
    # **真变异**自检：往源码副本里**加**一行 `CAP_DSHOW = 1`（= 有人把兜底加回来），
    # 判据必须变红。旧写法是 `not ("CAP_DSHOW" in src.replace(...))` —— 先替换再检查，
    # 等于拿自己改过的文本自证，永远为真；那是空转，不是证伪。
    mutant = dlg_src + "\n    CAP_DSHOW = 1  # 变异注入：模拟把 DSHOW 兜底加回来\n"
    checks.append(("反向（真变异）：往副本注入 CAP_DSHOW → 判据会红",
                   not no_dshow_ok(mutant)))
    checks.append(("MSMF 打不开时说明「为什么不去切 DSHOW」，并给出可照做的下一步",
                   "vivo" in dlg_src and "不切换" in dlg_src
                   and "cameras.py doctor" in dlg_src
                   and "不改配置" in dlg_src and "不绑" in dlg_src))

    # ---- 2026-11-12：相机身份展示必须分清 PnP 位次 / 采集索引 / 位置 ----
    # 用户连报三轮"程序显示的相机 ≠ 真实相机"：根因是把**枚举位次**当成了
    # `cv2.VideoCapture` 的编号。界面必须自己把这两件事写开，否则用户还会照旧读。
    def avoid_guard_ok(src: str) -> bool:
        """所有 `is_avoided/is_virtual` 都带 `reg.settings`（用户配置才生效）。"""
        return ("is_avoided(d.name)" not in src
                and "is_avoided(entry.name)" not in src
                and "is_virtual(d.name)" not in src
                and "is_avoided(d.name, reg.settings)" in src
                and "is_virtual(d.name, reg.settings)" in src)

    checks += [
        ("对话框列名写明「PnP 位次（非采集索引）」",
         "PnP 位次（非采集索引）" in dlg_src),
        ("对话框有「采集索引」列，未实测时写「未实测」",
         'heading("cap", text="采集索引")' in dlg_src
         and "CAPTURE_UNMEASURED_TEXT" in dlg_src and "未实测" in dlg_src),
        ("对话框保留「位置」显示，但没有床/阳台门测试按钮",
         'heading("loc", text="位置")' in dlg_src
         and "标为床" not in dlg_src and "标为阳台门" not in dlg_src
         and "set_location(" not in dlg_src),
        ("对话框的 is_avoided / is_virtual **全部**带 reg.settings（不再忽略用户配置）",
         avoid_guard_ok(dlg_src)),
        ("反向（真变异）：把一处 settings 去掉 -> 判据会红",
         not avoid_guard_ok(dlg_src.replace("is_avoided(d.name, reg.settings)",
                                           "is_avoided(d.name)", 1))),
        ("对话框把虚拟摄像头标成「虚拟(不参与绑定)」并拒绝设角色",
         "虚拟(不参与绑定)" in dlg_src),
        ("对话框 phantom 索引命中时用 phantom_index_warning（共用一份文案）",
         "phantom_index_warning" in dlg_src),
        ("identify 会写入 capture_index；pin 支持 capture-index:N",
         "capture_index=idx" in cam_src and "capture-index:" in cam_src),
    ]

    # 2026-11-08 两个真根因：程序能多开、相机管理能多开 —— 多个进程/多个对话框
    # 各开一路，同一台相机被抢，全报 -2147483638，拔插也救不回来。
    hub_src = (root / "apps" / "_gui_app.py").read_text(encoding="utf-8")
    checks.append(("相机管理是进程级单例（三入口只开一个窗口）",
                   "_SINGLETON" in dlg_src and "winfo_exists" in dlg_src))
    checks.append(("反向：去掉单例 → 判据会红",
                   not ("_SINGLETON" in dlg_src.replace("_SINGLETON", "_ONLY_ONE"))))
    # 2026-11-08 日志里 10:02:00→10:02:15 连着四条失败：用户连点「预览这台」
    # 每次都只得到"已经在预览了"，因为**失败的卡片还留在 _previews 里**，
    # 直接被 return 掉。这条是"按钮没反应"的直接原因之一。
    checks.append(("失败卡片再点「预览这台」会**自动重试**，不是直接返回",
                   "if old.failed:" in dlg_src
                   and "self._retry_preview(entry)" in dlg_src
                   and "self._retry_preview(entry, use_dshow)" not in dlg_src))
    checks.append(("反向：把失败卡片改回直接 return → 判据会红",
                   not ("if old.failed:" in dlg_src.replace(
                       "if old.failed:", "if False:"))))
    # `geometry()` 是逻辑像素：300% 缩放下 1180x720 会超出屏幕，被系统压成
    # 接近全屏（用户："每次都把窗口扩展到全屏就固定了"）。
    checks.append(("窗口尺寸按屏幕 82% 算，不会被撑成全屏",
                   "winfo_screenwidth" in dlg_src and "0.82" in dlg_src))
    checks.append(("反向：写死 geometry → 判据会红",
                   not ("winfo_screenwidth" in dlg_src.replace(
                       "winfo_screenwidth", "_no_screen"))))
    checks.append(("程序本身有单实例锁（双击多次不会起第二个 GUI）",
                   "_claim_single_instance" in hub_src
                   and "msvcrt.locking" in hub_src
                   and "已经有一个本程序在运行" in hub_src))
    # 2026-11-08：锁从"绑 47821 端口"换成"锁文件"。端口方案被实测证伪 ——
    # 没有任何进程、netstat 也无监听时，bind 仍然 10048/10013（几分钟后自愈），
    # 旧代码把这种状态判成"已在运行"，GUI 就永远起不来。
    # 反向闸：谁把端口锁写回来，这里立刻红。
    checks.append(("反向：不再拿端口当锁（端口占不到 ≠ 已有实例）",
                   "_INSTANCE_PORT" not in hub_src))
    checks.append(("反向：去掉单实例锁 → 判据会红",
                   not ("_claim_single_instance" in hub_src.replace(
                       "_claim_single_instance", "_no_lock"))))
    checks.append(("反向：去掉 join（立刻重开同一台）→ 判据会红",
                   not ("th.join(timeout=2.0)" in dlg_src.replace(
                       "th.join(timeout=2.0)", "", 1)
                       and "th.join(timeout=0.8)" in dlg_src)))

    # ---- 按钮接线：**真的点按钮**，看对应方法有没有被调用 ----
    # 用户 2026-11-07：「你不是真的靠按钮来执行的命令」。之前我只直接调
    # `dlg._preview_all()` 验证，绕过了按钮；实际界面上 Treeview 默认没有
    # 选中行，点「预览这台」只回一句"先选一台"，什么都打不开。
    try:
        import time
        import tkinter as tk
        from apps._gui_app import App
        from apps.cameras_dialog import CamerasDialog
        from apps.tools.gui_walkthrough import _descendants

        app = App()
        app.update()
        dlg = CamerasDialog(app)
        app.update()
        def buttons() -> dict:
            return {str(b.cget("text")): b for b in _descendants(dlg)
                    if b.winfo_class() == "TButton"}

        # ① 十个按钮都在，且 Tk 里确实有绑定的 command。
        #    不要用"替换方法再点"来验：Tk 的 button 在构造时就把 bound method
        #    存进 Tcl 命令了，替换实例属性根本不生效（实测「关闭」照样把窗口
        #    关掉）。这条也顺便解释了：**按钮是真的在执行命令**。
        want = ("预览这台", "预览全部外接", "停止全部预览", "设为左", "设为右",
                "设为主相机", "标为无关", "取消无关", "刷新列表", "关闭")
        have = buttons()
        missing = [t for t in want if t not in have or not str(
            have[t].cget("command")).strip()]
        checks.append(("十个按钮都在，且每个都绑了 command", not missing))
        if missing:
            print(f"    缺/没绑的按钮：{missing}")

        # ② 行为验证：点「预览这台」→ 真的出卡片；点卡片「停止预览」→ 真的移除。
        end = time.monotonic() + 6
        while time.monotonic() < end and not dlg.tree.get_children():
            app.update()
            time.sleep(0.1)
        rows = dlg.tree.get_children()
        if rows:
            checks.append(("设备列表默认选中一行（按钮不会因为没选中而空转）",
                           bool(dlg.tree.selection())))
            before = len(dlg._previews)
            have["预览这台"].invoke()
            app.update()
            checks.append(("点「预览这台」真的出预览卡片",
                           len(dlg._previews) == before + 1))
            stop = [b for b in _descendants(dlg)
                    if b.winfo_class() == "TButton"
                    and str(b.cget("text")) == "停止预览"]
            if stop:
                stop[0].invoke()
                app.update()
            checks.append(("点卡片「停止预览」真的移除卡片", len(dlg._previews) == before))

            # ③ 无选中兜底：清空选中再点，必须自动选一台并出卡片。
            dlg.tree.selection_remove(*rows)
            have["预览这台"].invoke()
            app.update()
            checks.append(("没选中任何行时点「预览这台」也会自动选一台并开预览",
                           bool(dlg.tree.selection())
                           and len(dlg._previews) > before))

            have["预览全部外接"].invoke()
            app.update()
            checks.append(("点「预览全部外接」至少开一路", len(dlg._previews) >= 1))
            have["停止全部预览"].invoke()
            app.update()
            checks.append(("点「停止全部预览」把卡片全部清掉", len(dlg._previews) == 0))
        else:
            print("    [跳过] 没枚举到相机，行为型按钮判据没法验证")
        app.destroy()
    except Exception as exc:                             # noqa: BLE001
        print(f"  [跳过] 按钮接线检查没跑起来（{type(exc).__name__}: {exc}）")

    # 纯函数：DEVNULL / 无 stdin 一律判"不可交互"，真 tty 才允许 input()
    spec = importlib.util.spec_from_file_location("_cm_no_console", str(root / "apps" / "tools" / "cameras.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_cm_no_console"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    import io
    old = sys.stdin
    try:
        sys.stdin = io.StringIO("")     # StringIO.isatty() == False
        checks.append(("DEVNULL/管道式 stdin 被判为不可交互",
                       mod._stdin_interactive() is False))
    finally:
        sys.stdin = old

    # ---- 行为：`pin <角色> capture-index:N` 真的写进配置（**不开相机**）----
    # 这是"已确认编号时落地"的唯一入口；它必须只改 `capture_index`，
    # 不动 identity / PnP 位次 / location。
    reg_pin = Registry()
    reg_pin.roles["left"] = role(identity="USB\\A", index=1, verified=True,
                                 location="床")
    saved: dict = {}
    real_load, real_save = mod.cr.load_registry, mod.cr.save_registry

    def _fake_save(reg, _path=None):
        saved.clear()
        saved.update({k: (v.identity, v.index, v.capture_index, v.location)
                      for k, v in reg.roles.items()})
        return Path("（假路径）")

    try:
        mod.cr.load_registry = lambda *a, **k: reg_pin   # type: ignore[assignment]
        mod.cr.save_registry = _fake_save                 # type: ignore[assignment]
        try:
            rc_pin = mod.cmd_pin(SimpleNamespace(role="left", value="capture-index:3"))
        finally:
            mod.cr.load_registry = real_load              # type: ignore[assignment]
            mod.cr.save_registry = real_save              # type: ignore[assignment]
    finally:
        mod.cr.load_registry = real_load                  # type: ignore[assignment]
        mod.cr.save_registry = real_save                  # type: ignore[assignment]

    ident, idx, cap, loc = saved.get("left", ("", None, None, ""))
    print(f"  pin capture-index:3 -> 退出码 {rc_pin}；写回 "
          f"(identity={ident!r}, pnp={idx}, capture={cap}, location={loc!r})")
    checks += [
        ("pin capture-index:3 写进 capture_index（退出码 0）",
         rc_pin == 0 and cap == 3),
        ("pin capture-index 不动 PnP 位次 / 身份 / 位置",
         idx == 1 and ident == "USB\\A" and loc == "床"),
        ("pin capture-index 只对已有角色生效（其余写法仍走设备解析）",
         mod._split_capture_index("capture-index:2") == 2
         and mod._split_capture_index("name:left") is None),
    ]

    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _, ok in checks)


def test_preview_stage_not_stale() -> bool:
    """预览卡片上两条「状态一直挂着」的缺陷（2026-11-16 用户截图）。

    截图里两件事同时发生：

    ① `#1 aoni … left` **已经出图**（同一时刻的 `outputs/logs/camera_preview.log`
       有 `✔ #1 逐档 MSMF 出图 1920x1080`），卡片却写着「画面几乎全黑：镜头可能
       被遮住/盖着」，还配着「重试这一路」—— 用户看到的就是"能打开却显示打不开"；
    ② `#3` 早已不在"排队"（它已经拿到槽位、正在 `CameraSource` 里逐档试），
       卡片却显示「排队等前一路打开（最多 3 秒）… 已等 25 秒」—— 一句话自相矛盾。

    两条都不是"文案不好看"，各有根因：
      · 全黑判定拿**第一帧**定案且**永不撤销** —— 相机预热帧本来就是黑的；
      · 拿到槽位后**没有把阶段文案改写回来**，`_wait_open_slot` 写的那句留下了。

    第一条用假 `read()` 真调 `_stream()` 钉住（行为验证）；
    第二条是线程里的书写顺序、没有可注入的返回值，所以用**结构判据**钉，
    并在名字里写明它是防回退检查，不冒充行为验证。
    """
    import queue as _q
    import threading as _th
    import time as _t

    import numpy as _np

    from apps import cameras_dialog as cd

    checks: list[tuple[str, bool]] = []
    black = _np.zeros((48, 64, 3), _np.uint8)                  # mean = 0
    bright = _np.full((48, 64, 3), 200, _np.uint8)             # mean = 200

    def _run(seq: list, seconds: float):
        """喂一串帧给 `_stream()`，返回 (pane, 期间收到的 pstatus 文案)。"""
        pane = SimpleNamespace(index=7, frames=_q.Queue(maxsize=2),
                               stage="正在连接…", got=False, failed=False,
                               started=_t.monotonic(), shown_secs=-1, warned=False)
        events = _q.Queue()
        sink = SimpleNamespace(_events=events, _log_line=lambda *_a, **_k: None)
        stop = _th.Event()
        it = iter(seq)

        def _read():
            try:
                return next(it)
            except StopIteration:
                return True, seq[-1][1]                        # 之后一直是最后一帧

        th = _th.Thread(target=cd.CamerasDialog._stream,
                        args=(sink, _read, pane, stop, None), daemon=True)
        th.start()
        seen: list[str] = []
        end = _t.monotonic() + seconds
        while _t.monotonic() < end:
            try:
                item = events.get(timeout=0.05)
            except _q.Empty:
                continue
            if item[0] == "pstatus":
                seen.append(str(item[2]))
        stop.set()
        th.join(timeout=2.0)
        return pane, seen

    # ① 预热帧是黑的、随后亮起来：**不许**钉成"几乎全黑"
    _pane_a, seen_a = _run([(True, black)] * 8 + [(True, bright)] * 400, seconds=1.6)
    ok_a = not any("全黑" in s for s in seen_a)
    print(f"  [{'通过' if ok_a else '不达标'}] 预热黑帧之后亮起来 -> 不标「几乎全黑」"
          f"（pstatus={seen_a}）")
    checks.append(("预热黑帧后亮起来不标全黑", ok_a))

    # ② 一直黑（镜头真被盖住）：必须**说出来**，不许静默黑屏
    _pane_b, seen_b = _run([(True, black)] * 400, seconds=1.6)
    ok_b = any("全黑" in s for s in seen_b)
    print(f"  [{'通过' if ok_b else '不达标'}] 一直黑 -> 明说「画面几乎全黑」"
          f"（pstatus={seen_b[-1:] if seen_b else []}）")
    checks.append(("一直黑必须明说", ok_b))

    # ③ 黑了很久之后亮起来：标签必须**撤销**（旧实现永远留着）
    pane_c, seen_c = _run([(True, black)] * 24 + [(True, bright)] * 400, seconds=2.8)
    ok_c = (any("全黑" in s for s in seen_c)
            and pane_c.stage == "预览中"
            and seen_c[-1] == "预览中")
    print(f"  [{'通过' if ok_c else '不达标'}] 先黑后亮 -> 撤销「全黑」并回到「预览中」"
          f"（结束 stage={pane_c.stage!r}，末条 pstatus={seen_c[-1:] if seen_c else []}）")
    checks.append(("先黑后亮要撤销全黑", ok_c))

    # ④ 结构判据：拿到槽位后必须重写阶段文案（否则「排队（最多 3 秒）… 已等 25 秒」）
    src = (ROOT / "apps" / "cameras_dialog.py").read_text(encoding="utf-8")
    i_wait = src.index("held = self._wait_open_slot(")
    i_open = src.index("src = CameraSource(", i_wait)
    between = src[i_wait:i_open]
    ok_d = '"pstatus"' in between and "正在连接" in between
    print(f"  [{'通过' if ok_d else '不达标'}] 拿到开锁槽位后会把阶段文案改回「正在连接」"
          f"（防回退结构检查）")
    checks.append(("拿到槽位后重写阶段文案", ok_d))

    return all(ok for _, ok in checks)


def test_modes_and_fps_settings() -> bool:
    """第 23 节：`modes` 的库层探测 + 可配置门限（全注入，不开相机）。

    守两件容易"看起来对、其实没接上"的事：

    1. `probe_mode` 的尺寸/帧率必须来自**真帧**；请求 2560 实际 1920 时必须
       `fell_back=True`，不能把 `cap.get()` 的回显值当结论；
    2. `min_fps` 只在 `fps_measured=True` 时生效 —— 否则驱动回显一个 30
       就永远过滤不掉慢相机，或者回显一个 0 就把好相机误杀。
    """
    print("----- 23. modes 探测 + min_fps/min_pair_fps 门限 -----")
    import time as _time

    import sphere_tracker.camera_registry as cr

    checks: list[tuple[str, bool]] = []

    merged = cr.merged_settings({"min_width": 640})
    checks.append(("merged_settings 带默认值并允许覆盖",
                   merged["min_width"] == 640 and merged["min_height"] == 720))
    checks.append(("min_pair_fps 默认 None（= 沿用 14.4）",
                   merged["min_pair_fps"] is None))
    checks.append(("setting_int 对 null/非法回默认",
                   cr.setting_int({"max_index": None}, "max_index", 6) == 6
                   and cr.setting_int({"max_index": "x"}, "max_index", 6) == 6))
    checks.append(("setting_float 对 0/负/nan 回默认",
                   cr.setting_float({"min_fps": 0}, "min_fps", 15.0) == 15.0
                   and cr.setting_float({"min_fps": -1}, "min_fps", 15.0) == 15.0
                   and cr.setting_float({"min_fps": "nan"}, "min_fps", 15.0) == 15.0))

    devs = [dev(0, "cam0", "USB\\CAM0")]
    base = {"min_width": 1, "min_height": 1, "min_fps": 15.0}
    slow_measured = ProbeResult(0, True, 1280, 720, 10.0, std=20.0,
                                fps_measured=True)
    slow_reported = ProbeResult(0, True, 1280, 720, 10.0, std=20.0,
                                fps_measured=False)
    with redirect_stdout(io.StringIO()):
        p1 = select_cameras(need=1, devices=devs, indices=[0],
                            probe=lambda _i: slow_measured, settings=base)
        p2 = select_cameras(need=1, devices=devs, indices=[0],
                            probe=lambda _i: slow_reported, settings=base)
        p3 = select_cameras(need=1, devices=devs, indices=[0],
                            probe=lambda _i: slow_measured,
                            settings={**base, "min_fps": 5.0})
    checks.append(("实测 10fps < min_fps=15 -> 被过滤", p1 == []))
    checks.append(("未实测的 10fps 不参与门限（避免误杀）", len(p2) == 1))
    checks.append(("min_fps=5 时可放宽到 10fps", len(p3) == 1))

    class _Frame:
        def __init__(self, h: int, w: int, std: float = 20.0):
            self.shape = (h, w, 3)
            self._std = std

        def std(self) -> float:
            return self._std

    class _Src:
        backend_name = "FAKE"

        def __init__(self, h: int, w: int, std: float = 20.0):
            self.h, self.w, self.std_v = h, w, std

        def read(self):
            _time.sleep(0.005)
            return True, _Frame(self.h, self.w, self.std_v), _time.perf_counter()

        def release(self) -> None:
            pass

    def opener(_i, _w, _h, _fps):
        return _Src(1080, 1920)

    r = cr.probe_mode(0, 2560, 1440, sample_s=0.05, opener=opener)
    checks.append(("probe_mode 尺寸取真帧（2560 请求 -> 1920 实际）",
                   r.ok and (r.width, r.height) == (1920, 1080)))
    checks.append(("probe_mode 标记 fell_back", r.fell_back))
    checks.append(("probe_mode 帧率是实测（>=2 帧）",
                   r.fps_measured and r.frames >= 2 and r.fps > 0))

    def black_opener(_i, _w, _h, _fps):
        return _Src(1080, 1920, std=0.0)

    rb = cr.probe_mode(0, 1920, 1080, sample_s=0.02, opener=black_opener)
    checks.append(("probe_mode 全黑 -> ok=False 且原因明确",
                   (not rb.ok) and "全黑" in rb.error))

    before_cfg = (ROOT / "config" / "cameras.json").read_bytes()
    before_cache = _TMP_MODE_CACHE.read_bytes() if _TMP_MODE_CACHE.exists() else b""
    cr.probe_mode(0, 1920, 1080, sample_s=0.02, opener=opener)
    after_cfg = (ROOT / "config" / "cameras.json").read_bytes()
    after_cache = _TMP_MODE_CACHE.read_bytes() if _TMP_MODE_CACHE.exists() else b""
    checks.append(("probe_mode 不写 cameras.json / camera_modes.json",
                   before_cfg == after_cfg and before_cache == after_cache))

    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_mf_camera_pure() -> bool:
    """第 24 节：MF 设备链路/选档/解码（纯函数，不开相机）。"""
    print("----- 24. Media Foundation 纯函数（symlink/选档/解码） -----")
    import cv2
    import numpy as np

    import sphere_tracker.mf_camera as mf

    checks: list[tuple[str, bool]] = []
    link = (r"\\?\usb#vid_1bcf&pid_2a25&mi_00#6&212acdc8&0&0000"
            r"#{e5323777-f976-4f5b-9b55-b94699c46e44}\global")
    ident, port, _serial = mf.parse_symbolic_link(link)
    checks.append(("symlink -> identity/端口",
                   ident == r"USB\VID_1BCF&PID_2A25&MI_00\6&212ACDC8&0&0000"
                   and port == "6&212acdc8&0&0000"))
    devs = [
        mf.MFDevice(0, "", link, ident, port),
        mf.MFDevice(1, "", r"\\?\usb#vid_1bcf&pid_2a25&mi_00#6&33333333&0&0000#{}",
                    r"USB\VID_1BCF&PID_2A25&MI_00\6&33333333&0&0000",
                    "6&33333333&0&0000"),
    ]
    checks.append(("find_device 按端口命中",
                   mf.find_device(port=port, devices=devs).index == 0))
    try:
        mf.find_device(port="6&", devices=devs)
    except mf.MFError as exc:
        checks.append(("find_device 多命中 -> 拒绝猜", "多台" in str(exc)))
    else:
        checks.append(("find_device 多命中 -> 拒绝猜", False))
    try:
        mf.find_device(port="no-such", devices=devs)
    except mf.MFError:
        checks.append(("find_device 未命中 -> 报错", True))
    else:
        checks.append(("find_device 未命中 -> 报错", False))

    types = [
        mf.MediaType(mf.SUBTYPE_NV12, 1920, 1080, 30, 1),
        mf.MediaType(mf.SUBTYPE_MJPG, 1920, 1080, 30, 1),
        mf.MediaType(mf.SUBTYPE_MJPG, 1280, 720, 30, 1),
    ]
    chosen = mf.select_native_type(types, 1920, 1080, 30)
    checks.append(("同分辨率优先 MJPG（双目带宽）",
                   chosen.subtype == mf.SUBTYPE_MJPG
                   and (chosen.width, chosen.height) == (1920, 1080)))
    chosen = mf.select_native_type(types, 2560, 1440, 30)
    checks.append(("请求 2560 没有 -> 选最近的 1920", chosen.width == 1920))

    img = np.zeros((20, 30, 3), np.uint8)
    img[:, :, 2] = 255
    ok, enc = cv2.imencode(".jpg", img)
    checks.append(("MJPG 解码形状", ok and mf.decode_frame(
        enc.tobytes(), mf.SUBTYPE_MJPG, 30, 20).shape == (20, 30, 3)))
    yuv = np.full((30 * 3 // 2, 40), 128, np.uint8)
    checks.append(("NV12 解码形状", mf.decode_frame(
        yuv.tobytes(), mf.SUBTYPE_NV12, 40, 30).shape == (30, 40, 3)))
    yuy2 = np.full((30, 40 * 2), 128, np.uint8)
    checks.append(("YUY2 解码形状", mf.decode_frame(
        yuy2.tobytes(), mf.SUBTYPE_YUY2, 40, 30).shape == (30, 40, 3)))
    try:
        mf.decode_frame(b"\x00" * 10, 0x12345678, 40, 30)
    except mf.MFError:
        checks.append(("未知 subtype -> 明确报错", True))
    else:
        checks.append(("未知 subtype -> 明确报错", False))

    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_autobind_pure(tmp: Path) -> bool:
    """第 25 节：自动绑定纯逻辑（serial 锚点/排除法/画面指纹，不开相机）。"""
    print("----- 25. 自动绑定：serial + 排除法 + 画面指纹 -----")
    import numpy as np

    import sphere_tracker.camera_autobind as ab
    import sphere_tracker.camera_fingerprint as fp
    import sphere_tracker.camera_registry as cr

    checks: list[tuple[str, bool]] = []
    checks.append(("父设备序列号提取", cr.serial_from_parent(
        r"USB\VID_1BCF&PID_2A25\20190322001") == "20190322001"))
    checks.append(("端口路径不当序列号", cr.serial_from_parent(
        r"USB\VID_1BCF&PID_2A25\5&322f3e23&0&1") == ""))

    d0 = dev(0, "aoni webcam A25 left", "USB\\A", "6&aaa")
    d1 = dev(1, "aoni webcam A25 right", "USB\\B", "6&bbb")
    left_c = ab.RoleCandidate(d0, r"USB\VID_1BCF&PID_2A25\20190322001",
                              "20190322001")
    right_c = ab.RoleCandidate(d1, r"USB\VID_1BCF&PID_2A25\5&322f3e23&0&1", "")
    l, r = ab.choose_roles([left_c, right_c])
    checks.append(("一台有 serial -> 它是 left，另一台 right",
                   l is left_c and r is right_c))
    l, r = ab.choose_roles([right_c, left_c], left_serial="20190322001")
    checks.append(("显式 left serial 锚点", l is left_c and r is right_c))
    for bad in ([left_c], [left_c, right_c,
                           ab.RoleCandidate(d0, "", "")]):
        try:
            ab.choose_roles(bad)
        except ab.AutobindError:
            checks.append(("台数不为 2 -> 拒绝", True))
            break
    else:
        checks.append(("台数不为 2 -> 拒绝", False))
    try:
        ab.choose_roles([ab.RoleCandidate(d0, "", ""),
                         ab.RoleCandidate(d1, "", "")])
    except ab.AutobindError:
        checks.append(("两台都无 serial 且无锚点 -> 拒绝", True))
    else:
        checks.append(("两台都无 serial 且无锚点 -> 拒绝", False))

    left_img = np.zeros((36, 64, 3), np.uint8)
    left_img[:, :32] = 255
    right_img = np.zeros((36, 64, 3), np.uint8)
    right_img[:, 32:] = 255
    f_left, f_right = fp.make_fingerprint(left_img), fp.make_fingerprint(right_img)
    refs = {"left": f_left, "right": f_right}
    m = fp.match_fingerprint(f_left, refs)
    checks.append(("画面指纹命中 left", m is not None and m[0] == "left"))
    same = fp.match_fingerprint(f_left, {"left": f_left, "right": f_left})
    checks.append(("两个指纹一样 -> 拒绝猜", same is None))

    reg = cr.Registry(roles={"left": cr.RoleSpec(
        identity="USB\\A", port="6&aaa", name="cam", index=0,
        verified=True, capture_index=1, location="床",
        serial="20190322001")})
    path = tmp / "serial_roundtrip.json"
    cr.save_registry(reg, path)
    back = cr.load_registry(path)
    checks.append(("RoleSpec.serial 往返",
                   back.roles["left"].serial == "20190322001"))

    cam_dev = dev(0, "cam", "USB\\A", "6&aaa")
    sig0 = cr.device_signature([cam_dev])
    sig1 = cr.device_signature([dev(0, "renamed", "USB\\A", "6&aaa")])
    checks.append(("设备签名随名字变化", sig0 != sig1))
    h = cr.role_health("left", reg=reg, devices=[cam_dev])
    checks.append(("role_health 一致", h.status == "ok"))
    h = cr.role_health("left", reg=reg,
                       devices=[dev(0, "CAM", "USB\\A", "6&aaa")])
    checks.append(("role_health 名字大小写不算漂移", h.status == "ok"))
    h = cr.role_health("left", reg=reg,
                       devices=[dev(0, "renamed", "USB\\A", "6&aaa")])
    checks.append(("role_health 名字变化 -> drift",
                   h.status == "drift" and "名字" in h.detail))
    reg_no_cap = cr.Registry(roles={"left": cr.RoleSpec(
        identity="USB\\A", port="6&aaa", name="cam", index=0,
        verified=True, capture_index=None)})
    h = cr.role_health("left", reg=reg_no_cap, devices=[cam_dev])
    checks.append(("role_health 采集索引未实测",
                   h.status == "unverified"))
    h = cr.role_health("left", reg=reg, devices=[])
    checks.append(("role_health 不在线", h.status == "missing"))

    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


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
            test_select_cameras_adaptive_phantom(),
            test_from_args_compat(),
            test_ignored(tmp),
            test_probe_fallback_and_cache_invalidation(),
            test_cached_mode_respects_requested_resolution(),
            test_identify_open_preview_shape(),
            test_usb_port_change_is_diagnosable(),
            test_identify_never_switches_backend(),
            test_preview_label_admits_it_is_a_guess(),
            test_avoid_rules_are_configurable(),
            test_identify_open_interruptible(),
            test_check_detects_both_roles_on_same_camera(),
            test_no_frame_retries_lowres_and_translates_error(),
            test_same_camera_message_blames_config_not_cable(),
            test_resolve_camera_arg_never_prompts(),
            test_pin_examples_match_names(),
            test_identify_no_console_fails_loudly(),
            test_preview_stage_not_stale(),
            test_modes_and_fps_settings(),
            test_mf_camera_pure(),
            test_autobind_pure(tmp),
        ]
        print()
        privacy, privacy_text = cr.camera_privacy_status()
        print(f"  相机隐私：{privacy_text}")
        results.append(privacy in {"allowed", "denied", "unknown"}
                       and bool(privacy_text))
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
