"""相机可用性判定自测：**没有相机时必须立刻说清，而且指对方向**。

为什么需要它
------------
本轮用户明确说明"**外接摄像头没有接上**"。在这个前提下，
"没有相机"会是**最常见的运行状态** —— 而它此前的表现一团糟。

2026-10-10 实测，四个入口给出**四种不同的失败**：

| 入口 | 表现 | 问题 |
| ---- | ---- | ---- |
| `run_realtime.py` | 5.1 s 后说"没找到可用的摄像头" | 慢；且先刷一堆 OpenCV 原始 `VIDEOIO` 警告，把信息淹了 |
| `calibrate_stereo.py` | "取帧源装配失败，退出" | **没说是相机的问题** |
| `verify_stereo.py` | "找不到标定文件 config\\stereo_calibration.npz" | 真因往往正是**没有相机所以从没标定成功过**，但提示把人引向"文件丢了" |
| `check_stereo_setup.py` | 要 `--left/--right` | 合理，但也没回答"到底有没有相机" |

没有任何一处**集中**回答"有没有相机、为什么不行、接下来做什么"。

## 判据为什么必须可注入

`camera_availability()` 的外部依赖（设备列表 / 试开函数 / 注册表）**全部可注入**，
所以没有相机的机器上也能把**每条分支**走一遍。
真去开相机的测试在沙箱里只能得到"全都打不开"一种结果，**等于没测**。

反向自证（改了就会红）
--------------------
========================================  ==========================================  ========
改动                                      注入方式                                    结果
========================================  ==========================================  ========
把"没有设备"当成可用                      空设备列表也返回 ok=True                    exit=1
不说清"有设备但都不可用"                  设备数 >0 时沿用"没有检测到"的文案          exit=1
不给可执行的下一步                        把 remedy 置空                              exit=1
分辨率不达标也算可用                      跳过 min_width/min_height 检查              exit=1
试开异常直接崩掉                          去掉 try/except                             exit=1
========================================  ==========================================  ========

用法::

    python tests/selftest_camera_presence.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
while not (ROOT / "sphere_tracker").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sphere_tracker.camera_registry import (  # noqa: E402
    Availability,
    DeviceEntry,
    ProbeResult,
    camera_availability,
    select_cameras_detailed,
)


def device(order: int) -> DeviceEntry:
    return DeviceEntry(order=order)


def ok_probe(index: int, width: int = 1280, height: int = 720, fps: float = 30.0):
    """返回一个"试开成功"的假 probe。"""
    def _p(i: int) -> ProbeResult:
        if i != index:
            return ProbeResult(i, False, 0, 0, 0.0, 0.0, "打不开")
        return ProbeResult(i, True, width, height, fps, 40.0, "")
    return _p


def bad_probe(error: str = "被占用"):
    def _p(i: int) -> ProbeResult:
        return ProbeResult(i, False, 0, 0, 0.0, 0.0, error)
    return _p


# --------------------------------------------------------------------------
def test_no_camera_is_reported_clearly() -> bool:
    """第 1 节：一台都没有 —— 必须 ok=False，且**说清怎么办**。"""
    print("----- 1. 完全没有相机 -----")
    av = camera_availability(need=1, devices=[], probe=bad_probe("打不开"))
    print(f"  ok={av.ok} device_count={av.device_count} usable={av.usable_count}")
    print(f"  reason : {av.reason}")
    print(f"  remedy : {av.remedy}")

    checks = [
        ("ok 为 False（不许把「没有」当成「可用」）", av.ok is False),
        ("device_count 为 0", av.device_count == 0),
        ("reason 明确说了「没有检测到」", "没有检测到" in av.reason),
        ("reason 报出试开了几个索引（否则像没查）", "试开了" in av.reason),
        ("remedy 非空（要能引导下一步，不只是报告问题）", len(av.remedy) > 10),
        ("remedy 给了**可替换的退路**（没有相机也能试功能）",
         "run_video" in av.remedy or "视频" in av.remedy),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_existing_but_unusable_is_a_different_message() -> bool:
    """第 2 节：**有设备但都不可用**必须与"没有设备"分开说。

    两者的排查方向完全不同：
    - 没有设备 -> 查线/口/设备管理器；
    - 有设备但不可用 -> 查占用 / 带宽 / 驱动。

    第一版一律报"没有检测到任何摄像头"，于是"相机被会议软件占着"这种情形
    会把人引去查 USB 线。
    """
    print("----- 2. 有设备但一台都用不了（例如被占用）-----")
    av = camera_availability(need=1, devices=[device(0), device(1)],
                             probe=bad_probe("被其它程序占用"))
    print(f"  ok={av.ok} device_count={av.device_count} usable={av.usable_count}")
    print(f"  reason : {av.reason[:110]}")
    print(f"  remedy : {av.remedy[:100]}")

    checks = [
        ("ok 为 False", av.ok is False),
        ("device_count 反映了真实设备数（2）", av.device_count == 2),
        ("reason **没有**说「没有检测到」（措辞必须区分）",
         "没有检测到" not in av.reason),
        ("reason 说清了「有 N 个但能用的不够」", "检测到" in av.reason and "能用的" in av.reason),
        ("reason 带上了具体原因（试开报的错）", "占用" in av.reason),
        ("remedy 指向**占用/带宽**方向，而不是查线",
         "占用" in av.remedy or "diag" in av.remedy),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_resolution_below_requirement_counts_as_unusable() -> bool:
    """第 3 节：能打开但分辨率不达标 —— 不算可用（否则下游会静默降级）。"""
    print("----- 3. 能打开但分辨率不够 -----")
    av = camera_availability(need=1, devices=[device(0)],
                             probe=ok_probe(0, width=160, height=120))
    print(f"  ok={av.ok} usable={av.usable_count}")
    print(f"  reason : {av.reason[:110]}")
    checks = [
        ("ok 为 False（分辨率不够不算可用）", av.ok is False),
        ("reason 里点出是分辨率问题（不是笼统的「打不开」）",
         "分辨率" in av.reason),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_probe_exception_does_not_crash() -> bool:
    """第 4 节：试开本身抛异常也不能把判定拖崩。"""
    print("----- 4. 试开抛异常 -----")

    def boom(i: int):
        raise OSError("驱动器拒绝访问")

    av = camera_availability(need=1, devices=[device(0)], probe=boom)
    print(f"  ok={av.ok} reason={av.reason[:100]}")
    checks = [
        ("不抛异常，返回 ok=False", av.ok is False),
        ("reason 里留下了异常类型（便于排查）", "OSError" in av.reason),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_available_camera_reports_ok() -> bool:
    """第 5 节：有相机时必须是 ok=True，且**不再废话**（reason 为空）。"""
    print("----- 5. 有一台可用 -----")
    av = camera_availability(need=1, devices=[device(0)], probe=ok_probe(0))
    print(f"  ok={av.ok} usable={av.usable_count} probed={av.probed} "
          f"reason={av.reason!r}")
    checks = [
        ("ok 为 True", av.ok is True),
        ("usable_count 为 1", av.usable_count == 1),
        ("reason 为空（能用就别吓人）", av.reason == ""),
        ("报出了试开次数（>0，说明真的试了）", len(av.probed) > 0),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_need_two_requires_two() -> bool:
    """第 6 节：双目要 2 台 —— 只有 1 台时必须 ok=False（不许凑合）。"""
    print("----- 6. 需要两台但只有一台 -----")
    only0 = ok_probe(0)
    av = camera_availability(need=2, devices=[device(0), device(1)],
                             probe=only0)
    av1 = camera_availability(need=1, devices=[device(0), device(1)],
                              probe=only0)
    print(f"  need=2 -> ok={av.ok} usable={av.usable_count}（试开 {av.probed}）")
    print(f"  need=1 -> ok={av1.ok} usable={av1.usable_count}（试开 {av1.probed}）")
    checks = [
        ("need=2 时 ok 为 False（一台不许当两台用）", av.ok is False),
        ("need=1 时同一批设备能通过", av1.ok is True),
        ("need=2 时确实试开了更多索引找第二台", len(av.probed) >= 2),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_probe_count_is_bounded() -> bool:
    """第 7 节：**试开次数必须有上限** —— 没相机时每个索引都要等后端超时。

    实测：`probe_index()` 单次最长 **4.4 s**（1280x720 输出全黑后自动降级再试），
    所以 `run_realtime` 没相机时整轮耗 5~9 s，大头全在逐个试开上。
    真机实测里这一段是用户唯一能感知的延迟。

    `camera_availability` 本身不再管上限（它委托 `select_cameras`，
    而上限属于 `select_cameras` 内部的候选表长度）。这里注入一个**会计数的 probe**，
    断言"试开次数不超过候选数量"，把这件事钉住。
    """
    print("----- 7. 试开次数有上限（没相机时最慢的一步）-----")
    calls: list[int] = []

    def counting_probe(i: int) -> ProbeResult:
        calls.append(i)
        return ProbeResult(i, False, 0, 0, 0.0, 0.0, "打不开")

    # 候选都不行时，`select_cameras` 会**回退去试 max_index 范围**（有意的：
    # 防止漏掉设备枚举还没列出来的相机）。所以上限是 max_index+1，不是候选数。
    av = camera_availability(need=1, devices=[device(0), device(1)],
                             probe=counting_probe)
    from sphere_tracker.camera_registry import DEFAULT_SETTINGS  # noqa: E402
    max_index = int(DEFAULT_SETTINGS["max_index"])
    limit = max_index + 1
    print(f"  候选 2 个都打不开 -> 试开 {len(calls)} 次：{calls}")
    print(f"  max_index={max_index} -> 回退范围 0..{max_index}，上限 {limit} 次")
    print(f"  reason={av.reason[:78]}")

    checks = [
        (f"试开次数不超过 {limit}（= max_index + 1，这是真正的契约）",
         len(calls) <= limit),
        ("**确实试了**（不是一次都没试就下结论）", len(calls) > 0),
        ("回报的 probed 与实际调用逐一对得上",
         list(av.probed) == list(range(len(calls)))),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_availability_and_selection_agree() -> bool:
    """第 8 节：**两个判定必须一致** —— 同一个事实不许实现两遍。

    第一版 `camera_availability` 自己重实现了"合格"判据，结果同一台机器上：

    ```
    select_cameras_detailed(need=1) -> ok=True  picks=[#0]
    camera_availability(need=1)     -> ok=False usable=0
    ```

    这正是本项目反复出现的头号问题（同一个量两处各算一遍）。
    现在 `camera_availability` **委托** `select_cameras`，本测试把这条契约钉住：
    给定同样的注入，两者的 ok / 台数必须**永远相同**。
    """
    print("----- 8. 可用性判定与挑选结果必须一致 -----")
    cases = [
        ("没有设备", [], bad_probe("打不开")),
        ("有设备但都打不开", [device(0), device(1)], bad_probe("被占用")),
        ("分辨率不够", [device(0)], ok_probe(0, width=160, height=120)),
        ("有一台可用", [device(0)], ok_probe(0)),
        ("要两台只有一台", [device(0), device(1)], ok_probe(0)),
    ]
    checks = []
    for need in (1, 2):
        for label, devs, probe in cases:
            av = camera_availability(need=need, devices=devs, probe=probe)
            sel = select_cameras_detailed(need=need, devices=devs, probe=probe)
            agree = (av.ok == sel.ok) and (av.usable_count == len(sel.picks))
            print(f"  need={need} {label:<16} av.ok={av.ok!s:<5} "
                  f"usable={av.usable_count}  sel.ok={sel.ok!s:<5} "
                  f"picks={len(sel.picks)}  一致={agree}")
            checks.append((f"need={need} / {label}：两者一致", agree))
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def main() -> int:
    print("=" * 70)
    print("相机可用性判定自测（没有相机时必须说清并指对方向）")
    print("=" * 70)
    results = []
    for fn in (test_no_camera_is_reported_clearly,
               test_existing_but_unusable_is_a_different_message,
               test_resolution_below_requirement_counts_as_unusable,
               test_probe_exception_does_not_crash,
               test_available_camera_reports_ok,
               test_need_two_requires_two,
               test_probe_count_is_bounded,
               test_availability_and_selection_agree):
        results.append(fn())
        print()
    ok = all(results)
    print("判定结果：" + ("全部通过" if ok else "有不达标项"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
