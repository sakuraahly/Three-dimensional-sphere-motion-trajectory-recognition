"""采集健康度自测：`LatestFrameCamera` 必须能说出"相机没了"，而不只是"球没动"。

为什么需要它
------------
改之前，`LatestFrameCamera._run()` 的采集循环是这样：

```python
while not self._stop:
    try:
        ok, frame, _t = self.source.read()
    except Exception:
        time.sleep(0.05)
        continue          # ← 纯静默重试
    if not ok or frame is None:
        time.sleep(0.005)
        continue
    with self._lock:
        self._frame = frame
        self._seq += 1
```

**相机被拔掉 / USB 掉带宽 / 驱动崩掉**之后会发生什么：

1. 采集线程每 50 ms 静默重试一次，**重试到进程结束**；
2. `self._seq` 不再增长，`self._frame` 停在最后一帧；
3. 主循环照常 `latest()` 取帧 —— 拿到的是**同一张旧帧**，于是
   检测、跟踪、预测、HUD **全部继续正常运转**；
4. 程序**不报错、不退出、日志里一行都没有**。

操作员看到画面卡住，而 HUD 上的三维位置是静止的 ——
**"球没动"和"相机没了"长得一模一样**。这是本项目头号成本
（"出事了但没人知道"）最纯粹的形态，也是静默失效普查里唯一的 high。

修法：把"采集出问题了"变成**可观测**的 ——
`read_errors`（连续失败次数，成功清零）+ `last_ok_t`（最后成功时刻），
再由 `stalled()` 给出一个调用方能直接问的判据。

判据
----
- 正常取帧时 `read_errors == 0`、`stalled()` 为 False；
- 读抛异常时 `read_errors` 增长；
- **读"成功但返回空"时也要增长** —— 拔线之后 `read()` 往往返回 `(False, None)`
  而不是抛异常，只计异常的话拔线仍然看不出来（这一条是刻意分开测的）；
- 连续失败超过超时后 `stalled()` 为 True；
- **成功一帧之后必须清零**，否则一次瞬断会永久显示"卡住"。

反向自证（改了就会红）
--------------------
======================================  ==========================================  ========
改动                                    注入方式                                    结果
======================================  ==========================================  ========
不统计异常失败                          删 `self.read_errors += 1`（except 分支）     exit=1
不统计"空帧"失败                        删 `self.read_errors += 1`（空帧分支）        exit=1
成功不清零                              删 `self.read_errors = 0`                    exit=1
不更新 last_ok_t                        删 `self.last_ok_t = time.monotonic()`       exit=1
`stalled()` 恒为 False                  改成 `return False`                          exit=1
======================================  ==========================================  ========

用法::

    python tests/selftest_capture_health.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
while not (ROOT / "sphere_tracker").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from sphere_tracker.sources import LatestFrameCamera

FRAME = np.zeros((8, 8, 3), np.uint8)


class ScriptedSource:
    """按剧本返回结果的假采集源。

    `script` 里每项是 `"ok"` / `"empty"` / `"raise"`；用完之后按最后一项重复，
    这样"一直失败"的场景不用写长剧本。

    ⚠️ **每次 read 都要睡一下**（`delay`）：第一版没睡，采集线程在毫秒内跑了
    **5 万帧**，既白烧 CPU，又让"距最后一次成功多久"这类时间判据在负载下变得脆弱。
    假源也要有接近真实的节奏（真相机 ~30 fps）。

    ⚠️ **时间戳必须用 `perf_counter`**（与 `CameraSource.read()` 同钟）。
    这里原来写的是 `time.monotonic()` —— 两个时钟在 Windows 上起点不同，
    而 `FramePump` 用 `perf_counter` 算帧年龄，混起来会得到"负的年龄"或荒唐的大数。
    测试替身和生产代码用不同的钟，是**测试自己造出来的 bug**，见第 6 节。
    """

    def __init__(self, script: list[str], delay: float = 0.004) -> None:
        self.script = list(script)
        self.calls = 0
        self.delay = delay

    def read(self):
        i = min(self.calls, len(self.script) - 1)
        self.calls += 1
        kind = self.script[i]
        # 按剧本走到哪一步，就在这一步的"节奏"上停一下
        time.sleep(self.delay * (10 if kind != "ok" else 1))
        if kind == "raise":
            raise RuntimeError("模拟驱动崩掉")
        if kind == "empty":
            return False, None, 0.0
        return True, FRAME.copy(), time.perf_counter()


def wait_for(pred, timeout: float = 2.0) -> bool:
    """等 `pred()` 为真（采集是异步的，不能直接断言）。"""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


def make(script: list[str]) -> tuple[LatestFrameCamera, ScriptedSource]:
    src = ScriptedSource(script)
    cam = LatestFrameCamera(src)          # type: ignore[arg-type]
    return cam, src


def stop(cam: LatestFrameCamera) -> None:
    cam._stop = True
    cam._thread.join(timeout=1.0)


# --------------------------------------------------------------------------
def test_healthy_capture_has_no_errors() -> bool:
    """第 1 节：正常取帧时，计数器必须是干净的，且 `last_ok_t` 要**在推进**。"""
    print("----- 1. 正常采集：read_errors 保持 0、last_ok_t 推进 -----")
    cam, src = make(["ok"])
    try:
        wait_for(lambda: cam.latest()[0] >= 3)
        t_first = cam.last_ok_t
        seq, frame, _t_cap = cam.latest()
        # 再等几帧，last_ok_t 必须往前走
        wait_for(lambda: cam.latest()[0] >= seq + 5)
        t_later = cam.last_ok_t
        ok_stalled = cam.stalled(timeout=0.5)
        print(f"  取到 {cam.latest()[0]} 帧（源被读 {src.calls} 次）；"
              f"read_errors={cam.read_errors}；stalled={ok_stalled}")
        print(f"  last_ok_t：{t_first:.3f} -> {t_later:.3f}（必须递增）")
        checks = [
            ("确实采到了帧", frame is not None),
            ("read_errors == 0", cam.read_errors == 0),
            ("stalled() 为 False（正常时不许误报卡住）", ok_stalled is False),
            # ⚠️ 这一条不能写成"距最后成功 > 0.1s" —— `last_ok_t` 的**初值**就满足它，
            # 于是"成功时忘了更新 last_ok_t"这个变异照样能过（实测 M4 没红）。
            # 要证明它在更新，只能比较**两次读取之间**它有没有前进。
            ("last_ok_t 确实随成功帧推进（不是只有初值）", t_later > t_first),
        ]
    finally:
        stop(cam)
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_exceptions_are_counted() -> bool:
    """第 2 节：采集抛异常要计入 `read_errors`（不是静默重试）。"""
    print("----- 2. 采集抛异常 → read_errors 增长 -----")
    cam, _src = make(["ok", "raise"])
    try:
        wait_for(lambda: cam.latest()[0] >= 1)
        grew = wait_for(lambda: cam.read_errors >= 3)
        print(f"  read_errors={cam.read_errors}（期望 ≥3）；seq={cam.latest()[0]}")
        checks = [
            ("异常被计入 read_errors", grew),
            ("seq 停止增长（旧帧被冻结 —— 正因如此才必须能看出卡住）",
             cam.latest()[0] <= 2),
        ]
    finally:
        stop(cam)
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_empty_frames_are_counted_too() -> bool:
    """第 3 节：**读成功但返回空**也要计入 —— 拔线走的是这条路径。

    这一节是刻意与第 2 节分开的：只统计异常的实现会在拔线时给出
    "一切正常"，而那正是最常见的故障。
    """
    print("----- 3. 空帧（拔线的典型形态）→ read_errors 增长 -----")
    cam, _src = make(["ok", "empty"])
    try:
        wait_for(lambda: cam.latest()[0] >= 1)
        grew = wait_for(lambda: cam.read_errors >= 3)
        print(f"  read_errors={cam.read_errors}（期望 ≥3，且**没有抛异常**）")
        checks = [
            ("空帧被计入 read_errors（不抛异常也要能看出来）", grew),
        ]
    finally:
        stop(cam)
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_success_resets_the_counter() -> bool:
    """第 4 节：成功一帧必须清零 —— 否则一次瞬断会永久显示"卡住"。"""
    print("----- 4. 恢复之后 read_errors 清零 -----")
    cam, _src = make(["ok", "raise", "raise", "ok"])
    try:
        wait_for(lambda: cam.read_errors >= 1)
        peak = cam.read_errors
        # 剧本走完 raise 之后回到 ok，计数应当回到 0
        recovered = wait_for(lambda: cam.read_errors == 0 and cam.latest()[0] >= 3)
        print(f"  失败时峰值 {peak}；恢复后 read_errors={cam.read_errors}；"
              f"seq={cam.latest()[0]}")
        checks = [
            ("失败期间计数确实涨过", peak >= 1),
            ("恢复成功后计数清零", recovered),
            ("且不再报 stalled", cam.stalled(timeout=0.5) is False),
        ]
    finally:
        stop(cam)
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_stalled_reports_true_after_timeout() -> bool:
    """第 5 节：连续失败超过超时后，`stalled()` 必须为 True。"""
    print("----- 5. 持续失败 → stalled() 为 True -----")
    cam, _src = make(["raise"])
    try:
        wait_for(lambda: cam.read_errors >= 1)
        # 等超过超时
        became = wait_for(lambda: cam.stalled(timeout=0.15), timeout=2.0)
        age = time.monotonic() - cam.last_ok_t
        print(f"  read_errors={cam.read_errors}；距最后一次成功 {age:.2f}s；"
              f"stalled(timeout=0.15)={cam.stalled(timeout=0.15)}")
        checks = [
            ("stalled() 报 True（调用方这才分得清「球没动」与「相机没了」）", became),
            ("并且 last_ok_t 在那之前（不是一个恒定值）", age > 0.1),
        ]
    finally:
        stop(cam)
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_frame_time_travels_with_frame() -> bool:
    """第 6 节：**采集时刻必须跟着帧一起交到消费方手里**（交接文档 §P0-1 判据①②）。

    ## 这条守的是什么（2026-10-27）

    `LatestFrameCamera._run` 原来把 `read()` 的第三个值直接扔了：

    ```python
    ok, frame, _t = self.source.read()      # ← 时间在这儿被丢掉
    ...
    def latest(self, last_seq=-1): return self._seq, self._frame   # 只回两个
    ```

    后果链：消费方拿不到帧时间 → 只能用"我开始处理它的时刻"顶替 →
    处理慢一点，整条时间轴就往后移，而且**随负载抖动** → 预测器吃的是这条轴。
    项目自己量化过：滞后 1 帧（17 ms）→ 三维误差 **440 mm**（34 倍）。

    所以判据是三条，缺一不可：

    1. **单调**：seq 前进时 `t_capture` 不许倒退；
    2. **同一帧两次取出同一个时间**（若每次刷新 = 还是"取的时刻"，不是采集时刻）；
    3. **同钟**：`t_capture` 与消费方 `perf_counter()` 之差必须是个**小正数**
       —— 这一条专门抓"假源用 monotonic、生产用 perf_counter"这种测试自己造的 bug。
    """
    print("----- 6. 采集时刻随帧走（单调 / 同一帧不刷新 / 与消费方同钟）-----")
    cam, _src = make(["ok"])
    try:
        wait_for(lambda: cam.latest()[0] >= 5)
        s1, _f1, t1 = cam.latest()
        # 同一帧连取两次：时间不许变（这是"采集时刻"与"取的时刻"的分水岭）
        s2, _f2, t2 = cam.latest()
        same_frame_same_t = (s1 == s2) and (t1 == t2)
        # 再等几帧，时间必须往前走
        wait_for(lambda: cam.latest()[0] >= s1 + 5)
        s3, _f3, t3 = cam.latest()
        # 同钟验证：拿 perf_counter 去减，差值必须落在"刚采到"的范围内
        age_ms = (time.perf_counter() - t3) * 1000.0
        print(f"  seq {s1}->{s3}：t {t1:.4f} -> {t3:.4f}"
              f"（递增={t3 > t1}）；同一帧两次取 t 相同={same_frame_same_t}")
        print(f"  消费方用 perf_counter 算出的帧年龄 = {age_ms:.1f}ms"
              f"（同一条时钟链，应落在 0~200ms）")
        checks = [
            ("seq 前进时 t_capture 递增（单调）", t3 > t1),
            ("同一帧被取两次时 t_capture 不刷新", same_frame_same_t),
            ("t_capture 与消费方同钟（年龄是合理的小正数，不是垃圾值）",
             0.0 <= age_ms <= 200.0),
        ]
    finally:
        stop(cam)
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def main() -> int:
    print("=" * 70)
    print("采集健康度自测（相机没了 ≠ 球没动）")
    print("=" * 70)
    results = []
    for fn in (test_healthy_capture_has_no_errors,
               test_exceptions_are_counted,
               test_empty_frames_are_counted_too,
               test_success_resets_the_counter,
               test_stalled_reports_true_after_timeout,
               test_frame_time_travels_with_frame):
        results.append(fn())
        print()
    ok = all(results)
    print("判定结果：" + ("全部通过" if ok else "有不达标项"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
