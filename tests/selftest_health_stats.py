"""运行健康度指标自测：`Health` 的字段语义、派生量、往返一致、以及"来源纯净"。

为什么需要它
------------
项目里"帧率"一词同时指三件事（采集 / 主循环 / 检测耗时），**已经因此走错过一次方向**
—— `交接文档.md` §10.10 记着：界面上那个 `detect 19 ms/frame (52 fps)` 量的是
**检测计算耗时**，被读成"相机采集帧率"，于是整轮排查都在查错方向。

所以本测试钉四件事：

**1. 三个帧率口径必须分得开。** 造一个"采集 30 fps、处理只有 5 fps"的场景，
断言 `capture_fps≈30`、`loop_fps≈5`、`detect_ms` 各归各 —— 混在一起就报红。

**2. `frame_age_ms` 与 `dropped` 的语义。** 这两个指标回答的是"用户看到的画面有多旧"。
`dropped` 的定义写死为「**本帧在被取走之前被下一帧覆盖**」——
因为现有 `LatestFrameCamera` 是**锁内原地覆盖、没有队列**，"丢"是"被覆盖"，
不是"进队列失败"。定义不写死，这个计数就永远说不清。

**3. 派生量不是字段。** `capture_fps` / `loop_fps` / `drop_rate` 都由方法算，
`to_dict` 里出现它们只是**只读快照**，`from_dict` 必须**丢弃**它们
（否则就有两个真相：一个是累加出来的，一个是反序列化读回来的）。

**4. 模块来源必须纯净**（计划书 §5.1 的边界：纯数据、无 IO、无本项目依赖）。

反向自证（改了就会红）
--------------------
===============================  ========================================  ========
改动                             注入方式                                  结果
===============================  ========================================  ========
让 loop_fps 去用采集帧率          `_rate(self.captured_frames, ...)`        exit=1
让 from_dict 也读派生量           `known = {...} | {"loop_fps"}`            exit=1
给 stats.py 加一个本项目 import   `from .types import FrameResult`          exit=1
把 dropped 改成"取帧就 +1"        `note_capture` 里加 `dropped_frames += 1` exit=1
===============================  ========================================  ========

> 后两条（`from_dict` 读派生量、加本项目 import）在变异下是**直接抛异常**而不是
> 逐项报红 —— 所以本测试把 `from_dict` 的调用包了 try，让"崩"也变成一条可读的
> `[不达标]`，而不是一屏 traceback。

用法::

    python tests/selftest_health_stats.py
"""

from __future__ import annotations

import ast
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
while not (ROOT / "sphere_tracker").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sphere_tracker.stats import FramePump, Health, StageTimes  # noqa: E402

STATS_PY = ROOT / "sphere_tracker" / "stats.py"


def test_three_frame_rates_are_distinct() -> bool:
    """第 1 节：采集 / 主循环 / 检测耗时 三个口径分得开。"""
    print("----- 1. 三个帧率口径分得开（本项目走错过方向的那件事） -----")
    h = Health()
    # 采集侧：30 fps（每帧 33.3 ms），采了 300 帧
    for _ in range(300):
        h.note_capture(capture_ms=1000.0 / 30.0, frame_age_ms=120.0)
    # 处理侧：只有 5 fps（每帧 200 ms），其中检测 180 ms
    for _ in range(50):
        h.begin_frame()
        h.end_frame(detect_ms=180.0, track_ms=15.0, render_ms=5.0)

    print(f"  capture_fps = {h.capture_fps:7.2f}（期望 ≈30）")
    print(f"  loop_fps    = {h.loop_fps:7.2f}（期望 ≈5）")
    print(f"  detect_ms   = {h.stages.detect_ms:7.2f}（期望 180）")
    print(f"  frame_age_ms= {h.frame_age_ms:7.2f}（期望 120）")
    print(f"  drop_rate   = {h.drop_rate:7.4f}")
    print(f"  summary_line: {h.summary_line()}")

    checks = [
        ("capture_fps ≈ 30（采集线程的速率）", abs(h.capture_fps - 30.0) < 0.5),
        ("loop_fps ≈ 5（主循环的速率，**与采集差 6 倍**）", abs(h.loop_fps - 5.0) < 0.2),
        ("两者确实不同（差 > 5 倍）—— 混成一个数就报红",
         h.capture_fps / max(h.loop_fps, 1e-9) > 5.0),
        ("detect_ms 单独可读（不被帧率吞掉）", abs(h.stages.detect_ms - 180.0) < 1e-6),
        ("frame_age_ms 单独可读", abs(h.frame_age_ms - 120.0) < 1e-6),
        ("summary_line 里**没有**裸的 `FPS` 字样（只有 cap / loop / detect）",
         "FPS" not in h.summary_line().upper().replace("CAP", "").replace("LOOP", "")),
        ("summary_line 同时含 cap 与 loop 两个数",
         "cap" in h.summary_line() and "loop" in h.summary_line()),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_frame_age_and_dropped_semantics() -> bool:
    """第 2 节：`dropped` = 被下一帧覆盖的次数；`frame_age_ms` = 画面有多旧。"""
    print("----- 2. dropped（被覆盖）与 frame_age_ms（画面有多旧） -----")
    h = Health()
    # 采集 10 帧；其中 3 帧在被取走之前就被覆盖了
    for i in range(10):
        h.note_capture(capture_ms=10.0, frame_age_ms=float(i))
    h.note_dropped(3)
    print(f"  captured={h.captured_frames}  dropped={h.dropped_frames}"
          f"  drop_rate={h.drop_rate:.3f}  frame_age_ms={h.frame_age_ms:.1f}")

    checks = [
        ("captured_frames = 10", h.captured_frames == 10),
        ("dropped_frames = 3（**只由 note_dropped 贡献**，取帧不加）",
         h.dropped_frames == 3),
        ("drop_rate = 3/(10+3)", abs(h.drop_rate - 3 / 13) < 1e-9),
        ("frame_age_ms = 最后一帧的年龄（9.0）", abs(h.frame_age_ms - 9.0) < 1e-9),
        ("note_dropped 不接受负数（不把计数搞坏）",
         (lambda g: (g.note_dropped(-5), g.dropped_frames == 0)[1])(Health())),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_derived_values_are_not_fields() -> bool:
    """第 3 节：派生量只读快照，反序列化时必须丢弃（不许有两个真相）。"""
    print("----- 3. 派生量的往返一致（不许有两个真相） -----")
    h = Health()
    for _ in range(30):
        h.note_capture(capture_ms=50.0, frame_age_ms=70.0)
    h.note_dropped(2)
    h.skew_ms, h.skew_ms_valid = -17.5, True
    for _ in range(10):
        h.begin_frame()
        h.end_frame(detect_ms=40.0, track_ms=5.0, render_ms=1.0)

    d = h.to_dict()
    roundtrip_error = ""
    try:
        h2 = Health.from_dict(d)
        d2 = h2.to_dict()
    except Exception as exc:                      # noqa: BLE001
        # 变异测试逼出来的写法：如果 `from_dict` 把派生量也当成字段读回来，
        # 这里会抛 TypeError。让它变成一条可读的 [不达标]，而不是一屏 traceback。
        h2, d2 = None, {}
        roundtrip_error = f"{type(exc).__name__}: {exc}"

    print(f"  to_dict 键：{sorted(d)}")
    if roundtrip_error:
        print(f"  **往返失败**：{roundtrip_error}")
    else:
        print(f"  往返后 capture_fps {d['capture_fps']} -> {d2['capture_fps']}"
              f" / loop_fps {d['loop_fps']} -> {d2['loop_fps']}")
        print(f"  skew_ms={h2.skew_ms} valid={h2.skew_ms_valid}"
              f"  stages={h2.stages}")

    checks = [
        ("to_dict 里含派生量 capture_fps / loop_fps / drop_rate",
         {"capture_fps", "loop_fps", "drop_rate"} <= set(d)),
        (f"往返不抛异常（{roundtrip_error or '正常'}）", not roundtrip_error),
        ("往返后全部字段逐字一致", bool(d2) and d == d2),
        ("往返后派生量也一致（cap/loop/drop_rate）",
         bool(d2) and d["capture_fps"] == d2["capture_fps"]
         and d["loop_fps"] == d2["loop_fps"]
         and d["drop_rate"] == d2["drop_rate"]),
        ("**派生量不是 dataclass 字段**（不在 __dataclass_fields__ 里）",
         not ({"capture_fps", "loop_fps", "drop_rate"}
              & set(Health.__dataclass_fields__))),
        ("stages 正确还原为 StageTimes（不是 dict）",
         isinstance(h2.stages, StageTimes) if h2 else False),
        ("skew_ms_valid 往返保留（False 不许被当成 0 真值）",
         bool(h2) and h2.skew_ms_valid is True and abs(h2.skew_ms + 17.5) < 1e-9),
        ("默认 skew_ms_valid 是 False（没估出来就说没估出来）",
         Health().skew_ms_valid is False),
        ("summary_line 在 skew 无效时**不报** skew",
         "skew" not in Health().summary_line()),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_module_is_pure() -> bool:
    """第 4 节：`stats.py` 必须纯 —— 无本项目依赖、无 IO、无 print（计划书 §5.1）。"""
    print("----- 4. stats.py 的来源纯净（纯数据模块的边界） -----")
    src = STATS_PY.read_text(encoding="utf-8")
    tree = ast.parse(src)

    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mod = ("." * node.level) + (node.module or "")
            imported += [mod]

    calls = {n.func.id for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}

    project_imports = [m for m in imported
                       if m.startswith(".") or m.split(".")[0] in {
                           "sphere_tracker", "calib", "stereo", "apps"}]
    io_calls = sorted(calls & {"open", "print", "input"})

    print(f"  import：{imported}")
    print(f"  本项目 import：{project_imports or '（无）✅'}")
    print(f"  IO / 输出调用：{io_calls or '（无）✅'}")

    checks = [
        ("没有 import 本项目任何模块（纯数据，最底层）", not project_imports),
        ("没有 open / print / input（不碰 IO、不往终端写）", not io_calls),
        ("模块可以单独 import（不依赖包内其他模块初始化）",
         _imports_standalone()),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def _imports_standalone() -> bool:
    """把 stats.py 当独立文件加载（不经包），能成功即说明它自足。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_stats_standalone", STATS_PY)
    if spec is None or spec.loader is None:
        return False
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_stats_standalone"] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:                      # noqa: BLE001
        print(f"    （独立加载失败：{type(exc).__name__}: {exc}）")
        return False
    return hasattr(mod, "Health")


def test_frame_pump_bookkeeping() -> bool:
    """`FramePump`：序号跳变算丢帧、重复帧不计数、停住能报出来。

    ## 为什么必须有这一节

    这段算术原先**手写在 `apps/run_realtime.py` 的主循环里**：
    比序号算跳过帧、按 `last_ok_t` 算帧年龄、按两次取帧间隔算采集间隔。
    它**不可测** —— `run_realtime` 没有相机就跑不起来。

    证据（变异测试实测）：把 `health=health` 从 `run_realtime` 的
    `pipe.render(...)` 调用里删掉，**整套回归全绿** ——
    因为没有任何测试经过那条路径。抽成 `FramePump` 之后，用假源就能走一遍。
    """
    print("----- 5. FramePump：取帧簿记 -----")

    class FakeCam:
        """假的"最新帧"源，按剧本给出 `(seq, frame[, t_capture])`，并可切换 stalled。

        `with_time=False` 用来模拟"源忘了回采集时刻"——那必须当场炸（见下面第 8 条）。
        """

        def __init__(self, steps, *, with_time=True):
            self.steps = list(steps)
            self.i = -1
            self.with_time = with_time
            self.last_ok_t = time.monotonic()
            self._stalled = False

        def latest(self, last_seq=-1):
            if self.i + 1 < len(self.steps):
                self.i += 1
            step = self.steps[self.i]
            seq, frame = step[0], step[1]
            if not self.with_time:
                return seq, frame
            t_cap = step[2] if len(step) > 2 else time.perf_counter()
            return seq, frame, t_cap

        def stalled(self, timeout=1.0):
            return self._stalled

    def run(steps, n):
        h = Health()
        cam = FakeCam(steps)
        pump = FramePump(cam, h, stalled_after_s=0.001)
        return h, [pump.next() for _ in range(n)]

    f = object()
    checks = []

    # 1) 正常逐帧推进
    h, out = run([(0, f), (1, f), (2, f)], 3)
    print(f"  逐帧推进：seq={[o.seq for o in out]} dropped={h.dropped_frames} "
          f"captured={h.captured_frames}")
    checks += [
        ("逐帧推进时 dropped 为 0", h.dropped_frames == 0),
        ("captured_frames 等于新帧数（3）", h.captured_frames == 3),
        ("每帧都标成 is_new", all(o.is_new for o in out)),
    ]

    # 2) 序号跳变 = 中间的帧被覆盖了
    h, out = run([(0, f), (5, f)], 2)
    print(f"  seq 0 -> 5：skipped={out[1].skipped} dropped={h.dropped_frames}"
          f"（中间 1/2/3/4 共 4 帧被盖）")
    checks += [
        ("序号跳变算出的 skipped 是差额减一（4）", out[1].skipped == 4),
        ("dropped_frames 累加了它", h.dropped_frames == 4),
    ]

    # 3) 同一帧重复取：不许改任何计数（否则 capture_fps 会虚高）
    h, out = run([(0, f), (0, f), (0, f)], 3)
    print(f"  同一帧取三次：captured={h.captured_frames} is_new={[o.is_new for o in out]}")
    checks += [
        ("重复取同一帧只算一次采集", h.captured_frames == 1),
        ("后续两次标成 is_new=False（调用方据此跳过处理）",
         bool(out[0].is_new) and not out[1].is_new and not out[2].is_new),
    ]

    # 4) 还没有帧（相机刚打开）
    h, out = run([(-1, None)], 1)
    print(f"  还没有帧：frame={out[0].frame} captured={h.captured_frames}")
    checks += [
        ("frame 为 None 时不算采集", h.captured_frames == 0),
        ("也不抛异常", out[0].frame is None),
    ]

    # 5) 停住要能报出来（HUD 的 STALL 靠它）
    h = Health()
    cam = FakeCam([(0, f)])
    pump = FramePump(cam, h, stalled_after_s=0.001)
    pump.next()
    cam._stalled = True
    got = pump.next()
    print(f"  源报 stalled：PumpResult.stalled={got.stalled}")
    checks += [("stalled 能透传出来", got.stalled is True)]

    # 6) 源不支持 stalled 时不许谎报
    class NoStall:
        last_ok_t = time.monotonic()

        def latest(self, last_seq=-1):
            return 0, f, time.perf_counter()

    got = FramePump(NoStall(), Health()).next()
    print(f"  源没有 stalled()：报 {got.stalled}（必须 False —— 不谎报停住）")
    checks += [("源不支持 stalled 时报 False（不谎报）", got.stalled is False)]

    # 7) 帧年龄必须按**这一帧自带的采集时刻**算（交接文档 §P0-1 判据②）
    #
    # 改之前：`age = now - source.last_ok_t`。`last_ok_t` 是"最后一次成功取帧"的时刻，
    # 对**当前这一帧**来说它是错的（拿到旧帧时会算成 0）—— 也就是"看起来永远新鲜"。
    t_cap = time.perf_counter() - 0.12
    got = FramePump(FakeCam([(0, f, t_cap)]), Health()).next()
    print(f"  注入 120ms 前采集的帧：frame_age_ms={got.frame_age_ms:.1f}（应 ≈120）"
          f" frame_t 透传={abs(got.frame_t - t_cap) < 1e-9}")
    checks += [
        ("帧年龄按帧自带的 t_capture 算（≈120ms，不是 0）",
         100.0 <= got.frame_age_ms <= 160.0),
        ("PumpResult.frame_t 原样透传（调用方要用它当时间轴）",
         abs(got.frame_t - t_cap) < 1e-9),
    ]

    # 8) 源不回第三个值 -> **当场炸**，不许静默退化成"年龄 0"
    class NoTime:
        def latest(self, last_seq=-1):
            return 0, f

    try:
        FramePump(NoTime(), Health()).next()
        checks.append(("latest() 缺 t_capture 必须抛 TypeError", False))
        print("  源只回两个值：没抛异常（不达标）")
    except TypeError as exc:
        ok_msg = "t_capture" in str(exc)
        print(f"  源只回两个值：抛 TypeError，消息提到 t_capture={ok_msg}")
        checks.append(("latest() 缺 t_capture 抛 TypeError 且点名原因", ok_msg))

    checks += [("stats 仍是纯数据 + 取帧簿记（不 import 本项目其他模块）",
                not _imports_project_modules())]

    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def _imports_project_modules() -> bool:
    """`stats.py` 有没有 import 本项目其他模块（它必须是叶子）。"""
    import re
    src = (ROOT / "sphere_tracker" / "stats.py").read_text(encoding="utf-8")
    mods = set(re.findall(r"^\s*(?:from|import)\s+([\w.]+)", src, flags=re.MULTILINE))
    return any(m.split(".")[0] in {"sphere_tracker", "calib", "stereo"} for m in mods)


def main() -> int:
    print("=" * 70)
    print("运行健康度指标自测（Health / StageTimes / FramePump）")
    print("=" * 70)
    results = []
    for fn in (test_three_frame_rates_are_distinct,
               test_frame_age_and_dropped_semantics,
               test_derived_values_are_not_fields,
               test_module_is_pure,
               test_frame_pump_bookkeeping):
        results.append(fn())
        print()
    ok = all(results)
    print("判定结果：" + ("全部通过" if ok else "有不达标项"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
