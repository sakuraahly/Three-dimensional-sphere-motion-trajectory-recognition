"""高帧率 / 高分辨率档位扩容自测：把 2026-11-18 那两包建立的不变量钉死。

为什么需要它
------------
这一轮给"以后要接 >=500 fps、4K 以上分辨率相机"做铺垫，动了四类**默认值**：
档位表、默认档、帧率类型、三个"按帧说话"的门限。它们的共同点是
**改错了不会报错**，只会让录制/标定静默换一个档、或让门限变成形同虚设：

- 在 ``PROFILES`` 顶部加两档，``PROFILES[1]`` 就从 2560x1440 静默漂成 5120x2880，
  而默认档漂移**不会报错**，只会让所有录制/标定静默换分辨率（已踩过）；
- 自动协商若跟着档位表从 8K 起试，每多试一档就多一次相机打开尝试
  （个别后端单次 20 s+），所有**现有**相机都会变慢；
- ``--fps`` 用 ``int`` 会把 29.97 截成 29，录 1 分钟少 1.8 帧；
- 30 fps 下调出来的 60 ms / 180 帧门限，在 500 fps 下分别是"30 帧"和"0.36 秒" ——
  一个是形同不设防，一个是搜索窗窄到必然失配。反过来，**24/30 fps 必须逐项不变**，
  否则就是拿现有相机的行为去换一个还没接上的高帧率能力。

所以本文件守的不是"新功能能用"，而是"**扩了档，老行为一个字节都没动**"。

九组判据
--------
① ``probe_order(None)`` 与扩容前**逐项相同**（7 项，不含 5K/8K），显式档不受上限限制；
② ``DEFAULT`` / ``RECORD_DEFAULT`` 是**对象引用**（不是下标），且源码里不许出现
   ``PROFILES[数字]``；
③ ``fps_type`` 保留小数、非法输入抛 ``ArgumentTypeError`` 且带原文；
④ 5 个 CLI 入口的 ``--fps`` 真的收小数（``_common`` 功能验证，其余源码扫描）；
⑤ ``default_max_skew_ms``：24/30 fps 恒 60 ms，31 fps 起收紧，下限 2 ms；
⑥ ``config_for_fps``：24/30 fps 必须等于旧硬编码的 180/30；
⑦ ``stream_bandwidth`` 的估算值与告警判据（按**单路**判，不是两路合计）;
⑧ ``timestamp_tolerance_s`` 固定 0.033，**不**按帧率派生；
⑨ ``fps: int`` 注解已清（帧率全链是浮点）。

反向自证（改了就会红，2026-11-18 实测）
------------------------------------
1. 自动协商跟着档位表从 8K 起试（``if profile.pixels <= ceiling.pixels:`` -> ``if True:``）
   -> ① 里 5 项变红：逐项比较（拿到 9 项、8K 打头）、长度 9 项、含 5K、含 8K、首项不是 4K。
2. 30 fps 边界抬到 100（``f <= 30.0`` -> ``f <= 100.0``）
   -> ⑤ 里 2 项变红：「31 fps 起才开始收紧」（实测 60.0000 ms）、
   「31 fps == 60*30/31」（实测 60.0）。
3. 闪光搜索窗下限去掉（``max(180, ...)`` -> ``max(0, ...)``）
   -> ⑥ 里 1 项变红：「24 fps -> 180/30」（实测 144/30）。
   **30 fps 那条不会红**（6s*30 本来就是 180）—— 所以 24 fps 这一条是唯一守门人。
4. 默认档退回下标（``DEFAULT = P1440`` -> ``PROFILES[1]``）
   -> ② 里 4 项变红：``DEFAULT is P1440``、``DEFAULT is RECORD_DEFAULT``、
   本应 2560x1440 实测 5120x2880、源码扫描报 ``video_profiles.py:129``。

每次变异后都已还原（三个文件的 sha256 与改前逐字节相同），本文件不修改任何源码。

用法::

    python tests/selftest_highspeed.py
"""

from __future__ import annotations

import argparse
import inspect
import io
import re
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent
while not (ROOT / "sphere_tracker").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sphere_tracker import video_profiles as vp        # noqa: E402


# ---------------------------------------------------------------------------
# 源码扫描的小工具（三条判据靠它，不是靠"读一眼代码"）
# ---------------------------------------------------------------------------
def _src(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _line_offsets(src: str) -> list[int]:
    """每一行起始字符的绝对偏移（tokenize 给的是 (行,列)，换算成切片要用）。"""
    offs = [0]
    for line in src.splitlines(keepends=True):
        offs.append(offs[-1] + len(line))
    return offs


def _code_only(src: str) -> str:
    """把注释与字符串字面量**挖成空格**，其余原样保留（行号、列号都不变）。

    为什么不能只做 `grep`：本轮源码里**故意**留着 ``PROFILES[1]`` 与 ``0.020``
    的注释，解释"为什么不能这么写"。只看原始文本会把解释当成违规（假红），
    而删掉解释又等于把踩过的坑抹掉。所以判据只看**代码**：
    注释、docstring、字符串里的同名字样都不算数。

    用 tokenize 而不是"跳过以 # 开头的行"：后者漏掉行尾注释，
    也分不清 ``"...#..."`` 这种字符串。3.12+ 的 f-string 是
    FSTRING_START/MIDDLE/END 三个 token，MIDDLE 才是字面量部分，一并挖掉。
    """
    toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    offs = _line_offsets(src)
    blank_types = {tokenize.COMMENT, tokenize.STRING,
                   getattr(tokenize, "FSTRING_MIDDLE", -1)}

    def pos(p: tuple[int, int]) -> int:
        return offs[p[0] - 1] + p[1]

    chars = list(src)
    for tok in toks:
        if tok.type not in blank_types:
            continue
        for i in range(pos(tok.start), pos(tok.end)):
            if chars[i] not in "\r\n":
                chars[i] = " "
    return "".join(chars)


def _add_argument_slices(src: str) -> list[tuple[int, str]]:
    """每个 ``add_argument(...)`` 调用的 ``(起始行, 原文)``（括号按 token 配对）。

    为什么要"整段"而不是逐行：``--fps`` 的 ``type=`` 与它常在**不同行**
    （多行 help 会把它挤下去），逐行 grep 会漏；而"文件里同时出现 --fps 和
    type=int"又会误报（别的参数用 int）。按调用切段才问得准。
    """
    toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    offs = _line_offsets(src)

    def pos(p: tuple[int, int]) -> int:
        return offs[p[0] - 1] + p[1]

    out: list[tuple[int, str]] = []
    for i, tok in enumerate(toks):
        if tok.type != tokenize.NAME or tok.string != "add_argument":
            continue
        j = i + 1
        while j < len(toks) and not (toks[j].type == tokenize.OP
                                     and toks[j].string == "("):
            j += 1
        if j >= len(toks):
            continue
        depth = 0
        for k in range(j, len(toks)):
            tk = toks[k]
            if tk.type != tokenize.OP:
                continue
            if tk.string in "([{":
                depth += 1
            elif tk.string in ")]}":
                depth -= 1
                if depth == 0:
                    out.append((tok.start[0], src[pos(tok.start):pos(tk.end)]))
                    break
    return out


def _ck(checks: list[tuple[str, bool]], label: str, ok: bool,
        detail: str = "") -> None:
    print(f"  [{'通过' if ok else '不达标'}] {label}"
          + (f"（{detail}）" if detail else ""))
    checks.append((label, ok))


def _flat(text: str, limit: int = 220) -> str:
    """把多行片段压成一行打印（失败时给人看"它实际长什么样"）。"""
    one = " ".join(text.split())
    return one if len(one) <= limit else one[:limit] + "…"


# ---------------------------------------------------------------------------
# ① 自动协商的梯子不许被加档改变
# ---------------------------------------------------------------------------
def test_probe_ladder_unchanged() -> bool:
    """``probe_order(None)`` 必须与扩容前**逐项相同**。

    这不是"差不多"：每多一项就是多一次相机打开尝试，个别后端单次 20 s+。
    现有相机全是 1080p/1440p，从 8K 起试对它们只有代价、没有任何收益。
    """
    print("----- ① 自动协商仍从 4K 起、仍只 7 项（5K/8K 只在显式选择时生效） -----")
    checks: list[tuple[str, bool]] = []
    expected = [(3840, 2160), (2560, 1440), (1920, 1080),
                (1280, 720), (960, 540), (640, 480), (0, 0)]

    got = vp.probe_order(None)
    _ck(checks, "probe_order(None) 逐项与扩容前相同", got == expected, str(got))
    _ck(checks, "长度仍是 7", len(got) == 7, f"{len(got)} 项")
    _ck(checks, "自动协商不含 5120x2880", (5120, 2880) not in got)
    _ck(checks, "自动协商不含 7680x4320", (7680, 4320) not in got)
    _ck(checks, "PROBE_CEILING 是 P4K 这个对象", vp.PROBE_CEILING is vp.P4K)
    _ck(checks, "自动协商首项仍是 4K（不是 8K）", got[0] == (3840, 2160), str(got[0]))

    o8 = vp.probe_order(vp.P8K)
    _ck(checks, "显式 8K：首项是 8K 自身（不受 ceiling 限制）", o8[0] == (7680, 4320),
        str(o8[:2]))
    _ck(checks, "显式 8K：次项是 5K（往下走，不跳档）", o8[1] == (5120, 2880),
        str(o8[:2]))
    _ck(checks, "显式 8K：仍含 4K 与 1080p（降档链没断）",
        (3840, 2160) in o8 and (1920, 1080) in o8)

    return all(ok for _n, ok in checks)


# ---------------------------------------------------------------------------
# ② 默认档不许被新档挤走
# ---------------------------------------------------------------------------
def test_default_is_object_reference() -> bool:
    """默认档必须是 ``P1440`` 这个**对象**，且源码里不许按下标取。

    对象同一性（``is``）而不是相等（``==``）：按值相等的话，
    "两个 2560x1440 的实例"也能过，而那种写法正是下一次加档漂移的温床。
    """
    print("----- ② 默认档是对象引用（不是 PROFILES[下标]） -----")
    checks: list[tuple[str, bool]] = []

    _ck(checks, "DEFAULT is P1440", vp.DEFAULT is vp.P1440)
    _ck(checks, "RECORD_DEFAULT is P1440", vp.RECORD_DEFAULT is vp.P1440)
    _ck(checks, "DEFAULT is RECORD_DEFAULT（仍是同一个对象）",
        vp.DEFAULT is vp.RECORD_DEFAULT)
    _ck(checks, 'DEFAULT.label == "2560x1440"', vp.DEFAULT.label == "2560x1440",
        vp.DEFAULT.label)
    _ck(checks, "P1440 在档位表里", vp.P1440 in vp.PROFILES)

    # 源码扫描：加档只改 PROFILES 那张表，别碰对象引用（否则下一次加档又漂）。
    pattern = re.compile(r"PROFILES\s*\[\s*\d")
    hits: list[str] = []
    files = sorted((ROOT / "sphere_tracker").rglob("*.py")) + \
        sorted((ROOT / "apps").rglob("*.py"))
    for path in files:
        code = _code_only(path.read_text(encoding="utf-8"))
        for lineno, line in enumerate(code.splitlines(), 1):
            if pattern.search(line):
                hits.append(f"{path.relative_to(ROOT).as_posix()}:{lineno}: "
                            f"{line.strip()}")
    _ck(checks, "sphere_tracker/ 与 apps/ 的**代码**里没有 PROFILES[数字下标]",
        not hits,
        "; ".join(hits[:4]) if hits else f"扫描 {len(files)} 个 .py，0 命中")

    return all(ok for _n, ok in checks)


# ---------------------------------------------------------------------------
# ③ 帧率小数不许被截断
# ---------------------------------------------------------------------------
def test_fps_type_keeps_fraction() -> bool:
    """``29.97`` 必须原样保留：截成 29 会以"录 1 分钟少 1.8 帧"的形式累积。"""
    print("----- ③ fps_type 保留小数、非法输入变用法错误 -----")
    checks: list[tuple[str, bool]] = []

    got = vp.fps_type("29.97")
    _ck(checks, 'fps_type("29.97") == 29.97', got == 29.97, repr(got))
    _ck(checks, 'fps_type("29.97") 的类型是 float', type(got) is float,
        type(got).__name__)
    _ck(checks, 'fps_type("29.97") != 29（没被截断）', got != 29)
    _ck(checks, 'fps_type("500") == 500.0', vp.fps_type("500") == 500.0)

    for bad in ("abc", "", "0", "-1", "1001"):
        raised = False
        msg = ""
        try:
            vp.fps_type(bad)
        except argparse.ArgumentTypeError as exc:
            raised = True
            msg = str(exc)
        except Exception as exc:                              # noqa: BLE001
            msg = f"{type(exc).__name__}: {exc}"
        # 异常文本必须带**收到的原文**：argparse 会把这句话原样打给用户，
        # 不带原文的话"我明明填了 1001 为什么不行"就没法自查。
        _ck(checks, f"fps_type({bad!r}) 抛 ArgumentTypeError 且带原文",
            raised and repr(bad) in msg, msg[:80])

    _ck(checks, "FPS_MAX 仍是 1000（1001 越界）", vp.FPS_MAX == 1000.0, repr(vp.FPS_MAX))
    _ck(checks, "FPS_TIERS 仍是 7 档建议值（只做下拉，不限制输入）",
        vp.FPS_TIERS == (24, 30, 60, 120, 240, 500, 1000), str(vp.FPS_TIERS))

    return all(ok for _n, ok in checks)


# ---------------------------------------------------------------------------
# ④ 5 个 CLI 入口的 --fps 真的能收小数
# ---------------------------------------------------------------------------
CLI_FPS_FILES = (
    "apps/record_stereo.py",
    "apps/_common.py",
    "apps/run_realtime.py",
    "apps/tools/capture_board.py",
    "apps/tools/list_cameras.py",
)


def test_cli_fps_accepts_fraction() -> bool:
    """``_common`` 真调一次；其余 4 个的 parser 在 ``main()`` 里造，取不到就扫源码。

    为什么不全靠源码扫描：``type=fps_type`` 写在文件里不等于它挂在 ``--fps`` 上
    （可能是别的参数），所以能真跑的那一个必须真跑。
    """
    print("----- ④ 5 个 CLI 入口的 --fps 收小数 -----")
    checks: list[tuple[str, bool]] = []

    from apps._common import add_stereo_source_args

    ap = argparse.ArgumentParser()
    add_stereo_source_args(ap)
    args = ap.parse_args(["--fps", "29.97"])
    _ck(checks, "_common: --fps 29.97 -> 29.97", args.fps == 29.97, repr(args.fps))
    _ck(checks, "_common: --fps 29.97 的类型是 float", type(args.fps) is float,
        type(args.fps).__name__)
    _ck(checks, "_common: 不给 --fps 时默认仍是 24（老行为不变）",
        ap.parse_args([]).fps == 24, repr(ap.parse_args([]).fps))

    for rel in CLI_FPS_FILES:
        src = _src(rel)
        slices = [text for _line, text in _add_argument_slices(src)
                  if '"--fps"' in text]
        ok = (len(slices) == 1
              and ("type=fps_type" in slices[0]
                   or "type=video_profiles.fps_type" in slices[0])
              and "type=int" not in slices[0])
        detail = (_flat(slices[0]) if slices
                  else f"没找到 --fps 的 add_argument（共 {len(slices)} 段）")
        _ck(checks, f"{rel}: --fps 用 fps_type、不是 type=int", ok, detail)

    return all(ok for _n, ok in checks)


# ---------------------------------------------------------------------------
# ⑤ 24/30 fps 的现有行为逐字节不变
# ---------------------------------------------------------------------------
def test_max_skew_derivation() -> bool:
    """``default_max_skew_ms``：30 fps 及以下恒 60 ms，更高按 60*30/fps 收紧。

    浮点比较一律用 ``abs(a-b) < 1e-9``：``60*30/500`` 的实际结果是
    3.5999999999999996，直接 ``== 3.6`` 会假红。
    """
    print("----- ⑤ 两路时间差门限：24/30 fps 不变，高帧率收紧 -----")
    checks: list[tuple[str, bool]] = []

    from apps import run_realtime_stereo as rrs

    cases = ((24, 60.0), (30, 60.0), (500, 3.6), (1000, 2.0), (2000, 2.0),
             (0, 60.0), (-5, 60.0))
    for fps, want in cases:
        got = rrs.default_max_skew_ms(fps)
        _ck(checks, f"default_max_skew_ms({fps}) == {want:g}", abs(got - want) < 1e-9,
            f"{got!r}")

    at31 = rrs.default_max_skew_ms(31)
    _ck(checks, "31 fps 起才开始收紧（< 60）", at31 < 60.0, f"{at31:.4f} ms")
    _ck(checks, "31 fps == 60*30/31", abs(at31 - 60.0 * 30.0 / 31.0) < 1e-9,
        f"{at31!r}")
    _ck(checks, "DEFAULT_MAX_SKEW_MS 常量本身仍是 60.0",
        abs(rrs.DEFAULT_MAX_SKEW_MS - 60.0) < 1e-9, repr(rrs.DEFAULT_MAX_SKEW_MS))

    # `None` 在 should_drop() 里是"不设门限"，绝不能漏进主循环 ——
    # 所以 parse_args() 之后必须是**数字**。
    a_default = rrs.parse_args([])
    _ck(checks, "parse_args([]) 的 max_skew_ms 已是数字 60（None 不许漏进去）",
        a_default.max_skew_ms is not None
        and abs(a_default.max_skew_ms - 60.0) < 1e-9, repr(a_default.max_skew_ms))
    a500 = rrs.parse_args(["--fps", "500"])
    _ck(checks, "--fps 500 -> max_skew_ms 派生为 3.6",
        abs(a500.max_skew_ms - 3.6) < 1e-9, repr(a500.max_skew_ms))
    a_set = rrs.parse_args(["--fps", "500", "--max-skew-ms", "10"])
    _ck(checks, "显式 --max-skew-ms 10 覆盖派生值",
        abs(a_set.max_skew_ms - 10.0) < 1e-9, repr(a_set.max_skew_ms))

    return all(ok for _n, ok in checks)


# ---------------------------------------------------------------------------
# ⑥ 闪光同步搜索窗
# ---------------------------------------------------------------------------
def test_sync_window_derivation() -> bool:
    """24/30 fps 下必须**等于旧硬编码的 180/30** —— 否则就是改了现有相机行为。"""
    print("----- ⑥ 闪光同步搜索窗：24/30 fps 与旧常量 180/30 一致 -----")
    checks: list[tuple[str, bool]] = []

    from stereo.sync import SyncConfig, config_for_fps

    c24 = config_for_fps(24)
    _ck(checks, "24 fps -> max_offset=180 / min_overlap=30（旧硬编码）",
        c24.max_offset == 180 and c24.min_overlap == 30,
        f"{c24.max_offset}/{c24.min_overlap}")
    c30 = config_for_fps(30)
    _ck(checks, "30 fps -> max_offset=180 / min_overlap=30（旧硬编码）",
        c30.max_offset == 180 and c30.min_overlap == 30,
        f"{c30.max_offset}/{c30.min_overlap}")
    c500 = config_for_fps(500)
    _ck(checks, "500 fps -> 6 秒/1 秒 = 3000/500",
        c500.max_offset == 3000 and c500.min_overlap == 500,
        f"{c500.max_offset}/{c500.min_overlap}")
    c1000 = config_for_fps(1000)
    _ck(checks, "1000 fps -> max_offset=6000", c1000.max_offset == 6000,
        f"{c1000.max_offset}")

    over = config_for_fps(30, smooth=3)
    _ck(checks, "overrides 能覆盖 smooth，且不动搜索窗（30 fps 仍 180/30）",
        over.smooth == 3 and over.max_offset == 180 and over.min_overlap == 30,
        f"smooth={over.smooth} {over.max_offset}/{over.min_overlap}")

    # 类默认值本身也是 180/30：直接构造 SyncConfig() 的老代码不受影响。
    d = SyncConfig()
    _ck(checks, "SyncConfig() 的默认仍是 180/30（直接构造的老调用点不变）",
        d.max_offset == 180 and d.min_overlap == 30,
        f"{d.max_offset}/{d.min_overlap}")

    return all(ok for _n, ok in checks)


# ---------------------------------------------------------------------------
# ⑦ 带宽估算
# ---------------------------------------------------------------------------
def test_stream_bandwidth() -> bool:
    """告警按**单路** 0.5 GB/s 判 —— 两台相机各占一个 USB3 口、各吃自己的链路。

    按两路合计判会把本机**已实测跑通**的 2560x1440@24（单路 0.265）也报成告警，
    那就变成每天一条假警报，最后没人看它。
    """
    print("----- ⑦ 录制启动期带宽估算（BGR24 裸流） -----")
    checks: list[tuple[str, bool]] = []

    from apps.record_stereo import USB3_EFFECTIVE_GBPS, stream_bandwidth

    _ck(checks, "USB3_EFFECTIVE_GBPS == 0.5（单路有效带宽）",
        abs(USB3_EFFECTIVE_GBPS - 0.5) < 1e-12, repr(USB3_EFFECTIVE_GBPS))

    cases = (
        (2560, 1440, 24, 0.265, False),
        (1920, 1080, 60, 0.373, False),
        (3840, 2160, 24, 0.597, True),
        (7680, 4320, 24, 2.389, True),
        (5120, 2880, 30, 1.327, True),
    )
    for w, h, fps, per, alarmed in cases:
        frame_mb, per_cam, total = stream_bandwidth(w, h, fps)
        expect_per = w * h * fps * 3 / 1e9
        _ck(checks, f"{w}x{h}@{fps}: 单路 {round(per_cam, 3)} GB/s、"
                    f"告警={alarmed}",
            round(per_cam, 3) == per
            and (per_cam > USB3_EFFECTIVE_GBPS) is alarmed,
            f"实测 {per_cam:.6f}（公式 {expect_per:.6f}）")
        _ck(checks, f"{w}x{h}@{fps}: 单帧 {frame_mb:.3f} MB、合计 == 单路 × 2",
            abs(total - per_cam * 2.0) < 1e-12
            and abs(frame_mb - w * h * 3 / 1e6) < 1e-12,
            f"{frame_mb:.4f} MB / {total:.6f} GB/s")

    return all(ok for _n, ok in checks)


# ---------------------------------------------------------------------------
# ⑧ 容差不许被按帧率派生（曾经改错过）
# ---------------------------------------------------------------------------
def test_timestamp_tolerance_fixed() -> bool:
    """``timestamp_tolerance_s`` 只用于**视频文件**的时间戳 sidecar 配对。

    帧周期来自**容器**；构造函数的 ``fps=`` 是**相机请求帧率**（视频路径下默认 60，
    与视频真实帧率毫无关系）。按 ``1/fps`` 派生会得到 0.0167 —— 比原值紧一倍，
    是对 ``--left-video/--right-video`` 老入口的行为变更，且对高帧率相机零收益
    （相机路径根本不走这条）。这一条上一包专门试过又撤回，所以要钉死。
    """
    print("----- ⑧ 视频时间戳容差固定 0.033（不按帧率派生） -----")
    checks: list[tuple[str, bool]] = []

    import stereo.pair_source as ps
    from stereo import PairSource

    default = inspect.signature(
        PairSource.__init__).parameters["timestamp_tolerance_s"].default
    _ck(checks, "PairSource.__init__ 的 timestamp_tolerance_s 默认 0.033",
        default == 0.033, repr(default))

    whole = inspect.getsource(ps)
    _ck(checks, "模块里不存在 _resolve_timestamp_tolerance 这类派生实现",
        "_resolve_timestamp_tolerance" not in whole)

    src = _src("stereo/pair_source.py")
    _ck(checks, 'from_args 视频分支的兜底是 0.033（与类默认一致）',
        'getattr(args, "timestamp_tolerance", 0.033)' in src)

    code = _code_only(src)
    hits = [f"第 {i} 行：{line.strip()}"
            for i, line in enumerate(code.splitlines(), 1) if "0.020" in line]
    _ck(checks, "0.020（旧 fallback）只出现在注释里，代码里没有", not hits,
        "; ".join(hits) if hits else "注释已剥除，代码 0 命中")

    return all(ok for _n, ok in checks)


# ---------------------------------------------------------------------------
# ⑨ 帧率类型注解不再说谎
# ---------------------------------------------------------------------------
def test_fps_annotations() -> bool:
    """帧率全链是浮点：注解写 ``fps: int`` 而实参是 float，会把后来人带沟里。"""
    print("----- ⑨ fps 类型注解与实现一致（不再是 int） -----")
    checks: list[tuple[str, bool]] = []

    pattern = re.compile(r"fps\s*:\s*int\b")
    for rel in ("stereo/pair_source.py", "sphere_tracker/sources.py",
                "apps/_common.py"):
        code = _code_only(_src(rel))
        hits = [f"第 {i} 行：{line.strip()}"
                for i, line in enumerate(code.splitlines(), 1)
                if pattern.search(line)]
        _ck(checks, f"{rel} 里没有 fps: int 注解", not hits,
            "; ".join(hits) if hits else "0 命中")

    pair_code = _code_only(_src("stereo/pair_source.py"))
    _ck(checks, "pair_source 的帧率是 float(fps)（不是 int(fps) 截断）",
        "self._fps = float(fps)" in pair_code)

    return all(ok for _n, ok in checks)


# ---------------------------------------------------------------------------
def main() -> int:
    print("=" * 70)
    print("高帧率/高分辨率扩容自测（8K/5K 档不进自动协商、帧率浮点、门限按 fps 派生）")
    print("=" * 70)
    results = []
    for fn in (test_probe_ladder_unchanged,
               test_default_is_object_reference,
               test_fps_type_keeps_fraction,
               test_cli_fps_accepts_fraction,
               test_max_skew_derivation,
               test_sync_window_derivation,
               test_stream_bandwidth,
               test_timestamp_tolerance_fixed,
               test_fps_annotations):
        results.append(fn())
        print()
    ok = all(results)
    print("判定结果：" + ("全部通过" if ok else "有不达标项"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
