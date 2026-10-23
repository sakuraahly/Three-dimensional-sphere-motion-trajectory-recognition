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
from sphere_tracker import video_profiles  # noqa: E402
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


def _open_preview(index: int, backend: int):
    """按**给定的那一个后端**打开预览；返回 (src, 后端名)，打不开返回 (None, None)。

    ## 为什么**绝不**在这里换后端（2026-10-18 第三次修正，用户实测反馈）

    教训链（两次都是我改坏的）：

    1. f8bb30 加了"按索引换后端" → 用户看到 **aoni 被标成 Integrated**
       （DSHOW 的编号与 PnP 错位）；
    2. 33b7164 把回退整个删掉 → MSMF 打不开的那台 **从界面消失**
       （"只能检测到一个"）；
    3. 我又改成"回退只为显示、不许绑定" → 用户看到 **同一台 aoni left 出现两次**
       而 right 不见 —— 因为回退让 DSHOW #2 与 MSMF #0 **可能是同一台**。

    **结论：按索引回退在"显示"这一层也站不住** —— 它会**重复**同一台物理相机。
    PnP 与各后端的编号是**互不相容的三套命名空间**，任何"混着用"的方案都会
    要么贴错标签、要么重复、要么遗漏。

    所以这里只做一件事：**在同一个后端内**开一个编号。
    想确认"另一台在别的后端下能不能出图"，用 cameras.py probe（它逐后端
    分别列出，本来就**不下结论说"哪台是哪台"**），或显式 --backend dshow。
    """
    from sphere_tracker.sources import backend_name

    used_name = backend_name(backend)
    try:
        src = _try_open(index, backend)
    except RuntimeError as exc:
        print(f"  #{index}（{used_name}）打不开：{_first_line(exc)}")
        return None, None
    _backend_used[index] = used_name
    return src, used_name


def _try_open(index: int, api: int):
    """按给定后端开一路预览；失败**抛 RuntimeError**，由调用方决定是否回退。

    optimize=False + 短预算（2026-10-18，用户反馈"卡顿明显"）：
    设备识别只需要"把画面抓出来让人按键"，**不需要**挑最优分辨率/帧率。
    之前用默认参数时：optimize=True 会跑模式搜索（冷缓存最坏 10~40 s），
    search_budget_s 默认 60 s —— 遇到打不开的相机会干等一分钟。两者叠加
    就是用户看到的"卡顿"。
    """
    # ⚠️ **请求 1920x1080（原生）而不是 640x480**。
    #
    # 2026-10-18 实测这台 aoni：MSMF 下**只认 1920x1080** ——
    # 请求 640x480 / 1280x720 都是"打开成功但一帧都不给"（10.6 s 后判死），
    # 而 1920x1080 能出帧（约 23 s；另一台约 2 s）。
    # 旧代码请求 640x480，于是那台在识别界面里**彻底消失**。
    return CameraSource(index=index, width=1920, height=1080, backend=api,
                        optimize=False, search_budget_s=PREVIEW_BUDGET_S,
                        first_frame_timeout_s=PREVIEW_FIRST_FRAME_S)


# 预览里"等第一帧"的上限（秒）。**慢相机实测要 ~24 秒**才给第一帧
#（aoni #0 + MSMF）；3 秒会把它误判成"一帧都拿不到"，于是它在界面上彻底消失。
# 预览窗口的**显示宽度**（超过就等比降采样）。识别只需看清是哪台，
# 而 1920x1080 每帧约 6 MB，直接 imshow 在核显上会明显掉帧。
PREVIEW_DISPLAY_W = 960

PREVIEW_FIRST_FRAME_S = 30.0

# 单台预览的**总**预算：要能容纳上面那次首帧等待 + 一次分辨率降级。
PREVIEW_BUDGET_S = 40.0


def _preview_label(index: int, entry: cr.DeviceEntry) -> str:
    """预览窗口上的标签。**必须写明"推测"** —— 这不是事实，是我们猜的。

    ## 为什么必须标"推测"（2026-10-18，用户第二次纠正）

    用户问："你的相机管理的视频画面的内容是不是根本不是通过相机的名字来设置的?
    为什么我的命名为 right 的相机却在画面上是 left?"

    **他说得对，而且这条比上一轮更根本。**

    `cv2.VideoCapture(N, backend)` **只能按编号打开** —— 名字根本不参与。
    窗口上那行 `#N <名字>` 是"编号 N 对应设备列表里第 N 台"这个**假设**的产物。

    我的证据缺口在这里：

    | 我验证过的 | 结论 |
    | --- | --- |
    | aoni vs Integrated（MSMF `#2` 黑 = PnP `#2` 内置） | 成立 |
    | **两台一模一样的 aoni 谁在 `#0`、谁在 `#1`** | **从未验证** |

    拿"`#2` 是内置"外推到"整个顺序都对"，是**用一条弱证据支撑了一个强结论**。
    两台 aoni 同 VID/PID、同分辨率、同画面特征 —— **在软件层面不可区分**，
    所以这个顺序**原理上无法由 OpenCV 证明**。

    于是：这条标签**永远只能是推测**。把推测写成事实，用户就会按错的标签绑定角色
    （他说"命名为 right 的相机在画面上是 left"就是这么来的）。

    正确做法：**标签照实说它是推测**，绑定走 `cameras.py pin <role> name:"<完整名字>"`
    —— 那条路径按**用户赋予的名字**解析到身份，**完全不开相机、也不碰编号**。
    """
    return f"#{index}  推测:{entry.name or '(无名字)'}  端口 {entry.port or '-'}"


def _preview_and_assign(index: int, entry: cr.DeviceEntry, status: str = "",
                        *, backend: int) -> str | None:
    """开一台相机显示实时画面，返回用户按下的角色（None=跳过，''=退出）。

    ⚠️ `_open_preview` **必须**返回 `(src, 后端名)` 两元组 —— 两个分支形状一致。
    这里崩过一次（2026-10-15）：早退分支写成 `return None`，解包时抛
    `TypeError: cannot unpack non-iterable NoneType object`，整个 identify 崩掉。
    判据放在 `tests/selftest_camera_registry.py` 第 14 节（**穷举返回路径**验形状）。

    ⚠️ `backend` 是**整个 identify 会话共用的**，不允许按索引换（见 `_open_preview`
    里的长注释与那张实测表）。`keyword-only` 是为了让每个调用点都必须显式交代它。
    """
    label = _preview_label(index, entry)
    try:
        src, backend_used = _open_preview(index, backend)
    except RuntimeError as exc:
        print(f"  #{index} 打不开：{_first_line(exc)}")
        return None
    if src is None:
        return None                     # 这个后端出不了图 -> 跳过这台（**不崩、也不换后端**）
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
                    # ⚠️ **显示前先降采样**（2026-10-18，用户反馈"画面依然很卡顿"）。
                    #
                    # 这台相机只肯给 1920x1080；而识别只需要"能看出这是哪台"。
                    # 每帧直接 imshow 1080p（约 6 MB/帧）在中低端核显上会明显掉帧，
                    # 再加上 putText 也在全尺寸上做，就更慢。
                    # 降到 960 宽后，imshow 的拷贝量降到 1/4。
                    disp = frame
                    if frame.shape[1] > PREVIEW_DISPLAY_W:
                        ratio = PREVIEW_DISPLAY_W / float(frame.shape[1])
                        disp = cv2.resize(
                            frame,
                            (PREVIEW_DISPLAY_W, max(1, int(frame.shape[0] * ratio))),
                            interpolation=cv2.INTER_AREA)
                    cv2.putText(disp, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 255), 2, cv2.LINE_AA)
                    if status:
                        cv2.putText(disp, f"now: {status}", (8, 54),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                                    cv2.LINE_AA)
                    try:
                        wins.show(win, disp)
                    except cv2.error:
                        gui_ok = False
                        break
                else:
                    miss += 1
                    if miss == 30:                    # 约 1 秒
                        print(f"  ⚠ #{index} 读不到画面了…（镜头被挡 / 被别的程序抢走 / "
                              f"USB 掉线？）再等 1 秒，还不行就跳过这台")
                    if miss >= 60:                    # 约 2 秒
                        print(f"  ✘ #{index} 连续 2 秒读不到画面，跳过这台相机。")
                        # ⚠️ 排查顺序**先问"这台是哪台"**（2026-10-15 加，坑 37）：
                        # 本机的**笔记本内置相机被用户有意遮住**，它必然读不到画面 ——
                        # 那是**预期行为**。原来的提示只写"是不是别的程序占着"，
                        # 会把内置相机这种正常情况也引向"去查占用/拔插"。
                        print(f"     排查（按顺序）：① 这台是不是**笔记本内置相机**"
                              f"（名字含 `Integrated`）？本机它是**有意被遮住**的，"
                              f"读不到画面属正常 → 按 `n` 标「无关」即可；")
                        print(f"     ② 若是外接那台：镜头是否被挡/盖 → 别的程序"
                              f"（QQ/会议/浏览器）是否占着 → 拔插 USB。")
                        return None
                key = cv2.waitKey(1) & 0xFF
                if wins.handle_key(win, key, disp):
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



def cmd_probe(args) -> int:
    """回答一个**不需要知道"哪台是哪台"**的问题：**现在有几台相机真的能出图？**

    ## 为什么必须有这个命令（2026-10-18 用户的真实困惑）

    用户反馈：「刷新设备列表和检查映射不能保证有用」+「aoni 摄像头又是只能检测到一个」。

    前半句是**对的**，因为那两个功能都**只读配置/枚举，根本不打开相机**：

    | 入口 | 实际做了什么 | 能不能回答"相机活着吗" |
    | ---- | ------------ | ----------------------- |
    | 刷新设备列表 | list_devices()（读注册表/PnP） | ❌ 只能列出**存在**的设备 |
    | 检查相机映射 | cameras.py check（比对配置 vs 枚举） | ❌ 只看配置有没有漂移 |

    而后半句的机制是本机 **MSMF 间歇性变慢/打不开**（实测 MSMF #0 一次要 20 s 以上，
    另一次直接超时），于是"单个后端试着开"的界面就只剩一台。

    ## 这个命令的口径（**只说它敢保证的事**）

    对**每个后端**、每个编号，真的开一次、读几帧，报「有图 / 纯黑 / 打不开 / 读不到帧」。
    于是"两台 aoni 是不是都活着"有了确定答案。

    ⚠️ **它不能用来绑角色**：两个后端的编号**不是同一台设备**
    （实测 DSHOW #1 = 笔记本内置、#2 才是外接）。本命令只回答"有几台能出图"，
    绑角色请用 pin <role> name:"<完整名字>" —— 那条路不开相机、不碰编号。
    """
    from sphere_tracker.sources import backend_name

    import cv2

    devices = cr.list_devices()
    names = {d.order: d.name for d in devices}
    max_index = args.max_index
    which = args.backends
    backends = []
    if which in ("both", "msmf"):
        backends.append(cv2.CAP_MSMF)
    if which in ("both", "dshow"):
        backends.append(cv2.CAP_DSHOW)

    print("=" * 68)
    print("相机能不能出图（真的开一次，逐后端 × 逐编号）")
    print("=" * 68)
    if devices:
        print(f"系统里存在 {len(devices)} 台：")
        for d in devices:
            print(f"    #{d.order}  {d.name or '(无名字)'}")
    else:
        print("（枚举不到设备 —— 只能按编号盲探）")
    print()
    print("⚠️ **不同后端的编号不是同一台设备**，所以下面只回答"
          "「这个后端下这个编号能不能出图」，")
    print("   **不能**用来决定哪台是 left / 哪台是 right（绑角色请用 pin name:）。")
    print()

    # "能打开"不等于"有可用画面"：**每帧几乎恒定的纯色**（std 很小）说明镜头被挡/
    # 被遮住/没曝光。本机那台内置相机就是被有意遮住的 —— 若把它算进"能出图"，
    # 用户会得到"两台都在"的错误结论，而这正是他这次困惑的来源之一。
    DARK_STD = 5.0

    counts: dict[str, int] = {}
    working: dict[str, set[int]] = {}          # 每个后端下"真正出图"的编号
    for api in backends:
        nm = backend_name(api)
        ok_n = 0
        dark_n = 0
        working[nm] = set()
        cells = []
        for idx in range(max_index + 1):
            r = cr.probe_index(idx, width=640, height=480, fps=30.0, backend=api)
            if r.ok and r.std >= DARK_STD:
                ok_n += 1
                working[nm].add(idx)
                cells.append(f"#{idx} ✅ 有图 {r.width}x{r.height} std{r.std:.0f}")
            elif r.ok:
                dark_n += 1
                cells.append(f"#{idx} ⚠ 近乎纯黑（std{r.std:.1f}，std<{DARK_STD:.0f}）"
                             f" —— 可能是镜头被挡/被遮住")
            else:
                first = (r.error or "").splitlines()[0][:26]
                cells.append(f"#{idx} ✘ {first}")
        counts[nm] = ok_n
        print(f"  {nm:5} **真正出图 {ok_n} 台**" +
              (f"（另有 {dark_n} 台能打开但画面近乎纯黑）" if dark_n else ""))
        for c in cells:
            print(f"        {c}")
        print()

    best = max(counts.values()) if counts else 0
    print("-" * 68)
    print(f"结论：**最多 {best} 台**能真正出图"
          f"（{'、'.join(f'{k} {v} 台' for k, v in counts.items())}）。")
    if best >= 2:
        print("  → 有一个后端能同时开出两台，可用来做双目。"
              "标定时请**显式指定**该后端（--backend dshow|msmf）。")
    elif len(working) >= 2 and sum(len(v) for v in working.values()) >= 2 \
            and not (set.intersection(*working.values()) if working else set()):
        # 每个后端各只有一台、且**编号互不相同** —— 本机实测就是这种状态
        print("  → ⚠️ **两台各自只在不同的后端下工作，没有任何后端能同时开出两台。**")
        for nm, idxs in working.items():
            print(f"        {nm}：编号 {sorted(idxs)} 能出图")
        print("     这不是识别逻辑的问题，是**后端/驱动层面的状态**。"
              "双目要求同一个后端同时开两路，所以现在**标定跑不起来**。")
        print("     按顺序试：① 两台都拔掉、等 5 秒、插回（重置驱动状态）；"
              "② 换到机身不同的 USB 口；")
        print("     ③ 关掉可能占用相机的程序；④ 再跑一次本命令，"
              "看是否变成「两台都在同一个后端」。")
    else:
        print("  → 只有一台能出图。按顺序查：① 是否有一台**镜头被挡/被遮住**"
              "（近乎纯黑就是它）；")
        print("     ② 拔插换 USB 口；③ 是否被别的程序（会议/浏览器）占着。")
    return 0 if best >= 1 else 1


def _pin_example_lines(devices: list[cr.DeviceEntry]) -> list[str]:
    """按**当前实际设备**生成 `pin` 示例命令（不写死任何型号名）。

    ## 为什么改成动态生成（2026-10-18）

    原来这两行是写死的：

        python apps/tools/cameras.py pin left  name:"aoni webcam A25 left"
        python apps/tools/cameras.py pin right name:"aoni webcam A25 right"

    换一台相机、或用户改了名，这段提示就变成**错的/看不懂的** ——
    而它正是"名字不可靠时该怎么办"的关键指引，错不起。
    现在从 `list_devices()` 取**真实存在**、且未被标「无关」/未被判为内置的设备。
    """
    reg = cr.load_registry()
    usable = [d for d in devices
              if d.name
              and not cr.is_device_ignored(d, reg)
              and not cr.is_avoided(d.name, reg.settings)]
    if len(usable) < 2:
        return ["     python apps/tools/cameras.py list   # 先看有哪些设备名可用"]
    a, b = usable[0], usable[1]
    return [
        f'     python apps/tools/cameras.py pin left  name:"{a.name}"',
        f'     python apps/tools/cameras.py pin right name:"{b.name}"',
    ]


def cmd_identify(args) -> int:
    devices = cr.list_devices()
    by_order = {d.order: d for d in devices}
    indices = parse_indices(args.indices, devices, args.max_index)
    if not indices:
        print("没有可识别的索引。先跑 python apps/tools/cameras.py list 看看设备。")
        return 1

    # ---- 选定**本次会话唯一**的后端（绝不在索引之间切换）-----------------
    #
    # 为什么必须在会话级定死：`#N` 的标签来自 PnP 枚举，而 cv2 用的是**后端自己的
    # 枚举**。只有 MSMF 实测与 PnP 位次一致；DSHOW 整体错位（#1=内置、#2=aoni）。
    # 所以换后端 = 标签与画面脱钩 = 用户把角色绑到错的相机上（2026-10-18 的真 bug）。
    from sphere_tracker.sources import INDEX_BACKEND, INDEX_BACKEND_NAME, backend_name

    backend = INDEX_BACKEND if not getattr(args, "backend", None) else {
        "msmf": __import__("cv2").CAP_MSMF,
        "dshow": __import__("cv2").CAP_DSHOW,
    }[args.backend]
    used = backend_name(backend)
    print(f"后端：**{used}**（整场只用这一个，不会中途换）")
    if backend != INDEX_BACKEND:
        print("=" * 70)
        print(f"⚠️ 你指定了 {used}。**它的枚举顺序与本机设备列表不一致**：")
        print("   实测本机 DSHOW 的编号整体错位（`#1` 是笔记本内置、"
              "`#2` 才是外接那台），")
        print(f"   而 {INDEX_BACKEND_NAME} 与设备列表一致。")
        print(f"   用 {used} 识别，**可能把画面上的相机贴成另一台的名字**。")
        print("   除非你已经确认过这个后端的编号，否则建议：按 Ctrl+C 退出，改用")
        for line in _pin_example_lines(devices):
            print(line)
        print("   —— 按名字直接绑，**根本不用开相机**，也就没有错位问题。")
        print("=" * 70)

    reg = cr.load_registry()
    # 🔴 这段警告是 2026-10-18 用户第二次纠正后加的，**不要删**。
    #
    # 用户问："视频画面的内容是不是根本不是通过相机的名字来设置的?
    #          为什么我的命名为 right 的相机却在画面上是 left?"
    #
    # 答案：**是，画面只按编号打开，名字根本不参与。**
    # 窗口上 `#N 推测:<名字>` 里的名字，是"编号 N ↔ 设备列表第 N 台"这个**假设**的产物。
    # 而这条假设 **只在"两台 aoni 谁在 #0/#1"这一层上从未被验证过** ——
    # 两台同 VID/PID、同分辨率、同画面特征，**软件层面不可区分**。
    # 所以标签可能是反的，**必须以你眼睛看到的画面为准**；
    # 要"名字说了算"，请用 `pin name:`（那条路径不开相机、不碰编号）。
    print("⚠️ 下面每台窗口的标签写着「推测:<名字>」——**名字是猜的，不是事实。**")
    print("   `cv2.VideoCapture` 只能按**编号**打开，名字不参与；"
          "而编号与设备列表的对应关系")
    print("   在两台同型号相机之间**无法由程序证明**（它们同 VID/PID、同分辨率）。")
    print("   所以：**请以画面为准**按键。若你已用名字区分了左右，更可靠的做法是")
    for line in _pin_example_lines(devices):
        print(line)
    print("   —— 按名字绑身份，**不开相机、不碰编号**，也就不存在左右颠倒。")
    print("\n逐个索引显示画面，请按提示确认哪台是左、哪台是右、哪台「无关」：")
    print(f"（按键：{_KEY_HINT}）\n")

    _unopened: list[int] = []
    for idx in indices:
        entry = by_order.get(idx) or cr.DeviceEntry(order=idx)
        status = _describe_status(reg, idx, entry)
        role = _preview_and_assign(idx, entry, status, backend=backend)
        if role is None and idx not in _unopened:
            _unopened.append(idx)
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
    if _unopened:
        print()
        print(f"以下编号在 {used} 下**没能打开**（不是没插，是这个后端打不开）：{_unopened}")
        print("  想确认它们在**别的后端**下是否可用：")
        print(f"     python apps/tools/cameras.py probe --max-index {max(_unopened)}")
        print("  （probe 逐后端分别列出 —— **不同后端的编号不是同一台**，它也不下结论说哪台是哪台）")
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
            # ⚠️ **解析失败必须算危险项**（2026-10-15 修）。
            # 这里原来只 `continue`，**没有置 danger** —— 于是"角色解析失败"
            # 却返回退出码 0，而本文档与用户手册都承诺「有 [危险] 项时退出码为 2」。
            # 用户 2026-10-15 换 USB 口后跑 check 就是这样：报了一行失败、退出码 0，
            # 看起来像"没事"。
            danger = True
            print(f"[危险] 「{role}」({tag}) **解析失败**：{_first_line(exc)}")
            # 光说"不在线"没用 —— 换 USB 口也会这样，而那是本工具最容易让人困惑的一步。
            # 补一句"当前有没有同名的设备"（**只报告，不自动改绑**）。
            hint = cr.describe_moved_device(spec, devices, role)
            if hint:
                for line in hint.splitlines():
                    print(line)
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
# doctor：一次把"能不能跑双目"全查完
# --------------------------------------------------------------------------- #
# 为什么要有它（2026-08 ~ 10 的教训都指向同一件事）：
#
# 这台机器上"相机能不能用"是**浮动**的，而且失败时会以**五六种不同的面孔**出现：
# 打不开 / 能打开但无帧 / 出帧但恒为纯黑 / 只有某个后端行 / 两路分辨率不一致 /
# 配置指向已换口的旧身份。这些我这几轮**一条一条手跑命令去查**，
# 每次都写一段临时 Python —— 按项目规矩（重复 3 遍就固化）该收进工具里。
#
# 与 `check` 的分工：
#   `check`  = 配置 vs 设备（**不开相机**，秒级）—— 查"配置还对不对"
#   `doctor` = **真开相机**，查"现在到底能不能出图、能不能凑成一对"
# 两者都**只报告不改配置**。
def cmd_doctor(args) -> int:
    """打开每台相机实测一遍，回答"现在能不能跑双目"。"""
    import tempfile

    from sphere_tracker import camera_registry as cr
    from sphere_tracker.sources import backend_name

    reg = cr.load_registry()
    devices = cr.list_devices()
    want_w, want_h = (args.width, args.height)

    print("=" * 68)
    print("相机体检（真开相机；只报告，不改配置）")
    print("=" * 68)
    if not devices:
        print("枚举不到任何相机设备（非 Windows / 注册表不可读 / 真的没插）。")
        return 1

    # ---- ① 枚举 ----
    print(f"\n① 枚举到 {len(devices)} 台：")
    for d in devices:
        mark = "  [无关]" if cr.is_device_ignored(d, reg) else ""
        print(f"    #{d.order}  {d.name or '(无名字)':22} 端口 {d.port or '-'}{mark}")

    # ---- ② 角色解析（不开相机）----
    print("\n② 角色解析：")
    resolved: dict[str, int | None] = {}
    fail = False
    for role in ("left", "right"):
        spec = reg.roles.get(role)
        if spec is None:
            print(f"    「{role}」：配置里没有这个角色 → 先跑 identify")
            resolved[role] = None
            fail = True
            continue
        try:
            r = cr.resolve_role(role, reg, devices, interactive=False)
            resolved[role] = r.index
            print(f"    「{role}」-> #{r.index}（how={r.how}）")
        except RuntimeError as exc:
            resolved[role] = None
            fail = True
            print(f"    「{role}」-> **解析失败**：{_first_line(exc)}")
            for line in cr.describe_moved_device(spec, devices, role).splitlines():
                print(line)

    # ---- ③ 每台实测：**在唯一可信的后端上**真能出图吗 ----
    #
    # ⚠️ 2026-10-18 修：原来用 `pick_backend_for_device()` 给每个索引单独挑后端 ——
    # 那是错的。`resolved[role]` 是 **PnP 位次**，只有 MSMF 与它一致；
    # 换成 DSHOW 后同一个编号指向**另一台物理相机**（实测 #1=内置、#2=aoni），
    # 于是"实测这一路"其实测的是别的相机，结论整个失真。
    # 现在：**整场只用一个后端**，与 `identify` 同一条纪律。
    from sphere_tracker.sources import INDEX_BACKEND, INDEX_BACKEND_NAME

    probe_backend = INDEX_BACKEND
    print(f"\n③ 逐台实测（后端固定为 **{INDEX_BACKEND_NAME}** —— 唯一与设备位次一致的后端；"
          f"判据是「真能出图」）：")
    usable: dict[str, tuple[int, int, int]] = {}      # role -> (index, api, std)
    for role in ("left", "right"):
        idx = resolved.get(role)
        if idx is None:
            continue
        probe = cr.probe_index(idx, width=want_w, height=want_h,
                               fps=30.0, backend=probe_backend)
        if not probe.ok:
            fail = True
            print(f"    「{role}」#{idx}：**出不了图** —— {probe.error}")
            print(f"       （不换后端试：换后端后 #{idx} 会指到**另一台**相机，"
                  f"测出来的不是这一台）")
            continue
        usable[role] = (idx, probe_backend, int(probe.std))
        print(f"    「{role}」#{idx}：{probe.width}x{probe.height}，"
              f"std {probe.std:.1f} ✅")

    # ---- ④ 两路能不能凑成一对（同后端 + 同分辨率）----
    print(f"\n④ 两路一致性（双目硬要求：**同一个后端 + 同一个分辨率**）：")
    if len(usable) < 2:
        fail = True
        print("    ✘ 不足两路可用，没法组双目。")
    else:
        (li, la, _), (ri, ra, _) = usable["left"], usable["right"]
        if la != ra:
            fail = True
            print(f"    ✘ 两路**后端不同**（left={backend_name(la)} / "
                  f"right={backend_name(ra)}）—— 同一进程混用会挂一路。")
        else:
            print(f"    ✓ 两路同后端（{backend_name(la)}）")
        # 用**临时缓存**做一致性测试，绝不污染真缓存（"缓存把分辨率钉死"
        # 正是踩过的坑：`config/camera_modes.json` 里记着 640x480，
        # 于是显式请求 1920x1080 也被盖过）。
        import sphere_tracker.sources as _src

        saved_cache = _src._MODE_CACHE
        res: dict[str, tuple[int, int] | None] = {}
        try:
            with tempfile.TemporaryDirectory() as td:
                _src._MODE_CACHE = Path(td) / "probe_modes.json"
                for role in ("left", "right"):
                    # ⚠️ 这里也**只测同一个后端**（同一条纪律），不按索引换后端。
                    probe = cr.probe_index(usable[role][0], width=want_w,
                                           height=want_h, fps=30.0,
                                           backend=usable[role][1])
                    res[role] = (probe.width, probe.height) if probe.ok else None
        finally:
            _src._MODE_CACHE = saved_cache
        print(f"    实测分辨率：left={res.get('left')}  right={res.get('right')}")
        if res.get("left") and res.get("left") == res.get("right"):
            print(f"    ✓ 两路一致 {res['left']} —— 满足标定要求")
        else:
            fail = True
            print(f"    ✘ 两路**不一致** —— 标定会直接失败。"
                  f"本机实测：只有请求 **1920x1080** 时两路才可能一致"
                  f"（`#0` 那台无视请求恒给 1920x1080）。")

    # ---- ⑤ 结论 ----
    print("\n" + "=" * 68)
    if fail:
        print("结论：**还不能跑双目。** 按上面 ✘ 那几行的指引处理；")
        print("     换过 USB 口先跑 `identify` 重新确认（InstanceId 里编着端口号）。")
        return 2
    print("结论：**可以跑双目。** 下一步：")
    print(f"    启动程序.bat precheck      （举棋盘确认两路都看得到）")
    print(f"    启动程序.bat calibrate     （标定；基线核对记得填你量到的中心距）")
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
    pi.add_argument(
        "--backend", choices=("msmf", "dshow"), default=None,
        help="整场用哪个后端。**默认 msmf** —— 只有它与系统设备列表的顺序一致；"
             "dshow 会整体错位，用它识别可能把标签贴到别的相机上")
    pi.set_defaults(func=cmd_identify)

    pp = sub.add_parser("pin", help="非交互地把某角色固定到某设备")
    pp.add_argument("role", help="角色名，如 left / right / main")
    pp.add_argument("value", help="0 | port:Port_#0001.Hub_#0003 | id:USB\\... | name:<名字片段>")
    pp.set_defaults(func=cmd_pin)

    pig = sub.add_parser("ignore", help="把某台相机标为「无关」（自动挑相机时跳过）")
    pig.add_argument("value", help="0 | port:Port_#0001.Hub_#0003 | id:USB\\... | name:<名字片段>")
    pig.set_defaults(func=cmd_ignore)

    pu = sub.add_parser("unignore", help="取消「无关」标记（单台，或 --all 清空整份清单）")
    pu.add_argument("value", nargs="?", default=None, help="同 ignore 的取值；与 --all 二选一")
    pu.add_argument("--all", action="store_true", help="清空整份「无关」清单")
    pu.set_defaults(func=cmd_unignore)

    pg = sub.add_parser("ignored", help="列出当前「无关」清单（并标注在不在线）")
    pg.set_defaults(func=cmd_ignored)

    pc = sub.add_parser("check", help="比对配置与当前设备：有没有漂移/被占用")
    pc.set_defaults(func=cmd_check)

    ppr = sub.add_parser(
        "probe", help="真的开一次：逐后端 × 逐编号报「能不能出图」（只诊断，不绑角色）")
    ppr.add_argument("--max-index", type=int, default=3, help="探测到第几个编号（默认 3）")
    ppr.add_argument("--backends", choices=("both", "msmf", "dshow"), default="both",
                     help="探测哪些后端（默认 both）")
    ppr.set_defaults(func=cmd_probe)

    pd = sub.add_parser(
        "doctor", help="**真开相机**体检一遍：现在能不能跑双目（只报告，不改配置）")
    pd.add_argument("--width", type=int, default=video_profiles.DEFAULT.width,
                    help=f"目标宽度（默认档 {video_profiles.DEFAULT.label}）")
    pd.add_argument("--height", type=int, default=video_profiles.DEFAULT.height,
                    help=f"目标高度（默认档 {video_profiles.DEFAULT.label}）")
    pd.set_defaults(func=cmd_doctor)
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
