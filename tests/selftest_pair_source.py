"""成对帧源自测：同步偏移到底取的是哪两帧。

为什么必须有这个测试
--------------------
`sync_offset` 的正负号是最容易**静默搞错**的地方：搞反了程序照跑，不报任何错，
只是把"左相机 t 时刻的球"和"右相机 t±8 帧的球"配成一对，三角测量出来一个
看起来完全正常、实际系统性错掉的三维点。这种错不会自己暴露，只能靠测试。

做法：把每一帧的**帧号编进画面**（左下、右下两个色块分别存帧号的低字节和高字节），
于是"取到的是第几帧"可以直接从像素里读回来比对，不依赖任何相机或标定。

验证十件事：
  1. `sync_offset=0`：左第 i 帧 配 右第 i 帧；
  2. `sync_offset=+k`：左第 i 帧 配 右第 i+k 帧；
  3. `sync_offset=-k`：左第 i 帧 配 右第 i-k 帧；
  4. **与 `stereo.find_offset` 串起来**：把一条连续时间线切成两段错开的视频，
     让 find_offset 自己算出偏移，再拿它去取帧 —— 取出来的必须是同一物理时刻。
     第 4 条才是真正要保证的契约，前三条是它的分解；
  5. 参数校验与"短的那路读完了"能干净结束；
  6. `from_args` 的命令行参数映射（重构最易漏的是调用点，不是库）；
  7. **采集会话**（`session.json`）：写→读→校验；坏会话必须逐条报错。
  8. **并行打开的整体超时**（`_join_all`）：改前裸 `th.join()` 会永久挂住主线程；
     改后"永不返回"的线程能在 0.2s 内被判定为"仍存活"并返回（不依赖真相机）。
  9. **自适应成对协商的纯逻辑**：档位标签、成对帧率取"两路较小者"、
     刚打开测不到时先等首帧再测（A5）、帧率下限与 `DEFAULT_FPS` 的换算关系。
  10. **A3 真帧守卫**（驱动真实的 `_negotiate`）：两路**上报一致但真帧不一致**
      必须被**拒绝**（不能静默接受一对尺寸不一致的相机），任一路取不到真帧
      也必须被拒绝；对照"都一致"要被接受。

图片文件夹与视频两条路径都测（两者的取帧实现完全不同：
文件夹是随机访问，视频要按帧号定位）。

    python tests/selftest_pair_source.py
    python tests/selftest_pair_source.py --keep     # 保留合成素材，便于人工排查
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
while not (ROOT / "sphere_tracker").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from stereo import (
    CameraInfo,
    CalibrationInfo,
    PairSource,
    SyncConfig,
    SyncInfo,
    canary_safe,
    create_session,
    find_offset,
    load_session,
    plan_timestamp_pairs,
    save_session,
    sha256_file,
    validate_session,
)
from calib.params import StereoRig

W, H = 96, 64
N = 40             # 合成多少帧
# 帧号写成 16 位灰度再拆成两个字节，分别放进左下 / 右下两个色块。
#
# 为什么不用"整幅图涂灰、灰度 = 帧号 x 步长"这种更简单的写法：
# 8 位灰度最多表示 256 个值，一乘步长就只能表示十几帧（步长 20 时到第 12 帧
# 就撞上 255 上限），更靠后的帧全都变成同一个灰度 —— 测试会以为"取错帧了"，
# 其实是素材自己分不出来。两个字节分开存就精确到 40 帧以上，还能扛视频压缩。
MARK = 24          # 每个字节色块的边长（够大，压缩后取值仍然稳定）


# --------------------------------------------------------------------------
def make_frame(index: int, brighten: float = 0.0, bg: int = 60) -> np.ndarray:
    """造一帧，并把帧号编进画面里（左右两个色块分别存低字节 / 高字节）。

    `brighten`（模拟闪光）只改背景、不动那两个色块 ——
    否则"变亮"会被读成"帧号变了"，测试自己就把自己坑了。
    `bg` 是背景基准亮度：两台相机自动曝光不同，基准亮度本来就该不一样。
    """
    lo, hi = index & 0xFF, (index >> 8) & 0xFF
    f = np.full((H, W, 3), np.clip(bg + brighten, 0, 255).astype(np.uint8), np.uint8)
    f[H - MARK:, 0:MARK] = lo
    f[H - MARK:, W - MARK:W] = hi
    return f


def read_index(frame: np.ndarray) -> int:
    """从一帧里把帧号读回来。用中位数，避开压缩噪声。"""
    lo = int(round(float(np.median(frame[H - MARK:, 0:MARK]))))
    hi = int(round(float(np.median(frame[H - MARK:, W - MARK:W]))))
    return (hi << 8) | lo


def write_images(base: Path, name: str, n: int) -> Path:
    d = base / name
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        cv2.imwrite(str(d / f"{i:03d}.png"), make_frame(i))
    return d


def make_scratch_dir(keep: bool = False) -> Path:
    """造一个可写的临时目录放合成素材。

    优先用系统临时目录；**沙箱/受限环境里那里可能不可写**（实测
    Windows 下 `%TEMP%` 会 `PermissionError [WinError 5]`），
    所以退回项目内的 `outputs/`（该目录本来就不入库、可随时重建）。
    """
    try:
        d = Path(tempfile.mkdtemp(prefix="pair_source_"))
        (d / ".probe").write_text("ok", encoding="utf-8")   # 真写一下才算数
        (d / ".probe").unlink()
        return d
    except OSError:
        pass
    d = ROOT / "outputs" / ("_pair_source_keep" if keep else "_pair_source_tmp")
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_video(path: Path, n: int, bg: int = 60) -> Path:
    """写一段"内容帧 0..n-1、逐帧对齐"的视频（用于最基础的偏移语义测试）。

    **编解码器的选择是有讲究的**：`mp4v` / `XVID` 这类有帧间预测的编码器会把
    "帧号"这种每帧都变的内容当残差压掉，实测解出来前几帧全被预测成同一个值，
    于是测试会误报成"取错帧了"（真正的问题是素材自己不可信）。
    MJPG（逐帧独立 JPEG）与 FFV1（无损）解出来与写进去完全一致。

    写完之后这里会**验证一遍**：确认每帧的标记都读得回来。
    素材不可信时立刻报错，而不是让后面的断言给出误导性的结论。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # 后缀决定容器：MJPG 塞进 .mp4 会被 FFMPEG 拒绝并悄悄退回 mp4v（有帧间预测），
    # 所以容器选 AVI —— MJPG/FFV1 都是 AVI 的原生编码。
    last_err = ""
    for cc in ("MJPG", "FFV1", "mp4v"):
        path = path.with_suffix(".avi" if cc != "mp4v" else ".mp4")
        wr = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*cc), 30.0, (W, H))
        if not wr.isOpened():
            last_err = f"{cc} 不可用"
            continue
        try:
            for i in range(n):
                wr.write(make_frame(i, bg=bg))
        finally:
            wr.release()

        got = read_video_indices(path)
        if got == list(range(n)):
            return path
        last_err = f"{cc} 编解码后有损：读出 {got[:8]}…（应为 {list(range(n))[:8]}…）"
        print(f"    [注意] {cc} 保不住帧号标记，换下一个编码器")

    raise RuntimeError(f"没有任何编码器能无损保存测试素材（{last_err}）。"
                       f"这是测试素材的问题，不是被测代码的问题。")


def read_video_indices(path: Path) -> list[int]:
    """顺序读一段视频，返回每一帧里编好的帧号。"""
    return [read_index(f) for f in read_video_frames(path)]


def read_video_frames(path: Path) -> list[np.ndarray]:
    """顺序读一段视频的全部帧。"""
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, f = cap.read()
        if not ok or f is None:
            break
        out.append(f)
    cap.release()
    return out


def write_pair_from_timeline(base: Path, head: int, n_left: int, n_right: int,
                             flashes: tuple[int, ...] = (20, 44),
                             bg_left: int = 60, bg_right: int = 80) -> tuple[Path, Path]:
    """把**一条连续时间线**切成两段文件：右路早 head 帧开拍（两段有重叠）。

    为什么必须这么做（踩过才知道）：同步模块 `find_offset` 靠的是"同一次闪光
    在两段序列里出现的位置差"。如果两段各自从头拍、内容逐字节相同，
    那么任何偏移处的相关都并列最高，正确答案本就是 0 —— 拿它去驱动取帧，
    自然"找不回"那个人为设定的偏移。真实的错拍也确实是这个形状：
    两台相机同时开录，但各自的**文件第 0 帧**差了一段时间，
    同一物理事件因此在两个文件里的帧号不同。

        时间线   : 0 ......... head ......... head+n_right-1
        左路文件 :              |-- 0 .. n_left-1 --|          （晚 head 帧开拍）
        右路文件 : |----------- 0 .. n_right-1 -----------|

    于是左路内容第 i 帧 与 右路内容第 i+head 帧 是同一物理时刻。
    """
    left = base / "e2e_left.mp4"
    right = base / "e2e_right.mp4"
    left = write_timeline_video(left, start=head, n=n_left, flashes=flashes, bg=bg_left)
    right = write_timeline_video(right, start=0, n=n_right, flashes=flashes, bg=bg_right)
    return left, right


def write_timeline_video(path: Path, start: int, n: int, flashes: tuple[int, ...],
                         bg: int) -> Path:
    """从时间线的 [start, start+n) 区间写一段视频（帧号按位置编，便于校验取帧）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    last_err = ""
    for cc in ("MJPG", "FFV1", "mp4v"):
        path = path.with_suffix(".avi" if cc != "mp4v" else ".mp4")
        wr = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*cc), 30.0, (W, H))
        if not wr.isOpened():
            last_err = f"{cc} 不可用"
            continue
        try:
            for i in range(n):
                t = start + i                       # 时间线上的绝对帧号
                extra = 60.0 if t in flashes else 0.0
                wr.write(make_frame(i, extra, bg=bg))   # 编进去的是**文件内**位置
        finally:
            wr.release()
        got = read_video_indices(path)
        if got == list(range(n)):
            return path
        last_err = f"{cc} 编解码后有损：读出 {got[:8]}…"
        print(f"    [注意] {cc} 保不住帧号标记，换下一个编码器")
    raise RuntimeError(f"没有任何编码器能无损保存测试素材（{last_err}）")


# --------------------------------------------------------------------------
def test_dir_offset(base: Path) -> bool:
    print("----- 图片文件夹：偏移语义 -----")
    left = write_images(base, "dir_left", N)
    right = write_images(base, "dir_right", N)

    cases = [(0, 0), (5, 5), (-4, -4)]
    ok = True
    for offset, expected in cases:
        src = PairSource(directory=(left, right), sync_offset=offset)
        got = []
        while len(got) < 12:
            good, a, b = src.read()
            if not good:
                break
            got.append((read_index(a), read_index(b)))
        src.release()
        # 左路应当从第 max(0, -offset) 帧开始，右路差恒等于 offset
        diffs = {b - a for a, b in got}
        start = got[0][0] if got else None
        pass_ = (diffs == {expected}) and start == max(0, -expected) and len(got) == 12
        ok = ok and pass_
        print(f"  sync_offset={offset:+d} -> 取到 {len(got):2d} 对，"
              f"首对 (左{start}, 右{got[0][1] if got else '-'})，"
              f"帧差 {sorted(diffs)}（应为 [{expected}]）  {'通过' if pass_ else '不达标'}")
    return ok


def test_video_offset(base: Path) -> bool:
    print("----- 视频文件：偏移语义 -----")
    left = write_video(base / "vid_left.mp4", N)
    right = write_video(base / "vid_right.mp4", N)

    ok = True
    for offset in (0, 7, -3):
        src = PairSource(video=(left, right), sync_offset=offset)
        got = []
        while len(got) < 12:
            good, a, b = src.read()
            if not good:
                break
            got.append((read_index(a), read_index(b)))
        src.release()
        diffs = {b - a for a, b in got}
        pass_ = diffs == {offset} and len(got) == 12
        ok = ok and pass_
        print(f"  sync_offset={offset:+d} -> 取到 {len(got):2d} 对，帧差 {sorted(diffs)}"
              f"（应为 [{offset}]）  {'通过' if pass_ else '不达标'}")
    return ok


def test_end_to_end(base: Path) -> bool:
    """造一对真实错开 k 帧的视频 -> find_offset 算出偏移 -> 拿它取帧 -> 必须对齐。

    这是本测试的核心：偏移量从同步模块流到取帧模块，中间不能掉链子。

    场景设定：把一条连续时间线切成两段文件，**右路早 head 帧开拍**，两段有重叠。
    于是左路文件第 i 帧 与 右路文件第 i+head 帧 是同一物理时刻，
    find_offset 应当报出 +head，PairSource 用这个偏移取帧时左右帧号差应恒为 +head。

    一开始这里写成"右路晚 head 帧开拍"，那是**做不到**的：
    左路开拍那一刻右路还没开拍，画面里根本没有对应内容，任何取帧都不可能对齐。
    这个方向性错误不会报错，只会让测试永远不通过。
    """
    print("----- 端到端：find_offset 算出的偏移直接喂给 PairSource -----")
    head = 9
    n_left, n_right = 40, 49
    left, right = write_pair_from_timeline(base, head=head, n_left=n_left, n_right=n_right)
    print(f"  右路比左路早开拍 {head} 帧（左 {n_left} 帧 / 右 {n_right} 帧，有重叠）")

    from stereo import brightness_series
    res = find_offset(brightness_series(read_video_frames(left)),
                      brightness_series(read_video_frames(right)),
                      SyncConfig(max_offset=30))
    print(f"  find_offset 给出 {res.offset:+d}（置信度 {res.confidence:.3f}）")

    src = PairSource(video=(left, right), sync_offset=res.offset)
    got = []
    while len(got) < 10:
        good, a, b = src.read()
        if not good:
            break
        got.append((read_index(a), read_index(b)))
    src.release()

    diffs = {b - a for a, b in got}
    checks = [
        (f"find_offset 找回 {head} 帧", res.offset == head),
        (f"左右帧号差 == {head}（实测 {sorted(diffs)}，前几对 {got[:4]}）", diffs == {head}),
    ]
    ok = True
    for name, passed in checks:
        print(f"  [{'通过' if passed else '不达标'}] {name}")
        ok = ok and passed
    return ok


def test_guards(base: Path) -> bool:
    """构造错误必须报错，不能悄悄给错结果。"""
    print("----- 参数校验 -----")
    left = write_images(base, "guard_left", 6)
    right = write_images(base, "guard_right", 4)     # 故意少两张

    checks: list[tuple[str, bool]] = []

    def expect_raises(name, fn, exc=Exception):
        try:
            fn()
        except exc:
            checks.append((name, True))
        except Exception as e:                       # noqa: BLE001
            print(f"    （期望 {exc.__name__}，实际 {type(e).__name__}: {e}）")
            checks.append((name, False))
        else:
            checks.append((name, False))

    expect_raises("一种帧源都不给 -> 报错", lambda: PairSource(), RuntimeError)
    expect_raises("同时给两种帧源 -> 报错",
                  lambda: PairSource(camera=(0, 1), directory=(left, right)), RuntimeError)
    expect_raises("图片数不一致 -> 报错",
                  lambda: PairSource(directory=(left, right)), RuntimeError)
    expect_raises("单张图片 -> 报错",
                  lambda: PairSource(directory=(left / "000.png", right / "000.png")),
                  RuntimeError)
    expect_raises("左右同一台相机 -> 报错",
                  lambda: PairSource(camera=(3, 3)), RuntimeError)

    ok = True
    for name, passed in checks:
        print(f"  [{'通过' if passed else '不达标'}] {name}")
        ok = ok and passed
    return ok


def test_from_args(base: Path) -> bool:
    """命令行参数 -> PairSource 的映射（两个 app 共用这一份）。

    为什么值得单独测：重构时最容易漏掉的不是库，而是**调用点** ——
    实测 `verify_stereo.py` 里就漏改了一处 `PairSource(args)`，
    于是复检工具在运行时报 `TypeError: 'Namespace' object is not subscriptable`。
    这里用 argparse 真造一个 Namespace，把三种映射都走一遍。
    """
    print("----- 命令行参数映射（from_args） -----")
    import argparse

    from stereo import from_args

    left = write_images(base, "args_left", 4)
    right = write_images(base, "args_right", 4)

    def ns(**kw):
        d = dict(left=None, right=None, left_video=None, right_video=None,
                 left_dir=None, right_dir=None, width=1920, height=1080, fps=60,
                 backup_camera=None, backend=None, auto_cameras=False)
        d.update(kw)
        return argparse.Namespace(**d)

    checks: list[tuple[str, bool]] = []
    src = from_args(ns(left_dir=str(left), right_dir=str(right)))
    checks.append(("--left-dir/--right-dir -> dir 帧源", src.kind == "dir"))
    src.release()

    # ---- `--backend auto` 不许被折叠成某个具体后端（2026-10-14 加）----
    #
    # 守的真缺陷：`from_args` 原来把 `None`（= auto）
    # **又赋成 `INDEX_BACKEND`（MSMF）**，于是 `PairSource` 里那份
    # `[("DSHOW", ...), ("MSMF", ...)]` 成对候选顺序**从来没被走到过** ——
    # 函数名叫 `auto`，行为却是"强制 MSMF"。实测后果（真机标定）：
    #
    #     MSMF : idx1=None（读 12 帧全空）  idx0=1920x1080
    #     DSHOW: idx1=1920x1080            idx0=1920x1080
    #
    # 即 MSMF 对其中一台根本不给帧，而 `auto` 必选它 → 标定跑不起来，
    # 还打出一条把人引向"换 USB 口"的错误建议。
    #
    # 判据是**源码级**的（与 `selftest_doc_facts` 里那几条同一手法）：
    # 这一点要真起相机才验得了，而"auto 传了什么给 PairSource"是纯源码事实。
    import cv2 as _cv2

    from stereo.pair_source import resolve_backend
    checks.append(("resolve_backend('auto') == None（auto 就是「不指定」）",
                   resolve_backend("auto") is None))
    checks.append(("resolve_backend('') == None", resolve_backend("") is None))
    checks.append(("resolve_backend('dshow') == CAP_DSHOW",
                   resolve_backend("dshow") == _cv2.CAP_DSHOW))
    checks.append(("resolve_backend('msmf') == CAP_MSMF",
                   resolve_backend("msmf") == _cv2.CAP_MSMF))

    pair_src = (ROOT / "stereo" / "pair_source.py").read_text(encoding="utf-8")
    body = pair_src.split("def from_args(")[1].split("\ndef ")[0]

    # ⚠️ **2026-10-18 改了口径**（旧判据现在恰好反了）：
    # 上一版判据是"from_args 里**不许**出现 `api = INDEX_BACKEND`"，
    # 理由是"auto 应该真的成对挑后端"。
    # 但那次改成"auto → DSHOW 优先"引出了更严重的 bug：
    # `left_i/right_i` 是 **PnP 位次**，DSHOW 的编号空间与它错位
    # （实测 DSHOW #1=内置、#2=aoni），于是**角色 right 会开到内置相机**，
    # 两路都报 1920x1080、界面上完全看不出 —— 比"打不开"危险得多。
    # 所以现在的正确行为是：**索引来自角色/auto 时锁死 INDEX_BACKEND**；
    # 想用别的后端必须显式给数字编号。
    checks.append(("索引来自角色/auto 时**锁死** INDEX_BACKEND（PnP 位次只对它成立）",
                   "api = INDEX_BACKEND" in body))
    checks.append(("from_args 确实把 api 传给了 PairSource（不是传了个死值）",
                   "backend=api" in body))
    checks.append(("显式数字编号时**不**锁后端（用户自认编号空间）",
                   "_explicit_index" in body))

    # 守卫：角色名 + dshow 必须被拒绝（这是"静默用错相机"的唯一入口）
    #
    # ⚠️ **2026-10-19 修假红**：这一段原来直接调 `from_args`，它会去查**真实的**
    # `config/cameras.json` 与真实设备表。没插相机时，角色解析（`resolve_camera_arg`）
    # 在 `from_args` 里就先抛"没有为「left」选定相机"，**走不到后面的后端守卫**
    # —— 判据要的"不能一起用 / PnP"当然对不上，于是 fast 档在没有相机的机器上必红。
    # 这是项目记过的坑 24（"替身测试必须断言替身真的走到了那一步"）的同型问题。
    #
    # 现在只替身**输入**（角色解析 + PairSource 构造）：
    #   - 守卫本身仍走产品代码，不是把判据改写成源码字符串；
    #   - 两条"非空转"判据见下面：left/right 各解析一次 + 桩没被构造；
    #   - PairSource 换成桩，万一守卫没抛，也不会真的去开相机。
    from stereo import pair_source as _pair_source

    real_resolve = _pair_source.resolve_camera_arg
    real_pair_source = _pair_source.PairSource
    resolved: list[str] = []
    built: list[tuple] = []

    class _StubPairSource:
        """守卫没拦住时才会被构造 —— 只记录参数，绝不碰相机。"""

        def __init__(self, *args, **kwargs) -> None:
            built.append((args, kwargs))

    def _fake_resolve(value, **_kw):
        # `**_kw` 不是偷懒：真函数是 `resolve_camera_arg(value, *, interactive=True)`，
        # 而 `pair_source` 现在显式传 `interactive=False`（后台/GUI 子进程没有 tty 输入，
        # 交互 `input()` 会永远挂住）。替身签名必须与真函数一致，否则这道守卫
        # **根本走不到**，红色会被误读成"守卫坏了"（替身陷阱，坑 24）。
        resolved.append(str(value))
        return {"left": 0, "right": 1}.get(str(value), 0)

    _pair_source.resolve_camera_arg = _fake_resolve
    _pair_source.PairSource = _StubPairSource
    try:
        try:
            from_args(ns(left="left", right="right", backend="dshow"))
        except RuntimeError as exc:
            msg = str(exc)
            ok_guard = ("不能一起用" in msg) and ("PnP" in msg)
            print(f"  守卫拒绝：{msg.splitlines()[0][:74]}")
            checks.append(("角色名 + --backend dshow -> **拒绝**（否则静默用错相机）", ok_guard))
        except Exception as exc:                     # noqa: BLE001
            checks.append(("角色名 + --backend dshow -> **拒绝**（否则静默用错相机）", False))
            print(f"  守卫抛了非 RuntimeError：{type(exc).__name__}: {exc}")
        else:
            checks.append(("角色名 + --backend dshow -> **拒绝**（否则静默用错相机）", False))
            print("  守卫没有拒绝：from_args 成功返回了（已用桩，没有真开相机）")
    finally:
        _pair_source.resolve_camera_arg = real_resolve
        _pair_source.PairSource = real_pair_source

    # 两条"非空转"判据，分工不同：
    #   ① 角色解析被调用了 left/right **各一次**（顺序不算契约，用 Counter 抹掉）；
    #   ② 守卫在**构造 PairSource 之前**就拒绝 —— 否则守卫若被挪到构造之后，
    #      真机会先开相机、而异常照样让测试变绿（`built` 非空才抓得住）。
    checks.append(("替身确实被调用：left/right 各一次",
                   Counter(resolved) == Counter({"left": 1, "right": 1})))
    checks.append(("守卫在构造 PairSource 之前就拒绝（没真开相机）", built == []))

    try:
        from_args(ns())
    except RuntimeError:
        checks.append(("什么都没给 -> 报错", True))
    except Exception:                                # noqa: BLE001
        checks.append(("什么都没给 -> 报错", False))
    else:
        checks.append(("什么都没给 -> 报错", False))

    ok = True
    for name, passed in checks:
        print(f"  [{'通过' if passed else '不达标'}] {name}")
        ok = ok and passed
    return ok


def test_short_side(base: Path) -> bool:
    """右路先读完时必须干净结束（返回 False），不能抛异常、不能给错帧。"""
    print("----- 右路比左路短（右 5 张 / 左 10 张，靠偏移错开） -----")
    left = write_images(base, "short_left", 10)
    right = write_images(base, "short_right", 5)
    # 图片文件夹要求两路张数一致（那是"一一对应"的语义），
    # 想测长度不一致的结束行为就得用视频：两段视频长度本来就可以不同。
    vl = write_video(base / "short_left.mp4", 10)
    vr = write_video(base / "short_right.mp4", 5)
    src = PairSource(video=(vl, vr))
    pairs = []
    while True:
        good, a, b = src.read()
        if not good:
            break
        pairs.append((read_index(a), read_index(b)))
        if len(pairs) > 50:                          # 死循环保护
            break
    src.release()

    checks = [
        ("配对在右路用完时停止（5 对）", len(pairs) == 5),
        ("取到的都是逐帧对齐的", pairs == [(i, i) for i in range(len(pairs))]),
    ]
    ok = True
    for name, passed in checks:
        print(f"  [{'通过' if passed else '不达标'}] {name}")
        ok = ok and passed
    return ok


def test_sessions(base: Path) -> bool:
    """采集会话：写→读→校验；坏会话必须逐条报错（不能静默通过）。

    为什么值得测：`session.json` 是数据采集的唯一契约。schema 少一个字段、
    校验把"分辨率不一致"放过，都会让后面几十次采集白录。
    """
    print("----- 采集会话（session.json 读写与校验） -----")
    checks: list[tuple[str, bool]] = []

    session_dir, session = create_session(base / "sessions", label="selftest")
    for side in ("left", "right"):
        path = session_dir / f"{side}.avi"
        wr = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"FFV1"), 30.0, (W, H))
        if wr.isOpened():
            for i in range(5):
                wr.write(make_frame(i))
            wr.release()
        checks.append((f"{side}.avi 用 FFV1 写出", path.exists()))

    K = np.array([[100.0, 0.0, W / 2], [0.0, 100.0, H / 2], [0.0, 0.0, 1.0]])
    D = np.zeros((1, 5))
    T = np.array([[-0.065], [0.0], [0.0]])
    rig = StereoRig(K1=K, D1=D, K2=K.copy(), D2=D.copy(), R=np.eye(3), T=T,
                    image_size=(W, H), quality={"rms": 0.5},
                    meta={"created": "2026-01-01 00:00:00",
                          "stereo_mode": "angled", "angle_deg": 90.0})
    rig.save(session_dir / "calib" / "stereo_calibration")

    session.left = CameraInfo("left", "left.avi", W, H, 5, 30.0, 30.0)
    session.right = CameraInfo("right", "right.avi", W, H, 5, 30.0, 30.0)
    session.calibration = CalibrationInfo(
        "calib/stereo_calibration.npz",
        sha256_file(session_dir / "calib" / "stereo_calibration.npz"),
        [W, H], 65.0, 65.0, "2026-01-01 00:00:00", "测试",
    )
    session.sync = SyncInfo("flash", 2, 0.9)
    session.stereo_mode = "angled"
    session.rig = {"stereo_mode": "angled", "angle_deg": 90.0,
                   "baseline_mm": 65.0}
    save_session(session, session_dir)

    errors, warnings = validate_session(session_dir)
    checks.append(("干净会话：无错误", not errors))
    checks.append(("干净会话：无警告", not warnings))

    session.stereo_mode = "parallel"
    save_session(session, session_dir)
    e_mix, w_mix = validate_session(session_dir)
    checks.append(("会话 parallel + 标定 angled -> 警告",
                   not e_mix and any("不一致" in x for x in w_mix)))
    session.stereo_mode = "angled"
    save_session(session, session_dir)

    session.sync = SyncInfo("none", 0, 0.0)
    save_session(session, session_dir)
    e_none, w_none = validate_session(session_dir)
    checks.append(("未同步 -> 警告（三维不可直接用）",
                   not e_none and any("未同步" in x for x in w_none)))
    session.sync = SyncInfo("flash", 2, 0.9)
    save_session(session, session_dir)

    _d, reloaded = load_session(session_dir)
    checks.append(("重新读回 label", reloaded.label == "selftest"))
    checks.append(("重新读回同步偏移", reloaded.sync.offset == 2))
    checks.append(("重新读回双目模式/夹角",
                   reloaded.stereo_mode == "angled"
                   and reloaded.rig.get("angle_deg") == 90.0))

    # 坏会话必须报错，不能静默通过
    session.left.width = W + 1
    save_session(session, session_dir)
    e, _w = validate_session(session_dir)
    checks.append(("分辨率不一致 -> 报错", any("不一致" in x for x in e)))
    session.left.width = W
    save_session(session, session_dir)

    session.calibration.sha256 = "deadbeef"
    save_session(session, session_dir)
    e, _w = validate_session(session_dir)
    checks.append(("标定 sha256 不对 -> 报错", any("sha256" in x for x in e)))
    session.calibration.sha256 = sha256_file(session_dir / "calib" / "stereo_calibration.npz")
    save_session(session, session_dir)

    session.calibration.baseline_measured_mm = 100.0
    save_session(session, session_dir)
    e, _w = validate_session(session_dir)
    checks.append(("物理基线差 >20% -> 报错", any("物理基线" in x for x in e)))
    session.calibration.baseline_measured_mm = 65.0
    save_session(session, session_dir)

    # 判定分三档：容器 fps 标称偏差（≤25%）只是提示；差 >25% 是时间基真错，
    # 必须警告；缺 sidecar 时容器是唯一时间基，差 >10% 就警告。
    session.left.fps_measured = 60.0        # 容器写的是 30 fps，主循环实测 60
    save_session(session, session_dir)

    def _write_sidecars(fps: float) -> None:
        for side in ("left", "right"):
            with (session_dir / f"{side}_times.csv").open(
                    "w", encoding="utf-8", newline="") as f:
                f.write("frame,perf_counter_s,unix_time_s\n")
                for i in range(5):              # 5 帧 @ fps，容器写的是 30 fps
                    f.write(f"{i},{100.0 + i / fps:.6f},"
                            f"{1700000000.0 + i / fps:.6f}\n")

    _write_sidecars(36.0)                   # |30-36|/36 = 17% -> 标称偏差
    e_n, w_n, notes = validate_session(session_dir, with_notes=True)
    checks.append(("容器 fps 与 sidecar 差 17% -> 只算提示，不算警告",
                   not e_n and not any("容器 fps" in x for x in w_n)
                   and any("sidecar 实测" in x for x in notes)))

    _write_sidecars(60.0)                   # |30-60|/60 = 50% -> 时间基错误
    e_r, w_r, n_r = validate_session(session_dir, with_notes=True)
    checks.append(("容器 fps 与 sidecar 差 50% -> 警告（时间基错误）",
                   not e_r and any("时间基错误" in x for x in w_r)
                   and not any("sidecar 实测" in x for x in n_r)))

    _write_sidecars(30.0)
    (session_dir / "left_times.csv").rename(session_dir / "left_times.csv.off")
    e_s, w_s, _no = validate_session(session_dir, with_notes=True)
    checks.append(("缺 sidecar -> fps 差回到警告（时间基不可信）",
                   not e_s and any("容器 fps" in x for x in w_s)))
    (session_dir / "left_times.csv.off").rename(session_dir / "left_times.csv")

    # sidecar 不完整必须独立上报：fps_measured 缺失时也不能静默
    session.left.fps_measured = 0.0
    save_session(session, session_dir)
    with (session_dir / "left_times.csv").open("w", encoding="utf-8", newline="") as f:
        f.write("frame,perf_counter_s,unix_time_s\n")
        for i in range(3):                      # 5 帧视频只写 3 行
            f.write(f"{i},{100.0 + i / 30.0:.6f},{1700000000.0 + i / 30.0:.6f}\n")
    e_i, w_i, _ni = validate_session(session_dir, with_notes=True)
    checks.append(("sidecar 不完整 -> 独立警告（含行数/帧数）",
                   not e_i and any("sidecar 不完整（3 行 / 5 帧）" in x for x in w_i)))

    for side in ("left", "right"):
        (session_dir / f"{side}_times.csv").unlink()
    session.left.fps_measured = 30.0
    save_session(session, session_dir)

    (session_dir / "right.avi").unlink()
    e, _w = validate_session(session_dir)
    checks.append(("缺右路视频 -> 报错", any("视频不存在" in x for x in e)))

    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_live_camera_skipped() -> bool:
    """实时相机只做一次试探：没有相机就跳过（不算不达标）。"""
    print("----- 实时相机（有相机才测） -----")
    try:
        src = PairSource(camera=(0, 1), width=640, height=480)
    except Exception as e:                           # noqa: BLE001
        print(f"  [跳过] 当前没有可用的双相机（{type(e).__name__}）：{e}")
        print("         真机验证留到「真机标定」那一步一起做")
        return True
    try:
        align = []
        for _ in range(5):
            good, a, b = src.read()
            if good:
                align.append((int(np.median(a)), int(np.median(b))))
        print(f"  取到 {len(align)} 对实时帧（时间差已压到一帧内，测不出精确同步）")
        print(f"  [跳过] 实时偏移本就不支持；要精确同步请录视频后用 find_offset")
        return True
    finally:
        src.release()


# --------------------------------------------------------------------------
def test_timestamp_pairing(base: Path) -> bool:
    """时间戳配对（金丝雀路径）：掉帧要跳过，不能拿常数偏移硬配。"""
    print("----- 时间戳配对 / 金丝雀 -----")
    checks: list[tuple[str, bool]] = []

    lt = np.arange(40, dtype=np.float64) / 30.0
    rt = (np.arange(40, dtype=np.float64) + 0.001) / 30.0
    rt = np.delete(rt, 10)                       # 右路第 10 帧丢了
    plan = plan_timestamp_pairs(lt, rt, tolerance_s=0.02)
    pairs = list(plan.pairs)
    checks.append(("掉帧后仍能配对，且左 10 被跳过",
                   len(pairs) == 39 and all(a != 10 for a, _b in pairs)))
    checks.append(("配对严格一一对应、单调递增",
                   len({a for a, _b in pairs}) == len(pairs)
                   and len({b for _a, b in pairs}) == len(pairs)
                   and all(pairs[i][1] < pairs[i + 1][1]
                           for i in range(len(pairs) - 1))))
    checks.append(("残差不超过容差",
                   plan.max_residual_ms <= 20.0))
    checks.append(("金丝雀允许接管", canary_safe(plan)[0]))
    low = plan_timestamp_pairs(lt, rt[:10], tolerance_s=0.02)
    checks.append(("覆盖率太低时金丝雀拒绝接管", not canary_safe(low)[0]))
    print(f"  {plan.summary()}")

    # 视频 + sidecar 集成：左 40 帧 / 右 40 帧，但右路时间戳缺第 10 帧
    left = write_video(base / "ts_left.mp4", N)
    right = write_video(base / "ts_right.mp4", N)
    t0 = 1000.0
    write_times(base / "ts_left_times.csv", [t0 + i / 30.0 for i in range(N)])
    write_times(base / "ts_right_times.csv",
                [t0 + 0.001 + i / 30.0 for i in range(N) if i != 10])
    src = PairSource(video=(left, right), pairing="timestamp")
    got = []
    while len(got) < 12:
        good, a, b = src.read()
        if not good:
            break
        got.append((read_index(a), read_index(b)))
    src.release()
    checks.append(("PairSource 用 timestamp 计划取帧",
                   got[:10] == [(i, i) for i in range(10)]
                   and got[10][0] == 11 and got[10][1] == 10))
    print(f"  前 10 对：{got}")

    # 转正：不显式传 pairing 时也应走 canary（有 sidecar 就接管）
    default_src = PairSource(video=(left, right))
    default_pairs = default_src._pair_plan
    default_ok = default_pairs is not None and len(default_pairs.pairs) == 39
    default_got = []
    while len(default_got) < 3:
        good, _a, _b = default_src.read()
        if not good:
            break
        default_got.append(1)
    default_src.release()
    checks.append(("默认 pairing=canary 会接管时间戳计划",
                   default_ok and len(default_got) == 3))

    # 缺 sidecar -> 金丝雀必须回退，而不是静默改变输出
    no_ts_left = write_video(base / "no_ts_left.mp4", 6)
    no_ts_right = write_video(base / "no_ts_right.mp4", 6)
    fallback = PairSource(video=(no_ts_left, no_ts_right), pairing="canary")
    fallback_ok = fallback._pair_plan is None
    fallback.release()
    checks.append(("缺时间戳时金丝雀回退索引", fallback_ok))

    for name, passed in checks:
        print(f"  [{'通过' if passed else '不达标'}] {name}")
    return all(p for _n, p in checks)


def write_times(path: Path, times: list[float]) -> None:
    """写 sidecar，格式与 record_stereo.py 一致。"""
    lines = ["frame,perf_counter_s,unix_time_s"]
    for i, t in enumerate(times):
        lines.append(f"{i},{t:.9f},{t:.9f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
def test_open_join_timeout() -> bool:
    """并行打开相机的**整体超时**：改前裸 `th.join()` 会永久挂住主线程。

    ⚠️ **这就是 Q3 的证伪**（对照数值）：改前 `CameraPairReader.__init__` 里是
    裸 `th.join()`（无 timeout），本判据会**永久挂住** —— 进程永不返回，
    这正是"相机管理卡死"的机制。改成 `_join_all(threads, timeout_s)` 之后，
    同一个"永不返回"的线程会在 ~0.2s 内被判定为"仍存活"并返回。

    判据不依赖任何真相机：被 join 的是纯 `time.sleep` 线程。
    实测耗时（改后）见下面打印出来的 `dt`。
    """
    print("----- 并行打开：整体超时（不依赖真相机） -----")
    import threading
    import time

    from stereo.pair_source import _join_all

    checks: list[tuple[str, bool]] = []

    # 负向（关键）：一个 30s 不返回的线程，超时 0.2s 必须**很快**返回，
    # 且它必须出现在"仍存活"名单里（= 是"可控失败"，不是"静默丢相机"）。
    th = threading.Thread(target=lambda: time.sleep(30), daemon=True, name="never")
    th.start()
    t0 = time.perf_counter()
    got = _join_all([th], 0.2)
    dt = time.perf_counter() - t0
    checks.append((f"永不返回的线程：{dt:.3f}s 内返回（应 < 1.0s）", dt < 1.0))
    checks.append(("永不返回的线程出现在「仍存活」名单里", th in got))
    print(f"  改后实测 dt = {dt:.3f}s（改前这里是裸 join()，会永久挂住，进程不返回）")

    # 正向：很快结束的线程必须**不在**存活名单里（返回空列表）。
    th2 = threading.Thread(target=lambda: time.sleep(0.05), daemon=True, name="quick")
    th2.start()
    got2 = _join_all([th2], 0.5)
    checks.append(("正常结束的线程：不在「仍存活」名单里（返回空列表）", got2 == []))

    ok = True
    for name, passed in checks:
        print(f"  [{'通过' if passed else '不达标'}] {name}")
        ok = ok and passed
    return ok


# --------------------------------------------------------------------------
def test_auto_negotiate_helpers() -> bool:
    """自适应成对协商的**纯逻辑**部分（不依赖真相机）。

    覆盖 S4 协商里最容易写错、又最容易被"合成真值漂亮"掩盖的几点：
    档位标签、成对帧率必须取**两路较小者**（不是某一台）、刚打开测不到帧时
    要**先等首帧再测**（A5）、帧率下限与 `DEFAULT_FPS` 的换算。判据不依赖相机：
    被测量的是纯 `time` 驱动的假相机。
    """
    print("----- 自适应成对协商：纯逻辑（不依赖相机） -----")
    import time

    from sphere_tracker.video_profiles import DEFAULT_FPS, probe_order
    from stereo.pair_source import (
        AUTO_MIN_PAIR_FPS,
        _measure_pair_fps,
        _pair_fps_sample,
        _profile_label,
    )

    checks: list[tuple[str, bool]] = []

    # 档位标签：原生档 (0,0) 必须说成"原生"，不能打印成会误导的 "0x0"
    checks.append(("原生档标签", _profile_label(0, 0) == "原生"))
    checks.append(("普通档标签", _profile_label(1920, 1080) == "1920x1080"))

    # 帧率下限 = DEFAULT_FPS * 0.6
    checks.append((f"帧率下限 = DEFAULT_FPS({DEFAULT_FPS}) * 0.6",
                   abs(AUTO_MIN_PAIR_FPS - DEFAULT_FPS * 0.6) < 1e-9))

    # 协商的档序来源：probe_order(None) 从最高档到原生档
    order = probe_order(None)
    checks.append(("档序首个是最高档", order[0] == (3840, 2160)))
    checks.append(("档序末尾是原生档", order[-1] == (0, 0)))

    class _FakeCam:
        """够用的假相机：`seq` 按 `rate` 匀速增长；`first_delay` 内不出帧。"""

        def __init__(self, rate: float, first_delay: float = 0.0) -> None:
            self._rate = float(rate)
            self._t0 = time.perf_counter()
            self._delay = float(first_delay)

        def _seq(self) -> int:
            dt = time.perf_counter() - self._t0 - self._delay
            return max(0, int(dt * self._rate))

        def latest(self, last_seq: int = -1):
            return self._seq(), None, 0.0

        def wait_first(self, timeout: float = 10.0):
            if self._delay <= 0:
                return object()
            time.sleep(min(self._delay, timeout))
            return object()

    # 成对帧率 = **两路较小者**：快的 100、慢的 20 -> 结果 ≈20，而不是 ≈60。
    rate = _pair_fps_sample([_FakeCam(100.0), _FakeCam(20.0)], budget=0.4)
    checks.append((f"成对帧率取两路较小者（测得 {rate:.0f}，应 ≈20 而非 ≈60）",
                   10.0 <= rate <= 30.0))

    # A5：刚打开（首个 0.8s 不出帧）短测会得到 0 -> 必须先等首帧再测。
    rate2 = _measure_pair_fps([_FakeCam(25.0, first_delay=0.8)],
                              budget=0.2, settle_max=2.0)
    checks.append((f"刚打开测不到时先等首帧再测（测得 {rate2:.0f}，应 > 0）",
                   rate2 > 1.0))

    ok_all = True
    for name, passed in checks:
        print(f"  [{'通过' if passed else '不达标'}] {name}")
        ok_all = ok_all and passed
    return ok_all


# --------------------------------------------------------------------------
def test_auto_negotiate_guards() -> bool:
    """A3 **真帧**守卫：**上报一致但真帧不一致**必须被拒（不静默接受）。

    这是本次修复（独立 Critic 判出的高）的**证伪点**：老代码只比
    `cap.get()` 的**上报值**，两台上报同为 1920x1080、真帧一个 1280x1080
    一个 1920x1080 时会被**静默接受**一对尺寸不一致的相机。

    这里**驱动真实的 `_negotiate`**（不复制逻辑）：只把"打开两路"替身掉，
    A3/A2/A5 守卫本身照走产品代码。
    """
    print("----- 自适应成对协商：A3 真帧守卫（驱动真实 _negotiate） -----")
    import time
    from types import SimpleNamespace

    from stereo.pair_source import CameraPairReader

    class _FakeCam:
        """假相机：`info` 是**上报值**，`wait_first` 给的是**真帧**。"""

        def __init__(self, reported, frame, seq_rate: float = 30.0) -> None:
            self.info = SimpleNamespace(width=reported[0], height=reported[1])
            self._frame = frame
            self._t0 = time.perf_counter()
            self._rate = float(seq_rate)
            self.stopped = False

        def latest(self, last_seq: int = -1):
            return max(0, int((time.perf_counter() - self._t0) * self._rate)), \
                self._frame, 0.0

        def wait_first(self, timeout: float = 10.0):
            return self._frame

        def stop(self) -> None:
            self.stopped = True

    def _frame(w: int, h: int):
        return np.zeros((h, w, 3), np.uint8)

    class _StubReader(CameraPairReader):
        """跳过真开相机：只替身 `_try_open_pair`，`_negotiate` 照走产品代码。"""

        def __init__(self, factory) -> None:
            # 不调 super().__init__（那会真开相机）；只补 `_negotiate` 需要的属性。
            self._left_index, self._right_index = 0, 1
            self._auto = True
            self.requested_size = "auto"
            self.resolved_size = None
            self.actual_sizes = []
            self.pair_fps = None
            self._fps = 30
            self._backup_camera = None
            self.cams = []
            self._factory = factory

        def _try_open_pair(self, name, api, rw, rh, errors, deadline=None):
            return self._factory(rw, rh)

    checks: list[tuple[str, bool]] = []

    # ① 上报一致（都报 1920x1080）但**真帧不一致**（左 1920x1080 / 右 1280x1080）
    #    -> 每一档都必须被拒；全档走完 -> cams 空 + attempts 写明真帧尺寸不一致。
    r1 = _StubReader(lambda rw, rh: [_FakeCam((1920, 1080), _frame(1920, 1080)),
                                     _FakeCam((1920, 1080), _frame(1280, 1080))])
    errs1: list[str] = []
    r1._negotiate([("指定", None)], errs1)
    joined1 = "\n".join(errs1)
    checks.append(("上报一致、真帧不一致 -> **拒绝**（cams 为空，未静默接受）",
                   r1.cams == []))
    checks.append(("拒绝原因写明「实际帧尺寸不一致」且带两路尺寸",
                   "实际帧尺寸" in joined1
                   and "1920x1080" in joined1 and "1280x1080" in joined1))

    # ② 任一路拿不到真帧（右路 `wait_first` 永远 None）-> 拒绝、降档。
    r2 = _StubReader(lambda rw, rh: [_FakeCam((1920, 1080), _frame(1920, 1080)),
                                     _FakeCam((1920, 1080), None)])
    errs2: list[str] = []
    r2._negotiate([("指定", None)], errs2)
    joined2 = "\n".join(errs2)
    checks.append(("任一路取不到真帧 -> **拒绝**（cams 为空）", r2.cams == []))
    checks.append(("拒绝原因写明「取不到真帧」", "取不到真帧" in joined2))

    # ③ 对照：上报一致、真帧也一致、帧率够快 -> **接受**（证明①②不是恒拒绝）。
    r3 = _StubReader(lambda rw, rh: [_FakeCam((1920, 1080), _frame(1920, 1080)),
                                     _FakeCam((1920, 1080), _frame(1920, 1080))])
    errs3: list[str] = []
    r3._negotiate([("指定", None)], errs3)
    checks.append(("对照：上报/真帧都一致、帧率够 -> **接受**", len(r3.cams) == 2))
    checks.append(("接受时 resolved_size / actual_sizes 记的是**真帧**尺寸",
                   r3.resolved_size == (1920, 1080)
                   and r3.actual_sizes == [(1920, 1080), (1920, 1080)]))

    ok_all = True
    for name, passed in checks:
        print(f"  [{'通过' if passed else '不达标'}] {name}")
        ok_all = ok_all and passed
    return ok_all


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="成对帧源自测")
    ap.add_argument("--keep", action="store_true",
                    help="保留合成的素材目录（默认跑完删掉），便于人工排查")
    args = ap.parse_args()

    print("=" * 62)
    print("成对帧源自测（合成真值，不依赖相机与标定）")
    print("=" * 62)

    base = make_scratch_dir(keep=args.keep)
    print(f"素材目录：{base}")
    try:
        results = [
            test_dir_offset(base),
            test_video_offset(base),
            test_end_to_end(base),
            test_guards(base),
            test_from_args(base),
            test_sessions(base),
            test_short_side(base),
            test_timestamp_pairing(base),
            test_open_join_timeout(),
            test_auto_negotiate_helpers(),
            test_auto_negotiate_guards(),
        ]
        print()
        results.append(test_live_camera_skipped())
    finally:
        if args.keep:
            print(f"\n已保留素材目录：{base}")
        else:
            shutil.rmtree(base, ignore_errors=True)

    print()
    ok = all(results)
    print("判定结果：" + ("全部通过" if ok else "有不达标项"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
