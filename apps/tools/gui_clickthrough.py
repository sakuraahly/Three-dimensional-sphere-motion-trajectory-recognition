"""按**真实用户操作**走一遍相机预览：启动 GUI → 点按钮 → 截图取证。

## 为什么需要它

2026-11-08 用户连着报「相机管理打不开」，而我前几轮都是用 `dlg._preview_all()`
直接调方法验证的 —— 按钮和它的前置条件（Treeview 选中行、单例）一次都没走过。
`AGENTS.md §7.3` 因此规定：GUI 功能验证必须从入口点进去、点真按钮。

这个工具就是那条规则的执行器：用 `pywinauto` 发**真实鼠标事件**，每一步截图，
找不到目标窗口/按钮就立刻报失败并留下现场，不往下猜。

## 用法

    python apps/tools/gui_clickthrough.py            # 走一遍，截图存 outputs/gui_click/
    python apps/tools/gui_clickthrough.py --dry-run  # 只检查环境（桌面是否解锁、库是否在）

**桌面锁屏时无法执行**（前台窗口 hwnd=0、`OpenInputDesktop` 失败、截图也会
`screen grab failed`）。脚本会直接说明并退出，不会假装点过。

## 坐标从哪来

Tk 的控件不是独立 HWND，pywinauto 拿不到按钮对象，只能按客户区比例点击。
三个比例都来自实际窗口截图（1960x1652 / 920x818 / 1180x720），改了布局要一起改。
"""

from __future__ import annotations

import argparse
import ctypes
import subprocess
import sys
import time
from pathlib import Path

# 按标记上溯到含 `sphere_tracker/` 的仓库根，不写死 parents[N]：
# 本文件现在也 import 项目包（stereo.profiles），必须能独立引导。
ROOT = Path(__file__).resolve()
while not (ROOT / "sphere_tracker").is_dir():
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: 三个点击点在各自窗口**客户区**里的相对位置（x 比例, y 比例）。
#:
#: ⚠️ 已知缺陷（2026-11-09 静态查明，未实机复验）：**比例坐标依赖窗口尺寸**。
#: 门厅三张卡是贴顶、固定高度排下来的，窗口变高时多出来的空白落在别处，
#: 按钮并不按比例下移。历史校准截图 `outputs/gui_click/calibrate_hub.png`
#: 是 996x865，而 11-08 那次 exit 5 的点击落在 (213,1615) —— y 远超窗口
#: 高度，等于点在窗口外。下次实机要先固定窗口尺寸（SetWindowPos 带 w/h）
#: 再重新量三个按钮的绝对像素；本工具现在会在坐标落到别的窗口/桌面时
#: 用 exit 8 明确报「没点过」，不再含糊地报 exit 5。
HUB_PARALLEL = (0.101, 0.464)      # 门厅 ·「平行双目 · 进入」（按 2026-11-08 实拍重算）
#: 大角度档案下平行卡被门控禁用，要点这张（同一比例尺实测 y≈0.72）。
HUB_ANGLED = (0.101, 0.720)        # 门厅 ·「大角度双目 · 进入」
STEREO_CAMERAS = (0.235, 0.191)    # 双目 ① ·「换相机…」
CAM_PREVIEW = (0.036, 0.855)       # 相机管理 ·「预览这台」（底栏按钮，不是状态条）

SHOTS = ROOT / "outputs" / "gui_click"


def _dpi_aware() -> None:
    """整个进程统一到**物理像素**。

    不声明的话会混两套坐标：`GetClientRect` 给逻辑宽（1080），`ClientToScreen`
    给物理原点（416），一乘就偏 88px，点到的全是空白（2026-11-08 实测：
    算出来 (525,1275)，按钮其实在 (614,1304)）。
    """
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)   # PER_MONITOR_AWARE
    except Exception:                                     # noqa: BLE001
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:                                 # noqa: BLE001
            pass


def _desktop_ready() -> tuple[bool, str]:
    """桌面能不能接收鼠标/截图。锁屏、RDP 断开都会返回 False。"""
    try:
        import ctypes
        from ctypes import wintypes

        u = ctypes.windll.user32
        u.OpenInputDesktop.restype = wintypes.HANDLE
        desk = u.OpenInputDesktop(0, False, 0x0100)
        if not desk:
            return False, "打不开输入桌面（锁屏 / 无交互会话）"
        if not u.GetForegroundWindow():
            return False, "没有前台窗口（锁屏 / 会话已断开）"
        from PIL import ImageGrab

        ImageGrab.grab().close()
        return True, "可交互"
    except Exception as exc:                              # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def _windows(desk, pattern: str):
    """按标题找**内容窗口**（Tk 根窗与门厅同标题，只取宽 > 400 的那个）。"""
    out = []
    for w in desk.windows(title_re=pattern):
        try:
            if w.is_visible() and w.rectangle().width() > 400:
                out.append(w)
        except Exception:                                 # noqa: BLE001
            continue
    return out


def _preview_log_stamp() -> tuple[float, int]:
    """相机预览日志的 (mtime, size)：用来证明「预览这台」真的被点过。"""
    p = ROOT / "outputs" / "logs" / "camera_preview.log"
    try:
        st = p.stat()
        return (st.st_mtime, st.st_size)
    except OSError:
        return (0.0, 0)


def _ensure_foreground(win) -> None:
    """点之前**确认目标窗口真的在前台**，否则立刻失败，不许"点完"了事。

    2026-11-08 15:45 现场：工具报「✔ 全流程点完」，但 4 张截图里有 3 张
    是 WorkBuddy 窗口、且 `camera_preview.log` 的 mtime 停在点击前 ——
    点击全落在别的窗口上了。`SetWindowPos(HWND_TOPMOST)` 只改 Z 序，
    **不保证前台**；而 Tk 的窗口被压在下面时，按客户区算出的绝对坐标
    会点中上面那个窗口。这是 §7.3 最不能接受的一种"假通过"。
    """
    import ctypes
    from ctypes import wintypes

    hwnd = win.handle
    u = ctypes.windll.user32
    u.ShowWindow(hwnd, 9)                  # SW_RESTORE
    u.BringWindowToTop(hwnd)
    u.SetForegroundWindow(hwnd)
    time.sleep(0.3)
    fg = u.GetForegroundWindow()
    if not fg:
        pass
    else:
        root = u.GetAncestor(fg, 2)        # GA_ROOT
        if root == u.GetAncestor(hwnd, 2):
            return
    print("✘ 目标窗口没能切到前台（很可能被其它置顶窗口挡住）：本次 **没有点过**，"
          "别把这次结果当成通过。", flush=True)
    raise SystemExit(6)


def _point_on_window(hwnd, x: int, y: int) -> bool:
    """该**屏幕坐标**下的窗口是不是目标窗口？

    `_ensure_foreground` 只保证「窗口在前台」，不保证「这个坐标在窗口上」：
    被置顶窗口压住、或比例坐标算到窗口外时，前台核对照样通过，鼠标却点在
    别处（11-08 抓到过一次"✔ 点完"其实全点在 WorkBuddy 上）。
    Tk 控件是自绘、没有独立 HWND，所以客户区内 `WindowFromPoint` 返回的就是
    根窗口；取 `GA_ROOT` 再比一次，能同时覆盖这两类假点击。
    """
    import ctypes
    from ctypes import wintypes

    u = ctypes.windll.user32
    u.WindowFromPoint.argtypes = [wintypes.POINT]
    u.WindowFromPoint.restype = wintypes.HWND
    u.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
    u.GetAncestor.restype = wintypes.HWND
    hit = u.WindowFromPoint(wintypes.POINT(int(x), int(y)))
    if not hit:
        return False
    return (u.GetAncestor(hit, 2) == u.GetAncestor(hwnd, 2))


def _require_point_on_window(hwnd, x: int, y: int, tag: str) -> None:
    """坐标不在目标窗口上 → 立刻失败，**绝不假装点过**（退出码 8）。"""
    if _point_on_window(hwnd, x, y):
        return
    print(f"✘ 「{tag}」算出的坐标 ({x},{y}) 不在目标窗口上"
          "（比例坐标与窗口尺寸对不上，或被别的窗口盖住）。本次 **没有点过**。",
          flush=True)
    raise SystemExit(8)


def _click(win, frac: tuple[float, float], tag: str) -> None:
    """在该窗口客户区按比例点一下（真实鼠标事件）。"""
    _ensure_foreground(win)
    import ctypes
    from ctypes import wintypes
    from pywinauto import mouse

    hwnd = win.handle
    u = ctypes.windll.user32
    # ⚠️ 必须用**客户区**尺寸算比例：窗口 rect 比客户区高一个标题栏（~30px），
    # 拿 rect 算 y 会整体偏上，点到的就是卡片空白（2026-11-08 第一次跑踩过）。
    client = wintypes.RECT()
    u.GetClientRect(hwnd, ctypes.byref(client))
    pt = wintypes.POINT(0, 0)
    u.ClientToScreen(hwnd, ctypes.byref(pt))
    x = pt.x + int(frac[0] * client.right)
    y = pt.y + int(frac[1] * client.bottom)
    _require_point_on_window(hwnd, x, y, tag)
    mouse.click(button="left", coords=(x, y))
    time.sleep(0.5)
    print(f"  点击 {tag} @ ({x},{y})", flush=True)


def _click_bottom(win, x_px: int, up_px: int, tag: str) -> None:
    """按"距客户区底部多少像素"点一下。

    底栏按钮（「预览这台」等）不能用**比例**定位：窗口高度每次不同，
    多出来的空白全在列表里，按钮始终贴着底部 —— 用比例会随窗口高度漂
    （2026-11-08 实测：同一条比例，一次点到列表空白、一次点到状态条）。
    """
    _ensure_foreground(win)
    import ctypes
    from ctypes import wintypes
    from pywinauto import mouse

    hwnd = win.handle
    u = ctypes.windll.user32
    client = wintypes.RECT()
    u.GetClientRect(hwnd, ctypes.byref(client))
    pt = wintypes.POINT(0, 0)
    u.ClientToScreen(hwnd, ctypes.byref(pt))
    x = pt.x + int(x_px)
    y = pt.y + int(client.bottom) - int(up_px)
    _require_point_on_window(hwnd, x, y, tag)
    mouse.click(button="left", coords=(x, y))
    time.sleep(0.5)
    print(f"  点击 {tag} @ ({x},{y})", flush=True)


def _shot(name: str) -> str:
    """截图；桌面中途锁定时如实返回失败，不让整条走查崩在取证上。"""
    from PIL import ImageGrab

    SHOTS.mkdir(parents=True, exist_ok=True)
    path = SHOTS / f"{name}.png"
    try:
        ImageGrab.grab().save(path)
        return str(path)
    except OSError as exc:
        return f"(截图失败：{exc}；桌面可能已锁定)"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="真实点击走查相机预览")
    ap.add_argument("--dry-run", action="store_true", help="只检查环境")
    ap.add_argument("--preview-up", type=int, default=210,
                    help="「预览这台」距客户区底部多少像素（窗口大小不同会变；"
                         "点空了就按 preview_*.png 量一下再传）")
    args = ap.parse_args(argv)
    _dpi_aware()

    ok, why = _desktop_ready()
    print(f"桌面状态：{'可交互' if ok else '不可用'}（{why}）", flush=True)
    if not ok:
        print("本次**没有**执行任何点击。请解锁桌面（或接上显示器会话）后重跑。",
              flush=True)
        return 2

    try:
        from pywinauto import Desktop
    except ImportError:
        print("缺 pywinauto：conda install -n myenv -c conda-forge pywinauto",
              flush=True)
        return 3
    if args.dry_run:
        print("库与桌面都就绪；去掉 --dry-run 就会真的点。", flush=True)
        return 0

    pyw = Path(sys.executable).with_name("pythonw.exe")
    launcher = pyw if pyw.is_file() else Path(sys.executable)
    print(f"启动：{launcher.name} apps/gui_main.py", flush=True)
    subprocess.Popen([str(launcher), "apps/gui_main.py"], cwd=str(ROOT),
                     creationflags=0x8)
    time.sleep(9)

    desk = Desktop(backend="win32")
    hubs = _windows(desk, ".*球形物体三维运动轨迹识别.*")
    if not hubs:
        print("✘ 主窗口没出现", flush=True)
        return 4
    hub = hubs[0]
    ctypes.windll.user32.SetWindowPos(hub.handle, -1, 0, 0, 0, 0, 0x0001 | 0x0002)
    # 门厅按当前装置档案的模式禁用不适用那张卡。硬点「平行双目」在
    # active=大角度档案时必然点空（2026-11-08 实测 exit 5）。按档案选可达的那张。
    from stereo import profiles
    _mode = str(getattr(profiles.get_active(), "stereo_mode", "") or "")
    angled = (_mode == "angled")
    frac, tag, title_re = ((HUB_ANGLED, "大角度双目·进入", ".*大角度双目.*")
                           if angled else
                           (HUB_PARALLEL, "平行双目·进入", ".*平行双目.*"))
    print(f"① 门厅 →「{tag}」（当前档案模式 {_mode or '未设置'}）", flush=True)
    _click(hub, frac, tag)
    time.sleep(2.5)

    sts = _windows(desk, title_re)
    if not sts:
        print(f"✘ 点了「{tag}」但没有出现双目窗", flush=True)
        # 静态查明（2026-11-09）：门厅卡片贴顶固定高度，比例坐标只在量坐标
        # 时的那个窗口尺寸下成立。把现场尺寸打出来，下次一眼能看出对没对上。
        try:
            _r = hub.rectangle()
            print(f"  诊断：门厅 rect={_r}，本次用的比例={frac}；"
                  f"校准截图是 996x865（outputs/gui_click/calibrate_hub.png）。"
                  "尺寸不同就会点空 —— 需固定窗口尺寸后重量绝对像素。", flush=True)
        except Exception:                                  # noqa: BLE001
            pass
        print(f"  现场截图：{_shot('fail_no_stereo')}", flush=True)
        return 5
    st = sts[0]
    ctypes.windll.user32.SetWindowPos(st.handle, -1, 0, 0, 0, 0, 0x0001 | 0x0002)
    time.sleep(0.5)
    print("② 双目 ① →「换相机…」", flush=True)
    _click(st, STEREO_CAMERAS, "换相机…")
    time.sleep(2.5)

    cams = _windows(desk, ".*相机管理.*")
    if not cams:
        print("✘ 点了「换相机…」但没有出现相机管理", flush=True)
        print(f"  现场截图：{_shot('fail_no_cameras')}", flush=True)
        return 6
    cam = cams[0]
    ctypes.windll.user32.SetWindowPos(cam.handle, -1, 0, 0, 0, 0, 0x0001 | 0x0002)
    time.sleep(0.5)
    print("③ 相机管理 →「预览这台」", flush=True)
    before = _preview_log_stamp()
    _click_bottom(cam, 216, args.preview_up, "预览这台")
    prev = 0
    for sec in (3, 8, 15, 25):
        time.sleep(sec - prev)
        prev = sec
        print(f"  等待 {sec}s → {_shot(f'preview_{sec}s')}", flush=True)
    if _preview_log_stamp() == before:
        print("✘ 判失败：camera_preview.log 在点击前后**没有变化** —— "
              "没有证据表明「预览这台」真的被点过（可能点到了别的窗口）。"
              "截图只作证据、不作通过依据。", flush=True)
        return 7
    print("✔ 全流程点完，且预览日志有新增。检查 preview_25s.png：", flush=True)
    print("  · 右侧出现卡片且有画面 → 通", flush=True)
    print("  · 右侧仍空白、状态条还是「相机管理已打开」→ 最后一下没点中，"
          "量一下 preview_25s.png 里按钮距窗口底部的像素，用 "
          "--preview-up 传进来重跑", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
