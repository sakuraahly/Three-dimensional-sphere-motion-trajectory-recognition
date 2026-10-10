"""球形物体三维运动轨迹识别 —— 命令行彩色启动器（人类友好型）。

与图形版（启动程序.py）功能一致，但没有窗口，适合：
  - 无图形界面 / 远程终端 / 服务器环境；
  - 喜欢键盘操作、想看更详细输出的使用者。

交互约定：两台相机左右的定义 —— **人面对镜头时，左手边那台为"左"**。

运行：
    C:\\Users\\<USER>\\miniforge3\\envs\\myenv\\python.exe 启动程序_命令行.py
    （或直接双击 启动程序_命令行.bat）
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
while not (ROOT / "apps").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PY = sys.executable


# --------------------------------------------------------------------------
class C:
    """ANSI 颜色（Windows 下由 enable_ansi() 打开虚拟终端支持）。"""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"


def enable_ansi() -> None:
    """在 Windows 控制台开启 ANSI 转义序列（否则会打印出一堆乱码）。"""
    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)          # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)   # ENABLE_VT
    except Exception:
        pass


def clear() -> None:
    os.system("cls" if os.name == "nt" else "clear")


# --------------------------------------------------------------------------
CFG: dict[str, str] = {
    "camera": "auto",            # auto = 启动时自动挑最优（优先外接、避开内置/红外、非全黑）
    "diameter": "0.04",
    "method": "fusion",
    "preset": "不使用",
    "horizon": "0.6",
    "focal": "",
    "calib_left": "left",        # 角色名：按设备身份绑定，换 USB 口也不漂
    "calib_right": "right",      # 也可填具体编号（如 0 / 2），但换接口可能指向别的相机
    "resolution": "1920x1080",
    "fps": "60",
}


def banner() -> None:
    print(f"{C.CYAN}{'=' * 62}{C.RESET}")
    print(f"{C.BOLD}{C.WHITE}  球形物体三维运动轨迹识别 · 启动器（命令行版）{C.RESET}")
    print(f"{C.DIM}  检测 → 跟踪 → 三维定位 → 轨迹预测{C.RESET}")
    print(f"{C.CYAN}{'=' * 62}{C.RESET}")


def show_config() -> None:
    print(f"{C.BOLD}当前参数{C.RESET}  {C.DIM}(选 7 可修改){C.RESET}")
    rows = [
        ("摄像头编号", CFG["camera"]),
        ("球直径(米)", CFG["diameter"]),
        ("检测方法", CFG["method"]),
        ("颜色预设", CFG["preset"]),
        ("预测视野(秒)", CFG["horizon"]),
        ("焦距fx", CFG["focal"] or "自动"),
        ("标定 左/右相机", f'{CFG["calib_left"]} / {CFG["calib_right"]}'),
        ("两路统一分辨率", CFG["resolution"]),
        ("帧率", CFG["fps"]),
    ]
    for k, v in rows:
        print(f"  {C.DIM}{k:<16}{C.RESET}{C.WHITE}{v}{C.RESET}")
    print(f"  {C.DIM}{'说明':<16}{C.RESET}{C.YELLOW}"
          f"auto = 自动挑最优；left/right = 按身份绑定的角色名（推荐，换 USB 口也不漂）{C.RESET}")
    print()


def menu() -> str:
    items = [
        ("1", "实时摄像头检测", C.GREEN),
        ("2", "离线视频分析", C.GREEN),
        ("3", "标定前置检查（两路是否都看到棋盘）", C.MAGENTA),
        ("4", "双目标定（两台相机 + 棋盘格）", C.MAGENTA),
        ("5", "标定复检（装置是否被碰动）", C.MAGENTA),
        ("6", "相机设备管理（识别 / 固定 / 检查漂移）", C.BLUE),
        ("7", "修改参数", C.YELLOW),
        ("8", "相机探测（核对两路参数是否一致）", C.BLUE),
        ("9", "环境自测（无需摄像头）", C.BLUE),
        ("0", "退出", C.RED),
    ]
    print(f"{C.BOLD}请选择操作：{C.RESET}")
    for key, label, color in items:
        print(f"  {color}[{key}]{C.RESET} {label}")
    print()
    return input(f"{C.CYAN}输入编号并回车 > {C.RESET}").strip()


def common_args() -> list[str]:
    args: list[str] = []
    if CFG["preset"] and CFG["preset"] != "不使用":
        args += ["--preset", CFG["preset"]]
    if CFG["diameter"]:
        args += ["--ball-diameter", CFG["diameter"]]
    if CFG["horizon"]:
        args += ["--horizon", CFG["horizon"]]
    if CFG["focal"]:
        args += ["--focal", CFG["focal"]]
    return args


def run(cmd: list[str], title: str) -> None:
    print()
    print(f"{C.CYAN}▶ {title}{C.RESET}")
    print(f"{C.DIM}> " + " ".join(cmd) + C.RESET)
    print()
    try:
        code = subprocess.call(cmd, cwd=str(ROOT))
    except KeyboardInterrupt:
        print(f"\n{C.YELLOW}已中断{C.RESET}")
        code = -1
    if code == 0:
        print(f"\n{C.GREEN}✔ 完成（返回码 0）{C.RESET}")
    else:
        print(f"\n{C.RED}✘ 结束（返回码 {code}）{C.RESET}")
    input(f"{C.DIM}按回车返回主菜单…{C.RESET}")


# --------------------------------------------------------------------------
def ask(prompt: str, current: str) -> str:
    value = input(f"{prompt} {C.DIM}[{current}]{C.RESET}: ").strip()
    return value or current


def edit_config() -> None:
    clear()
    banner()
    print(f"{C.BOLD}修改参数{C.RESET}  {C.DIM}(直接回车 = 保持原值){C.RESET}\n")
    print(f"{C.DIM}摄像头编号可填 auto（自动挑最优）/ 具体编号；"
          f"左/右建议填角色名 left / right（按身份绑定，换 USB 口也不漂）{C.RESET}")
    CFG["camera"] = ask("摄像头编号 (auto/0/1/2)", CFG["camera"])
    CFG["diameter"] = ask("球直径(米)", CFG["diameter"])
    print(f"{C.DIM}检测方法可选：fusion / color / hough / motion / yolo / yolo_sam{C.RESET}")
    CFG["method"] = ask("检测方法", CFG["method"])
    print(f"{C.DIM}颜色预设可选：不使用 / white_ball / orange_ball / green_ball / "
          f"blue_ball / red_ball / yellow_ball{C.RESET}")
    CFG["preset"] = ask("颜色预设", CFG["preset"])
    CFG["horizon"] = ask("预测视野(秒)", CFG["horizon"])
    CFG["focal"] = ask("焦距fx（留空=自动）", CFG["focal"])
    print(f"{C.DIM}相机的编号会随插口漂移；写 left / right 角色名最稳"
          f"（需先跑『相机设备管理』固定）{C.RESET}")
    CFG["calib_left"] = ask("标定用左相机 (left/编号)", CFG["calib_left"])
    CFG["calib_right"] = ask("标定用右相机 (right/编号)", CFG["calib_right"])
    print(f"{C.DIM}分辨率可选：1920x1080 / 1280x720 / 960x540 / 640x480{C.RESET}")
    CFG["resolution"] = ask("两路统一分辨率", CFG["resolution"])
    CFG["fps"] = ask("帧率", CFG["fps"])
    print(f"\n{C.GREEN}参数已更新{C.RESET}")
    input(f"{C.DIM}按回车返回主菜单…{C.RESET}")


def action_realtime() -> None:
    cmd = [PY, str(ROOT / "apps" / "run_realtime.py"),
           "--camera", CFG["camera"], "--method", CFG["method"]]
    cfg = ROOT / "config" / "sphere_config.json"
    if cfg.exists():
        cmd += ["--config", str(cfg)]
    cmd += common_args()
    run(cmd, "实时摄像头检测")


def action_video() -> None:
    path = input(f"{C.CYAN}请输入视频文件路径 > {C.RESET}").strip().strip('"')
    if not path or not Path(path).exists():
        print(f"{C.RED}文件不存在：{path}{C.RESET}")
        input(f"{C.DIM}按回车返回主菜单…{C.RESET}")
        return
    stamp = Path(path).stem
    out = ROOT / "outputs" / "videos" / f"{stamp}_annotated.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [PY, str(ROOT / "apps" / "run_video.py"), path,
           "--out", str(out),
           "--csv", str(ROOT / "outputs" / f"{stamp}_traj.csv"),
           "--method", CFG["method"]]
    cmd += common_args()
    run(cmd, "离线视频分析")


def action_calibrate() -> None:
    if not confirm_stereo_cameras():
        return
    width, height = CFG["resolution"].split("x")
    cmd = [PY, str(ROOT / "apps" / "calibrate_stereo.py"),
           "--left", CFG["calib_left"], "--right", CFG["calib_right"],
           "--cols", "8", "--rows", "5", "--square", "25",
           "--width", width, "--height", height, "--fps", CFG["fps"]]
    run(cmd, f'双目标定（左{CFG["calib_left"]} / 右{CFG["calib_right"]}，收满约 25 对后自动标定）')


def action_verify() -> None:
    calib = ROOT / "config" / "stereo_calibration"
    if not calib.with_suffix(".npz").exists():
        print(f"{C.RED}找不到标定文件，请先执行一次双目标定。{C.RESET}")
        input(f"{C.DIM}按回车返回主菜单…{C.RESET}")
        return
    if not confirm_stereo_cameras():
        return
    width, height = CFG["resolution"].split("x")
    cmd = [PY, str(ROOT / "apps" / "verify_stereo.py"),
           "--calib", str(calib),
           "--left", CFG["calib_left"], "--right", CFG["calib_right"],
           "--width", width, "--height", height, "--fps", CFG["fps"]]
    run(cmd, "标定复检（判断装置是否被碰动）")


def confirm_stereo_cameras() -> bool:
    """左/右选到内置或红外相机时确认一次：这种相机没法刚性固定在横梁上，标定会白做。

    解析失败（角色还没固定、枚举不可用等）就静默放行，不挡用户的路。
    """
    try:
        from sphere_tracker import camera_registry as cr
        devices = cr.list_devices()
    except Exception:
        return True
    by_order = {d.order: d for d in devices}
    for role, key in (("左", "calib_left"), ("右", "calib_right")):
        try:
            index = cr.resolve_camera_arg(CFG[key], interactive=False)
        except RuntimeError:
            continue                                     # 角色没固定/对不上：不拦
        if index is None:
            continue
        entry = by_order.get(index)
        name = entry.name if entry else ""
        if cr.is_avoided(name):
            print(f"{C.YELLOW}提示：{role}相机（编号 {index}，{name or '未知名'}）看起来是"
                  f"笔记本内置 / 红外相机，不适合双目（没法固定在同一根横梁上，标定会白做）。"
                  f"双目请用外接相机。{C.RESET}")
            return input(f"{C.CYAN}仍要继续？(y/N) > {C.RESET}").strip().lower() in ("y", "yes")
    return True


def action_precheck() -> None:
    if not confirm_stereo_cameras():
        return
    width, height = CFG["resolution"].split("x")
    cmd = [PY, str(ROOT / "apps" / "tools" / "check_stereo_setup.py"),
           "--left", CFG["calib_left"], "--right", CFG["calib_right"],
           "--width", width, "--height", height, "--fps", CFG["fps"]]
    run(cmd, f'标定前置检查（左{CFG["calib_left"]} / 右{CFG["calib_right"]}：'
             f'两路是否都看到棋盘）')


def action_probe() -> None:
    run([PY, str(ROOT / "apps" / "tools" / "list_cameras.py"), "--max-index", "3"],
        "相机探测（核对两路分辨率 / 帧率是否一致）")


def action_cameras() -> None:
    """相机设备管理子菜单（索引会漂移，身份不会）。"""
    clear()
    banner()
    print(f"{C.BOLD}相机设备管理{C.RESET}  {C.DIM}（索引会漂移，身份不会）{C.RESET}\n")
    print(f"  {C.CYAN}[1]{C.RESET} 列出设备（不开相机，任何机器都能跑）")
    print(f"  {C.CYAN}[2]{C.RESET} 列出设备 + 逐个试开（看真实分辨率 / 是否全黑）")
    print(f"  {C.CYAN}[3]{C.RESET} 识别并固定：逐个显示画面，确认哪台是左 / 右 / 主")
    print(f"  {C.CYAN}[4]{C.RESET} 检查映射：重插相机后比对有没有漂移 / 被占用")
    print()
    pick = input(f"{C.CYAN}输入编号并回车 > {C.RESET}").strip()
    cmds = {
        "1": (["list"], "列出相机设备"),
        "2": (["list", "--probe"], "列出相机设备（含实测分辨率）"),
        "3": (["identify"], "识别并固定相机角色（结果写入 config/cameras.json）"),
        "4": (["check"], "检查相机映射是否漂移"),
    }
    if pick not in cmds:
        return
    sub, title = cmds[pick]
    run([PY, str(ROOT / "apps" / "tools" / "cameras.py"), *sub], title)


def action_selftest() -> None:
    run([PY, str(ROOT / "tests" / "selftest_synthetic.py")],
        "环境自测（无需摄像头，约 1 分钟）")


ACTIONS = {
    "1": action_realtime,
    "2": action_video,
    "3": action_precheck,
    "4": action_calibrate,
    "5": action_verify,
    "6": action_cameras,
    "8": action_probe,
    "9": action_selftest,
}


def main() -> int:
    enable_ansi()
    while True:
        clear()
        banner()
        show_config()
        choice = menu()
        if choice == "0":
            print(f"{C.CYAN}再见。{C.RESET}")
            return 0
        if choice == "7":
            edit_config()
            continue
        action = ACTIONS.get(choice)
        if action is None:
            print(f"{C.RED}无效输入，请重试。{C.RESET}")
            input(f"{C.DIM}按回车继续…{C.RESET}")
            continue
        try:
            action()
        except KeyboardInterrupt:
            print(f"\n{C.YELLOW}已取消。{C.RESET}")
            input(f"{C.DIM}按回车返回主菜单…{C.RESET}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n已退出。")
