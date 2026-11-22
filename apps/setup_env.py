"""一键环境安装器：建好隔离环境、装齐依赖，并把它登记给所有启动入口。

为什么不能继续"pip install 到当前 Python"
----------------------------------------
`run.bat` 原来的兜底是：随便找一台能跑的 Python，直接把
`requirements.txt` 装进去。三处硬伤：

1. 装进用户级 conda / 系统 Python，跟别的项目互相踩。本项目自己就踩过
   `opencv-python` 与 `opencv-contrib-python` 抢同一个 `cv2` 模块的坑；
2. 换台机器、换个人、换个盘，能不能跑全靠"碰运气看谁装过 cv2"；
3. 装完不留凭据，每次启动都要把 8 个候选解释器重新跑一遍 `import cv2`。

现在的项目内布局::

    .venv/                      隔离环境（默认全忽略，删了能重建）
    config/python_env.txt       环境指针（本机状态，不入库）
    logs/misc/<月>/setup_env_<时间戳>.log   安装日志（形状同 paths.log_dir("misc")）

指针是**纯文本 key=value**，不是 JSON：`.bat` 用 `for /f`、Go 用按行切分、
Python 用 `partition("=")`，三方都不需要"先有一个能跑的 Python 才能读它"
—— 这恰恰是"环境还没装好"那一刻的约束。项目内的解释器写**相对路径**，
整个仓库拷走后仍指向同一处；`--adopt` 登记的仓库外环境才写绝对路径。

默认策略（省流量、不打断已有的 AI 后端）
--------------------------------------
1. 指针已存在且验证通过 → 什么都不做；
2. 本机已有装齐核心依赖的 Python（例如 conda `myenv`）→ **登记它**，零下载。
   这台机器上的 torch / ultralytics 不会被一个空的 .venv 挡住；
3. 都没有 → 新建项目内 `.venv` 并装依赖。

要强制完全隔离，用 `--isolate`；要重建，用 `--force`。

用法::

    python apps/setup_env.py                    自动配置（复用优先，缺失才建）
    python apps/setup_env.py --verify          只体检，不改任何东西
    python apps/setup_env.py --isolate         强制新建项目内 .venv
    python apps/setup_env.py --adopt [PYTHON]  登记指定/现有的环境，不下载
    python apps/setup_env.py --ai              额外装 YOLO/SAM 组（需 CUDA）
    python apps/setup_env.py --print-python    只输出解释器绝对路径（给 bat 用）

双击入口：`install.bat`（分发版的环境安装器）、`launcher.exe install`，
或 `run.bat install`（参数路由的唯一产地，前两者都转发到这里）。
根目录只放英文名的运行 / 安装 / 卸载脚本，见 AGENTS.md 规则 5。

本文件只用标准库：运行它的时候，环境本来就可能还没装好。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "sphere_tracker").is_dir())
DEFAULT_VENV = ROOT / ".venv"
POINTER = ROOT / "config" / "python_env.txt"
REQUIREMENTS = ROOT / "requirements.txt"
MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"

#: 核心依赖对应的 import 名。`tkinter` 单独判：Windows 图形界面必需，
#: 但某些无头 Linux 根本不带它，不能因此把整个环境判死。
CORE_IMPORTS = ("cv2", "numpy", "matplotlib", "PIL", "yaml")
ALL_IMPORTS = (*CORE_IMPORTS, "tkinter")

PROBE = r'''
import importlib, json, sys
mods = ["cv2", "numpy", "matplotlib", "PIL", "yaml", "tkinter", "scipy"]
out = {}
for m in mods:
    try:
        mod = importlib.import_module(m)
        out[m] = getattr(mod, "__version__", "ok")
    except Exception:
        out[m] = None
print("__ENV_PROBE__" + json.dumps({
    "version": ".".join(map(str, sys.version_info[:3])), "mods": out,
    "executable": sys.executable}))
'''

def setup_log_path(when: datetime | None = None) -> Path:
    """安装日志路径，形状与 `sphere_tracker.paths.log_dir("misc")` 一致。

    这里**不能** import `sphere_tracker.paths`：那个包的 `__init__` 会拉起
    pipeline → cv2 / numpy，而安装器跑的时候它们很可能还没装好。所以按同一
    套规则自己拼一份；`selftest_launcher_params` 会用真正的 `log_dir("misc")`
    比对形状，将来分片粒度改了会当场判红，不会悄悄漂移。
    """
    w = when or datetime.now()
    return (ROOT / "logs" / "misc" / w.strftime("%Y-%m")
            / f"setup_env_{w:%Y%m%d_%H%M%S}.log")


LOG_PATH = setup_log_path()
_LOG_LINES: list[str] = []


def log(msg: str, *, echo: bool = True) -> None:
    """同时进控制台和安装日志。日志写不出来绝不影响安装本身。"""
    _LOG_LINES.append(msg)
    if echo:
        print(msg, flush=True)


def flush_log() -> None:
    if not _LOG_LINES:
        return
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} =====\n")
            fh.write("\n".join(_LOG_LINES) + "\n")
        _LOG_LINES.clear()
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# 环境指针：纯文本，Python / bat / Go 三方都能读
# --------------------------------------------------------------------------- #
def parse_pointer(text: str, root: Path) -> Path | None:
    """从指针文本里解出解释器绝对路径；相对值按 ``root`` 还原。"""
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() == "python" and value.strip():
            p = Path(value.strip())
            return p if p.is_absolute() else (root / p).resolve()
    return None


def pointer_value(py: Path, root: Path) -> str:
    """仓库内写相对路径，仓库外才写绝对路径（换盘/换机器仍指对地方）。"""
    resolved = py.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def read_pointer() -> Path | None:
    try:
        return parse_pointer(POINTER.read_text(encoding="utf-8"), ROOT)
    except (OSError, UnicodeError):
        return None


def write_pointer(py: Path, kind: str, data: dict) -> str:
    value = pointer_value(py, ROOT)
    verified = [m for m, v in data.get("mods", {}).items() if v is not None]
    text = "\n".join([
        "# sphere-python-env v1 - generated by apps/setup_env.py; do not edit",
        f"python={value}",
        f"kind={kind}",
        f"created={datetime.now().isoformat(timespec='seconds')}",
        f"python_version={data.get('version', '?')}",
        "verified=" + ",".join(verified),
        "",
    ])
    POINTER.parent.mkdir(parents=True, exist_ok=True)
    POINTER.write_text(text, encoding="utf-8")
    return value


# --------------------------------------------------------------------------- #
# 子进程
# --------------------------------------------------------------------------- #
def run(cmd: list[str], *, quiet: bool = False) -> tuple[int, str]:
    """跑一条命令，逐行回显并留日志；不抛异常，返回码与合并后的输出。"""
    shown = " ".join(str(c) for c in cmd)
    log("$ " + shown, echo=not quiet)
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"          # 中文 pip 输出不按 GBK 炸
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    try:
        proc = subprocess.Popen(
            [str(c) for c in cmd], cwd=str(ROOT), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            errors="replace", env=env)
    except OSError as exc:
        log(f"  启动失败：{exc}", echo=not quiet)
        return 1, str(exc)
    assert proc.stdout is not None
    lines: list[str] = []
    for line in proc.stdout:
        line = line.rstrip("\r\n")
        lines.append(line)
        log("  " + line, echo=not quiet)
    rc = proc.wait()
    flush_log()                                 # 每步落盘，装到一半断电也不丢
    return rc, "\n".join(lines)


def check_imports(py: Path) -> dict:
    """真的在目标解释器里 import 一遍，不靠 pip list 猜。"""
    rc, text = run([str(py), "-c", PROBE], quiet=True)
    if rc != 0:
        log(f"  探针执行失败（rc={rc}）")
    for line in text.splitlines():
        if "__ENV_PROBE__" in line:
            try:
                return json.loads(line.split("__ENV_PROBE__", 1)[1])
            except (ValueError, IndexError):
                break
    return {"version": "?", "mods": {m: None for m in ALL_IMPORTS},
            "executable": str(py)}


def verify_report(data: dict) -> tuple[bool, list[str]]:
    """核心依赖必须齐；tkinter 只在 Windows 上算必需。

    scipy 是**推荐但非致命**依赖，故意不进 CORE_IMPORTS：缺了它 3D 处理
    不会崩，只是平滑/外参精化静默降级。所以这里判 True，另由
    `warn_if_scipy_missing()` 单独提醒，避免"环境就绪"掩盖精度损失。
    """
    mods = data.get("mods", {})
    missing = [m for m in CORE_IMPORTS if not mods.get(m)]
    if os.name == "nt" and not mods.get("tkinter"):
        missing.append("tkinter")
    return not missing, missing


def warn_if_scipy_missing(data: dict, emit=print) -> None:
    """缺 scipy 时给明确警告，但不改变"核心能跑"的判定。"""
    if not data.get("mods", {}).get("scipy"):
        emit("[警告] 缺 scipy：3D 轨迹平滑与外参精化会静默降级为"
             "滑动平均/跳过，建议 pip install scipy")


# --------------------------------------------------------------------------- #
# 找 Python / 建 venv / 装依赖
# --------------------------------------------------------------------------- #
def venv_python(base: Path | None = None) -> Path:
    base = base or DEFAULT_VENV
    return base / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def bootstrap_candidates() -> list[Path]:
    """与 `run.bat` / `launcher.exe` 同口径，再加当前解释器与已登记环境。"""
    out: list[Path] = []

    def add(p) -> None:
        if not p:
            return
        p = Path(p)
        if p not in out:
            out.append(p)

    add(read_pointer())
    add(sys.executable)
    for key in ("USERPROFILE", "LOCALAPPDATA", "ProgramData"):
        base = os.environ.get(key)
        if base:
            for distro in ("miniforge3", "anaconda3"):
                name = "python.exe" if os.name == "nt" else "python"
                add(Path(base) / distro / "envs" / "myenv" / name)
    for name in ("python", "python3", "py"):
        add(shutil.which(name))
    return out


def usable_bootstrap(py: Path) -> bool:
    if not py.is_file():
        return False
    rc, _ = run([str(py), "-c",
                 "import sys,venv; assert sys.version_info >= (3, 10)"], quiet=True)
    return rc == 0


def find_existing_env() -> Path | None:
    """找一台已经装齐核心依赖的 Python（用户现成的 conda 环境优先）。"""
    for py in bootstrap_candidates():
        if not py.is_file():
            continue
        data = check_imports(py)
        ok, _ = verify_report(data)
        if ok:
            return py
    return None


def create_venv(base: Path, force: bool) -> tuple[Path | None, str]:
    py = venv_python(base)
    if py.is_file() and not force:
        rc, _ = run([str(py), "-m", "pip", "--version"], quiet=True)
        if rc == 0:
            return py, "复用已有虚拟环境"
        log("  现有虚拟环境里 pip 不可用，准备重建")
    if base.exists():
        # 只删自己人：必须长得像 venv，避免 --venv 指错目录时误删数据。
        if not (base / "pyvenv.cfg").is_file():
            return None, f"{base} 已存在但不是虚拟环境（没有 pyvenv.cfg），拒绝删除"
        log(f"  删除旧虚拟环境：{base}")
        try:
            shutil.rmtree(base)
        except OSError as exc:
            return None, f"删不掉旧虚拟环境：{exc}"
    for boot in bootstrap_candidates():
        if not usable_bootstrap(boot):
            continue
        log(f"  用 {boot} 创建虚拟环境 …")
        rc, _ = run([str(boot), "-m", "venv", str(base)])
        if rc == 0 and venv_python(base).is_file():
            rc2, _ = run([str(venv_python(base)), "-m", "pip", "--version"],
                         quiet=True)
            if rc2 == 0:
                return venv_python(base), f"新建（引导 {boot.name}）"
            log("  这个引导建出来的环境没有 pip，换下一个")
        shutil.rmtree(base, ignore_errors=True)
    return None, "所有候选 Python 都建不出可用的虚拟环境（请安装 Python 3.10+）"


def pip_install(py: Path, args: list[str], *, official: bool = False,
                mirror: bool = True) -> bool:
    base = [str(py), "-m", "pip", "install", "--no-input",
            "--disable-pip-version-check"]
    if mirror and not official:
        rc, _ = run(base + ["-i", MIRROR, *args])
        if rc == 0:
            return True
        log("  镜像源失败，改用官方 PyPI 重试")
    rc, _ = run(base + args)
    return rc == 0


def install_ai(py: Path) -> bool:
    """AI 组按 `requirements-ai.txt` 的顺序装；失败只警告，不推翻核心环境。

    那份文件是**说明文档**（含 CUDA 版本、--no-deps 的先后约束），不是能
    直接 `-r` 的清单，所以顺序写在这里。torch 必须走它自己的 index，
    不能被镜像参数覆盖。
    """
    steps = (
        ["torch==2.11.0", "torchvision==0.26.0", "--index-url",
         "https://download.pytorch.org/whl/cu128"],
        ["ultralytics", "--no-deps"],
        ["pyyaml", "psutil", "polars", "timm", "lap"],
    )
    ok = True
    for args in steps:
        rc, _ = run([str(py), "-m", "pip", "install", "--no-input", *args])
        if rc != 0:
            log("  [警告] 这一步失败；AI 后端可能不可用，核心功能不受影响")
            ok = False
    return ok


# --------------------------------------------------------------------------- #
# 命令
# --------------------------------------------------------------------------- #
def cmd_print_python() -> int:
    py = read_pointer()
    if py and py.is_file():
        print(py)
        return 0
    return 1


def cmd_verify() -> int:
    py = read_pointer()
    if py is None:
        print("还没有登记运行环境。先执行：install.bat（或 run.bat install）")
        return 1
    if not py.is_file():
        print(f"登记的解释器不存在：{py}")
        print("虚拟环境可能被删了。重跑：install.bat")
        return 1
    data = check_imports(py)
    ok, missing = verify_report(data)
    print(f"解释器：{py}")
    print(f"Python ：{data.get('version', '?')}")
    for name, ver in data.get("mods", {}).items():
        print(f"  {name:<12}{ver or '缺失'}")
    if ok:
        print("判定结果：环境就绪")
    else:
        print(f"判定结果：缺 {', '.join(missing)}，重跑 install.bat")
    warn_if_scipy_missing(data, print)
    return 0 if ok else 1


def cmd_install(args: argparse.Namespace) -> int:
    base = Path(args.venv).expanduser().resolve() if args.venv else DEFAULT_VENV
    steps = [
        f"仓库根      {ROOT}",
        f"环境位置    {base}",
        f"依赖清单    {REQUIREMENTS.relative_to(ROOT).as_posix()}",
        f"AI 组       {'装' if args.ai else '不装（默认；要 YOLO/SAM 加 --ai）'}",
    ]
    if args.dry_run:
        print("计划（--dry-run，不改动任何东西）：")
        print("\n".join("  " + s for s in steps))
        return 0

    # --ai 是显式诉求：装不全却仍报"环境已就绪"是在骗用户。这里记账，
    # 收尾统一以非零码结束；核心环境本身不受影响，也不改核心缺依赖的判定。
    ai_failed = False

    def _install_ai(py: Path) -> None:
        nonlocal ai_failed
        if not install_ai(py):
            ai_failed = True
            log("[警告] AI 组没装全，核心功能不受影响；收尾会以非零码结束")

    def _finish(rc: int = 0) -> int:
        if ai_failed:
            log("[失败] AI 组未装全（核心环境不受影响）；可看日志重试")
            return 3
        return rc

    log("=" * 66)
    log("一键环境安装器")
    log("\n".join(steps))
    log("=" * 66)

    # ① 已经登记且能用 → 直接收工（AI 组仍可补装）
    if not args.isolate and not args.force:
        py = read_pointer()
        if py and py.is_file():
            data = check_imports(py)
            ok, _ = verify_report(data)
            if ok:
                log(f"环境已就绪：{py}")
                if not args.ai:
                    log("无需安装。要重建加 --force，要隔离环境加 --isolate。")
                    return 0
                _install_ai(py)
                return _finish()
            log("  已登记的环境缺依赖，继续修复")

    # ② 登记指定环境 / 复用本机现成环境（零下载）
    if args.adopt is not None:
        py = (Path(args.adopt).expanduser().resolve()
              if args.adopt != "auto" else find_existing_env())
        if py is None or not py.is_file():
            log("找不到可登记的环境。去掉 --adopt 让安装器新建一个。")
            return 1
        data = check_imports(py)
        ok, missing = verify_report(data)
        if not ok:
            log(f"这个环境缺 {', '.join(missing)}，不登记（免得启动器认了个坏环境）")
            return 1
        if args.ai:
            _install_ai(py)
        value = ("" if args.no_pointer else
                 write_pointer(py, "adopted", data))
        log(f"已登记现成环境：{py}" + (f"（指针 {value}）" if value else ""))
        return _finish()

    if not args.isolate:
        existing = find_existing_env()
        if existing is not None:
            data = check_imports(existing)
            if args.ai:
                _install_ai(existing)
            value = ("" if args.no_pointer else
                     write_pointer(existing, "adopted", data))
            log(f"本机已有装齐依赖的环境，直接复用（零下载）：{existing}")
            log("要一份完全隔离的环境，加 --isolate。")
            return _finish()

    # ③ 新建虚拟环境 + 装核心依赖
    py, why = create_venv(base, args.force)
    if py is None:
        log(f"[失败] {why}")
        return 1
    log(f"虚拟环境就绪（{why}）：{py}")
    if not pip_install(py, ["-r", REQUIREMENTS.relative_to(ROOT).as_posix()],
                       official=args.official):
        log("[失败] 依赖安装失败，看上面的报错或日志：" + str(LOG_PATH))
        return 1
    if args.ai:
        _install_ai(py)

    data = check_imports(py)
    ok, missing = verify_report(data)
    if not ok:
        log(f"[失败] 装完仍缺 {', '.join(missing)} —— 不写入环境指针，"
            "免得启动器认了个坏环境")
        return 1
    value = "" if args.no_pointer else write_pointer(py, "venv", data)
    log("依赖验证通过：")
    for name, ver in data.get("mods", {}).items():
        log(f"  {name:<12}{ver or '缺失'}")
    warn_if_scipy_missing(data, log)
    if value:
        log(f"已写入环境指针：{POINTER.relative_to(ROOT).as_posix()} -> {value}")
    log("完成。双击 launcher.exe 或运行 run.bat 即可启动。")
    log(f"安装日志：{LOG_PATH.relative_to(ROOT).as_posix()}")
    return _finish()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="一键配置项目运行环境（隔离 venv / 复用现成环境 / 体检）")
    ap.add_argument("--verify", action="store_true",
                    help="只体检已登记的环境，不改任何东西")
    ap.add_argument("--adopt", nargs="?", const="auto", metavar="PYTHON",
                    help="登记现成的 Python（不写路径则自动找），不下载依赖")
    ap.add_argument("--isolate", action="store_true",
                    help="不复用现成环境，强制新建项目内虚拟环境")
    ap.add_argument("--ai", action="store_true",
                    help="额外安装 YOLO/SAM 组（需 CUDA，默认不装）")
    ap.add_argument("--force", action="store_true", help="重建虚拟环境")
    ap.add_argument("--venv", metavar="DIR",
                    help="虚拟环境目录（默认项目内 .venv）")
    ap.add_argument("--official", action="store_true",
                    help="直接用官方 PyPI，不走镜像")
    ap.add_argument("--no-pointer", action="store_true",
                    help="不写环境指针（给测试 / 临时环境用）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印将要做的事，不创建任何文件")
    ap.add_argument("--print-python", action="store_true",
                    help="只输出已登记解释器的绝对路径，供启动脚本读取")
    args = ap.parse_args(argv)
    try:
        if args.print_python:
            return cmd_print_python()
        if args.verify:
            return cmd_verify()
        return cmd_install(args)
    finally:
        flush_log()


if __name__ == "__main__":
    raise SystemExit(main())
