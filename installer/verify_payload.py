"""安装程序 payload 自检：不依赖 Inno Setup，先把"要装的东西齐不齐"验完。

为什么需要它（2026-11-22）：
  编译 setup.exe 的前置是 Inno Setup，但那台机器可能没装。
  把"payload 完整性"从编译步骤里拆出来，就能在装 Inno 之前先把
  最容易错的部分（纯净包漏文件 / launcher.exe 没重建 / 脚本引用的图标不存在）
  验掉 —— 装好 Inno 后编译一次就过，不用来回试。

用法::

    python installer/verify_payload.py

它会检查：
  1. staging 目录存在且含关键入口（run.bat / install.bat / apps/gui_main.py / launcher.exe）；
  2. launcher.exe 是合法 PE（MZ 头 + PE 偏移），不是 0 字节占位；
  3. .iss 里 Source/SetupIconFile 引用的相对路径在 staging 里都能找到；
  4. .iss 的 AppId 是合法 GUID（Inno 对格式敏感，写错编译期才报太晚）。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGE = ROOT / "outputs" / "release_stage"
ISS = ROOT / "installer" / "sphere_motion.iss"

REQUIRED = (
    "run.bat", "install.bat", "uninstall.bat", "launcher.exe",
    "apps/gui_main.py", "apps/cli_main.py", "README.md",
    "requirements.txt", "assets/boards/chessboard_8x5_20mm_a4-portrait.png",
)


def main() -> int:
    problems: list[str] = []

    if not STAGE.is_dir():
        return _fail([f"找不到 staging 目录：{STAGE}（先跑 installer\\build_installer.bat 的 [1/3][2/3]）"])

    for rel in REQUIRED:
        if not (STAGE / rel).is_file():
            problems.append(f"payload 缺关键文件：{rel}")

    exe = STAGE / "launcher.exe"
    if exe.is_file():
        head = exe.read_bytes()[:64]
        if head[:2] != b"MZ":
            problems.append("launcher.exe 不是 PE 文件（没有 MZ 头）")
        else:
            pe = int.from_bytes(head[60:64], "little")
            if not (0 < pe < exe.stat().st_size):
                problems.append(f"launcher.exe PE 偏移非法：{pe}")
        if exe.stat().st_size < 100_000:
            problems.append("launcher.exe 体积异常小，可能是占位文件")

    if not ISS.is_file():
        problems.append(f"找不到安装脚本：{ISS}")
    else:
        txt = ISS.read_text(encoding="utf-8")
        m = re.search(r'#define\s+AppId\s+"([0-9A-Fa-f-]{36})"', txt)
        if not m:
            problems.append(".iss 的 AppId 不是 36 位 GUID（Inno 编译期会报）")
        icon = re.search(r'SetupIconFile=([^\r\n]+)', txt)
        if icon:
            rel = icon.group(1).strip().replace("{#StageDir}\\", "").replace("/", "\\")
            if not (STAGE / rel).is_file():
                problems.append(f".iss 引用的图标不存在：{rel}")
        if "{#StageDir}\\*" not in txt:
            problems.append(".iss 的 [Files] Source 没指向整个 staging 目录")

    if problems:
        return _fail(problems)

    # runtime 是几百 MB 的构建产物，单独报，不计入源码文件数
    n = sum(1 for p in STAGE.rglob("*") if p.is_file() and "runtime" not in p.relative_to(STAGE).parts)
    print(f"payload 自检通过：{n} 个源码文件，关键入口齐全，launcher.exe 合法，.iss 引用可达")
    rt = STAGE / "runtime" / "python" / "python.exe"
    print(f"内嵌运行时：{'已就绪 ' + str(round(rt.stat().st_size/1024/1024,1)) + ' MB' if rt.is_file() else '未构建（运行 build_runtime.py 或 build_installer.bat 会补上）'}")
    print("下一步：装好 Inno Setup 6 后双击 installer\\build_installer.bat")
    return 0


def _fail(problems: list[str]) -> int:
    print(f"[失败] payload 自检不通过（{len(problems)} 项）：", file=sys.stderr)
    for p in problems:
        print("  - " + p, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
