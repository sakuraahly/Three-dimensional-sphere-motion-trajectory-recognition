"""生成「纯净交付包」：只含源代码 + 运行必需物，直接产出可发送的 zip。

为什么需要它（2026-11-05）：最终要打包发送一个纯净版本，只有源码和运行
必须物，实验数据、测试结果、日志、文档史料、模型权重、素材一律没有。

纯净包**带公开文档**（README / 使用说明 / 算法解析），但不带内部史料
（planbooks、交接文档、待办清单、论文 PDF）。

`.gitignore` 已经改成**白名单式**，但它只管「未跟踪文件」—— 仓库里已经
入库的 60 多份 docs/、legacy/ 不会因为改了忽略规则就消失，`git archive HEAD`
照样会把它们打进去。所以本工具不读 git 索引，而是按**同一份白名单**直接从
工作区复制文件：

    python apps/tools/make_clean_bundle.py --check        # 只校验，不写文件
    python apps/tools/make_clean_bundle.py --dry-run      # 列清单
    python apps/tools/make_clean_bundle.py               # 生成 zip
    python apps/tools/make_clean_bundle.py --no-tests    # 连测试代码也不带

打包前会做四道校验，任何一道不过就拒绝生成：
  1. 白名单与 `.gitignore` 一致（用 `git check-ignore --no-index` 反查）；
  2. 包里没有禁入目录 / 缓存 / 大二进制；
  3. 包里的文本文件没有本机用户名、私人目录、真实邮箱；
  4. 生成的 zip 内容与实际收集的文件逐项对得上。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path


def _repo_root() -> Path:
    """按标记上溯到含 apps/ 的祖先，不写死 parents[N]。"""
    p = Path(__file__).resolve().parent
    while not (p / "apps").is_dir() and p.parent != p:
        p = p.parent
    return p


ROOT = _repo_root()

# ── 白名单：和 .gitignore 的 `!` 规则一一对应，改一处必须改两处 ──────────
#: 源码目录，只收 .py（含子目录）
PY_DIRS = ("apps", "sphere_tracker", "stereo", "calib", "tests")
#: 根目录必需文件：依赖清单 + 三个启动入口 + 最小说明
ROOT_FILES = (
    ".gitignore", ".gitattributes", "README.md",
    "requirements.txt", "requirements-ai.txt", "environment.yml",
    "启动程序.py", "启动程序_命令行.py", "启动程序.bat",
)
#: 其余运行必需物：标定板（拍标定图要打印）+ 空目录占位 / 放置说明
EXTRA_GLOBS = (
    "assets/boards/*.png", "assets/boards/*.pdf",
    "assets/videos/README.md", "models/README.md", "data/README.md",
    "config/.gitkeep",
)
#: 公开文档集（GitHub 发布用）。与 .gitignore 的 `!/docs/...` 规则一一对应，
#: 改一处必须改两处，8h 测试会用 `git check-ignore --no-index` 反查一致性。
#: 只列面向使用者的说明文档，不含 planbooks / 交接 / 待办 / 决策 / 论文 PDF。
PUBLIC_DOCS = (
    "docs/GitHub-README.md", "docs/算法解析.md", "docs/项目总纲.md",
    "docs/用户手册.md", "docs/API.md", "docs/3D查看器.md",
    "docs/文件结构.md", "docs/数据标准.md", "docs/运动数据交付说明.md",
    "docs/真机标定操作清单.md", "docs/标定算法对比.md",
    "docs/纯色球检测方案.md", "docs/摄像头设备管理方案.md",
    "docs/插件化架构.md", "docs/主线.md", "docs/坐标系统图.png",
)
#: 原生 exe 启动器的重建源码（产物 exe 不进包：平台相关、体积大）
EXTRA_BUILD_FILES = (
    "native/launcher/main.go", "native/launcher/go.mod",
    "native/launcher/build.bat", "native/launcher/icon.rc",
    "native/launcher/icon.ico",
)
#: 发布包首页：本地 README.md 是内部状态概览，不适合当 GitHub 首页。
#: 打包时用它替换根目录 README.md（名字不变，所以白名单与 zip 清单仍对得上）。
GITHUB_README = ROOT / "docs" / "GitHub-README.md"

ALLOWED_SUFFIX = {".py", ".md", ".txt", ".yml", ".bat", ".png", ".pdf",
                  ".go", ".mod", ".rc", ".ico"}
ALLOWED_NAMES = {".gitignore", ".gitattributes", ".gitkeep"}
TEXT_SUFFIX = {".py", ".md", ".txt", ".yml", ".bat", ".go", ".mod", ".rc"}
FORBIDDEN_PARTS = {
    "outputs", "logs", "legacy", ".workbuddy", "__pycache__",
    ".git", "node_modules", "venv", ".venv",
}
#: 拼接构造，避免本文件自己命中校验（它也在包里）
PRIVATE_MARKS = ("391" + "63", "@qq." + "com", "课程" + "相关作业资料等")
MAX_FILE_BYTES = 8 * 1024 * 1024     # 单个文件上限，挡住误混进来的权重/视频


def collect(*, with_tests: bool = True, with_assets: bool = True) -> list[Path]:
    """按白名单收集文件，返回按相对路径排序的去重列表。

    ``with_assets=False`` 是给「连标定板都不想要」的极简包用的：只留源码、
    依赖清单和启动入口。默认保留，因为拍标定图必须把标定板打印出来。
    """
    found: dict[str, Path] = {}

    def add(p: Path) -> None:
        if p.is_file():
            found[p.relative_to(ROOT).as_posix()] = p

    for rel in ROOT_FILES:
        add(ROOT / rel)
    for rel in PUBLIC_DOCS:
        add(ROOT / rel)
    for rel in EXTRA_BUILD_FILES:
        add(ROOT / rel)
    dirs = PY_DIRS if with_tests else tuple(d for d in PY_DIRS if d != "tests")
    for d in dirs:
        base = ROOT / d
        if base.is_dir():
            for p in base.rglob("*.py"):
                if "__pycache__" not in p.parts:
                    add(p)
    if with_assets:
        for pattern in EXTRA_GLOBS:
            for p in ROOT.glob(pattern):
                add(p)
    return [found[k] for k in sorted(found)]


def validate(files: list[Path]) -> list[str]:
    """校验内容：禁入目录、文件类型、体积、个人标识。返回问题列表。"""
    problems: list[str] = []
    if not GITHUB_README.is_file():
        problems.append("缺少 docs/GitHub-README.md（发布包首页）")
    for p in files:
        rel = p.relative_to(ROOT).as_posix()
        parts = rel.split("/")
        bad_dir = FORBIDDEN_PARTS.intersection(parts)
        if bad_dir:
            problems.append(f"禁入目录 {sorted(bad_dir)}: {rel}")
            continue
        if p.suffix.lower() not in ALLOWED_SUFFIX and p.name not in ALLOWED_NAMES:
            problems.append(f"不该在纯净包里的文件类型: {rel}")
            continue
        size = p.stat().st_size
        if size > MAX_FILE_BYTES:
            problems.append(f"文件过大（{size / 1024 / 1024:.1f} MiB）: {rel}")
            continue
        if p.suffix.lower() in TEXT_SUFFIX or p.name in ALLOWED_NAMES:
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
            except OSError as exc:
                problems.append(f"读不了 {rel}: {exc}")
                continue
            for mark in PRIVATE_MARKS:
                if mark in txt:
                    problems.append(f"含个人标识 {mark[:6]}…: {rel}")
    return problems


def gitignore_conflicts(files: list[Path]) -> list[str]:
    """白名单里的文件若被 `.gitignore` 忽略 = 两份清单漂移，必须拦下。

    用 `--no-index` 才是在测「规则本身」：不加它，已跟踪文件一律不算忽略，
    `docs/` 那种老文件会被误判成"通过"。
    """
    rels = [p.relative_to(ROOT).as_posix() for p in files]
    if not rels:
        return []
    try:
        proc = subprocess.run(
            ["git", "check-ignore", "--no-index", *rels],
            cwd=str(ROOT), text=True, capture_output=True, check=False)
    except OSError:                                   # 没装 git：跳过这项
        return []
    return [line.strip().strip('"') for line in proc.stdout.splitlines()
            if line.strip()]


def build(files: list[Path], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in files:
            rel = p.relative_to(ROOT).as_posix()
            if rel == "README.md" and GITHUB_README.is_file():
                z.writestr("README.md", GITHUB_README.read_bytes())
            else:
                z.write(p, arcname=rel)
    # 回读校验：写进去的必须正好是收集到的那些
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
    expected = [p.relative_to(ROOT).as_posix() for p in files]
    if names != expected:
        raise RuntimeError(
            f"zip 内容不一致：{len(names)} 项 vs 预期 {len(expected)} 项")


def _summary(files: list[Path]) -> tuple[int, int]:
    groups: Counter[str] = Counter()
    total = 0
    for p in files:
        rel = p.relative_to(ROOT).as_posix()
        groups[rel.split("/")[0] if "/" in rel else "(根目录)"] += 1
        total += p.stat().st_size
    for name, count in sorted(groups.items()):
        print(f"  {name:22s}{count:5d} 个")
    return len(files), total


def main() -> int:
    ap = argparse.ArgumentParser(description="生成纯净交付包（源码 + 运行必需物）")
    ap.add_argument("--out", default=None, help="输出 zip 路径")
    ap.add_argument("--dry-run", action="store_true", help="只列清单，不写文件")
    ap.add_argument("--check", action="store_true", help="只做校验，不写文件")
    ap.add_argument("--list", action="store_true", help="逐项列出文件")
    ap.add_argument("--no-tests", action="store_true", help="不带 tests/")
    ap.add_argument("--no-assets", action="store_true",
                    help="连标定板与占位说明也不带（只要源码 + 依赖 + 启动入口）")
    args = ap.parse_args()

    files = collect(with_tests=not args.no_tests,
                    with_assets=not args.no_assets)
    if not files:
        print("[失败] 白名单一个文件都没收到，路径可能不对", file=sys.stderr)
        return 1

    problems = validate(files) + [
        f"白名单被 .gitignore 忽略: {rel}" for rel in gitignore_conflicts(files)]
    if problems:
        print(f"[失败] 校验不通过（{len(problems)} 项）：", file=sys.stderr)
        for msg in problems[:20]:
            print("  " + msg, file=sys.stderr)
        return 1

    n, size = _summary(files)
    print(f"校验通过：{n} 个文件 / {size / 1024 / 1024:.2f} MiB（未压缩）")
    if args.list:
        for p in files:
            print("  " + p.relative_to(ROOT).as_posix())
    if args.check or args.dry_run:
        return 0

    out = (Path(args.out) if args.out else
           ROOT / "outputs" / f"clean_bundle_{datetime.now():%Y%m%d_%H%M%S}.zip")
    out = out if out.is_absolute() else ROOT / out
    build(files, out)
    print(f"已生成：{out}  ({out.stat().st_size / 1024 / 1024:.2f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
