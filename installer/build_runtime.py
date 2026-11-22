# -*- coding: utf-8 -*-
"""构建内嵌 Python 运行时（安装包里"开箱即用"的那一份）。

单一真相：版本、下载地址、sha256、依赖、精简策略全在本文件顶部。
产物：outputs/release_stage/runtime/python/（构建产物，gitignore，不入库）。
缓存：outputs/runtime_cache/<archive>（重复构建不重下）。

用法：
    python installer/build_runtime.py            # 缺则构建，有则跳过
    python installer/build_runtime.py --force    # 强制重建
    python installer/build_runtime.py --check    # 只校验现有产物，不改动

为什么要自己下载而不是复制开发机 conda 环境：
    开发机 myenv 有 5.1 GB（含 CUDA 等无关大包），不能塞进安装包。
    python-build-standalone 的 install_only_stripped 是可重定位的完整 CPython
    （含 tkinter + pip），这里再装齐 requirements.txt 即可，约 390 MB。
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE_TAG = "20261110"
PY_VERSION = "3.14.10"
# 与 requirements.txt 的 pin 保持一致；版本不符时必须重建，不能复用旧 stage。
EXPECTED_CV2 = "5.0.0"
ARCHIVE = f"cpython-{PY_VERSION}+{RELEASE_TAG}-x86_64-pc-windows-msvc-install_only_stripped.tar.gz"
URL = (f"https://github.com/astral-sh/python-build-standalone/releases/download/"
       f"{RELEASE_TAG}/cpython-{PY_VERSION}%2B{RELEASE_TAG}-x86_64-pc-windows-msvc-install_only_stripped.tar.gz")
SHA256 = "32b6cf90a03e2bc15fda8d2ab7ff78c4870f138887d612932650ce764efaab8f"

CACHE = ROOT / "outputs" / "runtime_cache" / ARCHIVE
STAGE = ROOT / "outputs" / "release_stage" / "runtime" / "python"
REQUIREMENTS = ROOT / "requirements.txt"
# 必须能 import 的模块（与 setup_env.py 的 CORE_IMPORTS 对齐）
PROBE = "import cv2, numpy, scipy, tkinter, matplotlib, yaml, PIL; print('OK')"


def log(msg: str) -> None:
    print(f"[runtime] {msg}", flush=True)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download() -> None:
    if CACHE.is_file() and sha256_of(CACHE) == SHA256:
        log(f"命中缓存 {CACHE.name}")
        return
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    log(f"下载 {URL}")
    tmp = CACHE.with_suffix(".part")
    urllib.request.urlretrieve(URL, tmp)
    got = sha256_of(tmp)
    if got != SHA256:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"sha256 不符：期望 {SHA256}，实际 {got}")
    tmp.replace(CACHE)
    log("sha256 校验通过")


def size_mb(base: Path) -> float:
    total = 0
    for root, _dirs, files in os.walk(base):
        for f in files:
            try:
                total += (Path(root) / f).stat().st_size
            except OSError:
                pass
    return total / 1024 / 1024


def prune_runtime() -> int:
    """删掉运行时不需要的东西，尤其是会触发 Windows MAX_PATH 的深层文件。

    为什么必须在 import 探针**之后**再跑一次：Python 导入 numpy/scipy 时会
    重新生成 `__pycache__`；只在 pip 后清理，最终 staging 里又会冒出几十层
    `__pycache__`。CI 上 Inno 正是在压缩这些深层路径时失败。
    """
    removed = 0
    for root, dirs, files in os.walk(STAGE, topdown=True):
        root_path = Path(root)
        for d in list(dirs):
            full = root_path / d
            rel = full.relative_to(STAGE).as_posix()
            drop = d == "__pycache__"
            # 包自带测试/头文件不参与运行；它们的路径最深，直接去掉。
            if rel.startswith("Lib/site-packages/") and d in {"tests", "test"}:
                drop = True
            if rel.startswith("Lib/site-packages/numpy/") and d == "include":
                drop = True
            if rel == "Lib/site-packages/numpy/_pyinstaller":
                drop = True
            if drop:
                shutil.rmtree(full, ignore_errors=True)
                removed += 1
                if d in dirs:
                    dirs.remove(d)
        for f in files:
            if f.endswith(".pyc"):
                try:
                    (root_path / f).unlink()
                    removed += 1
                except OSError:
                    pass
    # 兼容旧的显式清理项（即使上面的通用规则改坏也仍能减掉这些）
    for rel in ("Lib/site-packages/scipy/_lib/tests",
                "Lib/site-packages/numpy/_core/tests",
                "Lib/site-packages/matplotlib/tests",
                "Lib/site-packages/cv2/data",
                "Lib/site-packages/numpy/_core/include"):
        p = STAGE / rel
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
            removed += 1
    return removed


def cv2_version() -> str:
    exe = STAGE / "python.exe"
    if not exe.is_file():
        return ""
    r = subprocess.run(
        [str(exe), "-c", "import cv2; print(cv2.__version__)"],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.stdout.strip() if r.returncode == 0 else ""


def build(force: bool) -> None:
    download()
    if STAGE.is_dir() and not force:
        got = cv2_version()
        if got == EXPECTED_CV2:
            removed = prune_runtime()
            log(f"已有运行时 {STAGE}（{size_mb(STAGE):.0f} MB，OpenCV {got}），"
                f"跳过构建；清理 {removed} 处")
            return
        log(f"已有运行时 OpenCV={got or '无法读取'}，期望 {EXPECTED_CV2}，重建…")

    if STAGE.is_dir():
        log("删除旧运行时…")
        shutil.rmtree(STAGE)
    STAGE.parent.mkdir(parents=True, exist_ok=True)
    log("解压…")
    with tarfile.open(CACHE, "r:gz") as tf:
        tf.extractall(STAGE.parent)
    # 归档顶层是 python/，正好落在 STAGE
    if not (STAGE / "python.exe").is_file():
        raise SystemExit(f"解压后没找到 {STAGE / 'python.exe'}，检查归档结构")

    py = STAGE / "python.exe"
    log("安装依赖…")
    subprocess.run([str(py), "-m", "pip", "install", "--no-cache-dir",
                    "--disable-pip-version-check", "-r", str(REQUIREMENTS)],
                   check=True, cwd=ROOT)

    log("精简（__pycache__ / 包自带 tests / numpy 头文件）…")
    removed = prune_runtime()
    log(f"清理 {removed} 处")

    log("验证 import…")
    r = subprocess.run([str(py), "-c", PROBE], capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise SystemExit(f"内嵌环境 import 失败：{r.stderr}")
    # import 会重新生成 __pycache__，必须在验证后再清一次。
    removed = prune_runtime()
    log(f"import 后清理 {removed} 处")
    log(f"完成：{size_mb(STAGE):.0f} MB @ {STAGE}")


def check() -> int:
    if not (STAGE / "python.exe").is_file():
        print(f"[runtime] 缺失 {STAGE / 'python.exe'}")
        return 1
    r = subprocess.run([str(STAGE / "python.exe"), "-c", PROBE],
                       capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    got = cv2_version()
    ok = r.returncode == 0 and got == EXPECTED_CV2
    print(f"[runtime] {size_mb(STAGE):.0f} MB, OpenCV {got or '?'}, "
          f"import {'OK' if r.returncode == 0 else 'FAIL'}")
    if not ok:
        if r.returncode != 0:
            print(r.stderr)
        if got != EXPECTED_CV2:
            print(f"[runtime] 版本不符：期望 OpenCV {EXPECTED_CV2}")
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="强制重建运行时")
    ap.add_argument("--check", action="store_true", help="只校验现有产物")
    args = ap.parse_args()
    if args.check:
        return check()
    build(args.force)
    rc = check()
    # check 会 import cv2/scipy，重新生成 __pycache__；必须在打包前再清一次，
    # 否则 Inno 会去压缩几百个深层 .pyc（CI 的 MAX_PATH 就是这么爆的）。
    removed = prune_runtime()
    log(f"check 后清理 {removed} 处")
    return rc


if __name__ == "__main__":
    sys.exit(main())
