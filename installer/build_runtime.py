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


def build(force: bool) -> None:
    download()
    if STAGE.is_dir() and not force:
        log(f"已有运行时 {STAGE}（{size_mb(STAGE):.0f} MB），跳过；--force 可重建")
        return

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

    log("精简（__pycache__ / 包自带 tests）…")
    removed = 0
    for root, dirs, _files in os.walk(STAGE, topdown=False):
        if "__pycache__" in dirs:
            shutil.rmtree(os.path.join(root, "__pycache__"), ignore_errors=True)
            removed += 1
    for rel in ("Lib/site-packages/scipy/_lib/tests",
                "Lib/site-packages/numpy/_core/tests",
                "Lib/site-packages/matplotlib/tests",
                "Lib/site-packages/cv2/data"):
        p = STAGE / rel
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
            removed += 1
    log(f"清理 {removed} 处")

    log("验证 import…")
    r = subprocess.run([str(py), "-c", PROBE], capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise SystemExit(f"内嵌环境 import 失败：{r.stderr}")
    log(f"完成：{size_mb(STAGE):.0f} MB @ {STAGE}")


def check() -> int:
    if not (STAGE / "python.exe").is_file():
        print(f"[runtime] 缺失 {STAGE / 'python.exe'}")
        return 1
    r = subprocess.run([str(STAGE / "python.exe"), "-c", PROBE],
                       capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    print(f"[runtime] {size_mb(STAGE):.0f} MB, import {'OK' if r.returncode == 0 else 'FAIL'}")
    if r.returncode != 0:
        print(r.stderr)
    return r.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="强制重建运行时")
    ap.add_argument("--check", action="store_true", help="只校验现有产物")
    args = ap.parse_args()
    if args.check:
        return check()
    build(args.force)
    return check()


if __name__ == "__main__":
    sys.exit(main())
