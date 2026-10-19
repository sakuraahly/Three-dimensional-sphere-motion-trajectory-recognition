"""生成 / 更新 SBOM（软件物料清单）。

产物::

    docs/SBOM.md          人读清单（直接 / 传递 / 环境内 / 原生组件 / 许可证 / 风险）
    docs/sbom.cdx.json    CycloneDX 1.5（机器读；purl + 依赖图 + 许可证）

用法::

    python apps/tools/build_sbom.py            # 生成并覆盖 docs/ 下的两份文件
    python apps/tools/build_sbom.py --check    # 只检查是否过期；过期退出码 1
    python apps/tools/build_sbom.py --out-dir /tmp/sbom --quiet

为什么要有这个脚本
------------------
SBOM 是**当前机器环境的快照**：换机器、升级依赖、装/删包都会让上一份过期。
手工维护必然漂移，所以把它固化成"跑一次就更新 + `--check` 能报红"的确定性产物。

边界
----
- 只统计**本环境已安装**的 Python 发行版；`environment.yml` 里的 conda 原生包
  （MKL / Qt / 字体等）不逐条展开，只列直接影响运行链路的原生组件。
- 许可证优先用包元数据/Classifier；元数据空白的项用本文件的 `LICENSE_HINTS`
  补全（标 `curated`）。**对外分发前按 curated 项逐条复核。**
- 不联网查许可证/漏洞；这是物料清单，不是法务审查或 CVE 扫描。
"""

from __future__ import annotations

import argparse
import copy
import datetime as _dt
import importlib.metadata as md
import json
import platform
import re
import subprocess
import sys
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "sphere_tracker").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA_FILE = "sbom.cdx.json"
MD_FILE = "SBOM.md"
DEFAULT_OUT = ROOT / "docs"

SPDX_IDS = {
    "MIT", "BSD-2-Clause", "BSD-3-Clause", "Apache-2.0", "MPL-2.0",
    "AGPL-3.0", "LGPL-3.0-only", "PSF-2.0", "Unlicense",
}

# 直接依赖：角色 / 声明处 / 是否必需。
# ⚠️ 新增 requirements.txt 里的包时，必须在这里登记（脚本会报"未登记"）。
DIRECT: dict[str, tuple[str, str, bool]] = {
    "numpy": ("核心：数组 / 几何 / 数值", "requirements.txt", True),
    "opencv-contrib-python": ("核心：检测 / 标定 / 视频 I/O", "requirements.txt", True),
    "matplotlib": ("核心：绘图；3D 可视化备选", "requirements.txt", True),
    "pyyaml": ("可选：标定参数导出 YAML", "requirements.txt", False),
    "scipy": ("规划：M4 Savitzky-Golay（当前代码未 import）", "docs/计划书.md", False),
    "torch": ("可选：AI 后端（YOLO / SAM）", "requirements-ai.txt", False),
    "torchvision": ("可选：AI 后端", "requirements-ai.txt", False),
    "ultralytics": ("可选：YOLO / SAM（AGPL）", "requirements-ai.txt", False),
    "python-pptx": ("可选：开题报告 PPT 工具", "docs/开发指南.md", False),
    "pillow": ("可选：PPT 预览渲染", "docs/开发指南.md", False),
    "lxml": ("可选：python-pptx 的 XML 依赖", "docs/开发指南.md", False),
}

# 许可证补全表：只在包元数据/Classifier 给不出可用值时使用。
# source=metadata 表示与元数据一致；curated 表示人工补全。
LICENSE_HINTS: dict[str, tuple[str, str]] = {
    "anyio": ("MIT", "curated"),
    "certifi": ("MPL-2.0", "metadata"),
    "charset-normalizer": ("MIT", "metadata"),
    "click": ("BSD-3-Clause", "curated"),
    "colorama": ("BSD-3-Clause", "classifier"),
    "contourpy": ("BSD-3-Clause", "curated"),
    "cycler": ("BSD-3-Clause", "classifier"),
    "filelock": ("Unlicense", "curated"),
    "fonttools": ("MIT", "metadata"),
    "fsspec": ("BSD-3-Clause", "curated"),
    "h11": ("MIT", "metadata"),
    "hf-xet": ("Apache-2.0", "classifier"),
    "httpcore": ("BSD-3-Clause", "classifier"),
    "httpx": ("BSD-3-Clause", "metadata"),
    "huggingface-hub": ("Apache-2.0", "metadata"),
    "idna": ("BSD-3-Clause", "curated"),
    "jinja2": ("BSD-3-Clause", "classifier"),
    "kiwisolver": ("BSD-3-Clause", "classifier"),
    "lap": ("BSD-2-Clause", "metadata"),
    "lxml": ("BSD-3-Clause", "metadata"),
    "markupsafe": ("BSD-3-Clause", "curated"),
    "matplotlib": ("Matplotlib License (PSF-based)", "classifier"),
    "mpmath": ("BSD-3-Clause", "metadata"),
    "munkres": ("Apache-2.0", "classifier"),
    "networkx": ("BSD-3-Clause", "curated"),
    "numpy": ("BSD-3-Clause", "curated"),
    "nvidia-ml-py": ("BSD-3-Clause", "metadata"),
    "opencv-contrib-python": ("Apache-2.0", "metadata"),
    "packaging": ("Apache-2.0 OR BSD-2-Clause", "curated"),
    "pandas": ("BSD-3-Clause", "metadata"),
    "pillow": ("MIT-CMU", "curated"),
    "pip": ("MIT", "curated"),
    "polars": ("MIT", "metadata"),
    "polars-runtime-32": ("MIT", "metadata"),
    "psutil": ("BSD-3-Clause", "metadata"),
    "py-cpuinfo": ("MIT", "metadata"),
    "pyparsing": ("MIT", "curated"),
    "pyside6": ("LGPL-3.0-only", "curated"),
    "python-dateutil": ("Apache-2.0 OR BSD-3-Clause", "classifier"),
    "python-pptx": ("MIT", "metadata"),
    "pyyaml": ("MIT", "metadata"),
    "requests": ("Apache-2.0", "metadata"),
    "safetensors": ("Apache-2.0", "classifier"),
    "scipy": ("BSD-3-Clause", "classifier"),
    "setuptools": ("MIT", "curated"),
    "shiboken6": ("LGPL-3.0-only", "curated"),
    "six": ("MIT", "metadata"),
    "sympy": ("BSD-3-Clause", "metadata"),
    "timm": ("Apache-2.0", "metadata"),
    "torch": ("BSD-3-Clause", "metadata"),
    "torchvision": ("BSD-3-Clause", "metadata"),
    "tornado": ("Apache-2.0", "metadata"),
    "tqdm": ("MPL-2.0 AND MIT", "metadata"),
    "typing-extensions": ("PSF-2.0", "curated"),
    "tzdata": ("Apache-2.0", "metadata"),
    "ultralytics": ("AGPL-3.0", "metadata"),
    "ultralytics-thop": ("AGPL-3.0", "metadata"),
    "urllib3": ("MIT", "curated"),
    "xlsxwriter": ("BSD-2-Clause", "metadata"),
}

ENV_NOTES = {
    "anyio": "HF/httpx 可选链（环境残留）",
    "h11": "httpx 的依赖（环境残留）",
    "hf-xet": "HuggingFace 下载加速（AI 可选）",
    "httpcore": "httpx 的依赖（环境残留）",
    "httpx": "HuggingFace 客户端依赖（AI 可选）",
    "huggingface-hub": "模型下载（AI 可选）",
    "py-cpuinfo": "环境信息工具（timm/ultralytics 可选链）",
    "pyside6": "conda Qt 绑定；matplotlib Qt 后端可能用",
    "safetensors": "模型权重格式（AI 可选）",
    "shiboken6": "PySide6 的绑定层",
    "timm": "torch 视觉模型库（ultralytics 可选链）",
    "tornado": "Jupyter/异步依赖（环境残留）",
}

_REQ_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def canon(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name.strip()).lower()


# --------------------------------------------------------------------------
# 收集环境事实
# --------------------------------------------------------------------------
def collect_dists() -> dict[str, md.Distribution]:
    return {
        canon(d.metadata["Name"]): d
        for d in md.distributions()
        if d.metadata.get("Name")
    }


def versions_of(dists: dict[str, md.Distribution]) -> dict[str, str]:
    return {n: d.version.strip() for n, d in dists.items()}


def deps_of(name: str, dists: dict[str, md.Distribution]) -> list[str]:
    """只保留本环境已安装的依赖；跳过 extras 里的自引用。"""
    out: set[str] = set()
    for raw in (dists[name].requires or []):
        m = _REQ_NAME.match(raw)
        if not m:
            continue
        dep = canon(m.group(1))
        if dep in dists and dep != name:
            out.add(dep)
    return sorted(out)


def closure(roots: set[str], dists: dict[str, md.Distribution]) -> set[str]:
    seen: set[str] = set()
    stack = list(roots)
    while stack:
        n = stack.pop()
        if n in seen or n not in dists:
            continue
        seen.add(n)
        stack.extend(deps_of(n, dists))
    return seen


def requirements_names() -> list[str]:
    path = ROOT / "requirements.txt"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.split("#", 1)[0].strip()
        m = _REQ_NAME.match(s)
        if m:
            out.append(canon(m.group(1)))
    return out


def license_from_metadata(dist: md.Distribution) -> str:
    for cls in (dist.metadata.get_all("Classifier") or []):
        if not cls.startswith("License ::"):
            continue
        low = cls.lower()
        if "mit" in low:
            return "MIT"
        if "apache" in low:
            return "Apache-2.0"
        if "mozilla" in low:
            return "MPL-2.0"
        if "affero" in low:
            return "AGPL-3.0"
        if "lgpl" in low:
            return "LGPL-3.0-only"
        if "bsd" in low:
            return "BSD-3-Clause"
    lic = (dist.metadata.get("License") or "").strip()
    return lic if lic in SPDX_IDS else "NOASSERTION"


def lic_obj(text: str) -> dict:
    if " OR " in text or " AND " in text:
        return {"expression": text}
    if text in SPDX_IDS:
        return {"license": {"id": text}}
    return {"license": {"name": text}}


def purl(name: str, version: str) -> str:
    return f"pkg:pypi/{name}@{quote(version, safe='')}"


def home_of(dist: md.Distribution) -> str:
    home = (dist.metadata.get("Home-page") or "").strip()
    if home:
        return home
    for value in (dist.metadata.get_all("Project-URL") or []):
        if "," in value:
            _label, url = value.split(",", 1)
            if url.strip():
                return url.strip()
    return ""


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT),
            stderr=subprocess.DEVNULL, timeout=5,
        ).decode("ascii", "replace").strip()
    except Exception:
        return "unknown"


def _nvidia(query: str) -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, timeout=5,
        ).decode("utf-8", "replace").strip().splitlines()
        return out[0].strip() if out else ""
    except Exception:
        return ""


def mkl_version() -> str:
    path = ROOT / "environment.yml"
    if not path.exists():
        return ""
    m = re.search(r"^\s*-\s*mkl=([^\s=]+)", path.read_text(encoding="utf-8"), re.M)
    return m.group(1) if m else ""


def native_components() -> list[tuple[str, str, str, str, str]]:
    """返回 (name, version, license, scope, note)。缺什么就跳过什么。"""
    out: list[tuple[str, str, str, str, str]] = [
        ("CPython", platform.python_version(), "PSF-2.0", "required",
         "解释器；绝对路径 %USERPROFILE%\\miniforge3\\envs\\myenv\\python.exe"),
    ]
    try:
        import cv2
        build = cv2.getBuildInformation()
        if "FFMPEG:" in build:
            out.append((
                "FFmpeg (OpenCV wheel bundled)", "prebuilt-unknown",
                "LGPL-2.1-or-later OR GPL-2.0-or-later", "required",
                "OpenCV 的 FFMPEG 后端；许可证取决于 wheel 构建，分发前复核",
            ))
    except Exception as exc:
        print(f"[警告] 读不到 OpenCV build info：{exc}（SBOM 会缺 FFmpeg）", file=sys.stderr)
    try:
        import torch
        cuda = torch.version.cuda
        if cuda:
            out.append(("NVIDIA CUDA Runtime", str(cuda),
                        "NVIDIA proprietary (EULA)", "optional",
                        f"torch {torch.__version__}；AI 后端才需要"))
        cudnn = torch.backends.cudnn.version()
        if cudnn:
            ver = f"{cudnn // 10000}.{(cudnn // 100) % 100}.{cudnn % 100}"
            out.append(("NVIDIA cuDNN", ver,
                        "NVIDIA proprietary (EULA)", "optional",
                        f"torch.backends.cudnn 报 {cudnn}"))
    except Exception as exc:
        print(f"[警告] 读不到 torch/CUDA 信息：{exc}（SBOM 会缺 CUDA/cuDNN）", file=sys.stderr)
    driver = _nvidia("driver_version")
    if driver:
        gpu = _nvidia("name")
        out.append(("NVIDIA Display Driver", driver,
                    "NVIDIA proprietary (EULA)", "required",
                    f"{gpu or 'GPU'}；CUDA 的最低驱动"))
    mkl = mkl_version()
    if mkl:
        out.append(("Intel MKL", mkl, "Intel Simplified Software License", "required",
                    "conda 环境里的 BLAS/LAPACK；numpy 使用"))
    out.append((platform.system(), platform.version(), "Proprietary", "required", "操作系统"))
    return out


# --------------------------------------------------------------------------
# 组数据
# --------------------------------------------------------------------------
def build_data() -> tuple[dict, list[str]]:
    dists = collect_dists()
    versions = versions_of(dists)
    warnings: list[str] = []

    direct = dict(DIRECT)
    req = requirements_names()
    for name in req:
        if name not in direct:
            direct[name] = ("（未登记角色：请补 build_sbom.py 的 DIRECT）",
                            "requirements.txt", True)
            warnings.append(f"requirements.txt 里的 `{name}` 未在 DIRECT 登记")
    for name, (_role, _src, required) in DIRECT.items():
        if required and name not in req:
            warnings.append(f"必需依赖 `{name}` 不在 requirements.txt 里")
        if name not in dists:
            warnings.append(f"直接依赖 `{name}` 当前环境未安装")

    direct_present = {n for n in direct if n in dists}
    req_roots = {n for n, (_r, _s, required) in direct.items()
                 if required and n in dists}
    opt_roots = direct_present - req_roots
    req_closure = closure(req_roots, dists)
    opt_closure = closure(opt_roots, dists) - req_closure
    reachable = req_closure | opt_closure
    env_only = set(dists) - reachable

    parents: dict[str, list[str]] = defaultdict(list)
    for n in dists:
        for dep in deps_of(n, dists):
            parents[dep].append(n)

    def license_of(name: str) -> tuple[str, str]:
        if name in LICENSE_HINTS:
            return LICENSE_HINTS[name]
        text = license_from_metadata(dists[name])
        if text != "NOASSERTION":
            return text, "metadata"
        warnings.append(f"`{name}` 的许可证元数据为空，已记 NOASSERTION")
        return "NOASSERTION", "missing"

    def scope_of(name: str) -> str:
        return "required" if name in req_closure else "optional"

    def rel_of(name: str) -> str:
        if name in direct:
            return "direct-required" if direct[name][2] else "direct-optional"
        if name in env_only:
            return "environment-only"
        return "transitive"

    def who(name: str) -> str:
        ps = [p for p in sorted(set(parents.get(name, []))) if p in reachable]
        if not ps:
            return "—"
        return ", ".join(ps[:2]) + ("…" if len(ps) > 2 else "")

    stamp = _dt.datetime.now(_dt.timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    commit = git_commit()
    native = native_components()

    data = {
        "dists": dists,
        "versions": versions,
        "direct": direct,
        "direct_present": direct_present,
        "req_closure": req_closure,
        "reachable": reachable,
        "env_only": env_only,
        "parents": parents,
        "license_of": license_of,
        "scope_of": scope_of,
        "rel_of": rel_of,
        "who": who,
        "stamp": stamp,
        "commit": commit,
        "native": native,
    }
    return data, warnings


# --------------------------------------------------------------------------
# 生成 Markdown
# --------------------------------------------------------------------------
def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join(["---"] * len(headers)) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


def build_markdown(data: dict) -> str:
    dists = data["dists"]
    versions = data["versions"]
    direct = data["direct"]
    native = data["native"]
    license_of = data["license_of"]
    who = data["who"]
    lines: list[str] = []
    lines += [
        "# SBOM — 软件物料清单",
        "",
        f"> 生成时间：{data['stamp']}　生成时 HEAD：`{data['commit']}`（仅参考）",
        f"> 环境：`myenv`（Python {platform.python_version()}，{platform.platform()}）",
        f"> 范围：本机运行环境里 **{len(dists)} 个 Python 发行版** + "
        f"**{len(native)} 个原生运行时组件**。",
        "> 机器可读版本：[`docs/sbom.cdx.json`](sbom.cdx.json)（CycloneDX 1.5）。",
        "",
        "> ⚠️ **这是物料清单，不是法务审查。** 许可证优先取包元数据；"
        "元数据空白的项按上游常见许可证补全，并在表里标 `curated`。"
        "对外分发前请按 `curated` 项逐条复核。",
        "",
        f"## 1. 直接依赖（{len(data['direct_present'])}）",
        "",
    ]
    direct_rows = []
    for name in sorted(data["direct_present"]):
        role, src, _required = direct[name]
        lic, how = license_of(name)
        direct_rows.append([f"`{name}`", versions[name], f"{lic}（{how}）", role, f"`{src}`"])
    lines += _table(["组件", "已装版本", "许可证", "用途", "声明处"], direct_rows)
    lines += [
        "",
        "> `scipy` 是 M4 规划项（当前代码未 import）；`torch/torchvision/ultralytics` "
        "是可选 AI 后端；`python-pptx/Pillow/lxml` 只给 PPT 工具用。",
        "",
        f"## 2. 传递依赖（本环境已安装、从直接依赖可达，{len(data['reachable'] - set(data['direct_present']))}）",
        "",
    ]
    trans = sorted(data["reachable"] - set(data["direct_present"]))
    trans_rows = []
    for name in trans:
        lic, how = license_of(name)
        trans_rows.append([f"`{name}`", versions[name], f"{lic}（{how}）", who(name)])
    lines += _table(["组件", "已装版本", "许可证", "引入者（已安装）"], trans_rows)
    lines += [
        "",
        f"## 3. 环境内但不在声明依赖闭包（{len(data['env_only'])}）",
        "",
        "> 这些包装在这台机器上，但**从项目声明的直接依赖走不到**；"
        "多半是上游可选依赖（HF 下载、Qt 后端）或环境残留。项目代码不直接 import。",
        "",
    ]
    env_rows = []
    for name in sorted(data["env_only"]):
        lic, how = license_of(name)
        env_rows.append([f"`{name}`", versions[name], f"{lic}（{how}）",
                         ENV_NOTES.get(name, "环境残留")])
    lines += _table(["组件", "已装版本", "许可证", "可能来源"], env_rows)
    lines += [
        "",
        "## 4. 原生 / 系统组件",
        "",
    ]
    lines += _table(["组件", "版本", "许可证", "范围", "说明"],
                    [list(x) for x in native])
    lines += [
        "",
        "## 5. 许可证汇总",
        "",
    ]
    counts = Counter(license_of(n)[0] for n in dists)
    lic_rows = [[lic, str(c), ", ".join(f"`{n}`" for n in sorted(dists)
                                        if license_of(n)[0] == lic)]
                for lic, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    lines += _table(["许可证", "组件数", "组件"], lic_rows)
    lines += [
        "",
        "## 6. 风险与注意",
        "",
        "- **AGPL-3.0**：`ultralytics`、`ultralytics-thop`。只在使用 `--method yolo*` 时进入运行链路；"
        "若对外分发包含它们的产物，AGPL 义务需要单独评估。",
        "- **LGPL / 商业**：`pyside6`、`shiboken6`（conda Qt 绑定，LGPLv3 / GPLv3 / 商业三重许可）。"
        "项目代码不直接 import；matplotlib 的 Qt 后端可能间接使用。",
        "- **NVIDIA 专有**：CUDA / cuDNN / 驱动不可再分发，受 NVIDIA EULA 约束。",
        "- **OpenCV wheel 内置 FFmpeg**：构建时可能启用 LGPL/GPL 组件；"
        "`cv2.getBuildInformation()` 只报 `FFMPEG: YES (prebuilt binaries)`，不暴露版本。",
        "- **不要同时装 `opencv-python`**：本环境只有 `opencv-contrib-python`；"
        "两者都提供 `cv2`，会互相覆盖（`requirements-ai.txt` 已写明 `--no-deps`）。",
        "- **许可证元数据缺口**：标 `curated` 的项是按上游常见许可证补的，不是包元数据原文。",
        "",
        "## 7. 方法、边界与复现",
        "",
        "- Python 包：`importlib.metadata` 读版本 / 许可证 / `Requires-Dist`；`pip freeze` 交叉核对。",
        "- 直接依赖：`requirements.txt`、`requirements-ai.txt`、`docs/开发指南.md`（PPT 可选）、"
        "`docs/计划书.md`（scipy 规划）。",
        "- 传递闭包：从直接依赖按 `Requires-Dist` 做 BFS，只统计本环境已安装的包。",
        "- 原生组件：`cv2.getBuildInformation()`、`torch.version.cuda`、"
        "`torch.backends.cudnn.version()`、`nvidia-smi`、`environment.yml`。",
        "- **更新**：`python apps/tools/build_sbom.py`；"
        "**检查是否过期**：`python apps/tools/build_sbom.py --check`（退出码 1 = 该更新）。",
        "- 边界：`environment.yml` 里还有约 90 个 conda 原生包（MKL / Qt / 字体等），"
        "本文只列了直接影响运行链路的原生组件；完整 conda 清单以 `environment.yml` 为准。",
        "",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# 生成 CycloneDX
# --------------------------------------------------------------------------
def build_bom(data: dict) -> dict:
    dists = data["dists"]
    versions = data["versions"]
    commit = data["commit"]
    project_ref = f"pkg:generic/three-dimensional-sphere-motion-trajectory-recognition@{commit}"
    components = []
    dep_entries = []

    for name in sorted(dists):
        lic, how = data["license_of"](name)
        ref = purl(name, versions[name])
        comp = {
            "type": "library",
            "bom-ref": ref,
            "name": dists[name].metadata["Name"].strip(),
            "version": versions[name],
            "purl": ref,
            "scope": data["scope_of"](name),
            "licenses": [lic_obj(lic)],
            "properties": [
                {"name": "dsh:relationship", "value": data["rel_of"](name)},
                {"name": "dsh:license_source", "value": how},
            ],
        }
        summary = (dists[name].metadata.get("Summary") or "").strip()
        if summary:
            comp["description"] = summary[:240]
        home = home_of(dists[name])
        if home:
            comp["externalReferences"] = [{"type": "website", "url": home}]
        if name in data["direct"]:
            comp["properties"].append(
                {"name": "dsh:declared_by", "value": data["direct"][name][1]})
        components.append(comp)
        dep_entries.append({
            "ref": ref,
            "dependsOn": [purl(dep, versions[dep])
                          for dep in deps_of(name, dists) if dep in dists],
        })

    for name, version, lic, scope, note in data["native"]:
        ref = (f"pkg:generic/{quote(name.lower().replace(' ', '-'), safe='')}"
               f"@{quote(version, safe='')}")
        components.append({
            "type": ("operating-system" if name in {"Windows", "Linux", "Darwin"}
                     else "platform" if name == "CPython" else "library"),
            "bom-ref": ref,
            "name": name,
            "version": version,
            "purl": ref,
            "scope": scope,
            "licenses": [lic_obj(lic)],
            "properties": [
                {"name": "dsh:relationship", "value": "native"},
                {"name": "dsh:note", "value": note},
            ],
        })
        dep_entries.append({"ref": ref, "dependsOn": []})

    direct_refs = [purl(n, versions[n]) for n in sorted(data["direct_present"])]
    dep_entries.insert(0, {"ref": project_ref, "dependsOn": direct_refs})

    gpu = _nvidia("name")
    props = [
        {"name": "dsh:python", "value": platform.python_version()},
        {"name": "dsh:platform", "value": platform.platform()},
    ]
    if gpu:
        props.append({"name": "dsh:gpu", "value": gpu})

    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{uuid.uuid4()}",
        "version": 1,
        "metadata": {
            "timestamp": _dt.datetime.now(_dt.timezone.utc).astimezone()
            .isoformat(timespec="seconds"),
            "tools": [{"vendor": "DeepSeek Harness",
                       "name": "build_sbom.py", "version": "1.0"}],
            "component": {
                "type": "application",
                "bom-ref": project_ref,
                "name": "Three-dimensional-sphere-motion-trajectory-recognition",
                "version": commit,
                "properties": props,
            },
        },
        "components": components,
        "dependencies": dep_entries,
    }


# --------------------------------------------------------------------------
# 比较 / 写文件
# --------------------------------------------------------------------------
def _norm_md(text: str) -> str:
    # 生成时间 / 项目提交是"生成上下文"，不参与过期判断（否则每次提交都红）
    return "\n".join(ln for ln in text.splitlines()
                     if not ln.startswith("> 生成时间")).strip()


def _norm_bom(bom: dict) -> str:
    b = copy.deepcopy(bom)
    b.pop("serialNumber", None)
    meta = b.get("metadata", {})
    meta.pop("timestamp", None)
    comp = meta.get("component")
    if isinstance(comp, dict):
        comp.pop("version", None)
        comp["bom-ref"] = "pkg:generic/project"
    # `build_bom()` 把生成时的 commit 编进 project 的 ref；它只是"生成上下文"，
    # 不参与过期判断（否则每提交一次 --check 就红）。
    prefix = "pkg:generic/three-dimensional-sphere-motion-trajectory-recognition@"
    for dep in b.get("dependencies", []):
        if str(dep.get("ref", "")).startswith(prefix):
            dep["ref"] = "pkg:generic/project"
    return json.dumps(b, sort_keys=True, ensure_ascii=False)


def check(out_dir: Path, quiet: bool = False) -> int:
    md_path = out_dir / MD_FILE
    json_path = out_dir / SCHEMA_FILE
    if not md_path.exists() or not json_path.exists():
        print(f"[过期] {md_path} 或 {json_path} 不存在")
        print("  更新：python apps/tools/build_sbom.py")
        return 1
    data, warnings = build_data()
    want_md = build_markdown(data)
    want_bom = build_bom(data)
    if not quiet:
        for w in warnings:
            print(f"  [警告] {w}")
    old_md = md_path.read_text(encoding="utf-8")
    old_bom = json.loads(json_path.read_text(encoding="utf-8"))
    if _norm_md(old_md) == _norm_md(want_md) and _norm_bom(old_bom) == _norm_bom(want_bom):
        if not quiet:
            print("SBOM 已是最新（组件 / 版本 / 许可证 / 原生组件一致）")
        return 0

    old_map = {c.get("name"): c.get("version") for c in old_bom.get("components", [])}
    new_map = {c.get("name"): c.get("version") for c in want_bom.get("components", [])}
    diff: list[str] = []
    for name in sorted(set(old_map) - set(new_map)):
        diff.append(f"- 移除 {name} {old_map[name]}")
    for name in sorted(set(new_map) - set(old_map)):
        diff.append(f"+ 新增 {name} {new_map[name]}")
    for name in sorted(set(old_map) & set(new_map)):
        if old_map[name] != new_map[name]:
            diff.append(f"~ 版本 {name}: {old_map[name]} -> {new_map[name]}")
    print("[过期] SBOM 与当前环境不一致：")
    for line in diff[:20]:
        print(f"  {line}")
    if len(diff) > 20:
        print(f"  … 还有 {len(diff) - 20} 条")
    print("  更新：python apps/tools/build_sbom.py")
    return 1


def write(out_dir: Path, quiet: bool = False) -> int:
    data, warnings = build_data()
    if not quiet:
        for w in warnings:
            print(f"  [警告] {w}")
    out_dir.mkdir(parents=True, exist_ok=True)
    md_text = build_markdown(data)
    bom_text = json.dumps(build_bom(data), indent=2, ensure_ascii=False) + "\n"
    for path, text in ((out_dir / MD_FILE, md_text), (out_dir / SCHEMA_FILE, bom_text)):
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
        if not quiet:
            print(f"已更新：{path}")
    if not quiet:
        print(f"  Python 发行版 {len(data['dists'])}；原生组件 {len(data['native'])}；"
              f"直接依赖 {len(data['direct_present'])}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="生成 / 更新 SBOM")
    ap.add_argument("--check", action="store_true", help="只检查是否过期，不写文件")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT), help="输出目录（默认 docs/）")
    ap.add_argument("--quiet", action="store_true", help="少打印")
    args = ap.parse_args()
    out = Path(args.out_dir)
    if args.check:
        return check(out, quiet=args.quiet)
    return write(out, quiet=args.quiet)


if __name__ == "__main__":
    raise SystemExit(main())
