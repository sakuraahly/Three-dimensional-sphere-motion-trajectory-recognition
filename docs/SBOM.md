# SBOM — 软件物料清单

> 生成时间：2026-10-19 10:48:33 +0800　项目提交：`7b78c88`
> 环境：`myenv`（Python 3.14.7，Windows-11-10.0.26300-SP0）
> 范围：本机运行环境里 **59 个 Python 发行版** + 原生运行时组件。
> 机器可读版本：[`docs/sbom.cdx.json`](sbom.cdx.json)（CycloneDX 1.5）。

> ⚠️ **这是物料清单，不是法务审查。** 许可证优先取包元数据；元数据空白的项按上游常见许可证补全，并在表里标 `curated`。对外分发前请按 `curated` 项逐条复核。

## 1. 直接依赖（11）

| 组件 | 已装版本 | 许可证 | 用途 | 声明处 |
| --- | --- | --- | --- | --- |
| `lxml` | 6.1.3 | BSD-3-Clause（metadata） | 可选：python-pptx 的 XML 依赖 | `docs/开发指南.md` |
| `matplotlib` | 3.11.2 | Matplotlib License (PSF-based)（classifier） | 核心：绘图；3D 可视化备选 | `requirements.txt` |
| `numpy` | 2.5.3 | BSD-3-Clause（curated） | 核心：数组 / 几何 / 数值 | `requirements.txt` |
| `opencv-contrib-python` | 5.0.0.93 | Apache-2.0（metadata） | 核心：检测 / 标定 / 视频 I/O | `requirements.txt` |
| `pillow` | 12.3.0 | MIT-CMU（curated） | 可选：PPT 预览渲染 | `docs/开发指南.md` |
| `python-pptx` | 1.0.2 | MIT（metadata） | 可选：开题报告 PPT 工具 | `docs/开发指南.md` |
| `pyyaml` | 6.0.3 | MIT（metadata） | 可选：标定参数导出 YAML | `requirements.txt` |
| `scipy` | 1.18.1 | BSD-3-Clause（classifier） | 规划：M4 Savitzky-Golay（当前代码未 import） | `docs/计划书.md` |
| `torch` | 2.11.0+cu128 | BSD-3-Clause（metadata） | 可选：AI 后端（YOLO / SAM） | `requirements-ai.txt` |
| `torchvision` | 0.26.0+cu128 | BSD-3-Clause（metadata） | 可选：AI 后端 | `requirements-ai.txt` |
| `ultralytics` | 8.4.159 | AGPL-3.0（metadata） | 可选：YOLO / SAM（AGPL） | `requirements-ai.txt` |

> `scipy` 是 M4 规划项（当前代码未 import）；`torch/torchvision/ultralytics` 是可选 AI 后端；`python-pptx/Pillow/lxml` 只给 PPT 工具用。

## 2. 传递依赖（本环境已安装、从直接依赖可达，36）

| 组件 | 已装版本 | 许可证 | 引入者（已安装） |
| --- | --- | --- | --- |
| `certifi` | 2026.7.22 | MPL-2.0（metadata） | requests |
| `charset-normalizer` | 3.5.1 | MIT（metadata） | requests |
| `click` | 8.5.0 | BSD-3-Clause（curated） | scipy |
| `colorama` | 0.4.6 | BSD-3-Clause（classifier） | psutil, tqdm |
| `contourpy` | 1.4.0 | BSD-3-Clause（curated） | matplotlib |
| `cycler` | 0.12.1 | BSD-3-Clause（classifier） | matplotlib |
| `filelock` | 4.0.1 | Unlicense（curated） | setuptools, torch… |
| `fonttools` | 4.65.0 | MIT（metadata） | matplotlib |
| `fsspec` | 2026.9.0 | BSD-3-Clause（curated） | pandas, polars… |
| `idna` | 3.20 | BSD-3-Clause（curated） | requests |
| `jinja2` | 3.1.6 | BSD-3-Clause（classifier） | fsspec, pandas… |
| `kiwisolver` | 1.5.1 | BSD-3-Clause（classifier） | matplotlib |
| `lap` | 0.5.13 | BSD-2-Clause（metadata） | ultralytics |
| `markupsafe` | 3.0.3 | BSD-3-Clause（curated） | jinja2 |
| `mpmath` | 1.3.0 | BSD-3-Clause（metadata） | scipy, sympy |
| `munkres` | 1.1.4 | Apache-2.0（classifier） | fonttools |
| `networkx` | 3.6.1 | BSD-3-Clause（curated） | torch |
| `nvidia-ml-py` | 13.610.43 | BSD-3-Clause（metadata） | ultralytics |
| `packaging` | 26.3 | Apache-2.0 OR BSD-2-Clause（curated） | matplotlib, pillow… |
| `pandas` | 3.0.6 | BSD-3-Clause（metadata） | fsspec, networkx… |
| `pip` | 26.2.1 | MIT（curated） | setuptools |
| `polars` | 1.44.2 | MIT（metadata） | ultralytics |
| `polars-runtime-32` | 1.44.2 | MIT（metadata） | polars |
| `psutil` | 7.2.2 | BSD-3-Clause（metadata） | ultralytics |
| `pyparsing` | 3.3.3 | MIT（curated） | matplotlib |
| `python-dateutil` | 2.9.0.post0 | Apache-2.0 OR BSD-3-Clause（classifier） | matplotlib, pandas |
| `requests` | 2.34.2 | Apache-2.0（metadata） | fsspec, psutil… |
| `setuptools` | 81.0.0 | MIT（curated） | pillow, psutil… |
| `six` | 1.17.0 | MIT（metadata） | python-dateutil |
| `sympy` | 1.14.0 | BSD-3-Clause（metadata） | fonttools, networkx… |
| `tqdm` | 4.70.1 | MPL-2.0 AND MIT（metadata） | fsspec |
| `typing-extensions` | 4.16.0 | PSF-2.0（curated） | python-pptx, scipy… |
| `tzdata` | 2026.4 | Apache-2.0（metadata） | pandas, polars |
| `ultralytics-thop` | 2.1.6 | AGPL-3.0（metadata） | ultralytics |
| `urllib3` | 2.8.0 | MIT（curated） | fsspec, requests |
| `xlsxwriter` | 3.2.9 | BSD-2-Clause（metadata） | pandas, polars… |

## 3. 环境内但不在声明依赖闭包（12）

> 这些包装在这台机器上，但**从项目声明的直接依赖走不到**；多半是上游可选依赖（HF 下载、Qt 后端）或环境残留。项目代码不直接 import。

| 组件 | 已装版本 | 许可证 | 可能来源 |
| --- | --- | --- | --- |
| `anyio` | 4.15.1 | MIT（curated） | HF/httpx 可选链（环境残留） |
| `h11` | 0.16.0 | MIT（metadata） | httpx 的依赖（环境残留） |
| `hf-xet` | 1.6.0 | Apache-2.0（classifier） | HuggingFace 下载加速（AI 可选） |
| `httpcore` | 1.0.9 | BSD-3-Clause（classifier） | httpx 的依赖（环境残留） |
| `httpx` | 0.28.1 | BSD-3-Clause（metadata） | HuggingFace 客户端依赖（AI 可选） |
| `huggingface-hub` | 1.32.0 | Apache-2.0（metadata） | 模型下载（AI 可选） |
| `py-cpuinfo` | 9.0.0 | MIT（metadata） | 环境信息工具（timm/ultralytics 可选链） |
| `pyside6` | 6.11.2 | LGPL-3.0-only（curated） | conda Qt 绑定；matplotlib Qt 后端可能用 |
| `safetensors` | 0.8.0 | Apache-2.0（classifier） | 模型权重格式（AI 可选） |
| `shiboken6` | 6.11.2 | LGPL-3.0-only（curated） | PySide6 的绑定层 |
| `timm` | 1.0.29 | Apache-2.0（metadata） | torch 视觉模型库（ultralytics 可选链） |
| `tornado` | 6.5.10 | Apache-2.0（metadata） | Jupyter/异步依赖（环境残留） |

## 4. 原生 / 系统组件

| 组件 | 版本 | 许可证 | 范围 | 说明 |
| --- | --- | --- | --- | --- |
| CPython | 3.14.7 | PSF-2.0 | required | 解释器；绝对路径 `%USERPROFILE%\miniforge3\envs\myenv\python.exe` |
| FFmpeg（OpenCV wheel 内置） | prebuilt（版本未暴露） | LGPL-2.1-or-later OR GPL-2.0-or-later | required | OpenCV 的 FFMPEG 后端；许可证取决于 wheel 构建，分发前复核 |
| NVIDIA CUDA Runtime | 12.8 | NVIDIA proprietary (EULA) | optional | torch 2.11.0+cu128；AI 后端才需要 |
| NVIDIA cuDNN | 9.19.0 | NVIDIA proprietary (EULA) | optional | torch.backends.cudnn 报 91900 |
| NVIDIA Display Driver | 577.05 | NVIDIA proprietary (EULA) | required | RTX 5060 Laptop 8 GB；CUDA 12.8 的最低驱动 |
| Intel MKL | 2026.1.0 | Intel Simplified Software License | required | conda 环境里的 BLAS/LAPACK；numpy 使用 |
| Windows | 11 10.0.26300 | Proprietary | required | 操作系统 |

## 5. 许可证汇总

| 许可证 | 组件数 | 组件 |
| --- | --- | --- |
| BSD-3-Clause | 22 | `click`, `colorama`, `contourpy`, `cycler`, `fsspec`, `httpcore`, `httpx`, `idna`, `jinja2`, `kiwisolver`, `lxml`, `markupsafe`, `mpmath`, `networkx`, `numpy`, `nvidia-ml-py`, `pandas`, `psutil`, `scipy`, `sympy`, `torch`, `torchvision` |
| MIT | 14 | `anyio`, `charset-normalizer`, `fonttools`, `h11`, `pip`, `polars`, `polars-runtime-32`, `py-cpuinfo`, `pyparsing`, `python-pptx`, `pyyaml`, `setuptools`, `six`, `urllib3` |
| Apache-2.0 | 9 | `hf-xet`, `huggingface-hub`, `munkres`, `opencv-contrib-python`, `requests`, `safetensors`, `timm`, `tornado`, `tzdata` |
| AGPL-3.0 | 2 | `ultralytics`, `ultralytics-thop` |
| BSD-2-Clause | 2 | `lap`, `xlsxwriter` |
| LGPL-3.0-only | 2 | `pyside6`, `shiboken6` |
| Apache-2.0 OR BSD-2-Clause | 1 | `packaging` |
| Apache-2.0 OR BSD-3-Clause | 1 | `python-dateutil` |
| MIT-CMU | 1 | `pillow` |
| MPL-2.0 | 1 | `certifi` |
| MPL-2.0 AND MIT | 1 | `tqdm` |
| Matplotlib License (PSF-based) | 1 | `matplotlib` |
| PSF-2.0 | 1 | `typing-extensions` |
| Unlicense | 1 | `filelock` |

## 6. 风险与注意

- **AGPL-3.0**：`ultralytics`、`ultralytics-thop`。只在使用 `--method yolo*` 时进入运行链路；若对外分发包含它们的产物，AGPL 义务需要单独评估。
- **LGPL-3.0**：`pyside6`、`shiboken6`（conda Qt 绑定）。项目代码不直接 import；matplotlib 的 Qt 后端可能间接使用。分发二进制前按 LGPL 处理。
- **NVIDIA 专有**：CUDA / cuDNN / 驱动不可再分发，受 NVIDIA EULA 约束。
- **OpenCV wheel 内置 FFmpeg**：构建时可能启用 LGPL/GPL 组件；`cv2.getBuildInformation()` 只报 `FFMPEG: YES (prebuilt binaries)`，不暴露版本。
- **不要同时装 `opencv-python`**：本环境只有 `opencv-contrib-python`；两者都提供 `cv2`，会互相覆盖（`requirements-ai.txt` 已写明 `--no-deps`）。
- **许可证元数据缺口**：标 `curated` 的项是按上游常见许可证补的，不是包元数据原文。

## 7. 方法、边界与复现

- Python 包：`importlib.metadata` 读版本 / 许可证 / `Requires-Dist`；`pip freeze` 交叉核对。
- 直接依赖：`requirements.txt`、`requirements-ai.txt`、`docs/开发指南.md`（PPT 可选）、`docs/计划书.md`（scipy 规划）。
- 传递闭包：从直接依赖按 `Requires-Dist` 做 BFS，只统计本环境已安装的包。
- 原生组件：`cv2.getBuildInformation()`、`torch.version.cuda`、`torch.backends.cudnn.version()`、`nvidia-smi`、`environment.yml`。
- 复现：`pip freeze` + `environment.yml`；本文件是一次性生成的快照，**换机器 / 升级依赖后必须重新生成**。
- 边界：`environment.yml` 里还有约 90 个 conda 原生包（MKL / Qt / 字体等），本文只列了直接影响运行链路的原生组件；完整 conda 清单以 `environment.yml` 为准。

