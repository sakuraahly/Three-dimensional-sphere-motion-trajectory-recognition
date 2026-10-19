"""规范落差自查：文档 / 代码 / 规范之间"说的"与"实际"是否一致。

为什么需要它
------------
这个项目的头号反复问题是**漂移**：文档里写着一条规则或一个路径，
实际却早就不成立了。而在此之前，**没有任何东西守着它** ——
全靠某次审查时有人想起来去看一眼。已经抓到的实例：

- `AGENTS.md` 五处引用 `STC/...`，其中一个脚本
  （`STC\\scripts\\Check-Standards.ps1`）**在磁盘上根本不存在**；
- `AGENTS.md` 声明"基于 STC v0.3"，而 `..\\STC\\` 里的标准版本对不上；
- `用户手册.md` 承诺标定会存 `config/stereo_calibration.npz`，
  而标定判定不通过时**按设计就不保存**，失败路径文档里没写。

本工具把这些做成**可执行的三类检查**：

| 类别 | 查什么 |
| ---- | ------ |
| **A 引用** | 文档里反引号引用的仓库路径是否存在（**区分「引用」与「谈及」**） |
| **B 计数** | `AGENTS.md` §3/§7 的计数断言是否与磁盘一致（委托 `sync_facts.py`） |
| **C 来源** | `AGENTS.md` 引用的规范文件（`STC/...`）是否真的存在 |

分工（刻意如此，避免"同一个逻辑实现两遍"）
----------------------------------------
- **A 的权威实现在这里**；`tests/selftest_doc_facts.py` 改为调用本模块。
- **B 委托给 `apps/tools/sync_facts.py`** —— 那是计数的权威实现，不在这里重写一遍。
- `启动程序.bat standards` → 本模块。

边界
----
本工具**只读**，不改任何文件（计划书 §5.1 定死的职能）。

用法::

    python apps/tools/check_standards.py
    python apps/tools/check_standards.py --quiet     # 只出结论与失败项
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "sphere_tracker").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 放置规则里的顶层目录：只有落在这些下面的反引号路径才算"仓库路径引用"
KNOWN_TOP = {"sphere_tracker", "calib", "stereo", "apps", "tests", "docs", "assets",
             "config", "outputs", "models", "legacy", "STC"}
PATH_RE = re.compile(
    r"`([A-Za-z_][\w./\\-]*\.(?:py|md|json|npz|bat|txt|yml|ps1|pptx|jpg|png))"
    r"(?::(\d+)(?:-(\d+))?)?`")

# 计划书里指向"将要新建"的文件是正常的，但**不能整份豁免** ——
# 第一版把计划书整个放进 `SELF_DOCS`，结果它自己那 9 处指向"已被删掉的临时探针"
# 的引用**一声不响地过去了**（临时探针提升为常驻测试后就删了）。
# 教训：**豁免范围要按"具体引用"给，不能按"整份文档"给。**
SELF_DOC_NOTES = {
    "docs/planbooks/2026-10-10_架构接缝与缺失基础设施.md": (
        "计划书会引用「将要新建」的文件（`sphere_tracker/frames.py` 等），"
        "那是计划本身，不是漂移"),
}
# --------------------------------------------------------------------------
# 失效引用的白名单。键 = "<文档相对路径>:<引用的原文>"，**刻意不带行号**
#（行号随任何编辑漂移，用它做键会让白名单变成噪声源 —— 实测踩过，
#  13 条白名单里 4 条一改文档就失配、工具开始误报）。
# --------------------------------------------------------------------------
KNOWN_MENTIONS: dict[str, str] = {
    "AGENTS.md:_overflow_lib.py":
        "在讲「它曾经叫这个名字、后来改名了」—— 历史说明，不是引用",
    "docs/交接文档.md:apps/calibrate_camera.py":
        "在「原来写的 / 实际」表格里指认旧路径不存在",
    "docs/交接文档.md:rements.txt":
        "引述 cmd 行同步错乱时的实际报错碎词，不是路径",
    "docs/交接文档.md:_overflow_lib.py":
        "记录改名前后（`_overflow_lib.py` -> `check_overflow.py`）",
    "docs/开题报告_PPT逐页文稿.md:stereo/match.py":
        "代码表里明写「**没有** `stereo/match.py`」，是在澄清旧文稿",
    "docs/项目总纲.md:stereo/match.py":
        "同上，指认规划中的路径从未落地",
    "docs/开发指南.md:apps/calibrate_camera.py":
        "在「原来写的 / 实际」表格里（表内提及）",
    "docs/交接文档.md:session.json":
        "§12.6 在讲**计划中**的会话目录字段（P0-4 还没做），不是已存在的文件",
    # ---- 计划书：**按具体引用豁免，不整份放过** ----
    # 第一版把计划书整个放进 `SELF_DOCS`，结果它自己 10 处失效引用
    # （含指向"已被删掉的临时探针"的）**一声不响地过去了**。
    # 教训：豁免范围要按"具体引用"给，不能按"整份文档"给。
    "docs/planbooks/2026-10-10_架构接缝与缺失基础设施.md:sphere_tracker/frames.py":
        "P0-1 **计划新建**的模块（时间层），尚未实现",
    "docs/planbooks/2026-10-10_架构接缝与缺失基础设施.md:frames.py":
        "同上，§9 执行记录里的简写",
    "docs/planbooks/2026-10-10_架构接缝与缺失基础设施.md:stereo/sessions.py":
        "P0-4 **计划新建**的模块（会话读写），尚未实现",
    "docs/planbooks/2026-10-10_架构接缝与缺失基础设施.md:sessions.py":
        "同上，§5.1 职能表里的简写",
    "docs/planbooks/2026-10-10_架构接缝与缺失基础设施.md:session.json":
        "P0-4 计划中的会话元数据文件名",
    "docs/planbooks/2026-10-10_架构接缝与缺失基础设施.md:apps/tools/zzz_tmp.py":
        "§9.5 反向自证的**注入示例**，举例说明用的假文件名",
    "docs/planbooks/2026-10-10_架构接缝与缺失基础设施.md:New-Project.ps1":
        "§2.1 在陈述「`STC/scripts/` 里只有 New-Project.ps1」——是在讲仓库外有什么",
    "docs/planbooks/2026-10-10_架构接缝与缺失基础设施.md:STC\\scripts\\Check-Standards.ps1":
        "§2.1 在陈述**这个脚本不存在**（这是那条发现的结论本身）",
    "docs/planbooks/2026-10-10_架构接缝与缺失基础设施.md:config/stereo_calibration.npz":
        "§2.1 在陈述「`config/` 里**从来没有过**这两个文件」",
    # ---- 2026-10-19 离线双目视频计划：P0/P1 **尚未实现**的入口 ----
    "docs/planbooks/2026-10-19_离线双目视频与标定导出.md:apps/run_stereo_video.py":
        "P0 **计划新增**的离线双目入口，尚未实现",
    "docs/planbooks/2026-10-19_离线双目视频与标定导出.md:apps/record_stereo.py":
        "P1 **计划新增**的双路录制入口，等相机接上再做",
    "docs/planbooks/2026-10-19_离线双目视频与标定导出.md:session.json":
        "P1 计划中的会话元数据文件名",
    "docs/planbooks/2026-10-19_数据采集与结构整理.md:apps/record_stereo.py":
        "P1 **计划新增**的双路录制入口，等相机接上再做",
    "docs/planbooks/2026-10-19_数据采集与结构整理.md:apps/run_stereo_video.py":
        "P2 **计划新增**的离线双目入口，尚未实现",
    "docs/planbooks/2026-10-19_相机接入与数据采集.md:apps/record_stereo.py":
        "P2 **计划新增**的双路录制入口，相机已接但尚未实现",
    "docs/planbooks/2026-10-19_相机接入与数据采集.md:apps/run_stereo_video.py":
        "P3 **计划新增**的离线双目入口，尚未实现",
    "docs/planbooks/2026-10-19_相机接入与数据采集.md:world_frame.json":
        "P6 计划中的世界系变换文件，尚未实现",
    "docs/planbooks/2026-10-10_架构接缝与缺失基础设施.md:stereo_calibration.npz":
        "同上（`AGENTS.md:STC` 之外的简写形式）",
    "docs/planbooks/2026-10-10_架构接缝与缺失基础设施.md:stereo_calibration.json":
        "同上",
    "README.md:video_edge_detect.py": "在「历史遗留脚本」说明表格里",
    "README.md:test.py": "同上",
    "docs/开题报告_研究思路四页文稿.md:calib/calibrate_stereo.py":
        "在路径对照表里（旧路径 vs 实际路径）",
    "docs/开题报告_研究思路四页文稿.md:stereo/match.py":
        "同表的旧路径列（模块路径 -> 实际为 apps/...）",
    # ---- config/ 是**运行期可再生产物**，被"谈及"不等于被承诺 ----
    #
    # `config/camera_modes.json` 是"上次探测出的后端+分辨率"的备忘录：
    # 只在启用 `CameraSource(optimize=True)`（实时单目那条路）时才生成。
    # 2026-10-14 起**立体路径刻意不生成它**（`PairSource` 传 `optimize=False`，
    # 因为"各自优化"会把两路分辨率拉开、让标定直接失败），
    # 所以这个文件可能**合理地不存在**。而文档里提到的三处都不是"承诺它存在"：
    # 一处讲打开很重、一处是**历史记录**（"一度被钉在 640x480"）、
    # 一处是"参考基线（缓存随插口/协商变化，必须当场复测）"。
    # 教训与 `KNOWN_MENTIONS` 开头那句一致：**"引用"与"谈及"要分开**，
    # 而"可再生的运行期产物"属于谈及。
    "docs/API.md:config/camera_modes.json": "讲 CameraSource 打开很重（缓存的用途），非承诺存在",
    "docs/交接文档.md:config/camera_modes.json": "坑 35 的**历史记录**（一度被钉在 640x480）",
    "docs/用户手册.md:config\\camera_modes.json": "「参考基线」并自己写明缓存会变、须当场复测",
}

# --------------------------------------------------------------------------
# 已知**真漂移**：文档承诺了却没发生。单独列出并断言它们仍然存在 ——
# 修好之后应当变空，而不是被悄悄从白名单里删掉。
#
# ✅ 2026-10-15：**这一节已经清空，头一次。**
# 2026-10-14 23:08 用户真机标定**成功并落盘**（`config/stereo_calibration.npz` +
# `.json`，基线 65.81 mm、判定"勉强可用 —— 已保存"），于是那两条
# 「文档承诺了却没发生」的 `KNOWN_DRIFT` **不再出现** —— 检查器自己报了出来：
#
#     已知真漂移里有条目已不再出现（修好了？）——请从 KNOWN_DRIFT 里删掉，
#     否则它会变成永久豁免
#
# 这正是这个白名单设计成"必须手动删"的意义：**它不会自己变成永久豁免**。
# 保留这段文字是为了让下一个人知道：这里曾经挂过两条，以及它们是怎么消掉的。
# --------------------------------------------------------------------------
KNOWN_DRIFT: dict[str, str] = {}

# 指向**仓库外**（`..\STC\`，属用户的其他资产）：文件在磁盘上确实存在，
# 只是不在本仓库内 —— 单独归类、如实计数。这是 `AGENTS.md` §6 规范第 3 层的边界。
OUTSIDE_REPO_MARK = "STC"


def doc_files() -> list[Path]:
    return [ROOT / "AGENTS.md", ROOT / "README.md",
            *sorted((ROOT / "docs").rglob("*.md"))]


def _src_literals() -> set[str]:
    """仓库里作为字符串字面量出现过的路径（运行时生成的，如 config/sphere_config.json）。

    ⚠️ **必须排除 `outputs/`**（以及 `__pycache__` / `.git`）：
    `outputs/` 是"删了能再生"的临时目录，里面全是调试脚本与探针。
    第一版扫了它，结果**我自己写在 `outputs/` 里的一行调试代码，
    就让检查器把那个假路径当成"合法运行时路径"放行了** —— 反向自证因此静默失效。
    **临时产物不能当事实来源。**
    """
    out: set[str] = set()
    for py in ROOT.rglob("*.py"):
        parts = set(py.parts)
        if parts & {".git", "__pycache__", "outputs", "legacy"}:
            continue
        for m in re.finditer(
                r"""["']([^"'\n]{3,120}\.(?:py|md|json|npz|bat|txt|yml|ps1|pptx|jpg|png))["']""",
                py.read_text(encoding="utf-8", errors="replace")):
            out.add(m.group(1).replace("\\", "/").lstrip("./"))
    return out


def collect_stale_refs() -> dict[str, str]:
    """扫全库文档，返回 {「文档:引用原文」: 「L行号: 该行」}。"""
    literals = _src_literals()
    found: dict[str, str] = {}
    for doc in doc_files():
        rel = doc.relative_to(ROOT).as_posix()
        for lineno, line in enumerate(
                doc.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            for m in PATH_RE.finditer(line):
                raw = m.group(1)
                cand = raw.replace("\\", "/").lstrip("./")
                parts = cand.split("/")
                if len(parts) > 1 and parts[0] not in KNOWN_TOP:
                    continue
                if (ROOT / cand).exists() or cand in literals:
                    continue
                if len(parts) == 1:
                    hits = [p for p in ROOT.rglob(parts[0])
                            if p.is_file() and ".git" not in p.parts
                            and "__pycache__" not in p.parts]
                    if hits:
                        continue
                found[f"{rel}:{raw}"] = f"L{lineno}: {line.strip()[:100]}"
    return found


# --------------------------------------------------------------------------
def check_refs(verbose: bool) -> list[str]:
    """A 类：引用 vs 谈及。"""
    print("----- A. 文档里的仓库路径引用 -----")
    found = collect_stale_refs()
    outside = {k: v for k, v in found.items() if OUTSIDE_REPO_MARK in k.split(":")[1]}
    inside = {k: v for k, v in found.items() if k not in outside}
    known = set(KNOWN_MENTIONS) | set(KNOWN_DRIFT)
    unexpected = {k: v for k, v in inside.items() if k not in known}

    print(f"  扫到 {len(found)} 处失效引用 = 仓库外 {len(outside)} + 仓库内 {len(inside)}")
    if verbose:
        for k in sorted(outside):
            print(f"    [仓库外] {k}")
    print(f"  仓库内 {len(inside)} 处：白名单（有意保留的历史信息）"
          f"{len(KNOWN_MENTIONS)} 条、已知真漂移 {len(KNOWN_DRIFT)} 条")
    for k in sorted(KNOWN_DRIFT):
        print(f"    [真漂移，待修] {k}")

    problems: list[str] = []
    if unexpected:
        problems.append(f"白名单之外出现 {len(unexpected)} 处新失效引用：")
        for k, v in sorted(unexpected.items()):
            problems.append(f"    · {k}\n        {v}")
    vanished = [k for k in KNOWN_DRIFT if k not in found]
    if vanished:
        # 修好了是好消息，但必须**把白名单同步删掉**，否则它变成"永久豁免"
        problems.append("已知真漂移里有条目已不再出现（修好了？）——"
                        "请从 check_standards.py 的 KNOWN_DRIFT 里删掉，"
                        "否则它会变成永久豁免：\n"
                        + "\n".join(f"    · {k}" for k in vanished))
    return problems


def check_sources(verbose: bool) -> list[str]:
    """C 类：`AGENTS.md` 引用的规范文件是否真的存在。"""
    print("----- C. 规范来源是否存在 -----")
    text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    stc_refs = sorted(set(re.findall(r"`(STC[/\\][^`]+)`", text)))
    problems: list[str] = []
    # `STC/` 在仓库外（`..\STC\`），所以按**父目录**去找
    outside_root = ROOT.parent
    for ref in stc_refs:
        cand = outside_root / ref.replace("\\", "/")
        ok = cand.exists()
        print(f"  [{'存在' if ok else '**不存在**'}] {ref}"
              f"   -> {cand.relative_to(outside_root.parent).as_posix()}")
        if not ok:
            problems.append(
                f"AGENTS.md 引用了不存在的规范文件：`{ref}`\n"
                f"        期望位置：{cand}\n"
                f"        修法：删掉这处引用，或把 STC 补上（STC 在仓库外，属用户其他资产）")
    if not stc_refs:
        print("  （AGENTS.md 里没有 STC/ 引用）")
    return problems


def check_facts(verbose: bool) -> list[str]:
    """B 类：计数与清单 —— **委托给权威实现**，不在这里重写一遍。"""
    print("----- B. 派生事实（计数 / 清单） -----")
    script = ROOT / "apps" / "tools" / "sync_facts.py"
    if not script.exists():
        return [f"找不到 {script.relative_to(ROOT).as_posix()} —— "
                f"计数检查没有权威实现可委托"]
    proc = subprocess.run([sys.executable, str(script), "--check"],
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", cwd=str(ROOT))
    out = (proc.stdout or "") + (proc.stderr or "")
    if verbose or proc.returncode != 0:
        for ln in out.splitlines():
            print(f"    {ln}")
    else:
        for ln in out.splitlines():
            if ln.startswith("✅") or "判定结果" in ln:
                print(f"    {ln}")
    if proc.returncode == 0:
        return []
    fails = [ln.strip() for ln in out.splitlines()
             if ln.strip().startswith(("1)", "2)", "3)", "4)", "5)"))]
    return ["派生事实检查未通过（详见上面 sync_facts.py 的输出）"
            + ("：\n" + "\n".join(f"    {x}" for x in fails) if fails else "")]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="规范落差自查（只读）")
    ap.add_argument("--quiet", action="store_true", help="少打细节")
    args = ap.parse_args(argv)
    verbose = not args.quiet

    print("=" * 78)
    print("规范落差自查（文档 / 代码 / 规范 三者是否自洽）")
    print("=" * 78)
    print(f"仓库：{ROOT.name}")
    print()

    problems: list[str] = []
    problems += check_refs(verbose)
    print()
    problems += check_sources(verbose)
    print()
    problems += check_facts(verbose)
    print()

    print("=" * 78)
    if problems:
        print(f"发现 {len(problems)} 类问题：")
        for i, msg in enumerate(problems, 1):
            print(f"  {i}) {msg}")
        print()
        print("判定结果：有不达标项")
        return 1
    print("三类检查全部通过：引用、来源、计数都与磁盘一致。")
    print("判定结果：全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
