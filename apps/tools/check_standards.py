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
- `run.bat standards` → 本模块。

边界
----
本工具**只读**，不改任何文件（计划书 §5.1 定死的职能）。

用法::

    python apps/tools/check_standards.py
    python apps/tools/check_standards.py --quiet     # 只出结论与失败项
"""

from __future__ import annotations

import argparse
import functools
import re
import subprocess
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "sphere_tracker").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 放置规则里的顶层目录：只有落在这些下面的反引号路径才算"仓库路径引用"
#
# ⚠️ `data` **故意不在这里**（2026-11-11 明确下来，以前是"碰巧不在"）：
# `data/` 装的是用户自己录的实验数据，会话目录会被用户随时删/移/归档 ——
# 把 `data/...` 当"文档对仓库的承诺"来校验，等于每删一段录像就打红一次。
# 这与 `config/` / `outputs/` 归 `RUNTIME_REPRO_MARK` 是同一个道理：
# **运行期产物不是文档承诺**。所以文档里写 `data/` 路径既不会被校验、也不会
# 因此报错；真要保证数据目录结构，得靠 `sphere_tracker/discovery.py` 那类
# 代码判据，不是靠文档字符串。
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
    # ✅ 2026-10-24：**清掉了 23 条死条目。** 原因是它们承诺的文件真的补上了：
    # `apps/record_stereo.py`、`apps/run_stereo_video.py`、`apps/view_3d.py`、
    # `apps/_workbench.py`、`docs/3D查看器.md`、`sphere_tracker/detectors/specular.py`、
    # `config/camera_modes.json`、`config/stereo_calibration.npz` 都已落盘
    #（标定那两条是 2026-10-14 真机成功保存）。既然引用不再失效，豁免就没有对象了 ——
    # 留着就是**永久豁免**。这也是从这版起 `KNOWN_MENTIONS` 有了 vanished 检测的原因：
    # 它和 `KNOWN_DRIFT` 一样，**必须有人来删，不许自己沉默**。
    # ⚠️ 2026-10-24 二次修正：其中 `config/` 的那几条**删错了** —— 它是可再生产物，
    # "文件此刻存在"不代表"引用是承诺"。已改由 `RUNTIME_REPRO_MARK` 按**路径类别**
    # 分类（见上方注释）。存不存在都不影响结论，所以不需要、也不允许回到白名单里。
    # ---- 历史说明 / 表格里指认旧路径（"谈及"而非"承诺"） ----
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
    # ---- 计划书：**按具体引用豁免，不整份放过** ----
    # 第一版把计划书整个放进 `SELF_DOCS`，结果它自己 10 处失效引用
    # （含指向"已被删掉的临时探针"的）**一声不响地过去了**。
    # 教训：豁免范围要按"具体引用"给，不能按"整份文档"给。
    "docs/planbooks/2026-10-10_架构接缝与缺失基础设施.md:sphere_tracker/frames.py":
        "P0-1 **计划新建**的模块（时间层），尚未实现",
    "docs/planbooks/2026-10-10_架构接缝与缺失基础设施.md:frames.py":
        "同上，§9 执行记录里的简写",
    "docs/planbooks/2026-10-10_架构接缝与缺失基础设施.md:apps/tools/zzz_tmp.py":
        "§9.5 反向自证的**注入示例**，举例说明用的假文件名",
    "docs/planbooks/2026-10-10_架构接缝与缺失基础设施.md:New-Project.ps1":
        "§2.1 在陈述「`STC/scripts/` 里只有 New-Project.ps1」——是在讲仓库外有什么",
    # ---- 2026-11-11 计划书「目录化数据管理」P2：**已落地，豁免随之删除** ----
    # 曾经在这里的四条（`docs/planbooks/2026-11-09_目录化数据管理.md` 与
    # `docs/交接文档.md` 各两条，分别指向 `apps/package_export.py` 与
    # `package.json`）在 P2 打包层落地当天删掉了 —— 豁免的对象已经存在，
    # 留着就是**永久豁免**（工具自带的 vanished 检测就是为逼这一步写的）。
    # 这条注释是给后来者的说明，**不是豁免**，别照着它加回去。
    # ---- 2026-10-27 计划书：**按具体引用豁免**（同理，不整份放过） ----
    # 这些是本轮「单目主线与 GUI 分窗」计划 §2.5 / §3 / §5 列明要新建的文件，
    # 目前尚不存在 —— 计划的引用对象是"将要新建"，不是漂移。
    # ⚠️ 工具自带 vanished 检测：这些文件一旦落盘，本段必须删掉，
    # 否则就退化成"永久豁免"（2026-10-24 已清过 23 条，别重犯）。
    # ---- 2026-10-29：这份计划书列的文件**全部落地**，本段豁免清空 ----
    # 曾经在这里的有：_task_runner / _gui_widgets / _gui_app / _gui_console /
    # _gui_mono / _gui_stereo / _gui_bench / calibrate_mono / mono_calib /
    # selftest_mono_calib / tests/selftest_gui_budget，以及 capture_board 的
    # `_capture.json`。每落地一个就必须删一条（工具自带的 vanished 检测会逼你删），
    # 否则白名单退化成"永久豁免" —— 2026-10-24 已清过 23 条，别重犯。
    # 留这段说明是提醒：**计划书里的"计划新建"不该再进白名单**，
    # 要豁免也只能按具体引用给、并在文件落地当天删掉。
    # ---- 2026-10-28：`_capture.json` **已实现**（P0b），豁免条目随之删除 ----
    # 它原先在这里有一条"§3.1 计划新建"的豁免。`capture_board.write_capture_sidecar`
    # 落地后该豁免变成**死条目**，`selftest_doc_facts` 第 3/4/5 节连锁变红
    # （死条目让"删掉注入引用后应恢复通过"也失败）。
    # 这条注释是给后来者的说明，不是豁免 —— 别照着它再往白名单里加回去。
    # ---- 仓库外（`STC/`）：两处都是在**陈述"这个脚本不存在"** ----
    # 2026-10-24 前 `STC/` 引用从不进 problems，这条豁免等于空转；
    # 现在按 `ROOT.parent` 解析后它真的会被判失效，所以必须给出理由。
    "docs/planbooks/2026-10-10_架构接缝与缺失基础设施.md:STC\\scripts\\Check-Standards.ps1":
        "§2.1 在陈述**这个脚本不存在**（这是那条发现的结论本身）",
    "docs/交接文档.md:STC/scripts/Check-Standards.ps1":
        "§历史小记在讲「`AGENTS.md` 原先写着…并称它负责落差检测」，是在归档旧说法",
    # ---- 世界系变换文件：2026-10-25 起**已实现**，豁免条目随之作废 ----
    # `calib/world_frame.py` + `apps/tools/set_world_frame.py` 落地后，
    # `world_frame.json` 出现在 .py 的字面量里（stereo/visualization.py 的默认路径等），
    # `_src_literals()` 会把"源码里写过的路径"视为合法运行时路径 ——
    # 于是这 4 条豁免永远不再命中，留着只会让检查器报"死白名单"。
    # 记一笔原因，免得以后有人以为是漏配。
    "README.md:video_edge_detect.py": "在「历史遗留脚本」说明表格里",
    "README.md:test.py": "同上",
    "docs/开题报告_研究思路四页文稿.md:calib/calibrate_stereo.py":
        "在路径对照表里（旧路径 vs 实际路径）",
    "docs/开题报告_研究思路四页文稿.md:stereo/match.py":
        "同表的旧路径列（模块路径 -> 实际为 apps/...）",
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

# --------------------------------------------------------------------------
# **运行期可再生产物**（`config/`）：刻意单独归类，**显式报出但绝不阻断**。
#
# 依据三条，缺一不可：
# 1. `.gitignore:59-73` 把 `config/` **整个目录**忽略，只跟踪 `config/.gitkeep` ——
#    它压根不进版本库；
# 2. `AGENTS.md` §3 规则 3：**"能重新生成的东西不进版本库"（`outputs/`、`config/`）**；
# 3. 与 `_src_literals()` 里"运行时生成的路径属于谈及、不属于承诺"是**同一口径**。
#
# 为什么必须按**路径类别**豁免、而不是按单条 mention 豁免：
# 2026-10-24 那轮把 3 条 `config/camera_modes.json` 的白名单条目当成"死条目"删了
# （只因那一刻文件恰好存在），于是**一份可再生产物被升级成了硬承诺** ——
# 实测 `mv config/camera_modes.json` 后本工具报"实际失效 3 处"、exit 1。
# 换台机器、清一次缓存、重标一次，就会误红。豁免挂在单条 mention 上时，
# 它的寿命取决于"此刻磁盘上有没有那个文件"，这本身就是漂移。
#
# 2026-10-27 扩到 `outputs/`：理由完全一样 —— 它同样**整目录不入库**
# （`AGENTS.md` §3 规则 3），所以任何机器上都没有义务存在。文档里引它是在说
# "去哪儿找证据"，不是承诺"这个文件必须在"。实测：干净 checkout 里
# `docs/团队角色与派发记录.md` 等引了 24 处 `outputs/...`，全部被判失效 →
# `check_standards` 红 → `selftest_doc_facts` 跟着红。
# --------------------------------------------------------------------------
RUNTIME_REPRO_MARK = ("config", "outputs")


def doc_files() -> list[Path]:
    """要检查的 Markdown 清单。

    ⚠️ **不要按名字排除临时探针**（2026-10-26 的教训）：
    `selftest_doc_facts` 做"反向自证"时，会在 `docs/` 下造一个 `_zz_tmp_*_probe.md`
    再跑本检查、期望被抓到。按名字过滤会把**它的探针也一起屏蔽**，
    自证于是变成"永远通过" —— 我为了让并发不误报而加的那行过滤，恰好干掉了这件事。

    正确处理是：**容忍扫描期间文件消失**（见 `_read_text`），而不是不去看它。
    """
    return [ROOT / "AGENTS.md", ROOT / "README.md",
            *sorted((ROOT / "docs").rglob("*.md"))]


def _read_text(path: Path) -> str | None:
    """读文件；扫描期间被别的进程删掉就返回 None（不是错误）。

    并发场景：`selftest_doc_facts` 建/删探针与 `check_standards` 枚举同时发生，
    glob 拿到名字、读取时文件已经没了。这不是"规范漂移"，是竞态。
    """
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


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


def _outside_path(cand: str) -> Path | None:
    """`STC/...` 引用 → 它在**仓库外**的真实位置；不是 `STC/` 引用则返回 None。

    `STC/` 是用户的其他资产，实际在 `..\\STC\\`（即 `ROOT.parent` 下）。
    只在仓库内找它必然找不到 —— 这就是 P2 那个假阳性的根因。
    这里与 `check_sources()` 用**同一个**解析根，避免"同一件事两套口径"。
    """
    parts = cand.split("/")
    if parts[0] != OUTSIDE_REPO_MARK:
        return None
    return ROOT.parent / cand


def _iter_refs():
    """逐条产出文档里的路径引用：(文档相对路径, 引用原文, 行号, 该行, 候选路径, 分段)。"""
    for doc in doc_files():
        text = _read_text(doc)
        if text is None:            # 扫描期间被删（并发跑测试的探针）—— 跳过，不算问题
            continue
        rel = doc.relative_to(ROOT).as_posix()
        for lineno, line in enumerate(
                (text or "").splitlines(), 1):
            for m in PATH_RE.finditer(line):
                raw = m.group(1)
                cand = raw.replace("\\", "/").lstrip("./")
                parts = cand.split("/")
                if len(parts) > 1 and parts[0] not in KNOWN_TOP:
                    continue
                yield rel, raw, lineno, line, cand, parts


def _is_runtime_reproducible(parts: list[str]) -> bool:
    """候选路径首段是 `config/` / `outputs/` → **运行期可再生产物**，不参与"承诺/失效"判定。"""
    return parts[0] in RUNTIME_REPRO_MARK


@functools.lru_cache(maxsize=1)
def _repo_basenames() -> frozenset[str]:
    """仓库内所有**文件名**的集合（一次遍历，供"裸文件名引用"查表）。

    ## 为什么要有它（2026-10-27 实测）

    原实现是"每遇到一条单段引用就 `ROOT.rglob(name)` 走一遍全仓"。
    文档里这类引用有几百条，于是**一次检查要递归 695 遍、调 16 万次 `scandir`**，
    光 `nt.scandir` 就 9.6s，整个 `check_standards` 跑一次 **15.9s** ——
    而它只是在回答"仓库里有没有叫这个名字的文件"。

    更贵的是它被 `selftest_doc_facts` **调了 10 次**（每次都重启进程，缓存不跨进程），
    于是那一个测试占了全套 125s（全套 600s 的 1/5）。

    改法：**建一次索引，之后都是集合查找**。判定语义完全不变
    （原代码也只是看 `hits` 是否非空），成本从 O(引用数 × 仓库)**降到 O(仓库)**。
    """
    names: set[str] = set()
    for p in ROOT.rglob("*"):
        parts = p.parts
        if ".git" in parts or "__pycache__" in parts:
            continue
        if p.is_file():
            names.add(p.name)
    return frozenset(names)


def collect_refs() -> tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, str]]:
    """扫一遍文档，返回 (失效引用, 仓库外但确实存在的引用, **本机没有的外部资产**, 运行期可再生产物)。

    四者的差别就是几次"检查器自己不可移植"的修法：

    - **存在的不算失效**，只有"不存在"的才进第一个返回值 —— 于是
      "扫到的失效数"与"进入 problems 的条数"天然同口径。
    - `config/`（第三变第四类）**先于存在性判定**分流，所以它**永远**不进第一个返回值：
      文件在不在都不影响结论（2026-10-24 的回归就是这么修的）。
    - **"整棵外部资产不在本机" ≠ "引用失效"**（2026-10-27，B-27）：
      `STC/` 在仓库外，别人的机器 / CI runner / 全新 checkout 上**整棵树都不在**。
      第一版把这 4 处当作失效 → 干净副本里 `check_standards` 直接红 →
      `selftest_doc_facts` 跟着红（实测"干净副本 4 红"之一）。
      判据应当是"**树在、文件没了**"才算漂移；树根本不在，属于环境事实，如实归类即可。
    """
    literals = _src_literals()
    found: dict[str, str] = {}
    outside_present: dict[str, str] = {}
    outside_absent: dict[str, str] = {}
    runtime_repro: dict[str, str] = {}
    outside_tree = ROOT.parent / OUTSIDE_REPO_MARK
    for rel, raw, lineno, line, cand, parts in _iter_refs():
        key = f"{rel}:{raw}"
        detail = f"L{lineno}: {line.strip()[:100]}"
        outside = _outside_path(cand)
        if outside is not None:
            # 仓库外：按父目录解析。存在 → 归类"仓库外（存在）"，**不计入失效**；
            # 整棵树不在 → 归类"本机没有该外部资产"，**也不计入失效**；
            # 只有"树在、文件没了"才计入失效（白名单之外的会进 problems）。
            if outside.exists():
                outside_present[key] = outside.relative_to(ROOT.parent).as_posix()
            elif not outside_tree.is_dir():
                outside_absent[key] = cand
            else:
                found[key] = detail
            continue
        if _is_runtime_reproducible(parts):
            # 可再生产物：**不做存在性判定**，直接归类、如实计数、不进 problems。
            runtime_repro[key] = cand
            continue
        if (ROOT / cand).exists() or cand in literals:
            continue
        # 文件不在磁盘上（干净 checkout / CI）：先问 git 它是不是被 .gitignore
        # 明确排除的本机资产（planbooks / outputs 报告 / 散落产物）—— 这类文件
        # 本就不入库，不构成"承诺存在的仓库文件"，归入可再生产物不进 stale。
        # 顺序很关键：本地存在时上面已 continue，绝不走到这里 —— 否则每个引用
        # 都要 spawn 一次 git（实测 3 分钟），CI 上只有少量缺失路径才需判断。
        if _git_ignored(cand):
            runtime_repro[key] = cand
            continue
        if len(parts) == 1 and parts[0] in _repo_basenames():
            continue
        found[key] = detail
    return found, outside_present, outside_absent, runtime_repro


def collect_stale_refs() -> dict[str, str]:
    """扫全库文档，返回 {「文档:引用原文」: 「L行号: 该行」}（= 第一个返回值）。"""
    return collect_refs()[0]


def outer_mentions(stale: dict[str, str]) -> dict[str, str]:
    """从失效集合里挑出"仓库外（`STC/`）且不存在"的那些。"""
    return {k: v for k, v in stale.items()
            if OUTSIDE_REPO_MARK in k.split(":")[1].replace("\\", "/").split("/")[0]}


# --------------------------------------------------------------------------
def dead_whitelist_entries(stale: dict[str, str]) -> list[str]:
    """`KNOWN_MENTIONS` 里"已不再命中任何失效引用"的条目。

    **唯一实现**：`check_refs()` 与 `tests/selftest_doc_facts.py` 都调这里 ——
    这条规则第一次是各写一份的（测试自己又算了一遍），于是"树不在本机"那个例外
    只修进了一边，干净副本里测试照样红。规则有两份 = 迟早对不上。

    例外：白名单里指向**仓库外资产**（`STC/`）的条目，在"整棵树不在本机"时
    本来就无从命中（它被归进 `outside_absent`，不是 `stale`）。
    把它们判成"死条目"等于要求"这台机器必须装 STC" —— 又一条环境断言。
    """
    dead = sorted(k for k in KNOWN_MENTIONS if k not in stale)
    if not dead:
        return dead
    # 两类条目"没命中 stale"并非因为漂移已修好，而是被归到别的桶：
    #   1) 指向仓库外 STC/ 且整棵树不在本机 → outside_absent（原有例外）；
    #   2) 指向被 .gitignore 排除的本机资产/产物 → runtime_repro（2026-11-22 CI 首跑新增）。
    # 把它们判成"死条目"等于要求"这台机器必须入库这些文件"，又一条环境断言。
    outside_keys = [
        k for k in dead
        if OUTSIDE_REPO_MARK in k.split(":")[1].replace("\\", "/").split("/")[0]]
    if outside_keys and not (ROOT.parent / OUTSIDE_REPO_MARK).is_dir():
        print(f"  （本机没有 {OUTSIDE_REPO_MARK}/ 树，白名单里 {len(outside_keys)} 条"
              f"指向它的条目无从命中，不算「死条目」）")
    elif not (ROOT.parent / OUTSIDE_REPO_MARK).is_dir():
        outside_keys = []  # 树不在但白名单里没有 STC 条目：别误伤
    def _real_doc_entry(k: str) -> bool:
        # 测试会就地注入一条假白名单（源文档根本不存在）来探 vanished 检测是否
        # 还活着；对那种假 key 不做任何豁免，否则反向自证失效。
        doc, _, ref = k.partition(":")
        return (ROOT / doc).exists() and _git_ignored(ref)

    ignored_keys = [
        k for k in dead if k not in outside_keys and _real_doc_entry(k)]
    if ignored_keys:
        print(f"  （白名单里 {len(ignored_keys)} 条指向被 .gitignore 排除的本机资产，"
              f"已归入「不入库产物」，不算「死条目」）")
    return [k for k in dead if k not in outside_keys and k not in ignored_keys]


def check_refs(verbose: bool) -> list[str]:
    """A 类：引用 vs 谈及。

    ⚠️ **口径必须唯一**：这里打印的"实际失效引用 N 处"就是进入 problems 的条数。
    第一版把 `outside` 算出来却从不放进 problems，于是出现
    "报告 23 处失效 / 同时判定全部通过"这种自相矛盾的输出 —— 守漂移的工具自己漂了。
    """
    print("----- A. 文档里的仓库路径引用 -----")
    stale, outside_present, outside_absent, runtime_repro = collect_refs()
    outside_missing = outer_mentions(stale)
    inside = {k: v for k, v in stale.items() if k not in outside_missing}
    known = set(KNOWN_MENTIONS) | set(KNOWN_DRIFT)
    unexpected = {k: v for k, v in stale.items() if k not in known}
    exempted = len(stale) - len(unexpected)

    print(f"  STC/ 引用（仓库外，按 {ROOT.parent.name}/ 解析）："
          f"存在 {len(outside_present)} 处（不计失效）、不存在 {len(outside_missing)} 处（计入失效）")
    if outside_absent:
        print(f"  本机没有整棵 {OUTSIDE_REPO_MARK}/（{ROOT.parent.as_posix()}）—— "
              f"{len(outside_absent)} 处引用**如实归类但不计失效**"
              f"（别人的机器/CI 上本来就没有这棵树，不是漂移）")
    if verbose:
        for k in sorted(outside_present):
            print(f"    [仓库外（存在）] {k} -> {outside_present[k]}")
        for k in sorted(outside_missing):
            print(f"    [仓库外（不存在）] {k}\n        {outside_missing[k]}")
    print(f"  运行期可再生产物（config/，不阻断）{len(runtime_repro)} 处"
          f" —— 由运行/重标生成、整目录不入库，缺了不算漂移")
    if verbose:
        for k in sorted(runtime_repro):
            print(f"    [可再生产物] {k} -> {runtime_repro[k]}")
    print(f"  候选失效引用 {len(stale)} 处 = 仓库外（不存在）{len(outside_missing)}"
          f" + 仓库内 {len(inside)}")
    # 这个数字**不因 config 降级而变**：那批引用改动前就靠"文件恰好存在"通过，
    # 改动后靠"路径类别"通过 —— 都是不进 stale。真正的差别只在**文件缺失时**：
    # 改动前会涨（于是误红），改动后恒定。
    print(f"  其中白名单豁免 {exempted} 处（KNOWN_MENTIONS {len(KNOWN_MENTIONS)} 条命中 "
          f"{len(set(KNOWN_MENTIONS) & set(stale))} 条、已知真漂移 {len(KNOWN_DRIFT)} 条）")
    print(f"  实际失效引用 {len(unexpected)} 处 —— 与 problems 同口径（这些全部进入 problems）")
    for k in sorted(KNOWN_DRIFT):
        print(f"    [真漂移，待修] {k}")

    problems: list[str] = []
    if unexpected:
        problems.append(f"白名单之外出现 {len(unexpected)} 处新失效引用：")
        for k, v in sorted(unexpected.items()):
            problems.append(f"    · {k}\n        {v}")
    vanished = [k for k in KNOWN_DRIFT if k not in stale]
    if vanished:
        # 修好了是好消息，但必须**把白名单同步删掉**，否则它变成"永久豁免"
        problems.append("已知真漂移里有条目已不再出现（修好了？）——"
                        "请从 check_standards.py 的 KNOWN_DRIFT 里删掉，"
                        "否则它会变成永久豁免：\n"
                        + "\n".join(f"    · {k}" for k in vanished))
    dead_mentions = dead_whitelist_entries(stale)
    if dead_mentions:
        # 与 KNOWN_DRIFT 同理：白名单只增不减就会变成永久豁免。
        # 条目为什么"死"通常是好理由（文档删了那句、或那个文件真的补上了），
        # 但**必须有人来删**，不能靠它自己沉默。
        problems.append(f"KNOWN_MENTIONS 里有 {len(dead_mentions)} 条已不再命中任何失效引用"
                        "（文档改了？还是那个文件补上了？）"
                        "——请从 check_standards.py 的 KNOWN_MENTIONS 里删掉，"
                        "否则它会变成永久豁免：\n"
                        + "\n".join(f"    · {k}" for k in dead_mentions))

    # 内部口径不变式：凡被计为"实际失效"的，必须真的出现在 problems 里。
    # 这条**自己会红**，用来钉住 P1 那种"算了却不用"的回潮。
    for k in unexpected:
        if not any(k in msg for msg in problems):
            raise RuntimeError(
                f"内部口径不一致：`{k}` 被计为失效引用，却没有进入 problems —— "
                f"这就是 2026-10-24 那版「报 23 处失效却判定全部通过」的缺陷形态")
    return problems


def check_sources(verbose: bool) -> list[str]:
    """C 类：`AGENTS.md` 引用的规范文件是否真的存在。

    ⚠️ **外部资产的两种"没有"必须分开**（2026-10-27）：
    `STC/` 在仓库**外面**（`..\\STC\\`，属用户的其他资产）。全新 checkout 或别人的机器上
    **整棵树都不在**，这不叫漂移；而"树在、某个文件没了"才是真的漂移。
    第一版把两者都判红，后果是：`git archive` 出来的干净副本里 `check_standards` 直接红，
    连带 `selftest_doc_facts` 也红 —— **CI 上线第一天就挂，而且挂得莫名其妙**。
    实测过（B-27）：干净副本 23 个测试里 4 红，这条是其中之一。
    """
    print("----- C. 规范来源是否存在 -----")
    text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    stc_refs = sorted(set(re.findall(r"`(STC[/\\][^`]+)`", text)))
    problems: list[str] = []
    # `STC/` 在仓库外（`..\STC\`），所以按**父目录**去找
    outside_root = ROOT.parent
    stc_tree = outside_root / "STC"
    if stc_refs and not stc_tree.is_dir():
        print(f"  （本机没有仓库外的 {stc_tree.as_posix()} —— STC 属用户其他资产，"
              f"C 类跳过，不算漂移）")
        print(f"    引用 {len(stc_refs)} 处：{', '.join(stc_refs[:3])}"
              f"{' …' if len(stc_refs) > 3 else ''}")
        return []
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


# ------------------------------------------------------------------ #
# D 类：待办清单
# ------------------------------------------------------------------ #
# 为什么这一类存在（2026-10-27）：待办原来散在 3000 行 `交接文档.md` 的 6 张表
# 加一个 planbook 的 13 个 P 项里，**没有 ID、没有状态、互相滞后**。于是
# "还有哪些活"这个问题每次都要靠人翻 —— 包括 AI。翻出来的答案还会互相矛盾
# （§12.7 说时间层待办，§P0-1 的判据已满足一半）。
#
# 所以：状态**只认** `docs/待办清单.md` 一份，本类负责三件事：
#   1. 表自己自洽（ID 唯一、状态在词汇表里、blocked/wontfix 必须写原因）；
#   2. `done` 必须能对上 git —— "改完必验"里最容易被跳过的就是"真的提交了吗"；
#   3. **反向**：文档里每个"待办 / 还没做"章节，必须至少被表里一条引用。
#      少了这一条，新写的待办就会绕过本表，"唯一来源"当场失效。
BACKLOG = ROOT / "docs" / "待办清单.md"
BACKLOG_STATES = {"open", "doing", "blocked", "done", "wontfix"}
BACKLOG_ID_RE = re.compile(r"^B-(\d+)$")
# 章节标题形如 "### 11.6 待办" / "### 硬件待办"；只有带这些词的才算"待办章节"
TODO_HEADING_RE = re.compile(r"^#{2,4}\s+(?:(\d+(?:\.\d+)?)[.、]?\s*)?(.+)$")
TODO_WORDS = ("待办", "还没做", "尚未处理")
COMMIT_RE = re.compile(r"commit=([0-9a-f]{6,40})")


def _split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _git_commits() -> set[str] | None:
    """`git log` 里所有短 hash；没有 git 就返回 None（不静默当成通过）。

    注意 CI 上 `actions/checkout` 默认 `fetch-depth: 1`（浅克隆），`git log`
    只有 1 条，所有历史 commit hash 都会误报「查不到」。gate.yml 已设
    `fetch-depth: 0`；这里仍只取 -2000 条，足够覆盖待办表里的历史 hash。
    """
    try:
        proc = subprocess.run(["git", "log", "--format=%h", "-2000"],
                              capture_output=True, text=True, encoding="utf-8",
                              errors="replace", cwd=str(ROOT), timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return {ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()}


@functools.lru_cache(maxsize=None)
def _git_ignored(rel: str) -> bool:
    """该路径是否被 .gitignore 明确排除 —— 用于区分「文件丢失」与「本就不入库」。

    待办表详情列会指路 planbooks/（内部计划）、outputs/（跑出来的报告）这类
    **故意不入库**的文件：本地有、CI 的干净 checkout 里没有。若一律按「磁盘
    上没有」判失败，CI 必然红 —— 这是环境差异，不是文档漂移。用 git 自己
    的 check-ignore 判断（单一真相，不另写一份前缀黑名单），被忽略就跳过
    磁盘存在性检查；没被忽略却丢了，仍照常报错。
    """
    try:
        proc = subprocess.run(["git", "check-ignore", "-q", rel],
                              capture_output=True, cwd=str(ROOT), timeout=10)
    except (OSError, subprocess.SubprocessError):
        return False  # 问不到 git → 保守：照常校验存在性
    return proc.returncode == 0


@functools.lru_cache(maxsize=16)
def _heading_index(path: Path) -> dict[str, str]:
    """`交接文档.md` 里所有 "编号 -> 标题" 的索引（用来核对 `§NN.N` 是否真的存在）。

    带 `lru_cache`：30 行待办 × 每行 1~2 份文档，不缓存就是 60 次全量读那个 3000 行文件。
    """
    out: dict[str, str] = {}
    text = _read_text(path)
    if text is None:
        return out
    for line in text.splitlines():
        m = TODO_HEADING_RE.match(line)
        if not m:
            continue
        num, title = m.group(1), m.group(2)
        if num:
            out[num] = title
    return out


def _hook_installed() -> bool | None:
    """`core.hooksPath` 是否指向了 `.githooks`；问不到 git 就返回 None。

    **只做提示，不算失败**：新 clone 下来本来就还没装（安装是每台机器一次的手工步骤），
    把它做成失败会让所有人在第一次跑 standards 时撞红。
    但也不能完全不说 —— "写着的闸门不拦人"正是本项目最烦的一类问题，
    所以让它每次跑 standards 都自己喊一声。
    """
    try:
        proc = subprocess.run(["git", "config", "--get", "core.hooksPath"],
                              capture_output=True, text=True, encoding="utf-8",
                              errors="replace", cwd=str(ROOT), timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return False
    return (proc.stdout or "").strip().replace("\\", "/").strip("/") == ".githooks"


def check_backlog(verbose: bool) -> list[str]:
    """D 类：待办清单自洽 + `done` 能对上 git + 文档待办章节全覆盖。"""
    print("----- D. 待办清单（唯一来源 / 状态 / commit 对账） -----")
    rel = BACKLOG.relative_to(ROOT).as_posix()
    text = _read_text(BACKLOG)
    if text is None:
        return [f"找不到 {rel} —— 待办的唯一来源不在了；"
                f"别把状态再写回各章节（那正是这次要收掉的问题）"]

    rows: list[list[str]] = []
    for line in text.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = _split_row(line)
        if cells and BACKLOG_ID_RE.match(cells[0]):
            rows.append(cells)
    if not rows:
        return [f"{rel} 里没有解析出任何 `B-NN` 行 —— 表格式改了？检查器会因此变成空转"]

    problems: list[str] = []
    commits = _git_commits()
    if commits is None:
        print("    （没有可用的 git，`done` 的 commit 对账跳过 —— 其余照查）")

    seen: dict[str, int] = {}
    counted = {s: 0 for s in sorted(BACKLOG_STATES)}
    for cells in rows:
        if len(cells) < 5:
            problems.append(f"{cells[0]}：列数不足 5（应是 ID/状态/一句话/卡在什么/详情）")
            continue
        bid, state, _what, why, detail = cells[0], cells[1], cells[2], cells[3], cells[4]
        seen[bid] = seen.get(bid, 0) + 1
        if state not in BACKLOG_STATES:
            problems.append(f"{bid}：状态 {state!r} 不在词汇表 {sorted(BACKLOG_STATES)}")
            continue
        counted[state] += 1
        # blocked / wontfix 必须写清原因：只写"待定""—"等于没有
        if state in ("blocked", "wontfix") and why.strip() in ("", "—", "-", "/"):
            problems.append(f"{bid}：{state} 必须写清「卡在什么 / 为什么不做」")
        # done 必须能对上 git
        if state == "done":
            m = COMMIT_RE.search("|".join(cells))
            if not m:
                problems.append(f"{bid}：done 必须带 `commit=<短 hash>`（改完必验要能查）")
            elif commits is not None and not any(c.startswith(m.group(1))
                                                  for c in commits):
                problems.append(f"{bid}：commit={m.group(1)} 在 `git log` 里查不到")
        # 详情指路必须真的存在。
        # `§NN` 按**同一格里点名的那份 .md** 解析（没点名才默认 `交接文档.md`）——
        # 第一版只认交接文档，于是 `§2.4`（开发指南的章节）被误报成不存在。
        # 判据写错比代码错更容易骗人，所以这里宁可多一层解析。
        for sec in re.findall(r"§(\d+(?:\.\d+)*)", detail):
            docs = re.findall(r"`([^`]+\.md)`", detail) or ["docs/交接文档.md"]
            if not any(sec in _heading_index(ROOT / d) for d in docs
                       if (ROOT / d).exists()):
                problems.append(
                    f"{bid}：详情指向 §{sec}，但 {' / '.join(docs)} 里没有这个编号章节")
        for ref in re.findall(r"`([^`]+\.(?:md|py|bat|json))`", detail):
            if not (ROOT / ref).exists() and not _git_ignored(ref):
                problems.append(f"{bid}：详情指向 `{ref}`，但磁盘上没有")

    for bid, n in seen.items():
        if n > 1:
            problems.append(f"{bid}：ID 重复出现 {n} 次")

    # ---- 反向：文档里每个"待办章节"都必须被本表引用 ----
    doc = ROOT / "docs" / "交接文档.md"
    uncited: list[str] = []
    doc_text = _read_text(doc) or ""
    for line in doc_text.splitlines():
        m = TODO_HEADING_RE.match(line)
        if not m or not any(w in line for w in TODO_WORDS):
            continue
        num, title = m.group(1), m.group(2)
        key = f"§{num}" if num else re.sub(r"[^\w\u4e00-\u9fff]", "", title)
        if key not in text:
            uncited.append(f"{key}（{title.strip()}）")
    if uncited:
        problems.append(
            "这些待办章节没有被 `docs/待办清单.md` 引用，等于「有个待办没人管」："
            + "、".join(uncited))

    summary = "；".join(f"{k} {v}" for k, v in counted.items() if v)
    print(f"    {len(rows)} 条（{summary}）")
    hooked = _hook_installed()
    if hooked is False:
        print("    [提示] 提交前自动门没装：`git config core.hooksPath .githooks`"
              "（每台机器一次；不装则 pre-push 不会跑本检查）")
    elif hooked is None:
        print("    （问不到 git，自动门装没装未知）")
    if verbose or problems:
        for cells in rows[:3]:
            print(f"    · {cells[0]:<5} {cells[1]:<8} {cells[2][:40]}")
    if problems:
        for p in problems:
            print(f"    [不达标] {p}")
    return problems


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
    problems += check_backlog(verbose)
    print()

    print("=" * 78)
    if problems:
        print(f"发现 {len(problems)} 类问题：")
        for i, msg in enumerate(problems, 1):
            print(f"  {i}) {msg}")
        print()
        print("判定结果：有不达标项")
        return 1
    print("四类检查全部通过：引用、来源、计数、待办都与磁盘一致。")
    print("判定结果：全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
