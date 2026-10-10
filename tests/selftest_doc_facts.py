"""文档事实自测：测试清单/数量、`AGENTS.md` 的计数断言、已知的引用漂移。

为什么需要它
------------
本项目的头号反复问题是**文档漂移**。实测（本轮取证）：
- **测试数量在 6 处说法是过时的**（`交接文档.md:713` 说"五个"、`:831` 说"九个"、
  `开发指南.md:313/334` 说"六个"、`开题报告_研究思路四页文稿.md:408` 说"九个"、
  `摄像头设备管理方案.md:246` 说"五个"，实际 10）；
- `AGENTS.md` §3 里「相机/标定（7）」「开题报告 PPT（8）」与「引用 55 处」是**硬编码计数**，
  加一个文件就漂移，而**在此之前没有任何东西守着它们**。

本测试把这几个数变成**可执行的断言**（那些过时的散文说法留给 P2-8 的生成器去修；
这里先把"会漂的东西"钉住，而不是先去改 6 处文字）。

三类断言
--------
1. **测试清单**：`tests/selftest_*.py` 的数量、命名规范（`AGENTS.md` §4 要求
   `selftest_<被守护的模块>.py`）、以及**每一个都被 `AGENTS.md` §7 的表登记**。
2. **计数断言**：`AGENTS.md` §3 的「(7)」「(8)」与磁盘事实一致。
3. **引用漂移**：反引号里的仓库路径若指向不存在的文件，必须落在一份**显式白名单**里。

关于第 3 条 —— "引用"与"谈及"必须分开（本轮实测踩到的坑）
--------------------------------------------------------
天真扫描（"文件不存在就算漂移"）会报 20 处，**其中大部分是假阳性**：
它们出现在"**原来写的 / 实际**"这类描述性表格或"这些目录**从未存在过**"这类
说明句里 —— 那是**有意保留的历史信息**，抹掉反而丢掉了"为什么改"。
所以本测试用一份**白名单逐条列名**，每条都写清"为什么不算漂移"。
白名单之外出现新的失效引用 -> 报红（那才是真漂移）。

反向自证（改了就会红）
--------------------
====================================  ==========================================  ========
改动                                  注入方式                                    结果
====================================  ==========================================  ========
谎报测试数量                          `REAL_TEST_COUNT = 9`                       exit=1
把 `AGENTS.md` 的 (7) 改成 (8)        见 `_group_counts_ok` 的注入说明            exit=1
新增一个未登记的测试文件              在 `tests/` 放一个 `selftest_zzz.py`        exit=1
白名单之外新增失效引用                在 `docs/` 里写一个不存在的路径             exit=1
====================================  ==========================================  ========

用法::

    python tests/selftest_doc_facts.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
while not (ROOT / "sphere_tracker").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

AGENTS_MD = ROOT / "AGENTS.md"

# 本测试自己也是测试，也要被数进去
REAL_TEST_COUNT = 14
# `AGENTS.md` §3 里写死的两个计数。原文形如：
#   `| **相机 / 标定**（7） | ...` / `| **开题报告 PPT**（8） | ...`
# 注意：**全角括号**、标题里带 `**`、斜杠两侧有空格 —— 正则要容忍这些排版差异。
CLAIM_CAM_RE = re.compile(r"相机\s*/\s*标定[^\d（(]{0,8}[（(]\s*(\d+)\s*[）)]")
CLAIM_PPT_RE = re.compile(r"开题报告\s*PPT[^\d（(]{0,8}[（(]\s*(\d+)\s*[）)]")
# 归属：apps/tools 里这些名字属于 PPT 组，其余属于相机/标定组
PPT_TOOL_RE = re.compile(r"opening|preview|overflow|ppt")
# 测试命名规范（AGENTS.md §4）
TEST_NAME_RE = re.compile(r"^selftest_[a-z0-9_]+\.py$")

# 放置规则里的顶层目录：只有落在这些下面的反引号路径才算"仓库路径引用"
KNOWN_TOP = {"sphere_tracker", "calib", "stereo", "apps", "tests", "docs", "assets",
             "config", "outputs", "models", "legacy", "STC"}
PATH_RE = re.compile(
    r"`([A-Za-z_][\w./\\-]*\.(?:py|md|json|npz|bat|txt|yml|ps1|pptx|jpg|png))"
    r"(?::(\d+)(?:-(\d+))?)?`")
TABLE_ROW_RE = re.compile(r"^\s*\|")

# 本测试自己引用的"将要新建"的文件（属计划，不算漂移）
SELF_DOC = "docs/planbooks/2026-10-10_架构接缝与缺失基础设施.md"

# 指向**仓库外**（`..\STC\`，属用户的其他资产）的引用。它们的特殊性是：
# 文件在磁盘上**确实存在**，只是不在本仓库内 —— 所以既不进白名单、
# 也不该当漂移报红，单独归一类并**如实计数**（这是 `AGENTS.md` §6 规范第 3 层的边界）。
OUTSIDE_REPO_PREFIX = "STC"

# --------------------------------------------------------------------------
# 失效引用的白名单：每条都写清"为什么不算漂移"
# 键 = "<文档相对路径>:<引用的原文>" —— **刻意不带行号**：
# 文档一改行号就全变，用行号做键会让这个测试变成噪声源（实测踩过）。
# 值为理由。
# --------------------------------------------------------------------------
KNOWN_STALE_REFS: dict[str, str] = {
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
    "README.md:video_edge_detect.py": "在「历史遗留脚本」说明表格里",
    "README.md:test.py": "同上",
    "docs/开题报告_研究思路四页文稿.md:calib/calibrate_stereo.py":
        "在路径对照表里（旧路径 vs 实际路径）",
    "docs/开题报告_研究思路四页文稿.md:stereo/match.py":
        "同表的旧路径列（`│ 模块路径 | ... stereo/match.py | 实际为 apps/...`）",
}

# **真漂移**：白名单之外、但已确认是"文档承诺了却没发生"的两条。
# 它们与本轮 P0-2/P0-4 直接相关，所以**单独列出并断言它们确实还在**
# —— 等 P2-10 修好之后，这里改成"必须为空"。
# （见 selftest_calib_facts.py 第 3 节：标定判定不通过就不保存。
#   用户手册承诺会存 config/stereo_calibration.npz，而失败路径没写。）
ACTIVE_DRIFT: dict[str, str] = {
    r"docs/用户手册.md:config\stereo_calibration.npz":
        "标定失败时按设计不保存，用户手册的失败路径没写（P2-10 待修）",
    r"docs/用户手册.md:config\stereo_calibration.json":
        "同上",
}


def _doc_files() -> list[Path]:
    return [AGENTS_MD, ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]


def _test_files() -> list[Path]:
    return sorted((ROOT / "tests").glob("selftest_*.py"))


def _src_literals() -> set[str]:
    """仓库里作为字符串字面量出现过的路径（运行时生成的文件也算，如 config/sphere_config.json）。"""
    out: set[str] = set()
    for py in ROOT.rglob("*.py"):
        if ".git" in py.parts or "__pycache__" in py.parts:
            continue
        for m in re.finditer(
                r"""["']([^"'\n]{3,120}\.(?:py|md|json|npz|bat|txt|yml|ps1|pptx|jpg|png))["']""",
                py.read_text(encoding="utf-8", errors="replace")):
            out.add(m.group(1).replace("\\", "/").lstrip("./"))
    return out


def collect_stale_refs() -> dict[str, str]:
    """扫全库文档，返回 {doc:引用原文: 该行原文} —— 只含"仓库路径引用但文件不存在"。

    键**不含行号**（行号会随任何编辑漂移，用它做键会让白名单变成噪声源）。
    行号仍打进值里，便于定位。
    """
    literals = _src_literals()
    found: dict[str, str] = {}
    for doc in _doc_files():
        rel = doc.relative_to(ROOT).as_posix()
        if rel == SELF_DOC:
            continue
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
def test_count_and_naming() -> bool:
    """第 1 节：测试数量与命名规范。"""
    print("----- 1. 测试清单与数量 -----")
    files = _test_files()
    names = [f.name for f in files]
    bad_names = [n for n in names if not TEST_NAME_RE.match(n)]

    print(f"  tests/selftest_*.py 实际 {len(files)} 个（本测试认为应当是 {REAL_TEST_COUNT}）")
    for n in names:
        print(f"    {n}")

    checks = [
        (f"数量 = {REAL_TEST_COUNT}（实际 {len(files)}）", len(files) == REAL_TEST_COUNT),
        ("命名都符合 `selftest_<被守护的模块>.py`（全小写下划线）",
         not bad_names),
    ]
    if bad_names:
        print(f"  不合规的命名：{bad_names}")
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_tests_are_registered_in_agents_md() -> bool:
    """第 2 节：每个测试都被 `AGENTS.md` §7 的表登记（新加的不能漏写）。"""
    print("----- 2. 每个测试都在 AGENTS.md §7 登记 -----")
    text = AGENTS_MD.read_text(encoding="utf-8")
    registered = set(re.findall(r"`(selftest_[a-z0-9_]+\.py)`", text))
    actual = {f.name for f in _test_files()}
    missing = sorted(actual - registered)
    extra = sorted(registered - actual)

    print(f"  AGENTS.md 登记 {len(registered)} 个 / 磁盘上有 {len(actual)} 个")
    if missing:
        print(f"  磁盘上有、AGENTS.md 没写：{missing}")
    if extra:
        print(f"  AGENTS.md 写了、磁盘上没有：{extra}")
    checks = [
        ("没有「磁盘有但文档没登记」的测试", not missing),
        ("没有「文档登记了但磁盘没有」的测试", not extra),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_agents_md_count_claims() -> bool:
    """第 3 节：`AGENTS.md` §3 的计数断言与磁盘事实一致。"""
    print("----- 3. AGENTS.md 的硬编码计数 -----")
    tools = sorted((ROOT / "apps" / "tools").glob("*.py"))
    ppt = [p for p in tools if PPT_TOOL_RE.search(p.stem)]
    cam = [p for p in tools if not PPT_TOOL_RE.search(p.stem)]

    text = AGENTS_MD.read_text(encoding="utf-8")
    cam_m = CLAIM_CAM_RE.search(text)
    ppt_m = CLAIM_PPT_RE.search(text)
    claim_cam = int(cam_m.group(1)) if cam_m else None
    claim_ppt = int(ppt_m.group(1)) if ppt_m else None

    print(f"  apps/tools 实际：相机/标定 {len(cam)} 个、开题报告 PPT {len(ppt)} 个"
          f"（合计 {len(tools)}）")
    print(f"    相机组：{[p.name for p in cam]}")
    print(f"    PPT 组：{[p.name for p in ppt]}")
    print(f"  AGENTS.md 声明：相机/标定 "
          f"{claim_cam if claim_cam is not None else '(找不到)'} 个、"
          f"PPT {claim_ppt if claim_ppt is not None else '(找不到)'} 个")

    checks = [
        ("AGENTS.md 里有「相机/标定（N）」这个声明", claim_cam is not None),
        ("AGENTS.md 里有「开题报告 PPT（N）」这个声明", claim_ppt is not None),
        (f"相机/标定组 = {len(cam)}（声明 {claim_cam}）", claim_cam == len(cam)),
        (f"PPT 组 = {len(ppt)}（声明 {claim_ppt}）", claim_ppt == len(ppt)),
        ("两组相加不超过 apps/tools 全部文件数（说明分组确实覆盖了）",
         0 < len(cam) + len(ppt) <= len(tools)),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_ref_vs_mention_partition() -> bool:
    """第 4 节：失效引用必须落进三类之一，不能有第四个来源。"""
    print("----- 4. 引用 vs 谈及：失效引用必须逐条列名 -----")
    found = collect_stale_refs()

    outside = {k: v for k, v in found.items() if OUTSIDE_REPO_PREFIX in k.split(":")[1]}
    inside = {k: v for k, v in found.items() if k not in outside}
    known = set(KNOWN_STALE_REFS) | set(ACTIVE_DRIFT)
    unexpected = {k: v for k, v in inside.items() if k not in known}
    vanished_active = sorted(k for k in ACTIVE_DRIFT if k not in found)

    print(f"  扫到失效引用 {len(found)} 处 = 仓库外引用 {len(outside)} + 仓库内 {len(inside)}")
    print()
    print(f"  【仓库外】{len(outside)} 处（`STC/` 在 `..\\STC\\`，属用户其他资产；"
          f"文件在磁盘上存在，只是不在本仓库）")
    for k in sorted(outside):
        print(f"    · {k}")
    print()
    print(f"  【白名单】{len(KNOWN_STALE_REFS)} 条：有意保留的历史信息（表格里「原来写的 / 实际」）")
    print(f"  【ACTIVE_DRIFT】{len(ACTIVE_DRIFT)} 条：真漂移，待 P2-10 修")
    for k, why in ACTIVE_DRIFT.items():
        print(f"    · {k}")
        print(f"        {why}")
    if unexpected:
        print()
        print("  **白名单之外的新失效引用**（这才是要抓的漂移）：")
        for k, line in sorted(unexpected.items()):
            print(f"    · {k}")
            print(f"        {line[:110]}")
    vanished_known = sorted(k for k in KNOWN_STALE_REFS if k not in found)
    if vanished_known:
        print()
        print(f"  （提示：白名单里有 {len(vanished_known)} 条已不再出现，说明文档被改过了）")
        for k in vanished_known[:6]:
            print(f"      · {k}")

    checks = [
        ("**没有白名单之外的仓库内失效引用**（新漂移会被当场抓住）", not unexpected),
        ("ACTIVE_DRIFT 的两条仍然存在（说明还没修，别假装修完了）", not vanished_active),
        ("仓库外引用确实指向 `STC/`（属已知边界，`AGENTS.md` §6 第 3 层）",
         bool(outside) and all(OUTSIDE_REPO_PREFIX in k for k in outside)),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def main() -> int:
    print("=" * 70)
    print("文档事实自测（测试清单 / 计数断言 / 引用漂移）")
    print("=" * 70)
    results = []
    for fn in (test_count_and_naming,
               test_tests_are_registered_in_agents_md,
               test_agents_md_count_claims,
               test_ref_vs_mention_partition):
        results.append(fn())
        print()
    ok = all(results)
    print("判定结果：" + ("全部通过" if ok else "有不达标项"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
