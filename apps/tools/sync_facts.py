"""派生事实同步器：文档里的"计数"与"清单"由磁盘算出来，不手抄。

为什么需要它
------------
这个项目最反复出现的问题是**文档漂移**，而漂移里最常见的一类是**计数**：

- 测试数量：`tests/selftest_*.py` 从 10 涨到 15 的过程中，每一轮都要手工改遍全库。
  实测**散布在 16 处**，其中 6 处一度同时是错的（"五个""六个""九个""十个"并存）。
- 模块分组：`AGENTS.md` §3 写死「相机 / 标定（7）」「开题报告 PPT（8）」，
  加一个文件就漂，而在此之前没有任何东西守着它们。

本模块把这两类**派生事实**变成"由磁盘算、可校验"：

    python apps/tools/sync_facts.py --check     # 校验，不一致就非零退出

设计约束（都是踩过的）

1. **只认"描述现状"的说法，不碰"历史记录"。**
   "原状态：6 个回归测试一个都没跑""六 → 八""新增第 10 个回归测试"是**当时的实况**，
   改了就是篡改历史。所以检查目标是**逐条写死**的期望串，不做全库正则替换。
2. **不做自动改写正文。** 第一版想做，权衡后放弃：正文改写风险（误伤历史记录、
   破坏排版）高于收益，而"指出哪份文档哪一行不符"已经能让人一分钟改完。
   本工具的价值在**发现**，不在**代笔**。
3. **分组以 `AGENTS.md` §3 的清单为准，不按文件名猜。**
   第一版按 `opening|preview|ppt` 猜，结果新建的 `sync_facts.py` 被猜进
   "相机/标定"组、把计数从 7 变成 8 —— **靠命名约定推断归属，本身就是漂移源**。
   现在反过来：解析 `AGENTS.md` 里那两张表的文件清单，与磁盘求差集。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "sphere_tracker").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CN_DIGITS = "零一二三四五六七八九十"


def cn_number(n: int) -> str:
    """1..99 的中文写法（文档里用的那种）。"""
    if n <= 10:
        return CN_DIGITS[n]
    if n < 20:
        return "十" + CN_DIGITS[n - 10]
    tens, ones = divmod(n, 10)
    return CN_DIGITS[tens] + "十" + (CN_DIGITS[ones] if ones else "")


# --------------------------------------------------------------------------
# 事实来源
# --------------------------------------------------------------------------
def test_files() -> list[Path]:
    return sorted((ROOT / "tests").glob("selftest_*.py"))


def agents_groups() -> tuple[list[tuple[str, int, list[str]]], int]:
    """解析 `AGENTS.md` §3 的分组表。

    返回 `[(组名, 声明计数, 文件名列表), ...]`。

    **返回全部组**，不做"相机组 + 其它组"这种二分 —— 第一版按二分写，
    结果新增第三组之后，多出来的那个文件被无条件算进"PPT 组"。
    **变量名和内容对不上，本身就是一种漂移。**

    解析踩过两次，都记在这里：

    1. `next(ln for ln in lines if "开题报告 PPT" in ln)` 取到了**说明段落**
       （那段文字里提到了"开题报告 PPT"），组被解析成错的东西；
    2. 只按"我已知有两行"来取，新增第三组时漏掉它，
       表现是 `sync_facts.py` 把自己报成"未归类" —— 报错信息就是它自己，极易误判。

    教训：**解析表格要按结构取全部，锚定要锚到整体形状，不能只锚到片段。**
    """
    text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    rows = [ln for ln in text.splitlines() if ln.strip().startswith("|")]
    pat = re.compile(r"`([a-zA-Z_0-9]+\.py)`")

    # 分组行必须**精确锚定**：`| **组名**（N） |`。用 search 会误伤说明文字里的管道符。
    group_row = re.compile(
        r"^\|\s*\*\*(?P<name>[^*]+)\*\*\s*[（(](?P<count>\d+)[）)]\s*\|")
    groups: list[tuple[str, int, list[str]]] = []
    for ln in rows:
        m = group_row.match(ln)
        if m:
            groups.append((m.group("name").strip(), int(m.group("count")),
                           pat.findall(ln)))
    if not groups:
        raise RuntimeError(
            "AGENTS.md §3 里找不到任何分组行（形如 `| **组名**（N） | ...`）。"
            "本工具靠这张表判断分组；表被改过就得同步改这里。")
    return groups


def tools_on_disk() -> set[str]:
    return {p.name for p in (ROOT / "apps" / "tools").glob("*.py")}


# --------------------------------------------------------------------------
# 期望串：逐条写死"描述现状"的说法。
#
# ⚠️ 数量词这里**接受两种写法**（中文数字 / 阿拉伯数字）：文档里本来就两种并存
#（`十五条` 与 `15 个`），强行统一只会制造一次无意义的大改。
# 本工具要抓的是**数值过期**，不是**文体**。
# --------------------------------------------------------------------------
def expected_test_count_strings(n: int) -> list[tuple[tuple[str, ...], str]]:
    """返回 [(可接受的写法, 文档)]。

    ⚠️ 这份清单要跟着**写作方式的改变**走。2026-10-10 把交接文档 §4/§6 里
    "逐条手抄 5 个测试路径"改成"指向 `AGENTS.md` §7 + 一个命令"之后，
    原来那几条期望串就不再存在了 —— 校验器当场报了出来（这正是它该做的）。
    所以这里改成检查**新的说法**。
    """
    cn = cn_number(n)
    return [
        # ---- 交接文档：现在用命令而不是逐条路径 ----
        ((f"**{cn}个**回归测试", f"**{n} 个**回归测试"), "docs/交接文档.md"),
        ((f"启动程序.bat test full         # 全套 {n} 个",), "docs/交接文档.md"),
        ((f"全套 {n} 个",), "docs/交接文档.md"),
        ((f"全套 {n} 个，约 6 分钟",), "docs/交接文档.md"),
        # ---- 项目总纲 ----
        ((f"**{cn}个回归测试都必须输出", f"**{n} 个回归测试都必须输出"),
         "docs/项目总纲.md"),
        ((f"| {cn}个回归测试 ", f"| {n} 个回归测试 "), "docs/项目总纲.md"),
        ((f"回归测试（{cn}个，改动后必跑）", f"回归测试（{n} 个，改动后必跑）"),
         "docs/项目总纲.md"),
        # ---- AGENTS.md ----
        ((f"回归测试（{cn}个，改动后必跑）", f"回归测试（{n} 个，改动后必跑）"),
         "AGENTS.md"),
        ((f"改动代码后 {n} 个回归测试必须能一键复跑",), "AGENTS.md"),
        ((f"质量底线：{n} 个回归测试",), "AGENTS.md"),
        ((f"全套 {n} 个（改代码后必跑，约 6 分钟）",), "AGENTS.md"),
        # ---- 开发指南 ----
        ((f"现有 **{n} 个** `selftest_*.py`",), "docs/开发指南.md"),
        ((f"表里的 {n} 个文件",), "docs/开发指南.md"),
        ((f"→ **{n} 个全跑**",), "docs/开发指南.md"),
        ((f"**{n} 个测试要在命令行自己跑**",), "docs/开发指南.md"),
        # ---- 文件结构 ----
        ((f"回归测试（{cn}个，改动后必跑）", f"回归测试（{n} 个，改动后必跑）"),
         "docs/文件结构.md"),
        # ---- 开题报告文稿 ----
        ((f"合成真值自测（{n} 个回归测试）",), "docs/开题报告_研究思路四页文稿.md"),
    ]


def _relevant_lines(path: Path, keywords: tuple[str, ...]) -> str:
    out = []
    for i, ln in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if any(k in ln for k in keywords):
            out.append(f"L{i}: {ln.strip()[:88]}")
    return " ； ".join(out[:3])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="派生事实校验器（计数 / 清单）")
    ap.add_argument("--check", action="store_true",
                    help="只校验（默认行为也是校验；保留该参数为了与文档里的写法一致）")
    ap.parse_args(argv)

    n = len(test_files())
    cn = cn_number(n)
    groups = agents_groups()
    disk = tools_on_disk()

    print("=" * 76)
    print("派生事实校验（计数 / 清单）")
    print("=" * 76)
    print(f"磁盘事实：tests/selftest_*.py = {n} 个（中文写法「{cn}」）")
    print(f"          apps/tools/*.py     = {len(disk)} 个")
    for gname, gcount, gfiles in groups:
        flag = "" if gcount == len(gfiles) else "  ← 声明与清单不符"
        print(f"          AGENTS.md「{gname}」：声明 {gcount}、清单 {len(gfiles)} 个{flag}")
    print()

    problems: list[str] = []

    # ---- 1) 分组清单 vs 磁盘 ----
    listed = {f for _n, _c, files in groups for f in files}
    unlisted = sorted(disk - listed)
    missing = sorted(listed - disk)
    if unlisted:
        problems.append(
            f"apps/tools 里有 {len(unlisted)} 个文件**没被 AGENTS.md §3 分组**：\n"
            + "\n".join(f"      · {x}" for x in unlisted)
            + "\n      修法：把它加进那张表，并同时更新该组的 (N)")
    if missing:
        problems.append(
            f"AGENTS.md §3 列了但磁盘上**不存在**的文件：{missing}")
    for gname, gcount, gfiles in groups:
        if gcount != len(gfiles):
            problems.append(f"AGENTS.md「{gname}」组：写「({gcount})」"
                            f"但表里列了 {len(gfiles)} 个：{gfiles}")

    # ---- 2) 测试数量说法 ----
    stale: list[str] = []
    for wants, rel in expected_test_count_strings(n):
        p = ROOT / rel
        if not p.exists():
            stale.append(f"{rel}：文件不存在")
            continue
        text = p.read_text(encoding="utf-8")
        if not any(w in text for w in wants):
            stale.append(f"{rel}：期望含有「{wants[0]}」"
                         + (f"或「{wants[1]}」" if len(wants) > 1 else "")
                         + f"\n      实际相关行 "
                           f"{_relevant_lines(p, ('回归测试', 'selftest_*.py'))}")
    if stale:
        problems.append(f"测试数量有 {len(stale)} 处与磁盘不符（应为 {n}）：「{cn}」\n"
                        + "\n".join(f"      · {x}" for x in stale))

    # ---- 3) 测试登记 ----
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    registered = set(re.findall(r"`(selftest_[a-z0-9_]+\.py)`", agents))
    actual = {p.name for p in test_files()}
    if actual - registered:
        problems.append("磁盘上有、AGENTS.md §7 没登记的测试："
                        + ", ".join(sorted(actual - registered)))
    if registered - actual:
        problems.append("AGENTS.md §7 登记了、磁盘上没有的测试："
                        + ", ".join(sorted(registered - actual)))

    if problems:
        print(f"**发现 {len(problems)} 类不一致**：")
        for i, msg in enumerate(problems, 1):
            print(f"  {i}) {msg}")
        print()
        print("判定结果：有不达标项")
        return 1

    print(f"✅ 分组清单：{len(disk)} 个工具全部被 AGENTS.md §3 归类，计数一致")
    print(f"✅ 测试数量：{len(expected_test_count_strings(n))} 处说法全部为 {n}"
          f"（「{cn}」或「{n}」），AGENTS.md §7 登记完整")
    print()
    print("判定结果：全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
