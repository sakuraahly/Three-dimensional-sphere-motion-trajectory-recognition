"""测试数量的**陈旧说法**扫描：只报告，**绝不改写**。

用法::

    python apps/tools/check_test_count.py            # 报告
    python apps/tools/check_test_count.py --strict   # 有陈旧说法就退出码 1

## 为什么它只报告、不改写（这是本工具存在的核心理由）

我**先写了一个会自动改写的版本**，然后看它打算改什么 —— 立刻删了：

| 它想改的原文 | 改完会变成 | 问题 |
| ------------ | ---------- | ---- |
| `这 8 个文件的路径在 7 份文档里被引用 55 处` | `这 19 个文件…` | 说的是 **ppt 工具**，跟测试数量毫无关系 |
| `原状态：6 个回归测试一个都没跑` | `原状态：19 个…` | **历史记录**，改了就是说谎 |
| `新增第 10 个回归测试 selftest_config_wiring.py` | `新增第 19 个…` | 它**当时确实是第 10 个**，改了反而失真 |
| `参数化替代了那 6 个文件` | `那 19 个文件` | 说的是**启动脚本**数量 |

**数字在正文里同时承担"当前事实"和"历史记录"两种角色，工具分不出来。**
自动改写会把后者一起改掉 —— 那是**伪造历史**，比漂移更坏。

所以本工具的契约是：

- **找出**可疑的陈旧说法（带足够的上下文让人判断）；
- **指出**哪些是"当前事实"（该改）、哪些是"历史叙述"（不该改）；
- **一个字都不改** —— 改由人做，人知道那段话在说什么。

`AGENTS.md` 里那条"不做大范围样板统一"的判断，和这里是同一个道理：
**拿机器的确定性去换人的判断，在语义层面是亏的。**
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]

CN = "零一二三四五六七八九"


def cn_number(n: int) -> str:
    if n < 10:
        return CN[n]
    if n < 20:
        return "十" + (CN[n % 10] if n % 10 else "")
    return CN[n // 10] + "十" + (CN[n % 10] if n % 10 else "")


DOCS = ["AGENTS.md", "docs/交接文档.md", "docs/项目总纲.md",
        "docs/开发指南.md", "docs/文件结构.md",
        "docs/开题报告_研究思路四页文稿.md"]

# 「当前事实」的形态：数字紧贴"回归测试/测试"这类词，且没有历史标记。
CURRENT_RE = re.compile(
    r"(?P<num>\d{1,2}|[零一二三四五六七八九十]{1,3})\s*"
    r"(?P<unit>个|条)?\s*(?P<what>回归测试|selftest_|测试要在命令行)")
# 「历史叙述」的标记：出现这些词，说明在讲过去或举例，改了就是说谎／失真。
HISTORY_MARKERS = ("原状态", "原先", "当时", "以前", "曾经", "历史", "第 ",
                   "新增第", "已修", "先补跑", "还原", "回退", "改成",
                   # 提交信息模板里的示例、风格指南里的引号示例 —— 示范的是**格式**
                   "示例", "风格", "feat:", "docs:", "fix:", "尽量带上")
# 章节号（`### 5.3 加一个回归测试`）不是计数声明，必须排除。
SECTION_RE = re.compile(r"^\s*#{2,6}\s*\d+(?:\.\d+)*")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    tests = sorted((ROOT / "tests").glob("selftest_*.py"))
    n = len(tests)
    cn = cn_number(n)
    print("=" * 74)
    print(f"测试数量陈旧说法扫描（磁盘实际 {n} 个 / 「{cn}」）—— **只报告，不改写**")
    print("=" * 74)

    stale: list[tuple[str, int, str, bool]] = []   # (文件, 行号, 行, 疑似历史)
    for rel in DOCS:
        p = ROOT / rel
        if not p.exists():
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if SECTION_RE.match(line):
                continue          # `### 5.3 加一个回归测试` 里的 5.3 是章节号
            for m in CURRENT_RE.finditer(line):
                num = m.group("num")
                if num.isdigit():
                    if int(num) == n:
                        continue
                elif num == cn:
                    continue
                is_history = any(k in line for k in HISTORY_MARKERS)
                stale.append((rel, i, line.strip(), is_history))

    if not stale:
        print(f"\n没有发现陈旧说法：所有「N 个测试」的说法都是 {n}。")
        print("判定结果：全部通过")
        return 0

    cur = [s for s in stale if not s[3]]
    hist = [s for s in stale if s[3]]
    print(f"\n扫到 {len(stale)} 处可疑 —— 其中**像是当前事实** {len(cur)} 处、"
          f"**像是历史叙述** {len(hist)} 处\n")

    print("【该改的：当前事实】")
    for rel, i, line, _ in cur:
        print(f"  {rel}:{i}")
        print(f"    {line[:110]}")
    if not cur:
        print("  （无）")

    print("\n【**不该改**的：历史叙述 —— 改了就是伪造历史】")
    for rel, i, line, _ in hist:
        print(f"  {rel}:{i}")
        print(f"    {line[:110]}")
    if not hist:
        print("  （无）")

    print("\n" + "-" * 74)
    print("本工具**不会替你改**：数字在正文里同时承担「当前事实」与「历史记录」")
    print("两种角色，工具分不出后者。请只改上面【该改的】那批。")
    print("权威口径由 `sync_facts.py --check` 与 `selftest_doc_facts.py` 守。")
    if args.strict and cur:
        print("\n判定结果：有陈旧的当前事实说法")
        return 1
    print("\n判定结果：全部通过（扫描完成）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
