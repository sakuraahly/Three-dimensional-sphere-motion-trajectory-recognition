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
    python apps/tools/sync_facts.py --fix       # 按同一份期望表改对，改完自动再校验

设计约束（都是踩过的）

1. **只认"描述现状"的说法，不碰"历史记录"。**
   "原状态：6 个回归测试一个都没跑""六 → 八""新增第 10 个回归测试"是**当时的实况**，
   改了就是篡改历史。所以检查目标是**逐条写死**的期望串，不做全库正则替换。
2. **改写范围严格限定在"期望串"里。** 第一版想过全库正则替换，被否掉了：
   正文改写会误伤历史记录、破坏排版。现在 `--fix` 只做一件事 ——
   拿**期望串本身**当模板，把其中紧邻「个」的那个数字换成新值。
   同一行里的「约 6 分钟」「20 对」不在模板内，因此不会被碰到。
   仍用不了模板的行（说明文档的**说法形态**变了）不静默跳过，而是列出来并返回非零。
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
        ((f"（{n} 个 selftest，清单见",), "docs/项目总纲.md"),
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
        ((f"### 2.1 {cn}个测试，各守什么", f"### 2.1 {n} 个测试，各守什么"),
         "docs/开发指南.md"),
        ((f"表里的 {n} 个文件",), "docs/开发指南.md"),
        ((f"→ **{n} 个全跑**",), "docs/开发指南.md"),
        ((f"**{n} 个测试要在命令行自己跑**",), "docs/开发指南.md"),
        # ---- 用户手册（2026-10-14 补：这一份此前只被间接覆盖，
        #      所以 §2.3 的 9 个说法与 13 行的表并存了很久都没人报）----
        ((f"完整的一套是 {n} 个", f"完整的一套是 {n}个"), "docs/用户手册.md"),
        # 2026-10-26 补：扫全库时发现这三处同义说法一直没进表（都是"描述现状"，
        # 不是历史记录）—— 表覆盖不到的地方，就是漂移能长期躺着的地方。
        ((f"加 full 跑全套 {n} 个",), "docs/用户手册.md"),
        ((f"启动程序.bat test full         # 全套 {n} 个（改代码后必跑）",), "docs/用户手册.md"),
        ((f"{n} 个测试的 `main()`",), "docs/开发指南.md"),
        ((f"——{n} 个测试都不需要真相机",), "docs/交接文档.md"),
        # ---- 文件结构 ----
        ((f"回归测试（{cn}个，改动后必跑）", f"回归测试（{n} 个，改动后必跑）"),
         "docs/文件结构.md"),
        # ---- README（2026-10-14 补：正文有两处计数，此前一处都没被检查）----
        ((f"回归测试（改动代码后必跑，共 {n} 个）",), "README.md"),
        # ---- 开题报告文稿 ----
        ((f"合成真值自测（{n} 个回归测试）",), "docs/开题报告_研究思路四页文稿.md"),
        # 架构图里省了量词，只写「回归测试 ×22」——2026-10-26 补：
        # 它一直在图上，却因为形态不同（没有「个」）躲过了所有检查。
        ((f"回归测试 ×{n}",), "docs/开题报告_研究思路四页文稿.md"),
        ((f"{cn}个测试覆盖", f"{n} 个测试覆盖"), "docs/开题报告_研究思路四页文稿.md"),
    ]


def _relevant_lines(path: Path, keywords: tuple[str, ...]) -> str:
    out = []
    for i, ln in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if any(k in ln for k in keywords):
            out.append(f"L{i}: {ln.strip()[:88]}")
    return " ； ".join(out[:3])


# --------------------------------------------------------------------------
# --fix：按期望串改写文档
#
# 关键取舍：**不猜"哪个数字是测试数"，而是用期望串里数字的上下文当锚点**。
# 期望串本身就是模板（"全套 {n} 个，约 6 分钟"），把其中的数字挖成捕获组，
# 得到 "全套 (?P<num>…) (?P<gap>\s*)个，约 6 分钟"——
# 「6 分钟」在模板里是字面量，改不到它。这比"整行找数字再判断"可靠得多，
# 也不会因为文档里多了一个数字就误伤。
# --------------------------------------------------------------------------
_NUM_TOKEN = rf"(?:[0-9]+|[{CN_DIGITS}]+)"

# 数量 token 的**两种锚点**（都是"描述测试数量"的既有写法）：
#   1. `24 个` / `二十四个` —— 正文；
#   2. `回归测试 ×22` —— 架构图里省掉量词，只留乘号（`×22` 读作"22 个"）。
# 锚点必须是**形状**（数字紧邻某个固定字），不能是"这行里第一个数字"：
# 架构图那行还有「实时程序」「离线批处理」，按"第一个数字"会改错东西。
_ANCHORS = (
    re.compile(rf"(?P<num>{_NUM_TOKEN})(?P<gap>\s*)个"),
    re.compile(rf"×\s*(?P<num>{_NUM_TOKEN})"),
)


def _count_pattern(target: str) -> re.Pattern[str] | None:
    """把期望串转成"可改写"的正则；认不出数量 token 则返回 None。

    只认上面两种锚点之内的数字，据此锁定 token，其它数字一律留在模板外
    （模板里「约 6 分钟」「20 对」是字面量，`sub` 碰不到）。
    模板里数字两侧的空白放宽成 `\\s*`：同一句话在正文里可能多一个或少一个空格
    （模板写 `**二十四个**`、文档写 `**23 个**`，就是这一处差异）。
    """
    m = None
    for anchor in _ANCHORS:
        m = anchor.search(target)
        if m:
            break
    if m is None:
        return None

    pre, post = target[:m.start("num")], target[m.end("num"):]
    pre_rx = re.escape(pre.rstrip()) + r"\s*"
    post_rx = r"\s*" + re.escape(post.lstrip())
    return re.compile(pre_rx + rf"(?P<num>{_NUM_TOKEN})" + post_rx)




def _rewrite(pat: re.Pattern[str], text: str, n: int) -> tuple[str, int]:
    """按文档**原有写法**选数字样式：原来写中文就写中文，写阿拉伯就写阿拉伯。

    强行统一文体是一次无意义的大改（见上面"两种写法并存"的说明）。

    ⚠️ 替换的是**整个匹配**，所以要把数字之外的部分原样拼回去 ——
    第一版直接 `return new + gap + "个"`，把模板前缀（"启动程序.bat test full # 全套"）
    一起吞掉了，4 行命令只剩"24 个，约 6 分钟"。**改写必须是等长替换思维**：
    只动 token，其余字节逐字保留。
    """
    def repl(mo: re.Match[str]) -> str:
        old = mo.group("num")
        new = cn_number(n) if old[0] in CN_DIGITS else str(n)
        start = mo.start("num") - mo.start(0)
        whole = mo.group(0)
        return whole[:start] + new + whole[start + len(old):]
    return pat.subn(repl, text)


def _test_description(path: Path) -> str:
    """从测试模块的文档串首行取一句说明，供 §7 表使用。

    不手写描述：手写的说明在这张表里已经过期过（加了测试却忘了补一行），
    而从被测文件自己身上取，永远不会对不上。
    """
    src = path.read_text(encoding="utf-8")
    m = re.match(r'\s*[rbuf]*("""|\'\'\')(.*?)\1', src, re.S)
    if not m:
        return "（自动登记：模块无文档串）"
    first = next((ln.strip() for ln in m.group(2).splitlines() if ln.strip()), "")
    first = first.rstrip("。.;；")
    if len(first) > 60:
        first = first[:60] + "…"
    return first or "（自动登记：模块无文档串）"


def fix_counts(n: int) -> tuple[list[str], list[str], int]:
    """按期望表改文档。返回 (已改清单, 改不了清单, 总处数)。"""
    cache: dict[str, str] = {}
    hits: dict[str, int] = {}
    unfixable: list[str] = []

    for wants, rel in expected_test_count_strings(n):
        p = ROOT / rel
        if not p.exists():
            unfixable.append(f"{rel}：文件不存在")
            continue
        text = cache.get(rel) or p.read_text(encoding="utf-8")
        if any(w in text for w in wants):
            continue                      # 这条已经是对的，跳过
        total = 0
        for w in wants:
            pat = _count_pattern(w)
            if pat is None:
                continue
            text, k = _rewrite(pat, text, n)
            total += k
        if total:
            cache[rel] = text
            hits[rel] = hits.get(rel, 0) + total
        else:
            # 不静默：说明"说法形态变了"（换了措辞/换了句式），模板套不上。
            unfixable.append(f"{rel}：找不到可改的旧说法（期望「{wants[0]}」）")

    for rel, text in cache.items():
        (ROOT / rel).write_text(text, encoding="utf-8")
    return [f"{rel}：{c} 处" for rel, c in hits.items()], unfixable, sum(hits.values())


def fix_agents_registry() -> tuple[list[str], list[str]]:
    """把磁盘上多出来、`AGENTS.md` §7 却没登记的测试补进表里。

    插入位置取**字母序**（`run_tests.py --list` 的枚举顺序也是字母序，两者一致）。
    理由：§7 表原是按"加入时间"排的，那个顺序不在任何地方留档，"上次加到哪了"
    只能靠记忆——记错就是下一次漂移；字母序有唯一解。
    副作用要知道：表本身不是字母序，所以新行可能落在表中部甚至表头
    （`selftest_plugins.py` 就排在 `selftest_synthetic.py` 之前）。
    这里选"位置可复现"而不是"位置好看"。
    """
    path = ROOT / "AGENTS.md"
    text = path.read_text(encoding="utf-8")
    actual = {p.name for p in test_files()}
    missing = sorted(actual - set(re.findall(r"`(selftest_[a-z0-9_]+\.py)`", text)))
    if not missing:
        return [], []

    lines = text.splitlines()
    try:
        hdr = next(i for i, ln in enumerate(lines) if ln.strip() == "| 测试 | 守什么 |")
    except StopIteration:
        return [], ["AGENTS.md：找不到 §7 的「| 测试 | 守什么 |」表头，请手工登记"]

    # 表体 = 表头后的分隔行 + 连续的 `|` 行
    end = hdr + 2
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    rows = lines[hdr + 2:end]

    added: list[str] = []
    for name in missing:
        row = f"| `{name}` | {_test_description(ROOT / 'tests' / name)} |"
        # 第一个"文件名比它大"的行之前插入；没有则追加到表尾
        pos = len(rows)
        for i, ln in enumerate(rows):
            m = re.search(r"`(selftest_[a-z0-9_]+\.py)`", ln)
            if m and m.group(1) > name:
                pos = i
                break
        rows.insert(pos, row)
        added.append(name)

    new_text = "\n".join(lines[:hdr + 2] + rows + lines[end:]) + "\n"
    path.write_text(new_text, encoding="utf-8")
    return added, []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="派生事实校验器（计数 / 清单）")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true",
                      help="只校验（默认行为也是校验；保留该参数为了与文档里的写法一致）")
    mode.add_argument("--fix", action="store_true",
                      help="按同一份期望表把文档改对；改完再校验一次")
    args = ap.parse_args(argv)

    n = len(test_files())
    cn = cn_number(n)

    if args.fix:
        print("=" * 76)
        print(f"派生事实同步：按期望表改文档（测试数应为 {n}）")
        print("=" * 76)
        fixed, unfixable, total = fix_counts(n)
        added, reg_problems = fix_agents_registry()
        unfixable.extend(reg_problems)
        if fixed:
            print(f"已改 {len(fixed)} 份文档、{total} 处：")
            for x in fixed:
                print(f"  · {x}")
        else:
            print("文档无需修改（测试数量说法已是最新）")
        if added:
            print(f"AGENTS.md §7 补登记：{', '.join(added)}")
        else:
            print("AGENTS.md §7 无需补登记")
        if unfixable:
            print()
            print(f"**{len(unfixable)} 处改不了**（说法形态变了，模板套不上，请手工核对）：")
            for x in unfixable:
                print(f"  · {x}")
            print()
            print("判定结果：部分未改，请手工处理后重跑")
            return 1
        print()
        print("现在再校验一次：")
        print()

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
        if args.fix:
            print("（--fix 之后仍有不一致：多半是 §3 分组/缺文件这类结构性事实，"
                  "本工具只改计数与 §7 登记，其余请手工处理）")
        return 1

    print(f"✅ 分组清单：{len(disk)} 个工具全部被 AGENTS.md §3 归类，计数一致")
    print(f"✅ 测试数量：{len(expected_test_count_strings(n))} 处说法全部为 {n}"
          f"（「{cn}」或「{n}」），AGENTS.md §7 登记完整")
    print()
    print("判定结果：全部通过")
    if args.fix:
        print("✅ 现在一致了（文档已按磁盘事实对齐，再跑 --fix 不会有改动）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
