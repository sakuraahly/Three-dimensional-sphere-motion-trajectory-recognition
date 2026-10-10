"""静默失效自查：找出"声明/处理了，但其实没有出口"的地方。

为什么需要它（这一类 bug 是本项目的头号成本）
--------------------------------------------
已经踩过的同类：配置项定义了没人读（`min_score` / `max_q_scale`）、
`LatestFrameCamera.dropped` 声明后从不自增、叠加层坐标没跟 `viz.scale`。
共同形态都是 —— **程序照跑，输出看起来正常，但某条信息没有任何出口**。

三次都是靠人肉审查发现的。而在最近一次系统审查里，
"字段声明了没人读"这个扫描器**我写了 4 版才对**（详见下面的"踩过的坑"）。
同一个扫描器写四遍，就是该固化的信号（全局 §二：重复 3 遍就自动化）。

查五类
------
| 类 | 查什么 | 为什么危险 |
| -- | ------ | ---------- |
| **A** | 数据类的字段**从来没有被读**（只有赋值/构造） | 你以为它在传递信息，实际是死的 |
| **B** | `except` 块里**只吞不记**（`pass` / 只 `continue` / 只 `return`） | 出事没有任何出口，主循环继续跑 |
| **C** | 与字面量比较的**阈值**散落在多处 | 改一处忘一处，且没有提示 |
| **D** | 标记（`TODO` / `FIXME` / `XXX` / `HACK`） | 多半已解决但注释没删 |
| **E** | 公开函数/方法**零调用者**（疑似死代码） | 与 A 同类 |

它**不判对错**，只把候选列出来 —— 因为"这个字段是给人看的诊断料"这类判断
只有人能做。工具负责不漏，人负责定性。

踩过的坑（写进代码免得重犯）
--------------------------
1. **构造传参不是"读"**。`FrameContext(t=1.0, gray=g)` 里的 `gray=` 是**写**，
   v1 版把它当成了使用，于是报了 21 个候选、**全是假阳性**。
2. **`getattr(obj, "gray", None)` 是读**，但对 `obj.gray` 做正则匹配抓不到它。
   漏了这类写法会给出假的"零使用"。
3. **正则不可靠 → 用字面量包含**。`Select-String` 的 `\\.` 转义在本机对
   `frame.std()) > 1.0` 返回 0 命中，害我把"未完成"写成了"已完成"。
   所以本工具**凡是判断"有没有出现"，都用 `in` 做子串判断**，不用正则。

用法::

    python apps/tools/check_silent_failure.py            # 全部五类
    python apps/tools/check_silent_failure.py --only A B # 只看某几类
    python apps/tools/check_silent_failure.py --strict   # 有 high 就退出码 1
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT,):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SKIP_DIRS = {".git", "__pycache__", "outputs", "legacy", ".venv", "node_modules"}
# 库层：这里的静默失效最严重（工具层出错用户能看见）
LIB_DIRS = ("sphere_tracker", "calib", "stereo")
# 与字面量比较时，这些"显然不是阈值"的常量不算（0/1/2 多为索引、布尔）
TRIVIAL_NUMBERS = {"0", "1", "2", "1.0", "0.0", "0.5", "2.0", "100"}


@dataclass
class Finding:
    kind: str          # A / B / C / D / E
    severity: str      # high / medium / low
    where: str         # 路径:行号
    what: str
    why: str
    verify: str = ""   # 可复现的复核方式


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    scanned_files: int = 0
    scanned_lines: int = 0

    def add(self, f: Finding) -> None:
        self.findings.append(f)


def py_files() -> list[Path]:
    out = []
    for p in ROOT.rglob("*.py"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        out.append(p)
    return sorted(out)


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def is_lib(p: Path) -> bool:
    r = rel(p)
    return any(r.startswith(d + "/") for d in LIB_DIRS)


# --------------------------------------------------------------------------- #
# A. 数据类字段：声明了但从来没被"读"
# --------------------------------------------------------------------------- #
def check_unread_fields(files: list[Path], rep: Report) -> None:
    """`@dataclass` 的字段，全仓只有"写"没有"读"。

    读 = `obj.field`（ast.Attribute 且不是赋值目标）或 `getattr(obj, "field")`。
    写 = 构造关键字传参 / `self.field = ...` / `obj.field = ...` / 注解。
    """
    # 先收集所有 dataclass 的字段
    decls: list[tuple[str, str, int, str]] = []      # (path, class, line, field)
    for p in files:
        try:
            tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if not any(isinstance(d, ast.Name) and d.id == "dataclass" or
                       isinstance(d, ast.Attribute) and d.attr == "dataclass"
                       for d in node.decorator_list):
                continue
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    decls.append((rel(p), node.name, stmt.lineno, stmt.target.id))

    # 再统计全仓的"读"
    reads: dict[str, int] = {}
    for p in files:
        src = p.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        # 赋值目标里的属性不算读
        written: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    for sub in ast.walk(t):
                        if isinstance(sub, ast.Attribute):
                            written.add(id(sub))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and id(node) not in written:
                reads[node.attr] = reads.get(node.attr, 0) + 1
            # getattr(obj, "name") 也是读 —— 这是 v2 版漏掉的
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "getattr" and len(node.args) >= 2
                    and isinstance(node.args[1], ast.Constant)
                    and isinstance(node.args[1].value, str)):
                reads[node.args[1].value] = reads.get(node.args[1].value, 0) + 1

    for path, cls, line, name in decls:
        if reads.get(name, 0) == 0:
            rep.add(Finding(
                "A", "medium", f"{path}:{line}",
                f"`{cls}.{name}` 全仓零读取",
                "字段只被写、没有任何地方读 —— 它传递的信息没有出口",
                f"python -c \"import pathlib;"
                f"print(sum(str(p) for p in pathlib.Path('.').rglob('*.py') "
                f"if '.{name}' in p.read_text(encoding='utf-8',errors='replace')))\"",
            ))


# --------------------------------------------------------------------------- #
# B. except 只吞不记
# --------------------------------------------------------------------------- #
def _except_quality(body: list[ast.stmt]) -> tuple[str, str]:
    """判定一个 except 块的可观测性，返回 (等级, 说明)。

    ⚠️ **"没有 log 调用"不等于"静默"** —— 第一版只看"有没有 print/log"，
    于是 91 条候选里绝大部分是良性的，工具失去意义。本项目的优良做法恰恰是
    **把异常转成结构化返回值**（`ProbeResult(ok=False, error=...)`、
    缓存失败返回 `{}`），那比打日志更好：调用方能接管。

    所以真正的判据是：**调用方能不能区分"失败了"与"本来就是空/无"**。
    区分不了 → 静默。
    """
    # 块里重新抛出 = 没有吞，直接跳过
    for s in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(s, ast.Raise):
            return "ok", "重新抛出（没有吞）"
        # 把异常对象本身存下来 / 转成结构化结果 = 可观测
        if isinstance(s, ast.Call):
            f = s.func
            nm = f.attr if isinstance(f, ast.Attribute) else (
                f.id if isinstance(f, ast.Name) else "")
            # ⚠️ 这里是**精确相等**，不是子串包含。
            # 第一版名单里混进了 `setdefault`，而判据用了 `nm in {…}` —— 看起来没问题，
            # 但真正的坑在**名单本身**：我曾把判据写成"nm 是名单里某项的子串"，
            # 于是 `time.sleep(0.05)` 的 `sleep` 命中了 `setdefault`，
            # 把"纯静默重试"误判成"有记录"，害得 `sources.py:149`
            # （本轮最该抓的那条）只报了 medium。
            # **字符串包含判断在短名字上极不可靠** —— 同 `std()) > 1.0` 那个坑。
            if nm in {"warning", "warn", "error", "info", "debug", "exception",
                      "critical", "print", "append", "setdefault", "note",
                      "record", "log"}:
                return "ok", f"有记录动作（{nm}）"
        # `return XxxResult(ok=False, error=...)` / `return {}` / `return None, err`
        if isinstance(s, ast.Return) and s.value is not None:
            if isinstance(s.value, ast.Call):
                return "ok", "返回结构化结果（调用方可判）"
            if isinstance(s.value, ast.Dict):
                return "ok", "返回空字典（等价于「没有」）"
            if isinstance(s.value, ast.Constant) and s.value.value in (None, False, "", 0):
                return "ok", "返回默认值（None/False/空）"
            if isinstance(s.value, ast.Tuple):
                return "ok", "返回元组（多半带错误信息）"
        # 存进对象属性 = 至少可观测（Health/failed 之类）
        if isinstance(s, (ast.Assign, ast.AugAssign)):
            targets = s.targets if isinstance(s, ast.Assign) else [s.target]
            for t in targets:
                for sub in ast.walk(t):
                    if isinstance(sub, ast.Attribute):
                        return "ok", "写进对象属性（可观测）"
            return "ok", "写进变量"

    # 走到这里 = 块里只有 pass / continue / break / 裸 return
    kinds = sorted({type(s).__name__ for s in body})
    only_continue = all(isinstance(s, ast.Continue) for s in body)
    only_pass = all(isinstance(s, ast.Pass) for s in body)
    if only_continue:
        return "high", "循环里静默重试（只有 continue）"
    if only_pass:
        return "high", "完全静默（只有 pass）"
    return "medium", f"只吞不记（块内只有 {', '.join(kinds)}）"


def _enclosing_loop_kinds(tree: ast.AST) -> dict[int, str]:
    """给每个节点的行号标注"它被哪种循环包着"。用于判断"静默重试"。"""
    out: dict[int, str] = {}

    def walk(node: ast.AST, loop: str) -> None:
        for child in ast.iter_child_nodes(node):
            kind = loop
            if isinstance(child, ast.While):
                # while True / while not stop —— 采集线程的典型形态
                kind = "while"
            elif isinstance(child, (ast.For, ast.AsyncFor)):
                kind = loop or "for"
            # 不是所有子节点都有 lineno（如 ast.Store / ast.Load 这类上下文节点）
            lineno = getattr(child, "lineno", None)
            if lineno is not None:
                out[lineno] = kind
            walk(child, kind)

    walk(tree, "")
    return out


def check_silent_except(files: list[Path], rep: Report) -> None:
    """`except` 块里**调用方无法区分"失败"与"本来就是空"**的那些。

    ⚠️ **严重性只由一条可证明的规则决定：`except` 块里有 `continue`/`break`
    且在循环体内 → 「静默重试」→ high。** 走过的弯路：

    - 第一版"库层 + 无出口 = high" → 44 条 high，PPT 工具里的裸 `pass` 全进来了；
    - 第二版"库层 + 只有 pass = high" → 3 条，但 `sources.py:118`
      （**保存**缓存失败）也被报 high —— 那一条是**可接受**的
      （缓存是派生数据，失败只是下次启动重探）。
    - 最终：**语法上分不出"读失败"与"写失败"的严重性**，
      只有"循环里静默重试"能被证明危险（主循环会一直跑、拿到冻结的旧帧）。
      所以 high 只留这一条，其余一律 medium/low。**宁可漏报严重性，不要谎报。**
    """
    for p in files:
        src = p.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        loops = _enclosing_loop_kinds(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            grade, why = _except_quality(node.body)
            if grade == "ok":
                continue
            in_loop = loops.get(node.lineno, "") != ""
            # `continue` = 循环继续转（静默重试）；`break` = 循环退出，不是重试
            silent_retry = any(isinstance(s, ast.Continue) for s in node.body) and in_loop
            if silent_retry and is_lib(p):
                sev = "high"
            elif silent_retry or (grade == "high" and is_lib(p)):
                sev = "medium"
            elif is_lib(p):
                sev = "medium"
            else:
                sev = "low"
            hint = ""
            if sev == "high":
                hint = ("库层循环里静默重试：调用方无法区分「失败了」与「本来就是空」，"
                        "循环会一直转下去（采集线程 → 主循环拿到冻结的旧帧，日志无一行）")
            rep.add(Finding(
                "B", sev, f"{rel(p)}:{node.lineno}",
                f"`except` {why}" + ("，且在循环里" if in_loop else ""),
                hint or "调用方无法区分「失败了」与「本来就是空/无」",
                f"见 {rel(p)}:{node.lineno}-{node.body[-1].lineno}",
            ))


# --------------------------------------------------------------------------- #
# C. 阈值与字面量比较且散落
# --------------------------------------------------------------------------- #
def check_scattered_thresholds(files: list[Path], rep: Report) -> None:
    """同一个数值在**多个文件**里与变量比较 —— 疑似同一个阈值抄了多份。

    只在"跨文件、≥2 个文件"时报，避免把"一个文件里出现两次"也算上。
    """
    hits: dict[str, list[str]] = {}
    for p in files:
        try:
            tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            for c in node.comparators:
                if isinstance(c, ast.Constant) and isinstance(c.value, (int, float)):
                    key = repr(c.value)
                    if key in TRIVIAL_NUMBERS or isinstance(c.value, bool):
                        continue
                    # 左边要是个"算出来的东西"，不是配置读取（配置项本来就在 config.py 定义）
                    left = node.left
                    if isinstance(left, ast.Attribute) and isinstance(left.value, ast.Name) \
                            and left.value.id in {"cfg", "self"}:
                        continue
                    hits.setdefault(key, []).append(f"{rel(p)}:{node.lineno}")
    for val, where in sorted(hits.items()):
        fileset = {w.split(":")[0] for w in where}
        if len(where) >= 2 and len(fileset) >= 2:
            rep.add(Finding(
                "C", "medium", where[0],
                f"数值 {val} 在 {len(fileset)} 个文件、{len(where)} 处与变量比较",
                "疑似同一个阈值抄了多份 —— 改一处忘一处不会有提示",
                "；".join(where[:6]),
            ))


# --------------------------------------------------------------------------- #
# D. 标记
# --------------------------------------------------------------------------- #
MARKERS = ("TODO", "FIXME", "XXX", "HACK")


def check_markers(files: list[Path], rep: Report) -> None:
    """代码里的 TODO / FIXME / XXX / HACK。用字面量包含（不用正则）。"""
    for p in files:
        for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            for m in MARKERS:
                if m in line and "noqa" not in line and "MARKER" not in line:
                    rep.add(Finding(
                        "D", "low", f"{rel(p)}:{i}",
                        f"标记 {m}：{line.strip()[:70]}",
                        "多半已解决但注释没删 —— 会让人以为还是空模板",
                    ))
                    break


# --------------------------------------------------------------------------- #
# E. 公开函数零调用者
# --------------------------------------------------------------------------- #
def check_zero_caller_functions(files: list[Path], rep: Report) -> None:
    """库层里定义的公开函数/方法，全仓没有调用点（疑似死代码）。"""
    defined: list[tuple[str, int, str, str]] = []   # (path, line, name, kind)
    for p in files:
        if not is_lib(p):
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("_"):
                    continue
                if any(isinstance(d, ast.Name) and d.id in {"property", "staticmethod",
                                                            "classmethod"}
                       for d in node.decorator_list):
                    pass
                defined.append((rel(p), node.lineno, node.name, "def"))

    # 全仓的名字出现次数（字面量包含，含注释/文档 —— 只要有人提到就算"有出口可能"）
    corpus = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in files)
    for path, line, name, _kind in defined:
        # 出现 1 次 = 只有定义那一行本身
        if corpus.count(name) <= 1:
            rep.add(Finding(
                "E", "low", f"{path}:{line}",
                f"`{name}()` 只出现 1 次（就是定义处）",
                "疑似死代码；也可能是留给外部调用的公开 API",
            ))


CHECKS = {
    "A": ("数据类字段零读取", check_unread_fields),
    "B": ("except 只吞不记", check_silent_except),
    "C": ("阈值散落", check_scattered_thresholds),
    "D": ("TODO/FIXME 标记", check_markers),
    "E": ("库层零调用者函数", check_zero_caller_functions),
}


def main() -> int:
    ap = argparse.ArgumentParser(description="静默失效自查")
    ap.add_argument("--only", nargs="*", choices=sorted(CHECKS), default=None,
                    help="只跑指定类别")
    ap.add_argument("--strict", action="store_true",
                    help="有 high 级别发现时退出码 1（默认永远 0：它是普查工具，不是闸门）")
    ap.add_argument("--severity", choices=["high", "medium", "low"], default="low",
                    help="只显示不低于该级别的发现")
    args = ap.parse_args()

    files = py_files()
    rep = Report(scanned_files=len(files))
    rep.scanned_lines = sum(len(p.read_text(encoding="utf-8", errors="replace").splitlines())
                            for p in files)
    print("=" * 74)
    print("静默失效自查（声明了但没出口 / 吞异常 / 阈值散落 / 标记 / 死代码）")
    print("=" * 74)
    print(f"扫了 {rep.scanned_files} 个 .py、{rep.scanned_lines} 行\n")

    wanted = args.only or sorted(CHECKS)
    for k in wanted:
        title, fn = CHECKS[k]
        before = len(rep.findings)
        fn(files, rep)
        print(f"----- {k}. {title}：{len(rep.findings) - before} 条 -----")

    order = {"high": 0, "medium": 1, "low": 2}
    rep.findings.sort(key=lambda f: (order[f.severity], f.kind, f.where))
    shown = [f for f in rep.findings if order[f.severity] <= order[args.severity]]

    for sev in ("high", "medium", "low"):
        group = [f for f in shown if f.severity == sev]
        if not group:
            continue
        print(f"\n===== {sev.upper()}（{len(group)} 条）=====")
        for f in group:
            print(f"  [{f.kind}] {f.where}")
            print(f"       {f.what}")
            print(f"       为什么危险：{f.why}")
            if f.verify:
                print(f"       复核：{f.verify}")

    counts = {s: sum(1 for f in rep.findings if f.severity == s) for s in order}
    print(f"\n合计 {len(rep.findings)} 条："
          f"high {counts['high']} / medium {counts['medium']} / low {counts['low']}")
    print("本工具只负责**不漏**；哪条是真问题要靠人定性（很多是刻意留的诊断料）。")

    if args.strict and counts["high"]:
        print("\n判定结果：有 high 级别发现")
        return 1
    print("判定结果：全部通过（普查完成，不代表没有上述候选）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
