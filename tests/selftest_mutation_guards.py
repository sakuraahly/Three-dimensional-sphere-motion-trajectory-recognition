"""变异自证总集：把这一轮手写的那些变异脚本收进回归套件。

## 为什么要有它

2026-10-14 ~ 15 这一轮，为了证明"判据真的会失败"，我临时写了 **8 个变异脚本**，
散在 `outputs/_handoff_read/` 里。而 `outputs/` **被整个 gitignore** ——
也就是说那些证据**不受版本管理，删了就没了**，而且"该跑哪些变异"只活在我的对话里。

按项目规矩（重复 3 遍就固化；确定性产物要有出口），把它们收进 `tests/`：

| 之前 | 现在 |
| ---- | ---- |
| 8 个 `outputs/_handoff_read/mutate_*.py`，各跑各的 | 一个表 + 一个运行器 |
| 手工 `python outputs/.../mutate_x.py` | `python tests/selftest_mutation_guards.py` |
| 不在版本库，不受检查器管 | 入库；`--list` 可查、`--only` 可选跑 |
| 每个脚本各自 `write_text` 恢复 | **统一的备份/恢复契约**（见下） |

## 安全契约（这是它敢改源码的前提）

每条变异都会**临时改写一个真源码文件**，所以：

1. 改动前把原文**读进内存**，并记下 SHA256；
2. 跑完（无论成败）**在 `finally` 里写回**；
3. 写回后**再算一次 SHA256 并断言相等** —— 不等就**大声报错并中止**，
   而不是继续跑下一条（否则会在已污染的工作区上叠加更多变异）；
4. 注入前**先断言锚点唯一**（命中 1 处）。命中 0 处说明源码变了、锚点过期；
   命中 >1 处说明改这里会连带改到别处 —— 两种都**判不合格并中止**（不是"跳过"），
   并打印最接近的那一行 + 可直接粘贴的 `grep -n`；
5. 注入后**先 `compile()`** —— 注入成语法错是"变异体写错了"，不许被"目标测试崩溃"
   伪装成"抓到判据"；
6. 想放行过期锚点只有一条路：`--allow-stale`（默认关），打开后**必须**看到
   "守卫已失效，结果不可信"的大字横幅。

## 它不进 fast 档（会跑十几分钟）

每条变异都要**真跑一遍目标测试**，而 `selftest_task_manager` 一条就 20 秒、
`selftest_stereo_branch` 十几秒。所以本文件**故意设计成慢测试**：
`apps/run_tests.py` 按实测耗时分层，它会自然落到 `full` 档
（`--slow-threshold` 之下的保底名单不包含它）。

用法::

    python tests/selftest_mutation_guards.py            # 全跑（含崩溃恢复自测）
    python tests/selftest_mutation_guards.py --list     # 只列出有哪些变异
    python tests/selftest_mutation_guards.py --only kill,backend-guard
    python tests/selftest_mutation_guards.py --allow-stale   # 逃生口：结果不可信
"""

# 锚点写作约定（2026-10-26 审计后）：
#   * 优先钉**函数名 + 一行关键语句**，别复制整段带缩进的代码 ——
#     改缩进、调列表顺序、加注释都会让整段锚点失配，而测试会"看起来还在跑"；
#   * `old` 必须在目标文件里恰好出现 1 次（本文件会断言）；
#   * 变异体 `new` 必须仍然是合法 Python（本文件会先 compile 再跑测试）。

from __future__ import annotations

import argparse
import contextlib
import difflib
import hashlib
import io
import json
import pathlib
import re
import shlex
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent
while not (ROOT / "sphere_tracker").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------------------- #
# 变异表：每一项 = 一处"把当初的缺陷放回去"，用来证明对应判据真的会红
# --------------------------------------------------------------------------- #
# 字段：
#   key        短名（`--only` 用）
#   target     被注入的源码文件（仓库相对路径）
#   test       跑哪个测试来观察是否报红
#   why        这条变异守的是什么（写清"当初的缺陷"，否则下一个人会删掉它）
#   old/new    注入内容（old 必须在 target 里**恰好出现 1 次**）
MUTATIONS: list[dict[str, str]] = [
    {
        "key": "kill",
        "target": "apps/gui_main.py",
        "test": "tests/selftest_task_manager.py",
        "why": "stop_task 用 capture_output 调 taskkill 却**不看返回码** —— "
               "「没杀掉」被报成「停好了」，界面显示已停、进程还在、相机仍被占着",
        "old": "        if r.returncode == 0:",
        "new": "        if r.returncode >= 0:   # ← 注入：返回码不再有意义（失败也算停好了）",
    },
    {
        "key": "kill-lie",
        "target": "apps/gui_main.py",
        "test": "tests/selftest_task_manager.py",
        "why": "stop_task 永远 return True —— 返回值失去意义（原来是「是否真的下手了」）",
        "old": "        self._kill_leftover(kids, task)\n"
               "        return self._confirm_dead(task, kids)",
        "new": "        self._kill_leftover(kids, task)\n        return True",
    },
    {
        "key": "hud-font",
        "target": "sphere_tracker/pipeline.py",
        "test": "tests/selftest_viz_scale.py",
        "why": "字号退回乘 display_width —— 实时显示（960/1920）把字缩成 10 像素高，"
               "用户原话「字还是如此模糊」",
        # 语义锚点：只钉"把 display_width 折进 scale"的那一行，缩进/注释变了也不易失配
        "old": "            scale *= float(display_width) / float(frame.shape[1])",
        "new": "            scale *= float(display_width) / float(frame.shape[1])\n"
               "        overlay_scale = scale   # ← 注入：字号退回乘 display_width",
    },
    {
        "key": "launcher-method",
        "target": "apps/gui_main.py",
        "test": "tests/selftest_launcher_params.py",
        "why": "界面默认后端/方法写死回旧值 —— 界面与命令行/文档脱节"
               "（当年界面默认 fusion，从界面走必然吃 265 ms/帧）",
        # 2026-10-26 插件化后：界面默认值来自注册表（`plugins.default("detector")`），
        # `METHODS[0]` 这种"列表首项即默认"的隐式约定已被删掉。锚点跟着改成
        # `set(DEFAULT_METHOD)` —— 语义不变：把默认值写死回 fusion。
        "old": "        self.method.set(DEFAULT_METHOD)",
        "new": '        self.method.set("fusion")',
    },
    {
        "key": "launcher-methods-order",
        # 2026-10-26 插件化后：`METHODS` 由注册表按注册顺序生成，"改列表顺序"已无处可改。
        # 语义等价的新锚点：把界面改回**手写清单**、且首项是 fusion ——
        # 正是插件化之前的缺陷（GUI 7 项 / 命令行 5 项各自漂移，首项还被埋在 fusion）。
        "target": "apps/gui_main.py",
        "test": "tests/selftest_launcher_params.py",
        "why": "界面回到手写清单且首项是 fusion —— 清单与注册表漂移，"
               "下拉首项不再是不传参数时真正生效的那个",
        "old": 'METHODS = _plugin_names("detector")',
        "new": 'METHODS = ["fusion", "specular", "color", "hough", "motion", "yolo", "yolo_sam"]'
               "   # ← 注入：手写清单，首项 fusion",
    },
    {
        "key": "backend-wiring",
        "target": "apps/gui_main.py",
        "test": "tests/selftest_launcher_params.py",
        "why": "标定命令把 --backend 写死 —— 界面没有选择权，"
               "而本机 MSMF 打开两路 34.6 s、DSHOW 11.8 s",
        "old": "        args += self._backend_args()",
        "new": '        args += ["--backend", "msmf"]   # ← 注入：写死回旧行为',
    },
    {
        # ⚠️ 这条替换了旧的 `auto-backend`（它的语义已被 2026-10-18 的修复反转：
        # 现在"索引来自角色时锁死 INDEX_BACKEND"才是**正确**行为）。
        # 新变异守住的是同一个真实危害的**反面**：
        # 一旦放手允许"角色名 + DSHOW"，就会**静默开到另一台相机**。
        "key": "backend-guard",
        "target": "stereo/pair_source.py",
        "test": "tests/selftest_pair_source.py",
        "why": "关掉「PnP 位次 + 非 MSMF 后端」守卫 —— "
               "DSHOW 的 #1 是笔记本内置、#2 才是 aoni，于是角色 right 开到内置相机，"
               "两路都报 1920x1080、界面看不出，**标定静默废掉**",
        "old": "        if pnp_derived:",
        "new": "        if False:   # ← 注入：关掉守卫（允许角色名 + 任意后端）",
    },
    {
        "key": "probe-fallback",
        "target": "sphere_tracker/camera_registry.py",
        "test": "tests/selftest_camera_registry.py",
        "why": "探测不回退（只试主后端）—— 主后端坏就整体判「不可用」，"
               "而换个后端往往就能救回来",
        "old": "    for name, api in WINDOWS_BACKENDS:",
        "new": "    for name, api in ():   # ← 注入：不回退",
    },
    {
        "key": "cache-drop",
        "target": "sphere_tracker/sources.py",
        "test": "tests/selftest_camera_registry.py",
        "why": "_drop_mode_cache 永远不删 —— 坏缓存被永久钉住，"
               "每次启动白等 34 s（坑 35「缓存没有过期概念」）",
        "old": "    if not cache.pop(str(index), None):",
        "new": "    if True:   # ← 注入：永远不删（坏缓存被永久钉住）",
    },
    {
        "key": "cache-resolution",
        "target": "sphere_tracker/sources.py",
        "test": "tests/selftest_camera_registry.py",
        "why": "缓存分辨率守卫失效 —— 显式请求 1920x1080 会被缓存里的 640x480 盖过，"
               "两路分辨率不一致、标定直接失败",
        "old": "            if cached and want_wh and cache_wh and cache_wh != want_wh:",
        "new": "            if False:   # ← 注入：不再检查请求分辨率",
    },
    {
        "key": "open-preview-shape",
        "target": "apps/tools/cameras.py",
        "test": "tests/selftest_camera_registry.py",
        "why": "_open_preview 早退分支返回单值 None —— 调用点解包两个值 → "
               "TypeError，**整个 identify 崩掉**（我引入过的真崩溃）",
        # 2026-10-18：_open_preview 拆出 _try_open 后，旧锚点失效。
        # 2026-10-29：加「等多久 + 在线/不在线」文案后**又失效一次** ——
        # 那次改动把那行 print 整个换掉了，于是注入命中 0 处、本条判据静默失效
        # （变异自证自己报了"锚点失效"，这正是它存在的意义）。
        # 现在钉在**当前**唯一那处早退分支上：连 `why = ...` 一起带上 ——
        # 文件里还有三处 `return None, None`（在 `_parse_pick` 里），只给返回行不唯一。
        "old": '        why = ("在线但开不出图" if enumerated else "设备清单里没有这台")\n'
               '        print(f"  ✘ #{index}（{used_name}）{why}，用了 {took:.1f} s："\n'
               '              f"{_first_line(exc)}", flush=True)\n'
               '        return None, None',
        "new": '        why = ("在线但开不出图" if enumerated else "设备清单里没有这台")\n'
               '        print(f"  ✘ #{index}（{used_name}）{why}，用了 {took:.1f} s："\n'
               '              f"{_first_line(exc)}", flush=True)\n'
               "        return None          # ← 注入：形状不一致",
    },
    {
        "key": "port-change-hint",
        "target": "sphere_tracker/camera_registry.py",
        "test": "tests/selftest_camera_registry.py",
        "why": "explain_role_miss 返回空 —— 换 USB 口后只说「不在线」，"
               "不说「换口就会这样」，用户以为相机丢了",
        "old": "    out: list[str] = []\n    if not spec.name:\n        return out",
        "new": "    return []          # ← 注入：不再解释换口\n"
               "    out: list[str] = []\n    if not spec.name:\n        return out",
    },
    {
        "key": "port-change-in-error",
        "target": "sphere_tracker/camera_registry.py",
        "test": "tests/selftest_camera_registry.py",
        "why": "resolve_role 的**非交互报错**里去掉线索行 —— 脚本/子进程里"
               "只有那一句可见，去掉后用户看不到任何下一步",
        "old": '    tips = explain_role_miss(spec, devices, str(role))\n'
               '    extra = ("\\n" + "\\n".join(f"  · {t}" for t in tips)) if tips else ""',
        "new": '    tips = explain_role_miss(spec, devices, str(role))\n'
               '    extra = ""          # ← 注入：非交互报错里不带线索',
    },
]


# --------------------------------------------------------------------------- #
try:                      # 与 selftest 同目录，直接 import；单跑本文件时也在 path 上
    import _journal as _journal_mod
    from _journal import begin, end, recover
except ImportError:       # 被别处以模块方式运行时兜底
    import tests._journal as _journal_mod               # type: ignore
    from tests._journal import begin, end, recover      # type: ignore


# 变异总数下限：删掉一条变异等于关掉一个守卫，而"少跑几条"最容易悄悄发生
MIN_MUTATIONS = 13


def _sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_test(rel: str) -> tuple[int, list[str], bool]:
    """跑一个测试，返回 (退出码, 报红项, 是否崩溃)。"""
    r = subprocess.run([sys.executable, str(ROOT / rel)], capture_output=True,
                       text=True, encoding="utf-8", errors="replace", cwd=str(ROOT))
    out = r.stdout or ""
    bad = [ln.strip() for ln in out.splitlines() if "[不达标]" in ln]
    crashed = "Traceback" in (r.stderr or "")
    return r.returncode, bad, crashed


def _anchor_diagnosis(m: dict[str, str], text: str) -> list[str]:
    """锚点失配时的现场：最接近的一行 + 能直接粘贴的 grep。"""
    rel = m["target"]
    src = text.splitlines()
    probes = [ln.strip() for ln in m["old"].splitlines() if ln.strip()]
    best: tuple[float, int, str] | None = None
    for probe in probes:
        for lineno, line in enumerate(src, 1):
            ratio = difflib.SequenceMatcher(None, probe, line.strip()).ratio()
            if best is None or ratio > best[0]:
                best = (ratio, lineno, line.strip())
    out: list[str] = []
    if best is not None:
        ratio, lineno, line = best
        out.append(f"    ↳ 最接近的是第 {lineno} 行（相似度 {ratio:.0%}）：{line[:100]}")
        key = re.sub(r"\s+", " ", line)[:60]
        if key:
            out.append(f"    ↳ 修锚点：grep -n {shlex.quote(key)} {rel}")
    named = re.search(r"def\s+([A-Za-z_]\w*)\s*\(", m["old"])
    if named and not re.search(rf"def\s+{re.escape(named.group(1))}\s*\(", text):
        out.append(f"    ↳ ⚠ 目标函数 {named.group(1)}() 在 {rel} 里已不存在 —— 改名或搬文件了？")
    return out


def run_one(m: dict[str, str]) -> tuple[str, list[str], list[str], str]:
    """跑一条变异；返回 (状态, 报红项, 诊断行, 说明)。状态 ∈ ok/stale/fail。

    `stale` = 锚点失配（源码变了 / 函数改名 / 文件没了）—— 默认**判不合格**，
    只有 `--allow-stale` 才降级成跳过（见 main）。
    """
    target = ROOT / m["target"]
    if not target.exists():
        return ("stale", [], [
            f"    ✗ 目标文件不存在：{m['target']} —— 被删除还是改名了？",
            "      守卫不能因为源码搬家就消失：请同步改 MUTATIONS 里的 target/old。",
        ], "目标文件缺失")
    original = target.read_text(encoding="utf-8")
    before = _sha(target)
    hits = original.count(m["old"])
    if hits != 1:
        diag = [f"    ✗ 锚点失配：{m['target']} 里命中 {hits} 处（期望恰好 1 处）"]
        diag += _anchor_diagnosis(m, original)
        return ("stale", [], diag,
                f"锚点失效（在 {m['target']} 命中 {hits} 处）—— 守卫已失效")
    try:
        begin(target, original)                 # 崩溃安全：先记账，再动源码
        injected = original.replace(m["old"], m["new"], 1)
        target.write_text(injected, encoding="utf-8")
        if _sha(target) == before:
            return "fail", [], ["    注入后源码**没变** —— 锚点命中的是空串？"], \
                   "注入无效（源码未变）"
        try:
            compile(injected, str(target), "exec")
        except SyntaxError as exc:
            return "fail", [], [
                f"    ✗ 注入无效：注入后的代码不是合法 Python：{exc.msg}（第 {exc.lineno} 行）",
                "      这是变异体写错了，不是「测试抓到了」—— 不许用测试崩溃来掩盖。",
            ], "注入无效（SyntaxError）"
        rc, bad, crashed = _run_test(m["test"])
        if crashed:
            return "fail", bad, [
                "    目标测试**崩溃**（不是报红）—— 已确认注入后语法合法，",
                "    仍判不合格：请修测试或换一种注入表达。",
            ], "目标测试崩溃"
        if rc == 1 and bad:
            return "ok", bad, [], ""
        return "fail", bad, [], f"**没报红**（rc={rc}，报红 {len(bad)} 项）—— 判据是弱的或注入无效"
    finally:
        target.write_text(original, encoding="utf-8")
        if _sha(target) == before:
            end()                               # 只在本条变异彻底还原后销账
        if _sha(target) != before:
            # 恢复失败：**立刻中止**，别在污染的工作区上继续叠加变异
            raise SystemExit(
                f"\n!! 致命：{m['target']} 恢复后 SHA256 不一致 —— 已中止。\n"
                f"   请先 `git diff` 检查并按备份还原，再跑本测试。")


def _selftest_journal() -> list[str]:
    """自测 `_journal.recover()`：未走完的日志要还原并播报；被篡改要中止。

    全程只在临时目录里伪造，**不碰仓库源码**。返回问题清单（空 = 通过）。
    """
    problems: list[str] = []
    clean = "def f():\n    return 1\n"
    dirty = "def f():\n    return 1\nif False:   # ← 注入：上次没走完的残留\n"
    old_journal = _journal_mod.JOURNAL
    with tempfile.TemporaryDirectory() as td:
        target = pathlib.Path(td) / "fake_source.py"
        _journal_mod.JOURNAL = pathlib.Path(td) / "journal.json"
        try:
            # ① 上次未走完：日志里是原文，磁盘上是变异体
            target.write_text(clean, encoding="utf-8")
            begin(target, clean)
            target.write_text(dirty, encoding="utf-8")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                restored = recover()
            if restored != target:
                problems.append("未识别出「上次没走完」的日志（recover 返回空）")
            if target.read_text(encoding="utf-8") != clean:
                problems.append("没有用日志原文还原，或还原内容不对")
            if "[恢复]" not in buf.getvalue():
                problems.append("还原了但没有播报警告")
            if _journal_mod.JOURNAL.exists():
                problems.append("还原成功后日志没销账")

            # ② 日志被篡改（内容与 SHA 不符）：必须中止，且**不覆盖源码**
            target.write_text(dirty, encoding="utf-8")
            begin(target, clean)
            data = json.loads(_journal_mod.JOURNAL.read_text(encoding="utf-8"))
            data["content"] = clean.replace("return 1", "return 999")
            _journal_mod.JOURNAL.write_text(
                json.dumps(data, ensure_ascii=False), encoding="utf-8")
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    recover()
            except SystemExit as exc:
                if "SHA" not in str(exc):
                    problems.append(f"中止了但没说清原因：{exc}")
            else:
                problems.append("日志内容与 SHA 不符时**没有中止**（会拿坏内容覆盖源码）")
            if target.read_text(encoding="utf-8") != dirty:
                problems.append("中止路径竟然改了源码 —— 不能拿不可信的日志去覆盖")
        finally:
            _journal_mod.JOURNAL = old_journal
    return problems



def main() -> int:
    ap = argparse.ArgumentParser(description="变异自证总集")
    ap.add_argument("--list", action="store_true", help="只列出变异，不跑")
    ap.add_argument("--only", default="", help="只跑这些 key（逗号分隔）")
    ap.add_argument("--allow-stale", action="store_true",
                    help="锚点失配时降级为「跳过」（默认是硬失败）。"
                         "打开后结果不可信，只用于临时排查。")
    args = ap.parse_args()

    # 上次若被硬杀，先把变异体还原再动手（否则会在污染的工作区上叠加新变异）
    recovered = recover()
    if recovered is not None and not args.list:
        print(f"  [警告] 上次未走完，已恢复 {recovered.name}；本次结果才可信")

    picked = MUTATIONS
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        picked = [m for m in MUTATIONS if m["key"] in want]
        if not picked:
            print(f"没有匹配的 key：{sorted(m['key'] for m in MUTATIONS)}")
            return 1

    print("=" * 70)
    print("变异自证（把当初的缺陷放回去，证明判据真的会红）")
    print("=" * 70)
    if args.list:
        for m in picked:
            print(f"  {m['key']:24} {m['target']:36} -> {m['test']}")
            print(f"      {m['why']}")
        return 0

    results: list[tuple[str, str, bool, str]] = []      # (key, 状态, 是否合格, 说明)
    extra_fail: list[str] = []
    if len(MUTATIONS) < MIN_MUTATIONS:
        extra_fail.append(f"变异只剩 {len(MUTATIONS)} 条（应 ≥ {MIN_MUTATIONS}）—— 守卫被删了？")

    print("\n----- 日志恢复自测（_journal.recover）-----")
    journal_problems = _selftest_journal()
    if journal_problems:
        print("  [不达标] 崩溃恢复自测未通过：")
        for p in journal_problems:
            print(f"         {p}")
        extra_fail.append("崩溃恢复自测未通过")
    else:
        print("  [通过] 伪造「上次没走完」→ 已还原并警告；日志被篡改 → 中止且不覆盖源码")

    for m in picked:
        print(f"\n----- {m['key']} -----")
        print(f"  注入 {m['target']}；跑 {m['test']}")
        print(f"  守的是：{m['why']}")
        try:
            kind, bad, diag, note = run_one(m)
        except SystemExit:
            raise
        except Exception as exc:                          # noqa: BLE001
            kind, bad, diag, note = "fail", [], [], f"运行异常：{type(exc).__name__}: {exc}"
        if kind == "ok":
            print(f"  [通过] 注入后目标测试报红 {len(bad)} 项：")
            for b in bad[:3]:
                print(f"         {b[:110]}")
            results.append((m["key"], "ok", True, ""))
        elif kind == "stale" and args.allow_stale:
            print(f"  [跳过] {note}（--allow-stale 已放行；这条守卫现在是**哑的**）")
            for d in diag:
                print(d)
            results.append((m["key"], "stale", True, note))
        else:
            print(f"  [不达标] {note}")
            for d in diag:
                print(d)
            for b in bad[:3]:
                print(f"         {b[:110]}")
            results.append((m["key"], kind, False, note))

    bad_keys = [k for k, _, ok, _ in results if not ok]
    stale_keys = [k for k, kind, _, _ in results if kind == "stale"]
    skipped = stale_keys if args.allow_stale else []
    print("\n" + "=" * 70)
    print(f"合计 {len(results)} 条变异：合格 {len(results) - len(bad_keys)}，"
          f"不合格 {len(bad_keys)}，其中被跳过 {len(skipped)}")
    if bad_keys:
        print("不合格的：" + "、".join(bad_keys))
    if skipped:
        print("被跳过的：" + "、".join(skipped))
    if extra_fail:
        print("附加检查不达标：" + "；".join(extra_fail))
    failed = bool(bad_keys or extra_fail)
    print("判定结果：" + ("全部通过" if not failed else "有不达标项"))
    if skipped:
        print("!" * 70)
        print(f"!!  守卫已失效：{len(skipped)} 条变异因锚点失配被跳过，结果不可信  !!")
        print("!!  请去掉 --allow-stale、修好锚点后重跑。                          !!")
        print("!" * 70)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
