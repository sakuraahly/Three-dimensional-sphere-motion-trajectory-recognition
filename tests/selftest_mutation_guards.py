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
   命中 >1 处说明改这里会连带改到别处 —— 两种都**跳过并报出**，绝不硬改。

## 它不进 fast 档（会跑十几分钟）

每条变异都要**真跑一遍目标测试**，而 `selftest_task_manager` 一条就 20 秒、
`selftest_stereo_branch` 十几秒。所以本文件**故意设计成慢测试**：
`apps/run_tests.py` 按实测耗时分层，它会自然落到 `full` 档
（`--slow-threshold` 之下的保底名单不包含它）。

用法::

    python tests/selftest_mutation_guards.py            # 全跑
    python tests/selftest_mutation_guards.py --list     # 只列出有哪些变异
    python tests/selftest_mutation_guards.py --only hud_font,kill
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import subprocess
import sys

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
        "target": "启动程序.py",
        "test": "tests/selftest_task_manager.py",
        "why": "stop_task 用 capture_output 调 taskkill 却**不看返回码** —— "
               "「没杀掉」被报成「停好了」，界面显示已停、进程还在、相机仍被占着",
        "old": "        if r.returncode == 0:\n            return \"\"",
        "new": "        if r.returncode == 0:\n            return \"\"\n        return \"\"",
    },
    {
        "key": "kill-lie",
        "target": "启动程序.py",
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
        "old": "        overlay_scale = render_scale * (viz.scale if viz.scale and viz.scale > 0 else 1.0)\n"
               "        if display_width and frame.shape[1] > 0:\n"
               "            scale *= float(display_width) / float(frame.shape[1])",
        "new": "        overlay_scale = render_scale * (viz.scale if viz.scale and viz.scale > 0 else 1.0)\n"
               "        if display_width and frame.shape[1] > 0:\n"
               "            scale *= float(display_width) / float(frame.shape[1])\n"
               "        overlay_scale = scale   # ← 注入：字号退回乘 display_width",
    },
    {
        "key": "launcher-method",
        "target": "启动程序.py",
        "test": "tests/selftest_launcher_params.py",
        "why": "界面默认后端/方法写死回旧值 —— 界面与命令行/文档脱节"
               "（当年界面默认 fusion，从界面走必然吃 265 ms/帧）",
        "old": "        self.method.set(METHODS[0])          # 取列表首项 = 与命令行/文档同一个默认值",
        "new": '        self.method.set("fusion")',
    },
    {
        "key": "launcher-methods-order",
        "target": "启动程序.py",
        "test": "tests/selftest_launcher_params.py",
        "why": "METHODS 首项换回 fusion —— 默认值又变成最慢的那个",
        "old": 'METHODS = ["color", "fusion", "hough", "motion", "yolo", "yolo_sam"]',
        "new": 'METHODS = ["fusion", "color", "hough", "motion", "yolo", "yolo_sam"]',
    },
    {
        "key": "backend-wiring",
        "target": "启动程序.py",
        "test": "tests/selftest_launcher_params.py",
        "why": "标定命令把 --backend 写死 —— 界面没有选择权，"
               "而本机 MSMF 打开两路 34.6 s、DSHOW 11.8 s",
        "old": "        args += self._backend_args()",
        "new": '        args += ["--backend", "msmf"]   # ← 注入：写死回旧行为',
    },
    {
        "key": "auto-backend",
        "target": "stereo/pair_source.py",
        "test": "tests/selftest_pair_source.py",
        "why": "`--backend auto` 被折叠成 INDEX_BACKEND（MSMF）—— "
               "PairSource 里那份 DSHOW→MSMF 成对候选顺序**从没被走到过**",
        "old": '            print("[后端] 未指定 --backend（auto）：按 DSHOW → MSMF 的顺序"\n'
               '                  "**成对**试着打开，两路必须用同一个后端")',
        "new": "            from sphere_tracker.sources import INDEX_BACKEND\n"
               "            api = INDEX_BACKEND          # ← 注入：把 auto 折叠回固定后端",
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
        "old": "    cache = _load_mode_cache()\n"
               "    if not cache.pop(str(index), None):\n        return False",
        "new": "    cache = _load_mode_cache()\n    if True:\n        return False",
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
        "old": '        print(f"  #{index} 打不开：{_first_line(exc)}")\n'
               "        return None, None",
        "new": '        print(f"  #{index} 打不开：{_first_line(exc)}")\n'
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


def run_one(m: dict[str, str]) -> tuple[str, bool, list[str]]:
    """跑一条变异；(状态, 是否合格, 报红项)。状态 ∈ ok/skip/fail/污染。"""
    target = ROOT / m["target"]
    original = target.read_text(encoding="utf-8")
    before = _sha(target)
    hits = original.count(m["old"])
    if hits != 1:
        return (f"锚点命中 {hits} 处（应为 1）—— **源码变了或锚点过期**，已跳过",
                False, [])
    try:
        target.write_text(original.replace(m["old"], m["new"], 1), encoding="utf-8")
        if _sha(target) == before:
            return "注入后源码**没变** —— 判据不可信", False, []
        rc, bad, crashed = _run_test(m["test"])
        if crashed:
            return "目标测试**崩溃**（不是报红）—— 这条变异不算数", False, bad
        if rc == 1 and bad:
            return "", True, bad
        return f"**没报红**（rc={rc}，报红 {len(bad)} 项）—— 判据是弱的或注入无效", False, bad
    finally:
        target.write_text(original, encoding="utf-8")
        if _sha(target) != before:
            # 恢复失败：**立刻中止**，别在污染的工作区上继续叠加变异
            raise SystemExit(
                f"\n!! 致命：{m['target']} 恢复后 SHA256 不一致 —— 已中止。\n"
                f"   请先 `git diff` 检查并按备份还原，再跑本测试。")


def main() -> int:
    ap = argparse.ArgumentParser(description="变异自证总集")
    ap.add_argument("--list", action="store_true", help="只列出变异，不跑")
    ap.add_argument("--only", default="", help="只跑这些 key（逗号分隔）")
    args = ap.parse_args()

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

    results: list[tuple[str, bool, str]] = []
    for m in picked:
        print(f"\n----- {m['key']} -----")
        print(f"  注入 {m['target']}；跑 {m['test']}")
        print(f"  守的是：{m['why']}")
        try:
            note, good, bad = run_one(m)
        except SystemExit:
            raise
        except Exception as exc:                          # noqa: BLE001
            note, good, bad = f"运行异常：{type(exc).__name__}: {exc}", False, []
        if good:
            print(f"  [通过] 注入后目标测试报红 {len(bad)} 项：")
            for b in bad[:3]:
                print(f"         {b[:110]}")
        else:
            print(f"  [不达标] {note}")
            for b in bad[:3]:
                print(f"         {b[:110]}")
        results.append((m["key"], good, note))

    bad_keys = [k for k, ok, _ in results if not ok]
    print("\n" + "=" * 70)
    print(f"合计 {len(results)} 条变异：合格 {len(results) - len(bad_keys)}，"
          f"不合格 {len(bad_keys)}")
    if bad_keys:
        print("不合格的：" + "、".join(bad_keys))
    print("判定结果：" + ("全部通过" if not bad_keys else "有不达标项"))
    return 0 if not bad_keys else 1


if __name__ == "__main__":
    raise SystemExit(main())
