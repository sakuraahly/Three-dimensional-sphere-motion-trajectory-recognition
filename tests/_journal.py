"""变异测试的**崩溃安全**日志：注入前记账，恢复后销账，下次启动先对账。

## 为什么需要它（2026-10-26 真实事故）

`selftest_mutation_guards` 原本用 `try/finally` 还原源码，看起来是安全的。
但 `finally` **只在进程正常退出时执行**：超时被杀、Ctrl-C 落在 C 层、
后台任务被回收 —— 任何一次"硬死"都会把变异体留在工作区。

实际发生过：一次后台全套测试在后半段被回收，
`sphere_tracker/sources.py` 里留下了 `if False:   # ← 注入：不再检查请求分辨率`。
更糟的是**没有任何提示** —— 下一次跑全套时，前 13 个测试都是在这份被注入的代码上跑的，
结果"看起来正常"，而库里躺着一段坏代码。

## 契约

1. `begin(target, original)`：把**原文件内容**写进日志（不依赖 git —— 改动可能还没提交，
   `git checkout` 会把用户的工作一起冲掉），再让调用方去写变异体；
2. `end()`：恢复成功后**删日志**；
3. `recover()`：启动时调用。日志存在 → 说明上次没走完，**用日志里的原文还原**、
   校验、删日志，并返回被还原的路径供调用方大声播报；日志不存在 → 返回 None。

日志只记**一条**在途变异（本测试是串行执行的），多条会互相覆盖才是真 bug。
"""

from __future__ import annotations

import hashlib
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
JOURNAL = ROOT / "outputs" / ".mutation_journal.json"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def begin(target: pathlib.Path | str, original: str) -> None:
    """注入前记账：记下**原文**，这样恢复不依赖 git 状态。"""
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    JOURNAL.write_text(
        json.dumps(
            {
                "target": str(pathlib.Path(target).resolve()),
                "sha": _sha(original),
                "content": original,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def end() -> None:
    """恢复成功后销账。"""
    JOURNAL.unlink(missing_ok=True)


def recover(*, quiet: bool = False) -> pathlib.Path | None:
    """对账：上次留下的变异体若还在，用日志原文还原。返回被还原的文件路径。"""
    if not JOURNAL.exists():
        return None
    try:
        data = json.loads(JOURNAL.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        raise SystemExit(
            f"\n!! 变异日志 {JOURNAL} 读不出来。\n"
            f"   请 `git diff` 检查源码，确认没有残留 `# ← 注入` 后再手工删除该日志。"
        )

    target = pathlib.Path(data["target"])
    if not target.exists():
        JOURNAL.unlink(missing_ok=True)
        raise SystemExit(f"\n!! 变异日志指向的文件已不存在：{target}")

    text = data.get("content", "")
    if _sha(text) != data.get("sha"):
        raise SystemExit(
            f"\n!! 变异日志自检失败（内容与 SHA 不符）：{JOURNAL}\n"
            f"   不要用它覆盖源码，先手工核对 {target}。"
        )

    current = target.read_text(encoding="utf-8")
    if _sha(current) == data["sha"]:
        # 文件已经是干净的（上次恢复成功但没来得及销账）
        JOURNAL.unlink(missing_ok=True)
        return None

    target.write_text(text, encoding="utf-8")
    if _sha(target.read_text(encoding="utf-8")) != data["sha"]:
        raise SystemExit(f"\n!! 还原 {target} 后 SHA256 仍不一致 —— 已中止，请手工处理。")
    JOURNAL.unlink(missing_ok=True)
    if not quiet:
        try:
            shown = target.relative_to(ROOT)
        except ValueError:                       # 目标在仓外（测试用的临时文件）
            shown = target
        print(f"[恢复] 上次变异测试未走完，已还原：{shown}")
    return target
