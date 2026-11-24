"""多语言**渲染级**判据：在 en-US 下真建窗口、注入报错、读控件文本。

为什么单独要这一层（2026-11-23 / 11-24 两次踩坑的教训）：
  * 静态词典自洽（i18n_extract --check 全绿）+ 提取器自证，都挡不住"整块漏翻"——
    上一版相机管理窗口的失败结论写在模块级常量与 f-string 里，提取器看不见，
    但 --check 报"与源码一致"，英文界面照旧满屏中文。
  * 所以必须**跑起来看**：切到 en-US，走一遍报错路径，递归扫控件 text，
    界面骨架里不许残留中文（用户自定义装置名 / 设备名是数据，不算）。

覆盖：
  1. 相机失败结论常量在 en-US 下翻成真英文；
  2. 注入 perror 事件后，卡片状态行 + 主状态条无中文残留；
  3. 切回 zh-CN 再触发，中文正确恢复（_i18n_refresh / forget 没把译文当原文）。

无 GUI 也能跑：Tk 用 withdraw() 隐藏根窗口，不 map、不看屏幕。
"""
from __future__ import annotations

import re
import sys
import tkinter as tk
from pathlib import Path

# 按标记上溯到含 sphere_tracker/ 的仓库根（不能用 parents[1]：深度会变，
# selftest_entrypoints 的第 10 节专门守这条，2026-11-24 实测）。
ROOT = Path(__file__).resolve()
while not (ROOT / "sphere_tracker").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_CN = re.compile(r"[\u4e00-\u9fff]")


def _widget_texts(widget) -> list[str]:
    out: list[str] = []
    try:
        txt = widget.cget("text")
        if isinstance(txt, str) and txt:
            out.append(txt)
    except Exception:                                     # noqa: BLE001
        pass
    for c in widget.winfo_children():
        out.extend(_widget_texts(c))
    return out


def main() -> int:
    checks: list[tuple[str, bool]] = []
    try:
        from apps import i18n
        from apps.cameras_dialog import (
            CamerasDialog, REASON_INVALIDATED, REASON_CANNOT_OPEN,
        )
    except Exception as exc:                              # noqa: BLE001
        print(f"  FAIL 无法导入 i18n / 相机窗口：{exc}")
        print("判定结果：有不达标项")
        return 1

    old_lang = i18n.language()
    old_file = i18n.LANG_FILE
    root = None
    dlg = None
    try:
        i18n.LANG_FILE = ROOT / "config" / "language.txt"
        i18n._current = "en-US"
        checks.append(("en-US 下失败结论翻成真英文（不含中文）",
                       bool(i18n.t(REASON_INVALIDATED))
                       and not _CN.search(i18n.t(REASON_INVALIDATED))
                       and bool(i18n.t(REASON_CANNOT_OPEN))
                       and not _CN.search(i18n.t(REASON_CANNOT_OPEN))))

        root = tk.Tk()
        root.withdraw()
        dlg = CamerasDialog(root)
        dlg.update_idletasks()
        dlg._events.put(("perror", 77, REASON_INVALIDATED,
                         "detail (english-safe)", "QQ"))
        dlg._drain()
        texts = _widget_texts(dlg)
        cn = [t for t in texts if _CN.search(t)]
        checks.append((f"英文模式相机报错路径无中文残留（残留 {len(cn)} 条）",
                       not cn))

        # 切回中文：_i18n_refresh + forget 应让同一句正确恢复中文，
        # 而不是停留在英文（旧版会把译文当原文记账，切不回来）。
        i18n._current = "zh-CN"
        dlg._i18n_refresh()
        zh = [t for t in _widget_texts(dlg) if "采集会话失效" in t]
        checks.append(("切回中文后报错结论正确恢复", bool(zh)))

        # 2026-11-24 第二轮截图：双目窗（大角度）四段说明 + 动态状态行
        # 在英文模式仍是中文。这里真建 App + 双目窗，扫控件文本；用户自定义
        # 的装置名属数据（如"实验性质大角度1"），只剔除它，骨架必须全英文。
        i18n._current = "en-US"
        from apps._gui_app import App
        app = App()
        app.withdraw()
        app.update_idletasks()
        stereo = app.open_stereo("angled")
        stereo.update_idletasks()
        s_texts = _widget_texts(stereo) + [stereo.title()]
        # 剔除用户数据：装置名出现在 "Rig profile: …" 这类句子里
        s_cn = [t for t in s_texts
                if _CN.search(t) and "Rig profile:" not in t
                and "｜" not in t]
        checks.append((f"英文模式双目窗骨架无中文残留（残留 {len(s_cn)} 条）",
                       not s_cn))
        checks.append(("双目窗标题翻成英文",
                       "Wide-angle stereo" in stereo.title()))
        stereo.destroy()
        app.destroy()
    except Exception as exc:                              # noqa: BLE001
        checks.append((f"渲染级检查本身不抛异常（{exc}）", False))
    finally:
        try:
            if dlg is not None:
                dlg.destroy()
        except Exception:                                 # noqa: BLE001
            pass
        try:
            if root is not None:
                root.destroy()
        except Exception:                                 # noqa: BLE001
            pass
        i18n.LANG_FILE = old_file
        i18n._current = old_lang

    bad = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(("  OK   " if ok else "  FAIL ") + name)
    print(f"判定结果：{'全部通过' if not bad else '有不达标项 ' + str(len(bad))}")
    print("=" * 70)
    for b in bad:
        print("  FAIL", b)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
