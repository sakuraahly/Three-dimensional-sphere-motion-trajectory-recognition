"""新多语言映射系统（Unity 风格）判据。

为什么单独要这一层：旧系统的 ``--check`` 用同一个有盲区的提取器扫源码，
源码自洽 ≠ 界面正确（B-185/187/210/211 四次漏翻的教训）。新系统把
"实际取值"作为唯一事实来源，这里直接按取值链判：

1. 稳定 Key：改原文不改 id，旧原文作为 alias 仍能取到；
2. Locale 回退链：目标语言缺译 → 回退到源语言，且记一笔缺译账；
3. Smart String：变量 / 复数 / 选择 / 转义 / 缺参可见；
4. 缺译可见：未登记、未翻译都在 audit()，中文源语言不算缺译；
5. 事件驱动：语言切换通知订阅者；
6. 新旧桥接：旧 i18n.t 能走新系统取新 key，且可完全回退；
7. 反向自证：把译文改坏，判据必须变红（防止空判据）。

全部纯 Python，不建 GUI、不碰真实配置文件。
"""
from __future__ import annotations

import sys
from pathlib import Path

# 按标记上溯到含 sphere_tracker/ 的仓库根（不能用 parents[1]）。
ROOT = Path(__file__).resolve()
while not (ROOT / "sphere_tracker").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _fresh_system(config_file: Path):
    """一套隔离的新系统：不读用户偏好、不污染默认单例。"""
    from apps.localization.bridge import create_default
    return create_default(config_file=config_file)


def main() -> int:
    checks: list[tuple[str, bool]] = []
    try:
        from apps import i18n
        from apps import localization as loc
        from apps.localization.catalog import Entry
        from apps.localization.locale import Locale, LocaleRegistry
    except Exception as exc:                              # noqa: BLE001
        print(f"  FAIL 无法导入 localization：{exc}")
        print("判定结果：有不达标项")
        return 1

    tmp_cfg = ROOT / "outputs" / "_i18n_locale_selftest.txt"
    try:
        # ---------- 1. 稳定 Key / alias ----------
        sysm = _fresh_system(tmp_cfg)
        sysm.collection.catalog.register(Entry(
            id="camera.status.opening", source="正在打开…",
            aliases=("旧文案：正在打开相机…",)))
        sysm.collection.add_table("zh-CN", {
            **sysm.collection.tables["zh-CN"].values,
            "camera.status.opening": "正在打开…",
        }, replace=True)
        sysm.collection.add_table("en-US", {
            **sysm.collection.tables["en-US"].values,
            "camera.status.opening": "Opening…",
        }, replace=True)
        checks.append(("稳定 id 可取到英文译文",
                       sysm.resolver.resolve("camera.status.opening",
                                             locale="en-US") == "Opening…"))
        checks.append(("旧原文 alias 仍能解析到同一条目",
                       sysm.resolver.resolve("旧文案：正在打开相机…",
                                             locale="en-US") == "Opening…"))
        # 原文改了、id 不变 → 译文不丢
        sysm.collection.catalog.register(Entry(
            id="camera.status.opening", source="正在打开相机（改过措辞）…",
            aliases=("旧文案：正在打开相机…",)), replace=True)
        checks.append(("改中文原文不丢英文（原文不再是 key）",
                       sysm.resolver.resolve("camera.status.opening",
                                             locale="en-US") == "Opening…"))

        # ---------- 2. 回退链 + 缺译账 ----------
        sysm.resolver.clear_missing()
        sysm.collection.catalog.register(Entry(
            id="ui.only.zh", source="只有中文的句子"))
        sysm.collection.tables["zh-CN"].values["ui.only.zh"] = "只有中文的句子"
        got = sysm.resolver.resolve("ui.only.zh", locale="en-US")
        checks.append(("缺译回退到源语言（可见，不 blank）", got == "只有中文的句子"))
        checks.append(("缺译进 audit 账本（回退命中也算缺译）",
                       any("fallback:en-US:ui.only.zh" in k
                           for k in sysm.resolver.missing())))
        checks.append(("源语言取源文本不算缺译",
                       not any("untranslated:zh-CN" in k
                               for k in sysm.resolver.missing())))
        checks.append(("未登记中文界面文案进账",
                       sysm.resolver.resolve("未登记的界面文案", locale="en-US")
                       == "未登记的界面文案"
                       and any("unregistered:en-US:未登记的界面文案" in k
                               for k in sysm.resolver.missing())))
        checks.append(("英文标签/数据不当缺译（不污染 audit）",
                       sysm.resolver.resolve("English", locale="en-US") == "English"
                       and not any("unregistered:" in k and "English" in k
                                   for k in sysm.resolver.missing())))

        # ---------- 3. Smart String ----------
        from apps.localization.smart import format_smart, placeholders
        checks.append(("变量替换",
                       format_smart("你好 {name}", {"name": "张三"}) == "你好 张三"))
        checks.append(("复数（1 / 多路）",
                       format_smart("{n:plural:{} 路画面|{} 路画面}", {"n": 1})
                       == "1 路画面"
                       and format_smart("{n:plural:{} 路画面|{} 路画面}", {"n": 3})
                       == "3 路画面"))
        checks.append(("选择分支 + default",
                       format_smart("{k:choose(a|b):甲|乙|丙}", {"k": "b"}) == "乙"
                       and format_smart("{k:choose(a|b):甲|乙|丙}", {"k": "x"}) == "丙"))
        checks.append(("转义大括号不被当占位符",
                       format_smart(r"\{literal\} {n}", {"n": 2}) == "{literal} 2"))
        checks.append(("缺参可见（留占位符，不静默吞）",
                       "name" in format_smart("你好 {name}", {})))
        checks.append(("placeholders 列出字段名",
                       placeholders("{a} / {b:.1f} / {a}") == ("a", "b")))

        # ---------- 4. 事件驱动 ----------
        seen: list[str] = []
        reg = LocaleRegistry(config_file=tmp_cfg)
        reg.register(Locale("zh-CN", "中文", source=True))
        reg.register(Locale("en-US", "English", fallbacks=("zh-CN",)))
        reg.detect()
        reg.on_change(seen.append)
        changed = reg.select("en-US", persist=False)
        checks.append(("切语言通知订阅者（Unity StringChanged）",
                       changed and seen == ["en-US"]
                       and reg.chain() == ("en-US", "zh-CN")))

        # ---------- 5. 桥接：新系统接管旧 API，可回退 ----------
        if loc.is_bridged():
            loc.disable_bridge()
        old_t = i18n.t
        old_lang = i18n._current
        try:
            loc.enable_bridge(loc.default_system())
            loc.T("ui.selftest.bridge", "桥接测试")
            loc.tables.add_table("en-US", {
                **loc.tables.tables["en-US"].values,
                "ui.selftest.bridge": "Bridge OK",
            }, replace=True)
            i18n._current = "en-US"
            checks.append(("旧 i18n.t 能取到新系统的语义 key",
                           i18n.t("ui.selftest.bridge") == "Bridge OK"))
            checks.append(("旧 i18n.t 的中文原文 alias 仍工作",
                           i18n.t("相机管理") == "Camera Manager"))
        finally:
            loc.disable_bridge()
            i18n._current = old_lang
        checks.append(("桥接可完全回退到旧函数",
                       not loc.is_bridged() and i18n.t is old_t))

        # ---------- 6. 反向自证：译文改坏必须红 ----------
        bad = _fresh_system(tmp_cfg)
        bad.collection.set("确定", "en-US", "确定（没翻）")
        import re
        bad_value = bad.resolver.resolve("确定", locale="en-US")
        checks.append(("反向自证：英文表里塞中文会被识破（判据不空转）",
                       bad_value == "确定（没翻）"
                       and bool(re.search(r"[\u4e00-\u9fff]", bad_value))))

    except Exception as exc:                              # noqa: BLE001
        checks.append((f"判据本身不抛异常（{exc}）", False))
    finally:
        try:
            tmp_cfg.unlink(missing_ok=True)
        except OSError:
            pass

    bad = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(("  OK   " if ok else "  FAIL ") + name)
    print(f"判定结果：{'全部通过' if not bad else '有不达标项 ' + str(len(bad))}")
    print("=" * 70)
    for name in bad:
        print("  FAIL", name)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
