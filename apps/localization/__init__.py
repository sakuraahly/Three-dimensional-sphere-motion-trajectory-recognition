"""全新多语言映射系统（Unity Localization 风格）——与旧 ``apps.i18n`` 并存。

## 为什么重做

旧系统把**中文原文当 key**：原文一改（``.replace("**")``、f-string 拼接、
条件表达式换措辞）key 就失配，只能靠 400 行 AST 猜测 + 正则猜运行时整句，
一轮轮补盲区，而 ``--check`` 用同一个有盲区的提取器 → 永远假绿。
这套按 Unity 的官方结构重做，从根上断掉"原文即 key"：

| Unity 概念 | 本系统 | 解决什么 |
| ---------- | ------ | -------- |
| SharedTableData（稳定 ID） | ``Entry.id`` + alias | 改原文不丢译文 |
| ResourceTable（每语言一张） | ``StringTable`` | 加语言只加一张表 |
| Smart Strings | ``format_smart`` | 整句模板 + 变量/复数，不拼碎片 |
| Fallback Locale | ``Locale.fallbacks`` | 缺译有确定顺序且可见 |
| StringChanged | ``registry.on_change`` | 语言切换事件驱动刷新 |
| Missing translation | ``Resolver.missing()`` | 按**实际取值**记账，不假绿 |

## 快速上手

```python
from apps.localization import L, loc, audit

L("camera.status.opening")                  # 当前语言
L("camera.frame.ok", index=0, n=3)          # Smart String 带变量
loc.select("en-US")                         # 切语言（通知订阅者）
audit()                                     # 缺译 / 未登记清单
```

## 与旧系统的边界

- 旧 ``apps.i18n.py`` **一行不改**，仍服务现有 120+ 处调用点；
- ``enable_bridge()`` 可让旧 API 转发到新 resolver（可逆，``disable_bridge()``）；
- 旧词典由 ``import_legacy()`` 导入，已有译文一条不浪费。

新写的文案请用**语义化稳定 id**（如 ``camera.status.opening``），
不要再用中文原文当 id —— 那是这套系统要消灭的东西。
"""
from __future__ import annotations

from apps.localization.bridge import (DEFAULT_LOCALES, create_default,
                                      default_system, disable_bridge,
                                      enable_bridge, import_legacy, is_bridged)
from apps.localization.catalog import Catalog, Entry, StringTable, TableCollection
from apps.localization.locale import Locale, LocaleRegistry
from apps.localization.resolver import Resolver
from apps.localization.smart import format_smart, placeholders

# 默认单例：导入本包即可用；与 enable_bridge() 共用同一实例。
_system = default_system()
loc = _system.registry                 # LocaleRegistry
catalog = _system.collection.catalog   # Catalog
tables = _system.collection           # TableCollection
resolver = _system.resolver            # Resolver


def L(ref: str, **args) -> str:
    """取一条文案（Unity: StringDatabase.GetLocalizedString）。"""
    return resolver.resolve(ref, **args)


def T(entry_id: str, source: str, **kwargs) -> Entry:
    """注册一条新文案，返回其 Entry（幂等：同 id 覆盖）。

    同时把 source 写进源语言表：否则 zh-CN 下取值会走"未译回退"路径，
    audit() 会把自己的源语言误报成缺译（冒烟测试实测踩过）。
    """
    entry = Entry(id=entry_id, source=source, **kwargs)
    catalog.register(entry, replace=True)
    source_table = tables.tables.get("zh-CN")
    if source_table is not None:
        source_table.values[entry_id] = source
    return entry


def set_language(code: str, *, persist: bool = True) -> bool:
    """切换语言（旧系统 UI 仍走自己的 set_language；桥接开启后两者同步）。"""
    return loc.select(code, persist=persist)


def language() -> str:
    return loc.current()


def audit() -> dict[str, int]:
    """当前进程里实际发生的缺译 / 未登记记录。空字典 = 干净。"""
    return resolver.missing()


def coverage() -> dict[str, float]:
    return tables.coverage(loc.codes())


__all__ = [
    "L", "T", "loc", "catalog", "tables", "resolver",
    "set_language", "language", "audit", "coverage",
    "Locale", "LocaleRegistry", "Entry", "Catalog", "StringTable",
    "TableCollection", "Resolver", "format_smart", "placeholders",
    "DEFAULT_LOCALES", "create_default", "import_legacy",
    "enable_bridge", "disable_bridge", "is_bridged",
]
