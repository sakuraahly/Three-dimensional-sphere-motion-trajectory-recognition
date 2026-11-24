"""新旧桥接：旧 ``apps.i18n`` 一行不改，新系统可逐步接管。

两个方向：

1. **导入旧词典**（``import_legacy``）：把 ``i18n_data.EN_US`` + ``i18n_seed``
   的"中文原文 → 英文"搬进新表。每条原文注册成稳定 id ``legacy.<hash>``，
   原文同时作为 alias —— 已有译文一条不浪费，旧文件仍是权威。
   id 用原文哈希仅用于过渡；新写的文案请用语义化 id（``camera.status.*``）。

2. **可选接管**（``enable_bridge``）：把旧 ``i18n.t/tf/dynamic`` 转发到新
   Resolver。现有 120+ 处调用点**一行都不用改**就能用上新系统；
   不调用本函数则旧行为完全不变。可 ``disable_bridge()`` 随时回退。

为什么不直接改旧系统：用户明确要求"目前的先不要去除"。
桥接是单向的、可逆的，新旧可长期并存，迁移可逐模块进行。
"""
from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import Any

from apps.localization.catalog import Entry, TableCollection
from apps.localization.locale import Locale, LocaleRegistry
from apps.localization.resolver import Resolver

#: 新系统默认支持的语言；fallback 到源语言让缺译可见（Unity: fallback locale）。
DEFAULT_LOCALES = (
    Locale("zh-CN", "中文", source=True),
    Locale("en-US", "English", fallbacks=("zh-CN",)),
)


def _legacy_id(source: str) -> str:
    """过渡用的稳定 id：同样原文总得到同样 id，不依赖遍历顺序。"""
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:10]
    return f"legacy.{digest}"


def _read_legacy_pairs() -> dict[str, str]:
    """读旧词典：种子优先（人工润色过的口径压过自动资源）。"""
    from apps import i18n_data, i18n_seed              # 延迟导入，避免环

    pairs: dict[str, str] = {}
    en_us = getattr(i18n_data, "EN_US", {}) or {}
    pairs.update({str(k): str(v) for k, v in en_us.items() if v})
    pairs.update({str(k): str(v) for k, v in i18n_seed.SEED.items() if v})
    pairs.update({str(k): str(v) for k, v in i18n_seed.PUNCT.items() if v})
    return pairs


def import_legacy(collection: TableCollection, *,
                  locales: tuple[str, ...] = ("zh-CN", "en-US")) -> int:
    """把旧词典导入新表，返回导入条数。重复调用是幂等的。"""
    pairs = _read_legacy_pairs()
    zh_values: dict[str, str] = {}
    en_values: dict[str, str] = {}
    for source, english in pairs.items():
        entry_id = _legacy_id(source)
        if collection.catalog.find(entry_id) is None:
            collection.catalog.register(Entry(
                id=entry_id, source=source, aliases=(source,),
                smart="{" in source,
                comment="从 apps.i18n 词典导入的过渡条目",
            ), replace=True)
        zh_values[entry_id] = source
        en_values[entry_id] = english

    if "zh-CN" in locales and not collection.has_table("zh-CN"):
        collection.add_table("zh-CN", zh_values)
    if "en-US" in locales:
        merged = dict(en_values)
        if collection.has_table("en-US"):
            merged.update(collection.tables["en-US"].values)
        collection.add_table("en-US", merged, replace=True)
    return len(pairs)


def create_default(*, config_file=None) -> SimpleNamespace:
    """建一套可用的默认系统：注册语言 → 选启动语言 → 导入旧词典。"""
    registry = LocaleRegistry(config_file=config_file) if config_file \
        else LocaleRegistry()
    for locale in DEFAULT_LOCALES:
        registry.register(locale)
    registry.detect()

    collection = TableCollection()
    import_legacy(collection)
    resolver = Resolver(collection, registry)
    return SimpleNamespace(registry=registry, collection=collection,
                           resolver=resolver)


_default_system: SimpleNamespace | None = None


def default_system() -> SimpleNamespace:
    """进程级默认系统：``L/T`` 与 ``enable_bridge()`` 必须看同一份数据。

    否则 ``T()`` 注册的新 key 在桥接后取不到（冒烟测试实测踩过）。
    """
    global _default_system
    if _default_system is None:
        _default_system = create_default()
    return _default_system


# --------------------------------------------------------------------------- #
# 可选接管：旧 API → 新 resolver（可回退）
# --------------------------------------------------------------------------- #
_BRIDGED: list = []                    # [(module, name, original)]，供 disable


def enable_bridge(system: SimpleNamespace | None = None) -> Resolver:
    """把旧 ``i18n.t/tf/dynamic`` 转发到新 resolver（幂等）。"""
    import apps.i18n as legacy

    system = system or default_system()
    resolver: Resolver = system.resolver

    def sync() -> None:
        # 旧测试/代码会直接改 legacy._current；以旧系统的当前语言为准，
        # 只同步内存状态，不写偏好文件、不触发订阅。
        if resolver.registry.current() != legacy.language():
            resolver.registry._current = legacy.language()

    def new_t(text):
        if not isinstance(text, str) or not text:
            return text
        if legacy.language() == "zh-CN":
            return text
        sync()
        return resolver.resolve(text)

    def new_tf(template: str, **kwargs) -> str:
        if not isinstance(template, str) or not template:
            return template
        if legacy.language() == "zh-CN":
            try:
                return template.format(**kwargs)
            except (KeyError, IndexError, ValueError):
                return template
        sync()
        return resolver.resolve(template, **kwargs)

    old_dynamic = legacy.dynamic

    def new_dynamic(text):
        if not isinstance(text, str) or not text or legacy.language() == "zh-CN":
            return text
        sync()
        hit = resolver.resolve(text)
        if hit != text:
            return hit
        return old_dynamic(text)       # 旧正则兜底：动态整句仍有得救

    old_set_language = legacy.set_language

    def new_set_language(lang: str) -> None:
        old_set_language(lang)
        if resolver.registry.current() != lang:
            resolver.registry.select(lang, persist=False)

    if not _BRIDGED:
        for module, name, replacement in (
            (legacy, "t", new_t), (legacy, "tf", new_tf),
            (legacy, "dynamic", new_dynamic),
            (legacy, "set_language", new_set_language),
        ):
            _BRIDGED.append((module, name, getattr(module, name)))
            setattr(module, name, replacement)
    return resolver


def disable_bridge() -> None:
    """还原旧 ``i18n`` 的四个函数（回到未接管的原始行为）。"""
    while _BRIDGED:
        module, name, original = _BRIDGED.pop()
        setattr(module, name, original)


def is_bridged() -> bool:
    return bool(_BRIDGED)


__all__ = ["DEFAULT_LOCALES", "create_default", "default_system",
           "import_legacy", "enable_bridge", "disable_bridge", "is_bridged"]
