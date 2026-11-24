"""解析链（Unity: LocalizedString resolve / StringDatabase.GetLocalizedString）。

一次取值走完这条链，每步都有明确结果，**不静默吞掉缺译**：

    ref → Catalog 解析出稳定 id
        → 当前 locale 的表
        → fallback 链上的表
        → 源语言文本（source）
        → ref 本身（未注册时回退显示引用，一眼可见）

``missing()`` 给出所有"未注册 / 缺译"记录，是判据的抓手：
旧系统的 ``--check`` 用同一个有盲区的提取器扫源码，会假绿；
这里按**实际发生的取值**记账，漏了什么一查就知道。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_CN = re.compile(r"[\u4e00-\u9fff]")

from apps.localization.catalog import TableCollection
from apps.localization.locale import LocaleRegistry
from apps.localization.smart import format_smart


@dataclass(frozen=True)
class Resolution:
    """一次解析的结果。``hit`` 为 None 表示没在任何语言表里命中。"""

    text: str
    entry_id: str | None
    hit: str | None
    missing: bool


class Resolver:
    def __init__(self, collection: TableCollection, registry: LocaleRegistry) -> None:
        self.collection = collection
        self.registry = registry
        self._missing: dict[str, int] = {}

    # ---- 取值 ------------------------------------------------------------- #
    def resolve(self, ref: str, *, locale: str | None = None, **args) -> str:
        return self.resolve_detail(ref, locale=locale, **args).text

    def resolve_detail(self, ref: str, *, locale: str | None = None,
                       **args) -> Resolution:
        """按回退链解析；中文模式直接返回源文本（免查表）。"""
        at = locale or self.registry.current()
        chain = self.registry.chain(locale)
        entry, value, hit = self.collection.lookup(ref, chain)

        # 源语言：源表命中就用源表（允许人工在 zh-CN 表里润色），
        # 否则用 Entry.source。未注册则原样返回引用。
        if entry is None:
            self._note("unregistered", at, str(ref))
            return Resolution(str(ref), None, None, True)
        if value is None:
            # 源语言（zh-CN）取源文本天经地义，不算缺译；只有非源语言
            # 在整条回退链上都没命中才记账，否则 audit 会一片假红。
            at_locale = self.registry.get(at) if self.registry.has(at) else None
            if at_locale is None or not at_locale.source:
                self._note("untranslated", at, entry.id)
            source_table = self.collection.tables.get("zh-CN")
            source = (source_table.get(entry.id) if source_table else None) \
                or entry.source
            text = format_smart(source, args) if (entry.smart or "{" in source) \
                else source
            return Resolution(text, entry.id, None, True)

        # 命中了回退链上的别的语言 = 目标语言其实缺译。记一笔，让 audit
        # 能报出"哪些句子在 en-US 还是中文"（Unity 的 missing translation）。
        if hit != at:
            self._note("fallback", at, entry.id)
        text = format_smart(value, args) if (entry.smart or "{" in value) else value
        return Resolution(text, entry.id, hit, False)

    # ---- 缺译账本 --------------------------------------------------------- #
    def _note(self, kind: str, locale: str, ref: str) -> None:
        # 未登记引用只有含汉字才算"漏翻界面文案"。"English"/"Lang" 这类
        # 本来就是英文标签，格式片段/数据更不是句子；全收进来 audit 就
        # 被噪声淹没，真缺译反而看不见（实测一次渲染报 63 条假阳性）。
        if kind == "unregistered" and not _CN.search(ref):
            return
        key = f"{kind}:{locale}:{ref}"
        self._missing[key] = self._missing.get(key, 0) + 1

    def missing(self) -> dict[str, int]:
        """当前进程里发生过的缺译/未登记记录（key → 次数）。"""
        return dict(self._missing)

    def missing_count(self) -> int:
        return len(self._missing)

    def clear_missing(self) -> None:
        self._missing.clear()


__all__ = ["Resolution", "Resolver"]
