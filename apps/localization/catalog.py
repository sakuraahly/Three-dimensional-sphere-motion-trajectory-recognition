"""Key Registry + 每 Locale 一张 String Table（Unity: SharedTableData + ResourceTable）。

## 与旧系统的根本差别

旧 ``apps.i18n`` 把**中文原文当 key**：原文一改（``.replace("**")``、
f-string 拼接、条件表达式换措辞），key 就对不上，英文悄悄丢。
本模块按 Unity 的做法把两件事拆开：

- ``Entry.id`` 是**稳定标识**（如 ``camera.status.opening``），改名/改原文都不动它；
- ``Entry.source`` 是源语言文本，``Entry.aliases`` 收旧原文供平滑迁移；
- 代码引用 id 或 alias，**不依赖原文长什么样**。

## 数据表

``TableCollection`` 持有一个共享 Catalog + 每个 locale 一张 ``StringTable``：
查词时先由 Catalog 把引用解析成稳定 id，再按回退链逐张表取值。
新增一种语言 = ``registry.register(Locale(...))`` + ``add_table(lang, {...})``，
不碰任何现有代码。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Entry:
    """一条可本地化的文案（Unity: Shared Table Entry）。"""

    id: str                              # 稳定 ID：改名/改原文都不变
    source: str                          # 源语言（zh-CN）文本
    key: str = ""                        # 可读别名，默认同 id
    smart: bool | None = None            # None = 按 source 是否含 {..} 推断
    aliases: tuple[str, ...] = ()        # 旧中文原文等，供迁移期查找
    comment: str = ""                    # 给译者/维护者看的注释

    def __post_init__(self) -> None:
        if not self.key:
            self.key = self.id
        if self.smart is None:
            self.smart = "{" in self.source or any("{" in a for a in self.aliases)


class Catalog:
    """所有语言共享的 key 注册表：id / key / alias → Entry。"""

    def __init__(self) -> None:
        self._by_id: dict[str, Entry] = {}
        self._alias: dict[str, str] = {}          # alias/key/source → id

    def register(self, entry: Entry, *, replace: bool = False) -> None:
        if entry.id in self._by_id and not replace:
            raise ValueError(f"重复的关键字 id：{entry.id!r}")
        self._by_id[entry.id] = entry
        for alias in (entry.key, entry.source, *entry.aliases):
            if alias:
                self._alias[alias] = entry.id

    def find(self, ref: str) -> Entry | None:
        """按 id → key/alias/source 顺序解析一个引用。"""
        if not isinstance(ref, str) or not ref:
            return None
        hit = self._by_id.get(ref)
        if hit is not None:
            return hit
        entry_id = self._alias.get(ref)
        return self._by_id.get(entry_id) if entry_id else None

    def entries(self) -> tuple[Entry, ...]:
        return tuple(self._by_id.values())

    def __len__(self) -> int:
        return len(self._by_id)


class StringTable:
    """单个 locale 的取值表（Unity: ResourceTable）。"""

    def __init__(self, locale: str, values: dict[str, str]) -> None:
        self.locale = locale
        self.values: dict[str, str] = dict(values)

    def get(self, entry_id: str) -> str | None:
        value = self.values.get(entry_id)
        return value if value else None        # 空串 = 未翻译


class TableCollection:
    """共享 key 注册表 + 每个 locale 一张表（Unity: String Table Collection）。"""

    def __init__(self) -> None:
        self.catalog = Catalog()
        self.tables: dict[str, StringTable] = {}

    def add_table(self, locale: str, values: dict[str, str], *,
                  replace: bool = False) -> None:
        if locale in self.tables and not replace:
            raise ValueError(f"重复的语言表：{locale!r}")
        self.tables[locale] = StringTable(locale, values)

    def has_table(self, locale: str) -> bool:
        return locale in self.tables

    def set(self, ref: str, locale: str, value: str) -> None:
        """按语义 key / alias 写一条译文（调用方不必知道内部稳定 id）。"""
        entry = self.catalog.find(ref)
        if entry is None:
            raise KeyError(f"未登记的引用：{ref!r}")
        table = self.tables.get(locale)
        if table is None:
            raise KeyError(f"不存在的语言表：{locale!r}")
        table.values[entry.id] = value

    def lookup(self, ref: str, chain: tuple[str, ...]):
        """解析引用并按回退链取值。

        返回 ``(entry, value, hit_locale)``：

        - 引用没注册 → ``(None, None, None)``；
        - 注册了但所有语言都没译 → ``(entry, None, None)``；
        - 命中 → ``(entry, 译文, 实际命中的 locale)``。
        """
        entry = self.catalog.find(ref)
        if entry is None:
            return None, None, None
        for locale in chain:
            table = self.tables.get(locale)
            if table is None:
                continue
            value = table.get(entry.id)
            if value is not None:
                return entry, value, locale
        return entry, None, None

    def coverage(self, locales: tuple[str, ...]) -> dict[str, float]:
        """每种语言对已注册 key 的覆盖率（0~1）。"""
        total = len(self.catalog)
        if total == 0:
            return {loc: 1.0 for loc in locales}
        out: dict[str, float] = {}
        for loc in locales:
            table = self.tables.get(loc)
            done = sum(1 for e in self.catalog.entries()
                       if table is not None and table.get(e.id))
            out[loc] = done / total
        return out


__all__ = ["Entry", "Catalog", "StringTable", "TableCollection"]
