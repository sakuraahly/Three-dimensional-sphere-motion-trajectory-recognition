"""插件注册表：把"按名字选实现"收成**一张显式表**（见 `docs/插件化架构.md` §1/§2）。

这里只做三件事：`kind → name → factory` 的登记、按名字取实现、出错时列出候选。
登记是**显式**的（import 时执行装饰器），不扫描目录、不用字符串 import ——
这样"有哪些实现"是静态可读的事实，打包、改名、静态分析都不会静默失效（R1）。

边界写在实现里，每条都注明"为什么"：

- **R2** 重名注册默认抛 `ValueError`，`replace=True` 才覆盖 ——
  静默覆盖等于让"最后 import 的那个"偷偷决定行为，是又一个看不出来的坑。
- **R3** 找不到就抛 `UnknownPlugin` 并列出候选，**绝不 fallback 到默认** ——
  拼错名字时悄悄给一个默认实现，比直接报错危险得多。
- **R4** `default(kind)` 必须"恰好一个"，零个或多个都报错 ——
  默认值不允许由 import 顺序决定。
- **R6** `summary` 必填（空串视为缺失），`needs` 声明可选依赖 ——
  GUI/CLI 直接拿 summary 当说明；缺依赖的项可以灰掉，而不是运行到一半 ImportError。
- 错误消息带**可选来源模块**：表为空多半是漏 import，指出模块名比"未知名字"有用。
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

__all__ = [
    "UnknownPlugin",
    "plugin",
    "register",
    "resolve",
    "names",
    "default",
    "describe",
    "clear",
]


class UnknownPlugin(LookupError):
    """按名字找不到插件（R3）。`str(exc)` 里带 kind、请求的名字、候选与来源模块。"""


@dataclasses.dataclass
class _Entry:
    name: str
    factory: Callable[..., Any]
    summary: str
    needs: tuple[str, ...]
    is_default: bool
    source: str


# kind -> {name: entry}；dict 保序，所以 "注册顺序 = 展示顺序" 不用额外维护。
_REGISTRY: dict[str, dict[str, _Entry]] = {}


def _key(value: str) -> str:
    return str(value).strip().lower()


def _sources(kind: str) -> list[str]:
    out: list[str] = []
    for entry in (_REGISTRY.get(kind) or {}).values():
        if entry.source and entry.source not in out:
            out.append(entry.source)
    return out


def _not_found_message(kind: str, name: str) -> str:
    table = _REGISTRY.get(kind) or {}
    if not table:
        known = " / ".join(f"{k}（{len(v)} 项）" for k, v in _REGISTRY.items()) or "（无）"
        return (
            f"插件类型 {kind!r} 下没有任何实现，也就没有 {name!r}。\n"
            f"  已注册的类型：{known}\n"
            "  多半是注册模块还没 import —— R1 要求注册显式发生，不是自动扫描。"
        )
    srcs = "\n".join(f"    - {s}" for s in _sources(kind))
    return (
        f"插件类型 {kind!r} 下没有名为 {name!r} 的实现。\n"
        f"  可用名字（{len(table)} 个，按注册顺序）：{' / '.join(table)}\n"
        f"  这些名字由以下模块注册（若列表不对，先检查是否漏 import）：\n{srcs}"
    )


def register(
    kind: str,
    name: str,
    factory: Callable[..., Any],
    *,
    summary: str,
    default: bool = False,
    needs: tuple[str, ...] | list[str] = (),
    replace: bool = False,
) -> None:
    """登记一个实现。`summary` 必填，`needs` 是它需要的可选依赖（pip 包名）。"""
    kind, name = _key(kind), _key(name)
    if not kind or not name:
        raise ValueError(f"插件 kind/name 不能为空：kind={kind!r} name={name!r}")
    text = str(summary or "").strip()
    if not text:  # R6：summary 是 GUI/CLI 的用户可见说明，空串等于没有说明
        raise ValueError(
            f"插件 {kind}/{name} 缺少 summary（空串视为缺失）。"
            "它会被 CLI/GUI 直接当说明文字用，不是可选项。"
        )
    if not callable(factory):
        raise TypeError(f"插件 {kind}/{name} 的 factory 不可调用：{factory!r}")
    table = _REGISTRY.setdefault(kind, {})
    if name in table and not replace:  # R2：重复注册默认报错，覆盖要显式表态
        raise ValueError(
            f"插件 {kind}/{name} 已由 {table[name].source} 注册过。"
            "确需覆盖请传 replace=True，并在注册处写清覆盖理由。"
        )
    table[name] = _Entry(
        name=name,
        factory=factory,
        summary=text,
        needs=tuple(str(n).strip() for n in needs if str(n).strip()),
        is_default=bool(default),
        source=getattr(factory, "__module__", "") or "?",
    )


def plugin(
    kind: str,
    name: str,
    *,
    summary: str,
    default: bool = False,
    needs: tuple[str, ...] | list[str] = (),
    replace: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """装饰器形式：`@plugin("detector", "color", summary="…", default=True)`。"""

    def _decorate(factory: Callable[..., Any]) -> Callable[..., Any]:
        register(kind, name, factory, summary=summary, default=default,
                 needs=needs, replace=replace)
        return factory

    return _decorate


def resolve(kind: str, name: str) -> Callable[..., Any]:
    """取实现（工厂）。未知名字抛 `UnknownPlugin`，**不 fallback**（R3）。"""
    kind, name = _key(kind), _key(name)
    entry = (_REGISTRY.get(kind) or {}).get(name)
    if entry is None:
        raise UnknownPlugin(_not_found_message(kind, name))
    return entry.factory


def names(kind: str) -> list[str]:
    """该 kind 下所有名字，按注册顺序（= 下拉框展示顺序，R7 的唯一样式来源）。"""
    return list((_REGISTRY.get(_key(kind)) or {}))


def default(kind: str) -> str:
    """取该 kind 的唯一默认名字。零个或多个都报错（R4）。"""
    kind = _key(kind)
    table = _REGISTRY.get(kind) or {}
    hits = [e.name for e in table.values() if e.is_default]
    if len(hits) != 1:  # R4：不靠 import 顺序，也不允许"多个都算默认"
        raise LookupError(
            f"插件类型 {kind!r} 有 {len(hits)} 个默认项（应恰好 1 个）：{hits or '无'}"
            f"{'；已注册：' + ' / '.join(table) if table else ''}"
        )
    return hits[0]


def describe(kind: str) -> list[dict[str, Any]]:
    """给 GUI/CLI 用的元数据：`[{name, summary, needs, is_default}]`，顺序同上。"""
    return [
        {"name": e.name, "summary": e.summary, "needs": list(e.needs),
         "is_default": e.is_default}
        for e in (_REGISTRY.get(_key(kind)) or {}).values()
    ]


def clear() -> None:
    """清空注册表 —— **仅供测试**，生产代码不要调用。"""
    _REGISTRY.clear()
