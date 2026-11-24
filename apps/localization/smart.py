"""Smart String：整句模板 + 变量 / 复数 / 选择（Unity: Smart Strings）。

旧系统用 ``dynamic()`` 拿正则去**猜**运行时拼出来的中文整句，每加一种
句式就要补一条正则，且数据位容易被误翻。Unity 的做法是：模板里显式
写占位符，翻译时把值传进去，词的顺序由各语言自己的模板决定。

支持的语法（保持小而确定，不做完整 Smart Format）：

===========================  ============================================
``{name}``                   变量；用 ``args[name]`` 填
``{n:plural:one|many}``      复数；``n==1`` 取前者，否则后者，``{}`` 代值
``{k:choose(a|b):X|Y|Z}``    按 ``args[k]`` 选分支，最后是 default
``{x:.1f}``                  其他格式说明透传给 Python ``format``
``\\{`` / ``\\}``            字面大括号
===========================  ============================================

设计约束：**绝不逐词拼接译文**。句子是整条的，变量只是数据槽；
未提供的值原样留成 ``{name}``，让人一眼看出调用点漏传，而不是静默吞掉。
"""
from __future__ import annotations

import re

# 允许一层嵌套的 {}：plural/choose 分支里 `{}` 代表实际值
# （{n:plural:{} file|{} files}）。最内层的 {} 不能被先匹配掉。
_TOKEN = re.compile(r"\\([{}])|\{((?:[^{}]|\{[^{}]*\})*)\}")
_MISSING = object()


def _stringify(value) -> str:
    if value is _MISSING:
        return ""
    return str(value)


def format_smart(template: str, args: dict | None = None) -> str:
    """按 Smart String 语法渲染一条模板。

    非法/缺参都不抛异常：留着占位符或原样值，让缺失在界面上可见。
    """
    if not isinstance(template, str) or not template:
        return template
    args = args or {}

    def replace(match: re.Match) -> str:
        if match.group(1):                       # \{ 或 \}
            return match.group(1)
        body = match.group(2).strip()
        if not body:
            return ""
        if ":" not in body:
            value = args.get(body, _MISSING)
            return "{" + body + "}" if value is _MISSING else _stringify(value)

        name, rest = body.split(":", 1)
        value = args.get(name, _MISSING)

        if rest.startswith("plural:"):
            forms = rest[len("plural:"):].split("|")
            if value is _MISSING:
                return "{" + body + "}"
            chosen = forms[0] if value == 1 else (forms[1] if len(forms) > 1
                                                  else forms[0])
            return chosen.replace("{}", _stringify(value))

        if rest.startswith("choose(") and "):" in rest:
            spec, branches = rest[len("choose("):].split("):", 1)
            keys = [k.strip() for k in spec.split("|")]
            options = branches.split("|")
            if value is _MISSING:
                return "{" + body + "}"
            key = _stringify(value)
            idx = keys.index(key) if key in keys else len(keys)
            idx = min(idx, len(options) - 1) if options else 0
            return (options[idx] if options else "").replace("{}", _stringify(value))

        if value is _MISSING:
            return "{" + body + "}"
        try:
            return format(value, rest)           # .1f / >8 / , 等标准格式
        except (ValueError, TypeError):
            return _stringify(value)

    return _TOKEN.sub(replace, template)


def placeholders(template: str) -> tuple[str, ...]:
    """列出模板里的字段名（供校验：调用点传的参数是否齐全）。"""
    if not isinstance(template, str):
        return ()
    out: list[str] = []
    for match in _TOKEN.finditer(template):
        if match.group(1):
            continue
        body = match.group(2).strip()
        name = body.split(":", 1)[0] if body else ""
        if name and name not in out:
            out.append(name)
    return tuple(out)


__all__ = ["format_smart", "placeholders"]
