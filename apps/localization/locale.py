"""Locale 注册、启动选择与回退链（Unity: LocalizationSettings + Locale）。

职责边界：只管"语言代码 → 显示名 → 回退顺序"，不碰任何翻译内容。
与旧 ``apps.i18n.LANGS`` 完全解耦：这里可注册任意语言，新增语言
不必改本文件之外的任何东西。

启动选择顺序（与 Unity 的 Startup Locale Selectors 同构）：

1. 环境变量 ``SM_LANG``（测试 / 临时强制，旧系统也认它，口径一致）；
2. ``config/language.txt``（用户持久化偏好，双系统读同一份，不会打架）；
3. 默认 ``zh-CN``（项目源语言）。

回退链（Unity: fallback locale）：每个 Locale 自带 ``fallbacks``，
查找顺序为"自己 → 逐级回退"，去重保序。en-US 配 ``zh-CN`` 意味着
缺译时显示源语言（中文）——缺译是**可见**的，不是静默 blank。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "sphere_tracker").is_dir())
LANG_FILE = ROOT / "config" / "language.txt"


@dataclass(frozen=True)
class Locale:
    """一种语言/区域。``code`` 用 BCP-47 写法（zh-CN / en-US）。"""

    code: str
    name: str                       # 自显示名（"中文" / "English"）
    fallbacks: tuple[str, ...] = ()  # 缺译时按顺序回退到哪些语言
    source: bool = False            # 是否源语言：源语言取不到译文不算缺译

    def chain(self) -> tuple[str, ...]:
        """本 locale 的查找顺序：自己 → 逐级 fallback，去重保序。"""
        out = [self.code]
        for code in self.fallbacks:
            if code not in out:
                out.append(code)
        return tuple(out)


class LocaleRegistry:
    """Locale 集合 + 当前选择 + 变更订阅（Unity: AvailableLocales / SelectedLocale）。"""

    def __init__(self, *, default: str = "zh-CN", env: str = "SM_LANG",
                 config_file: str | Path = LANG_FILE) -> None:
        self._locales: dict[str, Locale] = {}
        self._default = default
        self._current = default
        self._env = env
        self._config = Path(config_file)
        self._listeners: list = []

    # ---- 注册 / 查询 ------------------------------------------------------ #
    def register(self, locale: Locale) -> None:
        self._locales[locale.code] = locale

    def codes(self) -> tuple[str, ...]:
        return tuple(self._locales)

    def get(self, code: str) -> Locale:
        try:
            return self._locales[code]
        except KeyError:
            raise KeyError(f"未注册的语言 {code!r}；可选 {self.codes()}") from None

    def has(self, code: str) -> bool:
        return code in self._locales

    def current(self) -> str:
        return self._current

    def current_locale(self) -> Locale:
        return self.get(self._current)

    def label(self, code: str | None = None) -> str:
        code = code or self._current
        return self._locales[code].name if code in self._locales else code

    def chain(self, code: str | None = None) -> tuple[str, ...]:
        """给定语言（缺省=当前）的回退查找顺序。"""
        return self.get(code or self._current).chain()

    # ---- 切换 / 订阅 ------------------------------------------------------ #
    def select(self, code: str, *, persist: bool = True) -> bool:
        """切换当前语言；返回是否真的变了。变动时通知订阅者。

        ``persist=False`` 给同步/测试用：只改内存状态，不写用户偏好文件。
        """
        if code not in self._locales:
            raise ValueError(f"不支持的语言 {code!r}；可选 {self.codes()}")
        if code == self._current:
            return False
        self._current = code
        if persist:
            try:
                self._config.parent.mkdir(parents=True, exist_ok=True)
                self._config.write_text(code, encoding="utf-8")
            except OSError:
                pass
        for fn in list(self._listeners):
            try:
                fn(code)
            except Exception:                              # noqa: BLE001
                pass                      # 一个订阅者坏了不该挡住其他刷新
        return True

    def on_change(self, fn) -> None:
        """订阅语言切换（Unity: SelectedLocaleChanged / StringChanged）。"""
        if fn not in self._listeners:
            self._listeners.append(fn)

    def off_change(self, fn) -> None:
        if fn in self._listeners:
            self._listeners.remove(fn)

    # ---- 启动 ------------------------------------------------------------- #
    def detect(self) -> str:
        """按"环境变量 → 本机偏好 → 默认"选出启动语言，不触发订阅。"""
        forced = os.environ.get(self._env)
        if forced in self._locales:
            self._current = forced
            return forced
        try:
            saved = self._config.read_text(encoding="utf-8").strip()
        except OSError:
            saved = ""
        if saved in self._locales:
            self._current = saved
            return saved
        self._current = self._default
        return self._default


__all__ = ["Locale", "LocaleRegistry", "LANG_FILE"]
