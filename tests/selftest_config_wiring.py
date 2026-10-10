"""配置接线自测：每个配置项都必须真的被读。

为什么需要它
------------
`config.py` 是"调参的唯一入口"，但如果某个字段**定义了却没人读**，
它就变成了一个**静默的谎**：

    你以为改了阈值 -> 实际代码里是硬编码 -> 行为毫无变化 -> 你还以为调生效了

本项目真实发生过两次（`docs/交接文档.md` §10.7-1）：

- `DetectorConfig.min_score`（默认 0.30）—— `detectors/fusion.py` 实际用硬编码 `0.55`；
- `TrackerConfig.max_q_scale`（默认 64.0）—— `track.py` 实际写死 `self.q_base * 64.0`。

两次都是**只能靠逐行读代码才发现**的，合成真值测试完全测不出来
（因为它不改变行为，只改变"能否被改变"）。

本测试把这类问题变成**机械可查**：

1. **结构检查**：`config.py` 里每个 dataclass 字段，必须在本文件之外**至少被引用一次**；
2. **功能检查**：把配置改成极端值，消费方拿到的值必须跟着变（比 grep 更硬的证据）；
3. **反向自证**：故意构造一个"死字段"，检测器必须报出来 —— 否则本测试是空转的。

已知边界（**别高估结构检查**）
------------------------------
结构检查证明的是"**这个名字在别处被属性访问或关键字实参引用过**"，
**不等于**"这个配置字段真的被读了"。同名的函数参数、别的对象的同名属性，
都能满足它。

反向验证时的实测：把一个已修好的字段只退回**调用点**、却保留同名形参，
结构检查就检不出来 —— 此时是**功能检查**（第 2 项）把它抓住的。

所以两部分是互补的：结构检查便宜且覆盖面广（91 个字段一次扫完），
功能检查精确但只能覆盖逐个写明的字段。**新增关键配置项时，请同时加一条功能检查。**

用法::

    python tests/selftest_config_wiring.py
"""

from __future__ import annotations

import ast
import dataclasses
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
while not (ROOT / "sphere_tracker").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sphere_tracker.config import DetectorConfig, TrackerConfig          # noqa: E402
from sphere_tracker.detectors.fusion import FusionDetector               # noqa: E402
from sphere_tracker.track import KalmanFilter                            # noqa: E402

CONFIG_PY = ROOT / "sphere_tracker" / "config.py"

# 允许"未被读取"的字段 —— 必须写明理由。
# 留空是常态：**新增一条就得解释一次**，这正是这条规矩的价值。
ALLOW_UNREAD: dict[str, str] = {}


def _dataclass_fields() -> dict[str, list[str]]:
    """用 AST 读出 config.py 里所有 dataclass 的字段名（不 import，避免副作用）。"""
    tree = ast.parse(CONFIG_PY.read_text(encoding="utf-8"))
    out: dict[str, list[str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        names = [
            st.target.id
            for st in node.body
            if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name)
        ]
        if names:
            out[node.name] = names
    return out


def _repo_sources() -> list[tuple[Path, str]]:
    out: list[tuple[Path, str]] = []
    for p in ROOT.rglob("*.py"):
        if ".git" in p.parts or p == CONFIG_PY:
            continue
        try:
            out.append((p, p.read_text(encoding="utf-8", errors="ignore")))
        except OSError:
            continue
    return out


def _referenced_names() -> set[str]:
    """收集全仓库里"真正被引用的名字"。

    **用 AST，不用文本 grep。** 为什么这点很重要：

    文本 grep 会把**注释、文档字符串、以及本测试文件自身**里出现的字段名
    也算成"被引用"。于是：

      - 一个只在注释里被提过的死配置，会被误判为"活着"；
      - 本测试用来做反向自证的假字段名，会因为写在自己文件里而**永远检不出来**
        （这个漏洞就是被反向自证那一步抓到的）。

    这里只认两种真实引用：

      - `ast.Attribute`：`self.cfg.min_score` 这种属性访问；
      - `ast.keyword` ：`KalmanFilter(..., max_q_scale=...)` 这种关键字实参。

    刻意**不认** `ast.Name`（裸变量名）—— 局部变量重名会造成大量假阳性，
    而配置字段的现实用法就是上面两种。
    """
    names: set[str] = set()
    me = Path(__file__).resolve()
    for p, text in _repo_sources():
        if p.resolve() == me:
            continue          # 跳过本文件，否则测试里的字面量会自我命中
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.keyword) and node.arg:
                names.add(node.arg)
    return names


def _unread_fields(fields: dict[str, list[str]],
                   referenced: set[str]) -> list[tuple[str, str]]:
    """返回 [(类名, 字段名)]：在本文件之外**没有任何真实引用**的字段。"""
    return [
        (cls, name)
        for cls, names in fields.items()
        for name in names
        if name not in referenced
    ]


# --------------------------------------------------------------------------
def test_no_dead_config() -> bool:
    """结构检查：不存在"定义了却没人读"的配置项。"""
    print("----- 结构检查：配置项是否都被读 -----")
    fields = _dataclass_fields()
    referenced = _referenced_names()
    total = sum(len(v) for v in fields.values())
    print(f"  config.py 里 {len(fields)} 个 dataclass、共 {total} 个字段")
    print(f"  扫描范围：{len(_repo_sources())} 个 .py（AST 解析，"
          f"排除 config.py 与本文件；注释与字符串不算引用）")
    print(f"  收集到 {len(referenced)} 个被属性访问/关键字实参引用过的名字")

    dead = _unread_fields(fields, referenced)
    # 允许清单里的不算
    dead = [(c, n) for c, n in dead if n not in ALLOW_UNREAD]

    if dead:
        print(f"  发现 {len(dead)} 个零引用字段（= 死配置，改了不生效）：")
        for cls, name in dead:
            print(f"    {cls}.{name}")
        print("  修法二选一：把它接到消费方，或删掉它。")
        print("  若确属有意保留，写进本文件的 ALLOW_UNREAD 并说明理由。")
    else:
        print(f"  {total} 个字段全部被引用  通过")
        if ALLOW_UNREAD:
            for name, why in ALLOW_UNREAD.items():
                print(f"    （已豁免 {name}：{why}）")
    return not dead


def test_detector_is_not_vacuous() -> bool:
    """反向自证：故意喂一个死字段，检测器必须报出来。

    不做这一步，本测试可能因为正则写错、路径写错而**永远返回"没问题"** ——
    那就是一个假装在守护的测试。
    """
    print("----- 反向自证：检测器是否真的能发现死字段 -----")
    fake = {"FakeConfig": ["this_field_is_definitely_never_used_anywhere_xyz"]}
    dead = _unread_fields(fake, _referenced_names())
    ok = (len(dead) == 1)
    print(f"  喂入 1 个假死字段，检测器报出 {len(dead)} 个  {'通过' if ok else '不达标'}")

    # 再喂一个**确实存在**的字段，应当**不**被报（证明不是"一律报死"）
    real = {"FakeConfig": ["threshold_px"]}
    dead2 = _unread_fields(real, _referenced_names())
    ok2 = (len(dead2) == 0)
    print(f"  喂入 1 个真实在用的字段（threshold_px），报出 {len(dead2)} 个  "
          f"{'通过' if ok2 else '不达标'}")
    return ok and ok2


def test_wiring_is_functional() -> bool:
    """功能检查：改配置，消费方拿到的值必须跟着变（比 grep 硬的证据）。"""
    print("----- 功能检查：改配置 -> 消费方是否真的跟着变 -----")
    checks: list[tuple[str, bool]] = []

    # --- min_score -> FusionDetector.cfg ---
    d = dataclasses.replace(DetectorConfig(), min_score=0.99)
    got = FusionDetector(d).cfg.min_score
    ok = abs(got - 0.99) < 1e-9
    print(f"  DetectorConfig.min_score=0.99 -> 融合检测器读到 {got}  "
          f"{'通过' if ok else '不达标'}")
    checks.append(("min_score 接到融合检测器", ok))

    # 默认值必须与旧硬编码一致，否则是悄悄改了行为
    default_ms = DetectorConfig().min_score
    ok_default = abs(default_ms - 0.55) < 1e-9
    print(f"  min_score 默认值 = {default_ms}（应为 0.55，与历史硬编码一致）  "
          f"{'通过' if ok_default else '不达标'}")
    checks.append(("min_score 默认值未改变行为", ok_default))

    # --- max_q_scale -> KalmanFilter.q_max ---
    k = KalmanFilter(2, 1.5e2, 2.0, max_q_scale=8.0)
    expect = 1.5e2 * 8.0
    ok_q = abs(k.q_max - expect) < 1e-6
    print(f"  max_q_scale=8.0 -> q_max={k.q_max}（期望 {expect}）  "
          f"{'通过' if ok_q else '不达标'}")
    checks.append(("max_q_scale 接到卡尔曼滤波", ok_q))

    t = TrackerConfig()
    k2 = KalmanFilter(2, t.process_noise, t.measurement_noise, max_q_scale=t.max_q_scale)
    expect2 = t.process_noise * t.max_q_scale
    ok_q2 = abs(k2.q_max - expect2) < 1e-6
    print(f"  走 TrackerConfig 默认值 -> q_max={k2.q_max}（期望 {expect2}）  "
          f"{'通过' if ok_q2 else '不达标'}")
    checks.append(("默认配置下 q_max 与历史一致", ok_q2))

    return all(p for _n, p in checks)


def main() -> int:
    print("=" * 62)
    print("配置接线自测（防止「死配置」：定义了却没人读）")
    print("=" * 62)
    a = test_no_dead_config()
    print()
    b = test_detector_is_not_vacuous()
    print()
    c = test_wiring_is_functional()
    print()
    ok = a and b and c
    print("判定结果：" + ("全部通过" if ok else "有不达标项"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
