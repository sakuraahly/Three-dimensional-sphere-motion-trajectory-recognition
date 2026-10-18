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


def test_default_method_is_the_fast_one() -> bool:
    """默认检测方法必须是**跑得动**的那个（2026-10-14 改）。

    ## 为什么这条要有护栏

    默认曾经是 `fusion`，而 `fusion` 会把 `color` + `hough` + `motion`
    **依次跑完**、且全在原始分辨率上（检测器不做降采样），于是单帧时间
    约等于三者之和。实测 1920x1080：

        color    17.7 ms  ->  56 fps
        hough    79.9 ms  ->  12.5 fps
        fusion  109.2 ms  ->   9.2 fps   ← 曾是默认

    **9 fps 的实时跟踪是不能用的**（球一帧就移动很远），而"打开就是这样"，
    用户没有对照物，只会觉得"这软件很卡"。用户的原话是"帧率很低，十分不好用"。

    这条断言把"默认必须是快的那个"钉住：谁把它改回 `fusion`，
    这个测试就红，并且红的地方直接带着 fps 对比。
    """
    print("----- 默认检测方法：必须是跑得动的那个 -----")
    checks: list[tuple[str, bool]] = []

    default = DetectorConfig().method
    # 实测单帧耗时（同机、1080p、合成场景）—— 数字写在这里是为了让红的时候
    # 一眼看到代价，而不是去翻历史。
    measured_ms = {"color": 18.4, "motion": 18.5, "hough": 79.9, "fusion": 118.0}
    print(f"  DetectorConfig().method = {default!r}"
          f"（实测 {measured_ms.get(default, '?')} ms/帧）")

    checks.append(("默认方法固定在 **color**（约 56 fps 档）", default == "color"))
    checks.append((f"默认方法的实测耗时 <= 25 ms（实际 "
                   f"{measured_ms.get(default, float('nan'))} ms）",
                   measured_ms.get(default, 1e9) <= 25.0))
    checks.append(("**fusion 不再是默认**（它要 109 ms/帧 = 9 fps）",
                   default != "fusion"))
    # 融合成员表本身没变（改默认不该顺手改融合的内容）
    members = DetectorConfig().fusion_members
    checks.append(("fusion_members 未被顺手改掉（仍是 color/hough/motion）",
                   members == ["color", "hough", "motion"]))
    # fusion 必须仍然存在且能选 —— 不许为了"快"把它删掉
    from sphere_tracker.detectors import build_detector          # noqa: PLC0415
    try:
        build_detector(dataclasses.replace(DetectorConfig(), method="fusion"))
        fusion_ok = True
    except Exception as exc:                                     # noqa: BLE001
        print(f"    构造 fusion 失败：{type(exc).__name__}: {exc}")
        fusion_ok = False
    checks.append(("fusion 仍然能构造（只是要显式选，不是被删了）", fusion_ok))

    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_camera_index_backend_is_explicit() -> bool:
    """用编号开相机的地方，必须**显式**给出后端。

    这条规则是被同一个 bug 咬了两次之后加的：

        `list_devices()` 的编号来自 Windows PnP 枚举，而
        `cv2.VideoCapture(index, backend)` 用的是各后端自己的枚举 —— 两者不同。
        本机实测 PnP 是 [aoni, aoni, 内置, 内置红外]，DSHOW 却是
        [aoni, 内置, aoni, -]。于是"用编号 1 开相机"在 DSHOW 下开到的是**内置**。

    实际踩到的两次：
      1. 双目标定把 `right` 开成了笔记本内置（画面全黑，看起来像相机坏了）；
      2. `cameras.py identify` 里用户看着 A 的画面按 r，配置却记到了 B 上 ——
         因为预览用 auto 后端、标签用 PnP 名字，两者不是同一台。

    所以：`CameraSource(index=...)` 必须带 `backend=`；
    只有把 backend 透传下去的工厂（形如 `**kwargs`）可以豁免 —— 它自己没有决策权。

    这是**静态检查**，不开相机、不需要硬件，所以能进无人值守的回归。
    """
    print("[检查] CameraSource(index=...) 是否显式指定后端")
    print("       （编号跨后端不通用，不指定就会开到另一台相机）")

    bad: list[str] = []
    ok_count = 0
    for path, text in _repo_sources():
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "CameraSource"):
                continue
            kw = {k.arg for k in node.keywords}
            if None in kw:          # **kwargs：后端由调用方决定，工厂本身不决策
                continue
            if "index" not in kw:   # 走默认 index=0，不属于"编号跨枚举"的场景
                continue
            if "backend" in kw:
                ok_count += 1
                continue
            rel = path.relative_to(ROOT)
            bad.append(f"{rel}:{node.lineno}")

    if bad:
        print("  [不达标] 以下调用用编号开相机却没给后端：")
        for b in bad:
            print(f"           {b}")
        print("           应传 backend=sphere_tracker.sources.INDEX_BACKEND"
              "（或明确的 cv2.CAP_*）")
    else:
        print(f"  [通过] {ok_count} 处用编号开相机的地方都显式给了后端")
    return not bad



def test_coast_window_is_frame_rate_independent() -> bool:
    """max_coast_s 必须让「遮挡容忍度」与帧率无关（审查 C3）。

    ## 守的是什么（2026-10-18 实测）

    原来 Track.dead 只数帧：misses > max_coast_frames（12 帧）。
    同一段 **0.3 秒**的遮挡：

        30 fps -> 9 帧  -> 轨迹延续（id 1->1）
        60 fps -> 18 帧 -> 轨迹**断开**（id 1->2，起了一条新轨迹）

    即**帧率翻倍、遮挡容忍度减半**。本项目两路常跑 30/60 fps，
    于是同一套配置在不同帧率下行为不同；而"轨迹断开"会让拖尾与预测历史清零，
    跟踪连续性**静默变差**。

    修法：TrackerConfig.max_coast_s（秒）优先，max_coast_frames 兜底。

    ## 判据

    1. 默认配置里有 max_coast_s 且 > 0；
    2. 同一段 0.3 s 遮挡，30 fps 与 60 fps 下**轨迹 id 都延续**；
    3. max_coast_s = 0 时退回按帧数（老配置仍然能用）。
    """
    print("----- max_coast_s：遮挡容忍度与帧率无关 -----")
    import numpy as np

    from sphere_tracker.config import TrackerConfig
    from sphere_tracker.track import BallTracker
    from sphere_tracker.types import Detection

    def traj(t: float) -> tuple[float, float]:
        return 400.0 * t + 200.0, -900.0 * t + 900.0 * t * t + 400.0

    def run(fps: int, cfg: TrackerConfig, gap_s: float = 0.3,
            seed: int = 0) -> tuple[int | None, int | None]:
        rng = np.random.default_rng(seed)
        tr = BallTracker(cfg)
        dt = 1.0 / fps
        id_before = None
        for i in range(int(1.2 * fps)):
            t = i * dt
            x, y = traj(t)
            dets = ([] if 0.4 <= t < 0.4 + gap_s else
                    [Detection(x=x + rng.normal(0, 1.0),
                               y=y + rng.normal(0, 1.0), radius=12.0, score=0.9)])
            if id_before is None and t >= 0.4:
                id_before = tr.primary.id if tr.primary is not None else None
            tr.update(dets, t)
        after = tr.primary.id if tr.primary is not None else None
        return id_before, after

    base = TrackerConfig()
    checks = [
        ("TrackerConfig 有 max_coast_s 且 > 0", getattr(base, "max_coast_s", 0) > 0),
    ]
    ids = {fps: run(fps, base) for fps in (30, 60)}
    for fps, (a, b) in ids.items():
        print(f"  {fps} fps：id {a}->{b}")
        checks.append((f"{fps} fps 下 0.3 s 遮挡后轨迹 id **延续**", a is not None and a == b))
    checks.append(("30 与 60 fps 的行为一致（同一段物理遮挡）",
                   ids[30][1] is not None and ids[30][1] == ids[60][1]))

    legacy = TrackerConfig()
    legacy.max_coast_s = 0.0          # 退回按帧数
    a, b = run(60, legacy)
    print(f"  兜底（max_coast_s=0，60 fps）：id {a}->{b}")
    checks.append(("max_coast_s=0 时退回按帧数（老配置仍可用，60fps 会断开）",
                   a is not None and a != b))

    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


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
    d = test_camera_index_backend_is_explicit()
    print()
    e = test_default_method_is_the_fast_one()
    print()
    f = test_coast_window_is_frame_rate_independent()
    print()
    ok = a and b and c and d and e and f
    print("判定结果：" + ("全部通过" if ok else "有不达标项"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
