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


def test_rig_profiles_are_validated_and_persisted() -> bool:
    """装置档案层：空 store 可读、往返一致、非法值被拒、几何提示、active 持久化。

    ## 守的是什么

    多套双目装置需要"这套参数 + 这份标定 + 几何预期"能一起存下来
    （`stereo/profiles.py`）。这一层最容易出的错不是算错，而是：

    - 文件不存在时抛异常 -> 首次使用就崩（所以必须**返回空 store**）；
    - 存进去的 `stereo_mode` 是别名或 `auto` -> 读回来歧义（必须**只收 canonical**）；
    - `calibration_path` 写成 `../x.json` -> 指向仓库外（必须**拒绝逃逸**）；
    - 几何提示把"估算夹角"说成标定值 -> 用户拿估算当真相（必须**标注非标定值**，
      并在 B/D 明显不满足 90° 时说"不支持"）。

    全部用临时目录，不碰真 `config/`（那是运行期产物，且已被 git 忽略）。
    """
    print("----- 装置档案：读写 / 校验 / 几何提示 / active -----")
    import tempfile

    from stereo import profiles as rp

    checks: list[tuple[str, bool]] = []

    def rejects(fn) -> bool:
        try:
            fn()
        except ValueError:
            return True
        return False

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "profiles.json"

        # 1) 文件不存在 -> 空 store，不抛（首次使用要能跑）
        empty = rp.load_store(path)
        checks.append(("文件不存在时返回空 store（active=None、profiles={}）",
                       empty["active"] is None and empty["profiles"] == {}))
        checks.append(("空 store 上 get_active()/list_profiles() 分别为 None/[]",
                       rp.get_active(path) is None and rp.list_profiles(path) == []))

        # 2) 写 -> 读往返一致
        profile = rp.RigProfile(
            name="parallel-desk",
            stereo_mode="parallel",
            width=2560,
            height=1440,
            fps=24,
            calibration_path="config/stereo_calibration.json",
            notes="近平行装置",
        )
        rp.upsert(profile, path=path)
        back = rp.get_profile("parallel-desk", path)
        checks.append(("写->读往返一致（含 calibration_path / notes / created）",
                       back is not None and back.to_dict() == profile.to_dict()))

        # 3) 非法输入必须被拒
        checks.append(("重名被拒", rejects(lambda: rp.upsert(profile, path=path))))
        checks.append(("stereo_mode='auto' 被拒", rejects(lambda: rp.upsert(
            rp.RigProfile(name="auto-rig", stereo_mode="auto"), path=path))))
        checks.append(("stereo_mode='orthogonal'（别名）被拒", rejects(lambda: rp.upsert(
            rp.RigProfile(name="alias-rig", stereo_mode="orthogonal"), path=path))))
        checks.append(("calibration_path='../x.json' 被拒", rejects(lambda: rp.upsert(
            rp.RigProfile(name="escape", calibration_path="../x.json"), path=path))))
        checks.append(("被拒后 store 未被污染（仍只有 1 个档案）",
                       [p.name for p in rp.list_profiles(path)] == ["parallel-desk"]))

        # 4) 2026-11-07：档案不再存几何预期，也不再由它估角度。
        # 这里的判据换成"字段真的没了"，并验一份**旧档案**（还带那两个键）能被读、
        # 旧键被忽略 —— 否则老机器升级后 load_store 会整份炸掉。
        import json

        legacy = Path(tmp) / "legacy.json"
        legacy.write_text(json.dumps({
            "schema": "sphere-rig-profiles", "version": 1, "active": "old-rig",
            "profiles": {"old-rig": {
                "name": "old-rig", "stereo_mode": "angled",
                "expected_baseline_mm": 600.0, "working_distance_m": 0.30,
                "notes": "", "created": "2026-10-24T00:00:00"}}},
            ensure_ascii=False), encoding="utf-8")
        old_back = rp.get_profile("old-rig", legacy)
        checks.append(("RigProfile 不再有 expected_baseline_mm / working_distance_m",
                       not hasattr(rp.RigProfile("x"), "expected_baseline_mm")
                       and not hasattr(rp.RigProfile("x"), "working_distance_m")))
        checks.append(("to_dict 里不再出现几何预期键",
                       not ({"expected_baseline_mm", "working_distance_m"}
                            & set(profile.to_dict()))))
        checks.append(("geometry_hint 已删除（不许再拿猜测值估角度）",
                       not hasattr(rp, "geometry_hint")))
        checks.append(("旧档案带这两个键仍能读、且旧键被忽略",
                       old_back is not None
                       and "expected_baseline_mm" not in old_back.to_dict()))

        # 5) active 持久化（重开程序要能记住用哪套装置）
        rp.set_active("parallel-desk", path)
        reloaded = rp.load_store(path)
        checks.append(("set_active 落盘（重新 load 后 active 仍是它）",
                       reloaded["active"] == "parallel-desk"
                       and rp.get_active(path).name == "parallel-desk"))
        checks.append(("set_active 指向不存在的档案被拒",
                       rejects(lambda: rp.set_active("nope", path))))
        rp.set_active(None, path)
        checks.append(("set_active(None) 可清除", rp.get_active(path) is None))

        # 5b) FOV 记录跟着装置走（2026-11-04）
        #
        # 守的是什么：两路焦距差是**装置属性**（同型号相机出厂焦距就不同），
        # 以前它只活在"某一次标定"的 meta 里 —— 重标就没、估失败就 null，
        # 于是每次标定都重猜一遍 crop_rect，内参跟着漂而屏幕上看不出来。
        # 存进档案后它会跟着 active profile 切换，所以校验必须**复用
        # FovAlignment 自己的规则**（两份校验迟早会漂）。
        fov_ok = {
            "reference_side": "left", "crop_side": "right",
            "crop_rect": [319, 179, 1281, 721], "target_size": [1280, 720],
            "scale": 0.6667, "method": "sift", "inliers": 793, "quality": 0.992,
            "left_size": [1920, 1080], "right_size": [1920, 1080],
            "measured_at": "2026-11-04T08:00:00", "source": "manual:test",
        }
        rp.upsert(rp.RigProfile(name="with-fov", stereo_mode="parallel",
                                fov=fov_ok), path=path)
        got = rp.get_profile("with-fov", path)
        checks.append(("fov 往返一致（含 left_size / measured_at / source）",
                       got is not None and got.fov == fov_ok))
        checks.append(("crop_rect/target_size 落盘是 list（JSON 友好，不是 tuple）",
                       isinstance(got.fov["crop_rect"], list)
                       and isinstance(got.fov["target_size"], list)))
        # 旧档案（没有 fov 这个键）必须照常可读 —— 向前兼容
        rp.upsert(rp.RigProfile(name="no-fov", stereo_mode="parallel"), path=path)
        checks.append(("没有 fov 键的旧档案可读（fov=None）",
                       rp.get_profile("no-fov", path).fov is None))
        # 非法 fov 必须被拒（每条都是真会算错外参的形状）
        for label, bad in (
            ("crop_side 与 reference_side 相同（参考帧不允许被裁）",
             {**fov_ok, "crop_side": "left"}),
            ("crop_rect 含负数", {**fov_ok, "crop_rect": [0, 0, -1, 10]}),
            ("scale 为 0", {**fov_ok, "scale": 0.0}),
            ("quality 超出 [0,1]", {**fov_ok, "quality": 1.5}),
        ):
            checks.append((f"fov：{label} 被拒", rejects(lambda b=bad: rp.upsert(
                rp.RigProfile(name="bad-fov", stereo_mode="parallel", fov=b),
                path=path))))
        checks.append(("fov 不是对象（字符串）被拒", rejects(lambda: rp.upsert(
            rp.RigProfile(name="bad-fov2", stereo_mode="parallel", fov="x"),
            path=path))))
        # 清掉这两条临时档案，否则下面"delete 后列表为空"会被它们打破
        rp.delete("with-fov", path)
        rp.delete("no-fov", path)

        # 6) 删除档案例（含引用计数清零）
        checks.append(("delete 已存在档案返回 True", rp.delete("parallel-desk", path)))
        checks.append(("delete 后列表为空、active 已清",
                       rp.list_profiles(path) == [] and rp.active_name(path) is None))

        # 7) resolve_calibration_base：档案接入主链路的唯一入口（2026-11-04）
        #
        # 守的是什么：半成品档案的典型状态是"路径指向从没标定过的文件"，
        # 无条件跟随会让录制直接报"找不到标定"，而顶层那份真实标定被忽略。
        # 三种情形必须分清：配了且存在 -> 跟随；配了但不存在 -> **显式报错**
        # （2026-11-08 改：以前静默回落顶层，结果切了装置却拿别套装置的几何在跑）；
        # 没配 -> 回落。strict=False 只留给本来就不需要现存标定的调用方，
        # 但仍跟随档案路径（不报错 ≠ 换几何，不许再回落顶层，2026-11-10 改）。
        import os
        fb = Path(tmp) / "top_level"          # 模拟顶层 stereo_calibration
        fb.with_suffix(".npz").write_bytes(b"x")
        # 7a) 档案配了 calib_path 且文件存在 -> 跟随档案（相对 ROOT 解析）
        #     calibration_path 是**仓库内相对路径**，必须相对 rp.ROOT（仓库根）。
        #     用仓库里真实存在的顶层标定路径来验证"跟随"这条，不用造文件：
        #     它指向 config/stereo_calibration，.npz 确实存在。
        rp.upsert(rp.RigProfile(name="rig-a", stereo_mode="parallel",
                                calibration_path="config/stereo_calibration"),
                  path=path)
        rp.set_active("rig-a", path)
        got_a = rp.resolve_calibration_base(path, fallback=fb)
        checks.append(("档案配了且 .npz 存在 -> 跟随档案（相对 ROOT）",
                       got_a == (rp.ROOT / "config" / "stereo_calibration")))
        # 7b) 档案配了但文件不存在 -> 显式报错（半成品档案的关键情形）
        rp.upsert(rp.RigProfile(name="rig-b", stereo_mode="parallel",
                                calibration_path="sub/rig_b"), path=path)
        rp.set_active("rig-b", path)
        try:
            rp.resolve_calibration_base(path, fallback=fb)
            raised = False
        except rp.ProfileError as exc:
            raised = "拒绝回落" in str(exc) and "rig-b" in str(exc)
        checks.append(("档案配了但标定不存在 -> 显式报错，不再拿顶层几何顶替",
                       raised))
        got_b_loose = rp.resolve_calibration_base(path, fallback=fb,
                                                  strict=False)
        checks.append(("strict=False 不报错但必须跟随档案自己的路径（不回落顶层）",
                       got_b_loose == (rp.ROOT / "sub" / "rig_b")
                       and got_b_loose != fb))
        # 7c) 档案没配 calib_path -> 回落 fallback
        rp.upsert(rp.RigProfile(name="rig-c", stereo_mode="parallel"), path=path)
        rp.set_active("rig-c", path)
        got_c = rp.resolve_calibration_base(path, fallback=fb)
        checks.append(("档案没配 calib_path -> 回落 fallback", got_c == fb))
        # 7d) 没激活任何档案 -> 回落 fallback
        rp.set_active(None, path)
        got_d = rp.resolve_calibration_base(path, fallback=fb)
        checks.append(("无激活档案 -> 回落 fallback", got_d == fb))

        # 8) 采集档（宽×高@fps）：能往返，且宽高必须成对（2026-11-05）
        #
        # 守的是什么：档案里的采集档是**拍标定图与录制共用的那一档**。若允许
        # 只填宽，采集会变成「指定宽 × 相机默认高」，而那个尺寸会被记进标定的
        # capture_size，决定之后所有录制请求 —— 错了要到录制失败才发现。
        rp.upsert(rp.RigProfile(name="rig-cap", stereo_mode="parallel",
                                width=1280, height=720, fps=60), path=path)
        got_cap = rp.get_profile("rig-cap", path)
        checks.append(("采集档能往返（1280x720@60）",
                       got_cap is not None and got_cap.width == 1280
                       and got_cap.height == 720 and got_cap.fps == 60))
        try:
            rp.upsert(rp.RigProfile(name="rig-half", width=1280), path=path)
            checks.append(("只给宽被拒（不许存半对的采集档）", False))
        except rp.ProfileError:
            checks.append(("只给宽被拒（不许存半对的采集档）", True))
        checks.append(("被拒后半对档案没落盘",
                       rp.get_profile("rig-half", path) is None))

    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_baseline_in_profile_is_reference_only() -> bool:
    """档案里的期望基线只是**参考**：与标定差 384% 也不许让 precheck 失败。

    ## 守的是什么（2026-10-24，用户原话）

        "我不能精确确定摄像头的位置，所以这个不该设置成强相关固定，
         相反，应该以标定计算得到结果为主。"

    旧实现拿"我估的装置尺寸"去判死"标定算出的结果"：
    `check_stereo_setup.py` 里 ``ratio > 0.30`` 就 ``errors.append(...)`` → 退出码 1。
    一个填错 600 mm 的档案能让一台标定良好的装置过不了前置检查。

    ## 判据

    1. 造一个期望基线 600 mm 的临时档案（真标定实测 ≈124 mm，差 384%），
       跑**真实 CLI** `check_stereo_setup.py --profiles-only`，**退出码必须是 0**；
    2. 输出里"标定实测"必须排在"参考：档案期望基线"**之前**（权威值在前）；
    3. 只填 ``name`` + ``stereo_mode`` 的档案能 upsert 成功（B/D 是可选的）。

    ## 怎么测的（为什么是子进程 + 打补丁）

    子进程能拿到**真正的进程退出码**（直接调函数只能看返回值，测不出 `sys.exit`）。
    档案路径不能走真 `config/profiles.json`（那是用户的运行期产物，测试不许改它），
    所以子进程里先把 ``stereo.profiles.DEFAULT_PATH`` 指到临时 store，
    再调 ``main()`` —— 这是该模块唯一读路径的入口，不打补丁就得改产品代码加环境变量。
    """
    print("----- 档案基线只是参考：precheck 以标定为准 -----")
    import json
    import subprocess
    import tempfile

    from stereo import profiles as rp

    checks: list[tuple[str, bool]] = []
    tool = ROOT / "apps" / "tools" / "check_stereo_setup.py"

    with tempfile.TemporaryDirectory() as tmp:
        store = Path(tmp) / "profiles.json"
        store.write_text(json.dumps({
            "schema": "sphere-rig-profiles", "version": 1, "active": "ref-600",
            # 故意留着两个"旧键"：2026-11-07 起档案不再存它们，检查脚本
            # 必须能读这份旧档案、并且不再拿它写"参考基线"提示。
            "profiles": {"ref-600": {
                "name": "ref-600", "stereo_mode": "angled",
                "expected_baseline_mm": 600.0, "working_distance_m": 0.30,
                "notes": "", "created": "2026-10-24T00:00:00"}}},
            ensure_ascii=False), encoding="utf-8")

        # ⚠️ 也要自带一份"标定实测"（2026-10-27）：这一节的判据是
        # 「标定实测排在参考之前」，没有标定就没有那一行，定位比较会拿 -1 去比 → 假红。
        # 干净 checkout 里 `config/` 是空的（gitignore），实测就是 4 红之一。
        # 造一份只带 `baseline_mm` 的 json：`_calibration_measured` 在缺 .npz 时
        # 会退回读这个字段（角度留空），足够触发"权威在前"的排序。
        fake_calib = Path(tmp) / "stereo_calibration.json"
        fake_calib.write_text(json.dumps({"baseline_mm": 124.0}), encoding="utf-8")

        code = (
            "import sys\n"
            f"sys.path.insert(0, {str(ROOT)!r})\n"
            "from pathlib import Path\n"
            "from stereo import profiles as rp\n"
            f"rp.DEFAULT_PATH = Path({str(store)!r})\n"
            "import importlib.util\n"
            "spec = importlib.util.spec_from_file_location('chk_setup', "
            f"{str(tool)!r})\n"
            "m = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(m)\n"
            f"m.CALIBRATION_PATH = Path({str(fake_calib)!r})\n"
            "sys.argv = ['check_stereo_setup.py', '--profiles-only']\n"
            "raise SystemExit(m.main())\n"
        )
        proc = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                              capture_output=True, text=True, encoding="utf-8",
                              errors="replace")
        out = proc.stdout or ""
        print("  --- 子进程输出（节选）---")
        for line in out.splitlines():
            if "标定实测" in line or "档案" in line or "几何提示" in line:
                print(f"  {line}")
        print(f"  退出码 = {proc.returncode}")

        checks.append(("旧档案里残留的期望基线不影响 precheck（仍返回 0）",
                       proc.returncode == 0))
        pos_measured = out.find("标定实测")
        checks.append(("输出仍以标定实测为准（有那一行）", pos_measured >= 0))
        # 2026-11-07：参考估算那一行整段删除，不能再出现。
        checks.append(("输出不再出现「参考：档案期望基线」/「档案参考值」/「几何提示」",
                       "参考：档案期望基线" not in out
                       and "档案参考值" not in out
                       and "几何提示" not in out))

    # 3) 只给 name + stereo_mode 也能建档（几何量一律由标定给出）
    with tempfile.TemporaryDirectory() as tmp2:
        p = Path(tmp2) / "profiles.json"
        saved = rp.upsert(rp.RigProfile(name="minimal", stereo_mode="parallel"), path=p)
        back = rp.get_profile("minimal", p)
        print(f"  只给 name+stereo_mode -> upsert 返回 {saved.name!r}，"
              f"字段 {sorted(saved.to_dict())}")
        checks.append(("只填 name+stereo_mode 的档案能 upsert 并读回",
                       back is not None and back.name == "minimal"
                       and back.to_dict() == saved.to_dict()))
        checks.append(("最小档案里没有几何预期键，也没有 geometry_hint",
                       not ({"expected_baseline_mm", "working_distance_m"}
                            & set(back.to_dict()))
                       and not hasattr(rp, "geometry_hint")))

    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_resolution_ladder_comes_from_video_profiles() -> bool:
    """分辨率降档梯子必须是**单一来源**（`video_profiles`），且**含 1920x1080**。

    ## 守的是什么（2026-10-31，planbook §2 / §8）

    "分辨率档位"曾在**四处**各写一遍（GUI 的档位表、`sources.py` 的降档梯子、
    `CANDIDATE_MODES`、`LOWRES_RETRY_MODE`）。其中 `sources.py` 那份手写梯子
    ``[(申请),(1280,720),(640,480),(原生)]`` **不含 1920x1080** —— 而 1920x1080
    恰是本机 aoni 在 MSMF 下**唯一稳定可用**的档；梯子里另外两个备选档
    （720p/480p）反而是本机已知打不开的。于是"一帧读不到"时程序跳到已知无效档，
    把"能用的相机"诊断成"坏的"。

    ## 判据

    1. `video_profiles.probe_order(None)` 从最高档到原生档，按像素降序、无重复；
    2. `probe_order(start)` 从该档往下、**不越级往上**，且**含 start 自身**（区间闭）；
    3. 梯子**含 1920x1080**（修复点）；
    4. `sources.CANDIDATE_MODES` 从 `probe_order` 派生（原生档剔除）；
    5. `video_profiles` 仍**不** import cv2（可在无相机、无显示器环境自测）。

    2026-11-18 加 8K/5K 档后补一条：**自动协商（`start=None`）仍只到 4K**，
    5K/8K 只有显式选才出现 —— 自动协商每多试一档就多一次相机打开尝试
    （个别后端单次 20 s+），从 8K 起试会拖慢所有现有相机；同时钉住
    `DEFAULT`/`RECORD_DEFAULT` 不随加档漂移（仍是 2560x1440 的同一对象）。

    这是静态/纯逻辑检查，不开相机、不需要硬件，所以能进无人值守的回归。
    """
    print("----- 分辨率降档梯子：单一来源 + 必须含 1920x1080 -----")
    from sphere_tracker import sources
    from sphere_tracker import video_profiles as vp

    auto = vp.probe_order(None)
    explicit = vp.probe_order(vp.parse("1920x1080"))
    custom = vp.probe_order(vp.parse("1024x768"))
    explicit_8k = vp.probe_order(vp.parse("7680x4320"))
    print(f"  probe_order(None)      = {auto}")
    print(f"  probe_order(1920x1080) = {explicit}")
    print(f"  probe_order(1024x768)  = {custom}")
    print(f"  probe_order(7680x4320) = {explicit_8k}")

    vp_src = (ROOT / "sphere_tracker" / "video_profiles.py").read_text(encoding="utf-8")
    # 只看真正的 import 语句（模块 docstring 里"不 import cv2"那句话不算）
    vp_imports_cv2 = any(
        ln.strip().startswith(("import cv2", "from cv2"))
        for ln in vp_src.splitlines()
    )
    checks: list[tuple[str, bool]] = [
        ("probe_order(None)：最高档 -> 原生档，按像素降序",
         auto == [(3840, 2160), (2560, 1440), (1920, 1080),
                  (1280, 720), (960, 540), (640, 480), vp.NATIVE]),
        ("probe_order(1920x1080)：从该档往下、不越级往上",
         explicit == [(1920, 1080), (1280, 720), (960, 540), (640, 480), vp.NATIVE]),
        ("降档梯子**含 1920x1080**（本机唯一稳定档；旧梯子缺它）",
         (1920, 1080) in auto and (1920, 1080) in explicit),
        ("显式档本身在结果里（区间是闭的）",
         explicit[0] == (1920, 1080) and custom[0] == (1024, 768)),
        ("非标准显式档也含自身，再跟更小的档 + 原生",
         custom == [(1024, 768), (960, 540), (640, 480), vp.NATIVE]),
        ("无重复项、末尾恒为原生档 NATIVE=(0,0)",
         all(len(o) == len(set(o)) and o[-1] == vp.NATIVE
             for o in (auto, explicit, custom))),
        ("sources.CANDIDATE_MODES 从 probe_order 派生（原生档不进模式搜索）",
         sources.CANDIDATE_MODES
         == tuple(p for p in auto if p != vp.NATIVE)
         and vp.NATIVE not in sources.CANDIDATE_MODES),
        ("sources.LOWRES_RETRY_MODE 仍是 (640,480)（补试最低带宽档语义不变）",
         sources.LOWRES_RETRY_MODE == (640, 480)),
        ("自动协商不含 5K/8K（仍是 4K 起 7 项；加档不拖慢现有相机）",
         len(auto) == 7
         and (7680, 4320) not in auto and (5120, 2880) not in auto
         and (3840, 2160) == auto[0]),
        ("显式 8K 以自身开头，并仍往下含 5K/4K/1080p",
         explicit_8k == [(7680, 4320), (5120, 2880), (3840, 2160), (2560, 1440),
                         (1920, 1080), (1280, 720), (960, 540), (640, 480),
                         vp.NATIVE]),
        ("DEFAULT/RECORD_DEFAULT 不随加档漂移（仍是 2560x1440 的同一对象）",
         vp.DEFAULT is vp.RECORD_DEFAULT
         and vp.DEFAULT.label == "2560x1440"),
        ("video_profiles 仍不 import cv2（无相机环境可自测）",
         not vp_imports_cv2),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_fov_measure_roundtrip() -> bool:
    """FOV 测量 → 装置档案 → 读回：合成真值精度 + 端到端搬运。

    ## 守的是什么（2026-11-04）

    用户原话："两个相同摄像机打开后焦距好像不一样，但我又不能调整"。
    这种**固定偏差**以前每次标定都重估一遍，`crop_rect` 每次差几像素，
    内参跟着漂而屏幕上看不出来。改成"量一次、存进装置档案、之后复用"之后，
    这条链上有三个地方会静默错：

    1. **尺度估错**（合成真值能验，合成 1.5× 视场差应得 0.6667）；
    2. **量出来却存不进去**（档案校验比 `FovAlignment` 更严 -> 白量一趟）；
    3. **标定产物搬不进档案**（`meta.fov` 的键名/形状对不上 -> 自动收尾静默失效）。

    用程序生成的纹理图跑，不依赖 data/ 下的任何真实录像。
    """
    print("----- FOV 测量 → 档案 → 读回 -----")
    import json
    import tempfile

    import cv2
    import numpy as np

    from apps import _fov_store
    from stereo import profiles as rp
    from stereo.fov_align import estimate_fov_alignment

    checks: list[tuple[str, bool]] = []

    # 有纹理、可复现（不依赖外部素材）
    rng = np.random.default_rng(7)
    scene = np.full((1080, 1920, 3), 240, np.uint8)
    for x in range(0, 1920, 60):
        cv2.line(scene, (x, 0), (x, 1080), (110, 110, 110), 1)
    for y in range(0, 1080, 60):
        cv2.line(scene, (0, y), (1920, y), (110, 110, 110), 1)
    for _ in range(60):
        cv2.circle(scene,
                   (int(rng.integers(0, 1920)), int(rng.integers(0, 1080))),
                   int(rng.integers(8, 40)),
                   tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
    for k in range(30):
        cv2.putText(scene, f"T{k}", (int(rng.integers(0, 1750)),
                                     int(rng.integers(40, 1040))),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (25, 25, 25), 2)

    factor = 1.5
    ch, cw = int(1080 / factor), int(1920 / factor)
    y0, x0 = (1080 - ch) // 2, (1920 - cw) // 2
    left = cv2.resize(scene[y0:y0 + ch, x0:x0 + cw], (1920, 1080))
    try:
        fov = estimate_fov_alignment(left, scene)
    except ValueError as exc:
        print(f"  [不达标] 合成图估计失败：{exc}")
        return False
    err = abs(fov.scale - 1.0 / factor) / (1.0 / factor) * 100
    print(f"  合成 {factor}× 视场差 -> scale={fov.scale:.4f}"
          f"（期望 {1 / factor:.4f}，误差 {err:.2f}%）")
    checks.append((f"scale 误差 < 2%（实测 {err:.2f}%）", err < 2.0))
    checks.append(("宽的视场那一路被裁（crop_side != reference_side）",
                   fov.crop_side != fov.reference_side))

    payload = dict(fov.to_dict(),
                   left_size=[1920, 1080], right_size=[1920, 1080],
                   measured_at="2026-11-04T08:00:00", source="test")

    with tempfile.TemporaryDirectory() as tmp:
        store = Path(tmp) / "profiles.json"
        prof = rp.RigProfile(name="fov-rig", stereo_mode="parallel", fov=payload)
        rp.upsert(prof, path=store)
        got = rp.get_profile("fov-rig", store)
        checks.append(("估出来的参数能通过档案校验并原样读回",
                       got is not None and got.fov == payload))

        # 标定产物 -> 档案（自动收尾走的就是这条）
        #
        # ⚠️ 格式必须是**顶层 fov**：`StereoRig.load()` 把整个 JSON 顶层塞进
        # `rig.meta`，所以 `record_stereo` 的 `rig.meta.get("fov")` 读的就是
        # 顶层那份。本判据第一版按 `{"meta": {"fov": ...}}` 造，结果把
        # `_fov_store` 读错层级这个真 bug **一起验证成了绿** —— 判据跟着错的
        # 格式走，比没有判据更糟。
        calib = Path(tmp) / "calib.json"
        calib.write_text(json.dumps(
            {"stereo_mode": "parallel", "image_size": [1920, 1080],
             "fov": dict(fov.to_dict(), left_size=[1920, 1080],
                         right_size=[1920, 1080])}),
            encoding="utf-8")
        note = _fov_store.record_fov_from_calibration(got, calib, store)
        back = rp.get_profile("fov-rig", store)
        checks.append(("标定 JSON 顶层 fov 能搬进档案（自动收尾路径）",
                       bool(note) and back is not None
                       and abs(float(back.fov["scale"]) - fov.scale) < 1e-9))
        checks.append(("搬运时补上 source=calibrate:<文件名>",
                       back is not None
                       and str(back.fov.get("source", "")).startswith("calibrate:")))
        # 兼容分层写法（别人手工造的 json）：仍然认
        nested = Path(tmp) / "nested.json"
        nested.write_text(json.dumps({"meta": {"fov": fov.to_dict()}}),
                          encoding="utf-8")
        checks.append(("兼容 meta.fov 的分层写法",
                       _fov_store.fov_payload_from_calibration(nested) is not None))
        # 标定没有 fov（旧流程/降级）时**不许**覆盖档案里的记录
        bare = Path(tmp) / "bare.json"
        bare.write_text(json.dumps({"fov": None}), encoding="utf-8")
        checks.append(("fov=null 时不覆盖已有记录",
                       _fov_store.record_fov_from_calibration(back, bare, store) == ""
                       and rp.get_profile("fov-rig", store).fov is not None))

        # ---- 已裁剪的录像必须被拒绝（否则量到的是"二次裁剪"的假值）----
        #
        # 2026-11-04 实测：平行装置的录像通常是已按 fov.target_size 裁剪过的。
        # 在那种画面上再估一次单应，得到 scale=1.486 / quality=0.988 的
        # "残余视场差"——数字自洽、指标漂亮，套到原始画面上裁剪框会落错位置。
        import importlib.util as _ilu
        root_dir = Path(__file__).resolve().parents[1]
        spec = _ilu.spec_from_file_location(
            "_measure_fov_under_test", root_dir / "apps" / "tools" / "measure_fov.py")
        mf = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mf)
        cropped_dir = Path(tmp) / "cropped_sess"
        cropped_dir.mkdir()
        (cropped_dir / "session.json").write_text(json.dumps(
            {"rig": {"fov": {"target_size": [1718, 972], "scale": 0.665,
                             "crop_side": "right", "crop_rect": [404, 195, 1718, 973]}}}),
            encoding="utf-8")
        checks.append(("已裁剪录像被拒绝（target_size == 视频分辨率）",
                       bool(mf.already_cropped_problem(cropped_dir / "left.mkv",
                                                       (1718, 972)))))
        checks.append(("未裁剪录像不被误拦（分辨率与 target_size 不同）",
                       mf.already_cropped_problem(cropped_dir / "left.mkv",
                                                  (1920, 1080)) == ""))
        plain_dir = Path(tmp) / "plain_sess"
        plain_dir.mkdir()
        (plain_dir / "session.json").write_text(json.dumps({"rig": {"fov": None}}),
                                                encoding="utf-8")
        checks.append(("没有 fov 记录的录像不拦（那是 90° 的正常情况）",
                       mf.already_cropped_problem(plain_dir / "left.mkv",
                                                  (1920, 1080)) == ""))

    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_session_calibration_file_is_dynamic() -> bool:
    """会话标定文件路径只有一份产地，且跟着 session.json 动态走。

    ## 守的是什么（2026-11-10，装置档案修复 A）

    标定文件名是跟着激活装置档案**动态生成**的（录制时按档案的
    ``calibration_path`` 拷进 ``calib/``）：老会话是
    ``calib/stereo_calibration.npz``，新会话是 ``calib/<档案名>.npz``。
    消费方若把文件名写死，新会话就断链。这条判据钉住 5 处收敛：
    非默认名能解析、老式默认名照样解析（向后兼容）、绝对路径原样返回、
    缺 ``calibration.path`` / 缺 ``session.json`` 抛明确 ``SessionError``
    （不是裸 ``KeyError``、不静默返回 ``None``）。
    """
    print("----- 会话标定文件路径：动态解析（唯一产地）-----")
    import json
    import tempfile

    from stereo.sessions import SessionError, session_calibration_file

    checks: list[tuple[str, bool]] = []
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "20261110-120000"
        d.mkdir()

        def write_session(cal) -> None:
            (d / "session.json").write_text(
                json.dumps({"schema": "sphere-stereo-session", "version": 1,
                            "calibration": cal}, ensure_ascii=False),
                encoding="utf-8")

        # 1) 新会话：非默认文件名，必须原样跟随（写死文件名就会断链）
        write_session({"path": "calib/实验性质平行装置1.npz"})
        checks.append(("新会话 calib/<档案名>.npz 动态跟随",
                       session_calibration_file(d)
                       == d / "calib" / "实验性质平行装置1.npz"))
        # 2) 老会话：默认名照样解析（向后兼容，不许破坏）
        write_session({"path": "calib/stereo_calibration.npz"})
        checks.append(("老会话 calib/stereo_calibration.npz 向后兼容",
                       session_calibration_file(d)
                       == d / "calib" / "stereo_calibration.npz"))
        # 3) 绝对路径原样返回
        abs_path = (Path(tmp) / "abs.npz").absolute()
        write_session({"path": str(abs_path)})
        checks.append(("绝对路径原样返回",
                       session_calibration_file(d) == abs_path))
        # 4) 缺 calibration / path -> 明确 SessionError，不是 KeyError / None。
        #    护栏（team-lead 要求）：连**异常类型**和**消息点名字段**都钉住 ——
        #    否则将来有人把 helper 改回静默返回 None / 抛裸 KeyError，夹具会变成
        #    "看似能跑、悄悄退回种子半径"，正是这个仓库最忌讳的形态。
        for bad in (None, {}, {"path": ""}):
            write_session(bad)
            try:
                session_calibration_file(d)
                raised = False
            except SessionError as exc:
                raised = "calibration.path" in str(exc)
            except Exception:               # noqa: BLE001 —— KeyError 之类算不达标
                raised = False
            checks.append((f"缺 calibration.path（{bad!r}）抛 SessionError 且点名字段",
                           raised))
        # 5) 没有 session.json 的目录 -> 明确报错（不硬套）
        (d / "session.json").unlink()
        try:
            session_calibration_file(d)
            raised = False
        except SessionError as exc:
            raised = "session.json" in str(exc)
        except Exception:                   # noqa: BLE001
            raised = False
        checks.append(("没有 session.json 抛 SessionError 且点名字段", raised))

        # 6) 唯一产地：两处解析（要路径就抛错的 / 要路径记 error 不抛的）共用
        #    ``_session_relative``。钉的是**具体调用行**而不是"行为"——
        #    反向自证过：把 helper 里的绝对路径分支删掉，三个测试**全都不红**，
        #    因为 pathlib 的 ``/`` 运算符本来就丢左侧，删了结果不变。所以
        #    "绝对路径原样"这条约定没有可红的行为判据，只能钉"只有一处写它"。
        src = (ROOT / "stereo/sessions.py").read_text(encoding="utf-8")
        pins = (
            'cal_path = _session_relative(p.parent, str(cal["path"]))',
            "cal_path = _session_relative(session_dir, cal.path)",
        )
        checks.append(("session_calibration_file 与 validate_session 共用同一份路径解析",
                       all(code in src for code in pins)))

    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_calibration_follows_active_profile_everywhere() -> bool:
    """全链路按**当前装置档案**判定标定 —— 不许再有第二处写死顶层。

    ## 守的是什么（2026-11-10 用户裁决 B5）

    以前是两套口径：GUI 的 ``AppState._stereo_calib_path()`` 按档案
    ``calibration_path`` 解析，而 CLI / ``run_all`` / 四个工具各自写一份
    ``ROOT / "config" / "stereo_calibration"``。用户把标定写进
    ``config/calib/<档案名>.npz`` 之后：
      - CLI 复检报「找不到标定文件」（标定就在档案路径下）
      - 更坏：读到顶层那份**别套装置**的几何去复检、去判「模式选错」

    判据两层，缺一不可：

    1. **行为**：``apps._common.current_calibration()`` 的四种档案状态。
       四种都要用**子进程**跑：``stereo.profiles.ROOT`` 与 ``DEFAULT_PATH``
       在 import 时就绑定，必须在新进程里同时改 ``rp.ROOT``（``cand`` 靠它拼）
       与 ``rp.DEFAULT_PATH``（档案库路径）才干净。
    2. **源码**：活链路里不许再出现 ``ROOT / "config" / "stereo_calibration"``
       这种**拼接表达式**，也不许 argparse 的 ``default="config/stereo_calibration"``。
       ⚠️ ``stereo/profiles.py`` 例外：那是 fallback 的**唯一产地**
       （``fb = ROOT / "config" / "stereo_calibration"``），不是第二套口径。

    为什么不靠「跑一遍 GUI/CLI 看输出」：``action_verify`` 是交互式
    ``input()``、GUI 要开窗口；判据得能无人值守跑。
    """
    print("----- 标定路径：全链路按当前装置档案（B5）-----")
    import json
    import subprocess
    import tempfile

    checks: list[tuple[str, bool]] = []
    CALIB_REL = "config/calib/p1"

    def _store(active: str, path: str | None) -> dict:
        prof: dict = {"name": active, "stereo_mode": "parallel",
                      "notes": "", "created": "2026-11-10T00:00:00"}
        if path is not None:
            prof["calibration_path"] = path
        return {"schema": "sphere-rig-profiles", "version": 1, "active": active,
                "profiles": {active: prof}}

    def probe(store: dict, files: tuple[str, ...]) -> tuple[str, str]:
        """造一份临时档案库 + 标定文件，在子进程里返回 (基名, 说明)。"""
        with tempfile.TemporaryDirectory(prefix="cw_calib_") as tmp:
            root = Path(tmp)
            for rel in files:
                p = root / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"x")           # 内容无关，只判存在性
            sp = root / "profiles.json"
            sp.write_text(json.dumps(store, ensure_ascii=False), encoding="utf-8")
            code = (
                "import sys\n"
                f"sys.path.insert(0, {str(ROOT)!r})\n"
                "from pathlib import Path\n"
                "from stereo import profiles as rp\n"
                f"rp.ROOT = Path({str(root)!r})\n"
                f"rp.DEFAULT_PATH = Path({str(sp)!r})\n"
                "from apps._common import current_calibration\n"
                "b, w = current_calibration()\n"
                "print(str(b))\n"
                "print(w)\n"
            )
            proc = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                                  capture_output=True, text=True, encoding="utf-8",
                                  errors="replace")
            lines = (proc.stdout or "").splitlines()
            if len(lines) < 2:
                return "", f"<子进程失败 rc={proc.returncode}> {(proc.stderr or '')[:300]}"
            return lines[0], lines[1]

    # ① 档案配了路径 + 文件在 → 用档案路径，无说明
    b1, w1 = probe(_store("p1", CALIB_REL), (f"{CALIB_REL}.npz",))
    checks.append(("档案配了标定且文件在 → 返回档案路径、无告警",
                   (b1.endswith(CALIB_REL.replace("/", "\\"))
                    or b1.endswith(CALIB_REL)) and w1 == ""))
    # ② 档案配了路径 + 文件不在 → **仍返回档案路径**，绝不回落顶层
    b2, w2 = probe(_store("p1", CALIB_REL), ())
    checks.append(("档案配了标定但文件不在 → 不回落顶层（返回档案路径）",
                   (b2.endswith(CALIB_REL.replace("/", "\\"))
                    or b2.endswith(CALIB_REL))
                   and "指定的标定文件不存在" in w2))
    # ③ 档案没配路径 + 顶层在 → 回落顶层，无说明
    b3, w3 = probe(_store("p2", None), ("config/stereo_calibration.npz",))
    checks.append(("档案没配标定 → 回落顶层（这是合法回落，不是第二套口径）",
                   b3.endswith("stereo_calibration") and w3 == ""))
    # ④ 档案没配 + 顶层也没有 → 顶层路径 + 明确说明
    b4, w4 = probe(_store("p2", None), ())
    checks.append(("档案没配且顶层也没有 → 明确说「找不到标定文件」",
                   b4.endswith("stereo_calibration")
                   and "找不到标定文件" in w4))

    # ⑤ 源码：活链路不许再有第二份顶层标定硬编码
    #    ⚠ `stereo/profiles.py` 不在扫描列表里 —— 它是 fallback 的**唯一产地**。
    hardcodes = ('ROOT / "config" / "stereo_calibration"',
                 "ROOT / 'config' / 'stereo_calibration'",
                 'default="config/stereo_calibration"',
                 "default='config/stereo_calibration'")
    scan = ("apps/gui_main.py", "apps/cli_main.py", "apps/run_all.py",
            "apps/_gui_app.py", "apps/_gui_stereo.py", "apps/verify_stereo.py",
            "apps/tools/check_stereo_setup.py", "apps/tools/set_world_frame.py",
            "apps/tools/export_calibration.py")
    offenders: list[str] = []
    for rel in scan:
        text = (ROOT / rel).read_text(encoding="utf-8")
        offenders += [f"{rel}: {lit}" for lit in hardcodes if lit in text]
    checks.append(("九个活链路文件里不再有第二份顶层标定硬编码", not offenders))
    if offenders:
        print(f"    {offenders}")

    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_capture_size_never_falls_back_to_2560() -> bool:
    """采集档来源已解耦到装置档案，未标定 / 档案没填也不许落 2560x1440。

    ## 守的是什么（2026-11-10，装置档案修复 A 的 blocking 配套）

    采集档是**装置档案的属性**（``RigProfile.width/height``），不是标定 json 里
    ``capture_size`` 的属性。改 ``return cand`` 后，未标定档案的标定 json 不存在
    → ``calib_capture_size`` 返回 ``None`` → 旧写法会把 doctor / GUI 拍图回落到
    ``video_profiles.DEFAULT``（2560x1440），而双路同时开 2560 会让一路 MSMF
    会话**持久失效、必须物理拔插**（``apps/tools/cameras.py`` doctor 段与
    ``apps/_gui_stereo.py::_capture_chain_frames`` 的注释都记着）。

    这条判据钉住：helper 与真实消费方 ``cameras.py`` 在「档案 width/height 为
    null + 未标定」的真实情形下都走 1080 安全档、不是 2560，且文案不谎称
    「标定记录的采集档 / 与录制同档 / 相机最高档」。安全档取
    ``video_profiles`` 的 1080 档对象，不写第二份 1920/1080 字面量。
    """
    print("----- 采集档：装置档案优先 + 1080 安全档（不许落 2560）-----")
    import argparse

    from apps._common import capture_size_from_profile
    from apps.tools.cameras import build_parser
    from sphere_tracker import video_profiles
    from stereo.profiles import get_active

    checks: list[tuple[str, bool]] = []
    safe = video_profiles.parse("1920x1080")
    default = video_profiles.DEFAULT
    dishonest = ("标定记录", "与录制同档", "相机最高档")

    # 1) helper：档案没填宽高（真实激活档案就是 null）→ 1080 安全档
    class _EmptyProfile:
        width = None
        height = None
        fps = None
    w, h, note = capture_size_from_profile(_EmptyProfile())
    checks.append(("档案没填宽高 -> 1080 安全档、不是 2560",
                   (w, h) == (safe.width, safe.height)
                   and (w, h) != (default.width, default.height)))
    checks.append(("安全档文案不谎称标定记录 / 与录制同档 / 相机最高档",
                   all(s not in note for s in dishonest)))
    # 2) helper：档案填了宽高 -> 用档案值（装置档案优先，既有设计意图）
    class _FilledProfile:
        width = 1280
        height = 720
        fps = 30
    w2, h2, _note2 = capture_size_from_profile(_FilledProfile())
    checks.append(("档案填了宽高 -> 用档案值（装置档案优先）",
                   (w2, h2) == (1280, 720)))
    # 3) 真实消费方：cameras.py doctor 的默认宽高与 --help 文案。
    #    期望值按「真实激活档案 + 同一 helper」算，不把 1080 写死到判据里：
    #    档案以后填了宽高，doctor 就该跟着变，但**永远不许变成 2560**。
    expect_w, expect_h, _ = capture_size_from_profile(get_active())
    p = build_parser()
    sub = next(a for a in p._actions
               if isinstance(a, argparse._SubParsersAction))
    doctor = sub.choices["doctor"]
    dw = next(a for a in doctor._actions if a.dest == "width")
    dh = next(a for a in doctor._actions if a.dest == "height")
    checks.append(("doctor 默认档 = 装置档案/安全档解析结果（真实 config）",
                   (dw.default, dh.default) == (expect_w, expect_h)))
    checks.append(("doctor 默认档不是 2560x1440（不许静默回落最高档）",
                   (dw.default, dh.default) != (default.width, default.height)))
    help_text = (dw.help or "") + (dh.help or "")
    checks.append(("doctor --width/--height 文案不谎称标定记录 / 与录制同档 / 相机最高档",
                   all(s not in help_text for s in dishonest)))

    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_profile_calibration_status_four_states() -> bool:
    """B1：profile_calibration_status 四态 + 三文件存在性唯一判定 + 收敛。

    全部用临时目录 + monkeypatch ``profiles.ROOT``，不碰真实 ``config/calib``
    任何文件。四态判据、三文件任一存在、老文件（无 profile）不误报，以及
    ``check_stereo_setup.py`` 不再对「配了 calibration_path 且产物存在」误报。
    """
    print("----- B1：装置档案标定状态四态 / 三文件判定 -----")
    import json
    import subprocess
    import tempfile

    from stereo import profiles as rp

    checks: list[tuple[str, bool]] = []
    orig_root = rp.ROOT
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        calib = root / "calib"
        calib.mkdir()
        rp.ROOT = root                       # 纯函数相对临时根解析，finally 还原
        try:
            def write_json(name, meta):
                (calib / f"{name}.json").write_text(
                    json.dumps(meta, ensure_ascii=False), encoding="utf-8")

            # uncalibrated：路径合法、三者都不存在（实测要求 1 的形状）
            s = rp.profile_calibration_status(rp.RigProfile(
                name="A", stereo_mode="parallel", calibration_path="calib/A"))
            checks.append(("文件都不存在 -> uncalibrated（不是 path_mismatch）",
                           s.state == rp.CALIB_STATUS_UNCALIBRATED))

            # calibrated：.json 存在、模式匹配、老文件无 profile
            write_json("B", {"stereo_mode": "parallel"})
            s = rp.profile_calibration_status(rp.RigProfile(
                name="B", stereo_mode="parallel", calibration_path="calib/B"))
            checks.append((".json + 模式匹配 + 无 profile（老文件）-> calibrated",
                           s.state == rp.CALIB_STATUS_CALIBRATED))

            # mode_conflict
            write_json("C", {"stereo_mode": "angled"})
            s = rp.profile_calibration_status(rp.RigProfile(
                name="C", stereo_mode="parallel", calibration_path="calib/C"))
            checks.append(("标定 stereo_mode 与档案不一致 -> mode_conflict",
                           s.state == rp.CALIB_STATUS_MODE_CONFLICT))

            # path_mismatch：meta.profile 指向别处 / 坏 JSON / 非法路径
            write_json("D", {"stereo_mode": "parallel", "profile": "别的装置"})
            s = rp.profile_calibration_status(rp.RigProfile(
                name="D", stereo_mode="parallel", calibration_path="calib/D"))
            checks.append(("meta.profile 指向别套档案 -> path_mismatch",
                           s.state == rp.CALIB_STATUS_PATH_MISMATCH))
            (calib / "E.json").write_text("{not json", encoding="utf-8")
            s = rp.profile_calibration_status(rp.RigProfile(
                name="E", stereo_mode="parallel", calibration_path="calib/E"))
            checks.append(("标定 .json 损坏 -> path_mismatch",
                           s.state == rp.CALIB_STATUS_PATH_MISMATCH))
            s = rp.profile_calibration_status(rp.RigProfile(
                name="F", stereo_mode="parallel", calibration_path="../escape"))
            checks.append(("calibration_path 含 ..（越界）-> path_mismatch",
                           s.state == rp.CALIB_STATUS_PATH_MISMATCH))

            # 三文件存在性唯一判定：.npz / .json / 无扩展名基名 任一
            (calib / "G.npz").write_bytes(b"x")
            s = rp.profile_calibration_status(rp.RigProfile(
                name="G", stereo_mode="parallel", calibration_path="calib/G"))
            checks.append(("只有 .npz 也算存在（不误报未标定）",
                           rp.calibration_files_exist(calib / "G")
                           and s.state == rp.CALIB_STATUS_CALIBRATED))
            (calib / "H").write_bytes(b"x")
            checks.append(("无扩展名基名存在也算存在",
                           rp.calibration_files_exist(calib / "H")))
            checks.append(("三者都不存在时判 False",
                           not rp.calibration_files_exist(calib / "NOPE")))

            # C1 新文件（meta.profile == 档案名）不能误报
            write_json("I", {"stereo_mode": "parallel", "profile": "I"})
            s = rp.profile_calibration_status(rp.RigProfile(
                name="I", stereo_mode="parallel", calibration_path="calib/I"))
            checks.append(("meta.profile == 当前档案名 -> calibrated（不误报）",
                           s.state == rp.CALIB_STATUS_CALIBRATED))
            # 存在性是**唯一开关**：写出来就是 calibrated，删掉立刻退回 uncalibrated。
            # 这一条是"文件不存在 -> uncalibrated"的真正守卫（旧判据把"用户当次
            # 有没有标定"当成期望值，用户一标定它就红 —— 那是判据依赖了用户数据）。
            write_json("J", {"stereo_mode": "parallel"})
            s_have = rp.profile_calibration_status(rp.RigProfile(
                name="J", stereo_mode="parallel", calibration_path="calib/J"))
            (calib / "J.json").unlink()
            s_gone = rp.profile_calibration_status(rp.RigProfile(
                name="J", stereo_mode="parallel", calibration_path="calib/J"))
            checks.append(("档案配了 calibration_path：文件在 -> calibrated，"
                           "删掉文件 -> 立刻退回 uncalibrated（存在性是唯一开关）",
                           s_have.state == rp.CALIB_STATUS_CALIBRATED
                           and s_gone.state == rp.CALIB_STATUS_UNCALIBRATED))
        finally:
            rp.ROOT = orig_root

    # 真实激活档案（只读 config/profiles.json，不改）：**只断言结构不变量**。
    #
    # 旧判据写死 `state == uncalibrated`，等于把"用户 2026-11-11 真机还没标定"
    # 当成期望值；用户标定过之后标定文件就存在，`state` 变 calibrated，判据必然红。
    # 测试不该依赖用户当前录了什么 —— 要守的是**状态与磁盘上的文件是否对得上**：
    #   * 状态必须落在四态全集里（不出野值）；
    #   * `uncalibrated` **当且仅当** 标定三文件确实不存在（当场用 Path 判断，
    #     不写死任何一边）。这样"用户标没标定"都绿，而"状态与文件不符"这种真 bug
    #     仍然会被抓住（两个方向都查）。
    a = rp.get_active()
    print(f"  真实激活档案 = {a.name if a else None!r}")
    if a is None:
        checks.append(("真实环境没有激活档案（无可断言，跳过而不判失败）", True))
    else:
        s = rp.profile_calibration_status(a)
        try:
            base = rp._calibration_base_of(a)     # 与 profile_calibration_status 同源
        except rp.ProfileError:
            base = None
        files_exist = rp.calibration_files_exist(base) if base is not None else None
        print(f"      标定基名 = {base}")
        print(f"      -> state={s.state!r}  reason={s.reason!r}"
              f"  磁盘三文件存在={files_exist}（当场判断，不写死）")
        checks.append(("真实激活档案的标定状态落在四态全集里（没有野值）",
                       s.state in {rp.CALIB_STATUS_CALIBRATED,
                                   rp.CALIB_STATUS_UNCALIBRATED,
                                   rp.CALIB_STATUS_PATH_MISMATCH,
                                   rp.CALIB_STATUS_MODE_CONFLICT}))
        if base is None:      # 路径非法：此时不该谈"文件在不在"
            checks.append(("标定路径非法 <=> path_mismatch",
                           s.state == rp.CALIB_STATUS_PATH_MISMATCH))
        else:
            checks.append(("标定三文件不存在 <=> uncalibrated（状态必须与磁盘一致，两个方向都查）",
                           (s.state == rp.CALIB_STATUS_UNCALIBRATED) == (not files_exist)))

    # check_stereo_setup.py：配了 calibration_path 且产物存在时不误报。
    # 用临时 store 打补丁 DEFAULT_PATH，不改真实 config/profiles.json；
    # REPO_ROOT 指向真实仓库，config/calib/平行装置.json/.npz 确实存在。
    tool = ROOT / "apps" / "tools" / "check_stereo_setup.py"
    with tempfile.TemporaryDirectory() as tmp2:
        store = Path(tmp2) / "profiles.json"
        store.write_text(json.dumps({
            "schema": "sphere-rig-profiles", "version": 1, "active": "exists-rig",
            "profiles": {"exists-rig": {
                "name": "exists-rig", "stereo_mode": "parallel",
                "calibration_path": "config/calib/平行装置"}}},
            ensure_ascii=False), encoding="utf-8")
        code = (
            "import sys\n"
            f"sys.path.insert(0, {str(ROOT)!r})\n"
            "from pathlib import Path\n"
            "from stereo import profiles as rp\n"
            f"rp.DEFAULT_PATH = Path({str(store)!r})\n"
            "import importlib.util\n"
            f"spec = importlib.util.spec_from_file_location('chk', {str(tool)!r})\n"
            "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
            "sys.argv = ['check_stereo_setup.py', '--profiles-only']\n"
            "raise SystemExit(m.main())\n"
        )
        proc = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                              capture_output=True, text=True, encoding="utf-8",
                              errors="replace")
        out = proc.stdout or ""
    checks.append(("check_stereo_setup 对产物存在的档案不再误报「标定文件不存在」",
                   "档案指向的标定文件不存在" not in out and proc.returncode == 0))

    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_calibration_identity_three_states() -> bool:
    """C2：meta.profile 三态（向后兼容老文件）+ GUI 侧分流 + 不阻断。"""
    print("----- C2：标定身份三态（向后兼容老文件）-----")
    import types

    from stereo import profiles as rp
    from apps._gui_app import AppState

    checks: list[tuple[str, bool]] = []

    matched = rp.calibration_identity({"profile": "甲"}, "甲")
    mismatch = rp.calibration_identity({"profile": "甲"}, "乙")
    unreg = rp.calibration_identity({}, "甲")
    legacy = rp.calibration_identity({"stereo_mode": "parallel"}, "甲")
    checks.append(("有 profile 且 == 当前档案 -> matched",
                   matched.state == rp.IDENTITY_MATCHED and matched.message == ""))
    checks.append(("有 profile 且 != 当前档案 -> mismatch（点名两个档案）",
                   mismatch.state == rp.IDENTITY_MISMATCH
                   and "甲" in mismatch.message and "乙" in mismatch.message))
    checks.append(("meta 没有 profile（老文件）-> unregistered",
                   unreg.state == rp.IDENTITY_UNREGISTERED
                   and legacy.state == rp.IDENTITY_UNREGISTERED))
    checks.append(("unregistered 不误报成「属于别套装置」",
                   "属于装置档案「" not in unreg.message))

    def make(meta):
        s = AppState.__new__(AppState)
        s.profile = types.SimpleNamespace(name="乙", stereo_mode="parallel")
        s.rig = types.SimpleNamespace(meta=meta)
        s.mono = None
        s.notes = []
        s.infos = []
        return s

    s_match = make({"profile": "乙", "stereo_mode": "parallel"})
    s_mis = make({"profile": "甲", "stereo_mode": "parallel"})
    s_unreg = make({"stereo_mode": "parallel"})
    for s in (s_match, s_mis, s_unreg):
        s._collect_identity()
    checks.append(("matched：notes/infos 都不加、level=info",
                   not s_match.notes and not s_match.infos
                   and s_match.status_text()[1] == "info"))
    checks.append(("mismatch：进 notes(warn)、mode_allowed 仍 True（不阻断）",
                   len(s_mis.notes) == 1 and not s_mis.infos
                   and s_mis.status_text()[1] == "warn"
                   and s_mis.mode_allowed("parallel") is True))
    checks.append(("unregistered：只进 infos 说明、level=info、不阻断",
                   not s_unreg.notes and len(s_unreg.infos) == 1
                   and s_unreg.status_text()[1] == "info"
                   and s_unreg.mode_allowed("parallel") is True))

    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_reload_reuses_refresh_and_broadcasts() -> bool:
    """B3/B4：_reload_state 不写死常量、状态条随档案变、广播到已开窗口。

    静态判据（检查 ``_reload_state`` 源码）用于反向自证：把 :833-834 改回
    写死常量，下面第 3 / 5 条会红，还原后复绿。
    """
    print("----- B3/B4：重新读取复用刷新结果 + 广播已开窗口 -----")
    import ast
    import types

    from apps._gui_app import App, AppState

    checks: list[tuple[str, bool]] = []
    src = (ROOT / "apps" / "_gui_app.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "_reload_state"), None)
    seg = ast.get_source_segment(src, fn) if fn is not None else ""

    def _set_status_first_is_literal_cn(fnode) -> bool:
        if fnode is None:
            return False
        for n in ast.walk(fnode):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "set_status" and n.args
                    and isinstance(n.args[0], ast.Constant)
                    and isinstance(n.args[0].value, str)
                    and any("\u4e00" <= ch <= "\u9fff" for ch in n.args[0].value)):
                return True
        return False

    def _calls(fnode, attr: str) -> bool:
        if fnode is None:
            return False
        return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr == attr for n in ast.walk(fnode))

    def _has_literal(fnode, literal: str) -> bool:
        if fnode is None:
            return False
        return any(isinstance(n, ast.Constant) and n.value == literal
                   for n in ast.walk(fnode))

    # 用 AST 而非源码 substring：注释里提到那句写死常量不应算「还在用」。
    # 反向自证改回 `ok, msg = True, "已重新读取装置与标定状态"` 时，它是真的
    # ast.Constant，_has_literal 会命中 -> 下面第 3 条判据真的红。
    checks.append(("apps/_gui_app.py 能找到 _reload_state", fn is not None))
    checks.append(("_reload_state 调用 self._refresh()（复用刷新结果）",
                   _calls(fn, "_refresh")))
    checks.append(("_reload_state 不再写死「已重新读取装置与标定状态」常量",
                   not _has_literal(fn, "已重新读取装置与标定状态")))
    checks.append(("_reload_state 的 set_status 参数不是写死中文常量",
                   not _set_status_first_is_literal_cn(fn)))
    checks.append(("_reload_state 调用 broadcast_state_refresh()（B4）",
                   _calls(fn, "broadcast_state_refresh")))

    def make_state(name, meta):
        s = AppState.__new__(AppState)
        s.profile = types.SimpleNamespace(name=name, stereo_mode="parallel")
        s.rig = types.SimpleNamespace(meta=meta)
        s.mono = None
        s.notes = []
        s.infos = []
        return s

    t1 = make_state("甲", {"stereo_mode": "parallel"})
    t2 = make_state("乙", {"stereo_mode": "angled"})
    checks.append(("切换档案后 status_text() 文案不同且各含档案名",
                   t1.status_text()[0] != t2.status_text()[0]
                   and "甲" in t1.status_text()[0] and "乙" in t2.status_text()[0]))
    t3 = make_state("甲", {})
    t3.rig = None
    checks.append(("有档案却没标定 -> level=warn（ok 反映真实状态）",
                   t3.status_text()[1] == "warn"))

    class W:
        def __init__(self, exists=True):
            self.calls = 0
            self.exists = exists

        def winfo_exists(self):
            return self.exists

        def _refresh_state(self):
            self.calls += 1

    w_mono, w_st, w_dead = W(), W(), W(exists=False)
    no_refresh = types.SimpleNamespace(winfo_exists=lambda: True)
    ns = types.SimpleNamespace(_windows={
        "mono": w_mono, "stereo_parallel": w_st,
        "stereo_angled": w_dead, "console": no_refresh})
    App.broadcast_state_refresh(ns)
    checks.append(("B4：mono/stereo 各重绘一次，已关闭 / 无 _refresh_state 的窗口不重绘",
                   w_mono.calls == 1 and w_st.calls == 1 and w_dead.calls == 0))

    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_calibration_specs_display() -> bool:
    """B-189：`calibration_specs()` 是标定数值展示的唯一产地，缺字段不许编造。

    用户原话「装置档案要能看到标定后的具体信息」。这条判据钉的不是"能不能显示"，
    而是**显示的数必须真的来自那份标定的 meta**：

    - 全字段逐项对得上（防止字段名写错、嵌套取错层）；
    - 缺字段一律 ``None`` / ``""``，**不许**补 0.0、补 0.00 px、补 1920；
    - ``bool`` 不算数字（``True`` 当 1.0 是撒谎）；
    - 只有 ``.npz`` 没有 ``.json`` 仍是"已标定（无详情）"，不是"未标定"；
    - ``include_path=False`` 时别把绝对路径又贴一遍。

    全部在临时目录里 monkeypatch ``profiles.ROOT``，不读也不写真实
    ``config/profiles.json``（用户正在用 GUI 真机测试，``active`` 不能被动）。
    """
    print("----- B-189：装置档案显示标定实测（唯一产地 / 缺字段不编造）-----")
    import json
    import shutil
    import tempfile

    from stereo import profiles as rp

    checks: list[tuple[str, bool]] = []
    orig_root = rp.ROOT
    tmp = Path(tempfile.mkdtemp())
    try:
        calib = tmp / "calib"
        calib.mkdir()
        rp.ROOT = tmp                       # 相对路径按临时根解析，finally 还原

        def write_json(name: str, meta: dict) -> None:
            (calib / f"{name}.json").write_text(
                json.dumps(meta, ensure_ascii=False), encoding="utf-8")

        def make(path: str) -> rp.RigProfile:
            # calibration_path 只接受仓库内相对路径，故不能指到 tmp 的绝对路径
            return rp.RigProfile(name="测试装置", stereo_mode="parallel",
                                 calibration_path=path)

        # 1) 全字段解析：meta 里有什么就搬什么，一个不漏
        write_json("full", {
            "baseline_mm": 68.84, "angle_deg": 0.653,
            "created": "2026-11-08 15:01:40", "capture_size": [1920, 1080],
            "profile": "测试装置", "board": "棋盘格",
            "quality": {"rms": 0.688, "epipolar_rms": 0.713,
                        "epipolar_max": 1.326, "n_pairs_used": 33,
                        "n_pairs_in": 40, "grade": "勉强可用",
                        "depth_sigma_mm_at_0_6m": 5.06},
        })
        s = rp.calibration_specs(make("calib/full"))
        checks.append((
            "全字段解析：state/path/基线/夹角/三项 RMS/配对数/判定/时间/采集档/登记/标定板",
            s.state == rp.CALIB_STATUS_CALIBRATED
            and s.path == calib / "full"
            and s.baseline_mm == 68.84
            and s.angle_deg == 0.653
            and s.rms_px == 0.688
            and s.epipolar_rms_px == 0.713
            and s.epipolar_max_px == 1.326
            and s.n_pairs_used == 33
            and s.n_pairs_in == 40
            and s.grade == "勉强可用"
            and s.created == "2026-11-08 15:01:40"
            and s.capture_size == (1920, 1080)
            and s.depth_sigma_mm_0_6m == 5.06
            and s.bound_profile == "测试装置"
            and s.board == "棋盘格"))
        checks.append(("全字段：inline 只放基线/夹角/极线三个数（1 位 / 1 位 / 2 位）",
                       rp.format_specs_inline(s)
                       == "68.8 mm / 0.7° / 0.71 px"))

        # 2) 缺字段不编造：只有基线，其余必须是 None / ""
        write_json("sparse", {"baseline_mm": 70.0})
        s = rp.calibration_specs(make("calib/sparse"))
        checks.append((
            "缺字段不编造：数值字段全 None、字符串字段全 \"\"、capture_size None",
            s.baseline_mm == 70.0
            and s.angle_deg is None and s.rms_px is None
            and s.epipolar_rms_px is None and s.epipolar_max_px is None
            and s.n_pairs_used is None and s.n_pairs_in is None
            and s.depth_sigma_mm_0_6m is None and s.capture_size is None
            and s.grade == "" and s.created == ""
            and s.bound_profile == "" and s.board == ""))
        checks.append(("缺字段：inline 只有一段「70.0 mm」（不补 0.00 px / 不补 0°）",
                       rp.format_specs_inline(s) == "70.0 mm"))
        tip = rp.format_specs_tooltip(s)
        # 不查裸 "0.00"：基线自己就是 70.00 mm；要查的是**凭空补出来的**段
        checks.append(("缺字段：tooltip 不出现 夹角 / 极线 / 未知 / 补充的 0.00 px / 1920",
                       all(k not in tip for k in
                           ("夹角", "极线", "未知", "0.00 px", "0.000°", "1920"))))
        # 连基线都没有的空 meta：最容易被人手补成 0.0 的那种，必须还是 None
        write_json("nobase", {})
        s0 = rp.calibration_specs(make("calib/nobase"))
        checks.append(("缺字段：meta 连基线都没有 -> baseline_mm 仍是 None（不补 0.0）",
                       s0.baseline_mm is None and s0.angle_deg is None
                       and rp.format_specs_inline(s0) == "已标定（无详情）"))

        # 3) bool 不算数字：True 当 1.0 是撒谎
        write_json("boolish", {"baseline_mm": True, "angle_deg": 1.0})
        s = rp.calibration_specs(make("calib/boolish"))
        checks.append(("bool 不算数字：baseline_mm=True -> None，angle_deg=1.0 保留",
                       s.baseline_mm is None and s.angle_deg == 1.0))

        # 4) 只有 .npz 没有 .json：仍算已标定，只是不知道参数
        (calib / "onlynpz.npz").write_bytes(b"x")
        s = rp.calibration_specs(make("calib/onlynpz"))
        tip = rp.format_specs_tooltip(s)
        checks.append(("只有 .npz 没有 .json -> state 仍是 calibrated（不误报未标定）",
                       s.state == rp.CALIB_STATUS_CALIBRATED))
        checks.append(("只有 .npz：inline 说「已标定（无详情）」，不是「—」",
                       rp.format_specs_inline(s) == "已标定（无详情）"))
        checks.append(("只有 .npz：tooltip 不写「基线」占位，但明说未登记来源",
                       "基线" not in tip and "登记装置：（未登记来源" in tip))

        # 5) 未标定：连标定文件都没有
        s = rp.calibration_specs(make("calib/nothing"))
        checks.append(("未标定 -> inline「—」、tooltip 空串（不显示任何数）",
                       s.state == rp.CALIB_STATUS_UNCALIBRATED
                       and rp.format_specs_inline(s) == "—"
                       and rp.format_specs_tooltip(s) == ""))

        # 6) include_path：调用方已经显示了路径就别再贴一遍
        s = rp.calibration_specs(make("calib/full"))
        with_path = rp.format_specs_tooltip(s)
        no_path = rp.format_specs_tooltip(s, include_path=False)
        checks.append((
            "include_path=False 末行不是「文件：」；默认参数末行是它",
            with_path.splitlines()[-1].startswith("文件：")
            and "文件：" not in no_path
            and not no_path.splitlines()[-1].startswith("文件：")))

        # 7) 唯一产地：GUI 只许调 profiles 的格式化函数，不许自己解析 meta
        src = (ROOT / "apps" / "profile_dialog.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        own_loads = any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "loads"
            and isinstance(n.func.value, ast.Name) and n.func.value.id == "json"
            for n in ast.walk(tree))
        calls = {n.func.attr for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        checks.append(("apps/profile_dialog.py 不自己 json.loads 标定 meta",
                       not own_loads))
        checks.append(("apps/profile_dialog.py 调用了 calibration_specs / format_specs_inline",
                       {"calibration_specs", "format_specs_inline"} <= calls))
    finally:
        rp.ROOT = orig_root
        shutil.rmtree(tmp, ignore_errors=True)

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
    g = test_rig_profiles_are_validated_and_persisted()
    print()
    h = test_baseline_in_profile_is_reference_only()
    print()
    i = test_resolution_ladder_comes_from_video_profiles()
    print()
    j = test_fov_measure_roundtrip()
    print()
    k = test_session_calibration_file_is_dynamic()
    print()
    l = test_capture_size_never_falls_back_to_2560()
    print()
    m = test_profile_calibration_status_four_states()
    print()
    n = test_calibration_identity_three_states()
    print()
    o = test_reload_reuses_refresh_and_broadcasts()
    print()
    p = test_calibration_follows_active_profile_everywhere()
    print()
    q = test_calibration_specs_display()
    print()
    ok = a and b and c and d and e and f and g and h and i and j and k and l \
        and m and n and o and p and q
    print("判定结果：" + ("全部通过" if ok else "有不达标项"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
