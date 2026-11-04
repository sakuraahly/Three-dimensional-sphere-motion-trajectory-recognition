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
            expected_baseline_mm=124.0,
            working_distance_m=0.30,
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
        checks.append(("expected_baseline_mm=-1 被拒", rejects(lambda: rp.upsert(
            rp.RigProfile(name="neg", expected_baseline_mm=-1.0), path=path))))
        checks.append(("被拒后 store 未被污染（仍只有 1 个档案）",
                       [p.name for p in rp.list_profiles(path)] == ["parallel-desk"]))

        # 4) 几何提示：124/0.3 远小于 90°，600/0.3 ≈ 90°
        small = rp.RigProfile(name="angled-124", stereo_mode="angled",
                              expected_baseline_mm=124.0, working_distance_m=0.30)
        good = rp.RigProfile(name="angled-600", stereo_mode="angled",
                             expected_baseline_mm=600.0, working_distance_m=0.30)
        hint_small = rp.geometry_hint(small)
        hint_good = rp.geometry_hint(good)
        print(f"  B=124 mm / D=0.30 m -> {hint_small}")
        print(f"  B=600 mm / D=0.30 m -> {hint_good}")
        # 2026-10-24：文案从"不支持 90°"（命令式判定）改成"参考估算 / 以标定为准"。
        # 断言改成**语气**而不是字面句子：必须同时出现"标定"与"参考"，
        # 且**不许**再出现"不支持"这类判定语义（估算错的是数字，不是装置）。
        checks.append(("B=124/D=0.3 的提示含「标定」+「参考」（不是命令式判定）",
                       "标定" in hint_small and "参考" in hint_small))
        checks.append(("B=600/D=0.3 的提示同样含「标定」+「参考」",
                       "标定" in hint_good and "参考" in hint_good))
        checks.append(("两种提示都不含「不支持」这类判定语义",
                       "不支持" not in hint_small and "不支持" not in hint_good))
        checks.append(("两种提示都标注「非标定值」（不许把估算当标定）",
                       "非标定值" in hint_small and "非标定值" in hint_good))

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
            if "标定实测" in line or "档案" in line or "参考" in line or "几何提示" in line:
                print(f"  {line}")
        print(f"  退出码 = {proc.returncode}（旧实现在差 >30% 时会是 1）")

        checks.append(("期望 600 mm vs 标定 124 mm（差 384%）precheck 仍返回 0",
                       proc.returncode == 0))
        pos_measured = out.find("标定实测")
        pos_ref = out.find("参考：档案期望基线")
        checks.append(("输出里「标定实测」排在「参考：档案期望基线」之前（权威在前）",
                       pos_measured >= 0 and pos_ref >= 0 and pos_measured < pos_ref))

    # 3) 只给 name + stereo_mode 也能建档（B/D 是可选的）
    with tempfile.TemporaryDirectory() as tmp2:
        p = Path(tmp2) / "profiles.json"
        saved = rp.upsert(rp.RigProfile(name="minimal", stereo_mode="parallel"), path=p)
        back = rp.get_profile("minimal", p)
        print(f"  只给 name+stereo_mode -> upsert 返回 {saved.name!r}，"
              f"B={saved.expected_baseline_mm}、D={saved.working_distance_m}")
        print(f"  geometry_hint -> {rp.geometry_hint(saved)}")
        checks.append(("只填 name+stereo_mode 的档案能 upsert 并读回（B/D 可空）",
                       back is not None and back.name == "minimal"
                       and back.expected_baseline_mm is None))
        checks.append(("缺 B/D 时的几何提示仍含「标定」+「参考」（不装成缺项报错）",
                       "标定" in rp.geometry_hint(saved)
                       and "参考" in rp.geometry_hint(saved)))

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

    这是静态/纯逻辑检查，不开相机、不需要硬件，所以能进无人值守的回归。
    """
    print("----- 分辨率降档梯子：单一来源 + 必须含 1920x1080 -----")
    from sphere_tracker import sources
    from sphere_tracker import video_profiles as vp

    auto = vp.probe_order(None)
    explicit = vp.probe_order(vp.parse("1920x1080"))
    custom = vp.probe_order(vp.parse("1024x768"))
    print(f"  probe_order(None)      = {auto}")
    print(f"  probe_order(1920x1080) = {explicit}")
    print(f"  probe_order(1024x768)  = {custom}")

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
        calib = Path(tmp) / "calib.json"
        calib.write_text(json.dumps(
            {"meta": {"fov": fov.to_dict(),
                      "left_size": [1920, 1080], "right_size": [1920, 1080]}}),
            encoding="utf-8")
        note = _fov_store.record_fov_from_calibration(got, calib, store)
        back = rp.get_profile("fov-rig", store)
        checks.append(("标定 JSON 的 meta.fov 能搬进档案（自动收尾路径）",
                       bool(note) and back is not None
                       and abs(float(back.fov["scale"]) - fov.scale) < 1e-9))
        checks.append(("搬运时补上 source=calibrate:<文件名>",
                       back is not None
                       and str(back.fov.get("source", "")).startswith("calibrate:")))
        # 标定没有 fov（旧流程/降级）时**不许**覆盖档案里的记录
        bare = Path(tmp) / "bare.json"
        bare.write_text(json.dumps({"meta": {"fov": None}}), encoding="utf-8")
        checks.append(("meta.fov=null 时不覆盖已有记录",
                       _fov_store.record_fov_from_calibration(back, bare, store) == ""
                       and rp.get_profile("fov-rig", store).fov is not None))

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
    ok = a and b and c and d and e and f and g and h and i and j
    print("判定结果：" + ("全部通过" if ok else "有不达标项"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
