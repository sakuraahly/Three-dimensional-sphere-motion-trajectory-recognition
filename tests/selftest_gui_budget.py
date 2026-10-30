"""GUI 预算自测：防止分窗重做之后又长回"一个窗口装全部"。

为什么它是最重要的一个测试（2026-10-27 计划书 §5.1）
----------------------------------------------------
旧界面的问题不是"某天突然变复杂"，而是**每次都只加不删**：四天里页面自然高度
从 2131 px 涨到 2828 px（+33%），第 1 步一张卡 1154 px。没有脚本守着的结构，
下一次加功能就会照原样再长回去 —— 所以预算必须**可执行**，而不是写在文档里。

判据（全部可脚本化，不靠"看起来好"）
------------------------------------
1. 每个模式窗的**每一步**（L1，不含折叠区）：输入控件 ≤ 8、按钮 ≤ 5；
2. 同一步**展开高级后**（L2）：可点控件 ≤ 12；
3. 任何窗口里都**没有**"模式选择"控件（下拉值出现 auto/parallel/angled 即判红）
   —— 模式 = 你进的哪个窗口，不能第二次出现在控件里；
4. 每一步**恰好一个**主操作（``Go.TButton``）：一步两个等于没有主操作；
5. 界面文案里不出现 Markdown 标记 ``**``（B-96 的教训：它会被原样印在按钮上）。
6. **新窗口的命令行拼装**（2026-10-29 增）：板参数 / 方法分支 / ``--skip-fov``
   的归属 / ``--backend`` 只出现一次 / 录制不许传 ``--stereo-mode`` /
   高级项留空不传 —— 这一段**不依赖 Tk**，无显示器也照跑。
   为什么它属于这个测试：结构判据只数控件，**一个字都不管参数怎么拼**，
   而"界面选的板参数有没有真的进到命令行"正是这两个窗口唯一干的活
   （老窗口里同一件事有 7 节判据守着，见 `selftest_launcher_params.py`）。

无显示器环境：``tk.Tk()`` 会抛 ``TclError``，此时打印 ``[跳过]`` 并返回 0 ——
**不许假装通过**（计划书 §5.1 明写）。

跑法::

    python tests/selftest_gui_budget.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
while not (ROOT / "sphere_tracker").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent                      # 按标记上溯（不写死 parents[N]）
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CLICKABLE = {"TButton", "TEntry", "TCombobox", "TCheckbutton", "TRadiobutton"}
INPUTS = {"TEntry", "TCombobox", "TCheckbutton", "TRadiobutton"}
STEP_MARKS = "①②③④⑤⑥"
MODE_WORDS = {"auto", "parallel", "angled"}

L1_INPUTS_MAX = 8
L1_BUTTONS_MAX = 5
L2_CLICKABLE_MAX = 12


def walk(widget):
    yield widget
    for child in widget.winfo_children():
        yield from walk(child)


def _inside_collapsible(widget, stop_at) -> bool:
    """控件是否位于 ``stop_at`` 内某个折叠区（``CollapsibleSection``）里。"""
    from apps._collapsible import CollapsibleSection

    node = getattr(widget, "master", None)
    while node is not None and node is not stop_at:
        if isinstance(node, CollapsibleSection):
            return True
        node = getattr(node, "master", None)
    return False


def step_cards(win) -> list:
    return [w for w in walk(win)
            if w.winfo_class() == "TLabelframe"
            and str(w.cget("text") or "")[:1] in STEP_MARKS]


def measure(card) -> dict[str, int]:
    """返回该步骤卡的 L1（不含折叠）与 L2（含折叠）计数。"""
    l1_inputs = l1_buttons = 0
    l2_clickable = 0
    for w in walk(card):
        cls = w.winfo_class()
        if cls not in CLICKABLE:
            continue
        l2_clickable += 1
        if _inside_collapsible(w, card):
            continue
        if cls in INPUTS:
            l1_inputs += 1
        elif cls == "TButton":
            l1_buttons += 1
    primary = 0
    for w in walk(card):
        try:
            if str(w.cget("style")) == "Go.TButton":
                primary += 1
        except Exception:                                   # noqa: BLE001
            pass
    return {"l1_inputs": l1_inputs, "l1_buttons": l1_buttons,
            "l2_clickable": l2_clickable, "primary": primary}


def collect_texts(win) -> list[str]:
    """窗口里所有可见文案 + 所有下拉的候选值（用来查模式控件与 Markdown）。"""
    out: list[str] = []
    for w in walk(win):
        for key in ("text", "values"):
            try:
                val = w.cget(key)
            except Exception:                               # noqa: BLE001
                continue
            if isinstance(val, (list, tuple)):
                out.extend(str(v) for v in val)
            elif val:
                out.append(str(val))
    return out


# --------------------------------------------------------------------------- #
# 新窗口的参数拼装（2026-10-29 增）
# --------------------------------------------------------------------------- #
# 为什么这一节必须存在
# --------------------
# 上面的结构判据只守"每步几个控件"，**一个字都不管参数怎么拼**。
# 而新窗口真正干活的就是那两行：界面选的板参数有没有正确进到
# `calibrate_stereo.py` 的命令行、录制有没有乱传 `--stereo-mode`。
# 老窗口（`启动程序.py`）里同一件事有 7 节判据守着（见
# `tests/selftest_launcher_params.py` 的第 1/2/2.5/4/5/6/7 节），新窗口却是空的 ——
# 于是"抽成纯函数 + 在这里盯住"就是补上这个缺口。
#
# 这一节**不依赖 Tk**（直接调 `apps/_gui_stereo.py` 的模块级函数），
# 所以无显示器环境也要照跑：结构可以跳过，参数拼装不能跳过。
def _has_seq(argv: list[str], seq: list[str]) -> bool:
    """``seq`` 是否在 ``argv`` 里**连续**出现。

    为什么不用 ``" ".join(argv)`` 的子串判断：``--cols 8`` 是 ``--cols 80``
    的前缀，字符串包含会把错的拼装判成对的。逐段比对才守得住"值真的进来了"。
    """
    n = len(seq)
    if n == 0:
        return False
    return any(argv[i:i + n] == list(seq) for i in range(len(argv) - n + 1))


def _stereo_param_checks() -> list[tuple[str, bool]]:
    """新窗口的命令行拼装判据。返回 (判据名, 是否通过) 列表。"""
    from apps._gui_stereo import calibrate_args, record_args, record_out_dir

    out: list[tuple[str, bool]] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'通过' if ok else '不达标'}] {label}")
        if not ok and detail:
            print(f"      实际：{detail}")
        out.append((label, ok))

    print("\n----- 新窗口的参数拼装（apps/_gui_stereo.py）-----")

    # ---- 对标老窗口 §2：板参数真的来自界面，不是硬编码 ----
    a = calibrate_args("joint", {"cols": "7", "rows": "6", "square": "30",
                                 "target": "25"}, "left", "right", "parallel")
    check("标定：板参数来自界面（7×6 / 30mm / 目标 25 对）",
          _has_seq(a, ["--cols", "7", "--rows", "6", "--square", "30",
                       "--target", "25"]),
          " ".join(a))
    b = calibrate_args("joint", {}, "", "", "parallel")
    check("标定：板参数留空 = 脚本默认 8×5 / 25mm / 40 对（不是落空串）",
          _has_seq(b, ["--cols", "8", "--rows", "5", "--square", "25",
                       "--target", "40"]),
          " ".join(b))
    check("标定：左右相机留空 = left/right（不传空串）",
          _has_seq(b, ["--left", "left", "--right", "right"]), " ".join(b))

    # ---- 对标老窗口 §6/§8b：方法分支的参数形态 ----
    s = calibrate_args("joint_skip_fov", {}, "", "", "angled")
    check("joint_skip_fov：走 --method joint + --stereo-mode angled",
          _has_seq(s, ["--method", "joint"])
          and _has_seq(s, ["--stereo-mode", "angled"]), " ".join(s))
    p = calibrate_args("joint", {}, "", "", "parallel")
    check("普通 joint（平行窗）：带当前模式 parallel",
          _has_seq(p, ["--stereo-mode", "parallel"]), " ".join(p))
    ang = calibrate_args("joint", {}, "", "", "angled")
    check("普通 joint（大角度窗）：带当前模式 angled",
          _has_seq(ang, ["--stereo-mode", "angled"]), " ".join(ang))

    # ---- 对标老窗口 §8b/§8c：--skip-fov 只属于 90° 专用入口 ----
    # 新窗口没有那个入口（等效做法是 joint_skip_fov → --stereo-mode angled）。
    # 一旦有人把 flag 塞进普通 joint，裁剪与内参口径会同时变，属于静默改行为。
    branches = {"joint/parallel": p, "joint/angled": ang,
                "joint_skip_fov": s}
    offenders = [k for k, v in branches.items() if "--skip-fov" in v]
    check("--skip-fov 不出现在任何普通标定分支（那是 90° 专用入口的参数）",
          not offenders,
          "；".join(f"{k}: {' '.join(v)}" for k, v in branches.items()))
    check("若将来引入 90° 入口，--skip-fov 必须与 --stereo-mode angled 成对出现",
          all(("--skip-fov" not in v)
              or _has_seq(v, ["--stereo-mode", "angled"])
              for v in branches.values()))

    # ---- 对标老窗口 §2.5：--backend 是**一个**参数、两台相机共用 ----
    # 出现两次 = 有人给左右各拼了一份（"左 dshow / 右 msmf"会把一路弄挂）；
    # 给了具体后端却一次都不传 = 界面接了不传，用户以为选了。
    with_be = calibrate_args("joint", {}, "", "", "parallel", backend="dshow")
    n_be = with_be.count("--backend")
    check("后端：选 dshow 时 --backend 恰好一次（两台相机共用一个）",
          n_be == 1 and _has_seq(with_be, ["--backend", "dshow"]),
          f"出现 {n_be} 次：{' '.join(with_be)}")
    auto_be = calibrate_args("joint", {}, "", "", "parallel", backend="auto")
    check("后端：auto 时不传 --backend（落到脚本默认，不替用户改行为）",
          "--backend" not in auto_be, " ".join(auto_be))

    # ---- 对标老窗口 §3b/§4：高级项留空不传、填了就传 ----
    e = calibrate_args("joint", {}, "", "", "parallel",
                       stationary="", sharpness="", relaxed=False)
    check("高级项留空 = 不传 --stationary-px / --min-sharpness / --relaxed",
          not any(x in e for x in ("--stationary-px", "--min-sharpness",
                                   "--relaxed")), " ".join(e))
    blank = calibrate_args("joint", {}, "", "", "parallel",
                           stationary="  ", sharpness="\t", relaxed=False)
    check("高级项只填空格也算空（不许把空白串传下去）",
          "--stationary-px" not in blank and "--min-sharpness" not in blank,
          " ".join(blank))
    filled = calibrate_args("joint", {}, "", "", "parallel", stationary="5.5",
                            sharpness="1200", relaxed=True)
    check("高级项填了就真的传下去（5.5 / 1200 / --relaxed）",
          _has_seq(filled, ["--stationary-px", "5.5"])
          and _has_seq(filled, ["--min-sharpness", "1200"])
          and "--relaxed" in filled, " ".join(filled))

    # ---- 对标老窗口 §5/§7：chain 分支与参考段 ----
    import tempfile
    with tempfile.TemporaryDirectory(prefix="stereo_args_") as tmp:
        base = Path(tmp)
        dirs = {"left": base / "left", "right": base / "right",
                "ref_left": base / "ref_left", "ref_right": base / "ref_right"}
        c = calibrate_args("chain", {}, "", "", "angled", dirs=dirs)
        check("chain：走 --method chain --same-board --left-dir/--right-dir",
              _has_seq(c, ["--method", "chain", "--same-board",
                           "--left-dir", str(dirs["left"]),
                           "--right-dir", str(dirs["right"])]), " ".join(c))
        check("chain：参考段不存在时不带 --pair-frame（别指向空目录）",
              "--pair-frame" not in c, " ".join(c))
        dirs["ref_left"].mkdir(parents=True)
        c2 = calibrate_args("chain", {}, "", "", "angled", dirs=dirs)
        check("chain：参考段存在时带 --pair-frame <ref_left> --pair-camera left",
              _has_seq(c2, ["--pair-frame", str(dirs["ref_left"]),
                            "--pair-camera", "left"]), " ".join(c2))
        check("chain：板参数仍然进命令（不是只有 joint 才带）",
              _has_seq(c2, ["--cols", "8", "--rows", "5", "--square", "25",
                            "--target", "40"]), " ".join(c2))

    # ---- 对标老窗口 §8：录制参数 ----
    out_dir = record_out_dir("ball01")
    r = record_args(left="left", right="right", duration="10", out=str(out_dir))
    check("录制：调用 apps/record_stereo.py",
          any("record_stereo.py" in x for x in r), " ".join(r))
    # B-107 的教训：传错模式会先把两台相机开起来，再在体检里报"模式冲突"
    check("录制：不传 --stereo-mode（走 auto = 按标定，B-107）",
          "--stereo-mode" not in r, " ".join(r))
    check("录制：带 --left left --right right --duration 10 --out <目录>",
          _has_seq(r, ["--left", "left", "--right", "right"])
          and _has_seq(r, ["--duration", "10"])
          and _has_seq(r, ["--out", str(out_dir)]), " ".join(r))
    r2 = record_args(left="", right="", duration="", out="x")
    check("录制：左右与时长留空 = left/right/10（不传空串）",
          _has_seq(r2, ["--left", "left"])
          and _has_seq(r2, ["--duration", "10"]), " ".join(r2))
    check("录制：标签 → data/<标签>",
          record_out_dir("ball01") == ROOT / "data" / "ball01",
          str(record_out_dir("ball01")))
    check("录制：标签留空 → data/stereo_session",
          record_out_dir("   ") == ROOT / "data" / "stereo_session",
          str(record_out_dir("   ")))

    # ---- 结构：拼装必须留在函数里，不许被抄回方法（否则上面全部失效）----
    # 这条守的是"抽出来"这件事本身：只要有人把 argv 拼装又写回
    # `_start_calibrate` / `_start_record`，上面的判据就变成在验一段没人调的代码。
    src = (ROOT / "apps" / "_gui_stereo.py").read_text(encoding="utf-8")

    def _body(name: str) -> str:
        head = src.index(f"def {name}(")
        tail = src.find("\n    def ", head)
        return src[head:tail if tail != -1 else len(src)]

    cal_body, rec_body = _body("_start_calibrate"), _body("_start_record")
    check("_start_calibrate 只剩『取控件值 → calibrate_args → tasks.start』",
          "calibrate_args(" in cal_body and '"--cols"' not in cal_body,
          cal_body.replace("\n", " ")[:200])
    check("_start_record 只剩『取控件值 → record_args → tasks.start』",
          "record_args(" in rec_body
          and '"--duration"' not in rec_body
          and '"--stereo-mode"' not in rec_body,
          rec_body.replace("\n", " ")[:200])
    return out


def _summary(results: list[tuple[str, bool]]) -> int:
    print()
    ok = 0
    for name, passed in results:
        print(f"  [{'通过' if passed else '不达标'}] {name}")
        ok += 1 if passed else 0
    failed = len(results) - ok
    print(f"\n合计 {len(results)} 项：通过 {ok}，不达标 {failed}")
    print("判定结果：全部通过" if not failed else "判定结果：有不达标项")
    return 0 if not failed else 1


def _mono_session_checks() -> list[tuple[str, bool]]:
    """单目会话（2026-10-29）：五条判据，**不需要 Tk**。

    为什么放在 `App()` 之前跑：这些守的是"扫目录 / 状态口径 / 软删除"，
    与窗口无关。塞进 Tk 分支的后果是——没有显示器时它们会被一起跳过，
    而"跳过"在这里等于没有防线（`selftest_preview_lib` 已经吃过这个亏）。
    全部用 `tempfile`，不许往真实 `data/mono/` 写。
    """
    import json
    import tempfile

    from apps import _mono_sessions as ms

    out: list[tuple[str, bool]] = []

    def ck(label: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'通过' if ok else '不达标'}] {label}"
              + (f"（{detail}）" if detail else ""))
        out.append((label, ok))

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "mono"
        good = root / "20260101-000000"
        (good / "derived").mkdir(parents=True)
        (good / "session.json").write_text(json.dumps({
            "schema": "sphere-mono-session", "version": 1, "camera": "main",
            "width": 1280, "height": 720, "fps": 30.0, "frame_count": 300,
            "duration_s": 10.0, "encoder": "HFYU", "video": "clip.mkv",
        }, ensure_ascii=False), encoding="utf-8")
        bad = root / "20260102-000000"
        bad.mkdir()
        (bad / "session.json").write_text("{坏掉的", encoding="utf-8")

        rows = ms.scan_mono_sessions(root)
        ck("坏会话也返回，不静默跳过",
           len(rows) == 2 and any(r.error for r in rows)
           and any(not r.error for r in rows), f"{len(rows)} 条")
        ck("按目录名倒序（不看 mtime）",
           [r.id for r in rows] == ["20260102-000000", "20260101-000000"],
           str([r.id for r in rows]))

        def state_of() -> str:
            return next(r for r in ms.scan_mono_sessions(root)
                        if r.id == "20260101-000000").state

        before = state_of()
        (good / "derived" / "traj.csv").write_text("frame,x\n", encoding="utf-8")
        after = state_of()
        ck("derived 里有轨迹才算「已检测」",
           before != after and "检测" in after, f"{before} -> {after}")

        ms.delete_mono_session(good, root)
        ck("删除是软删除（原目录消失 + 进 _trash + 扫描少一条）",
           (not good.exists())
           and any((root.parent / "_trash").glob("*_mono"))
           and len(ms.scan_mono_sessions(root)) == 1)

    # 按标记上溯取仓库根：`ROOT` 这个变量在本文件里指 `tests/` 的**上上层**
    # （它同时被别的路径用），拿它 `.parent` 拼会少一层。
    repo = next(p for p in Path(__file__).resolve().parents
                if (p / "sphere_tracker").is_dir())
    src = (repo / "apps" / "_gui_mono.py").read_text(encoding="utf-8")
    ck("单目窗仍提供「本次录制存到」与「历史会话」",
       "本次录制存到" in src and "历史会话" in src)

    # ---- 球径留空时命令行传什么（2026-10-29 增，纯函数、不需要 Tk） ----
    # 守的坑：三处 ``or "40"`` 让"界面显示 21.6、命令行传 40"成立过。
    # Z ∝ D，40 对这颗 21.6 的球等于让深度偏大 85%，而界面上看不出来。
    # 断言的是**函数返回值**（界面与命令行共用同一个入口），
    # 把兜底改回 "40" 这条立刻红 —— 反向自证见交接文档。
    from apps._gui_mono import EFFECTIVE_DIAMETER_MM, diameter_arg

    want = f"{EFFECTIVE_DIAMETER_MM:g}"
    ck("球径留空 → 命令行用有效直径（不是旧默认 40）",
       diameter_arg("") == want and diameter_arg("40") == "40",
       f"空→{diameter_arg('')!r}，填 40→{diameter_arg('40')!r}")
    ck("球径只有空白 → 也退回有效直径",
       diameter_arg("   ") == want, repr(diameter_arg("   ")))
    ck("球径填了就用填的（用户的数不被兜底顶掉）",
       diameter_arg(" 30 ") == "30", repr(diameter_arg(" 30 ")))
    return out


def main() -> int:
    print("=" * 74)
    print("GUI 预算自测：分窗后的每一步都必须留在一屏之内")
    print("=" * 74)

    # 参数拼装判据先跑：它直接调纯函数、不需要 Tk，所以**没有显示器也要跑**。
    # 结构判据可以跳过，参数拼装不行 —— 跳过的那部分一律不算验证过。
    results: list[tuple[str, bool]] = _stereo_param_checks()
    results += _mono_session_checks()

    try:
        from apps._gui_app import App
        app = App()
    except Exception as exc:                                # noqa: BLE001
        if type(exc).__name__ in ("TclError", "TclObjTypeError"):
            print(f"\n[跳过] 没有可用显示环境，无法实例化 Tk（{exc}）")
            print("       上面的参数拼装判据已经跑过；结构与文案判据未验证")
            if all(ok for _n, ok in results):
                print("判定结果：跳过（未验证，不算通过）")
                return 0
            return _summary(results)
        raise

    try:
        app.update_idletasks()
        windows = [("mono", app.open_mono()),
                   ("stereo(parallel)", app.open_stereo("parallel")),
                   ("stereo(angled)", app.open_stereo("angled")),
                   ("console", app.open_console())]
        app.update_idletasks()

        for name, win in windows:
            print(f"\n----- {name} -----")
            cards = step_cards(win)
            if name.startswith("stereo") and not cards:
                results.append((f"{name} 有步骤卡", False))
            for card in cards:
                title = str(card.cget("text"))
                m = measure(card)
                ok_l1 = (m["l1_inputs"] <= L1_INPUTS_MAX
                         and m["l1_buttons"] <= L1_BUTTONS_MAX)
                ok_l2 = m["l2_clickable"] <= L2_CLICKABLE_MAX
                ok_primary = m["primary"] <= 1
                print(f"  {title[:26]:28s} L1 输入 {m['l1_inputs']}/{L1_INPUTS_MAX}"
                      f"  按钮 {m['l1_buttons']}/{L1_BUTTONS_MAX}"
                      f"  L2 可点 {m['l2_clickable']}/{L2_CLICKABLE_MAX}"
                      f"  主操作 {m['primary']}")
                results.append((f"{name}「{title}」L1 预算", ok_l1))
                results.append((f"{name}「{title}」L2 预算", ok_l2))
                results.append((f"{name}「{title}」主操作 ≤1", ok_primary))

            # 模式控件：窗口里不许有 auto/parallel/angled 作候选或文案
            texts = collect_texts(win)
            bad = [t for t in texts if t.strip().lower() in MODE_WORDS]
            print(f"  模式选择控件 = {len(bad)}（必须为 0）")
            results.append((f"{name} 无模式选择控件", not bad))

            # Markdown 星号：B-96 的教训（会被原样印出来）
            stars = [t for t in texts if "**" in t]
            print(f"  未渲染 Markdown = {len(stars)}（必须为 0）")
            results.append((f"{name} 无 Markdown 星号", not stars))

        # ---------- 界面契约（2026-10-28 增，防回退） ----------
        # 守的是本轮踩过的坑：分窗时把跨模式工具与装置身份弄丢/藏起来，
        # 用户找不到相机管理、也看不出左右相机是哪台（B-103/B-104/B-106）。
        print("\n----- 入口与相机身份（防回退） -----")
        hub_texts = " ".join(collect_texts(app.hub))
        for label, needle in (("门厅有「相机设备」入口", "相机设备"),
                              ("门厅有「任务管理器和运行日志」入口",
                               "任务管理器和运行日志")):
            ok = needle in hub_texts
            print(f"  [{'通过' if ok else '不达标'}] {label}")
            results.append((label, ok))

        for name, win in windows:
            if name == "console":
                continue
            texts = " ".join(collect_texts(win))
            ok = "相机" in texts
            print(f"  [{'通过' if ok else '不达标'}] {name} 第①步显示相机身份")
            results.append((f"{name} 第①步显示相机身份", ok))

        # ---------- 单目窗：球径留空时命令行到底传了什么 ----------
        # 守的坑（2026-10-29）：三处 ``or "40"`` 兜底让"界面显示 21.6、命令行
        # 传 40"成为可能。Z ∝ D，那等于用户拿着偏大 85% 的深度当测量，
        # 而界面上一个字都看不出来。
        # 为什么断言的是**命令行实际拿到什么**、而不是源码里有没有某个字面量：
        # 后者一改写法（抽函数、换常量）就失效，且抓不到"留空走另一条分支"。
        # 这一条可以反向自证：把兜底改回 "40"，这里立刻红。
        from apps._gui_mono import EFFECTIVE_DIAMETER_MM, LEGACY_DIAMETER

        mono = dict(windows)["mono"]
        captured: list[list[str]] = []

        class _StubTasks:
            """只收集命令行，不真起进程（否则测试会去开相机）。"""

            def start(self, cmd, *_a, **_k):
                captured.append(list(cmd))

        real_tasks = mono.app.tasks
        mono.app.tasks = _StubTasks()
        try:
            mono.var_diameter.set("")           # 用户把输入框清空
            mono._start_realtime()
        finally:
            mono.app.tasks = real_tasks
        got = ""
        if captured and "--ball-diameter" in captured[0]:
            cmd = captured[0]
            got = cmd[cmd.index("--ball-diameter") + 1]
        try:
            ok_diam = (got != LEGACY_DIAMETER
                       and abs(float(got) - EFFECTIVE_DIAMETER_MM) < 1e-9)
        except ValueError:
            ok_diam = False
        print(f"  [{'通过' if ok_diam else '不达标'}] 单目窗球径留空 → "
              f"命令行 --ball-diameter {got or '(缺失)'}"
              f"（应为 {EFFECTIVE_DIAMETER_MM:g}，不是旧默认 {LEGACY_DIAMETER}）")
        results.append(("单目窗球径留空传有效直径", ok_diam))

        # ---------- 单目窗：离线视频传不传 fx、焦距估计传不传球径 ----------
        # 与上面同源的失效方式：界面里填了、命令行里没有（或反过来）。
        # 用假的会话目录 + 桩掉 find_clip，避免真去读磁盘、也避免弹文件选择框。
        import apps._gui_mono as mono_mod

        class _FakeSess:
            path = Path("outputs/_fake_mono_session")

        real_find = mono_mod.find_clip
        real_sess = mono._session
        mono.app.tasks = _StubTasks()
        try:
            mono_mod.find_clip = lambda _p: Path("outputs/_fake_mono_session/clip.mkv")
            mono._session = _FakeSess()

            captured.clear()
            mono.var_diameter.set("")
            mono.var_focal.set("1500")
            mono._start_video()
            cmd = captured[0] if captured else []
            ok_fx = ("--focal" in cmd
                     and cmd[cmd.index("--focal") + 1] == "1500")
            print(f"  [{'通过' if ok_fx else '不达标'}] 离线视频：fx 填 1500 → "
                  f"命令里有 --focal 1500（实际 {' '.join(cmd[-6:]) or '(无命令)'}）")
            results.append(("离线视频传手动 fx", ok_fx))

            captured.clear()
            mono.var_focal.set("")
            mono._start_video()
            cmd = captured[0] if captured else []
            ok_nofx = "--focal" not in cmd
            print(f"  [{'通过' if ok_nofx else '不达标'}] 离线视频：fx 留空 → "
                  f"命令里没有 --focal（实际 {' '.join(cmd[-6:]) or '(无命令)'}）")
            results.append(("离线视频留空不传 fx", ok_nofx))

            captured.clear()
            mono.var_focal.set("abc")
            mono._start_video()
            cmd = captured[0] if captured else []
            note = mono.status._label.cget("text")
            ok_badfx = "--focal" not in cmd and "不是数字" in note
            print(f"  [{'通过' if ok_badfx else '不达标'}] fx 填非数字 → 不传且状态条提示"
                  f"（状态条尾部：…{note[-30:]}）")
            results.append(("fx 非数字不静默", ok_badfx))

            captured.clear()
            mono.var_focal.set("")
            mono._start_focal_estimate()
            cmd = captured[0] if captured else []
            got_d = (cmd[cmd.index("--ball-diameter") + 1]
                     if "--ball-diameter" in cmd else "")
            ok_est = got_d == f"{EFFECTIVE_DIAMETER_MM:g}"
            print(f"  [{'通过' if ok_est else '不达标'}] 单点焦距估计：留空 → 传 "
                  f"--ball-diameter {got_d or '(缺失)'}（应为 "
                  f"{EFFECTIVE_DIAMETER_MM:g}）")
            results.append(("焦距估计传球径", ok_est))

            # 换了球：框里写 30，命令里就必须是 30（f 与 D 同比例，错一个都白标）
            captured.clear()
            mono.var_diameter.set("30")
            mono._start_focal_estimate()
            cmd = captured[0] if captured else []
            got_d30 = (cmd[cmd.index("--ball-diameter") + 1]
                       if "--ball-diameter" in cmd else "")
            ok_est30 = got_d30 == "30"
            print(f"  [{'通过' if ok_est30 else '不达标'}] 单点焦距估计：框里填 30 → "
                  f"传 --ball-diameter {got_d30 or '(缺失)'}（应为 30）")
            results.append(("焦距估计用框里的球径", ok_est30))
        finally:
            mono_mod.find_clip = real_find
            mono._session = real_sess
            mono.var_focal.set("")
            mono.var_diameter.set("")
            mono.app.tasks = real_tasks

        # ---------- 单目窗：检测器下拉 / --vis-json / 「看 3D」（2026-10-29 增）----------
        # 守的是用户报的那个问题的**上游**：「单目那里没有查看器」。
        # 三条判据分别对应三处可能"界面有、命令行没有"的静默失效：
        #   ① 界面选了 hough，命令行还是 color（这颗球 color 40 帧检出 0 帧）；
        #   ② 跑完检测但没写 visualization.json（于是"看 3D"永远灰着）；
        #   ③ smart 承诺"0 检出自动换 hough"，实际没有真的重跑。
        from apps._gui_mono import (SOURCE_OFFLINE, SOURCE_REALTIME,
                                   detector_choices)

        captured.clear()
        mono.app.tasks = _StubTasks()
        try:
            mono_mod.find_clip = lambda _p: Path("outputs/_fake_mono_session/clip.mkv")
            mono._session = _FakeSess()
            mono.var_diameter.set("")
            mono.var_focal.set("")

            mono.var_detector.set("hough")
            mono._start_video()
            cmd = captured[0] if captured else []
            got_m = cmd[cmd.index("--method") + 1] if "--method" in cmd else ""
            ok_m = got_m == "hough"
            print(f"  [{'通过' if ok_m else '不达标'}] 检测器选 hough → 命令行 "
                  f"--method {got_m or '(缺失)'}（界面选了必须真的传下去）")
            results.append(("单目窗检测器真的进命令行", ok_m))

            got_vis = cmd[cmd.index("--vis-json") + 1] if "--vis-json" in cmd else ""
            ok_vis = got_vis.replace("\\", "/").endswith(
                "_fake_mono_session/derived/visualization.json")
            print(f"  [{'通过' if ok_vis else '不达标'}] 离线检测带 --vis-json → "
                  f"{got_vis or '(缺失)'}（落在会话 derived/ 里，「看 3D」才找得到）")
            results.append(("离线检测导出可视化", ok_vis))

            # smart：起步用 color，且记下"0 检出就换 hough"
            captured.clear()
            mono.var_detector.set("smart")
            mono._start_video()
            cmd = captured[0] if captured else []
            got_auto = cmd[cmd.index("--method") + 1] if "--method" in cmd else ""
            print(f"  [{'通过' if got_auto == 'color' else '不达标'}] smart 第一遍用 "
                  f"--method {got_auto or '(缺失)'}（smart 不是注册表里的名字，"
                  f"起步只能给 color）")
            results.append(("smart 起步用 color", got_auto == "color"))

            class _FailTask:
                """桩：模拟 run_video 的 0 检出（退出码 2）。"""
                exit_code = 2

            captured.clear()
            mono._on_detected(_FailTask())
            cmd = captured[0] if captured else []
            got_retry = cmd[cmd.index("--method") + 1] if "--method" in cmd else ""
            ok_retry = got_retry == "hough"
            print(f"  [{'通过' if ok_retry else '不达标'}] smart 遇到退出码 2 → 自动重跑 "
                  f"--method {got_retry or '(缺失)'}（承诺了就得真的重跑）")
            results.append(("smart 0 检出自动换 hough", ok_retry))

            # 「看 3D」：没有结果时必须禁用（可点却打不开的按钮最伤信任）
            mono._session = _FakeSess()
            mono._refresh_session()
            disabled = "disabled" in mono.btn_view3d.state()
            print(f"  [{'通过' if disabled else '不达标'}] 没有三维结果时「看 3D」禁用"
                  f"（state={mono.btn_view3d.state()}）")
            results.append(("看 3D 无结果时禁用", disabled))

            # 有结果 → 启用，且命令里真的是 view_3d.py + 那个文件
            import json
            import tempfile
            with tempfile.TemporaryDirectory(prefix="mono_view_") as tmp:
                sess = Path(tmp) / "20260101-000000"
                (sess / "derived").mkdir(parents=True)
                vis = sess / "derived" / "visualization.json"
                vis.write_text(json.dumps({
                    "schema": "sphere-mono-visualization", "version": 1,
                    "mode": "mono", "session": sess.name,
                    "trajectory": [{"frame": 0, "t": 0.0, "xyz": [0, 0, 1.0],
                                    "confident": True}]}), encoding="utf-8")

                class _RealSess:
                    path = sess

                mono._session = _RealSess()
                mono._refresh_session()
                enabled = "disabled" not in mono.btn_view3d.state()
                captured.clear()
                mono._open_3d()
                cmd = captured[0] if captured else []
                ok_3d = (enabled and any("view_3d.py" in x for x in cmd)
                         and any(str(vis) == x for x in cmd))
                print(f"  [{'通过' if ok_3d else '不达标'}] 有结果时「看 3D」可点，"
                      f"命令 = {Path(cmd[1]).name if len(cmd) > 1 else '(无)'} "
                      f"+ …（enabled={enabled}）")
                results.append(("看 3D 打开查看器", ok_3d))

            # 「开始识别」按来源分发：实时不能误落到离线（那会去找 clip 而静默失败）
            captured.clear()
            mono.var_source.set(SOURCE_REALTIME)
            mono._start_by_source()
            cmd = captured[0] if captured else []
            ok_rt = any("run_realtime.py" in x for x in cmd)
            print(f"  [{'通过' if ok_rt else '不达标'}] 来源=实时 → 调 run_realtime.py"
                  f"（{' '.join(cmd[1:2]) or '(无命令)'}）")
            results.append(("开始识别按来源分发", ok_rt))
            print(f"  检测器候选 = {list(detector_choices())}"
                  f"（默认 {mono.var_detector.get()!r}）")
        finally:
            mono_mod.find_clip = real_find
            mono._session = real_sess
            mono.var_focal.set("")
            mono.var_diameter.set("")
            mono.var_detector.set("color")
            mono.var_source.set(SOURCE_OFFLINE)
            mono._auto_retry_pending = False
            mono.app.tasks = real_tasks
            mono._refresh_session()

        try:
            from apps._gui_stereo import SessionPicker, scan_sessions
            rows = scan_sessions()
            ok_scan = isinstance(rows, list)
            ok_cls = callable(SessionPicker)
        except Exception as exc:                        # noqa: BLE001
            print(f"  会话模块导入失败：{exc}")
            rows, ok_scan, ok_cls = [], False, False
        print(f"  [{'通过' if ok_scan else '不达标'}] 会话扫描可用（{len(rows)} 个）")
        print(f"  [{'通过' if ok_cls else '不达标'}] 会话选择器可导入")
        results.append(("会话扫描可用", ok_scan))
        results.append(("会话选择器可导入", ok_cls))
    finally:
        try:
            app.destroy()
        except Exception:                                   # noqa: BLE001
            pass

    return _summary(results)


if __name__ == "__main__":
    raise SystemExit(main())
