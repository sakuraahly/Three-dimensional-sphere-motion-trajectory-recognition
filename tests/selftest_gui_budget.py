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
# 大角度（angled）第②步：链式标定比 joint 多「标定方法下拉 + 采集标定板图 +
# 检查链式输入」三个控件，是**真实且必要**的内容差异，不是"长回一个窗口装全部"。
# 单独给它放宽 L2 上限，其余窗口仍守 12。
ANGLED_L2_CLICKABLE_MAX = 14


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
    from apps._gui_stereo import (calibrate_args, offline_args, record_args,
                                  record_out_dir)

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

    # ---- 录制：预览开关必须传下去（"标定完实时画面没了"的根因）----
    #
    # 老界面一直传 `--preview`，新界面重做后**从没传过** —— 而它是 store_true，
    # 不传就是没有窗口，且一声不响。这类"界面上少一个参数"的缺陷截图看不出来，
    # 只能靠比对命令行。
    # ---- 勾了锚点但没点选球心：界面必须先说清，而不是让人等一次进程启动 ----
    import tempfile
    from apps._actions import has_seeds
    with tempfile.TemporaryDirectory() as td:
        sess = Path(td)
        no_seed = not has_seeds(sess)
        (sess / "derived").mkdir()
        (sess / "derived" / "seeds.json").write_text("{}", encoding="utf-8")
        with_seed = has_seeds(sess)
    check("has_seeds：没有 derived/seeds.json → False；有 → True",
          no_seed and with_seed)

    rec = record_args(out="data/stereo", duration="10")
    check("录制必须带 --preview（否则录制时没有任何实时画面，且不报错）",
          "--preview" in rec, " ".join(rec))
    check("录制必须带「--wait-key space」：先出画面，按空格才进倒计时",
          _has_seq(rec, ["--wait-key", "space"]), " ".join(rec))

    # ---- 半径锚点：勾选框 → 命令行，必须真的传下去 ----
    #
    # 这条判据存在的理由：本项目已经在"界面上有、命令行没收到"上栽过三次
    # （球径按 mm 传给按米收的参数、「检查装置」一个参数都不传、后端与编号打架）。
    # 所以这里不比"有没有这个控件"，只比"勾了以后命令行长什么样"。
    offl_off = offline_args("data/stereo/20261024-173619", "parallel")
    offl_on = offline_args("data/stereo/20261024-173619", "parallel", anchor=True)
    check("半径锚点默认关：不勾就不传 --anchor-gate", "--anchor-gate" not in offl_off,
          " ".join(offl_off))
    check("半径锚点勾上：命令行必须出现「--anchor-gate on」",
          _has_seq(offl_on, ["--anchor-gate", "on"]), " ".join(offl_on))
    check("--stereo-mode 只出现一次（这个窗口存在的意义就是按本装置解算）",
          offl_on.count("--stereo-mode") == 1, " ".join(offl_on))

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

    # ---- 装置档案：标定落盘按 active profile 走（2026-11-03）----
    # 标定要写进"当前装置专属"的 calibration_path，而不是总覆盖 config/stereo_calibration
    # —— 否则换 90° 装置重新标定就把平行装置的标定冲掉了。
    no_out = calibrate_args("joint", {}, "", "", "parallel")
    check("不传 out 时不带 --out（走脚本默认 config/stereo_calibration）",
          "--out" not in no_out, " ".join(no_out))
    with_out = calibrate_args("joint", {}, "", "", "parallel",
                              out="config/calib/parallel_rig")
    check("传 out=config/calib/parallel_rig 时带 --out 该路径",
          _has_seq(with_out, ["--out", "config/calib/parallel_rig"]),
          " ".join(with_out))
    blank_out = calibrate_args("joint", {}, "", "", "parallel", out="  ")
    check("out 只填空格也算空（不传 --out 空串）",
          "--out" not in blank_out, " ".join(blank_out))

    # ---- 对标老窗口 §5/§7：chain 分支与参考段 ----
    import tempfile
    with tempfile.TemporaryDirectory(prefix="stereo_args_") as tmp:
        base = Path(tmp)
        dirs = {"left": base / "left", "right": base / "right",
                "ref_left": base / "ref_left", "ref_right": base / "ref_right",
                "board_b": base / "board_b", "pair": base / "pair"}
        c = calibrate_args("chain", {}, "", "", "angled", dirs=dirs)
        check("chain：走 --method chain --same-board --left-dir/--right-dir",
              _has_seq(c, ["--method", "chain", "--same-board",
                           "--left-dir", str(dirs["left"]),
                           "--right-dir", str(dirs["right"])]), " ".join(c))
        # 2026-11-03 修正：单板模式（--same-board）的参考段参数是 **--ref-left/--ref-right**，
        # 不是 --pair-frame（那是「双板」模式 --board-b-dir 才用的；calibrate_stereo 的
        # _chain_input_error 明确说单板模式不许给 pair-frame）。旧判据把 pair-frame 写死了，
        # 导致这个错误参数名一直没被发现——chain 标定在真机上拿不到参考段图。
        check("chain：参考段不存在时也不带 --pair-frame（单板模式不认这个参数）",
              "--pair-frame" not in c, " ".join(c))
        dirs["ref_left"].mkdir(parents=True)
        dirs["ref_right"].mkdir(parents=True)
        c2 = calibrate_args("chain", {}, "", "", "angled", dirs=dirs)
        check("chain：参考段存在时带 --ref-left <ref_left> --ref-right <ref_right>",
              _has_seq(c2, ["--ref-left", str(dirs["ref_left"]),
                            "--ref-right", str(dirs["ref_right"])]), " ".join(c2))
        check("chain：绝不带 --pair-frame / --pair-camera（那是双板模式的参数）",
              "--pair-frame" not in c2 and "--pair-camera" not in c2, " ".join(c2))
        check("chain：板参数仍然进命令（不是只有 joint 才带）",
              _has_seq(c2, ["--cols", "8", "--rows", "5", "--square", "25",
                            "--target", "40"]), " ".join(c2))

        # 双板链式（chain_dual，2026-11-03）：两块刚性固定的板 A/B，各看一块。
        # 用 --board-b-dir + --pair-frame + --pair-camera，**不加 --same-board**。
        dirs["board_b"].mkdir(parents=True)
        dirs["pair"].mkdir(parents=True)
        c3 = calibrate_args("chain_dual", {}, "", "", "angled", dirs=dirs)
        check("chain_dual：--board-b-dir + --pair-frame + --pair-camera，无 --same-board",
              _has_seq(c3, ["--method", "chain"])
              and _has_seq(c3, ["--board-b-dir", str(dirs["board_b"])])
              and _has_seq(c3, ["--pair-frame", str(dirs["pair"]), "--pair-camera", "left"])
              and "--same-board" not in c3 and "--ref-left" not in c3, " ".join(c3))

        # ---- 基线核对 / FOV 复用（2026-11-04）----
        # 两条都是"档案里的值要真的进命令"：不传等于后端那段物理护栏不执行，
        # 而它不会报错、只会静默少一项证据。
        cb = calibrate_args("chain_dual", {}, "", "", "angled", baseline=124.0)
        check("chain_dual：档案里的期望基线进 --baseline-mm",
              _has_seq(cb, ["--baseline-mm", "124"]), " ".join(cb))
        check("baseline 非法值不拼进命令（否则 argparse 直接 exit 2）",
              "--baseline-mm" not in calibrate_args("joint", {}, "", "", "parallel",
                                                    baseline="abc"), "abc 被忽略")
        cf = calibrate_args("joint", {}, "", "", "parallel",
                            fov_from_profile="平行装置")
        check("joint：档案名进 --fov-from-profile（复用已量好的裁剪参数）",
              _has_seq(cf, ["--fov-from-profile", "平行装置"]), " ".join(cf))
        check("chain：不下传 --fov-from-profile（chain 不做裁剪，后端会报互斥）",
              "--fov-from-profile" not in calibrate_args(
                  "chain", {}, "", "", "angled", fov_from_profile="平行装置"),
              "chain 不传")

        # ---- 链式输入自检：双板要查 pair（2026-11-04）----
        from apps._actions import check_chain_inputs as _chk
        cd1 = _chk(cols="8", rows="5", square="25", mode="dual")
        check("双板自检：--mode dual（查 left/board_b/pair 三段）",
              _has_seq(cd1, ["--mode", "dual"]), " ".join(cd1))
        check("单板自检仍默认 --mode single（行为没变）",
              _has_seq(_chk(cols="8", rows="5", square="25"),
                       ["--mode", "single"]), "默认 single")

    # ---- 对标老窗口 §8：录制参数 ----
    out_dir = record_out_dir()
    r = record_args(left="left", right="right", duration="10", out=str(out_dir))
    check("录制：调用 apps/record_stereo.py",
          any("record_stereo.py" in x for x in r), " ".join(r))
    # B-107 的教训是"别把错模式写死"；但 2026-11-04 起口径变了：
    # `record_stereo.plan_record` 的体检已经**前移到开相机之前**（不符只读一个 npz
    # 就返回 2），所以"传了会白开相机"不再成立；而"不传"的代价变成了
    # 「在 90° 窗口用平行标定录 → 会话被标成 parallel」——元数据与装置不符，
    # 隔离失效（用户实测 20261103-093820 就是这样）。
    r_ang = record_args(left="left", right="right", duration="10",
                        out=str(out_dir), mode="angled")
    check("录制：显式传本窗装置几何（--stereo-mode angled）",
          _has_seq(r_ang, ["--stereo-mode", "angled"]), " ".join(r_ang))
    check("录制：没给模式就不许凭空写 auto（保持脚本默认）",
          "--stereo-mode" not in r, " ".join(r))
    check("录制：带 --left left --right right --duration 10 --out <目录>",
          _has_seq(r, ["--left", "left", "--right", "right"])
          and _has_seq(r, ["--duration", "10"])
          and _has_seq(r, ["--out", str(out_dir)]), " ".join(r))
    r2 = record_args(left="", right="", duration="", out="x")
    check("录制：左右与时长留空 = left/right/10（不传空串）",
          _has_seq(r2, ["--left", "left"])
          and _has_seq(r2, ["--duration", "10"]), " ".join(r2))
    # 2026-10-31：`--out` 的语义改成会话**根**（时间戳子目录由 record_stereo 的
    # `create_session` 建，重名由它加 -01），标签改走 `--label` 进 session.json。
    # 旧断言"标签 → data/<标签>"因此不再成立，换成这两条（口径仍各只有一处）。
    from sphere_tracker.paths import stereo_roots

    check("录制：--out 落在会话扫描根里（否则录完第④步/工作台都列不出来）",
          any(Path(record_out_dir()).resolve() == Path(p).resolve()
              for p in stereo_roots()), str(record_out_dir()))
    # 2026-11-04：新录的会话按装置分区（data/stereo/<mode>/）。这里守住
    # "分区根只比会话根多一层" —— 扫描器（candidate_session_dirs）往里走一层
    # 就能找到会话；多套几层它就不认了，录完在列表里看不到。
    from sphere_tracker.paths import STEREO_ROOT

    for _m in ("parallel", "angled"):
        _d = Path(record_out_dir(_m))
        check(f"录制：{_m} 落 data/stereo/{_m}（分区根，只多一层）",
              _d.name == _m and _d.parent.resolve() == STEREO_ROOT.resolve(), str(_d))
    check("录制：标签 → --label（不拼路径；空白不传）",
          _has_seq(record_args(out=str(out_dir), label="ball01"),
                   ["--label", "ball01"])
          and "--label" not in record_args(out=str(out_dir), label="   "),
          " ".join(record_args(out=str(out_dir), label="ball01")))

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


def _mono_seed_checks() -> list[tuple[str, bool]]:
    """人工种子（2026-10-30 增）：**不需要 Tk** 的那一半判据。

    为什么要有这一节
    ----------------
    镜面球（镀铬、宿舍背景）没有可靠的自动初始化：288 帧实测 ``color`` 0/288
    落在球上、``hough`` 只有 1.4%、半径中位 69.5 px 而真值约 91 px。
    于是"点一下球"成了唯一正路 —— 而它整条链是
    ``点击 → canvas_to_image → 半径(via 第二点) → seed.json → --seed x,y,r``，
    中间任何一环写反（缩放比取倒数、半径用了直径、mm 当 m、带着另一套分辨率的
    坐标）都**不会报错**，只会让轨迹悄悄偏移。所以每一环都要有断言。

    这一节不碰 Tk：坐标映射 / 两点定圆 / 深度换算 / 读 seed.json 全是纯函数，
    没有显示器也照跑（跳过 = 未验证）。
    """
    import json
    import tempfile

    from apps._gui_mono import SEED_DETECTOR, detector_choices, seed_arg_from_file
    from apps.seed_picker import canvas_to_image, radius_from_points, z_estimate_m

    out: list[tuple[str, bool]] = []

    def ck(label: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'通过' if ok else '不达标'}] {label}"
              + (f"（{detail}）" if detail else ""))
        out.append((label, ok))

    # ---- 画布 → 原图 ----
    # 缩放比取倒数（乘而不是除）是最容易犯的错：760 宽的预览点 (400,300)
    # 会算成 (160,120)，于是种子落在左上角、跟踪立刻跟丢。
    px = canvas_to_image(400, 300, 0.4, (0.0, 0.0))
    ck("画布 → 原图：scale=0.4 时 (400,300) → (1000,750)",
       abs(px[0] - 1000.0) < 1e-6 and abs(px[1] - 750.0) < 1e-6, f"实际 {px}")

    # ---- 两点定圆 ----
    r = radius_from_points((100.0, 100.0), (100.0, 190.0))
    ck("两点定圆：球心 (100,100) 到球边 (100,190) → 半径 90",
       abs(r - 90.0) < 1e-9, f"实际 r={r:g}")

    # ---- Z = f·D/(2r)，D 是 mm、Z 要 m ----
    z = z_estimate_m(1387.5, 21.6, 91.0)
    want = 1387.5 * 0.0216 / (2 * 91.0)
    ck("深度换算：fx 1387.5 / D 21.6 mm / r 91 px → Z 约 0.165 m（mm→m 没漏）",
       z is not None and abs(z - want) < 1e-12, f"实际 Z={z}，应为 {want:.6f}")

    with tempfile.TemporaryDirectory(prefix="mono_seed_") as tmp:
        sess = Path(tmp) / "20260101-000000"
        (sess / "derived").mkdir(parents=True)
        seed = sess / "derived" / "seed.json"
        seed.write_text(json.dumps({"x": 100, "y": 50, "radius": 91, "frame": 0,
                                    "image_size": [1920, 1080],
                                    "created": "2026-01-01 00:00:00"},
                                   ensure_ascii=False), encoding="utf-8")

        # ---- 读 seed.json → --seed 的值 ----
        value, note = seed_arg_from_file(seed, image_size=(1920, 1080))
        try:
            parsed = [float(v) for v in value.split(",")]
        except ValueError:
            parsed = []
        ck("有 seed.json → --seed 的值就是文件里的 x,y,r（3 个数，能直接解析）",
           parsed == [100.0, 50.0, 91.0] and not note,
           f"值 {value!r}，提示 {note!r}")

        # 检测器名必须真的在注册表里：写错一个字母，run_video 会在 argparse
        # 就退出（"点了没反应"），而界面照样显示"已按种子跟踪"。
        ck("带种子时强制用的检测器在注册表里（specular）",
           SEED_DETECTOR == "specular" and SEED_DETECTOR in detector_choices(),
           f"{SEED_DETECTOR} in {list(detector_choices())}")

        # ---- 分辨率不符：坐标是另一套画布的 ----
        other, other_note = seed_arg_from_file(seed, image_size=(1280, 720))
        ck("种子分辨率与录像不符 → 不用它（退回自动检测），并说明原因",
           other == "" and "对不上" in other_note, f"值 {other!r}，提示 {other_note!r}")

        # ---- 读不动 / 字段不全：一律不抛 ----
        broken = sess / "derived" / "broken.json"
        broken.write_text("{坏掉的", encoding="utf-8")
        b_value, b_note = seed_arg_from_file(broken, image_size=(1920, 1080))
        ck("seed.json 坏了 → 不抛，值空 + 一句话",
           b_value == "" and bool(b_note), f"值 {b_value!r}，提示 {b_note!r}")

        gone_value, gone_note = seed_arg_from_file(sess / "derived" / "nope.json")
        ck("seed.json 不存在 → 不抛，值空 + 一句话",
           gone_value == "" and bool(gone_note),
           f"值 {gone_value!r}，提示 {gone_note!r}")

        seed.write_text(json.dumps({"x": 100, "y": 50}, ensure_ascii=False),
                        encoding="utf-8")
        m_value, m_note = seed_arg_from_file(seed)
        ck("缺 radius 的 seed.json → 不用（半径是 Z 的分母，不能猜）",
           m_value == "" and "radius" in m_note, f"值 {m_value!r}，提示 {m_note!r}")
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

    # ---- 球径：界面是 mm、命令行是米（2026-10-31 改口径） ----
    # 守的坑本来有两条：三处 ``or "40"`` 让"界面显示 21.6、命令行传 40"成立过；
    # 现在多了一条更狠的 —— **单位**。界面按 mm 收（标签「球直径（有效值）」，
    # 默认 21.6），而 ``--ball-diameter`` 与 ``AppConfig.ball_diameter_m`` 按米收
    # （``sphere_tracker/config.py`` = 0.0216）。原样传 21.6 等于"球有 21.6 米大"，
    # 实测 ``data/mono/20261029-205522`` 上 40 帧**一个三维点都没有**。
    # 所以这里的期望值全部是米，并多看一条"永远 < 1"。
    from apps._gui_mono import diameter_arg, diameter_problem

    # 2026-11-03 改：球径**没有默认值**了。留空 = 不传参数（而不是兜底成 21.6），
    # 由 `run_video` 在启动前拒跑、界面在这里先拦住。原因见 config.py 那段注释：
    # 测试球尺寸跨度大，猜一个数会让 Z 按 1:1 比例错。
    ck("球径留空 → 不传 --ball-diameter（不再兜底成有效直径）",
       diameter_arg("") == "" and diameter_arg("40") == "0.04",
       f"空→{diameter_arg('')!r}，填 40→{diameter_arg('40')!r}")
    ck("球径只有空白 → 同样不传",
       diameter_arg("   ") == "", repr(diameter_arg("   ")))
    ck("留空必须能说清后果（不是静默）",
       bool(diameter_problem("")) and "Z" in diameter_problem(""), repr(diameter_problem("")))
    ck("球径填了就用填的，且换算成米（mm ÷ 1000）",
       diameter_arg(" 30 ") == "0.03", repr(diameter_arg(" 30 ")))
    ck("反向：界面填 21.6（mm）→ 命令行 0.0216（米），不许出现 21.6",
       diameter_arg("21.6") == "0.0216" and float(diameter_arg("21.6")) < 1,
       f"{diameter_arg('21.6')!r}")
    return out



def _ball_settings_checks() -> list[tuple[str, bool]]:
    """球种类与检测参数（2026-11-03 增）：**不需要 Tk** 的那一半。

    为什么必须有这一节
    ------------------
    "球种类"不是一个装饰性的下拉 —— 它到 ``--method`` / ``--preset`` 之间
    是一条**联动**，而联动的失效方式全是静默的：选「纯色塑料球 + 红」而命令行
    只有 ``--method color``（预设丢了 → 默认按白球分割 → 红球 0 检出）、
    选「镜面金属球」却还带着 ``--preset red_ball``（把高光定位换成颜色门限）。
    这些在界面上一个字都看不出来，只能比对命令行。

    判据全部打在**产出 argv 的那一层**（``BallSettings.resolve`` +
    ``_actions``），而不是"源码里有没有某个字面量" —— 后者一改写法就失效。
    """
    from apps._actions import offline_stereo, run_realtime, run_video
    from apps._gui_stereo import detector_methods, preset_choices
    from apps._gui_widgets import (BALL_CUSTOM, BALL_METAL, BALL_PLASTIC,
                                   BallSettings, detector_needs_note,
                                   method_for_kind, preset_label, preset_options)
    from sphere_tracker.config import ColorConfig
    from sphere_tracker.plugins import names as plugin_names

    out: list[tuple[str, bool]] = []

    def ck(label: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'通过' if ok else '不达标'}] {label}"
              + (f"（{detail}）" if detail else ""))
        out.append((label, ok))

    print("\n----- 球种类 → 检测参数（BallSettings → argv）-----")

    # ---- 三档的定义值 ----
    ck("球种类的三档：镜面金属球 / 纯色塑料球 / 自定义",
       (method_for_kind(BALL_METAL), method_for_kind(BALL_PLASTIC),
        method_for_kind(BALL_CUSTOM)) == ("specular", "color", ""),
       f"{method_for_kind(BALL_METAL)} / {method_for_kind(BALL_PLASTIC)} / "
       f"{method_for_kind(BALL_CUSTOM)!r}")

    # ---- ① 纯色塑料球 + 红：--method color 且带对应 --preset ----
    # 预设名取自注册表（`ColorConfig.preset_names()`），不是测试里手写的"red"
    presets = list(preset_choices())
    red = next((p for p in presets if p.startswith("red")), "")
    m, p = BallSettings.resolve(BALL_PLASTIC, "", red)
    argv = run_video("clip.mkv", diameter_m="0.0216", method=m, preset=p)
    ck("纯色塑料球 + 红 → --method color --preset red_ball",
       red == "red_ball" and _has_seq(argv, ["--method", "color"])
       and _has_seq(argv, ["--preset", "red_ball"]), " ".join(argv))
    rt = run_realtime(diameter_m="0.0216", method=m, preset=p)
    ck("同一档在实时入口也带 --preset（两个入口同一套规则）",
       _has_seq(rt, ["--method", "color", "--preset", "red_ball"]), " ".join(rt))

    # ---- ② 镜面金属球：--method specular，且**不许**带 --preset ----
    m, p = BallSettings.resolve(BALL_METAL, "hough", red)
    argv = run_video("clip.mkv", diameter_m="0.0216", method=m, preset=p)
    ck("镜面金属球 → --method specular（即便别处填了别的方法）",
       m == "specular" and _has_seq(argv, ["--method", "specular"]), " ".join(argv))
    ck("镜面金属球不传 --preset（那是颜色分割的参数，传了等于换了个检测器）",
       p == "" and "--preset" not in argv, " ".join(argv))
    ck("自定义 + 真选了颜色 → --preset 仍然传（只有金属球才禁用它）",
       _has_seq(run_video("c.mkv", diameter_m="0.0216", method="hough",
                          preset=red), ["--preset", "red_ball"]))
    ck("自定义档里手动落到 specular → 同样不传 --preset（这条跟方法走，不跟档位走）",
       BallSettings.resolve("自定义", "specular", red)[1] == "")
    ck("预设选「不使用」= 不传 --preset（与经典界面同一个哨兵词）",
       BallSettings.resolve(BALL_PLASTIC, "", "不使用")[1] == "")

    # ---- ③ 自定义：下拉项 == 注册表（不是界面里手抄的清单）----
    methods = detector_methods()
    ck("自定义档的方法候选 == 注册表里的全部检测器（R7：不许手写名字）",
       bool(methods) and set(methods) == set(plugin_names("detector")),
       f"{list(methods)}")
    ck("颜色预设候选 == ColorConfig.preset_names()（同理：不许手抄颜色清单）",
       list(preset_choices()) == ColorConfig.preset_names(),
       f"{list(preset_choices())}")

    # ---- 预设**显示**：中文标签从预设表推，不是手写的"红=red_ball"字典 ----
    # 判据认的是"能不能从内部名推出来"，不是某几个字长什么样：
    # 表里加一种球色，这里自动跟着多一行；而写死字典会立刻露馅。
    labels = preset_options(preset_choices())
    ck("预设下拉显示中文（red_ball → 红球），且与预设表一一对应",
       labels.get("红球") == "red_ball" and labels.get("绿球") == "green_ball"
       and set(labels.values()) == set(preset_choices()),
       f"{labels}")
    ck("没见过的预设名照样显示（退回英文词干，不把它从下拉里藏掉）",
       preset_label("magenta_ball") == "magenta"
       and preset_label("不使用") == "不使用")
    # 首字母撞名：合并会让其中一项永远选不到，而界面看不出来
    clash = preset_options(["red_ball", "rose_ball"])
    ck("首字母撞名的预设不许互相覆盖（退回内部名，两项都还在）",
       len(clash) == 2 and set(clash.values()) == {"red_ball", "rose_ball"},
       f"{clash}")

    # ---- 可选依赖：注册表声明的 needs 要提前说，别跑到一半炸在 import ----
    note = detector_needs_note("yolo")
    ck("yolo 的依赖提示：没装就点名 ultralytics（装了则为空）",
       note == "" or ("ultralytics" in note and "pip install" in note),
       note or "（本机已装 ultralytics，提示为空）")
    ck("没有 needs 的方法不产生提示（color / specular 不该被这份提示污染）",
       detector_needs_note("color") == "" and detector_needs_note("specular") == "")

    # ---- 参数按窗口裁剪：脚本没有的开关，界面的 argv 里就不许出现 ----
    # run_video 的 argparse 里**没有** --min-radius/--max-radius；
    # run_stereo_video 的 argparse 里**没有** --ball-diameter。
    ck("单目 argv 不含半径参数（run_video 没这两个开关，传了必被 argparse 拒）",
       not any(x in run_video("c.mkv", diameter_m="0.0216", method="color")
               for x in ("--min-radius", "--max-radius")))
    ck("双目 argv 不含 --ball-diameter（run_stereo_video 没这个开关）",
       "--ball-diameter" not in offline_stereo("s", method="color"))

    # ---- 半径：非法值**不许**混进命令行 ----
    ok_r = _has_seq(offline_stereo("s", min_radius="12", max_radius="90"),
                    ["--min-radius", "12", "--max-radius", "90"])
    bad_r = any(x in offline_stereo("s", min_radius="-3", max_radius="abc")
                for x in ("--min-radius", "--max-radius"))
    ck("半径范围：正数照传；负数/乱码一项都不传（不静默变成 0）", ok_r and not bad_r,
       " ".join(offline_stereo("s", min_radius="12", max_radius="90")))

    # ---- 工作台：第③步的 argv 真的跟着球种类走（不是写死 specular） ----
    import tempfile
    from types import SimpleNamespace

    from apps import _actions
    from apps._gui_bench import offline_cmd

    with tempfile.TemporaryDirectory(prefix="bench_ball_") as tmp:
        sess = Path(tmp) / "20260104-000000"
        (sess / "derived").mkdir(parents=True)

        plain = SimpleNamespace(path=sess, has_seeds=False)
        argv = offline_cmd(plain, method="color", preset=red)
        ck("工作台：选纯色球+红 → 命令行 --method color --preset red_ball",
           _has_seq(argv, ["--method", "color", "--preset", "red_ball"])
           and "--stereo-mode" not in argv, " ".join(argv))

        # 有 seeds.json 时必须 specular：--seed-json 只有它读
        (sess / "derived" / "seeds.json").write_text("{}", encoding="utf-8")
        seeded = SimpleNamespace(path=sess, has_seeds=True)
        argv = offline_cmd(seeded, method="color")
        ck("工作台：会话里有点选球心 → 强制 specular + --seed-json（种子不被静默丢掉）",
           _has_seq(argv, ["--method", "specular"])
           and _has_seq(argv, ["--seed-json", str(sess / "derived" / "seeds.json")]),
           " ".join(argv))

        # 行为没变的证据（搬迁不是顺手改命令）：老写法是
        # `offline_stereo(path)` 之后在 GUI 里手工追加 `--method/--seed-json`；
        # 新写法把这两段交给 `_actions`。两条 argv 必须**逐字相等**（连顺序）。
        old_seeded = _actions.offline_stereo(sess) + [
            "--method", "specular",
            "--seed-json", str(sess / "derived" / "seeds.json")]
        old_plain = _actions.offline_stereo(sess)
        ck("工作台：有 seeds 时 argv 与改动前逐字相同（顺序也一样）",
           old_seeded == offline_cmd(seeded), " ".join(old_seeded[-4:]))
        ck("工作台：无 seeds 且不传参数时，argv 与改动前逐字相同",
           old_plain == offline_cmd(plain), " ".join(old_plain))

    # ---- 工作台：--method 不许再写死在函数里 ----
    # 这一条守的是"第 167 行写死 specular"那个具体缺陷：写死之后球种类那一栏
    # 怎么改都改不动命令行，而界面看起来一切正常。
    #
    # 用 AST 找**完整的字符串常量**，不做全文子串匹配：docstring 里解释这个缺陷
    # 时会写到 `--method`，子串匹配会把"解释"当成"又拼了一遍"（假红会让下一个人
    # 删掉注释保绿，那才是真损失）。
    import ast

    bench_tree = ast.parse((ROOT / "apps" / "_gui_bench.py").read_text(encoding="utf-8"))
    inline = sorted({n.value for n in ast.walk(bench_tree)
                     if isinstance(n, ast.Constant)
                     and n.value in ("--method", "--seed-json", "--preset")})
    ck("工作台不再自己拼 --method / --seed-json（跟随球种类，而不是写死）",
       not inline, f"仍在字面量里：{inline}" if inline else "")
    return out


def _argv_single_source_checks() -> list[tuple[str, bool]]:
    """argv 只能有一个产地（`apps/_actions.py`）—— 用 AST 扫，不看人。

    判据故意是"**零字面量**"而不是"有没有调 `_actions`"：后者可以一边调
    一边又内联一份（那正是从前那种漂移）。只要界面文件里再出现 ``"xxx.py"``，
    就说明有人在手工拼命令行，紧接着就是"同一个动作两份参数清单"。

    经典界面（``启动程序.py``）是**已知豁免**：它有 22 处内联，一次改完风险大。
    但豁免是**棘轮**——只许减少、不许增加，否则它会悄悄长出新的内联 argv。
    """
    import ast
    from pathlib import Path as _P

    root = _P(__file__).resolve().parent.parent
    legacy_budget = 22

    def py_literals(path):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return [n.lineno for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)
                and n.value.endswith(".py")]

    out: list[tuple[str, bool]] = []
    print("\n----- argv 只有一个产地（apps/_actions.py）-----")
    for f in sorted((root / "apps").glob("_gui_*.py")):
        hits = py_literals(f)
        ok = not hits
        print(f"  [{'通过' if ok else '不达标'}] {f.name}: 脚本字面量 {len(hits)} 处"
              + ("" if ok else f"（行 {hits[:6]}）"))
        out.append((f"{f.name} 不许手写脚本路径", ok))

    legacy = py_literals(root / "启动程序.py")
    ok = len(legacy) <= legacy_budget
    print(f"  [{'通过' if ok else '不达标'}] 经典界面内联数 {len(legacy)} ≤ {legacy_budget}"
          "（棘轮：只许减）")
    out.append(("经典界面内联 argv 只减不增", ok))

    # 动作模块本身必须真的承载这些脚本（否则"零字面量"可以靠删功能达成）
    acts = (root / "apps" / "_actions.py").read_text(encoding="utf-8")
    need = ["record_stereo.py", "calibrate_stereo.py", "run_stereo_video.py",
            "check_stereo_setup.py", "record_mono.py", "run_video.py",
            "run_realtime.py", "view_3d.py", "cameras.py", "capture_board.py",
            "check_dataset.py", "export_visualization.py", "data_manager.py",
            "calibrate_mono.py", "calibrate_focal.py"]
    missing = [n for n in need if n not in acts]
    print(f"  [{'通过' if not missing else '不达标'}] 动作模块覆盖 {len(need) - len(missing)}/{len(need)} 个脚本"
          + ("" if not missing else f"（缺 {missing}）"))
    out.append(("_actions 覆盖全部界面动作", not missing))
    return out


def main() -> int:
    print("=" * 74)
    print("GUI 预算自测：分窗后的每一步都必须留在一屏之内")
    print("=" * 74)

    # 参数拼装判据先跑：它直接调纯函数、不需要 Tk，所以**没有显示器也要跑**。
    # 结构判据可以跳过，参数拼装不行 —— 跳过的那部分一律不算验证过。
    results: list[tuple[str, bool]] = _stereo_param_checks()
    results += _ball_settings_checks()
    results += _argv_single_source_checks()
    results += _mono_session_checks()
    results += _mono_seed_checks()

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
                l2_max = (ANGLED_L2_CLICKABLE_MAX
                          if name == "stereo(angled)" else L2_CLICKABLE_MAX)
                ok_l2 = m["l2_clickable"] <= l2_max
                ok_primary = m["primary"] <= 1
                print(f"  {title[:26]:28s} L1 输入 {m['l1_inputs']}/{L1_INPUTS_MAX}"
                      f"  按钮 {m['l1_buttons']}/{L1_BUTTONS_MAX}"
                      f"  L2 可点 {m['l2_clickable']}/{l2_max}"
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

            # 第②步（标定）状态栏不得用"现有标定模式"去拒绝"即将标定"（2026-11-03）。
            # 标定是产出动作：大角度窗选 joint/chain 都是合法的，即使当前 active 标定
            # 还是上次的 parallel，也不该写"请用平行窗口"（那套判定属于第③录制/第④分析）。
            if name.startswith("stereo"):
                lbl = getattr(win, "lbl_calib", None)
                if lbl is not None:
                    try:
                        calib_txt = str(lbl.cget("text"))
                    except Exception:                       # noqa: BLE001
                        calib_txt = ""
                    ok = ("请用" not in calib_txt
                          and "请用平行" not in calib_txt
                          and "请用「平行双目」窗口" not in calib_txt)
                    print(f"  标定状态栏不误导 = {'通过' if ok else '不达标'}"
                          f"（{calib_txt[:40]}）")
                    results.append((f"{name} 第②步标定不误判模式", ok))

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
        # 2026-10-31 再补一条**单位**：界面按 mm 收，命令行按米收。期望值因此
        # 是 0.0216 而不是 21.6 —— 传 21.6 会让 run_video 一个三维点都不给。
        # 为什么断言的是**命令行实际拿到什么**、而不是源码里有没有某个字面量：
        # 后者一改写法（抽函数、换常量）就失效，且抓不到"留空走另一条分支"。
        # 这一条可以反向自证：把 ``diameter_arg`` 里的 ``/ 1000`` 去掉，这里立刻红。
        from apps._gui_mono import EFFECTIVE_DIAMETER_MM, LEGACY_DIAMETER

        want_m = EFFECTIVE_DIAMETER_MM / 1000.0

        mono = dict(windows)["mono"]
        captured: list[list[str]] = []

        # ---------- 单目窗：「点选球心」在没会话时必须是灰的（2026-10-30 增） ----------
        # 一进窗口就没有当前会话，此时可点只会弹一句"先录一段" ——
        # 可点却干不了事的按钮比灰着的按钮更伤人（与「看 3D」同一条道理）。
        seed_texts = " ".join(collect_texts(mono))
        seed_off = "disabled" in mono.btn_seed.state()
        ok_seed_ui = ("点选球心" in seed_texts) and seed_off
        print(f"  [{'通过' if ok_seed_ui else '不达标'}] 单目窗有「点选球心」，"
              f"没会话时禁用（state={mono.btn_seed.state()}）")
        results.append(("点选球心无会话时禁用", ok_seed_ui))

        class _StubTasks:
            """只收集命令行，不真起进程（否则测试会去开相机）。"""

            def start(self, cmd, *_a, **_k):
                captured.append(list(cmd))

        real_tasks = mono.app.tasks
        mono.app.tasks = _StubTasks()
        try:
            # 2026-11-03：球径**没有默认值**了。留空必须**拦住**，
            # 而不是拿 21.6 mm 顶上 —— 测试球尺寸跨度大，猜一个数会让 Z 按比例错。
            captured.clear()
            mono.var_diameter.set("")
            mono._start_realtime()
            note = mono.status._label.cget("text")
            ok_diam = (not captured) and ("球径" in note) and ("Z" in note)
            print(f"  [{'通过' if ok_diam else '不达标'}] 单目窗球径留空 → 不起任务且说清为什么"
                  f"（命令 {len(captured)} 条，状态条尾部：…{note[-34:]}）")
            results.append(("球径留空先拦住", ok_diam))
        finally:
            mono.app.tasks = real_tasks

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
            mono.var_diameter.set("21.6")   # 2026-11-03：球径无默认值，必须先填才能起任务
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
            ok_est = got_d == f"{want_m:g}"
            print(f"  [{'通过' if ok_est else '不达标'}] 单点焦距估计：留空 → 传 "
                  f"--ball-diameter {got_d or '(缺失)'}（应为 "
                  f"{want_m:g} 米 = {EFFECTIVE_DIAMETER_MM:g} mm）")
            results.append(("焦距估计传球径", ok_est))

            # 换了球：框里写 30（mm），命令里就必须是 0.03（米）——
            # f 与 D 同比例，错一个都白标；单位错了更是差三个数量级。
            captured.clear()
            mono.var_diameter.set("30")
            mono._start_focal_estimate()
            cmd = captured[0] if captured else []
            got_d30 = (cmd[cmd.index("--ball-diameter") + 1]
                       if "--ball-diameter" in cmd else "")
            ok_est30 = got_d30 == "0.03"
            print(f"  [{'通过' if ok_est30 else '不达标'}] 单点焦距估计：框里填 30 → "
                  f"传 --ball-diameter {got_d30 or '(缺失)'}（应为 0.03 米）")
            results.append(("焦距估计用框里的球径", ok_est30))

            # ---- 反向断言：命令行球径**永远是米**，必须 < 1 ----
            # 界面那一栏写的是 21.6（mm），所以"传下去的值"一定是 0.0216。
            # 谁把 ``diameter_arg`` 里的 ``/1000`` 去掉，这一条立刻红；
            # 而 21.6 米的球会让 run_video 一个三维点都不给（实测 40 帧全丢），
            # 界面上却只写"已完成"—— 那正是最坏的一类静默失效。
            captured.clear()
            mono.var_diameter.set("21.6")
            mono._start_video()
            cmd = captured[0] if captured else []
            got_216 = (cmd[cmd.index("--ball-diameter") + 1]
                       if "--ball-diameter" in cmd else "")
            try:
                as_m = float(got_216)
            except ValueError:
                as_m = 999.0
            ok_unit = 0 < as_m < 1 and abs(as_m - 0.0216) < 1e-9
            print(f"  [{'通过' if ok_unit else '不达标'}] 界面填 21.6 mm → 命令行 "
                  f"--ball-diameter {got_216 or '(缺失)'}（必须是米且 <1，"
                  f"出现 21.6 这种数就是单位断了）")
            results.append(("单目窗球径永远是米（<1）", ok_unit))
        finally:
            mono_mod.find_clip = real_find
            mono._session = real_sess
            mono.var_focal.set("")
            mono.var_diameter.set("21.6")   # 2026-11-03：球径无默认值，必须先填才能起任务
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
            mono.var_diameter.set("21.6")   # 2026-11-03：球径无默认值，必须先填才能起任务
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

            # ---------- 单目窗：点了球心之后命令行带什么（2026-10-30 增） ----------
            # 守的坑：用户辛苦点完两点，命令行里没有 --seed（或 --method 还是
            # 自动那个检测器）—— 界面写着"已按种子跟踪"，实际仍在全图瞎找，
            # 那颗镜面球只有 1.4% 的帧会碰巧落在球上。
            import json
            import tempfile
            with tempfile.TemporaryDirectory(prefix="mono_seed_cmd_") as tmp:
                sess = Path(tmp) / "20260102-000000"
                (sess / "derived").mkdir(parents=True)
                clip = sess / "clip.mkv"
                clip.write_bytes(b"")               # find_clip 已被桩掉，只是要个路径
                seed = sess / "derived" / "seed.json"
                seed.write_text(json.dumps({
                    "x": 812, "y": 530, "radius": 91, "frame": 0,
                    "image_size": [1920, 1080],
                    "created": "2026-01-02 00:00:00"},
                    ensure_ascii=False), encoding="utf-8")

                class _SeedSess:
                    path = sess
                    width = 1920
                    height = 1080
                    error = ""              # 真 MonoSession 有这个字段（坏会话时非空）

                mono_mod.find_clip = lambda _p: clip
                mono._session = _SeedSess()
                # 可点性：会话里有视频时必须能点。永远灰着的按钮 = 整个功能不可达，
                # 而"灰着"在截图里看起来完全正常。
                mono._sync_seed_button()
                ok_on = "disabled" not in mono.btn_seed.state()
                print(f"  [{'通过' if ok_on else '不达标'}] 会话有视频时「点选球心」"
                      f"可点（state={mono.btn_seed.state()}）")
                results.append(("点选球心在会话就绪后可点", ok_on))
                mono.app.tasks = _StubTasks()
                mono.var_source.set(SOURCE_OFFLINE)
                mono.var_focal.set("")
                mono.var_diameter.set("21.6")   # 2026-11-03：球径无默认值，必须先填才能起任务

                captured.clear()
                mono.var_detector.set("color")      # 用户没动检测器：默认 color
                mono._start_video()
                cmd = captured[0] if captured else []
                got_seed = cmd[cmd.index("--seed") + 1] if "--seed" in cmd else ""
                got_m = cmd[cmd.index("--method") + 1] if "--method" in cmd else ""
                ok_seed = (got_seed == "812,530,91" and got_m == "specular"
                           and mono.var_detector.get() == "specular")
                print(f"  [{'通过' if ok_seed else '不达标'}] 有 seed.json → 命令行 "
                      f"--seed {got_seed or '(缺失)'} 且 --method "
                      f"{got_m or '(缺失)'}（下拉也同步成 {mono.var_detector.get()!r}）")
                results.append(("有种子时命令行带 --seed 且检测器 specular", ok_seed))

                # 反向：把文件删掉 → 不许再带 --seed。
                # 否则就是"跟踪一个不存在的位置"，而且界面不会有任何提示。
                seed.unlink()
                mono.var_detector.set("color")
                captured.clear()
                mono._start_video()
                cmd = captured[0] if captured else []
                ok_noseed = "--seed" not in cmd
                print(f"  [{'通过' if ok_noseed else '不达标'}] 没有 seed.json → "
                      f"命令行不带 --seed（实际 {' '.join(cmd) or '(无命令)'}）")
                results.append(("没有种子时不传 --seed", ok_noseed))

                # 分辨率不符的种子：要退回自动检测，而不是把旧坐标套到新画布上
                seed.write_text(json.dumps({
                    "x": 812, "y": 530, "radius": 91, "frame": 0,
                    "image_size": [1280, 720],
                    "created": "2026-01-02 00:00:00"},
                    ensure_ascii=False), encoding="utf-8")
                mono.var_detector.set("color")
                captured.clear()
                mono._start_video()
                cmd = captured[0] if captured else []
                note = mono.status._label.cget("text")
                ok_bad = "--seed" not in cmd and "对不上" in note
                print(f"  [{'通过' if ok_bad else '不达标'}] 种子分辨率不符 → 不传 "
                      f"--seed 且状态条说明（尾部：…{note[-34:]}）")
                results.append(("种子分辨率不符时退回自动检测", ok_bad))

                # ---------- 球种类与检测参数（2026-11-03 增）----------
                # 这一段守的是"界面选了什么、命令行就是什么"：
                #   ① 纯色塑料球 + 红 → --method color --preset red_ball
                #   ② 镜面金属球     → --method specular，且**不带** --preset
                #   ③ 自定义档的下拉项 == 注册表（不是界面里手抄的清单）
                #   ④ 切球种类**不动**用户填的球直径
                # ①② 抓的是 argv（界面看不错、命令行错的那种静默失效）；
                # ④ 抓的是"联动越界"——改显示不成就顺手改了参数，那比不做联动更糟。
                from apps._gui_widgets import BALL_METAL, BALL_PLASTIC
                from apps._gui_stereo import detector_methods

                def _argv_of() -> list[str]:
                    captured.clear()
                    mono._start_video()
                    got = captured[0] if captured else []
                    print(f"      实际 argv：{' '.join(got) or '(无命令)'}")
                    return got

                def _val(cmd: list[str], flag: str) -> str:
                    return cmd[cmd.index(flag) + 1] if flag in cmd else ""

                mono.var_diameter.set("30")         # 用户填的球径（mm）
                mono.var_focal.set("")
                mono.var_preset.set("red_ball")
                mono.ball.set_kind(BALL_PLASTIC)
                cmd = _argv_of()
                ok_plastic = (_val(cmd, "--method") == "color"
                              and _val(cmd, "--preset") == "red_ball"
                              and _val(cmd, "--ball-diameter") == "0.03")
                print(f"  [{'通过' if ok_plastic else '不达标'}] 界面选「纯色塑料球 + 红」"
                      f"→ --method {_val(cmd, '--method') or '(缺失)'} "
                      f"--preset {_val(cmd, '--preset') or '(缺失)'} "
                      f"--ball-diameter {_val(cmd, '--ball-diameter') or '(缺失)'}")
                results.append(("球种类：纯色球+红 → color + red_ball", ok_plastic))

                mono.ball.set_kind(BALL_METAL)
                cmd = _argv_of()
                ok_metal = (_val(cmd, "--method") == "specular"
                            and "--preset" not in cmd)
                ok_keep = (mono.var_diameter.get().strip() == "30"
                           and _val(cmd, "--ball-diameter") == "0.03")
                print(f"  [{'通过' if ok_metal else '不达标'}] 界面选「镜面金属球」"
                      f"→ --method {_val(cmd, '--method') or '(缺失)'}，"
                      f"--preset {'仍在（错）' if '--preset' in cmd else '不传（对）'}")
                print(f"  [{'通过' if ok_keep else '不达标'}] 切球种类不动球直径："
                      f"框里仍是 {mono.var_diameter.get()!r}，命令行 "
                      f"{_val(cmd, '--ball-diameter') or '(缺失)'}（mm→m 换算是 0.03）")
                results.append(("球种类：镜面球 → specular 且不带 preset", ok_metal))
                results.append(("切球种类不改动球直径", ok_keep))

                # 预设：下拉显示中文、变量里仍是内部名（argv 只认 red_ball）
                mono.ball._var_preset_label.set("红球")
                mono.ball._on_preset_selected()
                ok_lab = (mono.var_preset.get() == "red_ball"
                          and "红球" in tuple(mono.ball.cb_preset.cget("values")))
                print(f"  [{'通过' if ok_lab else '不达标'}] 预设下拉显示「红球」，"
                      f"写回变量的是 {mono.var_preset.get()!r}（argv 只认内部名）")
                results.append(("预设显示中文但下传内部名", ok_lab))

                # 参数按窗口裁剪：单目没有半径控件（run_video 没这两个开关）
                ok_scope_m = mono.ball.radius_vars is None
                print(f"  [{'通过' if ok_scope_m else '不达标'}] 单目窗不放半径"
                      f"（run_video 的 argparse 里没有 --min/max-radius）")
                results.append(("单目窗不含半径控件", ok_scope_m))

                # ③ 下拉项 == 注册表：把界面上的候选与注册表逐个比，
                # 而不是"看看有没有 color 这一项"（少一个也是错，多一个更是）。
                stereo_win = dict(windows)["stereo(parallel)"]
                shown = tuple(stereo_win.ball.cb_method.cget("values"))
                ok_reg = shown == detector_methods()
                print(f"  [{'通过' if ok_reg else '不达标'}] 双目窗「检测方法」下拉项 == "
                      f"注册表（{len(shown)} 项：{list(shown)}）")
                results.append(("自定义档的下拉项等于注册表", ok_reg))
                mono.var_diameter.set("21.6")   # 2026-11-03：球径无默认值，必须先填才能起任务
                mono.var_preset.set("不使用")
                mono.ball.set_kind(BALL_PLASTIC)

            print(f"  检测器候选 = {list(detector_choices())}"
                  f"（默认 {mono.var_detector.get()!r}）")
        finally:
            mono_mod.find_clip = real_find
            mono._session = real_sess
            mono.var_focal.set("")
            mono.var_diameter.set("21.6")   # 2026-11-03：球径无默认值，必须先填才能起任务
            mono.var_detector.set("color")
            mono.var_source.set(SOURCE_OFFLINE)
            mono._auto_retry_pending = False
            mono.app.tasks = real_tasks
            mono._refresh_session()

        # ---------- 双目窗：第④步的 argv 真的来自「球种类」（2026-11-03 增）----------
        # 与单目那三条同一件事的另一半：第④步原来一个检测参数都不传，
        # 于是"这颗是红球"在界面上无处可说；现在说在哪、就必须传到位。
        import tempfile

        from apps._gui_widgets import BALL_METAL, BALL_PLASTIC

        stereo_win = dict(windows)["stereo(parallel)"]
        captured.clear()
        stereo_win.app.tasks = _StubTasks()
        try:
            with tempfile.TemporaryDirectory(prefix="stereo_ball_") as tmp:
                sess = Path(tmp) / "20260105-000000"
                sess.mkdir()
                stereo_win.var_session.set(str(sess))
                stereo_win.ball.set_kind(BALL_PLASTIC)
                stereo_win.var_preset.set("red_ball")
                stereo_win.var_min_radius.set("12")
                stereo_win.var_max_radius.set("90")
                stereo_win._start_offline()
                cmd = captured[0] if captured else []
                ok_st = (_has_seq(cmd, ["--method", "color", "--preset", "red_ball"])
                         and _has_seq(cmd, ["--min-radius", "12", "--max-radius", "90"])
                         and _has_seq(cmd, ["--stereo-mode", "parallel"]))
                print(f"  [{'通过' if ok_st else '不达标'}] 双目第④步（纯色球+红）→ "
                      f"{' '.join(cmd[1:]) or '(无命令)'}")
                results.append(("双目窗球种类进命令行", ok_st))

                # 切球种类**只许改方法**：半径范围是用户填的，不许跟着动
                stereo_win.ball.set_kind(BALL_METAL)
                ok_keep_r = (stereo_win.var_min_radius.get() == "12"
                             and stereo_win.var_max_radius.get() == "90")
                print(f"  [{'通过' if ok_keep_r else '不达标'}] 切球种类不动半径范围："
                      f"仍是 {stereo_win.var_min_radius.get()!r}–"
                      f"{stereo_win.var_max_radius.get()!r}")
                results.append(("切球种类不改动半径范围", ok_keep_r))
                # 双目没有球直径控件（run_stereo_video 的 argparse 里没有这个开关）
                ok_scope_s = stereo_win.ball.var_diameter is None
                print(f"  [{'通过' if ok_scope_s else '不达标'}] 双目窗不放球直径"
                      f"（run_stereo_video 没有 --ball-diameter）")
                results.append(("双目窗不含球直径控件", ok_scope_s))

                # 镜面球没有点选球心：拦住并说清，别让人等一次进程启动再看 0 配对
                captured.clear()
                stereo_win._start_offline()
                note = stereo_win.status._label.cget("text")
                ok_blk = not captured and "点选球心" in note
                print(f"  [{'通过' if ok_blk else '不达标'}] 双目镜面球没点选球心 → "
                      f"不起进程，状态条说明（尾部：…{note[-26:]}）")
                results.append(("双目镜面球缺种子时先拦住", ok_blk))
        finally:
            stereo_win.app.tasks = real_tasks
            stereo_win.var_session.set("")
            stereo_win.var_preset.set("不使用")
            stereo_win.var_min_radius.set("")
            stereo_win.var_max_radius.set("")
            stereo_win.ball.set_kind(BALL_PLASTIC)

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
