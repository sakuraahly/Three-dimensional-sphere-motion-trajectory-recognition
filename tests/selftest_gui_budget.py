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
7. **实时会话的可见性**（2026-11-17 增）：实时双目落的会话要在数据管理里
   看得出是实时（`source` 读进 `SessionItem` + 名称前缀），且第④步
   「跑离线三维」必须**先拦**（它没有原始录像，起子进程只会 validate 失败）。
   拦截的接线用 AST 断言"在全部 `tasks.start(` 之前"。同样**不依赖 Tk**。

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
# 12 → 13（2026-11-08）：第④步「球种类与检测参数」新增「尺寸门自动阶梯」勾选框，
# 是**一个**真实且必要的新增控件（决定要不要下传 --no-radius-auto），不是窗口长回去。
# 13 → 14（2026-11-14，team-lead 最终裁决）：用户点名要求新增「球直径（mm）」
# 物理输入（接 --anchor-diameter-mm）+ 承载它的「高级」折叠区 header
# （CollapsibleSection 的 header_button 也是 TButton、计入 L2）；同时按用户
# 裁决**移除**「尺寸门自动阶梯」勾选框（阶梯改为默认且唯一路径，GUI 不再
# 下传 --no-radius-auto），故 ④ 卡片净增只有 1（15 → 14）。其余窗口仍守
# 这个上限；L1_INPUTS_MAX / L1_BUTTONS_MAX 一律不动。
L2_CLICKABLE_MAX = 14
# 大角度（angled）第②步：链式标定比 joint 多「标定方法下拉 + 采集标定板图 +
# 检查链式输入」三个控件，是**真实且必要**的内容差异，不是"长回一个窗口装全部"。
# 单独给它放宽 L2 上限，其余窗口仍守 L2_CLICKABLE_MAX。
# 2026-11-04 再加「FOV 对齐」勾选（用户报"FOV 影响平行标定"，要能关掉）——
# 它与「放宽模式」同为改变标定口径的进阶开关，落在高级折叠区，仍算 L2 可点。
# 14 → 15：也是真实且必要的新增，不是结构长回去。
# 15 → 16（2026-11-09）：接 §67 A1 老缺陷 —— 大角度窗「高级」里加一个
# 「卷尺量两镜头中心距 mm（可留空）」输入框，把 --baseline-mm 真正下传。
# 零共视链路（双板链式）没有跨路一致性证据，这个卷尺值是**唯一独立于图像
# 之外的物理核对锚**，不是可省的装饰。它接在 line3 同一行、只在 angled 窗
# 出现、折叠区展开高度不变（selftest_window_mode 已验证），平行窗不受影响。
# 16 → 17（2026-11-11）：大角度 ② 标定卡的 action_row 加「重拍某一步…」
# 一个按钮 —— 让用户只重拍出错的那一步，不必四步全重来。它挂在既有按钮行
# （不新增一行，selftest_window_mode 已验证窗口高度不变），是 16→17 的
# 唯一来源；L1 按钮 3→4 仍在 ≤5 内。
# ⚠️ 前四次放宽（12→13→15→16→17）都是真实功能新增，但**连续放宽会削弱这道闸**；
#    **后续待办**：把「高级」折叠区的控件单独计一档，而不是每次都抬高整窗上限。
#    这条警示不因本次放宽而移除。
# 17 → 18（2026-11-14）：**第五次放宽**，也是唯一一次「按用户明确要求」——
# 用户原话「标定不要那么限制严苛，要增加 GUI 界面的强制通过指令」。② 标定卡
# **复用 `:854` 已有的高级折叠区**新增一个 secondary「强制通过标定（仅链路
# 测试）…」按钮（不新增 header），是真实功能新增（带二次确认 + 质量状态
# 记录），不是 UI 膨胀；主操作仍恰好 1、window_mode 高度不变。② 卡片其余
# 可点控件全是必要项（采集/重拍/检查链式输入/放宽/FOV/卷尺/相机名），
# **无法在不删真实功能、不改 measure() 计数、不钻控件类型空子**的前提下
# 零净增消化，故只把这一个上限从 17 调到 18。通用 L2_CLICKABLE_MAX 另因
# ④ 卡片净增 1 从 13 调到 14（见上），L1_INPUTS_MAX / L1_BUTTONS_MAX 不动。
# ✅ 已按 team-lead 2026-11-14 最终裁决登记（一次做到全绿）。
ANGLED_L2_CLICKABLE_MAX = 18


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
# 老窗口（`apps/gui_main.py`）里同一件事有 7 节判据守着（见
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

    # ---- FOV 开关（2026-11-04）：默认不动老行为，显式关才 skip ----
    # 用户报"FOV 有点影响平行双目标定了"，第②步加了勾选（默认关）。
    #
    # 旧判据是"--skip-fov 不许出现在任何普通分支"：那时界面没有开关，
    # 谁把 flag 塞进去就等于静默改裁剪口径。现在它变成**用户显式选择**，
    # 判据跟着改成三件事：默认不出现 / 关掉必须落命令行 / 关掉不能同时
    # 带 --fov-from-profile（后端 1055 行判互斥，会直接 argparse 报错）。
    p_default = calibrate_args("joint", {}, "", "", "parallel")
    p_off = calibrate_args("joint", {}, "", "", "parallel", use_fov=False)
    p_off_prof = calibrate_args("joint", {}, "", "", "parallel",
                                fov_from_profile="rig", use_fov=False)
    p_on_prof = calibrate_args("joint", {}, "", "", "parallel",
                               fov_from_profile="rig")
    check("默认（不传 use_fov）不出现 --skip-fov（老行为不变）",
          "--skip-fov" not in p_default, " ".join(p_default))
    check("界面关掉 FOV → 命令行真的带 --skip-fov",
          "--skip-fov" in p_off, " ".join(p_off))
    check("关掉时不再带 --fov-from-profile（后端会判互斥）",
          "--skip-fov" in p_off_prof and "--fov-from-profile" not in p_off_prof,
          " ".join(p_off_prof))
    check("打开且档案有名字 → 带 --fov-from-profile、不带 skip",
          "--fov-from-profile" in p_on_prof
          and "--skip-fov" not in p_on_prof, " ".join(p_on_prof))
    chain_off = calibrate_args("chain_dual", {}, "", "", "angled", use_fov=False)
    check("链式（none_raw）不下传 --skip-fov（它本来就不做 FOV）",
          "--skip-fov" not in chain_off, " ".join(chain_off))

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

    # 种子里记的球种类：界面据此决定"点选后仍用 color 还是切 specular"。
    # 旧文件没有该字段 → 空串（读取方按 metal 兼容），坏 json 不许抛。
    import json as _json
    from apps._actions import seed_ball_kind
    with tempfile.TemporaryDirectory() as td:
        sess = Path(td)
        (sess / "derived").mkdir()
        seed = sess / "derived" / "seeds.json"
        seed.write_text(_json.dumps({"ball_kind": "plastic"}), encoding="utf-8")
        k_plastic = seed_ball_kind(sess)
        seed.write_text(_json.dumps({"ball_kind": "metal"}), encoding="utf-8")
        k_metal = seed_ball_kind(sess)
        seed.write_text("{}", encoding="utf-8")
        k_old = seed_ball_kind(sess)
        seed.write_text("{坏 json", encoding="utf-8")
        k_bad = seed_ball_kind(sess)
    check("seed_ball_kind：plastic/metal 原样读，旧文件空串，坏 json 不抛",
          (k_plastic, k_metal, k_old, k_bad) == ("plastic", "metal", "", ""),
          f"{k_plastic!r}/{k_metal!r}/{k_old!r}/{k_bad!r}")

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
    check("不传 out 时不带 --out（脚本默认跟随激活装置档案的标定路径，不再写死顶层）",
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

        # ---- 基线核对 / FOV 复用（2026-11-04，2026-11-07 改口径）----
        # 2026-11-07：装置档案不再存"期望基线"，这里验的是**显式传进来的
        # 单次卷尺测量**（CLI / 经典界面的标定输入框）。它仍是 chain_dual
        # 物理护栏的开关：不传等于后端那段核对不执行，且不报错、静默少一项证据。
        cb = calibrate_args("chain_dual", {}, "", "", "angled", baseline=124.0)
        check("chain_dual：显式给的基线锚进 --baseline-mm",
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
    # 录制格式（2026-11-20）：老界面有 mkv/avi 下拉，新界面此前恒录 mkv。
    # 只给两档，mp4 刻意不进来（无损编码进 mp4 会被 FFmpeg 静默换有损）。
    r_avi = record_args(left="left", right="right", duration="10",
                        out=str(out_dir), container="avi")
    check("录制：格式选 avi → 命令行带 --container avi（老界面能力补回）",
          _has_seq(r_avi, ["--container", "avi"]), " ".join(r_avi))
    check("录制：默认不传 --container（空 = 跟随后端默认 mkv）",
          "--container" not in r, " ".join(r))
    _gui_src = (ROOT / "apps" / "_gui_stereo.py").read_text(encoding="utf-8")
    check("③ 录制卡有 mkv/avi 格式下拉（没有 mp4）",
          "var_container" in _gui_src
          and 'values=("mkv", "avi")' in _gui_src
          and 'values=("mkv", "avi", "mp4")' not in _gui_src)
    check("_start_record 把 var_container 传给 record_args",
          "container=self.var_container.get()" in _gui_src)
    check("录制：带 --left left --right right --duration 10 --out <目录>",
          _has_seq(r, ["--left", "left", "--right", "right"])
          and _has_seq(r, ["--duration", "10"])
          and _has_seq(r, ["--out", str(out_dir)]), " ".join(r))
    r2 = record_args(left="", right="", duration="", out="x")
    check("录制：左右与时长留空 = left/right/10（不传空串）",
          _has_seq(r2, ["--left", "left"])
          and _has_seq(r2, ["--duration", "10"]), " ".join(r2))
    # 自由采集档（2026-11-21）：空 = 不传；填了才下传。非法值绝不拼进
    # argparse（"abc" 会让后端 exit 2，用户看到的是参数错误而不是输入错误）。
    r_free = record_args(left="left", right="right", duration="10",
                         out=str(out_dir), width="1920", height="1080",
                         fps="29.97")
    check("录制：自由宽高/fps 真的下传（29.97 不截断）",
          _has_seq(r_free, ["--width", "1920"])
          and _has_seq(r_free, ["--height", "1080"])
          and _has_seq(r_free, ["--fps", "29.97"]), " ".join(r_free))
    r_bad = record_args(left="left", right="right", duration="10",
                        out=str(out_dir), width="", height="abc", fps="0")
    check("录制：空/非法/0 的采集档不传（交给后端按标定决定）",
          "--width" not in r_bad and "--height" not in r_bad
          and "--fps" not in r_bad, " ".join(r_bad))
    c_free = calibrate_args("joint", {}, "", "", "parallel",
                            width="1280", height="720", fps="60")
    check("标定 joint：自由宽高/fps 下传",
          _has_seq(c_free, ["--width", "1280"])
          and _has_seq(c_free, ["--height", "720"])
          and _has_seq(c_free, ["--fps", "60"]), " ".join(c_free))
    c_chain_free = calibrate_args("chain", {}, "", "", "angled",
                                  width="1280", height="720", fps="60")
    check("标定 chain：不传采集档（读图片目录，不是开相机）",
          "--width" not in c_chain_free and "--height" not in c_chain_free
          and "--fps" not in c_chain_free, " ".join(c_chain_free))
    check("_start_record 把三个采集档输入传给 record_args",
          "width=self.var_cap_width.get()" in _gui_src
          and "height=self.var_cap_height.get()" in _gui_src
          and "fps=self.var_cap_fps.get()" in _gui_src)
    check("_start_calibrate 把三个采集档输入传给 calibrate_args",
          "width=self.var_cap_width.get()" in _gui_src
          and "height=self.var_cap_height.get()" in _gui_src
          and "fps=self.var_cap_fps.get()" in _gui_src)
    # 注意：这里只看"是否走 _actions"，**不要**用 `"cameras.py" not in 源码`
    # 判"无字面量" —— 注释/docstring 里提到文件名是允许的，AST 判据在
    # `_argv_single_source_checks` 里，那里才是"界面不许有脚本字面量"的守卫。
    check("「使用相机实际模式」走 _actions.camera_modes",
          "_actions.camera_modes(" in _gui_src)
    from apps._actions import camera_modes as _camera_modes
    cm = _camera_modes(left="left", right="right")
    check("实际模式探测：cameras.py modes --json，按角色解析",
          any("cameras.py" in x for x in cm) and "modes" in cm
          and "--json" in cm and _has_seq(cm, ["--left", "left", "--right", "right"]),
          " ".join(cm))
    cm_idx = _camera_modes(indices="1,0")
    check("实际模式探测：显式采集编号走 --indices（不再传角色）",
          _has_seq(cm_idx, ["--indices", "1,0"]) and "--left" not in cm_idx,
          " ".join(cm_idx))
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


def _regression_2026_11_06_static() -> list[tuple[str, bool]]:
    """2026-11-06 GUI 走查遗留项：**不依赖 Tk** 的那一半回归判据。

    ① ③「开始录制」的禁用与起录判据同源：`_start_record` /
       `_sync_record_enabled` 都调 `_record_gate`，不各写一份
       parallel_status/angled_status 分支；
    ② 「试开这台相机」的 argv 只给 `--indices N`；`camera_quality` 的
       `resolve_targets` 在只给 indices 时确实走 indices 分支（带
       left/right 时优先、indices 被忽略）。

    成败**不按退出码**判 —— 该脚本打不开相机也 return 0，这里只验参数
    形态与目标解析，不碰硬件、不把「退出码 0」当「能出图」的证据。
    """
    import argparse
    import sys

    from apps import _actions
    from apps.tools import camera_quality as cq

    out: list[tuple[str, bool]] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'通过' if ok else '不达标'}] {label}")
        if not ok and detail:
            print(f"      实际：{detail}")
        out.append((label, ok))

    print("\n----- 2026-11-06 遗留项回归（源码 / argv，不依赖 Tk）-----")

    # ---- 判据同源：status 分支只许在 _record_gate 一份 ----
    src = (ROOT / "apps" / "_gui_stereo.py").read_text(encoding="utf-8")

    def body(name: str) -> str:
        head = src.index(f"    def {name}(")
        tail = src.find("\n    def ", head + 1)
        return src[head:tail if tail != -1 else len(src)]

    rec = body("_start_record")
    sync = body("_sync_record_enabled")
    gate = body("_record_gate")
    ok_same = ("_record_gate(" in rec and "_record_gate(" in sync
               and "parallel_status" not in rec and "angled_status" not in rec
               and "parallel_status" not in sync and "angled_status" not in sync
               and "parallel_status" in gate and "angled_status" in gate)
    check("B-181 判据同源：_start_record / _sync_record_enabled 都调"
          " _record_gate，status 分支只在 _record_gate 一份",
          ok_same,
          f"rec含status={('parallel_status' in rec or 'angled_status' in rec)},"
          f" sync含status={('parallel_status' in sync or 'angled_status' in sync)}")

    # ---- 试开 argv：只含 --indices N ----
    argv = _actions.camera_probe("0")
    scripts = [a for a in argv if a.endswith("camera_quality.py")]
    rest = argv[2:]
    ok_argv = (len(argv) >= 2 and argv[0] == sys.executable
               and len(scripts) == 1 and rest == ["--indices", "0"]
               and "--left" not in argv and "--right" not in argv)
    check("试开 argv 只含 --indices 0（无 --left/--right、脚本名以外无多余参数）",
          ok_argv, " ".join(argv))

    # ---- resolve_targets：只给 --indices 时走 indices 分支 ----
    # 构造 namespace 实调（left/right 留空 → 不碰相机解析），再读源码确认
    # 「left/right 优先于 indices」这个顺序仍成立（带 left/right 会忽略它）。
    targets = cq.resolve_targets(
        argparse.Namespace(left="", right="", indices="0"))
    cq_src = (ROOT / "apps" / "tools" / "camera_quality.py").read_text(
        encoding="utf-8")
    rt = cq_src[cq_src.index("def resolve_targets("):
                cq_src.index("\ndef main(")]
    ok_targets = (targets == [(0, "")]
                  and rt.index("if out:") < rt.index("if args.indices:")
                  and "return [(int(i)" in rt)
    check("camera_quality.resolve_targets：只给 --indices 走 indices 分支，"
          "带 left/right 时优先（忽略 indices）；不用退出码判成功",
          ok_targets, f"targets={targets}")

    # ---- measure() 首帧预算：显式 ≥ 30 s，别退回库默认 3.0 ----
    # 独立 Critic 证伪：底层诊断工具只等 3 s 会把慢相机（aoni 首帧 ~23 s）
    # 报成「打不开」。这里静态钉住，防止以后又被删掉、退回 3 s 把慢相机判死。
    import inspect
    import re

    measure_src = inspect.getsource(cq.measure)
    m = re.search(
        r"CameraSource\(.*?first_frame_timeout_s\s*=\s*([0-9.]+)",
        measure_src, re.S)
    timeout = float(m.group(1)) if m else 0.0
    check("camera_quality.measure 给 CameraSource 显式传 first_frame_timeout_s ≥ 30"
          "（防止以后又被删掉、退回 3 s 把慢相机判死）",
          bool(m) and timeout >= 30.0, f"first_frame_timeout_s={timeout}")
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
    # 预设名取自注册表（`ColorConfig.preset_choices()`，含 auto），不是测试里手写的"red"
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
    ck("颜色取色候选 == ColorConfig.preset_choices()（同理：不许手抄颜色清单）",
       list(preset_choices()) == ColorConfig.preset_choices(),
       f"{list(preset_choices())}")
    ck("颜色预设候选里 auto 是第一个（默认最不容易静默 0 检出的一档）",
       list(preset_choices())[:1] == ["auto"]
       and "auto" not in ColorConfig.preset_names(), f"{list(preset_choices())[:1]}")

    # ---- 预设**显示**：中文标签从预设表推，不是手写的"红=red_ball"字典 ----
    # 判据认的是"能不能从内部名推出来"，不是某几个字长什么样：
    # 表里加一种球色，这里自动跟着多一行；而写死字典会立刻露馅。
    labels = preset_options(preset_choices())
    ck("预设下拉显示中文（red_ball → 红球），且与预设表一一对应",
       labels.get("红球") == "red_ball" and labels.get("绿球") == "green_ball"
       and labels.get("自动取色") == "auto"
       and set(labels.values()) == set(preset_choices()),
       f"{labels}")
    ck("auto 的界面名字是「自动取色」而不是英文/空白",
       preset_label("auto") == "自动取色")
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

        # 2026-11-06 改：有 seeds.json **不再**强制 specular。纯色球的种子应继续
        # 交给 color（取色 + 半径锚），否则会把能跑的检测器换成跑不动的那个。
        (sess / "derived" / "seeds.json").write_text("{}", encoding="utf-8")
        seeded = SimpleNamespace(path=sess, has_seeds=True)
        argv = offline_cmd(seeded, method="color")
        ck("工作台：纯色球有点选球心 → 保留 --method color，并把 --seed-json 交给它",
           _has_seq(argv, ["--method", "color"])
           and _has_seq(argv, ["--seed-json", str(sess / "derived" / "seeds.json")])
           and "--method" in argv and argv[argv.index("--method") + 1] == "color",
           " ".join(argv))

        argv = offline_cmd(seeded, method="specular")
        ck("工作台：镜面球有点选球心 → --method specular + --seed-json",
           _has_seq(argv, ["--method", "specular"])
           and _has_seq(argv, ["--seed-json", str(sess / "derived" / "seeds.json")]),
           " ".join(argv))

        # 行为没变的证据（搬迁不是顺手改命令）：老写法是
        # `offline_stereo(path)` 之后在 GUI 里手工追加 `--method/--seed-json`；
        # 新写法把这两段交给 `_actions`。**没给方法**时仍是老行为（specular），
        # 两条 argv 必须逐字相等（连顺序）。
        old_seeded = _actions.offline_stereo(sess) + [
            "--method", "specular",
            "--seed-json", str(sess / "derived" / "seeds.json")]
        old_plain = _actions.offline_stereo(sess)
        ck("工作台：有 seeds 且没给方法时 argv 与改动前逐字相同（顺序也一样）",
           old_seeded == offline_cmd(seeded), " ".join(old_seeded[-4:]))
        ck("工作台：无 seeds 且不传参数时，argv 与改动前逐字相同",
           old_plain == offline_cmd(plain), " ".join(old_plain))

    # ---- 门厅「双目实时 3D」：argv 默认值就是"能用实时"的那一份 ----
    # 守两件事：① 默认必须带 `--detect-width 960`（全分辨率两路检测 35.6 ms，
    # 帧率会掉到 20 fps 以下，"实时"就名不副实）；② 空串不许变成 `--left ""`
    # —— 那会让子进程的角色解析报错，而界面上看不出差别。
    rt = _actions.stereo_realtime_args()
    ck("门厅：双目实时 argv 默认带 --detect-width 960 与 --max-skew-ms 60",
       _has_seq(rt, ["--detect-width", "960"])
       and _has_seq(rt, ["--max-skew-ms", "60"]),
       " ".join(rt))
    ck("门厅：双目实时 argv 空值不拼参数（不产生 --left \"\"）",
       "--left" not in rt and "--right" not in rt and '""' not in rt,
       " ".join(rt))
    rt2 = _actions.stereo_realtime_args(left="0", right="2", method="motion")
    ck("门厅：双目实时 argv 传了相机与方法就原样带上",
       _has_seq(rt2, ["--left", "0"]) and _has_seq(rt2, ["--right", "2"])
       and _has_seq(rt2, ["--method", "motion"]),
       " ".join(rt2[-6:]))

    # ---- 「双目实时 3D」的准入判据：必须走 current_calibration ----
    # 2026-11-16 用户点按钮得到「还没有可用的双目标定（实验性质平行装置1 不存在）」，
    # 而 config/calib/实验性质平行装置1.json + .npz **都在** —— 因为我当时写的是
    # `Path(base).exists()`，而标定基名是**无扩展名**的，`exists()` 恒为假。
    # 这正是 stereo/profiles.py:597 `calibration_files_exist` 文档里记着的
    # 「必然误报」（check_stereo_setup.py:198 当年一模一样）。两条一起钉：
    # ① 唯一判定本身的行为；② 门厅那段代码不许自己再拼一份。
    import tempfile as _tf
    from stereo.profiles import calibration_files_exist
    with _tf.TemporaryDirectory() as _td:
        _base = Path(_td) / "rigA"
        _bare = calibration_files_exist(_base) is False
        _base.with_suffix(".json").write_text("{}", encoding="utf-8")
        _json = calibration_files_exist(_base) is True
    ck("标定存在性：<base>.json 存在即算有标定（只判 base.exists() 会恒假）",
       _bare and _json, f"裸基名={_bare}, 有 .json={_json}")
    # 「准入不许自己拼 path.exists()」这条判据在下面各窗口的循环里
    # （要看 `_gui_stereo.py` 的 `_start_realtime_3d`，不是门厅了）。

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

    # ---- 2026-11-14 极化清单：物理球直径 / 强制通过标定的 argv 与二次确认 ----
    # 判据全部打在产出 argv 的那一层（_actions）+ handler 的结构上，不需要 Tk。
    from apps._actions import calibrate_args
    import ast as _ast
    import inspect
    import textwrap
    from apps._gui_stereo import StereoWindow

    cmd_d = offline_stereo("s", anchor=True, anchor_diameter_mm="21.6")
    ck("物理球直径 → --anchor-diameter-mm（勾锚点门且合法才下传）",
       _has_seq(cmd_d, ["--anchor-gate", "on",
                        "--anchor-diameter-mm", "21.6"]), " ".join(cmd_d[1:]))
    ck("锚点门未勾：物理球直径不下传（填了也不生效）",
       "--anchor-diameter-mm" not in offline_stereo(
           "s", anchor=False, anchor_diameter_mm="21.6"))
    for bad in ("", "abc", "-3", "0"):
        ck(f"物理球直径非法/空（{bad!r}）不拼进命令行，别让 CLI 崩",
           "--anchor-diameter-mm" not in offline_stereo(
               "s", anchor=True, anchor_diameter_mm=bad))
    ck("px 框留空仍走阶梯（GUI 恒走、不加 --no-radius-auto）；CLI 显式关才加",
       "--no-radius-auto" not in offline_stereo(
           "s", anchor=True, radius_auto=True)
       and "--no-radius-auto" in offline_stereo("s", radius_auto=False))
    ck("--force-save 默认不出现，只有 force_save=True 才加",
       "--force-save" not in calibrate_args("joint", {}, "", "", "parallel")
       and "--force-save" in calibrate_args(
           "joint", {}, "", "", "parallel", force_save=True))

    # 二次确认：--force-save 只在确认通过后出现，且文案三点齐全。
    # 用 AST 读 _force_calibrate 的源码与字符串常量，不做全文子串碰运气。
    force_src = textwrap.dedent(inspect.getsource(StereoWindow._force_calibrate))
    force_tree = _ast.parse(force_src)
    literals = "".join(n.value for n in _ast.walk(force_tree)
                       if isinstance(n, _ast.Constant)
                       and isinstance(n.value, str))
    ck("强制通过二次确认文案三点齐全（正式路径 / 产物不可用 / 仅链路测试）",
       all(k in literals for k in (
           "正式标定路径", "forced_unusable", "仅链路测试", "不得用于数据结论")))
    ck("--force-save 只在二次确认通过之后才下传（先问、取消就 return）",
       "askyesno(" in force_src and "if not" in force_src
       and "return" in force_src
       and 0 <= force_src.find("askyesno") < force_src.find("force_save=True")
       and "force_save: bool = False" in inspect.getsource(
           StereoWindow._start_calibrate))
    return out


def _argv_single_source_checks() -> list[tuple[str, bool]]:
    """argv 只能有一个产地（`apps/_actions.py`）—— 用 AST 扫，不看人。

    判据故意是"**零字面量**"而不是"有没有调 `_actions`"：后者可以一边调
    一边又内联一份（那正是从前那种漂移）。只要界面文件里再出现 ``"xxx.py"``，
    就说明有人在手工拼命令行，紧接着就是"同一个动作两份参数清单"。

    经典界面（``apps/gui_main.py``）是**已知豁免**：它有 22 处内联，一次改完风险大。
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
    # `tools_dialog.py` 手工列进来：它不叫 `_gui_*`，但同样是界面文件 ——
    # 2026-11-07 补六条离线流程时新建的，漏扫等于给"内联 argv"留了个后门。
    gui_files = sorted((root / "apps").glob("_gui_*.py")) + [
        root / "apps" / "tools_dialog.py"]
    for f in gui_files:
        hits = py_literals(f)
        ok = not hits
        print(f"  [{'通过' if ok else '不达标'}] {f.name}: 脚本字面量 {len(hits)} 处"
              + ("" if ok else f"（行 {hits[:6]}）"))
        out.append((f"{f.name} 不许手写脚本路径", ok))

    legacy = py_literals(root / "apps/gui_main.py")
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
            "run_all.py", "calibrate_mono.py", "calibrate_focal.py",
            # 2026-11-07 补进界面的六条（原先只能在命令行里走）
            "make_calibration_board.py", "calibrate_color.py",
            "set_world_frame.py", "export_calibration.py",
            "export_ball_dataset.py", "train_ball_detector.py"]
    missing = [n for n in need if n not in acts]
    print(f"  [{'通过' if not missing else '不达标'}] 动作模块覆盖 {len(need) - len(missing)}/{len(need)} 个脚本"
          + ("" if not missing else f"（缺 {missing}）"))
    out.append(("_actions 覆盖全部界面动作", not missing))

    # ---- 计划书 M8：一条命令跑完整系统（run_all）必须有 GUI 入口 ----
    bench_src = (root / "apps" / "_gui_bench.py").read_text(encoding="utf-8")
    out.append(("工作台有「一键跑全流程」入口",
                "一键跑全流程" in bench_src and "_run_all_pipeline" in bench_src))
    out.append(("一键全流程调用 _actions.run_all（不在这里重写三步）",
                "_actions.run_all(s.path)" in bench_src))

    # 双目第①步的相机行必须用短标签（设备名），否则两路完整状态一拼
    # 就会换行，大角度窗内容高出 16 px（selftest_window_mode 的实测）。
    app_src = (root / "apps" / "_gui_app.py").read_text(encoding="utf-8")
    stereo_src = (root / "apps" / "_gui_stereo.py").read_text(encoding="utf-8")
    out.append(("相机行用短标签（设备名，不把完整采集状态塞进第①步）",
                "def camera_role_short" in app_src
                and 'camera_role_short("left")' in stereo_src
                and 'camera_role_text("left")' not in stereo_src))

    # 文件选择框不能跟着 Windows 的“上次文件夹”（可能在文档/桌面）。
    tools_src = (root / "apps" / "tools_dialog.py").read_text(encoding="utf-8")
    mono_src = (root / "apps" / "_gui_mono.py").read_text(encoding="utf-8")
    legacy_src = (root / "apps" / "gui_main.py").read_text(encoding="utf-8")
    out.append(("文件浏览默认回项目内目录（内参/轴向/工具台/单目）",
                'PROJECT_ROOT / "config" / "calib"' in stereo_src
                and "if not start.is_dir()" in stereo_src
                and "initialdir=str(start)" in stereo_src
                and "initialdir=self._browse_start" in stereo_src
                and "initialdir=start" in tools_src
                and "initialdir=str(start)" in mono_src
                and "initialdir=start" in legacy_src))
    return out


def _tools_action_checks() -> list[tuple[str, bool]]:
    """「标定与训练工具」六条动作的 argv 拼装（2026-11-07 补的界面入口）。

    这六条在补之前**只能在命令行里走**：README 把它们写成"下一步做什么"，
    界面上一个入口都没有。判据钉在这里、而不是"按钮存在"上 —— 新按钮最常见
    的错不是没画出来，是点了跑的是另一件事（或者把空串原样传下去）。
    """
    import ast

    from apps import _actions

    out: list[tuple[str, bool]] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'通过' if ok else '不达标'}] {label}")
        if not ok and detail:
            print(f"      实际：{detail}")
        out.append((label, ok))

    print("\n----- 标定与训练工具：六条动作的 argv（apps/_actions.py）-----")

    # ① 标定板：填了就进命令行；留空一律不传（落到脚本默认，不落空串）
    a = _actions.make_board(cols="7", rows="6", square="30")
    check("标定板：板参数进命令行",
          _has_seq(a, ["--cols", "7", "--rows", "6", "--square", "30"]), " ".join(a))
    b = _actions.make_board()
    check("标定板：留空 = 不传（交给脚本默认 8×5 / 25 mm）",
          not any(x in b for x in ("--cols", "--rows", "--square", "--page", "--out")),
          " ".join(b))

    # ② 取色：给了图片就只用图片；什么都没给也必须给一个来源
    c = _actions.calibrate_color(image="a.png", camera="2")
    check("取色：给了图片就只用图片（不再带 --camera）",
          _has_seq(c, ["--image", "a.png"]) and "--camera" not in c, " ".join(c))
    d = _actions.calibrate_color()
    check("取色：什么都没给 → --camera 0（不是空 argv：空 argv 会起个立刻退出的进程）",
          _has_seq(d, ["--camera", "0"]), " ".join(d))

    # ③ 世界坐标系：--image 必填；板参数留空不传；试算只加 --dry-run
    e = _actions.set_world_frame(image="g.png", cols="9", rows="6", square="40",
                                 note="地面")
    check("世界系：图必填 + 板参数进命令行",
          _has_seq(e, ["--image", "g.png", "--board-cols", "9", "--board-rows", "6",
                       "--board-square", "40", "--note", "地面"]), " ".join(e))
    f = _actions.set_world_frame(image="g.png", dry_run=True)
    check("世界系：试算带 --dry-run，且不带空的板参数",
          "--dry-run" in f and "--board-cols" not in f and "--note" not in f,
          " ".join(f))

    # ③b 世界系的符号与标定来源（2026-11-20 补）：以前 GUI 根本没有这两个输入，
    # Z 朝上/朝下只能手改文件；而符号写错的矩阵"看起来完全正常"。
    z = _actions.set_world_frame(image="g.png", ground_axis="-z", rig="x")
    check("世界系：Z 轴方向与标定文件真的进命令行",
          _has_seq(z, ["--ground-axis", "-z"]) and _has_seq(z, ["--rig", "x"]),
          " ".join(z))
    z2 = _actions.set_world_frame(image="g.png", ground_axis="z")
    check("世界系：Z 朝下也要显式传 --ground-axis z（不靠脚本默认值碰运气）",
          _has_seq(z2, ["--ground-axis", "z"]), " ".join(z2))
    blank = _actions.set_world_frame(image="g.png")
    check("世界系：两项留空 = 一个都不传（落到当前档案 / 脚本默认）",
          "--ground-axis" not in blank and "--rig" not in blank, " ".join(blank))

    # ④ 导出标定参数
    g = _actions.export_calibration(calib="c", out="o", formats="json")
    check("导出标定：目录与格式进命令行",
          _has_seq(g, ["--calib", "c", "--out", "o", "--formats", "json"]), " ".join(g))

    # ⑤ 数据集：显式会话优先于 --all；两者都不给就不硬塞
    h = _actions.export_ball_dataset(out="d", session="s1")
    check("数据集：给了会话就只导它（不再带 --all）",
          _has_seq(h, ["--session", "s1"]) and "--all" not in h, " ".join(h))
    i = _actions.export_ball_dataset(out="d")
    check("数据集：没给会话 → --all", "--all" in i, " ".join(i))
    j = _actions.export_ball_dataset(all_sessions=False)
    check("数据集：既没会话也不要全导 → 两个都不传（让脚本自己报缺参数）",
          "--all" not in j and "--session" not in j, " ".join(j))

    # ⑥ 训练：L3 参数留空一律不传（不在这里重复写脚本默认值）
    k = _actions.train_ball_detector(data="ds", epochs="80")
    check("训练：数据集与轮数进命令行",
          _has_seq(k, ["--data", "ds", "--epochs", "80"]), " ".join(k))
    m2 = _actions.train_ball_detector()
    check("训练：参数留空一律不传（少一处默认值就少一处漂移）",
          not any(x in m2 for x in ("--data", "--epochs", "--imgsz", "--batch",
                                    "--device", "--out")), " ".join(m2))

    # ⑦ 环境切换：GUI 只构造 `setup_env.py --adopt`，指针格式仍由 setup_env 唯一实现。
    env = _actions.setup_env_adopt(r"C:\Python313\python.exe")
    check("环境切换：走 setup_env.py --adopt",
          any("setup_env.py" in x for x in env)
          and _has_seq(env, ["--adopt", r"C:\Python313\python.exe"]),
          " ".join(env))

    # 界面侧：对话框只许调动作模块，且六条流程都要真的调
    tree = ast.parse((ROOT / "apps" / "tools_dialog.py").read_text(encoding="utf-8"))
    lits = [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and n.value.endswith(".py")]
    check("工具对话框里没有脚本字面量（argv 仍只有 _actions 一个产地）",
          not lits, str(lits))
    called = {n.func.attr for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
              and isinstance(n.func.value, ast.Name) and n.func.value.id == "_actions"}
    need = {"make_board", "calibrate_color", "set_world_frame", "export_calibration",
            "export_ball_dataset", "train_ball_detector"}
    check("工具对话框覆盖全部六条流程（缺 %s）" % sorted(need - called),
          need <= called, str(sorted(called)))
    tools_src = (ROOT / "apps" / "tools_dialog.py").read_text(encoding="utf-8")
    check("工具台有 Python 运行环境卡片 + 验证切换按钮",
          "Python 运行环境" in tools_src and "验证并切换" in tools_src)
    check("环境切换只调 _actions.setup_env_adopt（不自己写指针）",
          "_actions.setup_env_adopt" in tools_src
          and "write_pointer(" not in tools_src
          and "_setup_env.read_pointer" in tools_src)

    # 世界系那一卡的两个新控件必须**真的接到 argv 上**：只加输入框、不往下传，
    # 用户填了等于没填（本仓库最常见的缺陷形态）。
    kw = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "set_world_frame"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "_actions"):
            kw |= {k.arg for k in node.keywords}
    check("工具对话框把 ground_axis / rig 真的传给 set_world_frame（缺 %s）"
          % sorted({"ground_axis", "rig"} - kw),
          {"ground_axis", "rig"} <= kw, str(sorted(kw)))

    # 「Z 轴朝上/朝下」必须是下拉里的两个候选（写在说明里不算控件）。
    combos = [n for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
              and n.func.attr == "param_combo"]
    axis_combo = None
    for call in combos:
        for arg in call.args:
            if (isinstance(arg, ast.List)
                    and {getattr(e, "value", None) for e in arg.elts} == {"-z", "z"}):
                axis_combo = call
    check("工具台有「世界 Z 轴 -z / z」下拉（控件存在，不是只有说明文字）",
          axis_combo is not None, str([ast.unparse(c)[:60] for c in combos]))
    return out


def _realtime_source_checks() -> list[tuple[str, bool]]:
    """实时双目落的会话在「读」这一侧的三件事（2026-11-17 用户要求）。

    实时窗退出时会往 ``data/stereo/<模式>/<时间戳>/`` 落一段会话（有
    ``derived/`` 三样产物、**没有原始左右录像**）。用户的原话是"实时双目也要
    能在数据管理里看到、能在 3D 查看器里打开"，所以这里钉三条**可见性**判据：

    ① ``scan_items`` 把 ``session.json`` 的 ``source`` 读进 ``SessionItem``
       —— 读不到的话数据管理里就跟普通录制会话一模一样，没法区分；
    ② 名称列加「实时 ·」前缀，且**只改显示、不改 label**（搜索/导出仍按原名）；
    ③ 第④步「跑离线三维」在 GUI 里**先拦**：实时会话没有原始视频，
       ``run_stereo_video`` 会 validate 失败退出，用户白等一次进程启动。

    ③ 另有一条**结构**判据（AST）：拦截调用必须在函数体里**所有**
    ``tasks.start(`` 之前 —— 光断言"函数返回非空"不管接线，删掉那三行
    也照样是绿的。
    """
    import ast
    import json
    import shutil
    import tempfile

    from apps import _data_export

    out: list[tuple[str, bool]] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'通过' if ok else '不达标'}] {label}")
        if not ok and detail:
            print(f"      实际：{detail}")
        out.append((label, ok))

    print("\n----- 实时会话（source=realtime）在数据管理/双目窗里的可见性 -----")

    base = Path(tempfile.mkdtemp(prefix="selftest-realtime-"))
    try:
        rt = base / "parallel" / "20261117-100000"
        rt.mkdir(parents=True)
        (rt / "session.json").write_text(json.dumps({
            "schema": "sphere-stereo-session", "version": 1, "id": "x",
            "created": "2026-11-17T10:00:00", "source": "realtime",
            "label": "测试", "left": {}, "right": {},
        }, ensure_ascii=False), encoding="utf-8")
        plain = base / "parallel" / "20261117-110000"
        plain.mkdir(parents=True)
        (plain / "session.json").write_text(json.dumps({
            "schema": "sphere-stereo-session", "version": 1, "id": "y",
            "created": "2026-11-17T11:00:00", "label": "普通录制",
            "left": {}, "right": {},
        }, ensure_ascii=False), encoding="utf-8")

        items = {it.path.name: it for it in _data_export.scan_items(base)}
        got = items.get("20261117-100000")
        ref = items.get("20261117-110000")
        check("scan_items 把 session.json 的 source 读进 SessionItem",
              got is not None and got.source == "realtime",
              repr(getattr(got, "source", "<没扫到>")))

        from apps.data_manager import name_cell_text
        check("名称列对实时会话加「实时 ·」前缀（不改 label 本身）",
              got is not None and name_cell_text(got) == "实时 · 测试"
              and got.label == "测试",
              repr(getattr(got, "label", "")) + " / "
              + repr(name_cell_text(got) if got is not None else ""))
        check("普通会话（session.json 没有 source 键）名称原样、source 为空串",
              ref is not None and name_cell_text(ref) == "普通录制"
              and ref.source == "",
              repr((getattr(ref, "source", "?"),
                    name_cell_text(ref) if ref is not None else "")))

        from apps._gui_stereo import realtime_session_blocker
        why = realtime_session_blocker(rt)
        check("第④步前置判据：实时会话被拦住，且说清「不能重跑」",
              bool(why) and "不能重跑" in why, repr(why))
        check("第④步前置判据：普通会话放行（返回空串）",
              realtime_session_blocker(plain) == "",
              repr(realtime_session_blocker(plain)))
        broken = base / "broken"
        broken.mkdir()
        (broken / "session.json").write_text("{不是 json", encoding="utf-8")
        check("第④步前置判据：session.json 坏了当普通会话（不误拦）",
              realtime_session_blocker(broken) == "",
              repr(realtime_session_blocker(broken)))
        check("第④步前置判据：目录里根本没有 session.json 也放行"
              "（由 run_stereo_video 报更具体的问题）",
              realtime_session_blocker(base / "no-such-dir") == "",
              repr(realtime_session_blocker(base / "no-such-dir")))
    finally:
        shutil.rmtree(base, ignore_errors=True)

    # 结构判据：拦截必须真的接在 `_start_offline` 里、在起子进程之前。
    tree = ast.parse((ROOT / "apps" / "_gui_stereo.py").read_text(encoding="utf-8"))
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "_start_offline"),
              None)
    guard = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name)
             and n.func.id == "realtime_session_blocker"] if fn else []
    starts = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Call)
              and isinstance(n.func, ast.Attribute) and n.func.attr == "start"
              and isinstance(n.func.value, ast.Attribute)
              and n.func.value.attr == "tasks"] if fn else []
    check("_start_offline 里调用了 realtime_session_blocker",
          bool(guard), "没找到调用")
    check("拦截调用在**全部** tasks.start( 之前（删掉就红）",
          bool(guard) and bool(starts) and max(guard) < min(starts),
          f"guard={guard} starts={starts}")
    return out


def _status_symbol_checks() -> list[tuple[str, bool]]:
    """数据管理的状态符号要覆盖「◑ 三维点过少」（2026-11-18 用户报的 bug）。

    用户原话是"有 3D 处理后的数据和视频的没有 ✔ 的状态符号，没有的反而有"。
    根因有两层：① `seeds.json`（输入）被当成 ✓ 的闸门；② 匹配点只有一两个
    也报 ✓。第二层修完以后多出一个中间态 `traj_thin`（◑），数据管理的
    筛选下拉 + 图例 + 过滤映射三处都要跟上，否则用户按「✓ 有3D」筛，
    那些根本画不出轨迹的会话仍然混在里面。

    这里用源码扫描（这些文案是 `data_manager.py` 方法内的局部常量，
    没有模块级符号可 import），符号源仍从 `_workbench.STATUS_SYMBOLS` 取，
    保证"两套图标各说各话"不会再发生。
    """
    from apps._workbench import STATUS_SYMBOLS

    src = (ROOT / "apps" / "data_manager.py").read_text(encoding="utf-8")
    out: list[tuple[str, bool]] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'通过' if ok else '不达标'}] {label}")
        if not ok and detail:
            print(f"      实际：{detail}")
        out.append((label, ok))

    print("\n----- 数据管理状态符号（◑ 三维点过少）-----")
    start = src.find("self.status_legend")
    legend = src[start:start + 1200] if start != -1 else ""
    has_option = '"◑ 三维点过少"' in src
    has_mapping = '"◑ 三维点过少": "traj_thin"' in src
    legend_ok = "◑ 三维点过少 →" in legend
    symbol_ok = STATUS_SYMBOLS.get("traj_thin") == "◑"
    check("状态下拉含「◑ 三维点过少」，图例文本含 ◑，过滤映射到 traj_thin，"
          "符号与 STATUS_SYMBOLS 一致",
          has_option and has_mapping and legend_ok and symbol_ok,
          f"option={has_option} mapping={has_mapping} legend={legend_ok} "
          f"symbol={STATUS_SYMBOLS.get('traj_thin')!r}")
    # 2026-11-18：`—` 从"空目录"改成真会出现的"没录成（0 帧且零产物）"。
    # 图例必须跟着判据一起改，否则用户按老说法去翻空目录，永远找不到那段会话。
    check("图例把「—」解释为「没录成」而不是「空目录」（与 empty 判据同步）",
          "没录成" in legend and "空目录" not in legend,
          f"legend={legend!r}")
    return out


def _legacy_deprecation_checks() -> list[tuple[str, bool]]:
    """经典界面必须保持「入口废止」（2026-11-20 用户裁决）。

    为什么放进本测试：门厅按钮、``gui_main.main()`` 与 ``run.bat`` 是三个
    不同入口，任何一个被顺手接回去，用户就会重新看到旧流程 —— 这件事
    **不会报错**，只会"多一个入口"。旧实现仍保留（多个回归测试按文本与
    导入引用它），所以这里守的是**可达性**，不是"文件存不存在"。

    旧界面的 AVI 录制格式下拉不再可用（新界面录制不传 `--container`，恒录
    mkv）；raw 视频仍可从「数据管理 → 查看原始视频」进入。要恢复任何一项，
    必须先在 `docs/决策记录.md` 改 2026-11-20 的决策，而不是只改这里一行。
    """
    print("\n----- 经典界面：入口必须保持废止（2026-11-20）-----")
    hub_src = (ROOT / "apps" / "_gui_app.py").read_text(encoding="utf-8")
    main_src = (ROOT / "apps" / "gui_main.py").read_text(encoding="utf-8")
    bat_src = (ROOT / "run.bat").read_text(encoding="utf-8-sig",
                                            errors="replace")
    # rem 行里可以写历史说明；判据只看真正会执行的正文。
    bat_body = "\n".join(ln for ln in bat_src.splitlines()
                         if not ln.strip().lower().startswith("rem"))

    def hub_ok(src: str) -> bool:
        """看**真实调用**，不看文本 —— docstring 里解释「不调 tasks.start」
        不该把自己判红（第一版就这么误报过一次）。"""
        import ast as _ast

        tree = _ast.parse(src)
        fn = next(n for n in _ast.walk(tree)
                  if isinstance(n, _ast.FunctionDef) and n.name == "_open_legacy")

        def dotted(call: _ast.Call) -> str:
            parts: list[str] = []
            cur = call.func
            while isinstance(cur, _ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, _ast.Name):
                parts.append(cur.id)
            return ".".join(reversed(parts))

        names = [dotted(n) for n in _ast.walk(fn) if isinstance(n, _ast.Call)]
        # 钉**真正创建按钮的那一行**，不是"源码里出现过这个词"：
        # 注释与弹窗标题里也有「已废止」，那样写反向自证会失效。
        button = ('self.surface.secondary(row2, "经典界面（已废止）", '
                  'self._open_legacy)')
        return (button in src
                and any(n.endswith("showinfo") for n in names)
                and not any(n.endswith("tasks.start") for n in names)
                and not any("legacy_gui" in n for n in names))

    def main_ok(src: str) -> bool:
        body = src[src.index("def main("):]
        return ('if "--legacy" in args:' in body
                and "return 2" in body
                and "Launcher().mainloop()" not in body)

    def bat_ok(src: str) -> bool:
        start = src.index(":ui_legacy")
        body = src[start:src.index("\n:ui_cli", start)]
        return "已废止" in body and "gui_main.py --legacy" not in body

    actions_src = (ROOT / "apps" / "_actions.py").read_text(encoding="utf-8")
    results = [
        ("门厅按钮写明「经典界面（已废止）」", "经典界面（已废止）" in hub_src),
        ("点击「经典界面」只弹提示、不起任务", hub_ok(hub_src)),
        ("`--legacy` 被 main() 明确拒绝（不启动 Launcher）", main_ok(main_src)),
        ("`run.bat legacy` 只打印提示、不再起旧界面", bat_ok(bat_body)),
        ("`_actions` 里没有 legacy_gui 第二入口",
         "def legacy_gui" not in actions_src),
    ]

    # ---- 反向自证：三个入口各接回一处，判据必须红（不是空转）----
    hub_start = hub_src.index("def _open_legacy(")
    hub_end = hub_src.index("\n    def ", hub_start + 1)
    broken_hub = (hub_src[:hub_start]
                  + hub_src[hub_start:hub_end].replace(
                      "        from tkinter import messagebox\n",
                      "        self.app.tasks.start([], \"x\")\n"
                      "        from tkinter import messagebox\n", 1)
                  + hub_src[hub_end:])
    main_at = main_src.index("def main(")
    broken_main = (main_src[:main_at]
                   + main_src[main_at:].replace(
                       "        return 2\n", "        Launcher().mainloop()\n", 1))
    bat_start = bat_body.index(":ui_legacy")
    bat_end = bat_body.index("\n:ui_cli", bat_start)
    broken_bat = (bat_body[:bat_start]
                  + bat_body[bat_start:bat_end].replace(
                      "exit /b 0", '"%PY%" apps\\gui_main.py --legacy', 1)
                  + bat_body[bat_end:])
    results += [
        ("反向：把 tasks.start 加回 _open_legacy → 判据会红",
         not hub_ok(broken_hub)),
        ("反向：让 --legacy 重新 mainloop → 判据会红",
         not main_ok(broken_main)),
        ("反向：让 run.bat legacy 重新起旧界面 → 判据会红",
         not bat_ok(broken_bat)),
        ("反向：按钮名称改回「过渡」→ 判据会红",
         not hub_ok(hub_src.replace(
             'self.surface.secondary(row2, "经典界面（已废止）", '
             'self._open_legacy)',
             'self.surface.secondary(row2, "经典界面（过渡）", '
             'self._open_legacy)', 1))),
    ]
    for name, ok in results:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return results


def main() -> int:
    print("=" * 74)
    print("GUI 预算自测：分窗后的每一步都必须留在一屏之内")
    print("=" * 74)

    # 参数拼装判据先跑：它直接调纯函数、不需要 Tk，所以**没有显示器也要跑**。
    # 结构判据可以跳过，参数拼装不行 —— 跳过的那部分一律不算验证过。
    results: list[tuple[str, bool]] = _stereo_param_checks()
    results += _ball_settings_checks()
    results += _argv_single_source_checks()
    results += _tools_action_checks()
    results += _realtime_source_checks()
    results += _status_symbol_checks()
    results += _mono_session_checks()
    results += _mono_seed_checks()
    results += _regression_2026_11_06_static()
    results += _legacy_deprecation_checks()

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
        # 「标定与训练」工具台（2026-11-07 新窗口）：**必须进同一套预算与文案判据**。
        # 不进的话它就是"没人量的那一屏"—— 实测这条加进来之前，卡片说明里的
        # `**` 星号（B-96 同型）就原样印在截图上了。
        app.hub._open_tools()
        _tools = getattr(app.hub, "_tools_window", None)
        if _tools is not None and _tools.winfo_exists():
            windows.append(("tools", _tools))
        else:                                               # 开不出来本身就是缺陷
            results.append(("请点「工具 · 标定与训练…」能把工具台开出来", False))
        app.update_idletasks()

        # ---- 2026-11-07：平行 / 大角度第②步不许再是同一套标定设计 ----
        # 用户两次纠正，最终口径：平行**只有「目标对数」**（收够就停），
        # 不许出现任何覆盖约束 ——「五个角都要拍到 / 5/5 全绿」是上一轮我
        # 加错的：既然收够对数就停，再挂一个"必须集齐五个角"自相矛盾，
        # 而且后端挑帧根本没有这条门限。分化线 / 4/5 放宽属于大角度链式，
        # 同样不许出现在平行窗。大角度一侧保持不动。
        by_name = dict(windows)
        def _card_text(win) -> str:
            return " ".join(collect_texts(win))
        def _parallel_ok(text: str) -> bool:
            return ("目标对数" in text
                    and not any(bad in text for bad in (
                        "五个角", "5/5", "4/5", "分化线", "全绿", "覆盖要求")))
        p_txt = _card_text(by_name["stereo(parallel)"])
        a_txt = _card_text(by_name["stereo(angled)"])
        shared = "目标对数 分化线 张 （拍够后 4/5 即可；之前要求 5/5 全绿）"
        for label, ok in (
            ("平行第②步只有「目标对数」，没有任何覆盖/5角/4-5 限制",
             _parallel_ok(p_txt)),
            ("大角度第②步保留「分化线 / 4/5 即可」（本次不许被改）",
             "分化线" in a_txt and "4/5 即可" in a_txt),
            # 反证：把上一版"5/5 全绿"或"共用文案"喂回来，判据必须变红。
            ("反向：写回「五个角都要拍到，5/5 全绿」会被抓到",
             not _parallel_ok("目标对数 张 （五个角都要拍到，5/5 全绿；收够就停）")),
            ("反向：共用文案（分化线 + 4/5 放宽）会被平行判据抓到",
             not _parallel_ok(shared)),
            ("反向：平行文案漏掉「目标对数」也会被抓到",
             not _parallel_ok("（收够就停，没有覆盖要求）")),
        ):
            print(f"  [{'通过' if ok else '不达标'}] {label}")
            results.append((label, ok))

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

        # 「实时 3D」必须在**每种双目窗口自己的第③步**（2026-11-16 用户裁决：
            # 实时与离线是同一套装置的两种处理方式，不属于门厅的跨模式工具）。
            # 除了"按钮在"，还要钉两件事：绑了命令；准入走 current_calibration
            # 而不是自己 Path().exists()（标定基名无扩展名，只判 exists 恒为假）。
            if name.startswith("stereo"):
                _rt = getattr(win, "btn_realtime", None)
                _ok_btn = (_rt is not None
                           and str(_rt.cget("text")).startswith("实时 3D")
                           and bool(str(_rt.cget("command"))))
                print(f"  [{'通过' if _ok_btn else '不达标'}] "
                      f"{name} 第③步有「实时 3D」按钮且已绑命令")
                results.append((f"{name} 第③步有「实时 3D」入口", _ok_btn))
                _gs = (ROOT / "apps" / "_gui_stereo.py").read_text(encoding="utf-8")
                _a = _gs.index("def _start_realtime_3d")
                _b = _gs.index("def _sync_record_enabled", _a)
                _seg = _gs[_a:_b]
                _ok_gate = ("current_calibration(" in _seg
                            and "path.exists()" not in _seg
                            and "_record_gate()" in _seg)
                print(f"  [{'通过' if _ok_gate else '不达标'}] "
                      f"{name} 实时 3D 的准入复用 current_calibration + _record_gate")
                results.append((f"{name} 实时 3D 准入判据", _ok_gate))
                # 分辨率必须跟着**这套标定**走：内参是像素量纲的，喂错档后端
                # 只会 exit 2；而大角度 2560x1440 / 平行 1920x1080 本来就不同档，
                # 不传等于让用户自己去翻标定文件。
                _ok_size = "image_size" in _seg
                _arg_ok = ("stereo_realtime_args(" in _seg
                           and "width=" in _seg and "height=" in _seg)
                print(f"  [{'通过' if (_ok_size and _arg_ok) else '不达标'}] "
                      f"{name} 实时 3D 按标定 image_size 传分辨率")
                results.append((f"{name} 实时 3D 分辨率跟随标定",
                                _ok_size and _arg_ok))
                # **行为判据**：球种类/预设要真的传到 argv（2026-11-17 用户实测
                # "塑料球和金属球都检不出来"—— 按钮原来只传 --method color、
                # 不传 --preset，于是永远吃白球默认阈值；镜面球则根本不该启动）。
                _ready, _ = win._record_gate()
                _got: list = []
                _orig_start = win.app.tasks.start
                win.app.tasks.start = lambda cmd, title, **k: _got.append(cmd)
                try:
                    win._start_realtime_3d()
                finally:
                    win.app.tasks.start = _orig_start
                if not _ready:
                    _ok_rt = not _got        # 标定不配这个窗 → 一个任务都不许起
                    _why_txt = "未就绪时拦住了"
                elif win.ball.method() == "specular":
                    _ok_rt = not _got        # 镜面球实时必然 0 检出 → 拦住
                    _why_txt = "镜面球拦住了"
                else:
                    _cmd = _got[-1] if _got else []
                    _m = win.ball.method()
                    _p = win.ball.preset()
                    _ok_rt = (_has_seq(_cmd, ["--method", _m])
                              and (("--preset" in _cmd) == bool(_p))
                              and (not _p or _has_seq(_cmd, ["--preset", _p])))
                    _why_txt = f"method={_m} preset={_p or '(空)'}"
                print(f"  [{'通过' if _ok_rt else '不达标'}] "
                      f"{name} 实时 3D 把球种类/预设传下去（{_why_txt}）")
                results.append((f"{name} 实时 3D 传球种类与预设", _ok_rt))

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

                from apps._gui_widgets import BALL_METAL, BALL_PLASTIC

                captured.clear()
                # 镜面球 + 种子：必须用 specular（种子初始化 CSRT）。
                mono.ball.set_kind(BALL_METAL)
                mono._start_video()
                cmd = captured[0] if captured else []
                got_seed = cmd[cmd.index("--seed") + 1] if "--seed" in cmd else ""
                got_m = cmd[cmd.index("--method") + 1] if "--method" in cmd else ""
                ok_seed = (got_seed == "812,530,91" and got_m == "specular"
                           and mono.var_detector.get() == "specular")
                print(f"  [{'通过' if ok_seed else '不达标'}] 镜面球 + seed.json → 命令行 "
                      f"--seed {got_seed or '(缺失)'} 且 --method "
                      f"{got_m or '(缺失)'}（下拉也同步成 {mono.var_detector.get()!r}）")
                results.append(("镜面球有种子时命令行带 --seed 且检测器 specular", ok_seed))

                # 纯色球 + 种子：**不许**换成 specular；种子交给 color 取色/锚半径。
                # 2026-11-06 之前这里会被强制成 specular，纯色球点选后反而 0 检出。
                captured.clear()
                mono.ball.set_kind(BALL_PLASTIC)
                mono.var_preset.set("red_ball")
                mono._start_video()
                cmd = captured[0] if captured else []
                got_seed = cmd[cmd.index("--seed") + 1] if "--seed" in cmd else ""
                got_m = cmd[cmd.index("--method") + 1] if "--method" in cmd else ""
                ok_plastic_seed = (got_seed == "812,530,91" and got_m == "color"
                                   and mono.var_detector.get() == "color")
                print(f"  [{'通过' if ok_plastic_seed else '不达标'}] 纯色球 + seed.json → "
                      f"--method {got_m or '(缺失)'}（**不换 specular**）、"
                      f"--seed {got_seed or '(缺失)'}")
                results.append(("纯色球有种子时仍用 color（种子取色/锚半径）", ok_plastic_seed))

                # 打开一段会话时，seed.json 的 ball_kind 要能恢复球种类：
                # 旧文件没字段 → metal（兼容）；plastic → 纯色球。
                seed.write_text(json.dumps({
                    "x": 812, "y": 530, "radius": 91, "frame": 0,
                    "image_size": [1920, 1080],
                    "created": "2026-01-02 00:00:00",
                    "ball_kind": "plastic"}, ensure_ascii=False), encoding="utf-8")
                mono._seed_kind_for = None
                mono._apply_seed_kind(sess)
                ok_restore_plastic = mono.ball.kind() == BALL_PLASTIC
                seed.write_text(json.dumps({
                    "x": 812, "y": 530, "radius": 91, "frame": 0,
                    "image_size": [1920, 1080],
                    "created": "2026-01-02 00:00:00"}, ensure_ascii=False),
                    encoding="utf-8")
                mono._seed_kind_for = None
                mono._apply_seed_kind(sess)
                ok_restore_old = mono.ball.kind() == BALL_METAL
                print(f"  [{'通过' if ok_restore_plastic and ok_restore_old else '不达标'}] "
                      f"打开会话时按 seed.json 恢复球种类：plastic→纯色、旧文件→金属")
                results.append(("按 seed.json 恢复球种类", ok_restore_plastic and ok_restore_old))

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

                # 镜面球时「颜色预设」下拉必须显示「不使用」：只把值留成 red_ball、
                # 靠 resolve 不传，界面上仍写着"红球"，用户会以为红球在生效。
                mono.var_preset.set("red_ball")
                mono.ball.set_kind(BALL_METAL)
                hidden_label = mono.ball._var_preset_label.get()
                ok_preset_hide = (hidden_label == "不使用"
                                  and mono.ball.preset() == "")
                mono.ball.set_kind(BALL_PLASTIC)
                ok_preset_back = (mono.ball._var_preset_label.get() == "红球"
                                  and mono.ball.preset() == "red_ball")
                print(f"  [{'通过' if ok_preset_hide else '不达标'}] 镜面球时预设下拉"
                      f"显示「{hidden_label}」、不传 --preset")
                print(f"  [{'通过' if ok_preset_back else '不达标'}] 切回纯色球后"
                      f"恢复原选择「{mono.ball._var_preset_label.get()}」")
                results.append(("镜面球时预设显示不使用", ok_preset_hide))
                results.append(("切回纯色球恢复预设", ok_preset_back))

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

        # ---------- 2026-11-06 GUI 走查遗留项：需要窗口的回归判据 ----------
        # 放在既有判据之后：这些会改 btn_record / mono._session / ball 状态，
        # 前面的结构判据量的是初始界面，不能被它们污染。
        def tcheck(label: str, ok: bool, detail: str = "") -> None:
            print(f"  [{'通过' if ok else '不达标'}] {label}")
            if not ok and detail:
                print(f"      实际：{detail}")
            results.append((label, ok))

        print("\n----- 2026-11-06 遗留项回归（窗口）-----")
        import json as _json
        import tempfile
        from unittest.mock import patch

        from apps._gui_widgets import BALL_METAL, BALL_PLASTIC
        from sphere_tracker.camera_registry import Registry, RoleSpec

        wins = dict(windows)
        # 2026-11-20：录制门禁现在也要求相机角色有**实测采集索引**。
        # 这两张表只替换 `load_registry()` 的返回值，不碰真实 config/cameras.json。
        fake_reg = Registry(roles={
            "left": RoleSpec(name="aoni left", capture_index=0, location="床"),
            "right": RoleSpec(name="aoni", capture_index=1, location="阳台门"),
        })
        null_cap_reg = Registry(roles={
            "left": RoleSpec(name="aoni left", location="床"),
            "right": RoleSpec(name="aoni", location="阳台门"),
        })

        # ① B-181 录制门禁：_refresh_state 同步的 btn_record 跟着 _record_gate 走。
        # 打桩 AppState.parallel_status / angled_status（返回 (bool, str)），
        # 不去真读 config 里的标定档案。
        gate_ok = True
        gate_detail: list[str] = []
        for _mode, _key, _status in (
                ("parallel", "stereo(parallel)", "parallel_status"),
                ("angled", "stereo(angled)", "angled_status")):
            w = wins[_key]
            with patch("sphere_tracker.camera_registry.load_registry",
                       return_value=fake_reg), \
                    patch.object(app.state, _status,
                                 return_value=(False, "**不匹配**")):
                w._refresh_state()
                off_state = tuple(w.btn_record.state())
                off_text = str(w.btn_record.cget("text"))
            with patch("sphere_tracker.camera_registry.load_registry",
                       return_value=fake_reg), \
                    patch.object(app.state, _status, return_value=(True, "")):
                w._refresh_state()
                on_state = tuple(w.btn_record.state())
                on_text = str(w.btn_record.cget("text"))
                on_cfg = str(w.btn_record.cget("state"))
            part = ("disabled" in off_state and "标定" in off_text
                    and "disabled" not in on_state
                    and ("normal" in on_cfg or on_text == "开始录制"))
            gate_ok = gate_ok and part
            gate_detail.append(
                f"{_mode}: off=({off_state},{off_text}) "
                f"on=({on_state},{on_text},{on_cfg})")
        tcheck("B-181 录制门禁：标定不匹配时 btn_record 禁用且文字带「标定」，"
               "匹配时 normal /「开始录制」（由 _refresh_state 同步）",
               gate_ok, "；".join(gate_detail))

        # 采集索引未实测：即使标定就绪，录制也必须禁用，且按钮指向**相机**
        # 而不是错误地写"先解决标定"。
        w = wins["stereo(parallel)"]
        with patch("sphere_tracker.camera_registry.load_registry",
                   return_value=null_cap_reg), \
                patch.object(app.state, "parallel_status", return_value=(True, "")):
            w._refresh_state()
            null_state = tuple(w.btn_record.state())
            null_text = str(w.btn_record.cget("text"))
        tcheck("采集索引未实测 -> 录制禁用，按钮指向「先确认相机」",
               "disabled" in null_state and "先确认相机" in null_text,
               f"state={null_state} text={null_text}")

        # 用户显式填纯数字采集索引 -> 那是他自己的选择，不该被角色门禁误拦。
        w.var_left.set("0")
        w.var_right.set("2")
        try:
            with patch("sphere_tracker.camera_registry.load_registry",
                       return_value=null_cap_reg), \
                    patch.object(app.state, "parallel_status", return_value=(True, "")):
                w._refresh_state()
                num_state = tuple(w.btn_record.state())
                num_text = str(w.btn_record.cget("text"))
            tcheck("显式数字采集索引不被角色 capture_index 门禁误拦",
                   "disabled" not in num_state, f"state={num_state} text={num_text}")
        finally:
            w.var_left.set("left")
            w.var_right.set("right")
            with patch("sphere_tracker.camera_registry.load_registry",
                       return_value=fake_reg):
                w._refresh_state()

        # ③ 无种子会话球种类复位（双目 _set_session / 单目 _apply_seed_kind）。
        # 全部用 tempfile，且双目/单目各用独立子根：单目 scan_mono_sessions 会
        # 扫子根的全部直接子目录，混在一起会互相干扰。
        with tempfile.TemporaryDirectory(prefix="gui_reset_") as _td:
            _root = Path(_td)
            _sr = _root / "stereo"
            _mr = _root / "mono"
            _sr.mkdir()
            _mr.mkdir()

            # 双目：_set_session 是会话选择器的 on_pick 回调，直接调它。
            sw = wins["stereo(parallel)"]
            p_no_seed = _sr / "no_seed"
            p_no_seed.mkdir()
            sw.ball.set_kind(BALL_METAL)
            sw._set_session(str(p_no_seed))
            s_kind = sw.ball.kind()
            s_note = str(sw.status._label.cget("text"))
            s_reset = (s_kind == BALL_PLASTIC and "复位" in s_note)

            p_seeded = _sr / "seeded"
            (p_seeded / "derived").mkdir(parents=True)
            seeds = p_seeded / "derived" / "seeds.json"
            sw.ball.set_kind(BALL_PLASTIC)
            seeds.write_text(_json.dumps({"ball_kind": "metal"}), encoding="utf-8")
            sw._set_session(str(p_seeded))
            s_metal = sw.ball.kind() == BALL_METAL
            sw.ball.set_kind(BALL_METAL)
            seeds.write_text(_json.dumps({"ball_kind": "plastic"}), encoding="utf-8")
            sw._set_session(str(p_seeded))
            s_plastic = sw.ball.kind() == BALL_PLASTIC
            tcheck("双目 _set_session：无 seeds.json → 复位纯色球且状态条含「复位」；"
                   "有 ball_kind=metal/plastic 时按种子恢复（不被复位覆盖）",
                   s_reset and s_metal and s_plastic,
                   f"no_seed={s_kind!r}/{s_reset}, metal={s_metal}, plastic={s_plastic}")

            # 单目：走真实入口 _on_session_picked（内部 _select_session →
            # _apply_seed_kind，并把「复位」说明追加到最终那条状态文本）。
            mono = wins["mono"]
            real_sess = mono._session

            def _make_mono(name: str, ball_kind=None) -> Path:
                d = _mr / name
                (d / "derived").mkdir(parents=True)
                (d / "session.json").write_text(_json.dumps({
                    "schema": "sphere-mono-session", "version": 1,
                    "camera": "main", "width": 1280, "height": 720,
                    "fps": 30.0, "frame_count": 300, "duration_s": 10.0,
                    "encoder": "HFYU", "video": "clip.mkv"},
                    ensure_ascii=False), encoding="utf-8")
                if ball_kind is not None:
                    (d / "derived" / "seed.json").write_text(
                        _json.dumps({"ball_kind": ball_kind}), encoding="utf-8")
                return d

            m_no_seed = _make_mono("no_seed")
            mono.ball.set_kind(BALL_METAL)
            mono._seed_kind_for = None
            mono._on_session_picked(str(m_no_seed))
            m_kind = mono.ball.kind()
            m_note = str(mono.status._label.cget("text"))
            m_reset = (m_kind == BALL_PLASTIC and "复位" in m_note)

            m_metal_d = _make_mono("metal", "metal")
            mono.ball.set_kind(BALL_PLASTIC)
            mono._seed_kind_for = None
            mono._on_session_picked(str(m_metal_d))
            m_metal = mono.ball.kind() == BALL_METAL
            m_plastic_d = _make_mono("plastic", "plastic")
            mono.ball.set_kind(BALL_METAL)
            mono._seed_kind_for = None
            mono._on_session_picked(str(m_plastic_d))
            m_plastic = mono.ball.kind() == BALL_PLASTIC
            tcheck("单目 _apply_seed_kind：无 seed.json → 复位纯色球且状态条含「复位」；"
                   "有 ball_kind=metal/plastic 时按种子恢复（不被复位覆盖）",
                   m_reset and m_metal and m_plastic,
                   f"no_seed={m_kind!r}/{m_reset}, metal={m_metal}, plastic={m_plastic}")
            mono._session = real_sess
            mono._seed_kind_for = None
            mono.ball.set_kind(BALL_PLASTIC)

        # ⑤ L1 预算 / primary 不退化：单目① primary==1 & L1≤5，
        # 「试开这台相机」在 L1、「替代：单点焦距估计」在折叠区；双目④ L1≤5。
        def _buttons(card):
            return [(w, str(w.cget("text"))) for w in walk(card)
                    if w.winfo_class() == "TButton"]

        c1 = next((c for c in step_cards(wins["mono"])
                   if str(c.cget("text"))[:1] == "①"), None)
        mm1 = measure(c1) if c1 is not None else {}
        probe_l1 = est_fold = False
        if c1 is not None:
            probe_l1 = any(t == "试开这台相机" and not _inside_collapsible(b, c1)
                           for b, t in _buttons(c1))
            est_fold = any(t == "替代：单点焦距估计" and _inside_collapsible(b, c1)
                           for b, t in _buttons(c1))
        ok_budget = (c1 is not None and mm1.get("primary") == 1
                     and mm1.get("l1_buttons", 99) <= 5
                     and probe_l1 and est_fold)
        c4 = next((c for c in step_cards(wins["stereo(parallel)"])
                   if str(c.cget("text"))[:1] == "④"), None)
        mm4 = measure(c4) if c4 is not None else {}
        ok_budget4 = c4 is not None and mm4.get("l1_buttons", 99) <= 5
        tcheck("L1 预算/primary 不退化：单目① primary==1 & L1≤5，"
               "「试开这台相机」在 L1、「替代」在折叠区；双目④ L1≤5",
               ok_budget and ok_budget4,
               f"mono①={mm1}/probe_l1={probe_l1}/est_fold={est_fold}, "
               f"stereo④={mm4}")
    finally:
        try:
            app.destroy()
        except Exception:                                   # noqa: BLE001
            pass

    return _summary(results)


if __name__ == "__main__":
    raise SystemExit(main())
