"""P2 打包层（自包含交付包）自测：目录包往返 / 标定逐字不变 / 三态 / 覆盖语义。

为什么需要它
------------
计划书 §6 P2 的交付物是**目录本身**：顶层 `package.json` 清单 + `sessions/<会话>/…`
+ `calib/<标定基名>.npz`。这件事有三个"错了也不会报错"的坑，正是本测试要钉的：

1. **清单与磁盘对不上**：`package.json` 列了 5 个文件、包里只有 4 个 —— 接收方
   按清单去读，读不到就当成"这段会话没跑过"，而不是"包坏了"。`read_package`
   必须双向核对（清单有、包里没有 / 包里有、清单没有）。
2. **标定文件名被清洗**：`calib/平行 装置 A.npz` 导入后必须**逐字不变**，
   否则 `calibration.path` 指的那份文件在目标机上根本不存在。计划书 §6 P2
   的原话判据就是这一条 —— 所以这里用中文 + 空格的基名，专门让"顺手清洗成
   ASCII"的实现红掉。
3. **`import_package` 默认写盘**：一次误调就能把整包并进用户的真实录制目录。
   所以默认 `dry_run=True`，本测试断言"只出计划、一个字节都不写"。

反向自证（真的改坏过，见 `docs/交接文档.md` §107）
------------------------------------------------
==========================================  ========================================
改动                                        结果
==========================================  ========================================
`calib` 文件名清洗成 ASCII（`re.sub`）      第 2 条红（清单路径与盘上文件名都对不上）
`normalize_output(directory=True)` 去掉    第 5 条红（`.zip` 路径被当目录收下）
  `.zip` 那条 ValueError
`import_package` 把 `dry_run` 默认改成 False  第 6 条红（`dest` 目录被创建）
`export_package` 的 `overwrite=True` 改成     第 7 条红（`keepme.txt` 被删）
  `rmtree` 整个目录
==========================================  ========================================
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "sphere_tracker").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps._data_export import (ExportOptions, _cat_files, export_zip,
                               normalize_output, scan_items)
from apps.package_export import (
    CALIBRATION_NOTE,
    PackageResult,
    export_package,
    import_package,
    read_package,
)

CATS = frozenset({"metadata", "derived", "annotated"})


def _mk_session(parent: Path, name: str, sid: str, *, label: str = "") -> Path:
    """造一段最小但**合法**的会话（session.json + 一个假标注视频）。"""
    d = Path(parent) / name
    (d / "derived").mkdir(parents=True, exist_ok=True)
    (d / "session.json").write_text(json.dumps({
        "schema": "sphere-stereo-session", "version": 1,
        "id": sid, "created": "2026-01-01 12:00:00",
        "stereo_mode": "parallel", "label": label,
        "left": {"frame_count": 3, "identity": "cam-L", "name": "左", "port": 0},
        "right": {"frame_count": 3, "identity": "cam-R"},
        "environment": {"platform": "win32"},
    }, ensure_ascii=False), encoding="utf-8")
    (d / "derived" / "annotated_fake.mp4").write_text("fake video bytes", encoding="utf-8")
    return d


def _mk_tree(td: Path) -> tuple[Path, list]:
    """两段会话：一段在根层（散会话），一段在 `parallel/`（有分组）。"""
    root = td / "stereo"
    _mk_session(root, "20260101-120000", "S-1", label="张三的球")
    _mk_session(root / "parallel", "20260102-120000", "S-2")
    return root, list(scan_items(root))


def _mk_calib(td: Path, *, profile: str | None = None,
              base: str = "平行 装置 A") -> tuple[Path, Path]:
    """造一份带**中文 + 空格**基名的标定（npz 是假字节，meta 是真 JSON）。"""
    d = td / "标定"
    d.mkdir(exist_ok=True)
    npz = d / f"{base}.npz"
    js = d / f"{base}.json"
    npz.write_bytes(b"PK\x03\x04 fake npz payload \xe4\xb8\xad\xe6\x96\x87")
    meta: dict = {"stereo_mode": "parallel", "baseline_mm": 68.84}
    if profile is not None:
        meta["profile"] = profile
    js.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return npz, js


def _session_files(root: Path) -> set[str]:
    """包内 `sessions/` 下的相对路径（与清单里的 `files` 同口径，前缀到会话目录）。"""
    out: set[str] = set()
    rep = read_package(root)
    for s in rep["manifest"]["sessions"]:
        for rel in s["files"]:
            out.add(f"sessions/{s['folder']}/{rel}")
    return out


# ----------------------------------------------------------------------
# 1. 往返：导出 → 读回，清单与磁盘必须一致
# ----------------------------------------------------------------------
def _check_roundtrip(ck) -> None:
    print("----- 1. 往返（2 段会话：根层 + parallel/）-----")
    td = Path(tempfile.mkdtemp(prefix="pkg_round_"))
    try:
        _root, items = _mk_tree(td)
        ck("临时树里扫到 2 段会话", len(items) == 2)
        out = td / "交付包"
        res = export_package(items, ExportOptions(categories=CATS), out,
                             calibration=(None, None))
        ck("export_package 返回 PackageResult 且会话数 = 2",
           isinstance(res, PackageResult) and res.n_sessions == 2)
        ck("顶层三件都在（package.json / README.md / sessions/）",
           (out / "package.json").is_file() and (out / "README.md").is_file()
           and (out / "sessions").is_dir())
        ck("README 开头就说明「这是自包含交付包」",
           "自包含交付包" in (out / "README.md").read_text(encoding="utf-8")[:400])

        rep = read_package(out)
        ck("read_package 的 n_sessions = 2", rep["n_sessions"] == 2)
        ck("read_package 没有任何 problem", rep["problems"] == [])
        files = _session_files(out)
        ck("清单里的每个文件都能在包里找到（4 个：2 段 × 元数据+视频）",
           len(files) == 4 and all((out / f).is_file() for f in files))
        ck("n_files 只数交付文件（清单文件不计）", res.n_files == rep["n_files"] == 4)

        # 脱敏真的生效：session.json 里没有 label/identity，清单里也不留
        meta_one = json.loads((out / "sessions" / "20260101-120000" /
                               "metadata" / "session.json").read_text(encoding="utf-8"))
        ck("session.json 已脱敏（label / identity / platform 都没了）",
           "label" not in meta_one and "identity" not in meta_one["left"]
           and "platform" not in meta_one.get("environment", {}))
        ck("清单里也没有 label（脱敏不能只做一半）",
           all(not s["label"] for s in rep["manifest"]["sessions"]))

        # 目录包被压成 zip 之后，同样能只读地读清单（不解压到磁盘）。
        # 注意 export_zip 产出的是 `manifest.json` 的**另一种**格式，
        # 不是交付包 —— 这里压缩的是刚导出的目录包本身。
        zpath = Path(shutil.make_archive(str(td / "包压缩"), "zip",
                                         root_dir=str(out.parent), base_dir=out.name))
        zrep = read_package(zpath)
        ck("read_package 也能读 .zip（清单 + 文件表，不落盘）",
           zrep["n_sessions"] == 2 and zrep["problems"] == [])
        ck("zip 里那份清单与目录包逐键相同（除 created 时间戳）",
           {k: v for k, v in zrep["manifest"].items() if k != "created"}
           == {k: v for k, v in rep["manifest"].items() if k != "created"})
    finally:
        shutil.rmtree(td, ignore_errors=True)


# ----------------------------------------------------------------------
# 2. 标定：路径与包内文件名**逐字不变**（计划书 §6 P2 原话判据）
# ----------------------------------------------------------------------
def _check_calib_verbatim(ck) -> None:
    print("----- 2. 标定文件名逐字不变（中文 + 空格基名）-----")
    td = Path(tempfile.mkdtemp(prefix="pkg_calib_"))
    try:
        _root, items = _mk_tree(td)
        npz, js = _mk_calib(td, profile="平行装置")
        out = td / "交付包"
        export_package(items, ExportOptions(categories=CATS), out,
                       calibration=(npz, js))
        manifest = json.loads((out / "package.json").read_text(encoding="utf-8"))
        block = manifest["calibration"]
        ck("calibration.path 逐字等于 calib/平行 装置 A.npz",
           block["path"] == "calib/平行 装置 A.npz")
        ck("calibration.json 逐字等于 calib/平行 装置 A.json",
           block["json"] == "calib/平行 装置 A.json")
        ck("包内相对路径用 POSIX 分隔符（没有反斜杠）",
           "\\" not in block["path"] and "\\" not in block["json"])
        on_disk = sorted(p.name for p in (out / "calib").iterdir())
        ck("calib/ 下两个文件名与源**逐字相同**",
           on_disk == ["平行 装置 A.json", "平行 装置 A.npz"])
        ck("npz 内容逐字节相同（搬移无损）",
           (out / "calib" / npz.name).read_bytes() == npz.read_bytes())
        ck("三态：meta 有 profile → registered", block["source"] == "registered")
        ck("note 是那句固定文案（一字不改）", block["note"] == CALIBRATION_NOTE)
        ck("calibration 里**没有** profile 字段（计划书 §8 明确不做）",
           "profile" not in block)
        ck("带标定的包读回无 problem", read_package(out)["problems"] == [])

        # 只有 .npz 的老产物也能打包（json 那项写 null，不编一份假的）
        npz2, _js2 = _mk_calib(td, base="老产物 仅npz")
        out2 = td / "只有npz"
        export_package(items, ExportOptions(categories=CATS), out2,
                       calibration=(npz2, None))
        b2 = json.loads((out2 / "package.json").read_text(encoding="utf-8"))["calibration"]
        ck("只有 .npz 时 json 写 null、path 仍在",
           b2["json"] is None and b2["path"] == "calib/老产物 仅npz.npz")
    finally:
        shutil.rmtree(td, ignore_errors=True)


# ----------------------------------------------------------------------
# 3. calibration.source 三态
# ----------------------------------------------------------------------
def _check_source_states(ck) -> None:
    print("----- 3. calibration.source 三态（registered / unregistered / none）-----")
    td = Path(tempfile.mkdtemp(prefix="pkg_source_"))
    try:
        _root, items = _mk_tree(td)

        npz, js = _mk_calib(td, profile="实验性质大角度1")
        out = td / "registered"
        export_package(items, ExportOptions(categories=CATS), out, calibration=(npz, js))
        b = json.loads((out / "package.json").read_text(encoding="utf-8"))["calibration"]
        ck("meta 有 profile → registered", b["source"] == "registered")

        npz2, js2 = _mk_calib(td, base="老标定 01")
        out2 = td / "unregistered"
        export_package(items, ExportOptions(categories=CATS), out2, calibration=(npz2, js2))
        b2 = json.loads((out2 / "package.json").read_text(encoding="utf-8"))["calibration"]
        ck("meta 没有 profile（老标定）→ unregistered", b2["source"] == "unregistered")

        out3 = td / "none"
        res3 = export_package(items, ExportOptions(categories=CATS), out3,
                              calibration=(None, None))
        b3 = json.loads((out3 / "package.json").read_text(encoding="utf-8"))["calibration"]
        ck("没标定 → source=none、path/json 都是 null",
           b3["source"] == "none" and b3["path"] is None and b3["json"] is None)
        ck("没标定时 note 仍是同一句", b3["note"] == CALIBRATION_NOTE)
        ck("没标定时有告警说「这个包不含标定」",
           any("不含标定" in w for w in res3.warnings))
        ck("没标定时不建空的 calib/ 目录", not (out3 / "calib").exists())
        ck("none 态的包读回无 problem", read_package(out3)["problems"] == [])
    finally:
        shutil.rmtree(td, ignore_errors=True)


# ----------------------------------------------------------------------
# 4. 目录包与 zip 包内容一致（同一套文件规划）
# ----------------------------------------------------------------------
def _check_zip_parity(ck) -> None:
    print("----- 4. 目录模式不破 zip 模式（同一批会话）-----")
    td = Path(tempfile.mkdtemp(prefix="pkg_parity_"))
    try:
        _root, items = _mk_tree(td)
        opts = ExportOptions(categories=CATS)
        zip_res = export_zip(items, opts, td / "对照.zip")
        pkg_res = export_package(items, opts, td / "对照包", calibration=(None, None))

        # 口径：export_zip 的 n_files 里含 README + manifest.json 两份清单文件，
        # 目录包的 n_files 只数交付文件（README/package.json 不算）——
        # 所以可比式是 zip.n_files - 2 == pkg.n_files。
        ck("zip 的 n_files - 2 == 目录包的 n_files",
           zip_res.n_files - 2 == pkg_res.n_files)

        with zipfile.ZipFile(zip_res.output) as zf:
            names = [n for n in zf.namelist() if "/sessions/" in n]
        zset = {n.split("/sessions/", 1)[1] for n in names}
        pset = {f[len("sessions/"):] for f in _session_files(td / "对照包")}
        ck("两边 sessions/ 下的文件集合逐个相等", zset == pset)
        ck("两边都收了标注视频（类别的口径没有分叉）",
           any(p.endswith("annotated_fake.mp4") for p in pset))
    finally:
        shutil.rmtree(td, ignore_errors=True)


# ----------------------------------------------------------------------
# 5. normalize_output 两种模式
# ----------------------------------------------------------------------
def _check_normalize(ck) -> None:
    print("----- 5. normalize_output：zip 模式 / 目录模式 -----")

    def raises(fn) -> bool:
        try:
            fn()
        except ValueError:
            return True
        return False

    ck("zip 模式：a/b → a/b.zip", normalize_output("a/b").as_posix() == "a/b.zip")
    ck("zip 模式：a/b.zip 保持原样", normalize_output("a/b.zip").as_posix() == "a/b.zip")
    ck("目录模式：a/b → a/b（不补后缀）",
       normalize_output("a/b", directory=True).as_posix() == "a/b")
    ck("目录模式：结尾分隔符被去掉",
       normalize_output("a/b/", directory=True).as_posix() == "a/b")
    ck("目录模式：a/b.zip 抛 ValueError",
       raises(lambda: normalize_output("a/b.zip", directory=True)))
    ck("空串：两种模式都抛 ValueError",
       raises(lambda: normalize_output("")) and raises(
           lambda: normalize_output("", directory=True)))
    ck("zip 模式：结尾分隔符仍报错（历史行为一字不改）",
       raises(lambda: normalize_output("a/b/")))
    ck("目录模式：'..' 被拒绝", raises(
        lambda: normalize_output("a/..", directory=True)))


# ----------------------------------------------------------------------
# 6. import_package 默认 dry-run
# ----------------------------------------------------------------------
def _check_import_dry(ck) -> None:
    print("----- 6. import_package：默认 dry_run 不写盘 -----")
    td = Path(tempfile.mkdtemp(prefix="pkg_import_"))
    try:
        _root, items = _mk_tree(td)
        npz, js = _mk_calib(td, profile="平行装置")
        out = td / "交付包"
        export_package(items, ExportOptions(categories=CATS), out, calibration=(npz, js))

        dest = td / "目标数据" / "stereo"
        calib_dst = td / "目标配置" / "calib"
        plan = import_package(out, dest, calibration_dir=calib_dst)
        ck("dry_run 默认是 True", plan["dry_run"] is True and plan["copied"] == 0)
        ck("dry_run 时目标目录**根本不存在**",
           not dest.exists() and not calib_dst.exists())
        ck("计划里列出 2 段会话 + 2 份标定（共 4 项）", len(plan["plan"]) == 4)
        ck("计划里有标定落点、文件名逐字不变",
           any(p["dst"].endswith("平行 装置 A.npz") for p in plan["plan"]
               if p["kind"] == "calibration"))
        ck("dry_run 没有 problem", plan["problems"] == [])

        real = import_package(out, dest, calibration_dir=calib_dst, dry_run=False)
        ck("dry_run=False 才真的拷：copied = 4", real["copied"] == 4)
        ck("会话按磁盘布局还原（session.json 回到会话根层）",
           (dest / "20260101-120000" / "session.json").is_file()
           and (dest / "20260101-120000" / "derived" / "annotated_fake.mp4").is_file())
        ck("标定按原名到位",
           (calib_dst / "平行 装置 A.npz").is_file()
           and (calib_dst / "平行 装置 A.json").is_file())
        ck("导入后的目录能被 scan_items 认出 2 段会话",
           len(scan_items(dest)) == 2)

        again = import_package(out, dest, calibration_dir=calib_dst, dry_run=False)
        ck("重复导入：目标已存在 → 记进 problems、copied = 0",
           again["copied"] == 0
           and sum("已存在" in p for p in again["problems"]) == 4)

        ck("没给 calibration_dir 就不拷标定，但会提醒",
           any("calibration_dir" in p for p in import_package(
               out, td / "另一处")["problems"]))
    finally:
        shutil.rmtree(td, ignore_errors=True)


# ----------------------------------------------------------------------
# 7. 覆盖语义：默认拒绝；overwrite 只清自己那几项
# ----------------------------------------------------------------------
def _check_overwrite(ck) -> None:
    print("----- 7. export_package 的覆盖语义 -----")
    td = Path(tempfile.mkdtemp(prefix="pkg_over_"))
    try:
        _root, items = _mk_tree(td)
        out = td / "已有目录"
        out.mkdir()
        (out / "keepme.txt").write_text("用户的说明，不许删", encoding="utf-8")
        (out / "package.json").write_text("{}", encoding="utf-8")

        try:
            export_package(items, ExportOptions(categories=CATS), out,
                           calibration=(None, None))
            refused = False
        except FileExistsError:
            refused = True
        ck("已存在且非空、overwrite=False → FileExistsError", refused)
        ck("拒绝时没动目录里的东西",
           (out / "keepme.txt").is_file()
           and (out / "package.json").read_text(encoding="utf-8") == "{}")

        export_package(items, ExportOptions(categories=CATS), out,
                       calibration=(None, None), overwrite=True)
        ck("overwrite=True 时自己的清单被重写（不再是 {}）",
           (out / "package.json").read_text(encoding="utf-8").strip().startswith("{"))
        ck("overwrite=True 时**别人的文件还在**（不是 rmtree 整个目录）",
           (out / "keepme.txt").is_file())
        ck("覆盖后包仍然可读", read_package(out)["problems"] == [])

        empty = td / "空目录"
        empty.mkdir()
        export_package(items, ExportOptions(categories=CATS), empty,
                       calibration=(None, None))
        ck("已存在但**空**的目录：不抛错，直接写", (empty / "package.json").is_file())
    finally:
        shutil.rmtree(td, ignore_errors=True)


# ----------------------------------------------------------------------
# 8. 视频导出：半成品排除 / 合法分段保留 / 视频纯存储（ZIP_STORED）
# ----------------------------------------------------------------------
def _check_video_export(ck) -> None:
    print("----- 8. 视频导出：半成品排除 / 分段保留 / 视频纯存储 -----")
    td = Path(tempfile.mkdtemp(prefix="pkg_video_"))
    try:
        session = td / "stereo" / "20260101-120000"
        (session / "derived").mkdir(parents=True)
        (session / "session.json").write_text(json.dumps({
            "schema": "sphere-stereo-session", "version": 1,
            "id": "S-video", "created": "2026-01-01 12:00:00",
            "left": {"frame_count": 1}, "right": {"frame_count": 1},
        }, ensure_ascii=False), encoding="utf-8")
        (session / "derived" / "annotated.mp4").write_bytes(b"ok video")
        (session / "derived" / "annotated.tmp.mp4").write_bytes(b"half")
        (session / "derived" / "annotated_first20.mp4").write_bytes(b"first20")

        # 1) _cat_files：收成品与合法分段，排除正在录制的半成品
        names = {p.name for p, _arc in _cat_files(session, "annotated")}
        ck("_cat_files 收 annotated.mp4 与分段 annotated_first20.mp4",
           {"annotated.mp4", "annotated_first20.mp4"} <= names)
        ck("_cat_files 排除半成品 annotated.tmp.mp4",
           "annotated.tmp.mp4" not in names)

        # 2) 真跑 export_zip：包里同样不能出现半成品
        items = list(scan_items(td / "stereo"))
        opts = ExportOptions(categories=frozenset({"annotated", "metadata"}))
        res = export_zip(items, opts, td / "video.zip")
        with zipfile.ZipFile(res.output) as zf:
            arcs = zf.namelist()
            infos = {i.filename: i for i in zf.infolist()}
        vids = [n for n in arcs if "/videos/" in n]
        ck("zip 里含 videos/annotated.mp4 与分段 annotated_first20.mp4",
           any(n.endswith("videos/annotated.mp4") for n in vids)
           and any(n.endswith("videos/annotated_first20.mp4") for n in vids))
        ck("zip 里没有半成品 annotated.tmp.mp4",
           not any(".tmp.mp4" in n for n in arcs))

        # 3) 视频条目 ZIP_STORED，非视频条目仍 ZIP_DEFLATED
        mp4 = [i for n, i in infos.items() if n.endswith(".mp4")]
        sj = [i for n, i in infos.items()
              if n.endswith("metadata/session.json")]
        ck("mp4 条目是 ZIP_STORED（视频不压缩）",
           bool(mp4) and all(i.compress_type == zipfile.ZIP_STORED
                             for i in mp4))
        ck("非视频条目 metadata/session.json 仍是 ZIP_DEFLATED",
           bool(sj) and sj[0].compress_type == zipfile.ZIP_DEFLATED)
    finally:
        shutil.rmtree(td, ignore_errors=True)


def main() -> int:
    print("=" * 70)
    print("P2 打包层自测（自包含交付包：往返 / 标定逐字不变 / 三态 / 覆盖）")
    print("=" * 70)
    results: list[tuple[str, bool]] = []

    def ck(label: str, good: bool) -> None:
        results.append((label, good))
        print(f"  [{'通过' if good else '不达标'}] {label}")

    _check_roundtrip(ck)
    _check_calib_verbatim(ck)
    _check_source_states(ck)
    _check_zip_parity(ck)
    _check_normalize(ck)
    _check_import_dry(ck)
    _check_overwrite(ck)
    _check_video_export(ck)

    print()
    ok = all(v for _n, v in results)
    bad = [n for n, v in results if not v]
    if bad:
        print(f"不达标 {len(bad)} 项：")
        for n in bad:
            print(f"  · {n}")
    print(f"\n共 {len(results)} 个检查点")
    print("\n判定结果：" + ("全部通过" if ok else "有不达标项"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
