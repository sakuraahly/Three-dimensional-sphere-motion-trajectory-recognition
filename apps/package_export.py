"""P2 打包层：**目录形态**的自包含交付包（写盘 / 读回校验 / 导入计划）。

为什么单独一层
--------------
``_data_export.export_zip`` 交付的是"一个压缩包"：接收方得先解压，还得信压缩包
没坏；用户也没法在资源管理器里直接翻。计划书 §6 P2 要的是**目录本身就是交付物**：
顶层一份 ``package.json`` 清单 + ``sessions/<会话>/…`` + ``calib/<标定基名>.npz``。

三条不许自作主张的规则（计划书 §8 / §9 第 8 条）
------------------------------------------------
1. **不加 ``profile`` 字段**。装置档案的身份问题归另一批修复；打包层只记
   "标定来自哪个文件 + 是否登记过"。
2. ``calibration.source`` 三态（``registered`` / ``unregistered`` / ``none``）
   **只走标定层的判定**（``stereo.profiles``），不在这里 ``json.loads`` 那份 meta
   再挑字段 —— 两份判据迟早漂。
3. ``calibration.note`` 固定是「sha256 只证搬移无损、不证来源」，一字不改。

复用而不是重写
--------------
文件规划（哪些类别收哪些文件）、目录名清洗与去重、``session.json`` 脱敏，
**全部来自** ``_data_export``（``_plan_files`` / ``_unique_folders`` /
``redact_session``）。目录包与 zip 包各写一份的后果不是多几行代码，而是两边交出的
内容会悄悄分叉（某天只给 zip 补了 ``derived/plots``，目录包就少一类文件，而两边
都不会报错）。

冷启动
------
顶层**不 import** ``stereo.profiles``：它会拖进 cv2 + scipy（``import apps.data_manager``
0.06 s → 1.5 s）。需要时在函数内 import（与 ``apps/_common.current_calibration``
同一做法）。
"""

from __future__ import annotations

import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, NamedTuple

from ._data_export import (
    ExportOptions,
    SessionItem,
    _plan_files,
    _read_session,
    _readme,
    _unique_folders,
    normalize_output,
    redact_session,
)

PACKAGE_SCHEMA = "sphere-package"
PACKAGE_VERSION = 1
GENERATOR = "sphere_tracker"
#: 固定文案。**一字不改**：它说的是 sha256 能证什么、不能证什么 ——
#: 改成"已校验来源"之类的说法，等于替一份没登记过的标定背书。
CALIBRATION_NOTE = "sha256 只证搬移无损、不证来源"
#: ``overwrite=True`` 时**只清这些顶层项**（见 ``_clear_own_entries``）。
OWN_TOP_ENTRIES = ("package.json", "README.md", "sessions", "calib")
SOURCE_STATES = ("registered", "unregistered", "none")


class PackageResult(NamedTuple):
    """一次目录包导出的结果。

    ``n_files`` 只数**交付文件**（``sessions/`` 下 + ``calib/`` 下），
    ``README.md`` / ``package.json`` 这两份清单文件不算 —— 与
    ``export_zip`` 的 ``n_files``（= 会话文件数 + 2）对齐后可比：
    ``zip.n_files - 2 == pkg.n_files``。判定见 ``tests/selftest_package.py`` 第 4 条。
    """

    root: Path
    manifest: dict
    n_sessions: int
    n_files: int
    total_bytes: int
    warnings: list[str]


# --------------------------------------------------------------------------- #
# 标定：从哪来、算不算"登记过"
# --------------------------------------------------------------------------- #
def _base_name(path: Path) -> str:
    """标定基名（去掉 ``.npz`` / ``.json`` 后缀，**其余逐字保留**）。

    中文、空格、点号都不清洗：``calib/<基名>.npz`` 里的文件名必须与源文件
    一字不差，导入方才能"照原名拷回去"（计划书 §6 P2 的原话判据）。
    """
    name = Path(path).name
    for ext in (".npz", ".json"):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return name


def _resolve_calibration(calibration
                         ) -> tuple[Path | None, Path | None, str | None, list[str]]:
    """决定这个包带哪份标定 → ``(npz, json, 基名, 告警)``。

    ``calibration=None``（默认）跟随**当前装置档案**：
    ``current_calibration(need_existing=False)`` —— 注意是 ``need_existing=False``，
    没标定也要能打包（此时两块都是 ``None``，清单写 ``source: "none"``）。
    显式传 ``(npz, json)``（测试用）时任一侧可给 ``None``；不存在的文件直接丢掉
    并告警，不写进清单 —— 清单里指着一个不存在的文件比不写更坏。
    """
    warnings: list[str] = []
    if calibration is None:
        from apps._common import current_calibration

        base_path, _why = current_calibration(need_existing=False)
        npz = base_path.with_name(base_path.name + ".npz")
        json_path = base_path.with_name(base_path.name + ".json")
        return (npz if npz.is_file() else None,
                json_path if json_path.is_file() else None,
                base_path.name, warnings)

    raw_npz, raw_json = calibration
    npz = Path(raw_npz) if raw_npz else None
    json_path = Path(raw_json) if raw_json else None
    if npz is not None and not npz.is_file():
        warnings.append(f"指定的标定 npz 不存在，未打包：{npz}")
        npz = None
    if json_path is not None and not json_path.is_file():
        warnings.append(f"指定的标定 json 不存在，未打包：{json_path}")
        json_path = None
    base = _base_name(npz or json_path) if (npz or json_path) else None
    if npz is not None and json_path is not None and _base_name(npz) != _base_name(json_path):
        warnings.append(
            f"标定两份文件的基名不同（{_base_name(npz)} / {_base_name(json_path)}）；"
            f"包内统一用 {_base_name(npz)}")
        base = _base_name(npz)
        json_path = None if _base_name(json_path) != base else json_path
    return npz, json_path, base, warnings


def _calibration_source(npz: Path | None, json_path: Path | None) -> str:
    """这份标定的 ``source``：``registered`` / ``unregistered`` / ``none``。

    判据**只走标定层**（计划书 §9 第 8 条），不自己解析 meta：

    - 先看它是否落在某套装置档案名下 —— 用
      ``stereo.profiles.calibration_specs(profile).bound_profile``
      （``bound_profile`` 就是 meta 里的 ``profile``，空串 = 未登记；
      ``calibration_specs`` 是"标定 meta 结构化摘要"的唯一产地）；
    - 不在任何档案名下（调用方显式传了外部/临时标定）→ 仍不 ``json.loads``
      挑字段，用同层的 ``calibration_identity(meta, None)`` 判定，meta 由
      ``stereo.profiles`` **自己的读取器**给出。
    """
    base = npz or json_path
    if base is None:
        return "none"
    from stereo import profiles                     # 函数内 import：别拖 cv2/scipy

    base = Path(base).with_suffix("")
    for prof in profiles.list_profiles():
        try:
            specs = profiles.calibration_specs(prof)
        except profiles.ProfileError:
            continue
        if specs.path is not None and Path(specs.path) == base:
            return "registered" if specs.bound_profile else "unregistered"
    # 标定层自己的 meta 读取器（不是本模块的 json.loads）
    meta = profiles._read_calibration_meta(base)    # noqa: SLF001
    ident = profiles.calibration_identity(meta, None)
    return "unregistered" if ident.state == profiles.IDENTITY_UNREGISTERED else "registered"


def _calibration_block(npz: Path | None, json_path: Path | None, base: str | None,
                       warnings: list[str]) -> dict:
    """``package.json`` 的 ``calibration`` 块。**键名与取值照计划书，别自创。**"""
    if npz is None and json_path is None:
        warnings.append("这个包不含标定：calibration.source 是 none，接收方得自己标定")
        return {"path": None, "json": None, "source": "none", "note": CALIBRATION_NOTE}
    return {
        # 包内相对路径 + POSIX 分隔符；文件名取标定基名**原样**
        "path": f"calib/{_base_name(npz)}.npz" if npz is not None else None,
        "json": f"calib/{_base_name(json_path)}.json" if json_path is not None else None,
        "source": _calibration_source(npz, json_path),
        "note": CALIBRATION_NOTE,
    }


# --------------------------------------------------------------------------- #
# 导出
# --------------------------------------------------------------------------- #
def _clear_own_entries(out: Path) -> None:
    """覆盖前只清**本函数自己会写的顶层项**，绝不 ``rmtree`` 整个目录。

    用户完全可能把输出指到一个已有目录（``D:/交付/第 3 批``），里面还有他自己的
    说明文档、照片、上一版手工整理的表。``rmtree`` 会一起删掉 —— 这种事不能靠
    "用户应该挑个空目录"来防。
    """
    for name in OWN_TOP_ENTRIES:
        p = out / name
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p)
        elif p.exists() or p.is_symlink():
            p.unlink()


def _package_readme(items: list[SessionItem], options: ExportOptions,
                    block: dict, warnings: list[str]) -> str:
    """README = 「这是自包含交付包 + 怎么导入」+ ``_data_export._readme`` 原文。

    后半段**不重写**：类别说明、脱敏口径、变量字典全在 ``_readme`` 里，两边共用
    一份口径。前半段只补目录包特有的两件事 —— 清单文件名、``calib/`` 的落点。
    """
    if block["source"] == "none":
        calib_line = ("- 标定：**本包不含标定**（`calibration.source = \"none\"`）——"
                      "接收方要用自己的标定，或先给这套装置标定一次。")
    else:
        files = [p for p in (block["path"], block["json"]) if p]
        calib_line = (f"- 标定：{('、'.join(f'`{p}`' for p in files))}"
                      f"（`source = \"{block['source']}\"`），导入时按**原名**拷到目标机"
                      f"的标定目录。\n  {block['note']} —— 别把它当成\"来源已核实\"。")
    head = f"""> **这是自包含交付包（目录形态）。** 整个目录就是交付物：不解压、不联网、
> 不依赖本仓库也能看懂。`package.json` 是机器读的清单（会话、类别、文件、标定
> 来源），本文件是给人看的说明。
>
> **怎么导入**
>
> 1. 先看计划：`python -c "from apps.package_export import import_package;
>    print(import_package(r'<这个目录>', r'<数据根>/stereo', dry_run=True))"`
>    —— 默认 `dry_run=True`，**一个字节都不写**。
> 2. 确认清单无误后再 `dry_run=False`；目标已存在的会话会被记进 `problems` 并跳过，
>    不会覆盖你现有的数据。
> 3. 标定落点用 `calibration_dir=` 显式给；不给就不拷（不猜落点）。
> 4. 手工导入也行：把 `sessions/<目录名>/` 整个目录拷到 `data/stereo/` 下即可，
>    目录名（时间戳）就是界面上的排序键。
>
> 本包内容：{len(items)} 段会话、清单 `package.json`、`sessions/`（分类文件）、
> `calib/`（标定，若有）。
{calib_line}

---

"""
    text = head + _readme(items, options)
    if warnings:
        text += "\n## 打包时的告警\n\n" + "\n".join(f"- {w}" for w in warnings) + "\n"
    return text


def export_package(items: Iterable[SessionItem], options: ExportOptions,
                   output: str | Path, *, calibration=None,
                   progress: Callable[[int, int, str], None] | None = None,
                   overwrite: bool = False) -> PackageResult:
    """把选中的会话写成一个**目录交付包**（布局与 ``package.json`` 见模块头）。

    ``output`` 走 ``normalize_output(output, directory=True)``：给 ``.zip`` 直接报错，
    不替用户改名。``calibration=None`` 跟随当前装置档案；传 ``(npz, json)`` 显式
    指定（任一侧可 ``None``）。

    已存在且非空时：``overwrite=False`` 抛 ``FileExistsError``（与 ``export_zip``
    一致）；``overwrite=True`` **只清本函数自己会写的那几项**，目录里别的东西不动。
    """
    options = options.normalized()
    items = list(items)
    if not items:
        raise ValueError("没有选择任何会话")
    out = normalize_output(output, directory=True)

    if "metadata" in options.categories:
        bad = [it.key for it in items if not _read_session(it.path)]
        if bad:
            raise ValueError(
                "这些会话缺/坏 session.json：" + "、".join(bad) +
                "；取消 metadata 类别或先修好会话")

    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise FileExistsError(f"输出目录已存在且非空：{out}")
        _clear_own_entries(out)
    out.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    planned = _plan_files(items, options)
    folders = _unique_folders(items)
    npz, json_path, base, calib_warnings = _resolve_calibration(calibration)
    warnings.extend(calib_warnings)
    block = _calibration_block(npz, json_path, base, warnings)
    calib_sources = [p for p in (npz, json_path) if p is not None]

    file_count = sum(len(rows) for _it, rows in planned) + len(calib_sources) + 2
    written = 0
    total_bytes = 0
    session_rows: list[dict] = []

    def tick(label: str) -> None:
        nonlocal written
        written += 1
        if progress:
            progress(written, file_count, label)

    readme = _package_readme(items, options, block, list(warnings))
    readme_bytes = readme.encode("utf-8")
    (out / "README.md").write_bytes(readme_bytes)
    total_bytes += len(readme_bytes)
    tick("README.md")

    for it, rows in planned:
        folder = folders[it.key]
        sess_dir = out / "sessions" / folder
        files: list[str] = []
        for src, rel in rows:
            if src == it.path / "session.json":
                if "metadata" not in options.categories:
                    continue
                data = _read_session(it.path)
                if not data:
                    raise ValueError(f"{it.key} 的 session.json 不可读")
                payload = json.dumps(redact_session(data, options),
                                     ensure_ascii=False, indent=2).encode("utf-8")
                target = sess_dir / "metadata" / "session.json"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
                total_bytes += len(payload)
                files.append("metadata/session.json")
                tick(f"sessions/{folder}/metadata/session.json")
                continue
            if not src.is_file():
                continue
            target = sess_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
            total_bytes += target.stat().st_size
            files.append(rel)
            tick(f"sessions/{folder}/{rel}")
        session_rows.append({
            "sid": it.sid,
            "folder": folder,
            "group": it.group or None,
            # 勾了脱敏就**不写**这两项：清单落盘后是要交出去的，session.json 里
            # 删掉了 label/notes、清单里又留着，等于脱敏只做了一半。
            "label": "" if options.redact_labels else it.label,
            "notes": "" if options.redact_labels else it.notes,
            # 没记录就写 null（与 zip 包口径一致）：写 "parallel" 会让接收方
            # 拿平行几何去解 90° 的数据，深度系统性偏掉还不报错。
            "stereo_mode": it.stereo_mode or None,
            "files": files,
        })

    for p in calib_sources:
        calib_dir = out / "calib"
        calib_dir.mkdir(parents=True, exist_ok=True)
        target = calib_dir / p.name          # 文件名逐字不变
        shutil.copy2(p, target)
        total_bytes += target.stat().st_size
        tick(f"calib/{p.name}")

    manifest = {
        "schema": PACKAGE_SCHEMA,
        "version": PACKAGE_VERSION,
        "created": datetime.now().isoformat(timespec="seconds"),
        "generator": GENERATOR,
        "options": {
            "categories": sorted(options.categories),
            "redact_camera": bool(options.redact_camera),
            "redact_labels": bool(options.redact_labels),
            "redact_platform": bool(options.redact_platform),
        },
        "sessions": session_rows,
        "calibration": block,
    }
    payload = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    # 清单**最后**写：目录里出现 package.json 时，它列的每个文件都已经落盘。
    (out / "package.json").write_bytes(payload.encode("utf-8"))
    tick("package.json")

    n_files = sum(len(row["files"]) for row in session_rows) + len(calib_sources)
    return PackageResult(root=out, manifest=manifest, n_sessions=len(items),
                         n_files=n_files, total_bytes=total_bytes,
                         warnings=warnings)


# --------------------------------------------------------------------------- #
# 读回与校验（只读）
# --------------------------------------------------------------------------- #
def _report(manifest: dict | None, problems: list[str], n_sessions: int,
            n_files: int) -> dict:
    return {"manifest": manifest, "problems": problems,
            "n_sessions": n_sessions, "n_files": n_files}


def read_package(root: str | Path) -> dict:
    """读 ``package.json`` 并**校验**；**只读，不写任何东西**。

    ``root`` 可以是目录，也可以是 ``.zip``（zip 只当只读包读清单与文件表，
    不解压到磁盘）。返回 ``{"manifest", "problems", "n_sessions", "n_files"}``；
    ``problems`` 非空即这份包不完整/不可信（缺文件、清单与包内容对不上、
    schema 不对、``calibration`` 三态自相矛盾）。
    """
    root = Path(root)
    problems: list[str] = []
    manifest: dict | None = None
    actual: set[str] = set()          # 包内文件（POSIX 相对路径，已去顶层前缀）

    if root.is_dir():
        mf = root / "package.json"
        if not mf.is_file():
            return _report(None, [f"这里没有 package.json：{mf}"], 0, 0)
        try:
            manifest = json.loads(mf.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            return _report(None, [f"package.json 读不了：{type(exc).__name__}: {exc}"], 0, 0)
        actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    elif root.is_file() and zipfile.is_zipfile(root):
        with zipfile.ZipFile(root) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            cand = [n for n in names
                    if n == "package.json"
                    or (n.endswith("/package.json") and n.count("/") == 1)]
            if not cand:
                return _report(None, [f"zip 里没有 package.json：{root}"], 0, 0)
            name = sorted(cand, key=len)[0]
            prefix = name[: -len("package.json")]
            try:
                manifest = json.loads(zf.read(name).decode("utf-8"))
            except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                return _report(None, [f"zip 里的 package.json 读不了：{type(exc).__name__}"],
                               0, 0)
            actual = {n[len(prefix):] for n in names if n.startswith(prefix)}
    else:
        return _report(None, [f"既不是目录包也不是 zip：{root}"], 0, 0)

    if not isinstance(manifest, dict):
        return _report(None, ["package.json 不是 JSON 对象"], 0, 0)

    if manifest.get("schema") != PACKAGE_SCHEMA:
        problems.append(f"schema 不是 {PACKAGE_SCHEMA!r}：{manifest.get('schema')!r}")
    if not isinstance(manifest.get("version"), int):
        problems.append(f"version 不是整数：{manifest.get('version')!r}")

    sessions = manifest.get("sessions")
    if not isinstance(sessions, list):
        problems.append("sessions 不是数组")
        sessions = []

    listed: set[str] = set()
    for i, s in enumerate(sessions):
        if not isinstance(s, dict):
            problems.append(f"sessions[{i}] 不是对象")
            continue
        folder = s.get("folder")
        if not isinstance(folder, str) or not folder:
            problems.append(f"sessions[{i}] 缺 folder")
            continue
        if "/" in folder or "\\" in folder:
            problems.append(f"sessions[{i}] 的 folder 含路径分隔符：{folder}")
            continue
        files = s.get("files")
        if not isinstance(files, list):
            problems.append(f"sessions[{i}]（{folder}）缺 files")
            continue
        for rel in files:
            if not isinstance(rel, str) or not rel:
                problems.append(f"sessions[{i}]（{folder}）的 files 里有非法项：{rel!r}")
                continue
            arc = f"sessions/{folder}/{rel}"
            listed.add(arc)
            if arc not in actual:
                problems.append(f"清单里的文件不在包里：{arc}")

    calib = manifest.get("calibration")
    if not isinstance(calib, dict):
        problems.append("缺 calibration 块")
    else:
        for key in ("path", "json"):
            rel = calib.get(key)
            if rel is None:
                continue
            if not isinstance(rel, str) or not rel:
                problems.append(f"calibration.{key} 不是路径：{rel!r}")
                continue
            listed.add(rel)
            if rel not in actual:
                problems.append(f"清单里的标定不在包里：{rel}")
        source = calib.get("source")
        if source not in SOURCE_STATES:
            problems.append(f"calibration.source 取值不认识：{source!r}")
        has_file = bool(calib.get("path") or calib.get("json"))
        if source == "none" and has_file:
            problems.append("calibration.source 是 none，却带着标定文件")
        if source in ("registered", "unregistered") and not has_file:
            problems.append(f"calibration.source 是 {source}，却没有标定文件")
        if "profile" in calib:
            problems.append("calibration 里不该有 profile 字段（计划书 §8 明确不做）")

    # 反方向也要查：包内有、清单里没有的文件（拷贝残留 / .DS_Store / 手改过）
    for arc in sorted(actual):
        if arc in ("package.json", "README.md"):
            continue
        if (arc.startswith("sessions/") or arc.startswith("calib/")) and arc not in listed:
            problems.append(f"包内有清单外的文件：{arc}")

    n_files = len([a for a in actual if a not in ("package.json", "README.md")])
    return _report(manifest, problems, len(sessions), n_files)


# --------------------------------------------------------------------------- #
# 导入（默认 dry-run）
# --------------------------------------------------------------------------- #
def _session_target(rel: str) -> str:
    """包内会话文件 → 磁盘会话目录里的相对路径（``_cat_files`` 的逆映射）。

    为什么不能"原样拷过去"：包里 ``metadata/session.json`` 对应磁盘上根层的
    ``session.json``，``videos/annotated_*.mp4`` 对应 ``derived/annotated_*.mp4``。
    原样拷出来的目录里 ``session.json`` 埋在 ``metadata/`` 下，``discover_session_dirs``
    会把它当成"半成品目录"，界面上根本认不出这段会话 —— 导入等于白导。
    认不出的路径保持原样（宁可多留一个文件，也不丢数据）。
    """
    parts = rel.split("/")
    if len(parts) == 1:
        return rel
    head, rest = parts[0], parts[1:]
    if head == "metadata":
        return "/".join(rest)
    if head == "derived" and rest and rest[0] == "plots":
        return "derived/" + "/".join(rest[1:])
    if head == "derived":
        return "/".join(parts)
    if head == "videos" and rest and rest[0] == "raw":
        return "/".join(rest[1:])
    if head == "videos":
        return "derived/" + "/".join(rest)
    if head == "calibration":
        return "calib/" + "/".join(rest)
    return rel


def _copy_session_tree(src: Path, target: Path) -> None:
    """逐文件按 :func:`_session_target` 还原到 ``target``（不整体 rmtree/整体拷）。"""
    for p in sorted(src.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(src).as_posix()
        out = target / _session_target(rel)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)


def import_package(root: str | Path, dest, *, calibration_dir=None,
                   dry_run: bool = True) -> dict:
    """把包里的会话/标定**拷到** ``dest`` / ``calibration_dir``。

    会话不是"原样搬目录"：包内布局（``metadata/`` ``videos/``）会按
    ``_session_target`` 还原成磁盘上的会话布局（``session.json`` 回到会话根层），
    否则导进去的会话在界面上认不出来。

    - ``dest`` **必须显式给**，永远不默认到 ``data/``：这是用户的真实录制目录，
      一次误调就是一个包悄悄并进去，且分不清哪些是新的。
    - ``dry_run=True``（默认）只返回"将会写哪些路径"的清单，**一个字节都不写**；
    - 目标已存在 → 记进 ``problems``、跳过，**不覆盖**（要不要覆盖是用户的事，
      发布语义不替用户决定）；
    - ``calibration_dir`` 不给就不拷标定（不猜落点），但会提醒包里带了标定；
    - 标定的文件名**逐字不变**（``Path(rel).name`` 原样拼回去）。

    返回 ``{"plan": [{"kind", "src", "dst"}, ...], "copied": n,
    "problems": [...], "dry_run": bool}``。
    """
    root = Path(root)
    if dest is None or not str(dest).strip():
        raise ValueError("import_package 必须显式给出目标数据目录（不许默认落到 data/）")
    dest = Path(dest)

    if root.is_file() and zipfile.is_zipfile(root):
        return {"plan": [], "copied": 0, "dry_run": bool(dry_run),
                "problems": ["zip 包的反向导入还没做：先用解压工具展开，"
                             "再对解压出来的目录调用本函数"]}

    report = read_package(root)
    problems = list(report["problems"])
    manifest = report["manifest"]
    if manifest is None:
        return {"plan": [], "copied": 0, "dry_run": bool(dry_run), "problems": problems}

    plan: list[dict] = []
    blocked: set[int] = set()
    for s in manifest.get("sessions") or []:
        folder = s.get("folder") if isinstance(s, dict) else None
        if not folder:
            continue
        target = dest / folder
        plan.append({"kind": "session", "src": str(root / "sessions" / folder),
                     "dst": str(target)})
        if target.exists():
            problems.append(f"目标已存在，不覆盖：{target}")
            blocked.add(len(plan) - 1)

    calib = manifest.get("calibration") or {}
    for key in ("path", "json"):
        rel = calib.get(key) if isinstance(calib, dict) else None
        if not rel:
            continue
        if calibration_dir is None:
            problems.append(f"包里有标定 {rel}，但没给 calibration_dir —— 不拷（不猜落点）")
            continue
        target = Path(calibration_dir) / Path(rel).name
        plan.append({"kind": "calibration", "src": str(root / Path(rel)),
                     "dst": str(target)})
        if target.exists():
            problems.append(f"标定已存在，不覆盖：{target}")
            blocked.add(len(plan) - 1)

    copied = 0
    if not dry_run:
        for i, item in enumerate(plan):
            if i in blocked:
                continue
            src, target = Path(item["src"]), Path(item["dst"])
            if item["kind"] == "session":
                _copy_session_tree(src, target)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, target)
            copied += 1

    return {"plan": plan, "copied": copied, "problems": problems, "dry_run": bool(dry_run)}
