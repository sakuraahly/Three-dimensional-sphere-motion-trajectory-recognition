"""数据管理的**数据层**自测：虚拟集合树 / 会话枚举 / 状态表重建。

为什么需要它
------------
用户要的是"像文件资源管理器那样看小目录里的数据"。这件事的地基是三件
互不重叠、但**必须只有一份口径**的事（分组与状态表在 ``apps/_session_index.py``，
枚举在 ``sphere_tracker/discovery.py``）：

1. **分组**：``collections.json`` 里的虚拟树（``实验1/梨子/第一篮``），
   允许任意嵌套、允许空目录、允许中文名；磁盘一动不动。
2. **枚举**：一个根下面到底有哪些会话。递归、命中即剪枝、认半成品
   （没有 ``session.json`` 但有 ``left.mkv``）、跳过 ``_trash``。
   **全仓库只有这一份实现**：``apps/_session_index.discover_session_dirs``
   与 ``apps/_stereo_layout.candidate_session_dirs``（多根包装）都是转发，
   ``stereo.sessions.list_sessions`` 直接调它。
3. **状态表**：``index.json`` 是派生缓存，删了必须能重建出**逐键相同**的内容。

这三件事以前各有两三套：数据管理、会话列表、``check_dataset`` 各扫一遍，
同一份磁盘数出 **29 / 44** 两个数（``parallel/`` 里那 15 段只有数据管理看得见，
"看得见用不了"）；而"按父目录名拼分组"在树、索引、导出包里各写一遍，
迟早对不上。所以这里逐条钉住：

- ``group_of`` 只有三种分支（集合里配了 / 没配按磁盘推 / 配的是根层），
  多一种或少一种都会让"树上"和"表里"分成两套集合；
- ``candidate_session_dirs`` / ``discover_session_dirs(strict=True)`` /
  ``list_sessions`` **三个入口必须是同一个集合**（第 9 条）—— 这是
  "三套扫描器合并成一套"的护栏；
- ``recursive=False`` 是 ``recursive=True`` 的**真子集**（第 3 层看不到），
  ``strict=True`` 是宽判据的**子集**（半成品看不见），两条都用临时树验，
  因为真实数据恰好没有第 3 层、也没有半成品 —— 不造树就等于没验；

- ``scan_items`` 枚举到的 sid 集合，必须等于按 ``group`` 建树后能覆盖到的
  sid 集合 —— 少一段就是"列表里有、树上没有"或者反过来。

反向自证（改了就会红）
--------------------
==================================  ========================================  ========
改动                                注入方式                                  结果
==================================  ========================================  ========
``group_of`` 去掉集合优先           直接 ``return disk_group_of(rel_path)``    exit=1
``all_group_paths`` 不补中间层       只保留叶子路径                             exit=1
中文名被转义                        ``json.dump`` 去掉 ``ensure_ascii=False``  exit=1
坏 JSON 被回写成空配置              ``load_collections`` 的 except 里加写入     exit=1
``n_unfiled`` 数成全部会话           ``build_index`` 里改成 ``len(sessions)``    exit=1
``normalize`` 不处理父子成环         删掉 ``looped`` 那段                        exit=1
``all_node_paths`` 只收有会话的节点  改用 ``collection_map``                     exit=1
``discover`` 不跳过 ``_trash``       从 ``SKIP_DIR_NAMES`` 去掉 ``"_trash"``     exit=1
枚举退回两层口径                     ``candidate`` 改传 ``recursive=False``      exit=1（第 9 条）
``load_index`` 不校验 root           去掉 ``raw["root"]`` 比较                   exit=1
``sid_of`` 退回目录名                删掉 ``session.json`` 的 ``id`` 分支        exit=1
sid 撞车不加确定性后缀               去掉 ``while sid in sids`` 循环             exit=1
22 份保护漏了一份                   从 ``PROTECTED_OUTPUTS`` 删掉一条           exit=1（第 13 节）
clean_junk 删掉硬保护               去掉 ``is_protected_output`` 那道拒绝       exit=1（第 13 节）
四处又抄一份备份格式串              任一调用方复制 ``<name>.bak-<stamp>``      exit=1（第 14 节）
备份排序键改回文件名字典序          ``key=_backup_sort_key`` → ``q.name``      exit=1（第 14 节）
裁剪不再保护刚创建的 backup         ``removable`` 去掉 ``q != backup``          exit=1（第 14 节）
冲突行去掉 ⚠ 提示                   ``mode_cell_text`` 直接返回 mode_label     exit=1（第 15 节）
索引不过滤无 ``session.json`` 的目录  删掉 ``if not (... session.json).is_file()``  exit=1（第 16 节）
``scan_items`` 也把孤儿藏起来        ``scan_items`` 改用 ``strict=True`` 枚举    exit=1（第 16 节）
==================================  ========================================  ========

**实测过的四处"改坏验证"**（本测试写完后真的改坏被测代码再跑，确认会红，
之后已还原）：① ``load_collections`` 坏 JSON 时回写空配置 → 第 5 条红；
② ``SKIP_DIR_NAMES`` 去掉 ``_trash`` → 第 8 条红（3 条断言）；
③ ``sid_of`` 不用 ``session.json`` 的 ``id`` → 第 11 条红，并连带第 4、12 条
一起红（同一个根因，符合预期）。写坏的注入方式就是上面表格里那三种；
④ 2026-11-09 三套扫描器合并后：把 ``candidate_session_dirs`` 里的
``recursive=True`` 改成 ``recursive=False`` → 第 9 条红 1 项（"candidate 也能
看到第 3 层"），exit=1。注意真实 ``data/stereo`` 恰好没有第 3 层，所以光靠
真实数据这条**不会**红 —— 第 9 条里那棵临时树就是为它准备的。
⑤ 2026-11-10 同秒裁剪真 bug 修复后，又真改坏两次跑红再还原：
把 ``_backup_sort_key`` 换回 ``key=lambda q: q.name`` → 第 14 节
「-10 数值序 / 裁剪留最新的 2 份」红；去掉 ``removable`` 里的
``q != backup`` 保护 → 第 14 节「同秒连写 7 次仍留 v=6 回滚点」红。
两次都 exit=1，还原后 69 个检查点全绿。
⑥ 2026-11-18 索引收口（列表宽 / 索引窄）后真改坏两次再跑红、再还原：
删掉 ``build_index`` 里 ``if not (it.path / "session.json").is_file(): continue``
→ 第 16 节 3 条 + 第 15 节「真实数据：索引里没有幽灵会话」红（4 项，exit=1）；
把 ``scan_items`` 的枚举改成 ``strict=True``（连孤儿一起从界面藏起来）
→ 第 16 节「items 两个都在」「数量差 1」红 + 第 8 节 3 条红（5 项，exit=1）。
两次还原后两个文件的 ``sha256sum`` 与改坏前逐字节相同，74 个检查点全绿。

**这条测试第一次跑就抓到一个真实缺陷**：``scan_items`` 的撞车分支原来写成
``sids`` 同时当 "rel_path → sid" 映射和 "已占用的 sid 集合" 用，
``while sid in sids`` 拿 sid 去比相对路径，**永远不命中** —— 两段同为
``same-id`` 的会话会拿到同一个 sid，且 ``ItemList.warnings`` 为空（静默）。
第 12 条就是为它写的；修好之后才可能出 ``same-id~2``。

用法::

    python tests/selftest_session_index.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
while not (ROOT / "apps").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps._data_export import SessionItem, scan_items
from apps._session_index import (
    COLLECTIONS_SCHEMA,
    ROOT_NODE,
    all_group_paths,
    all_node_paths,
    build_index,
    collection_map,
    collection_paths_by_node,
    discover_session_dirs,
    group_of,
    load_collections,
    load_index,
    looks_like_session_dir,
    node_children,
    normalize_collections,
    rel_key,
    save_collections,
    save_index,
    sid_of,
)
from apps._stereo_layout import candidate_session_dirs


def _mk_session(parent: Path, name: str, sid: str | None = None, *,
                mode: str = "parallel", frames_l: int = 3,
                frames_r: int = 3) -> Path:
    """造一段最小但**合法**的会话目录（字段参照 selftest_launcher_params 第 13 节）。"""
    d = Path(parent) / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "session.json").write_text(json.dumps({
        "schema": "sphere-stereo-session", "version": 1,
        "id": sid if sid is not None else name,
        "created": "2026-01-01 12:00:00",
        "stereo_mode": mode,
        "left": {"frame_count": frames_l},
        "right": {"frame_count": frames_r},
    }, ensure_ascii=False), encoding="utf-8")
    return d


def _collections_with(nodes: list[dict]) -> dict:
    """用原始 JSON 形状造一份集合配置并过 ``normalize_collections``。"""
    raw = {"schema": COLLECTIONS_SCHEMA,
           "nodes": [{"id": ROOT_NODE, "name": "全部会话",
                      "parent_id": None, "session_ids": []}, *nodes]}
    data, _warns = normalize_collections(raw)
    return data


def _tree_covered_sids(items) -> set[str]:
    """按 ``it.group`` 建树后，树上**能覆盖到**的 sid 集合。

    口径：每个分组路径的每一层（``实验1``、``实验1/梨子`` …）都必须在
    ``all_group_paths`` 里存在，否则那批会话在 Treeview 里插不进去（父节点缺失）。
    """
    by_group: dict[str, list[str]] = {}
    for it in items:
        by_group.setdefault(it.group, []).append(it.sid)
    tree = set(all_group_paths(by_group))
    covered: set[str] = set()
    for group, sids in by_group.items():
        parts = group.split("/") if group else []
        if all("/".join(parts[:i]) in tree for i in range(1, len(parts) + 1)):
            covered.update(sids)
    return covered


# ----------------------------------------------------------------------
# 1. group_of 的三个分支
# ----------------------------------------------------------------------
def _check_group_of(ck) -> None:
    print("----- 1. group_of 三分支（集合优先 / 磁盘兜底 / 根层）-----")
    mapping = {"S-1": "实验1/梨子"}
    ck("集合里配了 S → 返回配置的路径（忽略 rel_path）",
       group_of("S-1", mapping, "parallel/20260101-120000") == "实验1/梨子")
    ck("集合里没配 + rel_path=parallel/x → 按磁盘推成「平行双目」",
       group_of("S-2", mapping, "parallel/20260101-120000") == "平行双目")
    ck("集合里没配 + 根层时间戳 → 空串（散会话）",
       group_of("S-3", mapping, "20261021-120000") == "")
    ck("集合里配的是根层（\"\"）→ 返回空串，**不回退**磁盘层级",
       group_of("S-4", {"S-4": ""}, "parallel/20260101-120000") == "")


# ----------------------------------------------------------------------
# 2. 三层嵌套
# ----------------------------------------------------------------------
def _check_three_levels(ck) -> None:
    print("----- 2. 三层嵌套 实验1/梨子/第一篮 -----")
    data = _collections_with([
        {"id": "n1", "name": "实验1", "parent_id": ROOT_NODE, "session_ids": []},
        {"id": "n2", "name": "梨子", "parent_id": "n1", "session_ids": []},
        {"id": "n3", "name": "第一篮", "parent_id": "n2", "session_ids": ["S-A"]},
    ])
    paths = collection_paths_by_node(data)
    ck("collection_paths_by_node 三层路径全对",
       (paths.get("n1"), paths.get("n2"), paths.get("n3"))
       == ("实验1", "实验1/梨子", "实验1/梨子/第一篮"))

    leaves = collection_map(data).values()
    filled = set(all_group_paths(leaves))
    ck("all_group_paths 补出全部中间层（否则干活的会话是孤儿）",
       {"实验1", "实验1/梨子", "实验1/梨子/第一篮"} <= filled)

    kids = node_children(data)
    chain_ok = (
        [n["id"] for n in kids.get(ROOT_NODE, [])] == ["n1"]
        and [n["id"] for n in kids.get("n1", [])] == ["n2"]
        and [n["id"] for n in kids.get("n2", [])] == ["n3"]
        and kids.get("n3", []) == []
    )
    ck("node_children 把 根 → 实验1 → 梨子 → 第一篮 串起来", chain_ok)


# ----------------------------------------------------------------------
# 3. 中文名往返（文件必须是人能直接看懂的 UTF-8）
# ----------------------------------------------------------------------
def _check_chinese_roundtrip(ck) -> None:
    print("----- 3. 中文节点名往返（不许被 \\uXXXX 转义）-----")
    with tempfile.TemporaryDirectory(prefix="si_zh_") as td:
        p = Path(td) / "collections.json"
        data = _collections_with([
            {"id": "c1", "name": "实验1", "parent_id": ROOT_NODE, "session_ids": []},
            {"id": "c2", "name": "梨子", "parent_id": "c1", "session_ids": []},
        ])
        save_collections(data, p)
        raw_bytes = p.read_bytes()
        back, warns = load_collections(p)
        names_saved = {n["id"]: n["name"] for n in data["nodes"]}
        names_back = {n["id"]: n["name"] for n in back["nodes"]}
        ck("中文节点名保存 → 读回后逐字相同（且无警告）",
           names_saved == names_back and not warns)
        ck("原始文件字节里能直接找到「实验」的 UTF-8 编码（人看得懂）",
           "实验".encode("utf-8") in raw_bytes)
        ck("原始文件里没有 \\u 转义（ensure_ascii=False 真的生效了）",
           b"\\u5b9e" not in raw_bytes and b"\\u9ece" not in raw_bytes)


# ----------------------------------------------------------------------
# 4. 散会话
# ----------------------------------------------------------------------
def _check_unfiled(ck) -> None:
    print("----- 4. 散会话：group 为空串 + n_unfiled 计数 -----")
    with tempfile.TemporaryDirectory(prefix="si_unfiled_") as td:
        root = Path(td) / "stereo"
        _mk_session(root, "20260101-120000", "loose-1")      # 根层，未归入集合
        _mk_session(root, "20260102-120000", "filed-1")       # 归入「实验1」
        coll = _collections_with([
            {"id": "c1", "name": "实验1", "parent_id": ROOT_NODE,
             "session_ids": ["filed-1"]},
        ])
        index, items = build_index(root, collections=coll)
        by_sid = {it.sid: it for it in items}
        loose = by_sid.get("loose-1")
        filed = by_sid.get("filed-1")
        # 用 .get 而不是 []：sid 算错时这一条要**红**，不该让整份测试崩在 KeyError
        # 上（崩掉就看不到后面每一条的结论了）。
        ck("不在集合里的会话 group == \"\"（散会话）",
           loose is not None and loose.group == "")
        ck("集合里的会话 group == 配置的路径（不受磁盘层影响）",
           filed is not None and filed.group == "实验1")
        ck("index['n_unfiled'] 只数散会话（1，而不是全部 2）",
           index["n_unfiled"] == 1 and index["n_sessions"] == 2)


# ----------------------------------------------------------------------
# 5. 坏 JSON 不覆盖
# ----------------------------------------------------------------------
def _check_bad_json_untouched(ck) -> None:
    print("----- 5. collections.json 坏掉：返回空配置但**一个字节都不改** -----")
    with tempfile.TemporaryDirectory(prefix="si_bad_") as td:
        p = Path(td) / "collections.json"
        bad = b'{oops'
        p.write_bytes(bad)
        data, warns = load_collections(p)
        after = p.read_bytes()
        ck("坏 JSON → 可用空配置（只剩根节点），不抛异常",
           [n["id"] for n in data["nodes"]] == [ROOT_NODE])
        ck("坏 JSON → 警告非空（不能静默）", bool(warns))
        ck("坏 JSON → 原文件逐字节没变（否则用户的组织被程序删了）",
           after == bad)


# ----------------------------------------------------------------------
# 6. normalize_collections 的三条修正
# ----------------------------------------------------------------------
def _check_normalize_fixes(ck) -> None:
    print("----- 6. normalize_collections：成环 / 悬空 parent / sid 重复 -----")
    # ① 父子成环
    try:
        data, warns = normalize_collections({
            "schema": COLLECTIONS_SCHEMA,
            "nodes": [
                {"id": ROOT_NODE, "name": "全部会话", "parent_id": None},
                {"id": "x", "name": "X", "parent_id": "y"},
                {"id": "y", "name": "Y", "parent_id": "x"},
            ],
        })
        parents = {n["id"]: n["parent_id"] for n in data["nodes"]}
        loop_ok = (parents.get("x") == ROOT_NODE or parents.get("y") == ROOT_NODE)
        ck("① parent 成环（x→y→x）→ 至少一个挂根，且不抛异常",
           loop_ok and bool(warns))
    except Exception as exc:                                 # noqa: BLE001
        ck(f"① parent 成环 → 不许抛异常（实际抛出 {type(exc).__name__}: {exc}）",
           False)

    # ② parent 指向不存在的 id
    data, warns = normalize_collections({
        "schema": COLLECTIONS_SCHEMA,
        "nodes": [
            {"id": ROOT_NODE, "name": "全部会话", "parent_id": None},
            {"id": "z", "name": "走丢的目录", "parent_id": "no-such-node"},
        ],
    })
    parents = {n["id"]: n["parent_id"] for n in data["nodes"]}
    ck("② parent 指向不存在的 id → 挂根 + 警告",
       parents.get("z") == ROOT_NODE and any("不存在" in w for w in warns))

    # ③ 同一个 sid 出现在两个节点
    data, warns = normalize_collections({
        "schema": COLLECTIONS_SCHEMA,
        "nodes": [
            {"id": ROOT_NODE, "name": "全部会话", "parent_id": None},
            {"id": "a", "name": "先出现的", "parent_id": ROOT_NODE,
             "session_ids": ["dup"]},
            {"id": "b", "name": "后出现的", "parent_id": ROOT_NODE,
             "session_ids": ["dup", "other"]},
        ],
    })
    nodes = {n["id"]: n for n in data["nodes"]}
    a_ids = nodes.get("a", {}).get("session_ids")
    b_ids = nodes.get("b", {}).get("session_ids")
    ck("③ 同一个 sid 出现在两个节点 → 只保留第一个",
       a_ids == ["dup"] and b_ids == ["other"])
    ck("③ 只保留第一个的同时必须警告（不能静默丢用户的分组）",
       any("dup" in w for w in warns))


# ----------------------------------------------------------------------
# 7. 空目录不消失
# ----------------------------------------------------------------------
def _check_empty_dirs_survive(ck) -> None:
    print("----- 7. 一个会话都没有的节点也必须留在树里 -----")
    data = _collections_with([
        {"id": "e1", "name": "空目录", "parent_id": ROOT_NODE, "session_ids": []},
        {"id": "e2", "name": "空子", "parent_id": "e1", "session_ids": []},
    ])
    paths = all_node_paths(data)
    ck("all_node_paths 含空目录（刚建完还没放会话时不能消失）",
       "空目录" in paths)
    ck("all_node_paths 含空的嵌套子目录", "空目录/空子" in paths)
    ck("collection_map 此时确实为空（证明上面不是靠会话撑出来的）",
       collection_map(data) == {})


# ----------------------------------------------------------------------
# 8. 两视图同集合（核心护栏）
# ----------------------------------------------------------------------
def _check_two_views_same_set(ck) -> None:
    print("----- 8. 枚举与 scan_items 必须是同一个集合（含半成品 / 跳过回收站）-----")
    with tempfile.TemporaryDirectory(prefix="si_two_views_") as td:
        root = Path(td) / "stereo"
        a = _mk_session(root, "实验1/20260101-120000", "sid-a")
        b = _mk_session(root, "实验1/子/20260102-120000", "sid-b")
        c = _mk_session(root, "20260103-120000", "sid-c")
        half = root / "实验1" / "半成品"
        half.mkdir(parents=True)
        (half / "left.mkv").write_bytes(b"raw")              # 没有 session.json
        trashed = _mk_session(root, "_trash/20260104-120000", "sid-trash")

        dirs = discover_session_dirs(root, recursive=True)
        rels = [rel_key(root, d) for d in dirs]
        ck("discover_session_dirs 恰好 4 段（半成品算，回收站不算）",
           len(dirs) == 4)
        ck("半成品的目录 looks_like_session_dir 为 True（不靠 session.json 认）",
           looks_like_session_dir(half) is True)
        ck("_trash 里的完整会话**没有**被枚举（回收站里的不许出现在列表）",
           all("_trash" not in r for r in rels)
           and rel_key(root, trashed) not in rels)
        ck("递归枚举看到了两层嵌套里的会话（实验1/子/…）",
           "实验1/子/20260102-120000" in rels and "实验1/20260101-120000" in rels)

        items = scan_items(root)
        sids = {it.sid for it in items}
        ck("scan_items 的 sid 集合 = 物理枚举出来的 sid 集合（不多不少）",
           sids == {sid_of(d) for d in dirs})
        ck("scan_items 的 sid 集合 = 按 group 建树后树能覆盖到的 sid 集合",
           sids == _tree_covered_sids(items))
        keys = [it.key for it in items]
        ck("it.key（相对路径）无重复（Treeview 的 iid 不能撞）",
           len(keys) == len(set(keys)) and len(keys) == 4)
        half_item = next((it for it in items if it.sid == "半成品"), None)
        ck("半成品出现在列表里且 status == bad（缺 session.json 仍要给人看）",
           half_item is not None and half_item.status == "bad"
           and "缺 session.json" in half_item.problems)


# ----------------------------------------------------------------------
# 9. 三套扫描器合并成一套（只读真实数据 + 一棵临时嵌套树）
# ----------------------------------------------------------------------
def _check_legacy_unchanged(ck) -> None:
    print("----- 9. candidate / discover / list_sessions 是同一份枚举 -----")
    real = ROOT / "data" / "stereo"
    if not real.is_dir():
        print("  [跳过] 跳过（无真实数据）")
        return
    from stereo.sessions import list_sessions

    deep = {rel_key(real, d)
            for d in discover_session_dirs(real, recursive=True, strict=True)}
    cand = {rel_key(real, d) for d in candidate_session_dirs([real])}
    lib = {rel_key(real, d) for d in list_sessions(real)}
    print(f"    实测：discover(strict) {len(deep)} 段，candidate {len(cand)} 段，"
          f"list_sessions {len(lib)} 段")
    # 为什么不再是「recursive=False 与 candidate 相等」：candidate 自己已经递归了，
    # 那条断言就变成同义反复（等于没验）。现在验的是三个**入口**给出同一个集合。
    ck("candidate_session_dirs 与 discover_session_dirs(strict=True) 集合完全相等"
       "（它只是多根包装，不是第二份实现）",
       cand == deep)
    ck(f"stereo.sessions.list_sessions 与上面同一集合（{len(lib)} 段）"
       "—— 三套扫描器合并成一套",
       lib == deep)

    # 造一棵带第 3 层的临时树：旧口径看不到，递归口径与候选枚举都必须看到。
    # 真实数据里恰好没有第 3 层，所以这条只能用临时树验（否则"递归"这个性质
    # 在任何一台机器上都没被真正测过）。
    with tempfile.TemporaryDirectory(prefix="si_depth_") as td:
        root = Path(td) / "stereo"
        _mk_session(root, "parallel/20260101-120000", "lvl2")
        _mk_session(root, "实验1/梨子/20260102-120000", "lvl3")
        half = root / "半成品"
        half.mkdir()
        (half / "left.mkv").write_bytes(b"raw")          # 没有 session.json

        shallow = {rel_key(root, d)
                   for d in discover_session_dirs(root, recursive=False)}
        wide = {rel_key(root, d) for d in discover_session_dirs(root, recursive=True)}
        strict = {rel_key(root, d)
                  for d in discover_session_dirs(root, recursive=True, strict=True)}
        cand2 = {rel_key(root, d) for d in candidate_session_dirs([root])}

        ck("recursive=False 是 recursive=True 的**真**子集（第 3 层看不到）",
           shallow < wide
           and "实验1/梨子/20260102-120000" in wide
           and "实验1/梨子/20260102-120000" not in shallow)
        ck("candidate_session_dirs 也能看到第 3 层（旧的两层口径就是在这里丢会话的）",
           "实验1/梨子/20260102-120000" in cand2 and cand2 == strict)
        ck("strict=True 是 strict=False 的子集（只有 left.mkv 的半成品宽判据看得见）",
           strict < wide and "半成品" in wide and "半成品" not in strict)


# ----------------------------------------------------------------------
# 10. 索引可重建
# ----------------------------------------------------------------------
def _check_index_rebuild(ck) -> None:
    print("----- 10. index.json 删掉能重建出逐键相同的内容 -----")
    with tempfile.TemporaryDirectory(prefix="si_index_") as td:
        root = Path(td) / "stereo"
        _mk_session(root, "20260101-120000", "i-1")
        _mk_session(root, "parallel/20260102-120000", "i-2")
        coll = _collections_with([])
        p = Path(td) / "index.json"

        first, _items = build_index(root, collections=coll)
        save_index(first, p)
        raw1 = json.loads(p.read_text(encoding="utf-8"))
        p.unlink()
        second, _items2 = build_index(root, collections=coll)
        save_index(second, p)
        raw2 = json.loads(p.read_text(encoding="utf-8"))

        for d in (raw1, raw2):
            d.pop("built", None)
        ck("两次重建除 built 字段外逐键相同（用 json 全量比对）", raw1 == raw2)
        ck("load_index(root=别的路径) 返回 None（根不匹配 = 需要重建）",
           load_index(p, root=Path(td) / "another-root") is None)
        ck("load_index(root=本根) 能读回同一份索引",
           isinstance(load_index(p, root=root), dict))


# ----------------------------------------------------------------------
# 11. sid 稳定（搬目录不变）
# ----------------------------------------------------------------------
def _check_sid_stable(ck) -> None:
    print("----- 11. 把会话搬进 parallel/ 之后 sid 不变 -----")
    with tempfile.TemporaryDirectory(prefix="si_move_") as td:
        root = Path(td) / "stereo"
        src = _mk_session(root, "20260105-120000", "stable-id")
        sid_before = sid_of(src)
        rel_before = rel_key(root, src)
        group_before = group_of(sid_before, {}, rel_before)

        (root / "parallel").mkdir()
        dst = root / "parallel" / "20260105-120000"
        shutil.move(str(src), str(dst))
        sid_after = sid_of(dst)
        rel_after = rel_key(root, dst)
        group_after = group_of(sid_after, {}, rel_after)

        ck("搬目录后 sid 不变（来自 session.json 的 id，不是目录名）",
           sid_before == "stable-id" and sid_after == "stable-id")
        ck("rel_key 从 <时间戳> 变成 parallel/<时间戳>",
           rel_before == "20260105-120000"
           and rel_after == "parallel/20260105-120000")
        ck("group_of 从空串变成「平行双目」（磁盘兜底跟着新位置走）",
           group_before == "" and group_after == "平行双目")


# ----------------------------------------------------------------------
# 12. sid 撞车确定性
# ----------------------------------------------------------------------
def _build_collision_root(base: Path, reverse: bool) -> tuple[dict, list[str]]:
    root = base / "stereo"
    rows = [("a/20260101-120000", "a"), ("b/20260102-120000", "b")]
    for rel, _tag in (reversed(rows) if reverse else rows):
        _mk_session(root, rel, "same-id")
    items = scan_items(root)
    return {it.sid: it.rel_path for it in items}, list(items.warnings)


def _check_sid_collision(ck) -> None:
    print("----- 12. sid 撞车：确定性加后缀，枚举顺序变了结果也一样 -----")
    with tempfile.TemporaryDirectory(prefix="si_collide1_") as td1, \
            tempfile.TemporaryDirectory(prefix="si_collide2_") as td2:
        m1, w1 = _build_collision_root(Path(td1), reverse=False)
        m2, w2 = _build_collision_root(Path(td2), reverse=True)
        print(f"    第一次：{m1}")
        print(f"    第二次（目录创建顺序相反）：{m2}")
        ck("两段 same-id → 得到 same-id 与 same-id~2",
           set(m1) == {"same-id", "same-id~2"})
        ck("撞车必须进 warnings（自动改名不能静默）", bool(w1))
        ck("枚举顺序变化时 sid → rel_path 映射完全不变（不是看谁先扫到）",
           m1 == m2)


# ----------------------------------------------------------------------
# 13. 22 份不可再生交付物：任意一份都不会被判为可删
# ----------------------------------------------------------------------
def _check_protected_outputs(ck) -> None:
    """D：用户裁决原地保留的 22 份，逐一被清理判定与执行处保护。

    为什么并进本文件而不新建 `selftest_junk.py`：`selftest_doc_facts` 的
    `REAL_TEST_COUNT` 是签字预算，新增一个 `selftest_*.py` 会连锁改动十来处
    计数说法。这里只加判据，不动文件数。
    """
    print("----- 13. 22 份交付物：任意一份都不会被判为可删 -----")
    from apps._junk import (
        DANGER_SAFE, JunkFile, JunkItem, JunkPlan, PROTECTED_OUTPUTS,
        clean_junk, is_protected_output, looks_like_research_artifact,
    )

    ck("保护清单恰好 22 份", len(PROTECTED_OUTPUTS) == 22)

    with tempfile.TemporaryDirectory(prefix="si_protected_") as td:
        repo = Path(td) / "repo"
        for rel in sorted(PROTECTED_OUTPUTS):
            p = repo / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("不可再生", encoding="utf-8")

        # ① 现有语义判据（looks_like_research_artifact）必须认出每一份
        judged = [rel for rel in sorted(PROTECTED_OUTPUTS)
                  if looks_like_research_artifact(
                      [JunkFile(repo / rel, 1, 0.0)]) is not None]
        ck("22 份逐一被 looks_like_research_artifact 认成不可删",
           len(judged) == len(PROTECTED_OUTPUTS)
           and all(judged))

        # ② 即使被伪装成「推荐」档塞进计划，clean_junk 也必须拒绝删除
        refused = []
        for rel in sorted(PROTECTED_OUTPUTS):
            p = repo / rel
            plan = JunkPlan(items=[JunkItem(
                "fake", "伪造", "", DANGER_SAFE, [JunkFile(p, 1, 0.0)])])
            deleted, _freed, failures = clean_junk(plan, repo)
            if deleted == 0 and p.is_file() and failures:
                refused.append(rel)
        ck("22 份逐一被 clean_junk 拒绝删除（即使勾了推荐档）",
           len(refused) == len(PROTECTED_OUTPUTS))

        # 反向自证：保护不能扩大到普通调试产物，否则清理器就废了
        other = repo / "outputs/agent_x/f1.bin"
        other.parent.mkdir(parents=True, exist_ok=True)
        other.write_bytes(b"x")
        plan2 = JunkPlan(items=[JunkItem(
            "fake2", "伪造2", "", DANGER_SAFE, [JunkFile(other, 1, 0.0)])])
        d2, _f2, _e2 = clean_junk(plan2, repo)
        ck("普通调试产物仍可删（保护没有扩大化）", d2 == 1 and not other.exists())

        # 13 份运行期自动产物不该被误列进保护清单
        walk = ("anchor", "anchor2", "anchor3", "anchor4", "e4_partial",
                "fixA", "fixB", "full", "run", "smoke", "verify")
        auto = {f"outputs/gui_walk/{n}/report.md" for n in walk} | {
            "outputs/audit_critic_75/report.md",
            "outputs/audit_critic_final/report.md"}
        ck("13 份自动产物没被误列进保护清单",
           len(auto) == 13 and not (auto & PROTECTED_OUTPUTS))
        ck("is_protected_output 按相对 outputs/ 路径匹配，与盘符无关",
           is_protected_output(repo / "outputs/cam/doctor_report.md")
           and not is_protected_output(repo / "data/cam/doctor_report.md"))


# ----------------------------------------------------------------------
# 14. 备份 / 原子写：格式、撞车、保留份数、唯一实现
# ----------------------------------------------------------------------
def _check_backup_helper(ck) -> None:
    """E：四处实现收敛到 `apps._common.atomic_write_with_backup` 一份。"""
    print("----- 14. 备份命名规范：唯一实现 + 5/3 保留差异 -----")
    import re
    from unittest import mock

    import apps._common as _cm
    from apps._common import (
        DEFAULT_KEEP_BACKUPS, SESSION_KEEP_BACKUPS,
        atomic_write_with_backup, backup_glob, tmp_name,
    )
    from apps._junk import KEEP_BAK_PER_SESSION
    from apps._session_index import empty_collections, save_collections

    ck("保留份数差异通过常量/参数保留（5 / 3，不抹平）",
       DEFAULT_KEEP_BACKUPS == 5 and SESSION_KEEP_BACKUPS == 3
       and KEEP_BAK_PER_SESSION == SESSION_KEEP_BACKUPS)
    ck("backup_glob / tmp_name 与写备份同源",
       backup_glob("x/collections.json") == "collections.json.bak-*"
       and tmp_name("x/session.json").startswith("session.json.tmp-"))

    # 反向自证：四处调用方不许再留第二份**备份**格式字符串。
    # ⚠️ 不含 `strftime("%Y%m%d-%H%M%S")`：`data_manager` 里剩下两处是
    # 「导出目录」的时间戳，不是备份命名，属另一件事，不该被这条判据误伤。
    legacy = ('f"{p.name}.bak-{stamp}"', 'f"{p.name}.bak-{stamp}-{n}"',
              'f"{BAK_PREFIX}{stamp}"', 'f"{BAK_PREFIX}{stamp}-{n}"',
              'f"session.json.bak-{stamp}"', 'f"session.json.bak-{stamp}-{n}"',
              'f"{p.name}.tmp-{os.getpid()}"', 'f"{jf.name}.tmp-{os.getpid()}"',
              'glob("session.json.bak-*")')
    offenders = []
    for rel in ("apps/_session_index.py", "apps/data_manager.py",
                "apps/_stereo_sessions.py", "apps/_junk.py"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        offenders += [f"{rel}: {lit}" for lit in legacy if lit in text]
    ck("四处不再留第二份备份格式字符串（只有 _common.py 一份）",
       not offenders)
    if offenders:
        print(f"    {offenders}")

    with tempfile.TemporaryDirectory(prefix="si_backup_") as td:
        p = Path(td) / "session.json"
        original = {"id": "x", "label": "原件", "nested": {"a": 1}}
        p.write_text(json.dumps(original, ensure_ascii=False, indent=2),
                     encoding="utf-8")
        before = p.read_bytes()
        bak = atomic_write_with_backup(p, {"id": "x", "label": "新"},
                                       keep_backups=None)
        ck("返回备份路径；备份是原件逐字节副本（不重新序列化）",
           bak is not None and bak.read_bytes() == before)
        after = p.read_text(encoding="utf-8")
        ck("写入是 UTF-8 / 中文不转义 / 末尾换行",
           json.loads(after)["label"] == "新" and "\\u65b0" not in after
           and after.endswith("\n"))
        ck("临时文件已被 os.replace 收走",
           not list(Path(td).glob("*.tmp-*")))
        ck("备份名符合 <原名>.bak-<YYYYmmdd-HHMMSS>",
           bak is not None and re.fullmatch(
               r"session\.json\.bak-\d{8}-\d{6}", bak.name) is not None)

        # 冻住时钟，逼出同秒撞车的 -2 / -3，并一路造到 -11 验数值序。
        # 用**独立**临时目录 td2：上面 si_backup_ 里已留下一份真实时间戳的
        # 备份，混进来会干扰「裁剪后剩哪两份」的判定。
        class _FrozenClock:
            @staticmethod
            def now():
                return datetime(2026, 1, 2, 3, 4, 5)

        stamp = "20260102-030405"
        with tempfile.TemporaryDirectory(prefix="si_frozen_") as td2, \
                mock.patch.object(_cm, "datetime", _FrozenClock):
            q = Path(td2) / "session.json"
            q.write_text("{}", encoding="utf-8")
            (Path(td2) / f"session.json.bak-{stamp}").write_bytes(b"old1")
            b2 = atomic_write_with_backup(q, {"id": "x"}, keep_backups=None)
            ck("同秒撞车自动加 -2",
               b2 is not None and b2.name == f"session.json.bak-{stamp}-2")
            b3 = atomic_write_with_backup(q, {"id": "x"}, keep_backups=None)
            # 再写 8 次 → -4…-11（同秒，keep_backups=None 不裁剪）
            for _n in range(4, 12):
                atomic_write_with_backup(q, {"id": "x"}, keep_backups=None)
            names = {x.name for x in Path(td2).glob("session.json.bak-*")}
            expected = {f"session.json.bak-{stamp}"} | {
                f"session.json.bak-{stamp}-{n}" for n in range(2, 12)}
            numeric_named = expected <= names
            # 第 12 次写开 keep_backups=2：按 (stamp,n) 数值序，留下的必须是
            # 序号最大的 -11 / -12。若排序键退回文件名字典序，-10/-11 会排到
            # -2 前面被当成「旧」删掉，留下 -9 之流 → 这条会红。
            last = atomic_write_with_backup(q, {"id": "x"}, keep_backups=2)
            kept = {x.name for x in Path(td2).glob("session.json.bak-*")}
            numeric_kept = kept == {
                f"session.json.bak-{stamp}-11",
                f"session.json.bak-{stamp}-12"}
            ck("再撞一次加 -3 / -10 数值序 / 裁剪留最新的 2 份",
               b3 is not None and b3.name == f"session.json.bak-{stamp}-3"
               and numeric_named and last is not None and numeric_kept)

    # 保留份数：>0 按 (stamp,n) 数值序裁剪；None 写时不裁剪
    with tempfile.TemporaryDirectory(prefix="si_keep5_") as td:
        p = Path(td) / "x.json"
        p.write_text("{}", encoding="utf-8")
        for i in range(7):
            atomic_write_with_backup(p, {"i": i}, keep_backups=5)
        keep5_ok = len(list(Path(td).glob("x.json.bak-*"))) == 5

    # 真 bug 回归：同秒连写 7 次 + keep_backups=2，最新一次写入前的回滚点
    # （{"v":6}）必须还在，且备份数不超过 2。去掉对刚创建 backup 的显式保护
    # 会让新备份被自己这次的裁剪立刻删掉 → 这条会红。
    with tempfile.TemporaryDirectory(prefix="si_keep2_") as td:
        q = Path(td) / "cfg.json"
        q.write_text(json.dumps({"v": 0}), encoding="utf-8")

        class _Frozen2:
            @staticmethod
            def now():
                return datetime(2026, 11, 10, 20, 20, 51)

        with mock.patch.object(_cm, "datetime", _Frozen2):
            for i in range(1, 8):
                atomic_write_with_backup(q, {"v": i}, keep_backups=2)
        baks = sorted(Path(td).glob("cfg.json.bak-*"))
        contents = [json.loads(b.read_text(encoding="utf-8"))["v"]
                    for b in baks]
        keep2_ok = (len(baks) <= 2 and 6 in contents
                    and json.loads(q.read_text(encoding="utf-8"))["v"] == 7)

    ck("keep_backups=5 只留最近 5 份 + 同秒连写 7 次(keep=2) 仍留 v=6 回滚点",
       keep5_ok and keep2_ok)

    with tempfile.TemporaryDirectory(prefix="si_keepnone_") as td:
        p = Path(td) / "x.json"
        p.write_text("{}", encoding="utf-8")
        for i in range(7):
            atomic_write_with_backup(p, {"i": i}, keep_backups=None)
        ck("keep_backups=None 写时不裁剪（7 份都在）",
           len(list(Path(td).glob("x.json.bak-*"))) == 7)

    # save_collections 的 .bak-* glob 清理依赖文件名前缀 —— 收敛后必须仍生效
    with tempfile.TemporaryDirectory(prefix="si_coll_bak_") as td:
        p = Path(td) / "collections.json"
        p.write_text("{}", encoding="utf-8")
        for _i in range(7):
            save_collections(empty_collections(), p)
        ck("save_collections 的 .bak-* 前缀清理仍生效（留 5 份）",
           len(list(Path(td).glob("collections.json.bak-*"))) == 5)


# ----------------------------------------------------------------------
# 15. 冲突会话：「模式」列带可见提示，数据一动不动
# ----------------------------------------------------------------------
def _check_mode_conflict_hint(ck) -> None:
    """冲突会话：界面提示与 build_index 同源；只提示、不改数据。"""
    print("----- 15. 冲突会话：「模式」列可见提示（纯数据层）-----")
    from apps.data_manager import mode_cell_text

    with tempfile.TemporaryDirectory(prefix="si_conflict_") as td:
        root = Path(td) / "stereo"
        _mk_session(root, "parallel/20261104-140552", "s-conflict", mode="angled")
        _mk_session(root, "parallel/20261105-120000", "s-normal", mode="parallel")
        items = {it.sid: it for it in scan_items(root)}
        c, n = items.get("s-conflict"), items.get("s-normal")
        text_c = mode_cell_text(c) if c is not None else ""
        text_n = mode_cell_text(n) if n is not None else ""
        print(f"    冲突行：{text_c!r}；普通行：{text_n!r}")
        ck("冲突行文案带 ⚠ 且写清 目录/记录",
           "⚠" in text_c and "目录=parallel" in text_c
           and "记录=angled" in text_c)
        ck("冲突行仍保留原模式文案（大角度）", text_c.startswith("大角度"))
        ck("非冲突行不带 ⚠，等于原 mode_label",
           "⚠" not in text_n and n is not None and text_n == n.mode_label)

        index, _ = build_index(root, collections=_collections_with([]))
        by_rel = {s["rel_path"]: s for s in index["sessions"]}
        ck("提示判据与 build_index 的 mode_conflict 同源",
           by_rel["parallel/20261104-140552"]["mode_conflict"] is True
           and by_rel["parallel/20261105-120000"]["mode_conflict"] is False)
        ck("未记录模式的会话不误报冲突（不算「不知道」为冲突）",
           "⚠" not in mode_cell_text(SessionItem(
               path=root / "parallel" / "20261106-120000", id="u", created="",
               frames_left=0, frames_right=0, n_matched=0, status="ok",
               size_bytes=0, stereo_mode="")))

    # 真实数据：口径不变、索引里不许有幽灵会话
    #
    # 2026-11-11：原断言是「唯一 1 处冲突仍在 parallel/20261104-140552、未搬目录」。
    # 用户当天 08:35 在 GUI 里用「改装置模式…」把这段会话从 `parallel/` 归位到
    # `angled/`（该目录里留着 session.json.bak-20261111-083535），冲突已解决 ——
    # 旧断言等于"要求用户永远别修数据"，它红是对的、但红错了对象。
    # 改成断言**归位生效 + 索引没有幽灵会话**：不再钉"冲突必须存在"
    # （用户下次录错分区还会产生新冲突，那时不该红），但仍在钉"索引不许记录
    # 磁盘上不存在的会话"——`data/index.json` 是派生缓存，最容易和磁盘脱节。
    stereo = ROOT / "data" / "stereo"
    if not stereo.is_dir():
        print("  [跳过] 无真实 data/stereo")
        return
    index, _ = build_index(stereo, collections=_collections_with([]))
    ghosts = [s["rel_path"] for s in index["sessions"]
              if not (stereo / s["rel_path"] / "session.json").is_file()]
    if ghosts:
        print(f"    幽灵会话：{ghosts[:5]}")
    # 注意别用 all([])：冲突清零后 hits 为空，all() 恒真、判据白给。
    ck("真实数据：索引里没有幽灵会话（每段会话的 session.json 都在磁盘上）",
       bool(index["sessions"]) and not ghosts)
    # 原来这里钉死 `angled/20261104-140552/session.json` 必须存在 —— 那是
    # 2026-11-11 用户手工归位的一次性快照。2026-11-18 用户自己清掉了那段目录，
    # 判据就变成假红：索引里 mode_conflict 实际为 0，意图早已达成，红的只是
    # "那段历史数据必须永远在那儿"。改成钉**不变量**：索引里不许有任何
    # 记录模式与目录分区冲突的会话。仍保留 bool(index["sessions"]) 前置，
    # 避免冲突清零后 any() 恒假的空断言（上面 ghosts 那条同理）。
    ck("真实数据：索引里没有模式冲突（记录模式与磁盘分区一致）",
       bool(index["sessions"])
       and not any(s.get("mode_conflict") for s in index["sessions"]))


# ----------------------------------------------------------------------
# 16. items 全收 / index["sessions"] 只收有 session.json 的
# ----------------------------------------------------------------------
def _check_index_scope(ck) -> None:
    """列表与索引的**收口差异**：孤儿看得见，但不进时间状态表。

    为什么必须有一条**正向**判据：第 15 节末尾那条是"索引里没有幽灵会话"，
    它只是否定式 —— 把 ``build_index`` 的过滤删掉会红，但把 ``scan_items``
    也一起改成只收 ``session.json``（= 干脆从界面上把孤儿藏起来）它也照样绿。
    真正要同时钉住的是两件事：

    1. 数据管理列表**宽**：只有 ``derived/annotated.mp4``、没有 ``session.json``
       的目录（实时预览被中断留下的孤儿）必须看得见，用户才知道磁盘上多了
       几十 MB 的东西；
    2. ``index.json`` **窄**：它是"时间状态表"，记一个没有元数据的目录没有
       意义，且它是最容易和磁盘脱节的派生缓存。
    """
    print("----- 16. items 全收 / index['sessions'] 只收有 session.json 的 -----")
    with tempfile.TemporaryDirectory(prefix="si_scope_") as td:
        root = Path(td) / "stereo"
        real = _mk_session(root, "parallel/20261110-120000", "real-1")
        (real / "derived").mkdir()
        # 孤儿：形状与 data/stereo/parallel/20261118-151122 一致（只有标注视频）
        orphan = root / "parallel" / "20261110-130000" / "derived"
        orphan.mkdir(parents=True)
        (orphan / "annotated.mp4").write_bytes(b"\x00" * 32)

        index, items = build_index(root, collections=_collections_with([]))
        item_rels = {it.rel_path for it in items}
        sess_rels = {s["rel_path"] for s in index["sessions"]}
        print(f"    items={sorted(item_rels)}")
        print(f"    sessions={sorted(sess_rels)}")

        ck("items 两个都在（宽判据：孤儿也要在数据管理列表里看得见）",
           item_rels == {"parallel/20261110-120000",
                         "parallel/20261110-130000"})
        ck("index['sessions'] 只有第一个（时间状态表不收没有 session.json 的）",
           sess_rels == {"parallel/20261110-120000"})
        ck("两者数量差 1，差集恰好是那个孤儿目录（不多不少）",
           len(item_rels) - len(sess_rels) == 1
           and item_rels - sess_rels == {"parallel/20261110-130000"}
           and not (sess_rels - item_rels))
        ck("index['n_sessions'] 与 sessions 条数一致（计数不跟着 items 走）",
           index["n_sessions"] == len(sess_rels) == 1)


def main() -> int:
    print("=" * 70)
    print("数据管理数据层自测（集合树 / 会话枚举 / 索引重建）")
    print("=" * 70)
    results: list[tuple[str, bool]] = []

    def ck(label: str, good: bool) -> None:
        results.append((label, good))
        print(f"  [{'通过' if good else '不达标'}] {label}")

    _check_group_of(ck)
    _check_three_levels(ck)
    _check_chinese_roundtrip(ck)
    _check_unfiled(ck)
    _check_bad_json_untouched(ck)
    _check_normalize_fixes(ck)
    _check_empty_dirs_survive(ck)
    _check_two_views_same_set(ck)
    _check_legacy_unchanged(ck)
    _check_index_rebuild(ck)
    _check_sid_stable(ck)
    _check_sid_collision(ck)
    _check_protected_outputs(ck)
    _check_backup_helper(ck)
    _check_mode_conflict_hint(ck)
    _check_index_scope(ck)

    print()
    ok = all(v for _n, v in results)
    bad = [n for n, v in results if not v]
    if bad:
        print(f"不达标 {len(bad)} 项：")
        for n in bad:
            print(f"  · {n}")
    print("\n判定结果：" + ("全部通过" if ok else "有不达标项"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
