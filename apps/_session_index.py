"""数据管理的**组织层**：会话怎么分组、索引怎么建、状态表存在哪。

三件互不重叠的事，都放在这里，因为它们的**唯一裁决源**必须是同一份：

1. **分组**（``collections.json``）：用户怎么把会话组织成"实验1 / 实验2"这种
   小目录，允许任意嵌套。**虚拟的**——存在一个 JSON 里，不动磁盘上的目录。
2. **枚举**（``discover_session_dirs``）：一个根下面到底有哪些会话。递归、
   命中即剪枝，只此一份口径。实现**不在这里**，在
   ``sphere_tracker/discovery.py``（见下面"枚举为什么不在这个文件里"）。
3. **状态表**（``index.json``）：时间 + 状态 + 帧数 + 大小的快照，**派生、
   可重建**，删了不心疼。

枚举为什么不在这个文件里（2026-11-09）
--------------------------------------
两个约束把它推出去了：``stereo/sessions.py`` 的 ``list_sessions`` 也要用同一份
枚举（所以不能放 ``apps/``，``stereo/`` 不许 import ``apps/``）；又不能放
``stereo/`` —— 那会让任何 import 它的进程先执行 ``stereo/__init__.py``，
把 cv2 + scipy 一起拉起来（实测数据管理窗口启动 0.06s → 1.56s）。所以放在
分层最底层的 ``sphere_tracker/discovery.py``（纯 stdlib，上三层都依赖它）。
这里只转发导入，名字保持可从 ``apps._session_index`` 取（现有测试与调用方
都这么导入）。以前三处各扫一遍，同一份磁盘数出 29 / 44 两个数：
"数据管理看得见，会话列表看不见"。同一个事实只能有一个产地。

为什么分组是"虚拟"的而不是直接挪目录
------------------------------------
用户的原话是要"像文件资源管理器那样"看小目录里的数据。真去挪目录有四个问题：

- 会话目录的路径被 **7 处代码**假设成两层（``<根>/<时间戳>`` 或
  ``<根>/<模式>/<时间戳>``），多一层就有扫描器看不见（见
  ``apps/_stereo_layout.py`` 的历史说明）；
- ``session.json`` 里记的标定路径、``derived/`` 里的相对引用都会跟着错；
- 挪 1.3 GB 会失败在半途（U 盘、占用、断电），而"挪了一半"没有任何回滚；
- 用户想要的其实是**视图**，不是**搬家**——同一段会话同时属于"11 月实验"
  和"梨子"是很自然的事，物理目录做不到。

所以：``collections.json`` 只存 ``sid → 节点``，磁盘一动不动。真要交付给别人时
用 ``apps/package_export.py`` 物化成自包含包（P2），而不是在本地搬家。

为什么 sid 用 session.json 的 id 而不是路径
-------------------------------------------
"移动后 sid 不变"是硬要求：用户在资源管理器里把 ``stereo/<ts>`` 拖进
``stereo/parallel/`` 是很常见的整理动作，索引不该因此失效、勾选不该丢。
目录名会变、相对路径会变，而 ``session.json`` 的 ``id`` 跟着文件走。

缺 ``id`` 的老会话退回目录名（目录名是时间戳，本仓库里 44 段全都不重复）；
万一真撞了，``build_index`` 按**路径排序**确定性加后缀 ``~2``、``~3``，
重建两次结果一样 —— 不能是"看谁先扫到"。

分组只有一个产地
----------------
``group_of()`` 是唯一入口。树节点、``index.json`` 的 ``group`` 字段、
导出包的 ``manifest.folder`` 全部调它。任何地方再写一次"按父目录名拼分组"，
就等着两处对不上吧。
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Iterable

#: 枚举的**唯一实现**在 ``sphere_tracker/discovery.py``（最底层，纯 stdlib）。
#: 这里只转发导入：调用方和测试仍然
#: ``from apps._session_index import discover_session_dirs``，但磁盘上只有一份遍历
#: 逻辑。为什么不在 ``stereo/``：那会让任何 import 它的进程先执行
#: ``stereo/__init__.py``（cv2 + scipy，实测 +1.5s），数据管理窗口白付这笔账。
from apps._common import (  # noqa: E402 —— 备份/原子写的唯一实现（AGENTS.md §4 第 7 条）
    DEFAULT_KEEP_BACKUPS,
    atomic_write_with_backup,
    tmp_name,
)
from sphere_tracker.discovery import (  # noqa: F401 —— 全部是转发导出
    discover_session_dirs,
    looks_like_session_dir,
    rel_key,
    sid_of,
)

__all__ = [
    "COLLECTIONS_PATH", "COLLECTIONS_SCHEMA", "INDEX_PATH", "INDEX_SCHEMA",
    "ROOT_NODE", "UNFILED_LABEL",
    "all_group_paths", "all_node_paths", "build_index", "collection_map",
    "collection_paths_by_node", "disk_group_of",
    "discover_session_dirs", "empty_collections", "group_of", "load_collections",
    "load_index", "looks_like_session_dir", "new_node_id", "node_children",
    "normalize_collections", "paths_for_root", "rel_key", "save_collections",
    "save_index", "sid_of",
]

#: 虚拟集合树。放在 ``data/`` 而不是 ``data/stereo/``：它描述的是**所有**会话
#: 的组织方式，不该被某个装置分区框住。
COLLECTIONS_PATH = Path(__file__).resolve().parents[1] / "data" / "collections.json"
#: 时间状态表（派生、可重建）。
INDEX_PATH = Path(__file__).resolve().parents[1] / "data" / "index.json"

COLLECTIONS_SCHEMA = "sphere-collections"
INDEX_SCHEMA = "sphere-session-index"
_SCHEMA_VERSION = 1

#: 集合树的根节点 id。**固定字符串**，别改成 uuid：配置文件是人写的，
#: 根节点是唯一一个必须永远存在的节点，写死才能"读坏了也能重建"。
ROOT_NODE = "col-root"
#: 没有归入任何集合的会话，在树上挂在这个名字下面。
UNFILED_LABEL = "散会话"


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ----------------------------------------------------------------------
# 虚拟集合树（collections.json）
# ----------------------------------------------------------------------
def paths_for_root(root: str | Path) -> tuple[Path, Path]:
    """给定数据根，返回 ``(集合配置, 状态表)`` 的路径：**放在根的同级**。

    为什么不是写死 ``data/collections.json``：用户会用『数据目录…』把面板指向
    别的盘（``E:/实验数据/stereo``），配置得跟着那份数据走。写死的话，
    换一个根以后界面显示的是**另一棵树**的分组，点开就是一堆找不到的会话。

    对默认根 ``data/stereo`` 来说，结果正好是 ``data/collections.json`` 与
    ``data/index.json`` —— 谁也没搬走。
    """
    base = Path(root).resolve().parent
    return base / "collections.json", base / "index.json"


def empty_collections() -> dict:
    """空的集合配置：只有一个根节点，所有会话都落在「散会话」里。"""
    return {
        "schema": COLLECTIONS_SCHEMA,
        "version": _SCHEMA_VERSION,
        "updated": _now(),
        "nodes": [{"id": ROOT_NODE, "name": "全部会话",
                   "parent_id": None, "session_ids": []}],
    }


def new_node_id(taken: Iterable[str], name: str = "") -> str:
    """生成一个没被占用的节点 id。

    不用自增序号：两份配置合并（``col-3`` 各指一个节点）时会撞车，而撞车
    在树上的表现是"两个目录里出现同一段会话"，很难查。名字 + 时间的短哈希
    既唯一又可读。
    """
    used = set(taken)
    for n in range(1, 1000):
        seed = f"{name}|{datetime.now().isoformat(timespec='microseconds')}|{n}"
        cand = "col-" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:8]
        if cand not in used:
            return cand
    raise RuntimeError("生成集合节点 id 失败：连续 1000 次都撞车，检查 taken 集合")


def normalize_collections(raw: object) -> tuple[dict, list[str]]:
    """把读进来的 JSON 修成**一定能用**的形状，并把修了什么记进 warnings。

    修而不是拒绝，因为这是用户手也能改的配置文件：一个 ``parent_id`` 打错
    不该让整个数据管理打不开。但**绝不静默** —— 每一处修正都进 ``warnings``，
    由调用方显示出来。

    规则：
    - 顶层不是对象 / schema 不认识 → 空配置；
    - 节点必须有非空 ``id`` 与 ``name``，否则丢弃；
    - ``parent_id`` 指向不存在的节点、指向自己、或成环 → 挂到根；
    - 同一个 ``sid`` 出现在多个节点 → **先出现的赢**，后面的移除（树形结构
      里一段会话只能有一个位置，否则"在树上"和"在表里"两套集合对不上）；
    - 根节点必须存在，名字为空就补回默认。
    """
    warns: list[str] = []
    data = empty_collections()
    if not isinstance(raw, dict):
        warns.append(f"collections.json 顶层不是对象（{type(raw).__name__}），已按空配置处理")
        return data, warns
    if raw.get("schema") not in (None, COLLECTIONS_SCHEMA):
        warns.append(f"collections.json 的 schema 是 {raw.get('schema')!r}，"
                     f"不是 {COLLECTIONS_SCHEMA}，已按空配置处理")
        return data, warns

    src_nodes = raw.get("nodes")
    if not isinstance(src_nodes, list):
        if src_nodes is not None:
            warns.append("collections.json 的 nodes 不是数组，已按空配置处理")
        return data, warns

    nodes: list[dict] = []
    by_id: dict[str, dict] = {}
    for i, item in enumerate(src_nodes):
        if not isinstance(item, dict):
            warns.append(f"nodes[{i}] 不是对象，已跳过")
            continue
        nid = item.get("id")
        name = item.get("name")
        if not isinstance(nid, str) or not nid.strip():
            warns.append(f"nodes[{i}] 缺少 id，已跳过")
            continue
        if nid in by_id:
            warns.append(f"节点 id 重复：{nid}，只保留第一个")
            continue
        node = {
            "id": nid,
            "name": str(name).strip() if isinstance(name, str) and name.strip() else nid,
            "parent_id": item.get("parent_id"),
            "session_ids": [],
        }
        by_id[nid] = node
        nodes.append(node)

    if ROOT_NODE not in by_id:
        root = {"id": ROOT_NODE, "name": "全部会话", "parent_id": None, "session_ids": []}
        by_id[ROOT_NODE] = root
        nodes.insert(0, root)
    else:
        by_id[ROOT_NODE]["parent_id"] = None

    # parent 修正：没有父 / 指向自己 / 不存在 / 成环 → 挂根。
    # 挂根而不是丢弃：用户手改错一个 parent_id，那个目录连同里面的会话
    # 不该整块消失（会话本身没坏，只是放错位置了）。
    for node in nodes:
        if node["id"] == ROOT_NODE:
            node["parent_id"] = None
            continue
        pid = node["parent_id"]
        if pid is None:
            node["parent_id"] = ROOT_NODE
            continue
        if not isinstance(pid, str) or pid not in by_id:
            warns.append(f"节点「{node['name']}」的 parent_id={pid!r} 不存在，已挂到根")
            node["parent_id"] = ROOT_NODE
            continue
        if pid == node["id"]:
            warns.append(f"节点「{node['name']}」的父节点是自己，已挂到根")
            node["parent_id"] = ROOT_NODE
            continue
        # 顺着 parent 往上走；超过节点数还没到顶就是环
        cur: str | None = pid
        steps = 0
        looped = False
        while cur is not None and steps <= len(nodes):
            if cur == node["id"]:
                looped = True
                break
            nxt = by_id.get(cur)
            cur = nxt["parent_id"] if nxt else None
            steps += 1
        if looped:
            warns.append(f"节点「{node['name']}」的父子关系成环，已挂到根")
            node["parent_id"] = ROOT_NODE

    # session_ids：全局唯一，先出现的赢；根节点也允许挂（= 根层散会话）
    owner: dict[str, str] = {}
    for i, item in enumerate(src_nodes):
        if not isinstance(item, dict):
            continue
        nid = item.get("id")
        if not isinstance(nid, str) or nid not in by_id:
            continue
        ids = item.get("session_ids")
        if not isinstance(ids, list):
            if ids is not None:
                warns.append(f"节点「{by_id[nid]['name']}」的 session_ids 不是数组，已清空")
            continue
        for sid in ids:
            if not isinstance(sid, str) or not sid.strip():
                continue
            sid = sid.strip()
            if sid in owner and owner[sid] != nid:
                warns.append(f"会话 {sid} 同时出现在两个目录，已保留在"
                             f"「{by_id[owner[sid]]['name']}」")
                continue
            if sid in by_id[nid]["session_ids"]:
                continue                       # 同一个节点里写了两遍，去重不报警
            owner[sid] = nid
            by_id[nid]["session_ids"].append(sid)

    data["nodes"] = nodes
    data["updated"] = str(raw.get("updated") or _now())
    return data, warns


def load_collections(path: str | Path | None = None) -> tuple[dict, list[str]]:
    """读集合配置；文件不存在/坏掉都返回**可用**的空配置 + 警告。

    文件坏掉时**不改写原文件**：用户可能手改错了一个逗号，程序一启动就把它
    覆盖成空配置，等于把用户的组织全删了。只有在用户真的改了东西
    （``save_collections``）时才覆盖，而那时已经留了备份。
    """
    p = Path(path) if path is not None else COLLECTIONS_PATH
    if not p.is_file():
        return empty_collections(), []
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return empty_collections(), [f"collections.json 读不了（{exc}）；"
                                     f"已按空配置显示，原文件未改动：{p}"]
    return normalize_collections(raw)


def save_collections(data: dict, path: str | Path | None = None,
                     keep_backups: int = DEFAULT_KEEP_BACKUPS) -> Path:
    """原子写集合配置，写前留备份（``collections.json.bak-<时间>``）。

    为什么留备份：这是**唯一一份**用户手工组织的结构，丢了要重新点一遍。
    格式、撞车、保留份数、临时文件 + ``os.replace`` 全部委托给
    ``apps._common.atomic_write_with_backup``（AGENTS.md §4 第 7 条唯一实现），
    这里只负责补齐 ``schema`` / ``version`` / ``updated`` 三个派生字段。
    """
    p = Path(path) if path is not None else COLLECTIONS_PATH
    data = dict(data)
    data["schema"] = COLLECTIONS_SCHEMA
    data["version"] = _SCHEMA_VERSION
    data["updated"] = _now()
    atomic_write_with_backup(p, data, keep_backups=keep_backups)
    return p


def collection_map(data: dict) -> dict[str, str]:
    """``sid → 逻辑路径``（``"实验1/梨子"``；根节点下是 ``""``）。

    逻辑路径**不含根节点名**：根是"全部会话"，它不是一个真的分组。把它拼进
    路径会让树节点名变成「全部会话/实验1」，既长又容易和用户自己建的
    「全部会话」撞名。

    路径本身由 ``collection_paths_by_node`` 算 —— 这里不再写第二份"顺着
    parent 往上拼名字"的循环，那样迟早会有一处漏了环保护。
    """
    paths = collection_paths_by_node(data)
    out: dict[str, str] = {}
    for node in data.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        nid = node.get("id")
        if nid not in paths:
            continue
        for sid in node.get("session_ids") or []:
            if isinstance(sid, str) and sid:
                out[sid] = paths[nid]
    return out


def all_node_paths(data: dict) -> set[str]:
    """``collections.json`` 里**所有**节点的逻辑路径（含一个会话都没有的空目录）。

    为什么要单独一份：``collection_map`` 只返回"有会话"的节点。用户刚点完
    「新建目录」还没往里放东西，如果树只按 ``collection_map`` 建，这个目录
    会当场消失 —— 用户会以为"没建成"，然后反复点。空目录必须显示。
    """
    out: set[str] = set()
    for path in collection_paths_by_node(data).values():
        if path:
            out.add(path)
    return out


def collection_paths_by_node(data: dict) -> dict[str, str]:
    """``节点 id → 逻辑路径``（不含根；根是 ``""``）。"""
    nodes = {n["id"]: n for n in data.get("nodes") or [] if isinstance(n, dict)
             and isinstance(n.get("id"), str)}
    out: dict[str, str] = {}
    for nid in nodes:
        if nid == ROOT_NODE:
            out[nid] = ""
            continue
        parts: list[str] = []
        cur: str | None = nid
        guard = 0
        while cur is not None and cur != ROOT_NODE and guard <= len(nodes):
            n = nodes.get(cur)
            if n is None:
                break
            parts.append(str(n.get("name") or cur))
            cur = n.get("parent_id")
            guard += 1
        out[nid] = "/".join(reversed(parts))
    return out


def node_children(data: dict) -> dict[str | None, list[dict]]:
    """``parent_id → 子节点列表``（按名字排序），供界面一次性建树。"""
    kids: dict[str | None, list[dict]] = {}
    for node in data.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        kids.setdefault(node.get("parent_id"), []).append(node)
    for lst in kids.values():
        lst.sort(key=lambda n: str(n.get("name") or ""))
    return kids


def disk_group_of(rel_path: str) -> str:
    """按**磁盘相对路径**推断分组：``parallel/20261104-140552`` → ``平行双目``。

    装置分区目录名（``parallel`` / ``angled``）换成给人看的说法，其余保持原样
    —— 用户自己建的 ``实验1`` 不该被改写。根层的散会话返回 ``""``。
    """
    from apps._stereo_layout import MODE_LABELS

    parts = [p for p in PurePosixPath(str(rel_path or "")).parts[:-1] if p and p != "."]
    return "/".join(MODE_LABELS.get(p.lower(), p) for p in parts)


def group_of(sid: str, mapping: dict[str, str], rel_path: str = "") -> str:
    """这段会话属于哪个分组：**虚拟集合优先，没有就按磁盘层级**。

    这是分组定义的**唯一产地**。树节点、``index.json``、导出包目录名全部
    调它；任何地方再写一次"父目录名就是分组"，就等着两处对不上。

    为什么要磁盘兜底而不是一律扔进「散会话」：用户完全可能直接在资源管理器里
    建 ``data/stereo/实验1/`` 然后把会话拖进去 —— 那是他的自由，界面必须
    如实显示，而不是视而不见。
    """
    if sid in mapping:
        return mapping[sid]
    return disk_group_of(rel_path)


def all_group_paths(groups: Iterable[str]) -> list[str]:
    """补出所有**中间层**分组路径，排序后返回（``""`` 在最前）。

    树是逐层建的：只有 ``实验1/梨子`` 里有会话时，``实验1`` 这一层也必须
    出现，否则那批会话在树上就是"孤儿"（父节点不存在，Treeview 里根本插不进去）。
    """
    seen: set[str] = {""}
    for g in groups:
        g = str(g or "")
        if not g:
            continue
        parts = g.split("/")
        for i in range(1, len(parts) + 1):
            seen.add("/".join(parts[:i]))
    return sorted(seen, key=lambda s: (s.count("/") if s else -1, s))


# ----------------------------------------------------------------------
# 时间状态表（index.json）
# ----------------------------------------------------------------------
def build_index(root: str | Path, *, recursive: bool = True,
                collections: dict | None = None,
                progress=None) -> tuple[dict, list]:
    """扫一遍根，返回 ``(index 字典, SessionItem 列表)``。

    只扫一次：``SessionItem`` 是给界面/导出用的内存形态，``index`` 是给
    ``index.json`` 用的落盘形态，两者同一份数据派生，不各自扫一遍（扫两遍
    就会出现"表里有、索引里没有"这种没法解释的现象）。

    **两者收的范围不同，这是有意的**：

    - ``items``（返回的第二项）**全收**，用的是枚举的宽判据（有 ``derived/``
      或 ``left.mkv``/``right.mkv``/``clip.mkv`` 也算）。它是"数据管理列表"，
      半成品、孤儿目录也得让用户看见 —— 否则用户不知道磁盘上多了个几十 MB
      的东西，也没法自己判断要不要清理。
    - ``index["sessions"]`` **只收磁盘上真有 ``session.json`` 的目录**。它是
      "时间状态表"（按时间排序 + 状态快照），记一个没有元数据的目录没有意义；
      而且 ``index.json`` 是最容易和磁盘脱节的派生缓存，少记一条不会有人发现。

    实测过的一条：``data/stereo/parallel/20261118-151122`` 只有
    ``derived/annotated.mp4``（实时预览被中断，``_SessionRecorder.save()``
    没跑完，``_discard()`` 没机会删目录），原本被当成会话写进索引。
    """
    from apps._data_export import scan_items

    root_p = Path(root)
    coll = collections if collections is not None else load_collections()[0]
    items = scan_items(root_p, recursive=recursive, collections=coll, progress=progress)
    mapping = collection_map(coll)

    sessions = []
    for it in items:
        # `items` 与 `index["sessions"]` 职责不同，**这里要过滤**：
        # items 是数据管理列表，宽判据（有 derived/ 或视频就算）是对的 ——
        # 半成品、孤儿目录也得让用户看见，否则他不知道磁盘上多了个几十 MB 的东西。
        # 但 index 是"时间状态表"，记一个没有 session.json 的目录没有意义；
        # 实测 data/stereo/parallel/20261118-151122 只有 derived/annotated.mp4，
        # 被当成会话收进索引 —— 而索引是最容易和磁盘脱节的派生缓存。
        if not (it.path / "session.json").is_file():
            continue
        # 摆的位置与记录的模式不一致：记录优先（落盘前已核对过模式与标定），
        # 但这件事必须**留下痕迹** —— 实测 `parallel/20261104-140552` 记的是
        # angled，是 2026-11-04 分区迁移时按错的元数据归的类。
        from apps._stereo_layout import layout_mode_of

        layout = layout_mode_of(it.path)
        conflict = bool(it.stereo_mode and layout and layout != it.stereo_mode)
        sessions.append({
            "sid": it.sid,
            "rel_path": it.rel_path,
            "group": it.group,
            "id": it.id,
            "label": it.label,
            "created": it.created,
            "frames_left": it.frames_left,
            "frames_right": it.frames_right,
            "status": it.status,
            "symbol": it.symbol,
            # 没记录就是 null：**不默认 parallel**。老会话缺这个字段，
            # 默认成 parallel 会让 90° 装置录的会话看起来"可用"，
            # 用错几何解算得到系统性错的深度而且不报错。
            "stereo_mode": it.stereo_mode or None,
            "mode_recorded": bool(it.stereo_mode),
            "layout_mode": layout,
            "mode_conflict": conflict,
            "size_bytes": it.size_bytes,
            "problems": list(it.problems),
        })

    index = {
        "schema": INDEX_SCHEMA,
        "version": _SCHEMA_VERSION,
        "built": _now(),
        "root": str(root_p),
        "recursive": bool(recursive),
        "n_sessions": len(sessions),
        "n_unfiled": sum(1 for s in sessions if not s["group"]),
        "n_mode_unrecorded": sum(1 for s in sessions if not s["mode_recorded"]),
        "n_mode_conflict": sum(1 for s in sessions if s["mode_conflict"]),
        "groups": all_group_paths(s["group"] for s in sessions),
        "sessions": sessions,
        "warnings": list(getattr(items, "warnings", []) or []),
        "note": ("派生缓存，删掉会自动重建；权威数据是每段会话的 session.json "
                 "与 data/collections.json"),
    }
    return index, items


def save_index(index: dict, path: str | Path | None = None) -> Path:
    """写状态表（同样原子替换，但**不留备份**：它是可再生的）。"""
    p = Path(path) if path is not None else INDEX_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    # 临时文件名同样取自 `apps._common.tmp_name`，不在这里写第二份格式串。
    tmp = p.with_name(tmp_name(p))
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(index, fh, ensure_ascii=False, indent=2, allow_nan=False)
            fh.write("\n")
        os.replace(tmp, p)
    except (OSError, ValueError) as exc:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise type(exc)(f"写入 {p} 失败：{exc}") from exc
    return p


def load_index(path: str | Path | None = None,
               root: str | Path | None = None) -> dict | None:
    """读状态表；不存在 / schema 不符 / 根不匹配都返回 ``None``（= 需要重建）。

    校验 ``root`` 是关键：用户用『数据目录…』把根换到另一块盘之后，
    旧索引里全是**别的树**的会话，直接拿来用会显示一堆点开就报错的幽灵行。
    """
    p = Path(path) if path is not None else INDEX_PATH
    if not p.is_file():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict) or raw.get("schema") != INDEX_SCHEMA:
        return None
    if raw.get("version") != _SCHEMA_VERSION:
        return None
    if not isinstance(raw.get("sessions"), list):
        return None
    if root is not None and str(raw.get("root") or "") != str(Path(root)):
        return None
    return raw
