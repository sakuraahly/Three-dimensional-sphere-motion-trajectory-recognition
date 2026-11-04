"""FOV 记录的读写：**标定产物 ↔ 装置档案**。

为什么单独一个模块
------------------
同一段逻辑要在两个地方用：双目窗标定成功后的自动收尾（`_gui_stereo`），
和 FOV 管理器里的「从当前标定读入」按钮（`fov_dialog`）。抄两份的代价很具体
—— 一边改了字段白名单、另一边没改，就会出现"标定自动记进去的能用、
手动读入的缺字段"这种只在某条路径上炸的缺陷。

本模块只做文件读写与字段搬运：不 print、不起 GUI、不读 `sys.argv`。
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime
from pathlib import Path
from typing import Any

#: 从标定 `meta.fov` 搬进档案时要带的字段。
#: 前 8 个是 `FovAlignment` 的完整字段（少一个 `FovAlignment.from_dict` 会拒）；
#: 后 2 个是"这份记录还能不能信"的依据（像素量纲，换分辨率就失效）。
FOV_KEYS = ("reference_side", "crop_side", "crop_rect", "target_size",
            "scale", "method", "inliers", "quality",
            "left_size", "right_size")


def fov_payload_from_calibration(json_path: str | Path) -> dict[str, Any] | None:
    """从标定 JSON 里取出 FOV 记录，补上 ``measured_at`` / ``source``。

    ## 读的是**顶层** ``fov``，不是 ``meta.fov``（2026-11-04 修的真 bug）

    `calib.params.StereoRig.load()` 是把**整个 JSON 的顶层**塞进 ``rig.meta``：

        meta = json.loads(js.read_text(...));  return cls(..., meta=meta)

    所以 `record_stereo` 里的 ``rig.meta.get("fov")`` 读的就是顶层 ``fov``。
    本函数第一版写成 ``data["meta"]["fov"]``，而标定 JSON 根本没有 ``meta``
    这一层 —— 于是「从当前标定读入」永远报"没有 FOV 记录"，标定成功后的
    自动收尾也永远不生效，而**两边都不报错**。

    返回 ``None`` 表示"这份标定没有 FOV 记录"（旧流程 / ``--skip-fov`` /
    现场估计降级成 ``fov=null``）—— 调用方**不要**把它当成空记录去覆盖档案。
    文件读不出 / JSON 坏了也返回 ``None``：这是收尾动作，不该让标定被它拖挂。
    """
    path = Path(json_path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    raw = data.get("fov")
    if raw is None:
        # 兼容分层写法（别人手工造的 json / 将来的格式）：优先顶层，其次 meta。
        nested = data.get("meta")
        raw = nested.get("fov") if isinstance(nested, dict) else None
    if not isinstance(raw, dict) or not raw:
        return None
    payload = {k: raw[k] for k in FOV_KEYS if k in raw}
    if "target_size" not in payload or "scale" not in payload:
        # 连最基本的两项都没有：不是一份能用的记录，别往档案里写。
        return None
    payload["measured_at"] = datetime.now().isoformat(timespec="seconds")
    payload["source"] = f"calibrate:{path.name}"
    return payload


def save_to_profile(profile, payload: dict[str, Any],
                    path: str | Path | None = None) -> None:
    """把 FOV 记录写进这个装置档案（同名覆盖其他字段不动）。

    ``path`` 是档案文件路径；``None`` = 默认 ``config/profiles.json``。
    **测试必须显式传临时路径** —— 不传就会写到真实档案里，多出一条测试装置。
    这不是假设：本模块第一版漏了透传，判据跑一遍就往真配置里塞了条 `fov-rig`。

    校验交给 `stereo.profiles.upsert` → `_check_fov`：档案层的规则只该有一份。
    """
    from stereo import profiles

    profiles.upsert(dataclasses.replace(profile, fov=payload),
                    replace_existing=True, path=path)


def record_fov_from_calibration(profile, calib_json: str | Path,
                                path: str | Path | None = None) -> str:
    """一条龙：读标定 JSON → 写档案 → 返回一句给人看的结果。

    返回空串表示这次没有可记的东西（标定没带 FOV）；异常情况返回带括号的
    说明句，**不抛** —— 调用方通常刚跑完一次成功的标定，不该因为档案更新
    失败而让用户以为标定挂了。
    """
    name = getattr(profile, "name", "") or ""
    if not name:
        return ""
    payload = fov_payload_from_calibration(calib_json)
    if payload is None:
        return ""
    try:
        save_to_profile(profile, payload, path)
    except Exception as exc:                                # noqa: BLE001
        return f"（FOV 没能记进档案：{exc}）"
    scale = payload.get("scale")
    tail = f"，scale={scale:.4f}" if isinstance(scale, (int, float)) else ""
    return f"FOV 已记进装置档案「{name}」{tail}（下次标定直接复用）"


def calibration_json_path(base: str | Path) -> Path:
    """标定基名 → 它的 JSON 路径（`x` / `x.npz` / `x.json` 都归一成 `x.json`）。"""
    p = Path(base)
    return p if p.suffix.lower() == ".json" else p.with_suffix(".json")
