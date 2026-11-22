"""自动绑定纯逻辑：serial 锚点 + 排除法。

不打开相机、不写配置；只回答“这两台外接相机里哪台是 left/right”。
命令行/未来 GUI 负责取父设备属性和写 `config/cameras.json`。
"""

from __future__ import annotations

from dataclasses import dataclass

from . import camera_registry as cr


class AutobindError(RuntimeError):
    """自动绑定失败（不够两台 / 出现第三台 / serial 对不上）。"""


@dataclass(frozen=True)
class RoleCandidate:
    device: cr.DeviceEntry
    parent: str = ""
    serial: str = ""

    @property
    def label(self) -> str:
        return (f"#{self.device.order} {self.device.name or '(无名字)'} "
                f"port={self.device.port or '-'} serial={self.serial or '-'}")


def external_candidates(
    devices: list[cr.DeviceEntry],
    *,
    settings: dict | None = None,
) -> list[cr.DeviceEntry]:
    """外接、非虚拟、非避免名单的相机；内置/IR/虚拟不参与自动绑定。"""
    out: list[cr.DeviceEntry] = []
    for d in devices:
        if not d.present or not d.identity.upper().startswith("USB\\"):
            continue
        if cr.is_virtual(d.name, settings) or cr.is_avoided(d.name, settings):
            continue
        out.append(d)
    return out


def choose_roles(
    candidates: list[RoleCandidate],
    *,
    left_serial: str = "",
    right_serial: str = "",
) -> tuple[RoleCandidate, RoleCandidate]:
    """确定 (left,right)。

    优先级：

    1. 两个 serial 都给了 -> 各自精确匹配；
    2. 只给 left serial -> left 精确匹配，right = 剩下那台；
    3. 没给锚点且恰好只有一台有 serial -> 有 serial 的是 left，另一台 right
       （这是本机双目的实际情况：left 有 `20190322001`，right 没有）；
    4. 其它情况一律报错，不猜。
    """
    if len(candidates) != 2:
        raise AutobindError(
            f"需要恰好 2 台外接相机才能自动绑定，当前 {len(candidates)} 台："
            + "；".join(c.label for c in candidates))
    one, two = candidates
    ls = str(left_serial or "").strip().casefold()
    rs = str(right_serial or "").strip().casefold()

    def _by_serial(want: str) -> RoleCandidate | None:
        return next((c for c in candidates
                     if want and c.serial.casefold() == want), None)

    if ls and rs:
        left, right = _by_serial(ls), _by_serial(rs)
        if left is None or right is None or left is right:
            raise AutobindError(
                f"serial 对不上：left={left_serial!r} right={right_serial!r}；"
                + "；".join(c.label for c in candidates))
        return left, right
    if ls:
        left = _by_serial(ls)
        if left is None:
            raise AutobindError(
                f"left serial={left_serial!r} 不在当前设备里；"
                + "；".join(c.label for c in candidates))
        return left, two if left is one else one
    if rs:
        right = _by_serial(rs)
        if right is None:
            raise AutobindError(
                f"right serial={right_serial!r} 不在当前设备里；"
                + "；".join(c.label for c in candidates))
        return two if right is one else one, right
    serialized = [c for c in candidates if c.serial]
    if len(serialized) == 1:
        left = serialized[0]
        return left, two if left is one else one
    raise AutobindError(
        "无法用 serial 自动定左右：两台都没有 serial，或两台都有但没有锚点；"
        "请给 --left-serial/--right-serial，或改走画面指纹。"
        + "；".join(c.label for c in candidates))


__all__ = ["AutobindError", "RoleCandidate", "choose_roles", "external_candidates"]
