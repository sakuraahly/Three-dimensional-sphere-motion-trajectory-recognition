"""相机画质会诊自测：把"糊"和"卡"分开的判据，全部用合成图验证（不需要相机）。

为什么需要它
------------
这一块的价值全在"结论对不对"上：如果会诊把**带宽问题**说成**光学问题**，
人就会去擦一个本来没脏的镜头，白折腾半天还是标定不了。
所以每条判据都要用**已知答案的合成图**钉住。

守着的三件事：
1. `calib.capture.diagnose_frame()` —— 棋盘没检测到时，能不能指出"为什么"；
2. `apps/tools/camera_quality.py::_is_partial()` —— 能不能认出"下半幅是常量"
   这种**传输没传完**的残帧，同时**不误伤**正常画面；
3. `AutoPairCollector.side_lines()` —— 逐路状态在"什么都没检测到"时也不能崩。

用法::

    python tests/selftest_camera_quality.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
while not (ROOT / "sphere_tracker").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from calib.capture import AutoPairCollector, diagnose_frame  # noqa: E402


def _load_tool():
    """按路径加载 apps/tools/camera_quality.py（它是脚本，不在包里）。

    必须先把模块登记进 `sys.modules`：`@dataclass` 在解析类型注解时会去
    `sys.modules[cls.__module__]` 取模块的命名空间，不登记就报
    `'NoneType' object has no attribute '__dict__'`。
    """
    spec = importlib.util.spec_from_file_location(
        "camera_quality", ROOT / "apps" / "tools" / "camera_quality.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _textured(h: int = 480, w: int = 640, seed: int = 0) -> np.ndarray:
    """一张"有细节"的灰度图（真实传感器一定带噪声）。"""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (h, w), dtype=np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    cv2.rectangle(img, (80, 60), (300, 260), 240, -1)
    cv2.rectangle(img, (360, 140), (560, 380), 15, -1)
    return img


# --------------------------------------------------------------------------- #
def test_diagnose() -> bool:
    print("----- 1. 棋盘没找到时，能不能说出「为什么」 -----")
    sharp = _textured()
    checks: list[tuple[str, bool]] = []

    # 清晰有细节的图：不该乱报问题（否则会把用户引到错误方向）
    note = diagnose_frame(sharp)
    checks.append(("正常画面不误报（返回空）", note == ""))
    print(f"  正常画面   -> {note or '(无异常，符合预期)'}")

    # 蒙雾：对比被压扁 + 整体抬亮 + 轻微失焦 —— "镜头脏/起雾/保护膜"的指纹。
    # 注意判据用的是**动态范围**（p95-p5）而不是整幅标准差：画面里有大块纯色
    # （白墙、木桌）会把标准差抬高，用标准差会漏判。
    haze = cv2.GaussianBlur(sharp, (0, 0), 1.2)
    haze = (haze.astype(np.float32) * 0.28 + 160).clip(0, 255).astype(np.uint8)
    note = diagnose_frame(haze)
    ok = ("雾" in note) or ("范围" in note)
    checks.append(("蒙雾画面 -> 指到镜头/雾", ok))
    print(f"  蒙雾画面   -> {note}")

    # 过曝：整幅接近 255
    over = np.full((480, 640), 252, np.uint8)
    note = diagnose_frame(over)
    ok = "过曝" in note
    checks.append(("过曝画面 -> 指到逆光/曝光", ok))
    print(f"  过曝画面   -> {note}")

    # 过暗
    dark = (sharp.astype(np.float32) * 0.06).astype(np.uint8)
    note = diagnose_frame(dark)
    ok = ("暗" in note) or ("黑" in note)
    checks.append(("过暗画面 -> 指到补光/曝光", ok))
    print(f"  过暗画面   -> {note}")

    # 死黑（镜头被挡）
    blocked = np.zeros((480, 640), np.uint8)
    note = diagnose_frame(blocked)
    ok = bool(note)          # 必须给出一句话，不能沉默
    checks.append(("全黑画面 -> 有话说", ok))
    print(f"  全黑画面   -> {note}")

    ok_all = True
    print()
    for name, passed in checks:
        print(f"  [{'通过' if passed else '不达标'}] {name}")
        ok_all = ok_all and passed
    return ok_all


# --------------------------------------------------------------------------- #
def test_partial_frame() -> bool:
    print("----- 2. 残帧判定（传输没传完 = 下半幅是常量） -----")
    tool = _load_tool()
    checks: list[tuple[str, bool]] = []

    # 正常画面：绝不能误判 —— 真实传感器一定有噪声
    normal = _textured()
    checks.append(("正常画面不误判", tool._is_partial(normal) is False))

    # 下半幅被填成常量：这是残帧的指纹
    partial = normal.copy()
    partial[int(480 * 0.8):, :] = 0
    checks.append(("下半幅常量 -> 判为残帧", tool._is_partial(partial) is True))

    # 上半幅本来就是纯色（比如对着白墙）时，不该说是残帧
    wall = np.full((480, 640), 200, np.uint8)
    checks.append(("整幅纯色不判残帧（无内容可比）", tool._is_partial(wall) is False))

    # 带噪声的暗背景：有噪声 = 真实传输，不判残帧
    rng = np.random.default_rng(7)
    noisy = rng.integers(0, 6, (480, 640), dtype=np.uint8) + 10
    checks.append(("带噪声的暗背景不判残帧", tool._is_partial(noisy) is False))

    # 太小的图不该崩
    checks.append(("超小图不崩", tool._is_partial(np.zeros((10, 10), np.uint8)) is False))

    ok_all = True
    print()
    for name, passed in checks:
        print(f"  [{'通过' if passed else '不达标'}] {name}")
        ok_all = ok_all and passed
    return ok_all


# --------------------------------------------------------------------------- #
def test_side_lines() -> bool:
    print("----- 3. 逐路状态（标定窗口右上角那两行） -----")
    c = AutoPairCollector.__new__(AutoPairCollector)
    c.last_obs = (None, None)
    lines = c.side_lines()
    ok1 = len(lines) == 2 and all("-" in x for x in lines)
    print(f"  什么都没检测到 -> {lines}")
    print(f"  [{'通过' if ok1 else '不达标'}] 不崩且给出两路占位")
    return ok1


def main() -> int:
    print("=" * 62)
    print("相机画质会诊自测（全合成图，不打开任何相机）")
    print("=" * 62)
    results = [test_diagnose(), test_partial_frame(), test_side_lines()]
    print()
    ok = all(results)
    print("判定结果：" + ("全部通过" if ok else "有不达标项"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
