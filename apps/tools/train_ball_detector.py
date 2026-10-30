"""用"已经点选过的数据"自举训练一个单类球检测器（YOLO11）。

## 为什么需要它（2026-10-30）

现场是一颗**镜面金属球**，经典方法全灭：color 0%、hough 1.4%、
稠密视差 58% 有效、最近物体法 0/12。唯一的共同点是"没有一个方法知道球长什么样"，
而用户其实已经在逐帧点过球心 —— 那份点选数据就是训练集。

所以这条链是：**点选 → 导出数据集（另一支） → 本脚本训练 → 接进 YoloDetector**。
本脚本只负责后半段，数据集切分由导出端按**会话**做好，这里不重切。

## 验收不只看 mAP，看"圆心误差"

检测器的输出是**种子**：后面跟踪、测深、预测全从它出发，圆心错 10 px，
半径再准也没用。而 mAP 高不等于圆心准（框大一圈也算 TP）。
所以校验时额外量：

- 每张图取置信度最高的框 → 圆心，与标签圆心比欧氏距离（**原始分辨率像素**）；
- 报中位 / p90 / recall@30px / recall@60px；
- 框再交给 ``refine_circle_in_box`` 细化一次，比半径相对误差（中位）。

30/60 px 这两个门是给下游挑的：本项目的球半径约 90 px，跟踪器的搜索窗
（``REFINE_SEARCH_PX=3``）只修半径不修圆心 —— 圆心误差超过球半径的一半，
后面就再也拉不回来。

## 前置检查为什么卡"会话数"和"正样本数"

- **会话数 < 2**：同一段录像整段进 train 或整段进 val（见数据契约），
  只有一个会话时 val 与 train 是同一段画面，mAP 会虚高到没有信息量；
- **正样本 < 200**：几十个框训出来的东西在复杂背景上等于没训，
  与其花 20 分钟得到一个"看起来能用"的权重，不如先说"数据不够"。

两条都不满足时**训练直接拒绝**（退出码 2）。``--dry-run`` 只做体检、不训练，
所以它始终按 0 退出 —— 它存在的意义就是"在数据还不够的时候也能跑一下看看差多少"。

用法::

    python apps/tools/train_ball_detector.py --data outputs/ball_dataset --dry-run
    python apps/tools/train_ball_detector.py --data outputs/ball_dataset --epochs 60
    python apps/tools/train_ball_detector.py --data outputs/ball_dataset --device 0 --out models/ball_yolo.pt
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# 与其余 apps 一致：从本文件往上找到含 sphere_tracker 的目录当项目根，
# 这样在任何工作目录下跑都能 import 到库，也能按项目根解析相对路径。
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "sphere_tracker").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: 训练集至少要有几个会话（同一录像只能落在一侧，见模块 docstring）。
MIN_SESSIONS = 2
#: 正样本框数下限。200 是"勉强能学出一类外观"的量级，不是"够好"。
MIN_POSITIVE_BOXES = 200
#: 圆心命中门（像素）。与球半径（约 90 px）的关系见模块 docstring。
CENTER_HIT_PX = (30.0, 60.0)
#: 默认基座权重（仓库自带，**绝不在脚本里下载**：现场常常没有网络）。
DEFAULT_BASE = "models/yolo11n.pt"
DEFAULT_OUT = "models/ball_yolo.pt"
#: 单次推理的最低阈值：下面那张阈值表都是在**这一次**原始输出上再筛出来的，
#: 不是跑几遍（同一个模型跑几遍只会多花几倍时间，数字一模一样）。
EVAL_CONF = 0.02
#: 阈值表的行。0.25 = 运行时默认（``DetectorConfig.yolo_conf``），加两档更宽的：
#: 20 epoch 的模型在 val 上最高分中位只有 ~0.06，0.25 一刀下去是 0 检出 ——
#: 只报默认值会把"阈值不匹配"误读成"模型没学会"。也加一档更严的做对照。
CONF_GRID = (0.02, 0.05, 0.10, 0.25, 0.50)
#: 按**标签框长边**分桶（像素）。24 px 的球在 imgsz=640 下只剩 8 px，
#: 拿一个平均值会把这种"必然检不到"的样本藏起来，所以要分开报。
SIZE_BUCKETS = ((0.0, 48.0, "<48 px"), (48.0, 120.0, "48~120 px"), (120.0, 1e9, ">120 px"))
#: 统计框尺寸时最多读多少张图（只为报一个分布，不值得把几千张全解码）。
SIZE_SAMPLE = 300

#: 小数据必须开增强 —— 200 张图在 640 分辨率下几轮就过拟合，
#: 增强是"用手上这几张变出更多视角"的唯一手段。
AUGMENT = dict(
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=10.0, translate=0.1, scale=0.5, fliplr=0.5,
)

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class DatasetError(RuntimeError):
    """数据集不可用（目录 / yaml / 目录结构 / 标签格式）。消息本身就要能当用户提示。"""


@dataclass
class DatasetStats:
    """数据集体检结果。字段都直接进报告，不再二次加工。"""

    root: Path
    yaml_path: Path
    train_images: list[Path] = field(default_factory=list)
    val_images: list[Path] = field(default_factory=list)
    train_boxes: int = 0
    val_boxes: int = 0
    train_negatives: int = 0
    val_negatives: int = 0
    train_sessions: dict[str, int] = field(default_factory=dict)
    val_sessions: dict[str, int] = field(default_factory=dict)
    box_px: list[float] = field(default_factory=list)

    @property
    def session_count(self) -> int:
        return len(self.train_sessions)


# --------------------------------------------------------------------------
# 纯函数：能被测试直接喂数字，不必造数据集
# --------------------------------------------------------------------------
def parse_session(stem: str) -> str:
    """从文件名前缀解析会话名：``s20261029_1430_f0001`` → ``s20261029``。

    解析不出来的（文件名里没有下划线）一律归到"未知会话"：
    宁可让它们挤成一个会话、让"会话数 < 2"的检查报出来，
    也不要按每张图算一个会话 —— 那样检查永远通过，等于没有检查。
    """
    text = str(stem)
    if "_" not in text:
        return "未知会话"
    head = text.split("_", 1)[0].strip()
    return head or "未知会话"


def preflight_problems(sessions: int, positives: int, has_ultralytics: bool) -> list[str]:
    """返回"现在不能训练"的中文原因列表；空列表 = 可以训。

    抽成纯函数是为了能直接断言（真造一份 200 框的数据集太贵）。
    每条都写清**数字 + 门槛 + 后果**，因为用户看到的只有这几行。
    """
    problems: list[str] = []
    if sessions < MIN_SESSIONS:
        problems.append(
            f"训练集只有 {sessions} 个会话（至少 {MIN_SESSIONS} 个）："
            "同一段录像整段进 train 或整段进 val，只有一个会话时 val 就是训练画面本身，"
            "mAP 会虚高到看不出任何问题")
    if positives < MIN_POSITIVE_BOXES:
        problems.append(
            f"正样本框只有 {positives} 个（至少 {MIN_POSITIVE_BOXES} 个）："
            "这么少的框在复杂背景上学不到稳定的球外观，训出来的权重「看起来能用」但会乱检")
    if not has_ultralytics:
        problems.append(
            "没装 ultralytics，无法训练。请运行：\n"
            "    pip install ultralytics\n"
            "（本脚本不会自动下载任何东西，包括它自己）")
    return problems


def labels_dir_for(images_dir: Path) -> Path:
    """``.../images/train`` → ``.../labels/train``（ultralytics 的约定）。"""
    parts = list(images_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts)
    return images_dir.parent / "labels"


def read_label_file(path: Path) -> list[tuple[float, float, float, float]]:
    """读一个 YOLO 标签文件，返回 ``[(cx, cy, w, h), ...]``（归一化）。

    坏行**跳过并计数**而不是整体报错：导出的数据里出现一行空行/半个框很常见，
    为此拒绝整个数据集只会让人去手改几千个 txt。
    """
    boxes: list[tuple[float, float, float, float]] = []
    if not path.is_file():
        return boxes
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            _cls, cx, cy, w, h = (float(v) for v in parts[:5])
        except ValueError:
            continue
        if w > 0.0 and h > 0.0:
            boxes.append((cx, cy, w, h))
    return boxes


def ultralytics_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("ultralytics") is not None


# --------------------------------------------------------------------------
# 数据集体检
# --------------------------------------------------------------------------
def _list_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def _resolve_dataset_dirs(yaml_data: dict, yaml_path: Path) -> tuple[Path, Path]:
    """按 ultralytics 的语义解析 train/val 目录。

    ``path`` 缺省时相对 yaml 所在目录；``train``/``val`` 再相对 ``path``。
    ``path`` 写成绝对路径也允许（导出端就是这么写的）。
    """
    base = Path(str(yaml_data.get("path") or "."))
    if not base.is_absolute():
        base = (yaml_path.parent / base)
    dirs = []
    for key in ("train", "val"):
        value = yaml_data.get(key)
        if not value:
            raise DatasetError(
                f"dataset.yaml 缺少 {key}：不知道该去哪找{'训练' if key == 'train' else '验证'}图片")
        p = Path(str(value))
        dirs.append(p if p.is_absolute() else (base / p))
    return dirs[0], dirs[1]


def _box_sizes_px(images: list[Path], labels_dir: Path, limit: int = SIZE_SAMPLE) -> tuple[list[float], int]:
    """抽样算"框长边多少像素"。

    为什么需要它：``--imgsz`` 该开多大取决于球在原始画面里占多少像素。
    球只有 20 px 时用 640 训练等于把球缩到 20 px 里再缩一次，学不到东西。
    """
    import cv2  # 局部 import：--dry-run 的体检用得上，但不该拖慢模块导入

    sizes: list[float] = []
    used = 0
    for img in images[:limit]:
        frame = cv2.imread(str(img))
        if frame is None:
            continue
        used += 1
        h_px, w_px = frame.shape[:2]
        for _cx, _cy, bw, bh in read_label_file(labels_dir / f"{img.stem}.txt"):
            sizes.append(max(bw * w_px, bh * h_px))
    return sizes, used


def collect_stats(data_dir: Path) -> DatasetStats:
    """读 dataset.yaml + 两个目录 + 标签，产出体检结果；任何致命问题抛 DatasetError。"""
    import yaml  # ultralytics 自带依赖，要求装了 ultralytics 就一定在

    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise DatasetError(
            f"数据集目录不存在：{data_dir}\n"
            "       先跑导出（apps/tools/export_ball_dataset.py）生成，"
            "或在 --data 里指到已导出的目录")
    yaml_path = data_dir / "dataset.yaml"
    if not yaml_path.is_file():
        raise DatasetError(f"找不到 {yaml_path}：数据集目录里没有 dataset.yaml，无法确定 train/val")
    try:
        payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001  yaml 的异常类型很杂，统一转成用户能读的一句
        raise DatasetError(f"{yaml_path} 解析失败：{exc}") from None
    if not isinstance(payload, dict):
        raise DatasetError(f"{yaml_path} 的内容不是一个映射（dict），读不出 train/val")
    names = payload.get("names")
    if not names:
        raise DatasetError(f"{yaml_path} 缺少 names（应该是 {{0: ball}}）")

    train_dir, val_dir = _resolve_dataset_dirs(payload, yaml_path)
    stats = DatasetStats(root=data_dir, yaml_path=yaml_path)
    stats.train_images = _list_images(train_dir)
    stats.val_images = _list_images(val_dir)
    if not stats.train_images:
        raise DatasetError(f"训练图片目录是空的：{train_dir}\n"
                           "       （导出没跑完 / 路径写错 / 图片后缀不是常见格式）")
    if not stats.val_images:
        raise DatasetError(f"验证图片目录是空的：{val_dir}\n"
                           "       没有 val 就无法回答「训完到底准不准」")

    for split, images in (("train", stats.train_images), ("val", stats.val_images)):
        labels_dir = labels_dir_for(images[0].parent)
        for img in images:
            boxes = read_label_file(labels_dir / f"{img.stem}.txt")
            session = parse_session(img.stem)
            bucket = stats.train_sessions if split == "train" else stats.val_sessions
            bucket[session] = bucket.get(session, 0) + 1
            if split == "train":
                stats.train_boxes += len(boxes)
            else:
                stats.val_boxes += len(boxes)
            if not boxes:
                if split == "train":
                    stats.train_negatives += 1
                else:
                    stats.val_negatives += 1

    stats.box_px, _used = _box_sizes_px(
        stats.train_images, labels_dir_for(stats.train_images[0].parent))
    return stats


def _mtime_text(path: Path) -> str:
    """文件的修改时间（本地时间，秒级）。取不到就明说，不假装有。"""
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime))
    except OSError:
        return "（读不到修改时间）"


def format_stats(stats: DatasetStats) -> list[str]:
    """数据集体检的人话摘要。"""

    def sessions_text(bucket: dict[str, int]) -> str:
        items = sorted(bucket.items(), key=lambda kv: (-kv[1], kv[0]))
        shown = "、".join(f"{name}×{n}" for name, n in items[:6])
        more = "" if len(items) <= 6 else f" 等 {len(items)} 个"
        return f"{len(bucket)} 个会话（{shown}{more}）"

    lines = [
        f"[数据] {stats.root}",
        # dataset.yaml 的 mtime 是"训练时数据集可能正在被重导"的唯一事后线索：
        # 2026-10-30 就出现过训练跑到一半导出端重写数据集（val 目录一度被删空），
        # 那次只能靠日志时间戳猜。写上它，至少能对一眼。
        f"       dataset.yaml 修改于 {_mtime_text(stats.yaml_path)}",
        f"       train {len(stats.train_images)} 图 / {stats.train_boxes} 框 / "
        f"{stats.train_negatives} 负样本 / {sessions_text(stats.train_sessions)}",
        f"       val   {len(stats.val_images)} 图 / {stats.val_boxes} 框 / "
        f"{stats.val_negatives} 负样本 / {sessions_text(stats.val_sessions)}",
    ]
    if stats.box_px:
        import numpy as np

        arr = np.asarray(stats.box_px, dtype=float)
        p10, p50, p90 = (float(v) for v in np.percentile(arr, [10, 50, 90]))
        lines.append(f"       框长边（抽样 {min(len(stats.train_images), SIZE_SAMPLE)} 图）："
                     f"中位 {p50:.0f} px，p10 {p10:.0f} / p90 {p90:.0f} px")
        if p90 < 64:
            lines.append("       [注意] 球在原始画面里只有几十像素 —— imgsz 开太大会把球缩没，"
                         "建议 320~480")
    else:
        lines.append("       框长边：读不到图片（解码失败），跳过尺寸分布")
    unknown = sum(n for name, n in stats.train_sessions.items() if name == "未知会话")
    if unknown:
        lines.append(f"       [警告] {unknown} 张图的文件名里没有下划线，会话按「未知会话」计；"
                     "会话数检查可能因此判错")
    return lines


# --------------------------------------------------------------------------
# 训练与验收
# --------------------------------------------------------------------------
def _per_class_map(metrics) -> dict[str, float]:
    """把 ultralytics 的 per-class mAP 摊平成 ``{类名: mAP50-95}``。

    不同版本字段名漂过（``maps`` / ``ap_class_index``），取不到就返回空字典 ——
    报告里少一格比因为一格取不到而整个训练失败划算得多。
    """
    try:
        maps = list(getattr(metrics.box, "maps", []) or [])
        names = getattr(metrics, "names", {}) or {}
        if isinstance(names, list):
            names = {i: n for i, n in enumerate(names)}
        out: dict[str, float] = {}
        for i, value in enumerate(maps):
            out[str(names.get(i, i))] = float(value)
        return out
    except Exception:  # noqa: BLE001
        return {}


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def refine_radius_for_report(frame, box: tuple[float, float, float, float],
                             base_cfg) -> float | None:
    """把框交给 ``refine_circle_in_box`` 细化，返回半径；细化不出来返回 ``None``。

    这里**刻意放宽半径搜索窗**：默认 ``hough_max_radius=80`` 比本项目的球
    （约 90 px）还小，细化必然失败 —— 那样报出来的"半径误差"量的是配置而不是方法。
    报告要回答的是"框交给细化器能到多准"，所以给它一个够大的窗。
    """
    from sphere_tracker.config import DetectorConfig
    from sphere_tracker.detectors.neural import refine_circle_in_box
    from sphere_tracker.types import Detection

    box_w = max(1.0, box[2] - box[0])
    box_h = max(1.0, box[3] - box[1])
    det = Detection(x=(box[0] + box[2]) / 2.0, y=(box[1] + box[3]) / 2.0,
                    radius=0.5 * max(box_w, box_h), score=1.0, method="eval",
                    extra={"box": box})
    wide = DetectorConfig(**{**base_cfg.__dict__,
                             "min_radius": 3.0,
                             "max_radius": 5000.0,
                             "hough_max_radius": 5000.0})
    refined = refine_circle_in_box(frame, det, wide)
    return None if refined is None else float(refined.radius)


def size_bucket(side_px: float) -> str:
    """按标签框长边归桶。纯函数，测试可以直接喂数字。"""
    for lo, hi, name in SIZE_BUCKETS:
        if lo <= side_px < hi:
            return name
    return SIZE_BUCKETS[-1][2]


def _threshold_row(deltas: list[float], with_boxes: int, n_all: int,
                   radii: list[float], refined: int, matched: int) -> dict:
    """把某一阈值的样本列表算成一行指标。

    两个命中率并存，因为分母该是谁取决于问题：
    - ``recall_30px`` 分母是**带框的 val 图**（能算圆心误差的那些），
      回答"模型在有球的时候准不准"；
    - ``recall_30px_all`` 分母是**全部 val 图**（含负样本），
      回答"打开这个检测器，一帧里能不能拿到可信种子"。
    只报第一个会让负样本上的误检消失，只报第二个会让"没有球"的图拖低模型评分。
    """
    import numpy as np

    arr = np.asarray([d for d in deltas if math.isfinite(d)], dtype=float)
    all_arr = np.asarray(deltas, dtype=float)
    return {
        "n_matched": matched,
        "n_missed": with_boxes - matched,
        "median_px": float(np.median(arr)) if arr.size else None,
        "p90_px": float(np.percentile(arr, 90)) if arr.size else None,
        "recall_30px": float(np.mean(all_arr <= CENTER_HIT_PX[0])) if all_arr.size else 0.0,
        "recall_60px": float(np.mean(all_arr <= CENTER_HIT_PX[1])) if all_arr.size else 0.0,
        "recall_30px_all": (float(np.sum(all_arr <= CENTER_HIT_PX[0])) / n_all) if n_all else 0.0,
        "radius_median_rel_error": float(np.median(radii)) if radii else None,
        "radius_refined_ratio": (refined / matched) if matched else 0.0,
    }


def _bucket_table(records: list[dict], threshold: float) -> dict[str, dict]:
    """按目标大小分桶统计命中率（分母是桶内全部带框图，漏检算失败）。"""
    import numpy as np

    table: dict[str, dict] = {name: {"n_labeled": 0, "n_detected": 0, "deltas": []}
                              for _lo, _hi, name in SIZE_BUCKETS}
    for rec in records:
        row = table[rec["bucket"]]
        row["n_labeled"] += 1
        d = rec["deltas"].get(threshold, float("inf"))
        row["deltas"].append(d)
        if math.isfinite(d):
            row["n_detected"] += 1
    out: dict[str, dict] = {}
    for name, row in table.items():
        finite = [d for d in row["deltas"] if math.isfinite(d)]
        out[name] = {
            "n_labeled": row["n_labeled"],
            "n_detected": row["n_detected"],
            "recall_30px": (sum(1 for d in row["deltas"] if d <= CENTER_HIT_PX[0])
                            / len(row["deltas"])) if row["deltas"] else 0.0,
            "median_px": float(np.median(finite)) if finite else None,
        }
    return out


def evaluate(model, stats: DatasetStats, *, imgsz: int, device: str,
             runtime_conf: float | None = None) -> dict:
    """在 val 上量"圆心误差 / 半径误差"，并按**运行时阈值**与**目标大小**分开报。

    为什么要分阈值（2026-10-30，team-lead 指出）：原来只在 conf=0.01 上评，
    top-1 框常常是背景里的低分框，于是"中位 15.7 px / p90 352 px"——22 倍的差距
    说明中位数是从另一批图上来的，这种报告没法回答"能不能替代点选"。
    运行时用的是 ``DetectorConfig.yolo_conf``（0.25），评估就必须落在同一个门槛上。

    为什么要分桶：同一个模型在 24 px 和 159 px 的球上是两件事。混在一起平均，
    小的必然全错、大的几乎全对，平均值看起来"还行"，而实际可用的只有一半数据。
    """
    import numpy as np

    from sphere_tracker.config import DetectorConfig

    cfg = DetectorConfig()
    rt = float(runtime_conf if runtime_conf is not None else cfg.yolo_conf)
    # 表的行 = 常驻网格 ∪ 运行时阈值（配置里改成别的值也照样有对应一行）
    thresholds = sorted(set(CONF_GRID) | {rt})
    labels_dir = labels_dir_for(stats.val_images[0].parent)
    paths = [str(p) for p in stats.val_images]
    # 一次推理，最低阈值；更高的阈值在下面用 numpy 再筛（重复推理只是多花时间）
    results = model.predict(paths, conf=EVAL_CONF, imgsz=imgsz, device=device, verbose=False)

    deltas: dict[float, list[float]] = {t: [] for t in thresholds}
    radii: dict[float, list[float]] = {t: [] for t in thresholds}
    refined: dict[float, int] = {t: 0 for t in thresholds}
    matched: dict[float, int] = {t: 0 for t in thresholds}
    detected: dict[float, int] = {t: 0 for t in thresholds}
    refined_cache: dict[tuple, float | None] = {}
    records: list[dict] = []      # 每张带框图一条：用于任意阈值下再分桶
    top1_confs: list[float] = []  # 每图最高分：回答"置信度尺度对不对得上运行时阈值"
    with_boxes = 0
    negatives = 0

    for idx, res in enumerate(results):
        img_path = Path(getattr(res, "path", ""))
        labels = read_label_file(labels_dir / f"{img_path.stem}.txt")
        if not labels:
            # 负样本图没有圆心可比；它只在"分母含全部 val 图"的那个命中率里起作用
            negatives += 1
            continue
        with_boxes += 1
        h_px, w_px = (res.orig_shape[0], res.orig_shape[1]) if res.orig_shape else (
            res.orig_img.shape[:2] if getattr(res, "orig_img", None) is not None else (0, 0))
        if not h_px or not w_px:
            continue
        # 标签：归一化 (cx,cy,w,h) → 像素框 + 真值半径（w/h 的平均一半）
        gt_boxes = [((cx - w / 2) * w_px, (cy - h / 2) * h_px,
                     (cx + w / 2) * w_px, (cy + h / 2) * h_px) for cx, cy, w, h in labels]
        gt_centers = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in gt_boxes]

        boxes = getattr(res, "boxes", None)
        xyxy = boxes.xyxy.cpu().numpy() if boxes is not None and len(boxes) else None
        confs = boxes.conf.cpu().numpy() if boxes is not None and len(boxes) else None
        frame = getattr(res, "orig_img", None)
        if confs is not None and confs.size:
            top1_confs.append(float(np.max(confs)))

        rt_side = None                       # 仅用于调试，不参与分桶
        hit: set[float] = set()
        for t in thresholds:
            if xyxy is None or confs is None:
                deltas[t].append(float("inf"))
                continue
            keep = confs >= t
            if not keep.any():
                deltas[t].append(float("inf"))   # 漏检记 inf：命中率自然不过门
                continue
            best = int(np.argmax(np.where(keep, confs, -1.0)))
            detected[t] += 1
            pred_box = tuple(float(v) for v in xyxy[best])
            pcx = (pred_box[0] + pred_box[2]) / 2.0
            pcy = (pred_box[1] + pred_box[3]) / 2.0
            # 用 IoU 匹配标签（不用"最近圆心"：那会拿预测去挑对自己最有利的真值）
            gi = int(np.argmax([_iou(pred_box, b) for b in gt_boxes]))
            gcx, gcy = gt_centers[gi]
            deltas[t].append(math.hypot(pcx - gcx, pcy - gcy))
            matched[t] += 1
            hit.add(t)

            gt_box = gt_boxes[gi]
            gt_r = 0.25 * ((gt_box[2] - gt_box[0]) + (gt_box[3] - gt_box[1]))
            key = (idx, round(pred_box[0], 1), round(pred_box[1], 1),
                   round(pred_box[2], 1), round(pred_box[3], 1))
            if key not in refined_cache:
                refined_cache[key] = (refine_radius_for_report(frame, pred_box, cfg)
                                      if frame is not None else None)
            r_pred = refined_cache[key]
            if r_pred is None:
                r_pred = 0.5 * max(pred_box[2] - pred_box[0], pred_box[3] - pred_box[1])
            else:
                refined[t] += 1
            if gt_r > 1e-6:
                radii[t].append(abs(r_pred - gt_r) / gt_r)

            if t == rt:
                rt_side = max(gt_box[2] - gt_box[0], gt_box[3] - gt_box[1])

        # 分桶：每张带框的图都必须进桶（漏检也要进，否则"小目标全错"会从表里消失）。
        # 一张图多个标签时按**最大**那个定桶：一帧里最显眼的球决定这张图好不好认。
        side = max(max(b[2] - b[0], b[3] - b[1]) for b in gt_boxes)
        records.append({"bucket": size_bucket(side),
                        "deltas": {t: (deltas[t][-1] if t in hit else float("inf"))
                                   for t in thresholds}})

    n_all = len(stats.val_images)
    by_conf = {f"{t:.2f}": _threshold_row(deltas[t], with_boxes, n_all, radii[t],
                                          refined[t], matched[t])
               for t in thresholds}
    by_size = _bucket_table(records, rt)
    # 运行时阈值一行框都没有时，分桶表全空 —— 再给一张低阈值的表，
    # 否则"小目标必然检不到"这个结论没有数字支撑。
    by_size_low = _bucket_table(records, float(EVAL_CONF)) if detected[rt] == 0 else None

    headline = dict(by_conf[f"{rt:.2f}"])
    headline.update({
        "runtime_conf": rt,
        "eval_conf": EVAL_CONF,
        "n_val_images": n_all,
        "n_val_with_boxes": with_boxes,
        "n_negative": negatives,
        "n_detected": detected[rt],
        # 这一位数解释"为什么运行时阈值下 0 检出"：分数本身压在 0.0x 时，
        # 0.25 的门不是"滤掉误检"，是"滤掉全部"。
        "median_top1_conf": float(np.median(top1_confs)) if top1_confs else None,
    })
    return {"headline": headline, "by_conf": by_conf, "by_size": by_size,
            "by_size_low": by_size_low, "by_size_low_conf": EVAL_CONF}


def summary_lines(center: dict) -> list[str]:
    """屏幕总结：运行时阈值那一行 + 阈值表 + 分桶表 + 一句结论。

    "能不能当种子"只由运行时阈值那一行的 30 px 命中率回答；中位误差低但命中率低
    （一半图没有检测）就是不能用，别用中位去安慰自己。
    """
    head = center.get("headline", center)
    rt = head.get("runtime_conf", 0.25)
    lines: list[str] = []
    if not head.get("n_val_with_boxes"):
        return ["[结果] 圆心误差：val 的标签里一个框都没有（导出时没写标签？），量不出"]
    if not head.get("n_detected"):
        lines.append(f"[结果] conf={rt:.2f}（运行时默认）下这些带框的 val 图**一张都没检出** —— "
                     "这个权重不能当种子，问题在检出率而不是圆心精度")
    elif head.get("median_px") is not None:
        lines.append(
            f"[结果] conf={rt:.2f}（运行时默认）：圆心误差 中位 {head['median_px']:.1f} px / "
            f"p90 {head['p90_px']:.1f} px；{CENTER_HIT_PX[0]:.0f} px 内命中 "
            f"{head['recall_30px'] * 100:.0f}%（分母 {head['n_val_with_boxes']} 张带框 val 图，"
            f"漏检算失败；按全部 {head['n_val_images']} 张算 "
            f"{head['recall_30px_all'] * 100:.0f}%）"
            + (f"；半径误差 {head['radius_median_rel_error'] * 100:.1f}%"
               if head.get("radius_median_rel_error") is not None else "；半径误差：无数据"))

    if center.get("by_conf"):
        lines.append(f"[阈值] {'conf':>6} {'检出图数':>8} {'中位px':>8} {'p90px':>8} "
                     f"{'30px命中':>9} {'半径误差':>9}")
        for conf_text, row in center["by_conf"].items():
            med = f"{row['median_px']:.1f}" if row.get("median_px") is not None else "—"
            p90 = f"{row['p90_px']:.1f}" if row.get("p90_px") is not None else "—"
            rad = (f"{row['radius_median_rel_error'] * 100:.1f}%"
                   if row.get("radius_median_rel_error") is not None else "—")
            mark = " ←运行时" if abs(float(conf_text) - rt) < 1e-9 else ""
            lines.append(f"       {conf_text:>6} {row['n_matched']:>8} {med:>8} {p90:>8} "
                         f"{row['recall_30px'] * 100:>8.0f}% {rad:>9}{mark}")

    if center.get("by_size"):
        lines.append(f"[分桶] {'框长边':>10} {'带框张数':>8} {'有检出':>7} {'中位px':>8} {'30px命中':>9}")
        for name, row in center["by_size"].items():
            med = f"{row['median_px']:.1f}" if row.get("median_px") is not None else "—"
            lines.append(f"       {name:>10} {row['n_labeled']:>8} {row['n_detected']:>7} "
                         f"{med:>8} {row['recall_30px'] * 100:>8.0f}%")
        # 运行时阈值 0 检出时这张表三行全 0，逐桶提醒只会重复三遍同一件事，
        # 真正该看的是下面那张降阈值的诊断表。
        if head.get("n_detected"):
            for name, row in center["by_size"].items():
                if row["n_labeled"] >= 5 and row["recall_30px"] < 0.3:
                    lines.append(f"       [注意] {name} 这一桶只有 {row['recall_30px'] * 100:.0f}% 命中："
                                 "这类样本当不了种子（别用一个平均值把它盖掉）")

    if center.get("by_size_low") and not head.get("n_detected"):
        # 运行时阈值 0 检出时，上面那张分桶表全是 0，看不出"是全都不会"还是"小的不会"。
        # 这张表把阈值降到评估下限，只用于诊断。
        conf_low = center.get("by_size_low_conf", EVAL_CONF)
        lines.append(f"[分桶@{conf_low:.2f}] 运行时阈值 0 检出，降阈值只为诊断"
                     f"（{'框长边':>10} 带框/有检出/30px命中）")
        for name, row in center["by_size_low"].items():
            lines.append(f"       {name:>10} {row['n_labeled']:>5}/{row['n_detected']:<5} "
                         f"{row['recall_30px'] * 100:>8.0f}%"
                         + (f"  中位 {row['median_px']:.1f} px" if row.get("median_px") is not None else ""))

    recall30 = head.get("recall_30px") or 0.0
    if head.get("median_px") is None:
        lines.append("       结论：运行时阈值下没有可用的框，先看上面的 mAP 和分桶表")
        # 分两种原因：阈值不匹配（降阈值立刻好用）vs 真的没学会。
        # 混成一句"不能用"会让人去做错的事（去加数据，而其实只要调门）。
        best_low = None
        for conf_text, row in (center.get("by_conf") or {}).items():
            if float(conf_text) >= rt or not row.get("n_matched"):
                continue
            if best_low is None or row["recall_30px"] > best_low[1]["recall_30px"]:
                best_low = (conf_text, row)
        if best_low is not None and best_low[1]["recall_30px"] >= 0.5:
            conf_text, row = best_low
            med_conf = head.get("median_top1_conf")
            lead = (f"本模型 val 上 top-1 分数中位只有 {med_conf:.2f}，"
                    if med_conf is not None else "")
            lines.append(f"       [阈值] {lead}而运行时门槛是 {rt:.2f}，等于把所有框都滤掉："
                         f"降到 {conf_text} 时 30 px 命中 "
                         f"{row['recall_30px'] * 100:.0f}%、中位 {row['median_px']:.1f} px。"
                         "这是**阈值不匹配**，不是模型没学会 —— 用 --conf 调门槛，"
                         "或继续训到分数抬上来")
        elif best_low is not None:
            conf_text, row = best_low
            lines.append(f"       [阈值] 降到 {conf_text} 也只有 "
                         f"{row['recall_30px'] * 100:.0f}% 命中：不是门槛问题，是模型没学会"
                         "（看分桶和 mAP，先补数据/加 epoch 或提分辨率）")
    elif recall30 < 0.8:
        lines.append(f"       结论：[注意] 只有 {recall30 * 100:.0f}% 的 val 图在 "
                     f"{CENTER_HIT_PX[0]:.0f} px 内 —— 当种子用会丢掉一批帧，"
                     "先看分桶表判断是「小目标检不到」还是「框整体偏」")
    else:
        lines.append("       结论：可以当种子用（一半以上帧的圆心误差在 30 px 内）")
    return lines


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="用已点选的数据训练单类球检测器（YOLO11），并验收圆心/半径误差")
    p.add_argument("--data", default="outputs/ball_dataset", help="数据集目录（含 dataset.yaml）")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="cpu",
                   help="cpu（默认）或显卡编号 0；默认 CPU 是因为很多人没有 GPU，"
                        "继承配置里的 \"0\" 会直接报 CUDA 错、看起来像代码坏了")
    p.add_argument("--base", default=DEFAULT_BASE, help="基座权重（仓库自带，不自动下载）")
    p.add_argument("--out", default=DEFAULT_OUT, help="最好的权重拷到这里")
    p.add_argument("--dry-run", action="store_true",
                   help="只体检数据集并打印训练计划，不训练（数据不够时也能跑）")
    return p.parse_args(argv)


def _project_path(text: str) -> Path:
    p = Path(text)
    return p if p.is_absolute() else (ROOT / p)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        stats = collect_stats(_project_path(args.data))
    except DatasetError as exc:
        print("[失败] 数据集不可用：")
        for line in str(exc).splitlines():
            print(f"       {line}" if not line.startswith("       ") else line)
        print("       退出码 2 —— 没有可训练的数据，而不是「训完了但效果差」。")
        return 2

    for line in format_stats(stats):
        print(line)

    problems = preflight_problems(stats.session_count, stats.train_boxes, ultralytics_available())
    for problem in problems:
        print(f"[不达标] {problem}")

    print(f"[计划] epochs={args.epochs} imgsz={args.imgsz} batch={args.batch} "
          f"device={args.device} base={args.base} out={args.out}")
    print(f"       增强：hsv {AUGMENT['hsv_h']}/{AUGMENT['hsv_s']}/{AUGMENT['hsv_v']}，"
          f"旋转 ±{AUGMENT['degrees']:g}°，平移 {AUGMENT['translate'] * 100:g}%，"
          f"缩放 {AUGMENT['scale']:g}，水平翻转 {AUGMENT['fliplr']:g}；"
          "patience=15 workers=2")

    if args.dry_run:
        if problems:
            print(f"[结论] 数据还不够，真训练会被拒绝（退出码 2）。"
                  f"--dry-run 只做体检，按 0 退出：差 {max(0, MIN_POSITIVE_BOXES - stats.train_boxes)} 个框、"
                  f"{max(0, MIN_SESSIONS - stats.session_count)} 个会话。")
        else:
            print("[结论] 体检通过，可以去掉 --dry-run 开训")
        return 0

    if problems:
        print("[失败] 前置检查未通过，不训练（退出码 2）。"
              "先补数据，或用 --dry-run 反复看还差多少。")
        return 2

    base_path = _project_path(args.base)
    if not base_path.is_file():
        print(f"[失败] 基座权重不存在：{base_path}\n"
              "       本脚本不下载任何权重（现场常常没网络）。把官方权重放到 models/ 下，"
              "或用 --base 指到本地已有的 .pt")
        return 2

    try:
        from ultralytics import YOLO
    except ImportError as exc:  # 与前置检查重复兜底：这里只在"检查之后被卸载"时才会到
        print(f"[失败] 导入 ultralytics 失败：{exc}\n    pip install ultralytics")
        return 2

    started = time.time()
    run_name = time.strftime("%Y%m%d_%H%M%S")
    project = ROOT / "outputs" / "train_ball"
    print(f"[训练] 开始：{run_name}（小数据开着增强；CPU 上会慢，可先 --epochs 3 试流程）")
    model = YOLO(str(base_path))
    model.train(
        data=str(stats.yaml_path), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, patience=15, workers=2, seed=0, cache=False,
        project=str(project), name=run_name, exist_ok=True, **AUGMENT,
    )
    elapsed = time.time() - started

    save_dir = Path(getattr(model.trainer, "save_dir", project / run_name))
    best = save_dir / "weights" / "best.pt"
    if not best.is_file():
        print(f"[失败] 训练结束但找不到最好的权重：{best}")
        return 2
    out_path = _project_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, out_path)
    print(f"[权重] 已拷贝：{out_path}（源 {best}）")

    # ---- 验收：先 mAP，再圆心/半径 ----
    print("[验收] 在 val 上跑 mAP 与圆心误差（这一步比训练慢不了多少，但只有它能回答"
          "「能不能当种子用」）")
    metrics = model.val(data=str(stats.yaml_path), imgsz=args.imgsz, device=args.device,
                        project=str(project / "val"), name=run_name, exist_ok=True,
                        verbose=False)
    center = evaluate(model, stats, imgsz=args.imgsz, device=args.device)
    for line in summary_lines(center):
        print(line)

    report = {
        "schema": "ball-detector-report",
        "version": 1,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": {
            "path": str(stats.root),
            "train_images": len(stats.train_images),
            "val_images": len(stats.val_images),
            "train_boxes": stats.train_boxes,
            "val_boxes": stats.val_boxes,
            "train_sessions": stats.train_sessions,
            "val_sessions": stats.val_sessions,
        },
        "train": {
            "epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch,
            "device": args.device, "base": str(base_path), "out": str(out_path),
            "run_dir": str(save_dir), "seconds": round(elapsed, 1),
            "augment": AUGMENT,
        },
        "metrics": {
            "map50": float(getattr(metrics.box, "map50", float("nan"))),
            "map50_95": float(getattr(metrics.box, "map", float("nan"))),
            "per_class": _per_class_map(metrics),
        },
        "center": center,
        "notes": ("圆心误差按原始分辨率像素计；漏检记入 n_missed 且不进中位；"
                  "半径由 refine_circle_in_box 在放宽的搜索窗内细化，失败时退回框长边的一半"),
    }
    report_path = out_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[报告] {report_path}（mAP50={report['metrics']['map50']:.3f}，"
          f"mAP50-95={report['metrics']['map50_95']:.3f}，耗时 {elapsed:.0f} s）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
