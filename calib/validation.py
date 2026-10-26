"""交叉验证：标定结果"换一批噪声/换一批位姿"还站得住吗。

## 为什么需要这个文件

`calib/stereo_calib.py` 的 RMS 是**训练集自评**：`stereoCalibrate` 用同一批角点
既拟合又打分。这个数字对"过拟合"几乎免疫 —— 本项目的合成实验里，
3 对共视数据可以把 3 px 的真实误差报成 0.24 px（见
`MIN_COVISIBLE_PAIRS_FOR_JOINT_ANGLED` 的注释）。用户看到 "RMS 0.2 px" 就以为
可靠，而**没有任何一处代码验证过"换一批图还能复现"**。

交叉验证是这个问题最便宜的解：把图分成 k 折，用 k-1 折标定、剩下 1 折**只用来打分**。
训练 RMS 与测试 RMS 的比值就是"泛化差距"的直接度量：

    ratio = test_rms_median / train_rms_median

## 本模块**不依赖** `stereo_calib`

拟合过程由调用方以 `fit_fn(train_idx, test_idx) -> dict` 注入。这样：

1. 本模块可以被单目标定、chain 外参、甚至完全不同的拟合流程复用；
2. 测试可以注入"假拟合器"（例如恒定输出 train=0.2 / test=0.6）来**钉住判定逻辑本身**，
   而不必等一次真实的慢速标定跑完；
3. 不会形成 `stereo_calib` <-> `validation` 的循环依赖。

## 判定的三个档

`ratio <= 1.5`            -> "泛化良好"
`1.5 < ratio <= 2.5`      -> "轻度过拟合：样本偏少或位姿单一"
`ratio > 2.5`             -> "过拟合：结果不可用于测量"

门限取值的依据：真实标定里角点噪声同一量级时，k 折测试集略大于训练集是正常的
（测试集只用 k-1 折的位姿去解释，公式上有 √(k/(k-1)) 量级的放大，k=4 时约 1.15 倍）。
1.5 给了它一倍余量；2.5 以上就不是"余量"能解释的了，而是**模型在记住噪声**。

## 边界（都是显式失败，不静默）

- `n_items < k`：抛 `ValueError`（并说明至少需要多少帧）。k 折每折至少留 1 个样本，
  拆不出来就是拆不出来 —— 这种情况下"算一个 ratio 出来"必然是无意义的。
- 某一折 `fit_fn` 抛异常：**记录**到 `failed_folds`，不重试、不跳过、
  不把它当成 0；只要 `failed_folds` 非空，`verdict` 一定带 "部分折失败" 前缀，
  哪怕剩下的折恰好都"泛化良好" —— 因为"4 折里炸了 1 折"本身就是结论。
- `k=1`：退化成"全量训练 + 训练集自评"。此时 `test_rms == train_rms`、ratio 恒为 1，
  **结论不构成任何泛化证据**，所以 `degenerate=True` 且 verdict 明确写"退化"。

用法::

    from calib.validation import cross_validate

    def fit(train_idx, test_idx):
        res = calibrate_stereo([objp[i] for i in train_idx], ...)
        test = rms_of(res, test_idx)
        return {"train_rms": res.rms, "test_rms": test}

    rep = cross_validate(fit, n_items=len(pairs), k=4, seed=0)
    print(rep["verdict"], rep["ratio"])
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

# ratio 门限（见模块文档"判定的三个档"）
GOOD_RATIO_MAX = 1.5
WARN_RATIO_MAX = 2.5

FitFn = Callable[[Sequence[int], Sequence[int]], dict[str, Any]]


def _ratio_verdict(ratio: float) -> str:
    """按 ratio 给判定。非有限值单独处理（训练 RMS 为 0 时出现过）。"""
    if not np.isfinite(ratio):
        return "无法评估：训练 RMS 为 0 或非有限，比值无定义"
    if ratio <= GOOD_RATIO_MAX:
        return "泛化良好"
    if ratio <= WARN_RATIO_MAX:
        return "轻度过拟合：样本偏少或位姿单一"
    return "过拟合：结果不可用于测量"


def _median(values: list[float]) -> float:
    """中位数；空列表返回 nan（**不用 0** —— 0 会被后续比值当成真值）。"""
    return float(np.median(values)) if values else float("nan")


def _fold_rms(out: Any) -> tuple[float, float]:
    """从 `fit_fn` 返回值里取 (train_rms, test_rms)，缺项/非有限一律算失败折。

    **不默认为 0**：一个"忘了返回 test_rms"的拟合器如果被当成 0，
    会得到 ratio=0 的"完美泛化"结论 —— 比抛异常危险得多。
    """
    if not isinstance(out, dict):
        raise TypeError(f"fit_fn 必须返回 dict，实际是 {type(out).__name__}")
    missing = [k for k in ("train_rms", "test_rms") if k not in out]
    if missing:
        raise KeyError(f"fit_fn 返回值缺少 {missing}（至少要有 train_rms / test_rms）")
    train = float(out["train_rms"])
    test = float(out["test_rms"])
    if not (np.isfinite(train) and np.isfinite(test)):
        raise ValueError(f"fit_fn 返回非有限的 RMS（train={train!r}, test={test!r}）")
    return train, test


def cross_validate(
    fit_fn: FitFn,
    n_items: int,
    *,
    k: int = 4,
    seed: int = 0,
) -> dict:
    """k 折交叉验证，返回训练/测试 RMS 与泛化判定。

    参数
    ----
    fit_fn
        `fit_fn(train_idx, test_idx) -> {"train_rms": float, "test_rms": float, ...}`。
        `train_idx`/`test_idx` 是**整数索引列表**（调用方用它们去取原始角点）。
        其余键会被原样带进 `per_fold`，方便调用方塞诊断信息。
    n_items
        样本总数（本项目 = 图像对数 / 姿态数）。
    k
        折数。`k=1` 退化为训练集自评（见模块文档）。
    seed
        划分用的随机种子。**同一 seed 必须给出同一划分**，否则"上次 ratio 1.4
        这次 2.8"没法比较。

    返回
    ----
    dict，键：
        `train_rms_median` / `test_rms_median` / `ratio` / `verdict` / `per_fold` /
        `failed_folds` / `n_folds_ok` / `degenerate` / `k` / `n_items` / `seed` / `note`
    """
    n_items = int(n_items)
    k = int(k)
    if k < 1:
        raise ValueError(f"k={k} 非法：交叉验证至少需要 1 折")
    if n_items < k:
        raise ValueError(
            f"交叉验证需要至少 {k} 帧，实际只有 {n_items} 帧"
            f"（{k} 折每折至少留 1 帧做测试）。\n"
            f"  下一步：补到 ≥{k} 帧；本项目建议 ≥15 对，"
            f"低于 {8} 对连外参都不该解（见 MIN_COVISIBLE_PAIRS_FOR_JOINT_ANGLED）。"
        )
    if not callable(fit_fn):
        raise TypeError("fit_fn 必须可调用：fit_fn(train_idx, test_idx) -> dict")

    rng = np.random.default_rng(seed)
    order = rng.permutation(n_items)

    if k == 1:
        # 退化：全量训练、全量自评。test_rms 与 train_rms 来自同一批数据。
        folds: list[tuple[list[int], list[int]]] = [
            (order.tolist(), order.tolist())
        ]
    else:
        folds = []
        for chunk in np.array_split(order, k):
            test = chunk.tolist()
            train = [i for i in order.tolist() if i not in set(test)]
            folds.append((train, test))

    per_fold: list[dict] = []
    failed: list[dict] = []
    trains: list[float] = []
    tests: list[float] = []
    unverified = False
    estimator_unclear = False

    for i, (train_idx, test_idx) in enumerate(folds):
        entry: dict[str, Any] = {
            "fold": i,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "train_idx": list(train_idx),
            "test_idx": list(test_idx),
            "train_rms": float("nan"),
            "test_rms": float("nan"),
            "ratio": float("nan"),
            "error": None,
        }
        try:
            out = fit_fn(train_idx, test_idx)
            train, test = _fold_rms(out)
            # ---- 泄漏守卫（2026-10-26 加）----
            # 这个工具最容易的误用是：fit_fn 忽略传入的 train_idx、每折都拿全量训练，
            # 于是 test_rms==train_rms、比值恒 1.0，一路绿灯。**外部看不出来**。
            # 所以要求 fit_fn 把真正用于拟合的索引报出来 `used_train_idx`：
            #   - 报了但对不上 → 该折判失败（不是警告，是数据错）；
            #   - 没报 → 整份结果降级为"未验证"，宁可说不知道，也不发假合格证。
            # ---- 估计量守卫（2026-10-26 加）----
            # 训练用 `stereoCalibrate` 的 rms、测试用"位姿焊死的重投影"，两把尺子
            # 长度不同，比值天生偏乐观（实测偏高约 0.2）。要求 fit_fn 声明两个估计量，
            # 不一致直接判该折失败；未声明则结论降级，别让 reader 以为那是干净的 1.45。
            if isinstance(out, dict):
                est_t = out.get("train_rms_estimator")
                est_s = out.get("test_rms_estimator")
                if est_t and est_s and est_t != est_s:
                    raise ValueError(
                        f"训练/测试不是同一个估计量（{est_t} vs {est_s}）——"
                        f"这种比值只能说明两把尺子不一样长"
                    )
                if not (est_t and est_s):
                    estimator_unclear = True
            if isinstance(out, dict):
                used = out.get("used_train_idx")
                if used is None:
                    unverified = True
                elif set(int(i) for i in used) != set(int(i) for i in train_idx):
                    raise ValueError(
                        f"fit_fn 实际用 {len(set(used))} 帧拟合，与本折 train_idx "
                        f"（{len(set(train_idx))} 帧）不一致 —— 训练集与测试集可能重叠"
                    )
        except Exception as exc:  # noqa: BLE001 — 记录，不吞：见 failed_folds
            entry["error"] = f"{type(exc).__name__}: {exc}"
            failed.append(dict(entry))
            per_fold.append(entry)
            continue
        entry["train_rms"] = train
        entry["test_rms"] = test
        entry["ratio"] = (test / train) if train > 0 else float("inf")
        if isinstance(out, dict):
            for key, val in out.items():
                entry.setdefault(key, val)
        per_fold.append(entry)
        trains.append(train)
        tests.append(test)

    train_med = _median(trains)
    test_med = _median(tests)
    if np.isfinite(train_med) and train_med > 0 and np.isfinite(test_med):
        ratio = float(test_med / train_med)
    else:
        ratio = float("inf") if np.isfinite(test_med) else float("nan")

    base = _ratio_verdict(ratio)
    if estimator_unclear and not unverified:
        base += "（⚠️ 训练/测试未见估计量声明，比值可能混了尺子，仅供参考）"
    if unverified:
        # 把降级信息放进 base，这样"部分折失败"那条分支也不会把结论说满
        base = (f"未验证（fit_fn 未报告 used_train_idx，无法排除训练/测试泄漏）："
                f"按比值看是「{base}」，但不可采信")
    n_ok = len(trains)
    if k == 1:
        degenerate = True
        verdict = ("退化（k=1）：训练集自评，test_rms 恒等于 train_rms，"
                   "**不能**作为泛化证据")
    elif not trains:
        degenerate = False
        verdict = f"部分折失败（{len(failed)}/{k} 折抛异常）：没有任何折成功，无法评估泛化"
    elif failed:
        degenerate = False
        verdict = f"部分折失败（{len(failed)}/{k} 折抛异常）—— 成功折的判定：{base}"
    else:
        degenerate = False
        verdict = base

    return {
        "k": k,
        "n_items": n_items,
        "seed": int(seed),
        "degenerate": degenerate,
        "train_rms_median": train_med,
        "test_rms_median": test_med,
        "ratio": ratio,
        "ratio_per_fold_median": _median([f["ratio"] for f in per_fold
                                          if f["error"] is None]),
        "verdict": verdict,
        "leak_check": "unverified" if unverified else "ok",
        "estimator_declared": not estimator_unclear,
        "n_folds_ok": n_ok,
        "per_fold": per_fold,
        "failed_folds": failed,
        "note": (
            f"ratio = test_rms_median / train_rms_median；"
            f"门限 {GOOD_RATIO_MAX} / {WARN_RATIO_MAX}。"
            f"ratio 只说明「换一批图还能不能复现」，"
            f"不说明绝对精度是否够用（后者要看 RMS 本身与板占画面比例）。"
        ),
    }
