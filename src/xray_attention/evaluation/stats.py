"""统计检验工具。

对应 spec v2.0 §8 评估协议：
- bootstrap 95% CI for AUC
- Wilcoxon signed-rank（vs 0.5 基线、配对比较）
- rank-biserial 效应量
- Brier score（概率校准）
"""
from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon, rankdata
from sklearn.metrics import roc_auc_score


def bootstrap_auc_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    B: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """AUC 的 bootstrap 95% 置信区间。

    Returns
    -------
    (lo, hi, mean) : 95% CI 下界、上界、bootstrap 均值
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    n = len(y_true)
    aucs: list[float] = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], scores[idx]))
    if not aucs:
        return 0.0, 1.0, 0.5
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi), float(np.mean(aucs))


def wilcoxon_vs_baseline(aucs: np.ndarray, baseline: float = 0.5) -> float:
    """单边 Wilcoxon signed-rank：检验 AUC 是否显著高于 baseline。"""
    aucs = np.asarray(aucs, dtype=float)
    diffs = aucs - baseline
    # 若所有 diff 相同或全为 0，wilcoxon 会报错
    if np.all(diffs == 0):
        return 1.0
    stat, p = wilcoxon(diffs, alternative="greater")
    return float(p)


def wilcoxon_paired(a: np.ndarray, b: np.ndarray) -> float:
    """配对 Wilcoxon signed-rank：检验 a 是否显著大于 b（单边）。

    用于「融合 AUC 是否显著优于单模态 AUC」。
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diffs = a - b
    if np.all(diffs == 0):
        return 1.0
    stat, p = wilcoxon(diffs, alternative="greater")
    return float(p)


def rank_biserial_r(aucs: np.ndarray, baseline: float = 0.5) -> float:
    """rank-biserial 效应量 r（与 Wilcoxon 配对）。

    r = (2 * W+) / (n(n+1)/2) - 1，范围 [-1, 1]。
    """
    aucs = np.asarray(aucs, dtype=float)
    diffs = aucs - baseline
    n = len(diffs)
    if n == 0:
        return 0.0
    ranks = rankdata(np.abs(diffs))
    W_pos = float(np.sum(ranks[diffs > 0]))
    total = n * (n + 1) / 2
    return float((2 * W_pos / total) - 1) if total > 0 else 0.0


def brier_score(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Brier score（均方误差），越低越好。完美=0。"""
    y_true = np.asarray(y_true, dtype=float)
    proba = np.asarray(proba, dtype=float)
    return float(np.mean((proba - y_true) ** 2))
