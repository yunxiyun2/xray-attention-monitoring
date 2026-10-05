from __future__ import annotations

import numpy as np
import pytest
from xray_attention.evaluation.stats import (
    bootstrap_auc_ci,
    wilcoxon_vs_baseline,
    rank_biserial_r,
    brier_score,
    wilcoxon_paired,
)


def test_bootstrap_auc_ci_bounds():
    y = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    scores = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.7, 0.6, 0.4])
    lo, hi, mean = bootstrap_auc_ci(y, scores, B=200, seed=42)
    assert 0.0 <= lo <= mean <= hi <= 1.0


def test_bootstrap_auc_perfect():
    y = np.array([0, 1])
    scores = np.array([0.0, 1.0])
    lo, hi, mean = bootstrap_auc_ci(y, scores, B=100, seed=42)
    assert abs(mean - 1.0) < 1e-6


def test_wilcoxon_vs_baseline_returns_p():
    aucs = np.array([0.7, 0.75, 0.8, 0.65, 0.9])
    p = wilcoxon_vs_baseline(aucs, baseline=0.5)
    assert 0.0 <= p <= 1.0


def test_wilcoxon_paired_returns_p():
    a = np.array([0.7, 0.8, 0.75, 0.85])
    b = np.array([0.6, 0.7, 0.65, 0.7])
    p = wilcoxon_paired(a, b)
    assert 0.0 <= p <= 1.0


def test_rank_biserial_range():
    r = rank_biserial_r(np.array([0.6, 0.7, 0.8]), baseline=0.5)
    assert -1.0 <= r <= 1.0
    # 全部高于 baseline → r > 0
    assert r > 0


def test_rank_biserial_all_below():
    r = rank_biserial_r(np.array([0.3, 0.2, 0.1]), baseline=0.5)
    assert r < 0


def test_brier_score_zero_perfect():
    y = np.array([0, 1])
    p = np.array([0.0, 1.0])
    assert brier_score(y, p) == 0.0


def test_brier_score_worst():
    y = np.array([0, 1])
    p = np.array([1.0, 0.0])
    assert brier_score(y, p) == 1.0


def test_brier_score_partial():
    y = np.array([0, 1])
    p = np.array([0.5, 0.5])
    # (0.5)^2 + (0.5)^2 / 2 = 0.25
    assert abs(brier_score(y, p) - 0.25) < 1e-6
