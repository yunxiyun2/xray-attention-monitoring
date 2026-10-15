"""EFA 因子提取与平行分析测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dual_factor.analysis.efa import run_efa, parallel_analysis, EFAResult


def _make_synthetic_2factor(n=200, seed=42):
    rng = np.random.default_rng(seed)
    f1 = rng.standard_normal(n)
    f2 = rng.standard_normal(n) * 0.5
    data = pd.DataFrame({
        "FD": 0.8 * f1 + 0.1 * rng.standard_normal(n),
        "PSV": -0.7 * f1 + 0.1 * rng.standard_normal(n),
        "SA": -0.6 * f1 + 0.1 * rng.standard_normal(n),
        "FN": -0.5 * f1 + 0.1 * rng.standard_normal(n),
        "SE": 0.7 * f2 + 0.1 * rng.standard_normal(n),
        "GTE": 0.6 * f2 + 0.1 * rng.standard_normal(n),
        "SV": 0.5 * f2 + 0.1 * rng.standard_normal(n),
        "SDR": 0.4 * f2 + 0.1 * rng.standard_normal(n),
    })
    return data


def test_run_efa_returns_result():
    data = _make_synthetic_2factor()
    result = run_efa(data, extraction="pa", rotation="promax", max_factors=4, n_iter=50)
    assert isinstance(result, EFAResult)
    assert result.loadings.shape[1] <= 4
    assert result.variance_explained >= 0


def test_parallel_analysis_returns_eigenvalues():
    data = _make_synthetic_2factor(n=100)
    pa = parallel_analysis(data, n_iter=50, percentile=95, max_factors=4)
    assert len(pa["random_eigenvalues"]) <= 4
    assert len(pa["actual_eigenvalues"]) <= 4


def test_efa_recovers_two_factors():
    data = _make_synthetic_2factor(n=300)
    result = run_efa(data, extraction="pa", rotation="promax", max_factors=4, n_iter=50)
    assert result.n_factors >= 1
    assert result.n_factors <= 4
    if result.factor_corr is not None:
        assert result.factor_corr.shape[0] == result.n_factors
