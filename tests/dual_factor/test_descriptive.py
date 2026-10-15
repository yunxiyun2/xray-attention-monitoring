"""描述性统计与方向性检验测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dual_factor.analysis.descriptive import (
    descriptive_stats, directionality_test, HYPOTHESIS_DIRECTIONS, INDICATORS,
)


def test_descriptive_stats_columns():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "subject": ["1"] * 20 + ["2"] * 20,
        "kss": [3] * 20 + [7] * 20,
        **{ind: rng.standard_normal(40) for ind in INDICATORS},
    })
    stats = descriptive_stats(df, INDICATORS)
    assert set(stats.columns) >= {"indicator", "mean_alert", "mean_drowsy", "std_alert", "std_drowsy"}
    assert len(stats) == 8


def test_directionality_test_returns_match():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "subject": ["1"] * 20 + ["2"] * 20,
        "kss": [3] * 20 + [7] * 20,
        "FD": np.concatenate([rng.standard_normal(20), rng.standard_normal(20) + 1.0]),
        "PSV": np.concatenate([rng.standard_normal(20) + 1.0, rng.standard_normal(20)]),
    })
    out = directionality_test(df, ["FD", "PSV"], alert_max=4, drowsy_min=6)
    assert set(out.columns) >= {"indicator", "diff", "p_value", "direction", "expected", "match"}
    fd_row = out[out["indicator"] == "FD"].iloc[0]
    assert fd_row["direction"] == "increase"
    assert fd_row["expected"] == "increase"
    assert bool(fd_row["match"]) is True
