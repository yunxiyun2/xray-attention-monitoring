"""窗口化与被试内标准化测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dual_factor.features.windowing import (
    slice_windows, build_indicator_matrix, within_subject_zscore,
)


def test_slice_windows_count():
    fs = 512
    n = 512 * 100  # 100 秒
    h = np.random.randn(n)
    v = np.random.randn(n)
    wins = slice_windows(h, v, fs, window_sec=30, stride_sec=10)
    # 100 秒 / 30 秒窗口 / 10 秒步长 → (100-30)/10 + 1 = 8 个窗口
    assert len(wins) == 8
    assert wins[0].start_idx == 0
    assert wins[1].start_idx == 10 * fs


def test_build_indicator_matrix_columns():
    fs = 512
    n = 512 * 100
    rng = np.random.default_rng(42)
    h = rng.standard_normal(n)
    v = rng.standard_normal(n)
    df = build_indicator_matrix(
        subject="1", test="1", kss=3,
        eog_h=h, eog_v=v, fs=fs,
        window_sec=30, stride_sec=10, bins=5,
    )
    assert len(df) == 8
    expected_cols = {"subject", "test", "kss", "window_idx",
                     "FD", "PSV", "SA", "FN", "SE", "GTE", "SV", "SDR"}
    assert expected_cols <= set(df.columns)


def test_within_subject_zscore_centers_per_subject():
    df = pd.DataFrame({
        "subject": ["1"] * 5 + ["2"] * 5,
        "FD": [1, 2, 3, 4, 5, 10, 20, 30, 40, 50],
    })
    out = within_subject_zscore(df, ["FD"])
    g1 = out[out["subject"] == "1"]["FD_z"]
    g2 = out[out["subject"] == "2"]["FD_z"]
    assert abs(g1.mean()) < 1e-6
    assert abs(g2.mean()) < 1e-6
