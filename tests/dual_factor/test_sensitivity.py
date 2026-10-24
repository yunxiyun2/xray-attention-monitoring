"""Tests for sensitivity analysis functions in sensitivity.py."""
import numpy as np
import pandas as pd
import pytest

from dual_factor.analysis.sensitivity import (
    aggregate_subject_level,
    subject_level_efa,
    loso_efa,
    nonoverlap_efa,
)

# Base indicator names; functions append _z suffix internally
INDICATORS = ['FD', 'PSV', 'SA', 'FN', 'SE', 'GTE', 'SV', 'SDR']
# Actual column names in the synthetic DataFrame
INDICATOR_Z_COLS = [f'{ind}_z' for ind in INDICATORS]

N_SUBJECTS = 20
N_WINDOWS = 50


@pytest.fixture
def synthetic_df():
    """Create synthetic data: 20 subjects x 50 windows, 8 indicators
    generated from a correlated 2-factor structure."""
    np.random.seed(42)
    n = N_SUBJECTS * N_WINDOWS

    # Two latent factors
    factor1 = np.random.randn(n) * 1.0
    factor2 = np.random.randn(n) * 1.0

    data = {}
    for i, col in enumerate(INDICATOR_Z_COLS):
        if i < 4:
            # First 4 indicators load on factor 1
            data[col] = 0.8 * factor1 + 0.3 * np.random.randn(n)
        else:
            # Last 4 indicators load on factor 2
            data[col] = 0.8 * factor2 + 0.3 * np.random.randn(n)

    data['subject'] = np.repeat(
        [f'S{j+1}' for j in range(N_SUBJECTS)], N_WINDOWS
    )
    data['valid'] = [True] * n
    data['window_idx'] = np.tile(np.arange(N_WINDOWS), N_SUBJECTS)

    return pd.DataFrame(data)


def test_aggregate_subject_level(synthetic_df):
    """aggregate_subject_level should return 1 row per subject with _z columns."""
    result = aggregate_subject_level(synthetic_df, 'subject', INDICATORS)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == N_SUBJECTS
    for ind in INDICATORS:
        assert f'{ind}_z' in result.columns


def test_subject_level_efa(synthetic_df):
    """subject_level_efa should return a dict with 'status' and 'n_factors'."""
    result = subject_level_efa(synthetic_df, 'subject', INDICATORS)
    assert isinstance(result, dict)
    assert 'status' in result
    assert 'n_factors' in result


def test_loso_efa(synthetic_df):
    """loso_efa should return a DataFrame with one row per held-out subject."""
    result = loso_efa(synthetic_df, 'subject', INDICATORS)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == N_SUBJECTS


def test_nonoverlap_efa(synthetic_df):
    """nonoverlap_efa should return a dict."""
    result = nonoverlap_efa(
        synthetic_df, 'subject', INDICATORS,
        window_col='window_idx', window_sec=10, stride_sec=5,
    )
    assert isinstance(result, dict)
