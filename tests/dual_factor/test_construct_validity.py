"""Tests for construct_validity functions in construct_validity.py."""
import numpy as np
import pandas as pd
import pytest

from dual_factor.analysis.construct_validity import (
    proxy_validity_report,
    amplitude_normalization,
    construct_validity_summary,
)

INDICATORS = ['FD', 'PSV', 'SA', 'FN', 'SE', 'GTE', 'SV', 'SDR']


@pytest.fixture
def synthetic_df():
    """Create a synthetic DataFrame with subject, kss, valid, and
    8 indicator columns."""
    np.random.seed(42)
    n = 100
    data = {
        'subject': np.repeat(['S1', 'S2', 'S3', 'S4', 'S5'], 20),
        'kss': np.random.choice([3, 5, 7, 9], n),
        'valid': [True] * n,
    }
    for ind in INDICATORS:
        data[ind] = np.abs(np.random.randn(n) * 10 + 50)
    return pd.DataFrame(data)


def test_proxy_validity_report(synthetic_df):
    """proxy_validity_report should return a DataFrame with columns:
    indicator, proxy_category, risk_level, amplitude_sensitive."""
    result = proxy_validity_report(synthetic_df, INDICATORS)
    assert isinstance(result, pd.DataFrame)
    for col in ['indicator', 'proxy_category', 'risk_level',
                'amplitude_sensitive']:
        assert col in result.columns


def test_amplitude_normalization(synthetic_df):
    """amplitude_normalization should return a DataFrame with _norm columns."""
    result = amplitude_normalization(synthetic_df, 'subject', INDICATORS)
    assert isinstance(result, pd.DataFrame)
    for ind in INDICATORS:
        assert f'{ind}_norm' in result.columns


def test_construct_validity_summary(synthetic_df):
    """construct_validity_summary should return a markdown string containing
    '代理效度' and '方向性检验'."""
    result = construct_validity_summary(synthetic_df, INDICATORS)
    assert isinstance(result, str)
    assert '代理效度' in result
    assert '方向性检验' in result
