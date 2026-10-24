"""Tests for indicator_distribution_report in descriptive.py."""
import numpy as np
import pandas as pd
import pytest

from dual_factor.analysis.descriptive import indicator_distribution_report

INDICATORS = ['FD', 'PSV', 'SA', 'FN', 'SE', 'GTE', 'SV', 'SDR']


@pytest.fixture
def synthetic_df():
    """Create a synthetic DataFrame with 8 indicator columns and metadata."""
    np.random.seed(42)
    n = 100
    data = {
        'subject': np.repeat(['S1', 'S2', 'S3', 'S4', 'S5'], 20),
        'test': np.tile(['T1', 'T2'], 50),
        'kss': np.random.choice([3, 5, 7, 9], n),
        'valid': np.random.choice([True, False], n, p=[0.9, 0.1]),
    }
    for ind in INDICATORS:
        data[ind] = np.random.randn(n) * 10 + 50
    return pd.DataFrame(data)


def test_report_has_8_rows_with_expected_columns(synthetic_df):
    """Report should have one row per indicator (8 total) and contain
    mean_all, std_all, missing_rate, shapiro_p columns."""
    report = indicator_distribution_report(synthetic_df, INDICATORS)
    assert len(report) == 8
    expected_cols = {'mean_all', 'std_all', 'missing_rate', 'shapiro_p'}
    assert expected_cols.issubset(set(report.columns))


def test_missing_rate_positive_when_nan(synthetic_df):
    """missing_rate should be > 0 when some indicator values are NaN."""
    df = synthetic_df.copy()
    df.loc[:10, 'FD'] = np.nan
    report = indicator_distribution_report(df, INDICATORS)
    fd_row = report[report['indicator'] == 'FD'].iloc[0]
    assert fd_row['missing_rate'] > 0
