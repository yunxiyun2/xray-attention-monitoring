"""Tests for run_efa_compare and run_efa_fixed in efa.py."""
import numpy as np
import pandas as pd
import pytest

from dual_factor.analysis.efa import run_efa_compare, run_efa_fixed


@pytest.fixture
def synthetic_data():
    """Create synthetic 2-factor data: 200 rows x 8 columns.
    First 4 columns load on factor 1, last 4 on factor 2."""
    np.random.seed(42)
    n = 200
    factor1 = np.random.randn(n) * 1.0
    factor2 = np.random.randn(n) * 1.0

    data = {}
    cols = [f'V{i+1}' for i in range(8)]
    for i, col in enumerate(cols):
        if i < 4:
            data[col] = 0.8 * factor1 + 0.3 * np.random.randn(n)
        else:
            data[col] = 0.8 * factor2 + 0.3 * np.random.randn(n)
    return pd.DataFrame(data)


def test_run_efa_fixed_returns_result_with_n_factors(synthetic_data):
    """run_efa_fixed(data, n_factors=2) should return an EFAResult with
    n_factors == 2."""
    result = run_efa_fixed(synthetic_data, n_factors=2)
    assert result is not None
    assert result.n_factors == 2


def test_run_efa_compare_returns_dict_with_expected_keys(synthetic_data):
    """run_efa_compare(data, [2, 3]) should return a dict with
    'parallel_analysis_n_factors' and 'comparisons' keys, where
    'comparisons' contains '2_factor' and '3_factor'."""
    result = run_efa_compare(synthetic_data, [2, 3])
    assert isinstance(result, dict)
    assert 'parallel_analysis_n_factors' in result
    assert 'comparisons' in result
    assert '2_factor' in result['comparisons']
    assert '3_factor' in result['comparisons']


def test_2_factor_cross_loading_low(synthetic_data):
    """For clean 2-factor data, the 2-factor comparison's
    cross_loading_max should be < 0.3."""
    result = run_efa_compare(synthetic_data, [2, 3])
    comp_2f = result['comparisons']['2_factor']
    assert comp_2f['cross_loading_max'] < 0.3
