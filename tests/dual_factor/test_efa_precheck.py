"""EFA 前置检验测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dual_factor.analysis.efa import (
    calculate_kmo, bartlett_sphericity, precheck_efa,
)


def test_calculate_kmo_returns_value():
    rng = np.random.default_rng(42)
    data = pd.DataFrame(rng.standard_normal((200, 8)), columns=list("ABCDEFGH"))
    kmo_overall, kmo_per_var = calculate_kmo(data)
    assert 0 <= kmo_overall <= 1
    assert len(kmo_per_var) == 8


def test_bartlett_p_value_in_range():
    rng = np.random.default_rng(42)
    data = pd.DataFrame(rng.standard_normal((200, 8)), columns=list("ABCDEFGH"))
    chi2, p = bartlett_sphericity(data)
    assert chi2 >= 0
    assert 0 <= p <= 1


def test_precheck_efa_returns_all():
    rng = np.random.default_rng(42)
    data = pd.DataFrame(rng.standard_normal((200, 8)), columns=list("ABCDEFGH"))
    out = precheck_efa(data)
    assert set(out.keys()) >= {"kmo_overall", "kmo_per_variable", "bartlett_chi2", "bartlett_p"}
