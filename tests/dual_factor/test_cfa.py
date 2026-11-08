"""CFA 模块测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dual_factor.analysis.cfa import (
    CFAResult,
    DRIVE_DECAY,
    SPATIAL_DISORG,
    RETAINED,
    run_cfa,
    per_subject_cfa,
    _build_model_spec,
    CFI_THRESHOLD,
    RMSEA_THRESHOLD,
    SRMR_THRESHOLD,
)


# ============ 合成数据 ============

def _make_synthetic_data(n_obs: int = 500, seed: int = 42) -> pd.DataFrame:
    """生成符合两因子结构的合成数据。

    F1 -> FD, PSV, SA, FN（驱动衰减）
    F2 -> SE, GTE（空间失序）
    因子相关 ~0.1
    """
    rng = np.random.default_rng(seed)
    f1 = rng.normal(0, 1, n_obs)
    f2 = rng.normal(0, 1, n_obs) * 0.3 + f1 * 0.1  # 近乎独立

    data = pd.DataFrame({
        "FD_z": 0.7 * f1 + rng.normal(0, 0.5, n_obs),
        "PSV_z": 0.8 * f1 + rng.normal(0, 0.4, n_obs),
        "SA_z": 0.85 * f1 + rng.normal(0, 0.4, n_obs),
        "FN_z": 0.6 * f1 + rng.normal(0, 0.6, n_obs),
        "SE_z": 0.9 * f2 + rng.normal(0, 0.4, n_obs),
        "GTE_z": 0.95 * f2 + rng.normal(0, 0.3, n_obs),
    })
    return data


def _make_synthetic_per_subject(n_subjects: int = 15, n_windows: int = 80) -> pd.DataFrame:
    """生成多被试合成数据。"""
    frames = []
    for sid in range(1, n_subjects + 1):
        sub_data = _make_synthetic_data(n_windows, seed=sid * 100)
        sub_data["subject"] = sid
        frames.append(sub_data)
    return pd.concat(frames, ignore_index=True)


# ============ 模型语法测试 ============

class TestBuildModelSpec:
    def test_default_indicators(self):
        cols = [f"{c}_z" for c in RETAINED]
        spec = _build_model_spec(cols)
        assert "DriveDecay =~" in spec
        assert "SpatialDisorg =~" in spec
        assert "DriveDecay ~~ SpatialDisorg" in spec
        assert "FD_z" in spec
        assert "GTE_z" in spec

    def test_custom_indicators(self):
        cols = ["a", "b", "c", "d", "e", "f"]
        spec = _build_model_spec(cols, indicator_groups={"DriveDecay": cols[:4], "SpatialDisorg": cols[4:]})
        assert "DriveDecay =~ a + b + c + d" in spec
        assert "SpatialDisorg =~ e + f" in spec


# ============ 常量测试 ============

class TestConstants:
    def test_retained_indicators(self):
        assert RETAINED == ["FD", "PSV", "SA", "FN", "SE", "GTE"]
        assert len(RETAINED) == 6
        assert DRIVE_DECAY == ["FD", "PSV", "SA", "FN"]
        assert SPATIAL_DISORG == ["SE", "GTE"]

    def test_thresholds(self):
        assert CFI_THRESHOLD == 0.90
        assert RMSEA_THRESHOLD == 0.08
        assert SRMR_THRESHOLD == 0.08


# ============ CFA 拟合测试 ============

class TestRunCFA:
    def test_returns_cfa_result(self):
        data = _make_synthetic_data(500)
        result = run_cfa(data)
        assert isinstance(result, CFAResult)
        assert result.n_obs == 500

    def test_fit_indices_present(self):
        data = _make_synthetic_data(500)
        result = run_cfa(data)
        assert not np.isnan(result.cfi)
        assert not np.isnan(result.rmsea)
        assert not np.isnan(result.srmr)

    def test_good_fit_passes(self):
        """合成数据拟合良好时应通过判据。"""
        data = _make_synthetic_data(1000)
        result = run_cfa(data)
        assert result.cfi > 0.90, f"CFI={result.cfi}"
        assert result.srmr < 0.08, f"SRMR={result.srmr}"

    def test_main_loadings_extracted(self):
        data = _make_synthetic_data(500)
        result = run_cfa(data)
        assert len(result.main_loadings) == 6
        # 驱动衰减因子指标应该有正载荷
        for ind in ["FD_z", "PSV_z", "SA_z", "FN_z"]:
            assert ind in result.main_loadings

    def test_factor_corr_present(self):
        data = _make_synthetic_data(500)
        result = run_cfa(data)
        assert not np.isnan(result.factor_corr)

    def test_missing_columns_raises(self):
        data = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="缺少列"):
            run_cfa(data)

    def test_wrong_indicator_count_raises(self):
        data = pd.DataFrame({"a_z": range(100), "b_z": range(100)})
        with pytest.raises(ValueError, match="需要恰好6项"):
            run_cfa(data, ["a_z", "b_z"])

    def test_insufficient_obs_raises(self):
        data = _make_synthetic_data(10)
        with pytest.raises(ValueError, match="有效观测数过少"):
            run_cfa(data)

    def test_all_pass_flag(self):
        data = _make_synthetic_data(1000)
        result = run_cfa(data)
        # 好的合成数据应该 all_pass 为 True
        assert result.all_pass == (result.cfi_pass and result.rmsea_pass and result.srmr_pass)


# ============ 逐被试 CFA 测试 ============

class TestPerSubjectCFA:
    def test_returns_dataframe(self):
        df = _make_synthetic_per_subject(n_subjects=5, n_windows=80)
        result = per_subject_cfa(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5

    def test_columns_present(self):
        df = _make_synthetic_per_subject(n_subjects=3, n_windows=80)
        result = per_subject_cfa(df)
        expected_cols = {"subject", "n_windows", "status", "cfi", "rmsea", "srmr", "all_pass"}
        assert expected_cols.issubset(set(result.columns))

    def test_insufficient_data_handling(self):
        """被试数据不足时应标记 insufficient_data。"""
        frames = []
        for sid in range(1, 4):
            sub = _make_synthetic_data(80, seed=sid)
            sub["subject"] = sid
            frames.append(sub)
        # 添加一个数据不足的被试
        small = _make_synthetic_data(5, seed=99)
        small["subject"] = 99
        frames.append(small)
        df = pd.concat(frames, ignore_index=True)
        result = per_subject_cfa(df)
        row = result[result["subject"] == 99].iloc[0]
        assert row["status"] == "insufficient_data"
        assert row["all_pass"] == False

    def test_subject_ids_preserved(self):
        df = _make_synthetic_per_subject(n_subjects=7, n_windows=60)
        result = per_subject_cfa(df)
        assert set(result["subject"]) == set(range(1, 8))
