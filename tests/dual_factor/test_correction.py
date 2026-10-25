"""SPEC v3.0 路径 B：修剪阶梯单元测试。

覆盖：
  - prune_rule 四条规则的优先级与确定性（SPEC §1.3）
  - floor_violation 下限检查（SPEC §1.4）
  - run_correction_ladder 端到端：合成双因子数据 + 2 个噪声指标 → 恰好剔除噪声并在 rung 2 判定 supported
  - run_correction_ladder 端到端：全噪声数据 → 触底判定 unsupported_structural
  - loso_fixed_2factor 冒烟测试
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dual_factor.analysis.correction import (
    ALL_INDICATORS, DEFAULT_THRESHOLDS, DEFAULT_CORRECTION_CONFIG,
    evaluate_criteria, floor_violation, prune_rule,
    run_correction_ladder, loso_fixed_2factor, summarize_loso_2factor,
    two_factor_solution,
)


TH = dict(DEFAULT_THRESHOLDS)


def make_solution(names, primary, cross, variance=0.70, corr=-0.10):
    """构造 two_factor_solution 形状的合成 dict（不跑 EFA）。"""
    return {
        "n_factors": 2,
        "variable_names": list(names),
        "per_var_max": list(primary),
        "per_var_cross": list(cross),
        "variance_explained": variance,
        "factor_corr": corr,
        "min_primary_loading": float(np.min(primary)),
        "max_cross_loading": float(np.max(cross)),
        "max_primary_loading": float(np.max(primary)),
    }


# ============ prune_rule 单元测试 ============

class TestPruneRule:
    def test_rule1_cross_loading_highest_pruned(self):
        sol = make_solution(
            ["FD", "PSV", "SA", "FN", "SE", "GTE", "SV", "SDR"],
            [0.7, 0.7, 0.7, 0.7, 0.9, 0.9, 0.5, 0.6],
            [0.1, 0.1, 0.1, 0.1, 0.05, 0.1, 0.45, 0.2],  # SV 交叉最高
        )
        target, reason, detail = prune_rule(sol, TH)
        assert target == "SV"
        assert reason == "cross_loading"
        assert "SV" in detail and "0.45" in detail

    def test_rule2_lowest_primary_pruned(self):
        # 交叉全过，但 SDR 主载荷 0.12 < 0.50
        sol = make_solution(
            ["FD", "PSV", "SA", "FN", "SE", "GTE", "SV", "SDR"],
            [0.7, 0.7, 0.7, 0.6, 0.9, 0.9, 0.55, 0.12],
            [0.1] * 8,
        )
        target, reason, _ = prune_rule(sol, TH)
        assert target == "SDR"
        assert reason == "low_loading"

    def test_rule3_variance_driven(self):
        # 交叉过、主载荷全过，仅方差不足
        sol = make_solution(
            ["FD", "PSV", "SA", "FN", "SE", "GTE", "SV", "SDR"],
            [0.8, 0.75, 0.7, 0.55, 0.9, 0.9, 0.6, 0.65],
            [0.1] * 8,
            variance=0.50,
        )
        target, reason, _ = prune_rule(sol, TH)
        assert target == "FN"  # 主载荷最低（0.55）
        assert reason == "variance_driven"

    def test_rule4_factor_collinear_no_prune(self):
        sol = make_solution(
            ["FD", "PSV", "SA", "FN", "SE", "GTE", "SV", "SDR"],
            [0.7] * 8, [0.1] * 8,
            corr=0.85,
        )
        target, reason, _ = prune_rule(sol, TH)
        assert target is None
        assert reason == "factor_collinear"

    def test_priority_cross_over_low_loading(self):
        # 交叉超标 + 主载荷过低同时存在 → 交叉优先
        sol = make_solution(
            ["FD", "PSV", "SA", "FN", "SE", "GTE", "SV", "SDR"],
            [0.7, 0.7, 0.7, 0.7, 0.9, 0.9, 0.5, 0.12],
            [0.1, 0.1, 0.1, 0.1, 0.05, 0.1, 0.45, 0.05],
        )
        target, reason, _ = prune_rule(sol, TH)
        assert reason == "cross_loading"
        assert target == "SV"  # 交叉最高的 SV，而非主载荷最低的 SDR

    def test_all_pass_returns_none(self):
        sol = make_solution(
            ["FD", "PSV", "SA", "FN", "SE", "GTE"],
            [0.7, 0.7, 0.7, 0.6, 0.9, 0.9],
            [0.1] * 6,
            variance=0.70,
        )
        target, reason, _ = prune_rule(sol, TH)
        assert target is None
        assert reason == "pass"


# ============ floor_violation 单元测试 ============

class TestFloorViolation:
    CFG = {"min_indicators": 5, "min_per_factor": 2}

    def test_too_few_indicators(self):
        hit, why = floor_violation(["FD", "PSV", "SA", "SE"], self.CFG)
        assert hit and "4" in why

    def test_hypoactive_below_floor(self):
        hit, why = floor_violation(["FD", "SE", "GTE", "SV", "SDR"], self.CFG)
        assert hit and "呆滞" in why

    def test_diffuse_below_floor(self):
        hit, why = floor_violation(["FD", "PSV", "SA", "FN", "SE"], self.CFG)
        assert hit and "发散" in why

    def test_valid_set_passes(self):
        hit, why = floor_violation(["FD", "PSV", "SA", "SE", "GTE", "SV"], self.CFG)
        assert not hit


# ============ 评估判据 ============

class TestEvaluateCriteria:
    def test_c2_strict_per_indicator(self):
        # 全矩阵最大载荷 0.96，但有一个指标主载荷 0.12 → C2' 必须不达标
        sol = make_solution(
            ["FD", "PSV", "SA", "FN", "SE", "GTE", "SV", "SDR"],
            [0.7, 0.7, 0.7, 0.7, 0.9, 0.96, 0.55, 0.12],
            [0.1] * 8,
        )
        crit = evaluate_criteria(sol, TH)
        assert not crit["C2_loading_clarity"]["passed"]
        assert crit["C2_loading_clarity"]["value"] == pytest.approx(0.12)


# ============ 阶梯端到端（合成数据） ============

def _synth_two_factor_data(n_obs=600, seed=42, noise_vars=()):
    """合成 8 指标数据：6 个按双因子结构生成 + 指定指标为纯噪声。

    呆滞簇 FD/PSV/SA/FN 载荷 F1，发散簇 SE/GTE/SV 载荷 F2；
    noise_vars 中的指标用独立噪声替换（SDR 默认噪声 → 模拟 v2 实况）。
    """
    rng = np.random.default_rng(seed)
    f1 = rng.normal(0, 1, n_obs)
    f2 = rng.normal(0, 1, n_obs) * 0.9
    data = {}
    for name, f, load in [
        ("FD", f1, 0.85), ("PSV", f1, 0.80), ("SA", f1, 0.75), ("FN", f1, 0.65),
        ("SE", f2, 0.85), ("GTE", f2, 0.80), ("SV", f2, 0.70),
    ]:
        data[name] = load * f + rng.normal(0, np.sqrt(1 - load ** 2) * 0.6, n_obs)
    for name in noise_vars:
        data[name] = rng.normal(0, 1, n_obs)
    # SDR 默认为噪声（模拟 v2 基线中 SDR 主载荷仅 0.124）
    if "SDR" not in noise_vars and "SDR" not in data:
        data["SDR"] = rng.normal(0, 1, n_obs)
    return pd.DataFrame(data)[ALL_INDICATORS]


class TestCorrectionLadder:
    def test_clean_data_supported_rung0(self):
        # 全部 8 指标都按双因子结构生成（含 SDR） → rung 0 直接通过
        rng = np.random.default_rng(7)
        n = 600
        f1 = rng.normal(0, 1, n)
        f2 = rng.normal(0, 1, n)
        data = {}
        for name, f, load in [
            ("FD", f1, 0.85), ("PSV", f1, 0.80), ("SA", f1, 0.75), ("FN", f1, 0.65),
            ("SE", f2, 0.85), ("GTE", f2, 0.80), ("SV", f2, 0.70), ("SDR", f2, 0.60),
        ]:
            data[name] = load * f + rng.normal(0, np.sqrt(1 - load ** 2) * 0.6, n)
        df = pd.DataFrame(data)
        ladder = run_correction_ladder(df, correction_cfg={"enabled": True})
        assert ladder["final"]["verdict"] == "supported"
        assert ladder["final"]["n_prunes"] == 0
        assert ladder["final"]["criteria"]["all_pass"] is True

    def test_noisy_indicators_pruned_exactly(self):
        # SV 交叉噪声 + SDR 无载荷噪声 → 阶梯应剔除这两个并在 6 指标上判定 supported
        df = _synth_two_factor_data(noise_vars=("SV", "SDR"))
        ladder = run_correction_ladder(df, correction_cfg={"enabled": True})
        pruned = {p["indicator"] for p in ladder["pruned"]}
        assert pruned == {"SV", "SDR"}
        assert ladder["final"]["verdict"] == "supported"
        assert ladder["final"]["n_indicators"] == 6
        assert set(ladder["final"]["active_indicators"]) == {"FD", "PSV", "SA", "FN", "SE", "GTE"}
        # 每个 rung 记录完整
        assert len(ladder["rungs"]) == 3
        for rg in ladder["rungs"]:
            assert "criteria" in rg and "parallel_analysis_n_factors" in rg
        # 剔除记录带代理效度类别
        for p in ladder["pruned"]:
            assert p["proxy_class"] in ("time", "amplitude", "distribution")

    def test_pure_noise_unsupported_structural(self):
        # 全噪声数据 → 阶梯触底，判定路径 C
        rng = np.random.default_rng(3)
        df = pd.DataFrame(rng.normal(0, 1, (600, 8)), columns=ALL_INDICATORS)
        ladder = run_correction_ladder(df, correction_cfg={"enabled": True})
        assert ladder["final"]["verdict"] == "unsupported_structural"
        assert ladder["final"]["n_prunes"] == 3
        assert ladder["stop_reason"]

    def test_floor_stops_ladder(self):
        # max_prunes 压到 0 → rung 0 不通过即止
        rng = np.random.default_rng(3)
        df = pd.DataFrame(rng.normal(0, 1, (600, 8)), columns=ALL_INDICATORS)
        ladder = run_correction_ladder(df, correction_cfg={"enabled": True, "max_prunes": 0})
        assert ladder["final"]["verdict"] == "unsupported_structural"
        assert len(ladder["rungs"]) == 1


# ============ loso_fixed_2factor ============

class TestLosoFixed2Factor:
    def _make_df(self):
        rng = np.random.default_rng(11)
        rows = []
        for subj in range(1, 5):
            n = 60
            f1 = rng.normal(0, 1, n)
            f2 = rng.normal(0, 1, n)
            for i in range(n):
                rows.append({
                    "subject": subj, "valid": True, "window_idx": i,
                    "FD_z": 0.85 * f1[i] + rng.normal(0, 0.4),
                    "PSV_z": 0.8 * f1[i] + rng.normal(0, 0.4),
                    "SE_z": 0.85 * f2[i] + rng.normal(0, 0.4),
                    "GTE_z": 0.8 * f2[i] + rng.normal(0, 0.4),
                    "SA_z": 0.7 * f1[i] + rng.normal(0, 0.4),
                    "FN_z": 0.6 * f1[i] + rng.normal(0, 0.4),
                    "SV_z": 0.7 * f2[i] + rng.normal(0, 0.4),
                    "SDR_z": 0.6 * f2[i] + rng.normal(0, 0.4),
                })
        return pd.DataFrame(rows)

    def test_smoke_runs_and_columns(self):
        df = self._make_df()
        out = loso_fixed_2factor(df, subject_col="subject",
                                 indicator_cols=["FD", "PSV", "SA", "FN", "SE", "GTE", "SV", "SDR"])
        assert len(out) == 4
        for col in ["held_out_subject", "variance_explained", "min_primary_loading",
                    "max_cross_loading", "factor_corr", "status"]:
            assert col in out.columns
        assert (out["status"] == "ok").all()

    def test_summary_stats(self):
        df = self._make_df()
        out = loso_fixed_2factor(df, subject_col="subject",
                                 indicator_cols=["FD", "PSV", "SA", "FN", "SE", "GTE", "SV", "SDR"])
        summary = summarize_loso_2factor(out, cross_threshold=0.30)
        assert summary["n_folds"] == 4
        assert summary["n_ok"] == 4
        assert 0.0 <= summary["rebound_ratio"] <= 1.0
        assert summary["cross_max_median"] >= 0


# ============ two_factor_solution 集成冒烟 ============

class TestTwoFactorSolution:
    def test_real_efa_on_synthetic(self):
        df = _synth_two_factor_data(noise_vars=("SDR",))
        sol = two_factor_solution(df)
        assert sol["n_factors"] == 2
        assert sol["variable_names"] == ALL_INDICATORS
        assert 0 < sol["variance_explained"] < 1
        assert sol["factor_corr"] is not None
        # 呆滞簇应主要载入同一因子
        loadings = np.array(sol["loadings"])
        assign = np.argmax(np.abs(loadings), axis=1)
        assert assign[0] == assign[1] == assign[2] == assign[3]  # FD/PSV/SA/FN 同因子
        assert assign[4] == assign[5] == assign[6]  # SE/GTE/SV 同因子
