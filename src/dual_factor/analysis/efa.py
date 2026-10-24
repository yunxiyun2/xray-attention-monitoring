"""EFA：前置检验 + 因子提取 + 平行分析。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import (
    calculate_kmo as _fa_kmo,
    calculate_bartlett_sphericity as _fa_bartlett,
)


# ============ 前置检验 ============

def calculate_kmo(data: pd.DataFrame) -> tuple[float, dict[str, float]]:
    """KMO 采样充分性检验。返回 (overall, per_variable)。"""
    kmo_per_var, kmo_overall = _fa_kmo(data)
    return float(kmo_overall), dict(zip(data.columns, kmo_per_var))


def bartlett_sphericity(data: pd.DataFrame) -> tuple[float, float]:
    """Bartlett 球形检验。返回 (chi2, p_value)。"""
    chi2, p = _fa_bartlett(data)
    return float(chi2), float(p)


def precheck_efa(data: pd.DataFrame) -> dict:
    """EFA 前置检验汇总。"""
    kmo_overall, kmo_per_var = calculate_kmo(data)
    chi2, p = bartlett_sphericity(data)
    return {
        "kmo_overall": kmo_overall,
        "kmo_per_variable": kmo_per_var,
        "bartlett_chi2": chi2,
        "bartlett_p": p,
        "kmo_pass": kmo_overall > 0.50,
        "bartlett_pass": p < 0.05,
    }


# ============ 因子提取与平行分析 ============

@dataclass
class EFAResult:
    n_factors: int
    loadings: np.ndarray              # (n_vars, n_factors)
    variance_explained: float        # 累计方差解释率
    factor_corr: np.ndarray | None   # (n_factors, n_factors) 或 None
    variable_names: list[str]
    parallel_analysis: dict | None


def run_efa(
    data: pd.DataFrame,
    extraction: str = "pa",
    rotation: str = "promax",
    max_factors: int = 4,
    variance_threshold: float = 0.60,
    n_iter: int = 1000,
    percentile: float = 95.0,
) -> EFAResult:
    """主轴因子法 + Promax 旋转，按平行分析确定因子数。"""
    pa = parallel_analysis(data, n_iter=n_iter, percentile=percentile, max_factors=max_factors)
    actual = pa["actual_eigenvalues"]
    rand_refs = pa["random_eigenvalues"]
    n_factors = 0
    for k in range(min(len(actual), len(rand_refs))):
        if actual[k] > rand_refs[k]:
            n_factors = k + 1
        else:
            break

    # 若无信号，回退到 Kaiser 准则（>1）
    if n_factors == 0:
        n_factors = int(np.sum(np.array(actual) > 1.0)) or 1
    n_factors = min(n_factors, max_factors)

    method_map = {"pa": "principal", "ml": "ml", "minres": "minres"}
    method = method_map.get(extraction, "principal")

    rot_map = {"promax": "promax", "varimax": "varimax", "oblimin": "oblimin"}
    rot = rot_map.get(rotation, "promax")

    fa = FactorAnalyzer(n_factors=n_factors, rotation=rot, method=method)
    fa.fit(data)
    loadings = fa.loadings_
    variance = fa.get_factor_variance()
    cum_var = float(variance[2][-1]) if len(variance) >= 3 else 0.0
    corr = fa.phi_ if hasattr(fa, "phi_") and fa.phi_ is not None else None

    return EFAResult(
        n_factors=n_factors,
        loadings=loadings,
        variance_explained=cum_var,
        factor_corr=corr,
        variable_names=list(data.columns),
        parallel_analysis=pa,
    )


def run_efa_fixed(
    data: pd.DataFrame,
    n_factors: int,
    extraction: str = "pa",
    rotation: str = "promax",
) -> EFAResult:
    """提取指定因子数的 EFA（不自动确定因子数，用于 2vs3 对比）。"""
    method_map = {"pa": "principal", "ml": "ml", "minres": "minres"}
    method = method_map.get(extraction, "principal")
    rot_map = {"promax": "promax", "varimax": "varimax", "oblimin": "oblimin"}
    rot = rot_map.get(rotation, "promax")

    fa = FactorAnalyzer(n_factors=n_factors, rotation=rot, method=method)
    fa.fit(data)
    loadings = fa.loadings_
    variance = fa.get_factor_variance()
    cum_var = float(variance[2][-1]) if len(variance) >= 3 else 0.0
    corr = fa.phi_ if hasattr(fa, "phi_") and fa.phi_ is not None else None

    return EFAResult(
        n_factors=n_factors,
        loadings=loadings,
        variance_explained=cum_var,
        factor_corr=corr,
        variable_names=list(data.columns),
        parallel_analysis=None,
    )


def run_efa_compare(
    data: pd.DataFrame,
    n_factors_list: list[int] | None = None,
    extraction: str = "pa",
    rotation: str = "promax",
    n_iter: int = 1000,
    percentile: float = 95.0,
    max_factors: int = 4,
) -> dict:
    """并行提取多个因子解并输出对比表。

    默认对比 2 因子解和 3 因子解，回答：
    1. 平行分析建议几个因子？
    2. 2 因子解的交叉载荷是否 <0.3？
    3. 3 因子解是否比 2 因子解显著降低交叉载荷？
    4. 第三因子包含哪些指标？有无实质意义？
    """
    if n_factors_list is None:
        n_factors_list = [2, 3]

    # 平行分析确定建议因子数
    pa = parallel_analysis(data, n_iter=n_iter, percentile=percentile, max_factors=max_factors)
    actual_eigs = pa["actual_eigenvalues"]
    rand_refs = pa["random_eigenvalues"]
    pa_n_factors = 0
    for k in range(min(len(actual_eigs), len(rand_refs))):
        if actual_eigs[k] > rand_refs[k]:
            pa_n_factors = k + 1
        else:
            break
    if pa_n_factors == 0:
        pa_n_factors = int(np.sum(np.array(actual_eigs) > 1.0)) or 1

    results = {}
    for nf in n_factors_list:
        try:
            res = run_efa_fixed(data, nf, extraction=extraction, rotation=rotation)
            abs_loadings = np.abs(res.loadings)
            max_loading = float(np.max(abs_loadings))

            # 每变量最大载荷
            per_var_max = np.max(abs_loadings, axis=1) if abs_loadings.ndim > 1 else abs_loadings
            # 每变量次大载荷（交叉载荷）
            if abs_loadings.ndim > 1 and abs_loadings.shape[1] > 1:
                sorted_abs = np.sort(abs_loadings, axis=1)
                per_var_cross = sorted_abs[:, -2]
                cross_max = float(np.max(per_var_cross))
            else:
                cross_max = 0.0

            # 指标归属（每指标最大载荷的因子号）
            if abs_loadings.ndim > 1:
                assignments = np.argmax(abs_loadings, axis=1).tolist()
            else:
                assignments = [0] * len(res.variable_names)

            corr_val = float(res.factor_corr[0, 1]) if (
                res.factor_corr is not None and res.factor_corr.shape[0] >= 2
            ) else None

            results[f"{nf}_factor"] = {
                "n_factors": nf,
                "variance_explained": res.variance_explained,
                "loadings": res.loadings.tolist(),
                "variable_names": res.variable_names,
                "max_loading": max_loading,
                "cross_loading_max": cross_max,
                "factor_corr": corr_val,
                "assignments": assignments,
                "per_variable_max": per_var_max.tolist(),
                "factor_corr_matrix": res.factor_corr.tolist() if res.factor_corr is not None else None,
            }
        except Exception as e:
            results[f"{nf}_factor"] = {
                "n_factors": nf,
                "status": "error",
                "message": str(e),
            }

    return {
        "parallel_analysis_n_factors": pa_n_factors,
        "parallel_analysis": pa,
        "comparisons": results,
    }


def parallel_analysis(
    data: pd.DataFrame,
    n_iter: int = 1000,
    percentile: float = 95,
    max_factors: int = 4,
    seed: int = 42,
) -> dict:
    """平行分析：与随机数据特征值比较。"""
    rng = np.random.default_rng(seed)
    n_obs, n_vars = data.shape
    k = min(max_factors, n_vars)

    # 实际特征值
    corr = np.corrcoef(data.values, rowvar=False)
    actual_eigs = np.sort(np.linalg.eigvalsh(corr))[::-1][:k]

    # 随机数据特征值
    rand_eigs_all = np.zeros((n_iter, k))
    for i in range(n_iter):
        rand_data = rng.standard_normal((n_obs, n_vars))
        rand_corr = np.corrcoef(rand_data, rowvar=False)
        rand_eigs_all[i] = np.sort(np.linalg.eigvalsh(rand_corr))[::-1][:k]
    rand_eigs = np.percentile(rand_eigs_all, percentile, axis=0)

    return {
        "actual_eigenvalues": actual_eigs.tolist(),
        "random_eigenvalues": rand_eigs.tolist(),
        "n_iter": n_iter,
        "percentile": percentile,
    }
