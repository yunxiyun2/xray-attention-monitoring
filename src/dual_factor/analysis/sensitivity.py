"""窗口非独立性敏感性分析：被试级聚合 EFA、留一被试者 EFA、非重叠窗口子集 EFA。"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from dual_factor.analysis.efa import run_efa, precheck_efa


def aggregate_subject_level(
    df: pd.DataFrame,
    subject_col: str,
    indicator_cols: list[str],
    suffix: str = "_z",
) -> pd.DataFrame:
    """将被试的多个窗口聚合为被试级均值（1 被试 → 1 行 8 指标）。"""
    z_cols = [f"{c}{suffix}" for c in indicator_cols]
    available = [c for c in z_cols if c in df.columns]
    if not available:
        available = [c for c in indicator_cols if c in df.columns]
    if not available:
        raise ValueError(f"指标列 {indicator_cols} 在 DataFrame 中不存在")
    valid = df[df["valid"]] if "valid" in df.columns else df
    agg = valid.groupby(subject_col)[available].mean().reset_index()
    return agg


def subject_level_efa(
    df: pd.DataFrame,
    subject_col: str,
    indicator_cols: list[str],
    suffix: str = "_z",
    **efa_kwargs,
) -> dict:
    """在 14 行被试级聚合数据上做 EFA。"""
    agg = aggregate_subject_level(df, subject_col, indicator_cols, suffix)
    data = agg.drop(columns=[subject_col]).dropna()

    if data.shape[0] < len(indicator_cols):
        return {
            "status": "insufficient_samples",
            "n_subjects": int(data.shape[0]),
            "message": f"被试级样本量 {data.shape[0]} 小于指标数 {len(indicator_cols)}，EFA 可能不收敛",
            "efa_result": None,
        }

    precheck = precheck_efa(data)
    try:
        result = run_efa(data, **efa_kwargs)
        loadings = result.loadings
        abs_loadings = np.abs(loadings)
        max_loading = float(np.max(abs_loadings))
        if loadings.shape[1] > 1:
            sorted_abs = np.sort(abs_loadings, axis=1)
            cross_max = float(np.max(sorted_abs[:, -2]))
        else:
            cross_max = 0.0
        corr_val = float(result.factor_corr[0, 1]) if (
            result.factor_corr is not None and result.factor_corr.shape[0] >= 2
        ) else None
        return {
            "status": "ok",
            "n_subjects": int(data.shape[0]),
            "n_factors": result.n_factors,
            "variance_explained": result.variance_explained,
            "max_loading": max_loading,
            "cross_loading_max": cross_max,
            "factor_corr": corr_val,
            "precheck": precheck,
            "efa_result": {
                "n_factors": result.n_factors,
                "variance_explained": result.variance_explained,
                "loadings": result.loadings.tolist(),
                "variable_names": result.variable_names,
                "factor_corr": result.factor_corr.tolist() if result.factor_corr is not None else None,
            },
        }
    except Exception as e:
        return {
            "status": "error",
            "n_subjects": int(data.shape[0]),
            "message": str(e),
            "efa_result": None,
        }


def loso_efa(
    df: pd.DataFrame,
    subject_col: str,
    indicator_cols: list[str],
    suffix: str = "_z",
    **efa_kwargs,
) -> pd.DataFrame:
    """留一被试者 EFA：逐个留出 1 名被试，在剩余被试上做 EFA。"""
    z_cols = [f"{c}{suffix}" for c in indicator_cols]
    available = [c for c in z_cols if c in df.columns]
    if not available:
        available = [c for c in indicator_cols if c in df.columns]
    if not available:
        raise ValueError(f"指标列 {indicator_cols} 在 DataFrame 中不存在")

    valid = df[df["valid"]] if "valid" in df.columns else df
    subjects = sorted(valid[subject_col].unique())
    results = []

    for held_out in subjects:
        train = valid[valid[subject_col] != held_out][available].dropna()
        if train.shape[0] < len(indicator_cols):
            results.append({
                "held_out_subject": held_out,
                "n_train": int(train.shape[0]),
                "n_factors": None, "variance_explained": None,
                "max_loading": None, "cross_loading_max": None, "factor_corr": None,
                "status": "insufficient",
            })
            continue
        try:
            result = run_efa(train, **efa_kwargs)
            abs_loadings = np.abs(result.loadings)
            max_loading = float(np.max(abs_loadings))
            if result.loadings.shape[1] > 1:
                sorted_abs = np.sort(abs_loadings, axis=1)
                cross_max = float(np.max(sorted_abs[:, -2]))
            else:
                cross_max = 0.0
            corr_val = float(result.factor_corr[0, 1]) if (
                result.factor_corr is not None and result.factor_corr.shape[0] >= 2
            ) else None
            results.append({
                "held_out_subject": held_out,
                "n_train": int(train.shape[0]),
                "n_factors": result.n_factors,
                "variance_explained": result.variance_explained,
                "max_loading": max_loading,
                "cross_loading_max": cross_max,
                "factor_corr": corr_val,
                "status": "ok",
            })
        except Exception as e:
            results.append({
                "held_out_subject": held_out,
                "n_train": int(train.shape[0]),
                "n_factors": None, "variance_explained": None,
                "max_loading": None, "cross_loading_max": None, "factor_corr": None,
                "status": f"error: {e}",
            })

    return pd.DataFrame(results)


def nonoverlap_efa(
    df: pd.DataFrame,
    subject_col: str,
    indicator_cols: list[str],
    window_col: str = "window_idx",
    window_sec: float = 30.0,
    stride_sec: float = 10.0,
    suffix: str = "_z",
    **efa_kwargs,
) -> dict:
    """在非重叠窗口子集上做 EFA（每隔 stride_sec/window_sec 取一个窗口）。"""
    z_cols = [f"{c}{suffix}" for c in indicator_cols]
    available = [c for c in z_cols if c in df.columns]
    if not available:
        available = [c for c in indicator_cols if c in df.columns]
    if not available:
        raise ValueError(f"指标列 {indicator_cols} 在 DataFrame 中不存在")

    valid = df[df["valid"]] if "valid" in df.columns else df
    # 每隔 N 个窗口取 1 个，使窗口不重叠
    step = int(window_sec / stride_sec)  # e.g. 30/10 = 3 → 每 3 个取 1 个
    if step < 1:
        step = 1
    # 按 subject 分组，每组取非重叠子集
    parts = []
    for subj, grp in valid.groupby(subject_col):
        grp_sorted = grp.sort_values(window_col)
        non_overlap = grp_sorted.iloc[::step]
        parts.append(non_overlap)
    if not parts:
        return {"status": "no_data", "efa_result": None}
    subset = pd.concat(parts, ignore_index=True)
    data = subset[available].dropna()

    if data.shape[0] < len(indicator_cols):
        return {
            "status": "insufficient_samples",
            "n_windows": int(data.shape[0]),
            "message": f"非重叠窗口数 {data.shape[0]} 小于指标数 {len(indicator_cols)}",
            "efa_result": None,
        }

    precheck = precheck_efa(data)
    try:
        result = run_efa(data, **efa_kwargs)
        abs_loadings = np.abs(result.loadings)
        max_loading = float(np.max(abs_loadings))
        if result.loadings.shape[1] > 1:
            sorted_abs = np.sort(abs_loadings, axis=1)
            cross_max = float(np.max(sorted_abs[:, -2]))
        else:
            cross_max = 0.0
        corr_val = float(result.factor_corr[0, 1]) if (
            result.factor_corr is not None and result.factor_corr.shape[0] >= 2
        ) else None
        return {
            "status": "ok",
            "n_windows": int(data.shape[0]),
            "n_factors": result.n_factors,
            "variance_explained": result.variance_explained,
            "max_loading": max_loading,
            "cross_loading_max": cross_max,
            "factor_corr": corr_val,
            "precheck": precheck,
            "efa_result": {
                "n_factors": result.n_factors,
                "variance_explained": result.variance_explained,
                "loadings": result.loadings.tolist(),
                "variable_names": result.variable_names,
                "factor_corr": result.factor_corr.tolist() if result.factor_corr is not None else None,
            },
        }
    except Exception as e:
        return {
            "status": "error",
            "n_windows": int(data.shape[0]),
            "message": str(e),
            "efa_result": None,
        }


def sensitivity_summary(
    window_efa_result: dict,
    subject_efa_result: dict,
    loso_df: pd.DataFrame,
    nonoverlap_efa_result: dict,
) -> dict:
    """汇总三种敏感性分析结论，判断窗口级结论是否稳健。"""
    window_factors = window_efa_result.get("n_factors")
    subject_factors = subject_efa_result.get("n_factors") if subject_efa_result.get("status") == "ok" else None
    nonoverlap_factors = nonoverlap_efa_result.get("n_factors") if nonoverlap_efa_result.get("status") == "ok" else None

    loso_ok = loso_df[loso_df["status"] == "ok"] if "status" in loso_df.columns else pd.DataFrame()
    if len(loso_ok) > 0:
        loso_factors_list = loso_ok["n_factors"].dropna().tolist()
        loso_factors_mode = max(set(loso_factors_list), key=loso_factors_list.count) if loso_factors_list else None
    else:
        loso_factors_list = []
        loso_factors_mode = None

    factor_counts = [f for f in [window_factors, subject_factors, nonoverlap_factors, loso_factors_mode] if f is not None]
    consistent = len(set(factor_counts)) <= 1 if factor_counts else False

    return {
        "window_level_n_factors": window_factors,
        "subject_level_n_factors": subject_factors,
        "nonoverlap_n_factors": nonoverlap_factors,
        "loso_n_factors_mode": loso_factors_mode,
        "loso_n_factors_list": loso_factors_list,
        "consistent": consistent,
        "conclusion": "被试级结论与窗口级一致，因子结构跨方法稳健" if consistent
                      else "被试级结论与窗口级不一致，因子结构跨方法不稳定，需如实报告",
    }
