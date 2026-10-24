"""描述性统计与方向性检验。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, shapiro

INDICATORS = ["FD", "PSV", "SA", "FN", "SE", "GTE", "SV", "SDR"]

# 假设方向：alert→drowsy 的预期变化
HYPOTHESIS_DIRECTIONS = {
    "FD": "increase",
    "PSV": "decrease",
    "SA": "decrease",
    "FN": "decrease",
    "SE": "increase",
    "GTE": "increase",
    "SV": "increase",
    "SDR": "increase",
}


def _split_alert_drowsy(df: pd.DataFrame, alert_max: int = 4, drowsy_min: int = 6):
    """按 KSS 二分类。"""
    alert = df[df["kss"] <= alert_max]
    drowsy = df[df["kss"] >= drowsy_min]
    return alert, drowsy


def descriptive_stats(
    df: pd.DataFrame,
    indicators: list[str],
    alert_max: int = 4,
    drowsy_min: int = 6,
) -> pd.DataFrame:
    """各指标在 alert/drowsy 条件下的描述性统计。"""
    alert, drowsy = _split_alert_drowsy(df, alert_max, drowsy_min)
    rows = []
    for ind in indicators:
        a = alert[ind].dropna() if ind in alert.columns else pd.Series(dtype=float)
        d = drowsy[ind].dropna() if ind in drowsy.columns else pd.Series(dtype=float)
        rows.append({
            "indicator": ind,
            "n_alert": len(a), "n_drowsy": len(d),
            "mean_alert": float(a.mean()) if len(a) else np.nan,
            "mean_drowsy": float(d.mean()) if len(d) else np.nan,
            "std_alert": float(a.std()) if len(a) else np.nan,
            "std_drowsy": float(d.std()) if len(d) else np.nan,
        })
    return pd.DataFrame(rows)


def directionality_test(
    df: pd.DataFrame,
    indicators: list[str],
    alert_max: int = 4,
    drowsy_min: int = 6,
) -> pd.DataFrame:
    """各指标 alert vs drowsy 的方向性检验（Mann-Whitney U）。"""
    alert, drowsy = _split_alert_drowsy(df, alert_max, drowsy_min)
    rows = []
    for ind in indicators:
        a = alert[ind].dropna().values if ind in alert.columns else np.array([])
        d = drowsy[ind].dropna().values if ind in drowsy.columns else np.array([])
        if len(a) == 0 or len(d) == 0:
            rows.append({"indicator": ind, "diff": np.nan, "p_value": np.nan,
                         "direction": "unknown", "expected": HYPOTHESIS_DIRECTIONS.get(ind),
                         "match": False})
            continue
        diff = float(d.mean() - a.mean())
        try:
            _, p = mannwhitneyu(a, d, alternative="two-sided")
        except ValueError:
            p = np.nan
        direction = "increase" if diff > 0 else "decrease"
        expected = HYPOTHESIS_DIRECTIONS.get(ind, "unknown")
        match = (direction == expected)
        rows.append({
            "indicator": ind, "diff": diff, "p_value": float(p),
            "direction": direction, "expected": expected, "match": match,
        })
    return pd.DataFrame(rows)


def indicator_distribution_report(
    df: pd.DataFrame,
    indicators: list[str],
    alert_max: int = 4,
    drowsy_min: int = 6,
) -> pd.DataFrame:
    """各指标的完整分布报告：均值±标准差、中位数与IQR、缺失率、正态性检验。"""
    alert, drowsy = _split_alert_drowsy(df, alert_max, drowsy_min)
    rows = []
    for ind in indicators:
        col = df[ind] if ind in df.columns else pd.Series(dtype=float)
        a = alert[ind].dropna() if ind in alert.columns else pd.Series(dtype=float)
        d = drowsy[ind].dropna() if ind in drowsy.columns else pd.Series(dtype=float)
        all_vals = col.dropna()

        # Shapiro-Wilk 正态性检验（样本量 ≤ 5000 时可用）
        if len(all_vals) >= 3 and len(all_vals) <= 5000:
            try:
                sw_stat, sw_p = shapiro(all_vals.values)
            except Exception:
                sw_stat, sw_p = np.nan, np.nan
        else:
            sw_stat, sw_p = np.nan, np.nan

        rows.append({
            "indicator": ind,
            "n_total": int(len(all_vals)),
            "n_alert": int(len(a)),
            "n_drowsy": int(len(d)),
            "missing_rate": float(col.isna().mean()),
            "mean_all": float(all_vals.mean()) if len(all_vals) else np.nan,
            "std_all": float(all_vals.std()) if len(all_vals) else np.nan,
            "mean_alert": float(a.mean()) if len(a) else np.nan,
            "std_alert": float(a.std()) if len(a) else np.nan,
            "mean_drowsy": float(d.mean()) if len(d) else np.nan,
            "std_drowsy": float(d.std()) if len(d) else np.nan,
            "median_all": float(all_vals.median()) if len(all_vals) else np.nan,
            "iqr_all": float(all_vals.quantile(0.75) - all_vals.quantile(0.25)) if len(all_vals) else np.nan,
            "shapiro_stat": float(sw_stat),
            "shapiro_p": float(sw_p),
            "normal": bool(sw_p > 0.05) if not np.isnan(sw_p) else None,
        })
    return pd.DataFrame(rows)
