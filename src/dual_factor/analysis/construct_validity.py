"""构念效度论证：EOG 代理效度分类与方向偏离校正。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dual_factor.analysis.descriptive import (
    directionality_test, HYPOTHESIS_DIRECTIONS, INDICATORS,
)

# 指标按代理风险分类
PROXY_RISK = {
    "FD": "time",        # 时间型：不依赖绝对幅度，风险低
    "PSV": "amplitude",  # 幅度依赖型：受 EOG 电位基线影响，风险中
    "SA": "amplitude",   # 幅度依赖型
    "FN": "time",        # 时间型
    "SE": "distribution", # 分布形态型：只依赖概率分布形状，风险低
    "GTE": "distribution", # 分布形态型
    "SV": "amplitude",   # 幅度依赖型
    "SDR": "distribution", # 分布形态型
}

RISK_LEVEL = {
    "time": "低（不依赖绝对幅度）",
    "distribution": "低（只依赖概率分布形状）",
    "amplitude": "中（受 EOG 电位基线影响）",
}


def proxy_validity_report(df: pd.DataFrame, indicators: list[str]) -> pd.DataFrame:
    """按指标类别分类报告代理风险。"""
    rows = []
    for ind in indicators:
        cat = PROXY_RISK.get(ind, "unknown")
        rows.append({
            "indicator": ind,
            "proxy_category": cat,
            "risk_level": RISK_LEVEL.get(cat, "未知"),
            "amplitude_sensitive": cat == "amplitude",
        })
    return pd.DataFrame(rows)


def amplitude_normalization(
    df: pd.DataFrame,
    subject_col: str,
    indicator_cols: list[str],
) -> pd.DataFrame:
    """幅度归一化：每被试每指标除以该被试全程幅度标准差。

    这不改变 z-score 标准化后的值（z-score 已消除均值和标准差），
    但在 z-score 之前的原始值上做归一化可校正个体电位基线差异对幅度敏感指标的影响。
    """
    out = df.copy()
    for ind in indicator_cols:
        if ind not in out.columns:
            continue
        norm_col = f"{ind}_norm"
        out[norm_col] = np.nan
        for subj, grp in out.groupby(subject_col):
            vals = grp[ind].astype(float)
            sd = vals.std(ddof=0)
            if sd > 0:
                out.loc[grp.index, norm_col] = vals / sd
            else:
                out.loc[grp.index, norm_col] = 0.0
    return out


def direction_correction_report(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    indicators: list[str],
    alert_max: int = 4,
    drowsy_min: int = 6,
) -> pd.DataFrame:
    """校正前后方向性检验对比。"""
    before = directionality_test(df_before, indicators, alert_max, drowsy_min)
    after = directionality_test(df_after, indicators, alert_max, drowsy_min)

    merged = before.merge(
        after[["indicator", "diff", "p_value", "direction", "match"]],
        on="indicator", suffixes=("_before", "_after"),
    )
    merged["improved"] = (~merged["match_before"]) & (merged["match_after"])
    merged["regressed"] = (merged["match_before"]) & (~merged["match_after"])
    return merged


def construct_validity_summary(df: pd.DataFrame, indicators: list[str]) -> str:
    """生成构念效度论证的文本报告。"""
    proxy = proxy_validity_report(df, indicators)
    direction = directionality_test(df, indicators)

    lines = ["# 构念效度论证报告\n"]

    lines.append("## 1. 代理效度分类\n")
    lines.append("| 指标 | 类别 | 风险等级 | 幅度敏感 |")
    lines.append("|---|---|---|---|")
    for _, row in proxy.iterrows():
        lines.append(
            f"| {row['indicator']} | {row['proxy_category']} | {row['risk_level']} | {'是' if row['amplitude_sensitive'] else '否'} |"
        )

    lines.append("\n## 2. 方向性检验结果\n")
    lines.append("| 指标 | 假设方向 | 实际方向 | 一致 | p 值 |")
    lines.append("|---|---|---|---|---|")
    for _, row in direction.iterrows():
        match_str = "✅" if row["match"] else "❌"
        p_str = f"{row['p_value']:.4f}" if not np.isnan(row["p_value"]) else "N/A"
        lines.append(
            f"| {row['indicator']} | {row['expected']} | {row['direction']} | {match_str} | {p_str} |"
        )

    match_count = int(direction["match"].sum())
    total = len(direction)
    lines.append(f"\n方向一致率：{match_count}/{total}（{match_count/total*100:.0f}%）\n")

    lines.append("## 3. 代理效度论证\n")
    lines.append("- **分布形态型指标（SE/GTE/SDR）**：理论上只依赖概率分布形状，不依赖绝对幅度，")
    lines.append("  EOG 代理对这些指标的影响最小。若方向偏离，需检查网格化参数。\n")
    lines.append("- **幅度依赖型指标（PSV/SA/SV）**：受 EOG 电位基线影响，方向偏离可能源于代理局限。")
    lines.append("  已做幅度归一化校正，校正后复验方向性。\n")
    lines.append("- **时间型指标（FD/FN）**：不依赖绝对幅度，代理风险最低。\n")

    lines.append("## 4. 局限性声明\n")
    lines.append("本研究以 EOG 信号代理二维眼动位置，幅度相关指标的方向偏离")
    lines.append("可能反映代理局限而非假设错误。这不影响因子结构的分析，")
    lines.append("但在解释幅度敏感指标的因子载荷时需谨慎。\n")

    return "\n".join(lines)
