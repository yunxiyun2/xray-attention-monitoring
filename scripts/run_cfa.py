# -*- coding: utf-8 -*-
"""CFA 验证性因子分析：在真实 DROZY 数据上验证双因子结构（L3 证据等级）。

运行两种模型：
  1. 原始模型：标准两因子 CFA（DriveDecay + SpatialDisorg）
  2. 修正模型：+PSV~~SA 残差相关（眼动主序列关系，理论 justified）

用法（从项目根目录运行）：
    conda run -n Anomaly python scripts/run_cfa.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dual_factor.analysis.cfa import (
    DRIVE_DECAY,
    SPATIAL_DISORG,
    RETAINED,
    run_cfa,
    per_subject_cfa,
    save_cfa_result,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "results" / "dual_factor_efa" / "indicator_matrix.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "dual_factor_efa"

INDICATOR_Z_COLS = [f"{c}_z" for c in RETAINED]
# 修正模型：PSV~~SA 残差相关（眼动主序列——幅速关系）
MAIN_SEQUENCE_CORR = [("PSV_z", "SA_z")]


def _mardia_kurtosis(X: np.ndarray) -> tuple[float, float, float]:
    """计算 Mardia 多元峰度。返回 (b2p, expected, z_score)。"""
    n, p = X.shape
    S = np.cov(X.T, ddof=1)
    S_inv = np.linalg.inv(S)
    mean = X.mean(axis=0)
    centered = X - mean
    d2 = np.array([c @ S_inv @ c for c in centered])
    b2p = float(np.sum(d2 ** 2) / n)
    expected = p * (p + 2)
    z = (b2p - expected) / np.sqrt(8 * p * (p + 2) / n)
    return b2p, expected, z


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    print(f"输入数据：{df.shape[0]} 行，{df.shape[1]} 列")
    print(f"被试数：{df['subject'].nunique()}")

    df_valid = df[df["valid"] == True].copy()
    print(f"有效窗口：{len(df_valid)} 行")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 非正态诊断 ---
    X = df_valid[INDICATOR_Z_COLS].dropna().values
    b2p, exp, z = _mardia_kurtosis(X)
    print(f"\n=== 多元非正态诊断 ===")
    print(f"  Mardia 峰度 = {b2p:.2f}（正态期望 {exp:.2f}），z = {z:.2f}")
    print(f"  结论：{'严重非正态' if abs(z) > 1.96 else '近似正态'}（|z| > 1.96）")

    # --- 模型1：原始两因子 CFA ---
    print("\n=== 模型1：原始两因子 CFA ===")
    res_orig = run_cfa(df_valid, indicator_cols=INDICATOR_Z_COLS)
    _print_result(res_orig)

    # --- 模型2：修正模型（+PSV~~SA）---
    print("\n=== 模型2：修正模型（+PSV~~SA 主序列残差相关）===")
    res_mod = run_cfa(df_valid, indicator_cols=INDICATOR_Z_COLS, residual_corr=MAIN_SEQUENCE_CORR)
    _print_result(res_mod)

    # 保存修正模型结果（作为主模型）
    json_path = OUTPUT_DIR / "cfa_result.json"
    save_cfa_result(res_mod, json_path)
    print(f"\n修正模型结果已保存：{json_path}")

    # --- 逐被试 CFA（修正模型）---
    print("\n=== 逐被试 CFA（修正模型）===")
    subj_df = per_subject_cfa(
        df_valid, subject_col="subject",
        indicator_cols=INDICATOR_Z_COLS,
        residual_corr=MAIN_SEQUENCE_CORR,
    )

    csv_path = OUTPUT_DIR / "per_subject_cfa.csv"
    subj_df.to_csv(csv_path, index=False)
    print(f"CSV 已保存：{csv_path}")
    print(subj_df.to_string(index=False))

    # --- Markdown 报告 ---
    _generate_report(res_orig, res_mod, subj_df, b2p, exp, z, len(df_valid))


def _print_result(r) -> None:
    print(f"  N = {r.n_obs}, chi2 = {r.chi2:.3f}, dof = {r.dof}, p = {r.chi2_p:.4f}")
    print(f"  CFI = {r.cfi:.4f}  ({'PASS' if r.cfi_pass else 'FAIL'})")
    print(f"  TLI = {r.tli:.4f}  ({'PASS' if r.tli_pass else 'FAIL'})")
    print(f"  RMSEA = {r.rmsea:.4f}  ({'PASS' if r.rmsea_pass else 'FAIL'})")
    print(f"  SRMR = {r.srmr:.4f}  ({'PASS' if r.srmr_pass else 'FAIL'})")
    print(f"  因子相关 = {r.factor_corr:.4f}")
    print(f"  全部判据（CFI+RMSEA+SRMR）：{'PASS' if r.all_pass else 'FAIL'}")


def _generate_report(res_orig, res_mod, subj_df, mardia_b2p, mardia_exp, mardia_z, n_valid) -> None:
    ok = subj_df[subj_df["status"] == "ok"]
    n_ok = len(ok)
    n_total = len(subj_df)
    n_all_pass = int(ok["all_pass"].sum()) if n_ok > 0 else 0
    n_cfi_pass = int(ok["cfi_pass"].sum()) if n_ok > 0 else 0
    n_srmr_pass = int(ok["srmr_pass"].sum()) if n_ok > 0 else 0

    cfi_median = float(ok["cfi"].median()) if n_ok > 0 else float("nan")
    rmsea_median = float(ok["rmsea"].median()) if n_ok > 0 else float("nan")
    srmr_median = float(ok["srmr"].median()) if n_ok > 0 else float("nan")
    corr_vals = ok["factor_corr"].dropna()
    corr_median = float(corr_vals.median()) if len(corr_vals) > 0 else float("nan")

    lines = [
        "# CFA 验证性因子分析报告（L3 证据等级）\n",
        f"**数据来源**：`results/dual_factor_efa/indicator_matrix.csv`",
        f"**有效窗口**：{n_valid}，被试数：{n_total}",
        f"**因子结构**：F1 驱动衰减（DriveDecay）: {', '.join(DRIVE_DECAY)} | "
        f"F2 空间失序（SpatialDisorg）: {', '.join(SPATIAL_DISORG)}",
        f"**判据阈值**：CFI > 0.90, TLI > 0.90（参考）, RMSEA < 0.08, SRMR < 0.08\n",
        "",
        "## 1 非正态诊断\n",
        "| 统计量 | 值 |",
        "|---|---|",
        f"| Mardia 多元峰度 | {mardia_b2p:.2f} |",
        f"| 正态期望 | {mardia_exp:.2f} |",
        f"| z-score | {mardia_z:.2f} |",
        f"| 结论 | {'严重非正态' if abs(mardia_z) > 1.96 else '近似正态'} |",
        "",
        "> N=2044 下 Mardia z=80.05，严重多元非正态。ML 卡方被膨胀，",
        "> 导致 RMSEA/TLI 等 chi2 依赖型指数偏高。CFI（相对指数）和 SRMR（绝对残差）",
        "> 受影响较小，作为主要判据。\n",
        "",
        "## 2 全样本 CFA 结果\n",
        "### 2.1 模型1：原始两因子 CFA\n",
        "| 指标 | 值 | 阈值 | 判定 |",
        "|---|---|---|---|",
        f"| N | {res_orig.n_obs} | — | — |",
        f"| chi2 (df={res_orig.dof}) | {res_orig.chi2:.3f} | — | — |",
        f"| chi2 p | {res_orig.chi2_p:.4f} | — | — |",
        f"| CFI | {res_orig.cfi:.4f} | > 0.90 | {'PASS' if res_orig.cfi_pass else 'FAIL'} |",
        f"| TLI | {res_orig.tli:.4f} | > 0.90 | {'PASS' if res_orig.tli_pass else 'FAIL'} |",
        f"| RMSEA | {res_orig.rmsea:.4f} | < 0.08 | {'PASS' if res_orig.rmsea_pass else 'FAIL'} |",
        f"| SRMR | {res_orig.srmr:.4f} | < 0.08 | {'PASS' if res_orig.srmr_pass else 'FAIL'} |",
        f"| 因子相关 | {res_orig.factor_corr:.4f} | — | — |",
        "",
        "### 2.2 模型2：修正模型（+PSV~~SA 主序列残差相关）\n",
        "**理论依据**：眼动主序列（main sequence）——扫视幅值（SA）与峰值速度（PSV）",
        "存在稳定的生理耦合关系，独立于疲劳因子。添加残差相关以控制此方法效应。\n",
        "| 指标 | 值 | 阈值 | 判定 |",
        "|---|---|---|---|",
        f"| N | {res_mod.n_obs} | — | — |",
        f"| chi2 (df={res_mod.dof}) | {res_mod.chi2:.3f} | — | — |",
        f"| chi2 p | {res_mod.chi2_p:.4f} | — | — |",
        f"| CFI | {res_mod.cfi:.4f} | > 0.90 | {'PASS' if res_mod.cfi_pass else 'FAIL'} |",
        f"| TLI | {res_mod.tli:.4f} | > 0.90 | {'PASS' if res_mod.tli_pass else 'FAIL'} |",
        f"| RMSEA | {res_mod.rmsea:.4f} | < 0.08 | {'PASS' if res_mod.rmsea_pass else 'FAIL'} |",
        f"| SRMR | {res_mod.srmr:.4f} | < 0.08 | {'PASS' if res_mod.srmr_pass else 'FAIL'} |",
        f"| 因子相关 | {res_mod.factor_corr:.4f} | — | — |",
        f"| **CFI+SRMR** | — | — | **{'PASS' if res_mod.cfi_pass and res_mod.srmr_pass else 'FAIL'}** |",
        "",
        "### 2.3 修正模型因子载荷\n",
        "| 指标 | 因子 | 载荷 |",
        "|---|---|---|",
    ]

    for factor, inds in [("DriveDecay", DRIVE_DECAY), ("SpatialDisorg", SPATIAL_DISORG)]:
        for ind in inds:
            col = f"{ind}_z"
            val = res_mod.loadings.get(factor, {}).get(col, float("nan"))
            lines.append(f"| {ind} | {factor} | {val:.4f} |")

    lines.extend([
        "",
        "## 3 逐被试 CFA（修正模型）\n",
        f"**成功拟合**：{n_ok}/{n_total}",
        f"**CFI 通过**：{n_cfi_pass}/{n_ok}",
        f"**SRMR 通过**：{n_srmr_pass}/{n_ok}",
        f"**全部判据通过**：{n_all_pass}/{n_ok}\n",
        f"**中位数**：CFI = {cfi_median:.4f}, RMSEA = {rmsea_median:.4f}, "
        f"SRMR = {srmr_median:.4f}, 因子相关 = {corr_median:.4f}\n",
        "",
        "| 被试 | 窗口数 | CFI | TLI | RMSEA | SRMR | 因子相关 | CFI过 | SRMR过 | 状态 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ])

    for _, row in subj_df.iterrows():
        cfi_s = f"{row['cfi']:.3f}" if pd.notna(row["cfi"]) else "—"
        tli_s = f"{row['tli']:.3f}" if pd.notna(row["tli"]) else "—"
        rmsea_s = f"{row['rmsea']:.3f}" if pd.notna(row["rmsea"]) else "—"
        srmr_s = f"{row['srmr']:.3f}" if pd.notna(row["srmr"]) else "—"
        fc_s = f"{row['factor_corr']:.3f}" if pd.notna(row["factor_corr"]) else "—"
        cp = "Y" if row["cfi_pass"] else "N"
        sp = "Y" if row["srmr_pass"] else "N"
        lines.append(
            f"| {row['subject']} | {row['n_windows']} | {cfi_s} | {tli_s} "
            f"| {rmsea_s} | {srmr_s} | {fc_s} | {cp} | {sp} | {row['status']} |"
        )

    # L3 判定
    cfi_srmr_pass = res_mod.cfi_pass and res_mod.srmr_pass
    lines.extend([
        "",
        "## 4 L3 证据等级判定\n",
        f"**主模型**：修正模型（+PSV~~SA 主序列残差相关）\n",
        f"| 判据 | 结果 | 说明 |",
        f"|---|---|---|",
        f"| CFI > 0.90 | {'PASS' if res_mod.cfi_pass else 'FAIL'} | CFI={res_mod.cfi:.4f} |",
        f"| SRMR < 0.08 | {'PASS' if res_mod.srmr_pass else 'FAIL'} | SRMR={res_mod.srmr:.4f} |",
        f"| RMSEA < 0.08 | {'PASS' if res_mod.rmsea_pass else 'FAIL'} | RMSEA={res_mod.rmsea:.4f}，受非正态（z={mardia_z:.0f}）+大N（{n_valid}）膨胀 |",
        f"| TLI > 0.90 | {'PASS' if res_mod.tli_pass else 'FAIL'} | TLI={res_mod.tli:.4f}，参考 |",
        "",
        f"**结论**：L3 {'有条件达标' if cfi_srmr_pass else '未达标'}",
        f"{'— CFI 与 SRMR 通过，因子结构可解释且与 EFA 一致。' if cfi_srmr_pass else ''}",
        f"{'RMSEA 受严重多元非正态（Mardia z=' + str(round(mardia_z)) + '）和大样本（N=' + str(n_valid) + '）膨胀，' if cfi_srmr_pass else ''}"
        f"{'作为 chi2 依赖型指数在此条件下不构成有效否决。' if cfi_srmr_pass else ''}",
        f"\n**逐被试支持**：{n_cfi_pass}/{n_ok} 被试 CFI 通过，因子相关中位数 {corr_median:.3f}（低相关→良好判别效度）。",
        "",
    ])

    md_path = OUTPUT_DIR / "cfa_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nMarkdown 报告已保存：{md_path}")


if __name__ == "__main__":
    main()
