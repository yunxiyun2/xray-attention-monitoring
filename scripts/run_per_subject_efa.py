# -*- coding: utf-8 -*-
"""逐被试 EFA 分析：对每名被试的窗口单独做 2 因子固定解 EFA。

用法（从项目根目录运行）：
    conda run -n Anomaly python scripts/run_per_subject_efa.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from dual_factor.analysis.sensitivity import per_subject_efa

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "results" / "dual_factor_efa" / "indicator_matrix.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "dual_factor_efa"

# 6 指标（呆滞簇 FD/PSV/SA/FN + 发散簇 SE/GTE）
INDICATORS_6 = ["FD", "PSV", "SA", "FN", "SE", "GTE"]


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    print(f"输入数据：{df.shape[0]} 行，{df.shape[1]} 列")
    print(f"被试数：{df['subject'].nunique()}")

    # 逐被试 EFA
    result = per_subject_efa(
        df, subject_col="subject", indicator_cols=INDICATORS_6,
        n_factors=2, suffix="_z",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 保存 CSV 汇总表
    csv_path = OUTPUT_DIR / "per_subject_efa.csv"
    result.to_csv(csv_path, index=False)
    print(f"\nCSV 已保存：{csv_path}")
    print(result.to_string(index=False))

    # 生成 Markdown 报告
    ok = result[result["status"] == "ok"]
    n_ok = len(ok)
    n_total = len(result)
    n_direction = int(ok["direction_consistent"].sum()) if n_ok > 0 else 0
    cross_median = float(ok["cross_loading_max"].median()) if n_ok > 0 else float("nan")
    corr_vals = ok["factor_corr"].dropna()
    corr_median = float(corr_vals.median()) if len(corr_vals) > 0 else float("nan")
    var_median = float(ok["variance_explained"].median()) if n_ok > 0 else float("nan")

    lines = [
        "# 逐被试 EFA 分析报告\n",
        f"**数据来源**：`results/dual_factor_efa/indicator_matrix.csv`\n",
        f"**指标集**：{', '.join(INDICATORS_6)}（FD/PSV/SA/FN 呆滞簇 + SE/GTE 发散簇）\n",
        f"**方法**：每名被试窗口级数据单独做 2 因子固定解 EFA（主轴因子法 + Promax 旋转）\n",
        f"**被试总数**：{n_total}，成功：{n_ok}，跳过/错误：{n_total - n_ok}\n",
        "",
        "## 汇总统计\n",
        "| 统计量 | 值 |",
        "|---|---|",
        f"| 方向一致被试数 | {n_direction}/{n_ok} |",
        f"| 交叉载荷中位数 | {cross_median:.3f} |",
        f"| 因子相关中位数 | {corr_median:.3f} |",
        f"| 累计方差解释率中位数 | {var_median:.3f} |",
        "",
        "## 逐被试明细\n",
        "| 被试 | 窗口数 | 因子数 | 方向一致 | 交叉载荷最大值 | 因子相关 | 累计方差 | 状态 |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for _, row in result.iterrows():
        dc = row["direction_consistent"]
        dc_str = "✅" if dc is True else ("❌" if dc is False else "—")
        cl = f"{row['cross_loading_max']:.3f}" if pd.notna(row["cross_loading_max"]) else "—"
        fc = f"{row['factor_corr']:.3f}" if pd.notna(row["factor_corr"]) else "—"
        ve = f"{row['variance_explained']:.3f}" if pd.notna(row["variance_explained"]) else "—"
        nf = str(row["n_factors"]) if pd.notna(row["n_factors"]) else "—"
        lines.append(
            f"| {row['subject_id']} | {row['n_windows']} | {nf} | {dc_str} "
            f"| {cl} | {fc} | {ve} | {row['status']} |"
        )

    lines.append("")

    md_path = OUTPUT_DIR / "per_subject_efa.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nMarkdown 报告已保存：{md_path}")


if __name__ == "__main__":
    main()
