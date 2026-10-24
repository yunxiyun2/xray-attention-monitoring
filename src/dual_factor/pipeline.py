"""一键运行：DROZY → 预处理 → 8 指标 → 窗口化 → EFA → 敏感性分析 → 构念效度 → 决策报告。

集成 SPEC v2.0 全部要求：
  - 窗口非独立性敏感性分析（被试级聚合 EFA / LOSO-EFA / 非重叠窗口 EFA）
  - 2因子 vs 3因子强制对比
  - 构念效度论证（EOG 代理效度分类 + 幅度归一化校正）
  - 严格判据 C1-C5 执行与三选一决策

用法（从项目根目录）：
    conda run -n Anomaly python -m dual_factor.pipeline --config configs/dual_factor_efa.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from dual_factor.data.drozy_loader import load_all_records
from dual_factor.data.preprocess import preprocess_eog
from dual_factor.features.windowing import build_indicator_matrix, within_subject_zscore
from dual_factor.analysis.efa import precheck_efa, run_efa, run_efa_compare
from dual_factor.analysis.descriptive import (
    descriptive_stats, directionality_test, indicator_distribution_report, INDICATORS,
)
from dual_factor.analysis.sensitivity import (
    subject_level_efa, loso_efa, nonoverlap_efa, sensitivity_summary,
)
from dual_factor.analysis.construct_validity import (
    proxy_validity_report, amplitude_normalization,
    direction_correction_report, construct_validity_summary,
)
from dual_factor.analysis.plots import (
    plot_scree_parallel, plot_loading_matrix, plot_factor_scatter, plot_direction_check,
)
from dual_factor.analysis.decision import make_decision

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_pipeline(config: dict) -> None:
    """端到端执行 EFA 实验。"""
    results_dir = Path(config["output"]["results_dir"])
    figures_dir = Path(config["output"]["figures_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 保存配置快照
    with open(results_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True)

    # 1. 数据读取
    print("[1/10] Loading DROZY records...")
    records = load_all_records(
        config["data"]["drozy_root"], config["data"]["channels"]
    )
    print(f"  Loaded {len(records)} records")

    # 2. 预处理 + 窗口化 + 指标计算
    print("[2/10] Preprocessing + windowing + indicators...")
    frames = []
    pp = config["preprocess"]
    ed = config["event_detection"]
    wn = config["windowing"]
    bins = config["spatial_grid"]["bins"]
    for rec in records:
        h_clean, v_clean = preprocess_eog(
            rec.eog_h, rec.eog_v, rec.sampling_rate,
            highcut=pp["lowpass"], lowcut=pp["highpass"],
            filter_order=pp["filter_order"],
            hampel_window=pp["hampel_window"], hampel_threshold=pp["hampel_threshold"],
        )
        df = build_indicator_matrix(
            subject=rec.subject, test=rec.test, kss=rec.kss,
            eog_h=h_clean, eog_v=v_clean, fs=rec.sampling_rate,
            window_sec=wn["window_sec"], stride_sec=wn["stride_sec"], bins=bins,
            min_fixations=wn["min_fixations"], min_saccades=wn["min_saccades"],
            velocity_threshold=ed["velocity_threshold"],
            velocity_percentile=ed["velocity_percentile"],
            min_fixation_ms=ed["min_fixation_ms"], min_saccade_ms=ed["min_saccade_ms"],
        )
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)

    # 标准化
    z_cols = [f"{c}_z" for c in INDICATORS]
    full = within_subject_zscore(full, INDICATORS)

    # 输出指标矩阵
    full.to_csv(results_dir / "indicator_matrix.csv", index=False, encoding="utf-8")
    print(f"  Indicator matrix: {len(full)} rows, missing rate="
          f"{full['valid'].eq(False).mean():.2%}")

    # 3. 描述性统计 + 方向性检验 + 指标分布质量报告
    print("[3/10] Descriptive stats + directionality test + distribution report...")
    valid_df = full[full["valid"]].copy()
    desc = descriptive_stats(
        valid_df, INDICATORS,
        alert_max=config["labels"]["alert_threshold"],
        drowsy_min=config["labels"]["drowsy_threshold"],
    )
    desc.to_csv(results_dir / "descriptive_stats.csv", index=False, encoding="utf-8")

    # 指标分布质量报告（SPEC v2.0 Task 3：均值±SD、中位数IQR、缺失率、Shapiro-Wilk正态性）
    dist_report = indicator_distribution_report(
        valid_df, INDICATORS,
        alert_max=config["labels"]["alert_threshold"],
        drowsy_min=config["labels"]["drowsy_threshold"],
    )
    dist_report.to_csv(results_dir / "indicator_distribution_report.csv",
                       index=False, encoding="utf-8")

    # 4. EFA 前置检验
    print("[4/10] KMO + Bartlett pre-check...")
    efa_input = valid_df[z_cols].dropna()
    efa_input.columns = INDICATORS  # 重命名列去掉 _z 后缀
    precheck = precheck_efa(efa_input)
    with open(results_dir / "kmo_bartlett.json", "w", encoding="utf-8") as f:
        json.dump(precheck, f, ensure_ascii=False, indent=2)
    print(f"  KMO={precheck['kmo_overall']:.3f}, Bartlett p={precheck['bartlett_p']:.2e}")

    # 5. EFA 因子提取
    print("[5/10] EFA extraction + parallel analysis...")
    efa_cfg = config["efa"]
    result = run_efa(
        efa_input, extraction=efa_cfg["extraction"],
        rotation=efa_cfg["rotation"], max_factors=efa_cfg["max_factors"],
        variance_threshold=efa_cfg["variance_threshold"],
        n_iter=efa_cfg.get("parallel_analysis_n_iter", 1000),
        percentile=efa_cfg.get("parallel_analysis_percentile", 95),
    )
    efa_json = {
        "n_factors": result.n_factors,
        "variance_explained": result.variance_explained,
        "loadings": result.loadings.tolist(),
        "variable_names": result.variable_names,
        "factor_corr": result.factor_corr.tolist() if result.factor_corr is not None else None,
        "parallel_analysis": result.parallel_analysis,
    }
    with open(results_dir / "efa_results.json", "w", encoding="utf-8") as f:
        json.dump(efa_json, f, ensure_ascii=False, indent=2)
    print(f"  Factors={result.n_factors}, Var={result.variance_explained:.3f}")

    # 6. 窗口非独立性敏感性分析（SPEC v2.0 Task 4）
    print("[6/10] Sensitivity analysis (subject-level / LOSO / non-overlap)...")
    wn = config["windowing"]
    subj_efa = subject_level_efa(
        valid_df, subject_col="subject", indicator_cols=INDICATORS,
        suffix="_z", extraction=efa_cfg["extraction"], rotation=efa_cfg["rotation"],
        max_factors=efa_cfg["max_factors"],
        variance_threshold=efa_cfg["variance_threshold"],
    )
    with open(results_dir / "sensitivity_subject_efa.json", "w", encoding="utf-8") as f:
        json.dump(subj_efa, f, ensure_ascii=False, indent=2, default=str)

    loso_df = loso_efa(
        valid_df, subject_col="subject", indicator_cols=INDICATORS,
        suffix="_z", extraction=efa_cfg["extraction"], rotation=efa_cfg["rotation"],
        max_factors=efa_cfg["max_factors"],
        variance_threshold=efa_cfg["variance_threshold"],
    )
    loso_df.to_csv(results_dir / "sensitivity_loso.csv", index=False, encoding="utf-8")

    nonoverlap = nonoverlap_efa(
        valid_df, subject_col="subject", indicator_cols=INDICATORS,
        window_col="window_idx", window_sec=wn["window_sec"], stride_sec=wn["stride_sec"],
        suffix="_z", extraction=efa_cfg["extraction"], rotation=efa_cfg["rotation"],
        max_factors=efa_cfg["max_factors"],
        variance_threshold=efa_cfg["variance_threshold"],
    )
    with open(results_dir / "sensitivity_nonoverlap.json", "w", encoding="utf-8") as f:
        json.dump(nonoverlap, f, ensure_ascii=False, indent=2, default=str)

    sens = sensitivity_summary(
        window_efa_result={"n_factors": result.n_factors},
        subject_efa_result=subj_efa,
        loso_df=loso_df,
        nonoverlap_efa_result=nonoverlap,
    )
    with open(results_dir / "sensitivity_summary.json", "w", encoding="utf-8") as f:
        json.dump(sens, f, ensure_ascii=False, indent=2, default=str)
    print(f"  Sensitivity consistent={sens['consistent']}: {sens['conclusion']}")

    # 7. 2因子 vs 3因子强制对比（SPEC v2.0 Task 5）
    print("[7/10] 2-factor vs 3-factor comparison...")
    compare = run_efa_compare(
        efa_input, n_factors_list=[2, 3],
        extraction=efa_cfg["extraction"], rotation=efa_cfg["rotation"],
        n_iter=efa_cfg.get("parallel_analysis_n_iter", 1000),
        percentile=efa_cfg.get("parallel_analysis_percentile", 95),
        max_factors=efa_cfg["max_factors"],
    )
    with open(results_dir / "efa_compare_2v3.json", "w", encoding="utf-8") as f:
        json.dump(compare, f, ensure_ascii=False, indent=2, default=str)
    for key, comp in compare["comparisons"].items():
        if "status" not in comp:
            print(f"  {key}: var={comp['variance_explained']:.3f}, "
                  f"cross_max={comp['cross_loading_max']:.3f}, "
                  f"corr={comp['factor_corr']}")

    # 8. 构念效度论证（SPEC v2.0 Task 7）
    print("[8/10] Construct validity report...")
    proxy = proxy_validity_report(valid_df, INDICATORS)
    proxy.to_csv(results_dir / "proxy_validity.csv", index=False, encoding="utf-8")

    # 幅度归一化校正
    full_norm = amplitude_normalization(valid_df, subject_col="subject", indicator_cols=INDICATORS)
    dir_correction = direction_correction_report(
        valid_df, full_norm, INDICATORS,
        alert_max=config["labels"]["alert_threshold"],
        drowsy_min=config["labels"]["drowsy_threshold"],
    )
    dir_correction.to_csv(results_dir / "direction_correction.csv",
                          index=False, encoding="utf-8")

    cv_summary = construct_validity_summary(valid_df, INDICATORS)
    with open(results_dir / "construct_validity.md", "w", encoding="utf-8") as f:
        f.write(cv_summary)

    # 9. 可视化
    print("[9/10] Generating figures...")
    plot_scree_parallel(result, figures_dir / "scree_parallel.png")
    plot_loading_matrix(result, figures_dir / "loading_matrix.png")
    plot_factor_scatter(result, figures_dir / "factor_scatter.png")
    plot_direction_check(valid_df, INDICATORS, figures_dir / "direction_check.png")

    # 10. 严格判据决策报告（SPEC v2.0 Task 6：C1-C5 逐项检查 + 三选一决策）
    print("[10/10] Writing decision report (strict criteria C1-C5)...")
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
    subject_consistent = sens.get("consistent", None)
    report = make_decision(
        n_factors=result.n_factors, factor_corr=corr_val,
        max_loading=max_loading, cross_loading_max=cross_max,
        variance_explained=result.variance_explained,
        subject_level_consistent=subject_consistent,
    )
    with open(results_dir / "decision_report.md", "w", encoding="utf-8") as f:
        f.write(report.to_markdown())
    print(f"  Verdict: {report.verdict_label}")
    print(f"Done. Results at {results_dir}")


def main():
    parser = argparse.ArgumentParser(description="DROZY 双因子 EFA 一键运行")
    parser.add_argument("--config", default="configs/dual_factor_efa.yaml")
    args = parser.parse_args()
    cfg_path = PROJECT_ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    with open(cfg_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    run_pipeline(config)


if __name__ == "__main__":
    main()
