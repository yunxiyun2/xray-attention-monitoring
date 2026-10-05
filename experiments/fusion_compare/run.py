"""E3 三档融合对比实验入口。

对应 spec v2.0 §7.1 E3，回答 RQ3（F1/F2/F3 融合是否优于单模态基线）。
读 multimodal_features.csv（gaze 列在前、face 列在后），对 F1/F2/F3 各跑一次
nested LOSO，产出对比表与 Wilcoxon 显著性检验。

用法（从项目根目录）：
    PYTHONPATH=src conda run -n Anomaly python experiments/fusion_compare/run.py \
        --config configs/multimodal.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    ConfusionMatrixDisplay, roc_curve, auc as sklearn_auc,
)

from xray_attention.data.dataset_builder import DatasetBuilder
from xray_attention.evaluation.nested_loso import evaluate_nested_loso
from xray_attention.evaluation.stats import wilcoxon_paired
from xray_attention.models.fusion_sklearn_wrappers import (
    CrossAttentionFusionWrapper, EarlyFusionWrapper, LateFusionWrapper,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

_META_COLS = {"subject", "label", "state", "difficulty", "window_order"}


def _fusion_specs(gaze_dim: int) -> dict[str, tuple[Callable, dict]]:
    """返回 {fusion_name: (factory, param_grid)}。

    gaze_dim 为数据属性（非超参），在 factory 构造时固定传入。
    """
    return {
        "F1_early": (
            lambda: EarlyFusionWrapper(gaze_dim=gaze_dim, epochs=30, batch_size=32),
            {"hidden": [[16], [32, 16]], "lr": [1e-3]},
        ),
        "F2_late": (
            lambda: LateFusionWrapper(gaze_dim=gaze_dim, epochs=30, batch_size=32),
            {"hidden": [16, 32], "lr": [1e-3]},
        ),
        "F3_cross_attn": (
            lambda: CrossAttentionFusionWrapper(
                gaze_dim=gaze_dim, epochs=30, batch_size=32,
            ),
            {"d_model": [16, 32], "lr": [1e-3]},
        ),
    }


def _load_multimodal(
    multimodal_csv: Path, gaze_csv: Path, face_csv: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, list[str]]:
    """加载多模态特征矩阵，返回 (X, y, groups, gaze_dim, feature_names)。

    若 multimodal_csv 不存在则从 gaze/face 实时构建并写盘。
    """
    if not multimodal_csv.exists():
        print(f"[fusion] 多模态矩阵不存在，从 gaze+face 对齐构建...")
        df = DatasetBuilder.build_multimodal_feature_matrix(gaze_csv, face_csv)
        multimodal_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(multimodal_csv, index=False)
        print(f"[fusion] → {multimodal_csv}  ({len(df)} 行, {len(df.columns)} 列)")
    else:
        df = pd.read_csv(multimodal_csv, dtype={"subject": str})

    feature_cols = [c for c in df.columns if c not in _META_COLS]
    # gaze 特征 = 不以 face_ 开头的特征列
    gaze_cols = [c for c in feature_cols if not c.startswith("face_")]
    face_cols = [c for c in feature_cols if c.startswith("face_")]
    gaze_dim = len(gaze_cols)

    # 保证 X 中 gaze 列在前（build_multimodal_feature_matrix 已保证）
    X = df[gaze_cols + face_cols].values.astype(float)
    y = df["label"].values.astype(int)
    groups = df["subject"].values
    return X, y, groups, gaze_dim, gaze_cols + face_cols


def _plot_auc_boxplot(per_fold_all: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    models = sorted(per_fold_all["model"].unique())
    data = [per_fold_all[per_fold_all["model"] == m]["auc"].values for m in models]
    ax.boxplot(data, labels=models)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="chance")
    ax.set_ylabel("AUC (per LOSO fold)")
    ax.set_title("Fusion AUC distribution by strategy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "fusion_auc_boxplot.png", dpi=120)
    plt.close(fig)


def _plot_roc_pooled(
    pooled_y: np.ndarray, pooled_scores: np.ndarray, name: str, out_dir: Path
) -> float:
    fpr, tpr, _ = roc_curve(pooled_y, pooled_scores)
    roc_auc = float(sklearn_auc(fpr, tpr))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Pooled ROC ({name})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / f"roc_curve_{name}.png", dpi=120)
    plt.close(fig)
    return roc_auc


def _plot_confusion_matrix(
    pooled_y: np.ndarray, pooled_pred: np.ndarray, name: str, out_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        pooled_y, pooled_pred, display_labels=["sleepy", "alert"],
        ax=ax, colorbar=False,
    )
    ax.set_title(f"Confusion matrix ({name})")
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / f"confusion_matrix_{name}.png", dpi=120)
    plt.close(fig)


def run_fusion_compare(
    multimodal_csv: Path,
    gaze_csv: Path,
    face_csv: Path,
    output_dir: Path,
    config_path: Path | None = None,
    inner_cv_folds: int = 3,
    random_state: int = 42,
    gaze_baseline_dir: Path | None = None,
    face_baseline_dir: Path | None = None,
) -> None:
    output_dir = Path(output_dir)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    X, y, groups, gaze_dim, feat_names = _load_multimodal(
        multimodal_csv, gaze_csv, face_csv
    )
    print(
        f"[fusion] 多模态矩阵: X={X.shape}, n_subjects={len(np.unique(groups))}, "
        f"gaze_dim={gaze_dim}, face_dim={X.shape[1] - gaze_dim}"
    )

    specs = _fusion_specs(gaze_dim)
    all_per_fold: list[pd.DataFrame] = []
    summaries: dict = {}
    pooled_by_model: dict[str, np.ndarray] = {}
    per_fold_auc: dict[str, np.ndarray] = {}

    for name, (factory, grid) in specs.items():
        print(f"[fusion] 跑 {name} ...")
        try:
            res = evaluate_nested_loso(
                X=X, y=y, groups=groups,
                model_factory=factory, param_grid=grid,
                inner_cv_folds=inner_cv_folds, random_state=random_state,
            )
        except Exception as e:
            print(f"      [警告] {name} 失败: {e}")
            continue
        folds = res["per_fold"].copy()
        folds["model"] = name
        folds["test_subject"] = folds["test_subject"].astype(int)
        all_per_fold.append(folds)
        summaries[name] = res["summary"]
        pooled_by_model[name] = np.asarray(res["pooled_scores"])
        # per-fold AUC 按 test_subject 数值序排序，保证配对对齐
        per_fold_auc[name] = (
            folds.sort_values("test_subject")["auc"].values
        )
        _plot_roc_pooled(res["pooled_y"], res["pooled_scores"], name, output_dir)
        _plot_confusion_matrix(res["pooled_y"], res["pooled_pred"], name, output_dir)
        print(
            f"      AUC mean={res['summary']['auc_mean']:.3f} "
            f"CI=[{res['summary']['auc_ci_lo']:.3f}, {res['summary']['auc_ci_hi']:.3f}] "
            f"p={res['summary']['wilcoxon_vs_0.5_p']:.4g}"
        )

    if not all_per_fold:
        raise RuntimeError("[fusion] 所有融合策略都失败了")

    metrics_df = pd.concat(all_per_fold, ignore_index=True)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    print(f"[fusion] → {output_dir / 'metrics.csv'}  ({len(metrics_df)} 行)")

    # ---- Wilcoxon 配对检验（per-fold AUC）：F3 vs F1, F3 vs F2 ----
    pairwise = {}
    if "F3_cross_attn" in per_fold_auc and "F1_early" in per_fold_auc:
        p, r = _pairwise_test(
            per_fold_auc["F3_cross_attn"], per_fold_auc["F1_early"],
        )
        pairwise["F3_vs_F1"] = {"wilcoxon_p": p, "rank_biserial_r": r}
    if "F3_cross_attn" in per_fold_auc and "F2_late" in per_fold_auc:
        p, r = _pairwise_test(
            per_fold_auc["F3_cross_attn"], per_fold_auc["F2_late"],
        )
        pairwise["F3_vs_F2"] = {"wilcoxon_p": p, "rank_biserial_r": r}

    # ---- 跨实验对比（spec §8.2）：F3 vs E1_best, F3 vs E2_best ----
    if "F3_cross_attn" in per_fold_auc and gaze_baseline_dir is not None:
        res_e1 = _load_baseline_best_per_fold_auc(
            gaze_baseline_dir / "metrics.csv",
            gaze_baseline_dir / "summary.json",
        )
        if res_e1 is not None:
            e1_model, e1_auc = res_e1
            p, r = _pairwise_test(per_fold_auc["F3_cross_attn"], e1_auc)
            pairwise["F3_vs_E1_best"] = {
                "baseline_model": e1_model,
                "baseline_auc_mean": float(e1_auc.mean()),
                "wilcoxon_p": p, "rank_biserial_r": r,
            }
            print(
                f"[fusion] F3 vs E1({e1_model}) p={p:.4g} r={r:.3f}"
            )
    if "F3_cross_attn" in per_fold_auc and face_baseline_dir is not None:
        res_e2 = _load_baseline_best_per_fold_auc(
            face_baseline_dir / "metrics.csv",
            face_baseline_dir / "summary.json",
        )
        if res_e2 is not None:
            e2_model, e2_auc = res_e2
            p, r = _pairwise_test(per_fold_auc["F3_cross_attn"], e2_auc)
            pairwise["F3_vs_E2_best"] = {
                "baseline_model": e2_model,
                "baseline_auc_mean": float(e2_auc.mean()),
                "wilcoxon_p": p, "rank_biserial_r": r,
            }
            print(
                f"[fusion] F3 vs E2({e2_model}) p={p:.4g} r={r:.3f}"
            )

    summary_out = {
        "n_samples": int(len(y)),
        "n_subjects": int(len(np.unique(groups))),
        "gaze_dim": int(gaze_dim),
        "face_dim": int(X.shape[1] - gaze_dim),
        "feature_names": feat_names,
        "models": summaries,
        "pairwise_tests": pairwise,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_out, f, ensure_ascii=False, indent=2)
    print(f"[fusion] → {output_dir / 'summary.json'}")

    _plot_auc_boxplot(pd.concat(all_per_fold, ignore_index=True), output_dir)
    print(f"[fusion] → {output_dir / 'figures'}/")

    cfg_used = {
        "gaze_dim": int(gaze_dim),
        "face_dim": int(X.shape[1] - gaze_dim),
        "inner_cv_folds": inner_cv_folds,
        "random_state": random_state,
        "models": list(specs.keys()),
        "source_config": str(config_path) if config_path else None,
    }
    with open(output_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_used, f, allow_unicode=True)

    best = max(summaries.items(), key=lambda kv: kv[1]["auc_mean"])
    print(f"[fusion] 完成。best by AUC mean: {best[0]} ({best[1]['auc_mean']:.3f})")


def _pairwise_test(
    auc_a: np.ndarray, auc_b: np.ndarray
) -> tuple[float, float]:
    """检验 A 的 per-fold AUC 是否显著大于 B（单边 Wilcoxon 配对符号秩）。

    对应 spec §8 评估协议：基于 per-fold AUC 配对比较，而非 per-sample 错误。

    返回 (p_value, rank_biserial_r)：
    - p：单边 Wilcoxon H1: AUC_a > AUC_b
    - r：rank-biserial 效应量 ∈ [-1, 1]，正值表示 A 优于 B
    """
    from scipy.stats import rankdata

    auc_a = np.asarray(auc_a, dtype=float)
    auc_b = np.asarray(auc_b, dtype=float)
    # wilcoxon_paired(a, b) 检验 a > b
    p = wilcoxon_paired(auc_a, auc_b)
    diffs = auc_a - auc_b
    n = len(diffs)
    if n == 0:
        return 1.0, 0.0
    ranks = rankdata(np.abs(diffs))
    W_pos = float(np.sum(ranks[diffs > 0]))
    total = n * (n + 1) / 2
    r = (2 * W_pos / total) - 1 if total > 0 else 0.0
    return float(p), float(r)


def _load_baseline_best_per_fold_auc(
    metrics_csv: Path, summary_json: Path
) -> tuple[str, np.ndarray] | None:
    """从单模态基线实验加载最佳模型的 per-fold AUC（按 test_subject 排序）。

    用于跨实验配对比较（spec §8.2：E3 vs E1, E3 vs E2）。
    返回 (best_model_name, auc_array)；若文件缺失或为空返回 None。
    """
    if not metrics_csv.exists() or not summary_json.exists():
        return None
    with open(summary_json, encoding="utf-8") as f:
        s = json.load(f)
    models = s.get("models")
    if not models:
        return None
    best_model = max(models.items(), key=lambda kv: kv[1].get("auc_mean", 0))[0]
    df = pd.read_csv(metrics_csv)
    # test_subject 统一转 int 以保证与 E3 的 per_fold_auc 排序一致（数值序）
    df["test_subject"] = df["test_subject"].astype(int)
    df = df[df["model"] == best_model].sort_values("test_subject")
    if df.empty:
        return None
    return best_model, df["auc"].values.astype(float)


def main() -> None:
    parser = argparse.ArgumentParser(description="E3 融合对比实验")
    parser.add_argument("--config", default="configs/multimodal.yaml")
    parser.add_argument(
        "--features", default="dataset/features/multimodal_features.csv",
        help="多模态特征 CSV（不存在时自动从 gaze+face 构建）",
    )
    parser.add_argument(
        "--output", default="results/multimodal/fusion_compare",
        help="输出目录（相对项目根）",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(open(_PROJECT_ROOT / args.config))
    inner_cv_folds = cfg.get("nested_loso", {}).get("inner_cv_folds", 3)
    random_state = cfg.get("nested_loso", {}).get("random_state", 42)

    feat_dir = _PROJECT_ROOT / "dataset" / "features"
    gaze_baseline_dir = _PROJECT_ROOT / "results" / "multimodal" / "gaze_baseline"
    face_baseline_dir = _PROJECT_ROOT / "results" / "multimodal" / "face_baseline"
    run_fusion_compare(
        multimodal_csv=_PROJECT_ROOT / args.features,
        gaze_csv=feat_dir / "gaze_features.csv",
        face_csv=feat_dir / "face_features.csv",
        output_dir=_PROJECT_ROOT / args.output,
        config_path=_PROJECT_ROOT / args.config,
        inner_cv_folds=inner_cv_folds,
        random_state=random_state,
        gaze_baseline_dir=gaze_baseline_dir,
        face_baseline_dir=face_baseline_dir,
    )


if __name__ == "__main__":
    main()
