"""E6 最终报告：best 模型全特征性能汇总。

对应 spec v2.0 §7.1 E6、§8.3 次要指标、§8.4 报告模板。
对 F3（cross-attention 融合，E3 中 AUC 最高）跑 nested LOSO，
产出 confusion_matrix / calibration_curve / per_subject_auc / roc_curve
及 summary.json + metrics.csv。

用法（从项目根目录）：
    PYTHONPATH=src conda run -n Anomaly python experiments/final_report/run.py \
        --config configs/multimodal.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay, roc_curve, auc as sklearn_auc,
)

from xray_attention.data.dataset_builder import DatasetBuilder
from xray_attention.evaluation.nested_loso import evaluate_nested_loso
from xray_attention.models.fusion_sklearn_wrappers import (
    CrossAttentionFusionWrapper,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

_META_COLS = {"subject", "label", "state", "difficulty", "window_order"}


def _load_multimodal(
    multimodal_csv: Path, gaze_csv: Path, face_csv: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, list[str]]:
    """加载多模态特征矩阵，返回 (X, y, groups, gaze_dim, feature_names)。"""
    if not multimodal_csv.exists():
        print("[final] 多模态矩阵不存在，从 gaze+face 对齐构建...")
        df = DatasetBuilder.build_multimodal_feature_matrix(gaze_csv, face_csv)
        multimodal_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(multimodal_csv, index=False)
    else:
        df = pd.read_csv(multimodal_csv, dtype={"subject": str})
    feature_cols = [c for c in df.columns if c not in _META_COLS]
    gaze_cols = [c for c in feature_cols if not c.startswith("face_")]
    face_cols = [c for c in feature_cols if c.startswith("face_")]
    gaze_dim = len(gaze_cols)
    X = df[gaze_cols + face_cols].values.astype(float)
    y = df["label"].values.astype(int)
    groups = df["subject"].values
    return X, y, groups, gaze_dim, gaze_cols + face_cols


def _plot_confusion_matrix(
    pooled_y: np.ndarray, pooled_pred: np.ndarray, out_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        pooled_y, pooled_pred,
        display_labels=["sleepy", "alert"],
        ax=ax, colorbar=False,
    )
    ax.set_title("Confusion matrix (F3, pooled LOSO)")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=120)
    plt.close(fig)


def _plot_calibration_curve(
    pooled_y: np.ndarray, pooled_scores: np.ndarray, out_dir: Path
) -> None:
    prob_true, prob_pred = calibration_curve(
        pooled_y, pooled_scores, n_bins=10, strategy="uniform",
    )
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(prob_pred, prob_true, "s-", label="F3 calibration")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration curve (F3, pooled LOSO)")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_dir / "calibration_curve.png", dpi=120)
    plt.close(fig)


def _plot_per_subject_auc(per_fold: pd.DataFrame, out_dir: Path) -> None:
    df = per_fold.copy()
    df["test_subject"] = df["test_subject"].astype(int)
    df = df.sort_values("test_subject")
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#e74c3c" if a < 0.5 else "#2ecc71" for a in df["auc"]]
    ax.bar(range(len(df)), df["auc"], color=colors)
    ax.axhline(df["auc"].mean(), color="navy", linestyle="--", linewidth=1,
               label=f"mean={df['auc'].mean():.3f}")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, label="chance")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([f"s{int(s):02d}" for s in df["test_subject"]],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("AUC")
    ax.set_title("Per-subject AUC (F3, LOSO)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "per_subject_auc.png", dpi=120)
    plt.close(fig)


def _plot_roc_curve(
    pooled_y: np.ndarray, pooled_scores: np.ndarray, out_dir: Path
) -> float:
    fpr, tpr, _ = roc_curve(pooled_y, pooled_scores)
    roc_auc = float(sklearn_auc(fpr, tpr))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"F3 (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Pooled ROC (F3)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curve.png", dpi=120)
    plt.close(fig)
    return roc_auc


def run_final_report(
    multimodal_csv: Path,
    gaze_csv: Path,
    face_csv: Path,
    output_dir: Path,
    config_path: Path | None = None,
    inner_cv_folds: int = 3,
    random_state: int = 42,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, groups, gaze_dim, feat_names = _load_multimodal(
        multimodal_csv, gaze_csv, face_csv
    )
    print(
        f"[final] 多模态矩阵: X={X.shape}, "
        f"n_subjects={len(np.unique(groups))}, "
        f"gaze_dim={gaze_dim}, face_dim={X.shape[1] - gaze_dim}"
    )

    # F3 best 模型（E3 中 AUC 最高）
    factory = lambda: CrossAttentionFusionWrapper(
        gaze_dim=gaze_dim, epochs=30, batch_size=32,
        random_state=random_state,
    )
    grid = {"d_model": [16, 32], "lr": [1e-3]}

    print("[final] 跑 F3 cross-attention nested LOSO ...")
    res = evaluate_nested_loso(
        X=X, y=y, groups=groups,
        model_factory=factory, param_grid=grid,
        inner_cv_folds=inner_cv_folds, random_state=random_state,
    )
    summary = res["summary"]
    per_fold = res["per_fold"]
    pooled_y = res["pooled_y"]
    pooled_scores = res["pooled_scores"]
    pooled_pred = res["pooled_pred"]

    print(
        f"[final] AUC mean={summary['auc_mean']:.3f} "
        f"CI=[{summary['auc_ci_lo']:.3f}, {summary['auc_ci_hi']:.3f}] "
        f"pooled={summary['auc_pooled']:.3f} "
        f"Brier={summary['brier_score']:.4f}"
    )

    # 指标表
    per_fold_out = per_fold.copy()
    per_fold_out["test_subject"] = per_fold_out["test_subject"].astype(int)
    per_fold_out = per_fold_out.sort_values("test_subject")
    per_fold_out.to_csv(output_dir / "metrics.csv", index=False)

    # 图表
    _plot_confusion_matrix(pooled_y, pooled_pred, output_dir)
    _plot_calibration_curve(pooled_y, pooled_scores, output_dir)
    _plot_per_subject_auc(per_fold, output_dir)
    roc_auc = _plot_roc_curve(pooled_y, pooled_scores, output_dir)

    summary_out = {
        "model": "F3_cross_attn",
        "n_samples": int(len(y)),
        "n_subjects": int(len(np.unique(groups))),
        "gaze_dim": int(gaze_dim),
        "face_dim": int(X.shape[1] - gaze_dim),
        "feature_names": feat_names,
        "auc_mean": summary["auc_mean"],
        "auc_std": summary["auc_std"],
        "auc_ci_lo": summary["auc_ci_lo"],
        "auc_ci_hi": summary["auc_ci_hi"],
        "auc_pooled": summary["auc_pooled"],
        "roc_auc_pooled": roc_auc,
        "brier_score": summary["brier_score"],
        "balacc_mean": summary["balacc_mean"],
        "f1_mean": summary["f1_mean"],
        "wilcoxon_vs_0.5_p": summary["wilcoxon_vs_0.5_p"],
        "rank_biserial_r": summary["rank_biserial_r"],
        "n_folds": summary["n_folds"],
        "n_positive_auc": summary["n_positive_auc"],
        "per_subject_auc": {
            str(int(r["test_subject"])): float(r["auc"])
            for _, r in per_fold_out.iterrows()
        },
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_out, f, ensure_ascii=False, indent=2)
    print(f"[final] → {output_dir / 'summary.json'}")
    print(f"[final] → {output_dir}/confusion_matrix.png")
    print(f"[final] → {output_dir}/calibration_curve.png")
    print(f"[final] → {output_dir}/per_subject_auc.png")
    print(f"[final] → {output_dir}/roc_curve.png")
    print(f"[final] 完成。")


def main() -> None:
    parser = argparse.ArgumentParser(description="E6 最终报告")
    parser.add_argument("--config", default="configs/multimodal.yaml")
    parser.add_argument(
        "--features", default="dataset/features/multimodal_features.csv",
        help="多模态特征 CSV",
    )
    parser.add_argument(
        "--output", default="results/multimodal/final_report",
        help="输出目录（相对项目根）",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(open(_PROJECT_ROOT / args.config))
    inner_cv_folds = cfg.get("nested_loso", {}).get("inner_cv_folds", 3)
    random_state = cfg.get("nested_loso", {}).get("random_state", 42)

    feat_dir = _PROJECT_ROOT / "dataset" / "features"
    run_final_report(
        multimodal_csv=_PROJECT_ROOT / args.features,
        gaze_csv=feat_dir / "gaze_features.csv",
        face_csv=feat_dir / "face_features.csv",
        output_dir=_PROJECT_ROOT / args.output,
        config_path=_PROJECT_ROOT / args.config,
        inner_cv_folds=inner_cv_folds,
        random_state=random_state,
    )


if __name__ == "__main__":
    main()
