"""单模态基线实验共享运行器（E1 注视 / E2 面部通用）。

对应 spec v2.0 §7.1 E1/E2、§8 评估协议。
读取特征 CSV，对 LR/SVM/RF/MLP/LSTM 各跑一次 nested LOSO，
产出 metrics.csv / summary.json / config_used.yaml / figures/。

用法（被 gaze_baseline/run.py 与 face_baseline/run.py 复用）：
    from experiments._unimodal_baseline import run_unimodal_baseline
    run_unimodal_baseline(feature_csv=..., output_dir=..., modality="gaze")
"""
from __future__ import annotations

import json
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

from xray_attention.evaluation.nested_loso import evaluate_nested_loso
from xray_attention.models.classical import make_classical
from xray_attention.models.sklearn_wrappers import (
    MLPSklearnWrapper, LSTMSklearnWrapper,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 元数据列（不作为特征）
_META_COLS = {"subject", "label", "state", "difficulty", "window_idx"}


def _model_specs() -> dict[str, tuple[Callable, dict]]:
    """返回 {model_name: (factory, param_grid)}。"""
    return {
        "logreg": (
            lambda: make_classical("logreg"),
            {"C": [0.1, 1.0, 10.0]},
        ),
        "svm": (
            lambda: make_classical("svm"),
            {"C": [0.1, 1.0, 10.0], "kernel": ["rbf"]},
        ),
        "rf": (
            lambda: make_classical("rf"),
            {"n_estimators": [100, 200], "max_depth": [6, 8, None]},
        ),
        "mlp": (
            lambda: MLPSklearnWrapper(),
            {"hidden": [[16], [32, 16]], "epochs": [30], "lr": [1e-3]},
        ),
        "lstm": (
            lambda: LSTMSklearnWrapper(),
            {"hidden": [16, 32], "epochs": [30]},
        ),
    }


def _load_features(feature_csv: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """加载特征矩阵，返回 (X, y, groups, feature_names)。"""
    df = pd.read_csv(feature_csv)
    feature_cols = [c for c in df.columns if c not in _META_COLS]
    X = df[feature_cols].values.astype(float)
    y = df["label"].values.astype(int)
    groups = df["subject"].values
    return X, y, groups, feature_cols


def _plot_per_subject_auc(per_fold_all: pd.DataFrame, out_dir: Path) -> None:
    """每模型每折 AUC 箱线图。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    models = sorted(per_fold_all["model"].unique())
    data = [per_fold_all[per_fold_all["model"] == m]["auc"].values for m in models]
    ax.boxplot(data, labels=models)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="chance")
    ax.set_ylabel("AUC (per LOSO fold)")
    ax.set_title("Per-subject AUC distribution by model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "per_subject_auc_boxplot.png", dpi=120)
    plt.close(fig)


def _plot_roc_curve_pooled(
    pooled_y: np.ndarray, pooled_scores: np.ndarray, model_name: str, out_dir: Path
) -> float:
    """池化 ROC 曲线，返回 AUC。"""
    fpr, tpr, _ = roc_curve(pooled_y, pooled_scores)
    roc_auc = float(sklearn_auc(fpr, tpr))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Pooled ROC curve ({model_name})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / f"roc_curve_pooled_{model_name}.png", dpi=120)
    plt.close(fig)
    return roc_auc


def _plot_confusion_matrix(
    pooled_y: np.ndarray, pooled_pred: np.ndarray, model_name: str, out_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        pooled_y, pooled_pred, display_labels=["sleepy", "alert"],
        ax=ax, colorbar=False,
    )
    ax.set_title(f"Confusion matrix ({model_name})")
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / f"confusion_matrix_{model_name}.png", dpi=120)
    plt.close(fig)


def run_unimodal_baseline(
    feature_csv: Path,
    output_dir: Path,
    modality: str,
    config_path: Path | None = None,
    inner_cv_folds: int = 3,
    random_state: int = 42,
) -> None:
    """跑单模态 nested LOSO 基线。

    Parameters
    ----------
    feature_csv : 特征矩阵 CSV 路径
    output_dir : 输出目录（results/multimodal/<modality>_baseline/）
    modality : "gaze" | "face"
    """
    output_dir = Path(output_dir)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    X, y, groups, feat_names = _load_features(feature_csv)
    print(f"[{modality}] 特征矩阵: X={X.shape}, n_subjects={len(np.unique(groups))}, "
          f"n_features={len(feat_names)}")

    specs = _model_specs()
    all_per_fold: list[pd.DataFrame] = []
    summaries: dict = {}
    pooled_best: tuple[str, np.ndarray, np.ndarray, np.ndarray] | None = None
    best_auc_mean = -1.0

    for name, (factory, grid) in specs.items():
        print(f"[{modality}] 跑 {name} ...")
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
        all_per_fold.append(folds)
        summaries[name] = res["summary"]
        # 收集 best 模型的池化预测用于 ROC/CM
        if res["summary"]["auc_mean"] > best_auc_mean:
            best_auc_mean = res["summary"]["auc_mean"]
            pooled_best = (name, res["pooled_y"], res["pooled_scores"], res["pooled_pred"])
        # 也为每个模型画 ROC + CM
        _plot_roc_curve_pooled(res["pooled_y"], res["pooled_scores"], name, output_dir)
        _plot_confusion_matrix(res["pooled_y"], res["pooled_pred"], name, output_dir)
        print(f"      AUC mean={res['summary']['auc_mean']:.3f} "
              f"CI=[{res['summary']['auc_ci_lo']:.3f}, {res['summary']['auc_ci_hi']:.3f}] "
              f"p={res['summary']['wilcoxon_vs_0.5_p']:.4g}")

    if not all_per_fold:
        raise RuntimeError(f"[{modality}] 所有模型都失败了")

    metrics_df = pd.concat(all_per_fold, ignore_index=True)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    print(f"[{modality}] → {output_dir / 'metrics.csv'}  ({len(metrics_df)} 行)")

    # summary.json：加 feature_names 与 modality 元信息
    summary_out = {
        "modality": modality,
        "n_samples": int(len(y)),
        "n_subjects": int(len(np.unique(groups))),
        "n_features": int(len(feat_names)),
        "feature_names": feat_names,
        "models": summaries,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_out, f, ensure_ascii=False, indent=2)
    print(f"[{modality}] → {output_dir / 'summary.json'}")

    # 汇总箱线图（所有模型一图）
    _plot_per_subject_auc(pd.concat(all_per_fold, ignore_index=True), output_dir)
    print(f"[{modality}] → {output_dir / 'figures'}/")

    # config_used.yaml
    cfg_used = {
        "modality": modality,
        "feature_csv": str(feature_csv),
        "inner_cv_folds": inner_cv_folds,
        "random_state": random_state,
        "models": list(specs.keys()),
        "source_config": str(config_path) if config_path else None,
    }
    with open(output_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_used, f, allow_unicode=True)

    print(f"[{modality}] 完成。best model by AUC mean: "
          f"{pooled_best[0] if pooled_best else 'N/A'} ({best_auc_mean:.3f})")
