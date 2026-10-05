"""E4 难度分层实验入口。

对应 spec v2.0 §7.1 E4，回答 RQ4（任务难度 easy/hard 是否调节识别性能）。
按 easy/hard 分别重跑 E1（gaze-best）/E2（face-best）/E3（F3 best），
产出难度 × 模态 AUC 矩阵 + 调节效应条形图。

用法（从项目根目录）：
    PYTHONPATH=src conda run -n Anomaly python experiments/difficulty_stratified/run.py \
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

from xray_attention.data.dataset_builder import DatasetBuilder
from xray_attention.evaluation.nested_loso import evaluate_nested_loso
from xray_attention.models.classical import make_classical
from xray_attention.models.fusion_sklearn_wrappers import (
    CrossAttentionFusionWrapper,
)
from xray_attention.models.sklearn_wrappers import MLPSklearnWrapper

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

_META_COLS = {"subject", "label", "state", "difficulty", "window_order"}


def _model_specs(gaze_dim: int, face_dim: int) -> dict[str, tuple[Callable, dict, str]]:
    """返回 {name: (factory, param_grid, modality_slice)}。

    modality_slice 取值：
    - "gaze"：仅用注视列（X[:, :gaze_dim]）
    - "face"：仅用面部列（X[:, gaze_dim:]）
    - "all"：用全部列
    """
    return {
        # E1 best: logreg（gaze_baseline summary.json 中 auc_mean=0.902 最高）
        "E1_gaze_logreg": (
            lambda: make_classical("logreg"),
            {"C": [0.1, 1.0, 10.0]},
            "gaze",
        ),
        # E2 best: mlp（face_baseline summary.json 中 auc_mean=0.872 最高）
        "E2_face_mlp": (
            lambda: MLPSklearnWrapper(),
            {"hidden": [[16], [32, 16]], "epochs": [30], "lr": [1e-3]},
            "face",
        ),
        # E3 best: F3 cross-attn（fusion_compare summary.json 中 auc_mean=0.929 最高）
        "E3_F3_cross_attn": (
            lambda: CrossAttentionFusionWrapper(
                gaze_dim=gaze_dim, epochs=30, batch_size=32,
            ),
            {"d_model": [16, 32], "lr": [1e-3]},
            "all",
        ),
    }


def _load_multimodal(
    multimodal_csv: Path, gaze_csv: Path, face_csv: Path
) -> tuple[pd.DataFrame, int]:
    """加载多模态 DataFrame，返回 (df, gaze_dim)。"""
    if not multimodal_csv.exists():
        df = DatasetBuilder.build_multimodal_feature_matrix(gaze_csv, face_csv)
        multimodal_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(multimodal_csv, index=False)
    else:
        df = pd.read_csv(multimodal_csv, dtype={"subject": str})
    feature_cols = [c for c in df.columns if c not in _META_COLS]
    gaze_cols = [c for c in feature_cols if not c.startswith("face_")]
    return df, len(gaze_cols)


def _slice_X(X: np.ndarray, gaze_dim: int, slice_key: str) -> tuple[np.ndarray, int]:
    """按 slice_key 切片 X，返回 (X_slice, gaze_dim_slice)。"""
    if slice_key == "gaze":
        return X[:, :gaze_dim], gaze_dim
    if slice_key == "face":
        return X[:, gaze_dim:], 0
    return X, gaze_dim


def _plot_modulation_barplot(
    auc_matrix: pd.DataFrame, out_dir: Path
) -> None:
    """调节效应条形图：x=难度，分组=模型。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    models = list(auc_matrix.index)
    difficulties = list(auc_matrix.columns)
    x = np.arange(len(difficulties))
    width = 0.8 / len(models)
    for i, m in enumerate(models):
        ax.bar(
            x + i * width - 0.4 + width / 2,
            auc_matrix.loc[m].values,
            width, label=m,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(difficulties)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="chance")
    ax.set_ylabel("AUC (mean over LOSO folds)")
    ax.set_title("Difficulty modulation: AUC by model × difficulty")
    ax.set_ylim(0.4, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "difficulty_modulation_barplot.png", dpi=120)
    plt.close(fig)


def run_difficulty_stratified(
    multimodal_csv: Path,
    gaze_csv: Path,
    face_csv: Path,
    output_dir: Path,
    config_path: Path | None = None,
    inner_cv_folds: int = 3,
    random_state: int = 42,
) -> None:
    output_dir = Path(output_dir)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    df, gaze_dim = _load_multimodal(multimodal_csv, gaze_csv, face_csv)
    feature_cols = [c for c in df.columns if c not in _META_COLS]
    gaze_cols = [c for c in feature_cols if not c.startswith("face_")]
    face_cols = [c for c in feature_cols if c.startswith("face_")]
    print(
        f"[E4] 多模态矩阵: n={len(df)}, n_subjects={df['subject'].nunique()}, "
        f"gaze_dim={len(gaze_cols)}, face_dim={len(face_cols)}"
    )

    specs = _model_specs(len(gaze_cols), len(face_cols))
    difficulties = ["all", "easy", "hard"]

    all_per_fold: list[pd.DataFrame] = []
    auc_matrix: dict[str, dict[str, float]] = {m: {} for m in specs}
    per_fold_by_cell: dict[str, dict[str, np.ndarray]] = {m: {} for m in specs}

    for diff in difficulties:
        if diff == "all":
            sub = df
        else:
            sub = df[df["difficulty"] == diff]
        X_all = sub[gaze_cols + face_cols].values.astype(float)
        y = sub["label"].values.astype(int)
        groups = sub["subject"].values
        print(f"[E4] difficulty={diff}: n={len(sub)}, n_subjects={len(np.unique(groups))}")

        for name, (factory, grid, slice_key) in specs.items():
            X_slice, _ = _slice_X(X_all, len(gaze_cols), slice_key)
            print(f"      跑 {name} on {slice_key} (X={X_slice.shape}) ...")
            try:
                res = evaluate_nested_loso(
                    X=X_slice, y=y, groups=groups,
                    model_factory=factory, param_grid=grid,
                    inner_cv_folds=inner_cv_folds, random_state=random_state,
                )
            except Exception as e:
                print(f"      [警告] {name}/{diff} 失败: {e}")
                auc_matrix[name][diff] = float("nan")
                continue
            folds = res["per_fold"].copy()
            folds["model"] = name
            folds["difficulty"] = diff
            folds["test_subject"] = folds["test_subject"].astype(int)
            all_per_fold.append(folds)
            auc_mean = res["summary"]["auc_mean"]
            auc_matrix[name][diff] = auc_mean
            per_fold_by_cell[name][diff] = (
                folds.sort_values("test_subject")["auc"].values
            )
            print(
                f"        AUC mean={auc_mean:.3f} "
                f"CI=[{res['summary']['auc_ci_lo']:.3f}, "
                f"{res['summary']['auc_ci_hi']:.3f}]"
            )

    # ---- 输出 AUC 矩阵 ----
    auc_df = pd.DataFrame(auc_matrix).T  # rows=model, cols=difficulty
    auc_df.to_csv(output_dir / "difficulty_auc_matrix.csv")
    print(f"[E4] → {output_dir / 'difficulty_auc_matrix.csv'}")

    metrics_df = pd.concat(all_per_fold, ignore_index=True)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    print(f"[E4] → {output_dir / 'metrics.csv'}  ({len(metrics_df)} 行)")

    _plot_modulation_barplot(auc_df, output_dir)

    # 难度调节效应：hard 相对 easy 的 AUC 下降
    modulation = {}
    for m in specs:
        e = auc_matrix[m].get("easy", float("nan"))
        h = auc_matrix[m].get("hard", float("nan"))
        modulation[m] = {
            "auc_easy": float(e), "auc_hard": float(h),
            "delta_easy_minus_hard": float(e - h) if not np.isnan(e) and not np.isnan(h) else None,
        }

    summary_out = {
        "n_samples_total": int(len(df)),
        "n_subjects": int(df["subject"].nunique()),
        "difficulties": difficulties,
        "models": list(specs.keys()),
        "auc_matrix": {m: {d: v for d, v in row.items()} for m, row in auc_matrix.items()},
        "modulation_effect": modulation,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_out, f, ensure_ascii=False, indent=2)
    print(f"[E4] → {output_dir / 'summary.json'}")
    print(f"[E4] → {output_dir / 'figures'}/")

    cfg_used = {
        "inner_cv_folds": inner_cv_folds,
        "random_state": random_state,
        "models": list(specs.keys()),
        "difficulties": difficulties,
        "source_config": str(config_path) if config_path else None,
    }
    with open(output_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_used, f, allow_unicode=True)

    print("[E4] 完成。")


def main() -> None:
    parser = argparse.ArgumentParser(description="E4 难度分层实验")
    parser.add_argument("--config", default="configs/multimodal.yaml")
    parser.add_argument(
        "--features", default="dataset/features/multimodal_features.csv",
    )
    parser.add_argument(
        "--output", default="results/multimodal/difficulty_stratified",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(open(_PROJECT_ROOT / args.config))
    inner_cv_folds = cfg.get("nested_loso", {}).get("inner_cv_folds", 3)
    random_state = cfg.get("nested_loso", {}).get("random_state", 42)

    feat_dir = _PROJECT_ROOT / "dataset" / "features"
    run_difficulty_stratified(
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
