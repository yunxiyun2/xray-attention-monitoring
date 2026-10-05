"""E5 特征消融实验入口。

对应 spec v2.0 §7.1 E5、§9.1，回答 RQ5（哪类特征贡献最大）。
对 F3（cross-attention best）逐类剔除特征（gaze / eye / mouth），
记录 AUC 下降，出条形图。

特征分组（基于 multimodal_features.csv）：
- gaze（10 列）：mean_error 等
- eye（9 列）：face_ear_*, face_perclos_*, face_blink_*, face_long_closure_*
- mouth（3 列）：face_mar_*, face_yawn_count

用法（从项目根目录）：
    PYTHONPATH=src conda run -n Anomaly python experiments/ablation/run.py \
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
from xray_attention.models.fusion_sklearn_wrappers import (
    CrossAttentionFusionWrapper,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

_META_COLS = {"subject", "label", "state", "difficulty", "window_order"}


def _feature_groups(feature_cols: list[str]) -> dict[str, list[str]]:
    """返回 {group_name: [feature_cols]}。

    - gaze：不以 face_ 开头的特征列
    - eye：face_ear_* / face_perclos_* / face_blink_* / face_long_closure_*
    - mouth：face_mar_* / face_yawn_count
    """
    gaze = [c for c in feature_cols if not c.startswith("face_")]
    eye = [
        c for c in feature_cols
        if c.startswith("face_") and (
            c.startswith("face_ear") or c.startswith("face_perclos")
            or c.startswith("face_blink") or c.startswith("face_long_closure")
        )
    ]
    mouth = [
        c for c in feature_cols
        if c.startswith("face_") and (
            c.startswith("face_mar") or c.startswith("face_yawn")
        )
    ]
    # 校验覆盖
    covered = set(gaze) | set(eye) | set(mouth)
    uncovered = [c for c in feature_cols if c not in covered]
    if uncovered:
        print(f"[E5] 警告：未分组特征 {uncovered}（归入「other」，不剔除）")
    return {"gaze": gaze, "eye": eye, "mouth": mouth}


def _ablation_conditions(
    feature_cols: list[str]
) -> dict[str, list[str]]:
    """返回 {condition_name: kept_feature_cols}。

    条件：
    - all：保留全部（baseline）
    - drop_gaze：剔除注视列
    - drop_eye：剔除眼部列
    - drop_mouth：剔除嘴部列
    - gaze_only：仅注视
    - face_only：仅面部（eye+mouth）
    """
    groups = _feature_groups(feature_cols)
    all_cols = list(feature_cols)
    return {
        "all": all_cols,
        "drop_gaze": [c for c in all_cols if c not in groups["gaze"]],
        "drop_eye": [c for c in all_cols if c not in groups["eye"]],
        "drop_mouth": [c for c in all_cols if c not in groups["mouth"]],
        "gaze_only": list(groups["gaze"]),
        "face_only": groups["eye"] + groups["mouth"],
    }


def _plot_ablation_barplot(
    ablation_df: pd.DataFrame, out_dir: Path
) -> None:
    """消融条形图：x=条件，y=AUC，标注相对 baseline 的下降。"""
    fig, ax = plt.subplots(figsize=(9, 5))
    conds = list(ablation_df["condition"])
    aucs = ablation_df["auc_mean"].values
    base_auc = aucs[0]
    colors = ["#2ca02c" if a >= base_auc - 1e-6 else "#d62728" for a in aucs]
    colors[0] = "#1f77b4"  # baseline
    bars = ax.bar(range(len(conds)), aucs, color=colors)
    ax.axhline(base_auc, color="gray", linestyle="--", linewidth=0.8, label="baseline")
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(conds, rotation=20, ha="right")
    ax.set_ylabel("AUC (mean over LOSO folds)")
    ax.set_title("Feature ablation: AUC by condition (F3 cross-attn)")
    ax.set_ylim(0.5, 1.0)
    for i, (b, a) in enumerate(zip(bars, aucs)):
        delta = a - base_auc
        label = "baseline" if i == 0 else f"Δ={delta:+.3f}"
        ax.text(b.get_x() + b.get_width() / 2, a + 0.005, label,
                ha="center", va="bottom", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "ablation_barplot.png", dpi=120)
    plt.close(fig)


def run_ablation(
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

    if not multimodal_csv.exists():
        df = DatasetBuilder.build_multimodal_feature_matrix(gaze_csv, face_csv)
        multimodal_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(multimodal_csv, index=False)
    else:
        df = pd.read_csv(multimodal_csv, dtype={"subject": str})

    feature_cols = [c for c in df.columns if c not in _META_COLS]
    groups = _feature_groups(feature_cols)
    print(f"[E5] 特征分组: " + ", ".join(f"{k}={len(v)}" for k, v in groups.items()))
    print(f"[E5] 总特征数: {len(feature_cols)}")

    conditions = _ablation_conditions(feature_cols)
    y = df["label"].values.astype(int)
    groups_arr = df["subject"].values

    all_per_fold: list[pd.DataFrame] = []
    rows: list[dict] = []

    for cond, cols in conditions.items():
        print(f"[E5] 条件 {cond}: {len(cols)} 特征")
        X = df[cols].values.astype(float)
        # F3 wrapper 需 gaze_dim：kept gaze 列数
        kept_gaze = [c for c in cols if not c.startswith("face_")]
        gaze_dim_cond = len(kept_gaze)
        factory = lambda gdc=gaze_dim_cond: CrossAttentionFusionWrapper(
            gaze_dim=gdc, epochs=30, batch_size=32,
        )
        grid = {"d_model": [16, 32], "lr": [1e-3]}
        try:
            res = evaluate_nested_loso(
                X=X, y=y, groups=groups_arr,
                model_factory=factory, param_grid=grid,
                inner_cv_folds=inner_cv_folds, random_state=random_state,
            )
        except Exception as e:
            print(f"      [警告] {cond} 失败: {e}")
            rows.append({
                "condition": cond, "n_features": len(cols),
                "gaze_dim": gaze_dim_cond,
                "auc_mean": float("nan"),
                "auc_ci_lo": float("nan"), "auc_ci_hi": float("nan"),
            })
            continue
        folds = res["per_fold"].copy()
        folds["model"] = "F3_cross_attn"
        folds["condition"] = cond
        folds["test_subject"] = folds["test_subject"].astype(int)
        all_per_fold.append(folds)
        s = res["summary"]
        rows.append({
            "condition": cond, "n_features": len(cols),
            "gaze_dim": gaze_dim_cond,
            "auc_mean": s["auc_mean"], "auc_ci_lo": s["auc_ci_lo"],
            "auc_ci_hi": s["auc_ci_hi"], "brier_score": s["brier_score"],
        })
        print(
            f"        AUC={s['auc_mean']:.3f} "
            f"CI=[{s['auc_ci_lo']:.3f}, {s['auc_ci_hi']:.3f}]"
        )

    ablation_df = pd.DataFrame(rows)
    ablation_df.to_csv(output_dir / "ablation_metrics.csv", index=False)
    print(f"[E5] → {output_dir / 'ablation_metrics.csv'}")

    metrics_df = pd.concat(all_per_fold, ignore_index=True)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    print(f"[E5] → {output_dir / 'metrics.csv'}  ({len(metrics_df)} 行)")

    _plot_ablation_barplot(ablation_df, output_dir)

    # baseline AUC 与下降量
    base_auc = float(ablation_df.iloc[0]["auc_mean"])
    drops = {}
    for _, r in ablation_df.iloc[1:].iterrows():
        drops[r["condition"]] = {
            "auc": float(r["auc_mean"]),
            "delta_vs_all": float(r["auc_mean"] - base_auc),
        }

    summary_out = {
        "n_samples": int(len(df)),
        "n_subjects": int(df["subject"].nunique()),
        "feature_groups": {k: v for k, v in groups.items()},
        "baseline_condition": "all",
        "baseline_auc": base_auc,
        "conditions": {r["condition"]: {k: (float(v) if isinstance(v, float) and not pd.isna(v) else v) for k, v in r.items()} for r in rows},
        "delta_vs_baseline": drops,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_out, f, ensure_ascii=False, indent=2)
    print(f"[E5] → {output_dir / 'summary.json'}")
    print(f"[E5] → {output_dir / 'figures'}/")

    cfg_used = {
        "inner_cv_folds": inner_cv_folds,
        "random_state": random_state,
        "model": "F3_cross_attn",
        "conditions": list(conditions.keys()),
        "source_config": str(config_path) if config_path else None,
    }
    with open(output_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_used, f, allow_unicode=True)

    print("[E5] 完成。")


def main() -> None:
    parser = argparse.ArgumentParser(description="E5 特征消融实验")
    parser.add_argument("--config", default="configs/multimodal.yaml")
    parser.add_argument(
        "--features", default="dataset/features/multimodal_features.csv",
    )
    parser.add_argument(
        "--output", default="results/multimodal/ablation",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(open(_PROJECT_ROOT / args.config))
    inner_cv_folds = cfg.get("nested_loso", {}).get("inner_cv_folds", 3)
    random_state = cfg.get("nested_loso", {}).get("random_state", 42)

    feat_dir = _PROJECT_ROOT / "dataset" / "features"
    run_ablation(
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
