"""可解释性分析：排列重要性 + SHAP + F3 注意力热图。

对应 spec v2.0 §9.1/§9.2，回答 RQ5（哪类特征贡献最大）。
- 排列重要性（RF）：打乱每类特征，记录 AUC 下降
- TreeSHAP（RF）：top-10 特征重要性条形图 + 典型受试者 force plot
- F3 cross-attention 权重热图：验证困倦时段面部/注视协同的生理合理性

用法（从项目根目录）：
    PYTHONPATH=src conda run -n Anomaly python -m xray_attention.explain.shap_analysis \
        --features dataset/features/multimodal_features.csv \
        --output results/multimodal/explain
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

_META_COLS = {"subject", "label", "state", "difficulty", "window_order"}


def _load_data(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(csv_path, dtype={"subject": str})
    feat = [c for c in df.columns if c not in _META_COLS]
    X = df[feat].values.astype(float)
    y = df["label"].values.astype(int)
    g = df["subject"].values
    return X, y, g, feat


def _train_test_split_by_subject(
    X, y, groups, n_test_subjects: int = 4, random_state: int = 42
) -> tuple:
    """按受试者划分 train/test，保证无泄漏。"""
    gss = GroupShuffleSplit(
        n_splits=1, test_size=n_test_subjects, random_state=random_state,
    )
    tr, te = next(gss.split(X, y, groups))
    return (
        X[tr], X[te], y[tr], y[te],
        groups[tr], groups[te],
    )


def _per_subject_auc(
    y_test: np.ndarray, scores: np.ndarray, subj_test: np.ndarray
) -> dict[str, float]:
    """计算 test 集每个受试者的 AUC。"""
    out = {}
    for s in np.unique(subj_test):
        mask = subj_test == s
        if len(np.unique(y_test[mask])) < 2:
            out[s] = float("nan")
            continue
        out[s] = float(roc_auc_score(y_test[mask], scores[mask]))
    return out


def compute_permutation_importance(
    rf: RandomForestClassifier,
    X_test: np.ndarray, y_test: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """排列重要性：打乱每列，记录 AUC 下降（spec §9.1）。"""
    base_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    rng = np.random.RandomState(random_state)
    drops = []
    for j, name in enumerate(feature_names):
        vals = X_test[:, j].copy()
        aucs = []
        for _ in range(n_repeats):
            perm = rng.permutation(len(vals))
            X_test[:, j] = vals[perm]
            aucs.append(roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]))
        X_test[:, j] = vals  # 还原
        drops.append({
            "feature": name,
            "auc_drop": base_auc - np.mean(aucs),
            "auc_drop_std": np.std(aucs),
        })
    df = pd.DataFrame(drops).sort_values("auc_drop", ascending=False)
    df["base_auc"] = base_auc
    return df.reset_index(drop=True)


def plot_permutation_importance(imp_df: pd.DataFrame, out_dir: Path, top: int = 15) -> None:
    df = imp_df.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(5, top * 0.35)))
    ax.barh(df["feature"], df["auc_drop"], xerr=df["auc_drop_std"])
    ax.set_xlabel("AUC drop (permutation)")
    ax.set_title(f"Permutation importance (top {top})")
    fig.tight_layout()
    fig.savefig(out_dir / "permutation_importance.png", dpi=120)
    plt.close(fig)


def plot_shap_top_features(
    shap_values: np.ndarray, feature_names: list[str], out_dir: Path, top: int = 10
) -> None:
    """SHAP top-10 特征重要性条形图（spec §9.1）。"""
    # shap_values shape: (n_samples, n_features) — 取 |shap| 均值
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]  # 二分类取正类
    importance = np.mean(np.abs(shap_values), axis=0)
    order = np.argsort(importance)[::-1][:top]
    fig, ax = plt.subplots(figsize=(8, max(5, top * 0.4)))
    names = [feature_names[i] for i in order]
    ax.barh(names[::-1], importance[order][::-1])
    ax.set_xlabel("mean(|SHAP value|)")
    ax.set_title(f"SHAP feature importance (top {top})")
    fig.tight_layout()
    fig.savefig(out_dir / "shap_top10_barplot.png", dpi=120)
    plt.close(fig)


def plot_force_plots(
    shap_values: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    subj_test: np.ndarray,
    feature_names: list[str],
    per_subj_auc: dict,
    out_dir: Path,
    base_values: np.ndarray | None = None,
) -> None:
    """对 2-3 个典型受试者（高/低 AUC）出 SHAP force plot（spec §9.2）。"""
    if shap_values.ndim == 3:
        shap_pos = shap_values[:, :, 1]
        # base_values 可能是 [ev0, ev1]
        if base_values is not None and np.ndim(base_values) == 1 and len(base_values) >= 2:
            base = float(base_values[1])
        elif base_values is not None:
            base = float(np.ravel(base_values)[0])
        else:
            base = float(np.mean(np.abs(shap_pos)))
    else:
        shap_pos = shap_values
        base = float(base_values[0]) if base_values is not None else float(np.mean(np.abs(shap_pos)))
    valid = {s: a for s, a in per_subj_auc.items() if not np.isnan(a)}
    if not valid:
        return
    sorted_s = sorted(valid.items(), key=lambda kv: kv[1])
    picks = []
    if len(sorted_s) >= 1:
        picks.append(("low_auc", sorted_s[0]))     # 最低 AUC
    if len(sorted_s) >= 2:
        picks.append(("high_auc", sorted_s[-1]))    # 最高 AUC
    if len(sorted_s) >= 3:
        picks.append(("mid_auc", sorted_s[len(sorted_s) // 2]))  # 中间
    for tag, (subj, auc) in picks:
        mask = subj_test == subj
        if mask.sum() == 0:
            continue
        idx = np.where(mask)[0][0]
        sv = shap.Explanation(
            values=shap_pos[idx], base_values=base,
            data=X_test[idx], feature_names=feature_names,
        )
        try:
            shap.plots.waterfall(sv, max_display=10, show=False)
            fig = plt.gcf()
            fig.suptitle(f"SHAP waterfall: subject={subj} (AUC={auc:.3f}, label={y_test[idx]})")
            fig.tight_layout()
            fig.savefig(out_dir / f"shap_force_{tag}_subject{subj}.png", dpi=120)
            plt.close(fig)
        except Exception as e:
            print(f"[shap] force plot for subject={subj} 失败: {e}")


def plot_f3_attention_heatmap(
    multimodal_csv: Path,
    gaze_dim: int,
    out_dir: Path,
    random_state: int = 42,
) -> None:
    """F3 跨模态贡献比可视化（spec §9.2）。

    单 token 序列下 MultiheadAttention 权重恒为 1.0，无生理区分度；
    改用各流 attended 表示的 L2 范数平方比作为贡献代理：
    gaze_ratio = ||face_stream_output||² / (||gaze_stream||² + ||face_stream||²)
    反映各样本中 gaze/face 特征对融合层的相对贡献，验证生理合理性。
    """
    import torch
    from xray_attention.models.fusion_sklearn_wrappers import (
        CrossAttentionFusionWrapper,
    )

    df = pd.read_csv(multimodal_csv, dtype={"subject": str})
    feat = [c for c in df.columns if c not in _META_COLS]
    X = df[feat].values.astype(float)
    y = df["label"].values.astype(int)

    wrapper = CrossAttentionFusionWrapper(
        gaze_dim=gaze_dim, d_model=32, epochs=30, batch_size=32,
        random_state=random_state,
    )
    wrapper.fit(X, y)
    model = wrapper._model
    model.eval()

    gaze_t = torch.tensor(X[:, :gaze_dim], dtype=torch.float32)
    face_t = torch.tensor(X[:, gaze_dim:], dtype=torch.float32)

    # 单 token 序列下 nn.MultiheadAttention 的注意力权重恒为 1.0
    # （softmax over 单元素），无法反映生理差异。
    # 改用跨模态贡献比：gaze 流经 face-attention 后的范式占总融合表示范式比，
    # 衡量各样本中 gaze/face 特征对融合层的相对贡献（spec §9.2 生理合理性验证）。
    with torch.no_grad():
        g = model.gaze_proj(gaze_t).unsqueeze(1)  # (B, 1, d)
        f = model.face_proj(face_t).unsqueeze(1)  # (B, 1, d)
        # g_attn: gaze stream attended to face（face 信息流入 gaze 流）
        g_attn, _ = model.attn_g2f(g, f, f, need_weights=False)
        # f_attn: face stream attended to gaze（gaze 信息流入 face 流）
        f_attn, _ = model.attn_f2g(f, g, g, need_weights=False)
        g_norm = g_attn.squeeze(1).norm(dim=1).pow(2).numpy()  # face-derived
        f_norm = f_attn.squeeze(1).norm(dim=1).pow(2).numpy()  # gaze-derived
        gaze_ratio = f_norm / (g_norm + f_norm + 1e-8)  # (B,) gaze 贡献比

    # 按标签排序的热图
    order = np.argsort(y)
    ratio_sorted = gaze_ratio[order]
    y_sorted = y[order]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             gridspec_kw={"width_ratios": [1, 20]})
    # 左侧标签色条
    label_colors = np.array([[1, 0.6, 0.6] if v == 1 else [0.6, 0.8, 1]
                             for v in y_sorted]).reshape(-1, 1, 3)
    axes[0].imshow(label_colors, aspect="auto")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_title("label")
    for i, v in enumerate(y_sorted):
        if i == 0 or v != y_sorted[i - 1]:
            axes[0].axhline(i, color="k", linewidth=0.5)
    # 右侧贡献比热图
    im = axes[1].imshow(ratio_sorted.reshape(-1, 1), aspect="auto",
                        cmap="RdYlBu_r", vmin=0, vmax=1)
    axes[1].set_xlabel("gaze contribution ratio")
    axes[1].set_ylabel("sample (sorted by label)")
    axes[1].set_title("F3 modality contribution (gaze ratio)")
    axes[1].set_yticks([])
    axes[1].set_xticks([0])
    axes[1].set_xticklabels(["gaze"], rotation=45, ha="right")
    fig.colorbar(im, ax=axes[1], fraction=0.02)
    fig.tight_layout()
    fig.savefig(out_dir / "f3_attention_heatmap.png", dpi=120)
    plt.close(fig)

    # 另出按 label 分组的箱线图（更易读）
    fig2, ax = plt.subplots(figsize=(7, 5))
    alert = gaze_ratio[y == 0]
    sleepy = gaze_ratio[y == 1]
    ax.boxplot([alert, sleepy], tick_labels=["alert", "sleepy"])
    ax.set_ylabel("gaze contribution ratio")
    ax.set_title("F3 modality contribution by state")
    ax.set_ylim(0, 1)
    fig2.tight_layout()
    fig2.savefig(out_dir / "f3_attention_by_state.png", dpi=120)
    plt.close(fig2)
    return gaze_ratio


def run_explain(
    multimodal_csv: Path,
    output_dir: Path,
    n_test_subjects: int = 4,
    random_state: int = 42,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, groups, feat_names = _load_data(multimodal_csv)
    gaze_dim = len([c for c in feat_names if not c.startswith("face_")])
    print(
        f"[explain] 数据: X={X.shape}, gaze_dim={gaze_dim}, "
        f"face_dim={X.shape[1] - gaze_dim}"
    )

    # ---- 1. train/test split + RF ----
    X_tr, X_te, y_tr, y_te, _, subj_te = _train_test_split_by_subject(
        X, y, groups, n_test_subjects=n_test_subjects, random_state=random_state,
    )
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=random_state,
        class_weight="balanced",
    )
    rf.fit(X_tr, y_tr)
    scores_te = rf.predict_proba(X_te)[:, 1]
    per_subj_auc = _per_subject_auc(y_te, scores_te, subj_te)
    print(f"[explain] test per-subject AUC: {per_subj_auc}")

    # ---- 2. 排列重要性 ----
    print("[explain] 计算排列重要性 ...")
    imp_df = compute_permutation_importance(
        rf, X_te.copy(), y_te, feat_names, random_state=random_state,
    )
    imp_df.to_csv(output_dir / "permutation_importance.csv", index=False)
    plot_permutation_importance(imp_df, output_dir)
    print(f"[explain] → {output_dir / 'permutation_importance.png'}")

    # ---- 3. TreeSHAP ----
    print("[explain] 计算 SHAP 值 (TreeExplainer) ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.TreeExplainer(rf)
        sv = explainer.shap_values(X_te)
    # 二分类 RF：sv 可能是 list[2] 或 (n, f, 2)
    if isinstance(sv, list):
        sv_arr = np.stack(sv, axis=-1)  # (n, f, 2)
    else:
        sv_arr = sv
    plot_shap_top_features(sv_arr, feat_names, output_dir)
    print(f"[explain] → {output_dir / 'shap_top10_barplot.png'}")

    # ---- 4. Force/waterfall plots for typical subjects ----
    print("[explain] 生成典型受试者 force plot ...")
    expected_val = getattr(explainer, "expected_value", None)
    plot_force_plots(
        sv_arr, X_te, y_te, subj_te, feat_names, per_subj_auc, output_dir,
        base_values=expected_val,
    )

    # ---- 5. F3 attention heatmap ----
    print("[explain] 生成 F3 attention heatmap ...")
    try:
        plot_f3_attention_heatmap(multimodal_csv, gaze_dim, output_dir, random_state)
        print(f"[explain] → {output_dir / 'f3_attention_heatmap.png'}")
    except Exception as e:
        print(f"[explain] [警告] F3 attention heatmap 失败: {e}")

    summary = {
        "n_samples": int(len(y)),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "test_subjects": sorted(np.unique(subj_te).tolist()),
        "test_per_subject_auc": {str(k): float(v) for k, v in per_subj_auc.items()},
        "base_auc_test": float(roc_auc_score(y_te, scores_te)),
        "top5_permutation_features": imp_df.head(5)[["feature", "auc_drop"]].to_dict("records"),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[explain] → {output_dir / 'summary.json'}")
    print("[explain] 完成。")


def main() -> None:
    parser = argparse.ArgumentParser(description="SHAP/排列重要性分析")
    parser.add_argument(
        "--features", default="dataset/features/multimodal_features.csv",
    )
    parser.add_argument(
        "--output", default="results/multimodal/explain",
    )
    parser.add_argument("--n-test-subjects", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    run_explain(
        multimodal_csv=_PROJECT_ROOT / args.features,
        output_dir=_PROJECT_ROOT / args.output,
        n_test_subjects=args.n_test_subjects,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
