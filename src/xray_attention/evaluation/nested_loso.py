"""嵌套留一受试者评估（Nested LOSO）。

对应 spec v2.0 §7.2 超参搜索、§8 评估协议。

外层：Leave-One-Subject-Out 估泛化性能（20 折）。
内层：Leave-One-Subject-In 在 19 名训练受试者上网格搜索超参，
       内层 scoring = balanced_accuracy。
"""
from __future__ import annotations

from typing import Callable, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, f1_score, confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.preprocessing import StandardScaler

from xray_attention.evaluation.stats import (
    bootstrap_auc_ci, wilcoxon_vs_baseline, rank_biserial_r,
)


def evaluate_nested_loso(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_factory: Callable,
    param_grid: dict,
    inner_scoring: str = "balanced_accuracy",
    inner_cv_folds: int = 3,
    random_state: int = 42,
) -> dict:
    """嵌套 LOSO 评估。

    Parameters
    ----------
    X : (n_samples, n_features) 特征矩阵
    y : (n_samples,) 二值标签
    groups : (n_samples,) 受试者 ID（外层 LOSO 分组）
    model_factory : 无参 callable，返回新模型实例（含默认超参）
    param_grid : 超参网格字典，如 {"C": [0.1, 1, 10]}
    inner_scoring : 内层 GridSearch scoring
    inner_cv_folds : 内层 CV 折数（GroupKFold 按受试者分组）
    random_state : 随机种子

    Returns
    -------
    dict 含 per_fold DataFrame、summary（mean/std/95% CI/Wilcoxon p/效应量）
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    groups = np.asarray(groups)
    subjects = np.unique(groups)

    per_fold: list[dict] = []
    # 收集外层测试折的 per-sample 真值与预测分数，用于 bootstrap CI
    pooled_y: list = []
    pooled_scores: list = []
    pooled_pred: list = []
    for test_subj in subjects:
        train_mask = groups != test_subj
        test_mask = groups == test_subj
        X_tr, X_te = X[train_mask], X[test_mask]
        y_tr, y_te = y[train_mask], y[test_mask]
        g_tr = groups[train_mask]

        # 标准化仅在训练折拟合（防泄漏）
        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # 内层 GroupKFold（按受试者分组，防受试者级泄漏）
        inner_cv = GroupKFold(n_splits=min(inner_cv_folds, len(np.unique(g_tr))))
        gs = GridSearchCV(
            model_factory(), param_grid, scoring=inner_scoring,
            cv=inner_cv, n_jobs=-1, refit=True,
        )
        gs.fit(X_tr_s, y_tr, groups=g_tr)
        best = gs.best_estimator_

        # 外层测试折预测
        if hasattr(best, "predict_proba"):
            scores = best.predict_proba(X_te_s)[:, 1]
        else:
            scores = best.decision_function(X_te_s)
        y_pred = best.predict(X_te_s)

        pooled_y.extend(y_te.tolist())
        pooled_scores.extend(np.asarray(scores).ravel().tolist())
        pooled_pred.extend(np.asarray(y_pred).ravel().tolist())

        # 指标
        if len(np.unique(y_te)) < 2:
            auc = 0.5
        else:
            auc = float(roc_auc_score(y_te, scores))
        bal = float(balanced_accuracy_score(y_te, y_pred))
        f1 = float(f1_score(y_te, y_pred, zero_division=0))
        cm = confusion_matrix(y_te, y_pred, labels=[0, 1]).ravel().tolist()

        per_fold.append({
            "test_subject": test_subj,
            "auc": auc,
            "balanced_accuracy": bal,
            "f1": f1,
            "n_test": int(test_mask.sum()),
            "best_params": str(gs.best_params_),
            "tn": int(cm[0]), "fp": int(cm[1]),
            "fn": int(cm[2]), "tp": int(cm[3]),
        })

    folds_df = pd.DataFrame(per_fold)
    aucs = folds_df["auc"].values

    # bootstrap AUC CI 在外层测试折的 per-sample 池化预测上计算
    pooled_y_arr = np.asarray(pooled_y)
    pooled_scores_arr = np.asarray(pooled_scores)
    lo, hi, mean = bootstrap_auc_ci(
        pooled_y_arr, pooled_scores_arr, B=500, seed=random_state,
    )
    # 池化级 AUC（全外层样本）
    if len(np.unique(pooled_y_arr)) >= 2:
        pooled_auc = float(roc_auc_score(pooled_y_arr, pooled_scores_arr))
    else:
        pooled_auc = 0.5
    # Brier score（概率校准，用池化预测）
    brier = float(np.mean(
        (pooled_scores_arr - np.asarray(pooled_y_arr, dtype=float)) ** 2
    ))
    # 折级 AUC Wilcoxon vs 0.5
    try:
        p_vs_random = wilcoxon_vs_baseline(aucs, baseline=0.5)
    except Exception:
        p_vs_random = 1.0
    eff = rank_biserial_r(aucs, baseline=0.5)

    summary = {
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "auc_ci_lo": lo,
        "auc_ci_hi": hi,
        "auc_pooled": pooled_auc,
        "brier_score": brier,
        "balacc_mean": float(folds_df["balanced_accuracy"].mean()),
        "f1_mean": float(folds_df["f1"].mean()),
        "wilcoxon_vs_0.5_p": p_vs_random,
        "rank_biserial_r": eff,
        "n_folds": int(len(aucs)),
        "n_positive_auc": int(np.sum(aucs > 0.5)),
    }
    return {
        "per_fold": folds_df,
        "summary": summary,
        "pooled_y": pooled_y_arr,
        "pooled_scores": pooled_scores_arr,
        "pooled_pred": np.asarray(pooled_pred),
    }
