from __future__ import annotations

from math import nan
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


REGIMES = ("overall", "easy", "hard")


def _score_column(threshold: int | float) -> str:
    return f"inside_ratio_{int(threshold)}"


def _has_two_classes(labels: Iterable[int]) -> bool:
    return len(set(labels)) == 2


def _roc_auc(labels: pd.Series, scores: pd.Series) -> float:
    if not _has_two_classes(labels):
        return nan
    return float(roc_auc_score(labels, scores))


def _candidate_cutoffs(scores: pd.Series) -> list[float]:
    values = sorted(float(value) for value in scores.dropna().unique())
    if len(values) <= 1:
        return values
    return [(left + right) / 2.0 for left, right in zip(values, values[1:])]


def _best_cutoff(labels: pd.Series, scores: pd.Series) -> tuple[float, float]:
    best_cutoff = nan
    best_balanced_accuracy = -np.inf
    for cutoff in _candidate_cutoffs(scores):
        predictions = (scores >= cutoff).astype(int)
        score = float(balanced_accuracy_score(labels, predictions))
        if (score, -cutoff) > (best_balanced_accuracy, -best_cutoff):
            best_balanced_accuracy = score
            best_cutoff = cutoff
    if best_balanced_accuracy == -np.inf:
        return nan, nan
    return best_cutoff, best_balanced_accuracy


def _threshold_summary(
    frame: pd.DataFrame, threshold: int | float, regime: str | None = None
) -> dict[str, float | int | str]:
    column = _score_column(threshold)
    labels = frame["label"]
    scores = frame[column]
    positive_scores = scores[labels == 1]
    negative_scores = scores[labels == 0]
    cutoff, cutoff_balanced_accuracy = _best_cutoff(labels, scores)

    u_statistic = nan
    rank_biserial = nan
    if len(positive_scores) > 0 and len(negative_scores) > 0:
        result = mannwhitneyu(positive_scores, negative_scores, alternative="two-sided")
        u_statistic = float(result.statistic)
        rank_biserial = 1 - (2 * u_statistic) / (
            len(positive_scores) * len(negative_scores)
        )

    summary: dict[str, float | int | str] = {
        "threshold": int(threshold),
        "n_tasks": int(len(frame)),
        "positive_count": int(len(positive_scores)),
        "negative_count": int(len(negative_scores)),
        "roc_auc": _roc_auc(labels, scores),
        "best_balanced_accuracy": cutoff_balanced_accuracy,
        "decision_cutoff": cutoff,
        "mannwhitney_u": u_statistic,
        "rank_biserial": rank_biserial,
    }
    if regime is not None:
        summary["regime"] = regime
    return summary


def choose_threshold(training: pd.DataFrame, thresholds: list[int]) -> tuple[int, float]:
    summaries = [_threshold_summary(training, threshold) for threshold in thresholds]
    summaries.sort(
        key=lambda item: (
            -float(item["roc_auc"]) if not pd.isna(item["roc_auc"]) else np.inf,
            -float(item["best_balanced_accuracy"])
            if not pd.isna(item["best_balanced_accuracy"])
            else np.inf,
            int(item["threshold"]),
        )
    )
    best = summaries[0]
    return int(best["threshold"]), float(best["decision_cutoff"])


def _regime_frame(features: pd.DataFrame, regime: str) -> pd.DataFrame:
    if regime == "overall":
        return features
    return features[features["difficulty"] == regime]


def _validate_inputs(features: pd.DataFrame, thresholds: list[int]) -> None:
    required = {"subject_id", "difficulty", "label"}
    required.update(_score_column(threshold) for threshold in thresholds)
    missing = sorted(required.difference(features.columns))
    if missing:
        raise ValueError(f"features is missing required columns: {missing}")
    if not thresholds:
        raise ValueError("thresholds must not be empty")


def evaluate_nested_loso(
    features: pd.DataFrame, thresholds: list[int]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    _validate_inputs(features, thresholds)

    candidate_rows: list[dict[str, float | int | str]] = []
    fold_rows: list[dict[str, float | int | str]] = []
    recommendations: dict[str, float] = {}

    for regime in REGIMES:
        regime_features = _regime_frame(features, regime).copy()
        if regime_features.empty:
            continue

        for threshold in thresholds:
            candidate_rows.append(_threshold_summary(regime_features, threshold, regime))

        selected_threshold, _ = choose_threshold(regime_features, thresholds)
        recommendations[regime] = float(selected_threshold)

        for subject_id in sorted(regime_features["subject_id"].unique()):
            train = regime_features[regime_features["subject_id"] != subject_id]
            test = regime_features[regime_features["subject_id"] == subject_id]
            selected_threshold, decision_cutoff = choose_threshold(train, thresholds)
            score_column = _score_column(selected_threshold)
            predictions = (test[score_column] >= decision_cutoff).astype(int)
            fold_rows.append(
                {
                    "regime": regime,
                    "test_subject": subject_id,
                    "train_subject_count": int(train["subject_id"].nunique()),
                    "selected_threshold": selected_threshold,
                    "decision_cutoff": decision_cutoff,
                    "balanced_accuracy": float(
                        balanced_accuracy_score(test["label"], predictions)
                    ),
                    "roc_auc": _roc_auc(test["label"], test[score_column]),
                }
            )

    return (
        pd.DataFrame(candidate_rows),
        pd.DataFrame(fold_rows),
        recommendations,
    )
