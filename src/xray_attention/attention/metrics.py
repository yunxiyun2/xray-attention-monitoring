from __future__ import annotations

from typing import Any

import numpy as np

from xray_attention.data.records import TaskRecord


def longest_true_run(mask: np.ndarray) -> int:
    best = current = 0
    for value in mask:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def summarize_errors(errors: np.ndarray, thresholds: list[float]) -> dict[str, float]:
    summary = {
        "mean_error": float(np.mean(errors)),
        "median_error": float(np.median(errors)),
        "std_error": float(np.std(errors, ddof=1)),
        "n_samples": float(errors.size),
    }
    for threshold in thresholds:
        mask = errors <= threshold
        suffix = str(int(threshold))
        summary[f"inside_ratio_{suffix}"] = float(np.mean(mask))
        summary[f"longest_run_{suffix}"] = float(longest_true_run(mask))
    return summary


def compute_task_features(
    record: TaskRecord, thresholds: list[float]
) -> dict[str, str | int | float]:
    features: dict[str, Any] = {
        "subject_id": record.subject_id,
        "difficulty": record.difficulty,
        "label_name": record.label_name,
        "label": record.label,
    }
    features.update(summarize_errors(record.errors, thresholds))
    return features
