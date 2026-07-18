import numpy as np

from xray_attention.attention.metrics import longest_true_run, summarize_errors
from xray_attention.attention.metrics import compute_task_features
from xray_attention.data.records import TaskRecord


def test_longest_true_run_counts_contiguous_samples() -> None:
    assert longest_true_run(np.array([False, True, True, False, True])) == 2


def test_summarize_errors_computes_threshold_ratio_and_run() -> None:
    summary = summarize_errors(np.array([10.0, 40.0, 80.0, 20.0]), [25.0, 50.0])
    assert summary["inside_ratio_25"] == 0.5
    assert summary["longest_run_50"] == 2
    assert summary["mean_error"] == 37.5


def test_compute_task_features_includes_record_identity_and_summary() -> None:
    record = TaskRecord(
        subject_id="01",
        difficulty="easy",
        label_name="alert",
        label=1,
        errors=np.array([10.0, 40.0, 80.0, 20.0]),
    )
    features = compute_task_features(record, [25.0])
    assert features["subject_id"] == "01"
    assert features["difficulty"] == "easy"
    assert features["label_name"] == "alert"
    assert features["label"] == 1
    assert features["inside_ratio_25"] == 0.5
