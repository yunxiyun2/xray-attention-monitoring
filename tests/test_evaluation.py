import pandas as pd

from xray_attention.attention.evaluation import evaluate_nested_loso


def test_nested_loso_never_uses_test_subject_to_select_threshold() -> None:
    frame = pd.DataFrame(
        {
            "subject_id": ["01", "01", "02", "02", "03", "03"],
            "difficulty": ["easy"] * 6,
            "label": [1, 0, 1, 0, 1, 0],
            "inside_ratio_50": [0.9, 0.1, 0.8, 0.2, 0.2, 0.8],
            "inside_ratio_100": [0.95, 0.05, 0.9, 0.1, 0.85, 0.15],
        }
    )
    _, folds, _ = evaluate_nested_loso(frame, [50, 100])
    assert set(folds["test_subject"]) == {"01", "02", "03"}
    assert all(folds["train_subject_count"] == 2)
