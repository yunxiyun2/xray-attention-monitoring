import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from dual_factor.config_v7 import config_hash, validate_config
from dual_factor.predictive_v7 import (
    build_paired_dataset,
    decide,
    fit_loso,
    run_experiment,
)
from dual_factor.protocol_v7 import MODELS, RETAINED


def synthetic_frame(subjects=(1, 2, 3, 4, 5, 6, 7, 8), records=("1", "2")):
    frames = []
    for subject in subjects:
        for test_index, test in enumerate(records):
            label = test_index
            values = np.full(20, 50.0 * label)
            frame = pd.DataFrame({
                "subject": str(subject),
                "test": test,
                "kss": 3 + 5 * label,
                "window_idx": np.arange(20) * 3,
                "valid": True,
                **{column: values for column in RETAINED},
            })
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def base_analysis():
    return {
        "window_sec": 30.0,
        "stride_sec": 10.0,
        "primary_phase": 0,
        "alert_max": 4,
        "drowsy_min": 6,
        "min_pairs": 8,
        "min_state_windows": 10,
        "delta_auc_min": 0.05,
    }


def synthetic_config(tmp_path):
    return validate_config({
        "protocol": "v7.0",
        "input": {"path": str(tmp_path / "matrix.csv")},
        "analysis": base_analysis(),
        "model": {"type": "logistic", "regularization_c": .1, "max_iter": 100, "seed": 7},
        "bootstrap": {"iterations": 20, "seed": 11},
        "output": {"root": str(tmp_path / "results"), "overwrite": False},
    })


def write_matrix(tmp_path):
    path = tmp_path / "matrix.csv"
    synthetic_frame().to_csv(path, index=False)
    return path


def test_config_rejects_missing_sections_and_overwrite():
    with pytest.raises(ValueError, match="Missing configuration sections"):
        validate_config({"protocol": "v7.0"})
    with pytest.raises(ValueError, match="overwrite is prohibited"):
        validate_config({
            "protocol": "v7.0",
            "input": {"path": "x"},
            "analysis": base_analysis(),
            "model": {"type": "logistic", "regularization_c": .1, "max_iter": 10, "seed": 0},
            "bootstrap": {"iterations": 10, "seed": 0},
            "output": {"root": "y", "overwrite": True},
        })
    config = synthetic_config(Path("/tmp/v7-test"))
    assert config_hash(config)


def test_paired_dataset_keeps_subjects_and_phase():
    data, audit = build_paired_dataset(synthetic_frame(), base_analysis())
    assert audit["status"] == "ok"
    assert data.subject.nunique() == 8
    assert set(data.state) == {"alert", "drowsy"}
    assert data.n_windows.min() >= 10
    assert set(data.records) == {"1", "2"}
    assert (data["global"] == (data["f1"] + data["f2"]) / 2).all()


def test_loso_holds_out_whole_subjects():
    data, _ = build_paired_dataset(synthetic_frame(), base_analysis())
    model_config = {"regularization_c": .1, "max_iter": 100, "seed": 7}
    predictions, folds = fit_loso(data, model_config)
    assert set(predictions.model) == set(MODELS)
    assert predictions.groupby(["model", "subject"]).size().eq(2).all()
    assert folds.groupby("held_subject").size().eq(len(MODELS)).all()
    for subject in data.subject.unique():
        training = folds.loc[folds.held_subject == subject, "train_subjects"]
        assert all(subject not in value.split("|") for value in training)


def test_decision_boundaries():
    assert decide(.06, .01, .10, .05) == "incremental_supported"
    assert decide(.06, -.02, .10, .05) == "inconclusive"
    assert decide(-.06, -.12, -.02, .05) == "global_better"
    assert decide(-.06, -.10, .01, .05) == "inconclusive"


def test_run_creates_reproducible_outputs(tmp_path):
    path = write_matrix(tmp_path)
    config = synthetic_config(tmp_path)
    result = run_experiment(config, tmp_path)
    assert result["audit"]["status"] == "ok"
    assert result["decision"]["efa_v6_decision"] == "B"
    run_dir = Path(result["run_dir"])
    for name in ("decision.json", "fold_predictions.csv", "bootstrap_samples.csv"):
        assert (run_dir / name).exists()
    assert (run_dir / "figures/roc_and_auc.png").exists()
