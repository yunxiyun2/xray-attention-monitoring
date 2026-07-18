from pathlib import Path
import yaml
from experiments.threshold_selection.run import run_experiment


def _write_sequence(root, difficulty, subject, label, values):
    target = root / difficulty / subject / label
    target.mkdir(parents=True)
    (target / "all_errors.txt").write_text(values, encoding="utf-8")


def test_run_experiment_writes_required_outputs(tmp_path):
    data_root = tmp_path / "errors"
    for difficulty in ["easy", "hard"]:
        for subject in ["01", "02", "03"]:
            _write_sequence(data_root, difficulty, subject, "alert", "10\n20\n30\n")
            _write_sequence(data_root, difficulty, subject, "sleepy", "200\n220\n240\n")
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({
        "data_root": str(data_root),
        "output_dir": str(tmp_path / "out"),
        "thresholds_px": {"start": 50, "stop": 100, "step": 50},
        "labels": {"alert": 1, "sleepy": 0},
        "task_difficulties": ["easy", "hard"],
        "random_seed": 20260712,
    }), encoding="utf-8")
    run_experiment(config, tmp_path / "out")
    expected = {
        "config_used.yaml", "task_features.csv", "candidate_summary.csv",
        "nested_loso_folds.csv", "recommended_threshold.json",
        "threshold_performance.png", "selected_thresholds.png",
    }
    assert expected <= {path.name for path in (tmp_path / "out").iterdir()}
