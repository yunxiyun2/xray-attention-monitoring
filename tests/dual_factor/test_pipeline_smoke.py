"""Pipeline smoke test。"""
from __future__ import annotations

from pathlib import Path

import pytest

from dual_factor.pipeline import run_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DROZY_ROOT = PROJECT_ROOT / "data_public" / "DROZY"


def test_pipeline_writes_required_outputs(tmp_path: Path) -> None:
    if not DROZY_ROOT.exists():
        pytest.skip("DROZY dataset not available")
    config = {
        "data": {"drozy_root": str(DROZY_ROOT), "channels": ["EOG-H", "EOG-V", "Cam-Sync"],
                 "sampling_rate": 512, "subjects": list(range(1, 15)),
                 "exclude_tests": ["7-1"]},
        "preprocess": {"highpass": 0.5, "lowpass": 30.0, "filter_order": 4,
                       "hampel_window": 51, "hampel_threshold": 3.0},
        "event_detection": {"algorithm": "I-VT", "velocity_threshold": "auto",
                            "velocity_percentile": 70, "min_fixation_ms": 100, "min_saccade_ms": 20},
        "windowing": {"window_sec": 30, "stride_sec": 10, "min_fixations": 5,
                      "min_saccades": 3, "normalization": "within_subject_zscore"},
        "spatial_grid": {"bins": 5},
        "labels": {"kss_file": "KSS.txt", "alert_threshold": 4,
                   "drowsy_threshold": 6, "exclude_middle": True},
        "efa": {"extraction": "pa", "rotation": "promax", "max_factors": 4,
                "variance_threshold": 0.60, "loading_threshold": 0.50,
                "cross_loading_threshold": 0.30, "factor_corr_threshold": 0.70,
                "parallel_analysis_n_iter": 50, "parallel_analysis_percentile": 95},
        "output": {"results_dir": str(tmp_path / "results"), "figures_dir": str(tmp_path / "results" / "figures")},
    }
    run_pipeline(config)
    expected = {
        "indicator_matrix.csv", "descriptive_stats.csv", "kmo_bartlett.json",
        "efa_results.json", "decision_report.md", "config_used.yaml",
    }
    assert expected <= {p.name for p in tmp_path.joinpath("results").iterdir()}
    figures = tmp_path / "results" / "figures"
    assert figures.exists()
    fig_files = {p.name for p in figures.iterdir()}
    assert {"scree_parallel.png", "loading_matrix.png", "factor_scatter.png", "direction_check.png"} <= fig_files
