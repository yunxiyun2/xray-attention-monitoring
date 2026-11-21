from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eog_si_mset.config import config_hash, load_config, validate_config
from eog_si_mset.data import (
    _resample_pair,
    _robust_normalize,
    paired_loso_folds,
    paired_subjects,
    prepare_cache,
)
from dual_factor.data.drozy_loader import DrozyRecord


def _config(tmp_path):
    return {
        "protocol": "si-mset-v1",
        "data": {
            "drozy_root": str(tmp_path / "DROZY"),
            "cache_dir": str(tmp_path / "cache"),
            "subjects": [1, 2],
            "exclude_tests": [],
            "channels": ["EOG-H", "EOG-V"],
            "kss_file": "KSS.txt",
            "target_fs": 128,
            "window_sec": 30.0,
            "stride_sec": 10.0,
            "primary_phase": 0,
            "min_fixations": 5,
            "min_saccades": 3,
            "alert_max": 4,
            "drowsy_min": 6,
            "preprocess": {
                "highpass": .5, "lowpass": 30., "filter_order": 4,
                "hampel_window": 51, "hampel_threshold": 3.,
            },
            "event_detection": {
                "velocity_threshold": "auto", "velocity_percentile": 70.,
                "min_fixation_ms": 100., "min_saccade_ms": 20.,
            },
        },
        "model": {
            "variant": "full",
            "d_model": 96,
            "transformer_layers": 3,
            "attention_heads": 4,
            "ffn_dim": 192,
            "sequence_tokens": 96,
            "dropout": 0.15,
        },
        "training": {
            "epochs": 2,
            "patience": 1,
            "batches_per_epoch": 1,
            "batch_subjects": 2,
            "windows_per_state": 2,
            "windows_per_intermediate": 0,
            "validation_windows": 2,
            "seed": 1,
            "optimizer": "adamw",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "bce_weight": 1.0,
            "ordinal_weight": .5,
            "pair_weight": .25,
            "adversarial_weight": .1,
            "pair_margin": .25,
            "noise_std": .01,
            "device": "cpu",
            "amp": False,
            "adversarial_warmup_epochs": 1,
        },
        "evaluation": {
            "bootstrap_iterations": 10,
            "seed": 1,
            "min_auc": .75,
            "min_auc_lower": .5,
            "min_delta_auc": .05,
            "min_delta_lower": 0.0,
            "baseline_run": None,
        },
        "output": {"root": str(tmp_path / "runs"), "overwrite": False, "save_checkpoints": False},
    }


def _metadata():
    rows = []
    for subject in range(1, 11):
        for state, label, kss in (("alert", 0, 3), ("drowsy", 1, 7)):
            for window_idx in range(3):
                rows.append({
                    "subject": subject,
                    "test": 1 if state == "alert" else 3,
                    "kss": kss,
                    "label": label,
                    "state": state,
                    "window_idx": window_idx,
                })
    return pd.DataFrame(rows)


def test_config_hash_and_invalid_overwrite(tmp_path):
    config = validate_config(_config(tmp_path))
    assert len(config_hash(config)) == 64
    raw = _config(tmp_path)
    raw["output"]["overwrite"] = True
    with pytest.raises(ValueError, match="overwrite"):
        validate_config(raw)


def test_long_run_config_is_valid():
    project_root = Path(__file__).resolve().parents[2]
    config = load_config(project_root / "configs/eog_si_mset_v1_1.yaml")
    assert config["protocol"] == "si-mset-v1.1"
    assert config["training"]["validation_mode"] == "training_subjects"
    assert config["training"]["epochs"] == 130
    assert config["model"]["transformer_layers"] == 5


def test_v12_config_is_valid():
    project_root = Path(__file__).resolve().parents[2]
    config = load_config(project_root / "configs/eog_si_mset_v1_2.yaml")
    assert config["protocol"] == "si-mset-v1.2"
    assert len(config["training"]["replicate_seeds"]) == 3
    assert config["evaluation"]["calibration_method"] == "rank"


def test_paired_subjects_and_loso_are_disjoint():
    metadata = _metadata()
    assert paired_subjects(metadata) == [str(i) for i in range(1, 11)]
    first = paired_loso_folds(metadata, "1", validation_offset=1)
    second = paired_loso_folds(metadata, "2", validation_offset=1)
    assert set(first["train"]).isdisjoint(first["validation"] + first["test"])
    assert set(second["train"]).isdisjoint(second["validation"] + second["test"])
    assert first["test"] == ["1"]
    assert second["test"] == ["2"]


def test_robust_signal_conversion():
    values = np.array([0., 1., 0., 10., -10.])
    normalized = _robust_normalize(values)
    assert np.all(np.isfinite(normalized))
    assert np.max(np.abs(normalized)) <= 8
    h = np.sin(np.linspace(0, 10, 512))
    v = np.cos(np.linspace(0, 10, 512))
    h128, v128 = _resample_pair(h, v, 512, 128)
    assert len(h128) == 128
    assert len(v128) == 128


def test_prepare_cache_pipeline_with_fake_records(tmp_path, monkeypatch):
    root = tmp_path / "DROZY"
    (root / "psg").mkdir(parents=True)
    (root / "KSS.txt").write_text("1 3 7\n2 3 7\n", encoding="utf-8")
    records = []
    for subject in ("1", "2"):
        for test, kss in (("1", 3), ("3", 7)):
            (root / "psg" / f"{subject}-{test}.edf").write_bytes(b"fixture")
            time = np.arange(512 * 70) / 512
            records.append(DrozyRecord(
                subject=subject,
                test=test,
                eog_h=np.sin(2 * np.pi * 1.2 * time) + .5 * np.sin(2 * np.pi * 7 * time),
                eog_v=np.cos(2 * np.pi * .9 * time) + .5 * np.sin(2 * np.pi * 6 * time),
                cam_sync=np.array([]),
                timestamps=time,
                kss=kss,
                pvt_rt=np.array([]),
                sampling_rate=512,
            ))
    import eog_si_mset.data as data_module
    monkeypatch.setattr(data_module, "load_all_records", lambda *args, **kwargs: records)
    config = _config(tmp_path)
    config["data"].update({
        "drozy_root": str(root),
        "cache_dir": str(tmp_path / "cache"),
        "subjects": [1, 2],
        "exclude_tests": [],
        "kss_file": "KSS.txt",
        "channels": ["EOG-H", "EOG-V"],
        "min_fixations": 1,
        "min_saccades": 1,
        "preprocess": {
            "highpass": .5, "lowpass": 30., "filter_order": 4,
            "hampel_window": 51, "hampel_threshold": 3.,
        },
        "event_detection": {
            "velocity_threshold": "auto", "velocity_percentile": 70.,
            "min_fixation_ms": 100., "min_saccade_ms": 20.,
        },
    })
    result = prepare_cache(validate_config(config), tmp_path)
    assert result["n_windows"] > 0
    assert result["n_paired_subjects"] == 2
    assert (tmp_path / "cache" / "windows.npy").exists()
    metadata = pd.read_csv(tmp_path / "cache" / "metadata.csv")
    assert set(metadata.state) == {"alert", "drowsy"}
