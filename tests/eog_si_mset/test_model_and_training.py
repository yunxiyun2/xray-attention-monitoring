import numpy as np
import pandas as pd
import pytest
from pathlib import Path

torch = pytest.importorskip("torch")

from eog_si_mset.config import load_config
from eog_si_mset.model import GradientReversal, SIMSET
import eog_si_mset.train as train_module
from eog_si_mset.train import (
    PairBatchGenerator,
    calibrate_subject_scores,
    compute_loss,
    ensemble_seed_predictions,
    state_scores,
    train_fold,
)


def _model_config():
    return {
        "model": {
            "variant": "full",
            "d_model": 32,
            "transformer_layers": 1,
            "attention_heads": 4,
            "ffn_dim": 64,
            "sequence_tokens": 32,
            "dropout": 0.0,
        },
        "training": {
            "bce_weight": 1.0,
            "ordinal_weight": .5,
            "pair_weight": .25,
            "pair_margin": .25,
        },
    }


def test_gradient_reversal_direction():
    values = torch.tensor([2.0], requires_grad=True)
    output = GradientReversal.apply(values, 0.5)
    output.backward()
    assert values.grad.item() == pytest.approx(-0.5)


@pytest.mark.parametrize("variant", ["full", "single_scale", "no_adversarial", "binary_only"])
def test_model_forward_shapes(variant):
    model = SIMSET(
        n_subjects=4,
        variant=variant,
        d_model=32,
        transformer_layers=1,
        attention_heads=4,
        ffn_dim=64,
        sequence_tokens=32,
        dropout=0.0,
    )
    inputs = torch.randn(2, 3, 32 * 4)
    outputs = model(inputs, reversal_weight=.1)
    assert outputs["binary_logit"].shape == (2,)
    assert outputs["kss"].shape == (2,)
    assert outputs["embedding"].shape == (2, 32)
    if variant == "no_adversarial":
        assert outputs["subject_logits"] is None
    else:
        assert outputs["subject_logits"].shape == (2, 4)
    assert torch.isfinite(outputs["binary_logit"]).all()


def test_pair_batch_uses_original_indices_and_loss_terms():
    rows = []
    for subject in (1, 2):
        for state, label in (("alert", 0), ("drowsy", 1)):
            for window_idx in range(3):
                rows.append({
                    "subject": str(subject), "test": 1, "kss": 3 + 4 * label,
                    "label": label, "state": state, "window_idx": window_idx,
                })
    metadata = pd.DataFrame(rows)
    generator = PairBatchGenerator(metadata.loc[[1, 2, 3, 4, 5, 6]].copy(), ("1",), 1, 2, 0)
    batch = generator.sample(np.random.default_rng(0))
    assert set(batch["binary"]) == {0.0, 1.0}
    assert min(batch["indices"]) >= 1
    model = SIMSET(
        n_subjects=1, variant="full", d_model=32, transformer_layers=1,
        attention_heads=4, ffn_dim=64, sequence_tokens=32, dropout=0.0,
    )
    tensors = {
        "values": torch.randn(len(batch["indices"]), 3, 128),
        "binary": torch.from_numpy(batch["binary"]),
        "kss": torch.from_numpy(batch["kss"]),
        "subject_code": torch.from_numpy(batch["subject_code"]),
    }
    outputs = model(tensors["values"], reversal_weight=.1)
    losses = compute_loss(outputs, tensors, _model_config(), .1)
    assert all(torch.isfinite(value) for value in losses.values())


def test_one_fold_end_to_end_smoke(tmp_path):
    rows = []
    windows = []
    for subject in range(1, 11):
        for state, label, kss in (("alert", 0, 3), ("drowsy", 1, 7)):
            for window_idx in (0, 3, 6):
                rows.append({
                    "subject": str(subject), "test": 1, "kss": kss,
                    "label": label, "state": state, "window_idx": window_idx,
                })
                windows.append(np.random.default_rng(subject * 10 + label).normal(
                    label, 1, size=(3, 128)
                ).astype(np.float32))
    metadata = pd.DataFrame(rows)
    windows = np.stack(windows)
    config = _model_config()
    config["model"].update({"variant": "full", "sequence_tokens": 32})
    config["data"] = {"window_sec": 30.0, "stride_sec": 10.0, "primary_phase": 0}
    config["training"].update({
        "epochs": 1,
        "patience": 1,
        "batches_per_epoch": 1,
        "batch_subjects": 4,
        "windows_per_state": 2,
        "windows_per_intermediate": 0,
        "validation_windows": 2,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "noise_std": 0.0,
        "adversarial_weight": .1,
        "adversarial_warmup_epochs": 1,
        "amp": False,
        "seed": 1,
        "validation_mode": "training_subjects",
        "validation_windows_per_subject": 2,
    })
    config["output"] = {"save_checkpoints": False}
    result = train_fold(
        windows,
        metadata,
        fold_index=0,
        heldout_subject="1",
        config=config,
        root=tmp_path,
        device=torch.device("cpu"),
    )
    assert len(result["predictions"]) == 6
    assert result["predictions"].probability.between(0, 1).all()
    assert result["predictions"].subject.eq("1").all()


def test_calibration_uses_only_unlabelled_subject_scores():
    rows = []
    for subject in ("1", "2"):
        for index in range(8):
            rows.append({
                "subject": subject,
                "test": "1",
                "kss": 3 if index < 4 else 7,
                "label": 0 if index < 4 else 1,
                "state": "alert" if index < 4 else "drowsy",
                "window_idx": index,
                "probability": .1 + .1 * index,
                "kss_prediction": .2 + .05 * index,
            })
    frame = pd.DataFrame(rows)
    calibrated = calibrate_subject_scores(frame, "rank", min_windows=8)
    relabelled = frame.copy()
    relabelled["label"] = 1 - relabelled["label"]
    relabelled["state"] = relabelled.state.map({"alert": "drowsy", "drowsy": "alert"})
    recalibrated = calibrate_subject_scores(relabelled, "rank", min_windows=8)
    assert np.allclose(calibrated.score, recalibrated.score)
    assert set(state_scores(calibrated).state) == {"alert", "drowsy"}


def test_seed_ensemble_averages_logits():
    base = pd.DataFrame({
        "subject": ["1", "1"], "test": ["1", "1"], "kss": [3, 7],
        "label": [0, 1], "state": ["alert", "drowsy"], "window_idx": [0, 3],
        "kss_prediction": [.2, .8],
    })
    first = base.assign(probability=[.2, .8])
    second = base.assign(probability=[.8, .2])
    _, ensemble = ensemble_seed_predictions([(1, first), (2, second)])
    assert np.allclose(ensemble.probability, .5)
    assert ensemble.n_seeds.eq(2).all()


def test_v12_pipeline_with_stubbed_fold_training(tmp_path, monkeypatch):
    rows = []
    for subject in range(1, 11):
        for state, label, kss in (("alert", 0, 3), ("drowsy", 1, 7)):
            for window_idx in range(20):
                rows.append({
                    "subject": str(subject),
                    "test": 1,
                    "kss": kss,
                    "label": label,
                    "state": state,
                    "window_idx": window_idx * 3,
                })
    metadata = pd.DataFrame(rows)

    def fake_train_fold(windows, metadata, fold_index, heldout_subject, config, root,
                        device, training_seed=None):
        test = metadata.loc[metadata.subject.eq(str(heldout_subject))].copy()
        offset = int(heldout_subject) / 100
        test["probability"] = np.clip(
            .08 + .75 * test.label + offset + (training_seed % 7) / 1000, .001, .999
        )
        test["kss_prediction"] = test.probability
        history = pd.DataFrame([{
            "epoch": 1, "train_loss": 1., "validation_loss": .1, "learning_rate": 1e-3,
        }])
        return {
            "predictions": test,
            "history": history,
            "split": {"train": [str(i) for i in range(1, 11) if str(i) != str(heldout_subject)]},
            "validation_description": "stub",
            "best_epoch": 1,
            "best_validation_loss": .1,
        }

    monkeypatch.setattr(train_module, "load_cache", lambda path: (None, metadata))
    monkeypatch.setattr(train_module, "train_fold", fake_train_fold)
    project_root = Path(__file__).resolve().parents[2]
    config = load_config(project_root / "configs/eog_si_mset_v1_2.yaml")
    config["evaluation"]["bootstrap_iterations"] = 20
    config["evaluation"]["baseline_run"] = None
    config["output"]["root"] = str(tmp_path / "v12")
    result = train_module.run_training_v12(config, tmp_path)
    run_dir = Path(result["run_dir"])
    assert result["decision"]["decision"] == "not_evaluable"
    assert (run_dir / "calibration_summary.csv").exists()
    assert (run_dir / "bootstrap_samples.csv").exists()
    assert (run_dir / "figures/v12_performance.png").exists()
