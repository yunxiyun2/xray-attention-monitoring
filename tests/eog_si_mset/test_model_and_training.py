import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from eog_si_mset.config import load_config
from eog_si_mset.model import GradientReversal, SIMSET
from eog_si_mset.train import PairBatchGenerator, compute_loss, train_fold


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
