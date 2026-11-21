"""Training, LOSO evaluation and bootstrap for SI-MSET."""
from copy import deepcopy
from pathlib import Path
import json
import platform
import random
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as functional
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
import yaml

from dual_factor.run_manifest import save_json
from eog_si_mset.config import config_hash
from eog_si_mset.data import (
    load_cache,
    package_code_hash,
    paired_loso_folds,
    paired_subjects,
    split_indices,
)
from eog_si_mset.model import SIMSET
from eog_si_mset.protocol import variant_flags


def resolve_device(requested):
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sample(values, count, rng):
    replace = len(values) < count
    return rng.choice(values, count, replace=replace)


class PairBatchGenerator:
    def __init__(self, metadata, subjects, batch_subjects, windows_per_state,
                 windows_per_intermediate=0):
        self.metadata = metadata
        self.subjects = sorted(str(subject) for subject in subjects)
        self.batch_subjects = min(batch_subjects, len(self.subjects))
        self.windows_per_state = windows_per_state
        self.windows_per_intermediate = windows_per_intermediate
        self.subject_codes = {subject: index for index, subject in enumerate(self.subjects)}
        self.indices = {}
        for subject in self.subjects:
            for state in ("alert", "drowsy"):
                positions = np.flatnonzero(
                    metadata.subject.astype(str).eq(subject).to_numpy()
                    & metadata.state.eq(state).to_numpy()
                )
                indices = metadata.index.to_numpy()[positions]
                if not len(indices):
                    raise ValueError(f"Missing {state} windows for training subject {subject}")
                self.indices[(subject, state)] = indices
            intermediate = metadata.index.to_numpy()[
                np.flatnonzero(
                    metadata.subject.astype(str).eq(subject).to_numpy()
                    & metadata.state.eq("intermediate").to_numpy()
                )
            ]
            self.indices[(subject, "intermediate")] = intermediate

    def sample(self, rng):
        chosen_subjects = rng.choice(
            self.subjects, self.batch_subjects, replace=False
        )
        indices, binary, kss, subject_code = [], [], [], []
        for subject in chosen_subjects:
            for state in ("alert", "drowsy"):
                sampled = _sample(
                    self.indices[(str(subject), state)],
                    self.windows_per_state,
                    rng,
                )
                indices.extend(sampled.tolist())
                binary.extend([0 if state == "alert" else 1] * len(sampled))
                kss.extend(self.metadata.loc[sampled, "kss"].to_numpy(float))
                subject_code.extend([self.subject_codes[str(subject)]] * len(sampled))
            intermediate = self.indices[(str(subject), "intermediate")]
            if self.windows_per_intermediate and len(intermediate):
                sampled = _sample(intermediate, self.windows_per_intermediate, rng)
                indices.extend(sampled.tolist())
                binary.extend([-1] * len(sampled))
                kss.extend(self.metadata.loc[sampled, "kss"].to_numpy(float))
                subject_code.extend([self.subject_codes[str(subject)]] * len(sampled))
        return {
            "indices": np.asarray(indices, dtype=np.int64),
            "binary": np.asarray(binary, dtype=np.float32),
            "kss": (np.asarray(kss, dtype=np.float32) - 1.0) / 8.0,
            "subject_code": np.asarray(subject_code, dtype=np.int64),
        }


def balanced_subject_indices(metadata, subject, windows_per_state, rng):
    indices, labels, kss = [], [], []
    subject_metadata = metadata.loc[metadata.subject.astype(str) == str(subject)]
    for state, label in (("alert", 0), ("drowsy", 1)):
        candidates = np.flatnonzero(subject_metadata.state.eq(state).to_numpy())
        absolute = subject_metadata.index.to_numpy()[candidates]
        if not len(absolute):
            raise ValueError(f"Validation subject {subject} lacks {state} windows")
        sampled = _sample(absolute, windows_per_state, rng)
        indices.extend(sampled.tolist())
        labels.extend([label] * len(sampled))
        kss.extend(metadata.loc[sampled, "kss"].to_numpy(float))
    return {
        "indices": np.asarray(indices, dtype=np.int64),
        "binary": np.asarray(labels, dtype=np.float32),
        "kss": (np.asarray(kss, dtype=np.float32) - 1.0) / 8.0,
        "subject_code": np.zeros(len(indices), dtype=np.int64),
    }


def _as_tensors(batch, windows, device):
    values = np.asarray(windows[batch["indices"]], dtype=np.float32)
    return {
        "values": torch.from_numpy(values).to(device),
        "binary": torch.from_numpy(batch["binary"]).to(device),
        "kss": torch.from_numpy(batch["kss"]).to(device),
        "subject_code": torch.from_numpy(batch["subject_code"]).to(device),
    }


def compute_loss(outputs, batch, config, adversarial_scale):
    training = config["training"]
    known = batch["binary"] >= 0
    binary_loss = functional.binary_cross_entropy_with_logits(
        outputs["binary_logit"][known], batch["binary"][known]
    )
    ordinal_loss = functional.mse_loss(outputs["kss"], batch["kss"])
    pair_terms = []
    for code in torch.unique(batch["subject_code"]):
        mask = batch["subject_code"] == code
        alert = mask & (batch["binary"] == 0)
        drowsy = mask & (batch["binary"] == 1)
        if alert.any() and drowsy.any():
            gap = outputs["binary_logit"][drowsy].mean() - outputs["binary_logit"][alert].mean()
            pair_terms.append(functional.relu(training["pair_margin"] - gap))
    pair_loss = torch.stack(pair_terms).mean() if pair_terms else outputs["kss"].new_zeros(())
    adversarial_loss = outputs["kss"].new_zeros(())
    if outputs["subject_logits"] is not None and adversarial_scale > 0:
        adversarial_loss = functional.cross_entropy(
            outputs["subject_logits"], batch["subject_code"]
        )
    if config["model"]["variant"] == "binary_only":
        ordinal_loss = ordinal_loss.detach() * 0
        pair_loss = pair_loss.detach() * 0
    total = (
        training["bce_weight"] * binary_loss
        + training["ordinal_weight"] * ordinal_loss
        + training["pair_weight"] * pair_loss
        + adversarial_loss
    )
    return {
        "total": total,
        "binary": binary_loss,
        "ordinal": ordinal_loss,
        "pair": pair_loss,
        "adversarial": adversarial_loss,
    }


def _loss_float(losses):
    return {name: float(value.detach().cpu()) for name, value in losses.items()}


def predict_indices(model, windows, indices, device, amp_enabled, batch_size=256):
    model.eval()
    probabilities, predictions = [], []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            selected = indices[start:start + batch_size]
            values = torch.from_numpy(np.asarray(windows[selected], dtype=np.float32)).to(device)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                output = model(values)
            probabilities.append(torch.sigmoid(output["binary_logit"]).float().cpu().numpy())
            predictions.append(output["kss"].float().cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(predictions)


def _state_predictions(predictions):
    required = {"subject", "state", "label", "probability"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    table = (
        predictions.groupby(["subject", "state", "label"], as_index=False)
        .agg(probability=("probability", "mean"), kss_prediction=("kss_prediction", "mean"),
             n_windows=("probability", "size"))
    )
    counts = table.groupby("subject").size()
    if not counts.eq(2).all():
        raise ValueError("State aggregation did not produce two rows per subject")
    return table


def _bootstrap(subject_states, baseline_states, iterations, seed):
    subjects = sorted(subject_states.subject.unique())
    rng = np.random.default_rng(seed)
    rows = []
    deep_map = {subject: frame for subject, frame in subject_states.groupby("subject")}
    base_map = (
        {subject: frame for subject, frame in baseline_states.groupby("subject")}
        if baseline_states is not None else {}
    )
    for iteration in range(iterations):
        sampled = rng.choice(subjects, len(subjects), replace=True)
        deep_sample = pd.concat([deep_map[subject] for subject in sampled], ignore_index=True)
        deep_auc = roc_auc_score(deep_sample.label, deep_sample.probability)
        row = {"iteration": iteration, "deep_auc": deep_auc}
        if baseline_states is not None:
            baseline_sample = pd.concat([base_map[subject] for subject in sampled], ignore_index=True)
            baseline_auc = roc_auc_score(baseline_sample.label, baseline_sample.probability)
            row.update(baseline_auc=baseline_auc, delta_auc=deep_auc - baseline_auc)
        rows.append(row)
    return pd.DataFrame(rows)


def _interval(values):
    return np.quantile(np.asarray(values, dtype=float), [0.025, 0.975]).tolist()


def _decision(deep_auc, deep_ci, delta_auc, delta_ci, config, baseline_available):
    evaluation = config["evaluation"]
    if not baseline_available:
        return "not_evaluable"
    if (
        deep_auc >= evaluation["min_auc"]
        and deep_ci[0] > evaluation["min_auc_lower"]
        and delta_auc >= evaluation["min_delta_auc"]
        and delta_ci[0] > evaluation["min_delta_lower"]
    ):
        return "supported"
    if deep_ci[1] < 0.50:
        return "below_chance"
    return "inconclusive"


def load_baseline_states(path):
    if path is None:
        return None
    path = Path(path)
    prediction_path = path / "fold_predictions.csv" if path.is_dir() else path
    if not prediction_path.exists():
        return None
    predictions = pd.read_csv(prediction_path)
    required = {"model", "subject", "state", "label", "probability"}
    if not required.issubset(predictions.columns):
        raise ValueError("Baseline prediction schema is incompatible")
    global_predictions = predictions.loc[predictions.model == "global"].copy()
    if not len(global_predictions):
        raise ValueError("Baseline run has no global predictions")
    global_predictions["subject"] = global_predictions["subject"].astype(str)
    global_predictions["state"] = global_predictions["state"].astype(str)
    return (
        global_predictions.groupby(["subject", "state", "label"], as_index=False)
        .probability.mean()
    )


def _write_figures(run, state_predictions, baseline_states, bootstrap, history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    directory = run / "figures"
    directory.mkdir()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for frame, name, color in (
        (state_predictions, "SI-MSET", "#246591"),
        (baseline_states, "v7 global", "#8c5f38"),
    ):
        if frame is None:
            continue
        fpr, tpr, _ = roc_curve(frame.label, frame.probability)
        auc = roc_auc_score(frame.label, frame.probability)
        axes[0].plot(fpr, tpr, label=f"{name} ({auc:.3f})", color=color)
    axes[0].plot([0, 1], [0, 1], "--", color="#9aa5ad")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate",
                title="Leave-one-subject-out state ROC")
    axes[0].legend(frameon=False)
    for subject, fold_history in history.groupby("heldout_subject"):
        axes[1].plot(fold_history.epoch, fold_history.validation_loss,
                     label=subject, alpha=0.7)
    axes[1].set(xlabel="Epoch", ylabel="Validation loss", title="Per-fold optimization")
    axes[1].legend(frameon=False, fontsize=7, ncol=2)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(directory / "loso_performance.png", dpi=220)
    fig.savefig(directory / "loso_performance.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    intervals = []
    labels = []
    if baseline_states is not None:
        intervals.append((
            _interval(bootstrap.baseline_auc),
            roc_auc_score(baseline_states.label, baseline_states.probability),
        ))
        labels.append("v7 global")
    intervals.append((
        _interval(bootstrap.deep_auc),
        roc_auc_score(state_predictions.label, state_predictions.probability),
    ))
    labels.append("SI-MSET")
    for position, ((low, high), estimate) in enumerate(intervals):
        ax.plot([low, high], [position, position], color="#52768e", linewidth=3)
        ax.scatter([estimate], [position], color="#246591", zorder=3)
    ax.set(yticks=range(len(labels)), yticklabels=labels, xlim=(0, 1),
           xlabel="Subject-state AUC with cluster bootstrap 95% CI")
    ax.axvline(0.5, linestyle="--", color="#ba5545")
    ax.set_title("Out-of-fold performance")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.15)
    fig.tight_layout()
    fig.savefig(directory / "auc_intervals.png", dpi=220)
    fig.savefig(directory / "auc_intervals.pdf")
    plt.close(fig)


def _write_report(run, result, folds, config):
    lines = [
        "# SI-MSET 训练报告",
        "",
        f"- 变体：`{config['model']['variant']}`",
        f"- 判定：**{result['decision']}**",
        f"- 被试状态 AUC：{result['deep_auc']:.3f} "
        f"（95% CI {result['deep_ci'][0]:.3f}–{result['deep_ci'][1]:.3f}）",
    ]
    if result.get("baseline_auc") is not None:
        lines.extend([
            f"- v7 global AUC：{result['baseline_auc']:.3f}",
            f"- 配对 ΔAUC：{result['delta_auc']:.3f} "
            f"（95% CI {result['delta_ci'][0]:.3f}–{result['delta_ci'][1]:.3f}）",
        ])
    lines.extend([
        "",
        "## 分折状态",
        "",
        "| Held-out subject | Best epoch | Validation loss | Status |",
        "|---|---:|---:|---|",
    ])
    for row in folds.itertuples():
        lines.append(
            f"| {row.heldout_subject} | {row.best_epoch or '-'} | "
            f"{row.best_validation_loss or '-'} | {row.status} |"
        )
    lines.extend([
        "",
        "## 解释边界",
        "",
        "- 评价单位是被试状态；窗口不当作独立被试。",
        "- 所有折均保持 held-out subject 完全隔离。",
        "- 结果只适用于 DROZY、当前预处理和预注册模型，不外推到设备、任务或临床场景。",
        "- 消融变体需要单独运行；本次报告不自动推断各模块有效。",
    ])
    Path(run / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_fold(windows, metadata, fold_index, heldout_subject, config, root, device):
    history = []
    rng = np.random.default_rng(config["training"]["seed"] + fold_index)
    splits = paired_loso_folds(metadata, heldout_subject, validation_offset=fold_index + 1)
    train_indices = split_indices(metadata, splits["train"])
    validation_indices = split_indices(metadata, splits["validation"])
    test_indices = split_indices(metadata, splits["test"])
    phase_step = int(round(
        config["data"]["window_sec"] / config["data"]["stride_sec"]
    ))
    test_indices = test_indices[
        metadata.loc[test_indices, "window_idx"].astype(int).mod(phase_step).eq(
            config["data"]["primary_phase"]
        ).to_numpy()
    ]
    test_indices = test_indices[
        metadata.loc[test_indices, "label"].to_numpy() >= 0
    ]
    if not len(train_indices) or not len(validation_indices) or not len(test_indices):
        raise ValueError(f"Empty split for held-out subject {heldout_subject}")

    model = SIMSET(
        n_subjects=len(splits["train"]),
        variant=config["model"]["variant"],
        d_model=config["model"]["d_model"],
        transformer_layers=config["model"]["transformer_layers"],
        attention_heads=config["model"]["attention_heads"],
        ffn_dim=config["model"]["ffn_dim"],
        sequence_tokens=config["model"]["sequence_tokens"],
        dropout=config["model"]["dropout"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["epochs"]
    )
    generator = PairBatchGenerator(
        metadata.loc[train_indices].copy(),
        splits["train"],
        config["training"]["batch_subjects"],
        config["training"]["windows_per_state"],
            config["training"]["windows_per_intermediate"],
    )
    amp_enabled = bool(config["training"]["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler(device="cuda", enabled=amp_enabled)
    best_loss, best_epoch, best_state, stale = float("inf"), 0, None, 0
    for epoch in range(1, config["training"]["epochs"] + 1):
        model.train()
        progress = (epoch - 1) / max(config["training"]["epochs"] - 1, 1)
        adversarial_scale = (
            config["training"]["adversarial_weight"]
            * min(1.0, epoch / config["training"]["adversarial_warmup_epochs"])
            if variant_flags(config["model"]["variant"])["use_adversarial"] else 0.0
        )
        train_losses = []
        for _ in range(config["training"]["batches_per_epoch"]):
            batch = generator.sample(rng)
            tensors = _as_tensors(batch, windows, device)
            if training_noise := config["training"]["noise_std"]:
                tensors["values"] = tensors["values"] + torch.randn_like(tensors["values"]) * training_noise
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                outputs = model(tensors["values"], reversal_weight=adversarial_scale)
                losses = compute_loss(outputs, tensors, config, adversarial_scale)
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(losses["total"].detach().cpu()))
        if not np.isfinite(train_losses).all():
            raise ValueError(f"Nonfinite training loss in fold {fold_index}")
        scheduler.step()
        validation_batch = balanced_subject_indices(
            metadata,
            splits["validation"][0],
            config["training"]["validation_windows"],
            rng,
        )
        tensors = _as_tensors(validation_batch, windows, device)
        model.eval()
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp_enabled
        ):
            outputs = model(tensors["values"], reversal_weight=0.0)
            validation_losses = compute_loss(outputs, tensors, config, 0.0)
        validation_loss = float(validation_losses["total"].detach().cpu())
        if not np.isfinite(validation_loss):
            raise ValueError(f"Nonfinite validation loss in fold {fold_index}")
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "validation_loss": validation_loss,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        })
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_epoch = epoch
            stale = 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            stale += 1
        if stale >= config["training"]["patience"]:
            print(
                f"[fold {fold_index}] held-out={heldout_subject} early stop "
                f"epoch={epoch} best_epoch={best_epoch} best_val={best_loss:.4f}",
                flush=True,
            )
            break
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"[fold {fold_index}] held-out={heldout_subject} epoch={epoch} "
                f"train={np.mean(train_losses):.4f} val={validation_loss:.4f}",
                flush=True,
            )
    if best_state is None:
        raise ValueError(f"No finite validation checkpoint for fold {fold_index}")
    model.load_state_dict(best_state)
    if config["output"]["save_checkpoints"]:
        torch.save(best_state, root / f"fold_{fold_index:02d}_subject_{heldout_subject}.pt")
    probabilities, kss_prediction = predict_indices(
        model,
        windows,
        test_indices,
        device,
        amp_enabled,
    )
    predictions = metadata.loc[test_indices, ["subject", "test", "kss", "label", "state", "window_idx"]].copy()
    predictions["probability"] = probabilities
    predictions["kss_prediction"] = kss_prediction
    return {
        "model": model,
        "predictions": predictions,
        "history": pd.DataFrame(history),
        "split": splits,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
    }


def run_training(config, root, variant=None):
    torch.manual_seed(config["training"]["seed"])
    device = resolve_device(config["training"]["device"])
    if variant is not None:
        if variant not in ("full", "single_scale", "no_adversarial", "binary_only"):
            raise ValueError(f"Unknown variant: {variant}")
        config = deepcopy(config)
        config["model"]["variant"] = variant
    root = Path(root)
    cache_dir = (
        Path(config["data"]["cache_dir"])
        if Path(config["data"]["cache_dir"]).is_absolute()
        else root / config["data"]["cache_dir"]
    )
    windows, metadata = load_cache(cache_dir)
    paired = paired_subjects(metadata)
    output_root = Path(config["output"]["root"])
    if not output_root.is_absolute():
        output_root = root / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    run = output_root / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    )
    run.mkdir()
    (run / "config_used.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    save_json(run / "manifest.json", {
        "protocol": config["protocol"],
        "cache_dir": str(cache_dir),
        "paired_subjects": paired,
        "config_sha256": config_hash(config),
        "code_sha256": package_code_hash(root),
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "platform": platform.platform(),
    })

    fold_rows, prediction_frames, histories = [], [], []
    for fold_index, heldout_subject in enumerate(paired):
        print(
            f"[fold {fold_index + 1}/{len(paired)}] held-out subject {heldout_subject}",
            flush=True,
        )
        seed_everything(config["training"]["seed"] + fold_index)
        try:
            result = train_fold(
                windows, metadata, fold_index, heldout_subject, config, run, device
            )
            predictions = result["predictions"]
            prediction_frames.append(predictions)
            histories.append(result["history"].assign(heldout_subject=heldout_subject))
            result["history"].to_csv(run / f"fold_{fold_index:02d}_history.csv", index=False)
            fold_rows.append({
                "fold": fold_index,
                "heldout_subject": heldout_subject,
                "validation_subject": result["split"]["validation"][0],
                "train_subjects": "|".join(result["split"]["train"]),
                "best_epoch": result["best_epoch"],
                "best_validation_loss": result["best_validation_loss"],
                "status": "ok",
                "reason": "",
            })
        except Exception as error:
            fold_rows.append({
                "fold": fold_index,
                "heldout_subject": heldout_subject,
                "validation_subject": "",
                "train_subjects": "",
                "best_epoch": None,
                "best_validation_loss": None,
                "status": "failed",
                "reason": f"{type(error).__name__}: {error}",
            })
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(run / "folds.csv", index=False)
    if len(prediction_frames) != len(paired):
        save_json(run / "result.json", {
            "decision": "not_evaluable",
            "reason": "At least one LOSO fold failed",
        })
        return {"run_dir": str(run), "decision": "not_evaluable"}

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(run / "window_predictions.csv", index=False)
    state_predictions = _state_predictions(predictions)
    state_predictions.to_csv(run / "state_predictions.csv", index=False)
    baseline_path = config["evaluation"]["baseline_run"]
    if baseline_path and not Path(baseline_path).is_absolute():
        baseline_path = str(root / baseline_path)
    baseline_states = load_baseline_states(baseline_path)
    if baseline_states is not None:
        if set(state_predictions.subject) != set(baseline_states.subject):
            raise ValueError("SI-MSET and baseline subject sets differ")
        baseline_states = baseline_states.set_index(["subject", "state", "label"]).loc[
            state_predictions.set_index(["subject", "state", "label"]).index
        ].reset_index()
    bootstrap = _bootstrap(
        state_predictions,
        baseline_states,
        config["evaluation"]["bootstrap_iterations"],
        config["evaluation"]["seed"],
    )
    bootstrap.to_csv(run / "bootstrap_samples.csv", index=False)
    deep_auc = float(roc_auc_score(state_predictions.label, state_predictions.probability))
    deep_ci = _interval(bootstrap.deep_auc)
    baseline_auc = (
        float(roc_auc_score(baseline_states.label, baseline_states.probability))
        if baseline_states is not None else None
    )
    delta_auc = deep_auc - baseline_auc if baseline_auc is not None else None
    delta_ci = _interval(bootstrap.delta_auc) if "delta_auc" in bootstrap else None
    decision = _decision(
        deep_auc, deep_ci, delta_auc, delta_ci, config, baseline_states is not None
    )
    result = {
        "decision": decision,
        "variant": config["model"]["variant"],
        "n_subjects": int(state_predictions.subject.nunique()),
        "deep_auc": deep_auc,
        "deep_ci": deep_ci,
        "baseline_auc": baseline_auc,
        "delta_auc": delta_auc,
        "delta_ci": delta_ci,
        "window_auc": float(roc_auc_score(predictions.label, predictions.probability)),
        "average_precision": float(average_precision_score(state_predictions.label, state_predictions.probability)),
        "brier": float(brier_score_loss(state_predictions.label, state_predictions.probability)),
        "log_loss": float(log_loss(state_predictions.label, np.clip(state_predictions.probability, 1e-12, 1 - 1e-12))),
        "bootstrap_iterations": int(config["evaluation"]["bootstrap_iterations"]),
        "baseline_available": baseline_states is not None,
    }
    save_json(run / "result.json", result)
    summary = pd.DataFrame([
        {"metric": "deep_auc", "estimate": deep_auc, "ci_low": deep_ci[0], "ci_high": deep_ci[1]},
        {"metric": "window_auc", "estimate": result["window_auc"], "ci_low": None, "ci_high": None},
        {"metric": "average_precision", "estimate": result["average_precision"], "ci_low": None, "ci_high": None},
        {"metric": "brier", "estimate": result["brier"], "ci_low": None, "ci_high": None},
        {"metric": "log_loss", "estimate": result["log_loss"], "ci_low": None, "ci_high": None},
    ])
    if delta_auc is not None:
        summary = pd.concat([summary, pd.DataFrame([{
            "metric": "delta_auc_vs_v7_global",
            "estimate": delta_auc,
            "ci_low": delta_ci[0],
            "ci_high": delta_ci[1],
        }])], ignore_index=True)
    summary.to_csv(run / "metrics.csv", index=False)
    history = pd.concat(histories, ignore_index=True)
    _write_figures(run, state_predictions, baseline_states, bootstrap, history)
    _write_report(run, result, folds, config)
    return {"run_dir": str(run), "decision": result}
