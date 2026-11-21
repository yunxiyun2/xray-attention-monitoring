"""Strict YAML configuration for SI-MSET."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import yaml

from eog_si_mset.protocol import PROTOCOL, VARIANTS


def validate_config(raw):
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a mapping")
    config = deepcopy(raw)
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"Only {PROTOCOL} is supported")
    for section in ("data", "model", "training", "evaluation", "output"):
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"Missing configuration section: {section}")

    data = config["data"]
    for key in ("drozy_root", "cache_dir", "subjects", "exclude_tests", "channels", "kss_file",
                "target_fs", "window_sec", "stride_sec", "primary_phase",
                "preprocess", "event_detection",
                "min_fixations", "min_saccades", "alert_max", "drowsy_min"):
        if key not in data:
            raise ValueError(f"Missing data setting: {key}")
    if not str(data["drozy_root"]).strip() or not str(data["cache_dir"]).strip():
        raise ValueError("DROZY root and cache directory are required")
    if data["target_fs"] <= 0 or data["window_sec"] <= 0 or data["stride_sec"] <= 0:
        raise ValueError("Sampling and window settings must be positive")
    if not float(data["target_fs"]).is_integer():
        raise ValueError("target_fs must be an integer")
    ratio = data["window_sec"] / data["stride_sec"]
    if not ratio.is_integer() or ratio < 1:
        raise ValueError("Window/stride must have a positive integer ratio")
    if not 0 <= data["primary_phase"] < int(ratio):
        raise ValueError("primary_phase is outside the nonoverlap grid")
    if not 1 <= data["alert_max"] < data["drowsy_min"] <= 9:
        raise ValueError("Invalid KSS thresholds")
    if data["min_fixations"] < 1 or data["min_saccades"] < 1:
        raise ValueError("Event thresholds must be positive")
    if not data["subjects"] or not set(data["channels"]) >= {"EOG-H", "EOG-V"}:
        raise ValueError("Subjects and both EOG channels are required")
    preprocess = data["preprocess"]
    for key in ("highpass", "lowpass", "filter_order", "hampel_window", "hampel_threshold"):
        if key not in preprocess:
            raise ValueError(f"Missing preprocessing setting: {key}")
    if not 0 < preprocess["highpass"] < preprocess["lowpass"]:
        raise ValueError("Invalid preprocessing filter band")
    if preprocess["hampel_window"] < 1 or preprocess["hampel_window"] % 2 != 1:
        raise ValueError("Hampel window must be a positive odd integer")
    event_detection = data["event_detection"]
    for key in ("velocity_threshold", "velocity_percentile", "min_fixation_ms", "min_saccade_ms"):
        if key not in event_detection:
            raise ValueError(f"Missing event setting: {key}")
    if event_detection["velocity_threshold"] != "auto" or not 0 < event_detection["velocity_percentile"] < 100:
        raise ValueError("Only automatic I-VT velocity thresholds are supported")

    model = config["model"]
    if model.get("variant") not in VARIANTS:
        raise ValueError("Invalid model variant")
    for key in ("d_model", "transformer_layers", "attention_heads", "ffn_dim", "sequence_tokens"):
        if not isinstance(model.get(key), int) or model[key] < 1:
            raise ValueError(f"Invalid model setting: {key}")
    if model["d_model"] % model["attention_heads"]:
        raise ValueError("d_model must be divisible by attention_heads")
    if not 0 <= model.get("dropout", -1) < 1:
        raise ValueError("dropout must be in [0,1)")

    training = config["training"]
    for key in ("epochs", "patience", "batches_per_epoch", "batch_subjects",
                "windows_per_state", "validation_windows", "seed"):
        if not isinstance(training.get(key), int) or training[key] < 1:
            raise ValueError(f"Invalid training setting: {key}")
    if training["batch_subjects"] < 2:
        raise ValueError("At least two subjects are needed for adversarial training")
    if not isinstance(training.get("windows_per_intermediate"), int) or training["windows_per_intermediate"] < 0:
        raise ValueError("windows_per_intermediate must be a nonnegative integer")
    if training.get("optimizer") != "adamw":
        raise ValueError("Only AdamW is supported")
    if training.get("learning_rate", 0) <= 0 or training.get("weight_decay", -1) < 0:
        raise ValueError("Invalid optimizer hyperparameters")
    if training.get("device") not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu or cuda")
    for key in ("bce_weight", "ordinal_weight", "pair_weight", "adversarial_weight"):
        if training.get(key, -1) < 0:
            raise ValueError(f"Invalid loss weight: {key}")
    if training.get("pair_margin", -1) < 0:
        raise ValueError("pair_margin must be nonnegative")
    if not 0 <= training.get("noise_std", -1) < 1:
        raise ValueError("noise_std must be in [0,1)")
    if training.get("adversarial_warmup_epochs", 0) < 1:
        raise ValueError("adversarial_warmup_epochs must be positive")
    if not isinstance(training.get("amp"), bool):
        raise ValueError("amp must be boolean")

    evaluation = config["evaluation"]
    for key in ("bootstrap_iterations", "seed"):
        if not isinstance(evaluation.get(key), int) or evaluation[key] < 1:
            raise ValueError(f"Invalid evaluation setting: {key}")
    for key in ("min_auc", "min_auc_lower", "min_delta_auc", "min_delta_lower"):
        if not isinstance(evaluation.get(key), (int, float)):
            raise ValueError(f"Invalid evaluation threshold: {key}")
    baseline = evaluation.get("baseline_run")
    if baseline is not None and not isinstance(baseline, str):
        raise ValueError("baseline_run must be a path string or null")

    output = config["output"]
    if not isinstance(output.get("root"), str) or not output["root"]:
        raise ValueError("Output root is required")
    if output.get("overwrite") is not False:
        raise ValueError("Output overwrite is prohibited")
    if output.get("save_checkpoints") not in (True, False):
        raise ValueError("save_checkpoints must be boolean")
    return config


def load_config(path):
    return validate_config(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def config_hash(config):
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
