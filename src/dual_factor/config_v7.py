"""Validation for the v7 predictive-validity configuration."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import yaml


def validate_config(raw):
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a mapping")
    config = deepcopy(raw)
    if config.get("protocol") != "v7.0":
        raise ValueError("Only protocol v7.0 is supported")

    required = {"input", "analysis", "model", "bootstrap", "output"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")

    source = config["input"]
    if not isinstance(source.get("path"), str) or not source["path"]:
        raise ValueError("Input path is required")
    for key in ("window_sec", "stride_sec", "primary_phase", "alert_max", "drowsy_min",
                "min_pairs", "min_state_windows"):
        if key not in config["analysis"]:
            raise ValueError(f"Missing analysis setting: {key}")
    analysis = config["analysis"]
    if analysis["window_sec"] <= 0 or analysis["stride_sec"] <= 0:
        raise ValueError("Window and stride must be positive")
    ratio = analysis["window_sec"] / analysis["stride_sec"]
    if not ratio.is_integer() or ratio < 1:
        raise ValueError("Window/stride must have an integer ratio")
    if not 0 <= analysis["primary_phase"] < int(ratio):
        raise ValueError("Primary phase is outside the nonoverlap grid")
    if not 1 <= analysis["alert_max"] < analysis["drowsy_min"] <= 9:
        raise ValueError("Invalid KSS thresholds")
    if analysis["min_pairs"] < 2 or analysis["min_state_windows"] < 1:
        raise ValueError("Invalid minimum sample thresholds")

    model = config["model"]
    if model.get("type") != "logistic":
        raise ValueError("v7 permits only the fixed logistic model")
    if not isinstance(model.get("regularization_c"), (int, float)) or model["regularization_c"] <= 0:
        raise ValueError("Regularization C must be positive")
    if not isinstance(model.get("max_iter"), int) or model["max_iter"] < 1:
        raise ValueError("max_iter must be a positive integer")
    if not isinstance(model.get("seed"), int) or model["seed"] < 0:
        raise ValueError("Model seed must be a nonnegative integer")

    bootstrap = config["bootstrap"]
    if not isinstance(bootstrap.get("iterations"), int) or bootstrap["iterations"] < 1:
        raise ValueError("Bootstrap iterations must be positive")
    if not isinstance(bootstrap.get("seed"), int) or bootstrap["seed"] < 0:
        raise ValueError("Bootstrap seed must be nonnegative")

    output = config["output"]
    if not isinstance(output.get("root"), str) or not output["root"]:
        raise ValueError("Output root is required")
    if output.get("overwrite") is not False:
        raise ValueError("Historical output overwrite is prohibited")
    return config


def load_config(path):
    return validate_config(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def config_hash(config):
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
