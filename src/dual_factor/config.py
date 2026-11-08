"""Strict v5 protocol configuration; no decorative or silently ignored keys."""
from copy import deepcopy
from pathlib import Path
import json
import hashlib
import math
import re
import yaml
from dual_factor.protocol import RETAINED, CRITERIA, STABILITY

DEFAULT_CONFIG = {
    "protocol_version": "v5.0",
    "input": {"mode": "matrix", "path": "results/dual_factor_efa/indicator_matrix.csv",
              "source_config": "results/dual_factor_efa/config_used.yaml"},
    "windowing": {"window_sec": 30., "stride_sec": 10., "primary_phase": 0,
                  "sensitivity_phases": [1, 2], "min_fixations": 5, "min_saccades": 3},
    "structure": {"retained": list(RETAINED), "extraction": "paf", "rotation": "promax",
                  "paf_tol": 1e-6, "paf_max_iter": 500,
                  "weighting": "equal_subject_equal_record", "criteria": CRITERIA.copy(),
                  "stability": STABILITY.copy()},
    "parallel_analysis": {"iterations": 1000, "percentile": 95., "seed": 2026110801,
                          "min_segment": 8, "min_subjects": 10},
    "cluster_bootstrap": {"iterations": 2000, "seed": 2026110802},
    "signal_detection": {"iterations": 2000, "seed": 2026110806, "coherence_min": .5, "eigen_floor": 1.},
    "validity": {"alert_max": 4, "drowsy_min": 6, "iterations": 2000, "seed": 2026110803},
    "preprocess": {"highpass": .5, "lowpass": 30., "filter_order": 4,
                   "hampel_window": 51, "hampel_threshold": 3.},
    "event_detection": {"velocity_threshold": "auto", "velocity_percentile": 70.,
                        "min_fixation_ms": 100., "min_saccade_ms": 20.},
    "spatial_grid": {"bins": 5},
    "data": {"subjects": list(range(1, 15)), "exclude_tests": ["7-1"],
             "channels": ["EOG-H", "EOG-V"], "kss_file": "KSS.txt"},
    "stages": {"efa": True, "cfa": False, "prediction": False, "external": False, "invariance": False},
    "output": {"root": "results/dual_factor_efa_v5", "overwrite": False},
}


def _shape(value, reference, location="config"):
    if not isinstance(value, dict) or set(value) != set(reference):
        raise ValueError(f"Missing/unknown configuration keys at {location}")
    for key, default in reference.items():
        if isinstance(default, dict):
            _shape(value[key], default, f"{location}.{key}")
        elif isinstance(default, bool):
            if not isinstance(value[key], bool):
                raise ValueError(f"{location}.{key} must be boolean")
        elif isinstance(default, (int, float)) and (isinstance(value[key], bool) or not isinstance(value[key], (int, float))):
            raise ValueError(f"{location}.{key} must be numeric")
        elif isinstance(default, (int, float)) and not math.isfinite(value[key]):
            raise ValueError(f"{location}.{key} must be finite")
        elif isinstance(default, (str, list)) and not isinstance(value[key], type(default)):
            raise ValueError(f"Invalid type at {location}.{key}")


def validate_config(config):
    _shape(config, DEFAULT_CONFIG)
    if config["protocol_version"] != "v5.0" or config["input"]["mode"] not in ("matrix", "raw"):
        raise ValueError("Unsupported protocol/input mode")
    window = config["windowing"]
    if (window["window_sec"], window["stride_sec"], window["primary_phase"], window["sensitivity_phases"]) != (30., 10., 0, [1, 2]):
        raise ValueError("v5 fixes a 30s/10s sampling grid with phases 0,1,2")
    structure = config["structure"]
    if structure["retained"] != list(RETAINED) or structure["extraction"] != "paf" or structure["rotation"] != "promax":
        raise ValueError("v5 fixes the six-indicator PAF/Promax candidate")
    if structure["weighting"] != "equal_subject_equal_record":
        raise ValueError("Unsupported structural weighting")
    for group in (structure["criteria"], structure["stability"]):
        if any(not 0 < float(v) < 1 for v in group.values()):
            raise ValueError("Criterion/probability values must be in (0,1)")
    for section in ("parallel_analysis", "cluster_bootstrap", "validity", "signal_detection"):
        if not isinstance(config[section]["iterations"], int) or config[section]["iterations"] < 1:
            raise ValueError("Iteration counts must be positive integers")
        if not isinstance(config[section]["seed"], int) or config[section]["seed"] < 0:
            raise ValueError("Seeds must be nonnegative integers")
    if not 0 < config["parallel_analysis"]["percentile"] < 100:
        raise ValueError("Invalid PA percentile")
    if not 0 < config["signal_detection"]["coherence_min"] < 1 or config["signal_detection"]["eigen_floor"] <= 0:
        raise ValueError("Invalid observable-signal diagnostic thresholds")
    if not 1 <= config["validity"]["alert_max"] < config["validity"]["drowsy_min"] <= 9:
        raise ValueError("Invalid KSS class thresholds")
    if any(config["stages"][s] for s in ("cfa", "prediction", "external", "invariance")):
        raise ValueError("Later stages require separate implementations and real EFA acceptance; v5 EFA never launches them")
    if not config["stages"]["efa"] or config["output"]["overwrite"]:
        raise ValueError("EFA stage must be enabled; historical overwrite is prohibited")
    if structure["paf_tol"] <= 0 or not isinstance(structure["paf_max_iter"], int) or structure["paf_max_iter"] < 1:
        raise ValueError("Invalid PAF convergence settings")
    for key in ("min_segment", "min_subjects"):
        if not isinstance(config["parallel_analysis"][key], int) or config["parallel_analysis"][key] < 2:
            raise ValueError("Invalid dependence-reference sample threshold")
    for key in ("min_fixations", "min_saccades"):
        if not isinstance(window[key], int) or window[key] < 1:
            raise ValueError("Invalid event-count threshold")
    pp = config["preprocess"]
    if not 0 < pp["highpass"] < pp["lowpass"] or pp["hampel_threshold"] <= 0:
        raise ValueError("Invalid filter/MAD settings")
    if not isinstance(pp["hampel_window"], int) or pp["hampel_window"] < 1 or pp["hampel_window"] % 2 != 1:
        raise ValueError("Hampel window must be a positive odd integer")
    if not isinstance(pp["filter_order"], int) or pp["filter_order"] < 1:
        raise ValueError("Filter order must be a positive integer")
    ed = config["event_detection"]
    if ed["velocity_threshold"] != "auto" or not 0 < ed["velocity_percentile"] < 100 or min(ed["min_fixation_ms"], ed["min_saccade_ms"]) <= 0:
        raise ValueError("Invalid event settings")
    if not isinstance(config["spatial_grid"]["bins"], int) or config["spatial_grid"]["bins"] < 2:
        raise ValueError("Invalid spatial bins")
    subjects = config["data"]["subjects"]
    if not subjects or len(set(subjects)) != len(subjects) or any(not isinstance(s, int) or isinstance(s, bool) or not 1 <= s <= 14 for s in subjects):
        raise ValueError("Invalid DROZY subjects")
    if not {"EOG-H", "EOG-V"} <= set(config["data"]["channels"]):
        raise ValueError("Both EOG channels required")
    if any(not isinstance(x, str) or not re.fullmatch(r"(?:[1-9]|1[0-4])-[1-3]", x) for x in config["data"]["exclude_tests"]):
        raise ValueError("Invalid recording exclusion ID")
    return deepcopy(config)


def load_config(path):
    return validate_config(yaml.safe_load(Path(path).read_text()))


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()
