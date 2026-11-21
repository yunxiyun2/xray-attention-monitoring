"""Raw EOG caching and paired-subject index construction for SI-MSET."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from dual_factor.data.drozy_loader import load_all_records
from dual_factor.data.preprocess import preprocess_eog
from dual_factor.features.event_detection import detect_events_ivt
from dual_factor.run_manifest import save_json


def _path(root, value):
    path = Path(value)
    return path if path.is_absolute() else Path(root) / path


def file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def package_code_hash(root):
    root = Path(root)
    digest = hashlib.sha256()
    for path in sorted((root / "src/eog_si_mset").rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    for path in sorted((root / "scripts").glob("*eog*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _robust_normalize(values):
    values = np.asarray(values, dtype=np.float64)
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError("Degenerate robust scale")
    return np.clip((values - center) / scale, -8.0, 8.0)


def _resample_pair(h, v, source_fs, target_fs):
    if target_fs > source_fs:
        raise ValueError("target_fs cannot exceed source sampling rate")
    if np.isclose(source_fs, target_fs):
        return h.copy(), v.copy()
    ratio = Fraction(float(target_fs) / float(source_fs)).limit_denominator(1000)
    if not np.isclose(float(ratio), target_fs / source_fs, atol=1e-9):
        raise ValueError("Sampling-rate conversion cannot be represented safely")
    return (
        resample_poly(h, ratio.numerator, ratio.denominator),
        resample_poly(v, ratio.numerator, ratio.denominator),
    )


def _record_features(record, data_config):
    preprocess = data_config["preprocess"]
    cleaned_h, cleaned_v = preprocess_eog(
        record.eog_h,
        record.eog_v,
        record.sampling_rate,
        lowcut=preprocess["highpass"],
        highcut=preprocess["lowpass"],
        filter_order=preprocess["filter_order"],
        hampel_window=preprocess["hampel_window"],
        hampel_threshold=preprocess["hampel_threshold"],
    )
    target_fs = int(data_config["target_fs"])
    h_resampled, v_resampled = _resample_pair(
        cleaned_h, cleaned_v, record.sampling_rate, target_fs
    )
    h_norm = _robust_normalize(h_resampled)
    v_norm = _robust_normalize(v_resampled)
    dh = np.diff(h_norm, prepend=h_norm[0])
    dv = np.diff(v_norm, prepend=v_norm[0])
    speed = _robust_normalize(np.sqrt(dh ** 2 + dv ** 2))
    signals = np.stack([h_norm, v_norm, speed]).astype(np.float32)
    return cleaned_h, cleaned_v, signals


def prepare_cache(config, root):
    root = Path(root)
    data_config = config["data"]
    drozy_root = _path(root, data_config["drozy_root"])
    cache_dir = _path(root, data_config["cache_dir"])
    if not drozy_root.exists():
        raise FileNotFoundError(f"DROZY root not found: {drozy_root}")
    if cache_dir.exists():
        raise FileExistsError(f"Cache directory already exists: {cache_dir}")
    staging_dir = cache_dir.with_name(cache_dir.name + f".tmp-{uuid4().hex[:8]}")
    staging_dir.mkdir(parents=True)

    records = load_all_records(
        drozy_root,
        channels=data_config["channels"],
        subjects=data_config["subjects"],
        exclude_tests=data_config["exclude_tests"],
        kss_file=data_config["kss_file"],
    )
    if not records:
        raise ValueError("No DROZY records found")

    target_fs = int(data_config["target_fs"])
    window_size = int(data_config["window_sec"] * target_fs)
    stride_size = int(data_config["stride_sec"] * target_fs)
    raw_window = int(data_config["window_sec"] * records[0].sampling_rate)
    raw_stride = int(data_config["stride_sec"] * records[0].sampling_rate)
    if min(window_size, stride_size, raw_window, raw_stride) < 1:
        raise ValueError("Invalid window dimensions")
    event_config = data_config["event_detection"]

    windows, rows, rejected = [], [], []
    record_counts = {}
    source_hashes = {}
    for record in records:
        cleaned_h, cleaned_v, signals = _record_features(record, data_config)
        if not np.isfinite(signals).all():
            raise ValueError(f"Nonfinite signal after preprocessing: {record.subject}-{record.test}")
        n_windows = max(0, (signals.shape[1] - window_size) // stride_size + 1)
        raw_n_windows = max(0, (len(cleaned_h) - raw_window) // raw_stride + 1)
        if n_windows != raw_n_windows:
            raise ValueError("Source/resampled window counts disagree")
        valid_record = 0
        for window_idx in range(n_windows):
            start = window_idx * stride_size
            raw_start = window_idx * raw_stride
            events = detect_events_ivt(
                cleaned_h[raw_start:raw_start + raw_window],
                cleaned_v[raw_start:raw_start + raw_window],
                record.sampling_rate,
                velocity_threshold=event_config["velocity_threshold"],
                velocity_percentile=event_config["velocity_percentile"],
                min_fixation_ms=event_config["min_fixation_ms"],
                min_saccade_ms=event_config["min_saccade_ms"],
            )
            reason = []
            if len(events.fixations) < data_config["min_fixations"]:
                reason.append("insufficient_fixations")
            if len(events.saccades) < data_config["min_saccades"]:
                reason.append("insufficient_saccades")
            window = signals[:, start:start + window_size]
            if not np.isfinite(window).all():
                reason.append("nonfinite_window")
            if reason:
                rejected.append({
                    "subject": record.subject,
                    "test": record.test,
                    "kss": record.kss,
                    "window_idx": window_idx,
                    "reason": "|".join(reason),
                    "n_fixations": len(events.fixations),
                    "n_saccades": len(events.saccades),
                })
                continue
            label = 0 if record.kss <= data_config["alert_max"] else (
                1 if record.kss >= data_config["drowsy_min"] else -1
            )
            windows.append(window)
            rows.append({
                "subject": record.subject,
                "test": record.test,
                "kss": record.kss,
                "label": label,
                "state": "alert" if label == 0 else ("drowsy" if label == 1 else "intermediate"),
                "window_idx": window_idx,
                "start_sec": window_idx * data_config["stride_sec"],
                "end_sec": window_idx * data_config["stride_sec"] + data_config["window_sec"],
                "n_fixations": len(events.fixations),
                "n_saccades": len(events.saccades),
            })
            valid_record += 1
        record_counts[f"{record.subject}-{record.test}"] = {
            "generated": n_windows,
            "valid": valid_record,
            "kss": record.kss,
            "sampling_rate": record.sampling_rate,
        }
        edf_path = drozy_root / "psg" / f"{record.subject}-{record.test}.edf"
        source_hashes[f"psg/{edf_path.name}"] = file_sha256(edf_path)

    if not windows:
        raise ValueError("No valid EOG windows were generated")
    arrays = np.stack(windows).astype(np.float32, copy=False)
    metadata = pd.DataFrame(rows)
    paired = paired_subjects(metadata, data_config["alert_max"], data_config["drowsy_min"])
    if len(paired) < 2:
        raise ValueError("Fewer than two paired subjects in cached data")
    np.save(staging_dir / "windows.npy", arrays, allow_pickle=False)
    metadata.to_csv(staging_dir / "metadata.csv", index=False)
    pd.DataFrame(rejected).to_csv(staging_dir / "rejected_windows.csv", index=False)
    save_json(staging_dir / "record_counts.json", record_counts)
    kss_path = drozy_root / data_config["kss_file"]
    save_json(staging_dir / "manifest.json", {
        "protocol": config["protocol"],
        "n_records": len(records),
        "n_valid_windows": len(metadata),
        "n_rejected_windows": len(rejected),
        "n_paired_subjects": len(paired),
        "paired_subjects": paired,
        "windows_shape": list(arrays.shape),
        "target_fs": target_fs,
        "window_sec": data_config["window_sec"],
        "stride_sec": data_config["stride_sec"],
        "kss_sha256": file_sha256(kss_path),
        "source_files": source_hashes,
        "windows_sha256": file_sha256(staging_dir / "windows.npy"),
        "metadata_sha256": file_sha256(staging_dir / "metadata.csv"),
        "code_sha256": package_code_hash(root),
    })
    staging_dir.rename(cache_dir)
    return {
        "cache_dir": str(cache_dir),
        "n_windows": len(metadata),
        "n_rejected": len(rejected),
        "n_paired_subjects": len(paired),
        "paired_subjects": paired,
    }


def paired_subjects(metadata, alert_max=4, drowsy_min=6):
    required = {"subject", "state"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")
    subjects = []
    for subject, frame in metadata.groupby("subject", sort=True):
        states = set(frame.state)
        if "alert" in states and "drowsy" in states:
            subjects.append(str(subject))
    return subjects


def load_cache(cache_dir):
    cache_dir = Path(cache_dir)
    windows_path = cache_dir / "windows.npy"
    metadata_path = cache_dir / "metadata.csv"
    if not windows_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Incomplete SI-MSET cache: {cache_dir}")
    windows = np.load(windows_path, mmap_mode="r", allow_pickle=False)
    metadata = pd.read_csv(metadata_path)
    if len(windows) != len(metadata):
        raise ValueError("Window array and metadata lengths disagree")
    required = {"subject", "test", "kss", "label", "state", "window_idx"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")
    metadata["subject"] = metadata["subject"].astype(str)
    metadata["state"] = metadata["state"].astype(str)
    return windows, metadata


def paired_loso_folds(metadata, heldout_subject, validation_offset=1):
    subjects = paired_subjects(metadata)
    heldout_subject = str(heldout_subject)
    if heldout_subject not in subjects:
        raise ValueError(f"Unknown held-out subject: {heldout_subject}")
    pool = [subject for subject in subjects if subject != heldout_subject]
    if len(pool) < 3:
        raise ValueError("Insufficient paired subjects for train/validation/test")
    validation_subject = pool[int(validation_offset) % len(pool)]
    train_subjects = [subject for subject in pool if subject != validation_subject]
    splits = {
        "train": train_subjects,
        "validation": [validation_subject],
        "test": [heldout_subject],
    }
    overlap = (
        set(splits["train"]).intersection(splits["validation"])
        or set(splits["train"]).intersection(splits["test"])
        or set(splits["validation"]).intersection(splits["test"])
    )
    if overlap:
        raise ValueError("Subject split overlap")
    return splits


def split_indices(metadata, subjects):
    mask = metadata.subject.astype(str).isin([str(subject) for subject in subjects])
    return np.flatnonzero(mask.to_numpy())


def json_dump(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
