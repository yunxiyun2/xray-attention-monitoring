"""Audited legacy matrix migration; raw feature provenance remains explicit."""
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from dual_factor.protocol import ALL_INDICATORS, require_columns
from dual_factor.run_manifest import file_hash


def audit_matrix(df, window_sec=30., stride_sec=10., allow_reconstructed_time=False):
    df = df.copy()
    require_columns(df.columns, ["subject", "test", "kss", "window_idx", "valid", *ALL_INDICATORS])
    for column in ("subject", "test"):
        if df[column].isna().any():
            raise ValueError("Missing subject/record identity")
        df[column] = df[column].astype(str)
    if not pd.api.types.is_bool_dtype(df.valid) or df.valid.isna().any():
        raise ValueError("valid must be a nonmissing boolean")
    for column in ("kss", "window_idx"):
        values = pd.to_numeric(df[column], errors="raise")
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"Invalid integer column: {column}")
        df[column] = values.astype(int)
    if not df.kss.between(1, 9).all() or (df.window_idx < 0).any():
        raise ValueError("KSS/index outside supported range")
    if df.duplicated(["subject", "test", "window_idx"]).any():
        raise ValueError("Duplicate window primary key")
    if df.groupby(["subject", "test"]).kss.nunique().max() != 1:
        raise ValueError("KSS changes within a recording")
    time_source = "explicit"
    if "start_sec" not in df or "end_sec" not in df:
        if not allow_reconstructed_time:
            raise ValueError("Missing times require a matching source configuration snapshot")
        df["start_sec"] = df.window_idx * stride_sec
        df["end_sec"] = df.start_sec + window_sec
        time_source = "reconstructed_from_source_config"
    times = df[["start_sec", "end_sec"]].to_numpy(float)
    if not np.isfinite(times).all() or not np.allclose(times[:, 0], df.window_idx * stride_sec) or not np.allclose(times[:, 1]-times[:, 0], window_sec):
        raise ValueError("Window interval and configured index grid disagree")
    values = df[list(ALL_INDICATORS)].to_numpy(float)
    finite = np.isfinite(values).all(axis=1)
    df["complete_case"] = df.valid & finite
    # Do not hide nonfinite valid features or relabel them as original valid=False.
    legacy_reason = df["invalid_reason"].fillna("") if "invalid_reason" in df else pd.Series("", index=df.index)
    df["invalid_reason"] = np.where(~df.valid, np.where(legacy_reason != "", legacy_reason, "event_count_or_legacy_invalid"), np.where(~finite, "nonfinite_feature", ""))
    records = df.groupby(["subject", "test", "kss"], as_index=False).agg(
        n_generated=("valid", "size"), n_valid=("valid", "sum"), n_complete=("complete_case", "sum"))
    records["invalid_rate"] = 1 - records.n_valid/records.n_generated
    audit = {"n_generated": len(df), "n_valid": int(df.valid.sum()), "n_complete": int(df.complete_case.sum()),
             "n_subjects": int(df.subject.nunique()), "n_records": len(records),
             "invalid_window_rate": float((~df.valid).mean()),
             "missing_rate_by_indicator": dict(zip(ALL_INDICATORS, (~np.isfinite(values)).mean(axis=0))),
             "time_source": time_source, "feature_validation": "not_run",
             "n_valid_nonfinite": int((df.valid & ~finite).sum())}
    return df, audit, records


def load_matrix(path, source_config, window_sec=30., stride_sec=10.):
    source = yaml.safe_load(Path(source_config).read_text()) if source_config else {}
    old = source.get("windowing", {})
    allowed = (old.get("window_sec"), old.get("stride_sec")) == (window_sec, stride_sec)
    df, audit, records = audit_matrix(pd.read_csv(path), window_sec, stride_sec, allowed)
    audit.update(input_sha256=file_hash(path), source_config_sha256=file_hash(source_config) if source_config else None)
    return df, audit, records
