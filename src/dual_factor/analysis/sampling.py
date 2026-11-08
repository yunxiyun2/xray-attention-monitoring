"""Time-interval selection and equal-person/equal-record correlation estimates."""
import numpy as np
import pandas as pd
from dual_factor.protocol import require_columns
from dual_factor.analysis.paf import validate_correlation


def select_nonoverlap(df, window_sec=30., stride_sec=10., phase=0):
    require_columns(df.columns, ["subject", "window_idx"])
    ratio = window_sec / stride_sec if stride_sec > 0 else 0
    if window_sec <= 0 or not np.isclose(ratio, round(ratio)) or ratio < 1:
        raise ValueError("Window/stride must be a positive integer ratio")
    step = int(round(ratio))
    if not 0 <= phase < step:
        raise ValueError("Phase is outside the sampling grid")
    out = df.copy()
    # Single-record synthetic callers may omit test; real migration supplies it.
    if "test" not in out:
        out["test"] = "single_record"
    keys = ["subject", "test", "window_idx"]
    if out.duplicated(keys).any():
        raise ValueError("Duplicate window primary key")
    idx = pd.to_numeric(out.window_idx, errors="raise").to_numpy()
    if (idx < 0).any() or not np.equal(idx, np.floor(idx)).all():
        raise ValueError("Window indices must be nonnegative integers")
    if "start_sec" not in out:
        out["start_sec"] = idx * stride_sec
        out["end_sec"] = out.start_sec + window_sec
    require_columns(out.columns, ["start_sec", "end_sec"])
    times = out[["start_sec", "end_sec"]].to_numpy(float)
    if not np.isfinite(times).all() or not np.allclose(times[:, 1]-times[:, 0], window_sec):
        raise ValueError("Invalid window time interval")
    if not np.allclose(times[:, 0], idx * stride_sec):
        raise ValueError("Window indices and recording-relative times disagree")
    out = out.loc[(idx.astype(int) % step) == phase].sort_values(["subject", "test", "start_sec"])
    for _, recording in out.groupby(["subject", "test"], sort=False):
        if np.any(recording.start_sec.to_numpy()[1:] < recording.end_sec.to_numpy()[:-1] - 1e-10):
            raise ValueError("Selected windows overlap within a recording")
    # Filter only AFTER locking the phase; invalid windows are never backfilled.
    if "valid" in out:
        if not pd.api.types.is_bool_dtype(out.valid):
            raise ValueError("valid must contain booleans, not truthy strings")
        out = out.loc[out.valid]
    return out.reset_index(drop=True)


def person_correlations(df, columns, subject_col="subject"):
    require_columns(df.columns, [subject_col, "test", *columns])
    matrices, identities = [], []
    for identity, person in df.groupby(subject_col, sort=True):
        x = person[columns].to_numpy(float)
        if not np.isfinite(x).all():
            raise ValueError("Structural input contains nonfinite values")
        counts = person.groupby("test")["test"].transform("size").to_numpy(float)
        weights = 1 / (person.test.nunique() * counts)
        mean = weights @ x
        variance = weights @ (x - mean) ** 2
        if np.any(variance <= 1e-20):
            raise ValueError(f"degenerate_indicator for subject {identity}")
        z = (x - mean) / np.sqrt(variance)
        matrices.append((z * weights[:, None]).T @ z)
        identities.append(identity)
    if not matrices:
        raise ValueError("No usable subjects")
    return np.asarray(matrices), identities


def structural_correlation(df, columns, subject_col="subject"):
    matrices, identities = person_correlations(df, columns, subject_col)
    return validate_correlation(matrices.mean(axis=0)), matrices, identities


def weighted_covariance_decomposition(df, columns):
    """Within-person standardized, equal-record covariance decomposition."""
    within, between = [], []
    for _, person in df.groupby("subject"):
        x = person[columns].to_numpy(float)
        counts = person.groupby("test")["test"].transform("size").to_numpy(float)
        weights = 1 / (person.test.nunique() * counts)
        mean = weights @ x
        var = weights @ (x-mean)**2
        if np.any(var <= 1e-20):
            raise ValueError("Degenerate covariance decomposition input")
        z = (x-mean)/np.sqrt(var)
        w, b = np.zeros((len(columns), len(columns))), np.zeros((len(columns), len(columns)))
        for test in person.test.unique():
            values = z[person.test.to_numpy() == test]
            mu = values.mean(axis=0)
            w += (values-mu).T @ (values-mu)/len(values)/person.test.nunique()
            b += np.outer(mu, mu)/person.test.nunique()
        within.append(w)
        between.append(b)
    w, b = np.mean(within, axis=0), np.mean(between, axis=0)
    return {"within_record": w.tolist(), "between_record_means": b.tolist(), "total": (w+b).tolist()}
