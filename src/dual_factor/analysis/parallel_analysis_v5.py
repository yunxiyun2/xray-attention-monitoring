"""Distinct iid and record-centered, dependence-preserving dimension references."""
import numpy as np
from dual_factor.analysis.sampling import person_correlations


def suggested_count(actual, reference):
    count = 0
    for value, cutoff in zip(actual, reference):
        if value <= cutoff:
            break
        count += 1
    return count


def _correlation(df, columns):
    matrices, _ = person_correlations(df, columns)
    return matrices.mean(axis=0)


def dimension_reference(df, columns, iterations=1000, percentile=95, seed=2026110801,
                        mode="iid", min_segment=8, min_subjects=10):
    if iterations < 1 or not 0 < percentile < 100 or mode not in ("iid", "record_centered_shift"):
        raise ValueError("Invalid dimension-reference settings")
    data = df.copy()
    rng = np.random.default_rng(seed)
    segments = []
    if mode == "record_centered_shift":
        parts = []
        for (subject, test), group in data.groupby(["subject", "test"]):
            group = group.sort_values("window_idx")
            breaks = group.window_idx.diff().ne(3).cumsum()
            for number, segment in group.groupby(breaks):
                if len(segment) >= min_segment:
                    segment = segment.copy()
                    segment[columns] -= segment[columns].mean()
                    segment["test"] = f"{test}:segment:{number}"
                    parts.append(segment)
                    segments.append({"subject": str(subject), "test": str(test), "n_windows": len(segment),
                                     "first_window": int(segment.window_idx.iloc[0])})
        if not parts:
            return {"status": "not_evaluable", "mode": mode, "reason": "no_long_contiguous_segments"}
        import pandas as pd
        data = pd.concat(parts, ignore_index=True)
        if data.subject.nunique() < min_subjects:
            return {"status": "not_evaluable", "mode": mode, "n_subjects": data.subject.nunique(),
                    "reason": "too_few_contributing_subjects", "segments": segments}
    actual = np.linalg.eigvalsh(_correlation(data, columns))[::-1]
    random_eigenvalues, shifts = [], []
    for iteration in range(iterations):
        surrogate = data.copy()
        if mode == "iid":
            surrogate[columns] = rng.standard_normal((len(data), len(columns)))
        else:
            offsets = []
            for _, group in data.groupby(["subject", "test"]):
                for column in columns:
                    offset = int(rng.integers(0, len(group)))
                    surrogate.loc[group.index, column] = np.roll(group[column].to_numpy(), offset)
                    offsets.append(offset)
            shifts.append(offsets)
        random_eigenvalues.append(np.linalg.eigvalsh(_correlation(surrogate, columns))[::-1])
    reference = np.percentile(random_eigenvalues, percentile, axis=0)
    return {"status": "ok", "mode": mode, "n_subjects": int(data.subject.nunique()),
            "n_records_or_segments": int(data.groupby(["subject", "test"]).ngroups), "n_windows": len(data),
            "actual_eigenvalues": actual.tolist(), "random_eigenvalues": reference.tolist(),
            "suggested_n_factors": suggested_count(actual, reference), "iterations": iterations,
            "percentile": percentile, "segments": segments, "shifts": shifts,
            "scope": "descriptive_iid_reference" if mode == "iid" else "within_record_centered_covariation"}
