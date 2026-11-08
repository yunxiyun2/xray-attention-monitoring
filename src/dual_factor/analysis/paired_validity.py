"""Record-balanced, within-person KSS contrasts. Windows are not independent people."""
from itertools import product
import numpy as np
import pandas as pd
from dual_factor.protocol import RETAINED, DIRECTIONS, require_columns


def paired_validity(df, columns=None, alert_max=4, drowsy_min=6, iterations=2000, seed=2026110803):
    columns = list(columns or RETAINED)
    require_columns(df.columns, ["subject", "test", "kss", *columns])
    if not 1 <= alert_max < drowsy_min <= 9 or iterations < 1:
        raise ValueError("Invalid KSS thresholds or bootstrap count")
    if df.groupby(["subject", "test"]).kss.nunique().max() > 1:
        raise ValueError("KSS changes within a recording")
    records = df.groupby(["subject", "test", "kss"], as_index=False)[columns].mean()
    deltas, excluded = [], []
    for identity, person in records.groupby("subject"):
        alert = person.loc[person.kss <= alert_max, columns]
        drowsy = person.loc[person.kss >= drowsy_min, columns]
        if alert.empty or drowsy.empty:
            excluded.append({"subject": str(identity), "reason": "missing_alert_or_drowsy"})
            continue
        # Equal-record variance, not a scale dominated by the longest recording.
        windows = df.loc[df.subject == identity]
        counts = windows.groupby("test")["test"].transform("size").to_numpy(float)
        w = 1 / (windows.test.nunique() * counts)
        x = windows[columns].to_numpy(float)
        variance = w @ (x - w @ x) ** 2
        if not np.isfinite(variance).all() or np.any(variance <= 1e-20):
            raise ValueError(f"Degenerate paired-effect scale: subject {identity}")
        delta = (drowsy.mean()-alert.mean()).to_numpy()
        deltas.append({"subject": str(identity), **dict(zip(columns, delta)),
                       **{c+"_standardized": v for c, v in zip(columns, delta/np.sqrt(variance))}})
    table = pd.DataFrame(deltas)
    if len(table) < 2:
        return table, pd.DataFrame(), {"status": "not_evaluable", "n_pairs": len(table), "excluded": excluded}
    if len(table) > 20:
        raise ValueError("Exact sign enumeration limited to 20 paired subjects")
    values = table[[c+"_standardized" for c in columns]].to_numpy(float)
    observed = values.mean(axis=0)
    signs = np.asarray(list(product([-1., 1.], repeat=len(table))))
    null = signs @ values / len(table)
    raw_p = (np.abs(null) >= np.abs(observed)[None, :] - 1e-12).mean(axis=0)
    order = np.argsort(raw_p)
    adjusted = np.empty(len(columns))
    adjusted[order] = np.minimum(1., np.maximum.accumulate(raw_p[order] * np.arange(len(columns), 0, -1)))
    rng = np.random.default_rng(seed)
    sampled_means = values[rng.integers(0, len(table), (iterations, len(table)))].mean(axis=1)
    ci = np.quantile(sampled_means, [.025, .975], axis=0)
    results = []
    for i, column in enumerate(columns):
        expected = DIRECTIONS[column]
        correct = observed[i] * expected > 0
        conclusion = ("supported" if correct else "opposite") if adjusted[i] < .05 else "inconclusive"
        results.append({"indicator": column, "n_pairs": len(table), "mean_delta_standardized": observed[i],
                        "ci_low": ci[0, i], "ci_high": ci[1, i], "p_value": raw_p[i],
                        "p_holm": adjusted[i], "expected_sign": expected, "conclusion": conclusion})
    return table, pd.DataFrame(results), {"status": "ok", "n_pairs": len(table), "excluded": excluded,
        "assumption": "Sign-flip inference assumes symmetric exchangeable within-person differences; not causal randomization"}
