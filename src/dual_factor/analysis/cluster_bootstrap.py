"""Whole-person bootstrap, preserving all within-person multivariate dependence."""
from itertools import permutations, product
import numpy as np
import pandas as pd
from dual_factor.protocol import RETAINED, CRITERIA, STABILITY
from dual_factor.analysis.paf import fit_paf
from dual_factor.analysis.sampling import structural_correlation


def aligned_loadings(reference, candidate, phi):
    """Only signed permutations; never an unconstrained Procrustes rotation."""
    if reference.shape != candidate.shape:
        raise ValueError("Alignment dimensions differ")
    best = None
    for permutation in permutations(range(candidate.shape[1])):
        current = candidate[:, permutation]
        norms = np.linalg.norm(reference, axis=0) * np.linalg.norm(current, axis=0)
        if np.any(norms == 0):
            continue
        cosine = (reference * current).sum(axis=0) / norms
        signs = np.where(cosine < 0, -1., 1.)
        score = np.abs(cosine)
        if best is None or score.sum() > best[0]:
            transformed_phi = phi[np.ix_(permutation, permutation)] * np.outer(signs, signs)
            best = (score.sum(), current * signs, transformed_phi, score)
    if best is None:
        raise ValueError("Degenerate factor alignment")
    return best[1:]


def anchor_fit(fit, columns):
    indices = [columns.index("FD"), columns.index("SE")]
    assignments = np.argmax(np.abs(fit.loadings), axis=1)
    order = assignments[indices].tolist()
    if order[0] == order[1]:
        raise ValueError("FD and SE do not anchor distinct factors")
    loading = fit.loadings[:, order].copy()
    signs = np.sign(loading[indices, [0, 1]])
    if np.any(signs == 0):
        raise ValueError("Zero anchor loading")
    return loading * signs, fit.phi[np.ix_(order, order)] * np.outer(signs, signs)


def criterion_values(loadings, phi, communalities, thresholds=None):
    thresholds = CRITERIA | (thresholds or {})
    order = np.sort(np.abs(loadings), axis=1)
    return {
        "C1": float(np.mean(communalities)) > thresholds["variance_min"],
        "C2": bool(np.all(order[:, -1] >= thresholds["loading_min"])),
        "C3": float(order[:, -2].max()) < thresholds["cross_max"],
        "C4": abs(float(phi[0, 1])) < thresholds["correlation_max"],
    }


def bootstrap_structure(df, columns=None, iterations=2000, seed=2026110802,
                        thresholds=None, stability=None, tol=1e-6, max_iter=500):
    columns = list(columns or RETAINED)
    if set(columns) != set(RETAINED) or iterations < 1:
        raise ValueError("Bootstrap requires all six retained indicators and positive iterations")
    r, person_matrices, identities = structural_correlation(df, columns)
    reference = fit_paf(r, 2, tol=tol, max_iter=max_iter)
    reference_loading, reference_phi = anchor_fit(reference, columns)
    expected = np.array([0 if c in RETAINED[:4] else 1 for c in columns])
    thresholds = CRITERIA | (thresholds or {})
    stability = STABILITY | (stability or {})
    random = np.random.default_rng(seed)
    rows, successful = [], []
    for replicate in range(iterations):
        draw = random.integers(0, len(identities), len(identities))
        row = {"replicate": replicate, "source_subjects": [str(identities[i]) for i in draw],
               "status": "failed", "joint_pass": False, "assignment_match": [False] * len(columns)}
        try:
            # Averaging the complete within-person covariance contributions is exactly
            # equivalent to materializing all drawn person/record/window copies.
            fit = fit_paf(person_matrices[draw].mean(axis=0), 2, tol=tol, max_iter=max_iter)
            loading, phi, congruence = aligned_loadings(reference_loading, fit.loadings, fit.phi)
            matches = np.argmax(np.abs(loading), axis=1) == expected
            checks = criterion_values(loading, phi, fit.communalities, thresholds)
            row.update(status="ok", loadings=loading.tolist(), factor_corr=float(phi[0, 1]),
                       common_variance=float(fit.communalities.mean()), congruence=congruence.tolist(),
                       assignment_match=matches.tolist(), joint_pass=all(checks.values()), criteria=checks)
            successful.append(row)
        except (ValueError, np.linalg.LinAlgError) as error:
            row["failure_reason"] = str(error)
        rows.append(row)
    summary = {"iterations": iterations, "n_subjects": len(identities), "n_valid": len(successful),
               "valid_fraction": len(successful)/iterations,
               "joint_fraction": sum(x["joint_pass"] for x in rows)/iterations,
               "candidate_selection": "conditional_on_historical_six_indicator_set",
               "ci_scope": "percentile_intervals_condition_on_numerically_valid_replicates",
               "reference_loadings": reference_loading.tolist(), "reference_phi": reference_phi.tolist(),
               "reference_criteria": criterion_values(reference_loading, reference_phi, reference.communalities, thresholds)}
    assignment = np.array([row["assignment_match"] for row in rows]).mean(axis=0)
    summary["assignment_probability"] = dict(zip(columns, assignment.tolist()))
    checks = {"valid": summary["valid_fraction"] >= stability["valid_fraction"],
              "joint": summary["joint_fraction"] >= stability["joint_fraction"],
              "assignment": bool(np.all(assignment >= stability["assignment_probability"]))}
    if successful:
        loadings = np.asarray([x["loadings"] for x in successful])
        ordered = np.sort(np.abs(loadings), axis=2)
        primary_ci = np.quantile(ordered[:, :, -1], [.025, .975], axis=0)
        cross_ci = np.quantile(ordered[:, :, -2], [.025, .975], axis=0)
        correlation_ci = np.quantile([x["factor_corr"] for x in successful], [.025, .975])
        congruence_p5 = np.quantile([x["congruence"] for x in successful], .05, axis=0)
        summary.update(primary_loading_ci=primary_ci.T.tolist(), cross_loading_ci=cross_ci.T.tolist(),
                       factor_corr_ci=correlation_ci.tolist(), congruence_p5=congruence_p5.tolist())
        checks.update(primary_ci=bool(np.all(primary_ci[0] >= thresholds["loading_min"])),
                      cross_ci=bool(np.all(cross_ci[1] < thresholds["cross_max"])),
                      correlation_ci=bool(np.all(np.abs(correlation_ci) < thresholds["correlation_max"])),
                      congruence=bool(np.all(congruence_p5 >= stability["congruence_min"])))
    else:
        checks["estimable"] = False
    p = summary["joint_fraction"]
    summary.update(C5_checks=checks, C5_pass=all(checks.values()),
                   joint_mc_se=float(np.sqrt(p*(1-p)/iterations)))
    return pd.DataFrame(rows), summary


def bootstrap_single_factor(df, columns=None, iterations=2000, seed=2026110805, thresholds=None, tol=1e-6, max_iter=500):
    """Evidence for a clear one-factor alternative, not a fallback for failed PAF."""
    columns = list(columns or RETAINED)
    thresholds = CRITERIA | (thresholds or {})
    r, matrices, identities = structural_correlation(df, columns)
    reference = fit_paf(r, 1, tol=tol, max_iter=max_iter)
    random = np.random.default_rng(seed)
    values, primary = [], []
    passed = 0
    for _ in range(iterations):
        draw = random.integers(0, len(identities), len(identities))
        try:
            fit = fit_paf(matrices[draw].mean(axis=0), 1, tol=tol, max_iter=max_iter)
            load = np.abs(fit.loadings[:, 0])
            variance = float(fit.communalities.mean())
            values.append(variance)
            primary.append(load)
            passed += variance > thresholds["variance_min"] and np.all(load >= thresholds["loading_min"])
        except (ValueError, np.linalg.LinAlgError):
            continue
    result = {"n_subjects": len(identities), "iterations": iterations, "valid_fraction": len(values)/iterations,
              "joint_fraction": passed/iterations, "reference_common_variance": float(reference.communalities.mean()),
              "alternative_supported": False}
    if values:
        variance_ci = np.quantile(values, [.025, .975])
        load_ci = np.quantile(primary, [.025, .975], axis=0)
        result.update(common_variance_ci=variance_ci.tolist(), primary_loading_ci=load_ci.T.tolist(),
                      alternative_supported=bool(len(values)/iterations >= .95 and passed/iterations >= .80
                        and variance_ci[0] > thresholds["variance_min"] and np.all(load_ci[0] >= thresholds["loading_min"])))
    return result
