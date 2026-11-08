"""Regression tests for statistical failures, not just return shapes."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from dual_factor.config import DEFAULT_CONFIG, validate_config
from dual_factor.protocol import RETAINED
from dual_factor.analysis.paf import fit_paf
from dual_factor.analysis.sampling import select_nonoverlap, structural_correlation, weighted_covariance_decomposition
from dual_factor.analysis.cluster_bootstrap import bootstrap_structure, aligned_loadings
from dual_factor.analysis.paired_validity import paired_validity
from dual_factor.analysis.construct_validity import direction_correction_report, amplitude_normalization
from dual_factor.analysis.sensitivity import subject_level_efa
from dual_factor.data.matrix_io import audit_matrix
from dual_factor.run_manifest import require_teacher_acceptance, file_hash


def fixture_matrix(subjects=8, seed=17):
    random = np.random.default_rng(seed)
    parts = []
    for subject in range(subjects):
        for test in range(2):
            loading = np.zeros((6, 2))
            loading[:4, 0] = [.85, -.85, -.85, -.85]
            loading[4:, 1] = .85
            phi = np.array([[1., .4], [.4, 1.]])
            correlation = loading @ phi @ loading.T + np.eye(6)*(1-.85**2)
            x = np.empty((120, 6))
            # Exact known covariance in EACH phase isolates sampling invariance
            # from numerical identifiability. Stochastic/weak cases are tested separately.
            for phase in range(3):
                values = random.normal(size=(40, 6))
                values -= values.mean(axis=0)
                cholesky = np.linalg.cholesky(values.T @ values/40)
                white = np.linalg.solve(cholesky, values.T).T
                x[phase::3] = white @ np.linalg.cholesky(correlation).T
            x += test*np.array([.5, -.5, -.5, -.5, .5, .5])
            frame = pd.DataFrame(x, columns=RETAINED)
            frame["subject"], frame["test"] = str(subject), str(test)
            frame["window_idx"], frame["valid"] = np.arange(120), True
            frame["kss"] = 3 if test == 0 else 7
            frame["SV"], frame["SDR"] = random.normal(size=(2, 120))
            parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def test_three_record_indices_do_not_interleave_or_backfill():
    frame = fixture_matrix(1)
    frame.loc[(frame.test == "0") & (frame.window_idx == 3), "valid"] = False
    selected = select_nonoverlap(frame.sample(frac=1, random_state=4))
    assert 3 not in selected.loc[selected.test == "0", "window_idx"].to_list()
    assert not selected.window_idx.mod(3).any()
    for _, group in selected.groupby(["subject", "test"]):
        assert np.all(group.start_sec.to_numpy()[1:] >= group.end_sec.to_numpy()[:-1])


def test_non_integer_sampling_ratio_rejected():
    with pytest.raises(ValueError):
        select_nonoverlap(fixture_matrix(1), window_sec=25, stride_sec=10)


def test_duplicate_record_key_rejected():
    frame = fixture_matrix(1)
    with pytest.raises(ValueError, match="Duplicate"):
        select_nonoverlap(pd.concat([frame, frame.iloc[:1]]))


def test_paf_recovers_analytic_common_covariance_not_pca():
    loading = np.zeros((8, 2))
    loading[:4, 0], loading[4:, 1] = .8, .7
    common = loading @ loading.T
    correlation = common + np.diag(1-np.diag(common))
    fit = fit_paf(correlation)
    np.testing.assert_allclose(fit.reproduced, common, atol=1e-5)
    np.testing.assert_allclose(fit.communalities, np.diag(common), atol=1e-5)
    assert fit.iterations > 1


def test_paf_fails_instead_of_clipping_or_faking_convergence():
    r = np.full((6, 6), .6)
    np.fill_diagonal(r, 1)
    with pytest.raises(ValueError):
        fit_paf(r, 1, max_iter=1)
    with pytest.raises(ValueError, match="singular"):
        fit_paf(np.ones((6, 6)))


def test_duplicate_windows_do_not_increase_person_evidence():
    frame = select_nonoverlap(fixture_matrix())
    r, matrices, people = structural_correlation(frame, list(RETAINED))
    duplicated = pd.concat([frame, frame])
    r2, matrices2, people2 = structural_correlation(duplicated, list(RETAINED))
    np.testing.assert_allclose(r, r2)
    np.testing.assert_allclose(matrices, matrices2)
    assert people == people2
    draws, summary = bootstrap_structure(frame, iterations=20, seed=42)
    draws2, summary2 = bootstrap_structure(duplicated, iterations=20, seed=42)
    np.testing.assert_allclose(summary["factor_corr_ci"], summary2["factor_corr_ci"], atol=1e-8)
    assert all(len(ids) == len(people) for ids in draws.source_subjects)
    assert any(len(set(ids)) < len(ids) for ids in draws.source_subjects)


def test_signed_permutation_alignment_preserves_covariance():
    random = np.random.default_rng(1)
    reference = random.normal(size=(6, 2))
    phi = np.array([[1, .3], [.3, 1]])
    swapped = reference[:, ::-1] * [1, -1]
    swapped_phi = phi[::-1, ::-1] * np.outer([1, -1], [1, -1])
    aligned, aligned_phi, congruence = aligned_loadings(reference, swapped, swapped_phi)
    np.testing.assert_allclose(aligned, reference)
    np.testing.assert_allclose(aligned_phi, phi)
    np.testing.assert_allclose(congruence, 1)


def test_zero_subject_z_mean_is_not_a_subject_level_factor_signal():
    frame = fixture_matrix()
    for col in RETAINED:
        frame[col+"_z"] = frame.groupby("subject")[col].transform(lambda x: (x-x.mean())/x.std(ddof=0))
    result = subject_level_efa(frame, "subject", list(RETAINED))
    assert result["status"] == "degenerate_input"
    deltas, tests, info = paired_validity(select_nonoverlap(frame), iterations=30)
    assert info["n_pairs"] == 8
    assert tests.loc[tests.indicator == "FD", "mean_delta_standardized"].item() > 0


def test_pairing_does_not_invent_second_class():
    frame = fixture_matrix(3)
    frame.loc[frame.subject == "2", "kss"] = 3
    deltas, _, info = paired_validity(select_nonoverlap(frame), iterations=10)
    assert info["n_pairs"] == 2
    assert "2" not in deltas.subject.to_list()


def test_normalized_column_mapping_is_really_used():
    frame = fixture_matrix(3)
    after = frame.copy()
    after["FD_norm"] = -frame.FD
    result = direction_correction_report(frame, after, ["FD"])
    assert result.diff_before.item() == pytest.approx(-result.diff_after.item())
    scaled = amplitude_normalization(frame, "subject", list(RETAINED))
    for _, person in scaled.groupby("subject"):
        a = (person.FD-person.FD.mean())/person.FD.std(ddof=0)
        b = (person.FD_norm-person.FD_norm.mean())/person.FD_norm.std(ddof=0)
        np.testing.assert_allclose(a, b, atol=1e-12)


def test_covariance_decomposition_uses_consistent_weights():
    frame = select_nonoverlap(fixture_matrix())
    decomposition = weighted_covariance_decomposition(frame, list(RETAINED))
    r, _, _ = structural_correlation(frame, list(RETAINED))
    np.testing.assert_allclose(decomposition["total"], r, atol=1e-12)


@pytest.mark.parametrize("mutation", [
    lambda c: c.update(unknown=True),
    lambda c: c["stages"].update(cfa=True),
    lambda c: c["structure"].update(extraction="principal"),
    lambda c: c["output"].update(overwrite=True),
    lambda c: c["parallel_analysis"].update(iterations=0),
])
def test_protocol_rejects_silent_parameter_or_stage_changes(mutation):
    config = deepcopy(DEFAULT_CONFIG)
    mutation(config)
    with pytest.raises(ValueError):
        validate_config(config)


def test_audit_missing_rate_uses_all_windows():
    frame = fixture_matrix(2)
    frame.loc[:9, "valid"] = False
    frame.loc[:9, "FD"] = np.nan
    _, audit, records = audit_matrix(frame, allow_reconstructed_time=True)
    assert audit["missing_rate_by_indicator"]["FD"] == pytest.approx(10/len(frame))
    assert audit["invalid_window_rate"] == pytest.approx(10/len(frame))
    assert records.n_generated.sum() == len(frame)


def test_teacher_gate_binds_the_exact_report(tmp_path):
    report = tmp_path/"report.md"
    report.write_text("candidate only")
    approval = tmp_path/"approval.json"
    approval.write_text(json.dumps({"accepted": True, "source": "teacher review", "report_sha256": file_hash(report)}))
    require_teacher_acceptance(approval, report)
    report.write_text("modified")
    with pytest.raises(ValueError):
        require_teacher_acceptance(approval, report)


def test_true_paf_failure_is_not_relabelled_as_a_structural_negative():
    from dual_factor.analysis.correction import run_correction_ladder, format_final_decision
    from dual_factor.protocol import ALL_INDICATORS
    rng = np.random.default_rng(3)
    data = pd.DataFrame(rng.normal(size=(600, 8)), columns=ALL_INDICATORS)
    result = run_correction_ladder(data, extraction="paf")
    assert result["final"]["verdict"] == "not_evaluable"
    assert result["rungs"][-1]["status"] == "failed"
    assert "不能据此" in result["stop_reason"]
    assert "不进入CFA" in format_final_decision(result)


def test_bootstrap_fit_failures_stay_in_denominator(monkeypatch):
    import dual_factor.analysis.cluster_bootstrap as module
    original = module.fit_paf
    calls = 0
    def only_reference_succeeds(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise ValueError("PAF did not converge")
        return original(*args, **kwargs)
    monkeypatch.setattr(module, "fit_paf", only_reference_succeeds)
    rows, summary = module.bootstrap_structure(select_nonoverlap(fixture_matrix()), iterations=7)
    assert len(rows) == 7
    assert summary["n_valid"] == 0
    assert summary["valid_fraction"] == 0
    assert summary["C5_pass"] is False


def test_observable_signal_is_not_a_latent_factor_acceptance():
    from dual_factor.analysis.covariance_signal import covariance_signal
    result = covariance_signal(select_nonoverlap(fixture_matrix()), iterations=30)
    assert result["observable_signal_detected"] is True
    assert result["latent_acceptance"] is False
    assert result["can_unlock_cfa"] is False


def test_observable_signal_rejects_a_single_common_axis():
    from dual_factor.analysis.covariance_signal import covariance_signal
    frame = fixture_matrix()
    random = np.random.default_rng(3)
    latent = random.normal(size=len(frame))
    for column in RETAINED:
        frame[column] = latent + random.normal(size=len(frame))*.1
    result = covariance_signal(select_nonoverlap(frame), iterations=30)
    assert result["observable_signal_detected"] is False
