from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
import yaml
from dual_factor.config import DEFAULT_CONFIG
from dual_factor.config import config_hash
from dual_factor.run_manifest import code_hash
from dual_factor.v5_pipeline import run_experiment
from dual_factor.analysis.parallel_analysis_v5 import dimension_reference
from dual_factor.data.preprocess import hampel_filter
from dual_factor.analysis.cfa import _build_model_spec, run_cfa
from test_v5_regressions import fixture_matrix


def test_matrix_pipeline_dry_run_and_nonpromotable_diagnostics(tmp_path):
    config = deepcopy(DEFAULT_CONFIG)
    frame = fixture_matrix(4)
    matrix = tmp_path/"input.csv"
    frame.to_csv(matrix, index=False)
    source = tmp_path/"source.yaml"
    source.write_text(yaml.safe_dump(config))
    config["input"].update(path=str(matrix), source_config=str(source))
    config["output"]["root"] = str(tmp_path/"runs")
    config["parallel_analysis"]["iterations"] = 3
    config["cluster_bootstrap"]["iterations"] = 10
    config["validity"]["iterations"] = 10
    result = run_experiment(config, tmp_path, dry_run=True)
    assert result["writes"] is False
    assert not (tmp_path/"runs").exists()
    with pytest.raises(ValueError, match="certificate"):
        run_experiment(config, tmp_path)
    result = run_experiment(config, tmp_path, diagnostic_only=True)
    output = Path(result["run_dir"])
    assert result["decision"] == "pending"
    facts = json.loads((output/"facts.json").read_text())
    assert facts["teacher_accepted"] is False
    assert facts["groups"]["f2"] == ["SE", "GTE"]
    assert (output/"efa_models.json").exists()
    assert (output/"efa_decision_report.md").exists()
    assert (output/"figures"/"paired_validity.png").exists()
    manifest = json.loads((output/"manifest.json").read_text())
    assert manifest["stages"]["cfa"] == "not_run"
    certificate = tmp_path/"diagnostic_certificate.json"
    certificate.write_text(json.dumps({"diagnostic_checks_passed": True,
        "latent_power_validated": False, "passed": False, "datasets_per_scenario": 200,
        "bootstrap_iterations": 200, "code_sha256": code_hash(tmp_path), "config_sha256": config_hash(config)}))
    evaluated = run_experiment(config, tmp_path, validation=certificate)
    assert evaluated["decision"] == "B"
    evaluated_facts = json.loads((Path(evaluated["run_dir"])/"facts.json").read_text())
    assert evaluated_facts["latent_power_validated"] is False
    assert evaluated_facts["teacher_accepted"] is False
    # Word generation cannot inherit the previous report's numeric claims.
    project = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("biweekly", project/"scripts/gen_biweekly_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    document = module.build_document(output)
    text = "\n".join(p.text for p in document.paragraphs)
    assert "诊断运行" in text
    assert "每因子三项" not in text


def test_changed_matrix_extraction_parameters_are_rejected(tmp_path):
    config = deepcopy(DEFAULT_CONFIG)
    matrix = tmp_path/"input.csv"
    fixture_matrix(2).to_csv(matrix, index=False)
    source = tmp_path/"source.yaml"
    source.write_text(yaml.safe_dump(config))
    config["input"].update(path=str(matrix), source_config=str(source))
    config["event_detection"]["velocity_percentile"] = 60
    with pytest.raises(ValueError, match="changed extraction"):
        run_experiment(config, tmp_path, dry_run=True)


def test_dependence_reference_does_not_bridge_missing_windows():
    data = fixture_matrix(2)
    data = data.loc[data.window_idx.mod(3) == 0].copy()
    data = data.loc[~data.window_idx.isin([30, 60])].reset_index(drop=True)
    result = dimension_reference(data, ["FD", "PSV", "SA", "FN", "SE", "GTE"],
        mode="record_centered_shift", iterations=3, min_subjects=2)
    assert result["status"] == "ok"
    assert len(result["segments"]) > 4
    assert len(result["shifts"]) == 3


def test_exact_hampel_uses_one_median_per_neighborhood():
    signal = np.array([0., 0., 0., 10., 0., 0., 0., 1., 2., 3., 10., 4.])
    padded = np.pad(signal, 2, mode="edge")
    reference = signal.copy()
    for i in range(len(signal)):
        neighborhood = padded[i:i+5]
        median = np.median(neighborhood)
        mad = 1.4826*np.median(abs(neighborhood-median))
        if abs(signal[i]-median) > 3*mad:
            reference[i] = median
    np.testing.assert_allclose(hampel_filter(signal, window=5), reference)
    assert reference[3] == 0


def test_cfa_named_groups_ignore_dataframe_order():
    names = ["GTE_z", "FN_z", "SA_z", "SE_z", "FD_z", "PSV_z"]
    model = _build_model_spec(names)
    assert "DriveDecay =~ FD_z + PSV_z + SA_z + FN_z" in model
    assert "SpatialDisorg =~ SE_z + GTE_z" in model
    with pytest.raises(ValueError):
        _build_model_spec(["a", "b", "c", "d", "e", "f"])


def test_cfa_reports_true_correlation_and_standardized_loadings():
    rng = np.random.default_rng(7)
    latent = rng.multivariate_normal([0., 0.], [[1., .5], [.5, 1.]], size=1200)
    data = fixture_matrix(1).iloc[:120].copy()
    import pandas as pd
    data = pd.DataFrame({c+"_z": (latent[:, 0] if i < 4 else latent[:, 1]) * (i+1) + rng.normal(size=1200)
                         for i, c in enumerate(["FD", "PSV", "SA", "FN", "SE", "GTE"])})
    result = run_cfa(data)
    assert -.1 < result.factor_corr < .9
    assert result.factor_corr != pytest.approx(result.factor_covariance)
    assert all(abs(x) <= 1.01 for x in result.main_loadings.values())
    assert "jacobian_singular_ratio" in result.identification
    assert result.model_role == "internal_consistency"
