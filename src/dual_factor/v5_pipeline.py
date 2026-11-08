"""Auditable v5 EFA runner. Later scientific stages cannot launch automatically."""
from pathlib import Path
import json
import sys
import platform
import numpy as np
import pandas as pd
import yaml
from dual_factor.config import validate_config, config_hash
from dual_factor.protocol import RETAINED, ALL_INDICATORS
from dual_factor.run_manifest import create_run, save_json, file_hash, code_hash
from dual_factor.data.matrix_io import load_matrix, audit_matrix
from dual_factor.analysis.sampling import select_nonoverlap, structural_correlation, weighted_covariance_decomposition
from dual_factor.analysis.paf import fit_paf
from dual_factor.analysis.cluster_bootstrap import bootstrap_structure, criterion_values, anchor_fit, bootstrap_single_factor
from dual_factor.analysis.paired_validity import paired_validity
from dual_factor.analysis.parallel_analysis_v5 import dimension_reference
from dual_factor.analysis.covariance_signal import covariance_signal


def _path(root, value):
    p = Path(value)
    return p if p.is_absolute() else Path(root)/p


def _raw_input(config, root):
    from dual_factor.data.drozy_loader import load_all_records
    from dual_factor.data.preprocess import preprocess_eog
    from dual_factor.features.windowing import build_indicator_matrix
    records = load_all_records(_path(root, config["input"]["path"]), **config["data"])
    frames = []
    for record in records:
        pp = config["preprocess"]
        h, v = preprocess_eog(record.eog_h, record.eog_v, record.sampling_rate,
                             lowcut=pp["highpass"], highcut=pp["lowpass"], filter_order=pp["filter_order"],
                             hampel_window=pp["hampel_window"], hampel_threshold=pp["hampel_threshold"])
        wn = config["windowing"]
        frame = build_indicator_matrix(record.subject, record.test, record.kss, h, v, record.sampling_rate,
            window_sec=wn["window_sec"], stride_sec=wn["stride_sec"],
            min_fixations=wn["min_fixations"], min_saccades=wn["min_saccades"],
            bins=config["spatial_grid"]["bins"], **config["event_detection"])
        frames.append(frame)
    if not frames:
        raise ValueError("No raw records available")
    df, audit, records = audit_matrix(pd.concat(frames, ignore_index=True))
    audit.update(feature_validation="not_run", raw_extraction="executed_not_independently_validated",
                 hampel_algorithm="v5_exact_local_mad", algorithm_changed_from_legacy=True)
    return df, audit, records


def _precheck(r):
    inverse = np.linalg.inv(r)
    partial = -inverse / np.sqrt(np.outer(np.diag(inverse), np.diag(inverse)))
    partial2, correlations2 = partial**2, r**2
    np.fill_diagonal(partial2, 0.)
    np.fill_diagonal(correlations2, 0.)
    denom = correlations2.sum() + partial2.sum()
    kmo = float(correlations2.sum()/denom) if denom else 0.
    return {"kmo": kmo, "kmo_pass": kmo > .50,
            "bartlett_scope": "Independent-window p-values are not used as cluster-valid evidence"}


def _model(data, columns, factors, config):
    r = None
    try:
        r, _, identities = structural_correlation(data, columns)
        fit = fit_paf(r, factors, tol=config["structure"]["paf_tol"], max_iter=config["structure"]["paf_max_iter"])
        result = {"status": "ok", "columns": columns, "requested_n_factors": factors,
                  "fit_success": True, "hypothesis_sign_match": None,
                  "n_subjects": len(identities), "n_records": data.groupby(["subject", "test"]).ngroups,
                  "n_windows": len(data), "loadings": fit.loadings.tolist(), "structure": fit.structure.tolist(),
                  "phi": fit.phi.tolist(), "communalities": fit.communalities.tolist(),
                  "uniquenesses": fit.uniquenesses.tolist(), "reproduced": fit.reproduced.tolist(),
                  "correlation": r.tolist(), "common_variance": float(fit.communalities.mean()),
                  "method": "iterative_paf", "iterations": fit.iterations, "precheck": _precheck(r)}
        if factors == 2 and set(columns) == set(RETAINED):
            try:
                loading, phi = anchor_fit(fit, columns)
                assignment = np.argmax(np.abs(loading), axis=1)
                expected = np.array([0 if c in RETAINED[:4] else 1 for c in columns])
                expected_sign = np.array([1 if c in ("FD", "SE", "GTE") else -1 for c in columns])
                result.update(anchored_loadings=loading.tolist(), anchored_phi=phi.tolist(),
                              assignment_match=bool(np.array_equal(assignment, expected)),
                              hypothesis_sign_match=bool(np.array_equal(np.sign(loading[np.arange(len(columns)), expected]), expected_sign)),
                              criteria=criterion_values(loading, phi, fit.communalities, config["structure"]["criteria"]))
            except ValueError as error:
                result.update(assignment_match=False, anchor_reason=str(error),
                              criteria=criterion_values(fit.loadings, fit.phi, fit.communalities, config["structure"]["criteria"]))
        return result
    except (ValueError, np.linalg.LinAlgError) as error:
        return {"status": "failed", "reason": str(error), "columns": columns,
                "requested_n_factors": factors, "fit_success": False, "n_windows": len(data),
                "correlation": r.tolist() if r is not None else None,
                "precheck": _precheck(r) if r is not None else None}


def _certificate_valid(path, root, config):
    if not path:
        return False
    value = json.loads(Path(path).read_text())
    return (value.get("diagnostic_checks_passed") is True and value.get("datasets_per_scenario", 0) >= 200
            and value.get("bootstrap_iterations", 0) >= 200
            and value.get("code_sha256") == code_hash(root)
            and value.get("config_sha256") == config_hash(config))


def run_experiment(config, project_root, *, dry_run=False, diagnostic_only=False, validation=None):
    config = validate_config(config)
    root = Path(project_root).resolve()
    if config["input"]["mode"] == "matrix":
        source = yaml.safe_load(_path(root, config["input"]["source_config"]).read_text())
        for section in ("preprocess", "event_detection", "spatial_grid"):
            if any(source.get(section, {}).get(key) != value for key, value in config[section].items()):
                raise ValueError("Matrix features cannot reflect changed extraction parameters; use raw mode")
        for key in ("min_fixations", "min_saccades"):
            if source.get("windowing", {}).get(key) != config["windowing"][key]:
                raise ValueError("Matrix event-count rules differ from source configuration")
        df, audit, quality = load_matrix(_path(root, config["input"]["path"]),
                                        _path(root, config["input"]["source_config"]))
    else:
        df, audit, quality = _raw_input(config, root)
    source_count = len(df)
    mask = df.subject.isin({str(x) for x in config["data"]["subjects"]})
    excluded = {tuple(x.split("-")) for x in config["data"]["exclude_tests"]}
    mask &= np.array([tuple(x) not in excluded for x in df[["subject", "test"]].to_numpy()], dtype=bool)
    if not mask.all():
        if not mask.any():
            raise ValueError("No rows remain after configured subject/record exclusions")
        df, selection_audit, quality = audit_matrix(df.loc[mask].copy())
        audit.update(selection_audit)
    audit["n_source_generated"] = source_count
    validated = _certificate_valid(validation, root, config)
    latent_power = validated and json.loads(Path(validation).read_text()).get("latent_power_validated") is True
    if dry_run:
        phases = {str(p): len(select_nonoverlap(df, phase=p).loc[lambda x: x.complete_case]) for p in (0, 1, 2)}
        return {"config_sha256": config_hash(config), "audit": audit,
                "phase_complete_window_counts": phases, "implementation_validated": validated,
                "latent_power_validated": latent_power,
                "stages": config["stages"], "writes": False}
    if not diagnostic_only and not validated:
        raise ValueError("Formal EFA requires a matching full simulation certificate; use --diagnostic-only for non-promotable checks")
    run = create_run(_path(root, config["output"]["root"]))
    manifest = {"protocol_version": "v5.0", "config_sha256": config_hash(config), "code_sha256": code_hash(root),
                "python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__,
                "diagnostic_only": diagnostic_only, "implementation_validated": validated,
                "latent_power_validated": latent_power,
                "validation_scope": "observable-signal and negative controls; latent acceptance has its own power gate",
                "stages": {"efa": "running", "cfa": "not_run", "prediction": "not_run", "external": "not_run", "invariance": "not_run"}}
    save_json(run/"manifest.json", manifest)
    (run/"config_used.yaml").write_text(yaml.safe_dump(config, allow_unicode=True))
    save_json(run/"data_audit.json", audit)
    quality.to_csv(run/"quality_by_record.csv", index=False)
    df.to_csv(run/"indicator_matrix.csv", index=False)
    try:
        samples, models, dimensions = {}, {}, {}
        pa_cfg = config["parallel_analysis"]
        for phase in [config["windowing"]["primary_phase"], *config["windowing"]["sensitivity_phases"]]:
            subset = select_nonoverlap(df, phase=phase)
            subset = subset.loc[subset.complete_case].reset_index(drop=True)
            samples[phase] = subset
            subset.to_csv(run/f"nonoverlap_phase_{phase}.csv", index=False)
            for columns, label in [(list(ALL_INDICATORS), "eight"), (list(RETAINED), "six")]:
                for factors in ([1, 2, 3] if label == "six" else [2, 3]):
                    key = f"phase_{phase}_{label}_{factors}"
                    models[key] = _model(subset, columns, factors, config)
                for mode in ("iid", "record_centered_shift"):
                    key = f"phase_{phase}_{label}_{mode}"
                    try:
                        dimensions[key] = dimension_reference(subset, columns, mode=mode,
                            iterations=pa_cfg["iterations"], percentile=pa_cfg["percentile"],
                            seed=pa_cfg["seed"]+phase, min_segment=pa_cfg["min_segment"], min_subjects=pa_cfg["min_subjects"])
                    except (ValueError, np.linalg.LinAlgError) as error:
                        dimensions[key] = {"status": "not_evaluable", "reason": str(error)}
        primary = samples[0]
        try:
            signal = covariance_signal(primary, **config["signal_detection"])
        except (ValueError, np.linalg.LinAlgError) as error:
            signal = {"observable_signal_detected": False, "status": "not_evaluable", "reason": str(error)}
        save_json(run/"observable_covariance_signal.json", signal)
        save_json(run/"efa_models.json", models)
        save_json(run/"parallel_analysis.json", dimensions)
        try:
            draws, summary = bootstrap_structure(primary, thresholds=config["structure"]["criteria"],
                stability=config["structure"]["stability"], tol=config["structure"]["paf_tol"],
                max_iter=config["structure"]["paf_max_iter"], **config["cluster_bootstrap"])
            save_json(run/"cluster_bootstrap.json", draws.to_dict("records"))
            draws.to_csv(run/"cluster_bootstrap.csv", index=False)
        except (ValueError, np.linalg.LinAlgError) as error:
            summary = {"C5_pass": False, "status": "not_evaluable", "reason": str(error)}
        save_json(run/"cluster_summary.json", summary)
        paired, tests, validity = paired_validity(primary, **config["validity"])
        paired.to_csv(run/"paired_subject_deltas.csv", index=False)
        tests.to_csv(run/"validity_tests.csv", index=False)
        save_json(run/"validity_summary.json", validity)
        auxiliary = []
        for identity, person in primary.groupby("subject"):
            model = _model(person, list(RETAINED), 2, config)
            try:
                person_pa = dimension_reference(person, list(RETAINED), mode="iid", iterations=pa_cfg["iterations"],
                    percentile=pa_cfg["percentile"], seed=pa_cfg["seed"])
                model["pa_suggested_n_factors"] = person_pa["suggested_n_factors"]
            except (ValueError, np.linalg.LinAlgError):
                model["pa_suggested_n_factors"] = None
            auxiliary.append({"subject": identity, "kind": "per_subject_fixed_two_factor", **model})
            auxiliary.append({"subject": identity, "kind": "leave_one_subject_out", **_model(primary.loc[primary.subject != identity], list(RETAINED), 2, config)})
        save_json(run/"subject_structure.json", auxiliary)
        try:
            decomposition = weighted_covariance_decomposition(primary, list(RETAINED))
        except ValueError as error:
            decomposition = {"status": "not_evaluable", "reason": str(error)}
        save_json(run/"covariance_decomposition.json", decomposition)
        diagnostics = {"feature_validation": audit["feature_validation"],
            "FD_FN_correlation": float(primary.FD.corr(primary.FN)),
            "PSV_SA_correlation": float(primary.PSV.corr(primary.SA)),
            "SE_GTE_correlation": float(primary.SE.corr(primary.GTE)),
            "GTE_definition": "joint_entropy_of_adjacent_grid_pairs",
            "SDR_definition": "KL_to_uniform_smaller_is_more_random",
            "raw_parameter_sensitivity": "not_run", "event_count_entropy_control": "not_run",
            "scope": "Proxy covariance does not establish a fatigue mechanism"}
        save_json(run/"measurement_diagnostics.json", diagnostics)
        phase_pass = all(models[f"phase_{p}_six_2"].get("status") == "ok"
                         and models[f"phase_{p}_six_2"].get("assignment_match") is True
                         and models[f"phase_{p}_six_2"].get("precheck", {}).get("kmo_pass") is True
                         and all(models[f"phase_{p}_six_2"].get("criteria", {}).values()) for p in samples)
        dimensions_ok = all(dimensions[f"phase_{p}_six_iid"].get("suggested_n_factors") == 2 for p in samples)
        centered = [dimensions[f"phase_{p}_six_record_centered_shift"] for p in samples]
        center_conflict = all(x.get("status") == "ok" for x in centered) and (
            all(x["suggested_n_factors"] == 1 for x in centered) or all(x["suggested_n_factors"] >= 3 for x in centered))
        ready = phase_pass and dimensions_ok and not center_conflict and summary["C5_pass"] and audit["invalid_window_rate"] < .10 and audit["n_valid_nonfinite"] == 0
        decision = "pending" if diagnostic_only or not validated else ("A" if ready and latent_power else "B")
        single_points = all(models[f"phase_{p}_six_1"].get("status") == "ok"
            and models[f"phase_{p}_six_1"].get("precheck", {}).get("kmo_pass") is True
            and models[f"phase_{p}_six_1"].get("common_variance", 0) > config["structure"]["criteria"]["variance_min"] for p in samples)
        if (decision == "B" and single_points and audit["invalid_window_rate"] < .10 and audit["n_valid_nonfinite"] == 0
                and all(dimensions[f"phase_{p}_six_iid"].get("suggested_n_factors") == 1 for p in samples)
                and all(x.get("status") == "ok" and x.get("suggested_n_factors") == 1 for x in centered)):
            alternative = bootstrap_single_factor(primary, thresholds=config["structure"]["criteria"],
                tol=config["structure"]["paf_tol"], max_iter=config["structure"]["paf_max_iter"], **config["cluster_bootstrap"])
            save_json(run/"single_factor_stability.json", alternative)
            if alternative["alternative_supported"]:
                decision = "C"
        facts = {"run_id": run.name, "scientific_decision": decision,
                 "evidence_level": "L1" if decision == "A" else "L0",
                 "teacher_accepted": False, "data": audit, "n_pairs": validity["n_pairs"],
                 "C5_pass": summary["C5_pass"], "phase_pass": phase_pass, "dimensions_ok": dimensions_ok,
                 "observable_signal_detected": signal["observable_signal_detected"],
                 "latent_power_validated": latent_power,
                 "method_correction_required": not latent_power,
                 "measurement_validation": "not_run", "groups": {"f1": list(RETAINED[:4]), "f2": list(RETAINED[4:])},
                 "hypothesis_validity": tests.to_dict("records"),
                 "limitations": ["历史六指标的条件性复验", "14人规模下的区间可能很宽", "EOG代理及原始事件效度尚未独立验证", "无CFA、预测或跨域结果"],
                 "source_files": ["data_audit.json", "cluster_summary.json", "validity_tests.csv", "efa_models.json", "parallel_analysis.json", "observable_covariance_signal.json"]}
        save_json(run/"facts.json", facts)
        save_json(run/"decision.json", facts)
        write_figures(run, models, dimensions, tests)
        write_report(run, facts, models, summary, tests, dimensions)
        manifest["stages"]["efa"] = "diagnostic_completed" if diagnostic_only else "executed"
        manifest["report_sha256"] = file_hash(run/"efa_decision_report.md")
        save_json(run/"manifest.json", manifest)
        return {"run_dir": str(run), "decision": decision, "n_pairs": validity["n_pairs"]}
    except Exception as error:
        manifest.update(failure_reason=str(error))
        manifest["stages"]["efa"] = "failed"
        save_json(run/"manifest.json", manifest)
        raise


def write_report(run, facts, models, summary, tests, dimensions):
    lines = ["# v5 EFA 决策报告", "", "候选六指标4/2结构：驱动衰减假设成分与分布—转移失序假设成分。",
             "", f"分析决策：{facts['scientific_decision']}；教师验收：未完成。", "",
             "## 数据与方法", "", json.dumps(facts["data"], ensure_ascii=False, default=str), "",
             "采用真非重叠相位、按人/记录等权、迭代PAF；旧principal/SVD与旧零均值聚合结果不作当前证据。", "",
             "## 完整模型对照与载荷", ""]
    for key, model in models.items():
        lines += [f"### {key}", "", json.dumps(model, ensure_ascii=False), ""]
    lines += ["## 平行分析参照", "", "iid参照没有校正时间依赖；中心化循环位移回答不同的记录内问题。", ""]
    for key, value in dimensions.items():
        lines += [key, "", json.dumps({k:v for k,v in value.items() if k != "shifts"}, ensure_ascii=False), ""]
    lines += ["## 被试cluster稳健性", "", json.dumps(summary, ensure_ascii=False), "",
              "## 疲劳方向及测量效度", "", f"配对被试数：{facts['n_pairs']}。", "",
              tests.to_csv(index=False), "", "## 决策、证据等级和后续", "",
              f"决策 {facts['scientific_decision']}；等级 {facts['evidence_level']}。", "",
              f"可观测共变信号：{facts['observable_signal_detected']}；这不是潜因子成立，也不能豁免PAF/C5或开启CFA。", "",
              "诊断运行不能升级为正式验收。CFA、预测、外部和不变性均未执行。", "",
              *["- "+x for x in facts["limitations"]]]
    (run/"efa_decision_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_figures(run, models, dimensions, tests):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = run/"figures"
    directory.mkdir()
    for key in ("phase_0_eight_2", "phase_0_eight_3", "phase_0_six_2", "phase_0_six_3"):
        model = models[key]
        if model["status"] != "ok":
            continue
        values = np.asarray(model["loadings"])
        fig, ax = plt.subplots(figsize=(5, 5))
        heat = ax.imshow(values, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_yticks(range(len(model["columns"])), labels=model["columns"])
        ax.set_xticks(range(values.shape[1]), labels=[f"F{i+1}" for i in range(values.shape[1])])
        for i, j in np.ndindex(values.shape):
            ax.text(j, i, f"{values[i,j]:.2f}", ha="center", va="center")
        ax.set_title(key+" (conditional candidate)")
        fig.colorbar(heat, ax=ax)
        fig.tight_layout()
        fig.savefig(directory/(key+".png"), dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, label in zip(axes, ("eight", "six")):
        item = dimensions[f"phase_0_{label}_iid"]
        if item.get("status") == "ok":
            x = np.arange(1, len(item["actual_eigenvalues"])+1)
            axis.plot(x, item["actual_eigenvalues"], "o-", label="Actual")
            axis.plot(x, item["random_eigenvalues"], "s--", label="IID P95 reference")
            axis.set_title(label+" (not cluster-corrected)")
            axis.legend()
    fig.tight_layout()
    fig.savefig(directory/"parallel_analysis_iid.png", dpi=150)
    plt.close(fig)
    if len(tests):
        fig, ax = plt.subplots(figsize=(7, 4))
        values = tests.mean_delta_standardized.to_numpy()
        low = tests.ci_low.to_numpy()
        high = tests.ci_high.to_numpy()
        # Intervals need not contain the sample point; plot independent segments.
        ax.hlines(np.arange(len(tests)), low, high)
        ax.scatter(values, np.arange(len(tests)))
        ax.set_yticks(np.arange(len(tests)), labels=tests.indicator.to_list())
        ax.axvline(0, color="grey")
        ax.set_title("Paired person-level effects (Holm tests in CSV)")
        fig.tight_layout()
        fig.savefig(directory/"paired_validity.png", dpi=150)
        plt.close(fig)
