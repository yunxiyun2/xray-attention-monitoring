"""Auditable v7 incremental predictive-validity analysis."""
from pathlib import Path
import json
import platform

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from dual_factor.config_v7 import config_hash
from dual_factor.protocol_v7 import (
    F1_COMPONENTS,
    F2_COMPONENTS,
    MODELS,
    RETAINED,
    Z_COLUMNS,
    require_columns,
)
from dual_factor.run_manifest import code_hash, create_run, file_hash, save_json


SOURCE_COLUMNS = ("subject", "test", "kss", "window_idx", "valid", *RETAINED)


def _path(root, value):
    path = Path(value)
    return path if path.is_absolute() else Path(root) / path


def _validate_source(frame, analysis):
    require_columns(frame.columns, SOURCE_COLUMNS)
    if frame.duplicated(["subject", "test", "window_idx"]).any():
        raise ValueError("Duplicate subject/test/window_idx primary key")
    if not pd.api.types.is_bool_dtype(frame["valid"]):
        raise ValueError("valid must contain booleans")
    frame["subject"] = frame["subject"].astype(str)
    frame["test"] = frame["test"].astype(str)
    kss = pd.to_numeric(frame["kss"], errors="raise").to_numpy(float)
    if not np.isfinite(kss).all() or not np.equal(kss, np.floor(kss)).all():
        raise ValueError("KSS must contain finite integers")
    frame["kss"] = kss.astype(int)
    if frame.groupby(["subject", "test"]).kss.nunique().max() > 1:
        raise ValueError("KSS changes within a recording")
    for column in ("window_idx", *RETAINED):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    valid = frame.loc[frame.valid, list(RETAINED)].to_numpy(float)
    if not np.isfinite(valid).all():
        raise ValueError("Valid rows contain nonfinite indicators")
    if "start_sec" in frame.columns or "end_sec" in frame.columns:
        require_columns(frame.columns, ["start_sec", "end_sec"])
        for column in ("start_sec", "end_sec"):
            frame[column] = pd.to_numeric(frame[column], errors="raise")
    else:
        frame["start_sec"] = frame["window_idx"] * analysis["stride_sec"]
        frame["end_sec"] = frame["start_sec"] + analysis["window_sec"]
    times = frame[["start_sec", "end_sec"]].to_numpy(float)
    expected_start = frame["window_idx"].to_numpy(float) * analysis["stride_sec"]
    if not np.isfinite(times).all() or not np.allclose(times[:, 0], expected_start):
        raise ValueError("Window indices and recording-relative time intervals disagree")
    if not np.allclose(times[:, 1] - times[:, 0], analysis["window_sec"]):
        raise ValueError("Window durations disagree with configuration")
    if (analysis["alert_max"] >= analysis["drowsy_min"]):
        raise ValueError("Invalid KSS thresholds")


def _within_subject_standardize(valid):
    standardized = valid.copy()
    for subject, person in valid.groupby("subject", sort=True):
        for column in RETAINED:
            values = person[column].astype(float)
            mean = float(values.mean())
            scale = float(values.std(ddof=0))
            if not np.isfinite(scale) or scale <= 1e-20:
                raise ValueError(f"Degenerate within-subject scale: {subject}/{column}")
            standardized.loc[person.index, f"{column}_z"] = (values - mean) / scale
    return standardized


def _nonoverlap_phase(frame, analysis):
    step = int(round(analysis["window_sec"] / analysis["stride_sec"]))
    phase_frame = frame.loc[
        frame["window_idx"].astype(int).mod(step).eq(analysis["primary_phase"])
    ].copy()
    phase_frame = phase_frame.sort_values(["subject", "test", "start_sec"])
    for _, recording in phase_frame.groupby(["subject", "test"], sort=False):
        start = recording["start_sec"].to_numpy(float)
        end = recording["end_sec"].to_numpy(float)
        if np.any(start[1:] < end[:-1] - 1e-10):
            raise ValueError("Selected phase contains overlapping windows")
    return phase_frame


def _score_rows(state_records):
    rows = []
    for state, state_data in state_records.items():
        records = state_data["records"]
        record_means = []
        for record in records:
            record_means.append(record[list(Z_COLUMNS)].mean(axis=0))
        mean = pd.DataFrame(record_means).mean(axis=0)
        f1 = np.mean([mean[f"{column}_z"] * direction for column, direction in F1_COMPONENTS.items()])
        f2 = np.mean([mean[f"{column}_z"] * direction for column, direction in F2_COMPONENTS.items()])
        rows.append({
            "state": state,
            "label": 1 if state == "drowsy" else 0,
            "f1": float(f1),
            "f2": float(f2),
            "global": float((f1 + f2) / 2),
            **{column: float(mean[column]) for column in Z_COLUMNS},
        })
    return rows


def build_paired_dataset(frame, analysis):
    _validate_source(frame, analysis)
    valid = frame.loc[frame.valid].copy()
    standardized = _within_subject_standardize(valid)
    phase = _nonoverlap_phase(standardized, analysis)

    subjects = []
    excluded = []
    for subject, person in phase.groupby("subject", sort=True):
        states = {}
        for state, mask in (
            ("alert", person.kss <= analysis["alert_max"]),
            ("drowsy", person.kss >= analysis["drowsy_min"]),
        ):
            recordings = []
            record_tests = []
            for test, record in person.loc[mask].groupby("test", sort=True):
                recordings.append(record)
                record_tests.append(test)
            if recordings and sum(len(record) for record in recordings) >= analysis["min_state_windows"]:
                states[state] = {"records": recordings, "tests": record_tests}
        if len(states) != 2:
            excluded.append({
                "subject": subject,
                "reason": "missing_alert_or_drowsy_after_window_gate",
                "alert_windows": sum(len(record) for record in states.get("alert", {"records": []})["records"]),
                "drowsy_windows": sum(len(record) for record in states.get("drowsy", {"records": []})["records"]),
            })
            continue
        scored = _score_rows(states)
        for row, state in zip(scored, ("alert", "drowsy")):
            rows = states[state]["records"]
            scaled_kss = person.loc[
                person.test.isin(states[state]["tests"]), "kss"
            ]
            row.update({
                "subject": subject,
                "kss_min": int(scaled_kss.min()),
                "kss_max": int(scaled_kss.max()),
                "records": "|".join(sorted(states[state]["tests"])),
                "n_records": len(rows),
                "n_windows": sum(len(record) for record in rows),
            })
            subjects.append(row)

    data = pd.DataFrame(subjects)
    audit = {
        "status": "ok" if len(data) else "not_evaluable",
        "n_subjects": int(data.subject.nunique()) if len(data) else 0,
        "n_subject_state_rows": int(len(data)),
        "excluded_subjects": excluded,
        "minimum_windows_per_record": analysis["min_state_windows"],
        "primary_phase": analysis["primary_phase"],
        "window_sec": analysis["window_sec"],
        "stride_sec": analysis["stride_sec"],
    }
    return data.reset_index(drop=True), audit


def fit_loso(data, model_config):
    subjects = sorted(data.subject.unique())
    predictions = []
    folds = []
    for held_subject in subjects:
        train = data.loc[data.subject != held_subject].copy()
        test = data.loc[data.subject == held_subject].copy()
        if set(train.subject).intersection(test.subject):
            raise ValueError("Training and held-out subjects overlap")
        if train.label.nunique() != 2:
            raise ValueError(f"Training fold lacks both classes for {held_subject}")
        for model_name, features in MODELS.items():
            scaler = StandardScaler()
            x_train = scaler.fit_transform(train[list(features)])
            x_test = scaler.transform(test[list(features)])
            classifier = LogisticRegression(
                C=model_config["regularization_c"],
                max_iter=model_config["max_iter"],
                random_state=model_config["seed"],
                solver="lbfgs",
            )
            classifier.fit(x_train, train.label)
            probability = classifier.predict_proba(x_test)[:, 1]
            for (_, row), prob, decision in zip(
                test.iterrows(), probability, classifier.decision_function(x_test)
            ):
                predictions.append({
                    "model": model_name,
                    "held_subject": held_subject,
                    "subject": row.subject,
                    "state": row.state,
                    "label": int(row.label),
                    "probability": float(prob),
                    "decision_score": float(decision),
                })
            folds.append({
                "model": model_name,
                "held_subject": held_subject,
                "n_train_subjects": int(train.subject.nunique()),
                "n_train_rows": int(len(train)),
                "n_test_rows": int(len(test)),
                "train_subjects": "|".join(sorted(train.subject.unique())),
            })
    return pd.DataFrame(predictions), pd.DataFrame(folds)


def metrics_for_predictions(predictions):
    rows = []
    for model_name in MODELS:
        model_predictions = predictions.loc[predictions.model == model_name]
        probability = model_predictions.probability.to_numpy(float)
        label = model_predictions.label.to_numpy(int)
        rows.extend([
            {"model": model_name, "metric": "roc_auc", "estimate": roc_auc_score(label, probability)},
            {"model": model_name, "metric": "average_precision", "estimate": average_precision_score(label, probability)},
            {"model": model_name, "metric": "brier", "estimate": brier_score_loss(label, probability)},
            {"model": model_name, "metric": "log_loss", "estimate": log_loss(label, np.clip(probability, 1e-12, 1 - 1e-12))},
        ])
    return pd.DataFrame(rows)


def _bootstrap_metrics(predictions, iterations, seed):
    subjects = sorted(predictions.subject.unique())
    grouped = {subject: group for subject, group in predictions.groupby("subject")}
    rng = np.random.default_rng(seed)
    rows = []
    for iteration in range(iterations):
        sampled_subjects = rng.choice(subjects, len(subjects), replace=True)
        sample = pd.concat([grouped[subject] for subject in sampled_subjects], ignore_index=True)
        values = metrics_for_predictions(sample).pivot(index="model", columns="metric", values="estimate")
        for model_name in MODELS:
            for metric in ("roc_auc", "average_precision", "brier", "log_loss"):
                rows.append({
                    "iteration": iteration,
                    "model": model_name,
                    "metric": metric,
                    "value": float(values.loc[model_name, metric]),
                })
        rows.append({
            "iteration": iteration,
            "model": "two_factor_minus_global",
            "metric": "roc_auc_difference",
            "value": float(values.loc["two_factor", "roc_auc"] - values.loc["global", "roc_auc"]),
        })
    return pd.DataFrame(rows)


def summarize_bootstrap(point_metrics, bootstrap):
    rows = []
    for (model_name, metric), group in bootstrap.groupby(["model", "metric"], sort=True):
        if model_name == "two_factor_minus_global":
            estimate = (
                point_metrics.set_index(["model", "metric"]).loc[("two_factor", "roc_auc"), "estimate"]
                - point_metrics.set_index(["model", "metric"]).loc[("global", "roc_auc"), "estimate"]
            )
        else:
            estimate = point_metrics.set_index(["model", "metric"]).loc[(model_name, metric), "estimate"]
        low, high = np.quantile(group.value.to_numpy(float), [0.025, 0.975])
        rows.append({
            "model": model_name,
            "metric": metric,
            "estimate": float(estimate),
            "ci_low": float(low),
            "ci_high": float(high),
            "bootstrap_iterations": int(len(group)),
        })
    return pd.DataFrame(rows)


def decide(delta_auc, ci_low, ci_high, delta_auc_min):
    if delta_auc >= delta_auc_min and ci_low > 0:
        return "incremental_supported"
    if delta_auc < 0 and ci_high < 0:
        return "global_better"
    return "inconclusive"


def write_report(path, decision, audit, metrics, bootstrap, input_hash):
    metric_table = metrics.pivot(index="model", columns="metric", values="estimate")
    display_lines = [
        "| Model | ROC AUC | Average precision | Brier | Log loss |",
        "|---|---:|---:|---:|---:|",
    ]
    for model_name in MODELS:
        row = metric_table.loc[model_name]
        display_lines.append(
            f"| {model_name} | {row['roc_auc']:.3f} | {row['average_precision']:.3f} | "
            f"{row['brier']:.3f} | {row['log_loss']:.3f} |"
        )
    delta = bootstrap.loc[
        (bootstrap.model == "two_factor_minus_global")
        & (bootstrap.metric == "roc_auc_difference")
    ].iloc[0]
    lines = [
        "# v7 预测效度实验报告",
        "",
        f"- 输入 SHA256：`{input_hash}`",
        f"- 配对被试：{audit['n_subjects']}",
        f"- 分析单位：被试级清醒/困倦记录；phase {audit['primary_phase']} 非重叠窗口",
        "",
        "## 结果",
        "",
        *display_lines,
        "",
        f"主差值 ΔAUC（two_factor - global）= {delta.estimate:.3f}，"
        f"被试整群 bootstrap 95% CI [{delta.ci_low:.3f}, {delta.ci_high:.3f}]。",
        "",
        f"判定：**{decision['decision']}**。",
        "",
        "## 解释边界",
        "",
        "- 本实验检验预定义行为分数的增量预测价值，不检验或升级 v6 的潜因子结构。",
        "- 10 名左右配对被试的外样本 AUC 和区间可能很宽；未检出差异不等于证明没有差异。",
        "- KSS 离散分组、DROZY 场景和单一数据源限制外推。",
        "- 本次未进行超参数搜索、CFA、跨域验证或因果推断。",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figures(run, predictions, bootstrap):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    figure_dir = run / "figures"
    figure_dir.mkdir()
    colors = {
        "global": "#4c6575",
        "f1_only": "#8c5f38",
        "f2_only": "#497d70",
        "two_factor": "#246591",
        "seven_feature": "#8f4b6b",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for model_name in MODELS:
        subset = predictions.loc[predictions.model == model_name]
        fpr, tpr, _ = roc_curve(subset.label, subset.probability)
        auc = roc_auc_score(subset.label, subset.probability)
        axes[0].plot(fpr, tpr, label=f"{model_name} ({auc:.3f})", color=colors[model_name])
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#9aa5ad")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="Leave-one-subject-out ROC")
    axes[0].legend(frameon=False, fontsize=8)

    auc = bootstrap.loc[bootstrap.metric == "roc_auc"].groupby("model").value.quantile([0.025, 0.5, 0.975]).unstack()
    model_names = list(MODELS)
    y = np.arange(len(model_names))
    axes[1].hlines(y, auc.loc[model_names, 0.025], auc.loc[model_names, 0.975],
                   color="#52768e", linewidth=2)
    axes[1].scatter(auc.loc[model_names, 0.5], y, color="#246591", zorder=3)
    axes[1].set(yticks=y, yticklabels=model_names, xlim=(0, 1.02),
                xlabel="AUC with subject-cluster bootstrap 95% CI")
    axes[1].axvline(0.5, color="#ba5545", linestyle="--", linewidth=1)
    axes[1].set_title("Out-of-fold discrimination")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(figure_dir / "roc_and_auc.png", dpi=220)
    fig.savefig(figure_dir / "roc_and_auc.pdf")
    plt.close(fig)


def _save_not_evaluable(run, config, audit, input_hash):
    decision = {
        "decision": "not_evaluable",
        "reason": "insufficient paired subjects or windows",
        "required_min_pairs": config["analysis"]["min_pairs"],
        "n_pairs": audit["n_subjects"],
    }
    audit = {**audit, "input_sha256": input_hash}
    save_json(run / "data_audit.json", audit)
    save_json(run / "decision.json", decision)
    (run / "report.md").write_text(
        "# v7 预测效度实验报告\n\n"
        "判定：**not_evaluable**。\n\n"
        f"仅获得 {audit['n_subjects']} 名合格配对被试，低于预注册门槛 "
        f"{config['analysis']['min_pairs']}。未运行模型或补充样本。\n",
        encoding="utf-8",
    )
    return decision


def run_experiment(config, project_root, dry_run=False):
    root = Path(project_root)
    input_path = _path(root, config["input"]["path"])
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    frame = pd.read_csv(input_path)
    data, audit = build_paired_dataset(frame, config["analysis"])
    input_hash = file_hash(input_path)
    audit.update({
        "input_path": str(input_path.relative_to(root)),
        "input_sha256": input_hash,
        "min_pairs": config["analysis"]["min_pairs"],
    })
    ready = audit["n_subjects"] >= config["analysis"]["min_pairs"]
    if not ready:
        audit["status"] = "not_evaluable"
    if dry_run:
        return {"status": audit["status"], "ready": ready, "audit": audit}

    output_root = _path(root, config["output"]["root"])
    output_root.mkdir(parents=True, exist_ok=True)
    run = create_run(output_root)
    (run / "config_used.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    manifest = {
        "protocol": "v7.0",
        "run": run.name,
        "input_sha256": input_hash,
        "config_sha256": config_hash(config),
        "code_sha256": code_hash(root),
        "platform": platform.platform(),
    }
    save_json(run / "manifest.json", manifest)
    if not ready:
        decision = _save_not_evaluable(run, config, audit, input_hash)
        return {"run_dir": str(run), "decision": decision, "audit": audit}

    predictions, folds = fit_loso(data, config["model"])
    point_metrics = metrics_for_predictions(predictions)
    raw_bootstrap = _bootstrap_metrics(
        predictions, config["bootstrap"]["iterations"], config["bootstrap"]["seed"]
    )
    bootstrap = summarize_bootstrap(
        point_metrics, raw_bootstrap
    )
    delta = bootstrap.loc[
        (bootstrap.model == "two_factor_minus_global")
        & (bootstrap.metric == "roc_auc_difference")
    ].iloc[0]
    verdict = decide(
        float(delta.estimate),
        float(delta.ci_low),
        float(delta.ci_high),
        float(config["analysis"]["delta_auc_min"]),
    )
    decision = {
        "decision": verdict,
        "n_pairs": audit["n_subjects"],
        "primary_metric": "two_factor_minus_global_auc",
        "delta_auc": float(delta.estimate),
        "ci_low": float(delta.ci_low),
        "ci_high": float(delta.ci_high),
        "delta_auc_min": float(config["analysis"]["delta_auc_min"]),
        "efa_v6_decision": "B",
        "latent_structure_claim": "not established",
        "bootstrap": "subject-cluster, paired states retained",
        "bootstrap_iterations": int(config["bootstrap"]["iterations"]),
    }
    save_json(run / "data_audit.json", audit)
    data.to_csv(run / "subject_summary.csv", index=False)
    predictions.to_csv(run / "fold_predictions.csv", index=False)
    folds.to_csv(run / "fold_audit.csv", index=False)
    point_metrics.to_csv(run / "metrics.csv", index=False)
    bootstrap.to_csv(run / "bootstrap_summary.csv", index=False)
    raw_bootstrap.to_csv(run / "bootstrap_samples.csv", index=False)
    save_json(run / "decision.json", decision)
    write_report(run / "report.md", decision, audit, point_metrics, bootstrap, input_hash)
    write_figures(run, predictions, raw_bootstrap)
    return {"run_dir": str(run), "decision": decision, "audit": audit}
