"""Run and summarize the preregistered SI-MSET v1.2 component ablation."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eog_si_mset.config import load_config
from eog_si_mset.protocol import VARIANTS
from eog_si_mset.train import resolve_device, run_training, torch


ABLATIONS = ("single_scale", "no_adversarial", "binary_only")
REQUIRED_FILES = (
    "config_used.yaml",
    "manifest.json",
    "result.json",
    "folds.csv",
    "calibration_summary.csv",
    "calibration_state_predictions.csv",
    "bootstrap_samples.csv",
)


def _load_run(path):
    path = Path(path).expanduser().resolve()
    missing = [name for name in REQUIRED_FILES if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{path} is missing required files: {missing}")
    config = yaml.safe_load((path / "config_used.yaml").read_text(encoding="utf-8"))
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    result = json.loads((path / "result.json").read_text(encoding="utf-8"))
    folds = pd.read_csv(path / "folds.csv", dtype={"heldout_subject": str})
    calibration = pd.read_csv(path / "calibration_summary.csv")
    states = pd.read_csv(path / "calibration_state_predictions.csv", dtype={"subject": str})
    bootstrap = pd.read_csv(path / "bootstrap_samples.csv")

    if config.get("protocol") != "si-mset-v1.2" or manifest.get("protocol") != "si-mset-v1.2":
        raise ValueError(f"{path} is not an SI-MSET v1.2 run")
    variant = config["model"]["variant"]
    if variant not in VARIANTS or result.get("variant") != variant:
        raise ValueError(f"Variant metadata disagree in {path}")
    if result.get("decision") == "not_evaluable":
        raise ValueError(f"{path} is not evaluable: {result.get('reason', 'unknown reason')}")
    if folds.status.ne("ok").any():
        raise ValueError(f"{path} contains failed LOSO folds")
    if set(calibration.method) != {"raw", "center", "zscore", "rank"}:
        raise ValueError(f"{path} does not contain all four calibration summaries")
    if "rank_auc" not in bootstrap or "iteration" not in bootstrap:
        raise ValueError(f"{path} does not contain rank-calibration bootstrap samples")
    if len(bootstrap) != int(config["evaluation"]["bootstrap_iterations"]):
        raise ValueError(f"{path} has an unexpected bootstrap iteration count")
    rank_states = states.loc[states.calibration == "rank"].copy()
    keys = ["subject", "state", "label"]
    if rank_states.duplicated(keys).any() or len(rank_states) != 2 * len(manifest["paired_subjects"]):
        raise ValueError(f"{path} has an unexpected rank-calibrated state table")
    return {
        "path": path,
        "config": config,
        "manifest": manifest,
        "result": result,
        "folds": folds,
        "calibration": calibration.set_index("method"),
        "states": rank_states,
        "bootstrap": bootstrap.set_index("iteration").sort_index(),
    }


def _comparable_config(config):
    value = deepcopy(config)
    value["model"].pop("variant", None)
    # Output location and checkpoint retention do not change model fitting.
    value["output"].pop("root", None)
    value["output"].pop("save_checkpoints", None)
    return value


def _validate_reference_config(config, run):
    if _comparable_config(config) != _comparable_config(run["config"]):
        raise ValueError(
            "The reference run differs from the selected v1.2 config in a training, "
            "data, seed, or evaluation setting."
        )
    if run["config"]["model"]["variant"] != "full":
        raise ValueError("--reuse-full-run must point to a full-model run")
    if run["config"]["evaluation"].get("calibration_method") != "rank":
        raise ValueError("The v1.2 ablation requires rank as the primary calibration")


def _validate_runtime(config, reference):
    device = resolve_device(config["training"]["device"])
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else None
    expected = reference["manifest"]
    if str(device) != expected.get("device") or device_name != expected.get("cuda_device"):
        raise RuntimeError(
            "The current execution device does not match the full-model reference "
            f"(current={device}/{device_name}, reference={expected.get('device')}/"
            f"{expected.get('cuda_device')}). Run the ablations on the reference device."
        )


def _validate_runs(runs):
    full = runs["full"]
    fold_columns = ["fold", "heldout_subject", "validation_subject", "train_subjects", "seeds"]
    reference_keys = full["states"].sort_values(["subject", "state", "label"])[
        ["subject", "state", "label"]
    ].reset_index(drop=True)
    for name, run in runs.items():
        if _comparable_config(run["config"]) != _comparable_config(full["config"]):
            raise ValueError(f"{name} differs from full in a non-variant config setting")
        if run["config"]["evaluation"].get("calibration_method") != "rank":
            raise ValueError(f"{name} does not use rank as the primary calibration")
        for field in ("code_sha256", "cache_dir", "device", "cuda_device"):
            if run["manifest"].get(field) != full["manifest"].get(field):
                raise ValueError(f"{name} differs from full on manifest field {field}")
        if run["manifest"].get("paired_subjects") != full["manifest"].get("paired_subjects"):
            raise ValueError(f"{name} has a different paired-subject set")
        if run["manifest"].get("replicate_seeds") != full["manifest"].get("replicate_seeds"):
            raise ValueError(f"{name} used different random seeds")
        if len(run["folds"]) != len(full["folds"]):
            raise ValueError(f"{name} has a different LOSO fold count")
        if not run["folds"][fold_columns].reset_index(drop=True).equals(
            full["folds"][fold_columns].reset_index(drop=True)
        ):
            raise ValueError(f"{name} has different train/validation/test folds")
        if not run["bootstrap"].index.equals(full["bootstrap"].index):
            raise ValueError(f"{name} has different bootstrap iteration IDs")
        keys = run["states"].sort_values(["subject", "state", "label"])[
            ["subject", "state", "label"]
        ].reset_index(drop=True)
        if not keys.equals(reference_keys):
            raise ValueError(f"{name} has different evaluated subject-state rows")


def _summary(runs):
    full = runs["full"]
    summary_rows, delta_rows = [], []
    for name in VARIANTS:
        run = runs[name]
        calibration = run["calibration"]
        rank_boot = run["bootstrap"]["rank_auc"]
        rank_low, rank_high = np.quantile(rank_boot.to_numpy(float), [0.025, 0.975])
        rank = calibration.loc["rank"]
        summary_rows.append({
            "variant": name,
            "run_dir": str(run["path"]),
            "rank_auc": float(rank.auc),
            "rank_auc_ci_low": float(rank_low),
            "rank_auc_ci_high": float(rank_high),
            "rank_average_precision": float(rank.average_precision),
            "rank_correct_subjects": int(rank.correct_subjects),
            "raw_auc": float(calibration.loc["raw"].auc),
            "center_auc": float(calibration.loc["center"].auc),
            "zscore_auc": float(calibration.loc["zscore"].auc),
            "raw_brier": float(run["result"]["brier"]),
            "raw_log_loss": float(run["result"]["log_loss"]),
        })
        if name == "full":
            continue
        paired_delta = full["bootstrap"]["rank_auc"] - run["bootstrap"]["rank_auc"]
        low, high = np.quantile(paired_delta.to_numpy(float), [0.025, 0.975])
        if low > 0:
            interpretation = "full_higher"
        elif high < 0:
            interpretation = "ablation_higher"
        else:
            interpretation = "interval_includes_zero"
        delta_rows.append({
            "comparison": f"full_minus_{name}",
            "point_delta_auc": float(full["calibration"].loc["rank"].auc - rank.auc),
            "ci_low": float(low),
            "ci_high": float(high),
            "bootstrap_fraction_positive": float((paired_delta > 0).mean()),
            "interpretation": interpretation,
        })
    return pd.DataFrame(summary_rows), pd.DataFrame(delta_rows)


def _write_figures(output, summary, paired):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hlines(y, summary.rank_auc_ci_low, summary.rank_auc_ci_high,
              color="#52768e", linewidth=3)
    ax.scatter(summary.rank_auc, y, color="#246591", s=55, zorder=3)
    ax.set(yticks=y, yticklabels=summary.variant, xlim=(0, 1),
           xlabel="Subject-state AUC (95% subject-bootstrap CI)")
    ax.axvline(0.5, linestyle="--", color="#ba5545")
    ax.set_title("SI-MSET v1.2 component ablation")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=.16)
    fig.tight_layout()
    fig.savefig(output / "rank_auc_intervals.png", dpi=220)
    fig.savefig(output / "rank_auc_intervals.pdf")
    plt.close(fig)

    y = np.arange(len(paired))
    fig, ax = plt.subplots(figsize=(8, 4.3))
    ax.hlines(y, paired.ci_low, paired.ci_high, color="#52768e", linewidth=3)
    ax.scatter(paired.point_delta_auc, y, color="#246591", s=55, zorder=3)
    ax.axvline(0, linestyle="--", color="#ba5545")
    ax.set(yticks=y,
           yticklabels=[item.replace("full_minus_", "full − ") for item in paired.comparison],
           xlabel="Paired bootstrap ΔAUC (full − ablation)")
    ax.set_title("Contribution of v1.2 model components")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=.16)
    fig.tight_layout()
    fig.savefig(output / "paired_delta_auc.png", dpi=220)
    fig.savefig(output / "paired_delta_auc.pdf")
    plt.close(fig)


def _write_report(output, runs, summary, paired):
    config = runs["full"]["config"]
    subjects = len(runs["full"]["manifest"]["paired_subjects"])
    seeds = config["training"]["replicate_seeds"]
    lines = [
        "# SI-MSET v1.2 消融实验",
        "",
        "## 实验目的与方案",
        "",
        "本实验在 SI-MSET v1.2 固定协议下，评估多尺度编码、受试者对抗约束和辅助序数监督对清醒—困倦被试级判别的贡献。完整模型与三个单组件消融模型使用同一数据缓存、LOSO 折、训练/验证边界、优化配置和随机种子；每个变体均在全部 LOSO 折上运行三种子训练，并按预先指定方式融合 logits。",
        "",
        "主评价指标为 subject-state AUC。各模型均先对每名测试被试的未标注窗口分数执行 v1.2 预注册的 rank 校准，再聚合为该被试的清醒与困倦状态分数。校准不读取测试标签。配对差值定义为 ΔAUC = 完整模型 − 消融模型；置信区间使用相同随机抽取的被试级 bootstrap 样本计算。",
        "",
        f"本次纳入 {subjects} 名配对被试、{len(seeds)} 个随机种子（{', '.join(map(str, seeds))}），每个变体共 {subjects * len(seeds)} 次折内拟合；bootstrap 重复 {config['evaluation']['bootstrap_iterations']} 次。",
        "",
        "| 模型变体 | rank AUC（95% CI） | rank AP | 正确排序被试 | raw AUC（补充） |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples():
        lines.append(
            f"| {row.variant} | {row.rank_auc:.3f} "
            f"（{row.rank_auc_ci_low:.3f}～{row.rank_auc_ci_high:.3f}） | "
            f"{row.rank_average_precision:.3f} | {row.rank_correct_subjects}/{subjects} | "
            f"{row.raw_auc:.3f} |"
        )
    lines.extend([
        "",
        "## 配对组件效应",
        "",
        "| 比较 | ΔAUC（full − ablation） | 95% 配对 bootstrap CI | bootstrap 中 ΔAUC > 0 的比例 |",
        "|---|---:|---:|---:|",
    ])
    for row in paired.itertuples():
        lines.append(
            f"| {row.comparison} | {row.point_delta_auc:+.3f} | "
            f"[{row.ci_low:+.3f}, {row.ci_high:+.3f}] | "
            f"{row.bootstrap_fraction_positive:.1%} |"
        )
    lines.extend(["", "## 结果解释", ""])
    for row in paired.itertuples():
        label = row.comparison.removeprefix("full_minus_")
        if row.ci_low > 0:
            text = "完整模型的 rank AUC 高于该消融模型，配对区间支持相应组件的正向贡献。"
        elif row.ci_high < 0:
            text = "该消融模型的 rank AUC 高于完整模型，配对区间支持移除该组件后的提升。"
        else:
            text = "配对区间覆盖 0；当前样本对完整模型与该变体的 rank AUC 差异未形成明确区分。"
        lines.append(
            f"- {label}：ΔAUC = {row.point_delta_auc:+.3f}，"
            f"95% CI [{row.ci_low:+.3f}, {row.ci_high:+.3f}]。{text}"
        )
    lines.extend([
        "",
        "rank AUC 是本消融实验的预先指定主结果；raw AUC 与其它校准分数仅作补充描述，不替代主比较。各变体原始概率的 Brier 与 log loss 见 summary.csv。",
        "",
        "## 复核材料",
        "",
    ])
    for name in VARIANTS:
        lines.append(f"- {name}：{runs[name]['path']}")
    lines.extend([
        "- 逐变体指标：summary.csv",
        "- 配对 AUC 差值：paired_deltas.csv",
        "- 本次运行清单：run_manifest.json",
    ])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_manifest(path, runs, config):
    payload = {
        "protocol": "si-mset-v1.2",
        "primary_calibration": "rank",
        "config_sha256_by_variant": {
            name: run["manifest"].get("config_sha256") for name, run in runs.items()
        },
        "code_sha256_by_variant": {
            name: run["manifest"].get("code_sha256") for name, run in runs.items()
        },
        "run_dir_by_variant": {name: str(run["path"]) for name, run in runs.items()},
        "bootstrap_iterations": config["evaluation"]["bootstrap_iterations"],
        "bootstrap_seed": config["evaluation"]["seed"],
        "replicate_seeds": config["training"]["replicate_seeds"],
    }
    (path / "run_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/eog_si_mset_v1_2.yaml"))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--reuse-full-run", help="Reuse a completed full v1.2 run as the control")
    group.add_argument("--train-all", action="store_true", help="Train full and all ablations")
    parser.add_argument("--output", help="Summary directory; defaults to a unique timestamped path")
    args = parser.parse_args()

    config = load_config(args.config)
    if config["protocol"] != "si-mset-v1.2":
        raise ValueError("This runner requires the si-mset-v1.2 protocol")
    if config["evaluation"].get("calibration_method") != "rank":
        raise ValueError("v1.2 ablation requires rank as its primary calibration")

    runs = {}
    if args.reuse_full_run:
        runs["full"] = _load_run(args.reuse_full_run)
        _validate_reference_config(config, runs["full"])
        _validate_runtime(config, runs["full"])

    # Checkpoints are not used by the paired analysis; keep only reproducibility artifacts.
    config["output"]["save_checkpoints"] = False

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(args.output).expanduser() if args.output else (
        Path(config["output"]["root"]) / "ablation_v12" / f"{timestamp}-{uuid4().hex[:8]}"
    )
    if not output.is_absolute():
        output = ROOT / output
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    variants = VARIANTS if args.train_all else ABLATIONS
    try:
        for variant in variants:
            print(f"[v1.2 ablation] variant={variant}", flush=True)
            run_result = run_training(config, ROOT, variant=variant)
            run = _load_run(run_result["run_dir"])
            runs[variant] = run
            _write_manifest(output, runs, config)

        _validate_runs(runs)
        summary, paired = _summary(runs)
        summary.to_csv(output / "summary.csv", index=False)
        paired.to_csv(output / "paired_deltas.csv", index=False)
        _write_figures(output, summary, paired)
        _write_report(output, runs, summary, paired)
        _write_manifest(output, runs, config)
    except Exception as error:
        _write_manifest(output, runs, config)
        (output / "failure.txt").write_text(
            f"{type(error).__name__}: {error}\n", encoding="utf-8"
        )
        raise

    print(f"Summary directory: {output}")
    print(summary.to_string(index=False))
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
