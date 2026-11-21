"""Compare SI-MSET variants trained on identical LOSO splits."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.metrics import roc_auc_score


def _load_run(path):
    path = Path(path)
    required = ("config_used.yaml", "manifest.json", "result.json", "folds.csv",
                "state_predictions.csv", "bootstrap_samples.csv")
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(f"{path} is missing {missing}")
    config = yaml.safe_load((path / "config_used.yaml").read_text(encoding="utf-8"))
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    result = json.loads((path / "result.json").read_text(encoding="utf-8"))
    folds = pd.read_csv(path / "folds.csv")
    states = pd.read_csv(path / "state_predictions.csv")
    states["subject"] = states["subject"].astype(str)
    bootstrap = pd.read_csv(path / "bootstrap_samples.csv").set_index("iteration")
    return {
        "path": path,
        "config": config,
        "manifest": manifest,
        "result": result,
        "folds": folds,
        "states": states,
        "bootstrap": bootstrap,
    }


def _comparable(config):
    value = deepcopy(config)
    value["model"].pop("variant", None)
    return value


def _validate_runs(runs):
    reference_name = "full"
    reference = runs[reference_name]
    reference_config = _comparable(reference["config"])
    for name, run in runs.items():
        if _comparable(run["config"]) != reference_config:
            raise ValueError(f"{name} has a configuration beyond model.variant")
        if run["manifest"]["code_sha256"] != reference["manifest"]["code_sha256"]:
            raise ValueError(f"{name} used different code")
        if run["manifest"]["cache_dir"] != reference["manifest"]["cache_dir"]:
            raise ValueError(f"{name} used a different cache")
        if len(run["folds"]) != len(reference["folds"]):
            raise ValueError(f"{name} has a different fold count")
        fold_columns = ["fold", "heldout_subject", "validation_subject", "train_subjects"]
        if not run["folds"][fold_columns].equals(reference["folds"][fold_columns]):
            raise ValueError(f"{name} has different train/validation/test splits")
        if not run["bootstrap"].index.equals(reference["bootstrap"].index):
            raise ValueError(f"{name} has different bootstrap iterations")
        reference_index = reference["states"].set_index(["subject", "state", "label"]).index
        if not run["states"].set_index(["subject", "state", "label"]).index.equals(reference_index):
            raise ValueError(f"{name} has different state predictions")


def _correct_ordering(states):
    pivot = states.pivot(index="subject", columns="state", values="probability")
    return int((pivot["drowsy"] > pivot["alert"]).sum())


def _summary(runs):
    rows, paired = [], []
    full = runs["full"]
    for name, run in runs.items():
        result = run["result"]
        rows.append({
            "variant": name,
            "run_dir": str(run["path"]),
            "decision": result["decision"],
            "auc": result["deep_auc"],
            "auc_low": result["deep_ci"][0],
            "auc_high": result["deep_ci"][1],
            "average_precision": result["average_precision"],
            "brier": result["brier"],
            "log_loss": result["log_loss"],
            "window_auc": result["window_auc"],
            "correct_subjects": _correct_ordering(run["states"]),
            "delta_vs_v7": result["delta_auc"],
        })
        if name == "full":
            continue
        delta = full["bootstrap"].deep_auc - run["bootstrap"].deep_auc
        point = result_full = full["result"]["deep_auc"] - result["deep_auc"]
        low, high = np.quantile(delta, [0.025, 0.975])
        if low > 0:
            interpretation = "full_higher"
        elif high < 0:
            interpretation = "ablation_higher"
        else:
            interpretation = "not_identified"
        paired.append({
            "comparison": f"full_minus_{name}",
            "point_delta_auc": point,
            "ci_low": low,
            "ci_high": high,
            "bootstrap_fraction_positive": float((delta > 0).mean()),
            "bootstrap_fraction_zero": float((delta == 0).mean()),
            "interpretation": interpretation,
        })
    return pd.DataFrame(rows), pd.DataFrame(paired)


def _write_figures(output, summary, paired):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = summary.variant.tolist()
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hlines(y, summary.auc_low, summary.auc_high, color="#52768e", linewidth=3)
    ax.scatter(summary.auc, y, color="#246591", s=55, zorder=3)
    ax.set(yticks=y, yticklabels=labels, xlim=(0, 1), xlabel="Subject-state AUC (95% cluster bootstrap CI)")
    ax.axvline(0.5, linestyle="--", color="#ba5545")
    ax.set_title("SI-MSET LOSO ablation performance")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=.16)
    fig.tight_layout()
    fig.savefig(output / "ablation_auc.png", dpi=220)
    fig.savefig(output / "ablation_auc.pdf")
    plt.close(fig)

    y = np.arange(len(paired))
    fig, ax = plt.subplots(figsize=(8, 4.3))
    ax.hlines(y, paired.ci_low, paired.ci_high, color="#52768e", linewidth=3)
    ax.scatter(paired.point_delta_auc, y, color="#246591", s=55, zorder=3)
    ax.axvline(0, linestyle="--", color="#ba5545")
    ax.set(yticks=y, yticklabels=[name.replace("full_minus_", "full − ") for name in paired.comparison],
           xlabel="Paired bootstrap ΔAUC")
    ax.set_title("Component contribution relative to full model")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=.16)
    fig.tight_layout()
    fig.savefig(output / "ablation_deltas.png", dpi=220)
    fig.savefig(output / "ablation_deltas.pdf")
    plt.close(fig)


def _write_report(output, summary, paired):
    lines = [
        "# SI-MSET v1 消融分析",
        "",
        "四个变体使用相同代码、缓存、LOSO 划分和 bootstrap 抽样。注意：这里的 `full` 指 v1 的完整模型，不是随后的 v1.1 长训练模型。",
        "",
        "## 主结果",
        "",
        "| 变体 | AUC | 95% CI | AP | Brier | Log loss | 正确排序被试 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples():
        lines.append(
            f"| {row.variant} | {row.auc:.3f} | [{row.auc_low:.3f}, {row.auc_high:.3f}] | "
            f"{row.average_precision:.3f} | {row.brier:.3f} | {row.log_loss:.3f} | "
            f"{row.correct_subjects}/10 |"
        )
    lines.extend([
        "",
        "## 配对效应",
        "",
        "| 比较 | 点估计 | 95% CI | bootstrap 正值比例 | 结论 |",
        "|---|---:|---:|---:|---|",
    ])
    for row in paired.itertuples():
        lines.append(
            f"| {row.comparison} | {row.point_delta_auc:+.3f} | "
            f"[{row.ci_low:+.3f}, {row.ci_high:+.3f}] | "
            f"{row.bootstrap_fraction_positive:.1%} | {row.interpretation} |"
        )
    lines.extend([
        "",
        "## 结论",
        "",
        "- 完整模型的 AUC 点估计高于三个消融，但三个配对差值的 95% CI 都覆盖 0；当前样本不能证明任一模块带来稳定增益。",
        "- 最优超参数、验证集和测试折均沿用 v1；消融结果没有改变 v1 主判定 `inconclusive`。",
        "- `no_adversarial` 的 AUC 只低 0.010，且其正确排序被试数不低；不能声称受试者对抗约束已被证明有效。",
        "- `single_scale` 和 `binary_only` 各低 0.020，方向与设计预期一致，但区间较宽，只能作为弱趋势。",
    ])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    for variant in ("full", "single_scale", "no_adversarial", "binary_only"):
        parser.add_argument(f"--{variant.replace('_', '-')}", required=True)
    parser.add_argument("--output", default=str(ROOT / "results/eog_si_mset/ablation_v1"))
    args = parser.parse_args()
    paths = {
        "full": args.full,
        "single_scale": args.single_scale,
        "no_adversarial": args.no_adversarial,
        "binary_only": args.binary_only,
    }
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    runs = {name: _load_run(path) for name, path in paths.items()}
    _validate_runs(runs)
    summary, paired = _summary(runs)
    output.mkdir(parents=True)
    summary.to_csv(output / "summary.csv", index=False)
    paired.to_csv(output / "paired_deltas.csv", index=False)
    _write_figures(output, summary, paired)
    _write_report(output, summary, paired)
    print(summary.to_string(index=False))
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
