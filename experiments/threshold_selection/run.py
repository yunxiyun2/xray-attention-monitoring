from pathlib import Path
import yaml
import json
import pandas as pd

from xray_attention.data.records import discover_records
from xray_attention.attention.metrics import summarize_errors
from xray_attention.attention.evaluation import evaluate_nested_loso
from xray_attention.attention.plots import (
    plot_threshold_performance,
    plot_selected_thresholds,
)


def run_experiment(config_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_root = Path(config["data_root"])
    labels = config["labels"]
    difficulties = config["task_difficulties"]
    thresholds_spec = config["thresholds_px"]
    thresholds = list(
        range(
            thresholds_spec["start"],
            thresholds_spec["stop"] + 1,
            thresholds_spec["step"],
        )
    )

    records = discover_records(data_root, labels, difficulties)

    task_features = []
    for record in records:
        summary = summarize_errors(record.errors, thresholds)
        row = {
            "subject_id": record.subject_id,
            "difficulty": record.difficulty,
            "label_name": record.label_name,
            "label": record.label,
        }
        row.update(summary)
        task_features.append(row)

    features_df = pd.DataFrame(task_features)
    features_df.to_csv(
        output_dir / "task_features.csv", index=False, encoding="utf-8"
    )

    candidate_summary, nested_loso_folds, recommendations = evaluate_nested_loso(
        features_df, thresholds
    )

    candidate_summary.to_csv(
        output_dir / "candidate_summary.csv", index=False, encoding="utf-8"
    )
    nested_loso_folds.to_csv(
        output_dir / "nested_loso_folds.csv", index=False, encoding="utf-8"
    )

    with open(output_dir / "recommended_threshold.json", "w", encoding="utf-8") as f:
        json.dump(recommendations, f)

    plot_threshold_performance(candidate_summary, output_dir)
    plot_selected_thresholds(nested_loso_folds, output_dir)

    with open(output_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    output_dir = Path(config["output_dir"])
    run_experiment(args.config, output_dir)
