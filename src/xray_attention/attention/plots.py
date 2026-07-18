from pathlib import Path
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_threshold_performance(candidate_summary, output_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    regimes = candidate_summary["regime"].unique() if "regime" in candidate_summary.columns else ["overall"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    
    for i, regime in enumerate(regimes):
        if "regime" in candidate_summary.columns:
            subset = candidate_summary[candidate_summary["regime"] == regime]
        else:
            subset = candidate_summary
        ax.plot(
            subset["threshold"],
            subset["roc_auc"],
            marker="o",
            label=regime,
            color=colors[i % len(colors)]
        )
    
    ax.set_xlabel("Threshold (px)")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Threshold Performance by ROC-AUC")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.0, 1.0])
    
    fig.tight_layout()
    fig.savefig(output_dir / "threshold_performance.png", dpi=150)
    plt.close(fig)


def plot_selected_thresholds(nested_loso_folds, output_dir):
    regimes = nested_loso_folds["regime"].unique()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    
    fig, axes = plt.subplots(1, len(regimes), figsize=(12, 5), sharey=True)
    if len(regimes) == 1:
        axes = [axes]
    
    for i, regime in enumerate(regimes):
        subset = nested_loso_folds[nested_loso_folds["regime"] == regime]
        thresholds = subset["selected_threshold"].values
        
        unique_thresholds, counts = np.unique(thresholds, return_counts=True)
        
        axes[i].bar(unique_thresholds, counts, color=colors[i % len(colors)], alpha=0.7)
        axes[i].set_xlabel("Selected Threshold (px)")
        axes[i].set_ylabel("Frequency")
        axes[i].set_title(f"Selected Thresholds - {regime}")
        axes[i].grid(axis="y", alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(output_dir / "selected_thresholds.png", dpi=150)
    plt.close(fig)
