"""EFA 可视化图表。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dual_factor.analysis.efa import EFAResult
from dual_factor.analysis.descriptive import directionality_test


def plot_scree_parallel(efa_result: EFAResult, out_path: str | Path) -> None:
    """碎石图 + 平行分析图。"""
    pa = efa_result.parallel_analysis
    actual = pa["actual_eigenvalues"]
    random = pa["random_eigenvalues"]
    k = len(actual)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, k + 1), actual, "o-", label="Actual eigenvalues", color="#1f77b4")
    ax.plot(range(1, k + 1), random, "s--", label="Parallel (95th pct)", color="#d62728")
    ax.axhline(1.0, color="gray", linestyle=":", label="Kaiser criterion (>=1)")
    ax.set_xlabel("Factor number")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Scree Plot with Parallel Analysis")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(out_path), dpi=150)
    plt.close(fig)


def plot_loading_matrix(efa_result: EFAResult, out_path: str | Path) -> None:
    """因子载荷热力图。"""
    loadings = efa_result.loadings
    vars_names = efa_result.variable_names
    n_factors = efa_result.n_factors
    fig, ax = plt.subplots(figsize=(6, 8))
    im = ax.imshow(loadings, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n_factors))
    ax.set_xticklabels([f"F{i+1}" for i in range(n_factors)])
    ax.set_yticks(range(len(vars_names)))
    ax.set_yticklabels(vars_names)
    for i in range(loadings.shape[0]):
        for j in range(loadings.shape[1]):
            ax.text(j, i, f"{loadings[i, j]:.2f}", ha="center", va="center",
                    color="white" if abs(loadings[i, j]) > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, label="Loading")
    ax.set_title("Factor Loading Matrix (Promax rotated)")
    fig.tight_layout()
    fig.savefig(Path(out_path), dpi=150)
    plt.close(fig)


def plot_factor_scatter(efa_result: EFAResult, out_path: str | Path) -> None:
    """因子相关散点图（若 PHI 矩阵可用）。"""
    if efa_result.factor_corr is None:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    corr = efa_result.factor_corr
    n = corr.shape[0]
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"F{i+1}" for i in range(n)])
    ax.set_yticklabels([f"F{i+1}" for i in range(n)])
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=10)
    fig.colorbar(im, ax=ax, label="Correlation")
    ax.set_title("Factor Correlation Matrix (Phi)")
    fig.tight_layout()
    fig.savefig(Path(out_path), dpi=150)
    plt.close(fig)


def plot_direction_check(df: pd.DataFrame, indicators: list[str], out_path: str | Path) -> None:
    """各指标 alert vs drowsy 方向检验图。"""
    direction_df = directionality_test(df, indicators)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2ca02c" if m else "#d62728" for m in direction_df["match"]]
    ax.barh(direction_df["indicator"], direction_df["diff"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean difference (drowsy - alert)")
    ax.set_title("Directionality Check: Indicator Changes (alert -> drowsy)\nGreen=matches hypothesis, Red=contradicts")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(out_path), dpi=150)
    plt.close(fig)
