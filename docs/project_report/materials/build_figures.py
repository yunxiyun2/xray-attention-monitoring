"""Render existing v6 artifacts for the practice report; no experiment is rerun.

Run from any directory with the project's .venv Python.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "results/dual_factor_efa_v6/20261115T142308Z-fd3e4386"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.facecolor": "white"})
SOURCES = {}


def read_json(path):
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


models = read_json(RUN / "efa_models.json")
pa = read_json(RUN / "parallel_analysis.json")
facts = read_json(RUN / "facts.json")
certificate = read_json(ROOT / "results/dual_factor_efa_v6/simulation_certificate.json")
effects = pd.DataFrame(facts["hypothesis_validity"])
model = models["phase_0_seven_2"]


def correlation(ax):
    values = np.array(model["correlation"])
    heat = ax.imshow(values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(7), model["columns"])
    ax.set_yticks(range(7), model["columns"])
    for i, j in np.ndindex(values.shape):
        ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center",
                fontsize=9, color="white" if abs(values[i, j]) > .65 else "#263244")
    ax.set_title("A. Indicator correlation\nPhase 0 | subject-balanced | 706 windows", pad=12)
    ax.figure.colorbar(heat, ax=ax, fraction=.045, pad=.04, label="Correlation")


def parallel(ax):
    colors = ["#246591", "#228879", "#bd7430"]
    for phase, color in enumerate(colors):
        item = pa[f"phase_{phase}_seven_record_centered_shift"]
        x = np.arange(1, 8)
        ax.plot(x, item["actual_eigenvalues"], "o-", color=color, label=f"Phase {phase}: observed")
        ax.plot(x, item["random_eigenvalues"], ":", color=color, alpha=.7)
    ax.set(xlabel="Eigenvalue rank", ylabel="Eigenvalue", xticks=range(1, 8), ylim=(0, 2.8))
    ax.set_title("B. Parallel analysis\nRecord-centered shift | dotted: null P95", pad=12)
    ax.legend(frameon=False, fontsize=9)
    ax.text(.97, .52, "All phases suggest 2 dimensions.\nThis does not establish the factor model.",
            ha="right", transform=ax.transAxes, fontsize=9)
    ax.grid(axis="y", alpha=.16)


def paired(ax):
    y = np.arange(len(effects))
    ax.hlines(y, effects.ci_low, effects.ci_high, color="#52768e", linewidth=2)
    ax.scatter(effects.mean_delta_standardized, y, color="#246591", s=45, zorder=3)
    ax.axvline(0, color="#ba5545", linestyle="--", linewidth=1)
    ax.set(yticks=y, yticklabels=effects.indicator, xlabel="Mean standardized difference (sleepy - alert)")
    ax.invert_yaxis()
    ax.set_title("C. Paired subject-level differences\nn = 10 | marginal 95% CIs | all Holm p >= 0.902", pad=12)
    ax.grid(axis="x", alpha=.16)


def simulation(ax):
    scenarios = list(certificate["outcomes"])
    labels = ["Null", "Single", "Two (4/3)", "Correlated two", "Cross-loading", "Heterogeneous", "Nonnormal two"]
    y = np.arange(len(scenarios))
    rates = [certificate["outcomes"][s]["latent_fraction"] for s in scenarios]
    counts = [certificate["outcomes"][s]["latent_supported"] for s in scenarios]
    ax.barh(y, rates, color=["#228879" if r else "#a5b4be" for r in rates], height=.55)
    for i, (r, count) in enumerate(zip(rates, counts)):
        ax.text(r + .025, i, f"{count}/200", va="center", fontsize=9)
    ax.set(yticks=y, yticklabels=labels, xlim=(0, 1.3), xticks=[0, .5, 1],
           xticklabels=["0%", "50%", "100%"], xlabel="Candidate latent-structure acceptance rate")
    ax.invert_yaxis()
    ax.set_title("D. Simulation calibration (synthetic data)\n200 datasets/scenario | bootstrap B = 200", pad=12)
    ax.grid(axis="x", alpha=.16)


drawers = [("01_correlation", correlation), ("02_parallel_analysis", parallel),
           ("03_paired_effects", paired), ("04_simulation", simulation)]
for name, draw in drawers:
    fig, ax = plt.subplots(figsize=(8, 6))
    draw(ax)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=220)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
for ax, (_, draw) in zip(axes.flat, drawers):
    draw(ax)
fig.suptitle("Existing v6 evidence | DROZY proxy indicators | real-data decision: B (not supported)", fontsize=16)
fig.tight_layout(rect=(0, 0, 1, .96), h_pad=3, w_pad=3)
fig.savefig(OUT / "overview.png", dpi=140)
plt.close(fig)
(OUT / "sources.json").write_text(json.dumps({"run_id": facts["run_id"],
    "source_sha256": SOURCES, "note": "Visualization only. No training, simulation or raw feature extraction rerun."},
    indent=2) + "\n")
print(OUT)
