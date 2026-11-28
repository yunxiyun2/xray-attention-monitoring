"""Generate overview flowchart for the X-Ray project practice report (English labels)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.facecolor": "white"
})

COLORS = {
    "data": "#246591",
    "feature": "#228879",
    "structure": "#bd7430",
    "predictive": "#8b5a9e",
    "deep": "#d9534f",
    "success": "#228879",
    "inconclusive": "#bd7430",
    "failure": "#ba5545",
}

def draw_arrow(ax, x1, y1, x2, y2, color="#666", linewidth=1.5):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="->,head_width=0.3,head_length=0.3",
        color=color, linewidth=linewidth, mutation_scale=15
    )
    ax.add_patch(arrow)

fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis("off")

# Title
ax.text(7, 6.6, "X-Ray Project Research Flow", ha="center", fontsize=14, fontweight="bold")

# Data layer
data_box = Rectangle((0.5, 5.2), 3, 0.9, facecolor=COLORS["data"], alpha=0.15,
                     edgecolor=COLORS["data"], linewidth=2)
ax.add_patch(data_box)
ax.text(2, 5.65, "DROZY Dataset", ha="center", fontsize=11, fontweight="bold")
ax.text(2, 5.35, "14 subjects, 36 recs\n2044 windows", ha="center", fontsize=9)

# Feature layer
feat_box = Rectangle((4, 5.2), 3, 0.9, facecolor=COLORS["feature"], alpha=0.15,
                     edgecolor=COLORS["feature"], linewidth=2)
ax.add_patch(feat_box)
ax.text(5.5, 5.65, "Proxy Indicators", ha="center", fontsize=11, fontweight="bold")
ax.text(5.5, 5.35, "FD, PSV, SA, FN\nSE, GTE, SV", ha="center", fontsize=9)

# Structure layer
struct_box = Rectangle((7.5, 5.2), 3, 0.9, facecolor=COLORS["structure"], alpha=0.15,
                       edgecolor=COLORS["structure"], linewidth=2)
ax.add_patch(struct_box)
ax.text(9, 5.65, "Structure Test (v6)", ha="center", fontsize=11, fontweight="bold")
ax.text(9, 5.35, "PAF, Parallel Analysis\nHeywood case", ha="center", fontsize=9)

# Predictive layer
pred_box = Rectangle((11, 5.2), 2.5, 0.9, facecolor=COLORS["predictive"], alpha=0.15,
                     edgecolor=COLORS["predictive"], linewidth=2)
ax.add_patch(pred_box)
ax.text(12.25, 5.65, "Predictive Validity (v7)", ha="center", fontsize=11, fontweight="bold")
ax.text(12.25, 5.35, "LOSO, AUC = 0.670\nΔAUC = -0.150", ha="center", fontsize=9)

# Arrows top row
draw_arrow(ax, 3.6, 5.65, 3.9, 5.65)
draw_arrow(ax, 7.1, 5.65, 7.4, 5.65)
draw_arrow(ax, 10.6, 5.65, 10.9, 5.65)

# Deep learning branch
deep_box = Rectangle((4, 3.5), 9.5, 1.2, facecolor=COLORS["deep"], alpha=0.15,
                     edgecolor=COLORS["deep"], linewidth=2)
ax.add_patch(deep_box)
ax.text(8.75, 4.15, "SI-MSET Deep Model (v1-v1.2)", ha="center", fontsize=11, fontweight="bold")
ax.text(8.75, 3.85, "Raw EOG, Transformer, Adversarial, 3-seed ensemble, Subject calibration",
        ha="center", fontsize=9)

# v6 result
v6_box = Rectangle((7.5, 2.5), 3, 0.7, facecolor=COLORS["inconclusive"], alpha=0.3,
                   edgecolor=COLORS["inconclusive"], linewidth=1.5)
ax.add_patch(v6_box)
ax.text(9, 2.85, "v6: not_evaluable", ha="center", fontsize=10, fontweight="bold",
        color=COLORS["inconclusive"])

# v7 result
v7_box = Rectangle((11, 2.5), 2.5, 0.7, facecolor=COLORS["inconclusive"], alpha=0.3,
                   edgecolor=COLORS["inconclusive"], linewidth=1.5)
ax.add_patch(v7_box)
ax.text(12.25, 2.85, "v7: inconclusive", ha="center", fontsize=10, fontweight="bold",
        color=COLORS["inconclusive"])

# Downward arrows
draw_arrow(ax, 9, 5.15, 9, 3.3, color=COLORS["inconclusive"], linewidth=2)
draw_arrow(ax, 12.25, 5.15, 12.25, 3.3, color=COLORS["inconclusive"], linewidth=2)

# v1.2 result
v12_box = Rectangle((4, 1.8), 9.5, 1.2, facecolor=COLORS["success"], alpha=0.2,
                    edgecolor=COLORS["success"], linewidth=2)
ax.add_patch(v12_box)
ax.text(8.75, 2.45, "v1.2: SUPPORTED", ha="center", fontsize=13, fontweight="bold",
        color=COLORS["success"])
ax.text(8.75, 2.1, "AUC = 0.880 [0.61, 1.00] | ΔAUC vs v7 = +0.21 [0.03, 0.42]",
        ha="center", fontsize=9)

# Arrow to v1.2
draw_arrow(ax, 8.75, 3.5, 8.75, 3.05, color=COLORS["success"], linewidth=2)

# Key statistics
stats = """Key Statistics:
• 14 subjects, 36 recordings, 2044 valid windows
• 10 paired subjects (alert + sleepy)
• 10-fold LOSO, 3 seeds/fold (30 trainings)
• Bootstrap B = 2000"""

ax.text(0.5, 0.4, stats, fontsize=9, va="top",
        bbox=dict(boxstyle="round", facecolor="#f8f8f8", edgecolor="#ddd"))

# Conclusions
conclusions = """Conclusions:
• Proxy indicators: Reproducible null result (structure not supported)
• Deep model: Conditionally supported (requires subject calibration, small-sample LOSO)"""

ax.text(7, 0.4, conclusions, fontsize=9, va="top", ha="center", fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="#e8f4e8", edgecolor=COLORS["success"]))

# Legend
legend_y = 1.1
legend_items = [
    (COLORS["data"], "Data"),
    (COLORS["feature"], "Features"),
    (COLORS["structure"], "Structure"),
    (COLORS["predictive"], "Predictive"),
    (COLORS["deep"], "Deep Learning"),
    (COLORS["success"], "supported"),
    (COLORS["inconclusive"], "inconclusive"),
]

x_start = 0.5
for i, (color, label) in enumerate(legend_items):
    x = x_start + (i % 7) * 1.85
    y = legend_y - 0.35 * (i // 7)
    rect = Rectangle((x, y - 0.12), 0.3, 0.22, facecolor=color, alpha=0.5, edgecolor=color)
    ax.add_patch(rect)
    ax.text(x + 0.35, y, label, fontsize=8, va="center")

fig.tight_layout()
out_dir = Path(__file__).resolve().parent / "figures"
out_dir.mkdir(exist_ok=True)
fig.savefig(out_dir / "project_overview.png", dpi=220)
fig.savefig(out_dir / "project_overview.pdf")
plt.close(fig)

print(f"Overview figure saved to {out_dir}")
