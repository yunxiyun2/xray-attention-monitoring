"""Generate SI-MSET model architecture diagram."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.facecolor": "white"
})

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis("off")

# Colors
COLORS = {
    "input": "#246591",
    "encoder": "#228879",
    "transformer": "#bd7430",
    "output": "#8b5a9e",
    "loss": "#d9534f",
}

# Title
ax.text(6, 7.7, "SI-MSET Model Architecture", ha="center", fontsize=14, fontweight="bold")

# Input branches
branch_width = 1.2
branch_height = 0.8
y_positions = [5.5, 4.5, 3.5]
branch_labels = ["EOG-H", "EOG-V", "Velocity"]

for i, (y, label) in enumerate(zip(y_positions, branch_labels)):
    # Input box
    rect = Rectangle((0.5, y), branch_width, branch_height,
                     facecolor=COLORS["input"], alpha=0.3, edgecolor=COLORS["input"], linewidth=2)
    ax.add_patch(rect)
    ax.text(1.1, y + 0.4, label, ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(1.1, y + 0.2, "Input", ha="center", va="center", fontsize=9)

# Multi-scale encoding
enc_rect = Rectangle((2.5, 3.5), 2.5, 2.8,
                     facecolor=COLORS["encoder"], alpha=0.3, edgecolor=COLORS["encoder"], linewidth=2)
ax.add_patch(enc_rect)
ax.text(3.75, 5.0, "Multi-Scale\nTemporal Encoding", ha="center", va="center",
        fontsize=11, fontweight="bold")
ax.text(3.75, 4.4, "3 scales", ha="center", va="center", fontsize=9)

# Arrows from branches to encoder
for y in y_positions:
    arrow = FancyArrowPatch((1.8, y + 0.4), (2.4, y + 0.4),
                            arrowstyle="->,head_width=0.3,head_length=0.3",
                            color=COLORS["input"], linewidth=2)
    ax.add_patch(arrow)

# Transformer
trans_rect = Rectangle((5.5, 3.5), 3, 2.8,
                       facecolor=COLORS["transformer"], alpha=0.3, edgecolor=COLORS["transformer"], linewidth=2)
ax.add_patch(trans_rect)
ax.text(7, 5.2, "Transformer Encoder", ha="center", va="center",
        fontsize=11, fontweight="bold")
ax.text(7, 4.7, "3 layers, 96 dim, 4 heads", ha="center", va="center", fontsize=9)
ax.text(7, 4.2, "+ Adversarial Constraint", ha="center", va="center", fontsize=9)

# Arrow from encoder to transformer
arrow = FancyArrowPatch((5.0, 4.9), (5.4, 4.9),
                        arrowstyle="->,head_width=0.3,head_length=0.3",
                        color=COLORS["encoder"], linewidth=2)
ax.add_patch(arrow)

# Output head
out_rect = Rectangle((9, 4.5), 2, 1.2,
                     facecolor=COLORS["output"], alpha=0.3, edgecolor=COLORS["output"], linewidth=2)
ax.add_patch(out_rect)
ax.text(10, 5.1, "Classification", ha="center", va="center",
        fontsize=11, fontweight="bold")
ax.text(10, 4.7, "Head", ha="center", va="center", fontsize=9)

# Arrow from transformer to output
arrow = FancyArrowPatch((8.6, 4.9), (8.9, 5.1),
                        arrowstyle="->,head_width=0.3,head_length=0.3",
                        color=COLORS["transformer"], linewidth=2)
ax.add_patch(arrow)

# Loss box
loss_rect = Rectangle((9, 2.8), 2, 1.2,
                      facecolor=COLORS["loss"], alpha=0.3, edgecolor=COLORS["loss"], linewidth=2)
ax.add_patch(loss_rect)
ax.text(10, 3.4, "Pairwise", ha="center", va="center",
        fontsize=11, fontweight="bold")
ax.text(10, 3.0, "Ranking Loss", ha="center", va="center", fontsize=9)

# Arrow from output to loss
arrow = FancyArrowPatch((10, 4.4), (10, 4.0),
                        arrowstyle="->,head_width=0.3,head_length=0.3",
                        color=COLORS["output"], linewidth=2)
ax.add_patch(arrow)

# Legend
legend_y = 1.5
legend_items = [
    (COLORS["input"], "Input Branches (3)"),
    (COLORS["encoder"], "Multi-Scale Encoding"),
    (COLORS["transformer"], "Transformer Encoder"),
    (COLORS["output"], "Classification Head"),
    (COLORS["loss"], "Pairwise Ranking Loss"),
]

for i, (color, label) in enumerate(legend_items):
    x = 0.5 + (i % 3) * 3.5
    y = legend_y - 0.4 * (i // 3)
    rect = Rectangle((x, y - 0.15), 0.5, 0.3, facecolor=color, alpha=0.5, edgecolor=color)
    ax.add_patch(rect)
    ax.text(x + 0.6, y, label, fontsize=10, va="center")

# Key statistics
stats = """
Key Specifications:
• Input: Raw EOG-H, EOG-V, Velocity (3 branches)
• Encoding: Multi-scale temporal convolution (3 scales)
• Transformer: 3 layers, 96 dimensions, 4 attention heads
• Regularization: Subject adversarial constraint
• Loss: Pairwise ranking (alert vs sleepy)
• Training: 3 seeds per LOSO fold, max 80 epochs
"""

ax.text(0.5, 0.5, stats, fontsize=10, va="top",
        bbox=dict(boxstyle="round", facecolor="#f8f8f8", edgecolor="#ddd"))

# Output info
output_info = """
Output: Fatigue state probability (per window)
Evaluation: 10-fold LOSO, subject-level AUC
"""
ax.text(6.5, 0.5, output_info, fontsize=10, va="top", ha="center",
        bbox=dict(boxstyle="round", facecolor="#e8f4e8", edgecolor=COLORS["output"]))

fig.tight_layout()
out_dir = Path(__file__).resolve().parent / "figures"
fig.savefig(out_dir / "model_architecture.png", dpi=220)
fig.savefig(out_dir / "model_architecture.pdf")
plt.close(fig)

print(f"Model architecture figure saved to {out_dir}")
