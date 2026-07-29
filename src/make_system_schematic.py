#!/usr/bin/env python3
"""Figure 1 — system schematic for Paper 1. Renders the data->substrate->dispatch->objectives
->controllers pipeline plus the leakage-control protocol, to paper/figures/fig1_system_schematic.png.
Pure matplotlib (no data dependency); reproducible."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(HERE, "paper", "figures")
os.makedirs(FIGDIR, exist_ok=True)

fig, ax = plt.subplots(figsize=(9.2, 6.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

def box(x, y, w, h, text, fc, fs=8.5, ec="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.5",
                                linewidth=1.0, edgecolor=ec, facecolor=fc, alpha=0.95))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)

def arrow(x1, y1, x2, y2, style="-|>", color="#444", lw=1.3):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=12,
                                 linewidth=lw, color=color))

BLUE, GREEN, ORANGE, PURPLE, GREY = "#cfe2f3", "#d9ead3", "#fce5cd", "#e6d0f0", "#efefef"

# Row 1: real data inputs
box(3, 86, 20, 10, "Real workload trace\n(Alibaba GPU-2020)", BLUE)
box(27, 86, 18, 10, "ERCOT LMP\nprices", BLUE)
box(49, 86, 18, 10, "EIA fuel-mix\n× IPCC carbon", BLUE)
box(71, 86, 26, 10, "Weather (temp, solar,\nwind) — Open-Meteo / NASA", BLUE)

# Row 2: substrate models
box(6, 66, 26, 11, "IT power\nP_nom·[φ+(1−φ)·u]  (Fan)", GREEN)
box(37, 66, 26, 11, "Cooling\nIT·[PUE(T)−1]", GREEN)
box(68, 66, 29, 11, "On-site supply models\nsolar / wind / battery / gas", GREEN)

# Row 3: env core
box(20, 46, 60, 12, "Single-facility environment (hourly, 168-h episodes)\n"
                    "demand D = IT + cooling;  dispatch order: renewables → battery → gas → grid\n"
                    "actions: defer (≤30%) · cooling offset (±3°C) · battery · gas", ORANGE, fs=8)

# Row 4: objectives
box(30, 30, 40, 9, "Objectives → reward\nR = −Σ α·(cost, carbon, water, SLA)/divisor", ORANGE, fs=8.5)

# Row 5: controllers
box(4, 10, 27, 12, "Learned controllers\nPPO · SAC · TD3 · A2C", PURPLE, fs=8.5)
box(34, 10, 32, 12, "Non-learned baselines\nDoNothing · RuleBased ·\nGreedy · MPC", PURPLE, fs=8.5)
box(69, 10, 28, 12, "Offline oracle\n(perfect hindsight,\nceiling only)", GREY, fs=8.5)

# Protocol side-note
box(2, 30, 24, 9, "Leakage control:\ntrain 2020–23 / test 2024–25;\nno-lookahead (persistence)", "#fff2cc", fs=7.2)

# Arrows: inputs -> substrate
arrow(13, 86, 16, 77.5)          # trace -> IT
arrow(84, 86, 55, 77.5)          # weather -> cooling
arrow(84, 86, 82, 77.5)          # weather -> supply
# substrate -> env
for cx in (19, 50, 82):
    arrow(cx, 66, 50, 58.5)
# price/carbon -> env (cost/carbon accounting)
arrow(36, 86, 45, 58.5)
arrow(58, 86, 55, 58.5)
# env -> objectives
arrow(50, 46, 50, 39.5)
# controllers act on env (bidirectional feel): objectives/env -> controllers evaluated
arrow(50, 30, 17, 22.5)
arrow(50, 30, 50, 22.5)
arrow(50, 30, 83, 22.5)

ax.text(50, 98.5, "Figure 1. Honest single-facility behind-the-meter dispatch benchmark",
        ha="center", va="center", fontsize=11, fontweight="bold")

fig.tight_layout()
out = os.path.join(FIGDIR, "fig1_system_schematic.png")
fig.savefig(out, dpi=170, bbox_inches="tight")
plt.close(fig)
print("wrote", out)
