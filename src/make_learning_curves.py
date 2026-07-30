"""
Learning-curve figure: held-out evaluation reward vs training steps, per algorithm,
aggregated (mean +/- std across the 5 seeds) from the checkpoint_curve logged every
100k steps in the main-run worker files. Demonstrates convergence/plateau by 1M steps.

Reads:  results/rl_results_temporal_worker_*.json
Writes: paper/figures/fig_learning_curves.png
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
FIGDIR = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIGDIR, exist_ok=True)

# Two representative configs: a renewable-only RL win and the hardest storage loss.
PANELS = [("grid_solar_wind", "grid_solar_wind  (RL win, +1.6%)"),
          ("all_sources", "all_sources  (RL loss, -5.0%)")]
ALGOS = ["PPO", "SAC", "TD3", "A2C"]
COLORS = {"PPO": "#1f77b4", "SAC": "#d62728", "TD3": "#2ca02c", "A2C": "#9467bd"}

recs = []
for f in glob.glob(os.path.join(RESULTS, "rl_results_temporal_worker_*.json")):
    recs += json.load(open(f))

def curve_for(algo, config):
    """Return (steps, mean_reward_mean, mean_reward_std) aggregated across seeds."""
    rows = [r for r in recs if r["algorithm"] == algo and r["config"] == config
            and isinstance(r.get("checkpoint_curve"), list) and r["checkpoint_curve"]]
    if not rows:
        return None
    # align on the common step grid
    steps = sorted({e["steps"] for r in rows for e in r["checkpoint_curve"]})
    per_step = {s: [] for s in steps}
    for r in rows:
        for e in r["checkpoint_curve"]:
            per_step[e["steps"]].append(e["mean_reward"])
    mean = np.array([np.mean(per_step[s]) for s in steps])
    std = np.array([np.std(per_step[s]) for s in steps])
    return np.array(steps), mean, std

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
for ax, (cfg, title) in zip(axes, PANELS):
    for algo in ALGOS:
        c = curve_for(algo, cfg)
        if c is None:
            continue
        steps, mean, std = c
        xs = steps / 1e6
        ax.plot(xs, mean, marker="o", ms=3, lw=1.6, color=COLORS[algo], label=algo)
        ax.fill_between(xs, mean - std, mean + std, color=COLORS[algo], alpha=0.12)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("training steps (millions)")
    ax.grid(True, alpha=0.3)
axes[0].set_ylabel("held-out evaluation reward")
axes[0].legend(fontsize=8, loc="lower right")
fig.suptitle("Learning curves: held-out evaluation reward vs training steps "
             "(mean \u00b1 std over 5 seeds)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(FIGDIR, "fig_learning_curves.png")
fig.savefig(out, dpi=150)
print("wrote", out)

# Also print a compact plateau check: last-200k-step reward change per algo/config.
print("\nPlateau check (reward change over final 200k steps, mean across seeds):")
for cfg, _ in PANELS:
    for algo in ALGOS:
        c = curve_for(algo, cfg)
        if c is None:
            continue
        steps, mean, _ = c
        if len(mean) >= 3:
            tail = mean[-1] - mean[-3]
            rel = tail / (abs(mean[-1]) + 1e-9) * 100
            print(f"  {cfg:22s} {algo:4s}  delta(last 200k)={tail:+.3f}  ({rel:+.2f}%)")
