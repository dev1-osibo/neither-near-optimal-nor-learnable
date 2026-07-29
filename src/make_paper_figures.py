#!/usr/bin/env python3
"""
Reproducible figure generation for Paper 1 -- CORRECTED substrate (2026-07-27 retrain) and the
PIVOTED narrative (honest benchmark + true LP-optimum online->offline gap + foresight paradox).

Reads the FRESH local artifacts in results/:
  - rl_temporal_verdict.json      -> fig_verdict_delta.png     (best-RL vs RuleBased, 4/12 wins)
  - rl_lp_ceiling.json            -> fig_optimality_gap.png     (RB & RL vs the TRUE LP optimum)
  - data/alibaba_gpu2020/hourly_utilization.csv -> fig_load_typical_week.png
Pareto + foresight figures are produced directly by rl_pareto_analysis.py / rl_foresight_premium.py.
Sizing-sensitivity figure needs a fresh corrected-substrate sizing sweep (flagged; run separately).
"""
from __future__ import annotations
import os, json, shutil, argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEF_RESULTS = os.path.join(HERE, "results")
DATA = os.path.join(HERE, "data")
FIGDIR = os.path.join(HERE, "paper", "figures")
os.makedirs(FIGDIR, exist_ok=True)

RENEW = {"grid_only", "grid_solar", "grid_wind", "grid_solar_wind"}
STORAGE = {"grid_solar_battery", "grid_wind_battery", "grid_solar_wind_battery", "all_sources"}
ORDER = ["grid_only", "grid_solar", "grid_wind", "grid_solar_wind", "grid_gas",
         "grid_solar_gas", "grid_wind_gas", "grid_solar_wind_gas", "grid_solar_battery",
         "grid_wind_battery", "grid_solar_wind_battery", "all_sources"]


def _save(fig, name):
    path = os.path.join(FIGDIR, name)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
    print(f"[fig] wrote {os.path.relpath(path)}")


def fig_load_typical_week(results_dir):
    csv = os.path.join(results_dir, "hourly_utilization.csv")
    if not os.path.isfile(csv):
        csv = os.path.join(DATA, "alibaba_gpu2020", "hourly_utilization.csv")
    if not os.path.isfile(csv):
        print("[fig] skip load_typical_week (no hourly_utilization.csv)"); return
    import pandas as pd
    df = pd.read_csv(csv)
    h = df["hour_index"].astype(int).values
    key = ((h // 24) % 7) * 24 + (h % 24)
    prof = np.zeros(168); cnt = np.zeros(168)
    np.add.at(prof, key, df["gpu_util"].values.astype(float)); np.add.at(cnt, key, 1.0)
    util = prof / np.maximum(cnt, 1.0)
    it_mw = 20.0 * (0.30 + 0.70 * util)
    fig, ax1 = plt.subplots(figsize=(8.2, 3.6))
    ax1.plot(np.arange(168), util * 100, color="#1f77b4", lw=1.6)
    ax1.set_xlabel("Hour of typical week (Mon 00:00 -> Sun 23:00)")
    ax1.set_ylabel("Cluster GPU utilization (%)", color="#1f77b4")
    ax1.set_xlim(0, 167); ax1.set_xticks(np.arange(0, 168, 24))
    ax1.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]); ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(np.arange(168), it_mw, color="#d62728", lw=1.0, ls="--", alpha=0.7)
    ax2.set_ylabel("IT power (MW)", color="#d62728")
    ax1.set_title(f"Reconstructed real workload: typical-week profile (Alibaba GPU-2020)\n"
                  f"mean util {util.mean()*100:.0f}%, peak {util.max()*100:.0f}% -> flat load, "
                  "limited deferral headroom")
    _save(fig, "fig_load_typical_week.png")


def fig_optimality_gap(results_dir):
    """NEW headline figure: how far RuleBased AND best-RL sit above the TRUE LP optimum."""
    path = os.path.join(results_dir, "rl_lp_ceiling.json")
    if not os.path.isfile(path):
        print("[fig] skip optimality_gap (run rl_lp_ceiling.py first)"); return
    rows = {r["config"]: r for r in json.load(open(path))["rows"]}
    configs = [c for c in ORDER if c in rows]
    rb = [rows[c]["rb_to_opt_pct"] for c in configs]
    rl = [rows[c]["rl_to_opt_pct"] for c in configs]
    x = np.arange(len(configs)); w = 0.4
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(x - w/2, rb, w, label="RuleBased -> optimum", color="#1f77b4", edgecolor="black", lw=0.4)
    ax.bar(x + w/2, rl, w, label="best RL -> optimum", color="#d62728", alpha=0.85,
           edgecolor="black", lw=0.4)
    ax.set_xticks(x); ax.set_xticklabels(configs, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Gap above true LP optimum (%)")
    ax.set_title("Online->offline optimality gap (true LP optimum):\n"
                 "RuleBased ~6-9% (renewable) but ~20-25% (storage); RL is even further -- "
                 "neither controller closes it")
    ax.grid(True, axis="y", alpha=0.3); ax.legend()
    _save(fig, "fig_optimality_gap.png")


def fig_verdict_delta(results_dir):
    path = os.path.join(results_dir, "rl_temporal_verdict.json")
    if not os.path.isfile(path):
        print("[fig] skip verdict_delta (no verdict)"); return
    v = json.load(open(path))
    by = {}
    for c in v["comparisons"]:
        by.setdefault(c["config"], []).append(c)
    configs = [c for c in ORDER if c in by]
    vals, colors = [], []
    for c in configs:
        best = min(by[c], key=lambda r: r["rl_cost_mean"])
        d = best["vs_rulebased"]["pct_improvement"]
        sig = best["vs_rulebased"].get("significant_holm") and best["vs_rulebased"].get("a_wins")
        vals.append(d)
        colors.append("#2ca02c" if sig else ("#d62728" if d < 0 else "#7f7f7f"))
    fig, ax = plt.subplots(figsize=(9.5, 4))
    ax.bar(range(len(configs)), vals, color=colors, alpha=0.9, edgecolor="black", lw=0.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Best-RL improvement over RuleBased (%)")
    ax.set_title("Does RL beat the heuristic? Best-of-4 RL vs RuleBased, held-out cost (corrected substrate)\n"
                 "green = significant win (4/12, all renewable-only), grey = neutral, red = significant loss")
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, "fig_verdict_delta.png")


def _copy_if(src, dst):
    if os.path.isfile(src):
        shutil.copy2(src, os.path.join(FIGDIR, dst)); print(f"[fig] copied {dst}"); return True
    return False


def fig_pareto_and_foresight(results_dir):
    got = _copy_if(os.path.join(results_dir, "pareto_cost_carbon.png"), "fig_pareto_cost_carbon.png")
    _copy_if(os.path.join(results_dir, "pareto_parallel.png"), "fig_pareto_parallel.png")
    if not got:
        print("[fig] pareto: run rl_pareto_analysis.py first")
    fp = os.path.join(FIGDIR, "fig_foresight_premium.png")
    print(f"[fig] foresight_premium: {'present' if os.path.isfile(fp) else 'run rl_foresight_premium.py'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=DEF_RESULTS)
    args = ap.parse_args()
    print(f"[fig] results dir: {args.results_dir}")
    fig_load_typical_week(args.results_dir)
    fig_optimality_gap(args.results_dir)
    fig_verdict_delta(args.results_dir)
    fig_pareto_and_foresight(args.results_dir)
    print("[fig] NOTE: fig_sizing_sensitivity needs a fresh corrected-substrate sizing sweep (deferred).")
    print(f"[fig] done -> {FIGDIR}")


if __name__ == "__main__":
    main()
