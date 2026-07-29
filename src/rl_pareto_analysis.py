#!/usr/bin/env python3
"""
Pareto-front analysis of the objective-weight sweep (Paper 1, AWS-7 / D13).

WHAT THIS IS
------------
The Pareto sweep trained PPO on the single `all_sources` configuration under four
objective weightings (cost / carbon / water / SLA):

    cost-heavy    0.7 / 0.1 / 0.1 / 0.1
    carbon-heavy  0.1 / 0.7 / 0.1 / 0.1
    water-heavy   0.1 / 0.1 / 0.7 / 0.1
    equal         0.3 / 0.3 / 0.3 / 0.1

Each weighting has 5 seeds, evaluated on the held-out test window (2024-25),
n_eval_episodes = 200 per seed, on the real-trace substrate.

INTEGRITY GUARDS (DECISION_LOG ACTION-ON-COMPLETION):
  * This analysis is SEPARATE from the 4-algorithm headline verdict. The Pareto
    files are NEVER pooled into that comparison — different objective weights make
    the scalar reward non-comparable across points.
  * We therefore rank/compare ONLY the raw physical objectives (cost $, carbon kg,
    water m^3, SLA violation count) across weightings — never `mean_reward`, which
    is a weighted scalar that differs by construction between weightings.

WHAT IT REPORTS
---------------
  1. Per-weighting aggregate (mean +/- std, SEM, 95% t-CI over the 5 seeds) for
     cost, carbon, water, SLA.
  2. "Does the knob work?" diagonal check: does each heavy weighting actually
     minimise its own objective relative to the others?
  3. Directional tradeoff vs the balanced `equal` point (% change per objective).
  4. Pareto dominance among the four points in (cost, carbon, water) space.
  5. Welch t-tests (heavy vs equal) per objective, with an explicit low-power flag
     (n = 5 seeds).
  6. Figures: cost-vs-carbon tradeoff scatter (water encoded) + normalised
     parallel-coordinates. Figure generation is guarded; analysis still completes
     if matplotlib is unavailable.

Outputs: <out_dir>/pareto_analysis.json, <out_dir>/PARETO_ANALYSIS.md,
         <fig_dir>/pareto_cost_carbon.png, <fig_dir>/pareto_parallel.png
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from typing import Dict, List

try:
    import numpy as np
except ImportError:  # numpy is a hard dependency for the stats here
    print("CRITICAL HALT: numpy required for this analysis.", file=sys.stderr)
    raise

try:
    from scipy import stats as _scipy_stats
    _HAVE_SCIPY = True
except ImportError:
    _scipy_stats = None
    _HAVE_SCIPY = False

# Weight points: (alpha_cost, alpha_carbon, alpha_water, alpha_sla)
WEIGHTS = {
    "cost":   (0.7, 0.1, 0.1, 0.1),
    "carbon": (0.1, 0.7, 0.1, 0.1),
    "water":  (0.1, 0.1, 0.7, 0.1),
    "equal":  (0.3, 0.3, 0.3, 0.1),
}

# Physical objectives we are allowed to compare across weightings.
# Each is minimised (lower = better).
OBJ_KEYS = {
    "cost":   ("mean_episode_cost",   "USD/week"),
    "carbon": ("mean_episode_carbon", "kg CO2/week"),
    "water":  ("mean_episode_water",  "m3/week"),
    "sla":    ("mean_sla_violations", "violations/week"),
}

BALANCED_REF = "equal"


def _t_crit(df: int, conf: float = 0.95) -> float:
    """Two-sided critical t value. Uses scipy when present, else a small table."""
    if _HAVE_SCIPY:
        return float(_scipy_stats.t.ppf(1 - (1 - conf) / 2.0, df))
    table_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
                6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
    return table_95.get(df, 1.96)


def load_sweep(results_dir: str) -> Dict[str, List[dict]]:
    """Load the four Pareto files; validate seed count and homogeneity."""
    data: Dict[str, List[dict]] = {}
    for tag in WEIGHTS:
        # Seeds are sharded across many workers (20-seed sweep => workers 0..19), so glob
        # ALL worker shards for this tag and concatenate, rather than reading worker_0 only.
        paths = sorted(glob.glob(os.path.join(
            results_dir, f"rl_results_temporal_pareto_{tag}_worker_*.json")))
        if not paths:
            raise FileNotFoundError(f"Missing Pareto files for '{tag}' in {results_dir}")
        recs: List[dict] = []
        for path in paths:
            with open(path) as fh:
                recs.extend(json.load(fh))
        if not recs:
            raise ValueError(f"'{tag}' files contained no records")
        algos = {r.get("algorithm") for r in recs}
        configs = {r.get("config") for r in recs}
        windows = {r.get("eval_window") for r in recs}
        if algos != {"PPO"}:
            raise ValueError(f"'{tag}' unexpected algorithms {algos} (expected PPO only)")
        if configs != {"all_sources"}:
            raise ValueError(f"'{tag}' unexpected configs {configs} (expected all_sources)")
        if windows != {"test"}:
            raise ValueError(f"'{tag}' eval_window not held-out test: {windows}")
        if len(recs) != 20:
            print(f"WARNING: '{tag}' has {len(recs)} seeds (expected 20).", file=sys.stderr)
        data[tag] = recs
    return data


def aggregate(data: Dict[str, List[dict]]) -> dict:
    """Per-weighting mean/std/SEM/95%CI over seeds for each physical objective."""
    agg: dict = {}
    for tag, recs in data.items():
        seeds = sorted(r["seed"] for r in recs)
        n = len(recs)
        tcrit = _t_crit(n - 1)
        per_obj = {}
        for obj, (key, unit) in OBJ_KEYS.items():
            vals = np.array([float(r[key]) for r in recs], dtype=float)
            mean = float(vals.mean())
            std = float(vals.std(ddof=1)) if n > 1 else 0.0
            sem = std / math.sqrt(n) if n > 1 else 0.0
            ci = tcrit * sem
            per_obj[obj] = {
                "unit": unit,
                "mean": mean,
                "std": std,
                "sem": sem,
                "ci95_halfwidth": ci,
                "ci95_low": mean - ci,
                "ci95_high": mean + ci,
                "per_seed": vals.tolist(),
            }
        agg[tag] = {
            "weights_cost_carbon_water_sla": list(WEIGHTS[tag]),
            "n_seeds": n,
            "seeds": seeds,
            "objectives": per_obj,
        }
    return agg


def diagonal_check(agg: dict) -> dict:
    """Does each heavy weighting actually minimise its own objective?"""
    out = {}
    for obj in ("cost", "carbon", "water"):
        means = {tag: agg[tag]["objectives"][obj]["mean"] for tag in agg}
        winner = min(means, key=means.get)
        expected = obj  # the weighting whose name matches the objective
        out[obj] = {
            "expected_minimiser": expected,
            "actual_minimiser": winner,
            "knob_behaves_as_designed": (winner == expected),
            "means_by_weighting": means,
        }
    return out


def tradeoff_vs_balanced(agg: dict, ref: str = BALANCED_REF) -> dict:
    """% change in each objective for each weighting relative to the balanced point."""
    out = {}
    ref_means = {obj: agg[ref]["objectives"][obj]["mean"] for obj in OBJ_KEYS}
    for tag in agg:
        if tag == ref:
            continue
        row = {}
        for obj in OBJ_KEYS:
            m = agg[tag]["objectives"][obj]["mean"]
            base = ref_means[obj]
            pct = ((m - base) / base * 100.0) if base != 0 else float("nan")
            row[obj] = {"value": m, "ref_value": base, "pct_change_vs_equal": pct}
        out[tag] = row
    return out


def pareto_dominance(agg: dict) -> dict:
    """Non-dominated set in (cost, carbon, water) minimisation space (mean values)."""
    objs = ("cost", "carbon", "water")
    pts = {tag: np.array([agg[tag]["objectives"][o]["mean"] for o in objs]) for tag in agg}
    dominated = {}
    for a in pts:
        dom_by = []
        for b in pts:
            if a == b:
                continue
            # b dominates a if b <= a on all objectives and < on at least one
            if np.all(pts[b] <= pts[a]) and np.any(pts[b] < pts[a]):
                dom_by.append(b)
        dominated[a] = dom_by
    non_dominated = [t for t, d in dominated.items() if not d]
    return {
        "objective_space": list(objs),
        "dominated_by": dominated,
        "non_dominated": non_dominated,
        "all_non_dominated": len(non_dominated) == len(pts),
    }


def significance(data: Dict[str, List[dict]], agg: dict, ref: str = BALANCED_REF) -> dict:
    """Welch t-test (heavy vs equal) per objective on seed-level means. Low power (n=5)."""
    out = {"note": "Welch two-sample t-test on per-seed means; n=20 seeds per group.",
           "have_scipy": _HAVE_SCIPY}
    ref_vals = {obj: np.array(agg[ref]["objectives"][obj]["per_seed"]) for obj in OBJ_KEYS}
    for tag in agg:
        if tag == ref:
            continue
        row = {}
        for obj in OBJ_KEYS:
            x = np.array(agg[tag]["objectives"][obj]["per_seed"])
            y = ref_vals[obj]
            if _HAVE_SCIPY and x.std() + y.std() > 0:
                t, p = _scipy_stats.ttest_ind(x, y, equal_var=False)
                row[obj] = {"t": float(t), "p": float(p), "significant_0.05": bool(p < 0.05)}
            else:
                row[obj] = {"t": None, "p": None, "significant_0.05": None}
        out[tag] = row
    return out


def make_figures(agg: dict, fig_dir: str) -> List[str]:
    """Cost-vs-carbon tradeoff scatter + normalised parallel coordinates."""
    written: List[str] = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("NOTE: matplotlib unavailable; skipping figures (analysis still complete).",
              file=sys.stderr)
        return written

    os.makedirs(fig_dir, exist_ok=True)
    tags = list(agg.keys())
    colors = {"cost": "#1f77b4", "carbon": "#2ca02c", "water": "#17becf", "equal": "#7f7f7f"}

    # --- Figure 1: cost vs carbon, marker size ~ water, SLA annotated ---
    cost = np.array([agg[t]["objectives"]["cost"]["mean"] for t in tags])
    cost_ci = np.array([agg[t]["objectives"]["cost"]["ci95_halfwidth"] for t in tags])
    carbon = np.array([agg[t]["objectives"]["carbon"]["mean"] for t in tags])
    carbon_ci = np.array([agg[t]["objectives"]["carbon"]["ci95_halfwidth"] for t in tags])
    water = np.array([agg[t]["objectives"]["water"]["mean"] for t in tags])
    sla = np.array([agg[t]["objectives"]["sla"]["mean"] for t in tags])

    w_min, w_max = water.min(), water.max()
    sizes = 200 + (600 * (water - w_min) / (w_max - w_min) if w_max > w_min else np.zeros_like(water))

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for i, t in enumerate(tags):
        ax.errorbar(cost[i], carbon[i], xerr=cost_ci[i], yerr=carbon_ci[i],
                    fmt="none", ecolor="#999999", elinewidth=1, capsize=3, zorder=1)
        ax.scatter(cost[i], carbon[i], s=sizes[i], color=colors.get(t, "#444"),
                   alpha=0.75, edgecolors="black", linewidths=1, zorder=2,
                   label=f"{t}-weighted")
        ax.annotate(f"{t}\nSLA={sla[i]:.2f}", (cost[i], carbon[i]),
                    textcoords="offset points", xytext=(8, 8), fontsize=8)
    ax.set_xlabel("Weekly cost (USD)  -->  lower is better")
    ax.set_ylabel("Weekly carbon (kg CO2)  -->  lower is better")
    ax.set_title("Cost-Carbon tradeoff by objective weighting\n(PPO, all_sources, held-out "
                 "test; marker size ~ water; bars = 95% CI over 5 seeds)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    f1 = os.path.join(fig_dir, "pareto_cost_carbon.png")
    fig.savefig(f1, dpi=160)
    plt.close(fig)
    written.append(f1)

    # --- Figure 2: normalised parallel coordinates over cost/carbon/water/SLA ---
    obj_order = ["cost", "carbon", "water", "sla"]
    raw = {o: np.array([agg[t]["objectives"][o]["mean"] for t in tags]) for o in obj_order}
    norm = {}
    for o in obj_order:
        v = raw[o]
        rng = v.max() - v.min()
        norm[o] = (v - v.min()) / rng if rng > 0 else np.zeros_like(v)

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    xs = np.arange(len(obj_order))
    for i, t in enumerate(tags):
        ys = [norm[o][i] for o in obj_order]
        ax.plot(xs, ys, marker="o", color=colors.get(t, "#444"),
                linewidth=2, alpha=0.8, label=f"{t}-weighted")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{o}\n({OBJ_KEYS[o][1]})" for o in obj_order], fontsize=8)
    ax.set_ylabel("Normalised objective (0 = best across points, 1 = worst)")
    ax.set_title("Objective-weight tradeoff (normalised parallel coordinates)\n"
                 "PPO, all_sources, held-out test, 5-seed means")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    f2 = os.path.join(fig_dir, "pareto_parallel.png")
    fig.savefig(f2, dpi=160)
    plt.close(fig)
    written.append(f2)
    return written


def _fmt(x: float, nd: int = 1) -> str:
    return f"{x:,.{nd}f}"


def write_markdown(agg, diag, trade, dom, sig, figs, out_md: str) -> None:
    L = []
    L.append("# Pareto-Front Analysis — Objective-Weight Sweep (Paper 1)\n")
    L.append("**Scope:** PPO, `all_sources` configuration, held-out test window (2024-25), "
             "real-trace (corrected, carbon-visible) substrate, n_eval_episodes=200 per seed, "
             "20 seeds per weighting.\n")
    L.append("**Integrity:** analysed SEPARATELY from the 4-algorithm headline verdict; "
             "raw physical objectives only (never the weighted scalar reward).\n")

    # 1. Aggregate table
    L.append("\n## 1. Aggregate objectives by weighting (mean +/- 95% CI over 20 seeds)\n")
    L.append("| Weighting (cost/carb/water/SLA) | Cost (USD/wk) | Carbon (kg/wk) | Water (m3/wk) | SLA (viol/wk) |")
    L.append("|---|---|---|---|---|")
    for tag in agg:
        w = "/".join(str(x) for x in agg[tag]["weights_cost_carbon_water_sla"])
        o = agg[tag]["objectives"]
        L.append(
            f"| **{tag}** ({w}) | {_fmt(o['cost']['mean'])} ± {_fmt(o['cost']['ci95_halfwidth'])} "
            f"| {_fmt(o['carbon']['mean'])} ± {_fmt(o['carbon']['ci95_halfwidth'])} "
            f"| {_fmt(o['water']['mean'],2)} ± {_fmt(o['water']['ci95_halfwidth'],2)} "
            f"| {_fmt(o['sla']['mean'],3)} ± {_fmt(o['sla']['ci95_halfwidth'],3)} |"
        )

    # 2. Diagonal check
    L.append("\n## 2. Does the weighting knob work? (own-objective minimisation)\n")
    L.append("| Objective | Expected minimiser | Actual minimiser | Behaves as designed? |")
    L.append("|---|---|---|---|")
    for obj, d in diag.items():
        L.append(f"| {obj} | {d['expected_minimiser']}-heavy | {d['actual_minimiser']}-heavy "
                 f"| {'YES' if d['knob_behaves_as_designed'] else 'NO'} |")

    # 3. Tradeoff vs equal
    L.append("\n## 3. Directional tradeoff vs the balanced `equal` point (% change)\n")
    L.append("Negative = better than equal (lower cost/carbon/water/SLA); positive = worse.\n")
    L.append("| Weighting | Cost % | Carbon % | Water % | SLA (abs delta) |")
    L.append("|---|---|---|---|---|")
    for tag, row in trade.items():
        # SLA reference (equal) is 0 -> % undefined; report absolute delta instead.
        sla_delta = row['sla']['value'] - row['sla']['ref_value']
        sla_cell = (f"{row['sla']['pct_change_vs_equal']:+.2f}%"
                    if math.isfinite(row['sla']['pct_change_vs_equal'])
                    else f"{sla_delta:+.3f} (ref=0)")
        L.append(f"| {tag}-heavy | {row['cost']['pct_change_vs_equal']:+.2f} "
                 f"| {row['carbon']['pct_change_vs_equal']:+.2f} "
                 f"| {row['water']['pct_change_vs_equal']:+.2f} "
                 f"| {sla_cell} |")

    # 4. Pareto dominance
    L.append("\n## 4. Pareto dominance in (cost, carbon, water) space\n")
    L.append(f"- Non-dominated points: **{', '.join(dom['non_dominated'])}**")
    L.append(f"- All four points mutually non-dominated? **{dom['all_non_dominated']}**")
    for tag, by in dom["dominated_by"].items():
        if by:
            L.append(f"  - `{tag}` is dominated by: {', '.join(by)}")

    # 5. Significance
    L.append("\n## 5. Welch t-tests (heavy vs equal), per objective\n")
    L.append(f"_{sig['note']}_\n")
    L.append("| Weighting | Cost p | Carbon p | Water p | SLA p |")
    L.append("|---|---|---|---|---|")
    for tag, row in sig.items():
        if tag in ("note", "have_scipy"):
            continue
        def cell(o):
            r = row[o]
            if r["p"] is None:
                return "n/a"
            star = "*" if r["significant_0.05"] else ""
            return f"{r['p']:.3f}{star}"
        L.append(f"| {tag}-heavy | {cell('cost')} | {cell('carbon')} | {cell('water')} | {cell('sla')} |")
    L.append("\n(* significant at p<0.05; n=20 seeds per group.)")

    if figs:
        L.append("\n## 6. Figures\n")
        for f in figs:
            L.append(f"- `{os.path.relpath(f)}`")

    L.append("\n---\n_Generated by `src/rl_pareto_analysis.py`. Do not pool with the headline verdict._\n")

    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


def main():
    default_results = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "rl_backup_20260716",
    )
    ap = argparse.ArgumentParser(description="Pareto-front analysis of the objective-weight sweep.")
    ap.add_argument("--results-dir", default=default_results,
                    help="Directory containing rl_results_temporal_pareto_*_worker_0.json")
    ap.add_argument("--out-dir", default=None,
                    help="Where to write pareto_analysis.json / PARETO_ANALYSIS.md "
                         "(default: <results-dir>/pareto_analysis)")
    ap.add_argument("--fig-dir", default=None,
                    help="Where to write figures (default: <out-dir>)")
    ap.add_argument("--no-figures", action="store_true", help="Skip figure generation")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.results_dir, "pareto_analysis")
    fig_dir = args.fig_dir or out_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f"[pareto] loading from: {args.results_dir}")
    data = load_sweep(args.results_dir)
    agg = aggregate(data)
    diag = diagonal_check(agg)
    trade = tradeoff_vs_balanced(agg)
    dom = pareto_dominance(agg)
    sig = significance(data, agg)

    figs = [] if args.no_figures else make_figures(agg, fig_dir)

    result = {
        "scope": {
            "algorithm": "PPO",
            "config": "all_sources",
            "eval_window": "test (held-out 2024-25)",
            "n_eval_episodes_per_seed": 200,
            "n_seeds": {t: agg[t]["n_seeds"] for t in agg},
            "weights_cost_carbon_water_sla": WEIGHTS,
            "substrate": "real-trace (Alibaba GPU-2020 replay, PUE(T) cooling)",
            "integrity_note": "SEPARATE from the 4-algorithm headline verdict; "
                              "physical objectives only, weighted reward excluded.",
        },
        "aggregate": agg,
        "diagonal_check": diag,
        "tradeoff_vs_equal": trade,
        "pareto_dominance": dom,
        "significance_welch": sig,
        "figures": [os.path.relpath(f) for f in figs],
    }
    out_json = os.path.join(out_dir, "pareto_analysis.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    out_md = os.path.join(out_dir, "PARETO_ANALYSIS.md")
    write_markdown(agg, diag, trade, dom, sig, figs, out_md)

    print(f"[pareto] wrote: {out_json}")
    print(f"[pareto] wrote: {out_md}")
    for f in figs:
        print(f"[pareto] wrote figure: {f}")

    # Console summary (honest headline)
    print("\n=== PARETO SUMMARY ===")
    for tag in agg:
        o = agg[tag]["objectives"]
        print(f"  {tag:7s}: cost={o['cost']['mean']:.0f}  carbon={o['carbon']['mean']:.0f}  "
              f"water={o['water']['mean']:.1f}  SLA={o['sla']['mean']:.3f}")
    print("  diagonal (knob works?):",
          {o: diag[o]["knob_behaves_as_designed"] for o in diag})
    print("  non-dominated:", dom["non_dominated"])


if __name__ == "__main__":
    main()
