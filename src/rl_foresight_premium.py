#!/usr/bin/env python3
"""
Foresight-premium ablation (Paper 1 §5.6) -- audit #6 corrected version.

Quantifies how much of the LEARNED controllers' behaviour is driven by price FORESIGHT rather
than control skill, by comparing the SAME algorithm under two observation regimes on the
identical held-out window and matched seeds (8000-8199):
  * persistence = the honest, no-lookahead main run   (rl_results_temporal_worker_*.json)
  * oracle      = perfect future-price foresight in the obs (rl_results_temporal_oracle_worker_*.json)

Audit #6 fixes vs the previous version:
  (a) covers BOTH PPO and SAC (SAC is the best algorithm in most configs, so a PPO-only
      ablation could not support a general "foresight doesn't help" claim);
  (b) adds a PER-EPISODE PAIRED TEST (t + Wilcoxon + bootstrap 95% CI, via rl_stats) so we can
      state whether the premium is statistically distinguishable from zero, rather than eyeballing
      means.

Foresight premium (per algo, per config) = (persistence_cost - oracle_cost)/persistence_cost*100
(positive => perfect foresight makes the learner cheaper). RL per-episode cost is the mean over
the 5 seeds (matched episode order), the same aggregation the headline verdict uses.

Integrity: same-algorithm comparisons only; oracle files are tagged and never pooled into the
headline verdict. Outputs: foresight_premium.json, FORESIGHT_PREMIUM.md, fig_foresight_premium.png
"""
from __future__ import annotations
import os, sys, json, glob, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_stats import paired_comparison

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEF = os.path.join(HERE, "results")
ALGOS = ["PPO", "SAC"]
ORDER = ["grid_only", "grid_solar", "grid_wind", "grid_solar_wind", "grid_gas",
         "grid_solar_gas", "grid_wind_gas", "grid_solar_wind_gas", "grid_solar_battery",
         "grid_wind_battery", "grid_solar_wind_battery", "all_sources"]


def _episode_costs_by_key(glob_pat, algos):
    """(algo, config) -> list of per-seed per-episode cost arrays, from all worker files."""
    d = {}
    for p in glob.glob(glob_pat):
        try:
            runs = json.load(open(p))
        except Exception:
            continue
        for r in runs:
            if isinstance(r, dict) and r.get("algorithm") in algos and "eval_costs" in r:
                d.setdefault((r["algorithm"], r["config"]), []).append(
                    np.asarray(r["eval_costs"], dtype=float))
    return d


def _seed_mean_vector(arrays):
    """Element-wise mean across seed arrays (truncated to the shortest), like the verdict."""
    m = min(len(a) for a in arrays)
    return np.mean(np.vstack([a[:m] for a in arrays]), axis=0)


def _rulebased_by_config(verdict_path):
    rb = {}
    if os.path.isfile(verdict_path):
        v = json.load(open(verdict_path))
        for c in v.get("comparisons", []):
            rb.setdefault(c["config"], float(c["rulebased_cost_mean"]))
    return rb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=DEF)
    ap.add_argument("--verdict", default=None,
                    help="verdict json for RuleBased costs (default: <results-dir>/rl_temporal_verdict.json)")
    args = ap.parse_args()
    rd = args.results_dir
    verdict_path = args.verdict or os.path.join(rd, "rl_temporal_verdict.json")

    persist = _episode_costs_by_key(os.path.join(rd, "rl_results_temporal_worker_*.json"), ALGOS)
    oracle = _episode_costs_by_key(os.path.join(rd, "rl_results_temporal_oracle_worker_*.json"), ALGOS)
    rb = _rulebased_by_config(verdict_path)

    rows = []
    for algo in ALGOS:
        for c in ORDER:
            kp, ko = (algo, c), (algo, c)
            if kp not in persist or ko not in oracle:
                continue
            pv = _seed_mean_vector(persist[kp])
            ov = _seed_mean_vector(oracle[ko])
            n = min(len(pv), len(ov))
            pv, ov = pv[:n], ov[:n]
            # paired_comparison(a=oracle, b=persistence): mean_diff = mean(persist - oracle)
            # >0 => oracle cheaper; pct_improvement = premium %; t/wilcoxon test premium != 0.
            st = paired_comparison(ov, pv)
            row = {"algo": algo, "config": c,
                   "persistence_cost": float(pv.mean()), "oracle_cost": float(ov.mean()),
                   "foresight_premium_pct": st["pct_improvement"],
                   "premium_t_p": st["t_pvalue"], "premium_wilcoxon_p": st["wilcoxon_pvalue"],
                   "premium_ci95": [st["boot_ci95_low"], st["boot_ci95_high"]],
                   "premium_sig": bool(st["ci_excludes_zero"])}
            if c in rb:
                row["rulebased_cost"] = rb[c]
                row["persistence_beats_rb"] = bool(pv.mean() < rb[c])
                row["oracle_beats_rb"] = bool(ov.mean() < rb[c])
            rows.append(row)

    by_algo = {a: [r["foresight_premium_pct"] for r in rows if r["algo"] == a] for a in ALGOS}
    summary = {a: {"mean_premium_pct": float(np.mean(v)) if v else None,
                   "min": float(np.min(v)) if v else None,
                   "max": float(np.max(v)) if v else None,
                   "n_sig": int(sum(1 for r in rows if r["algo"] == a and r["premium_sig"]))}
               for a, v in by_algo.items()}
    out = {"note": "Per-episode paired foresight premium (persistence vs oracle) for PPO and SAC; "
                   "premium=(persist-oracle)/persist*100; sig = bootstrap 95% CI excludes 0.",
           "summary": summary, "rows": rows}
    os.makedirs(rd, exist_ok=True)
    json.dump(out, open(os.path.join(rd, "foresight_premium.json"), "w"), indent=2)

    L = ["# Foresight-Premium Ablation (§5.6) -- PPO + SAC, paired test\n",
         "Held-out weekly cost under no-lookahead (persistence) vs perfect price foresight (oracle), "
         "matched seeds/window. Premium = how much cheaper the learner becomes with foresight. "
         "p = paired t-test that premium != 0; sig = bootstrap 95% CI on the mean per-episode "
         "difference excludes zero.\n",
         "| Algo | Config | Persist $ | Oracle $ | Premium | t-p | sig? | P<RB | O<RB |",
         "|---|---|---:|---:|---:|---:|:--:|:--:|:--:|"]
    for r in rows:
        pb = ("y" if r.get("persistence_beats_rb") else "n") if "persistence_beats_rb" in r else "-"
        ob = ("y" if r.get("oracle_beats_rb") else "n") if "oracle_beats_rb" in r else "-"
        L.append(f"| {r['algo']} | {r['config']} | {r['persistence_cost']:,.0f} | "
                 f"{r['oracle_cost']:,.0f} | {r['foresight_premium_pct']:+.2f}% | "
                 f"{r['premium_t_p']:.3g} | {'Y' if r['premium_sig'] else 'n'} | {pb} | {ob} |")
    for a in ALGOS:
        s = summary[a]
        if s["mean_premium_pct"] is not None:
            L.append(f"\n**{a}** mean premium {s['mean_premium_pct']:+.2f}% "
                     f"(range {s['min']:+.2f}..{s['max']:+.2f}); significant in {s['n_sig']}/"
                     f"{len(by_algo[a])} configs.")
    open(os.path.join(rd, "FORESIGHT_PREMIUM.md"), "w", encoding="utf-8").write("\n".join(L))

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        figdir = os.path.join(HERE, "paper", "figures"); os.makedirs(figdir, exist_ok=True)
        cfgs = [c for c in ORDER if any(r["config"] == c for r in rows)]
        x = np.arange(len(cfgs)); w = 0.38
        fig, ax = plt.subplots(figsize=(10, 4))
        for i, a in enumerate(ALGOS):
            vals = [next((r["foresight_premium_pct"] for r in rows if r["algo"] == a and r["config"] == c), 0)
                    for c in cfgs]
            ax.bar(x + (i - 0.5) * w, vals, w, label=a, alpha=0.85, edgecolor="black", lw=0.4)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(cfgs, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Foresight premium (%)")
        ax.set_title("Foresight-premium ablation (PPO & SAC): cost reduction from perfect price foresight")
        ax.legend(); ax.grid(True, axis="y", alpha=0.3); fig.tight_layout()
        fig.savefig(os.path.join(figdir, "fig_foresight_premium.png"), dpi=160); plt.close(fig)
        print("[foresight] wrote paper/figures/fig_foresight_premium.png")
    except ImportError:
        print("[foresight] matplotlib unavailable; skipped figure")

    for a in ALGOS:
        s = summary[a]
        if s["mean_premium_pct"] is not None:
            print(f"[foresight] {a}: mean premium {s['mean_premium_pct']:+.2f}% "
                  f"(sig in {s['n_sig']}/{len(by_algo[a])})")


if __name__ == "__main__":
    main()
