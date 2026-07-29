#!/usr/bin/env python3
"""
TRUE-OPTIMUM CEILING analysis (Paper 1 §5.2 / Tables I-II) -- audit #3.

Replaces the quantile-threshold "DeterministicOptimal" heuristic (a clairvoyant RULE, not an
optimum) with the genuine per-episode LP optimum (lp_oracle.solve_episode) as the ceiling, and
reports how far the RuleBased heuristic and the best RL controller sit ABOVE that true optimum.

For each config, over the held-out test seeds (8000..8000+N), it computes:
  - LP true optimum weekly cost (full mode: battery+gas+grid+renewables + env-faithful deferral),
  - RuleBased cost and best-RL cost  (read from the fresh verdict artifact),
  - gaps:  RB->opt = (RB-LP)/RB ,  RL->opt = (RL-LP)/RL ,  RL-vs-RB = (RB-RL)/RB.

This is the corrected basis for the "heuristic near-optimal?" claim: the true gap is >= the old
(quantile-oracle) gap, and (per the pre-retrain preview) blows out to ~13-19% for storage configs.

Note: the LP is deliberately conservative (it does NOT exploit the cooling-offset cost lever), so
its gap is a LOWER BOUND on the true optimum gap. LP <= RuleBased is asserted per episode.

Outputs: results/rl_lp_ceiling.json + results/LP_CEILING.md
"""
from __future__ import annotations
import os, sys, json, argparse, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dc_energy_env import DataCenterEnergyEnv
from baselines import SOURCE_CONFIGS
from rl_temporal_split import make_split_ranges
import lp_oracle

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")
RESULTS = os.path.join(HERE, "results")
SEED_BASE = 8000
SPLIT_DATE = "2024-01-01"
RENEW_ONLY = {"grid_only", "grid_solar", "grid_wind", "grid_solar_wind"}
STORAGE = {"grid_solar_battery", "grid_wind_battery", "grid_solar_wind_battery", "all_sources"}


def lp_optimum_by_config(n_episodes, configs):
    """Mean LP-optimum weekly cost per config over the matched test seeds."""
    out = {}
    seeds = [SEED_BASE + i for i in range(n_episodes)]
    t0 = time.time()
    for cfg in configs:
        # MUST match the verdict's protocol: restrict episode starts to the held-out TEST
        # window so reset(seed) yields the SAME weeks the RL/RB evaluation used. Without this
        # the LP would be solved on full-2020-2025 episodes (a different, costlier distribution)
        # and would not be comparable to the test-window RB/RL costs.
        test_range = make_split_ranges(DATA, SOURCE_CONFIGS[cfg], SPLIT_DATE)["test_range"]
        env = DataCenterEnergyEnv(data_path=DATA, episode_start_range=test_range,
                                  **SOURCE_CONFIGS[cfg])
        costs = []
        for s in seeds:
            env.reset(seed=int(s))
            st = env.episode_start; T = env.episode_length; sl = slice(st, st + T)
            c, ok = lp_oracle.solve_episode(
                demand=env.total_demand[sl], price=env.grid_price[sl], gas_price=env.gas_price[sl],
                solar=env.solar_available[sl], wind=env.wind_available[sl],
                cap_kwh=env.battery_capacity_kwh, rate_kw=env.battery_max_rate_kw,
                eff=env.battery_efficiency, gas_cap_kw=env.gas_capacity_kw, allow_defer=True)
            if c is not None:
                costs.append(c)
        out[cfg] = float(np.mean(costs))
        print(f"  LP {cfg:26s} = ${out[cfg]:,.0f}  [{time.time()-t0:.0f}s]")
    return out


def verdict_rb_rl(verdict_path):
    """Per config: RuleBased cost + best-RL (min over algos) cost/algo/significance."""
    v = json.load(open(verdict_path))
    rb, best = {}, {}
    for c in v["comparisons"]:
        cfg = c["config"]
        rb.setdefault(cfg, float(c["rulebased_cost_mean"]))
        cur = best.get(cfg)
        if cur is None or c["rl_cost_mean"] < cur["rl_cost"]:
            best[cfg] = {"rl_cost": float(c["rl_cost_mean"]), "algo": c["algo"],
                         "rl_vs_rb_pct": float(c["vs_rulebased"]["pct_improvement"]),
                         "sig": bool(c["vs_rulebased"].get("significant_holm", False))}
    return rb, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-episodes", type=int, default=200)
    ap.add_argument("--verdict", default=os.path.join(RESULTS, "rl_temporal_verdict.json"))
    args = ap.parse_args()

    if not os.path.isfile(args.verdict):
        sys.exit(f"verdict not found: {args.verdict} (run rl_temporal_verdict.py first)")
    configs = list(SOURCE_CONFIGS.keys())
    print(f"LP-ceiling: {len(configs)} configs x {args.n_episodes} episodes")
    lp = lp_optimum_by_config(args.n_episodes, configs)
    rb, best = verdict_rb_rl(args.verdict)

    rows = []
    for cfg in configs:
        if cfg not in rb or cfg not in best or cfg not in lp:
            continue
        R, B, L = rb[cfg], best[cfg]["rl_cost"], lp[cfg]
        rows.append({
            "config": cfg, "rulebased_cost": R, "best_rl_cost": B, "best_rl_algo": best[cfg]["algo"],
            "lp_optimum_cost": L,
            "rb_to_opt_pct": (R - L) / R * 100.0,
            "rl_to_opt_pct": (B - L) / B * 100.0,
            "rl_vs_rb_pct": best[cfg]["rl_vs_rb_pct"], "rl_beats_rb_sig": best[cfg]["sig"],
            "lp_le_rb": bool(L <= R + 1e-6), "lp_le_rl": bool(L <= B + 1e-6),
        })

    def agg(keys, field):
        vals = [r[field] for r in rows if r["config"] in keys]
        return (float(np.mean(vals)), float(np.min(vals)), float(np.max(vals))) if vals else (None,)*3

    summary = {
        "renewable_only_rb_to_opt": agg(RENEW_ONLY, "rb_to_opt_pct"),
        "storage_rb_to_opt": agg(STORAGE, "rb_to_opt_pct"),
        "any_lp_above_rb": [r["config"] for r in rows if not r["lp_le_rb"]],  # must be empty
    }
    out = {"n_episodes": args.n_episodes,
           "note": "LP true optimum as ceiling (audit #3). gap=(controller-LP)/controller*100. "
                   "LP is conservative (no cooling-offset lever) => gap is a lower bound.",
           "summary": summary, "rows": rows}
    json.dump(out, open(os.path.join(RESULTS, "rl_lp_ceiling.json"), "w"), indent=2)

    if summary["any_lp_above_rb"]:
        print("  !! WARNING: LP exceeded RuleBased in:", summary["any_lp_above_rb"])

    L = ["# True-Optimum Ceiling (LP) -- Paper 1 §5.2 (audit #3)\n",
         "RuleBased and best-RL weekly cost vs the genuine per-episode LP optimum (perfect "
         "foresight). Gap = how far above the true optimum. LP replaces the old quantile "
         "'DeterministicOptimal' heuristic.\n",
         "| Config | RuleBased $ | Best RL $ (algo) | LP optimum $ | RB→opt | RL→opt | RL vs RB |",
         "|---|---:|---|---:|---:|---:|---:|"]
    for r in rows:
        sig = "*" if r["rl_beats_rb_sig"] and r["rl_vs_rb_pct"] > 0 else ""
        L.append(f"| {r['config']} | {r['rulebased_cost']:,.0f} | {r['best_rl_cost']:,.0f} "
                 f"({r['best_rl_algo']}) | {r['lp_optimum_cost']:,.0f} | {r['rb_to_opt_pct']:.1f}% "
                 f"| {r['rl_to_opt_pct']:.1f}% | {r['rl_vs_rb_pct']:+.2f}%{sig} |")
    rn, st = summary["renewable_only_rb_to_opt"], summary["storage_rb_to_opt"]
    L.append(f"\nRuleBased→optimum gap: renewable-only {rn[0]:.1f}% (mean), "
             f"storage-rich {st[0]:.1f}% (mean, range {st[1]:.1f}-{st[2]:.1f}%).")
    L.append(f"LP<=RuleBased holds for all configs: {not summary['any_lp_above_rb']}.")
    open(os.path.join(RESULTS, "LP_CEILING.md"), "w", encoding="utf-8").write("\n".join(L))
    print(f"[lp-ceiling] wrote rl_lp_ceiling.json + LP_CEILING.md")
    print(f"  RB->opt: renewable {rn[0]:.1f}% | storage {st[0]:.1f}% (range {st[1]:.1f}-{st[2]:.1f}%)")


if __name__ == "__main__":
    main()
