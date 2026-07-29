#!/usr/bin/env python3
"""Summarise the leakage-free temporal verdict into a compact per-config table.

For each of the 12 source configs, reports:
  - best RL algo (lowest mean test cost) and its % improvement vs RuleBased
  - whether that best algo significantly beats RuleBased (Holm-corrected)
  - the RuleBased gap to the oracle (DeterministicOptimal) as a %  [= headroom ceiling]
  - per-algo Δ% vs RuleBased (to show the storage-degradation pattern)

Read-only. Grounds the Paper 1 results section in the actual artifact.
"""
import json
import os
import sys

CONFIG_ORDER = [
    "grid_only", "grid_solar", "grid_wind", "grid_solar_wind",
    "grid_gas", "grid_solar_gas", "grid_wind_gas", "grid_solar_wind_gas",
    "grid_solar_battery", "grid_wind_battery", "grid_solar_wind_battery",
    "all_sources",
]


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vpath = os.path.join(here, "results", "rl_backup_20260716",
                         "rl_temporal_verdict_realtrace.json")
    if len(sys.argv) > 1:
        vpath = sys.argv[1]
    with open(vpath) as fh:
        v = json.load(fh)

    comps = v["comparisons"]
    by_cfg = {}
    for c in comps:
        by_cfg.setdefault(c["config"], []).append(c)

    print(f"split_date={v.get('split_date')} test_range={v.get('test_range')} "
          f"n_episodes={v.get('n_episodes')} seed_base={v.get('seed_base')}\n")

    header = f"{'config':<24} {'best':<5} {'bestΔ%':>7} {'sig':>4} {'RB_gap_oracle%':>15}  per-algo Δ% vs RB"
    print(header)
    print("-" * len(header))

    n_sig_wins = 0
    for cfg in CONFIG_ORDER:
        rows = by_cfg.get(cfg, [])
        if not rows:
            continue
        # best = lowest RL cost
        best = min(rows, key=lambda r: r["rl_cost_mean"])
        best_algo = best["algo"]
        best_pct = best["vs_rulebased"]["pct_improvement"]
        best_sig = best["vs_rulebased"]["significant_holm"] and best["vs_rulebased"]["a_wins"]
        if best_sig:
            n_sig_wins += 1
        # RB gap to oracle: use any row (rulebased & oracle same across algos); derive %
        rb = best["rulebased_cost_mean"]
        orc = best["oracle_cost_mean"]
        rb_gap = (rb - orc) / rb * 100.0
        per = " ".join(
            f"{r['algo']}={r['vs_rulebased']['pct_improvement']:+.1f}"
            + ("*" if (r['vs_rulebased']['significant_holm'] and r['vs_rulebased']['a_wins']) else "")
            + ("!" if (r['vs_rulebased']['significant_holm'] and not r['vs_rulebased']['a_wins']) else "")
            for r in sorted(rows, key=lambda x: x["algo"])
        )
        print(f"{cfg:<24} {best_algo:<5} {best_pct:>+7.2f} {('YES' if best_sig else 'no'):>4} "
              f"{rb_gap:>14.2f}%  {per}")
        print(f"    abs: RB=${rb:,.0f}  oracle=${orc:,.0f}  best({best_algo})=${best['rl_cost_mean']:,.0f}")

    print(f"\nSIGNIFICANT best-algo wins over RuleBased (Holm): {n_sig_wins}/12")
    print("Legend: *=significantly beats RB; !=significantly WORSE than RB; Δ%>0 = cheaper than RB.")


if __name__ == "__main__":
    main()
