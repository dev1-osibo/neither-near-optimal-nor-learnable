#!/usr/bin/env python3
"""
Matched-protocol baseline evaluation (Gate 2) + battery-sizing sweep (Gate 4a).

Gate 2: evaluate ALL five baselines (DoNothing, RuleBased, Greedy, MPC,
DeterministicOptimal) on the SAME protocol as the RL agents in the verdict —
real-trace substrate, forecast_mode="persistence" (leakage-free, the main-run
condition), held-out test window, seeds 8000..8199 (n=200), all 12 configs.
This lets MPC/Greedy join the headline table on equal footing, and makes the
"MPC ran on a persistence (naive) forecast" fact explicit and reproducible.

Gate 4a: battery-sizing sensitivity of the heuristic-to-oracle gap. For the four
storage configs, re-evaluate RuleBased + DeterministicOptimal at battery
capacity in {10, 20, 40} MWh (rate = capacity/2), same matched protocol. Shows
whether "the heuristic sits ~near the oracle" is robust to sizing (defends the
headline against the "it's a sizing artifact" objection). Hardware sizing only —
NOT a load change (respects D18 anti-p-hacking).

Read-only w.r.t. models. Outputs:
  results/baseline_matched_realtrace.json
  results/sizing_sensitivity_realtrace.json
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from dc_energy_env import DataCenterEnergyEnv
from rl_temporal_split import make_split_ranges
from baselines import (SOURCE_CONFIGS, DoNothingPolicy, RuleBasedPolicy,
                       GreedyPolicy, MPCPolicy, DeterministicOptimalPolicy)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
SPLIT_DATE = "2024-01-01"
EVAL_SEED_BASE = 8000
N_EVAL = 200
STORAGE_CONFIGS = ["grid_solar_battery", "grid_wind_battery",
                   "grid_solar_wind_battery", "all_sources"]


def _make_policy(name, env):
    if name == "DoNothing":            return DoNothingPolicy()
    if name == "RuleBased":            return RuleBasedPolicy()
    if name == "Greedy":               return GreedyPolicy()
    if name == "MPC":                  return MPCPolicy()
    if name == "DeterministicOptimal": return DeterministicOptimalPolicy(env)
    raise ValueError(name)


def eval_policy(env, name, n_episodes=N_EVAL):
    """Run one baseline over n_episodes matched seeds; return per-metric means."""
    pol = _make_policy(name, env)
    costs, carbons, waters, slas = [], [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=EVAL_SEED_BASE + ep)
        if hasattr(pol, "reset"):      # DeterministicOptimal loads episode prices
            pol.reset()
        done, info = False, {}
        while not done:
            action, _ = pol.predict(obs, deterministic=True)
            obs, _r, term, trunc, info = env.step(action)
            done = term or trunc
        costs.append(info.get("episode_cost", 0.0))
        carbons.append(info.get("episode_carbon", 0.0))
        waters.append(info.get("episode_water", 0.0))
        slas.append(info.get("episode_sla_violations", 0))
    return {
        "mean_episode_cost": float(np.mean(costs)),
        "std_episode_cost": float(np.std(costs)),
        "mean_episode_carbon": float(np.mean(carbons)),
        "mean_episode_water": float(np.mean(waters)),
        "mean_sla_violations": float(np.mean(slas)),
        "n_eval_episodes": n_episodes,
        "eval_window": "test", "eval_costs": costs,
    }


def run_gate2(n_eval=N_EVAL):
    ranges = {c: make_split_ranges(DATA_DIR, SOURCE_CONFIGS[c], SPLIT_DATE) for c in SOURCE_CONFIGS}
    baselines = ["DoNothing", "RuleBased", "Greedy", "MPC", "DeterministicOptimal"]
    out = {"protocol": {"forecast_mode": "persistence", "seed_base": EVAL_SEED_BASE,
                        "n_eval_episodes": n_eval, "split_date": SPLIT_DATE,
                        "note": "matched to RL verdict; MPC uses a persistence (naive) forecast"},
           "results": []}
    for cfg_name in SOURCE_CONFIGS:
        test_range = ranges[cfg_name]["test_range"]
        cfg = SOURCE_CONFIGS[cfg_name]
        for b in baselines:
            env = DataCenterEnergyEnv(data_path=DATA_DIR, episode_start_range=test_range,
                                      forecast_mode="persistence", **cfg)
            rec = eval_policy(env, b, n_episodes=n_eval)
            rec.update({"algorithm": b, "config": cfg_name, "is_baseline": True})
            rec.pop("eval_costs", None)  # keep file small for the summary artifact
            out["results"].append(rec)
            print(f"  [G2] {cfg_name:<24} {b:<20} cost=${rec['mean_episode_cost']:,.0f}")
    path = os.path.join(RESULTS_DIR, "baseline_matched_realtrace.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"[G2] wrote {path}")


def run_gate4a():
    ranges = {c: make_split_ranges(DATA_DIR, SOURCE_CONFIGS[c], SPLIT_DATE) for c in STORAGE_CONFIGS}
    sizes_kwh = [10000, 20000, 40000]
    out = {"protocol": {"forecast_mode": "persistence", "seed_base": EVAL_SEED_BASE,
                        "n_eval_episodes": N_EVAL, "note": "battery-sizing sensitivity; "
                        "rate=capacity/2; heuristic (RuleBased) vs oracle (DeterministicOptimal)"},
           "results": []}
    for cfg_name in STORAGE_CONFIGS:
        test_range = ranges[cfg_name]["test_range"]
        cfg = SOURCE_CONFIGS[cfg_name]
        for cap in sizes_kwh:
            row = {"config": cfg_name, "battery_capacity_kwh": cap, "battery_max_rate_kw": cap // 2}
            for b in ["RuleBased", "DeterministicOptimal"]:
                env = DataCenterEnergyEnv(data_path=DATA_DIR, episode_start_range=test_range,
                                          forecast_mode="persistence",
                                          battery_capacity_kwh=cap, battery_max_rate_kw=cap // 2,
                                          **cfg)
                rec = eval_policy(env, b)
                row[b] = rec["mean_episode_cost"]
            gap = (row["RuleBased"] - row["DeterministicOptimal"]) / row["RuleBased"] * 100.0
            row["rb_gap_to_oracle_pct"] = gap
            out["results"].append(row)
            print(f"  [G4a] {cfg_name:<24} batt={cap/1000:>4.0f}MWh  RB=${row['RuleBased']:,.0f} "
                  f"oracle=${row['DeterministicOptimal']:,.0f}  gap={gap:.2f}%")
    path = os.path.join(RESULTS_DIR, "sizing_sensitivity_realtrace.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"[G4a] wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", choices=["2", "4a", "both"], default="both")
    ap.add_argument("--n-eval", type=int, default=N_EVAL, help="episodes per eval (smoke tests)")
    args = ap.parse_args()
    if args.gate in ("2", "both"):
        run_gate2(n_eval=args.n_eval)
    if args.gate in ("4a", "both"):
        run_gate4a()
