"""
Strategy-Level Adversarial Tests on Trained Agents (Part 2)
===========================================================
Takes the trained policies and attacks them with out-of-distribution regimes
to find where a learned policy generalizes vs where it breaks. Honest by
design: every scenario reports whether the RL agent STILL beats a simple
rule-based baseline, and flags scenarios where it LOSES.

Scenarios covered here (data/actuator transforms):
  price_shock, flat_market, demand_shock, missing_gas, biased_forecast,
  zero_flexibility, worst_case_regret

The temporal hold-out probe (train/test leakage) is run separately via
    rl_significance.py --split test
because it reuses the exact same paired-statistics machinery.

Outputs:
  results/rl_adversarial.json
  printed per-scenario regret table.

Usage:
  python rl_adversarial.py --algo PPO --n-episodes 100 --n-procs 8
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(__file__))
from rl_eval_common import (
    make_env, evaluate_policy_episodes, build_baselines, model_path,
    RESULTS_DIR, DATA_DIR,
)
from rl_stats import paired_comparison
from rl_scenario_wrapper import SCENARIOS, ClampFlexibilityPolicy
from baselines import SOURCE_CONFIGS

SEEDS = [42, 123, 456, 789, 1024]

# Configs where flexibility levers actually exist -> most meaningful stress.
DEFAULT_CONFIGS = ["all_sources", "grid_solar_wind_battery", "grid_wind_gas"]


def _load_sb3(algo, path):
    from stable_baselines3 import SAC, PPO, TD3, A2C
    cls = {"SAC": SAC, "PPO": PPO, "TD3": TD3, "A2C": A2C}[algo]
    return cls.load(path, device="cpu")


def _rl_mean_costs(algo, config, seeds, scenario, start_range):
    """Evaluate all 5 seeds of one algo under a scenario; return mean-per-episode
    cost array (averaged across seeds) plus the raw per-seed arrays."""
    per_seed = []
    for seed in SEEDS:
        env = make_env(config, episode_start_range=start_range)
        if scenario in SCENARIOS:
            env = SCENARIOS[scenario](env)
        model = _load_sb3(algo, model_path(algo, config, seed))
        policy = model
        if scenario == "zero_flexibility":
            policy = ClampFlexibilityPolicy(model)
        out = evaluate_policy_episodes(policy, env, seeds)
        per_seed.append(out["cost"])
    stacked = np.vstack(per_seed)
    return stacked.mean(axis=0), per_seed


def _baseline_costs(config, seeds, scenario, start_range):
    """Evaluate RuleBased + Oracle under the same scenario."""
    env = make_env(config, episode_start_range=start_range)
    if scenario in SCENARIOS:
        env = SCENARIOS[scenario](env)
    out = {}
    for name, policy, is_oracle in build_baselines(env):
        if name not in ("RuleBased", "DeterministicOptimal", "DoNothing"):
            continue
        # oracle/env must be reset fresh per policy; rebuild env for the oracle so
        # it reads the (already transformed) prices.
        e2 = make_env(config, episode_start_range=start_range)
        if scenario in SCENARIOS:
            e2 = SCENARIOS[scenario](e2)
        if is_oracle:
            from baselines import DeterministicOptimalPolicy
            policy = DeterministicOptimalPolicy(e2)
        res = evaluate_policy_episodes(policy, e2, seeds, is_oracle=is_oracle)
        out[name] = res["cost"]
    return out


def scenario_task(args):
    """Worker: run one (scenario, config) cell for the chosen algo."""
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass

    algo, config, scenario, n_episodes, seed_base = args
    seeds = [seed_base + i for i in range(n_episodes)]
    rl_mean, _ = _rl_mean_costs(algo, config, seeds, scenario, None)
    base = _baseline_costs(config, seeds, scenario, None)

    vs_rule = paired_comparison(rl_mean, base["RuleBased"])
    # Worst-case single-episode regret (RL - RuleBased), positive == RL worse.
    regret = rl_mean - base["RuleBased"]
    worst_idx = int(np.argmax(regret))

    return {
        "algo": algo,
        "config": config,
        "scenario": scenario,
        "rl_cost_mean": float(rl_mean.mean()),
        "rulebased_cost_mean": float(base["RuleBased"].mean()),
        "oracle_cost_mean": float(base["DeterministicOptimal"].mean()),
        "donothing_cost_mean": float(base["DoNothing"].mean()),
        "vs_rulebased": vs_rule,
        "rl_beats_rule": bool(vs_rule["mean_diff"] > 0),
        "worst_case_regret_usd": float(regret[worst_idx]),
        "worst_case_seed": int(seeds[worst_idx]),
        "mean_regret_usd": float(regret.mean()),
    }


def run(algo, configs, n_episodes, n_procs, seed_base):
    t0 = time.time()
    scenarios = list(SCENARIOS.keys()) + ["zero_flexibility"]
    tasks = [(algo, c, s, n_episodes, seed_base) for c in configs for s in scenarios]
    print(f"  Adversarial battery: algo={algo}, {len(configs)} configs x "
          f"{len(scenarios)} scenarios = {len(tasks)} cells on {n_procs} procs")

    results = []
    with mp.Pool(processes=n_procs) as pool:
        for i, r in enumerate(pool.imap_unordered(scenario_task, tasks), 1):
            results.append(r)
            print(f"    [{i}/{len(tasks)}] {r['scenario']:<16} {r['config']:<24} "
                  f"RL ${r['rl_cost_mean']:,.0f} vs Rule ${r['rulebased_cost_mean']:,.0f} "
                  f"-> {'WIN' if r['rl_beats_rule'] else 'LOSS'}")

    payload = {
        "algo": algo,
        "n_episodes": n_episodes,
        "seed_base": seed_base,
        "configs": configs,
        "results": results,
        "elapsed_min": (time.time() - t0) / 60.0,
    }
    outpath = os.path.join(RESULTS_DIR, "rl_adversarial.json")
    with open(outpath, "w") as f:
        json.dump(payload, f, indent=2)

    _print_table(results)
    losses = [r for r in results if not r["rl_beats_rule"]]
    print(f"\n  SCENARIOS WHERE {algo} LOSES TO RULE-BASED: {len(losses)}/{len(results)}")
    for r in losses:
        print(f"    - {r['scenario']} @ {r['config']}: "
              f"RL ${r['rl_cost_mean']:,.0f} > Rule ${r['rulebased_cost_mean']:,.0f}")
    print(f"\n  Saved: {outpath}  ({payload['elapsed_min']:.1f} min)")
    return outpath


def _print_table(results):
    print(f"\n{'='*104}")
    print(f"ADVERSARIAL RESULTS (positive Δ = RL cheaper than RuleBased)")
    print(f"{'='*104}")
    print(f"{'scenario':<16}{'config':<24}{'RL $':>10}{'Rule $':>10}"
          f"{'Δ$/wk':>9}{'%':>7}{'worstReg$':>11}{'win?':>6}")
    for r in sorted(results, key=lambda x: (x["scenario"], x["config"])):
        v = r["vs_rulebased"]
        print(f"{r['scenario']:<16}{r['config']:<24}{r['rl_cost_mean']:>10,.0f}"
              f"{r['rulebased_cost_mean']:>10,.0f}{v['mean_diff']:>9,.0f}"
              f"{v['pct_improvement']:>6.1f}%{r['worst_case_regret_usd']:>11,.0f}"
              f"{'YES' if r['rl_beats_rule'] else 'NO':>6}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="PPO", choices=["SAC", "PPO", "TD3", "A2C"])
    ap.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    ap.add_argument("--n-episodes", type=int, default=100)
    ap.add_argument("--n-procs", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--seed-base", type=int, default=7000)
    args = ap.parse_args()
    run(args.algo, args.configs, args.n_episodes, args.n_procs, args.seed_base)


if __name__ == "__main__":
    main()
