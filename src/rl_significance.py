"""
RL Significance Tests (Part 1)
==============================
Re-evaluates every trained agent (4 algos x 12 configs x 5 seeds) and all
5 baselines on a shared, enlarged set of episodes, then runs paired
statistics to decide whether the RL cost advantage is real.

Two evaluation windows are supported via --split:
  full : sample episodes across the whole dataset (matches how agents were
         trained/eval'd originally). This is the "with overlap" number.
  test : sample episodes only from the held-out period (>= split date). This
         is the leakage-aware smoke test on the EXISTING agents (step 'a').

Outputs:
  results/rl_significance_<split>.json  -- full statistics
  prints a paper-ready table to stdout / log.

Usage:
  python rl_significance.py --split full --n-episodes 200 --n-procs 8
  python rl_significance.py --split test --n-episodes 200 --n-procs 8
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
from rl_temporal_split import make_split_ranges
from rl_stats import paired_comparison, holm_bonferroni
from baselines import SOURCE_CONFIGS

ALGOS = ["SAC", "PPO", "TD3", "A2C"]
SEEDS = [42, 123, 456, 789, 1024]


def _load_sb3(algo, path):
    """Load an SB3 model by algo name. Import inside for clean multiprocessing."""
    from stable_baselines3 import SAC, PPO, TD3, A2C
    cls = {"SAC": SAC, "PPO": PPO, "TD3": TD3, "A2C": A2C}[algo]
    return cls.load(path, device="cpu")


def eval_model_task(args):
    """
    Worker: evaluate ONE trained model on the shared seeds. Returns per-episode
    cost array. Torch limited to 1 thread so N workers don't oversubscribe cores.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass

    algo, config, seed, seeds, start_range = args
    env = make_env(config, episode_start_range=start_range)
    model = _load_sb3(algo, model_path(algo, config, seed))
    out = evaluate_policy_episodes(model, env, seeds)
    return (algo, config, seed, out["cost"].tolist(),
            out["carbon"].tolist(), out["sla"].tolist())


def eval_baselines_for_config(config, seeds, start_range):
    """Evaluate all 5 baselines for one config (sequential; they are cheap)."""
    env = make_env(config, episode_start_range=start_range)
    results = {}
    for name, policy, is_oracle in build_baselines(env):
        out = evaluate_policy_episodes(policy, env, seeds, is_oracle=is_oracle)
        results[name] = out["cost"]
    return results


def run(split, n_episodes, n_procs, seed_base):
    t0 = time.time()

    # Resolve the episode-start window for the chosen split (per-config, since
    # each config filters rows slightly differently -> use all_sources as the
    # canonical index axis; configs share the same timestamp grid).
    start_range = None
    split_info = None
    if split == "test":
        split_info = make_split_ranges(
            DATA_DIR, SOURCE_CONFIGS["all_sources"], split_date="2024-01-01"
        )
        start_range = split_info["test_range"]
        print(f"  [SPLIT] TEST window (>=2024): episode-start {start_range}")
    else:
        print("  [SPLIT] FULL window (entire dataset)")

    seeds = [seed_base + i for i in range(n_episodes)]

    # --- Baselines (sequential per config) ---
    print(f"\n  Evaluating baselines on {n_episodes} episodes...")
    baseline_costs = {}   # config -> {baseline_name: cost_array}
    for config in SOURCE_CONFIGS:
        baseline_costs[config] = eval_baselines_for_config(config, seeds, start_range)
        print(f"    {config}: RuleBased ${baseline_costs[config]['RuleBased'].mean():,.0f} | "
              f"Oracle ${baseline_costs[config]['DeterministicOptimal'].mean():,.0f}")

    # --- RL models (parallel) ---
    tasks = [(algo, config, seed, seeds, start_range)
             for algo in ALGOS for config in SOURCE_CONFIGS for seed in SEEDS]
    print(f"\n  Evaluating {len(tasks)} trained models on {n_procs} procs...")

    rl_costs = {}  # (algo, config) -> list of per-seed cost arrays
    with mp.Pool(processes=n_procs) as pool:
        for i, res in enumerate(pool.imap_unordered(eval_model_task, tasks), 1):
            algo, config, seed, cost, carbon, sla = res
            rl_costs.setdefault((algo, config), []).append(np.asarray(cost))
            if i % 20 == 0:
                print(f"    ...{i}/{len(tasks)} models done")

    # --- Statistics per config ---
    print(f"\n  Running paired statistics...")
    comparisons = []   # flat list for Holm correction + table
    for config in SOURCE_CONFIGS:
        rb = baseline_costs[config]["RuleBased"]
        oracle = baseline_costs[config]["DeterministicOptimal"]
        for algo in ALGOS:
            seed_arrays = rl_costs.get((algo, config), [])
            if not seed_arrays:
                continue
            # Mean cost per episode across the 5 training seeds -> paired vs baseline.
            rl_mean_per_ep = np.mean(np.vstack(seed_arrays), axis=0)
            vs_rule = paired_comparison(rl_mean_per_ep, rb)
            vs_oracle = paired_comparison(oracle, rl_mean_per_ep)  # oracle - rl gap
            # Seed-level dispersion (how consistent across training seeds).
            seed_means = np.array([s.mean() for s in seed_arrays])
            comparisons.append({
                "config": config,
                "algo": algo,
                "rl_cost_mean": float(rl_mean_per_ep.mean()),
                "rl_seed_mean_std": float(seed_means.std(ddof=1)) if len(seed_means) > 1 else 0.0,
                "rulebased_cost_mean": float(rb.mean()),
                "oracle_cost_mean": float(oracle.mean()),
                "vs_rulebased": vs_rule,
                "gap_to_oracle": vs_oracle,
            })

    # Holm-Bonferroni across the whole family of RL-vs-RuleBased tests.
    raw_p = [c["vs_rulebased"]["wilcoxon_pvalue"] for c in comparisons]
    reject, adj = holm_bonferroni(raw_p, alpha=0.05)
    for c, r, a in zip(comparisons, reject, adj):
        c["vs_rulebased"]["holm_adjusted_p"] = float(a)
        c["vs_rulebased"]["significant_holm"] = bool(r)

    payload = {
        "split": split,
        "n_episodes": n_episodes,
        "seed_base": seed_base,
        "split_info": split_info,
        "comparisons": comparisons,
        "elapsed_min": (time.time() - t0) / 60.0,
    }
    outpath = os.path.join(RESULTS_DIR, f"rl_significance_{split}.json")
    with open(outpath, "w") as f:
        json.dump(payload, f, indent=2)

    _print_table(comparisons, split)
    print(f"\n  Saved: {outpath}  ({payload['elapsed_min']:.1f} min)")
    return outpath


def _print_table(comparisons, split):
    """Print a compact significance table (best algo per config highlighted)."""
    print(f"\n{'='*100}")
    print(f"SIGNIFICANCE TABLE ({split} window) — RL vs RuleBased (positive = RL cheaper)")
    print(f"{'='*100}")
    print(f"{'config':<26}{'algo':<6}{'RL $':>10}{'Rule $':>10}{'Δ$/wk':>9}"
          f"{'%':>7}{'d':>7}{'Holm p':>10}{'sig?':>6}")
    by_config = {}
    for c in comparisons:
        by_config.setdefault(c["config"], []).append(c)
    for config, rows in by_config.items():
        best = min(rows, key=lambda x: x["rl_cost_mean"])
        for c in rows:
            v = c["vs_rulebased"]
            star = " *BEST" if c is best else ""
            print(f"{c['config']:<26}{c['algo']:<6}{c['rl_cost_mean']:>10,.0f}"
                  f"{c['rulebased_cost_mean']:>10,.0f}{v['mean_diff']:>9,.0f}"
                  f"{v['pct_improvement']:>6.1f}%{v['cohens_d']:>7.2f}"
                  f"{v['holm_adjusted_p']:>10.4f}{'YES' if v['significant_holm'] else 'no':>6}{star}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["full", "test"], default="full")
    ap.add_argument("--n-episodes", type=int, default=200)
    ap.add_argument("--n-procs", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--seed-base", type=int, default=6000)
    args = ap.parse_args()
    run(args.split, args.n_episodes, args.n_procs, args.seed_base)


if __name__ == "__main__":
    main()
