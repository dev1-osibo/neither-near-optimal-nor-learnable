"""
Leakage-Free Verdict (all 4 algorithms, temporal split)
=======================================================
Combines the temporally-trained RL results (train 2020-2023, eval on held-out
2024-2025) with baselines evaluated on the SAME held-out window and the SAME
eval seeds (8000-8199), then runs paired statistics.

RL per-episode costs are read directly from the temporal results' `eval_costs`
field (already computed on seeds 8000-8199), so no RL re-evaluation is needed.
Baselines are evaluated fresh on the test window with matched seeds -> fully
paired comparison.

Output: results/rl_temporal_verdict.json + a paper-ready table.

Usage: python rl_temporal_verdict.py --n-procs 8
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(__file__))
from rl_eval_common import make_env, evaluate_policy_episodes, build_baselines, RESULTS_DIR, DATA_DIR
from rl_temporal_split import make_split_ranges
from rl_stats import paired_comparison, holm_bonferroni
from baselines import SOURCE_CONFIGS

ALGOS = ["SAC", "PPO", "TD3", "A2C"]
EVAL_SEED_BASE = 8000          # MUST match train_rl_temporal.EVAL_SEED_BASE
SPLIT_DATE = "2024-01-01"


def load_temporal_results():
    """Load all temporal run dicts from every worker file. Returns list."""
    runs = []
    for path in glob.glob(os.path.join(RESULTS_DIR, "rl_results_temporal_worker_*.json")):
        with open(path) as f:
            runs.extend(json.load(f))
    return [r for r in runs if "error" not in r and "eval_costs" in r]


def baseline_task(args):
    """Worker: evaluate the 5 baselines on the test window for one config."""
    os.environ["OMP_NUM_THREADS"] = "1"
    config, seeds, test_range = args
    env = make_env(config, episode_start_range=test_range)
    out = {}
    for name, policy, is_oracle in build_baselines(env):
        # Oracle needs its own env instance bound at construction; rebuild per policy.
        if is_oracle:
            from baselines import DeterministicOptimalPolicy
            e2 = make_env(config, episode_start_range=test_range)
            res = evaluate_policy_episodes(DeterministicOptimalPolicy(e2), e2, seeds, is_oracle=True)
        else:
            res = evaluate_policy_episodes(policy, env, seeds, is_oracle=False)
        out[name] = res["cost"].tolist()
    return config, out


def run(n_procs, n_episodes):
    seeds = [EVAL_SEED_BASE + i for i in range(n_episodes)]

    # Test-window bounds (same split the training used).
    split = make_split_ranges(DATA_DIR, SOURCE_CONFIGS["all_sources"], SPLIT_DATE)
    test_range = split["test_range"]
    print(f"  Test window: episode-start {test_range} (>= {SPLIT_DATE})")

    # --- RL: group temporal eval_costs by (algo, config), average across seeds ---
    runs = load_temporal_results()
    print(f"  Loaded {len(runs)} temporal RL runs")
    rl = {}  # (algo, config) -> list of per-seed cost arrays
    for r in runs:
        costs = np.asarray(r["eval_costs"], dtype=float)[:n_episodes]
        rl.setdefault((r["algorithm"], r["config"]), []).append(costs)

    # --- Baselines on the matched test window (parallel over configs) ---
    print(f"  Evaluating baselines on {n_episodes} matched episodes ({n_procs} procs)...")
    tasks = [(c, seeds, test_range) for c in SOURCE_CONFIGS]
    baseline_costs = {}
    with mp.Pool(processes=n_procs) as pool:
        for config, out in pool.imap_unordered(baseline_task, tasks):
            baseline_costs[config] = {k: np.asarray(v) for k, v in out.items()}
            print(f"    {config}: RuleBased ${baseline_costs[config]['RuleBased'].mean():,.0f} | "
                  f"Oracle ${baseline_costs[config]['DeterministicOptimal'].mean():,.0f}")

    # --- Paired stats per (algo, config) vs RuleBased + gap to oracle ---
    comparisons = []
    for config in SOURCE_CONFIGS:
        rb = baseline_costs[config]["RuleBased"]
        oracle = baseline_costs[config]["DeterministicOptimal"]
        for algo in ALGOS:
            seed_arrays = rl.get((algo, config), [])
            if not seed_arrays:
                continue
            m = min(len(a) for a in seed_arrays)
            rl_mean = np.mean(np.vstack([a[:m] for a in seed_arrays]), axis=0)
            n = min(len(rl_mean), len(rb))
            vs_rule = paired_comparison(rl_mean[:n], rb[:n])
            vs_oracle = paired_comparison(oracle[:n], rl_mean[:n])
            seed_means = np.array([a.mean() for a in seed_arrays])
            comparisons.append({
                "config": config, "algo": algo,
                "rl_cost_mean": float(rl_mean.mean()),
                "rl_seed_std": float(seed_means.std(ddof=1)) if len(seed_means) > 1 else 0.0,
                "rulebased_cost_mean": float(rb.mean()),
                "oracle_cost_mean": float(oracle.mean()),
                "vs_rulebased": vs_rule,
                "gap_to_oracle": vs_oracle,
            })

    # Holm across the full family of RL-vs-RuleBased tests.
    raw_p = [c["vs_rulebased"]["wilcoxon_pvalue"] for c in comparisons]
    reject, adj = holm_bonferroni(raw_p, alpha=0.05)
    for c, r, a in zip(comparisons, reject, adj):
        c["vs_rulebased"]["holm_adjusted_p"] = float(a)
        c["vs_rulebased"]["significant_holm"] = bool(r)

    payload = {"split_date": SPLIT_DATE, "test_range": test_range,
               "n_episodes": n_episodes, "seed_base": EVAL_SEED_BASE,
               "comparisons": comparisons}
    outpath = os.path.join(RESULTS_DIR, "rl_temporal_verdict.json")
    with open(outpath, "w") as f:
        json.dump(payload, f, indent=2)

    _print_table(comparisons)
    print(f"\n  Saved: {outpath}")
    return outpath


def _print_table(comparisons):
    print(f"\n{'='*100}")
    print("LEAKAGE-FREE VERDICT (train 2020-2023 -> test 2024-2025) | +Δ = RL cheaper than RuleBased")
    print(f"{'='*100}")
    print(f"{'config':<26}{'algo':<6}{'RL $':>10}{'Rule $':>10}{'Δ$/wk':>9}{'%':>7}"
          f"{'d':>7}{'HolmP':>9}{'sig':>5}")
    by_cfg = {}
    for c in comparisons:
        by_cfg.setdefault(c["config"], []).append(c)
    n_best_sig = 0
    for config, rows in by_cfg.items():
        best = min(rows, key=lambda x: x["rl_cost_mean"])
        for c in rows:
            v = c["vs_rulebased"]
            mark = " *" if c is best else "  "
            print(f"{c['config']:<26}{c['algo']:<6}{c['rl_cost_mean']:>10,.0f}"
                  f"{c['rulebased_cost_mean']:>10,.0f}{v['mean_diff']:>9,.0f}"
                  f"{v['pct_improvement']:>6.1f}%{v['cohens_d']:>7.2f}"
                  f"{v['holm_adjusted_p']:>9.3g}{'Y' if v['significant_holm'] else 'n':>5}{mark}")
        b = best["vs_rulebased"]
        if b["significant_holm"] and b["mean_diff"] > 0:
            n_best_sig += 1
    print(f"\n  Configs where the BEST algo significantly beats RuleBased (Holm): "
          f"{n_best_sig}/{len(by_cfg)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-procs", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--n-episodes", type=int, default=200)
    args = ap.parse_args()
    run(args.n_procs, args.n_episodes)


if __name__ == "__main__":
    main()
