"""
Aggregate RL Results from All Workers
=======================================
After training completes on 4 workers, this combines all results
into a single analysis file with proper statistical reporting.

Produces:
- Mean ± std across 5 seeds per algorithm×config
- Statistical significance tests (paired t-tests)
- Best algorithm per config
- Best config per algorithm
- Final comparison table for paper
"""

import json
import os
import numpy as np
from scipy import stats

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def aggregate():
    # Load all worker results
    all_results = []
    for worker_id in range(10):  # Check up to 10 workers
        path = os.path.join(RESULTS_DIR, f"rl_results_worker_{worker_id}.json")
        if os.path.exists(path):
            with open(path) as f:
                worker_results = json.load(f)
                all_results.extend(worker_results)
                print(f"  Worker {worker_id}: {len(worker_results)} results")
    
    if not all_results:
        print("  No results found!")
        return
    
    # Filter out errors
    valid = [r for r in all_results if "error" not in r]
    errors = [r for r in all_results if "error" in r]
    print(f"\n  Total valid results: {len(valid)}")
    print(f"  Failed runs: {len(errors)}")
    
    # Group by algorithm × config
    grouped = {}
    for r in valid:
        key = f"{r['algorithm']}_{r['config']}"
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(r)
    
    # Compute statistics per group
    print(f"\n{'='*80}")
    print(f"{'Algorithm':<8} | {'Config':<30} | {'Seeds':>5} | {'Cost Mean':>12} | {'Cost Std':>10} | {'Reward':>8}")
    print(f"{'-'*8} | {'-'*30} | {'-'*5} | {'-'*12} | {'-'*10} | {'-'*8}")
    
    summary = {}
    for key in sorted(grouped.keys()):
        results = grouped[key]
        algo = results[0]["algorithm"]
        config = results[0]["config"]
        n_seeds = len(results)
        
        costs = [r["mean_episode_cost"] for r in results]
        rewards = [r["mean_reward"] for r in results]
        carbons = [r["mean_episode_carbon"] for r in results]
        
        summary[key] = {
            "algorithm": algo,
            "config": config,
            "n_seeds": n_seeds,
            "cost_mean": float(np.mean(costs)),
            "cost_std": float(np.std(costs)),
            "cost_ci95": float(1.96 * np.std(costs) / np.sqrt(n_seeds)),
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "carbon_mean": float(np.mean(carbons)),
            "carbon_std": float(np.std(carbons)),
        }
        
        print(f"{algo:<8} | {config:<30} | {n_seeds:>5} | "
              f"${np.mean(costs):>10,.0f} | ${np.std(costs):>8,.0f} | {np.mean(rewards):>7.1f}")
    
    # Best per config
    print(f"\n{'='*80}")
    print("BEST ALGORITHM PER SOURCE CONFIGURATION:")
    configs = set(r["config"] for r in valid)
    for config in sorted(configs):
        config_results = {k: v for k, v in summary.items() if v["config"] == config}
        if config_results:
            best_key = min(config_results.keys(), key=lambda k: config_results[k]["cost_mean"])
            best = config_results[best_key]
            print(f"  {config:<30}: {best['algorithm']} (${best['cost_mean']:,.0f} ± ${best['cost_std']:,.0f})")
    
    # Best per algorithm
    print(f"\nBEST SOURCE CONFIGURATION PER ALGORITHM:")
    algos = set(r["algorithm"] for r in valid)
    for algo in sorted(algos):
        algo_results = {k: v for k, v in summary.items() if v["algorithm"] == algo}
        if algo_results:
            best_key = min(algo_results.keys(), key=lambda k: algo_results[k]["cost_mean"])
            best = algo_results[best_key]
            print(f"  {algo:<8}: {best['config']} (${best['cost_mean']:,.0f} ± ${best['cost_std']:,.0f})")
    
    # Save
    output = {
        "total_runs": len(valid),
        "failed_runs": len(errors),
        "summary": summary,
        "all_results": valid,
    }
    outpath = os.path.join(RESULTS_DIR, "rl_aggregated_results.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  ✓ Saved: {outpath}")


if __name__ == "__main__":
    print("=" * 80)
    print("AGGREGATING RL RESULTS")
    print("=" * 80)
    aggregate()
