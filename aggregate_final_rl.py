"""
Aggregate the final 240-run RL results and compare against baselines.
Reads both workers' progress files (source of truth for completed runs),
computes mean +/- std episode cost per (algorithm, config), and answers the
one decisive question: did ANY algorithm x config beat the rule-based baseline
with acceptable SLA compliance?
"""
import json
import os
from collections import defaultdict
import statistics

CKPT = "/home/ubuntu/optena/checkpoints"

# Baselines (weekly episode cost, all_sources config) from baseline_results.json / handoff note
BASELINES = {
    "DoNothing": 47116,
    "RuleBased": 39805,
    "Greedy": 43673,
    "MPC": 43677,
    "DeterministicOptimal": 38358,  # oracle / theoretical ceiling
}
RULE_BASED = BASELINES["RuleBased"]
ORACLE = BASELINES["DeterministicOptimal"]

results = []
for wid in (0, 1):
    p = os.path.join(CKPT, f"progress_worker_{wid}.json")
    with open(p) as f:
        d = json.load(f)
    for r in d.get("results", []):
        if "error" not in r:
            results.append(r)

print(f"Total result records: {len(results)}")

# Group by (algorithm, config)
by_algo_config = defaultdict(list)
by_algo = defaultdict(list)
for r in results:
    algo = r.get("algorithm")
    cfg = r.get("config")
    cost = r.get("mean_episode_cost")
    sla = r.get("mean_sla_violations")
    if cost is None:
        continue
    by_algo_config[(algo, cfg)].append((cost, sla, r.get("seed")))
    by_algo[algo].append(cost)

# Per-algorithm summary across ALL configs
print("\n=== PER-ALGORITHM MEAN EPISODE COST (across all configs/seeds) ===")
for algo in sorted(by_algo):
    costs = by_algo[algo]
    print(f"  {algo:5s}: mean=${statistics.mean(costs):,.0f}  "
          f"min=${min(costs):,.0f}  max=${max(costs):,.0f}  n={len(costs)}")

# The decisive comparison: best mean per (algo, config) vs rule-based.
# Focus especially on all_sources config (the target), but report all.
print(f"\n=== vs RULE-BASED (${RULE_BASED:,}) and ORACLE (${ORACLE:,}) ===")
beat_rulebased = []
for (algo, cfg), vals in sorted(by_algo_config.items()):
    costs = [v[0] for v in vals]
    slas = [v[1] for v in vals if v[1] is not None]
    mean_cost = statistics.mean(costs)
    std_cost = statistics.pstdev(costs) if len(costs) > 1 else 0.0
    mean_sla = statistics.mean(slas) if slas else float("nan")
    beats = mean_cost < RULE_BASED
    flag = "  <-- BEATS RULE-BASED" if beats else ""
    if beats:
        beat_rulebased.append((algo, cfg, mean_cost, mean_sla))
    # Only print all_sources + any that beat, to keep output focused
    if cfg == "all_sources" or beats:
        print(f"  {algo:5s} {cfg:24s} mean=${mean_cost:,.0f} +/- {std_cost:,.0f}  "
              f"SLA_viol/wk={mean_sla:.1f}{flag}")

print("\n=== VERDICT ===")
print(f"Configs (algo x source-config) beating rule-based: {len(beat_rulebased)} "
      f"of {len(by_algo_config)}")
if beat_rulebased:
    print("Beating configs (mean cost, SLA violations/week):")
    for algo, cfg, mc, ms in sorted(beat_rulebased, key=lambda x: x[2]):
        acceptable = "ACCEPTABLE SLA" if (ms == ms and ms <= 2) else "HIGH SLA VIOLATIONS"
        print(f"  {algo} {cfg}: ${mc:,.0f}  SLA={ms:.1f}/wk  [{acceptable}]")
else:
    print("NO algorithm x config beat the rule-based baseline on mean cost.")

# Also: absolute best single run
best = min(results, key=lambda r: r.get("mean_episode_cost", 1e12))
print(f"\nAbsolute best single run: {best['run_id']} = ${best['mean_episode_cost']:,.0f}, "
      f"SLA={best.get('mean_sla_violations')}/wk")
