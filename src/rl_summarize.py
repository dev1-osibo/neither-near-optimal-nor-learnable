"""Compact summary of significance (full & test windows) + adversarial results."""
import json, os
R = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

def load(n):
    p = os.path.join(R, n)
    return json.load(open(p)) if os.path.exists(p) else None

for split in ["full", "test"]:
    d = load(f"rl_significance_{split}.json")
    if not d:
        print(f"[{split}] MISSING"); continue
    comps = d["comparisons"]
    print(f"\n===== SIGNIFICANCE [{split}] ({d['n_episodes']} eps) =====")
    # best algo per config + its significance vs rulebased
    by_cfg = {}
    for c in comps:
        by_cfg.setdefault(c["config"], []).append(c)
    n_sig_best = 0
    for cfg, rows in by_cfg.items():
        best = min(rows, key=lambda x: x["rl_cost_mean"])
        v = best["vs_rulebased"]
        sig = v.get("significant_holm")
        n_sig_best += 1 if (sig and v["mean_diff"] > 0) else 0
        print(f"  {cfg:<26} best={best['algo']:<4} RL=${best['rl_cost_mean']:>8,.0f} "
              f"rule=${best['rulebased_cost_mean']:>8,.0f} d%={v['pct_improvement']:>5.1f} "
              f"holmP={v['holm_adjusted_p']:.3g} sig={'Y' if sig else 'N'} "
              f"CI=[{v['boot_ci95_low']:,.0f},{v['boot_ci95_high']:,.0f}]")
    # PPO specifically, all configs
    ppo = [c for c in comps if c["algo"] == "PPO"]
    n_ppo_sig = sum(1 for c in ppo if c["vs_rulebased"].get("significant_holm") and c["vs_rulebased"]["mean_diff"]>0)
    print(f"  -- best-algo beats rule (sig, Holm): {n_sig_best}/{len(by_cfg)} configs")
    print(f"  -- PPO beats rule (sig, Holm): {n_ppo_sig}/{len(ppo)} configs")

a = load("rl_adversarial.json")
if a:
    print(f"\n===== ADVERSARIAL PPO ({a['n_episodes']} eps) =====")
    for r in sorted(a["results"], key=lambda x:(x["scenario"], x["config"])):
        v = r["vs_rulebased"]
        print(f"  {r['scenario']:<16}{r['config']:<24} RL=${r['rl_cost_mean']:>8,.0f} "
              f"rule=${r['rulebased_cost_mean']:>8,.0f} d%={v['pct_improvement']:>5.1f} "
              f"win={'Y' if r['rl_beats_rule'] else 'N'} worstReg=${r['worst_case_regret_usd']:>7,.0f}")
    losses = [r for r in a["results"] if not r["rl_beats_rule"]]
    print(f"  -- PPO loses to rule in {len(losses)}/{len(a['results'])} scenario-cells")
