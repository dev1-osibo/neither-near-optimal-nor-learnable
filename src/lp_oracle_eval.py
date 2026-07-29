"""
RuleBased -> TRUE-optimum gap on the held-out test window (audit #3).
For every config, over the matched test seeds (8000..8000+n), compute:
  - RuleBased weekly cost (fresh rollout on the CORRECTED substrate),
  - LP true optimum, full mode (battery+gas+grid+renewables + env-faithful 1/12 deferral),
  - LP true optimum, storage-only mode (deferral disabled).
Report per-episode gaps (RB-LP)/RB, summarized by MEAN and MEDIAN (median is robust to the
rare price-spike weeks that dominate the mean). Saves results/lp_oracle_gap.json.

This is eval-only, no GPU, and tells us whether the paper's "heuristic near-optimal (3-7%)"
claim survives against a genuine optimum BEFORE committing to any retrain.
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dc_energy_env import DataCenterEnergyEnv
from baselines import RuleBasedPolicy, SOURCE_CONFIGS
import lp_oracle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
SEED_BASE = 8000
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
CONFIGS = list(SOURCE_CONFIGS.keys())


def rb_rollout(env, seed):
    obs, _ = env.reset(seed=int(seed)); pol = RuleBasedPolicy(); done = False; info = {}
    while not done:
        a, _ = pol.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(a); done = term or trunc
    return info["episode_cost"]


def episode_args(env, seed):
    env.reset(seed=int(seed))
    s = env.episode_start; T = env.episode_length; sl = slice(s, s + T)
    return dict(demand=env.total_demand[sl], price=env.grid_price[sl],
                gas_price=env.gas_price[sl], solar=env.solar_available[sl],
                wind=env.wind_available[sl], cap_kwh=env.battery_capacity_kwh,
                rate_kw=env.battery_max_rate_kw, eff=env.battery_efficiency,
                gas_cap_kw=env.gas_capacity_kw)


def summarize(gaps):
    g = np.asarray(gaps, float)
    return dict(mean=float(g.mean()), median=float(np.median(g)),
                p25=float(np.percentile(g, 25)), p75=float(np.percentile(g, 75)),
                min=float(g.min()), max=float(g.max()))


def main():
    seeds = [SEED_BASE + i for i in range(N)]
    out = {"n_episodes": N, "seed_base": SEED_BASE, "note":
           "gap%=(RB-LP)/RB on corrected substrate (5MW solar, 0.90 RT battery)", "configs": {}}
    print(f"LP-oracle gap eval: {len(CONFIGS)} configs x {N} episodes\n")
    t0 = time.time()
    for cfg in CONFIGS:
        env = DataCenterEnergyEnv(data_path=DATA, **SOURCE_CONFIGS[cfg])
        rb_costs, lpf, lps, gfull, gstor = [], [], [], [], []
        fail = 0
        for seed in seeds:
            rb = rb_rollout(env, seed)
            args = episode_args(env, seed)
            full, s1 = lp_oracle.solve_episode(allow_defer=True, **args)
            stor, s2 = lp_oracle.solve_episode(allow_defer=False, **args)
            if full is None or stor is None:
                fail += 1; continue
            rb_costs.append(rb); lpf.append(full); lps.append(stor)
            gfull.append((rb - full) / rb * 100.0)
            gstor.append((rb - stor) / rb * 100.0)
        out["configs"][cfg] = {
            "rb_mean_cost": float(np.mean(rb_costs)),
            "lp_full_mean_cost": float(np.mean(lpf)),
            "lp_storage_mean_cost": float(np.mean(lps)),
            "gap_full_pct": summarize(gfull),
            "gap_storage_pct": summarize(gstor),
            "n_failed": fail,
        }
        gf, gsu = out["configs"][cfg]["gap_full_pct"], out["configs"][cfg]["gap_storage_pct"]
        print(f"  {cfg:<26} full: median {gf['median']:5.1f}% mean {gf['mean']:5.1f}%  |  "
              f"storage: median {gsu['median']:5.1f}% mean {gsu['mean']:5.1f}%  "
              f"[{time.time()-t0:.0f}s]")
        with open(os.path.join(RESULTS, "lp_oracle_gap.json"), "w") as f:
            json.dump(out, f, indent=2)
    print(f"\nDone in {time.time()-t0:.0f}s -> results/lp_oracle_gap.json")


if __name__ == "__main__":
    main()
