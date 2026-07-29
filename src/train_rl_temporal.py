"""
RL Retraining with a PROPER TEMPORAL TRAIN/TEST SPLIT (Step b)
==============================================================
Fixes the leakage blindspot: agents are TRAINED only on episodes that start
and finish before the split date (default 2024-01-01), and EVALUATED only on
episodes drawn from the held-out period (>= split date). Nothing the agent
sees at test time was available during training.

Mirrors train_rl_checkpointed.py (checkpoint/resume, spot-safe, per-run model
save) but injects episode_start_range into both train and eval envs.

Usage (per worker, splits the run list across workers):
  python train_rl_temporal.py --worker-id 0 --total-workers 4 --algos PPO
  python train_rl_temporal.py --worker-id 0 --total-workers 4 --algos SAC PPO TD3 A2C
"""

import os
import sys
import json
import time
import signal
import argparse
import traceback
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from dc_energy_env import DataCenterEnergyEnv
from rl_temporal_split import make_split_ranges
from baselines import SOURCE_CONFIGS

from stable_baselines3 import SAC, PPO, TD3, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models_temporal")
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints_temporal")
for d in [RESULTS_DIR, MODELS_DIR, CHECKPOINT_DIR]:
    os.makedirs(d, exist_ok=True)

TOTAL_TIMESTEPS = 1_000_000
CHECKPOINT_FREQ = 100_000     # fewer checkpoints than before -> less serialization overhead
N_EVAL_EPISODES = 200         # final authoritative eval per run
N_CKPT_EVAL = 20              # quick held-out eval logged at EACH checkpoint (progress record)
OFF_POLICY = {"SAC", "TD3"}   # algorithms whose replay buffer must be persisted on checkpoint
# Master seed list. The FIRST FIVE are the original main-run seeds (reproducibility: the
# 5-seed main verdict is unchanged); the rest extend the Pareto sweep to more seeds so the
# carbon/water levers are not underpowered (audit #12). --n-seeds selects a prefix.
SEEDS_MASTER = [42, 123, 456, 789, 1024,
                2048, 3141, 5926, 5358, 9793,
                2384, 6264, 3383, 2795, 288,
                4197, 1693, 9937, 5105, 8209]
SEEDS = SEEDS_MASTER[:5]
SPLIT_DATE = "2024-01-01"
EVAL_SEED_BASE = 8000         # test-window eval seeds (disjoint from all others)
ALGORITHMS = {"SAC": SAC, "PPO": PPO, "TD3": TD3, "A2C": A2C}

# Objective weights passed to the env. Default = the main run's balance; overridden
# per weight-point in the Pareto sweep (AWS-7). SLA weight held fixed as a constraint.
ALPHAS = {"alpha_cost": 0.4, "alpha_carbon": 0.3, "alpha_water": 0.2, "alpha_sla": 0.1}

# Forecast mode threaded into BOTH train and eval envs. "persistence" (no lookahead)
# is the leakage-free default and produced the main verdict. "oracle" (perfect price
# foresight in the observation) is used ONLY for the labelled foresight-premium ablation
# (Gate 3): it must be run under its own --tag so it never pools into the main verdict.
FORECAST_MODE = "persistence"

# Optional infrastructure-sizing overrides (Gate 4 sizing sensitivity). None => the env's
# default sizing (5MW solar / 5MW wind / 20MWh battery @ 10MW / 2MW gas). Backward compatible.
CAP_OVERRIDES = {}


def _env_kwargs(episode_start_range, cfg):
    """Assemble env kwargs shared by train + eval so forecast_mode/sizing stay in sync."""
    kw = dict(data_path=DATA_DIR, episode_start_range=episode_start_range,
              forecast_mode=FORECAST_MODE, **cfg, **ALPHAS)
    kw.update(CAP_OVERRIDES)
    return kw


class SpotHandler:
    """Saves progress on SIGTERM (spot 2-min warning) / SIGINT."""
    def __init__(self):
        self.interrupted = False
        signal.signal(signal.SIGTERM, self._h)
        signal.signal(signal.SIGINT, self._h)
    def _h(self, *_):
        print("\n  interruption signal -> will checkpoint and exit")
        self.interrupted = True


def _make_model(algo_name, env, seed):
    """Identical hyperparameters to the original run for a fair comparison."""
    if algo_name == "SAC":
        return SAC("MlpPolicy", env, learning_rate=3e-4, buffer_size=100000,
                   batch_size=256, tau=0.005, gamma=0.99, train_freq=1,
                   gradient_steps=1, ent_coef="auto", verbose=0, seed=seed)
    if algo_name == "PPO":
        return PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=2048, batch_size=64,
                   n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                   verbose=0, seed=seed)
    if algo_name == "TD3":
        return TD3("MlpPolicy", env, learning_rate=3e-4, buffer_size=100000,
                   batch_size=256, tau=0.005, gamma=0.99, verbose=0, seed=seed)
    if algo_name == "A2C":
        return A2C("MlpPolicy", env, learning_rate=7e-4, n_steps=5, gamma=0.99,
                   gae_lambda=0.95, verbose=0, seed=seed)
    raise ValueError(algo_name)


def _evaluate_test_window(model, config_name, test_range, n_episodes=None):
    """Evaluate on the HELD-OUT test window (>= split date).

    n_episodes defaults to the authoritative N_EVAL_EPISODES; a smaller value is
    used for the lightweight per-checkpoint progress evaluation.
    """
    n_episodes = N_EVAL_EPISODES if n_episodes is None else n_episodes
    cfg = SOURCE_CONFIGS[config_name]
    env = DataCenterEnergyEnv(**_env_kwargs(test_range, cfg))
    costs, carbons, waters, slas, rewards = [], [], [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=EVAL_SEED_BASE + ep)
        ep_r, done, info = 0.0, False, {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            ep_r += r
            done = term or trunc
        costs.append(info.get("episode_cost", 0.0))
        carbons.append(info.get("episode_carbon", 0.0))
        waters.append(info.get("episode_water", 0.0))
        slas.append(info.get("episode_sla_violations", 0))
        rewards.append(ep_r)
    return {
        "mean_reward": float(np.mean(rewards)), "std_reward": float(np.std(rewards)),
        "mean_episode_cost": float(np.mean(costs)), "std_episode_cost": float(np.std(costs)),
        "mean_episode_carbon": float(np.mean(carbons)),
        "mean_episode_water": float(np.mean(waters)),
        "mean_sla_violations": float(np.mean(slas)),
        "n_eval_episodes": n_episodes,
        "eval_window": "test", "eval_costs": costs,
    }


def train_one(algo, config, seed, ranges, handler):
    """Train on train_range, evaluate on test_range. Checkpoint/resume aware."""
    run_id = f"{algo}_{config}_seed{seed}"
    ckpt = os.path.join(CHECKPOINT_DIR, run_id)
    final = os.path.join(MODELS_DIR, run_id)
    cfg = SOURCE_CONFIGS[config]
    train_range = ranges[config]["train_range"]
    test_range = ranges[config]["test_range"]

    def _mk():
        e = DataCenterEnergyEnv(**_env_kwargs(train_range, cfg))
        return Monitor(e)
    env = DummyVecEnv([lambda: _mk() for _ in range(4)])

    rb_path = ckpt + "_rb.pkl"          # replay buffer (off-policy only)
    ckpt_log = ckpt + "_ckptlog.jsonl"  # durable per-checkpoint progress record

    remaining = TOTAL_TIMESTEPS
    if os.path.exists(ckpt + ".zip"):
        print(f"  resume {run_id}")
        model = ALGORITHMS[algo].load(ckpt, env=env)
        # Restore replay buffer so off-policy resume does NOT lose learned
        # transitions (SB3 model.save() excludes the buffer by default).
        if algo in OFF_POLICY and os.path.exists(rb_path):
            try:
                model.load_replay_buffer(rb_path)
                print(f"  restored replay buffer ({model.replay_buffer.size():,} transitions)")
            except Exception as e:
                print(f"  WARN: could not restore replay buffer: {e}")
        if os.path.exists(ckpt + "_meta.json"):
            with open(ckpt + "_meta.json") as f:
                remaining = TOTAL_TIMESTEPS - json.load(f).get("steps_completed", 0)
    else:
        model = _make_model(algo, env, seed)

    steps_done = TOTAL_TIMESTEPS - remaining
    while remaining > 0 and not handler.interrupted:
        chunk = min(CHECKPOINT_FREQ, remaining)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        steps_done += chunk
        remaining -= chunk

        # --- Durable checkpoint: model + step count ---
        model.save(ckpt)
        with open(ckpt + "_meta.json", "w") as f:
            json.dump({"steps_completed": steps_done}, f)
        # Off-policy: persist replay buffer so a resume continues losslessly.
        if algo in OFF_POLICY:
            try:
                model.save_replay_buffer(rb_path)
            except Exception as e:
                print(f"  WARN: could not save replay buffer: {e}")

        # --- Per-checkpoint held-out eval, appended to a durable JSONL so partial
        #     progress is recorded even if the run later dies. This is the
        #     "log results at each checkpoint" guarantee. n=20 here is a quick
        #     progress indicator; the FINAL authoritative eval below uses n=200. ---
        try:
            quick = _evaluate_test_window(model, config, test_range, n_episodes=N_CKPT_EVAL)
            rec = {"steps": steps_done, "mean_episode_cost": quick["mean_episode_cost"],
                   "std_episode_cost": quick["std_episode_cost"],
                   "mean_reward": quick["mean_reward"],
                   "n_eval_episodes": N_CKPT_EVAL,
                   "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            with open(ckpt_log, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"  [{run_id}] {steps_done:,}/{TOTAL_TIMESTEPS:,} "
                  f"heldout-cost=${quick['mean_episode_cost']:,.0f} (n={N_CKPT_EVAL})")
        except Exception as e:
            print(f"  WARN: checkpoint eval failed at {steps_done}: {e}")

    if handler.interrupted:
        return None

    # --- Final authoritative eval (full N_EVAL_EPISODES) ---
    model.save(final)
    result = _evaluate_test_window(model, config, test_range)
    result.update({"algorithm": algo, "config": config, "seed": seed, "run_id": run_id})
    # Preserve the per-checkpoint learning curve alongside the final result.
    if os.path.exists(ckpt_log):
        try:
            with open(ckpt_log) as f:
                result["checkpoint_curve"] = [json.loads(l) for l in f if l.strip()]
        except Exception:
            pass
    # Clean up the in-flight checkpoint artifacts (run is durably recorded now).
    for ext in [".zip", "_meta.json", "_rb.pkl", "_ckptlog.jsonl"]:
        p = ckpt + ext
        if os.path.exists(p):
            os.remove(p)
    return result


def main():
    # Declared global up-front: main() may override these for smoke tests, and
    # Python forbids referencing a name (even as an argparse default) before its
    # global declaration in the same function.
    global TOTAL_TIMESTEPS, N_EVAL_EPISODES, CHECKPOINT_FREQ, MODELS_DIR, CHECKPOINT_DIR, RESULTS_DIR
    global FORECAST_MODE, CAP_OVERRIDES

    ap = argparse.ArgumentParser()
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--total-workers", type=int, default=1)
    ap.add_argument("--algos", nargs="*", default=["PPO"])
    ap.add_argument("--configs", nargs="*", default=list(SOURCE_CONFIGS.keys()),
                    help="subset of source configs (smoke tests only)")
    ap.add_argument("--n-seeds", type=int, default=5,
                    help="number of seeds from SEEDS_MASTER (default 5 = original main-run "
                         "seeds; use 20 for the higher-power Pareto sweep, audit #12).")
    ap.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS,
                    help="override total timesteps (smoke tests only)")
    ap.add_argument("--eval-episodes", type=int, default=N_EVAL_EPISODES,
                    help="override eval episodes (smoke tests only)")
    ap.add_argument("--checkpoint-freq", type=int, default=CHECKPOINT_FREQ,
                    help="override checkpoint frequency in steps (smoke tests only)")
    ap.add_argument("--tag", default="",
                    help="output tag; isolates Pareto-sweep artifacts from the main run")
    ap.add_argument("--alpha-cost", type=float, default=ALPHAS["alpha_cost"])
    ap.add_argument("--alpha-carbon", type=float, default=ALPHAS["alpha_carbon"])
    ap.add_argument("--alpha-water", type=float, default=ALPHAS["alpha_water"])
    ap.add_argument("--alpha-sla", type=float, default=ALPHAS["alpha_sla"])
    ap.add_argument("--forecast-mode", choices=["persistence", "oracle", "provided"],
                    default=FORECAST_MODE,
                    help="price-forecast mode fed to the agent. 'persistence' = leakage-free "
                         "default (main verdict); 'oracle' = perfect-foresight ablation "
                         "(Gate 3, MUST be run with its own --tag).")
    ap.add_argument("--battery-capacity-kwh", type=float, default=None,
                    help="Gate 4 sizing override; None keeps env default (20000).")
    ap.add_argument("--battery-max-rate-kw", type=float, default=None,
                    help="Gate 4 sizing override; None keeps env default (10000).")
    args = ap.parse_args()

    # Allow smoke-test overrides without editing the rigorous defaults.
    TOTAL_TIMESTEPS = args.timesteps
    N_EVAL_EPISODES = args.eval_episodes
    CHECKPOINT_FREQ = args.checkpoint_freq

    # Objective weights for this run (Pareto sweep overrides these per weight-point).
    ALPHAS["alpha_cost"] = args.alpha_cost
    ALPHAS["alpha_carbon"] = args.alpha_carbon
    ALPHAS["alpha_water"] = args.alpha_water
    ALPHAS["alpha_sla"] = args.alpha_sla

    # Forecast mode (Gate 3 oracle ablation) + optional sizing overrides (Gate 4).
    FORECAST_MODE = args.forecast_mode
    if args.battery_capacity_kwh is not None:
        CAP_OVERRIDES["battery_capacity_kwh"] = args.battery_capacity_kwh
    if args.battery_max_rate_kw is not None:
        CAP_OVERRIDES["battery_max_rate_kw"] = args.battery_max_rate_kw
    if FORECAST_MODE != "persistence" and not args.tag:
        print("REFUSING: non-persistence forecast_mode must use a --tag so it cannot pool "
              "into the main verdict.", file=sys.stderr)
        sys.exit(2)
    print(f"  forecast_mode={FORECAST_MODE} cap_overrides={CAP_OVERRIDES or 'none'} tag='{args.tag}'")

    # A non-empty --tag isolates all artifacts (models/checkpoints/results/progress) so a
    # Pareto weight-point never collides with or skips against the main run's outputs.
    if args.tag:
        MODELS_DIR = MODELS_DIR + "_" + args.tag
        CHECKPOINT_DIR = CHECKPOINT_DIR + "_" + args.tag
        RESULTS_DIR = RESULTS_DIR  # results filenames get the tag instead (see below)
        for d in [MODELS_DIR, CHECKPOINT_DIR]:
            os.makedirs(d, exist_ok=True)

    # Precompute split ranges per config (shared timestamp axis, but compute each
    # to be safe against any config-specific row filtering).
    ranges = {c: make_split_ranges(DATA_DIR, SOURCE_CONFIGS[c], SPLIT_DATE)
              for c in SOURCE_CONFIGS}
    sample = ranges["all_sources"]
    print(f"  SPLIT {SPLIT_DATE}: train {sample['train_range']} test {sample['test_range']} "
          f"| data {sample['first_ts']} .. {sample['last_ts']}")

    seeds_used = SEEDS_MASTER[:args.n_seeds]
    print(f"  seeds ({len(seeds_used)}): {seeds_used}")
    runs = [{"algo": a, "config": c, "seed": s}
            for a in args.algos for c in args.configs for s in seeds_used]
    my_runs = [r for i, r in enumerate(runs) if i % args.total_workers == args.worker_id]

    prog_file = os.path.join(CHECKPOINT_DIR, f"progress_temporal_worker_{args.worker_id}.json")
    progress = json.load(open(prog_file)) if os.path.exists(prog_file) else {"completed": [], "results": []}

    handler = SpotHandler()
    _tagpart = f"_{args.tag}" if args.tag else ""
    results_file = os.path.join(RESULTS_DIR, f"rl_results_temporal{_tagpart}_worker_{args.worker_id}.json")
    print(f"  worker {args.worker_id}/{args.total_workers}: {len(my_runs)} runs, "
          f"{len(progress['completed'])} already done")

    for run in my_runs:
        if handler.interrupted:
            break
        rid = f"{run['algo']}_{run['config']}_seed{run['seed']}"
        if rid in progress["completed"]:
            continue
        try:
            res = train_one(run["algo"], run["config"], run["seed"], ranges, handler)
            if res is None:
                break
            progress["completed"].append(rid)
            progress["results"].append(res)
            json.dump(progress, open(prog_file, "w"), indent=2)
            json.dump(progress["results"], open(results_file, "w"), indent=2)
            print(f"  DONE {rid}: test-cost ${res['mean_episode_cost']:,.0f} "
                  f"± ${res['std_episode_cost']:,.0f}")
        except Exception as e:
            print(f"  FAILED {rid}: {e}")
            traceback.print_exc()
            progress["completed"].append(rid)
            progress["results"].append({"run_id": rid, "error": str(e)})
            json.dump(progress, open(prog_file, "w"), indent=2)

    print(f"  worker {args.worker_id} finished: {len(progress['completed'])} total")


if __name__ == "__main__":
    main()
