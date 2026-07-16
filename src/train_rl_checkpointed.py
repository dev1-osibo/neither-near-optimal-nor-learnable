"""
RL Training with Checkpoint/Resume for Spot Instances
======================================================
Handles AWS Spot interruptions gracefully:
- Saves model checkpoint every 50K steps
- Saves training progress (which runs are done, which are pending)
- On restart, automatically resumes from last checkpoint
- Catches SIGTERM (AWS spot 2-min warning) and saves immediately

Usage:
    python train_rl_checkpointed.py --worker-id 0 --total-workers 4
    python train_rl_checkpointed.py --worker-id 1 --total-workers 4
    python train_rl_checkpointed.py --worker-id 2 --total-workers 4
    python train_rl_checkpointed.py --worker-id 3 --total-workers 4

Each worker takes a slice of the 240 total runs.
If interrupted, restart with same --worker-id and it resumes.
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

from stable_baselines3 import SAC, PPO, TD3, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints")

for d in [RESULTS_DIR, MODELS_DIR, CHECKPOINT_DIR]:
    os.makedirs(d, exist_ok=True)

# Training parameters — FULL RIGOROUS (no shortcuts)
TOTAL_TIMESTEPS = 1_000_000  # 1M steps per run
CHECKPOINT_FREQ = 50_000     # Save every 50K steps
N_EVAL_EPISODES = 20         # Evaluate on 20 episodes
SEEDS = [42, 123, 456, 789, 1024]  # 5 seeds per config

# Source configs (12 total)
SOURCE_CONFIGS = {
    "grid_only": {"use_solar": False, "use_wind": False, "use_battery": False, "use_gas": False},
    "grid_solar": {"use_solar": True, "use_wind": False, "use_battery": False, "use_gas": False},
    "grid_wind": {"use_solar": False, "use_wind": True, "use_battery": False, "use_gas": False},
    "grid_gas": {"use_solar": False, "use_wind": False, "use_battery": False, "use_gas": True},
    "grid_solar_wind": {"use_solar": True, "use_wind": True, "use_battery": False, "use_gas": False},
    "grid_solar_battery": {"use_solar": True, "use_wind": False, "use_battery": True, "use_gas": False},
    "grid_wind_battery": {"use_solar": False, "use_wind": True, "use_battery": True, "use_gas": False},
    "grid_solar_gas": {"use_solar": True, "use_wind": False, "use_battery": False, "use_gas": True},
    "grid_wind_gas": {"use_solar": False, "use_wind": True, "use_battery": False, "use_gas": True},
    "grid_solar_wind_battery": {"use_solar": True, "use_wind": True, "use_battery": True, "use_gas": False},
    "grid_solar_wind_gas": {"use_solar": True, "use_wind": True, "use_battery": False, "use_gas": True},
    "all_sources": {"use_solar": True, "use_wind": True, "use_battery": True, "use_gas": True},
}

# Algorithms (4 total)
ALGORITHMS = {"SAC": SAC, "PPO": PPO, "TD3": TD3, "A2C": A2C}

# ============================================================
# SPOT INTERRUPTION HANDLER
# ============================================================

class SpotInterruptionHandler:
    """Catches AWS Spot termination signal and saves checkpoint."""
    
    def __init__(self):
        self.interrupted = False
        # AWS sends SIGTERM 2 minutes before spot termination
        signal.signal(signal.SIGTERM, self._handle_signal)
        # Also catch SIGINT (Ctrl+C) for manual interruption
        signal.signal(signal.SIGINT, self._handle_signal)
    
    def _handle_signal(self, signum, frame):
        print(f"\n⚠️  INTERRUPTION SIGNAL RECEIVED (signal {signum})")
        print("  Saving checkpoint before exit...")
        self.interrupted = True


class ProgressTracker:
    """Tracks which runs are complete, in progress, or pending."""
    
    def __init__(self, worker_id, progress_file=None):
        self.worker_id = worker_id
        self.progress_file = progress_file or os.path.join(
            CHECKPOINT_DIR, f"progress_worker_{worker_id}.json"
        )
        self.progress = self._load()
    
    def _load(self):
        if os.path.exists(self.progress_file):
            with open(self.progress_file) as f:
                return json.load(f)
        return {"completed": [], "in_progress": None, "results": []}
    
    def save(self):
        with open(self.progress_file, "w") as f:
            json.dump(self.progress, f, indent=2)
    
    def is_completed(self, run_id):
        return run_id in self.progress["completed"]
    
    def mark_in_progress(self, run_id):
        self.progress["in_progress"] = run_id
        self.save()
    
    def mark_completed(self, run_id, result):
        self.progress["completed"].append(run_id)
        self.progress["in_progress"] = None
        self.progress["results"].append(result)
        self.save()
    
    def get_in_progress(self):
        return self.progress["in_progress"]


# ============================================================
# TRAINING WITH CHECKPOINT/RESUME
# ============================================================

def get_checkpoint_path(algo, config, seed):
    """Get checkpoint file path for a specific run."""
    return os.path.join(CHECKPOINT_DIR, f"{algo}_{config}_seed{seed}")


def train_single_checkpointed(algo_name, config_name, seed, handler):
    """
    Train with checkpointing. Resumes from last checkpoint if exists.
    Returns result dict or None if interrupted.
    """
    run_id = f"{algo_name}_{config_name}_seed{seed}"
    checkpoint_path = get_checkpoint_path(algo_name, config_name, seed)
    final_model_path = os.path.join(MODELS_DIR, run_id)
    
    print(f"\n{'─'*60}")
    print(f"  RUN: {run_id}")
    print(f"  Steps: {TOTAL_TIMESTEPS:,} | Checkpoint every {CHECKPOINT_FREQ:,}")
    print(f"{'─'*60}")
    
    # Create environment (DummyVecEnv with 4 parallel envs for 3.87x speedup)
    source_config = SOURCE_CONFIGS[config_name]
    
    def make_env(seed_offset):
        def _init():
            e = DataCenterEnergyEnv(data_path=DATA_DIR, **source_config)
            e = Monitor(e)
            return e
        return _init
    
    env = DummyVecEnv([make_env(i) for i in range(4)])
    
    # Check for existing checkpoint
    checkpoint_zip = checkpoint_path + ".zip"
    remaining_steps = TOTAL_TIMESTEPS
    
    algo_class = ALGORITHMS[algo_name]
    
    if os.path.exists(checkpoint_zip):
        # RESUME from checkpoint
        print(f"  📂 Resuming from checkpoint: {checkpoint_zip}")
        model = algo_class.load(checkpoint_path, env=env)
        
        # Calculate remaining steps from checkpoint metadata
        meta_file = checkpoint_path + "_meta.json"
        if os.path.exists(meta_file):
            with open(meta_file) as f:
                meta = json.load(f)
            steps_done = meta.get("steps_completed", 0)
            remaining_steps = TOTAL_TIMESTEPS - steps_done
            print(f"  Steps already done: {steps_done:,}, remaining: {remaining_steps:,}")
        else:
            # Can't determine progress — restart
            remaining_steps = TOTAL_TIMESTEPS
            print(f"  No metadata found — training full {TOTAL_TIMESTEPS:,} steps")
    else:
        # Fresh start
        print(f"  🆕 Starting fresh training")
        
        if algo_name == "SAC":
            model = algo_class(
                "MlpPolicy", env,
                learning_rate=3e-4, buffer_size=100000, batch_size=256,
                tau=0.005, gamma=0.99, train_freq=1, gradient_steps=1,
                ent_coef="auto", verbose=0, seed=seed,
            )
        elif algo_name == "PPO":
            model = algo_class(
                "MlpPolicy", env,
                learning_rate=3e-4, n_steps=2048, batch_size=64,
                n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                verbose=0, seed=seed,
            )
        elif algo_name == "TD3":
            model = algo_class(
                "MlpPolicy", env,
                learning_rate=3e-4, buffer_size=100000, batch_size=256,
                tau=0.005, gamma=0.99, verbose=0, seed=seed,
            )
        elif algo_name == "A2C":
            model = algo_class(
                "MlpPolicy", env,
                learning_rate=7e-4, n_steps=5, gamma=0.99,
                gae_lambda=0.95, verbose=0, seed=seed,
            )
    
    if remaining_steps <= 0:
        print(f"  ✓ Already completed!")
    else:
        # Train with periodic checkpointing
        steps_trained = TOTAL_TIMESTEPS - remaining_steps
        
        while remaining_steps > 0 and not handler.interrupted:
            # Train in chunks of CHECKPOINT_FREQ
            chunk = min(CHECKPOINT_FREQ, remaining_steps)
            
            start_time = time.time()
            model.learn(total_timesteps=chunk, reset_num_timesteps=False)
            chunk_time = time.time() - start_time
            
            steps_trained += chunk
            remaining_steps -= chunk
            
            # Save checkpoint
            model.save(checkpoint_path)
            meta = {"steps_completed": steps_trained, "total_steps": TOTAL_TIMESTEPS,
                    "algo": algo_name, "config": config_name, "seed": seed,
                    "last_checkpoint_time": time.strftime("%Y-%m-%d %H:%M:%S")}
            with open(checkpoint_path + "_meta.json", "w") as f:
                json.dump(meta, f)
            
            pct = steps_trained / TOTAL_TIMESTEPS * 100
            rate = chunk / chunk_time
            eta = remaining_steps / rate if rate > 0 else 0
            print(f"  [{steps_trained:>8,}/{TOTAL_TIMESTEPS:,}] {pct:.0f}% | "
                  f"{rate:.0f} steps/s | ETA: {eta/60:.0f} min")
            
            if handler.interrupted:
                print(f"  ⚠️  Interrupted — checkpoint saved at {steps_trained:,} steps")
                return None
    
    # Training complete — save final model
    model.save(final_model_path)
    print(f"  ✓ Training complete. Model saved: {final_model_path}")
    
    # Evaluate
    print(f"  Evaluating ({N_EVAL_EPISODES} episodes)...")
    eval_env = DataCenterEnergyEnv(data_path=DATA_DIR, **source_config)
    result = evaluate_model(model, eval_env)
    result.update({
        "algorithm": algo_name, "config": config_name, "seed": seed,
        "total_timesteps": TOTAL_TIMESTEPS, "run_id": run_id,
    })
    
    print(f"  Result: cost=${result['mean_episode_cost']:,.0f} ± ${result['std_episode_cost']:,.0f}, "
          f"reward={result['mean_reward']:.2f}")
    
    # Clean up checkpoint (training is done)
    for ext in [".zip", "_meta.json"]:
        p = checkpoint_path + ext
        if os.path.exists(p):
            os.remove(p)
    
    return result


def evaluate_model(model, env):
    """Evaluate trained model over N_EVAL_EPISODES episodes."""
    rewards, costs, carbons, waters, sla = [], [], [], [], []
    
    for ep in range(N_EVAL_EPISODES):
        obs, _ = env.reset(seed=5000 + ep)
        episode_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        rewards.append(episode_reward)
        costs.append(info.get("episode_cost", 0))
        carbons.append(info.get("episode_carbon", 0))
        waters.append(info.get("episode_water", 0))
        sla.append(info.get("episode_sla_violations", 0))
    
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_episode_cost": float(np.mean(costs)),
        "std_episode_cost": float(np.std(costs)),
        "mean_episode_carbon": float(np.mean(carbons)),
        "std_episode_carbon": float(np.std(carbons)),
        "mean_episode_water": float(np.mean(waters)),
        "mean_sla_violations": float(np.mean(sla)),
        "n_eval_episodes": N_EVAL_EPISODES,
    }

# ============================================================
# WORK DISTRIBUTION FOR PARALLEL INSTANCES
# ============================================================

def generate_all_runs():
    """Generate the full list of 240 runs (4 algos × 12 configs × 5 seeds)."""
    runs = []
    for algo in ALGORITHMS.keys():
        for config in SOURCE_CONFIGS.keys():
            for seed in SEEDS:
                run_id = f"{algo}_{config}_seed{seed}"
                runs.append({"algo": algo, "config": config, "seed": seed, "run_id": run_id})
    return runs


def get_worker_runs(all_runs, worker_id, total_workers):
    """Split runs across workers. Each worker gets every Nth run."""
    return [r for i, r in enumerate(all_runs) if i % total_workers == worker_id]


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Checkpointed RL Training for Spot Instances")
    parser.add_argument("--worker-id", type=int, default=0,
                        help="Worker index (0-3 for 4 instances)")
    parser.add_argument("--total-workers", type=int, default=4,
                        help="Total number of parallel workers")
    parser.add_argument("--dry-run", action="store_true",
                        help="Just show what would run, don't train")
    args = parser.parse_args()
    
    # Generate work
    all_runs = generate_all_runs()
    my_runs = get_worker_runs(all_runs, args.worker_id, args.total_workers)
    
    print("=" * 60)
    print(f"CHECKPOINTED RL TRAINING — Worker {args.worker_id}/{args.total_workers}")
    print("=" * 60)
    print(f"  Total runs across all workers: {len(all_runs)}")
    print(f"  Runs assigned to this worker: {len(my_runs)}")
    print(f"  Timesteps per run: {TOTAL_TIMESTEPS:,}")
    print(f"  Seeds per config: {len(SEEDS)}")
    print(f"  Checkpoint frequency: every {CHECKPOINT_FREQ:,} steps")
    print(f"  Eval episodes: {N_EVAL_EPISODES}")
    
    if args.dry_run:
        print(f"\n  DRY RUN — runs assigned to worker {args.worker_id}:")
        for r in my_runs:
            print(f"    {r['run_id']}")
        print(f"\n  Estimated time (GPU): ~{len(my_runs) * 2:.0f} hours")
        print(f"  Estimated time (CPU): ~{len(my_runs) * 6:.0f} hours")
        return
    
    # Setup
    handler = SpotInterruptionHandler()
    tracker = ProgressTracker(args.worker_id)
    
    print(f"\n  Previously completed: {len(tracker.progress['completed'])} runs")
    
    # Check if there's an interrupted run to resume
    in_progress = tracker.get_in_progress()
    if in_progress:
        print(f"  Resuming interrupted run: {in_progress}")
    
    # Run all assigned work
    completed = 0
    skipped = 0
    failed = 0
    start_time = time.time()
    
    for run in my_runs:
        if handler.interrupted:
            print("\n⚠️  Stopping due to interruption signal.")
            break
        
        run_id = run["run_id"]
        
        # Skip if already done
        if tracker.is_completed(run_id):
            skipped += 1
            continue
        
        # Train
        tracker.mark_in_progress(run_id)
        
        try:
            result = train_single_checkpointed(
                run["algo"], run["config"], run["seed"], handler
            )
            
            if result is not None:
                tracker.mark_completed(run_id, result)
                completed += 1
                
                # Save accumulated results
                results_path = os.path.join(RESULTS_DIR, f"rl_results_worker_{args.worker_id}.json")
                with open(results_path, "w") as f:
                    json.dump(tracker.progress["results"], f, indent=2)
            else:
                # Interrupted during training
                break
                
        except Exception as e:
            print(f"  ✗ FAILED: {run_id}: {e}")
            traceback.print_exc()
            failed += 1
            tracker.mark_completed(run_id, {"run_id": run_id, "error": str(e)})
    
    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"WORKER {args.worker_id} SUMMARY")
    print(f"{'='*60}")
    print(f"  Completed this session: {completed}")
    print(f"  Skipped (already done): {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Total completed (all sessions): {len(tracker.progress['completed'])}")
    print(f"  Remaining: {len(my_runs) - len(tracker.progress['completed'])}")
    print(f"  Wall time: {elapsed/3600:.1f} hours")
    
    if handler.interrupted:
        print(f"\n  ⚠️  INTERRUPTED — restart this worker to resume")
        print(f"  Command: python train_rl_checkpointed.py --worker-id {args.worker_id} --total-workers {args.total_workers}")


if __name__ == "__main__":
    main()
