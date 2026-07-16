"""
RL Training Script — Multi-Source DC Energy Optimization
=========================================================
Trains reinforcement learning agents on the DataCenterEnergyEnv.

Algorithms: SAC, PPO, TD3, A2C (from stable-baselines3)
Source combinations: 12 configurations
Training: 500K-1M timesteps per config

Usage:
    python train_rl.py --algo SAC --config all_sources --steps 500000
    python train_rl.py --algo PPO --config grid_solar_battery --steps 500000
    python train_rl.py --run-all  # Run all combinations (long!)
"""

import os
import sys
import json
import time
import argparse
import numpy as np

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))
from dc_energy_env import DataCenterEnergyEnv

from stable_baselines3 import SAC, PPO, TD3, A2C
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

# Results directory
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# Source combination configs
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

# Algorithm mapping
ALGORITHMS = {
    "SAC": SAC,
    "PPO": PPO,
    "TD3": TD3,
    "A2C": A2C,
}


def train_single(algo_name, config_name, total_timesteps=500000, seed=42):
    """
    Train a single RL agent on a specific source configuration.
    
    Args:
        algo_name: "SAC", "PPO", "TD3", or "A2C"
        config_name: Key from SOURCE_CONFIGS
        total_timesteps: Training steps
        seed: Random seed for reproducibility
    
    Returns:
        dict with training results
    """
    print(f"\n{'='*60}")
    print(f"TRAINING: {algo_name} on {config_name}")
    print(f"Steps: {total_timesteps:,} | Seed: {seed}")
    print(f"{'='*60}")
    
    # Create environment
    source_config = SOURCE_CONFIGS[config_name]
    env = DataCenterEnergyEnv(data_path=DATA_DIR, **source_config)
    env = Monitor(env)
    
    # Create eval environment (same config, different episodes)
    eval_env = DataCenterEnergyEnv(data_path=DATA_DIR, **source_config)
    eval_env = Monitor(eval_env)
    
    # Algorithm-specific hyperparameters (tuned for this domain)
    algo_class = ALGORITHMS[algo_name]
    
    if algo_name == "SAC":
        model = algo_class(
            "MlpPolicy", env,
            learning_rate=3e-4,
            buffer_size=100000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            verbose=0,
            seed=seed,
        )
    elif algo_name == "PPO":
        model = algo_class(
            "MlpPolicy", env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=0,
            seed=seed,
        )
    elif algo_name == "TD3":
        model = algo_class(
            "MlpPolicy", env,
            learning_rate=3e-4,
            buffer_size=100000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            verbose=0,
            seed=seed,
        )
    elif algo_name == "A2C":
        model = algo_class(
            "MlpPolicy", env,
            learning_rate=7e-4,
            n_steps=5,
            gamma=0.99,
            gae_lambda=0.95,
            verbose=0,
            seed=seed,
        )
    
    # Train
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps)
    train_time = time.time() - start_time
    
    # Save model
    model_path = os.path.join(MODELS_DIR, f"{algo_name}_{config_name}")
    model.save(model_path)
    print(f"  Model saved: {model_path}")
    
    # Evaluate: run 20 episodes and collect metrics
    print(f"  Evaluating (20 episodes)...")
    eval_results = evaluate_model(model, eval_env, n_episodes=20)
    
    result = {
        "algorithm": algo_name,
        "config": config_name,
        "sources": source_config,
        "total_timesteps": total_timesteps,
        "train_time_seconds": train_time,
        "seed": seed,
        **eval_results,
    }
    
    print(f"  Results:")
    print(f"    Avg episode cost: ${eval_results['mean_episode_cost']:,.0f}")
    print(f"    Avg episode reward: {eval_results['mean_reward']:.2f}")
    print(f"    Training time: {train_time:.0f}s")
    
    return result


def evaluate_model(model, env, n_episodes=20):
    """Evaluate a trained model over multiple episodes."""
    rewards = []
    costs = []
    carbons = []
    waters = []
    sla_violations = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=1000 + ep)
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
        sla_violations.append(info.get("episode_sla_violations", 0))
    
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_episode_cost": float(np.mean(costs)),
        "std_episode_cost": float(np.std(costs)),
        "mean_episode_carbon": float(np.mean(carbons)),
        "mean_episode_water": float(np.mean(waters)),
        "mean_sla_violations": float(np.mean(sla_violations)),
        "n_episodes": n_episodes,
    }


def run_all(total_timesteps=500000):
    """Run all algorithm × config combinations."""
    all_results = []
    
    # Start with key combinations (not all 48 — that would take days)
    priority_runs = [
        # Core algorithms on full system
        ("SAC", "all_sources"),
        ("PPO", "all_sources"),
        ("TD3", "all_sources"),
        # SAC across source tiers (to show scaling)
        ("SAC", "grid_only"),
        ("SAC", "grid_solar"),
        ("SAC", "grid_solar_battery"),
        ("SAC", "grid_solar_wind_battery"),
        # PPO for comparison on key configs
        ("PPO", "grid_only"),
        ("PPO", "grid_solar_wind_battery"),
    ]
    
    for algo, config in priority_runs:
        try:
            result = train_single(algo, config, total_timesteps=total_timesteps)
            all_results.append(result)
            
            # Save incrementally
            outpath = os.path.join(RESULTS_DIR, "rl_training_results.json")
            with open(outpath, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"  → Results saved ({len(all_results)} runs complete)")
            
        except Exception as e:
            print(f"  ✗ FAILED: {algo}/{config}: {e}")
            all_results.append({
                "algorithm": algo, "config": config, "error": str(e)
            })
    
    # Final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE — SUMMARY")
    print("=" * 60)
    
    print(f"\n  {'Algorithm':<8} | {'Config':<30} | {'Avg Cost':>10} | {'Reward':>8}")
    print(f"  {'-'*8} | {'-'*30} | {'-'*10} | {'-'*8}")
    
    for r in all_results:
        if "error" not in r:
            print(f"  {r['algorithm']:<8} | {r['config']:<30} | "
                  f"${r['mean_episode_cost']:>8,.0f} | {r['mean_reward']:>7.1f}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Train RL agents for DC energy optimization")
    parser.add_argument("--algo", choices=list(ALGORITHMS.keys()), default="SAC",
                        help="RL algorithm")
    parser.add_argument("--config", choices=list(SOURCE_CONFIGS.keys()), default="all_sources",
                        help="Source configuration")
    parser.add_argument("--steps", type=int, default=500000,
                        help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--run-all", action="store_true",
                        help="Run all priority combinations")
    args = parser.parse_args()
    
    if args.run_all:
        run_all(total_timesteps=args.steps)
    else:
        result = train_single(args.algo, args.config, args.steps, args.seed)
        
        # Save individual result
        outpath = os.path.join(RESULTS_DIR, f"rl_{args.algo}_{args.config}_result.json")
        with open(outpath, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  ✓ Saved: {outpath}")


if __name__ == "__main__":
    main()
