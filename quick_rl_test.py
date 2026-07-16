"""Quick 10K-step SAC test to verify pipeline works."""
import sys
sys.path.insert(0, "src")
import time
import numpy as np
from dc_energy_env import DataCenterEnergyEnv
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

env = DataCenterEnergyEnv(data_path="data")
env = Monitor(env)

print("Training SAC for 10K steps (quick test)...")
start = time.time()
model = SAC("MlpPolicy", env, learning_rate=3e-4, verbose=0, seed=42)
model.learn(total_timesteps=10000)
elapsed = time.time() - start
print(f"Done in {elapsed:.0f}s")

# Quick eval
eval_env = DataCenterEnergyEnv(data_path="data")
rewards = []
costs = []
for ep in range(5):
    obs, _ = eval_env.reset(seed=2000 + ep)
    r_total = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        r_total += reward
        done = terminated or truncated
    rewards.append(r_total)
    costs.append(info.get("episode_cost", 0))

print(f"Eval (5 eps): reward={np.mean(rewards):.2f}, cost=${np.mean(costs):,.0f}")
print("✓ PIPELINE WORKS" if np.mean(costs) > 0 else "✗ ERROR")
