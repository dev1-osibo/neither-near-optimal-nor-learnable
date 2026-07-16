"""Final environment verification: run multiple episodes to show consistency."""
import sys
sys.path.insert(0, "src")
import numpy as np
from dc_energy_env import DataCenterEnergyEnv

env = DataCenterEnergyEnv(data_path="data")

def run_episode(env, policy, seed):
    obs, _ = env.reset(seed=seed)
    total_reward = 0
    total_cost = 0
    for step in range(168):
        if policy == "random":
            action = env.action_space.sample()
        elif policy == "do_nothing":
            action = np.array([0.0, 0.0, 0.0, 0.0])
        elif policy == "smart":
            price_signal = obs[10]
            action = np.array([
                0.5,
                0.0,
                0.8 if price_signal > 0.2 else (-0.6 if price_signal < -0.2 else 0.0),
                0.9 if price_signal > 0.5 else 0.0,
            ])
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated:
            total_cost = info.get("episode_cost", 0)
            break
    return total_reward, total_cost

# Run 20 episodes each
N = 20
seeds = list(range(100, 100 + N))

print("=" * 60)
print("ENVIRONMENT VERIFICATION — 20 episodes each")
print("=" * 60)

for policy in ["do_nothing", "random", "smart"]:
    rewards = []
    costs = []
    for seed in seeds:
        r, c = run_episode(env, policy, seed)
        rewards.append(r)
        costs.append(c)
    
    rewards = np.array(rewards)
    costs = np.array(costs)
    print(f"\n  {policy:12s}: reward={rewards.mean():.2f} ± {rewards.std():.2f} | "
          f"cost=${costs.mean():,.0f} ± ${costs.std():,.0f}")

print("\n  EXPECTED ORDERING (by cost — lower is better):")
print("    smart < random < do_nothing")
print("\n  If smart has LOWEST average cost across 20 episodes → PASS")
