"""Debug: print reward component magnitudes to find the imbalance."""
import sys
sys.path.insert(0, "src")
import numpy as np
from dc_energy_env import DataCenterEnergyEnv

env = DataCenterEnergyEnv(data_path="data")

# Run do-nothing and collect component values
obs, _ = env.reset(seed=42)
costs, carbons, waters = [], [], []
for step in range(168):
    action = np.array([0.0, 0.0, 0.0, 0.0])
    obs, reward, terminated, truncated, info = env.step(action)
    costs.append(info["hour_cost"])
    carbons.append(info["hour_carbon"])
    waters.append(info["hour_water"])

costs = np.array(costs)
carbons = np.array(carbons)
waters = np.array(waters)

print("RAW COMPONENT MAGNITUDES (per hour):")
print(f"  Cost:   mean=${costs.mean():.0f}, range ${costs.min():.0f}-${costs.max():.0f}")
print(f"  Carbon: mean={carbons.mean():.0f}kg, range {carbons.min():.0f}-{carbons.max():.0f}")
print(f"  Water:  mean={waters.mean():.1f}m3, range {waters.min():.1f}-{waters.max():.1f}")

# What the normalization does
demand_mean = env.demand_mean
price_mean = env.price_mean
carbon_mean = env.grid_carbon.mean()

cost_norm = costs / (demand_mean * price_mean)
carbon_norm = carbons / (demand_mean * carbon_mean)
water_norm = waters / 5.0

print(f"\nNORMALIZED COMPONENTS (what the reward sees):")
print(f"  Cost norm:   mean={cost_norm.mean():.3f}, range {cost_norm.min():.3f}-{cost_norm.max():.3f}")
print(f"  Carbon norm: mean={carbon_norm.mean():.3f}, range {carbon_norm.min():.3f}-{carbon_norm.max():.3f}")
print(f"  Water norm:  mean={water_norm.mean():.3f}, range {water_norm.min():.3f}-{water_norm.max():.3f}")

print(f"\nWEIGHTED (alpha_cost=0.4, alpha_carbon=0.3, alpha_water=0.2):")
print(f"  Cost contribution:   {0.4 * cost_norm.mean():.4f}")
print(f"  Carbon contribution: {0.3 * carbon_norm.mean():.4f}")
print(f"  Water contribution:  {0.2 * water_norm.mean():.4f}")
print(f"  → Water is {0.2 * water_norm.mean() / (0.4 * cost_norm.mean()) * 100:.0f}% of cost contribution")
