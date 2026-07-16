"""Debug the environment reward signal."""
import sys
sys.path.insert(0, "src")
import numpy as np
from dc_energy_env import DataCenterEnergyEnv

env = DataCenterEnergyEnv(data_path="data")
obs, _ = env.reset(seed=42)

print("Smart heuristic - first 10 steps:")
for step in range(10):
    price_signal = obs[10]
    action = np.array([0.8, 0.3, 0.7 if price_signal > 0 else -0.5, 0.8 if price_signal > 0.3 else 0.0])
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"  Step {step}: reward={reward:.3f} | cost=${info['hour_cost']:.0f} | "
          f"water={info['hour_water']:.1f}m3 | grid={info['grid_used']:.0f}kW | "
          f"batt_d={info['battery_discharged']:.0f} | gas={info['gas_used']:.0f} | soc={info['battery_soc']:.2f}")

print("\nDo-nothing - first 10 steps:")
obs, _ = env.reset(seed=42)
for step in range(10):
    action = np.array([0.0, 0.0, 0.0, 0.0])
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"  Step {step}: reward={reward:.3f} | cost=${info['hour_cost']:.0f} | "
          f"water={info['hour_water']:.1f}m3 | grid={info['grid_used']:.0f}kW")

print("\n\nThe issue: check if water penalty dominates when cooling_offset > 0")
print("Also check: does deferral cost MORE because deferred load is served at bad times?")
