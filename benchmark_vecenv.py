"""
Benchmark: Vectorized env vs single env speed.
Measures ACTUAL steps/second for both approaches.
"""
import sys
import time
import numpy as np
sys.path.insert(0, "src")
from dc_energy_env import DataCenterEnergyEnv
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

DATA_PATH = "data"
TEST_STEPS = 10000  # 10K steps for each benchmark

def make_env(seed):
    def _init():
        env = DataCenterEnergyEnv(data_path=DATA_PATH)
        env = Monitor(env)
        return env
    return _init

print("=" * 60)
print("BENCHMARK: Single env vs Vectorized env")
print("=" * 60)

# Benchmark 1: Single environment (baseline — what we measured before)
print("\n[1] Single environment (1 env, sequential)...")
env_single = DataCenterEnergyEnv(data_path=DATA_PATH)
env_single = Monitor(env_single)

model_single = SAC("MlpPolicy", env_single, verbose=0, seed=42)
start = time.time()
model_single.learn(total_timesteps=TEST_STEPS)
elapsed_single = time.time() - start
rate_single = TEST_STEPS / elapsed_single
print(f"  Time: {elapsed_single:.1f}s | Rate: {rate_single:.0f} steps/sec")

# Benchmark 2: DummyVecEnv with 4 envs (same process, no multiprocessing)
print("\n[2] DummyVecEnv (4 envs, same process)...")
vec_env_4 = DummyVecEnv([make_env(i) for i in range(4)])

model_vec4 = SAC("MlpPolicy", vec_env_4, verbose=0, seed=42)
start = time.time()
model_vec4.learn(total_timesteps=TEST_STEPS)
elapsed_vec4 = time.time() - start
rate_vec4 = TEST_STEPS / elapsed_vec4
print(f"  Time: {elapsed_vec4:.1f}s | Rate: {rate_vec4:.0f} steps/sec")
print(f"  Speedup vs single: {rate_vec4/rate_single:.2f}x")

vec_env_4.close()

# Benchmark 3: SubprocVecEnv with 4 envs (multiprocessing — uses multiple cores)
print("\n[3] SubprocVecEnv (4 envs, 4 separate processes)...")
vec_env_sub4 = SubprocVecEnv([make_env(i) for i in range(4)])

model_sub4 = SAC("MlpPolicy", vec_env_sub4, verbose=0, seed=42)
start = time.time()
model_sub4.learn(total_timesteps=TEST_STEPS)
elapsed_sub4 = time.time() - start
rate_sub4 = TEST_STEPS / elapsed_sub4
print(f"  Time: {elapsed_sub4:.1f}s | Rate: {rate_sub4:.0f} steps/sec")
print(f"  Speedup vs single: {rate_sub4/rate_single:.2f}x")

vec_env_sub4.close()

# Benchmark 4: SubprocVecEnv with 8 envs
print("\n[4] SubprocVecEnv (8 envs, 8 separate processes)...")
vec_env_sub8 = SubprocVecEnv([make_env(i) for i in range(8)])

model_sub8 = SAC("MlpPolicy", vec_env_sub8, verbose=0, seed=42)
start = time.time()
model_sub8.learn(total_timesteps=TEST_STEPS)
elapsed_sub8 = time.time() - start
rate_sub8 = TEST_STEPS / elapsed_sub8
print(f"  Time: {elapsed_sub8:.1f}s | Rate: {rate_sub8:.0f} steps/sec")
print(f"  Speedup vs single: {rate_sub8/rate_single:.2f}x")

vec_env_sub8.close()

# Summary
print(f"\n{'='*60}")
print("BENCHMARK RESULTS")
print(f"{'='*60}")
print(f"  Single env:        {rate_single:.0f} steps/sec (baseline)")
print(f"  DummyVecEnv(4):    {rate_vec4:.0f} steps/sec ({rate_vec4/rate_single:.2f}x)")
print(f"  SubprocVecEnv(4):  {rate_sub4:.0f} steps/sec ({rate_sub4/rate_single:.2f}x)")
print(f"  SubprocVecEnv(8):  {rate_sub8:.0f} steps/sec ({rate_sub8/rate_single:.2f}x)")

# Project full training time
full_runs = 240
steps_per_run = 1_000_000
eval_overhead = 0.1  # 10% overhead for evaluation

for label, rate in [("Single", rate_single), ("Best vectorized", max(rate_vec4, rate_sub4, rate_sub8))]:
    time_per_run = steps_per_run / rate * (1 + eval_overhead)
    total_time = full_runs * time_per_run
    print(f"\n  Full 240 runs at {label} speed:")
    print(f"    Per run: {time_per_run/3600:.1f} hours")
    print(f"    Sequential (1 at a time): {total_time/3600:.0f} hours ({total_time/86400:.1f} days)")
    print(f"    On 8-core (1 concurrent run): {total_time/3600:.0f} hours")
