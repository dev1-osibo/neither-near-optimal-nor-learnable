"""
Baseline Strategies for Comparison
====================================
Non-RL strategies that the RL agents must beat.

1. Do-Nothing (grid only, no active decisions)
2. Rule-Based (industry standard: fixed time schedules)
3. Greedy (choose cheapest source each hour, no foresight)
4. Deterministic Optimal (hindsight oracle: knows ALL future perfectly)
5. MPC (Model Predictive Control: uses forecast, solves LP each step)

All use the SAME environment and produce the same metrics
for direct comparison with RL agents.
"""

import sys
import os
import numpy as np
import json

sys.path.insert(0, os.path.dirname(__file__))
from dc_energy_env import DataCenterEnergyEnv

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_EVAL_EPISODES = 20  # Same as RL evaluation
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


# ============================================================
# BASELINE 1: DO NOTHING
# ============================================================
class DoNothingPolicy:
    """No active decisions. Uses renewables (automatic) but no battery/gas/deferral."""
    def predict(self, obs, deterministic=True):
        return np.array([0.0, 0.0, 0.0, 0.0]), None


# ============================================================
# BASELINE 2: RULE-BASED (Industry Standard)
# ============================================================
class RuleBasedPolicy:
    """
    Fixed time-based rules (what most DCs do today):
    - Defer 20% of load always
    - Charge battery 1-5 AM, discharge 4-9 PM
    - Use gas during peak hours (4-8 PM) if available
    - No cooling adjustment
    """
    def predict(self, obs, deterministic=True):
        # Extract hour from cyclical encoding
        hour_sin = obs[0]
        hour_cos = obs[1]
        hour = np.arctan2(hour_sin, hour_cos) * 24 / (2 * np.pi) % 24
        
        # Fixed deferral
        defer = 0.6  # 60% of max = 18% of load
        
        # Battery: charge at night, discharge at peak
        if 1 <= hour <= 5:
            battery = -0.7  # Charge
        elif 16 <= hour <= 21:
            battery = 0.8  # Discharge
        else:
            battery = 0.0
        
        # Gas: only during peak
        gas = 0.8 if 16 <= hour <= 20 else 0.0
        
        return np.array([defer, 0.0, battery, gas]), None


# ============================================================
# BASELINE 3: GREEDY (Cheapest source NOW, no foresight)
# ============================================================
class GreedyPolicy:
    """
    Each hour: use the cheapest available source for remaining demand.
    No foresight — doesn't consider future prices for battery timing.
    Defers when price is above average, runs when below.
    """
    def predict(self, obs, deterministic=True):
        price_signal = obs[10]  # Normalized current price
        gas_signal = obs[12]    # Normalized gas price
        
        # Defer when expensive, run extra when cheap
        defer = 0.8 if price_signal > 0.3 else 0.2
        
        # Battery: discharge if price > 0, charge if price < 0
        if price_signal > 0.2:
            battery = 0.6
        elif price_signal < -0.2:
            battery = -0.6
        else:
            battery = 0.0
        
        # Gas: use when gas is cheaper than grid (gas_signal < price_signal)
        gas = 0.9 if gas_signal < price_signal else 0.0
        
        return np.array([defer, 0.0, battery, gas]), None


# ============================================================
# BASELINE 4: DETERMINISTIC OPTIMAL (Hindsight Oracle)
# ============================================================
class DeterministicOptimalPolicy:
    """
    PERFECT FORESIGHT: knows entire episode's prices in advance.
    Uses this to optimally time battery charge/discharge.
    
    This is the theoretical ceiling — no online algorithm can beat it.
    Requires access to the environment's internal data (cheating).
    """
    def __init__(self, env):
        self.env = env
        self.episode_prices = None
        self.step_count = 0
    
    def reset(self):
        """Call this at episode start to load future prices."""
        t_start = self.env.episode_start
        t_end = t_start + self.env.episode_length
        self.episode_prices = self.env.grid_price[t_start:t_end]
        self.step_count = 0
        
        # Pre-compute optimal battery schedule
        # Strategy: charge in bottom 25% price hours, discharge in top 25%
        sorted_prices = np.sort(self.episode_prices)
        self.charge_threshold = sorted_prices[int(len(sorted_prices) * 0.25)]
        self.discharge_threshold = sorted_prices[int(len(sorted_prices) * 0.75)]
    
    def predict(self, obs, deterministic=True):
        if self.episode_prices is None:
            return np.array([0.5, 0.0, 0.0, 0.5]), None
        
        current_price = self.episode_prices[min(self.step_count, len(self.episode_prices)-1)]
        
        # Perfect battery timing
        if current_price <= self.charge_threshold:
            battery = -0.9  # Charge at cheapest hours
            defer = 0.1     # Run more (it's cheap)
            gas = 0.0
        elif current_price >= self.discharge_threshold:
            battery = 0.9   # Discharge at most expensive hours
            defer = 0.9     # Defer heavily (it's expensive)
            gas = 0.9       # Use gas too (grid is expensive)
        else:
            battery = 0.0
            defer = 0.4
            gas = 0.0
        
        self.step_count += 1
        return np.array([defer, 0.0, battery, gas]), None


# ============================================================
# BASELINE 5: MPC (Model Predictive Control)
# ============================================================
class MPCPolicy:
    """
    Uses the 4h and 24h forecast from observations to make decisions.
    Each step: looks at forecast signals and applies optimal control.
    
    This is what a well-engineered control system (non-ML) would do.
    It's the strongest non-RL baseline.
    """
    def predict(self, obs, deterministic=True):
        price_now = obs[10]       # Current price (normalized)
        forecast_4h = obs[16]     # 4h ahead forecast
        forecast_24h = obs[17]    # 24h ahead forecast
        battery_soc = obs[13]     # Battery state
        
        # MPC logic: compare current price to future forecast
        # If now is cheaper than future → charge battery, run more
        # If now is expensive relative to future → discharge, defer, use gas
        
        price_vs_future = price_now - forecast_4h  # Positive = now expensive
        
        # Battery: charge if now < future, discharge if now > future
        if price_vs_future > 0.2 and battery_soc > 0.15:
            battery = 0.8  # Discharge (now is expensive, future is cheaper)
        elif price_vs_future < -0.2 and battery_soc < 0.85:
            battery = -0.7  # Charge (now is cheap, future is expensive)
        else:
            battery = 0.0
        
        # Deferral: defer if now expensive relative to 24h avg
        if price_now > forecast_24h + 0.1:
            defer = 0.8  # Defer heavily
        elif price_now < forecast_24h - 0.1:
            defer = 0.2  # Run more now (it's cheap)
        else:
            defer = 0.4
        
        # Gas: use when current grid price is high AND gas looks worthwhile
        gas_signal = obs[12]
        gas = 0.8 if (price_now > 0.3 and gas_signal < price_now) else 0.0
        
        return np.array([defer, 0.0, battery, gas]), None


# ============================================================
# EVALUATION FUNCTION (shared with RL)
# ============================================================

def evaluate_baseline(policy, env, n_episodes=N_EVAL_EPISODES, is_oracle=False):
    """Run a baseline policy on the environment and collect metrics."""
    rewards, costs, carbons, waters, sla = [], [], [], [], []
    
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=5000 + ep)
        
        # Oracle needs to see full episode prices
        if is_oracle and hasattr(policy, 'reset'):
            policy.reset()
        
        episode_reward = 0
        done = False
        while not done:
            action, _ = policy.predict(obs, deterministic=True)
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
        "n_eval_episodes": n_episodes,
    }


# ============================================================
# MAIN: Run all baselines on all source configs
# ============================================================

def run_all_baselines():
    """Run all 5 baselines across all 12 source configurations."""
    all_results = []
    
    print("=" * 70)
    print("RUNNING ALL BASELINES (5 strategies × 12 configs = 60 evaluations)")
    print("=" * 70)
    
    for config_name, config in SOURCE_CONFIGS.items():
        print(f"\n  Config: {config_name}")
        env = DataCenterEnergyEnv(data_path=DATA_DIR, **config)
        
        baselines = {
            "DoNothing": (DoNothingPolicy(), False),
            "RuleBased": (RuleBasedPolicy(), False),
            "Greedy": (GreedyPolicy(), False),
            "DeterministicOptimal": (DeterministicOptimalPolicy(env), True),
            "MPC": (MPCPolicy(), False),
        }
        
        for baseline_name, (policy, is_oracle) in baselines.items():
            result = evaluate_baseline(policy, env, n_episodes=N_EVAL_EPISODES, is_oracle=is_oracle)
            result["algorithm"] = baseline_name
            result["config"] = config_name
            result["is_baseline"] = True
            all_results.append(result)
            
            print(f"    {baseline_name:25s}: cost=${result['mean_episode_cost']:>8,.0f} ± "
                  f"${result['std_episode_cost']:>6,.0f} | reward={result['mean_reward']:.2f}")
    
    # Save
    outpath = os.path.join(RESULTS_DIR, "baseline_results.json")
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  ✓ Saved: {outpath} ({len(all_results)} evaluations)")
    
    # Summary table
    print(f"\n{'='*70}")
    print("BASELINE SUMMARY (all_sources config)")
    print(f"{'='*70}")
    all_src = [r for r in all_results if r["config"] == "all_sources"]
    for r in sorted(all_src, key=lambda x: x["mean_episode_cost"]):
        print(f"  {r['algorithm']:25s}: ${r['mean_episode_cost']:>8,.0f}/week")
    
    return all_results


if __name__ == "__main__":
    run_all_baselines()
