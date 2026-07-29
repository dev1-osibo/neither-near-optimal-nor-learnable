"""
Adversarial Scenario Transforms
===============================
Non-invasive out-of-distribution transforms applied to a fully-constructed
DataCenterEnergyEnv. Default env behavior is never changed; a scenario is
opt-in and mutates only the underlying data arrays a policy will experience.

IMPORTANT: normalization statistics (price_mean/std etc.) are intentionally
NOT recomputed after a transform. A deployed agent carries fixed preprocessing
learned at training time, so leaving these fixed is the honest representation
of how the trained policy would actually see a shifted world.

Oracle safety: because these transforms mutate env.grid_price BEFORE any
policy (including the hindsight oracle) reads it, the oracle ceiling is
computed against the shocked prices too — it never gets an unfair reference.
"""

import numpy as np


def apply_price_shock(env, scale=3.0, spike_frac=0.05, rng_seed=0):
    """
    Multiply all grid prices by `scale` and inject extreme spikes into a random
    `spike_frac` of hours (freeze-event style). Tests over/under-reaction.
    """
    rng = np.random.default_rng(rng_seed)
    env.grid_price = env.grid_price * scale
    n_spikes = int(len(env.grid_price) * spike_frac)
    idx = rng.integers(0, len(env.grid_price), size=n_spikes)
    env.grid_price[idx] = env.grid_price[idx] * 10.0  # 10x spike on top
    return env


def apply_flat_market(env):
    """
    Collapse all price volatility to a constant (mean price). Optimization has
    near-zero value here; a good policy must AVOID destroying value (needless
    battery cycling, pointless deferral causing SLA hits).
    """
    const = float(env.grid_price.mean())
    env.grid_price = np.full_like(env.grid_price, const)
    return env


def apply_demand_shock(env, scale=2.0):
    """Scale total facility demand (and its IT/cooling components) up by `scale`."""
    env.total_demand = env.total_demand * scale
    env.it_load = env.it_load * scale
    env.cooling_load = env.cooling_load * scale
    return env


def apply_missing_gas(env):
    """Force the gas generator offline at runtime (capacity -> 0) even though the
    agent was trained expecting gas. Tests missing-actuator robustness."""
    env.gas_capacity_kw = 0
    return env


def apply_biased_forecast(env, bias=0.30, noise=0.10, rng_seed=0):
    """
    Switch the env to 'provided' forecast mode and feed SYSTEMATICALLY WRONG
    forecasts (multiplicative bias + gaussian noise) built from the true future.
    Probes sensitivity of the forecast observation channels to corruption.
    """
    rng = np.random.default_rng(rng_seed)
    n = env.n_hours
    price = env.grid_price
    # True forward means (what an honest forecast would target).
    true_4h = np.array([price[t + 1:t + 5].mean() if t + 5 < n else price[t]
                        for t in range(n)])
    true_24h = np.array([price[t + 1:t + 25].mean() if t + 25 < n else price[t]
                         for t in range(n)])
    corrupt = lambda x: x * (1.0 + bias) * (1.0 + rng.normal(0, noise, size=x.shape))
    env.forecast_mode = "provided"
    env._pf4 = corrupt(true_4h)
    env._pf24 = corrupt(true_24h)
    return env


# Registry so the driver can look transforms up by name.
SCENARIOS = {
    "price_shock": apply_price_shock,
    "flat_market": apply_flat_market,
    "demand_shock": apply_demand_shock,
    "missing_gas": apply_missing_gas,
    "biased_forecast": apply_biased_forecast,
}


class ClampFlexibilityPolicy:
    """
    Wraps any policy and forces workload-deferral and battery actions to zero,
    simulating a deployment where those actuators are unavailable at runtime.
    The agent still 'thinks' it can use them (it was trained with them), so this
    tests graceful degradation vs thrashing.
    """
    def __init__(self, inner):
        self.inner = inner

    def predict(self, obs, deterministic=True):
        action, state = self.inner.predict(obs, deterministic=deterministic)
        action = np.array(action, dtype=float).copy()
        action[0] = 0.0   # no workload deferral
        action[2] = 0.0   # no battery action
        return action, state
