"""Leakage test for the price-forecast observation in DataCenterEnergyEnv.

Verifies the core integrity property for the paper:
  - In 'persistence' mode, the observation at time t does NOT depend on any
    price after t (no future leakage).
  - In 'oracle' mode, it DOES depend on future prices (confirming the ablation
    ceiling really uses foresight, and that the test can detect leakage).
"""
import os
import sys
import numpy as np

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
sys.path.insert(0, SRC)

from dc_energy_env import DataCenterEnergyEnv  # noqa: E402

FORECAST_DIMS = (16, 17)  # price_forecast_4h_norm, price_forecast_24h_norm


def _obs_at(env, t):
    env.episode_start = 0
    env.current_step = t
    return env._get_obs()


def test_persistence_mode_has_no_future_leakage():
    """Mutating prices AFTER t must not change the forecast observation at t."""
    env = DataCenterEnergyEnv(data_path=DATA, forecast_mode="persistence")
    t = 100
    before = _obs_at(env, t).copy()
    # Corrupt the entire future (everything after t) with a large spike.
    env.grid_price[t + 1:] = env.grid_price[t + 1:] + 999.0
    after = _obs_at(env, t)
    for d in FORECAST_DIMS:
        assert np.isclose(before[d], after[d]), f"persistence leaked future at obs[{d}]"


def test_oracle_mode_uses_future():
    """Oracle mode SHOULD react to future prices (sanity: test can detect leakage)."""
    env = DataCenterEnergyEnv(data_path=DATA, forecast_mode="oracle")
    t = 100
    before = _obs_at(env, t).copy()
    env.grid_price[t + 1:t + 25] = env.grid_price[t + 1:t + 25] + 999.0
    after = _obs_at(env, t)
    changed = any(not np.isclose(before[d], after[d]) for d in FORECAST_DIMS)
    assert changed, "oracle mode did not react to future prices (unexpected)"


def test_default_mode_is_leakage_free():
    """The default constructor must be the honest (persistence) mode."""
    env = DataCenterEnergyEnv(data_path=DATA)
    assert env.forecast_mode == "persistence"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
