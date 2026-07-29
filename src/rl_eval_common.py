"""
Shared RL / Baseline Evaluation Harness
========================================
One evaluation function used by BOTH the significance tests and the
adversarial tests, so every policy is scored identically and comparisons
stay apples-to-apples.

Key property: returns PER-EPISODE arrays (not just means). Paired statistics
require the individual episode outcomes, aligned by seed across policies.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from dc_energy_env import DataCenterEnergyEnv
from baselines import (
    DoNothingPolicy, RuleBasedPolicy, GreedyPolicy,
    DeterministicOptimalPolicy, MPCPolicy, SOURCE_CONFIGS,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

# Seed block for significance evaluation. Disjoint from the 5000-block used
# during training-time evaluation, so we are not re-using the exact scoring
# weeks the pipeline already reported on.
DEFAULT_SEED_BASE = 6000


def make_env(config_name, episode_start_range=None, data_path=DATA_DIR):
    """Construct an env for a named source config, optional temporal window."""
    cfg = SOURCE_CONFIGS[config_name]
    return DataCenterEnergyEnv(
        data_path=data_path, episode_start_range=episode_start_range, **cfg
    )


def evaluate_policy_episodes(policy, env, seeds, is_oracle=False):
    """
    Run a policy over a fixed list of episode seeds and return per-episode
    outcomes. Same seed -> same historical week -> paired across policies.

    Args:
        policy: object exposing .predict(obs, deterministic=True) -> (action, _).
                RL SB3 models and the baseline classes both satisfy this.
        env: a DataCenterEnergyEnv instance.
        seeds: iterable of int seeds (each defines one episode/week).
        is_oracle: if True and policy has .reset(), call it each episode
                   (DeterministicOptimal needs to preload the episode's prices).

    Returns:
        dict of numpy arrays keyed by metric, each length == len(seeds).
    """
    costs, carbons, waters, slas, rewards = [], [], [], [], []

    for s in seeds:
        obs, _ = env.reset(seed=int(s))
        # Oracle must reload future prices for THIS episode's window.
        if is_oracle and hasattr(policy, "reset"):
            policy.reset()

        ep_reward = 0.0
        done = False
        info = {}
        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        costs.append(info.get("episode_cost", 0.0))
        carbons.append(info.get("episode_carbon", 0.0))
        waters.append(info.get("episode_water", 0.0))
        slas.append(info.get("episode_sla_violations", 0))
        rewards.append(ep_reward)

    return {
        "cost": np.asarray(costs, dtype=float),
        "carbon": np.asarray(carbons, dtype=float),
        "water": np.asarray(waters, dtype=float),
        "sla": np.asarray(slas, dtype=float),
        "reward": np.asarray(rewards, dtype=float),
    }


def build_baselines(env):
    """Instantiate the 5 baseline policies for a given env. Returns
    list of (name, policy, is_oracle)."""
    return [
        ("DoNothing", DoNothingPolicy(), False),
        ("RuleBased", RuleBasedPolicy(), False),
        ("Greedy", GreedyPolicy(), False),
        ("MPC", MPCPolicy(), False),
        ("DeterministicOptimal", DeterministicOptimalPolicy(env), True),
    ]


def model_path(algo, config, seed):
    """Path to a trained SB3 model zip (without extension, as SB3 expects)."""
    return os.path.join(MODELS_DIR, f"{algo}_{config}_seed{seed}")
