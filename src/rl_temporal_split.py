"""
Temporal Train/Test Split Utilities
====================================
Converts calendar dates into episode-start index bounds for
DataCenterEnergyEnv, so that agents can be TRAINED on one time period
and EVALUATED on a disjoint, later period (leakage-free generalization).

Why this exists
---------------
The original RL pipeline sampled episode starts across the ENTIRE
2020-2025 dataset for both training and evaluation. That allows an agent
to be scored on weeks it also trained on (temporal leakage). These helpers
produce explicit, non-overlapping [lo, hi) index windows so the split is
enforced at the environment level.

All bounds are EPISODE-START indices (the first hour of a 168h episode),
already clamped so an episode cannot run past the end of its window.
"""

import os
import numpy as np
import pandas as pd

from dc_energy_env import DataCenterEnergyEnv


def _load_timestamps(data_path, config):
    """
    Build a throwaway env just to read the exact timestamp axis the env uses
    after its own filtering/merging. We must derive indices from the SAME
    dataframe the env produces, otherwise index bounds would not line up.

    Returns a numpy datetime64 array of length n_hours and the env's max_start.
    """
    env = DataCenterEnergyEnv(data_path=data_path, **config)
    return np.asarray(env.timestamps), int(env.max_start), int(env.episode_length)


def date_to_index(timestamps, date_str):
    """
    First row index whose timestamp is >= date_str. If date_str is beyond the
    last timestamp, returns len(timestamps).

    Args:
        timestamps: numpy datetime64 array (ascending).
        date_str: e.g. "2024-01-01".
    """
    target = np.datetime64(pd.Timestamp(date_str))
    idx = int(np.searchsorted(timestamps, target, side="left"))
    return idx


def make_split_ranges(data_path, config, split_date="2024-01-01"):
    """
    Produce train/test episode-start windows around a single split date.

    Train window : episodes that START and FULLY FIT before split_date.
    Test window  : episodes that START on/after split_date.

    Args:
        data_path: directory with the CSV data.
        config: source-config dict (use_solar/use_wind/... ) so the env's
            row filtering matches the models being evaluated.
        split_date: boundary between train and test.

    Returns:
        dict with:
          train_range = (lo, hi)  episode-start bounds for training
          test_range  = (lo, hi)  episode-start bounds for testing
          split_index = row index of split_date
          n_hours, episode_length
          coverage = human-readable description
    """
    timestamps, max_start, episode_length = _load_timestamps(data_path, config)
    split_idx = date_to_index(timestamps, split_date)

    # Train episode must fully finish before the split boundary.
    train_lo = 0
    train_hi = max(0, split_idx - episode_length)

    # Test episode may start any time from the split onward, but must still fit.
    test_lo = min(split_idx, max_start)
    test_hi = max_start

    if train_hi <= train_lo:
        raise ValueError(f"Train window empty for split_date={split_date}.")
    if test_hi <= test_lo:
        raise ValueError(f"Test window empty for split_date={split_date}.")

    return {
        "split_date": split_date,
        "split_index": split_idx,
        "n_hours": len(timestamps),
        "episode_length": episode_length,
        "max_start": max_start,
        "train_range": (train_lo, train_hi),
        "test_range": (test_lo, test_hi),
        "first_ts": str(timestamps[0]),
        "last_ts": str(timestamps[-1]),
        "coverage": (
            f"train starts in [{train_lo},{train_hi}) "
            f"(<= {split_date}); test starts in [{test_lo},{test_hi}) (>= {split_date})"
        ),
    }


if __name__ == "__main__":
    # Quick self-check against the default all_sources config.
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    cfg = {"use_solar": True, "use_wind": True, "use_battery": True, "use_gas": True}
    info = make_split_ranges(DATA_DIR, cfg, split_date="2024-01-01")
    for k, v in info.items():
        print(f"  {k}: {v}")
