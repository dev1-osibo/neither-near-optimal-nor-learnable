"""Unit tests for the Alibaba GPU trace reconstruction + workload model.

Uses a tiny synthetic dataset matching the verified cluster-trace-gpu-v2020
schema, so it runs without the (large) real download and validates the logic.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from workload_trace import reconstruct_hourly_utilization, WorkloadModel, HOURS_PER_WEEK  # noqa: E402


def _toy_tables():
    """Two machines (192 cores, 4 GPUs total); two instances with known usage."""
    machine_spec = pd.DataFrame({
        "machine": ["m1", "m2"],
        "gpu_type": ["V100", "V100"],
        "cap_cpu": [96, 96],   # 192 cores total
        "cap_mem": [512, 512],
        "cap_gpu": [2, 2],     # 4 GPUs total
    })
    # inst A: hours 0-1, uses 96 cores (cpu_usage=9600%) + 2 GPUs (gpu=200%)
    # inst B: hour 1 only, uses 96 cores + 2 GPUs
    instances = pd.DataFrame({
        "worker_name": ["A", "B"],
        "start_time": [0, 3600],
        "end_time": [7199, 7199],   # A spans hours 0&1; B spans hour 1
        "machine": ["m1", "m2"],
    })
    sensors = pd.DataFrame({
        "worker_name": ["A", "B"],
        "cpu_usage": [9600.0, 9600.0],   # 96 cores each
        "gpu_wrk_util": [200.0, 200.0],  # 2 GPUs each
    })
    return instances, sensors, machine_spec


def test_reconstruction_utilization_fractions():
    """Hour 0: only A active -> 96/192 CPU = 0.5, 2/4 GPU = 0.5.
       Hour 1: A+B active -> 192/192 = 1.0 CPU, 4/4 = 1.0 GPU."""
    instances, sensors, spec = _toy_tables()
    util = reconstruct_hourly_utilization(instances, sensors, spec)
    assert len(util) == 2
    assert np.isclose(util.loc[0, "cpu_util_frac"], 0.5)
    assert np.isclose(util.loc[0, "gpu_util_frac"], 0.5)
    assert np.isclose(util.loc[1, "cpu_util_frac"], 1.0)
    assert np.isclose(util.loc[1, "gpu_util_frac"], 1.0)


def test_util_bounded():
    """Utilization fractions never exceed 1.0 even if usage would."""
    instances, sensors, spec = _toy_tables()
    sensors["cpu_usage"] = [999999.0, 999999.0]
    util = reconstruct_hourly_utilization(instances, sensors, spec)
    assert util["cpu_util_frac"].max() <= 1.0


def test_workload_model_fit_and_sample():
    """Fitted model produces a bounded series of the requested length."""
    # Build a longer synthetic series with a diurnal shape over 3 weeks.
    hours = np.arange(24 * 21)
    diurnal = 0.4 + 0.3 * np.sin(2 * np.pi * (hours % 24) / 24)
    util_df = pd.DataFrame({
        "hour": hours,
        "cpu_util_frac": np.clip(diurnal, 0, 1),
        "gpu_util_frac": np.clip(diurnal * 0.8, 0, 1),
    })
    model = WorkloadModel.fit(util_df)
    assert model.cpu_mean.shape == (HOURS_PER_WEEK,)

    ts = pd.date_range("2020-07-01", periods=24 * 14, freq="h")
    sampled = model.sample(ts, seed=1)
    assert len(sampled) == len(ts)
    assert sampled["cpu_util_frac"].between(0, 1).all()
    assert sampled["gpu_util_frac"].between(0, 1).all()
    # Diurnal structure should survive: peak-hour mean > trough-hour mean.
    sampled["h"] = sampled["timestamp"].dt.hour
    hourly_mean = sampled.groupby("h")["cpu_util_frac"].mean()
    assert hourly_mean.max() > hourly_mean.min()


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
