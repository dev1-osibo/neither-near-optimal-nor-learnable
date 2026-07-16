"""
Alibaba 2020 GPU Trace -> Cluster Utilization Time Series + Workload Model
==========================================================================
The Alibaba GPU trace (cluster-trace-gpu-v2020) publishes UTILIZATION only, as
event records (instance start/end + per-instance sensor averages) plus machine
specs. It does NOT publish a ready time series, power, or cooling.

This module:
  1. Reconstructs a fixed-cadence (hourly) CLUSTER utilization series from the
     event records:  util_frac(t) = active resource usage / cluster capacity.
  2. Fits a WorkloadModel over the "hour-of-week" profile (the trace preserves
     real time-of-day and day-of-week even though calendar dates are
     desensitized) so the pattern can be instantiated over any target timeline
     (e.g. the 2020-2025 energy record).

Verified schema (cluster-trace-gpu-v2020 README):
  pai_instance_table: job_name, task_name, inst_name, worker_name, inst_id,
                      status, start_time, end_time, machine
  pai_sensor_table:   ... worker_name, machine, cpu_usage (%cores*100),
                      gpu_wrk_util (%GPU), avg_mem, ...
  pai_machine_spec:   machine, gpu_type, cap_cpu (cores), cap_mem, cap_gpu

Times are in seconds, relative to the trace start (desensitized).
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


SECONDS_PER_HOUR = 3600
HOURS_PER_WEEK = 168


def reconstruct_hourly_utilization(
    instances: pd.DataFrame,
    sensors: pd.DataFrame,
    machine_spec: pd.DataFrame,
) -> pd.DataFrame:
    """Reconstruct hourly cluster CPU & GPU utilization fractions from events.

    Method: each instance contributes its (constant, lifetime-average) sensor
    usage to every hour bin its [start_time, end_time) overlaps. Per-hour usage
    is summed across active instances and divided by total cluster capacity.

    Args:
        instances: pai_instance_table (needs worker_name, start_time, end_time).
        sensors:   pai_sensor_table (needs worker_name, cpu_usage, gpu_wrk_util).
        machine_spec: pai_machine_spec (needs cap_cpu, cap_gpu).

    Returns:
        DataFrame indexed by hour (int) with columns cpu_util_frac, gpu_util_frac in [0,1].
    """
    # Cluster capacity: total CPU cores and total GPUs across all machines.
    cap_cpu_cores = float(machine_spec["cap_cpu"].sum())
    cap_gpu_units = float(machine_spec["cap_gpu"].sum())
    if cap_cpu_cores <= 0 or cap_gpu_units <= 0:
        raise ValueError("machine_spec must yield positive CPU and GPU capacity")

    # Join per-instance usage onto lifetimes. cpu_usage is in %cores (100 = 1 core);
    # gpu_wrk_util is in %GPU (100 = 1 GPU). Convert to absolute units.
    df = instances.merge(
        sensors[["worker_name", "cpu_usage", "gpu_wrk_util"]],
        on="worker_name", how="inner",
    ).dropna(subset=["start_time", "end_time"])
    df = df[df["end_time"] > df["start_time"]]
    df["cpu_cores"] = df["cpu_usage"].fillna(0.0) / 100.0
    df["gpu_units"] = df["gpu_wrk_util"].fillna(0.0) / 100.0

    df["h_start"] = (df["start_time"] // SECONDS_PER_HOUR).astype(int)
    df["h_end"] = (df["end_time"] // SECONDS_PER_HOUR).astype(int)
    n_hours = int(df["h_end"].max()) + 1

    cpu_used = np.zeros(n_hours)
    gpu_used = np.zeros(n_hours)
    # Accumulate each instance's usage across the hours it spans.
    for h0, h1, c, g in zip(df["h_start"].values, df["h_end"].values,
                            df["cpu_cores"].values, df["gpu_units"].values):
        cpu_used[h0:h1 + 1] += c
        gpu_used[h0:h1 + 1] += g

    out = pd.DataFrame({
        "hour": np.arange(n_hours),
        "cpu_util_frac": np.clip(cpu_used / cap_cpu_cores, 0.0, 1.0),
        "gpu_util_frac": np.clip(gpu_used / cap_gpu_units, 0.0, 1.0),
    })
    return out


@dataclass
class WorkloadModel:
    """Hour-of-week utilization profile learned from a reconstructed series.

    Preserves the trace's real diurnal + weekly structure so it can be
    re-instantiated onto any calendar timeline. Calendar dates in the trace are
    desensitized, but time-of-day and day-of-week are real, so an
    hour-of-week (0..167) profile is the meaningful, honest summary.
    """
    cpu_mean: np.ndarray  # shape (168,)
    cpu_std: np.ndarray
    gpu_mean: np.ndarray
    gpu_std: np.ndarray

    @classmethod
    def fit(cls, util_df: pd.DataFrame) -> "WorkloadModel":
        """Fit per-hour-of-week mean/std for CPU and GPU utilization."""
        how = util_df["hour"].values % HOURS_PER_WEEK
        def profile(col):
            m = np.zeros(HOURS_PER_WEEK)
            s = np.zeros(HOURS_PER_WEEK)
            vals = util_df[col].values
            for k in range(HOURS_PER_WEEK):
                sel = vals[how == k]
                if len(sel):
                    m[k], s[k] = sel.mean(), sel.std()
            return m, s
        cpu_m, cpu_s = profile("cpu_util_frac")
        gpu_m, gpu_s = profile("gpu_util_frac")
        return cls(cpu_m, cpu_s, gpu_m, gpu_s)

    def sample(self, timestamps: pd.DatetimeIndex, seed: int = 42) -> pd.DataFrame:
        """Instantiate a utilization series aligned to a target timeline.

        Args:
            timestamps: target DatetimeIndex (e.g. hourly 2020-2025).
            seed: RNG seed for the noise term.

        Returns:
            DataFrame with timestamp, cpu_util_frac, gpu_util_frac in [0,1].
        """
        rng = np.random.default_rng(seed)
        how = (timestamps.dayofweek.values * 24 + timestamps.hour.values) % HOURS_PER_WEEK
        cpu = np.clip(self.cpu_mean[how] + rng.normal(0, 1, len(how)) * self.cpu_std[how], 0, 1)
        gpu = np.clip(self.gpu_mean[how] + rng.normal(0, 1, len(how)) * self.gpu_std[how], 0, 1)
        return pd.DataFrame({"timestamp": timestamps, "cpu_util_frac": cpu, "gpu_util_frac": gpu})
