"""
Alibaba 2020 GPU Trace Loader
=============================
Loads the real Alibaba PAI GPU cluster trace (Jul-Aug 2020, ~56 days, 1720
machines) and produces an hourly cluster-utilization series that drives the
power model (power_model.py).

VERIFIED against the real files (headerless CSVs):

  pai_machine_metric.csv  (2,009,423 rows) columns, in order:
    worker_name, machine, start_time, end_time,
    machine_cpu_iowait, machine_cpu_kernel, machine_cpu_usr,
    machine_gpu, machine_load_1, machine_net_receive,
    machine_num_worker, machine_cpu

  pai_machine_spec.csv (1,897 rows):
    machine, gpu_type, cap_cpu, cap_mem, cap_gpu

CRITICAL DATA SEMANTICS (verified empirically, do not assume otherwise):
  * machine_cpu  -> overall CPU utilization PERCENT in [0,100] (mean ~27.6).
                    Fraction = machine_cpu / 100.
  * machine_gpu  -> SUM of per-GPU utilization across the machine's installed
                    GPUs, in units where 100 == one fully-utilized GPU. Range
                    observed 0..787 (machines have cap_gpu of 2 or 8). Thus
                    GPU-equivalents utilized = machine_gpu / 100, and the true
                    per-installed-GPU utilization fraction = (machine_gpu/100)/cap_gpu.
                    (Clipping machine_gpu at 100 -- as a naive percent -- would
                    discard most of the GPU signal and is WRONG.)
  * timestamps    -> desensitized RELATIVE seconds; bin by hour = floor(t/3600).
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

SECONDS_PER_HOUR = 3600

METRIC_COLS = ["worker_name", "machine", "start_time", "end_time",
               "machine_cpu_iowait", "machine_cpu_kernel", "machine_cpu_usr",
               "machine_gpu", "machine_load_1", "machine_net_receive",
               "machine_num_worker", "machine_cpu"]
SPEC_COLS = ["machine", "gpu_type", "cap_cpu", "cap_mem", "cap_gpu"]

DEFAULT_TIME_COL = "start_time"
DEFAULT_CPU_COL = "machine_cpu"      # percent [0,100]
DEFAULT_GPU_COL = "machine_gpu"      # SUM of per-GPU util, 100 == one full GPU
DEFAULT_MACHINE_COL = "machine"
DEFAULT_CAP_GPU = 2                  # fallback installed-GPU count if spec absent


@dataclass
class HourlyUtilization:
    """Hourly cluster utilization series.

    cpu_util_fraction : mean per-machine CPU utilization, [0, 1].
    gpu_util_fraction : mean per-INSTALLED-GPU utilization, [0, 1]
                        (machine_gpu/100 divided by that machine's cap_gpu).
    gpu_equiv_utilized: mean GPU-equivalents utilized per reporting machine
                        (machine_gpu/100), the raw magnitude signal.
    n_samples         : rows aggregated into each hour (data-quality signal).
    """
    hour_index: np.ndarray
    cpu_util_fraction: np.ndarray
    gpu_util_fraction: np.ndarray
    gpu_equiv_utilized: np.ndarray
    n_samples: np.ndarray

    def __len__(self) -> int:
        return len(self.hour_index)


def load_machine_metric(path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """Load the headerless pai_machine_metric.csv with verified column names."""
    return pd.read_csv(path, header=None, names=METRIC_COLS, nrows=nrows,
                       usecols=[DEFAULT_MACHINE_COL, DEFAULT_TIME_COL,
                                DEFAULT_GPU_COL, DEFAULT_CPU_COL])


def load_machine_spec(path: str) -> pd.DataFrame:
    """Load the headerless pai_machine_spec.csv with verified column names."""
    return pd.read_csv(path, header=None, names=SPEC_COLS)


def to_hourly_utilization(
    df: pd.DataFrame,
    spec: Optional[pd.DataFrame] = None,
    time_col: str = DEFAULT_TIME_COL,
    cpu_col: str = DEFAULT_CPU_COL,
    gpu_col: str = DEFAULT_GPU_COL,
    machine_col: str = DEFAULT_MACHINE_COL,
) -> HourlyUtilization:
    """Aggregate machine-level rows into an hourly cluster utilization series.

    machine_gpu is interpreted as GPU-equivalents*100 and divided by each
    machine's installed cap_gpu (from `spec`) to recover a true per-GPU
    utilization fraction. Without `spec`, a cluster-default cap_gpu is used.
    """
    t = pd.to_numeric(df[time_col], errors="coerce")
    cpu_frac = (pd.to_numeric(df[cpu_col], errors="coerce") / 100.0).clip(0.0, 1.0)
    gpu_equiv = (pd.to_numeric(df[gpu_col], errors="coerce") / 100.0).clip(lower=0.0)

    # Per-machine installed GPU count for the per-GPU utilization fraction.
    if spec is not None and machine_col in spec.columns and "cap_gpu" in spec.columns:
        cap_map = pd.to_numeric(spec.set_index(machine_col)["cap_gpu"], errors="coerce")
        cap = df[machine_col].map(cap_map)
        cap = cap.where(cap > 0, np.nan)  # CPU-only machines (cap_gpu=0) -> no per-GPU frac
    else:
        cap = pd.Series(DEFAULT_CAP_GPU, index=df.index, dtype=float)
    gpu_per_gpu_frac = (gpu_equiv / cap).clip(0.0, 1.0)

    work = pd.DataFrame({
        "hour": (t // SECONDS_PER_HOUR).astype("Int64"),
        "cpu": cpu_frac,
        "gpu_frac": gpu_per_gpu_frac,
        "gpu_equiv": gpu_equiv,
    }).dropna(subset=["hour"])

    agg = work.groupby("hour").agg(
        cpu=("cpu", "mean"),
        gpu_frac=("gpu_frac", "mean"),
        gpu_equiv=("gpu_equiv", "mean"),
        n=("cpu", "size"),
    )
    full_index = np.arange(int(agg.index.min()), int(agg.index.max()) + 1)
    agg = agg.reindex(full_index)
    for c in ("cpu", "gpu_frac", "gpu_equiv"):
        agg[c] = agg[c].interpolate().ffill().bfill()
    agg["n"] = agg["n"].fillna(0)

    return HourlyUtilization(
        hour_index=full_index,
        cpu_util_fraction=agg["cpu"].to_numpy(dtype=float),
        gpu_util_fraction=agg["gpu_frac"].to_numpy(dtype=float),
        gpu_equiv_utilized=agg["gpu_equiv"].to_numpy(dtype=float),
        n_samples=agg["n"].to_numpy(dtype=float),
    )


def gpu_type_mix(spec: pd.DataFrame) -> dict:
    """Return {gpu_type: installed_gpu_count} for a TDP-weighted power model."""
    if spec is None or "gpu_type" not in spec.columns:
        return {}
    cap = pd.to_numeric(spec.get("cap_gpu", 0), errors="coerce").fillna(0)
    return spec.assign(_cap=cap).groupby("gpu_type")["_cap"].sum().astype(int).to_dict()
