"""
Reconstruct hourly cluster utilization from the Alibaba GPU-2020 trace (P2, step 1).
====================================================================================
Produces an hourly time series of cluster CPU and GPU utilization (fractions in
[0,1]) with the trace's REAL diurnal/weekly structure, to drive the data-center
IT-load model (util -> power) for the Paper 1 substrate rebuild.

Method (documented in DECISION_LOG D16):
- Each instance (worker_name) has a start/end interval (pai_instance_table) and an
  ACTUAL average usage (pai_sensor_table: cpu_usage in %-cores, gpu_wrk_util in %-GPU).
- Sum active instances' usage per hourly bin via a difference-array sweep, then
  normalize by total cluster capacity (pai_machine_spec: cap_cpu, cap_gpu).
- Only sensor-covered instances are used (actual measured usage). We need the SHAPE,
  which is mapped onto the facility idle->peak band later, so absolute undercount is
  immaterial. Disclosed as a modeling choice (Paper Caveat Register #1).

Output: data/alibaba_gpu2020/hourly_utilization.csv
  columns: hour_index, cpu_util, gpu_util, n_active_workers
"""

import os
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "alibaba_gpu2020")
HOUR = 3600.0

# Column orders are headerless; taken from the official README schema.
INSTANCE_COLS = ["job_name", "task_name", "inst_name", "worker_name", "inst_id",
                 "status", "start_time", "end_time", "machine"]
SENSOR_COLS = ["job_name", "task_name", "worker_name", "inst_id", "machine", "gpu_name",
               "cpu_usage", "gpu_wrk_util", "avg_mem", "max_mem", "avg_gpu_wrk_mem",
               "max_gpu_wrk_mem", "read", "write", "read_count", "write_count"]
SPEC_COLS = ["machine", "gpu_type", "cap_cpu", "cap_mem", "cap_gpu"]


def load_capacity():
    """Total cluster CPU cores and GPU count from machine_spec."""
    spec = pd.read_csv(os.path.join(DATA, "pai_machine_spec.csv"), header=None, names=SPEC_COLS)
    cap_cpu = pd.to_numeric(spec["cap_cpu"], errors="coerce").fillna(0).sum()
    cap_gpu = pd.to_numeric(spec["cap_gpu"], errors="coerce").fillna(0).sum()
    print(f"  [CAP] machines={len(spec):,} total_cap_cpu={cap_cpu:,.0f} cores "
          f"total_cap_gpu={cap_gpu:,.0f} GPUs")
    return float(cap_cpu), float(cap_gpu)


def build():
    cap_cpu, cap_gpu = load_capacity()

    # Instance intervals (only the columns we need).
    print("  [LOAD] pai_instance_table ...")
    inst = pd.read_csv(os.path.join(DATA, "pai_instance_table.csv"), header=None,
                       names=INSTANCE_COLS,
                       usecols=["worker_name", "start_time", "end_time"])
    inst["start_time"] = pd.to_numeric(inst["start_time"], errors="coerce")
    inst["end_time"] = pd.to_numeric(inst["end_time"], errors="coerce")
    inst = inst.dropna(subset=["worker_name", "start_time", "end_time"])
    inst = inst[inst["end_time"] > inst["start_time"]]
    print(f"    valid instances: {len(inst):,}")

    # Actual usage per worker.
    print("  [LOAD] pai_sensor_table ...")
    sens = pd.read_csv(os.path.join(DATA, "pai_sensor_table.csv"), header=None,
                       names=SENSOR_COLS,
                       usecols=["worker_name", "cpu_usage", "gpu_wrk_util"])
    sens["cpu_usage"] = pd.to_numeric(sens["cpu_usage"], errors="coerce").fillna(0.0)
    sens["gpu_wrk_util"] = pd.to_numeric(sens["gpu_wrk_util"], errors="coerce").fillna(0.0)
    # One usage row per worker (average if duplicated).
    sens = sens.groupby("worker_name", as_index=False).mean(numeric_only=True)

    # Join usage onto intervals; keep only workers with measured usage.
    df = inst.merge(sens, on="worker_name", how="inner")
    del inst, sens
    print(f"    instances with sensor usage: {len(df):,}")

    # Convert to cores / GPU-equivalents (percent -> absolute).
    df["cpu_cores"] = df["cpu_usage"] / 100.0
    df["gpu_eq"] = df["gpu_wrk_util"] / 100.0

    # Time bins (hourly) from a common origin.
    origin = df["start_time"].min()
    end_max = df["end_time"].max()
    n_hours = int(np.ceil((end_max - origin) / HOUR)) + 1
    print(f"  [TIME] origin={origin:.0f}s span_hours={n_hours} (~{n_hours/24:.1f} days)")

    s_bin = np.floor((df["start_time"].values - origin) / HOUR).astype(np.int64)
    e_bin = np.floor((df["end_time"].values - origin) / HOUR).astype(np.int64)
    s_bin = np.clip(s_bin, 0, n_hours - 1)
    e_bin = np.clip(e_bin, 0, n_hours - 1)

    # Difference-array sweep for interval sums (O(N+H)).
    def sweep(weights):
        diff = np.zeros(n_hours + 2, dtype=np.float64)
        np.add.at(diff, s_bin, weights)
        np.add.at(diff, e_bin + 1, -weights)
        return np.cumsum(diff)[:n_hours]

    cpu_used = sweep(df["cpu_cores"].values)
    gpu_used = sweep(df["gpu_eq"].values)
    active = sweep(np.ones(len(df)))

    cpu_util = np.clip(cpu_used / cap_cpu, 0, None) if cap_cpu > 0 else np.zeros(n_hours)
    gpu_util = np.clip(gpu_used / cap_gpu, 0, None) if cap_gpu > 0 else np.zeros(n_hours)

    out = pd.DataFrame({
        "hour_index": np.arange(n_hours),
        "cpu_util": cpu_util,
        "gpu_util": gpu_util,
        "n_active_workers": active.astype(int),
    })
    outpath = os.path.join(DATA, "hourly_utilization.csv")
    out.to_csv(outpath, index=False)

    print(f"\n  [RESULT] hours={n_hours}")
    print(f"    cpu_util: mean={cpu_util.mean():.3f} min={cpu_util.min():.3f} "
          f"max={cpu_util.max():.3f}")
    print(f"    gpu_util: mean={gpu_util.mean():.3f} min={gpu_util.min():.3f} "
          f"max={gpu_util.max():.3f}")
    # Diurnal sanity: mean gpu_util by hour-of-day should show a day/night pattern.
    hod = out["hour_index"] % 24
    by_hod = out.groupby(hod)["gpu_util"].mean()
    print(f"    gpu_util by hour-of-day spread: min={by_hod.min():.3f} max={by_hod.max():.3f} "
          f"(ratio={by_hod.max()/max(by_hod.min(),1e-6):.2f}x)")
    print(f"  [SAVED] {outpath}")


if __name__ == "__main__":
    build()
