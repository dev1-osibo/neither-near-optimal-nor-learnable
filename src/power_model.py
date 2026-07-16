"""
Utilization -> IT Power Model
=============================
Converts real cluster-trace utilization (CPU %, optional GPU %) into data-center
IT power draw (kW). Public traces (Alibaba 2018/2020, Google 2019) publish
UTILIZATION only -- never power -- so a util->power model is required. This
module implements the standard, literature-validated linear model.

References for the linear util->power approach (accuracy in real deployments):
- Google fleet PDU power model: <5% MAPE (Power Modeling for Datacenter Planning, 2021)
- Linear regression on CPU utilization: 2-7% error (arXiv:1411.3201)
- SPECpower-derived models (green-coding-solutions/spec-power-model)

Model (per server):
    P(u) = P_idle + (P_peak - P_idle) * u          # u = CPU utilization fraction [0,1]

GPU servers add a per-GPU term keyed on GPU type (published TDPs):
    P_gpu(u_gpu) = P_gpu_idle + (TDP - P_gpu_idle) * u_gpu

All parameters are explicit and configurable so the paper can report sensitivity.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# Published GPU thermal design power (watts). Used only when a GPU trace is driving load.
GPU_TDP_W = {
    "V100": 300.0, "V100M32": 300.0,
    "T4": 70.0,
    "P100": 250.0,
    "K80": 300.0, "K40m": 235.0, "M60": 300.0,
    "A100": 400.0,
    "MISC": 250.0,  # older-generation fallback (Alibaba GPU trace 'MISC')
}
GPU_IDLE_FRACTION = 0.10  # idle draw ~10% of TDP (typical)


@dataclass
class ServerPowerSpec:
    """Per-server power envelope for the linear CPU model.

    Defaults approximate a modern 2-socket rack server (SPECpower range).
    """
    p_idle_w: float = 150.0   # idle wall power (W)
    p_peak_w: float = 400.0   # 100%-CPU wall power (W)

    def __post_init__(self) -> None:
        if self.p_peak_w < self.p_idle_w:
            raise ValueError("p_peak_w must be >= p_idle_w")


def cpu_util_to_power_kw(
    cpu_util_fraction: np.ndarray,
    n_servers: int,
    spec: Optional[ServerPowerSpec] = None,
) -> np.ndarray:
    """Convert an aggregate CPU-utilization series to facility IT power (kW).

    Args:
        cpu_util_fraction: array of mean cluster CPU utilization in [0, 1].
        n_servers: number of servers the utilization represents.
        spec: per-server power envelope (defaults to ServerPowerSpec()).

    Returns:
        IT power draw in kW, same length as the input series.
    """
    spec = spec or ServerPowerSpec()
    u = np.clip(np.asarray(cpu_util_fraction, dtype=float), 0.0, 1.0)
    per_server_w = spec.p_idle_w + (spec.p_peak_w - spec.p_idle_w) * u
    total_w = per_server_w * n_servers
    return total_w / 1000.0


def gpu_util_to_power_kw(
    gpu_util_fraction: np.ndarray,
    n_gpus: int,
    gpu_type: str = "V100",
) -> np.ndarray:
    """Convert GPU-utilization series to GPU power (kW) using published TDPs.

    Args:
        gpu_util_fraction: array of mean GPU utilization in [0, 1].
        n_gpus: number of GPUs represented.
        gpu_type: key into GPU_TDP_W (e.g. 'V100', 'T4', 'A100', 'MISC').

    Returns:
        GPU power draw in kW.
    """
    tdp = GPU_TDP_W.get(gpu_type.upper(), GPU_TDP_W["MISC"])
    idle = GPU_IDLE_FRACTION * tdp
    u = np.clip(np.asarray(gpu_util_fraction, dtype=float), 0.0, 1.0)
    per_gpu_w = idle + (tdp - idle) * u
    return per_gpu_w * n_gpus / 1000.0


def facility_it_power_kw(
    cpu_util_fraction: np.ndarray,
    n_servers: int,
    spec: Optional[ServerPowerSpec] = None,
    gpu_util_fraction: Optional[np.ndarray] = None,
    n_gpus: int = 0,
    gpu_type: str = "V100",
) -> np.ndarray:
    """Total IT power (kW) = CPU-server power (+ optional GPU power).

    For CPU-only traces (Alibaba 2018, Google 2019) pass only cpu_util_fraction.
    For GPU traces (Alibaba 2020) also pass gpu_util_fraction/n_gpus/gpu_type.
    """
    it_kw = cpu_util_to_power_kw(cpu_util_fraction, n_servers, spec)
    if gpu_util_fraction is not None and n_gpus > 0:
        it_kw = it_kw + gpu_util_to_power_kw(gpu_util_fraction, n_gpus, gpu_type)
    return it_kw
