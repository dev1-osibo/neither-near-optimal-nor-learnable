"""
Physical Cooling / Facility Power Model
=======================================
Derives cooling load (kW) and total facility power (kW) from IT power and
ambient temperature. Real facility cooling telemetry is not publicly published
(verified), so cooling MUST be modeled -- this is standard practice
(Green-DCC, Sustain-Cluster, DeepMind all model it). The key discipline here,
unlike the earlier synthetic generator, is that cooling is a transparent
function of a REAL IT-load driver and REAL ambient weather, and every
coefficient is explicit and documented for sensitivity analysis.

Model:
    partial-PUE for cooling rises with ambient temperature (economizer loses
    effectiveness as it gets hot):

        cooling_kw = it_kw * k_base * (1 + beta * max(0, T_amb - T_ref))

    total_facility_kw = it_kw + cooling_kw + it_kw * loss_frac   # power/distribution loss

Defaults sit within ASHRAE / industry PUE norms (PUE ~1.3-1.8). They are NOT
tuned to any target; the paper reports sensitivity to k_base and beta.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class CoolingSpec:
    """Coefficients for the ambient-dependent cooling model."""
    k_base: float = 0.35      # cooling kW per IT kW at/below reference temp
    beta: float = 0.010       # fractional cooling increase per deg C above T_ref
    t_ref_c: float = 15.0     # reference ambient temp (economizer effective below)
    loss_frac: float = 0.05   # power distribution / conversion loss as frac of IT
    # Evaporative water model (m^3 per cooling kWh), humidity/temp adjusted:
    water_m3_per_cooling_kwh: float = 0.0018  # ~1.8 L/kWh baseline (industry range)

    def __post_init__(self) -> None:
        if self.k_base <= 0:
            raise ValueError("k_base must be positive")


def cooling_load_kw(
    it_power_kw: np.ndarray,
    ambient_temp_c: np.ndarray,
    spec: CoolingSpec | None = None,
) -> np.ndarray:
    """Cooling power (kW) as a function of IT load and ambient temperature."""
    spec = spec or CoolingSpec()
    it = np.asarray(it_power_kw, dtype=float)
    t = np.asarray(ambient_temp_c, dtype=float)
    temp_excess = np.clip(t - spec.t_ref_c, 0.0, None)
    return it * spec.k_base * (1.0 + spec.beta * temp_excess)


def total_facility_kw(
    it_power_kw: np.ndarray,
    ambient_temp_c: np.ndarray,
    spec: CoolingSpec | None = None,
) -> np.ndarray:
    """Total facility power (kW) = IT + cooling + distribution loss."""
    spec = spec or CoolingSpec()
    it = np.asarray(it_power_kw, dtype=float)
    cool = cooling_load_kw(it, ambient_temp_c, spec)
    return it + cool + it * spec.loss_frac


def pue(it_power_kw: np.ndarray, ambient_temp_c: np.ndarray, spec: CoolingSpec | None = None) -> np.ndarray:
    """Instantaneous PUE = total / IT (sanity metric; expect ~1.3-1.8)."""
    it = np.asarray(it_power_kw, dtype=float)
    total = total_facility_kw(it, ambient_temp_c, spec)
    return total / np.clip(it, 1e-9, None)


def cooling_water_m3(
    cooling_kwh: np.ndarray,
    ambient_temp_c: np.ndarray,
    relative_humidity_pct: np.ndarray,
    spec: CoolingSpec | None = None,
) -> np.ndarray:
    """Evaporative cooling water use (m^3), scaled by temp and humidity.

    Hotter/drier conditions -> more evaporative water per cooling kWh.
    """
    spec = spec or CoolingSpec()
    cool = np.asarray(cooling_kwh, dtype=float)
    t = np.asarray(ambient_temp_c, dtype=float)
    rh = np.asarray(relative_humidity_pct, dtype=float)
    temp_factor = np.where(t > 25, 1.2, np.where(t < 10, 0.6, 1.0))
    humidity_factor = 1.0 + (50.0 - np.clip(rh, 0, 100)) / 100.0  # drier -> more water
    return cool * spec.water_m3_per_cooling_kwh * temp_factor * humidity_factor
