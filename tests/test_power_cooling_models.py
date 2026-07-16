"""Unit tests for the util->power and cooling models.

Validate physical sanity: monotonicity, plausible magnitudes, PUE in
industry range (~1.3-1.8), and correct handling of edge inputs.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from power_model import (  # noqa: E402
    cpu_util_to_power_kw, gpu_util_to_power_kw, facility_it_power_kw, ServerPowerSpec,
)
from cooling_model import cooling_load_kw, total_facility_kw, pue, cooling_water_m3, CoolingSpec  # noqa: E402


def test_cpu_power_monotonic_and_bounds():
    """Power rises with utilization; idle and peak match the spec."""
    spec = ServerPowerSpec(p_idle_w=150, p_peak_w=400)
    p_idle = cpu_util_to_power_kw(np.array([0.0]), n_servers=1000, spec=spec)[0]
    p_peak = cpu_util_to_power_kw(np.array([1.0]), n_servers=1000, spec=spec)[0]
    p_mid = cpu_util_to_power_kw(np.array([0.5]), n_servers=1000, spec=spec)[0]
    assert np.isclose(p_idle, 150.0)      # 1000 * 150W = 150 kW
    assert np.isclose(p_peak, 400.0)      # 1000 * 400W = 400 kW
    assert p_idle < p_mid < p_peak


def test_util_clipping():
    """Out-of-range utilization is clipped to [0,1]."""
    p = cpu_util_to_power_kw(np.array([-0.5, 2.0]), n_servers=100)
    assert p[0] == cpu_util_to_power_kw(np.array([0.0]), 100)[0]
    assert p[1] == cpu_util_to_power_kw(np.array([1.0]), 100)[0]


def test_gpu_power_uses_tdp():
    """GPU power scales with published TDP; V100 draws more than T4."""
    v100 = gpu_util_to_power_kw(np.array([1.0]), n_gpus=100, gpu_type="V100")[0]
    t4 = gpu_util_to_power_kw(np.array([1.0]), n_gpus=100, gpu_type="T4")[0]
    assert np.isclose(v100, 30.0)   # 100 * 300W
    assert np.isclose(t4, 7.0)      # 100 * 70W
    assert v100 > t4


def test_facility_it_power_adds_gpu():
    """Combined CPU+GPU exceeds CPU-only."""
    cpu_only = facility_it_power_kw(np.array([0.5]), n_servers=500)
    with_gpu = facility_it_power_kw(np.array([0.5]), n_servers=500,
                                    gpu_util_fraction=np.array([0.5]), n_gpus=200, gpu_type="V100")
    assert with_gpu[0] > cpu_only[0]


def test_cooling_rises_with_ambient():
    """Cooling load increases with ambient temperature above reference."""
    it = np.array([1000.0, 1000.0])
    cool = cooling_load_kw(it, ambient_temp_c=np.array([10.0, 35.0]))
    assert cool[1] > cool[0]


def test_pue_in_industry_range():
    """PUE stays within a plausible 1.2-2.0 band across temperatures."""
    it = np.full(5, 1000.0)
    temps = np.array([0.0, 15.0, 25.0, 35.0, 45.0])
    p = pue(it, temps)
    assert p.min() >= 1.2
    assert p.max() <= 2.0


def test_water_higher_when_hot_and_dry():
    """Evaporative water use is higher in hot, dry conditions."""
    cool = np.array([100.0, 100.0])
    w = cooling_water_m3(cool, ambient_temp_c=np.array([35.0, 5.0]),
                         relative_humidity_pct=np.array([20.0, 90.0]))
    assert w[0] > w[1]


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
