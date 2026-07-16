"""Tests for the Alibaba GPU trace loader.

Synthetic tests validate the corrected GPU semantics (machine_gpu is a sum of
per-GPU utilization where 100 == one full GPU, divided by cap_gpu to recover a
per-GPU fraction). A real-data smoke test runs on the actual extracted file when
present, so the loader is validated against reality, not just assumptions.
"""
import os
import sys
import numpy as np
import pandas as pd
import pytest

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
sys.path.insert(0, SRC)

from trace_loader import (  # noqa: E402
    to_hourly_utilization, load_machine_metric, load_machine_spec,
    gpu_type_mix, HourlyUtilization,
)
from power_model import facility_it_power_kw  # noqa: E402


def _synthetic_metric():
    """machine_gpu is GPU-equivalents*100 (e.g. 2 GPUs @80% -> 160)."""
    return pd.DataFrame([
        {"machine": "m1", "start_time": 100,  "machine_cpu": 50, "machine_gpu": 160},
        {"machine": "m2", "start_time": 200,  "machine_cpu": 50, "machine_gpu": 160},
        {"machine": "m1", "start_time": 3700, "machine_cpu": 20, "machine_gpu": 40},
        {"machine": "m2", "start_time": 7300, "machine_cpu": 90, "machine_gpu": 200},
    ])


def _synthetic_spec():
    return pd.DataFrame([
        {"machine": "m1", "gpu_type": "V100", "cap_cpu": 96, "cap_mem": 512, "cap_gpu": 2},
        {"machine": "m2", "gpu_type": "V100", "cap_cpu": 96, "cap_mem": 512, "cap_gpu": 2},
    ])


def test_cpu_fraction_and_hourly_binning():
    hu = to_hourly_utilization(_synthetic_metric(), spec=_synthetic_spec())
    assert isinstance(hu, HourlyUtilization)
    assert list(hu.hour_index) == [0, 1, 2]
    assert np.isclose(hu.cpu_util_fraction[0], 0.5)
    assert np.isclose(hu.cpu_util_fraction[2], 0.9)


def test_gpu_equivalents_not_clipped_at_one():
    """The multi-GPU magnitude must survive (gpu_equiv up to cap_gpu)."""
    hu = to_hourly_utilization(_synthetic_metric(), spec=_synthetic_spec())
    assert np.isclose(hu.gpu_equiv_utilized[0], 1.6)   # 160/100
    assert np.isclose(hu.gpu_equiv_utilized[2], 2.0)   # 200/100


def test_per_gpu_fraction_uses_cap_gpu():
    """per-GPU fraction = (machine_gpu/100)/cap_gpu, in [0,1]."""
    hu = to_hourly_utilization(_synthetic_metric(), spec=_synthetic_spec())
    assert np.isclose(hu.gpu_util_fraction[0], 0.8)    # 1.6 / 2
    assert np.isclose(hu.gpu_util_fraction[2], 1.0)    # 2.0 / 2
    assert (hu.gpu_util_fraction >= 0).all() and (hu.gpu_util_fraction <= 1).all()


def test_contiguous_hours_filled():
    df = pd.DataFrame([
        {"machine": "m1", "start_time": 100, "machine_cpu": 40, "machine_gpu": 80},
        {"machine": "m1", "start_time": 100 + 3 * 3600, "machine_cpu": 60, "machine_gpu": 120},
    ])
    hu = to_hourly_utilization(df, spec=_synthetic_spec())
    assert list(hu.hour_index) == [0, 1, 2, 3]
    assert not np.isnan(hu.gpu_util_fraction).any()


def test_composes_with_power_model():
    hu = to_hourly_utilization(_synthetic_metric(), spec=_synthetic_spec())
    it_kw = facility_it_power_kw(
        hu.cpu_util_fraction, n_servers=1000,
        gpu_util_fraction=hu.gpu_util_fraction, n_gpus=4000, gpu_type="V100",
    )
    assert (it_kw > 0).all()
    assert it_kw[2] > it_kw[1]


def test_gpu_type_mix():
    mix = gpu_type_mix(_synthetic_spec())
    assert mix == {"V100": 4}   # 2 machines * cap_gpu 2


# ---------------------------------------------------------------------------
# Real-data smoke test: runs only if the extracted files are present.
# ---------------------------------------------------------------------------
_METRIC = os.path.join(DATA, "pai_machine_metric.csv")
_SPEC = os.path.join(DATA, "pai_machine_spec.csv")


@pytest.mark.skipif(not (os.path.exists(_METRIC) and os.path.exists(_SPEC)),
                    reason="real Alibaba trace not present")
def test_real_trace_sample_is_sane():
    """Validate on a 100k-row sample of the real extracted trace."""
    df = load_machine_metric(_METRIC, nrows=100_000)
    spec = load_machine_spec(_SPEC)
    hu = to_hourly_utilization(df, spec=spec)
    assert len(hu) > 0
    assert np.nanmin(hu.cpu_util_fraction) >= 0 and np.nanmax(hu.cpu_util_fraction) <= 1
    assert np.nanmin(hu.gpu_util_fraction) >= 0 and np.nanmax(hu.gpu_util_fraction) <= 1
    # GPU-equivalents per machine should be plausible (installed max is 8).
    assert np.nanmax(hu.gpu_equiv_utilized) <= 8.5
    # Spec mix should include known GPU types.
    mix = gpu_type_mix(spec)
    assert sum(mix.values()) > 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
