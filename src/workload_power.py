"""
Workload -> IT power and physical cooling models (P2 step-2 / Fix 1 + Fix 2).
=============================================================================
Turns the reconstructed real utilization (build_workload_load.py) into a
facility IT-power and cooling-power time series, replayed by (day-of-week,
hour-of-day) onto any target timestamp axis.

Models (all cited, see DECISION_LOG Caveat Register #8):
- util -> IT power: facility linear idle/peak envelope,
    IT = nameplate * (idle_frac + (1 - idle_frac) * util)
  (utilization is the real cluster fraction; nameplate/idle_frac are the
   facility sizing, disclosed and sensitivity-tested).
- IT -> cooling: PUE(T_ambient) model,
    cooling = IT * (PUE(T) - 1),  PUE(T) = pue_min + pue_slope*max(0, T - pue_tref)
  (ASHRAE/Uptime-style temperature dependence; real weather drives T).

NOTE on phase (Caveat Register #3): trace timestamps are desensitized, so the
diurnal phase is not calendar-anchored. We replay a consistent (dow,hod) profile;
the workload is an independent real-structure input, and the price/cooling
coupling that matters physically flows through the real weather signal, not the
workload phase.
"""

import numpy as np
import pandas as pd


def build_typical_week(util_csv, col="gpu_util"):
    """Collapse the reconstructed hourly utilization into a typical-week profile:
    168 values indexed by (day_of_week*24 + hour_of_day).

    Trace hour_index is used as the internal clock (phase is arbitrary but
    consistent, per the desensitized-timestamp caveat).
    Returns np.ndarray shape (168,).
    """
    df = pd.read_csv(util_csv)
    h = df["hour_index"].values.astype(int)
    dow = (h // 24) % 7
    hod = h % 24
    key = dow * 24 + hod
    util = df[col].values.astype(float)
    prof = np.zeros(168)
    cnt = np.zeros(168)
    np.add.at(prof, key, util)
    np.add.at(cnt, key, 1.0)
    prof = np.divide(prof, np.maximum(cnt, 1.0))
    # Fill any empty slot (shouldn't happen for a 56-day trace) with global mean.
    prof[cnt == 0] = util.mean()
    return prof


def replay_utilization(typical_week, dow_arr, hod_arr):
    """Map a target timestamp axis (its day-of-week and hour-of-day arrays) onto
    the typical-week utilization profile. Returns per-timestamp utilization."""
    idx = (dow_arr.astype(int) % 7) * 24 + (hod_arr.astype(int) % 24)
    return typical_week[idx]


def it_power_kw(util, nameplate_kw=20000.0, idle_frac=0.30):
    """Facility IT power from utilization fraction (linear idle/peak envelope)."""
    return nameplate_kw * (idle_frac + (1.0 - idle_frac) * np.asarray(util, dtype=float))


def pue(temp_c, pue_min=1.25, pue_slope=0.01, pue_tref=20.0):
    """Power Usage Effectiveness as a function of ambient temperature."""
    return pue_min + pue_slope * np.maximum(0.0, np.asarray(temp_c, dtype=float) - pue_tref)


def cooling_power_kw(it_kw, temp_c, pue_min=1.25, pue_slope=0.01, pue_tref=20.0):
    """Cooling power from IT power and ambient temperature via PUE(T)."""
    return np.asarray(it_kw, dtype=float) * (pue(temp_c, pue_min, pue_slope, pue_tref) - 1.0)
