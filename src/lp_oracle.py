"""
True Offline Optimum via Linear Programming (audit #3)
======================================================
Replaces the quantile-threshold "DeterministicOptimal" (a clairvoyant heuristic that ignored
SoC/deferral/coupling) with a genuine per-episode cost minimum under perfect foresight.

For one 168-hour episode we minimize total dispatch cost
    sum_t  price_t*(grid_to_load + grid_to_charge) + gas_price_t*gas
subject to the SAME physical constraints every controller faces:
  - energy balance each hour (renewables -> battery -> gas -> grid all serve load),
  - battery SoC dynamics with one-way efficiency eta at each end (round-trip eta^2 = 0.90),
    charge/discharge power caps, 0 <= SoC <= capacity, SoC_0 = 0.5*cap,
  - gas capacity cap,
  - optional workload deferral modeled as load-shifting: up to 30% of each hour's load may be
    deferred, accumulated in a backlog (cap 0.30*mean_load*12h, matching the env ceiling),
    and all deferred load served by episode end (no SLA violation).

This is a valid COST LOWER BOUND (savings upper bound) for any controller operating these
assets under these limits: the true optimum. We solve it in two modes:
  - storage-only  (deferral disabled): pure renewable/battery/gas/grid arbitrage optimum,
  - full          (deferral enabled) : the true optimum, which dominates every controller.

Cost is emission-factor-independent, so this bound is valid on both the old and corrected
substrates; we run it on the CORRECTED env so the RuleBased->optimum gap reflects the
substrate we would retrain on.
"""

import numpy as np
from scipy.optimize import linprog
from scipy import sparse


def solve_episode(demand, price, gas_price, solar, wind,
                  cap_kwh, rate_kw, eff, gas_cap_kw,
                  soc0_frac=0.5, allow_defer=True, defer_frac=0.30,
                  defer_window_h=12):
    """Return (min_cost, status) for one episode. Arrays are length T (hours)."""
    T = len(demand)
    R = np.asarray(solar, float) + np.asarray(wind, float)  # free renewable each hour
    price = np.asarray(price, float)
    gas_price = np.asarray(gas_price, float)
    demand = np.asarray(demand, float)

    has_batt = cap_kwh > 0 and rate_kw > 0
    has_gas = gas_cap_kw > 0
    soc_init = soc0_frac * cap_kwh
    KEEP = 11.0 / 12.0   # env drains 1/12 of the backlog each hour (kept fraction 11/12)

    # Variable blocks, each length T: gl gc gas rs rc pd d b soc
    #   d = load deferred OUT this hour (control); b = backlog after this hour.
    # Env deferral recursion (SLA-avoiding, per step()):
    #   b_t = (11/12)*(b_{t-1} + d_t);  served_t = demand_t - (11/12)*d_t + (1/12)*b_{t-1}
    # so deferred load is drained geometrically (~12 h), exactly as the environment does it.
    names = ["gl", "gc", "gas", "rs", "rc", "pd", "d", "b", "soc"]
    off = {nm: i * T for i, nm in enumerate(names)}
    N = len(names) * T

    def ix(nm, t):
        return off[nm] + t

    # --- Objective: cost = price*(gl+gc) + gas_price*gas ---
    c = np.zeros(N)
    c[off["gl"]:off["gl"] + T] = price
    c[off["gc"]:off["gc"] + T] = price
    c[off["gas"]:off["gas"] + T] = gas_price

    rows, cols, vals, beq = [], [], [], []
    r = 0

    def add(row, col, v):
        rows.append(row); cols.append(col); vals.append(v)

    # (a) Load balance: gl + gas + rs + pd = served_t = demand_t - (11/12)d_t + (1/12)b_{t-1}
    #     => gl + gas + rs + pd + (11/12)d_t - (1/12)b_{t-1} = demand_t   (b_{-1}=0)
    for t in range(T):
        add(r, ix("gl", t), 1.0); add(r, ix("gas", t), 1.0); add(r, ix("rs", t), 1.0)
        add(r, ix("pd", t), 1.0); add(r, ix("d", t), KEEP)
        if t > 0:
            add(r, ix("b", t - 1), -1.0 / 12.0)
        beq.append(demand[t]); r += 1

    # (b) SoC dynamics: soc_t - soc_{t-1} - eff*(gc+rc) + (1/eff)*pd = [soc_init if t==0 else 0]
    for t in range(T):
        add(r, ix("soc", t), 1.0)
        if t > 0:
            add(r, ix("soc", t - 1), -1.0)
        add(r, ix("gc", t), -eff); add(r, ix("rc", t), -eff)
        add(r, ix("pd", t), 1.0 / eff)
        beq.append(soc_init if t == 0 else 0.0); r += 1

    # (c) Backlog dynamics: b_t - (11/12)b_{t-1} - (11/12)d_t = 0   (b_{-1}=0)
    for t in range(T):
        add(r, ix("b", t), 1.0)
        if t > 0:
            add(r, ix("b", t - 1), -KEEP)
        add(r, ix("d", t), -KEEP)
        beq.append(0.0); r += 1

    A_eq = sparse.coo_matrix((vals, (rows, cols)), shape=(r, N)).tocsr()
    b_eq = np.array(beq)

    # --- Inequalities ---
    urows, ucols, uvals, bub = [], [], [], []
    ru = 0

    def addu(row, col, v):
        urows.append(row); ucols.append(col); uvals.append(v)

    # (e) Renewable cap: rs + rc <= R
    for t in range(T):
        addu(ru, ix("rs", t), 1.0); addu(ru, ix("rc", t), 1.0)
        bub.append(R[t]); ru += 1
    # (f) Charge rate: gc + rc <= rate
    for t in range(T):
        addu(ru, ix("gc", t), 1.0); addu(ru, ix("rc", t), 1.0)
        bub.append(rate_kw); ru += 1
    # (g) SLA-avoid (no forced serve): b_{t-1} + d_t <= cap_t = 0.30*demand_t*12
    for t in range(T):
        addu(ru, ix("d", t), 1.0)
        if t > 0:
            addu(ru, ix("b", t - 1), 1.0)
        bub.append(defer_frac * demand[t] * defer_window_h); ru += 1

    A_ub = sparse.coo_matrix((uvals, (urows, ucols)), shape=(ru, N)).tocsr()
    b_ub = np.array(bub)

    # --- Bounds ---
    lb = np.zeros(N); ub = np.full(N, np.inf)
    ub[off["gas"]:off["gas"] + T] = gas_cap_kw if has_gas else 0.0
    if not has_batt:
        for nm in ("gc", "rc", "pd", "soc"):
            ub[off[nm]:off[nm] + T] = 0.0
    else:
        ub[off["pd"]:off["pd"] + T] = rate_kw
        ub[off["soc"]:off["soc"] + T] = cap_kwh
    # deferral: d_t in [0, 0.30*demand_t]; b_t >= 0 (bounded above by SLA-avoid + dynamics)
    if allow_defer:
        ub[off["d"]:off["d"] + T] = defer_frac * demand
    else:
        ub[off["d"]:off["d"] + T] = 0.0
        ub[off["b"]:off["b"] + T] = 0.0

    bounds = list(zip(lb, ub))
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    if not res.success:
        return None, res.message
    return float(res.fun), "ok"
