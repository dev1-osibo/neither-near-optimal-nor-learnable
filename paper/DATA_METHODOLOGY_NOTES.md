# Data Provenance & Methodology — Behind-the-Meter Single-Facility Benchmark

This document describes the data provenance and modelling choices for the benchmark
(`PAPER1_DRAFT.md`, §3). It is the data-provenance companion for the reproducibility section
(§9): replayed real Alibaba GPU workload trace, real ERCOT market and environmental data, and
physically modelled IT power and cooling.

---

## Data provenance by signal

### 1. Workload trace → IT power — REAL trace, replayed (§3.2)
**Source:** Alibaba `cluster-trace-gpu-v2020` (Weng et al. 2022).
**Method:** Hourly cluster GPU utilization is reconstructed by replaying instance start/end
events weighted by per-instance usage, normalized by machine capacity. Because trace timestamps
are desensitized (time-of-day and day-of-week are real; calendar dates are not), the
reconstruction is collapsed into a **typical-week profile** — 168 values indexed by
(day-of-week × 24 + hour-of-day) — and replayed onto the 2020–2025 timestamp axis.
**Mapping to power:** linear idle/peak envelope (Fan et al. 2007):
`IT(t) = P_nom · [ φ + (1 − φ) · u(t) ]`, with `P_nom = 20 MW`, idle fraction `φ = 0.30`.
**Note:** load is diurnal but relatively flat — the replayed typical-week profile the agent
sees has mean ≈ 12%, peak ≈ 15% of capacity (raw pre-aggregation hourly peak ≈ 25%, smoothed by
the typical-week averaging) — a genuine property of ML clusters and a limit on deferral headroom
(Threats #1–#3).

### 2. Cooling — physical model (§3.3)
Temperature-dependent PUE: `cooling(t) = IT(t) · [ PUE(T(t)) − 1 ]`,
`PUE(T) = 1.25 + 0.01·max(0, T − 20 °C)`. Uncontrolled demand `D(t) = IT(t) + cooling(t)`
(mean ≈ 9.7 MW at design sizing).

### 3. On-site generation & storage — physical models (§3.4)
- **Solar (5 MW):** `P_solar = clip(G·32,679·0.18·0.85/1000, 0, 5000)` kW — a 32,679 m² array at
  18% module efficiency, 0.85 performance ratio, sized to reach 5 MW nameplate at 1000 W/m².
  Driven by the site's own ambient weather → ~17% capacity factor (conservative vs a
  purpose-sited farm).
- **Wind (5 MW):** cubic power curve in wind speed; cut-in 3.5, rated 12, cut-out 25 m/s.
- **Battery (20 MWh / 10 MW):** **0.90 round-trip efficiency** (√0.90 ≈ 0.949 applied at each of
  charge and discharge). Surplus renewable auto-charges up to 95% SoC regardless of the agent's
  action (disclosed dynamic, Threat #11).
- **Gas (2 MW):** Henry Hub cost, 0.41 kgCO₂/kWh (EIA/EPA natural-gas electricity).

### 4. Market data — REAL (§3.5)
Dispatch runs on **real ERCOT LMPs**; gas priced at **Henry Hub**. CAISO price history begins
only in 2023 and is excluded to keep the market signal consistent across the full 2020–2025 span
(Threat #6).

### 5. Carbon intensity — DERIVED (§3.5)
Real **EIA hourly fuel mix** × **IPCC lifecycle emission factors** → hourly grid carbon
intensity. This is the industry-standard method (ElectricityMaps, WattTime, EPA eGRID use the
same approach): known emission factors applied to real generation data, not a novel methodology.
The grid-carbon observation is normalized with data-derived mean/std so the agent can resolve it.

### 6. Weather — REAL (§3.5)
Temperature, humidity, solar irradiance, and wind speed from **Open-Meteo Archive / NASA POWER**.
Drives solar/wind generation, cooling PUE, and the evaporative-water model.

---

## Reward normalization (§3.7)
Reward is the negative weighted sum of four **normalized** objectives (cost/carbon/water/SLA),
weights α = (0.4, 0.3, 0.2, 0.1) unless the sweep overrides. Divisors are **data-derived** (mean
hourly value over the loaded substrate), not hand-set: **C = 592.1, K = 3683.7, W = 2.638**.
- Water-model humidity factor `h_f = 1 + (50 − RH)/100` — drier air evaporates more per unit of
  cooling.
- Divisors use full-record statistics (a train-only C would be ≈733, a 19% offset); this is a
  reward-shaping choice applied identically to all controllers, not an evaluation leak — every
  reported cost/carbon/water figure is an absolute physical quantity (Threat #10).

---

## Summary table (substrate)
| Signal | Type | Provenance | Role |
|---|---|---|---|
| GPU workload trace | REAL (replayed typical-week) | Alibaba cluster-trace-gpu-v2020 (Weng et al. 2022) | drives IT power |
| IT power | MODEL | Fan et al. 2007 linear envelope | facility load |
| Cooling | MODEL | temperature-dependent PUE(T) | facility load |
| Solar / wind | MODEL on real weather | irradiance / wind-speed power curves | on-site generation |
| Battery / gas | MODEL | SoC dynamics (η = 0.90) / Henry Hub + emission factor | storage & dispatchable |
| Electricity price | REAL | ERCOT LMP | cost objective |
| Grid carbon intensity | DERIVED | EIA fuel mix × IPCC factors | carbon objective |
| Weather | REAL | Open-Meteo / NASA POWER | generation, cooling, water |

---

## Key threats a reader should note
Power and cooling are **models**, not metered facility telemetry (Threat #4) — comparative
conclusions are more robust than any absolute figure, since every controller (RL, heuristics, and
the LP optimum) is scored on the same substrate and seeds. The load is real in structure but
**replayed**, not a real calendar sequence (Threat #3); the trace is a **GPU cluster** (Threat
#2); and results are for a **single facility**, mitigated by the sizing sweep across a fourfold
battery range (§5.8, Threat #5).
