# Paper Blueprint — Real-Trace Single-Facility Multi-Source DC Energy Control

**Status:** Blueprint (pre-build). Supersedes the synthetic-substrate approach.
**Decision:** Option 1 chosen. Target: IEEE-level journal.
**Date:** 2026-07-10

---

## 1. Focus (locked)

This paper asks whether deep reinforcement learning actually beats well-tuned
heuristics and model-predictive control for operating a single data center's
on-site multi-source energy mix (grid, solar, wind, battery, gas) across the
coupled objectives of cost, carbon, water, and SLA compliance. Unlike the
geo-distributed workload-shifting systems that dominate the literature
(Green-DCC, Sustain-Cluster, and the Nvidia/EPRI substation pilots), it targets
the single-site, behind-the-meter dispatch regime faced by operators who cannot
spatially route their load. Its central contribution is methodological honesty:
it evaluates every controller under realistic (non-oracle) price and carbon
forecasts on real ERCOT/CAISO market, carbon, and weather data with a real
workload trace, quantifying how much any learned advantage shrinks once perfect
foresight is removed. The result is a rigorous, reproducible benchmark that
reports where sophisticated learning genuinely helps and where a cheap heuristic
is already enough — the honest reality check the field currently lacks.

### Research arc (framing)
This work is the rigorous, real-data validation of the *buildable core* of the
author's prior vision paper (Osibo, "Transforming High-Energy Data Center Sites:
Sustainability with Predictive Analytics and Futuristic Technologies," IJSR
14(8), 2025). That paper proposed a context-aware, multi-signal energy
orchestration framework; this paper implements and honestly tests its central
dispatch mechanism. Industry has since turned toward the same "align compute
with available power" thesis (Nvidia/EPRI substation micro-DC pilots, IEEE
Spectrum 2026), motivating the timeliness of a grounded evaluation.

### Grid-stress dimension (rides the substation theme, stays in the niche)
In addition to cost/carbon/water/SLA, we evaluate a **grid-stress-aware**
objective: operating on-site dispatch to minimize grid draw during the rare
constrained windows the Nvidia/EPRI work highlights (grid peak <200 hrs/yr;
spatial shifting needed only ~0.1% of the time). This connects on-site
flexibility to grid relief *without* requiring spatial routing.

### Scope exclusions (integrity guardrail)
Explicitly OUT of scope (the speculative, never-built elements from the vision
paper that must not re-enter): quantum-enhanced AI, quantum-integrated
blockchain / energy trading, and wearable/human-activity/urban-mobility signals.
The paper uses only real, causal, reproducible inputs.

### Follow-on (after experiments — do not start yet)
Once the experiments are complete, extend toward the **underserved-communities /
unreliable-grid** direction discussed separately: on-site multi-source dispatch
under frequent unscheduled outages + diesel avoidance (emerging-market grids),
where outage prediction becomes a control input. Tracked as future work.

---

## 2. Positioning — the differentiator (verified against prior art)

| Prior work | Scope | What they optimize | Gap we fill |
|---|---|---|---|
| Google Carbon-Intelligent (2020) | Fleet | Temporal + spatial carbon shifting | Not on-site multi-source dispatch |
| Green-DCC (2025) | DC **cluster** | Hierarchical MARL, workload+cooling across sites | Geo-distributed, not single-site generation/storage |
| Sustain-Cluster / DCcluster-Opt (NeurIPS 2025 D&B) | **Geo-distributed** cluster | Where+when workload placement, carbon/cost/SLA/water | Assumes grid supply per site; no on-site solar/wind/battery/gas co-dispatch |

**Our niche:** the *single-facility, behind-the-meter multi-source* problem (grid + 5MW solar + 5MW wind + 20MWh battery + 2MW gas) co-optimized with workload deferral and cooling. Relevant to colo/edge/single-site operators who cannot geo-distribute.

**Second differentiator (the honest-science hook):** rigorous **oracle-free** evaluation. Most works (including our own earlier runs) leak perfect price foresight into the controller. We quantify how much the RL advantage shrinks when forecasts are realistic.

---

## 3. What we keep vs. rebuild (grounded in the actual repo)

**KEEP (real, reusable):**
- `data/real_lmp_ERCOT_2020_2025.csv`, `real_lmp_CAISO_2020_2025.csv` — real LMP.
- `data/carbon_intensity_{CISO,ERCO,PJM}_full.csv` — real (EIA fuel-mix × IPCC factors).
- `data/eia_demand_*`, gas prices, `weather_{ashburn,phoenix,the_dalles}` — real.
- `src/dc_energy_env.py` — single-facility multi-source env (the differentiator).
- `src/baselines.py` — DoNothing / RuleBased / Greedy / MPC / DeterministicOptimal.
- `src/tft_model.py`, RL harness (`train_rl_checkpointed.py`), `fair_model_comparison.py`.

**REBUILD (the fixes):**
1. **Replace synthetic `it_load`** with a **real-workload-driven** IT load.
2. **Replace circular cooling** with a documented physical cooling model.
3. **Remove oracle prices** from `dc_energy_env._get_obs()`; feed TFT forecasts instead.
4. **Regenerate baselines** (no `baseline_results.json` exists — must be produced).
5. **Proper temporal split** everywhere; no leakage.

---

## 4. Data pipeline (honest, layered, each layer sourced)

```
Real workload trace (Alibaba '18/'20 CPU/GPU util  OR  Google '19 May, CC-BY)
        │  util→power model  (linear / SPEC-power / GPU-TDP; lit: 2–7% MAPE, Google fleet <5%)
        ▼
IT load (kW)  ── physical cooling model (PUE + ambient temp, ASHRAE) ──►  cooling load (kW)
        │
        ├── real weather (have)      ── ambient for cooling + solar/wind availability
        ├── real LMP price (have)    ── cost objective
        ├── real carbon (have)       ── carbon objective
        └── real gas price (have)    ── gas dispatch cost
        ▼
Single-facility multi-source env  →  RL / heuristics / MPC  →  cost, carbon, water, SLA
```

**Timestamp alignment (the known obstacle) — two options, both documented:**
- **Option A (cleanest real slice):** Google 2019 trace = real **May 2019**. Join to real May 2019 LMP/carbon/weather → fully real-aligned 1-month experiment. Limitation: single month, limited seasonal variety.
- **Option B (multi-year):** learn a workload *model* (diurnal/weekly/burst distribution) from the real trace, instantiate across 2020–2025 to align with the full energy record. Limitation: workload becomes distributionally-real but not calendar-real. Disclose explicitly.
- **Plan:** primary results on A (defensible, real-aligned); generalization on B (breadth). Report both.

---

## 5. Experimental design

**Controller inputs (oracle-free):** current state + **TFT forecasts** of price and carbon at 4h/24h (TFT trained only on past → real forecasting task on *real* price/carbon targets, no circularity).

**Methods compared (same env, same seeds, same eval episodes):**
- Baselines: DoNothing, RuleBased (time-of-use), Greedy, **MPC (uses TFT forecasts)**.
- RL: PPO, SAC, TD3, A2C (already implemented).
- Upper bound: DeterministicOptimal (oracle) — reported as ceiling only, clearly labeled.

**Metrics:** weekly cost ($), carbon (kg CO₂), water (m³), SLA violations. Report mean ± std over ≥5 seeds, ≥20 eval episodes. Statistical tests (paired, multiple-comparison corrected).

**Core experiments:**
1. RL vs heuristics vs MPC across the 12 source configurations (grid_only … all_sources).
2. **Oracle vs realistic-forecast ablation** — the headline honesty result: Δ advantage when foresight is removed.
3. Forecast-quality sensitivity (TFT vs persistence vs perfect) on control outcome.
4. Objective-weight sweep (Pareto: cost↔carbon↔water).
5. Generalization: train on one price regime / test on another (temporal split).

**Expected honest findings (hypotheses, not assumed):** RL's margin over a well-tuned MPC is small and shrinks under realistic forecasts; RL's value (if any) concentrates in resource-rich configs; heuristics are a strong, cheap floor. If RL *does* win cleanly somewhere, that becomes the positive contribution. Either outcome is publishable because the *design* is honest.

---

## 6. Threats to validity (state up front in the paper)

- **Cooling & IT-power are modeled**, not metered (real facility telemetry is unpublished — verified). Mitigated by using real workload as driver + validated power models; disclosed as a limitation.
- **Workload–energy time alignment** is a modeling choice (Option A/B above).
- **Single-facility synthetic infrastructure sizing** (5MW solar etc.) — report sensitivity to sizing.
- **Simulation, not deployment** — standard in the field (Green-DCC, Sustain-Cluster are also sim).

---

## 7. Target venues (IEEE-level)

- IEEE Transactions on Sustainable Computing
- IEEE Transactions on Smart Grid (energy-systems framing)
- ACM e-Energy (if conference route preferred)
- Applied Energy / Energy and AI (Elsevier alternates)

---

## 8. Task breakdown (build order)

- **P0 — Novelty lock:** full read of Green-DCC + DCcluster-Opt claims; write the 1-paragraph delta. (Do before coding.)
- **P1 — Data layer:** acquire trace; build util→power model; build cooling model; align to energy data (Option A first).
- **P2 — Env fix:** remove oracle prices; wire TFT forecasts into `_get_obs()`; unit-test no-leakage.
- **P3 — Forecasts:** retrain TFT to forecast real price + carbon (past-only); validate MAPE + calibration.
- **P4 — Baselines:** run `baselines.py` end-to-end; produce `baseline_results.json`.
- **P5 — RL:** retrain 4 algos on the fixed env (real trace, forecast-driven); aggregate.
- **P6 — Analysis:** the 5 experiments; figures/tables; statistical tests.
- **P7 — Write-up:** IEEE format; threats-to-validity; reproducibility package.

---

## 9. Non-negotiables (integrity)

- No oracle foresight in any reported controller result (oracle = labeled ceiling only).
- Every modeled quantity disclosed as modeled, with its source/validation.
- Baseline numbers must come from a re-run artifact, never from prose.
- Temporal splits everywhere; no concurrent-feature leakage.
