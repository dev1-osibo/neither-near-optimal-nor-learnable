# Paper 1 — Results (DRAFT)

> ⚠️ **SUPERSEDED (2026-07-26).** This early results draft contains PRE-correction (buggy
> gas-EF) numbers. The authoritative, corrected Results now live inline in **`PAPER1_DRAFT.md`
> §5** (verified against the post-rebuild artifacts). Kept for history only — do not cite.

> **Status:** Draft results section. Every number is traceable to a re-run artifact
> (no prose-only figures, per blueprint §9). Sources:
> `results/rl_backup_20260716/rl_temporal_verdict_realtrace.json` (main verdict),
> `.../baseline_results_realtrace.json` (substrate sanity),
> `.../pareto_analysis/pareto_analysis.json` (objective-weight sweep).
> Framing follows DECISION_LOG **D21** (do not oversell the 4/12) and the
> **Paper Caveat Register**.
>
> **Integrity guard (D-guards):** the 4-algorithm benchmark (§R.2–R.3) and the
> objective-weight Pareto sweep (§R.4) are reported **separately** and never pooled —
> the sweep uses different objective weights, so its scalar reward is not comparable
> to the fixed-weight main run.

---

## R.1 Setup recap (grounding)

- **Substrate:** single facility, real-trace load (Alibaba GPU-2020 replay, aligned by
  hour-of-day/day-of-week), physical PUE(T_ambient) cooling, real ERCOT prices, real
  EIA-fuel-mix carbon, real weather. IT nameplate 20 MW, idle fraction 0.30,
  PUE = 1.25 + 0.01·max(0, T−20) → ~10 MW mean facility. On-site assets: 5 MW solar,
  5 MW wind, 20 MWh battery, 2 MW gas.
- **Leakage control:** temporal split — train on 2020–2023, evaluate on **held-out
  2024–2025** (env step indices [35064, 52272]). Agents never see the test window in
  training.
- **Protocol:** 4 algorithms (PPO, SAC, TD3, A2C) × 12 source configurations × 5 seeds
  = 240 training runs. Each policy evaluated on **200 test-window episodes**. Baselines
  (DoNothing, RuleBased, Greedy, MPC, DeterministicOptimal/oracle) evaluated on the
  **same real-trace substrate with matched seeds (8000–8199)**, so RL and baselines are
  compared like-for-like.
- **Statistics:** paired comparison of per-episode cost, Holm-corrected across the 12
  configurations; Wilcoxon and bootstrap 95% CIs computed alongside; Cohen's d reported.
  "Significant" = Holm-adjusted p < 0.05 **and** the direction favors the named method.

**Baseline-protocol note (must state in the paper):** an earlier substrate-sanity check
(`baseline_results_realtrace.json`, n = 20, full-range sampling) reported RuleBased within
~2% of the oracle. The **headline comparison uses the matched-seed, held-out-test
baselines** in the verdict artifact instead; on that protocol the RuleBased→oracle gap is
**3.0–6.9%** (Table R1). We report the matched-protocol numbers because they are the only
ones evaluated identically to the RL agents; the n = 20 figure was a sanity check, not a
comparator.

---

## R.2 The heuristic ceiling: RuleBased vs oracle

Across all 12 configurations the hand-tuned time-of-use **RuleBased** controller lands
within **3.0–6.9%** of the offline **DeterministicOptimal (oracle)** ceiling on held-out
weekly cost. The gap is smallest in renewable-only configs (grid_only 3.0%, grid_solar
3.0%, grid_wind 4.0%, grid_solar_wind 4.0%) and widens where dispatchable/storable assets
add decision complexity (gas and all_sources 5–7%). This establishes the central premise:
**a cheap heuristic already captures ~93–97% of the achievable cost reduction**, leaving a
narrow 3–7% band for any learned controller to compete over.

---

## R.2.1 Full baseline set at matched protocol (Table R1b) [Gate 2 — DONE]

**Table R1b.** Held-out weekly cost (USD), matched seeds 8000–8199, n=200, `persistence`
forecast. Lowest non-oracle in **bold**. "MPC vs RB" = MPC's cost penalty relative to
RuleBased.

| Configuration | DoNothing | RuleBased | Greedy | MPC | Oracle | MPC vs RB |
|---|---:|---:|---:|---:|---:|---:|
| grid_only | 50,747 | **50,095** | 50,263 | 50,312 | 48,599 | +0.4% |
| grid_solar | 49,959 | **49,307** | 49,476 | 49,525 | 47,811 | +0.4% |
| grid_wind | 38,545 | **37,893** | 38,061 | 38,110 | 36,397 | +0.6% |
| grid_solar_wind | 37,757 | **37,105** | 37,274 | 37,323 | 35,609 | +0.6% |
| grid_gas | 50,747 | 48,550 | **48,531** | 50,018 | 46,152 | +3.0% |
| grid_solar_gas | 49,959 | 47,762 | **47,743** | 49,230 | 45,364 | +3.1% |
| grid_wind_gas | 38,545 | 36,348 | **36,329** | 37,816 | 33,950 | +4.0% |
| grid_solar_wind_gas | 37,757 | 35,560 | **35,541** | 37,028 | 33,162 | +4.1% |
| grid_solar_battery | 49,959 | **47,352** | 48,844 | 49,525 | 45,207 | +4.6% |
| grid_wind_battery | 38,545 | **35,590** | 37,428 | 38,110 | 33,843 | +7.1% |
| grid_solar_wind_battery | 37,757 | **34,730** | 36,640 | 37,323 | 33,011 | +7.5% |
| all_sources | 37,757 | **33,593** | 34,962 | 37,028 | 31,263 | +10.2% |

**Findings.** (1) **RuleBased is the strongest non-oracle baseline** in every config except
the gas-only pairs (where Greedy edges it by <0.1%), which is why it is the headline
comparator. (2) **MPC underperforms RuleBased everywhere** (+0.4% to +10.2%), worst in
storage/all_sources — a direct consequence of running on a **persistence (naive) forecast**:
its price-vs-future signal is ~0, so its lookahead logic rarely fires (disclosed, Register
#9). (3) **The MPC weakness does not threaten the headline.** The offline oracle
(DeterministicOptimal, perfect price hindsight) upper-bounds what *any* forecast-driven
controller — including a perfect-forecast MPC — could achieve; RuleBased sits within 3–7% of
that bound, so "the heuristic is near-optimal" holds regardless of MPC forecast quality. A
TFT-forecast MPC arm is future work.

## R.3 Main benchmark: does RL beat the heuristic? (Table R1)

**Table R1.** Held-out weekly cost (USD), matched seeds, 200 test episodes. "Best RL" is
the lowest-cost of the four algorithms per config; Δ% is its improvement over RuleBased
(positive = cheaper). "Sig." = best RL significantly beats RuleBased (Holm). Last column =
RuleBased gap to oracle.

| Configuration | RuleBased | Oracle | Best RL (algo) | Δ% vs RB | Sig.? | RB→oracle gap |
|---|---:|---:|---|---:|:---:|---:|
| grid_only | 50,095 | 48,599 | 49,840 (SAC) | **+0.51** | ✅ | 2.99% |
| grid_solar | 49,307 | 47,811 | 48,984 (TD3) | **+0.66** | ✅ | 3.03% |
| grid_wind | 37,893 | 36,397 | 37,484 (PPO) | **+1.08** | ✅ | 3.95% |
| grid_solar_wind | 37,105 | 35,609 | 36,716 (SAC) | **+1.05** | ✅ | 4.03% |
| grid_gas | 48,550 | 46,152 | 48,336 (PPO) | +0.44 | ✗ | 4.94% |
| grid_solar_gas | 47,762 | 45,364 | 47,452 (PPO) | +0.65 | ✗ | 5.02% |
| grid_wind_gas | 36,348 | 33,950 | 35,975 (PPO) | +1.03 | ✗ | 6.60% |
| grid_solar_wind_gas | 35,560 | 33,162 | 35,307 (PPO) | +0.71 | ✗ | 6.74% |
| grid_solar_battery | 47,352 | 45,207 | 48,026 (PPO) | **−1.42** | ✗ (loss) | 4.53% |
| grid_wind_battery | 35,590 | 33,843 | 36,298 (PPO) | **−1.99** | ✗ (loss) | 4.91% |
| grid_solar_wind_battery | 34,730 | 33,011 | 35,565 (PPO) | **−2.40** | ✗ (loss) | 4.95% |
| all_sources | 33,593 | 31,263 | 34,148 (SAC) | **−1.65** | ✗ (loss) | 6.94% |

**Headline (honest, per D21):** the best RL controller significantly beats the RuleBased
heuristic in only **4 of 12** configurations — and all four are **renewable-only**
(grid_only, grid_solar, grid_wind, grid_solar_wind), with small margins of **0.5–1.1%**.
That is, even where RL wins, it recovers only about a quarter to a third of the 3–4%
heuristic-to-oracle headroom. In the four **gas** configurations RL is essentially
**neutral** (best-algo +0.4–1.0%, none Holm-significant). No single algorithm dominates:
SAC, TD3, and PPO each take the top slot in different configs.

## R.3.1 The storage-degradation pattern (the interesting finding)

The most consistent effect is a **failure**, not a win. In **every configuration
containing a battery** the best RL controller is **worse** than the fixed RuleBased
charge/discharge schedule, and the loss deepens as the asset mix grows:
grid_solar_battery **−1.4%**, grid_wind_battery **−2.0%**, grid_solar_wind_battery
**−2.4%**, all_sources **−1.7%**. These losses are broad across algorithms, and the
off-policy/weaker learners degrade far more (TD3 and A2C reach **−5% to −7%**). Per-algorithm
Holm tests confirm the losses are significant, not noise.

Interpretation: a simple deterministic storage rule is already close to optimal for this
diurnal price/renewable structure, and the learned policies mis-time charge/discharge
against it. **RL underperforms precisely where storage is present — the opposite of where
learning is commonly assumed to add the most value.** Combined with R.3, the take-away is
that sophistication does not beat a well-tuned heuristic in single-facility
behind-the-meter dispatch; it can actively hurt once storage introduces
inter-temporal coupling.

---

## R.4 Multi-objective control: the objective-weight Pareto sweep (Table R2)

*Reported separately from R.2–R.3 (see integrity guard). Scope: PPO, `all_sources`, the
same held-out test window, 200 episodes/seed, 5 seeds per weighting. This experiment asks
a different question — not "does RL beat the heuristic?" but "when RL **is** used, does the
reward weighting give the operator predictable control over the cost/carbon/water
tradeoff?"*

**Table R2.** Held-out weekly objectives by reward weighting (mean ± 95% CI over 5 seeds).
Weights are (cost / carbon / water / SLA).

| Weighting | Cost (USD) | Carbon (kg CO₂) | Water (m³) | SLA (viol.) |
|---|---:|---:|---:|---:|
| cost-heavy (0.7/0.1/0.1/0.1) | **33,240 ± 1,226** | 413,375 ± 14,621 | 627.2 ± 6.9 | 1.17 ± 2.44 |
| carbon-heavy (0.1/0.7/0.1/0.1) | 34,957 ± 643 | **353,817 ± 19,783** | 624.3 ± 14.7 | 1.00 ± 2.06 |
| water-heavy (0.1/0.1/0.7/0.1) | 36,336 ± 1,411 | 428,257 ± 22,939 | **613.1 ± 13.7** | 0.00 ± 0.00 |
| equal (0.3/0.3/0.3/0.1) | 35,330 ± 465 | 399,769 ± 15,864 | 619.5 ± 12.3 | 0.00 ± 0.00 |

**Findings:**
1. **The weighting knob works as designed.** Each single-objective-heavy weighting
   achieves the minimum of its own objective (bold diagonal): cost-heavy → lowest cost,
   carbon-heavy → lowest carbon, water-heavy → lowest water. All four points are mutually
   **Pareto-non-dominated** in (cost, carbon, water) space — i.e. a genuine tradeoff
   surface, not a single co-optimal point.
2. **Cost and carbon are the elastic, controllable objectives.** Relative to the balanced
   `equal` point, cost-heavy cuts cost by **−5.9%** (Welch p = 0.006) and carbon-heavy
   cuts carbon by **−11.5%** (p = 0.001). These are the levers with real authority.
3. **Water is nearly inelastic.** The water-heavy weighting reaches the lowest water but
   only ~1% below `equal`, and the difference is not significant (p = 0.36); worse, chasing
   water **significantly raises carbon (+7.1%, p = 0.025)**. Water is the weakest knob and
   trades off adversely against carbon.
4. **A cost/carbon–SLA tension emerges.** The two aggressive weightings (cost-heavy,
   carbon-heavy) incur ~1 SLA violation/week (high variance), whereas water-heavy and equal
   are clean (0). Pushing hard on cost or carbon defers load into riskier windows.

**Reconciliation with R.3 (state explicitly, do not pool):** the cost-heavy PPO reaches
$33,240 on all_sources — below the fixed-weight main-run RL result and close to the oracle
— but this is **not** evidence that RL beats the heuristic, because (a) the heuristics were
not re-run under a cost-only weighting, and (b) the low cost is bought by sacrificing
carbon, water, and SLA. The sweep characterizes RL's *internal* tradeoff behavior; it is
not a controller-vs-controller comparison.

**Figures:** `pareto_cost_carbon.png` (cost–carbon front, marker size ∝ water, 95% CI
bars) and `pareto_parallel.png` (normalized parallel coordinates across all four
objectives). Both in `results/rl_backup_20260716/pareto_analysis/`.

---

## R.4.2 Battery-sizing sensitivity of the heuristic ceiling (Table R3) [Gate 4a — DONE]

*Directly answers the "is the near-optimal heuristic a sizing artifact?" objection. Matched
protocol; RuleBased vs oracle at three battery sizes (rate = capacity/2).*

**Table R3.** RuleBased→oracle gap (%) by battery capacity.

| Config | 10 MWh | 20 MWh (design) | 40 MWh |
|---|---:|---:|---:|
| grid_solar_battery | 3.77% | 4.53% | 2.08% |
| grid_wind_battery | 4.93% | 4.91% | 1.30% |
| grid_solar_wind_battery | 5.06% | 4.95% | 1.32% |
| all_sources | 7.40% | 6.94% | 4.03% |

**Finding.** The heuristic-to-oracle gap stays modest (1.3–7.4%) across a 4× range of
battery sizing and **shrinks at 40 MWh** — i.e. with more storage the simple fixed schedule
gets *closer* to optimal, not further. The "heuristic is near-optimal" result is therefore
**not an artifact of the chosen 20 MWh sizing**. (Whether RL's storage-degradation persists
at 40 MWh is tested separately by a retrain — [PENDING: Gate 4 size40].)

## R.5 Threats to validity (cross-reference)

The results above are subject to the disclosures in the **Paper Caveat Register**
(DECISION_LOG). The ones most load-bearing for these findings:

- **Modeled power & cooling** (Register #4): load is real (replayed trace) but IT power
  (Fan et al.) and cooling (PUE(T)) are models, not metered telemetry. The absolute costs
  are therefore model outputs; the **comparative** RL-vs-heuristic conclusions are more
  robust than any single absolute figure.
- **Replayed, not calendar-real, load** (Register #3) and **flat reconstructed load**
  (D18): the trace is diurnally structured but relatively flat (~12% mean, ~25% peak),
  which limits deferral headroom — a plausible partial cause of RL's thin margins. Disclosed,
  not engineered away (anti-p-hacking, D10/D18).
- **Single facility, modeled sizing** (Register #5): the storage-degradation result is
  conditioned on the 20 MWh battery and the specific rule-based schedule; report sensitivity
  to sizing and to the heuristic's tuning.
- **ERCOT-only multi-year** (Register #6): CAISO history begins 2023, so the multi-year run
  is ERCOT-only; generalization across markets is future work.
- **Simulation, not deployment** (Register #7): standard for the field (Green-DCC,
  Sustain-Cluster), stated as scope.

---

## R.6 One-line summary for the abstract/conclusion

> On a leakage-free, real-trace single-facility benchmark, a well-tuned time-of-use
> heuristic stays within 3–7% of the offline optimum, and deep RL beats it only marginally
> (0.5–1.1%) in renewable-only dispatch, is neutral with gas, and **significantly
> underperforms it whenever a battery is present**. RL's reward weights do, however, give
> predictable and statistically significant control over the cost–carbon tradeoff (water is
> weakly controllable). Sophistication is not free, and for behind-the-meter dispatch a
> cheap heuristic is a strong, hard-to-beat baseline.
