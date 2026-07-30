# Neither Near-Optimal Nor Learnable

**A Honest, Leakage-Free Benchmark for Behind-the-Meter Data-Center Energy Dispatch**

Babasola Osibo · basola.osibo@gmail.com

This repository contains the environment, controllers, offline-optimum LP solver, training/evaluation
harness, and result artifacts for the paper. Every table and figure in the paper regenerates from
the scripts here.

---

## What this is

A leakage-free, real-data benchmark for **single-facility, behind-the-meter** energy dispatch: an
hourly control problem in which a data center serves its load from a mix of grid import, on-site
solar, wind, battery, and gas against cost / carbon / water / SLA objectives. Four RL algorithms
(PPO, SAC, TD3, A2C) are compared against four non-learned baselines (DoNothing, a time-of-use
RuleBased heuristic, Greedy, and a naive-forecast MPC) across **twelve** source configurations,
under a strict 2020–2023 / 2024–2025 temporal split with no forecast lookahead — and against a
**per-episode offline LP economic-dispatch optimum** used as the offline ceiling.

## Headline findings

- On the honest protocol, the best RL controller significantly beats the heuristic in only
  **4 of 12** configurations (all renewable-only, by 0.5–1.6%) and **significantly loses in every
  storage configuration** (to −5.0% at the full asset mix).
- Perfect price foresight barely moves the learners (mean premium ≈ 0 for PPO and SAC).
- Against the offline LP optimum, **neither** controller is near-optimal where it matters: ~6–9%
  above optimum for renewable-only dispatch, but **~20–25% above** in storage-rich configurations
  — a large online-to-offline gap that persists across a fourfold battery-sizing range.
- RL's one usable affordance is reward-weight steering on **cost** (~5%); carbon/water levers are
  statistically real but physically small (~1%).

## Data provenance

Real replayed workload trace (Alibaba `cluster-trace-gpu-v2020`, Weng et al. 2022), real ERCOT
LMPs, EIA fuel-mix × IPCC carbon intensity, and measured weather (Open-Meteo / NASA POWER); IT
power (Fan et al. 2007 envelope) and cooling (temperature-dependent PUE) are physical models. Full
details in `paper/DATA_METHODOLOGY_NOTES.md`.

> **Note on the workload trace:** the raw Alibaba trace is redistributed under the Alibaba Cluster
> Trace Program research license and is **not** committed here. Obtain it from the
> [Alibaba clusterdata repo](https://github.com/alibaba/clusterdata) and run the reconstruction
> step below.

## Repository layout

```
src/
  dc_energy_env.py            # the Gym environment (substrate + dispatch + reward)
  baselines.py               # DoNothing, RuleBased, Greedy, MPC controllers
  lp_oracle.py               # per-episode offline LP economic-dispatch optimum (HiGHS)
  workload_power.py          # trace → IT power (Fan envelope) + PUE(T) cooling
  build_workload_load.py     # reconstruct hourly utilization from the raw trace
  train_rl_temporal.py       # training harness (temporal split, checkpointed)
  rl_temporal_verdict.py     # Table II + Fig 4 (RL vs RuleBased)
  rl_lp_ceiling.py           # Table Ib + Fig 3 (distance above the LP optimum)
  rl_eval_baselines_matched.py  # Table I (matched-protocol baseline costs)
  rl_foresight_premium.py    # Table IIb + Fig 5 (foresight ablation, PPO+SAC)
  rl_pareto_analysis.py      # Table III + Fig 6 (multi-objective weight sweep)
  rl_sizing_sweep_lp.py      # Table IV + Fig 7 (battery-sizing sensitivity)
  make_paper_figures.py      # renders figures from the result JSONs
  make_system_schematic.py   # Fig 1
tests/
  test_env_forecast_leakage.py  # asserts no forecast leakage in persistence mode
  test_workload_trace.py        # asserts correct utilization reconstruction
results/                     # result artifacts (JSON) + summary markdown
paper/                       # manuscript, figures, provenance & manifest
```

## Reproduce

```bash
pip install -r requirements.txt
# Windows only, so Δ/→/± print cleanly:  set PYTHONIOENCODING=utf-8
```

Regenerate the analysis artifacts and figures (from the repo root):

```bash
python src/rl_temporal_verdict.py        # Table II,  Fig 4
python src/rl_lp_ceiling.py              # Table Ib,  Fig 3
python src/rl_eval_baselines_matched.py  # Table I
python src/rl_foresight_premium.py       # Table IIb, Fig 5
python src/rl_pareto_analysis.py         # Table III, Fig 6
python src/rl_sizing_sweep_lp.py         # Table IV,  Fig 7
python src/make_paper_figures.py         # renders all figures → paper/figures/
```

Retraining from scratch (the full 445-run campaign: 240 main + 120 foresight + 80 Pareto + 5
sizing) is driven by `src/train_rl_temporal.py`; see `src/run_retrain_split.sh` for the
multi-worker runner. Each run is 1×10⁶ steps.

## Tests

```bash
pytest tests/
```

The two tests back the paper's core honesty claims: no forecast leakage in the reported
`persistence` mode, and correct workload-trace reconstruction.

## Citation

If you use this benchmark, please cite the paper (preprint link to be added). BibTeX will be
provided with the arXiv posting.
