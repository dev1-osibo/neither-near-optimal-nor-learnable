# Paper 1 — Figures & Tables manifest

Every artifact below is regenerated from the result files by `src/make_paper_figures.py` (reads
result artifacts → `paper/figures/`); tables live inline in `PAPER1_DRAFT.md`.

> Console note: the analysis scripts print `Δ`/`→`/`±`; on Windows cp1252 set
> `$env:PYTHONIOENCODING="utf-8"` before running, or ignore the cosmetic exit-1 (outputs are
> written to disk *before* the print).

## Figures (8 files in `paper/figures/`)
| Fig | File | Section | Source artifact / script | Status |
|---|---|---|---|---|
| 1 | fig1_system_schematic.png | §3 | src/make_system_schematic.py | **READY** |
| 2 | fig_load_typical_week.png | §3.2 | reconstructed Alibaba GPU-2020 typical week | **READY** |
| 3 | fig_optimality_gap.png | §5.2 (**headline**) | rl_lp_ceiling.json → make_paper_figures.py | **READY** |
| 4 | fig_verdict_delta.png | §5.3–5.4 | rl_temporal_verdict.json → make_paper_figures.py | **READY** (12 configs, 4/12 wins) |
| 5 | fig_foresight_premium.png | §5.6 | rl_foresight_premium.py (PPO+SAC, paired) | **READY** |
| 6 | fig_pareto_cost_carbon.png + fig_pareto_parallel.png | §5.7 | rl_pareto_analysis.py (20 seeds) | **READY** |
| 7 | fig_sizing_sensitivity.png | §5.8 | sizing_sensitivity_lp.json → make_paper_figures.py | **READY** (RB→true-optimum gap) |

*Note:* the headline figure is `fig_optimality_gap.png` (distance above the true LP optimum).

## Tables (inline in PAPER1_DRAFT.md)
| # | Title | Section | Status |
|---|---|---|---|
| I | Matched-protocol weekly cost (DoNothing / RuleBased / Greedy / MPC) | §5.1 | **FINAL** (RuleBased lowest-cost non-oracle; MPC = naive-forecast) |
| Ib | Distance above the **true LP optimum** (RB→opt, RL→opt, vs old quantile-oracle gap) | §5.2 | **FINAL** (renewable 6–9%; storage 20–25%) |
| II | Best RL vs RuleBased (Δ%, Holm significance) | §5.3 | **FINAL** (4/12 sig wins, all renewable-only; every battery config a sig. loss) |
| IIb | Foresight premium, PPO & SAC (paired) | §5.6 | **FINAL** (PPO +0.08%, SAC −0.37%) |
| III | Multi-objective weight sweep (20 seeds, mean ± 95% CI) | §5.7 | **FINAL** (cost −5.1% p<0.001; carbon −1.4% p<0.001; water −1.1% p=0.019) |
| IV | Sizing sensitivity — RB→**true-optimum** gap by battery capacity | §5.8 | **FINAL** (14–25% across 10/20/40 MWh; quantile-oracle contrast column) |
| V | Substrate & control parameters | §3.8 | **FINAL** (corrected: solar 32,679 m²; η 0.90; divisors 592.1/3683.7/2.638) |
| VI | Modeled parameters & sensitivity scope | §7 (Threat 8) | **FINAL** |

## Regeneration
Run from `patent1-energy-orchestration/` (set `$env:PYTHONIOENCODING="utf-8"` first):

1. **Verdict** (Table II, Fig 4): `python src/rl_temporal_verdict.py` → `results/rl_temporal_verdict.json`
2. **LP ceiling** (Table Ib, Fig 3): `python src/rl_lp_ceiling.py` → `results/rl_lp_ceiling.json`, `results/LP_CEILING.md`
3. **Baselines** (Table I): `python src/rl_eval_baselines_matched.py`
4. **Foresight** (Table IIb, Fig 5): `python src/rl_foresight_premium.py` → `results/FORESIGHT_PREMIUM.md`
5. **Pareto** (Table III, Fig 6): `python src/rl_pareto_analysis.py` → `results/PARETO_ANALYSIS.md`, `fig_pareto_cost_carbon.png`, `fig_pareto_parallel.png`
6. **Sizing** (Table IV, Fig 7): `python src/rl_sizing_sweep_lp.py` → `results/sizing_sensitivity_lp.json`
7. **All figures**: `python src/make_paper_figures.py` (reads the JSON artifacts above → `paper/figures/`)
