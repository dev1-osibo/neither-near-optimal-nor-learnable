# Data-Centre Energy-Control Claim Qualification Benchmark

This publication package accompanies **Which Data-Centre Energy-Control Claims Survive Strict Qualification?** It presents a qualification-first benchmark for asking whether controller-performance claims remain defensible after causal information boundaries, complete terminal accounting, service qualification, repeated training seeds, and comparison with a same-environment hindsight reference are enforced together.

## Why this benchmark matters

A controller can report an attractive average while failing service in another training run, exporting unfinished work beyond the evaluation horizon, depleting storage at the boundary, or benefiting from information unavailable online. The benchmark makes those failure modes visible and prevents a favorable subset from supporting a general superiority claim.

## Main findings

The primary campaign evaluated PPO, SAC, TD3, and A2C across 12 energy-system configurations and five training seeds, producing 240 learned policies on the same 103 non-overlapping held-out weeks.

- 136 of 240 learned policies met the zero-service-violation screen.
- Only 4 of 48 algorithm–configuration recipes qualified across all five seeds.
- The planned primary comparison family was incomplete, so the evidence does not license a general multiplicity-corrected RL-versus-RuleBased cost-superiority claim.
- A descriptive fixed-policy price-input ablation recovered 7.414% of the opportunity-weighted persistence-to-hindsight gap while retaining zero modeled service violations in both arms. This was not MPC and is not a real-world savings claim.
- A separate 100-policy history/budget evaluation produced 50 individually eligible policies, but all six declared cost families remained incomplete. Its 12 budget transitions contained one temporary recovery, nine persistent failures, and two regressions.

These results do not show that reinforcement learning cannot help. They show that average performance and repeatable, service-qualified evidence are different scientific claims.

## Scope

The simulated single-facility setting is ERCOT-centred and evaluates predefined combinations of grid electricity, solar, wind, battery storage, and dispatchable gas generation. The hindsight reference shares the modeled accounting but sees the complete future episode and is not deployable. Conclusions apply only to the tested algorithms, configurations, objective, service definition, battery design, five seeds, and evaluation population.

## Contents

- [Complete preprint](docs/manuscript/PREPRINT.md)
- [Non-technical project overview](docs/manuscript/NONTECHNICAL_PROJECT_ONE_PAGER.md)
- [Publication evidence map](docs/manuscript/PUBLICATION_EVIDENCE_MAP.md)
- [Eight article figures and reproduction guide](docs/manuscript/figures/README.md)
- Compact publication results, a digest-gated figure renderer, and package validation tests

The public package intentionally excludes research-operation records, review prompts, custody material, temporary execution captures, raw datasets, and trained policy files. It publishes the reader-facing article, figures, compact results, and the code and tests needed to validate and reproduce those figures.

## Release status

The 2024–2025 results are a completed retrospective reanalysis, not a new prospective confirmation sample. A separately reserved 2026 interval remains future work. The historical [v0.1.0 release](https://github.com/dev1-osibo/neither-near-optimal-nor-learnable/releases/tag/v0.1.0) establishes earlier-project provenance but does not supply the results reported here.
