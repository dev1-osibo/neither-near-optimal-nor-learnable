# Patent 1 — Methodology Review & Non-Provisional Notes

**Date:** June 14, 2026  
**Status:** Working notes — awaiting user direction

---

## METHODOLOGY REVIEW — Paper 1 (Academic Paper)

### What We Claimed (Provisional Patent) vs What We've Done

| Provisional Claim | Current Evidence | Gap? |
|---|---|---|
| Multi-signal fusion improves prediction | ✅ EDA proves 22% improvement (linear), Granger causality confirms | No gap |
| TFT architecture | ⏳ Training on AWS now | Results pending |
| Multi-quantile forecasts (P10, P50, P90) | ⏳ TFT outputs these | Results pending |
| Multi-horizon (4h, 12h, 24h) | ⏳ TFT does this | Results pending |
| Variable selection (which signals matter) | ⏳ TFT attention weights | Results pending |
| Multi-agent RL (3 agents) | ❌ Not built yet | Need Phase 3 |
| 4-objective reward (cost+carbon+water+SLA) | ❌ Not tested yet | Need Phase 3 |
| Cross-facility coordination | ❌ Not tested | Could simulate with 3 regions |
| Contextual signals (events, transit) | ⚠️ We don't have event/transit data | We have weather, price, carbon — enough for proof |
| Water consumption optimization | ⚠️ Not in our dataset | Can derive from cooling + humidity |

### Methodology Concerns for the Paper

1. **Synthetic telemetry** — We're using calibrated synthetic DC data. The DATA_METHODOLOGY_NOTES.md handles this well. Not a gap, just needs clear disclosure.

2. **Price data is synthesized** — Monthly EIA modulated by demand. Weaker than real LMP data but defensible. The paper should note: "Hourly prices estimated from monthly EIA averages modulated by observed demand patterns, following NREL standard practice."

3. **No real production DC validation** — Common limitation. Every paper in this space (DeepMind, DCGen, LC-Opt) uses simulation or calibrated synthetic. We cite this as a limitation.

4. **Missing signals from patent** — Events, transit, social media. For the paper we don't need all of them. We prove the architecture works with weather + price + carbon + internal. The patent protects the broader concept.

---

## NON-PROVISIONAL PATENT — Requirements

### Key Differences from Provisional

| Aspect | Provisional (what was filed) | Non-Provisional (what we need) |
|--------|------------------------------|--------------------------------|
| Claims | Can be informal | Must be precise, formal patent language |
| Drawings | Optional | Required (system architecture diagram minimum) |
| Specification | Can be rough | Must enable "person skilled in the art" to reproduce |
| Experimental evidence | Not required | Strengthens prosecution (not mandatory but helps) |
| Prior art distinction | Brief | Explicit, detailed differentiation |
| Filing fee | $160 (micro) | $160 (micro) + any excess claims |

### What We Need to Produce

1. **Specification** — Expanded from provisional with experimental results (TFT comparison, EDA findings)
2. **Claims** — Tighten the 12 claims, possibly add new dependent claims based on findings
3. **Drawings** — System architecture (5 layers), TFT architecture, multi-agent RL diagram, data flow
4. **Prior art analysis** — Formal section distinguishing from DeepMind cooling, Microsoft carbon-aware, etc.

### Open Questions

- Exact filing date of provisional (needed for 12-month deadline calculation)
- Whether to include TFT + RL experimental results in non-provisional
- Whether to wait for all results or file now with available evidence

---

*Notes saved: June 14, 2026*
