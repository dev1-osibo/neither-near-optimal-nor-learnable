# Data Methodology Notes — For Paper Writing & Non-Provisional Filing

## IMPORTANT: How to Describe Each Data Type

### 1. REAL Data (Weather, EIA)
**Description in paper:**
"Hourly weather data was obtained from Open-Meteo Archive API for three major US data center locations (Ashburn VA, Phoenix AZ, The Dalles OR) covering January 2020 through December 2025. Grid operational data including hourly demand, generation by fuel type, and interchange was obtained from the US Energy Information Administration Form EIA-930 Hourly Electric Grid Monitor for the PJM, ERCOT, and CAISO balancing authorities."

No caveats needed — this is government/scientific public data.

---

### 2. DERIVED Data (Carbon Intensity)
**What we did:** Real EIA generation-by-fuel data × published IPCC/EPA emission factors = hourly carbon intensity

**Description in paper:**
"Grid carbon intensity was derived from hourly generation-by-fuel data (EIA Form 930) using IPCC 2014 lifecycle emission factors [cite IPCC AR5]. This methodology is consistent with approaches used by ElectricityMaps [cite] and WattTime [cite] for real-time grid carbon tracking, and follows the EPA eGRID methodology for emission factor assignment by fuel type."

**Key point:** This is industry standard. ElectricityMaps, WattTime, Google, Microsoft all use the same approach. Not novel methodology — just applying known factors to real data.

---

### 3. CALIBRATED SYNTHETIC Data (DC Internal Telemetry)
**What we did:** Generated synthetic time series, but EVERY parameter is calibrated from real ORNL Summit measurements (8.9M rows, DOE, CC BY 4.0).

**Description in paper:**
"Internal data center telemetry was generated using a synthetic model calibrated against 8,911,312 real measurements from the ORNL Summit supercomputer [Shin et al., 2022, DOI: 10.13139/OLCF/1861393]. Specific calibration parameters include:
- GPU power draw: 39-53W idle baseline, 5.12x P95 training spike (measured)
- Thermal behavior: 0.35°C/min rise rate, 22-70°C operating range (measured)
- Node power patterns: daily/weekly/seasonal cycles matching real HPC usage patterns
- Cooling load: physically modeled as f(IT_load, ambient_temperature) per ASHRAE TC 9.9

The synthetic approach is necessitated by the proprietary nature of production data center operational telemetry. The calibration against government-published real measurements from a 13MW, 4,626-node facility ensures physical realism. This approach is consistent with prior work including DCGen [arXiv:2604.09616], the NeurIPS LC-Opt benchmark, and Google DeepMind's data center optimization studies which similarly use simulators calibrated from real operational data."

**Key points for reviewers:**
1. NEVER claim it's real production data
2. ALWAYS cite the calibration source (Shin et al., ORNL Summit)
3. State WHY we used synthetic (proprietary nature of real DC data)
4. Reference other papers that do the same thing (DCGen, LC-Opt)
5. List WHICH parameters were calibrated and to what measured values

---

### 4. Pricing Data
**Current state:** Monthly state-level retail/industrial prices from EIA (REAL).
**For hourly resolution:** Would need PJM LMP data (available from PJM.com but requires separate data agreement). For the paper, we can:
- Use monthly real prices as a baseline
- Interpolate to hourly using the time-of-use patterns visible in our demand data
- Clearly label as "hourly prices estimated from monthly EIA averages modulated by observed demand patterns"

OR register for PJM Data Miner (free): https://dataminer2.pjm.com — provides actual hourly LMP.

---

## Summary Table for Paper

| Data Source | Type | Provenance | Rows | Citation |
|------------|------|-----------|------|----------|
| Weather (3 locations) | REAL | Open-Meteo Archive API | 157,392 | open-meteo.com |
| Grid demand/generation/interchange | REAL | EIA Form 930 | 3,095,000+ | eia.gov |
| Retail/Industrial prices | REAL | EIA retail-sales | 720 | eia.gov |
| Carbon intensity (3 regions) | DERIVED | EIA fuel mix × IPCC factors | 157,145 | IPCC AR5 + EIA |
| DC telemetry | CALIBRATED SYNTHETIC | Parameters from ORNL Summit | 52,464 | Shin et al. 2022, DOI:10.13139/OLCF/1861393 |

---

## For Non-Provisional Patent Filing

The non-provisional should reference:
- "Real-world weather and grid data from [Open-Meteo, EIA] demonstrate that the system operates with publicly available signal sources"
- "System parameters are calibrated from [ORNL Summit DOE dataset]"
- "Experimental results show the fused-signal model outperforms internal-only baselines by [X%] (cite paper results)"

The patent does NOT need to justify synthetic data the way a paper does — patents describe the METHOD, not prove scientific claims. The paper handles the scientific proof; the patent handles the legal protection.

---

*Notes created: June 13, 2026*
*For use in: Paper 2 (Energy Orchestration) + Non-Provisional filing for Patent 1*
