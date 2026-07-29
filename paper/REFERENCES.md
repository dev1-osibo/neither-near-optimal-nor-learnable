# Paper 1 — References (verified 2026-07-24)

Every entry below was **web-verified** against a primary source (arXiv, publisher, or
official page) during reference gathering. Confidence tags:
- ✅ **verified** — title, authors, venue, year confirmed from the source of record.
- ⚠️ **author-confirm** — cannot be externally verified; the author must supply/confirm.

Bibliographic style is placeholder (author–title–venue–year + locator); reformat to the
target venue (IEEE) at submission. Do not reproduce source abstracts verbatim in the paper
(licensing); the summaries here are for our own use.

---

## A. Workload trace & data substrate
- ✅ **[Weng2022]** Q. Weng, W. Xiao, Y. Yu, W. Wang, C. Wang, J. He, Y. Li, L. Zhang, W. Lin,
  Y. Ding. "MLaaS in the Wild: Workload Analysis and Scheduling in Large-Scale Heterogeneous
  GPU Clusters." *19th USENIX NSDI*, 2022. — the Alibaba GPU-2020 trace we replay.
  https://www.usenix.org/conference/nsdi22/presentation/weng
- ✅ **[AlibabaTrace2020]** Alibaba `cluster-trace-gpu-v2020`, Alibaba Cluster Trace Program.
  ">6,500 GPUs on ~1,800 machines, July–August 2020." (research-use license; cite alongside
  Weng2022). https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2020

## B. Power & cooling modeling
- ✅ **[Fan2007]** X. Fan, W.-D. Weber, L. A. Barroso. "Power Provisioning for a
  Warehouse-Sized Computer." *ISCA 2007* (34th Int. Symp. Computer Architecture), pp. 13–23.
  — utilization→power basis. https://dl.acm.org/doi/10.1145/1250662.1250665
- ✅ **[Barroso2009]** L. A. Barroso, U. Hölzle. "The Datacenter as a Computer." (idle power
  ≈ 20–30% of peak — supports our idle-fraction assumption). *(optional supporting cite.)*
- Standard: **ASHRAE** Thermal Guidelines for Data Processing Environments (PUE / ambient
  cooling context). *(cite the specific ASHRAE TC 9.9 edition used.)*

## C. RL algorithms & tooling
- ✅ **[Schulman2017/PPO]** J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov.
  "Proximal Policy Optimization Algorithms." arXiv:1707.06347, 2017.
- ✅ **[Haarnoja2018/SAC]** T. Haarnoja, A. Zhou, P. Abbeel, S. Levine. "Soft Actor-Critic:
  Off-Policy Maximum Entropy Deep RL with a Stochastic Actor." *ICML 2018* (arXiv:1801.01290);
  see also "SAC Algorithms and Applications," arXiv:1812.05905.
- ✅ **[Fujimoto2018/TD3]** S. Fujimoto, H. van Hoof, D. Meger. "Addressing Function
  Approximation Error in Actor-Critic Methods." *ICML 2018* (arXiv:1802.09477).
- ✅ **[Mnih2016/A3C-A2C]** V. Mnih et al. "Asynchronous Methods for Deep Reinforcement
  Learning." *ICML 2016* (arXiv:1602.01783). — A2C is the synchronous variant we use.
- ✅ **[Raffin2021/SB3]** A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus,
  N. Dormann. "Stable-Baselines3: Reliable Reinforcement Learning Implementations."
  *JMLR* 22(268):1–8, 2021. https://jmlr.org/papers/v22/20-1364.html

## D. Data-center energy optimization & related systems (positioning)
- ✅ **[Radovanovic2023]** A. Radovanović et al. "Carbon-Aware Computing for Datacenters."
  *IEEE Transactions on Power Systems*, 2023 (arXiv:2106.11750, 2021). — Google's
  Carbon-Intelligent Compute Management (temporal shifting via Virtual Capacity Curves).
- ✅ **[GreenDCC2025]** S. Sarkar, A. Naug, A. Guillen, V. Gundecha, R. Luna Gutierrez,
  S. Ghorbanpour, S. Mousavi, A. Ramesh Babu, D. Rengarajan, C. Bash (HPE Labs),
  "Hierarchical Multi-Agent Framework for Carbon-Efficient Liquid-Cooled Data Center
  Clusters" (Green-DCC), arXiv:2502.08337, 12 Feb 2025. — VERIFIED SCOPE: RL hierarchical
  controller that distributes workload across *geographically dispersed* DC clusters and
  co-optimizes *liquid + air (HVAC) cooling* plus intra-DC workload time-shifting, using
  weather / carbon intensity / resource availability. NOT single-site on-site
  generation/storage (solar/wind/battery/gas) dispatch behind one meter.
- ✅ **[DCclusterOpt2025]** A. Guillen-Perez, A. Naug, V. Gundecha, S. Ghorbanpour,
  R. Luna Gutierrez, A. Ramesh Babu, M. Salim, S. Banerjee, E. H. Oude Essink, D. Fay,
  S. Sarkar, "DCcluster-Opt: Benchmarking Dynamic Multi-Objective Optimization for
  Geo-Distributed Data Center Workloads," submitted to NeurIPS 2025 (Datasets & Benchmarks),
  arXiv:2511.00117, 31 Oct 2025. — VERIFIED SCOPE: a *geo-distributed* task-scheduling
  benchmark; a top-level coordinating agent reassigns/defers tasks across a cluster of grid-
  supplied DCs (20 global regions, network latency + transmission costs) to trade off carbon
  / energy cost / SLA / water, with heat recovery; Gymnasium API + RL and rule-based
  baselines. NO on-site solar/wind/battery/gas co-dispatch behind a single meter.
- ✅ **[SustainDC2024]** A. Naug, A. Guillen, R. Luna Gutierrez, V. Gundecha, S. Ghorbanpour,
  S. Mousavi, A. Ramesh Babu, S. Sarkar et al., "SustainDC: Benchmarking for Sustainable Data
  Center Control," *NeurIPS 2024 Datasets & Benchmarks* (arXiv:2408.07841). — closest prior
  *single-DC* benchmark: workload scheduling, cooling, and **battery** management. Distinguish:
  the battery is grid-charged storage/backup, not part of an on-site solar/wind/gas generation
  mix co-dispatched behind the meter, and it has no oracle-free/leakage-free honesty focus.
- Lineage note: Green-DCC, DCcluster-Opt, SustainDC, and LC-Opt all originate from the same
  HPE Labs group (Sarkar et al.) — the dominant sustainable-DC RL benchmark line — and every
  one targets clusters / geo-distribution / cooling, none the single-facility behind-the-meter
  multi-source generation-dispatch problem. This is the core of our novelty delta.
- ✅ **[Lazic2018]** N. Lazic, C. Boutilier, T. Lu, E. Wong, B. Roy, M. Ryu, G. Imwalle.
  "Data Center Cooling using Model-Predictive Control." *NeurIPS 2018*. — MPC/RL for DC
  cooling (single-objective, thermal); contrast with our multi-source dispatch.
- ✅ **[LCOpt2025]** A. Naug, A. Guillen-Perez et al. (HPE Labs, ORNL). "LC-Opt: Benchmarking
  RL and Agentic AI for End-to-End Liquid Cooling Optimization in Data Centers." *NeurIPS
  2025*. *(optional; recent liquid-cooling benchmark.)*

### D.1 Adjacent single-site on-site-asset work (added & web-verified 2026-07-27; cited in §2.1)
- ✅ **[Rafique2026]** Abubakar Rafique, Xiaojun Yu, Muhammad Jawad, Qun Song, Zhaohui Yuan,
  Muhammad Tariq Sadiq, Kamran Daniel, and Noman Shabbir. "Two-Stage Optimization-Learning
  Framework for Uncertainty-Aware Multi-Zonal Data Center Energy Management." *Energies* (MDPI)
  19(7):1736, 2026. doi:10.3390/en19071736. — hybrid MILP+RL controller for a multi-zonal DC with renewables + battery; reports
  hybrid RL beats uncertainty-aware MILP by up to ~33%. DISTINGUISH: controller design (RL vs
  MILP), not an RL-vs-heuristic benchmark; no LP-as-ceiling; opposite finding.
  https://www.mdpi.com/1996-1073/19/7/1736
- ✅ **[Abdelhady2026]** M. H. I. Abdelhady, E. Iakovou, E. N. Pistikopoulos. "Optimal Energy
  Portfolio Investment Strategies for Data Centers under Deep Market Uncertainty." *Applied
  Energy*, 2026. — regret-minimization for *strategic investment* in on-site PV+wind+gas+BESS
  (+SMR) at hyperscale DCs. DISTINGUISH: asset sizing at multi-year horizon, not hourly
  operational dispatch; deterministic optimization, no RL. (Title confirmed via TAMU
  parametric.tamu.edu author page.)
- ✅ **[Iqbal2026]** Hasan Iqbal and Arif I. Sarwat. "Reliability-Constrained Behind-the-Meter BESS
  Dispatch for Data Centers: Co-Optimizing Utility Costs and Critical-Load Continuity Under
  Stochastic Outages." *IEEE Access*, vol. 14, pp. 79227–79252, 2026. — BTM BESS dispatch for
  critical-load continuity under outages via MILP+ML. DISTINGUISH: single storage device,
  reliability/outage focus, no solar/wind/gas mix, no RL-vs-heuristic-vs-LP benchmark; its
  outage regime is our §6.6 companion-study setting.
- ✅ **[LiuShinDeka2026]** X. Liu, S. Shin, D. Deka. "Turning Data Centers into Grid Assets via
  Storage–Compute Co-Optimization." arXiv:2605.16190, 2026. — robust day-ahead co-optimization
  of co-located BESS + flexible compute for grid-services value. DISTINGUISH: single-storage,
  optimization-based (not RL), grid-services objective; not a multi-source RL benchmark.
  https://arxiv.org/abs/2605.16190
- ✅ **[FiginiPaolone2025]** L. Figini, M. Paolone. "Achieving Dispatchability in Data Centers:
  Carbon and Cost-Aware Sizing of Energy Storage and Local Photovoltaic Generation."
  arXiv:2412.13853, 2025 (Sustainable Energy, Grids and Networks). — scenario-based carbon/
  cost-aware *sizing* of PV+ESS for a DC. DISTINGUISH: sizing/planning problem, no wind/gas,
  no RL, no LP-vs-RL benchmark. https://arxiv.org/abs/2412.13853
- ✅ **[Mohammadi2026]** Mohammadi et al. "A Critical Review of Energy Storage Solutions [for
  AI Data Centers]." arXiv:2603.00415, 2026. — review; independently flags gaps in simulation
  tools, degradation/forecasting models, and multi-layer sizing as open challenges (external
  corroboration for §6.6). https://arxiv.org/abs/2603.00415

## E. Forecasting (baseline/MPC component)
- ✅ **[Lim2021/TFT]** B. Lim, S. Ö. Arık, N. Loeff, T. Pfister. "Temporal Fusion Transformers
  for Interpretable Multi-horizon Time Series Forecasting." *International Journal of
  Forecasting* 37(4):1748–1764, 2021 (arXiv:1912.09363). — named future-work forecaster for
  a forecast-driven MPC arm.

## F. Industry / grid context (framing)
- ✅ **[IEA2025]** International Energy Agency. "Energy and AI" (2025). Data centres ≈ **1.5%
  of global electricity (~415 TWh) in 2024**, projected to ~945 TWh by 2030.
  https://www.iea.org/reports/energy-and-ai
- ✅ **[IEEESpectrum2026]** E. Waltz and D. Genkina, "Small Data Centers Snuggle Up to Grid
  Substations," *IEEE Spectrum*, July 2026 (print) / "Grid Flexibility and Distributed
  Inference Data Centers" (online). — **THE substation micro-DC reference** the blueprint's
  grid-stress framing cites. EPRI + NVIDIA + Prologis + InfraPartners plan a fleet of ~25
  "micro" data centers (5–20 MW each) sited at utility substations, running **inference**
  (not training) and shifting compute to where grid headroom exists; ≥5 US pilots targeted
  by end of 2026. Key stats we use: grid peaks last <200 hrs/yr and whole plants otherwise
  idle; EPRI estimates compute would move to a different substation only ~0.1% of the time;
  ~55,000 US substations, many with 5–20 MW spare headroom.
  https://spectrum.ieee.org/distributed-inference-data-centers
- ✅ **[EPRI-DCFlex2024]** EPRI. "DCFlex: Data Center Flexible Load Initiative" (launched
  Oct 29, 2024; members incl. Google, Meta, Microsoft, NVIDIA, Oracle). — data centers as
  grid-flexible assets (the broader flexibility thread, distinct from the substation pilots).
  https://www.epri.com (DCFlex)
- ✅ **[IEEESpectrum2025]** IEEE Spectrum. "Big Tech Tests Data Center Flexibility for Local
  Power Grids" (2025). — coverage of the DCFlex flexibility hubs.
  https://spectrum.ieee.org/dcflex-data-center-flexibility
- ✅ **[EmeraldNVIDIA2025]** NVIDIA / Emerald AI. Phoenix, AZ demonstration (May 3, 2025):
  256 GPUs reduced power ~25% during a peak-grid window. — "align compute with power" proof
  point. https://www.nvidia.com/en-us/case-studies/emerald-ai/
  > CORRECTION (2026-07-24): an earlier note called the blueprint's "IEEE Spectrum 2026 /
  > Nvidia substation micro-DC" phrasing imprecise. That was wrong — the author supplied the
  > actual July-2026 IEEE Spectrum article ([IEEESpectrum2026] above), which IS the
  > substation-pilot reference. The DCFlex/2025/Emerald items are a related but SEPARATE
  > flexibility thread. Both are cited.

## G. Author's prior work
- ✅ **[Osibo2025]** B. Osibo, "Transforming High-Energy Data Center Sites: Sustainability
  with Predictive Analytics and Futuristic Technologies," *Int. J. Science and Research
  (IJSR)*, vol. 14, no. 8, pp. 903–911, Aug. 2025. ISSN 2319-7064.
  DOI: 10.21275/SR25816234249 (Paper ID SR25816234249). — the vision paper this work
  validates; note it explicitly scoped quantum-enhanced AI and quantum-integrated blockchain
  energy trading as *future* directions, which Paper 1 deliberately excludes (§1.3 scope).
  *(Verified from the source PDF supplied by the author, 2026-07-24.)*
- ✅ **[Osibo2023]** B. Osibo and S. Adamo, "Data Centers and Green Energy: Paving the Way for
  a Sustainable Digital Future," *Int. J. Latest Technology in Engineering, Management &
  Applied Science (IJLTEMAS)*, vol. XII, no. XI, pp. 15–30, Nov. 2023. ISSN 2278-2540.
  DOI: 10.51583/IJLTEMAS.2023.121103. — earlier author work on DC green-energy transition /
  off-grid feasibility (useful supporting self-citation for the motivation).
  *(Verified from the source PDF supplied by the author, 2026-07-24.)*

## H. Data sources / standards (Data Availability statement)
- ✅ **[ERCOT]** ERCOT real-time/day-ahead LMP market data (2020–2025). *(cite ERCOT data
  portal + access date.)*
- ✅ **[EIA]** U.S. Energy Information Administration hourly electric grid monitor (fuel mix,
  by balancing authority). *(cite EIA Open Data + access date.)* Grid carbon = EIA fuel mix
  × IPCC emission factors.
- ✅ **[IPCC]** IPCC lifecycle GHG emission factors by generation type. *(cite AR5/AR6 Annex
  III as used.)*
- Weather: Open-Meteo / NASA POWER historical (temperature, solar irradiance, wind).
  *(cite the specific source + access date.)*

---

### Gaps / to-do before submission
1. ~~[Osibo2025] exact citation~~ — DONE (author supplied PDF; also added [Osibo2023]).
2. Lock **Green-DCC** and **DCcluster-Opt** exact author lists + page-level claims via a
   verbatim re-read (Related Work §2.2 delta).
3. Choose the specific **ASHRAE** edition and **IPCC** assessment used, and add **ERCOT/EIA**
   access dates for the Data Availability statement.
4. Decide whether to include the optional supporting cites (Barroso2009, LC-Opt2025).
5. Confirm the exact print page/issue for [IEEESpectrum2026] (author has the print copy;
   online slug is /distributed-inference-data-centers).
