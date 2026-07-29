# True-Optimum Ceiling (LP) -- Paper 1 §5.2 (audit #3)

RuleBased and best-RL weekly cost vs the genuine per-episode LP optimum (perfect foresight). Gap = how far above the true optimum. LP replaces the old quantile 'DeterministicOptimal' heuristic.

| Config | RuleBased $ | Best RL $ (algo) | LP optimum $ | RB→opt | RL→opt | RL vs RB |
|---|---:|---|---:|---:|---:|---:|
| grid_only | 50,095 | 49,869 (SAC) | 47,091 | 6.0% | 5.6% | +0.45%* |
| grid_solar | 45,462 | 44,958 (SAC) | 42,457 | 6.6% | 5.6% | +1.11%* |
| grid_wind | 37,893 | 37,617 (SAC) | 34,889 | 7.9% | 7.3% | +0.73%* |
| grid_gas | 48,550 | 48,926 (PPO) | 43,844 | 9.7% | 10.4% | -0.78% |
| grid_solar_wind | 33,263 | 32,736 (TD3) | 30,265 | 9.0% | 7.5% | +1.58%* |
| grid_solar_battery | 42,919 | 43,461 (PPO) | 34,199 | 20.3% | 21.3% | -1.26% |
| grid_wind_battery | 35,041 | 35,979 (PPO) | 27,603 | 21.2% | 23.3% | -2.68% |
| grid_solar_gas | 43,916 | 44,041 (PPO) | 39,210 | 10.7% | 11.0% | -0.28% |
| grid_wind_gas | 36,348 | 36,258 (SAC) | 31,642 | 12.9% | 12.7% | +0.25% |
| grid_solar_wind_battery | 29,834 | 30,734 (PPO) | 23,349 | 21.7% | 24.0% | -3.02% |
| grid_solar_wind_gas | 31,719 | 31,586 (TD3) | 27,112 | 14.5% | 14.2% | +0.42% |
| all_sources | 28,998 | 30,450 (PPO) | 21,725 | 25.1% | 28.7% | -5.01% |

RuleBased→optimum gap: renewable-only 7.4% (mean), storage-rich 22.1% (mean, range 20.3-25.1%).
LP<=RuleBased holds for all configs: True.