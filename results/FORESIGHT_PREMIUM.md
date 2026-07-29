# Foresight-Premium Ablation (§5.6) -- PPO + SAC, paired test

Held-out weekly cost under no-lookahead (persistence) vs perfect price foresight (oracle), matched seeds/window. Premium = how much cheaper the learner becomes with foresight. p = paired t-test that premium != 0; sig = bootstrap 95% CI on the mean per-episode difference excludes zero.

| Algo | Config | Persist $ | Oracle $ | Premium | t-p | sig? | P<RB | O<RB |
|---|---|---:|---:|---:|---:|:--:|:--:|:--:|
| PPO | grid_only | 50,055 | 49,937 | +0.24% | 4.34e-07 | Y | y | y |
| PPO | grid_solar | 45,096 | 45,194 | -0.22% | 2.11e-06 | Y | y | y |
| PPO | grid_wind | 37,692 | 37,507 | +0.49% | 1.91e-13 | Y | y | y |
| PPO | grid_solar_wind | 32,935 | 32,825 | +0.33% | 1.99e-09 | Y | y | y |
| PPO | grid_gas | 48,926 | 48,810 | +0.24% | 0.00838 | Y | n | n |
| PPO | grid_solar_gas | 44,041 | 44,059 | -0.04% | 0.478 | n | n | n |
| PPO | grid_wind_gas | 36,338 | 36,671 | -0.92% | 6.09e-16 | Y | y | n |
| PPO | grid_solar_wind_gas | 31,746 | 31,571 | +0.55% | 3.06e-09 | Y | n | y |
| PPO | grid_solar_battery | 43,461 | 43,416 | +0.10% | 0.372 | n | n | n |
| PPO | grid_wind_battery | 35,979 | 36,048 | -0.19% | 0.294 | n | n | n |
| PPO | grid_solar_wind_battery | 30,734 | 31,064 | -1.07% | 3.82e-17 | Y | n | n |
| PPO | all_sources | 30,450 | 30,026 | +1.39% | 3.46e-06 | Y | n | n |
| SAC | grid_only | 49,869 | 49,867 | +0.00% | 0.95 | n | y | y |
| SAC | grid_solar | 44,958 | 44,963 | -0.01% | 0.818 | n | y | y |
| SAC | grid_wind | 37,617 | 37,524 | +0.25% | 0.00501 | Y | y | y |
| SAC | grid_solar_wind | 33,001 | 33,081 | -0.24% | 7.59e-05 | Y | y | y |
| SAC | grid_gas | 49,065 | 49,055 | +0.02% | 0.85 | n | n | n |
| SAC | grid_solar_gas | 44,158 | 44,178 | -0.05% | 0.674 | n | n | n |
| SAC | grid_wind_gas | 36,258 | 36,445 | -0.52% | 0.000158 | Y | y | n |
| SAC | grid_solar_wind_gas | 31,649 | 31,598 | +0.16% | 0.272 | n | y | y |
| SAC | grid_solar_battery | 44,115 | 44,026 | +0.20% | 0.259 | n | n | n |
| SAC | grid_wind_battery | 36,033 | 37,113 | -3.00% | 8.31e-19 | Y | n | n |
| SAC | grid_solar_wind_battery | 31,159 | 31,699 | -1.73% | 4.16e-06 | Y | n | n |
| SAC | all_sources | 30,912 | 30,778 | +0.43% | 0.145 | n | n | n |

**PPO** mean premium +0.08% (range -1.07..+1.39); significant in 9/12 configs.

**SAC** mean premium -0.37% (range -3.00..+0.43); significant in 5/12 configs.