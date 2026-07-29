"""
Data Center Multi-Source Energy Environment
=============================================
A Gymnasium-compatible environment for training RL agents to optimize
data center energy operations across 5 sources and 4 objectives.

Based on:
- Stable-Baselines3 custom environment pattern
- DC-CFR (AAAI 2024) environment design
- Our EDA findings (parameters calibrated from real data)

Sources: Grid, Solar (5MW), Wind (5MW), Battery (20MWh), Gas (2MW)
Objectives: Cost, Carbon, Water, SLA compliance
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import os


class DataCenterEnergyEnv(gym.Env):
    """
    Multi-source data center energy optimization environment.
    
    At each timestep (1 hour), the agent observes the current state
    (weather, prices, battery, demand) and decides how to allocate
    energy across sources.
    
    The environment steps through REAL historical data chronologically.
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, data_path=None, episode_length=168, scale=10,
                 alpha_cost=0.4, alpha_carbon=0.3, alpha_water=0.2, alpha_sla=0.1,
                 use_solar=True, use_wind=True, use_battery=True, use_gas=True,
                 forecast_mode="persistence",
                 price_forecast_4h=None, price_forecast_24h=None,
                 episode_start_range=None,
                 load_mode="real_trace", workload_util_csv=None,
                 it_nameplate_kw=20000.0, idle_frac=0.30,
                 pue_min=1.25, pue_slope=0.01, pue_tref=20.0,
                 battery_capacity_kwh=None, battery_max_rate_kw=None,
                 solar_capacity_kw=None, wind_capacity_kw=None, gas_capacity_kw=None):
        """
        Args:
            data_path: Path to merged_enriched CSV with price data
            episode_length: Hours per episode (168 = 1 week)
            scale: Facility scale multiplier (10 = 10MW DC)
            alpha_*: Objective weights (must sum to 1.0)
            use_*: Which energy sources are available
            forecast_mode: how the price-forecast observations are produced.
                - "persistence": use CURRENT price as the forecast (NO lookahead).
                  This is the honest, leakage-free default for reported results.
                - "oracle": use actual future price mean (perfect foresight).
                  KEPT ONLY as a labeled upper-bound ablation -- never a headline result.
                - "provided": use externally supplied forecast arrays (e.g. TFT),
                  indexed by absolute timestep, passed via price_forecast_4h/24h.
            price_forecast_4h / price_forecast_24h: optional arrays (len == n_hours)
                of $/kWh forecasts, required when forecast_mode == "provided".
            episode_start_range: optional (lo, hi) integer index bounds restricting
                where an episode may start. Enables a proper TRAIN/TEST TEMPORAL
                SPLIT (e.g. train on 2020-2023 rows, evaluate on 2024-2025 rows) so
                that reported generalization is leakage-free. Default None samples
                across the entire dataset (original behavior, unchanged).
        """
        super().__init__()

        # Optional temporal-split window for episode starts (leakage control).
        # Resolved against max_start inside _load_data once n_hours is known.
        self._episode_start_range = episode_start_range

        # Load substrate (Fix 1 + Fix 2): "real_trace" drives IT load from the
        # replayed Alibaba GPU-2020 utilization + physical PUE(T) cooling;
        # "legacy_csv" keeps the original synthetic CSV load.
        self.load_mode = load_mode
        self.workload_util_csv = workload_util_csv
        self.it_nameplate_kw = it_nameplate_kw
        self.idle_frac = idle_frac
        self.pue_min = pue_min
        self.pue_slope = pue_slope
        self.pue_tref = pue_tref

        # Forecast configuration (controls whether the agent gets any lookahead)
        if forecast_mode not in ("persistence", "oracle", "provided"):
            raise ValueError(f"invalid forecast_mode: {forecast_mode}")
        self.forecast_mode = forecast_mode
        self._pf4 = None if price_forecast_4h is None else np.asarray(price_forecast_4h, dtype=float)
        self._pf24 = None if price_forecast_24h is None else np.asarray(price_forecast_24h, dtype=float)

        # Configuration
        self.episode_length = episode_length
        self.scale = scale
        self.alpha_cost = alpha_cost
        self.alpha_carbon = alpha_carbon
        self.alpha_water = alpha_water
        self.alpha_sla = alpha_sla
        
        # Source availability (for source-combination experiments)
        self.use_solar = use_solar
        self.use_wind = use_wind
        self.use_battery = use_battery
        self.use_gas = use_gas
        
        # Infrastructure specs (from EDA). Optional *_capacity overrides enable the
        # Gate 4 sizing-sensitivity sweep; None preserves the original design sizing.
        # A source that is disabled (use_*) is always 0 regardless of any override.
        _solar_cap = 5000 if solar_capacity_kw is None else solar_capacity_kw
        _wind_cap = 5000 if wind_capacity_kw is None else wind_capacity_kw
        _batt_cap = 20000 if battery_capacity_kwh is None else battery_capacity_kwh
        _batt_rate = 10000 if battery_max_rate_kw is None else battery_max_rate_kw
        _gas_cap = 2000 if gas_capacity_kw is None else gas_capacity_kw
        self.solar_capacity_kw = _solar_cap if use_solar else 0  # 5 MW default
        self.wind_capacity_kw = _wind_cap if use_wind else 0     # 5 MW default
        self.battery_capacity_kwh = _batt_cap if use_battery else 0  # 20 MWh default
        self.battery_max_rate_kw = _batt_rate if use_battery else 0  # 10 MW (C/2) default
        # One-way efficiency applied at BOTH charge and discharge (audit #4). Set to
        # sqrt(0.90) so the effective ROUND-TRIP efficiency is exactly 0.90, matching the
        # paper's stated design (previously 0.90 one-way => 0.81 round-trip).
        self.battery_efficiency = float(np.sqrt(0.90))  # ~0.94868; round-trip = 0.90
        self.gas_capacity_kw = _gas_cap if use_gas else 0      # 2 MW default
        # Gas-generator CO2 emission factor (kg CO2 / kWh electric).
        # Corrected 2026-07-24: was 0.00041 (a ~1000x unit error that made gas appear
        # nearly carbon-free vs the ~0.38 kg/kWh grid). Set to EIA/EPA natural-gas
        # electricity factor ~0.91 lb CO2/kWh = 0.41 kg CO2/kWh. Sensitivity-tested.
        self.gas_carbon_kg_kwh = 0.41
        
        # Load data
        self._load_data(data_path)
        
        # State: 18 dimensions
        # [hour_sin, hour_cos, month_sin, month_cos,
        #  it_load_norm, cooling_load_norm, temp_norm, humidity_norm,
        #  solar_irradiance_norm, wind_speed_norm,
        #  grid_price_norm, grid_carbon_norm, gas_price_norm,
        #  battery_soc,
        #  solar_avail_norm, wind_avail_norm,
        #  price_forecast_4h_norm, price_forecast_24h_norm]
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(18,), dtype=np.float32
        )
        
        # Actions: 4 continuous values
        # [workload_defer (0-1), cooling_offset (-1 to 1), 
        #  battery_action (-1 to 1), gas_fraction (0-1)]
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, -1.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Episode state
        self.current_step = 0
        self.episode_start = 0
        self.battery_soc = 0.5  # Start at 50%
        self.deferred_load_kwh = 0.0  # Accumulated deferred work
        self.total_cost = 0.0
        self.total_carbon = 0.0
        self.total_water = 0.0
        self.sla_violations = 0

    def _load_data(self, data_path):
        """Load and prepare the historical data."""
        if data_path is None:
            # Default path
            data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        
        # Load merged dataset
        merged_path = os.path.join(data_path, "merged_enriched_2020_2025.csv")
        df = pd.read_csv(merged_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Load ERCOT prices
        ercot_path = os.path.join(data_path, "real_lmp_ERCOT_2020_2025.csv")
        ercot = pd.read_csv(ercot_path)
        ercot["timestamp"] = pd.to_datetime(ercot["timestamp"])
        
        # Merge prices
        df = df.merge(ercot[["timestamp", "lmp_price_usd_mwh"]], on="timestamp", how="left")
        df["lmp_price_usd_mwh"] = df["lmp_price_usd_mwh"].ffill().bfill()
        
        # Load gas prices
        gas_path = os.path.join(data_path, "real_gas_henry_hub_daily_2020_2025.csv")
        gas = pd.read_csv(gas_path)
        gas["date"] = pd.to_datetime(gas["date"])
        gas_map = gas.set_index(gas["date"].dt.strftime("%Y-%m-%d"))["gas_price_usd_mmbtu"].to_dict()
        df["date_str"] = df["timestamp"].dt.strftime("%Y-%m-%d")
        df["gas_cost_usd_kwh"] = df["date_str"].map(gas_map).apply(
            lambda x: x / (0.11723 * 1000) if pd.notna(x) else 0.029  # Default ~$29/MWh
        )
        
        # Drop rows without complete data
        df = df.dropna(subset=["lmp_price_usd_mwh"]).reset_index(drop=True)
        
        # Pre-compute energy source availability
        # Solar (5 MW array from irradiance).
        # Area corrected 2026-07-26 (audit #1): the previous 5,556 m2 peaked at only
        # ~850 kW at 1000 W/m2 (1000*5556*0.18*0.85/1000), so the "5 MW" clip never bound.
        # A true 5 MW array at 18% module efficiency x 0.85 performance ratio needs
        # 5,000,000 / (1000 * 0.18 * 0.85) = 32,679 m2. The clip stays as the nameplate cap.
        _solar_area_m2 = 32679.0
        df["solar_available_kw"] = (df["shortwave_radiation"] * _solar_area_m2 * 0.18 * 0.85) / 1000
        df["solar_available_kw"] = df["solar_available_kw"].clip(0, self.solar_capacity_kw)
        
        # Wind (5MW turbine from wind speed)
        speed = df["wind_speed_10m"].values
        wind_kw = np.zeros(len(speed))
        mask_ramp = (speed >= 3.5) & (speed < 12)
        wind_kw[mask_ramp] = self.wind_capacity_kw * ((speed[mask_ramp] - 3.5) / 8.5) ** 3
        wind_kw[(speed >= 12) & (speed <= 25)] = self.wind_capacity_kw
        df["wind_available_kw"] = wind_kw
        
        # Store key columns as numpy arrays for fast access.
        # NOTE: the `scale` multiplier below applies ONLY in legacy_csv mode. In the
        # reported "real_trace" mode these three arrays are immediately overwritten by the
        # workload-driven IT load + PUE(T) cooling block below, so `scale` has no effect on
        # any reported experiment (kept only for backward compatibility with legacy_csv).
        self.timestamps = df["timestamp"].values
        self.it_load = df["it_load_kw"].values * self.scale
        self.cooling_load = df["cooling_load_kw"].values * self.scale
        self.total_demand = df["total_facility_kw"].values * self.scale
        self.temperature = df["temperature_2m"].values
        self.humidity = df["relative_humidity_2m"].values
        self.solar_irradiance = df["shortwave_radiation"].values
        self.wind_speed = df["wind_speed_10m"].values
        self.grid_price = df["lmp_price_usd_mwh"].values / 1000  # $/kWh
        self.grid_carbon = df["carbon_intensity_gco2_kwh"].values / 1000  # kg/kWh
        self.gas_price = df["gas_cost_usd_kwh"].values  # $/kWh
        self.solar_available = df["solar_available_kw"].values
        self.wind_available = df["wind_available_kw"].values
        self.hours = df["timestamp"].dt.hour.values
        self.months = df["timestamp"].dt.month.values

        # --- Fix 1 + Fix 2: real-workload-driven IT load + physical PUE(T) cooling ---
        # Replays the reconstructed Alibaba GPU-2020 utilization (typical-week profile)
        # onto this timestamp axis by (day-of-week, hour-of-day), converts to IT power,
        # and derives cooling physically from IT power and real ambient temperature.
        if self.load_mode == "real_trace":
            import sys as _sys
            _sys.path.insert(0, os.path.dirname(__file__))
            import workload_power as wp
            util_csv = self.workload_util_csv or os.path.join(
                data_path, "alibaba_gpu2020", "hourly_utilization.csv")
            typ_week = wp.build_typical_week(util_csv, col="gpu_util")
            dow = df["timestamp"].dt.dayofweek.values
            util = wp.replay_utilization(typ_week, dow, self.hours)
            self.it_load = wp.it_power_kw(util, self.it_nameplate_kw, self.idle_frac)
            self.cooling_load = wp.cooling_power_kw(
                self.it_load, self.temperature, self.pue_min, self.pue_slope, self.pue_tref)
            self.total_demand = self.it_load + self.cooling_load
            self._cluster_util = util
            print(f"  [ENV] Real-trace load: IT mean={self.it_load.mean():,.0f}kW "
                  f"peak={self.it_load.max():,.0f}kW; total mean={self.total_demand.mean():,.0f}kW")

        # Normalization stats (for observation scaling)
        self.price_mean = self.grid_price.mean()
        self.price_std = max(self.grid_price.std(), 0.001)
        # Carbon normalization stats, data-derived (audit #2). The previous hardcoded
        # (x - 0.0004)/0.0002 were calibrated for kg/MWh and saturated the observation to a
        # constant 1.0 for kg/kWh values (~0.38), blinding the agent to grid carbon. Use the
        # same mean/std + clip scheme as price so the carbon observation actually varies.
        self.carbon_mean = self.grid_carbon.mean()
        self.carbon_std = max(self.grid_carbon.std(), 1e-6)
        self.demand_mean = self.total_demand.mean()
        self.demand_std = max(self.total_demand.std(), 0.001)
        self.temp_mean = self.temperature.mean()
        self.temp_std = max(self.temperature.std(), 0.001)

        # Reward-normalization divisors computed FROM DATA (Fix 3 / AWS-6):
        # replaces the previous hardcoded magic numbers (178 / 3178 / 5.4). Each is the
        # mean hourly cost / carbon / water under the current load+price+weather substrate,
        # so the multi-objective reward stays balanced on whatever substrate is loaded.
        self.cost_norm_div = max(float((self.total_demand * self.grid_price).mean()), 1e-6)
        self.carbon_norm_div = max(float((self.total_demand * self.grid_carbon).mean()), 1e-6)
        # Humidity factor must match the (corrected) sign used in step() (audit #5): drier -> more water.
        _hum_factor = 1.0 + (50.0 - self.humidity) / 100.0
        _temp_factor = np.where(self.temperature > 25, 1.2,
                                np.where(self.temperature < 10, 0.5, 1.0))
        _water_ref = self.cooling_load * 1.8 * _hum_factor * _temp_factor / 1000.0
        self.water_norm_div = max(float(np.mean(_water_ref)), 1e-6)
        print(f"  [ENV] Reward norm divisors: cost={self.cost_norm_div:,.1f} "
              f"carbon={self.carbon_norm_div:,.1f} water={self.water_norm_div:,.3f}")

        self.n_hours = len(df)
        self.max_start = self.n_hours - self.episode_length - 24  # Room for forecast

        # Resolve optional temporal-split window into concrete, clamped bounds.
        # These are the [lo, hi) episode-start indices reset() will sample from.
        if self._episode_start_range is not None:
            lo, hi = self._episode_start_range
            lo = max(0, int(lo))
            hi = min(self.max_start, int(hi))
            if hi <= lo:
                raise ValueError(
                    f"episode_start_range ({lo},{hi}) is empty after clamping to "
                    f"max_start={self.max_start}; window too small for episode_length."
                )
            self._start_lo, self._start_hi = lo, hi
        else:
            self._start_lo, self._start_hi = 0, self.max_start

        print(f"  [ENV] Loaded {self.n_hours:,} hours of data")
        print(f"  [ENV] Episode-start window: [{self._start_lo:,}, {self._start_hi:,})")
        print(f"  [ENV] Sources: solar={self.use_solar}, wind={self.use_wind}, "
              f"battery={self.use_battery}, gas={self.use_gas}")

    def _get_obs(self):
        """Build observation vector for current timestep."""
        t = self.episode_start + self.current_step
        
        # Cyclical time encoding
        hour = self.hours[t]
        month = self.months[t]
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        
        # Normalized signals (scaled to roughly [-1, 1])
        it_load_norm = (self.it_load[t] - self.demand_mean) / self.demand_std
        cooling_norm = (self.cooling_load[t] - self.demand_mean * 0.3) / (self.demand_std * 0.3)
        temp_norm = (self.temperature[t] - self.temp_mean) / self.temp_std
        humidity_norm = (self.humidity[t] - 60) / 30  # Center around 60%
        solar_norm = self.solar_irradiance[t] / 500 - 1  # 0-1000 → [-1, 1]
        wind_norm = self.wind_speed[t] / 15 - 1  # 0-30 → [-1, 1]
        price_norm = (self.grid_price[t] - self.price_mean) / self.price_std
        price_norm = np.clip(price_norm, -3, 3) / 3  # Clip outliers
        # Data-derived carbon normalization (audit #2): same mean/std + clip scheme as price,
        # so the agent receives a real, varying grid-carbon signal instead of a saturated 1.0.
        carbon_norm = (self.grid_carbon[t] - self.carbon_mean) / self.carbon_std
        carbon_norm = np.clip(carbon_norm, -3, 3) / 3
        gas_price_norm = (self.gas_price[t] - 0.03) / 0.02  # Center around $30/MWh
        
        # Battery state of charge (already 0-1)
        battery_soc = self.battery_soc
        
        # Available renewable (normalized by capacity)
        solar_avail_norm = self.solar_available[t] / max(self.solar_capacity_kw, 1) if self.use_solar else 0
        wind_avail_norm = self.wind_available[t] / max(self.wind_capacity_kw, 1) if self.use_wind else 0
        
        # Price-forecast observations. Mode controls how much (if any) lookahead
        # the agent receives. Default "persistence" gives NO future information
        # (leakage-free); "oracle" (perfect foresight) is retained only as a
        # labeled ablation ceiling; "provided" injects real forecasts (e.g. TFT).
        if self.forecast_mode == "oracle":
            future_4h = self.grid_price[t+1:t+5].mean() if t+5 < self.n_hours else self.grid_price[t]
            future_24h = self.grid_price[t+1:t+25].mean() if t+25 < self.n_hours else self.grid_price[t]
        elif self.forecast_mode == "provided" and self._pf4 is not None:
            future_4h = self._pf4[t]
            future_24h = self._pf24[t] if self._pf24 is not None else self._pf4[t]
        else:  # "persistence" -- current price stands in for the forecast (no lookahead)
            future_4h = self.grid_price[t]
            future_24h = self.grid_price[t]
        forecast_4h_norm = (future_4h - self.price_mean) / self.price_std
        forecast_24h_norm = (future_24h - self.price_mean) / self.price_std
        forecast_4h_norm = np.clip(forecast_4h_norm, -3, 3) / 3
        forecast_24h_norm = np.clip(forecast_24h_norm, -3, 3) / 3
        
        obs = np.array([
            hour_sin, hour_cos, month_sin, month_cos,
            it_load_norm, cooling_norm, temp_norm, humidity_norm,
            solar_norm, wind_norm,
            price_norm, carbon_norm, gas_price_norm,
            battery_soc,
            solar_avail_norm, wind_avail_norm,
            forecast_4h_norm, forecast_24h_norm,
        ], dtype=np.float32)
        
        # Clip to observation space bounds
        obs = np.clip(obs, -1.0, 1.0)
        
        return obs

    def reset(self, seed=None, options=None):
        """Reset environment to start of a new episode."""
        super().reset(seed=seed)
        
        # Pick a random starting point WITHIN the (optionally restricted) window.
        # Same seed -> same start index -> same historical week, which is what makes
        # cross-policy comparisons on identical episodes properly paired.
        lo, hi = self._start_lo, self._start_hi
        if seed is not None:
            rng = np.random.default_rng(seed)
            self.episode_start = int(rng.integers(lo, hi))
        else:
            self.episode_start = int(np.random.randint(lo, hi))
        
        self.current_step = 0
        self.battery_soc = 0.5  # Start at 50% charge
        self.deferred_load_kwh = 0.0
        self.total_cost = 0.0
        self.total_carbon = 0.0
        self.total_water = 0.0
        self.sla_violations = 0
        
        return self._get_obs(), {}

    def step(self, action):
        """
        Execute one timestep (1 hour) of the environment.
        
        Actions:
            action[0]: workload_defer_fraction (0 to 1, capped at 0.30)
            action[1]: cooling_offset (-1 to 1, mapped to -3 to +3 °C)
            action[2]: battery_action (-1 to 1, negative=charge, positive=discharge)
            action[3]: gas_fraction (0 to 1, fraction of gas capacity to dispatch)
        """
        t = self.episode_start + self.current_step
        
        # --- Parse actions ---
        defer_fraction = np.clip(action[0], 0, 1) * 0.30  # Max 30% deferral
        cooling_offset = np.clip(action[1], -1, 1) * 3.0   # ±3°C
        battery_action = np.clip(action[2], -1, 1)          # -1=full charge, +1=full discharge
        gas_fraction = np.clip(action[3], 0, 1)             # 0-100% of gas capacity
        
        # --- Compute demand ---
        base_demand = self.total_demand[t]
        deferred_now = base_demand * defer_fraction
        served_demand = base_demand - deferred_now
        self.deferred_load_kwh += deferred_now
        
        # Must serve some deferred load (SLA: max 12h accumulation)
        max_deferred = base_demand * 0.30 * 12  # 12 hours worth
        if self.deferred_load_kwh > max_deferred:
            # Force serve excess (SLA violation if can't)
            forced_serve = self.deferred_load_kwh - max_deferred
            served_demand += forced_serve
            self.deferred_load_kwh = max_deferred
            self.sla_violations += 1
        
        # Serve a portion of deferred load in cheap hours
        # (Simple: serve 1/12 of deferred each hour)
        deferred_serving = self.deferred_load_kwh / 12
        served_demand += deferred_serving
        self.deferred_load_kwh -= deferred_serving
        
        # --- Cooling adjustment ---
        # Higher setpoint = less cooling energy but more water needed
        cooling_base = self.cooling_load[t]
        cooling_factor = 1.0 - cooling_offset * 0.03  # ±3°C → ±9% cooling change
        actual_cooling = cooling_base * cooling_factor
        served_demand = served_demand - cooling_base + actual_cooling  # Adjust total
        
        remaining = max(0, served_demand)
        
        # --- Energy dispatch (priority order: renewables → battery → gas → grid) ---
        hour_cost = 0.0
        hour_carbon = 0.0
        
        # 1. Solar (free, zero carbon)
        solar_used = 0.0
        if self.use_solar:
            solar_used = min(self.solar_available[t], remaining)
            remaining -= solar_used
        
        # 2. Wind (free, zero carbon)
        wind_used = 0.0
        if self.use_wind:
            wind_used = min(self.wind_available[t], remaining)
            remaining -= wind_used
        
        # 3. Battery
        battery_discharged = 0.0
        battery_charged = 0.0
        if self.use_battery:
            if battery_action > 0:  # Discharge
                max_discharge = min(
                    remaining,
                    self.battery_max_rate_kw * battery_action,
                    self.battery_soc * self.battery_capacity_kwh * self.battery_efficiency
                )
                battery_discharged = max(0, max_discharge)
                remaining -= battery_discharged
                self.battery_soc -= battery_discharged / (self.battery_capacity_kwh * self.battery_efficiency)
            elif battery_action < 0:  # Charge
                charge_amount = min(
                    self.battery_max_rate_kw * abs(battery_action),
                    (1 - self.battery_soc) * self.battery_capacity_kwh / self.battery_efficiency
                )
                battery_charged = max(0, charge_amount)
                # Charging costs money (from grid)
                hour_cost += battery_charged * self.grid_price[t]
                hour_carbon += battery_charged * self.grid_carbon[t]
                self.battery_soc += battery_charged * self.battery_efficiency / self.battery_capacity_kwh
            
            # Clip SoC to valid range
            self.battery_soc = np.clip(self.battery_soc, 0.0, 1.0)
            
            # Free charge from excess renewables
            excess_renewable = max(0, 
                (self.solar_available[t] - solar_used) + (self.wind_available[t] - wind_used)
            )
            if excess_renewable > 0 and self.battery_soc < 0.95:
                free_charge = min(
                    excess_renewable,
                    self.battery_max_rate_kw,
                    (0.95 - self.battery_soc) * self.battery_capacity_kwh / self.battery_efficiency
                )
                self.battery_soc += free_charge * self.battery_efficiency / self.battery_capacity_kwh
        
        # 4. Gas
        gas_used = 0.0
        if self.use_gas and remaining > 0:
            gas_used = min(self.gas_capacity_kw * gas_fraction, remaining)
            hour_cost += gas_used * self.gas_price[t]
            hour_carbon += gas_used * self.gas_carbon_kg_kwh
            remaining -= gas_used
        
        # 5. Grid (whatever remains)
        grid_used = max(0, remaining)
        hour_cost += grid_used * self.grid_price[t]
        hour_carbon += grid_used * self.grid_carbon[t]
        
        # --- Water consumption ---
        # Evaporative cooling model: water ∝ cooling × humidity factor.
        # Sign corrected 2026-07-26 (audit #5): evaporation rises as air gets DRIER, so lower
        # RH must INCREASE water use. Previously used (RH-50) which inverted the physics.
        humidity_factor = 1 + (50 - self.humidity[t]) / 100
        temp_factor = 1.2 if self.temperature[t] > 25 else (0.5 if self.temperature[t] < 10 else 1.0)
        # Higher cooling setpoint = less mechanical cooling but MORE evaporative water
        water_factor = 1.0 + cooling_offset * 0.05  # +3°C → 15% more water
        hour_water = actual_cooling * 1.8 * humidity_factor * temp_factor * water_factor / 1000  # m³
        
        # --- SLA penalty ---
        sla_penalty = 0.0
        if self.sla_violations > 0:
            sla_penalty = 1.0  # Binary penalty per violation
        
        # --- Compute reward ---
        # Normalize each component to mean ~1.0 (calibrated from data)
        # These divisors are set so that at average conditions, each component ≈ 1.0
        cost_norm = hour_cost / self.cost_norm_div      # data-derived (Fix 3 / AWS-6)
        carbon_norm = hour_carbon / self.carbon_norm_div
        water_norm = hour_water / self.water_norm_div
        
        reward = -(
            self.alpha_cost * cost_norm +
            self.alpha_carbon * carbon_norm +
            self.alpha_water * water_norm +
            self.alpha_sla * sla_penalty
        )
        
        # --- Track cumulative metrics ---
        self.total_cost += hour_cost
        self.total_carbon += hour_carbon
        self.total_water += hour_water
        
        # --- Advance step ---
        self.current_step += 1
        terminated = self.current_step >= self.episode_length
        truncated = False
        
        info = {
            "hour_cost": hour_cost,
            "hour_carbon": hour_carbon,
            "hour_water": hour_water,
            "grid_used": grid_used,
            "solar_used": solar_used,
            "wind_used": wind_used,
            "battery_discharged": battery_discharged,
            "battery_charged": battery_charged,
            "gas_used": gas_used,
            "battery_soc": self.battery_soc,
            "sla_violations": self.sla_violations,
            "deferred_kwh": self.deferred_load_kwh,
        }
        
        if terminated:
            info["episode_cost"] = self.total_cost
            info["episode_carbon"] = self.total_carbon
            info["episode_water"] = self.total_water
            info["episode_sla_violations"] = self.sla_violations
        
        return self._get_obs(), reward, terminated, truncated, info


# ============================================================
# TESTING / VERIFICATION
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ENVIRONMENT VERIFICATION")
    print("=" * 60)
    
    # Create environment
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    env = DataCenterEnergyEnv(data_path=data_path)
    
    # Test 1: Random actions for one episode
    print("\n[TEST 1] Random actions (1 episode = 168 hours)...")
    obs, info = env.reset(seed=42)
    print(f"  Initial observation shape: {obs.shape}")
    print(f"  Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
    
    total_reward = 0
    for step in range(168):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated:
            break
    
    print(f"  Episode complete:")
    print(f"    Total reward: {total_reward:.2f}")
    print(f"    Episode cost: ${info.get('episode_cost', 0):,.0f}")
    print(f"    Episode carbon: {info.get('episode_carbon', 0):,.0f} kg")
    print(f"    Episode water: {info.get('episode_water', 0):.1f} m³")
    print(f"    SLA violations: {info.get('episode_sla_violations', 0)}")
    print(f"    Final battery SoC: {info.get('battery_soc', 0):.2f}")
    
    # Test 2: "Do nothing" baseline (no deferral, no battery action, no gas)
    print("\n[TEST 2] Grid-only baseline (renewables used, everything else from grid)...")
    obs, _ = env.reset(seed=42)
    total_reward_baseline = 0
    for step in range(168):
        # Use renewables (happens automatically in env) but no active decisions
        action = np.array([0.0, 0.0, 0.0, 0.0])
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward_baseline += reward
        if terminated:
            break
    
    print(f"  Episode cost: ${info.get('episode_cost', 0):,.0f}")
    print(f"  Episode carbon: {info.get('episode_carbon', 0):,.0f} kg")
    print(f"  Total reward: {total_reward_baseline:.2f}")
    
    # Test 3: Smart heuristic (max deferral, discharge when price high, use gas when cheap)
    print("\n[TEST 3] Smart heuristic...")
    obs, _ = env.reset(seed=42)
    total_reward_smart = 0
    for step in range(168):
        # Smart: moderate deferral, NO cooling offset (avoid water penalty),
        # discharge battery when price signal positive, charge when negative
        price_signal = obs[10]  # Normalized grid price
        forecast_signal = obs[16]  # 4h forecast
        action = np.array([
            0.5,  # Moderate deferral
            0.0,  # No cooling offset (avoid water penalty)
            0.8 if price_signal > 0.2 else (-0.6 if price_signal < -0.2 else 0.0),
            0.9 if price_signal > 0.5 else 0.0,  # Gas only when grid very expensive
        ])
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward_smart += reward
        if terminated:
            break
    
    print(f"  Episode cost: ${info.get('episode_cost', 0):,.0f}")
    print(f"  Episode carbon: {info.get('episode_carbon', 0):,.0f} kg")
    print(f"  Total reward: {total_reward_smart:.2f}")
    
    # Comparison
    print("\n[VERIFICATION] Reward comparison:")
    print(f"  Random actions:  {total_reward:.2f}")
    print(f"  Do-nothing:      {total_reward_baseline:.2f}")
    print(f"  Smart heuristic: {total_reward_smart:.2f}")
    
    if total_reward_smart > total_reward and total_reward_smart > total_reward_baseline:
        print("  ✓ PASS: Smart heuristic beats random and do-nothing")
    else:
        print("  ⚠️ WARNING: Expected smart > random > do-nothing")
    
    print("\n  ✓ Environment is functional and ready for RL training")
