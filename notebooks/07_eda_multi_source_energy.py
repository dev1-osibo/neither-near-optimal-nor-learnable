"""
EDA 07: Multi-Source Energy Analysis
======================================
Purpose: Explore all energy source signals before RL environment build.

Analyses:
1. Price signal distributions (ERCOT, CAISO, Gas) — volatility, extremes, patterns
2. Solar generation potential from irradiance data
3. Wind generation potential from wind speed data
4. Source complementarity (when one is low, is another high?)
5. Optimal battery sizing analysis
6. Cross-correlation between sources, demand, and external signals
7. Regime identification (distinct operating modes)
8. Gas vs Grid breakeven analysis

Output: JSON results + summary statistics for paper.
"""

import pandas as pd
import numpy as np
import json
import os
from scipy import stats

DATA_DIR = os.path.expanduser("~/optena/data")
RESULTS_DIR = os.path.expanduser("~/optena/results")
os.makedirs(RESULTS_DIR, exist_ok=True)

results = {}

print("=" * 70)
print("EDA 07: MULTI-SOURCE ENERGY ANALYSIS")
print("=" * 70)

# ============================================================
# 1. LOAD ALL DATA
# ============================================================
print("\n[1] Loading all datasets...")

# Core merged dataset (DC telemetry + weather + carbon)
merged = pd.read_csv(os.path.join(DATA_DIR, "merged_enriched_2020_2025.csv"))
merged["timestamp"] = pd.to_datetime(merged["timestamp"])
print(f"  Merged DC data: {len(merged):,} rows, {merged['timestamp'].min()} to {merged['timestamp'].max()}")

# Real LMP prices
ercot = pd.read_csv(os.path.join(DATA_DIR, "real_lmp_ERCOT_2020_2025.csv"))
ercot["timestamp"] = pd.to_datetime(ercot["timestamp"])
print(f"  ERCOT prices: {len(ercot):,} rows, {ercot['timestamp'].min()} to {ercot['timestamp'].max()}")

caiso = pd.read_csv(os.path.join(DATA_DIR, "real_lmp_CAISO_2020_2025.csv"))
caiso["timestamp"] = pd.to_datetime(caiso["timestamp"], utc=True)
caiso["timestamp"] = caiso["timestamp"].dt.tz_localize(None)  # Remove tz for join
print(f"  CAISO prices: {len(caiso):,} rows, {caiso['timestamp'].min()} to {caiso['timestamp'].max()}")

# Gas prices
gas = pd.read_csv(os.path.join(DATA_DIR, "real_gas_henry_hub_daily_2020_2025.csv"))
gas["date"] = pd.to_datetime(gas["date"])
print(f"  Gas prices: {len(gas):,} rows, {gas['date'].min()} to {gas['date'].max()}")

# ============================================================
# 2. PRICE SIGNAL ANALYSIS
# ============================================================
print("\n[2] Price Signal Analysis...")

# ERCOT stats
ercot_stats = {
    "count": len(ercot),
    "mean_usd_mwh": float(ercot["lmp_price_usd_mwh"].mean()),
    "median_usd_mwh": float(ercot["lmp_price_usd_mwh"].median()),
    "std_usd_mwh": float(ercot["lmp_price_usd_mwh"].std()),
    "min_usd_mwh": float(ercot["lmp_price_usd_mwh"].min()),
    "max_usd_mwh": float(ercot["lmp_price_usd_mwh"].max()),
    "pct_negative": float((ercot["lmp_price_usd_mwh"] < 0).mean() * 100),
    "pct_above_100": float((ercot["lmp_price_usd_mwh"] > 100).mean() * 100),
    "pct_above_500": float((ercot["lmp_price_usd_mwh"] > 500).mean() * 100),
    "pct_above_1000": float((ercot["lmp_price_usd_mwh"] > 1000).mean() * 100),
    "skewness": float(ercot["lmp_price_usd_mwh"].skew()),
    "kurtosis": float(ercot["lmp_price_usd_mwh"].kurtosis()),
}
print(f"  ERCOT: mean=${ercot_stats['mean_usd_mwh']:.1f}, median=${ercot_stats['median_usd_mwh']:.1f}, "
      f"max=${ercot_stats['max_usd_mwh']:.0f}, {ercot_stats['pct_above_100']:.2f}% above $100")

# CAISO stats
caiso_stats = {
    "count": len(caiso),
    "mean_usd_mwh": float(caiso["lmp_price_usd_mwh"].mean()),
    "median_usd_mwh": float(caiso["lmp_price_usd_mwh"].median()),
    "std_usd_mwh": float(caiso["lmp_price_usd_mwh"].std()),
    "min_usd_mwh": float(caiso["lmp_price_usd_mwh"].min()),
    "max_usd_mwh": float(caiso["lmp_price_usd_mwh"].max()),
    "pct_negative": float((caiso["lmp_price_usd_mwh"] < 0).mean() * 100),
    "pct_above_100": float((caiso["lmp_price_usd_mwh"] > 100).mean() * 100),
    "pct_above_500": float((caiso["lmp_price_usd_mwh"] > 500).mean() * 100),
    "skewness": float(caiso["lmp_price_usd_mwh"].skew()),
    "kurtosis": float(caiso["lmp_price_usd_mwh"].kurtosis()),
}
print(f"  CAISO: mean=${caiso_stats['mean_usd_mwh']:.1f}, median=${caiso_stats['median_usd_mwh']:.1f}, "
      f"max=${caiso_stats['max_usd_mwh']:.0f}, {caiso_stats['pct_negative']:.2f}% negative")

# Hourly price patterns
ercot["hour"] = ercot["timestamp"].dt.hour
ercot_hourly = ercot.groupby("hour")["lmp_price_usd_mwh"].agg(["mean", "median", "std"]).round(2)
print(f"\n  ERCOT hourly pattern (mean $/MWh):")
print(f"    Cheapest hour: {ercot_hourly['mean'].idxmin()}:00 (${ercot_hourly['mean'].min():.1f})")
print(f"    Most expensive hour: {ercot_hourly['mean'].idxmax()}:00 (${ercot_hourly['mean'].max():.1f})")
print(f"    Peak/Off-peak ratio: {ercot_hourly['mean'].max() / ercot_hourly['mean'].min():.2f}x")

caiso["hour"] = caiso["timestamp"].dt.hour
caiso_hourly = caiso.groupby("hour")["lmp_price_usd_mwh"].agg(["mean", "median", "std"]).round(2)
print(f"  CAISO hourly pattern (mean $/MWh):")
print(f"    Cheapest hour: {caiso_hourly['mean'].idxmin()}:00 (${caiso_hourly['mean'].min():.1f})")
print(f"    Most expensive hour: {caiso_hourly['mean'].idxmax()}:00 (${caiso_hourly['mean'].max():.1f})")
print(f"    Peak/Off-peak ratio: {caiso_hourly['mean'].max() / caiso_hourly['mean'].min():.2f}x")

results["price_analysis"] = {
    "ercot": ercot_stats,
    "caiso": caiso_stats,
    "ercot_hourly_mean": ercot_hourly["mean"].to_dict(),
    "caiso_hourly_mean": caiso_hourly["mean"].to_dict(),
}

# ============================================================
# 3. SOLAR GENERATION POTENTIAL
# ============================================================
print("\n[3] Solar Generation Potential...")

# Solar irradiance from merged dataset (shortwave_radiation, W/m²)
solar = merged["shortwave_radiation"].copy()
print(f"  Irradiance stats: mean={solar.mean():.1f} W/m², max={solar.max():.1f} W/m²")
print(f"  Hours with zero solar: {(solar == 0).sum()} ({(solar == 0).mean()*100:.1f}% — nighttime)")
print(f"  Hours with >500 W/m²: {(solar > 500).sum()} ({(solar > 500).mean()*100:.1f}%)")

# Model: 1 MW solar array (typical DC rooftop/parking lot)
# Panel efficiency: 18%, performance ratio: 85%
PANEL_AREA_M2 = 5556  # ~1 MW at 1000 W/m² × 18% efficiency
PANEL_EFFICIENCY = 0.18
PERFORMANCE_RATIO = 0.85

merged["solar_gen_kw"] = (solar * PANEL_AREA_M2 * PANEL_EFFICIENCY * PERFORMANCE_RATIO) / 1000
solar_gen = merged["solar_gen_kw"]

print(f"\n  1 MW solar array output:")
print(f"    Mean generation: {solar_gen.mean():.1f} kW")
print(f"    Peak generation: {solar_gen.max():.1f} kW")
print(f"    Capacity factor: {solar_gen.mean() / 1000 * 100:.1f}%")
print(f"    Annual energy: {solar_gen.sum() / 1000:.0f} MWh/year (avg over 6 years)")

# Solar by hour
merged["hour_temp"] = merged["timestamp"].dt.hour
solar_by_hour = merged.groupby("hour_temp")["solar_gen_kw"].mean()
print(f"    Peak solar hour: {solar_by_hour.idxmax()}:00 ({solar_by_hour.max():.0f} kW avg)")

results["solar"] = {
    "mean_gen_kw": float(solar_gen.mean()),
    "max_gen_kw": float(solar_gen.max()),
    "capacity_factor_pct": float(solar_gen.mean() / 1000 * 100),
    "hours_zero": int((solar == 0).sum()),
    "pct_zero": float((solar == 0).mean() * 100),
    "peak_hour": int(solar_by_hour.idxmax()),
    "by_hour": solar_by_hour.round(1).to_dict(),
}

# ============================================================
# 4. WIND GENERATION POTENTIAL
# ============================================================
print("\n[4] Wind Generation Potential...")

wind = merged["wind_speed_10m"].copy()
print(f"  Wind speed stats: mean={wind.mean():.1f} m/s, max={wind.max():.1f} m/s")

# Standard wind turbine power curve (simplified Vestas V90, 2MW)
# Cut-in: 3.5 m/s, rated: 12 m/s, cut-out: 25 m/s
def wind_power_curve(speed, rated_power_kw=2000):
    """Simplified cubic power curve for wind turbine."""
    power = np.zeros_like(speed, dtype=float)
    # Below cut-in (3.5 m/s): no power
    # Between cut-in and rated (3.5 - 12 m/s): cubic ramp
    mask_ramp = (speed >= 3.5) & (speed < 12)
    power[mask_ramp] = rated_power_kw * ((speed[mask_ramp] - 3.5) / (12 - 3.5)) ** 3
    # Between rated and cut-out (12 - 25 m/s): full rated power
    mask_rated = (speed >= 12) & (speed <= 25)
    power[mask_rated] = rated_power_kw
    # Above cut-out: shutdown
    return power

merged["wind_gen_kw"] = wind_power_curve(wind.values, rated_power_kw=2000)
wind_gen = merged["wind_gen_kw"]

print(f"\n  2 MW wind turbine output:")
print(f"    Mean generation: {wind_gen.mean():.1f} kW")
print(f"    Peak generation: {wind_gen.max():.1f} kW")
print(f"    Capacity factor: {wind_gen.mean() / 2000 * 100:.1f}%")
print(f"    Hours at zero: {(wind_gen == 0).sum()} ({(wind_gen == 0).mean()*100:.1f}%)")
print(f"    Hours at rated: {(wind_gen >= 1900).sum()} ({(wind_gen >= 1900).mean()*100:.1f}%)")

# Wind by hour
wind_by_hour = merged.groupby("hour_temp")["wind_gen_kw"].mean()
print(f"    Peak wind hour: {wind_by_hour.idxmax()}:00 ({wind_by_hour.max():.0f} kW avg)")

results["wind"] = {
    "mean_gen_kw": float(wind_gen.mean()),
    "max_gen_kw": float(wind_gen.max()),
    "capacity_factor_pct": float(wind_gen.mean() / 2000 * 100),
    "hours_zero": int((wind_gen == 0).sum()),
    "pct_zero": float((wind_gen == 0).mean() * 100),
    "peak_hour": int(wind_by_hour.idxmax()),
    "by_hour": wind_by_hour.round(1).to_dict(),
}

# ============================================================
# 5. SOURCE COMPLEMENTARITY
# ============================================================
print("\n[5] Source Complementarity...")

# When solar is low, is wind high? And vice versa
solar_norm = merged["solar_gen_kw"] / merged["solar_gen_kw"].max()
wind_norm = merged["wind_gen_kw"] / max(merged["wind_gen_kw"].max(), 1)

corr_solar_wind = solar_norm.corr(wind_norm)
print(f"  Solar-Wind correlation: {corr_solar_wind:.3f}")

# During nighttime (solar=0), what's wind availability?
night_mask = merged["solar_gen_kw"] == 0
night_wind_mean = merged.loc[night_mask, "wind_gen_kw"].mean()
day_wind_mean = merged.loc[~night_mask, "wind_gen_kw"].mean()
print(f"  Wind during night (solar=0): {night_wind_mean:.1f} kW avg")
print(f"  Wind during day (solar>0): {day_wind_mean:.1f} kW avg")

# Combined renewable availability
merged["combined_renewable_kw"] = merged["solar_gen_kw"] + merged["wind_gen_kw"]
combined_stats = merged["combined_renewable_kw"].describe()
print(f"  Combined (solar+wind): mean={combined_stats['mean']:.0f} kW, "
      f"max={combined_stats['max']:.0f} kW")
print(f"  Hours with ZERO renewable: {(merged['combined_renewable_kw'] == 0).sum()} "
      f"({(merged['combined_renewable_kw'] == 0).mean()*100:.1f}%)")

# Can renewables ever fully cover IT load?
it_load = merged["it_load_kw"]
renewable_covers = (merged["combined_renewable_kw"] >= it_load).sum()
print(f"  Hours where renewables >= IT load: {renewable_covers} "
      f"({renewable_covers/len(merged)*100:.1f}%)")

results["complementarity"] = {
    "solar_wind_correlation": float(corr_solar_wind),
    "night_wind_avg_kw": float(night_wind_mean),
    "day_wind_avg_kw": float(day_wind_mean),
    "combined_mean_kw": float(combined_stats["mean"]),
    "hours_zero_renewable": int((merged["combined_renewable_kw"] == 0).sum()),
    "pct_zero_renewable": float((merged["combined_renewable_kw"] == 0).mean() * 100),
    "hours_renewable_covers_load": int(renewable_covers),
    "pct_renewable_covers_load": float(renewable_covers / len(merged) * 100),
}

# ============================================================
# 6. GAS VS GRID BREAKEVEN
# ============================================================
print("\n[6] Gas vs Grid Breakeven Analysis...")

# Gas cost at 40% efficiency: $/MWh = price_per_mmbtu / (0.29307 * 0.40) 
# = price / 0.11723
gas["gas_cost_mwh"] = gas["gas_price_usd_mmbtu"] / 0.11723
gas_mean_cost = gas["gas_cost_mwh"].mean()
gas_median_cost = gas["gas_cost_mwh"].median()

print(f"  Gas generation cost (40% eff): mean=${gas_mean_cost:.1f}/MWh, median=${gas_median_cost:.1f}/MWh")

# How often is gas cheaper than grid?
# Merge gas daily price with ERCOT hourly
gas_daily = gas[["date", "gas_cost_mwh"]].copy()
ercot["date"] = ercot["timestamp"].dt.date.astype(str)
gas_daily["date"] = gas_daily["date"].dt.strftime("%Y-%m-%d")
ercot_with_gas = ercot.merge(gas_daily, on="date", how="inner")

gas_cheaper = (ercot_with_gas["gas_cost_mwh"] < ercot_with_gas["lmp_price_usd_mwh"]).mean() * 100
print(f"  Hours gas cheaper than ERCOT grid: {gas_cheaper:.1f}%")

# When is gas most valuable? During price spikes
spike_mask = ercot_with_gas["lmp_price_usd_mwh"] > 100
if spike_mask.sum() > 0:
    gas_savings_during_spikes = (ercot_with_gas.loc[spike_mask, "lmp_price_usd_mwh"] - 
                                  ercot_with_gas.loc[spike_mask, "gas_cost_mwh"]).mean()
    print(f"  Avg savings using gas during >$100 spikes: ${gas_savings_during_spikes:.1f}/MWh")

results["gas_analysis"] = {
    "gas_mean_cost_mwh": float(gas_mean_cost),
    "gas_median_cost_mwh": float(gas_median_cost),
    "pct_hours_gas_cheaper_than_ercot": float(gas_cheaper),
    "gas_emissions_kg_co2_kwh": 0.41,
}

# ============================================================
# 7. DEMAND-SUPPLY ALIGNMENT
# ============================================================
print("\n[7] Demand-Supply Alignment...")

# When is demand highest vs when is renewable supply highest?
demand_by_hour = merged.groupby("hour_temp")["it_load_kw"].mean()
solar_by_hour = merged.groupby("hour_temp")["solar_gen_kw"].mean()
wind_by_hour = merged.groupby("hour_temp")["wind_gen_kw"].mean()

peak_demand_hour = demand_by_hour.idxmax()
peak_solar_hour = solar_by_hour.idxmax()
peak_wind_hour = wind_by_hour.idxmax()

print(f"  Peak demand hour: {peak_demand_hour}:00 ({demand_by_hour.max():.0f} kW)")
print(f"  Peak solar hour: {peak_solar_hour}:00 ({solar_by_hour.max():.0f} kW)")
print(f"  Peak wind hour: {peak_wind_hour}:00 ({wind_by_hour.max():.0f} kW)")

# Mismatch: demand minus renewable
merged["net_grid_need_kw"] = merged["it_load_kw"] + merged["cooling_load_kw"] - merged["combined_renewable_kw"]
merged["net_grid_need_kw"] = merged["net_grid_need_kw"].clip(lower=0)

net_by_hour = merged.groupby("hour_temp")["net_grid_need_kw"].mean()
print(f"  Highest net grid need: {net_by_hour.idxmax()}:00 ({net_by_hour.max():.0f} kW)")
print(f"  Lowest net grid need: {net_by_hour.idxmin()}:00 ({net_by_hour.min():.0f} kW)")

results["demand_supply"] = {
    "peak_demand_hour": int(peak_demand_hour),
    "peak_solar_hour": int(peak_solar_hour),
    "peak_wind_hour": int(peak_wind_hour),
    "demand_by_hour": demand_by_hour.round(1).to_dict(),
    "solar_by_hour": solar_by_hour.round(1).to_dict(),
    "wind_by_hour": wind_by_hour.round(1).to_dict(),
    "net_grid_by_hour": net_by_hour.round(1).to_dict(),
}

# ============================================================
# 8. BATTERY SIZING ANALYSIS
# ============================================================
print("\n[8] Battery Sizing Analysis...")

# What battery size captures the most arbitrage value?
# Simple analysis: for each day, what's the max price spread?
ercot["date_only"] = ercot["timestamp"].dt.date
daily_spreads = ercot.groupby("date_only")["lmp_price_usd_mwh"].agg(
    daily_min="min", daily_max="max", daily_mean="mean"
)
daily_spreads["spread"] = daily_spreads["daily_max"] - daily_spreads["daily_min"]

print(f"  Daily price spread (ERCOT):")
print(f"    Mean spread: ${daily_spreads['spread'].mean():.1f}/MWh")
print(f"    Median spread: ${daily_spreads['spread'].median():.1f}/MWh")
print(f"    Max spread: ${daily_spreads['spread'].max():.0f}/MWh")

# Battery revenue potential (simple: buy low, sell high each day)
# Assume 4 MWh battery, 90% round-trip efficiency
BATTERY_MWH = 4.0
EFFICIENCY = 0.90
daily_revenue = daily_spreads["spread"] * BATTERY_MWH * EFFICIENCY / 1000  # Convert to $k
annual_revenue = daily_revenue.mean() * 365

print(f"\n  4 MWh battery arbitrage potential (ERCOT):")
print(f"    Daily avg revenue: ${daily_revenue.mean()*1000:.0f}")
print(f"    Annual revenue: ${annual_revenue*1000:.0f}")
print(f"    (Simple buy-low/sell-high, 90% RT efficiency)")

results["battery"] = {
    "assumed_capacity_mwh": BATTERY_MWH,
    "round_trip_efficiency": EFFICIENCY,
    "daily_spread_mean_mwh": float(daily_spreads["spread"].mean()),
    "daily_spread_median_mwh": float(daily_spreads["spread"].median()),
    "daily_spread_max_mwh": float(daily_spreads["spread"].max()),
    "annual_arbitrage_revenue_usd": float(annual_revenue * 1000),
}

# ============================================================
# 9. SEASONAL PATTERNS
# ============================================================
print("\n[9] Seasonal Patterns...")

merged["month_temp"] = merged["timestamp"].dt.month
monthly_solar = merged.groupby("month_temp")["solar_gen_kw"].mean()
monthly_wind = merged.groupby("month_temp")["wind_gen_kw"].mean()
monthly_demand = merged.groupby("month_temp")["it_load_kw"].mean()
monthly_cooling = merged.groupby("month_temp")["cooling_load_kw"].mean()

print(f"  Best solar months: {monthly_solar.nlargest(3).index.tolist()} "
      f"({monthly_solar.nlargest(3).values.round(0).tolist()} kW)")
print(f"  Best wind months: {monthly_wind.nlargest(3).index.tolist()} "
      f"({monthly_wind.nlargest(3).values.round(0).tolist()} kW)")
print(f"  Highest demand months: {monthly_demand.nlargest(3).index.tolist()}")
print(f"  Highest cooling months: {monthly_cooling.nlargest(3).index.tolist()}")

results["seasonal"] = {
    "solar_by_month": monthly_solar.round(1).to_dict(),
    "wind_by_month": monthly_wind.round(1).to_dict(),
    "demand_by_month": monthly_demand.round(1).to_dict(),
    "cooling_by_month": monthly_cooling.round(1).to_dict(),
}

# ============================================================
# SAVE RESULTS
# ============================================================
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

outpath = os.path.join(RESULTS_DIR, "eda_multi_source_energy_results.json")
with open(outpath, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"  ✓ Saved: {outpath}")

print("\n" + "=" * 70)
print("EDA 07 COMPLETE — KEY FINDINGS")
print("=" * 70)
print(f"""
  PRICES:
    ERCOT: ${ercot_stats['mean_usd_mwh']:.0f}/MWh avg, extreme spikes to ${ercot_stats['max_usd_mwh']:.0f}
    CAISO: ${caiso_stats['mean_usd_mwh']:.0f}/MWh avg, negative prices {caiso_stats['pct_negative']:.1f}% of time
    Gas:   ${gas_mean_cost:.0f}/MWh avg (at 40% efficiency)
    
  RENEWABLES:
    Solar CF: {results['solar']['capacity_factor_pct']:.1f}% | Peak hour: {results['solar']['peak_hour']}:00
    Wind CF:  {results['wind']['capacity_factor_pct']:.1f}% | Peak hour: {results['wind']['peak_hour']}:00
    Combined zero hours: {results['complementarity']['pct_zero_renewable']:.1f}%
    Solar-Wind correlation: {results['complementarity']['solar_wind_correlation']:.3f}
    
  OPTIMIZATION OPPORTUNITY:
    Gas cheaper than grid: {gas_cheaper:.1f}% of hours (ERCOT)
    Peak/off-peak ratio: {ercot_hourly['mean'].max() / ercot_hourly['mean'].min():.1f}x (ERCOT)
    Battery arbitrage: ${results['battery']['annual_arbitrage_revenue_usd']:.0f}/year (4 MWh)
""")
