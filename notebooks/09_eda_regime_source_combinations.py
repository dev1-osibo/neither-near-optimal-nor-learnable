"""
EDA 09: Regime-Specific Models & Source Combination Analysis
=============================================================
Deep dive into:
1. Regime-specific prediction models (does per-regime outperform global?)
2. All source combinations — which mix optimizes best?
3. Price spike analysis — what precedes extreme events?
4. Cross-regional price correlation (ERCOT vs CAISO arbitrage)
5. Renewable curtailment detection (negative prices = oversupply)
6. Temporal autocorrelation structure (how predictable is each signal?)
7. Mutual information (non-linear dependency beyond correlation)
8. Optimal scheduling windows per regime
9. Energy cost sensitivity analysis (which lever saves most $?)
10. Compound event detection (multiple signals aligning = extreme outcomes)

Output: JSON results for paper.
"""

import pandas as pd
import numpy as np
import json
import os
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.feature_selection import mutual_info_regression
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = os.path.expanduser("~/optena/data")
RESULTS_DIR = os.path.expanduser("~/optena/results")

results = {}

print("=" * 70)
print("EDA 09: REGIME-SPECIFIC MODELS & SOURCE COMBINATIONS")
print("=" * 70)

# Load data
print("\n[LOAD] Loading datasets...")
merged = pd.read_csv(os.path.join(DATA_DIR, "merged_enriched_2020_2025.csv"))
merged["timestamp"] = pd.to_datetime(merged["timestamp"])

ercot = pd.read_csv(os.path.join(DATA_DIR, "real_lmp_ERCOT_2020_2025.csv"))
ercot["timestamp"] = pd.to_datetime(ercot["timestamp"])

caiso = pd.read_csv(os.path.join(DATA_DIR, "real_lmp_CAISO_2020_2025.csv"))
caiso["timestamp"] = pd.to_datetime(caiso["timestamp"], utc=True)
caiso["timestamp"] = caiso["timestamp"].dt.tz_localize(None)

gas = pd.read_csv(os.path.join(DATA_DIR, "real_gas_henry_hub_daily_2020_2025.csv"))
gas["date"] = pd.to_datetime(gas["date"])

# Compute energy sources
PANEL_AREA = 5556
merged["solar_gen_kw"] = (merged["shortwave_radiation"] * PANEL_AREA * 0.18 * 0.85) / 1000

def wind_power(speed, rated=2000):
    p = np.zeros_like(speed, dtype=float)
    mask = (speed >= 3.5) & (speed < 12)
    p[mask] = rated * ((speed[mask] - 3.5) / 8.5) ** 3
    p[(speed >= 12) & (speed <= 25)] = rated
    return p

merged["wind_gen_kw"] = wind_power(merged["wind_speed_10m"].values)
merged["combined_renewable_kw"] = merged["solar_gen_kw"] + merged["wind_gen_kw"]

# Merge price
merged_p = merged.merge(ercot[["timestamp", "lmp_price_usd_mwh"]], on="timestamp", how="left")
merged_p["lmp_price_usd_mwh"] = merged_p["lmp_price_usd_mwh"].ffill()

# Gas cost
gas_map = gas.set_index(gas["date"].dt.strftime("%Y-%m-%d"))["gas_price_usd_mmbtu"].to_dict()
merged_p["date_str"] = merged_p["timestamp"].dt.strftime("%Y-%m-%d")
merged_p["gas_cost_mwh"] = merged_p["date_str"].map(gas_map).apply(
    lambda x: x / 0.11723 if pd.notna(x) else np.nan
)

df = merged_p.dropna(subset=["lmp_price_usd_mwh"]).copy()
df["hour"] = df["timestamp"].dt.hour
df["month"] = df["timestamp"].dt.month
df["dow"] = df["timestamp"].dt.dayofweek
print(f"  Working dataset: {len(df):,} rows")

# ============================================================
# 1. REGIME-SPECIFIC PREDICTION MODELS
# ============================================================
print("\n[1] Regime-Specific vs Global Model Comparison...")

# Build regimes using temperature + solar + wind + hour
regime_features = ["temperature_2m", "shortwave_radiation", "wind_speed_10m", "hour"]
regime_data = df[regime_features].dropna()
scaler = StandardScaler()
scaled = scaler.fit_transform(regime_data)

km = KMeans(n_clusters=4, random_state=42, n_init=10)
df.loc[regime_data.index, "regime"] = km.fit_predict(scaled)
df["regime"] = df["regime"].ffill().astype(int)

# Target: predict net demand (total - renewable)
df["net_demand_kw"] = (df["total_facility_kw"] - df["combined_renewable_kw"]).clip(lower=0)
target = "net_demand_kw"
features = ["temperature_2m", "wind_speed_10m", "shortwave_radiation", "cloud_cover",
            "it_load_kw", "carbon_intensity_gco2_kwh", "hour", "month", "dow"]
features = [f for f in features if f in df.columns]

# Global model
sub = df[features + [target]].dropna()
split = int(len(sub) * 0.8)
X_train, X_test = sub[features].values[:split], sub[features].values[split:]
y_train, y_test = sub[target].values[:split], sub[target].values[split:]

gb_global = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42, learning_rate=0.05)
gb_global.fit(X_train, y_train)
global_r2 = r2_score(y_test, gb_global.predict(X_test))
global_mape = mean_absolute_percentage_error(y_test[y_test>0], gb_global.predict(X_test)[y_test>0]) * 100
print(f"  Global model: R²={global_r2:.4f}, MAPE={global_mape:.2f}%")

# Per-regime models
regime_results = {}
regime_labels = df.loc[sub.index, "regime"].values
for r_id in range(4):
    r_mask_train = regime_labels[:split] == r_id
    r_mask_test = regime_labels[split:] == r_id
    
    if r_mask_train.sum() < 50 or r_mask_test.sum() < 20:
        continue
    
    gb_r = GradientBoostingRegressor(n_estimators=150, max_depth=4, random_state=42, learning_rate=0.05)
    gb_r.fit(X_train[r_mask_train], y_train[r_mask_train])
    pred_r = gb_r.predict(X_test[r_mask_test])
    actual_r = y_test[r_mask_test]
    
    r2_r = r2_score(actual_r, pred_r)
    mape_r = mean_absolute_percentage_error(actual_r[actual_r>0], pred_r[actual_r>0]) * 100
    
    # Compare to global model on same subset
    global_pred_r = gb_global.predict(X_test[r_mask_test])
    r2_global_on_regime = r2_score(actual_r, global_pred_r)
    
    improvement = ((r2_r - r2_global_on_regime) / abs(r2_global_on_regime)) * 100 if r2_global_on_regime != 0 else 0
    
    regime_results[f"regime_{r_id}"] = {
        "n_train": int(r_mask_train.sum()),
        "n_test": int(r_mask_test.sum()),
        "r2_regime_model": float(r2_r),
        "r2_global_on_regime": float(r2_global_on_regime),
        "mape_regime": float(mape_r),
        "improvement_pct": float(improvement),
    }
    better = "BETTER" if improvement > 0 else "WORSE"
    print(f"  Regime {r_id}: R²={r2_r:.4f} vs global={r2_global_on_regime:.4f} "
          f"({improvement:+.1f}% {better}, n={r_mask_test.sum()})")

results["regime_models"] = {
    "global_r2": float(global_r2),
    "global_mape": float(global_mape),
    "per_regime": regime_results,
}

# ============================================================
# 2. SOURCE COMBINATION ANALYSIS
# ============================================================
print("\n[2] Source Combination Analysis...")
print("  Testing: which energy source mix minimizes cost?")

# Simulate hourly cost for each source combination
# Assumptions: facility needs total_facility_kw each hour
# Sources: Grid (at ERCOT LMP), Solar (free), Wind (free), Gas (at gas_cost), Battery (stored)

facility_demand = df["total_facility_kw"].values
grid_price = df["lmp_price_usd_mwh"].values / 1000  # $/kWh
solar_avail = df["solar_gen_kw"].values
wind_avail = df["wind_gen_kw"].values
gas_cost_kwh = df["gas_cost_mwh"].values / 1000  # $/kWh (where available)
gas_cost_kwh = np.nan_to_num(gas_cost_kwh, nan=0.03)  # Default $30/MWh

# Carbon emissions per kWh
GRID_CARBON = df["carbon_intensity_gco2_kwh"].values / 1000  # kg/kWh
SOLAR_CARBON = 0.0  # Zero marginal
WIND_CARBON = 0.0
GAS_CARBON = 0.00041  # 0.41 kg/kWh

source_combinations = {
    "grid_only": {"solar": False, "wind": False, "gas": False, "battery": False},
    "grid_solar": {"solar": True, "wind": False, "gas": False, "battery": False},
    "grid_wind": {"solar": False, "wind": True, "gas": False, "battery": False},
    "grid_gas": {"solar": False, "wind": False, "gas": True, "battery": False},
    "grid_solar_wind": {"solar": True, "wind": True, "gas": False, "battery": False},
    "grid_solar_battery": {"solar": True, "wind": False, "gas": False, "battery": True},
    "grid_wind_battery": {"solar": False, "wind": True, "gas": False, "battery": True},
    "grid_solar_gas": {"solar": True, "wind": False, "gas": True, "battery": False},
    "grid_wind_gas": {"solar": False, "wind": True, "gas": True, "battery": False},
    "grid_solar_wind_battery": {"solar": True, "wind": True, "gas": False, "battery": True},
    "grid_solar_wind_gas": {"solar": True, "wind": True, "gas": True, "battery": False},
    "all_sources": {"solar": True, "wind": True, "gas": True, "battery": True},
}

combo_results = {}
BATTERY_CAP_KWH = 4000  # 4 MWh
BATTERY_MAX_RATE = 1900  # 1.9 MW charge/discharge
BATTERY_EFF = 0.90

for combo_name, sources in source_combinations.items():
    hourly_cost = np.zeros(len(df))
    hourly_carbon = np.zeros(len(df))
    remaining_demand = facility_demand.copy()
    battery_soc = BATTERY_CAP_KWH * 0.5  # Start at 50%
    
    for t in range(len(df)):
        demand = remaining_demand[t]
        
        # Use solar first (free, zero carbon)
        if sources["solar"]:
            solar_used = min(solar_avail[t], demand)
            demand -= solar_used
            hourly_carbon[t] += solar_used * SOLAR_CARBON
        
        # Use wind (free, zero carbon)
        if sources["wind"]:
            wind_used = min(wind_avail[t], demand)
            demand -= wind_used
            hourly_carbon[t] += wind_used * WIND_CARBON
        
        # Battery discharge if grid is expensive
        if sources["battery"] and demand > 0:
            if grid_price[t] > 0.05:  # Discharge when grid > $50/MWh
                discharge = min(demand, BATTERY_MAX_RATE, battery_soc * BATTERY_EFF)
                demand -= discharge
                battery_soc -= discharge / BATTERY_EFF
        
        # Use gas if cheaper than grid
        if sources["gas"] and demand > 0:
            if gas_cost_kwh[t] < grid_price[t]:
                gas_used = min(demand, 2000)  # 2 MW gas capacity
                demand -= gas_used
                hourly_cost[t] += gas_used * gas_cost_kwh[t]
                hourly_carbon[t] += gas_used * GAS_CARBON
        
        # Remaining from grid
        hourly_cost[t] += demand * grid_price[t]
        hourly_carbon[t] += demand * GRID_CARBON[t]
        
        # Charge battery when grid is cheap
        if sources["battery"]:
            if grid_price[t] < 0.03 and battery_soc < BATTERY_CAP_KWH * 0.9:
                charge = min(BATTERY_MAX_RATE, BATTERY_CAP_KWH - battery_soc)
                battery_soc += charge * BATTERY_EFF
                hourly_cost[t] += charge * grid_price[t]
    
    total_cost = hourly_cost.sum()
    total_carbon = hourly_carbon.sum()
    
    combo_results[combo_name] = {
        "total_cost_usd": float(total_cost),
        "annual_cost_usd": float(total_cost / 6),  # 6 years of data
        "total_carbon_kg": float(total_carbon),
        "annual_carbon_kg": float(total_carbon / 6),
    }

# Normalize to grid_only baseline
baseline_cost = combo_results["grid_only"]["total_cost_usd"]
baseline_carbon = combo_results["grid_only"]["total_carbon_kg"]

print(f"\n  Baseline (grid only): ${baseline_cost/6:,.0f}/year, {baseline_carbon/6:,.0f} kg CO2/year")
print(f"\n  Source combination results (vs grid-only baseline):")
for combo, vals in sorted(combo_results.items(), key=lambda x: x[1]["total_cost_usd"]):
    cost_saving = (1 - vals["total_cost_usd"] / baseline_cost) * 100
    carbon_saving = (1 - vals["total_carbon_kg"] / baseline_carbon) * 100
    print(f"    {combo:30s}: cost {cost_saving:+.1f}%, carbon {carbon_saving:+.1f}% "
          f"(${vals['annual_cost_usd']:,.0f}/yr)")

results["source_combinations"] = combo_results
results["source_combinations_baseline"] = {
    "grid_only_annual_cost": float(baseline_cost / 6),
    "grid_only_annual_carbon": float(baseline_carbon / 6),
}

# ============================================================
# 3. PRICE SPIKE PRECURSOR ANALYSIS
# ============================================================
print("\n[3] Price Spike Precursor Analysis...")
print("  What signals precede extreme price events?")

# Define spike: price > $200/MWh (top ~3% in ERCOT)
spike_threshold = 200
df["is_spike"] = df["lmp_price_usd_mwh"] > spike_threshold
n_spikes = df["is_spike"].sum()
print(f"  Spikes (>{spike_threshold} $/MWh): {n_spikes} hours ({df['is_spike'].mean()*100:.2f}%)")

# Look at conditions 1-6 hours BEFORE a spike
precursor_vars = ["temperature_2m", "wind_speed_10m", "shortwave_radiation",
                  "it_load_kw", "carbon_intensity_gco2_kwh", "combined_renewable_kw"]
precursor_vars = [v for v in precursor_vars if v in df.columns]

precursor_results = {}
for lag in [1, 2, 3, 6]:
    # Shift spike indicator back by lag hours
    df[f"spike_ahead_{lag}h"] = df["is_spike"].shift(-lag)
    
    for var in precursor_vars:
        pre_spike = df.loc[df[f"spike_ahead_{lag}h"] == True, var]
        normal = df.loc[df[f"spike_ahead_{lag}h"] == False, var]
        
        if len(pre_spike) < 10:
            continue
        
        # T-test: are pre-spike values different from normal?
        t_stat, p_val = stats.ttest_ind(pre_spike.dropna(), normal.dropna(), equal_var=False)
        
        key = f"{var}_lag{lag}h"
        precursor_results[key] = {
            "pre_spike_mean": float(pre_spike.mean()),
            "normal_mean": float(normal.mean()),
            "difference_pct": float((pre_spike.mean() - normal.mean()) / normal.mean() * 100),
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "significant": p_val < 0.001,
        }

# Print significant precursors
print(f"\n  Significant precursors (p < 0.001):")
sig_precursors = {k: v for k, v in precursor_results.items() if v["significant"]}
for key in sorted(sig_precursors.keys()):
    v = sig_precursors[key]
    direction = "HIGHER" if v["difference_pct"] > 0 else "LOWER"
    print(f"    {key}: {direction} by {abs(v['difference_pct']):.1f}% before spike (p={v['p_value']:.2e})")

results["spike_precursors"] = precursor_results

# ============================================================
# 4. CROSS-REGIONAL PRICE CORRELATION (ARBITRAGE)
# ============================================================
print("\n[4] Cross-Regional Price Correlation...")

# Merge ERCOT and CAISO on timestamp
cross_regional = ercot.merge(
    caiso[["timestamp", "lmp_price_usd_mwh"]],
    on="timestamp", how="inner", suffixes=("_ercot", "_caiso")
)
print(f"  Overlapping hours (ERCOT + CAISO): {len(cross_regional):,}")

if len(cross_regional) > 100:
    corr = cross_regional["lmp_price_usd_mwh_ercot"].corr(cross_regional["lmp_price_usd_mwh_caiso"])
    print(f"  Price correlation: {corr:.3f}")
    
    # Arbitrage: when one is cheap and other is expensive
    spread = cross_regional["lmp_price_usd_mwh_ercot"] - cross_regional["lmp_price_usd_mwh_caiso"]
    print(f"  Price spread (ERCOT - CAISO): mean=${spread.mean():.1f}, std=${spread.std():.1f}")
    print(f"  Hours ERCOT cheaper: {(spread < 0).mean()*100:.1f}%")
    print(f"  Hours CAISO cheaper: {(spread > 0).mean()*100:.1f}%")
    print(f"  Arbitrage potential (|spread| > $50): {(spread.abs() > 50).mean()*100:.1f}% of hours")
    
    # Hourly pattern of spread
    cross_regional["hour"] = cross_regional["timestamp"].dt.hour
    hourly_spread = cross_regional.groupby("hour")[["lmp_price_usd_mwh_ercot", "lmp_price_usd_mwh_caiso"]].mean()
    
    results["cross_regional"] = {
        "overlap_hours": len(cross_regional),
        "price_correlation": float(corr),
        "spread_mean": float(spread.mean()),
        "spread_std": float(spread.std()),
        "pct_ercot_cheaper": float((spread < 0).mean() * 100),
        "pct_large_spread": float((spread.abs() > 50).mean() * 100),
        "hourly_ercot": hourly_spread["lmp_price_usd_mwh_ercot"].to_dict(),
        "hourly_caiso": hourly_spread["lmp_price_usd_mwh_caiso"].to_dict(),
    }
else:
    print("  Insufficient overlap for cross-regional analysis")
    results["cross_regional"] = {"overlap_hours": len(cross_regional)}

# ============================================================
# 5. NEGATIVE PRICE ANALYSIS (RENEWABLE CURTAILMENT)
# ============================================================
print("\n[5] Negative Price / Renewable Curtailment Analysis...")

# CAISO has negative prices when solar oversupply
neg_caiso = caiso[caiso["lmp_price_usd_mwh"] < 0].copy()
print(f"  CAISO negative price hours: {len(neg_caiso)} ({len(neg_caiso)/len(caiso)*100:.1f}%)")

if len(neg_caiso) > 0:
    neg_caiso["hour"] = neg_caiso["timestamp"].dt.hour
    neg_by_hour = neg_caiso.groupby("hour").size()
    print(f"  Negative price hours by time of day:")
    peak_neg_hours = neg_by_hour.nlargest(5)
    for h, count in peak_neg_hours.items():
        print(f"    {h}:00 — {count} occurrences")
    
    results["negative_prices"] = {
        "caiso_count": len(neg_caiso),
        "caiso_pct": float(len(neg_caiso) / len(caiso) * 100),
        "mean_negative_price": float(neg_caiso["lmp_price_usd_mwh"].mean()),
        "min_price": float(neg_caiso["lmp_price_usd_mwh"].min()),
        "by_hour": neg_by_hour.to_dict(),
        "peak_hours": peak_neg_hours.index.tolist(),
    }
    print(f"  → This means: during solar peak, grid PAYS you to consume!")
    print(f"  → Battery strategy: charge for FREE (or get paid) during these hours")

# ============================================================
# 6. TEMPORAL AUTOCORRELATION
# ============================================================
print("\n[6] Temporal Autocorrelation Structure...")

# How predictable is each signal from its own history?
auto_vars = ["lmp_price_usd_mwh", "solar_gen_kw", "wind_gen_kw", 
             "it_load_kw", "cooling_load_kw", "temperature_2m"]
auto_vars = [v for v in auto_vars if v in df.columns]

autocorr_results = {}
lags_to_test = [1, 2, 3, 6, 12, 24, 48, 168]  # hours

for var in auto_vars:
    series = df[var].dropna()
    acorrs = {}
    for lag in lags_to_test:
        if lag < len(series):
            acorrs[lag] = float(series.autocorr(lag))
    autocorr_results[var] = acorrs
    # Find lag where autocorrelation drops below 0.5
    half_life = next((lag for lag, ac in sorted(acorrs.items()) if ac < 0.5), ">168h")
    print(f"  {var:30s}: AC(1h)={acorrs.get(1,0):.3f}, AC(24h)={acorrs.get(24,0):.3f}, "
          f"half-life={half_life}")

results["autocorrelation"] = autocorr_results

# ============================================================
# 7. MUTUAL INFORMATION (Non-linear dependencies)
# ============================================================
print("\n[7] Mutual Information Analysis...")

# MI captures non-linear dependencies that correlation misses
mi_target = "lmp_price_usd_mwh"
mi_features = ["temperature_2m", "wind_speed_10m", "shortwave_radiation",
               "cloud_cover", "it_load_kw", "carbon_intensity_gco2_kwh",
               "solar_gen_kw", "wind_gen_kw", "hour", "month"]
mi_features = [f for f in mi_features if f in df.columns]

mi_data = df[mi_features + [mi_target]].dropna()
if len(mi_data) > 10000:
    mi_sample = mi_data.sample(10000, random_state=42)
else:
    mi_sample = mi_data

X_mi = mi_sample[mi_features].values
y_mi = mi_sample[mi_target].values

mi_scores = mutual_info_regression(X_mi, y_mi, random_state=42)
mi_dict = dict(zip(mi_features, mi_scores))
mi_sorted = sorted(mi_dict.items(), key=lambda x: x[1], reverse=True)

print(f"  Mutual Information with price (non-linear dependency):")
for feat, mi in mi_sorted:
    bar = "█" * int(mi / max(mi_scores) * 30)
    print(f"    {feat:35s}: MI={mi:.4f} {bar}")

results["mutual_information"] = {
    "target": mi_target,
    "scores": {k: float(v) for k, v in mi_sorted},
}

# ============================================================
# 8. COMPOUND EVENT DETECTION
# ============================================================
print("\n[8] Compound Event Detection...")
print("  When multiple adverse signals align simultaneously")

# Define adverse conditions
df["high_temp_flag"] = df["temperature_2m"] > df["temperature_2m"].quantile(0.9)
df["low_renewable_flag"] = df["combined_renewable_kw"] < df["combined_renewable_kw"].quantile(0.1)
df["high_demand_flag"] = df["it_load_kw"] > df["it_load_kw"].quantile(0.9)
df["high_carbon_flag"] = df["carbon_intensity_gco2_kwh"] > df["carbon_intensity_gco2_kwh"].quantile(0.9)

# Count compound events
df["n_adverse"] = (df["high_temp_flag"].astype(int) + 
                   df["low_renewable_flag"].astype(int) +
                   df["high_demand_flag"].astype(int) +
                   df["high_carbon_flag"].astype(int))

compound_results = {}
for n in range(0, 5):
    mask = df["n_adverse"] == n
    count = mask.sum()
    if count > 0:
        avg_price = df.loc[mask, "lmp_price_usd_mwh"].mean()
        p95_price = df.loc[mask, "lmp_price_usd_mwh"].quantile(0.95)
        compound_results[n] = {
            "count": int(count),
            "pct": float(count / len(df) * 100),
            "avg_price": float(avg_price),
            "p95_price": float(p95_price),
        }
        print(f"  {n} adverse signals: {count:,} hours ({count/len(df)*100:.1f}%), "
              f"avg price=${avg_price:.1f}, p95=${p95_price:.0f}")

results["compound_events"] = compound_results

# ============================================================
# 9. ENERGY COST SENSITIVITY ANALYSIS
# ============================================================
print("\n[9] Energy Cost Sensitivity Analysis...")
print("  Which lever saves the most money?")

# Calculate savings from each intervention independently
baseline_annual_cost = combo_results["grid_only"]["annual_cost_usd"]

interventions = {
    "Add 1MW Solar": combo_results["grid_solar"]["annual_cost_usd"],
    "Add 2MW Wind": combo_results["grid_wind"]["annual_cost_usd"],
    "Add 2MW Gas": combo_results["grid_gas"]["annual_cost_usd"],
    "Add Solar+Wind": combo_results["grid_solar_wind"]["annual_cost_usd"],
    "Add Solar+Battery": combo_results["grid_solar_battery"]["annual_cost_usd"],
    "Add Wind+Battery": combo_results["grid_wind_battery"]["annual_cost_usd"],
    "Add All Sources": combo_results["all_sources"]["annual_cost_usd"],
}

print(f"\n  Baseline annual cost: ${baseline_annual_cost:,.0f}")
sensitivity = {}
for name, cost in sorted(interventions.items(), key=lambda x: x[1]):
    saving = baseline_annual_cost - cost
    pct = saving / baseline_annual_cost * 100
    sensitivity[name] = {"annual_cost": float(cost), "saving_usd": float(saving), "saving_pct": float(pct)}
    print(f"    {name:25s}: ${cost:,.0f}/yr (saves ${saving:,.0f}, {pct:.1f}%)")

results["cost_sensitivity"] = sensitivity

# ============================================================
# 10. OPTIMAL SCHEDULING WINDOWS
# ============================================================
print("\n[10] Optimal Scheduling Windows by Regime...")

# For each regime, when is the cheapest time to run workloads?
for regime_id in range(4):
    r_mask = df["regime"] == regime_id
    r_data = df[r_mask]
    if len(r_data) < 100:
        continue
    
    hourly_price = r_data.groupby("hour")["lmp_price_usd_mwh"].mean()
    cheapest_hours = hourly_price.nsmallest(4).index.tolist()
    expensive_hours = hourly_price.nlargest(4).index.tolist()
    
    hourly_renewable = r_data.groupby("hour")["combined_renewable_kw"].mean()
    best_renewable_hours = hourly_renewable.nlargest(4).index.tolist()
    
    print(f"\n  Regime {regime_id} (n={len(r_data):,}):")
    print(f"    Cheapest grid hours: {cheapest_hours} (${hourly_price[cheapest_hours].mean():.0f}/MWh)")
    print(f"    Most expensive hours: {expensive_hours} (${hourly_price[expensive_hours].mean():.0f}/MWh)")
    print(f"    Best renewable hours: {best_renewable_hours}")

# ============================================================
# SAVE ALL RESULTS
# ============================================================
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

outpath = os.path.join(RESULTS_DIR, "eda_regime_source_combinations_results.json")
with open(outpath, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"  ✓ Saved: {outpath}")

print("\n" + "=" * 70)
print("EDA 09 COMPLETE — KEY DISCOVERIES")
print("=" * 70)
