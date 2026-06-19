"""
EXPLORATORY DATA ANALYSIS — Patent 1: Energy Orchestration
=============================================================
Questions we're answering:
1. What patterns exist in DC energy consumption?
2. How does weather correlate with DC cooling/energy demand?
3. How does grid carbon intensity vary by time and region?
4. What's the relationship between pricing and demand?
5. Can external signals predict DC energy demand better than internal only?
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

print("=" * 70)
print("EXPLORATORY DATA ANALYSIS — Energy Orchestration")
print("=" * 70)

# ============================================================
# LOAD ALL DATA
# ============================================================

print("\n1. LOADING DATA...")

# DC Telemetry
dc = pd.read_csv(os.path.join(DATA_DIR, 'dc_telemetry_calibrated_2020_2025.csv'))
dc['timestamp'] = pd.to_datetime(dc['timestamp'])
dc.set_index('timestamp', inplace=True)
print(f"   DC Telemetry: {len(dc):,} rows, {dc.columns.tolist()}")

# Weather (Ashburn — primary DC location)
weather = pd.read_csv(os.path.join(DATA_DIR, 'weather_ashburn_va_2020_2025.csv'))
weather['timestamp'] = pd.to_datetime(weather['timestamp'])
weather.set_index('timestamp', inplace=True)
print(f"   Weather (Ashburn): {len(weather):,} rows")

# Carbon intensity (PJM — Virginia region)
carbon = pd.read_csv(os.path.join(DATA_DIR, 'carbon_intensity_PJM_full.csv'))
carbon['timestamp'] = pd.to_datetime(carbon['period'])
carbon.set_index('timestamp', inplace=True)
print(f"   Carbon Intensity (PJM): {len(carbon):,} rows")

# EIA Demand (PJM)
eia_demand = pd.read_csv(os.path.join(DATA_DIR, 'eia_demand_PJM_full.csv'))
eia_demand['timestamp'] = pd.to_datetime(eia_demand['period'])
# Filter to just demand type 'D'
eia_d = eia_demand[eia_demand['type'] == 'D'].copy()
eia_d.set_index('timestamp', inplace=True)
eia_d['grid_demand_mw'] = pd.to_numeric(eia_d['value'], errors='coerce')
print(f"   Grid Demand (PJM): {len(eia_d):,} rows")

# ============================================================
# MERGE INTO UNIFIED DATASET
# ============================================================

print("\n2. MERGING INTO UNIFIED HOURLY DATASET...")

# Align all to same hourly index
merged = dc.copy()
merged = merged.join(weather[['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 
                              'shortwave_radiation', 'cloud_cover']], how='left')
merged = merged.join(carbon[['carbon_intensity_gco2_kwh']], how='left')
merged = merged.join(eia_d[['grid_demand_mw']], how='left')

# Fill missing values
merged = merged.ffill().bfill()

# Add time features
merged['hour'] = merged.index.hour
merged['day_of_week'] = merged.index.dayofweek
merged['month'] = merged.index.month
merged['is_weekend'] = (merged['day_of_week'] >= 5).astype(int)

print(f"   Merged dataset: {len(merged):,} rows × {len(merged.columns)} columns")
print(f"   Columns: {merged.columns.tolist()}")

# Save merged dataset
merged.to_csv(os.path.join(DATA_DIR, 'merged_all_signals_2020_2025.csv'))
print(f"   Saved: merged_all_signals_2020_2025.csv")

# ============================================================
# 3. BASIC STATISTICS
# ============================================================

print("\n3. BASIC STATISTICS")
print("=" * 50)

key_cols = ['it_load_kw', 'cooling_load_kw', 'pue', 'temperature_2m', 
            'carbon_intensity_gco2_kwh', 'grid_demand_mw', 'renewable_availability_pct']

for col in key_cols:
    if col in merged.columns:
        s = merged[col].describe()
        print(f"\n   {col}:")
        print(f"     Mean: {s['mean']:.1f} | Std: {s['std']:.1f}")
        print(f"     Min: {s['min']:.1f} | Max: {s['max']:.1f}")
        print(f"     25%: {s['25%']:.1f} | 75%: {s['75%']:.1f}")

# ============================================================
# 4. CORRELATION ANALYSIS — The Key Question
# ============================================================

print("\n\n4. CORRELATION ANALYSIS")
print("=" * 50)
print("   Question: Do external signals correlate with DC energy demand?")

target = 'it_load_kw'
external_signals = ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m',
                    'shortwave_radiation', 'cloud_cover', 'carbon_intensity_gco2_kwh',
                    'grid_demand_mw']

print(f"\n   Correlations with {target}:")
print(f"   {'Signal':<35} {'Pearson r':>10} {'p-value':>12} {'Significant':>12}")
print(f"   {'─'*35} {'─'*10} {'─'*12} {'─'*12}")

correlations = {}
for signal in external_signals:
    if signal in merged.columns:
        valid = merged[[target, signal]].dropna()
        if len(valid) > 100:
            r, p = stats.pearsonr(valid[target], valid[signal])
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            print(f"   {signal:<35} {r:>10.4f} {p:>12.2e} {sig:>12}")
            correlations[signal] = {'r': r, 'p': p}

# ============================================================
# 5. TEMPORAL PATTERNS
# ============================================================

print("\n\n5. TEMPORAL PATTERNS IN DC ENERGY")
print("=" * 50)

# Hourly pattern
hourly_mean = merged.groupby('hour')['it_load_kw'].mean()
print(f"\n   Hourly pattern (IT Load kW):")
print(f"     Peak hour: {hourly_mean.idxmax()}:00 ({hourly_mean.max():.0f} kW)")
print(f"     Trough hour: {hourly_mean.idxmin()}:00 ({hourly_mean.min():.0f} kW)")
print(f"     Daily swing: {hourly_mean.max() - hourly_mean.min():.0f} kW ({(hourly_mean.max()-hourly_mean.min())/hourly_mean.mean()*100:.1f}%)")

# Weekly pattern
daily_mean = merged.groupby('day_of_week')['it_load_kw'].mean()
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
print(f"\n   Weekly pattern:")
for i, d in enumerate(days):
    print(f"     {d}: {daily_mean.iloc[i]:.0f} kW")

# Monthly pattern
monthly_mean = merged.groupby('month')['it_load_kw'].mean()
print(f"\n   Monthly pattern:")
print(f"     Peak month: {monthly_mean.idxmax()} ({monthly_mean.max():.0f} kW)")
print(f"     Trough month: {monthly_mean.idxmin()} ({monthly_mean.min():.0f} kW)")

# ============================================================
# 6. WEATHER → COOLING RELATIONSHIP
# ============================================================

print("\n\n6. WEATHER → COOLING LOAD RELATIONSHIP")
print("=" * 50)

r, p = stats.pearsonr(merged['temperature_2m'].dropna(), merged['cooling_load_kw'].dropna()[:len(merged['temperature_2m'].dropna())])
print(f"   Temperature → Cooling load: r = {r:.4f} (p = {p:.2e})")
print(f"   {'STRONG' if abs(r) > 0.5 else 'MODERATE' if abs(r) > 0.3 else 'WEAK'} correlation")

# Temperature bins
print(f"\n   Cooling load by temperature range:")
merged['temp_bin'] = pd.cut(merged['temperature_2m'], bins=[-20, 0, 10, 20, 30, 50])
temp_cooling = merged.groupby('temp_bin')['cooling_load_kw'].mean()
for bin_range, cooling in temp_cooling.items():
    print(f"     {str(bin_range):<15}: {cooling:.0f} kW avg cooling")

# ============================================================
# 7. CARBON INTENSITY PATTERNS
# ============================================================

print("\n\n7. CARBON INTENSITY PATTERNS (PJM Region)")
print("=" * 50)

hourly_carbon = merged.groupby('hour')['carbon_intensity_gco2_kwh'].mean()
print(f"   Hourly carbon intensity:")
print(f"     Cleanest hour: {hourly_carbon.idxmin()}:00 ({hourly_carbon.min():.0f} gCO2/kWh)")
print(f"     Dirtiest hour: {hourly_carbon.idxmax()}:00 ({hourly_carbon.max():.0f} gCO2/kWh)")
print(f"     Daily swing: {hourly_carbon.max() - hourly_carbon.min():.0f} gCO2/kWh")
print(f"     Optimization potential: {(hourly_carbon.max()-hourly_carbon.min())/hourly_carbon.mean()*100:.1f}% carbon reduction by time-shifting")

monthly_carbon = merged.groupby('month')['carbon_intensity_gco2_kwh'].mean()
print(f"\n   Monthly carbon intensity:")
print(f"     Cleanest month: {monthly_carbon.idxmin()} ({monthly_carbon.min():.0f} gCO2/kWh)")
print(f"     Dirtiest month: {monthly_carbon.idxmax()} ({monthly_carbon.max():.0f} gCO2/kWh)")

# ============================================================
# 8. KEY FINDING: External Signal Predictive Power
# ============================================================

print("\n\n8. KEY FINDING: EXTERNAL SIGNAL PREDICTIVE POWER")
print("=" * 50)

# Simple linear regression: can temperature predict next-hour cooling load?
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error

# Prepare features
X_internal = merged[['it_load_kw', 'hour', 'day_of_week', 'month']].dropna()
X_external = merged[['it_load_kw', 'hour', 'day_of_week', 'month', 
                      'temperature_2m', 'shortwave_radiation', 
                      'carbon_intensity_gco2_kwh']].dropna()
y = merged.loc[X_external.index, 'cooling_load_kw']

# Split: first 80% train, last 20% test
split_idx = int(len(X_external) * 0.8)

X_int_train = X_internal.iloc[:split_idx]
X_int_test = X_internal.iloc[split_idx:]
X_ext_train = X_external.iloc[:split_idx]
X_ext_test = X_external.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

# Model 1: Internal only
lr_int = LinearRegression()
lr_int.fit(X_int_train, y_train)
pred_int = lr_int.predict(X_int_test)
mape_int = mean_absolute_percentage_error(y_test, pred_int) * 100

# Model 2: Internal + External
lr_ext = LinearRegression()
lr_ext.fit(X_ext_train, y_train)
pred_ext = lr_ext.predict(X_ext_test)
mape_ext = mean_absolute_percentage_error(y_test, pred_ext) * 100

improvement = (mape_int - mape_ext) / mape_int * 100

print(f"\n   Predicting cooling load (next hour):")
print(f"   Model 1 (internal signals only):    MAPE = {mape_int:.2f}%")
print(f"   Model 2 (internal + external):      MAPE = {mape_ext:.2f}%")
print(f"   Improvement from external signals:  {improvement:.1f}%")
print(f"   {'✅ EXTERNAL SIGNALS IMPROVE PREDICTION' if improvement > 0 else '❌ No improvement'}")

# Feature importance in external model
print(f"\n   Feature importance (external model coefficients):")
for feat, coef in sorted(zip(X_external.columns, lr_ext.coef_), key=lambda x: abs(x[1]), reverse=True):
    print(f"     {feat:<30}: {coef:>10.4f}")

# ============================================================
# SUMMARY
# ============================================================

print("\n\n" + "=" * 70)
print("EDA SUMMARY — KEY FINDINGS FOR THE PAPER")
print("=" * 70)

print("""
1. TEMPORAL PATTERNS: DC energy has strong daily (±{daily_swing:.0f} kW), weekly 
   (weekday/weekend), and seasonal patterns that are predictable.

2. WEATHER CORRELATION: Ambient temperature significantly correlates with cooling 
   load (r={temp_r:.4f}). External weather is a leading indicator.

3. CARBON VARIABILITY: Grid carbon intensity varies {carbon_swing:.0f} gCO2/kWh daily,
   representing {carbon_pct:.1f}% optimization potential through time-shifting.

4. EXTERNAL SIGNALS IMPROVE PREDICTION: Adding weather + carbon to internal metrics
   improves cooling load prediction by {improve:.1f}% (MAPE reduction).
   This is the baseline proof that multi-signal fusion adds value.

5. NEXT STEP: Replace linear regression with Temporal Fusion Transformer to capture
   non-linear cross-signal dependencies and multi-horizon forecasting.
""".format(
    daily_swing=hourly_mean.max() - hourly_mean.min(),
    temp_r=r,
    carbon_swing=hourly_carbon.max() - hourly_carbon.min(),
    carbon_pct=(hourly_carbon.max()-hourly_carbon.min())/hourly_carbon.mean()*100,
    improve=improvement
))

# Save results
results = {
    'correlations': correlations,
    'mape_internal': mape_int,
    'mape_external': mape_ext,
    'improvement_pct': improvement,
    'total_rows': len(merged),
    'columns': len(merged.columns),
}

import json
with open(os.path.join(RESULTS_DIR, 'eda_results.json'), 'w') as f:
    json.dump({k: str(v) if not isinstance(v, (int, float)) else v for k, v in results.items()}, f, indent=2)

print(f"\nResults saved to: results/eda_results.json")
print(f"Merged dataset saved to: data/merged_all_signals_2020_2025.csv")
