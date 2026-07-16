"""
EDA 08: Deep Cross-Variable Interactions & Feature Variations
================================================================
Full deep-dive into all signal combinations for multi-source energy.

Analyses:
1. Full correlation matrix (all variables × all variables)
2. Granger causality: which signals CAUSE energy cost/demand changes
3. Non-linear interactions: polynomial, threshold effects
4. Variable recombination: all pairwise interactions
5. Regime detection: k-means clustering on multi-source state
6. Lag analysis: optimal lag for each variable pair
7. Feature importance: which variables predict each target
8. Conditional distributions: how does X behave given Y's state

Output: Comprehensive JSON results for paper tables.
"""

import pandas as pd
import numpy as np
import json
import os
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = os.path.expanduser("~/optena/data")
RESULTS_DIR = os.path.expanduser("~/optena/results")
os.makedirs(RESULTS_DIR, exist_ok=True)

results = {}

print("=" * 70)
print("EDA 08: DEEP CROSS-VARIABLE INTERACTIONS")
print("=" * 70)

# ============================================================
# 1. LOAD AND PREPARE DATA
# ============================================================
print("\n[1] Loading and preparing data...")

merged = pd.read_csv(os.path.join(DATA_DIR, "merged_enriched_2020_2025.csv"))
merged["timestamp"] = pd.to_datetime(merged["timestamp"])

ercot = pd.read_csv(os.path.join(DATA_DIR, "real_lmp_ERCOT_2020_2025.csv"))
ercot["timestamp"] = pd.to_datetime(ercot["timestamp"])

gas = pd.read_csv(os.path.join(DATA_DIR, "real_gas_henry_hub_daily_2020_2025.csv"))
gas["date"] = pd.to_datetime(gas["date"])

# Compute energy source signals
PANEL_AREA_M2 = 5556
merged["solar_gen_kw"] = (merged["shortwave_radiation"] * PANEL_AREA_M2 * 0.18 * 0.85) / 1000

def wind_power_curve(speed, rated=2000):
    power = np.zeros_like(speed, dtype=float)
    mask_ramp = (speed >= 3.5) & (speed < 12)
    power[mask_ramp] = rated * ((speed[mask_ramp] - 3.5) / (12 - 3.5)) ** 3
    mask_rated = (speed >= 12) & (speed <= 25)
    power[mask_rated] = rated
    return power

merged["wind_gen_kw"] = wind_power_curve(merged["wind_speed_10m"].values)
merged["combined_renewable_kw"] = merged["solar_gen_kw"] + merged["wind_gen_kw"]
merged["net_demand_kw"] = merged["total_facility_kw"] - merged["combined_renewable_kw"]
merged["net_demand_kw"] = merged["net_demand_kw"].clip(lower=0)

# Merge ERCOT prices into the main dataset
merged_with_price = merged.merge(
    ercot[["timestamp", "lmp_price_usd_mwh"]], on="timestamp", how="left"
)
# Forward fill price for any gaps
merged_with_price["lmp_price_usd_mwh"] = merged_with_price["lmp_price_usd_mwh"].ffill()

# Add gas price (daily → hourly via merge on date)
merged_with_price["date"] = merged_with_price["timestamp"].dt.date.astype(str)
gas["date_str"] = gas["date"].dt.strftime("%Y-%m-%d")
gas_map = gas.set_index("date_str")["gas_price_usd_mmbtu"].to_dict()
merged_with_price["gas_price_mmbtu"] = merged_with_price["date"].map(gas_map)
merged_with_price["gas_cost_mwh"] = merged_with_price["gas_price_mmbtu"] / 0.11723

# Drop rows without price data
df = merged_with_price.dropna(subset=["lmp_price_usd_mwh"]).copy()
print(f"  Working dataset: {len(df):,} rows with all signals")

# Define analysis variables
CORE_VARS = [
    "it_load_kw", "cooling_load_kw", "total_facility_kw", "pue",
    "ambient_temp_c", "temperature_2m", "relative_humidity_2m",
    "wind_speed_10m", "shortwave_radiation", "cloud_cover",
    "carbon_intensity_gco2_kwh", "solar_gen_kw", "wind_gen_kw",
    "combined_renewable_kw", "net_demand_kw", "lmp_price_usd_mwh",
]
# Filter to available columns
CORE_VARS = [v for v in CORE_VARS if v in df.columns]
print(f"  Analysis variables: {len(CORE_VARS)}")

# ============================================================
# 2. FULL CORRELATION MATRIX
# ============================================================
print("\n[2] Full Correlation Matrix...")

corr_matrix = df[CORE_VARS].corr()
print(f"  Matrix size: {corr_matrix.shape}")

# Find strongest correlations (excluding self)
strong_corrs = []
for i, v1 in enumerate(CORE_VARS):
    for j, v2 in enumerate(CORE_VARS):
        if i < j:
            r = corr_matrix.loc[v1, v2]
            strong_corrs.append((v1, v2, r))

strong_corrs.sort(key=lambda x: abs(x[2]), reverse=True)
print(f"\n  Top 15 strongest correlations:")
for v1, v2, r in strong_corrs[:15]:
    print(f"    {v1} × {v2}: r={r:.3f}")

print(f"\n  Top correlations with PRICE (lmp_price_usd_mwh):")
price_corrs = [(v, corr_matrix.loc[v, "lmp_price_usd_mwh"]) 
               for v in CORE_VARS if v != "lmp_price_usd_mwh"]
price_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
for v, r in price_corrs[:10]:
    print(f"    {v}: r={r:.3f}")

results["correlation_matrix"] = {
    "top_15_pairs": [(v1, v2, float(r)) for v1, v2, r in strong_corrs[:15]],
    "price_correlations": {v: float(r) for v, r in price_corrs},
}

# ============================================================
# 3. GRANGER CAUSALITY
# ============================================================
print("\n[3] Granger Causality Tests...")
from statsmodels.tsa.stattools import grangercausalitytests

# Test which variables Granger-cause price and demand changes
targets = ["lmp_price_usd_mwh", "net_demand_kw", "cooling_load_kw"]
predictors = ["temperature_2m", "wind_speed_10m", "shortwave_radiation", 
              "cloud_cover", "carbon_intensity_gco2_kwh", "it_load_kw",
              "solar_gen_kw", "wind_gen_kw"]
predictors = [p for p in predictors if p in df.columns]

granger_results = {}
max_lag = 6  # Test up to 6 hour lags

for target in targets:
    print(f"\n  Target: {target}")
    granger_results[target] = {}
    for pred in predictors:
        if pred == target:
            continue
        try:
            test_df = df[[target, pred]].dropna()
            if len(test_df) < 100:
                continue
            # Subsample if too large (Granger is slow on 50K+ rows)
            if len(test_df) > 10000:
                test_df = test_df.iloc[::5]  # Every 5th row
            
            result = grangercausalitytests(test_df[[target, pred]], maxlag=max_lag, verbose=False)
            # Get min p-value across all lags
            min_p = min(result[lag][0]["ssr_ftest"][1] for lag in range(1, max_lag+1))
            best_lag = min(range(1, max_lag+1), key=lambda l: result[l][0]["ssr_ftest"][1])
            
            causes = "YES" if min_p < 0.001 else ("WEAK" if min_p < 0.05 else "NO")
            granger_results[target][pred] = {
                "min_p_value": float(min_p),
                "best_lag": int(best_lag),
                "causes": causes
            }
            print(f"    {pred} → {target}: {causes} (p={min_p:.2e}, lag={best_lag}h)")
        except Exception as e:
            print(f"    {pred} → {target}: ERROR ({e})")

results["granger_causality"] = granger_results

# ============================================================
# 4. NON-LINEAR INTERACTIONS & THRESHOLDS
# ============================================================
print("\n[4] Non-Linear Interactions & Thresholds...")

# Test if relationships are non-linear by comparing linear vs polynomial R²
nonlinear_results = {}

test_pairs = [
    ("temperature_2m", "cooling_load_kw"),
    ("temperature_2m", "lmp_price_usd_mwh"),
    ("shortwave_radiation", "solar_gen_kw"),
    ("wind_speed_10m", "wind_gen_kw"),
    ("it_load_kw", "cooling_load_kw"),
    ("cloud_cover", "solar_gen_kw"),
    ("relative_humidity_2m", "cooling_load_kw"),
    ("carbon_intensity_gco2_kwh", "lmp_price_usd_mwh"),
]

for x_var, y_var in test_pairs:
    if x_var not in df.columns or y_var not in df.columns:
        continue
    x = df[x_var].dropna().values
    y = df[y_var].reindex(df[x_var].dropna().index).values
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    
    if len(x) < 100:
        continue
    
    # Linear fit
    slope, intercept, r_lin, p, se = stats.linregress(x, y)
    r2_linear = r_lin ** 2
    
    # Polynomial (degree 2) fit
    coeffs = np.polyfit(x, y, 2)
    y_pred_poly = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred_poly) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2_poly = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    improvement = (r2_poly - r2_linear) / max(r2_linear, 0.001) * 100
    
    # Find threshold (piecewise breakpoint)
    # Simple: split at median and compare slopes
    med = np.median(x)
    below = (x < med)
    above = (x >= med)
    if below.sum() > 10 and above.sum() > 10:
        slope_below = stats.linregress(x[below], y[below])[0]
        slope_above = stats.linregress(x[above], y[above])[0]
    else:
        slope_below = slope_above = slope
    
    nonlinear_results[f"{x_var} → {y_var}"] = {
        "r2_linear": float(r2_linear),
        "r2_polynomial": float(r2_poly),
        "improvement_pct": float(improvement),
        "slope_below_median": float(slope_below),
        "slope_above_median": float(slope_above),
        "slope_ratio": float(slope_above / slope_below) if slope_below != 0 else None,
        "is_nonlinear": improvement > 10,
    }
    
    nl = "NON-LINEAR" if improvement > 10 else "~linear"
    print(f"  {x_var} → {y_var}: R²_lin={r2_linear:.3f}, R²_poly={r2_poly:.3f} "
          f"(+{improvement:.0f}%) [{nl}]")

results["nonlinear_interactions"] = nonlinear_results

# ============================================================
# 5. ALL PAIRWISE INTERACTION FEATURES
# ============================================================
print("\n[5] Pairwise Interaction Feature Importance...")

# Create all pairwise products and test which predict price/demand best
interaction_vars = ["temperature_2m", "wind_speed_10m", "shortwave_radiation",
                    "cloud_cover", "it_load_kw", "carbon_intensity_gco2_kwh",
                    "relative_humidity_2m", "solar_gen_kw", "wind_gen_kw"]
interaction_vars = [v for v in interaction_vars if v in df.columns]

# Build interaction features
interaction_features = {}
for v1, v2 in combinations(interaction_vars, 2):
    feat_name = f"{v1}_x_{v2}"
    interaction_features[feat_name] = df[v1] * df[v2]

interaction_df = pd.DataFrame(interaction_features)
print(f"  Created {len(interaction_features)} pairwise interactions")

# Test which interactions best predict price
target = df["lmp_price_usd_mwh"].values
valid_mask = ~np.isnan(target)
target_valid = target[valid_mask]

interaction_importance = {}
for feat_name, feat_vals in interaction_features.items():
    vals = feat_vals.values[valid_mask]
    mask2 = ~np.isnan(vals)
    if mask2.sum() < 100:
        continue
    r = np.corrcoef(vals[mask2], target_valid[mask2])[0, 1]
    interaction_importance[feat_name] = float(abs(r))

# Sort by importance
sorted_interactions = sorted(interaction_importance.items(), key=lambda x: x[1], reverse=True)
print(f"\n  Top 10 interaction features for predicting PRICE:")
for name, imp in sorted_interactions[:10]:
    print(f"    {name}: |r|={imp:.4f}")

# Same for cooling load
target_cool = df["cooling_load_kw"].values
valid_cool = ~np.isnan(target_cool)
interaction_importance_cool = {}
for feat_name, feat_vals in interaction_features.items():
    vals = feat_vals.values[valid_cool]
    mask2 = ~np.isnan(vals)
    if mask2.sum() < 100:
        continue
    r = np.corrcoef(vals[mask2], target_cool[valid_cool][mask2])[0, 1]
    interaction_importance_cool[feat_name] = float(abs(r))

sorted_cool = sorted(interaction_importance_cool.items(), key=lambda x: x[1], reverse=True)
print(f"\n  Top 10 interaction features for predicting COOLING:")
for name, imp in sorted_cool[:10]:
    print(f"    {name}: |r|={imp:.4f}")

results["interaction_features"] = {
    "top_for_price": sorted_interactions[:20],
    "top_for_cooling": sorted_cool[:20],
}

# ============================================================
# 6. REGIME DETECTION (K-MEANS CLUSTERING)
# ============================================================
print("\n[6] Regime Detection...")

# Cluster on: temp, solar, wind, load, price
regime_vars = ["temperature_2m", "shortwave_radiation", "wind_speed_10m",
               "it_load_kw", "lmp_price_usd_mwh"]
regime_vars = [v for v in regime_vars if v in df.columns]

regime_data = df[regime_vars].dropna()
scaler = StandardScaler()
scaled = scaler.fit_transform(regime_data)

# Test different K values
inertias = {}
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(scaled)
    inertias[k] = km.inertia_

# Use K=4 (consistent with previous EDA finding of 4 regimes)
K = 4
km = KMeans(n_clusters=K, random_state=42, n_init=10)
labels = km.fit_predict(scaled)
regime_data_labeled = regime_data.copy()
regime_data_labeled["regime"] = labels

print(f"  K={K} clusters, sizes: {pd.Series(labels).value_counts().sort_index().to_dict()}")
print(f"\n  Regime centroids:")
centroids = pd.DataFrame(
    scaler.inverse_transform(km.cluster_centers_),
    columns=regime_vars
)
for i in range(K):
    row = centroids.iloc[i]
    print(f"    Regime {i}: temp={row['temperature_2m']:.0f}°C, "
          f"solar={row['shortwave_radiation']:.0f}W/m², "
          f"wind={row['wind_speed_10m']:.1f}m/s, "
          f"load={row['it_load_kw']:.0f}kW, "
          f"price=${row['lmp_price_usd_mwh']:.0f}/MWh")

# Name the regimes based on characteristics
regime_names = {}
for i in range(K):
    c = centroids.iloc[i]
    if c["temperature_2m"] < 10 and c["shortwave_radiation"] < 100:
        regime_names[i] = "Cold/Night"
    elif c["temperature_2m"] < 20 and c["shortwave_radiation"] > 200:
        regime_names[i] = "Cool/Sunny"
    elif c["temperature_2m"] > 25 and c["lmp_price_usd_mwh"] > 50:
        regime_names[i] = "Hot/Expensive"
    elif c["wind_speed_10m"] > 12:
        regime_names[i] = "Windy"
    else:
        regime_names[i] = f"Regime_{i}"

print(f"\n  Regime labels: {regime_names}")

results["regimes"] = {
    "k": K,
    "sizes": pd.Series(labels).value_counts().sort_index().to_dict(),
    "centroids": centroids.round(2).to_dict(),
    "regime_names": regime_names,
    "inertias": {str(k): float(v) for k, v in inertias.items()},
}

# ============================================================
# 7. OPTIMAL LAG ANALYSIS
# ============================================================
print("\n[7] Optimal Lag Analysis...")

# For each predictor → target pair, find the lag that maximizes correlation
lag_targets = ["lmp_price_usd_mwh", "cooling_load_kw", "net_demand_kw"]
lag_predictors = ["temperature_2m", "wind_speed_10m", "shortwave_radiation",
                  "carbon_intensity_gco2_kwh", "it_load_kw", "solar_gen_kw", "wind_gen_kw"]
lag_predictors = [p for p in lag_predictors if p in df.columns]
MAX_LAG = 24  # Test up to 24 hours

lag_results = {}
for target in lag_targets:
    if target not in df.columns:
        continue
    lag_results[target] = {}
    print(f"\n  Target: {target}")
    for pred in lag_predictors:
        if pred == target or pred not in df.columns:
            continue
        best_r = 0
        best_lag = 0
        for lag in range(0, MAX_LAG + 1):
            shifted = df[pred].shift(lag)
            valid = ~(shifted.isna() | df[target].isna())
            if valid.sum() < 100:
                continue
            r = abs(shifted[valid].corr(df[target][valid]))
            if r > best_r:
                best_r = r
                best_lag = lag
        
        lag_results[target][pred] = {"best_lag": int(best_lag), "correlation": float(best_r)}
        if best_lag > 0:
            print(f"    {pred}: best lag={best_lag}h (|r|={best_r:.3f})")

results["lag_analysis"] = lag_results

# ============================================================
# 8. FEATURE IMPORTANCE (GRADIENT BOOSTING)
# ============================================================
print("\n[8] Feature Importance via Gradient Boosting...")

feature_cols = [v for v in CORE_VARS if v not in ["lmp_price_usd_mwh", "net_demand_kw"]]
feature_cols = [v for v in feature_cols if v in df.columns]

# Add time features
df["hour_feat"] = df["timestamp"].dt.hour
df["month_feat"] = df["timestamp"].dt.month
df["dow_feat"] = df["timestamp"].dt.dayofweek
feature_cols_full = feature_cols + ["hour_feat", "month_feat", "dow_feat"]

importance_results = {}

for target_name in ["lmp_price_usd_mwh", "cooling_load_kw", "net_demand_kw"]:
    if target_name not in df.columns:
        continue
    print(f"\n  Target: {target_name}")
    
    avail_feats = [f for f in feature_cols_full if f in df.columns and f != target_name]
    sub = df[avail_feats + [target_name]].dropna()
    
    if len(sub) < 100:
        print(f"    Skipping — not enough data")
        continue
    
    X = sub[avail_feats].values
    y = sub[target_name].values
    
    # Train/test split (temporal — no shuffling)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42,
                                    learning_rate=0.05, subsample=0.8)
    gb.fit(X_train, y_train)
    y_pred = gb.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test[y_test != 0], y_pred[y_test != 0]) * 100
    
    # Feature importance
    importances = dict(zip(avail_feats, gb.feature_importances_))
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    
    print(f"    R² = {r2:.4f}, MAPE = {mape:.1f}%")
    print(f"    Top 10 features:")
    for feat, imp in sorted_imp[:10]:
        print(f"      {feat}: {imp:.4f} ({imp*100:.1f}%)")
    
    importance_results[target_name] = {
        "r2": float(r2),
        "mape": float(mape),
        "feature_importance": {k: float(v) for k, v in sorted_imp},
        "top_5": [(k, float(v)) for k, v in sorted_imp[:5]],
    }

results["feature_importance"] = importance_results

# ============================================================
# 9. VARIABLE ABLATION STUDY
# ============================================================
print("\n[9] Variable Ablation Study...")
print("  Testing: what happens when we remove each variable group?")

# Define variable groups (matching patent source categories)
var_groups = {
    "weather": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", 
                "shortwave_radiation", "cloud_cover"],
    "renewable_gen": ["solar_gen_kw", "wind_gen_kw", "combined_renewable_kw"],
    "carbon": ["carbon_intensity_gco2_kwh"],
    "internal_telemetry": ["it_load_kw", "cooling_load_kw", "pue"],
    "time_features": ["hour_feat", "month_feat", "dow_feat"],
}

# Baseline: all features predicting net_demand
target_name = "cooling_load_kw"
all_feats = [f for f in feature_cols_full if f in df.columns and f != target_name]
sub = df[all_feats + [target_name]].dropna()
X_all = sub[all_feats].values
y_all = sub[target_name].values
split = int(len(X_all) * 0.8)

gb_base = GradientBoostingRegressor(n_estimators=150, max_depth=4, random_state=42, learning_rate=0.05)
gb_base.fit(X_all[:split], y_all[:split])
baseline_r2 = r2_score(y_all[split:], gb_base.predict(X_all[split:]))
print(f"\n  Baseline (all features): R² = {baseline_r2:.4f}")

ablation_results = {}
for group_name, group_vars in var_groups.items():
    # Remove this group
    remaining = [f for f in all_feats if f not in group_vars]
    if not remaining:
        continue
    
    sub_abl = df[remaining + [target_name]].dropna()
    X_abl = sub_abl[remaining].values
    y_abl = sub_abl[target_name].values
    split_abl = int(len(X_abl) * 0.8)
    
    gb_abl = GradientBoostingRegressor(n_estimators=150, max_depth=4, random_state=42, learning_rate=0.05)
    gb_abl.fit(X_abl[:split_abl], y_abl[:split_abl])
    abl_r2 = r2_score(y_abl[split_abl:], gb_abl.predict(X_abl[split_abl:]))
    
    drop = (baseline_r2 - abl_r2) / baseline_r2 * 100
    ablation_results[group_name] = {
        "r2_without": float(abl_r2),
        "r2_drop_pct": float(drop),
        "vars_removed": group_vars,
    }
    print(f"  Without {group_name}: R² = {abl_r2:.4f} (drop: {drop:.1f}%)")

results["ablation_study"] = {
    "target": target_name,
    "baseline_r2": float(baseline_r2),
    "groups": ablation_results,
}

# ============================================================
# 10. CONDITIONAL DISTRIBUTIONS
# ============================================================
print("\n[10] Conditional Distributions...")

# How does price behave under different conditions?
conditions = {
    "high_temp": df["temperature_2m"] > df["temperature_2m"].quantile(0.9),
    "low_temp": df["temperature_2m"] < df["temperature_2m"].quantile(0.1),
    "high_solar": df["solar_gen_kw"] > df["solar_gen_kw"].quantile(0.9),
    "no_solar": df["solar_gen_kw"] == 0,
    "high_wind": df["wind_gen_kw"] > df["wind_gen_kw"].quantile(0.9),
    "low_wind": df["wind_gen_kw"] < df["wind_gen_kw"].quantile(0.1),
    "high_load": df["it_load_kw"] > df["it_load_kw"].quantile(0.9),
    "low_load": df["it_load_kw"] < df["it_load_kw"].quantile(0.1),
    "peak_hours": df["timestamp"].dt.hour.between(16, 20),
    "off_peak": df["timestamp"].dt.hour.between(1, 5),
}

cond_results = {}
print(f"\n  Grid price ($/MWh) under different conditions:")
for cond_name, mask in conditions.items():
    if mask.sum() < 10:
        continue
    prices = df.loc[mask, "lmp_price_usd_mwh"]
    cond_results[cond_name] = {
        "count": int(mask.sum()),
        "price_mean": float(prices.mean()),
        "price_median": float(prices.median()),
        "price_std": float(prices.std()),
        "price_p95": float(prices.quantile(0.95)),
    }
    print(f"    {cond_name:15s}: mean=${prices.mean():.1f}, "
          f"median=${prices.median():.1f}, p95=${prices.quantile(0.95):.0f} "
          f"(n={mask.sum():,})")

results["conditional_distributions"] = cond_results

# ============================================================
# SAVE ALL RESULTS
# ============================================================
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

outpath = os.path.join(RESULTS_DIR, "eda_deep_cross_variations_results.json")
with open(outpath, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"  ✓ Saved: {outpath}")

print("\n" + "=" * 70)
print("EDA 08 COMPLETE")
print("=" * 70)
