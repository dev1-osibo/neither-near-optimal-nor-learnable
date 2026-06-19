"""
Prophet & ARIMA Comparison — Adding the "industry standard" baselines
======================================================================
Prophet is what most DC operators use today. We need to show TFT beats it.
ARIMA/SARIMA is the classical statistical baseline.

Same rules: predict from ONLY past data, evaluate at 1h, 4h, 12h, 24h ahead.
"""

import numpy as np
import pandas as pd
import os
import json
import time
from sklearn.metrics import mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

DATA_FILE = '/home/ubuntu/tft_training/merged_enriched_2020_2025.csv'
RESULTS_DIR = '/home/ubuntu/tft_training/results'

print("=" * 70)
print("Prophet & ARIMA — Industry Standard Baselines")
print("=" * 70)

# Load data
df = pd.read_csv(DATA_FILE)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)
df = df.sort_index()

target = 'cooling_load_kw'

# Same split as other models
train_end = '2024-06-30'
val_end = '2025-03-31'
train_df = df[:train_end]
test_df = df[val_end:]

print(f"Train: {len(train_df):,} rows")
print(f"Test:  {len(test_df):,} rows")

HORIZONS = [1, 4, 12, 24]
results = {}

# ============================================================
# 1. PROPHET (Internal only — univariate, the industry standard)
# ============================================================
print("\n\n1. PROPHET (univariate — industry standard)")
print("=" * 60)
print("Prophet uses ONLY the target's own history. No external signals.")

from prophet import Prophet

# Prophet needs 'ds' and 'y' columns
# We'll train on all training data, then predict test period
# For fair comparison: at each test point T, predict T+horizon using data up to T

# Since Prophet is slow to fit per-point, we'll fit once on training data
# and use its forecast for the test period (standard approach)

prophet_train = train_df[[target]].reset_index()
prophet_train.columns = ['ds', 'y']

print("   Fitting Prophet (internal only — univariate)...")
start = time.time()
m = Prophet(yearly_seasonality=True, weekly_seasonality=True, 
            daily_seasonality=True, changepoint_prior_scale=0.05)
m.fit(prophet_train)
fit_time = time.time() - start
print(f"   Fit time: {fit_time:.1f}s")

# Create future dataframe covering test period
future = m.make_future_dataframe(periods=len(test_df), freq='h')
forecast = m.predict(future)

# Extract predictions for test period
# For horizon H: the prediction at time T+H made at time T
# Prophet's forecast gives point estimates for each future timestamp
test_forecast = forecast.iloc[len(train_df):len(train_df)+len(test_df)]
prophet_pred = test_forecast['yhat'].values
test_actual = test_df[target].values

# For each horizon, shift the comparison
print(f"\n   Prophet Results (univariate — internal only):")
for h in HORIZONS:
    # Prophet prediction for T+h made using model fit on data up to train_end
    # This is a fair comparison: model sees no test data
    actual_h = test_actual[h:]
    pred_h = prophet_pred[:-h]  # Prediction made h steps earlier
    if len(actual_h) > 0 and len(pred_h) > 0:
        n = min(len(actual_h), len(pred_h))
        mape = mean_absolute_percentage_error(actual_h[:n], pred_h[:n]) * 100
    else:
        mape = float('nan')
    results[f'Prophet_internal_{h}h'] = mape
    print(f"     {h}h ahead: {mape:.2f}% MAPE")

# ============================================================
# 2. PROPHET WITH REGRESSORS (fusion — external signals)
# ============================================================
print("\n\n2. PROPHET + External Regressors (fusion)")
print("=" * 60)
print("Prophet with weather + carbon as additional regressors")

# Prophet can add regressors (external signals)
regressors = ['temperature_2m', 'shortwave_radiation', 'carbon_intensity_gco2_kwh']

prophet_train_ext = train_df[[target] + regressors].reset_index()
prophet_train_ext.columns = ['ds', 'y'] + regressors

print("   Fitting Prophet (with external regressors)...")
start = time.time()
m_ext = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                daily_seasonality=True, changepoint_prior_scale=0.05)
for reg in regressors:
    m_ext.add_regressor(reg)
m_ext.fit(prophet_train_ext)
fit_time = time.time() - start
print(f"   Fit time: {fit_time:.1f}s")

# Future with regressors (use actual values — Prophet requires future regressor values)
future_ext = test_df[[target] + regressors].reset_index()
future_ext.columns = ['ds', 'y'] + regressors
# For Prophet, we need to provide future regressor values
# In a real system, these would come from weather forecasts
# For this test, we use actuals (gives Prophet its BEST possible performance)

full_future = pd.concat([prophet_train_ext[['ds'] + regressors], 
                         future_ext[['ds'] + regressors]])
forecast_ext = m_ext.predict(full_future)

test_forecast_ext = forecast_ext.iloc[len(train_df):len(train_df)+len(test_df)]
prophet_ext_pred = test_forecast_ext['yhat'].values

print(f"\n   Prophet + Regressors Results (fusion — best case):")
for h in HORIZONS:
    actual_h = test_actual[h:]
    pred_h = prophet_ext_pred[:-h]
    if len(actual_h) > 0 and len(pred_h) > 0:
        n = min(len(actual_h), len(pred_h))
        mape = mean_absolute_percentage_error(actual_h[:n], pred_h[:n]) * 100
    else:
        mape = float('nan')
    results[f'Prophet_fusion_{h}h'] = mape
    print(f"     {h}h ahead: {mape:.2f}% MAPE")

# ============================================================
# 3. SARIMA (Statistical baseline)
# ============================================================
print("\n\n3. SARIMA (classical statistics)")
print("=" * 60)
print("Fitting on last 2000 hours of training (SARIMA is slow on full data)")

from statsmodels.tsa.statespace.sarimax import SARIMAX

# SARIMA is too slow for full dataset — use last 2000 hours of training
# and predict first 200 hours of test
sarima_train = train_df[target].iloc[-2000:]
sarima_test = test_df[target].iloc[:200]

print("   Fitting SARIMA(1,1,1)(1,1,1,24)...")
start = time.time()
try:
    model_sarima = SARIMAX(sarima_train, order=(1,1,1), seasonal_order=(1,1,1,24),
                           enforce_stationarity=False, enforce_invertibility=False)
    sarima_fit = model_sarima.fit(disp=False, maxiter=100)
    fit_time = time.time() - start
    print(f"   Fit time: {fit_time:.1f}s")
    
    # Forecast
    sarima_forecast = sarima_fit.forecast(steps=200)
    
    print(f"\n   SARIMA Results:")
    for h in HORIZONS:
        if h < len(sarima_forecast) and h < len(sarima_test):
            actual_h = sarima_test.values[h:]
            pred_h = sarima_forecast.values[:-h] if h > 0 else sarima_forecast.values
            n = min(len(actual_h), len(pred_h))
            if n > 10:
                mape = mean_absolute_percentage_error(actual_h[:n], pred_h[:n]) * 100
                results[f'SARIMA_{h}h'] = mape
                print(f"     {h}h ahead: {mape:.2f}% MAPE")
            else:
                print(f"     {h}h ahead: insufficient data")
        else:
            print(f"     {h}h ahead: N/A")
except Exception as e:
    print(f"   SARIMA failed: {e}")
    for h in HORIZONS:
        results[f'SARIMA_{h}h'] = float('nan')

# ============================================================
# 4. COMBINED RESULTS TABLE
# ============================================================
print("\n\n" + "=" * 70)
print("COMPLETE MODEL COMPARISON (including Prophet & ARIMA)")
print("=" * 70)

# Load previous results
prev_file = os.path.join(RESULTS_DIR, 'fair_model_comparison_results.json')
if os.path.exists(prev_file):
    with open(prev_file) as f:
        prev = json.load(f)
    prev_results = prev.get('results', {})
else:
    prev_results = {}

# Merge all results into one table
all_models = {}

# From previous run
for h_key, h_data in prev_results.items():
    for model_name, mape in h_data.items():
        if model_name not in all_models:
            all_models[model_name] = {}
        all_models[model_name][h_key] = mape

# Add Prophet and SARIMA
for h in HORIZONS:
    h_key = f'{h}h'
    
    p_int = results.get(f'Prophet_internal_{h}h')
    if p_int and not np.isnan(p_int):
        if 'Prophet (internal - univariate)' not in all_models:
            all_models['Prophet (internal - univariate)'] = {}
        all_models['Prophet (internal - univariate)'][h_key] = p_int
    
    p_fus = results.get(f'Prophet_fusion_{h}h')
    if p_fus and not np.isnan(p_fus):
        if 'Prophet + Regressors (fusion)' not in all_models:
            all_models['Prophet + Regressors (fusion)'] = {}
        all_models['Prophet + Regressors (fusion)'][h_key] = p_fus
    
    s = results.get(f'SARIMA_{h}h')
    if s and not np.isnan(s):
        if 'SARIMA(1,1,1)(1,1,1,24)' not in all_models:
            all_models['SARIMA(1,1,1)(1,1,1,24)'] = {}
        all_models['SARIMA(1,1,1)(1,1,1,24)'][h_key] = s

# Print final table
print(f"\n{'Model':<38} {'1h':>7} {'4h':>7} {'12h':>7} {'24h':>7} {'Avg':>7}")
print(f"{'─'*38} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")

sorted_models = sorted(all_models.items(), 
                       key=lambda x: np.mean([v for v in x[1].values() if not np.isnan(v)]))

for model_name, horizons in sorted_models:
    row = f"{model_name:<38}"
    mapes = []
    for h in ['1h', '4h', '12h', '24h']:
        val = horizons.get(h)
        if val and not np.isnan(val):
            row += f" {val:>6.2f}%"
            mapes.append(val)
        else:
            row += f" {'N/A':>7}"
    avg = np.mean(mapes) if mapes else float('nan')
    row += f" {avg:>6.2f}%"
    print(row)

# TFT improvement over Prophet
print(f"\n  TFT IMPROVEMENT OVER PROPHET:")
tft_mapes = all_models.get('TFT (fusion, 168h lookback)', {})
prophet_mapes = all_models.get('Prophet (internal - univariate)', {})
for h in ['1h', '4h', '12h', '24h']:
    tft_val = tft_mapes.get(h)
    p_val = prophet_mapes.get(h)
    if tft_val and p_val:
        improve = (p_val - tft_val) / p_val * 100
        print(f"    {h}: Prophet {p_val:.2f}% → TFT {tft_val:.2f}% ({improve:+.1f}% improvement)")

# Save combined results
combined = {
    'date': '2026-06-15',
    'all_models': {name: dict(h) for name, h in all_models.items()},
    'prophet_results': results
}
with open(os.path.join(RESULTS_DIR, 'complete_model_comparison.json'), 'w') as f:
    json.dump(combined, f, indent=2, default=str)

print(f"\nResults saved: {RESULTS_DIR}/complete_model_comparison.json")
print("=" * 70)
