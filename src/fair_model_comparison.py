"""
Fair Model Comparison — All models predict from same point, see only past data
================================================================================
Task: Given data up to time T, predict cooling_load_kw at T+1h, T+4h, T+12h, T+24h
Rule: NO model sees any data beyond time T. Only past data allowed.

Models compared:
1. Persistence (naive: predict last known value)
2. Linear Regression with lag features (internal only)
3. Linear Regression with lag features (all signals — fusion)
4. Gradient Boosting with lag features (all signals)
5. LSTM (simple recurrent network)
6. TFT (our model — already trained, load results)

All use same train/val/test split. All evaluated at same horizons.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import json
import time
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'merged_enriched_2020_2025.csv')
if not os.path.exists(DATA_FILE):
    DATA_FILE = '/home/ubuntu/tft_training/merged_enriched_2020_2025.csv'

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
if not os.path.exists(RESULTS_DIR):
    RESULTS_DIR = '/home/ubuntu/tft_training/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 70)
print("FAIR MODEL COMPARISON — Same task, same data, only past visible")
print("=" * 70)

# ============================================================
# 1. LOAD AND PREPARE DATA
# ============================================================

print("\n1. Loading data...")
df = pd.read_csv(DATA_FILE)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)
df = df.sort_index()

target = 'cooling_load_kw'
all_features = ['temperature_2m', 'shortwave_radiation', 'carbon_intensity_gco2_kwh',
                'it_load_kw', 'hour', 'day_of_week', 'month']
internal_features = ['it_load_kw', 'hour', 'day_of_week', 'month']

df_clean = df[all_features + [target]].dropna()
print(f"   Rows: {len(df_clean):,}")

# Temporal split
train_end = '2024-06-30'
val_end = '2025-03-31'
train_df = df_clean[:train_end]
test_df = df_clean[val_end:]
print(f"   Train: {len(train_df):,} rows ({train_df.index.min().date()} to {train_df.index.max().date()})")
print(f"   Test:  {len(test_df):,} rows ({test_df.index.min().date()} to {test_df.index.max().date()})")

# ============================================================
# 2. CREATE LAG FEATURES (THE FAIR WAY)
# ============================================================

print("\n2. Creating lag features (only past data visible)...")

HORIZONS = [1, 4, 12, 24]  # hours ahead to predict
LOOKBACK_LAGS = [1, 2, 3, 4, 6, 12, 24, 48, 168]  # hours of history to use

def create_lag_dataset(data, features, target_col, horizon, lags):
    """
    For each time T, create features from lags [T-1, T-2, ..., T-168]
    and target at T+horizon. No future leakage.
    """
    df_lagged = pd.DataFrame(index=data.index)
    
    # Lag features: value of each feature at T-lag
    for feat in features:
        for lag in lags:
            df_lagged[f'{feat}_lag{lag}'] = data[feat].shift(lag)
    
    # Target: value at T+horizon (what we're predicting)
    df_lagged['target'] = data[target_col].shift(-horizon)
    
    # Drop rows with NaN (from lags and future target)
    df_lagged = df_lagged.dropna()
    
    return df_lagged

# ============================================================
# 3. TRAIN AND EVALUATE EACH MODEL
# ============================================================

print("\n3. Training and evaluating models...")
print("   (All models see ONLY past data, predict future)")

results = {}


def train_lstm_model(train_df, test_df, features, target_col, horizon, 
                     seq_len=72, hidden_size=32, epochs=15, batch_size=128):
    """Train a simple LSTM for fair comparison"""
    
    # Prepare sequences
    scaler = StandardScaler()
    train_vals = scaler.fit_transform(train_df[features + [target_col]].values)
    test_vals = scaler.transform(test_df[features + [target_col]].values)
    
    target_idx = len(features)  # last column
    target_mean = scaler.mean_[target_idx]
    target_std = scaler.scale_[target_idx]
    
    def make_sequences(data, seq_len, horizon, target_idx):
        X, y = [], []
        for i in range(len(data) - seq_len - horizon):
            X.append(data[i:i+seq_len, :])  # all features for seq_len steps
            y.append(data[i+seq_len+horizon-1, target_idx])  # target at T+horizon
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    
    X_train, y_train = make_sequences(train_vals, seq_len, horizon, target_idx)
    X_test, y_test = make_sequences(test_vals, seq_len, horizon, target_idx)
    
    if len(X_train) < 100 or len(X_test) < 100:
        return float('nan')
    
    # Simple LSTM model
    class SimpleLSTM(nn.Module):
        def __init__(self, input_size, hidden_size):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)
        
        def forward(self, x):
            _, (h_n, _) = self.lstm(x)
            return self.fc(h_n[-1]).squeeze(-1)
    
    model = SimpleLSTM(X_train.shape[2], hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    # Train
    train_tensor = torch.FloatTensor(X_train)
    target_tensor = torch.FloatTensor(y_train)
    
    dataset = torch.utils.data.TensorDataset(train_tensor, target_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        test_pred_scaled = model(torch.FloatTensor(X_test)).numpy()
    
    # Inverse scale
    test_pred = test_pred_scaled * target_std + target_mean
    test_actual = y_test * target_std + target_mean
    
    mape = mean_absolute_percentage_error(test_actual, test_pred) * 100
    return float(mape)


# === MAIN COMPARISON LOOP ===
for horizon in HORIZONS:
    print(f"\n   === Horizon: {horizon}h ahead ===")
    
    # --- Model 1: Persistence ---
    test_actual = test_df[target].iloc[horizon:].values
    test_persist = test_df[target].iloc[:-horizon].values
    persist_mape = mean_absolute_percentage_error(test_actual, test_persist) * 100
    
    # --- Model 2: Linear (internal only, lag features) ---
    train_lagged_int = create_lag_dataset(train_df, internal_features, target, horizon, LOOKBACK_LAGS)
    test_lagged_int = create_lag_dataset(test_df, internal_features, target, horizon, LOOKBACK_LAGS)
    
    feat_cols_int = [c for c in train_lagged_int.columns if c != 'target']
    X_train_int = train_lagged_int[feat_cols_int].values
    y_train_int = train_lagged_int['target'].values
    X_test_int = test_lagged_int[feat_cols_int].values
    y_test_int = test_lagged_int['target'].values
    
    lr_int = LinearRegression().fit(X_train_int, y_train_int)
    lr_int_mape = mean_absolute_percentage_error(y_test_int, lr_int.predict(X_test_int)) * 100
    
    # --- Model 3: Linear (all signals, lag features) ---
    train_lagged_all = create_lag_dataset(train_df, all_features, target, horizon, LOOKBACK_LAGS)
    test_lagged_all = create_lag_dataset(test_df, all_features, target, horizon, LOOKBACK_LAGS)
    
    feat_cols_all = [c for c in train_lagged_all.columns if c != 'target']
    X_train_all = train_lagged_all[feat_cols_all].values
    y_train_all = train_lagged_all['target'].values
    X_test_all = test_lagged_all[feat_cols_all].values
    y_test_all = test_lagged_all['target'].values
    
    lr_all = LinearRegression().fit(X_train_all, y_train_all)
    lr_all_mape = mean_absolute_percentage_error(y_test_all, lr_all.predict(X_test_all)) * 100
    
    # --- Model 4: Gradient Boosting (all signals, lag features) ---
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                                    subsample=0.8, random_state=42)
    gb.fit(X_train_all, y_train_all)
    gb_mape = mean_absolute_percentage_error(y_test_all, gb.predict(X_test_all)) * 100
    
    # --- Model 5: Simple LSTM ---
    lstm_mape = train_lstm_model(train_df, test_df, all_features, target, horizon)
    
    # Store results
    results[f'{horizon}h'] = {
        'Persistence (naive)': persist_mape,
        'Linear (internal lags)': lr_int_mape,
        'Linear (fusion lags)': lr_all_mape,
        'Gradient Boosting (fusion lags)': gb_mape,
        'LSTM (fusion)': lstm_mape,
    }
    
    print(f"     Persistence:           {persist_mape:.2f}%")
    print(f"     Linear (internal):     {lr_int_mape:.2f}%")
    print(f"     Linear (fusion):       {lr_all_mape:.2f}%")
    print(f"     Gradient Boosting:     {gb_mape:.2f}%")
    print(f"     LSTM:                  {lstm_mape:.2f}%")
# ============================================================

print("\n\n4. Adding TFT results from previous training...")

tft_results_file = os.path.join(RESULTS_DIR, 'tft_training_results.json')
if os.path.exists(tft_results_file):
    with open(tft_results_file) as f:
        tft_data = json.load(f)
    tft_horizons = tft_data.get('tft_performance', {}).get('horizons', {})
    for h_key, h_data in tft_horizons.items():
        if h_key in results:
            results[h_key]['TFT (fusion, 168h lookback)'] = h_data['mape']
    print("   TFT results loaded from previous training run")
else:
    print("   WARNING: No TFT results found. Run train_tft_standalone.py first.")

# ============================================================
# 5. SUMMARY TABLE
# ============================================================

print("\n\n" + "=" * 70)
print("FINAL COMPARISON — All models predicting future from ONLY past data")
print("=" * 70)

# Get all model names
all_models = set()
for h_results in results.values():
    all_models.update(h_results.keys())
all_models = sorted(all_models)

# Print table
print(f"\n{'Model':<35}", end='')
for h in HORIZONS:
    print(f" {h}h ahead", end='')
print(f"  {'Avg':>7}")
print(f"{'─'*35}", end='')
for h in HORIZONS:
    print(f" {'─'*8}", end='')
print(f"  {'─'*7}")

for model_name in all_models:
    row = f"{model_name:<35}"
    mapes = []
    for h in HORIZONS:
        mape = results.get(f'{h}h', {}).get(model_name)
        if mape is not None and not np.isnan(mape):
            row += f" {mape:>7.2f}%"
            mapes.append(mape)
        else:
            row += f" {'N/A':>8}"
    avg = np.mean(mapes) if mapes else float('nan')
    row += f"  {avg:>6.2f}%"
    print(row)

# Best model per horizon
print(f"\n  BEST MODEL PER HORIZON:")
for h in HORIZONS:
    h_results = results.get(f'{h}h', {})
    if h_results:
        best_name = min(h_results, key=h_results.get)
        best_mape = h_results[best_name]
        print(f"    {h}h: {best_name} ({best_mape:.2f}%)")

# Fusion improvement
print(f"\n  FUSION IMPROVEMENT (all signals vs internal only):")
for h in HORIZONS:
    h_results = results.get(f'{h}h', {})
    int_mape = h_results.get('Linear (internal lags)', 0)
    fus_mape = h_results.get('Linear (fusion lags)', 0)
    if int_mape > 0 and fus_mape > 0:
        improve = (int_mape - fus_mape) / int_mape * 100
        print(f"    {h}h: {int_mape:.2f}% → {fus_mape:.2f}% ({improve:+.1f}% improvement)")

# ============================================================
# 6. SAVE RESULTS
# ============================================================

output = {
    'date': '2026-06-15',
    'task': 'Predict cooling_load_kw at T+horizon using ONLY data up to T',
    'horizons': HORIZONS,
    'lookback_lags': LOOKBACK_LAGS,
    'train_period': f"{train_df.index.min().date()} to {train_df.index.max().date()}",
    'test_period': f"{test_df.index.min().date()} to {test_df.index.max().date()}",
    'results': results
}

output_file = os.path.join(RESULTS_DIR, 'fair_model_comparison_results.json')
with open(output_file, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\n\nResults saved: {output_file}")
print("=" * 70)
