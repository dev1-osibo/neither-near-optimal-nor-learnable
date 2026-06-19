"""
TFT Training Pipeline — Multi-Signal DC Energy Forecasting
============================================================
Full training script:
1. Data loading & preprocessing (temporal split, no leakage)
2. Sliding window dataset creation
3. TFT training with quantile loss
4. Baseline comparison (Prophet, Linear, GB)
5. Evaluation across multiple horizons
6. Variable importance extraction from attention weights
7. Results saving for paper
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import sys
import json
import time
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tft_model import TemporalFusionTransformer, QuantileLoss, create_tft_model

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')

print("=" * 70)
print("TFT Training — Multi-Signal DC Energy Forecasting")
print("=" * 70)

# ============================================================
# 1. DATA LOADING & PREPROCESSING
# ============================================================

print("\n1. Loading and preprocessing data...")

df = pd.read_csv(os.path.join(DATA_DIR, 'merged_enriched_2020_2025.csv'))
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)
df = df.sort_index()

print(f"   Loaded: {len(df):,} rows, {df.index.min()} to {df.index.max()}")

# Define features
target_col = 'cooling_load_kw'
past_features = ['temperature_2m', 'shortwave_radiation', 'carbon_intensity_gco2_kwh',
                 'it_load_kw', 'hour', 'day_of_week', 'month']
future_features = ['hour', 'day_of_week', 'month']  # Known in advance

# Ensure all columns exist
available_past = [f for f in past_features if f in df.columns]
available_future = [f for f in future_features if f in df.columns]
print(f"   Past features ({len(available_past)}): {available_past}")
print(f"   Future features ({len(available_future)}): {available_future}")

# Handle missing values
df_clean = df[available_past + [target_col]].dropna()
print(f"   After dropna: {len(df_clean):,} rows")

# TEMPORAL SPLIT (no leakage!)
# Train: 2020-01 to 2024-06 (4.5 years)
# Validation: 2024-07 to 2025-03 (9 months)
# Test: 2025-04 to 2025-12 (9 months)
train_end = '2024-06-30'
val_end = '2025-03-31'

train_df = df_clean[:train_end]
val_df = df_clean[train_end:val_end]
test_df = df_clean[val_end:]

print(f"   Train: {len(train_df):,} rows ({train_df.index.min().date()} to {train_df.index.max().date()})")
print(f"   Val:   {len(val_df):,} rows ({val_df.index.min().date()} to {val_df.index.max().date()})")
print(f"   Test:  {len(test_df):,} rows ({test_df.index.min().date()} to {test_df.index.max().date()})")

# Normalize using ONLY training statistics (prevent leakage)
scaler = StandardScaler()
train_scaled = pd.DataFrame(
    scaler.fit_transform(train_df[available_past + [target_col]]),
    columns=available_past + [target_col],
    index=train_df.index
)
val_scaled = pd.DataFrame(
    scaler.transform(val_df[available_past + [target_col]]),
    columns=available_past + [target_col],
    index=val_df.index
)
test_scaled = pd.DataFrame(
    scaler.transform(test_df[available_past + [target_col]]),
    columns=available_past + [target_col],
    index=test_df.index
)

# Save scaler params for inverse transform
target_idx = available_past.index('hour')  # Find target column index in scaler
target_scaler_mean = scaler.mean_[-1]  # Last column is target
target_scaler_std = scaler.scale_[-1]

print(f"   Target scaling: mean={target_scaler_mean:.2f}, std={target_scaler_std:.2f}")

# ============================================================
# 2. DATASET CLASS
# ============================================================

class TimeSeriesDataset(Dataset):
    """Sliding window dataset for TFT"""
    
    def __init__(self, data: pd.DataFrame, encoder_length: int = 168, 
                 decoder_length: int = 24, past_features: list = None,
                 future_features: list = None, target_col: str = 'cooling_load_kw'):
        self.data = data.values
        self.columns = list(data.columns)
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        self.total_length = encoder_length + decoder_length
        
        # Column indices
        self.past_indices = [self.columns.index(f) for f in past_features]
        self.future_indices = [self.columns.index(f) for f in future_features]
        self.target_idx = self.columns.index(target_col)
        
        # Valid starting indices
        self.n_samples = len(data) - self.total_length + 1
    
    def __len__(self):
        return max(0, self.n_samples)
    
    def __getitem__(self, idx):
        # Extract window
        window = self.data[idx:idx + self.total_length]
        
        # Past: encoder_length time steps, all past features
        past = window[:self.encoder_length, :][:, self.past_indices]
        
        # Future: decoder_length time steps, only known-future features
        future = window[self.encoder_length:, :][:, self.future_indices]
        
        # Target: decoder_length time steps of target variable
        target = window[self.encoder_length:, self.target_idx]
        
        return (torch.FloatTensor(past), 
                torch.FloatTensor(future), 
                torch.FloatTensor(target))


# Create datasets
ENCODER_LENGTH = 168  # 7 days lookback (FULL)
DECODER_LENGTH = 24   # 24h ahead

train_dataset = TimeSeriesDataset(train_scaled, ENCODER_LENGTH, DECODER_LENGTH, 
                                   available_past, available_future, target_col)
val_dataset = TimeSeriesDataset(val_scaled, ENCODER_LENGTH, DECODER_LENGTH,
                                 available_past, available_future, target_col)
test_dataset = TimeSeriesDataset(test_scaled, ENCODER_LENGTH, DECODER_LENGTH,
                                  available_past, available_future, target_col)

print(f"\n   Train samples: {len(train_dataset):,}")
print(f"   Val samples: {len(val_dataset):,}")
print(f"   Test samples: {len(test_dataset):,}")

# DataLoaders
BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ============================================================
# 3. TFT TRAINING
# ============================================================

print("\n\n2. Training TFT Model...")
print("=" * 60)

# Model config — FULL SIZE (no shortcuts)
config = {
    'num_past_inputs': len(available_past),
    'num_future_inputs': len(available_future),
    'hidden_size': 64,
    'lstm_layers': 2,
    'num_heads': 4,
    'dropout': 0.1,
    'num_quantiles': 3,  # P10, P50, P90
    'encoder_length': ENCODER_LENGTH,
    'decoder_length': DECODER_LENGTH,
}

model = create_tft_model(config)
quantiles = [0.1, 0.5, 0.9]
loss_fn = QuantileLoss(quantiles)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

# Training loop
EPOCHS = 50
PATIENCE = 10
best_val_loss = float('inf')
patience_counter = 0
train_losses = []
val_losses = []

print(f"   Config: {config}")
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"   Epochs: {EPOCHS}, Patience: {PATIENCE}, Batch: {BATCH_SIZE}")
print(f"\n   {'Epoch':<8} {'Train Loss':>12} {'Val Loss':>12} {'LR':>12} {'Time':>8}")
print(f"   {'─'*8} {'─'*12} {'─'*12} {'─'*12} {'─'*8}")

start_time = time.time()

for epoch in range(EPOCHS):
    epoch_start = time.time()
    
    # Training
    model.train()
    epoch_train_loss = 0
    n_batches = 0
    for past, future, target in train_loader:
        optimizer.zero_grad()
        predictions, _, _ = model(past, future)
        loss = loss_fn(predictions, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_train_loss += loss.item()
        n_batches += 1
    
    avg_train_loss = epoch_train_loss / max(n_batches, 1)
    train_losses.append(avg_train_loss)
    
    # Validation
    model.eval()
    epoch_val_loss = 0
    n_val_batches = 0
    with torch.no_grad():
        for past, future, target in val_loader:
            predictions, _, _ = model(past, future)
            loss = loss_fn(predictions, target)
            epoch_val_loss += loss.item()
            n_val_batches += 1
    
    avg_val_loss = epoch_val_loss / max(n_val_batches, 1)
    val_losses.append(avg_val_loss)
    
    # Learning rate scheduling
    scheduler.step(avg_val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        # Save best model
        torch.save(model.state_dict(), os.path.join(RESULTS_DIR, 'tft_best_model.pt'))
    else:
        patience_counter += 1
    
    epoch_time = time.time() - epoch_start
    
    if (epoch + 1) % 5 == 0 or epoch == 0 or patience_counter >= PATIENCE:
        print(f"   {epoch+1:<8} {avg_train_loss:>11.6f} {avg_val_loss:>11.6f} {current_lr:>11.6f} {epoch_time:>6.1f}s")
    
    if patience_counter >= PATIENCE:
        print(f"   Early stopping at epoch {epoch+1}")
        break

total_time = time.time() - start_time
print(f"\n   Training complete in {total_time:.0f}s ({total_time/60:.1f} min)")
print(f"   Best validation loss: {best_val_loss:.6f}")

# Load best model
model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, 'tft_best_model.pt'), weights_only=True))

# ============================================================
# 4. EVALUATION ON TEST SET
# ============================================================

print("\n\n3. Evaluating on Test Set...")
print("=" * 60)

model.eval()
all_predictions = []
all_targets = []
all_attention = []
all_past_weights = []

with torch.no_grad():
    for past, future, target in test_loader:
        predictions, attention, var_weights = model(past, future)
        all_predictions.append(predictions.numpy())
        all_targets.append(target.numpy())
        all_attention.append(attention.numpy())
        all_past_weights.append(var_weights['past_weights'].numpy())

predictions = np.concatenate(all_predictions, axis=0)  # (N, 24, 3)
targets = np.concatenate(all_targets, axis=0)            # (N, 24)

# Inverse transform to original scale
predictions_orig = predictions * target_scaler_std + target_scaler_mean
targets_orig = targets * target_scaler_std + target_scaler_mean

# Extract P50 (median) predictions for MAPE calculation
p50_predictions = predictions_orig[:, :, 1]  # Index 1 = P50

print(f"   Test samples: {len(targets_orig):,}")
print(f"   Predictions shape: {predictions_orig.shape}")

# Overall metrics
overall_mape = mean_absolute_percentage_error(targets_orig.flatten(), p50_predictions.flatten()) * 100
overall_mae = mean_absolute_error(targets_orig.flatten(), p50_predictions.flatten())
overall_r2 = r2_score(targets_orig.flatten(), p50_predictions.flatten())

print(f"\n   TFT Overall Performance:")
print(f"   MAPE: {overall_mape:.2f}%")
print(f"   MAE:  {overall_mae:.1f} kW")
print(f"   R²:   {overall_r2:.4f}")

# Per-horizon metrics
print(f"\n   Per-Horizon Performance:")
print(f"   {'Horizon':>8} {'MAPE%':>8} {'MAE kW':>8} {'R²':>8}")
print(f"   {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

horizon_results = {}
for h in [0, 3, 11, 23]:  # 1h, 4h, 12h, 24h (0-indexed)
    h_mape = mean_absolute_percentage_error(targets_orig[:, h], p50_predictions[:, h]) * 100
    h_mae = mean_absolute_error(targets_orig[:, h], p50_predictions[:, h])
    h_r2 = r2_score(targets_orig[:, h], p50_predictions[:, h])
    horizon_results[f'{h+1}h'] = {'mape': float(h_mape), 'mae': float(h_mae), 'r2': float(h_r2)}
    print(f"   {h+1:>6}h {h_mape:>7.2f}% {h_mae:>7.1f} {h_r2:>7.4f}")

# Quantile coverage
print(f"\n   Quantile Coverage (expected vs actual):")
for i, q in enumerate(quantiles):
    coverage = (targets_orig <= predictions_orig[:, :, i]).mean() * 100
    print(f"   P{int(q*100):>2}: expected {q*100:.0f}%, actual {coverage:.1f}%")

# ============================================================
# 5. BASELINE COMPARISONS
# ============================================================

print("\n\n4. Baseline Comparisons...")
print("=" * 60)

# Prepare test data for baselines (use same test period)
test_raw = test_df.copy()

baseline_results = {}

# --- Baseline 1: Persistence (last known value) ---
# Predict: cooling in 24h = cooling now
persistence_mape = mean_absolute_percentage_error(
    test_raw[target_col].iloc[24:].values,
    test_raw[target_col].iloc[:-24].values
) * 100
baseline_results['Persistence (naive)'] = {'mape': float(persistence_mape)}
print(f"   Persistence (naive):        {persistence_mape:.2f}% MAPE")

# --- Baseline 2: Linear Regression (internal only) ---
internal_features = ['it_load_kw', 'hour', 'day_of_week', 'month']
int_avail = [f for f in internal_features if f in train_df.columns]

X_train_int = train_df[int_avail].values
y_train_bl = train_df[target_col].values
X_test_int = test_df[int_avail].values
y_test_bl = test_df[target_col].values

lr_internal = LinearRegression().fit(X_train_int, y_train_bl)
lr_int_pred = lr_internal.predict(X_test_int)
lr_int_mape = mean_absolute_percentage_error(y_test_bl, lr_int_pred) * 100
baseline_results['Linear (internal only)'] = {'mape': float(lr_int_mape)}
print(f"   Linear (internal only):     {lr_int_mape:.2f}% MAPE")

# --- Baseline 3: Linear Regression (all signals) ---
X_train_all = train_df[available_past].values
X_test_all = test_df[available_past].values

lr_all = LinearRegression().fit(X_train_all, y_train_bl)
lr_all_pred = lr_all.predict(X_test_all)
lr_all_mape = mean_absolute_percentage_error(y_test_bl, lr_all_pred) * 100
baseline_results['Linear (all signals)'] = {'mape': float(lr_all_mape)}
print(f"   Linear (all signals):       {lr_all_mape:.2f}% MAPE")

# --- Baseline 4: Gradient Boosting (all signals) ---
gb = GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, 
                                subsample=0.8, random_state=42)
gb.fit(X_train_all, y_train_bl)
gb_pred = gb.predict(X_test_all)
gb_mape = mean_absolute_percentage_error(y_test_bl, gb_pred) * 100
baseline_results['Gradient Boosting'] = {'mape': float(gb_mape)}
print(f"   Gradient Boosting:          {gb_mape:.2f}% MAPE")

# --- Baseline 5: Linear (internal) for multi-horizon ---
# For fair comparison, also compute linear baselines at specific horizons
print(f"\n   Multi-horizon comparison:")
print(f"   {'Model':<30} {'1h':>8} {'4h':>8} {'12h':>8} {'24h':>8}")
print(f"   {'─'*30} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

# TFT results
tft_row = f"   {'TFT (ours)':<30}"
for h_key in ['1h', '4h', '12h', '24h']:
    tft_row += f" {horizon_results[h_key]['mape']:>7.2f}%"
print(tft_row)

# Linear baselines at each horizon
for model_name, features, X_tr, X_te in [
    ('Linear (internal)', int_avail, X_train_int, X_test_int),
    ('Linear (all signals)', available_past, X_train_all, X_test_all)
]:
    row = f"   {model_name:<30}"
    for h in [1, 4, 12, 24]:
        # Shift target
        y_tr_h = train_df[target_col].shift(-h).dropna().values[:len(X_tr)-h]
        y_te_h = test_df[target_col].shift(-h).dropna().values[:len(X_te)-h]
        X_tr_h = X_tr[:len(y_tr_h)]
        X_te_h = X_te[:len(y_te_h)]
        
        lr_h = LinearRegression().fit(X_tr_h, y_tr_h)
        pred_h = lr_h.predict(X_te_h)
        mape_h = mean_absolute_percentage_error(y_te_h, pred_h) * 100
        row += f" {mape_h:>7.2f}%"
    print(row)

# Add TFT to baseline results
baseline_results['TFT (ours)'] = {
    'mape': float(overall_mape),
    'mae': float(overall_mae),
    'r2': float(overall_r2),
    'horizons': horizon_results
}

# ============================================================
# 6. VARIABLE IMPORTANCE FROM ATTENTION
# ============================================================

print("\n\n5. Variable Importance (from TFT attention weights)...")
print("=" * 60)

# Aggregate attention weights across test set
all_pw = np.concatenate(all_past_weights, axis=0)  # (N*enc_len, 1, num_past)
avg_weights = all_pw.mean(axis=0).flatten()  # (num_past,)

print(f"\n   Past Variable Selection Weights (learned by TFT):")
print(f"   {'Variable':<30} {'Weight':>10} {'Rank':>6}")
print(f"   {'─'*30} {'─'*10} {'─'*6}")

# Normalize
weight_sum = avg_weights.sum()
var_importance = {}
sorted_indices = np.argsort(-avg_weights)
for rank, idx in enumerate(sorted_indices):
    feat = available_past[idx]
    weight = avg_weights[idx] / weight_sum * 100
    var_importance[feat] = float(weight)
    print(f"   {feat:<30} {weight:>8.1f}% {rank+1:>6}")

# ============================================================
# 7. IMPROVEMENT SUMMARY
# ============================================================

print("\n\n6. IMPROVEMENT SUMMARY")
print("=" * 60)

print(f"\n   {'Model':<30} {'MAPE%':>8} {'vs Internal':>12} {'vs GB':>10}")
print(f"   {'─'*30} {'─'*8} {'─'*12} {'─'*10}")

for name, res in sorted(baseline_results.items(), key=lambda x: x[1]['mape']):
    mape = res['mape']
    vs_int = (lr_int_mape - mape) / lr_int_mape * 100
    vs_gb = (gb_mape - mape) / gb_mape * 100
    marker = " ← OURS" if 'TFT' in name else ""
    print(f"   {name:<30} {mape:>7.2f}% {vs_int:>+11.1f}% {vs_gb:>+9.1f}%{marker}")

tft_vs_internal = (lr_int_mape - overall_mape) / lr_int_mape * 100
tft_vs_gb = (gb_mape - overall_mape) / gb_mape * 100

print(f"\n   KEY RESULT:")
print(f"   TFT with multi-signal fusion improves over:")
print(f"     • Internal-only linear: {tft_vs_internal:+.1f}%")
print(f"     • Gradient Boosting:    {tft_vs_gb:+.1f}%")
print(f"     • Persistence:          {(persistence_mape - overall_mape)/persistence_mape*100:+.1f}%")

# ============================================================
# 8. SAVE ALL RESULTS
# ============================================================

results = {
    'date': '2026-06-14',
    'model_config': config,
    'training': {
        'epochs_trained': len(train_losses),
        'best_val_loss': float(best_val_loss),
        'training_time_seconds': float(total_time),
        'train_losses': [float(l) for l in train_losses],
        'val_losses': [float(l) for l in val_losses],
    },
    'tft_performance': {
        'overall_mape': float(overall_mape),
        'overall_mae': float(overall_mae),
        'overall_r2': float(overall_r2),
        'horizons': horizon_results,
        'quantile_coverage': {
            f'P{int(q*100)}': float((targets_orig <= predictions_orig[:, :, i]).mean() * 100)
            for i, q in enumerate(quantiles)
        }
    },
    'baselines': baseline_results,
    'variable_importance': var_importance,
    'improvement': {
        'vs_internal_linear_pct': float(tft_vs_internal),
        'vs_gradient_boosting_pct': float(tft_vs_gb),
        'vs_persistence_pct': float((persistence_mape - overall_mape) / persistence_mape * 100)
    },
    'data_split': {
        'train': f"{train_df.index.min().date()} to {train_df.index.max().date()} ({len(train_df):,} rows)",
        'val': f"{val_df.index.min().date()} to {val_df.index.max().date()} ({len(val_df):,} rows)",
        'test': f"{test_df.index.min().date()} to {test_df.index.max().date()} ({len(test_df):,} rows)"
    }
}

with open(os.path.join(RESULTS_DIR, 'tft_training_results.json'), 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n\n   Results saved to: results/tft_training_results.json")
print(f"   Model saved to: results/tft_best_model.pt")

print("\n" + "=" * 70)
print("TFT TRAINING COMPLETE")
print("=" * 70)
