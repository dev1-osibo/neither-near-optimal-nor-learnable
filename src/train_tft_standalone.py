"""
TFT Standalone Training Script — For AWS Deployment
=====================================================
Self-contained: model + training + evaluation in one file.
No relative imports needed.

Run: python3 train_tft_standalone.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# TFT MODEL ARCHITECTURE
# ============================================================

class GatedLinearUnit(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(input_size, hidden_size)
    
    def forward(self, x):
        return torch.sigmoid(self.fc2(x)) * self.fc1(x)


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1, context_size=None):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.glu = GatedLinearUnit(hidden_size, output_size)
        self.layer_norm = nn.LayerNorm(output_size)
        self.context_fc = nn.Linear(context_size, hidden_size, bias=False) if context_size else None
        self.skip = nn.Linear(input_size, output_size) if input_size != output_size else None
    
    def forward(self, x, context=None):
        residual = self.skip(x) if self.skip else x
        hidden = F.elu(self.fc1(x))
        if self.context_fc and context is not None:
            hidden = hidden + self.context_fc(context)
        hidden = self.dropout(self.fc2(hidden))
        hidden = self.glu(hidden)
        return self.layer_norm(hidden + residual)


class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_sizes, hidden_size, num_inputs, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_inputs = num_inputs
        self.individual_grns = nn.ModuleList([
            GatedResidualNetwork(s, hidden_size, hidden_size, dropout) for s in input_sizes
        ])
        self.weight_grn = GatedResidualNetwork(sum(input_sizes), hidden_size, num_inputs, dropout)
    
    def forward(self, inputs):
        processed = [grn(inp) for grn, inp in zip(self.individual_grns, inputs)]
        flat = torch.cat(inputs, dim=-1)
        weights = F.softmax(self.weight_grn(flat), dim=-1)
        stacked = torch.stack(processed, dim=-1)
        if weights.dim() == 2:
            weights = weights.unsqueeze(1)
        combined = (stacked * weights).sum(dim=-1)
        return combined, weights


class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, self.head_size)
        self.out_linear = nn.Linear(self.head_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.scale = np.sqrt(self.head_size)
    
    def forward(self, query, key, value):
        batch_size = query.size(0)
        Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        V = self.v_linear(value)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = self.dropout(F.softmax(scores, dim=-1))
        avg_attention = attention_weights.mean(dim=1)
        context = torch.matmul(avg_attention, V)
        return self.out_linear(context), avg_attention


class TemporalFusionTransformer(nn.Module):
    def __init__(self, num_past_inputs=7, num_future_inputs=3, hidden_size=64,
                 lstm_layers=2, num_heads=4, dropout=0.1, num_quantiles=3,
                 encoder_length=168, decoder_length=24):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        self.num_past_inputs = num_past_inputs
        self.num_future_inputs = num_future_inputs
        self.num_quantiles = num_quantiles
        
        self.past_proj = nn.ModuleList([nn.Linear(1, hidden_size) for _ in range(num_past_inputs)])
        self.future_proj = nn.ModuleList([nn.Linear(1, hidden_size) for _ in range(num_future_inputs)])
        
        self.past_vsn = VariableSelectionNetwork([hidden_size]*num_past_inputs, hidden_size, num_past_inputs, dropout)
        self.future_vsn = VariableSelectionNetwork([hidden_size]*num_future_inputs, hidden_size, num_future_inputs, dropout)
        
        self.encoder_lstm = nn.LSTM(hidden_size, hidden_size, lstm_layers, batch_first=True, dropout=dropout if lstm_layers > 1 else 0)
        self.decoder_lstm = nn.LSTM(hidden_size, hidden_size, lstm_layers, batch_first=True, dropout=dropout if lstm_layers > 1 else 0)
        
        self.post_lstm_gate = GatedLinearUnit(hidden_size, hidden_size)
        self.post_lstm_norm = nn.LayerNorm(hidden_size)
        
        self.attention = InterpretableMultiHeadAttention(hidden_size, num_heads, dropout)
        self.post_attn_gate = GatedLinearUnit(hidden_size, hidden_size)
        self.post_attn_norm = nn.LayerNorm(hidden_size)
        
        self.ff_grn = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)
        self.output_proj = nn.Linear(hidden_size, num_quantiles)
    
    def forward(self, past_inputs, future_inputs):
        batch_size = past_inputs.size(0)
        enc_len = past_inputs.size(1)
        dec_len = future_inputs.size(1)
        
        # Embed
        past_emb = [self.past_proj[i](past_inputs[:, :, i:i+1]) for i in range(self.num_past_inputs)]
        future_emb = [self.future_proj[i](future_inputs[:, :, i:i+1]) for i in range(self.num_future_inputs)]
        
        # Variable selection
        past_flat = [e.reshape(-1, self.hidden_size) for e in past_emb]
        past_sel, past_wts = self.past_vsn(past_flat)
        past_sel = past_sel.reshape(batch_size, enc_len, self.hidden_size)
        
        future_flat = [e.reshape(-1, self.hidden_size) for e in future_emb]
        future_sel, future_wts = self.future_vsn(future_flat)
        future_sel = future_sel.reshape(batch_size, dec_len, self.hidden_size)
        
        # LSTM
        enc_out, (h_n, c_n) = self.encoder_lstm(past_sel)
        dec_out, _ = self.decoder_lstm(future_sel, (h_n, c_n))
        
        lstm_out = torch.cat([enc_out, dec_out], dim=1)
        gated = self.post_lstm_gate(lstm_out)
        residual = torch.cat([past_sel, future_sel], dim=1)
        lstm_out = self.post_lstm_norm(gated + residual)
        
        # Attention
        dec_positions = lstm_out[:, enc_len:, :]
        attn_out, attn_wts = self.attention(dec_positions, lstm_out, lstm_out)
        gated_attn = self.post_attn_gate(attn_out)
        post_attn = self.post_attn_norm(gated_attn + dec_positions)
        
        # FF + output
        ff_out = self.ff_grn(post_attn)
        predictions = self.output_proj(ff_out)
        
        return predictions, attn_wts, {'past_weights': past_wts, 'future_weights': future_wts}


class QuantileLoss(nn.Module):
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles
    
    def forward(self, predictions, targets):
        losses = []
        for i, q in enumerate(self.quantiles):
            error = targets - predictions[:, :, i]
            losses.append(torch.max(q * error, (q - 1) * error).mean())
        return sum(losses) / len(losses)


# ============================================================
# DATA PIPELINE
# ============================================================

class TimeSeriesDataset(Dataset):
    def __init__(self, data, encoder_length, decoder_length, past_features, future_features, target_col):
        self.data = data.values.copy()
        self.columns = list(data.columns)
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        self.total_length = encoder_length + decoder_length
        self.past_indices = [self.columns.index(f) for f in past_features]
        self.future_indices = [self.columns.index(f) for f in future_features]
        self.target_idx = self.columns.index(target_col)
        self.n_samples = len(data) - self.total_length + 1
    
    def __len__(self):
        return max(0, self.n_samples)
    
    def __getitem__(self, idx):
        window = self.data[idx:idx + self.total_length]
        past = window[:self.encoder_length][:, self.past_indices].astype(np.float32)
        future = window[self.encoder_length:][:, self.future_indices].astype(np.float32)
        target = window[self.encoder_length:][:, self.target_idx].astype(np.float32)
        return torch.from_numpy(past), torch.from_numpy(future), torch.from_numpy(target)


# ============================================================
# MAIN TRAINING
# ============================================================

if __name__ == '__main__':
    print("=" * 70)
    print("TFT Training — FULL MODEL (AWS)")
    print("=" * 70)
    
    # Paths
    DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'merged_enriched_2020_2025.csv')
    RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load data
    print("\n1. Loading data...")
    df = pd.read_csv(DATA_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df = df.sort_index()
    print(f"   Loaded: {len(df):,} rows")
    
    # Features
    target_col = 'cooling_load_kw'
    past_features = ['temperature_2m', 'shortwave_radiation', 'carbon_intensity_gco2_kwh',
                     'it_load_kw', 'hour', 'day_of_week', 'month']
    future_features = ['hour', 'day_of_week', 'month']
    
    available_past = [f for f in past_features if f in df.columns]
    available_future = [f for f in future_features if f in df.columns]
    
    df_clean = df[available_past + [target_col]].dropna()
    print(f"   Clean: {len(df_clean):,} rows, {len(available_past)} past + {len(available_future)} future features")
    
    # Temporal split
    train_end = '2024-06-30'
    val_end = '2025-03-31'
    train_df = df_clean[:train_end]
    val_df = df_clean[train_end:val_end]
    test_df = df_clean[val_end:]
    print(f"   Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    
    # Scale
    scaler = StandardScaler()
    train_scaled = pd.DataFrame(scaler.fit_transform(train_df), columns=train_df.columns, index=train_df.index)
    val_scaled = pd.DataFrame(scaler.transform(val_df), columns=val_df.columns, index=val_df.index)
    test_scaled = pd.DataFrame(scaler.transform(test_df), columns=test_df.columns, index=test_df.index)
    
    target_mean = scaler.mean_[-1]
    target_std = scaler.scale_[-1]
    
    # Config — FULL MODEL
    ENCODER_LENGTH = 168
    DECODER_LENGTH = 24
    BATCH_SIZE = 64
    EPOCHS = 50
    PATIENCE = 10
    
    config = {
        'num_past_inputs': len(available_past),
        'num_future_inputs': len(available_future),
        'hidden_size': 64,
        'lstm_layers': 2,
        'num_heads': 4,
        'dropout': 0.1,
        'num_quantiles': 3,
        'encoder_length': ENCODER_LENGTH,
        'decoder_length': DECODER_LENGTH,
    }
    
    # Datasets
    train_ds = TimeSeriesDataset(train_scaled, ENCODER_LENGTH, DECODER_LENGTH, available_past, available_future, target_col)
    val_ds = TimeSeriesDataset(val_scaled, ENCODER_LENGTH, DECODER_LENGTH, available_past, available_future, target_col)
    test_ds = TimeSeriesDataset(test_scaled, ENCODER_LENGTH, DECODER_LENGTH, available_past, available_future, target_col)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"   Samples — Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")
    
    # Model
    print(f"\n2. Training TFT (full model: 64 hidden, 2 LSTM layers, 168h lookback)...")
    model = TemporalFusionTransformer(**config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,}")
    
    loss_fn = QuantileLoss([0.1, 0.5, 0.9])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []
    
    print(f"   Epochs: {EPOCHS} | Patience: {PATIENCE} | Batch: {BATCH_SIZE}")
    print(f"\n   {'Epoch':<7} {'Train':>10} {'Val':>10} {'LR':>10} {'Time':>7} {'Status'}")
    print(f"   {'─'*7} {'─'*10} {'─'*10} {'─'*10} {'─'*7} {'─'*10}")
    
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        ep_start = time.time()
        
        # Train
        model.train()
        ep_loss = 0
        n = 0
        for past, future, target in train_loader:
            optimizer.zero_grad()
            pred, _, _ = model(past, future)
            loss = loss_fn(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()
            n += 1
        train_loss = ep_loss / max(n, 1)
        train_losses.append(train_loss)
        
        # Validate
        model.eval()
        ep_vloss = 0
        nv = 0
        with torch.no_grad():
            for past, future, target in val_loader:
                pred, _, _ = model(past, future)
                ep_vloss += loss_fn(pred, target).item()
                nv += 1
        val_loss = ep_vloss / max(nv, 1)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]['lr']
        
        status = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, 'tft_best_model.pt'))
            status = "★ best"
        else:
            patience_counter += 1
            status = f"wait {patience_counter}/{PATIENCE}"
        
        ep_time = time.time() - ep_start
        print(f"   {epoch+1:<7} {train_loss:>9.6f} {val_loss:>9.6f} {lr:>9.6f} {ep_time:>5.1f}s {status}")
        
        if patience_counter >= PATIENCE:
            print(f"   Early stopping at epoch {epoch+1}")
            break
    
    total_time = time.time() - start_time
    print(f"\n   Done in {total_time:.0f}s ({total_time/60:.1f} min)")
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, 'tft_best_model.pt'), weights_only=True))
    
    # ============================================================
    # EVALUATION
    # ============================================================
    print(f"\n3. Evaluating on test set...")
    
    model.eval()
    all_pred, all_tgt, all_pw = [], [], []
    with torch.no_grad():
        for past, future, target in test_loader:
            pred, attn, vw = model(past, future)
            all_pred.append(pred.numpy())
            all_tgt.append(target.numpy())
            all_pw.append(vw['past_weights'].numpy())
    
    predictions = np.concatenate(all_pred, axis=0)
    targets = np.concatenate(all_tgt, axis=0)
    
    # Inverse scale
    pred_orig = predictions * target_std + target_mean
    tgt_orig = targets * target_std + target_mean
    p50 = pred_orig[:, :, 1]  # Median
    
    # Overall
    overall_mape = mean_absolute_percentage_error(tgt_orig.flatten(), p50.flatten()) * 100
    overall_mae = mean_absolute_error(tgt_orig.flatten(), p50.flatten())
    overall_r2 = r2_score(tgt_orig.flatten(), p50.flatten())
    
    print(f"   TFT MAPE: {overall_mape:.2f}% | MAE: {overall_mae:.1f} kW | R²: {overall_r2:.4f}")
    
    # Per horizon
    horizon_results = {}
    print(f"\n   {'Horizon':>8} {'MAPE%':>8} {'MAE':>8} {'R²':>8}")
    print(f"   {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for h in [0, 3, 11, 23]:
        h_mape = mean_absolute_percentage_error(tgt_orig[:, h], p50[:, h]) * 100
        h_mae = mean_absolute_error(tgt_orig[:, h], p50[:, h])
        h_r2 = r2_score(tgt_orig[:, h], p50[:, h])
        horizon_results[f'{h+1}h'] = {'mape': float(h_mape), 'mae': float(h_mae), 'r2': float(h_r2)}
        print(f"   {h+1:>6}h {h_mape:>7.2f}% {h_mae:>7.1f} {h_r2:>7.4f}")
    
    # Quantile coverage
    quantiles = [0.1, 0.5, 0.9]
    print(f"\n   Quantile coverage:")
    for i, q in enumerate(quantiles):
        cov = (tgt_orig <= pred_orig[:, :, i]).mean() * 100
        print(f"   P{int(q*100):>2}: expected {q*100:.0f}%, actual {cov:.1f}%")
    
    # ============================================================
    # BASELINES
    # ============================================================
    print(f"\n4. Baselines...")
    
    X_train_int = train_df[['it_load_kw', 'hour', 'day_of_week', 'month']].values
    X_train_all = train_df[available_past].values
    y_train = train_df[target_col].values
    X_test_int = test_df[['it_load_kw', 'hour', 'day_of_week', 'month']].values
    X_test_all = test_df[available_past].values
    y_test = test_df[target_col].values
    
    # Persistence
    pers_mape = mean_absolute_percentage_error(y_test[24:], y_test[:-24]) * 100
    print(f"   Persistence:           {pers_mape:.2f}% MAPE")
    
    # Linear internal
    lr_int = LinearRegression().fit(X_train_int, y_train)
    lr_int_mape = mean_absolute_percentage_error(y_test, lr_int.predict(X_test_int)) * 100
    print(f"   Linear (internal):     {lr_int_mape:.2f}% MAPE")
    
    # Linear all
    lr_all = LinearRegression().fit(X_train_all, y_train)
    lr_all_mape = mean_absolute_percentage_error(y_test, lr_all.predict(X_test_all)) * 100
    print(f"   Linear (all signals):  {lr_all_mape:.2f}% MAPE")
    
    # Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, random_state=42)
    gb.fit(X_train_all, y_train)
    gb_mape = mean_absolute_percentage_error(y_test, gb.predict(X_test_all)) * 100
    print(f"   Gradient Boosting:     {gb_mape:.2f}% MAPE")
    
    # Multi-horizon baselines
    print(f"\n   Multi-horizon comparison:")
    print(f"   {'Model':<25} {'1h':>8} {'4h':>8} {'12h':>8} {'24h':>8}")
    print(f"   {'─'*25} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    
    row = f"   {'TFT (ours)':<25}"
    for h in ['1h', '4h', '12h', '24h']:
        row += f" {horizon_results[h]['mape']:>7.2f}%"
    print(row)
    
    for name, X_tr, X_te in [('Linear (internal)', X_train_int, X_test_int), ('Linear (all)', X_train_all, X_test_all)]:
        row = f"   {name:<25}"
        for h in [1, 4, 12, 24]:
            y_tr_h = train_df[target_col].shift(-h).dropna().values[:len(X_tr)-h]
            y_te_h = test_df[target_col].shift(-h).dropna().values[:len(X_te)-h]
            lr_h = LinearRegression().fit(X_tr[:len(y_tr_h)], y_tr_h)
            mape_h = mean_absolute_percentage_error(y_te_h, lr_h.predict(X_te[:len(y_te_h)])) * 100
            row += f" {mape_h:>7.2f}%"
        print(row)
    
    # Variable importance
    print(f"\n5. Variable importance (TFT attention):")
    all_pw_cat = np.concatenate(all_pw, axis=0)
    avg_w = all_pw_cat.mean(axis=0).flatten()
    w_sum = avg_w.sum()
    var_importance = {}
    for i in np.argsort(-avg_w):
        feat = available_past[i]
        pct = avg_w[i] / w_sum * 100
        var_importance[feat] = float(pct)
        print(f"   {feat:<30} {pct:>6.1f}%")
    
    # Summary
    tft_vs_int = (lr_int_mape - overall_mape) / lr_int_mape * 100
    tft_vs_gb = (gb_mape - overall_mape) / gb_mape * 100
    
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS")
    print(f"{'='*70}")
    print(f"   TFT MAPE:              {overall_mape:.2f}%")
    print(f"   vs Internal-only:      {tft_vs_int:+.1f}% improvement")
    print(f"   vs Gradient Boosting:  {tft_vs_gb:+.1f}% improvement")
    print(f"   vs Persistence:        {(pers_mape-overall_mape)/pers_mape*100:+.1f}% improvement")
    
    # Save results
    results = {
        'date': '2026-06-14',
        'model_config': config,
        'training': {
            'epochs': len(train_losses),
            'best_val_loss': float(best_val_loss),
            'time_seconds': float(total_time),
            'train_losses': [float(l) for l in train_losses],
            'val_losses': [float(l) for l in val_losses],
        },
        'tft_performance': {
            'overall_mape': float(overall_mape),
            'overall_mae': float(overall_mae),
            'overall_r2': float(overall_r2),
            'horizons': horizon_results,
            'quantile_coverage': {f'P{int(q*100)}': float((tgt_orig <= pred_orig[:,:,i]).mean()*100) for i, q in enumerate(quantiles)}
        },
        'baselines': {
            'persistence': {'mape': float(pers_mape)},
            'linear_internal': {'mape': float(lr_int_mape)},
            'linear_all_signals': {'mape': float(lr_all_mape)},
            'gradient_boosting': {'mape': float(gb_mape)},
            'tft': {'mape': float(overall_mape), 'mae': float(overall_mae), 'r2': float(overall_r2)}
        },
        'variable_importance': var_importance,
        'improvement': {
            'vs_internal_pct': float(tft_vs_int),
            'vs_gb_pct': float(tft_vs_gb),
            'vs_persistence_pct': float((pers_mape-overall_mape)/pers_mape*100)
        },
        'data_split': {
            'train': f"{train_df.index.min().date()} to {train_df.index.max().date()} ({len(train_df)} rows)",
            'val': f"{val_df.index.min().date()} to {val_df.index.max().date()} ({len(val_df)} rows)",
            'test': f"{test_df.index.min().date()} to {test_df.index.max().date()} ({len(test_df)} rows)"
        }
    }
    
    with open(os.path.join(RESULTS_DIR, 'tft_training_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n   Results: {RESULTS_DIR}/tft_training_results.json")
    print(f"   Model:   {RESULTS_DIR}/tft_best_model.pt")
    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")
