"""
Temporal Fusion Transformer (TFT) for Multi-Signal DC Energy Forecasting
=========================================================================
Based on: Lim et al. 2021 "Temporal Fusion Transformers for Interpretable
Multi-horizon Time Series Forecasting" (arXiv:1912.09363)

Simplified implementation focused on:
- Multi-horizon prediction (1h, 4h, 12h, 24h ahead)
- Multi-quantile output (P10, P50, P90)
- Variable selection network (learns which signals matter)
- Interpretable attention (which past time steps matter)
- Comparison vs baselines (Prophet, Linear, GB)

Architecture:
  Input → Variable Selection Network → LSTM Encoder → 
  Multi-Head Attention → Gated Residual Network → Quantile Output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List


class GatedLinearUnit(nn.Module):
    """GLU: element-wise gating mechanism"""
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.sigmoid(self.fc2(x)) * self.fc1(x)


class GatedResidualNetwork(nn.Module):
    """GRN: Gated Residual Network with optional context"""
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 dropout: float = 0.1, context_size: int = None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.glu = GatedLinearUnit(hidden_size, output_size)
        self.layer_norm = nn.LayerNorm(output_size)
        
        if context_size is not None:
            self.context_fc = nn.Linear(context_size, hidden_size, bias=False)
        else:
            self.context_fc = None
        
        if input_size != output_size:
            self.skip = nn.Linear(input_size, output_size)
        else:
            self.skip = None
    
    def forward(self, x, context=None):
        # Skip connection
        if self.skip is not None:
            residual = self.skip(x)
        else:
            residual = x
        
        # Main path
        hidden = self.fc1(x)
        if self.context_fc is not None and context is not None:
            hidden = hidden + self.context_fc(context)
        hidden = self.elu(hidden)
        hidden = self.fc2(hidden)
        hidden = self.dropout(hidden)
        hidden = self.glu(hidden)
        
        return self.layer_norm(hidden + residual)


class VariableSelectionNetwork(nn.Module):
    """VSN: Learns which input variables are most important"""
    def __init__(self, input_sizes: List[int], hidden_size: int, num_inputs: int,
                 dropout: float = 0.1, context_size: int = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_inputs = num_inputs
        
        # Individual GRNs for each input variable
        self.individual_grns = nn.ModuleList([
            GatedResidualNetwork(input_size, hidden_size, hidden_size, dropout)
            for input_size in input_sizes
        ])
        
        # Softmax GRN for variable weights
        self.weight_grn = GatedResidualNetwork(
            sum(input_sizes), hidden_size, num_inputs, dropout, context_size
        )
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, inputs: List[torch.Tensor], context=None):
        # Process each variable independently
        processed = [grn(inp) for grn, inp in zip(self.individual_grns, inputs)]
        
        # Compute variable weights
        flat_inputs = torch.cat(inputs, dim=-1)
        weights = self.softmax(self.weight_grn(flat_inputs, context))  # (batch, num_inputs)
        
        # Weighted combination
        processed_stack = torch.stack(processed, dim=-1)  # (batch, hidden, num_inputs)
        
        if weights.dim() == 2:
            weights = weights.unsqueeze(1)  # (batch, 1, num_inputs)
        
        combined = (processed_stack * weights).sum(dim=-1)  # (batch, hidden)
        
        return combined, weights


class InterpretableMultiHeadAttention(nn.Module):
    """Interpretable attention: share values across heads for interpretability"""
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, self.head_size)  # Shared values
        
        self.out_linear = nn.Linear(self.head_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.scale = np.sqrt(self.head_size)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Project Q, K (multi-head), V (single head, shared)
        Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        V = self.v_linear(value)  # (batch, seq, head_size) — shared across heads
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Average attention across heads (for interpretability)
        avg_attention = attention_weights.mean(dim=1)  # (batch, seq_q, seq_k)
        
        # Apply attention to shared values
        context = torch.matmul(avg_attention, V)  # (batch, seq_q, head_size)
        output = self.out_linear(context)
        
        return output, avg_attention


class TemporalFusionTransformer(nn.Module):
    """
    Simplified TFT for DC energy forecasting
    
    Input features:
    - Static: facility metadata (not used in this version)
    - Known future: hour, day_of_week, month (calendar features)
    - Observed past: temperature, solar, carbon, IT load, cooling (target)
    
    Output: Multi-quantile predictions at multiple horizons
    """
    
    def __init__(self, 
                 num_past_inputs: int = 7,      # observed past variables
                 num_future_inputs: int = 3,     # known future variables  
                 hidden_size: int = 64,
                 lstm_layers: int = 2,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 num_quantiles: int = 3,         # P10, P50, P90
                 encoder_length: int = 168,      # 7 days lookback
                 decoder_length: int = 24,       # 24h forecast
                 ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        self.num_quantiles = num_quantiles
        self.num_past_inputs = num_past_inputs
        self.num_future_inputs = num_future_inputs
        
        # Input embeddings (project each variable to hidden_size)
        self.past_input_projection = nn.ModuleList([
            nn.Linear(1, hidden_size) for _ in range(num_past_inputs)
        ])
        self.future_input_projection = nn.ModuleList([
            nn.Linear(1, hidden_size) for _ in range(num_future_inputs)
        ])
        
        # Variable Selection Networks
        self.past_vsn = VariableSelectionNetwork(
            [hidden_size] * num_past_inputs, hidden_size, num_past_inputs, dropout
        )
        self.future_vsn = VariableSelectionNetwork(
            [hidden_size] * num_future_inputs, hidden_size, num_future_inputs, dropout
        )
        
        # LSTM Encoder-Decoder
        self.encoder_lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers=lstm_layers,
            batch_first=True, dropout=dropout if lstm_layers > 1 else 0
        )
        self.decoder_lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers=lstm_layers,
            batch_first=True, dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Post-LSTM gating
        self.post_lstm_gate = GatedLinearUnit(hidden_size, hidden_size)
        self.post_lstm_norm = nn.LayerNorm(hidden_size)
        
        # Multi-head attention
        self.attention = InterpretableMultiHeadAttention(hidden_size, num_heads, dropout)
        self.post_attention_gate = GatedLinearUnit(hidden_size, hidden_size)
        self.post_attention_norm = nn.LayerNorm(hidden_size)
        
        # Position-wise feed-forward
        self.ff_grn = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)
        
        # Output projection (multi-quantile)
        self.output_projection = nn.Linear(hidden_size, num_quantiles)
    
    def forward(self, past_inputs: torch.Tensor, future_inputs: torch.Tensor):
        """
        Args:
            past_inputs: (batch, encoder_length, num_past_inputs)
            future_inputs: (batch, decoder_length, num_future_inputs)
        Returns:
            predictions: (batch, decoder_length, num_quantiles)
            attention_weights: (batch, decoder_length, encoder_length)
            variable_weights: dict with past and future selection weights
        """
        batch_size = past_inputs.size(0)
        
        # === 1. Input Embedding ===
        # Project each variable to hidden_size
        past_embedded = []
        for i in range(self.num_past_inputs):
            emb = self.past_input_projection[i](past_inputs[:, :, i:i+1])  # (batch, enc_len, hidden)
            past_embedded.append(emb)
        
        future_embedded = []
        for i in range(self.num_future_inputs):
            emb = self.future_input_projection[i](future_inputs[:, :, i:i+1])
            future_embedded.append(emb)
        
        # === 2. Variable Selection ===
        # Apply VSN at each time step (flatten time into batch for efficiency)
        enc_len = past_inputs.size(1)
        dec_len = future_inputs.size(1)
        
        # Reshape for VSN: (batch * time, hidden)
        past_flat = [e.reshape(-1, self.hidden_size) for e in past_embedded]
        past_selected, past_weights = self.past_vsn(past_flat)
        past_selected = past_selected.reshape(batch_size, enc_len, self.hidden_size)
        
        future_flat = [e.reshape(-1, self.hidden_size) for e in future_embedded]
        future_selected, future_weights = self.future_vsn(future_flat)
        future_selected = future_selected.reshape(batch_size, dec_len, self.hidden_size)
        
        # === 3. LSTM Encoder-Decoder ===
        encoder_output, (h_n, c_n) = self.encoder_lstm(past_selected)
        decoder_output, _ = self.decoder_lstm(future_selected, (h_n, c_n))
        
        # Post-LSTM gating and residual
        lstm_output = torch.cat([encoder_output, decoder_output], dim=1)
        gated = self.post_lstm_gate(lstm_output)
        lstm_residual = torch.cat([past_selected, future_selected], dim=1)
        lstm_output = self.post_lstm_norm(gated + lstm_residual)
        
        # === 4. Multi-Head Attention (decoder attends to encoder) ===
        # Query: decoder positions, Key/Value: all positions
        decoder_positions = lstm_output[:, enc_len:, :]  # (batch, dec_len, hidden)
        all_positions = lstm_output  # (batch, enc_len+dec_len, hidden)
        
        attention_output, attention_weights = self.attention(
            decoder_positions, all_positions, all_positions
        )
        
        # Post-attention gating
        gated_attention = self.post_attention_gate(attention_output)
        attention_residual = decoder_positions
        post_attention = self.post_attention_norm(gated_attention + attention_residual)
        
        # === 5. Position-wise Feed-Forward ===
        ff_output = self.ff_grn(post_attention)
        
        # === 6. Quantile Output ===
        predictions = self.output_projection(ff_output)  # (batch, dec_len, num_quantiles)
        
        return predictions, attention_weights, {
            'past_weights': past_weights,
            'future_weights': future_weights
        }


class QuantileLoss(nn.Module):
    """Quantile loss for multi-quantile prediction"""
    def __init__(self, quantiles: List[float] = [0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        predictions: (batch, horizon, num_quantiles)
        targets: (batch, horizon)
        """
        losses = []
        for i, q in enumerate(self.quantiles):
            pred = predictions[:, :, i]
            error = targets - pred
            loss = torch.max(q * error, (q - 1) * error)
            losses.append(loss.mean())
        
        return sum(losses) / len(losses)


def create_tft_model(config: Dict = None) -> TemporalFusionTransformer:
    """Factory function to create TFT with default or custom config"""
    default_config = {
        'num_past_inputs': 7,      # temp, solar, carbon, it_load, hour, dow, month
        'num_future_inputs': 3,     # hour, day_of_week, month (known in future)
        'hidden_size': 64,
        'lstm_layers': 2,
        'num_heads': 4,
        'dropout': 0.1,
        'num_quantiles': 3,
        'encoder_length': 168,      # 7 days lookback
        'decoder_length': 24,       # 24h forecast
    }
    
    if config:
        default_config.update(config)
    
    return TemporalFusionTransformer(**default_config)


if __name__ == '__main__':
    # Quick test
    print("Testing TFT model architecture...")
    model = create_tft_model()
    
    # Dummy inputs
    batch_size = 4
    past = torch.randn(batch_size, 168, 7)    # 7 days of 7 past variables
    future = torch.randn(batch_size, 24, 3)   # 24h of 3 known future variables
    
    predictions, attention, var_weights = model(past, future)
    
    print(f"  Past input shape: {past.shape}")
    print(f"  Future input shape: {future.shape}")
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Attention shape: {attention.shape}")
    print(f"  Past variable weights shape: {var_weights['past_weights'].shape}")
    print(f"  Future variable weights shape: {var_weights['future_weights'].shape}")
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Test loss
    targets = torch.randn(batch_size, 24)
    loss_fn = QuantileLoss([0.1, 0.5, 0.9])
    loss = loss_fn(predictions, targets)
    print(f"  Test loss: {loss.item():.4f}")
    print("  ✓ Model architecture validated")
