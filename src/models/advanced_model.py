import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional
from pathlib import Path
import json

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class MultiHorizonTransformer(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        input_size = config['input_size']
        d_model = config['d_model']
        nhead = config['nhead']
        num_layers = config['num_layers']
        dropout = config.get('dropout', 0.1)
        output_size = config['output_size']
        self.prediction_horizons = config.get('prediction_horizons', [15, 30, 60, 120, 1440])  # in minutes
        
        # Input projection
        self.input_proj = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Multi-horizon prediction heads
        self.prediction_heads = nn.ModuleDict()
        for horizon in self.prediction_horizons:
            self.prediction_heads[str(horizon)] = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, output_size * 2)  # Predict mean and log variance
            )
        
        # Horizon embedding
        self.horizon_embedding = nn.Embedding(len(self.prediction_horizons), d_model)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        # x shape: [batch_size, seq_len, input_size]
        batch_size = x.size(0)
        
        # Project input
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        
        # Process through transformer
        transformer_out = self.transformer(x)
        
        # Get the last time step's output
        last_output = transformer_out[:, -1, :]
        
        # Generate predictions for each horizon
        predictions = {}
        for i, (horizon, head) in enumerate(self.prediction_heads.items()):
            # Get horizon embedding
            horizon_idx = torch.tensor([i] * batch_size, device=x.device)
            horizon_emb = self.horizon_embedding(horizon_idx)
            
            # Combine with transformer output
            combined = last_output + horizon_emb
            
            # Get prediction (mean and log variance)
            pred = head(combined)
            mean = pred[..., :pred.size(-1)//2]
            log_var = pred[..., pred.size(-1)//2:]
            
            predictions[horizon] = {
                'mean': mean,
                'std': torch.exp(0.5 * log_var)  # Convert log variance to std
            }
        
        return predictions

class GaussianNLLLoss(nn.Module):
    def forward(self, pred_mean: torch.Tensor, target: torch.Tensor, pred_std: torch.Tensor) -> torch.Tensor:
        """
        Gaussian Negative Log Likelihood Loss.
        
        Args:
            pred_mean: Predicted means [batch_size, output_size]
            target: Target values [batch_size, output_size]
            pred_std: Predicted standard deviations [batch_size, output_size]
            
        Returns:
            Loss value
        """
        # Ensure numerical stability
        pred_std = pred_std.clamp(min=1e-9)
        
        # Calculate loss
        loss = 0.5 * (torch.log(pred_std**2) + ((target - pred_mean) / pred_std)**2).mean()
        return loss

def load_model(checkpoint_path: Path, config: Optional[Dict] = None) -> MultiHorizonTransformer:
    """
    Load a trained model from a checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        config: Model configuration (if None, will be loaded from checkpoint)
        
    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if config is None:
        config = checkpoint['config']
    
    model = MultiHorizonTransformer(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model
