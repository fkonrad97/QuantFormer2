import torch
import torch.nn as nn
import math

class SinusoidalTimePositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding for sequences (e.g., temporal IV surfaces).
    """

    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, T, ..., D) — supports (B, T, D) or (B, T, HW, D)
        Returns:
            Tensor of same shape with positional encoding added along time dimension
        """
        pe = self.pe[:, :x.size(1)].to(x.device)  # (1, T, D)
    
        if x.ndim == 3:
            # (B, T, D)
            return x + pe
        elif x.ndim == 4:
            # (B, T, HW, D) — need to expand PE to (1, T, 1, D)
            return x + pe.unsqueeze(2)  # Broadcast across HW
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")
