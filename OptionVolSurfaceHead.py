import torch
import torch.nn as nn
import torch.nn.functional as F

class VolSurfaceHead(nn.Module):
    """
    Maps Transformer encoder outputs to implied volatility surface predictions.
    Typically a small MLP.
    """
    def __init__(self, embed_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # ensures output > 0
            # nn.ReLU() or just remove it (no activation on final layer)
        )

    def forward(self, x):
        """
        Args:
            x: (B, HW, D)
        Returns:
            (B, HW)
        """
        out = self.net[:-1](x)  # before Softplus
        # print("[DEBUG] Pre-softplus stats:", out.min().item(), out.max().item(), out.mean().item())
        out = self.net(x)
        return F.softplus(out.squeeze(-1))  # Ensures output > 0  # old: out.squeeze(-1)