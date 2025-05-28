import torch
import torch.nn as nn

class LearnedGrid2DPositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, height: int, width: int):
        super().__init__()
        self.height = height
        self.width = width
        self.embed_dim = embed_dim

        self.row_embed = nn.Embedding(height, embed_dim // 2)
        self.col_embed = nn.Embedding(width, embed_dim // 2)

        # Precompute and register grid indices as buffers
        self.register_buffer("h_pos", torch.arange(height).long())  # shape (H,)
        self.register_buffer("w_pos", torch.arange(width).long())   # shape (W,)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, B: int):
        h_embed = self.row_embed(self.h_pos)  # (H, D/2)
        w_embed = self.col_embed(self.w_pos)  # (W, D/2)

        grid = torch.cat([
            h_embed.unsqueeze(1).expand(-1, self.width, -1),   # (H, W, D/2)
            w_embed.unsqueeze(0).expand(self.height, -1, -1)   # (H, W, D/2)
        ], dim=-1)  # (H, W, D)

        return grid.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, D)