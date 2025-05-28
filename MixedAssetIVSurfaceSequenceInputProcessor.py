import torch
import torch.nn as nn
from LearnedGrid2DPositionalEncoding import LearnedGrid2DPositionalEncoding
from SinusoidalTimePositionalEncoding import SinusoidalTimePositionalEncoding

class MixedAssetIVSurfaceSequenceInputProcessor(nn.Module):
    """
    Input processor for variable-A IV surface sequences with hybrid sector conditioning.
    
    Inputs:
        - input_iv:   (B, T, A, H, W)
        - sector_vec: (B, num_sectors)
    Output:
        - Embedded:   (B, T, A*H*W, D)
    """

    def __init__(self, embed_dim, max_seq_len, height, width, num_sectors, use_hybrid_sector_encoding=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.height = height
        self.width = width
        self.use_hybrid_sector_encoding = use_hybrid_sector_encoding

        self.input_proj = nn.Linear(1, embed_dim)
        self.grid_pos_enc = LearnedGrid2DPositionalEncoding(embed_dim, height, width)
        self.time_pos_enc = SinusoidalTimePositionalEncoding(embed_dim, max_len=max_seq_len)

        self.basket_sector_proj = nn.Linear(num_sectors, embed_dim)
        if self.use_hybrid_sector_encoding:
            self.asset_sector_proj = nn.Linear(num_sectors, embed_dim)

    def forward(
        self,
        input_iv: torch.Tensor,
        sector_vec: torch.Tensor,
        asset_sector_matrix: torch.Tensor = None,
        log_embeddings: bool = False
    ) -> torch.Tensor:
        """
        Args:
            input_iv:   (B, T, A, H, W)
            sector_vec: (B, num_sectors)
            asset_sector_matrix: (B, A, num_sectors) – optional
            log_embeddings: whether to print embedding samples for inspection
        Returns:
            Embedded:   (B, T, A*H*W, D)
        """
        B, T, A, H, W = input_iv.shape
    
        x = input_iv.clone()
        x[torch.isnan(x)] = 0.0
        x = x.unsqueeze(-1)  # (B, T, A, H, W, 1)
        x = self.input_proj(x)  # (B, T, A, H, W, D)
    
        grid_pe = self.grid_pos_enc(B * T * A)  # (B*T*A, H, W, D)
        grid_pe = grid_pe.view(B, T, A, H, W, self.embed_dim)
        x = x + grid_pe
    
        x = x.view(B, T, A * H * W, self.embed_dim)  # (B, T, A*H*W, D)
        x = self.time_pos_enc(x)  # (B, T, A*H*W, D)
    
        basket_emb = self.basket_sector_proj(sector_vec)  # (B, D)
    
        if self.use_hybrid_sector_encoding and asset_sector_matrix is not None:
            assert asset_sector_matrix.shape == (B, A, sector_vec.shape[-1]), \
                f"Expected asset_sector_matrix shape (B={B}, A={A}, num_sectors={sector_vec.shape[-1]}), got {asset_sector_matrix.shape}"
    
            asset_emb = self.asset_sector_proj(asset_sector_matrix)  # (B, A, D)
            asset_emb = asset_emb.unsqueeze(2).expand(-1, -1, H * W, -1)  # (B, A, H*W, D)
            asset_emb = asset_emb.reshape(B, A * H * W, self.embed_dim)  # (B, A*H*W, D)
            asset_emb = asset_emb.unsqueeze(1).expand(-1, T, -1, -1)     # (B, T, A*H*W, D)
    
            basket_emb = basket_emb.view(B, 1, 1, self.embed_dim).expand(-1, T, A * H * W, -1)  # (B, T, A*H*W, D)
            fused = x + asset_emb + basket_emb
        else:
            basket_emb = basket_emb.view(B, 1, 1, self.embed_dim)
            fused = x + basket_emb  # (B, T, A*H*W, D)
    
        if log_embeddings:
            print("[DEBUG] log_embeddings=True")
            print("  basket_emb[0,0,:5]:", basket_emb[0, 0, :5].detach().cpu().numpy())
            if self.use_hybrid_sector_encoding and asset_sector_matrix is not None:
                print("  asset_emb[0,:5,:5]:", asset_emb[0, :5, :5].detach().cpu().numpy())
    
        return fused