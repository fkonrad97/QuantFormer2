import torch
import torch.nn as nn
from TransformerEncoder import TransformerEncoder
from OptionVolSurfaceHead import VolSurfaceHead
from MixedAssetIVSurfaceSequenceInputProcessor import MixedAssetIVSurfaceSequenceInputProcessor

class MixedAssetSequenceIVSurfaceTransformerModel(nn.Module):
    """
    Transformer model for predicting IV surfaces with variable asset count per sample (mixed baskets).

    Inputs:
        - input_iv: (B, T, A, H, W)
        - sector_vec: (B, num_sectors)
        - asset_sector_matrix (optional): (B, A, num_sectors) – used only if hybrid encoding is enabled

    Output:
        - pred_iv: (B, A, H, W)
    """

    def __init__(
        self,
        height: int,
        width: int,
        embed_dim: int,
        num_heads: int,
        ff_hidden_dim: int,
        num_layers: int,
        num_sectors: int,
        max_seq_len: int,
        head_hidden_dim: int = 64,
        dropout: float = 0.1,
        use_hybrid_sector_encoding=False
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.use_hybrid_sector_encoding = use_hybrid_sector_encoding

        self.input_processor = MixedAssetIVSurfaceSequenceInputProcessor(
            embed_dim=embed_dim,
            max_seq_len=max_seq_len,
            height=height,
            width=width,
            num_sectors=num_sectors,
            use_hybrid_sector_encoding=use_hybrid_sector_encoding
        )

        self.encoder = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_hidden_dim=ff_hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        self.head = VolSurfaceHead(embed_dim, head_hidden_dim)

    def forward(self, input_iv, input_mask=None, sector_vec=None, asset_sector_matrix=None, verbose=False):
        B, T, A, H, W = input_iv.shape
        L = A * H * W

        x = self.input_processor(
            input_iv=input_iv,
            sector_vec=sector_vec,
            asset_sector_matrix=asset_sector_matrix if self.use_hybrid_sector_encoding else None,
            log_embeddings=verbose
        )  # (B, T, A*H*W, D)

        if verbose:
            print("[InputProcessor] output:", x.shape)

        x = x.view(B, T * L, -1)  # Flatten time × space
        x = self.encoder(x)       # (B, T*A*H*W, D)

        last_encoded = x[:, -L:]  # take only last time step's space
        pred = self.head(last_encoded)  # (B, A*H*W)
        pred = pred.view(B, A, H, W)    # reshape to surface

        return pred