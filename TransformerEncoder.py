import torch.nn as nn
from TransformerBlock import TransformerBlock

class TransformerEncoder(nn.Module):
    """
    Stacks multiple Transformer blocks.
    Accepts a custom block class that implements the BaseBlock interface.
    """
    def __init__(self, 
                 embed_dim: int, 
                 num_heads: int, 
                 ff_hidden_dim: int, 
                 num_layers: int = 6,
                 dropout: float = 0.1,
                 block_cls: type = TransformerBlock,
                 **block_kwargs):
        super().__init__()

        self.layers = nn.ModuleList([
            block_cls(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_hidden_dim=ff_hidden_dim,
                dropout=dropout,
                **block_kwargs
            )
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        # print("[ENCODER] output mean/std:", x.mean().item(), x.std().item())
        for layer in self.layers:
            x = layer(x, mask)
        return x
