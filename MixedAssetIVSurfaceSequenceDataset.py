import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional
import pandas as pd
import torch.nn.functional as F
import numpy as np
from GeneralIvVolMatrixUtils import TICKER_TO_SECTOR

class MixedAssetIVSurfaceSequenceDataset(Dataset):
    def __init__(
        self,
        iv_matrices: List,  # (T, A, H, W)
        masks: List,
        dates: List[pd.Timestamp],
        basket_names: List[str],
        sector_vectors: List[List[int]],
        aggregate_outputs: Optional[List[bool]] = None,
        weights: Optional[List[List[float]]] = None,
        basket_ticker_map: Optional[Dict[str, List[str]]] = None
    ):
        self.iv_matrices = iv_matrices
        self.masks = masks
        self.dates = dates
        self.basket_names = basket_names
        self.sector_vectors = sector_vectors
        self.aggregate_outputs = aggregate_outputs if aggregate_outputs is not None else [False] * len(iv_matrices)
        self.weights = weights if weights is not None else [None] * len(iv_matrices)
        self.basket_ticker_map = basket_ticker_map or {}

        self.unique_sectors = sorted(set(TICKER_TO_SECTOR.values()))
        self.sector_to_idx = {s: i for i, s in enumerate(self.unique_sectors)}

    def get_asset_sector_matrix(self, basket_name: str) -> torch.Tensor:
        tickers = self.basket_ticker_map.get(basket_name, [])
        matrix = torch.zeros((len(tickers), len(self.unique_sectors)))
        for i, t in enumerate(tickers):
            s = TICKER_TO_SECTOR.get(t)
            if s:
                matrix[i, self.sector_to_idx[s]] = 1.0
        return matrix

    def __len__(self):
        return len(self.iv_matrices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        input_iv = torch.tensor(self.iv_matrices[idx]).float()
        input_iv[torch.isnan(input_iv)] = 0.0
        input_mask = torch.tensor(self.masks[idx]).float()
        target_iv = input_iv[-1]
        target_mask = input_mask[-1]
        basket_name = self.basket_names[idx]

        if self.aggregate_outputs[idx]:  # basket-level output
            # If *any* asset is fully masked, skip
            if (target_mask.sum(dim=(1, 2)) == 0).any():  # shape: (A,)
                return None
        else:
            # Per-asset case — skip only if *all* assets are masked
            if target_mask.sum() == 0:
                return None

        if torch.all(target_mask == 0):
            print(f"[WARN] Empty target mask slipped into __getitem__! Basket: {basket_name}, Date: {self.dates[idx]}")

        return {
            "input_iv": input_iv,
            "input_mask": input_mask,
            "target_iv": target_iv,
            "target_mask": target_mask,
            "sector_vec": torch.tensor(self.sector_vectors[idx]).float(),
            "asset_sector_matrix": self.get_asset_sector_matrix(basket_name), 
            "aggregate_output": self.aggregate_outputs[idx],
            "weights": torch.tensor(self.weights[idx]).float() if self.weights[idx] is not None else None,
            "meta": {
                "date": self.dates[idx],
                "basket": basket_name,
                "num_assets": input_iv.shape[1]
            }
        }
    
def mixed_asset_iv_surface_collate_fn(batch, device=None):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    
    T = batch[0]['input_iv'].shape[0]
    H = batch[0]['input_iv'].shape[2]
    W = batch[0]['input_iv'].shape[3]
    max_A = max(item['input_iv'].shape[1] for item in batch)
    num_sectors = batch[0]['asset_sector_matrix'].shape[1]

    def pad(t, target_shape):
        pad_dims = []
        for i in reversed(range(len(t.shape))):
            diff = target_shape[i] - t.shape[i]
            pad_dims.extend([0, diff])
        return F.pad(t, pad_dims, value=0.0)

    batch_dict = {
        'input_iv': [], 'input_mask': [], 'target_iv': [], 'target_mask': [],
        'sector_vec': [], 'aggregate_output': [], 'weights': [],
        'asset_sector_matrix': [], 'meta': []
    }

    for item in batch:
        A = item['input_iv'].shape[1]
        batch_dict['input_iv'].append(pad(item['input_iv'], (T, max_A, H, W)))
        batch_dict['input_mask'].append(pad(item['input_mask'], (T, max_A, H, W)))
        batch_dict['target_iv'].append(pad(item['target_iv'], (max_A, H, W)))
        batch_dict['target_mask'].append(pad(item['target_mask'], (max_A, H, W)))
        batch_dict['sector_vec'].append(item['sector_vec'])
        batch_dict['asset_sector_matrix'].append(pad(item['asset_sector_matrix'], (max_A, num_sectors)))
        batch_dict['aggregate_output'].append(item['aggregate_output'])
        if item['weights'] is not None:
            padded_weights = F.pad(item['weights'], (0, max_A - item['weights'].shape[0]), value=0.0)
        else:
            padded_weights = torch.zeros(max_A, device=item['input_iv'].device)
        batch_dict['weights'].append(padded_weights)
        batch_dict['meta'].append(item['meta'])

    to_tensor = lambda x: torch.stack(x).to(device) if device else torch.stack(x)

    for i, m in enumerate(batch_dict["target_mask"]):
        if m.sum() == 0:
            print(f"[COLLATE-DEBUG] Empty mask in batch index {i}, Meta: {batch_dict['meta'][i]}")

    return {
        'input_iv': to_tensor(batch_dict['input_iv']),
        'input_mask': to_tensor(batch_dict['input_mask']),
        'target_iv': to_tensor(batch_dict['target_iv']),
        'target_mask': to_tensor(batch_dict['target_mask']),
        'sector_vec': to_tensor(batch_dict['sector_vec']),
        'asset_sector_matrix': to_tensor(batch_dict['asset_sector_matrix']),
        'aggregate_output': torch.tensor(batch_dict['aggregate_output'], dtype=torch.bool, device=device),
        'weights': to_tensor(batch_dict['weights']),
        'meta': batch_dict['meta']
    }