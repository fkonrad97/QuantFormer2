from typing import List, Dict, Tuple, Union, Optional
import numpy as np
import pandas as pd
from RealIvSurfaceDataProcessorUtils import load_all_iv_surfaces
import torch

TICKER_TO_SECTOR = {
    "BA": "Industrials",
    "BAC": "Financials",
    "BLK": "Financials",
    "BP": "Energy",
    "C": "Financials",
    "CVX": "Energy",
    "GM": "Industrials",
    "HD": "Consumer Discretionary",
    "IWM": "Index",
    "JPM": "Financials",
    "KO": "Consumer Staples",
    "MS": "Financials",
    "PFE": "Pharma",
    "PG": "Household Products",
    "ROG": "Pharma",
    "SPY": "Index",
    "TSM": "Semiconductors",
    "UPS": "Industrials",
    "V": "Financials",
    "VIX": "Index",
    "WFC": "Financials",
    "WMT": "Retail",
    "XOM": "Energy",
    "TTE": "Energy",
    "COP": "Energy",
    "ENB": "Energy",
    "PBR": "Energy",
    "WMB": "Energy",
    "SAP": "Tech",
    "CRM": "Tech",
    "IBM": "Tech",
    "MA": "Financials",
    "AXP": "Financials",
    "BX": "Financials",
    "GS": "Financials",
    "UBS": "Financials",
    "BCS": "Financials",
    "DB": "Financials",
    "LLY": "Pharma",
    "JNJ": "Pharma",
    "UNH": "Pharma",
    "NVO": "Pharma",
    "NVS": "Pharma",
    "GE": "Industrials",
    "RTX": "Industrials",
    "CAT": "Industrials",
    "ETN": "Industrials",
    "LMT": "Industrials",
    "GEV": "Industrials",
    "UNP": "Industrials"
}

def compute_empirical_grid(
    df: pd.DataFrame,
    num_logm_points: int = 11,
    num_maturity_points: int = 7,
    lower_percentile: float = 2.5,
    upper_percentile: float = 97.5
) -> Tuple[np.ndarray, np.ndarray]:
    logm_series = df['log_moneyness'].dropna()
    maturity_series = df['maturity'].dropna()
    logm_bounds = np.percentile(logm_series, [lower_percentile, upper_percentile])
    maturity_bounds = np.percentile(maturity_series, [lower_percentile, upper_percentile])
    log_m_grid = np.linspace(logm_bounds[0], logm_bounds[1], num_logm_points)
    maturity_grid = np.linspace(maturity_bounds[0], maturity_bounds[1], num_maturity_points)
    return log_m_grid, maturity_grid

def compute_quantile_grid(
    df: pd.DataFrame,
    num_logm_points: int = 11,
    num_maturity_points: int = 7
) -> Tuple[np.ndarray, np.ndarray]:
    log_m_grid = df['log_moneyness'].dropna().quantile(
        np.linspace(0.01, 0.99, num_logm_points)).values
    maturity_grid = df['maturity'].dropna().quantile(
        np.linspace(0.01, 0.99, num_maturity_points)).values
    return log_m_grid, maturity_grid

def are_dates_consecutive(dates: List[pd.Timestamp], max_gap_days: int = 3) -> bool:
    if len(dates) < 2:
        return True
    sorted_dates = sorted(dates)
    deltas = [(t2 - t1).days for t1, t2 in zip(sorted_dates[:-1], sorted_dates[1:])]
    return all(delta <= max_gap_days for delta in deltas)

def build_iv_matrix_with_mask(
    df: pd.DataFrame,
    log_m_grid: np.ndarray,
    maturity_grid: np.ndarray,
    method: str = "linear"
) -> Tuple[np.ndarray, np.ndarray]:
    from scipy.interpolate import griddata

    df = df.dropna(subset=['log_moneyness', 'maturity', 'iv'])
    if df.empty:
        shape = (len(maturity_grid), len(log_m_grid))
        return np.full(shape, np.nan), np.zeros(shape, dtype=np.uint8)

    points = df[['log_moneyness', 'maturity']].values
    values = df['iv'].values

    # Create grid to interpolate over
    grid_x, grid_y = np.meshgrid(log_m_grid, maturity_grid)  # shape (H, W)
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])  # shape (H*W, 2)

    try:
        iv_grid = griddata(points, values, grid_points, method=method)
        # print(f"[INFO] Interpolation succeeded for!")
    except Exception as e:
        print(f"[WARN] Interpolation failed: {e}")
        iv_grid = np.full(len(grid_points), np.nan)

    # Debug: confirm shape before reshaping
    expected_len = len(log_m_grid) * len(maturity_grid)
    if iv_grid is None or iv_grid.shape[0] != expected_len:
        raise ValueError(
            f"[ERROR] Unexpected iv_grid shape: {iv_grid.shape} — expected ({expected_len},). "
            f"Grid might be too sparse or interpolation failed."
        )

    iv_matrix = iv_grid.reshape(len(maturity_grid), len(log_m_grid))  # shape (H, W)

    # Final safety check
    assert iv_matrix.ndim == 2, f"[ERROR] Got invalid iv_matrix shape: {iv_matrix.shape}"

    mask = ~np.isnan(iv_matrix)
    return iv_matrix, mask.astype(np.uint8)

def generate_multi_asset_iv_surface_dataset_with_masks(
    df_all: pd.DataFrame,
    baskets: List[Dict],  # Each: {"name": ..., "tickers": [...], "sector_vector": [...]}
    log_m_grid: np.ndarray,
    maturity_grid: np.ndarray,
    sequence_length: int = 4,
    max_gap_days: int = 3
) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp], List[str], List[List[int]]]:
    """
    Builds IV surface sequences for multi-asset baskets.
    
    Returns:
    - iv_tensor: shape (N, T, A, H, W)
    - mask_tensor: same
    - dates: label dates
    - basket_names
    - basket_sector_vectors
    """
    df_all['date'] = pd.to_datetime(df_all['date'])
    df_all['log_moneyness'] = np.log(df_all['strike'] / df_all['forward'])

    iv_tensors = []
    mask_tensors = []
    label_dates = []
    basket_names = []
    basket_sector_vectors = []

    for basket in baskets:
        tickers = basket['tickers']
        basket_name = basket['name']
        sector_vector = basket['sector_vector']

        # Get intersection of dates across all assets in basket
        asset_date_sets = []
        for ticker in tickers:
            df_t = df_all[df_all['ticker'] == ticker]
            asset_date_sets.append(set(df_t['date'].unique()))
        common_dates = sorted(set.intersection(*asset_date_sets))

        for i in range(len(common_dates) - sequence_length + 1):
            window_dates = common_dates[i:i + sequence_length]
            if not are_dates_consecutive(window_dates, max_gap_days=max_gap_days):
                continue

            iv_seq = []
            mask_seq = []
            skip = False

            for date in window_dates:
                iv_matrices_at_t = []
                mask_matrices_at_t = []

                for ticker in tickers:
                    df_day = df_all[(df_all['ticker'] == ticker) & (df_all['date'] == date)]
                    iv_matrix, mask = build_iv_matrix_with_mask(df_day, log_m_grid, maturity_grid)
                    if np.isnan(iv_matrix).all():
                        skip = True
                        break
                    iv_matrices_at_t.append(iv_matrix)
                    mask_matrices_at_t.append(mask)

                if skip:
                    break

                iv_seq.append(np.stack(iv_matrices_at_t))   # (A, H, W)
                mask_seq.append(np.stack(mask_matrices_at_t))  # (A, H, W)

            if skip:
                continue

            iv_tensor = np.stack(iv_seq)   # (T, A, H, W)
            mask_tensor = np.stack(mask_seq)  # (T, A, H, W)

            iv_tensors.append(iv_tensor)
            mask_tensors.append(mask_tensor)
            label_dates.append(window_dates[-1])
            basket_names.append(basket_name)
            basket_sector_vectors.append(sector_vector)

    if not iv_tensors:
        raise ValueError("[ERROR] No valid IV sequences found for any basket")

    X = np.stack(iv_tensors)
    M = np.stack(mask_tensors)
    return X, M, label_dates, basket_names, basket_sector_vectors

def load_prepared_multi_asset_iv_surface_dataset(
    data_dir: str,
    baskets: List[Dict],
    log_m_grid: np.ndarray = None,
    maturity_grid: np.ndarray = None,
    sequence_length: int = 4,
    max_gap_days: int = 3,
    start_date: str = "2024-01-01",
    end_date: str = "2025-04-30"
) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp], List[str], List[List[int]]]:
    """
    Wrapper to load and process IV surface sequences for multi-asset baskets.

    Returns:
    - X: IV surface tensor of shape (N, T, A, H, W)
    - M: mask tensor, same shape
    - dates: label dates (length N)
    - basket_names: list of basket names (length N)
    - sector_vectors: list of one-hot sector vectors (length N)
    """
    # Flatten all unique tickers across all baskets
    all_basket_tickers = sorted(set(ticker for b in baskets for ticker in b["tickers"]))

    # Load raw surface data
    df_all = load_all_iv_surfaces(
        data_dir,
        instruments=all_basket_tickers,
        start_date=start_date,
        end_date=end_date
    )

    # Compute grids if not given
    if log_m_grid is None or maturity_grid is None:
        log_m_grid, maturity_grid = get_default_grid()

    return generate_multi_asset_iv_surface_dataset_with_masks(
        df_all=df_all,
        baskets=baskets,
        log_m_grid=log_m_grid,
        maturity_grid=maturity_grid,
        sequence_length=sequence_length,
        max_gap_days=max_gap_days
    )

def generate_mixed_asset_iv_surface_dataset_with_masks(
    df_all: pd.DataFrame,
    baskets: List[Dict],
    log_m_grid: np.ndarray,
    maturity_grid: np.ndarray,
    sequence_length: int = 4,
    max_gap_days: int = 3
) -> Tuple[
    List[np.ndarray],  # IV tensors
    List[np.ndarray],  # masks
    List[pd.Timestamp],
    List[str],
    List[List[int]],   # basket sector vecs
]:
    df_all['date'] = pd.to_datetime(df_all['date'])
    df_all['log_moneyness'] = np.log(df_all['strike'] / df_all['forward'])

    iv_tensors, mask_tensors = [], []
    label_dates, basket_names = [], []
    basket_sector_vectors = []
    aggregate_outputs = []
    weights_list = []

    for basket in baskets:
        tickers = basket["tickers"]
        name = basket["name"]
        sector_vector = basket["sector_vector"]

        df_basket = df_all[df_all["ticker"].isin(tickers)]
        sorted_dates = sorted(df_basket["date"].unique())

        for i in range(len(sorted_dates) - sequence_length + 1):
            window_dates = sorted_dates[i:i + sequence_length]
            if not are_dates_consecutive(window_dates, max_gap_days=max_gap_days):
                continue

            iv_stack, mask_stack = [], []
            skip = False

            for t in window_dates:
                iv_day, mask_day = [], []
                for ticker in tickers:
                    df_day = df_basket[(df_basket["ticker"] == ticker) & (df_basket["date"] == t)]
                    iv_matrix, mask = build_iv_matrix_with_mask(df_day, log_m_grid, maturity_grid)
                    if np.isnan(iv_matrix).all():
                        skip = True
                        print(f"[SKIP-NaN] Entire surface is NaN for {ticker} on {t}")
                        break
                    # print(f"[INFO] Interplation was a success for: {ticker} on {t}")
                    iv_day.append(iv_matrix)
                    mask_day.append(mask)
                if skip:
                    break

                iv_stack.append(np.stack(iv_day))
                mask_stack.append(np.stack(mask_day))

            if skip:
                continue

            iv_tensor = np.stack(iv_stack)         # (T, A, H, W)
            mask_tensor = np.stack(mask_stack)     # (T, A, H, W)

            final_mask = mask_tensor[-1] > 0
            final_iv = iv_tensor[-1][final_mask]

            # After: final_iv = iv_tensor[-1][final_mask]
            if final_iv.size == 0 or np.allclose(final_iv, 0.0) or np.isnan(iv_tensor[-1]).all():
                print(f"[SKIP] Empty, zero, or NaN target IV at {name}, {window_dates[-1]}")
                continue
            
            assert np.any(final_mask), f"[ERROR] All-zero mask unexpectedly passed for {name} at {window_dates[-1]}"
            aggregate_outputs.append(basket.get("aggregate_output", False))
            weights_list.append(basket.get("weights", None))
            iv_tensors.append(iv_tensor)
            mask_tensors.append(mask_tensor)
            label_dates.append(window_dates[-1])
            basket_names.append(name)
            basket_sector_vectors.append(sector_vector)

    assert all(np.any(m[-1] > 0) for m in mask_tensors), "[ERROR] Output contains empty target masks!"
    return (
        iv_tensors,
        mask_tensors,
        label_dates,
        basket_names,
        basket_sector_vectors,
        aggregate_outputs,
        weights_list
    )

def load_prepared_mixed_asset_iv_surface_dataset(
    data_dir: str,
    baskets: List[Dict],
    log_m_grid: np.ndarray = None,
    maturity_grid: np.ndarray = None,
    sequence_length: int = 4,
    max_gap_days: int = 3,
    start_date: str = "2024-01-01",
    end_date: str = "2025-04-30"
) -> Tuple[
    List[np.ndarray],         # iv_tensors
    List[np.ndarray],         # mask_tensors
    List[pd.Timestamp],       # label_dates
    List[str],                # basket_names
    List[List[int]],          # sector_vectors
    List[bool],               # aggregate_outputs
    List[Optional[List[float]]]  # weights
]:
    """
    Wrapper for loading and processing mixed-asset basket IV sequences.
    Returns:
        - List of IV tensors: (T, A, H, W)
        - List of mask tensors: same
        - List of label dates
        - List of basket names
        - List of sector one-hot vectors
    """
    # Load all IV surface data across tickers
    all_tickers = sorted(set(ticker for basket in baskets for ticker in basket["tickers"]))
    df_all = load_all_iv_surfaces(
        data_dir,
        instruments=all_tickers,
        start_date=start_date,
        end_date=end_date
    )
    df_all['log_moneyness'] = np.log(df_all['strike'] / df_all['forward'])

    # Use default grid if not provided
    if log_m_grid is None or maturity_grid is None:
        log_m_grid, maturity_grid = get_default_grid()

    return generate_mixed_asset_iv_surface_dataset_with_masks(
        df_all=df_all,
        baskets=baskets,
        log_m_grid=log_m_grid,
        maturity_grid=maturity_grid,
        sequence_length=sequence_length,
        max_gap_days=max_gap_days
    )

def aggregate_surfaces(
    iv_tensor: torch.Tensor,  # shape: (B, A, H, W)
    weights: torch.Tensor = None  # shape: (B, A) or (A,)
) -> torch.Tensor:
    """
    Aggregates asset-level surfaces into a single basket-level surface per sample.

    Args:
        iv_tensor: Tensor of shape (B, A, H, W)
        weights: Optional weights, shape (B, A) or (A,)

    Returns:
        Tensor of shape (B, H, W) — aggregated basket surfaces
    """
    if weights is None:
        return iv_tensor.mean(dim=1)  # simple average
    if weights.dim() == 1:
        weights = weights.unsqueeze(0).expand(iv_tensor.size(0), -1)  # (A,) → (B, A)
    weights = weights.unsqueeze(-1).unsqueeze(-1)  # (B, A, 1, 1)
    return (iv_tensor * weights).sum(dim=1)  # weighted sum

def get_sector_one_hot_matrix(tickers: List[str]) -> np.ndarray:
    sectors = sorted(set(TICKER_TO_SECTOR.values()))
    sector_to_idx = {sector: i for i, sector in enumerate(sectors)}
    matrix = np.zeros((len(tickers), len(sectors)))

    for i, ticker in enumerate(tickers):
        sector = TICKER_TO_SECTOR.get(ticker)
        if sector in sector_to_idx:
            matrix[i, sector_to_idx[sector]] = 1.0
    return matrix