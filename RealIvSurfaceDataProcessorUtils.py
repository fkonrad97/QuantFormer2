import os
import pandas as pd
import numpy as np

def load_all_iv_surfaces(folder_root_path, instruments=None, start_date=None, end_date=None):
    """
    Loads and combines IV surfaces from all CSVs in a folder (and subfolders),
    optionally filtered by ticker symbols and a date range.

    Args:
        folder_root_path (str): Root folder (e.g., 'data/real_iv_surfaces')
        instruments (list[str], optional): If provided, only load surfaces for these tickers.
        start_date (str or pd.Timestamp, optional): Start date (inclusive) to filter files.
        end_date (str or pd.Timestamp, optional): End date (inclusive) to filter files.

    Returns:
        pd.DataFrame: Combined surface data with inferred columns, including 'ticker' and 'date'
    """
    all_records = []

    # Parse optional dates
    if start_date:
        start_date = pd.to_datetime(start_date)
    if end_date:
        end_date = pd.to_datetime(end_date)

    for root, _, files in os.walk(folder_root_path):
        for file in files:
            if file.endswith('.csv') and "_iv_surface" in file:
                try:
                    parts = file.split("_")
                    if len(parts) < 3:
                        print(f"Invalid filename format: {file}")
                        continue

                    ticker = parts[0]
                    date_str = parts[1]
                    date = pd.to_datetime(date_str)

                    # Apply instrument filter
                    if instruments is not None and ticker not in instruments:
                        continue

                    # Apply date filters
                    if start_date and date < start_date:
                        continue
                    if end_date and date > end_date:
                        continue

                    filepath = os.path.join(root, file)
                    df = pd.read_csv(filepath)

                    df['ticker'] = ticker
                    df['date'] = date

                    if 'expiry' in df.columns:
                        df['expiry'] = pd.to_datetime(df['expiry'], errors='coerce')

                    all_records.append(df)

                except Exception as e:
                    print(f"Skipping {file}: {e}")

    if not all_records:
        print("No IV surface files found. Check folder path, filters, or date range.")
        return pd.DataFrame()

    print(f"Loaded {len(all_records)} files into dataframe")

    return pd.concat(all_records, ignore_index=True)


def build_iv_dataset(folder_path: str, tickers: list[str], method: str = "mean") -> pd.DataFrame:
    """
    Constructs a multi-asset or single asset implied volatility dataset from single-asset IV surfaces.

    Args:
        folder_path (str): Path to root folder containing CSVs per asset in subfolders.
        tickers (list[str]): List of tickers to combine (e.g., ["MS", "JPM", "AAPL"]).
        method (str): Aggregation method to combine IVs. Options: "mean", "median".

    Returns:
        pd.DataFrame: DataFrame with columns [strike_1, ..., strike_N, maturity, iv, date, tickers]
    """
    # Load all CSVs per ticker
    surfaces = {}
    for ticker in tickers:
        ticker_dir = os.path.join(folder_path, ticker)
        if not os.path.exists(ticker_dir):
            raise ValueError(f"Ticker folder not found: {ticker_dir}")
        dfs = []
        for file in os.listdir(ticker_dir):
            if file.endswith(".csv") and "_iv_surface" in file:
                df = pd.read_csv(os.path.join(ticker_dir, file))
                df["date"] = file.split("_")[1]
                dfs.append(df[["strike", "maturity", "iv", "date"]])
        surfaces[ticker] = pd.concat(dfs, ignore_index=True)

    # Find common dates
    date_sets = [set(df["date"].unique()) for df in surfaces.values()]
    common_dates = set.intersection(*date_sets)

    records = []
    for date in sorted(common_dates):
        per_ticker_grids = {}
        for ticker in tickers:
            df = surfaces[ticker]
            df_date = df[df["date"] == date]
            grouped = df_date.groupby("maturity")

            for maturity, group in grouped:
                strikes = group["strike"].values
                ivs = group["iv"].values
                per_ticker_grids.setdefault(maturity, {})[ticker] = (strikes, ivs)

        # Only keep maturities common to all tickers
        for maturity, ticker_data in per_ticker_grids.items():
            if set(ticker_data.keys()) != set(tickers):
                continue

            # Build strike meshgrid
            strike_lists = [ticker_data[ticker][0] for ticker in tickers]
            iv_lists = [ticker_data[ticker][1] for ticker in tickers]

            # Ensure strike grids are the same size
            min_len = min(len(s) for s in strike_lists)
            strike_lists = [s[:min_len] for s in strike_lists]
            iv_lists = [iv[:min_len] for iv in iv_lists]

            # Combine into grid
            mesh = np.stack(strike_lists, axis=-1)
            iv_stack = np.stack(iv_lists, axis=-1)

            if method == "mean":
                iv_combined = iv_stack.mean(axis=1)
            elif method == "median":
                iv_combined = np.median(iv_stack, axis=1)
            else:
                raise ValueError(f"Unsupported method: {method}")

            for i in range(min_len):
                row = list(mesh[i]) + [maturity, iv_combined[i], date, "_".join(tickers)]
                records.append(row)

    strike_cols = [f"strike_{i+1}" for i in range(len(tickers))]
    columns = strike_cols + ["maturity", "iv", "date", "tickers"]
    return pd.DataFrame(records, columns=columns)