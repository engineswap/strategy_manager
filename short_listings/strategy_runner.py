#!/usr/bin/env python3
"""
generate_last_day_weights.py

Reads OHLC data for Binance and Bybit perpetuals, computes daily equal-weight
short signals for newly listed coins, and saves only the last day's non-zero weight
per symbol to disk as CSV files in the specified output directory. Each
output file is named 'weights_short_listings_<exchange>.csv' and contains:

    symbol,weight
    AERGO,-0.1601
    LAYER,-0.1357
    BTC,1.3000
    ...
"""

import os
import pandas as pd
import numpy as np

# Paths to input CSVs (update these if your data location changes)
BINANCE_PATH = "/Users/ilyat/Documents/quant📈/ccxt_data/data/binance/ohlc/perps/combined_enriched.csv"
BYBIT_PATH   = "/Users/ilyat/Documents/quant📈/ccxt_data/data/bybit/ohlc/perps/combined_enriched.csv"

# Output directory for last-day weights
OUTPUT_DIR = "/Users/ilyat/Desktop/xs_research/ready_strategies/weights"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_last_day_weights(input_path: str, exchange_name: str) -> None:
    """
    Reads the CSV at input_path, computes daily equal-weight short weights for each symbol,
    adds a constant BTC hedge of +1.3, then extracts only the last day's non-zero weights
    and saves them to:
        OUTPUT_DIR/weights_short_listings_<exchange_name>.csv
    The saved CSV has two columns: symbol,weight.
    """
    # 1. Load data
    df = pd.read_csv(input_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['timestamp', 'symbol'])

    # 2. Compute true days since listing (first-ever timestamp per symbol)
    first_list = df.groupby('symbol')['timestamp'].transform('min')
    df['days_since_list'] = (df['timestamp'] - first_list).dt.days

    # 3. Generate short signal: -1 if days_since_list < 100, else 0
    df['signal'] = np.where(df['days_since_list'] < 100, -1, 0)

    # 4. Filter to date range starting from 2025-01-01 (inclusive)
    df = df[df['timestamp'] > '2025-01-01']

    # 5. Compute daily pct returns per symbol
    df['ret'] = df.groupby('symbol')['close'].pct_change()

    # 6. Compute equal-weight short weights on each date (avoid deprecation warning)
    #    For each timestamp, count how many symbols have signal == -1, then assign -1/n to those.
    weights = (
        df.groupby('timestamp', group_keys=False)['signal']
          .apply(lambda sig: pd.Series(
              np.where(sig == -1, -1.0 / sig.eq(-1).sum(), 0.0),
              index=sig.index
          ))
    )
    df['weight'] = weights.fillna(0.0)

    # 7. Pivot weights to wide format (timestamps × symbols), fill missing with 0
    weights_df = df.pivot(index='timestamp', columns='symbol', values='weight').fillna(0.0)

    # 8. Add constant BTC hedge of +1.3 (overrides any existing BTC weight)
    weights_df['BTC'] = 1.3

    # 9. Extract last day's weights
    if weights_df.empty:
        print(f"No weights computed for {exchange_name} (empty DataFrame).")
        return

    last_timestamp = weights_df.index.max()
    last_weights = weights_df.loc[last_timestamp].round(4)

    # Convert to two-column DataFrame: symbol, weight; then filter non-zero weights
    last_weights_df = last_weights.reset_index()
    last_weights_df.columns = ['symbol', 'weight']
    nonzero_weights_df = last_weights_df[last_weights_df['weight'] != 0.0]

    # 10. Save to CSV (no index, header included)
    filename = f"weights_short_listings_{exchange_name}.csv"
    output_path = os.path.join(OUTPUT_DIR, filename)
    nonzero_weights_df.to_csv(output_path, index=False)
    print(f"Non-zero last-day weights ({last_timestamp.date()}) saved to: {output_path}")

if __name__ == "__main__":
    # Generate and save last-day non-zero weights for Binance
    generate_last_day_weights(BINANCE_PATH, "binance")

    # Generate and save last-day non-zero weights for Bybit
    generate_last_day_weights(BYBIT_PATH, "bybit")
