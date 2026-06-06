"""
Real macro data downloader from Yahoo Finance (no yfinance lib needed).
Downloads SPY, DXY, GOLD, VIX daily candles → data/raw/macro/macro_features.csv

Bypasses geo-block: uses direct query1.finance.yahoo.com endpoint.
"""
import urllib.request
import json
import ssl
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ─── CONFIG ─────────────────────────────────────────────────────────────
TICKERS = {
    "spy":  "SPY",       # S&P 500 ETF
    "dxy":  "DX-Y.NYB",  # Dollar Index
    "gold": "GC=F",      # Gold futures
    "vix":  "^VIX",      # CBOE Volatility Index
}

# Span: last 5 years (covers all training/val/test ranges with margin)
END_TS = int(time.time())
START_TS = END_TS - 5 * 365 * 86400

OUT_DIR = "data/raw/macro"
OUT_CSV = os.path.join(OUT_DIR, "macro_features.csv")

# ─── HTTP CLIENT ────────────────────────────────────────────────────────
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
}

def fetch_yahoo(symbol: str, start: int, end: int, retries: int = 3) -> pd.DataFrame | None:
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?period1={start}&period2={end}&interval=1d&events=history"
    )
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            ctx = ssl.create_default_context()
            with urllib.request.urlopen(req, timeout=20, context=ctx) as r:
                data = json.loads(r.read())
            result = data["chart"]["result"][0]
            ts = result["timestamp"]
            quote = result["indicators"]["quote"][0]
            close = quote.get("close")
            volume = quote.get("volume", [None] * len(ts))
            df = pd.DataFrame({
                "timestamp": pd.to_datetime(ts, unit="s"),
                "close": close,
                "volume": volume,
            }).set_index("timestamp")
            df = df.dropna(subset=["close"])
            return df
        except Exception as e:
            last_err = e
            wait = 2 ** attempt
            print(f"  [{symbol}] attempt {attempt+1}/{retries} failed: {e}; retry in {wait}s")
            time.sleep(wait)
    print(f"  [{symbol}] FAILED after {retries} retries: {last_err}")
    return None

# ─── MAIN ───────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Downloading macro data for {list(TICKERS)} from "
          f"{datetime.fromtimestamp(START_TS):%Y-%m-%d} to "
          f"{datetime.fromtimestamp(END_TS):%Y-%m-%d}")

    series = {}
    for name, sym in TICKERS.items():
        print(f"\n[{name}] symbol={sym}")
        df = fetch_yahoo(sym, START_TS, END_TS)
        if df is None or len(df) < 30:
            print(f"  -> SKIP ({0 if df is None else len(df)} rows)")
            continue
        series[name] = df["close"].rename(name)
        print(f"  -> {len(df)} daily rows, {df.index.min():%Y-%m-%d} → {df.index.max():%Y-%m-%d}")

    if not series:
        print("ERROR: no macro series downloaded")
        sys.exit(1)

    macro_df = pd.concat(series.values(), axis=1).sort_index()
    macro_df.index.name = "date"
    macro_df = macro_df.ffill().dropna(how="all")

    macro_df.to_csv(OUT_CSV)
    print(f"\nSaved {len(macro_df)} rows × {macro_df.shape[1]} cols to {OUT_CSV}")
    print(f"Cols: {list(macro_df.columns)}")
    print(f"Range: {macro_df.index.min()} → {macro_df.index.max()}")
    print(macro_df.tail(3))

if __name__ == "__main__":
    main()
