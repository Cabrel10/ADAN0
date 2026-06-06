"""
Resilient BTC daily downloader — Session 12.

Strategy: try multiple data sources in order, with optional HTTP proxy support
to bypass geo-restrictions on GitHub Actions runners. Saves to
data/raw/btc_daily/btc_daily.csv used by ExogenousRegimeOracle.fit().

Sources tried (first one that yields ≥200 daily bars wins):
  1. Bitget   (CCXT, paginated, no API key)
  2. Kraken   (CCXT, paginated, no API key)
  3. KuCoin   (CCXT, paginated, no API key)
  4. Yahoo    (urllib direct, BTC-USD)
  5. CoinGecko public REST (no key, rate-limited but global)

Optional env var:
  ADAN_PROXY=http://user:pass@host:port   → injected into both CCXT 'proxies' AND
                                            urllib Yahoo / CoinGecko requests.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Callable

import pandas as pd

try:
    import ccxt  # noqa: E402
except ImportError:
    ccxt = None  # the script still has Yahoo + CoinGecko fallbacks


# ────────────────────────────────────────────────────────────────────────────
# Common helpers
# ────────────────────────────────────────────────────────────────────────────

def get_proxies() -> dict | None:
    """Return a {'http': url, 'https': url} dict if ADAN_PROXY is set, else None."""
    proxy_url = os.environ.get("ADAN_PROXY", "").strip()
    if proxy_url:
        return {"http": proxy_url, "https": proxy_url}
    return None


def _urlopen_via_proxy(url: str, timeout: int = 30):
    """urllib opener honoring ADAN_PROXY."""
    proxies = get_proxies()
    if proxies:
        proxy_handler = urllib.request.ProxyHandler(proxies)
        opener = urllib.request.build_opener(proxy_handler)
    else:
        opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "Mozilla/5.0 (compatible; ADAN0/1.0)")]
    return opener.open(url, timeout=timeout)


# ────────────────────────────────────────────────────────────────────────────
# Source 1-3: CCXT exchanges
# ────────────────────────────────────────────────────────────────────────────

def _ccxt_fetch(exchange_name: str, symbol: str = "BTC/USDT",
                days: int = 730, batch_size: int = 200) -> pd.DataFrame:
    if ccxt is None:
        raise RuntimeError("ccxt not installed")
    ex_cls = getattr(ccxt, exchange_name, None)
    if ex_cls is None:
        raise RuntimeError(f"ccxt has no exchange '{exchange_name}'")

    cfg = {"rateLimit": 400, "enableRateLimit": True, "timeout": 20000}
    proxies = get_proxies()
    if proxies:
        cfg["proxies"] = proxies
    exchange = ex_cls(cfg)

    since = exchange.milliseconds() - days * 86400 * 1000
    rows: list[list[float]] = []
    batch = 0
    while batch < 60:  # safety cap
        chunk = None
        last_err = None
        for attempt in range(3):
            try:
                chunk = exchange.fetch_ohlcv(symbol, "1d", since=since, limit=batch_size)
                break
            except Exception as e:
                last_err = e
                time.sleep(1.5 * (attempt + 1))
        if chunk is None:
            raise RuntimeError(f"{exchange_name}: {last_err}")
        if not chunk:
            break
        rows.extend(chunk)
        batch += 1
        last_ts = chunk[-1][0]
        if len(chunk) < batch_size:
            break
        new_since = last_ts + 86400_000
        if new_since <= since:
            break
        since = new_since
        time.sleep(0.4)
    if not rows:
        raise RuntimeError(f"{exchange_name}: no candles")

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset=["ts"])
    df["date"] = pd.to_datetime(df["ts"], unit="ms").dt.normalize()
    return df.set_index("date").sort_index()


# ────────────────────────────────────────────────────────────────────────────
# Source 4: Yahoo Finance BTC-USD
# ────────────────────────────────────────────────────────────────────────────

def _yahoo_fetch(days: int = 730) -> pd.DataFrame:
    """Yahoo Finance BTC-USD daily, via the v7 download endpoint (no key)."""
    period2 = int(time.time())
    period1 = period2 - days * 86400
    url = (
        "https://query1.finance.yahoo.com/v7/finance/download/BTC-USD"
        f"?period1={period1}&period2={period2}&interval=1d&events=history"
    )
    with _urlopen_via_proxy(url, timeout=30) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    df = pd.read_csv(io.StringIO(body))
    # Columns: Date,Open,High,Low,Close,Adj Close,Volume
    if "Date" not in df.columns or "Close" not in df.columns:
        raise RuntimeError(f"unexpected Yahoo schema: {df.columns.tolist()}")
    df["date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                             "Close": "close", "Volume": "volume"})
    return df[["date", "open", "high", "low", "close", "volume"]].set_index("date").sort_index()


# ────────────────────────────────────────────────────────────────────────────
# Source 5: CoinGecko public REST
# ────────────────────────────────────────────────────────────────────────────

def _coingecko_fetch(days: int = 365) -> pd.DataFrame:
    """CoinGecko free tier — capped at 365 days for non-Pro keys."""
    days = min(days, 365)
    url = (
        "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
        f"?vs_currency=usd&days={days}&interval=daily"
    )
    with _urlopen_via_proxy(url, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8", errors="replace"))
    prices = data.get("prices", [])
    if not prices:
        raise RuntimeError("CoinGecko returned no prices")
    df = pd.DataFrame(prices, columns=["ts", "close"])
    df["date"] = pd.to_datetime(df["ts"], unit="ms").dt.normalize()
    df = df.drop_duplicates(subset=["date"]).set_index("date").sort_index()
    # synthesize open/high/low from close (only 'close' is critical for the Oracle)
    df["open"] = df["close"].shift(1).fillna(df["close"])
    df["high"] = df["close"]
    df["low"] = df["close"]
    df["volume"] = 0.0
    return df[["open", "high", "low", "close", "volume"]]


# ────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ────────────────────────────────────────────────────────────────────────────

def main() -> int:
    out_dir = Path("data/raw/btc_daily")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "btc_daily.csv"

    proxies = get_proxies()
    print(f"ADAN_PROXY: {'SET (' + list(proxies.values())[0][:30] + '...)' if proxies else 'NOT SET'}")

    sources: list[tuple[str, Callable[[], pd.DataFrame]]] = [
        ("bitget",     lambda: _ccxt_fetch("bitget", days=730)),
        ("kraken",     lambda: _ccxt_fetch("kraken", symbol="BTC/USD", days=730)),
        ("kucoin",     lambda: _ccxt_fetch("kucoin", days=730)),
        ("yahoo",      lambda: _yahoo_fetch(days=730)),
        ("coingecko",  lambda: _coingecko_fetch(days=365)),
    ]

    last_err = None
    for name, fn in sources:
        print(f"\n[try] {name}…")
        try:
            df = fn()
            if df is None or len(df) < 100:
                raise RuntimeError(f"too few rows ({0 if df is None else len(df)})")
            df_close = df[["close"]].copy()
            df_close = df_close[df_close["close"].notna()]
            df_close = df_close[df_close["close"] > 0]
            df_close.to_csv(out_csv)
            print(f"[OK] {name}: {len(df_close)} rows saved → {out_csv}")
            print(f"     range: {df_close.index.min().date()} → {df_close.index.max().date()}")
            print(df_close.tail(3))
            return 0
        except Exception as e:
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
            last_err = e
            continue

    # All sources failed → fall back to keeping any existing file as-is
    if out_csv.exists():
        print(f"\n[WARN] all sources failed; keeping existing {out_csv}")
        existing = pd.read_csv(out_csv)
        print(f"       existing rows: {len(existing)}")
        return 0

    print(f"\n[FATAL] all sources failed and no existing file. last={last_err}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
