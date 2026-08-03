"""
Train the ExogenousRegimeOracle on REAL data and validate it.

Inputs:
  - data/raw/btc_daily/btc_daily.csv   (from scripts/download_btc_daily.py)
  - data/raw/macro/macro_features.csv  (from scripts/download_macro_data.py)

Outputs:
  - models/exog_oracle.pkl
  - models/exog_oracle_metadata.json   (training stats for audit)
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("oracle_trainer")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from adan_trading_bot.environment.exogenous_regime_oracle import ExogenousRegimeOracle  # noqa: E402


def load_btc() -> pd.Series:
    candidates = [
        REPO_ROOT / "data" / "raw" / "btc_daily" / "btc_daily.csv",
    ]
    for p in candidates:
        if p.is_file():
            df = pd.read_csv(p, parse_dates=["date"], index_col="date")
            s = df["close"].dropna()
            logger.info(f"BTC daily loaded: {len(s)} days from {p}")
            return s
    raise FileNotFoundError(
        "data/raw/btc_daily/btc_daily.csv missing — run scripts/download_btc_daily.py first"
    )


def load_macro() -> pd.DataFrame:
    p = REPO_ROOT / "data" / "raw" / "macro" / "macro_features.csv"
    if not p.is_file():
        logger.warning("macro_features.csv missing — oracle will be BTC-only")
        return pd.DataFrame()
    df = pd.read_csv(p, parse_dates=["date"], index_col="date")
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    logger.info(f"Macro daily loaded: {len(df)} days, cols={list(df.columns)} from {p}")
    return df


def validate_oracle(oracle: ExogenousRegimeOracle, n: int = 100) -> dict:
    rng = np.random.default_rng(42)
    preds = []
    for _ in range(n):
        feat = rng.standard_normal(oracle.scaler.n_features_in_) * 0.05
        preds.append(tuple(np.round(oracle.predict_proba_safe(feat), 3)))
    unique = len(set(preds))
    return {
        "n_samples_tested": n,
        "unique_predictions": unique,
        "diverse": unique > 5,
        "feature_count": int(oracle.scaler.n_features_in_),
    }


def main() -> int:
    out_path = REPO_ROOT / "models" / "exog_oracle.pkl"
    out_meta = REPO_ROOT / "models" / "exog_oracle_metadata.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    btc = load_btc()
    macro = load_macro()

    oracle = ExogenousRegimeOracle()
    oracle.fit(macro, btc)
    if not oracle.is_fitted:
        logger.error("Oracle did NOT fit — not enough data?")
        return 2
    oracle.save(str(out_path))

    val = validate_oracle(oracle)
    meta = {
        "btc_days": int(len(btc)),
        "btc_date_min": str(btc.index.min().date()),
        "btc_date_max": str(btc.index.max().date()),
        "macro_days": int(len(macro)),
        "macro_cols": list(macro.columns),
        "validation": val,
    }
    out_meta.write_text(json.dumps(meta, indent=2))

    logger.info(f"Oracle saved: {out_path}")
    logger.info(f"Metadata: {out_meta}")
    print(json.dumps(meta, indent=2))

    if not val["diverse"]:
        logger.error(f"Oracle DEGENERATE: only {val['unique_predictions']} unique predictions")
        return 3
    logger.info(f"PASS: {val['unique_predictions']}/100 unique predictions on random input")
    return 0


if __name__ == "__main__":
    sys.exit(main())
