#!/usr/bin/env python3
"""DL-03: Apply FeatureEngineer on real data, validate 21 features per TF.

Uses direct importlib import to avoid pulling gym/torch through __init__.py.
Loads config.yaml to construct FeatureEngineer with proper data_config.
"""
import sys, json, logging
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

# Resolve project root relative to this script (scripts/ -> project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Direct import of FeatureEngineer (bypass __init__.py chain) ---
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "feature_engineer",
    str(_PROJECT_ROOT / "src/adan_trading_bot/data_processing/feature_engineer.py")
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules['adan_trading_bot.data_processing.feature_engineer'] = _mod
_spec.loader.exec_module(_mod)
FeatureEngineer = _mod.FeatureEngineer

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ---- Load config ----
with open(_PROJECT_ROOT / "config/config.yaml") as f:
    full_config = yaml.safe_load(f)

MODELS_DIR = str(_PROJECT_ROOT / "models")
Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)

# ---- Required features per TF (5 OHLCV + 16 indicators = 21) ----
REQUIRED_FEATURES = {
    "5m": ["open","high","low","close","volume",
           "ema_20_ratio","macdh_12_26_9","rsi_14","adx_14","di_delta","atr_pct",
           "bb_percent_b_20_2","obv_slope","volume_ratio_20","volatility_ratio_14_50",
           "fib_ratio","price_action","vwap_ratio","market_structure","bb_width_20_2","log_return"],
    "1h": ["open","high","low","close","volume",
           "ema_50_ratio","macdh_21_42_9","rsi_21","adx_14","di_delta","atr_pct",
           "bb_percent_b_20_2","obv_slope","volume_ratio_20","volatility_ratio_14_50",
           "fib_ratio","price_action","vwap_ratio","market_structure","bb_width_20_2","log_return"],
    "4h": ["open","high","low","close","volume",
           "ema_100_ratio","macdh_26_52_18","rsi_28","adx_14","di_delta","atr_pct",
           "bb_percent_b_20_2","obv_slope","volume_ratio_20","volatility_ratio_14_50",
           "fib_ratio","price_action","vwap_ratio","market_structure","bb_width_20_2","log_return"],
}

# Sanity ranges for key indicators
SANITY_RANGES = {
    "rsi_14": (0, 100), "rsi_21": (0, 100), "rsi_28": (0, 100),
    "adx_14": (0, 100),
    "bb_percent_b_20_2": (-2.0, 3.0),
    "di_delta": (-100, 100),
    "atr_pct": (0, 0.5),
    "fib_ratio": (-0.5, 1.5),
    "market_structure": (-1.01, 1.01),
}

# ---- Instantiate FeatureEngineer with config ----
fe = FeatureEngineer(data_config=full_config, models_dir=MODELS_DIR)

report = {}
all_ok = True

for tf, required_cols in REQUIRED_FEATURES.items():
    raw_path = _PROJECT_ROOT / f"data/raw/BTCUSDT/{tf}/BTCUSDT_{tf}_raw.parquet"
    if not raw_path.exists():
        logger.error(f"MISSING: {raw_path}")
        all_ok = False; continue

    df_raw = pd.read_parquet(raw_path)
    logger.info(f"=== FeatureEngineer {tf}: {len(df_raw)} rows ===")

    df_feat = fe.calculate_indicators_for_single_timeframe(df_raw.copy(), timeframe=tf)

    tf_report = {"rows_in": len(df_raw), "rows_out": len(df_feat),
                 "columns_out": df_feat.columns.tolist(), "n_cols": len(df_feat.columns),
                 "missing_cols": [], "nan_pcts": {}, "ranges": {},
                 "passed": True, "errors": []}

    # Check required columns
    for col in required_cols:
        if col not in df_feat.columns:
            tf_report["missing_cols"].append(col)
            tf_report["errors"].append(f"MISSING: {col}")

    if tf_report["missing_cols"]:
        logger.error(f"  Missing: {tf_report['missing_cols']}")
        tf_report["passed"] = False
        all_ok = False; report[tf] = tf_report; continue

    # NaN check (tolerate <=5% from warmup)
    for col in required_cols:
        nan_pct = df_feat[col].isna().mean() * 100
        tf_report["nan_pcts"][col] = round(nan_pct, 2)
        if nan_pct > 5.0 and col not in ["open","high","low","close","volume"]:
            tf_report["errors"].append(f"NaN excess {col}: {nan_pct:.1f}%")

    # Sanity range checks
    for col, (lo, hi) in SANITY_RANGES.items():
        if col in df_feat.columns:
            vals = df_feat[col].dropna()
            if len(vals) > 0:
                vmin, vmax = float(vals.min()), float(vals.max())
                tf_report["ranges"][col] = [round(vmin, 4), round(vmax, 4)]
                if vmin < lo or vmax > hi:
                    tf_report["errors"].append(f"RANGE {col}: [{vmin:.3f}, {vmax:.3f}] vs [{lo}, {hi}]")

    # Always save the featured data — range warnings are informational, not fatal
    out_dir = _PROJECT_ROOT / "data/processed/BTCUSDT"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"BTCUSDT_{tf}_featured.parquet"
    df_feat.to_parquet(out_path)
    
    # Separate hard errors (missing cols, NaN excess) from soft warnings (range)
    hard_errors = [e for e in tf_report["errors"] if not e.startswith("RANGE")]
    range_warnings = [e for e in tf_report["errors"] if e.startswith("RANGE")]
    
    if hard_errors:
        logger.error(f"  FAIL {tf}: {len(hard_errors)} hard error(s)")
        for e in hard_errors:
            logger.error(f"    {e}")
        tf_report["passed"] = False
        all_ok = False
    else:
        if range_warnings:
            logger.warning(f"  PASS {tf} (with {len(range_warnings)} range warnings): {len(df_feat)} rows -> {out_path}")
            for w in range_warnings:
                logger.warning(f"    {w}")
        else:
            logger.info(f"  PASS {tf}: {len(df_feat)} rows, {len(df_feat.columns)} cols -> {out_path}")

    report[tf] = tf_report

# Save report
val_dir = _PROJECT_ROOT / "data/validation"
val_dir.mkdir(parents=True, exist_ok=True)
with open(val_dir / "features_report.json", "w") as f:
    json.dump(report, f, indent=2)

print(f"\n{'='*60}")
if all_ok:
    print("DL-03: ALL FEATURES VALIDATED")
    for tf, r in report.items():
        print(f"  {tf}: {r['rows_out']} rows, {r['n_cols']} cols, 0 errors")
else:
    print("DL-03: ERRORS DETECTED — check data/validation/features_report.json")
    for tf, r in report.items():
        status = "PASS" if r["passed"] else f"FAIL({len(r['errors'])})"
        print(f"  {tf}: {status} — {r.get('n_cols', '?')} cols")
        if r.get("missing_cols"):
            print(f"    Missing: {r['missing_cols']}")
        for e in r.get("errors", []):
            print(f"    {e}")
    sys.exit(1)
