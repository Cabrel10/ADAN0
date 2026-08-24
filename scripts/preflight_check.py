#!/usr/bin/env python3
"""
PRE-FLIGHT INVARIANT CONTROLLER — the single gate before any smoke/training run
of the frozen 5y BTC/DOGE controlled experiment.

It verifies, automatically and non-interactively, that the frozen contract holds
BEFORE the two independent brains are launched. It changes NOTHING; it only reads
and asserts. Output: a GO / NO-GO verdict + JSON report.

Checks (per asset):
  1. dataset = *_binance and present for train/val/test x 5m/1h/4h
  2. 21 feature columns, exact expected set, same for BTC & DOGE
  3. split 70/15/15 (chronological, TEST = final unseen segment: max(train_ts) < min(val_ts) < min(test_ts))
  4. scaler TRAIN-only enforced (ADAN_FORCE_FIT_SCALERS guard present in state_builder)
  5. reward / architecture / action space / DBE / Future Arena source hashes match a frozen snapshot
  6. FREE_SLTP state known + TP/SL mapping band = exactly the frozen commit's numbers
  7. no NaN/Inf in the loaded feature columns

Usage:
  PYTHONPATH=src preflight_check.py --asset BTCUSDT_binance [--freeze] [--out report.json]

--freeze : (re)write the frozen source-hash snapshot from the CURRENT tree. Run ONCE
           on the commit you declare frozen; afterwards the check compares against it.
"""
import os, sys, json, argparse, hashlib, re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# ── frozen expectations (ground-truth verified 2026-08-24) ──────────────────
# 21 features per timeframe. The trend/momentum triplet is TF-specific
# (short window on 5m, longer on 1h/4h); the other 18 are identical across TFs.
# BTC and DOGE MUST share the exact same per-TF schema (cross-asset invariant).
_COMMON_TAIL = [
    "adx_14", "di_delta", "atr_pct", "bb_percent_b_20_2", "obv_slope",
    "volume_ratio_20", "volatility_ratio_14_50", "fib_ratio", "price_action",
    "vwap_ratio", "market_structure", "bb_width_20_2", "log_return",
]
_OHLCV = ["open", "high", "low", "close", "volume"]
EXPECTED_COLS_BY_TF = {
    "5m": _OHLCV + ["ema_20_ratio",  "macdh_12_26_9", "rsi_14"] + _COMMON_TAIL,
    "1h": _OHLCV + ["ema_50_ratio",  "macdh_21_42_9", "rsi_21"] + _COMMON_TAIL,
    "4h": _OHLCV + ["ema_100_ratio", "macdh_26_52_18", "rsi_28"] + _COMMON_TAIL,
}
N_FEATURES = 21
SPLITS = ["train", "val", "test"]
TFS = ["5m", "1h", "4h"]
SPLIT_TARGET = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_TOL = 0.01  # +/- 1 percentage point

# Frozen FREE_SLTP mapping band (exact numbers from the frozen commit).
FROZEN_FREE_SLTP = {
    "sl_lo": 0.003, "sl_hi": 0.060,
    "tp_lo_base": 0.003, "tp_hi": 0.120,
    "tp_lo_floor_expr": "max(2*commission,0.005)*1.2",
    "rr_floor_skipped_when_free": True,
    "atr_floor_skipped_when_free": True,
    "config_tp_bypassed": True,  # config take_profit_pct: 0.04 is NOT applied under FREE_SLTP=1
}

# Source files whose content defines the frozen contract. A change to any of these
# (reward, architecture, action space, DBE, Future Arena, SL/TP mapping) flips NO-GO.
FROZEN_SOURCES = {
    "reward":        "src/adan_trading_bot/environment/reward_calculator.py",
    "reward_shaper": "src/adan_trading_bot/environment/reward_shaper.py",
    "env":           "src/adan_trading_bot/environment/multi_asset_chunked_env.py",
    "dbe":           "src/adan_trading_bot/environment/dynamic_behavior_engine.py",
    "action_space":  "src/adan_trading_bot/environment/action_routing.py",
    "arena":         "src/adan_trading_bot/future_arena",  # package (dir)
    "model":         "src/adan_trading_bot/agent/cnn_ppo_model.py",
    "feature_ext":   "src/adan_trading_bot/agent/feature_extractors.py",
    "state_builder": "src/adan_trading_bot/data_processing/state_builder.py",
}
SNAP_PATH = ROOT / "config" / "frozen_v29_sources.json"


def sha256_file(p: Path) -> str:
    """SHA256 of a file, or of a directory's *.py contents (sorted, path-tagged)."""
    if not p.exists():
        return "MISSING"
    h = hashlib.sha256()
    if p.is_dir():
        for f in sorted(p.rglob("*.py")):
            if "__pycache__" in f.parts:
                continue
            h.update(f.relative_to(p).as_posix().encode())
            h.update(f.read_bytes())
    else:
        h.update(p.read_bytes())
    return h.hexdigest()


def check_dataset_present(asset):
    res = {"ok": True, "detail": {}}
    for s in SPLITS:
        for tf in TFS:
            p = ROOT / "data" / "processed" / "indicators" / s / asset / f"{tf}.parquet"
            ex = p.exists()
            res["detail"][f"{s}/{tf}"] = ex
            if not ex:
                res["ok"] = False
    res["ok"] = res["ok"] and asset.endswith("_binance")
    res["is_binance"] = asset.endswith("_binance")
    return res


def check_features_and_nan(asset):
    res = {"ok": True, "detail": {}}
    for tf in TFS:
        expected = EXPECTED_COLS_BY_TF[tf]
        p = ROOT / "data" / "processed" / "indicators" / "train" / asset / f"{tf}.parquet"
        if not p.exists():
            res["ok"] = False
            res["detail"][tf] = "MISSING"
            continue
        df = pd.read_parquet(p)
        cols = list(df.columns)
        ncol_ok = len(cols) == N_FEATURES
        set_ok = cols == expected
        nan_ct = int(df[expected].isna().sum().sum()) if set_ok else -1
        inf_ct = int(np.isinf(df[expected].to_numpy(np.float64)).sum()) if set_ok else -1
        tf_ok = ncol_ok and set_ok and nan_ct == 0 and inf_ct == 0
        res["detail"][tf] = {"ncols": len(cols), "cols_match": set_ok,
                             "nan": nan_ct, "inf": inf_ct, "ok": tf_ok}
        if not tf_ok:
            res["ok"] = False
    return res


def check_split(asset):
    res = {"ok": True, "detail": {}}
    lens = {}
    ts = {}
    for s in SPLITS:
        p = ROOT / "data" / "processed" / "indicators" / s / asset / "5m.parquet"
        if not p.exists():
            res["ok"] = False
            res["detail"][s] = "MISSING"
            return res
        df = pd.read_parquet(p)
        lens[s] = len(df)
        idx = df.index
        ts[s] = (str(idx.min()), str(idx.max()))
    tot = sum(lens.values())
    for s in SPLITS:
        frac = lens[s] / tot
        ok = abs(frac - SPLIT_TARGET[s]) <= SPLIT_TOL
        res["detail"][s] = {"rows": lens[s], "frac": round(frac, 4),
                            "target": SPLIT_TARGET[s], "ok": ok,
                            "ts_range": ts[s]}
        if not ok:
            res["ok"] = False
    # chronology: max(train) <= min(val) <= min(test); TEST = final unseen
    try:
        chrono = (ts["train"][1] <= ts["val"][0]) and (ts["val"][1] <= ts["test"][0])
    except Exception:
        chrono = False
    res["chronological_no_leak"] = chrono
    res["ok"] = res["ok"] and chrono
    return res


def check_scaler_guard():
    """Verify (via AST) that _load_training_scalers contains an early guard:
    `if ADAN_FORCE_FIT_SCALERS: ... return` positioned BEFORE the first line that
    references 'prod_scalers' as executable code (the stale-scaler load block)."""
    import ast
    p = ROOT / FROZEN_SOURCES["state_builder"]
    txt = p.read_text() if p.exists() else ""
    has_guard = "ADAN_FORCE_FIT_SCALERS" in txt
    guard_before_load = False
    try:
        tree = ast.parse(txt)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_load_training_scalers":
                guard_line = None
                # find the If whose test references ADAN_FORCE_FIT_SCALERS and body returns
                for stmt in ast.walk(node):
                    if isinstance(stmt, ast.If):
                        src = ast.get_source_segment(txt, stmt.test) or ""
                        if "ADAN_FORCE_FIT_SCALERS" in src and any(
                            isinstance(b, ast.Return) for b in stmt.body
                        ):
                            guard_line = stmt.lineno
                            break
                # find first executable use of prod_scalers (Path("prod_scalers"))
                load_line = None
                for stmt in ast.walk(node):
                    if isinstance(stmt, ast.Call):
                        seg = ast.get_source_segment(txt, stmt) or ""
                        if "prod_scalers" in seg:
                            load_line = stmt.lineno
                            break
                if guard_line is not None and (load_line is None or guard_line < load_line):
                    guard_before_load = True
                break
    except Exception:
        guard_before_load = False
    return {"ok": has_guard and guard_before_load,
            "guard_present": has_guard,
            "guard_before_prod_load": guard_before_load,
            "note": "env must run with ADAN_FORCE_FIT_SCALERS=1 for TRAIN-only fit"}


def check_free_sltp_mapping():
    p = ROOT / FROZEN_SOURCES["env"]
    txt = p.read_text() if p.exists() else ""
    # Assert the exact frozen band literals are present in the FREE_SLTP branch.
    checks = {
        "sl_band_0.003_0.060": bool(re.search(r"sl_lo,\s*sl_hi\s*=\s*0\.003,\s*0\.060", txt)),
        "tp_band_0.003_0.120": bool(re.search(r"tp_lo,\s*tp_hi\s*=\s*0\.003,\s*0\.120", txt)),
        "roundtrip_floor_1.2x": "_round_trip * 1.2" in txt,
        "rr_floor_skipped_free": "not _free_sltp and tp_pct < sl_pct * 1.5" in txt,
        "atr_floor_skipped_free": 'not _free_sltp and _prof == "scalper"' in txt,
        "free_sltp_gate": 'os.environ.get("ADAN_FREE_SLTP", "0") == "1"' in txt,
    }
    ok = all(checks.values())
    return {"ok": ok, "detail": checks, "frozen_band": FROZEN_FREE_SLTP}


def check_frozen_sources(freeze=False):
    cur = {name: sha256_file(ROOT / rel) for name, rel in FROZEN_SOURCES.items()}
    if freeze:
        SNAP_PATH.parent.mkdir(parents=True, exist_ok=True)
        SNAP_PATH.write_text(json.dumps({"sources": FROZEN_SOURCES, "sha256": cur}, indent=2))
        return {"ok": True, "frozen_now": True, "snapshot": str(SNAP_PATH), "sha256": cur}
    if not SNAP_PATH.exists():
        return {"ok": False, "reason": "no frozen snapshot yet — run with --freeze on the frozen commit",
                "current_sha256": cur}
    snap = json.loads(SNAP_PATH.read_text())
    ref = snap.get("sha256", {})
    diffs = {n: {"frozen": ref.get(n, "?"), "current": cur[n]}
             for n in FROZEN_SOURCES if ref.get(n) != cur[n]}
    return {"ok": len(diffs) == 0, "changed": list(diffs.keys()), "diffs": diffs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="BTCUSDT_binance")
    ap.add_argument("--freeze", action="store_true",
                    help="(re)write the frozen source-hash snapshot from current tree")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    report = {"asset": args.asset, "checks": {}}
    report["checks"]["dataset_present"] = check_dataset_present(args.asset)
    report["checks"]["features_nan"] = check_features_and_nan(args.asset)
    report["checks"]["split_70_15_15"] = check_split(args.asset)
    report["checks"]["scaler_train_only"] = check_scaler_guard()
    report["checks"]["free_sltp_mapping"] = check_free_sltp_mapping()
    report["checks"]["frozen_sources"] = check_frozen_sources(freeze=args.freeze)

    all_ok = all(c.get("ok", False) for c in report["checks"].values())
    report["verdict"] = "GO" if all_ok else "NO-GO"

    print("=" * 78)
    print(f"PRE-FLIGHT — {args.asset}")
    print("=" * 78)
    for name, c in report["checks"].items():
        mark = "PASS" if c.get("ok") else "FAIL"
        print(f"  [{mark}] {name}")
        if not c.get("ok"):
            print(f"         -> {json.dumps({k: v for k, v in c.items() if k != 'detail'}, default=str)[:400]}")
    print("-" * 78)
    print(f"VERDICT: {report['verdict']}")

    out = args.out or f"/tmp/preflight_{args.asset}.json"
    Path(out).write_text(json.dumps(report, indent=2, default=str))
    print(f"saved {out}")
    sys.exit(0 if all_ok else 2)


if __name__ == "__main__":
    main()
