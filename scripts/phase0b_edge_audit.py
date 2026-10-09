#!/usr/bin/env python3
"""Phase 0b SPOT: LONG-only TRAIN then VAL economics, account costs required.

Uses the corrected plan-conditioned dataset and independent scalar first-touch
TP grid from audit_tp_mfe. OLD short/TEST/maker-discount verdicts are not authority.
This is geometry/plan EV, NOT realized portfolio EV nor a signal-selection proof.
No TEST tuning and no permanent TP_MAX change. A nonpositive LONG-only net EV
on either split is NO-GO. Missing account fee evidence also blocks the 500K.
"""

import numpy as np
import pandas as pd

PARQUET = "data/processed/BTCUSDT_binance/BTCUSDT_5m_featured.parquet"
WICK_MIN = 0.40
LOOKBACK = 10
MAX_HOLD = 288          # 24 h
PHASES = {11, 12}
FEES_TAKER = 0.0040     # 0.40 % RT (stress-test)
FEES_MAKER = 0.0008     # 0.08 % RT (post-only limit, production)
SL_GRID = [0.004, 0.008, 0.012]       # 0.4 % / 0.8 % / 1.2 %
TP_GRID = [1.5, 2.5, 3.5]             # en R
RNG_SEED = 42


def load():
    df = pd.read_parquet(PARQUET, columns=["open", "high", "low", "close"])
    # Do not drop malformed bars silently; the causal dataset integrity gate handles them.
    return df


def detect_signals(df, side="long"):
    """Indices des bougies-signal (sweep + réintégration + mèche, phase 11/12)."""
    from adan_trading_bot.policy.market_contract import load_market_contract
    load_market_contract().require_direction(side.upper())
    o = df["open"].to_numpy(float); h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float);  c = df["close"].to_numpy(float)
    idx = df.index
    phase = idx.minute.to_numpy() // 5 + 1
    prev_high = pd.Series(h).rolling(LOOKBACK).max().shift(1).to_numpy()
    prev_low = pd.Series(l).rolling(LOOKBACK).min().shift(1).to_numpy()
    sig, base = [], []
    for i in range(LOOKBACK, len(df) - MAX_HOLD - 1):
        if phase[i] not in PHASES:
            continue
        rng = h[i] - l[i]
        if rng <= 0:
            continue
        if side == "short":
            wick = h[i] - max(o[i], c[i])
            ok = (np.isfinite(prev_high[i]) and h[i] > prev_high[i]
                  and c[i] < prev_high[i] and wick >= WICK_MIN * rng)
        else:
            wick = min(o[i], c[i]) - l[i]
            ok = (np.isfinite(prev_low[i]) and l[i] < prev_low[i]
                  and c[i] > prev_low[i] and wick >= WICK_MIN * rng)
        (sig if ok else base).append(i)
    return np.array(sig), np.array(base)


def excursions(df, events, side="long"):
    """Niveaux 1-2 : MFE/MAE (% entry) + durées, pour chaque événement."""
    from adan_trading_bot.policy.market_contract import load_market_contract
    load_market_contract().require_direction(side.upper())
    o = df["open"].to_numpy(float); h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float);  c = df["close"].to_numpy(float)
    atr = atr14(df)
    out = []
    for i in events:
        entry = o[i + 1]
        mfe = mae = 0.0
        t_mfe = t_mae = MAX_HOLD
        for j in range(i + 1, i + 1 + MAX_HOLD):
            fav = (entry - l[j]) if side == "short" else (h[j] - entry)
            adv = (h[j] - entry) if side == "short" else (entry - l[j])
            if fav > mfe:
                mfe, t_mfe = fav, j - i
            if adv > mae:
                mae, t_mae = adv, j - i
        out.append((df.index[i], 100 * mfe / entry, 100 * mae / entry,
                    t_mfe, t_mae, atr[i] / entry * 100 if atr[i] > 0 else np.nan))
    return pd.DataFrame(out, columns=["ts", "mfe_pct", "mae_pct",
                                      "t_mfe", "t_mae", "atr_pct"])


def atr14(df):
    tr = np.maximum(df["high"] - df["low"],
                    np.maximum((df["high"] - df["close"].shift()).abs(),
                               (df["low"] - df["close"].shift()).abs()))
    return tr.rolling(14).mean().to_numpy()


def simulate_geometry(df, events, sl_pct, tp_r, side="long"):
    """Niveau 3 : EV brute (R) pour une géométrie SL% × TP(R)."""
    from adan_trading_bot.policy.market_contract import load_market_contract
    load_market_contract().require_direction(side.upper())
    o = df["open"].to_numpy(float); h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float);  c = df["close"].to_numpy(float)
    rs = []
    for i in events:
        entry = o[i + 1]
        risk = entry * sl_pct
        stop = entry + risk if side == "short" else entry - risk
        tp = entry - tp_r * risk if side == "short" else entry + tp_r * risk
        r = None
        for j in range(i + 1, i + 1 + MAX_HOLD):
            hit_sl = h[j] >= stop if side == "short" else l[j] <= stop
            hit_tp = l[j] <= tp if side == "short" else h[j] >= tp
            if hit_sl or hit_tp:
                r = -1.0 if hit_sl else tp_r   # ambigu → pire cas (SL d'abord)
                break
        if r is None:
            exit_p = c[i + MAX_HOLD]
            r = ((entry - exit_p) if side == "short" else (exit_p - entry)) / risk
        rs.append(r)
    return np.array(rs)


def split_mask(ts):
    return {"train": ts < "2022-01-01",
            "val": (ts >= "2022-01-01") & (ts < "2024-01-01"),
}


def main():
    import argparse
    import hashlib
    import json
    import subprocess
    from pathlib import Path
    from adan_trading_bot.offline.audit_tp_mfe import audit
    from adan_trading_bot.policy.market_contract import load_market_contract
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',type=Path,required=True)
    parser.add_argument('--val',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--diagnostic-unverified-fees',action='store_true')
    args = parser.parse_args()
    target = args.report.resolve()
    if not target.is_relative_to(Path('/home/ubuntu/webapp')) or not target.parent.is_dir():
        raise ValueError('Output must remain inside workspace with existing parent')
    market = load_market_contract()
    if not args.diagnostic_unverified_fees:
        market.require_verified_fees()
    train = audit(1000000,12,'train',args.train)
    val = audit(1000000,12,'val',args.val)
    reasons = []
    if not market.fee_verified:
        reasons.append('Account fee tier unavailable; configured costs are DIAGNOSTIC ONLY')
    for name, result in (('TRAIN',train),('VAL',val)):
        if result['baseline_rates']['NET_RETURN'] <= 0:
            reasons.append(name + ' LONG-only mean NET_RETURN <= 0 after costs')
    # No automatic GO: arithmetic passes cannot authorize a signal or portfolio strategy.
    report = {'market':'SPOT','actions':['BUY','SELL_EXIT','HOLD'],'train':train,'val':val,
              'stop_rule':'If LONG-only mean NET_RETURN <= 0 on TRAIN OR VAL after verified account costs: NO-GO, no 500K',
              'blocking_reasons':reasons,'verdict':'NO_GO' if reasons else 'ECONOMY_ONLY_PASS_OTHER_GATES_PENDING',
              'experiment_500k_authorized':False,'tp_max_locked':False,'test_touched':False,
              'git_commit':subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
              'train_manifest_sha256':hashlib.sha256((args.train/'manifest.json').read_bytes()).hexdigest(),
              'val_manifest_sha256':hashlib.sha256((args.val/'manifest.json').read_bytes()).hexdigest(),
              'interpretation':'Overlapping State×Plan alternatives, not executed trades. Positive aggregate EV is not proof of deployable alpha. TP grid and SL buckets identical on TRAIN and VAL.'}
    target.write_text(json.dumps(report,indent=2,default=str)+'\n')
    print(json.dumps({'verdict':report['verdict'],'blocking_reasons':reasons,
          'train':train['baseline_rates'],'val':val['baseline_rates']},indent=2))


if __name__ == '__main__':
    main()
