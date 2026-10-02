#!/usr/bin/env python3
"""
Non-neural benchmarks (ZERO training) — the reference bar ADAN must clear.

Strategies (long-only, one position at a time, next-bar-open execution):
  1. buy_hold      : enter bar 0, never exit (equity = price curve).
  2. random_entry  : each flat bar enters LONG with prob p; exit by TP/SL/horizon.
  3. random_exit   : always in market when flat; random hold length then re-enter.
  4. momentum      : enter when ema_20_ratio>1 & macdh>0 & rsi in (50,70).
  5. mean_reversion: enter when rsi<35 & bb_percent_b<0.1 (oversold).
  6. cluster       : enter when rsi<40 & adx>25 (the audit's oversold+ADX regime).

Trade contract for entry strategies (2..6): TP = TP_ATR*atr_pct, SL = SL_ATR*atr_pct,
max hold = H bars. Fee = FEE per side (round-trip 2*FEE). All on the SAME bars so
PF/Sharpe/DD are directly comparable.

Metrics: n_trades, win_rate, PF (gross win / gross loss), avg R per trade,
total return, Sharpe (per-trade, annualized approx), max drawdown on equity.

Usage: benchmark_nonneural.py <featured_5m_parquet> <label> [out_json]
Env: BM_TP_ATR (3.0) BM_SL_ATR (2.0) BM_HORIZON (40) BM_FEE (0.0006) BM_SEED (0)
"""
import os, sys, json
import numpy as np
import pandas as pd

parq = sys.argv[1]
LABEL = sys.argv[2] if len(sys.argv) > 2 else "ASSET"
OUT = sys.argv[3] if len(sys.argv) > 3 else f"/tmp/bench_{LABEL}.json"

TP_ATR = float(os.environ.get("BM_TP_ATR", "3.0"))
SL_ATR = float(os.environ.get("BM_SL_ATR", "2.0"))
H = int(os.environ.get("BM_HORIZON", "40"))
FEE = float(os.environ.get("BM_FEE", "0.0006"))   # 6 bps per side
SEED = int(os.environ.get("BM_SEED", "0"))
BARS_PER_YEAR = 105_120  # 5m bars/year (365*24*12)

rng = np.random.default_rng(SEED)
df = pd.read_parquet(parq).reset_index()
N = len(df)
openp = df["open"].to_numpy(np.float64)
high = df["high"].to_numpy(np.float64)
low = df["low"].to_numpy(np.float64)
close = df["close"].to_numpy(np.float64)
atr = df["atr_pct"].to_numpy(np.float64) if "atr_pct" in df else np.full(N, 0.0015)


def sig(colexpr):
    return colexpr.to_numpy() if hasattr(colexpr, "to_numpy") else colexpr


def col(name, default=np.nan):
    return df[name].to_numpy(np.float64) if name in df.columns else np.full(N, default)


rsi = col("rsi_14"); adx = col("adx_14"); ema_r = col("ema_20_ratio")
macdh = col("macdh_12_26_9"); bbp = col("bb_percent_b_20_2")


def simulate_entries(entry_mask):
    """Long trades: on each allowed flat bar where entry_mask True, enter at NEXT
    bar open; exit at first of TP / SL / horizon (intrabar TP/SL via high/low)."""
    trades = []
    i = 0
    while i < N - 1:
        if not entry_mask[i]:
            i += 1; continue
        entry_i = i + 1
        if entry_i >= N:
            break
        ep = openp[entry_i]
        a = atr[entry_i] if np.isfinite(atr[entry_i]) and atr[entry_i] > 0 else 0.0015
        tp = ep * (1 + TP_ATR * a)
        sl = ep * (1 - SL_ATR * a)
        exit_i = min(entry_i + H, N - 1)
        outcome = None
        for j in range(entry_i, min(entry_i + H, N)):
            if low[j] <= sl:
                exit_i = j; outcome = "SL"; xp = sl; break
            if high[j] >= tp:
                exit_i = j; outcome = "TP"; xp = tp; break
        if outcome is None:
            xp = close[exit_i]; outcome = "HORIZON"
        gross = xp / ep - 1.0
        net = gross - 2 * FEE
        trades.append((entry_i, exit_i, net, outcome))
        i = exit_i + 1  # flat until this trade closes
    return trades


def metrics_from_trades(trades):
    if not trades:
        return {"n_trades": 0}
    rets = np.array([t[2] for t in trades])
    wins = rets[rets > 0]; losses = rets[rets < 0]
    gross_win = wins.sum() if len(wins) else 0.0
    gross_loss = -losses.sum() if len(losses) else 0.0
    pf = (gross_win / gross_loss) if gross_loss > 0 else (np.inf if gross_win > 0 else 0.0)
    # compounded equity
    eq = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(eq)
    dd = (eq / peak - 1.0).min()
    total_ret = eq[-1] - 1.0
    # per-trade Sharpe, annualized by avg trades/year
    dur = np.mean([t[1] - t[0] for t in trades])
    trades_per_year = BARS_PER_YEAR / max(dur, 1)
    sharpe = (rets.mean() / rets.std() * np.sqrt(trades_per_year)) if rets.std() > 0 else 0.0
    n_tp = sum(1 for t in trades if t[3] == "TP")
    n_sl = sum(1 for t in trades if t[3] == "SL")
    n_h = sum(1 for t in trades if t[3] == "HORIZON")
    return {
        "n_trades": len(trades),
        "win_rate_pct": float(len(wins) / len(trades) * 100),
        "profit_factor": float(round(pf, 3)) if np.isfinite(pf) else "inf",
        "avg_net_ret_pct": float(rets.mean() * 100),
        "total_return_pct": float(total_ret * 100),
        "sharpe_annual": float(round(sharpe, 3)),
        "max_drawdown_pct": float(dd * 100),
        "exit_TP": n_tp, "exit_SL": n_sl, "exit_HORIZON": n_h,
        "avg_hold_bars": float(round(dur, 1)),
    }


results = {}

# 1. buy & hold
bh_ret = close[-1] / openp[0] - 1.0 - 2 * FEE
eq = close / openp[0]
peak = np.maximum.accumulate(eq)
results["buy_hold"] = {
    "total_return_pct": float(bh_ret * 100),
    "max_drawdown_pct": float((eq / peak - 1.0).min() * 100),
    "n_trades": 1,
}

# 2. random entry (p tuned so trade count is comparable to signal strategies)
p_entry = 0.02
rmask = rng.random(N) < p_entry
results["random_entry"] = metrics_from_trades(simulate_entries(rmask))

# 3. random exit: enter every flat bar (always-in), random horizon via mask of all True
# Model as "enter whenever flat" -> same engine, entry every bar
results["random_exit"] = metrics_from_trades(simulate_entries(np.ones(N, bool)))

# 4. momentum
mom = (ema_r > 1.0) & (macdh > 0) & (rsi > 50) & (rsi < 70)
results["momentum"] = metrics_from_trades(simulate_entries(np.nan_to_num(mom).astype(bool)))

# 5. mean reversion
mr = (rsi < 35) & (bbp < 0.1)
results["mean_reversion"] = metrics_from_trades(simulate_entries(np.nan_to_num(mr).astype(bool)))

# 6. cluster (audit signature: oversold + strong trend)
cl = (rsi < 40) & (adx > 25)
results["cluster_oversold_adx"] = metrics_from_trades(simulate_entries(np.nan_to_num(cl).astype(bool)))

summary = {
    "label": LABEL, "n_bars": N,
    "range": [str(df["timestamp"].iloc[0]), str(df["timestamp"].iloc[-1])],
    "contract": {"tp_atr": TP_ATR, "sl_atr": SL_ATR, "horizon": H, "fee_per_side": FEE},
    "results": results,
}

print("=" * 88)
print(f"BENCHMARKS NON-NEURONAUX — {LABEL}  n={N}  (TP={TP_ATR}xATR SL={SL_ATR}xATR H={H} fee={FEE})")
print("=" * 88)
print(f"{'strategy':22} {'trades':>7} {'win%':>6} {'PF':>7} {'avgR%':>7} {'totRet%':>9} {'Sharpe':>7} {'maxDD%':>8}")
for name, m in results.items():
    if name == "buy_hold":
        print(f"{name:22} {1:>7} {'-':>6} {'-':>7} {'-':>7} {m['total_return_pct']:>9.1f} {'-':>7} {m['max_drawdown_pct']:>8.1f}")
        continue
    if m.get("n_trades", 0) == 0:
        print(f"{name:22} {0:>7}  (no trades)"); continue
    print(f"{name:22} {m['n_trades']:>7} {m['win_rate_pct']:>6.1f} "
          f"{str(m['profit_factor']):>7} {m['avg_net_ret_pct']:>7.3f} "
          f"{m['total_return_pct']:>9.1f} {m['sharpe_annual']:>7.2f} {m['max_drawdown_pct']:>8.1f}")

json.dump(summary, open(OUT, "w"), indent=2, default=str)
print(f"\nsaved {OUT}")
