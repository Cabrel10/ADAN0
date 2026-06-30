"""Reliable Metrics Engine for ADAN Mission Control.

Strict architecture (Bloomberg / QuantConnect style):

    Raw data (CSV / parquet)  ->  Engine (independent recompute)
                              ->  Validator (recompute vs source)
                              ->  Cache (TTL)
                              ->  Dashboard / Agent

Core principles enforced here:
  * Every metric carries provenance: value, computed_at, source, sample window.
  * Metrics are RECOMPUTED from raw rows, never read from an in-memory training
    variable (which can be stale or corrupted).
  * The Validator re-derives the same number a second time via an independent
    code path and flags any mismatch.
  * Results are cached with a TTL so the dashboard reads validated numbers only.
"""
from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import analytics_service as A
from . import telemetry_service as T
from ..settings import TELEMETRY_CSV


# --------------------------------------------------------------------------- #
# provenance helper
# --------------------------------------------------------------------------- #
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def metric(value: Any, source: str, window: int | None = None,
           note: str | None = None) -> dict[str, Any]:
    """Wrap a value with provenance so the UI/agent can trust it."""
    out: dict[str, Any] = {
        "value": value,
        "computed_at": _now_iso(),
        "source": source,
    }
    if window is not None:
        out["window"] = window
    if note:
        out["note"] = note
    return out


# --------------------------------------------------------------------------- #
# tiny TTL cache
# --------------------------------------------------------------------------- #
_CACHE: dict[str, tuple[float, Any]] = {}


def cached(key: str, ttl: float, fn: Callable[[], Any]) -> Any:
    now = time.time()
    hit = _CACHE.get(key)
    if hit and (now - hit[0]) < ttl:
        return hit[1]
    val = fn()
    _CACHE[key] = (now, val)
    return val


# --------------------------------------------------------------------------- #
# Trade-derived performance metrics (recomputed from raw CSV)
# --------------------------------------------------------------------------- #
def _trade_returns(trades: list[dict[str, Any]]) -> list[float]:
    """Per-CLOSE return as pnl / size_usd (excludes OPEN rows)."""
    rets = []
    for t in trades:
        pnl = t.get("pnl_usd")
        sz = t.get("size_usd")
        reason = (t.get("reason") or "").upper()
        if pnl is not None and sz and sz != 0 and reason != "OPEN":
            rets.append(pnl / sz)
    return rets


def performance_block(limit: int = 5000) -> dict[str, Any]:
    """Full performance + risk block recomputed from raw paper CSV."""
    data = A.load_trades(limit=limit)
    trades = data["trades"]
    src = data["file"] or "no-file"
    rets = _trade_returns(trades)
    n = len(rets)
    if n == 0:
        return {
            "source": src,
            "n_closed": metric(0, src, 0, "no closed trades with pnl"),
        }

    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r < 0]
    mean = sum(rets) / n
    std = math.sqrt(sum((r - mean) ** 2 for r in rets) / n)
    downside = [r for r in rets if r < 0]
    dstd = math.sqrt(sum(r * r for r in downside) / len(downside)) if downside else 0.0

    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    pf = (gross_win / gross_loss) if gross_loss > 0 else None
    win_rate = len(wins) / n

    # equity / drawdown / ulcer
    eq = 1.0
    peak = 1.0
    max_dd = 0.0
    dd_sq_sum = 0.0
    eq_curve = []
    for r in rets:
        eq *= (1 + r)
        peak = max(peak, eq)
        dd = (eq - peak) / peak
        max_dd = min(max_dd, dd)
        dd_sq_sum += dd * dd
        eq_curve.append(eq)
    total_return = eq - 1.0
    ulcer = math.sqrt(dd_sq_sum / n)

    sharpe = (mean / std) * math.sqrt(n) if std > 0 else 0.0
    sortino = (mean / dstd) * math.sqrt(n) if dstd > 0 else 0.0
    calmar = (total_return / abs(max_dd)) if max_dd < 0 else 0.0
    mar = calmar  # MAR == CAGR/maxDD; here total_return proxy

    srt = sorted(rets)
    idx = max(0, int(0.05 * n) - 1)
    var95 = srt[idx]
    tail = srt[: idx + 1]
    cvar95 = sum(tail) / len(tail) if tail else var95

    avg_win = (sum(wins) / len(wins)) if wins else 0.0
    avg_loss = (sum(losses) / len(losses)) if losses else 0.0

    # exposure = fraction of rows that are open positions
    n_open = sum(1 for t in trades if (t.get("reason") or "").upper() == "OPEN")
    exposure = n_open / len(trades) if trades else 0.0

    w = n  # sample window = number of closed trades
    return {
        "source": src,
        "n_closed": metric(n, src, w),
        "win_rate": metric(round(win_rate, 4), src, w),
        "profit_factor": metric(round(pf, 4) if pf is not None else None, src, w),
        "expectancy": metric(round(mean, 5), src, w),
        "mean_return": metric(round(mean, 5), src, w),
        "std_return": metric(round(std, 5), src, w),
        "avg_winner": metric(round(avg_win, 5), src, len(wins)),
        "avg_loser": metric(round(avg_loss, 5), src, len(losses)),
        "total_return": metric(round(total_return, 4), src, w),
        "max_drawdown": metric(round(max_dd, 4), src, w),
        "ulcer_index": metric(round(ulcer, 5), src, w),
        "sharpe": metric(round(sharpe, 4), src, w),
        "sortino": metric(round(sortino, 4), src, w),
        "calmar": metric(round(calmar, 4), src, w),
        "mar": metric(round(mar, 4), src, w),
        "var95": metric(round(var95, 5), src, w),
        "cvar95": metric(round(cvar95, 5), src, w),
        "best": metric(round(max(rets), 4), src, w),
        "worst": metric(round(min(rets), 4), src, w),
        "exposure": metric(round(exposure, 4), src, len(trades)),
        "n_wins": metric(len(wins), src, w),
        "n_losses": metric(len(losses), src, w),
    }


# --------------------------------------------------------------------------- #
# RL / training metrics (from telemetry CSV) with provenance
# --------------------------------------------------------------------------- #
def rl_block() -> dict[str, Any]:
    rows = T.read_telemetry(since=0)
    src = str(TELEMETRY_CSV.name)
    if not rows:
        return {"source": src, "note": metric(None, src, 0, "no telemetry yet")}
    last = rows[-1]
    w = int(last.get("timesteps") or 0)
    histo = last.get("a0_histo") or []
    histo_total = sum(histo) if histo else 0
    # concentration: share of the most populated bin (collapse indicator)
    concentration = (max(histo) / histo_total) if histo_total else None
    return {
        "source": src,
        "timesteps": metric(w, src, len(rows)),
        "a0_mean": metric(last.get("a0_mean"), src, w),
        "a0_std": metric(last.get("a0_std"), src, w),
        "policy_entropy": metric(last.get("policy_entropy"), src, w),
        "illegal_ratio": metric(last.get("illegal_ratio"), src, w),
        "req_BUY_pct": metric(last.get("req_BUY_pct"), src, w),
        "req_SELL_pct": metric(last.get("req_SELL_pct"), src, w),
        "req_HOLD_pct": metric(last.get("req_HOLD_pct"), src, w),
        "steps_flat_pct": metric(last.get("steps_flat_pct"), src, w),
        "steps_open_pct": metric(last.get("steps_open_pct"), src, w),
        "histo_concentration": metric(
            round(concentration, 4) if concentration is not None else None, src, w,
            "share of the most populated action bin (1.0 = full collapse)"),
    }


# --------------------------------------------------------------------------- #
# Validator: recompute key numbers via an INDEPENDENT path and compare
# --------------------------------------------------------------------------- #
def _independent_recount(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """Second, deliberately different implementation used only for validation."""
    closed = 0
    wins = 0
    gw = 0.0
    gl = 0.0
    sum_ret = 0.0
    for t in trades:
        reason = (t.get("reason") or "").upper()
        if reason == "OPEN":
            continue
        pnl = t.get("pnl_usd")
        sz = t.get("size_usd")
        if pnl is None or not sz:
            continue
        r = pnl / sz
        closed += 1
        sum_ret += r
        if r > 0:
            wins += 1
            gw += r
        elif r < 0:
            gl += -r
    return {
        "n_closed": closed,
        "win_rate": round(wins / closed, 4) if closed else 0.0,
        "profit_factor": round(gw / gl, 4) if gl > 0 else None,
        "expectancy": round(sum_ret / closed, 5) if closed else 0.0,
    }


def validate(limit: int = 5000) -> dict[str, Any]:
    """Compare engine output vs an independent recompute; flag mismatches."""
    data = A.load_trades(limit=limit)
    trades = data["trades"]
    src = data["file"] or "no-file"

    engine = A.compute_metrics(trades)          # path 1 (existing)
    indep = _independent_recount(trades)         # path 2 (independent)

    def cmp(name: str, tol: float = 1e-6) -> dict[str, Any]:
        a = engine.get(name)
        b = indep.get(name)
        if a is None and b is None:
            match = True
        elif a is None or b is None:
            match = (a == b)
        else:
            match = abs(float(a) - float(b)) <= tol
        return {"dashboard": a, "recomputed": b, "match": bool(match)}

    checks = {
        "n_closed": cmp("n_closed"),
        "win_rate": cmp("win_rate", tol=1e-4),
        "profit_factor": cmp("profit_factor", tol=1e-3),
        "expectancy": cmp("expectancy", tol=1e-4),
    }
    all_ok = all(c["match"] for c in checks.values())
    return {
        "source": src,
        "computed_at": _now_iso(),
        "all_match": all_ok,
        "checks": checks,
    }


# --------------------------------------------------------------------------- #
# Trade-marker validation (candle window + price proximity)
# --------------------------------------------------------------------------- #
def validate_markers(timeframe: str = "5m", limit_trades: int = 5000,
                     limit_candles: int = 800) -> dict[str, Any]:
    """Validate each trade marker against the candle series.

    Because paper-trade timestamps are *simulation step counters* (not real
    epochs), we cannot do candle_open<=t<candle_close. Instead we validate that
    each trade PRICE falls within the [low, high] range of at least one candle
    in the visible window (i.e. the price is real for this market), and report
    how well the nearest-by-close mapping fits.
    """
    cdata = A.candles(timeframe=timeframe, limit=limit_candles)
    candles = cdata.get("candles", [])
    tdata = A.load_trades(limit=limit_trades)
    trades = tdata["trades"]
    if not candles or not trades:
        return {"checked": 0, "valid": 0, "invalid": 0, "markers": [],
                "note": "no candles or no trades"}

    lo = min(c["low"] for c in candles)
    hi = max(c["high"] for c in candles)
    # pre-extract arrays once (avoid repeated dict lookups in the inner loop)
    closes = [c["close"] for c in candles]
    times = [c["time"] for c in candles]
    valid = 0
    invalid = 0
    sample = []
    for i, t in enumerate(trades):
        price = t.get("price")
        in_range = price is not None and lo <= price <= hi
        # nearest candle by close (the mapping the chart uses)
        nearest = None
        if price is not None:
            best_d = float("inf")
            for j, cl in enumerate(closes):
                d = cl - price
                if d < 0:
                    d = -d
                if d < best_d:
                    best_d = d
                    nearest = times[j]
        if in_range:
            valid += 1
        else:
            invalid += 1
        if i < 60:
            sample.append({
                "idx": i,
                "side": t.get("side"),
                "reason": t.get("reason"),
                "price": price,
                "in_market_range": bool(in_range),
                "mapped_candle_time": nearest,
                "status": "VALID" if in_range else "OUT_OF_RANGE",
            })
    return {
        "timeframe": timeframe,
        "market_range": {"low": round(lo, 2), "high": round(hi, 2)},
        "checked": valid + invalid,
        "valid": valid,
        "invalid": invalid,
        "markers": sample,
    }


# --------------------------------------------------------------------------- #
# Equity & drawdown curves (for the dashboard)
# --------------------------------------------------------------------------- #
def equity_and_drawdown(limit: int = 5000) -> dict[str, Any]:
    data = A.load_trades(limit=limit)
    trades = data["trades"]
    src = data["file"] or "no-file"
    eq = 1000.0
    peak = eq
    pts = []
    i = 0
    for t in trades:
        pnl = t.get("pnl_usd") or 0.0
        fee = t.get("fee_usd") or 0.0
        eq += (pnl - fee)
        peak = max(peak, eq)
        dd = (eq - peak) / peak if peak else 0.0
        pts.append({"i": i, "equity": round(eq, 2), "drawdown": round(dd, 5)})
        i += 1
    return {"source": src, "start": 1000.0, "points": pts}
