"""Analytics over REAL artifacts: backtests, trades, OHLCV, metrics matrices.

Sources:
- logs/validation/*.json                  (aggregate backtest metrics per run)
- logs/validation/confidence_scan/bt_*.json  (per-checkpoint backtests)
- logs/validation/forensic/forensic_*.json (forensic per checkpoint)
- logs/paper/**/trades_paper_*.csv         (trade-by-trade)
- data/processed/BTCUSDT/BTCUSDT_<tf>_featured.parquet (OHLCV candles)
"""
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any

from .. import settings

VAL_DIR = settings.REPO_ROOT / "logs" / "validation"
SCAN_DIR = VAL_DIR / "confidence_scan"
FORENSIC_DIR = VAL_DIR / "forensic"
PAPER_DIR = settings.REPO_ROOT / "logs" / "paper"
DATA_DIR = settings.REPO_ROOT / "data" / "processed" / "BTCUSDT"

_CKPT_RE = re.compile(r"(\d+)")


# ----------------------------- Model registry ----------------------------- #
def model_registry() -> list[dict[str, Any]]:
    """Per-checkpoint backtest metrics from confidence_scan/bt_*.json."""
    out: list[dict[str, Any]] = []
    if SCAN_DIR.exists():
        for p in sorted(SCAN_DIR.glob("bt_*.json")):
            stem = p.stem.replace("bt_", "")
            if not stem.isdigit():
                continue
            try:
                d = json.loads(p.read_text())
            except Exception:
                continue
            out.append({
                "checkpoint": int(stem),
                "n_trades": d.get("n_trades") or d.get("env_total_trades"),
                "win_rate": d.get("win_rate"),
                "profit_factor": d.get("profit_factor"),
                "expectancy_pct": d.get("expectancy_pct") or d.get("avg_pnl_pct_per_trade"),
                "total_return_pct": d.get("total_return_pct"),
                "sharpe_like": d.get("sharpe_like"),
                "best_trade_pct": d.get("best_trade_pct"),
                "worst_trade_pct": d.get("worst_trade_pct"),
                "max_consecutive_losses": d.get("max_consecutive_losses"),
                "verdict": d.get("verdict"),
                "source": p.name,
            })
    out.sort(key=lambda x: x["checkpoint"])
    return out


def named_backtests() -> list[dict[str, Any]]:
    """Named validation backtests (logs/validation/backtest_*.json)."""
    out: list[dict[str, Any]] = []
    if VAL_DIR.exists():
        for p in sorted(VAL_DIR.glob("backtest_*.json")):
            try:
                d = json.loads(p.read_text())
            except Exception:
                continue
            out.append({
                "name": p.stem.replace("backtest_", ""),
                "checkpoint": d.get("checkpoint"),
                "split": d.get("split"),
                "n_trades": d.get("n_trades"),
                "win_rate": d.get("win_rate"),
                "profit_factor": d.get("profit_factor"),
                "total_return_pct": d.get("total_return_pct"),
                "sharpe_like": d.get("sharpe_like"),
                "expectancy_pct": d.get("expectancy_pct"),
                "verdict": d.get("verdict"),
            })
    return out


# ----------------------------- Trades & equity ----------------------------- #
def _find_paper_trade_files() -> list[Path]:
    if not PAPER_DIR.exists():
        return []
    files = list(PAPER_DIR.glob("**/trades_paper_*.csv"))
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def _read_paper_csv(f: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with f.open("r", newline="") as fh:
            for r in csv.DictReader(fh):
                rows.append(r)
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for r in rows[:limit]:
        def num(k):
            try:
                return float(r.get(k, "") or "nan")
            except ValueError:
                return None
        out.append({
            "timestamp": r.get("timestamp"),
            "side": r.get("side"),
            "symbol": r.get("symbol"),
            "price": num("price"),
            "size_usd": num("size_usd"),
            "size_asset": num("size_asset"),
            "sl_pct": num("sl_pct"),
            "tp_pct": num("tp_pct"),
            "fee_usd": num("fee_usd"),
            "pnl_usd": num("pnl_usd"),
            "reason": r.get("reason"),
            "source": r.get("source"),
            "order_id": r.get("order_id"),
        })
    return out


# Module-level TTL cache: the paper-file scan is expensive (up to 60 files), and
# several services (performance, validation, markers, equity) call load_trades
# within the same request cycle. Caching keeps the agent snapshot fast.
_LOAD_CACHE: dict[int, tuple[float, dict[str, Any]]] = {}
_LOAD_TTL = 8.0


def load_trades(limit: int = 2000) -> dict[str, Any]:
    """Trade-by-trade from the RICHEST recent paper CSV (TTL-cached).

    Prefers files that actually contain closed trades (non-zero pnl), so the
    metrics matrices are meaningful instead of showing an OPEN-only file.
    """
    import time as _t
    now = _t.time()
    hit = _LOAD_CACHE.get(limit)
    if hit and (now - hit[0]) < _LOAD_TTL:
        return hit[1]

    files = _find_paper_trade_files()
    best: tuple[int, Path, list[dict[str, Any]]] | None = None
    # Scan up to 60 most-recent files; pick the one with most pnl-bearing rows.
    for f in files[:60]:
        trades = _read_paper_csv(f, limit)
        if not trades:
            continue
        scored = sum(1 for t in trades if (t.get("pnl_usd") or 0.0) != 0.0)
        if best is None or scored > best[0]:
            best = (scored, f, trades)
        if scored >= 20:  # good enough, stop early
            break
    if best is None:
        result = {"file": None, "count": 0, "trades": []}
    else:
        _, f, trades = best
        result = {
            "file": str(f.relative_to(settings.REPO_ROOT)),
            "count": len(trades),
            "trades": trades,
        }
    _LOAD_CACHE[limit] = (now, result)
    return result


def equity_curve_from_trades(trades: list[dict[str, Any]], start: float = 1000.0):
    eq = start
    curve = []
    for i, t in enumerate(trades):
        pnl = t.get("pnl_usd") or 0.0
        fee = t.get("fee_usd") or 0.0
        eq += (pnl - fee)
        curve.append({"i": i, "equity": round(eq, 2),
                      "ts": t.get("timestamp"), "reason": t.get("reason")})
    return curve


# ----------------------------- Metrics matrices ---------------------------- #
def _pct_returns(trades: list[dict[str, Any]]) -> list[float]:
    """Per-CLOSE pnl in fraction of notional (approx)."""
    rets = []
    for t in trades:
        pnl = t.get("pnl_usd")
        sz = t.get("size_usd")
        if pnl is not None and sz and sz != 0 and (t.get("reason") not in ("OPEN",)):
            rets.append(pnl / sz)
    return rets


def compute_metrics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    rets = _pct_returns(trades)
    n = len(rets)
    if n == 0:
        return {"n_closed": 0, "note": "no closed trades with pnl in this file"}

    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r < 0]
    mean = sum(rets) / n
    var = sum((r - mean) ** 2 for r in rets) / n
    std = math.sqrt(var) if var > 0 else 0.0
    downside = [r for r in rets if r < 0]
    dstd = math.sqrt(sum(r * r for r in downside) / len(downside)) if downside else 0.0

    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
    win_rate = len(wins) / n
    expectancy = mean

    # equity & drawdown
    eq = 1.0
    peak = 1.0
    max_dd = 0.0
    eq_path = []
    for r in rets:
        eq *= (1 + r)
        peak = max(peak, eq)
        dd = (eq - peak) / peak
        max_dd = min(max_dd, dd)
        eq_path.append(eq)
    total_return = eq - 1.0

    sharpe = (mean / std) * math.sqrt(n) if std > 0 else 0.0
    sortino = (mean / dstd) * math.sqrt(n) if dstd > 0 else 0.0
    calmar = (total_return / abs(max_dd)) if max_dd < 0 else 0.0

    # VaR / CVaR at 95%
    srt = sorted(rets)
    idx = max(0, int(0.05 * n) - 1)
    var95 = srt[idx]
    tail = srt[: idx + 1]
    cvar95 = sum(tail) / len(tail) if tail else var95

    # max consecutive losses
    mcl = cur = 0
    for r in rets:
        if r < 0:
            cur += 1
            mcl = max(mcl, cur)
        else:
            cur = 0

    return {
        "n_closed": n,
        "win_rate": round(win_rate, 4),
        "profit_factor": round(pf, 4) if pf != float("inf") else None,
        "expectancy": round(expectancy, 5),
        "mean_return": round(mean, 5),
        "std_return": round(std, 5),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "calmar": round(calmar, 4),
        "max_drawdown": round(max_dd, 4),
        "total_return": round(total_return, 4),
        "best": round(max(rets), 4),
        "worst": round(min(rets), 4),
        "var95": round(var95, 5),
        "cvar95": round(cvar95, 5),
        "max_consecutive_losses": mcl,
        "n_wins": len(wins),
        "n_losses": len(losses),
    }


def confusion_buy_sell(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """Simple side/outcome breakdown (proxy confusion matrix)."""
    buckets = {"buy_open": 0, "sell_open": 0, "close_win": 0, "close_loss": 0}
    for t in trades:
        side = (t.get("side") or "").upper()
        reason = (t.get("reason") or "").upper()
        pnl = t.get("pnl_usd") or 0.0
        if reason == "OPEN":
            buckets["buy_open" if side == "BUY" else "sell_open"] += 1
        else:
            buckets["close_win" if pnl > 0 else "close_loss"] += 1
    return buckets


# ------------------------------- OHLCV candles ----------------------------- #
def candles(timeframe: str = "5m", limit: int = 500) -> dict[str, Any]:
    tf = timeframe if timeframe in ("5m", "1h", "4h") else "5m"
    path = DATA_DIR / f"BTCUSDT_{tf}_featured.parquet"
    if not path.exists():
        return {"timeframe": tf, "candles": [], "error": "parquet not found"}
    try:
        import pandas as pd
        df = pd.read_parquet(path, columns=["open", "high", "low", "close", "volume"])
        df = df.tail(limit)
        out = []
        for ts, row in df.iterrows():
            try:
                t = int(ts.timestamp())
            except Exception:
                continue
            out.append({
                "time": t,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            })
        return {"timeframe": tf, "count": len(out), "candles": out}
    except Exception as e:
        return {"timeframe": tf, "candles": [], "error": str(e)}
