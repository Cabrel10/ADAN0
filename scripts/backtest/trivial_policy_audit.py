"""ÉTAPE 1 — Trivial Policy Audit (The Oracle Test).

Runs 4 non-learning policies on the SAME env/data used for training/backtest and
measures the *terminal* economics of each. The question is purely about the MDP:

    Does the environment reward permanent long exposure?

If AlwaysLong dominates Random and AlwaysFlat on risk-adjusted terminal metrics,
the MDP is biased (risk externalised by the SL/TP oracle + future-guided reward)
and PPO's collapse to always-BUY is simply PPO discovering that bias.

Policies (a0 is the direction axis; dims 1-4 are Size/TF/SL/TP left neutral):
    AlwaysLong  : a0 = +1.0    (FLAT -> BUY, LONG -> HOLD; never voluntarily exits;
                                exits happen only via SL/TP oracle == the test)
    AlwaysFlat  : a0 = -1.0    (FLAT -> HOLD forever; if ever long, closes)
    AlwaysShort : a0 = -1.0 but we FORCE-open then rely on SELL route — in SPOT
                  shorting is impossible, so this is reported as ~= AlwaysFlat.
                  Kept for completeness / symmetry of the protocol.
    Random      : a0 ~ U(-1, 1) i.i.d. each step.

Metrics (computed from the portfolio's realised closed_positions == ground truth):
    n_trades, EV (mean PnL/trade, USDT), EV_pct (mean return/trade),
    WinRate, ProfitFactor, Sharpe (per-trade), MaxDD (on equity curve),
    terminal_return_pct, mean_hold_bars.

Usage:
    python scripts/backtest/trivial_policy_audit.py [--steps 6000] [--split val]
                                                    [--seed 42] [--out PATH]
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ.setdefault("ADAN_TRAINING_SILENT", "1")
os.environ.setdefault("ADAN_RICH_STEP_EVERY", "999999")
logging.disable(logging.WARNING)


def _build_env(split: str):
    from adan_trading_bot.common.config_loader import ConfigLoader
    from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader
    from adan_trading_bot.environment.multi_asset_chunked_env import (
        MultiAssetChunkedEnv,
    )

    cfg = ConfigLoader.load_config(str(REPO_ROOT / "config" / "config.yaml"))
    cfg.setdefault("environment", {})["rich_display_interval"] = 999999

    worker_idx = 0
    wc = copy.deepcopy(cfg.get("workers", {}).get("w1", {}))
    wc.update({
        "worker_id": worker_idx,
        "data_split_override": split,
        "timeframes": ["5m", "1h", "4h"],
        "assets": ["BTCUSDT"],
    })
    data = ChunkedDataLoader(config=cfg, worker_config=wc, worker_id=worker_idx).load_chunk(0)
    env = MultiAssetChunkedEnv(
        data=data, config=cfg, worker_config=wc, worker_id=worker_idx, live_mode=False
    )
    return env, cfg


def _action_dim(env) -> int:
    try:
        return int(np.prod(env.action_space.shape))
    except Exception:
        return 5


def _make_action(kind: str, dim: int, rng: np.random.Generator) -> np.ndarray:
    """Build a full action vector. a0 = direction; dims 1-4 neutral (0.0)."""
    a = np.zeros(dim, dtype=np.float32)
    if kind == "AlwaysLong":
        a[0] = 1.0
    elif kind == "AlwaysFlat":
        a[0] = -1.0
    elif kind == "AlwaysShort":
        a[0] = -1.0  # SPOT: no short; behaves ~ AlwaysFlat
    elif kind == "Random":
        a[0] = float(rng.uniform(-1.0, 1.0))
    else:
        raise ValueError(kind)
    # dims 1..4 (Size/TF/SL/TP): neutral mid so the oracle uses its defaults
    return a


def _extract_closed_trades(env):
    """Ground-truth per-trade PnL list from the portfolio manager."""
    pm = getattr(env, "portfolio_manager", None)
    if pm is None:
        return []
    try:
        metrics = pm.get_metrics() if hasattr(pm, "get_metrics") else {}
        closed = metrics.get("closed_positions") or []
    except Exception:
        closed = []
    trades = []
    for t in closed:
        pnl = float(t.get("pnl", t.get("realized_pnl", 0.0)))
        entry = float(t.get("entry_price", t.get("entry", 0.0)) or 0.0)
        exit_ = float(t.get("exit_price", t.get("exit", 0.0)) or 0.0)
        size = float(t.get("size", t.get("quantity", 0.0)) or 0.0)
        hold = t.get("hold_bars", t.get("duration", t.get("bars_held", None)))
        reason = t.get("close_reason", t.get("reason", t.get("exit_reason", "")))
        ret_pct = ((exit_ - entry) / entry) if entry > 0 else 0.0
        trades.append({
            "pnl": pnl, "entry": entry, "exit": exit_, "size": size,
            "ret_pct": ret_pct,
            "hold_bars": (float(hold) if hold is not None else np.nan),
            "reason": str(reason),
        })
    return trades


def _metrics_from_trades(trades, terminal_pv, initial_pv):
    if not trades:
        return {
            "n_trades": 0, "EV_usdt": 0.0, "EV_pct": 0.0, "win_rate": 0.0,
            "profit_factor": 0.0, "sharpe_per_trade": 0.0, "max_dd_pct": 0.0,
            "terminal_return_pct": (terminal_pv - initial_pv) / initial_pv * 100.0,
            "mean_hold_bars": 0.0, "gross_win": 0.0, "gross_loss": 0.0,
            "reason_hist": {},
        }
    pnls = np.array([t["pnl"] for t in trades], dtype=float)
    rets = np.array([t["ret_pct"] for t in trades], dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())
    pf = (gross_win / gross_loss) if gross_loss > 1e-12 else (np.inf if gross_win > 0 else 0.0)
    sharpe = float(pnls.mean() / (pnls.std() + 1e-12)) if len(pnls) > 1 else 0.0
    # equity curve max drawdown (cumulative realised PnL)
    eq = initial_pv + np.cumsum(pnls)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd = float(dd.min() * 100.0)
    holds = np.array([t["hold_bars"] for t in trades if not np.isnan(t["hold_bars"])])
    reason_hist = {}
    for t in trades:
        reason_hist[t["reason"]] = reason_hist.get(t["reason"], 0) + 1
    return {
        "n_trades": len(trades),
        "EV_usdt": float(pnls.mean()),
        "EV_pct": float(rets.mean() * 100.0),
        "win_rate": float((pnls > 0).mean()),
        "profit_factor": (float(pf) if np.isfinite(pf) else 999.0),
        "sharpe_per_trade": sharpe,
        "max_dd_pct": max_dd,
        "terminal_return_pct": (terminal_pv - initial_pv) / initial_pv * 100.0,
        "mean_hold_bars": (float(holds.mean()) if len(holds) else np.nan),
        "gross_win": gross_win, "gross_loss": gross_loss,
        "reason_hist": reason_hist,
    }


def run_policy(kind: str, steps: int, split: str, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    env, cfg = _build_env(split)
    dim = _action_dim(env)
    obs = env.reset(seed=seed)
    if isinstance(obs, tuple):
        obs = obs[0]
    initial_pv = 20.50
    last_pv = initial_pv
    done = False
    n_actual = 0
    for s in range(steps):
        a = _make_action(kind, dim, rng)
        out = env.step(a)
        if len(out) == 5:  # gymnasium
            obs, r, terminated, truncated, info = out
            done = bool(terminated) or bool(truncated)
        else:  # gym
            obs, r, done, info = out
        n_actual += 1
        if isinstance(info, dict):
            last_pv = float(info.get("portfolio_value", info.get("total_value", last_pv)) or last_pv)
        if done:
            gi = env.get_info() if hasattr(env, "get_info") else {}
            last_pv = float(gi.get("portfolio_value", last_pv) or last_pv)
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            done = False
    # final authoritative values
    gi = env.get_info() if hasattr(env, "get_info") else {}
    last_pv = float(gi.get("portfolio_value", last_pv) or last_pv)
    trades = _extract_closed_trades(env)
    m = _metrics_from_trades(trades, last_pv, initial_pv)
    m["policy"] = kind
    m["steps"] = n_actual
    m["terminal_pv"] = last_pv
    try:
        env.close()
    except Exception:
        pass
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--split", type=str, default="val")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str,
                    default=str(REPO_ROOT / "logs" / "validation" / "trivial_policy_audit.json"))
    args = ap.parse_args()

    policies = ["AlwaysLong", "AlwaysFlat", "AlwaysShort", "Random"]
    results = {}
    for p in policies:
        print(f"\n===== RUNNING {p} ({args.steps} steps, split={args.split}) =====", flush=True)
        try:
            m = run_policy(p, args.steps, args.split, args.seed)
        except Exception as e:
            import traceback
            traceback.print_exc()
            m = {"policy": p, "error": f"{type(e).__name__}: {e}"}
        results[p] = m
        if "error" not in m:
            print(f"  n_trades={m['n_trades']} EV={m['EV_usdt']:+.4f}usdt "
                  f"({m['EV_pct']:+.3f}%) WR={m['win_rate']:.3f} "
                  f"PF={m['profit_factor']:.2f} Sharpe={m['sharpe_per_trade']:+.3f} "
                  f"MaxDD={m['max_dd_pct']:.2f}% termRet={m['terminal_return_pct']:+.2f}% "
                  f"hold={m['mean_hold_bars']:.1f} reasons={m['reason_hist']}", flush=True)
        else:
            print(f"  ERROR: {m['error']}", flush=True)

    # ---- Verdict ----
    print("\n\n" + "=" * 78)
    print("ÉTAPE 1 VERDICT — is AlwaysLong dominant? (MDP bias test)")
    print("=" * 78)
    hdr = f"{'policy':<12} {'n_tr':>5} {'EV_usdt':>9} {'EV_%':>7} {'WR':>6} {'PF':>7} {'Sharpe':>8} {'MaxDD%':>8} {'termRet%':>9}"
    print(hdr)
    print("-" * len(hdr))
    for p in policies:
        m = results[p]
        if "error" in m:
            print(f"{p:<12} ERROR {m['error']}")
            continue
        print(f"{p:<12} {m['n_trades']:>5} {m['EV_usdt']:>+9.4f} {m['EV_pct']:>+7.3f} "
              f"{m['win_rate']:>6.3f} {m['profit_factor']:>7.2f} {m['sharpe_per_trade']:>+8.3f} "
              f"{m['max_dd_pct']:>8.2f} {m['terminal_return_pct']:>+9.2f}")

    def _val(p, k):
        return results.get(p, {}).get(k, None)

    al_sharpe = _val("AlwaysLong", "sharpe_per_trade")
    al_ev = _val("AlwaysLong", "EV_usdt")
    al_ret = _val("AlwaysLong", "terminal_return_pct")
    rnd_sharpe = _val("Random", "sharpe_per_trade")
    rnd_ret = _val("Random", "terminal_return_pct")
    flat_ret = _val("AlwaysFlat", "terminal_return_pct")

    verdict = {"bifurcation": None, "reason": ""}
    if None in (al_ev, al_ret, rnd_ret, flat_ret):
        verdict["reason"] = "insufficient data (a policy errored or made 0 trades)"
    else:
        long_beats_random = (al_ret > rnd_ret) and ((al_sharpe or 0) >= (rnd_sharpe or 0))
        long_beats_flat = al_ret > flat_ret
        long_positive_ev = (al_ev > 0)
        if long_beats_random and long_beats_flat:
            verdict["bifurcation"] = "A"
            verdict["reason"] = (
                f"AlwaysLong DOMINATES: termRet {al_ret:+.2f}% > Random {rnd_ret:+.2f}% "
                f"and > Flat {flat_ret:+.2f}%; EV/trade {al_ev:+.4f} "
                f"({'positive' if long_positive_ev else 'negative'}). "
                "MDP IS BIASED -> disable oracle / episodic reward.")
        else:
            verdict["bifurcation"] = "B"
            verdict["reason"] = (
                f"AlwaysLong does NOT dominate (termRet {al_ret:+.2f}% vs Random "
                f"{rnd_ret:+.2f}% / Flat {flat_ret:+.2f}%). Env plausibly sane; "
                "collapse is architectural (PPO/action-head) -> ÉTAPE 2.")

    print("\n>>> BIFURCATION:", verdict["bifurcation"])
    print(">>>", verdict["reason"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": results, "verdict": verdict,
                               "steps": args.steps, "split": args.split,
                               "seed": args.seed}, indent=2, default=str))
    print(f"\n[SAVED] {out}")


if __name__ == "__main__":
    main()
