"""Fixed-capital backtest — honest strategy return, no compounding.

Why this exists
---------------
`deterministic_backtest.py` reports `total_return_pct = (portfolio_value - 20.50)/20.50`
where `portfolio_value` is the RL env's COMPOUNDED equity, and the env-level
`total_realized_pnl` accumulates across episode resets. That mixes
position-sizing-on-growing-capital with strategy quality and produces
arithmetically impossible figures (e.g. +1485% on a -9% market).

This script measures the STRATEGY instead of the equity trajectory:

  * Each closed trade contributes its `pnl_pct` (percentage return on entry
    price, fees included) — read straight from the portfolio trade_log.
  * That percentage is applied to a FIXED notional ($100) per trade. No gains
    are reinvested, so 90 winning trades cannot snowball into +1485%.
  * Episodes are reset cleanly; we drain the trade_log per episode so trades
    are never double counted across resets.

Output metrics (the honest set):
  - n_trades, win_rate
  - avg_pnl_pct / trade, median, best, worst
  - total_return_pct  = sum(pnl_pct) * (FIXED_NOTIONAL / FIXED_CAPITAL)
  - profit_factor     = gross_win / gross_loss
  - sharpe_like        = mean(pnl_pct) / std(pnl_pct)
  - max_consecutive_losses, expectancy

Usage:
  PYTHONPATH=src python3 scripts/backtest_fixed_capital.py \
      --ckpt checkpoints/ppo_adan0_500k_FIXED.zip --split test --steps 5000 \
      --out logs/validation/backtest_CORRECTED_500k_test.json
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

# NOTE: this script lives in scripts/backtest/, so the repo root is THREE
# levels up (backtest/ -> scripts/ -> <repo root>). Using parent.parent only
# resolved to scripts/ and made config/data lookups fail with FileNotFoundError.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ.setdefault("ADAN_TRAINING_SILENT", "1")
os.environ.setdefault("ADAN_RICH_STEP_EVERY", "999999")
logging.disable(logging.WARNING)

# Honest fixed-capital convention: every trade risks the SAME notional.
FIXED_CAPITAL = 1000.0    # reference account, never grows
FIXED_NOTIONAL = 100.0    # dollars deployed per trade (10% of fixed account)


def _drain_trade_log(pm) -> list:
    """Pop all receipts from the portfolio trade_log (closed trades only)."""
    out = []
    tl = getattr(pm, "trade_log", None)
    if tl is None:
        return out
    while len(tl) > 0:
        out.append(tl.popleft())
    return out


def _extract_pnl_pct(receipt) -> float | None:
    """Return the trade's percentage return (fees included) or None if not a close."""
    if not isinstance(receipt, dict):
        return None
    # close_position receipts carry pnl_pct; entry receipts do not.
    if "pnl_pct" in receipt and receipt.get("pnl_pct") is not None:
        try:
            return float(receipt["pnl_pct"])
        except (TypeError, ValueError):
            return None
    # Fallback: derive from pnl / (entry_price*size) if present.
    pnl = receipt.get("pnl")
    ep = receipt.get("entry_price")
    sz = receipt.get("size")
    if pnl is not None and ep and sz:
        notional = abs(float(ep) * float(sz))
        if notional > 0:
            return float(pnl) / notional * 100.0
    return None


def run_fixed_capital_backtest(ckpt_path: str, steps: int, split: str,
                               agent: str = "model") -> dict:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from adan_trading_bot.common.config_loader import ConfigLoader
    from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader
    from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv

    cfg = ConfigLoader.load_config(str(REPO_ROOT / "config" / "config.yaml"))
    cfg.setdefault("environment", {})["rich_display_interval"] = 999999

    wc = copy.deepcopy(cfg.get("workers", {}).get("w1", {}))
    wc.update({
        "worker_id": 0,
        # MultiAssetChunkedEnv rebuilds its OWN ChunkedDataLoader internally and
        # reads worker_config["data_split"] (NOT "data_split_override") — see
        # multi_asset_chunked_env.py:_build_data_loader. We must set BOTH keys so
        # the env actually loads the requested split instead of defaulting to
        # "train". (Bug found when val results came out byte-identical to test.)
        "data_split": split,
        "data_split_override": split,
        "timeframes": ["5m", "1h", "4h"],
        "assets": ["BTCUSDT"],
    })

    try:
        data = ChunkedDataLoader(config=cfg, worker_config=wc, worker_id=0).load_chunk(0)
    except Exception as e:
        return {"error": f"no {split} data: {e}"}

    env = MultiAssetChunkedEnv(
        data=data, config=cfg, worker_config=wc, worker_id=0, live_mode=False
    )
    vec_env = DummyVecEnv([lambda: env])

    model = None
    rng = np.random.default_rng(42)
    if agent == "model":
        model = PPO.load(ckpt_path, device="cpu")
        model.set_env(vec_env)

    obs = vec_env.reset()
    underlying = vec_env.envs[0]
    pm = underlying.portfolio_manager
    _drain_trade_log(pm)  # discard anything from reset

    trade_returns: list[float] = []   # pnl_pct per closed trade
    n_episodes = 0
    act_dim = int(np.prod(vec_env.action_space.shape))

    for s in range(steps):
        if agent == "random":
            # Uniform random action in the Box(-1,1) action space.
            action = rng.uniform(-1.0, 1.0, size=(1, act_dim)).astype(np.float32)
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, _r, dones, _infos = vec_env.step(action)

        # Harvest any trades that closed this step.
        for receipt in _drain_trade_log(pm):
            pct = _extract_pnl_pct(receipt)
            if pct is not None:
                trade_returns.append(pct)

        done = bool(np.ravel(dones)[0])
        if done:
            n_episodes += 1
            obs = vec_env.reset()
            underlying = vec_env.envs[0]
            pm = underlying.portfolio_manager
            _drain_trade_log(pm)  # fresh episode → discard reset noise

    # ── Honest metrics on fixed notional, additive (no compounding) ──
    arr = np.array(trade_returns, dtype=float)
    n = len(arr)
    if n == 0:
        return {
            "agent": agent, "checkpoint": os.path.basename(ckpt_path),
            "split": split, "steps_tested": steps, "episodes": n_episodes,
            "n_trades": 0, "verdict": "NO_TRADES",
        }

    wins = arr[arr > 0]
    losses = arr[arr < 0]
    win_rate = len(wins) / n
    gross_win = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(-losses.sum()) if len(losses) else 0.0
    profit_factor = (gross_win / gross_loss) if gross_loss > 1e-9 else float("inf")
    # Fixed-notional return: each trade's % applied to FIXED_NOTIONAL, summed,
    # expressed against the FIXED_CAPITAL reference account.
    total_return_pct = float(arr.sum()) * (FIXED_NOTIONAL / FIXED_CAPITAL)
    sharpe_like = float(arr.mean() / arr.std()) if arr.std() > 1e-9 else 0.0
    expectancy = float(arr.mean())

    # Max consecutive losses
    max_consec_loss = 0
    cur = 0
    for x in arr:
        if x < 0:
            cur += 1
            max_consec_loss = max(max_consec_loss, cur)
        else:
            cur = 0

    return {
        "agent": agent,
        "checkpoint": os.path.basename(ckpt_path),
        "split": split,
        "steps_tested": steps,
        "episodes": n_episodes,
        "fixed_capital": FIXED_CAPITAL,
        "fixed_notional": FIXED_NOTIONAL,
        "n_trades": n,
        "win_rate": round(win_rate, 4),
        "avg_pnl_pct_per_trade": round(expectancy, 4),
        "median_pnl_pct": round(float(np.median(arr)), 4),
        "best_trade_pct": round(float(arr.max()), 4),
        "worst_trade_pct": round(float(arr.min()), 4),
        "gross_win_pct": round(gross_win, 4),
        "gross_loss_pct": round(gross_loss, 4),
        "profit_factor": round(profit_factor, 4) if np.isfinite(profit_factor) else 1e9,
        "total_return_pct": round(total_return_pct, 4),
        "sharpe_like": round(sharpe_like, 4),
        "expectancy_pct": round(expectancy, 4),
        "max_consecutive_losses": int(max_consec_loss),
        "verdict": _verdict(win_rate, expectancy, n),
    }


def _verdict(win_rate: float, expectancy: float, n: int) -> str:
    if n < 10:
        return f"INSUFFICIENT_SAMPLE ({n} trades)"
    if expectancy > 0 and win_rate > 0.5:
        return f"POSITIVE_EDGE (WR={win_rate*100:.1f}%, E={expectancy:+.3f}%/trade)"
    if expectancy > 0:
        return f"MARGINAL (WR={win_rate*100:.1f}%, E={expectancy:+.3f}%/trade)"
    return f"NO_EDGE (WR={win_rate*100:.1f}%, E={expectancy:+.3f}%/trade)"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--split", type=str, default="test", choices=["val", "test", "train"])
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--agent", type=str, default="model", choices=["model", "random"])
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    if args.agent == "model" and not args.ckpt:
        print("ERROR: --ckpt required for --agent model", file=sys.stderr)
        return 1

    print(f"[fixed-capital] agent={args.agent} ckpt={args.ckpt} split={args.split} steps={args.steps}",
          file=sys.stderr)
    result = run_fixed_capital_backtest(args.ckpt or "RANDOM", args.steps, args.split, args.agent)

    out_dir = REPO_ROOT / "logs" / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else out_dir / f"backtest_FIXED_{args.agent}_{args.split}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[fixed-capital] saved: {out_path}", file=sys.stderr)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
