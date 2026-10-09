"""
Counterfactual simulation battery — ADAN V27 checkpoint 320k (FROZEN reference).

Runs 6 arms on the SAME val data, SAME initial state, per the forensic
decision tree (docs/V27_320K_FORENSIC_VERDICT.md, section 8):

  A — real policy (checkpoint 320k, deterministic)
  B — forced HOLD  (direction overridden to 0.0)
  C — forced BUY   (direction overridden to +1.0)
  D — forced SELL  (direction overridden to -1.0)
  E — real policy, ZERO-FEE config (commission/slippage/round_trip zeroed)
  F — real policy, reward re-accounted with -0.05 per sterile SELL
      (sell_while_flat). Accounting-only arm: behavior is identical to A
      (no gradient in a single pass); it quantifies how the proposed
      Layer-B penalty WOULD have scored the observed behavior.

Forced arms keep the model's own size/sl/tp/tf outputs and override ONLY
the direction component a[0] — isolating the directional decision.

No repo code is modified. The checkpoint is never overwritten.
Writes per-arm JSON + combined summary to logs/validation/counterfactual_320k/.

Usage:
  python scripts/backtest/counterfactual_320k.py [--steps 3000] [--split val]
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

CKPT = REPO_ROOT / "checkpoints" / "ppo_adan0_v27_hmm_semantic_20260809T123040Z_checkpoint_320000_steps.zip"
OUT_DIR = REPO_ROOT / "logs" / "validation" / "counterfactual_320k"

SELL_THRESHOLD = 0.05          # 5m action threshold (config: action_thresholds.5m)
STERILE_SELL_PENALTY = -0.05   # arm F: hypothetical Layer-B penalty
INITIAL_PV = 20.50


def zero_fees(cfg: dict) -> dict:
    """Recursively zero every fee/slippage knob (arm E)."""
    FEE_KEYS = {"commission", "commission_pct", "slippage_pct", "round_trip_fees",
                "entry_fee", "exit_fee", "maker_fee", "taker_fee"}
    def _walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k in FEE_KEYS and isinstance(v, (int, float)):
                    node[k] = 0.0
                else:
                    _walk(v)
        elif isinstance(node, list):
            for item in node:
                _walk(item)
    cfg = copy.deepcopy(cfg)
    _walk(cfg)
    return cfg


def build_env(cfg, wc, worker_idx):
    from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader
    from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
    data = ChunkedDataLoader(config=cfg, worker_config=wc, worker_id=worker_idx).load_chunk(0)
    return MultiAssetChunkedEnv(data=data, config=cfg, worker_config=wc,
                                worker_id=worker_idx, live_mode=False)


def run_arm(model, cfg, wc, arm: str, steps: int) -> dict:
    from stable_baselines3.common.vec_env import DummyVecEnv
    env = build_env(cfg, wc, 0)
    vec_env = DummyVecEnv([lambda: env])
    obs = vec_env.reset()

    equity_curve = []
    pnls = []
    rewards = []
    adj_rewards = []           # arm F accounting
    sterile_sells = 0
    reject_agg: dict = {}
    attempts = 0
    last_pv = INITIAL_PV

    for s in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        a = np.asarray(action, dtype=np.float64).flatten().copy()

        open_pos = 0
        try:
            open_pos = int(env.get_info().get("open_positions", 0))
        except Exception:
            pass

        # --- sterile SELL accounting (arm F) : direction strongly negative while flat
        if arm == "F" and a[0] < -SELL_THRESHOLD and open_pos == 0:
            sterile_sells += 1

        # --- direction override for forced arms (keep size/sl/tp from policy)
        if arm == "B":
            a[0] = 0.0
        elif arm == "C":
            a[0] = 1.0
        elif arm == "D":
            a[0] = -1.0

        obs, r, dones, infos = vec_env.step(a.reshape(1, -1))
        info = infos[0] if isinstance(infos, (list, tuple)) and infos else (infos or {})

        rv = float(np.asarray(r).flat[0])
        rewards.append(rv)
        if arm == "F":
            adj_rewards.append(rv + (STERILE_SELL_PENALTY if (a[0] < -SELL_THRESHOLD and open_pos == 0) else 0.0))

        for reason, cnt in (info.get("rejection_reasons") or {}).items():
            reject_agg[reason] = reject_agg.get(reason, 0) + cnt
        attempts += int(info.get("trade_attempts", 0))

        pnl = float(info.get("realized_pnl", info.get("trade_pnl", 0.0)) or 0.0)
        if abs(pnl) > 1e-9:
            pnls.append(pnl)
        last_pv = float(info.get("portfolio_value", last_pv) or last_pv)
        equity_curve.append(last_pv)

        done = bool(np.asarray(dones).flat[0])
        if done:
            obs = vec_env.reset()

    # ---- env-authoritative counters
    env_info = {}
    try:
        env_info = env.get_info()
    except Exception as e:
        env_info = {"get_info_error": str(e)}

    eq = np.asarray(equity_curve, dtype=np.float64)
    peak = np.maximum.accumulate(eq) if len(eq) else np.array([INITIAL_PV])
    dd = (peak - eq) / np.maximum(peak, 1e-9) if len(eq) else np.array([0.0])
    rets = np.diff(eq) / np.maximum(eq[:-1], 1e-9) if len(eq) > 1 else np.array([0.0])
    sharpe = float(np.mean(rets) / (np.std(rets) + 1e-12) * np.sqrt(len(rets))) if len(rets) > 1 else 0.0

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    result = {
        "arm": arm,
        "steps": steps,
        "final_equity": float(last_pv),
        "total_return_pct": float((last_pv - INITIAL_PV) / INITIAL_PV * 100.0),
        "max_drawdown_pct": float(np.max(dd) * 100.0),
        "sharpe_like": sharpe,
        "trades_detected": len(pnls),
        "env_total_trades": int(env_info.get("total_trades", -1)),
        "win_rate": float(len(wins) / len(pnls)) if pnls else 0.0,
        "expectancy_per_trade": float(np.mean(pnls)) if pnls else 0.0,
        "gross_profit": float(sum(wins)),
        "gross_loss": float(sum(losses)),
        "profit_factor": float(sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else (float("inf") if wins else 0.0),
        "trade_attempts": attempts,
        "rejection_reasons": reject_agg,
        "reward_sum": float(np.sum(rewards)),
        "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
        "env_total_realized_pnl": float(env_info.get("total_realized_pnl", 0.0)),
        "exposure_steps_open_pct": None,  # filled below if available
    }
    if arm == "F":
        result["sterile_sell_count"] = sterile_sells
        result["adj_reward_sum"] = float(np.sum(adj_rewards))
        result["adj_reward_mean"] = float(np.mean(adj_rewards)) if adj_rewards else 0.0
        result["penalty_total"] = float(sterile_sells * STERILE_SELL_PENALTY)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--split", type=str, default="val")
    ap.add_argument("--arms", type=str, default="A,B,C,D,E,F")
    args = ap.parse_args()

    from stable_baselines3 import PPO
    from adan_trading_bot.common.config_loader import ConfigLoader

    cfg = ConfigLoader.load_config(str(REPO_ROOT / "config" / "config.yaml"))
    cfg.setdefault("environment", {})["rich_display_interval"] = 999999
    wc = copy.deepcopy(cfg.get("workers", {}).get("w1", {}))
    wc.update({"worker_id": 0, "data_split_override": args.split,
               "timeframes": ["5m", "1h", "4h"], "assets": ["BTCUSDT"]})

    print(f"[counterfactual] loading {CKPT.name}", file=sys.stderr)
    model = PPO.load(str(CKPT), device="cpu")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    for arm in [x.strip() for x in args.arms.split(",") if x.strip()]:
        arm_cfg = zero_fees(cfg) if arm == "E" else cfg
        print(f"[counterfactual] === ARM {arm} ({args.steps} steps, split={args.split}) ===", file=sys.stderr)
        try:
            res = run_arm(model, arm_cfg, wc, arm, args.steps)
        except Exception as e:
            res = {"arm": arm, "error": f"{type(e).__name__}: {e}"}
        summary[arm] = res
        (OUT_DIR / f"arm_{arm}.json").write_text(json.dumps(res, indent=2))
        print(f"[counterfactual] arm {arm} -> equity={res.get('final_equity')} "
              f"ret={res.get('total_return_pct')}% trades={res.get('env_total_trades')} "
              f"reward_sum={res.get('reward_sum')}", file=sys.stderr)

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
