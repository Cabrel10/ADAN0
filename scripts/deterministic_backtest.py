"""
Honest deterministic backtest — no SDE noise, no fake metrics.

Loads the most recent ppo_adan0_sandbox_*.zip checkpoint (and matching
_vecnorm.pkl), runs N steps on val/test data with deterministic=True, and
emits structured JSON to logs/validation/backtest_<cum_steps>.json.

Used by:
  - local sanity checks after each Session 10 fix
  - .github/workflows/adan0_relay.yml after each training segment
  - CI gate: exit code 0 always (informational), reads logs/validation/*.json

Usage:
  python scripts/deterministic_backtest.py [--steps 1000] [--split val]
                                            [--ckpt PATH] [--out PATH]
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# Silence env's INFO chatter during evaluation
os.environ.setdefault("ADAN_TRAINING_SILENT", "1")
os.environ.setdefault("ADAN_RICH_STEP_EVERY", "999999")
logging.disable(logging.WARNING)


def find_latest_checkpoint(ckpt_dir: str | None = None, worker_idx: int | None = None) -> str | None:
    """Find latest checkpoint. Support both sandbox and PBT structures.
    
    Args:
        ckpt_dir: Optional directory to search in (PBT ray results structure)
        worker_idx: Optional worker index to filter by (for PBT training)
    
    Returns:
        Path to the latest checkpoint, or None if not found
    """
    if ckpt_dir:
        # PBT structure: /path/to/adan_pbt_training/ADAN_PBT_Worker_b7791_00001_1_ent_coef=...
        if worker_idx is not None:
            pattern = f"{ckpt_dir}/ADAN_PBT_Worker_*_worker_idx={worker_idx}_*/checkpoint_*/model.zip"
        else:
            pattern = f"{ckpt_dir}/ADAN_PBT_Worker_*/checkpoint_*/model.zip"
        ckpts = sorted(glob.glob(pattern), key=os.path.getmtime)
        if ckpts:
            return ckpts[-1]
    
    # Fallback to sandbox structure
    ckpts = sorted(
        glob.glob(str(REPO_ROOT / "checkpoints" / "ppo_adan0_sandbox_*.zip")),
        key=os.path.getmtime,
    )
    return ckpts[-1] if ckpts else None


def run_backtest(ckpt_path: str, steps: int = 1000, split: str = "val") -> dict:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from adan_trading_bot.common.config_loader import ConfigLoader
    from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader
    from adan_trading_bot.environment.multi_asset_chunked_env import (
        MultiAssetChunkedEnv,
    )

    cfg = ConfigLoader.load_config(str(REPO_ROOT / "config" / "config.yaml"))
    cfg.setdefault("environment", {})["rich_display_interval"] = 999999
    
    # DEBUG: Print action_thresholds from config
    env_thresholds = cfg.get("environment", {}).get("action_thresholds", {})
    print(f"[DEBUG] Config action_thresholds: {env_thresholds}", file=sys.stderr)

    # DYNAMIC WORKER IDENTIFICATION - Extract worker_idx from checkpoint path
    import re
    worker_idx = 0
    match = re.search(r'worker_idx=(\d+)', ckpt_path)
    if match:
        worker_idx = int(match.group(1))
    worker_key = f"w{worker_idx + 1}"
    print(f"[DEBUG] Loading profile for {worker_key} (worker_idx={worker_idx})", file=sys.stderr)
    
    wc = copy.deepcopy(cfg.get("workers", {}).get(worker_key, {}))
    wc.update({
        "worker_id": worker_idx,
        "data_split_override": split,
        "timeframes": ["5m", "1h", "4h"],
        "assets": ["BTCUSDT"],
    })

    try:
        data = ChunkedDataLoader(config=cfg, worker_config=wc, worker_id=worker_idx).load_chunk(0)
    except Exception as e:
        return {"error": f"no {split} data: {e}"}

    # Create 2 envs for backtest (Ray PBT training uses n_envs=2)
    env1 = MultiAssetChunkedEnv(
        data=data, config=cfg, worker_config=wc, worker_id=worker_idx, live_mode=False
    )
    env2 = MultiAssetChunkedEnv(
        data=data, config=cfg, worker_config=wc, worker_id=worker_idx, live_mode=False
    )
    vec_env = DummyVecEnv([lambda: env1, lambda: env2])
    
    # FIX: Use VecNormalize from the same checkpoint directory (PBT), not sandbox
    vecnorm_path = None
    
    # Strategy 1: Look for vecnormalize.pkl in the same directory as checkpoint
    ckpt_dir = os.path.dirname(ckpt_path)
    candidate1 = os.path.join(ckpt_dir, "vecnormalize.pkl")
    if os.path.isfile(candidate1):
        vecnorm_path = candidate1
        print(f"[DEBUG] Found VecNormalize in checkpoint dir: {vecnorm_path}", file=sys.stderr)
    
    # Strategy 2: Look for _vecnorm.pkl next to checkpoint
    if not vecnorm_path:
        candidate2 = ckpt_path.replace(".zip", "_vecnorm.pkl")
        if os.path.isfile(candidate2):
            vecnorm_path = candidate2
            print(f"[DEBUG] Found VecNormalize with _vecnorm suffix: {vecnorm_path}", file=sys.stderr)
    
    # Strategy 3: Look in parent directory (for Ray results structure)
    if not vecnorm_path:
        parent_dir = os.path.dirname(ckpt_dir)
        candidate3 = os.path.join(parent_dir, "vecnormalize.pkl")
        if os.path.isfile(candidate3):
            vecnorm_path = candidate3
            print(f"[DEBUG] Found VecNormalize in parent dir: {vecnorm_path}", file=sys.stderr)
    
    # Strategy 4: Fallback to sandbox (only if nothing else found)
    if not vecnorm_path:
        canonical = str(REPO_ROOT / "checkpoints" / "vecnormalize_sandbox.pkl")
        if os.path.isfile(canonical):
            vecnorm_path = canonical
            print(f"[WARNING] Using fallback sandbox VecNormalize: {vecnorm_path}", file=sys.stderr)
    
    if vecnorm_path:
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    # FIX: Load model WITHOUT env to avoid double VecNormalize wrapping
    # PPO.load(ckpt, env=vec_env) would create a second VecNormalize internally
    model = PPO.load(ckpt_path, device="cpu")
    model.set_env(vec_env)

    obs = vec_env.reset()
    trades = 0
    pnls: list[float] = []
    actions_max: list[float] = []
    actions_mean: list[float] = []
    initial_pv = 20.50
    last_pv = initial_pv
    
    # DEBUG: Log which VecNormalize was used
    print(f"[DEBUG] VecNormalize path: {vecnorm_path}", file=sys.stderr)
    if vecnorm_path and os.path.isfile(vecnorm_path):
        print(f"[DEBUG] VecNormalize loaded from: {vecnorm_path}", file=sys.stderr)
    else:
        print(f"[DEBUG] WARNING: VecNormalize not found or not loaded", file=sys.stderr)

    # Detailed logging
    rejection_reasons = {}
    action_samples = []
    trade_attempts = 0
    step_logs = []
    
    for s in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        a = np.asarray(action).flatten()
        actions_max.append(float(np.max(np.abs(a))))
        actions_mean.append(float(np.mean(np.abs(a))))
        
        # DEBUG: Log action shape and size
        if s < 3:
            print(f"[DEBUG] Step {s}: action.shape={np.asarray(action).shape}, len(a)={len(a)}, expected={len(wc.get('assets', ['BTCUSDT'])) * 5}", file=sys.stderr)
        
        obs, _r, dones, infos = vec_env.step(action)
        info = infos[0] if isinstance(infos, (list, tuple)) and infos else (infos or {})
        
        # Extract detailed rejection info
        if "rejection_reasons" in info:
            for reason, count in info["rejection_reasons"].items():
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + count
        
        # Track trade attempts
        if "trade_attempts" in info:
            trade_attempts += info["trade_attempts"]
        
        pnl = float(info.get("realized_pnl", info.get("trade_pnl", 0.0)))
        if abs(pnl) > 1e-6:
            trades += 1
            pnls.append(pnl)
        
        last_pv = float(info.get("portfolio_value", last_pv))
        
        # Log EVERY step with full details
        # Handle reward conversion safely
        if isinstance(_r, (list, tuple)):
            reward_val = float(_r[0]) if len(_r) > 0 else 0.0
        elif isinstance(_r, np.ndarray):
            reward_val = float(_r.flat[0]) if _r.size > 0 else 0.0
        else:
            reward_val = float(_r)
        
        step_log = {
            "step": s,
            "action_raw": a[:5].tolist() if len(a) >= 5 else a.tolist(),
            "action_max": float(np.max(np.abs(a))),
            "action_mean": float(np.mean(np.abs(a))),
            "realized_pnl": pnl,
            "portfolio_value": last_pv,
            "trade_attempts": info.get("trade_attempts", 0),
            "rejection_reasons": info.get("rejection_reasons", {}),
            "open_positions": info.get("open_positions", 0),
            "total_trades": info.get("total_trades", 0),
            "reward": reward_val,
            "done": bool(dones[0]) if isinstance(dones, (list, tuple)) else (bool(dones.flat[0]) if isinstance(dones, np.ndarray) else bool(dones)),
        }
        step_logs.append(step_log)
        
        # Print EVERY step with detailed breakdown
        rejection_str = ""
        if step_log['rejection_reasons']:
            rejection_str = " | Rejections: " + ", ".join(
                f"{k}={v}" for k, v in step_log['rejection_reasons'].items() if v > 0
            )
        
        # Show full action vector for first 10 steps
        action_detail = ""
        if s < 10:
            action_detail = f" | Full_action={a.tolist()}"
        
        print(f"[STEP {s:5d}] "
              f"Action_max={step_log['action_max']:.4f} "
              f"Action_mean={step_log['action_mean']:.4f}{action_detail} | "
              f"Attempts={step_log['trade_attempts']} "
              f"Open_pos={step_log['open_positions']} "
              f"Total_trades={step_log['total_trades']}{rejection_str} | "
              f"PnL=${pnl:+.4f} | Portfolio=${last_pv:.2f} | "
              f"Reward={step_log['reward']:+.6f}", file=sys.stderr)
        
        if dones[0] if isinstance(dones, (list, tuple)) else (dones.flat[0] if isinstance(dones, np.ndarray) else dones):
            obs = vec_env.reset()

    # Pull authoritative trade count from the underlying env
    env_info = {}
    try:
        underlying = vec_env.envs[0] if hasattr(vec_env, "envs") else None
        if underlying is None and hasattr(vec_env, "venv"):
            underlying = vec_env.venv.envs[0]
        if underlying is not None and hasattr(underlying, "get_info"):
            env_info = underlying.get_info()
    except Exception as e:
        env_info = {"get_info_error": str(e)}

    total_return_pct = (last_pv - initial_pv) / initial_pv * 100.0
    win_rate = sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0.0
    
    # Print summary to stderr
    print(f"\n[BACKTEST_SUMMARY]", file=sys.stderr)
    print(f"  Total steps: {steps}", file=sys.stderr)
    print(f"  Trades executed: {trades}", file=sys.stderr)
    print(f"  Trade attempts: {trade_attempts}", file=sys.stderr)
    print(f"  Rejection reasons: {rejection_reasons}", file=sys.stderr)
    print(f"  Action stats: mean={np.mean(actions_max):.4f}, max={np.max(actions_max):.4f}", file=sys.stderr)
    print(f"  PnL: ${sum(pnls):.2f}, Portfolio: ${last_pv:.2f}", file=sys.stderr)
    
    return {
        "checkpoint": os.path.basename(ckpt_path),
        "ckpt_size_bytes": os.path.getsize(ckpt_path),
        "vecnorm_used": vecnorm_path,
        "split": split,
        "steps_tested": steps,
        # Per-step counts (from infos)
        "trades_detected_in_loop": trades,
        "trade_attempts": trade_attempts,
        "rejection_reasons": rejection_reasons,
        "pnl_sum": float(sum(pnls)),
        "pnl_avg": float(np.mean(pnls)) if pnls else 0.0,
        "win_rate": float(win_rate),
        # Action statistics — the true health check
        "action_max_mean": float(np.mean(actions_max)),
        "action_max_max": float(np.max(actions_max)),
        "action_max_pct_above_001": float(np.mean(np.array(actions_max) > 0.01)),
        "action_max_pct_above_005": float(np.mean(np.array(actions_max) > 0.05)),
        # Equity
        "initial_equity": initial_pv,
        "final_equity": last_pv,
        "total_return_pct": float(total_return_pct),
        # Underlying env counter (the source of truth)
        "env_total_trades": int(env_info.get("total_trades", -1)),
        "env_winning_trades": int(env_info.get("winning_trades", -1)),
        "env_losing_trades": int(env_info.get("losing_trades", -1)),
        "env_total_realized_pnl": float(env_info.get("total_realized_pnl", 0.0)),
        "env_drawdown_pct": float(env_info.get("drawdown_pct", 0.0)),
    }


def grade(result: dict) -> str:
    """Honest grading. No vanity metrics."""
    if "error" in result:
        return "ERROR"
    t = result.get("env_total_trades", 0)
    amm = result.get("action_max_mean", 0.0)
    ret = result.get("total_return_pct", 0.0)
    if t > 0 and ret > 0:
        return f"PROFITABLE ({t} trades, {ret:+.2f}%)"
    if t > 0 and ret <= 0:
        return f"TRADING_NONPROFITABLE ({t} trades, {ret:+.2f}%)"
    if t == 0 and amm > 0.01:
        return f"GATED ({amm:.4f} actions, threshold likely blocks)"
    if t == 0 and amm <= 0.01:
        return f"POLICY_TOO_SMALL (action_max_mean={amm:.4f}, need more steps)"
    return f"UNKNOWN (t={t}, amm={amm:.4f}, ret={ret:+.2f}%)"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--split", type=str, default="val", choices=["val", "test", "train"])
    parser.add_argument("--ckpt", type=str, default=None, help="Path to specific checkpoint")
    parser.add_argument("--ckpt-dir", type=str, default=None, help="PBT checkpoint directory (e.g., /path/to/adan_pbt_training)")
    parser.add_argument("--worker", type=int, default=None, help="Worker index for PBT checkpoints")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    # Determine checkpoint to use
    if args.ckpt:
        # User specified exact path
        ckpt = args.ckpt
    elif args.ckpt_dir or args.worker is not None:
        # Search PBT directory for worker checkpoint
        ckpt = find_latest_checkpoint(ckpt_dir=args.ckpt_dir, worker_idx=args.worker)
    else:
        # Try to find any latest checkpoint (sandbox or PBT)
        ckpt = find_latest_checkpoint()
    
    if not ckpt:
        error_msg = {"error": "no checkpoint found"}
        if args.ckpt_dir or args.worker is not None:
            error_msg["details"] = f"No PBT checkpoint found in {args.ckpt_dir} for worker {args.worker}"
        print(json.dumps(error_msg, indent=2))
        return 0
    
    print(f"[backtest] checkpoint: {ckpt}", file=sys.stderr)

    result = run_backtest(ckpt, steps=args.steps, split=args.split)
    result["verdict"] = grade(result)

    out_dir = REPO_ROOT / "logs" / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.out:
        out_path = Path(args.out)
    else:
        cum = ""
        m = os.path.basename(ckpt).replace(".zip", "")
        if "_" in m:
            cum = m.split("_")[-1].replace("steps", "")
        # Add worker suffix if applicable
        worker_suffix = f"_w{args.worker}" if args.worker is not None else ""
        out_path = out_dir / f"backtest_{cum or 'latest'}{worker_suffix}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[backtest] saved: {out_path}", file=sys.stderr)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
