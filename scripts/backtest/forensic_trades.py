"""Forensic per-trade backtest — answers the 4 blind spots with FACTS.

This builds on backtest_fixed_capital.py but, instead of only aggregating
pnl_pct, it captures EVERY closed trade with the context needed to answer:

  1. ZONES (GREEN/ORANGE/RED): for each trade we replay the env's own ex-post
     MFE/MAE on the 5m chunk at the entry index (identical math to
     multi_asset_chunked_env.py:_future_zone_contribution) and classify the
     zone with the SAME ZoneConfig the training uses. Then we report
     trades / WR / expectancy PER ZONE. If GREEN expectancy <= ORANGE,
     the zone system is broken.

  2. SL / ATR & TP / ATR: SL%/TP% are read from the close receipt; ATR% at
     entry is read from the 5m chunk feature `atr_pct` at the entry index.
     Distance_SL/ATR = SL% / ATR%. If SL_distance/ATR < 0.5 -> SL too tight.

  3. candles-before-SL_HIT: hold_steps = current_step - open_step per trade,
     bucketed; if 50% of SL hit in <3 candles -> noise stop-out.

  4. illegal_actions_ratio: read env.rejection_reasons at the end
     (sell_no_position + min_notional + ...) / total_steps.

Usage:
  PYTHONPATH=src python3 scripts/backtest/forensic_trades.py \
      --ckpt checkpoints/ppo_adan0_sandbox_checkpoint_430000_steps.zip \
      --split test --steps 5000 \
      --out logs/validation/forensic/forensic_430k.json
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ.setdefault("ADAN_TRAINING_SILENT", "1")
os.environ.setdefault("ADAN_RICH_STEP_EVERY", "999999")
logging.disable(logging.WARNING)


def _drain_trade_log(pm):
    out = []
    tl = getattr(pm, "trade_log", None)
    if tl is None:
        return out
    while len(tl) > 0:
        out.append(tl.popleft())
    return out


def _entry_atr_pct(env, asset, open_step):
    """Recover atr_pct at the trade's entry index on the 5m chunk.

    Uses the SAME index math as multi_asset_chunked_env._future_zone_contribution:
      entry_idx = step_in_chunk - (current_step - open_step)
    Returns (atr_pct, df, entry_idx, tf) or (None, None, None, None).
    """
    try:
        df, tf = env._get_chunk_df_for_asset(asset, preferred_tf="5m")
        if df is None or len(df) == 0:
            return None, None, None, None
        cur_local = int(getattr(env, "step_in_chunk", 0))
        cur_global = int(getattr(env, "current_step", 0))
        entry_idx = cur_local - (cur_global - int(open_step))
        if entry_idx < 0 or entry_idx >= len(df):
            return None, df, None, str(tf or "5m")
        atr = None
        for col in ("atr_pct", "ATR_pct", "atr_pct_14", "atr_14"):
            if col in df.columns:
                v = float(df[col].iloc[entry_idx])
                # atr_pct stored as ratio (0.01 = 1%); guard against absurd values
                atr = v
                break
        return atr, df, entry_idx, str(tf or "5m")
    except Exception:
        return None, None, None, None


def run_forensic(ckpt_path: str, steps: int, split: str) -> dict:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from adan_trading_bot.common.config_loader import ConfigLoader
    from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader
    from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
    from adan_trading_bot.future_arena.future_zones import (
        ZoneConfig, classify_zone, compute_mfe_mae, PivotDirection,
    )

    cfg = ConfigLoader.load_config(str(REPO_ROOT / "config" / "config.yaml"))
    cfg.setdefault("environment", {})["rich_display_interval"] = 999999

    wc = copy.deepcopy(cfg.get("workers", {}).get("w1", {}))
    wc.update({
        "worker_id": 0,
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

    model = PPO.load(ckpt_path, device="cpu")
    model.set_env(vec_env)

    obs = vec_env.reset()
    underlying = vec_env.envs[0]
    pm = underlying.portfolio_manager
    _drain_trade_log(pm)

    # Zone config — same defaults the env uses (ZoneConfig defaults).
    zcfg = getattr(underlying, "_future_zone_cfg", None) or ZoneConfig()
    horizon = int(getattr(zcfg, "horizon", 36))
    mae_floor = float(getattr(zcfg, "mae_floor", 0.0015))

    trades = []   # one dict per closed trade
    n_episodes = 0

    for s in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _r, dones, _infos = vec_env.step(action)

        for receipt in _drain_trade_log(pm):
            if not isinstance(receipt, dict):
                continue
            if receipt.get("action") != "close" and "pnl_pct" not in receipt:
                continue
            asset = receipt.get("asset", "BTCUSDT")
            open_step = int(receipt.get("open_step", -1))
            cur_global = int(getattr(underlying, "current_step", 0))
            hold = max(0, cur_global - open_step) if open_step >= 0 else None

            sl_pct = float(receipt.get("stop_loss_pct", 0.0) or 0.0)  # ratio
            tp_pct = float(receipt.get("take_profit_pct", 0.0) or 0.0)
            pnl_pct = receipt.get("pnl_pct")
            reason = str(receipt.get("close_reason", receipt.get("reason", "UNKNOWN")))
            tf = str(receipt.get("timeframe", "") or "")

            atr_pct, df, entry_idx, df_tf = _entry_atr_pct(underlying, asset, open_step)

            # Zone via ex-post MFE/MAE on the 5m chunk (env's own math).
            zone = None
            mfe = mae = None
            if df is not None and entry_idx is not None and 0 <= entry_idx < len(df):
                try:
                    mfe, mae = compute_mfe_mae(
                        df, entry_idx, PivotDirection.LOW, horizon, mae_floor=mae_floor
                    )
                    z, _q = classify_zone(mfe, mae, zcfg)
                    zone = z.value
                except Exception:
                    zone = None

            sl_over_atr = (sl_pct / atr_pct) if (atr_pct and atr_pct > 1e-9) else None
            tp_over_atr = (tp_pct / atr_pct) if (atr_pct and atr_pct > 1e-9) else None

            trades.append({
                "open_step": open_step,
                "hold": hold,
                "reason": reason,
                "tf": tf or df_tf or "",
                "sl_pct": round(sl_pct * 100, 4),
                "tp_pct": round(tp_pct * 100, 4),
                "atr_pct": round(atr_pct * 100, 4) if atr_pct is not None else None,
                "sl_over_atr": round(sl_over_atr, 3) if sl_over_atr is not None else None,
                "tp_over_atr": round(tp_over_atr, 3) if tp_over_atr is not None else None,
                "pnl_pct": round(float(pnl_pct), 4) if pnl_pct is not None else None,
                "zone": zone,
                "mfe": round(float(mfe), 5) if mfe is not None else None,
                "mae": round(float(mae), 5) if mae is not None else None,
            })

        done = bool(np.ravel(dones)[0])
        if done:
            n_episodes += 1
            obs = vec_env.reset()
            underlying = vec_env.envs[0]
            pm = underlying.portfolio_manager
            _drain_trade_log(pm)

    # ── illegal actions (read final rejection_reasons) ──
    rej = dict(getattr(underlying, "rejection_reasons", {}) or {})
    sell_no_pos = int(rej.get("sell_no_position", 0))
    min_notional = int(rej.get("min_notional", 0))
    total_rej = sum(int(v) for v in rej.values())
    # "illegal" = actions the policy requested that are structurally impossible
    illegal = sell_no_pos + min_notional
    illegal_ratio = illegal / steps if steps else 0.0

    return _aggregate(ckpt_path, split, steps, n_episodes, trades, rej,
                      illegal, illegal_ratio)


def _stats(vals):
    a = np.array([v for v in vals if v is not None], dtype=float)
    if a.size == 0:
        return {"n": 0}
    wins = a[a > 0]
    return {
        "n": int(a.size),
        "wr": round(float(len(wins) / a.size), 4),
        "expectancy": round(float(a.mean()), 4),
        "median": round(float(np.median(a)), 4),
        "best": round(float(a.max()), 4),
        "worst": round(float(a.min()), 4),
    }


def _aggregate(ckpt, split, steps, n_eps, trades, rej, illegal, illegal_ratio):
    ckpt_name = os.path.basename(ckpt)

    # ---- per ZONE ----
    by_zone = defaultdict(list)
    for t in trades:
        if t["zone"] is not None and t["pnl_pct"] is not None:
            by_zone[t["zone"]].append(t["pnl_pct"])
    zones = {z: _stats(by_zone.get(z, [])) for z in ("green", "orange", "red")}
    zone_covered = sum(len(v) for v in by_zone.values())

    # zone verdict: GREEN expectancy must beat ORANGE must beat RED
    ge = zones["green"].get("expectancy")
    oe = zones["orange"].get("expectancy")
    re = zones["red"].get("expectancy")
    if ge is not None and oe is not None:
        zone_ok = ge > oe and (re is None or oe >= re or oe > re)
        zone_verdict = "ZONES_WORK" if (ge > oe) else "ZONES_BROKEN (GREEN<=ORANGE)"
    else:
        zone_verdict = "INSUFFICIENT_ZONE_DATA"

    # ---- per REASON: hold buckets + pnl ----
    by_reason = defaultdict(list)
    for t in trades:
        by_reason[t["reason"]].append(t)
    reasons = {}
    for r, lst in by_reason.items():
        holds = [x["hold"] for x in lst if x["hold"] is not None]
        pnls = [x["pnl_pct"] for x in lst if x["pnl_pct"] is not None]
        le3 = sum(1 for h in holds if h <= 3)
        reasons[r] = {
            "n": len(lst),
            "pct_of_trades": round(len(lst) / max(1, len(trades)) * 100, 1),
            "hold_mean": round(float(np.mean(holds)), 1) if holds else None,
            "hold_median": round(float(np.median(holds)), 1) if holds else None,
            "pct_hold_le3": round(le3 / max(1, len(holds)) * 100, 1) if holds else None,
            "pnl_mean": round(float(np.mean(pnls)), 4) if pnls else None,
        }

    # ---- SL/ATR & TP/ATR (overall + for SL_HIT / TP_HIT) ----
    def atr_block(filt):
        sub = [t for t in trades if filt(t)]
        sl_a = [t["sl_over_atr"] for t in sub if t["sl_over_atr"] is not None]
        tp_a = [t["tp_over_atr"] for t in sub if t["tp_over_atr"] is not None]
        atr = [t["atr_pct"] for t in sub if t["atr_pct"] is not None]
        return {
            "n": len(sub),
            "n_with_atr": len(sl_a),
            "sl_over_atr_mean": round(float(np.mean(sl_a)), 3) if sl_a else None,
            "sl_over_atr_median": round(float(np.median(sl_a)), 3) if sl_a else None,
            "tp_over_atr_mean": round(float(np.mean(tp_a)), 3) if tp_a else None,
            "atr_pct_mean": round(float(np.mean(atr)), 4) if atr else None,
            "pct_sl_under_0p5_atr": round(
                sum(1 for x in sl_a if x < 0.5) / max(1, len(sl_a)) * 100, 1
            ) if sl_a else None,
        }

    def _is_sl(r):
        r = r.upper()
        return "SL" in r or "STOP_LOSS" in r or "STOPLOSS" in r

    def _is_tp(r):
        r = r.upper()
        return "TP" in r or "TAKE_PROFIT" in r or "TAKEPROFIT" in r

    atr_all = atr_block(lambda t: True)
    atr_slhit = atr_block(lambda t: _is_sl(t["reason"]))
    atr_tphit = atr_block(lambda t: _is_tp(t["reason"]))

    # ---- global PnL ----
    allp = _stats([t["pnl_pct"] for t in trades])

    return {
        "checkpoint": ckpt_name,
        "split": split,
        "steps_tested": steps,
        "episodes": n_eps,
        "n_trades": len(trades),
        "global_pnl": allp,
        "zones": {
            "verdict": zone_verdict,
            "covered_trades": zone_covered,
            "uncovered": len(trades) - zone_covered,
            "green": zones["green"],
            "orange": zones["orange"],
            "red": zones["red"],
        },
        "reasons": reasons,
        "sl_tp_atr": {
            "all": atr_all,
            "SL_HIT": atr_slhit,
            "TP_HIT": atr_tphit,
        },
        "illegal_actions": {
            "rejection_reasons": rej,
            "sell_no_position": int(rej.get("sell_no_position", 0)),
            "min_notional": int(rej.get("min_notional", 0)),
            "illegal_total": illegal,
            "illegal_ratio_per_step": round(illegal_ratio, 4),
        },
        # keep a sample of raw trades for the per-trade report (first 500)
        "trades_sample": trades[:500],
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--split", type=str, default="test", choices=["val", "test", "train"])
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    print(f"[forensic] ckpt={args.ckpt} split={args.split} steps={args.steps}",
          file=sys.stderr)
    res = run_forensic(args.ckpt, args.steps, args.split)

    out_dir = REPO_ROOT / "logs" / "validation" / "forensic"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else out_dir / f"forensic_{Path(args.ckpt).stem}.json"
    out_path.write_text(json.dumps(res, indent=2))
    print(f"[forensic] saved: {out_path}", file=sys.stderr)

    # compact stdout summary
    if "error" in res:
        print("ERROR:", res["error"])
        return 1
    z = res["zones"]
    print(f"\n=== {res['checkpoint']} | trades={res['n_trades']} eps={res['episodes']} ===")
    g = res["global_pnl"]
    print(f"GLOBAL: WR={g.get('wr')} exp={g.get('expectancy')} best={g.get('best')} worst={g.get('worst')}")
    print(f"ZONES [{z['verdict']}] covered={z['covered_trades']}/{res['n_trades']}")
    for zn in ("green", "orange", "red"):
        s = z[zn]
        print(f"  {zn:6s}: n={s.get('n')} WR={s.get('wr')} exp={s.get('expectancy')}")
    ia = res["illegal_actions"]
    print(f"ILLEGAL: sell_no_pos={ia['sell_no_position']} min_notional={ia['min_notional']} "
          f"ratio/step={ia['illegal_ratio_per_step']}")
    st = res["sl_tp_atr"]["SL_HIT"]
    print(f"SL_HIT ATR: n={st['n']} sl/atr_mean={st['sl_over_atr_mean']} "
          f"%sl<0.5ATR={st['pct_sl_under_0p5_atr']} atr%_mean={st['atr_pct_mean']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
