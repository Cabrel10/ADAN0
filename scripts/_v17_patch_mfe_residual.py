#!/usr/bin/env python3
"""V17-Fix B: feed `mfe_residual` to the Arena (RewardBridge.contribution).

Problem: mfe_residual (Maximum Favorable Excursion AFTER the agent's exit) was
never computed, so reward_service.lost_potential_penalty (the user's "Scenario B"
= "you sold too early, the market kept rising, here is an opportunity-cost malus")
stayed inactive.

Fix: compute the exit index (entry_idx + steps_held) and run compute_mfe_mae over
the horizon AFTER the exit, then pass mfe_residual into bridge.contribution(...).
This is OFFLINE (future known only during training) and NEVER injected into the
observation -> live-safe (Arena 1 design).

Edit tool FAILS on this 498KB file -> use Python io string-replace. Idempotent.
"""
import io, sys

ENV = "src/adan_trading_bot/environment/multi_asset_chunked_env.py"

# Anchor: the exact contribution(...) call block (entry MFE/MAE already computed
# just above via compute_mfe_mae at entry_idx).
OLD = '''                # duree en steps
                steps_held = max(0, cur_global - open_step)
                contrib = bridge.contribution(
                    profile=profile,
                    timeframe=str(tf or "5m"),
                    closed=True,
                    pnl_gross=float(receipt.get("pnl_gross", receipt.get("pnl", 0.0)) or 0.0),
                    steps_held=int(steps_held),
                    close_reason=str(receipt.get("reason", receipt.get("close_reason", "")) or ""),
                    direction=1.0,  # SPOT long
                    size=float(receipt.get("size", 0.0) or 0.0),
                    sl_chosen=float(receipt.get("stop_loss_pct", 0.0) or 0.0),
                    tp_chosen=float(receipt.get("take_profit_pct", 0.0) or 0.0),
                    mfe=float(mfe),
                    mae=float(mae),
                )'''

NEW = '''                # duree en steps
                steps_held = max(0, cur_global - open_step)
                # V17-Fix B: mfe_residual = MFE over the horizon AFTER the exit
                # index. Feeds Arena.lost_potential_penalty (Scenario B: "sold too
                # early, market kept rising -> opportunity-cost malus"). OFFLINE
                # only; never injected into the observation -> live-safe.
                mfe_residual = None
                try:
                    exit_idx = entry_idx + int(steps_held)
                    if 0 <= exit_idx < len(df):
                        _mfe_r, _ = compute_mfe_mae(
                            df, exit_idx, PivotDirection.LOW, horizon, mae_floor=mae_floor
                        )
                        mfe_residual = float(_mfe_r)
                except Exception:
                    mfe_residual = None
                contrib = bridge.contribution(
                    profile=profile,
                    timeframe=str(tf or "5m"),
                    closed=True,
                    pnl_gross=float(receipt.get("pnl_gross", receipt.get("pnl", 0.0)) or 0.0),
                    steps_held=int(steps_held),
                    close_reason=str(receipt.get("reason", receipt.get("close_reason", "")) or ""),
                    direction=1.0,  # SPOT long
                    size=float(receipt.get("size", 0.0) or 0.0),
                    sl_chosen=float(receipt.get("stop_loss_pct", 0.0) or 0.0),
                    tp_chosen=float(receipt.get("take_profit_pct", 0.0) or 0.0),
                    mfe=float(mfe),
                    mae=float(mae),
                    mfe_residual=mfe_residual,
                )'''

def main():
    with io.open(ENV, "r", encoding="utf-8") as f:
        src = f.read()
    if "V17-Fix B: mfe_residual" in src:
        print("PATCH_B_ALREADY_PRESENT")
        return 0
    if OLD not in src:
        print("PATCH_B_ANCHOR_NOT_FOUND")
        return 2
    src = src.replace(OLD, NEW, 1)
    with io.open(ENV, "w", encoding="utf-8") as f:
        f.write(src)
    print("PATCH_B_APPLIED")
    return 0

if __name__ == "__main__":
    sys.exit(main())
