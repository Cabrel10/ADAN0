#!/usr/bin/env python3
"""Prove/refute that the EV fee gate is structurally infeasible as configured.

Chain of evidence established in this session
---------------------------------------------
1. Canonical Gate C (logs/validation/gate_c_run_20260904_225928.log): NO_GO
     requested_hold_rate 0.548 -> executed_hold_rate 0.960 (FAIL, <=0.80)
     action divergence 0.354 (FAIL, <0.05)
2. Ventilation (logs/validation/routing_ventilation_*.json), 500 steps,
   uniform-random policy, exact accounting:
     requested: BUY 222 | SELL 225 | HOLD 53
     rejection_reasons: fee_gate 201  <-- dominant, 40.2% of ALL steps
     routing_reject_sell_while_flat 217, buy_while_long 17, deadband 39
     trade_executed 18
   => ~90% of BUY intent is killed by ONE gate: fee_gate.

This script closes the loop on *why* fee_gate fires, using the gate's own
formula from environment/action_routing.py:

    p_min_required = (sl + round_trip_fees) / (sl + tp)

and the launcher runtime invariants recorded in the Gate C log:

    ADAN_SL_HI = 0.0235   ADAN_TP_HI = 0.0222   ADAN_TP_LO = 0.0135
    commission_per_side = 0.002  ->  round_trip_fees = 0.004

The gate blocks unless p_hmm > p_min_required. An untrained / uninformative
HMM sits near 0.5 by construction, so if p_min_required >> 0.5 the gate is not
"selective", it is *closed*: no BUY can ever pass, the agent can only learn
HOLD, and executed HOLD saturates at ~96% exactly as measured.

Note this is NOT a bug in the gate. The formula is the correct EV break-even
win-rate. The defect is upstream, in the SL/TP configuration: SL_HI (2.35%) is
WIDER than TP_LO (1.35%), i.e. reward/risk < 1, which mathematically demands a
very high win rate to be EV-positive. Gate E ("fees_to_mean_tp 0.224 PASS")
could not see this because it compares fees to TP only and never computes the
required win rate.
"""
from __future__ import annotations

import json
import time
from itertools import product
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

FEES_ROUND_TRIP = 0.004  # 0.002 per side, from the Gate C log
SL_HI, TP_HI, TP_LO = 0.0235, 0.0222, 0.0135
P_HMM_UNINFORMATIVE = 0.5


def p_min_required(sl: float, tp: float, fees: float = FEES_ROUND_TRIP) -> float:
    if sl <= 0.0 or tp <= 0.0:
        return 0.99
    return (sl + fees) / (sl + tp)


def main() -> None:
    configured = {}
    for name, (sl, tp) in {
        "sl_hi__tp_lo (worst configured pair)": (SL_HI, TP_LO),
        "sl_hi__tp_hi (best configured pair)": (SL_HI, TP_HI),
    }.items():
        pmin = p_min_required(sl, tp)
        configured[name] = {
            "sl": sl,
            "tp": tp,
            "rr_ratio": round(tp / sl, 4),
            "p_min_required": round(pmin, 4),
            "gate_open_for_uninformative_hmm": P_HMM_UNINFORMATIVE > pmin,
            "excess_win_rate_needed_over_50pct":
                round(pmin - P_HMM_UNINFORMATIVE, 4),
        }

    # What SL/TP would actually let a coin-flip signal through?
    grid = []
    for sl, tp in product([0.005, 0.008, 0.010, 0.0135, 0.0235],
                          [0.0135, 0.020, 0.0222, 0.030, 0.040]):
        pmin = p_min_required(sl, tp)
        grid.append({
            "sl": sl, "tp": tp, "rr_ratio": round(tp / sl, 3),
            "p_min_required": round(pmin, 4),
            "open_at_p_hmm_0.50": bool(P_HMM_UNINFORMATIVE > pmin),
            "open_at_p_hmm_0.55": bool(0.55 > pmin),
        })

    feasible = [g for g in grid if g["open_at_p_hmm_0.55"]]

    verdict = (
        "CONFIRMED_STRUCTURALLY_CLOSED"
        if not configured["sl_hi__tp_hi (best configured pair)"][
            "gate_open_for_uninformative_hmm"]
        else "REFUTED"
    )

    report = {
        "hypothesis": (
            "The EV fee gate is closed by construction under the configured "
            "SL/TP, so BUY intent cannot reach execution and executed HOLD "
            "saturates near 96% (Gate C FAIL)."
        ),
        "verdict": verdict,
        "formula": "p_min_required = (sl + round_trip_fees) / (sl + tp)",
        "round_trip_fees": FEES_ROUND_TRIP,
        "p_hmm_uninformative_baseline": P_HMM_UNINFORMATIVE,
        "configured_pairs": configured,
        "measured_corroboration": {
            "source": "logs/validation/routing_ventilation_20260904_232121.json",
            "steps": 500,
            "requested_buy": 222,
            "fee_gate_rejections": 201,
            "share_of_buy_intent_killed_by_fee_gate": round(201 / 222, 4),
            "trade_executed": 18,
            "executed_hold_rate_gate_c": 0.960,
            "requested_hold_rate_gate_c": 0.548,
        },
        "why_gate_E_missed_it": (
            "Gate E compares round_trip_fees to mean TP (0.224 < 0.3 PASS). It "
            "never computes the break-even win rate, so an SL wider than TP "
            "passes E while making the EV gate unsatisfiable."
        ),
        "feasible_sl_tp_pairs_at_p_hmm_0.55": feasible,
        "recommended_minimal_change": (
            "Do NOT bypass the gate (ADAN_DISABLE_EV_FEE_GATE only hides it as "
            "advisory telemetry). Fix the economics: require TP > SL so RR > 1. "
            "With fees=0.004, sl=0.0135 and tp=0.0222 gives "
            f"p_min={p_min_required(0.0135, 0.0222):.4f}, still above 0.5; "
            f"sl=0.010/tp=0.030 gives {p_min_required(0.010, 0.030):.4f}, "
            "which an informative signal can clear."
        ),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    out = REPO_ROOT / "logs" / "validation"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"fee_gate_feasibility_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\n[WROTE] {path}")


if __name__ == "__main__":
    main()
