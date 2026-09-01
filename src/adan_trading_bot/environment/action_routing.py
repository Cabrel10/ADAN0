"""State-conditioned action routing for ADAN0 v12.

SINGLE SOURCE OF TRUTH for how the continuous ``action[0]`` axis is mapped to a
discrete trade decision (0=HOLD, 1=BUY/OPEN, 2=SELL/CLOSE), as a function of the
per-asset portfolio state.

Rationale (docs/ARCHITECTURE_ACTION_ROUTING_v12.md)
--------------------------------------------------
A SPOT portfolio slot has only TWO meaningful states:

  * FLAT (no open position on this asset)  -> can only OPEN (BUY) or HOLD.
    SELL is structurally meaningless (nothing to close).
  * LONG (position open on this asset)     -> can only CLOSE (SELL) or HOLD.
    BUY is structurally meaningless (already exposed).

The previous symmetric decode (BUY if a0>thr, SELL if a0<-thr, else HOLD) made
the agent spend most of its training experiencing *illegal* actions (SELL-while-
flat, BUY-while-open). Those illegal samples polluted the policy gradient and
drove the entropy collapse to always-BUY (V10 @70k, V11 @78k).

This module removes the ambiguity STRUCTURALLY, without any penalty:

  * FLAT : a0 > +threshold -> BUY ; otherwise (even a0 = -1.0) -> HOLD (neutral).
  * LONG : a0 < -threshold -> SELL ; otherwise (even a0 = +1.0) -> HOLD (neutral).
  * Slot beyond the tier's ``max_concurrent_positions`` and currently flat
    -> NOOP (forced HOLD, neutral).

The agent therefore learns a genuine state machine and NEVER samples a
direction-illegal action. There is no reward penalty for the "wrong sign" — the
negative half of the axis simply means "I don't want to enter" when flat, and
the positive half means "I hold my position" when long.

This scales natively from tier 1 (Micro, 1 slot -> pure binary) to tier 5
(Enterprise, 5 slots -> one independent routing per asset slot). It does NOT
touch the other 4 action dims (Size, Timeframe, StopLoss, TakeProfit) which are
driven by the Future Arena oracle and must remain intact.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

# Discrete action codes (kept identical to the legacy convention used across
# the env / execution engine / paper-trading monitor).
HOLD = 0
BUY = 1
SELL = 2


def route_action_by_state(
    a0: float,
    in_position: bool,
    slot_available: bool = True,
    threshold: float = 0.10,
    sell_threshold: float | None = None,
) -> int:
    """Map the continuous ``action[0]`` to a discrete decision by state.

    Parameters
    ----------
    a0 : float
        The raw first action dimension in [-1, 1] for this asset slot.
    in_position : bool
        True if an open position exists on this asset slot (LONG state).
    slot_available : bool, default True
        True if the tier allows opening a new position (n_open < max_concurrent).
        When False AND flat, the slot is beyond quota -> forced NOOP/HOLD.
    threshold : float, default 0.10
        Absolute action threshold (config-driven, not hard-coded upstream).

    Returns
    -------
    int
        0 = HOLD (neutral, no penalty), 1 = BUY/OPEN, 2 = SELL/CLOSE.

    Notes
    -----
    * FLAT + slot beyond quota -> HOLD (cannot open, nothing to close).
    * FLAT + slot available    -> BUY iff a0 > +threshold, else HOLD.
    * LONG                      -> SELL iff a0 < -threshold, else HOLD.
      (A LONG slot is always "manageable" regardless of slot_available: you can
      always close an existing position.)
    """
    a0 = float(a0)
    thr = abs(float(threshold))
    # FIX-D (ASYMMETRIC THRESHOLD, measured root cause 2026-07-05):
    # Diag archfix proved reqSELL (8%) << a0<0*open% (31.6%): a large mass of
    # negative a0 emitted WHILE LONG falls in the dead-zone |a0|<=thr and is
    # routed to HOLD -> the agent's exit intent is silently swallowed -> it
    # learns "SELL rarely fires" -> stops trying -> BUY runaway. Design fix:
    # ENTRY is a commitment (fees, risk) -> require conviction (buy thr).
    # EXIT is protection -> should be EASY (smaller sell thr). Backward
    # compatible: sell_threshold=None -> symmetric legacy behaviour.
    sthr = abs(float(sell_threshold)) if sell_threshold is not None else thr

    if in_position:
        # LONG state: only CLOSE or HOLD_POS are legal.
        if a0 < -sthr:
            return SELL
        return HOLD

    # FLAT state.
    if not slot_available:
        # Slot beyond the tier's concurrent-position quota -> cannot open.
        return HOLD
    if a0 > thr:
        return BUY
    return HOLD


def post_sell_reentry_penalty(
    *,
    current_step: int,
    last_sell_step: int,
    wait_steps: int,
    total_cooldown_steps: int | None,
    round_trip_fees: float,
    pnl_reward_scale: float = 0.5,
) -> tuple[float, float]:
    """Price an early re-entry without masking the policy's BUY.

    Returns ``(penalty, proximity)`` where proximity decreases linearly from
    one immediately after a close to zero when the post-SELL window expires.
    ``last_sell_step`` may intentionally point into the future after a stop
    loss; ``total_cooldown_steps`` then normalizes the doubled SL window.

    The maximum raw-reward penalty equals one real round trip expressed on the
    same scale as ``pnl_base_reward``: fee fraction × 100 percentage points ×
    the PnL reward coefficient (currently 0.5).  The environment therefore
    lets the action execute and lets PPO learn whether rapid re-entry was worth
    paying for, rather than silently replacing BUY with HOLD.
    """
    wait = max(0, int(wait_steps))
    if wait == 0:
        return 0.0, 0.0

    remaining = wait - (int(current_step) - int(last_sell_step))
    total = max(wait, int(total_cooldown_steps or wait), 1)
    proximity = min(1.0, max(0.0, remaining / float(total)))
    max_penalty = max(0.0, float(round_trip_fees)) * 100.0 * max(
        0.0, float(pnl_reward_scale)
    )
    return -max_penalty * proximity, proximity


def economic_round_trip_fees(
    config: Mapping[str, Any] | None,
    *,
    commission_pct: float = 0.002,
) -> float:
    """Return the round-trip fee contract used by economic decisions.

    ``reward_shaping.future_reward.round_trip_fees`` is the project's financial
    source of truth.  Falling back to twice the per-side commission keeps
    minimal/test configurations coherent.  This function intentionally does
    not inspect ``ADAN_FREE_SLTP``: that flag only releases SL/TP geometry and
    must never disable profitability checks.
    """
    configured: Any = None
    if isinstance(config, Mapping):
        reward_cfg = config.get("reward_shaping", {})
        if isinstance(reward_cfg, Mapping):
            future_cfg = reward_cfg.get("future_reward", {})
            if isinstance(future_cfg, Mapping):
                configured = future_cfg.get("round_trip_fees")

    try:
        configured_fees = float(configured)
    except (TypeError, ValueError):
        configured_fees = math.nan
    if math.isfinite(configured_fees) and configured_fees >= 0.0:
        return configured_fees

    try:
        per_side = float(commission_pct)
    except (TypeError, ValueError):
        per_side = 0.002
    if not math.isfinite(per_side):
        per_side = 0.002
    return 2.0 * max(0.0, per_side)


def minimum_profitable_win_probability(
    *,
    stop_loss_pct: float,
    take_profit_pct: float,
    round_trip_fees: float,
) -> float:
    """Return the break-even win probability after a complete fee round trip.

    For a win worth ``TP - fees`` and a loss worth ``SL + fees``, positive EV
    requires ``p > (SL + fees) / (SL + TP)``.  The fee argument is therefore
    the complete entry-plus-exit contract, not a one-sided commission.
    """
    sl = float(stop_loss_pct)
    tp = float(take_profit_pct)
    fees = max(0.0, float(round_trip_fees))
    if sl <= 0.0 or tp <= 0.0:
        return 0.99
    return (sl + fees) / (sl + tp)


def resolve_ev_fee_gate(
    *,
    p_hmm: float,
    p_min_required: float,
    disabled: bool,
) -> tuple[bool, str]:
    """Resolve the optional EV fee gate without hiding a bypass.

    The environment flag is intended for controlled diagnostics only. When it
    is active, a negative gate comparison remains observable as advisory
    telemetry but cannot suppress the policy's BUY action.
    """
    if float(p_hmm) > float(p_min_required):
        return False, "accepted"
    if disabled:
        return False, "disabled_advisory"
    return True, "negative_ev_fee_gate"


def resolve_agent_close_gate(
    *,
    exit_authority: bool,
    budget_blocked: bool,
    below_break_even: bool,
) -> tuple[bool, str]:
    """Resolve policy-close gates without disabling anti-churn controls.

    Returns ``(blocked, reason)``. The decision budget, daily quota, and minimum
    close gap are a hard structural gate. ``exit_authority`` only bypasses the
    profitability barrier *after* that gate is satisfied, allowing an eligible
    policy close to cut risk instead of forcing a losing position to remain
    open. Independent safety exits (SL/TP/MaxDuration) do not use this gate.
    """
    if budget_blocked:
        return True, "decision_budget_or_quota"
    if exit_authority:
        return False, "exit_authority"
    if below_break_even:
        return True, "below_break_even_barrier"
    return False, "accepted"


def describe_route(a0: float, in_position: bool, slot_available: bool = True,
                   threshold: float = 0.10) -> str:
    """Human-readable label for logging/telemetry (never used in control flow)."""
    code = route_action_by_state(a0, in_position, slot_available, threshold)
    state = "LONG" if in_position else ("FLAT" if slot_available else "FLAT/NOQUOTA")
    verb = {HOLD: "HOLD", BUY: "OPEN", SELL: "CLOSE"}[code]
    return f"{state}->{verb}(a0={a0:+.3f},thr={threshold:.2f})"
