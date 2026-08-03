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
