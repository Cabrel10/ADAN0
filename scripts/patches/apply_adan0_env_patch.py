#!/usr/bin/env python3
"""ADAN0 one-shot migration patch for multi_asset_chunked_env.py.

~11k-line file; the Edit tool fails on it. This patcher applies the ADAN0
changes via exact-anchor replacement, requiring each anchor to occur EXACTLY
ONCE before any write (fail-loud, no partial writes). Blocks A-H live in
adan0_env_blocks.py; blocks I-O are defined below.
"""
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adan0_env_blocks import BLOCKS

P = "src/adan_trading_bot/environment/multi_asset_chunked_env.py"

# ── I: DRAWDOWN_KILL -> economic reset (single authority) ─────────────
BLOCKS.append((
    "I_drawdown_economic_reset",
    """            except Exception as _e:
                self.logger.warning(f"[DRAWDOWN_KILL] Force-close failed: {_e}")
            return True
        return False""",
    """            except Exception as _e:
                self.logger.warning(f"[DRAWDOWN_KILL] Force-close failed: {_e}")
            # -- ADAN0: ECONOMIC RESET (single authority) ------------------
            # The ONLY path that restores capital to initial_capital. Updates
            # portfolio_lifetime_id / portfolio_reset_count / cumulative_pnl /
            # time_since_drawdown_reset inside PortfolioManager.economic_reset.
            try:
                self.portfolio_manager.economic_reset(reason="DRAWDOWN_KILL")
            except Exception as _er:
                self.logger.error(f"[DRAWDOWN_KILL] economic_reset failed: {_er}")
            # decision_budget restored to max ONLY here (never at episode reset)
            try:
                if hasattr(self, "decision_budget_max"):
                    self.decision_budget = float(self.decision_budget_max)
            except Exception:
                pass
            # Drawdown cooldown in the never-reset global_step frame so it
            # survives episode boundaries by construction.
            try:
                self._drawdown_cooldown_until_step = int(getattr(self, "global_step", 0)) + int(
                    getattr(self, "drawdown_cooldown_steps", 50)
                )
            except Exception:
                self._drawdown_cooldown_until_step = 0
            try:
                _pm = self.portfolio_manager
                self.logger.warning(
                    f"[DRAWDOWN_KILL_ECONOMIC_RESET] lifetime={_pm.portfolio_lifetime_id} "
                    f"resets={_pm.portfolio_reset_count} "
                    f"cum_pnl={_pm.cumulative_pnl:+.2f} global_peak={_pm.global_peak:.2f}"
                )
            except Exception:
                pass
            return True
        return False""",
))

# ── J: _build_observation -> push drawdown-persistence slots to PM ────
BLOCKS.append((
    "J_obs_push",
    """            try:
                self.portfolio._pending_decision_budget = float(
                    getattr(self, "decision_budget", 1.0))
                self.portfolio._pending_decision_budget_max = float(
                    getattr(self, "decision_budget_max", 1.0))
            except Exception:
                pass""",
    """            try:
                self.portfolio._pending_decision_budget = float(
                    getattr(self, "decision_budget", 1.0))
                self.portfolio._pending_decision_budget_max = float(
                    getattr(self, "decision_budget_max", 1.0))
            except Exception:
                pass
            # ADAN0-OBS: drawdown persistence slots [28-31]. Every constraint
            # that survives episode boundaries MUST be observable, else the
            # decoupling silently creates a POMDP.
            try:
                _lt = getattr(self, "_locked_tier", None) or {}
                _mdd = float(_lt.get("max_drawdown_pct", 40.0)) / 100.0
                self.portfolio._pending_max_dd_frac = max(_mdd, 1e-8)
                _until = int(getattr(self, "_drawdown_cooldown_until_step", 0))
                _rem = max(0, _until - int(getattr(self, "global_step", 0)))
                _tot = max(1, int(getattr(self, "drawdown_cooldown_steps", 50)))
                self.portfolio._pending_drawdown_cooldown = float(min(1.0, _rem / _tot))
            except Exception:
                pass""",
))

# ── K: routing guard init ──────────────────────────────────────────────
BLOCKS.append((
    "K_route_guard",
    """            if discrete_action == 0:
                _effective_thr = abs(float(_sell_thr)) if (_in_pos_route and _sell_thr is not None) else abs(float(action_threshold))""",
    """            _route_stage, _route_reason = None, None
            if discrete_action == 0:
                _effective_thr = abs(float(_sell_thr)) if (_in_pos_route and _sell_thr is not None) else abs(float(action_threshold))""",
))

# ── L: ventilate routing counters by exact reason ─────────────────────
BLOCKS.append((
    "L_ventilation",
    """                self._trace_action_pipeline(
                    _route_stage, asset, main_decision, 0, _route_reason,
                    in_position=_in_pos_route, slot_available=_slot_available,
                    threshold=_effective_thr,
                )""",
    """                # ADAN0: ventilate routing rejections by exact reason.
                try:
                    _apc = self.action_pipeline_counts
                    if _route_stage == "deadband_reject":
                        _apc["routing_reject_deadband"] += 1
                    elif _route_stage == "portfolio_reject":
                        _apc["routing_reject_position"] += 1
                    elif _route_stage == "routing_reject":
                        _apc["routing_reject_position"] += 1
                        _rk = f"routing_reject_{_route_reason}"
                        if _rk in _apc:
                            _apc[_rk] += 1
                        else:
                            _apc["routing_reject_other"] += 1
                    else:
                        _apc["routing_reject_other"] += 1
                except Exception:
                    pass
                self._trace_action_pipeline(
                    _route_stage, asset, main_decision, 0, _route_reason,
                    in_position=_in_pos_route, slot_available=_slot_available,
                    threshold=_effective_thr,
                )""",
))

# ── M: step-causal decision snapshot (first asset) ────────────────────
BLOCKS.append((
    "M_decision_snapshot",
    """            if i == 0:
                # Preserve policy intent separately from route and execution.
                self._last_raw_action0 = float(main_decision)
                self._last_route_action = int(discrete_action)
                first_discrete_action_requested = discrete_action  # routed, before gates
                first_discrete_action = discrete_action""",
    """            if i == 0:
                # Preserve policy intent separately from route and execution.
                self._last_raw_action0 = float(main_decision)
                self._last_route_action = int(discrete_action)
                first_discrete_action_requested = discrete_action  # routed, before gates
                first_discrete_action = discrete_action
                # ADAN0: step-causal decision snapshot (first asset = primary).
                self._last_step_decision = {
                    "asset": str(asset),
                    "raw_action": float(main_decision),
                    "discrete_action": int(discrete_action),
                    "routing_result": (_route_reason if discrete_action == 0 else "routed"),
                    "target_weight": None,
                    "position_before": bool(_in_pos_route),
                }""",
))

# ── N: capture target_weight for the causal record ────────────────────
BLOCKS.append((
    "N_target_weight",
    """            normalized_size = (size_raw + 1.0) / 2.0  # 0..1
            normalized_size = max(0.0, min(1.0, normalized_size))
            target_exposure_pct = min_exp + normalized_size * (max_exp - min_exp)""",
    """            normalized_size = (size_raw + 1.0) / 2.0  # 0..1
            normalized_size = max(0.0, min(1.0, normalized_size))
            target_exposure_pct = min_exp + normalized_size * (max_exp - min_exp)
            if i == 0:
                try:
                    self._last_step_decision["target_weight"] = float(target_exposure_pct)
                except Exception:
                    pass""",
))

# ── O: step() tail -> write the Step Causal Record ────────────────────
BLOCKS.append((
    "O_step_record",
    """            # Nettoyage explicite des observations precedentes
            if hasattr(self, 'last_observation') and self.last_observation is not None:
                del self.last_observation""",
    """            # ADAN0: Step Causal Record — one JSONL row linking the policy
            # decision of THIS step to its economic consequences. Opt-in,
            # telemetry never alters the trading trajectory.
            try:
                _rec = getattr(self, "_step_causal_recorder", None)
                if _rec is not None and getattr(_rec, "enabled", False):
                    _pm = self.portfolio_manager
                    _dec = getattr(self, "_last_step_decision", {}) or {}
                    _receipts = getattr(self, "_step_closed_receipts", []) or []
                    _fees = 0.0
                    _realized = 0.0
                    for _r in _receipts:
                        if isinstance(_r, dict):
                            _fees += float(_r.get("fees", 0.0) or 0.0)
                            _realized += float(_r.get("pnl_net", _r.get("pnl", 0.0)) or 0.0)
                    _rej_after = getattr(self, "rejection_reasons", {}) or {}
                    _rej_before = getattr(self, "_rejection_snapshot_before", {}) or {}
                    _rej_fired = [
                        k for k, v in _rej_after.items()
                        if int(v) > int(_rej_before.get(k, 0))
                    ]
                    _asset0 = _dec.get("asset") or (
                        str(self.assets[0]) if getattr(self, "assets", None) else None
                    )
                    _price = None
                    try:
                        _price = float(self._get_current_prices().get(_asset0))
                    except Exception:
                        _price = None
                    _cap_after = float(getattr(_pm, "equity", 0.0))
                    _cash_after = float(getattr(_pm, "cash", 0.0))
                    _ic = max(float(getattr(_pm, "initial_capital", 20.5)), 1e-9)
                    _rec.record(
                        step_id=int(getattr(self, "global_step", 0)),
                        episode_id=int(getattr(self, "episode_count", 0)),
                        asset=_asset0,
                        policy_decision_id=(
                            f"w{self.worker_id}:e{int(getattr(self, 'episode_count', 0))}"
                            f":s{int(getattr(self, 'global_step', 0))}"
                        ),
                        raw_action=_dec.get("raw_action"),
                        action_log_prob=None,  # owned by SB3 rollout buffer
                        value_estimate=None,   # owned by SB3 rollout buffer
                        entropy=None,          # owned by SB3 rollout buffer
                        discrete_action=_dec.get("discrete_action"),
                        target_weight=_dec.get("target_weight"),
                        routing_result=_dec.get("routing_result"),
                        rejection_reason=(_rej_fired[0] if _rej_fired else None),
                        budget_before=getattr(self, "_budget_before_step", None),
                        budget_after=float(getattr(self, "decision_budget", 0.0)),
                        drawdown_state={
                            "equity": _cap_after,
                            "drawdown_ratio": max(0.0, (_ic - _cap_after) / _ic),
                            "cooldown_until_step": int(getattr(self, "_drawdown_cooldown_until_step", 0)),
                            "lifetime_id": int(getattr(_pm, "portfolio_lifetime_id", 0)),
                            "reset_count": int(getattr(_pm, "portfolio_reset_count", 0)),
                        },
                        cooldown_state={
                            "ratio": float(self._cooldown_remaining_ratio()),
                            "agent_close_last_step": int(getattr(self, "_last_agent_close_step", -10**9)),
                        },
                        position_before=_dec.get("position_before"),
                        position_after=any(
                            getattr(p, "is_open", False)
                            for p in getattr(_pm, "positions", {}).values()
                        ),
                        price=_price,
                        capital_before=getattr(self, "_capital_before_step", None),
                        capital_after=_cap_after,
                        realized_pnl=float(_realized),
                        unrealized_pnl=float(_cap_after - _cash_after),
                        fees=float(_fees),
                        reward=float(reward),
                        reward_components=dict(getattr(self, "_last_reward_components", {}) or {}),
                    )
            except Exception:
                pass

            # Nettoyage explicite des observations precedentes
            if hasattr(self, 'last_observation') and self.last_observation is not None:
                del self.last_observation""",
))


def main() -> int:
    s = io.open(P, encoding="utf-8").read()
    for name, old, _new in BLOCKS:
        n = s.count(old)
        if n != 1:
            print(f"FAIL {name}: anchor count={n} (expected 1)")
            return 1
    for name, old, new in BLOCKS:
        s = s.replace(old, new, 1)
        print(f"OK   {name}")
    io.open(P, "w", encoding="utf-8").write(s)
    print("PATCH APPLIED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
