#!/usr/bin/env python3
"""ADAN0 migration blocks for multi_asset_chunked_env.py (data module).

Imported by apply_adan0_env_patch.py. Each block = (name, old, new); every
anchor must occur EXACTLY ONCE before any write happens.
"""

BLOCKS = []

BLOCKS.append((
    "A_const_32",
    """# Standard portfolio state vector dimension
DEFAULT_PORTFOLIO_STATE_SIZE = 28""",
    """# Standard portfolio state vector dimension
# 32 = 20 base + 8 ACM Capability Vector + 4 ADAN0 drawdown-persistence slots
DEFAULT_PORTFOLIO_STATE_SIZE = 32""",
))

BLOCKS.append((
    "B_counters",
    """            "portfolio_reject": 0,
            "trade_executed": 0,
        }""",
    """            "portfolio_reject": 0,
            "trade_executed": 0,
            # ADAN0: routing_reject ventilated by exact reason (real code names)
            "routing_reject_deadband": 0,
            "routing_reject_drawdown": 0,
            "routing_reject_budget": 0,
            "routing_reject_position": 0,
            "routing_reject_cooldown": 0,
            "routing_reject_other": 0,
            "routing_reject_sell_while_flat": 0,
            "routing_reject_buy_while_long": 0,
        }""",
))

BLOCKS.append((
    "C_recorder_init",
    """        if self._action_pipeline_trace_path:
            os.makedirs(os.path.dirname(self._action_pipeline_trace_path) or ".", exist_ok=True)""",
    """        if self._action_pipeline_trace_path:
            os.makedirs(os.path.dirname(self._action_pipeline_trace_path) or ".", exist_ok=True)
        # ADAN0: Step Causal Recorder (opt-in via ADAN_STEP_CAUSAL_PATH,
        # zero cost when disabled). One JSONL record per env step linking the
        # policy decision to its economic consequences.
        try:
            from ..instrumentation.step_causal_recorder import StepCausalRecorder
            self._step_causal_recorder = StepCausalRecorder.from_env(self.worker_id)
        except Exception:
            self._step_causal_recorder = None
        # ADAN0: drawdown cooldown state (global_step frame -> survives
        # episode boundaries by construction, never rebased).
        self._drawdown_cooldown_until_step: int = 0
        self.drawdown_cooldown_steps: int = int(
            self.config.get("risk_parameters", {}).get("drawdown_cooldown_steps", 50)
        )""",
))

BLOCKS.append((
    "D_reset_rebase",
    """        self.current_step = 0
        self.done = False
        self.episode_reward = 0.0""",
    """        # ADAN0 DECOUPLING: cooldown trackers are keyed on current_step, which
        # is about to restart at 0. To let cooldowns SURVIVE the episode
        # boundary we rebase every absolute timestamp into the new frame
        # (v - final_step). Remaining cooldown time is preserved exactly.
        _prev_final_step = int(getattr(self, "current_step", 0))
        if _prev_final_step > 0:
            try:
                for _d_name in ("_last_sell_step_by_asset", "_last_open_step_by_asset"):
                    _d = getattr(self, _d_name, None)
                    if isinstance(_d, dict):
                        for _k in list(_d.keys()):
                            _d[_k] = int(_d[_k]) - _prev_final_step
                # _buy_step_by_asset: positions are all closed at the boundary
                # (finalize_open_positions above) -> tracker is historical only,
                # cleared by _cooldown_remaining_ratio while FLAT.
                if isinstance(getattr(self, "_buy_step_by_asset", None), dict):
                    self._buy_step_by_asset.clear()
                _lac = int(getattr(self, "_last_agent_close_step", -10**9))
                if _lac > -10**8:
                    self._last_agent_close_step = _lac - _prev_final_step
            except Exception:
                pass

        self.current_step = 0
        self.done = False
        self.episode_reward = 0.0""",
))

BLOCKS.append((
    "E_budget_no_reset",
    """        # DECISION BUDGET (V3) — reset jauge + compteurs AGENT_CLOSE a chaque
        # nouvel episode (sinon l'etat fuit d'un episode a l'autre).
        self.agent_close_count_today = 0
        self.agent_close_consecutive = 0
        self._last_agent_close_step = -10**9
        if hasattr(self, "decision_budget_max"):
            self.decision_budget = float(self.decision_budget_max)""",
    """        # DECISION BUDGET (V3) — ADAN0 DECOUPLING: the budget is NOT restored
        # at episode boundaries anymore. It returns to max ONLY via the
        # DRAWDOWN_KILL economic reset (single authority). Daily AGENT_CLOSE
        # quota counters stay per-episode by design; the min-gap tracker was
        # rebased above so the cooldown survives the boundary.
        self.agent_close_count_today = 0
        self.agent_close_consecutive = 0""",
))

BLOCKS.append((
    "F_cooldown_dicts_survive",
    """        # OMEGA-3/4: Reset episode receipts and per-asset cooldown
        self._all_episode_receipts = deque(maxlen=50)  # REDUCED: Cap memory: keep last 50 receipts (was 500)
        self._last_open_step_by_asset = {}
        self._last_sell_step_by_asset = {}   # start of post-SELL re-entry friction
        self._buy_step_by_asset = {}          # hard HOLD_MIN post-BUY guard
        self._sell_cooldown_total_by_asset = {}  # friction normalization window""",
    """        # OMEGA-3/4: Reset episode receipts. ADAN0 DECOUPLING: the per-asset
        # cooldown trackers (_last_sell_step_by_asset, _last_open_step_by_asset,
        # _sell_cooldown_total_by_asset) were REBASED above and SURVIVE the
        # episode boundary — re-entry friction is an economic constraint, not
        # an episodic one. No dict reset here anymore.
        self._all_episode_receipts = deque(maxlen=50)  # REDUCED: Cap memory: keep last 50 receipts (was 500)""",
))

BLOCKS.append((
    "G_step_head_capture",
    """        self._step_closed_receipts = []
        self._step_invalid_penalty = 0.0  # Reset per-step penalty
        self._step_realized_pnl = 0.0""",
    """        self._step_closed_receipts = []
        self._step_invalid_penalty = 0.0  # Reset per-step penalty
        self._step_realized_pnl = 0.0
        # ADAN0: step-causal capture — pre-execution economic state.
        self._budget_before_step = float(getattr(self, "decision_budget", 1.0))
        try:
            self._capital_before_step = float(self.portfolio_manager.equity)
        except Exception:
            self._capital_before_step = None
        self._rejection_snapshot_before = dict(getattr(self, "rejection_reasons", {}))
        self._last_step_decision = {}""",
))

BLOCKS.append((
    "H_warmup_global",
    """        warmup = getattr(self, 'warmup_steps', getattr(self, 'warmup_period', 50))
        if getattr(self, 'current_step', 0) < warmup:
            return False""",
    """        warmup = getattr(self, 'warmup_steps', getattr(self, 'warmup_period', 50))
        # ADAN0 DECOUPLING: warmup is GLOBAL (once per run), not per-episode —
        # the portfolio now survives episodes, so a per-episode warmup would
        # blind the kill-switch for 50 steps after every boundary.
        if getattr(self, 'global_step', 0) < warmup:
            return False""",
))
