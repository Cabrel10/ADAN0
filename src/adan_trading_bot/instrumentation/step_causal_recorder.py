"""Step-level causal recorder for ADAN (ADAN0 chantier).

Writes ONE JSONL record per environment step, linking the PPO policy decision
to its economic consequences. Opt-in via the ``ADAN_STEP_CAUSAL_PATH``
environment variable (supports a ``{worker_id}`` placeholder, exactly like
``ADAN_PIPELINE_TRACE_PATH``). Zero cost when disabled.

Causal chain covered by each record::

    policy_decision_id -> raw_action -> discrete_action -> target_weight
    -> routing_result -> rejection_reason -> execution
    -> position_before/after -> capital_before/after
    -> realized/unrealized PnL -> fees -> reward_components -> reward

PPO-side fields (``action_log_prob``, ``value_estimate``, ``entropy``) are
owned by the SB3 rollout buffer, not by the env. They are left ``None`` here
and joined ex-post via ``policy_decision_id`` — never fabricated.

Telemetry must NEVER alter the trading trajectory: every write is wrapped in
a broad except so a broken sink cannot break a run.
"""

from __future__ import annotations

import io
import json
import os
import threading
from typing import Any, Dict, Optional, Tuple

# Exact schema imposed by the ADAN0 directive. Order is stable.
SCHEMA_FIELDS = (
    "step_id",
    "episode_id",
    "asset",
    "policy_decision_id",
    "raw_action",
    "action_log_prob",
    "value_estimate",
    "entropy",
    "discrete_action",
    "target_weight",
    "routing_result",
    "rejection_reason",
    "budget_before",
    "budget_after",
    "drawdown_state",
    "cooldown_state",
    "position_before",
    "position_after",
    "price",
    "capital_before",
    "capital_after",
    "realized_pnl",
    "unrealized_pnl",
    "fees",
    "reward",
    "reward_components",
)

# Canonical additive (non-aliased) reward component keys. The invariant
# ``raw_reward == sum(components)`` is checked against THIS set only — logger
# compatibility aliases (e.g. ``drawdown_penalty`` duplicated as ``drawdown``)
# are excluded by construction.
ADDITIVE_REWARD_KEYS = (
    "pnl_reward",
    "behavior_penalty",
    "cooldown_reentry_penalty",
    "action_anchor_penalty",
    "holding_cost",
    "smart_flat_reward",
    "time_decay_cost",
    "promotion_bonus",
    "demotion_penalty",
    "closure_bonus",
    "drawdown_penalty",
    "symmetry_penalty",
    "action_entropy_penalty",
    "future_contrib",
    "latent_pnl",
    "saturation_penalty",
)

# MTM override branch (ADAN_MTM_REWARD=1) replaces the crystallised-PnL raw
# reward with step-wise equity delta + friction + drawdown pressure.
ADDITIVE_REWARD_KEYS_MTM = (
    "mtm_delta_equity",
    "mtm_trade_cost",
    "behavior_penalty",
    "cooldown_reentry_penalty",
    "drawdown_penalty",
)


def check_reward_invariant(
    reward_total: float,
    components: Dict[str, Any],
    *,
    tol: float = 1e-6,
    mtm: bool = False,
) -> Tuple[bool, float, float]:
    """Per-transition reward invariant.

    Returns ``(ok, error, additive_sum)`` where
    ``error = abs(reward_total - additive_sum)``. This is a PER-TRANSITION
    check — comparing means is NOT accepted as proof.
    """
    keys = ADDITIVE_REWARD_KEYS_MTM if mtm else ADDITIVE_REWARD_KEYS
    additive_sum = 0.0
    for key in keys:
        try:
            additive_sum += float(components.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
    error = abs(float(reward_total) - additive_sum)
    return (error <= tol), error, additive_sum


class StepCausalRecorder:
    """Append-only JSONL writer for per-step causal records."""

    def __init__(self, path: str, *, enabled: bool = True):
        self.path = path
        self.enabled = bool(enabled and path)
        self._lock = threading.Lock()
        self._seq = 0
        if self.enabled:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)

    @classmethod
    def from_env(cls, worker_id: int = 0) -> "StepCausalRecorder":
        path = os.environ.get("ADAN_STEP_CAUSAL_PATH", "").strip()
        if path:
            path = path.format(worker_id=worker_id)
        return cls(path, enabled=bool(path))

    def record(self, **fields: Any) -> Optional[Dict[str, Any]]:
        """Write one record. Returns the record dict (or None when disabled).

        Missing fields default to ``None`` — a missing value is declared, never
        inferred. Extra fields are appended after the canonical schema keys.
        """
        if not self.enabled:
            return None
        try:
            with self._lock:
                self._seq += 1
                record: Dict[str, Any] = {
                    name: fields.get(name) for name in SCHEMA_FIELDS
                }
                for key, value in fields.items():
                    if key not in record:
                        record[key] = value
                record["record_seq"] = self._seq
                with io.open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(
                        json.dumps(record, separators=(",", ":"), default=str)
                        + "\n"
                    )
            return record
        except Exception:
            # Telemetry must never alter the trading trajectory.
            return None
