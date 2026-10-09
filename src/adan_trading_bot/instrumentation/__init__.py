"""ADAN0 instrumentation package.

Causal observability for the RL trading loop:

- :class:`StepCausalRecorder` — one JSONL record per env step linking the
  policy decision to its economic consequences (routing, rejection, execution,
  PnL, fees, reward components).
- :class:`PPORadarCallback` — SB3 callback writing one CSV row per PPO update,
  explicitly linked to the rollout window that produced it.
- :func:`check_reward_invariant` — per-transition verification that
  ``reward_total == sum(reward_components)`` within numeric tolerance.
"""

from .step_causal_recorder import (
    StepCausalRecorder,
    check_reward_invariant,
    SCHEMA_FIELDS,
)

__all__ = [
    "StepCausalRecorder",
    "check_reward_invariant",
    "SCHEMA_FIELDS",
]
