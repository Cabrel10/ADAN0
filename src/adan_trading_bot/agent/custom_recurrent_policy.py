#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom recurrent policy for PPO using CNN feature extractor + LSTM (via SB3-Contrib).

- Expects observation space as Dict with keys:
  - "observation": Box(shape=(3, 20, 15), dtype=float32)
  - "portfolio_state": Box(shape=(17,), dtype=float32)

- The CNN processes the market observation [C=3, H=20, W=15].
- A small MLP processes the portfolio_state (17-dim).
- Features are concatenated and projected to `features_dim`.
- LSTM layers are handled by RecurrentActorCriticPolicy in sb3_contrib.

Dependencies:
- stable-baselines3 >= 2.0.0
- sb3-contrib (for RecurrentActorCriticPolicy)
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

from adan_trading_bot.agent.feature_extractors import TemporalFusionExtractor

try:
    # sb3-contrib provides the recurrent policies
    from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
except Exception as e:  # pragma: no cover
    raise ImportError(
        "sb3-contrib is required for CustomRecurrentPolicy. Install with: pip install sb3-contrib"
    ) from e

class CustomRecurrentPolicy(RecurrentActorCriticPolicy):
    """Recurrent Actor-Critic policy using TemporalFusionExtractor as features extractor."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule,
        features_dim: int = 128,  # Doit correspondre à la sortie de TemporalFusionExtractor
        lstm_hidden_size: int = 256,
        n_lstm_layers: int = 1,
        **kwargs: Any,
    ) -> None:
        # S'assurer que notre nouvel extracteur est utilisé
        fe_kwargs = kwargs.pop("features_extractor_kwargs", None) or {"features_dim": features_dim}
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            features_extractor_class=TemporalFusionExtractor,
            features_extractor_kwargs=fe_kwargs,
            lstm_hidden_size=lstm_hidden_size,
            n_lstm_layers=n_lstm_layers,
            **kwargs,
        )


