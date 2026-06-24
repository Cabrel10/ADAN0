"""RewardBridge — pont ADDITIF entre l'environnement et ``reward_service``.

Rôle
----
L'environnement monolithe (`multi_asset_chunked_env`) calcule déjà sa récompense
(`raw_reward`) : PnL réalisé, barrière AGENT_CLOSE (A5), pénalités diverses, puis
``symlog``. Le ``RewardBridge`` ajoute UN TERME OPTIONNEL guidé par le futur,
**sans rien remplacer** :

    raw_reward += bridge.contribution(...)        # une seule ligne dans l'env

Modes
-----
* ``classic`` (DÉFAUT)  → retourne **0.0** immédiatement. Aucun changement de
  comportement : l'env se comporte exactement comme avant. Sert de baseline A/B.
* ``future_guided``     → construit un :class:`TradeOutcome` depuis l'état env,
  appelle :meth:`RewardService.compute`, et retourne **uniquement** la part
  guidée par le futur (``future_contrib``, déjà PLAFONNÉE à 0.60 par le service).
  Les termes ``pnl_net`` / ``agent_close`` / ``temporal`` ne sont PAS renvoyés :
  l'env les calcule déjà → on évite tout double comptage.
* ``stochastic_hybrid`` → idem ``future_guided`` mais le service peut détecter un
  effondrement de la tête TP (``tp_head_entropy < 0.1``) et l'annoter.

Garanties
---------
* En ``classic`` : zéro allocation, zéro effet de bord.
* Ne lève JAMAIS : toute exception interne renvoie 0.0 (l'entraînement ne doit
  jamais crasher à cause du pont). L'erreur est mémorisée dans ``last_error``.
* Le PnL net reste le roi : la contribution est bornée par ``max_future_contrib``.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .reward_service import (
    RewardConfig,
    RewardMode,
    RewardService,
    TradeOutcome,
    RewardBreakdown,
    ROUND_TRIP_FEES_DEFAULT,
)

__all__ = ["RewardBridge"]

_MODE_MAP: Dict[str, RewardMode] = {
    "classic": RewardMode.CLASSIC,
    "future_guided": RewardMode.FUTURE_GUIDED,
    "stochastic_hybrid": RewardMode.STOCHASTIC_HYBRID,
}


class RewardBridge:
    """Adaptateur fin ENV ↔ ``RewardService`` (additif, désactivable)."""

    def __init__(
        self,
        config: Optional[RewardConfig] = None,
        seed: Optional[int] = None,
        enabled: bool = False,
    ) -> None:
        self.config = config or RewardConfig(mode=RewardMode.CLASSIC)
        self.enabled = bool(enabled)
        self.seed = seed
        self._service = RewardService(self.config, seed=seed)
        self._last_breakdown: Optional[RewardBreakdown] = None
        self.last_error: Optional[str] = None
        # compteurs de diagnostic (visibles dans les logs sans casser l'entraînement)
        self.n_calls = 0
        self.n_active = 0

    # ── construction depuis la config YAML ────────────────────────────────────
    @classmethod
    def from_config(
        cls, full_config: Optional[Dict[str, Any]], seed: Optional[int] = None
    ) -> "RewardBridge":
        """Construit le pont depuis le bloc ``reward_shaping.future_reward``.

        Bloc attendu (tout est optionnel, valeurs sûres par défaut) ::

            reward_shaping:
              future_reward:
                enabled: false          # le pont est-il actif ?
                mode: classic           # classic | future_guided | stochastic_hybrid
                round_trip_fees: 0.008  # sinon 2×commission, sinon 0.008
                max_future_contrib: 0.60

        Si le bloc est absent → pont DÉSACTIVÉ en mode classic (no-op).
        """
        cfg = full_config or {}
        block = (
            (cfg.get("reward_shaping") or {}).get("future_reward")
            if isinstance(cfg.get("reward_shaping"), dict)
            else None
        ) or {}

        enabled = bool(block.get("enabled", False))
        mode_str = str(block.get("mode", "classic")).lower()
        mode = _MODE_MAP.get(mode_str, RewardMode.CLASSIC)

        # frais : explicite > 2×commission > défaut SPOT 0.80 %.
        rtf = block.get("round_trip_fees")
        if rtf is None:
            commission = None
            env_block = cfg.get("environment")
            if isinstance(env_block, dict):
                commission = env_block.get("commission")
            if commission is None:
                tr_block = cfg.get("trading_rules")
                if isinstance(tr_block, dict):
                    commission = tr_block.get("commission_pct")
            rtf = (2.0 * float(commission)) if commission is not None \
                else ROUND_TRIP_FEES_DEFAULT

        max_fc = float(block.get("max_future_contrib", 0.60))

        rcfg = RewardConfig(
            mode=mode,
            round_trip_fees=float(rtf),
            max_future_contrib=max_fc,
        )
        return cls(config=rcfg, seed=seed, enabled=enabled)

    # ── API principale (appelée par l'env) ────────────────────────────────────
    @property
    def is_noop(self) -> bool:
        """Vrai si le pont ne peut RIEN ajouter (désactivé ou mode classic)."""
        return (not self.enabled) or self.config.mode == RewardMode.CLASSIC

    def contribution(
        self,
        *,
        profile: str = "intraday",
        timeframe: str = "5m",
        closed: bool = False,
        pnl_gross: float = 0.0,
        steps_held: int = 0,
        close_reason: str = "",
        direction: float = 0.0,
        size: float = 0.0,
        sl_chosen: float = 0.0,
        tp_chosen: float = 0.0,
        mfe: Optional[float] = None,
        mae: Optional[float] = None,
        mfe_residual: Optional[float] = None,
        near_green: bool = False,
        nearest_green: Any = None,
        tp_head_entropy: Optional[float] = None,
    ) -> float:
        """Terme de récompense guidé par le futur à AJOUTER à ``raw_reward``.

        Retourne 0.0 si le pont est désactivé / en mode classic, ou en cas
        d'erreur (jamais d'exception propagée). Sinon ``future_contrib`` (borné).
        """
        self.n_calls += 1
        if self.is_noop:
            return 0.0
        try:
            ev = TradeOutcome(
                profile=str(profile),
                timeframe=str(timeframe),
                direction=float(direction),
                size=float(size),
                sl_chosen=float(sl_chosen),
                tp_chosen=float(tp_chosen),
                closed=bool(closed),
                pnl_gross=float(pnl_gross),
                steps_held=int(steps_held),
                close_reason=str(close_reason or ""),
                mfe=mfe,
                mae=mae,
                mfe_residual=mfe_residual,
                nearest_green=nearest_green,
                near_green=bool(near_green),
                tp_head_entropy=tp_head_entropy,
            )
            bd = self._service.compute(ev)
            self._last_breakdown = bd
            self.n_active += 1
            # On ne renvoie QUE la part futur (eqs/sl/tp/sizing/missed/lost_pot),
            # déjà plafonnée. pnl_net / agent_close / temporal sont déjà dans l'env.
            return float(bd.future_contrib)
        except Exception as exc:  # noqa: BLE001 — robustesse: jamais crasher l'env
            self.last_error = repr(exc)
            return 0.0

    # ── introspection / logging ───────────────────────────────────────────────
    def last_breakdown(self) -> Optional[RewardBreakdown]:
        return self._last_breakdown

    def snapshot(self) -> dict:
        return {
            "enabled": self.enabled,
            "mode": self.config.mode.value,
            "round_trip_fees": self.config.round_trip_fees,
            "max_future_contrib": self.config.max_future_contrib,
            "n_calls": self.n_calls,
            "n_active": self.n_active,
            "last_error": self.last_error,
            "service": self._service.snapshot(),
        }
