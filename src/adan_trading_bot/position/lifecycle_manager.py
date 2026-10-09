"""
lifecycle_manager.py — ADAN-System-One · Étape 10 (Vigie active de position)
=============================================================================

Dans ADAN0, une fois l'ordre envoyé, le trade était abandonné à un SL et un
TP **aveugles**. Ici, le cerveau reste allumé tant que la position est ouverte :
à chaque nouvelle clôture 5m, l'état S_t est réévalué et le gestionnaire agit.

Objectif économique (workflow étape 10) : transformer une perte pleine
−1.0R en −0.2R/−0.3R par sortie anticipée, et verrouiller les gains
(break-even) sans couper les gagnants.

Règles (dans l'ordre de priorité à chaque clôture 5m) :
  1. TP atteint                → CLOSE (gain verrouillé)
  2. SL atteint                → CLOSE (invalidation structurelle touchée)
  3. Anomalie critique (JEV)   → CLOSE au marché immédiatement
  4. Invalidation de thèse     → CLOSE au marché (sortie anticipée ~−0.3R) :
       après ≥ 6 barres (30 min), le cours clôture au-delà du plus extrême
       des 3 dernières barres CONTRE la position avec momentum adverse
       (close sous le plus bas des 3 dernières pour un LONG / au-dessus du
       plus haut pour un SHORT) ET la progression reste négative.
  5. Break-even                → dès +1.5R, le SL remonte à entry + frais
       (hors gap/slippage et coûts réalisés variables)
  6. Trailing                  → dès +2.0R, le SL suit le prix à 1.0 × ATR_1h
  7. Time-stop                 → au-delà de MAX_HOLD barres, CLOSE au marché

Le gestionnaire ne crée jamais de position — il ne fait que SURVEILLER,
PROTÉGER et SORTIR une position ouverte par la chaîne gate/geometry/risk.

Référence : workflow étape 10 + dev.md §ADAN-SYSTEM-ONE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

# ─── Seuils de gestion (workflow étape 10) ────────────────────────────────────
BREAKEVEN_TRIGGER_R = 1.5     # passage au break-even dès +1.5R
TRAILING_TRIGGER_R = 2.0      # trailing actif dès +2.0R
TRAILING_ATR_MULT = 1.0       # distance de trailing = 1.0 × ATR_1h
INVALIDATION_MIN_BARS = 6     # invalidation évaluée à partir de 6 barres (30 min)
INVALIDATION_LOOKBACK = 3     # extrême des 3 dernières barres
P_ANOMALY_CRITICAL = 0.70     # anomalie critique → sortie immédiate au marché
MAX_HOLD_BARS = 288           # time-stop : 24 h (288 × 5m)


@dataclass
class OpenPosition:
    """Position ouverte suivie par la vigie."""
    direction: str              # LONG inventory under SPOT
    entry: float
    stop_loss: float
    take_profit: float
    size_usd: float
    entry_index: int            # indice de la barre d'entrée (timeline 5m)
    fees_rt: Optional[float] = None  # expected RT costs from central SPOT contract
    atr_1h_pct: float = 0.012   # ATR contenant 1h en % (pour le trailing)
    # état courant (mis à jour à chaque barre)
    bars_held: int = 0
    mfe_r: float = 0.0          # meilleure excursion favorable (en R)
    mae_r: float = 0.0          # pire excursion adverse (en R)
    breakeven_done: bool = False
    trailing_done: bool = False


@dataclass(frozen=True)
class VigilAction:
    """Action décidée par la vigie à la clôture d'une 5m."""
    action: str                 # 'HOLD' | 'CLOSE' | 'MOVE_SL'
    reason: str
    new_stop: Optional[float] = None
    r_courant: float = 0.0
    exit_price: Optional[float] = None


class LifecycleManager:
    """Surveille une position ouverte à chaque clôture 5m (étape 10)."""

    def __init__(self, position: OpenPosition, market_contract=None):
        from adan_trading_bot.policy.market_contract import load_market_contract
        market = market_contract or load_market_contract()
        market.require_direction(position.direction)   # SPOT: only LONG positions can exist; exits are SELL_EXIT
        self.market_contract = market
        if position.fees_rt is None:
            position.fees_rt = market.cost_rt
        if not np.isfinite(position.fees_rt) or not np.isclose(position.fees_rt,market.cost_rt,rtol=0,atol=1e-15):
            raise ValueError("Position expected costs differ from central SPOT contract")
        self.pos = position
        self.risk = abs(position.entry - position.stop_loss)  # 1R en prix
        if self.risk <= 0:
            raise ValueError("stop_loss doit différer de l'entry")
        self._recent_lows: List[float] = []   # 3 dernières barres (invalidation)
        self._recent_highs: List[float] = []

    # ── utilitaires ──────────────────────────────────────────────────────────
    def _r(self, price: float) -> float:
        """PnL latent exprimé en R (1R = distance entry→SL initial)."""
        if self.pos.direction == "SHORT":
            return (self.pos.entry - price) / self.risk
        return (price - self.pos.entry) / self.risk

    def _push_recent(self, h: float, l: float):
        """Empile la barre dans la fenêtre d'invalidation (3 dernières)."""
        self._recent_lows.append(l); self._recent_highs.append(h)
        if len(self._recent_lows) > INVALIDATION_LOOKBACK:
            self._recent_lows.pop(0); self._recent_highs.pop(0)

    def _update_excursions(self, high: float, low: float):
        best = self._r(low) if self.pos.direction == "SHORT" else self._r(high)
        worst = self._r(high) if self.pos.direction == "SHORT" else self._r(low)
        self.pos.mfe_r = max(self.pos.mfe_r, best)
        self.pos.mae_r = min(self.pos.mae_r, worst)

    # ── décision à chaque clôture 5m ─────────────────────────────────────────
    def on_bar(self, o: float, h: float, l: float, c: float,
               judgment=None) -> VigilAction:
        """Réévalue la position à la clôture de la 5m (o/h/l/c).

        judgment : Judgment calibré optionnel (étape 4) pour l'anomalie critique.
        Retourne VigilAction (HOLD / MOVE_SL / CLOSE + motif).
        """
        p = self.pos
        self.market_contract.require_direction(p.direction)
        p.bars_held += 1
        self._update_excursions(h, l)

        # suivi des barres (pour l'invalidation de thèse) — la barre courante
        # est ajoutée APRÈS l'évaluation : la fenêtre d'invalidation compare la
        # clôture courante aux extrêmes des barres PRÉCÉDENTES uniquement.
        r_close = self._r(c)

        # Same conservative OHLC convention as the plan oracle: SL first on
        # an ambiguous bar; a gap through the stop fills at the worse open.
        if l <= p.stop_loss:
            exit_price = min(p.stop_loss, o)
            self._push_recent(h, l)
            return VigilAction("CLOSE", "stop-loss touché (SELL_EXIT)",
                               r_courant=self._r(exit_price), exit_price=exit_price)
        if h >= p.take_profit:
            self._push_recent(h, l)
            return VigilAction("CLOSE", "take-profit atteint (SELL_EXIT)",
                               r_courant=self._r(p.take_profit), exit_price=p.take_profit)

        # ── 3. Anomalie critique détectée par le JEV → sortie au marché ──────
        if judgment is not None:
            p_anom = judgment.noul.get("anomalie_donnees", 0.0)
            if p_anom > P_ANOMALY_CRITICAL:
                self._push_recent(h, l)
                return VigilAction("CLOSE",
                                   f"anomalie critique (P={p_anom:.2f}) — sortie au marché",
                                   r_courant=r_close, exit_price=c)

        # ── 4. Invalidation de thèse (sortie anticipée ~−0.3R) ───────────────
        # La fenêtre contient les INVALIDATION_LOOKBACK barres PRÉCÉDENTES
        # (la courante n'y est pas encore — voir ajout en fin de on_bar).
        if p.bars_held >= INVALIDATION_MIN_BARS and r_close < 0:
            if len(self._recent_lows) == INVALIDATION_LOOKBACK:
                extr_low = min(self._recent_lows)
                extr_high = max(self._recent_highs)
                if p.direction == "SHORT" and c > extr_high:
                    # le cours clôture au-dessus du plus haut récent contre le SHORT
                    self._push_recent(h, l)
                    return VigilAction("CLOSE",
                                       f"invalidation thèse SHORT (close {c:.2f} > plus haut "
                                       f"{extr_high:.2f}, {r_close:+.2f}R) — coupe anticipée",
                                       r_courant=r_close, exit_price=c)
                if p.direction == "LONG" and c < extr_low:
                    self._push_recent(h, l)
                    return VigilAction("CLOSE",
                                       f"invalidation thèse LONG (close {c:.2f} < plus bas "
                                       f"{extr_low:.2f}, {r_close:+.2f}R) — coupe anticipée",
                                       r_courant=r_close, exit_price=c)

        # ── 5. Break-even : dès +1.5R, SL → entry + frais attendus (pas une garantie face aux gaps) ──
        if not p.breakeven_done and p.mfe_r >= BREAKEVEN_TRIGGER_R - 1e-9:
            fees_price = p.entry * p.fees_rt
            if p.direction == "SHORT":
                new_stop = p.entry - fees_price   # sous l'entrée pour un SHORT
                if new_stop < p.stop_loss:
                    p.stop_loss = new_stop
                    p.breakeven_done = True
                    self._push_recent(h, l)
                    return VigilAction("MOVE_SL", "break-even : SL → entry − frais",
                                       new_stop=new_stop, r_courant=r_close)
            else:
                new_stop = p.entry + fees_price
                if new_stop > p.stop_loss:
                    p.stop_loss = new_stop
                    p.breakeven_done = True
                    self._push_recent(h, l)
                    return VigilAction("MOVE_SL", "break-even : SL → entry + frais",
                                       new_stop=new_stop, r_courant=r_close)

        # ── 6. Trailing : dès +2.0R, SL suit le prix à 1.0 × ATR_1h ──────────
        if p.mfe_r >= TRAILING_TRIGGER_R - 1e-9:
            trail_dist = p.entry * p.atr_1h_pct * TRAILING_ATR_MULT
            if p.direction == "SHORT":
                new_stop = min(p.stop_loss, (p.entry - p.mfe_r * self.risk) + trail_dist)
                if new_stop < p.stop_loss:
                    p.stop_loss = new_stop
                    p.trailing_done = True
                    self._push_recent(h, l)
                    return VigilAction("MOVE_SL", "trailing ATR (SHORT)",
                                       new_stop=new_stop, r_courant=r_close)
            else:
                new_stop = max(p.stop_loss, (p.entry + p.mfe_r * self.risk) - trail_dist)
                if new_stop > p.stop_loss:
                    p.stop_loss = new_stop
                    p.trailing_done = True
                    self._push_recent(h, l)
                    return VigilAction("MOVE_SL", "trailing ATR (LONG)",
                                       new_stop=new_stop, r_courant=r_close)

        # ── 7. Time-stop ──────────────────────────────────────────────────────
        if p.bars_held >= MAX_HOLD_BARS:
            self._push_recent(h, l)
            return VigilAction("CLOSE", f"time-stop ({p.bars_held} barres)",
                               r_courant=r_close, exit_price=c)

        # aucune règle déclenchée → la barre courante entre dans la fenêtre
        # d'invalidation pour les barres suivantes, puis on maintient.
        self._push_recent(h, l)
        return VigilAction("HOLD", f"thèse intacte ({r_close:+.2f}R, "
                                   f"MFE {p.mfe_r:+.2f}R, MAE {p.mae_r:+.2f}R)",
                           r_courant=r_close)
