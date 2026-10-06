"""
risk_engine.py — ADAN-System-One · Étape 8 (Dimensionnement du capital)
==========================================================================

Le sizing est DÉTERMINISTE — jamais appris, jamais délégué au réseau.
Le jugement calibré (P(win), étape 4) alimente Kelly fractionnaire, borné
par des plafonds stricts :

    f_kelly   = p − (1−p)/b          (b = ratio gain/risque = TP_R = 3.5)
    f_quarter = f_kelly / 4          (Kelly fractionnaire — survie d'abord)
    f_final   = min(f_quarter, 20 %) (plafond d'exposition par trade)

    Size_USD  = Capital × f_final    (ramené au risque réel via sl_pct)

Garde-fous réglementaires/techniques :
  · exposition courante + nouveau trade ≤ 40 % du capital (anti-concentration)
  · pertes du jour > 5 % du capital → arrêt du jour (circuit breaker)
  · taille finale ≥ 15 USDT (minimum notional Binance Futures)
  · f_kelly ≤ 0 → pas de trade (l'edge ne justifie aucun capital)

Référence : workflow étape 8 + dev.md §ADAN-SYSTEM-ONE.
"""

from __future__ import annotations

from dataclasses import dataclass

# ─── Plafonds (workflow) ──────────────────────────────────────────────────────
MAX_EXPOSURE_PER_TRADE = 0.20   # 20 % du capital max par trade
MAX_EXPOSURE_TOTAL = 0.40       # 40 % d'exposition cumulée max
DAILY_LOSS_CIRCUIT = 0.05       # arrêt du jour si pertes > 5 % du capital
KELLY_FRACTION = 0.25           # quart de Kelly (robustesse)
MIN_NOTIONAL_USD = 15.0         # minimum Binance Futures


@dataclass(frozen=True)
class PositionSize:
    """Résultat du dimensionnement (ou refus motivé)."""
    ok: bool
    size_usd: float = 0.0
    risk_usd: float = 0.0         # perte maximale si SL touché
    f_kelly: float = 0.0
    f_applied: float = 0.0
    motif_refus: str = ""


@dataclass(frozen=True)
class MicroCapitalRegime:
    """Current explicit configuration, not the legacy quarter-Kelly/$15 regime."""
    initial_capital: float
    allocation_min: float
    allocation_max: float
    risk_per_trade: float
    max_positions: int
    configured_min_notional: float
    effective_min_notional: float
    config_sha256: str
    min_notional_policy: str = "max(configured_min, latest_user_requirement_11_USD)"


def load_micro_capital_regime(config_path="config/config.yaml", required_min_notional=11.0):
    import hashlib
    import math
    from pathlib import Path
    import yaml
    payload = Path(config_path).read_bytes()
    config = yaml.safe_load(payload)
    tier = config['capital_tiers'][0]
    if 'micro' not in tier['name'].lower():
        raise ValueError("First capital tier is not the configured Micro regime")
    minimum, maximum = map(float, tier['exposure_range'])
    risk = float(tier['risk_per_trade_pct'])
    initial = float(config['sandbox']['initial_capital'])
    exchange_min = float(config['trading_rules']['min_order_value_usdt'])
    if not all(math.isfinite(x) for x in (minimum, maximum, risk, initial, exchange_min, required_min_notional)):
        raise ValueError("Nonfinite micro-capital configuration")
    if not (0 < minimum <= maximum <= 100 and 0 < risk <= 100 and initial > 0 and exchange_min > 0 and required_min_notional > 0):
        raise ValueError("Invalid micro-capital percentages or capital")
    if tier['max_concurrent_positions'] != 1 or float(tier['leverage']) != 1.0:
        raise ValueError("Current micro contract requires one position, leverage=1")
    return MicroCapitalRegime(initial, minimum/100, maximum/100, risk/100, 1,
                              exchange_min, max(exchange_min, required_min_notional),
                              hashlib.sha256(payload).hexdigest())


def size_micro_position(regime, capital, sl_fraction, *, allocation_fraction=None,
                        positions_open=0, exposure_current=0., available_cash=None):
    """Bounded sizing for a hypothetical flat micro portfolio, not a fill model.

    Four percent is a MAXIMUM risk budget, not an obligation to risk exactly 4%.
    A cash/min-order/risk conflict means abstention; never fall back to 20% or
    fabricate a zero portfolio. Probability/EV admission is a separate gate.
    """
    import math
    allocation = (regime.allocation_min + regime.allocation_max)/2 if allocation_fraction is None else allocation_fraction
    cash = capital if available_cash is None else available_cash
    if not all(math.isfinite(x) for x in (capital, sl_fraction, allocation, cash, exposure_current)):
        return PositionSize(False, motif_refus="nonfinite micro portfolio")
    if capital <= 0 or not 0 < sl_fraction < 1 or cash < 0 or exposure_current < 0:
        return PositionSize(False, motif_refus="invalid capital/SL/cash/exposure")
    if positions_open != 0 or exposure_current != 0:
        return PositionSize(False, motif_refus="micro maximum one simultaneous position")
    if not regime.allocation_min <= allocation <= regime.allocation_max:
        return PositionSize(False, motif_refus="allocation outside configured micro range")
    notional = min(capital*allocation, capital*regime.risk_per_trade/sl_fraction, cash)
    if notional < capital*regime.allocation_min:
        return PositionSize(False, motif_refus="risk/cash incompatible with micro minimum allocation")
    if notional < regime.effective_min_notional:
        return PositionSize(False, motif_refus="below effective conservative minimum order")
    return PositionSize(True, size_usd=notional, risk_usd=notional*sl_fraction,
                        f_applied=notional/capital)


def size_position(
    capital: float,
    p_win: float,
    tp_r: float,
    sl_distance_pct: float,
    exposure_current: float = 0.0,
    daily_pnl: float = 0.0,
) -> PositionSize:
    """Calcule la taille en USD d'un trade validé par le gate (étape 5-7).

    capital          : capital total disponible (USD)
    p_win            : probabilité calibrée de succès (JEV)
    tp_r             : ratio gain/risque de la géométrie (b de Kelly)
    sl_distance_pct  : distance du SL en % du prix (1R)
    exposure_current : exposition actuelle en USD (positions ouvertes)
    daily_pnl        : PnL réalisé du jour en USD (négatif = pertes)
    """
    if capital <= 0:
        return PositionSize(False, motif_refus="capital nul")

    # Circuit breaker journalier : on protège le capital avant tout
    if daily_pnl < -DAILY_LOSS_CIRCUIT * capital:
        return PositionSize(
            False,
            motif_refus=(f"circuit breaker journalier : pertes {daily_pnl:.0f}$ "
                         f"< −{DAILY_LOSS_CIRCUIT*100:.0f}% du capital"),
        )

    # Kelly fractionnaire sur la probabilité calibrée
    b = tp_r
    f_kelly = p_win - (1.0 - p_win) / b if b > 0 else 0.0
    if f_kelly <= 0:
        return PositionSize(
            False, f_kelly=f_kelly,
            motif_refus=f"Kelly ≤ 0 (p={p_win:.2f}, b={b}) — l'edge ne justifie aucun capital",
        )
    f_applied = min(f_kelly * KELLY_FRACTION, MAX_EXPOSURE_PER_TRADE)

    size_usd = capital * f_applied

    # Plafond d'exposition totale (anti-concentration)
    if exposure_current + size_usd > MAX_EXPOSURE_TOTAL * capital:
        allowed = MAX_EXPOSURE_TOTAL * capital - exposure_current
        if allowed <= 0:
            return PositionSize(
                False, f_kelly=f_kelly,
                motif_refus=(f"exposition totale au plafond "
                             f"({exposure_current/capital*100:.0f}% ≥ {MAX_EXPOSURE_TOTAL*100:.0f}%)"),
            )
        size_usd = allowed  # on réduit au reliquat plutôt que rejeter

    # Minimum notional Binance
    if size_usd < MIN_NOTIONAL_USD:
        return PositionSize(
            False, f_kelly=f_kelly, f_applied=f_applied,
            motif_refus=f"taille {size_usd:.2f}$ < minimum Binance {MIN_NOTIONAL_USD}$",
        )

    risk_usd = size_usd * sl_distance_pct  # perte si SL touché
    return PositionSize(
        True, size_usd=size_usd, risk_usd=risk_usd,
        f_kelly=f_kelly, f_applied=f_applied,
    )
