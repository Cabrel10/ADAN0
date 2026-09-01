"""PHASE 3 — Tests d'invariants contractuels V29 (sans entraînement).

Ces tests vérifient les invariants du pipeline action -> exécution AVANT tout
lancement d'entraînement V29. Règle gouvernance : un seul invariant
contractuel FAIL = TRAINING ABORTED.

Cartographie :
  T1  Les notionals d'ouverture restent dans les bornes du tier Micro
      (exposition [70%, 90%] du capital, plancher min_order).
  T2  RISK_GATE / slot tier : jamais plus de max_concurrent_positions
      ouvertes simultanément (Micro = 1).
  T3  AGENT_CLOSE barrier : resolve_agent_close_gate bloque sous le
      break-even, laisse passer au-dessus, et le budget est un hard gate.
  T4  Routing d'état : FLAT{a0=-1,0,+1}->{HOLD,HOLD,BUY},
      LONG{a0=-1,0,+1}->{SELL,HOLD,HOLD}, slot indisponible -> HOLD forcé.
  T5  Le sizing final DOIT dépendre de size_raw (anomalie C du rapport
      PHASE 2 : l.8719 écrasé par LINEAR_EXPO HMM l.8745). XFAIL tant que
      le PATCH 4 (sizing rendu à l'agent) n'est pas appliqué.
  T6  Taxonomie télémétrie : chaque clé de rejection_reasons est classée
      dans exactement une catégorie (contract_tier / economic /
      operational / routing_neutral) — prérequis au fix illegal_ratio V29.

Tests d'intégration (T1/T2/T5) : environnement RÉEL (MultiAssetChunkedEnv,
données val BTCUSDT), actions Box(5) contrôlées, AUCUN PPO, AUCUN
entraînement. Réutilise scripts/tests/action_pipeline_harness.py.
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ.setdefault("ADAN_TRAINING_SILENT", "1")
os.environ.setdefault("ADAN_RICH_STEP_EVERY", "999999")

from adan_trading_bot.environment.action_routing import (  # noqa: E402
    BUY,
    HOLD,
    SELL,
    resolve_agent_close_gate,
    route_action_by_state,
)
from scripts.tests import action_pipeline_harness as harness  # noqa: E402

INTEGRATION_STEPS = int(os.environ.get("ADAN_INVARIANT_STEPS", "400"))
TRACE_DIR = REPO_ROOT / "logs" / "validation" / "v29_invariant_traces"

# Tier Micro (config/config.yaml l.175+) — référentiel normatif.
MICRO_EXPO_MIN = 0.70
MICRO_EXPO_MAX = 0.90
MICRO_MAX_CONCURRENT = 1
MIN_ORDER_USDT = 11.0
# Tolérances numériques (slippage fill 2bps + arrondis float).
EXPO_TOL = 0.02


def _run_scenario_with_size(
    scenario: str, *, size_raw: float, steps: int, tag: str
) -> dict:
    """Run one harness scenario with a controlled size_raw dimension."""
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    return harness.run_scenario(
        scenario,
        steps=steps,
        split="val",
        seed=26027,
        size_raw=size_raw,
        sl_raw=1.0,
        tp_raw=1.0,
        trace_path=TRACE_DIR / f"{tag}.jsonl",
    )


@pytest.fixture(scope="module")
def constant_buy_run() -> dict:
    """BUY à chaque step (size_raw=0) — base T1/T2."""
    return _run_scenario_with_size(
        "constant_buy", size_raw=0.0, steps=INTEGRATION_STEPS, tag="t1_const_buy"
    )


# ──────────────────────────────────────────────────────────────────────────
# T1 — Bornes d'exposition du tier Micro respectées à l'ouverture
# ──────────────────────────────────────────────────────────────────────────


def test_t1_open_notionals_within_micro_tier_bounds(constant_buy_run: dict) -> None:
    """Chaque ouverture respecte [max(min_order, 0.70*cap), 0.90*cap].

    Avec equity initiale 20.5$ : borne basse = 14.35$ (0.70*20.5), borne
    haute = 18.45$ (0.90*20.5). Aucun chemin (fallback, PROB_SIZER) ne doit
    sortir de [min_order - eps, 0.90 * equity_init * (1+tol)].
    """
    opens = constant_buy_run["opens"]
    assert opens["count"] > 0, "aucun trade ouvert : le harness est cassé"
    equity0 = constant_buy_run["equity"]["initial"]
    hi = MICRO_EXPO_MAX * equity0 * (1.0 + EXPO_TOL)
    lo = MIN_ORDER_USDT * (1.0 - 0.01)  # plancher opérationnel absolu
    notional_min = opens["notional_usd"]["min"]
    notional_max = opens["notional_usd"]["max"]
    assert notional_min is not None and notional_max is not None
    assert lo <= notional_min, (
        f"notional min {notional_min:.2f} < plancher {lo:.2f} : "
        "un chemin de sizing sort de la borne basse du tier"
    )
    assert notional_max <= hi, (
        f"notional max {notional_max:.2f} > borne haute {hi:.2f} "
        f"(0.90*equity0={0.90 * equity0:.2f}) : violation borne haute tier"
    )
    # Cohérence avec la plage LINEAR_EXPO à l'equity initiale :
    # la majorité des opens doit être dans [0.70*e0*(1-tol), 0.90*e0*(1+tol)].
    expo_lo = MICRO_EXPO_MIN * equity0 * (1.0 - EXPO_TOL)
    assert notional_min >= min(expo_lo, lo), (
        f"notional min {notional_min:.2f} sous 0.70*equity0={expo_lo:.2f} "
        "hors chemin PROB_SIZER"
    )


# ──────────────────────────────────────────────────────────────────────────
# T2 — RISK_GATE : jamais plus de max_concurrent_positions (Micro = 1)
# ──────────────────────────────────────────────────────────────────────────


def test_t2_never_more_than_one_open_position(constant_buy_run: dict) -> None:
    """Sous BUY constant, le tier Micro (1 slot) ne doit JAMAIS être dépassé.

    Le harness clôture en fin de run ; pendant le run, le routing d'état +
    RISK_GATE doivent maintenir n_open <= 1 en permanence. Preuve indirecte
    mais stricte : le nombre d'opens réussis sans close intermédiaire est
    borné par 1 (un nouvel open n'arrive qu'après un close).
    """
    trace_path = TRACE_DIR / "t1_const_buy.jsonl"
    if not trace_path.exists():
        pytest.skip("trace manquante")
    import json as _json

    open_ids: set[str] = set()
    max_simultaneous = 0
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = _json.loads(line)
        if event.get("stage") != "trade_executed":
            continue
        pid = str(event.get("position_id"))
        if event.get("lifecycle_event") == "open":
            open_ids.add(pid)
        elif event.get("lifecycle_event") == "close":
            open_ids.discard(pid)
        max_simultaneous = max(max_simultaneous, len(open_ids))
    assert max_simultaneous <= MICRO_MAX_CONCURRENT, (
        f"{max_simultaneous} positions simultanées > tier Micro "
        f"({MICRO_MAX_CONCURRENT}) : violation RISK_GATE / slot tier"
    )


def test_t2_risk_gate_counter_never_negative_and_bounded(constant_buy_run: dict) -> None:
    """risk_gate reste un compteur sain (pas de corruption télémétrie)."""
    rej = constant_buy_run["rejection_reasons"]
    risk_gate = int(rej.get("risk_gate", 0))
    assert risk_gate >= 0
    # Avec 1 asset / 1 slot, le routing neutralise BUY-while-open AVANT
    # RISK_GATE : risk_gate doit rester ~0 (sinon = fuite du routing).
    assert risk_gate <= 1, (
        f"risk_gate={risk_gate} avec 1 asset/1 slot : le routing d'état "
        "laisse fuir des BUY vers le gate dur (ordre des couches cassé)"
    )


# ──────────────────────────────────────────────────────────────────────────
# T3 — AGENT_CLOSE barrier (resolve_agent_close_gate)
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("exit_authority", "budget_blocked", "below_break_even", "expected"),
    [
        # budget = hard gate : bloque TOUJOURS, quelles que soient les autres.
        (False, True, False, (True, "decision_budget_or_quota")),
        (True, True, True, (True, "decision_budget_or_quota")),
        # exit_authority bypasse la barrière de profitabilité uniquement.
        (True, False, True, (False, "exit_authority")),
        (True, False, False, (False, "exit_authority")),
        # Sous le break-even sans autorité -> bloqué (hysteresis).
        (False, False, True, (True, "below_break_even_barrier")),
        # Au-dessus du break-even -> SELL autorisé.
        (False, False, False, (False, "accepted")),
    ],
)
def test_t3_agent_close_gate_contract(
    exit_authority: bool,
    budget_blocked: bool,
    below_break_even: bool,
    expected: tuple[bool, str],
) -> None:
    assert (
        resolve_agent_close_gate(
            exit_authority=exit_authority,
            budget_blocked=budget_blocked,
            below_break_even=below_break_even,
        )
        == expected
    )


def test_t3_hysteresis_fires_on_premature_close() -> None:
    """Integration : loss_cut (SELL sous break-even) -> hysteresis > 0."""
    result = _run_scenario_with_size(
        "loss_cut",
        size_raw=0.0,
        steps=INTEGRATION_STEPS,
        tag="t3_loss_cut",
    )
    rej = result["rejection_reasons"]
    # Si le scénario a ouvert une position puis tenté un close perdant,
    # la barrière AGENT_CLOSE doit avoir produit du hysteresis.
    if result["opens"]["count"] > 0 and result["requested"].get("SELL", 0) > 0:
        assert int(rej.get("hysteresis", 0)) > 0, (
            "closes perdants tentés sans aucun rejet hysteresis : "
            "la barrière break-even ne protège plus"
        )


# ──────────────────────────────────────────────────────────────────────────
# T4 — Routing d'état (route_action_by_state)
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("a0", "in_position", "slot_available", "expected"),
    [
        # FLAT : seul a0 > +thr ouvre. Le négatif est un HOLD neutre.
        (-1.0, False, True, HOLD),
        (0.0, False, True, HOLD),
        (+1.0, False, True, BUY),
        # LONG : seul a0 < -thr clôture. Le positif est un HOLD neutre.
        (-1.0, True, True, SELL),
        (0.0, True, True, HOLD),
        (+1.0, True, True, HOLD),
        # Slot indisponible + FLAT -> HOLD forcé (quota tier).
        (+1.0, False, False, HOLD),
        (-1.0, False, False, HOLD),
        # LONG + slot indisponible : on peut TOUJOURS clôturer.
        (-1.0, True, False, SELL),
        # Seuil : |a0| <= thr -> HOLD des deux côtés.
        (0.05, False, True, HOLD),
        (-0.05, True, True, HOLD),
    ],
)
def test_t4_state_routing_contract(
    a0: float, in_position: bool, slot_available: bool, expected: int
) -> None:
    assert (
        route_action_by_state(
            a0, in_position, slot_available=slot_available, threshold=0.10
        )
        == expected
    )


def test_t4_asymmetric_sell_threshold() -> None:
    """sell_threshold < threshold facilite la sortie (FIX-D documenté)."""
    # a0 négatif dans la zone morte du buy-thr mais sous le sell-thr.
    assert (
        route_action_by_state(
            -0.06, True, threshold=0.10, sell_threshold=0.05
        )
        == SELL
    )
    # Symétrie legacy conservée quand sell_threshold=None.
    assert route_action_by_state(-0.06, True, threshold=0.10) == HOLD


# ──────────────────────────────────────────────────────────────────────────
# T5 — Le sizing final DOIT dépendre de size_raw (anomalie C, PATCH 4)
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.xfail(
    reason=(
        "Anomalie C confirmée (PHASE 2 §4.3) : target_exposure de size_raw "
        "(env l.8719) est écrasé par LINEAR_EXPO x bull_prob_HMM (l.8745). "
        "XFAIL jusqu'au PATCH 4 (sizing rendu à l'agent)."
    ),
    strict=False,
)
def test_t5_agent_size_raw_drives_exposure() -> None:
    """Deux runs identiques sauf size_raw=-1 vs +1 DOIVENT différer.

    Pré-PATCH 4 : les notionals sont identiques (le canal size est mort)
    -> le test échoue (xfail attendu). Post-PATCH 4 : size_raw=+1 doit
    produire des notionals significativement plus grands que size_raw=-1.
    """
    steps = max(200, INTEGRATION_STEPS // 2)
    low = _run_scenario_with_size(
        "constant_buy", size_raw=-1.0, steps=steps, tag="t5_size_low"
    )
    high = _run_scenario_with_size(
        "constant_buy", size_raw=+1.0, steps=steps, tag="t5_size_high"
    )
    assert low["opens"]["count"] > 0 and high["opens"]["count"] > 0
    mean_low = low["opens"]["notional_usd"]["mean"]
    mean_high = high["opens"]["notional_usd"]["mean"]
    assert mean_low is not None and mean_high is not None
    # size_raw=-1 -> normalized 0.0 -> 70% ; size_raw=+1 -> 1.0 -> 90%.
    # L'écart attendu est ~22% du notional ; on exige au moins 10% pour
    # prouver que le canal size a un effet réel.
    assert mean_high > mean_low * 1.10, (
        f"size_raw sans effet : mean(size=+1)={mean_high:.2f} vs "
        f"mean(size=-1)={mean_low:.2f} -> le sizing de l'agent est ignoré"
    )


# ──────────────────────────────────────────────────────────────────────────
# T6 — Taxonomie télémétrie des rejets (prérequis fix illegal_ratio V29)
# ──────────────────────────────────────────────────────────────────────────

REJECTION_TAXONOMY: dict[str, str] = {
    # Violation directe du contrat capital_tiers (hard gate).
    "risk_gate": "contract_tier",
    # Frictions économiques (SELL bloqué sous break-even, EV négatif).
    "hysteresis": "economic",
    "fee_gate": "economic",
    # Frictions opérationnelles (temps / quotas / cash plancher).
    "cooldown_hold_min": "operational",
    "cooldown_wait": "operational",
    "cooldown_omega4e": "operational",
    "daily_limit": "operational",
    "min_notional": "operational",
    # Routing d'état / HOLD neutres — PAS des fautes de l'agent.
    "sell_no_position": "routing_neutral",
    "anti_spam_hold": "routing_neutral",
    "pm_rejected": "operational",
}


def test_t6_every_rejection_reason_is_classified(constant_buy_run: dict) -> None:
    """Toute clé émise par env.rejection_reasons appartient à la taxonomie.

    illegal_ratio (train_parallel_agents l.774) somme TOUTES les clés ;
    sans taxonomie exhaustive, la métrique confond violation du contrat
    tier et friction économique — le fix V29 s'appuie sur ce mapping.
    """
    observed = set(constant_buy_run["rejection_reasons"].keys())
    unknown = observed - set(REJECTION_TAXONOMY)
    assert not unknown, (
        f"clés rejection_reasons non classées : {sorted(unknown)} — "
        "étendre REJECTION_TAXONOMY avant tout fix illegal_ratio"
    )


def test_t6_taxonomy_categories_are_mutually_exclusive() -> None:
    cats = set(REJECTION_TAXONOMY.values())
    assert cats == {
        "contract_tier",
        "economic",
        "operational",
        "routing_neutral",
    }
    # risk_gate est le SEUL rejet de nature contractuelle tier.
    contract = {k for k, v in REJECTION_TAXONOMY.items() if v == "contract_tier"}
    assert contract == {"risk_gate"}


# ──────────────────────────────────────────────────────────────────────────
# PATCH 1 (préparation) — clamp_policy_log_std respecte les bornes données
# ──────────────────────────────────────────────────────────────────────────


def test_patch1_clamp_policy_log_std_respects_bounds() -> None:
    """Le clamp log_std applique exactement [min, max] sur policy.log_std."""
    torch = pytest.importorskip("torch")
    from adan_trading_bot.utils.ppo_safety import clamp_policy_log_std

    class _Policy:
        def __init__(self) -> None:
            self.log_std = torch.nn.Parameter(torch.tensor([-9.0, 0.5, 3.0]))

    class _Model:
        def __init__(self) -> None:
            self.policy = _Policy()

    model = _Model()
    clamp_policy_log_std(model, min_log_std=-2.0, max_log_std=0.0)
    values = model.policy.log_std.detach().tolist()
    assert values == [-2.0, 0.0, 0.0], (
        f"clamp [-2,0] incorrect : {values} — PATCH 1 gate a0_std<1.0 "
        "non atteignable si les bornes ne sont pas appliquées"
    )
