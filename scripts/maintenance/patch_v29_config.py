#!/usr/bin/env python3
"""V29 — Corrections config + PATCH 3 (hysteresis/deadband) — idempotent.

Corrections appliquees (toutes validees par l'utilisateur) :
  CFG-1  min_order_value_usdt 11.0 -> 5.0 (hard_constraints + trading_rules
         + les 3 autres occurrences dead config harmonisees pour coherence)
  CFG-2  hard_constraints.max_position_size_pct / max_risk_per_trade_pct :
         commentes (dead config — aucun consommateur dans src/)
  CFG-3  environment.features_config (l.668) : supprime (doublon de
         data.features_config l.260, zero consommateur)
  CFG-4  environment.timeframe_trading_config (l.868) : supprime (fantome,
         zero consommateur — environment.action_thresholds fait foi)
  PATCH2 saturation_penalty lambda 0.10 -> 0.02, cap 0.20 -> 0.05
         (saturation_penalty dominait le reward : mean -0.1249 vs
         behavior_invalid_penalty -0.0087 — 14x)
  PATCH3 hysteresis : plancher exposition SELL 5% -> 2%, barriere
         AGENT_CLOSE multipliee par ADAN_BARRIER_MULT defaut 1.0 -> 0.75
         (gate smoke: illegal_ratio < 50%; hysteresis ~70% des rejets V28)
"""
from pathlib import Path

cfg_path = Path("config/config.yaml")
env_path = Path("src/adan_trading_bot/environment/multi_asset_chunked_env.py")
cfg = cfg_path.read_text(encoding="utf-8")
env = env_path.read_text(encoding="utf-8")

changes = []


def sub(text: str, old: str, new: str, name: str, count: int = 1) -> str:
    if new in text and old not in text:
        changes.append(f"{name}: deja applique — skip")
        return text
    n = text.count(old)
    if n != count:
        raise SystemExit(f"ERREUR {name}: {n} occurrence(s) de la cible (attendu {count}) — abort")
    changes.append(f"{name}: OK ({count}x)")
    return text.replace(old, new)


# ── CFG-1 : min_order 11.0 -> 5.0 (4 occurrences, toutes harmonisees) ──────
cfg = sub(
    cfg,
    "min_order_value_usdt: 11.0",
    "min_order_value_usdt: 5.0  # V29: was 11.0 (54% du capital Micro 20.5$) — plancher operationnel aligne tier",
    "CFG-1 min_order 11->5",
    count=4,
)

# ── CFG-2 : dead configs hard_constraints commentes ─────────────────────────
cfg = sub(
    cfg,
    "    max_position_size_pct: 0.5\n    max_risk_per_trade_pct: 0.02\n    min_order_value_usdt: 5.0",
    "    # V29: DEAD CONFIG — aucun consommateur dans src/ (verifie PHASE 1).\n"
    "    # Le domaine d'exposition est dicte EXCLUSIVEMENT par capital_tiers.\n"
    "    # max_position_size_pct: 0.5\n"
    "    # max_risk_per_trade_pct: 0.02\n"
    "    min_order_value_usdt: 5.0",
    "CFG-2 dead configs commentes",
)

# ── CFG-3 : environment.features_config (doublon, zero consommateur) ────────
fc_old = (
    "  features_config:\n"
    "    indicators:\n"
    "    - rsi_14\n"
    "    - macd_hist\n"
    "    - atr_14\n"
    "    - bb_upper\n"
    "    - bb_middle\n"
    "    - bb_lower\n"
    "    price:\n"
    "    - open\n"
    "    - high\n"
    "    - low\n"
    "    - close\n"
    "    volume:\n"
    "    - volume\n"
    "  frequency_validation:"
)
fc_new = (
    "  # V29: environment.features_config SUPPRIME (doublon de\n"
    "  # data.features_config, zero consommateur dans src/ — verifie PHASE 1).\n"
    "  frequency_validation:"
)
cfg = sub(cfg, fc_old, fc_new, "CFG-3 env.features_config supprime")

# ── CFG-4 : timeframe_trading_config (fantome, zero consommateur) ───────────
ttc_old = (
    "  timeframe_trading_config:\n"
    "    1h:\n"
    "      action_threshold: 0.02   # Session 9: was 0.08 — see action_thresholds note\n"
    "      description: Medium-term trading optimized\n"
    "      force_trade_steps: 120\n"
    "      min_magnitude: 0.08  # SESSION 15: Increased from 0.05 to filter weak signals\n"
    "    4h:\n"
    "      action_threshold: 0.03   # Session 9: was 0.10\n"
    "      description: Long-term trading enabled\n"
    "      force_trade_steps: 240\n"
    "      min_magnitude: 0.12  # SESSION 15: Increased from 0.08 to require stronger setups\n"
    "    5m:\n"
    "      action_threshold: 0.01   # Session 9: was 0.05\n"
    "      description: High-frequency trading enabled\n"
    "      force_trade_steps: 72\n"
    "      min_magnitude: 0.06  # SESSION 15: Increased from 0.03 to avoid micro-trades\n"
    "  trading_mode: backtest"
)
ttc_new = (
    "  # V29: timeframe_trading_config SUPPRIME (fantome, zero consommateur —\n"
    "  # environment.action_thresholds l.649 est la source de verite).\n"
    "  trading_mode: backtest"
)
cfg = sub(cfg, ttc_old, ttc_new, "CFG-4 timeframe_trading_config supprime")

# ── PATCH 2 : saturation_penalty reduit ─────────────────────────────────────
cfg = sub(
    cfg,
    "    lambda: 0.10              # poids de la penalite log\n"
    "    cap: 0.20                 # plafond absolu de la penalite par step",
    "    lambda: 0.02              # V29 PATCH2: was 0.10 — dominait le reward\n"
    "                              #   (mean -0.1249 = 14x behavior_invalid -0.0087)\n"
    "    cap: 0.05                 # V29 PATCH2: was 0.20 — plafond aligne lambda",
    "PATCH2 saturation_penalty 0.10/0.20 -> 0.02/0.05",
)

cfg_path.write_text(cfg, encoding="utf-8")

# ── PATCH 3 : hysteresis reduit (env) ───────────────────────────────────────
# 3a. plancher exposition SELL 5% -> 2%
env = sub(
    env,
    "                    if _exposure < 0.05:\n"
    "                        # Position trop petite pour valoir les frais",
    "                    # V29 PATCH3: 0.05 -> 0.02 — le plancher 5% bloquait\n"
    "                    # des SELL legitimes (hysteresis ~70% des rejets V28).\n"
    "                    if _exposure < 0.02:\n"
    "                        # Position trop petite pour valoir les frais",
    "PATCH3a plancher expo SELL 0.05->0.02",
)
# 3b. barriere AGENT_CLOSE : ADAN_BARRIER_MULT defaut 1.0 -> 0.75
env = sub(
    env,
    '_barrier_mult = float(_os_v17.environ.get("ADAN_BARRIER_MULT", "1.0"))',
    '# V29 PATCH3: defaut 1.0 -> 0.75 (reduit hysteresis, gate illegal<50%)\n'
    '                        _barrier_mult = float(_os_v17.environ.get("ADAN_BARRIER_MULT", "0.75"))',
    "PATCH3b barrier mult 1.0->0.75",
)

env_path.write_text(env, encoding="utf-8")

# ── Validation finale ───────────────────────────────────────────────────────
import ast

import yaml

yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
ast.parse(env_path.read_text(encoding="utf-8"))
print("\n".join(changes))
print("VALIDATION: YAML + Python syntax OK")
