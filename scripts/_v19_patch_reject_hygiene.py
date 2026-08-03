#!/usr/bin/env python3
"""V19 io-patcher: credit-assignment hygiene for REJECTED SELLs.

Tutor directive (§V19, most important): a SELL that is REFUSED must NOT be
punished as if the agent chose a bad action.

  - SELL refused by BARRIER (PnL < break-even): the agent tried to exit but
    the trade isn't profitable enough yet. Punishing it teaches nothing and
    re-imprisons the agent (the V16 collapse root-cause). -> NO penalty.
  - SELL refused by BUDGET (cooldown/quota): informative structural signal,
    not a moral failing. -> NO penalty (kept only as a soft trace optionally).
  - Only EXECUTED SELLs are judged by the Future Arena.

We keep the rejection COUNTERS (sell_lifecycle) for observability, but zero
out the reward penalties. Controlled by ADAN_V19_REJECT_HYGIENE (default 1).
Set to 0 to restore the old punitive behavior for A/B comparison.

Idempotent. Uses string-replace (Edit fails on this 498KB file).
"""
import io
import sys

ENV = "src/adan_trading_bot/environment/multi_asset_chunked_env.py"

with io.open(ENV, "r", encoding="utf-8") as f:
    src = f.read()

if "V19: reject hygiene" in src:
    print("[V19] already patched, skipping.")
    sys.exit(0)

# --- Patch 1: budget rejection penalty -> gated by hygiene flag ----------
OLD_BUDGET = '''                            # Penalite douce proportionnelle au deficit de budget
                            # (gradient, pas no-op) — l'agent apprend a economiser.
                            _deficit_b = max(0.0, (_cost_close - _budget) / max(_cost_close, 1e-9))
                            _q_pen = -0.10 - 0.10 * min(1.0, _deficit_b)
                            self._step_invalid_penalty += _q_pen'''
NEW_BUDGET = '''                            # V19: reject hygiene — un SELL refuse par le BUDGET est
                            # un signal STRUCTUREL informatif (cooldown/quota), pas une
                            # faute morale. Par defaut AUCUNE penalite (l'agent n'a pas
                            # "mal choisi": il a voulu sortir, la friction l'en empeche).
                            # ADAN_V19_REJECT_HYGIENE=0 restaure l'ancien comportement.
                            import os as _os_v19b
                            _hygiene_b = _os_v19b.environ.get("ADAN_V19_REJECT_HYGIENE", "1") == "1"
                            _deficit_b = max(0.0, (_cost_close - _budget) / max(_cost_close, 1e-9))
                            _q_pen = 0.0 if _hygiene_b else (-0.10 - 0.10 * min(1.0, _deficit_b))
                            self._step_invalid_penalty += _q_pen'''

if OLD_BUDGET not in src:
    print("[V19] ERROR: budget penalty block not found.")
    sys.exit(1)
src = src.replace(OLD_BUDGET, NEW_BUDGET, 1)
print("[V19] budget-reject hygiene applied.")

# --- Patch 2: barrier rejection penalty -> gated by hygiene flag ---------
OLD_BARRIER = '''                            # Penalite GRADIENT (cf. reward_service.agent_close_barrier):
                            # proportionnelle au manque de rentabilite, bornee, negative.
                            _deficit = (_barrier - unrealized_pnl_pct) / max(_barrier, 1e-9)
                            _ac_pen = -0.15 * min(1.0, max(0.0, _deficit))
                            self._step_invalid_penalty += _ac_pen'''
NEW_BARRIER = '''                            # V19: reject hygiene — un SELL refuse par la BARRIERE
                            # (PnL < break-even) NE DOIT PAS etre puni. L'agent a
                            # tente la bonne action (sortir), mais le trade n'est pas
                            # encore rentable. Le punir = re-emprisonner l'agent =
                            # cause racine du collapse V16. Par defaut AUCUNE penalite.
                            # Seuls les SELL EXECUTES sont juges par le Future Arena.
                            # ADAN_V19_REJECT_HYGIENE=0 restaure l'ancien comportement.
                            import os as _os_v19a
                            _hygiene_a = _os_v19a.environ.get("ADAN_V19_REJECT_HYGIENE", "1") == "1"
                            _deficit = (_barrier - unrealized_pnl_pct) / max(_barrier, 1e-9)
                            _ac_pen = 0.0 if _hygiene_a else (-0.15 * min(1.0, max(0.0, _deficit)))
                            self._step_invalid_penalty += _ac_pen'''

if OLD_BARRIER not in src:
    print("[V19] ERROR: barrier penalty block not found.")
    sys.exit(1)
src = src.replace(OLD_BARRIER, NEW_BARRIER, 1)
print("[V19] barrier-reject hygiene applied.")

with io.open(ENV, "w", encoding="utf-8") as f:
    f.write(src)
print("[V19] File written.")
