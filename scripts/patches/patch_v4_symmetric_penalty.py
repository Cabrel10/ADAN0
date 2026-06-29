#!/usr/bin/env python3
"""
DIAGNOSTIC-V4 patch (2026-06-30): break the entropy collapse at its ROOT.

PROVEN ROOT CAUSE (forensic + 28k diagnostic run):
  _inv_pen_weight = 0.0 (line 7279, "C6 fix") makes EVERY BUY-gate rejection
  cost ZERO reward. Meanwhile SELL-while-flat DOES get a geometric "sterile"
  penalty. This asymmetry teaches the agent that BUY is the only painless
  action -> it saturates action0 at +1 (BUY 97% of steps, illegal_ratio 97%,
  HOLD 0.3%, a0_std 13.5). Diagnostic V3 (ent_coef/cooldown/cap) FAILED 5/5
  because no hyperparameter can overcome a free-exploit in the reward.

THE FIX (symmetric, minimal, gradient-visible — NOT action masking):
  Make BUY-illegal cost the SAME geometric sterile penalty as SELL-illegal.
  We do NOT mask the action (it still resolves to HOLD), but the gradient now
  SEES that the requested BUY was invalid. Reuses the EXACT same config knobs
  (invalid_trade_penalty_weight / sterile_action_geom_ratio /
  sterile_action_penalty_cap) so the two penalties are symmetric by design.

  Fees are UNTOUCHED (commission=0.0025, round_trip_fees=0.005).

Targets (verified by reading the file):
  1. Add helper `_sterile_penalty_for_tier()` (DRY: SELL & BUY share it).
  2. anti_spam_hold (BUY-while-open, ~line 7507): add sterile penalty.
  3. min_notional CASH_FLOOR (BUY no-cash, ~line 7545): add sterile penalty.
  4. SELL-flat (~line 7965): refactor to call the helper (behavior identical).
"""
import sys

F = "src/adan_trading_bot/environment/multi_asset_chunked_env.py"
src = open(F, encoding="utf-8").read()
orig = src

# ---------------------------------------------------------------------------
# PART 1 — insert the shared helper just before the `_inv_pen_weight = 0.0`
# line (line ~7279). The helper computes the same tier-geometric penalty the
# SELL-flat branch already uses.
# ---------------------------------------------------------------------------
anchor1 = '        _inv_pen_weight = 0.0  # was 0.005 — C6 fix (all gate rejections = 0 reward)'
helper = '''        _inv_pen_weight = 0.0  # was 0.005 — C6 fix (all gate rejections = 0 reward)

        # DIAGNOSTIC-V4 (2026-06-30): symmetric sterile penalty helper.
        # Root cause of the entropy collapse = asymmetry. SELL-while-flat is
        # penalized (geometric sterile pen) but BUY-illegal was FREE, so the
        # agent saturated on BUY. This closure returns the SAME geometric
        # penalty for ANY invalid intent, making BUY/SELL symmetric. It reads
        # the same reward_shaping knobs; fees are not involved.
        def _sterile_penalty_for_tier():
            _tier_order = {"micro": 0, "small": 1, "medium": 2,
                           "high": 3, "enterprise": 4}
            _tname = "micro"
            try:
                _ct = self.portfolio_manager.get_current_tier()
                if isinstance(_ct, dict):
                    _tname = str(_ct.get("name", "Micro")).split()[0].lower()
            except Exception:
                pass
            _k = _tier_order.get(_tname, 0)
            _rs = self.config.get("reward_shaping", {})
            _base = float(_rs.get("invalid_trade_penalty_weight", 0.005))
            _r = float(_rs.get("sterile_action_geom_ratio", 1.6))
            _cap = float(_rs.get("sterile_action_penalty_cap", 0.05))
            return min(_cap, _base * (_r ** _k))'''

if anchor1 not in src:
    print("ERROR: helper anchor (_inv_pen_weight line) not found.")
    sys.exit(1)
if src.count(anchor1) != 1:
    print(f"ERROR: helper anchor found {src.count(anchor1)} times (expected 1).")
    sys.exit(1)
src = src.replace(anchor1, helper, 1)
print("PART 1 OK — helper inserted")

# ---------------------------------------------------------------------------
# PART 2 — BUY-while-open (anti_spam_hold). Add sterile penalty when we
# override BUY to HOLD because we're already in position.
# ---------------------------------------------------------------------------
anchor2 = '''                    if exposure_diff < 0.10:  # Within 10% -> no action needed (OMEGA-4E)
                        discrete_action = 0  # Override to HOLD
                        self.rejection_reasons["anti_spam_hold"] += 1
                        if self.current_step % 100 == 0:'''
repl2 = '''                    if exposure_diff < 0.10:  # Within 10% -> no action needed (OMEGA-4E)
                        discrete_action = 0  # Override to HOLD
                        self.rejection_reasons["anti_spam_hold"] += 1
                        # DIAGNOSTIC-V4: BUY-while-open is invalid intent. It is
                        # converted to HOLD but must NOT be free (root-cause fix).
                        self._step_invalid_penalty += -_sterile_penalty_for_tier()
                        if self.current_step % 100 == 0:'''
if anchor2 not in src:
    print("ERROR: anti_spam_hold anchor not found.")
    sys.exit(1)
if src.count(anchor2) != 1:
    print(f"ERROR: anti_spam anchor found {src.count(anchor2)} times (expected 1).")
    sys.exit(1)
src = src.replace(anchor2, repl2, 1)
print("PART 2 OK — BUY-while-open penalized")

# ---------------------------------------------------------------------------
# PART 3 — min_notional CASH_FLOOR (BUY with no cash). Add sterile penalty.
# ---------------------------------------------------------------------------
anchor3 = '''                    else:
                        # Truly cannot afford — hard HOLD
                        self.invalid_trade_attempts += 1
                        self.rejection_reasons["min_notional"] += 1
                        if self.current_step % 50 == 0:'''
repl3 = '''                    else:
                        # Truly cannot afford — hard HOLD
                        self.invalid_trade_attempts += 1
                        self.rejection_reasons["min_notional"] += 1
                        # DIAGNOSTIC-V4: spamming BUY with no cash was the main
                        # collapse exploit (5000+ free rejections). Now penalized.
                        self._step_invalid_penalty += -_sterile_penalty_for_tier()
                        if self.current_step % 50 == 0:'''
if anchor3 not in src:
    print("ERROR: min_notional CASH_FLOOR anchor not found.")
    sys.exit(1)
if src.count(anchor3) != 1:
    print(f"ERROR: min_notional anchor found {src.count(anchor3)} times (expected 1).")
    sys.exit(1)
src = src.replace(anchor3, repl3, 1)
print("PART 3 OK — min_notional BUY penalized")

# ---------------------------------------------------------------------------
# PART 4 — refactor SELL-flat to use the helper (identical numeric behavior,
# removes duplicated code so the two paths can never drift apart).
# ---------------------------------------------------------------------------
anchor4 = '''                _k = _tier_order.get(_tname, 0)
                # base = invalid_trade_penalty_weight (config), ratio r>1, cap borne.
                # NB: ces clefs sont sous reward_shaping (cf. structure config.yaml).
                _rs_cfg = self.config.get("reward_shaping", {})
                _base = float(_rs_cfg.get("invalid_trade_penalty_weight", 0.005))
                _r = float(_rs_cfg.get("sterile_action_geom_ratio", 1.6))
                _cap = float(_rs_cfg.get("sterile_action_penalty_cap", 0.05))
                _sterile_pen = min(_cap, _base * (_r ** _k))
                self._step_invalid_penalty += -_sterile_pen'''
repl4 = '''                _k = _tier_order.get(_tname, 0)
                # DIAGNOSTIC-V4: use the shared helper so SELL & BUY sterile
                # penalties stay symmetric. Numeric behavior identical to before.
                _sterile_pen = _sterile_penalty_for_tier()
                self._step_invalid_penalty += -_sterile_pen'''
if anchor4 not in src:
    print("WARN: SELL-flat refactor anchor not found (skipping PART 4, non-fatal).")
else:
    if src.count(anchor4) != 1:
        print(f"WARN: SELL-flat anchor found {src.count(anchor4)} times — skipping PART 4.")
    else:
        src = src.replace(anchor4, repl4, 1)
        print("PART 4 OK — SELL-flat refactored to helper")

if src == orig:
    print("ERROR: no changes made.")
    sys.exit(1)

open(F, "w", encoding="utf-8").write(src)
print("\nPATCH V4 WRITTEN.")
