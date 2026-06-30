#!/usr/bin/env python3
"""
DIAGNOSTIC-V6 patch (part 2) — penalty CALIBRATION.

User insights (FACTS-confirmed at 30k):
  BUG2: CASH_FLOOR (min_notional) punishes the agent for being fully invested
        / having no cash — a state it does NOT control (exposure_range [70,90]
        forces ~95% allocation). Punishing an uncontrollable variable teaches
        "BUY = pain". Fix: min_notional severity ~ 0 (signal only, no real pain).
        The real "don't BUY twice" lesson is taught by anti_spam_hold
        (BUY_WHILE_OPEN), which the agent DOES control.
  BUG3: "la penalite actuelle est trop agressive trop tot. Le PPO n'a pas le
        temps d'apprendre avant d'etre puni." Fix: a global WARMUP ramp scales
        every sterile penalty from 0 -> 1 over the first `sterile_warmup_steps`
        environment steps, so early exploration is not crushed.

Also makes the penalty keyed correctly per family with calibrated severities.
FEES UNTOUCHED.

Run: python scripts/patches/patch_v6_penalty_calibration.py
"""
import io, sys

F = "src/adan_trading_bot/environment/multi_asset_chunked_env.py"
s = io.open(F, encoding="utf-8").read()

if "sterile_warmup_steps" in s:
    print("[SKIP] already patched (sterile_warmup_steps present)")
    sys.exit(0)

# --- 1) Recalibrate severities: min_notional made (near) harmless. -----------
old_sev = (
    "            _sev = {\n"
    '                "sell_no_position": 1.5,\n'
    '                "anti_spam_hold": 1.2,   # BUY while already open\n'
    '                "min_notional": 0.8,     # BUY blocked by cash\n'
    '                "hysteresis": 0.7,\n'
    '                "cooldown_wait": 0.6,\n'
    '                "cooldown_hold_min": 0.6,\n'
    "            }\n"
    "            return float(_sev.get(reason, 1.0))"
)
new_sev = (
    "            # DIAGNOSTIC-V6: severities recalibrated. min_notional is an\n"
    "            # UNCONTROLLABLE state (no cash) -> near-zero so we do not teach\n"
    "            # 'BUY = pain'. The controllable mistake (BUY while open) keeps\n"
    "            # a real but moderate severity. SELL-no-position stays the worst.\n"
    "            _sev = {\n"
    '                "sell_no_position": 1.2,\n'
    '                "anti_spam_hold": 0.8,   # BUY while already open (controllable)\n'
    '                "min_notional": 0.05,    # BUY blocked by cash (NOT a fault)\n'
    '                "hysteresis": 0.4,\n'
    '                "cooldown_wait": 0.4,\n'
    '                "cooldown_hold_min": 0.4,\n'
    "            }\n"
    "            return float(_sev.get(reason, 0.8))"
)
assert s.count(old_sev) == 1, ("sev", s.count(old_sev))
s = s.replace(old_sev, new_sev, 1)

# --- 2) Add warmup ramp inside _sterile_penalty_v5. --------------------------
old_pen = (
    "            _mult = _invalid_ratio_mult_v5()\n"
    "            _pen = _base * min(_cap / _base if _base > 0 else _cap,\n"
    "                               1.0 + _alpha * _acc) * _mult\n"
    "            _pen = min(_pen, _cap)\n"
    "            return _pen, self._sterile_streak[reason], _acc, _mult"
)
new_pen = (
    "            _mult = _invalid_ratio_mult_v5()\n"
    "            # DIAGNOSTIC-V6: WARMUP ramp. Penalty scales 0->1 over the first\n"
    "            # sterile_warmup_steps env steps so PPO can explore/learn BEFORE\n"
    "            # being punished (user: 'trop agressive trop tot').\n"
    "            _warm = float(_rs.get('sterile_warmup_steps', 50000))\n"
    "            _cur = float(getattr(self, 'current_step', 0) or 0)\n"
    "            _ramp = 1.0 if _warm <= 0 else min(1.0, _cur / _warm)\n"
    "            _pen = _base * min(_cap / _base if _base > 0 else _cap,\n"
    "                               1.0 + _alpha * _acc) * _mult * _ramp\n"
    "            _pen = min(_pen, _cap)\n"
    "            return _pen, self._sterile_streak[reason], _acc, _mult"
)
assert s.count(old_pen) == 1, ("pen", s.count(old_pen))
s = s.replace(old_pen, new_pen, 1)

io.open(F, "w", encoding="utf-8").write(s)
print("[OK] V6 penalty calibration (min_notional harmless + warmup ramp) applied.")
