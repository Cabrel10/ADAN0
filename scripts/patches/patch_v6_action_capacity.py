#!/usr/bin/env python3
"""
DIAGNOSTIC-V6 patch (part 1) — expose ACTION CAPACITY in the observation.

User insight (proven by FACTS at 30k):
  - 91% of BUY are impossible (284 CASH_FLOOR / 28 executed) because the Micro
    tier exposure_range is [70,90]% -> a single BUY consumes ~95% of the $18-20
    capital, leaving ~$1.8 cash < $11 min_notional. The agent is then punished
    (CASH_FLOOR) for being fully invested -> learns "BUY = pain".
  - The 20-dim portfolio_state had slot [9] = reserved = 0.0 and NO explicit
    legal-capacity signal. The agent is BLIND to "can I BUY right now?".

Design (user): etat reel -> capacites legales -> action autorisee.
This patch fills slot [9] with `can_buy_now` in [0,1]:
    1.0  = cash margin comfortably above min_notional AND a free position slot
    0.0  = cannot open (no cash buffer OR all position slots used)
    in-between = soft margin ratio (smooth gradient for PPO).

Shape stays 20 dims (no obs-space resize, no model incompatibility).

Run: python scripts/patches/patch_v6_action_capacity.py
"""
import io, sys

F = "src/adan_trading_bot/portfolio/portfolio_manager.py"
s = io.open(F, encoding="utf-8").read()

if "can_buy_now" in s:
    print("[SKIP] already patched (can_buy_now present)")
    sys.exit(0)

old = (
    "                np.clip(profit_factor, 0.0, 1.0),                                # [8] profit_factor_norm\n"
    "                0.0,                                                              # [9] reserved\n"
    "            ]"
)
new = (
    "                np.clip(profit_factor, 0.0, 1.0),                                # [8] profit_factor_norm\n"
    "                0.0,  # [9] PLACEHOLDER -> overwritten below by can_buy_now\n"
    "            ]\n"
    "            # ----------------------------------------------------------------\n"
    "            # DIAGNOSTIC-V6: slot [9] = can_buy_now (action-capacity signal).\n"
    "            # The agent was BLIND to whether a BUY is even legal -> it kept\n"
    "            # requesting BUY while fully invested and got punished (CASH_FLOOR).\n"
    "            # Now it SEES its legal arsenal. etat reel -> capacites legales.\n"
    "            # ----------------------------------------------------------------\n"
    "            try:\n"
    "                _min_notional = float(getattr(self, 'min_trade_value', 11.0))\n"
    "                _max_pos = max(int(getattr(self, 'max_concurrent_positions', 1)), 1)\n"
    "                _free_slot = 1.0 if int(open_count) < _max_pos else 0.0\n"
    "                # soft cash margin: 0 at exactly min_notional, ->1 at 2x min_notional\n"
    "                _cash_margin = (cash - _min_notional) / max(_min_notional, 1e-8)\n"
    "                _cash_ok = float(np.clip(_cash_margin, 0.0, 1.0))\n"
    "                can_buy_now = _free_slot * _cash_ok\n"
    "            except Exception:\n"
    "                can_buy_now = 0.0\n"
    "            state[9] = float(np.clip(can_buy_now, 0.0, 1.0))"
)
assert s.count(old) == 1, ("anchor", s.count(old))
s = s.replace(old, new, 1)

io.open(F, "w", encoding="utf-8").write(s)
print("[OK] V6 action-capacity (can_buy_now in slot [9]) applied.")
