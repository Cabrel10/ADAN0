#!/usr/bin/env python3
"""V17-Audit: SELL lifecycle instrumentation (expert measures #1 + #5).

Tracks the full SELL funnel per episode so we can DISTINGUISH
  "the agent never wanted to sell"  vs  "its SELLs were refused".

Counters (self.sell_lifecycle):
  requested_open : agent emitted SELL on an OPEN position (discrete_action==2 & is_open)
  rej_hold_min   : refused because position held < HOLD_MIN
  rej_budget     : refused by decision-budget / gap / daily-quota
  rej_barrier    : refused by the break-even barrier   <-- THE measured culprit
  executed       : SELL actually executed (position closed)
  exec_profit    : executed SELLs whose realized PnL > 0

Purely diagnostic; does NOT change agent behaviour. Edit tool FAILS on the 498KB
file -> Python io string-replace. Idempotent.
"""
import io, sys

ENV = "src/adan_trading_bot/environment/multi_asset_chunked_env.py"

# --- 1) requested_open : the SELL branch entry -----------------------------
OLD1 = '''            if discrete_action == 2 and is_open:
                # Cooldown post-BUY par timeframe (HOLD_MIN)'''
NEW1 = '''            if discrete_action == 2 and is_open:
                # V17-Audit: SELL requested on an open position.
                try:
                    self.sell_lifecycle["requested_open"] += 1
                except Exception:
                    pass
                # Cooldown post-BUY par timeframe (HOLD_MIN)'''

# --- 2) rej_hold_min -------------------------------------------------------
OLD2 = '''                if _steps_held < _hold_min:
                    # SELL trop tôt — pénalité proportionnelle au manque
                    self.invalid_trade_attempts += 1
                    self.rejection_reasons["cooldown_hold_min"] += 1'''
NEW2 = '''                if _steps_held < _hold_min:
                    # SELL trop tôt — pénalité proportionnelle au manque
                    self.invalid_trade_attempts += 1
                    self.rejection_reasons["cooldown_hold_min"] += 1
                    try:
                        self.sell_lifecycle["rej_hold_min"] += 1
                    except Exception:
                        pass'''

# --- 3) rej_budget ---------------------------------------------------------
OLD3 = '''                        if _budget_blocked:
                            discrete_action = 0
                            self.rejection_reasons["hysteresis"] += 1'''
NEW3 = '''                        if _budget_blocked:
                            discrete_action = 0
                            self.rejection_reasons["hysteresis"] += 1
                            try:
                                self.sell_lifecycle["rej_budget"] += 1
                            except Exception:
                                pass'''

# --- 4) rej_barrier (THE culprit) ------------------------------------------
OLD4 = '''                        elif unrealized_pnl_pct < _barrier:
                            # Reject AGENT_CLOSE - rentabilite insuffisante vs frais.
                            discrete_action = 0
                            self.rejection_reasons["hysteresis"] += 1'''
NEW4 = '''                        elif unrealized_pnl_pct < _barrier:
                            # Reject AGENT_CLOSE - rentabilite insuffisante vs frais.
                            discrete_action = 0
                            self.rejection_reasons["hysteresis"] += 1
                            try:
                                self.sell_lifecycle["rej_barrier"] += 1
                            except Exception:
                                pass'''

# --- 5) executed + exec_profit ---------------------------------------------
OLD5 = '''                                self._apply_trade_results_safely(
                                    pnl_value=float(val), fees=float(fees))
                                trade_executed_this_step = True
                                # DIAGNOSTIC-V5: legal SELL -> relax sterile pressure'''
NEW5 = '''                                self._apply_trade_results_safely(
                                    pnl_value=float(val), fees=float(fees))
                                trade_executed_this_step = True
                                try:
                                    self.sell_lifecycle["executed"] += 1
                                    if float(val) > 0.0:
                                        self.sell_lifecycle["exec_profit"] += 1
                                except Exception:
                                    pass
                                # DIAGNOSTIC-V5: legal SELL -> relax sterile pressure'''

PATCHES = [(OLD1, NEW1, "requested_open"), (OLD2, NEW2, "rej_hold_min"),
           (OLD3, NEW3, "rej_budget"), (OLD4, NEW4, "rej_barrier"),
           (OLD5, NEW5, "executed/exec_profit")]

def main():
    with io.open(ENV, "r", encoding="utf-8") as f:
        src = f.read()
    if 'self.sell_lifecycle["requested_open"] += 1' in src:
        print("PATCH_LIFECYCLE_ALREADY_PRESENT")
        return 0
    for old, new, name in PATCHES:
        if old not in src:
            print(f"ANCHOR_NOT_FOUND: {name}")
            return 2
    for old, new, name in PATCHES:
        src = src.replace(old, new, 1)
    with io.open(ENV, "w", encoding="utf-8") as f:
        f.write(src)
    print("PATCH_LIFECYCLE_APPLIED (5 stages)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
