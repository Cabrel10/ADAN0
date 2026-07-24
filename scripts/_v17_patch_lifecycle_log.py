#!/usr/bin/env python3
"""V17-Audit: log the SELL lifecycle at episode reset, then (re)initialize it.

Placed exactly where rejection_reasons is reset each episode. Logs a single
[SELL_LIFECYCLE] line summarizing the funnel of the episode that just ended
(requested -> rejected[reason] -> executed -> profitable), then resets the dict.

Edit tool FAILS on the 498KB file -> Python io string-replace. Idempotent.
"""
import io, sys

ENV = "src/adan_trading_bot/environment/multi_asset_chunked_env.py"

OLD = '''        self.rejection_reasons = {
            "fee_gate": 0, "risk_gate": 0, "cooldown_wait": 0,
            "cooldown_hold_min": 0, "cooldown_omega4e": 0,
            "min_notional": 0, "hysteresis": 0, "anti_spam_hold": 0,
            "daily_limit": 0, "pm_rejected": 0, "sell_no_position": 0,
        }'''

NEW = '''        # V17-Audit: log the SELL funnel of the episode that just ended, then reset.
        _prev_sl = getattr(self, "sell_lifecycle", None)
        if isinstance(_prev_sl, dict) and _prev_sl.get("requested_open", 0) > 0:
            _req = max(1, int(_prev_sl.get("requested_open", 0)))
            _rejb = int(_prev_sl.get("rej_barrier", 0))
            try:
                self.logger.info(
                    "[SELL_LIFECYCLE] requested_open=%d rej_hold_min=%d rej_budget=%d "
                    "rej_barrier=%d (%.1f%%) executed=%d exec_profit=%d" % (
                        int(_prev_sl.get("requested_open", 0)),
                        int(_prev_sl.get("rej_hold_min", 0)),
                        int(_prev_sl.get("rej_budget", 0)),
                        _rejb, 100.0 * _rejb / _req,
                        int(_prev_sl.get("executed", 0)),
                        int(_prev_sl.get("exec_profit", 0)),
                    )
                )
            except Exception:
                pass
        self.sell_lifecycle = {
            "requested_open": 0, "rej_hold_min": 0, "rej_budget": 0,
            "rej_barrier": 0, "executed": 0, "exec_profit": 0,
        }
        self.rejection_reasons = {
            "fee_gate": 0, "risk_gate": 0, "cooldown_wait": 0,
            "cooldown_hold_min": 0, "cooldown_omega4e": 0,
            "min_notional": 0, "hysteresis": 0, "anti_spam_hold": 0,
            "daily_limit": 0, "pm_rejected": 0, "sell_no_position": 0,
        }'''

def main():
    with io.open(ENV, "r", encoding="utf-8") as f:
        src = f.read()
    if "[SELL_LIFECYCLE] requested_open=" in src:
        print("PATCH_LIFECYCLE_LOG_ALREADY_PRESENT")
        return 0
    if OLD not in src:
        print("ANCHOR_NOT_FOUND")
        return 2
    src = src.replace(OLD, NEW, 1)
    with io.open(ENV, "w", encoding="utf-8") as f:
        f.write(src)
    print("PATCH_LIFECYCLE_LOG_APPLIED")
    return 0

if __name__ == "__main__":
    sys.exit(main())
