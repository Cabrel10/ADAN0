#!/usr/bin/env python3
"""V17-Fix A: replace the STATIC break-even barrier (1.5 x round-trip fees)
with a DYNAMIC, present-only barrier scaled by ATR% and controlled by the
ADAN_BARRIER_MULT env-var.

Rationale (measured, 4 reproducible episodes): the static wall
`_barrier = 1.5 * _rt_fees` (~1.2% with 0.5% R/T fees) physically converted
95-99% of SELL intents into HOLD, imprisoning the agent. It never learned to
exit -> MAX_DURATION force-closed 69% of positions -> no SELL credit -> collapse.

The new barrier:
  - uses ONLY present information (ATR%, fees) -> NO future leak -> live-safe.
  - adapts to volatility: calm market -> low barrier (~fees), volatile -> higher.
  - default mult 1.0 (more permissive than 1.5) with fees now 0.40% R/T.
  - floored at _rt_fees (never below break-even) and capped at 2% for safety.
The barrier no longer BLOCKS learning; premature exits are punished PEDAGOGICALLY
by the Arena (lost_potential_penalty / Scenario B), not physically walled off.

Edit tool FAILS on this 498KB file -> use Python io string-replace.
Idempotent: re-running is a no-op if the marker is already present.
"""
import io, sys

ENV = "src/adan_trading_bot/environment/multi_asset_chunked_env.py"

OLD = '                        _barrier = 1.5 * _rt_fees  # ex: 1.5 x 0.8% = 1.2%'

NEW = '''                        # V17-Fix A: DYNAMIC break-even barrier (present-only).
                        # Was static `1.5 * _rt_fees` (~1.2% @ 0.5% R/T) which
                        # blocked 95-99% of SELLs (measured, 4 episodes). Now the
                        # barrier scales with ATR% (volatility) and is tunable via
                        # ADAN_BARRIER_MULT (default 1.0). ATR is PRESENT info ->
                        # no future leak -> live-compatible. Floored at fees (never
                        # below break-even), capped at 2%. Premature exits are now
                        # judged by the Arena (Scenario B), not physically walled.
                        import os as _os_v17
                        _barrier_mult = float(_os_v17.environ.get("ADAN_BARRIER_MULT", "1.0"))
                        _atr_pct_bar = 0.0
                        try:
                            _atr_pct_bar = float(self._get_atr_pct_for_asset(asset)) or 0.0
                        except Exception:
                            _atr_pct_bar = 0.0
                        # atr_scale in [0.5, 2.0]; reference volatility 0.4% (~1 R/T unit)
                        _atr_scale = 1.0
                        if _atr_pct_bar > 1e-9:
                            _atr_scale = min(2.0, max(0.5, _atr_pct_bar / 0.004))
                        _barrier = _barrier_mult * _rt_fees * _atr_scale
                        _barrier = max(_rt_fees, min(_barrier, 0.02))
                        # Log every 500 steps to trace the dynamic barrier.
                        try:
                            if int(getattr(self, "current_step", 0)) % 500 == 0:
                                self.logger.info(
                                    f"[BREAK_EVEN_DYN] {asset}: barrier={_barrier:.4%} "
                                    f"(mult={_barrier_mult}, atr%={_atr_pct_bar:.4%}, "
                                    f"scale={_atr_scale:.2f}, rt_fees={_rt_fees:.4%})"
                                )
                        except Exception:
                            pass'''

def main():
    with io.open(ENV, "r", encoding="utf-8") as f:
        src = f.read()
    if "V17-Fix A: DYNAMIC break-even barrier" in src:
        print("PATCH_A_ALREADY_PRESENT")
        return 0
    if OLD not in src:
        print("PATCH_A_ANCHOR_NOT_FOUND")
        return 2
    src = src.replace(OLD, NEW, 1)
    with io.open(ENV, "w", encoding="utf-8") as f:
        f.write(src)
    print("PATCH_A_APPLIED")
    return 0

if __name__ == "__main__":
    sys.exit(main())
