#!/usr/bin/env python3
"""V19 io-patcher (part 2): replace arbitrary 2% barrier cap with observed
MFE percentile, and track MFE distribution for adaptive calibration.

Tutor / V19 spec:
  - The hard 2% barrier cap is arbitrary. Replace it with percentile-95 of the
    MFE actually observed on real trades (a data-driven ceiling). Until enough
    samples are seen, fall back to 2%.
  - Maintain a rolling MFE buffer (self._mfe_observed) fed by the same
    _future_contrib_from_receipts loop that computes MFE.

Env-var ADAN_V19_DYNAMIC_CAP (default 1) toggles the behavior.
Idempotent. String-replace (Edit fails on this 498KB file).
"""
import io
import sys

ENV = "src/adan_trading_bot/environment/multi_asset_chunked_env.py"

with io.open(ENV, "r", encoding="utf-8") as f:
    src = f.read()

if "V19: dynamic MFE-percentile cap" in src:
    print("[V19B] already patched, skipping.")
    sys.exit(0)

# --- Patch A: helper method + rolling buffer init -----------------------
# Insert a helper right before _future_contrib_from_receipts definition.
DEF_ANCHOR = "    def _future_contrib_from_receipts(self) -> float:\n"
HELPER = '''    def _v19_barrier_cap(self) -> float:
        """V19: dynamic MFE-percentile cap (remplace le plafond arbitraire 2%).

        Retourne le percentile-95 des MFE reellement observees si assez
        d'echantillons (>=30), sinon le repli 0.02. Borne dans [0.008, 0.05]
        pour rester sain. Active par ADAN_V19_DYNAMIC_CAP (def 1).
        """
        import os as _os_cap
        if _os_cap.environ.get("ADAN_V19_DYNAMIC_CAP", "1") != "1":
            return 0.02
        buf = getattr(self, "_mfe_observed", None)
        if not buf or len(buf) < 30:
            return 0.02
        try:
            import numpy as _np_cap
            p95 = float(_np_cap.percentile(_np_cap.asarray(buf, dtype=float), 95))
            return float(max(0.008, min(0.05, p95)))
        except Exception:
            return 0.02

'''

if "def _v19_barrier_cap" in src:
    print("[V19B] helper already present.")
elif DEF_ANCHOR not in src:
    print("[V19B] ERROR: _future_contrib_from_receipts def not found.")
    sys.exit(1)
else:
    src = src.replace(DEF_ANCHOR, HELPER + DEF_ANCHOR, 1)
    print("[V19B] helper _v19_barrier_cap inserted.")

# --- Patch B: feed the MFE buffer inside the receipts loop --------------
# After mfe/mae computed. Anchor on the steps_held line right after MFE calc.
FEED_ANCHOR = "                # duree en steps\n                steps_held = max(0, cur_global - open_step)\n"
FEED = '''                # V19: alimente le buffer MFE observe (calibration du cap).
                try:
                    if not hasattr(self, "_mfe_observed"):
                        from collections import deque as _dq_v19
                        self._mfe_observed = _dq_v19(maxlen=2000)
                    if mfe is not None and float(mfe) >= 0.0:
                        self._mfe_observed.append(float(mfe))
                except Exception:
                    pass
'''
if FEED_ANCHOR not in src:
    print("[V19B] ERROR: steps_held anchor not found.")
    sys.exit(1)
src = src.replace(FEED_ANCHOR, FEED_ANCHOR + FEED, 1)
print("[V19B] MFE buffer feed inserted.")

# --- Patch C: use dynamic cap in V17 barrier clamp ----------------------
OLD_CLAMP = "                        _barrier = max(_rt_fees, min(_barrier, 0.02))\n"
NEW_CLAMP = '''                        # V19: dynamic MFE-percentile cap (remplace le 2% arbitraire).
                        _cap_v19 = 0.02
                        try:
                            _cap_v19 = float(self._v19_barrier_cap())
                        except Exception:
                            _cap_v19 = 0.02
                        _barrier = max(_rt_fees, min(_barrier, _cap_v19))
'''
if OLD_CLAMP not in src:
    print("[V19B] ERROR: barrier clamp line not found.")
    sys.exit(1)
src = src.replace(OLD_CLAMP, NEW_CLAMP, 1)
print("[V19B] dynamic cap wired into V17 clamp.")

# --- Patch D: use dynamic cap in V18 arena override ----------------------
OLD_ARENA_CLAMP = "                                    _barrier = max(_rt_fees, min(float(_be_arena), 0.02))\n"
NEW_ARENA_CLAMP = '''                                    _barrier = max(_rt_fees, min(float(_be_arena), _cap_v19 if 'in' in dir() or True else 0.02))
'''
# Simpler robust replacement: reference a locally-recomputed cap.
NEW_ARENA_CLAMP = '''                                    try:
                                        _cap_a19 = float(self._v19_barrier_cap())
                                    except Exception:
                                        _cap_a19 = 0.02
                                    _barrier = max(_rt_fees, min(float(_be_arena), _cap_a19))
'''
if OLD_ARENA_CLAMP not in src:
    print("[V19B] WARN: arena clamp line not found (skipping D).")
else:
    src = src.replace(OLD_ARENA_CLAMP, NEW_ARENA_CLAMP, 1)
    print("[V19B] dynamic cap wired into V18 arena override.")

with io.open(ENV, "w", encoding="utf-8") as f:
    f.write(src)
print("[V19B] File written.")
