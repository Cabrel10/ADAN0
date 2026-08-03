#!/usr/bin/env python3
"""V18 io-patcher: integrate ArenaEstimator + ArenaCollector into the env.

The env file is ~498KB; the Edit tool fails on it, so we use string-replace.
Idempotent: re-running detects the markers and does nothing.

Two insertions:
  1. Init block: after the RewardBridge init (self._reward_bridge = None on
     exception), create self._arena_estimator and self._arena_collector,
     lazily/robustly (never crash the env).
  2. Barrier override: inside the V17-Fix A dynamic-barrier block, if the
     Arena estimator is active, override `_barrier` with the Arena's
     adaptive break-even (mean + k*std) derived from the PRESENT market row.
"""
import io
import sys

ENV = "src/adan_trading_bot/environment/multi_asset_chunked_env.py"

with io.open(ENV, "r", encoding="utf-8") as f:
    src = f.read()

orig = src

# ---------------------------------------------------------------------- #
# INSERTION 1 — init arena estimator + collector after RewardBridge block
# ---------------------------------------------------------------------- #
MARKER1 = "self.last_trade_timestamps = {\"5m\": None, \"1h\": None, \"4h\": None}"
INIT_BLOCK = '''        # ============================================================
        # V18 — Arena Bayesien Predictif (present-only, live-safe).
        # Estimator: remplace la barriere/profils fixes par un modele appris.
        # Collector: enregistre (present -> params optimaux ex-post) en JSONL.
        # Tout est best-effort: jamais casser l'env si torch/modele absent.
        # Actives par env-vars ADAN_ARENA_PREDICT=1 / ADAN_ARENA_COLLECT=1.
        # ============================================================
        self._arena_estimator = None
        self._arena_collector = None
        try:
            from adan_trading_bot.arena_predictor import ArenaEstimator, ArenaCollector
            _rtf_arena = 0.004
            try:
                _rtf_arena = 2.0 * float(getattr(self.portfolio_manager, "fee_pct", 0.002))
            except Exception:
                _rtf_arena = 0.004
            self._arena_estimator = ArenaEstimator(round_trip_fees=_rtf_arena)
            self._arena_collector = ArenaCollector()
            if self._arena_estimator.enabled:
                logger.info("[ARENA_V18] Estimator init (active=%s, path=%s)",
                            self._arena_estimator.is_active, self._arena_estimator.model_path)
            if self._arena_collector.enabled:
                logger.info("[ARENA_V18] Collector ACTIVE -> %s", self._arena_collector.out_path)
        except Exception as _e_arena:  # pragma: no cover
            logger.warning("[ARENA_V18] init non disponible: %r", _e_arena)
            self._arena_estimator = None
            self._arena_collector = None

'''

if "V18 — Arena Bayesien Predictif" in src:
    print("[V18] INSERTION 1 already present, skipping.")
else:
    idx = src.find(MARKER1)
    if idx == -1:
        print("[V18] ERROR: marker1 not found:", MARKER1)
        sys.exit(1)
    src = src[:idx] + INIT_BLOCK + src[idx:]
    print("[V18] INSERTION 1 applied.")

# ---------------------------------------------------------------------- #
# INSERTION 2 — arena barrier override inside FIX-A block
# ---------------------------------------------------------------------- #
ANCHOR2 = "                        _barrier = max(_rt_fees, min(_barrier, 0.02))\n"
OVERRIDE = '''                        # V18: si l'Arena predictif est actif, la barriere
                        # devient la break-even ADAPTATIVE (mean + k*std) issue
                        # du modele appris sur l'etat PRESENT. Repli sur la
                        # barriere V17 (ATR-scale) si modele indisponible.
                        try:
                            _arena = getattr(self, "_arena_estimator", None)
                            if _arena is not None and getattr(_arena, "is_active", False):
                                from adan_trading_bot.arena_predictor import PresentState as _PS_v18
                                _row_v18 = {}
                                _tf_v18 = "5m"
                                try:
                                    _df_v18, _tf_got = self._get_chunk_df_for_asset(asset, "5m")
                                    if _df_v18 is not None and hasattr(_df_v18, "iloc") and len(_df_v18) > 0:
                                        _row_v18 = _df_v18.iloc[-1].to_dict()
                                        _tf_v18 = _tf_got or "5m"
                                except Exception:
                                    _row_v18 = {}
                                _st_v18 = _PS_v18.from_market_row(_row_v18, timeframe=_tf_v18)
                                _be_arena = _arena.estimate_break_even(_st_v18)
                                if _be_arena and _be_arena > 0:
                                    _barrier = max(_rt_fees, min(float(_be_arena), 0.02))
                                    self._last_arena_barrier = _barrier
                        except Exception:
                            pass
'''

if "V18: si l'Arena predictif est actif" in src:
    print("[V18] INSERTION 2 already present, skipping.")
else:
    if ANCHOR2 not in src:
        print("[V18] ERROR: anchor2 not found (barrier clamp line).")
        sys.exit(1)
    src = src.replace(ANCHOR2, ANCHOR2 + OVERRIDE, 1)
    print("[V18] INSERTION 2 applied.")

if src != orig:
    with io.open(ENV, "w", encoding="utf-8") as f:
        f.write(src)
    print("[V18] File written.")
else:
    print("[V18] No changes (idempotent).")
