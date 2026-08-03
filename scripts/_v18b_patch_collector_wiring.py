#!/usr/bin/env python3
"""V18-FINAL io-patcher: wire ArenaCollector.record() on real trade closures.

The tutor's critical note: the Collector must be fed by REAL closed trades,
otherwise the Predictor never learns from production reality.

Hook point: `_future_contrib_from_receipts` (line ~6581). For every closed
receipt it already computes:
    entry_idx, df, mfe, mae, steps_held, mfe_residual, pnl_gross.

We add, right after `total += float(contrib)`:
  1. Build PresentState FROM THE ENTRY ROW (df.iloc[entry_idx]) -> the state
     the market was in AT OPEN (not exit). This is the golden rule: predict
     the future from the entry, not the exit.
  2. Derive optimal params via ArenaCollector.optimal_params_from_future(...).
  3. collector.record(state, params, meta).

Everything is best-effort (never breaks training). Idempotent.
"""
import io
import sys

ENV = "src/adan_trading_bot/environment/multi_asset_chunked_env.py"

with io.open(ENV, "r", encoding="utf-8") as f:
    src = f.read()

if "V18-FINAL: collecte reelle sur cloture" in src:
    print("[V18B] already wired, skipping.")
    sys.exit(0)

ANCHOR = "                total += float(contrib)\n"
if ANCHOR not in src:
    print("[V18B] ERROR: anchor 'total += float(contrib)' not found.")
    sys.exit(1)

WIRE = '''                # ============================================================
                # V18-FINAL: collecte reelle sur cloture de trade.
                # Le Collector transforme (etat present A L'ENTREE -> params
                # optimaux ex-post) en echantillon supervise. REGLE D'OR: on
                # capture l'etat AU MOMENT DE L'ENTREE (df.iloc[entry_idx]),
                # pas a la sortie -> le Predictor apprend a anticiper depuis
                # l'entree. Best-effort: ne casse jamais l'entrainement.
                # ============================================================
                try:
                    _col = getattr(self, "_arena_collector", None)
                    if _col is not None and getattr(_col, "enabled", False):
                        from adan_trading_bot.arena_predictor import (
                            PresentState as _PS_col, ArenaCollector as _AC_col,
                        )
                        _entry_row = {}
                        try:
                            _entry_row = df.iloc[int(entry_idx)].to_dict()
                        except Exception:
                            _entry_row = {}
                        _entry_price = float(_entry_row.get("close", receipt.get("entry_price", 0.0)) or 0.0)
                        _st_col = _PS_col.from_market_row(_entry_row, timeframe=str(tf or "5m"))
                        _rtf_col = 0.008
                        try:
                            _rtf_col = 2.0 * float(getattr(self.portfolio_manager, "fee_pct", 0.002))
                        except Exception:
                            _rtf_col = 0.008
                        _pnl_net_col = float(
                            receipt.get("pnl_net", receipt.get("pnl", receipt.get("pnl_gross", 0.0))) or 0.0
                        )
                        _params_col = _AC_col.optimal_params_from_future(
                            entry_price=_entry_price,
                            mfe=float(mfe),
                            mae=float(mae),
                            steps_held=int(steps_held),
                            mfe_residual=mfe_residual,
                            round_trip_fees=_rtf_col,
                            pnl_net=_pnl_net_col,
                        )
                        _col.record(_st_col, _params_col, meta={
                            "asset": str(asset),
                            "tf": str(tf or "5m"),
                            "open_step": int(receipt.get("open_step", -1)),
                            "steps_held": int(steps_held),
                            "reason": str(receipt.get("reason", receipt.get("close_reason", "")) or ""),
                            "pnl_net": _pnl_net_col,
                            "global_step": int(cur_global),
                        })
                except Exception:
                    pass
'''

src = src.replace(ANCHOR, ANCHOR + WIRE, 1)

with io.open(ENV, "w", encoding="utf-8") as f:
    f.write(src)
print("[V18B] collector wiring applied.")
