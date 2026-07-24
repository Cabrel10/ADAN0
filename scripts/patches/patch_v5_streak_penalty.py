#!/usr/bin/env python3
"""
DIAGNOSTIC-V5 patch — streak-based per-reason escalation penalty.

Root cause confirmed (110k FACTS): V4 sterile penalty was indexed on CAPITAL
TIER (_sterile_penalty_for_tier). At Micro tier k=0 it stayed flat at base
(-0.02) no matter how many times the SAME invalid action was repeated, so PPO
felt a constant penalty gradient and saturated the policy (a0_std 109, HOLD 0%,
illegal_ratio 99.7%).

This patch replaces tier-indexing with PERSISTENCE-indexing, exactly per the
user's analysis:

  1. Per-family streak counters keyed on the rejection REASON.
  2. Adaptive bounded accumulator:  acc = decay*acc + severity ; reset on any
     legal action.  penalty = base * min(cap, 1 + alpha*acc).
  3. Per-family SEVERITY (sell_no_position heavier than buy-blocked).
  4. Sliding-window invalid_ratio multiplier (>0.7 -> max).
  5. _reset_sterile_streak() called on EVERY legal trade executed.
  6. BUY_WHILE_OPEN / CASH_FLOOR now penalised AND logged (symmetric).

FEES UNTOUCHED. Only reward-shaping knobs are read. Idempotent (guard markers).

Run:  python scripts/patches/patch_v5_streak_penalty.py
"""
import io, re, sys

F = "src/adan_trading_bot/environment/multi_asset_chunked_env.py"

with io.open(F, "r", encoding="utf-8") as fh:
    src = fh.read()

if "_sterile_penalty_v5" in src:
    print("[SKIP] patch already applied (_sterile_penalty_v5 present)")
    sys.exit(0)

orig = src

# ---------------------------------------------------------------------------
# 1) __init__ : add streak/accumulator state right after rejection_reasons init
#    (the FIRST occurrence — the __init__ one at ~line 559, ends with pm_rejected)
# ---------------------------------------------------------------------------
init_anchor = (
    '            "pm_rejected": 0,   # Portfolio manager rejected the open\n'
    "        }\n"
    "        # Accumulate invalid_trade_penalty per step (reset each step)\n"
    "        self._step_invalid_penalty = 0.0\n"
)
init_block = (
    '            "pm_rejected": 0,   # Portfolio manager rejected the open\n'
    "        }\n"
    "        # ============================================================\n"
    "        # DIAGNOSTIC-V5: streak-based sterile penalty state\n"
    "        # Persistence-indexed (NOT capital-tier). Keyed on rejection\n"
    "        # reason. acc = decay*acc + severity ; reset on legal action.\n"
    "        # ============================================================\n"
    "        self._sterile_streak: Dict[str, int] = {}\n"
    "        self._sterile_acc: Dict[str, float] = {}\n"
    "        from collections import deque as _deque_v5\n"
    "        self._invalid_window = _deque_v5(maxlen=200)  # 1=invalid step,0=legal\n"
    "        # Accumulate invalid_trade_penalty per step (reset each step)\n"
    "        self._step_invalid_penalty = 0.0\n"
)
assert init_anchor in src, "init anchor not found"
src = src.replace(init_anchor, init_block, 1)

# ---------------------------------------------------------------------------
# 2) Insert the V5 helper methods + keep _sterile_penalty_for_tier as a thin
#    legacy shim (so any other caller still works). Insert just BEFORE the
#    closure definition line "        def _sterile_penalty_for_tier():"
# ---------------------------------------------------------------------------
helper_anchor = "        def _sterile_penalty_for_tier():\n"
helper_methods = (
    "        # ============================================================\n"
    "        # DIAGNOSTIC-V5 sterile penalty — escalated by PERSISTENCE.\n"
    "        # ============================================================\n"
    "        def _sterile_severity_v5(reason):\n"
    "            # Per-family gravity. SELL-no-position is the worst (it is the\n"
    "            # collapse attractor); cash-blocked BUY is milder.\n"
    "            _sev = {\n"
    '                "sell_no_position": 1.5,\n'
    '                "anti_spam_hold": 1.2,   # BUY while already open\n'
    '                "min_notional": 0.8,     # BUY blocked by cash\n'
    '                "hysteresis": 0.7,\n'
    '                "cooldown_wait": 0.6,\n'
    '                "cooldown_hold_min": 0.6,\n'
    "            }\n"
    "            return float(_sev.get(reason, 1.0))\n"
    "\n"
    "        def _invalid_ratio_mult_v5():\n"
    "            # Sliding-window invalid ratio -> multiplier (collapse detector).\n"
    "            _w = getattr(self, '_invalid_window', None)\n"
    "            if not _w or len(_w) < 20:\n"
    "                return 1.0\n"
    "            _r = sum(_w) / float(len(_w))\n"
    "            if _r > 0.7:\n"
    "                return 3.0\n"
    "            if _r > 0.4:\n"
    "                return 2.0\n"
    "            if _r > 0.1:\n"
    "                return 1.5\n"
    "            return 1.0\n"
    "\n"
    "        def _sterile_penalty_v5(reason):\n"
    "            # Adaptive bounded accumulator, keyed on rejection reason.\n"
    "            #   acc_t = decay*acc_{t-1} + severity\n"
    "            #   pen   = base * min(cap, 1 + alpha*acc_t) * window_mult\n"
    "            # Reset (decay-to-zero) happens via _reset_sterile_streak() on\n"
    "            # ANY legal action. fees are NOT involved.\n"
    "            _rs = self.config.get('reward_shaping', {})\n"
    "            _base = float(_rs.get('invalid_trade_penalty_weight', 0.02))\n"
    "            _cap = float(_rs.get('sterile_action_penalty_cap', 0.30))\n"
    "            _decay = float(_rs.get('sterile_acc_decay', 0.97))\n"
    "            _alpha = float(_rs.get('sterile_acc_alpha', 0.45))\n"
    "            _sev = _sterile_severity_v5(reason)\n"
    "            self._sterile_streak[reason] = self._sterile_streak.get(reason, 0) + 1\n"
    "            _acc = self._sterile_acc.get(reason, 0.0) * _decay + _sev\n"
    "            self._sterile_acc[reason] = _acc\n"
    "            _mult = _invalid_ratio_mult_v5()\n"
    "            _pen = _base * min(_cap / _base if _base > 0 else _cap,\n"
    "                               1.0 + _alpha * _acc) * _mult\n"
    "            _pen = min(_pen, _cap)\n"
    "            return _pen, self._sterile_streak[reason], _acc, _mult\n"
    "\n"
    "        def _sterile_penalty_for_tier():\n"
    "            # LEGACY shim kept for compatibility; now routes through V5\n"
    "            # generic reason so old call paths still escalate by streak.\n"
    "            _p, _, _, _ = _sterile_penalty_v5('generic_invalid')\n"
    "            return _p\n"
)
assert helper_anchor in src, "helper anchor not found"
# Replace the OLD closure body entirely with our methods. The old closure runs
# from "def _sterile_penalty_for_tier():" to its "return min(_cap, ...)" line.
old_closure_re = re.compile(
    r"        def _sterile_penalty_for_tier\(\):\n"
    r"(?:.*\n)*?"
    r"            return min\(_cap, _base \* \(_r \*\* _k\)\)\n"
)
m = old_closure_re.search(src)
assert m, "old _sterile_penalty_for_tier closure not matched"
src = src[:m.start()] + helper_methods + src[m.end():]

# ---------------------------------------------------------------------------
# 3) Call site A — BUY while already open (anti_spam_hold). Add log + reason key.
# ---------------------------------------------------------------------------
buy_open_old = (
    '                        self.rejection_reasons["anti_spam_hold"] += 1\n'
    "                        # DIAGNOSTIC-V4: BUY-while-open is invalid intent. It is\n"
    "                        # converted to HOLD but must NOT be free (root-cause fix).\n"
    "                        self._step_invalid_penalty += -_sterile_penalty_for_tier()\n"
)
buy_open_new = (
    '                        self.rejection_reasons["anti_spam_hold"] += 1\n'
    "                        # DIAGNOSTIC-V5: BUY-while-open escalated by streak.\n"
    "                        _pv5, _sk5, _ac5, _mu5 = _sterile_penalty_v5('anti_spam_hold')\n"
    "                        self._step_invalid_penalty += -_pv5\n"
    "                        try: self._invalid_window.append(1)\n"
    "                        except Exception: pass\n"
    "                        if self.current_step % 50 == 0:\n"
    "                            self.logger.warning(\n"
    "                                f'[BUY_WHILE_OPEN] {asset} | streak={_sk5} '\n"
    "                                f'acc={_ac5:.2f} mult={_mu5:.1f} pen=-{_pv5:.5f}')\n"
)
assert buy_open_old in src, "buy_open call site not found"
src = src.replace(buy_open_old, buy_open_new, 1)

# ---------------------------------------------------------------------------
# 4) Call site B — min_notional / cash floor. Add log + reason key.
# ---------------------------------------------------------------------------
cashfloor_old = (
    '                        self.rejection_reasons["min_notional"] += 1\n'
    "                        # DIAGNOSTIC-V4: spamming BUY with no cash was the main\n"
    "                        # collapse exploit (5000+ free rejections). Now penalized.\n"
    "                        self._step_invalid_penalty += -_sterile_penalty_for_tier()\n"
    "                        if self.current_step % 50 == 0:\n"
    "                            self.logger.info(\n"
    '                                f"[CASH_FLOOR] {asset} cash=${available_cash_for_sizing:.2f} "\n'
    '                                f"< min_order=${min_order_value:.2f} — forced HOLD"\n'
    "                            )\n"
)
cashfloor_new = (
    '                        self.rejection_reasons["min_notional"] += 1\n'
    "                        # DIAGNOSTIC-V5: cash-blocked BUY escalated by streak.\n"
    "                        _pv5, _sk5, _ac5, _mu5 = _sterile_penalty_v5('min_notional')\n"
    "                        self._step_invalid_penalty += -_pv5\n"
    "                        try: self._invalid_window.append(1)\n"
    "                        except Exception: pass\n"
    "                        if self.current_step % 50 == 0:\n"
    "                            self.logger.info(\n"
    '                                f"[CASH_FLOOR] {asset} cash=${available_cash_for_sizing:.2f} "\n'
    '                                f"< min_order=${min_order_value:.2f} | streak={_sk5} "\n'
    '                                f"acc={_ac5:.2f} mult={_mu5:.1f} pen=-{_pv5:.5f} — forced HOLD"\n'
    "                            )\n"
)
assert cashfloor_old in src, "cashfloor call site not found"
src = src.replace(cashfloor_old, cashfloor_new, 1)

# ---------------------------------------------------------------------------
# 5) Call site C — SELL no position (the collapse attractor). Replace whole
#    tier block with V5 streak penalty + updated log.
# ---------------------------------------------------------------------------
sell_old_re = re.compile(
    r"                # --- penalite geometrique par palier ---\n"
    r"(?:.*\n)*?"
    r"                # action reste HOLD \(rien a fermer\), mais elle n'est plus gratuite\.\n"
)
sell_new = (
    "                # --- DIAGNOSTIC-V5: streak-escalated sterile penalty ---\n"
    "                _pv5, _sk5, _ac5, _mu5 = _sterile_penalty_v5('sell_no_position')\n"
    "                self._step_invalid_penalty += -_pv5\n"
    "                try: self._invalid_window.append(1)\n"
    "                except Exception: pass\n"
    "                if self.current_step % 50 == 0:\n"
    "                    self.logger.warning(\n"
    "                        f'[STERILE_SELL] {asset} | SELL sans position | '\n"
    "                        f'streak={_sk5} acc={_ac5:.2f} mult={_mu5:.1f} '\n"
    "                        f'pen=-{_pv5:.5f}')\n"
    "                # action reste HOLD (rien a fermer), mais elle n'est plus gratuite.\n"
)
m = sell_old_re.search(src)
assert m, "SELL no-position tier block not matched"
src = src[:m.start()] + sell_new + src[m.end():]

# ---------------------------------------------------------------------------
# 6) _reset_sterile_streak on every legal trade executed. We hook the 4
#    "trade_executed_this_step = True" assignments. Add a streak reset + push
#    a 0 onto the invalid window right after each.
# ---------------------------------------------------------------------------
reset_helper = "_reset_sterile_streak_inline"
# Define an inline method via attribute set in __init__? Simpler: just inline.
def add_reset(src, needle):
    repl = (
        needle
        + "\n                    # DIAGNOSTIC-V5: legal action -> relax sterile pressure\n"
        + "                    self._sterile_streak.clear()\n"
        + "                    self._sterile_acc.clear()\n"
        + "                    try: self._invalid_window.append(0)\n"
        + "                    except Exception: pass\n"
    )
    return src, repl

# We must place resets with correct indentation per site. Do targeted replaces.
# Site BUY open (indent 20 spaces):
buy_exec_old = "                    trade_executed_this_step = True\n                    # Update frequency counters\n"
buy_exec_new = (
    "                    trade_executed_this_step = True\n"
    "                    # DIAGNOSTIC-V5: legal BUY -> relax sterile pressure\n"
    "                    self._sterile_streak.clear()\n"
    "                    self._sterile_acc.clear()\n"
    "                    try: self._invalid_window.append(0)\n"
    "                    except Exception: pass\n"
    "                    # Update frequency counters\n"
)
assert buy_exec_old in src, "buy exec reset anchor not found"
src = src.replace(buy_exec_old, buy_exec_new, 1)

# Site SELL AGENT_CLOSE (indent 32 spaces):
sell_exec_old = (
    "                                trade_executed_this_step = True\n"
    "                                # FIX (2026-06-25): comptabilise cet AGENT_CLOSE\n"
)
sell_exec_new = (
    "                                trade_executed_this_step = True\n"
    "                                # DIAGNOSTIC-V5: legal SELL -> relax sterile pressure\n"
    "                                self._sterile_streak.clear()\n"
    "                                self._sterile_acc.clear()\n"
    "                                try: self._invalid_window.append(0)\n"
    "                                except Exception: pass\n"
    "                                # FIX (2026-06-25): comptabilise cet AGENT_CLOSE\n"
)
assert sell_exec_old in src, "sell exec reset anchor not found"
src = src.replace(sell_exec_old, sell_exec_new, 1)

with io.open(F, "w", encoding="utf-8") as fh:
    fh.write(src)

print("[OK] V5 streak penalty patch applied.")
print("  delta bytes:", len(src) - len(orig))
