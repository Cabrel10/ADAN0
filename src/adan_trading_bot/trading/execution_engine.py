"""
ExecutionEngine — Paper trading (virtual portfolio) and Live trading (CCXT orders).

Decodes the PPO Box(5,) action:
  action[0] = direction   ∈ [-1, 1]  → -1=SHORT, 0=HOLD, +1=LONG
  action[1] = size_pct    ∈ [-1, 1]  → mapped to [0, max_position_pct]
  action[2] = tf_pref     ∈ [-1, 1]  → timeframe preference (informational)
  action[3] = sl_pct      ∈ [-1, 1]  → mapped to [0.5%, 5%] stop-loss
  action[4] = tp_pct      ∈ [-1, 1]  → mapped to [0.5%, 10%] take-profit

Architecture:
  Paper mode:  virtual portfolio dict, no real orders, full logging
  Live mode:   ccxt.create_market_order() with kill switches
"""
from __future__ import annotations

import csv
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import ccxt
except ImportError:
    ccxt = None

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────
# Data classes
# ────────────────────────────────────────────────────────────────────────────

SLIPPAGE_BPS = 2.0  # 2 basis points per side (same as training env)


@dataclass
class TradeRecord:
    timestamp: float
    side: str       # "BUY" or "SELL"
    symbol: str
    price: float
    size_usd: float
    size_asset: float
    sl_pct: float
    tp_pct: float
    fee_usd: float
    source: str     # "paper" or "live"
    order_id: str = ""
    pnl_usd: float = 0.0
    reason: str = ""


@dataclass
class Position:
    symbol: str
    side: str
    entry_price: float
    size_usd: float
    size_asset: float
    sl_price: float
    tp_price: float
    open_time: float
    unrealized_pnl: float = 0.0

    def update_pnl(self, current_price: float):
        if self.side == "BUY":
            self.unrealized_pnl = (current_price - self.entry_price) * self.size_asset
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * self.size_asset


@dataclass
class KillSwitch:
    """Safety limits — breaching any of these shuts down the bot.

    max_position_pct = 100.0 : le bridage artificiel est supprimé.
    Le modèle s'expose selon son capital_tier d'entraînement
    (Micro ~90%, Small ~65%, Medium ~48%, High ~28%, Enterprise ~20%).
    Le minimum de 11$ par ordre est imposé dans _execute_open().
    """
    max_drawdown_pct: float = 10.0            # relative to equity high-water mark
    absolute_capital_floor_pct: float = 50.0  # equity < 50% of INITIAL capital = STOP
    max_trades_per_hour: int = 5
    max_loss_per_trade_pct: float = 3.0
    max_position_pct: float = 100.0           # Désactivé : exposition gérée par le tier
    min_trade_interval_sec: float = 30.0
    api_retry_limit: int = 3                  # max consecutive API failures before stop


class ExecutionEngine:
    """Unified execution engine for paper and live trading.

    Handles:
    - Action decoding (Box(5,) → direction, size, SL, TP)
    - Slippage simulation (2 bps directional, same as training)
    - Paper portfolio tracking
    - Live CCXT order placement
    - Kill switch enforcement
    - Trade logging to JSON
    """

    # ── SL/TP bounds per profile — MUST stay IDENTICAL to the training env ──
    # SINGLE SOURCE OF TRUTH = multi_asset_chunked_env.py `_BOUNDS` (line ~7451).
    # SYNC 2026-06-26 (FINDING #4 / capture-ratio): these were stale at the OLD
    # 8-40% bands (pre-FINDING#4). A tp_raw=0 used to decode to ~5% TP in live vs
    # ~1.25% in training -> total divergence. Re-aligned to the tight, real-wick
    # bands the PPO actually trained on. DO NOT EDIT without editing the env too.
    #   scalper : SL 0.3-1.2%  TP 0.5-2.0%
    #   intraday: SL 0.5-2.0%  TP 0.8-4.0%
    #   swing   : SL 1.0-3.5%  TP 1.5-7.0%
    #   position: SL 2.0-6.0%  TP 3.0-12.0%
    _PROFILE_BOUNDS = {
        "scalper":  {"sl": (0.003, 0.012), "tp": (0.005, 0.020)},
        "intraday": {"sl": (0.005, 0.020), "tp": (0.008, 0.040)},
        "swing":    {"sl": (0.010, 0.035), "tp": (0.015, 0.070)},
        "position": {"sl": (0.020, 0.060), "tp": (0.030, 0.120)},
    }

    def __init__(
        self,
        mode: str = "paper",          # "paper" or "live"
        exchange_id: str = "binance",
        symbol: str = "BTC/USDT",
        initial_capital: float = 20.50,
        kill_switch: KillSwitch = None,
        api_key: str = None,
        api_secret: str = None,
        testnet: bool = True,
        log_dir: str = "logs/trading",
        action_threshold: float = 0.01,
        capital_tiers: list = None,   # capital_tiers from config.yaml
        profile: str = "intraday",    # worker profile → SL/TP bounds (train coherence)
        stochastic_sltp: bool = False,  # if True, SL/TP come from ATR×regime calibrator
                                        # instead of the (collapsed) model action[3]/[4]
    ):
        self.mode = mode
        self.exchange_id = exchange_id
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.kill_switch = kill_switch or KillSwitch()
        self.action_threshold = action_threshold
        self.testnet = testnet

        # Capital tiers — used to enforce max_position_size_pct per tier
        # Format: list of dicts matching config.yaml capital_tiers structure
        self.capital_tiers = capital_tiers or []

        # Worker profile — drives SL/TP bounds IDENTICAL to the training env
        # (multi_asset_chunked_env.py:7009-7014). Single source of truth = config.
        _pmap = {"conservative": "scalper", "moderate": "intraday",
                 "balanced": "intraday", "aggressive": "swing",
                 "adaptive": "position"}
        self.profile = _pmap.get(str(profile).lower(), str(profile).lower())
        if self.profile not in self._PROFILE_BOUNDS:
            self.profile = "intraday"

        # Last HMM confidence (bull_prob from context_vector[3]); 0.5 == train default
        self._last_confidence = 0.5

        # STOCHASTIC SL/TP CALIBRATOR (separation of responsibilities) ──────
        # The PPO `tp` head collapsed (raw≈-10 → always TP=tp_lo) due to training
        # entropy collapse (see docs/SIZING_COHESION_AUDIT.md §12-§13). The `dir`
        # head is healthy (66% WR backtest). So when stochastic_sltp=True we KEEP
        # the model's direction/confidence but DERIVE SL/TP from market state
        # (ATR + HMM regime) instead of the saturated action[3]/action[4].
        self.stochastic_sltp = bool(stochastic_sltp)
        # Last derived SL/TP source for logging ("model" | "stochastic")
        self._last_sltp_source = "model"
        if self.stochastic_sltp:
            logger.info(
                "[ExecutionEngine] STOCHASTIC SL/TP calibrator ENABLED — "
                "SL/TP derived from ATR×regime (model tp/sl heads bypassed)."
            )

        # Portfolio state
        self.cash = initial_capital
        self.position: Optional[Position] = None
        self.equity_high = initial_capital
        self.trades: List[TradeRecord] = []
        self._last_trade_time = 0.0
        self._killed = False
        self._kill_reason = ""

        # Logging
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Live CCXT client
        self.live_client = None
        if mode == "live" and ccxt is not None:
            self._init_live_client(api_key, api_secret)

        logger.info(
            f"[ExecutionEngine] mode={mode} exchange={exchange_id} "
            f"symbol={symbol} capital=${initial_capital:.2f} "
            f"testnet={testnet}"
        )

    def _init_live_client(self, api_key: str, api_secret: str):
        """Initialize CCXT client for live trading."""
        # NEVER hardcode keys — must come from env vars or explicit args
        key = api_key or os.environ.get("ADAN_API_KEY", "")
        secret = api_secret or os.environ.get("ADAN_API_SECRET", "")
        passphrase = os.environ.get("ADAN_API_PASSPHRASE", "")

        if not key or not secret:
            logger.error("[ExecutionEngine] LIVE mode requires API keys. "
                         "Set ADAN_API_KEY and ADAN_API_SECRET env vars.")
            self._killed = True
            self._kill_reason = "Missing API keys"
            return

        exchange_cls = getattr(ccxt, self.exchange_id, None)
        if exchange_cls is None:
            raise ValueError(f"Unknown exchange: {self.exchange_id}")

        cfg = {
            "apiKey": key,
            "secret": secret,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        }
        if passphrase:
            cfg["password"] = passphrase
        if self.testnet:
            cfg["sandbox"] = True

        self.live_client = exchange_cls(cfg)
        if self.testnet:
            self.live_client.set_sandbox_mode(True)
        logger.info(f"[ExecutionEngine] LIVE client initialized: {self.exchange_id} "
                     f"testnet={self.testnet}")

    # ── Action decoding ────────────────────────────────────────────────

    def _get_tier_exposure_range(self, equity: float) -> tuple:
        """Return (exp_min, exp_max) as fractions (0.0–1.0) for the current tier.

        Reads exposure_range from config.yaml capital_tiers — the SAME source the
        training env uses (multi_asset_chunked_env.py:6810-6811). Falls back to the
        Micro Capital default [70, 90] used by the env when no tier matches.
        """
        if self.capital_tiers:
            for tier in sorted(self.capital_tiers, key=lambda t: t.get("min_capital", 0)):
                min_cap = float(tier.get("min_capital", 0))
                max_cap = tier.get("max_capital")
                max_cap = float(max_cap) if max_cap is not None else float("inf")
                if min_cap <= equity < max_cap:
                    er = tier.get("exposure_range", [70, 90])
                    return float(er[0]) / 100.0, float(er[1]) / 100.0
            # equity above all tiers → last tier
            last = sorted(self.capital_tiers, key=lambda t: t.get("min_capital", 0))[-1]
            er = last.get("exposure_range", [70, 90])
            return float(er[0]) / 100.0, float(er[1]) / 100.0
        # No tiers configured → env default
        return 0.70, 0.90

    def decode_action(
        self, action: np.ndarray, context_vector: np.ndarray = None,
        equity: float = None,
    ) -> Dict[str, float]:
        """Decode Box(5,) action — UNIFIED with the training env (single source of truth).

        Sizing replicates multi_asset_chunked_env.py:6908 (LINEAR_EXPO):
            confidence  = context_vector[3] (HMM bull_prob) if present, else 0.5
            target_exp  = exp_min + (exp_max − exp_min) × confidence
            size_pct    = target_exposure (then capped by tier max_position_size_pct
                          inside _execute_open, exactly like the env clamp).

        SL/TP replicate multi_asset_chunked_env.py:7009-7028 (profile bounds + R/R≥1.5).

        action[0] = direction   ∈ [-1, 1]
        action[1] = size_pref   ∈ [-1, 1]  (kept for logging; sizing now HMM-driven
                                            like training — model size is overridden
                                            by LINEAR_EXPO in the env, env:6908)
        action[2] = tf_pref     ∈ [-1, 1] (informational)
        action[3] = sl_pct      ∈ [-1, 1] → profile SL band
        action[4] = tp_pct      ∈ [-1, 1] → profile TP band
        """
        direction = float(action[0])
        raw_size = float(action[1])
        tf_pref = float(action[2])
        raw_sl = float(action[3])
        raw_tp = float(action[4])

        # ── SIZING : LINEAR_EXPO identical to training env (env:6908) ──
        equity = equity if equity is not None else self.cash
        exp_min, exp_max = self._get_tier_exposure_range(equity)

        # HMM bull probability — context_vector[3], same index as env:6903
        confidence = 0.5  # train default when HMM unavailable (env except branch:6986)
        if context_vector is not None:
            try:
                cv = np.asarray(context_vector, dtype=np.float32).flatten()
                if cv.shape[0] >= 4:
                    confidence = float(np.clip(cv[3], 0.01, 0.99))
            except Exception:
                confidence = 0.5
        self._last_confidence = confidence

        size_pct = exp_min + (exp_max - exp_min) * confidence

        # ── SL/TP : profile-specific bounds identical to env:7009-7028 ──
        b = self._PROFILE_BOUNDS.get(self.profile, self._PROFILE_BOUNDS["intraday"])
        sl_lo, sl_hi = b["sl"]
        tp_lo, tp_hi = b["tp"]
        tp_lo = max(tp_lo, 0.006)  # fee gate: 3× round-trip fees (env:7018)

        norm_sl = (raw_sl + 1.0) / 2.0
        sl_pct = float(np.clip(sl_lo + norm_sl * (sl_hi - sl_lo), sl_lo, sl_hi))
        norm_tp = (raw_tp + 1.0) / 2.0
        tp_pct = float(np.clip(tp_lo + norm_tp * (tp_hi - tp_lo), tp_lo, tp_hi))

        # Enforce R/R ≥ 1.5 exactly like the env (env:7027)
        if tp_pct < sl_pct * 1.5:
            tp_pct = float(min(sl_pct * 1.5, tp_hi))

        # ── ATR scalper SL floor — IDENTICAL to env:7523-7540 (N2 sync) ──
        # On scalper, SL must never sit below 3× market noise (~0.2% ATR),
        # else it is stopped out by noise. context_vector[0] = ATR/close ratio.
        if self.profile == "scalper":
            atr_pct_estimate = 0.002  # default 0.2% ATR (env default)
            try:
                if context_vector is not None:
                    cv = np.asarray(context_vector, dtype=np.float32).flatten()
                    if cv.shape[0] >= 1:
                        atr_pct_estimate = max(0.001, float(cv[0]))
            except Exception:
                pass
            min_scalp_sl = max(0.006, 3.0 * atr_pct_estimate)  # 3× ATR floor (env:7535)
            if sl_pct < min_scalp_sl:
                sl_pct = min_scalp_sl
                if tp_pct < sl_pct * 1.5:   # re-enforce R/R after SL bump
                    tp_pct = float(min(sl_pct * 1.5, tp_hi))

        # ── STOCHASTIC OVERRIDE ────────────────────────────────────────
        # When enabled, replace the (collapsed) model SL/TP with an
        # ATR×regime-derived pair. Direction & sizing stay model/HMM-driven.
        self._last_sltp_source = "model"
        model_sl_pct, model_tp_pct = sl_pct, tp_pct
        if self.stochastic_sltp:
            sl_pct, tp_pct = self._compute_stochastic_sltp(
                context_vector=context_vector, sl_lo=sl_lo, sl_hi=sl_hi,
                tp_lo=tp_lo, tp_hi=tp_hi, confidence=confidence,
            )
            self._last_sltp_source = "stochastic"

        return {
            "direction": direction,
            "size_pct": size_pct,
            "size_pref": raw_size,        # raw model preference (logged, not used)
            "confidence": confidence,
            "profile": self.profile,
            "exposure_range": [exp_min, exp_max],
            "tf_pref": tf_pref,
            "sl_pct": sl_pct,
            "tp_pct": tp_pct,
            "sltp_source": self._last_sltp_source,
            "model_sl_pct": model_sl_pct,   # what the model WOULD have produced
            "model_tp_pct": model_tp_pct,   # (logged for A/B comparison)
            "raw_action": action.tolist(),
        }

    def _compute_stochastic_sltp(
        self, context_vector: np.ndarray, sl_lo: float, sl_hi: float,
        tp_lo: float, tp_hi: float, confidence: float,
    ) -> tuple:
        """Derive (sl_pct, tp_pct) from market state instead of the model.

        Rationale (docs/SIZING_COHESION_AUDIT.md §13): the PPO `tp` head
        collapsed to -1.0 (always min TP). The `dir` head is healthy. We keep
        direction/sizing from the model but compute risk from observable market
        state, separating "where to trade" (model) from "how to size risk"
        (this calibrator).

        Logic:
          ATR  = context_vector[0]  (ATR/close ratio, same index as env:7044)
          regime (bull_prob) = context_vector[3]  (HMM, same index as env:6901)

          SL  = clip(ATR_MULT × ATR, sl_lo, sl_hi)      # volatility-scaled stop
          RR  = target risk/reward chosen by regime:
                  bull  (bull_prob ≥ 0.65) → RR = 2.5   (let winners run)
                  neutral                  → RR = 1.8
                  bear  (bull_prob ≤ 0.35) → RR = 1.5   (cut quickly)
          TP  = clip(SL × RR, tp_lo, tp_hi)

        SL/TP stay clamped to the SAME profile bands used in training/backtest,
        so the calibrator can only re-position WITHIN the validated envelope
        (never produce out-of-band risk). R/R≥1.5 is still enforced.
        """
        ATR_MULT = 2.0          # SL = 2× ATR (3× is the noise floor used for scalper)
        atr_pct = 0.005         # neutral default ≈ 0.5% if context unavailable
        bull_prob = float(confidence)  # already clipped [0.01, 0.99]

        if context_vector is not None:
            try:
                cv = np.asarray(context_vector, dtype=np.float32).flatten()
                if cv.shape[0] >= 1:
                    # context[0] = ATR/close ratio (env:7043-7044)
                    atr_pct = float(max(0.0005, min(0.10, abs(cv[0]))))
                if cv.shape[0] >= 4:
                    bull_prob = float(np.clip(cv[3], 0.01, 0.99))
            except Exception:
                pass

        # Regime → target risk/reward
        if bull_prob >= 0.65:
            target_rr = 2.5
        elif bull_prob <= 0.35:
            target_rr = 1.5
        else:
            target_rr = 1.8

        # SL scaled by volatility, clamped to profile band
        sl_pct = float(np.clip(ATR_MULT * atr_pct, sl_lo, sl_hi))
        # TP = SL × RR, clamped to profile band
        tp_pct = float(np.clip(sl_pct * target_rr, tp_lo, tp_hi))
        # Always keep R/R ≥ 1.5 (same invariant as the env)
        if tp_pct < sl_pct * 1.5:
            tp_pct = float(min(sl_pct * 1.5, tp_hi))

        return sl_pct, tp_pct

    # ── Kill switch checks ─────────────────────────────────────────────

    def _check_kill_switch(self) -> bool:
        """Return True if trading should stop.
        
        Three independent kill conditions:
        1. Absolute capital floor: equity < 50% of initial capital = STOP (non-negotiable)
        2. Relative drawdown: equity dropped > max_drawdown_pct from high-water mark
        3. Rate limiting: too many trades per hour or too fast
        """
        if self._killed:
            return True

        equity = self.get_equity(self.get_current_price_cached())

        # ABSOLUTE FLOOR: equity < 50% of initial capital = IMMEDIATE STOP
        # This is non-negotiable — even in testnet, prevents runaway bug loops
        capital_floor = self.initial_capital * (self.kill_switch.absolute_capital_floor_pct / 100.0)
        if equity < capital_floor:
            self._killed = True
            self._kill_reason = (
                f"ABSOLUTE_CAPITAL_FLOOR: equity ${equity:.2f} < "
                f"${capital_floor:.2f} ({self.kill_switch.absolute_capital_floor_pct}% of initial ${self.initial_capital:.2f})"
            )
            logger.error(f"[KILL SWITCH] {self._kill_reason}")
            return True

        # Relative drawdown from high-water mark
        if equity < self.equity_high:
            dd_pct = (self.equity_high - equity) / self.equity_high * 100
            if dd_pct > self.kill_switch.max_drawdown_pct:
                self._killed = True
                self._kill_reason = f"MAX_DRAWDOWN: {dd_pct:.2f}% > {self.kill_switch.max_drawdown_pct}%"
                logger.error(f"[KILL SWITCH] {self._kill_reason}")
                return True

        # Trades per hour
        now = time.time()
        recent_trades = [t for t in self.trades if now - t.timestamp < 3600]
        if len(recent_trades) >= self.kill_switch.max_trades_per_hour:
            logger.warning(f"[KILL SWITCH] Rate limit: {len(recent_trades)} trades/hour")
            return True

        # Min interval between trades
        if now - self._last_trade_time < self.kill_switch.min_trade_interval_sec:
            return True

        return False

    _price_cache: float = 0.0
    _price_cache_ts: float = 0.0

    def get_current_price_cached(self) -> float:
        """Return cached price (updated by process_tick)."""
        return self._price_cache or self.initial_capital

    # ── Core execution ─────────────────────────────────────────────────

    def process_tick(
        self,
        action: np.ndarray,
        current_price: float,
        timestamp: float = None,
        context_vector: np.ndarray = None,
    ) -> Dict[str, Any]:
        """Process one inference tick: decode action, check SL/TP, execute.

        context_vector (optional) carries the HMM regime probs (same vector used
        in training); context_vector[3] = bull_prob drives LINEAR_EXPO sizing.

        Returns a dict with tick results for logging.
        """
        ts = timestamp or time.time()
        self._price_cache = current_price
        self._price_cache_ts = ts

        equity_now = self.get_equity(current_price)
        decoded = self.decode_action(action, context_vector=context_vector,
                                     equity=equity_now)
        direction = decoded["direction"]
        result = {
            "timestamp": ts,
            "price": current_price,
            "decoded_action": decoded,
            "position": None,
            "trade": None,
            "portfolio": self._portfolio_snapshot(current_price),
            "killed": self._killed,
            "kill_reason": self._kill_reason,
        }

        # Update equity high water mark
        equity = self.get_equity(current_price)
        if equity > self.equity_high:
            self.equity_high = equity

        # Update existing position PnL
        if self.position is not None:
            self.position.update_pnl(current_price)
            result["position"] = asdict(self.position)

            # Check SL/TP
            sl_tp_result = self._check_sl_tp(current_price, ts)
            if sl_tp_result:
                result["trade"] = asdict(sl_tp_result)
                return result

        # Kill switch
        if self._check_kill_switch():
            result["killed"] = self._killed
            result["kill_reason"] = self._kill_reason
            return result

        # Decide action
        if abs(direction) < self.action_threshold:
            # HOLD — action too weak
            return result

        if direction > self.action_threshold and self.position is None:
            # BUY signal, no existing position
            trade = self._execute_open(
                side="BUY",
                price=current_price,
                size_pct=decoded["size_pct"],
                sl_pct=decoded["sl_pct"],
                tp_pct=decoded["tp_pct"],
                timestamp=ts,
            )
            if trade:
                result["trade"] = asdict(trade)

        elif direction < -self.action_threshold and self.position is not None:
            # SELL signal, close existing position
            trade = self._execute_close(
                price=current_price,
                reason="AGENT_CLOSE",
                timestamp=ts,
            )
            if trade:
                result["trade"] = asdict(trade)

        result["portfolio"] = self._portfolio_snapshot(current_price)
        return result

    # ── SL/TP check ────────────────────────────────────────────────────

    def _check_sl_tp(self, price: float, ts: float) -> Optional[TradeRecord]:
        """Check stop-loss and take-profit on the current position."""
        if self.position is None:
            return None

        pos = self.position
        if pos.side == "BUY":
            if price <= pos.sl_price:
                return self._execute_close(price, "STOP_LOSS", ts)
            if price >= pos.tp_price:
                return self._execute_close(price, "TAKE_PROFIT", ts)
        return None

    # ── Open / Close ───────────────────────────────────────────────────

    def _get_tier_cap(self, equity: float) -> float:
        """Return max_position_size_pct for the current capital tier (0.0–1.0).

        Reads self.capital_tiers (injected from config.yaml).
        Falls back to KillSwitch.max_position_pct if no tiers configured.
        """
        if not self.capital_tiers:
            return self.kill_switch.max_position_pct / 100.0

        for tier in sorted(self.capital_tiers, key=lambda t: t.get("min_capital", 0)):
            min_cap = float(tier.get("min_capital", 0))
            max_cap = tier.get("max_capital")
            max_cap = float(max_cap) if max_cap is not None else float("inf")
            if min_cap <= equity < max_cap:
                pct = float(tier.get("max_position_size_pct", 90))
                logger.debug(
                    f"[TIER_CAP] equity=${equity:.2f} → tier='{tier.get('name')}' "
                    f"max_position_size_pct={pct}%"
                )
                return pct / 100.0

        # equity above all tiers → use last tier's value
        last = sorted(self.capital_tiers, key=lambda t: t.get("min_capital", 0))[-1]
        return float(last.get("max_position_size_pct", 20)) / 100.0

    # Minimum notional par ordre (exigence de production)
    MIN_ORDER_VALUE: float = 11.0

    def _execute_open(
        self, side: str, price: float, size_pct: float,
        sl_pct: float, tp_pct: float, timestamp: float,
    ) -> Optional[TradeRecord]:
        """Open a new position (paper or live).

        SAFETY LAYERS (in order of priority):
        1. TIER CAP : size_pct vient du capital_tier d'entraînement (DBE/Oracle).
                      max_position_pct = 100% → le modèle s'expose librement.
        2. CASH RESERVE : jamais 100 % du cash (garde 5 %).
        3. MIN ORDER : notional < 11 $ → forcé à 11 $ si le capital le permet,
                       sinon ordre rejeté pour protéger le portefeuille.
        """
        if self.position is not None:
            return None  # Only one position at a time

        # Slippage : BUY at higher price, SELL at lower price
        fill_price = price * (1.0 + SLIPPAGE_BPS / 10000.0) if side == "BUY" else \
                     price * (1.0 - SLIPPAGE_BPS / 10000.0)

        # ── SAFETY LAYER 1: Respect du tier — cap issu de capital_tiers ──
        # size_pct est transmis tel quel par le modèle / DBE.
        # On plafonne au max_position_size_pct du tier courant (ex: 90% pour Micro).
        equity = self.get_equity(price)
        tier_cap = self._get_tier_cap(equity)
        size_pct = min(size_pct, tier_cap)

        # ── SAFETY LAYER 2: Cash reserve ──
        size_usd = self.cash * size_pct
        size_usd = min(size_usd, self.cash * 0.95)  # garde 5 % de cash

        # ── SAFETY LAYER 3: Minimum 11 $ par ordre ──
        if size_usd < self.MIN_ORDER_VALUE:
            if equity >= self.MIN_ORDER_VALUE:
                # Capital suffisant : on force au minimum
                size_usd = self.MIN_ORDER_VALUE
                size_pct = size_usd / self.cash if self.cash > 0 else size_pct
            else:
                logger.warning(
                    f"[ORDER_REJECTED] Capital insuffisant (${equity:.2f}) "
                    f"pour atteindre le minimum de ${self.MIN_ORDER_VALUE:.2f}"
                )
                return None

        logger.info(
            f"[SIZING] LINEAR_EXPO profile={self.profile} conf={self._last_confidence:.3f} "
            f"size_pct={size_pct:.4f} ({size_pct*100:.2f}%) "
            f"size_usd=${size_usd:.2f} of cash=${self.cash:.2f} "
            f"(tier_cap={tier_cap*100:.1f}%, min_order=${self.MIN_ORDER_VALUE:.2f})"
        )

        size_asset = size_usd / fill_price
        fee_usd = size_usd * 0.001  # 0.1% maker/taker fee

        # SL/TP prices
        if side == "BUY":
            sl_price = fill_price * (1.0 - sl_pct)
            tp_price = fill_price * (1.0 + tp_pct)
        else:
            sl_price = fill_price * (1.0 + sl_pct)
            tp_price = fill_price * (1.0 - tp_pct)

        # Execute
        if self.mode == "live" and self.live_client:
            order = self._place_live_order(side, size_asset)
            if order is None:
                return None
            fill_price = order.get("average", fill_price)
            fee_usd = order.get("fee", {}).get("cost", fee_usd)

        # Update portfolio
        self.cash -= (size_usd + fee_usd)
        self.position = Position(
            symbol=self.symbol,
            side=side,
            entry_price=fill_price,
            size_usd=size_usd,
            size_asset=size_asset,
            sl_price=sl_price,
            tp_price=tp_price,
            open_time=timestamp,
        )

        trade = TradeRecord(
            timestamp=timestamp,
            side=side,
            symbol=self.symbol,
            price=fill_price,
            size_usd=size_usd,
            size_asset=size_asset,
            sl_pct=sl_pct,
            tp_pct=tp_pct,
            fee_usd=fee_usd,
            source=self.mode,
            reason="OPEN",
        )
        self.trades.append(trade)
        self._last_trade_time = timestamp

        logger.info(
            f"[{self.mode.upper()}_TRADE] {side} {self.symbol} "
            f"size=${size_usd:.2f} price=${fill_price:.2f} "
            f"SL=${sl_price:.2f} TP=${tp_price:.2f} fee=${fee_usd:.4f}"
        )
        return trade

    def _execute_close(
        self, price: float, reason: str, timestamp: float,
    ) -> Optional[TradeRecord]:
        """Close the existing position."""
        if self.position is None:
            return None

        pos = self.position

        # Slippage: SELL at lower price
        fill_price = price * (1.0 - SLIPPAGE_BPS / 10000.0)

        # PnL
        if pos.side == "BUY":
            pnl = (fill_price - pos.entry_price) * pos.size_asset
        else:
            pnl = (pos.entry_price - fill_price) * pos.size_asset

        fee_usd = pos.size_usd * 0.001  # closing fee

        # Execute live order
        if self.mode == "live" and self.live_client:
            order = self._place_live_order("sell", pos.size_asset)
            if order:
                fill_price = order.get("average", fill_price)

        # Update portfolio
        self.cash += pos.size_usd + pnl - fee_usd
        self.position = None

        trade = TradeRecord(
            timestamp=timestamp,
            side="SELL",
            symbol=self.symbol,
            price=fill_price,
            size_usd=pos.size_usd,
            size_asset=pos.size_asset,
            sl_pct=0,
            tp_pct=0,
            fee_usd=fee_usd,
            source=self.mode,
            pnl_usd=pnl,
            reason=reason,
        )
        self.trades.append(trade)
        self._last_trade_time = timestamp

        logger.info(
            f"[{self.mode.upper()}_TRADE] CLOSE({reason}) {self.symbol} "
            f"price=${fill_price:.2f} PnL=${pnl:+.4f} fee=${fee_usd:.4f} "
            f"cash=${self.cash:.2f}"
        )
        return trade

    # ── Live order placement ───────────────────────────────────────────

    def _place_live_order(self, side: str, amount: float) -> Optional[Dict]:
        """Place a real order via CCXT. Returns order dict or None."""
        if not self.live_client:
            return None
        try:
            order = self.live_client.create_market_order(
                symbol=self.symbol,
                side=side.lower(),
                amount=amount,
            )
            logger.info(f"[LIVE ORDER] {side} {amount} {self.symbol} → {order['id']}")
            return order
        except Exception as e:
            logger.error(f"[LIVE ORDER FAILED] {e}")
            return None

    # ── Portfolio helpers ──────────────────────────────────────────────

    def get_equity(self, current_price: float) -> float:
        """Total equity = cash + unrealized position value."""
        eq = self.cash
        if self.position is not None:
            self.position.update_pnl(current_price)
            eq += self.position.size_usd + self.position.unrealized_pnl
        return eq

    def get_portfolio_state(self, current_price: float) -> np.ndarray:
        """Build portfolio_state vector (20,) — MUST mirror, slot-for-slot, the
        training layout produced by PortfolioManager.get_state_vector(), because
        the PPO policy was trained on that exact stationary-ratio layout.

        A previous version emitted an ad-hoc layout (e.g. state[7] = RAW trade
        count, growing 1,2,4,... unbounded) which is out-of-distribution for the
        network and drives the policy toward saturated actions in live/paper.
        This rebuild reproduces the canonical layout so live obs == train obs.

        Layout (identical to portfolio_manager.py:get_state_vector):
          [0] cash_ratio (clip 0-10)          [1] value_ratio (clip 0-10)
          [2] trading_pnl_pct (clip ±5)       [3] exposure_ratio (0-1)
          [4] drawdown (0-1)                  [5] sharpe/3 (-1..1)
          [6] open_positions_norm (0-1)       [7] win_rate (0-1)
          [8] profit_factor/5 (0-1)           [9] reserved (0)
          [10-14] position 1 features         [15-19] position 2 features
        """
        equity = self.get_equity(current_price)
        init_cap = max(self.initial_capital, 1e-8)
        total_value = max(equity, 1e-8)
        cash = self.cash

        # Trading PnL relative to initial capital (realised + unrealised).
        realised_pnl = float(sum(float(t.pnl_usd) for t in self.trades))
        unreal = float(self.position.unrealized_pnl) if self.position is not None else 0.0
        trading_pnl_pct = (realised_pnl + unreal) / init_cap

        # Performance metrics from closed trades (same convention as training).
        m = self.compute_metrics(current_price)
        sharpe = float(np.clip(m["sharpe_per_trade"], -3.0, 3.0))
        win_rate = float(np.clip(m["win_rate_pct"] / 100.0, 0.0, 1.0))
        pf_raw = m["profit_factor"]
        pf = 5.0 if pf_raw >= 9999.0 else float(pf_raw)
        profit_factor_norm = float(np.clip(pf, 0.0, 5.0) / 5.0)

        drawdown = ((self.equity_high - equity) / self.equity_high
                    if self.equity_high > 0 else 0.0)
        exposure_ratio = (total_value - cash) / total_value if total_value > 0 else 0.0
        open_count = 1 if self.position is not None else 0
        max_positions = max(int(getattr(self, "max_concurrent_positions", 1)), 1)

        state = [
            float(np.clip(cash / init_cap, 0.0, 10.0)),          # [0] cash_ratio
            float(np.clip(total_value / init_cap, 0.0, 10.0)),    # [1] value_ratio
            float(np.clip(trading_pnl_pct, -5.0, 5.0)),           # [2] trading_pnl_pct
            float(np.clip(exposure_ratio, 0.0, 1.0)),             # [3] exposure_ratio
            float(np.clip(drawdown, 0.0, 1.0)),                   # [4] drawdown
            float(sharpe / 3.0),                                   # [5] sharpe_norm
            float(min(open_count / max_positions, 1.0)),          # [6] positions_norm
            win_rate,                                              # [7] win_rate
            profit_factor_norm,                                    # [8] profit_factor_norm
            0.0,                                                   # [9] reserved
        ]

        # Position slots (2 × 5). The paper engine holds at most one position;
        # slot 0 = current position, slot 1 = zeros.
        for slot_idx in range(2):
            if slot_idx == 0 and self.position is not None:
                pos = self.position
                entry_p = max(pos.entry_price, 1e-8)
                direction = 1.0 if pos.side == "BUY" else -1.0
                notional = abs(pos.size_usd)
                # steps-in-position proxy: seconds held / (1000 * interval≈300s).
                held_s = max(0.0, (self._price_cache_ts - pos.open_time)
                             if hasattr(self, "_price_cache_ts") else 0.0)
                steps_held = held_s / 300.0
                # Derive SL/TP pct from stored prices.
                sl_pct = abs(entry_p - pos.sl_price) / entry_p
                tp_pct = abs(pos.tp_price - entry_p) / entry_p
                # NOTE (FINDING #4 / revue): ces clips 0.2 / 0.5 ne FIXENT PAS le TP/SL
                # d'un ordre — ils NORMALISENT une feature SL/TP pour le vecteur d'etat
                # (live trading uniquement; execution_engine n'est PAS importe en training).
                # La source de verite d'execution = _BOUNDS (env). Ne pas confondre.
                state.extend([
                    float(np.clip((current_price - entry_p) / entry_p * direction, -0.5, 0.5)),
                    float(np.clip(notional / total_value, 0.0, 1.0)),
                    float(np.clip(steps_held / 1000.0, 0.0, 1.0)),
                    float(np.clip(sl_pct, 0.0, 0.2)),
                    float(np.clip(tp_pct, 0.0, 0.5)),
                ])
            else:
                state.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        assert len(state) == 20, f"portfolio_state must be 20 dims, got {len(state)}"
        return np.array(state, dtype=np.float32)

    def _portfolio_snapshot(self, price: float) -> Dict[str, Any]:
        return {
            "cash": round(self.cash, 4),
            "equity": round(self.get_equity(price), 4),
            "drawdown_pct": round(
                (self.equity_high - self.get_equity(price)) / self.equity_high * 100
                if self.equity_high > 0 else 0, 2
            ),
            "n_trades": len(self.trades),
            "has_position": self.position is not None,
        }

    # ── Reporting ─────────────────────────────────────────────────────

    def compute_metrics(self, price: float = None) -> Dict[str, float]:
        """Compute Sharpe / Profit Factor / Drawdown / WinRate from closed trades.

        Only trades that REALISED PnL (reason in STOP_LOSS/TAKE_PROFIT/AGENT_CLOSE,
        i.e. pnl_usd != 0) are counted as round-trip results — consistent with the
        backtest metric convention.
        """
        price = price if price is not None else (self._price_cache or 0.0)
        pnls = [float(t.pnl_usd) for t in self.trades if abs(float(t.pnl_usd)) > 1e-12]
        n = len(pnls)
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        gross_win = sum(wins)
        gross_loss = abs(sum(losses))
        # Per-trade return relative to initial capital (proxy for Sharpe)
        rets = [p / self.initial_capital for p in pnls] if self.initial_capital > 0 else []
        sharpe = 0.0
        if len(rets) >= 2:
            mean = sum(rets) / len(rets)
            var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
            std = var ** 0.5
            sharpe = (mean / std) if std > 1e-12 else 0.0
        profit_factor = (gross_win / gross_loss) if gross_loss > 1e-12 else (
            float("inf") if gross_win > 0 else 0.0)
        win_rate = (len(wins) / n * 100.0) if n > 0 else 0.0
        max_dd = ((self.equity_high - min(self.get_equity(price), self.equity_high))
                  / self.equity_high * 100 if self.equity_high > 0 else 0.0)
        return {
            "n_closed_trades": n,
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else 9999.0,
            "sharpe_per_trade": round(sharpe, 4),
            "gross_win_usd": round(gross_win, 4),
            "gross_loss_usd": round(gross_loss, 4),
            "max_drawdown_pct": round(max_dd, 2),
            "expectancy_usd": round(sum(pnls) / n, 4) if n > 0 else 0.0,
        }

    def export_trades_csv(self, filename: str = None) -> str:
        """Export every trade to CSV (audit / post-mortem of the 72h run)."""
        if filename is None:
            filename = f"trades_{self.mode}_{self.exchange_id}_{int(time.time())}.csv"
        path = self.log_dir / filename
        fields = ["timestamp", "side", "symbol", "price", "size_usd", "size_asset",
                  "sl_pct", "tp_pct", "fee_usd", "pnl_usd", "reason", "source", "order_id"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for t in self.trades:
                row = {k: getattr(t, k, "") for k in fields}
                w.writerow(row)
        logger.info(f"[ExecutionEngine] Trades CSV exported: {path} ({len(self.trades)} rows)")
        return str(path)

    def save_report(self, filename: str = None) -> str:
        """Save trading session report to JSON (+ metrics) and export trades CSV."""
        if filename is None:
            filename = f"trading_{self.mode}_{self.exchange_id}_{int(time.time())}.json"

        price = self._price_cache or 0.0
        metrics = self.compute_metrics(price)
        report = {
            "mode": self.mode,
            "exchange": self.exchange_id,
            "symbol": self.symbol,
            "profile": self.profile,
            "initial_capital": self.initial_capital,
            "final_equity": round(self.get_equity(price), 4),
            "return_pct": round(
                (self.get_equity(price) - self.initial_capital) / self.initial_capital * 100, 4
            ),
            "n_trades": len(self.trades),
            "metrics": metrics,
            "trades": [asdict(t) for t in self.trades],
            "killed": self._killed,
            "kill_reason": self._kill_reason,
            "max_drawdown_pct": metrics["max_drawdown_pct"],
        }

        path = self.log_dir / filename
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"[ExecutionEngine] Report saved: {path}")
        logger.info(
            f"[METRICS] trades={metrics['n_closed_trades']} WR={metrics['win_rate_pct']}% "
            f"PF={metrics['profit_factor']} Sharpe={metrics['sharpe_per_trade']} "
            f"DD={metrics['max_drawdown_pct']}% E=${metrics['expectancy_usd']}"
        )
        # CSV export alongside the JSON
        csv_name = filename.replace(".json", ".csv").replace("trading_", "trades_")
        self.export_trades_csv(csv_name)
        return str(path)
