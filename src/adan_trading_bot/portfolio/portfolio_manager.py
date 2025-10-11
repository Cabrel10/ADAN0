#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module de gestion de portefeuille pour le bot de trading ADAN.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid

import numpy as np

from ..performance.metrics import PerformanceMetrics
from ..utils.smart_logger import create_smart_logger

logger = logging.getLogger(__name__)


class Position:
    """Représente une position de trading unique."""

    def __init__(self):
        self.is_open = False
        self.asset = ""
        self.entry_price = 0.0
        self.size = 0.0  # En unités de l'actif
        self.stop_loss_pct = 0.0
        self.take_profit_pct = 0.0
        self.open_step = 0
        self.current_price = 0.0
        self.opened_at: Optional[datetime] = None
        self.closed_at: Optional[datetime] = None
        self.timeframe: str = ""
        self.risk_horizon: float = 0.0

    def open(
        self,
        entry_price: float,
        size: float,
        stop_loss_pct: float,
        take_profit_pct: float,
        open_step: int,
        asset: str,
        open_time: Optional[datetime] = None,
        timeframe: str = "5m",
        risk_horizon: float = 0.0,
    ):
        """Ouvre la position."""
        self.is_open = True
        self.asset = asset
        self.entry_price = entry_price
        self.size = size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.open_step = open_step
        self.current_price = entry_price
        if open_time is None:
            raise ValueError("open_time must be provided when opening a position")
        self.opened_at = open_time
        self.closed_at = None
        self.timeframe = timeframe
        self.risk_horizon = risk_horizon

    def close(self, close_time: Optional[datetime] = None):
        """Ferme la position."""
        if close_time is None:
            raise ValueError("close_time must be provided when closing a position")
        self.is_open = False
        self.closed_at = close_time
        self.size = 0.0

    def get_status(self) -> str:
        """Retourne le statut de la position."""
        if self.is_open:
            return f"Open ({self.size} units @ {self.entry_price:.2f})"
        return "Closed"


class PortfolioManager:
    """Gère le portefeuille de trading, le capital, les positions et les métriques."""

    def __init__(
        self,
        config: Dict[str, Any],
        worker_id: int = 0,
        performance_metrics: Optional[PerformanceMetrics] = None,
    ):
        self.worker_id = worker_id
        self.config = config
        self.smart_logger = create_smart_logger(
            worker_id, total_workers=4, logger_name="portfolio_manager"
        )

        # Configuration du capital et des positions
        env_config = self.config.get("environment", {})
        self.initial_equity = env_config.get("initial_balance", 20.50)
        self.initial_capital = self.initial_equity
        self.assets = self.config.get("assets", [])

        # Métriques de performance
        self.metrics = (
            performance_metrics
            if performance_metrics
            else PerformanceMetrics(config=config, worker_id=worker_id)
        )

        self._last_market_timestamp: Optional[datetime] = None
        self._last_positions_snapshot: Dict[str, Dict[str, Any]] = {}

        self.reset()

    def reset(self, **kwargs):
        """Réinitialise le portefeuille à son état initial."""
        self.cash = self.initial_equity
        self.equity = self.initial_equity
        self.peak_equity = self.initial_equity
        self.portfolio_value = self.initial_equity
        # Paramètres de risque courants (par défaut)
        self.sl_pct = kwargs.get("stop_loss_pct", 0.02)
        self.tp_pct = kwargs.get("take_profit_pct", 0.05)
        self.pos_size_pct = kwargs.get("position_size_pct", 0.1)

        self.positions: Dict[str, Position] = {
            asset.upper(): Position() for asset in self.assets
        }
        self.trade_log: List[Dict[str, Any]] = []

        self._last_market_timestamp = None
        self._last_positions_snapshot = {}

        if hasattr(self, "metrics") and self.metrics:
            self.metrics.returns.clear()
            self.metrics.drawdowns.clear()
            self.metrics.equity_curve.clear()
            self.metrics.trades.clear()
            self.metrics.closed_positions.clear()
            self.metrics.frequency_history.clear()
            self.metrics.positions_frequency = {
                "5m": 0,
                "1h": 0,
                "4h": 0,
                "daily_total": 0,
            }
            self.metrics.update_open_positions_metrics([], {})
            self.metrics.record_equity_snapshot(self.equity)

        self.log_info(
            f"Portefeuille réinitialisé. Capital initial: ${self.initial_equity:.2f}"
        )

    def __getstate__(self):
        """Préparer l'état pour le pickling, en excluant le logger."""
        state = self.__dict__.copy()
        # Exclure le logger de la sérialisation de manière sécurisée
        state.pop("smart_logger", None)
        return state

    def __setstate__(self, state):
        """Restaurer l'état après le unpickling et ré-initialiser le logger."""
        self.__dict__.update(state)
        # Ré-initialiser le logger dans le nouveau processus
        self.smart_logger = create_smart_logger(
            getattr(self, "worker_id", 0),
            total_workers=4,
            logger_name="portfolio_manager",
        )

    @staticmethod
    def _normalize_timestamp(timestamp: Optional[Any]) -> Optional[datetime]:
        """Convertit différents formats de timestamps en datetime natif."""
        if timestamp is None:
            return None
        if isinstance(timestamp, datetime):
            return timestamp
        if hasattr(timestamp, "to_pydatetime"):
            try:
                return timestamp.to_pydatetime()
            except Exception:
                return None
        if isinstance(timestamp, np.datetime64):
            try:
                ns_timestamp = timestamp.astype("datetime64[ns]").astype("int64")
                return datetime.utcfromtimestamp(ns_timestamp / 1_000_000_000)
            except Exception:
                return None
        if isinstance(timestamp, str):
            try:
                return datetime.fromisoformat(timestamp)
            except ValueError:
                return None
        return None

    def register_market_timestamp(self, timestamp: Optional[Any]) -> None:
        """Enregistre le dernier horodatage de marché reçu depuis l'environnement."""
        normalized = self._normalize_timestamp(timestamp)
        if normalized is not None:
            self._last_market_timestamp = normalized

    def open_position(
        self,
        asset: str,
        price: float,
        size: float,
        stop_loss_pct: float,
        take_profit_pct: float,
        timestamp: Optional[Any] = None,
        current_prices: Optional[Dict[str, float]] = None,
        allocated_pct: Optional[float] = None,
        timeframe: str = "5m",
        current_step: int = 0,
        risk_horizon: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        """Ouvre une nouvelle position."""
        asset = asset.upper()
        if asset not in self.positions:
            logger.warning(
                f"Actif '{asset}' non trouvé dans le portefeuille. Ajout dynamique."
            )
            self.positions[asset] = Position()

        # Règle: limiter le nombre de positions ouvertes selon le palier
        try:
            tier_cfg = self.get_current_tier()
            limit = 1
            if isinstance(tier_cfg, dict):
                limit = int(tier_cfg.get("max_open_positions", 1))
            open_count = len(self._get_open_positions())
            if open_count >= max(1, limit):
                self.log_info(
                    f"[RISK] Position limit reached ({open_count}/{limit}). Refus d'ouverture pour {asset}."
                )
                try:
                    if hasattr(self, "metrics") and self.metrics:
                        self.metrics.record_trade_rejection(
                            reason="position_limit",
                            context={
                                "asset": asset,
                                "open_count": open_count,
                                "limit": limit,
                            },
                        )
                except Exception:
                    pass
                return None
        except Exception:
            # En cas d'erreur de lecture de palier, on continue sans bloquer
            pass

        position = self.positions[asset]
        if position.is_open:
            logger.warning(
                f"Tentative d'ouverture d'une position déjà ouverte pour {asset}. Ignoré."
            )
            return None

        cost = size * price
        if self.cash < cost:
            logger.warning(
                f"Cash insuffisant pour ouvrir une position de {size} {asset} à ${price:.2f}. Cash: ${self.cash:.2f}, Coût: ${cost:.2f}"
            )
            try:
                if hasattr(self, "metrics") and self.metrics:
                    self.metrics.record_trade_rejection(
                        reason="insufficient_cash",
                        context={
                            "asset": asset,
                            "cost": cost,
                            "cash": float(self.cash),
                        },
                    )
            except Exception:
                pass
            return None

        open_time = self._normalize_timestamp(timestamp) or self._last_market_timestamp
        if open_time is None:
            logger.error(
                f"[Worker {self.worker_id}] Impossible d'ouvrir {asset}: aucun horodatage marché valide disponible."
            )
            return None

        try:
            position.open(
                entry_price=price,
                size=size,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                open_step=current_step,
                asset=asset,
                open_time=open_time,
                timeframe=timeframe,
                risk_horizon=risk_horizon,
            )
        except ValueError as exc:
            logger.error(
                f"[Worker {self.worker_id}] Ouverture de {asset} impossible: {exc}"
            )
            return None
        self.cash -= cost
        position.current_price = price
        self._update_equity(current_prices)
        self.log_info(
            f"[POSITION OUVERTE] {asset}: {size:.6f} @ {price:.2f} | SL: {stop_loss_pct * 100:.2f}% | TP: {take_profit_pct * 100:.2f}% | RH: {risk_horizon:.2f}"
        )

        # Démarrer la traque
        if hasattr(self, "dbe") and self.dbe:
            try:
                duration_config = self.config.get("trading_rules", {}).get(
                    "duration_tracking", {}
                )
                # Utiliser l'horizon de risque pour déterminer la durée maximale
                # Mapper risk_horizon (-1 à 1) à un facteur (ex: 0.5 à 1.5)
                rh_factor = 1.0 + (risk_horizon * 0.5)  # -1 -> 0.5, 0 -> 1.0, 1 -> 1.5
                base_duration = duration_config.get(timeframe, {}).get(
                    "max_duration_steps", 48
                )
                duration_steps = int(base_duration * rh_factor)
                self.dbe.start_hunt(
                    self.worker_id, asset, timeframe, duration_steps, current_step
                )
            except Exception as e:
                logger.error(
                    f"[HUNT] Échec du démarrage de la traque pour le worker {self.worker_id}: {e}"
                )

        # Normalize to picklable primitives
        receipt = {
            "event": "open",
            "asset": str(asset),
            "price": float(price),
            "size": float(size),
            "notional": float(price * size),
            **(
                {"allocated_pct": float(allocated_pct)}
                if allocated_pct is not None
                else {}
            ),
            "timestamp": (
                open_time.isoformat()
                if isinstance(open_time, datetime)
                else str(open_time)
            ),
            "sl": float(stop_loss_pct),
            "tp": float(take_profit_pct),
            "order_id": str(uuid.uuid4()),
            "timeframe": timeframe,
            "risk_horizon": float(risk_horizon),
        }
        self.trade_log.append(receipt)
        try:
            # Journaliser l'ouverture auprès des métriques si disponible
            if hasattr(self, "metrics") and self.metrics:
                self.metrics.record_trade_open(receipt)
        except Exception:
            pass
        return receipt

    def close_position(
        self,
        asset: str,
        price: float,
        timestamp: Optional[Any] = None,
        current_prices: Optional[Dict[str, float]] = None,
        reason: Optional[str] = None,
        risk_horizon: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Ferme une position ouverte et retourne le PnL réalisé."""
        asset = asset.upper()
        if asset not in self.positions or not self.positions[asset].is_open:
            logger.warning(
                f"Tentative de fermeture d'une position inexistante ou déjà fermée pour {asset}."
            )
            return None

        position = self.positions[asset]
        entry_price = position.entry_price
        size = position.size
        open_time = position.opened_at
        close_time = self._normalize_timestamp(timestamp) or self._last_market_timestamp
        if close_time is None:
            logger.error(
                f"[Worker {self.worker_id}] Impossible de fermer {asset}: aucun horodatage marché valide disponible."
            )
            return None

        pnl = (price - entry_price) * size
        pnl_pct = ((price / entry_price) - 1) * 100 if entry_price > 0 else 0.0

        self.cash += price * size

        try:
            position.close(close_time)
        except ValueError as exc:
            logger.error(
                f"[Worker {self.worker_id}] Fermeture de {asset} impossible: {exc}"
            )
            return None

        # Terminer la traque
        if hasattr(self, "dbe") and self.dbe:
            try:
                self.dbe.end_hunt(self.worker_id)
            except Exception as e:
                logger.error(
                    f"[HUNT] Échec de la fin de la traque pour le worker {self.worker_id}: {e}"
                )

        # Normalize to picklable primitives
        log_entry = {
            "action": "close",
            "asset": str(asset),
            "exit_price": float(price),
            "entry_price": float(entry_price),
            "size": float(size),
            "pnl": float(pnl),
            "pnl_pct": float(pnl_pct),
            "timestamp": close_time.isoformat(),
            "opened_at": (open_time.isoformat() if open_time else None),
            "closed_at": close_time.isoformat(),
            "duration_seconds": (
                float((close_time - open_time).total_seconds())
                if (open_time and close_time)
                else None
            ),
            "order_id": str(uuid.uuid4()),
            **({"reason": str(reason)} if reason else {}),
            "risk_horizon": float(position.risk_horizon),
        }

        self._update_equity(current_prices)

        log_entry["equity"] = self.equity

        self.trade_log.append(log_entry)
        self.metrics.update_trade(log_entry)

        self.log_info(
            f"[POSITION FERMÉE] {asset}: {size:.6f} @ {entry_price:.2f} -> {price:.2f} | PnL: ${pnl:+.2f}"
        )
        return log_entry

    def update_market_price(
        self, current_prices: Dict[str, float], current_step: int
    ) -> tuple[float, list[dict[str, Any]]]:
        """Met à jour la valeur des positions, vérifie les SL/TP, et retourne le PnL et les reçus."""
 
        realized_pnl = 0.0
        closed_receipts = []
        for asset, position in self.positions.items():
            if position.is_open and asset in current_prices:
                price = current_prices[asset]
                position.current_price = price

                # Vérification Stop Loss
                sl_price = position.entry_price * (1 - position.stop_loss_pct)
                if price <= sl_price:
                    self.log_info(
                        f"STOP LOSS atteint pour {asset} @ {price:.2f} (SL: {sl_price:.2f})"
                    )
                    receipt = self.close_position(
                        asset,
                        price,
                        timestamp=self._last_market_timestamp,
                        current_prices=current_prices,
                        reason="SL",
                        risk_horizon=position.risk_horizon,
                    )
                    if isinstance(receipt, dict):
                        closed_receipts.append(receipt)
                        val = receipt.get("pnl")
                        if isinstance(val, (int, float)):
                            realized_pnl += float(val)
                    continue

                # Vérification Take Profit
                if position.take_profit_pct > 0:
                    tp_price = position.entry_price * (1 + position.take_profit_pct)
                    if price >= tp_price:
                        self.log_info(
                            f"TAKE PROFIT atteint pour {asset} @ {price:.2f} (TP: {tp_price:.2f})"
                        )
                        receipt = self.close_position(
                            asset,
                            price,
                            timestamp=self._last_market_timestamp,
                            current_prices=current_prices,
                            reason="TP",
                            risk_horizon=position.risk_horizon,
                        )
                        if isinstance(receipt, dict):
                            closed_receipts.append(receipt)
                            val = receipt.get("pnl")
                            if isinstance(val, (int, float)):
                                realized_pnl += float(val)
                        continue

                # Vérification de la durée maximale de la position
                duration_config = self.config.get("trading_rules", {}).get(
                    "duration_tracking", {}
                )
                timeframe = position.timeframe or "5m"
                max_duration = duration_config.get(timeframe, {}).get(
                    "max_duration_steps", 144
                )

                current_duration = current_step - position.open_step
                is_overstay = current_duration > max_duration

                if is_overstay:
                    self.log_info(
                        f"MAX DURATION atteinte pour {asset} @ {price:.2f} (durée: {current_duration} > {max_duration})"
                    )
                    receipt = self.close_position(
                        asset,
                        price,
                        timestamp=self._last_market_timestamp,
                        current_prices=current_prices,
                        reason="MaxDuration",
                        risk_horizon=position.risk_horizon,
                    )
                    if isinstance(receipt, dict):
                        closed_receipts.append(receipt)
                        val = receipt.get("pnl")
                        if isinstance(val, (int, float)):
                            realized_pnl += float(val)
                    continue

        self._update_equity(current_prices)

        # Vérification Stop Loss global (drawdown global)
        try:
            threshold_cfg = self.config.get("risk_management", {}).get(
                "global_sl_pct"
            ) or self.config.get("risk_management", {}).get("max_drawdown_pct")
            if threshold_cfg is not None:
                thr = float(threshold_cfg)
                if thr > 1.0:
                    thr = thr / 100.0
                dd_ratio = self.calculate_drawdown()
                if dd_ratio >= thr:
                    # Fermer toutes les positions ouvertes
                    for a, pos in list(self.positions.items()):
                        if pos.is_open:
                            p = current_prices.get(a, pos.current_price)
                            receipt = self.close_position(
                                a,
                                p,
                                timestamp=self._last_market_timestamp,
                                current_prices=current_prices,
                                reason="GlobalSL",
                                risk_horizon=position.risk_horizon,
                            )
                            if isinstance(receipt, dict):
                                closed_receipts.append(receipt)
                    # Recalculer equity après clôtures
                    self._update_equity(current_prices)
        except Exception as _e:
            # Ne jamais casser la boucle d'update prix pour une erreur de config
            pass

        return realized_pnl, closed_receipts

    def get_state_vector(self) -> np.ndarray:
        """Construit et retourne l'état du portefeuille sous forme de vecteur numpy."""
        try:
            metrics = self.get_metrics()
            total_value = metrics.get("total_value", 0.0)
            cash = metrics.get("cash", 0.0)

            # Obtenir les informations sur les flux de fonds
            fund_analysis = self.get_trading_pnl_vs_external_flows()
            trading_pnl_pct = (
                fund_analysis["trading_pnl"] / fund_analysis["adjusted_initial_capital"]
                if fund_analysis["adjusted_initial_capital"] > 0
                else 0.0
            )
            external_flow_pct = (
                fund_analysis["net_external_flow"] / self.initial_capital
                if self.initial_capital > 0
                else 0.0
            )

            # 10 features de base (incluant les flux de fonds)
            state = [
                cash,
                total_value,
                trading_pnl_pct,  # PnL de trading pur (excluant flux externes)
                external_flow_pct,  # Impact des flux externes (positif = dépôts nets)
                fund_analysis["total_deposits"] / self.initial_capital
                if self.initial_capital > 0
                else 0.0,
                fund_analysis["total_withdrawals"] / self.initial_capital
                if self.initial_capital > 0
                else 0.0,
                metrics.get("sharpe_ratio", 0.0),
                metrics.get("drawdown", 0.0) / 100.0,  # Convertir de % à ratio
                metrics.get("open_positions_count", 0),
                (total_value - cash) / total_value
                if total_value > 0
                else 0.0,  # Allocation
            ]

            # 10 features pour les positions (5 positions * 2 features)
            sorted_positions = sorted(
                metrics.get("positions", {}).items(),
                key=lambda item: abs(
                    item[1].get("size", 0.0) * item[1].get("current_price", 0.0)
                ),
                reverse=True,
            )[:5]

            for asset, pos_obj in sorted_positions:
                state.append(pos_obj.get("size", 0.0))
                state.append(hash(asset) % 1000 / 1000.0)  # Asset encodé et normalisé

            # Remplir les slots de positions restants avec des zéros
            num_pos_features = len(sorted_positions) * 2
            padding_needed = 10 - num_pos_features
            state.extend([0.0] * padding_needed)

            return np.array(state, dtype=np.float32)

        except Exception as e:
            logger.error(
                f"Erreur lors de la construction du vecteur d'état du portefeuille: {e}",
                exc_info=True,
            )
            return np.zeros(20, dtype=np.float32)  # Ajusté pour les nouvelles features

    def _update_equity(self, current_prices: Optional[Dict[str, float]] = None):
        """Met à jour la valeur totale du portefeuille (equity)."""
        prices = current_prices or {}
        positions_value = 0.0

        for asset, position in self.positions.items():
            if not position.is_open:
                continue

            price = prices.get(asset) if prices else None
            if price is None:
                price = position.current_price
            else:
                position.current_price = price

            positions_value += position.size * price

        self.equity = self.cash + positions_value
        self.portfolio_value = self.equity

        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        self.metrics.record_equity_snapshot(self.equity)
        self.metrics.update_open_positions_metrics(self._get_open_positions(), prices)
        self._last_positions_snapshot = self._serialize_positions(prices)

    def _get_open_positions(self) -> List[Position]:
        return [position for position in self.positions.values() if position.is_open]

    def _serialize_positions(
        self, prices: Dict[str, float]
    ) -> Dict[str, Dict[str, Any]]:
        snapshot: Dict[str, Dict[str, Any]] = {}

        for asset, position in self.positions.items():
            if not position.is_open:
                continue

            current_price = prices.get(asset, position.current_price)
            entry_price = position.entry_price
            unrealized = (
                (current_price - entry_price) * position.size
                if entry_price > 0
                else 0.0
            )

            snapshot[asset] = {
                "asset": asset,
                "size": position.size,
                "entry_price": entry_price,
                "current_price": current_price,
                "unrealized_pnl": unrealized,
                "opened_at": position.opened_at.isoformat()
                if position.opened_at
                else None,
                "stop_loss_pct": position.stop_loss_pct,
                "take_profit_pct": position.take_profit_pct,
            }

        return snapshot

    def calculate_drawdown(self) -> float:
        """Calcule le drawdown courant en ratio (0-1)."""
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - self.equity) / self.peak_equity)

    def get_portfolio_value(self) -> float:
        return self.portfolio_value

    def get_cash(self) -> float:
        """Retourne le solde de cash disponible."""
        return float(self.cash)

    def get_metrics(self) -> Dict[str, Any]:
        """Retourne un résumé enrichi des métriques de performance."""
        base_metrics = self.metrics.get_metrics_summary()

        # Agrégats PnL pour clarté Equity vs Capital
        try:
            unrealized_pnl_total = 0.0
            for pos in self._last_positions_snapshot.values():
                unrealized_pnl_total += float(pos.get("unrealized_pnl", 0.0))
        except Exception:
            unrealized_pnl_total = 0.0

        # Somme des PnL réalisés à partir des positions fermées connues
        try:
            realized_pnl_total = 0.0
            if hasattr(self.metrics, "closed_positions"):
                for tr in self.metrics.closed_positions:
                    realized_pnl_total += float(tr.get("pnl", 0.0))
        except Exception:
            realized_pnl_total = 0.0

        enriched_metrics = dict(base_metrics)
        enriched_metrics.update(
            {
                "total_value": float(self.portfolio_value),
                "cash": float(self.cash),
                "unrealized_pnl_total": unrealized_pnl_total,
                "realized_pnl_total": realized_pnl_total,
                "drawdown": self.calculate_drawdown() * 100,
                "max_drawdown": base_metrics.get("max_drawdown", 0.0),
                "positions": self._last_positions_snapshot,
                "open_positions_count": len(self._last_positions_snapshot),
                "equity_curve": list(self.metrics.equity_curve),
                "closed_positions": list(self.metrics.closed_positions),
                "last_market_timestamp": (
                    self._last_market_timestamp.isoformat()
                    if self._last_market_timestamp
                    else None
                ),
            }
        )

        return enriched_metrics

    def get_equity(self) -> float:
        """Retourne l'equity courante du portefeuille."""
        return self.equity

    def get_total_value(self) -> float:
        """Compatibilité: retourne la valeur totale du portefeuille (equity)."""
        return self.get_portfolio_value()

    def get_current_tier(self) -> Dict[str, Any]:
        """
        Compatibilité: retourne la configuration de palier de risque courante.
        Si aucune configuration n'est définie, retourne un dict vide.
        """
        # 1) Si un dict direct est fourni
        tiers = self.config.get("risk_tiers") or self.config.get("tiers")
        if isinstance(tiers, dict):
            return tiers

        # 2) Sélection basée sur capital_tiers (liste de paliers par capital)
        capital_tiers = self.config.get("capital_tiers")
        if isinstance(capital_tiers, list):
            current_capital = float(self.get_portfolio_value())
            for tier in capital_tiers:
                try:
                    min_cap = tier.get("min_capital", float("-inf"))
                    max_cap = tier.get("max_capital", float("inf"))
                    if min_cap is None:
                        min_cap = float("-inf")
                    if max_cap is None:
                        max_cap = float("inf")
                    if min_cap <= current_capital < max_cap:
                        return tier
                except Exception:
                    continue
            # Si aucun palier ne correspond, retourner le plus proche (fallback: premier)
            if capital_tiers:
                return capital_tiers[0]

        # 3) Fallback: dict vide
        return {}

    def log_info(self, message: str):
        """Log un message avec le préfixe du worker."""
        logger.info(f"[Worker {self.worker_id}] {message}")

    def update_risk_parameters(
        self, risk_params: Dict[str, Any], tier: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Met à jour les paramètres de risque avec normalisation aux paliers.

        Args:
            risk_params: Dictionnaire des paramètres de risque du DBE
            tier: Palier actuel (optionnel, sera calculé si None)
        """
        if tier is None:
            tier = self.get_current_tier()

        # Mise à jour des paramètres de base
        self.sl_pct = risk_params.get("stop_loss_pct", getattr(self, "sl_pct", 0.02))
        self.tp_pct = risk_params.get("take_profit_pct", getattr(self, "tp_pct", 0.05))

        pos_size = risk_params.get(
            "position_size_pct", getattr(self, "pos_size_pct", 0.1)
        )
        # Cap dur par le palier (max_position_size_pct)
        max_pos_size_pct = (
            (tier.get("max_position_size_pct", 90.0) / 100.0)
            if isinstance(tier, dict)
            else 0.9
        )
        capped_pos = min(max(0.0, pos_size), max_pos_size_pct)

        # Harmonisation avec exposure_range du palier si présent (intervale cible)
        clamped_by_range = None
        try:
            exposure_range = (
                tier.get("exposure_range") if isinstance(tier, dict) else None
            )
            if (
                exposure_range
                and isinstance(exposure_range, (list, tuple))
                and len(exposure_range) == 2
            ):
                min_pct = float(exposure_range[0]) / 100.0
                max_pct = float(exposure_range[1]) / 100.0
                clamped_by_range = min(max(capped_pos, min_pct), max_pct)
            else:
                clamped_by_range = capped_pos
        except Exception:
            clamped_by_range = capped_pos

        self.pos_size_pct = float(clamped_by_range)

        self.log_info(
            f"[RISK_UPDATE] Palier: {tier.get('name', 'N/A') if isinstance(tier, dict) else 'N/A'}, "
            f"PosSize: {self.pos_size_pct:.2%} (cap≤{max_pos_size_pct:.2%}{', range applied' if isinstance(tier, dict) and tier.get('exposure_range') else ''}), "
            f"SL: {self.sl_pct:.2%}, TP: {self.tp_pct:.2%}"
        )

    def check_emergency_condition(self, current_step: int) -> bool:
        """
                Vérifie les conditions d'urgence nécessitant un reset immédiat.

        {{ ... }}
                    current_step: L'étape actuelle de l'environnement

                Returns:
                    bool: True si un reset d'urgence est nécessaire, False sinon
        """
        if not self.config.get("enable_surveillance_mode", True):
            return False

        current_value = self.get_portfolio_value()
        emergency_threshold = self.config.get(
            "emergency_drawdown_threshold", 0.8
        )  # 80% de drawdown

        # Vérifier le drawdown d'urgence
        if current_value <= (self.initial_equity * (1 - emergency_threshold)):
            if not hasattr(self, "_emergency_reset_count"):
                self._emergency_reset_count = 0
            self._emergency_reset_count += 1
            logger.critical(
                "🚨 EMERGENCY RESET - Drawdown %.2f%% exceeds threshold (%.2f%%). Current: %.2f, Initial: %.2f",
                (1 - current_value / self.initial_equity) * 100
                if self.initial_equity > 0
                else 0,
                emergency_threshold * 100,
                current_value,
                self.initial_equity,
            )
            return True

        return False

    def check_protection_limits(self, current_prices: Dict[str, float]) -> None:
        """
        Vérifie les limites de protection (compatibilité). Ne déclenche pas d'exception.
        Met simplement à jour l'équity avec les prix courants.
        """
        try:
            self._update_equity(current_prices)
        except Exception:
            # En cas d'erreur de données, on ignore pour ne pas interrompre le step
            pass

    # ========================================
    # GESTION DES FONDS (DEPOSITS/WITHDRAWALS)
    # ========================================

    def __init_fund_management(self):
        """Initialise le système de gestion des fonds."""
        if not hasattr(self, "fund_operations_log"):
            self.fund_operations_log: List[Dict[str, Any]] = []
        if not hasattr(self, "total_deposits"):
            self.total_deposits = 0.0
        if not hasattr(self, "total_withdrawals"):
            self.total_withdrawals = 0.0

    def deposit_funds(
        self,
        amount: float,
        reason: str = "Manual deposit",
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Ajoute des fonds au portefeuille.

        Args:
            amount: Montant à déposer (doit être positif)
            reason: Raison du dépôt
            timestamp: Horodatage de l'opération

        Returns:
            Dict contenant les détails de l'opération
        """
        self.__init_fund_management()

        if amount <= 0:
            raise ValueError(f"Le montant du dépôt doit être positif: {amount}")

        # Enregistrer l'état avant
        old_cash = self.cash
        old_equity = self.equity
        old_portfolio_value = self.portfolio_value

        # Effectuer le dépôt
        self.cash += amount
        self.equity += amount
        self.portfolio_value += amount
        self.total_deposits += amount

        # Créer l'enregistrement de l'opération
        operation = {
            "id": str(uuid.uuid4()),
            "type": "DEPOSIT",
            "amount": amount,
            "reason": reason,
            "timestamp": timestamp or datetime.now(),
            "worker_id": self.worker_id,
            "portfolio_state_before": {
                "cash": old_cash,
                "equity": old_equity,
                "portfolio_value": old_portfolio_value,
            },
            "portfolio_state_after": {
                "cash": self.cash,
                "equity": self.equity,
                "portfolio_value": self.portfolio_value,
            },
        }

        self.fund_operations_log.append(operation)

        # Logger l'opération
        self.log_info(
            f"[DEPOSIT] +${amount:.2f} | Reason: {reason} | "
            f"New Balance: ${self.cash:.2f} | New Equity: ${self.equity:.2f}"
        )

        # Mettre à jour les métriques si disponibles
        if hasattr(self, "metrics") and self.metrics:
            self.metrics.record_equity_snapshot(self.equity)

        return operation

    def withdraw_funds(
        self,
        amount: float,
        reason: str = "Manual withdrawal",
        timestamp: Optional[datetime] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Retire des fonds du portefeuille.

        Args:
            amount: Montant à retirer (doit être positif)
            reason: Raison du retrait
            timestamp: Horodatage de l'opération
            force: Si True, permet le retrait même si cela crée un découvert

        Returns:
            Dict contenant les détails de l'opération
        """
        self.__init_fund_management()

        if amount <= 0:
            raise ValueError(f"Le montant du retrait doit être positif: {amount}")

        # Vérifier la disponibilité des fonds
        available_cash = self.cash
        if not force and amount > available_cash:
            raise ValueError(
                f"Fonds insuffisants pour le retrait. "
                f"Disponible: ${available_cash:.2f}, Demandé: ${amount:.2f}"
            )

        # Enregistrer l'état avant
        old_cash = self.cash
        old_equity = self.equity
        old_portfolio_value = self.portfolio_value

        # Effectuer le retrait
        self.cash -= amount
        self.equity -= amount
        self.portfolio_value -= amount
        self.total_withdrawals += amount

        # Créer l'enregistrement de l'opération
        operation = {
            "id": str(uuid.uuid4()),
            "type": "WITHDRAWAL",
            "amount": amount,
            "reason": reason,
            "timestamp": timestamp or datetime.now(),
            "worker_id": self.worker_id,
            "forced": force,
            "portfolio_state_before": {
                "cash": old_cash,
                "equity": old_equity,
                "portfolio_value": old_portfolio_value,
            },
            "portfolio_state_after": {
                "cash": self.cash,
                "equity": self.equity,
                "portfolio_value": self.portfolio_value,
            },
        }

        self.fund_operations_log.append(operation)

        # Logger l'opération
        self.log_info(
            f"[WITHDRAWAL] -${amount:.2f} | Reason: {reason} | "
            f"New Balance: ${self.cash:.2f} | New Equity: ${self.equity:.2f}"
            f"{' [FORCED]' if force else ''}"
        )

        # Mettre à jour les métriques si disponibles
        if hasattr(self, "metrics") and self.metrics:
            self.metrics.record_equity_snapshot(self.equity)

        return operation

    def get_fund_operations_summary(self) -> Dict[str, Any]:
        """
        Retourne un résumé des opérations de fonds.

        Returns:
            Dict contenant le résumé des opérations
        """
        self.__init_fund_management()

        deposits = [op for op in self.fund_operations_log if op["type"] == "DEPOSIT"]
        withdrawals = [
            op for op in self.fund_operations_log if op["type"] == "WITHDRAWAL"
        ]

        return {
            "total_deposits": sum(op["amount"] for op in deposits),
            "total_withdrawals": sum(op["amount"] for op in withdrawals),
            "net_external_flow": sum(op["amount"] for op in deposits)
            - sum(op["amount"] for op in withdrawals),
            "deposit_count": len(deposits),
            "withdrawal_count": len(withdrawals),
            "operations_count": len(self.fund_operations_log),
            "last_operation": self.fund_operations_log[-1]
            if self.fund_operations_log
            else None,
        }

    def get_trading_pnl_vs_external_flows(self) -> Dict[str, float]:
        """
        Calcule la performance pure du trading en excluant les flux externes.

        Returns:
            Dict avec les PnL séparés
        """
        self.__init_fund_management()

        # Capital initial + dépôts - retraits = capital ajusté
        fund_summary = self.get_fund_operations_summary()
        adjusted_initial_capital = (
            self.initial_equity + fund_summary["net_external_flow"]
        )

        # PnL total (actuel - initial)
        total_pnl = self.equity - self.initial_equity

        # PnL de trading (actuel - capital ajusté)
        trading_pnl = self.equity - adjusted_initial_capital

        return {
            "initial_capital": self.initial_equity,
            "current_equity": self.equity,
            "total_deposits": fund_summary["total_deposits"],
            "total_withdrawals": fund_summary["total_withdrawals"],
            "net_external_flow": fund_summary["net_external_flow"],
            "adjusted_initial_capital": adjusted_initial_capital,
            "total_pnl": total_pnl,
            "trading_pnl": trading_pnl,
            "external_flow_impact": fund_summary["net_external_flow"],
        }

    def get_fund_operations_log(
        self, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retourne l'historique des opérations de fonds.

        Args:
            limit: Nombre maximum d'opérations à retourner (les plus récentes)

        Returns:
            Liste des opérations de fonds
        """
        self.__init_fund_management()

        if limit is None:
            return self.fund_operations_log.copy()

        return self.fund_operations_log[-limit:] if limit > 0 else []
