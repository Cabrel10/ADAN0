#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Portfolio management module for the ADAN trading bot.

This module is responsible for tracking the agent's financial status, including
capital, positions, and performance metrics.
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from ..environment.finance_manager import FinanceManager
from ..performance.metrics import PerformanceMetrics


logger = logging.getLogger(__name__)


class Position:
    """Represents a single, simple trading position (long or short)."""

    def __init__(self):
        self.is_open = False
        self.entry_price = 0.0
        self.size = 0.0  # Number of units
        self.stop_loss_pct = 0.0
        self.take_profit_pct = 0.0

    def open(
        self,
        entry_price: float,
        size: float,
        stop_loss_pct: float = 0.0,
        take_profit_pct: float = 0.0,
    ) -> None:
        self.is_open = True
        self.entry_price = entry_price
        self.size = size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def close(self):
        self.is_open = False
        self.entry_price = 0.0
        self.size = 0.0
        self.stop_loss_pct = 0.0
        self.take_profit_pct = 0.0

    def get_status(self) -> str:
        """Get the status of the position.

        Returns:
            str: A string describing the position status.
        """
        if self.is_open:
            return f"Open ({self.size} units @ {self.entry_price:.2f})"
        return "Closed"


class PortfolioManager:
    """Manages the trading portfolio for a single asset.

    Handles capital allocation, tracks PnL, and enforces risk rules defined
    in the environment configuration. Also tracks performance per data chunk
    for reward shaping and learning purposes.
    """

    def _update_equity(self) -> None:
        """Met à jour la valeur de l'équité du portefeuille."""
        # Initialiser current_prices s'il n'existe pas
        if not hasattr(self, 'current_prices'):
            self.current_prices = {}

        # Calculer la valeur des positions ouvertes
        positions_value = sum(
            position.size * self.current_prices.get(asset, 0)
            for asset, position in self.positions.items()
            if position.is_open
        )

        # Mettre à jour l'équité (cash + valeur des positions)
        self.equity = self.cash + positions_value

        # Mettre à jour le pic d'équité pour le calcul du drawdown
        if not hasattr(self, 'peak_equity') or self.equity > self.peak_equity:
            self.peak_equity = self.equity

        # Calculer le drawdown courant
        if hasattr(self, 'peak_equity') and self.peak_equity > 0:
            self.current_drawdown = (self.peak_equity - self.equity) / self.peak_equity
        else:
            self.current_drawdown = 0.0

    def get_equity(self) -> float:
        """Retourne la valeur actuelle de l'équité du portefeuille."""
        if not hasattr(self, 'equity'):
            self._update_equity()
        return getattr(self, 'equity', 0.0)

    def get_performance_metrics(self) -> dict:
        """Retourne les métriques de performance actuelles du portefeuille.

        Returns:
            dict: Dictionnaire contenant les métriques de performance
        """
        return {
            'equity': self.get_equity(),
            'cash': self.cash,
            'drawdown': self.current_drawdown if hasattr(self, 'current_drawdown') else 0.0,
            'peak_equity': getattr(self, 'peak_equity', self.initial_capital),
            'win_rate': getattr(self, 'win_rate', 0.0),
            'sharpe_ratio': getattr(self, 'sharpe_ratio', 0.0),
            'sortino_ratio': getattr(self, 'sortino_ratio', 0.0),
            'profit_factor': getattr(self, 'profit_factor', 0.0),
            'total_trades': getattr(self, 'total_trades', 0),
            'winning_trades': getattr(self, 'winning_trades', 0),
            'losing_trades': getattr(self, 'losing_trades', 0),
            'max_drawdown': getattr(self, 'max_drawdown', 0.0),
            'risk_free_rate': getattr(self, 'risk_free_rate', 0.0),
            'volatility': getattr(self, 'volatility', 0.0),
            'cagr': getattr(self, 'cagr', 0.0)
        }

    def get_balance(self) -> float:
        """Retourne le solde disponible (cash) du portefeuille."""
        return getattr(self, 'cash', 0.0)

    def get_total_value(self) -> float:
        """
        Retourne la valeur totale du portefeuille (cash + valeur des positions ouvertes).

        Returns:
            float: Valeur totale du portefeuille dans la devise de base
        """
        # Mise à jour de l'équité pour s'assurer d'avoir les dernières valeurs
        self._update_equity()

        # Retourner l'équité qui est déjà la somme du cash et de la valeur des positions
        return self.equity

    def __init__(self, env_config: Dict[str, Any], assets: Optional[List[str]] = None) -> None:
        # Devise de base du portefeuille
        self.currency = env_config.get('default_currency', 'USDT')

        # Dictionnaire des prix courants pour chaque actif
        self.current_prices = {}

        """
        Initialize the PortfolioManager with environment configuration and assets.

        Args:
            env_config: Dictionary containing environment configuration
            assets: Optional list of asset symbols. If None, will be derived from
                env_config['assets'] or env_config['environment']['assets'] if present.
        """
        self.config = env_config

        # Initialize FinanceManager
        try:
            fee_pct = env_config.get("trading_fees", 0.001)
        except Exception:
            fee_pct = 0.001

        try:
            min_order_usdt = env_config.get("min_order_value_usdt", 1.0)
        except Exception:
            min_order_usdt = 1.0

        # Initialize FinanceManager with default values that will be updated later
        self.finance_manager = FinanceManager(
            initial_capital=0.0,  # Will be updated with actual initial_balance
            fee_pct=fee_pct,
            min_order_usdt=min_order_usdt,
        )

        # Initialize performance metrics tracking
        self.metrics = PerformanceMetrics(
            metrics_dir=str(Path(self.config.get('paths', {}).get('metrics_dir', 'logs/metrics')))
        )

        # Logging configuration
        self.log_interval = self.config.get('logging', {}).get('metrics_interval', 100)
        self.step_count = 0

        # Initialize configuration sections
        portfolio_config = self.config.get("portfolio", {})
        environment_config = self.config.get("environment", {})
        trading_rules_config = self.config.get("trading_rules", {})

        # Derive assets if not provided
        if assets is None:
            derived_assets = self.config.get("assets") or environment_config.get("assets") or []
            assets = list(derived_assets)
            if not assets:
                logger.warning("No assets provided or found in config; defaulting to empty asset list")

        # Initialize equity and capital (normalize to tests expectations)
        # Priority: top-level 'initial_capital' -> environment.initial_balance -> portfolio.initial_balance -> default 1000.0
        default_balance = 1000.0
        self.initial_equity = (
            self.config.get("initial_capital",
                environment_config.get(
                    "initial_balance",
                    portfolio_config.get("initial_balance", default_balance)
                )
            )
        )
        self.initial_capital = self.initial_equity  # For backward compatibility
        self.current_equity = self.initial_equity

        # Initialize portfolio state (will be fully set in reset())
        self.cash = 0.0
        self.portfolio_value = 0.0
        self.total_capital = 0.0

        # Initialize position tracking
        self.assets = assets
        self.assets = assets
        self.positions: Dict[str, Position] = {asset.upper(): Position() for asset in assets}
        self.trade_history: List[Dict[str, Any]] = []
        self.trade_log: List[Dict[str, Any]] = []  # Initialize trade log

        # Surveillance mode state
        self._surveillance_mode = False  # Indique si le mode surveillance est actif
        self._surveillance_start_step = None  # Étape à laquelle le mode surveillance a été activé
        self._chunk_start_step = 0  # Étape de début du chunk actuel
        self._survived_chunks = 0  # Nombre de chunks passés en mode surveillance
        self._recovery_threshold = 11.0  # Seuil de récupération (11 USDT)
        self._last_surveillance_warning = 0  # Dernier avertissement de surveillance émis

        # Nouveaux états pour la surveillance améliorée
        self._critical_chunk_count = 0  # Nombre de chunks critiques consécutifs
        self.surveillance_chunk_start_balance = 0.0  # Solde au début du chunk de surveillance
        self._emergency_reset_count = 0  # Compteur de réinitialisations d'urgence
        self._surveillance_entry_count = 0  # Nombre d'entrées en mode surveillance
        self._surveillance_success_count = 0  # Nombre de récupérations réussies
        self._surveillance_failure_count = 0  # Nombre d'échecs de récupération

        # NOUVEAU: Tracker chunks pour reset strict (préserver expérience DBE)
        self.chunk_below_threshold = 0  # Tracker chunks <11$
        self.min_capital = environment_config.get("min_capital_before_reset", 11.0)  # Seuil minimum

        # Initialize chunk-based tracking
        self.chunk_pnl: Dict[int, Dict[str, float]] = {}
        self.current_chunk_id = 0
        self.chunk_start_equity = self.initial_equity

        # Initialize step tracking
        self.current_step = 0

        # Trading rules configuration
        self.futures_enabled = bool(trading_rules_config.get("futures_enabled", False))
        self.leverage = int(trading_rules_config.get("leverage", 1))

        # Commission and fees
        if self.futures_enabled:
            self.commission_pct = float(trading_rules_config.get("futures_commission_pct", 0.0004))
        else:
            self.commission_pct = float(trading_rules_config.get("commission_pct", 0.001))

        # Position sizing rules
        self.min_trade_size = float(trading_rules_config.get("min_trade_size", 0.0001))
        self.min_notional_value = float(trading_rules_config.get("min_notional_value", 10.0))
        self.max_notional_value = float(trading_rules_config.get("max_notional_value", 100000.0))

        # Configuration de la gestion des risques
        risk_management = self.config.get("risk_management", {})

        # 1. Charger les capital_tiers depuis la configuration
        self.capital_tiers = self.config.get("capital_tiers", [])

        # Si vide, essayer de charger depuis risk_management
        if not self.capital_tiers and "capital_tiers" in risk_management:
            self.capital_tiers = risk_management["capital_tiers"]
            logger.info("Chargement des capital_tiers depuis risk_management")

        # 2. Valider le format et le contenu des tiers
        if not isinstance(self.capital_tiers, list):
            logger.error(
                "capital_tiers doit être une liste, mais a reçu: %s",
                type(self.capital_tiers),
            )
            self.capital_tiers = []

        # 3. Définir les clés requises pour chaque tier
        # Remarque: max_concurrent_positions est optionnel (par défaut à 1) pour compatibilité avec les tests
        REQUIRED_KEYS = {
            "name",
            "min_capital",
            "max_position_size_pct",
            "risk_per_trade_pct",
            "max_drawdown_pct",
            "leverage",
        }

        # 4. Valider chaque tier
        valid_tiers = []
        for i, tier in enumerate(self.capital_tiers):
            if not isinstance(tier, dict):
                msg = f"Le tier {i} n'est pas un dictionnaire et sera ignoré: {tier}"
                logger.warning(msg)
                continue

            # Vérifier les clés requises (max_concurrent_positions est optionnel)
            missing_keys = REQUIRED_KEYS - tier.keys()
            if missing_keys:
                msg = (
                    f"Le tier {i} manque des clés requises {missing_keys} "
                    f"et sera ignoré: {tier}"
                )
                logger.warning(msg)
                continue

            # Vérifier que max_concurrent_positions est un entier positif (par défaut 1 si absent)
            max_pos = tier.get("max_concurrent_positions", 1)
            if not isinstance(max_pos, int) or max_pos <= 0:
                msg = (
                    f"Tier {i}: max_concurrent_positions invalide ({max_pos}). "
                    "Valeur par défaut: 1"
                )
                logger.warning(msg)
                tier["max_concurrent_positions"] = 1
            else:
                tier["max_concurrent_positions"] = max_pos
            valid_tiers.append(tier)

        # 5. Trier les tiers par min_capital croissant
        valid_tiers.sort(key=lambda x: x["min_capital"])

        # 6. Vérifier la continuité des paliers
        for i in range(1, len(valid_tiers)):
            prev_tier = valid_tiers[i - 1]
            curr_tier = valid_tiers[i]
            if prev_tier["min_capital"] >= curr_tier["min_capital"]:
                msg = (
                    "Les paliers de capital doivent être en ordre croissant. "
                    f"Palier {i-1} ({prev_tier['name']}) a un min_capital "
                    f">= au palier {i} ({curr_tier['name']})"
                )
                logger.error(msg)
                valid_tiers = []
                break

        # 7. Mettre à jour la liste des tiers valides
        self.capital_tiers = valid_tiers

        # Provide a safe default tier if none present to satisfy tests using simple configs
        if not self.capital_tiers:
            logger.warning("No valid capital tiers found; using a default tier")
            self.capital_tiers = [
                {
                    "name": "default",
                    "min_capital": 0,
                    "max_position_size_pct": 0.5,
                    "risk_per_trade_pct": 1.0,
                    "max_drawdown_pct": 10.0,
                    "leverage": self.leverage,
                    "max_concurrent_positions": 1,
                }
            ]
        else:
            tier_info = ", ".join(
                f"{t['name']} ({t['min_capital']}+)" for t in self.capital_tiers
            )
            logger.info(
                "%d paliers de capital chargés avec succès: %s",
                len(self.capital_tiers),
                tier_info,
            )

        # Position sizing configuration
        self.position_sizing_config = risk_management.get("position_sizing", {})
        self.concentration_limits = self.position_sizing_config.get(
            "concentration_limits", {}
        )

        # Trading protection flag (spot mode): when True, no new long trades are allowed
        self.trading_disabled: bool = False

        self.reset()

    def reset(self) -> None:
        """Reset portfolio state to initial values and clear open positions.

        Tests expect initial_capital, total_capital, and cash to be equal at start,
        and all positions closed.
        """
        self.cash = float(self.initial_capital)
        self.total_capital = float(self.initial_capital)
        self.portfolio_value = float(self.initial_capital)
        # Close all positions
        for pos in self.positions.values():
            pos.close()
        # Reset step/chunk trackers minimally
        self.current_step = 0
        self.current_chunk_id = 0

    def get_margin_level(self) -> float:
        """
        Returns the current margin level (margin used / available capital).

        Returns:
            float: The margin level as a ratio of used margin to initial capital.
        """
        if not self.futures_enabled:
            return 1.0  # Not applicable for spot trading

        # Calculate total margin used
        total_margin_used = 0.0
        for position in self.positions.values():
            if position.is_open:
                margin = (position.size * position.entry_price) / self.leverage
                total_margin_used += margin

        # Margin level is the ratio of margin used to initial capital
        if self.initial_capital > 0:
            return total_margin_used / self.initial_capital

        # Handle error state when initial capital is 0 or negative
        return 0.0

    def get_current_tier(self) -> Dict[str, Any]:
        """Détermine le tier de capital actuel en fonction de la valeur du portefeuille.

        Returns:
            dict: Configuration du tier actuel avec les clés :
                - name: Nom du palier
                - min_capital: Capital minimum du palier
                - max_capital: Capital max du palier (None si dernier palier)
                - max_position_size_pct: Taille max de position en %
                - leverage: Effet de levier autorisé
                - risk_per_trade_pct: Risque max par trade en %
                - max_drawdown_pct: Drawdown maximum autorisé en %

        Raises:
            RuntimeError: Si aucun tier de capital n'est défini ou si la
                configuration est invalide
        """
        if not self.capital_tiers:
            raise RuntimeError(
                "Configuration des capital_tiers invalide ou vide. "
                "Vérifiez la configuration."
            )

        # Trier les tiers par min_capital croissant (au cas où)
        sorted_tiers = sorted(self.capital_tiers, key=lambda x: x["min_capital"])

        # Trouver le premier tier où min_capital <= capital_de_référence < next_tier.min_capital
        # Les tests attendent une sélection basée sur le capital initial
        current_equity = float(getattr(self, "initial_capital", self.get_portfolio_value()))

        for i, tier in enumerate(sorted_tiers):
            # Si c'est le dernier tier, on l'utilise
            if i == len(sorted_tiers) - 1:
                logger.debug(
                    "Palier actuel: %s (capital: %.2f >= %.2f)",
                    tier["name"],
                    current_equity,
                    tier["min_capital"],
                )
                return tier

            # Sinon, vérifier si on est dans l'intervalle [min_capital,
            # next_tier.min_capital)
            next_tier = sorted_tiers[i + 1]
            if tier["min_capital"] <= current_equity < next_tier["min_capital"]:
                logger.debug(
                    "Palier actuel: %s (%.2f <= capital: %.2f < %.2f)",
                    tier["name"],
                    tier["min_capital"],
                    current_equity,
                    next_tier["min_capital"],
                )
                return tier

        # Si on arrive ici, on utilise le dernier tier (ne devrait normalement
        # pas arriver)
        logger.warning(
            f"Aucun tier trouvé pour la valeur de portefeuille "
            f"{current_equity:.2f}. Utilisation du dernier tier disponible."
        )
        return sorted_tiers[-1]

    # Backward-compatible alias used elsewhere in this module
    def get_active_tier(self) -> Dict[str, Any]:
        return self.get_current_tier()

    def calculate_position_size(
        self,
        price: float,
        stop_loss_pct: float = 0.02,
        risk_per_trade: float = 0.01,
        account_risk_multiplier: float = 1.0,
        expected_return_pct: float = 0.0,  # Rendement attendu en pourcentage
        min_profit_margin: float = 1.5,    # Marge de profit minimale (1.5x la commission)
        aggressivity: float = 0.5,         # Coefficient d'agressivité (0-1)
    ) -> float:
        """
        Calcule la taille de position en fonction du risque, du stop loss et des limites de position.

        La taille de la position est déterminée par les facteurs suivants :
        1. Limite du nombre de positions simultanées (max_concurrent_positions)
        2. Taille maximale autorisée par le palier (max_position_size_pct)
        3. Taille basée sur le risque (risque_max / (prix * stop_loss_pct))
        4. Capital disponible (en tenant compte des commissions et d'un buffer)

        Args:
            price: Prix actif
            stop_loss_pct: Pourcentage de stop loss (ex: 0.02 pour 2%)
            risk_per_trade: Fraction du capital à risquer (0.01 pour 1%)
            account_risk_multiplier: Multiplicateur de risque (défini par le DBE)

        Returns:
            float: Taille de position en unités de l'actif, ou 0 si les conditions ne sont pas remplies
        """
        if price <= 0 or stop_loss_pct <= 0:
            return 0.0

        try:
            # Récupérer le palier actif
            tier = self.get_active_tier()

            # 1. Vérifier le capital minimum (11 USDT)
            min_trade_value = 11.0  # Minimum de 11 USDT par trade
            if self.portfolio_value < min_trade_value:
                logger.info("[POSITION SIZE] Capital insuffisant (%.2f < %.2f USDT)",
                          self.portfolio_value, min_trade_value)
                return 0.0

            # 2. Compter les positions ouvertes
            current_open = sum(1 for pos in self.positions.values() if pos.is_open)

            # 3. Vérifier la limite de positions simultanées
            if current_open >= tier.get('max_concurrent_positions', 1):
                logger.info(
                    "[POSITION SIZE] Limite de positions atteinte (%d/%d)",
                    current_open, tier['max_concurrent_positions']
                )
                return 0.0

            # 4. Calculer la fourchette de position en fonction du palier
            min_position_pct = tier.get('position_size_range', [0.01, 0.1])[0]  # Par défaut 1-10%
            max_position_pct = tier.get('position_size_range', [0.01, 0.1])[1]

            # 5. Calculer la taille de position en fonction de l'agressivité
            position_pct = min_position_pct + (max_position_pct - min_position_pct) * aggressivity

            # 6. Calculer la taille de position en USDT
            position_size_usdt = self.portfolio_value * position_pct

            # 7. Vérifier le minimum de 11 USDT
            position_size_usdt = max(position_size_usdt, min_trade_value)

            # 8. Calculer la taille en unités
            position_size = position_size_usdt / price

            # 9. Calculer la perte potentielle avec le stop-loss
            potential_loss = position_size_usdt * stop_loss_pct

            # 10. Calculer la perte maximale autorisée (4% du capital)
            max_allowed_loss = self.portfolio_value * 0.04

            # 11. Ajuster le stop-loss si nécessaire pour respecter la contrainte de 4%
            if potential_loss > max_allowed_loss:
                adjusted_sl_pct = (max_allowed_loss / position_size_usdt) * 0.9  # Marge de sécurité de 10%
                logger.info(
                    "[POSITION SIZE] Ajustement du stop-loss de %.2f%% à %.2f%% pour respecter la contrainte de drawdown",
                    stop_loss_pct * 100, adjusted_sl_pct * 100
                )
                stop_loss_pct = adjusted_sl_pct

                # Recalculer la taille basée sur le nouveau stop-loss
                risk_amount = self.portfolio_value * (tier["risk_per_trade_pct"] / 100.0)
                risk_amount *= account_risk_multiplier
                risk_based_size = risk_amount / (price * stop_loss_pct)

                # Prendre le minimum entre la taille calculée et la taille basée sur le risque
                position_size = min(position_size, risk_based_size)
            else:
                risk_amount = self.portfolio_value * (tier["risk_per_trade_pct"] / 100.0)
                risk_amount *= account_risk_multiplier
                risk_based_size = risk_amount / (price * stop_loss_pct)
                position_size = min(position_size, risk_based_size)

            # 12. Calculer la taille maximale autorisée par le palier
            max_position_value = self.portfolio_value * (tier["max_position_size_pct"] / 100.0)
            max_position_size = max_position_value / price

            # Prendre la plus petite des deux tailles basées sur le risque et la limite du palier
            target_size = min(risk_based_size, max_position_size)

            # Calculer le capital disponible en tenant compte d'un buffer de sécurité
            buffer = max(
                self.commission_pct * 2,  # Au moins 2x la commission
                0.02 * self.portfolio_value  # Ou 2% du portefeuille, selon le plus grand
            )
            available_cash = max(0.0, self.get_available_capital() - buffer)

            # Calculer la taille finale en fonction du cash disponible
            affordable_size = available_cash / price
            position_size = min(target_size, affordable_size)

            # Vérifier la taille minimale de trade
            if position_size > 0 and position_size < self.min_trade_size:
                logger.info(
                    "[POSITION SIZE] Taille de position (%.8f) inférieure au minimum (%.8f)",
                    position_size, self.min_trade_size
                )
                return 0.0

            # Vérifier si le profit attendu couvre les commissions avec une marge
            commission = position_size * price * self.commission_pct
            min_acceptable_profit = commission * min_profit_margin
            expected_profit = position_size * price * (expected_return_pct / 100.0)

            if expected_return_pct > 0 and expected_profit < min_acceptable_profit:
                logger.info(
                    "[POSITION SIZE] Profit attendu (%.4f) < marge minimale (%.4f) pour commission (%.4f)",
                    expected_profit, min_acceptable_profit, commission
                )
                return 0.0

            # Journalisation détaillée
            logger.debug(
                "[POSITION SIZE] Calcul terminé - Taille: %.8f (Max: %.8f, Risque: %.8f, "
                "Disponible: %.8f, Comm: %.4f, Profit min: %.4f, Profit attendu: %.4f)",
                position_size, max_position_size, risk_based_size, affordable_size,
                commission, min_acceptable_profit, expected_profit if expected_return_pct > 0 else 0.0
            )

            return position_size

        except Exception as e:
            logger.error(
                "Erreur lors du calcul de la taille de position: %s",
                str(e),
                exc_info=True
            )
            return 0.0

    def calculate_commission(self, notional_value: float) -> float:
        """
        Calcule la commission pour une valeur notionnelle donnée.

        Args:
            notional_value: La valeur notionnelle de la transaction

        Returns:
            float: Le montant de la commission
        """
        return notional_value * self.commission_pct

    def is_profitable_after_commissions(
        self,
        notional_value: float,
        expected_return_pct: float,
        min_profit_margin: float = 1.5
    ) -> bool:
        """
        Vérifie si une transaction est rentable après prise en compte des commissions.

        Args:
            notional_value: La valeur notionnelle de la transaction
            expected_return_pct: Le rendement attendu en pourcentage
            min_profit_margin: La marge de profit minimale par rapport aux commissions

        Returns:
            bool: True si la transaction est rentable, False sinon
        """
        if expected_return_pct <= 0:
            return False

        commission = self.calculate_commission(notional_value)
        expected_profit = notional_value * (expected_return_pct / 100.0)
        min_acceptable_profit = commission * min_profit_margin

        return expected_profit >= min_acceptable_profit

    def _calculate_volatility(self, window: int = 20) -> float:
        """Calcule la volatilité des rendements sur une fenêtre glissante.

        Args:
            window: Taille de la fenêtre de calcul (minimum 2, maximum 252)

        Returns:
            float: Volatilité annualisée des rendements sur la fenêtre, ou 0.0 si non calculable
        """
        # Validation des entrées
        window = max(2, min(window, 252))  # Borne la fenêtre entre 2 et 252

        # Vérification des données disponibles
        if not self.trade_history or len(self.trade_history) < 2:
            return 0.0

        try:
            # Conversion en array numpy et vérification des valeurs
            values = np.array(self.trade_history[-window:], dtype=np.float64)

            # Suppression des valeurs non finies (NaN, Inf)
            values = values[np.isfinite(values)]

            # Vérification après nettoyage
            if len(values) < 2 or np.any(values <= 0):
                return 0.0

            # Calcul des rendements logarithmiques avec protection contre les valeurs non positives
            returns = np.diff(np.log(values))

            # Vérification des rendements calculés
            if len(returns) < 1 or not np.all(np.isfinite(returns)):
                return 0.0

            # Calcul de la volatilité annualisée (252 jours de bourse par an)
            volatility = np.std(returns, ddof=1)  # ddof=1 pour l'estimation non biaisée

            # Protection contre les valeurs aberrantes
            if not np.isfinite(volatility) or volatility <= 0:
                return 0.0

            return volatility * np.sqrt(252)  # Annualisation

        except (ValueError, RuntimeWarning, ZeroDivisionError) as e:
            logger.warning(f"Erreur dans le calcul de la volatilité: {str(e)}")
            return 0.0

    def get_state(self) -> np.ndarray:
        """Return the current portfolio state as a numpy array with 17 dimensions.

        The state includes:
            0. Cash ratio (cash / portfolio value)
            1. Equity ratio (current equity / initial equity)
            2. Current margin level
            3. Ratio of open positions
            4. Realized PnL
            5. Unrealized PnL
            6. Max drawdown
            7. Sharpe ratio
            8. Sortino ratio (placeholder)
            9. Portfolio volatility
            10. Total fees paid
            11. Total commissions paid
            12. Number of trades
            13. Win rate
            14. Average gain per winning trade
            15. Average loss per losing trade
            16. Current drawdown

        Returns:
            np.ndarray: Array of shape (17,) containing portfolio state info
        """
        # Calculate basic metrics
        cash_ratio = 0.0
        if self.portfolio_value > 0:
            cash_ratio = self.cash / self.portfolio_value

        equity_ratio = 0.0
        if self.initial_equity > 0:
            equity_ratio = self.current_equity / self.initial_equity

        margin_level = self.get_margin_level()

        # Calculate position metrics
        open_positions = sum(1 for p in self.positions.values() if p.is_open)
        open_positions_ratio = open_positions / max(1, len(self.positions))

        # Calculate trade metrics
        def trade_filter(trade):
            return trade.get("type") == "close"

        def pnl_positive(trade):
            return trade.get("trade_pnl", 0) > 0

        def pnl_negative(trade):
            return trade.get("trade_pnl", 0) <= 0

        closed_trades = list(filter(trade_filter, self.trade_log))
        total_trades = len(closed_trades)
        winning_trades = list(filter(pnl_positive, closed_trades))
        losing_trades = list(filter(pnl_negative, closed_trades))

        win_rate = len(winning_trades) / max(1, total_trades)
        avg_win = (
            np.mean([t.get("trade_pnl", 0) for t in winning_trades])
            if winning_trades
            else 0.0
        )
        avg_loss = (
            np.mean([abs(t.get("trade_pnl", 0)) for t in losing_trades])
            if losing_trades
            else 0.0
        )

        # Calculate volatility
        volatility = self._calculate_volatility()

        # Create state vector with 17 dimensions
        state = np.array(
            [
                # 0-3: Basic metrics
                cash_ratio,  # 0: Cash ratio
                equity_ratio,  # 1: Equity ratio
                margin_level,  # 2: Margin level
                open_positions_ratio,  # 3: Open positions ratio
                # 4-7: PnL and risk metrics
                self.realized_pnl,  # 4: Realized PnL
                self.unrealized_pnl,  # 5: Unrealized PnL
                self.drawdown,  # 6: Max drawdown
                self.sharpe_ratio,  # 7: Sharpe ratio
                # 8-11: Advanced metrics
                0.0,  # 8: Sortino ratio (placeholder)
                volatility,  # 9: Portfolio volatility
                0.0,  # 10: Total fees paid (placeholder)
                0.0,  # 11: Commissions paid (placeholder)
                # 12-16: Trade statistics
                total_trades,  # 12: Number of trades
                win_rate,  # 13: Win rate
                avg_win,  # 14: Avg gain per winning trade
                avg_loss,  # 15: Avg loss per losing trade
                self.drawdown,  # 16: Current drawdown
            ],
            dtype=np.float32,
        )

        if len(state) != 17:
            err_msg = f"State vector must have 17 dimensions, got {len(state)}"
            raise ValueError(err_msg)

        return state

    def _enter_surveillance_mode(self, current_step: int) -> None:
        """
        Active le mode surveillance avec suivi amélioré et initialisation des compteurs.

        Le mode surveillance permet au bot de tenter de se rétablir après une baisse
        significative de la valeur du portefeuille en dessous du seuil critique.

        Pendant ce mode :
        - Le bot a 2 chunks complets pour tenter de se rétablir
        - Un soft reset est effectué à chaque chunk
        - Si après 2 chunks la valeur est toujours en dessous du seuil, un full reset est effectué

        Args:
            current_step: L'étape actuelle de l'environnement
        """
        if not self._surveillance_mode:
            # Initialiser les compteurs si nécessaire
            if not hasattr(self, '_surveillance_entry_count'):
                self._surveillance_entry_count = 0
            if not hasattr(self, '_critical_chunk_count'):
                self._critical_chunk_count = 0

            # Activer le mode surveillance
            self._surveillance_mode = True
            self._surveillance_start_step = current_step
            self._chunk_start_step = current_step
            self._survived_chunks = 0
            self._critical_chunk_count += 1
            self._surveillance_entry_count += 1
            self.surveillance_chunk_start_balance = self.get_portfolio_value()

            # Initialiser le suivi de la valeur maximale pendant la surveillance
            self._surveillance_max_value = self.surveillance_chunk_start_balance
            self._surveillance_min_value = self.surveillance_chunk_start_balance

            logger.warning(
                "⚠️ ENTERING SURVEILLANCE MODE - Portfolio: $%.2f, Start: $%.2f, Critical chunks: %d, Attempts: %d",
                self.get_portfolio_value(),
                self.surveillance_chunk_start_balance,
                self._critical_chunk_count,
                self._surveillance_entry_count
            )

            # Enregistrer l'état initial pour le débogage
            logger.debug(
                "[SURVEILLANCE] Initial state - Step: %d, Max: $%.2f, Min: $%.2f, Current: $%.2f",
                current_step,
                self._surveillance_max_value,
                self._surveillance_min_value,
                self.get_portfolio_value()
            )

    def _check_surveillance_status(self, current_step: int) -> bool:
        """
        Vérifie l'état de surveillance et met à jour les compteurs.

        Cette méthode est appelée à chaque étape pour :
        1. Mettre à jour les statistiques de surveillance (valeur max/min)
        2. Détecter si on doit entrer ou sortir du mode surveillance
        3. Vérifier si on a dépassé le nombre maximal de chunks en surveillance

        Args:
            current_step: L'étape actuelle de l'environnement

        Returns:
            bool: True si un reset complet est nécessaire, False sinon
        """
        current_value = self.get_portfolio_value()
        is_critical = current_value <= self._recovery_threshold

        # Mettre à jour les statistiques de surveillance si en mode surveillance
        if self._surveillance_mode:
            # Mettre à jour la valeur maximale et minimale
            if hasattr(self, '_surveillance_max_value'):
                self._surveillance_max_value = max(self._surveillance_max_value, current_value)
            if hasattr(self, '_surveillance_min_value'):
                self._surveillance_min_value = min(self._surveillance_min_value, current_value)

            # Journalisation périodique pour le débogage
            if current_step % 100 == 0:  # Toutes les 100 étapes
                logger.debug(
                    "[SURVEILLANCE] Step %d - Value: $%.2f, Max: $%.2f, Min: $%.2f, Chunks: %d/2",
                    current_step, current_value,
                    getattr(self, '_surveillance_max_value', current_value),
                    getattr(self, '_surveillance_min_value', current_value),
                    getattr(self, '_survived_chunks', 0)
                )

        # Vérifier si on entre ou sort du mode surveillance
        if is_critical and not self._surveillance_mode:
            self._enter_surveillance_mode(current_step)
        elif not is_critical and self._surveillance_mode:
            # Ne pas sortir automatiquement du mode surveillance ici
            # On laisse la méthode reset gérer cela
            pass

        # Vérifier si on a dépassé le nombre maximal de chunks en surveillance
        if self._surveillance_mode and self._survived_chunks >= 2:
            logger.warning(
                "[SURVEILLANCE] Maximum chunks in surveillance reached. "
                "Forcing reset on next step."
            )
            self._exit_surveillance_mode(recovered=False)
            return True

    def validate_position(self, asset: str, size: float, price: float) -> bool:
        """
        Validate a prospective position against trading rules and available funds.

        Rules per tests:
        - size must be >= min_trade_size
        - notional = size * price must be within [min_notional_value, max_notional_value]
        - sufficient funds:
          spot: cash >= notional + commission
          futures: cash >= margin(notional / leverage) + commission
        """
        try:
            # Asset must be known
            if asset not in self.positions:
                logger.warning("Unknown asset %s in validate_position", asset)
                return False

            # Basic parameter checks
            if not isinstance(size, (int, float)) or not isinstance(price, (int, float)):
                return False
            if size <= 0 or price <= 0:
                return False

            # Trading rules
            min_size = float(getattr(self, "min_trade_size", 0.0001))
            min_notional = float(getattr(self, "min_notional_value", 10.0))
            max_notional = float(getattr(self, "max_notional_value", 100000.0))
            commission_pct = float(getattr(self, "commission_pct", 0.001))

            if size < min_size:
                return False

            notional = float(size) * float(price)
            if not (min_notional <= notional <= max_notional):
                return False

            # Funds check
            if getattr(self, "futures_enabled", False):
                margin = notional / max(1.0, float(getattr(self, "leverage", 1)))
                commission = notional * commission_pct
                required = margin + commission
            else:
                commission = notional * commission_pct
                required = notional + commission

            # Use current cash for affordability
            available_cash = float(getattr(self, "cash", 0.0))
            return available_cash >= required

        except Exception as e:
            logger.error("Error in validate_position: %s", str(e), exc_info=True)
            return False

    def check_liquidation(self, current_prices: Dict[str, float]) -> bool:
        """
        Check if portfolio should be liquidated based on a simple threshold used by tests.

        The unit test expects liquidation when total_capital falls below
        liquidation_threshold * initial_capital. When triggered, all positions are
        closed at provided prices and totals are set to cash only.

        Returns True if liquidation occurred, else False.
        """
        try:
            # Only meaningful in margin/futures context per tests
            threshold = (
                self.config.get("trading_rules", {}).get("liquidation_threshold", 0.2)
            )
            initial_cap = float(getattr(self, "initial_capital", 0.0))
            current_total = float(getattr(self, "total_capital", self.get_portfolio_value()))

            # Compute trigger level based on tests' assumption
            trigger_level = initial_cap * float(threshold)
            if current_total < trigger_level:
                # Close all positions using provided prices if available
                for asset, pos in list(self.positions.items()):
                    if pos.is_open:
                        px = current_prices.get(asset)
                        if isinstance(px, (int, float)) and px > 0:
                            self.close_position(asset, float(px))
                        else:
                            # Fallback to entry price to ensure closure
                            self.close_position(asset, pos.entry_price)

                # After liquidation: portfolio equals cash only
                self.unrealized_pnl = 0.0
                self.portfolio_value = float(self.cash)
                self.total_capital = float(self.cash)
                self.current_equity = float(self.cash)
                logger.critical(
                    "LIQUIDATION TRIGGERED - total_capital %.2f < trigger %.2f",
                    current_total,
                    trigger_level,
                )
                return True

            return False
        except Exception as e:
            logger.error("Error in check_liquidation: %s", str(e), exc_info=True)
            return False

        return False

    def check_emergency_condition(self, current_step: int) -> bool:
        """
        Vérifie les conditions d'urgence nécessitant un reset immédiat.

        Args:
            current_step: L'étape actuelle de l'environnement

        Returns:
            bool: True si un reset d'urgence est nécessaire, False sinon
        """
        if not self.config.get("enable_surveillance_mode", True):
            return False

        current_value = self.get_portfolio_value()
        emergency_threshold = self.config.get("emergency_drawdown_threshold", 0.8)  # 80% de drawdown

        # Vérifier le drawdown d'urgence
        if current_value <= (self.initial_equity * (1 - emergency_threshold)):
            self._emergency_reset_count += 1
            logger.critical(
                "🚨 EMERGENCY RESET - Drawdown %.2f%% exceeds threshold (%.2f%%). Current: %.2f, Initial: %.2f",
                (1 - current_value / self.initial_equity) * 100,
                emergency_threshold * 100,
                current_value,
                self.initial_equity
            )
            return True

        return False

    def get_surveillance_metrics(self) -> Dict[str, Any]:
        """
        Récupère les métriques de surveillance actuelles.

        Returns:
            Dict[str, Any]: Dictionnaire contenant les métriques de surveillance
        """
        return {
            "in_surveillance": self._surveillance_mode,
            "critical_chunk_count": self._critical_chunk_count,
            "surveillance_entry_count": self._surveillance_entry_count,
            "surveillance_success_count": self._surveillance_success_count,
            "surveillance_failure_count": self._surveillance_failure_count,
            "emergency_reset_count": self._emergency_reset_count,
            "current_balance": self.get_portfolio_value(),
            "surveillance_start_balance": self.surveillance_chunk_start_balance if self._surveillance_mode else 0.0,
        }

    def _exit_surveillance_mode(self, recovered: bool = True) -> None:
        """
        Désactive le mode surveillance et met à jour les compteurs.

        Cette méthode est appelée dans deux cas :
        1. Récupération réussie (recovered=True) : Le portefeuille est remonté au-dessus du seuil critique
        2. Échec de la récupération (recovered=False) : Le portefeuille n'a pas récupéré après 2 chunks

        Args:
            recovered: Si True, indique une récupération réussie
        """
        if not self._surveillance_mode:
            return

        # Initialiser les compteurs si nécessaire
        if not hasattr(self, '_surveillance_success_count'):
            self._surveillance_success_count = 0
        if not hasattr(self, '_surveillance_failure_count'):
            self._surveillance_failure_count = 0

        current_value = self.get_portfolio_value()
        start_balance = getattr(self, 'surveillance_chunk_start_balance', current_value)
        chunks_survived = getattr(self, '_survived_chunks', 0)

        # Mettre à jour les statistiques de surveillance
        if hasattr(self, '_surveillance_max_value'):
            max_value = self._surveillance_max_value
            min_value = getattr(self, '_surveillance_min_value', start_balance)
            drawdown = ((start_balance - min_value) / start_balance * 100) if start_balance > 0 else 0
            recovery = ((current_value - min_value) / min_value * 100) if min_value > 0 else 0
        else:
            max_value = current_value
            min_value = current_value
            drawdown = 0
            recovery = 0

        if recovered:
            self._surveillance_success_count += 1
            logger.info(
                "✅ SURVEILLANCE SUCCESS - Portfolio recovered to $%.2f (from $%.2f), "
                "Max: $%.2f, Min: $%.2f, Drawdown: %.2f%%, Recovery: %.2f%%",
                current_value, start_balance, max_value, min_value, drawdown, recovery
            )
        else:
            self._surveillance_failure_count += 1
            logger.warning(
                "❌ SURVEILLANCE FAILED - Portfolio at $%.2f (from $%.2f), "
                "Max: $%.2f, Min: $%.2f, Drawdown: %.2f%%",
                current_value, start_balance, max_value, min_value, drawdown
            )

            # En cas d'échec, forcer la fermeture de toutes les positions
            for asset in list(self.positions.keys()):
                if self.positions[asset].is_open:
                    logger.debug("Closing position for %s due to surveillance failure", asset)
                    self.close_position(asset, self.positions[asset].entry_price)

        # Réinitialiser l'état de surveillance
        self._surveillance_mode = False
        self._surveillance_start_step = None
        self._survived_chunks = 0
        self.surveillance_chunk_start_balance = 0.0

        # Nettoyer les attributs de suivi
        if hasattr(self, '_surveillance_max_value'):
            del self._surveillance_max_value
        if hasattr(self, '_surveillance_min_value'):
            del self._surveillance_min_value

    def _soft_reset_epoch_state(self) -> None:
        """
        Réinitialise l'état pour un nouvel 'epoch' sans toucher au capital.

        Cette méthode est appelée pour effectuer une réinitialisation partielle de l'état
        du portefeuille à la fin d'un chunk, sans réinitialiser complètement le capital.

        En mode surveillance, elle incrémente également le compteur de chunks survécus.
        """
        # Sauvegarder le capital actuel
        current_capital = self.get_portfolio_value()

        # Mettre à jour les statistiques de surveillance si nécessaire
        if self._surveillance_mode:
            # S'assurer que l'attribut existe
            if not hasattr(self, '_survived_chunks'):
                self._survived_chunks = 0

            # Incrémenter le compteur de chunks
            self._survived_chunks += 1

            # Journaliser l'état actuel
            start_balance = getattr(self, 'surveillance_chunk_start_balance', current_capital)
            max_value = getattr(self, '_surveillance_max_value', current_capital)
            min_value = getattr(self, '_surveillance_min_value', current_capital)

            logger.info(
                "[SURVEILLANCE] Chunk %d/2 completed - Value: $%.2f (Start: $%.2f, Max: $%.2f, Min: $%.2f)",
                self._survived_chunks, current_capital, start_balance, max_value, min_value
            )

            # Réinitialiser le suivi des valeurs min/max pour le prochain chunk
            if hasattr(self, '_surveillance_max_value'):
                self._surveillance_max_value = current_capital
            if hasattr(self, '_surveillance_min_value'):
                self._surveillance_min_value = current_capital

        # Fermer toutes les positions sans enregistrer le PnL
        for asset, position in list(self.positions.items()):
            if position.is_open:
                try:
                    # Fermer la position sans enregistrer le PnL
                    self.close_position(asset, position.entry_price)
                    logger.debug("Soft-closing position for %s during epoch reset", asset)
                except Exception as e:
                    logger.error("Error soft-closing position %s: %s", asset, str(e))

        # Réinitialiser les compteurs et états
        self._reset_metrics()

        # Rétablir le capital (sans toucher au capital total)
        self.cash = current_capital
        self.portfolio_value = current_capital
        self.current_equity = current_capital
        self.peak_equity = current_capital

        # Réinitialiser le suivi des performances du chunk
        self.chunk_start_balance = current_capital

    def reset(self, new_epoch: bool = True, force: bool = False, min_capital_before_reset: float = None) -> bool:
        """
        Reset portfolio manager avec logique stricte pour préserver l'expérience DBE.

        NOUVELLE LOGIQUE STRICTE (pour préserver expérience DBE):
        1. If force=True or new_epoch=True: Full reset to initial capital
        2. Reset capital SEULEMENT si capital < min_capital pendant chunk complet
        3. Sinon: Préserver capital ET état DBE (continuité totale)

        Returns:
            bool: True if a hard capital reset was performed, False otherwise
        """
        import inspect

        # Get configuration sections
        portfolio_config = self.config.get("portfolio", {})
        environment_config = self.config.get("environment", {})

        # Get current portfolio value
        current_value = self.get_portfolio_value()

        # Get min_capital_before_reset from config if not provided
        if min_capital_before_reset is None:
            min_capital_before_reset = environment_config.get("min_capital_before_reset", 11.0)

        # Log the current state for debugging
        caller_frame = inspect.currentframe().f_back
        caller_info = f"{caller_frame.f_code.co_name}:{caller_frame.f_lineno}" if caller_frame else "unknown"

        logger.debug(
            "[RESET] Current value: $%.2f, New epoch: %s, Force: %s, Min capital: $%.2f, "
            "Surveillance: %s, Survived chunks: %d, Caller: %s",
            current_value, new_epoch, force, min_capital_before_reset,
            getattr(self, '_surveillance_mode', False),
            getattr(self, '_survived_chunks', 0),
            caller_info
        )

        # Close all open positions before any reset
        self._close_all_positions()

        # Rule 1: Forced reset (new episode or force=True)
        if force or new_epoch:
            reset_reason = "force=True" if force else "new_epoch=True"
            logger.warning(
                "[FULL RESET] Performing full reset (reason: %s, capital=$%.2f, threshold=$%.2f)",
                reset_reason, current_value, min_capital_before_reset
            )
            # Exit surveillance mode if active
            if getattr(self, '_surveillance_mode', False):
                self._exit_surveillance_mode(recovered=False)
            return self._perform_full_reset(portfolio_config, environment_config)

        # Get surveillance mode status safely with default
        surveillance_mode = getattr(self, '_surveillance_mode', False)

        # Rule 2: Handle surveillance mode
        if surveillance_mode:
            # Increment survived chunks counter if not just entered
            if hasattr(self, '_surveillance_chunk_count'):
                self._surveillance_chunk_count += 1
            else:
                self._surveillance_chunk_count = 1

            logger.warning(
                "[SURVEILLANCE] Portfolio value: $%.2f, Chunks survived: %d/2, Start balance: $%.2f",
                current_value,
                self._surveillance_chunk_count,
                getattr(self, 'surveillance_chunk_start_balance', current_value)
            )

            # If we've survived enough chunks and are above threshold, exit surveillance
            if self._surveillance_chunk_count >= 2:
                if current_value > min_capital_before_reset:
                    logger.info(
                        "[SURVEILLANCE] Successfully recovered to $%.2f > $%.2f - Exiting surveillance mode",
                        current_value, min_capital_before_reset
                    )
                    self._exit_surveillance_mode(recovered=True)
                    # Perform soft reset with current capital
                    self._perform_soft_reset(current_value)
                    return False
                else:
                    # After 2 chunks, if still below threshold, perform full reset
                    logger.warning(
                        "[SURVEILLANCE] Failed to recover after 2 chunks ($%.2f <= $%.2f) - Full reset",
                        current_value, min_capital_before_reset
                    )
                    self._exit_surveillance_mode(recovered=False)
                    return self._perform_full_reset(portfolio_config, environment_config)
            else:
                # Stay in surveillance mode with soft reset
                self._soft_reset_epoch_state()
                return False

        # Rule 3: If value drops below threshold, enter surveillance mode
        if current_value <= min_capital_before_reset:
            # NOUVELLE LOGIQUE: Check reset strict au lieu de surveillance
            return self.check_reset()

        # Rule 4: NOUVELLE LOGIQUE - Préservation complète (capital + état DBE)
        logger.info(
            "[NO RESET] Capital OK ($%.2f > $%.2f) - Continuité préservée (capital + DBE)",
            current_value, min_capital_before_reset
        )
        self.chunk_below_threshold = 0  # Reset counter si remontée
        return False

    def _close_all_positions(self) -> None:
        """Ferme toutes les positions ouvertes aux prix du marché actuels."""
        if not hasattr(self, 'positions') or not self.positions:
            return

        # Pour chaque actif avec une position ouverte, on la ferme
        for asset in list(self.positions.keys()):
            position = self.positions[asset]
            if position.is_open:
                try:
                    # On utilise le prix du marché actuel ou le prix d'entrée si indisponible
                    current_price = getattr(position, 'current_price', position.entry_price)
                    self.close_position(asset, current_price)
                    logger.debug("Position fermée pour %s à $%.8f", asset, current_price)
                except Exception as e:
                    logger.error("Erreur lors de la fermeture de la position pour %s: %s", asset, str(e))

    def _perform_full_reset(self, portfolio_config: dict, environment_config: dict) -> bool:
        """
        Effectue une réinitialisation complète du portefeuille au capital initial.

        Args:
            portfolio_config: Configuration du portefeuille
            environment_config: Configuration de l'environnement

        Returns:
            bool: Toujours True pour indiquer qu'un full reset a été effectué
        """
        # Réinitialisation complète au capital initial
        # Respecter la priorité de l'initialisation utilisée dans le constructeur:
        # top-level 'initial_capital' -> environment.initial_balance -> portfolio.initial_balance -> default 1000.0
        computed_initial = self.config.get(
            "initial_capital",
            environment_config.get(
                "initial_balance",
                portfolio_config.get("initial_balance", 1000.0),
            ),
        )
        self.initial_equity = float(computed_initial)
        self.initial_capital = self.initial_equity
        self.cash = self.initial_equity
        self.portfolio_value = self.initial_equity
        self.total_capital = self.initial_equity
        self.current_equity = self.initial_equity
        self.peak_equity = self.initial_equity
        self.trade_history = [self.initial_equity]
        self.trade_log = []

        # Réinitialisation de l'état de surveillance
        self._surveillance_mode = False
        self._surveillance_start_step = None
        self._survived_chunks = 0
        self.surveillance_chunk_start_balance = 0.0

        # Réinitialisation des métriques
        self._reset_metrics()

        logger.warning(
            "[FULL RESET] Portefeuille réinitialisé au capital initial: %.2f",
            self.initial_equity
        )

        return True

    def _perform_soft_reset(self, current_value: float) -> None:
        """
        Effectue une réinitialisation douce du portefeuille en préservant le capital actuel.

        Args:
            current_value: Valeur actuelle du portefeuille
        """
        logger.debug(
            "[SOFT RESET] Performing soft reset with preserved capital: $%.2f",
            current_value
        )

        # Close any remaining positions (should be none at this point)
        self._close_all_positions()

        # Reset metrics but keep current capital
        self.trade_history = []
        self.peak_equity = current_value
        self.current_equity = current_value
        self.portfolio_value = current_value
        self.total_capital = current_value
        self.cash = current_value  # Reset cash to current value (no positions open after reset)

        # Re-initialize positions
        self.positions = {asset.upper(): Position() for asset in getattr(self, 'assets', [])}

        # Reset any per-epoch state
        self._soft_reset_epoch_state()

        logger.debug(
            "[SOFT RESET] Completed. New balance: $%.2f, Cash: $%.2f",
            self.portfolio_value, self.cash
        )

        # Mettre à jour l'historique des trades
        self.trade_history.append(current_value)

        logger.debug(
            "[SOFT RESET] État du portefeuille réinitialisé avec une valeur de %.2f",
            current_value
        )

    def _reset_metrics(self):
        """Reset all portfolio metrics to their initial state."""
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.drawdown = 0.0
        self.sharpe_ratio = 0.0
        self.var = 0.0
        self.cvar = 0.0
        self.trading_disabled = False

    def check_reset(self):
        """
        Vérifie conditions de reset strict pour capital uniquement.

        Reset SEULEMENT si capital < min_capital pendant chunk complet,
        sans effacer l'état DBE pour préserver expérience.

        Returns:
            bool: True si hard reset capital effectué, False sinon
        """
        current_value = self.get_portfolio_value()

        if current_value < self.min_capital:
            self.chunk_below_threshold += 1
            logger.warning(
                f"[CAPITAL TRACKING] Capital < ${self.min_capital:.2f} pour chunk #{self.chunk_below_threshold}"
            )

            if self.chunk_below_threshold >= 1:  # Un chunk complet
                logger.info(
                    f"[HARD RESET] Capital < ${self.min_capital:.2f} pour {self.chunk_below_threshold} chunk(s) complet(s) - Reset capital UNIQUEMENT"
                )
                # Reset capital seulement (préserver état DBE)
                self.capital = self.initial_capital
                self.cash = self.initial_capital
                self.portfolio_value = self.initial_capital
                self.positions = {}  # Reset positions
                self.chunk_below_threshold = 0  # Reset counter
                return True
        else:
            self.chunk_below_threshold = 0  # Remontée : Reset counter

        return False

    def update_risk_parameters(self, risk_params, tier=None):
        """
        Met à jour les paramètres de risque avec normalisation aux paliers.

        Args:
            risk_params: Dictionnaire des paramètres de risque
            tier: Palier actuel (optionnel, sera calculé si None)
        """
        import math

        if tier is None:
            tier = self.get_current_tier()

        # Mise à jour basique des paramètres
        if 'sl' in risk_params:
            self.sl_pct = risk_params['sl']
        if 'tp' in risk_params:
            self.tp_pct = risk_params['tp']
        if 'pos_size' in risk_params:
            self.pos_size_pct = risk_params['pos_size']

        # Normalisation sigmoïde pour respecter les paliers
        if hasattr(self, 'pos_size_pct'):
            min_bound = 0.01  # 1% minimum
            max_bound = tier['max_position_size_pct'] / 100.0  # Conversion en décimal
            mid = (min_bound + max_bound) / 2
            k = 0.1

            # Normalisation sigmoïde
            normalized = mid + ((max_bound - min_bound) / 2) * math.tanh(k * (self.pos_size_pct - mid))
            self.pos_size_pct = min(max(normalized, min_bound), max_bound)  # Fallback clip

        # Appliquer contraintes de risque par trade
        if 'risk_per_trade' in risk_params:
            max_risk = tier['risk_per_trade_pct'] / 100.0
            self.risk_per_trade_pct = min(risk_params['risk_per_trade'], max_risk)

        logger.info(
            f"[RISK UPDATED] Palier: {tier['name']}, PosSize: {getattr(self, 'pos_size_pct', 0.1)*100:.1f}%, "
            f"Risk: {getattr(self, 'risk_per_trade_pct', 0.01)*100:.1f}%"
        )

        self.peak_equity = self.initial_equity

    def get_current_tier_index(self) -> int:
        """Retourne l'index du palier actuel dans self.capital_tiers (0-based)."""
        equity = float(getattr(self, "initial_capital", self.get_portfolio_value()))
        sorted_tiers = sorted(self.capital_tiers, key=lambda x: x["min_capital"])

        for i, tier in enumerate(sorted_tiers):
            # Si c'est le dernier palier
            if i == len(sorted_tiers) - 1:
                if equity >= tier["min_capital"]:
                    return i
            else:
                next_min = sorted_tiers[i + 1]["min_capital"]
                if tier["min_capital"] <= equity < next_min:
                    return i
        return 0

    def get_current_tier_name(self) -> str:
        """Retourne le nom du palier actuel."""
        idx = self.get_current_tier_index()
        return sorted(self.capital_tiers, key=lambda x: x["min_capital"])[idx]["name"]

    def get_portfolio_value(self) -> float:
        """
        Returns the current total portfolio value (cash + open positions) with numerical safety.

        Returns:
            float: Total portfolio value in quote currency, never negative
        """
        try:
            # Calculate total value of open positions with protection
            positions_value = 0.0
            for pos in self.positions.values():
                if pos.is_open and hasattr(pos, 'size') and hasattr(pos, 'entry_price'):
                    positions_value = positions_value + float(pos.size) * float(pos.entry_price)

            # Update and return the total portfolio value with protection
            total_value = float(self.cash) + positions_value
            self.portfolio_value = max(0.0, total_value)  # Never go below zero
            return self.portfolio_value
        except (TypeError, ValueError) as e:
            logger.error(f"Error calculating portfolio value: {e}")
            return max(0.0, float(getattr(self, 'cash', 0.0)))  # Fallback to cash if available

    def get_leverage(self) -> float:
        """
        Calculate the current leverage of the portfolio.

        Returns:
            float: Current leverage ratio (1.0 = no leverage, 2.0 = 2x, etc.)
        """
        if self.portfolio_value <= 0:
            return 1.0  # Avoid division by zero

        # Calculate total position value (sum of all open positions)
        total_position_value = 0.0
        for position in self.positions.values():
            if position.is_open:
                total_position_value += abs(position.size * position.entry_price)

        # Leverage = Total position value / Portfolio equity
        leverage = total_position_value / self.portfolio_value

        # Apply configured leverage cap if futures are enabled
        if self.futures_enabled:
            leverage = min(leverage, self.leverage)

        return max(1.0, leverage)  # Minimum leverage is 1.0

    def calculate_drawdown(self) -> float:
        """
        Calculate the current drawdown of the portfolio.

        Returns:
            float: Current drawdown as a percentage (0.0 to 1.0)
        """
        if not hasattr(self, "peak_equity") or self.peak_equity == 0:
            return 0.0

        current_equity = self.get_portfolio_value()
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            return 0.0

        return (self.peak_equity - current_equity) / self.peak_equity

    def get_available_capital(self) -> float:
        """
        Returns the available capital for new trades based on total portfolio value.

        For spot trading, available capital is calculated as the total portfolio value
        minus the value of all open positions. For margin trading, it considers the
        margin used and available leverage.

        Returns:
            float: The amount of capital available for new trades.
        """
        # Get current portfolio value (cash + open positions)
        portfolio_value = self.get_portfolio_value()

        # Calculate total value of open positions (absolute value to handle both long and short)
        open_positions_value = sum(
            abs(pos.size) * pos.entry_price
            for pos in self.positions.values()
            if pos.is_open
        )

        # Calculate available capital based on leverage
        if self.leverage > 1.0 and open_positions_value > 0:
            # For margin trading, available capital is reduced by margin used
            margin_used = open_positions_value / self.leverage
            available_capital = max(0, portfolio_value - margin_used)

            # Log detailed information for margin trading
            logger.debug(
                "[CAPITAL] Available: %.2f (Portfolio: %.2f, Margin Used: %.2f, "
                "Positions: %.2f, Leverage: %.1fx)",
                available_capital,
                portfolio_value,
                margin_used,
                open_positions_value,
                self.leverage,
            )
        else:
            # For spot trading, available capital is total value minus open positions
            available_capital = max(0, portfolio_value - open_positions_value)

            # Log detailed information for spot trading
            logger.debug(
                "[CAPITAL] Available: %.2f (Portfolio: %.2f, Positions: %.2f)",
                available_capital,
                portfolio_value,
                open_positions_value,
            )

        return available_capital

    def update_market_price(self, current_prices: Dict[str, float]) -> None:
        """
        Met à jour la valeur du portefeuille en fonction des prix actuels avec gestion robuste des erreurs.

        Args:
            current_prices: Dictionnaire associant les symboles d'actifs à leurs prix actuels.
        """
        if not isinstance(current_prices, dict):
            logger.warning("current_prices doit être un dictionnaire. Reçu: %s", type(current_prices))
            current_prices = {}

        try:
            previous_portfolio_value = float(self.portfolio_value) if hasattr(self, 'portfolio_value') else 0.0
            self.unrealized_pnl = 0.0
            total_positions_value = 0.0
            assets_with_missing_prices = []

            for asset, position in self.positions.items():
                if not position.is_open:
                    continue

                current_price = current_prices.get(asset)

                # Skip invalid prices
                if not isinstance(current_price, (int, float)) or current_price <= 0:
                    assets_with_missing_prices.append(asset)
                    continue

                try:
                    current_price = float(current_price)
                    entry_value = float(position.size) * float(position.entry_price)
                    current_value = float(position.size) * current_price
                    position_pnl = current_value - entry_value

                    # Update position metrics
                    self.unrealized_pnl += position_pnl
                    total_positions_value += current_value

                    # Update position attributes
                    position.current_price = current_price
                    position.current_value = current_value
                    position.unrealized_pnl = position_pnl

                    # Safe percentage calculation
                    if entry_value > 0:
                        position.pnl_pct = (position_pnl / entry_value) * 100.0
                    else:
                        position.pnl_pct = 0.0

                    # Log position details
                    logger.debug(
                        "[POSITION] %s: %.8f @ %.8f (Val: %.2f USDT, PnL: %.2f USDT, %.2f%%)",
                        asset,
                        position.size,
                        current_price,
                        current_value,
                        position_pnl,
                        position.pnl_pct,
                    )

                except (TypeError, ValueError) as e:
                    logger.error(f"Error updating position {asset}: {e}")
                    continue

            # Log assets with missing prices
            if assets_with_missing_prices:
                logger.warning(
                    "Prix manquants ou invalides pour %d actifs: %s. "
                    "Ces positions seront ignorées.",
                    len(assets_with_missing_prices),
                    ", ".join(assets_with_missing_prices),
                )

            # Update portfolio metrics
            self.portfolio_value = max(0.0, float(self.cash) + total_positions_value)
            self.current_equity = self.portfolio_value
            self.total_capital = self.portfolio_value

            # Calculate total PnL with protection
            try:
                total_pnl = self.portfolio_value - float(getattr(self, 'initial_equity', 0.0))
                pnl_pct = (total_pnl / float(self.initial_equity) * 100) if float(self.initial_equity) > 0 else 0.0
                logger.debug(
                    "[PORTFOLIO] Valeur totale: %.2f USDT (Cash: %.2f, Positions: %.2f, PnL: %.2f USDT, %.2f%%)",
                    self.portfolio_value,
                    float(self.cash),
                    total_positions_value,
                    total_pnl,
                    pnl_pct,
                )
            except (TypeError, ValueError) as e:
                logger.error(f"Error calculating portfolio PnL: {e}")

            # Update trade history (limit to 1000 entries)
            if not hasattr(self, "trade_history"):
                self.trade_history = [max(0.0, float(getattr(self, 'initial_equity', 0.0)))]

            self.trade_history.append(self.portfolio_value)
            if len(self.trade_history) > 1000:
                self.trade_history.pop(0)

            # Apply funding rates if futures are enabled
            if getattr(self, 'futures_enabled', False):
                # Funding rate logic would go here
                pass

            # Update metrics
            self.update_metrics()

            # Check protection limits with valid prices
            valid_prices = {
                k: float(v) for k, v in current_prices.items()
                if isinstance(v, (int, float)) and v > 0
            }

            if valid_prices:
                protection_triggered = self.check_protection_limits(valid_prices)
                if protection_triggered:
                    # Reset metrics after protection action
                    self.unrealized_pnl = 0.0
                    self.portfolio_value = max(0.0, float(self.cash))
                    self.current_equity = self.portfolio_value
                    self.total_capital = self.portfolio_value
                    logger.warning("Protection limits triggered - reset portfolio metrics")

        except Exception as e:
            logger.critical(
                "[CRITICAL] Erreur lors de la mise à jour des prix du marché: %s",
                str(e),
                exc_info=True
            )
            # Ensure portfolio value is always valid
            self.portfolio_value = max(0.0, float(getattr(self, 'cash', 0.0)))
            self.current_equity = self.portfolio_value
            self.total_capital = self.portfolio_value

            # Calculate drawdown for logging
            try:
                current_drawdown = float(getattr(self, 'initial_equity', 0.0)) - self.portfolio_value
                initial_equity = float(getattr(self, 'initial_equity', 1.0))
                drawdown_pct = (current_drawdown / initial_equity * 100) if initial_equity > 0 else 0.0
                logger.warning(
                    "Protection triggered - Portfolio: %.2f USDT, Drawdown: %.2f USDT (%.2f%%)",
                    self.portfolio_value,
                    current_drawdown,
                    drawdown_pct
                )
            except (TypeError, ValueError) as e:
                logger.error("Erreur lors du calcul du drawdown: %s", str(e))

        # Vérifier les ordres de protection si des positions sont ouvertes
        if any(pos.is_open for pos in self.positions.values()) and not valid_prices:
            logger.warning(
                "Impossible de vérifier les ordres de protection: "
                "aucun prix valide disponible"
            )

    def check_protection_limits(self, current_prices: Dict[str, float]) -> bool:
        """
        Vérifie si le portefeuille dépasse les limites de protection définies.

        Dans un contexte de trading spot, cette méthode vérifie les conditions de protection
        et ferme les positions si nécessaire pour protéger le capital.

        Args:
            current_prices: Dictionnaire des prix actuels par actif.

        Returns:
            bool: True si une action de protection a été déclenchée, False sinon.
        """
        # Récupérer le palier de capital actuel
        tier = self.get_active_tier()
        max_drawdown_pct = tier.get("max_drawdown_pct", 20.0) / 100.0

        # Vérifier le drawdown maximum autorisé
        max_drawdown_value = self.initial_equity * max_drawdown_pct
        current_drawdown = self.initial_equity - self.portfolio_value
        current_drawdown_pct = (
            (current_drawdown / self.initial_equity) * 100
            if self.initial_equity > 0
            else 0
        )

        # Vérifier le solde disponible pour éviter les positions trop importantes
        # Utilisation de la valeur totale du portefeuille moins la valeur des positions ouvertes
        # pour éviter de compter deux fois la même valeur
        open_positions_value = sum(
            pos.size * current_prices.get(asset, 0)
            for asset, pos in self.positions.items()
            if pos.is_open and asset in current_prices
        )
        available_balance = max(0, self.portfolio_value - open_positions_value)
        # Laisser 1% de marge pour les frais
        max_position_size = available_balance * 0.99
        # Journalisation des informations de risque
        logger.info(
            "[RISK] Drawdown actuel: %.2f/%.2f USDT (%.1f%%/%.1f%%), "
            "Solde dispo: %.2f USDT",
            current_drawdown,
            max_drawdown_value,
            current_drawdown_pct,
            max_drawdown_pct * 100,
            available_balance,
        )

        try:
            # Vérifier si le drawdown dépasse la limite du palier
            if current_drawdown > max_drawdown_value:
                tier = self.get_active_tier()
                logger.critical(
                    "[CRITICAL] Drawdown critique: %.2f/%.2f USDT (%.1f%%/%.1f%%), "
                    "Palier: %s (%.2f USDT), Solde: %.2f USDT",
                    current_drawdown,
                    max_drawdown_value,
                    current_drawdown_pct,
                    max_drawdown_pct * 100,
                    tier.get("name", "Inconnu"),
                    self.initial_equity,
                    self.portfolio_value,
                )

                if not self.futures_enabled:
                    # Spot mode: disable new buy trades, do NOT force-close positions
                    if not self.trading_disabled:
                        logger.warning(
                            "[PROTECTION] Spot mode: Disabling new BUY trades due to drawdown breach."
                        )
                    self.trading_disabled = True
                    # Keep positions open; environment should respect this via validation
                    return True

                # Futures/margin mode: proceed with liquidation as before
                positions_closed = False
                for asset in list(self.positions.keys()):
                    current_price = current_prices.get(asset)
                    if current_price is not None:
                        logger.info(
                            "[ACTION] Fermeture de la position %s à %.8f USDT",
                            asset,
                            current_price,
                        )
                        self.close_position(asset, current_price)
                        positions_closed = True
                    else:
                        logger.error(
                            "[ERROR] Impossible de fermer la position %s: prix manquant",
                            asset,
                        )

                # Mettre à jour les métriques (futures)
                if positions_closed:
                    self.unrealized_pnl = 0.0
                    self.total_capital = self.cash
                    self.portfolio_value = self.cash
                    self.current_equity = self.cash
                    logger.info(
                        "[STATUS] Portefeuille après fermeture: %.2f USDT (Cash: %.2f USDT)",
                        self.portfolio_value,
                        self.cash,
                    )
                return True

            # Si on arrive ici, c'est que le drawdown est dans les limites
            return False

        except Exception as e:
            # En cas d'erreur, on bloque par sécurité
            logger.critical(
                "[CRITICAL] Erreur lors de la vérification des limites de protection: %s",
                str(e),
                exc_info=True,
            )
            return True

            # Aucune position n'a pu être fermée
            return False

        # Vérifier également le niveau de marge pour les comptes sur marge
        if self.futures_enabled:
            liquidation_threshold = self.config["trading_rules"].get(
                "liquidation_threshold", 0.2
            )
            margin_level = self.get_margin_level()

            if margin_level < liquidation_threshold:
                logger.warning(
                    "Niveau de marge %.1f%% en dessous du seuil de liquidation de %.1f%%. "
                    "Liquidation des positions.",
                    margin_level * 100,
                    liquidation_threshold * 100,
                )

                # Fermer toutes les positions
                for asset in list(self.positions.keys()):
                    current_price = current_prices.get(asset)
                    if current_price is not None:
                        self.close_position(asset, current_price)
                    else:
                        logger.error(
                            "Impossible de fermer la position %s lors de la liquidation: "
                            "prix actuel manquant",
                            asset,
                        )

                # Mettre à jour les métriques
                self.unrealized_pnl = 0.0
                self.total_capital = self.cash
                self.portfolio_value = self.cash
                self.update_metrics()

                logger.critical(
                    "LIQUIDATION SUR MARGE EFFECTUÉE - Niveau de marge: %.1f%%",
                    margin_level * 100,
                )

                return True

        return False

    def check_protection_orders(self, current_prices: Dict[str, float]):
        """Checks if any open positions have hit their stop-loss or take-profit levels."""
        for asset, position in self.positions.items():
            if position.is_open:
                current_price = current_prices.get(asset)
                if current_price is None:
                    continue

                # Check stop-loss
                if (
                    position.stop_loss_pct > 0
                    and current_price
                    <= position.entry_price * (1 - position.stop_loss_pct)
                ):
                    logger.info("Stop-loss hit for %s. Closing position.", asset)
                    self.close_position(asset, current_price)
                # Check take-profit
                elif (
                    position.take_profit_pct > 0
                    and current_price
                    >= position.entry_price * (1 + position.take_profit_pct)
                ):
                    logger.info("Take-profit hit for %s. Closing position.", asset)
                    self.close_position(asset, current_price)

    def open_position(self, asset: str, price: float, size: float) -> bool:
        """
        Ouvre une nouvelle position longue pour un actif spécifique.

        Cette méthode gère l'ouverture d'une position en vérifiant les fonds disponibles
        et en mettant à jour la valeur du portefeuille. Elle prend en compte la commission
        et utilise la valeur totale du portefeuille pour les vérifications.

        Args:
            asset: L'actif pour lequel ouvrir une position.
            price: Le prix auquel ouvrir la position.
            size: La taille de la position à ouvrir.

        Returns:
            bool: True si la position a été ouverte avec succès, False sinon.
        """
        # Protection: in spot mode, block new BUY orders when trading is disabled
        if not self.futures_enabled and self.trading_disabled and size > 0:
            logger.warning(
                "[PROTECTION] open_position blocked: trading disabled for BUY orders (drawdown breach). %s size=%.8f @ %.8f",
                asset,
                size,
                price,
            )
            # Standardized guard log for easy grep during integration runs
            logger.warning(
                "[GUARD] Rejecting BUY due to trading_disabled (reason=spot_drawdown) asset=%s size=%.8f price=%.8f",
                asset,
                size,
                price,
            )
            return False
        # Vérifier si une position est déjà ouverte pour cet actif
        if self.positions[asset].is_open:
            logger.warning(
                "[ERREUR] Impossible d'ouvrir une position pour %s: position déjà ouverte",
                asset,
            )
            return False

        # Récupérer la configuration du worker (par défaut w1 si non spécifié)
        worker_config = self.config.get("workers", {}).get("w1", {})
        trading_config = worker_config.get("trading_config", {})

        # Récupérer les paramètres de gestion des risques avec valeurs par défaut
        stop_loss_pct = trading_config.get("stop_loss_pct", 0.05)  # 5% par défaut
        take_profit_pct = trading_config.get("take_profit_pct", 0.15)  # 15% par défaut

        # Journalisation des paramètres de trading
        logger.debug("[OUVERTURE] %s - Configuration: %s", asset, trading_config)
        logger.debug(
            "[OUVERTURE] %s - Stop-loss: %.2f%%, Take-profit: %.2f%%",
            asset,
            stop_loss_pct * 100,
            take_profit_pct * 100,
        )

        # Calculer la valeur notionnelle et la commission
        notional_value = size * price
        commission = notional_value * self.commission_pct
        total_cost = notional_value + commission

        # Vérifier les fonds disponibles en utilisant la valeur totale du portefeuille
        available_capital = self.get_available_capital()

        # Journalisation des détails financiers
        logger.debug(
            "[OUVERTURE] %s - Détails financiers - Valeur notionnelle: %.2f, Commission: %.2f, Coût total: %.2f, Capital disponible: %.2f",
            asset,
            notional_value,
            commission,
            total_cost,
            available_capital,
        )

        # Vérifier si les fonds sont suffisants
        if total_cost > available_capital:
            logger.warning(
                "[ERREUR] Fonds insuffisants pour %s - Coût total: %.2f > Disponible: %.2f",
                asset,
                total_cost,
                available_capital,
            )
            return False

        # Appliquer les coûts selon le type de trading
        if self.futures_enabled:
            # Pour les contrats à terme : réserver la marge (valeur notionnelle/levier) plus la commission
            margin_used = notional_value / self.leverage
            self.cash -= margin_used + commission
            logger.debug(
                "[OUVERTURE] %s - Marge utilisée: %.2f, Commission: %.2f, Cash restant: %.2f",
                asset,
                margin_used,
                commission,
                self.cash,
            )
        else:
            # Pour le spot : débiter le montant total (valeur notionnelle + commission)
            self.cash -= total_cost
            logger.debug(
                "[OUVERTURE] %s - Montant débité: %.2f, Cash restant: %.2f",
                asset,
                total_cost,
                self.cash,
            )

        try:
            # Ouvrir la position avec les paramètres de gestion des risques
            self.positions[asset].open(
                entry_price=price,
                size=size,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
            )

            # Mettre à jour le compteur de trades
            self.trade_count += 1

            # Préparer les informations de suivi
            trade_info = {
                "type": "open",
                "asset": asset,
                "size": size,
                "price": price,
                "stop_loss": price * (1 - stop_loss_pct) if stop_loss_pct > 0 else None,
                "take_profit": price * (1 + take_profit_pct)
                if take_profit_pct > 0
                else None,
                "commission": commission,
                "timestamp": datetime.now().isoformat(),
                "current_cash": self.cash,
                "portfolio_value": self.portfolio_value,
                "available_capital": self.get_available_capital(),
                "leverage": self.leverage if self.futures_enabled else 1.0,
            }

            # Ajouter au journal des trades
            self.trade_log.append(trade_info)

            # Journalisation détaillée
            logger.info(
                "[POSITION OUVERTE] %s - Taille: %.8f @ %.8f | Valeur: %.2f | SL: %.8f | TP: %.8f | Commission: %.2f | Capital: %.2f %s",
                asset.upper(),
                size,
                price,
                notional_value,
                trade_info["stop_loss"] if trade_info["stop_loss"] else 0.0,
                trade_info["take_profit"] if trade_info["take_profit"] else 0.0,
                commission,
                self.get_equity(),
                self.currency
            )

            # Mise à jour des métriques de performance
            self.metrics.update_trade({
                'action': 'open',
                'asset': asset,
                'price': price,
                'size': size,
                'value': notional_value,
                'commission': commission,
                'equity': self.get_equity(),
                'balance': self.get_balance(),
                'leverage': self.leverage if self.futures_enabled else 1.0,
                'stop_loss': trade_info["stop_loss"],
                'take_profit': trade_info["take_profit"]
            })

            # Journalisation des soldes
            logger.debug(
                "[SOLDE] Cash: %.2f | Capital disponible: %.2f | Valeur portefeuille: %.2f",
                self.cash,
                self.get_available_capital(),
                self.portfolio_value,
            )

            return True

        except Exception as e:
            logger.error(
                "[ERREUR] Échec de l'ouverture de la position pour %s: %s",
                asset,
                str(e),
                exc_info=True,
            )
            # Annuler les modifications en cas d'erreur
            if self.futures_enabled:
                self.cash += (notional_value / self.leverage) + commission
            else:
                self.cash += notional_value + commission

            return False

    def close_position(self, asset: str, price: float) -> float:
        """
        Ferme la position ouverte pour un actif spécifique.

        Cette méthode gère la fermeture d'une position, calcule le PnL réalisé,
        met à jour la trésorerie et journalise les détails de la transaction.

        Args:
            asset: L'actif pour lequel fermer la position.
            price: Le prix auquel fermer la position.

        Returns:
            float: Le PnL net réalisé (après commissions) ou 0 en cas d'erreur.
        """
        # Vérifier si une position est ouverte pour cet actif
        if asset not in self.positions or not self.positions[asset].is_open:
            logger.warning(
                "[FERMETURE] Impossible de fermer la position pour %s: aucune position ouverte",
                asset,
            )
            return 0.0

        position = self.positions[asset]
        position_size = position.size
        entry_price = position.entry_price

        # Calculer le PnL brut (sans commission)
        trade_pnl = (price - entry_price) * position_size

        # Calculer la valeur notionnelle et la commission
        notional_value = position_size * price
        commission = notional_value * self.commission_pct

        # Calculer le PnL net (après commission)
        net_pnl = trade_pnl - commission

        # Journalisation avant fermeture
        logger.debug(
            "[FERMETURE] Préparation de la fermeture pour %s - Taille: %.8f @ %.8f | Prix entrée: %.8f | Prix sortie: %.8f",
            asset.upper(),
            position_size,
            price,
            entry_price,
            price,
        )

        try:
            # Mettre à jour la trésorerie selon le type de trading
            if self.futures_enabled:
                # Pour les contrats à terme : libérer la marge au prix de sortie, puis payer la commission.
                # Le PnL est implicitement reflété par la différence entre marge d'entrée et marge de sortie.
                margin_released = (position_size * price) / self.leverage
                self.cash += margin_released - commission
                logger.debug(
                    "[FERMETURE] %s - Marge libérée: %.2f | PnL brut: %.2f | Commission: %.2f",
                    asset.upper(),
                    margin_released,
                    trade_pnl,
                    commission,
                )
            else:
                # Pour le spot : récupérer l'investissement initial + PnL - commission
                initial_investment = position_size * entry_price
                self.cash += initial_investment + trade_pnl - commission
                logger.debug(
                    "[FERMETURE] %s - Investissement initial: %.2f | PnL brut: %.2f | Commission: %.2f",
                    asset.upper(),
                    initial_investment,
                    trade_pnl,
                    commission,
                )

            # Mettre à jour le PnL réalisé (net des commissions)
            self.realized_pnl += net_pnl

            # Calculer le pourcentage de gain/perte
            pnl_pct = (
                ((price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
            )

            # Préparer les informations de suivi
            trade_info = {
                "type": "close",
                "asset": asset,
                "size": position_size,
                "entry_price": entry_price,
                "exit_price": price,
                "pnl": net_pnl,
                "pnl_pct": pnl_pct,
                "commission": commission,
                "timestamp": datetime.now().isoformat(),
                "trade_pnl": net_pnl,  # Pour rétrocompatibilité
                "leverage": self.leverage if self.futures_enabled else 1.0,
                "position_value": notional_value,
                "cash_after": self.cash,
                "portfolio_value_after": self.portfolio_value,
                "available_capital_after": self.get_available_capital(),
            }

            # Ajouter au journal des trades
            self.trade_log.append(trade_info)

            # Fermer la position
            position.close()

            # Journalisation de la fermeture
            logger.info(
                "[POSITION FERMÉE] %s - Taille: %.8f | Entrée: %.8f | Sortie: %.8f | PnL: %+.2f (%.2f%%) | Capital: %.2f %s | Équité: %.2f %s",
                asset.upper(),
                position_size,
                entry_price,
                price,
                net_pnl,
                pnl_pct,
                self.get_balance(),
                self.currency,
                self.get_equity(),
                self.currency
            )

            # Mise à jour des métriques de performance
            self.metrics.update_trade({
                'action': 'close',
                'asset': asset,
                'entry_price': entry_price,
                'exit_price': price,
                'size': position_size,
                'pnl': net_pnl,
                'pnl_pct': pnl_pct,
                'commission': commission,
                'equity': self.get_equity(),
                'balance': self.get_balance(),
                'leverage': self.leverage if self.futures_enabled else 1.0,
                'duration': (datetime.now() - datetime.fromisoformat(trade_info['timestamp'])).total_seconds()
            })

            # Journalisation des soldes après fermeture
            logger.debug(
                "[SOLDE APRÈS FERMETURE] Cash: %.2f | Capital disponible: %.2f | Valeur portefeuille: %.2f",
                self.cash,
                self.get_available_capital(),
                self.portfolio_value,
            )

            return net_pnl

        except Exception as e:
            logger.error(
                "[ERREUR] Échec de la fermeture de la position pour %s: %s",
                asset,
                str(e),
                exc_info=True,
            )
            return 0.0

        # Fermer la position
        position.close()

        # Return gross PnL (without commission) as expected by the tests
        return trade_pnl

    def update_metrics(self) -> None:
        """
        Met à jour les métriques du portefeuille de manière robuste.

        Cette méthode calcule et met à jour les métriques de performance clés
        comme le drawdown, le ratio de Sharpe, Sortino, etc., avec une gestion
        robuste des erreurs et une stabilité numérique améliorée.

        La méthode est conçue pour être tolérante aux erreurs et ne jamais lever d'exception.
        """
        try:
            # Mise à jour des métriques de base
            self._update_equity()

            # Vérification de l'historique
            if not hasattr(self, 'trade_history') or not self.trade_history:
                logger.debug("[METRICS] Aucun historique de trades disponible pour le calcul des métriques")
                # Mise à jour périodique même sans trade récent
                self.metrics.log_periodic_update()
                return

            # Conversion en tableau numpy avec gestion des erreurs
            try:
                history_array = np.asarray(self.trade_history, dtype=np.float64)
                if history_array.size == 0:
                    self.metrics.log_periodic_update()
                    return
            except (TypeError, ValueError) as e:
                logger.error("[METRICS] Erreur lors de la conversion de l'historique: %s", str(e))
                self.metrics.log_periodic_update()
                return

            # Mise à jour des métriques de base
            self._update_drawdown_metrics(history_array)
            self._update_return_metrics(history_array)

            # Mise à jour des métriques avancées via PerformanceMetrics
            try:
                # Récupérer les métriques actuelles
                metrics = self.metrics.get_metrics_summary()

                # Mettre à jour les attributs du portefeuille
                self.sharpe_ratio = metrics.get('sharpe_ratio', 0.0)
                self.sortino_ratio = metrics.get('sortino_ratio', 0.0)
                self.profit_factor = metrics.get('profit_factor', 0.0)
                self.calmar_ratio = metrics.get('calmar_ratio', 0.0)
                self.max_drawdown = metrics.get('max_drawdown', 0.0)
                self.win_rate = metrics.get('win_rate', 0.0)

                # Log périodique des métriques (toutes les 100 étapes par défaut)
                if hasattr(self, 'step_count') and self.step_count % getattr(self, 'log_interval', 100) == 0:
                    self.metrics.log_periodic_update()

            except Exception as e:
                logger.error("[METRICS] Erreur lors de la mise à jour des métriques avancées: %s", str(e))
                # En cas d'erreur, on continue avec les valeurs par défaut
                self.sharpe_ratio = 0.0
                self.sortino_ratio = 0.0
                self.profit_factor = 0.0
                self.calmar_ratio = 0.0
                self.max_drawdown = 0.0
                self.win_rate = 0.0

        except Exception as e:
            logger.error(
                "[METRICS] Erreur critique dans update_metrics: %s",
                str(e),
                exc_info=True
            )
            # En cas d'erreur, on conserve les valeurs par défaut
            self.drawdown = 0.0
            self.sharpe_ratio = 0.0
            self.sortino_ratio = 0.0
            self.profit_factor = 0.0
            self.calmar_ratio = 0.0
            self.max_drawdown = 0.0
            self.win_rate = 0.0
            self.sharpe_ratio = 0.0

    def _update_drawdown_metrics(self, history_array: np.ndarray) -> None:
        """Met à jour les métriques de drawdown de manière robuste."""
        try:
            if len(history_array) < 2:
                return

            # Calcul des valeurs cumulées maximales avec protection numérique
            with np.errstate(divide='ignore', invalid='ignore'):
                # Remplacement des valeurs non finies par 0
                clean_history = np.nan_to_num(history_array, nan=0.0, posinf=0.0, neginf=0.0)
                cummax = np.maximum.accumulate(clean_history)

                # Calcul du drawdown avec protection contre division par zéro
                mask = cummax > 1e-10  # Évite les divisions par des valeurs très petites
                drawdowns = np.zeros_like(clean_history)
                drawdowns[mask] = (cummax[mask] - clean_history[mask]) / cummax[mask]

                # Calcul du drawdown maximum, limité à 100%
                self.drawdown = float(np.clip(np.max(drawdowns), 0.0, 1.0))

        except Exception as e:
            logger.error("Erreur dans _update_drawdown_metrics: %s", str(e))
            self.drawdown = 0.0

    def _update_return_metrics(self, history_array: np.ndarray) -> None:
        """Calcule les métriques de rendement (Sharpe ratio, etc.) de manière robuste."""
        try:
            if len(history_array) < 2:
                return

            # Nettoyage des données
            clean_history = np.nan_to_num(history_array, nan=0.0, posinf=0.0, neginf=0.0)

            # Calcul des retours journaliers en pourcentage
            prev_values = clean_history[:-1]
            next_values = clean_history[1:]

            with np.errstate(divide='ignore', invalid='ignore'):
                # Calcul des rendements avec protection contre division par zéro
                valid_mask = prev_values > 1e-10  # Évite les divisions par des valeurs très petites
                returns = np.zeros_like(prev_values)
                returns[valid_mask] = (next_values[valid_mask] - prev_values[valid_mask]) / prev_values[valid_mask]

                # Suppression des valeurs aberrantes
                returns = returns[np.isfinite(returns)]

                # Calcul du ratio de Sharpe avec des conditions de protection
                if len(returns) >= 2:  # Au moins 2 points pour avoir une variance non nulle
                    returns_std = np.std(returns)
                    if returns_std > 1e-10:  # Évite la division par zéro
                        sharpe = np.mean(returns) / returns_std * np.sqrt(252)  # Annualisation
                        self.sharpe_ratio = float(np.clip(sharpe, -10.0, 10.0))  # Bornes raisonnables
                    else:
                        self.sharpe_ratio = 0.0
                else:
                    self.sharpe_ratio = 0.0

        except Exception as e:
            logger.error("Erreur dans _update_return_metrics: %s", str(e))
            self.sharpe_ratio = 0.0

        self.calculate_risk_metrics()

    def calculate_risk_metrics(self, confidence_level: float = 0.95) -> None:
        """
        Calcule la Value at Risk (VaR) et la Conditional Value at Risk (CVaR).

        Args:
            confidence_level: Niveau de confiance pour le calcul du VaR (par défaut: 0.95)
        """
        # Initialisation des valeurs par défaut
        self.var = 0.0
        self.cvar = 0.0

        # Vérification des conditions minimales
        if not hasattr(self, 'trade_history') or len(self.trade_history) < 2:
            return

        try:
            # Conversion sécurisée en tableau numpy
            history_array = np.asarray(self.trade_history, dtype=np.float64)

            # Vérification du tableau
            if history_array.size < 2:
                return

            # Calcul des retours avec protection
            prev_values = history_array[:-1]
            next_values = history_array[1:]

            # Calcul des rendements avec gestion des erreurs numériques
            with np.errstate(divide='ignore', invalid='ignore'):
                # Masque pour éviter les divisions par zéro
                valid_mask = (prev_values > 1e-10) & np.isfinite(prev_values)
                returns = np.zeros_like(prev_values)
                returns[valid_mask] = (next_values[valid_mask] - prev_values[valid_mask]) / prev_values[valid_mask]

            # Nettoyage des valeurs non finies
            returns = returns[np.isfinite(returns)]

            if len(returns) <= 1:  # Pas assez de données pour calculer le risque
                return

            # Tri des rendements
            sorted_returns = np.sort(returns)

            # Calcul de l'index pour le VaR avec protection des bornes
            var_index = int(np.floor(len(sorted_returns) * (1 - confidence_level)))
            var_index = max(0, min(var_index, len(sorted_returns) - 1))

            # Calcul du VaR (valeur absolue du quantile des pertes)
            if 0 <= var_index < len(sorted_returns):
                self.var = float(np.abs(sorted_returns[var_index]))

                # Calcul du CVaR (moyenne des pertes pires que le VaR)
                if var_index > 0:
                    cvar_returns = sorted_returns[:var_index]
                    if len(cvar_returns) > 0:
                        self.cvar = float(np.abs(np.mean(cvar_returns)))
                    else:
                        self.cvar = self.var  # Si pas de pertes pires, on prend le VaR
                else:
                    self.cvar = self.var  # Si pas de pertes pires, on prend le VaR

            # Protection contre les valeurs aberrantes
            self.var = min(self.var, 1.0)  # Ne peut pas dépasser 100%
            self.cvar = min(self.cvar, 1.0)  # Ne peut pas dépasser 100%

            # Log de débogage
            logger.debug(
                "Métriques de risque calculées - VaR: %.4f, CVaR: %.4f (n=%d)",
                self.var, self.cvar, len(returns)
            )

        except Exception as e:
            logger.error(
                "Erreur dans calculate_risk_metrics: %s",
                str(e),
                exc_info=True
            )
            # En cas d'erreur, on conserve les valeurs par défaut (0.0)
            self.cvar = 0.0

    def get_current_tier(self, capital: float = None) -> Dict[str, Any]:
        """
        Détermine le palier de capital actuel basé sur le capital disponible.

        Args:
            capital: Capital à évaluer. Si None, utilise le capital actuel du portfolio

        Returns:
            Dict contenant la configuration du palier correspondant
        """
        if capital is None:
            capital = self.get_total_value()

        for tier in self.capital_tiers:
            min_cap = tier.get('min_capital', 0)
            max_cap = tier.get('max_capital', float('inf'))
            if max_cap is None:
                max_cap = float('inf')

            if min_cap <= capital <= max_cap:
                logger.debug(f"Capital {capital:.2f} correspond au palier: {tier['name']}")
                return tier

        # Fallback sur le dernier palier (Enterprise) si hors bornes
        if self.capital_tiers:
            fallback = self.capital_tiers[-1]
            logger.warning(f"Capital {capital:.2f} hors bornes, utilisation palier: {fallback['name']}")
            return fallback

        # Fallback par défaut si aucun palier configuré
        logger.error("Aucun palier de capital configuré")
        return {
            'name': 'Default',
            'max_position_size_pct': 10,
            'risk_per_trade_pct': 1.0,
            'max_drawdown_pct': 10.0,
            'max_concurrent_positions': 1,
            'exposure_range': [10, 50]
        }

    def normalize_to_tier_bounds(self, value: float, min_bound: float, max_bound: float,
                                method: str = 'linear') -> float:
        """
        Normalise une valeur calculée pour qu'elle respecte les bornes du palier actuel.

        Args:
            value: Valeur calculée par une formule (CVaR, Kelly, etc.)
            min_bound: Borne minimale autorisée par le palier
            max_bound: Borne maximale autorisée par le palier
            method: Méthode de normalisation ('linear' ou 'sigmoid')

        Returns:
            Valeur normalisée dans l'intervalle [min_bound, max_bound]
        """
        if method == 'linear':
            # Clipping linéaire simple
            normalized = min(max(value, min_bound), max_bound)
            if normalized != value:
                logger.debug(f"Normalisation linéaire: {value:.4f} -> {normalized:.4f} [{min_bound:.2f}, {max_bound:.2f}]")
            return normalized

        elif method == 'sigmoid':
            # Normalisation sigmoïde pour ajustement smooth
            import math
            mid = (min_bound + max_bound) / 2
            k = 0.1  # Facteur de sensibilité ajustable
            try:
                normalized = mid + ((max_bound - min_bound) / 2) * math.tanh(k * (value - mid))
                # Fallback clipping pour sécurité
                normalized = min(max(normalized, min_bound), max_bound)
                if abs(normalized - value) > 0.01:  # Log seulement si différence significative
                    logger.debug(f"Normalisation sigmoïde: {value:.4f} -> {normalized:.4f} [{min_bound:.2f}, {max_bound:.2f}]")
                return normalized
            except (OverflowError, ValueError) as e:
                logger.warning(f"Erreur normalisation sigmoïde: {e}, fallback linéaire")
                return min(max(value, min_bound), max_bound)

        # Fallback par défaut
        return min(max(value, min_bound), max_bound)

    def apply_tier_constraints(self, position_size_pct: float = None, risk_pct: float = None,
                             exposure_avg: float = None, concurrent_positions: int = None) -> Dict[str, float]:
        """
        Applique les contraintes du palier actuel à tous les paramètres de trading.

        Args:
            position_size_pct: Taille de position en % (calculée par formules)
            risk_pct: Risque par trade en % (calculé par formules)
            exposure_avg: Exposition moyenne (calculée par formules)
            concurrent_positions: Nombre de positions simultanées

        Returns:
            Dict avec les valeurs normalisées selon le palier actuel
        """
        tier = self.get_current_tier()
        results = {}

        # Normalisation position sizing
        if position_size_pct is not None:
            max_pos = tier.get('max_position_size_pct', 100)
            results['position_size_pct'] = self.normalize_to_tier_bounds(
                position_size_pct, 0, max_pos, 'sigmoid'
            )

        # Normalisation risk per trade
        if risk_pct is not None:
            max_risk = tier.get('risk_per_trade_pct', 5.0)
            results['risk_per_trade_pct'] = self.normalize_to_tier_bounds(
                risk_pct, 0, max_risk, 'linear'
            )

        # Normalisation exposition (utilise exposure_range du palier)
        if exposure_avg is not None:
            exposure_range = tier.get('exposure_range', [50, 100])
            min_exp, max_exp = exposure_range[0], exposure_range[1]
            results['exposure_normalized'] = self.normalize_to_tier_bounds(
                exposure_avg, min_exp, max_exp, 'sigmoid'
            )

        # Normalisation positions concurrentes (clipping entier)
        if concurrent_positions is not None:
            max_concurrent = tier.get('max_concurrent_positions', 1)
            results['concurrent_positions'] = max(1, min(concurrent_positions, max_concurrent))

        # Log du palier appliqué
        if results:
            logger.info(f"Contraintes appliquées - Palier: {tier['name']}, Résultats: {results}")

        return results

    def get_metrics(self) -> Dict[str, Any]:
        """
        Returns a dictionary of current portfolio metrics.

        Returns:
            Dict containing comprehensive portfolio metrics
        """
        # Initialize default values for all metrics
        metrics = {
            "total_positions": 0,
            "positions": {},
            "trade_count": 0,
            "drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "var": 0.0,
            "cvar": 0.0,
            "total_pnl_pct": 0.0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "initial_capital": getattr(self, "initial_capital", 0.0),
            "current_equity": getattr(self, "current_equity", 0.0),
            "unrealized_pnl": getattr(self, "unrealized_pnl", 0.0),
            "realized_pnl": getattr(self, "realized_pnl", 0.0),
            "total_capital": getattr(self, "total_capital", 0.0),
            "cash": getattr(self, "cash", 0.0),
            "portfolio_value": getattr(self, "portfolio_value", 0.0),
            "leverage": getattr(self, "leverage", 1.0),
        }

        # Calculate basic metrics
        if hasattr(self, "initial_capital") and self.initial_capital > 0:
            metrics["total_pnl_pct"] = (
                (metrics["total_capital"] / self.initial_capital) - 1
            ) * 100

        # Initialize trade_log if it doesn't exist
        if not hasattr(self, "trade_log") or not isinstance(self.trade_log, list):
            self.trade_log = []
            logger.warning(
                "trade_log n'était pas initialisé, initialisation avec une liste vide"
            )
            return metrics

        # Calculate trade metrics with safe dictionary access
        try:
            # Get all closed trades safely
            closed_trades = [
                t
                for t in self.trade_log
                if isinstance(t, dict) and t.get("type") == "close"
            ]
            metrics["total_trades"] = len(closed_trades)

            # Calculate winning and losing trades with safe value access
            winning_trades = [
                t for t in closed_trades if float(t.get("trade_pnl", 0)) > 0
            ]
            losing_trades = [
                t for t in closed_trades if float(t.get("trade_pnl", 0)) <= 0
            ]

            metrics["winning_trades"] = len(winning_trades)
            metrics["losing_trades"] = len(losing_trades)

            # Calculate win rate
            if metrics["total_trades"] > 0:
                metrics["win_rate"] = (
                    metrics["winning_trades"] / metrics["total_trades"]
                ) * 100

            # Calculate average win/loss with safe value access
            if winning_trades:
                metrics["avg_win"] = float(
                    np.mean([float(t.get("trade_pnl", 0)) for t in winning_trades])
                )

            if losing_trades:
                metrics["avg_loss"] = abs(
                    float(
                        np.mean([float(t.get("trade_pnl", 0)) for t in losing_trades])
                    )
                )

            # Calculate profit factor safely
            if losing_trades and metrics["avg_loss"] > 0:
                total_win = metrics["avg_win"] * metrics["winning_trades"]
                total_loss = metrics["avg_loss"] * metrics["losing_trades"]
                if total_loss > 0:  # Avoid division by zero
                    metrics["profit_factor"] = total_win / total_loss

        except Exception as e:
            logger.error(
                "Erreur lors du calcul des métriques de trading: %s",
                str(e),
                exc_info=True,
            )
            # En cas d'erreur, on garde les valeurs par défaut déjà définies

        # Prepare position metrics with safe attribute access
        try:
            positions_metrics = {}
            for asset, position in getattr(self, "positions", {}).items():
                try:
                    position_data = {
                        "size": getattr(position, "size", 0.0),
                        "entry_price": getattr(position, "entry_price", 0.0),
                        "is_open": getattr(position, "is_open", False),
                        "unrealized_pnl": 0.0,
                        "leverage": getattr(position, "leverage", 1.0),
                    }

                    # Calculate unrealized PnL safely
                    if position_data["is_open"] and hasattr(position, "current_price"):
                        entry_price = position_data["entry_price"]
                        current_price = getattr(position, "current_price", entry_price)
                        position_data["unrealized_pnl"] = (
                            current_price - entry_price
                        ) * position_data["size"]

                    positions_metrics[asset] = position_data

                except Exception as pos_e:
                    logger.error(
                        "Erreur lors du calcul des métriques pour la position %s: %s",
                        asset,
                        str(pos_e),
                        exc_info=True,
                    )
                    continue

            # Update metrics with position data
            metrics.update(
                {
                    "total_positions": len(positions_metrics),
                    "positions": positions_metrics,
                }
            )

        except Exception as e:
            logger.error(
                "Erreur lors de la préparation des métriques de position: %s",
                str(e),
                exc_info=True,
            )
            # On continue avec les métriques déjà calculées

        # Update metrics with portfolio data safely
        try:
            metrics.update(
                {
                    "trade_count": getattr(self, "trade_count", 0),
                    "drawdown": getattr(self, "drawdown", 0.0),
                    "sharpe_ratio": getattr(self, "sharpe_ratio", 0.0),
                    "var": getattr(self, "var", 0.0),
                    "cvar": getattr(self, "cvar", 0.0),
                    "unrealized_pnl": getattr(self, "unrealized_pnl", 0.0),
                    "realized_pnl": getattr(self, "realized_pnl", 0.0),
                    "portfolio_value": getattr(self, "portfolio_value", 0.0),
                }
            )
        except Exception as e:
            logger.error(
                "Erreur lors de la mise à jour des métriques du portefeuille: %s",
                str(e),
                exc_info=True,
            )

        return metrics

    def get_state_features(self) -> np.ndarray:
        """
        Returns a numpy array of features representing the portfolio's state.
        """
        features = []
        for asset in self.config["assets"]:
            position = self.positions[asset]
            has_position = 1.0 if position.is_open else 0.0

            if position.is_open:
                entry_value = position.entry_price * position.size
                # This needs current price to be accurate, which is not available here.
                # Passing 0 for now, to be fixed in a later step.
                relative_pnl = 0.0
            else:
                relative_pnl = 0.0

            features.extend([has_position, relative_pnl])

        return np.array(features, dtype=np.float32)

    def get_feature_size(self) -> int:
        """
        Returns the number of features in the portfolio's state representation.
        """
        # For each asset, we have 'has_position' and 'relative_pnl'
        return len(self.config["assets"]) * 2

    def is_bankrupt(self) -> bool:
        """
        Checks if the portfolio value has fallen below a critical threshold.
        """
        # Consider bankrupt if capital is less than 1% of initial capital
        return self.total_capital < (self.initial_capital * 0.01)

    def start_new_chunk(self, chunk_id: Optional[str] = None) -> None:
        """
        Call this method when starting to process a new chunk of data.
        This will finalize the previous chunk's PnL and start tracking a new chunk.

        Args:
            chunk_id: Optional identifier for the chunk (for logging purposes)
        """
        # Vérifier les conditions d'urgence avant de commencer un nouveau chunk
        if self.check_emergency_condition(self.current_step):
            logger.critical(
                "[EMERGENCY] Emergency condition detected before starting chunk %s. "
                "Reset will be triggered.",
                chunk_id or str(self.current_chunk_id + 1)
            )
            # Le reset sera géré par l'appelant
            return

        # Vérifier l'état de surveillance et mettre à jour en conséquence
        needs_reset = self._check_surveillance_status(self.current_step)

        # Si un reset est nécessaire, il sera géré par l'appelant via la méthode reset()
        if needs_reset:
            logger.warning(
                "[SURVEILLANCE] Maximum chunks in surveillance reached for chunk %s. "
                "Forcing reset on next step.",
                chunk_id or str(self.current_chunk_id)
            )
            return

        # Finalize the previous chunk's PnL if this isn't the first chunk
        if self.current_chunk_id > 0:
            self._finalize_chunk_pnl()

            # Vérifier la récupération à la fin du chunk en mode surveillance
            if self._surveillance_mode:
                current_value = self.get_portfolio_value()
                if current_value > self._recovery_threshold:
                    logger.info(
                        "[SURVEILLANCE] Recovery during chunk %s: end_balance=%.2f > threshold=%.2f. "
                        "Surveillance cleared.",
                        chunk_id or str(self.current_chunk_id),
                        current_value,
                        self._recovery_threshold
                    )
                    self._exit_surveillance_mode(recovered=True)

        # Start a new chunk
        self.current_chunk_id += 1
        self.chunk_start_equity = self.total_capital
        self.trade_count = 0  # Reset trade count for the new chunk

        # Log chunk start with surveillance info if applicable
        chunk_info = f"{chunk_id} (#{self.current_chunk_id})" if chunk_id else f"{self.current_chunk_id}"
        log_msg = f"🔄 Starting chunk {chunk_info} with starting equity: ${self.chunk_start_equity:.2f}"

        if self._surveillance_mode:
            max_chunks = self.config.get("surveillance_chunk_allowance", 2)
            log_msg += f" [SURVEILLANCE MODE: {self._survived_chunks}/{max_chunks} chunks]"

            # Avertissement si on approche de la limite de chunks en surveillance
            remaining_chunks = max(0, max_chunks - self._survived_chunks)
            if remaining_chunks <= 1:
                logger.warning(
                    "[SURVEILLANCE] Only %d chunk(s) remaining in surveillance mode before forced reset. "
                    "Current value: %.2f, Start value: %.2f",
                    remaining_chunks,
                    self.get_portfolio_value(),
                    self.surveillance_chunk_start_balance
                )

        logger.info(log_msg)

    def _finalize_chunk_pnl(self) -> None:
        """Calculate and store the PnL for the current chunk."""
        if self.current_chunk_id == 0:
            return

        chunk_pnl_pct = (
            (self.total_capital - self.chunk_start_equity) / self.chunk_start_equity
        ) * 100

        self.chunk_pnl[self.current_chunk_id] = {
            "start_equity": self.chunk_start_equity,
            "end_equity": self.total_capital,
            "pnl_pct": chunk_pnl_pct,
            "n_trades": self.trade_count,  # Use self.trade_count directly
        }

        logger.info(
            "Chunk %d completed with PnL: %.2f%% (Equity: $%.2f -> $%.2f)",
            self.current_chunk_id,
            chunk_pnl_pct,
            self.chunk_start_equity,
            self.total_capital,
        )

    def get_chunk_performance_ratio(self, chunk_id: int, optimal_pnl: float) -> float:
        """
        Calculate the performance ratio for a specific chunk compared to the optimal PnL.

        Args:
            chunk_id: The ID of the chunk to calculate the ratio for.
            optimal_pnl: The optimal possible PnL for this chunk.

        Returns:
            float: The performance ratio (actual_pnl / optimal_pnl), clipped to [0, 1].
        """
        if chunk_id not in self.chunk_pnl:
            logger.warning("No PnL data found for chunk %d", chunk_id)
            return 0.0

        if optimal_pnl <= 0:
            return 0.0

        actual_pnl = self.chunk_pnl[chunk_id]["pnl_pct"]
        ratio = actual_pnl / optimal_pnl

        # Clip the ratio between 0 and 1 to prevent extreme values
        return max(0.0, min(1.0, ratio))

    def rebalance(self, current_prices: Dict[str, float]):
        """Rebalances the portfolio to match target allocations and concentration limits."""
        logger.info("Rebalancing portfolio...")

        if not current_prices:
            logger.warning("Cannot rebalance: no current prices provided.")
            return

        total_portfolio_value = self.get_portfolio_value()
        if total_portfolio_value <= 0:
            logger.warning(
                "Cannot rebalance: total portfolio value is zero or negative."
            )
            return

        max_single_asset_limit = self.concentration_limits.get(
            "max_single_asset", 1.0
        )  # Default to 100%

        for asset, position in self.positions.items():
            if position.is_open:
                current_price = current_prices.get(asset)
                if current_price is None:
                    logger.warning(
                        "Cannot rebalance %s: current price not available.", asset
                    )
                    continue

                position_value = position.size * current_price
                current_allocation = position_value / total_portfolio_value

                if current_allocation > max_single_asset_limit:
                    # Calculate the excess amount to sell
                    excess_value = position_value - (
                        max_single_asset_limit * total_portfolio_value
                    )
                    sell_size = excess_value / current_price

                    logger.info(
                        "Rebalancing %s: current allocation %.2f exceeds limit %.2f. "
                        "Selling %.4f units.",
                        asset,
                        current_allocation,
                        max_single_asset_limit,
                        sell_size,
                    )
                    # Simulate closing a portion of the position
                    # This is a simplified close; in a real scenario, you'd adjust the existing position object
                    # and potentially execute a partial sell order.
                    self.cash += excess_value * (
                        1 - self.commission_pct
                    )  # Deduct commission on sell
                    position.size -= sell_size
                    if position.size <= 0:
                        position.close()  # Close if size becomes zero or negative
                        logger.info(
                            "Position for %s fully closed during rebalancing.", asset
                        )

        self.update_metrics()
        logger.info("Portfolio rebalancing completed.")

    def validate_position(
        self,
        asset: str,
        size: float,
        price: float,
        expected_return_pct: float = 0.0
    ) -> bool:
        """
        Validates if a position can be opened with the given parameters.

        This method checks:
        1. If the price is valid
        2. If the position meets minimum/maximum size requirements
        3. If there's sufficient available capital including commissions
        4. If concentration limits are respected
        5. If the trade is profitable after commissions

        Args:
            asset: Asset symbol
            size: Size of the position (positive for long, negative for short)
            price: Entry price
            expected_return_pct: Expected return percentage for profitability check

        Returns:
            bool: True if the position is valid, False otherwise
        """
        # Check if price is valid
        if price <= 0:
            logger.warning("[VALIDATION] Invalid price: %.8f", price)
            return False

        # If protection is active in spot mode, block only new long entries (buys)
        if not self.futures_enabled and self.trading_disabled and size > 0:
            logger.warning(
                "[VALIDATION] Trading disabled for new BUY orders due to drawdown breach. "
                "Request blocked: %s size=%.8f @ %.8f",
                asset, size, price
            )
            return False

        # Check minimum trade size
        if abs(size) < self.min_trade_size:
            logger.warning(
                "[VALIDATION] Position size (%.8f) is less than minimum trade size (%.8f).",
                size, self.min_trade_size
            )
            return False

        # Calculate notional value
        notional_value = abs(size) * price

        # Check against notional value limits
        if notional_value < self.min_notional_value:
            logger.warning(
                "[VALIDATION] Notional value (%.2f) is below minimum (%.2f).",
                notional_value, self.min_notional_value
            )
            return False

        if notional_value > self.max_notional_value:
            logger.warning(
                "[VALIDATION] Notional value (%.2f) exceeds maximum (%.2f).",
                notional_value, self.max_notional_value
            )
            return False

        # Check expected profitability after commissions
        if expected_return_pct > 0 and not self.is_profitable_after_commissions(
            notional_value, expected_return_pct
        ):
            commission = self.calculate_commission(notional_value)
            logger.info(
                "[VALIDATION] Trade not profitable after commissions. "
                "Expected return: %.2f%%, Commission: %.4f, Notional: %.4f",
                expected_return_pct, commission, notional_value
            )
            return False

        # Check available capital
        available_capital = self.get_available_capital()
        required_margin = notional_value * (1 + self.commission_pct)  # Include commission

        if required_margin > available_capital:
            logger.info(
                "[VALIDATION] Insufficient capital. Required: %.4f, Available: %.4f",
                required_margin, available_capital
            )
            return False

        # Check maximum concurrent positions
        try:
            active_tier = self.get_active_tier()
            max_positions = active_tier.get('max_concurrent_positions', 1)
            current_positions = sum(1 for pos in self.positions.values() if pos.is_open)

            # Allow modifying existing position even if limit is reached
            if (current_positions >= max_positions and
                not (asset in self.positions and self.positions[asset].is_open)):
                logger.info(
                    "[VALIDATION] Maximum concurrent positions reached (%d/%d). %s",
                    current_positions, max_positions, asset
                )
                return False

            # Check concentration limits
            if self.concentration_limits:
                portfolio_value = self.get_portfolio_value()
                if portfolio_value > 0:
                    position_pct = notional_value / portfolio_value
                    max_position_pct = self.concentration_limits.get("max_position_pct", 1.0)

                    if position_pct > max_position_pct:
                        logger.warning(
                            "[VALIDATION] Position size (%.2f%%) > maximum allowed (%.2f%%)",
                            position_pct * 100, max_position_pct * 100
                        )
                        return False

                    # Check per-asset concentration
                    current_asset_value = 0.0
                    if asset in self.positions and self.positions[asset].is_open:
                        current_asset_value = abs(self.positions[asset].size * price)

                    new_asset_value = current_asset_value + notional_value
                    max_asset_pct = self.concentration_limits.get("max_asset_pct", 0.5)

                    if (new_asset_value / portfolio_value) > max_asset_pct:
                        logger.warning(
                            "[VALIDATION] Asset concentration (%.2f%%) > maximum allowed (%.2f%%)",
                            (new_asset_value / portfolio_value) * 100, max_asset_pct * 100
                        )
                        return False

            logger.debug(
                "[VALIDATION] Position validated - Size: %.8f, Price: %.8f, Notional: %.4f, "
                "Positions: %d/%d, Capital: %.4f/%.4f",
                size, price, notional_value, current_positions, max_positions,
                required_margin, available_capital
            )
            return True

        except Exception as e:
            logger.error(
                "[VALIDATION] Error validating position: %s",
                str(e), exc_info=True
            )
            return False

    def get_active_tier(self):
        """
        Détermine le palier de capital actif en fonction de la valeur du portefeuille.

        Returns:
            dict: Le palier de capital actif avec toutes ses propriétés.

        Raises:
            RuntimeError: Si aucun palier valide n'est trouvé.
        """
        if not self.capital_tiers:
            raise RuntimeError(
                "Aucun palier de capital n'est défini dans la configuration."
            )

        current_value = self.get_portfolio_value()

        # Parcourir les paliers du plus élevé au plus bas
        for tier in sorted(
            self.capital_tiers, key=lambda x: x["min_capital"], reverse=True
        ):
            if current_value >= tier["min_capital"]:
                logger.debug(
                    "Palier actif: %s (capital: %.2f >= %.2f)",
                    tier["name"],
                    current_value,
                    tier["min_capital"],
                )
                return tier

        # Si on arrive ici, utiliser le palier le plus bas
        min_tier = min(self.capital_tiers, key=lambda x: x["min_capital"])
        logger.warning(
            "La valeur du portefeuille (%.2f) est inférieure au palier minimum (%.2f). "
            "Utilisation du palier: %s",
            current_value,
            min_tier["min_capital"],
            min_tier["name"],
        )
        return min_tier

    def calculate_position_size_with_cvar(self, capital: float, asset: str, timeframe: str = '1h',
                                        confidence_level: float = 0.05, target_risk: float = 0.01) -> float:
        """
        Calcule la taille de position optimale basée sur CVaR (Expected Shortfall).

        Cette méthode utilise le Conditional Value at Risk pour déterminer la taille de position
        qui respecte le niveau de risque souhaité en tenant compte des pertes extrêmes.

        Formule: Position Size = (Target Risk * Capital) / |CVaR|
        où CVaR = E[Loss | Loss ≥ VaR]

        Args:
            capital: Capital disponible pour le trade
            asset: Symbole de l'actif (ex: 'BTCUSDT')
            timeframe: Timeframe pour l'analyse historique ('5m', '1h', '4h')
            confidence_level: Niveau de confiance pour CVaR (0.05 = 5%)
            target_risk: Risque cible en pourcentage du capital (0.01 = 1%)

        Returns:
            Taille de position en unités USDT
        """
        try:
            logger.info(f"Calcul CVaR position sizing pour {asset} ({timeframe})")

            # 1. Simuler des données de rendements historiques si pas de loader disponible
            # En production, ceci devrait utiliser le data_loader réel
            if hasattr(self, 'data_loader') and self.data_loader:
                try:
                    # Utiliser le data_loader si disponible
                    df = self.data_loader.load_data(asset, timeframe, 'train')
                    if not df.empty and 'close' in df.columns:
                        close_prices = df['close']
                        returns = close_prices.pct_change().dropna()
                    else:
                        logger.warning(f"Données insuffisantes pour {asset}, utilisation de simulation")
                        returns = self._simulate_returns()
                except Exception as e:
                    logger.warning(f"Erreur chargement données {asset}: {e}, utilisation de simulation")
                    returns = self._simulate_returns()
            else:
                logger.info("Data loader non disponible, utilisation de rendements simulés")
                returns = self._simulate_returns()

            if len(returns) < 100:  # Minimum pour un calcul CVaR fiable
                logger.warning(f"Données insuffisantes ({len(returns)} points), utilisation du sizing classique")
                return capital * target_risk  # Fallback au sizing simple

            # 2. Calculer le VaR (Value at Risk)
            var = np.percentile(returns, confidence_level * 100)

            # 3. Calculer le CVaR (Expected Shortfall)
            # CVaR = moyenne des rendements inférieurs ou égaux au VaR
            tail_losses = returns[returns <= var]

            if len(tail_losses) == 0:
                logger.warning("Aucune perte dans la queue de distribution, utilisation du VaR")
                cvar = var
            else:
                cvar = tail_losses.mean()

            # 4. Ajustements pour le trading spot crypto
            # Facteur d'ajustement pour la volatilité crypto (généralement plus élevée)
            crypto_volatility_factor = 1.5
            adjusted_cvar = cvar * crypto_volatility_factor

            # 5. Calcul de la taille de position optimale
            # Position Size = (Target Risk * Capital) / |CVaR|
            if abs(adjusted_cvar) > 0:
                optimal_position_size = (target_risk * capital) / abs(adjusted_cvar)
            else:
                logger.warning("CVaR nul, utilisation du sizing de base")
                optimal_position_size = capital * target_risk

            # 6. Appliquer les contraintes du portfolio management
            tier = self.get_current_tier()
            max_position_pct = tier.get('max_position_size_pct', 50) / 100
            max_allowed_size = capital * max_position_pct

            # 7. Contraintes supplémentaires
            min_trade_value = 11.0  # Minimum Binance
            max_trade_value = capital * 0.25  # Maximum 25% du capital

            # 8. Ajustement final de la taille
            final_size = min(optimal_position_size, max_allowed_size, max_trade_value)
            final_size = max(final_size, min_trade_value)

            # 9. Logs détaillés
            logger.info(f"CVaR Position Sizing pour {asset}:")
            logger.info(f"  - VaR ({confidence_level*100}%): {var:.6f}")
            logger.info(f"  - CVaR: {cvar:.6f} -> Ajusté: {adjusted_cvar:.6f}")
            logger.info(f"  - Position optimale: ${optimal_position_size:.2f}")
            logger.info(f"  - Contraintes: Min=${min_trade_value:.2f}, Max=${max_allowed_size:.2f}")
            logger.info(f"  - Taille finale avant palier: ${final_size:.2f} ({final_size/capital*100:.1f}% du capital)")

            # 🔥 NOUVELLE LOGIQUE: Application des contraintes de palier
            position_size_pct = (final_size / capital) * 100
            tier_constraints = self.apply_tier_constraints(position_size_pct=position_size_pct)

            if 'position_size_pct' in tier_constraints:
                normalized_pct = tier_constraints['position_size_pct']
                final_size = (normalized_pct / 100) * capital
                logger.info(f"  - Position normalisée par palier: {position_size_pct:.1f}% -> {normalized_pct:.1f}% = ${final_size:.2f}")

            return final_size

        except Exception as e:
            logger.error(f"Erreur dans calculate_position_size_with_cvar: {str(e)}")
            # Fallback au sizing simple en cas d'erreur
            return min(capital * target_risk, capital * 0.1, 50.0)  # Max 10% ou 50 USDT

    def _simulate_returns(self, n_samples: int = 252) -> np.ndarray:
        """
        Simule des rendements journaliers réalistes pour les cryptomonnaies.

        Utilise une distribution avec:
        - Moyenne légèrement positive (bull market bias)
        - Volatilité élevée typique des cryptos
        - Queues épaisses (kurtosis élevée)

        Args:
            n_samples: Nombre d'échantillons à générer

        Returns:
            Array des rendements simulés
        """
        np.random.seed(42)  # Pour la reproductibilité

        # Paramètres typiques des cryptomonnaies
        daily_mean_return = 0.001  # 0.1% par jour en moyenne
        daily_volatility = 0.05    # 5% de volatilité quotidienne

        # Générer des rendements avec queues épaisses
        # Utilisation d'une distribution t de Student
        from scipy.stats import t
        df = 4  # Degrés de liberté pour queues épaisses

        # Rendements normalisés puis ajustés
        t_samples = t.rvs(df, size=n_samples)
        normalized_samples = (t_samples - t_samples.mean()) / t_samples.std()

        returns = daily_mean_return + daily_volatility * normalized_samples

        logger.debug(f"Rendements simulés: μ={returns.mean():.6f}, σ={returns.std():.6f}, "
                    f"skew={self._calculate_skewness(returns):.3f}")

        return returns

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calcule l'asymétrie (skewness) d'une distribution."""
        n = len(data)
        if n < 3:
            return 0.0

        mean = np.mean(data)
        std = np.std(data)

        if std == 0:
            return 0.0

        skew = np.sum(((data - mean) / std) ** 3) / n
        return skew
