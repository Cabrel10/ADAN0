#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de Paper Trading en temps réel pour ADAN.
Exécute des trades simulés sur Binance Testnet en utilisant un modèle entraîné.
"""
import os
import sys
import time
import signal
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
import json

# Ajouter le répertoire src au PYTHONPATH
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, '..'))

from src.adan_trading_bot.common.utils import load_config, get_logger
from src.adan_trading_bot.exchange_api.connector import (
    get_exchange_client,
    test_exchange_connection,
    ExchangeConnectionError,
    ExchangeConfigurationError
)

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.live import Live
    from rich.text import Text
    from rich.progress import Progress, SpinnerColumn, TextColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Configuration du logger
logger = get_logger(__name__)
console = Console() if RICH_AVAILABLE else None

class PaperTradingAgent:
    """Agent de paper trading en temps réel."""

    def __init__(self, model_path, config, exchange_client, initial_capital=15.0):
        """
        Initialise l'agent de paper trading.

        Args:
            model_path: Chemin vers le modèle entraîné
            config: Configuration complète
            exchange_client: Client d'exchange CCXT
            initial_capital: Capital initial en USDT
        """
        self.model_path = model_path
        self.config = config
        self.exchange = exchange_client
        self.initial_capital = initial_capital
        self.current_capital = initial_capital

        # Charger le modèle
        self.model = self._load_model()

        # Configuration des actifs
        self.assets = config.get('data', {}).get('assets', ['ADAUSDT', 'BTCUSDT', 'ETHUSDT'])
        self.symbols = [f"{asset.replace('USDT', '')}/USDT" for asset in self.assets]

        # Positions actuelles
        self.positions = {symbol: 0.0 for symbol in self.symbols}
        self.usdt_balance = initial_capital

        # Historique des trades
        self.trade_history = []
        self.pnl_history = []

        # Métriques de performance
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0

        # État de fonctionnement
        self.running = True
        self.start_time = datetime.now()

        logger.info(f"🤖 Paper Trading Agent initialisé")
        logger.info(f"💰 Capital initial: ${initial_capital:.2f}")
        logger.info(f"📊 Actifs: {self.symbols}")

    def _load_model(self):
        """Charge le modèle PPO entraîné."""
        try:
            from stable_baselines3 import PPO
            model = PPO.load(self.model_path)
            logger.info(f"✅ Modèle chargé: {self.model_path}")
            return model
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement du modèle: {e}")
            raise

    def get_market_data(self, symbol, timeframe='1m', limit=50):
        """
        Récupère les données de marché en temps réel.

        Args:
            symbol: Symbole de trading (ex: 'BTC/USDT')
            timeframe: Timeframe ('1m', '5m', '1h')
            limit: Nombre de bougies à récupérer

        Returns:
            pd.DataFrame: Données OHLCV
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')

            return df
        except Exception as e:
            logger.error(f"❌ Erreur récupération données {symbol}: {e}")
            return None

    def calculate_features(self, market_data):
        """
        Calcule les features nécessaires pour le modèle.

        Args:
            market_data: Dict {symbol: DataFrame} avec données OHLCV

        Returns:
            np.array: Features formatées pour le modèle
        """
        try:
            # Simuler le format attendu par le modèle
            # En pratique, cela devrait utiliser StateBuilder
            features = []

            for symbol in self.symbols:
                if symbol in market_data and market_data[symbol] is not None:
                    df = market_data[symbol]
                    if not df.empty:
                        # Utiliser les dernières valeurs OHLCV
                        last_row = df.iloc[-1]
                        features.extend([
                            last_row['open'],
                            last_row['high'],
                            last_row['low'],
                            last_row['close'],
                            last_row['volume']
                        ])
                    else:
                        # Valeurs par défaut si pas de données
                        features.extend([1.0, 1.0, 1.0, 1.0, 1000.0])
                else:
                    # Valeurs par défaut si symbol manquant
                    features.extend([1.0, 1.0, 1.0, 1.0, 1000.0])

            # Ajouter des features supplémentaires pour atteindre la dimension attendue
            # (adapter selon le modèle entraîné)
            while len(features) < 100:  # Dimension approximative
                features.append(0.0)

            return np.array(features[:100])  # Limiter à la dimension attendue

        except Exception as e:
            logger.error(f"❌ Erreur calcul des features: {e}")
            return np.zeros(100)  # Fallback

    def execute_trade(self, symbol, action, amount):
        """
        Exécute un trade simulé (paper trading).

        Args:
            symbol: Symbole à trader
            action: 'buy' ou 'sell'
            amount: Montant en USDT ou quantité d'actif

        Returns:
            dict: Résultat du trade
        """
        try:
            # Récupérer le prix actuel
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']

            trade_result = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'action': action,
                'price': current_price,
                'amount': 0,
                'value_usdt': 0,
                'success': False,
                'pnl': 0
            }

            if action == 'buy':
                # Calculer la quantité à acheter
                max_amount = min(amount, self.usdt_balance * 0.95)  # Garder 5% de marge
                quantity = max_amount / current_price

                if max_amount >= 1.0:  # Minimum 1 USDT
                    # Simuler l'exécution
                    self.positions[symbol] += quantity
                    self.usdt_balance -= max_amount

                    trade_result.update({
                        'amount': quantity,
                        'value_usdt': max_amount,
                        'success': True
                    })

                    logger.info(f"✅ BUY {quantity:.6f} {symbol} @ ${current_price:.2f} = ${max_amount:.2f}")

            elif action == 'sell':
                # Vendre une partie ou la totalité de la position
                current_position = self.positions.get(symbol, 0)
                quantity_to_sell = min(amount, current_position)

                if quantity_to_sell > 0:
                    # Calculer la valeur de vente
                    sell_value = quantity_to_sell * current_price

                    # Simuler l'exécution
                    self.positions[symbol] -= quantity_to_sell
                    self.usdt_balance += sell_value

                    trade_result.update({
                        'amount': quantity_to_sell,
                        'value_usdt': sell_value,
                        'success': True
                    })

                    logger.info(f"✅ SELL {quantity_to_sell:.6f} {symbol} @ ${current_price:.2f} = ${sell_value:.2f}")

            # Enregistrer le trade
            if trade_result['success']:
                self.trade_history.append(trade_result)
                self.total_trades += 1

            return trade_result

        except Exception as e:
            logger.error(f"❌ Erreur exécution trade {action} {symbol}: {e}")
            return {'success': False, 'error': str(e)}

    def calculate_portfolio_value(self):
        """Calcule la valeur totale du portefeuille."""
        try:
            total_value = self.usdt_balance

            for symbol, quantity in self.positions.items():
                if quantity > 0:
                    ticker = self.exchange.fetch_ticker(symbol)
                    current_price = ticker['last']
                    total_value += quantity * current_price

            return total_value
        except Exception as e:
            logger.error(f"❌ Erreur calcul valeur portefeuille: {e}")
            return self.current_capital

    def get_trading_action(self, market_data):
        """
        Détermine l'action de trading basée sur le modèle.

        Args:
            market_data: Données de marché actuelles

        Returns:
            dict: Action de trading
        """
        try:
            # Calculer les features
            features = self.calculate_features(market_data)

            # Prédiction du modèle
            action, _ = self.model.predict(features.reshape(1, -1), deterministic=True)

            # Interpréter l'action (adapter selon votre environnement)
            # Action 0 = HOLD, 1 = BUY première crypto, 2 = SELL première crypto, etc.
            action_int = int(action[0])

            if action_int == 0:
                return {'type': 'hold'}
            elif action_int % 2 == 1:  # Actions impaires = BUY
                symbol_idx = (action_int - 1) // 2
                if symbol_idx < len(self.symbols):
                    return {
                        'type': 'buy',
                        'symbol': self.symbols[symbol_idx],
                        'amount': min(5.0, self.usdt_balance * 0.2)  # 20% du capital ou 5$ max
                    }
            else:  # Actions paires = SELL
                symbol_idx = (action_int - 2) // 2
                if symbol_idx < len(self.symbols):
                    symbol = self.symbols[symbol_idx]
                    position = self.positions.get(symbol, 0)
                    if position > 0:
                        return {
                            'type': 'sell',
                            'symbol': symbol,
                            'amount': position * 0.5  # Vendre 50% de la position
                        }

            return {'type': 'hold'}

        except Exception as e:
            logger.error(f"❌ Erreur détermination action: {e}")
            return {'type': 'hold'}

    def update_metrics(self):
        """Met à jour les métriques de performance."""
        try:
            current_value = self.calculate_portfolio_value()
            pnl = current_value - self.initial_capital
            roi = (pnl / self.initial_capital) * 100

            self.current_capital = current_value
            self.total_pnl = pnl

            # Ajouter à l'historique
            self.pnl_history.append({
                'timestamp': datetime.now(),
                'portfolio_value': current_value,
                'pnl': pnl,
                'roi': roi
            })

            return {
                'portfolio_value': current_value,
                'pnl': pnl,
                'roi': roi,
                'total_trades': self.total_trades,
                'usdt_balance': self.usdt_balance,
                'positions': dict(self.positions)
            }

        except Exception as e:
            logger.error(f"❌ Erreur mise à jour métriques: {e}")
            return {}

    def create_status_table(self, metrics):
        """Crée un tableau de statut Rich."""
        if not RICH_AVAILABLE:
            return f"Portfolio: ${metrics.get('portfolio_value', 0):.2f} | ROI: {metrics.get('roi', 0):.2f}%"

        table = Table(title="📊 Paper Trading Status", show_header=True, header_style="bold magenta")
        table.add_column("Métrique", style="dim", width=20)
        table.add_column("Valeur", style="bold", width=15)
        table.add_column("Status", width=10)

        portfolio_value = metrics.get('portfolio_value', 0)
        roi = metrics.get('roi', 0)

        # Statut ROI
        if roi > 5:
            roi_status = "🟢 Excellent"
        elif roi > 0:
            roi_status = "🟡 Positif"
        elif roi > -5:
            roi_status = "🟠 Attention"
        else:
            roi_status = "🔴 Critique"

        table.add_row("💰 Capital Actuel", f"${portfolio_value:.2f}", "")
        table.add_row("📈 PnL", f"${metrics.get('pnl', 0):.2f}", "")
        table.add_row("📊 ROI", f"{roi:.2f}%", roi_status)
        table.add_row("🔄 Total Trades", f"{metrics.get('total_trades', 0)}", "")
        table.add_row("💵 USDT Libre", f"${metrics.get('usdt_balance', 0):.2f}", "")

        return table

    def run(self, duration_minutes=30, update_interval=10):
        """
        Lance le paper trading pour une durée spécifiée.

        Args:
            duration_minutes: Durée en minutes
            update_interval: Intervalle entre les updates en secondes
        """
        logger.info(f"🚀 Démarrage du paper trading pour {duration_minutes} minutes")

        end_time = datetime.now() + timedelta(minutes=duration_minutes)

        def signal_handler(sig, frame):
            logger.info("⏹️  Arrêt demandé par l'utilisateur")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)

        try:
            if RICH_AVAILABLE:
                with Live(auto_refresh=False) as live:
                    while self.running and datetime.now() < end_time:
                        # Récupérer les données de marché
                        market_data = {}
                        for symbol in self.symbols:
                            market_data[symbol] = self.get_market_data(symbol)

                        # Déterminer l'action de trading
                        action = self.get_trading_action(market_data)

                        # Exécuter le trade si nécessaire
                        if action['type'] != 'hold':
                            self.execute_trade(action['symbol'], action['type'], action['amount'])

                        # Mettre à jour les métriques
                        metrics = self.update_metrics()

                        # Mettre à jour l'affichage
                        status_table = self.create_status_table(metrics)
                        remaining_time = end_time - datetime.now()

                        panel = Panel(
                            status_table,
                            title=f"⏰ Paper Trading - {remaining_time.total_seconds()/60:.1f}min restantes",
                            subtitle=f"🕐 Dernière MAJ: {datetime.now().strftime('%H:%M:%S')}"
                        )

                        live.update(panel)
                        live.refresh()

                        time.sleep(update_interval)
            else:
                # Mode sans Rich
                while self.running and datetime.now() < end_time:
                    market_data = {}
                    for symbol in self.symbols:
                        market_data[symbol] = self.get_market_data(symbol)

                    action = self.get_trading_action(market_data)

                    if action['type'] != 'hold':
                        self.execute_trade(action['symbol'], action['type'], action['amount'])

                    metrics = self.update_metrics()

                    remaining_time = end_time - datetime.now()
                    print(f"Portfolio: ${metrics.get('portfolio_value', 0):.2f} | "
                          f"ROI: {metrics.get('roi', 0):.2f}% | "
                          f"Restant: {remaining_time.total_seconds()/60:.1f}min")

                    time.sleep(update_interval)

        except Exception as e:
            logger.error(f"❌ Erreur pendant le paper trading: {e}")

        # Résumé final
        self.show_final_summary()

    def show_final_summary(self):
        """Affiche le résumé final du paper trading."""
        final_metrics = self.update_metrics()
        duration = datetime.now() - self.start_time

        logger.info("=" * 60)
        logger.info("🎉 PAPER TRADING TERMINÉ")
        logger.info("=" * 60)
        logger.info(f"⏱️  Durée: {duration.total_seconds()/60:.1f} minutes")
        logger.info(f"💰 Capital Initial: ${self.initial_capital:.2f}")
        logger.info(f"💰 Capital Final: ${final_metrics.get('portfolio_value', 0):.2f}")
        logger.info(f"📈 PnL Total: ${final_metrics.get('pnl', 0):.2f}")
        logger.info(f"📊 ROI: {final_metrics.get('roi', 0):.2f}%")
        logger.info(f"🔄 Total Trades: {final_metrics.get('total_trades', 0)}")
        logger.info("=" * 60)

        # Sauvegarder les résultats
        self.save_results()

    def save_results(self):
        """Sauvegarde les résultats du paper trading."""
        try:
            results = {
                'session_info': {
                    'start_time': self.start_time.isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'initial_capital': self.initial_capital,
                    'final_capital': self.current_capital,
                    'total_pnl': self.total_pnl,
                    'roi_percent': (self.total_pnl / self.initial_capital) * 100,
                    'total_trades': self.total_trades
                },
                'trade_history': [
                    {
                        'timestamp': trade['timestamp'].isoformat(),
                        'symbol': trade['symbol'],
                        'action': trade['action'],
                        'price': trade['price'],
                        'amount': trade['amount'],
                        'value_usdt': trade['value_usdt']
                    }
                    for trade in self.trade_history
                ],
                'pnl_history': [
                    {
                        'timestamp': pnl['timestamp'].isoformat(),
                        'portfolio_value': pnl['portfolio_value'],
                        'pnl': pnl['pnl'],
                        'roi': pnl['roi']
                    }
                    for pnl in self.pnl_history
                ]
            }

            # Créer le répertoire de reports
            reports_dir = Path('reports/paper_trading')
            reports_dir.mkdir(parents=True, exist_ok=True)

            # Sauvegarder avec timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = reports_dir / f'paper_trading_{timestamp}.json'

            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            logger.info(f"💾 Résultats sauvegardés: {results_file}")

        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde résultats: {e}")

def main():
    parser = argparse.ArgumentParser(description='ADAN Paper Trading Agent')
    parser.add_argument('--model_path', type=str, default='models/final_model.zip',
                       help='Chemin vers le modèle entraîné')
    parser.add_argument('--capital', type=float, default=15.0,
                       help='Capital initial en USDT')
    parser.add_argument('--duration', type=int, default=30,
                       help='Durée du paper trading en minutes')
    parser.add_argument('--testnet', action='store_true',
                       help='Utiliser Binance Testnet')
    parser.add_argument('--update_interval', type=int, default=10,
                       help='Intervalle de mise à jour en secondes')

    args = parser.parse_args()

    try:
        # Charger la configuration
        config = load_config('config/main_config.yaml')

        # Vérifier le modèle
        if not os.path.exists(args.model_path):
            # Essayer le modèle interrompu
            args.model_path = 'models/interrupted_model.zip'
            if not os.path.exists(args.model_path):
                logger.error(f"❌ Modèle non trouvé: {args.model_path}")
                return 1

        # Initialiser la connexion exchange
        if args.testnet:
            logger.info("🔌 Connexion au Binance Testnet...")
            exchange = get_exchange_client(config)

            # Test de connexion
            test_result = test_exchange_connection(exchange)
            if test_result.get('errors'):
                logger.error("❌ Problèmes de connexion détectés")
                return 1
        else:
            logger.error("❌ Mode Live non supporté pour le moment")
            return 1

        # Créer et lancer l'agent
        agent = PaperTradingAgent(
            model_path=args.model_path,
            config=config,
            exchange_client=exchange,
            initial_capital=args.capital
        )

        agent.run(
            duration_minutes=args.duration,
            update_interval=args.update_interval
        )

        return 0

    except KeyboardInterrupt:
        logger.info("⏹️  Paper trading interrompu par l'utilisateur")
        return 0
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
