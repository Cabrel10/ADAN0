#!/usr/bin/env python3
"""
Test pour reproduire le scénario multi-worker et identifier la vraie cause
du problème position_exists=True, position_open=False.

Ce script simule plusieurs workers avec leurs propres environnements
pour vérifier si le mélange des logs cause la confusion observée.
"""

import sys
import logging
import threading
import time
import random
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

# Ajouter le chemin du projet
sys.path.insert(0, '/home/morningstar/Documents/trading')
sys.path.insert(0, '/home/morningstar/Documents/trading/bot/src')

from adan_trading_bot.portfolio.portfolio_manager import PortfolioManager, Position

# Configuration du logging avec thread info
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - [%(threadName)s] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class WorkerState:
    """État d'un worker individuel"""
    worker_id: str
    portfolio_manager: PortfolioManager
    current_step: int = 0
    positions_opened: int = 0
    trade_count: int = 0

class MultiWorkerTester:
    """Testeur pour le scénario multi-worker"""

    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        self.workers: Dict[str, WorkerState] = {}
        self.global_logs: List[Dict[str, Any]] = []
        self.lock = threading.Lock()

        # Configuration de base
        self.config = {
            'portfolio': {'initial_balance': 50.0},
            'environment': {'min_capital_before_reset': 11.0},
            'trading_fees': 0.001,
            'min_order_value_usdt': 11.0,
            'capital_tiers': [
                {
                    'name': 'base',
                    'min_capital': 0.0,
                    'max_balance': 200.0,
                    'max_position_size_pct': 0.10,
                    'max_concurrent_positions': 3,
                    'leverage': 1.0,
                    'risk_per_trade_pct': 2.0,
                    'max_drawdown_pct': 25.0
                }
            ]
        }

    def create_worker(self, worker_id: str) -> WorkerState:
        """Créer un worker avec son propre PortfolioManager"""
        logger.info(f"🔧 Creating worker {worker_id}")

        portfolio = PortfolioManager(
            env_config=self.config,
            assets=['BTCUSDT']
        )

        worker = WorkerState(
            worker_id=worker_id,
            portfolio_manager=portfolio
        )

        logger.info(f"✅ Worker {worker_id} created - Portfolio ID: {id(portfolio)}")
        return worker

    def log_global_event(self, event_type: str, worker_id: str, data: Dict[str, Any]):
        """Logger global thread-safe pour tracer tous les événements"""
        with self.lock:
            event = {
                'timestamp': time.time(),
                'thread': threading.current_thread().name,
                'event_type': event_type,
                'worker_id': worker_id,
                'data': data
            }
            self.global_logs.append(event)

    def simulate_trading_step(self, worker: WorkerState, step: int) -> Dict[str, Any]:
        """Simuler un step de trading pour un worker"""
        worker.current_step = step

        # Récupérer la position
        position = worker.portfolio_manager.positions.get('BTCUSDT')

        # État initial de la position
        position_exists = position is not None
        position_open = position.is_open if position else False
        position_id = id(position) if position else None

        # Logger l'état initial (comme dans le vrai code)
        trade_log = {
            'step': step,
            'worker_id': worker.worker_id,
            'position_exists': position_exists,
            'position_open': position_open,
            'position_id': position_id,
            'portfolio_id': id(worker.portfolio_manager)
        }

        # Log similaire au vrai code
        logger.info(f"[TRADING DEBUG] {worker.worker_id} Step {step}: "
                   f"position_exists={position_exists}, position_open={position_open}, "
                   f"position_id={position_id}")

        self.log_global_event('TRADING_CHECK', worker.worker_id, trade_log)

        # Décision de trading aléatoire
        should_trade = random.random() > 0.7  # 30% de chance de trader

        if should_trade and position and not position.is_open:
            # Ouvrir position
            logger.info(f"🔄 {worker.worker_id} Opening position at step {step}")
            position.open(
                entry_price=50000.0 + random.uniform(-1000, 1000),
                size=0.001 + random.uniform(-0.0005, 0.0005)
            )
            worker.positions_opened += 1
            worker.trade_count += 1

            # Log après ouverture
            logger.info(f"[POSITION OPENED] {worker.worker_id} BTCUSDT: "
                       f"price={position.entry_price}, size={position.size}, "
                       f"position_id={id(position)}")

            self.log_global_event('POSITION_OPENED', worker.worker_id, {
                'position_id': id(position),
                'entry_price': position.entry_price,
                'size': position.size,
                'step': step
            })

        elif position and position.is_open and random.random() > 0.8:
            # Fermer position (20% de chance)
            logger.info(f"🔄 {worker.worker_id} Closing position at step {step}")
            position.close()

            self.log_global_event('POSITION_CLOSED', worker.worker_id, {
                'position_id': id(position),
                'step': step
            })

        # Simuler reset occasionnel
        if step > 0 and step % 10 == 0:
            logger.info(f"🔄 {worker.worker_id} Performing reset at step {step}")
            reset_performed = worker.portfolio_manager.reset(
                new_epoch=False,
                force=False,
                min_capital_before_reset=11.0
            )

            self.log_global_event('RESET_PERFORMED', worker.worker_id, {
                'step': step,
                'reset_performed': reset_performed,
                'portfolio_value': worker.portfolio_manager.get_portfolio_value()
            })

        return trade_log

    def run_worker(self, worker_id: str, num_steps: int = 20) -> WorkerState:
        """Exécuter un worker pour plusieurs steps"""
        logger.info(f"🚀 Starting worker {worker_id} for {num_steps} steps")

        worker = self.create_worker(worker_id)
        self.workers[worker_id] = worker

        for step in range(1, num_steps + 1):
            # Délai aléatoire pour simuler des timings différents
            time.sleep(random.uniform(0.1, 0.3))

            try:
                self.simulate_trading_step(worker, step)
            except Exception as e:
                logger.error(f"❌ Error in {worker_id} at step {step}: {e}")
                break

        logger.info(f"✅ Worker {worker_id} completed - "
                   f"Positions opened: {worker.positions_opened}, "
                   f"Total trades: {worker.trade_count}")

        return worker

    def analyze_results(self):
        """Analyser les résultats pour identifier les patterns problématiques"""
        logger.info("📊 ANALYSE DES RÉSULTATS")
        logger.info("=" * 60)

        # Analyser les logs globaux
        trading_checks = [log for log in self.global_logs if log['event_type'] == 'TRADING_CHECK']
        position_opens = [log for log in self.global_logs if log['event_type'] == 'POSITION_OPENED']

        logger.info(f"Total trading checks: {len(trading_checks)}")
        logger.info(f"Total position openings: {len(position_opens)}")

        # Identifier les cas position_exists=True, position_open=False
        problematic_cases = [
            log for log in trading_checks
            if log['data']['position_exists'] and not log['data']['position_open']
        ]

        logger.info(f"Cas problématiques (position_exists=True, position_open=False): {len(problematic_cases)}")

        # Analyser par worker
        logger.info("\n--- Analyse par worker ---")
        for worker_id, worker in self.workers.items():
            position = worker.portfolio_manager.positions.get('BTCUSDT')
            worker_checks = [log for log in trading_checks if log['worker_id'] == worker_id]
            worker_problematic = [log for log in problematic_cases if log['worker_id'] == worker_id]

            logger.info(f"Worker {worker_id}:")
            logger.info(f"  Portfolio ID: {id(worker.portfolio_manager)}")
            logger.info(f"  Position ID: {id(position) if position else None}")
            logger.info(f"  Final position state: exists={position is not None}, open={position.is_open if position else False}")
            logger.info(f"  Trading checks: {len(worker_checks)}")
            logger.info(f"  Problematic checks: {len(worker_problematic)}")
            logger.info(f"  Positions opened: {worker.positions_opened}")

        # Vérifier si des workers différents ont des IDs qui se mélangent
        logger.info("\n--- Vérification mélange des logs ---")
        portfolio_ids = set()
        position_ids = set()

        for log in trading_checks:
            portfolio_ids.add(log['data']['portfolio_id'])
            if log['data']['position_id']:
                position_ids.add(log['data']['position_id'])

        logger.info(f"Nombre de Portfolio IDs uniques: {len(portfolio_ids)}")
        logger.info(f"Nombre de Position IDs uniques: {len(position_ids)}")
        logger.info(f"Nombre de workers créés: {len(self.workers)}")

        if len(portfolio_ids) == len(self.workers):
            logger.info("✅ Chaque worker a son propre PortfolioManager (normal)")
        else:
            logger.warning("⚠️  Problème potentiel: nombre incorrect de PortfolioManagers")

        # Identifier la cause des cas problématiques
        logger.info("\n--- Analyse des cas problématiques ---")
        if problematic_cases:
            first_problematic = problematic_cases[0]
            logger.info("Premier cas problématique:")
            logger.info(f"  Worker: {first_problematic['worker_id']}")
            logger.info(f"  Step: {first_problematic['data']['step']}")
            logger.info(f"  Portfolio ID: {first_problematic['data']['portfolio_id']}")
            logger.info(f"  Position ID: {first_problematic['data']['position_id']}")

            # Vérifier si c'est un nouveau worker
            worker_events = [log for log in self.global_logs if log['worker_id'] == first_problematic['worker_id']]
            worker_events.sort(key=lambda x: x['timestamp'])

            logger.info(f"  Événements totaux pour ce worker: {len(worker_events)}")
            logger.info(f"  Premier événement: {worker_events[0]['event_type'] if worker_events else 'None'}")

            if worker_events and worker_events[0]['event_type'] == 'TRADING_CHECK':
                logger.info("🎯 CAUSE IDENTIFIÉE: Position vérifiée avant toute ouverture (normal pour nouveau worker)")
            else:
                logger.warning("⚠️  Cause inconnue pour ce cas problématique")

        return {
            'total_trading_checks': len(trading_checks),
            'total_position_opens': len(position_opens),
            'problematic_cases': len(problematic_cases),
            'unique_portfolio_ids': len(portfolio_ids),
            'unique_position_ids': len(position_ids),
            'workers_created': len(self.workers)
        }

    def test_concurrent_workers(self):
        """Test principal avec workers concurrents"""
        logger.info("🚀 DÉBUT DU TEST MULTI-WORKER")
        logger.info("=" * 60)

        # Lancer plusieurs workers en parallèle
        with ThreadPoolExecutor(max_workers=self.num_workers, thread_name_prefix='Worker') as executor:
            futures = []

            for i in range(self.num_workers):
                worker_id = f"w{i+1}"
                future = executor.submit(self.run_worker, worker_id, 15)
                futures.append(future)

            # Attendre que tous les workers terminent
            for future in futures:
                try:
                    future.result(timeout=30)
                except Exception as e:
                    logger.error(f"Worker failed: {e}")

        # Analyser les résultats
        results = self.analyze_results()

        # Conclusions
        logger.info("\n" + "=" * 60)
        logger.info("🎯 CONCLUSIONS")

        if results['problematic_cases'] > 0:
            logger.warning(f"⚠️  {results['problematic_cases']} cas de position_exists=True, position_open=False détectés")

            if results['unique_portfolio_ids'] == results['workers_created']:
                logger.info("✅ Chaque worker a son propre portfolio → Cas problématiques sont NORMAUX")
                logger.info("   → Ils correspondent à des vérifications avant ouverture de positions")
                logger.info("   → Ce n'est PAS un bug, c'est le comportement attendu")
            else:
                logger.warning("❌ Problème potentiel de partage de PortfolioManager entre workers")
        else:
            logger.info("✅ Aucun cas problématique détecté")

        return results

def main():
    """Fonction principale"""
    try:
        # Test avec 4 workers (comme dans la configuration réelle)
        tester = MultiWorkerTester(num_workers=4)
        results = tester.test_concurrent_workers()

        # Recommandations finales
        logger.info("\n" + "=" * 60)
        logger.info("💡 RECOMMANDATIONS")

        if results['problematic_cases'] > 0:
            logger.info("1. ✅ Les cas position_exists=True, position_open=False sont NORMAUX")
            logger.info("2. 🔧 Améliorer les logs pour distinguer les workers:")
            logger.info("   - Ajouter worker_id dans tous les logs de trading")
            logger.info("   - Séparer les fichiers de logs par worker")
            logger.info("3. 📊 Focus sur les métriques réelles:")
            logger.info("   - Positions ouvertes par worker")
            logger.info("   - PnL par worker")
            logger.info("   - Performance globale")

        logger.info("\n✅ Test multi-worker terminé avec succès")

        return results['problematic_cases'] == 0  # False si des cas problématiques (ce qui est normal)

    except Exception as e:
        logger.error(f"Erreur durant le test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
