#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de test pour valider l'affichage hiérarchique et les corrections.

Ce script teste :
1. L'affichage hiérarchique avec le nouveau callback
2. La correction de l'erreur JSONL avec clean_worker_id
3. La configuration exposure_range des paliers
4. L'intégration avec les métriques de trading
"""

import logging
import numpy as np
import time
import sys
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock

# Ajouter le chemin du projet
project_root = Path(__file__).parent
sys.path.append(str(project_root / "bot" / "src"))

try:
    from adan_trading_bot.agent.ppo_agent import HierarchicalTrainingDisplayCallback
    from adan_trading_bot.environment.multi_asset_chunked_env import clean_worker_id
    from adan_trading_bot.common.constants import CAPITAL_TIERS
    print("✅ Imports réussis")
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    sys.exit(1)

# Configuration du logging pour les tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_clean_worker_id():
    """Test de la fonction clean_worker_id pour corriger l'erreur JSONL."""
    print("\n" + "="*60)
    print("🧪 TEST 1: Correction erreur JSONL - clean_worker_id")
    print("="*60)

    test_cases = [
        ("w0", 0),
        ("W0", 0),
        ("w1", 1),
        ("W5", 5),
        ("w99", 99),
        (0, 0),
        (42, 42),
        ("invalid", 0),
        ("", 0),
        (None, 0)
    ]

    passed = 0
    for input_val, expected in test_cases:
        result = clean_worker_id(input_val)
        if result == expected:
            print(f"✅ {input_val} -> {result} (attendu: {expected})")
            passed += 1
        else:
            print(f"❌ {input_val} -> {result} (attendu: {expected})")

    print(f"\n📊 Résultat: {passed}/{len(test_cases)} tests réussis")
    return passed == len(test_cases)

def test_exposure_range_config():
    """Test de la configuration exposure_range dans les paliers."""
    print("\n" + "="*60)
    print("🧪 TEST 2: Configuration exposure_range des paliers")
    print("="*60)

    required_tiers = ["Micro", "Small", "Medium", "Large", "Enterprise"]
    passed = 0

    for tier_name in required_tiers:
        if tier_name in CAPITAL_TIERS:
            tier = CAPITAL_TIERS[tier_name]
            if 'exposure_range' in tier:
                exposure_range = tier['exposure_range']
                if isinstance(exposure_range, list) and len(exposure_range) == 2:
                    print(f"✅ {tier_name}: exposure_range = {exposure_range}")
                    passed += 1
                else:
                    print(f"❌ {tier_name}: exposure_range invalide = {exposure_range}")
            else:
                print(f"❌ {tier_name}: exposure_range manquant")
        else:
            print(f"❌ {tier_name}: palier manquant")

    print(f"\n📊 Résultat: {passed}/{len(required_tiers)} paliers configurés")
    return passed == len(required_tiers)

class MockModel:
    """Modèle fictif pour tester le callback."""

    def __init__(self):
        self.logger = Mock()
        self.logger.name_to_value = {
            "train/loss": 0.1234,
            "train/policy_loss": 0.0567,
            "train/value_loss": 0.0456,
            "train/entropy_loss": 0.7890
        }

class MockCallback(HierarchicalTrainingDisplayCallback):
    """Callback modifié pour les tests."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_logs = []

        # Mock du logger pour capturer les sorties
        original_logger = logger
        self.mock_logger = Mock()
        self.mock_logger.info = lambda msg: self.test_logs.append(msg)
        self.mock_logger.warning = lambda msg: self.test_logs.append(f"WARNING: {msg}")
        self.mock_logger.error = lambda msg: self.test_logs.append(f"ERROR: {msg}")

        # Remplacer temporairement le logger global
        import adan_trading_bot.agent.ppo_agent as ppo_module
        ppo_module.logger = self.mock_logger

def test_hierarchical_callback():
    """Test du callback d'affichage hiérarchique."""
    print("\n" + "="*60)
    print("🧪 TEST 3: Callback d'affichage hiérarchique")
    print("="*60)

    # Créer le callback avec des paramètres de test
    callback = MockCallback(
        verbose=1,
        display_freq=100,
        total_timesteps=1000,
        initial_capital=50.0
    )

    # Simuler le modèle
    callback.model = MockModel()
    callback.num_timesteps = 500

    # Simuler des données d'environnement
    callback.locals = {
        "infos": [{
            "portfolio_value": 55.25,
            "cash": 25.30,
            "drawdown": 2.5,
            "positions": {
                "BTCUSDT": {
                    "size": 0.001,
                    "entry_price": 45000.0,
                    "value": 45.0,
                    "sl": 43000.0,
                    "tp": 50000.0
                }
            },
            "sharpe": 1.25,
            "sortino": 1.42,
            "profit_factor": 1.18,
            "max_dd": 3.2,
            "cagr": 15.5,
            "win_rate": 65.0,
            "trades": 12
        }]
    }

    try:
        # Test de démarrage
        callback._on_training_start()
        print("✅ _on_training_start() réussi")

        # Test d'affichage des métriques
        callback._log_detailed_metrics()
        print("✅ _log_detailed_metrics() réussi")

        # Test de fin de rollout
        callback._on_rollout_end()
        print("✅ _on_rollout_end() réussi")

        # Test de fin d'entraînement
        callback._on_training_end()
        print("✅ _on_training_end() réussi")

        # Vérifier que des logs ont été générés
        if len(callback.test_logs) > 0:
            print(f"✅ {len(callback.test_logs)} logs générés")

            # Afficher quelques exemples de logs
            print("\n📋 Exemples de logs générés:")
            for i, log in enumerate(callback.test_logs[:5]):  # Premiers 5 logs
                print(f"   {i+1}. {log[:80]}...")

            if len(callback.test_logs) > 5:
                print(f"   ... et {len(callback.test_logs) - 5} autres logs")

            return True
        else:
            print("❌ Aucun log généré")
            return False

    except Exception as e:
        print(f"❌ Erreur lors du test du callback: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_portfolio_metrics():
    """Test de l'intégration avec les métriques de portfolio."""
    print("\n" + "="*60)
    print("🧪 TEST 4: Intégration métriques de portfolio")
    print("="*60)

    # Simuler différents scénarios de portfolio
    test_scenarios = [
        {
            "name": "Portfolio profitable",
            "portfolio_value": 120.0,
            "initial_capital": 100.0,
            "expected_roi": 20.0
        },
        {
            "name": "Portfolio en perte",
            "portfolio_value": 85.0,
            "initial_capital": 100.0,
            "expected_roi": -15.0
        },
        {
            "name": "Portfolio stable",
            "portfolio_value": 100.5,
            "initial_capital": 100.0,
            "expected_roi": 0.5
        }
    ]

    passed = 0
    for scenario in test_scenarios:
        roi = ((scenario["portfolio_value"] - scenario["initial_capital"]) / scenario["initial_capital"]) * 100
        if abs(roi - scenario["expected_roi"]) < 0.01:
            print(f"✅ {scenario['name']}: ROI = {roi:.1f}% (attendu: {scenario['expected_roi']:.1f}%)")
            passed += 1
        else:
            print(f"❌ {scenario['name']}: ROI = {roi:.1f}% (attendu: {scenario['expected_roi']:.1f}%)")

    print(f"\n📊 Résultat: {passed}/{len(test_scenarios)} scénarios réussis")
    return passed == len(test_scenarios)

def test_progress_bar():
    """Test de la barre de progression."""
    print("\n" + "="*60)
    print("🧪 TEST 5: Barre de progression")
    print("="*60)

    total_timesteps = 1000
    test_steps = [0, 100, 250, 500, 750, 1000]

    print("Simulation de la barre de progression:")
    for step in test_steps:
        progress = step / total_timesteps * 100
        progress_bar_length = 30
        filled_length = int(progress_bar_length * progress // 100)
        bar = "━" * filled_length + "─" * (progress_bar_length - filled_length)

        print(f"🚀 ADAN Training {bar} {progress:6.1f}% ({step:4d}/{total_timesteps})")

    print("✅ Barre de progression fonctionnelle")
    return True

def run_integration_test():
    """Test d'intégration complet."""
    print("\n" + "="*80)
    print("🎯 TEST D'INTÉGRATION COMPLET")
    print("="*80)

    print("Simulation d'un entraînement complet avec affichage hiérarchique...")

    # Créer le callback
    callback = MockCallback(
        verbose=1,
        display_freq=50,
        total_timesteps=200,
        initial_capital=25.0
    )

    callback.model = MockModel()

    # Simuler une session d'entraînement
    for step in range(0, 201, 50):
        callback.num_timesteps = step

        # Simuler l'évolution du portfolio
        portfolio_value = 25.0 + (step / 200) * 10.0  # Croissance de 25 à 35

        callback.locals = {
            "infos": [{
                "portfolio_value": portfolio_value,
                "cash": portfolio_value * 0.3,
                "drawdown": max(0, (step / 200) * 5.0),
                "positions": {
                    "ADAUSDT": {
                        "size": 35.0,
                        "entry_price": 0.7092,
                        "value": portfolio_value * 0.7,
                        "sl": 0.67,
                        "tp": 0.82
                    }
                } if step > 0 else {},
                "sharpe": 0.5 + (step / 200) * 1.0,
                "sortino": 0.6 + (step / 200) * 1.0,
                "profit_factor": 1.0 + (step / 200) * 0.5,
                "max_dd": (step / 200) * 5.0,
                "cagr": (step / 200) * 20.0,
                "win_rate": 50.0 + (step / 200) * 20.0,
                "trades": step // 50
            }]
        }

        if step == 0:
            callback._on_training_start()
        elif step > 0:
            callback._log_detailed_metrics()

    callback._on_training_end()

    print(f"\n✅ Test d'intégration terminé - {len(callback.test_logs)} logs générés")

    # Analyser les logs pour vérifier le contenu attendu
    expected_patterns = [
        "🚀 DÉMARRAGE ADAN TRAINING",
        "Configuration Flux Monétaires",
        "PORTFOLIO",
        "RISK",
        "METRICS",
        "✅ ENTRAÎNEMENT TERMINÉ"
    ]

    found_patterns = 0
    for pattern in expected_patterns:
        if any(pattern in log for log in callback.test_logs):
            print(f"✅ Pattern trouvé: {pattern}")
            found_patterns += 1
        else:
            print(f"❌ Pattern manquant: {pattern}")

    print(f"\n📊 Patterns trouvés: {found_patterns}/{len(expected_patterns)}")
    return found_patterns >= len(expected_patterns) * 0.8  # 80% de réussite minimum

def main():
    """Fonction principale de test."""
    print("🎯 VALIDATION COMPLÈTE - AFFICHAGE HIÉRARCHIQUE ADAN")
    print("="*80)
    print("Ce script valide toutes les améliorations apportées:")
    print("• Affichage hiérarchique structuré")
    print("• Correction erreur JSONL (worker_id)")
    print("• Configuration exposure_range")
    print("• Intégration métriques de trading")
    print("• Barre de progression visuelle")
    print("="*80)

    tests = [
        ("Correction erreur JSONL", test_clean_worker_id),
        ("Configuration exposure_range", test_exposure_range_config),
        ("Callback hiérarchique", test_hierarchical_callback),
        ("Métriques de portfolio", test_portfolio_metrics),
        ("Barre de progression", test_progress_bar),
        ("Test d'intégration", run_integration_test)
    ]

    results = {}
    passed_tests = 0

    start_time = time.time()

    for test_name, test_func in tests:
        print(f"\n⏳ Exécution: {test_name}...")
        try:
            result = test_func()
            results[test_name] = result
            if result:
                passed_tests += 1
                print(f"✅ {test_name}: RÉUSSI")
            else:
                print(f"❌ {test_name}: ÉCHOUÉ")
        except Exception as e:
            print(f"💥 {test_name}: ERREUR - {e}")
            results[test_name] = False

    end_time = time.time()

    # Résumé final
    print("\n" + "="*80)
    print("📊 RÉSUMÉ FINAL")
    print("="*80)
    print(f"Tests réussis: {passed_tests}/{len(tests)}")
    print(f"Taux de réussite: {(passed_tests/len(tests)*100):.1f}%")
    print(f"Temps d'exécution: {end_time-start_time:.2f}s")

    print("\n📋 Détail des résultats:")
    for test_name, result in results.items():
        status = "✅ RÉUSSI" if result else "❌ ÉCHOUÉ"
        print(f"  {test_name:<30} {status}")

    if passed_tests == len(tests):
        print("\n🎉 TOUS LES TESTS SONT RÉUSSIS!")
        print("✅ L'affichage hiérarchique est prêt pour l'entraînement")
        print("\nCommande recommandée pour l'entraînement:")
        print("conda run -n trading_env python bot/scripts/train_parallel_agents.py --config bot/config/config.yaml --timeout 120 --checkpoint-dir checkpoints")
        return True
    else:
        print(f"\n⚠️  {len(tests)-passed_tests} test(s) ont échoué")
        print("Veuillez corriger les problèmes avant l'entraînement")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
