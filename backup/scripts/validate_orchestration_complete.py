#!/usr/bin/env python3
"""
Script de validation finale pour la tâche 8.3.1 - Valider orchestration complète.
Vérifie que toutes les sous-tâches sont implémentées et fonctionnelles.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, '/home/morningstar/Documents/trading/ADAN/src')

def check_file_exists(filepath, description):
    """Vérifie qu'un fichier existe"""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} - MANQUANT")
        return False

def check_test_passes(test_script, description):
    """Vérifie qu'un test passe"""
    try:
        result = os.system(f"python {test_script} > /dev/null 2>&1")
        if result == 0:
            print(f"✅ {description}: RÉUSSI")
            return True
        else:
            print(f"❌ {description}: ÉCHEC")
            return False
    except Exception as e:
        print(f"❌ {description}: ERREUR - {e}")
        return False

def check_benchmark_results():
    """Vérifie que les résultats de benchmark existent"""
    logs_dir = Path("logs")

    # Cherche les fichiers de benchmark récents
    orchestration_files = list(logs_dir.glob("orchestration_benchmark_*.json"))
    parallel_files = list(logs_dir.glob("parallel_orchestration_benchmark_*.json"))

    has_orchestration = len(orchestration_files) > 0
    has_parallel = len(parallel_files) > 0

    if has_orchestration:
        latest_orch = max(orchestration_files, key=lambda f: f.stat().st_mtime)
        print(f"✅ Benchmark orchestration: {latest_orch.name}")
    else:
        print("❌ Benchmark orchestration: MANQUANT")

    if has_parallel:
        latest_parallel = max(parallel_files, key=lambda f: f.stat().st_mtime)
        print(f"✅ Benchmark parallèle: {latest_parallel.name}")
    else:
        print("❌ Benchmark parallèle: MANQUANT")

    return has_orchestration and has_parallel

def validate_orchestration_implementation():
    """Valide l'implémentation complète de l'orchestration"""
    print("🚀 Validation Orchestration Multi-Environnements")
    print("=" * 60)

    validation_results = {
        'timestamp': datetime.now().isoformat(),
        'task': '8.3.1 - Valider orchestration complète',
        'subtasks': {},
        'overall_status': 'PENDING'
    }

    # 1. Vérifier les fichiers d'implémentation
    print("\n📁 1. FICHIERS D'IMPLÉMENTATION")
    print("-" * 40)

    impl_files = [
        ("ADAN/src/adan_trading_bot/training/training_orchestrator.py", "TrainingOrchestrator"),
        ("ADAN/src/adan_trading_bot/online_learning_agent.py", "OnlineLearningAgent avec buffer"),
        ("ADAN/src/adan_trading_bot/environment/dynamic_behavior_engine.py", "DBE avec persistance")
    ]

    impl_status = all(check_file_exists(f, desc) for f, desc in impl_files)
    validation_results['subtasks']['implementation_files'] = impl_status

    # 2. Vérifier les tests unitaires
    print("\n🧪 2. TESTS UNITAIRES")
    print("-" * 40)

    test_files = [
        ("scripts/test_full_orchestration.py", "Test orchestration complète"),
        ("scripts/test_dbe_state_persistence.py", "Test persistance DBE"),
        ("scripts/test_continuous_experience_buffer.py", "Test buffer continu")
    ]

    test_status = all(check_test_passes(f, desc) for f, desc in test_files)
    validation_results['subtasks']['unit_tests'] = test_status

    # 3. Vérifier la stabilité 4 environnements
    print("\n🔄 3. STABILITÉ 4 ENVIRONNEMENTS")
    print("-" * 40)

    stability_status = check_test_passes(
        "scripts/test_full_orchestration.py",
        "Stabilité 4 environnements simultanés"
    )
    validation_results['subtasks']['stability_4_envs'] = stability_status

    # 4. Vérifier les mesures de performance
    print("\n📊 4. MESURES DE PERFORMANCE")
    print("-" * 40)

    benchmark_status = check_benchmark_results()
    validation_results['subtasks']['performance_benchmarks'] = benchmark_status

    # 5. Vérifier l'utilisation des ressources
    print("\n💾 5. UTILISATION DES RESSOURCES")
    print("-" * 40)

    resource_files = [
        ("logs/orchestration_benchmark_*.json", "Métriques ressources orchestration"),
        ("logs/parallel_orchestration_benchmark_*.json", "Métriques ressources parallèles")
    ]

    resource_status = True
    for pattern, desc in resource_files:
        files = list(Path(".").glob(pattern))
        if files:
            print(f"✅ {desc}: {len(files)} fichier(s)")
        else:
            print(f"❌ {desc}: MANQUANT")
            resource_status = False

    validation_results['subtasks']['resource_validation'] = resource_status

    # 6. Vérifier la documentation des gains
    print("\n📋 6. DOCUMENTATION DES GAINS")
    print("-" * 40)

    doc_status = check_file_exists(
        "ADAN/docs/orchestration_performance_report.md",
        "Rapport de performance"
    )
    validation_results['subtasks']['performance_documentation'] = doc_status

    # 7. Résumé final
    print("\n" + "=" * 60)
    print("📋 RÉSUMÉ DE VALIDATION")
    print("=" * 60)

    all_subtasks = [
        ("Fichiers d'implémentation", impl_status),
        ("Tests unitaires", test_status),
        ("Stabilité 4 environnements", stability_status),
        ("Benchmarks de performance", benchmark_status),
        ("Validation des ressources", resource_status),
        ("Documentation des gains", doc_status)
    ]

    passed_count = sum(1 for _, status in all_subtasks if status)
    total_count = len(all_subtasks)

    for desc, status in all_subtasks:
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {desc}")

    overall_success = passed_count == total_count
    validation_results['subtasks']['summary'] = {
        'passed': passed_count,
        'total': total_count,
        'success_rate': passed_count / total_count
    }
    validation_results['overall_status'] = 'SUCCESS' if overall_success else 'PARTIAL'

    print(f"\n🎯 Score: {passed_count}/{total_count} sous-tâches validées")

    if overall_success:
        print("🎉 TÂCHE 8.3.1 COMPLÈTEMENT VALIDÉE !")
        print("✅ Prêt pour le Sprint 9 - Optimisation Performance")
    else:
        print("⚠️  VALIDATION PARTIELLE - Quelques éléments à compléter")

    # Sauvegarde du rapport de validation
    validation_file = f"logs/orchestration_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("logs", exist_ok=True)

    with open(validation_file, 'w') as f:
        json.dump(validation_results, f, indent=2)

    print(f"\n📁 Rapport de validation sauvegardé: {validation_file}")

    return overall_success, validation_results

def main():
    """Fonction principale"""
    success, results = validate_orchestration_implementation()

    if success:
        print("\n🚀 VALIDATION RÉUSSIE - Sprint 8.3.1 COMPLET")
        return 0
    else:
        print("\n⚠️  VALIDATION PARTIELLE - Vérifier les éléments manquants")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
