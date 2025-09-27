#!/usr/bin/env python3
"""
Test de validation des points critiques - ADAN Trading Bot
Validation rapide des éléments essentiels pour la stabilité.
"""

import os
import sys
import subprocess
import time
import logging
import re
from pathlib import Path
from typing import Dict, List, Any

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CriticalValidator:
    """Validateur pour les points critiques de stabilité."""

    def __init__(self, bot_dir: str = "bot"):
        self.bot_dir = Path(bot_dir)
        self.config_path = self.bot_dir / "config" / "config.yaml"
        self.script_path = self.bot_dir / "scripts" / "train_parallel_agents.py"

    def run_stability_test(self, timeout_seconds: int = 25) -> Dict[str, Any]:
        """Lance un test de stabilité de 25 secondes."""
        logger.info("🚀 LANCEMENT TEST STABILITÉ (25s)")

        cmd = [
            "eval", "$(conda shell.bash hook)", "&&",
            "conda", "activate", "trading_env", "&&",
            "timeout", f"{timeout_seconds}s",
            "python", str(self.script_path),
            "--config", str(self.config_path),
            "--timeout", "3600"
        ]

        cmd_str = " ".join(cmd)
        logger.info(f"Commande : {cmd_str}")

        try:
            # Utiliser shell=True pour les commandes complexes avec eval
            process = subprocess.run(
                cmd_str,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout_seconds + 5  # Marge de sécurité
            )

            output_lines = process.stdout.split('\n') if process.stdout else []
            error_lines = process.stderr.split('\n') if process.stderr else []

            return {
                "success": True,
                "return_code": process.returncode,
                "output_lines": output_lines,
                "error_lines": error_lines,
                "timeout_reached": process.returncode == 124  # Code retour timeout
            }

        except subprocess.TimeoutExpired:
            logger.info("⏰ Timeout atteint (normal)")
            return {"success": False, "error": "timeout", "timeout_reached": True}
        except Exception as e:
            logger.error(f"Erreur exécution : {e}")
            return {"success": False, "error": str(e), "timeout_reached": False}

    def analyze_critical_points(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse les points critiques dans les logs."""
        analysis = {
            "startup_success": False,
            "config_loaded": False,
            "data_loaded": False,
            "environment_created": False,
            "risk_params_initialized": False,
            "no_crashes": True,
            "capital_management": {"found": False, "source": None, "amount": None},
            "position_logs": {"correct_entry_prices": True, "issues": []},
            "inappropriate_resets": {"found": False, "count": 0, "details": []},
            "worker_system": {"functioning": False, "workers_detected": 0},
            "overall_score": 0.0
        }

        if not result.get("success") and not result.get("timeout_reached"):
            analysis["no_crashes"] = False
            return analysis

        all_lines = result.get("output_lines", []) + result.get("error_lines", [])

        # Analyser ligne par ligne
        for line in all_lines:
            line = line.strip()

            # Points de démarrage
            if "ADAN Trading Bot v" in line:
                analysis["startup_success"] = True

            if "Configuration chargée depuis" in line:
                analysis["config_loaded"] = True

            if "Données chargées:" in line and ("lignes" in line):
                analysis["data_loaded"] = True

            if "MultiAssetChunkedEnv instance created" in line:
                analysis["environment_created"] = True

            if "Paramètres de risque initialisés" in line:
                analysis["risk_params_initialized"] = True

            # Capital management
            init_capital_match = re.search(r'\[INIT CAPITAL\] Source: (\w+), initial_equity: \$(\d+\.?\d*)', line)
            if init_capital_match:
                analysis["capital_management"]["found"] = True
                analysis["capital_management"]["source"] = init_capital_match.group(1)
                analysis["capital_management"]["amount"] = float(init_capital_match.group(2))

            # Position logs
            if "Position fermée" in line:
                entry_match = re.search(r'entrée: \$(\d+\.?\d*)', line)
                if entry_match:
                    entry_price = float(entry_match.group(1))
                    if entry_price == 0.0:
                        analysis["position_logs"]["correct_entry_prices"] = False
                        analysis["position_logs"]["issues"].append(f"Entry price 0.0: {line}")

            # Resets inappropriés
            if "_epoch_reset" in line and "step" in line.lower() and "reset()" not in line:
                analysis["inappropriate_resets"]["found"] = True
                analysis["inappropriate_resets"]["count"] += 1
                analysis["inappropriate_resets"]["details"].append(line)

            # Worker system
            if "[w" in line and "]" in line:
                analysis["worker_system"]["functioning"] = True
                # Compter workers uniques
                worker_match = re.search(r'\[w(\d+)\]', line)
                if worker_match:
                    worker_id = int(worker_match.group(1))
                    analysis["worker_system"]["workers_detected"] = max(
                        analysis["worker_system"]["workers_detected"],
                        worker_id + 1
                    )

            # Détection de crashes
            if "Traceback" in line or "Exception" in line or "Error:" in line:
                # Filtrer les erreurs connues non critiques
                if not any(ignore in line for ignore in ["import-error", "ModuleNotFoundError: gymnasium", "timeout"]):
                    analysis["no_crashes"] = False

        # Calculer score global
        score = 0.0
        weights = {
            "startup_success": 10,
            "config_loaded": 15,
            "data_loaded": 15,
            "environment_created": 10,
            "risk_params_initialized": 10,
            "no_crashes": 20,
            "capital_management": 10,
            "position_logs": 5,
            "inappropriate_resets": -15,  # Pénalité
            "worker_system": 5
        }

        if analysis["startup_success"]: score += weights["startup_success"]
        if analysis["config_loaded"]: score += weights["config_loaded"]
        if analysis["data_loaded"]: score += weights["data_loaded"]
        if analysis["environment_created"]: score += weights["environment_created"]
        if analysis["risk_params_initialized"]: score += weights["risk_params_initialized"]
        if analysis["no_crashes"]: score += weights["no_crashes"]
        if analysis["capital_management"]["found"]: score += weights["capital_management"]
        if analysis["position_logs"]["correct_entry_prices"]: score += weights["position_logs"]
        if analysis["inappropriate_resets"]["found"]: score += weights["inappropriate_resets"]
        if analysis["worker_system"]["functioning"]: score += weights["worker_system"]

        analysis["overall_score"] = max(0.0, score)

        return analysis

    def generate_report(self, analysis: Dict[str, Any]) -> str:
        """Génère un rapport de validation."""
        score = analysis["overall_score"]

        if score >= 90:
            status = "🟢 EXCELLENT"
        elif score >= 75:
            status = "🟡 GOOD"
        elif score >= 50:
            status = "🟠 MODERATE"
        else:
            status = "🔴 POOR"

        report = f"""
╭─────────────────────────────────────────────╮
│          RAPPORT VALIDATION CRITIQUE        │
╰─────────────────────────────────────────────╯

📊 SCORE GLOBAL : {score:.1f}/100 {status}

🚀 POINTS DE DÉMARRAGE :
  ✅ Startup Success        : {'✓' if analysis['startup_success'] else '✗'}
  ✅ Configuration Loaded   : {'✓' if analysis['config_loaded'] else '✗'}
  ✅ Data Loaded           : {'✓' if analysis['data_loaded'] else '✗'}
  ✅ Environment Created   : {'✓' if analysis['environment_created'] else '✗'}
  ✅ Risk Params Init      : {'✓' if analysis['risk_params_initialized'] else '✗'}

💰 CAPITAL MANAGEMENT :
  Source trouvée : {'✓' if analysis['capital_management']['found'] else '✗'}"""

        if analysis['capital_management']['found']:
            report += f"""
  Source : {analysis['capital_management']['source']}
  Montant : ${analysis['capital_management']['amount']:.2f}"""

        report += f"""

🔄 RESETS ET POSITIONS :
  Pas de resets inappropriés : {'✓' if not analysis['inappropriate_resets']['found'] else '✗'}"""

        if analysis['inappropriate_resets']['found']:
            report += f"""
  ⚠️  Resets détectés : {analysis['inappropriate_resets']['count']}"""

        report += f"""
  Entry prices corrects     : {'✓' if analysis['position_logs']['correct_entry_prices'] else '✗'}"""

        if analysis['position_logs']['issues']:
            report += f"""
  ⚠️  Issues positions : {len(analysis['position_logs']['issues'])}"""

        report += f"""

👷 WORKERS SYSTEM :
  Système fonctionnel : {'✓' if analysis['worker_system']['functioning'] else '✗'}
  Workers détectés    : {analysis['worker_system']['workers_detected']}

🛡️  STABILITÉ :
  Aucun crash critique : {'✓' if analysis['no_crashes'] else '✗'}
"""

        return report

    def run_full_validation(self) -> bool:
        """Lance la validation complète."""
        logger.info("🎯 VALIDATION POINTS CRITIQUES - ADAN Trading Bot")

        # Vérifications préliminaires
        if not self.config_path.exists():
            logger.error(f"❌ Configuration non trouvée : {self.config_path}")
            return False

        if not self.script_path.exists():
            logger.error(f"❌ Script non trouvé : {self.script_path}")
            return False

        logger.info("✅ Fichiers de base présents")

        # Test de stabilité
        result = self.run_stability_test(timeout_seconds=25)

        # Analyse
        analysis = self.analyze_critical_points(result)

        # Rapport
        report = self.generate_report(analysis)
        print(report)

        # Sauvegarder le rapport
        report_path = Path("analysis") / "critical_validation_report.txt"
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"📄 Rapport sauvegardé : {report_path}")

        # Critères de succès
        success_criteria = [
            analysis["startup_success"],
            analysis["config_loaded"],
            analysis["data_loaded"],
            analysis["environment_created"],
            analysis["no_crashes"],
            not analysis["inappropriate_resets"]["found"],
            analysis["overall_score"] >= 75
        ]

        success = all(success_criteria)

        if success:
            logger.info("🎉 VALIDATION CRITIQUE : SUCCÈS")
            logger.info("🚀 Système prêt pour entraînement long")
        else:
            logger.error("❌ VALIDATION CRITIQUE : ÉCHEC")
            logger.error("⚠️  Corrections nécessaires avant entraînement long")

        return success


def main():
    """Point d'entrée principal."""
    validator = CriticalValidator()
    success = validator.run_full_validation()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
