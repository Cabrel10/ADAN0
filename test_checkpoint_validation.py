#!/usr/bin/env python3
"""
Test de validation des checkpoints - ADAN Trading Bot
Ce script teste la fonctionnalité de sauvegarde/reprise de checkpoint
pour s'assurer qu'il n'y a pas de réinitialisations inappropriées.
"""

import os
import sys
import subprocess
import time
import signal
import tempfile
import shutil
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
import json

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CheckpointValidator:
    """Validateur pour les fonctionnalités de checkpoint."""

    def __init__(self, bot_dir: str = "bot"):
        self.bot_dir = Path(bot_dir)
        self.temp_dir = None
        self.original_config_path = self.bot_dir / "config" / "config.yaml"
        self.test_config_path = None
        self.checkpoint_dir = None

    def setup_test_environment(self) -> bool:
        """Configure l'environnement de test temporaire."""
        try:
            # Créer répertoire temporaire
            self.temp_dir = Path(tempfile.mkdtemp(prefix="adan_checkpoint_test_"))
            logger.info(f"Répertoire temporaire créé : {self.temp_dir}")

            # Créer répertoire checkpoints de test
            self.checkpoint_dir = self.temp_dir / "checkpoints"
            self.checkpoint_dir.mkdir(parents=True)

            # Copier et modifier la configuration
            with open(self.original_config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Modifier pour test rapide
            config['agent']['n_steps'] = 32  # Très petit pour test rapide
            config['agent']['batch_size'] = 16
            config['training']['max_timesteps'] = 64  # Minimum pour créer checkpoint
            config['agent']['checkpoint_freq'] = 32  # Checkpoint fréquent

            # Sauvegarder config de test
            self.test_config_path = self.temp_dir / "test_config.yaml"
            with open(self.test_config_path, 'w') as f:
                yaml.safe_dump(config, f)

            logger.info("Configuration de test préparée")
            return True

        except Exception as e:
            logger.error(f"Erreur setup environnement test : {e}")
            return False

    def cleanup_test_environment(self):
        """Nettoie l'environnement de test."""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            logger.info("Environnement temporaire nettoyé")

    def run_training_phase_1(self, timeout_seconds: int = 15) -> Dict[str, Any]:
        """Lance la première phase d'entraînement pour créer un checkpoint."""
        logger.info("=== PHASE 1 : Création du checkpoint initial ===")

        cmd = [
            "python", f"{self.bot_dir}/scripts/train_parallel_agents.py",
            "--config", str(self.test_config_path),
            "--models-dir", str(self.checkpoint_dir)
        ]

        logger.info(f"Commande : {' '.join(cmd)}")

        # Lancer le processus
        process = None
        output_lines = []
        error_lines = []

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            # Attendre timeout puis interrompre
            time.sleep(timeout_seconds)

            # Interruption propre
            if process.poll() is None:
                logger.info("Interruption propre du processus...")
                process.send_signal(signal.SIGTERM)
                try:
                    stdout, stderr = process.communicate(timeout=5)
                    output_lines = stdout.split('\n') if stdout else []
                    error_lines = stderr.split('\n') if stderr else []
                except subprocess.TimeoutExpired:
                    logger.warning("Timeout lors de l'interruption, force SIGKILL")
                    process.kill()
                    stdout, stderr = process.communicate()
                    output_lines = stdout.split('\n') if stdout else []
                    error_lines = stderr.split('\n') if stderr else []

            return {
                "success": True,
                "return_code": process.returncode,
                "output_lines": output_lines,
                "error_lines": error_lines,
                "interrupted": True
            }

        except Exception as e:
            logger.error(f"Erreur phase 1 : {e}")
            if process:
                process.kill()
            return {
                "success": False,
                "error": str(e),
                "output_lines": output_lines,
                "error_lines": error_lines
            }

    def check_checkpoint_created(self) -> bool:
        """Vérifie qu'un checkpoint a été créé."""
        checkpoint_files = list(self.checkpoint_dir.glob("**/*.zip"))
        logger.info(f"Checkpoints trouvés : {len(checkpoint_files)}")
        for cp in checkpoint_files:
            logger.info(f"  - {cp}")
        return len(checkpoint_files) > 0

    def run_training_phase_2(self, timeout_seconds: int = 10) -> Dict[str, Any]:
        """Lance la reprise depuis checkpoint."""
        logger.info("=== PHASE 2 : Reprise depuis checkpoint ===")

        cmd = [
            "python", f"{self.bot_dir}/scripts/train_parallel_agents.py",
            "--config", str(self.test_config_path),
            "--models-dir", str(self.checkpoint_dir),
            "--resume"
        ]

        logger.info(f"Commande reprise : {' '.join(cmd)}")

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            # Laisser tourner puis arrêter
            time.sleep(timeout_seconds)

            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
                try:
                    stdout, stderr = process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    stdout, stderr = process.communicate()
            else:
                stdout, stderr = process.communicate()

            return {
                "success": True,
                "return_code": process.returncode,
                "output_lines": stdout.split('\n') if stdout else [],
                "error_lines": stderr.split('\n') if stderr else []
            }

        except Exception as e:
            logger.error(f"Erreur phase 2 : {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def analyze_logs(self, phase1_result: Dict[str, Any], phase2_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse les logs pour détecter les problèmes."""
        analysis = {
            "phase1_issues": [],
            "phase2_issues": [],
            "capital_consistency": True,
            "no_inappropriate_resets": True,
            "checkpoint_loaded_correctly": False,
            "overall_health": "GOOD"
        }

        # Analyser Phase 1
        if phase1_result.get("success"):
            for line in phase1_result.get("output_lines", []):
                if "[INIT CAPITAL]" in line:
                    logger.info(f"Phase 1 - Capital initial : {line.strip()}")
                if "_epoch_reset" in line and "step" in line.lower():
                    analysis["phase1_issues"].append(f"Reset possible à chaud : {line.strip()}")
                if "Position fermée" in line and "entrée: $0.00" in line:
                    analysis["phase1_issues"].append(f"Entry price incorrect : {line.strip()}")

        # Analyser Phase 2
        if phase2_result.get("success"):
            resume_found = False
            for line in phase2_result.get("output_lines", []):
                if "Loading checkpoint" in line or "Resuming from" in line:
                    analysis["checkpoint_loaded_correctly"] = True
                    resume_found = True
                    logger.info(f"Phase 2 - Reprise détectée : {line.strip()}")
                if "[INIT CAPITAL]" in line:
                    logger.info(f"Phase 2 - Capital initial : {line.strip()}")
                if "_epoch_reset" in line and "step" in line.lower():
                    analysis["phase2_issues"].append(f"Reset inapproprié après resume : {line.strip()}")

            if not resume_found:
                analysis["phase2_issues"].append("Aucune indication de reprise de checkpoint détectée")

        # Déterminer santé globale
        total_issues = len(analysis["phase1_issues"]) + len(analysis["phase2_issues"])
        if total_issues == 0 and analysis["checkpoint_loaded_correctly"]:
            analysis["overall_health"] = "EXCELLENT"
        elif total_issues <= 1:
            analysis["overall_health"] = "GOOD"
        elif total_issues <= 3:
            analysis["overall_health"] = "MODERATE"
        else:
            analysis["overall_health"] = "POOR"

        return analysis

    def run_full_test(self) -> bool:
        """Lance le test complet de validation des checkpoints."""
        logger.info("🚀 DÉBUT TEST VALIDATION CHECKPOINTS")

        try:
            # Setup
            if not self.setup_test_environment():
                return False

            # Phase 1 : Créer checkpoint
            phase1_result = self.run_training_phase_1(timeout_seconds=20)
            if not phase1_result["success"]:
                logger.error("❌ ÉCHEC Phase 1")
                return False

            # Vérifier checkpoint créé
            if not self.check_checkpoint_created():
                logger.error("❌ ÉCHEC : Aucun checkpoint créé")
                return False

            logger.info("✅ Phase 1 réussie - Checkpoint créé")

            # Phase 2 : Reprendre depuis checkpoint
            time.sleep(2)  # Pause entre phases
            phase2_result = self.run_training_phase_2(timeout_seconds=15)
            if not phase2_result["success"]:
                logger.error("❌ ÉCHEC Phase 2")
                return False

            logger.info("✅ Phase 2 réussie - Reprise testée")

            # Analyse
            analysis = self.analyze_logs(phase1_result, phase2_result)

            # Rapport final
            logger.info("📊 RAPPORT FINAL :")
            logger.info(f"  Santé globale : {analysis['overall_health']}")
            logger.info(f"  Checkpoint chargé : {analysis['checkpoint_loaded_correctly']}")
            logger.info(f"  Issues Phase 1 : {len(analysis['phase1_issues'])}")
            logger.info(f"  Issues Phase 2 : {len(analysis['phase2_issues'])}")

            if analysis['phase1_issues']:
                logger.warning("⚠️  Issues Phase 1 :")
                for issue in analysis['phase1_issues']:
                    logger.warning(f"    - {issue}")

            if analysis['phase2_issues']:
                logger.warning("⚠️  Issues Phase 2 :")
                for issue in analysis['phase2_issues']:
                    logger.warning(f"    - {issue}")

            # Critères de succès
            success = (
                analysis['overall_health'] in ['EXCELLENT', 'GOOD'] and
                analysis['checkpoint_loaded_correctly'] and
                len(analysis['phase2_issues']) == 0
            )

            if success:
                logger.info("🎉 TEST CHECKPOINT : SUCCÈS")
            else:
                logger.error("❌ TEST CHECKPOINT : ÉCHEC")

            return success

        except Exception as e:
            logger.error(f"Erreur test complet : {e}")
            return False

        finally:
            self.cleanup_test_environment()


def main():
    """Point d'entrée principal."""
    logger.info("Test de validation des checkpoints - ADAN Trading Bot")

    validator = CheckpointValidator()
    success = validator.run_full_test()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
