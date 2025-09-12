"""
Orchestrateur de workflows pour ADAN Trading Bot.
Gère l'exécution des scripts CLI avec suivi temps réel.
Implémente les tâches 10B.1.1, 10B.1.2, 10B.1.3.
"""

import os
import sys
import subprocess
import threading
import time
import json
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import queue
import signal

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """États des workflows"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowStep:
    """Étape d'un workflow"""
    name: str
    script_path: str
    description: str
    args: List[str] = None
    env_vars: Dict[str, str] = None
    timeout: int = 3600  # 1 heure par défaut
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.args is None:
            self.args = []
        if self.env_vars is None:
            self.env_vars = {}
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class WorkflowExecution:
    """Exécution d'un workflow"""
    workflow_id: str
    step_name: str
    status: WorkflowStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    exit_code: Optional[int] = None
    output_lines: List[str] = None
    error_lines: List[str] = None
    progress_percent: float = 0.0
    
    def __post_init__(self):
        if self.output_lines is None:
            self.output_lines = []
        if self.error_lines is None:
            self.error_lines = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire pour sérialisation"""
        data = asdict(self)
        data['status'] = self.status.value
        if self.start_time:
            data['start_time'] = self.start_time.isoformat()
        if self.end_time:
            data['end_time'] = self.end_time.isoformat()
        return data


class WorkflowOrchestrator:
    """Orchestrateur principal des workflows"""
    
    def __init__(self, base_path: str = None):
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.scripts_path = self.base_path / "scripts"
        
        # État des exécutions
        self.executions: Dict[str, WorkflowExecution] = {}
        self.active_processes: Dict[str, subprocess.Popen] = {}
        self.execution_threads: Dict[str, threading.Thread] = {}
        
        # Callbacks pour UI
        self.progress_callbacks: List[Callable] = []
        self.completion_callbacks: List[Callable] = []
        self.log_callbacks: List[Callable] = []
        
        # Configuration des workflows prédéfinis
        self.workflows = self._define_workflows()
        
        # Queue pour communication thread-safe
        self.message_queue = queue.Queue()
        
        logger.info(f"WorkflowOrchestrator initialized with base_path: {self.base_path}")
    
    def _define_workflows(self) -> Dict[str, List[WorkflowStep]]:
        """Définit les workflows prédéfinis"""
        return {
            "download_data": [
                WorkflowStep(
                    name="fetch_data",
                    script_path="scripts/fetch_data_ccxt.py",
                    description="Télécharger les données de marché",
                    args=["--symbols", "BTCUSDT,ETHUSDT,SOLUSDT", "--timeframes", "5m,1h,4h"],
                    timeout=1800  # 30 minutes
                )
            ],
            
            "prepare_dataset": [
                WorkflowStep(
                    name="convert_data",
                    script_path="scripts/convert_real_data.py",
                    description="Convertir les données brutes",
                    timeout=900  # 15 minutes
                ),
                WorkflowStep(
                    name="merge_data",
                    script_path="scripts/merge_processed_data.py",
                    description="Fusionner les données multi-timeframes",
                    dependencies=["convert_data"],
                    timeout=600  # 10 minutes
                )
            ],
            
            "train_model": [
                WorkflowStep(
                    name="train_agent",
                    script_path="scripts/train_rl_agent.py",
                    description="Entraîner l'agent RL",
                    args=["--config", "config/training_prod.yaml"],
                    timeout=7200  # 2 heures
                )
            ],
            
            "backtest": [
                WorkflowStep(
                    name="evaluate_performance",
                    script_path="scripts/evaluate_performance.py",
                    description="Évaluer les performances (backtest)",
                    args=["--mode", "backtest"],
                    timeout=1800  # 30 minutes
                )
            ],
            
            "paper_trade": [
                WorkflowStep(
                    name="paper_trading",
                    script_path="scripts/paper_trade_agent.py",
                    description="Trading papier en temps réel",
                    args=["--duration", "24h"],
                    timeout=86400  # 24 heures
                )
            ],
            
            "full_pipeline": [
                WorkflowStep(
                    name="fetch_data",
                    script_path="scripts/fetch_data_ccxt.py",
                    description="Télécharger les données",
                    args=["--symbols", "BTCUSDT,ETHUSDT", "--timeframes", "5m,1h,4h"]
                ),
                WorkflowStep(
                    name="convert_data",
                    script_path="scripts/convert_real_data.py",
                    description="Convertir les données",
                    dependencies=["fetch_data"]
                ),
                WorkflowStep(
                    name="merge_data",
                    script_path="scripts/merge_processed_data.py",
                    description="Fusionner les données",
                    dependencies=["convert_data"]
                ),
                WorkflowStep(
                    name="train_agent",
                    script_path="scripts/train_rl_agent.py",
                    description="Entraîner l'agent",
                    dependencies=["merge_data"],
                    args=["--config", "config/training_prod.yaml"]
                ),
                WorkflowStep(
                    name="evaluate_performance",
                    script_path="scripts/evaluate_performance.py",
                    description="Évaluer les performances",
                    dependencies=["train_agent"],
                    args=["--mode", "backtest"]
                )
            ]
        }
    
    def add_progress_callback(self, callback: Callable[[str, float, str], None]) -> None:
        """Ajoute un callback pour les mises à jour de progression"""
        self.progress_callbacks.append(callback)
    
    def add_completion_callback(self, callback: Callable[[str, WorkflowStatus], None]) -> None:
        """Ajoute un callback pour les fins d'exécution"""
        self.completion_callbacks.append(callback)
    
    def add_log_callback(self, callback: Callable[[str, str, str], None]) -> None:
        """Ajoute un callback pour les logs"""
        self.log_callbacks.append(callback)
    
    def execute_workflow(self, workflow_name: str, 
                        custom_args: Dict[str, List[str]] = None) -> str:
        """
        Exécute un workflow complet.
        
        Args:
            workflow_name: Nom du workflow à exécuter
            custom_args: Arguments personnalisés par étape
            
        Returns:
            ID d'exécution du workflow
        """
        if workflow_name not in self.workflows:
            raise ValueError(f"Workflow '{workflow_name}' not found")
        
        workflow_id = f"{workflow_name}_{int(time.time())}"
        steps = self.workflows[workflow_name]
        
        # Appliquer les arguments personnalisés
        if custom_args:
            for step in steps:
                if step.name in custom_args:
                    step.args = custom_args[step.name]
        
        # Lancer l'exécution en arrière-plan
        thread = threading.Thread(
            target=self._execute_workflow_steps,
            args=(workflow_id, steps),
            daemon=True
        )
        thread.start()
        self.execution_threads[workflow_id] = thread
        
        logger.info(f"Started workflow '{workflow_name}' with ID: {workflow_id}")
        return workflow_id
    
    def execute_single_step(self, step: WorkflowStep) -> str:
        """
        Exécute une seule étape.
        
        Args:
            step: Étape à exécuter
            
        Returns:
            ID d'exécution
        """
        execution_id = f"{step.name}_{int(time.time())}"
        
        thread = threading.Thread(
            target=self._execute_single_step,
            args=(execution_id, step),
            daemon=True
        )
        thread.start()
        self.execution_threads[execution_id] = thread
        
        return execution_id
    
    def _execute_workflow_steps(self, workflow_id: str, steps: List[WorkflowStep]) -> None:
        """Exécute les étapes d'un workflow en séquence"""
        completed_steps = set()
        
        for step in steps:
            # Vérifier les dépendances
            if step.dependencies:
                missing_deps = set(step.dependencies) - completed_steps
                if missing_deps:
                    logger.error(f"Missing dependencies for {step.name}: {missing_deps}")
                    self._notify_completion(workflow_id, WorkflowStatus.FAILED)
                    return
            
            # Exécuter l'étape
            step_id = f"{workflow_id}_{step.name}"
            success = self._execute_single_step(step_id, step)
            
            if success:
                completed_steps.add(step.name)
            else:
                logger.error(f"Step {step.name} failed, stopping workflow")
                self._notify_completion(workflow_id, WorkflowStatus.FAILED)
                return
        
        self._notify_completion(workflow_id, WorkflowStatus.COMPLETED)
    
    def _execute_single_step(self, execution_id: str, step: WorkflowStep) -> bool:
        """Exécute une seule étape"""
        execution = WorkflowExecution(
            workflow_id=execution_id,
            step_name=step.name,
            status=WorkflowStatus.RUNNING,
            start_time=datetime.now()
        )
        self.executions[execution_id] = execution
        
        try:
            # Construire la commande
            script_full_path = self.base_path / step.script_path
            if not script_full_path.exists():
                raise FileNotFoundError(f"Script not found: {script_full_path}")
            
            cmd = [sys.executable, str(script_full_path)] + step.args
            
            # Préparer l'environnement
            env = os.environ.copy()
            env.update(step.env_vars)
            
            # Lancer le processus
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env,
                cwd=str(self.base_path)
            )
            
            self.active_processes[execution_id] = process
            
            # Lire la sortie en temps réel
            self._monitor_process_output(execution_id, process, step.timeout)
            
            # Attendre la fin
            exit_code = process.wait(timeout=step.timeout)
            
            # Mettre à jour l'exécution
            execution.status = WorkflowStatus.COMPLETED if exit_code == 0 else WorkflowStatus.FAILED
            execution.exit_code = exit_code
            execution.end_time = datetime.now()
            execution.progress_percent = 100.0
            
            # Nettoyer
            if execution_id in self.active_processes:
                del self.active_processes[execution_id]
            
            # Notifier
            self._notify_progress(execution_id, 100.0, "Completed")
            self._notify_completion(execution_id, execution.status)
            
            return exit_code == 0
            
        except subprocess.TimeoutExpired:
            logger.error(f"Step {step.name} timed out after {step.timeout} seconds")
            execution.status = WorkflowStatus.FAILED
            execution.end_time = datetime.now()
            self._cleanup_process(execution_id)
            return False
            
        except Exception as e:
            logger.error(f"Error executing step {step.name}: {e}")
            execution.status = WorkflowStatus.FAILED
            execution.end_time = datetime.now()
            self._cleanup_process(execution_id)
            return False
    
    def _monitor_process_output(self, execution_id: str, process: subprocess.Popen, timeout: int) -> None:
        """Surveille la sortie du processus en temps réel"""
        def read_output():
            try:
                for line in iter(process.stdout.readline, ''):
                    if line:
                        line = line.strip()
                        self.executions[execution_id].output_lines.append(line)
                        self._notify_log(execution_id, "stdout", line)
                        
                        # Essayer d'extraire le pourcentage de progression
                        progress = self._extract_progress(line)
                        if progress is not None:
                            self.executions[execution_id].progress_percent = progress
                            self._notify_progress(execution_id, progress, line)
            except Exception as e:
                logger.error(f"Error reading stdout: {e}")
        
        def read_errors():
            try:
                for line in iter(process.stderr.readline, ''):
                    if line:
                        line = line.strip()
                        self.executions[execution_id].error_lines.append(line)
                        self._notify_log(execution_id, "stderr", line)
            except Exception as e:
                logger.error(f"Error reading stderr: {e}")
        
        # Lancer les threads de lecture
        stdout_thread = threading.Thread(target=read_output, daemon=True)
        stderr_thread = threading.Thread(target=read_errors, daemon=True)
        
        stdout_thread.start()
        stderr_thread.start()
    
    def _extract_progress(self, line: str) -> Optional[float]:
        """Extrait le pourcentage de progression d'une ligne de log"""
        import re
        
        # Patterns courants pour la progression
        patterns = [
            r'(\d+)%',  # "50%"
            r'(\d+)/(\d+)',  # "50/100"
            r'Progress:\s*(\d+\.?\d*)%',  # "Progress: 50.5%"
            r'Epoch\s+(\d+)/(\d+)',  # "Epoch 5/10"
            r'Step\s+(\d+)/(\d+)',  # "Step 500/1000"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                if len(match.groups()) == 1:
                    return float(match.group(1))
                elif len(match.groups()) == 2:
                    current = float(match.group(1))
                    total = float(match.group(2))
                    return (current / total) * 100.0 if total > 0 else 0.0
        
        return None
    
    def cancel_execution(self, execution_id: str) -> bool:
        """Annule une exécution en cours"""
        if execution_id not in self.executions:
            return False
        
        execution = self.executions[execution_id]
        if execution.status != WorkflowStatus.RUNNING:
            return False
        
        # Terminer le processus
        if execution_id in self.active_processes:
            process = self.active_processes[execution_id]
            try:
                process.terminate()
                time.sleep(2)  # Attendre un peu
                if process.poll() is None:
                    process.kill()  # Force kill si nécessaire
            except Exception as e:
                logger.error(f"Error terminating process: {e}")
        
        # Mettre à jour le statut
        execution.status = WorkflowStatus.CANCELLED
        execution.end_time = datetime.now()
        
        self._cleanup_process(execution_id)
        self._notify_completion(execution_id, WorkflowStatus.CANCELLED)
        
        logger.info(f"Execution {execution_id} cancelled")
        return True
    
    def _cleanup_process(self, execution_id: str) -> None:
        """Nettoie les ressources d'un processus"""
        if execution_id in self.active_processes:
            del self.active_processes[execution_id]
        
        if execution_id in self.execution_threads:
            del self.execution_threads[execution_id]
    
    def get_execution_status(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Récupère le statut d'une exécution"""
        return self.executions.get(execution_id)
    
    def get_all_executions(self) -> Dict[str, WorkflowExecution]:
        """Récupère toutes les exécutions"""
        return self.executions.copy()
    
    def get_active_executions(self) -> Dict[str, WorkflowExecution]:
        """Récupère les exécutions actives"""
        return {
            eid: execution for eid, execution in self.executions.items()
            if execution.status == WorkflowStatus.RUNNING
        }
    
    def _notify_progress(self, execution_id: str, progress: float, message: str) -> None:
        """Notifie les callbacks de progression"""
        for callback in self.progress_callbacks:
            try:
                callback(execution_id, progress, message)
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")
    
    def _notify_completion(self, execution_id: str, status: WorkflowStatus) -> None:
        """Notifie les callbacks de fin d'exécution"""
        for callback in self.completion_callbacks:
            try:
                callback(execution_id, status)
            except Exception as e:
                logger.error(f"Error in completion callback: {e}")
    
    def _notify_log(self, execution_id: str, stream: str, message: str) -> None:
        """Notifie les callbacks de log"""
        for callback in self.log_callbacks:
            try:
                callback(execution_id, stream, message)
            except Exception as e:
                logger.error(f"Error in log callback: {e}")
    
    def save_execution_history(self, filepath: str = None) -> str:
        """Sauvegarde l'historique des exécutions"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"logs/workflow_history_{timestamp}.json"
        
        # Préparer les données
        history = {
            'timestamp': datetime.now().isoformat(),
            'executions': {
                eid: execution.to_dict() 
                for eid, execution in self.executions.items()
            }
        }
        
        # Créer le répertoire si nécessaire
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Execution history saved to: {filepath}")
        return filepath
    
    def cleanup_old_executions(self, max_age_hours: int = 24) -> int:
        """Nettoie les anciennes exécutions"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        to_remove = []
        for eid, execution in self.executions.items():
            if (execution.end_time and execution.end_time < cutoff_time and 
                execution.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]):
                to_remove.append(eid)
        
        for eid in to_remove:
            del self.executions[eid]
        
        logger.info(f"Cleaned up {len(to_remove)} old executions")
        return len(to_remove)
    
    def shutdown(self) -> None:
        """Arrêt propre de l'orchestrateur"""
        logger.info("Shutting down WorkflowOrchestrator...")
        
        # Annuler toutes les exécutions actives
        active_executions = list(self.get_active_executions().keys())
        for execution_id in active_executions:
            self.cancel_execution(execution_id)
        
        # Attendre que les threads se terminent
        for thread in self.execution_threads.values():
            if thread.is_alive():
                thread.join(timeout=5.0)
        
        # Sauvegarder l'historique
        self.save_execution_history()
        
        logger.info("WorkflowOrchestrator shutdown completed")