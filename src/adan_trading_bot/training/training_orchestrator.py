"""Module d'orchestration de la formation des modèles de trading."""

import logging
import time
import numpy as np
import torch
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable


class TrainingOrchestrator:
    """Orchestrateur principal pour la formation distribuée des modèles de trading."""

    def __init__(self, config: Dict[str, Any], shared_buffer=None):
        """Initialise l'orchestrateur de formation.

        Args:
            config: Configuration de l'orchestrateur
            shared_buffer: Buffer d'expérience partagé optionnel
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.worker_models = {}
        self.sync_lock = threading.Lock()
        self.global_step = 0
        self.last_sync_step = 0
        self.last_sync_time = time.time()
        self.averaged_model = None
        self.shared_buffer = shared_buffer

        # Métriques de suivi
        self.metrics = {
            "episode_rewards": [],
            "episode_lengths": [],
            "learning_rates": [],
            "entropy_coeffs": [],
            "best_mean_reward": -np.inf,
            "buffer_size": [],
            "buffer_additions": 0,
            "buffer_samples_used": 0,
            "last_buffer_stats": {}
        }
        
        # Configuration du monitoring
        self.monitoring_interval = config.get("monitoring_interval", 60)  # secondes
        self.last_monitor_update = time.time()

        # Configuration des intervalles
        # 5 min par défaut pour le monitoring
        self.monitoring_interval = config.get("monitoring_interval", 300)
        # Pas entre les synchronisations
        self.sync_frequency = config.get("sync_frequency", 100)
        self.last_monitor_update = time.time()

    def _log_buffer_stats(self, force: bool = False) -> None:
        """Enregistre les statistiques du buffer partagé.
        
        Args:
            force: Si True, force la journalisation même si l'intervalle n'est pas atteint
        """
        current_time = time.time()
        if not force and current_time - self.last_monitor_update < self.monitoring_interval:
            return

        if self.shared_buffer is not None:
            try:
                # Récupérer les statistiques détaillées du buffer
                stats = self.shared_buffer.get_stats()
                
                # Mettre à jour les métriques de base
                self.metrics["buffer_size"].append((self.global_step, stats.get("size", 0)))
                self.metrics["buffer_additions"] = stats.get("total_added", 0)
                self.metrics["buffer_samples_used"] = stats.get("total_sampled", 0)
                
                # Journalisation détaillée
                log_msg = [
                    "\n" + "="*50,
                    "BUFFER STATISTIQUES DÉTAILLÉES",
                    "="*50,
                    f"Taille: {stats.get('size', 0):,}/{stats.get('max_size', 0):,} "
                    f"({stats.get('utilization_percent', 0):.1f}%)",
                    f"Ajouts totaux: {stats.get('total_added', 0):,} "
                    f"({stats.get('add_rate_per_second', 0):.1f}/s)",
                    f"Échantillons utilisés: {stats.get('total_sampled', 0):,} "
                    f"({stats.get('sample_rate_per_second', 0):.1f}/s)",
                    f"Dernier ajout: il y a {stats.get('seconds_since_last_add', 0):.1f}s",
                    f"Dernier échantillonnage: il y a {stats.get('seconds_since_last_sample', 0):.1f}s",
                    f"Priorité max: {stats.get('priority_max', 0):.4f}, Beta: {stats.get('beta', 0):.4f}",
                    "="*50
                ]
                
                self.logger.info("\n".join(log_msg))
                
                # Mettre à jour les métriques pour TensorBoard/MLflow
                self.metrics["last_buffer_stats"] = {
                    **stats,  # Inclure toutes les statistiques du buffer
                    "timestamp": current_time,
                    "global_step": self.global_step
                }
                
                # Vérifier les problèmes potentiels
                if stats.get("size", 0) == 0:
                    self.logger.warning("Le buffer est vide! Vérifiez l'ajout d'expériences.")
                    
                if stats.get("seconds_since_last_add", 0) > 300:  # 5 minutes
                    self.logger.warning(
                        f"Aucun ajout au buffer depuis {stats['seconds_since_last_add']/60:.1f} minutes. "
                        "Vérifiez les workers d'expérience."
                    )
                    
                if stats.get("utilization_percent", 0) > 90:
                    self.logger.warning(
                        f"Le buffer est presque plein ({stats['utilization_percent']:.1f}%). "
                        "Envisagez d'augmenter sa taille ou d'ajuster la stratégie d'échantillonnage."
                    )
                
                # Ne pas réinitialiser les compteurs ici car ils sont maintenant gérés par le buffer
                
            except Exception as e:
                self.logger.error(f"Erreur lors de la récupération des stats du buffer: {e}", exc_info=True)
        else:
            self.logger.warning("Aucun buffer partagé configuré pour le monitoring")
        
        self.last_monitor_update = current_time

    def _synchronize_models(self, force: bool = False) -> bool:
        """Synchronise les modèles des workers en moyennant leurs poids.

        Args:
            force: Si True, force la synchronisation même si la fréquence
                   n'est pas atteinte

        Returns:
            bool: True si la synchronisation a eu lieu, False sinon
        """
        # Mettre à jour les statistiques du buffer
        self._log_buffer_stats(force)
        if not self._should_sync_models() and not force:
            return False

        start_time = time.time()
        with self.sync_lock:
            if len(self.worker_models) < 2:
                return False

            # Récupérer les modèles à synchroniser
            models = list(self.worker_models.values())

            # Calculer la moyenne des poids
            self._average_models(models)

            # Mettre à jour le dernier pas de synchronisation
            self.last_sync_step = self.global_step
            self.last_sync_time = time.time()

            self.logger.debug(
                f"Modèles synchronisés en {time.time() - start_time:.2f}s "
                f"(étape {self.global_step}, {len(models)} workers)"
            )

            return True

    def _average_models(self, models: List[Any]) -> None:
        """Calcule et applique la moyenne des poids des modèles.

        Args:
            models: Liste des modèles à moyenner
        """
        if not models:
            self.logger.warning("Aucun modèle à moyenner")
            return

        try:
            # Vérifier que tous les modèles sont du même type
            model_type = type(models[0])
            if not all(isinstance(m, model_type) for m in models):
                raise ValueError("Tous les modèles doivent être du même type")

            # Calculer les poids moyens
            avg_weights = {}
            param_keys = models[0].policy.state_dict().keys()

            for key in param_keys:
                # Récupérer les poids de chaque modèle pour ce paramètre
                weights = [model.policy.state_dict()[key].float() for model in models]

                # Calculer la moyenne des poids
                if weights:
                    if isinstance(weights[0], torch.Tensor):
                        avg_weights[key] = torch.stack(weights, dim=0).mean(dim=0)
                    else:
                        avg_weights[key] = np.mean(weights, axis=0)

            # Appliquer les poids moyens à chaque modèle
            for model in models:
                model.policy.load_state_dict(avg_weights, strict=True)

            self.logger.debug(
                f"Moyenne des poids calculée et appliquée sur {len(models)} modèles"
            )

        except Exception as e:
            self.logger.error(f"Erreur lors du calcul de la moyenne des modèles: {e}")
            raise

    def _should_sync_models(self) -> bool:
        """Détermine si une synchronisation des modèles est nécessaire.

        Returns:
            bool: True si une synchronisation est nécessaire, False sinon
        """
        steps_since_last_sync = self.global_step - self.last_sync_step
        time_since_last_sync = time.time() - self.last_sync_time

        # Synchroniser si on a atteint le nombre d'étapes défini
        if steps_since_last_sync >= self.sync_frequency:
            return True

        # Synchroniser si trop de temps s'est écoulé
        if time_since_last_sync > 300:  # 5 minutes
            return True

        return False

    def _cleanup(self) -> None:
        """Nettoie les ressources utilisées par l'orchestrateur."""
        # Libérer les ressources des workers
        for worker_id in list(self.worker_models.keys()):
            self.unregister_worker(worker_id)

        # Fermer les environnements
        if hasattr(self, "env") and self.env is not None:
            try:
                self.env.close()
            except Exception as e:
                self.logger.error(
                    f"Erreur lors de la fermeture de l'environnement: {e}"
                )

        self.logger.info("Nettoyage terminé")

    def unregister_worker(self, worker_id: str) -> None:
        """Désenregistre un worker de l'orchestrateur.

        Args:
            worker_id: Identifiant du worker à désenregistrer
        """
        with self.sync_lock:
            if worker_id in self.worker_models:
                del self.worker_models[worker_id]
                self.logger.info(f"Worker {worker_id} désenregistré")

    def train(
        self,
        total_timesteps: int,
        callback: Optional[Callable] = None,
    ) -> None:
        """Lance la boucle d'entraînement principale.

        Args:
            total_timesteps: Nombre total d'étapes d'entraînement
            callback: Fonction de rappel optionnelle à exécuter
        """
        self.logger.info(f"Début de l'entraînement pour {total_timesteps} pas")
        start_time = time.time()

        try:
            while self.global_step < total_timesteps:
                if self._should_stop_training():
                    msg = "Signal d'arrêt détecté, arrêt de l'entraînement"
                    self.logger.info(msg)
                    break

                self._train_step()
                self._synchronize_models()
                self._update_metrics()

                if callback is not None:
                    callback(locals(), globals())

                self.global_step += 1

        except KeyboardInterrupt:
            self.logger.info("Entraînement interrompu par l'utilisateur")
        except Exception as e:
            self.logger.error(f"Erreur lors de l'entraînement: {e}", exc_info=True)
        finally:
            self._cleanup()
            self._save_model("final_model")
            training_duration = time.time() - start_time
            self.logger.info(
                f"Entraînement terminé après {self.global_step} étapes "
                f"({training_duration:.2f} secondes)"
            )

    def _train_step(self) -> None:
        """Exécute une seule étape d'entraînement."""
        # Implémentation de l'étape d'entraînement
        # À adapter selon votre algorithme d'apprentissage
        pass

    def _update_metrics(self) -> None:
        """Met à jour les métriques de suivi."""
        # Mettre à jour les statistiques du buffer périodiquement
        self._log_buffer_stats()
        
        # Autres métriques à mettre à jour...
        if hasattr(self, 'env') and hasattr(self.env, 'get_episode_rewards'):
            episode_rewards = self.env.get_episode_rewards()
            if episode_rewards:
                self.metrics["episode_rewards"].append(episode_rewards[-1])
                
                # Mettre à jour la meilleure récompense moyenne
                mean_reward = np.mean(episode_rewards[-100:])  # Derniers 100 épisodes
                if mean_reward > self.metrics["best_mean_reward"]:
                    self.metrics["best_mean_reward"] = mean_reward
        
        # Journaliser les métriques périodiquement
        current_time = time.time()
        if current_time - getattr(self, 'last_metric_log', 0) > 300:  # Toutes les 5 minutes
            self.logger.info(
                f"Métriques - Étape: {self.global_step}, "
                f"Dernière récompense: {self.metrics['episode_rewards'][-1] if self.metrics['episode_rewards'] else 'N/A'}, "
                f"Meilleure récompense moyenne: {self.metrics['best_mean_reward']:.2f}"
            )
            self.last_metric_log = current_time

    def _log_metrics(self) -> None:
        """Enregistre les métriques actuelles."""
        if not self.metrics["episode_rewards"]:
            return

        latest = self.metrics["episode_rewards"][-1]
        mean = np.mean(self.metrics["episode_rewards"][-100:])
        std = np.std(self.metrics["episode_rewards"][-100:])

        self.logger.info("=" * 50)
        self.logger.info(f"Étape: {self.global_step}")
        self.logger.info(f"Dernière récompense: {latest:.2f}")
        self.logger.info(f"Moyenne (100): {mean:.2f} ± {std:.2f}")
        self.logger.info("=" * 50)

    def _should_stop_training(self) -> bool:
        """Vérifie si l'entraînement doit s'arrêter.

        Returns:
            bool: True si l'entraînement doit s'arrêter, False sinon
        """
        # Vérifier la présence d'un fichier d'arrêt
        stop_file = Path("stop_training")
        if stop_file.exists():
            stop_file.unlink()  # Supprimer le fichier pour les prochaines exécutions
            return True
        return False

    def _save_model(self, model_name: str) -> None:
        """Enregistre le modèle sur le disque.

        Args:
            model_name: Nom à utiliser pour l'enregistrement du modèle
        """
        if self.averaged_model is not None:
            try:
                # Adapter selon votre implémentation de sauvegarde de modèle
                save_path = f"models/{model_name}"
                self.averaged_model.save(save_path)
                self.logger.info(f"Modèle enregistré sous {save_path}")
            except Exception as e:
                self.logger.error(f"Erreur lors de l'enregistrement du modèle: {e}")
