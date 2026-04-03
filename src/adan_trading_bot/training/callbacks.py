"""
Callbacks personnalisés pour l'entraînement du bot de trading ADAN.
"""
import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.panel import Panel
from rich.rule import Rule
from rich.live import Live
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
import time

from ..common.utils import get_logger

logger = get_logger()
console = Console()

class CustomTrainingInfoCallback(BaseCallback):
    """
    Un callback pour afficher des informations détaillées pendant l'entraînement
    avec barre de progression et métriques en temps réel.
    """
    def __init__(self, check_freq: int, verbose: int = 1):
        super(CustomTrainingInfoCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.start_time = time.time()
        self.rollout_count = 0
        self.progress = None
        self.task_id = None
        self.last_progress_update = 0

    def _on_training_start(self) -> None:
        """
        Appelé au début de l'entraînement avec gestion des flux optimisée.
        """
        self.start_time = time.time()
        console.rule("[bold green]🚀 Début de l'Entraînement ADAN avec Flux Dynamiques[/bold green]")
        
        # Déterminer dynamiquement le capital initial depuis l'environnement/config
        initial_capital_display = 20.0
        try:
            # 1) Essayer via portfolio_manager.initial_capital
            pm_list = self.training_env.get_attr('portfolio_manager')
            if pm_list and hasattr(pm_list[0], 'initial_capital'):
                initial_capital_display = float(pm_list[0].initial_capital)
            else:
                # 2) Essayer via config (portfolio.initial_balance > environment.initial_balance)
                cfg_list = self.training_env.get_attr('config')
                cfg = cfg_list[0] if cfg_list else {}
                env_cfg = cfg.get('environment', {}) if isinstance(cfg, dict) else {}
                portfolio_cfg = cfg.get('portfolio', {}) if isinstance(cfg, dict) else {}
                initial_capital_display = float(
                    portfolio_cfg.get('initial_balance', env_cfg.get('initial_balance', initial_capital_display))
                )
        except Exception:
            # Fallback silencieux
            initial_capital_display = float(initial_capital_display)

        # Affichage des paramètres de flux
        console.print(Panel(
            f"[cyan]💰 Capital Initial: ${initial_capital_display:.2f}[/cyan]\n"
            "[yellow]🎯 Gestion Dynamique des Flux Activée[/yellow]\n"
            "[green]📊 Monitoring en Temps Réel[/green]",
            title="Configuration Flux Monétaires"
        ))
        
        # Initialiser la barre de progression optimisée
        self.progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=50),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed:,}/{task.total:,})"),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console
        )
        
        # Démarrer la barre de progression
        self.progress.start()
        self.task_id = self.progress.add_task(
            "🚀 ADAN Training", 
            total=self.model._total_timesteps
        )

    def _on_rollout_start(self) -> None:
        """
        Appelé au début de chaque collecte de rollout avec monitoring des flux.
        """
        self.rollout_count += 1
        if self.verbose > 0 and self.rollout_count % self.check_freq == 0:
            # Récupérer le capital actuel pour le monitoring
            try:
                current_capital_list = self.training_env.get_attr('capital')
                current_capital = np.mean(current_capital_list) if current_capital_list else 15.0
                console.print(f"[dim]💡 Rollout #{self.rollout_count} | Step: {self.num_timesteps:,} | Capital: ${current_capital:.2f}[/dim]")
            except:
                console.print(f"[dim]💡 Rollout #{self.rollout_count} | Step: {self.num_timesteps:,}[/dim]")

    def _on_step(self) -> bool:
        """
        Appelé à chaque pas. Met à jour la barre de progression avec gestion dynamique.
        """
        # Mettre à jour la barre de progression plus fréquemment
        if self.progress and self.task_id is not None:
            # Mise à jour avec description dynamique selon le capital
            try:
                current_capital_list = self.training_env.get_attr('capital')
                # Utiliser le capital initial dynamique comme valeur par défaut
                default_initial_capital = 20.0
                try:
                    pm_list = self.training_env.get_attr('portfolio_manager')
                    if pm_list and hasattr(pm_list[0], 'initial_capital'):
                        default_initial_capital = float(pm_list[0].initial_capital)
                except Exception:
                    pass
                current_capital = np.mean(current_capital_list) if current_capital_list else default_initial_capital
                
                # Adapter la description selon le capital
                if current_capital > 20.0:
                    description = "🚀 ADAN Training (💰 Profit)"
                elif current_capital < 10.0:
                    description = "🚀 ADAN Training (⚠️ Risk)"
                else:
                    description = "🚀 ADAN Training (📊 Stable)"
                    
                self.progress.update(
                    self.task_id, 
                    completed=self.num_timesteps,
                    description=description
                )
            except:
                self.progress.update(self.task_id, completed=self.num_timesteps)
        
        return True

    def _on_rollout_end(self) -> None:
        """
        Appelé à la fin de chaque collecte de rollout (avant la mise à jour).
        Affiche des métriques détaillées avec gestion optimisée des flux monétaires.
        """
        if self.verbose > 0 and self.rollout_count % self.check_freq == 0:
            # Calculer les métriques de base
            duration = time.time() - self.start_time
            fps = self.num_timesteps / duration if duration > 0 else 0
            progress_pct = (self.num_timesteps / self.model._total_timesteps) * 100
            
            # Récupérer les métriques de l'environnement avec gestion dynamique des flux
            try:
                # Capital actuel
                current_capital_list = self.training_env.get_attr('capital')
                # Déterminer capital initial dynamique pour ROI et valeurs par défaut
                initial_capital_dyn = 20.0
                try:
                    pm_list = self.training_env.get_attr('portfolio_manager')
                    if pm_list and hasattr(pm_list[0], 'initial_capital'):
                        initial_capital_dyn = float(pm_list[0].initial_capital)
                    else:
                        cfg_list = self.training_env.get_attr('config')
                        cfg = cfg_list[0] if cfg_list else {}
                        env_cfg = cfg.get('environment', {}) if isinstance(cfg, dict) else {}
                        portfolio_cfg = cfg.get('portfolio', {}) if isinstance(cfg, dict) else {}
                        initial_capital_dyn = float(
                            portfolio_cfg.get('initial_balance', env_cfg.get('initial_balance', initial_capital_dyn))
                        )
                except Exception:
                    pass

                current_capital = np.mean(current_capital_list) if current_capital_list else initial_capital_dyn
                
                cumulative_reward_list = self.training_env.get_attr('cumulative_reward')
                cumulative_reward = np.mean(cumulative_reward_list) if cumulative_reward_list else 0
                
                # Nouvelles métriques pour flux monétaires
                positions_list = self.training_env.get_attr('positions')
                active_positions = len(positions_list[0]) if positions_list and positions_list[0] else 0
                
                # Calculer le ROI dynamique
                roi_pct = ((current_capital - initial_capital_dyn) / initial_capital_dyn) * 100 if initial_capital_dyn > 0 else 0
                
            except Exception as e:
                # Valeurs de repli sûres
                current_capital = 0.0
                cumulative_reward = 0
                active_positions = 0
                roi_pct = 0
            
            # Récupérer les métriques SB3 avec gestion des erreurs
            try:
                policy_loss = self.model.logger.name_to_value.get('train/policy_loss', 0)
                value_loss = self.model.logger.name_to_value.get('train/value_loss', 0)
                ep_rew_mean = self.model.logger.name_to_value.get('rollout/ep_rew_mean', 0)
                ep_len_mean = self.model.logger.name_to_value.get('rollout/ep_len_mean', 0)
            except:
                policy_loss = value_loss = ep_rew_mean = ep_len_mean = 0
            
            # Affichage optimisé pour flux positifs (moins de logs si ROI > 0)
            if roi_pct > 0 and hasattr(self.model, 'quiet_positive') and self.model.quiet_positive:
                # Mode silencieux pour retours positifs - affichage minimal
                console.print(f"[green]✅ Step {self.num_timesteps:,} | ROI: +{roi_pct:.2f}% | Capital: ${current_capital:.2f}[/green]")
            else:
                # Affichage détaillé avec tableau pour flux négatifs ou mode normal
                metrics_table = Table(title=f"[bold cyan]📊 Rollout #{self.rollout_count} - Step {self.num_timesteps:,}[/bold cyan]")
                metrics_table.add_column("Métrique", style="dim cyan")
                metrics_table.add_column("Valeur", style="bright_white")
                
                # Métriques de flux monétaires
                metrics_table.add_row("💰 Capital Actuel", f"${current_capital:.2f}")
                metrics_table.add_row("📈 ROI", f"{roi_pct:+.2f}%")
                metrics_table.add_row("🎯 Positions Actives", f"{active_positions}")
                metrics_table.add_row("🏆 Récompense Cumulative", f"{cumulative_reward:.4f}")
                
                # Métriques de performance
                metrics_table.add_row("⚡ FPS", f"{fps:.1f}")
                metrics_table.add_row("📊 Progression", f"{progress_pct:.1f}%")
                metrics_table.add_row("🎲 Récompense Moy. Épisode", f"{ep_rew_mean:.4f}")
                
                # Métriques d'entraînement
                if policy_loss != 0:
                    metrics_table.add_row("🧠 Perte Politique", f"{policy_loss:.6f}")
                if value_loss != 0:
                    metrics_table.add_row("💎 Perte Valeur", f"{value_loss:.6f}")
                
                console.print(metrics_table)
                
                # Alerte spéciale pour capital critique
                if current_capital < 5.0:
                    console.print(Panel(
                        f"[bold red]⚠️ CAPITAL CRITIQUE: ${current_capital:.2f}[/bold red]\n"
                        f"Gestion des flux monétaires activée",
                        title="Alerte Flux"
                    ))
            ep_rew_mean = "N/A"
            ep_len_mean = "N/A"
            policy_loss = "N/A"
            
            try:
                latest_values = self.model.logger.name_to_value
                if 'rollout/ep_rew_mean' in latest_values:
                    ep_rew_mean = f"{latest_values['rollout/ep_rew_mean']:.4f}"
                if 'rollout/ep_len_mean' in latest_values:
                    ep_len_mean = f"{latest_values['rollout/ep_len_mean']:.0f}"
                if 'train/policy_loss' in latest_values:
                    policy_loss = f"{latest_values['train/policy_loss']:.6f}"
            except:
                pass
            
            # Créer un affichage compact
            metrics_table = Table(show_header=True, box=None, padding=(0,1))
            metrics_table.add_column("Métrique", style="bold cyan", width=20)
            metrics_table.add_column("Valeur", style="white", width=15)
            metrics_table.add_column("Métrique", style="bold cyan", width=20)
            metrics_table.add_column("Valeur", style="white", width=15)
            
            # Première ligne
            metrics_table.add_row(
                "📊 Progression", f"{progress_pct:.1f}%",
                "💰 Capital", f"${current_capital:,.0f}" if current_capital else "N/A"
            )
            
            # Deuxième ligne
            metrics_table.add_row(
                "⚡ FPS", f"{fps:.1f}",
                "🎯 Récompense Moy", ep_rew_mean
            )
            
            # Troisième ligne
            metrics_table.add_row(
                "🔄 Rollout", f"#{self.rollout_count}",
                "📏 Longueur Ép.", ep_len_mean
            )
            
            # Quatrième ligne
            eta_seconds = (self.model._total_timesteps - self.num_timesteps) / fps if fps > 0 else 0
            eta_minutes = eta_seconds / 60
            eta_display = f"{eta_minutes:.0f}min" if eta_minutes < 60 else f"{eta_minutes/60:.1f}h"
            
            metrics_table.add_row(
                "⏱️  ETA", eta_display,
                "🔧 Perte Pol.", policy_loss
            )
            
            # Affichage compact
            console.print(f"\n📈 [bold yellow]Step {self.num_timesteps:,}[/bold yellow] | [bold green]Rollout #{self.rollout_count}[/bold green]")
            console.print(metrics_table)
            console.print("─" * 80)
    
    def _on_training_end(self) -> None:
        """
        Appelé à la fin de l'entraînement.
        """
        # Arrêter la barre de progression
        if self.progress:
            self.progress.stop()
        
        duration = time.time() - self.start_time
        hours = duration // 3600
        minutes = (duration % 3600) // 60
        seconds = duration % 60
        
        time_str = f"{int(hours)}h{int(minutes):02d}m{int(seconds):02d}s" if hours > 0 else f"{int(minutes)}m{int(seconds):02d}s"
        
        console.rule(f"[bold green]✅ Entraînement ADAN Terminé - Durée: {time_str}[/bold green]")
        
        # Afficher un résumé final
        final_table = Table(title="[bold cyan]🎯 Résumé Final[/bold cyan]")
        final_table.add_column("Métrique", style="bold")
        final_table.add_column("Valeur", style="green")
        
        final_table.add_row("Timesteps Total", f"{self.num_timesteps:,}")
        final_table.add_row("Rollouts", str(self.rollout_count))
        final_table.add_row("Durée", time_str)
        final_table.add_row("FPS Moyenne", f"{self.num_timesteps / duration:.1f}")
        
        console.print(Panel(final_table, expand=False))


class EvaluationCallback(BaseCallback):
    """
    Callback pour évaluer l'agent pendant l'entraînement.
    """
    def __init__(self, eval_env, eval_freq=10000, n_eval_episodes=5, verbose=1):
        super(EvaluationCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_mean_reward = -np.inf
        self.last_eval_timestep = 0
    
    def _on_step(self) -> bool:
        """
        Évaluer l'agent à intervalles réguliers.
        """
        if self.num_timesteps - self.last_eval_timestep >= self.eval_freq:
            self.last_eval_timestep = self.num_timesteps
            
            # Évaluer l'agent
            episode_rewards = []
            episode_lengths = []
            
            for _ in range(self.n_eval_episodes):
                # Réinitialiser l'environnement
                obs, _ = self.eval_env.reset()
                done = False
                episode_reward = 0.0
                episode_length = 0
                
                while not done:
                    # Prédire l'action
                    action, _ = self.model.predict(obs, deterministic=True)
                    
                    # Exécuter l'action
                    obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                    done = terminated or truncated
                    
                    # Mettre à jour les compteurs
                    episode_reward += reward
                    episode_length += 1
                
                # Enregistrer les résultats
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
            
            # Calculer les moyennes
            mean_reward = np.mean(episode_rewards)
            mean_length = np.mean(episode_lengths)
            
            # Afficher les résultats
            eval_table = Table(title=f"[bold cyan]Évaluation à {self.num_timesteps} timesteps[/bold cyan]")
            eval_table.add_column("Métrique", style="dim cyan")
            eval_table.add_column("Valeur")
            
            eval_table.add_row("Récompense Moyenne", f"{mean_reward:.4f}")
            eval_table.add_row("Longueur Moyenne Épisode", f"{mean_length:.1f}")
            
            # Vérifier si c'est le meilleur modèle jusqu'à présent
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                eval_table.add_row("Meilleure Récompense", f"[bold green]{mean_reward:.4f}[/bold green] (Nouveau record!)")
            else:
                eval_table.add_row("Meilleure Récompense", f"{self.best_mean_reward:.4f}")
            
            console.print(Panel(eval_table, expand=False))
            
            # Logguer les métriques
            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/mean_ep_length", mean_length)
            self.logger.record("eval/best_mean_reward", self.best_mean_reward)
        
        return True
