
import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from .frequency_manager import FrequencyManager

class FrequencyAwareTradingEnv(gym.Env):
    """Extension de l'environnement de trading avec gestion des fréquences"""
    
    def __init__(self, base_env, config: Dict):
        self.base_env = base_env
        self.frequency_manager = FrequencyManager(config)
        
        # Hériter les espaces d'action et d'observation du base_env
        self.action_space = base_env.action_space
        self.observation_space = base_env.observation_space
        
        # Métriques de fréquence
        self.current_worker_id = None
        self.episode_start_time = None
        
    def set_worker_id(self, worker_id: str):
        """Définit l'ID du worker pour le suivi des fréquences"""
        self.current_worker_id = worker_id
        if self.episode_start_time is None:
            self.episode_start_time = pd.Timestamp.now()
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Exécute une étape avec suivi des fréquences"""
        obs, reward, terminated, truncated, info = self.base_env.step(action)
        
        if self.current_worker_id and self.frequency_manager.enabled:
            # Analyser l'action pour détecter le timeframe
            timeframe = self._extract_timeframe_from_action(action)
            if timeframe:
                timestamp = pd.Timestamp.now()
                self.frequency_manager.record_trade(
                    self.current_worker_id, 
                    timeframe, 
                    timestamp
                )
        
        return obs, reward, terminated, truncated, info
    
    def _extract_timeframe_from_action(self, action: np.ndarray) -> Optional[str]:
        """Extrait le timeframe depuis l'action du modèle"""
        # Logique simplifiée - à adapter selon votre format d'action
        if len(action) >= 2:
            # Supposons que l'action contient [position, timeframe]
            # timeframe_encoded = action[1]  # 0=5m, 1=1h, 2=4h
            # 
            # Pour l'instant, retourner None (à implémenter selon votre logique)
            pass
        
        return None
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Reset l'environnement avec gestion des fréquences"""
        obs, info = self.base_env.reset(**kwargs)
        
        if self.current_worker_id:
            # Calculer les métriques de fréquence avant reset
            frequency_metrics = self.frequency_manager.get_metrics(self.current_worker_id)
            info['frequency_metrics'] = frequency_metrics
            
            # Reset optionnel (pas à chaque épisode pour éviter les resets trop fréquents)
            if np.random.random() < 0.1:  # Reset seulement 10% du temps
                self.frequency_manager.reset_worker(self.current_worker_id)
        
        return obs, info
    
    def render(self):
        """Affichage avec métriques de fréquence"""
        self.base_env.render()
        
        if self.current_worker_id and self.frequency_manager.enabled:
            metrics = self.frequency_manager.get_metrics(self.current_worker_id)
            print(f"[FREQUENCY VALIDATION {self.current_worker_id}] ", end='')
            for tf, status in metrics['validation']['validation'].items():
                print(f'{tf}: {status} | ', end='')
            print(f'Total: {metrics["validation"]["valid_timeframes"]}/3 ✓')
