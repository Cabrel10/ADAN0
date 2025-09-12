"""
Hyperparameter Modulator for RL Agent

Ce module permet d'ajuster dynamiquement les hyperparamètres d'un agent RL
en fonction des signaux de modulation du Dynamic Behavior Engine (DBE).
"""
from typing import Dict, Any, Optional
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from ..common.utils import get_logger

logger = get_logger()

class HyperparameterModulator:
    """
    Module de modulation des hyperparamètres pour l'agent RL.
    
    Permet d'ajuster dynamiquement les hyperparamètres de l'agent en fonction
    des signaux de modulation du DBE.
    """
    
    def __init__(self, agent: BaseAlgorithm, config: Dict[str, Any]):
        """
        Initialise le modulateur d'hyperparamètres.
        
        Args:
            agent: L'agent RL dont les hyperparamètres doivent être modulés
            config: Configuration du modulateur
        """
        self.agent = agent
        self.config = config
        self.initial_params = self._get_current_params()
        
        # Paramètres de modulation
        self.learning_rate_bounds = (
            config.get('min_learning_rate', 1e-6),
            config.get('max_learning_rate', 1e-3)
        )
        self.ent_coef_bounds = (
            config.get('min_ent_coef', 1e-4),
            config.get('max_ent_coef', 0.1)
        )
        
        logger.info("HyperparameterModulator initialisé avec succès")
    
    def _get_current_params(self) -> Dict[str, Any]:
        """Récupère les paramètres actuels de l'agent."""
        return {
            'learning_rate': getattr(self.agent, 'learning_rate', None),
            'ent_coef': getattr(self.agent, 'ent_coef', None),
            'gamma': getattr(self.agent, 'gamma', None),
            'batch_size': getattr(self.agent, 'batch_size', None)
        }
    
    def adjust_params(self, modulation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ajuste les hyperparamètres de l'agent en fonction de la modulation.
        
        Args:
            modulation: Dictionnaire de modulation du DBE
            
        Returns:
            Dict contenant les modifications apportées aux paramètres
        """
        if not modulation:
            return {}
        
        changes = {}
        
        # Mode de risque du DBE
        risk_mode = modulation.get('risk_mode', 'NEUTRAL')
        
        # Ajustement du learning rate
        if 'learning_rate' in self.initial_params:
            base_lr = self.initial_params['learning_rate']
            
            if risk_mode == 'DEFENSIVE':
                # Réduire le learning rate pour stabiliser
                new_lr = max(
                    self.learning_rate_bounds[0],
                    base_lr * self.config.get('defensive_lr_factor', 0.9)
                )
                self.agent.learning_rate = new_lr
                changes['learning_rate'] = new_lr
                
            elif risk_mode == 'AGGRESSIVE':
                # Augmenter légèrement le learning rate
                new_lr = min(
                    self.learning_rate_bounds[1],
                    base_lr * self.config.get('aggressive_lr_factor', 1.05)
                )
                self.agent.learning_rate = new_lr
                changes['learning_rate'] = new_lr
        
        # Ajustement du coefficient d'entropie
        if 'ent_coef' in self.initial_params and self.initial_params['ent_coef'] is not None:
            current_ent_coef = self.agent.ent_coef
            
            if risk_mode == 'DEFENSIVE':
                # Augmenter l'entropie pour encourager l'exploration
                new_ent_coef = min(
                    self.ent_coef_bounds[1],
                    current_ent_coef * self.config.get('defensive_ent_factor', 1.1)
                )
                self.agent.ent_coef = new_ent_coef
                changes['ent_coef'] = new_ent_coef
                
            elif risk_mode == 'AGGRESSIVE':
                # Réduire l'entropie pour exploiter les opportunités
                new_ent_coef = max(
                    self.ent_coef_bounds[0],
                    current_ent_coef * self.config.get('aggressive_ent_factor', 0.95)
                )
                self.agent.ent_coef = new_ent_coef
                changes['ent_coef'] = new_ent_coef
        
        # Ajustement du gamma (facteur d'actualisation)
        if 'gamma' in self.initial_params:
            base_gamma = self.initial_params['gamma']
            
            if risk_mode == 'DEFENSIVE':
                # Réduire gamma pour se concentrer sur les récompenses à court terme
                new_gamma = max(0.9, base_gamma * 0.98)
                self.agent.gamma = new_gamma
                changes['gamma'] = new_gamma
                
            elif risk_mode == 'AGGRESSIVE':
                # Augmenter gamma pour prendre en compte les récompenses à plus long terme
                new_gamma = min(0.999, base_gamma * 1.01)
                self.agent.gamma = new_gamma
                changes['gamma'] = new_gamma
        
        if changes:
            logger.info(
                f"Hyperparamètres ajustés (mode {risk_mode}): {changes}"
            )
        
        return changes
    
    def reset_to_initial(self) -> None:
        """Réinitialise les paramètres de l'agent à leurs valeurs initiales."""
        for param, value in self.initial_params.items():
            if hasattr(self.agent, param) and value is not None:
                setattr(self.agent, param, value)
        
        logger.info("Hyperparamètres réinitialisés aux valeurs initiales")
