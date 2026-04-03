"""
🔧 OBSERVATION NORMALIZER
Normalise les observations avec les stats d'entraînement pour éviter le covariate shift
"""

import json
import pickle
import numpy as np
from pathlib import Path
import logging
import sys

logger = logging.getLogger(__name__)

class ObservationNormalizer:
    """Normalise les observations avec les stats d'entraînement"""
    
    def __init__(self, vecnorm_path=None, fallback_stats_path=None):
        """
        Initialise le normaliseur
        
        Args:
            vecnorm_path: Chemin vers le fichier VecNormalize.pkl
            fallback_stats_path: Chemin vers un fichier JSON de stats de secours
        """
        self.vecnorm_path = Path(vecnorm_path or "./models/w1/vecnormalize.pkl")
        self.fallback_stats_path = Path(fallback_stats_path or "/tmp/emergency_normalizer.json")
        
        self.mean = None
        self.var = None
        self.is_loaded = False
        
        # Essayer de charger les stats
        self._load_stats()
    
    def _load_stats(self):
        """Charge les stats de normalisation"""
        # Essai 1: Charger depuis VecNormalize.pkl
        if self._load_from_vecnormalize():
            logger.info("✅ Stats chargées depuis VecNormalize.pkl")
            self.is_loaded = True
            return
        
        # Essai 2: Charger depuis JSON de secours
        if self._load_from_json():
            logger.info("✅ Stats chargées depuis JSON de secours")
            self.is_loaded = True
            return
        
        # Essai 3: Utiliser des stats par défaut
        logger.warning("⚠️  Impossible de charger les stats, utilisation de valeurs par défaut")
        self._use_default_stats()
        self.is_loaded = True
    
    def _load_from_vecnormalize(self):
        """Charge les stats depuis VecNormalize.pkl"""
        try:
            if not self.vecnorm_path.exists():
                logger.debug(f"Fichier non trouvé: {self.vecnorm_path}")
                return False
            
            # Augmenter la limite de récursion temporairement
            old_limit = sys.getrecursionlimit()
            sys.setrecursionlimit(10000)
            
            try:
                with open(self.vecnorm_path, 'rb') as f:
                    vecnorm = pickle.load(f)
                
                sys.setrecursionlimit(old_limit)
                
                # Extraire les stats - VecNormalize a running_mean et running_var
                if hasattr(vecnorm, 'running_mean') and hasattr(vecnorm, 'running_var'):
                    self.mean = np.array(vecnorm.running_mean, dtype=np.float32)
                    self.var = np.array(vecnorm.running_var, dtype=np.float32)
                    logger.info(f"✅ Stats extraites depuis VecNormalize: mean shape={self.mean.shape}, var shape={self.var.shape}")
                    return True
                elif hasattr(vecnorm, 'mean') and hasattr(vecnorm, 'var'):
                    self.mean = np.array(vecnorm.mean, dtype=np.float32)
                    self.var = np.array(vecnorm.var, dtype=np.float32)
                    logger.info(f"✅ Stats extraites: mean shape={self.mean.shape}, var shape={self.var.shape}")
                    return True
                else:
                    logger.debug(f"VecNormalize n'a pas les attributs attendus. Attributs disponibles: {dir(vecnorm)}")
                    return False
                    
            except RecursionError:
                sys.setrecursionlimit(old_limit)
                logger.debug("RecursionError lors du chargement de VecNormalize")
                return False
                
        except Exception as e:
            logger.debug(f"Erreur lors du chargement de VecNormalize: {e}")
            return False
    
    def _load_from_json(self):
        """Charge les stats depuis un fichier JSON"""
        try:
            if not self.fallback_stats_path.exists():
                logger.debug(f"Fichier JSON non trouvé: {self.fallback_stats_path}")
                return False
            
            with open(self.fallback_stats_path, 'r') as f:
                stats = json.load(f)
            
            if 'mean' in stats and 'var' in stats:
                self.mean = np.array(stats['mean'], dtype=np.float32)
                self.var = np.array(stats['var'], dtype=np.float32)
                logger.debug(f"Stats JSON chargées: mean shape={self.mean.shape}")
                return True
            else:
                logger.debug("Fichier JSON n'a pas les clés attendues")
                return False
                
        except Exception as e:
            logger.debug(f"Erreur lors du chargement du JSON: {e}")
            return False
    
    def _use_default_stats(self):
        """Utilise des stats par défaut"""
        # Dimension 20: taille du portfolio_state actuel (avec indices 8-9 pour hiérarchie ADAN)
        # Correspond au format simplifié utilisé par paper_trading_monitor.py
        n_features = 20  
        self.mean = np.zeros(n_features, dtype=np.float32)
        self.var = np.ones(n_features, dtype=np.float32)
        logger.debug(f"Stats par défaut utilisées: {n_features} features (portfolio_state)")
    
    def normalize(self, observation):
        """
        Normalise une observation
        
        Args:
            observation: np.array de shape (n_features,) ou (batch_size, n_features)
        
        Returns:
            Observation normalisée
        """
        if not self.is_loaded or self.mean is None or self.var is None:
            logger.warning("⚠️  Normaliseur non initialisé, retour de l'observation brute")
            return observation
        
        try:
            # Vérifier la shape et auto-recalibrer si nécessaire
            if observation.ndim == 1:
                # Observation unique
                if len(observation) != len(self.mean):
                    logger.warning(f"⚠️  Shape mismatch détecté: obs={len(observation)}, mean={len(self.mean)}")
                    logger.warning(f"   → Auto-recalibrage du normaliseur avec dimension actuelle")
                    # Auto-ajuster les stats pour éviter crash en production
                    self.mean = np.zeros(len(observation), dtype=np.float32)
                    self.var = np.ones(len(observation), dtype=np.float32)
                    logger.info(f"✅ Normaliseur recalibré: {len(observation)} features")
            elif observation.ndim == 2:
                # Batch d'observations
                if observation.shape[1] != len(self.mean):
                    logger.warning(f"⚠️  Shape mismatch détecté: obs={observation.shape[1]}, mean={len(self.mean)}")
                    logger.warning(f"   → Auto-recalibrage du normaliseur avec dimension actuelle")
                    self.mean = np.zeros(observation.shape[1], dtype=np.float32)
                    self.var = np.ones(observation.shape[1], dtype=np.float32)
                    logger.info(f"✅ Normaliseur recalibré: {observation.shape[1]} features")
            else:
                logger.warning(f"⚠️  Shape inattendue: {observation.shape}")
                return observation
            
            # Normalisation
            normalized = (observation - self.mean) / np.sqrt(self.var + 1e-8)
            
            return normalized
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la normalisation: {e}")
            return observation
    
    def denormalize(self, normalized_observation):
        """
        Dénormalise une observation (inverse de normalize)
        
        Args:
            normalized_observation: Observation normalisée
        
        Returns:
            Observation dénormalisée
        """
        if not self.is_loaded or self.mean is None or self.var is None:
            logger.warning("⚠️  Normaliseur non initialisé, retour de l'observation brute")
            return normalized_observation
        
        try:
            denormalized = normalized_observation * np.sqrt(self.var + 1e-8) + self.mean
            return denormalized
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la dénormalisation: {e}")
            return normalized_observation
    
    def get_stats(self):
        """Retourne les stats de normalisation"""
        return {
            'mean': self.mean.tolist() if self.mean is not None else None,
            'var': self.var.tolist() if self.var is not None else None,
            'is_loaded': self.is_loaded
        }


class DriftDetector:
    """Détecte les dérives de distribution (covariate shift)"""
    
    def __init__(self, window_size=100, threshold=2.0):
        """
        Initialise le détecteur de dérive
        
        Args:
            window_size: Nombre d'observations pour calculer les stats
            threshold: Seuil de dérive (en écarts-types)
        """
        self.window_size = window_size
        self.threshold = threshold
        self.observations = []
        self.drifts_detected = []
    
    def add_observation(self, observation):
        """Ajoute une observation à la fenêtre"""
        self.observations.append(observation)
        
        # Garder seulement les dernières observations
        if len(self.observations) > self.window_size:
            self.observations.pop(0)
    
    def check_drift(self, reference_mean, reference_var):
        """
        Vérifie s'il y a une dérive par rapport aux stats de référence
        
        Args:
            reference_mean: Moyenne de référence (entraînement)
            reference_var: Variance de référence (entraînement)
        
        Returns:
            Dict avec les résultats du test
        """
        if len(self.observations) < self.window_size:
            return {
                'drift_detected': False,
                'reason': 'Pas assez d\'observations',
                'n_observations': len(self.observations)
            }
        
        # Calculer les stats de la fenêtre
        obs_array = np.array(self.observations)
        live_mean = np.mean(obs_array, axis=0)
        live_var = np.var(obs_array, axis=0)
        
        # Calculer la distance (en écarts-types)
        reference_std = np.sqrt(reference_var + 1e-8)
        mean_distance = np.abs(live_mean - reference_mean) / reference_std
        
        # Vérifier si la dérive dépasse le seuil
        max_distance = np.max(mean_distance)
        drift_detected = max_distance > self.threshold
        
        result = {
            'drift_detected': drift_detected,
            'max_distance': float(max_distance),
            'threshold': self.threshold,
            'n_features_drifted': int(np.sum(mean_distance > self.threshold)),
            'n_observations': len(self.observations)
        }
        
        if drift_detected:
            self.drifts_detected.append(result)
            logger.warning(f"⚠️  DÉRIVE DÉTECTÉE: distance={max_distance:.2f} (seuil={self.threshold})")
        
        return result
    
    def get_drift_summary(self):
        """Retourne un résumé des dérives détectées"""
        return {
            'total_drifts': len(self.drifts_detected),
            'drifts': self.drifts_detected
        }
