"""
Tests pour ObservationNormalizer et DriftDetector
"""

import pytest
import numpy as np
from pathlib import Path
import json
import tempfile

from adan_trading_bot.normalization import ObservationNormalizer, DriftDetector


class TestObservationNormalizer:
    """Tests pour ObservationNormalizer"""
    
    def test_normalizer_initialization(self):
        """Test l'initialisation du normaliseur"""
        normalizer = ObservationNormalizer()
        assert normalizer is not None
        assert normalizer.mean is not None
        assert normalizer.var is not None
    
    def test_normalize_single_observation(self):
        """Test la normalisation d'une observation unique"""
        normalizer = ObservationNormalizer()
        
        # Créer une observation
        obs = np.random.randn(68) * 10 + 50
        
        # Normaliser
        normalized = normalizer.normalize(obs)
        
        # Vérifier que la shape est préservée
        assert normalized.shape == obs.shape
        
        # Vérifier que les valeurs sont différentes
        assert not np.allclose(normalized, obs)
    
    def test_normalize_batch_observations(self):
        """Test la normalisation d'un batch d'observations"""
        normalizer = ObservationNormalizer()
        
        # Créer un batch
        batch = np.random.randn(10, 68) * 10 + 50
        
        # Normaliser
        normalized = normalizer.normalize(batch)
        
        # Vérifier que la shape est préservée
        assert normalized.shape == batch.shape
        
        # Vérifier que les valeurs sont différentes
        assert not np.allclose(normalized, batch)
    
    def test_denormalize(self):
        """Test la dénormalisation"""
        normalizer = ObservationNormalizer()
        
        # Créer une observation
        obs = np.random.randn(68) * 10 + 50
        
        # Normaliser puis dénormaliser
        normalized = normalizer.normalize(obs)
        denormalized = normalizer.denormalize(normalized)
        
        # Vérifier que nous retrouvons l'observation originale
        assert np.allclose(denormalized, obs, rtol=1e-5)
    
    def test_normalize_with_custom_stats(self):
        """Test la normalisation avec des stats personnalisées"""
        # Créer des stats personnalisées
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            stats = {
                'mean': [50.0] * 68,
                'var': [100.0] * 68
            }
            json.dump(stats, f)
            temp_path = f.name
        
        try:
            normalizer = ObservationNormalizer(fallback_stats_path=temp_path)
            
            # Créer une observation
            obs = np.ones(68) * 50
            
            # Normaliser
            normalized = normalizer.normalize(obs)
            
            # Vérifier que la normalisation est correcte
            # (50 - 50) / sqrt(100) = 0
            assert np.allclose(normalized, 0, atol=1e-5)
        
        finally:
            Path(temp_path).unlink()
    
    def test_get_stats(self):
        """Test la récupération des stats"""
        normalizer = ObservationNormalizer()
        
        stats = normalizer.get_stats()
        
        assert 'mean' in stats
        assert 'var' in stats
        assert 'is_loaded' in stats
        assert stats['is_loaded'] is True
        assert len(stats['mean']) == 68
        assert len(stats['var']) == 68


class TestDriftDetector:
    """Tests pour DriftDetector"""
    
    def test_drift_detector_initialization(self):
        """Test l'initialisation du détecteur"""
        detector = DriftDetector(window_size=100, threshold=2.0)
        
        assert detector.window_size == 100
        assert detector.threshold == 2.0
        assert len(detector.observations) == 0
    
    def test_add_observation(self):
        """Test l'ajout d'observations"""
        detector = DriftDetector(window_size=10)
        
        # Ajouter des observations
        for i in range(5):
            obs = np.random.randn(68)
            detector.add_observation(obs)
        
        assert len(detector.observations) == 5
    
    def test_window_size_limit(self):
        """Test que la fenêtre ne dépasse pas la taille limite"""
        detector = DriftDetector(window_size=10)
        
        # Ajouter plus d'observations que la taille de la fenêtre
        for i in range(20):
            obs = np.random.randn(68)
            detector.add_observation(obs)
        
        # Vérifier que la fenêtre ne dépasse pas la limite
        assert len(detector.observations) == 10
    
    def test_check_drift_not_enough_observations(self):
        """Test que check_drift retourne False si pas assez d'observations"""
        detector = DriftDetector(window_size=100)
        
        # Ajouter seulement 10 observations
        for i in range(10):
            obs = np.random.randn(68)
            detector.add_observation(obs)
        
        # Vérifier la dérive
        result = detector.check_drift(
            reference_mean=np.zeros(68),
            reference_var=np.ones(68)
        )
        
        assert result['drift_detected'] is False
        assert 'Pas assez' in result['reason']
    
    def test_check_drift_no_drift(self):
        """Test que check_drift détecte pas de dérive si les données sont similaires"""
        detector = DriftDetector(window_size=100, threshold=2.0)
        
        # Créer des observations similaires aux stats de référence
        reference_mean = np.zeros(68)
        reference_var = np.ones(68)
        
        # Ajouter des observations proches de la référence
        for i in range(100):
            obs = np.random.randn(68) * 0.5  # Petit écart
            detector.add_observation(obs)
        
        # Vérifier la dérive
        result = detector.check_drift(reference_mean, reference_var)
        
        # Devrait pas détecter de dérive (max_distance < threshold)
        assert result['max_distance'] < detector.threshold
    
    def test_check_drift_with_drift(self):
        """Test que check_drift détecte une dérive si les données sont très différentes"""
        detector = DriftDetector(window_size=100, threshold=2.0)
        
        # Créer des observations très différentes
        reference_mean = np.zeros(68)
        reference_var = np.ones(68)
        
        # Ajouter des observations très éloignées de la référence
        for i in range(100):
            obs = np.random.randn(68) * 10 + 50  # Grand écart
            detector.add_observation(obs)
        
        # Vérifier la dérive
        result = detector.check_drift(reference_mean, reference_var)
        
        # Devrait détecter une dérive
        assert result['drift_detected'] is True
        assert result['max_distance'] > detector.threshold
    
    def test_get_drift_summary(self):
        """Test la récupération du résumé des dérives"""
        detector = DriftDetector(window_size=100, threshold=2.0)
        
        # Ajouter des observations
        for i in range(100):
            obs = np.random.randn(68) * 10 + 50
            detector.add_observation(obs)
        
        # Vérifier la dérive
        detector.check_drift(
            reference_mean=np.zeros(68),
            reference_var=np.ones(68)
        )
        
        # Récupérer le résumé
        summary = detector.get_drift_summary()
        
        assert 'total_drifts' in summary
        assert 'drifts' in summary
        assert summary['total_drifts'] >= 0


class TestIntegration:
    """Tests d'intégration"""
    
    def test_normalizer_and_drift_detector_together(self):
        """Test l'utilisation du normaliseur et du détecteur ensemble"""
        normalizer = ObservationNormalizer()
        detector = DriftDetector(window_size=100, threshold=2.0)
        
        # Générer des observations
        for i in range(100):
            raw_obs = np.random.randn(68) * 10 + 50
            
            # Normaliser
            normalized_obs = normalizer.normalize(raw_obs)
            
            # Ajouter au détecteur
            detector.add_observation(raw_obs)
        
        # Vérifier la dérive
        result = detector.check_drift(normalizer.mean, normalizer.var)
        
        # Devrait pas détecter de dérive (les données sont normalisées)
        assert result['n_observations'] == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
