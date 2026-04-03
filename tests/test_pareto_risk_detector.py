#!/usr/bin/env python3
"""
Unit tests for Pareto Risk Detector
"""

import unittest
import numpy as np
import sys
sys.path.insert(0, 'src')

from adan_trading_bot.risk.pareto_risk_detector import ParetoRiskDetector


class TestParetoRiskDetector(unittest.TestCase):
    """Test suite for Pareto Risk Detector."""
    
    def test_initialization(self):
        """Test detector initializes correctly."""
        detector = ParetoRiskDetector(window_size=50, update_frequency=10)
        
        self.assertEqual(detector.window_size, 50)
        self.assertEqual(detector.update_frequency, 10)
        self.assertEqual(detector.current_regime, "NORMAL")
        self.assertEqual(detector.current_multiplier, 1.0)
        print("✓ Initialization correct")
    
    def test_normal_regime_detection(self):
        """Test that normal returns keep NORMAL regime."""
        print("\n" + "="*70)
        print("TEST 1: Normal Regime Detection")
        print("="*70)
        
        detector = ParetoRiskDetector(window_size=100, update_frequency=20)
        
        # Simulate normal returns (Gaussian, kurtosis ≈ 0)
        np.random.seed(42)
        for i in range(100):
            ret = np.random.normal(0, 0.01)  # 1% std
            detector.update(ret)
        
        info = detector.get_regime_info()
        
        print(f"Regime: {info['regime']}")
        print(f"Kurtosis: {info['kurtosis']:.2f}")
        print(f"Multiplier: {info['multiplier']:.2f}")
        
        self.assertEqual(info['regime'], "NORMAL")
        self.assertLess(abs(info['kurtosis']), 1.5, "Kurtosis should be near 0 for normal")
        self.assertEqual(info['multiplier'], 1.0)
        print("✓ PASS: Normal regime correctly detected")
    
    def test_high_vol_regime_detection(self):
        """Test HIGH_VOL regime detection with moderate fat tails."""
        print("\n" + "="*70)
        print("TEST 2: High Volatility Regime Detection")
        print("="*70)
        
        detector = ParetoRiskDetector(window_size=100, update_frequency=20)
        
        # Mix of normal (80%) + extreme (20%) to create fat tails
        np.random.seed(42)
        for i in range(100):
            if i % 5 == 0:  # 20% of time: extreme moves
                ret = np.random.normal(0, 0.05)  # 5% std (5x normal)
            else:
                ret = np.random.normal(0, 0.01)  # Normal
            detector.update(ret)
        
        info = detector.get_regime_info()
        
        print(f"Regime: {info['regime']}")
        print(f"Kurtosis: {info['kurtosis']:.2f}")
        print(f"Multiplier: {info['multiplier']:.2f}")
        
        # Should detect HIGH_VOL (kurtosis between 2 and 5)
        self.assertIn(info['regime'], ["HIGH_VOL", "EXTREME"])
        self.assertGreater(info['kurtosis'], 1.5)
        self.assertLess(info['multiplier'], 1.0, "Risk should be reduced")
        print("✓ PASS: High volatility regime detected")
    
    def test_extreme_regime_detection(self):
        """Test EXTREME regime detection with heavy fat tails."""
        print("\n" + "="*70)
        print("TEST 3: Extreme Regime Detection (Crisis)")
        print("="*70)
        
        detector = ParetoRiskDetector(window_size=100, update_frequency=10)
        
        # Simulate crisis: 70% normal + 30% huge spikes
        np.random.seed(42)
        for i in range(100):
            if i % 3 == 0:  # 33% extreme events
                ret = np.random.normal(0, 0.10)  # 10% std (10x normal)
            else:
                ret = np.random.normal(0, 0.01)
            detector.update(ret)
        
        info = detector.get_regime_info()
        
        print(f"Regime: {info['regime']}")
        print(f"Kurtosis: {info['kurtosis']:.2f}")
        print(f"Multiplier: {info['multiplier']:.2f}")
        
        # Should detect EXTREME (kurtosis > 5)
        self.assertEqual(info['regime'], "EXTREME")
        self.assertGreaterEqual(info['kurtosis'], 3.0)
        self.assertEqual(info['multiplier'], 0.5, "Risk should be halved")
        print("✓ PASS: Extreme regime detected, risk halved")
    
    def test_risk_adjustment(self):
        """Test that adjusted risk is correct."""
        print("\n" + "="*70)
        print("TEST 4: Risk Adjustment Calculation")
        print("="*70)
        
        detector = ParetoRiskDetector()
        
        base_risk = 0.04  # 4% (Micro Capital tier)
        
        # NORMAL regime
        detector.current_multiplier = 1.0
        adjusted = detector.get_risk_multiplier(base_risk)
        self.assertAlmostEqual(adjusted, 0.04)
        print(f"✓ NORMAL: 4.0% → {adjusted*100:.1f}%")
        
        # HIGH_VOL regime
        detector.current_multiplier = 0.75
        adjusted = detector.get_risk_multiplier(base_risk)
        self.assertAlmostEqual(adjusted, 0.03)
        print(f"✓ HIGH_VOL: 4.0% → {adjusted*100:.1f}% (-25%)")
        
        # EXTREME regime
        detector.current_multiplier = 0.5
        adjusted = detector.get_risk_multiplier(base_risk)
        self.assertAlmostEqual(adjusted, 0.02)
        print(f"✓ EXTREME: 4.0% → {adjusted*100:.1f}% (-50%)")
    
    def test_reset(self):
        """Test that reset clears state."""
        detector = ParetoRiskDetector()
        
        # Add some data
        for _ in range(50):
            detector.update(np.random.normal(0, 0.05))
        
        # Reset
        detector.reset()
        
        self.assertEqual(len(detector.returns_history), 0)
        self.assertEqual(detector.current_regime, "NORMAL")
        self.assertEqual(detector.current_multiplier, 1.0)
        print("✓ Reset clears all state")


if __name__ == "__main__":
    print("\n" + "🔒 " + "="*66)
    print("   PARETO RISK DETECTOR - UNIT TESTS")
    print("   " + "="*66)
    unittest.main(verbosity=2)
