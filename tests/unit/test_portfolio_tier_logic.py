#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests unitaires critiques pour la logique des paliers et CVaR.

Ce module teste les fonctionnalités essentielles de la Phase 1 :
- Logique des capital_tiers et transitions
- CVaR Position Sizing avec contraintes de paliers
- Normalisation des formules mathématiques
- Application des contraintes selon le capital disponible
"""

import pytest
import numpy as np
import logging
from unittest.mock import Mock, patch

logger = logging.getLogger(__name__)


class TestCapitalTiers:
    """Tests de la logique des paliers de capital."""

    def test_tier_detection_micro_capital(self, portfolio_manager_micro, helpers):
        """Test détection correcte du palier Micro Capital."""
        pm = portfolio_manager_micro

        # Test avec capital initial (20$)
        tier = pm.get_current_tier()

        assert tier['name'] == 'Micro Capital'
        assert tier['min_capital'] == 11.0
        assert tier['max_capital'] == 30.0
        assert tier['max_position_size_pct'] == 90
        assert tier['risk_per_trade_pct'] == 5.0
        assert tier['exposure_range'] == [75, 90]

        helpers.assert_tier_compliance(pm, 20.0)
        logger.info(f"✅ Palier Micro Capital correctement détecté: {tier['name']}")

    def test_tier_detection_medium_capital(self, portfolio_manager_medium, helpers):
        """Test détection correcte du palier Medium Capital."""
        pm = portfolio_manager_medium

        # Test avec capital initial (250$)
        tier = pm.get_current_tier()

        assert tier['name'] == 'Medium Capital'
        assert tier['min_capital'] == 101.0
        assert tier['max_capital'] == 500.0
        assert tier['max_position_size_pct'] == 60
        assert tier['risk_per_trade_pct'] == 3.0
        assert tier['exposure_range'] == [55, 70]

        helpers.assert_tier_compliance(pm, 250.0)
        logger.info(f"✅ Palier Medium Capital correctement détecté: {tier['name']}")

    def test_tier_transitions(self, config_data, tier_transition_scenarios):
        """Test des transitions entre paliers selon le capital."""
        from bot.src.adan_trading_bot.portfolio.portfolio_manager import PortfolioManager

        for scenario in tier_transition_scenarios:
            capital = scenario['capital']
            expected_tier = scenario['expected_tier']
            expected_max_pos = scenario['max_position_pct']

            # Créer PM avec capital spécifique
            config_data['portfolio']['initial_balance'] = capital
            pm = PortfolioManager(config_data)
            pm.initial_capital = capital
            pm.cash = capital

            # Vérifier détection du palier
            tier = pm.get_current_tier()
            assert tier['name'] == expected_tier, f"Capital {capital}: attendu {expected_tier}, obtenu {tier['name']}"
            assert tier['max_position_size_pct'] == expected_max_pos

            logger.info(f"✅ Transition capital {capital}$ -> {expected_tier} (max pos: {expected_max_pos}%)")

    def test_tier_edge_cases(self, config_data):
        """Test des cas limites de détection des paliers."""
        from bot.src.adan_trading_bot.portfolio.portfolio_manager import PortfolioManager

        # Test capital exactement à la limite
        test_cases = [
            (11.0, "Micro Capital"),    # Limite basse Micro
            (30.0, "Micro Capital"),    # Limite haute Micro
            (31.0, "Small Capital"),    # Limite basse Small
            (100.0, "Small Capital"),   # Limite haute Small
            (101.0, "Medium Capital"),  # Limite basse Medium
            (500.0, "Medium Capital"),  # Limite haute Medium
            (501.0, "Large Capital"),   # Limite basse Large
            (2000.0, "Large Capital"),  # Limite haute Large
            (2001.0, "Enterprise"),     # Limite basse Enterprise
            (10000.0, "Enterprise")     # Capital très élevé
        ]

        for capital, expected_tier in test_cases:
            config_data['portfolio']['initial_balance'] = capital
            pm = PortfolioManager(config_data)
            pm.initial_capital = capital
            pm.cash = capital

            tier = pm.get_current_tier()
            assert tier['name'] == expected_tier, f"Capital {capital}: attendu {expected_tier}, obtenu {tier['name']}"

            logger.info(f"✅ Cas limite capital {capital}$ -> {expected_tier}")


class TestCVaRPositionSizing:
    """Tests du système CVaR Position Sizing."""

    def test_cvar_basic_calculation(self, portfolio_manager_micro, helpers):
        """Test calcul CVaR de base avec contraintes de palier."""
        pm = portfolio_manager_micro

        # Test avec capital Micro (20$)
        position_size = pm.calculate_position_size_with_cvar(
            capital=20.0,
            asset="BTCUSDT",
            timeframe="1h",
            confidence_level=0.05,
            target_risk=0.02  # 2% de risque cible
        )

        # Vérifications de base
        assert isinstance(position_size, float)
        assert position_size > 0, "Position size must be positive"
        assert position_size >= 11.0, "Position must meet minimum order value"

        # Position ne doit pas dépasser les contraintes du palier Micro (90%)
        max_allowed = 20.0 * 0.90  # 18$
        assert position_size <= max_allowed, f"Position {position_size} exceeds Micro tier limit {max_allowed}"

        logger.info(f"✅ CVaR position sizing Micro: ${position_size:.2f} (max autorisé: ${max_allowed:.2f})")

    def test_cvar_with_different_risk_levels(self, portfolio_manager_medium):
        """Test CVaR avec différents niveaux de risque."""
        pm = portfolio_manager_medium
        capital = 250.0

        # Test avec risques croissants
        risk_levels = [0.01, 0.02, 0.03, 0.05]  # 1%, 2%, 3%, 5%
        positions = []

        for risk in risk_levels:
            position = pm.calculate_position_size_with_cvar(
                capital=capital,
                asset="BTCUSDT",
                timeframe="1h",
                confidence_level=0.05,
                target_risk=risk
            )
            positions.append(position)

            # Position doit respecter palier Medium (max 60%)
            max_allowed = capital * 0.60  # 150$
            assert position <= max_allowed, f"Risk {risk}: position {position} exceeds Medium tier limit {max_allowed}"

        # Positions doivent augmenter avec le risque (généralement)
        logger.info(f"✅ CVaR avec risques variables - Positions: {[f'{p:.1f}$' for p in positions]}")

    def test_cvar_extreme_scenarios(self, portfolio_manager_micro, sample_cvar_scenarios, helpers):
        """Test CVaR dans des scénarios de marché extrêmes."""
        pm = portfolio_manager_micro

        for scenario_name, scenario in sample_cvar_scenarios.items():
            logger.info(f"Test scénario: {scenario['description']}")

            # Simuler les rendements du scénario
            with patch.object(pm, '_simulate_returns') as mock_returns:
                mock_returns.return_value = scenario['returns']

                position = pm.calculate_position_size_with_cvar(
                    capital=20.0,
                    asset="BTCUSDT",
                    timeframe="1h",
                    confidence_level=0.05,
                    target_risk=0.02
                )

                # Vérifier les contraintes fondamentales du système
                position_pct = (position / 20.0) * 100
                tier = pm.get_current_tier()
                max_tier_pct = tier['max_position_size_pct']

                # Validations essentielles
                assert position > 0, f"Position doit être positive: {position}"
                assert position >= 11.0, f"Position ${position:.2f} < minimum Binance $11"
                assert position_pct <= max_tier_pct, f"Position {position_pct:.1f}% > limite palier {max_tier_pct}%"

                logger.info(f"✅ {scenario_name}: position ${position:.2f} ({position_pct:.1f}%) respecte contraintes (max: {max_tier_pct}%)")


class TestTierConstraintsNormalization:
    """Tests de la normalisation selon les contraintes des paliers."""

    def test_linear_normalization(self, portfolio_manager_micro):
        """Test normalisation linéaire (clipping)."""
        pm = portfolio_manager_micro

        # Test normalisation position size (palier Micro: max 90%)
        test_cases = [
            (50.0, 50.0),    # Dans les bornes
            (95.0, 90.0),    # Au-dessus, clippé à max
            (-5.0, 0.0),     # En-dessous, clippé à min
            (120.0, 90.0)    # Très au-dessus, clippé à max
        ]

        for input_val, expected in test_cases:
            result = pm.normalize_to_tier_bounds(input_val, 0, 90, 'linear')
            assert result == expected, f"Normalisation linéaire {input_val} -> {result}, attendu {expected}"

        logger.info("✅ Normalisation linéaire fonctionne correctement")

    def test_sigmoid_normalization(self, portfolio_manager_micro):
        """Test normalisation sigmoïde (smooth)."""
        pm = portfolio_manager_micro

        # Test normalisation sigmoïde pour position size (0-90%)
        test_cases = [
            (45.0, 40, 50),    # Valeur centrale, doit rester proche
            (120.0, 85, 90),   # Très au-dessus, ramené proche du max
            (-20.0, 0, 5),     # Très en-dessous, ramené proche du min
            (150.0, 88, 90)    # Extrêmement haut, très proche du max
        ]

        for input_val, min_expected, max_expected in test_cases:
            result = pm.normalize_to_tier_bounds(input_val, 0, 90, 'sigmoid')

            assert min_expected <= result <= max_expected, \
                f"Sigmoïde {input_val} -> {result:.2f}, attendu dans [{min_expected}, {max_expected}]"

        logger.info("✅ Normalisation sigmoïde fonctionne correctement")

    def test_tier_constraints_application(self, portfolio_manager_micro):
        """Test application complète des contraintes de palier."""
        pm = portfolio_manager_micro

        # Test avec valeurs hors bornes pour palier Micro
        constraints = pm.apply_tier_constraints(
            position_size_pct=120.0,    # Trop haut (max 90%)
            risk_pct=8.0,               # Trop haut (max 5.0%)
            exposure_avg=100.0,         # Trop haut (max 90 dans [75,90])
            concurrent_positions=3      # Trop haut (max 1)
        )

        # Vérifications
        assert constraints['position_size_pct'] <= 90, f"Position {constraints['position_size_pct']} dépasse 90%"
        assert constraints['risk_per_trade_pct'] <= 5.0, f"Risk {constraints['risk_per_trade_pct']} dépasse 5%"
        assert 75 <= constraints['exposure_normalized'] <= 90, f"Exposure {constraints['exposure_normalized']} hors [75,90]"
        assert constraints['concurrent_positions'] == 1, f"Positions concurrentes {constraints['concurrent_positions']} ≠ 1"

        logger.info(f"✅ Contraintes appliquées: {constraints}")

    def test_tier_constraints_different_tiers(self, config_data):
        """Test constraints pour différents paliers."""
        from bot.src.adan_trading_bot.portfolio.portfolio_manager import PortfolioManager

        test_scenarios = [
            (20.0, "Micro Capital", 90, 5.0, [75, 90], 1),
            (250.0, "Medium Capital", 60, 3.0, [55, 70], 3),
            (5000.0, "Enterprise", 30, 2.0, [35, 50], 5)
        ]

        for capital, tier_name, max_pos, max_risk, exp_range, max_concurrent in test_scenarios:
            config_data['portfolio']['initial_balance'] = capital
            pm = PortfolioManager(config_data)
            pm.initial_capital = capital
            pm.cash = capital

            # Vérifier le palier détecté
            tier = pm.get_current_tier()
            assert tier['name'] == tier_name

            # Test constraints avec valeurs excessives
            constraints = pm.apply_tier_constraints(
                position_size_pct=150.0,  # Toujours trop haut
                risk_pct=10.0,           # Toujours trop haut
                exposure_avg=120.0,      # Toujours trop haut
                concurrent_positions=10   # Toujours trop haut
            )

            # Vérifier que les contraintes sont respectées
            assert constraints['position_size_pct'] <= max_pos
            assert constraints['risk_per_trade_pct'] <= max_risk
            assert exp_range[0] <= constraints['exposure_normalized'] <= exp_range[1]
            assert constraints['concurrent_positions'] <= max_concurrent

            logger.info(f"✅ Contraintes {tier_name}: pos≤{max_pos}%, risk≤{max_risk}%, exp∈{exp_range}, concurrent≤{max_concurrent}")


class TestCVaRIntegrationWithTiers:
    """Tests d'intégration CVaR + logique des paliers."""

    def test_cvar_respects_tier_limits(self, portfolio_manager_micro, portfolio_manager_medium):
        """Test que CVaR respecte automatiquement les limites des paliers."""

        # Test Micro Capital (90% max)
        position_micro = portfolio_manager_micro.calculate_position_size_with_cvar(
            capital=20.0, asset="BTCUSDT", target_risk=0.05  # Risque élevé pour forcer contraintes
        )
        max_micro = 20.0 * 0.90  # 18$
        assert position_micro <= max_micro, f"CVaR Micro {position_micro} dépasse limite {max_micro}"

        # Test Medium Capital (60% max)
        position_medium = portfolio_manager_medium.calculate_position_size_with_cvar(
            capital=250.0, asset="BTCUSDT", target_risk=0.05  # Même risque élevé
        )
        max_medium = 250.0 * 0.60  # 150$
        assert position_medium <= max_medium, f"CVaR Medium {position_medium} dépasse limite {max_medium}"

        logger.info(f"✅ CVaR respecte limites: Micro ${position_micro:.2f}/{max_micro:.2f}, Medium ${position_medium:.2f}/{max_medium:.2f}")

    def test_cvar_tier_migration(self, config_data):
        """Test comportement CVaR lors du changement de palier."""
        from bot.src.adan_trading_bot.portfolio.portfolio_manager import PortfolioManager

        # Simuler croissance du capital et migration de paliers
        capitals = [25.0, 50.0, 150.0, 700.0, 3000.0]  # Migration through all tiers
        positions = []
        tiers = []

        for capital in capitals:
            config_data['portfolio']['initial_balance'] = capital
            pm = PortfolioManager(config_data)
            pm.initial_capital = capital
            pm.cash = capital

            # Mock data loader
            pm.data_loader = Mock()
            pm.data_loader.load_data.return_value = Mock()

            tier = pm.get_current_tier()
            tiers.append(tier['name'])

            # Calculer position avec même risque pour tous
            position = pm.calculate_position_size_with_cvar(
                capital=capital, asset="BTCUSDT", target_risk=0.02
            )
            positions.append(position)

            # Vérifier respect des contraintes du palier
            position_pct = (position / capital) * 100
            max_allowed_pct = tier['max_position_size_pct']
            assert position_pct <= max_allowed_pct, \
                f"Capital {capital}: position {position_pct:.1f}% dépasse limite {max_allowed_pct}%"

        logger.info("✅ Migration des paliers avec CVaR:")
        for i, (capital, tier, position) in enumerate(zip(capitals, tiers, positions)):
            pct = (position / capital) * 100
            logger.info(f"  Capital ${capital} -> {tier}: ${position:.2f} ({pct:.1f}%)")

    def test_cvar_mathematical_properties(self, portfolio_manager_medium, helpers):
        """Test des propriétés mathématiques du CVaR."""
        pm = portfolio_manager_medium
        capital = 250.0

        # Test 1: CVaR doit être cohérent (même inputs = même outputs)
        position1 = pm.calculate_position_size_with_cvar(capital, "BTCUSDT", target_risk=0.02)
        position2 = pm.calculate_position_size_with_cvar(capital, "BTCUSDT", target_risk=0.02)
        assert abs(position1 - position2) < 1.0, "CVaR should be deterministic with same inputs"

        # Test 2: Plus de risque cible = généralement plus de position (si pas limité par palier)
        low_risk_pos = pm.calculate_position_size_with_cvar(capital, "BTCUSDT", target_risk=0.01)
        high_risk_pos = pm.calculate_position_size_with_cvar(capital, "BTCUSDT", target_risk=0.02)

        # Si les deux ne sont pas limités par le palier, high_risk doit être >= low_risk
        max_tier_limit = capital * (pm.get_current_tier()['max_position_size_pct'] / 100)
        if high_risk_pos < max_tier_limit and low_risk_pos < max_tier_limit:
            assert high_risk_pos >= low_risk_pos, "Higher risk should generally allow larger positions"

        # Test 3: Propriétés CVaR attendues
        helpers.assert_cvar_properties(position1, capital, 0.05)  # Assume 5% CVaR typical

        logger.info(f"✅ Propriétés mathématiques CVaR validées: low_risk=${low_risk_pos:.2f}, high_risk=${high_risk_pos:.2f}")


class TestErrorHandling:
    """Tests de gestion d'erreur et cas limites."""

    def test_invalid_capital_amounts(self, config_data):
        """Test avec montants de capital invalides."""
        from bot.src.adan_trading_bot.portfolio.portfolio_manager import PortfolioManager

        # Test avec capital négatif (devrait fallback au tier minimum)
        config_data['portfolio']['initial_balance'] = -10.0
        pm = PortfolioManager(config_data)
        pm.initial_capital = -10.0
        pm.cash = -10.0

        # Devrait fallback sans crash
        tier = pm.get_current_tier()
        assert tier is not None
        logger.info(f"✅ Capital négatif géré: {tier['name']}")

    def test_cvar_with_insufficient_data(self, portfolio_manager_micro):
        """Test CVaR avec données insuffisantes."""
        pm = portfolio_manager_micro

        # Mock données insuffisantes
        with patch.object(pm, '_simulate_returns') as mock_returns:
            mock_returns.return_value = np.array([0.01, -0.02])  # Seulement 2 points

            position = pm.calculate_position_size_with_cvar(20.0, "BTCUSDT")

            # Devrait fallback au sizing simple sans crash
            assert position > 0
            assert position <= 20.0  # Ne doit pas dépasser le capital

        logger.info(f"✅ CVaR avec données insuffisantes géré: ${position:.2f}")

    def test_empty_capital_tiers_config(self, config_data):
        """Test avec configuration capital_tiers vide."""
        from bot.src.adan_trading_bot.portfolio.portfolio_manager import PortfolioManager

        # Vider la config des tiers
        config_data['capital_tiers'] = []

        pm = PortfolioManager(config_data)

        # Devrait créer un tier par défaut
        tier = pm.get_current_tier()
        assert tier is not None
        assert tier['name'] == 'default'

        logger.info("✅ Configuration tiers vide gérée avec tier par défaut")


if __name__ == "__main__":
    # Permet d'exécuter les tests directement
    pytest.main([__file__, "-v", "--tb=short"])
