#!/usr/bin/env python3
"""
Script de profilage pour identifier les goulots d'étranglement de performance
dans le système ADAN Trading Bot.
"""

import cProfile
import pstats
import io
import time
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, '/home/morningstar/Documents/trading/ADAN/src')

# Mock dependencies for testing
from unittest.mock import MagicMock
sys.modules['stable_baselines3'] = MagicMock()
sys.modules['gymnasium'] = MagicMock()

try:
    from adan_trading_bot.data_processing.feature_engineer import FeatureEngineer
    from adan_trading_bot.data_processing.state_builder import StateBuilder
    from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader
    from adan_trading_bot.environment.reward_calculator import RewardCalculator
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")


class PerformanceProfiler:
    """Profileur de performance pour identifier les goulots d'étranglement"""
    
    def __init__(self):
        self.results = {}
        self.sample_data = self._generate_sample_data()
    
    def _generate_sample_data(self) -> Dict[str, pd.DataFrame]:
        """Génère des données d'exemple pour les tests de performance"""
        print("📊 Génération des données d'exemple...")
        
        # Génère des données OHLCV réalistes
        np.random.seed(42)
        n_samples = 10000
        
        data = {}
        for timeframe in ['5m', '1h', '4h']:
            # Prix de base avec tendance
            base_price = 50000
            price_trend = np.cumsum(np.random.randn(n_samples) * 0.001) * base_price * 0.01
            prices = base_price + price_trend
            
            # OHLC avec volatilité réaliste
            volatility = np.random.rand(n_samples) * 0.02 + 0.005
            
            high = prices * (1 + volatility * np.random.rand(n_samples))
            low = prices * (1 - volatility * np.random.rand(n_samples))
            open_prices = prices + np.random.randn(n_samples) * volatility * prices * 0.5
            close_prices = prices + np.random.randn(n_samples) * volatility * prices * 0.5
            
            # Volume réaliste
            volume = np.random.exponential(1000000, n_samples)
            
            # Timestamps
            timestamps = pd.date_range(
                start='2023-01-01', 
                periods=n_samples, 
                freq='5min' if timeframe == '5m' else ('1h' if timeframe == '1h' else '4h')
            )
            
            data[timeframe] = pd.DataFrame({
                'timestamp': timestamps,
                'open': open_prices,
                'high': high,
                'low': low,
                'close': close_prices,
                'volume': volume
            })
        
        print(f"✅ Données générées: {len(data)} timeframes, {n_samples} points chacun")
        return data
    
    def profile_feature_engineering(self) -> Dict[str, Any]:
        """Profile le calcul des indicateurs techniques"""
        print("\n🔧 Profilage Feature Engineering...")
        
        def run_feature_engineering():
            try:
                # Configuration des indicateurs
                indicators_config = [
                    {'name': 'sma', 'params': {'length': 20}},
                    {'name': 'ema', 'params': {'length': 12}},
                    {'name': 'rsi', 'params': {'length': 14}},
                    {'name': 'macd', 'params': {'fast': 12, 'slow': 26, 'signal': 9}},
                    {'name': 'bbands', 'params': {'length': 20, 'std': 2}},
                    {'name': 'atr', 'params': {'length': 14}},
                ]
                
                # Dummy data_config for FeatureEngineer
                data_config = {
                    'feature_engineering': {
                        'timeframes': ['5m', '1h', '4h'],
                        'indicators': {
                            'sma': {'length': 20},
                            'ema': {'length': 12},
                            'rsi': {'length': 14},
                            'macd': {'fast': 12, 'slow': 26, 'signal': 9},
                            'bbands': {'length': 20, 'std': 2},
                            'atr': {'length': 14},
                        },
                        'columns_to_normalize': ['open', 'high', 'low', 'close', 'volume', 'SMA', 'EMA', 'RSI', 'MACD', 'BB_Upper', 'BB_Lower', 'BB_Middle', 'ATR']
                    }
                }
                engineer = FeatureEngineer(data_config=data_config)
                
                # Create a merged DataFrame for FeatureEngineer
                merged_df_data = {}
                for tf, df in self.sample_data.items():
                    for col in df.columns:
                        if col != 'timestamp': # Exclude timestamp for merging, it's the index
                            merged_df_data[f'{tf}_{col}'] = df[col]
                
                merged_df = pd.DataFrame(merged_df_data)
                merged_df.index = self.sample_data['5m']['timestamp'] # Use one of the timestamps as index
                
                processed_df = engineer.process_data(merged_df, fit_scaler=True)
                
                return True
            except Exception as e:
                print(f"Error in feature engineering: {e}")
                return False
        
        # Profile the function
        profiler = cProfile.Profile()
        start_time = time.time()
        
        profiler.enable()
        success = run_feature_engineering()
        profiler.disable()
        
        end_time = time.time()
        
        # Analyze results
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions
        
        return {
            'success': success,
            'execution_time': end_time - start_time,
            'profile_stats': s.getvalue(),
            'component': 'feature_engineering'
        }
    
    def profile_state_building(self) -> Dict[str, Any]:
        """Profile la construction des observations"""
        print("\n🏗️ Profilage State Building...")
        
        def run_state_building():
            try:
                # Add some basic indicators to the data first
                enhanced_data = {}
                for tf, df in self.sample_data.items():
                    df_copy = df.copy()
                    # Add simple moving averages
                    df_copy[f'sma_20_{tf}'] = df_copy['close'].rolling(20).mean()
                    df_copy[f'ema_12_{tf}'] = df_copy['close'].ewm(span=12).mean()
                    df_copy[f'rsi_{tf}'] = self._calculate_rsi(df_copy['close'])
                    enhanced_data[tf] = df_copy
                
                # Dummy features_config for StateBuilder
                features_config = {
                    '5m': ['open', 'high', 'low', 'close', 'volume', 'SMA_20_5m', 'EMA_12_5m', 'RSI_5m'],
                    '1h': ['open', 'high', 'low', 'close', 'volume', 'SMA_20_1h', 'EMA_12_1h', 'RSI_1h'],
                    '4h': ['open', 'high', 'low', 'close', 'volume', 'SMA_20_4h', 'EMA_12_4h', 'RSI_4h'],
                }
                state_builder = StateBuilder(
                    features_config=features_config,
                    window_size=100,
                    adaptive_window=True
                )
                
                # Fit scalers before building observations
                state_builder.fit_scalers(enhanced_data)

                # --- Dimension Validation ---
                print("📏 Validating state dimension before profiling...")
                try:
                    state_builder.validate_dimension(enhanced_data)
                    print("   ✅ State dimension validation successful.")
                except ValueError as e:
                    print(f"   ❌ State dimension validation failed: {e}")
                    # We can choose to raise e to stop profiling or just log it
                    # For a profiler, it's better to log and continue if possible
                    return False # Stop this specific profile run
                # --- End Validation ---

                # Build observations for multiple time points
                for i in range(200, min(1000, len(enhanced_data['5m']) - 100)):
                    obs = state_builder.build_adaptive_observation(i, enhanced_data)
                
                return True
            except Exception as e:
                print(f"Error in state building: {e}")
                return False
        
        # Profile the function
        profiler = cProfile.Profile()
        start_time = time.time()
        
        profiler.enable()
        success = run_state_building()
        profiler.disable()
        
        end_time = time.time()
        
        # Analyze results
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(20)
        
        return {
            'success': success,
            'execution_time': end_time - start_time,
            'profile_stats': s.getvalue(),
            'component': 'state_building'
        }
    
    def profile_reward_calculation(self) -> Dict[str, Any]:
        """Profile le calcul des récompenses"""
        print("\n💰 Profilage Reward Calculation...")
        
        def run_reward_calculation():
            try:
                # Mock portfolio state
                portfolio_state = {
                    'total_value': 100000,
                    'positions': {
                        'BTCUSDT': {'size': 1.0, 'entry_price': 50000, 'unrealized_pnl': 1000}
                    },
                    'cash': 50000,
                    'margin_used': 25000
                }
                
                # Mock market data
                market_data = {
                    'BTCUSDT': {
                        'price': 51000,
                        'volume': 1000000,
                        'volatility': 0.02
                    }
                }
                
                # Dummy env_config for RewardCalculator
                env_config = {
                    'reward_shaping': {
                        'realized_pnl_multiplier': 1.0,
                        'unrealized_pnl_multiplier': 0.1,
                        'inaction_penalty': -0.0001,
                        'reward_clipping_range': [-5.0, 5.0],
                        'optimal_trade_bonus': 1.0,
                        'performance_threshold': 0.8
                    }
                }
                reward_calc = RewardCalculator(env_config=env_config)
                
                # Calculate rewards for multiple steps
                for i in range(1000):
                    # Simulate price changes
                    price_change = np.random.randn() * 0.001
                    market_data['BTCUSDT']['price'] *= (1 + price_change)
                    
                    # Update portfolio
                    portfolio_state['positions']['BTCUSDT']['unrealized_pnl'] = (
                        market_data['BTCUSDT']['price'] - 
                        portfolio_state['positions']['BTCUSDT']['entry_price']
                    ) * portfolio_state['positions']['BTCUSDT']['size']
                    
                    # Calculate reward
                    reward = reward_calc.calculate(
                        portfolio_metrics=portfolio_state,
                        trade_pnl=0.0, # Assuming no trade closed for this profiling
                        action=0, # Assuming hold action for profiling
                        chunk_id=1, # Dummy chunk ID
                        optimal_chunk_pnl=100.0, # Dummy optimal PnL
                        performance_ratio=0.8 # Dummy performance ratio
                    )
                
                return True
            except Exception as e:
                print(f"Error in reward calculation: {e}")
                return False
        
        # Profile the function
        profiler = cProfile.Profile()
        start_time = time.time()
        
        profiler.enable()
        success = run_reward_calculation()
        profiler.disable()
        
        end_time = time.time()
        
        # Analyze results
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(20)
        
        return {
            'success': success,
            'execution_time': end_time - start_time,
            'profile_stats': s.getvalue(),
            'component': 'reward_calculation'
        }
    
    def profile_numpy_operations(self) -> Dict[str, Any]:
        """Profile les opérations NumPy critiques"""
        print("\n🔢 Profilage Opérations NumPy...")
        
        def run_numpy_operations():
            # Simulate common operations in trading systems
            n_samples = 10000
            n_features = 100
            
            # Matrix operations
            data_matrix = np.random.randn(n_samples, n_features)
            
            # Rolling calculations (common bottleneck)
            for i in range(20, n_samples):
                window = data_matrix[i-20:i, :]
                rolling_mean = np.mean(window, axis=0)
                rolling_std = np.std(window, axis=0)
                normalized = (data_matrix[i, :] - rolling_mean) / (rolling_std + 1e-8)
            
            # Correlation calculations
            correlation_matrix = np.corrcoef(data_matrix.T)
            
            # Technical indicator calculations
            prices = np.random.randn(n_samples) * 0.01 + 1
            prices = np.cumprod(1 + prices) * 50000
            
            # RSI calculation (vectorized)
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            # Moving averages
            for window in [14, 20, 50]:
                sma = np.convolve(prices, np.ones(window)/window, mode='valid')
            
            return True
        
        # Profile the function
        profiler = cProfile.Profile()
        start_time = time.time()
        
        profiler.enable()
        success = run_numpy_operations()
        profiler.disable()
        
        end_time = time.time()
        
        # Analyze results
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(20)
        
        return {
            'success': success,
            'execution_time': end_time - start_time,
            'profile_stats': s.getvalue(),
            'component': 'numpy_operations'
        }
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calcul RSI simple pour les tests"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def run_comprehensive_profiling(self) -> Dict[str, Any]:
        """Exécute un profilage complet du système"""
        print("🚀 Profilage Complet des Performances ADAN")
        print("=" * 60)
        
        components = [
            ('feature_engineering', self.profile_feature_engineering),
            ('state_building', self.profile_state_building),
            ('reward_calculation', self.profile_reward_calculation),
            ('numpy_operations', self.profile_numpy_operations)
        ]
        
        results = {}
        total_time = 0
        
        for name, profiler_func in components:
            try:
                result = profiler_func()
                results[name] = result
                total_time += result['execution_time']
                
                if result['success']:
                    print(f"✅ {name}: {result['execution_time']:.3f}s")
                else:
                    print(f"❌ {name}: ÉCHEC")
            except Exception as e:
                print(f"❌ {name}: ERREUR - {e}")
                results[name] = {
                    'success': False,
                    'execution_time': 0,
                    'error': str(e),
                    'component': name
                }
        
        # Summary
        print(f"\n📊 Temps total de profilage: {total_time:.3f}s")
        
        # Identify bottlenecks
        bottlenecks = []
        for name, result in results.items():
            if result['success'] and result['execution_time'] > 1.0:
                bottlenecks.append((name, result['execution_time']))
        
        bottlenecks.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n🔍 Goulots d'étranglement identifiés:")
        for name, exec_time in bottlenecks:
            print(f"  ⚠️  {name}: {exec_time:.3f}s")
        
        if not bottlenecks:
            print("  ✅ Aucun goulot d'étranglement majeur détecté")
        
        # Save detailed results
        self._save_profiling_results(results)
        
        return results
    
    def _save_profiling_results(self, results: Dict[str, Any]):
        """Sauvegarde les résultats de profilage"""
        from datetime import datetime
        import json
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save summary
        summary = {
            'timestamp': timestamp,
            'total_components': len(results),
            'successful_components': sum(1 for r in results.values() if r.get('success', False)),
            'total_execution_time': sum(r.get('execution_time', 0) for r in results.values()),
            'components': {
                name: {
                    'success': result.get('success', False),
                    'execution_time': result.get('execution_time', 0),
                    'component': result.get('component', name)
                }
                for name, result in results.items()
            }
        }
        
        os.makedirs("logs", exist_ok=True)
        
        # Save JSON summary
        with open(f"logs/performance_profiling_{timestamp}.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed profile stats
        for name, result in results.items():
            if result.get('success') and 'profile_stats' in result:
                with open(f"logs/profile_{name}_{timestamp}.txt", 'w') as f:
                    f.write(f"Profile for {name}\n")
                    f.write("=" * 50 + "\n")
                    f.write(result['profile_stats'])
        
        print(f"📁 Résultats sauvegardés: logs/performance_profiling_{timestamp}.json")


def main():
    """Fonction principale"""
    profiler = PerformanceProfiler()
    results = profiler.run_comprehensive_profiling()
    
    return results


if __name__ == "__main__":
    main()