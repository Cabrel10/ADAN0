#!/usr/bin/env python3
"""
Script de Paper Trading pour ADAN - Trading en temps réel avec agent RL pré-entraîné.

Ce script charge un agent PPO pré-entraîné et l'utilise pour prendre des décisions
de trading sur le Binance Testnet en temps réel.
"""

import os
import sys
import argparse
import time
import yaml
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Ajouter le répertoire src au PYTHONPATH
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent
sys.path.insert(0, str(project_root / "src"))

from adan_trading_bot.common.utils import get_logger, load_config
from adan_trading_bot.exchange_api.connector import get_exchange_client, validate_exchange_config
from adan_trading_bot.portfolio.portfolio_manager import PortfolioManager
from adan_trading_bot.trading.order_manager import OrderManager, Order, OrderType, OrderSide, OrderStatus
from adan_trading_bot.environment.state_builder import StateBuilder
from adan_trading_bot.agent.ppo_agent import load_agent

logger = get_logger(__name__)

class PaperTradingAgent:
    """Agent de paper trading en temps réel."""

    def __init__(self, config, model_path, initial_capital=15000.0):
        """
        Initialise l'agent de paper trading.

        Args:
            config: Configuration complète du système
            model_path: Chemin vers le modèle PPO pré-entraîné
            initial_capital: Capital initial pour le paper trading
        """
        self.config = config
        self.model_path = model_path
        self.initial_capital = initial_capital

        # Initialiser le PortfolioManager
        # Le config passé ici doit contenir toutes les sections nécessaires (trading_rules, risk_management, etc.)
        self.portfolio_manager = PortfolioManager(env_config=config)
        self.current_capital = self.portfolio_manager.total_capital # Utiliser le capital du portfolio manager
        self.positions = self.portfolio_manager.positions # Utiliser les positions du portfolio manager

        # Actifs à trader
        self.assets = config.get('data', {}).get('assets', [])
        logger.info(f"📊 Assets configured: {self.assets}")

        # Configuration du trading
        self.training_timeframe = config.get('data', {}).get('training_timeframe', '1m')
        self.data_source_type = config.get('data', {}).get('data_source_type', 'precomputed_features')

        # Initialiser les composants
        self._initialize_exchange()
        self._load_agent_and_scaler()
        self._initialize_order_manager()
        self._initialize_state_builder()

        # Historique des trades
        self.trade_history = []
        self.decision_history = []

        logger.info(f"🚀 PaperTradingAgent initialized - Capital: ${self.current_capital:.2f}")

    def _initialize_exchange(self):
        """Initialise la connexion à l'exchange."""
        try:
            # Valider la configuration
            if not validate_exchange_config(self.config):
                raise ValueError("Configuration d'exchange invalide")

            # Créer le client d'exchange
            self.exchange = get_exchange_client(self.config)
            logger.info(f"✅ Exchange connected: {self.exchange.id}")

            # Charger les marchés
            self.markets = self.exchange.load_markets()
            logger.info(f"📈 Markets loaded: {len(self.markets)} pairs")

        except Exception as e:
            logger.error(f"❌ Exchange initialization failed: {e}")
            logger.warning("🔧 Falling back to simulation mode")
            self.exchange = None
            self.markets = None

    def _load_agent_and_scaler(self):
        """Charge l'agent PPO et le scaler correspondant au training_timeframe."""
        try:
            # Charger l'agent
            logger.info(f"🤖 Loading agent from: {self.model_path}")
            self.agent = load_agent(self.model_path)
            logger.info("✅ Agent loaded successfully")

            # Charger le scaler approprié pour le training_timeframe
            self.scaler = self._load_appropriate_scaler()

        except Exception as e:
            logger.error(f"❌ Failed to load agent or scaler: {e}")
            raise

    def _load_appropriate_scaler(self):
        """Charge le scaler approprié selon le training_timeframe et data_source_type."""
        scalers_dir = project_root / "data" / "scalers_encoders"

        # Strategy 1: Scaler spécifique au timeframe
        scaler_candidates = [
            scalers_dir / f"scaler_{self.training_timeframe}.joblib",
            scalers_dir / f"scaler_{self.training_timeframe}_cpu.joblib",
            scalers_dir / f"unified_scaler_{self.training_timeframe}.joblib"
        ]

        # Strategy 2: Fallback scaler générique
        fallback_candidates = [
            scalers_dir / "scaler_cpu.joblib",
            scalers_dir / "scaler.joblib",
            scalers_dir / "unified_scaler.joblib"
        ]

        all_candidates = scaler_candidates + fallback_candidates

        for scaler_path in all_candidates:
            if scaler_path.exists():
                try:
                    scaler = joblib.load(scaler_path)
                    logger.info(f"✅ Scaler loaded from: {scaler_path}")

                    # Validation du scaler
                    if hasattr(scaler, 'transform'):
                        logger.info(f"📊 Scaler features: {getattr(scaler, 'n_features_in_', 'Unknown')}")
                        return scaler
                    else:
                        logger.warning(f"⚠️ Invalid scaler format in {scaler_path}")

                except Exception as e:
                    logger.warning(f"⚠️ Failed to load scaler from {scaler_path}: {e}")

        # Strategy 3: Créer un scaler à partir des données d'entraînement
        logger.warning("⚠️ No pre-saved scaler found, attempting to create from training data")
        return self._create_scaler_from_training_data()

    def _create_scaler_from_training_data(self):
        """Crée un scaler à partir des données d'entraînement du timeframe."""
        try:
            from sklearn.preprocessing import StandardScaler

            # Charger les données d'entraînement correspondantes
            processed_dir = project_root / "data" / "processed" / "unified"
            train_file = processed_dir / f"{self.training_timeframe}_train_merged.parquet"

            if not train_file.exists():
                # Fallback vers l'ancien format
                old_processed_dir = project_root / "data" / "processed" / "merged" / "unified"
                train_file = old_processed_dir / f"{self.training_timeframe}_train_merged.parquet"

            if not train_file.exists():
                logger.error(f"❌ No training data found for {self.training_timeframe}")
                return None

            logger.info(f"📊 Creating scaler from training data: {train_file}")
            train_df = pd.read_parquet(train_file)

            # Identifier les colonnes à normaliser (exclure OHLC)
            ohlc_patterns = ['open_', 'high_', 'low_', 'close_']
            cols_to_normalize = []

            for col in train_df.columns:
                should_normalize = True
                for pattern in ohlc_patterns:
                    if col.startswith(pattern):
                        should_normalize = False
                        break
                if should_normalize:
                    cols_to_normalize.append(col)

            if not cols_to_normalize:
                logger.warning("⚠️ No columns to normalize found")
                return None

            # Créer et ajuster le scaler
            scaler = StandardScaler()
            scaler.fit(train_df[cols_to_normalize])

            # Sauvegarder pour usage futur
            scaler_path = project_root / "data" / "scalers_encoders" / f"runtime_scaler_{self.training_timeframe}.joblib"
            scaler_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(scaler, scaler_path)

            logger.info(f"✅ Runtime scaler created and saved: {scaler_path}")
            logger.info(f"📊 Normalizable features: {len(cols_to_normalize)}")

            # Stocker les colonnes pour usage futur
            self.normalizable_columns = cols_to_normalize

            return scaler

        except Exception as e:
            logger.error(f"❌ Failed to create scaler from training data: {e}")
            return None

    def _initialize_order_manager(self):
        """Initialise le gestionnaire d'ordres."""
        self.order_manager = OrderManager(portfolio_manager=self.portfolio_manager)
        logger.info("✅ OrderManager initialized")

    def _initialize_state_builder(self):
        """Initialise le constructeur d'état."""
        # Pour le paper trading, nous utiliserons une version simplifiée
        # qui peut construire des états à partir de données live

        timeframe = self.config.get('data', {}).get('training_timeframe', '1m')
        logger.info(f"Initializing StateBuilder for timeframe: {timeframe}")
        base_features_per_asset = []

        if timeframe == '1m':
            base_features_per_asset = self.config.get('data', {}).get('base_market_features', ['open', 'high', 'low', 'close', 'volume'])
            logger.info(f"Using 'base_market_features' for 1m: {base_features_per_asset}")
        else:  # '1h' or '1d'
            base_features_per_asset.extend(['open', 'high', 'low', 'close', 'volume'])
            # indicators_by_timeframe should be at the root of data_config content
            indicators_cfg = self.config.get('data', {}).get('indicators_by_timeframe', {}).get(timeframe, [])

            logger.info(f"Found {len(indicators_cfg)} indicator specs for {timeframe} in config['data']['indicators_by_timeframe']")
            for ind_spec in indicators_cfg:
                name_to_use = ind_spec.get('output_col_name') or \
                              (ind_spec.get('output_col_names') and ind_spec['output_col_names'][0]) or \
                              ind_spec.get('alias') or \
                              ind_spec.get('name')
                if name_to_use:
                    base_features_per_asset.append(f"{name_to_use}_{timeframe}")
                else:
                    logger.warning(f"Could not determine name for indicator spec: {ind_spec}")

        seen = set()
        unique_base_features = [x for x in base_features_per_asset if not (x in seen or seen.add(x))]
        logger.info(f"Unique base features determined: {unique_base_features}")

        self.state_builder = StateBuilder(
            config=self.config, # Pass the main config dictionary
            assets=self.assets,
            scaler=self.scaler, # self.scaler should be loaded before this
            encoder=None,
            base_feature_names=unique_base_features,
            cnn_input_window_size=self.config.get('data', {}).get('cnn_input_window_size', 20)
        )
        logger.info(f"✅ StateBuilder initialized with {len(unique_base_features)} base features per asset for timeframe {timeframe}.")

    def get_live_market_data(self, symbol_ccxt, limit=50):
        """
        Récupère les données de marché en temps réel.

        Args:
            symbol_ccxt: Symbole au format CCXT (ex: "BTC/USDT")
            limit: Nombre de bougies à récupérer

        Returns:
            pd.DataFrame: Données OHLCV ou None si erreur
        """
        try:
            if not self.exchange:
                logger.warning("❌ No exchange connection - cannot fetch live data")
                return None

            # Récupérer les données OHLCV
            ohlcv = self.exchange.fetch_ohlcv(
                symbol_ccxt,
                timeframe='1m',  # Toujours récupérer en 1m pour flexibilité
                limit=limit
            )

            if not ohlcv:
                logger.warning(f"❌ No OHLCV data received for {symbol_ccxt}")
                return None

            # Convertir en DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            logger.debug(f"📊 Fetched {len(df)} candles for {symbol_ccxt}")
            return df

        except Exception as e:
            logger.error(f"❌ Error fetching market data for {symbol_ccxt}: {e}")
            return None

    def process_market_data_for_agent(self, market_data_dict):
        """
        Traite les données de marché pour créer une observation pour l'agent selon le training_timeframe.

        Args:
            market_data_dict: Dict avec asset_id -> DataFrame OHLCV 1m

        Returns:
            np.array: Observation normalisée pour l'agent ou None si erreur
        """
        try:
            logger.info(f"🔄 Processing market data for {self.training_timeframe} timeframe")

            # Étape 1: Créer les données au timeframe approprié
            processed_data = self._prepare_timeframe_data(market_data_dict)
            if processed_data is None:
                logger.error("❌ Failed to prepare timeframe data")
                return None

            # Étape 2: Calculer les features selon le timeframe
            features_data = self._calculate_features_for_timeframe(processed_data)
            if features_data is None:
                logger.error("❌ Failed to calculate features")
                return None

            # Étape 3: Normaliser les features
            normalized_data = self._normalize_features(features_data)
            if normalized_data is None:
                logger.error("❌ Failed to normalize features")
                return None

            # Étape 4: Combiner les features des actifs en un seul DataFrame
            window_size = self.config.get('data', {}).get('cnn_input_window_size', 20)
            all_asset_dfs_for_window = []

            for asset_id in self.assets: # Iterate in defined order
                if asset_id not in normalized_data:
                    logger.warning(f"No normalized data for {asset_id} to combine. Skipping this asset for observation.")
                    # How to handle missing asset? StateBuilder expects all.
                    # For now, let's assume StateBuilder will handle missing columns if padded correctly later.
                    # Or, this should ideally not happen if data fetching is robust.
                    continue

                asset_df = normalized_data[asset_id]
                if len(asset_df) < window_size:
                    logger.warning(f"Insufficient data for {asset_id} in normalized_data ({len(asset_df)} < {window_size}). Padding with last known value.")
                    if asset_df.empty: # No data at all
                        # This is problematic. StateBuilder needs some data.
                        # Create a dummy df with NaNs for this asset's features, StateBuilder might handle NaNs.
                        # This part needs careful consideration on how StateBuilder handles missing asset columns.
                        # For now, log error and skip, which will likely cause downstream issues.
                        logger.error(f"CRITICAL: No data for {asset_id} to build observation. Observation will be incomplete.")
                        continue # Or raise error

                    # Padding: repeat the last row
                    last_row = asset_df.iloc[[-1]] # Keep as DataFrame
                    padding_needed = window_size - len(asset_df)
                    padding_df = pd.concat([last_row] * padding_needed, ignore_index=False) # keep index of last_row for alignment
                    asset_df_window = pd.concat([asset_df, padding_df])
                    # Ensure the index is consistent after padding if it was messed up
                    # This part is tricky; ideally, data fetching ensures enough points.
                    # For now, we assume simple concat is okay if indices are simple.
                else:
                    asset_df_window = asset_df.tail(window_size)

                # Rename columns to feature_asset_id
                renamed_df = asset_df_window.rename(columns=lambda col: f"{col}_{asset_id}")
                all_asset_dfs_for_window.append(renamed_df)

            if not all_asset_dfs_for_window:
                logger.error("❌ No asset dataframes to combine for observation.")
                return None

            # Merge all asset dataframes horizontally
            # Assuming they share the same DatetimeIndex from resampling/fetching
            combined_features_df = pd.concat(all_asset_dfs_for_window, axis=1)

            # Ensure the combined_features_df has exactly window_size rows.
            # This can be tricky if different assets had slightly misaligned timestamps at the resampling stage.
            # For now, we trust the resampling and tail(window_size) per asset.
            # If issues arise, a reindex to a common DatetimeIndex might be needed before concat.
            if len(combined_features_df) != window_size:
                 logger.warning(f"Combined features df has {len(combined_features_df)} rows, expected {window_size}. Using tail.")
                 combined_features_df = combined_features_df.tail(window_size)
                 if len(combined_features_df) < window_size:
                     logger.error(f"Still not enough rows ({len(combined_features_df)}) after tail. Observation will be flawed.")
                     # Potentially pad again here if necessary, or let StateBuilder handle it.


            # Étape 5: Valider les colonnes avant de construire l'observation
            if self.state_builder is None or self.state_builder.global_scaler_feature_order is None:
                if self.state_builder and not self.state_builder.global_scaler_feature_order and not self.state_builder.global_scaler:
                    # This is acceptable if no scaler is loaded, means we use canonical order and no scaling by SB
                    logger.info("PaperTradingAgent: StateBuilder has no global_scaler or specific feature order. Proceeding with canonical feature order and no scaling by StateBuilder.")
                else:
                    logger.error("PaperTradingAgent: StateBuilder or its global_scaler_feature_order is not initialized. Cannot validate/build observation.")
                    return None # Cannot proceed if scaler was expected but order is missing

            expected_feature_columns = None
            if self.state_builder.global_scaler_feature_order:
                expected_feature_columns = self.state_builder.global_scaler_feature_order
            elif hasattr(self.state_builder, 'canonical_fallback_feature_order'): # Check if fallback exists
                expected_feature_columns = self.state_builder.canonical_fallback_feature_order

            if not expected_feature_columns:
                 logger.error("PaperTradingAgent: Could not determine expected feature columns for validation.")
                 return None

            missing_columns = [col for col in expected_feature_columns if col not in combined_features_df.columns]

            if missing_columns:
                logger.error(f"PaperTradingAgent: Critical features missing from combined_features_df for StateBuilder. Missing ({len(missing_columns)}): {missing_columns}")
                logger.error(f"Available columns in combined_features_df (first {min(10, len(combined_features_df.columns))}): {combined_features_df.columns.tolist()[:min(10, len(combined_features_df.columns))]}")
                return None # Stop if critical features for the model are missing

            # All expected columns are present, proceed to build observation
            observation_dict = self._build_final_observation(combined_features_df)

            return observation_dict # This is now a dictionary

        except Exception as e:
            logger.error(f"❌ Error processing market data: {e}", exc_info=True)
            return None

    def _prepare_timeframe_data(self, market_data_dict):
        """Prépare les données au timeframe d'entraînement."""
        try:
            timeframe_data = {}

            for asset_id in self.assets:
                if asset_id not in market_data_dict:
                    logger.warning(f"⚠️ No market data for {asset_id}")
                    continue

                df_1m = market_data_dict[asset_id]

                if self.training_timeframe == "1m":
                    # Utiliser directement les données 1m
                    timeframe_data[asset_id] = df_1m.copy()

                elif self.training_timeframe in ["1h", "1d"]:
                    # Ré-échantillonner vers le timeframe cible
                    freq = '1H' if self.training_timeframe == '1h' else '1D'

                    # Assurer que l'index est datetime
                    if not isinstance(df_1m.index, pd.DatetimeIndex):
                        df_1m.index = pd.to_datetime(df_1m.index)

                    # Ré-échantillonner OHLCV
                    resampled = df_1m.resample(freq).agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).dropna()

                    if len(resampled) < 2:
                        logger.warning(f"⚠️ Insufficient resampled data for {asset_id}: {len(resampled)}")
                        continue

                    timeframe_data[asset_id] = resampled
                    logger.debug(f"📊 {asset_id}: {len(df_1m)} 1m bars → {len(resampled)} {self.training_timeframe} bars")

                else:
                    logger.error(f"❌ Unsupported training_timeframe: {self.training_timeframe}")
                    return None

            return timeframe_data if timeframe_data else None

        except Exception as e:
            logger.error(f"❌ Error preparing timeframe data: {e}")
            return None

    def _calculate_features_for_timeframe(self, timeframe_data):
        """Calcule les features selon le timeframe et data_source_type."""
        try:
            features_data = {}

            for asset_id, df in timeframe_data.items():
                if self.training_timeframe == "1m" and self.data_source_type in ["precomputed_features_1m_resample", "precomputed_features"]:
                    # Mode 1m avec features pré-calculées : simuler avec OHLCV + quelques indicateurs simples
                    logger.warning(f"⚠️ {asset_id}: Simulating precomputed features with basic indicators")
                    features_df = self._calculate_basic_indicators(df)

                elif self.training_timeframe in ["1h", "1d"]:
                    # Mode 1h/1d : calculer les indicateurs selon la configuration
                    indicators_config = self.config.get('data', {}).get('indicators_by_timeframe', {}).get(self.training_timeframe, [])
                    features_df = self._calculate_timeframe_indicators(df, indicators_config)

                else:
                    # Fallback : OHLCV uniquement
                    features_df = df.copy()

                features_data[asset_id] = features_df
                logger.debug(f"📊 {asset_id}: {features_df.shape[1]} features calculated")

            return features_data if features_data else None

        except Exception as e:
            logger.error(f"❌ Error calculating features: {e}")
            return None

    def _calculate_basic_indicators(self, df):
        """Calcule des indicateurs de base pour simuler les features pré-calculées."""
        try:
            import pandas_ta as ta

            features_df = df.copy()

            # Indicateurs simples calculables à la volée
            features_df['SMA_short'] = ta.sma(df['close'], length=10)
            features_df['SMA_long'] = ta.sma(df['close'], length=50)
            features_df['EMA_short'] = ta.ema(df['close'], length=12)
            features_df['EMA_long'] = ta.ema(df['close'], length=26)
            features_df['RSI'] = ta.rsi(df['close'], length=14)

            # MACD
            macd_data = ta.macd(df['close'], fast=12, slow=26, signal=9)
            if macd_data is not None and len(macd_data.columns) >= 3:
                features_df['MACD'] = macd_data.iloc[:, 0]
                features_df['MACDs'] = macd_data.iloc[:, 1]
                features_df['MACDh'] = macd_data.iloc[:, 2]

            # Bollinger Bands
            bb_data = ta.bbands(df['close'], length=20, std=2)
            if bb_data is not None and len(bb_data.columns) >= 3:
                features_df['BBL'] = bb_data.iloc[:, 0]
                features_df['BBM'] = bb_data.iloc[:, 1]
                features_df['BBU'] = bb_data.iloc[:, 2]

            # ATR
            features_df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)

            # Nettoyer les NaN
            features_df = features_df.fillna(method='ffill').fillna(method='bfill')

            logger.debug(f"📊 Basic indicators calculated: {features_df.shape[1]} features")
            return features_df

        except Exception as e:
            logger.error(f"❌ Error calculating basic indicators: {e}")
            return df  # Fallback to OHLCV only

    def _calculate_timeframe_indicators(self, df, indicators_config):
        """Calcule les indicateurs selon la configuration pour 1h/1d."""
        try:
            if not indicators_config:
                logger.warning(f"⚠️ No indicators configured for {self.training_timeframe}")
                return df

            # Import de la fonction add_technical_indicators si disponible
            try:
                from adan_trading_bot.data_processing.feature_engineer import add_technical_indicators
                features_df, _ = add_technical_indicators(df, indicators_config, self.training_timeframe) # add_technical_indicators returns df, added_features
                logger.debug(f"📊 Timeframe indicators calculated: {features_df.shape[1]} features")
                return features_df
            except ImportError:
                logger.warning("⚠️ add_technical_indicators not available, using basic indicators")
                return self._calculate_basic_indicators(df)

        except Exception as e:
            logger.error(f"❌ Error calculating timeframe indicators: {e}")
            return df

    def _normalize_features(self, features_data):
        """Normalise les features avec le scaler approprié."""
        try:
            if not self.scaler:
                logger.warning("⚠️ No scaler available, using raw features")
                return features_data

            normalized_data = {}

            for asset_id, df in features_data.items():
                # Identifier les colonnes à normaliser
                normalizable_cols = getattr(self, 'normalizable_columns', None)

                if normalizable_cols is None:
                    # Auto-detect normalizable columns (exclude OHLC)
                    ohlc_patterns = ['open', 'high', 'low', 'close']
                    normalizable_cols = [col for col in df.columns
                                       if not any(col.startswith(pattern) for pattern in ohlc_patterns)]

                # Filtrer les colonnes qui existent réellement
                available_cols = [col for col in normalizable_cols if col in df.columns]

                if not available_cols:
                    logger.warning(f"⚠️ {asset_id}: No normalizable columns found")
                    normalized_data[asset_id] = df
                    continue

                # Normaliser
                df_normalized = df.copy()
                try:
                    df_normalized[available_cols] = self.scaler.transform(df[available_cols])
                    logger.debug(f"📊 {asset_id}: {len(available_cols)} features normalized")
                except Exception as e:
                    logger.warning(f"⚠️ {asset_id}: Normalization failed: {e}")

                normalized_data[asset_id] = df_normalized

            return normalized_data

        except Exception as e:
            logger.error(f"❌ Error normalizing features: {e}")
            return features_data  # Return unnormalized data as fallback

    def _build_final_observation(self, combined_features_df):
        """
        Construit l'observation finale pour l'agent en utilisant StateBuilder.

        Args:
            combined_features_df: DataFrame contenant les features de tous les actifs pour la fenêtre requise,
                                  avec des colonnes nommées (ex: 'rsi_1h_ADAUSDT', 'open_BTCUSDT').
                                  L'index doit être un DatetimeIndex.
        Returns:
            dict: Dictionnaire d'observation attendu par l'agent (image_features, vector_features).
        """
        try:
            if self.state_builder is None:
                logger.error("❌ StateBuilder not initialized. Cannot build observation.")
                return None

            if not isinstance(combined_features_df.index, pd.DatetimeIndex):
                logger.warning("Converting index of combined_features_df to DatetimeIndex.")
                try:
                    combined_features_df.index = pd.to_datetime(combined_features_df.index)
                except Exception as e_conv:
                    logger.error(f"Failed to convert index to DatetimeIndex: {e_conv}")
                    return None

            # StateBuilder expects market_data_window, capital, positions
            # image_shape is handled internally by StateBuilder based on its config
            observation_dict = self.state_builder.build_observation(
                market_data_window=combined_features_df, # This df should have the window_size rows
                capital=self.current_capital,
                positions=self.positions,
                apply_scaling=True # Explicitly set to True for live/paper trading data
            )

            # Validation de l'observation (StateBuilder devrait déjà gérer ça, mais double-check)
            if isinstance(observation_dict, dict):
                for key, value in observation_dict.items():
                    if value is None: # Statebuilder can return None for a feature type if error.
                        logger.error(f"Observation component '{key}' is None.")
                        return None
                    if np.any(np.isnan(value)):
                        logger.warning(f"⚠️ NaN values detected in observation component '{key}', replacing with 0")
                        observation_dict[key] = np.nan_to_num(value, nan=0.0)
                    if np.any(np.isinf(value)):
                        logger.warning(f"⚠️ Infinite values detected in observation component '{key}', clipping")
                        observation_dict[key] = np.clip(value, -1e6, 1e6)
            else: # Should be a dict
                 logger.error(f"❌ Observation_dict is not a dict: {type(observation_dict)}")
                 return None

            logger.debug(f"📊 Final observation dict built. Keys: {list(observation_dict.keys())}")
            if "image_features" in observation_dict:
                 logger.debug(f"   Image features shape: {observation_dict['image_features'].shape}")
            if "vector_features" in observation_dict:
                 logger.debug(f"   Vector features shape: {observation_dict['vector_features'].shape}")

            return observation_dict

        except Exception as e:
            logger.error(f"❌ Error building final observation with StateBuilder: {e}", exc_info=True)
            return None

    def convert_asset_to_ccxt_symbol(self, asset_id):
        """Convertit un asset_id en symbole CCXT."""
        if asset_id.endswith('USDT'):
            base = asset_id[:-4]
            return f"{base}/USDT"
        elif asset_id.endswith('BTC'):
            base = asset_id[:-3]
            return f"{base}/BTC"
        else:
            logger.warning(f"⚠️ Unknown quote currency for {asset_id}")
            return None

    def translate_action(self, action):
        """
        Traduit l'action numérique en asset_id et type de trade.

        Args:
            action: Action numérique de l'agent

        Returns:
            tuple: (asset_id, trade_type) ou (None, "HOLD")
        """
        try:
            if action == 0:  # HOLD
                return None, "HOLD"

            num_assets = len(self.assets)

            # Actions BUY: 1 à num_assets
            if 1 <= action <= num_assets:
                asset_index = action - 1
                asset_id = self.assets[asset_index]
                return asset_id, "BUY"

            # Actions SELL: num_assets+1 à 2*num_assets
            elif num_assets + 1 <= action <= 2 * num_assets:
                asset_index = action - num_assets - 1
                asset_id = self.assets[asset_index]
                return asset_id, "SELL"

            else:
                logger.warning(f"⚠️ Unknown action: {action}")
                return None, "HOLD"

        except Exception as e:
            logger.error(f"❌ Error translating action {action}: {e}")
            return None, "HOLD"

    def execute_trading_decision(self, asset_id, trade_type, current_prices):
        """
        Exécute une décision de trading.

        Args:
            asset_id: ID de l'actif à trader
            trade_type: Type de trade ("BUY" ou "SELL")
            current_prices: Dict des prix actuels

        Returns:
            dict: Résultat de l'exécution
        """
        try:
            if trade_type == "HOLD":
                return {"status": "HOLD", "message": "No action taken"}

            if asset_id not in current_prices:
                logger.error(f"❌ No current price for {asset_id}")
                return {"status": "ERROR", "message": f"No price for {asset_id}"}

            current_price = current_prices[asset_id]

            if trade_type == "BUY":
                # Calculer la taille de la position en utilisant le PortfolioManager
                # Pour le paper trading, nous allons utiliser une confiance de 1.0 pour simplifier
                size = self.portfolio_manager.calculate_position_size(action_type="buy", asset=asset_id, current_price=current_price, confidence=1.0)

                if self.portfolio_manager.validate_position(asset_id, size, current_price):
                    logger.info(f"🔄 Executing BUY {asset_id}: {size:.6f} units at ${current_price:.6f}")
                    self.portfolio_manager.open_position(asset_id, current_price, size)
                    return {"status": "SUCCESS", "message": f"BUY {asset_id}"}
                else:
                    logger.warning(f"⚠️ BUY order for {asset_id} invalid. Skipping.")
                    return {"status": "INVALID_ORDER", "message": f"BUY order for {asset_id} invalid"}

            elif trade_type == "SELL":
                if self.portfolio_manager.positions[asset_id].is_open:
                    size = self.portfolio_manager.positions[asset_id].size
                    logger.info(f"🔄 Executing SELL {asset_id}: {size:.6f} units at ${current_price:.6f}")
                    self.portfolio_manager.close_position(asset_id, current_price)
                    return {"status": "SUCCESS", "message": f"SELL {asset_id}"}
                else:
                    logger.warning(f"⚠️ Cannot SELL {asset_id}: No open position.")
                    return {"status": "NO_POSITION", "message": f"No position to sell for {asset_id}"}

        except Exception as e:
            logger.error(f"❌ Error executing {trade_type} for {asset_id}: {e}")
            return {"status": "ERROR", "message": str(e)}

    def run_trading_loop(self, max_iterations=100, sleep_seconds=60):
        """
        Exécute la boucle principale de trading.

        Args:
            max_iterations: Nombre maximum d'itérations
            sleep_seconds: Temps d'attente entre chaque décision
        """
        logger.info(f"🚀 Starting paper trading loop - Max iterations: {max_iterations}")
        logger.info(f"⏰ Decision frequency: Every {sleep_seconds} seconds")

        iteration = 0

        try:
            while iteration < max_iterations:
                iteration += 1
                logger.info(f"\n{'='*60}")
                logger.info(f"🔄 ITERATION {iteration}/{max_iterations}")
                logger.info(f"💰 Current Capital: ${self.portfolio_manager.total_capital:.2f}")
                logger.info(f"📊 Open Positions: {len([p for p in self.portfolio_manager.positions.values() if p.is_open])}")

                # 1. Récupérer les données de marché
                market_data_dict = {}
                current_prices = {}

                for asset_id in self.assets:
                    symbol_ccxt = self.convert_asset_to_ccxt_symbol(asset_id)
                    if symbol_ccxt:
                        market_data = self.get_live_market_data(symbol_ccxt, limit=30)
                        if market_data is not None and not market_data.empty:
                            market_data_dict[asset_id] = market_data
                            current_prices[asset_id] = float(market_data['close'].iloc[-1])
                            logger.debug(f"📈 {asset_id}: ${current_prices[asset_id]:.6f}")

                if not market_data_dict:
                    logger.warning("⚠️ No market data available - skipping iteration")
                    time.sleep(sleep_seconds)
                    continue

                # Mettre à jour le PortfolioManager avec les prix actuels
                self.portfolio_manager.update_market_price(current_prices)

                # 2. Construire l'observation pour l'agent
                observation = self.process_market_data_for_agent(market_data_dict)
                if observation is None:
                    logger.warning("⚠️ Failed to build observation - skipping iteration")
                    time.sleep(sleep_seconds)
                    continue

                # 3. Obtenir la décision de l'agent
                try:
                    action, _ = self.agent.predict(observation, deterministic=True)
                    asset_id, trade_type = self.translate_action(action)

                    logger.info(f"🤖 Agent Decision: Action={action} -> {trade_type} {asset_id or 'N/A'}")

                except Exception as e:
                    logger.error(f"❌ Agent prediction failed: {e}")
                    asset_id, trade_type = None, "HOLD"

                # 4. Exécuter la décision
                execution_result = self.execute_trading_decision(asset_id, trade_type, current_prices)

                # 5. Enregistrer l'historique
                decision_record = {
                    "timestamp": datetime.now(),
                    "iteration": iteration,
                    "action": action if 'action' in locals() else None,
                    "asset_id": asset_id,
                    "trade_type": trade_type,
                    "execution_result": execution_result,
                    "capital_before": self.portfolio_manager.total_capital, # Utiliser le capital du portfolio manager
                    "positions_count": len([p for p in self.portfolio_manager.positions.values() if p.is_open]) # Compter les positions ouvertes
                }

                self.decision_history.append(decision_record)

                # 6. Afficher le résumé
                pnl = self.portfolio_manager.total_capital - self.initial_capital
                pnl_pct = (pnl / self.initial_capital) * 100

                logger.info(f"💼 Portfolio Summary:")
                logger.info(f"   💰 Capital: ${self.portfolio_manager.total_capital:.2f}")
                logger.info(f"   📈 PnL: ${pnl:.2f} ({pnl_pct:+.2f}%)")
                logger.info(f"   🎯 Positions: {list(self.portfolio_manager.positions.keys())}")

                # 7. Attendre avant la prochaine décision
                if iteration < max_iterations:
                    logger.info(f"⏰ Sleeping {sleep_seconds} seconds...")
                    time.sleep(sleep_seconds)

        except KeyboardInterrupt:
            logger.info("\n🛑 Trading loop interrupted by user")
        except Exception as e:
            logger.error(f"❌ Trading loop error: {e}")
        finally:
            self.save_trading_summary()

    def save_trading_summary(self):
        """Sauvegarde un résumé de la session de trading."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_file = project_root / f"paper_trading_summary_{timestamp}.json"

            # Calculer les statistiques finales
            final_pnl = self.portfolio_manager.total_capital - self.initial_capital
            final_pnl_pct = (final_pnl / self.initial_capital) * 100

            summary = {
                "session_info": {
                    "start_time": datetime.now().isoformat(),
                    "model_path": str(self.model_path),
                    "initial_capital": self.initial_capital,
                    "final_capital": self.portfolio_manager.total_capital,
                    "total_pnl": final_pnl,
                    "pnl_percentage": final_pnl_pct,
                    "total_decisions": len(self.decision_history),
                    "assets_traded": self.assets
                },
                "final_positions": {asset: {attr: getattr(position, attr) for attr in ['is_open', 'entry_price', 'size', 'stop_loss_pct', 'take_profit_pct']} for asset, position in self.portfolio_manager.positions.items()},
                "decision_history": [
                    {k: v if k != "timestamp" else v.isoformat() for k, v in record.items()}
                    for record in self.decision_history
                ]
            }

            import json
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)

            logger.info(f"📊 Trading summary saved: {summary_file}")
            logger.info(f"🎯 Final Results:")
            logger.info(f"   💰 Capital: ${self.initial_capital:.2f} -> ${self.portfolio_manager.total_capital:.2f}")
            logger.info(f"   📈 PnL: ${final_pnl:.2f} ({final_pnl_pct:+.2f}%)")
            logger.info(f"   🔄 Decisions: {len(self.decision_history)}")

        except Exception as e:
            logger.error(f"❌ Failed to save trading summary: {e}")


def main():
    """Fonction principale du script de paper trading."""
    parser = argparse.ArgumentParser(description="ADAN Paper Trading Agent")
    parser.add_argument("--exec_profile", type=str, default="cpu",
                       choices=["cpu", "gpu"], help="Profil d'exécution")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Chemin vers le modèle PPO pré-entraîné")
    parser.add_argument("--initial_capital", type=float, default=15000.0,
                       help="Capital initial pour le paper trading")
    parser.add_argument("--max_iterations", type=int, default=100,
                       help="Nombre maximum d'itérations de trading")
    parser.add_argument("--sleep_seconds", type=int, default=60,
                       help="Temps d'attente entre chaque décision (secondes)")
    parser.add_argument(
        '--training_timeframe',
        type=str,
        default='1m', # Defaulting to '1m'
        choices=['1m', '1h', '1d'],
        help="Operational timeframe, should match the model's training. (default: 1m)"
    )

    args = parser.parse_args()

    try:
        # Charger la configuration
        logger.info(f"🔄 Loading configurations for profile: {args.exec_profile} from project root: {project_root}")
        main_cfg_path = project_root / 'config' / 'main_config.yaml'
        data_cfg_path = project_root / 'config' / f'data_config_{args.exec_profile}.yaml'
        env_cfg_path = project_root / 'config' / 'environment_config.yaml'

        main_config = load_config(str(main_cfg_path))
        data_config = load_config(str(data_cfg_path))
        environment_config = load_config(str(env_cfg_path))

        if not main_config or not data_config or not environment_config:
            logger.error("❌ Critical error: One or more configuration files could not be loaded.")
            sys.exit(1)

        config = {
            'main': main_config,
            'paths': main_config.get('paths', {}),
            'data': data_config,
            'environment': environment_config.get('environment', {}) # Ensure 'environment' key exists
        }

        # Override training_timeframe from args
        # args.training_timeframe has a default, so it will always be set.
        config['data']['training_timeframe'] = args.training_timeframe
        logger.info(f"Paper trading timeframe set to: {args.training_timeframe} (from command line or default).")

        logger.info(f"✅ Configuration loaded for profile: {args.exec_profile}")

        # Vérifier que le modèle existe
        model_path = Path(args.model_path)
        if not model_path.exists():
            logger.error(f"❌ Model not found: {model_path}")
            sys.exit(1)

        # Initialiser l'agent de paper trading
        trading_agent = PaperTradingAgent(
            config=config,
            model_path=str(model_path),
            initial_capital=args.initial_capital
        )

        # Lancer la boucle de trading
        trading_agent.run_trading_loop(
            max_iterations=args.max_iterations,
            sleep_seconds=args.sleep_seconds
        )

    except Exception as e:
        logger.error(f"❌ Paper trading failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
