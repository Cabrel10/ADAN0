"""
Exogenous Regime Oracle — XGBoost-based macro-economic regime classifier.

Predicts P(bull|macro), P(bear|macro), P(sideways|macro) from exogenous features
(S&P500, DXY, Gold, VIX) to augment the endogenous HMM regime detection.

When macro data is unavailable, falls back to BTC-only features (momentum, volatility).
When the model is not yet trained, returns uniform [1/3, 1/3, 1/3].
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Try to import xgboost; fall back to sklearn GradientBoosting if unavailable
try:
    from xgboost import XGBClassifier
    _CLASSIFIER_CLS = XGBClassifier
    _CLASSIFIER_KWARGS = {
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.1,
        "use_label_encoder": False,
        "eval_metric": "mlogloss",
        "random_state": 42,
        "verbosity": 0,
    }
    logger.debug("ExogenousRegimeOracle: using XGBClassifier")
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    _CLASSIFIER_CLS = GradientBoostingClassifier
    _CLASSIFIER_KWARGS = {
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.1,
        "random_state": 42,
    }
    logger.debug("ExogenousRegimeOracle: XGBoost unavailable, using sklearn GradientBoosting")

from sklearn.preprocessing import RobustScaler


class ExogenousRegimeOracle:
    """Exogenous regime predictor using macro-economic indicators.

    Public API:
        fit(macro_df, btc_series)   — train from macro DataFrame + BTC daily close
        predict_proba_safe(X)       — return [P_bear, P_side, P_bull] (always sums to 1)
        save(path) / load(path)     — persist / restore model
        is_fitted                   — property: True if model has been trained
    """

    def __init__(self):
        self.model = None
        self.scaler = RobustScaler()
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, macro_df, btc_series):
        """Train the oracle on macro features + BTC daily close.

        Args:
            macro_df: DataFrame with macro columns (spy, dxy, gold, etc.).
                      Can be empty — in that case, BTC-only features are used.
            btc_series: Series of BTC daily close prices.
        """
        import pandas as pd

        # Build feature matrix from BTC
        btc = btc_series.copy().dropna()
        if len(btc) < 30:
            logger.warning("[Oracle] Not enough BTC data to train (<30 days)")
            return

        features = pd.DataFrame(index=btc.index)

        # BTC momentum features
        features["btc_ret_1d"] = np.log(btc / btc.shift(1))
        features["btc_ret_5d"] = np.log(btc / btc.shift(5))
        features["btc_ret_10d"] = np.log(btc / btc.shift(10))
        features["btc_vol_5d"] = features["btc_ret_1d"].rolling(5).std()
        features["btc_vol_20d"] = features["btc_ret_1d"].rolling(20).std()

        # Add macro features if available
        if macro_df is not None and len(macro_df) > 0:
            for col in macro_df.columns:
                aligned = macro_df[col].reindex(features.index, method="ffill")
                features[f"{col}_ret_1d"] = np.log(aligned / aligned.shift(1))
                features[f"{col}_ret_5d"] = np.log(aligned / aligned.shift(5))

        # Drop NaN rows
        features = features.replace([np.inf, -np.inf], np.nan).dropna()

        if len(features) < 30:
            logger.warning(f"[Oracle] Too few valid samples ({len(features)}), skipping training")
            return

        # Create labels: 0=bear, 1=sideways, 2=bull based on forward 5-day return
        btc_aligned = btc.reindex(features.index)
        fwd_ret = np.log(btc_aligned.shift(-5) / btc_aligned)
        fwd_ret = fwd_ret.reindex(features.index).dropna()
        features = features.loc[fwd_ret.index]

        # Tercile-based labeling
        q33 = fwd_ret.quantile(0.33)
        q66 = fwd_ret.quantile(0.66)
        labels = pd.Series(1, index=fwd_ret.index)  # sideways by default
        labels[fwd_ret < q33] = 0  # bear
        labels[fwd_ret > q66] = 2  # bull

        X = features.values.astype(np.float64)
        y = labels.values.astype(int)

        # Scale
        X_scaled = self.scaler.fit_transform(X)

        # Train
        self.model = _CLASSIFIER_CLS(**_CLASSIFIER_KWARGS)
        self.model.fit(X_scaled, y)
        self._is_fitted = True

        # Log training info
        unique, counts = np.unique(y, return_counts=True)
        label_map = {0: "bear", 1: "sideways", 2: "bull"}
        dist = {label_map.get(u, str(u)): int(c) for u, c in zip(unique, counts)}
        logger.info(f"[Oracle] Trained on {len(X)} samples, label distribution: {dist}")

    def predict_proba_safe(self, X: np.ndarray) -> np.ndarray:
        """Predict regime probabilities. Always returns [P_bear, P_side, P_bull].

        Args:
            X: Feature vector of shape (n_features,) or (1, n_features).

        Returns:
            np.ndarray of shape (3,) summing to 1.0.
        """
        uniform = np.array([1/3, 1/3, 1/3], dtype=np.float32)

        if not self._is_fitted or self.model is None:
            return uniform

        try:
            X_2d = np.atleast_2d(X).astype(np.float64)

            # Check feature count matches
            expected = self.scaler.n_features_in_
            if X_2d.shape[1] != expected:
                # Pad or truncate
                if X_2d.shape[1] < expected:
                    X_2d = np.pad(X_2d, ((0, 0), (0, expected - X_2d.shape[1])))
                else:
                    X_2d = X_2d[:, :expected]

            X_scaled = self.scaler.transform(X_2d)
            probs = self.model.predict_proba(X_scaled)[0]

            # Ensure exactly 3 classes
            if len(probs) < 3:
                full_probs = np.array([1/3, 1/3, 1/3], dtype=np.float32)
                for i, p in enumerate(probs):
                    if i < 3:
                        full_probs[i] = p
                probs = full_probs / full_probs.sum()
            else:
                probs = probs[:3]

            # Normalize to sum to 1
            s = probs.sum()
            if s > 0:
                probs = probs / s
            else:
                return uniform

            return probs.astype(np.float32)
        except Exception as e:
            logger.debug(f"[Oracle] predict_proba_safe fallback: {e}")
            return uniform

    def save(self, path: str):
        """Save model + scaler to pickle file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler, "is_fitted": self._is_fitted}, f)
        logger.info(f"[Oracle] Saved to {path}")

    def load(self, path: str):
        """Load model + scaler from pickle file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self._is_fitted = data.get("is_fitted", True)
        logger.info(f"[Oracle] Loaded from {path}")
