#!/usr/bin/env python3
"""
ADAN Walk-Forward Validation Engine
====================================
Rigorous chronological train/test validation methodology that prevents
look-ahead bias and measures true out-of-sample performance across
different market regimes.

Methodology (anchored walk-forward):
  For K folds, the dataset is split into K+1 contiguous slices:
    Fold 0: Train [0..s1]           Test [s1..s2]
    Fold 1: Train [0..s2]           Test [s2..s3]
    ...
    Fold K: Train [0..sK]           Test [sK..end]

  Each fold:
    A) Train PPO on the training window (N steps).
    B) Save model + VecNormalize checkpoint.
    C) Load checkpoint in INFERENCE mode (training=False, norm_reward=False).
    D) Run deterministic inference on the test window.
    E) Log PnL, Sharpe, max drawdown, trade count.

  The expanding (anchored) window ensures the model always trains on
  all available historical data, matching production deployment.

Usage:
    python scripts/walk_forward_validation.py --steps-per-fold 500 --folds 2
    python scripts/walk_forward_validation.py --steps-per-fold 5000 --folds 4

Author: ADAN Quant Team
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Project root
_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("walk_forward")


# ---------------------------------------------------------------------------
# Fold result dataclass
# ---------------------------------------------------------------------------
@dataclass
class FoldResult:
    """Metrics for a single walk-forward fold."""
    fold_id: int
    train_rows: int
    test_rows: int
    train_steps: int
    test_steps: int
    # Test-set metrics
    total_reward: float = 0.0
    total_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    trade_count: int = 0
    final_balance: float = 0.0
    duration_seconds: float = 0.0
    # Auxiliary
    avg_kelly_modifier: float = 0.0
    aux_loss_mean: float = 0.0


# ---------------------------------------------------------------------------
# Walk-Forward core engine
# ---------------------------------------------------------------------------
class WalkForwardEngine:
    """Anchored walk-forward cross-validation for RL trading models."""

    def __init__(
        self,
        config_path: str = "config/config.yaml",
        data_dir: str = "data/processed/indicators/train",
        asset: str = "BTCUSDT",
        n_folds: int = 4,
        steps_per_fold: int = 5000,
        output_dir: str = "results/walk_forward",
    ):
        self.config_path = str(PROJECT_ROOT / config_path)
        self.data_dir = str(PROJECT_ROOT / data_dir)
        self.asset = asset
        self.n_folds = n_folds
        self.steps_per_fold = steps_per_fold
        self.output_dir = str(PROJECT_ROOT / output_dir)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        self.results: List[FoldResult] = []

    # -----------------------------------------------------------------
    # Data loading & splitting
    # -----------------------------------------------------------------
    def _load_data(self) -> Dict[str, pd.DataFrame]:
        """Load multi-timeframe parquet data."""
        frames = {}
        for tf in ["5m", "1h", "4h"]:
            p = Path(self.data_dir) / self.asset / f"{tf}.parquet"
            if not p.exists():
                raise FileNotFoundError(
                    f"Missing {p}. Run: python scripts/generate_colab_dataset.py first."
                )
            frames[tf] = pd.read_parquet(p)
            logger.info(f"  Loaded {tf}: {len(frames[tf])} rows")
        return frames

    def _split_folds(
        self, total_rows: int
    ) -> List[Dict[str, int]]:
        """Compute anchored walk-forward fold boundaries.

        Returns list of dicts with keys: train_start, train_end, test_start, test_end.
        """
        # Reserve at least 20% of data for testing across all folds
        min_train = max(100, total_rows // (self.n_folds + 2))
        fold_size = (total_rows - min_train) // (self.n_folds + 1)
        fold_size = max(50, fold_size)

        folds = []
        for i in range(self.n_folds):
            train_end = min_train + fold_size * (i + 1)
            test_start = train_end
            test_end = min(train_end + fold_size, total_rows)
            if test_start >= total_rows or test_end <= test_start:
                break
            folds.append({
                "train_start": 0,  # anchored
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            })
        return folds

    def _slice_data(
        self, frames: Dict[str, pd.DataFrame], start: int, end: int
    ) -> Dict[str, pd.DataFrame]:
        """Slice all timeframes to [start:end]."""
        return {tf: df.iloc[start:end].copy() for tf, df in frames.items()}

    def _save_temp_parquets(
        self, sliced: Dict[str, pd.DataFrame], tmp_dir: str
    ) -> str:
        """Save sliced data as parquets in a temp directory structure."""
        out = Path(tmp_dir) / "train" / self.asset
        out.mkdir(parents=True, exist_ok=True)
        for tf, df in sliced.items():
            df.to_parquet(out / f"{tf}.parquet", engine="pyarrow")
        return str(Path(tmp_dir))

    # -----------------------------------------------------------------
    # Environment & model creation
    # -----------------------------------------------------------------
    def _create_env(self, data_dir: str, is_training: bool = True):
        """Create the full ADAN environment stack."""
        from adan_trading_bot.common.config_loader import ConfigLoader
        from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader
        from adan_trading_bot.environment.multi_asset_chunked_env import (
            MultiAssetChunkedEnv,
        )
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        # Load config and override data_dirs
        config = ConfigLoader.load_config(self.config_path)
        config.setdefault("data", {})["data_dirs"] = {
            "base": data_dir,
            "train": str(Path(data_dir) / "train"),
            "test": str(Path(data_dir) / "train"),
            "val": str(Path(data_dir) / "train"),
        }

        # Override paths in config
        config.setdefault("paths", {})["indicators_data_dir"] = data_dir

        # Worker config: use w1 defaults with correct timeframes and asset
        worker_config = config.get("workers", {}).get("w1", {})
        if not worker_config:
            worker_config = {}
        worker_config.setdefault("timeframes", ["5m", "1h", "4h"])
        worker_config.setdefault("data_split", "train")
        worker_config.setdefault("assets", [self.asset])

        loader = ChunkedDataLoader(
            config=config, worker_config=worker_config, worker_id=0,
        )
        data = loader.load_chunk(0)

        def make_env():
            env = MultiAssetChunkedEnv(
                data=data,
                config=config,
                worker_config=config.get("workers", {}).get("w1", {}),
                worker_id=0,
            )
            return env

        vec_env = DummyVecEnv([make_env])
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=is_training,
            clip_obs=10.0,
            clip_reward=10.0,
            training=is_training,
        )
        return vec_env

    def _create_model(self, env, learning_rate: float = 3e-5):
        """Create a fresh WorldModelPPO (or fallback PPO)."""
        from stable_baselines3 import PPO

        try:
            from adan_trading_bot.agent.feature_extractors import (
                WorldModelPPO,
                ContextualTemporalFusionExtractor,
            )
            PPOClass = WorldModelPPO
        except ImportError:
            PPOClass = PPO
            ContextualTemporalFusionExtractor = None

        policy_kwargs = {}
        if ContextualTemporalFusionExtractor is not None:
            policy_kwargs["features_extractor_class"] = ContextualTemporalFusionExtractor

        kwargs = dict(
            policy="MultiInputPolicy",
            env=env,
            learning_rate=learning_rate,
            n_steps=min(512, self.steps_per_fold),
            batch_size=64,
            n_epochs=4,
            gamma=0.97,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs if policy_kwargs else None,
            verbose=0,
            device="cpu",
        )
        if PPOClass.__name__ == "WorldModelPPO":
            kwargs["aux_loss_coef"] = 0.1
        model = PPOClass(**kwargs)
        return model

    # -----------------------------------------------------------------
    # Single fold execution
    # -----------------------------------------------------------------
    def _run_fold(
        self,
        fold_id: int,
        frames: Dict[str, pd.DataFrame],
        boundaries: Dict[str, int],
        prev_model_path: Optional[str] = None,
    ) -> FoldResult:
        """Execute one fold: train then test."""
        t0 = time.time()
        logger.info(f"{'='*60}")
        logger.info(
            f"FOLD {fold_id}/{self.n_folds} | "
            f"Train [0..{boundaries['train_end']}] | "
            f"Test [{boundaries['test_start']}..{boundaries['test_end']}]"
        )
        logger.info(f"{'='*60}")

        result = FoldResult(
            fold_id=fold_id,
            train_rows=boundaries["train_end"],
            test_rows=boundaries["test_end"] - boundaries["test_start"],
            train_steps=self.steps_per_fold,
            test_steps=0,
        )

        # ---- A) TRAIN ----
        logger.info(f"[Fold {fold_id}] Phase A: Training for {self.steps_per_fold} steps...")
        train_data = self._slice_data(frames, 0, boundaries["train_end"])

        with tempfile.TemporaryDirectory() as tmp_train:
            train_dir = self._save_temp_parquets(train_data, tmp_train)

            try:
                train_env = self._create_env(train_dir, is_training=True)
            except Exception as e:
                logger.error(f"[Fold {fold_id}] Environment creation failed: {e}")
                result.duration_seconds = time.time() - t0
                return result

            # Load previous model or create fresh
            try:
                from stable_baselines3 import PPO
                try:
                    from adan_trading_bot.agent.feature_extractors import WorldModelPPO
                    PPOClass = WorldModelPPO
                except ImportError:
                    PPOClass = PPO

                if prev_model_path and os.path.exists(prev_model_path):
                    logger.info(f"  Warm-starting from {prev_model_path}")
                    model = PPOClass.load(prev_model_path, env=train_env)
                else:
                    model = self._create_model(train_env)
            except Exception as e:
                logger.error(f"[Fold {fold_id}] Model creation failed: {e}")
                train_env.close()
                result.duration_seconds = time.time() - t0
                return result

            # Train
            try:
                model.learn(total_timesteps=self.steps_per_fold)
            except Exception as e:
                logger.error(f"[Fold {fold_id}] Training failed: {e}")
                train_env.close()
                result.duration_seconds = time.time() - t0
                return result

            # Save checkpoint
            ckpt_dir = Path(self.output_dir) / f"fold_{fold_id}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model_path = str(ckpt_dir / "model.zip")
            vec_path = str(ckpt_dir / "vecnormalize.pkl")
            model.save(model_path)
            train_env.save(vec_path)

            # Record aux_loss if available
            if hasattr(model, '_aux_loss_history') and model._aux_loss_history:
                result.aux_loss_mean = float(np.mean(model._aux_loss_history))
                logger.info(
                    f"  Auxiliary loss mean: {result.aux_loss_mean:.6f} "
                    f"({len(model._aux_loss_history)} samples)"
                )

            train_env.close()
            del model

        # ---- B) TEST (Out-of-Sample) ----
        logger.info(f"[Fold {fold_id}] Phase B: OOS Inference on test window...")
        test_data = self._slice_data(
            frames, boundaries["test_start"], boundaries["test_end"]
        )

        with tempfile.TemporaryDirectory() as tmp_test:
            test_dir = self._save_temp_parquets(test_data, tmp_test)

            try:
                test_env = self._create_env(test_dir, is_training=False)
            except Exception as e:
                logger.error(f"[Fold {fold_id}] Test env creation failed: {e}")
                result.duration_seconds = time.time() - t0
                return result

            # Load VecNormalize in INFERENCE mode (CRITICAL)
            from stable_baselines3.common.vec_env import VecNormalize
            from stable_baselines3 import PPO
            try:
                from adan_trading_bot.agent.feature_extractors import WorldModelPPO
                PPOClass = WorldModelPPO
            except ImportError:
                PPOClass = PPO

            try:
                venv = test_env.venv if hasattr(test_env, 'venv') else test_env
                test_env = VecNormalize.load(vec_path, venv)
                test_env.training = False        # CRITICAL: inference mode
                test_env.norm_reward = False      # CRITICAL: raw rewards
                logger.info("  VecNormalize loaded: training=False, norm_reward=False")
            except Exception as e:
                logger.warning(f"  VecNormalize load failed, using default: {e}")

            # Load model
            try:
                model = PPOClass.load(model_path, env=test_env)
            except Exception as e:
                logger.error(f"[Fold {fold_id}] Model load failed: {e}")
                test_env.close()
                result.duration_seconds = time.time() - t0
                return result

            # Run deterministic inference
            obs = test_env.reset()
            total_reward = 0.0
            rewards_list = []
            equity_curve = []
            step_count = 0
            kelly_mods = []

            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, dones, infos = test_env.step(action)
                r = float(reward[0])
                total_reward += r
                rewards_list.append(r)
                step_count += 1

                # Track equity & Kelly from info
                info = infos[0] if infos else {}
                balance = info.get("portfolio_value", info.get("balance", 0))
                equity_curve.append(balance)
                km = info.get("kelly_modifier", 1.0)
                kelly_mods.append(km)

                if dones[0]:
                    obs = test_env.reset()
                    done = True  # one episode for OOS test

                # Safety cap: max 2x test rows as steps
                if step_count >= 2 * result.test_rows:
                    break

            result.test_steps = step_count
            result.total_reward = total_reward

            # Compute financial metrics
            if rewards_list:
                rets = np.array(rewards_list)
                # Sharpe (annualised, assuming 5-min bars)
                if rets.std() > 1e-9:
                    result.sharpe_ratio = float(
                        rets.mean() / rets.std() * np.sqrt(252 * 288)
                    )

                # Max drawdown on equity curve
                if equity_curve:
                    eq = np.array(equity_curve, dtype=float)
                    if len(eq) > 0 and eq.max() > 0:
                        running_max = np.maximum.accumulate(eq)
                        drawdowns = (running_max - eq) / np.where(
                            running_max > 0, running_max, 1
                        )
                        result.max_drawdown = float(drawdowns.max())
                    if len(eq) > 0:
                        result.final_balance = float(eq[-1])

            result.trade_count = info.get("total_trades", 0)
            if kelly_mods:
                result.avg_kelly_modifier = float(np.mean(kelly_mods))
            result.duration_seconds = time.time() - t0

            test_env.close()
            del model

        logger.info(
            f"[Fold {fold_id}] DONE | Steps={result.test_steps} | "
            f"Reward={result.total_reward:.4f} | Sharpe={result.sharpe_ratio:.3f} | "
            f"MaxDD={result.max_drawdown:.3%} | Trades={result.trade_count} | "
            f"Balance={result.final_balance:.2f} | "
            f"AvgKelly={result.avg_kelly_modifier:.3f} | "
            f"Time={result.duration_seconds:.1f}s"
        )
        return result

    # -----------------------------------------------------------------
    # Main orchestrator
    # -----------------------------------------------------------------
    def run(self) -> List[FoldResult]:
        """Execute the full walk-forward validation."""
        logger.info("=" * 70)
        logger.info("ADAN WALK-FORWARD VALIDATION ENGINE")
        logger.info(f"  Folds: {self.n_folds}")
        logger.info(f"  Steps per fold: {self.steps_per_fold}")
        logger.info(f"  Asset: {self.asset}")
        logger.info(f"  Data dir: {self.data_dir}")
        logger.info("=" * 70)

        # 1. Load data
        frames = self._load_data()
        total_rows = min(len(df) for df in frames.values())
        logger.info(f"Total aligned rows: {total_rows}")

        # 2. Compute fold boundaries
        folds = self._split_folds(total_rows)
        if not folds:
            logger.error("Not enough data for requested fold count. Use more candles.")
            return []
        logger.info(f"Planned {len(folds)} folds:")
        for f in folds:
            logger.info(f"  Train [0..{f['train_end']}] -> Test [{f['test_start']}..{f['test_end']}]")

        # 3. Execute folds
        prev_model = None
        for i, boundaries in enumerate(folds):
            result = self._run_fold(i, frames, boundaries, prev_model)
            self.results.append(result)
            # Chain model for warm-starting next fold
            ckpt = Path(self.output_dir) / f"fold_{i}" / "model.zip"
            if ckpt.exists():
                prev_model = str(ckpt)

        # 4. Summary
        self._print_summary()
        self._save_report()
        return self.results

    def _print_summary(self):
        """Print a formatted summary table."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("WALK-FORWARD VALIDATION SUMMARY")
        logger.info("=" * 80)
        header = (
            f"{'Fold':>4} | {'Train':>6} | {'Test':>6} | "
            f"{'Reward':>10} | {'Sharpe':>8} | {'MaxDD':>8} | "
            f"{'Trades':>6} | {'Balance':>10} | {'Kelly':>6} | {'AuxLoss':>8}"
        )
        logger.info(header)
        logger.info("-" * 80)
        for r in self.results:
            row = (
                f"{r.fold_id:>4} | {r.train_rows:>6} | {r.test_rows:>6} | "
                f"{r.total_reward:>+10.4f} | {r.sharpe_ratio:>8.3f} | "
                f"{r.max_drawdown:>7.2%} | {r.trade_count:>6} | "
                f"{r.final_balance:>10.2f} | {r.avg_kelly_modifier:>6.3f} | "
                f"{r.aux_loss_mean:>8.6f}"
            )
            logger.info(row)

        # Aggregate
        if self.results:
            avg_sharpe = np.mean([r.sharpe_ratio for r in self.results])
            avg_dd = np.mean([r.max_drawdown for r in self.results])
            total_trades = sum(r.trade_count for r in self.results)
            survived = sum(
                1 for r in self.results if r.final_balance > 0 and r.test_steps > 0
            )
            logger.info("-" * 80)
            logger.info(
                f"Avg Sharpe: {avg_sharpe:.3f} | Avg MaxDD: {avg_dd:.2%} | "
                f"Total Trades: {total_trades} | "
                f"Survival: {survived}/{len(self.results)} folds"
            )

            # Regime robustness verdict
            if survived == len(self.results) and avg_dd < 0.5:
                logger.info("VERDICT: Model demonstrates regime robustness (all folds survived)")
            elif survived > 0:
                logger.info(f"VERDICT: Partial robustness ({survived}/{len(self.results)} folds survived)")
            else:
                logger.info("VERDICT: Model failed walk-forward validation")
        logger.info("=" * 80)

    def _save_report(self):
        """Save JSON report."""
        report = {
            "n_folds": self.n_folds,
            "steps_per_fold": self.steps_per_fold,
            "asset": self.asset,
            "folds": [asdict(r) for r in self.results],
            "summary": {
                "avg_sharpe": float(np.mean([r.sharpe_ratio for r in self.results])) if self.results else 0,
                "avg_max_drawdown": float(np.mean([r.max_drawdown for r in self.results])) if self.results else 0,
                "total_trades": sum(r.trade_count for r in self.results),
                "folds_survived": sum(1 for r in self.results if r.final_balance > 0 and r.test_steps > 0),
            },
        }
        out_path = Path(self.output_dir) / "walk_forward_report.json"
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="ADAN Walk-Forward Validation Engine"
    )
    parser.add_argument(
        "--config", default="config/config.yaml",
        help="Path to config YAML (relative to project root)"
    )
    parser.add_argument(
        "--data-dir", default="data/processed/indicators/train",
        help="Path to training parquets"
    )
    parser.add_argument("--asset", default="BTCUSDT")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--steps-per-fold", type=int, default=5000)
    parser.add_argument(
        "--output", default="results/walk_forward",
        help="Output directory for checkpoints and reports"
    )
    args = parser.parse_args()

    engine = WalkForwardEngine(
        config_path=args.config,
        data_dir=args.data_dir,
        asset=args.asset,
        n_folds=args.folds,
        steps_per_fold=args.steps_per_fold,
        output_dir=args.output,
    )
    results = engine.run()
    sys.exit(0 if results else 1)


if __name__ == "__main__":
    main()
