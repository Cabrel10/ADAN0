#!/usr/bin/env python3
"""
Reward logging module for the ADAN trading bot.

This module provides specialized logging functionality for reward tracking,
analysis, and debugging of the reward system.
"""

import json
import logging
import time
from collections import deque, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)

class RewardLogger:
    """
    Specialized logging system for reward tracking and analysis.

    Provides detailed logging of:
    - Individual reward calculations and components
    - Reward trends and patterns
    - Performance bonus tracking
    - Risk-adjusted reward analysis
    - Debugging information for reward system
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the reward logger.

        Args:
            config: Configuration dictionary containing logging parameters
        """
        self.config = config.get('reward_logging', {})
        self.enabled = self.config.get('enabled', True)
        self.log_level = self.config.get('log_level', 'INFO')
        self.save_interval = self.config.get('save_interval', 100)
        self.max_history = self.config.get('max_history', 10000)

        # Storage paths
        self.base_path = Path(self.config.get('base_path', 'logs/rewards'))
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Reward tracking
        self.reward_history = deque(maxlen=self.max_history)
        self.component_history = defaultdict(lambda: deque(maxlen=self.max_history))
        self.bonus_history = deque(maxlen=self.max_history)
        self.episode_rewards = {}

        # Statistics
        self.total_rewards_logged = 0
        self.session_start = datetime.now()
        self.last_save_time = time.time()

        # Configure detailed logging
        self._setup_detailed_logger()

        logger.info(f"RewardLogger initialized with base path: {self.base_path}")

    def _setup_detailed_logger(self) -> None:
        """Setup detailed file logger for rewards."""
        if not self.enabled:
            return

        # Create detailed reward logger
        self.detailed_logger = logging.getLogger('reward_detailed')
        self.detailed_logger.setLevel(getattr(logging, self.log_level))

        # Remove existing handlers to avoid duplicates
        for handler in self.detailed_logger.handlers[:]:
            self.detailed_logger.removeHandler(handler)

        # File handler for detailed reward logs
        log_file = self.base_path / f"reward_detailed_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, self.log_level))

        # Detailed formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)

        self.detailed_logger.addHandler(file_handler)
        self.detailed_logger.propagate = False

    def log_reward_calculation(self, reward_data: Dict[str, Any]) -> None:
        """
        Log a detailed reward calculation.

        Args:
            reward_data: Dictionary containing reward calculation details
        """
        if not self.enabled:
            return

        timestamp = datetime.now()

        # Extract reward components
        total_reward = reward_data.get('total_reward', 0.0)
        components = reward_data.get('components', {})
        metadata = reward_data.get('metadata', {})

        # Store in history
        reward_record = {
            'timestamp': timestamp.isoformat(),
            'total_reward': total_reward,
            'components': components.copy(),
            'metadata': metadata.copy()
        }

        self.reward_history.append(reward_record)

        # Store component history
        for component, value in components.items():
            self.component_history[component].append({
                'timestamp': timestamp.isoformat(),
                'value': value
            })

        # Convertir les valeurs NumPy en types natifs Python
        def convert_numpy(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj) if isinstance(obj, np.floating) else int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy(x) for x in obj]
            return obj

        # Convertir les données avant la sérialisation
        components_serializable = convert_numpy(components)
        metadata_serializable = convert_numpy(metadata)

        # Detailed logging
        self.detailed_logger.info(
            f"REWARD_CALC | Total: {total_reward:.6f} | "
            f"Components: {json.dumps(components_serializable, default=str)} | "
            f"Metadata: {json.dumps(metadata_serializable, default=str)}"
        )

        self.total_rewards_logged += 1

        # Auto-save periodically
        if self.total_rewards_logged % self.save_interval == 0:
            self.save_reward_logs()

    def log_performance_bonus(self, bonus_data: Dict[str, Any]) -> None:
        """
        Log performance bonus details.

        Args:
            bonus_data: Dictionary containing bonus calculation details
        """
        if not self.enabled:
            return

        timestamp = datetime.now()

        bonus_record = {
            'timestamp': timestamp.isoformat(),
            'chunk_id': bonus_data.get('chunk_id'),
            'optimal_pnl': bonus_data.get('optimal_pnl', 0.0),
            'actual_pnl': bonus_data.get('actual_pnl', 0.0),
            'performance_ratio': bonus_data.get('performance_ratio', 0.0),
            'bonus_amount': bonus_data.get('bonus_amount', 0.0),
            'threshold': bonus_data.get('threshold', 0.0)
        }

        self.bonus_history.append(bonus_record)

        # Detailed logging
        self.detailed_logger.info(
            f"PERFORMANCE_BONUS | Chunk: {bonus_record['chunk_id']} | "
            f"Ratio: {bonus_record['performance_ratio']:.3f} | "
            f"Bonus: {bonus_record['bonus_amount']:.6f} | "
            f"Optimal PnL: {bonus_record['optimal_pnl']:.2f}%"
        )

    def log_episode_reward(self, episode_id: int, episode_data: Dict[str, Any]) -> None:
        """
        Log episode-level reward summary.

        Args:
            episode_id: Episode identifier
            episode_data: Dictionary containing episode reward data
        """
        if not self.enabled:
            return

        timestamp = datetime.now()

        episode_record = {
            'timestamp': timestamp.isoformat(),
            'episode_id': episode_id,
            'total_reward': episode_data.get('total_reward', 0.0),
            'average_reward': episode_data.get('average_reward', 0.0),
            'reward_std': episode_data.get('reward_std', 0.0),
            'min_reward': episode_data.get('min_reward', 0.0),
            'max_reward': episode_data.get('max_reward', 0.0),
            'episode_length': episode_data.get('episode_length', 0),
            'performance_bonuses': episode_data.get('performance_bonuses', 0),
            'risk_penalties': episode_data.get('risk_penalties', 0)
        }

        self.episode_rewards[episode_id] = episode_record

        # Detailed logging
        self.detailed_logger.info(
            f"EPISODE_REWARD | Episode: {episode_id} | "
            f"Total: {episode_record['total_reward']:.4f} | "
            f"Avg: {episode_record['average_reward']:.6f} | "
            f"Length: {episode_record['episode_length']}"
        )

    def log_reward_trend_analysis(self) -> None:
        """Log reward trend analysis."""
        if not self.enabled or len(self.reward_history) < 10:
            return

        # Extract recent rewards
        recent_rewards = [r['total_reward'] for r in list(self.reward_history)[-100:]]

        # Calculate trend metrics
        trend_metrics = {
            'mean': np.mean(recent_rewards),
            'std': np.std(recent_rewards),
            'min': np.min(recent_rewards),
            'max': np.max(recent_rewards),
            'trend': self._calculate_trend(recent_rewards),
            'volatility': np.std(recent_rewards) / abs(np.mean(recent_rewards)) if np.mean(recent_rewards) != 0 else 0
        }

        self.detailed_logger.info(
            f"REWARD_TREND | Mean: {trend_metrics['mean']:.6f} | "
            f"Std: {trend_metrics['std']:.6f} | "
            f"Trend: {trend_metrics['trend']} | "
            f"Volatility: {trend_metrics['volatility']:.3f}"
        )

    def log_component_analysis(self) -> None:
        """Log analysis of reward components."""
        if not self.enabled or not self.component_history:
            return

        component_analysis = {}

        for component, history in self.component_history.items():
            if len(history) < 5:
                continue

            values = [h['value'] for h in list(history)[-50:]]

            component_analysis[component] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'contribution_pct': abs(np.mean(values)) / sum(abs(np.mean([h['value'] for h in list(hist)[-50:]]))
                                                              for hist in self.component_history.values()) * 100
            }

        # Log component contributions
        for component, analysis in component_analysis.items():
            self.detailed_logger.info(
                f"COMPONENT_ANALYSIS | {component} | "
                f"Mean: {analysis['mean']:.6f} | "
                f"Contribution: {analysis['contribution_pct']:.1f}%"
            )

    def get_reward_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive reward statistics.

        Returns:
            Dictionary containing reward statistics
        """
        if not self.reward_history:
            return {'error': 'No reward data available'}

        # Extract all rewards
        all_rewards = [r['total_reward'] for r in self.reward_history]

        # Basic statistics
        stats = {
            'total_rewards_logged': len(all_rewards),
            'session_duration_hours': (datetime.now() - self.session_start).total_seconds() / 3600,
            'mean_reward': np.mean(all_rewards),
            'std_reward': np.std(all_rewards),
            'min_reward': np.min(all_rewards),
            'max_reward': np.max(all_rewards),
            'median_reward': np.median(all_rewards),
            'reward_trend': self._calculate_trend(all_rewards[-100:]) if len(all_rewards) >= 100 else 'insufficient_data'
        }

        # Percentiles
        stats['reward_percentiles'] = {
            '5th': np.percentile(all_rewards, 5),
            '25th': np.percentile(all_rewards, 25),
            '75th': np.percentile(all_rewards, 75),
            '95th': np.percentile(all_rewards, 95)
        }

        # Component statistics
        stats['component_stats'] = {}
        for component, history in self.component_history.items():
            if history:
                values = [h['value'] for h in history]
                stats['component_stats'][component] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'count': len(values)
                }

        # Bonus statistics
        if self.bonus_history:
            bonus_amounts = [b['bonus_amount'] for b in self.bonus_history]
            stats['bonus_stats'] = {
                'total_bonuses': len(bonus_amounts),
                'total_bonus_amount': sum(bonus_amounts),
                'average_bonus': np.mean(bonus_amounts),
                'bonus_frequency': len(bonus_amounts) / len(all_rewards) if all_rewards else 0
            }

        return stats

    def save_reward_logs(self, filename: Optional[str] = None) -> None:
        """
        Save reward logs to file.

        Args:
            filename: Optional custom filename
        """
        if not self.enabled:
            return

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reward_logs_{timestamp}.json"

        filepath = self.base_path / filename

        # Prepare data for saving
        save_data = {
            'session_info': {
                'start_time': self.session_start.isoformat(),
                'save_time': datetime.now().isoformat(),
                'total_rewards_logged': self.total_rewards_logged
            },
            'reward_statistics': self.get_reward_statistics(),
            'recent_rewards': list(self.reward_history)[-1000:],  # Last 1000 rewards
            'recent_bonuses': list(self.bonus_history)[-100:],    # Last 100 bonuses
            'episode_summaries': dict(list(self.episode_rewards.items())[-50:])  # Last 50 episodes
        }

        try:
            with open(filepath, 'w') as f:
                json.dump(save_data, f, indent=2, default=str)

            self.last_save_time = time.time()
            logger.info(f"Reward logs saved to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save reward logs: {str(e)}")

    def load_reward_logs(self, filename: str) -> bool:
        """
        Load reward logs from file.

        Args:
            filename: Name of the file to load

        Returns:
            True if successful, False otherwise
        """
        filepath = self.base_path / filename

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Restore reward history
            if 'recent_rewards' in data:
                for reward_record in data['recent_rewards']:
                    self.reward_history.append(reward_record)

            # Restore bonus history
            if 'recent_bonuses' in data:
                for bonus_record in data['recent_bonuses']:
                    self.bonus_history.append(bonus_record)

            # Restore episode data
            if 'episode_summaries' in data:
                self.episode_rewards.update(data['episode_summaries'])

            # Update counters
            if 'session_info' in data:
                self.total_rewards_logged = data['session_info'].get('total_rewards_logged', 0)

            logger.info(f"Reward logs loaded from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to load reward logs from {filename}: {str(e)}")
            return False

    def reset_logs(self) -> None:
        """Reset all reward logs."""
        self.reward_history.clear()
        self.component_history.clear()
        self.bonus_history.clear()
        self.episode_rewards.clear()

        self.total_rewards_logged = 0
        self.session_start = datetime.now()

        logger.info("All reward logs have been reset")

    def _calculate_trend(self, values: List[float]) -> str:
        """
        Calculate trend direction for a series of values.

        Args:
            values: List of numeric values

        Returns:
            Trend direction: 'up', 'down', or 'stable'
        """
        if len(values) < 2:
            return 'stable'

        # Simple linear regression slope
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]

        if slope > 0.001:
            return 'up'
        elif slope < -0.001:
            return 'down'
        else:
            return 'stable'

    def generate_reward_report(self) -> str:
        """
        Generate a human-readable reward report.

        Returns:
            Formatted reward report string
        """
        stats = self.get_reward_statistics()

        if 'error' in stats:
            return "No reward data available for report generation."

        report_lines = [
            "=" * 60,
            "REWARD SYSTEM REPORT",
            "=" * 60,
            f"Session Duration: {stats['session_duration_hours']:.1f} hours",
            f"Total Rewards Logged: {stats['total_rewards_logged']:,}",
            "",
            "REWARD STATISTICS:",
            f"  Mean Reward: {stats['mean_reward']:.6f}",
            f"  Std Deviation: {stats['std_reward']:.6f}",
            f"  Min/Max: {stats['min_reward']:.6f} / {stats['max_reward']:.6f}",
            f"  Median: {stats['median_reward']:.6f}",
            f"  Trend: {stats['reward_trend']}",
            "",
            "PERCENTILES:",
            f"  5th: {stats['reward_percentiles']['5th']:.6f}",
            f"  25th: {stats['reward_percentiles']['25th']:.6f}",
            f"  75th: {stats['reward_percentiles']['75th']:.6f}",
            f"  95th: {stats['reward_percentiles']['95th']:.6f}",
        ]

        # Add component statistics
        if stats['component_stats']:
            report_lines.extend([
                "",
                "COMPONENT CONTRIBUTIONS:",
            ])
            for component, comp_stats in stats['component_stats'].items():
                report_lines.append(f"  {component}: {comp_stats['mean']:.6f} (±{comp_stats['std']:.6f})")

        # Add bonus statistics
        if 'bonus_stats' in stats:
            bonus_stats = stats['bonus_stats']
            report_lines.extend([
                "",
                "PERFORMANCE BONUSES:",
                f"  Total Bonuses: {bonus_stats['total_bonuses']}",
                f"  Total Amount: {bonus_stats['total_bonus_amount']:.4f}",
                f"  Average Bonus: {bonus_stats['average_bonus']:.6f}",
                f"  Bonus Frequency: {bonus_stats['bonus_frequency']:.1%}",
            ])

        report_lines.append("=" * 60)

        return "\n".join(report_lines)
