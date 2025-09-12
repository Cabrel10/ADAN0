#!/usr/bin/env python3
"""
Metrics tracking module for the ADAN trading bot.

This module provides comprehensive tracking of trading performance metrics,
including real-time monitoring, historical analysis, and performance reporting.
"""

import json
import logging
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)

class MetricsTracker:
    """
    Comprehensive metrics tracking system for trading performance.
    
    Tracks various metrics including:
    - Trading performance (PnL, win rate, etc.)
    - Risk metrics (drawdown, Sharpe ratio, etc.)
    - Execution metrics (latency, slippage, etc.)
    - Learning metrics (reward, exploration, etc.)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the metrics tracker.
        
        Args:
            config: Configuration dictionary containing tracking parameters
        """
        self.config = config.get('metrics_tracking', {})
        self.enabled = self.config.get('enabled', True)
        self.save_interval = self.config.get('save_interval', 100)  # Save every N updates
        self.history_length = self.config.get('history_length', 10000)  # Keep last N records
        
        # Storage paths
        self.base_path = Path(self.config.get('base_path', 'logs/metrics'))
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Metrics storage
        self.metrics = defaultdict(deque)
        self.aggregated_metrics = {}
        self.session_start = datetime.now()
        self.update_count = 0
        
        # Performance tracking
        self.trade_history = []
        self.reward_history = deque(maxlen=self.history_length)
        self.loss_history = deque(maxlen=self.history_length)
        
        # Real-time metrics
        self.current_episode = 0
        self.current_step = 0
        self.last_save_time = time.time()
        
        logger.info(f"MetricsTracker initialized with base path: {self.base_path}")
    
    def update_trading_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update trading-related metrics.
        
        Args:
            metrics: Dictionary containing trading metrics
        """
        if not self.enabled:
            return
            
        timestamp = datetime.now()
        
        # Core trading metrics
        for key in ['pnl', 'realized_pnl', 'unrealized_pnl', 'total_capital', 
                   'drawdown', 'sharpe_ratio', 'win_rate', 'trade_count']:
            if key in metrics:
                self._add_metric(f'trading.{key}', metrics[key], timestamp)
        
        # Portfolio metrics
        if 'positions' in metrics:
            active_positions = sum(1 for pos in metrics['positions'].values() if pos.get('is_open', False))
            self._add_metric('trading.active_positions', active_positions, timestamp)
        
        # Risk metrics
        if 'var' in metrics:
            self._add_metric('risk.var', metrics['var'], timestamp)
        if 'cvar' in metrics:
            self._add_metric('risk.cvar', metrics['cvar'], timestamp)
    
    def update_learning_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update learning-related metrics.
        
        Args:
            metrics: Dictionary containing learning metrics
        """
        if not self.enabled:
            return
            
        timestamp = datetime.now()
        
        # Reward metrics
        if 'reward' in metrics:
            self.reward_history.append(metrics['reward'])
            self._add_metric('learning.reward', metrics['reward'], timestamp)
        
        # Loss metrics
        if 'loss' in metrics:
            self.loss_history.append(metrics['loss'])
            self._add_metric('learning.loss', metrics['loss'], timestamp)
        
        # Exploration metrics
        for key in ['epsilon', 'exploration_rate', 'action_distribution']:
            if key in metrics:
                self._add_metric(f'learning.{key}', metrics[key], timestamp)
        
        # Model metrics
        for key in ['learning_rate', 'gradient_norm', 'model_updates']:
            if key in metrics:
                self._add_metric(f'model.{key}', metrics[key], timestamp)
    
    def update_execution_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update execution-related metrics.
        
        Args:
            metrics: Dictionary containing execution metrics
        """
        if not self.enabled:
            return
            
        timestamp = datetime.now()
        
        # Latency metrics
        for key in ['order_latency', 'data_latency', 'decision_latency']:
            if key in metrics:
                self._add_metric(f'execution.{key}', metrics[key], timestamp)
        
        # Slippage and fees
        for key in ['slippage', 'commission', 'spread']:
            if key in metrics:
                self._add_metric(f'execution.{key}', metrics[key], timestamp)
        
        # Order metrics
        for key in ['orders_placed', 'orders_filled', 'orders_cancelled']:
            if key in metrics:
                self._add_metric(f'execution.{key}', metrics[key], timestamp)
    
    def update_market_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update market-related metrics.
        
        Args:
            metrics: Dictionary containing market metrics
        """
        if not self.enabled:
            return
            
        timestamp = datetime.now()
        
        # Market condition metrics
        for key in ['volatility', 'trend_strength', 'volume', 'spread']:
            if key in metrics:
                self._add_metric(f'market.{key}', metrics[key], timestamp)
        
        # Price metrics
        for asset in ['BTC', 'ETH', 'ADA', 'SOL', 'XRP']:
            if f'{asset}_price' in metrics:
                self._add_metric(f'market.{asset.lower()}_price', metrics[f'{asset}_price'], timestamp)
    
    def log_trade(self, trade_info: Dict[str, Any]) -> None:
        """
        Log a completed trade.
        
        Args:
            trade_info: Dictionary containing trade information
        """
        if not self.enabled:
            return
            
        trade_record = {
            'timestamp': datetime.now().isoformat(),
            'asset': trade_info.get('asset'),
            'action': trade_info.get('action'),
            'size': trade_info.get('size'),
            'price': trade_info.get('price'),
            'pnl': trade_info.get('pnl', 0.0),
            'commission': trade_info.get('commission', 0.0),
            'duration': trade_info.get('duration', 0.0)
        }
        
        self.trade_history.append(trade_record)
        
        # Update trade-related metrics
        if trade_record['pnl'] > 0:
            self._add_metric('trading.winning_trades', 1, datetime.now())
        else:
            self._add_metric('trading.losing_trades', 1, datetime.now())
    
    def start_episode(self, episode_id: int) -> None:
        """
        Mark the start of a new episode.
        
        Args:
            episode_id: Unique identifier for the episode
        """
        self.current_episode = episode_id
        self.current_step = 0
        self._add_metric('episodes.started', episode_id, datetime.now())
    
    def end_episode(self, episode_metrics: Dict[str, Any]) -> None:
        """
        Mark the end of an episode and log episode metrics.
        
        Args:
            episode_metrics: Dictionary containing episode summary metrics
        """
        timestamp = datetime.now()
        
        for key, value in episode_metrics.items():
            self._add_metric(f'episodes.{key}', value, timestamp)
        
        self._add_metric('episodes.completed', self.current_episode, timestamp)
    
    def step(self) -> None:
        """Mark a single step in the current episode."""
        self.current_step += 1
        self.update_count += 1
        
        # Auto-save periodically
        if self.update_count % self.save_interval == 0:
            self.save_metrics()
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """
        Get current aggregated metrics.
        
        Returns:
            Dictionary containing current metrics summary
        """
        current_metrics = {}
        
        # Calculate recent averages
        if self.reward_history:
            current_metrics['avg_reward_recent'] = np.mean(list(self.reward_history)[-100:])
            current_metrics['reward_trend'] = self._calculate_trend(list(self.reward_history)[-50:])
        
        if self.loss_history:
            current_metrics['avg_loss_recent'] = np.mean(list(self.loss_history)[-100:])
            current_metrics['loss_trend'] = self._calculate_trend(list(self.loss_history)[-50:])
        
        # Trading performance
        if self.trade_history:
            recent_trades = self.trade_history[-50:]
            winning_trades = sum(1 for trade in recent_trades if trade['pnl'] > 0)
            current_metrics['recent_win_rate'] = winning_trades / len(recent_trades)
            current_metrics['recent_avg_pnl'] = np.mean([trade['pnl'] for trade in recent_trades])
        
        # Session info
        current_metrics['session_duration'] = (datetime.now() - self.session_start).total_seconds()
        current_metrics['current_episode'] = self.current_episode
        current_metrics['current_step'] = self.current_step
        current_metrics['total_updates'] = self.update_count
        
        return current_metrics
    
    def get_metric_history(self, metric_name: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get historical data for a specific metric.
        
        Args:
            metric_name: Name of the metric to retrieve
            limit: Maximum number of records to return
            
        Returns:
            List of metric records with timestamps and values
        """
        if metric_name not in self.metrics:
            return []
        
        history = list(self.metrics[metric_name])
        if limit:
            history = history[-limit:]
        
        return history
    
    def save_metrics(self, filename: Optional[str] = None) -> None:
        """
        Save current metrics to file.
        
        Args:
            filename: Optional custom filename
        """
        if not self.enabled:
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{timestamp}.json"
        
        filepath = self.base_path / filename
        
        # Prepare data for saving
        save_data = {
            'session_info': {
                'start_time': self.session_start.isoformat(),
                'save_time': datetime.now().isoformat(),
                'total_updates': self.update_count,
                'current_episode': self.current_episode
            },
            'current_metrics': self.get_current_metrics(),
            'trade_history': self.trade_history[-1000:],  # Save last 1000 trades
            'metric_summaries': self._generate_metric_summaries()
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(save_data, f, indent=2, default=str)
            
            self.last_save_time = time.time()
            logger.info(f"Metrics saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save metrics: {str(e)}")
    
    def load_metrics(self, filename: str) -> bool:
        """
        Load metrics from file.
        
        Args:
            filename: Name of the file to load
            
        Returns:
            True if successful, False otherwise
        """
        filepath = self.base_path / filename
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Restore trade history
            if 'trade_history' in data:
                self.trade_history.extend(data['trade_history'])
            
            # Restore session info
            if 'session_info' in data:
                session_info = data['session_info']
                self.update_count = session_info.get('total_updates', 0)
                self.current_episode = session_info.get('current_episode', 0)
            
            logger.info(f"Metrics loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load metrics from {filename}: {str(e)}")
            return False
    
    def reset_metrics(self) -> None:
        """Reset all metrics and start fresh."""
        self.metrics.clear()
        self.trade_history.clear()
        self.reward_history.clear()
        self.loss_history.clear()
        self.aggregated_metrics.clear()
        
        self.current_episode = 0
        self.current_step = 0
        self.update_count = 0
        self.session_start = datetime.now()
        
        logger.info("All metrics have been reset")
    
    def _add_metric(self, name: str, value: Union[int, float], timestamp: datetime) -> None:
        """
        Add a metric value with timestamp.
        
        Args:
            name: Metric name
            value: Metric value
            timestamp: Timestamp for the metric
        """
        metric_record = {
            'timestamp': timestamp.isoformat(),
            'value': value
        }
        
        # Add to deque with max length
        if len(self.metrics[name]) >= self.history_length:
            self.metrics[name].popleft()
        
        self.metrics[name].append(metric_record)
    
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
        
        if slope > 0.01:
            return 'up'
        elif slope < -0.01:
            return 'down'
        else:
            return 'stable'
    
    def _generate_metric_summaries(self) -> Dict[str, Dict[str, float]]:
        """
        Generate summary statistics for all metrics.
        
        Returns:
            Dictionary containing summary statistics for each metric
        """
        summaries = {}
        
        for metric_name, metric_data in self.metrics.items():
            if not metric_data:
                continue
            
            values = [record['value'] for record in metric_data if isinstance(record['value'], (int, float))]
            
            if values:
                summaries[metric_name] = {
                    'count': len(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'latest': values[-1] if values else 0
                }
        
        return summaries