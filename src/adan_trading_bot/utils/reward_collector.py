"""
Reward/Penalty Collector for ADAN Trading Bot - COMPLETE VERSION
Collects detailed reward breakdown per worker per step for debugging
Captures 50+ metrics including all reward/penalty components
"""
import json
import os
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, Any, Optional, List
import numpy as np


class RewardCollector:
    """Collecte detaillee des rewards et penalites par worker par step."""
    
    def __init__(self, log_dir: str = None):
        # Configurable log directory with safe fallback (no hardcoded /mnt/new_data)
        if log_dir is None:
            log_dir = os.environ.get("ADAN_REWARD_LOG_DIR", "./logs/rewards")
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.step_data = defaultdict(list)
        self.episode_data = defaultdict(list)
        self.current_episode = defaultdict(int)
        self.current_step = defaultdict(int)
        self.log_files = {}
        self.global_stats = {"total_steps_logged": 0, "total_episodes_logged": 0, "inconsistencies_found": 0}
        
    def _get_log_file(self, worker_id: str) -> str:
        if worker_id not in self.log_files:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"worker_{worker_id}_rewards_{timestamp}.jsonl"
            filepath = os.path.join(self.log_dir, filename)
            self.log_files[worker_id] = filepath
        return self.log_files[worker_id]
    
    def log_step(self, worker_id: str, step: int, episode: int, profile: str,
                 reward: float, reward_breakdown: Dict[str, float],
                 realized_pnl: float, unrealized_pnl: float, total_commission: float, pnl_net: float,
                 cash_before: float, cash_after: float, portfolio_value: float,
                 initial_capital: float, equity: float, balance: float,
                 num_positions: int, num_positions_before: int,
                 open_positions_details: List[Dict], closed_positions_details: List[Dict],
                 position_sizes: Dict[str, float], position_limits: Dict[str, int],
                 action_taken: str, action_raw: np.ndarray,
                 sl_triggered: bool, tp_triggered: bool,
                 trade_duration_seconds: float, trade_reason: str,
                 drawdown_pct: float, max_drawdown_pct: float,
                 sharpe_ratio: float, sortino_ratio: float, profit_factor: float, calmar_ratio: float,
                 win_rate: float, volatility: float, risk_level: float,
                 trades_count_5m: int, trades_count_1h: int, trades_count_4h: int, trades_count_daily: int,
                 trades_total: int, winning_trades: int, losing_trades: int, neutral_trades: int,
                 trade_attempts: int, invalid_trade_attempts: int, steps_since_last_trade: int,
                 inaction_penalty: float, survival_bonus: float, trade_cost_penalty: float,
                 drawdown_penalty: float, position_limit_penalty: float,
                 outcome_tp_bonus: float, outcome_sl_penalty: float, passivity_penalty: float,
                 early_close_bonus: float, duration_penalty: float, capacity_reward: float,
                 frequency_penalty: float, consistency_bonus: float, invalid_sell_penalty: float,
                 sharpe_excellence_bonus: float, winning_streak_bonus: float, confluence_bonus: float,
                 consistency_excellence_bonus: float, profit_factor_bonus: float,
                 chunk_progression_penalty: float, winning_streak_count: int, consecutive_losses: int,
                 failsafe_triggered: bool, ev_norm: float,
                 tier_name: str, tier_level: int, tier_progress_pct: float,
                 regime: str, market_regime: str, chunk_number: int, step_in_chunk: int, total_chunks: int,
                 capital_usage_pct: float, cash_utilization: float, capacity_usage_pct: float,
                 margin_used: float, margin_available: float, leverage_used: float, buying_power: float,
                 action_buy_score: float, action_sell_score: float, action_hold_score: float,
                 action_size_pct: float, action_confidence: float,
                 current_price: float, entry_price: float, stop_loss_price: float, take_profit_price: float,
                 unrealized_pnl_pct: float, timestamp: Optional[float] = None, **kwargs) -> None:
        """Log comprehensive step data with 50+ metrics."""
        try:
            data = {
                "timestamp": timestamp or time.time(),
                "datetime": datetime.now().isoformat(),
                "worker_id": str(worker_id),
                "step": int(step),
                "episode": int(episode),
                "profile": str(profile),
                "reward": {
                    "total": float(reward),
                    "breakdown": reward_breakdown,
                    "raw_before_symlog": reward_breakdown.get("raw", 0.0) if reward_breakdown else 0.0,
                },
                "pnl": {
                    "realized": float(realized_pnl),
                    "unrealized": float(unrealized_pnl),
                    "total_commission": float(total_commission),
                    "net": float(pnl_net),
                    "unrealized_pct": float(unrealized_pnl_pct),
                },
                "portfolio": {
                    "cash_before": float(cash_before),
                    "cash_after": float(cash_after),
                    "cash_delta": float(cash_after - cash_before),
                    "total_value": float(portfolio_value),
                    "initial_capital": float(initial_capital),
                    "equity": float(equity),
                    "balance": float(balance),
                    "return_pct": ((portfolio_value - initial_capital) / initial_capital * 100) if initial_capital > 0 else 0.0,
                },
                "positions": {
                    "count": int(num_positions),
                    "count_before": int(num_positions_before),
                    "delta": int(num_positions - num_positions_before),
                    "open_details": open_positions_details if open_positions_details else [],
                    "closed_details": closed_positions_details if closed_positions_details else [],
                    "sizes": position_sizes if position_sizes else {},
                    "limits": position_limits if position_limits else {},
                },
                "action": {
                    "type": str(action_taken),
                    "raw": action_raw.tolist() if isinstance(action_raw, np.ndarray) else action_raw,
                    "buy_score": float(action_buy_score),
                    "sell_score": float(action_sell_score),
                    "hold_score": float(action_hold_score),
                    "size_pct": float(action_size_pct),
                    "confidence": float(action_confidence),
                },
                "triggers": {
                    "sl_triggered": bool(sl_triggered),
                    "tp_triggered": bool(tp_triggered),
                    "duration_seconds": float(trade_duration_seconds),
                    "reason": str(trade_reason),
                },
                "risk": {
                    "drawdown_pct": float(drawdown_pct),
                    "max_drawdown_pct": float(max_drawdown_pct),
                    "sharpe_ratio": float(sharpe_ratio),
                    "sortino_ratio": float(sortino_ratio),
                    "profit_factor": float(profit_factor),
                    "calmar_ratio": float(calmar_ratio),
                    "win_rate": float(win_rate),
                    "volatility": float(volatility),
                    "risk_level": float(risk_level),
                },
                "frequency": {
                    "trades_5m": int(trades_count_5m),
                    "trades_1h": int(trades_count_1h),
                    "trades_4h": int(trades_count_4h),
                    "trades_daily": int(trades_count_daily),
                    "total": int(trades_total),
                    "winning": int(winning_trades),
                    "losing": int(losing_trades),
                    "neutral": int(neutral_trades),
                    "attempts": int(trade_attempts),
                    "invalid_attempts": int(invalid_trade_attempts),
                    "steps_since_last_trade": int(steps_since_last_trade),
                },
                "penalties": {
                    "inaction": float(inaction_penalty),
                    "trade_cost": float(trade_cost_penalty),
                    "drawdown": float(drawdown_penalty),
                    "position_limit": float(position_limit_penalty),
                    "passivity": float(passivity_penalty),
                    "duration": float(duration_penalty),
                    "frequency": float(frequency_penalty),
                    "invalid_sell": float(invalid_sell_penalty),
                    "chunk_progression": float(chunk_progression_penalty),
                },
                "bonuses": {
                    "survival": float(survival_bonus),
                    "outcome_tp": float(outcome_tp_bonus),
                    "early_close": float(early_close_bonus),
                    "capacity": float(capacity_reward),
                    "consistency": float(consistency_bonus),
                    "sharpe_excellence": float(sharpe_excellence_bonus),
                    "winning_streak": float(winning_streak_bonus),
                    "confluence": float(confluence_bonus),
                    "consistency_excellence": float(consistency_excellence_bonus),
                    "profit_factor": float(profit_factor_bonus),
                },
                "excellence": {
                    "winning_streak": int(winning_streak_count),
                    "consecutive_losses": int(consecutive_losses),
                    "failsafe_triggered": bool(failsafe_triggered),
                    "ev_norm": float(ev_norm),
                },
                "tier": {
                    "name": str(tier_name),
                    "level": int(tier_level),
                    "progress_pct": float(tier_progress_pct),
                },
                "market": {
                    "regime": str(regime),
                    "market_regime": str(market_regime),
                    "chunk": int(chunk_number),
                    "step_in_chunk": int(step_in_chunk),
                    "total_chunks": int(total_chunks),
                },
                "capital": {
                    "usage_pct": float(capital_usage_pct),
                    "cash_utilization": float(cash_utilization),
                    "capacity_usage_pct": float(capacity_usage_pct),
                    "margin_used": float(margin_used),
                    "margin_available": float(margin_available),
                    "leverage": float(leverage_used),
                    "buying_power": float(buying_power),
                },
                "prices": {
                    "current": float(current_price),
                    "entry": float(entry_price),
                    "stop_loss": float(stop_loss_price),
                    "take_profit": float(take_profit_price),
                },
                "inconsistency_flags": {
                    "positive_reward_negative_pnl": reward > 0 and realized_pnl < 0,
                    "zero_cash_positive_reward": cash_after <= 0 and reward > 0,
                    "high_reward_low_portfolio": reward > 1 and portfolio_value < initial_capital * 0.5,
                    "sl_tp_both_triggered": sl_triggered and tp_triggered,
                    "excessive_drawdown": drawdown_pct > 20,
                    "negative_cash": cash_after < 0,
                },
            }
            
            log_file = self._get_log_file(worker_id)
            with open(log_file, 'a') as f:
                f.write(json.dumps(data) + '\n')
            
            self.step_data[worker_id].append(data)
            self.global_stats["total_steps_logged"] += 1
            
        except Exception as e:
            error_file = os.path.join(self.log_dir, "collector_errors.log")
            with open(error_file, 'a') as f:
                f.write(f"[{datetime.now().isoformat()}] Error logging step for worker {worker_id}: {e}\n")
    
    def log_episode_summary(self, worker_id: str, episode: int, summary: Dict[str, Any]) -> None:
        """Log episode summary data."""
        try:
            data = {
                "type": "episode_summary",
                "timestamp": time.time(),
                "datetime": datetime.now().isoformat(),
                "worker_id": str(worker_id),
                "episode": int(episode),
                "summary": summary
            }
            log_file = self._get_log_file(worker_id)
            with open(log_file, 'a') as f:
                f.write(json.dumps(data) + '\n')
            self.episode_data[worker_id].append(data)
            self.global_stats["total_episodes_logged"] += 1
        except Exception:
            pass
    
    def get_worker_data(self, worker_id: str) -> List[Dict]:
        """Get all logged data for a specific worker."""
        return self.step_data.get(worker_id, [])
    
    def analyze_inconsistencies(self, worker_id: Optional[str] = None) -> Dict[str, Any]:
        """Analyze logged data for reward/cash/PnL inconsistencies."""
        inconsistencies = []
        workers_to_check = [worker_id] if worker_id else list(self.step_data.keys())
        
        for wid in workers_to_check:
            for data in self.step_data.get(wid, []):
                flags = data.get("inconsistency_flags", {})
                
                if flags.get("positive_reward_negative_pnl"):
                    inconsistencies.append({
                        "type": "positive_reward_negative_pnl",
                        "worker_id": wid, "step": data["step"], "episode": data["episode"],
                        "reward": data["reward"]["total"], "realized_pnl": data["pnl"]["realized"],
                        "cash_after": data["portfolio"]["cash_after"], "severity": "HIGH"
                    })
                
                if flags.get("zero_cash_positive_reward"):
                    inconsistencies.append({
                        "type": "zero_cash_positive_reward",
                        "worker_id": wid, "step": data["step"], "episode": data["episode"],
                        "reward": data["reward"]["total"], "cash_after": data["portfolio"]["cash_after"],
                        "severity": "CRITICAL"
                    })
                
                if flags.get("high_reward_low_portfolio"):
                    inconsistencies.append({
                        "type": "high_reward_low_portfolio",
                        "worker_id": wid, "step": data["step"], "episode": data["episode"],
                        "reward": data["reward"]["total"], "portfolio_value": data["portfolio"]["total_value"],
                        "severity": "MEDIUM"
                    })
                
                if flags.get("excessive_drawdown"):
                    inconsistencies.append({
                        "type": "excessive_drawdown",
                        "worker_id": wid, "step": data["step"], "episode": data["episode"],
                        "drawdown_pct": data["risk"]["drawdown_pct"], "severity": "HIGH"
                    })
        
        self.global_stats["inconsistencies_found"] = len(inconsistencies)
        
        return {
            "total_inconsistencies": len(inconsistencies),
            "by_type": self._group_by_type(inconsistencies),
            "by_worker": self._group_by_worker(inconsistencies),
            "critical_count": sum(1 for i in inconsistencies if i["severity"] == "CRITICAL"),
            "high_count": sum(1 for i in inconsistencies if i["severity"] == "HIGH"),
            "details": inconsistencies[:50]
        }
    
    def _group_by_type(self, inconsistencies: List[Dict]) -> Dict[str, int]:
        result = defaultdict(int)
        for inc in inconsistencies:
            result[inc["type"]] += 1
        return dict(result)
    
    def _group_by_worker(self, inconsistencies: List[Dict]) -> Dict[str, int]:
        result = defaultdict(int)
        for inc in inconsistencies:
            result[inc["worker_id"]] += 1
        return dict(result)
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate comprehensive analysis report."""
        lines = []
        lines.append("="*80)
        lines.append("REWARD COLLECTOR ANALYSIS REPORT")
        lines.append("="*80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Log directory: {self.log_dir}")
        lines.append("")
        lines.append("-"*80)
        lines.append("GLOBAL STATISTICS")
        lines.append("-"*80)
        lines.append(f"Total steps logged: {self.global_stats['total_steps_logged']}")
        lines.append(f"Total episodes logged: {self.global_stats['total_episodes_logged']}")
        lines.append(f"Workers tracked: {len(self.step_data)}")
        lines.append(f"Total inconsistencies found: {self.global_stats['inconsistencies_found']}")
        lines.append("")
        
        analysis = self.analyze_inconsistencies()
        lines.append("-"*80)
        lines.append("INCONSISTENCY ANALYSIS")
        lines.append("-"*80)
        lines.append(f"Critical issues: {analysis['critical_count']}")
        lines.append(f"High severity issues: {analysis['high_count']}")
        lines.append("")
        
        if analysis['by_type']:
            lines.append("By Type:")
            for inc_type, count in sorted(analysis['by_type'].items(), key=lambda x: -x[1]):
                lines.append(f"  {inc_type}: {count}")
        lines.append("")
        
        if analysis['by_worker']:
            lines.append("By Worker:")
            for worker, count in sorted(analysis['by_worker'].items(), key=lambda x: -x[1]):
                lines.append(f"  Worker {worker}: {count}")
        lines.append("")
        
        if analysis['details']:
            lines.append("-"*80)
            lines.append("DETAILED INCONSISTENCIES (first 50)")
            lines.append("-"*80)
            for i, inc in enumerate(analysis['details'][:50], 1):
                lines.append(f"\n{i}. {inc['type'].upper()} | Severity: {inc['severity']}")
                lines.append(f"   Worker {inc['worker_id']} | Episode {inc['episode']} | Step {inc['step']}")
                lines.append(f"   Reward: {inc.get('reward', 'N/A'):+.4f}")
                if 'realized_pnl' in inc:
                    lines.append(f"   Realized PnL: {inc['realized_pnl']:+.2f}")
                if 'cash_after' in inc:
                    lines.append(f"   Cash After: {inc['cash_after']:.2f}")
                if 'drawdown_pct' in inc:
                    lines.append(f"   Drawdown: {inc['drawdown_pct']:.2f}%")
        
        lines.append("")
        lines.append("="*80)
        
        report_text = "\n".join(lines)
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
        return report_text
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for all workers."""
        stats = {"workers": {}, "global": self.global_stats.copy()}
        for worker_id, data_list in self.step_data.items():
            if not data_list:
                continue
            rewards = [d["reward"]["total"] for d in data_list]
            pnls = [d["pnl"]["realized"] for d in data_list]
            cash_deltas = [d["portfolio"]["cash_delta"] for d in data_list]
            stats["workers"][worker_id] = {
                "steps_logged": len(data_list),
                "avg_reward": np.mean(rewards) if rewards else 0.0,
                "std_reward": np.std(rewards) if rewards else 0.0,
                "min_reward": min(rewards) if rewards else 0.0,
                "max_reward": max(rewards) if rewards else 0.0,
                "total_realized_pnl": sum(pnls),
                "avg_cash_delta": np.mean(cash_deltas) if cash_deltas else 0.0,
                "positive_reward_count": sum(1 for r in rewards if r > 0),
                "negative_reward_count": sum(1 for r in rewards if r < 0),
            }
        return stats


# Singleton instance pour usage global
_reward_collector: Optional[RewardCollector] = None


def get_reward_collector(log_dir: Optional[str] = None) -> RewardCollector:
    """Get or create global reward collector instance."""
    global _reward_collector
    if _reward_collector is None:
        effective_dir = log_dir or os.environ.get("ADAN_REWARD_LOG_DIR", "./logs/rewards")
        _reward_collector = RewardCollector(log_dir=effective_dir)
    return _reward_collector


def reset_reward_collector():
    """Reset the global reward collector."""
    global _reward_collector
    _reward_collector = None


# Variable pour verifier la disponibilite
REWARD_COLLECTOR_AVAILABLE = True
