"""Regression test for the post-learn Ray financial snapshot."""

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "train_parallel_agents",
    ROOT / "scripts" / "train_parallel_agents.py",
)
TRAINING = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(TRAINING)


class _Logger:
    name_to_value = {}


class _Model:
    def __init__(self):
        self.num_timesteps = 0
        self.ep_info_buffer = []
        self.logger = _Logger()

    def learn(self, *, total_timesteps, callback, reset_num_timesteps):
        del callback, reset_num_timesteps
        self.num_timesteps += total_timesteps


class _VecEnv:
    def __init__(self):
        self.run_pnl = -4.266501973013799
        self.portfolio_pnl = self.run_pnl
        self.last_episode_pnl = 0.0
        self.current_episode_pnl = self.run_pnl
        self.cash = 3.76
        self.equity = 16.23
        self.realized_equity = 20.5 + self.portfolio_pnl
        self.open_positions = 1
        self.calls = []

    def env_method(self, name, *args, **kwargs):
        self.calls.append((name, args, kwargs))
        if name == "finalize_open_positions":
            terminal_pnl = 0.020871911946740068
            self.portfolio_pnl += terminal_pnl
            self.cash = 20.5 + self.portfolio_pnl
            self.equity = self.cash
            self.realized_equity = self.cash
            self.open_positions = 0
            return [[{"pnl_net": terminal_pnl}]]
        if name == "_finalize_episode_financial_telemetry":
            terminal_pnl = kwargs["reset_close_pnl"]
            self.current_episode_pnl += terminal_pnl
            self.run_pnl += terminal_pnl
            self.last_episode_pnl = self.current_episode_pnl
            self.current_episode_pnl = 0.0
            return [None]
        raise AssertionError(f"Unexpected env_method call: {name}")


class _MetricsMonitor:
    _LISTS = (
        "sharpe_ratios",
        "portfolio_values",
        "realized_pnls",
        "realized_pnl_steps",
        "realized_pnl_episodes",
        "realized_pnl_episode_currents",
        "realized_pnl_cumulatives",
        "cash_values",
        "equity_values",
        "realized_equity_values",
    )

    def __init__(self, vec_env):
        self.vec_env = vec_env
        self.collect_count = 0
        self.worker_metrics = {
            0: {name: [] for name in self._LISTS}
        }

    def _collect_worker_metrics(self):
        self.collect_count += 1
        metrics = self.worker_metrics[0]
        values = {
            "sharpe_ratios": 0.0,
            "portfolio_values": self.vec_env.equity,
            "realized_pnls": self.vec_env.portfolio_pnl,
            "realized_pnl_steps": 0.0,
            "realized_pnl_episodes": self.vec_env.last_episode_pnl,
            "realized_pnl_episode_currents": self.vec_env.current_episode_pnl,
            "realized_pnl_cumulatives": self.vec_env.run_pnl,
            "cash_values": self.vec_env.cash,
            "equity_values": self.vec_env.equity,
            "realized_equity_values": self.vec_env.realized_equity,
        }
        for name, value in values.items():
            metrics[name].append(value)


def test_final_ray_result_uses_post_learn_terminal_finance():
    worker = object.__new__(TRAINING.ADAN_PBT_Worker)
    vec_env = _VecEnv()
    worker.vec_env = vec_env
    worker.model = _Model()
    worker._callbacks = object()
    worker._metrics_monitor = _MetricsMonitor(vec_env)
    worker._sync_mutable_hyperparameters = lambda: None
    worker.interval_timesteps = 2048
    worker._total_timesteps = 0
    worker._last_checkpoint_step = 0
    worker.checkpoint_dir = "unused"
    worker.learning_rate = 1e-4
    worker.ent_coef = 0.0
    worker.gamma = 0.99
    worker.sl_pct = 0.01
    worker.tp_pct = 0.02
    worker._completed_iterations = 0
    worker._max_iterations = 1

    result = worker.step()

    expected_pnl = -4.245630061067059
    expected_equity = 20.5 + expected_pnl
    assert result["done"] is True
    assert result["timesteps_total"] == 2048
    assert result["realized_pnl"] == expected_pnl
    assert result["realized_pnl_episode"] == expected_pnl
    assert result["realized_pnl_episode_current"] == 0.0
    assert result["realized_pnl_cumulative"] == expected_pnl
    assert result["cash"] == expected_equity
    assert result["equity"] == expected_equity
    assert result["realized_equity"] == expected_equity
    assert worker._metrics_monitor.collect_count == 1
    assert [call[0] for call in vec_env.calls] == [
        "finalize_open_positions",
        "_finalize_episode_financial_telemetry",
    ]
