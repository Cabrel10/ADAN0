#!/usr/bin/env python3
"""
Benchmark script to measure performance improvements with TrainingOrchestrator
vs single environment training.
"""

import time
import psutil
import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Any
from unittest.mock import MagicMock

# Add src to path
sys.path.insert(0, '/home/morningstar/Documents/trading/ADAN/src')

# Mock dependencies
sys.modules['adan_trading_bot.online_learning_agent'] = MagicMock()
sys.modules['adan_trading_bot.environment.multi_asset_env'] = MagicMock()

from adan_trading_bot.training_orchestrator import TrainingOrchestrator


class MockAgent:
    """Mock agent that simulates training workload"""

    def __init__(self, model=None, env=None, config=None, experience_buffer=None):
        self.model = model
        self.env = env
        self.config = config
        self.experience_buffer = experience_buffer
        self.training_steps = config.get('training_steps', 1000) if config else 1000

    def run(self):
        """Simulate training workload"""
        for step in range(self.training_steps):
            # Simulate computation
            dummy_calc = sum(i * 0.001 for i in range(100))

            # Simulate experience buffer interaction
            if self.experience_buffer and hasattr(self.experience_buffer, 'add'):
                self.experience_buffer.add(
                    obs=f"obs_{step}",
                    action=step % 3,
                    reward=dummy_calc,
                    next_obs=f"next_obs_{step}",
                    done=False,
                    priority=0.5
                )

            # Small delay to simulate real training
            time.sleep(0.001)


class PerformanceBenchmark:
    """Benchmark orchestration performance"""

    def __init__(self):
        self.results = {}

    def measure_resource_usage(self, func, *args, **kwargs):
        """Measure CPU and memory usage during function execution"""
        process = psutil.Process(os.getpid())

        # Initial measurements
        cpu_before = process.cpu_percent()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        start_time = time.time()

        # Execute function
        result = func(*args, **kwargs)

        end_time = time.time()

        # Final measurements
        cpu_after = process.cpu_percent()
        memory_after = process.memory_info().rss / 1024 / 1024  # MB

        return {
            'result': result,
            'execution_time': end_time - start_time,
            'cpu_usage': max(cpu_after, cpu_before),
            'memory_usage_mb': memory_after,
            'memory_delta_mb': memory_after - memory_before
        }

    def benchmark_single_environment(self, training_steps: int = 1000):
        """Benchmark single environment training"""
        print("🔄 Benchmarking single environment training...")

        config = {'training_steps': training_steps}

        def single_training():
            agent = MockAgent(config=config)
            agent.run()
            return 1  # 1 agent trained

        metrics = self.measure_resource_usage(single_training)

        self.results['single_env'] = {
            'num_agents': 1,
            'training_steps_per_agent': training_steps,
            'total_training_steps': training_steps,
            **metrics
        }

        print(f"✅ Single env completed in {metrics['execution_time']:.2f}s")
        return metrics

    def benchmark_orchestrated_training(self, num_envs: int = 4, training_steps: int = 1000):
        """Benchmark orchestrated multi-environment training"""
        print(f"🔄 Benchmarking orchestrated training with {num_envs} environments...")

        env_configs = [{'id': f'env{i}'} for i in range(num_envs)]
        agent_config = {'training_steps': training_steps, 'batch_size': 64}

        def orchestrated_training():
            orchestrator = TrainingOrchestrator(
                env_configs=env_configs,
                agent_config=agent_config,
                agent_class=MockAgent
            )

            orchestrator.setup_environments()
            orchestrator.train()

            return len(orchestrator.agents)

        metrics = self.measure_resource_usage(orchestrated_training)

        self.results['orchestrated'] = {
            'num_agents': num_envs,
            'training_steps_per_agent': training_steps,
            'total_training_steps': training_steps * num_envs,
            **metrics
        }

        print(f"✅ Orchestrated training completed in {metrics['execution_time']:.2f}s")
        return metrics

    def calculate_performance_gains(self):
        """Calculate performance improvements"""
        if 'single_env' not in self.results or 'orchestrated' not in self.results:
            return None

        single = self.results['single_env']
        orchestrated = self.results['orchestrated']

        # Calculate theoretical vs actual speedup
        theoretical_time = single['execution_time'] * orchestrated['num_agents']
        actual_time = orchestrated['execution_time']

        gains = {
            'theoretical_sequential_time': theoretical_time,
            'actual_orchestrated_time': actual_time,
            'speedup_factor': theoretical_time / actual_time,
            'time_saved_seconds': theoretical_time - actual_time,
            'time_saved_percentage': ((theoretical_time - actual_time) / theoretical_time) * 100,
            'efficiency': (orchestrated['total_training_steps'] / actual_time) / (single['total_training_steps'] / single['execution_time']),
            'memory_overhead_mb': orchestrated['memory_usage_mb'] - single['memory_usage_mb'],
            'cpu_overhead': orchestrated['cpu_usage'] - single['cpu_usage']
        }

        self.results['performance_gains'] = gains
        return gains

    def run_full_benchmark(self, training_steps: int = 1000):
        """Run complete benchmark suite"""
        print("🚀 Starting orchestration performance benchmark...")
        print(f"Training steps per agent: {training_steps}")
        print("-" * 60)

        # Benchmark single environment
        self.benchmark_single_environment(training_steps)

        # Benchmark orchestrated training
        self.benchmark_orchestrated_training(4, training_steps)

        # Calculate gains
        gains = self.calculate_performance_gains()

        # Print results
        self.print_results()

        # Save results
        self.save_results()

        return self.results

    def print_results(self):
        """Print benchmark results"""
        print("\n" + "=" * 60)
        print("📊 BENCHMARK RESULTS")
        print("=" * 60)

        if 'single_env' in self.results:
            single = self.results['single_env']
            print(f"\n🔹 Single Environment:")
            print(f"   Execution time: {single['execution_time']:.2f}s")
            print(f"   Memory usage: {single['memory_usage_mb']:.1f} MB")
            print(f"   CPU usage: {single['cpu_usage']:.1f}%")

        if 'orchestrated' in self.results:
            orch = self.results['orchestrated']
            print(f"\n🔹 Orchestrated (4 environments):")
            print(f"   Execution time: {orch['execution_time']:.2f}s")
            print(f"   Memory usage: {orch['memory_usage_mb']:.1f} MB")
            print(f"   CPU usage: {orch['cpu_usage']:.1f}%")

        if 'performance_gains' in self.results:
            gains = self.results['performance_gains']
            print(f"\n🚀 Performance Gains:")
            print(f"   Speedup factor: {gains['speedup_factor']:.2f}x")
            print(f"   Time saved: {gains['time_saved_seconds']:.2f}s ({gains['time_saved_percentage']:.1f}%)")
            print(f"   Training efficiency: {gains['efficiency']:.2f}x")
            print(f"   Memory overhead: {gains['memory_overhead_mb']:.1f} MB")
            print(f"   CPU overhead: {gains['cpu_overhead']:.1f}%")

        print("\n" + "=" * 60)

    def save_results(self):
        """Save benchmark results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"orchestration_benchmark_{timestamp}.json"
        filepath = os.path.join("logs", filename)

        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)

        # Add metadata
        self.results['metadata'] = {
            'timestamp': timestamp,
            'python_version': sys.version,
            'cpu_count': psutil.cpu_count(),
            'total_memory_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024
        }

        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"📁 Results saved to: {filepath}")


def main():
    """Main benchmark execution"""
    benchmark = PerformanceBenchmark()

    # Run benchmark with different training step counts
    print("Running benchmark with 1000 training steps per agent...")
    results = benchmark.run_full_benchmark(training_steps=1000)

    return results


if __name__ == "__main__":
    main()
