#!/usr/bin/env python3
"""
Memory Management System for ADAN Trading Bot.

This module provides comprehensive memory management capabilities including:
- Automatic garbage collection triggers
- Memory usage monitoring and alerting
- Tensor memory allocation optimization
- Mixed-precision training support
"""

import gc
import logging
import psutil
import threading
import time
import torch
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
from functools import wraps
import numpy as np


class MemoryPressureLevel(Enum):
    """Memory pressure levels for different management strategies."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MemoryStats:
    """Memory statistics container."""
    total_memory: float
    available_memory: float
    used_memory: float
    memory_percent: float
    gpu_memory_used: Optional[float] = None
    gpu_memory_total: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    tensor_count: int = 0
    tensor_memory: float = 0.0
    timestamp: float = field(default_factory=time.time)


class MemoryManager:
    """
    Comprehensive memory management system for the trading bot.

    Features:
    - Automatic garbage collection based on memory thresholds
    - Memory usage monitoring and alerting
    - GPU memory management for PyTorch tensors
    - Mixed-precision training support
    - Memory leak detection
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Memory Manager.

        Args:
            config: Configuration dictionary with memory management settings
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Memory thresholds (as percentages)
        self.thresholds = {
            MemoryPressureLevel.MEDIUM: config.get('medium_threshold', 70.0),
            MemoryPressureLevel.HIGH: config.get('high_threshold', 85.0),
            MemoryPressureLevel.CRITICAL: config.get('critical_threshold', 95.0)
        }

        # Monitoring settings
        self.monitoring_enabled = config.get('enable_monitoring', True)
        self.monitoring_interval = config.get('monitoring_interval', 30.0)  # seconds
        self.auto_gc_enabled = config.get('enable_auto_gc', True)
        self.gpu_monitoring = config.get('enable_gpu_monitoring', True)

        # Mixed precision settings
        self.mixed_precision_enabled = config.get('enable_mixed_precision', True)
        self.amp_enabled = config.get('enable_amp', True)

        # State tracking
        self.current_pressure_level = MemoryPressureLevel.LOW
        self.memory_history: List[MemoryStats] = []
        self.max_history_size = config.get('max_history_size', 1000)

        # Callbacks for memory events
        self.pressure_callbacks: Dict[MemoryPressureLevel, List[Callable]] = {
            level: [] for level in MemoryPressureLevel
        }

        # Monitoring thread
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        self._lock = threading.RLock()

        # GPU availability check
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu_device_count = torch.cuda.device_count()
            self.logger.info(f"GPU monitoring enabled for {self.gpu_device_count} devices")
        else:
            self.gpu_device_count = 0
            self.logger.info("No GPU available, CPU-only memory monitoring")

        # Initialize mixed precision scaler if enabled
        self.scaler = None
        if self.mixed_precision_enabled and self.gpu_available:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
            self.logger.info("Mixed precision training enabled with AMP")

        # Start monitoring if enabled
        if self.monitoring_enabled:
            self.start_monitoring()

        self.logger.info("Memory Manager initialized")

    def get_memory_stats(self) -> MemoryStats:
        """
        Get current memory statistics.

        Returns:
            MemoryStats: Current memory usage statistics
        """
        # System memory
        memory = psutil.virtual_memory()
        stats = MemoryStats(
            total_memory=memory.total / (1024**3),  # GB
            available_memory=memory.available / (1024**3),  # GB
            used_memory=memory.used / (1024**3),  # GB
            memory_percent=memory.percent
        )

        # GPU memory if available
        if self.gpu_available and self.gpu_monitoring:
            try:
                gpu_memory = torch.cuda.memory_stats()
                allocated = gpu_memory.get('allocated_bytes.all.current', 0)
                reserved = gpu_memory.get('reserved_bytes.all.current', 0)

                # Get total GPU memory
                total_memory = torch.cuda.get_device_properties(0).total_memory

                stats.gpu_memory_used = allocated / (1024**3)  # GB
                stats.gpu_memory_total = total_memory / (1024**3)  # GB
                stats.gpu_memory_percent = (allocated / total_memory) * 100

            except Exception as e:
                self.logger.warning(f"Failed to get GPU memory stats: {e}")

        # Tensor statistics
        if self.gpu_available:
            try:
                stats.tensor_count = len([obj for obj in gc.get_objects()
                                        if isinstance(obj, torch.Tensor)])
                stats.tensor_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
            except Exception as e:
                self.logger.warning(f"Failed to get tensor stats: {e}")

        return stats

    def get_pressure_level(self, memory_percent: float) -> MemoryPressureLevel:
        """
        Determine memory pressure level based on usage percentage.

        Args:
            memory_percent: Memory usage percentage

        Returns:
            MemoryPressureLevel: Current pressure level
        """
        if memory_percent >= self.thresholds[MemoryPressureLevel.CRITICAL]:
            return MemoryPressureLevel.CRITICAL
        elif memory_percent >= self.thresholds[MemoryPressureLevel.HIGH]:
            return MemoryPressureLevel.HIGH
        elif memory_percent >= self.thresholds[MemoryPressureLevel.MEDIUM]:
            return MemoryPressureLevel.MEDIUM
        else:
            return MemoryPressureLevel.LOW

    def trigger_garbage_collection(self, force: bool = False) -> Dict[str, int]:
        """
        Trigger garbage collection with optional force.

        Args:
            force: Force full garbage collection

        Returns:
            Dict with collection statistics
        """
        start_time = time.time()

        # Clear PyTorch cache if GPU is available
        if self.gpu_available:
            torch.cuda.empty_cache()

        # Run garbage collection
        if force:
            # Force full collection of all generations
            collected = {
                'gen0': gc.collect(0),
                'gen1': gc.collect(1),
                'gen2': gc.collect(2)
            }
        else:
            # Normal collection
            collected = {'total': gc.collect()}

        collection_time = time.time() - start_time

        self.logger.info(f"Garbage collection completed in {collection_time:.3f}s, "
                        f"collected: {collected}")

        return collected

    def handle_memory_pressure(self, pressure_level: MemoryPressureLevel):
        """
        Handle memory pressure based on the current level.

        Args:
            pressure_level: Current memory pressure level
        """
        if pressure_level == MemoryPressureLevel.MEDIUM:
            self.logger.warning("Medium memory pressure detected, triggering GC")
            self.trigger_garbage_collection()

        elif pressure_level == MemoryPressureLevel.HIGH:
            self.logger.warning("High memory pressure detected, forcing full GC")
            self.trigger_garbage_collection(force=True)

        elif pressure_level == MemoryPressureLevel.CRITICAL:
            self.logger.error("Critical memory pressure detected, emergency cleanup")
            self.trigger_garbage_collection(force=True)

            # Additional emergency measures
            if self.gpu_available:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        # Execute registered callbacks
        for callback in self.pressure_callbacks[pressure_level]:
            try:
                callback(pressure_level)
            except Exception as e:
                self.logger.error(f"Error in pressure callback: {e}")

    def register_pressure_callback(self, level: MemoryPressureLevel,
                                 callback: Callable[[MemoryPressureLevel], None]):
        """
        Register a callback for memory pressure events.

        Args:
            level: Memory pressure level to monitor
            callback: Function to call when pressure level is reached
        """
        self.pressure_callbacks[level].append(callback)
        self.logger.info(f"Registered callback for {level.value} memory pressure")

    def monitor_memory(self):
        """Memory monitoring loop (runs in separate thread)."""
        self.logger.info("Memory monitoring started")

        while not self._stop_monitoring.is_set():
            try:
                # Get current memory stats
                stats = self.get_memory_stats()

                with self._lock:
                    # Add to history
                    self.memory_history.append(stats)
                    if len(self.memory_history) > self.max_history_size:
                        self.memory_history.pop(0)

                    # Check pressure level
                    new_pressure_level = self.get_pressure_level(stats.memory_percent)

                    # Handle pressure changes
                    if new_pressure_level != self.current_pressure_level:
                        self.logger.info(f"Memory pressure changed: {self.current_pressure_level.value} -> {new_pressure_level.value}")
                        self.current_pressure_level = new_pressure_level

                        if self.auto_gc_enabled:
                            self.handle_memory_pressure(new_pressure_level)

                # Log periodic stats
                if len(self.memory_history) % 10 == 0:  # Every 10 cycles
                    self.logger.debug(f"Memory: {stats.memory_percent:.1f}% "
                                    f"({stats.used_memory:.1f}GB/{stats.total_memory:.1f}GB)")
                    if stats.gpu_memory_percent:
                        self.logger.debug(f"GPU Memory: {stats.gpu_memory_percent:.1f}% "
                                        f"({stats.gpu_memory_used:.1f}GB/{stats.gpu_memory_total:.1f}GB)")

            except Exception as e:
                self.logger.error(f"Error in memory monitoring: {e}")

            # Wait for next monitoring cycle
            self._stop_monitoring.wait(self.monitoring_interval)

        self.logger.info("Memory monitoring stopped")

    def start_monitoring(self):
        """Start memory monitoring in a separate thread."""
        if self._monitoring_thread is None or not self._monitoring_thread.is_alive():
            self._stop_monitoring.clear()
            self._monitoring_thread = threading.Thread(
                target=self.monitor_memory,
                name="MemoryMonitor",
                daemon=True
            )
            self._monitoring_thread.start()
            self.logger.info("Memory monitoring thread started")

    def stop_monitoring(self):
        """Stop memory monitoring."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._stop_monitoring.set()
            self._monitoring_thread.join(timeout=5.0)
            self.logger.info("Memory monitoring stopped")

    def get_memory_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive memory usage summary.

        Returns:
            Dict with memory usage summary
        """
        with self._lock:
            if not self.memory_history:
                return {"error": "No memory history available"}

            recent_stats = self.memory_history[-10:]  # Last 10 measurements

            summary = {
                "current": {
                    "memory_percent": self.memory_history[-1].memory_percent,
                    "used_memory_gb": self.memory_history[-1].used_memory,
                    "available_memory_gb": self.memory_history[-1].available_memory,
                    "pressure_level": self.current_pressure_level.value
                },
                "averages": {
                    "memory_percent": np.mean([s.memory_percent for s in recent_stats]),
                    "used_memory_gb": np.mean([s.used_memory for s in recent_stats])
                },
                "thresholds": {
                    level.value: threshold for level, threshold in self.thresholds.items()
                },
                "monitoring": {
                    "enabled": self.monitoring_enabled,
                    "auto_gc_enabled": self.auto_gc_enabled,
                    "history_size": len(self.memory_history)
                }
            }

            # Add GPU info if available
            if self.gpu_available and self.memory_history[-1].gpu_memory_percent:
                summary["gpu"] = {
                    "memory_percent": self.memory_history[-1].gpu_memory_percent,
                    "used_memory_gb": self.memory_history[-1].gpu_memory_used,
                    "total_memory_gb": self.memory_history[-1].gpu_memory_total,
                    "tensor_count": self.memory_history[-1].tensor_count,
                    "tensor_memory_gb": self.memory_history[-1].tensor_memory
                }

            return summary

    def optimize_for_training(self):
        """Optimize memory settings for training workloads."""
        self.logger.info("Optimizing memory for training workload")

        # Force garbage collection
        self.trigger_garbage_collection(force=True)

        # Set PyTorch memory management
        if self.gpu_available:
            # Enable memory fraction to prevent OOM
            torch.cuda.set_per_process_memory_fraction(0.9)

            # Enable memory pool for faster allocation
            torch.backends.cudnn.benchmark = True

            # Clear cache
            torch.cuda.empty_cache()

        # Adjust monitoring for training
        original_interval = self.monitoring_interval
        self.monitoring_interval = min(self.monitoring_interval, 10.0)  # More frequent during training

        self.logger.info(f"Memory optimization complete, monitoring interval: {self.monitoring_interval}s")

        return original_interval

    def restore_monitoring_interval(self, original_interval: float):
        """Restore original monitoring interval after training."""
        self.monitoring_interval = original_interval
        self.logger.info(f"Monitoring interval restored to {self.monitoring_interval}s")

    def create_mixed_precision_context(self):
        """
        Create a context manager for mixed precision training.

        Returns:
            Context manager for mixed precision operations
        """
        if self.mixed_precision_enabled and self.scaler:
            return torch.cuda.amp.autocast(enabled=True)
        else:
            # Return a no-op context manager
            from contextlib import nullcontext
            return nullcontext()

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Scale loss for mixed precision training.

        Args:
            loss: Loss tensor to scale

        Returns:
            Scaled loss tensor
        """
        if self.scaler:
            return self.scaler.scale(loss)
        return loss

    def step_optimizer(self, optimizer: torch.optim.Optimizer):
        """
        Step optimizer with mixed precision support.

        Args:
            optimizer: PyTorch optimizer to step
        """
        if self.scaler:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

    def memory_efficient_decorator(self, clear_cache: bool = True):
        """
        Decorator for memory-efficient function execution.

        Args:
            clear_cache: Whether to clear GPU cache after execution

        Returns:
            Decorator function
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Pre-execution cleanup
                if self.auto_gc_enabled:
                    gc.collect()

                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    # Post-execution cleanup
                    if clear_cache and self.gpu_available:
                        torch.cuda.empty_cache()

                    if self.auto_gc_enabled:
                        gc.collect()

            return wrapper
        return decorator

    def shutdown(self):
        """Shutdown the memory manager and cleanup resources."""
        self.logger.info("Shutting down Memory Manager")

        # Stop monitoring
        self.stop_monitoring()

        # Final cleanup
        self.trigger_garbage_collection(force=True)

        if self.gpu_available:
            torch.cuda.empty_cache()

        self.logger.info("Memory Manager shutdown complete")


# Utility functions for memory management
def get_tensor_memory_usage() -> Dict[str, float]:
    """
    Get detailed tensor memory usage information.

    Returns:
        Dict with tensor memory statistics
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    try:
        stats = torch.cuda.memory_stats()
        return {
            "allocated_gb": stats.get('allocated_bytes.all.current', 0) / (1024**3),
            "reserved_gb": stats.get('reserved_bytes.all.current', 0) / (1024**3),
            "max_allocated_gb": stats.get('allocated_bytes.all.peak', 0) / (1024**3),
            "max_reserved_gb": stats.get('reserved_bytes.all.peak', 0) / (1024**3),
            "num_alloc_retries": stats.get('num_alloc_retries', 0),
            "num_ooms": stats.get('num_ooms', 0)
        }
    except Exception as e:
        return {"error": str(e)}


def memory_profile(func):
    """
    Decorator to profile memory usage of a function.

    Args:
        func: Function to profile

    Returns:
        Decorated function with memory profiling
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get initial memory
        initial_memory = psutil.virtual_memory().percent
        initial_gpu = None

        if torch.cuda.is_available():
            initial_gpu = torch.cuda.memory_allocated() / (1024**3)

        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # Get final memory
            end_time = time.time()
            final_memory = psutil.virtual_memory().percent
            final_gpu = None

            if torch.cuda.is_available():
                final_gpu = torch.cuda.memory_allocated() / (1024**3)

            # Log memory usage
            logger = logging.getLogger(func.__module__)
            logger.info(f"Memory profile for {func.__name__}:")
            logger.info(f"  Execution time: {end_time - start_time:.3f}s")
            logger.info(f"  Memory change: {final_memory - initial_memory:.1f}%")

            if initial_gpu is not None and final_gpu is not None:
                logger.info(f"  GPU memory change: {final_gpu - initial_gpu:.3f}GB")

    return wrapper
