#!/usr/bin/env python3
"""
System metrics monitoring module for the ADAN trading bot.

This module provides comprehensive tracking of system-level metrics including
CPU, memory, and GPU usage.
"""

import os
import time
import psutil
import platform
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

# Try to import GPUtil for GPU metrics
try:
    import GPUtil
except ImportError:
    GPUtil = None

logger = logging.getLogger(__name__)

class SystemMetricsCollector:
    """
    Collects and tracks system-level metrics including CPU, memory, and GPU usage.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the system metrics collector.

        Args:
            config: Configuration dictionary for the metrics collector
        """
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
        self.update_interval = self.config.get('update_interval', 5.0)  # seconds
        self.last_update = 0.0

        # Initialize metrics storage
        self.metrics = {}
        self.gpu_available = False

        # Check for GPU availability
        self._check_gpu_availability()

        logger.info(f"SystemMetricsCollector initialized. GPU available: {self.gpu_available}")

    def _check_gpu_availability(self) -> None:
        """Check if GPU is available and initialize GPU metrics if possible."""
        self.gpu_available = False
        if GPUtil is not None:
            try:
                gpus = GPUtil.getGPUs()
                self.gpu_available = len(gpus) > 0
                if self.gpu_available:
                    self.metrics['gpu'] = {
                        'count': len(gpus),
                        'devices': [{
                            'id': gpu.id,
                            'name': gpu.name,
                            'load': 0.0,
                            'memory_used': 0.0,
                            'memory_total': gpu.memoryTotal,
                            'temperature': 0.0
                        } for gpu in gpus]
                    }
            except Exception as e:
                logger.warning(f"Could not initialize GPU monitoring: {e}")

    def update_metrics(self) -> Dict[str, Any]:
        """
        Update all system metrics.

        Returns:
            Dictionary containing the latest system metrics
        """
        if not self.enabled:
            return {}

        current_time = time.time()
        if current_time - self.last_update < self.update_interval:
            return self.metrics

        # Update CPU metrics
        self._update_cpu_metrics()

        # Update memory metrics
        self._update_memory_metrics()

        # Update GPU metrics if available
        if self.gpu_available:
            self._update_gpu_metrics()

        # Update system info
        self._update_system_info()

        self.last_update = current_time
        return self.metrics

    def _update_cpu_metrics(self) -> None:
        """Update CPU-related metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()

            self.metrics['cpu'] = {
                'usage_percent': cpu_percent,
                'cores': {
                    'physical': psutil.cpu_count(logical=False),
                    'logical': cpu_count,
                },
                'frequency': {
                    'current': cpu_freq.current if cpu_freq else None,
                    'min': cpu_freq.min if cpu_freq else None,
                    'max': cpu_freq.max if cpu_freq else None,
                },
                'load_avg': {
                    '1min': os.getloadavg()[0] if hasattr(os, 'getloadavg') else None,
                    '5min': os.getloadavg()[1] if hasattr(os, 'getloadavg') and len(os.getloadavg()) > 1 else None,
                    '15min': os.getloadavg()[2] if hasattr(os, 'getloadavg') and len(os.getloadavg()) > 2 else None
                },
                'times': dict(psutil.cpu_times_percent()._asdict())
            }
        except Exception as e:
            logger.error(f"Error updating CPU metrics: {e}")

    def _update_memory_metrics(self) -> None:
        """Update memory-related metrics."""
        try:
            virtual_mem = psutil.virtual_memory()
            swap_mem = psutil.swap_memory()

            self.metrics['memory'] = {
                'virtual': {
                    'total': virtual_mem.total,
                    'available': virtual_mem.available,
                    'used': virtual_mem.used,
                    'free': virtual_mem.free,
                    'percent': virtual_mem.percent,
                },
                'swap': {
                    'total': swap_mem.total,
                    'used': swap_mem.used,
                    'free': swap_mem.free,
                    'percent': swap_mem.percent,
                    'sin': swap_mem.sin,
                    'sout': swap_mem.sout
                }
            }
        except Exception as e:
            logger.error(f"Error updating memory metrics: {e}")

    def _update_gpu_metrics(self) -> None:
        """Update GPU-related metrics if a GPU is available."""
        if not self.gpu_available or GPUtil is None:
            return

        try:
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                if i < len(self.metrics['gpu']['devices']):
                    self.metrics['gpu']['devices'][i].update({
                        'load': gpu.load * 100,  # Convert to percentage
                        'memory_used': gpu.memoryUsed,
                        'memory_free': gpu.memoryFree,
                        'memory_util': gpu.memoryUtil * 100,  # Convert to percentage
                        'temperature': gpu.temperature
                    })
        except Exception as e:
            logger.error(f"Error updating GPU metrics: {e}")

    def _update_system_info(self) -> None:
        """Update general system information."""
        try:
            self.metrics['system'] = {
                'platform': {
                    'system': platform.system(),
                    'node': platform.node(),
                    'release': platform.release(),
                    'version': platform.version(),
                    'machine': platform.machine(),
                    'processor': platform.processor(),
                },
                'python': {
                    'version': platform.python_version(),
                    'implementation': platform.python_implementation(),
                    'compiler': platform.python_compiler(),
                },
                'boot_time': psutil.boot_time(),
                'users': [user._asdict() for user in psutil.users()],
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error updating system info: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get the current system metrics.

        Returns:
            Dictionary containing the current system metrics
        """
        return self.metrics

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current system metrics in a more compact format.

        Returns:
            Dictionary containing a summary of system metrics
        """
        if not self.metrics:
            return {}

        summary = {
            'cpu_usage': self.metrics.get('cpu', {}).get('usage_percent', 0),
            'memory_usage': self.metrics.get('memory', {}).get('virtual', {}).get('percent', 0),
            'timestamp': self.metrics.get('system', {}).get('timestamp', datetime.utcnow().isoformat())
        }

        if 'gpu' in self.metrics and self.gpu_available:
            gpu = self.metrics['gpu']['devices'][0]  # Primary GPU
            summary.update({
                'gpu_usage': gpu.get('load', 0),
                'gpu_memory_usage': gpu.get('memory_util', 0),
                'gpu_temperature': gpu.get('temperature', 0)
            })

        return summary
