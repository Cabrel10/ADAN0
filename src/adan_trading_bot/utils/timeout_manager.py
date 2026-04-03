import logging
import signal
import threading
from contextlib import contextmanager
from functools import wraps
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class TimeoutException(Exception):
    """Raised when an operation exceeds the configured timeout."""


class TimeoutManager:
    """Manage timeouts using SIGALRM when available, otherwise threading.Timer.

    Usage:
      manager = TimeoutManager(timeout=60, cleanup_callback=save_checkpoint)
      with manager.timeout():
          run_training()

      @manager.decorator
      def fn(...):
          ...
    """

    def __init__(self, timeout: float, cleanup_callback: Optional[Callable[[], None]] = None) -> None:
        self.timeout = float(timeout)
        self.cleanup_callback = cleanup_callback
        self._use_signal = hasattr(signal, "SIGALRM")
        if not self._use_signal:
            logger.warning("SIGALRM not available on this platform, using threading.Timer fallback")

    def _invoke_cleanup(self) -> None:
        if self.cleanup_callback:
            try:
                self.cleanup_callback()
            except Exception:
                logger.exception("Error in cleanup_callback")

    def _signal_handler(self, signum, frame) -> None:  # pragma: no cover (hard to hit reliably in CI)
        self._invoke_cleanup()
        raise TimeoutException(f"Operation exceeded timeout of {self.timeout}s")

    @contextmanager
    def limit(self):
        if self._use_signal:
            prev_handler = signal.getsignal(signal.SIGALRM)
            try:
                signal.signal(signal.SIGALRM, self._signal_handler)
                signal.setitimer(signal.ITIMER_REAL, self.timeout)
                yield
            finally:
                # Disable timer and restore previous handler
                try:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                except Exception:
                    logger.debug("Failed to disable itimer", exc_info=True)
                try:
                    signal.signal(signal.SIGALRM, prev_handler)
                except Exception:
                    logger.debug("Failed to restore previous SIGALRM handler", exc_info=True)
        else:
            # Fallback: raise from a timer thread by setting a flag and checking is not trivial;
            # instead we call cleanup and raise in the main thread on next check. Here we simulate
            # by raising in the timer target; tests will assert TimeoutException via decorator or context.
            # For practical use, prefer SIGALRM on Linux.
            event = threading.Event()

            def _timeout_target():
                self._invoke_cleanup()
                event.set()

            timer = threading.Timer(self.timeout, _timeout_target)
            timer.start()
            try:
                yield
                if event.is_set():
                    raise TimeoutException(f"Operation exceeded timeout of {self.timeout}s")
            finally:
                timer.cancel()

    def decorator(self, func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self.limit():
                return func(*args, **kwargs)
        return wrapper
