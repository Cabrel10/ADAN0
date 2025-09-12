"""
Hierarchical Error Management System for ADAN Trading Bot.

This module provides a comprehensive error handling framework with classification,
automatic retry logic, and escalation mechanisms.
"""
import time
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Callable, Type, Dict, Any, List, TypeVar, Generic

# Type variable for the return type of the function being wrapped
T = TypeVar('T')

class ErrorSeverity(Enum):
    """Classification of error severity levels."""
    DEBUG = auto()      # Minor issues that don't affect functionality
    INFO = auto()       # Informational messages
    WARNING = auto()    # Potential issues that don't prevent execution
    ERROR = auto()      # Errors that affect functionality but are recoverable
    CRITICAL = auto()   # Critical errors that require immediate attention

class ErrorCategory(Enum):
    """Classification of error categories."""
    CONFIGURATION = auto()      # Configuration-related errors
    NETWORK = auto()           # Network and connectivity issues
    DATA = auto()              # Data-related errors
    TRADING = auto()           # Trading operation errors
    RESOURCE = auto()          # Resource constraints
    VALIDATION = auto()        # Data validation errors
    UNKNOWN = auto()           # Unclassified errors

@dataclass
class ErrorContext:
    """Contextual information about where an error occurred."""
    module: str
    function: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

class TradingError(Exception):
    """Base exception class for all trading-related errors."""
    def __init__(self,
                 message: str,
                 severity: ErrorSeverity = ErrorSeverity.ERROR,
                 category: ErrorCategory = ErrorCategory.UNKNOWN,
                 context: Optional[ErrorContext] = None,
                 cause: Optional[Exception] = None):
        self.message = message
        self.severity = severity
        self.category = category
        self.context = context
        self.cause = cause
        super().__init__(self.message)

class ConfigurationError(TradingError):
    """Raised when there's a configuration error."""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.CONFIGURATION,
            **kwargs
        )

class NetworkError(TradingError):
    """Raised when there are network-related errors."""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.NETWORK,
            **kwargs
        )

class DataError(TradingError):
    """Raised when there are data-related errors."""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.DATA,
            **kwargs
        )

class ErrorHandler:
    """
    A class to handle errors with automatic retry and escalation capabilities.
    """
    def __init__(self,
                 max_retries: int = 3,
                 initial_delay: float = 1.0,
                 backoff_factor: float = 2.0,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the error handler.

        Args:
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay between retries in seconds
            backoff_factor: Multiplier for delay between retries
            logger: Logger instance to use (creates one if None)
        """
        self.max_retries = max(max_retries, 0)
        self.initial_delay = max(initial_delay, 0)
        self.backoff_factor = max(backoff_factor, 1.0)
        self.logger = logger or logging.getLogger(__name__)

    def with_retry(self,
                  func: Callable[..., T],
                  retry_exceptions: tuple[Type[Exception], ...] = (Exception,),
                  context: Optional[ErrorContext] = None) -> Callable[..., T]:
        """
        Decorator to add retry logic to a function.

        Args:
            func: The function to wrap
            retry_exceptions: Tuple of exception types to retry on
            context: Optional context information

        Returns:
            Wrapped function with retry logic
        """
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            delay = self.initial_delay

            for attempt in range(self.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retry_exceptions as e:
                    last_exception = e
                    if attempt < self.max_retries:
                        self._log_retry_attempt(e, attempt, delay, context)
                        time.sleep(delay)
                        delay *= self.backoff_factor
                    else:
                        self._log_retry_failure(e, context)
                        raise self._enhance_exception(e, context) from e
                except Exception as e:
                    # Don't retry on unexpected exceptions
                    self._log_unexpected_error(e, context)
                    raise self._enhance_exception(e, context) from e

            # This should never be reached due to the raise in the except block
            raise RuntimeError("Unexpected error in retry logic")

        return wrapper

    def _log_retry_attempt(self,
                          exception: Exception,
                          attempt: int,
                          delay: float,
                          context: Optional[ErrorContext] = None):
        """Log a retry attempt."""
        self.logger.warning(
            f"Attempt {attempt + 1}/{self.max_retries + 1} failed: {str(exception)}. "
            f"Retrying in {delay:.2f}s..."
        )

    def _log_retry_failure(self,
                          exception: Exception,
                          context: Optional[ErrorContext] = None):
        """Log a retry failure."""
        self.logger.error(
            f"All {self.max_retries + 1} attempts failed. Last error: {str(exception)}",
            exc_info=True
        )

    def _log_unexpected_error(self,
                             exception: Exception,
                             context: Optional[ErrorContext] = None):
        """Log an unexpected error."""
        self.logger.error(
            f"Unexpected error: {str(exception)}",
            exc_info=True
        )

    def _enhance_exception(self,
                          exception: Exception,
                          context: Optional[ErrorContext] = None) -> Exception:
        """Enhance an exception with additional context."""
        if isinstance(exception, TradingError):
            if context and not exception.context:
                exception.context = context
            return exception

        # Create a new TradingError with the original as cause
        return TradingError(
            message=str(exception),
            cause=exception,
            context=context
        )

# Global instance for convenience
default_error_handler = ErrorHandler()

def handle_errors(
    retry_exceptions: tuple[Type[Exception], ...] = (Exception,),
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0
) -> Callable:
    '''
    Decorator factory for error handling with retry logic.

    Example:
        @handle_errors(retry_exceptions=(NetworkError,), max_retries=3)
        def fetch_data():
            # Implementation here
            pass
    '''
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        handler = ErrorHandler(
            max_retries=max_retries,
            initial_delay=initial_delay,
            backoff_factor=backoff_factor
        )
        return handler.with_retry(func, retry_exceptions)
    return decorator
