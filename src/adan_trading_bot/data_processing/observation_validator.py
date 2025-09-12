#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Observation validation module for the ADAN Trading Bot.

This module provides comprehensive validation for observations generated
by the StateBuilder to ensure they meet design specifications and quality
standards.
"""

import datetime
import json
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Add type hints for TensorFlow if available
try:
    import tensorflow as tf  # noqa: F401
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation severity levels."""
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


@dataclass
class ValidationResult:
    """Result of a validation operation.

    Attributes:
        is_valid: Whether the validation passed
        message: Result description
        level: Severity level
        details: Additional validation details
        passed: Whether the validation passed (for compatibility)
    """
    is_valid: bool
    message: str
    level: ValidationLevel = ValidationLevel.INFO
    details: Dict[str, Any] = field(default_factory=dict)
    passed: bool = True

    def __post_init__(self):
        """Ensure passed is consistent with is_valid."""
        self.passed = self.is_valid

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'is_valid': self.is_valid,
            'message': self.message,
            'level': self.level.name,
            'details': self.details,
            'passed': self.passed
        }


class ObservationValidator:
    """Comprehensive validator for multi-timeframe observations.

    This class implements validation checks to ensure observations are suitable
    for reinforcement learning models.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the validator with optional configuration.

        Args:
            config: Configuration dictionary with validation parameters
        """
        self.config = config or {}
        self._setup_default_config()
        self.results: List[ValidationResult] = []

        # Initialize validation statistics
        self.validation_stats = {
            'total_validations': 0,
            'passed_validations': 0,
            'failed_validations': 0,
            'warnings_count': 0,
            'errors_count': 0,
            'critical_count': 0
        }

        # Set up validation thresholds
        self.value_range_threshold = 100.0
        if 'value_range_threshold' in self.config:
            self.value_range_threshold = self.config['value_range_threshold']
        self.nan_tolerance = self.config.get('nan_tolerance', 0.0)
        self.inf_tolerance = self.config.get('inf_tolerance', 0.0)
        self.timeframes = self.config.get('timeframes', ['5m', '1h', '4h'])

    def _setup_default_config(self) -> None:
        """Set up default validation configuration."""
        defaults = {
            'check_shape': True,
            'check_dtype': True,
            'check_finite': True,
            'check_nan_inf': True,
            'check_value_ranges': True,
            'check_temporal_consistency': True,
            'check_statistics': True,
            'warn_on_errors': True,
            'raise_on_critical': False,
            'value_range_threshold': 100.0,
            'nan_tolerance': 0.0,
            'inf_tolerance': 0.0,
        }
        # Update config with defaults if not provided
        for key, value in defaults.items():
            self.config.setdefault(key, value)

    def _make_result(
        self,
        is_valid: bool,
        message: str,
        level: ValidationLevel,
        details: Optional[Dict[str, Any]] = None,
        passed: Optional[bool] = None,
    ) -> ValidationResult:
        """Safely construct a ValidationResult, tolerant to test mocks.

        Some tests may monkeypatch ValidationResult with a simplified constructor
        that doesn't accept 'details' or 'passed'. This helper gracefully
        downgrades arguments to avoid TypeError while preserving core fields.
        """
        try:
            if passed is None:
                return ValidationResult(
                    is_valid=is_valid, message=message, level=level, details=details
                )
            return ValidationResult(
                is_valid=is_valid, message=message, level=level, details=details, passed=passed
            )
        except TypeError:
            try:
                if passed is None:
                    return ValidationResult(
                        is_valid=is_valid, message=message, level=level
                    )
                return ValidationResult(
                    is_valid=is_valid, message=message, level=level, passed=passed
                )
            except TypeError:
                return ValidationResult(
                    is_valid=is_valid, message=message, level=level
                )

    def validate_observation(
        self,
        observation: np.ndarray,
        expected_shape: Optional[Tuple[int, ...]] = None,
        expected_dtype: type = np.float32,
        strict: bool = False
    ) -> Tuple[bool, List[ValidationResult]]:
        """Validate an observation against specifications.

        Args:
            observation: The observation array to validate
            expected_shape: Expected shape of the observation
            expected_dtype: Expected data type of the observation
            strict: If True, warnings are treated as failures

        Returns:
            Tuple of (is_valid, list_of_validation_results)
        """
        results = []

        # Update validation statistics
        self.validation_stats['total_validations'] += 1

        if self.config['check_shape']:
            results.extend(self._validate_shape(observation, expected_shape))

        if self.config['check_dtype']:
            results.extend(self._validate_dtype(observation))

        if self.config['check_nan_inf']:
            results.extend(self._validate_nan_inf(observation))

        if self.config['check_value_ranges']:
            results.extend(self._validate_value_ranges(observation))

        if self.config['check_statistics']:
            results.extend(self._validate_statistics(observation))

        if self.config['check_temporal_consistency']:
            results.extend(self._validate_temporal_consistency(observation))

        # Update statistics based on results
        has_errors = any(
            r.level == ValidationLevel.ERROR for r in results
        )
        has_warnings = any(
            r.level == ValidationLevel.WARNING for r in results
        )
        has_critical = any(
            r.level == ValidationLevel.CRITICAL for r in results
        )

        self.validation_stats['warnings_count'] += sum(
            1 for r in results if r.level == ValidationLevel.WARNING
        )
        self.validation_stats['errors_count'] += sum(
            1 for r in results if r.level == ValidationLevel.ERROR
        )
        self.validation_stats['critical_count'] += sum(
            1 for r in results if r.level == ValidationLevel.CRITICAL
        )

        # Determine overall validity
        is_valid = not has_critical and not has_errors
        if strict:
            is_valid = is_valid and not has_warnings

        if is_valid:
            self.validation_stats['passed_validations'] += 1
        else:
            self.validation_stats['failed_validations'] += 1

        return is_valid, results

    def _validate_shape(
        self,
        observation: np.ndarray,
        expected_shape: Optional[Tuple[int, ...]]
    ) -> List[ValidationResult]:
        """Validate observation shape."""
        results = []



        if expected_shape is None:
            results.append(
                self._make_result(
                    is_valid=True,
                    message=(
                        'No expected shape provided, skipping shape validation'
                    ),
                    level=ValidationLevel.INFO
                )
            )
            return results

        if observation.shape != expected_shape:
            results.append(
                self._make_result(
                    is_valid=False,
                    message=(
                        f'Invalid shape: expected {expected_shape}, '
                        f'got {observation.shape}'
                    ),
                    level=ValidationLevel.ERROR,
                    details={
                        'expected_shape': expected_shape,
                        'actual_shape': observation.shape
                    },
                    passed=False
                )
            )
        else:
            results.append(
                self._make_result(
                    is_valid=True,
                    message=f'Shape validation passed: {observation.shape}',
                    level=ValidationLevel.INFO
                )
            )

        return results

    def _validate_dtype(
        self, observation: np.ndarray
    ) -> List[ValidationResult]:
        """Validate observation data type."""
        results = []

        if not np.issubdtype(observation.dtype, np.floating):
            results.append(
                self._make_result(
                    is_valid=False,
                    message=(
                        "Observation dtype is not floating point: "
                        f"{observation.dtype}"
                    ),
                    level=ValidationLevel.WARNING,
                    details={'dtype': str(observation.dtype)},
                    passed=False,
                )
            )
        else:
            results.append(
                self._make_result(
                    is_valid=True,
                    message=(
                        "Observation dtype is appropriate: "
                        f"{observation.dtype}"
                    ),
                    level=ValidationLevel.INFO,
                    details={'dtype': str(observation.dtype)},
                )
            )

        return results

    def _validate_value_ranges(
        self, observation: np.ndarray
    ) -> List[ValidationResult]:
        """Validate observation value ranges."""
        results = []

        try:
            # Check for values outside expected range
            min_val = np.min(observation)
            max_val = np.max(observation)
            max_abs_val = max(abs(min_val), abs(max_val))

            if max_abs_val > self.value_range_threshold:
                results.append(
                    self._make_result(
                        is_valid=False,
                        message=(
                            "Values exceed expected range: "
                            f"max abs value is {max_abs_val:.4f} > "
                            f"{self.value_range_threshold}"
                        ),
                        level=ValidationLevel.WARNING,
                        details={
                            'min_value': float(min_val),
                            'max_value': float(max_val),
                            'max_abs_value': float(max_abs_val),
                            'threshold': self.value_range_threshold,
                        },
                        passed=False,
                    )
                )
            else:
                results.append(
                    self._make_result(
                        is_valid=True,
                        message=(
                            "Values within expected range: "
                            f"max abs value is {max_abs_val:.4f}"
                        ),
                        level=ValidationLevel.INFO,
                        details={
                            'min_value': float(min_val),
                            'max_value': float(max_val),
                            'max_abs_value': float(
                                max_abs_val
                            ),
                        },
                    )
                )

        except Exception as e:
            results.append(
                self._make_result(
                    is_valid=False,
                    message=(
                        "Value range validation failed: "
                        f"{str(e)}"
                    ),
                    level=ValidationLevel.WARNING,
                    passed=False,
                )
            )

        return results

    def _validate_nan_inf(
        self, observation: np.ndarray
    ) -> List[ValidationResult]:
        """Validate for NaN and infinite values."""
        results = []

        try:
            # Check for NaN values
            nans = np.isnan(observation)
            if np.any(nans):
                nan_count = np.sum(nans)
                nan_indices = list(zip(*np.where(nans)))
                results.append(
                    self._make_result(
                        is_valid=False,
                        message=f"Found {nan_count} NaN values in observation",
                        level=ValidationLevel.ERROR,
                        details={
                            'nan_count': int(nan_count),
                            'nan_indices': nan_indices[:100],  # Limit indices
                            'total_elements': int(observation.size)
                        },
                    )
                )
            else:
                results.append(
                    self._make_result(
                        is_valid=True,
                        message="No NaN values detected",
                        level=ValidationLevel.INFO
                    )
                )

            # Check for infinite values
            infs = np.isinf(observation)
            if np.any(infs):
                inf_count = np.sum(infs)
                inf_indices = list(zip(*np.where(infs)))
                results.append(
                    self._make_result(
                        is_valid=False,
                        message=(
                            f"Found {inf_count} infinite values in observation"
                        ),
                        level=ValidationLevel.ERROR,
                        details={
                            'inf_count': int(inf_count),
                            'inf_indices': inf_indices[:100],  # Limit indices
                            'total_elements': int(observation.size)
                        },
                    )
                )
            else:
                results.append(
                    self._make_result(
                        is_valid=True,
                        message="No infinite values detected",
                        level=ValidationLevel.INFO
                    )
                )

        except Exception as e:
            results.append(
                self._make_result(
                    is_valid=False,
                    message=(
                        "NaN and infinite value validation failed: "
                        f"{str(e)}"
                    ),
                    level=ValidationLevel.WARNING,
                    passed=False,
                )
            )

        return results

    def _validate_statistics(
        self, observation: np.ndarray
    ) -> List[ValidationResult]:
        """Validate statistical properties of the observation."""
        results = []

        try:
            # Support 2D observations by treating as a single timeframe
            is_2d = observation.ndim == 2
            timeframes_iter = (
                [(0, self.timeframes[0])] if is_2d else list(enumerate(self.timeframes))
            )

            for idx, timeframe in timeframes_iter:
                if idx < observation.shape[0]:
                    tf_data = observation if is_2d else observation[idx]

                    mean = np.mean(tf_data)
                    std = np.std(tf_data)

                    # Check for constant values (zero variance)
                    if std < 1e-6:  # More lenient threshold
                        results.append(
                            self._make_result(
                                is_valid=False,
                                message=(
                                    f"{timeframe} has very low variance "
                                    f"(std={std:.8f}), data might be constant"
                                ),
                                level=ValidationLevel.WARNING,
                                details={
                                    'timeframe': timeframe,
                                    'mean': float(mean),
                                    'std': float(std),
                                },
                                passed=False,
                            )
                        )

                    # Check for features with zero variance individually
                    if tf_data.ndim == 2:
                        for feature_idx in range(tf_data.shape[1]):
                            feature_data = tf_data[:, feature_idx]
                            feature_std = np.std(feature_data)
                            if feature_std < 1e-8:
                                msg = (
                                    f"{timeframe} feature {feature_idx} "
                                    "has zero variance (constant values)"
                                )
                                results.append(
                                    self._make_result(
                                        is_valid=False,
                                        message=msg,
                                        level=ValidationLevel.WARNING,
                                        details={
                                            'timeframe': timeframe,
                                            'feature_index': int(feature_idx),
                                            'std': float(feature_std),
                                            'value': float(np.mean(feature_data)),
                                        },
                                        passed=False,
                                    )
                                )

                    # Check for extreme skewness (might indicate data issues)
                    if np.abs(mean) > 5 * std and std > 0:
                        results.append(
                            self._make_result(
                                is_valid=False,
                                message=(
                                    f"{timeframe} shows extreme skewness "
                                    f"(mean={mean:.4f}, std={std:.4f})"
                                ),
                                level=ValidationLevel.WARNING,
                                details={
                                    'timeframe': timeframe,
                                    'mean': float(mean),
                                    'std': float(std),
                                },
                                passed=False,
                            )
                        )

        except Exception as e:
            results.append(
                self._make_result(
                    is_valid=False,
                    message=f"Statistical validation failed: {str(e)}",
                    level=ValidationLevel.WARNING,
                    passed=False,
                )
            )

        return results

    def _validate_temporal_consistency(
        self, observation: np.ndarray
    ) -> List[ValidationResult]:
        """Validate temporal consistency within the observation."""
        results = []

        try:
            # Support 2D observations by treating as a single timeframe
            is_2d = observation.ndim == 2
            timeframes_iter = (
                [(0, self.timeframes[0])] if is_2d else list(enumerate(self.timeframes))
            )

            for idx, timeframe in timeframes_iter:
                if idx < observation.shape[0]:
                    tf_data = observation if is_2d else observation[idx]

                    # Check for repeated patterns
                    # (might indicate data duplication)
                    if tf_data.shape[0] > 1:
                        # Calculate differences between consecutive time steps
                        diffs = np.diff(tf_data, axis=0)

                        # Determine which steps are identical repeats
                        if diffs.ndim == 1:
                            identical_rows = diffs == 0
                        else:
                            identical_rows = np.all(diffs == 0, axis=1)

                        consecutive_identical = 0
                        max_consecutive = 0
                        for is_identical in identical_rows:
                            if is_identical:
                                consecutive_identical += 1
                                max_consecutive = max(max_consecutive, consecutive_identical)
                            else:
                                consecutive_identical = 0

                        # More than 5 or 5% consecutive identical
                        threshold = max(5, int(tf_data.shape[0] * 0.05))
                        if max_consecutive > threshold:
                            results.append(
                                self._make_result(
                                    is_valid=False,
                                    message=(
                                        f"{timeframe} has {max_consecutive} "
                                        "consecutive identical time steps"
                                    ),
                                    level=ValidationLevel.WARNING,
                                    details={
                                        'timeframe': timeframe,
                                        'max_consecutive_identical': int(max_consecutive),
                                        'percentage': float(max_consecutive / tf_data.shape[0] * 100),
                                    },
                                    passed=False,
                                )
                            )

                        # Additionally, detect repeated rows even if not consecutive
                        try:
                            # Use a contiguous view to compute unique rows
                            rows_view = np.ascontiguousarray(tf_data).view(
                                np.dtype((np.void, tf_data.dtype.itemsize * tf_data.shape[1]))
                            )
                            _, counts = np.unique(rows_view, return_counts=True)
                            max_dup = int(counts.max()) if counts.size > 0 else 1
                            dup_rows = int(np.sum(counts > 1)) if counts.size > 0 else 0
                            if max_dup > 1 and dup_rows > 0:
                                results.append(
                                    self._make_result(
                                        is_valid=False,
                                        message=(
                                            f"{timeframe} contains repeated rows that may indicate "
                                            "consecutive identical time steps"
                                        ),
                                        level=ValidationLevel.WARNING,
                                        details={
                                            'timeframe': timeframe,
                                            'duplicate_row_groups': dup_rows,
                                            'max_duplicate_count': max_dup,
                                            'total_rows': int(tf_data.shape[0]),
                                        },
                                        passed=False,
                                    )
                                )
                        except Exception:
                            # Be robust: ignore failures in duplicate detection
                            pass

        except Exception as e:
            results.append(
                self._make_result(
                    is_valid=False,
                    message=(
                        "Temporal consistency validation failed: "
                        f"{str(e)}"
                    ),
                    level=ValidationLevel.WARNING,
                    passed=False,
                )
            )

        return results

    def validate_batch(
        self, observations: np.ndarray, strict: bool = True
    ) -> Tuple[bool, List[List[ValidationResult]]]:
        """Validate a batch of observations.

        Args:
            observations: Batch of observations (batch_size, n_timeframes,
                window_size, n_features)
            strict: If True, warnings are treated as failures

        Returns:
            Tuple of (all_valid, list_of_validation_results_per_observation)
        """
        # Accept 3D (batch, window, features) by adding a timeframe dimension
        if observations.ndim == 3:
            observations = observations[:, np.newaxis, :, :]
        elif observations.ndim != 4:
            raise ValueError(
                f"Expected 3D or 4D batch array, got {observations.ndim}D"
            )

        batch_results = []
        all_valid = True

        for i in range(observations.shape[0]):
            obs_valid, obs_results = self.validate_observation(
                observations[i], strict=strict
            )
            batch_results.append(obs_results)
            all_valid = all_valid and obs_valid

        return all_valid, batch_results

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get a summary of validation statistics."""
        total = self.validation_stats['total_validations']
        if total == 0:
            return {
                'total_validations': 0,
                'success_rate': 0.0,
                'failure_rate': 0.0,
                'warnings_per_validation': 0.0,
                'errors_per_validation': 0.0,
                'critical_per_validation': 0.0,
                'stats': self.validation_stats.copy(),
            }

        return {
            'total_validations': total,
            'success_rate': (
                self.validation_stats['passed_validations'] / total * 100
            ),
            'failure_rate': (
                self.validation_stats['failed_validations'] / total * 100
            ),
            'warnings_per_validation': (
                self.validation_stats['warnings_count'] / total
            ),
            'errors_per_validation': (
                self.validation_stats['errors_count'] / total
            ),
            'critical_per_validation': (
                self.validation_stats['critical_count'] / total
            ),
            'stats': self.validation_stats.copy(),
        }

    def reset_stats(self) -> None:
        """Reset validation statistics."""
        for key in self.validation_stats:
            self.validation_stats[key] = 0

    def log_validation_results(
        self, results: List[ValidationResult], observation_id: str = "unknown"
    ) -> None:
        """Log validation results with appropriate log levels."""
        for result in results:
            log_message = f"[{observation_id}] {result.message}"

            if result.level == ValidationLevel.INFO:
                logger.info(log_message)
            elif result.level == ValidationLevel.WARNING:
                logger.warning(log_message)
            elif result.level == ValidationLevel.ERROR:
                logger.error(log_message)
            elif result.level == ValidationLevel.CRITICAL:
                logger.critical(log_message)

    def save_validation_report(
        self,
        results: List[ValidationResult],
        filepath: Union[str, Path]
    ) -> None:
        """Save validation results to a JSON file.

        Args:
            results: List of validation results
            filepath: Path to save the report
        """
        def _to_py(o):
            # Convert numpy scalars to Python native types
            if isinstance(o, (np.generic,)):
                if isinstance(o, (np.integer,)):
                    return int(o)
                if isinstance(o, (np.floating,)):
                    return float(o)
                if isinstance(o, (np.bool_,)):
                    return bool(o)
            return o

        def _convert(obj):
            if isinstance(obj, dict):
                return {str(k): _convert(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_convert(x) for x in obj]
            return _to_py(obj)

        def _result_to_dict(r: ValidationResult) -> Dict[str, Any]:
            # Support mocks or simplified result objects without to_dict
            if hasattr(r, 'to_dict') and callable(getattr(r, 'to_dict')):
                try:
                    return r.to_dict()  # type: ignore[attr-defined]
                except Exception:
                    pass
            # Fallback: build dict from attributes if present
            is_valid = bool(getattr(r, 'is_valid', getattr(r, 'passed', False)))
            message = str(getattr(r, 'message', ''))
            level_obj = getattr(r, 'level', None)
            level = level_obj.name if hasattr(level_obj, 'name') else str(level_obj or 'INFO')
            details = getattr(r, 'details', {}) or {}
            passed = bool(getattr(r, 'passed', is_valid))
            return {
                'is_valid': is_valid,
                'message': message,
                'level': level,
                'details': _convert(details),
                'passed': passed,
            }

        report = {
            'timestamp': datetime.datetime.now().isoformat(),
            'results': [_convert(_result_to_dict(r)) for r in results],
            'summary': {
                'total': len(results),
                'passed': sum(1 for r in results if getattr(r, 'is_valid', False) or getattr(r, 'passed', False)),
                'failed': sum(1 for r in results if not (getattr(r, 'is_valid', False) or getattr(r, 'passed', False))),
                'levels': {
                    level.name: sum(1 for r in results if getattr(r, 'level', None) == level)
                    for level in ValidationLevel
                }
            }
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)

        logger.info('Validation report saved to %s', filepath)


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Create a sample observation
    obs = np.random.randn(10, 20, 5).astype(np.float32)

    # Create validator with default config
    validator = ObservationValidator()

    # Run validation
    is_valid, results = validator.validate_observation(obs)

    # Print results
    for result in results:
        status = "PASS" if result.is_valid else "FAIL"
        print(f"[{status}] {result.message}")
        if result.details:
            print(f"  Details: {result.details}")

    # Save full report
    validator.save_validation_report(
        results,
        'validation_report.json'
    )
