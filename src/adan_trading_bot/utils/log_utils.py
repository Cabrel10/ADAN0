"""
Logging utilities for the ADAN Trading Bot.

This module provides utilities for log management including log rotation,
compression, and search functionality.
"""
import gzip
import json
import logging
import os
import shutil
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator, Union

from dateutil.parser import parse as parse_date

logger = logging.getLogger(__name__)

class LogManager:
    """
    A class to manage log files including rotation, compression, and search.
    """

    def __init__(
        self,
        log_dir: Union[str, Path] = "logs",
        log_file: str = "adan_trading_bot.log",
        json_log_file: str = "adan_trading_bot.json",
        max_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        compress_backups: bool = True
    ) -> None:
        """
        Initialize the LogManager.

        Args:
            log_dir: Directory where log files are stored
            log_file: Name of the main log file
            json_log_file: Name of the JSON log file
            max_size: Maximum size of log files before rotation (in bytes)
            backup_count: Number of backup files to keep
            compress_backups: Whether to compress backup files
        """
        self.log_dir = Path(log_dir)
        self.log_file = self.log_dir / log_file
        self.json_log_file = self.log_dir / json_log_file
        self.max_size = max_size
        self.backup_count = backup_count
        self.compress_backups = compress_backups

        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def rotate_logs(self) -> None:
        """
        Rotate log files if they exceed the maximum size.
        """
        self._rotate_file(self.log_file)
        self._rotate_file(self.json_log_file)

    def _rotate_file(self, file_path: Path) -> None:
        """
        Rotate a single log file if it exceeds the maximum size.

        Args:
            file_path: Path to the log file to rotate
        """
        if not file_path.exists() or file_path.stat().st_size < self.max_size:
            return

        # Find the next available backup number
        base_name = file_path.name
        backup_paths = []

        # Find all existing backup files
        for f in self.log_dir.glob(f"{base_name}*"):
            if f != file_path:
                backup_paths.append(f)

        # Sort backup files by creation time (oldest first)
        backup_paths.sort(key=os.path.getctime)

        # Remove old backups if we have too many
        while len(backup_paths) >= self.backup_count:
            old_backup = backup_paths.pop(0)
            try:
                old_backup.unlink()
            except OSError as e:
                logger.warning("Failed to remove old log file %s: %s", old_backup, e)

        # Create a new backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = file_path.parent / f"{file_path.name}.{timestamp}"

        try:
            # Rename the current log file to the backup name
            file_path.rename(backup_path)

            # Compress the backup if enabled
            if self.compress_backups and not str(backup_path).endswith('.gz'):
                self._compress_file(backup_path)

        except OSError as e:
            logger.error("Failed to rotate log file %s: %s", file_path, e)

    @staticmethod
    def _compress_file(file_path: Path) -> None:
        """
        Compress a file using gzip.

        Args:
            file_path: Path to the file to compress
        """
        compressed_path = Path(f"{file_path}.gz")

        try:
            with open(file_path, 'rb') as f_in, gzip.open(compressed_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

            # Remove the original file if compression was successful
            file_path.unlink()

        except (IOError, OSError) as e:
            logger.error("Failed to compress log file %s: %s", file_path, e)

            # Remove the partial compressed file if it exists
            if compressed_path.exists():
                try:
                    compressed_path.unlink()
                except OSError:
                    pass

    def search_logs(
        self,
        query: str = "",
        level: Optional[str] = None,
        start_time: Optional[Union[str, datetime]] = None,
        end_time: Optional[Union[str, datetime]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search through JSON log files.

        Args:
            query: Text to search for in log messages
            level: Log level to filter by (e.g., 'INFO', 'ERROR')
            start_time: Only include logs after this time (ISO format or datetime)
            end_time: Only include logs before this time (ISO format or datetime)
            limit: Maximum number of results to return

        Returns:
            List of matching log entries as dictionaries
        """
        # Parse time filters
        start_dt = self._parse_datetime(start_time) if start_time else None
        end_dt = self._parse_datetime(end_time) if end_time else None

        results = []

        # Check all JSON log files (current and backups)
        log_files = [self.json_log_file]
        log_files.extend(sorted(
            self.log_dir.glob(f"{self.json_log_file.name}.*"),
            key=os.path.getmtime,
            reverse=True
        ))

        for log_file in log_files:
            if not log_file.exists():
                continue

            try:
                # Handle gzipped files
                open_func = gzip.open if log_file.suffix == '.gz' else open
                mode = 'rt' if log_file.suffix == '.gz' else 'r'

                with open_func(log_file, mode, encoding='utf-8') as f:
                    for line in f:
                        try:
                            entry = json.loads(line)

                            # Apply filters
                            if self._matches_filters(entry, query, level, start_dt, end_dt):
                                results.append(entry)

                                if len(results) >= limit:
                                    return results

                        except json.JSONDecodeError:
                            logger.warning("Invalid JSON in log file: %s", log_file)
                            continue

            except (IOError, OSError) as e:
                logger.error("Error reading log file %s: %s", log_file, e)
                continue

        return results

    @staticmethod
    def _parse_datetime(dt: Union[str, datetime]) -> datetime:
        """
        Parse a datetime from a string or return a datetime object.

        Args:
            dt: Datetime string or datetime object

        Returns:
            datetime: Parsed datetime object
        """
        if isinstance(dt, datetime):
            return dt
        return parse_date(dt)

    @staticmethod
    def _matches_filters(
        entry: Dict[str, Any],
        query: str,
        level: Optional[str],
        start_dt: Optional[datetime],
        end_dt: Optional[datetime]
    ) -> bool:
        """
        Check if a log entry matches all the specified filters.

        Args:
            entry: Log entry to check
            query: Text to search for in the message
            level: Log level to filter by
            start_dt: Only include entries after this time
            end_dt: Only include entries before this time

        Returns:
            bool: True if the entry matches all filters
        """
        # Check level filter
        if level and entry.get('level') != level.upper():
            return False

        # Check time filters
        timestamp = entry.get('timestamp')
        if timestamp:
            try:
                entry_dt = parse_date(timestamp)

                if start_dt and entry_dt < start_dt:
                    return False

                if end_dt and entry_dt > end_dt:
                    return False

            except (ValueError, TypeError):
                # If timestamp is invalid, include it to be safe
                pass

        # Check text search
        if query:
            query = query.lower()
            message = str(entry.get('message', '')).lower()

            if query not in message:
                return False

        return True

    def cleanup_old_logs(self, days_to_keep: int = 30) -> None:
        """
        Remove log files older than the specified number of days.

        Args:
            days_to_keep: Number of days of logs to keep
        """
        cutoff_time = datetime.now() - timedelta(days=days_to_keep)

        for log_file in self.log_dir.glob('*'):
            try:
                mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                if mtime < cutoff_time:
                    log_file.unlink()
                    logger.info("Removed old log file: %s", log_file)

            except OSError as e:
                logger.warning("Failed to remove old log file %s: %s", log_file, e)

def setup_log_management(
    config_path: Optional[Union[str, Path]] = None,
    log_dir: Union[str, Path] = "logs",
    log_file: str = "adan_trading_bot.log",
    json_log_file: str = "adan_trading_bot.json",
    max_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    compress_backups: bool = True,
    cleanup_days: Optional[int] = None
) -> LogManager:
    """
    Set up log management with the specified configuration.

    Args:
        config_path: Path to a YAML configuration file
        log_dir: Directory to store log files
        log_file: Name of the main log file
        json_log_file: Name of the JSON log file
        max_size: Maximum size of log files before rotation (in bytes)
        backup_count: Number of backup files to keep
        compress_backups: Whether to compress backup files
        cleanup_days: If specified, remove log files older than this many days

    Returns:
        LogManager: Configured LogManager instance
    """
    print(f"DEBUG - setup_log_management - config_path: {config_path}")
    print(f"DEBUG - setup_log_management - initial log_dir: {log_dir}")

    # Convert log_dir to Path if it's a string
    log_dir = Path(log_dir)
    print(f"DEBUG - setup_log_management - log_dir as Path: {log_dir}")

    # Load config if provided
    if config_path and os.path.exists(str(config_path)):
        try:
            print(f"DEBUG - Loading config from: {config_path}")
            config_path = Path(config_path).resolve()
            print(f"DEBUG - Resolved config path: {config_path}")

            with open(str(config_path), 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            print(f"DEBUG - Config loaded: {config}")

            # Update settings from config
            if 'log_dir' in config:
                print(f"DEBUG - Found log_dir in config: {config['log_dir']}")
                # If log_dir is relative, make it relative to the config file's directory
                log_dir = Path(config['log_dir'])
                print(f"DEBUG - log_dir as Path: {log_dir}")
                print(f"DEBUG - is_absolute(): {log_dir.is_absolute()}")

                if not log_dir.is_absolute():
                    print(f"DEBUG - log_dir is relative, resolving against: {config_path.parent}")
                    log_dir = config_path.parent / log_dir
                    print(f"DEBUG - Resolved log_dir: {log_dir}")

            print(f"DEBUG - Final log_dir: {log_dir}")
            log_file = config.get('log_file', log_file)
            json_log_file = config.get('json_log_file', json_log_file)
            max_size = config.get('max_log_size', max_size)
            backup_count = config.get('backup_count', backup_count)
            compress_backups = config.get('compress_backups', compress_backups)

            print(f"DEBUG - Final settings:")
            print(f"  log_dir: {log_dir}")
            print(f"  log_file: {log_file}")
            print(f"  json_log_file: {json_log_file}")
            print(f"  max_size: {max_size}")
            print(f"  backup_count: {backup_count}")
            print(f"  compress_backups: {compress_backups}")

        except Exception as e:
            logger.warning("Error loading log config: %s. Using defaults.", str(e))
            import traceback
            traceback.print_exc()

    # Create and configure the log manager
    print(f"DEBUG - Creating LogManager with log_dir: {log_dir}")
    log_manager = LogManager(
        log_dir=log_dir,
        log_file=log_file,
        json_log_file=json_log_file,
        max_size=max_size,
        backup_count=backup_count,
        compress_backups=compress_backups
    )

    # Rotate logs if needed
    log_manager.rotate_logs()

    # Clean up old logs if configured
    if cleanup_days is not None:
        log_manager.cleanup_old_logs(days_to_keep=cleanup_days)

    return log_manager
