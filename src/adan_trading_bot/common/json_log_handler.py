"""
JSON Log Handler for structured logging.
"""
import json
import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional
from datetime import datetime
import gzip
import shutil
from pathlib import Path

class JsonLogHandler(RotatingFileHandler):
    """
    A handler class which writes formatted logging records to a JSON file.
    """

    def __init__(
        self,
        filename: str,
        mode: str = 'a',
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        encoding: Optional[str] = None,
        delay: bool = False,
        compress_backups: bool = True
    ) -> None:
        """
        Initialize the handler.

        Args:
            filename: Log file name
            mode: File open mode
            max_bytes: Maximum file size before rotation
            backup_count: Number of backup files to keep
            encoding: File encoding
            delay: If True, file opening is deferred until the first write
            compress_backups: If True, compress rotated log files
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filename)) or '.', exist_ok=True)

        super().__init__(
            filename=filename,
            mode=mode,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding=encoding,
            delay=delay
        )

        self.compress_backups = compress_backups
        self.backup_count = backup_count
        self.base_filename = filename

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the specified record as a JSON string.

        Args:
            record: Log record to format

        Returns:
            str: JSON formatted log entry
        """
        # Extract standard fields
        log_record: Dict[str, Any] = {
            'timestamp': datetime.utcfromtimestamp(record.created).isoformat() + 'Z',
            'level': record.levelname,
            'name': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'process': record.process,
            'thread': record.thread,
            'thread_name': record.threadName,
        }

        # Add exception info if present
        if record.exc_info:
            # Use attached formatter if present, otherwise a temporary one
            try:
                formatter = self.formatter or logging.Formatter()
                log_record['exception'] = formatter.formatException(record.exc_info)
            except Exception:
                # Fallback to string conversion
                log_record['exception'] = str(record.exc_info)

        # Add any extra attributes
        for key, value in record.__dict__.items():
            if key not in ('args', 'asctime', 'created', 'exc_info', 'exc_text',
                         'filename', 'funcName', 'id', 'levelname', 'levelno',
                         'lineno', 'module', 'msecs', 'msecs', 'message',
                         'msg', 'name', 'pathname', 'process', 'processName',
                         'relativeCreated', 'stack_info', 'thread', 'threadName'):
                log_record[key] = value

        return json.dumps(log_record, ensure_ascii=False)

    def doRollover(self) -> None:
        """
        Do a rollover, as described in __init__().
        """
        # Let the parent class handle the actual rollover
        super().doRollover()

        if not self.compress_backups:
            return

        # Compress all backup files
        for i in range(1, self.backup_count + 1):
            sfn = f"{self.base_filename}.{i}"
            dfn = f"{self.base_filename}.{i}.gz"

            # Skip if the file doesn't exist or is already compressed
            if not os.path.exists(sfn) or os.path.exists(dfn):
                continue

            try:
                with open(sfn, 'rb') as f_in, gzip.open(dfn, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                os.remove(sfn)
            except (IOError, OSError):
                # If compression fails, just leave the file as is
                pass

class JsonLogFormatter(logging.Formatter):
    """
    A formatter that converts log records to JSON format.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the specified record as a JSON string.

        Args:
            record: Log record to format

        Returns:
            str: JSON formatted log entry
        """
        # Extract standard fields
        log_record: Dict[str, Any] = {
            'timestamp': datetime.utcfromtimestamp(record.created).isoformat() + 'Z',
            'level': record.levelname,
            'name': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_record['exception'] = self.formatException(record.exc_info)

        # Add any extra attributes
        for key, value in record.__dict__.items():
            if key not in ('args', 'asctime', 'created', 'exc_info', 'exc_text',
                         'filename', 'funcName', 'id', 'levelname', 'levelno',
                         'lineno', 'module', 'msecs', 'msecs', 'message',
                         'msg', 'name', 'pathname', 'process', 'processName',
                         'relativeCreated', 'stack_info', 'thread', 'threadName'):
                log_record[key] = value

        return json.dumps(log_record, ensure_ascii=False)
