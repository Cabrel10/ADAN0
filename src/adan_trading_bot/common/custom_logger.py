"""
Custom logger setup for the ADAN trading bot with support for both console and JSON logging.
"""
import logging
import os
from pathlib import Path
from typing import Optional, Union

import yaml
import sys
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# Import the JSON log handler
from .json_log_handler import JsonLogHandler, JsonLogFormatter

# Define a custom theme for rich
custom_theme = Theme({
    "info": "green",
    "warning": "yellow",
    "error": "bold red",
    "critical": "bold white on red",
    "success": "bold green",
    "trade_buy": "bold blue",
    "trade_sell": "bold purple",
    "profit": "bold green",
    "loss": "bold red",
})

console = Console(theme=custom_theme)

def setup_logging(
    config_path: Optional[Union[str, Path]] = None,
    default_level: int = logging.INFO,
    log_dir: Union[str, Path] = "logs",
    log_file: str = "adan_trading_bot.log",
    enable_json_logs: bool = False,  # Désactivé par défaut
    enable_console_logs: bool = True,
    json_log_file: str = "adan_trading_bot.json",
    max_log_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    compress_backups: bool = True,
    force_plain_console: bool = True  # Nouveau paramètre pour forcer la console simple
) -> logging.Logger:
    """
    Setup logging configuration with support for both console and JSON logging.

    Args:
        config_path: Path to the logging configuration file.
        default_level: Default logging level if config is not found.
        log_dir: Directory to store log files.
        log_file: Name of the log file.
        enable_json_logs: Whether to enable JSON logging.
        enable_console_logs: Whether to enable console logging.
        json_log_file: Name of the JSON log file.
        max_log_size: Maximum size of log files before rotation.
        backup_count: Number of backup files to keep.
        compress_backups: Whether to compress rotated log files.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger("adan_trading_bot")

    # If logger is already configured, return it
    if logger.handlers:
        return logger

    # Set the log level
    log_level = default_level

    # Load config if provided
    if config_path and os.path.exists(str(config_path)):
        try:
            with open(str(config_path), 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # Get log level from config
            log_level_str = config.get('level', 'INFO')
            log_level = getattr(
                logging,
                log_level_str.upper(),
                default_level
            )

            # Update settings from config if available
            log_dir = config.get('log_dir', log_dir)
            log_file = config.get('log_file', log_file)
            enable_json_logs = config.get('enable_json_logs', enable_json_logs)
            enable_console_logs = config.get(
                'enable_console_logs',
                enable_console_logs
            )
            json_log_file = config.get('json_log_file', json_log_file)
            max_log_size = config.get('max_log_size', max_log_size)
            backup_count = config.get('backup_count', backup_count)
            compress_backups = config.get('compress_backups', compress_backups)

        except Exception as e:
            logger.warning(
                "Error loading logging config: %s. Using default settings.",
                str(e)
            )

    # Ensure log directory exists
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    # Clear any existing handlers
    logger.handlers = []

    # Set up console handler if enabled
    if enable_console_logs:
        if not force_plain_console and not any(isinstance(h, RichHandler) for h in logging.root.handlers):
            # Utiliser RichHandler si disponible et non forcé en mode simple
            try:
                rich_handler = RichHandler(
                    console=console,
                    rich_tracebacks=True,
                    markup=True,
                    show_time=True,
                    show_path=True
                )
                rich_handler.setFormatter(
                    logging.Formatter(
                        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S'
                    )
                )
                rich_handler.setLevel(log_level)
                logger.addHandler(rich_handler)
            except Exception as e:
                logger.warning(f"Failed to initialize Rich handler: {e}")
                force_plain_console = True

        if force_plain_console:
            # Utiliser un handler de console simple
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            console_handler.setLevel(log_level)
            logger.addHandler(console_handler)

    # Set up JSON file handler if enabled
    if enable_json_logs:
        json_log_path = log_dir_path / json_log_file

        json_handler = JsonLogHandler(
            filename=str(json_log_path),
            max_bytes=max_log_size,
            backup_count=backup_count,
            compress_backups=compress_backups
        )

        # Use the JSON formatter
        json_handler.setFormatter(JsonLogFormatter())
        json_handler.setLevel(log_level)
        logger.addHandler(json_handler)

    # Set the log level for the root logger
    logger.setLevel(log_level)

    # Suppress logging from other libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)

    return logger

def get_logger():
    """
    Get the configured logger instance.

    Returns:
        logging.Logger: Logger instance.
    """
    return logging.getLogger("adan_trading_bot")

# Convenience functions for logging with rich formatting
def log_info(message):
    """Log an info message with rich formatting."""
    get_logger().info(f"[info]{message}[/info]")

def log_warning(message):
    """Log a warning message with rich formatting."""
    get_logger().warning(f"[warning]{message}[/warning]")

def log_error(message):
    """Log an error message with rich formatting."""
    get_logger().error(f"[error]{message}[/error]")

def log_critical(message):
    """Log a critical message with rich formatting."""
    get_logger().critical(f"[critical]{message}[/critical]")

def log_success(message):
    """Log a success message with rich formatting."""
    get_logger().info(f"[success]{message}[/success]")

def log_trade_buy(message):
    """Log a buy trade with rich formatting."""
    get_logger().info(f"[trade_buy]{message}[/trade_buy]")

def log_trade_sell(message):
    """Log a sell trade with rich formatting."""
    get_logger().info(f"[trade_sell]{message}[/trade_sell]")

def log_profit(message):
    """Log a profit message with rich formatting."""
    get_logger().info(f"[profit]{message}[/profit]")

def log_loss(message):
    """Log a loss message with rich formatting."""
    get_logger().info(f"[loss]{message}[/loss]")
