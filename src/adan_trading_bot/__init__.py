"""ADAN Trading Bot Package."""
import logging
from pathlib import Path

# Import core components for easier access
from .data_processing import feature_engineer
from .environment import MultiAssetChunkedEnv
from .trading import OrderManager
from .portfolio import PortfolioManager
from .common.custom_logger import setup_logging
from .utils.log_utils import setup_log_management

__version__ = '0.1.0'

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_CONFIG_DIR = PROJECT_ROOT / 'config'

# Initialize logging
logger = logging.getLogger(__name__)

# Only configure logging if it hasn't been configured yet
if not logging.root.handlers:
    # Try to load logging config from file
    config_path = DEFAULT_CONFIG_DIR / 'logging_config.yaml'

    # Setup logging with config file if it exists, otherwise use defaults
    if config_path.exists():
        logger = setup_logging(config_path=config_path)
    else:
        logger = setup_logging()

    # Setup log management
    try:
        log_manager = setup_log_management(
            config_path=config_path if config_path.exists() else None,
            cleanup_days=30
        )
    except Exception as e:
        logger.warning("Failed to initialize log management: %s", str(e))

logger.info("ADAN Trading Bot v%s initialized", __version__)

# Export public API
__all__ = [
    'feature_engineer',
    'MultiAssetChunkedEnv',
    'OrderManager',
    'PortfolioManager',
    'logger',
    'setup_logging',
    'setup_log_management'
]
