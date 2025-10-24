
import os
import sys
import logging

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
sys.path.insert(0, project_root)

from src.adan_trading_bot.common.config_loader import ConfigLoader
from src.adan_trading_bot.data_processing.state_builder import StateBuilder
from src.adan_trading_bot.portfolio.portfolio_manager import PortfolioManager

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
config_path = os.path.join(project_root, "config", "config.yaml")
config = ConfigLoader.load_config(config_path)

# Dummy portfolio manager
portfolio_manager = PortfolioManager(config=config, worker_id=0)


# Instantiate StateBuilder
state_builder = StateBuilder(
    features_config=config["data"]["features_config"]["timeframes"],
    timeframes=config["data"]["timeframes"],
    window_size=config["environment"]["window_size"],
    include_portfolio_state=config["portfolio"]["include_portfolio_state"],
    normalize=config["preprocessing"]["normalization"]["method"] == "minmax",
)

# Get and print the portfolio state dimension
try:
    # We need to set the portfolio_manager manually since it's not in the constructor
    state_builder.portfolio_manager = portfolio_manager

    portfolio_dim = state_builder.get_portfolio_state_dim()
    logger.info(f"Successfully retrieved portfolio_state_dim: {portfolio_dim}")
    print(f"Portfolio State Dimension: {portfolio_dim}")

    # Also check the shape of the built portfolio state
    portfolio_state = state_builder.build_portfolio_state(portfolio_manager)
    logger.info(f"Successfully built portfolio_state with shape: {portfolio_state.shape}")
    print(f"Built Portfolio State Shape: {portfolio_state.shape}")

except Exception as e:
    logger.error(f"An error occurred: {e}", exc_info=True)

