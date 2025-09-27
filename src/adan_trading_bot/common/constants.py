"""
Constants used throughout the ADAN trading bot.
"""

# Action space constants
HOLD = 0
BUY = 1
SELL = 2

# Action codes for 11 discrete actions
ACTION_HOLD = 0
ACTION_BUY_ASSET_0 = 1
ACTION_BUY_ASSET_1 = 2
ACTION_BUY_ASSET_2 = 3
ACTION_BUY_ASSET_3 = 4
ACTION_BUY_ASSET_4 = 5
ACTION_SELL_ASSET_0 = 6
ACTION_SELL_ASSET_1 = 7
ACTION_SELL_ASSET_2 = 8
ACTION_SELL_ASSET_3 = 9
ACTION_SELL_ASSET_4 = 10

# Order types
ORDER_TYPE_MARKET = "MARKET"
ORDER_TYPE_LIMIT = "LIMIT"
ORDER_TYPE_STOP_LOSS = "STOP_LOSS"
ORDER_TYPE_TAKE_PROFIT = "TAKE_PROFIT"
ORDER_TYPE_TRAILING_STOP = "TRAILING_STOP"
ORDER_TYPE_STOP_LIMIT = "STOP_LIMIT"

# Executed order types
ORDER_TYPE_EXECUTED_LIMIT = "EXECUTED_LIMIT"
ORDER_TYPE_EXECUTED_STOP_LOSS = "EXECUTED_STOP_LOSS"
ORDER_TYPE_EXECUTED_TAKE_PROFIT = "EXECUTED_TAKE_PROFIT"
ORDER_TYPE_EXECUTED_TRAILING_STOP = "EXECUTED_TRAILING_STOP"
ORDER_TYPE_EXECUTED_STOP_LIMIT = "EXECUTED_STOP_LIMIT"

# Order status
ORDER_STATUS_PENDING = "PENDING"
ORDER_STATUS_EXECUTED = "EXECUTED"
ORDER_STATUS_EXPIRED = "EXPIRED"
ORDER_STATUS_CANCELED = "CANCELED"

# Invalid order reasons
INVALID_ORDER_TOO_SMALL = "INVALID_ORDER_TOO_SMALL"
INVALID_ORDER_BELOW_TOLERABLE = "INVALID_ORDER_BELOW_TOLERABLE"
INVALID_NO_CAPITAL = "INVALID_NO_CAPITAL"
INVALID_MAX_POSITIONS = "INVALID_MAX_POSITIONS"
INVALID_NO_POSITION = "INVALID_NO_POSITION"

# Chunk and window configuration
DEFAULT_CHUNK_SIZES = {
    '5m': 2048,   # >> window_size=100
    '1h': 1024,
    '4h': 512
}
DEFAULT_WINDOW_SIZE = 100

# Position sizing tiers (capital in base currency)
CAPITAL_TIERS = {
    "Micro": {
        "min": 11.0,
        "max": 30.0,
        "max_position_size_pct": 30,
        "risk_per_trade_pct": 5.0,
        "exposure_range": [5, 25]  # 5% à 25% d'exposition
    },
    "Small": {
        "min": 30.0,
        "max": 100.0,
        "max_position_size_pct": 20,
        "risk_per_trade_pct": 2.0,
        "exposure_range": [8, 35]  # 8% à 35% d'exposition
    },
    "Medium": {
        "min": 100.0,
        "max": 300.0,
        "max_position_size_pct": 15,
        "risk_per_trade_pct": 1.5,
        "exposure_range": [10, 50]  # 10% à 50% d'exposition
    },
    "Large": {
        "min": 300.0,
        "max": 1000.0,
        "max_position_size_pct": 10,
        "risk_per_trade_pct": 1.0,
        "exposure_range": [15, 70]  # 15% à 70% d'exposition
    },
    "Enterprise": {
        "min": 1000.0,
        "max": float('inf'),
        "max_position_size_pct": 8,
        "risk_per_trade_pct": 0.5,
        "exposure_range": [20, 90]  # 20% à 90% d'exposition
    }
}

# Trading parameters
DEFAULT_COMMISSION = 0.0002  # 0.02%
DEFAULT_SLIPPAGE = 0.0005    # 0.05%
MIN_TRADE_VALUE = 10.0       # Minimum trade value in base currency

# Episode parameters
MIN_STEPS_PER_EPISODE = 500
MAX_STEPS_PER_EPISODE = 1000

# Reward parameters
COMMISSION_PENALTY_WEIGHT = 1.0
HOLDING_BONUS_PER_STEP = 0.0  # Can be increased to encourage longer holds

# Standard column names
COL_TIMESTAMP = "timestamp"
COL_PAIR = "pair"
COL_OPEN = "open"
COL_HIGH = "high"
COL_LOW = "low"
COL_CLOSE = "close"
COL_VOLUME = "volume"

# Standard indicator names
COL_RSI_14 = "rsi_14"
COL_EMA_10 = "ema_10"
COL_EMA_20 = "ema_20"
COL_SMA_50 = "sma_50"
COL_MACD = "macd_12_26_9"
COL_MACD_SIGNAL = "macd_signal_9"
COL_MACD_HIST = "macd_hist"
COL_BB_UPPER = "bb_upper_20_2"
COL_BB_MIDDLE = "bb_middle_20_2"
COL_BB_LOWER = "bb_lower_20_2"
COL_ATR_14 = "atr_14"
COL_ADX_14 = "adx_14"

# Penalties and rewards
PENALTY_INVALID_ORDER = -0.2
PENALTY_ORDER_TOO_SMALL = -0.5
PENALTY_BELOW_TOLERABLE = -0.2
PENALTY_NO_CAPITAL = -0.2
PENALTY_MAX_POSITIONS = -0.2
PENALTY_NO_POSITION = -0.2
PENALTY_LIMIT_EXPIRY = -0.1
PENALTY_STOP_LOSS_EXPIRY = -0.1
PENALTY_TRAILING_STOP_EXPIRY = -0.05
PENALTY_TIME = -0.001  # Small penalty for each step to encourage action

# Reward clipping
REWARD_MIN = -10.0
REWARD_MAX = 10.0

# Minimum order values
MIN_ORDER_VALUE_TOLERABLE = 10.0
MIN_ORDER_VALUE_ABSOLUTE = 9.0

# Default order expiry (in steps)
DEFAULT_ORDER_EXPIRY = 10
