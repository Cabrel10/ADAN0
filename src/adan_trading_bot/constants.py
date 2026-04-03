"""
Constantes centralisées pour ADAN Trading Bot.
Source unique de vérité pour tous les paramètres critiques.
"""

# === PORTFOLIO STATE ===
PORTFOLIO_STATE_SIZE = 20

# === OBSERVATION SPACE ===
OBSERVATION_TIMEFRAMES = {
    "5m": {"window": 20, "features": 14},
    "1h": {"window": 14, "features": 14},
    "4h": {"window": 5, "features": 14},
}

# === ACTION SPACE ===
ACTION_SPACE_DIM = 15  # 5 actions * 3 assets

# === FREQUENCY VALIDATION ===
FORCE_TRADE_CONFIG = {
    "enabled": True,
    "max_trades_per_day_per_timeframe": 2,
    "escalation_steps": [15, 30, 50],
    "reset_on_new_day": True,
}

GLOBAL_RULES = {
    "min_total_trades_per_episode": 1,
    "max_total_trades_per_episode": 50,
    "min_episode_duration_minutes": 30,
}

# === DATA VALIDATION ===
REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}
VALID_TIMEFRAMES = ["5m", "1h", "4h"]
