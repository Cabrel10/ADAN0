"""
Utility functions for the ADAN trading bot.
"""
import os
import yaml
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from typing import Dict, Any, Optional, List, Tuple, Union, Callable

def _is_running_in_colab():
    """Check if the code is running in Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def get_project_root():
    """
    Get the absolute path to the project root directory.

    Returns:
        str: Absolute path to the project root directory.
    """
    # utils.py est dans src/adan_trading_bot/common/
    # Remonter de 3 niveaux pour atteindre la racine ADAN/
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))

    # Vérification que nous sommes bien dans le bon répertoire
    # en cherchant des fichiers caractéristiques du projet ADAN
    expected_files = ["config", "src", "scripts", "data"]
    if all(os.path.exists(os.path.join(project_root, f)) for f in expected_files):
        return project_root

    # Fallback 1: detection Colab
    if _is_running_in_colab():
        colab_path = "/content/ADAN"
        if os.path.exists(colab_path):
            return colab_path

    # Fallback 2: chercher ADAN dans le chemin courant ou parent
    current_path = os.getcwd()
    path_parts = current_path.split(os.sep)

    # Chercher "ADAN" dans le chemin actuel
    for i, part in enumerate(path_parts):
        if part == "ADAN":
            potential_root = os.sep.join(path_parts[:i + 1])
            if all(os.path.exists(os.path.join(potential_root, f)) for f in expected_files):
                return potential_root

    # Fallback 3: remonter depuis le répertoire courant jusqu'à trouver un répertoire ADAN valide
    search_path = current_path
    for _ in range(10):  # Limiter la recherche à 10 niveaux
        if os.path.basename(search_path) == "ADAN":
            if all(os.path.exists(os.path.join(search_path, f)) for f in expected_files):
                return search_path
        parent = os.path.dirname(search_path)
        if parent == search_path:  # Arrivé à la racine du système
            break
        search_path = parent

    # Fallback 4: chercher un répertoire ADAN dans les répertoires parents
    search_path = current_path
    for _ in range(10):
        adan_candidate = os.path.join(search_path, "ADAN")
        if os.path.exists(adan_candidate) and all(os.path.exists(os.path.join(adan_candidate, f)) for f in expected_files):
            return adan_candidate
        parent = os.path.dirname(search_path)
        if parent == search_path:
            break
        search_path = parent

    # Si tout échoue, retourner le calcul initial et espérer que ça marche
    return project_root

def get_path(path_key, main_config_path_relative_to_root="config/main_config.yaml"):
    """
    Get an absolute path based on the project root and the path key.

    Args:
        path_key: Key for the path (e.g., 'data', 'models', etc.)
        main_config_path_relative_to_root: Path to main config relative to project root.

    Returns:
        str: Absolute path for the specified key.
    """
    project_root = get_project_root()

    # Construire le chemin absolu vers main_config.yaml
    full_main_config_path = os.path.join(project_root, main_config_path_relative_to_root)

    # Essayer de lire le nom du répertoire depuis la configuration
    try:
        with open(full_main_config_path, 'r') as f:
            main_cfg = yaml.safe_load(f)

        # Chercher la clé correspondante dans la section 'paths'
        dir_name_key = f"{path_key}_dir_name"  # ex: data_dir_name
        dir_name = main_cfg.get('paths', {}).get(dir_name_key, path_key)  # Fallback sur la clé
        return os.path.join(project_root, dir_name)

    except Exception as e:
        # Fallback: utiliser directement la clé comme nom de répertoire
        logger = logging.getLogger("adan_trading_bot")
        logger.warning(f"Could not load directory name from config for '{path_key}': {e}")
        return os.path.join(project_root, path_key)

def load_config(config_path):
    """
    Load a YAML configuration file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        dict: Configuration as a dictionary.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_to_pickle(obj, file_path):
    """
    Save an object to a pickle file.

    Args:
        obj: Object to save.
        file_path: Path where to save the object.
    """
    joblib.dump(obj, file_path)

def load_from_pickle(file_path):
    """
    Load an object from a pickle file.

    Args:
        file_path: Path to the pickle file.

    Returns:
        Object loaded from the pickle file.
    """
    return joblib.load(file_path)

def create_directories(dir_paths: Union[str, List[str]]) -> None:
    """
    Create the specified directories if they don't exist.

    Args:
        dir_paths: Path or list of paths to create
    """
    if isinstance(dir_paths, str):
        dir_paths = [dir_paths]

    for dir_path in dir_paths:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

def ensure_dir_exists(dir_path):
    """
    Ensure that a directory exists, creating it if necessary.

    Args:
        dir_path: Path to the directory.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

def timestamp_to_datetime(timestamp):
    """
    Convert a timestamp to a datetime object.

    Args:
        timestamp: Timestamp as an integer (milliseconds since epoch) or string.

    Returns:
        datetime: Datetime object.
    """
    if isinstance(timestamp, str):
        try:
            return datetime.fromisoformat(timestamp)
        except ValueError:
            # Try to parse as integer
            timestamp = int(timestamp)

    # Assume milliseconds if timestamp is large
    if timestamp > 1e10:
        return datetime.fromtimestamp(timestamp / 1000.0)
    else:
        return datetime.fromtimestamp(timestamp)

def format_currency(value, precision=2):
    """
    Format a value as currency.

    Args:
        value: Value to format.
        precision: Number of decimal places.

    Returns:
        str: Formatted currency string.
    """
    return f"${value:.{precision}f}"

def calculate_pnl(entry_price, exit_price, quantity, is_long=True):
    """
    Calculate the PnL for a trade.

    Args:
        entry_price: Entry price.
        exit_price: Exit price.
        quantity: Quantity traded.
        is_long: Whether the position is long (True) or short (False).

    Returns:
        float: PnL amount.
    """
    if is_long:
        return (exit_price - entry_price) * quantity
    else:
        return (entry_price - exit_price) * quantity

def calculate_return_pct(entry_price, exit_price, is_long=True):
    """
    Calculate the percentage return for a trade.

    Args:
        entry_price: Entry price.
        exit_price: Exit price.
        is_long: Whether the position is long (True) or short (False).

    Returns:
        float: Percentage return.
    """
    if is_long:
        return (exit_price - entry_price) / entry_price * 100
    else:
        return (entry_price - exit_price) / entry_price * 100

def calculate_log_return(old_value, new_value):
    """
    Calculate the logarithmic return between two values.

    Args:
        old_value: Old value.
        new_value: New value.

    Returns:
        float: Logarithmic return.
    """
    return np.log(new_value / old_value)

def get_dataframe_numeric_columns(df):
    """
    Get the numeric columns of a DataFrame.

    Args:
        df: Pandas DataFrame.

    Returns:
        list: List of numeric column names.
    """
    return df.select_dtypes(include=[np.number]).columns.tolist()

def ensure_number_or_callable(
    val: Any,
    name: str = "param",
    default: Optional[float] = None,
    positive: bool = True,
    positive_strict: bool = False,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None
) -> Union[float, Callable]:
    """
    Valide et convertit une valeur en nombre natif ou vérifie qu'elle est callable.

    Args:
        val: Valeur à valider (peut être un nombre, une chaîne, ou une callable)
        name: Nom du paramètre pour les messages d'erreur
        default: Valeur par défaut si la conversion échoue (None pour lever une exception)
        positive: Si True, vérifie que le nombre est strictement positif
        positive_strict: Si True, vérifie que le nombre est > 0 (et non >= 0)
        min_val: Valeur minimale autorisée (inclusive)
        max_val: Valeur maximale autorisée (inclusive)

    Returns:
        float ou Callable: La valeur convertie en float natif ou la callable d'origine

    Raises:
        ValueError: Si la validation échoue et qu'aucune valeur par défaut n'est fournie
    """
    logger = get_logger("adan_trading_bot.utils")

    # Si la valeur est déjà un callable, on la retourne telle quelle
    if callable(val):
        logger.debug("%s est un callable, conservation de la valeur", name)
        return val

    # Si la valeur est None et qu'une valeur par défaut est fournie
    if val is None and default is not None:
        logger.debug("%s est None, utilisation de la valeur par défaut: %s", name, default)
        return float(default) if not callable(default) else default

    try:
        # Conversion en float natif
        v = float(val) if val is not None else None

        # Vérification des valeurs None après conversion
        if v is None:
            if default is not None:
                return float(default) if not callable(default) else default
            raise ValueError(f"{name} ne peut pas être None")

        # Vérification de la positivité si demandé
        if positive:
            if positive_strict and v <= 0:
                if default is not None:
                    logger.warning(
                        "%s doit être > 0 (reçu %s). Utilisation de la valeur par défaut: %s",
                        name, v, default
                    )
                    return float(default) if not callable(default) else default
                raise ValueError(f"{name} doit être > 0, reçu: {v}")
            elif not positive_strict and v < 0:
                if default is not None:
                    logger.warning(
                        "%s doit être >= 0 (reçu %s). Utilisation de la valeur par défaut: %s",
                        name, v, default
                    )
                    return float(default) if not callable(default) else default
                raise ValueError(f"{name} doit être >= 0, reçu: {v}")

        # Vérification des valeurs minimales et maximales
        if min_val is not None and v < min_val:
            if default is not None:
                logger.warning(
                    "%s doit être >= %s (reçu %s). Utilisation de la valeur par défaut: %s",
                    name, min_val, v, default
                )
                return float(default) if not callable(default) else default
            raise ValueError(f"{name} doit être >= {min_val}, reçu: {v}")

        if max_val is not None and v > max_val:
            if default is not None:
                logger.warning(
                    "%s doit être <= %s (reçu %s). Utilisation de la valeur par défaut: %s",
                    name, max_val, v, default
                )
                return float(default) if not callable(default) else default
            raise ValueError(f"{name} doit être <= {max_val}, reçu: {v}")

        logger.debug(
            "%s validé avec succès: %s (type: %s)",
            name, v, type(v).__name__
        )
        return v

    except (TypeError, ValueError) as e:
        if default is not None:
            logger.warning(
                "Impossible de convertir %s (%s) en float. "
                "Utilisation de la valeur par défaut: %s. Erreur: %s",
                name, val, default, str(e)
            )
            return float(default) if not callable(default) else default
        raise ValueError(
            f"{name} doit être un nombre ou un callable, reçu: {val} (type: {type(val).__name__})"
        ) from e


def safe_serialize(obj: Any) -> Union[dict, list, str, int, float, bool, None]:
    """Safely serialize an object to a JSON-serializable format.

    Handles common types including numpy arrays, pandas objects, and custom objects
    by converting them to native Python types.

    Args:
        obj: Object to serialize

    Returns:
        JSON-serializable representation of the object
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, dict):
        return {str(k): safe_serialize(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [safe_serialize(item) for item in obj]

    # Handle numpy types
    if hasattr(obj, 'item') and callable(obj.item):
        try:
            return obj.item()
        except (ValueError, TypeError):
            pass

    # Handle numpy arrays
    if hasattr(obj, 'tolist') and callable(obj.tolist):
        try:
            return obj.tolist()
        except (ValueError, TypeError):
            pass

    # Handle pandas objects
    if hasattr(obj, 'to_dict') and callable(obj.to_dict):
        try:
            return safe_serialize(obj.to_dict())
        except (ValueError, TypeError):
            pass

    # Handle datetimes
    if hasattr(obj, 'isoformat') and callable(obj.isoformat):
        try:
            return obj.isoformat()
        except (ValueError, TypeError):
            pass

    # Last resort: convert to string
    try:
        return str(obj)
    except Exception:
        return "[unserializable object]"


def sanitize_ppo_params(params: dict) -> dict:
    """
    Validate and sanitize PPO hyperparameters to ensure numerical stability.

    Args:
        params: Dictionary of PPO hyperparameters

    Returns:
        dict: Sanitized parameters with safe values
    """
    if not isinstance(params, dict):
        logger.warning(
            "Invalid params type %s, using empty dict",
            type(params).__name__
        )
        return {}

    params = params.copy()

    # Learning rate validation
    if "learning_rate" in params:
        try:
            lr = float(params["learning_rate"])
            if lr <= 0 or not np.isfinite(lr):
                logger.warning(
                    "Invalid learning_rate %s -> using 1e-4",
                    lr
                )
                lr = 1e-4
            params["learning_rate"] = lr
        except (TypeError, ValueError) as e:
            logger.warning(
                "Error parsing learning_rate: %s -> using 1e-4",
                str(e)
            )
            params["learning_rate"] = 1e-4

    # Entropy coefficient validation
    if "ent_coef" in params:
        try:
            ent_coef = float(params["ent_coef"])
            if ent_coef < 0 or not np.isfinite(ent_coef):
                logger.warning("Invalid ent_coef %s -> using 0.0", ent_coef)
                ent_coef = 0.0
            params["ent_coef"] = ent_coef
        except (TypeError, ValueError) as e:
            logger.warning("Error parsing ent_coef: %s -> using 0.0", str(e))
            params["ent_coef"] = 0.0

    # Gamma validation
    if "gamma" in params:
        try:
            gamma = float(params["gamma"])
            if gamma <= 0 or gamma >= 1.0 or not np.isfinite(gamma):
                logger.warning("Invalid gamma %s -> using 0.99", gamma)
                gamma = 0.99
            params["gamma"] = gamma
        except (TypeError, ValueError) as e:
            logger.warning("Error parsing gamma: %s -> using 0.99", str(e))
            params["gamma"] = 0.99

    # Clip range validation
    if "clip_range" in params:
        try:
            clip = float(params["clip_range"])
            if clip <= 0 or not np.isfinite(clip):
                logger.warning("Invalid clip_range %s -> using 0.2", clip)
                clip = 0.2
            params["clip_range"] = clip
        except (TypeError, ValueError) as e:
            logger.warning(
                "Error parsing clip_range: %s -> using 0.2",
                str(e)
            )
            params["clip_range"] = 0.2

    # Clip range vf validation
    if "clip_range_vf" in params:
        try:
            clip_vf = float(params["clip_range_vf"])
            if clip_vf <= 0 or not np.isfinite(clip_vf):
                logger.warning("Invalid clip_range_vf %s -> using None", clip_vf)
                clip_vf = None
            params["clip_range_vf"] = clip_vf
        except (TypeError, ValueError) as e:
            logger.warning(
                "Error parsing clip_range_vf: %s -> using None",
                str(e)
            )
            params["clip_range_vf"] = None

    # Max grad norm validation
    if "max_grad_norm" in params:
        try:
            max_grad_norm = float(params["max_grad_norm"])
            if max_grad_norm <= 0 or not np.isfinite(max_grad_norm):
                logger.warning(
                    "Invalid max_grad_norm %s -> using 0.5",
                    max_grad_norm
                )
                max_grad_norm = 0.5
            params["max_grad_norm"] = max_grad_norm
        except (TypeError, ValueError) as e:
            logger.warning(
                "Error parsing max_grad_norm: %s -> using 0.5",
                str(e)
            )
            params["max_grad_norm"] = 0.5

    return params


def get_logger(name=None):
    """
    Get a logger with the specified name.

    Args:
        name: Name of the logger. If None, returns the root logger.


    Returns:
        logging.Logger: Logger object.
    """
    if name is None:
        return logging.getLogger()
    return logging.getLogger(name)
