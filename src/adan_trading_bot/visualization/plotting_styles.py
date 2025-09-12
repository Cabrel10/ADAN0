""
Définition des couleurs et styles pour les visualisations.
"""
from typing import Dict, Any

# Couleurs pour les graphiques
COLORS = {
    # Bougies et prix
    'candle_up': "#2ECC71",      # Vert
    'candle_down': "#E74C3C",    # Rouge
    'wick': "#34495E",           # Gris foncé
    'close': "#3498DB",          # Bleu

    # Moyennes mobiles
    'ma_short': "#F1C40F",       # Jaune
    'ma_medium': "#3498DB",      # Bleu
    'ma_long': "#8E44AD",        # Violet
    'bb_upper': "#E67E22",       # Orange
    'bb_middle': "#2980B9",      # Bleu foncé
    'bb_lower': "#E67E22",       # Orange

    # Volume
    'volume_up': "#2ECC71",      # Vert clair
    'volume_down': "#E74C3C",    # Rouge clair
    'volume_alpha': 0.3,         # Transparence

    # Indicateurs
    'rsi': "#9B59B6",            # Violet
    'rsi_overbought': "#E74C3C", # Rouge
    'rsi_oversold': "#2ECC71",   # Vert
    'macd': "#3498DB",           # Bleu
    'signal': "#E67E22",         # Orange
    'hist_positive': "#2ECC71",  # Vert
    'hist_negative': "#E74C3C",  # Rouge
    'atr': "#9B59B6",            # Violet
    'obv': "#3498DB",            # Bleu
    'obv_ema': "#E67E22",        # Orange
    'vwap': "#8E44AD",           # Violet foncé
    'supertrend_up': "#2ECC71",  # Vert
    'supertrend_down': "#E74C3C",# Rouge
    'ichimoku_kijun': "#9B59B6", # Violet
    'ichimoku_tenkan': "#E67E22",# Orange
    'ichimoku_senkou_a': "#2ECC71", # Vert clair
    'ichimoku_senkou_b': "#E74C3C", # Rouge clair
    'ichimoku_chikou': "#3498DB",   # Bleu
    'support': "#2ECC71",         # Vert
    'resistance': "#E74C3C",     # Rouge
    'fib_0': "#E74C3C",          # Rouge
    'fib_23': "#E67E22",         # Orange
    'fib_38': "#F1C40F",         # Jaune
    'fib_50': "#2ECC71",         # Vert
    'fib_61': "#3498DB",         # Bleu
    'fib_100': "#8E44AD",        # Violet
}

def get_mpl_style() -> Dict[str, Any]:
    """
    Retourne les paramètres de style pour matplotlib.

    Returns:
        Dictionnaire de paramètres de style
    """
    return {
        'figure.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.color': '#BDC3C7',
        'axes.facecolor': 'white',
        'figure.figsize': (14, 10),
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'sans-serif'],
        'lines.linewidth': 1.5,
        'lines.markersize': 4,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    }

def apply_style():
    """Applique le style aux graphiques matplotlib."""
    import matplotlib.pyplot as plt
    plt.style.use('seaborn')
    plt.rcParams.update(get_mpl_style())
