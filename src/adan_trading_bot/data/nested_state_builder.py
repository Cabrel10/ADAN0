"""
nested_state_builder.py — ADAN-System-One · Étape 1 (État temporel)
====================================================================

Principe cardinal du workflow : **la clôture 5m est l'horloge maîtresse
unique**. Les contenants 1h et 4h ne sont jamais des bougies figées chargées
depuis un autre fichier — ce sont des **contenants vivants en cours de
formation**, reconstruits depuis la séquence 5m et tronqués à la seconde
exacte de clôture de la 5m observée.

    1 contenant 1h  = 12 × 5m   (phase k = minute//5 + 1,  k ∈ 1..12)
    1 contenant 4h  = 48 × 5m   (phase m = (heure%4)*12 + k, m ∈ 1..48)

Conséquence structurelle : le tenseur d'état est **totalement causal** —
il est impossible qu'il contienne une fuite de données futures.

Pourquoi ce module remplace le MTF statique (erreur documentée) :
  L'ancien ADAN injectait les bougies 1h/4h DÉJÀ FERMÉES (latence déguisée
  en confluence). Ici, à k=7/12 par exemple, le contenant 1h n'est qu'à
  58 % de sa formation : son high/low/range sont RUNNING, pas définitifs.

Référence : dev.md §ADAN-SYSTEM-ONE, workflow étape 1 + LivingStateSnapshot.
Validé par Phase 0b (edge des contenants confirmé POSITIF sur test).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ─── Dimensions canoniques ────────────────────────────────────────────────────
BARS_PER_1H = 12      # 12 × 5m
BARS_PER_4H = 48      # 48 × 5m
LOOKBACK_SEQ = 36     # fenêtre de la séquence 5m (features micro)
N_FEATURES_5M = 8     # [O, H, L, C, V, mèche_haute, mèche_basse, corps]


@dataclass(frozen=True)
class LivingStateSnapshot:
    """Tenseur d'état vivant S_t — extrait à la clôture de la 5m d'indice t.

    Garantie causale : seules les barres d'indice ≤ t sont utilisées.
    """

    timestamp: pd.Timestamp
    price: float

    # 1. Précision : la 5m qui vient de fermer + sa séquence récente
    bar_5m: np.ndarray          # shape (8,) — [O, H, L, C, V, wick_up, wick_dn, body]
    seq_5m: np.ndarray          # shape (LOOKBACK_SEQ, 8)

    # 2. Perception : contenant 1h EN COURS de formation
    phase_1h: float             # k / 12  (0.083 … 1.0)
    running_1h: np.ndarray      # [open_1h, running_high, running_low, running_vol]
    pos_in_1h: float            # (close_5m − running_low) / running_range  ∈ [0,1]
    sweep_high_1h: int          # 1 si high_5m > high_1h_précédent ET close réintègre
    sweep_low_1h: int           # 1 si low_5m < low_1h_précédent ET close réintègre
    prev_1h: np.ndarray         # [open, high, low, close, volume] du contenant 1h FERMÉ précédent

    # 3. Horizon : contenant 4h EN COURS de formation
    phase_4h: float             # m / 48  (0.021 … 1.0)
    running_4h: np.ndarray      # [open_4h, running_high, running_low, running_vol]
    pos_in_4h: float            # position dans le range 4h courant ∈ [0,1]
    sweep_high_4h: int          # balayage de l'extrême haut de la 4h précédente
    sweep_low_4h: int           # balayage de l'extrême bas de la 4h précédente
    prev_4h: np.ndarray         # [open, high, low, close, volume] du contenant 4h FERMÉ précédent

    # 4. Proprioception : état du trader (injecté par l'appelant)
    portfolio: np.ndarray       # [cash, exposure, pnl_latent, trades_today, cooldown_step]

    # 5. Intégrité (étape 2) : False → ABSTENTION obligatoire
    integrity_ok: bool = True


class NestedStateBuilder:
    """Construit les LivingStateSnapshot depuis une timeline 5m unique.

    Deux modes :
      - `build_series(df)`   : pré-calcul vectorisé pour le backtest/recherche
                               (retourne des tableaux de contenants alignés) ;
      - `snapshot(i)`        : extraction causale à la clôture de la barre i,
                               utilisable en live (barre par barre).

    Invariants vérifiés (étape 2 — contrôle d'intégrité) :
      OHLC cohérent (H ≥ max(O,C), L ≤ min(O,C)), pas de NaN/inf, volume ≥ 0,
      historique minimum (48 barres 4h + 1 contenant fermé de chaque échelle).
    """

    MIN_BARS = 2 * BARS_PER_4H + 1   # 4h courante + 4h précédente + marge

    def __init__(self, df: pd.DataFrame):
        """df : DataFrame 5m indexé par timestamp, colonnes open/high/low/close/volume."""
        cols = ["open", "high", "low", "close", "volume"]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes dans la timeline 5m : {missing}")
        self.df = df
        self.o = df["open"].to_numpy(np.float64)
        self.h = df["high"].to_numpy(np.float64)
        self.l = df["low"].to_numpy(np.float64)
        self.c = df["close"].to_numpy(np.float64)
        self.v = df["volume"].to_numpy(np.float64)
        self.ts = df.index
        self.n = len(df)

        # Phases dans les contenants (vectorisées — positions absolues dans la grille)
        minutes = self.ts.minute.to_numpy()
        hours = self.ts.hour.to_numpy()
        self.k_1h = minutes // 5 + 1                      # 1..12
        self.m_4h = (hours % 4) * BARS_PER_1H + self.k_1h  # 1..48

        # Rang de chaque barre dans son contenant (0-based) : k-1 et m-1
        self._pos_1h = self.k_1h - 1
        self._pos_4h = self.m_4h - 1

        self._precompute_containers()

    # ─── Pré-calcul vectorisé des contenants vivants ──────────────────────────
    def _precompute_containers(self):
        """Calcule, pour CHAQUE barre i, l'état RUNNING de ses contenants
        1h/4h (jusqu'à i inclus) et les niveaux des contenants FERMÉS
        précédents — le tout strictement causal."""
        n = self.n

        # Identifiant entier de chaque contenant (changement → nouveau groupe).
        # pandas 2+/3 : ts.view('int64') supprimé — conversion explicite via numpy.
        ts_ns = self.ts.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        # Contenant 1h : l'heure civile ; contenant 4h : bloc de 4 heures.
        id_1h = ts_ns // (3600 * 10**9)
        id_4h = ts_ns // (4 * 3600 * 10**9)

        def running(grp_id):
            """OHLCV running du contenant en cours, à chaque barre incluse."""
            # open du contenant = open de la 1re barre du groupe
            first_idx = pd.Series(np.arange(n)).groupby(grp_id).transform("first")
            open_run = self.o[first_idx.to_numpy()]
            high_run = pd.Series(self.h).groupby(grp_id).cummax().to_numpy()
            low_run = pd.Series(self.l).groupby(grp_id).cummin().to_numpy()
            vol_run = pd.Series(self.v).groupby(grp_id).cumsum().to_numpy()
            return open_run, high_run, low_run, vol_run

        self.run_1h = running(id_1h)
        self.run_4h = running(id_4h)

        # Niveaux des contenants FERMÉS précédents (complets, donc causaux)
        def prev_closed(grp_id):
            g_o = pd.Series(self.o).groupby(grp_id).first()
            g_h = pd.Series(self.h).groupby(grp_id).max()
            g_l = pd.Series(self.l).groupby(grp_id).min()
            g_c = pd.Series(self.c).groupby(grp_id).last()
            g_v = pd.Series(self.v).groupby(grp_id).sum()
            closed = pd.DataFrame({"o": g_o, "h": g_h, "l": g_l, "c": g_c, "v": g_v})
            closed_prev = closed.shift(1)  # contenant précédent (NaN pour le 1er)
            # ré-indexe sur chaque barre via son id de contenant
            aligned = closed_prev.reindex(grp_id).to_numpy()
            return aligned

        self.prev_1h = prev_closed(id_1h)   # (n, 5) — NaN si pas de contenant précédent
        self.prev_4h = prev_closed(id_4h)

        # Séquence 5m brute (O,H,L,C,V,wicks,corps) — fenêtre LOOKBACK_SEQ
        wick_up = self.h - np.maximum(self.o, self.c)
        wick_dn = np.minimum(self.o, self.c) - self.l
        body = self.c - self.o
        self.bars_5m = np.column_stack([self.o, self.h, self.l, self.c, self.v,
                                        wick_up, wick_dn, body])

    # ─── Intégrité (étape 2) ─────────────────────────────────────────────────
    def _integrity(self, i: int) -> bool:
        if i < self.MIN_BARS:
            return False
        win = self.bars_5m[i - BARS_PER_4H:i + 1]
        if not np.isfinite(win).all():
            return False
        o, h, l, v = self.o[i], self.h[i], self.l[i], self.v[i]
        if h < max(o, self.c[i]) or l > min(o, self.c[i]) or l > h or v < 0:
            return False
        if not np.isfinite(self.prev_1h[i]).all() or not np.isfinite(self.prev_4h[i]).all():
            return False
        return True

    # ─── Extraction du snapshot à la clôture de la barre i ───────────────────
    def snapshot(self, i: int,
                 portfolio: Optional[np.ndarray] = None) -> LivingStateSnapshot:
        """Construit S_i — strictement causal (barres ≤ i uniquement)."""
        if portfolio is None:
            portfolio = np.zeros(5, dtype=np.float64)

        o1, h1, l1, v1 = (self.run_1h[k][i] for k in range(4))
        o4, h4, l4, v4 = (self.run_4h[k][i] for k in range(4))
        close = self.c[i]

        rng1 = max(h1 - l1, 1e-12)
        rng4 = max(h4 - l4, 1e-12)

        # Sweeps vs contenants FERMÉS précédents (réintégration exigée)
        ph1, pl1 = self.prev_1h[i][1], self.prev_1h[i][2]
        ph4, pl4 = self.prev_4h[i][1], self.prev_4h[i][2]
        sweep_h1 = int(np.isfinite(ph1) and self.h[i] > ph1 and close < ph1)
        sweep_l1 = int(np.isfinite(pl1) and self.l[i] < pl1 and close > pl1)
        sweep_h4 = int(np.isfinite(ph4) and self.h[i] > ph4 and close < ph4)
        sweep_l4 = int(np.isfinite(pl4) and self.l[i] < pl4 and close > pl4)

        seq_start = max(0, i - LOOKBACK_SEQ + 1)
        seq = self.bars_5m[seq_start:i + 1]
        if len(seq) < LOOKBACK_SEQ:  # padding zéro à gauche (début de série)
            seq = np.vstack([np.zeros((LOOKBACK_SEQ - len(seq), N_FEATURES_5M)), seq])

        return LivingStateSnapshot(
            timestamp=self.ts[i],
            price=close,
            bar_5m=self.bars_5m[i].copy(),
            seq_5m=seq,
            phase_1h=self.k_1h[i] / BARS_PER_1H,
            running_1h=np.array([o1, h1, l1, v1]),
            pos_in_1h=float(np.clip((close - l1) / rng1, 0.0, 1.0)),
            sweep_high_1h=sweep_h1,
            sweep_low_1h=sweep_l1,
            prev_1h=self.prev_1h[i].copy(),
            phase_4h=self.m_4h[i] / BARS_PER_4H,
            running_4h=np.array([o4, h4, l4, v4]),
            pos_in_4h=float(np.clip((close - l4) / rng4, 0.0, 1.0)),
            sweep_high_4h=sweep_h4,
            sweep_low_4h=sweep_l4,
            prev_4h=self.prev_4h[i].copy(),
            portfolio=np.asarray(portfolio, dtype=np.float64),
            integrity_ok=self._integrity(i),
        )

    def __len__(self):
        return self.n
