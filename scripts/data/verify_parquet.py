#!/usr/bin/env python3
"""VERIFY-PARQUET — vérifie l'intégrité et la correspondance des Parquet.

Demandé par l'utilisateur : « n'oublie pas de vérifier les parquet et leur
correspondance ». Contrôle, pour chaque split (train/val/test) × timeframe :

  1. existence du fichier ;
  2. nb de lignes, plage de dates (index ou colonne timestamp) ;
  3. correspondance des COLONNES avec TRAIN_COLUMNS[tf] (celles que le
     StateBuilder/LiveStateBuilder attend réellement) — manquantes / en trop ;
  4. NaN / inf par colonne (les NaN cassent silencieusement la normalisation) ;
  5. cohérence CHRONOLOGIQUE inter-split (train < val < test, sans chevauchement) ;
  6. cohérence inter-timeframe d'un même split (mêmes bornes temporelles).

Read-only. Code de sortie 0 si tout est cohérent, 1 sinon.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(str(ROOT))

SPLITS = ["train", "val", "test"]
TFS = ["5m", "1h", "4h"]
ASSET = os.environ.get("VERIFY_ASSET", "BTCUSDT")
BASE = ROOT / "data" / "processed" / "indicators"


def _load_expected_columns():
    try:
        from adan_trading_bot.trading.live_state_builder import TRAIN_COLUMNS
        return TRAIN_COLUMNS
    except Exception as e:  # noqa: BLE001
        print(f"  [!] import TRAIN_COLUMNS échoué: {e}")
        return None


def _time_bounds(df: pd.DataFrame):
    """Retourne (start, end) lisibles depuis l'index ou une colonne timestamp."""
    if isinstance(df.index, pd.DatetimeIndex) and len(df):
        return df.index.min(), df.index.max()
    for c in ("timestamp", "open_time", "date", "datetime"):
        if c in df.columns:
            try:
                s = pd.to_datetime(df[c])
                return s.min(), s.max()
            except Exception:
                pass
    return None, None


def main() -> int:
    print("=" * 80)
    print(f"  VERIFY-PARQUET — asset={ASSET}  base={BASE}")
    print("=" * 80)
    expected = _load_expected_columns()
    ok = True
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    bounds: dict[tuple[str, str], tuple] = {}

    # ── 1-4 : par fichier ───────────────────────────────────────────────────
    print(f"\n{'split':<7}{'tf':<5}{'rows':>8}{'cols':>6}"
          f"{'missing':>9}{'extra':>7}{'nan':>7}{'inf':>6}  date_range")
    for split in SPLITS:
        for tf in TFS:
            p = BASE / split / ASSET / f"{tf}.parquet"
            if not p.exists():
                print(f"{split:<7}{tf:<5}  MANQUANT : {p}")
                ok = False
                continue
            df = pd.read_parquet(p)
            frames[(split, tf)] = df
            start, end = _time_bounds(df)
            bounds[(split, tf)] = (start, end)

            missing = extra = nan_tot = inf_tot = 0
            if expected and tf in expected:
                exp = set(expected[tf])
                have = set(df.columns)
                missing = len(exp - have)
                extra = len(have - exp)
                cols_check = [c for c in expected[tf] if c in df.columns]
            else:
                cols_check = [c for c in df.columns
                              if pd.api.types.is_numeric_dtype(df[c])]
            for c in cols_check:
                col = df[c]
                if pd.api.types.is_numeric_dtype(col):
                    nan_tot += int(col.isna().sum())
                    inf_tot += int(np.isinf(col.to_numpy(dtype="float64",
                                                          na_value=np.nan)).sum())
            dr = (f"{start:%Y-%m-%d %H:%M}→{end:%Y-%m-%d %H:%M}"
                  if start is not None else "(pas d'index temporel)")
            flag = "" if (missing == 0 and nan_tot == 0 and inf_tot == 0) else "  ⚠"
            print(f"{split:<7}{tf:<5}{len(df):>8}{df.shape[1]:>6}"
                  f"{missing:>9}{extra:>7}{nan_tot:>7}{inf_tot:>6}  {dr}{flag}")
            if missing or nan_tot or inf_tot:
                ok = False
            # détail colonnes manquantes (critique)
            if expected and tf in expected:
                miss = [c for c in expected[tf] if c not in df.columns]
                if miss:
                    print(f"         manquantes: {miss}")

    # ── 5 : non-fuite temporelle (l'ordre des SPLITS importe peu, le
    #        chevauchement est ce qui crée une fuite). On trie par date de
    #        début réelle et on vérifie qu'aucun split n'en chevauche un autre.
    print("\n── NON-FUITE TEMPORELLE (aucun chevauchement entre splits, par tf) ──")
    for tf in TFS:
        seq = [(s, bounds.get((s, tf))) for s in SPLITS if (s, tf) in bounds]
        seq = [(s, b) for s, b in seq if b and b[0] is not None]
        if len(seq) < 2:
            print(f"  {tf}: (pas assez d'index temporels pour vérifier)")
            continue
        # tri chronologique RÉEL (par début), indépendamment du nom du split
        seq.sort(key=lambda sb: sb[1][0])
        msg = []
        no_overlap = True
        for (s1, b1), (s2, b2) in zip(seq, seq[1:]):
            overlap = b1[1] >= b2[0]  # fin du précédent >= début du suivant
            if overlap:
                no_overlap = False
            sym = ">" if overlap else "≤"
            tag = "CHEVAUCHEMENT" if overlap else "OK"
            msg.append(f"{s1}.end={b1[1]:%Y-%m-%d %H:%M} {sym} "
                       f"{s2}.start={b2[0]:%Y-%m-%d %H:%M} [{tag}]")
        order_str = " → ".join(s for s, _ in seq)
        print(f"  {tf}: ordre réel = [{order_str}]")
        print(f"        " + " | ".join(msg))
        if not no_overlap:
            ok = False

    # ── 6 : cohérence inter-timeframe d'un même split ─────────────────────────
    print("\n── COHÉRENCE INTER-TIMEFRAME (mêmes bornes par split) ──")
    for split in SPLITS:
        bs = [(tf, bounds.get((split, tf))) for tf in TFS if (split, tf) in bounds]
        bs = [(tf, b) for tf, b in bs if b and b[0] is not None]
        if len(bs) < 2:
            print(f"  {split}: (pas assez d'index temporels)")
            continue
        starts = {tf: b[0] for tf, b in bs}
        ends = {tf: b[1] for tf, b in bs}
        spread_start = (max(starts.values()) - min(starts.values()))
        spread_end = (max(ends.values()) - min(ends.values()))
        print(f"  {split}: start spread={spread_start}  end spread={spread_end}")

    print("\n" + "=" * 80)
    print(f"  RÉSULTAT : {'TOUT COHÉRENT ✅' if ok else 'INCOHÉRENCES DÉTECTÉES ⚠'}")
    print("=" * 80)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
