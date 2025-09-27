import os
import pandas as pd
import pyarrow.parquet as pq

def inspect_parquet(file_path):
    """
    Renvoie un dict d'informations clés sur le fichier Parquet.
    """
    info = {'exists': False}
    if not os.path.exists(file_path):
        return info

    info['exists'] = True

    # Metadata via PyArrow (rapide, sans charger les données)
    pf = pq.ParquetFile(file_path)
    meta = pf.metadata
    info['num_rows'] = meta.num_rows
    info['num_columns'] = meta.num_columns
    info['num_row_groups'] = meta.num_row_groups

    # Stats du premier row group (si disponibles)
    rg_meta = meta.row_group(0)
    stats = {}
    for i in range(rg_meta.num_columns):
        col_meta = rg_meta.column(i)
        col_name = col_meta.path_in_schema
        if col_meta.is_stats_set:
            st = col_meta.statistics
            stats[col_name] = {
                'min': st.min,
                'max': st.max,
                'null_count': st.null_count
            }
    info['stats'] = stats

    # Aperçu avec Pandas pour diagnostic plus fin
    df = pd.read_parquet(file_path)
    info['columns'] = df.columns.tolist()
    info['nans'] = df.isnull().sum().to_dict()
    info['first_rows'] = df.head(3).to_dict(orient='list')
    return info

def detailed_diagnostics_all(base_path='bot/data/processed/backups'):
    splits = ['train', 'test', 'val']
    assets = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT', 'XRPUSDT']
    timeframes = ['5m', '1h', '4h']
    indicator_cols = ['RSI_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3', 'MACD_HIST',
                      'ATR_14', 'EMA_5', 'EMA_12', 'BB_UPPER', 'BB_MIDDLE', 'BB_LOWER']

    for split in splits:
        print(f"\n=== SPLIT: {split.upper()} ===")
        for asset in assets:
            print(f"\n-- Actif: {asset} --")
            for tf in timeframes:
                file_path = os.path.join(base_path, split, asset, f"{tf}.parquet")
                print(f"\n> Timeframe: {tf} — Fichier: {file_path}")
                info = inspect_parquet(file_path)

                if not info.get('exists'):
                    print("  ❌ Fichier non trouvé.")
                    continue

                print(f"  Lignes: {info['num_rows']}, Colonnes: {info['num_columns']}, RowGroups: {info['num_row_groups']}")
                print(f"  Colonnes: {info['columns']}")
                print("  Aperçu des 3 premières lignes (par colonne):")
                for col, vals in info['first_rows'].items():
                    print(f"    {col}: {vals}")

                # Stats statistiques rapides
                print("  Statistiques (row-group 0):")
                for col, st in info['stats'].items():
                    print(f"    {col}: min={st['min']}, max={st['max']}, nulls={st['null_count']}")

                print("  Valeurs manquantes (NaN) par colonne:")
                for col, cnt in info['nans'].items():
                    pct = cnt / info['num_rows'] * 100 if info['num_rows'] else 0
                    print(f"    {col}: {cnt}/{info['num_rows']} ({pct:.1f}%)")

                # Statistiques des indicateurs si présents
                df = pd.read_parquet(file_path)
                print("  Stats colonnes indicateurs:")
                for col in indicator_cols:
                    if col in df.columns:
                        vals = df[col].dropna()
                        if len(vals) > 0:
                            print(f"    {col}: min={vals.min():.4f}, max={vals.max():.4f}, count={len(vals)}")
                        else:
                            print(f"    {col}: aucune valeur valide (toutes NaN)")

