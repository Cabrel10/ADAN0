import pandas as pd
import os

# Chemins vers les fichiers de données
base_path = "/home/morningstar/Documents/trading/bot/data/processed/indicators/train/SOLUSDT"
timeframes = ['5m', '1h', '4h']

print("Analyse détaillée des données BTC/USDT par timeframe:")
print("=" * 80)

for tf in timeframes:
    file_path = os.path.join(base_path, f"{tf}.parquet")
    try:
        # Lire le fichier parquet
        df = pd.read_parquet(file_path)
        
        # Calculer le nombre de NaN par colonne
        nan_counts = df.isna().sum()
        total_rows = len(df)
        
        # Afficher les informations
        print(f"\n{'='*80}")
        print(f"TIMEFRAME: {tf}")
        print(f"Nombre de lignes: {total_rows:,}")
        print(f"Nombre de colonnes: {len(df.columns)}")
        
        # Afficher les colonnes avec des NaN
        print("\nColonnes avec des valeurs manquantes (NaN):")
        has_nan = False
        for col, count in nan_counts.items():
            if count > 0:
                has_nan = True
                print(f"- {col}: {count} NaN ({count/total_rows*100:.1f}%)")
        
        if not has_nan:
            print("Aucune valeur manquante détectée.")
            
        # Compter les colonnes entièrement vides
        empty_cols = [col for col in df.columns if df[col].isna().all()]
        if empty_cols:
            print(f"\nColonnes entièrement vides (100% NaN): {len(empty_cols)}")
            for col in empty_cols:
                print(f"- {col}")
        
        # Afficher un aperçu des données
        print("\nAperçu des données (2 premières lignes):")
        print(df.head(2).to_string())
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"ERREUR avec le fichier {file_path}:")
        print(str(e))
    
    print("\n" + "="*80 + "\n")
