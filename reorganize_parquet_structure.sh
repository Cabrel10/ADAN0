#!/bin/bash

# Script pour réorganiser la structure des fichiers Parquet
# De: {split}/{timeframe}/{asset}.parquet
# Vers: {split}/{asset}/{timeframe}.parquet

BASE_DIR="bot/data/processed/indicators"

for split in "train" "test" "val"; do
    for timeframe in "1h" "4h" "5m"; do
        if [ -d "$BASE_DIR/$split/$timeframe" ]; then
            echo "Traitement de $split/$timeframe"
            
            for asset_file in "$BASE_DIR/$split/$timeframe"/*.parquet; do
                if [ -f "$asset_file" ]; then
                    # Extraire le nom de l'asset (sans extension)
                    asset_name=$(basename "$asset_file" .parquet)
                    
                    # Créer le dossier de destination
                    mkdir -p "$BASE_DIR/$split/$asset_name"
                    
                    # Déplacer le fichier
                    mv "$asset_file" "$BASE_DIR/$split/$asset_name/$timeframe.parquet"
                    echo "  Déplacé: $asset_file → $BASE_DIR/$split/$asset_name/$timeframe.parquet"
                fi
            done
            
            # Supprimer le dossier timeframe vide
            rmdir "$BASE_DIR/$split/$timeframe" 2>/dev/null
        fi
    done
done

echo "Réorganisation terminée!"
