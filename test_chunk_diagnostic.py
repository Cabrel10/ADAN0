#!/usr/bin/env python3
"""
Script de diagnostic pour les problèmes de chunks dans le système de trading.

Ce script teste et diagnostique :
1. La structure des fichiers de données
2. Le chargement des données par timeframe
3. Le calcul du nombre de chunks
4. La configuration des chemins

Usage:
    cd trading/
    python test_chunk_diagnostic.py
"""

import sys
import os
import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, List

# Ajouter le chemin du bot
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'bot', 'src'))

try:
    from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader
except ImportError as e:
    print(f"Erreur d'import: {e}")
    sys.exit(1)

class ChunkDiagnostic:
    """Diagnostic du système de chunks."""

    def __init__(self):
        self.config_path = Path("bot/config/config.yaml")
        self.config = self.load_config()

    def load_config(self) -> dict:
        """Charge la configuration."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"❌ Erreur chargement config: {e}")
            return {}

    def diagnose_data_structure(self):
        """Diagnostic de la structure des données."""
        print("🔍 DIAGNOSTIC STRUCTURE DES DONNÉES")
        print("=" * 50)

        # Chemins de base
        base_data_dir = Path("data/processed/indicators")
        splits = ['train', 'val', 'test']

        print(f"📁 Répertoire de base: {base_data_dir}")
        print(f"   Existe: {base_data_dir.exists()}")

        if not base_data_dir.exists():
            print("❌ Répertoire de base des données n'existe pas!")
            return

        # Vérifier chaque split
        for split in splits:
            split_dir = base_data_dir / split
            print(f"\n📂 Split '{split}': {split_dir}")
            print(f"   Existe: {split_dir.exists()}")

            if split_dir.exists():
                # Lister les actifs
                asset_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
                print(f"   Actifs trouvés: {len(asset_dirs)}")

                for asset_dir in asset_dirs[:3]:  # Limiter à 3 pour l'affichage
                    print(f"     - {asset_dir.name}/")

                    # Lister les timeframes
                    parquet_files = list(asset_dir.glob("*.parquet"))
                    timeframes = [f.stem for f in parquet_files]
                    print(f"       Timeframes: {timeframes}")

                    # Taille des fichiers
                    for pf in parquet_files:
                        size_mb = pf.stat().st_size / (1024*1024)
                        print(f"       {pf.name}: {size_mb:.1f} MB")

                if len(asset_dirs) > 3:
                    print(f"     ... et {len(asset_dirs) - 3} autres actifs")

    def test_data_loading(self):
        """Test du chargement des données."""
        print("\n🧪 TEST CHARGEMENT DES DONNÉES")
        print("=" * 50)

        # Configuration du data loader
        assets = self.config.get('environment', {}).get('assets', ['btcusdt'])
        timeframes = self.config.get('data', {}).get('timeframes', ['5m', '1h', '4h'])

        print(f"📋 Configuration:")
        print(f"   Actifs: {assets}")
        print(f"   Timeframes: {timeframes}")

        # Test pour chaque combinaison
        for asset in assets:
            print(f"\n🔍 Test actif: {asset}")

            for tf in timeframes:
                print(f"  📊 Timeframe {tf}:")

                try:
                    # Test direct du chemin
                    base_dir = Path("data/processed/indicators/train")

                    # Variantes de nommage à tester
                    asset_variants = [
                        asset.lower(),
                        asset.upper(),
                        asset.replace('/', '').replace('-', '').lower(),
                        asset.replace('/', '').replace('-', '').upper()
                    ]

                    tf_variants = [tf.lower(), tf.upper()]

                    found_file = None
                    for asset_var in asset_variants:
                        for tf_var in tf_variants:
                            test_path = base_dir / asset_var / f"{tf_var}.parquet"
                            if test_path.exists():
                                found_file = test_path
                                break
                        if found_file:
                            break

                    if found_file:
                        print(f"    ✅ Fichier trouvé: {found_file}")

                        # Tester le chargement
                        df = pd.read_parquet(found_file)
                        print(f"    📈 Lignes: {len(df)}")
                        print(f"    📊 Colonnes: {list(df.columns)}")

                        # Vérifier la période de données
                        if hasattr(df.index, 'min'):
                            print(f"    📅 Période: {df.index.min()} → {df.index.max()}")

                    else:
                        print(f"    ❌ Fichier non trouvé")
                        print(f"       Variantes testées actif: {asset_variants}")
                        print(f"       Variantes testées TF: {tf_variants}")

                except Exception as e:
                    print(f"    ❌ Erreur: {e}")

    def test_chunk_calculation(self):
        """Test du calcul des chunks."""
        print("\n🧮 TEST CALCUL DES CHUNKS")
        print("=" * 50)

        try:
            # Créer un data loader pour test
            test_config = {
                'paths': {
                    'processed_data_dir': 'data/processed'
                },
                'data': {
                    'data_dirs': {
                        'base': 'data/processed/indicators',
                        'train': 'data/processed/indicators/train'
                    },
                    'timeframes': ['5m', '1h', '4h']
                },
                'environment': {
                    'assets': ['btcusdt'],
                    'max_chunks_per_episode': 10
                }
            }

            print("📋 Configuration de test créée")

            # Test manuel du calcul
            assets = ['btcusdt']
            timeframes = ['5m', '1h', '4h']
            chunk_sizes = {'5m': 1000, '1h': 1000, '4h': 1000}  # Tailles par défaut

            print(f"🎯 Test manuel:")
            print(f"   Actifs: {assets}")
            print(f"   Timeframes: {timeframes}")
            print(f"   Tailles chunks: {chunk_sizes}")

            min_chunks = float('inf')
            successful_loads = 0

            for asset in assets:
                print(f"\n  🔍 Actif: {asset}")

                for tf in timeframes:
                    print(f"    📊 Timeframe {tf}:")

                    try:
                        # Chemin direct
                        file_path = Path(f"data/processed/indicators/train/{asset.upper()}/{tf}.parquet")
                        if not file_path.exists():
                            file_path = Path(f"data/processed/indicators/train/{asset.lower()}/{tf}.parquet")

                        if file_path.exists():
                            df = pd.read_parquet(file_path)
                            data_size = len(df)
                            chunk_size = chunk_sizes.get(tf, 1000)
                            num_chunks = max(1, data_size // chunk_size)

                            print(f"      ✅ Données: {data_size} lignes")
                            print(f"      📦 Taille chunk: {chunk_size}")
                            print(f"      🔢 Nombre chunks: {num_chunks}")

                            if num_chunks < min_chunks:
                                min_chunks = num_chunks

                            successful_loads += 1

                        else:
                            print(f"      ❌ Fichier non trouvé: {file_path}")

                    except Exception as e:
                        print(f"      ❌ Erreur: {e}")

            print(f"\n🎯 RÉSULTAT:")
            print(f"   Chargements réussis: {successful_loads}/{len(assets) * len(timeframes)}")

            if min_chunks != float('inf'):
                print(f"   Chunks minimum calculé: {min_chunks}")
            else:
                print(f"   ❌ Impossible de calculer les chunks!")

            # Comparaison avec config
            config_max = test_config.get('environment', {}).get('max_chunks_per_episode', 'Non défini')
            print(f"   Config max_chunks_per_episode: {config_max}")

        except Exception as e:
            print(f"❌ Erreur dans le test: {e}")

    def test_data_loader_integration(self):
        """Test d'intégration avec le ChunkedDataLoader."""
        print("\n🔗 TEST INTÉGRATION DATA LOADER")
        print("=" * 50)

        try:
            # Configuration simplifiée
            config = {
                'paths': {
                    'processed_data_dir': 'data/processed'
                },
                'data': {
                    'data_dirs': {
                        'base': 'data/processed/indicators',
                        'train': 'data/processed/indicators/train'
                    },
                    'timeframes': ['5m', '1h', '4h']
                },
                'environment': {
                    'assets': ['btcusdt'],
                    'max_chunks_per_episode': 10
                }
            }

            print("🏗️  Création du ChunkedDataLoader...")

            # Test avec données train
            loader = ChunkedDataLoader(
                config=config,
                assets=['btcusdt'],
                timeframes=['5m', '1h', '4h'],
                data_split='train'
            )

            print(f"✅ ChunkedDataLoader créé")
            print(f"📊 Total chunks détecté: {loader.total_chunks}")

            if loader.total_chunks > 0:
                print(f"🎯 Test chargement chunk 0...")

                # Tester le chargement du premier chunk
                chunk_data = loader.get_chunk_data(0)

                if chunk_data:
                    print(f"✅ Chunk 0 chargé avec succès")
                    print(f"📊 Actifs dans le chunk: {list(chunk_data.keys())}")

                    for asset, timeframes_data in chunk_data.items():
                        print(f"   {asset}:")
                        for tf, df in timeframes_data.items():
                            print(f"     {tf}: {len(df)} lignes")
                else:
                    print(f"❌ Échec du chargement du chunk 0")
            else:
                print(f"❌ Aucun chunk détecté par le loader")

        except Exception as e:
            print(f"❌ Erreur intégration: {e}")
            import traceback
            traceback.print_exc()

    def run_full_diagnostic(self):
        """Lance le diagnostic complet."""
        print("🩺 DIAGNOSTIC COMPLET DU SYSTÈME DE CHUNKS")
        print("=" * 60)

        self.diagnose_data_structure()
        self.test_data_loading()
        self.test_chunk_calculation()
        self.test_data_loader_integration()

        print("\n🎯 RÉCAPITULATIF")
        print("=" * 30)
        print("Si des problèmes ont été détectés:")
        print("1. Vérifiez la structure des données dans data/processed/indicators/")
        print("2. Vérifiez les noms d'actifs dans la configuration")
        print("3. Vérifiez les permissions des fichiers")
        print("4. Vérifiez la cohérence entre config et noms de fichiers")

def main():
    """Fonction principale."""
    diagnostic = ChunkDiagnostic()
    diagnostic.run_full_diagnostic()

if __name__ == "__main__":
    main()
