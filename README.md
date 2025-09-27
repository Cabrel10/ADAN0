# GUIDE D'UTILISATION DU PROJET ADAN

Ce guide explique comment utiliser le projet ADAN de bout en bout, du téléchargement des données brutes à leur traitement pour l'entraînement des modèles.

---

## ÉTAPE 1 : TÉLÉCHARGEMENT DES DONNÉES BRUTES

**NOTE IMPORTANTE :** Le script `fetch_data_ccxt.py`, mentionné dans la documentation des scripts, est actuellement manquant. En attendant sa création, les données doivent être téléchargées manuellement.

Les données brutes sont nécessaires pour que le script de traitement puisse générer les fichiers Parquet avec les indicateurs techniques.

### 1. Structure des dossiers

Assurez-vous que le dossier `ADAN/data/raw/` existe. À l'intérieur, créez des sous-dossiers pour chaque timeframe que vous souhaitez utiliser (par exemple, `5m`, `1h`, `4h`).

La structure attendue est la suivante :

```
ADAN/
└── data/
    └── raw/
        ├── 5m/
        │   ├── BTCUSDT.csv
        │   └── ETHUSDT.csv
        │   └── ...
        ├── 1h/
        │   ├── BTCUSDT.csv
        │   └── ...
        └── 4h/
            ├── BTCUSDT.csv
            └── ...
```

### 2. Format des fichiers CSV

Chaque fichier CSV doit contenir les données pour un seul asset et un seul timeframe. Le nom du fichier doit suivre le format `{ASSET}USDT.csv` (par exemple, `BTCUSDT.csv`).

Les colonnes requises dans chaque fichier CSV sont :

-   `timestamp`: L'horodatage de la bougie (format : `YYYY-MM-DD HH:MM:SS`).
-   `open`: Le prix d'ouverture.
-   `high`: Le prix le plus haut.
-   `low`: Le prix le plus bas.
-   `close`: Le prix de clôture.
-   `volume`: Le volume des transactions.

---

## ÉTAPE 2 : TRAITEMENT DES DONNÉES ET GÉNÉRATION DES INDICATEURS

Une fois les données brutes en place, vous pouvez exécuter le script pour calculer les indicateurs techniques et générer les fichiers Parquet.

### 1. Configuration de l'environnement

Assurez-vous que votre environnement Conda `trading_env` est activé. Cet environnement doit contenir toutes les dépendances nécessaires (listées dans `requirements.txt`).

### 2. Exécution du script

Depuis la racine du projet (`/home/morningstar/Documents/trading/`), exécutez la commande suivante dans votre terminal :

```bash
conda activate trading_env && python ADAN/scripts/generate_sample_data.py
```

### 3. Résultat du traitement

Le script va :

-   Lire les fichiers CSV depuis `ADAN/data/raw/`.
-   Calculer les indicateurs techniques définis dans `ADAN/config/config.yaml`.
-   Sauvegarder les données traitées sous forme de fichiers Parquet dans le dossier `ADAN/data/processed/indicators/`.

Les fichiers de sortie seront organisés par asset, comme ceci :

```
ADAN/
└── data/
    └── processed/
        └── indicators/
            ├── ADA/
            │   ├── 1h.parquet
            │   ├── 4h.parquet
            │   └── 5m.parquet
            └── ...
```

Le projet est maintenant prêt pour l'entraînement des modèles d'intelligence artificielle.
