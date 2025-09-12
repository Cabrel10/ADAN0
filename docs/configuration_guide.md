# Guide de Configuration des Workers

## Structure de base

Chaque worker est défini dans le fichier `config.yaml` sous la section `workers`. Voici la structure de base :

```yaml
workers:
  w1:  # Identifiant unique du worker
    name: "Nom du Worker"
    description: "Description détaillée"
    assets: ["BTC", "ETH"]  # Actifs à trader
    timeframes: ["5m", "1h"]  # Périodes temporelles
    data_split: "train"  # train/validation/test
    
    # Configuration des récompenses
    reward_config:
      short_trade_bonus: 0.1
      holding_penalty: -0.001
      
    # Configuration du moteur de comportement dynamique
    dbe_config:
      volatility_impact: 2.0
      drawdown_risk_multiplier: 1.5
      
    # Configuration spécifique à l'agent
    agent_config:
      learning_rate: 3e-4
      ent_coef: 0.05
```

## Configuration requise

### Actifs et Timeframes
- `assets`: Liste des symboles d'actifs à trader (ex: ["BTC", "ETH"])
- `timeframes`: Liste des périodes temporelles (ex: ["5m", "1h", "4h"])
- `data_split`: Jeu de données à utiliser ("train", "validation" ou "test")

### Répertoires de données
Le système recherche les fichiers de données dans les emplacements suivants :
1. `data/processed/{data_split}/indicators/{asset}/{timeframe}.parquet`
2. `data/processed/indicators/{asset}/{timeframe}.parquet` (fallback)

## Exemples de configuration

### Worker de haute fréquence
```yaml
w1:
  name: "Temporal Precision"
  description: "Scalping optimisé sur micro-régimes de marché"
  assets: ["BTC", "ETH"]
  timeframes: ["5m"]
  data_split: "train"
  reward_config:
    short_trade_bonus: 0.1
    holding_penalty: -0.001
```

### Worker basse fréquence
```yaml
w2:
  name: "Low-Frequency Sentinel"
  description: "Stratégie de suivi de tendance"
  assets: ["BTC", "ETH", "SOL"]
  timeframes: ["1h", "4h"]
  data_split: "train"
  reward_config:
    drawdown_penalty_multiplier: 1.5
    holding_bonus: 0.0005
```

## Bonnes pratiques

1. **Configuration minimale** : Seules les clés `assets` et `timeframes` sont obligatoires.
2. **Héritage** : Les valeurs non spécifiées héritent de la configuration par défaut.
3. **Validation** : La configuration est validée au démarrage pour détecter les erreurs.
4. **Logs** : Consultez les logs pour identifier les problèmes de configuration.

## Dépannage

### Fichiers manquants
Si vous voyez des avertissements concernant des fichiers manquants :
1. Vérifiez que les fichiers existent dans `data/processed/indicators/`
2. Vérifiez les noms des actifs et des timeframes
3. Assurez-vous que les données ont été générées avec `scripts/generate_sample_data.py`

### Problèmes de configuration
1. Vérifiez l'indentation YAML
2. Assurez-vous que toutes les valeurs sont correctement typées
3. Consultez les logs pour des messages d'erreur détaillés
