# Journal des modifications (Changelog)

## [1.0.1] - 2025-08-06

### Corrigé
- **Gestion des positions** : Correction de la méthode `calculate_position_size` dans `PortfolioManager`
- **Gestion des ordres** : Mise à jour de la méthode `open_position` dans `OrderManager` pour accepter les paramètres `size`, `stop_loss` et `take_profit`
- **Exécution des trades** : Correction des erreurs d'appel de méthode dans `multi_asset_chunked_env.py`
- **Gestion des risques** : Amélioration de la gestion des limites de drawdown et de la taille des positions

## [1.0.0] - 2025-07-27

### Ajouté
- **Environnement Multi-Worker** : Implémentation de l'environnement `MultiAssetChunkedEnv` pour exécuter plusieurs stratégies en parallèle
- **Gestion des configurations** : Système de fusion des configurations globales et spécifiques aux workers
- **Chargement des données** : Support du chargement de données spécifiques à chaque worker
- **Tests d'intégration** : Mise en place de tests pour valider le bon fonctionnement des workers
- **Documentation** : Ajout de guides de configuration et de démarrage rapide

### Modifié
- **Refactorisation** : Amélioration de la structure du code pour une meilleure maintenabilité
- **Gestion des erreurs** : Amélioration des messages d'erreur pour faciliter le débogage

### Corrigé
- **Configuration** : Correction des problèmes de chargement des configurations
- **Tests** : Mise à jour des tests pour refléter les changements dans l'API
- **Documentation** : Mise à jour de la documentation pour refléter les dernières modifications

## [0.1.0] - 2025-07-20

### Ajouté
- **Structure initiale du projet**
- **Configuration de base**
- **Documentation initiale**
