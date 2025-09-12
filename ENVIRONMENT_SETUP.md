# Configuration des Variables d'Environnement

## Aperçu

Ce guide explique comment configurer de manière sécurisée les clés API en utilisant des variables d'environnement au lieu de fichiers codés en dur.

## Variables d'Environnement Requises

### Binance
```bash
export BINANCE_API_KEY="votre_cle_api_binance"
export BINANCE_API_SECRET="votre_secret_api_binance"
export BINANCE_SANDBOX="true"  # Mettez à "false" pour la production
```

### Binance Futures
```bash
export BINANCE_FUTURES_API_KEY="votre_cle_api_binance_futures"
export BINANCE_FUTURES_API_SECRET="votre_secret_api_binance_futures"
export BINANCE_FUTURES_SANDBOX="true"  # Mettez à "false" pour la production
```

### Bitget
```bash
export BITGET_API_KEY="votre_cle_api_bitget"
export BITGET_API_SECRET="votre_secret_api_bitget"
export BITGET_PASSPHRASE="votre_phrase_secrete_bitget"
export BITGET_SANDBOX="true"  # Mettez à "false" pour la production
```

### Bybit
```bash
export BYBIT_API_KEY="votre_cle_api_bybit"
export BYBIT_API_SECRET="votre_secret_api_bybit"
export BYBIT_SANDBOX="true"  # Mettez à "false" pour la production
```

### OKEx
```bash
export OKEX_API_KEY="votre_cle_api_okex"
export OKEX_API_SECRET="votre_secret_api_okex"
export OKEX_PASSPHRASE="votre_phrase_secrete_okex"
export OKEX_SANDBOX="true"  # Mettez à "false" pour la production
```

### Kraken
```bash
export KRAKEN_API_KEY="votre_cle_api_kraken"
export KRAKEN_API_SECRET="votre_secret_api_kraken"
export KRAKEN_SANDBOX="true"  # Mettez à "false" pour la production
```

## Méthodes de Configuration

### Méthode 1 : Fichier .env (Recommandé pour le Développement)

1. Créez un fichier `.env` à la racine de votre projet :
```bash
# Fichier .env
BINANCE_API_KEY=votre_cle_api_binance
BINANCE_API_SECRET=votre_secret_api_binance
BINANCE_SANDBOX=true
```

2. Ajoutez `.env` à votre fichier `.gitignore` pour éviter de commettre des secrets :
```bash
echo ".env" >> .gitignore
```

3. Chargez le fichier .env dans votre application :
```python
from dotenv import load_dotenv
load_dotenv()
```

### Méthode 2 : Variables d'Environnement Système

#### Linux/macOS
Ajoutez à votre `~/.bashrc` ou `~/.zshrc` :
```bash
export BINANCE_API_KEY="votre_cle_api_binance"
export BINANCE_API_SECRET="votre_secret_api_binance"
```

Puis rechargez votre shell :
```bash
source ~/.bashrc  # ou ~/.zshrc
```

#### Windows
```cmd
setx BINANCE_API_KEY "votre_cle_api_binance"
setx BINANCE_API_SECRET "votre_secret_api_binance"
```

### Méthode 3 : Variables d'Environnement Docker

```dockerfile
ENV BINANCE_API_KEY=votre_cle_api_binance
ENV BINANCE_API_SECRET=votre_secret_api_binance
```

Ou en utilisant docker-compose :
```yaml
environment:
  - BINANCE_API_KEY=votre_cle_api_binance
  - BINANCE_API_SECRET=votre_secret_api_binance
```

## Bonnes Pratiques de Sécurité

1. **Ne jamais commettre de clés API dans le contrôle de version**
2. **Utilisez des clés différentes pour le développement et la production**
3. **Faites tourner régulièrement vos clés API**
4. **Définissez des permissions appropriées sur vos comptes d'échange**
5. **Surveillez l'utilisation de vos clés API et configurez des alertes**
6. **Utilisez des environnements sandbox/testnet pour le développement**

## Vérification

Pour vérifier que vos variables d'environnement sont correctement définies :

```bash
# Vérifiez si les variables sont définies (affichera des valeurs masquées)
echo $BINANCE_API_KEY | sed 's/./*/g'
echo $BINANCE_API_SECRET | sed 's/./*/g'
```

## Migration depuis des Fichiers Codés en Dur

Si vous avez des clés API codées en dur :

1. Copiez vos clés API dans des variables d'environnement en utilisant l'une des méthodes ci-dessus
2. Supprimez tous les fichiers contenant des clés codées en dur (par exemple, `gemini_api_keys.txt`)
3. Redémarrez votre application
4. Vérifiez que l'application charge correctement les identifiants depuis les variables d'environnement

## Dépannage

### Problèmes Courants

1. **Variables d'environnement non chargées** : Assurez-vous de redémarrer votre terminal/application après avoir défini les variables
2. **Permission refusée** : Vérifiez que vos clés API ont les bonnes permissions sur l'échange
3. **Mode Sandbox** : Assurez-vous que les variables `SANDBOX` sont définies à "true" pour les tests
