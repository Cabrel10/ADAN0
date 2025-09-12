#!/usr/bin/env python3
"""
Script de statut pour le connecteur d'exchange ADAN.
Vérifie l'état de la connexion au Binance Testnet et affiche un rapport détaillé.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import traceback

# Ajouter le répertoire src au PYTHONPATH
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

try:
    from src.adan_trading_bot.common.utils import load_config
    from src.adan_trading_bot.exchange_api.connector import (
        get_exchange_client,
        validate_exchange_config,
        ExchangeConnectionError,
        ExchangeConfigurationError
    )
    import ccxt
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    print("Assurez-vous d'être dans le répertoire racine ADAN et que l'environnement conda est activé")
    sys.exit(1)

def print_header():
    """Affiche l'en-tête du rapport de statut."""
    print("=" * 80)
    print(" 🔌 ADAN EXCHANGE CONNECTOR - RAPPORT DE STATUT")
    print("=" * 80)
    print(f" Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Système: Binance Testnet Connection Check")
    print("=" * 80)

def check_environment():
    """Vérifie l'environnement et les prérequis."""
    print("\n📋 VÉRIFICATION DE L'ENVIRONNEMENT")
    print("-" * 50)

    status = {
        'conda_env': False,
        'ccxt_available': False,
        'api_keys': False,
        'config_file': False
    }

    # Vérifier l'environnement conda
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'Non défini')
    if conda_env == 'trading_env':
        print("✅ Environnement Conda: trading_env (OK)")
        status['conda_env'] = True
    else:
        print(f"⚠️  Environnement Conda: {conda_env} (Recommandé: trading_env)")

    # Vérifier CCXT
    try:
        print(f"✅ CCXT Version: {ccxt.__version__}")
        status['ccxt_available'] = True
    except Exception as e:
        print(f"❌ CCXT: Erreur - {e}")

    # Vérifier les variables d'environnement
    api_key = os.environ.get("BINANCE_TESTNET_API_KEY")
    secret_key = os.environ.get("BINANCE_TESTNET_SECRET_KEY")

    if api_key and secret_key:
        print(f"✅ API Key: {api_key[:8]}... (Masquée)")
        print(f"✅ Secret Key: {secret_key[:8]}... (Masquée)")
        status['api_keys'] = True
    else:
        print("❌ Clés API: Non définies")
        print("   Commandes pour définir:")
        print("   export BINANCE_TESTNET_API_KEY='VOTRE_CLE'")
        print("   export BINANCE_TESTNET_SECRET_KEY='VOTRE_SECRET'")

    # Vérifier le fichier de configuration
    config_path = project_root / "config" / "main_config.yaml"
    if config_path.exists():
        print(f"✅ Fichier de config: {config_path}")
        status['config_file'] = True
    else:
        print(f"❌ Fichier de config: {config_path} (Non trouvé)")

    return status

def check_configuration():
    """Vérifie la configuration du paper trading."""
    print("\n⚙️  VÉRIFICATION DE LA CONFIGURATION")
    print("-" * 50)

    try:
        config_path = project_root / "config" / "main_config.yaml"
        config = load_config(str(config_path))

        paper_config = config.get('paper_trading', {})
        if paper_config:
            print("✅ Section paper_trading trouvée:")
            print(f"   - exchange_id: {paper_config.get('exchange_id', 'Non défini')}")
            print(f"   - use_testnet: {paper_config.get('use_testnet', 'Non défini')}")

            # Vérifier la configuration complète
            if paper_config.get('exchange_id') == 'binance' and paper_config.get('use_testnet') is True:
                print("✅ Configuration correcte pour Binance Testnet")
                return True, config
            else:
                print("⚠️  Configuration incomplète ou incorrecte")
                return False, config
        else:
            print("❌ Section paper_trading manquante")
            return False, config

    except Exception as e:
        print(f"❌ Erreur lors du chargement de la configuration: {e}")
        return False, None

def test_connection():
    """Teste la connexion au Binance Testnet."""
    print("\n🌐 TEST DE CONNEXION BINANCE TESTNET")
    print("-" * 50)

    # Vérifier d'abord les prérequis
    env_status = check_environment()
    if not env_status['api_keys']:
        print("❌ Impossible de tester la connexion: Clés API manquantes")
        return False

    config_valid, config = check_configuration()
    if not config_valid or not config:
        print("❌ Impossible de tester la connexion: Configuration invalide")
        return False

    try:
        # Valider la configuration
        print("🔍 Validation de la configuration...")
        if not validate_exchange_config(config):
            print("❌ Échec de la validation de la configuration")
            return False
        print("✅ Configuration validée")

        # Créer le client d'exchange
        print("🔌 Création du client d'exchange...")
        exchange = get_exchange_client(config)
        print(f"✅ Client créé: {exchange.id}")

        # Test de base: heure du serveur
        print("⏰ Test de l'heure du serveur...")
        server_time = exchange.fetch_time()
        print(f"✅ Heure du serveur: {exchange.iso8601(server_time)}")

        # Test: chargement des marchés
        print("📊 Chargement des marchés...")
        markets = exchange.load_markets()
        print(f"✅ Marchés chargés: {len(markets)} paires")

        # Vérifier les paires importantes
        important_pairs = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'BNB/USDT', 'XRP/USDT']
        available_pairs = [pair for pair in important_pairs if pair in markets]
        print(f"✅ Paires ADAN disponibles: {', '.join(available_pairs)}")

        # Test: récupération du solde
        print("💰 Récupération du solde...")
        balance = exchange.fetch_balance()

        # Afficher les soldes non nuls
        non_zero_balances = {k: v for k, v in balance['total'].items() if v > 0}
        if non_zero_balances:
            print("✅ Soldes Testnet disponibles:")
            for currency, amount in non_zero_balances.items():
                print(f"   - {currency}: {amount}")
        else:
            print("ℹ️  Aucun solde affiché (normal sur testnet)")

        # Test: données de marché en temps réel
        print("📈 Test des données de marché...")
        ticker = exchange.fetch_ticker('BTC/USDT')
        print(f"✅ BTC/USDT - Prix: {ticker['last']}, Volume: {ticker['baseVolume']}")

        print("\n🎉 CONNEXION TESTNET RÉUSSIE !")
        return True

    except ExchangeConfigurationError as e:
        print(f"❌ Erreur de configuration: {e}")
        return False
    except ExchangeConnectionError as e:
        print(f"❌ Erreur de connexion: {e}")
        return False
    except ccxt.AuthenticationError as e:
        print(f"❌ Erreur d'authentification: {e}")
        print("   Vérifiez vos clés API Binance Testnet")
        return False
    except ccxt.NetworkError as e:
        print(f"❌ Erreur réseau: {e}")
        print("   Vérifiez votre connexion internet")
        return False
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}")
        traceback.print_exc()
        return False

def show_next_steps(success):
    """Affiche les prochaines étapes selon le résultat."""
    print("\n🚀 PROCHAINES ÉTAPES")
    print("-" * 50)

    if success:
        print("✅ Connexion Exchange opérationnelle !")
        print("\n📋 Actions recommandées:")
        print("1. 🧪 Tester les scripts:")
        print("   python test_ccxt_connection.py")
        print("   python test_exchange_connector.py")
        print("\n2. 🔧 Intégrer dans OrderManager:")
        print("   Modifier src/adan_trading_bot/environment/order_manager.py")
        print("\n3. 📝 Créer les scripts paper trading:")
        print("   scripts/paper_trading_agent.py")
        print("   scripts/live_order_test.py")
        print("\n4. 🎯 Tester des ordres réels (Testnet):")
        print("   Ordres market BUY/SELL avec gestion PnL")
    else:
        print("❌ Connexion Exchange NON opérationnelle")
        print("\n🔧 Actions requises:")
        print("1. ⚙️  Vérifier l'environnement:")
        print("   conda activate trading_env")
        print("\n2. 🔑 Définir les clés API:")
        print("   export BINANCE_TESTNET_API_KEY='VOTRE_CLE'")
        print("   export BINANCE_TESTNET_SECRET_KEY='VOTRE_SECRET'")
        print("\n3. 🌐 Vérifier la connexion internet")
        print("\n4. 📖 Consulter le guide:")
        print("   cat GUIDE_TEST_EXCHANGE_CONNECTOR.md")

def generate_summary():
    """Génère un résumé du statut."""
    print("\n📊 RÉSUMÉ DU STATUT")
    print("-" * 50)

    env_status = check_environment()
    config_valid, _ = check_configuration()
    connection_ok = False

    # Test de connexion simplifié
    if env_status['api_keys'] and config_valid:
        try:
            # Test rapide sans output détaillé
            config_path = project_root / "config" / "main_config.yaml"
            config = load_config(str(config_path))
            exchange = get_exchange_client(config)
            exchange.fetch_time()  # Test simple
            connection_ok = True
        except:
            connection_ok = False

    # Calcul du score
    checks = [
        env_status['conda_env'],
        env_status['ccxt_available'],
        env_status['api_keys'],
        env_status['config_file'],
        config_valid,
        connection_ok
    ]

    score = sum(checks)
    total = len(checks)
    percentage = (score / total) * 100

    print(f"Score global: {score}/{total} ({percentage:.1f}%)")

    if percentage >= 80:
        status_icon = "🟢"
        status_text = "OPÉRATIONNEL"
    elif percentage >= 60:
        status_icon = "🟡"
        status_text = "PARTIELLEMENT FONCTIONNEL"
    else:
        status_icon = "🔴"
        status_text = "NON OPÉRATIONNEL"

    print(f"Statut: {status_icon} {status_text}")

    return connection_ok

def main():
    """Fonction principale."""
    print_header()

    # Tests principaux
    env_status = check_environment()
    config_valid, config = check_configuration()
    connection_success = test_connection()

    # Résumé et prochaines étapes
    final_success = generate_summary()
    show_next_steps(final_success)

    # Footer
    print("\n" + "=" * 80)
    print(" 📞 Support: Consultez GUIDE_TEST_EXCHANGE_CONNECTOR.md")
    print(" 🔄 Mise à jour: Relancez ce script après modifications")
    print("=" * 80)

    return final_success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n❌ Statut interrompu par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Erreur fatale: {e}")
        traceback.print_exc()
        sys.exit(1)
