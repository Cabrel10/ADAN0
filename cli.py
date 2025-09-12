import os
import re
import sys
import logging
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt, Confirm
from rich.table import Table
from pyfiglet import Figlet # Keep Figlet for now, but will change how it's used

# Assurez-vous que le chemin vers src est dans PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from adan_trading_bot.trading.secure_api_manager import SecureAPIManager, ExchangeType, APICredentials
from adan_trading_bot.trading.manual_trading_interface import ManualTradingInterface

# Configuration du logger
# Supprimer les messages INFO du logger pour une interface plus propre
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

console = Console()
api_manager = SecureAPIManager()
trading_interface = ManualTradingInterface(api_manager)

def display_main_menu():
    # Generate ADAN text with Figlet
    f = Figlet(font='standard')
    adan_text_raw = f.renderText('ADAN')

    # Manually apply rainbow gradient
    rainbow_colors = [
        "#FF0000", "#FFA500", "#FFFF00", "#00FF00", "#0000FF", "#4B0082", "#EE82EE"
    ]
    lines = adan_text_raw.splitlines()
    for i, line in enumerate(lines):
        color_index = i % len(rainbow_colors)
        console.print(Text(line, justify="center", style=f"bold {rainbow_colors[color_index]}"))
    console.print(Panel(Text("ADAN Trading Bot - Menu Principal", justify="center", style="bold green"), style="blue"))

    menu_table = Table(show_header=False, box=None, padding=0)
    menu_table.add_column("Option", style="bold cyan", no_wrap=True)
    menu_table.add_column("Description", style="dim")

    menu_options = [
        ("1.", "Sélectionner l'Exchange", "Choisissez la plateforme de trading (Binance, Bybit, etc.)"),
        ("2.", "Gérer les Clés API", "Ajouter, lister ou supprimer vos clés API sécurisées"),
        ("3.", "Sélectionner les Paires de Trading", "Définissez les actifs sur lesquels vous souhaitez trader (ex: BTC/USDT)"),
        ("4.", "Définir les Paramètres de Trading", "Configurez les détails de vos ordres (type, stop-loss, take-profit)"),
        ("5.", "Définir le Modèle et le Capital", "Associez un modèle de trading et allouez un capital"),
        ("6.", "Afficher l'Historique des Actions", "Consultez les transactions passées et les performances"),
        ("7.", "Démarrer le Trading", "Lancez le bot de trading avec les configurations actuelles"),
        ("8.", "Quitter", "Fermez l'application", "bold red") # Special style for Quit
    ]

    for i, (num, title, desc, *style) in enumerate(menu_options):
        option_text = Text(f"{num} {title}", style=style[0] if style else "bold cyan")
        menu_table.add_row(option_text, desc)

    console.print(menu_table)

def select_exchange():
    console.print(Panel(Text("Sélection de l'Exchange", justify="center", style="bold green"), style="blue"))
    exchanges = [e.value for e in ExchangeType]

    exchange_table = Table(show_header=False, box=None, padding=0)
    exchange_table.add_column("Number", style="bold cyan", no_wrap=True)
    exchange_table.add_column("Exchange", style="bold")

    for i, exchange in enumerate(exchanges):
        if exchange.lower() in ["binance", "bitget"]:
            style = "bold green"
        else:
            style = "bold orange3" # Using orange3 for a distinct orange
        exchange_table.add_row(f"{i+1}.", Text(exchange.upper(), style=style))

    console.print(exchange_table)

    choice = Prompt.ask("Entrez le numéro de l'exchange", choices=[str(i+1) for i in range(len(exchanges))])
    selected_exchange = ExchangeType(exchanges[int(choice)-1])
    trading_interface.set_default_exchange(selected_exchange)
    console.print(f"Exchange par défaut défini sur [bold green]{selected_exchange.value.upper()}[/bold green].")
    Prompt.ask("Appuyez sur Entrée pour continuer...")

def manage_api_keys():
    console.print(Panel(Text("Gestion des Clés API", justify="center", style="bold green"), style="blue"))

    # Gérer le mot de passe maître
    master_password_set = False
    if api_manager.config_path.exists():
        # Le fichier de configuration existe, demander le mot de passe existant
        while not master_password_set:
            master_password = Prompt.ask("Entrez le mot de passe maître pour déverrouiller les clés API", password=True)
            if api_manager.set_master_password(master_password):
                console.print("[bold green]Clés API déverrouillées.[/bold green]")
                master_password_set = True
            else:
                console.print("[bold red]Mot de passe maître incorrect ou erreur lors du déverrouillage. Veuillez réessayer.[/bold red]")
                if not Confirm.ask("Voulez-vous réessayer ?", default=True):
                    Prompt.ask("Appuyez sur Entrée pour continuer...")
                    return
    else:
        # Le fichier de configuration n'existe pas, créer un nouveau mot de passe
        console.print("[bold yellow]Aucun mot de passe maître n'est défini. Vous devez en créer un.[/bold yellow]")
        while not master_password_set:
            new_master_password = Prompt.ask("Créez un nouveau mot de passe maître", password=True)
            confirm_password = Prompt.ask("Confirmez le nouveau mot de passe maître", password=True)

            if new_master_password == confirm_password:
                if api_manager.set_master_password(new_master_password):
                    console.print("[bold green]Nouveau mot de passe maître créé avec succès.[/bold green]")
                    master_password_set = True
                else:
                    console.print("[bold red]Erreur critique lors de la création du mot de passe maître. Veuillez contacter le support.[/bold red]")
                    Prompt.ask("Appuyez sur Entrée pour continuer...")
                    return
            else:
                console.print("[bold red]Les mots de passe ne correspondent pas. Veuillez réessayer.[/bold red]")
                if not Confirm.ask("Voulez-vous réessayer ?", default=True):
                    Prompt.ask("Appuyez sur Entrée pour continuer...")
                    return

    while True:
        key_menu_table = Table(show_header=False, box=None, padding=0)
        key_menu_table.add_column("Option", style="bold cyan", no_wrap=True)
        key_menu_table.add_column("Description", style="dim")

        key_options = [
            ("1.", "Ajouter/Mettre à jour une clé API", ""),
            ("2.", "Lister les clés API", ""),
            ("3.", "Supprimer une clé API", ""),
            ("4.", "Retour au menu principal", "", "bold red")
        ]

        for i, (num, title, desc, *style) in enumerate(key_options):
            option_text = Text(f"{num} {title}", style=style[0] if style else "bold cyan")
            key_menu_table.add_row(option_text, desc)

        console.print(key_menu_table)

        key_choice = Prompt.ask("Votre choix", choices=["1", "2", "3", "4"])

        if key_choice == "1":
            exchange_name = Prompt.ask("Nom de l'exchange (binance, bybit, etc.)").lower()
            try:
                exchange_type = ExchangeType(exchange_name)
            except ValueError:
                console.print("[bold red]Exchange non supporté.[/bold red]")
                continue

            api_key = Prompt.ask("Entrez la clé API")
            api_secret = Prompt.ask("Entrez le secret API", password=True)
            is_sandbox = Confirm.ask("Utiliser le mode Sandbox (Testnet) ?", default=True)

            creds = APICredentials(exchange=exchange_type, api_key=api_key, api_secret=api_secret, sandbox=is_sandbox)
            if api_manager.add_credentials(creds):
                console.print("[bold green]Clés API ajoutées/mises à jour avec succès.[/bold green]")
            else:
                console.print("[bold red]Échec de l'ajout/mise à jour des clés API.[/bold red]")

        elif key_choice == "2":
            console.print(Panel(Text("Clés API configurées", justify="center", style="bold green"), style="blue"))
            creds_list = api_manager.list_credentials()
            if not creds_list:
                console.print("Aucune clé API configurée.")
            else:
                creds_table = Table(show_header=True, header_style="bold magenta", box=None)
                creds_table.add_column("Exchange")
                creds_table.add_column("Name")
                creds_table.add_column("Sandbox")
                creds_table.add_column("API Key (Partial)")

                for cred in creds_list:
                    creds_table.add_row(
                        cred['exchange'],
                        cred['name'],
                        str(cred['sandbox']),
                        f"{cred['api_key'][:4]}...{cred['api_key'][-4:]}" # Show partial key for security
                    )
                console.print(creds_table)

        elif key_choice == "3":
            exchange_name = Prompt.ask("Nom de l'exchange à supprimer (binance, bybit, etc.)").lower()
            try:
                exchange_type = ExchangeType(exchange_name)
            except ValueError:
                console.print("[bold red]Exchange non supporté.[/bold red]")
                continue

            if api_manager.remove_credentials(exchange_type):
                console.print("[bold green]Clés API supprimées avec succès.[/bold green]")
            else:
                console.print("[bold red]Échec de la suppression des clés API.[/bold red]")

        elif key_choice == "4":
            break
    Prompt.ask("Appuyez sur Entrée pour continuer...")

import re

def select_trading_pairs():
    console.print(Panel(Text("Sélection des Paires de Trading", justify="center", style="bold green"), style="blue"))

    while True:
        current_pairs = trading_interface.get_trading_pairs()
        if current_pairs:
            console.print(f"[bold yellow]Paires de trading actuelles : {', '.join(current_pairs)}[/bold yellow]")
        else:
            console.print("[bold yellow]Aucune paire de trading n'est actuellement sélectionnée.[/bold yellow]")

        pair_menu_table = Table(show_header=False, box=None, padding=0)
        pair_menu_table.add_column("Option", style="bold cyan", no_wrap=True)
        pair_menu_table.add_column("Description", style="dim")

        pair_options = [
            ("1.", "Ajouter/Remplacer des paires", "Saisir de nouvelles paires (remplace les existantes)"),
            ("2.", "Ajouter plus de paires", "Ajouter des paires à la liste actuelle"),
            ("3.", "Supprimer une paire", "Supprimer une paire spécifique de la liste"),
            ("4.", "Effacer toutes les paires", "Vider la liste des paires de trading"),
            ("5.", "Retour au menu principal", "", "bold red")
        ]

        for i, (num, title, desc, *style) in enumerate(pair_options):
            option_text = Text(f"{num} {title}", style=style[0] if style else "bold cyan")
            pair_menu_table.add_row(option_text, desc)

        console.print(pair_menu_table)

        pair_choice = Prompt.ask("Votre choix", choices=[str(i) for i in range(1, 6)])

        if pair_choice == "1":
            # Ajouter/Remplacer des paires
            pairs_input = Prompt.ask("Entrez les paires de trading (ex: BTC/USDT, ETH/BTC). Séparées par des virgules")
            raw_pairs = [p.strip() for p in pairs_input.split(',') if p.strip()]

            valid_pairs = []
            invalid_pairs = []
            pair_pattern = re.compile(r"^[A-Z]{2,5}/[A-Z]{2,5}$")

            for pair in raw_pairs:
                if pair_pattern.match(pair.upper()):
                    valid_pairs.append(pair.upper())
                else:
                    invalid_pairs.append(pair)

            if invalid_pairs:
                console.print(f"[bold red]Paires invalides ignorées : {', '.join(invalid_pairs)}[/bold red]")

            if valid_pairs:
                trading_interface.set_trading_pairs(valid_pairs)
                console.print(f"[bold green]Paires de trading sélectionnées : {', '.join(trading_interface.get_trading_pairs())}[/bold green]")
            else:
                console.print("[bold red]Aucune paire valide n'a été entrée. La liste des paires n'a pas été modifiée.[/bold red]")

        elif pair_choice == "2":
            # Ajouter plus de paires
            pairs_input = Prompt.ask("Entrez les paires de trading à ajouter (ex: LTC/USDT, ADA/BTC). Séparées par des virgules")
            raw_pairs = [p.strip() for p in pairs_input.split(',') if p.strip()]

            valid_new_pairs = []
            invalid_new_pairs = []
            pair_pattern = re.compile(r"^[A-Z]{2,5}/[A-Z]{2,5}$")

            for pair in raw_pairs:
                if pair_pattern.match(pair.upper()):
                    if pair.upper() not in current_pairs:
                        valid_new_pairs.append(pair.upper())
                    else:
                        console.print(f"[bold yellow]La paire {pair.upper()} existe déjà et ne sera pas ajoutée.[/bold yellow]")
                else:
                    invalid_new_pairs.append(pair)

            if invalid_new_pairs:
                console.print(f"[bold red]Paires invalides ignorées : {', '.join(invalid_new_pairs)}[/bold red]")

            if valid_new_pairs:
                trading_interface.set_trading_pairs(current_pairs + valid_new_pairs)
                console.print(f"[bold green]Paires de trading mises à jour : {', '.join(trading_interface.get_trading_pairs())}[/bold green]")
            else:
                console.print("[bold red]Aucune nouvelle paire valide n'a été entrée ou toutes les paires existent déjà.[/bold red]")

        elif pair_choice == "3":
            # Supprimer une paire
            if not current_pairs:
                console.print("[bold red]Aucune paire à supprimer.[/bold red]")
                Prompt.ask("Appuyez sur Entrée pour continuer...")
                continue

            pair_to_remove = Prompt.ask("Entrez la paire à supprimer (ex: BTC/USDT)").upper()
            if pair_to_remove in current_pairs:
                updated_pairs = [p for p in current_pairs if p != pair_to_remove]
                trading_interface.set_trading_pairs(updated_pairs)
                console.print(f"[bold green]Paire {pair_to_remove} supprimée avec succès.[/bold green]")
                if updated_pairs:
                    console.print(f"[bold green]Nouvelles paires : {', '.join(updated_pairs)}[/bold green]")
                else:
                    console.print("[bold green]Toutes les paires ont été supprimées.[/bold green]")
            else:
                console.print(f"[bold red]La paire {pair_to_remove} n'est pas dans la liste actuelle.[/bold red]")

        elif pair_choice == "4":
            # Effacer toutes les paires
            if Confirm.ask("Êtes-vous sûr de vouloir effacer toutes les paires de trading ?", default=False):
                trading_interface.set_trading_pairs([])
                console.print("[bold green]Toutes les paires de trading ont été effacées.[/bold green]")
            else:
                console.print("[bold yellow]Opération annulée.[/bold yellow]")

        elif pair_choice == "5":
            break

    Prompt.ask("Appuyez sur Entrée pour continuer...")

def define_trading_parameters():
    console.print(Panel(Text("Définir les Paramètres de Trading", justify="center", style="bold green"), style="blue"))

    while True:
        console.print(f"\n[bold yellow]Paramètres de trading actuels:[/bold yellow]")
        console.print(f"  - Stop Loss (%): [cyan]{trading_interface.stop_loss_pct * 100:.2f}%[/cyan]")
        console.print(f"  - Take Profit (%): [cyan]{trading_interface.take_profit_pct * 100:.2f}%[/cyan]")
        console.print(f"  - Trailing Stop (%): [cyan]{trading_interface.trailing_stop * 100:.2f}%[/cyan]")

        param_menu_table = Table(show_header=False, box=None, padding=0)
        param_menu_table.add_column("Option", style="bold cyan", no_wrap=True)
        param_menu_table.add_column("Description", style="dim")

        param_options = [
            ("1.", "Modifier le Stop Loss", ""),
            ("2.", "Modifier le Take Profit", ""),
            ("3.", "Modifier le Trailing Stop", ""),
            ("4.", "Retour au menu principal", "", "bold red")
        ]

        for i, (num, title, desc, *style) in enumerate(param_options):
            option_text = Text(f"{num} {title}", style=style[0] if style else "bold cyan")
            param_menu_table.add_row(option_text, desc)

        console.print(param_menu_table)

        param_choice = Prompt.ask("Votre choix", choices=[str(i) for i in range(1, 5)])

        if param_choice == "1":
            try:
                new_sl = Prompt.ask("Entrez la nouvelle valeur de Stop Loss (en pourcentage, ex: 5 pour 5%)", default=str(trading_interface.stop_loss_pct * 100))
                new_sl = float(new_sl) / 100.0
                if 0 <= new_sl <= 1:
                    trading_interface.set_stop_loss_pct(new_sl)
                    console.print(f"[bold green]Stop Loss défini à {new_sl * 100:.2f}%[/bold green]")
                else:
                    console.print("[bold red]La valeur du Stop Loss doit être entre 0% et 100%.[/bold red]")
            except ValueError:
                console.print("[bold red]Valeur invalide. Veuillez entrer un nombre.[/bold red]")
        elif param_choice == "2":
            try:
                new_tp = Prompt.ask("Entrez la nouvelle valeur de Take Profit (en pourcentage, ex: 15 pour 15%)", default=str(trading_interface.take_profit_pct * 100))
                new_tp = float(new_tp) / 100.0
                if 0 <= new_tp <= 1:
                    trading_interface.set_take_profit_pct(new_tp)
                    console.print(f"[bold green]Take Profit défini à {new_tp * 100:.2f}%[/bold green]")
                else:
                    console.print("[bold red]La valeur du Take Profit doit être entre 0% et 100%.[/bold red]")
            except ValueError:
                console.print("[bold red]Valeur invalide. Veuillez entrer un nombre.[/bold red]")
        elif param_choice == "3":
            try:
                new_ts = Prompt.ask("Entrez la nouvelle valeur de Trailing Stop (en pourcentage, ex: 1 pour 1%)", default=str(trading_interface.trailing_stop * 100))
                new_ts = float(new_ts) / 100.0
                if 0 <= new_ts <= 1:
                    trading_interface.set_trailing_stop(new_ts)
                    console.print(f"[bold green]Trailing Stop défini à {new_ts * 100:.2f}%[/bold green]")
                else:
                    console.print("[bold red]La valeur du Trailing Stop doit être entre 0% et 100%.[/bold red]")
            except ValueError:
                console.print("[bold red]Valeur invalide. Veuillez entrer un nombre.[/bold red]")
        elif param_choice == "4":
            break
    Prompt.ask("Appuyez sur Entrée pour continuer...")

def define_model_and_capital():
    console.print(Panel(Text("Définir le Modèle et le Capital", justify="center", style="bold green"), style="blue"))

    while True:
        current_model_algo = trading_interface.config.get('agent', {}).get('algorithm', 'Non défini')
        current_model_policy = trading_interface.config.get('agent', {}).get('policy', 'Non définie')
        current_initial_balance = trading_interface.config.get('portfolio', {}).get('initial_balance', 'Non défini')
        current_strategy_weights = trading_interface.get_strategy_weights()

        console.print(f"\n[bold yellow]Configuration actuelle:[/bold yellow]")
        console.print(f"  - Algorithme du Modèle: [cyan]{current_model_algo}[/cyan]")
        console.print(f"  - Politique du Modèle: [cyan]{current_model_policy}[/cyan]")
        console.print(f"  - Capital Initial: [cyan]{current_initial_balance} USDT[/cyan]")
        console.print(f"  - Poids des Stratégies: [cyan]{current_strategy_weights}[/cyan]")

        model_capital_menu_table = Table(show_header=False, box=None, padding=0)
        model_capital_menu_table.add_column("Option", style="bold cyan", no_wrap=True)
        model_capital_menu_table.add_column("Description", style="dim")

        model_capital_options = [
            ("1.", "Définir le Capital Initial", ""),
            ("2.", "Définir le Modèle (Algorithme et Politique)", ""),
            ("3.", "Définir les Poids des Stratégies", ""),
            ("4.", "Retour au menu principal", "", "bold red")
        ]

        for i, (num, title, desc, *style) in enumerate(model_capital_options):
            option_text = Text(f"{num} {title}", style=style[0] if style else "bold cyan")
            model_capital_menu_table.add_row(option_text, desc)

        console.print(model_capital_menu_table)

        model_capital_choice = Prompt.ask("Votre choix", choices=[str(i) for i in range(1, 5)])

        if model_capital_choice == "1":
            try:
                new_balance = Prompt.ask("Entrez le nouveau capital initial (en USDT)", default=str(current_initial_balance))
                new_balance = float(new_balance)
                if new_balance > 0:
                    trading_interface.set_initial_balance(new_balance)
                    console.print(f"[bold green]Capital initial défini à {new_balance} USDT[/bold green]")
                else:
                    console.print("[bold red]Le capital initial doit être un nombre positif.[/bold red]")
            except ValueError:
                console.print("[bold red]Valeur invalide. Veuillez entrer un nombre.[/bold red]")
        elif model_capital_choice == "2":
            new_algo = Prompt.ask("Entrez le nom de l'algorithme du modèle (ex: PPO)", default=current_model_algo)
            new_policy = Prompt.ask("Entrez le nom de la politique du modèle (ex: MultiInputPolicy)", default=current_model_policy)
            trading_interface.set_model_config(new_algo, new_policy)
            console.print(f"[bold green]Modèle défini sur Algorithme: {new_algo}, Politique: {new_policy}[/bold green]")
        elif model_capital_choice == "3":
            console.print(Panel(Text("Définir les Poids des Stratégies", justify="center", style="bold green"), style="blue"))
            current_weights = trading_interface.config.get('agent', {}).get('strategy_weights', {'w1': 0, 'w2': 0, 'w3': 0, 'w4': 0})
            weights = {}
            valid_input = False
            while not valid_input:
                try:
                    console.print("\n[bold yellow]Entrez les poids pour chaque stratégie (la somme doit être 10):[/bold yellow]")
                    w1 = float(Prompt.ask(f"Poids pour w1 (actuel: {current_weights.get('w1')}):", default=str(current_weights.get('w1'))))
                    w2 = float(Prompt.ask(f"Poids pour w2 (actuel: {current_weights.get('w2')}):", default=str(current_weights.get('w2'))))
                    w3 = float(Prompt.ask(f"Poids pour w3 (actuel: {current_weights.get('w3')}):", default=str(current_weights.get('w3'))))
                    w4 = float(Prompt.ask(f"Poids pour w4 (actuel: {current_weights.get('w4')}):", default=str(current_weights.get('w4'))))

                    total_weight = w1 + w2 + w3 + w4
                    if abs(total_weight - 10.0) < 0.001: # Allow for floating point inaccuracies
                        weights = {'w1': w1, 'w2': w2, 'w3': w3, 'w4': w4}
                        trading_interface.set_strategy_weights(weights)
                        console.print(f"[bold green]Poids des stratégies définis: {weights}[/bold green]")
                        valid_input = True
                    else:
                        console.print(f"[bold red]La somme des poids doit être 10. Actuel: {total_weight}[/bold red]")
                except ValueError:
                    console.print("[bold red]Valeur invalide. Veuillez entrer un nombre pour chaque poids.[/bold red]")
        elif model_capital_choice == "4":
            break
    Prompt.ask("Appuyez sur Entrée pour continuer...")

def display_action_history():
    console.print(Panel(Text("Historique des Actions du Modèle", justify="center", style="bold green"), style="blue"))

    order_history = trading_interface.get_order_history()

    if not order_history:
        console.print("[bold yellow]Aucune action enregistrée pour le moment.[/bold yellow]")
    else:
        history_table = Table(show_header=True, header_style="bold magenta", box=None)
        history_table.add_column("ID Ordre")
        history_table.add_column("Symbole")
        history_table.add_column("Côté")
        history_table.add_column("Type")
        history_table.add_column("Quantité")
        history_table.add_column("Prix")
        history_table.add_column("Statut")
        history_table.add_column("Créé le")

        for order in order_history:
            history_table.add_row(
                order.order_id[:8] + "...",
                order.symbol,
                order.side.value,
                order.order_type.value,
                f"{order.quantity:.4f}",
                f"{order.price:.4f}" if order.price else "N/A",
                order.status.value,
                order.created_at.strftime("%Y-%m-%d %H:%M:%S")
            )
        console.print(history_table)

    Prompt.ask("Appuyez sur Entrée pour continuer...")

def start_trading():
    console.print(Panel(Text("Démarrage du Trading", justify="center", style="bold green"), style="blue"))

    console.print("[bold green]Démarrage du bot de trading avec les configurations actuelles...[/bold green]")

    # Afficher un résumé des configurations actuelles avant de démarrer
    console.print("\n[bold yellow]Résumé des configurations:[/bold yellow]")
    console.print(f"  - Exchange par défaut: [cyan]{trading_interface.default_exchange.value.upper()}[/cyan]")
    console.print(f"  - Paires de trading: [cyan]{', '.join(trading_interface.get_trading_pairs()) if trading_interface.get_trading_pairs() else 'Non définies'}[/cyan]")
    console.print(f"  - Stop Loss: [cyan]{trading_interface.stop_loss_pct * 100:.2f}%[/cyan]")
    console.print(f"  - Take Profit: [cyan]{trading_interface.take_profit_pct * 100:.2f}%[/cyan]")
    console.print(f"  - Trailing Stop: [cyan]{trading_interface.trailing_stop * 100:.2f}%[/cyan]")
    console.print(f"  - Capital Initial: [cyan]{trading_interface.config.get('portfolio', {}).get('initial_balance', 'Non défini')} USDT[/cyan]")
    console.print(f"  - Algorithme du Modèle: [cyan]{trading_interface.config.get('agent', {}).get('algorithm', 'Non défini')}[/cyan]")
    console.print(f"  - Politique du Modèle: [cyan]{trading_interface.config.get('agent', {}).get('policy', 'Non définie')}[/cyan]")
    console.print(f"  - Poids des Stratégies: [cyan]{trading_interface.get_strategy_weights()}[/cyan]")

    if Confirm.ask("Confirmez-vous le démarrage du trading avec ces paramètres ?", default=True):
        trading_interface.start_live_trading()
        console.print("[bold green]Bot de trading démarré. Appuyez sur Entrée pour revenir au menu principal.[/bold green]")
    else:
        console.print("[bold yellow]Démarrage du trading annulé.[/bold yellow]")

    Prompt.ask("Appuyez sur Entrée pour continuer...")

def main():
    while True:
        display_main_menu()
        choice = Prompt.ask("Votre choix", choices=[str(i) for i in range(1, 9)])

        if choice == "1":
            select_exchange()
        elif choice == "2":
            manage_api_keys()
        elif choice == "3":
            select_trading_pairs()
        elif choice == "4":
            define_trading_parameters()
        elif choice == "5":
            define_model_and_capital()
        elif choice == "6":
            display_action_history()
        elif choice == "7":
            start_trading()
        elif choice == "8":
            console.print("[bold yellow]Arrêt du bot de trading. Au revoir ![/bold yellow]")
            break

if __name__ == "__main__":
    main()
