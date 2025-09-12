# %%
# Titre, auteur, date
# Jupytext formats: py:percent,ipynb

# %%
# Imports standards
import os
import yaml
import logging
from pathlib import Path

# Imports ADAN
from adan_trading_bot.training.training_orchestrator import TrainingOrchestrator
from adan_trading_bot.common.config_watcher import ConfigWatcher
# Autres imports pour l’analyse
import matplotlib.pyplot as plt
import pandas as pd

# Configuration du logger (pour s'assurer que les messages DEBUG sont visibles)
# Ajouter le répertoire src au path si ce script est exécuté seul
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from adan_trading_bot.common.utils import get_logger
logger = get_logger(__name__)
logger.setLevel(logging.DEBUG) # S'assurer que les messages DEBUG sont traités

logger.info("Notebook d'analyse initialisé.")

# %%
# Chargement des Configurations
# Définitions des chemins
project_root = Path(__file__).parent.parent
config_dir    = project_root / "config"

# Chargement de la config principale
with open(config_dir / "main_config.yaml") as f:
    main_cfg = yaml.safe_load(f)
# Affichage sommaire
print("Main config keys:", list(main_cfg.keys()))

# %%
# Initialisation de l’Orchestrator et du ConfigWatcher
# Instanciation
orch = TrainingOrchestrator(device="cpu", config_dir=str(config_dir))

# Vérifier que le ConfigWatcher est actif
print("ConfigWatcher:", orch.config_watcher)

# %%
# Préparation d’un Entraînement Court de Test Dynamique
# Définir des paramètres très courts
test_kwargs = {
    "total_timesteps":   2000,
    "batch_size":        32,
    "n_steps":           64,
    "callback":          None,  # sera à remplacer par un LoggingCallback plus tard
}

# %%
# Exécuter un entraînement “brut” sans callback pour collecter des logs de base
# REMARQUE : commentez la ligne suivante si vous passez au callback.
# orch.train_agent(**test_kwargs)

# %%
# Visualisation des Courbes d’Entraînement
# Charger les logs TensorBoard ou CSV exporté
# Par exemple : reports/metrics/train_metrics.csv
df = pd.read_csv(project_root / "reports" / "metrics" / "training_metrics.csv")

# %%
# Tracer la récompense moyenne vs timesteps
plt.figure()
plt.plot(df["timesteps"], df["ep_rew_mean"])
plt.title("Récompense moyenne par timestep")
plt.xlabel("Timestep")
plt.ylabel("Récompense")
plt.show()

# %%
# Test de Reload Dynamique avec LoggingCallback
# Importer et configurer votre LoggingCallback
from scripts.test_orchestrator_config_reload import LoggingCallback

cb = LoggingCallback(orch.config_watcher)

# %%
# Relancer un entraînement court AVEC callback
test_kwargs["callback"] = cb
orch.train_agent(**test_kwargs)

# %%
# Analyse des Logs de Callback
# Extraire du fichier log (logs/adan.log ou custom) les lignes “[Callback]”
log_path = project_root / "logs" / "adan.log"
lines = [l for l in open(log_path) if "[Callback]" in l]

# %%
# Afficher les 20 premières lignes
for l in lines[:20]:
    print(l.strip())

# %%
# Endurance Test & Monitoring Mémoire
# Pour un endurance test, modifier :
long_kwargs = test_kwargs.copy()
long_kwargs.update({"total_timesteps": 200_000, "n_steps": 1024})

# %%
# Lancer l’endurance test
# orch.train_agent(**long_kwargs)

# %%
# Charger les logs mémoire : logs/memory/memory.log
mem_df = pd.read_json(project_root / "logs" / "memory" / "memory.log", lines=True)

# %%
# Tracer la consommation mémoire vs timestamp
plt.figure()
plt.plot(pd.to_datetime(mem_df["asctime"]), mem_df["rss_mb"])
plt.title("Consommation RAM au fil du temps")
plt.xlabel("Heure")
plt.ylabel("RAM (MB)")
plt.show()

# %%
# Backtest “Paper Live” avec Perturbations
# Charger un backtest en mode paper
main_cfg["environment"]["mode"] = "paper"
with open(config_dir / "environment_config.yaml", "w") as f:
    yaml.dump(main_cfg["environment"], f)

# %%
# Lancer un backtest (script dédié)
# !python scripts/paper_trade_agent.py --config environment_config.yaml

# %%
# Conclusion & Export des Résultats
# Récapitulatif final
print("✅ Analyse interactive terminée.")
print("Résultats sauvegardés dans reports/figures et logs/")
