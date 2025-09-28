# INSTRUCTIONS D'UTILISATION - SYSTÈME ADAN CORRIGÉ
## Guide Complet pour Entraînement, Monitoring et Reprise

**Version :** Post-corrections TensorBoard & Dashboard  
**Date :** 28 septembre 2025  
**Statut :** ✅ SYSTÈME ENTIÈREMENT OPÉRATIONNEL

---

## 🚀 DÉMARRAGE RAPIDE

### 1. Entraînement Standard (Recommandé)
```bash
cd /home/morningstar/Documents/trading
timeout 30s /home/morningstar/miniconda3/envs/trading_env/bin/python \
  bot/scripts/train_parallel_agents.py \
  --config bot/config/config.yaml \
  --checkpoint-dir bot/checkpoints
```

### 2. Monitoring en Temps Réel
```bash
# Terminal 2 - Lancer le dashboard
cd /home/morningstar/Documents/trading/bot/scripts
/home/morningstar/miniconda3/envs/trading_env/bin/python training_dashboard.py

# Accès Web : http://localhost:8050
```

### 3. Reprise d'Entraînement  
```bash
# Reprendre depuis le dernier checkpoint
timeout 30s /home/morningstar/miniconda3/envs/trading_env/bin/python \
  bot/scripts/train_parallel_agents.py \
  --config bot/config/config.yaml \
  --checkpoint-dir bot/checkpoints \
  --resume
```

---

## 📖 GUIDE DÉTAILLÉ

### Configuration de l'Environnement

**1. Activation de l'environnement conda :**
```bash
conda activate trading_env
# Ou directement :
/home/morningstar/miniconda3/envs/trading_env/bin/python
```

**2. Vérification des répertoires :**
```bash
cd /home/morningstar/Documents/trading

# Vérifier la structure
ls -la bot/config/config.yaml     # Configuration principale
ls -la bot/checkpoints/           # Répertoire des sauvegardes  
ls -la reports/tensorboard_logs/  # Logs TensorBoard
```

### Commandes d'Entraînement

**Nouveau projet :**
```bash
# Nettoyer les anciens fichiers (optionnel)
rm -rf bot/checkpoints/* reports/tensorboard_logs/*

# Lancer un nouvel entraînement
timeout 3600s /home/morningstar/miniconda3/envs/trading_env/bin/python \
  bot/scripts/train_parallel_agents.py \
  --config bot/config/config.yaml \
  --checkpoint-dir bot/checkpoints
```

**Entraînement longue durée :**
```bash
# Pour entraînement de plusieurs heures
nohup timeout 14400s /home/morningstar/miniconda3/envs/trading_env/bin/python \
  bot/scripts/train_parallel_agents.py \
  --config bot/config/config.yaml \
  --checkpoint-dir bot/checkpoints \
  > training_$(date +%Y%m%d_%H%M%S).log 2>&1 &

echo $! > training.pid  # Sauvegarder le PID
```

**Reprise intelligente :**
```bash
# Le système détecte automatiquement le dernier checkpoint
timeout 7200s /home/morningstar/miniconda3/envs/trading_env/bin/python \
  bot/scripts/train_parallel_agents.py \
  --config bot/config/config.yaml \
  --checkpoint-dir bot/checkpoints \
  --resume
```

---

## 📊 MONITORING & DASHBOARD

### Dashboard Web (Recommandé)

**1. Lancement :**
```bash
cd bot/scripts
/home/morningstar/miniconda3/envs/trading_env/bin/python training_dashboard.py
```

**2. Accès :**
- **URL :** http://localhost:8050  
- **Interface :** Dashboard interactif temps réel
- **Données :** TensorBoard + métriques personnalisées

**3. Fonctionnalités disponibles :**
- 📈 Graphiques de performance en temps réel
- 📊 Comparaison multi-workers  
- 📋 Métriques détaillées portfolio
- 💾 Export des données

### TensorBoard Classique

**Lancement TensorBoard :**
```bash
# Si TensorBoard fonctionne dans votre environnement
tensorboard --logdir reports/tensorboard_logs --port 6006

# Accès : http://localhost:6006
```

**Note :** En cas d'erreur TensorBoard, utilisez le dashboard personnalisé qui lit les mêmes données.

### Vérification des Données

**Contrôle des fichiers générés :**
```bash
# Vérifier les logs TensorBoard
ls -la reports/tensorboard_logs/
# Devrait contenir : events.out.tfevents.* et progress.csv

# Vérifier les checkpoints
ls -la bot/checkpoints/
# Devrait contenir : checkpoint_YYYYMMDD_HHMMSS_ep*_step*/
```

**Test de lecture TensorBoard :**
```bash
/home/morningstar/miniconda3/envs/trading_env/bin/python -c "
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
tb_files = glob.glob('reports/tensorboard_logs/events.out.tfevents.*')
if tb_files:
    acc = EventAccumulator(tb_files[0])
    acc.Reload()
    print(f'Tags disponibles: {acc.Tags()[\"scalars\"]}')
    print('✅ TensorBoard lisible')
else:
    print('❌ Aucun fichier TensorBoard')
"
```

---

## 💾 GESTION DES CHECKPOINTS

### Sauvegarde Automatique

**Configuration actuelle :**
- **Fréquence :** Tous les 10,000 steps  
- **Maximum :** 5 checkpoints conservés
- **Sauvegarde finale :** Toujours à l'arrêt

**Structure des checkpoints :**
```
bot/checkpoints/checkpoint_YYYYMMDD_HHMMSS_epXXXXXX_stepXXXXXXXXXX/
├── metadata.json          # Métadonnées complètes
├── optimizer.pt           # État de l'optimiseur  
└── [autres fichiers modèle]
```

### Utilisation des Checkpoints

**Lister les checkpoints disponibles :**
```bash
ls -lat bot/checkpoints/ | grep checkpoint_
```

**Reprendre automatiquement :**
```bash
# Reprend depuis le plus récent automatiquement
python bot/scripts/train_parallel_agents.py --resume [autres options]
```

**Informations d'un checkpoint :**
```bash
# Voir les métadonnées
cat bot/checkpoints/checkpoint_*/metadata.json | jq .
```

---

## 🔧 CONFIGURATION AVANCÉE

### Personnalisation config.yaml

**Paramètres d'entraînement :**
```yaml
training:
  total_timesteps: 1000000        # Nombre total d'étapes
  checkpointing:
    enabled: true
    save_freq: 10000             # Fréquence sauvegarde
    save_path: ${paths.trained_models_dir}
```

**Paramètres de monitoring :**
```yaml
agent:
  checkpoint_freq: 10000          # Fréquence checkpoints
  logging_level: ERROR            # Niveau de logs console
```

### Variables d'Environnement

**Optimisation performance :**
```bash
export CUDA_VISIBLE_DEVICES=0    # GPU spécifique
export OMP_NUM_THREADS=4         # Threads CPU
export MKL_NUM_THREADS=4         # Intel MKL threads
```

---

## 🐛 DÉPANNAGE

### Problèmes Courants

**1. "Aucun fichier TensorBoard créé"**
```bash
# Vérifier les permissions
ls -la reports/
mkdir -p reports/tensorboard_logs
chmod 755 reports/tensorboard_logs
```

**2. "Dashboard affiche écran noir"**
```bash
# Vérifier que l'entraînement a généré des données
ls -la reports/tensorboard_logs/events.out.tfevents.*

# Relancer le dashboard
cd bot/scripts
python training_dashboard.py
```

**3. "Checkpoint non trouvé pour resume"**
```bash
# Vérifier les checkpoints existants  
ls -la bot/checkpoints/checkpoint_*/

# Lancer sans --resume pour nouveau démarrage
python bot/scripts/train_parallel_agents.py --config bot/config/config.yaml
```

**4. "Erreur de mémoire"**
```bash
# Réduire le nombre de workers
python bot/scripts/train_parallel_agents.py --workers 2 [autres options]
```

### Logs de Débogage

**Vérifier les logs d'entraînement :**
```bash
# Si lancé avec nohup
tail -f training_*.log

# En temps réel
python bot/scripts/train_parallel_agents.py [options] 2>&1 | tee debug.log
```

**Tester les composants individuellement :**
```bash
# Test rapide de validation
python test_tensorboard_checkpoint_validation.py
```

---

## 📋 BONNES PRATIQUES

### Workflow Recommandé

**1. Préparation :**
```bash
# Nettoyer si nécessaire
rm -rf bot/checkpoints/* reports/tensorboard_logs/*

# Vérifier la configuration  
cat bot/config/config.yaml | grep -A5 training
```

**2. Lancement :**
```bash
# Terminal 1 : Entraînement
timeout 7200s python bot/scripts/train_parallel_agents.py \
  --config bot/config/config.yaml --checkpoint-dir bot/checkpoints

# Terminal 2 : Monitoring
cd bot/scripts && python training_dashboard.py
```

**3. Surveillance :**
- Dashboard : http://localhost:8050
- Vérifier checkpoints toutes les heures
- Surveiller l'utilisation disque

**4. Reprise après interruption :**
```bash
# Reprendre automatiquement
python bot/scripts/train_parallel_agents.py \
  --config bot/config/config.yaml --checkpoint-dir bot/checkpoints --resume
```

### Maintenance

**Nettoyage périodique :**
```bash
# Archiver anciens logs (garder les 10 derniers)
cd reports/tensorboard_logs
ls -t events.out.tfevents.* | tail -n +11 | xargs rm -f

# Nettoyer anciens checkpoints (garder les 3 derniers)
cd bot/checkpoints  
ls -t -d checkpoint_* | tail -n +4 | xargs rm -rf
```

**Sauvegarde critique :**
```bash
# Sauvegarder les meilleurs checkpoints
cp -r bot/checkpoints/checkpoint_BEST_* ~/backup/
```

---

## ✅ VALIDATION DU SYSTÈME

### Test Complet Automatisé
```bash
# Valider toutes les fonctionnalités
python test_tensorboard_checkpoint_validation.py

# Doit afficher : "🎉 SUCCÈS: Les corrections principales fonctionnent!"
```

### Vérification Manuelle Rapide
```bash
# 1. Lancer entraînement court
timeout 60s python bot/scripts/train_parallel_agents.py \
  --config bot/config/config.yaml --checkpoint-dir bot/checkpoints

# 2. Vérifier fichiers générés
ls reports/tensorboard_logs/events.out.tfevents.* && echo "✅ TensorBoard OK"
ls bot/checkpoints/checkpoint_* && echo "✅ Checkpoints OK"

# 3. Tester dashboard
cd bot/scripts && timeout 10s python training_dashboard.py && echo "✅ Dashboard OK"
```

---

## 🎯 RÉSUMÉ DES COMMANDES ESSENTIELLES

```bash
# ENTRAÎNEMENT STANDARD
timeout 30s /home/morningstar/miniconda3/envs/trading_env/bin/python \
  bot/scripts/train_parallel_agents.py --config bot/config/config.yaml \
  --checkpoint-dir bot/checkpoints

# DASHBOARD  
cd bot/scripts && /home/morningstar/miniconda3/envs/trading_env/bin/python \
  training_dashboard.py

# REPRISE
timeout 30s /home/morningstar/miniconda3/envs/trading_env/bin/python \
  bot/scripts/train_parallel_agents.py --config bot/config/config.yaml \
  --checkpoint-dir bot/checkpoints --resume

# VALIDATION
python test_tensorboard_checkpoint_validation.py
```

**🎉 Le système est maintenant entièrement opérationnel et prêt pour une utilisation en production !**