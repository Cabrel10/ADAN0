# Notebooks d'analyse et de test

Ce répertoire contient des notebooks Jupyter pour tester et analyser les différentes composantes du projet.

## Comment utiliser ces notebooks

1. **Prérequis** :
   - Python 3.8+
   - Jupyter Lab ou Jupyter Notebook
   - Jupytext

2. **Démarrer l'environnement** :
   ```bash
   # Rendre le script exécutable si ce n'est pas déjà fait
   chmod +x ../start_notebook.sh
   
   # Démarrer Jupyter
   ./../start_notebook.sh
   ```

3. **Ouvrir un notebook** :
   - Accédez à `http://localhost:8888` dans votre navigateur
   - Ouvrez le notebook souhaité (fichiers `.py` ou `.ipynb`)

## Liste des notebooks disponibles

- `01_test_shared_experience_buffer.py` : Test et benchmark du SharedExperienceBuffer
  - Test de base avec un seul processus
  - Test avec plusieurs processus en parallèle
  - Analyse des performances avec différentes configurations
  - Visualisation des résultats

## Fonctionnalités Jupytext

Les notebooks sont synchronisés avec des scripts Python (format `py:percent`) qui offrent :

- Meilleure intégration avec git (diffs plus propres)
- Édition dans un IDE
- Exécution en tant que script Python standard

### Commandes utiles

Pour convertir un notebook en script Python :
```bash
jupytext --to py:percent notebook.ipynb
```

Pour convertir un script Python en notebook :
```bash
jupytext --to ipynb script.py
```

## Bonnes pratiques

1. Exécutez toujours les cellules dans l'ordre
2. Sauvegardez régulièrement votre travail
3. Utilisez des cellules markdown pour documenter votre raisonnement
4. Gardez les cellules courtes et ciblées
5. Utilisez des sous-titres pour organiser votre notebook
