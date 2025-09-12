# Configuration for Jupyter Notebook
c = get_config()  # noqa

# Activer Jupytext
c.NotebookApp.contents_manager_class = "jupytext.TextFileContentsManager"

# Configurer Jupytext pour utiliser le format py:percent par défaut
c.ContentsManager.default_jupytext_formats = "ipynb,py:percent"

# Désactiver la création automatique de nouveaux notebooks
c.FileContentsManager.always_delete_nonempty_dir = True

# Configurer le répertoire racine
import os
c.NotebookApp.notebook_dir = os.path.expanduser(os.path.join(os.getcwd(), 'notebooks'))

# Désactiver la vérification du jeton (pour les connexions locales)
c.NotebookApp.token = ''
c.NotebookApp.password = ''

# Désactiver l'ouverture automatique du navigateur
c.NotebookApp.open_browser = False

# Configurer le port (par défaut: 8888)
c.NotebookApp.port = 8888
