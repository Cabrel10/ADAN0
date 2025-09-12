.PHONY: lint dead-code vulture analyze test

# Installation des dépendances
install:
	uv pip install -r requirements.txt

# Exécuter tous les linteurs
lint:
	pylint src/adan_trading_bot/
	flake8 src/adan_trading_bot/

# Détecter le code mort avec Vulture
vulture:
	vulture src/adan_trading_bot/ vulture.ini

# Détecter le code mort avec deadcode
dead-code:
	deadcode .

# Exécuter tous les tests
test:
	pytest tests/

# Exécuter toutes les analyses
analyze: lint vulture dead-code

# Nettoyer les fichiers temporaires
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name ".mypy_cache" -exec rm -r {} +
