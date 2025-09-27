# Design Document - Plan A: Interface & Visualisation

## Overview

Ce document présente l'architecture technique pour finaliser le Plan A du Sprint 10 d'ADAN Trading Bot. Le design se base sur l'infrastructure UI existante (PySide6/PyQt6) et l'étend avec de nouveaux composants d'analyse, une localisation complète, et des optimisations de performance.

## Architecture

### Architecture Globale

```
ADAN Trading Bot UI
├── Core UI Framework (PySide6/PyQt6)
├── Localization System (QTranslator + .ts/.qm files)
├── Existing Components (Enhanced)
│   ├── MainWindow (main_window.py)
│   ├── ChartWidget (chart_widget.py) 
│   ├── SidePanel (side_panel.py)
│   └── ConfigDialog (config_dialog.py)
├── New Analysis Components
│   ├── PerformanceWidget
│   ├── DBEAnalysisWidget
│   └── ReportingWidget
├── Theme System (TradingView-style)
└── Integration Layer (Plan B Components)
```

### Stratégie de Localisation

La localisation utilise une approche hybride combinant :
- **Regex-based String Extraction** : Pour identifier et envelopper les chaînes traduisibles
- **Qt Linguist Workflow** : Pour la gestion des traductions
- **Runtime Language Switching** : Pour le changement de langue dynamique

## Components and Interfaces

### 1. Localization Engine

#### LocalizationProcessor
```python
class LocalizationProcessor:
    """Processeur pour automatiser la localisation des fichiers UI"""
    
    def __init__(self):
        self.translatable_patterns = [
            r'QLabel\("([^"]+)"\)',
            r'QGroupBox\("([^"]+)"\)',
            r'QPushButton\("([^"]+)"\)',
            r'setWindowTitle\("([^"]+)"\)',
            r'setText\("([^"]+)"\)',
            # ... autres patterns
        ]
    
    def process_file(self, file_path: str) -> str:
        """Traite un fichier et enveloppe les chaînes avec self.tr()"""
        
    def extract_strings(self, content: str) -> List[str]:
        """Extrait toutes les chaînes traduisibles"""
        
    def wrap_with_tr(self, content: str) -> str:
        """Enveloppe les chaînes avec self.tr()"""
```

#### TranslationManager
```python
class TranslationManager:
    """Gestionnaire des traductions runtime"""
    
    def __init__(self, app: QApplication):
        self.app = app
        self.translator = QTranslator()
        
    def load_translation(self, language: str) -> bool:
        """Charge une traduction spécifique"""
        
    def get_available_languages(self) -> List[str]:
        """Retourne les langues disponibles"""
        
    def switch_language(self, language: str) -> None:
        """Change la langue de l'interface"""
```

### 2. Enhanced Existing Components

#### ChartWidget (Optimisé)
```python
class ChartWidget(QWidget):
    """Widget graphique optimisé pour haute performance"""
    
    def __init__(self):
        super().__init__()
        self.plot_widget = pg.PlotWidget()
        self.data_cache = {}  # Cache intelligent
        self.render_timer = QTimer()  # Throttling des updates
        
    def update_timeframe(self, timeframe: str) -> None:
        """Mise à jour optimisée du timeframe (<500ms)"""
        
    def add_indicators(self, indicators: Dict) -> None:
        """Ajout d'indicateurs avec lazy loading"""
        
    def optimize_rendering(self) -> None:
        """Optimisations de rendu pour 1000+ candles"""
```

#### SidePanel (Métriques Temps Réel)
```python
class SidePanel(QWidget):
    """Panel latéral avec métriques temps réel optimisées"""
    
    def __init__(self):
        super().__init__()
        self.metrics_timer = QTimer()
        self.metrics_timer.timeout.connect(self.update_metrics)
        self.metrics_timer.start(100)  # Update every 100ms
        
    def update_metrics(self) -> None:
        """Mise à jour des métriques (<100ms)"""
        
    def add_tooltip_support(self) -> None:
        """Ajoute des tooltips informatifs"""
```

### 3. New Analysis Components

#### PerformanceWidget
```python
class PerformanceWidget(QWidget):
    """Widget d'analyse de performance avec graphiques avancés"""
    
    def __init__(self):
        super().__init__()
        self.equity_chart = pg.PlotWidget()
        self.drawdown_chart = pg.PlotWidget()
        self.heatmap_widget = QWidget()
        self.benchmark_selector = QComboBox()
        
    def create_equity_curve(self, data: pd.DataFrame) -> None:
        """Crée la courbe d'équité"""
        
    def create_drawdown_chart(self, data: pd.DataFrame) -> None:
        """Crée le graphique de drawdown avec zones critiques"""
        
    def create_trade_heatmap(self, trades: List[Dict]) -> None:
        """Crée la heatmap des trades profit/loss"""
        
    def compare_with_benchmark(self, benchmark: str) -> None:
        """Compare avec un benchmark"""
        
    def export_charts(self, format: str = "png") -> str:
        """Exporte les graphiques en haute résolution"""
```

#### DBEAnalysisWidget
```python
class DBEAnalysisWidget(QWidget):
    """Widget d'analyse du Dynamic Behavior Engine"""
    
    def __init__(self):
        super().__init__()
        self.mode_histogram = pg.PlotWidget()
        self.sl_tp_evolution = pg.PlotWidget()
        self.correlation_matrix = QWidget()
        self.adaptation_metrics = QWidget()
        
    def create_mode_histogram(self, dbe_data: pd.DataFrame) -> None:
        """Crée l'histogramme des modes DBE"""
        
    def create_sl_tp_evolution(self, dbe_data: pd.DataFrame) -> None:
        """Visualise l'évolution des paramètres SL/TP"""
        
    def calculate_performance_correlation(self, data: pd.DataFrame) -> Dict:
        """Calcule la corrélation performance/régime"""
        
    def update_adaptation_metrics(self, metrics: Dict) -> None:
        """Met à jour les métriques d'adaptation"""
```

#### ReportingWidget
```python
class ReportingWidget(QWidget):
    """Widget de génération de rapports automatiques"""
    
    def __init__(self):
        super().__init__()
        self.template_selector = QComboBox()
        self.export_options = QGroupBox()
        self.schedule_options = QGroupBox()
        self.progress_bar = QProgressBar()
        
    def generate_pdf_report(self, template: str, data: Dict) -> str:
        """Génère un rapport PDF professionnel"""
        
    def export_csv_data(self, data: pd.DataFrame, filename: str) -> None:
        """Exporte les données en CSV"""
        
    def setup_scheduled_reports(self, schedule: Dict) -> None:
        """Configure la génération automatique de rapports"""
        
    def create_custom_template(self, template_config: Dict) -> str:
        """Crée un template personnalisé"""
```

### 4. Theme System

#### ThemeManager
```python
class ThemeManager:
    """Gestionnaire de thèmes TradingView-style"""
    
    def __init__(self):
        self.themes = {
            'light': self.load_light_theme(),
            'dark': self.load_dark_theme()
        }
        
    def load_light_theme(self) -> Dict:
        """Charge le thème clair"""
        
    def load_dark_theme(self) -> Dict:
        """Charge le thème sombre TradingView-style"""
        
    def apply_theme(self, app: QApplication, theme_name: str) -> None:
        """Applique un thème à l'application"""
        
    def create_tradingview_stylesheet(self) -> str:
        """Crée le stylesheet TradingView"""
```

### 5. Integration Layer

#### PlanBIntegrator
```python
class PlanBIntegrator:
    """Intégrateur pour les composants du Plan B"""
    
    def __init__(self, main_window):
        self.main_window = main_window
        self.workflow_orchestrator = None
        self.api_manager = None
        self.trading_interface = None
        self.health_monitor = None
        
    def connect_workflow_orchestrator(self, orchestrator) -> None:
        """Connecte le WorkflowOrchestrator à l'UI"""
        
    def connect_api_manager(self, api_manager) -> None:
        """Connecte le SecureAPIManager à l'UI"""
        
    def connect_trading_interface(self, trading_interface) -> None:
        """Connecte le ManualTradingInterface à l'UI"""
        
    def connect_health_monitor(self, health_monitor) -> None:
        """Connecte le SystemHealthMonitor à l'UI"""
        
    def setup_real_time_updates(self) -> None:
        """Configure les mises à jour temps réel"""
```

## Data Models

### LocalizationData
```python
@dataclass
class LocalizationData:
    """Modèle de données pour la localisation"""
    source_file: str
    translatable_strings: List[str]
    processed_content: str
    language_files: Dict[str, str]  # language -> file_path
```

### PerformanceData
```python
@dataclass
class PerformanceData:
    """Modèle de données pour l'analyse de performance"""
    equity_curve: pd.Series
    drawdown_series: pd.Series
    trades: List[Dict]
    benchmark_data: pd.Series
    metrics: Dict[str, float]
```

### DBEAnalysisData
```python
@dataclass
class DBEAnalysisData:
    """Modèle de données pour l'analyse DBE"""
    mode_history: pd.DataFrame
    sl_tp_evolution: pd.DataFrame
    regime_correlation: Dict[str, float]
    adaptation_metrics: Dict[str, float]
```

## Error Handling

### Stratégie de Gestion d'Erreurs

1. **Localization Errors**
   - Fallback vers l'anglais si traduction manquante
   - Logging des chaînes non traduites
   - Validation des fichiers .ts/.qm

2. **Performance Errors**
   - Throttling automatique si lag détecté
   - Fallback vers rendu simplifié
   - Monitoring des performances UI

3. **Data Errors**
   - Validation des données avant affichage
   - Messages d'erreur utilisateur clairs
   - Récupération gracieuse des erreurs

4. **Integration Errors**
   - Gestion des déconnexions Plan B
   - Retry automatique avec backoff
   - État de fallback pour chaque composant

## Testing Strategy

### Tests Unitaires
- **LocalizationProcessor** : Tests de regex et extraction
- **PerformanceWidget** : Tests de génération de graphiques
- **DBEAnalysisWidget** : Tests de calculs statistiques
- **ReportingWidget** : Tests de génération PDF/CSV

### Tests d'Intégration
- **UI Responsiveness** : Tests de performance (<100ms, <500ms)
- **Theme Switching** : Tests de changement de thème
- **Plan B Integration** : Tests de communication avec composants Plan B
- **Localization** : Tests de changement de langue

### Tests de Performance
- **Chart Rendering** : 1000+ candles à >30 FPS
- **Memory Usage** : Monitoring de la consommation mémoire
- **Startup Time** : Interface principale <3 secondes
- **Real-time Updates** : Métriques <100ms

### Tests d'Accessibilité
- **Keyboard Navigation** : Tous les éléments accessibles au clavier
- **Screen Reader** : Compatibilité avec lecteurs d'écran
- **High Contrast** : Support des thèmes à fort contraste
- **Tooltips** : Informations contextuelles disponibles

## Implementation Notes

### Priorités d'Implémentation
1. **Phase 1** : Finalisation localisation (config_dialog.py + autres fichiers)
2. **Phase 2** : Optimisation composants existants (performance, tooltips)
3. **Phase 3** : Nouveaux composants d'analyse (PerformanceWidget, DBEAnalysisWidget)
4. **Phase 4** : Système de rapports (ReportingWidget)
5. **Phase 5** : Thème et polish final

### Considérations Techniques
- **Regex Patterns** : Utilisation de patterns robustes pour l'extraction de chaînes
- **Memory Management** : Cache intelligent avec LRU pour les graphiques
- **Threading** : QThread pour les calculs lourds (éviter le blocage UI)
- **Lazy Loading** : Chargement différé des composants lourds
- **Signal/Slot** : Communication asynchrone entre composants

### Compatibilité
- **Qt Version** : Compatible PySide6/PyQt6
- **Python Version** : Python 3.8+
- **OS Support** : Windows, macOS, Linux
- **Resolution** : Support écrans haute résolution (4K+)