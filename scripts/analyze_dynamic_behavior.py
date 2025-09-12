# scripts/analyze_dynamic_behavior.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple

class DBEAnalyzer:
    def __init__(self, logs_dir: str = "logs/dbe"):
        self.logs_dir = Path(logs_dir)
        self.df = self._load_logs()

    def _load_logs(self) -> pd.DataFrame:
        """Charge et prépare les logs du DBE."""
        logs = []
        # Look for logs in the specified directory relative to the project root
        log_files = list(self.logs_dir.glob("dbe_*.jsonl"))

        # If no logs found, create dummy data for demonstration
        if not log_files:
            print(f"Aucun fichier de log trouvé dans '{self.logs_dir}'. Création de données de démonstration.")
            self._create_dummy_logs()
            log_files = list(self.logs_dir.glob("dbe_*.jsonl"))

        for log_file in log_files:
            with open(log_file) as f:
                for line in f:
                    try:
                        logs.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # Ignore malformed lines

        if not logs:
            raise ValueError(f"Impossible de charger les logs depuis {self.logs_dir}")

        df = pd.DataFrame(logs)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.sort_values('timestamp').reset_index(drop=True)

    def _create_dummy_logs(self):
        """Crée des fichiers de logs factices pour la démonstration."""
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        dummy_data = [
            {'timestamp': '2025-07-13T10:00:00Z', 'drawdown': 1.5, 'sl_pct': 2.0, 'risk_mode': 'NORMAL', 'reward_boost': 1.0, 'penalty_inaction': 0.0, 'action_frequency': 0.5},
            {'timestamp': '2025-07-13T10:01:00Z', 'drawdown': 3.0, 'sl_pct': 2.2, 'risk_mode': 'DEFENSIVE', 'reward_boost': 1.0, 'penalty_inaction': 0.0, 'action_frequency': 0.4},
            {'timestamp': '2025-07-13T10:02:00Z', 'drawdown': 6.0, 'sl_pct': 2.5, 'risk_mode': 'DEFENSIVE', 'reward_boost': 1.5, 'penalty_inaction': -0.01, 'action_frequency': 0.1},
            {'timestamp': '2025-07-13T10:03:00Z', 'drawdown': 1.0, 'sl_pct': 2.0, 'risk_mode': 'NORMAL', 'reward_boost': 1.8, 'penalty_inaction': 0.0, 'action_frequency': 0.05}
        ]
        with open(self.logs_dir / 'dbe_demo.jsonl', 'w') as f:
            for entry in dummy_data:
                f.write(json.dumps(entry) + '\n')

    def analyze_drawdown_behavior(self) -> Dict:
        """Analyse le comportement du DBE par rapport au drawdown."""
        df = self.df.copy()

        if 'drawdown' not in df.columns or 'sl_pct' not in df.columns or 'risk_mode' not in df.columns:
            return {'recommendations': ["Colonnes 'drawdown', 'sl_pct' ou 'risk_mode' manquantes."]}

        drawdown_bins = pd.cut(df['drawdown'], bins=[0, 2, 5, 10, 20, 100], right=False)
        metrics = df.groupby(drawdown_bins).agg(
            sl_pct_mean=('sl_pct', 'mean'),
            sl_pct_std=('sl_pct', 'std'),
            defensive_mode_pct=('risk_mode', lambda x: (x == 'DEFENSIVE').mean() * 100)
        ).reset_index()

        # Visualisation
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        sns.barplot(data=metrics, x='drawdown', y='sl_pct_mean', ax=ax1)
        ax1.set_title('Stop Loss moyen par niveau de drawdown')
        ax1.set_ylabel('SL moyen (%)')

        sns.barplot(data=metrics, x='drawdown', y='defensive_mode_pct', ax=ax2)
        ax2.set_title('% de temps en mode DEFENSIVE par niveau de drawdown')
        ax2.set_xlabel('Niveau de Drawdown (%)')
        ax2.set_ylabel('% du temps')
        plt.xticks(rotation=45)
        plt.tight_layout()

        report_path = Path('reports/figures')
        report_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(report_path / 'dbe_drawdown_analysis.png')
        plt.close(fig)

        recommendations = self._generate_drawdown_recommendations(metrics)
        metrics['drawdown'] = metrics['drawdown'].astype(str)
        return {
            'sensitivity_analysis': metrics.set_index('drawdown').to_dict('index'),
            'recommendations': recommendations
        }

    def _generate_drawdown_recommendations(self, metrics: pd.DataFrame) -> List[str]:
        recs = []
        if metrics['sl_pct_mean'].pct_change().fillna(0).mean() < 0.05:
            recs.append("La sensibilité du SL au drawdown est faible. Envisagez d'augmenter `drawdown_sl_factor`.")
        
        defensive_at_low_drawdown = metrics[metrics['drawdown'].apply(lambda x: x.left < 2)]
        if not defensive_at_low_drawdown.empty and (defensive_at_low_drawdown['defensive_mode_pct'] > 10).any():
            recs.append("Le mode DEFENSIVE se déclenche pour un drawdown faible (<2%). Envisagez d'ajuster `min_drawdown_threshold`.")
        
        if not recs:
            recs.append("Le comportement face au drawdown semble nominal.")
        return recs

    def analyze_reward_structure(self) -> Dict:
        """Analyse la structure des récompenses."""
        df = self.df
        recs = []
        results = {}

        if 'reward_boost' in df.columns:
            boost_freq = (df['reward_boost'] > 1.0).mean() * 100
            results['boost_frequency_pct'] = boost_freq
            if boost_freq < 10:
                recs.append(f"Le `reward_boost` est rare ({boost_freq:.1f}%). Vérifiez `winrate_threshold`.")
        
        if 'penalty_inaction' in df.columns:
            penalty_freq = (df['penalty_inaction'] < 0).mean() * 100
            results['penalty_frequency_pct'] = penalty_freq
            if penalty_freq > 30:
                recs.append(f"La pénalité d'inaction est fréquente ({penalty_freq:.1f}%). Vérifiez `inaction_factor`.")

        if not recs:
            recs.append("La structure des récompenses semble équilibrée.")
        results['recommendations'] = recs
        return results

    def generate_report(self):
        """Génère et sauvegarde un rapport complet."""
        report = {
            'report_generated_at': pd.Timestamp.now().isoformat(),
            'drawdown_analysis': self.analyze_drawdown_behavior(),
            'reward_analysis': self.analyze_reward_structure()
        }

        report_path = Path('reports')
        report_path.mkdir(parents=True, exist_ok=True)
        with open(report_path / 'dbe_analysis_report.json', 'w') as f:
            class NpEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, (np.integer, np.floating)):
                        return obj.item()
                    if isinstance(obj, pd.Interval):
                        return str(obj)
                    return super(NpEncoder, self).default(obj)
            json.dump(report, f, indent=2, cls=NpEncoder)

def main():
    """Point d'entrée du script."""
    print("Lancement de l'analyse du comportement du DBE...")
    try:
        # Assumes the script is run from the ADAN project root
        analyzer = DBEAnalyzer(logs_dir="logs/dbe")
        analyzer.generate_report()
        
        print("\n✅ Rapport généré : 'reports/dbe_analysis_report.json'")
        print("   Graphique sauvegardé : 'reports/figures/dbe_drawdown_analysis.png'")

        # Display recommendations from the generated report
        with open('reports/dbe_analysis_report.json') as f:
            report = json.load(f)
        print("\n=== Recommandations Clés ===")
        for section in ['drawdown_analysis', 'reward_analysis']:
            if report.get(section, {}).get('recommendations'):
                print(f"\n--- {section.replace('_', ' ').title()} ---")
                for rec in report[section]['recommendations']:
                    print(f"  - {rec}")

    except (ValueError, FileNotFoundError) as e:
        print(f"\nErreur : {e}")
    except Exception as e:
        print(f"\nUne erreur inattendue est survenue : {e}")

if __name__ == "__main__":
    main()
