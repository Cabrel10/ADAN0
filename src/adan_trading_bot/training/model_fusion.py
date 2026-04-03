import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class ModelFusion:
    """
    Module pour la fusion des décisions de plusieurs workers de trading.
    """
    def __init__(self, config: Dict[str, Any] = None):
        if config is None:
            config = {}
        # Utiliser la configuration de 'model_fusion' si elle existe
        fusion_config = config.get('model_fusion', {})
        self.consensus_threshold = fusion_config.get('consensus_threshold', 0.75)

    def aggregate(self, worker_outputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Agrège les sorties des workers en utilisant une moyenne pondérée par la performance.
        """
        action_scores = {}
        action_confidences = {}

        for worker_id, output in worker_outputs.items():
            action = output.get('action')
            confidence = output.get('confidence', 0.5)
            sharpe = output.get('sharpe', 1.0)
            
            # Le score est une combinaison de la confiance et de la performance (Sharpe)
            score = confidence * sharpe

            if action not in action_scores:
                action_scores[action] = 0.0
                action_confidences[action] = []
            
            action_scores[action] += score
            action_confidences[action].append(confidence)
        
        # Choisir l'action avec le score le plus élevé
        best_action = max(action_scores, key=action_scores.get)
        
        # La confiance de l'action finale peut être la moyenne des confiances des workers l'ayant choisie
        avg_confidence = np.mean(action_confidences[best_action])

        return {'action': best_action, 'confidence': avg_confidence}

    def has_consensus(self, worker_outputs: Dict[str, Dict[str, Any]]) -> bool:
        """
        Vérifie si un consensus (>= 75%) est atteint parmi les workers.
        """
        actions = [output.get('action') for output in worker_outputs.values()]
        if not actions:
            return False
        
        # Compter les occurrences de chaque action
        action_counts = {}
        for action in actions:
            action_counts[action] = action_counts.get(action, 0) + 1
            
        # Vérifier si une action atteint le seuil de consensus
        total_workers = len(actions)
        for action, count in action_counts.items():
            if (count / total_workers) >= self.consensus_threshold:
                return True
        
        return False

    def resolve_conflict(self, worker_outputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Résout les conflits en donnant la priorité au worker de plus haut tier
        et avec les meilleures métriques de performance.
        """
        best_worker = None
        highest_tier = -1
        best_sharpe = -np.inf

        for worker_id, output in worker_outputs.items():
            tier = output.get('tier', 0)
            sharpe = output.get('sharpe', 0.0)

            # Priorité 1: Tier le plus élevé
            if tier > highest_tier:
                highest_tier = tier
                best_sharpe = sharpe
                best_worker = worker_id
            # Priorité 2: En cas d'égalité de tier, meilleur Sharpe
            elif tier == highest_tier:
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_worker = worker_id
        
        if best_worker is None:
            # Fallback de sécurité : retourner la décision du premier worker
            best_worker = next(iter(worker_outputs.keys()))

        final_decision = worker_outputs[best_worker]
        final_decision['source'] = best_worker # Ajouter la source pour la validation

        return final_decision