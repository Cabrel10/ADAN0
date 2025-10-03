
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import gymnasium as gym

class MultiTimeframeCNN(nn.Module):
    """CNN pour extraction de features multi-timeframes"""
    
    def __init__(self, input_shape: Tuple[int, int, int], n_features_per_tf: int = 10):
        super(MultiTimeframeCNN, self).__init__()
        
        self.n_timeframes = input_shape[0]  # 3 timeframes
        self.window_size = input_shape[1]   # Taille de fenêtre temporelle
        self.n_features = input_shape[2]    # Nombre de features par timeframe
        
        # Couches CNN par timeframe
        self.tf_convs = nn.ModuleList([
            nn.Sequential(
                # Conv1D sur la dimension temporelle
                nn.Conv1d(n_features_per_tf, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(16),  # Réduire la dimension temporelle
                nn.Flatten()
            ) for _ in range(self.n_timeframes)
        ])
        
        # Couche de fusion temporelle
        tf_output_size = 64 * 16  # 64 filtres × 16 positions après pooling
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.n_timeframes * tf_output_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Initialisation des poids
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Traitement avant: [batch, timeframes, window, features]"""
        
        # x shape: [batch_size, 3, window_size, n_features]
        batch_size = x.shape[0]
        
        # Traiter chaque timeframe indépendamment
        tf_features = []
        for i in range(self.n_timeframes):
            # Sélectionner un timeframe: [batch, window, features]
            tf_data = x[:, i, :, :]
            
            # Transposer pour Conv1D: [batch, features, window]
            tf_data = tf_data.transpose(1, 2)
            
            # Appliquer le CNN: [batch, features_out]
            features = self.tf_convs[i](tf_data)
            tf_features.append(features)
        
        # Fusion des features temporelles: [batch, n_timeframes × features_out]
        combined_features = torch.cat(tf_features, dim=1)
        
        # Fusion finale: [batch, 128]
        fused = self.fusion_layer(combined_features)
        
        return fused
    
    def _initialize_weights(self):
        """Initialisation optimisée des poids"""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)

class PPOMemoryNetwork(nn.Module):
    """Réseau PPO avec mémoire pour mémoriser les schémas"""
    
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 128):
        super(PPOMemoryNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.action_dim = action_dim
        
        # Réseau politique (acteur)
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, action_dim * 2)  # mean et std pour chaque action
        )
        
        # Réseau de valeur (critique)
        self.value_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)  # Valeur unique
        )
        
        # Mémoire des schémas (LSTM pour séquences)
        self.memory_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            dropout=0.1 if hidden_dim > 64 else 0.0
        )
        
        # Couche d'attention pour pondération temporelle
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim // 2,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
    
    def forward(self, state: torch.Tensor, hidden_state: Optional[Tuple] = None) -> Dict:
        """Traitement avant avec mémoire"""
        
        # LSTM pour mémoire des schémas
        if hidden_state is not None:
            lstm_out, new_hidden = self.memory_lstm(state.unsqueeze(1), hidden_state)
        else:
            lstm_out, new_hidden = self.memory_lstm(state.unsqueeze(1))
        
        # Aplatir la sortie LSTM
        lstm_features = lstm_out.squeeze(1)  # [batch, hidden_dim//2]
        
        # Attention sur les features mémoire
        attn_out, _ = self.attention(lstm_features.unsqueeze(1), 
                                   lstm_features.unsqueeze(1), 
                                   lstm_features.unsqueeze(1))
        attn_features = attn_out.squeeze(1)
        
        # Combiner mémoire et état actuel
        combined_input = lstm_features + attn_features  # Fusion résiduelle
        
        # Politique (acteur)
        policy_logits = self.policy_net(combined_input)
        action_mean, action_logstd = policy_logits.chunk(2, dim=1)
        action_std = torch.exp(action_logstd)
        
        # Valeur (critique)
        value = self.value_net(combined_input)
        
        return {
            'action_mean': action_mean,
            'action_std': action_std,
            'value': value,
            'hidden_state': new_hidden,
            'lstm_features': lstm_features,
            'attention_features': attn_features
        }

class IntegratedCNNPPOModel(nn.Module):
    """Modèle intégré CNN+PPO pour trading multi-timeframes"""
    
    def __init__(self, input_shape: Tuple, action_dim: int = 2):
        super(IntegratedCNNPPOModel, self).__init__()
        
        # CNN pour extraction de features
        self.cnn = MultiTimeframeCNN(input_shape)
        cnn_output_dim = 128  # Sortie de la couche de fusion
        
        # PPO avec mémoire
        self.ppo = PPOMemoryNetwork(cnn_output_dim, action_dim)
        
        # Métriques pour analyse
        self.feature_importance = {}
    
    def forward(self, observation: torch.Tensor, hidden_state: Optional[Tuple] = None) -> Dict:
        """Flux complet: CNN → Mémoire → PPO"""
        
        # Étape 1: Extraction de features par le CNN
        cnn_features = self.cnn(observation)  # [batch, 128]
        
        # Étape 2: Traitement PPO avec mémoire des schémas
        ppo_output = self.ppo(cnn_features, hidden_state)
        
        # Métriques pour analyse (optionnel)
        self._compute_feature_importance(cnn_features, ppo_output)
        
        return ppo_output
    
    def _compute_feature_importance(self, cnn_features: torch.Tensor, ppo_output: Dict):
        """Calcule l'importance des features pour analyse"""
        # Analyse de l'importance relative des timeframes
        # (À étendre selon les besoins d'analyse)
        pass
    
    def get_action(self, observation: np.ndarray, hidden_state: Optional[Tuple] = None) -> Tuple:
        """Génère une action depuis l'observation"""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
            output = self.forward(obs_tensor, hidden_state)
            
            # Échantillonnage de l'action
            action_mean = output['action_mean'].numpy()[0]
            action_std = output['action_std'].numpy()[0]
            
            # Échantillonnage gaussien
            action = np.random.normal(action_mean, action_std)
            
            # Clipping pour stabilité numérique
            action = np.clip(action, -1.0, 1.0)
            
            return action, output['hidden_state']
    
    def evaluate_action(self, observation: np.ndarray, action: np.ndarray, 
                       hidden_state: Optional[Tuple] = None) -> Dict:
        """Évalue une action pour calcul de l'advantage"""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
            output = self.forward(obs_tensor, hidden_state)
            
            # Log-probabilité de l'action
            action_tensor = torch.FloatTensor(action).unsqueeze(0)
            action_mean = output['action_mean']
            action_std = output['action_std']
            
            # Calcul de la log-probabilité (distribution gaussienne)
            var = action_std ** 2
            logp = -0.5 * torch.sum(
                torch.log(2 * np.pi * var) + ((action_tensor - action_mean) ** 2) / var,
                dim=1
            )
            
            return {
                'value': output['value'].numpy()[0, 0],
                'logp': logp.numpy()[0],
                'action_mean': action_mean.numpy()[0],
                'action_std': action_std.numpy()[0]
            }

class CNNPPOAgent:
    """Agent PPO+CNN intégré pour le trading"""
    
    def __init__(self, observation_shape: Tuple, action_dim: int = 2):
        self.observation_shape = observation_shape
        self.action_dim = action_dim
        
        # Modèle intégré
        self.model = IntegratedCNNPPOModel(observation_shape, action_dim)
        
        # État caché pour mémoire des schémas
        self.hidden_state = None
        
        # Métriques de performance
        self.step_count = 0
        self.action_history = []
        self.reward_history = []
    
    def act(self, observation: np.ndarray) -> np.ndarray:
        """Sélectionne une action basée sur l'observation actuelle"""
        
        # Conversion en tenseur
        obs_tensor = torch.FloatTensor(observation)
        
        # Génération de l'action
        action, new_hidden_state = self.model.get_action(observation, self.hidden_state)
        
        # Mise à jour de l'état caché (mémoire des schémas)
        self.hidden_state = new_hidden_state
        
        # Logging pour analyse
        self.step_count += 1
        self.action_history.append(action.copy())
        
        return action
    
    def remember(self, observation: np.ndarray, action: np.ndarray, 
                reward: float, next_observation: np.ndarray, done: bool):
        """Mémorise l'expérience pour apprentissage"""
        self.reward_history.append(reward)
        
        # La mémoire est gérée par le modèle PPO interne
        # Cette méthode sera utilisée lors de l'implémentation complète
    
    def learn(self):
        """Apprentissage PPO avec les expériences mémorisées"""
        # Implémentation de l'algorithme PPO
        # (À implémenter selon les besoins spécifiques)
        pass

# Exemple d'utilisation
if __name__ == '__main__':
    # Configuration typique pour le trading
    observation_shape = (3, 50, 10)  # 3 timeframes, 50 points, 10 features
    action_dim = 2  # [position_size, timeframe_selection]
    
    # Création de l'agent
    agent = CNNPPOAgent(observation_shape, action_dim)
    
    # Exemple d'observation multi-timeframes
    sample_observation = np.random.randn(*observation_shape)
    
    # Génération d'une action
    action = agent.act(sample_observation)
    
    print(f'✅ Agent créé avec observation_shape: {observation_shape}')
    print(f'✅ Action générée: {action}')
    print(f'✅ Étapes effectuées: {agent.step_count}')
    
    print('
📊 Architecture:')
    print(f'   CNN Timeframes: {agent.model.cnn.n_timeframes}')
    print(f'   CNN Features: {agent.model.cnn.n_features}')
    print(f'   PPO Hidden: {agent.model.ppo.policy_net[0].out_features}')
    print(f'   LSTM Layers: {agent.model.ppo.memory_lstm.num_layers}')
