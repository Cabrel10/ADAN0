"""Feature extractors for the ADAN trading agent.

This module contains custom feature extractors for processing market data
in the form of images and vectors for use with reinforcement learning agents.
"""

# Standard library imports
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

# Third-party imports
import gymnasium as gym
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# Local application imports
from adan_trading_bot.common.utils import get_logger

# Set up logging
logger = get_logger(__name__)

# Define type hints for better code readability
Tensor = th.Tensor

class ChannelAttention(nn.Module):
    """Channel Attention Module.

    This module implements channel attention mechanism which learns to focus on
    important channels in the input feature maps.

    Args:
        num_channels: Number of input channels
        reduction_ratio: Reduction ratio for the bottleneck layer.
            Defaults to 16.
    """

    def __init__(self, num_channels: int, reduction_ratio: int = 16) -> None:
        """Initialize the ChannelAttention module.

        Args:
            num_channels: Number of input channels
            reduction_ratio: Reduction ratio for the bottleneck layer
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP with bottleneck
        hidden_channels = max(num_channels // reduction_ratio, 4)
        self.mlp = nn.Sequential(
            nn.Linear(num_channels, hidden_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, num_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for channel attention.

        Args:
            x: Input feature map of shape (batch_size, channels, height, width)

        Returns:
            Output feature map with channel attention applied
        """
        # Average and max pooling along spatial dimensions
        avg_out = self.mlp(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.mlp(self.max_pool(x).view(x.size(0), -1))

        # Combine and apply sigmoid
        channel_weights = self.sigmoid(avg_out + max_out)

        # Apply attention weights to input
        return x * channel_weights.view(x.size(0), x.size(1), 1, 1)

class TemporalAttention(nn.Module):
    """Temporal Attention module for sequence data.

    Applies multi-head self-attention along the temporal dimension to capture
    long-range dependencies in sequential data.

    Args:
        num_features: Number of input features
        num_heads: Number of attention heads (default: 4)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(self, num_features: int, num_heads: int = 4,
                 dropout: float = 0.1) -> None:
        """Initialize the Temporal Attention module.

        Args:
            num_features: Number of input features
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        if num_features % num_heads != 0:
            raise ValueError(
                f"Number of features ({num_features}) must be divisible by "
                f"number of heads ({num_heads})")

        self.num_heads = num_heads
        self.head_dim = num_features // num_heads
        self.scale = self.head_dim ** -0.5

        # Projections for query, key, value
        self.query = nn.Linear(num_features, num_features)
        self.key = nn.Linear(num_features, num_features)
        self.value = nn.Linear(num_features, num_features)

        # Output projection
        self.out_proj = nn.Linear(num_features, num_features)

        # Dropout and layer norm
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(num_features)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Forward pass through the temporal attention module.

        Args:
            x: Input tensor of shape [batch, seq_len, features]
            mask: Optional mask tensor of shape [batch, seq_len] or
                [batch, seq_len, seq_len] where True/1 indicates positions
                that should be masked out

        Returns:
            Output tensor with temporal attention applied
        """
        batch_size, seq_len, _ = x.size()

        # Project queries, keys and values
        q = self.query(x).view(
            batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(
            batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(
            batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = th.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 2:  # [batch, seq_len]
                mask = mask.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, seq_len]
            elif mask.dim() == 3:  # [batch, seq_len, seq_len]
                mask = mask.unsqueeze(1)  # [batch, 1, seq_len, seq_len]

            # Apply mask (1s are masked out, 0s are kept)
            scores = scores.masked_fill(mask, float('-inf'))

        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = th.matmul(attn_weights, v)

        # Reshape and project back to original dimension
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, -1)
        output = self.out_proj(output)

        # Add residual connection and layer norm
        return self.norm(x + output)

class MultiScaleFeatureExtractor(nn.Module):
    """Multi-scale feature extractor using parallel convolutions.

    Applies multiple parallel convolutional branches with different kernel sizes
    to capture features at different scales, then combines the results.

    Args:
        in_channels: Number of input channels
        out_channels_per_branch: Number of output channels per branch
        kernel_sizes: List of kernel sizes for each branch. Defaults to [3, 5, 7, 9]
    """

    def __init__(self, in_channels: int, out_channels_per_branch: int = 32,
                 kernel_sizes: Optional[List[int]] = None) -> None:
        """Initialize the multi-scale feature extractor.

        Args:
            in_channels: Number of input channels
            out_channels_per_branch: Number of output channels per branch
            kernel_sizes: List of kernel sizes for each branch
        """
        super().__init__()

        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7, 9]

        self.branches = nn.ModuleList()

        for kernel_size in kernel_sizes:
            # Calculate padding to maintain spatial dimensions
            padding = (kernel_size - 1) // 2

            branch = nn.Sequential(
                nn.Conv2d(in_channels, out_channels_per_branch,
                         kernel_size=kernel_size, padding=padding, bias=False),
                nn.BatchNorm2d(out_channels_per_branch),
                nn.ReLU(inplace=True)
            )
            self.branches.append(branch)

        # Number of output channels is num_branches * out_channels_per_branch
        self.out_channels = len(kernel_sizes) * out_channels_per_branch

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the multi-scale feature extractor.

        Args:
            x: Input tensor of shape [batch, in_channels, height, width]

        Returns:
            Concatenated output from all branches with shape
            [batch, num_branches * out_channels_per_branch, height, width]
        """
        # Process input through all branches in parallel
        branch_outputs = [branch(x) for branch in self.branches]

        # Concatenate along the channel dimension
        return th.cat(branch_outputs, dim=1)

class CustomCNNFeatureExtractor(BaseFeaturesExtractor):
    """Custom CNN feature extractor for processing market data as images.

    This feature extractor processes both image-like market data and vector
    features, applies several convolutional layers with channel and temporal
    attention, and combines the processed features with additional vector
    features for use in reinforcement learning.

    Args:
        observation_space: The observation space (must be a gym.spaces.Dict)
        features_dim: Dimension of the output features
        num_input_channels: Number of input channels in the image data
        cnn_config: Configuration dictionary for the CNN
    """

    def __init__(self,
                 observation_space: gym.spaces.Space,
                 features_dim: int = 256,
                 num_input_channels: int = 3,
                 cnn_config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the custom CNN feature extractor with channel groups and attention.

        Args:
            observation_space: The observation space (gym.spaces.Dict or gym.spaces.Box)
                For Dict: with 'image_features' and 'vector_features' keys
                For Box: treated as image_features only
            features_dim: Dimension of the output features
            num_input_channels: Number of input channels in the image data
            cnn_config: Configuration dictionary for the CNN with the following structure:
                - channel_groups: Dict defining channel groups and their processing
                - conv_layers: List of dicts for convolutional layers
                - pooling: Type of pooling ('max' or 'avg')
                - dropout: Dropout probability
                - activation: Activation function ('relu' or 'leaky_relu')
                - use_channel_attention: Whether to use channel attention
                - use_temporal_attention: Whether to use temporal attention
        """
        # Initialize the parent class
        super().__init__(observation_space, features_dim)

        # Store configuration
        self.num_input_channels = num_input_channels
        self.features_dim = features_dim
        self.cnn_config = cnn_config or {
            'channel_groups': {
                'price': {'indices': [0, 1, 2], 'out_channels': 32},
                'volume': {'indices': [3], 'out_channels': 16},
                'indicators': {'indices': [4, 5, 6], 'out_channels': 32}
            },
            'conv_layers': [
                {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                {'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            ],
            'pooling': 'max',
            'dropout': 0.2,
            'activation': 'leaky_relu',
            'use_channel_attention': True,
            'use_temporal_attention': True,
            'temporal_attention_heads': 4,
            'temporal_attention_dropout': 0.1
        }

        # Initialize the parent class
        super().__init__(observation_space, features_dim)

        # Build the CNN
        self.cnn = self._build_cnn(self.cnn_config)

        # Get the output dimension of the CNN
        with th.no_grad():
            # Create a dummy input to determine the output dimension - handle both Dict and Box
            if isinstance(observation_space, gym.spaces.Dict):
                # Dict observation space
                dummy_height = observation_space['image_features'].shape[-2]
                dummy_width = observation_space['image_features'].shape[-1]
            else:
                # Box observation space (C, H, W)
                dummy_height = observation_space.shape[-2]
                dummy_width = observation_space.shape[-1]

            dummy_input = th.zeros(1, num_input_channels, dummy_height, dummy_width)

            # Process through channel groups if enabled
            if hasattr(self, 'channel_groups') and self.channel_groups is not None:
                group_outputs = []
                for group_name, group in self.channel_groups.items():
                    group_indices = self.cnn_config['channel_groups'][group_name]['indices']
                    group_input = dummy_input[:, group_indices, :, :]
                    group_outputs.append(group(group_input))
                cnn_input = th.cat(group_outputs, dim=1)
            else:
                cnn_input = dummy_input

            # Process through shared CNN
            cnn_output = self.shared_cnn(cnn_input)
            self.cnn_output_dim = cnn_output.view(-1).shape[0]

        # Get the vector features dimension - handle both Dict and Box observation spaces
        if isinstance(observation_space, gym.spaces.Dict) and 'vector_features' in observation_space.spaces:
            self.vector_features_dim = observation_space['vector_features'].shape[0]
        else:
            # For Box observations, use default vector features dim
            self.vector_features_dim = 64

        # Vector features processing layer
        self.vector_fc = nn.Sequential(
            nn.Linear(self.vector_features_dim, self.vector_features_dim),
            nn.ReLU(),
            nn.Dropout(cnn_config.get('dropout', 0.2))
        )

        # Final fully connected layers
        self.combined_fc = nn.Sequential(
            nn.Linear(self.cnn_output_dim + self.vector_features_dim, 512),
            nn.ReLU(),
            nn.Dropout(cnn_config.get('dropout', 0.2)),
            nn.Linear(512, features_dim),
            nn.LayerNorm(features_dim)
        )

        # Initialize weights
        self._initialize_weights()

    def _build_channel_group(
        self,
        group_cfg: Dict[str, Any],
        global_cfg: Dict[str, Any]
    ) -> nn.Sequential:
        """Build a processing block for a channel group.

        Args:
            group_cfg: Group configuration with 'indices' and 'out_channels'
            global_cfg: Global configuration

        Returns:
            A sequential module for processing the channel group
        """
        layers = []
        in_channels = len(group_cfg['indices'])
        out_channels = group_cfg['out_channels']

        # Initial 1x1 conv to project to desired channels
        layers.extend([
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        ])

        # Add channel attention if enabled
        if global_cfg.get('use_channel_attention', False):
            layers.append(ChannelAttention(
                num_channels=out_channels,
                reduction_ratio=8
            ))

        return nn.Sequential(*layers)

    def _build_cnn(self, config: Dict[str, Any]) -> nn.Module:
        """Build the CNN architecture with channel groups and attention.

        Constructs a CNN with the following components:
        1. Parallel processing of channel groups
        2. Shared convolutional layers
        3. Optional temporal attention

        Args:
            config: Configuration dictionary containing:
                - channel_groups: Dict of channel group configurations
                - conv_layers: List of dicts for shared conv layers
                - pooling: Type of pooling ('max' or 'avg')
                - dropout: Dropout probability
                - activation: Activation function ('relu' or 'leaky_relu')
                - use_channel_attention: Whether to use channel attention
                - use_temporal_attention: Whether to use temporal attention
                - temporal_attention_heads: Number of attention heads
                - temporal_attention_dropout: Dropout for attention

        Returns:
            A module that processes the input through the CNN architecture
        """
        # Initialize channel groups if specified
        if 'channel_groups' in config and config['channel_groups']:
            self.channel_groups = nn.ModuleDict()
            for group_name, group_cfg in config['channel_groups'].items():
                # Create a processing block for each channel group
                self.channel_groups[group_name] = self._build_channel_group(
                    group_cfg, config
                )

            # Calculate total output channels from all groups
            total_out_channels = sum(
                cfg['out_channels'] for cfg in config['channel_groups'].values()
            )
            in_channels = total_out_channels
        else:
            self.channel_groups = None
            in_channels = self.num_input_channels

        # Shared convolutional layers
        shared_layers = []

        for i, layer_cfg in enumerate(config['conv_layers']):
            # Add conv layer
            shared_layers.extend([
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=layer_cfg['out_channels'],
                    kernel_size=layer_cfg['kernel_size'],
                    stride=layer_cfg.get('stride', 1),
                    padding=layer_cfg.get('padding', 0),
                    dilation=layer_cfg.get('dilation', 1),
                    groups=layer_cfg.get('groups', 1),
                    bias=layer_cfg.get('bias', True)
                ),
                nn.BatchNorm2d(layer_cfg['out_channels'])
            ])

            # Add activation
            if config.get('activation', 'leaky_relu') == 'leaky_relu':
                shared_layers.append(nn.LeakyReLU(0.1, inplace=True))
            else:
                shared_layers.append(nn.ReLU(inplace=True))

            # Add pooling if specified
            if 'pooling' in layer_cfg:
                if layer_cfg['pooling'] == 'max':
                    shared_layers.append(
                        nn.MaxPool2d(
                            kernel_size=layer_cfg.get('pool_kernel', 2),
                            stride=layer_cfg.get('pool_stride', 2)
                        )
                    )
                elif layer_cfg['pooling'] == 'avg':
                    shared_layers.append(
                        nn.AvgPool2d(
                            kernel_size=layer_cfg.get('pool_kernel', 2),
                            stride=layer_cfg.get('pool_stride', 2)
                        )
                    )

            # Add dropout if specified
            if 'dropout' in layer_cfg and layer_cfg['dropout'] > 0:
                shared_layers.append(nn.Dropout2d(layer_cfg['dropout']))

            # Add channel attention if enabled
            if config.get('use_channel_attention', False):
                shared_layers.append(
                    ChannelAttention(num_channels=layer_cfg['out_channels'])
                )

            # Update input channels for next layer
            in_channels = layer_cfg['out_channels']

        # Add global average pooling and flatten
        shared_layers.extend([
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        ])

        # Combine into sequential
        self.shared_cnn = nn.Sequential(*shared_layers)

        # Add temporal attention if enabled
        self.temporal_attention = None
        if config.get('use_temporal_attention', False):
            # Calculate the feature dimension after CNN
            with th.no_grad():
                dummy = th.zeros(1, self.num_input_channels, 32, 32)  # Assuming min 32x32 input
                if hasattr(self, 'channel_groups') and self.channel_groups is not None:
                    # Process through channel groups first
                    group_outputs = []
                    for group_name, group in self.channel_groups.items():
                        group_outputs.append(group(dummy))
                    dummy = th.cat(group_outputs, dim=1)
                dummy = self.shared_cnn(dummy)
                temporal_dim = dummy.size(1)

            self.temporal_attention = TemporalAttention(
                num_features=temporal_dim,
                num_heads=config.get('temporal_attention_heads', 4),
                dropout=config.get('temporal_attention_dropout', 0.1)
            )

        return self

    def _initialize_weights(self) -> None:
        """Initialize weights for Conv2d and Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, observations) -> Tensor:
        """Forward pass through the feature extractor.

        Processes input through:
        1. Channel group processing (if enabled)
        2. Shared CNN layers
        3. Temporal attention (if enabled)
        4. Vector feature processing
        5. Feature combination

        Args:
            observations: Dictionary containing:
                - 'image_features': Input tensor of shape (batch_size, C, H, W)
                - 'vector_features': Input tensor of shape (batch_size, vector_dim)

        Returns:
            Extracted features of shape (batch_size, features_dim)
        """
        # Extract inputs - Handle both Dict and Box observations
        if isinstance(observations, dict):
            image_features = observations["image_features"]
            vector_features = observations.get("vector_features", th.zeros(observations["image_features"].shape[0], self.vector_features_dim, device=observations["image_features"].device))
        else:
            # Box Tensor: treat as image, create empty vector
            image_features = observations
            vector_features = th.zeros(image_features.shape[0], getattr(self, 'vector_features_dim', 64), device=image_features.device)

        # Check for NaN/Inf values in input features and clip extreme values
        image_features = th.nan_to_num(image_features, nan=0.0, posinf=1e6, neginf=-1e6)
        if not th.all(th.isfinite(image_features)):
            logger.warning("Non-finite values detected in image_features")
            image_features = th.clamp(image_features, -1e5, 1e5)

        vector_features = th.nan_to_num(vector_features, nan=0.0, posinf=1e6, neginf=-1e6)
        if not th.all(th.isfinite(vector_features)):
            logger.warning("Non-finite values detected in vector_features")
            vector_features = th.clamp(vector_features, -1e5, 1e5)

        # Process image features through channel groups if enabled
        if hasattr(self, 'channel_groups') and self.channel_groups is not None:
            group_outputs = []
            for group_name, group in self.channel_groups.items():
                # Extract channels for this group
                group_indices = self.cnn_config['channel_groups'][group_name]['indices']
                group_input = image_features[:, group_indices, :, :]
                group_outputs.append(group(group_input))

            # Concatenate all group outputs along channel dimension
            cnn_input = th.cat(group_outputs, dim=1)
        else:
            cnn_input = image_features

        # Process through shared CNN layers
        cnn_features = self.shared_cnn(cnn_input)

        # Apply temporal attention if enabled
        if self.temporal_attention is not None:
            # Reshape for temporal attention (add sequence dimension)
            seq_len = cnn_features.size(1) // self.features_dim
            cnn_features = cnn_features.view(cnn_features.size(0), seq_len, -1)

            # Apply temporal attention
            cnn_features = self.temporal_attention(cnn_features)

            # Flatten back
            cnn_features = cnn_features.reshape(cnn_features.size(0), -1)

        # Process vector features through MLP
        vector_features = self.vector_fc(vector_features)

        # Combine features
        combined = th.cat([cnn_features, vector_features], dim=1)

        # Process through final fully connected layers
        features = self.combined_fc(combined)

        # Debug logging
        if self.training:
            logger.debug(f"CNN input shape: {image_features.shape}")
            if hasattr(self, 'channel_groups') and self.channel_groups is not None:
                logger.debug(f"Processed channel groups, combined shape: {cnn_input.shape}")
            logger.debug(f"CNN features shape: {cnn_features.shape}")
            if self.temporal_attention is not None:
                logger.debug(f"After temporal attention: {cnn_features.shape}")
            logger.debug(f"Vector features shape: {vector_features.shape}")
            logger.debug(f"Combined features shape: {combined.shape}")
            logger.debug(f"Output features shape: {features.shape}")

        return features

class MultiTimeframeCNN(nn.Module):
    """Applique un CNN 1D distinct à chaque timeframe pour extraire les features."""
    def __init__(self, n_timeframes: int, n_features: int, window_sizes: List[int]):
        super().__init__()
        self.n_timeframes = n_timeframes
        self.n_features = n_features
        self.window_sizes = window_sizes

        self.tf_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten()
            ) for _ in range(n_timeframes)
        ])

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        """Traite chaque timeframe avec son CNN dédié."""
        tf_features = []
        # Itérer sur les timeframes dans un ordre défini pour la cohérence
        sorted_timeframes = sorted(observations.keys())
        for i, tf_name in enumerate(sorted_timeframes):
            # Input shape pour Conv1d: (batch, channels, length)
            # Notre observation est (batch, length, channels), donc on permute
            tf_data = observations[tf_name]
            
            # Ajoute une dimension de batch si elle est manquante
            if tf_data.dim() == 2:
                tf_data = tf_data.unsqueeze(0)

            tf_data = tf_data.permute(0, 2, 1)
            features = self.tf_convs[i](tf_data)
            tf_features.append(features)
        
        # Concaténer les features de tous les timeframes
        return th.cat(tf_features, dim=1)

class TemporalFusionExtractor(BaseFeaturesExtractor):
    """
    Extracteur de features qui combine un CNN multi-timeframe avec une couche de fusion.
    Prend en entrée un dictionnaire d'observations (une par timeframe).
    """
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        # Séparer les clés de timeframes et les autres (ex: portfolio_state)
        self.timeframe_keys = [k for k, v in observation_space.spaces.items() if len(v.shape) == 2]
        self.portfolio_state_key = 'portfolio_state'
        
        if self.portfolio_state_key not in observation_space.spaces:
            raise ValueError(f"'{self.portfolio_state_key}' not found in observation space.")

        n_timeframes = len(self.timeframe_keys)
        
        # S'assurer qu'on a au moins un timeframe
        if n_timeframes == 0:
            raise ValueError("No 2D timeframe data found in observation space.")

        # Dimensions des données de marché (on suppose qu'elles sont identiques)
        sample_tf_space = observation_space.spaces[self.timeframe_keys[0]]
        n_features = sample_tf_space.shape[1]
        window_sizes = [observation_space.spaces[tf].shape[0] for tf in self.timeframe_keys]

        # Module CNN Multi-Timeframe
        self.multitimeframe_cnn = MultiTimeframeCNN(n_timeframes, n_features, window_sizes)

        # Calculer la dimension de sortie du CNN
        # Chaque CNN sort (64 features * 16 positions temporelles) = 1024
        cnn_output_dim = n_timeframes * 64

        # Dimension de l'état du portefeuille
        portfolio_state_dim = observation_space.spaces[self.portfolio_state_key].shape[0]

        # Couche de fusion intermédiaire pour apprendre les interactions entre les timeframes
        self.inter_timeframe_fusion_layer = nn.Sequential(
            nn.Linear(cnn_output_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
        )

        # Couche de fusion finale pour combiner les features des timeframes et l'état du portefeuille
        self.fusion_layer = nn.Sequential(
            nn.Linear(128 + portfolio_state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        """
        Le forward pass de l'extracteur.
        `observations` est un dictionnaire de tenseurs, une clé par timeframe.
        """
        # Séparer les observations du marché de l'état du portefeuille
        market_observations = {k: observations[k] for k in self.timeframe_keys}
        portfolio_state = observations[self.portfolio_state_key]

        # 1. Extraction des features par les CNNs pour chaque timeframe
        cnn_features = self.multitimeframe_cnn(market_observations)

        # 2. Fusion intermédiaire des features des différents timeframes
        timeframe_fused_features = self.inter_timeframe_fusion_layer(cnn_features)

        # 3. Concaténer les features fusionnées avec l'état du portefeuille
        # Assurer que portfolio_state a une dimension de batch si elle est manquante
        if portfolio_state.dim() == 1:
            portfolio_state = portfolio_state.unsqueeze(0)
            
        combined_features = th.cat([timeframe_fused_features, portfolio_state], dim=1)

        # 3. Fusion des features
        fused_features = self.fusion_layer(combined_features)

        return fused_features

