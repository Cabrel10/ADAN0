"""
Implémentation d'un CNN personnalisé avec attention pour l'extraction de caractéristiques.
Enhanced with memory optimizations and mixed-precision training support.
"""
import gc
import logging
from functools import wraps
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class SqueezeExcitation(nn.Module):
    """
    Module Squeeze-and-Excitation pour l'attention par canal.
    """
    def __init__(self, channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MultiScaleResidualBlock(nn.Module):
    """
    Bloc résiduel multi-échelle avec Squeeze-and-Excitation.
    """
    def __init__(self, in_channels: int, out_channels: int, config: Dict):
        super().__init__()
        self.scales = nn.ModuleList()

        # Configuration des branches multi-échelles
        for scale_cfg in config.get('multi_scale', []):
            self.scales.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels // len(config['multi_scale']),
                        kernel_size=scale_cfg['kernel_size'],
                        padding=scale_cfg['padding'],
                        dilation=scale_cfg.get('dilation', 1),
                        bias=False
                    ),
                    nn.BatchNorm2d(out_channels // len(config['multi_scale'])),
                    nn.LeakyReLU(negative_slope=config.get('leaky_relu_negative_slope', 0.01), inplace=True),
                    nn.Dropout2d(p=config.get('dropout', 0.1))
                )
            )

        # Connexion résiduelle
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        # Attention par canal
        self.se = SqueezeExcitation(out_channels, reduction_ratio=config.get('se_ratio', 16))

    def forward(self, x: Tensor) -> Tensor:
        # Concaténer les sorties des différentes échelles
        out = torch.cat([scale(x) for scale in self.scales], dim=1)
        # Ajouter la connexion résiduelle
        out = out + self.shortcut(x)
        # Appliquer l'attention par canal
        out = self.se(out)
        return out

class TemporalAttention(nn.Module):
    """
    Module d'attention temporelle multi-tête.
    """
    def __init__(self, in_channels: int, num_heads: int = 4, dropout: float = 0.1, use_residual: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_residual = use_residual

        self.qkv = nn.Linear(in_channels, in_channels * 3)
        self.proj = nn.Linear(in_channels, in_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape

        # Projection Q, K, V
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Calcul des scores d'attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Application des poids d'attention aux valeurs
        out = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, -1)
        out = self.proj(out)

        # Connexion résiduelle
        if self.use_residual:
            out = out + x

        return out

def memory_efficient_forward(func):
    """Decorator for memory-efficient forward passes."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Clear cache before forward pass if GPU memory is tight
        if torch.cuda.is_available() and hasattr(self, '_memory_efficient') and self._memory_efficient:
            torch.cuda.empty_cache()

        try:
            return func(self, *args, **kwargs)
        finally:
            # Optional cleanup after forward pass
            if torch.cuda.is_available() and hasattr(self, '_aggressive_cleanup') and self._aggressive_cleanup:
                torch.cuda.empty_cache()
                gc.collect()

    return wrapper


class CustomCNN(BaseFeaturesExtractor):
    """
    CNN personnalisé avec attention pour l'extraction de caractéristiques.
    Enhanced with memory optimizations and mixed-precision training support.
    """
    def __init__(
        self,
        observation_space,
        features_dim: int = 256,
        cnn_configs: Optional[Dict] = None,
        diagnostics: Optional[Dict] = None,
        memory_config: Optional[Dict] = None
    ):
        super().__init__(observation_space, features_dim=features_dim)

        # Configuration par défaut si non spécifiée
        if cnn_configs is None:
            cnn_configs = {
                'block_a': {
                    'out_channels': 64,
                    'kernel_size': 3,
                    'padding': 1,
                    'leaky_relu_negative_slope': 0.01,
                    'dropout': 0.1
                },
                'block_b': {
                    'multi_scale': [
                        {'kernel_size': 3, 'dilation': 1, 'padding': 1},
                        {'kernel_size': 5, 'dilation': 1, 'padding': 2},
                        {'kernel_size': 3, 'dilation': 2, 'padding': 2}
                    ],
                    'se_ratio': 16,
                    'dropout': 0.2,
                    'leaky_relu_negative_slope': 0.01
                },
                'attention': {
                    'num_heads': 4,
                    'dropout': 0.1,
                    'use_residual': True
                },
                'head': {
                    'hidden_units': [256, 128],
                    'dropout': 0.3,
                    'activation': 'leaky_relu'
                }
            }

        # Memory optimization configuration
        self.memory_config = memory_config or {}
        self._memory_efficient = self.memory_config.get('enable_memory_efficient', True)
        self._aggressive_cleanup = self.memory_config.get('aggressive_cleanup', False)
        self._mixed_precision = self.memory_config.get('enable_mixed_precision', True)
        self._gradient_checkpointing = self.memory_config.get('enable_gradient_checkpointing', False)

        # Logger for memory tracking
        self.logger = logging.getLogger(self.__class__.__name__)

        # Sauvegarder la configuration pour la visualisation
        self.diagnostics = diagnostics or {}
        self.attention_maps = []

        # Dimensions d'entrée (C, H, W) pour l'image
        in_channels = observation_space.shape[0]

        # Bloc A: Convolutions séparées par canal
        self.block_a = nn.Sequential(
            nn.Conv2d(in_channels, cnn_configs['block_a']['out_channels'],
                     kernel_size=cnn_configs['block_a']['kernel_size'],
                     padding=cnn_configs['block_a']['padding']),
            nn.BatchNorm2d(cnn_configs['block_a']['out_channels']),
            nn.LeakyReLU(negative_slope=cnn_configs['block_a']['leaky_relu_negative_slope']),
            nn.Dropout2d(p=cnn_configs['block_a']['dropout'])
        )

        # Bloc B: Multi-échelle avec connexions résiduelles et SE
        self.block_b = MultiScaleResidualBlock(
            in_channels=cnn_configs['block_a']['out_channels'],
            out_channels=cnn_configs['block_a']['out_channels'] * 2,
            config=cnn_configs['block_b']
        )

        # Attention temporelle
        self.temporal_attention = TemporalAttention(
            in_channels=cnn_configs['block_a']['out_channels'] * 2,
            num_heads=cnn_configs['attention']['num_heads'],
            dropout=cnn_configs['attention']['dropout'],
            use_residual=cnn_configs['attention']['use_residual']
        )

        # Tête de classification
        head_units = [cnn_configs['block_a']['out_channels'] * 2] + cnn_configs['head']['hidden_units']
        head_layers = []

        for i in range(len(head_units) - 1):
            head_layers.extend([
                nn.Linear(head_units[i], head_units[i+1]),
                nn.LeakyReLU(negative_slope=cnn_configs['block_a']['leaky_relu_negative_slope']),
                nn.Dropout(p=cnn_configs['head']['dropout'])
            ])

        self.head = nn.Sequential(*head_layers)

        # Initialisation des poids
        self._init_weights()

        # Memory optimization setup
        if self._gradient_checkpointing:
            self._setup_gradient_checkpointing()

        # Model compilation for inference
        self._compiled = False
        # 'reduce-overhead' pour moins de mémoire
        self._compile_mode = 'max-autotune'

        # Log memory configuration
        self.logger.info(
            "CustomCNN initialized with memory optimizations: "
            f"efficient={self._memory_efficient}, "
            f"mixed_precision={self._mixed_precision}, "
            f"gradient_checkpointing={self._gradient_checkpointing}, "
            f"compilation={self._compile_mode}"
        )

        # Compile the model if CUDA is available
        self._maybe_compile()

    def _init_weights(self):
        """Initialisation des poids du modèle."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _maybe_compile(self):
        """Compile the model if conditions are met."""
        if (torch.__version__ >= '2.0.0' and
                torch.cuda.is_available() and
                not self._compiled):
            try:
                # Compile the model for inference
                self._forward_impl = torch.compile(
                    self._forward_impl,
                    mode=self._compile_mode,
                    fullgraph=False,
                    dynamic=True
                )
                self._compiled = True
                self.logger.info(
                    f"Model compiled with mode: {self._compile_mode}"
                )
            except Exception as e:
                self.logger.warning(f"Model compilation failed: {e}")
                self._compiled = False

    def _setup_gradient_checkpointing(self):
        """Setup gradient checkpointing for memory efficiency."""
        self.logger.info("Setting up gradient checkpointing for memory efficiency")

        # Enable gradient checkpointing for major blocks
        if hasattr(self.block_b, 'scales'):
            for scale in self.block_b.scales:
                for module in scale:
                    if hasattr(module, 'gradient_checkpointing'):
                        module.gradient_checkpointing = True

    def enable_mixed_precision(self):
        """Enable mixed precision training optimizations."""
        if torch.cuda.is_available():
            self._mixed_precision = True
            self.logger.info("Mixed precision training enabled")
        else:
            self.logger.warning("Mixed precision requires CUDA, keeping disabled")

    def disable_mixed_precision(self):
        """Disable mixed precision training."""
        self._mixed_precision = False
        self.logger.info("Mixed precision training disabled")

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage of the model.

        Returns:
            Dict with memory usage statistics
        """
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}

        # Calculate model parameters memory
        param_memory = sum(p.numel() * p.element_size() for p in self.parameters()) / (1024**2)  # MB

        # Get current GPU memory
        allocated = torch.cuda.memory_allocated() / (1024**2)  # MB
        cached = torch.cuda.memory_reserved() / (1024**2)  # MB

        return {
            "model_parameters_mb": param_memory,
            "gpu_allocated_mb": allocated,
            "gpu_cached_mb": cached,
            "memory_efficient": self._memory_efficient,
            "mixed_precision": self._mixed_precision,
            "gradient_checkpointing": self._gradient_checkpointing
        }

    def optimize_for_inference(self):
        """Optimize model for inference (disable training-specific features)."""
        self.eval()  # Set to evaluation mode
        if self._gradient_checkpointing:
            self.logger.info("Disabling gradient checkpointing for inference")
            self._gradient_checkpointing = False

        # Disable dropout and other training-specific layers
        for module in self.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d)):
                module.p = 0.0

        # Compile the model for inference
        self._maybe_compile()

        # Clear CUDA cache and run garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Enable cuDNN benchmarking for inference
        torch.backends.cudnn.benchmark = True

        self.logger.info("Model optimized for inference")

    def cleanup_memory(self):
        """Manual memory cleanup."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        self.logger.debug("Memory cleanup performed")

    @memory_efficient_forward
    def forward(self, observations: Tensor) -> Tensor:
        """
        Passe avant du réseau avec optimisations mémoire.

        Args:
            observations: Tenseur d'entrée de forme (batch_size, C, H, W)

        Returns:
            Tensor: Caractéristiques extraites de forme (batch_size, features_dim)
        """
        # Use mixed precision if enabled
        if self._mixed_precision and torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                return self._forward_impl(observations)
        else:
            return self._forward_impl(observations)

    def _forward_impl(self, observations: Tensor) -> Tensor:
        """Internal forward implementation."""
        # Bloc A
        x = self.block_a(observations)

        # Bloc B with optional gradient checkpointing
        if self._gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            x = checkpoint(self.block_b, x)
        else:
            x = self.block_b(x)

        # Réduction de dimension spatiale
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)

        # Tête de classification
        features = self.head(x)

        return features

    def get_attention_map(self, observations: Tensor) -> Tensor:
        """
        Récupère la carte d'attention pour la visualisation.

        Args:
            observations: Tenseur d'entrée de forme (batch_size, C, H, W)

        Returns:
            Tensor: Carte d'attention de forme (batch_size, H, W)
        """
        with torch.no_grad():
            # Passe avant jusqu'au bloc B
            x = self.block_a(observations)
            x = self.block_b(x)

            # Calcul de l'attention par canal (moyenne des canaux)
            attention_map = x.mean(dim=1, keepdim=True)

            # Normalisation pour la visualisation
            attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-5)

            return attention_map.squeeze(1)
