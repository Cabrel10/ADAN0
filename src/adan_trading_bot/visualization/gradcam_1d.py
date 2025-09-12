"""
Grad-CAM 1D pour visualiser l'attention du modèle sur les séries temporelles.
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class GradCAM1D:
    """
    Grad-CAM 1D pour visualiser les régions importantes des séries temporelles.
    """
    def __init__(self, model, target_layer):
        """
        Initialise Grad-CAM 1D.

        Args:
            model: Modèle PyTorch
            target_layer: Couche cible pour l'extraction des activations
        """
        self.model = model
        self.target_layer = target_layer
        self.grad = None
        self.activations = None

        # Enregistre les hooks
        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.grad = grad_out[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def __call__(self, input_tensor, class_idx=None):
        """
        Calcule la carte d'activation Grad-CAM.

        Args:
            input_tensor: Tenseur d'entrée [B, C, T, F] ou [B, C, 1, T]
            class_idx: Index de classe pour le calcul du gradient (None pour régression)

        Returns:
            Carte d'activation normalisée [T,]
        """
        self.model.zero_grad()

        # Forward pass
        out = self.model(input_tensor)  # [B, N] ou [B, 1] pour la régression

        # Sélectionne la sortie pour laquelle calculer le gradient
        if class_idx is None:
            # Pour la régression ou sortie unique
            score = out.squeeze()
        else:
            # Pour la classification
            score = out[0, class_idx]

        # Backward pass
        score.backward(retain_graph=True)

        # Récupère les gradients et activations
        grads = self.grad
        activations = self.activations

        # Calcule la pondération des canaux
        if grads.ndim == 4:
            # [B, C, f, T] -> moyenne sur les dimensions spatiales
            weights = grads.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
            cam = (weights * activations).sum(dim=1)  # [B, f, T]
            cam = cam.mean(dim=1)  # [B, T]
        elif grads.ndim == 3:
            # [B, C, T]
            weights = grads.mean(dim=2, keepdim=True)  # [B, C, 1]
            cam = (weights * activations).sum(dim=1)  # [B, T]

        # Applique ReLU et normalise
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)

        return cam

def plot_gradcam(cam, timestamps, prices, save_path=None):
    """
    Affiche la heatmap Grad-CAM superposée au prix.

    Args:
        cam: Carte d'activation Grad-CAM [T,]
        timestamps: Timestamps pour l'axe des x
        prices: Prix pour le tracé de la courbe
        save_path: Chemin pour sauvegarder la figure (optionnel)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})

    # Courbe des prix
    ax1.plot(timestamps, prices, 'b-', linewidth=1.5, label='Prix')
    ax1.set_ylabel('Prix', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # Heatmap Grad-CAM
    cmap = plt.cm.Reds
    norm = mcolors.Normalize(vmin=0, vmax=1)
    ax2.pcolormesh([timestamps], [0, 1], [cam], cmap=cmap, norm=norm, shading='auto')
    ax2.set_yticks([])
    ax2.set_xlabel('Temps')
    ax2.set_title('Grad-CAM')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close(fig)

    return fig
