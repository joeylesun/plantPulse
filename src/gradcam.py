"""
Grad-CAM: visualize which image regions the CNN attends to when classifying.

Rubric item addressed: Grad-CAM heatmaps for explainability.

How it works (short version):
  1. Forward pass - compute class score.
  2. Backward pass of that score w.r.t. the final conv layer's feature maps.
  3. Global-average-pool the gradients -> per-channel importance weights.
  4. Weighted sum of feature maps -> heatmap showing 'where the model looked.'

We hand-roll this instead of pulling in pytorch-grad-cam because the project
needs to show the ML concept in the code, and the dependency is small enough
that it's cleaner as an explicit implementation.

AI attribution: Core Grad-CAM algorithm is standard (Selvaraju et al. 2017);
this PyTorch implementation was drafted with Claude (Anthropic) and adapted
by the author.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """Grad-CAM for any CNN with an identifiable final conv block.

    For our ResNet-50, target_layer should be model.layer4 (the last conv stage).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # Register hooks on the target layer so we can grab its output (forward)
        # and the gradient of the class score w.r.t. that output (backward).
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        # grad_output is a tuple; we want the first (and only) element
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        """Produce a heatmap in [0,1] at the input spatial resolution.

        input_tensor: shape (1, 3, H, W), normalized as the model expects.
        target_class: which class to explain. If None, uses model's top prediction.
        """
        self.model.eval()
        # We need gradients flowing even though model.eval() disables dropout etc.
        logits = self.model(input_tensor)

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        # Zero out existing grads, then backprop only from the target class score
        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward()

        # gradients: (1, C, h, w); activations: (1, C, h, w)
        # weights[c] = mean over spatial dims of gradients[c]
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)  # keep only positive contributions

        # Upsample to input size
        cam = F.interpolate(
            cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1] for visualization
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def overlay_heatmap_on_image(
    pil_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> Image.Image:
    """Blend a heatmap over an image for display.

    Uses matplotlib's jet colormap (standard for Grad-CAM visualizations).
    """
    import matplotlib.cm as cm

    img = np.array(pil_image.convert("RGB").resize(heatmap.shape[::-1])) / 255.0
    colored = cm.jet(heatmap)[:, :, :3]  # drop alpha channel from colormap
    blended = (1 - alpha) * img + alpha * colored
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)
